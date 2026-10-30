//! Per-page font cache.
//!
//! [`FontCache`] maps a PDF font resource name (e.g. `b"F1"`) to a loaded
//! [`FontFace`].  Faces are loaded on first use during rendering, then reused
//! for all subsequent glyphs in the same page.
//!
//! When a font has no embedded bytes, or its embedded program cannot be
//! parsed, the cache draws it with a substitute of the same family class
//! and variant (sans-serif, serif or monospaced; regular, bold, italic or
//! bold italic), located by [`super::substitutes`].
//!
//! # Threading
//!
//! One instance is created per page renderer (one per thread when pages are
//! rendered in parallel) and owns that page's [`FontEngine`].  What depends
//! only on an indirect font dictionary — its descriptor, its parsed program,
//! its code → glyph table and the identifier of its face at each matrix — is
//! kept in a [`DocumentFonts`] shared by every page of the document, with the
//! glyph bitmaps of those faces.

use std::collections::HashMap;
use std::ops::Deref;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use font::{
    cache::GlyphCache,
    engine::{FaceParams, FontEngine},
    face::FontFace,
    key::{FaceId, FaceIds},
    program::FontProgram,
};
use pdf::ObjectId;

use super::encodings::{base_encoding_table, glyph_name_for_code};
use super::substitutes::substitute_program;
use crate::resources::cmap_predefined::cid_to_unicode;
use crate::resources::font::{BaseEncoding, FontDescriptor, PdfFontKind, fallback_font};
use crate::resources::{FontKey, PageResources};

// ── Cache ─────────────────────────────────────────────────────────────────────

/// Key type for [`FontCache::faces`]: the font dictionary's identity paired
/// with the 2×2 Trm matrix bit pattern.
type FaceKey = (FontKey, [u64; 4]);

/// The font selected by `Tf`: the identity of its dictionary and the
/// descriptor resolved from it.
#[derive(Clone, Debug)]
pub struct ActiveFont {
    /// Identity of the font dictionary.
    pub key: FontKey,
    /// Descriptor resolved from the dictionary, shared by every selection of
    /// the same dictionary on the page.
    pub descriptor: Arc<FontDescriptor>,
}

/// A face loaded for a page: the [`FontFace`] it draws with, and whether
/// that face is a substitute program rather than one the document embeds.
pub struct PageFace {
    face: FontFace,
    /// `true` when the face draws a substitute: the font has no embedded
    /// program, or its embedded program could not be parsed.
    pub substitute: bool,
}

impl Deref for PageFace {
    type Target = FontFace;

    fn deref(&self) -> &FontFace {
        &self.face
    }
}

/// The program a descriptor draws with and whether it is a substitute.
type ResolvedProgram = (Arc<FontProgram>, bool);

/// The fonts of one document, resolved once and shared by every page
/// rendered from it.  Entries are keyed by the object number of an indirect
/// font dictionary, so a store serves one document only.
#[derive(Default)]
pub struct DocumentFonts {
    fonts: Mutex<HashMap<ObjectId, Arc<SharedFont>>>,
    /// Source of the identifier of every face made for the document.
    face_ids: Arc<FaceIds>,
    /// Glyph bitmaps of every face made for the document.
    glyphs: GlyphCache,
}

impl DocumentFonts {
    /// The entry of the font dictionary `id`, created empty on first use.
    /// Its fields are filled outside the store's lock.
    fn font(&self, id: ObjectId) -> Arc<SharedFont> {
        let mut fonts = self.fonts.lock().unwrap_or_else(PoisonError::into_inner);
        Arc::clone(fonts.entry(id).or_default())
    }
}

/// What one indirect font dictionary resolves to, independent of the page
/// and of the size it is drawn at.  `None` inside a cell records a failure.
#[derive(Default)]
struct SharedFont {
    descriptor: OnceLock<Arc<FontDescriptor>>,
    program: OnceLock<Option<ResolvedProgram>>,
    /// The `code_to_gid` table computed from the program, `None` when the
    /// descriptor's own table (or the program's charmap) applies.
    code_to_gid: OnceLock<Option<Vec<u32>>>,
    /// Face identifier per Trm bit pattern, so every page draws the font at
    /// one matrix from the same cached bitmaps.
    face_ids: Mutex<HashMap<[u64; 4], FaceId>>,
}

impl SharedFont {
    /// The identifier of the face at the matrix `trm`, drawn from `ids` on
    /// first use.
    fn face_id(&self, trm: [u64; 4], ids: &FaceIds) -> FaceId {
        let mut face_ids = self.face_ids.lock().unwrap_or_else(PoisonError::into_inner);
        *face_ids.entry(trm).or_insert_with(|| ids.next())
    }
}

/// Per-page cache of resolved font descriptors and loaded font faces.
pub struct FontCache {
    engine: FontEngine,
    /// Document-wide store for indirect font dictionaries.
    shared: Arc<DocumentFonts>,
    /// The document's glyph bitmap cache.
    glyph_cache: GlyphCache,
    /// Descriptor of each font selected on this page.
    descriptors: HashMap<FontKey, Arc<FontDescriptor>>,
    /// Map from (font dictionary, four f64-as-u64 bits of the Trm 2×2 matrix)
    /// to loaded face.  The full matrix is the key because the same font can
    /// appear at different sizes and skew angles within a single page.
    faces: HashMap<FaceKey, Option<PageFace>>,
}

impl FontCache {
    /// Create a new cache resolving through the document store `shared`.
    #[must_use]
    pub fn new(shared: Arc<DocumentFonts>) -> Self {
        Self {
            engine: FontEngine::with_face_ids(true, Arc::clone(&shared.face_ids)),
            glyph_cache: shared.glyphs.clone(),
            shared,
            descriptors: HashMap::new(),
            faces: HashMap::new(),
        }
    }

    /// The font the resource `name` of `resources` selects, resolving its
    /// descriptor on the first selection of its dictionary.
    ///
    /// A name absent from the `/Font` resources, or whose entry is not a
    /// font dictionary, selects [`fallback_font`] under a key private to
    /// the resource context, so the text it shows is drawn rather than
    /// dropped.
    pub fn select(&mut self, resources: &PageResources<'_>, name: &[u8]) -> ActiveFont {
        let (key, entry) = resources.font_entry(name).unwrap_or_else(|| {
            let key = FontKey::Direct {
                context: resources.resource_context_id(),
                name: name.into(),
            };
            (key, pdf::Object::Null)
        });
        let shared = &self.shared;
        let descriptor = self
            .descriptors
            .entry(key.clone())
            .or_insert_with(|| {
                let resolve = || {
                    resources
                        .resolve_font_entry(&entry)
                        .unwrap_or_else(|| fallback_font(resources.doc()))
                };
                match &key {
                    FontKey::Object(id) => Arc::clone(
                        shared
                            .font(*id)
                            .descriptor
                            .get_or_init(|| Arc::new(resolve())),
                    ),
                    FontKey::Direct { .. } => Arc::new(resolve()),
                }
            })
            .clone();
        ActiveFont { key, descriptor }
    }

    /// The document store this cache resolves indirect fonts through.
    #[must_use]
    pub const fn document_fonts(&self) -> &Arc<DocumentFonts> {
        &self.shared
    }

    /// Return the glyph bitmap cache shared by every face of the document.
    #[must_use]
    pub const fn glyph_cache(&self) -> &GlyphCache {
        &self.glyph_cache
    }

    /// Return a reference to the [`PageFace`] for `font`, loading it on first
    /// use.  `name` is the resource name `font` was selected by, for
    /// diagnostics.
    ///
    /// `trm` is the 2×2 submatrix of the **text rendering matrix** in device
    /// pixels: `font_size × Tm[2×2] × CTM[2×2]`.  It determines both the
    /// rendered pixel size and any shear/rotation.
    ///
    /// Returns `None` if the face cannot be loaded.  The caller should skip
    /// glyph rendering — missing glyphs are silently omitted.
    pub fn get_or_load(
        &mut self,
        font: &ActiveFont,
        name: &[u8],
        trm: [f64; 4],
    ) -> Option<&PageFace> {
        // The full 2×2 Trm matrix, as bit patterns, is part of the key: the
        // same font at different sizes or skews needs distinct faces.  Failed
        // loads are stored as `None`, so a font that cannot load is tried once.
        let key: FaceKey = (font.key.clone(), trm.map(f64::to_bits));
        if !self.faces.contains_key(&key) {
            let face = self.load_face(&font.key, &font.descriptor, trm);
            if face.is_none() {
                log::warn!(
                    "font_cache: could not load face for /{} trm={trm:?}",
                    String::from_utf8_lossy(name)
                );
            }
            return self.faces.entry(key).or_insert(face).as_ref();
        }
        self.faces.get(&key)?.as_ref()
    }

    fn load_face(
        &self,
        key: &FontKey,
        desc: &FontDescriptor,
        trm: [f64; 4],
    ) -> Option<PageFace> {
        // Pre-flight: reject degenerate matrices before parsing the program.
        // The y-column magnitude (hypot of indices 2 & 3) is the pixel size used by
        // FontFace::new; a value < 1.0 or non-finite produces an unusable face.
        if !trm_pixel_size_valid(trm) {
            log::debug!(
                "font_cache: degenerate trm (size={:.1}), skipping face load",
                f64::hypot(trm[2], trm[3])
            );
            return None;
        }

        let params = FaceParams {
            code_to_gid: desc.code_to_gid.clone(),
            mat: trm,
        };
        let shared = match key {
            FontKey::Object(id) => Some(self.shared.font(*id)),
            FontKey::Direct { .. } => None,
        };

        let (program, substitute) = match &shared {
            Some(shared) => shared.program.get_or_init(|| parse_program(desc)).clone()?,
            None => parse_program(desc)?,
        };
        let mut face = match &shared {
            Some(shared) => {
                let id = shared.face_id(trm.map(f64::to_bits), &self.shared.face_ids);
                self.engine.load_program_face_as(id, program, params)
            }
            None => self.engine.load_program_face(program, params),
        }
        .inspect_err(|e| log::debug!("font_cache: {e}"))
        .ok()?;

        // A substitute resolves codes as the program of a non-embedded font.
        let embedded = desc.bytes.is_some() && !substitute;
        let table = shared.as_ref().map_or_else(
            || resolved_code_to_gid(&face, desc, embedded),
            |shared| {
                shared
                    .code_to_gid
                    .get_or_init(|| resolved_code_to_gid(&face, desc, embedded))
                    .clone()
            },
        );
        if let Some(table) = table {
            face.code_to_gid = table;
        }
        Some(PageFace { face, substitute })
    }
}

/// Parse the program a descriptor draws with: its embedded font program,
/// else the substitute of its style — for a font without a program, and
/// for one whose program does not parse (truncated stream, unsupported
/// format), which is logged.  `None` when no substitute is available
/// either.
fn parse_program(desc: &FontDescriptor) -> Option<ResolvedProgram> {
    if let Some(bytes) = &desc.bytes {
        match FontProgram::parse(bytes.clone(), 0) {
            Ok(program) => return Some((Arc::new(program), false)),
            Err(e) => log::warn!(
                "font_cache: embedded font program unusable ({e}) — drawing a substitute"
            ),
        }
    }
    substitute_program(desc.substitute).map(|program| (program, true))
}

/// The `code_to_gid` table a face of `desc` needs in place of the
/// descriptor's, or `None` when the descriptor's table (or the program's own
/// charmap) applies.
///
/// A simple embedded Type1/Type1C/CFF font is resolved by glyph *name*: its
/// char codes are not glyph indices (subset programs pack glyphs
/// arbitrarily) and it often has no usable Unicode cmap, so each code's name
/// (`Differences` override, else the base encoding) is looked up in the
/// program's charset.  An embedded font drawn through a Unicode cmap, and
/// every substitute (whose charmap is Unicode, from a `cmap` subtable or
/// from its glyph names), is resolved through the names' Unicode values.
/// CID/Type0 keep the descriptor's table; symbolic TrueType keeps the
/// program's own charmap.
///
/// An embedded simple sfnt with no `cmap` subtable and no `post` names
/// (a legacy subset) offers no mapping at all; its codes are its glyph
/// indices, the only reading under which such a subset draws.
///
/// `embedded` is whether `face` draws the descriptor's own program.
fn resolved_code_to_gid(
    face: &FontFace,
    desc: &FontDescriptor,
    embedded: bool,
) -> Option<Vec<u32>> {
    let simple = desc.cid_encoding.is_none();
    let outline = desc.type3.is_none();
    if embedded && simple && outline && !face.program().has_charmap() {
        Some((0..256).collect())
    } else if wants_name_resolution(embedded, simple, outline, desc.kind) {
        Some(resolve_simple_to_gid(
            face,
            &desc.differences,
            desc.base_encoding,
        ))
    } else if wants_unicode_resolution(embedded, simple, outline, desc.kind)
        && (!embedded || face.has_unicode_cmap())
    {
        Some(resolve_via_unicode_cmap(
            face,
            &desc.differences,
            desc.base_encoding,
            embedded,
        ))
    } else if !embedded
        && !simple
        && let Some(table) = desc.substitute.script.and_then(cid_to_unicode)
    {
        let vertical = desc.cid_encoding.as_ref().is_some_and(|enc| enc.vertical);
        Some(resolve_cids_via_unicode(face, &table, vertical))
    } else {
        None
    }
}

/// Build a 65 536-entry CID → glyph table for a composite font drawn by a
/// substitute, through the CID → Unicode table of its character
/// collection and the substitute's Unicode charmap; in `vertical`
/// writing mode each glyph is replaced by its vertical form.  The table
/// spans the whole CID space so that no CID falls through to the charmap.
fn resolve_cids_via_unicode(
    face: &FontFace,
    cid_to_unicode: &HashMap<u32, u32>,
    vertical: bool,
) -> Vec<u32> {
    let mut table = vec![0u32; 1 << 16];
    for (&cid, &unicode) in cid_to_unicode {
        if let Some(slot) = table.get_mut(cid as usize) {
            let gid = face.raw_get_char_index(unicode);
            *slot = if vertical {
                face.program().vertical_gid(gid)
            } else {
                gid
            };
        }
    }
    table
}

// ── Shared size-validation ────────────────────────────────────────────────────

/// Return `true` if the y-column magnitude of `trm` (the face's nominal
/// pixel size) is a finite value ≥ 1.0.
///
/// This mirrors the validation the font engine applies and rejects
/// obviously degenerate matrices before the font program is parsed.
pub(crate) fn trm_pixel_size_valid(trm: [f64; 4]) -> bool {
    let size = f64::hypot(trm[2], trm[3]);
    size.is_finite() && size >= 1.0
}

// ── Simple-font name → GID resolution ────────────────────────────────────────

/// Decide whether the glyph-name → charset resolution path applies, given
/// the descriptor's shape.  This is the load-bearing dispatch gate that
/// must fire *only* for a simple embedded Type1/Type1C/CFF font:
///
/// - `has_bytes` false  → non-embedded standard-14: the fallback face's own
///   active charmap resolves codes; name resolution on a substitute face
///   would mis-map.
/// - `not_cid` false    → CID/Type0: the descriptor's `code_to_gid` (built
///   from `CIDToGIDMap` / the `CMap`) already owns resolution.
/// - `not_type3` false  → Type 3: glyphs are content-stream procedures, not
///   a font program with a charset.
/// - `kind` `TrueType`  → symbolic/non-symbolic TrueType keeps the cmap
///   the font program selects; only `Type1` (incl. `MMType1`) and
///   `Other` (OpenType-CFF wrappers) are name-resolved.
const fn wants_name_resolution(
    has_bytes: bool,
    not_cid: bool,
    not_type3: bool,
    kind: PdfFontKind,
) -> bool {
    has_bytes && not_cid && not_type3 && matches!(kind, PdfFontKind::Type1 | PdfFontKind::Other)
}

/// Decide whether a simple font's codes are resolved through glyph names
/// and the Unicode cmap of the loaded face ([`resolve_via_unicode_cmap`]).
///
/// - Embedded TrueType: PDF §9.6.5.4 resolves code → glyph name → Unicode
///   → (3,1) cmap for a non-symbolic font.
/// - Non-embedded font of any simple kind: the substitute face is a Unicode
///   TrueType font that knows nothing of the PDF encoding, so a code means
///   the glyph its encoding names, never the substitute's glyph at that
///   code point.
///
/// CID/Type0 and Type 3 fonts own their resolution and never qualify.
const fn wants_unicode_resolution(
    has_bytes: bool,
    not_cid: bool,
    not_type3: bool,
    kind: PdfFontKind,
) -> bool {
    not_cid && not_type3 && (!has_bytes || matches!(kind, PdfFontKind::TrueType))
}

/// Build a 256-entry `code_to_gid` table for a simple embedded
/// Type1/Type1C/CFF font by resolving each char code's glyph *name* against
/// the font program's own charset.
///
/// For every code 0..256 the glyph name is `differences[code]` if the
/// `Encoding/Differences` array overrides it, otherwise the name the
/// `base_encoding` assigns to that code (PDF §9.6.6).  With no base encoding
/// named ([`BaseEncoding::FontProgram`]) a code without a `Differences`
/// entry resolves through the program's built-in encoding, the only
/// mapping a subset with a custom encoding (glyphs named for their codes,
/// no standard names) carries.  A name is resolved to a GID through the
/// program's own glyph names (charset / `CharStrings` / `post` table — the
/// only correct source for a subsetted program where the char code is
/// neither the GID nor a usable Unicode-cmap key).
///
/// Resolution order per code:
/// 1. no `Differences` entry and no base encoding named → the program's
///    built-in encoding (`raw_get_builtin_index`), when it assigns the code
/// 2. name (Differences ▸ base encoding, `StandardEncoding` when none is
///    named) → `raw_get_name_index`
/// 3. name → Adobe Glyph List → Unicode → `raw_get_char_index` (handles
///    programs that expose a usable Unicode cmap but no charset names)
/// 4. `.notdef` (0) — a truly absent glyph.  We deliberately do NOT fall
///    back to code-identity: identity is the original garble bug.
fn resolve_simple_to_gid(
    face: &FontFace,
    differences: &[Option<Box<str>>; 256],
    base: BaseEncoding,
) -> Vec<u32> {
    let mut table = vec![0u32; 256];
    for (code, slot) in (0u8..=u8::MAX).zip(table.iter_mut()) {
        if base == BaseEncoding::FontProgram
            && differences[usize::from(code)].is_none()
            && let Some(gid) = face.raw_get_builtin_index(code)
        {
            *slot = gid;
            continue;
        }
        let Some(name) = glyph_name_for_code(differences, base, usize::from(code)) else {
            continue;
        };

        let mut gid = face.raw_get_name_index(name);
        if gid == 0
            && let Some(unicode) = font::glyph_name_to_unicode(name)
        {
            gid = face.raw_get_char_index(unicode);
        }
        *slot = gid;
    }
    table
}

/// Build a 256-entry `code_to_gid` table for a simple font drawn with a face
/// whose active cmap is Unicode (PDF §9.6.5.4: code → glyph name → Unicode →
/// (3,1) cmap): an embedded TrueType program, or the substitute face of a
/// non-embedded font.
///
/// A code is resolved through its glyph name when the name comes from a
/// `Differences` entry or from an explicit `WinAnsi` / `MacRoman` base
/// encoding and the font maps the name's Unicode value; this is what places
/// `WinAnsi` 0x80–0x9F (`oe` at 0x9C, `quoteright` at 0x92, …) on their glyphs.
/// A substitute program carries the name itself in its charset, which is
/// consulted first.  Every other code is looked up as a raw Unicode code
/// point, the identity that holds for the ASCII and Latin-1 ranges of those
/// encodings.
///
/// For an `embedded` program, a code that neither lookup maps resolves
/// through the program's byte-keyed (3,0) / (1,0) subtables
/// ([`FontFace::raw_get_code_index`], PDF §9.6.6.4).  A Unicode charmap
/// derived from `post` glyph names keys its glyphs by the Unicode value of
/// the name (`therefore` → U+2234), so the code a symbolic subset assigns
/// such a glyph is reachable only through those subtables.
fn resolve_via_unicode_cmap(
    face: &FontFace,
    differences: &[Option<Box<str>>; 256],
    base: BaseEncoding,
    embedded: bool,
) -> Vec<u32> {
    let nonzero = |gid: u32| (gid != 0).then_some(gid);
    (0..=u8::MAX)
        .map(|code| {
            let idx = usize::from(code);
            let name = differences[idx].as_deref().or_else(|| {
                matches!(
                    base,
                    BaseEncoding::WinAnsi
                        | BaseEncoding::MacRoman
                        | BaseEncoding::Symbol
                        | BaseEncoding::ZapfDingbats
                )
                .then(|| base_encoding_table(base)[idx])
                .flatten()
                .filter(|n| !n.is_empty())
            });
            name.and_then(|name| {
                (!embedded)
                    .then(|| face.raw_get_name_index(name))
                    .and_then(nonzero)
                    .or_else(|| {
                        font::glyph_name_to_unicode(name)
                            .map(|unicode| face.raw_get_char_index(unicode))
                            .and_then(nonzero)
                    })
            })
            .or_else(|| nonzero(face.raw_get_char_index(u32::from(code))))
            .or_else(|| embedded.then(|| face.raw_get_code_index(code)))
            .unwrap_or(0)
        })
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::font_substitute::SubstituteStyle;

    /// `desc` selected as the indirect font dictionary `1 0 R`.
    fn active(desc: FontDescriptor) -> ActiveFont {
        ActiveFont {
            key: FontKey::Object((1, 0)),
            descriptor: Arc::new(desc),
        }
    }

    fn empty_diffs() -> [Option<Box<str>>; 256] {
        std::array::from_fn(|_| None)
    }

    /// The name-resolution path must fire ONLY for a simple embedded
    /// Type1/Type1C/CFF font.  A misfire would steal resolution from CID,
    /// symbolic TrueType, Type 3, or the non-embedded standard-14 path.
    #[test]
    fn name_resolution_dispatch_is_isolated() {
        // Simple embedded Type1 / OpenType-CFF (`Other`): the only YES cases.
        assert!(wants_name_resolution(true, true, true, PdfFontKind::Type1));
        assert!(wants_name_resolution(true, true, true, PdfFontKind::Other));

        // Non-embedded standard-14 (no font bytes): fallback face owns it.
        assert!(!wants_name_resolution(
            false,
            true,
            true,
            PdfFontKind::Type1
        ));
        // CID/Type0 (cid_encoding present → not_cid == false).
        assert!(!wants_name_resolution(
            true,
            false,
            true,
            PdfFontKind::Other
        ));
        // Type 3 (type3 present → not_type3 == false).
        assert!(!wants_name_resolution(
            true,
            true,
            false,
            PdfFontKind::Type1
        ));
        // Embedded TrueType: keeps the program's own charmap.
        assert!(!wants_name_resolution(
            true,
            true,
            true,
            PdfFontKind::TrueType
        ));
    }

    /// The simple-font name path must yield `.notdef` (0) for an unresolvable
    /// code — never code-identity.  Code-identity for a subsetted program
    /// produces wrong characters on a pixel-correct page; this pins that
    /// regression.
    #[test]
    fn unresolvable_code_is_notdef_never_identity() {
        let diffs = empty_diffs();
        // StandardEncoding leaves 0x80 unmapped: no name → no GID lookup.
        assert_eq!(
            glyph_name_for_code(&diffs, BaseEncoding::Standard, 0x80),
            None
        );
        // The control range is unmapped in every base encoding.
        for base in [
            BaseEncoding::Standard,
            BaseEncoding::WinAnsi,
            BaseEncoding::MacRoman,
        ] {
            assert_eq!(glyph_name_for_code(&diffs, base, 0), None);
            assert_eq!(glyph_name_for_code(&diffs, base, 31), None);
        }
    }

    #[test]
    fn unicode_resolution_dispatch() {
        // Non-embedded simple fonts of every kind resolve through the
        // substitute face's Unicode cmap.
        for kind in [
            PdfFontKind::Type1,
            PdfFontKind::TrueType,
            PdfFontKind::Other,
        ] {
            assert!(
                wants_unicode_resolution(false, true, true, kind),
                "{kind:?}"
            );
        }
        // Embedded: only TrueType; Type1/CFF programs resolve by name.
        assert!(wants_unicode_resolution(
            true,
            true,
            true,
            PdfFontKind::TrueType
        ));
        assert!(!wants_unicode_resolution(
            true,
            true,
            true,
            PdfFontKind::Type1
        ));
        assert!(!wants_unicode_resolution(
            true,
            true,
            true,
            PdfFontKind::Other
        ));
        // CID/Type0 and Type 3 never qualify.
        assert!(!wants_unicode_resolution(
            false,
            false,
            true,
            PdfFontKind::Other
        ));
        assert!(!wants_unicode_resolution(
            false,
            true,
            false,
            PdfFontKind::Other
        ));
    }

    /// A non-embedded font whose `Differences` array places glyph names on
    /// low codes (`0 /Q /eacute`) must draw those glyphs, not the substitute
    /// face's glyphs at code points 0 and 1.
    #[test]
    fn non_embedded_font_resolves_differences_on_substitute_face() {
        let mut differences: Box<[Option<Box<str>>; 256]> = Box::new(std::array::from_fn(|_| None));
        differences[0] = Some("Q".into());
        differences[1] = Some("eacute".into());
        let desc = FontDescriptor {
            kind: PdfFontKind::Type1,
            bytes: None,
            widths: vec![],
            first_char: 0,
            missing_width: 0,
            code_to_gid: vec![],
            differences,
            base_encoding: BaseEncoding::MacRoman,
            cid_encoding: None,
            type3: None,
            substitute: SubstituteStyle::default(),
        };
        let mut cache = FontCache::new(Arc::default());
        let face = cache
            .get_or_load(&active(desc), b"F1", [20.0, 0.0, 0.0, 20.0])
            .expect("substitute face loads");
        let q = face.raw_get_char_index(u32::from('Q'));
        let eacute = face.raw_get_char_index(0xE9);
        assert_ne!(q, 0);
        assert_ne!(eacute, 0);
        assert_eq!(face.code_to_gid.first(), Some(&q));
        assert_eq!(face.code_to_gid.get(1), Some(&eacute));
        // A code left to the base encoding resolves through its name too:
        // MacRoman 0x8E is `eacute`, which is not U+008E.
        assert_eq!(face.code_to_gid.get(0x8E), Some(&eacute));
    }

    /// An embedded non-symbolic TrueType font whose `Differences` array
    /// shifts the digit names off their ASCII codes (`0x33 /zero /one …`)
    /// draws the glyph each name selects through the (3,1) cmap, not the
    /// glyph the program's cmap assigns to the raw code.
    #[test]
    fn embedded_truetype_differences_shift_digits() {
        const DIGITS: [&str; 10] = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        ];
        let mut differences: Box<[Option<Box<str>>; 256]> = Box::new(std::array::from_fn(|_| None));
        for (slot, name) in differences[0x33..].iter_mut().zip(DIGITS) {
            *slot = Some(name.into());
        }
        let desc = FontDescriptor {
            kind: PdfFontKind::TrueType,
            bytes: Some(include_bytes!("../../../font_test_data/test_data/ttf/ahem.ttf").to_vec()),
            widths: vec![],
            first_char: 0,
            missing_width: 0,
            code_to_gid: vec![],
            differences,
            base_encoding: BaseEncoding::FontProgram,
            cid_encoding: None,
            type3: None,
            substitute: SubstituteStyle::default(),
        };
        let mut cache = FontCache::new(Arc::default());
        let face = cache
            .get_or_load(&active(desc), b"F1", [20.0, 0.0, 0.0, 20.0])
            .expect("embedded TrueType face loads");
        for (offset, digit) in ('0'..='9').enumerate() {
            let code = 0x33 + offset;
            let expected = face.raw_get_char_index(u32::from(digit));
            assert_ne!(expected, 0, "font maps U+{:04X}", u32::from(digit));
            assert_eq!(
                face.code_to_gid.get(code),
                Some(&expected),
                "code {code:#04x} draws {digit:?}"
            );
        }
    }

    /// A composite font without a program, of a known character
    /// collection, draws through a CID-indexed table built from the
    /// collection's CID → Unicode mapping and the substitute's charmap:
    /// half-width `A` (Japan1 CID 264) selects the substitute's `A`.
    #[test]
    fn non_embedded_cid_font_resolves_cids_through_unicode() {
        use crate::resources::cmap::CMap;
        use crate::resources::font::CidEncoding;
        use crate::resources::font_substitute::Script;
        let desc = FontDescriptor {
            kind: PdfFontKind::TrueType,
            bytes: None,
            widths: vec![],
            first_char: 0,
            missing_width: 0,
            code_to_gid: vec![],
            differences: Box::new(empty_diffs()),
            base_encoding: BaseEncoding::Standard,
            cid_encoding: Some(CidEncoding {
                encoding_cmap: Arc::new(CMap::identity()),
                cid_to_gid: None,
                default_width: 1000,
                widths: vec![],
                vertical: false,
                default_vertical: [880, -1000],
                vertical_metrics: vec![],
            }),
            type3: None,
            substitute: SubstituteStyle {
                script: Some(Script::Japanese),
                ..SubstituteStyle::default()
            },
        };
        let mut cache = FontCache::new(Arc::default());
        let face = cache
            .get_or_load(&active(desc), b"F1", [20.0, 0.0, 0.0, 20.0])
            .expect("substitute face loads");
        assert_eq!(face.code_to_gid.len(), 1 << 16);
        let a = face.raw_get_char_index(u32::from('A'));
        assert_ne!(a, 0);
        assert_eq!(face.code_to_gid[264], a);
        assert_eq!(face.code_to_gid[34], a);
        assert_eq!(face.code_to_gid[0], 0);
    }
}
