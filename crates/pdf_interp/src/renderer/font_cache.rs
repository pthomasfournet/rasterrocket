//! Per-page font cache.
//!
//! [`FontCache`] maps a PDF font resource name (e.g. `b"F1"`) to a loaded
//! [`FontFace`].  Faces are loaded on first use during rendering, then reused
//! for all subsequent glyphs in the same page.
//!
//! When a font has no embedded bytes the cache draws it with an installed
//! substitute of the same family class and variant (sans-serif, serif or
//! monospaced; regular, bold, italic or bold italic), located through a
//! fixed search path.
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
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use font::{
    cache::GlyphCache,
    engine::{FaceParams, FontEngine},
    face::FontFace,
    key::{FaceId, FaceIds},
    program::FontProgram,
};
use pdf::ObjectId;

use crate::resources::font::{BaseEncoding, FontDescriptor, PdfFontKind};
use crate::resources::font_substitute::{SubstituteFamily, SubstituteStyle};
use crate::resources::{FontKey, PageResources};

// ── Substitute font discovery ─────────────────────────────────────────────────

/// Installed typefaces per [`SubstituteStyle`], indexed by
/// [`substitute_index`]: `[metric-compatible free family stem, macOS
/// supplemental file stem, Windows file stem]`.
const SUBSTITUTES: [[&str; 3]; 12] = [
    ["LiberationSans-Regular", "Arial", "arial"],
    ["LiberationSans-Italic", "Arial Italic", "ariali"],
    ["LiberationSans-Bold", "Arial Bold", "arialbd"],
    ["LiberationSans-BoldItalic", "Arial Bold Italic", "arialbi"],
    ["LiberationSerif-Regular", "Times New Roman", "times"],
    ["LiberationSerif-Italic", "Times New Roman Italic", "timesi"],
    ["LiberationSerif-Bold", "Times New Roman Bold", "timesbd"],
    [
        "LiberationSerif-BoldItalic",
        "Times New Roman Bold Italic",
        "timesbi",
    ],
    ["LiberationMono-Regular", "Courier New", "cour"],
    ["LiberationMono-Italic", "Courier New Italic", "couri"],
    ["LiberationMono-Bold", "Courier New Bold", "courbd"],
    [
        "LiberationMono-BoldItalic",
        "Courier New Bold Italic",
        "courbi",
    ],
];

/// Last-resort sans-serif candidates outside the [`SUBSTITUTES`] layout.
const LAST_RESORT: &[&str] = &[
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
];

/// Row of [`SUBSTITUTES`] for `style`.
const fn substitute_index(style: SubstituteStyle) -> usize {
    let family = match style.family {
        SubstituteFamily::Sans => 0,
        SubstituteFamily::Serif => 1,
        SubstituteFamily::Mono => 2,
    };
    family * 4 + (style.bold as usize) * 2 + (style.italic as usize)
}

/// Candidate paths for one [`SUBSTITUTES`] row, in search order.
fn substitute_candidates([stem, mac, win]: [&str; 3]) -> [String; 4] {
    [
        format!("/usr/share/fonts/truetype/liberation/{stem}.ttf"),
        format!("/usr/share/fonts/liberation/{stem}.ttf"),
        format!("/System/Library/Fonts/Supplemental/{mac}.ttf"),
        format!("C:/Windows/Fonts/{win}.ttf"),
    ]
}

/// Locate the installed substitute for `style`: the exact family and
/// variant, else the family's regular face, else the sans-serif regular
/// face, else a last-resort system sans-serif.  `None` if nothing usable
/// is installed.  Resolved once per process.
fn substitute_font_path(style: SubstituteStyle) -> Option<&'static str> {
    static CACHE: OnceLock<[Option<Box<str>>; 12]> = OnceLock::new();
    let table = CACHE.get_or_init(|| {
        let installed = |row: usize| {
            substitute_candidates(SUBSTITUTES[row])
                .into_iter()
                .find(|p| PathBuf::from(p).exists())
        };
        std::array::from_fn(|row| {
            let family_regular = row - row % 4;
            installed(row)
                .or_else(|| installed(family_regular))
                .or_else(|| installed(0))
                .or_else(|| {
                    LAST_RESORT
                        .iter()
                        .find(|p| PathBuf::from(p).exists())
                        .map(|&p| p.to_owned())
                })
                .map(String::into_boxed_str)
        })
    });
    table[substitute_index(style)].as_deref()
}

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
    descriptor: OnceLock<Option<Arc<FontDescriptor>>>,
    program: OnceLock<Option<Arc<FontProgram>>>,
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
    /// Descriptor of each font dictionary selected on this page; `None` for a
    /// dictionary that does not resolve.
    descriptors: HashMap<FontKey, Option<Arc<FontDescriptor>>>,
    /// Map from (font dictionary, four f64-as-u64 bits of the Trm 2×2 matrix)
    /// to loaded face.  The full matrix is the key because the same font can
    /// appear at different sizes and skew angles within a single page.
    faces: HashMap<FaceKey, Option<FontFace>>,
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
    /// Returns `None` if the name is absent or its entry is not a dictionary.
    pub fn select(&mut self, resources: &PageResources<'_>, name: &[u8]) -> Option<ActiveFont> {
        let (key, entry) = resources.font_entry(name)?;
        let shared = &self.shared;
        let descriptor = self
            .descriptors
            .entry(key.clone())
            .or_insert_with(|| {
                let resolve = || resources.resolve_font_entry(&entry).map(Arc::new);
                match &key {
                    FontKey::Object(id) => shared.font(*id).descriptor.get_or_init(resolve).clone(),
                    FontKey::Direct { .. } => resolve(),
                }
            })
            .clone()?;
        Some(ActiveFont { key, descriptor })
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

    /// Return a reference to the [`FontFace`] for `font`, loading it on first
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
    ) -> Option<&FontFace> {
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
        &mut self,
        key: &FontKey,
        desc: &FontDescriptor,
        trm: [f64; 4],
    ) -> Option<FontFace> {
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

        let mut face = if let Some(shared) = &shared {
            let program = shared
                .program
                .get_or_init(|| parse_program(desc))
                .as_ref()?;
            let id = shared.face_id(trm.map(f64::to_bits), &self.shared.face_ids);
            self.engine
                .load_program_face_as(id, Arc::clone(program), params)
                .inspect_err(|e| log::debug!("font_cache: {e}"))
                .ok()?
        } else if let Some(bytes) = &desc.bytes {
            self.engine
                .load_memory_face(bytes.clone(), 0, params)
                .inspect_err(|e| log::debug!("font_cache: embedded font program: {e}"))
                .ok()?
        } else {
            let path = substitute_font_path(desc.substitute)?;
            self.engine
                .load_file_face(path, 0, params)
                .inspect_err(|e| log::debug!("font_cache: fallback font {path}: {e}"))
                .ok()?
        };

        let table = shared.as_ref().map_or_else(
            || resolved_code_to_gid(&face, desc),
            |shared| {
                shared
                    .code_to_gid
                    .get_or_init(|| resolved_code_to_gid(&face, desc))
                    .clone()
            },
        );
        if let Some(table) = table {
            face.code_to_gid = table;
        }
        Some(face)
    }
}

/// Parse the program a descriptor draws with: its embedded font program, or
/// the installed substitute for a font without one.  `None`, logged, when
/// neither yields a usable program.
fn parse_program(desc: &FontDescriptor) -> Option<Arc<FontProgram>> {
    let parsed = if let Some(bytes) = &desc.bytes {
        FontProgram::parse(bytes.clone(), 0)
            .inspect_err(|e| log::debug!("font_cache: embedded font program: {e}"))
    } else {
        let path = substitute_font_path(desc.substitute)?;
        std::fs::read(path)
            .map_err(font::engine::LoadError::Io)
            .and_then(|data| FontProgram::parse(data, 0))
            .inspect_err(|e| log::debug!("font_cache: fallback font {path}: {e}"))
    };
    parsed.ok().map(Arc::new)
}

/// The `code_to_gid` table a face of `desc` needs in place of the
/// descriptor's, or `None` when the descriptor's table (or the program's own
/// charmap) applies.
///
/// A simple embedded Type1/Type1C/CFF font is resolved by glyph *name*: its
/// char codes are not glyph indices (subset programs pack glyphs
/// arbitrarily) and it often has no usable Unicode cmap, so each code's name
/// (`Differences` override, else the base encoding) is looked up in the
/// program's charset.  A font drawn through a Unicode cmap is resolved
/// through the names' Unicode values.  CID/Type0 keep the descriptor's
/// table; symbolic TrueType keeps the program's own charmap.
fn resolved_code_to_gid(face: &FontFace, desc: &FontDescriptor) -> Option<Vec<u32>> {
    let embedded = desc.bytes.is_some();
    let simple = desc.cid_encoding.is_none();
    let outline = desc.type3.is_none();
    if wants_name_resolution(embedded, simple, outline, desc.kind) {
        Some(resolve_simple_to_gid(
            face,
            &desc.differences,
            desc.base_encoding,
        ))
    } else if wants_unicode_resolution(embedded, simple, outline, desc.kind)
        && face.has_unicode_cmap()
    {
        Some(resolve_via_unicode_cmap(
            face,
            &desc.differences,
            desc.base_encoding,
            embedded,
        ))
    } else {
        None
    }
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
            && let Some(unicode) = adobe_glyph_to_unicode(name)
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
/// Every other code is looked up as a raw Unicode code point, the identity
/// that holds for the ASCII and Latin-1 ranges of those encodings.
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
    (0..=u8::MAX)
        .map(|code| {
            let idx = usize::from(code);
            let name = differences[idx].as_deref().or_else(|| {
                matches!(base, BaseEncoding::WinAnsi | BaseEncoding::MacRoman)
                    .then(|| base_encoding_table(base)[idx])
                    .flatten()
                    .filter(|n| !n.is_empty())
            });
            name.and_then(adobe_glyph_to_unicode)
                .map(|unicode| face.raw_get_char_index(unicode))
                .filter(|&gid| gid != 0)
                .or_else(|| Some(face.raw_get_char_index(u32::from(code))).filter(|&gid| gid != 0))
                .or_else(|| embedded.then(|| face.raw_get_code_index(code)))
                .unwrap_or(0)
        })
        .collect()
}

/// Resolve the glyph name for one char code: the `Differences` override if
/// present, otherwise the `base` encoding's name (PDF §9.6.6 — Differences
/// take precedence over the base encoding).  `None` = no glyph at this code.
fn glyph_name_for_code(
    differences: &[Option<Box<str>>; 256],
    base: BaseEncoding,
    code: usize,
) -> Option<&str> {
    differences[code]
        .as_deref()
        .or_else(|| base_encoding_table(base)[code].filter(|n| !n.is_empty()))
}

/// Return the code→glyph-name table for a PDF base encoding (PDF Appendix D).
const fn base_encoding_table(base: BaseEncoding) -> &'static [Option<&'static str>; 256] {
    match base {
        BaseEncoding::Standard | BaseEncoding::FontProgram => &STANDARD_ENCODING,
        BaseEncoding::WinAnsi => &WIN_ANSI_ENCODING,
        BaseEncoding::MacRoman => &MAC_ROMAN_ENCODING,
    }
}

// ── Adobe Glyph List ─────────────────────────────────────────────────────────

/// Adobe Glyph List subset: glyph name → Unicode codepoint, strictly
/// ascending by name in byte order.
///
/// Used by [`adobe_glyph_to_unicode`] for binary search: an entry out of
/// order is unreachable, as are the entries the search steps past because of
/// it.  Implemented as a module-level static rather than a local `static`
/// inside the function to avoid a `clippy::items_after_statements` suppress
/// and to make the table visually distinct from the lookup logic.
///
/// A `match` expression covering every entry compiles to an LLVM decision
/// tree that costs hundreds of MiB of working memory per codegen unit in debug
/// mode (OOM'd on this machine); a static array is pure data, zero LLVM IR.
static AGL: &[(&str, u32)] = &[
    ("A", 0x0041),
    ("AE", 0x00C6),
    ("Aacute", 0x00C1),
    ("Abreve", 0x0102),
    ("Acircumflex", 0x00C2),
    ("Adieresis", 0x00C4),
    ("Agrave", 0x00C0),
    ("Alpha", 0x0391),
    ("Amacron", 0x0100),
    ("Aogonek", 0x0104),
    ("Aring", 0x00C5),
    ("Atilde", 0x00C3),
    ("B", 0x0042),
    ("Beta", 0x0392),
    ("C", 0x0043),
    ("Cacute", 0x0106),
    ("Ccaron", 0x010C),
    ("Ccedilla", 0x00C7),
    ("Ccircumflex", 0x0108),
    ("Cdotaccent", 0x010A),
    ("Chi", 0x03A7),
    ("D", 0x0044),
    ("Dcaron", 0x010E),
    ("Dcroat", 0x0110),
    ("Delta", 0x0394),
    ("E", 0x0045),
    ("Eacute", 0x00C9),
    ("Ebreve", 0x0114),
    ("Ecaron", 0x011A),
    ("Ecircumflex", 0x00CA),
    ("Edieresis", 0x00CB),
    ("Edotaccent", 0x0116),
    ("Egrave", 0x00C8),
    ("Emacron", 0x0112),
    ("Eogonek", 0x0118),
    ("Epsilon", 0x0395),
    ("Eta", 0x0397),
    ("Eth", 0x00D0),
    ("Euro", 0x20AC),
    ("F", 0x0046),
    ("G", 0x0047),
    ("Gamma", 0x0393),
    ("Gcircumflex", 0x011C),
    ("H", 0x0048),
    ("I", 0x0049),
    ("Iacute", 0x00CD),
    ("Icircumflex", 0x00CE),
    ("Idieresis", 0x00CF),
    ("Igrave", 0x00CC),
    ("Iota", 0x0399),
    ("J", 0x004A),
    ("K", 0x004B),
    ("Kappa", 0x039A),
    ("L", 0x004C),
    ("Lambda", 0x039B),
    ("Lslash", 0x0141),
    ("M", 0x004D),
    ("Mu", 0x039C),
    ("N", 0x004E),
    ("Nacute", 0x0143),
    ("Ncaron", 0x0147),
    ("Ntilde", 0x00D1),
    ("Nu", 0x039D),
    ("O", 0x004F),
    ("OE", 0x0152),
    ("Oacute", 0x00D3),
    ("Ocircumflex", 0x00D4),
    ("Odieresis", 0x00D6),
    ("Ograve", 0x00D2),
    ("Omega", 0x03A9),
    ("Omicron", 0x039F),
    ("Oslash", 0x00D8),
    ("Otilde", 0x00D5),
    ("P", 0x0050),
    ("Phi", 0x03A6),
    ("Pi", 0x03A0),
    ("Psi", 0x03A8),
    ("Q", 0x0051),
    ("R", 0x0052),
    ("Racute", 0x0154),
    ("Rcaron", 0x0158),
    ("Rho", 0x03A1),
    ("S", 0x0053),
    ("Sacute", 0x015A),
    ("Scaron", 0x0160),
    ("Scedilla", 0x015E),
    ("Scircumflex", 0x015C),
    ("Sigma", 0x03A3),
    ("T", 0x0054),
    ("Tau", 0x03A4),
    ("Tcaron", 0x0164),
    ("Theta", 0x0398),
    ("Thorn", 0x00DE),
    ("U", 0x0055),
    ("Uacute", 0x00DA),
    ("Ucircumflex", 0x00DB),
    ("Udieresis", 0x00DC),
    ("Ugrave", 0x00D9),
    ("Uhungarumlaut", 0x0170),
    ("Uogonek", 0x0172),
    ("Upsilon", 0x03A5),
    ("Uring", 0x016E),
    ("V", 0x0056),
    ("W", 0x0057),
    ("X", 0x0058),
    ("Xi", 0x039E),
    ("Y", 0x0059),
    ("Yacute", 0x00DD),
    ("Ydieresis", 0x0178),
    ("Z", 0x005A),
    ("Zacute", 0x0179),
    ("Zcaron", 0x017D),
    ("Zdotaccent", 0x017B),
    ("Zeta", 0x0396),
    ("a", 0x0061),
    ("aacute", 0x00E1),
    ("abreve", 0x0103),
    ("acircumflex", 0x00E2),
    ("acute", 0x00B4),
    ("adieresis", 0x00E4),
    ("ae", 0x00E6),
    ("agrave", 0x00E0),
    ("alpha", 0x03B1),
    ("amacron", 0x0101),
    ("ampersand", 0x0026),
    ("aogonek", 0x0105),
    ("approxequal", 0x2248),
    ("aring", 0x00E5),
    ("asciicircum", 0x005E),
    ("asciitilde", 0x007E),
    ("asterisk", 0x002A),
    ("at", 0x0040),
    ("atilde", 0x00E3),
    ("b", 0x0062),
    ("backslash", 0x005C),
    ("bar", 0x007C),
    ("beta", 0x03B2),
    ("braceleft", 0x007B),
    ("braceright", 0x007D),
    ("bracketleft", 0x005B),
    ("bracketright", 0x005D),
    ("breve", 0x02D8),
    ("brokenbar", 0x00A6),
    ("bullet", 0x2022),
    ("c", 0x0063),
    ("cacute", 0x0107),
    ("caron", 0x02C7),
    ("ccaron", 0x010D),
    ("ccedilla", 0x00E7),
    ("ccircumflex", 0x0109),
    ("cdotaccent", 0x010B),
    ("cedilla", 0x00B8),
    ("cent", 0x00A2),
    ("chi", 0x03C7),
    ("circumflex", 0x02C6),
    ("colon", 0x003A),
    ("comma", 0x002C),
    ("copyright", 0x00A9),
    ("currency", 0x00A4),
    ("d", 0x0064),
    ("dagger", 0x2020),
    ("daggerdbl", 0x2021),
    ("dcaron", 0x010F),
    ("dcroat", 0x0111),
    ("degree", 0x00B0),
    ("delta", 0x03B4),
    ("dieresis", 0x00A8),
    ("divide", 0x00F7),
    ("dollar", 0x0024),
    ("dotaccent", 0x02D9),
    ("e", 0x0065),
    ("eacute", 0x00E9),
    ("ebreve", 0x0115),
    ("ecaron", 0x011B),
    ("ecircumflex", 0x00EA),
    ("edieresis", 0x00EB),
    ("edotaccent", 0x0117),
    ("egrave", 0x00E8),
    ("eight", 0x0038),
    ("ellipsis", 0x2026),
    ("emacron", 0x0113),
    ("emdash", 0x2014),
    ("endash", 0x2013),
    ("eogonek", 0x0119),
    ("epsilon", 0x03B5),
    ("equal", 0x003D),
    ("eta", 0x03B7),
    ("eth", 0x00F0),
    ("exclam", 0x0021),
    ("exclamdown", 0x00A1),
    ("f", 0x0066),
    ("fi", 0xFB01),
    ("five", 0x0035),
    ("fl", 0xFB02),
    ("four", 0x0034),
    ("fraction", 0x2044),
    ("g", 0x0067),
    ("gamma", 0x03B3),
    ("gcircumflex", 0x011D),
    ("germandbls", 0x00DF),
    ("grave", 0x0060),
    ("greater", 0x003E),
    ("greaterequal", 0x2265),
    ("guillemotleft", 0x00AB),
    ("guillemotright", 0x00BB),
    ("guilsinglleft", 0x2039),
    ("guilsinglright", 0x203A),
    ("h", 0x0068),
    ("hungarumlaut", 0x02DD),
    ("hyphen", 0x002D),
    ("i", 0x0069),
    ("iacute", 0x00ED),
    ("icircumflex", 0x00EE),
    ("idieresis", 0x00EF),
    ("igrave", 0x00EC),
    ("infinity", 0x221E),
    ("integral", 0x222B),
    ("iota", 0x03B9),
    ("j", 0x006A),
    ("k", 0x006B),
    ("kappa", 0x03BA),
    ("l", 0x006C),
    ("lambda", 0x03BB),
    ("less", 0x003C),
    ("lessequal", 0x2264),
    ("logicalnot", 0x00AC),
    ("lozenge", 0x25CA),
    ("lslash", 0x0142),
    ("m", 0x006D),
    ("macron", 0x00AF),
    ("minus", 0x2212),
    ("mu", 0x00B5),
    ("multiply", 0x00D7),
    ("n", 0x006E),
    ("nacute", 0x0144),
    ("ncaron", 0x0148),
    ("nine", 0x0039),
    ("notequal", 0x2260),
    ("ntilde", 0x00F1),
    ("nu", 0x03BD),
    ("numbersign", 0x0023),
    ("o", 0x006F),
    ("oacute", 0x00F3),
    ("ocircumflex", 0x00F4),
    ("odieresis", 0x00F6),
    ("oe", 0x0153),
    ("ogonek", 0x02DB),
    ("ograve", 0x00F2),
    ("omicron", 0x03BF),
    ("one", 0x0031),
    ("onehalf", 0x00BD),
    ("onequarter", 0x00BC),
    ("onesuperior", 0x00B9),
    ("ordfeminine", 0x00AA),
    ("ordmasculine", 0x00BA),
    ("oslash", 0x00F8),
    ("otilde", 0x00F5),
    ("p", 0x0070),
    ("paragraph", 0x00B6),
    ("parenleft", 0x0028),
    ("parenright", 0x0029),
    ("partialdiff", 0x2202),
    ("percent", 0x0025),
    ("period", 0x002E),
    ("periodcentered", 0x00B7),
    ("perthousand", 0x2030),
    ("phi", 0x03C6),
    ("pi", 0x03C0),
    ("plus", 0x002B),
    ("plusminus", 0x00B1),
    ("product", 0x220F),
    ("psi", 0x03C8),
    ("q", 0x0071),
    ("question", 0x003F),
    ("questiondown", 0x00BF),
    ("quotedbl", 0x0022),
    ("quotedblbase", 0x201E),
    ("quotedblleft", 0x201C),
    ("quotedblright", 0x201D),
    ("quoteleft", 0x2018),
    ("quoteright", 0x2019),
    ("quotesinglbase", 0x201A),
    ("quotesingle", 0x0027),
    ("r", 0x0072),
    ("racute", 0x0155),
    ("radical", 0x221A),
    ("rcaron", 0x0159),
    ("registered", 0x00AE),
    ("rho", 0x03C1),
    ("ring", 0x02DA),
    ("s", 0x0073),
    ("sacute", 0x015B),
    ("scaron", 0x0161),
    ("scedilla", 0x015F),
    ("scircumflex", 0x015D),
    ("section", 0x00A7),
    ("semicolon", 0x003B),
    ("seven", 0x0037),
    ("sigma", 0x03C3),
    ("sigma1", 0x03C2),
    ("six", 0x0036),
    ("slash", 0x002F),
    ("space", 0x0020),
    ("sterling", 0x00A3),
    ("summation", 0x2211),
    ("t", 0x0074),
    ("tau", 0x03C4),
    ("tcaron", 0x0165),
    ("theta", 0x03B8),
    ("thorn", 0x00FE),
    ("three", 0x0033),
    ("threequarters", 0x00BE),
    ("threesuperior", 0x00B3),
    ("tilde", 0x02DC),
    ("two", 0x0032),
    ("twosuperior", 0x00B2),
    ("u", 0x0075),
    ("uacute", 0x00FA),
    ("ucircumflex", 0x00FB),
    ("udieresis", 0x00FC),
    ("ugrave", 0x00F9),
    ("uhungarumlaut", 0x0171),
    ("underscore", 0x005F),
    ("uogonek", 0x0173),
    ("upsilon", 0x03C5),
    ("uring", 0x016F),
    ("v", 0x0076),
    ("w", 0x0077),
    ("x", 0x0078),
    ("xi", 0x03BE),
    ("y", 0x0079),
    ("yacute", 0x00FD),
    ("ydieresis", 0x00FF),
    ("yen", 0x00A5),
    ("z", 0x007A),
    ("zacute", 0x017A),
    ("zcaron", 0x017E),
    ("zdotaccent", 0x017C),
    ("zero", 0x0030),
    ("zeta", 0x03B6),
];

/// Map an Adobe glyph name to its Unicode codepoint via [`AGL`].
///
/// Fast paths handle the `uni<XXXX>` (4- or 8-digit) and `u<XXXX>` (4–6 digit)
/// naming conventions that encode the codepoint directly without a table lookup.
fn adobe_glyph_to_unicode(name: &str) -> Option<u32> {
    // "uni" prefix: exactly 4 hex digits (BMP) or 8 hex digits (non-BMP).
    if let Some(hex) = name
        .strip_prefix("uni")
        .filter(|s| (s.len() == 4 || s.len() == 8) && s.chars().all(|c| c.is_ascii_hexdigit()))
    {
        return u32::from_str_radix(hex, 16).ok();
    }
    // "u" prefix: 4–6 hex digits.
    if let Some(hex) = name
        .strip_prefix('u')
        .filter(|s| (4..=6).contains(&s.len()) && s.chars().all(|c| c.is_ascii_hexdigit()))
    {
        return u32::from_str_radix(hex, 16).ok();
    }
    AGL.binary_search_by_key(&name, |&(k, _)| k)
        .ok()
        .map(|i| AGL[i].1)
}

// ── PDF base-encoding code → glyph-name tables (PDF 1.7 Appendix D.2) ─────────
//
// Static arrays, not `match`: a 256-arm `match` per encoding compiles to a
// large LLVM decision tree that OOMs debug codegen on this box (the same
// reason `AGL` above is a static).  These are pure data, zero LLVM IR.

/// Adobe `StandardEncoding`: char code → glyph name (`None` = unused code).
static STANDARD_ENCODING: [Option<&str>; 256] = [
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    Some("space"),
    Some("exclam"),
    Some("quotedbl"),
    Some("numbersign"),
    Some("dollar"),
    Some("percent"),
    Some("ampersand"),
    Some("quoteright"),
    Some("parenleft"),
    Some("parenright"),
    Some("asterisk"),
    Some("plus"),
    Some("comma"),
    Some("hyphen"),
    Some("period"),
    Some("slash"),
    Some("zero"),
    Some("one"),
    Some("two"),
    Some("three"),
    Some("four"),
    Some("five"),
    Some("six"),
    Some("seven"),
    Some("eight"),
    Some("nine"),
    Some("colon"),
    Some("semicolon"),
    Some("less"),
    Some("equal"),
    Some("greater"),
    Some("question"),
    Some("at"),
    Some("A"),
    Some("B"),
    Some("C"),
    Some("D"),
    Some("E"),
    Some("F"),
    Some("G"),
    Some("H"),
    Some("I"),
    Some("J"),
    Some("K"),
    Some("L"),
    Some("M"),
    Some("N"),
    Some("O"),
    Some("P"),
    Some("Q"),
    Some("R"),
    Some("S"),
    Some("T"),
    Some("U"),
    Some("V"),
    Some("W"),
    Some("X"),
    Some("Y"),
    Some("Z"),
    Some("bracketleft"),
    Some("backslash"),
    Some("bracketright"),
    Some("asciicircum"),
    Some("underscore"),
    Some("quoteleft"),
    Some("a"),
    Some("b"),
    Some("c"),
    Some("d"),
    Some("e"),
    Some("f"),
    Some("g"),
    Some("h"),
    Some("i"),
    Some("j"),
    Some("k"),
    Some("l"),
    Some("m"),
    Some("n"),
    Some("o"),
    Some("p"),
    Some("q"),
    Some("r"),
    Some("s"),
    Some("t"),
    Some("u"),
    Some("v"),
    Some("w"),
    Some("x"),
    Some("y"),
    Some("z"),
    Some("braceleft"),
    Some("bar"),
    Some("braceright"),
    Some("asciitilde"),
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    Some("exclamdown"),
    Some("cent"),
    Some("sterling"),
    Some("fraction"),
    Some("yen"),
    Some("florin"),
    Some("section"),
    Some("currency"),
    Some("quotesingle"),
    Some("quotedblleft"),
    Some("guillemotleft"),
    Some("guilsinglleft"),
    Some("guilsinglright"),
    Some("fi"),
    Some("fl"),
    None,
    Some("endash"),
    Some("dagger"),
    Some("daggerdbl"),
    Some("periodcentered"),
    None,
    Some("paragraph"),
    Some("bullet"),
    Some("quotesinglbase"),
    Some("quotedblbase"),
    Some("quotedblright"),
    Some("guillemotright"),
    Some("ellipsis"),
    Some("perthousand"),
    None,
    Some("questiondown"),
    None,
    Some("grave"),
    Some("acute"),
    Some("circumflex"),
    Some("tilde"),
    Some("macron"),
    Some("breve"),
    Some("dotaccent"),
    Some("dieresis"),
    None,
    Some("ring"),
    Some("cedilla"),
    None,
    Some("hungarumlaut"),
    Some("ogonek"),
    Some("caron"),
    Some("emdash"),
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    Some("AE"),
    None,
    Some("ordfeminine"),
    None,
    None,
    None,
    None,
    Some("Lslash"),
    Some("Oslash"),
    Some("OE"),
    Some("ordmasculine"),
    None,
    None,
    None,
    None,
    None,
    Some("ae"),
    None,
    None,
    None,
    Some("dotlessi"),
    None,
    None,
    Some("lslash"),
    Some("oslash"),
    Some("oe"),
    Some("germandbls"),
    None,
    None,
    None,
    None,
];

/// `WinAnsiEncoding` (Windows CP-1252 superset): char code → glyph name.
static WIN_ANSI_ENCODING: [Option<&str>; 256] = [
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    Some("space"),
    Some("exclam"),
    Some("quotedbl"),
    Some("numbersign"),
    Some("dollar"),
    Some("percent"),
    Some("ampersand"),
    Some("quotesingle"),
    Some("parenleft"),
    Some("parenright"),
    Some("asterisk"),
    Some("plus"),
    Some("comma"),
    Some("hyphen"),
    Some("period"),
    Some("slash"),
    Some("zero"),
    Some("one"),
    Some("two"),
    Some("three"),
    Some("four"),
    Some("five"),
    Some("six"),
    Some("seven"),
    Some("eight"),
    Some("nine"),
    Some("colon"),
    Some("semicolon"),
    Some("less"),
    Some("equal"),
    Some("greater"),
    Some("question"),
    Some("at"),
    Some("A"),
    Some("B"),
    Some("C"),
    Some("D"),
    Some("E"),
    Some("F"),
    Some("G"),
    Some("H"),
    Some("I"),
    Some("J"),
    Some("K"),
    Some("L"),
    Some("M"),
    Some("N"),
    Some("O"),
    Some("P"),
    Some("Q"),
    Some("R"),
    Some("S"),
    Some("T"),
    Some("U"),
    Some("V"),
    Some("W"),
    Some("X"),
    Some("Y"),
    Some("Z"),
    Some("bracketleft"),
    Some("backslash"),
    Some("bracketright"),
    Some("asciicircum"),
    Some("underscore"),
    Some("grave"),
    Some("a"),
    Some("b"),
    Some("c"),
    Some("d"),
    Some("e"),
    Some("f"),
    Some("g"),
    Some("h"),
    Some("i"),
    Some("j"),
    Some("k"),
    Some("l"),
    Some("m"),
    Some("n"),
    Some("o"),
    Some("p"),
    Some("q"),
    Some("r"),
    Some("s"),
    Some("t"),
    Some("u"),
    Some("v"),
    Some("w"),
    Some("x"),
    Some("y"),
    Some("z"),
    Some("braceleft"),
    Some("bar"),
    Some("braceright"),
    Some("asciitilde"),
    None,
    Some("Euro"),
    None,
    Some("quotesinglbase"),
    Some("florin"),
    Some("quotedblbase"),
    Some("ellipsis"),
    Some("dagger"),
    Some("daggerdbl"),
    Some("circumflex"),
    Some("perthousand"),
    Some("Scaron"),
    Some("guilsinglleft"),
    Some("OE"),
    None,
    Some("Zcaron"),
    None,
    None,
    Some("quoteleft"),
    Some("quoteright"),
    Some("quotedblleft"),
    Some("quotedblright"),
    Some("bullet"),
    Some("endash"),
    Some("emdash"),
    Some("tilde"),
    Some("trademark"),
    Some("scaron"),
    Some("guilsinglright"),
    Some("oe"),
    None,
    Some("zcaron"),
    Some("Ydieresis"),
    // 0xA0: WinAnsiEncoding maps the no-break space to `space` (PDF App. D.2);
    // a bare NBSP in body text must be a blank glyph, not `.notdef`.
    Some("space"),
    Some("exclamdown"),
    Some("cent"),
    Some("sterling"),
    Some("currency"),
    Some("yen"),
    Some("brokenbar"),
    Some("section"),
    Some("dieresis"),
    Some("copyright"),
    Some("ordfeminine"),
    Some("guillemotleft"),
    Some("logicalnot"),
    // 0xAD: WinAnsiEncoding maps the soft hyphen to `hyphen` (PDF App. D.2);
    // omitting it drops discretionary hyphens to `.notdef` mid-word.
    Some("hyphen"),
    Some("registered"),
    Some("macron"),
    Some("degree"),
    Some("plusminus"),
    Some("twosuperior"),
    Some("threesuperior"),
    Some("acute"),
    Some("mu"),
    Some("paragraph"),
    Some("periodcentered"),
    Some("cedilla"),
    Some("onesuperior"),
    Some("ordmasculine"),
    Some("guillemotright"),
    Some("onequarter"),
    Some("onehalf"),
    Some("threequarters"),
    Some("questiondown"),
    Some("Agrave"),
    Some("Aacute"),
    Some("Acircumflex"),
    Some("Atilde"),
    Some("Adieresis"),
    Some("Aring"),
    Some("AE"),
    Some("Ccedilla"),
    Some("Egrave"),
    Some("Eacute"),
    Some("Ecircumflex"),
    Some("Edieresis"),
    Some("Igrave"),
    Some("Iacute"),
    Some("Icircumflex"),
    Some("Idieresis"),
    Some("Eth"),
    Some("Ntilde"),
    Some("Ograve"),
    Some("Oacute"),
    Some("Ocircumflex"),
    Some("Otilde"),
    Some("Odieresis"),
    Some("multiply"),
    Some("Oslash"),
    Some("Ugrave"),
    Some("Uacute"),
    Some("Ucircumflex"),
    Some("Udieresis"),
    Some("Yacute"),
    Some("Thorn"),
    Some("germandbls"),
    Some("agrave"),
    Some("aacute"),
    Some("acircumflex"),
    Some("atilde"),
    Some("adieresis"),
    Some("aring"),
    Some("ae"),
    Some("ccedilla"),
    Some("egrave"),
    Some("eacute"),
    Some("ecircumflex"),
    Some("edieresis"),
    Some("igrave"),
    Some("iacute"),
    Some("icircumflex"),
    Some("idieresis"),
    Some("eth"),
    Some("ntilde"),
    Some("ograve"),
    Some("oacute"),
    Some("ocircumflex"),
    Some("otilde"),
    Some("odieresis"),
    Some("divide"),
    Some("oslash"),
    Some("ugrave"),
    Some("uacute"),
    Some("ucircumflex"),
    Some("udieresis"),
    Some("yacute"),
    Some("thorn"),
    Some("ydieresis"),
];

/// `MacRomanEncoding` (Mac OS Roman): char code → glyph name.
///
/// The C2–D7 mathematical-symbol slots (notequal, infinity, lessequal,
/// greaterequal, partialdiff, summation, product, pi, integral, Omega,
/// radical, approxequal, Delta, lozenge) and the 0xCA no-break space are
/// part of PDF Appendix D.2; leaving them unmapped silently routed those
/// glyphs to `.notdef` in MacRoman-encoded text.  0xF0 (the Apple-logo
/// glyph) is intentionally absent: it is a non-spec vendor extension with
/// no Adobe-Glyph-List name, so it can never resolve regardless.
static MAC_ROMAN_ENCODING: [Option<&str>; 256] = [
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    Some("space"),
    Some("exclam"),
    Some("quotedbl"),
    Some("numbersign"),
    Some("dollar"),
    Some("percent"),
    Some("ampersand"),
    Some("quotesingle"),
    Some("parenleft"),
    Some("parenright"),
    Some("asterisk"),
    Some("plus"),
    Some("comma"),
    Some("hyphen"),
    Some("period"),
    Some("slash"),
    Some("zero"),
    Some("one"),
    Some("two"),
    Some("three"),
    Some("four"),
    Some("five"),
    Some("six"),
    Some("seven"),
    Some("eight"),
    Some("nine"),
    Some("colon"),
    Some("semicolon"),
    Some("less"),
    Some("equal"),
    Some("greater"),
    Some("question"),
    Some("at"),
    Some("A"),
    Some("B"),
    Some("C"),
    Some("D"),
    Some("E"),
    Some("F"),
    Some("G"),
    Some("H"),
    Some("I"),
    Some("J"),
    Some("K"),
    Some("L"),
    Some("M"),
    Some("N"),
    Some("O"),
    Some("P"),
    Some("Q"),
    Some("R"),
    Some("S"),
    Some("T"),
    Some("U"),
    Some("V"),
    Some("W"),
    Some("X"),
    Some("Y"),
    Some("Z"),
    Some("bracketleft"),
    Some("backslash"),
    Some("bracketright"),
    Some("asciicircum"),
    Some("underscore"),
    Some("grave"),
    Some("a"),
    Some("b"),
    Some("c"),
    Some("d"),
    Some("e"),
    Some("f"),
    Some("g"),
    Some("h"),
    Some("i"),
    Some("j"),
    Some("k"),
    Some("l"),
    Some("m"),
    Some("n"),
    Some("o"),
    Some("p"),
    Some("q"),
    Some("r"),
    Some("s"),
    Some("t"),
    Some("u"),
    Some("v"),
    Some("w"),
    Some("x"),
    Some("y"),
    Some("z"),
    Some("braceleft"),
    Some("bar"),
    Some("braceright"),
    Some("asciitilde"),
    None,
    Some("Adieresis"),
    Some("Aring"),
    Some("Ccedilla"),
    Some("Eacute"),
    Some("Ntilde"),
    Some("Odieresis"),
    Some("Udieresis"),
    Some("aacute"),
    Some("agrave"),
    Some("acircumflex"),
    Some("adieresis"),
    Some("atilde"),
    Some("aring"),
    Some("ccedilla"),
    Some("eacute"),
    Some("egrave"),
    Some("ecircumflex"),
    Some("edieresis"),
    Some("iacute"),
    Some("igrave"),
    Some("icircumflex"),
    Some("idieresis"),
    Some("ntilde"),
    Some("oacute"),
    Some("ograve"),
    Some("ocircumflex"),
    Some("odieresis"),
    Some("otilde"),
    Some("uacute"),
    Some("ugrave"),
    Some("ucircumflex"),
    Some("udieresis"),
    Some("dagger"),
    Some("degree"),
    Some("cent"),
    Some("sterling"),
    Some("section"),
    Some("bullet"),
    Some("paragraph"),
    Some("germandbls"),
    Some("registered"),
    Some("copyright"),
    Some("trademark"),
    Some("acute"),
    Some("dieresis"),
    Some("notequal"),
    Some("AE"),
    Some("Oslash"),
    Some("infinity"),
    Some("plusminus"),
    Some("lessequal"),
    Some("greaterequal"),
    Some("yen"),
    Some("mu"),
    Some("partialdiff"),
    Some("summation"),
    Some("product"),
    Some("pi"),
    Some("integral"),
    Some("ordfeminine"),
    Some("ordmasculine"),
    Some("Omega"),
    Some("ae"),
    Some("oslash"),
    Some("questiondown"),
    Some("exclamdown"),
    Some("logicalnot"),
    Some("radical"),
    Some("florin"),
    Some("approxequal"),
    Some("Delta"),
    Some("guillemotleft"),
    Some("guillemotright"),
    Some("ellipsis"),
    Some("space"),
    Some("Agrave"),
    Some("Atilde"),
    Some("Otilde"),
    Some("OE"),
    Some("oe"),
    Some("endash"),
    Some("emdash"),
    Some("quotedblleft"),
    Some("quotedblright"),
    Some("quoteleft"),
    Some("quoteright"),
    Some("divide"),
    Some("lozenge"),
    Some("ydieresis"),
    Some("Ydieresis"),
    Some("fraction"),
    Some("currency"),
    Some("guilsinglleft"),
    Some("guilsinglright"),
    Some("fi"),
    Some("fl"),
    Some("daggerdbl"),
    Some("periodcentered"),
    Some("quotesinglbase"),
    Some("quotedblbase"),
    Some("perthousand"),
    Some("Acircumflex"),
    Some("Ecircumflex"),
    Some("Aacute"),
    Some("Edieresis"),
    Some("Egrave"),
    Some("Iacute"),
    Some("Icircumflex"),
    Some("Idieresis"),
    Some("Igrave"),
    Some("Oacute"),
    Some("Ocircumflex"),
    None,
    Some("Ograve"),
    Some("Uacute"),
    Some("Ucircumflex"),
    Some("Ugrave"),
    Some("dotlessi"),
    Some("circumflex"),
    Some("tilde"),
    Some("macron"),
    Some("breve"),
    Some("dotaccent"),
    Some("ring"),
    Some("cedilla"),
    Some("hungarumlaut"),
    Some("ogonek"),
    Some("caron"),
];

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn base_encoding_known_codes() {
        // WinAnsi: ASCII letters are identity-positioned; 0x80 = Euro; 0xC9 =
        // Eacute (CP-1252 high range).
        assert_eq!(WIN_ANSI_ENCODING[0x41], Some("A"));
        assert_eq!(WIN_ANSI_ENCODING[0x61], Some("a"));
        assert_eq!(WIN_ANSI_ENCODING[0x80], Some("Euro"));
        assert_eq!(WIN_ANSI_ENCODING[0xC9], Some("Eacute"));

        // StandardEncoding: 0x27 is quoteright (not quotesingle), 0x60 is
        // quoteleft — the classic distinction WinAnsi does not make.
        assert_eq!(STANDARD_ENCODING[0x27], Some("quoteright"));
        assert_eq!(STANDARD_ENCODING[0x60], Some("quoteleft"));
        assert_eq!(STANDARD_ENCODING[0x41], Some("A"));

        // MacRoman: ASCII identity; 0xC7 = guillemotleft (Mac OS Roman).
        assert_eq!(MAC_ROMAN_ENCODING[0x41], Some("A"));
        assert_eq!(MAC_ROMAN_ENCODING[0xC7], Some("guillemotleft"));
    }

    /// PDF Appendix D.2 conformance for the slots that *differ* between the
    /// three base encodings and the slots a single wrong byte would turn into
    /// a systematic silent-wrong-glyph defect.  A wrong entry here mis-maps
    /// that code in every PDF using that base encoding, so pin them hard.
    #[test]
    fn base_encoding_appendix_d2_differing_slots() {
        // All three tables are exactly 256 entries (index == char code).
        assert_eq!(STANDARD_ENCODING.len(), 256);
        assert_eq!(WIN_ANSI_ENCODING.len(), 256);
        assert_eq!(MAC_ROMAN_ENCODING.len(), 256);

        // 0x27 / 0x60: the classic quote distinction.  Standard uses the
        // typographic quoteright/quoteleft; WinAnsi & MacRoman use the
        // straight quotesingle/grave.
        assert_eq!(STANDARD_ENCODING[0x27], Some("quoteright"));
        assert_eq!(WIN_ANSI_ENCODING[0x27], Some("quotesingle"));
        assert_eq!(MAC_ROMAN_ENCODING[0x27], Some("quotesingle"));
        assert_eq!(STANDARD_ENCODING[0x60], Some("quoteleft"));
        assert_eq!(WIN_ANSI_ENCODING[0x60], Some("grave"));
        assert_eq!(MAC_ROMAN_ENCODING[0x60], Some("grave"));

        // WinAnsi CP-1252 high specials (0x80–0x9F) the other two leave unset.
        assert_eq!(WIN_ANSI_ENCODING[0x80], Some("Euro"));
        assert_eq!(WIN_ANSI_ENCODING[0x85], Some("ellipsis"));
        assert_eq!(WIN_ANSI_ENCODING[0x91], Some("quoteleft"));
        assert_eq!(WIN_ANSI_ENCODING[0x92], Some("quoteright"));
        assert_eq!(WIN_ANSI_ENCODING[0x96], Some("endash"));
        assert_eq!(WIN_ANSI_ENCODING[0x97], Some("emdash"));
        assert_eq!(STANDARD_ENCODING[0x80], None);
        assert_eq!(MAC_ROMAN_ENCODING[0x80], Some("Adieresis"));

        // Spec-mandated slots that were previously unmapped (the bug this
        // hardening pass fixed): WinAnsi NBSP/soft-hyphen and the MacRoman
        // C2–D7 math symbols + 0xCA no-break space.
        assert_eq!(WIN_ANSI_ENCODING[0xA0], Some("space"));
        assert_eq!(WIN_ANSI_ENCODING[0xAD], Some("hyphen"));
        assert_eq!(MAC_ROMAN_ENCODING[0xAD], Some("notequal"));
        assert_eq!(MAC_ROMAN_ENCODING[0xB0], Some("infinity"));
        assert_eq!(MAC_ROMAN_ENCODING[0xB2], Some("lessequal"));
        assert_eq!(MAC_ROMAN_ENCODING[0xB3], Some("greaterequal"));
        assert_eq!(MAC_ROMAN_ENCODING[0xBD], Some("Omega"));
        assert_eq!(MAC_ROMAN_ENCODING[0xC3], Some("radical"));
        assert_eq!(MAC_ROMAN_ENCODING[0xCA], Some("space"));
        assert_eq!(MAC_ROMAN_ENCODING[0xD7], Some("lozenge"));

        // MacRoman accented-letter high range (Mac OS Roman layout, not
        // CP-1252) — proves the table is the Mac vector, not a WinAnsi copy.
        assert_eq!(MAC_ROMAN_ENCODING[0x81], Some("Aring"));
        assert_eq!(MAC_ROMAN_ENCODING[0xAE], Some("AE"));
        assert_eq!(MAC_ROMAN_ENCODING[0xD0], Some("endash"));
        assert_eq!(WIN_ANSI_ENCODING[0xD0], Some("Eth"));

        // 0xF0: the Apple-logo glyph is a non-spec vendor extension absent
        // from PDF Appendix D.2 and the AGL — it stays unmapped by design.
        assert_eq!(MAC_ROMAN_ENCODING[0xF0], None);

        // StandardEncoding high-range distinctives (PDF App. D.2).
        assert_eq!(STANDARD_ENCODING[0xA1], Some("exclamdown"));
        assert_eq!(STANDARD_ENCODING[0xAB], Some("guillemotleft"));
        assert_eq!(STANDARD_ENCODING[0xD0], Some("emdash"));
        assert_eq!(STANDARD_ENCODING[0xE1], Some("AE"));
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
    fn differences_override_base_encoding() {
        let mut diffs = empty_diffs();
        // pdfTeX idiom: code 2 → /fi, code 39 → /quoteright.
        diffs[2] = Some("fi".to_string().into_boxed_str());
        diffs[39] = Some("quoteright".to_string().into_boxed_str());

        // Code 2 has no WinAnsi name; the Differences override supplies "fi".
        assert_eq!(
            glyph_name_for_code(&diffs, BaseEncoding::WinAnsi, 2),
            Some("fi")
        );
        // Code 39 IS WinAnsi 'quotesingle', but Differences wins → quoteright.
        assert_eq!(
            glyph_name_for_code(&diffs, BaseEncoding::WinAnsi, 39),
            Some("quoteright")
        );
        // Code 65 has no override → falls back to the base encoding ('A').
        assert_eq!(
            glyph_name_for_code(&diffs, BaseEncoding::WinAnsi, 65),
            Some("A")
        );
    }

    #[test]
    fn unmapped_code_yields_no_name() {
        let diffs = empty_diffs();
        // StandardEncoding leaves the C0 control range unmapped: no glyph.
        assert_eq!(glyph_name_for_code(&diffs, BaseEncoding::Standard, 0), None);
        // 0x80 is unmapped in StandardEncoding (unlike WinAnsi's Euro).
        assert_eq!(
            glyph_name_for_code(&diffs, BaseEncoding::Standard, 0x80),
            None
        );
    }

    #[test]
    fn base_encoding_table_selects_correct_table() {
        assert_eq!(
            base_encoding_table(BaseEncoding::Standard)[0x27],
            Some("quoteright")
        );
        assert_eq!(
            base_encoding_table(BaseEncoding::WinAnsi)[0x27],
            Some("quotesingle")
        );
        assert_eq!(
            base_encoding_table(BaseEncoding::MacRoman)[0x27],
            Some("quotesingle")
        );
    }

    #[test]
    fn substitute_rows_match_styles() {
        for family in [
            SubstituteFamily::Sans,
            SubstituteFamily::Serif,
            SubstituteFamily::Mono,
        ] {
            for bold in [false, true] {
                for italic in [false, true] {
                    let style = SubstituteStyle {
                        family,
                        bold,
                        italic,
                    };
                    let stem = SUBSTITUTES[substitute_index(style)][0];
                    let family_name = match family {
                        SubstituteFamily::Sans => "Sans-",
                        SubstituteFamily::Serif => "Serif-",
                        SubstituteFamily::Mono => "Mono-",
                    };
                    assert!(stem.contains(family_name), "{style:?} -> {stem}");
                    assert_eq!(stem.contains("Bold"), bold, "{style:?} -> {stem}");
                    assert_eq!(stem.contains("Italic"), italic, "{style:?} -> {stem}");
                }
            }
        }
    }

    /// A serif request resolves to a different installed face than the
    /// sans-serif default whenever a serif substitute is installed.
    #[test]
    fn serif_substitute_is_not_the_sans_face() {
        let serif = SubstituteStyle {
            family: SubstituteFamily::Serif,
            ..SubstituteStyle::default()
        };
        let installed = substitute_candidates(SUBSTITUTES[substitute_index(serif)])
            .iter()
            .any(|p| PathBuf::from(p).exists());
        if !installed {
            eprintln!("no serif substitute installed; skipping");
            return;
        }
        assert_ne!(
            substitute_font_path(serif),
            substitute_font_path(SubstituteStyle::default())
        );
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
        if substitute_font_path(SubstituteStyle::default()).is_none() {
            eprintln!("no substitute font installed; skipping");
            return;
        }
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

    /// [`adobe_glyph_to_unicode`] binary-searches [`AGL`], so the table must
    /// be strictly ascending in byte order; every entry then resolves to its
    /// own code point.
    #[test]
    fn agl_is_sorted_and_every_entry_resolves() {
        for pair in AGL.windows(2) {
            assert!(
                pair[0].0.as_bytes() < pair[1].0.as_bytes(),
                "AGL out of order: {:?} before {:?}",
                pair[0].0,
                pair[1].0
            );
        }
        for &(name, unicode) in AGL {
            assert_eq!(adobe_glyph_to_unicode(name), Some(unicode), "{name}");
        }
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
}
