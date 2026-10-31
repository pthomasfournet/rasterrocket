//! Per-page font cache.
//!
//! [`FontCache`] maps a PDF font resource name (e.g. `b"F1"`) to a loaded
//! [`FontFace`].  Faces are loaded on first use during rendering, then reused
//! for all subsequent glyphs in the same page.
//!
//! When a font has no embedded program, or its embedded program cannot be
//! parsed, the cache draws it with a substitute of the same family class
//! and variant (sans-serif, serif or monospaced; regular, bold, italic or
//! bold italic), located by [`super::substitutes`].
//!
//! # Threading
//!
//! One instance is created per page renderer (one per thread when pages are
//! rendered in parallel) and owns that page's [`FontEngine`].  What depends
//! only on an indirect font dictionary — its descriptor (which holds the
//! parsed program), its code → glyph table, its fallback resolver and the
//! identifier of its face at each matrix — is
//! kept in a [`DocumentFonts`] shared by every page of the document, with the
//! glyph bitmaps of those faces.

use std::collections::HashMap;
use std::ops::Deref;
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

use super::encodings::{base_encoding_table, glyph_name_for_code};
use super::fallback::{FallbackGlyph, FallbackResolver, first_scalar};
use super::substitutes::substitute_program;
use super::system_fonts::FontLibrary;
use crate::resources::cmap::CMap;
use crate::resources::cmap_predefined::{CollectionUnicode, cid_to_unicode};
use crate::resources::font::{
    BaseEncoding, CidEncoding, FontDescriptor, PdfFontKind, fallback_font,
};
use crate::resources::{PageResources, ResourceKey};

// ── Cache ─────────────────────────────────────────────────────────────────────

/// Key type for [`FontCache::faces`]: the font dictionary's identity paired
/// with the 2×2 Trm matrix bit pattern.
type FaceKey = (ResourceKey, [u64; 4]);

/// The font selected by `Tf`: the identity of its dictionary and the
/// descriptor resolved from it.
#[derive(Clone, Debug)]
pub struct ActiveFont {
    /// Identity of the font dictionary.
    pub key: ResourceKey,
    /// Descriptor resolved from the dictionary, shared by every selection of
    /// the same dictionary on the page.
    pub descriptor: Arc<FontDescriptor>,
}

/// A face loaded for a page.
///
/// It holds the [`FontFace`] the font draws with, whether that face is a
/// substitute program rather than one the document embeds, and, for a
/// substitute, the installed faces that draw the characters it lacks.
pub struct PageFace {
    face: FontFace,
    /// `true` when the face draws a substitute: the font has no embedded
    /// program, or its embedded program could not be parsed.
    pub substitute: bool,
    /// Resolver of the characters a substitute lacks; `None` for an
    /// embedded program.
    fallback: Option<Arc<FallbackResolver>>,
    /// Fallback faces at this face's matrix, by resolver slot, loaded as
    /// shown codes need them.
    fallback_faces: Vec<Option<FontFace>>,
    /// The fallback glyph of each (code, CID) shown with this face.
    fallback_glyphs: HashMap<(u32, u32), Option<FallbackGlyph>>,
}

impl PageFace {
    /// The fallback face and its glyph index that draw the shown `code`
    /// (CID `cid` of a composite font; the code itself for a simple font)
    /// in place of this face, as prepared by [`FontCache::get_or_load_showing`].
    /// `None` when this face draws the code itself.
    #[must_use]
    pub fn fallback_glyph(&self, code: u32, cid: u32) -> Option<(&FontFace, u32)> {
        let (slot, gid) = (*self.fallback_glyphs.get(&(code, cid))?)?;
        Some((self.fallback_faces.get(slot)?.as_ref()?, gid))
    }
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
pub struct DocumentFonts {
    fonts: Mutex<HashMap<ObjectId, Arc<SharedFont>>>,
    /// The installed fonts substitutes and fallbacks are drawn from.
    library: Arc<FontLibrary>,
    /// Source of the identifier of every face made for the document.
    face_ids: Arc<FaceIds>,
    /// Glyph bitmaps of every face made for the document.
    glyphs: GlyphCache,
}

impl Default for DocumentFonts {
    /// A store drawing substitutes from the standard font directories.
    fn default() -> Self {
        Self::new(&[], true)
    }
}

impl DocumentFonts {
    /// A store drawing substitutes and fallbacks from the fonts in
    /// `font_dirs` (searched in order) and then, when `system_fonts` is
    /// set, the standard font directories.
    #[must_use]
    pub fn new(font_dirs: &[PathBuf], system_fonts: bool) -> Self {
        Self {
            fonts: Mutex::default(),
            library: FontLibrary::shared(font_dirs, system_fonts),
            face_ids: Arc::default(),
            glyphs: GlyphCache::default(),
        }
    }

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
    /// The `code_to_gid` table computed from the program, `None` when the
    /// descriptor's own table (or the program's charmap) applies.
    code_to_gid: OnceLock<Option<Vec<u32>>>,
    /// Fallback resolver of a font drawn by a substitute.
    fallback: OnceLock<Option<Arc<FallbackResolver>>>,
    /// Face identifier per Trm bit pattern and face slot (0 for the font's
    /// own face, `k + 1` for fallback program `k`), so every page draws the
    /// font at one matrix from the same cached bitmaps.
    face_ids: Mutex<HashMap<([u64; 4], usize), FaceId>>,
}

impl SharedFont {
    /// The identifier of face `slot` at the matrix `trm`, drawn from `ids`
    /// on first use.
    fn face_id(&self, trm: [u64; 4], slot: usize, ids: &FaceIds) -> FaceId {
        let mut face_ids = self.face_ids.lock().unwrap_or_else(PoisonError::into_inner);
        *face_ids.entry((trm, slot)).or_insert_with(|| ids.next())
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
    descriptors: HashMap<ResourceKey, Arc<FontDescriptor>>,
    /// Map from (font dictionary, four f64-as-u64 bits of the Trm 2×2 matrix)
    /// to loaded face.  The full matrix is the key because the same font can
    /// appear at different sizes and skew angles within a single page.
    faces: HashMap<FaceKey, Option<PageFace>>,
}

impl FontCache {
    /// Create a new cache resolving through the document store `shared`,
    /// rasterising glyphs with 8-bit coverage when `antialias` is set and
    /// bilevel otherwise.
    #[must_use]
    pub fn new(shared: Arc<DocumentFonts>, antialias: bool) -> Self {
        Self {
            engine: FontEngine::with_face_ids(antialias, Arc::clone(&shared.face_ids)),
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
            let key = ResourceKey::Direct {
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
                    ResourceKey::Object(id) => Arc::clone(
                        shared
                            .font(*id)
                            .descriptor
                            .get_or_init(|| Arc::new(resolve())),
                    ),
                    ResourceKey::Direct { .. } => Arc::new(resolve()),
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

    /// The face identifier under which the coverage of the Type 3 font
    /// dictionary `id`, drawn with the glyph-to-device matrix `matrix`
    /// (2×2 part), is cached; one per document, font and matrix.
    #[must_use]
    pub fn type3_face_id(&self, id: ObjectId, matrix: [f64; 4]) -> FaceId {
        self.shared
            .font(id)
            .face_id(matrix.map(f64::to_bits), 0, &self.shared.face_ids)
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
        self.get_or_load_showing(font, name, trm, &[])
    }

    /// [`Self::get_or_load`] for showing the string `bytes`: the fallback
    /// glyph of every code of `bytes` a substitute lacks is resolved, and
    /// its fallback face loaded, so that [`PageFace::fallback_glyph`]
    /// answers for each.
    pub fn get_or_load_showing(
        &mut self,
        font: &ActiveFont,
        name: &[u8],
        trm: [f64; 4],
        bytes: &[u8],
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
            let _ = self.faces.insert(key.clone(), face);
        }
        let face = self.faces.get_mut(&key)?.as_mut()?;
        if !bytes.is_empty() && face.fallback.is_some() {
            let shared = match &font.key {
                ResourceKey::Object(id) => Some(self.shared.font(*id)),
                ResourceKey::Direct { .. } => None,
            };
            let loader = FallbackLoader {
                engine: &self.engine,
                ids: &self.shared.face_ids,
                shared: shared.as_deref(),
                trm,
            };
            loader.prepare(face, &font.descriptor, bytes);
        }
        self.faces.get(&key)?.as_ref()
    }

    fn load_face(
        &self,
        key: &ResourceKey,
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
            code_to_gid: Vec::new(),
            mat: trm,
        };
        let shared = match key {
            ResourceKey::Object(id) => Some(self.shared.font(*id)),
            ResourceKey::Direct { .. } => None,
        };

        let (program, substitute) = resolve_program(desc, &self.shared.library)?;
        let mut face = match &shared {
            Some(shared) => {
                let id = shared.face_id(trm.map(f64::to_bits), 0, &self.shared.face_ids);
                self.engine
                    .load_program_face_as(id, Arc::clone(&program), params)
            }
            None => self.engine.load_program_face(Arc::clone(&program), params),
        }
        .inspect_err(|e| log::debug!("font_cache: {e}"))
        .ok()?;

        // A substitute resolves codes as the program of a non-embedded font.
        let embedded = !substitute;
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
        let resolve = || {
            (!embedded)
                .then(|| {
                    FallbackResolver::new(desc, program, Arc::clone(&self.shared.library))
                        .map(Arc::new)
                })
                .flatten()
        };
        let fallback = match &shared {
            Some(shared) => shared.fallback.get_or_init(resolve).clone(),
            None => resolve(),
        };
        Some(PageFace {
            face,
            substitute,
            fallback,
            fallback_faces: Vec::new(),
            fallback_glyphs: HashMap::new(),
        })
    }
}

/// What loading the fallback faces of a [`PageFace`] needs from its
/// [`FontCache`].
struct FallbackLoader<'a> {
    engine: &'a FontEngine,
    ids: &'a FaceIds,
    /// Document store entry of an indirect font, for stable face IDs.
    shared: Option<&'a SharedFont>,
    trm: [f64; 4],
}

impl FallbackLoader<'_> {
    /// Resolve the fallback glyph of every code of `bytes` not yet seen by
    /// `face`, loading the fallback faces they select.  A code of a simple
    /// font is looked up only when the substitute maps it to `.notdef`; a
    /// CID of a composite font whenever its character is known, since the
    /// substitute's table keeps a glyph-order reading for a character it
    /// lacks.
    fn prepare(&self, face: &mut PageFace, desc: &FontDescriptor, bytes: &[u8]) {
        let Some(resolver) = face.fallback.clone() else {
            return;
        };
        let codes: Vec<(u32, u32)> = desc.cid_encoding.as_ref().map_or_else(
            || {
                bytes
                    .iter()
                    .map(|&b| (u32::from(b), u32::from(b)))
                    .collect()
            },
            |enc| enc.codes(bytes).map(|d| (d.code, d.value)).collect(),
        );
        let simple = desc.cid_encoding.is_none();
        for (code, cid) in codes {
            if face.fallback_glyphs.contains_key(&(code, cid)) {
                continue;
            }
            let glyph = (!simple || face.face.glyph_index(code) == 0)
                .then(|| resolver.glyph(code, cid))
                .flatten()
                .filter(|&(slot, _)| self.ensure_face(face, &resolver, slot));
            let _ = face.fallback_glyphs.insert((code, cid), glyph);
        }
    }

    /// Load the face of fallback program `slot` at this matrix into
    /// `face`, once; `false` when it cannot be loaded.
    fn ensure_face(&self, face: &mut PageFace, resolver: &FallbackResolver, slot: usize) -> bool {
        if face.fallback_faces.len() <= slot {
            face.fallback_faces.resize_with(slot + 1, || None);
        }
        if face.fallback_faces[slot].is_some() {
            return true;
        }
        let Some(program) = resolver.program(slot) else {
            return false;
        };
        let params = FaceParams {
            code_to_gid: Vec::new(),
            mat: self.trm,
        };
        let loaded = match self.shared {
            Some(shared) => {
                let id = shared.face_id(self.trm.map(f64::to_bits), slot + 1, self.ids);
                self.engine.load_program_face_as(id, program, params)
            }
            None => self.engine.load_program_face(program, params),
        }
        .inspect_err(|e| log::debug!("font_cache: fallback face: {e}"))
        .ok();
        let loaded_ok = loaded.is_some();
        face.fallback_faces[slot] = loaded;
        loaded_ok
    }
}

/// The program a descriptor draws with: its embedded program, else the
/// substitute of its style from `library`, for a font without a usable
/// program.  `None` when no substitute is available either.
fn resolve_program(desc: &FontDescriptor, library: &FontLibrary) -> Option<ResolvedProgram> {
    desc.program.clone().map_or_else(
        || substitute_program(desc.substitute, library).map(|program| (program, true)),
        |program| Some((program, false)),
    )
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
/// An embedded simple sfnt with no code-keyed `cmap` subtable (a legacy
/// subset) resolves through [`resolve_cmapless`].
///
/// `embedded` is whether `face` draws the descriptor's own program.
fn resolved_code_to_gid(
    face: &FontFace,
    desc: &FontDescriptor,
    embedded: bool,
) -> Option<Vec<u32>> {
    let simple = desc.cid_encoding.is_none();
    let outline = desc.type3.is_none();
    if embedded && simple && outline && face.program().lacks_cmap() {
        Some(resolve_cmapless(
            face,
            &desc.differences,
            desc.base_encoding,
        ))
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
        && let Some(enc) = &desc.cid_encoding
    {
        desc.substitute.script.and_then(cid_to_unicode).map_or_else(
            || {
                enc.to_unicode
                    .as_ref()
                    .map(|to_unicode| resolve_cids_via_to_unicode(face, enc, to_unicode))
            },
            |table| Some(resolve_cids_via_unicode(face, &table, enc.vertical)),
        )
    } else {
        None
    }
}

/// Build a 256-entry `code_to_gid` table for an embedded simple sfnt that
/// has no code-keyed `cmap` subtable.
///
/// A code whose name the font's encoding declares (a `Differences` entry,
/// else an explicit base encoding) selects the glyph of that name in the
/// program's `post` table (PDF §9.6.6.4).  Every other code is read as a
/// glyph index: without a `cmap`, the glyph order is the program's only
/// built-in encoding, and producers that drop the table address glyphs by
/// index.  A Unicode charmap derived from the `post` names is not
/// consulted for raw codes, since a code is not a Unicode value here.
fn resolve_cmapless(
    face: &FontFace,
    differences: &[Option<Box<str>>; 256],
    base: BaseEncoding,
) -> Vec<u32> {
    (0..=u8::MAX)
        .map(|code| {
            let idx = usize::from(code);
            differences[idx]
                .as_deref()
                .or_else(|| {
                    (base != BaseEncoding::FontProgram)
                        .then(|| base_encoding_table(base)[idx])
                        .flatten()
                })
                .map(|name| face.raw_get_name_index(name))
                .filter(|&gid| gid != 0)
                .unwrap_or_else(|| u32::from(code))
        })
        .collect()
}

/// Build a 65 536-entry CID → glyph table for a composite font of no
/// known character collection drawn by a substitute, through the font's
/// `ToUnicode` map: each code it maps is read as its CID through the
/// `Encoding` `CMap` and its glyph found by Unicode value in the
/// substitute's charmap (vertical form in vertical writing mode).  A CID
/// the map leaves out, or whose glyph the substitute lacks, keeps the
/// identity reading, under which the substitutes of the metric-compatible
/// families draw by glyph order.
fn resolve_cids_via_to_unicode(face: &FontFace, enc: &CidEncoding, to_unicode: &CMap) -> Vec<u32> {
    let mut table: Vec<u32> = (0..1u32 << 16).collect();
    for (&code, &value) in &to_unicode.map {
        let Some(unicode) = first_scalar(value) else {
            continue;
        };
        let gid = face.raw_get_char_index(unicode);
        if gid == 0 {
            continue;
        }
        if let Some(slot) = table.get_mut(enc.encoding_cmap.lookup(code) as usize) {
            *slot = if enc.vertical {
                face.program().vertical_gid(gid)
            } else {
                gid
            };
        }
    }
    table
}

/// Build a 65 536-entry CID → glyph table for a composite font drawn by a
/// substitute, through the CID → Unicode table of its character
/// collection and the substitute's Unicode charmap: the glyph of the
/// first of a CID's code points the substitute maps.  In `vertical`
/// writing mode, and for a CID that is the collection's vertical form of
/// its code point in either mode, each glyph is replaced by its vertical
/// form.  The table spans the whole CID space so that no CID falls
/// through to the charmap.
fn resolve_cids_via_unicode(
    face: &FontFace,
    collection: &CollectionUnicode,
    vertical: bool,
) -> Vec<u32> {
    let mut table = vec![0u32; 1 << 16];
    for (cid, codes) in collection.iter() {
        if let Some(slot) = table.get_mut(cid as usize) {
            let gid = codes
                .iter()
                .map(|&unicode| face.raw_get_char_index(unicode))
                .find(|&gid| gid != 0)
                .unwrap_or(0);
            *slot = if vertical || collection.is_vertical(cid) {
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
/// A substitute that lacks the glyph of a name it does understand draws
/// nothing for the code: the glyph at the raw code point would be an
/// unrelated character (`uni4E2D` at code 65 is not `A`).
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
            });
            let unicode = name.and_then(font::glyph_name_to_unicode);
            name.and_then(|name| {
                (!embedded)
                    .then(|| face.raw_get_name_index(name))
                    .and_then(nonzero)
                    .or_else(|| {
                        unicode
                            .map(|u| face.raw_get_char_index(u))
                            .and_then(nonzero)
                    })
            })
            .or_else(|| {
                (embedded || unicode.is_none())
                    .then(|| face.raw_get_char_index(u32::from(code)))
                    .and_then(nonzero)
            })
            .or_else(|| embedded.then(|| face.raw_get_code_index(code)))
            .unwrap_or(0)
        })
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::font::CidMetrics;
    use crate::resources::font_substitute::SubstituteStyle;

    /// `desc` selected as the indirect font dictionary `1 0 R`.
    fn active(desc: FontDescriptor) -> ActiveFont {
        ActiveFont {
            key: ResourceKey::Object((1, 0)),
            descriptor: Arc::new(desc),
        }
    }

    fn empty_diffs() -> [Option<Box<str>>; 256] {
        std::array::from_fn(|_| None)
    }

    /// The program `bytes`, parsed.
    fn parsed(bytes: Vec<u8>) -> Arc<FontProgram> {
        Arc::new(FontProgram::parse(bytes, 0).expect("test program parses"))
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
            program: None,
            widths: vec![],
            first_char: 0,
            missing_width: 0,
            differences,
            base_encoding: BaseEncoding::MacRoman,
            cid_encoding: None,
            type3: None,
            substitute: SubstituteStyle::default(),
        };
        let mut cache = FontCache::new(Arc::default(), true);
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

    /// A `Differences` name the substitute understands but has no glyph for
    /// draws nothing: `uni4E2D` at code 65 must not select the Latin `A`
    /// the substitute keeps at that code point.
    #[test]
    fn missing_glyph_of_a_known_name_is_notdef_on_a_substitute() {
        let mut differences: Box<[Option<Box<str>>; 256]> = Box::new(std::array::from_fn(|_| None));
        differences[65] = Some("uni4E2D".into());
        differences[66] = Some("afii10017".into());
        let desc = FontDescriptor {
            kind: PdfFontKind::TrueType,
            program: None,
            widths: vec![],
            first_char: 0,
            missing_width: 0,
            differences,
            base_encoding: BaseEncoding::WinAnsi,
            cid_encoding: None,
            type3: None,
            substitute: SubstituteStyle::default(),
        };
        let mut cache = FontCache::new(Arc::default(), true);
        let face = cache
            .get_or_load(&active(desc), b"F1", [20.0, 0.0, 0.0, 20.0])
            .expect("substitute face loads");
        assert_eq!(
            face.raw_get_char_index(0x4E2D),
            0,
            "the Latin substitute has no CJK"
        );
        assert_eq!(face.code_to_gid.get(65), Some(&0));
        let cyrillic_a = face.raw_get_char_index(0x0410);
        assert_ne!(cyrillic_a, 0);
        assert_eq!(face.code_to_gid.get(66), Some(&cyrillic_a));
        assert_eq!(
            face.code_to_gid.get(67),
            Some(&face.raw_get_char_index(u32::from('C')))
        );
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
            program: Some(parsed(
                include_bytes!("../../../font_test_data/test_data/ttf/ahem.ttf").to_vec(),
            )),
            widths: vec![],
            first_char: 0,
            missing_width: 0,
            differences,
            base_encoding: BaseEncoding::FontProgram,
            cid_encoding: None,
            type3: None,
            substitute: SubstituteStyle::default(),
        };
        let mut cache = FontCache::new(Arc::default(), true);
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

    /// `ahem.ttf` with its `cmap` table hidden: the table-directory tag is
    /// renamed, so the program keeps its `post` names and no code-keyed
    /// subtable.
    fn ahem_without_cmap() -> Vec<u8> {
        let mut data = include_bytes!("../../../font_test_data/test_data/ttf/ahem.ttf").to_vec();
        let tables = usize::from(u16::from_be_bytes([data[4], data[5]]));
        let at = (0..tables)
            .map(|i| 12 + 16 * i)
            .find(|&at| &data[at..at + 4] == b"cmap")
            .expect("ahem has a cmap table");
        data[at..at + 4].copy_from_slice(b"xmap");
        data
    }

    /// An embedded simple TrueType program `bytes` under `differences` and
    /// `base_encoding`.
    fn embedded_truetype(
        bytes: Vec<u8>,
        differences: Box<[Option<Box<str>>; 256]>,
        base_encoding: BaseEncoding,
    ) -> FontDescriptor {
        FontDescriptor {
            kind: PdfFontKind::TrueType,
            program: Some(parsed(bytes)),
            widths: vec![],
            first_char: 0,
            missing_width: 0,
            differences,
            base_encoding,
            cid_encoding: None,
            type3: None,
            substitute: SubstituteStyle::default(),
        }
    }

    /// An embedded TrueType subset with `post` names but no `cmap` and no
    /// `Encoding` addresses its glyphs by index: code 7 draws glyph 7, not
    /// the `.notdef` a Unicode lookup of U+0007 through the names gives.
    #[test]
    fn cmapless_truetype_without_encoding_reads_codes_as_glyph_indices() {
        let desc = embedded_truetype(
            ahem_without_cmap(),
            Box::new(empty_diffs()),
            BaseEncoding::FontProgram,
        );
        let mut cache = FontCache::new(Arc::default(), true);
        let face = cache
            .get_or_load(&active(desc), b"F1", [20.0, 0.0, 0.0, 20.0])
            .expect("embedded TrueType face loads");
        assert!(face.program().lacks_cmap());
        assert_eq!(face.raw_get_name_index("dollar"), 7);
        for code in [2usize, 3, 7, 0x41, 0xFF] {
            assert_eq!(face.code_to_gid[code] as usize, code, "code {code:#04x}");
        }
    }

    /// A cmap-less subset under a declared encoding resolves each named
    /// code through the `post` table: `Differences` first, then the base
    /// encoding; a code whose name the program lacks keeps its glyph index.
    #[test]
    fn cmapless_truetype_resolves_declared_names_through_post() {
        let mut differences: Box<[Option<Box<str>>; 256]> = Box::new(empty_diffs());
        differences[0x41] = Some("dollar".into());
        differences[0x42] = Some("nosuchglyph".into());
        let desc = embedded_truetype(ahem_without_cmap(), differences, BaseEncoding::WinAnsi);
        let mut cache = FontCache::new(Arc::default(), true);
        let face = cache
            .get_or_load(&active(desc), b"F1", [20.0, 0.0, 0.0, 20.0])
            .expect("embedded TrueType face loads");
        let exclam = face.raw_get_name_index("exclam");
        assert_ne!(exclam, 0x21, "ahem's glyph order is not its code order");
        assert_eq!(face.code_to_gid[0x41], 7, "Differences name");
        assert_eq!(face.code_to_gid[0x21], exclam, "WinAnsi name");
        assert_eq!(face.code_to_gid[0x42], 0x42, "unknown name keeps the index");
        assert_eq!(face.code_to_gid[0x05], 0x05, "unnamed code keeps the index");
    }

    /// Whether an installed face maps U+0915 (Devanagari KA), the probe of
    /// the fallback tests; they are skipped on a host without one.
    fn devanagari_installed() -> bool {
        let found = FontLibrary::shared(&[], true)
            .fallback_program(SubstituteStyle::default(), 0x0915)
            .is_some();
        if !found {
            eprintln!("no installed face maps U+0915; skipping");
        }
        found
    }

    /// A non-embedded simple font naming a Devanagari glyph draws it from
    /// an installed face that maps it, at the code's position in the
    /// shown string; codes the substitute draws stay with it.
    #[test]
    fn substitute_lacking_a_glyph_draws_it_from_a_fallback_face() {
        if !devanagari_installed() {
            return;
        }
        let mut differences: Box<[Option<Box<str>>; 256]> = Box::new(empty_diffs());
        differences[65] = Some("uni0915".into());
        let desc = FontDescriptor {
            kind: PdfFontKind::TrueType,
            program: None,
            widths: vec![],
            first_char: 0,
            missing_width: 0,
            differences,
            base_encoding: BaseEncoding::WinAnsi,
            cid_encoding: None,
            type3: None,
            substitute: SubstituteStyle::default(),
        };
        let mut cache = FontCache::new(Arc::default(), true);
        let face = cache
            .get_or_load_showing(&active(desc), b"F1", [20.0, 0.0, 0.0, 20.0], b"AB")
            .expect("substitute face loads");
        assert_eq!(face.code_to_gid[65], 0, "the substitute lacks U+0915");
        let (fallback, gid) = face
            .fallback_glyph(65, 65)
            .expect("a fallback draws U+0915");
        assert!(!std::ptr::eq(fallback.program(), face.program()));
        assert_eq!(gid, fallback.program().char_index(0x0915));
        assert!(
            fallback
                .make_glyph_by_gid(gid, 0)
                .is_some_and(|g| g.data.iter().any(|&c| c != 0))
        );
        assert!(
            face.fallback_glyph(66, 66).is_none(),
            "B stays with the substitute"
        );
        assert!(face.fallback_glyph(67, 67).is_none(), "C was not shown");
    }

    /// A composite font without a program, through its `ToUnicode` map,
    /// draws a character its substitute lacks from a fallback face, in
    /// place of the glyph-order reading its table keeps for that CID.
    #[test]
    fn non_embedded_cid_font_draws_missing_characters_from_a_fallback() {
        use crate::resources::cmap::parse_cmap;
        if !devanagari_installed() {
            return;
        }
        let to_unicode = parse_cmap(
            b"1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n\
              2 beginbfchar\n<0003> <0915>\n<0004> <0048>\nendbfchar\n",
        )
        .expect("parses");
        let mut cache = FontCache::new(Arc::default(), true);
        let face = cache
            .get_or_load_showing(
                &active(cid_substitute(to_unicode)),
                b"F1",
                [20.0, 0.0, 0.0, 20.0],
                b"\x00\x03\x00\x04",
            )
            .expect("substitute face loads");
        assert_eq!(face.code_to_gid[3], 3, "glyph-order reading in the table");
        assert!(
            face.fallback_glyph(3, 3).is_some(),
            "a fallback draws U+0915"
        );
        assert!(
            face.fallback_glyph(4, 4).is_none(),
            "H stays with the substitute"
        );
    }

    /// A `ToUnicode` map spanning the whole code space resolves only the
    /// codes shown: one shown code costs one lookup, however many the map
    /// covers.
    #[test]
    fn full_range_to_unicode_resolves_only_shown_codes() {
        use crate::resources::cmap::parse_cmap;
        let to_unicode = parse_cmap(
            b"1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n\
              1 beginbfrange\n<0000> <FFFF> <0000>\nendbfrange\n",
        )
        .expect("parses");
        let mut cache = FontCache::new(Arc::default(), true);
        let face = cache
            .get_or_load_showing(
                &active(cid_substitute(to_unicode)),
                b"F1",
                [20.0, 0.0, 0.0, 20.0],
                b"\x00\x48\x09\x15",
            )
            .expect("substitute face loads");
        assert_eq!(face.fallback_glyphs.len(), 2);
        assert!(
            face.fallback_glyph(0x48, 0x48).is_none(),
            "H is the substitute's"
        );
    }

    /// An embedded program draws its own glyphs only: it has no resolver.
    #[test]
    fn embedded_font_has_no_fallbacks() {
        let mut differences: Box<[Option<Box<str>>; 256]> = Box::new(empty_diffs());
        differences[65] = Some("uni0915".into());
        let desc = embedded_truetype(
            include_bytes!("../../../font_test_data/test_data/ttf/ahem.ttf").to_vec(),
            differences,
            BaseEncoding::WinAnsi,
        );
        let mut cache = FontCache::new(Arc::default(), true);
        let face = cache
            .get_or_load_showing(&active(desc), b"F1", [20.0, 0.0, 0.0, 20.0], b"A")
            .expect("embedded face loads");
        assert!(face.fallback.is_none());
        assert!(face.fallback_glyph(65, 65).is_none());
    }

    /// A non-embedded Identity-H font of no known collection with the
    /// `ToUnicode` map `to_unicode`.
    fn cid_substitute(to_unicode: CMap) -> FontDescriptor {
        FontDescriptor {
            kind: PdfFontKind::TrueType,
            program: None,
            widths: vec![],
            first_char: 0,
            missing_width: 0,
            differences: Box::new(empty_diffs()),
            base_encoding: BaseEncoding::Standard,
            cid_encoding: Some(CidEncoding {
                encoding_cmap: Arc::new(CMap::identity()),
                cid_to_gid: None,
                default_width: 1000,
                widths: CidMetrics::new(Vec::new()),
                vertical: false,
                default_vertical: [880, -1000],
                vertical_metrics: CidMetrics::new(Vec::new()),
                to_unicode: Some(Arc::new(to_unicode)),
            }),
            type3: None,
            substitute: SubstituteStyle::default(),
        }
    }

    /// A composite font without a program and of no known collection draws
    /// each code its `ToUnicode` map covers by Unicode value: Identity-H
    /// code 3 → `H` selects the substitute's `H`, while an unmapped code
    /// keeps the identity reading.
    #[test]
    fn non_embedded_cid_font_resolves_codes_through_to_unicode() {
        use crate::resources::cmap::parse_cmap;
        let to_unicode = parse_cmap(
            b"1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n\
              1 beginbfchar\n<0003> <0048>\nendbfchar\n",
        )
        .expect("parses");
        let desc = FontDescriptor {
            kind: PdfFontKind::TrueType,
            program: None,
            widths: vec![],
            first_char: 0,
            missing_width: 0,
            differences: Box::new(empty_diffs()),
            base_encoding: BaseEncoding::Standard,
            cid_encoding: Some(CidEncoding {
                encoding_cmap: Arc::new(CMap::identity()),
                cid_to_gid: None,
                default_width: 1000,
                widths: CidMetrics::new(Vec::new()),
                vertical: false,
                default_vertical: [880, -1000],
                vertical_metrics: CidMetrics::new(Vec::new()),
                to_unicode: Some(Arc::new(to_unicode)),
            }),
            type3: None,
            substitute: SubstituteStyle::default(),
        };
        let mut cache = FontCache::new(Arc::default(), true);
        let face = cache
            .get_or_load(&active(desc), b"F1", [20.0, 0.0, 0.0, 20.0])
            .expect("substitute face loads");
        let h = face.raw_get_char_index(u32::from('H'));
        assert_ne!(h, 0);
        assert_eq!(face.code_to_gid[3], h);
        assert_eq!(
            face.code_to_gid[43], 43,
            "unmapped CIDs keep the identity reading"
        );
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
            program: None,
            widths: vec![],
            first_char: 0,
            missing_width: 0,
            differences: Box::new(empty_diffs()),
            base_encoding: BaseEncoding::Standard,
            cid_encoding: Some(CidEncoding {
                encoding_cmap: Arc::new(CMap::identity()),
                cid_to_gid: None,
                default_width: 1000,
                widths: CidMetrics::new(Vec::new()),
                vertical: false,
                default_vertical: [880, -1000],
                vertical_metrics: CidMetrics::new(Vec::new()),
                to_unicode: None,
            }),
            type3: None,
            substitute: SubstituteStyle {
                script: Some(Script::Japanese),
                ..SubstituteStyle::default()
            },
        };
        let mut cache = FontCache::new(Arc::default(), true);
        let face = cache
            .get_or_load(&active(desc), b"F1", [20.0, 0.0, 0.0, 20.0])
            .expect("substitute face loads");
        assert_eq!(face.code_to_gid.len(), 1 << 16);
        let a = face.raw_get_char_index(u32::from('A'));
        assert_ne!(a, 0);
        assert_eq!(face.code_to_gid[264], a);
        assert_eq!(face.code_to_gid[34], a);
        assert_eq!(face.code_to_gid[0], 0);
        // The vertical-form CIDs a `-V` `CMap` selects (long-vowel mark
        // 7891, ideographic comma 7887, wave dash 7894) draw the vertical
        // form of their character, in horizontal mode too.
        for (cid, unicode) in [(7891, 0x30FC), (7887, 0x3001), (7894, 0x301C)] {
            let gid = face.raw_get_char_index(unicode);
            let vertical = face.program().vertical_gid(gid);
            assert_ne!(gid, 0, "U+{unicode:04X}");
            assert_eq!(face.code_to_gid[cid], vertical, "CID {cid}");
        }
    }
}
