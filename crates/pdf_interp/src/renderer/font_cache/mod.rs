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

use super::fallback::{FallbackGlyph, FallbackResolver};
use super::substitutes::substitute_program;
use super::system_fonts::FontLibrary;
use crate::resources::font::{FontDescriptor, fallback_font};
use crate::resources::{PageResources, ResourceKey};

mod code_to_gid;
#[cfg(test)]
mod test_fonts;

use code_to_gid::resolved_code_to_gid;

// ── Cache ─────────────────────────────────────────────────────────────────────

/// Key type for [`FontCache::faces`]: the font dictionary's identity paired
/// with the 2×2 Trm matrix bit pattern.
type FaceKey = (ResourceKey, [u64; 4]);

/// The font selected by `Tf`: the identity of its dictionary and the
/// descriptor resolved from it.
#[derive(Clone, Debug)]
pub(crate) struct ActiveFont {
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
pub(crate) struct PageFace {
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
    pub(crate) fn fallback_glyph(&self, code: u32, cid: u32) -> Option<(&FontFace, u32)> {
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

    /// The entry of the font `key` selects: that of its dictionary when it
    /// is indirect; `None` for a direct dictionary, which has no
    /// document-wide identity.
    fn entry(&self, key: &ResourceKey) -> Option<Arc<SharedFont>> {
        match key {
            ResourceKey::Object(id) => Some(self.font(*id)),
            ResourceKey::Direct { .. } => None,
        }
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
pub(crate) struct FontCache {
    engine: FontEngine,
    /// Document-wide store for indirect font dictionaries and the glyph
    /// bitmaps of the document.
    shared: Arc<DocumentFonts>,
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
    pub(crate) fn new(shared: Arc<DocumentFonts>, antialias: bool) -> Self {
        Self {
            engine: FontEngine::with_face_ids(antialias, Arc::clone(&shared.face_ids)),
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
    pub(crate) fn select(&mut self, resources: &PageResources<'_>, name: &[u8]) -> ActiveFont {
        let (key, entry) = resources.font_entry(name).unwrap_or_else(|| {
            let key = ResourceKey::Direct {
                context: resources.resource_context_id(),
                name: name.into(),
            };
            (key, &pdf::Object::Null)
        });
        let shared = &self.shared;
        let descriptor = self
            .descriptors
            .entry(key.clone())
            .or_insert_with(|| {
                let resolve = || {
                    Arc::new(
                        resources
                            .resolve_font_entry(entry)
                            .unwrap_or_else(|| fallback_font(resources.doc())),
                    )
                };
                shared.entry(&key).map_or_else(resolve, |font| {
                    Arc::clone(font.descriptor.get_or_init(resolve))
                })
            })
            .clone();
        ActiveFont { key, descriptor }
    }

    /// The document store this cache resolves indirect fonts through.
    #[must_use]
    pub(crate) const fn document_fonts(&self) -> &Arc<DocumentFonts> {
        &self.shared
    }

    /// The face identifier under which the coverage of the Type 3 font
    /// dictionary `id`, drawn with the glyph-to-device matrix `matrix`
    /// (2×2 part), is cached; one per document, font and matrix.
    #[must_use]
    pub(crate) fn type3_face_id(&self, id: ObjectId, matrix: [f64; 4]) -> FaceId {
        self.shared
            .font(id)
            .face_id(matrix.map(f64::to_bits), 0, &self.shared.face_ids)
    }

    /// Return the glyph bitmap cache shared by every face of the document.
    #[must_use]
    pub(crate) fn glyph_cache(&self) -> &GlyphCache {
        &self.shared.glyphs
    }

    /// Return a reference to the [`PageFace`] for `font`, loading it on first
    /// use.  `name` is the resource name `font` was selected by, for
    /// diagnostics.
    ///
    /// `trm` is the 2×2 submatrix of the **text rendering matrix** in device
    /// pixels: `font_size × Tm[2×2] × CTM[2×2]`.  It determines both the
    /// rendered pixel size and any shear/rotation.
    ///
    /// `None` (logged once per font and matrix) when the face cannot be
    /// loaded: the matrix is degenerate, or no program, embedded or
    /// substitute, is usable.
    pub(crate) fn get_or_load(
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
    pub(crate) fn get_or_load_showing(
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
            let shared = self.shared.entry(&font.key);
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
        let shared = self.shared.entry(key);

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
        let simple = desc.cid_encoding.is_none();
        let mut visit = |code: u32, cid: u32| {
            if face.fallback_glyphs.contains_key(&(code, cid)) {
                return;
            }
            let glyph = (!simple || face.face.glyph_index(code) == 0)
                .then(|| resolver.glyph(code, cid))
                .flatten()
                .filter(|&(slot, _)| self.ensure_face(face, &resolver, slot));
            let _ = face.fallback_glyphs.insert((code, cid), glyph);
        };
        match &desc.cid_encoding {
            None => bytes
                .iter()
                .for_each(|&b| visit(u32::from(b), u32::from(b))),
            Some(enc) => enc.codes(bytes).for_each(|d| visit(d.code, d.value)),
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::test_fonts::{ahem, cache, composite_substitute, diffs, face_showing, simple};
    use super::*;
    use crate::resources::cmap::parse_cmap;
    use crate::resources::font::{BaseEncoding, PdfFontKind};
    use crate::resources::font_substitute::SubstituteStyle;

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
        let desc = simple(
            PdfFontKind::TrueType,
            None,
            diffs(&[(65, "uni0915")]),
            BaseEncoding::WinAnsi,
        );
        let mut cache = cache();
        let face = face_showing(&mut cache, desc, b"AB");
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
        if !devanagari_installed() {
            return;
        }
        let to_unicode = parse_cmap(
            b"1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n\
              2 beginbfchar\n<0003> <0915>\n<0004> <0048>\nendbfchar\n",
        )
        .expect("parses");
        let desc = composite_substitute(Some(to_unicode), SubstituteStyle::default());
        let mut cache = cache();
        let face = face_showing(&mut cache, desc, b"\x00\x03\x00\x04");
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
        let to_unicode = parse_cmap(
            b"1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n\
              1 beginbfrange\n<0000> <FFFF> <0000>\nendbfrange\n",
        )
        .expect("parses");
        let desc = composite_substitute(Some(to_unicode), SubstituteStyle::default());
        let mut cache = cache();
        let face = face_showing(&mut cache, desc, b"\x00\x48\x09\x15");
        assert_eq!(face.fallback_glyphs.len(), 2);
        assert!(
            face.fallback_glyph(0x48, 0x48).is_none(),
            "H is the substitute's"
        );
    }

    /// An embedded program draws its own glyphs only: it has no resolver.
    #[test]
    fn embedded_font_has_no_fallbacks() {
        let desc = simple(
            PdfFontKind::TrueType,
            Some(ahem()),
            diffs(&[(65, "uni0915")]),
            BaseEncoding::WinAnsi,
        );
        let mut cache = cache();
        let face = face_showing(&mut cache, desc, b"A");
        assert!(face.fallback.is_none());
        assert!(face.fallback_glyph(65, 65).is_none());
    }
}
