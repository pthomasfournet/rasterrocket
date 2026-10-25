//! A font program at one text rendering matrix.
//!
//! A [`FontFace`] pairs a shared [`FontProgram`] with the 2×2 matrix that
//! maps em units to device pixels (PDF text rendering matrix `Trm` without
//! its translation, y up).  It resolves character codes to glyphs, renders
//! glyph coverage bitmaps and builds glyph outline paths in that space.
//!
//! Glyph indices passed to the `*_by_gid` methods follow the composite-font
//! convention: for a CID-keyed CFF program they are CIDs and are mapped
//! through the program's charset; for every other program they are glyph
//! indices, which in a CID-keyed Type 1 program are the CIDs.

use std::sync::{Arc, OnceLock};

use raster::path::Path;
use skrifa::outline::HintingInstance;

use crate::bitmap::GlyphBitmap;
use crate::cache::GlyphCache;
use crate::coverage::rasterize;
use crate::engine::FaceParams;
use crate::key::{FRACTION, FaceId, GlyphKey};
use crate::outline::PathPen;
use crate::program::FontProgram;

/// 2×2 matrix `[a, b, c, d]` mapping em units to device pixels:
/// `x' = a·x + c·y`, `y' = b·x + d·y`.
pub type FontMatrix = [f64; 4];

/// Largest em, in device pixels, whose glyphs [`FontFace`] renders as
/// bitmaps by preference; see [`FontFace::prefers_path_fill`].
pub const MAX_BITMAP_EM_PX: f64 = 2048.0;

/// A font program bound to one em → device matrix.
pub struct FontFace {
    /// Opaque identifier for this face (for cache keying).
    pub id: FaceId,
    /// Glyph-index map: `code_to_gid[char_code]` → glyph index.  An empty
    /// `Vec` defers to the program's own charmap
    /// ([`FontProgram::char_index`]).
    pub code_to_gid: Vec<u32>,
    /// Nominal pixel size: `round(hypot(c, d))` of the matrix.
    pub size_px: u16,
    /// Whether glyph bitmaps carry 8-bit coverage (`true`) or 1-bit masks.
    pub aa: bool,
    program: Arc<FontProgram>,
    matrix: FontMatrix,
    /// Interpreter state at `size_px` pixels per em for a program that
    /// requires hinting ([`FontProgram::requires_hinting`]), built on first
    /// use; `None` inside when the program needs none or it cannot be set
    /// up.
    hinting: OnceLock<Option<HintingInstance>>,
}

impl std::fmt::Debug for FontFace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FontFace")
            .field("id", &self.id)
            .field("size_px", &self.size_px)
            .field("matrix", &self.matrix)
            .field("program", &self.program)
            .finish_non_exhaustive()
    }
}

impl FontFace {
    /// Bind `program` to the matrix in `params`.
    ///
    /// Returns `None` when the matrix is degenerate: a non-finite entry, or
    /// a nominal pixel size outside `[1, 65535]`.
    #[must_use]
    pub(crate) fn new(
        id: FaceId,
        program: Arc<FontProgram>,
        params: FaceParams,
        aa: bool,
    ) -> Option<Self> {
        let FaceParams { code_to_gid, mat } = params;
        if !mat.iter().all(|v| v.is_finite()) {
            return None;
        }
        let size_f = f64::hypot(mat[2], mat[3]).round();
        if !(1.0..=f64::from(u16::MAX)).contains(&size_f) {
            return None;
        }
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "size_f is finite and within [1, 65535] by the check above"
        )]
        let size_px = size_f as u16;
        Some(Self {
            id,
            code_to_gid,
            size_px,
            aa,
            program,
            matrix: mat,
            hinting: OnceLock::new(),
        })
    }

    /// The font program this face renders.
    #[must_use]
    pub fn program(&self) -> &FontProgram {
        &self.program
    }

    /// `true` when an em spans more than [`MAX_BITMAP_EM_PX`] pixels along
    /// either matrix axis.  Glyphs that large are better filled as paths
    /// into the page ([`Self::glyph_path`]) than rasterised into cached
    /// bitmaps, whose size grows with the square of the em.
    #[must_use]
    pub fn prefers_path_fill(&self) -> bool {
        let [a, b, c, d] = self.matrix;
        a.hypot(b).max(c.hypot(d)) > MAX_BITMAP_EM_PX
    }

    /// Rasterise the glyph for character code `char_code`.
    ///
    /// `x_frac` is a sub-pixel x offset in `[0, FRACTION)` quarter pixels.
    /// Returns `None` for a glyph without outline (`.notdef` in an empty
    /// subset, a space) or one that fails to load.
    #[must_use]
    pub fn make_glyph(&self, char_code: u32, x_frac: u8) -> Option<GlyphBitmap> {
        self.render_gid(self.resolve_gid(char_code), x_frac)
    }

    /// Cached form of [`Self::make_glyph`].  The cache key is the resolved
    /// glyph index, so codes that select the same glyph share an entry.
    #[must_use]
    pub fn make_glyph_cached(
        &self,
        cache: &GlyphCache,
        char_code: u32,
        x_frac: u8,
    ) -> Option<Arc<GlyphBitmap>> {
        let gid = self.resolve_gid(char_code);
        let key = GlyphKey::new(self.id, gid, self.size_px, x_frac, self.aa);
        cache.get_or_render(key, || self.render_gid(gid, x_frac))
    }

    /// Rasterise the glyph a composite font selects with `gid` (see the
    /// module documentation), bypassing character-code resolution.
    #[must_use]
    pub fn make_glyph_by_gid(&self, gid: u32, x_frac: u8) -> Option<GlyphBitmap> {
        self.render_gid(self.program.cid_to_gid(gid)?, x_frac)
    }

    /// Cached form of [`Self::make_glyph_by_gid`].
    ///
    /// The key includes [`Self::id`], which is allocated per face — per
    /// font program and matrix — so glyphs at different sizes, skews or
    /// rotations never alias even though [`GlyphKey`] records only
    /// `size_px`.
    #[must_use]
    pub fn make_glyph_by_gid_cached(
        &self,
        cache: &GlyphCache,
        gid: u32,
        x_frac: u8,
    ) -> Option<Arc<GlyphBitmap>> {
        let gid = self.program.cid_to_gid(gid)?;
        let key = GlyphKey::new(self.id, gid, self.size_px, x_frac, self.aa);
        cache.get_or_render(key, || self.render_gid(gid, x_frac))
    }

    /// Outline of the glyph for `char_code` in device pixels relative to
    /// the pen position, y up.  `None` when the glyph has no outline.
    #[must_use]
    pub fn glyph_path(&self, char_code: u32) -> Option<Path> {
        self.outline(self.resolve_gid(char_code), 0.0)
    }

    /// Outline of the glyph a composite font selects with `gid`; see
    /// [`Self::glyph_path`].
    #[must_use]
    pub fn glyph_path_by_gid(&self, gid: u32) -> Option<Path> {
        self.outline(self.program.cid_to_gid(gid)?, 0.0)
    }

    /// Advance width of the glyph for `char_code` in em units (multiply by
    /// the font size for text-space units).  `None` when the program
    /// records none.
    #[must_use]
    pub fn glyph_advance(&self, char_code: u32) -> Option<f64> {
        self.program.advance(self.resolve_gid(char_code))
    }

    /// Advance width of the glyph a composite font selects with `gid`; see
    /// [`Self::glyph_advance`].
    #[must_use]
    pub fn glyph_advance_by_gid(&self, gid: u32) -> Option<f64> {
        self.program.advance(self.program.cid_to_gid(gid)?)
    }

    /// Glyph index for `code` through the program's charmap (see
    /// [`FontProgram::char_index`]); 0 when unmapped.
    #[must_use]
    pub fn raw_get_char_index(&self, code: u32) -> u32 {
        self.program.char_index(code)
    }

    /// Glyph index of the single-byte `code` through the program's
    /// byte-keyed (3,0) / (1,0) `cmap` subtables (see
    /// [`FontProgram::code_index`]); 0 when unmapped.
    #[must_use]
    pub fn raw_get_code_index(&self, code: u8) -> u32 {
        self.program.code_index(code)
    }

    /// `true` when the program is an sfnt whose charmap takes Unicode
    /// values (see [`FontProgram::has_unicode_cmap`]).
    #[must_use]
    pub fn has_unicode_cmap(&self) -> bool {
        self.program.has_unicode_cmap()
    }

    /// Glyph index of the glyph named `name` in the program, 0 when absent
    /// (see [`FontProgram::name_index`]).
    #[must_use]
    pub fn raw_get_name_index(&self, name: &str) -> u32 {
        self.program.name_index(name)
    }

    /// Glyph index of `code` through the program's built-in encoding, `None`
    /// when it assigns none (see [`FontProgram::builtin_encoding_index`]).
    #[must_use]
    pub fn raw_get_builtin_index(&self, code: u8) -> Option<u32> {
        self.program.builtin_encoding_index(code)
    }

    /// Resolve a character code: through `code_to_gid` when it covers the
    /// code, otherwise through the program's charmap.
    fn resolve_gid(&self, char_code: u32) -> u32 {
        usize::try_from(char_code)
            .ok()
            .and_then(|i| self.code_to_gid.get(i))
            .copied()
            .unwrap_or_else(|| self.program.char_index(char_code))
    }

    fn render_gid(&self, gid: u32, x_frac: u8) -> Option<GlyphBitmap> {
        let shift = f64::from(u32::from(x_frac) % FRACTION) / f64::from(FRACTION);
        rasterize(&self.outline(gid, shift)?, self.aa)
    }

    /// Outline of glyph `gid` in device pixels, translated right by `shift`.
    ///
    /// A program that requires hinting is grid-fitted at `size_px` pixels
    /// per em and the result mapped by the matrix divided by `size_px`;
    /// when that fails the glyph is drawn unhinted.
    fn outline(&self, gid: u32, shift: f64) -> Option<Path> {
        if let Some(path) = self.hinted_outline(gid, shift) {
            return Some(path);
        }
        let scale = self.program.upem().recip();
        let [a, b, c, d] = self.matrix.map(|v| v * scale);
        let mut pen = PathPen::new([a, b, c, d, shift, 0.0]);
        let _advance = self.program.draw(gid, &mut pen)?;
        let path = pen.finish();
        (!path.pts.is_empty()).then_some(path)
    }

    fn hinted_outline(&self, gid: u32, shift: f64) -> Option<Path> {
        if !self.program.requires_hinting() {
            return None;
        }
        let instance = self
            .hinting
            .get_or_init(|| self.program.hinting_instance(f32::from(self.size_px)))
            .as_ref()?;
        let scale = f64::from(self.size_px).recip();
        let [a, b, c, d] = self.matrix.map(|v| v * scale);
        let mut pen = PathPen::new([a, b, c, d, shift, 0.0]);
        self.program.draw_hinted(gid, instance, &mut pen)?;
        let path = pen.finish();
        (!path.pts.is_empty()).then_some(path)
    }
}
