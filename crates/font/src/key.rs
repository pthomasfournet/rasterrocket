//! Cache key types for glyph lookup.

/// Opaque identifier for a loaded font face.
///
/// Assigned by a [`crate::engine::FontEngine`] once per face (font program
/// plus matrix) and never reused by that engine.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FaceId(pub(crate) u32);

/// Number of sub-pixel x-fraction bits.
///
/// Only 4 distinct x-fraction positions are cached: 0, 1, 2, 3 (quarters
/// of a pixel).
pub const FRACTION_BITS: u32 = 2;

/// `1 << FRACTION_BITS` — the denominator for fractional coordinates.
pub const FRACTION: u32 = 1 << FRACTION_BITS;

/// Full cache key for a single rendered glyph.
///
/// Glyphs are rendered at integral vertical positions, so the key carries
/// no y-fraction.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct GlyphKey {
    /// Which face the glyph belongs to.
    pub face_id: FaceId,
    /// Glyph index within the face's font program.
    pub glyph_id: u32,
    /// Font size in pixels (non-zero; `round(hypot(c, d))` of the face matrix).
    pub size_px: u16,
    /// Sub-pixel x-fraction in `[0, FRACTION)`.
    pub x_frac: u8,
    /// Whether anti-aliasing was enabled when the glyph was rasterized.
    pub aa: bool,
}

impl GlyphKey {
    /// Construct a `GlyphKey`, masking `x_frac` to the valid range `[0, FRACTION)`.
    #[must_use]
    pub const fn new(face_id: FaceId, glyph_id: u32, size_px: u16, x_frac: u8, aa: bool) -> Self {
        // FRACTION = 4 = 0b100; FRACTION as u8 - 1 = 3 = 0b11.
        // FRACTION fits in u8 (it is 4), so the cast is safe.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "FRACTION = 4; FRACTION as u8 = 4 fits in u8 without truncation"
        )]
        let mask = FRACTION as u8 - 1;
        Self {
            face_id,
            glyph_id,
            size_px,
            x_frac: x_frac & mask,
            aa,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glyph_key_equality() {
        let a = GlyphKey::new(FaceId(1), 42, 12, 0, true);
        let b = GlyphKey::new(FaceId(1), 42, 12, 0, true);
        assert_eq!(a, b);
    }

    #[test]
    fn glyph_key_aa_distinguishes() {
        let a = GlyphKey::new(FaceId(1), 42, 12, 0, true);
        let b = GlyphKey::new(FaceId(1), 42, 12, 0, false);
        assert_ne!(a, b);
    }

    #[test]
    fn x_frac_clamped_to_fraction_bits() {
        // x_frac 7 (0b111) is masked to 3 (0b011) by the 2-bit mask.
        let k = GlyphKey::new(FaceId(0), 0, 10, 7, true);
        assert_eq!(k.x_frac, 3);
    }

    #[test]
    fn fraction_constant_matches_bits() {
        assert_eq!(FRACTION, 4, "2 fraction bits → 4 positions");
    }
}
