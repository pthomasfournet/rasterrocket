//! Cache key types for glyph lookup.

use std::sync::atomic::{AtomicU32, Ordering};

/// Opaque identifier for a loaded font face.
///
/// Drawn from a [`FaceIds`] source once per face (font program plus
/// matrix); no source yields the same identifier twice.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FaceId(pub(crate) u32);

/// A source of [`FaceId`]s.  Faces whose identifiers come from one source
/// are distinct keys in a [`crate::cache::GlyphCache`] they share.
#[derive(Debug, Default)]
pub struct FaceIds(AtomicU32);

impl FaceIds {
    /// A fresh identifier.
    pub fn next(&self) -> FaceId {
        let id = self.0.fetch_add(1, Ordering::Relaxed);
        debug_assert!(id < u32::MAX, "FaceId counter wrapped");
        FaceId(id)
    }
}

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
    /// Horizontal scale of the outline in units of `1 / u16::MAX`;
    /// `u16::MAX` is unscaled.  A substitute glyph condensed to the
    /// advance the document specifies is keyed by its scale.
    pub x_scale: u16,
    /// Whether anti-aliasing was enabled when the glyph was rasterized.
    pub aa: bool,
}

impl GlyphKey {
    /// Construct an unscaled `GlyphKey`, masking `x_frac` to the valid
    /// range `[0, FRACTION)`.
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
            x_scale: u16::MAX,
            aa,
        }
    }

    /// This key with the horizontal scale `x_scale`.
    #[must_use]
    pub const fn condensed(self, x_scale: u16) -> Self {
        Self { x_scale, ..self }
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
    fn x_scale_distinguishes() {
        let a = GlyphKey::new(FaceId(1), 42, 12, 0, true);
        let b = a.condensed(u16::MAX / 2);
        assert_ne!(a, b);
        assert_eq!(a, a.condensed(u16::MAX));
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
