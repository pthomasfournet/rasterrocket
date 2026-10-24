//! Numeric conversions between pixel indices, counts and `f64`.

/// `usize` → `f64`.  Every value passed is a pixel coordinate or count
/// bounded by the image area, below 2⁵³ and therefore exact in `f64`.
#[expect(
    clippy::cast_precision_loss,
    reason = "pixel coordinates and counts are below 2^53, exact in f64"
)]
pub(crate) const fn to_f64(v: usize) -> f64 {
    v as f64
}

/// `u64` → `f64` for histogram masses, bounded by the image area (exact).
#[expect(
    clippy::cast_precision_loss,
    reason = "histogram masses are bounded by the image area (< 2^53), exact in f64"
)]
pub(crate) const fn to_f64_u64(v: u64) -> f64 {
    v as f64
}

/// `⌊v⌋` as an index; negative and non-finite values map to 0.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is clamped to [0, 2^31] before the cast"
)]
pub(crate) const fn floor_index(v: f64) -> usize {
    if v.is_finite() {
        v.floor().clamp(0.0, 2_147_483_648.0) as usize
    } else {
        0
    }
}

/// `round(v)` as an index; negative and non-finite values map to 0.
pub(crate) fn round_index(v: f64) -> usize {
    floor_index(v + 0.5)
}

/// Round and saturate to `u8`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is clamped to [0, 255] before the cast"
)]
pub(crate) fn clamp_u8(v: f64) -> u8 {
    (v + 0.5).clamp(0.0, 255.0) as u8
}
