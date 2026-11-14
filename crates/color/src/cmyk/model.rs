//! Arithmetic shared by the solvers over the process-ink model of
//! [`CORNERS`](super::CORNERS): the weight of a multilinear corner and ink
//! amounts as bytes.

/// The multilinear weight at fractional position `t` of the corner whose
/// offset on axis `a` is `offsets[a]` (0 or 1): the product over the axes
/// of `t` at offset 1 and `1 − t` at offset 0, in axis order.
pub(super) fn corner_weight<const N: usize>(offsets: [usize; N], t: [f64; N]) -> f64 {
    offsets
        .iter()
        .zip(t)
        .map(|(&bit, t)| if bit == 1 { t } else { 1.0 - t })
        .product()
}

/// `[0, 1]` ink amounts as bytes.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is clamped to [0, 255] and rounded first"
)]
pub(super) fn ink_bytes<const N: usize>(inks: [f64; N]) -> [u8; N] {
    inks.map(|v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
}
