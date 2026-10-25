//! `DeviceCMYK` → sRGB conversion by multilinear interpolation over process-ink corners.
//!
//! A CMYK value is a point in the unit hypercube `[0, 1]⁴`.  Each of the
//! sixteen vertices — every combination of no ink and solid ink on C, M, Y
//! and K — is assigned the sRGB colour of that overprint printed on white
//! paper ([`CORNERS`]).  An arbitrary CMYK value is the multilinear
//! (quadrilinear) interpolation of those sixteen colours.  The model
//! reproduces solids and overprints exactly; the additive complement
//! `R = 1 − (C + K)` instead treats process inks as ideal block filters and
//! renders them far lighter and more saturated than print.
//!
//! # Fixed-point evaluation
//!
//! Every entry point evaluates the same integer arithmetic, so the scalar
//! code, the SIMD tiers in `simd` and the GPU kernel produce identical
//! bytes:
//!
//! 1. An ink byte `x ∈ [0, 255]` becomes the weight `w = x + (x >> 7)`
//!    on the scale `[0, 256]`; `0 ↦ 0` and `255 ↦ 256`, so solids select a
//!    corner exactly, and `|w / 256 − x / 255| ≤ 1/512` elsewhere.
//! 2. The sixteen corners are reduced along Y, M, C and K in that order by
//!    `lerp(a, b, w) = 256·a + (b − a)·w`.  Values carry 8 fractional
//!    bits between stages; after the M and C stages the 16-bit product is
//!    rounded back to 8 fractional bits with `(v + 128) >> 8`, and the
//!    final K stage rounds to an integer with `(v + 32768) >> 16`.
//! 3. Every intermediate is a convex combination of corner bytes, so it is
//!    non-negative and at most `255 · 2¹⁶ < 2³¹`; `i32` lanes never
//!    overflow and the result never leaves `[0, 255]`.
//!
//! Normalised (`f64`) components are quantised to the same weight scale by
//! `round(256 · t)`, so a vector fill and an image sample of equal colour
//! convert through one code path.

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod simd;

#[cfg(test)]
mod tests;

/// sRGB colour of each solid / overprint corner, indexed `k·8 + c·4 + m·2 + y`
/// with every index in `{0, 1}` (0 = no ink, 1 = solid ink).
///
/// The index order is the CMYK CLUT layout used throughout the workspace
/// (K the slowest axis, Y the fastest), so the table is also a 2⁴-node CLUT.
///
/// # Provenance
///
/// The values are the FOGRA39 characterisation data set (ISO 12647-2:2004
/// Amd 1, offset printing on paper types 1/2, gloss- and matte-coated) in
/// the form of the ECI *ISO Coated v2* output profile, evaluated at the
/// sixteen corners and converted to sRGB (IEC 61966-2-1) with the ICC
/// media-relative colorimetric intent and black point compensation.  Paper
/// white therefore maps to sRGB white and the four-colour solid to sRGB
/// black, while solid K alone keeps the slight lift of printed black.
///
/// Solid K is the neutral grey of the converted corner's luminance (the
/// conversion gives `[24, 23, 21]`).  The characterisation aims solid K at
/// `a* = b* = 0`; the warm cast is the media-relative mapping of the bluish
/// paper white, not a property of the ink.  A neutral corner keeps the whole
/// K axis achromatic: K-only content converts to exact greys.
pub const CORNERS: [[u8; 3]; 16] = [
    // K = 0
    [255, 255, 255], // paper
    [255, 237, 0],   // Y
    [229, 0, 125],   // M
    [227, 0, 11],    // M + Y
    [0, 158, 227],   // C
    [0, 150, 62],    // C + Y
    [46, 35, 131],   // C + M
    [47, 47, 45],    // C + M + Y
    // K = 1
    [23, 23, 23], // K
    [22, 24, 0],  // Y + K
    [34, 0, 0],   // M + K
    [30, 0, 0],   // M + Y + K
    [0, 3, 26],   // C + K
    [0, 11, 0],   // C + Y + K
    [0, 0, 0],    // C + M + K
    [0, 0, 0],    // C + M + Y + K
];

/// [`CORNERS`] flattened to 48 bytes: the 2⁴-node CLUT handed to GPU kernels.
pub const CORNER_TABLE: [u8; 48] = {
    let mut table = [0u8; 48];
    let mut i = 0;
    while i < 48 {
        table[i] = CORNERS[i / 3][i % 3];
        i += 1;
    }
    table
};

/// First-stage constants for one output channel.
///
/// Entry `j = k·4 + c·2 + m` holds the pair of corners that differ only in
/// Y: `base = 256 · V(y = 0)` and `step = V(y = 1) − V(y = 0)`, so the Y
/// stage is `base + step · w_y`.  `pair = V(y = 0) | V(y = 1) << 16` packs
/// the same two corners as 16-bit halves for multiply-add instructions:
/// `V(y = 0)·(256 − w_y) + V(y = 1)·w_y` is the identical integer.
#[derive(Clone, Copy)]
pub(crate) struct YStage {
    pub(crate) base: [i32; 8],
    pub(crate) step: [i32; 8],
    #[cfg_attr(
        not(target_arch = "x86_64"),
        expect(dead_code, reason = "read only by the x86-64 multiply-add tiers")
    )]
    pub(crate) pair: [i32; 8],
}

const fn y_stage(channel: usize) -> YStage {
    let mut base = [0i32; 8];
    let mut step = [0i32; 8];
    let mut pair = [0i32; 8];
    let mut j = 0;
    while j < 8 {
        let lo = CORNERS[2 * j][channel] as i32;
        let hi = CORNERS[2 * j + 1][channel] as i32;
        base[j] = lo << 8;
        step[j] = hi - lo;
        pair[j] = lo | (hi << 16);
        j += 1;
    }
    YStage { base, step, pair }
}

/// [`YStage`] for the R, G and B channels.
pub(crate) const Y_STAGES: [YStage; 3] = [y_stage(0), y_stage(1), y_stage(2)];

/// Interpolation weight on `[0, 256]` of an ink byte.
#[inline]
#[must_use]
pub(crate) const fn weight(x: u8) -> i32 {
    let x = x as i32;
    x + (x >> 7)
}

/// `256·a + (b − a)·w`: one interpolation step, adding 8 fractional bits.
#[inline]
const fn lerp(a: i32, b: i32, w: i32) -> i32 {
    (a << 8) + (b - a) * w
}

/// Drop 8 fractional bits with round-half-up.
#[inline]
const fn round8(v: i32) -> i32 {
    (v + 128) >> 8
}

/// One output channel from the four ink weights `[c, m, y, k]` on `[0, 256]`.
#[inline]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a convex combination of corner bytes lies in [0, 255]"
)]
const fn channel(stage: &YStage, w: [i32; 4]) -> u8 {
    let [wc, wm, wy, wk] = w;
    let s = &stage.base;
    let d = &stage.step;
    // Y stage: index j = k·4 + c·2 + m, 8 fractional bits.
    let y0 = s[0] + d[0] * wy;
    let y1 = s[1] + d[1] * wy;
    let y2 = s[2] + d[2] * wy;
    let y3 = s[3] + d[3] * wy;
    let y4 = s[4] + d[4] * wy;
    let y5 = s[5] + d[5] * wy;
    let y6 = s[6] + d[6] * wy;
    let y7 = s[7] + d[7] * wy;
    // M stage: index k·2 + c.
    let m0 = round8(lerp(y0, y1, wm));
    let m1 = round8(lerp(y2, y3, wm));
    let m2 = round8(lerp(y4, y5, wm));
    let m3 = round8(lerp(y6, y7, wm));
    // C stage: index k.
    let c0 = round8(lerp(m0, m1, wc));
    let c1 = round8(lerp(m2, m3, wc));
    // K stage, rounded to an integer in [0, 255].
    ((lerp(c0, c1, wk) + (1 << 15)) >> 16) as u8
}

/// sRGB of the CMYK value with ink weights `[c, m, y, k]` on `[0, 256]`.
#[inline]
const fn from_weights(w: [i32; 4]) -> [u8; 3] {
    [
        channel(&Y_STAGES[0], w),
        channel(&Y_STAGES[1], w),
        channel(&Y_STAGES[2], w),
    ]
}

/// Convert one CMYK pixel (bytes, 0 = no ink, 255 = solid ink) to sRGB.
///
/// This is the reference evaluation; [`cmyk_to_rgb_row`] produces the same
/// bytes for every input.
#[inline]
#[must_use]
pub const fn cmyk_to_rgb(c: u8, m: u8, y: u8, k: u8) -> [u8; 3] {
    from_weights([weight(c), weight(m), weight(y), weight(k)])
}

/// Weight on `[0, 256]` of a normalised ink amount; NaN counts as no ink.
#[inline]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the clamped product lies in [0.0, 256.0] and is rounded first"
)]
fn unit_weight(t: f64) -> i32 {
    if t.is_nan() {
        return 0;
    }
    (t.clamp(0.0, 1.0) * 256.0).round() as i32
}

/// Convert normalised PDF CMYK components (`[0, 1]` per ink) to sRGB bytes.
///
/// Components are clamped to `[0, 1]`; a NaN component counts as no ink.
#[inline]
#[must_use]
pub fn cmyk_f64_to_rgb(c: f64, m: f64, y: f64, k: f64) -> [u8; 3] {
    from_weights([
        unit_weight(c),
        unit_weight(m),
        unit_weight(y),
        unit_weight(k),
    ])
}

/// Convert interleaved CMYK pixels (4 bytes each) to interleaved sRGB (3 bytes each).
///
/// Output is byte-identical to [`cmyk_to_rgb`] per pixel; the best SIMD
/// tier available at runtime handles whole blocks:
/// - x86-64: AVX-512 (F, BW, VBMI; 16 px/block) → AVX2 (8 px/block) → scalar
/// - aarch64: NEON (16 px/block)
/// - other targets: scalar
///
/// # Panics
///
/// Panics if `cmyk.len()` is not a multiple of 4 or
/// `rgb.len() != cmyk.len() / 4 * 3`.
pub fn cmyk_to_rgb_row(cmyk: &[u8], rgb: &mut [u8]) {
    assert!(
        cmyk.len().is_multiple_of(4),
        "cmyk_to_rgb_row: CMYK length {} is not a multiple of 4",
        cmyk.len()
    );
    assert_eq!(
        rgb.len(),
        cmyk.len() / 4 * 3,
        "cmyk_to_rgb_row: RGB length must be 3 bytes per CMYK pixel"
    );
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    simd::convert(cmyk, rgb);
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    convert_scalar(cmyk, rgb);
}

/// Allocating form of [`cmyk_to_rgb_row`].
///
/// # Panics
///
/// Panics if `cmyk.len()` is not a multiple of 4.
#[must_use]
pub fn cmyk_to_rgb_vec(cmyk: &[u8]) -> Vec<u8> {
    let mut rgb = vec![0u8; cmyk.len() / 4 * 3];
    cmyk_to_rgb_row(cmyk, &mut rgb);
    rgb
}

/// Per-pixel loop: the tail of every SIMD tier and the whole conversion on
/// targets without one.  Lengths are validated by [`cmyk_to_rgb_row`].
#[inline]
pub(crate) fn convert_scalar(cmyk: &[u8], rgb: &mut [u8]) {
    for (src, dst) in cmyk
        .as_chunks::<4>()
        .0
        .iter()
        .zip(rgb.as_chunks_mut::<3>().0)
    {
        *dst = cmyk_to_rgb(src[0], src[1], src[2], src[3]);
    }
}
