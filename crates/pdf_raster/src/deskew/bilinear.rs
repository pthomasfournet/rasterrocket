//! The scalar bilinear rotation shared by every sample depth: the
//! per-row inverse-map parameters, the 2×2 footprint, the blend, and the
//! row driver the SIMD row kernels plug into.
//!
//! Rotation is clockwise-positive about the image centre; a source
//! coordinate outside `[0, w−1] × [0, h−1]` is white (`Sample::MAX`).  The
//! blend is evaluated in `f32` and rounded half away from zero, which is
//! exact for samples of up to 16 bits.

use color::Sample;
use raster::Bitmap;

use super::DeskewPixel;

/// Parameters shared between the scalar and SIMD row kernels.
///
/// Passed by value to avoid threading individual scalars through dispatch helpers.
pub(super) struct RowParams {
    pub(super) cos_a: f32,
    pub(super) sin_a: f32,
    pub(super) sx_base: f32,
    pub(super) sy_base: f32,
    pub(super) sx_max: i32,
    pub(super) sy_max: i32,
}

/// The 2×2 bilinear footprint `[p00, p10, p01, p11]` at origin `(x0, y0)`.
///
/// `x0` and `y0` must already be bounds-checked (`0 ≤ x0 ≤ sx_max = w-1`,
/// `0 ≤ y0 ≤ sy_max = h-1`). On the last column/row the footprint is clamped
/// to the edge, so the missing neighbour duplicates the edge pixel and the
/// interpolation degrades gracefully to nearest-neighbour there.
#[inline]
fn footprint<P: DeskewPixel>(src: &Bitmap<P>, x0: usize, y0: usize) -> [P::Sample; 4] {
    let x1 = (x0 + 1).min(src.width as usize - 1);
    let y1 = (y0 + 1).min(src.height as usize - 1);
    // y0 ≤ sy_max = h-1 ≤ 32767 and y1 ≤ h-1 — safe casts.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "y0 ≤ h-1 ≤ 32767 and y1 ≤ h-1; both fit in u32"
    )]
    let (r0, r1) = (src.row_samples(y0 as u32), src.row_samples(y1 as u32));
    [r0[x0], r0[x1], r1[x0], r1[x1]]
}

/// Bilinear interpolation at source `(sx, sy)` given pre-floored origin `(x0, y0)`
/// and fractions `(fx, fy)`; the origin contract is that of [`footprint`].
#[inline]
fn bilinear_sample<P: DeskewPixel>(
    src: &Bitmap<P>,
    x0: usize,
    y0: usize,
    fx: f32,
    fy: f32,
) -> P::Sample {
    let [p00, p10, p01, p11] = footprint(src, x0, y0).map(P::to_f32);

    let top = (p10 - p00).mul_add(fx, p00);
    let bot = (p11 - p01).mul_add(fx, p01);
    let val = (bot - top).mul_add(fy, top);

    // A bilinear blend of samples in [0, MAX] stays in [0, MAX], which f32
    // represents exactly at 16 bits; the rounded value converts exactly.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the rounded blend lies in [0, 65535]"
    )]
    P::Sample::from_signed(val.round() as i64)
}

/// Compute a single output pixel at column `ox`.
///
/// Returns white ([`Sample::MAX`]) when the source coordinates are out of
/// the valid bilinear-sampling range.
#[inline]
fn rotate_pixel_scalar<P: DeskewPixel>(src: &Bitmap<P>, ox: f32, p: &RowParams) -> P::Sample {
    let sx = p.cos_a.mul_add(ox, p.sx_base);
    let sy = p.sin_a.mul_add(ox, p.sy_base);

    // floor() → i32: coordinates bounded by ≤ 32768·√2 ≈ 46341 — fits in i32.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "image coordinates bounded by rotated extent ≤ 32768 * sqrt(2) ≈ 46341; fits in i32"
    )]
    let x0 = sx.floor() as i32;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "image coordinates bounded by rotated extent ≤ 32768 * sqrt(2) ≈ 46341; fits in i32"
    )]
    let y0 = sy.floor() as i32;

    if x0 < 0 || y0 < 0 || x0 > p.sx_max || y0 > p.sy_max {
        return P::Sample::MAX;
    }

    // x0 ≥ 0, y0 ≥ 0 verified above.
    #[expect(
        clippy::cast_sign_loss,
        reason = "x0 ≥ 0 and y0 ≥ 0 verified by the guard above"
    )]
    let (x0u, y0u) = (x0 as usize, y0 as usize);
    // x0 ≤ 46341 ≤ 2^23 — exactly representable in f32; subtraction is exact.
    #[expect(
        clippy::cast_precision_loss,
        reason = "x0 ≤ 46341 ≤ 2^23 — exactly representable in f32"
    )]
    let fx = sx - x0 as f32;
    #[expect(
        clippy::cast_precision_loss,
        reason = "y0 ≤ 46341 ≤ 2^23 — exactly representable in f32"
    )]
    let fy = sy - y0 as f32;

    bilinear_sample(src, x0u, y0u, fx, fy)
}

/// Scalar per-pixel loop over output columns `first .. first + span.len()`.
pub(super) fn rotate_span_scalar<P: DeskewPixel>(
    span: &mut [P::Sample],
    first: usize,
    src: &Bitmap<P>,
    p: &RowParams,
) {
    for (i, dst_px) in span.iter_mut().enumerate() {
        #[expect(
            clippy::cast_precision_loss,
            reason = "column ≤ DESKEW_MAX_SIDE = 32768; exact in f32"
        )]
        {
            *dst_px = rotate_pixel_scalar(src, (first + i) as f32, p);
        }
    }
}

/// Distance, in source pixels, from each edge of the sampled source range
/// within which [`rotate_border_scalar`] recomputes a sample.
#[cfg(feature = "gpu-deskew")]
const BORDER: f32 = 2.0;

/// The output columns of `0..w` whose source coordinate `k·ox + b` may lie
/// in `[lo, hi)`, widened by a column on each side.
#[cfg(feature = "gpu-deskew")]
fn columns_within(k: f32, b: f32, (lo, hi): (f32, f32), w: usize) -> std::ops::Range<usize> {
    if k == 0.0 {
        return if (lo..hi).contains(&b) { 0..w } else { 0..0 };
    }
    let (t0, t1) = ((lo - b) / k, (hi - b) / k);
    // w ≤ DESKEW_MAX_SIDE = 32768; exact in f32.
    #[expect(
        clippy::cast_precision_loss,
        reason = "w ≤ DESKEW_MAX_SIDE = 32768; exact in f32"
    )]
    let wf = w as f32;
    // Clamped to [0, w] before the casts; the bounds are finite for a
    // finite angle and non-zero `k`.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to [0, w] with w ≤ 32768"
    )]
    let (first, end) = (
        (t0.min(t1).floor() - 1.0).clamp(0.0, wf) as usize,
        (t0.max(t1).ceil() + 2.0).clamp(0.0, wf) as usize,
    );
    first..end.max(first)
}

/// Overwrite, with the scalar kernel's samples, every sample of the rows
/// of `dst` (`dst[oy * stride..][..width]`, a rotation of `src` by
/// `angle_deg` computed by another kernel) whose source position lies
/// within [`BORDER`] pixels of an edge of the sampled range
/// `[0, w) × [0, h)`: the samples where the white background meets the
/// edge-clamped footprint.  Every other sample is left as it is.
#[cfg(feature = "gpu-deskew")]
pub(super) fn rotate_border_scalar<P: DeskewPixel>(
    src: &Bitmap<P>,
    angle_deg: f32,
    dst: &mut [P::Sample],
    stride: usize,
) {
    let w = src.width as usize;
    // w, h ≤ DESKEW_MAX_SIDE = 32768; exact in f32.
    #[expect(
        clippy::cast_precision_loss,
        reason = "image dimensions ≤ DESKEW_MAX_SIDE = 32768; exact in f32"
    )]
    let (wf, hf) = (src.width as f32, src.height as f32);
    let bands = |end: f32| [(-BORDER, BORDER), (end - 1.0 - BORDER, end + BORDER)];
    rotate_rows(src, angle_deg, dst, stride, |row, src, p| {
        let [x_low, x_high] = bands(wf).map(|band| columns_within(p.cos_a, p.sx_base, band, w));
        let [y_low, y_high] = bands(hf).map(|band| columns_within(p.sin_a, p.sy_base, band, w));
        let mut spans = [x_low, x_high, y_low, y_high];
        spans.sort_unstable_by_key(|span| span.start);
        // An empty span has `start >= end`, so `first < span.end` skips it.
        let mut done = 0;
        for span in spans {
            let first = span.start.max(done);
            if first < span.end {
                rotate_span_scalar(&mut row[first..span.end], first, src, p);
                done = span.end;
            }
        }
    });
}

/// [`rotate_rows`] into a new packed bitmap.
pub(super) fn rotate_cpu_with<P: DeskewPixel>(
    src: &Bitmap<P>,
    angle_deg: f32,
    row_fn: impl Fn(&mut [P::Sample], &Bitmap<P>, &RowParams),
) -> Bitmap<P> {
    let mut dst = Bitmap::<P>::new(src.width, src.height, 1, false);
    let stride = dst.stride;
    rotate_rows(src, angle_deg, dst.data_mut(), stride, row_fn);
    dst
}

/// Row-loop driver shared by the rotations of every tier and the per-row
/// parity tests: computes the per-row inverse-map parameters and
/// hands each output row, `dst[oy * stride..][..width]`, to `row_fn`.
#[expect(
    clippy::similar_names,
    reason = "sx_*/sy_* are paired coordinate variables; renaming would obscure the symmetry"
)]
pub(super) fn rotate_rows<P: DeskewPixel>(
    src: &Bitmap<P>,
    angle_deg: f32,
    dst: &mut [P::Sample],
    stride: usize,
    row_fn: impl Fn(&mut [P::Sample], &Bitmap<P>, &RowParams),
) {
    let w = src.width as usize;
    let h = src.height as usize;

    let rad = angle_deg.to_radians();
    let cos_a = rad.cos();
    let sin_a = rad.sin();

    // Rotation centre: image centre (may be fractional).
    // w, h ≤ DESKEW_MAX_SIDE = 32768; exact in f32 (24-bit mantissa covers integers to 16 M).
    #[expect(
        clippy::cast_precision_loss,
        reason = "image dimensions ≤ DESKEW_MAX_SIDE = 32768; exact in f32"
    )]
    let cx = (w as f32 - 1.0) * 0.5;
    #[expect(
        clippy::cast_precision_loss,
        reason = "image dimensions ≤ DESKEW_MAX_SIDE = 32768; exact in f32"
    )]
    let cy = (h as f32 - 1.0) * 0.5;

    // Valid source coordinate range: x ∈ [0, w-1], y ∈ [0, h-1].
    // `bilinear_sample` clamps its 2×2 footprint to the edge, so the last
    // column/row are sampleable rather than white-filled.
    // w, h ≤ 32768 — safe to cast to i32 (fits in i32 range).
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "image dimensions ≤ DESKEW_MAX_SIDE = 32768; fits in i32"
    )]
    let sx_max = (w as i32) - 1;
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "image dimensions ≤ DESKEW_MAX_SIDE = 32768; fits in i32"
    )]
    let sy_max = (h as i32) - 1;

    for oy in 0..h {
        // oy < h ≤ 32768; safe to cast to u32.
        let dst_row = &mut dst[oy * stride..][..w];

        // Pre-compute y-dependent terms outside the x loop.
        // oy ≤ 32768; exact in f32.
        #[expect(
            clippy::cast_precision_loss,
            reason = "oy ≤ DESKEW_MAX_SIDE = 32768; exact in f32"
        )]
        let dy = oy as f32 - cy;
        let sx_base = cos_a.mul_add(-cx, -(sin_a * dy)) + cx;
        let sy_base = sin_a.mul_add(-cx, cos_a * dy) + cy;

        let params = RowParams {
            cos_a,
            sin_a,
            sx_base,
            sy_base,
            sx_max,
            sy_max,
        };
        row_fn(dst_row, src, &params);
    }
}
