//! Area-averaging resampling of an image drawn smaller than its resolution.
//!
//! The blit samples one source pixel per device pixel (see
//! [`super::image_sampler`]).  When an image is drawn below its resolution,
//! that point sampling skips source pixels, so features narrower than the
//! sampling step — thin strokes of lettering, fine rules — break up or
//! vanish, or, for a bilevel stencil, land as whole dark pixels.  Along
//! every axis on which the draw reduces, the image is first resampled with
//! an exact box filter: each output sample is the mean of the source
//! samples under its footprint, each weighted by the length of its overlap.
//!
//! When the image axes map onto the device axes (any combination of scale,
//! flip and quarter turn), the footprint of an output sample is exactly one
//! device pixel: output sample `i` along a reduced axis covers device pixel
//! `k₀ + i` (or `k₁ − 1 − i` on a flipped axis), where `[k₀, k₁)` are the
//! pixels whose centres the image covers.  The resampled image comes with a
//! matrix that maps it onto those pixels one sample per pixel, so the
//! nearest-sample blit reads each device pixel's exact area average.  On a
//! rotated or skewed draw a device pixel's footprint is a parallelogram in
//! image space; there the image is resampled along its own axes to its drawn
//! extent, and the blit samples the result at about one sample per pixel.
//!
//! Axes the draw does not reduce — including those within
//! [`REDUCTION_MARGIN`] of one sample per device pixel — keep their
//! samples, so full-resolution and upscaled draws are untouched.
//!
//! The filter is separable: a vertical pass accumulates whole source rows,
//! each scaled by its weight, into one output row, and a horizontal pass
//! then reduces that row.  Weights are 16-bit fixed point summing to
//! exactly 65 536 per output sample, rounded cumulatively so that each
//! partial sum is within half a count of exact; the vertical pass keeps 8
//! fraction bits, so the result is the exact mean to within one count.
//! Every intermediate fits `u32`: a vertical sum is at most `255 · 2¹⁶` and
//! a horizontal one at most `(255 · 2⁸) · 2¹⁶`.

use crate::renderer::gstate::Ctm;
use crate::resources::SoftMask;
use crate::resources::image::{ImageData, ImageDescriptor, ImageFilter};

/// Fraction bits of a filter weight; the weights of one output sum to
/// `1 << WEIGHT_BITS`.
const WEIGHT_BITS: u32 = 16;

/// Fraction bits the vertical pass keeps in its output.
const MID_BITS: u32 = 8;

/// A CTM component at most this large (device pixels per unit square) is
/// treated as zero when classifying a draw as axis-aligned: the image edge
/// it skews then drifts by under half a device pixel over its length.
pub(super) const AXIS_TOLERANCE: f64 = 0.5;

/// An axis reduces when its source samples exceed its device pixels by more
/// than this relative margin; below it, a draw counts as full resolution.
/// Within the margin point sampling skips at most one sample in 33, while
/// the box filter would blend almost every device pixel with a neighbour
/// by the sub-pixel phase of the two grids.
pub(super) const REDUCTION_MARGIN: f64 = 1.0 / 32.0;

/// Resample `img` for drawing through `ctm`.
///
/// Returns the resampled image and the matrix to draw it through, or `None`
/// — draw `img` as is through `ctm` — when the draw reduces neither axis or
/// the samples are not host-resident.  A stencil mask resamples like a grey
/// image: its samples are 0 (paint) and 255 (transparent), so each output
/// sample is 255 minus the paint coverage of its footprint.  The result
/// carries no mask: the caller resamples the original's mask separately
/// with [`reduce_mask_for_draw`], on the mask's own grid.
pub(super) fn reduce_for_draw(img: &ImageDescriptor, ctm: &Ctm) -> Option<(ImageDescriptor, Ctm)> {
    let plan = Plan::new(ctm, img.width, img.height)?;
    let src = img.data.as_cpu()?;
    let out = plan.resample(src, img.color_space.bytes_per_pixel())?;
    Some((
        ImageDescriptor {
            width: plan.out_w,
            height: plan.out_h,
            color_space: img.color_space,
            data: ImageData::Cpu(out),
            smask: None,
            filter: ImageFilter::Raw,
        },
        plan.ctm,
    ))
}

/// Resample `mask` for drawing through `ctm`, as [`reduce_for_draw`] does
/// an image: each output value is the mean alpha of its footprint, so a
/// bilevel mask becomes the fraction of each device pixel it admits.
pub(super) fn reduce_mask_for_draw(mask: &SoftMask, ctm: &Ctm) -> Option<(SoftMask, Ctm)> {
    let plan = Plan::new(ctm, mask.width(), mask.height())?;
    let out = plan.resample(mask.alpha(), 1)?;
    Some((SoftMask::new(plan.out_w, plan.out_h, out)?, plan.ctm))
}

/// Box-filter taps along one axis: output sample `i` is the weighted sum of
/// source samples `start[i] .. start[i] + width` with weights
/// `weights[i * width ..][.. width]`, which sum to `1 << WEIGHT_BITS`.
///
/// Every output has the same number of taps — the widest footprint's,
/// zero-padded — so the per-output loop runs a constant trip count.
struct AxisFilter {
    width: usize,
    start: Vec<u32>,
    weights: Vec<u32>,
}

impl AxisFilter {
    /// Taps for one output sample per footprint in `footprints`, each a
    /// source interval `[s0, s1)` in sample units over `0 .. n`.
    fn new(n: u32, footprints: impl ExactSizeIterator<Item = (f64, f64)>) -> Self {
        let len = footprints.len();
        let mut first = Vec::with_capacity(len);
        let mut offset = Vec::with_capacity(len + 1);
        let mut ragged = Vec::with_capacity(len * 4);
        offset.push(0);
        for (s0, s1) in footprints {
            first.push(footprint_taps(n, s0, s1, &mut ragged));
            offset.push(ragged.len());
        }
        let width = offset.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(1);
        let mut filter = Self {
            width,
            start: Vec::with_capacity(len),
            weights: vec![0; len * width],
        };
        // A footprint's taps lie in `0 .. n`, so no window is wider than `n`.
        let last_start = n - u32::try_from(width).expect("filter width is at most n");
        for (i, (&f, span)) in first.iter().zip(offset.windows(2)).enumerate() {
            // Shift the window left where it would run off the end; the
            // footprint's taps then sit `f - start` into it.
            let start = f.min(last_start);
            let at = (f - start) as usize;
            filter.weights[i * width + at..][..span[1] - span[0]]
                .copy_from_slice(&ragged[span[0]..span[1]]);
            filter.start.push(start);
        }
        filter
    }

    /// `(first source sample, weights)` of output sample `i`.
    #[inline]
    fn taps(&self, i: usize) -> (usize, &[u32]) {
        (
            self.start[i] as usize,
            &self.weights[i * self.width..][..self.width],
        )
    }
}

/// Append to `weights` the taps of footprint `[s0, s1)`, clipped to
/// `[0, n)`, and return the first sample they weight.  A footprint that
/// clips to nothing takes the nearest sample.
///
/// Each weight is the difference of consecutive rounded partial sums of the
/// scaled overlaps, so the weights sum to exactly one.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "sample indices are clamped to [0, n) before the cast; partial sums are in [0, 2^16]"
)]
fn footprint_taps(n: u32, s0: f64, s1: f64, weights: &mut Vec<u32>) -> u32 {
    let nf = f64::from(n);
    let (s0, s1) = (s0.clamp(0.0, nf), s1.clamp(0.0, nf));
    let first = (s0.floor() as u32).min(n - 1);
    let last = ((s1.ceil() as u32).max(first + 1)).min(n);
    let total = s1 - s0;
    let one = 1u32 << WEIGHT_BITS;
    if total <= 0.0 {
        weights.push(one);
        return first;
    }
    let scale = f64::from(one) / total;
    let mut covered = 0.0;
    let mut prev = 0u32;
    for j in first..last {
        covered += (s1.min(f64::from(j + 1)) - s0.max(f64::from(j))).max(0.0);
        let next = if j + 1 == last {
            one
        } else {
            ((covered * scale).round() as u32).min(one)
        };
        weights.push(next.saturating_sub(prev));
        prev = next.max(prev);
    }
    first
}

/// How one source axis maps onto the device, and its filter if reduced.
struct AxisPlan {
    /// Output samples along the axis.
    n: u32,
    /// Device coordinate of the axis origin and its signed device extent:
    /// source position `s` of `n` lies at `origin + extent * s / n`.
    origin: f64,
    extent: f64,
    filter: Option<AxisFilter>,
}

impl AxisPlan {
    /// A reduced axis of `n` samples spanning device `origin ..
    /// origin + extent` along a device axis whose pixel edges are integers:
    /// one output sample per device pixel whose centre the axis covers.
    /// Unreduced when the axis has no more samples than device pixels.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "pixel counts are bounded by n (checked) and pixel indices are finite"
    )]
    fn device_aligned(n: u32, origin: f64, extent: f64) -> Option<Self> {
        let unreduced = || Self {
            n,
            origin,
            extent,
            filter: None,
        };
        if !reduces(n, extent.abs()) {
            return Some(unreduced());
        }
        let (lo, hi) = (origin.min(origin + extent), origin.max(origin + extent));
        let (k0, k1) = ((lo - 0.5).ceil(), (hi - 0.5).ceil());
        let count = k1 - k0;
        if !(count >= 1.0 && count < f64::from(n)) {
            // No pixel centre covered (the image draws nothing on this
            // axis), or the rounding to whole pixels leaves nothing to
            // reduce.
            return (count >= 1.0).then(unreduced);
        }
        let count_u = count as u32;
        let to_source = f64::from(n) / extent;
        let flipped = extent < 0.0;
        let footprints = (0..count_u).map(|i| {
            let k = if flipped {
                k1 - 1.0 - f64::from(i)
            } else {
                k0 + f64::from(i)
            };
            let (a, b) = ((k - origin) * to_source, (k + 1.0 - origin) * to_source);
            (a.min(b), a.max(b))
        });
        Some(Self {
            n: count_u,
            origin: if flipped { k1 } else { k0 },
            extent: if flipped { -count } else { count },
            filter: Some(AxisFilter::new(n, footprints)),
        })
    }

    /// An axis of `n` samples resampled along itself to `drawn` samples, the
    /// ceiling of its device length, when that reduces it.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "drawn is finite, at least 1 and below n (checked)"
    )]
    fn uniform(n: u32, length: f64) -> Self {
        let drawn = length.ceil();
        if !(drawn >= 1.0 && reduces(n, drawn)) {
            return Self {
                n,
                origin: 0.0,
                extent: 0.0,
                filter: None,
            };
        }
        let m = drawn as u32;
        let step = f64::from(n) / drawn;
        let footprints = (0..m).map(|i| (f64::from(i) * step, f64::from(i + 1) * step));
        Self {
            n: m,
            origin: 0.0,
            extent: 0.0,
            filter: Some(AxisFilter::new(n, footprints)),
        }
    }
}

/// `true` when `n` samples drawn over `pixels` device pixels reduce.
fn reduces(n: u32, pixels: f64) -> bool {
    f64::from(n) > pixels * (1.0 + REDUCTION_MARGIN)
}

/// The resampling of one image for one draw.
struct Plan {
    /// Source dimensions.
    src_w: u32,
    src_h: u32,
    /// Output dimensions.
    out_w: u32,
    out_h: u32,
    /// Filters along the source columns (x) and rows (y); `None` keeps the
    /// axis.  At least one is `Some`.
    cols: Option<AxisFilter>,
    rows: Option<AxisFilter>,
    /// The matrix the output is drawn through.
    ctm: Ctm,
}

impl Plan {
    /// Plan the resampling of a `width × height` image drawn through `ctm`,
    /// or `None` when no axis reduces.
    #[expect(
        clippy::many_single_char_names,
        reason = "a–f are the PDF matrix components"
    )]
    fn new(ctm: &Ctm, width: u32, height: u32) -> Option<Self> {
        if width == 0 || height == 0 || ctm.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let [a, b, c, d, e, f] = *ctm;
        // Device space here has integer pixel edges on both axes: x is the
        // bitmap column, y the bitmap row counted up from the bottom edge.
        // The image's u axis (columns, left to right) runs from (e, f)
        // along (a, b); its rows run top-down, from v = 1 to v = 0, so row
        // position t of h lies at v = 1 − t / h.
        let (cols, rows, ctm) = if b.abs() <= AXIS_TOLERANCE && c.abs() <= AXIS_TOLERANCE {
            // Columns along device x, rows along device y.
            let cols = AxisPlan::device_aligned(width, e, a)?;
            let rows = AxisPlan::device_aligned(height, f + d, -d)?;
            let ctm = [
                cols.extent,
                0.0,
                0.0,
                -rows.extent,
                cols.origin,
                rows.origin + rows.extent,
            ];
            (cols, rows, ctm)
        } else if a.abs() <= AXIS_TOLERANCE && d.abs() <= AXIS_TOLERANCE {
            // A quarter turn: columns along device y, rows along device x.
            let cols = AxisPlan::device_aligned(width, f, b)?;
            let rows = AxisPlan::device_aligned(height, e + c, -c)?;
            let ctm = [
                0.0,
                cols.extent,
                -rows.extent,
                0.0,
                rows.origin + rows.extent,
                cols.origin,
            ];
            (cols, rows, ctm)
        } else {
            (
                AxisPlan::uniform(width, a.hypot(b)),
                AxisPlan::uniform(height, c.hypot(d)),
                *ctm,
            )
        };
        if cols.filter.is_none() && rows.filter.is_none() {
            return None;
        }
        Some(Self {
            src_w: width,
            src_h: height,
            out_w: cols.n,
            out_h: rows.n,
            cols: cols.filter,
            rows: rows.filter,
            ctm,
        })
    }

    /// Resample `src`, `src_w × src_h` pixels of `bpp` bytes, into
    /// `out_w × out_h`.  `None` when `src` is not exactly that size.
    fn resample(&self, src: &[u8], bpp: usize) -> Option<Vec<u8>> {
        let (w, h) = (self.src_w as usize, self.src_h as usize);
        let row_len = w.checked_mul(bpp)?;
        if src.len() != row_len.checked_mul(h)? {
            return None;
        }
        let (out_w, out_h) = (self.out_w as usize, self.out_h as usize);
        let out_row_len = out_w * bpp;
        let mut out = vec![0u8; out_row_len.checked_mul(out_h)?];
        // `acc` holds the vertical pass's row carrying `frac` fraction bits.
        let mut acc = vec![0u32; row_len];

        for (oy, out_row) in out.chunks_exact_mut(out_row_len).enumerate() {
            let frac = if let Some(rows) = &self.rows {
                acc.fill(0);
                let (first, weights) = rows.taps(oy);
                for (j, &wt) in weights.iter().enumerate() {
                    if wt != 0 {
                        accumulate_row(&mut acc, &src[(first + j) * row_len..][..row_len], wt);
                    }
                }
                WEIGHT_BITS
            } else {
                for (a, &v) in acc.iter_mut().zip(&src[oy * row_len..][..row_len]) {
                    *a = u32::from(v);
                }
                0
            };
            match &self.cols {
                Some(cols) => {
                    // At most MID_BITS fraction bits enter the second pass.
                    let shift = frac.saturating_sub(MID_BITS);
                    if shift > 0 {
                        for a in &mut acc {
                            *a = (*a + (1 << (shift - 1))) >> shift;
                        }
                    }
                    reduce_row(&acc, cols, bpp, out_row, frac - shift + WEIGHT_BITS);
                }
                None => {
                    for (o, &a) in out_row.iter_mut().zip(&acc) {
                        *o = round_shift(a, frac);
                    }
                }
            }
        }
        Some(out)
    }
}

/// `acc[x] += weight * row[x]` over a whole row.
#[inline]
fn accumulate_row(acc: &mut [u32], row: &[u8], weight: u32) {
    for (a, &v) in acc.iter_mut().zip(row) {
        *a += weight * u32::from(v);
    }
}

/// `round(v / 2^shift)` as a byte; the filter's sums never exceed
/// `255 << shift`.
#[inline]
fn round_shift(v: u32, shift: u32) -> u8 {
    ((v + (1 << (shift - 1))) >> shift).min(255) as u8
}

/// Reduce one row of `bpp`-channel samples carrying `shift − WEIGHT_BITS`
/// fraction bits through `cols` into `out`.
///
/// The common shapes — one channel up to twelve taps, three channels up to
/// eight — run a fully unrolled loop.
fn reduce_row(row: &[u32], cols: &AxisFilter, bpp: usize, out: &mut [u8], shift: u32) {
    macro_rules! fixed {
        ($($n:literal: [$($t:literal),*]),*) => {
            match (bpp, cols.width) {
                $($(($n, $t) => return reduce_row_fixed::<$n, $t>(row, cols, out, shift),)*)*
                _ => {}
            }
        };
    }
    fixed!(1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 3: [1, 2, 3, 4, 5, 6, 7, 8]);
    let width = cols.width;
    let pixels = out.chunks_exact_mut(bpp).zip(&cols.start);
    for ((px, &start), weights) in pixels.zip(cols.weights.chunks_exact(width)) {
        let src = &row[start as usize * bpp..][..width * bpp];
        for (ch, o) in px.iter_mut().enumerate() {
            let sum: u32 = weights
                .iter()
                .zip(src.chunks_exact(bpp))
                .map(|(&wt, s)| wt * s[ch])
                .sum();
            *o = round_shift(sum, shift);
        }
    }
}

/// [`reduce_row`] for `N` channels and a filter `T` taps wide.
#[inline]
fn reduce_row_fixed<const N: usize, const T: usize>(
    row: &[u32],
    cols: &AxisFilter,
    out: &mut [u8],
    shift: u32,
) {
    let pixels = out.as_chunks_mut::<N>().0.iter_mut().zip(&cols.start);
    for ((px, &start), weights) in pixels.zip(cols.weights.as_chunks::<T>().0) {
        let src = &row[start as usize * N..][..T * N];
        let mut sum = [0u32; N];
        for t in 0..T {
            for ch in 0..N {
                sum[ch] += weights[t] * src[t * N + ch];
            }
        }
        for (o, s) in px.iter_mut().zip(sum) {
            *o = round_shift(s, shift);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::image::ImageColorSpace;

    fn gray(width: u32, height: u32, data: Vec<u8>) -> ImageDescriptor {
        ImageDescriptor {
            width,
            height,
            color_space: ImageColorSpace::Gray,
            data: ImageData::Cpu(data),
            smask: None,
            filter: ImageFilter::Raw,
        }
    }

    /// Overlap of footprint `[s0, s1)` with sample `j`.
    fn overlap(s0: f64, s1: f64, j: usize) -> f64 {
        #[expect(clippy::cast_precision_loss, reason = "test indices are tiny")]
        let j = j as f64;
        (s1.min(j + 1.0) - s0.max(j)).max(0.0)
    }

    /// Floating-point mean of `values` over footprint `[s0, s1)`.
    fn exact_mean(values: &[u8], s0: f64, s1: f64) -> f64 {
        values
            .iter()
            .enumerate()
            .map(|(j, &v)| overlap(s0, s1, j) * f64::from(v))
            .sum::<f64>()
            / (s1 - s0)
    }

    /// A one-pixel-wide black stroke in every fourth column survives a 4:1
    /// reduction as a uniform quarter-grey, where point sampling at that
    /// step lands on the stroke either always or never.
    #[test]
    fn thin_stroke_contributes_to_every_pixel() {
        let (w, h) = (16u32, 4u32);
        let data = (0..h)
            .flat_map(|_| (0..w).map(|x| if x % 4 == 1 { 0 } else { 255 }))
            .collect();
        // Drawn 4 × 1 device pixels at the origin.
        let ctm = [4.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let (out, out_ctm) = reduce_for_draw(&gray(w, h, data), &ctm).expect("reduced");
        assert_eq!((out.width, out.height), (4, 1));
        assert_eq!(out.data.as_cpu().unwrap(), &[191, 191, 191, 191]);
        assert_eq!(out_ctm, [4.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    /// Ten samples over three device pixels: each pixel averages 3⅓
    /// samples, splitting the boundary samples by their overlap.
    #[test]
    fn fractional_ratio_weights_boundary_samples_by_overlap() {
        let data: Vec<u8> = vec![0, 30, 60, 90, 120, 150, 180, 210, 240, 255];
        let ctm = [3.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let (out, _) = reduce_for_draw(&gray(10, 1, data.clone()), &ctm).expect("reduced");
        let out = out.data.as_cpu().unwrap();
        let third = 10.0 / 3.0;
        for (i, &got) in out.iter().enumerate() {
            let i = f64::from(u8::try_from(i).unwrap());
            let want = exact_mean(&data, i * third, (i + 1.0) * third);
            assert!((f64::from(got) - want).abs() <= 0.6, "{i}: {got} vs {want}");
        }
    }

    /// A flipped, offset, non-integer draw reproduces the floating-point
    /// area average of every device pixel to within one count, and the
    /// returned matrix places output sample i on device pixel k₁ − 1 − i.
    #[test]
    fn flipped_fractional_draw_matches_exact_means() {
        let (w, h) = (37usize, 23usize);
        let data: Vec<u8> = (0..w * h)
            .map(|i| u8::try_from((i * 97 + i / 7) % 256).unwrap())
            .collect();
        // x: 37 samples from device 20.3 back to 9.1 — flipped, ratio ≈ 3.3.
        // y: 23 rows from device 10.6 (top row) down to 3.7 — ratio ≈ 3.3.
        let ctm = [-11.2, 0.0, 0.0, 6.9, 20.3, 3.7];
        let plan = Plan::new(&ctm, 37, 23).expect("plan");
        let out = plan.resample(&data, 1).expect("resampled");
        // Pixel centres covered: x 9.5 ..= 19.5 → 11 px; y 4.5 ..= 10.5 → 7 px.
        assert_eq!((plan.out_w, plan.out_h), (11, 7));
        assert_eq!(plan.ctm, [-11.0, 0.0, 0.0, 7.0, 20.0, 4.0]);
        // Footprint of device pixel k along an axis of n samples running
        // from `origin` with signed `extent`, in sample units.
        let footprint = |k: f64, n: f64, origin: f64, extent: f64| {
            let (a, b) = ((k - origin) * n / extent, (k + 1.0 - origin) * n / extent);
            (a.min(b).clamp(0.0, n), a.max(b).clamp(0.0, n))
        };
        for oy in 0..7u8 {
            // Output row 0 is the source top, on device row 10.
            let (t0, t1) = footprint(10.0 - f64::from(oy), 23.0, 10.6, -6.9);
            for ox in 0..11u8 {
                // Output column 0 is the source left, on device column 19.
                let (s0, s1) = footprint(19.0 - f64::from(ox), 37.0, 20.3, -11.2);
                let mut sum = 0.0;
                for (y, row) in data.chunks_exact(w).enumerate() {
                    let wy = overlap(t0, t1, y);
                    if wy > 0.0 {
                        sum += wy * exact_mean(row, s0, s1);
                    }
                }
                let want = sum / (t1 - t0);
                let got = f64::from(out[usize::from(oy) * 11 + usize::from(ox)]);
                assert!((got - want).abs() <= 1.0, "({ox}, {oy}): {got} vs {want}");
            }
        }
    }

    #[test]
    fn rgb_channels_average_independently() {
        // 4×1 RGB drawn 2 px wide.
        let img = ImageDescriptor {
            color_space: ImageColorSpace::Rgb,
            ..gray(4, 1, vec![255, 0, 0, 0, 0, 255, 10, 20, 30, 30, 40, 50])
        };
        let (out, _) = reduce_for_draw(&img, &[2.0, 0.0, 0.0, 1.0, 0.0, 0.0]).expect("reduced");
        assert_eq!((out.width, out.height), (2, 1));
        assert_eq!(out.data.as_cpu().unwrap(), &[128, 0, 128, 20, 30, 40]);
    }

    /// A quarter-turn draw reduces the source rows along device x and the
    /// columns along device y.
    #[test]
    fn quarter_turn_reduces_along_the_rotated_axes() {
        // 8 columns × 4 rows; rows 0..2 black, 2..4 white.
        let data: Vec<u8> = (0..4u8)
            .flat_map(|y| [if y < 2 { 0 } else { 255 }; 8])
            .collect();
        // u (columns) along +y over 4 px, v along −x: rows run along +x.
        let ctm = [0.0, 4.0, -2.0, 0.0, 2.0, 0.0];
        let plan = Plan::new(&ctm, 8, 4).expect("plan");
        let out = plan.resample(&data, 1).expect("resampled");
        assert_eq!((plan.out_w, plan.out_h), (4, 2));
        // Row t of 4 lies at device x = (2 − 2) + 2·t/4: the black top rows
        // are at x ∈ [0, 1).  The output keeps source orientation.
        assert_eq!(out, [0, 0, 0, 0, 255, 255, 255, 255]);
        assert_eq!(plan.ctm, [0.0, 4.0, -2.0, 0.0, 2.0, 0.0]);
    }

    /// Only the reduced axis is resampled; the upscaled one keeps its
    /// samples and its mapping.
    #[test]
    fn unreduced_axis_keeps_its_samples() {
        let data: Vec<u8> = (0..2u8).flat_map(|y| [y * 100; 8]).collect();
        let ctm = [2.0, 0.0, 0.0, 50.0, 0.0, 0.0];
        let (out, out_ctm) = reduce_for_draw(&gray(8, 2, data), &ctm).expect("reduced");
        assert_eq!((out.width, out.height), (2, 2));
        assert_eq!(out.data.as_cpu().unwrap(), &[0, 0, 100, 100]);
        assert_eq!(out_ctm, [2.0, 0.0, 0.0, 50.0, 0.0, 0.0]);
    }

    #[test]
    fn full_resolution_and_upscaled_draws_are_not_resampled() {
        let img = gray(10, 10, vec![0; 100]);
        for ctm in [
            [10.0, 0.0, 0.0, 10.0, 0.5, 0.25],
            [9.75, 0.0, 0.0, -9.75, 0.3, 20.0],
            [25.0, 0.0, 0.0, -12.0, 3.0, 40.0],
            [0.0, 10.0, 10.0, 0.0, 0.0, 0.0],
            [9.0, 4.4, -4.4, 9.0, 0.0, 0.0],
        ] {
            assert!(reduce_for_draw(&img, &ctm).is_none(), "{ctm:?}");
        }
        let mask = SoftMask::new(10, 10, vec![0; 100]).unwrap();
        assert!(reduce_mask_for_draw(&mask, &[6.0, 0.0, 0.0, 12.0, 0.0, 0.0]).is_some());
        assert!(reduce_mask_for_draw(&mask, &[12.0, 0.0, 0.0, 12.0, 0.0, 0.0]).is_none());
    }

    /// A rotated draw resamples along the image's own axes to its drawn
    /// extent and keeps the matrix.
    #[test]
    fn skewed_draw_resamples_to_its_drawn_extent() {
        let img = gray(40, 40, vec![255; 1600]);
        let ctm = [6.0, 8.0, -8.0, 6.0, 50.0, 50.0];
        let (out, out_ctm) = reduce_for_draw(&img, &ctm).expect("reduced");
        assert_eq!((out.width, out.height), (10, 10));
        assert!(out.data.as_cpu().unwrap().iter().all(|&v| v == 255));
        assert_eq!(out_ctm, ctm);
    }

    /// A stencil's resampled sample is 255 minus the paint coverage of its
    /// footprint: one painted sample in four leaves 191.
    #[test]
    fn stencil_reduces_to_its_paint_coverage() {
        let stencil = ImageDescriptor {
            color_space: ImageColorSpace::Mask,
            ..gray(4, 2, vec![0, 255, 0, 0, 255, 255, 0, 0])
        };
        let (out, _) = reduce_for_draw(&stencil, &[2.0, 0.0, 0.0, 1.0, 0.0, 0.0]).expect("reduced");
        assert_eq!(out.color_space, ImageColorSpace::Mask);
        assert_eq!((out.width, out.height), (2, 1));
        assert_eq!(out.data.as_cpu().unwrap(), &[191, 0]);
    }

    /// A bilevel mask reduces to the fraction of each pixel it admits.
    #[test]
    fn mask_reduces_to_its_mean_alpha() {
        let mask = SoftMask::new(6, 3, [[255, 0, 0, 255, 255, 255]; 3].concat()).unwrap();
        let (out, _) =
            reduce_mask_for_draw(&mask, &[2.0, 0.0, 0.0, 1.0, 0.0, 0.0]).expect("reduced");
        assert_eq!((out.width(), out.height()), (2, 1));
        assert_eq!(out.alpha(), &[85, 255]);
    }

    /// Weights of every output sum to exactly one, whatever the ratio.
    #[test]
    fn weights_always_sum_to_one() {
        for (n, m) in [(7u32, 3u32), (1000, 299), (65, 64), (3, 1), (5000, 7)] {
            let step = f64::from(n) / f64::from(m);
            let f = AxisFilter::new(
                n,
                (0..m).map(|i| (f64::from(i) * step, f64::from(i + 1) * step)),
            );
            for i in 0..f.start.len() {
                let (first, w) = f.taps(i);
                assert_eq!(w.iter().sum::<u32>(), 1 << WEIGHT_BITS, "{n}/{m} #{i}");
                assert!(first + w.len() <= n as usize);
            }
        }
    }
}
