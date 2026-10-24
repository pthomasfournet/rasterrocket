//! Analysis planes, pixel-format abstraction and inverse-mapped resampling.
//!
//! Detectors operate on a [`Plane`]: a tightly packed 8-bit luma grid,
//! usually box-downsampled.  Resampling operates on the caller's
//! [`Bitmap`] in its own pixel format, so a colour page keeps its colour.
//!
//! Coordinate convention: integer coordinates address pixel centres; the
//! continuous extent of a `w`-pixel row is `[-0.5, w - 0.5]`.

use color::{Gray8, Pixel, Rgb8};
use raster::Bitmap;
use rayon::prelude::*;

// ── Pixel formats ───────────────────────────────────────────────────────────

mod sealed {
    pub trait Sealed {}
    impl Sealed for color::Gray8 {}
    impl Sealed for color::Rgb8 {}
}

/// Pixel formats accepted by the camera-capture cleanup stages.
///
/// Detection always runs on luma; resampling interpolates every channel
/// of the native format.  Implemented for [`Gray8`] and [`Rgb8`] only.
pub trait CameraPixel: Pixel + sealed::Sealed {
    /// The luma plane of `img` (BT.709 for RGB, identity for gray).
    #[doc(hidden)]
    fn luma_plane(img: &Bitmap<Self>) -> Plane;
}

impl CameraPixel for Gray8 {
    fn luma_plane(img: &Bitmap<Self>) -> Plane {
        Plane::from_rows(img.width, img.height, |y| {
            &img.row_bytes(y)[..img.width as usize]
        })
    }
}

impl CameraPixel for Rgb8 {
    fn luma_plane(img: &Bitmap<Self>) -> Plane {
        let gray = crate::rgb_to_gray(img);
        Gray8::luma_plane(&gray)
    }
}

// ── Plane ───────────────────────────────────────────────────────────────────

/// A tightly packed 8-bit single-channel image, row-major, top-down.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct Plane {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

impl Plane {
    fn from_rows<'a>(width: u32, height: u32, row: impl Fn(u32) -> &'a [u8]) -> Self {
        let (w, h) = (width as usize, height as usize);
        let mut data = Vec::with_capacity(w * h);
        for y in 0..height {
            data.extend_from_slice(row(y));
        }
        Self { w, h, data }
    }

    /// Value at `(x, y)`.  Caller guarantees `x < w`, `y < h`.
    pub fn at(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.w + x]
    }

    /// Row `y` as a slice of `w` values.
    pub fn row(&self, y: usize) -> &[u8] {
        &self.data[y * self.w..(y + 1) * self.w]
    }

    /// Smallest integer factor `f ≥ 1` with `max(w, h) / f ≤ max_side`.
    pub fn factor_for(&self, max_side: usize) -> usize {
        self.w.max(self.h).div_ceil(max_side.max(1)).max(1)
    }

    /// Box-filter downsample by an integer `factor`; trailing pixels that do
    /// not fill a whole block are discarded.  Downsampled pixel `i` covers
    /// source pixels `[i·f, (i+1)·f)`, so its centre is at `(i + 0.5)·f − 0.5`
    /// in source coordinates (see [`Self::upscale_coord`]).
    pub fn downsample(&self, factor: usize) -> Self {
        if factor <= 1 {
            return self.clone();
        }
        let (w, h) = (self.w / factor, self.h / factor);
        let area = u32::try_from(factor * factor).unwrap_or(u32::MAX);
        let mut data = vec![0u8; w * h];
        data.par_chunks_mut(w.max(1))
            .enumerate()
            .for_each(|(y, out)| {
                let mut acc = vec![0u32; w];
                for sy in y * factor..(y + 1) * factor {
                    let src = self.row(sy);
                    for (x, a) in acc.iter_mut().enumerate() {
                        *a += src[x * factor..(x + 1) * factor]
                            .iter()
                            .map(|&v| u32::from(v))
                            .sum::<u32>();
                    }
                }
                for (o, a) in out.iter_mut().zip(acc) {
                    *o = u8::try_from((a + area / 2) / area).unwrap_or(u8::MAX);
                }
            });
        Self { w, h, data }
    }

    /// Map a coordinate of a plane downsampled by `factor` back to the
    /// source pixel grid.
    pub fn upscale_coord(v: f64, factor: usize) -> f64 {
        to_f64(factor).mul_add(v + 0.5, -0.5)
    }

    /// Bilinear sample at continuous `(x, y)`, clamped to the plane.
    pub fn sample(&self, x: f64, y: f64) -> f64 {
        let xc = x.clamp(0.0, to_f64(self.w - 1));
        let yc = y.clamp(0.0, to_f64(self.h - 1));
        let (x0, y0) = (floor_index(xc), floor_index(yc));
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let (fx, fy) = (xc - to_f64(x0), yc - to_f64(y0));
        let top = f64::from(self.at(x0, y0)).mul_add(1.0 - fx, f64::from(self.at(x1, y0)) * fx);
        let bot = f64::from(self.at(x0, y1)).mul_add(1.0 - fx, f64::from(self.at(x1, y1)) * fx);
        top.mul_add(1.0 - fy, bot * fy)
    }
}

// ── Histograms ──────────────────────────────────────────────────────────────

/// 256-bin histogram of `values`.
pub fn histogram(values: impl Iterator<Item = u8>) -> [u64; 256] {
    let mut hist = [0u64; 256];
    for v in values {
        hist[usize::from(v)] += 1;
    }
    hist
}

/// Otsu's threshold of a histogram: the `t` maximising the between-class
/// variance of `{v < t}` and `{v ≥ t}`, together with the separability
/// `σ_between² / σ_total²` in `[0, 1]`.  `None` for a constant histogram.
pub fn otsu(hist: &[u64; 256]) -> Option<(u8, f64)> {
    let total: u64 = hist.iter().sum();
    if total == 0 {
        return None;
    }
    let n = to_f64_u64(total);
    let sum: f64 = hist
        .iter()
        .enumerate()
        .map(|(v, &c)| to_f64(v) * to_f64_u64(c))
        .sum();
    let mean = sum / n;
    let var_total: f64 = hist
        .iter()
        .enumerate()
        .map(|(v, &c)| (to_f64(v) - mean).powi(2) * to_f64_u64(c))
        .sum::<f64>()
        / n;
    if var_total <= f64::EPSILON {
        return None;
    }
    let (mut w0, mut sum0) = (0.0f64, 0.0f64);
    let (mut best_t, mut best_var) = (0usize, -1.0f64);
    for t in 1..256 {
        w0 += to_f64_u64(hist[t - 1]);
        sum0 = to_f64(t - 1).mul_add(to_f64_u64(hist[t - 1]), sum0);
        let w1 = n - w0;
        if w0 <= 0.0 || w1 <= 0.0 {
            continue;
        }
        let m0 = sum0 / w0;
        let m1 = (sum - sum0) / w1;
        let var_between = w0 * w1 * (m0 - m1).powi(2) / (n * n);
        if var_between > best_var {
            best_var = var_between;
            best_t = t;
        }
    }
    let t = u8::try_from(best_t).ok()?;
    Some((t, best_var / var_total))
}

/// The value below which a fraction `q ∈ [0, 1]` of the histogram mass lies.
pub fn percentile(hist: &[u64; 256], q: f64) -> u8 {
    let total: u64 = hist.iter().sum();
    let target = q.clamp(0.0, 1.0) * to_f64_u64(total);
    let mut acc = 0u64;
    for (v, &c) in hist.iter().enumerate() {
        acc += c;
        if to_f64_u64(acc) >= target && acc > 0 {
            return u8::try_from(v).unwrap_or(u8::MAX);
        }
    }
    u8::MAX
}

// ── Resampling ──────────────────────────────────────────────────────────────

/// Build an `out_w × out_h` bitmap whose pixel `(x, y)` is the bilinear
/// sample of `src` at `map(x, y)`.
///
/// Samples outside the continuous extent of `src` are white (255 in every
/// channel).  Inside it, the 2×2 footprint is clamped to the edge pixels.
pub fn resample<P, F>(src: &Bitmap<P>, out_w: u32, out_h: u32, map: F) -> Bitmap<P>
where
    P: CameraPixel,
    F: Fn(f64, f64) -> (f64, f64) + Sync,
{
    let mut dst = Bitmap::<P>::new(out_w, out_h, 1, false);
    if out_w == 0 || out_h == 0 {
        return dst;
    }
    let stride = dst.stride;
    let ch = P::BYTES;
    let (sw, sh) = (src.width as usize, src.height as usize);
    if sw == 0 || sh == 0 {
        dst.data_mut().fill(u8::MAX);
        return dst;
    }
    let (max_x, max_y) = (to_f64(sw) - 0.5, to_f64(sh) - 0.5);
    dst.data_mut()
        .par_chunks_mut(stride)
        .enumerate()
        .for_each(|(oy, row)| {
            for (ox, px) in row[..out_w as usize * ch].chunks_exact_mut(ch).enumerate() {
                let (sx, sy) = map(to_f64(ox), to_f64(oy));
                if !(sx >= -0.5 && sy >= -0.5 && sx <= max_x && sy <= max_y) {
                    px.fill(u8::MAX);
                    continue;
                }
                let xc = sx.clamp(0.0, to_f64(sw - 1));
                let yc = sy.clamp(0.0, to_f64(sh - 1));
                let (x0, y0) = (floor_index(xc), floor_index(yc));
                let (x1, y1) = ((x0 + 1).min(sw - 1), (y0 + 1).min(sh - 1));
                let (fx, fy) = (xc - to_f64(x0), yc - to_f64(y0));
                let r0 = src.row_bytes(u32::try_from(y0).unwrap_or(u32::MAX));
                let r1 = src.row_bytes(u32::try_from(y1).unwrap_or(u32::MAX));
                for (c, out) in px.iter_mut().enumerate() {
                    let a = f64::from(r0[x0 * ch + c]);
                    let b = f64::from(r0[x1 * ch + c]);
                    let d = f64::from(r1[x0 * ch + c]);
                    let e = f64::from(r1[x1 * ch + c]);
                    let top = a.mul_add(1.0 - fx, b * fx);
                    let bot = d.mul_add(1.0 - fx, e * fx);
                    *out = clamp_u8(top.mul_add(1.0 - fy, bot * fy));
                }
            }
        });
    dst
}

/// Copy columns `[x0, x1)` of `src` into a new bitmap.
pub fn crop_columns<P: CameraPixel>(src: &Bitmap<P>, x0: u32, x1: u32) -> Bitmap<P> {
    let w = x1.saturating_sub(x0);
    let mut dst = Bitmap::<P>::new(w, src.height, 1, false);
    let (a, b) = (x0 as usize * P::BYTES, x1 as usize * P::BYTES);
    for y in 0..src.height {
        dst.row_bytes_mut(y)[..b - a].copy_from_slice(&src.row_bytes(y)[a..b]);
    }
    dst
}

// ── Numeric conversions ─────────────────────────────────────────────────────

/// `usize` → `f64`.  Every value passed is a pixel coordinate or count
/// bounded by the page area limit (< 2³⁰), which `f64` represents exactly.
#[expect(
    clippy::cast_precision_loss,
    reason = "pixel coordinates and counts are < 2^30, exact in f64"
)]
pub const fn to_f64(v: usize) -> f64 {
    v as f64
}

/// `u64` → `f64` for histogram masses (< 2³⁰, exact).
#[expect(
    clippy::cast_precision_loss,
    reason = "histogram masses are bounded by the page area (< 2^30), exact in f64"
)]
pub const fn to_f64_u64(v: u64) -> f64 {
    v as f64
}

/// `⌊v⌋` as an index; negative and non-finite values map to 0.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is clamped to [0, 2^31] before the cast"
)]
pub const fn floor_index(v: f64) -> usize {
    if v.is_finite() {
        v.floor().clamp(0.0, 2_147_483_648.0) as usize
    } else {
        0
    }
}

/// `round(v)` as an index; negative and non-finite values map to 0.
pub fn round_index(v: f64) -> usize {
    floor_index(v + 0.5)
}

/// Round and saturate to `u8`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is clamped to [0, 255] before the cast"
)]
pub fn clamp_u8(v: f64) -> u8 {
    (v + 0.5).clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::testutil::plane;

    #[test]
    fn downsample_averages_blocks() {
        let p = plane(4, 2, |x, _| if x < 2 { 0 } else { 200 });
        let d = p.downsample(2);
        assert_eq!((d.w, d.h), (2, 1));
        assert_eq!(d.data, vec![0, 200]);
    }

    #[test]
    fn otsu_separates_bimodal() {
        let mut hist = [0u64; 256];
        hist[40] = 500;
        hist[220] = 500;
        let (t, eta) = otsu(&hist).unwrap();
        assert!((41..=220).contains(&t), "threshold {t}");
        assert!(eta > 0.99, "separability {eta}");
    }

    #[test]
    fn otsu_constant_is_none() {
        let mut hist = [0u64; 256];
        hist[128] = 10;
        assert!(otsu(&hist).is_none());
    }

    #[test]
    fn percentile_bounds() {
        let hist = histogram([10u8, 20, 30, 40].into_iter());
        assert_eq!(percentile(&hist, 0.0), 10);
        assert_eq!(percentile(&hist, 0.5), 20);
        assert_eq!(percentile(&hist, 1.0), 40);
    }

    #[test]
    fn resample_identity_preserves_rgb() {
        let mut src = Bitmap::<Rgb8>::new(3, 2, 1, false);
        for (i, b) in src.data_mut().iter_mut().enumerate() {
            *b = u8::try_from(i * 13).unwrap();
        }
        let dst = resample(&src, 3, 2, |x, y| (x, y));
        assert_eq!(dst.data(), src.data());
    }

    #[test]
    fn resample_outside_is_white() {
        let src = Bitmap::<Gray8>::new(2, 2, 1, false);
        let dst = resample(&src, 2, 1, |x, y| (x + 10.0, y));
        assert_eq!(dst.data(), &[255, 255]);
    }
}
