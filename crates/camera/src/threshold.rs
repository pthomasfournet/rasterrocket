//! Binarisation of 8-bit luma: a fixed threshold, or Sauvola's local
//! threshold from the mean and deviation of a window around each pixel.
//!
//! # Sauvola
//!
//! A pixel of value `v` is black when `v < t`, with
//!
//! ```text
//! t = m · (1 + k · (s / R − 1)),    R = 128,
//! ```
//!
//! `m` and `s` the mean and standard deviation of the samples in the
//! square window centred on the pixel, truncated at the image edges and
//! taken over the samples it actually covers.  On 8-bit samples `s ≤ 127.5
//! < R`, so `t ≤ m`, and `t` falls as `k` rises: a larger `k` never adds
//! black pixels.  Samples below [`BLACK_FLOOR`] are black whatever the
//! window, which keeps the interior of a solid area, where `s = 0` and
//! `t = m (1 − k)`, from turning white.
//!
//! The window sums come from `u64` integral images of the value and of its
//! square, held as a ring of rows.  The comparison is rearranged into
//! integers (see [`is_black`]), so the result is exact and identical on
//! every platform.
//!
//! # Limits
//!
//! The method assumes dark ink on light paper: a reversed-out page must be
//! inverted first.  A halftoned photograph is thresholded dot by dot.

use crate::error::CameraError;
use crate::image::{GrayPage, GrayView};

/// The divisor `R` of the standard deviation: the largest deviation of
/// 8-bit samples, rounded up.
const DEVIATION_RANGE: u128 = 128;
/// Samples below this luma are black under every method's window.
const BLACK_FLOOR: u8 = 32;
/// Largest half-width of a Sauvola window; wider windows are clamped, so a
/// window covers at most `2047²` samples.
const MAX_HALF_WINDOW: usize = 1023;

/// How [`binarize`] chooses black.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Threshold {
    /// Samples below the value are black.
    Fixed(u8),
    /// Sauvola's local threshold.
    Sauvola {
        /// Side of the square window, in pixels; an even side is rounded up
        /// to the next odd one so the window is centred, and a side above
        /// 2047 is clamped to 2047.
        window_px: u32,
        /// The sensitivity `k`, in thousandths: higher values make less
        /// black.
        k_per_mille: u16,
    },
}

impl Threshold {
    /// Sauvola's threshold for a page rendered at `dpi`: a window of
    /// 0.33 inch (at least 1 px) and `k = 0.34`.
    #[must_use]
    pub const fn sauvola_at(dpi: u32) -> Self {
        let window = dpi / 100 * 33 + dpi % 100 * 33 / 100;
        Self::Sauvola {
            window_px: if window == 0 { 1 } else { window },
            k_per_mille: 340,
        }
    }
}

/// Binarise the 8-bit luma `img` to black (0) and white (255) by `method`.
///
/// Never declines: every image has a binarisation.
///
/// # Errors
///
/// [`CameraError`] when `img` is malformed, or [`CameraError::ZeroSize`]
/// for a Sauvola window of 0 px.
pub fn binarize(img: GrayView<'_>, method: Threshold) -> Result<GrayPage, CameraError> {
    img.validate()?;
    match method {
        Threshold::Fixed(t) => Ok(map_samples(img, |v| v < t)),
        Threshold::Sauvola {
            window_px,
            k_per_mille,
        } => {
            if window_px == 0 {
                return Err(CameraError::ZeroSize {
                    parameter: "Sauvola window",
                });
            }
            let half = usize::try_from(window_px / 2)
                .unwrap_or(usize::MAX)
                .min(MAX_HALF_WINDOW);
            Ok(sauvola(img, half, k_per_mille))
        }
    }
}

/// The page with 0 where `black(v)` and 255 elsewhere.
fn map_samples(img: GrayView<'_>, black: impl Fn(u8) -> bool) -> GrayPage {
    let mut data = Vec::with_capacity(img.width * img.height);
    for y in 0..img.height {
        data.extend(img.row(y).iter().map(|&v| if black(v) { 0 } else { 255 }));
    }
    GrayPage {
        width: img.width,
        height: img.height,
        data,
    }
}

/// Window sums of one pixel: sample count, sum of values, sum of squares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Window {
    n: u64,
    sum: u64,
    squares: u64,
}

/// Rows of the integral images of value and squared value: row `r` holds,
/// for every `x ≤ width`, the sums over rows `[0, r)` and columns `[0, x)`.
/// Only the last `slots` rows are kept.
struct IntegralRows<'a> {
    img: GrayView<'a>,
    stride: usize,
    slots: usize,
    sums: Vec<u64>,
    squares: Vec<u64>,
    /// Number of rows computed so far.
    rows: usize,
}

impl<'a> IntegralRows<'a> {
    fn new(img: GrayView<'a>, slots: usize) -> Self {
        let stride = img.width + 1;
        // Row 0 is all zeros.
        Self {
            img,
            stride,
            slots,
            sums: vec![0; stride * slots],
            squares: vec![0; stride * slots],
            rows: 1,
        }
    }

    /// Compute rows up to and including `r`.
    fn advance_to(&mut self, r: usize) {
        while self.rows <= r {
            let src = self.img.row(self.rows - 1);
            let prev = (self.rows - 1) % self.slots * self.stride;
            let cur = self.rows % self.slots * self.stride;
            let (mut run, mut run_sq) = (0u64, 0u64);
            self.sums[cur] = 0;
            self.squares[cur] = 0;
            for (x, &v) in src.iter().enumerate() {
                let v = u64::from(v);
                // A row sums to at most 255² · 2³⁰ < 2⁶⁴.
                run += v;
                run_sq += v * v;
                // Wrapping: a window sum is a difference of integral
                // entries and exact modulo 2⁶⁴; every window sum is below
                // 255² · 2047² < 2⁶⁴, so the difference is the true sum.
                self.sums[cur + x + 1] = self.sums[prev + x + 1].wrapping_add(run);
                self.squares[cur + x + 1] = self.squares[prev + x + 1].wrapping_add(run_sq);
            }
            self.rows += 1;
        }
    }

    /// Sums over rows `[y0, y1)` and columns `[x0, x1)`.  Caller guarantees
    /// both rows are computed and still held.
    fn window(&self, (y0, y1): (usize, usize), (x0, x1): (usize, usize)) -> Window {
        let (a, b) = (y0 % self.slots * self.stride, y1 % self.slots * self.stride);
        let rect = |t: &[u64]| {
            t[b + x1]
                .wrapping_sub(t[b + x0])
                .wrapping_sub(t[a + x1])
                .wrapping_add(t[a + x0])
        };
        let area = (y1 - y0) * (x1 - x0);
        Window {
            n: u64::try_from(area).unwrap_or(u64::MAX),
            sum: rect(&self.sums),
            squares: rect(&self.squares),
        }
    }
}

/// Sauvola binarisation with windows of half-width `half`
/// (`half ≤ MAX_HALF_WINDOW`).
fn sauvola(img: GrayView<'_>, half: usize, k_per_mille: u16) -> GrayPage {
    let (w, h) = (img.width, img.height);
    // A window spans at most 2·half + 1 rows, so its two integral rows lie
    // within 2·half + 2 consecutive rows.
    let slots = (2 * half + 2).min(h + 1);
    let mut ring = IntegralRows::new(img, slots);
    let mut data = Vec::with_capacity(w * h);
    for y in 0..h {
        let rows = (y.saturating_sub(half), (y + half + 1).min(h));
        ring.advance_to(rows.1);
        for (x, &v) in img.row(y).iter().enumerate() {
            let cols = (x.saturating_sub(half), (x + half + 1).min(w));
            let black = v < BLACK_FLOOR || is_black(v, ring.window(rows, cols), k_per_mille);
            data.push(if black { 0 } else { 255 });
        }
    }
    GrayPage {
        width: w,
        height: h,
        data,
    }
}

/// Whether `v < m · (1 + k · (s / R − 1))` for the window `win`, with
/// `k = k_per_mille / 1000`, decided exactly in integers.
///
/// With `n`, `S1`, `S2` the window's count, sum and sum of squares,
/// `m = S1 / n` and `s = √D / n` for `D = n · S2 − S1² ≥ 0`.  Multiplying
/// through by `1000 · R · n²` gives
///
/// ```text
/// R · n · A < K · S1 · √D,    A = 1000 · n · v − (1000 − K) · S1,
/// ```
///
/// true when `A < 0`, and otherwise equivalent to
/// `(R · n · A)² < (K · S1)² · D`.
///
/// Bounds, for `n ≤ 2047² < 2²²` and 8-bit samples (`S1 < 2³⁰`,
/// `S2 < 2³⁸`): `|A| < 2⁴⁷`, so `R · n · A < 2⁷⁶`; `K · S1 < 2⁴⁶`, so
/// `(K · S1)² < 2⁹²`; `D ≤ n · S2 < 2⁶⁰`.  Each factor fits `u128`, and
/// both products are compared at 256 bits ([`mul_wide`]).
fn is_black(v: u8, win: Window, k_per_mille: u16) -> bool {
    let (count, sum, squares) = (
        i128::from(win.n),
        i128::from(win.sum),
        i128::from(win.squares),
    );
    let sensitivity = i128::from(k_per_mille);
    let excess = 1000 * count * i128::from(v) - (1000 - sensitivity) * sum;
    if excess < 0 {
        return true;
    }
    let spread = (count * squares - sum.pow(2)).max(0);
    let (Ok(excess), Ok(count), Ok(sum), Ok(spread)) = (
        u128::try_from(excess),
        u128::try_from(count),
        u128::try_from(sum),
        u128::try_from(spread),
    ) else {
        return false;
    };
    let lhs = DEVIATION_RANGE * count * excess;
    let rhs = u128::from(k_per_mille) * sum;
    mul_wide(lhs, lhs) < mul_wide(rhs * rhs, spread)
}

/// The full 256-bit product `a · b` as `(high, low)` halves; tuples compare
/// as the products do.
const fn mul_wide(a: u128, b: u128) -> (u128, u128) {
    const LOW: u128 = u64::MAX as u128;
    let (a1, a0) = (a >> 64, a & LOW);
    let (b1, b0) = (b >> 64, b & LOW);
    let (p00, p01, p10, p11) = (a0 * b0, a0 * b1, a1 * b0, a1 * b1);
    // Each partial product is below 2¹²⁸; `mid` sums three terms below 2⁶⁴.
    let mid = (p00 >> 64) + (p01 & LOW) + (p10 & LOW);
    let low = (p00 & LOW) | (mid << 64);
    let high = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
    (high, low)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{digest, plane, ramp_lit};

    /// Sauvola by direct summation over every window.
    fn naive(img: &GrayPage, half: usize, k: u16) -> GrayPage {
        let (w, h) = (img.width, img.height);
        plane(w, h, |x, y| {
            let sample = img.data[y * w + x];
            let mut win = Window {
                n: 0,
                sum: 0,
                squares: 0,
            };
            for yy in y.saturating_sub(half)..(y + half + 1).min(h) {
                for xx in x.saturating_sub(half)..(x + half + 1).min(w) {
                    let s = u64::from(img.data[yy * w + xx]);
                    win.n += 1;
                    win.sum += s;
                    win.squares += s * s;
                }
            }
            if sample < BLACK_FLOOR || is_black(sample, win, k) {
                0
            } else {
                255
            }
        })
    }

    fn sauvola_of(img: &GrayPage, window_px: u32, k_per_mille: u16) -> GrayPage {
        binarize(
            img.view(),
            Threshold::Sauvola {
                window_px,
                k_per_mille,
            },
        )
        .expect("valid")
    }

    /// Deterministic noise in `-2..=2`.
    fn noise(x: usize, y: usize) -> i32 {
        let h =
            (x.wrapping_mul(0x9E37_79B9) ^ y.wrapping_mul(0x85EB_CA6B)).wrapping_mul(0xC2B2_AE35);
        i32::try_from((h >> 7) % 5).unwrap_or(0) - 2
    }

    #[test]
    fn integral_sums_match_direct_summation() {
        let img = plane(41, 29, |x, y| {
            u8::try_from((x * 37 + y * 91 + x * y * 13) % 256).unwrap_or(0)
        });
        for (half, k) in [(0, 340), (1, 340), (3, 0), (7, 340), (20, 1000), (40, 2000)] {
            let window = u32::try_from(2 * half + 1).unwrap_or(1);
            assert_eq!(
                sauvola_of(&img, window, k),
                naive(&img, half, k),
                "half {half}"
            );
        }
    }

    #[test]
    fn ramp_lit_page_binarises_to_its_ink() {
        let (lit, ink) = ramp_lit(1200, 1000);
        let black = |p: &GrayPage| p.data.iter().map(|&v| v == 0).collect::<Vec<_>>();
        let fixed = binarize(lit.view(), Threshold::Fixed(128)).expect("valid");
        assert_ne!(black(&fixed), ink);
        let out = binarize(lit.view(), Threshold::sauvola_at(256)).expect("valid");
        assert!(out.data.iter().all(|&v| v == 0 || v == 255));
        assert_eq!(black(&out), ink);
        let got = digest(&[out, fixed]);
        assert_eq!(got, GOLDEN, "{got:#x}");
    }

    const GOLDEN: u64 = 0x48e3_2282_b928_d775;

    #[test]
    fn noisy_blank_page_is_white() {
        let img = plane(300, 200, |x, y| {
            u8::try_from(230 + noise(x, y)).unwrap_or(0)
        });
        let out = binarize(img.view(), Threshold::sauvola_at(150)).expect("valid");
        assert!(out.data.iter().all(|&v| v == 255));
    }

    #[test]
    fn wide_black_square_stays_solid() {
        let Threshold::Sauvola { window_px, .. } = Threshold::sauvola_at(150) else {
            unreachable!("sauvola_at builds a Sauvola threshold");
        };
        let side = 3 * usize::try_from(window_px).unwrap_or(0);
        let (x0, y0) = (40, 30);
        let img = plane(side + 80, side + 60, |x, y| {
            let inside = (x0..x0 + side).contains(&x) && (y0..y0 + side).contains(&y);
            if inside { 0 } else { 230 }
        });
        let out = binarize(img.view(), Threshold::sauvola_at(150)).expect("valid");
        assert_eq!(
            out.data,
            img.data
                .iter()
                .map(|&v| if v == 0 { 0 } else { 255 })
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn fixed_threshold_splits_at_its_value() {
        let img = plane(3, 1, |x, _| [0, 127, 128][x]);
        let out = binarize(img.view(), Threshold::Fixed(128)).expect("valid");
        assert_eq!(out.data, vec![0, 0, 255]);
    }

    #[test]
    fn zero_window_is_an_error() {
        let img = plane(3, 3, |_, _| 0);
        let zero = Threshold::Sauvola {
            window_px: 0,
            k_per_mille: 340,
        };
        assert!(matches!(
            binarize(img.view(), zero),
            Err(CameraError::ZeroSize { .. })
        ));
    }

    #[test]
    fn resolution_sets_the_window() {
        assert_eq!(
            Threshold::sauvola_at(300),
            Threshold::Sauvola {
                window_px: 99,
                k_per_mille: 340
            }
        );
        assert!(matches!(
            Threshold::sauvola_at(0),
            Threshold::Sauvola { window_px: 1, .. }
        ));
        assert!(matches!(
            Threshold::sauvola_at(u32::MAX),
            Threshold::Sauvola {
                window_px: 1_417_339_207,
                ..
            }
        ));
    }

    /// The integer rule agrees with the formula evaluated in `f64` wherever
    /// the sample is not within rounding of the threshold.
    #[test]
    fn integer_rule_matches_the_formula() {
        let mut checked = 0;
        for n in [1u64, 2, 9, 49, 361] {
            for seed in 0..12u64 {
                let samples: Vec<u64> = (0..n)
                    .map(|i| (i * 97 + seed * 31 + i * i * seed) % 256)
                    .collect();
                let win = Window {
                    n,
                    sum: samples.iter().sum(),
                    squares: samples.iter().map(|s| s * s).sum(),
                };
                let nf = f64::from(u32::try_from(n).unwrap_or(0));
                let mean = f64::from(u32::try_from(win.sum).unwrap_or(0)) / nf;
                let var = mean.mul_add(
                    -mean,
                    f64::from(u32::try_from(win.squares).unwrap_or(0)) / nf,
                );
                let dev = var.max(0.0).sqrt();
                for k in [0u16, 340, 1000, 2500] {
                    let kf = f64::from(k) / 1000.0;
                    let t = mean * kf.mul_add(dev / 128.0 - 1.0, 1.0);
                    for v in (0..=255u8).step_by(3) {
                        if (f64::from(v) - t).abs() > 1e-6 {
                            assert_eq!(
                                is_black(v, win, k),
                                f64::from(v) < t,
                                "n {n} seed {seed} k {k} v {v}"
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
        assert!(checked > 10_000, "{checked}");
    }

    #[test]
    fn wide_products_compare_exactly() {
        assert_eq!(mul_wide(u128::MAX, u128::MAX), (u128::MAX - 1, 1));
        assert_eq!(mul_wide(1 << 64, 1 << 64), (1, 0));
        assert!(mul_wide(3, 1 << 127) > mul_wide(1 << 127, 2));
    }
}
