//! Shared arithmetic primitives used throughout the rasterizer.
//!
//! All compositing math lives here — never copy-pasted into callers.
//!
//! # Functions
//!
//! **Integer blend math**
//! - [`div255`] — division by 255, rounded to nearest
//! - [`lerp_u8`] — linear interpolation between two bytes
//!
//! **Color-space conversion (f64 → u8, normalised PDF values)**
//! (`DeviceCMYK` conversion lives in [`crate::cmyk`].)
//! - [`gray_to_u8`] — normalised grey \[0,1\] → byte
//! - [`rgb_to_bytes`] — normalised RGB \[0,1\] → 3-byte array
//!
//! **Geometry rounding**
//! - [`floor_i32`] — floor toward −∞, returns i32
//! - [`ceil_i32`] — ceil toward +∞, returns i32
//! - [`round_half_up_i32`] — round half-integers toward +∞

use crate::sample::{Sample, Wide};

// ── Integer blend math ────────────────────────────────────────────────────────

/// Division by 255, rounded to nearest, clamped to a byte.
///
/// Computes `(t + (t >> 8)) >> 8` with `t = x + 0x80`, which equals
/// `x / 255` rounded to nearest for every `x` in \[0, 65535\] (an exact
/// half never occurs, 255 being odd).
///
/// # Valid input range
///
/// `x` must be in \[0, 65535\]. In practice `x` is a product `a * b` of two
/// bytes, at most 255 × 255 = 65025.
///
/// # Output range
///
/// Always \[0, 255\]: from `x` = 65153 the nearest integer is 256, which is
/// clamped to 255; no product of two bytes reaches it.
#[inline]
#[must_use]
pub fn div255(x: u32) -> u8 {
    // At most 65535 + 128 + 256 = 65919 before the shift, 257 after it.
    let t = x + 0x80;
    let shifted = (t + (t >> 8)) >> 8;
    // `.min(255)` makes the `as u8` cast lossless.
    shifted.min(255) as u8
}

/// Linear interpolation between two bytes at weight `t` on the scale `[0, 256]`.
///
/// Computes `(a · (256 − t) + b · t + 128) >> 8`: the weights sum to 256, so
/// the result is `a + (b − a) · t / 256` rounded half up, and every output lies
/// between `a` and `b`.
///
/// # Valid input range
///
/// `t` must be in \[0, 256\]; a larger `t` is a caller bug and is caught by a
/// `debug_assert!`.
///
/// # Endpoints
///
/// `t = 0` returns `a` and `t = 256` returns `b` exactly; `t = 128` returns
/// the midpoint `(a + b + 1) / 2`.
#[inline]
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    reason = "a convex combination of two bytes, shifted down by 8 bits, is at most 255"
)]
pub fn lerp_u8(a: u8, b: u8, t: u32) -> u8 {
    debug_assert!(t <= 256, "lerp_u8: t={t} out of range [0, 256]");
    ((u32::from(a) * (256 - t) + u32::from(b) * t + 128) >> 8) as u8
}

// ── Color space conversion ────────────────────────────────────────────────────

/// Full-range (JFIF) Y′CbCr → RGB, ITU-R BT.601 coefficients.
///
/// `R = Y + 1.402·Cr′`, `G = Y − 0.344136·Cb′ − 0.714136·Cr′`,
/// `B = Y + 1.772·Cb′`, with `Cb′ = Cb − 128` and `Cr′ = Cr − 128`.  The
/// coefficients are 16.16 fixed point, rounded half-up, and each channel is
/// clamped to \[0, 255\].
#[inline]
#[must_use]
pub fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
    const HALF: i32 = 1 << 15;
    let y = i32::from(y);
    let cb = i32::from(cb) - 128;
    let cr = i32::from(cr) - 128;
    let r = y + ((91_881 * cr + HALF) >> 16);
    let g = y + ((-22_554 * cb - 46_802 * cr + HALF) >> 16);
    let b = y + ((116_130 * cb + HALF) >> 16);
    #[expect(
        clippy::cast_sign_loss,
        reason = "clamped to [0, 255] before narrowing"
    )]
    let clamp = |v: i32| v.clamp(0, 255) as u8;
    (clamp(r), clamp(g), clamp(b))
}

// ── f64 → u8 conversions ─────────────────────────────────────────────────────

/// Convert a normalised PDF value \[0.0, 1.0\] to a `u8` byte.
///
/// Clamps then rounds. `f64::clamp(NaN, 0.0, 1.0)` returns `NaN` (clamp does
/// not sanitise NaN); the subsequent `as u8` cast then saturates `NaN` to 0
/// per Rust's float-to-int rules, so NaN inputs map to 0.
/// Used for PDF colour operators where channel components are normalised floats.
#[inline]
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "value is clamped to [0, 1] and scaled to [0.0, 255.0]; round() output fits u8"
)]
pub fn gray_to_u8(v: f64) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Convert three normalised PDF RGB components to `[r, g, b]` bytes.
///
/// Each channel is clamped to \[0.0, 1.0\] independently.
#[inline]
#[must_use]
pub fn rgb_to_bytes(r: f64, g: f64, b: f64) -> [u8; 3] {
    [gray_to_u8(r), gray_to_u8(g), gray_to_u8(b)]
}

/// The weights `[R, G, B]` of the `DeviceRGB` → `DeviceGray` conversion,
/// `0.30 R + 0.59 G + 0.11 B` (ISO 32000-2 §10.4.2.2), in 1.15 fixed
/// point; they sum to exactly `1 << GRAY_SHIFT`, so a neutral maps to
/// itself.
pub const GRAY_WEIGHTS: [u32; 3] = [9830, 19333, 3605];
/// The rounding bias of [`GRAY_WEIGHTS`]: one half in 1.15 fixed point.
pub const GRAY_BIAS: u32 = 1 << (GRAY_SHIFT - 1);
/// The fraction bits of [`GRAY_WEIGHTS`].
pub const GRAY_SHIFT: u32 = 15;

/// The gray of an RGB pixel at the depth of `S` by [`GRAY_WEIGHTS`],
/// rounded half up: `(9830 R + 19333 G + 3605 B + 16384) >> 15`.
#[inline]
#[must_use]
pub fn gray_at<S: Sample>(rgb: [S; 3]) -> S {
    let sum: i64 = rgb
        .iter()
        .zip(GRAY_WEIGHTS)
        .map(|(v, w)| v.signed() * i64::from(w))
        .sum();
    S::from_signed((sum + i64::from(GRAY_BIAS)) >> GRAY_SHIFT)
}

/// The luminosity of an 8-bit RGB pixel, `0.30 R + 0.59 G + 0.11 B`
/// rounded to the nearest level (PDF §11.3.5.3).
#[inline]
#[must_use]
pub fn luminosity(r: u8, g: u8, b: u8) -> u8 {
    luminosity_at(r, g, b)
}

/// [`luminosity`] at the depth of `S`: `(30 R + 59 G + 11 B + 50) / 100`,
/// formed in [`Sample::Wide`].
#[inline]
#[must_use]
pub fn luminosity_at<S: Sample>(r: S, g: S, b: S) -> S {
    let w = <S::Wide as Wide>::of;
    S::from_wide((w(30) * r.wide() + w(59) * g.wide() + w(11) * b.wide() + w(50)) / w(100))
}

// ── Geometry rounding ─────────────────────────────────────────────────────────

/// Saturating cast of an integer-valued `f64` to `i32`.
///
/// Non-finite inputs map to `i32::MAX` for `+∞` and `i32::MIN` for `-∞` / NaN.
/// Finite values outside the `i32` range saturate at the nearest endpoint.
#[inline]
fn saturate_f64_to_i32(x: f64) -> i32 {
    if !x.is_finite() {
        return if x == f64::INFINITY {
            i32::MAX
        } else {
            i32::MIN
        };
    }
    // x is finite: casting to i64 is well-defined for any finite f64 whose
    // magnitude fits in i64 (which covers all practical PDF coordinates);
    // try_from saturates the rare case of very large floats.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "f64 → i64 cast; try_from on the next line saturates out-of-range values"
    )]
    let v = x as i64;
    i32::try_from(v).unwrap_or(if v > 0 { i32::MAX } else { i32::MIN })
}

/// Floor toward −∞, returning `i32`.
///
/// # Valid input range
///
/// Any `f64`. For PDF coordinates, values are always finite and well within
/// i32 range.
///
/// # Edge cases
///
/// - Finite values outside \[`i32::MIN`, `i32::MAX`\]: saturate to
///   `i32::MIN` or `i32::MAX` respectively.
/// - `NaN` or ±infinity: `is_finite()` check returns `i32::MIN` for any
///   non-finite input (conservatively safe — callers must not rely on this
///   specific value for non-finite inputs).
///
/// # Panic
///
/// Never panics.
#[inline]
#[must_use]
pub fn floor_i32(x: f64) -> i32 {
    saturate_f64_to_i32(x.floor())
}

/// Ceil toward +∞, returning `i32`.
///
/// # Valid input range
///
/// Any `f64`. See [`floor_i32`] for edge-case behaviour.
///
/// # Edge cases
///
/// Same as [`floor_i32`]: non-finite inputs return `i32::MAX` (for +∞) or
/// `i32::MIN` (for −∞ and NaN).
///
/// # Panic
///
/// Never panics.
#[inline]
#[must_use]
pub fn ceil_i32(x: f64) -> i32 {
    saturate_f64_to_i32(x.ceil())
}

/// Round to the nearest integer, half-integers toward +∞, returning `i32`.
///
/// - 0.5 rounds to 1 (toward +∞).
/// - −0.5 rounds to 0 (toward +∞, i.e. not away from zero).
///
/// The fraction `x − floor(x)` is exact in `f64`, so a value just below a
/// half-integer rounds down: `floor(x + 0.5)` would round the sum itself
/// and take 0.49999999999999994 to 1.
///
/// # Valid input range
///
/// Any `f64`. See [`floor_i32`] for edge-case behaviour on non-finite inputs.
///
/// # Panic
///
/// Never panics.
#[inline]
#[must_use]
pub fn round_half_up_i32(x: f64) -> i32 {
    let below = x.floor();
    // Non-finite `x` gives a NaN fraction and keeps `below`.
    floor_i32(if x - below >= 0.5 { below + 1.0 } else { below })
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gray_weights_sum_to_one_and_keep_neutrals() {
        assert_eq!(GRAY_WEIGHTS.iter().sum::<u32>(), 1 << GRAY_SHIFT);
        for v in 0..=255u8 {
            assert_eq!(gray_at([v; 3]), v);
        }
        assert_eq!(gray_at([65_535u16, 0, 0]), 19_660);
        assert_eq!(gray_at([255u8, 0, 0]), 76);
        assert_eq!(gray_at([0u8, 60, 10]), 36);
    }

    #[test]
    fn ycbcr_neutral_axis_is_gray() {
        for y in 0..=255u8 {
            assert_eq!(ycbcr_to_rgb(y, 128, 128), (y, y, y));
        }
    }

    #[test]
    fn ycbcr_to_rgb_within_one_of_exact_formula() {
        let exact = |v: f64| v.round().clamp(0.0, 255.0);
        for y in 0..=255u8 {
            for cb in 0..=255u8 {
                for cr in 0..=255u8 {
                    let (yf, cbf, crf) =
                        (f64::from(y), f64::from(cb) - 128.0, f64::from(cr) - 128.0);
                    let want = [
                        exact(1.402f64.mul_add(crf, yf)),
                        exact((-0.344_136f64).mul_add(cbf, (-0.714_136f64).mul_add(crf, yf))),
                        exact(1.772f64.mul_add(cbf, yf)),
                    ];
                    let (r, g, b) = ycbcr_to_rgb(y, cb, cr);
                    for (got, want) in [r, g, b].into_iter().zip(want) {
                        assert!(
                            (f64::from(got) - want).abs() <= 1.0,
                            "ycbcr_to_rgb({y}, {cb}, {cr}) = {:?}, exact {want:?}",
                            (r, g, b)
                        );
                    }
                }
            }
        }
    }

    /// `div255` is `x / 255` rounded to nearest, clamped to 255, over its
    /// whole input range; 173 × 241 / 255 = 163.502 rounds up.
    #[test]
    fn div255_rounds_to_nearest() {
        for x in 0u32..=65535 {
            let nearest = ((2 * x + 255) / 510).min(255);
            assert_eq!(u32::from(div255(x)), nearest, "div255({x})");
        }
        assert_eq!(div255(173 * 241), 164);
    }

    /// Endpoints are exact and every interpolant lies between them.
    #[test]
    fn lerp_is_exact_at_the_endpoints_and_bounded() {
        for a in 0u8..=255 {
            for b in 0u8..=255 {
                assert_eq!(lerp_u8(a, b, 0), a, "lerp_u8({a}, {b}, 0)");
                assert_eq!(lerp_u8(a, b, 256), b, "lerp_u8({a}, {b}, 256)");
                for t in [1, 64, 128, 200, 255] {
                    let v = lerp_u8(a, b, t);
                    assert!(
                        a.min(b) <= v && v <= a.max(b),
                        "lerp_u8({a}, {b}, {t}) = {v} outside [{a}, {b}]"
                    );
                }
            }
        }
    }

    /// Interpolating between equal bytes returns that byte at every weight.
    #[test]
    fn lerp_between_equal_bytes_is_constant() {
        for a in 0u8..=255 {
            for t in 0..=256 {
                assert_eq!(lerp_u8(a, a, t), a, "lerp_u8({a}, {a}, {t})");
            }
        }
    }

    #[test]
    fn lerp_midpoint_rounds_half_up() {
        assert_eq!(lerp_u8(0, 255, 128), 128);
        assert_eq!(lerp_u8(10, 11, 128), 11);
        assert_eq!(lerp_u8(200, 100, 128), 150);
    }

    #[test]
    fn floor_ceil_round() {
        let cases = [
            (0.0f64, 0, 0, 0),
            (0.5, 0, 1, 1),
            (0.9, 0, 1, 1),
            (1.0, 1, 1, 1),
            (-0.1, -1, 0, 0),
            (-0.5, -1, 0, 0),
            (-0.6, -1, 0, -1),
            (-1.0, -1, -1, -1),
        ];
        for (x, fl, ce, ro) in cases {
            assert_eq!(floor_i32(x), fl, "floor({x})");
            assert_eq!(ceil_i32(x), ce, "ceil({x})");
            assert_eq!(round_half_up_i32(x), ro, "round({x})");
        }
    }

    /// Half-integer tie-breaking: 0.5 → 1, −0.5 → 0 (toward +∞).
    #[test]
    fn round_half_up_half_integers() {
        assert_eq!(round_half_up_i32(0.5), 1, "0.5 rounds toward +inf");
        assert_eq!(
            round_half_up_i32(-0.5),
            0,
            "-0.5 rounds toward +inf (i.e. 0)"
        );
        assert_eq!(round_half_up_i32(1.5), 2);
        assert_eq!(round_half_up_i32(-1.5), -1);
    }

    /// A value one ulp below a half-integer rounds down, also where adding
    /// one half would round up to the next integer.
    #[test]
    fn round_half_up_just_below_a_half() {
        let below_half = 0.5f64.next_down();
        assert_eq!(round_half_up_i32(below_half), 0);
        assert_eq!(round_half_up_i32((-0.5f64).next_down()), -1);
        assert_eq!(round_half_up_i32(2.5f64.next_down()), 2);
    }

    /// Non-finite inputs must not invoke UB and must return a defined sentinel.
    #[test]
    fn floor_ceil_round_non_finite() {
        // +∞
        assert_eq!(floor_i32(f64::INFINITY), i32::MAX);
        assert_eq!(ceil_i32(f64::INFINITY), i32::MAX);
        assert_eq!(round_half_up_i32(f64::INFINITY), i32::MAX);
        // −∞
        assert_eq!(floor_i32(f64::NEG_INFINITY), i32::MIN);
        assert_eq!(ceil_i32(f64::NEG_INFINITY), i32::MIN);
        assert_eq!(round_half_up_i32(f64::NEG_INFINITY), i32::MIN);
        // NaN — treated as non-positive (returns i32::MIN).
        assert_eq!(floor_i32(f64::NAN), i32::MIN);
        assert_eq!(ceil_i32(f64::NAN), i32::MIN);
        assert_eq!(round_half_up_i32(f64::NAN), i32::MIN);
    }

    // ── gray_to_u8 ────────────────────────────────────────────────────────────

    #[test]
    fn gray_extremes() {
        assert_eq!(gray_to_u8(0.0), 0);
        assert_eq!(gray_to_u8(1.0), 255);
    }

    #[test]
    fn gray_clamped() {
        assert_eq!(gray_to_u8(-1.0), 0);
        assert_eq!(gray_to_u8(2.0), 255);
    }

    /// NaN must map to 0 — `f64::clamp` returns NaN unchanged, then Rust's
    /// float-to-int saturation maps NaN → 0.
    #[test]
    fn gray_nan_is_zero() {
        assert_eq!(gray_to_u8(f64::NAN), 0);
    }
}
