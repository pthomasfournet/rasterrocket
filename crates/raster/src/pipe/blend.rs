//! PDF blend modes (PDF spec §11.3.5).
//!
//! All separable modes operate per-channel with `src` and `dst` in
//! `[0, M]`, `M = S::MAX`, and return a blended value in `[0, M]`.  For
//! CMYK/DeviceN the caller inverts the channel values before calling
//! (`M - v`) and inverts again after, blending the subtractive complement in
//! additive space.
//!
//! Non-separable modes (`Hue`, `Saturation`, `Color`, `Luminosity`) operate on
//! full RGB triples (always in additive space); CMYK callers convert first.
//!
//! The arithmetic is integer, in `i64`, with `M` in place of the 8-bit
//! constant 255 and every threshold a fixed fraction of `M + 1`; no
//! intermediate overflows at 16 bits.

use crate::types::BlendMode;
use color::{Sample, Wide};

/// `M`, the largest sample value, as an `i64`.
fn max<S: Sample>() -> i64 {
    S::MAX.signed()
}

// ── Separable blend primitives ────────────────────────────────────────────────

/// Normal: result is the source (Porter-Duff over handled in the pipe, not here).
#[must_use]
pub(crate) const fn blend_normal<S: Sample>(src: S, _dst: S) -> S {
    src
}

/// Multiply: `div_max(src * dst)`.
#[must_use]
pub(crate) fn blend_multiply<S: Sample>(src: S, dst: S) -> S {
    S::div_max(src.wide() * dst.wide())
}

/// Screen: `src + dst - div_max(src * dst)`, which lies in `[0, M]`.
#[must_use]
pub(crate) fn blend_screen<S: Sample>(src: S, dst: S) -> S {
    let (s, d) = (src.wide(), dst.wide());
    S::from_wide(s + d - S::div_max(s * d).wide())
}

/// Overlay: hard-light of dst over src.
#[must_use]
pub(crate) fn blend_overlay<S: Sample>(src: S, dst: S) -> S {
    let m = max::<S>();
    let (s, d) = (src.signed(), dst.signed());
    if d < (m + 1) / 2 {
        S::div_max(src.wide() * dst.wide() * S::Wide::of(2))
    } else {
        let (s, d) = (m - s, m - d);
        let prod = u64::try_from(2 * s * d).expect("a product of non-negative samples");
        S::from_signed(m - S::div_max(S::Wide::of_u64(prod)).signed())
    }
}

/// Darken: `min(src, dst)`.
#[must_use]
pub(crate) fn blend_darken<S: Sample>(src: S, dst: S) -> S {
    src.min(dst)
}

/// Lighten: `max(src, dst)`.
#[must_use]
pub(crate) fn blend_lighten<S: Sample>(src: S, dst: S) -> S {
    src.max(dst)
}

/// Color dodge: brighten dst to reflect src.
///
/// ISO 32000-2 §11.3.5.2: `0` when `cb == 0`, `1` when `cb ≥ 1 − cs`, and
/// `cb / (1 − cs)` otherwise.  A black backdrop is preserved for every source
/// value, including `src == M`, mirroring the white backdrop of
/// [`blend_color_burn`].
#[must_use]
pub(crate) fn blend_color_dodge<S: Sample>(src: S, dst: S) -> S {
    let m = max::<S>();
    if dst == S::default() {
        S::default()
    } else if src == S::MAX {
        S::MAX
    } else {
        S::from_signed((dst.signed() * m) / (m - src.signed()))
    }
}

/// Color burn: darken dst to reflect src.
///
/// A white backdrop is preserved for every source value, including `src == 0`:
/// the spec's `1 - min(1, (1 - cb)/cs)` evaluates to 1 when `cb == 1`, since the
/// numerator is 0 whatever the source. Returning 0 there would turn a white
/// backdrop black under a zero source.
#[must_use]
pub(crate) fn blend_color_burn<S: Sample>(src: S, dst: S) -> S {
    let m = max::<S>();
    if dst == S::MAX {
        S::MAX
    } else if src == S::default() {
        S::default()
    } else {
        let x = (m - dst.signed()) * m / src.signed();
        if x >= m {
            S::default()
        } else {
            S::from_signed(m - x)
        }
    }
}

/// Hard light: `Overlay(dst, src)` — the src/dst roles are swapped vs Overlay.
#[must_use]
pub(crate) fn blend_hard_light<S: Sample>(src: S, dst: S) -> S {
    // hard_light(s, d) == overlay(d, s)
    blend_overlay(dst, src)
}

/// Soft light (PDF spec §11.3.5.3).
#[must_use]
pub(crate) fn blend_soft_light<S: Sample>(src: S, dst: S) -> S {
    let m = max::<S>();
    let s = src.signed();
    let d = dst.signed();
    let result = if s < (m + 1) / 2 {
        d - (m - 2 * s) * d * (m - d) / (m * m)
    } else {
        let x = if d < (m + 1) / 4 {
            (((16 * d - 12 * m) * d / m + 4 * m) * d) / m
        } else {
            // sqrt(M · d): the square root of d / M, scaled to [0, M].
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_precision_loss,
                reason = "M · d < 2^32 is exact in f64; its square root is at most M"
            )]
            {
                (d as f64 * m as f64).sqrt() as i64
            }
        };
        d + (2 * s - m) * (x - d) / m
    };
    S::from_signed(result)
}

/// Difference: `|src - dst|`.
#[must_use]
pub(crate) fn blend_difference<S: Sample>(src: S, dst: S) -> S {
    S::from_signed((src.signed() - dst.signed()).abs())
}

/// Exclusion: `src + dst - 2 × src × dst / M`.
///
/// The subtrahend reaches `2M` at `src = dst = M`, so it cannot be routed
/// through [`Sample::div_max`], which saturates at `M`. Saturating there would
/// leave `blend_exclusion(M, M)` at `M` (white) where the spec requires 0 (black).
#[must_use]
pub(crate) fn blend_exclusion<S: Sample>(src: S, dst: S) -> S {
    let m = max::<S>();
    let (s, d) = (src.signed(), dst.signed());
    // The rounded quotient of 2·s·d by M, at most 2M.
    let prod = (2 * s * d + m / 2) / m;
    S::from_signed((s + d - prod).max(0))
}

// ── Non-separable helpers (RGB additive space) ────────────────────────────────

/// Luminance of an RGB triple using the PDF weights 0.30/0.59/0.11 (integer rounding).
#[must_use]
const fn get_lum(r: i64, g: i64, b: i64) -> i64 {
    (r * 77 + g * 151 + b * 28 + 0x80) >> 8
}

/// Saturation: max channel − min channel.
#[must_use]
fn get_sat(r: i64, g: i64, b: i64) -> i64 {
    r.max(g).max(b) - r.min(g).min(b)
}

/// Clip an out-of-range RGB triple back into `[0, m]` while preserving luminance.
fn clip_color(m: i64, c: [i64; 3]) -> [i64; 3] {
    let lum = get_lum(c[0], c[1], c[2]);
    let rgb_min = c[0].min(c[1]).min(c[2]);
    let rgb_max = c[0].max(c[1]).max(c[2]);
    if rgb_min < 0 {
        let d = lum - rgb_min;
        c.map(|v| (lum + (v - lum) * lum / d).clamp(0, m))
    } else if rgb_max > m {
        let d = rgb_max - lum;
        c.map(|v| (lum + (v - lum) * (m - lum) / d).clamp(0, m))
    } else {
        c
    }
}

/// Shift the luminance of an RGB triple to `lum` while preserving hue/saturation.
fn set_lum(m: i64, c: [i64; 3], lum: i64) -> [i64; 3] {
    let d = lum - get_lum(c[0], c[1], c[2]);
    clip_color(m, c.map(|v| v + d))
}

/// Scale an RGB triple so its saturation equals `sat`, while preserving hue.
fn set_sat(m: i64, c: [i64; 3], sat: i64) -> [i64; 3] {
    // Sort channels into (min, mid, max) keeping track of which index is which.
    let mut channels = [(c[0], 0usize), (c[1], 1), (c[2], 2)];
    channels.sort_unstable_by_key(|&(v, _)| v);
    let [(lo, slot_lo), (mid, slot_mid), (hi, slot_hi)] = channels;

    let mut out = [0i64; 3];
    if hi > lo {
        out[slot_mid] = ((mid - lo) * sat / (hi - lo)).clamp(0, m);
        out[slot_hi] = sat.clamp(0, m);
    }
    out[slot_lo] = 0;
    out
}

/// The luminance of an RGB triple.
const fn lum_of(c: [i64; 3]) -> i64 {
    get_lum(c[0], c[1], c[2])
}

/// The saturation of an RGB triple.
fn sat_of(c: [i64; 3]) -> i64 {
    get_sat(c[0], c[1], c[2])
}

/// The triple of samples as `i64`.
fn signed3<S: Sample>(v: [S; 3]) -> [i64; 3] {
    v.map(Sample::signed)
}

/// The triple, clamped by [`clip_color`] to `[0, M]`, as samples.
fn samples3<S: Sample>(v: [i64; 3]) -> [S; 3] {
    v.map(S::from_signed)
}

/// Non-separable Hue blend: hue of src, saturation and luminosity of dst.
#[must_use]
pub(crate) fn blend_hue_rgb<S: Sample>(src: [S; 3], dst: [S; 3]) -> [S; 3] {
    let m = max::<S>();
    let d = signed3(dst);
    samples3(set_lum(m, set_sat(m, signed3(src), sat_of(d)), lum_of(d)))
}

/// Non-separable Saturation blend: saturation of src, hue and luminosity of dst.
#[must_use]
pub(crate) fn blend_saturation_rgb<S: Sample>(src: [S; 3], dst: [S; 3]) -> [S; 3] {
    let m = max::<S>();
    let d = signed3(dst);
    samples3(set_lum(m, set_sat(m, d, sat_of(signed3(src))), lum_of(d)))
}

/// Non-separable Color blend: hue and saturation of src, luminosity of dst.
#[must_use]
pub(crate) fn blend_color_rgb<S: Sample>(src: [S; 3], dst: [S; 3]) -> [S; 3] {
    let m = max::<S>();
    samples3(set_lum(m, signed3(src), lum_of(signed3(dst))))
}

/// Non-separable Luminosity blend: luminosity of src, hue and saturation of dst.
#[must_use]
pub(crate) fn blend_luminosity_rgb<S: Sample>(src: [S; 3], dst: [S; 3]) -> [S; 3] {
    let m = max::<S>();
    samples3(set_lum(m, signed3(dst), lum_of(signed3(src))))
}

// ── Dispatch ──────────────────────────────────────────────────────────────────

/// Apply a separable blend mode to each corresponding sample pair from `src` and `dst`.
///
/// For CMYK/DeviceN modes the caller must invert before and after (subtractive complement).
/// `src.len()`, `dst.len()`, and `out.len()` must all be equal.
///
/// # Panics
///
/// Panics if `mode` is a non-separable variant (`Hue`, `Saturation`, `Color`,
/// `Luminosity`).  Use [`apply_nonseparable_rgb`] for those.
pub(crate) fn apply_separable<S: Sample>(mode: BlendMode, src: &[S], dst: &[S], out: &mut [S]) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len(), out.len());
    let f: fn(S, S) -> S = match mode {
        BlendMode::Normal => blend_normal,
        BlendMode::Multiply => blend_multiply,
        BlendMode::Screen => blend_screen,
        BlendMode::Overlay => blend_overlay,
        BlendMode::Darken => blend_darken,
        BlendMode::Lighten => blend_lighten,
        BlendMode::ColorDodge => blend_color_dodge,
        BlendMode::ColorBurn => blend_color_burn,
        BlendMode::HardLight => blend_hard_light,
        BlendMode::SoftLight => blend_soft_light,
        BlendMode::Difference => blend_difference,
        BlendMode::Exclusion => blend_exclusion,
        BlendMode::Hue | BlendMode::Saturation | BlendMode::Color | BlendMode::Luminosity => {
            panic!("apply_separable called with non-separable mode {mode:?}")
        }
    };
    for ((s, d), o) in src.iter().zip(dst).zip(out.iter_mut()) {
        *o = f(*s, *d);
    }
}

/// Apply a non-separable blend mode (`Hue`/`Saturation`/`Color`/`Luminosity`) to an RGB triple.
///
/// `src` and `dst` are in additive space — CMYK callers convert first
/// (`M - channel`).  Only the three components are blended; the caller
/// handles channel `[3]` (K/alpha) separately.
///
/// # Panics
///
/// Panics if `mode` is a separable variant.  Use [`apply_separable`] for those.
#[must_use]
pub(crate) fn apply_nonseparable_rgb<S: Sample>(
    mode: BlendMode,
    src: [S; 3],
    dst: [S; 3],
) -> [S; 3] {
    match mode {
        BlendMode::Hue => blend_hue_rgb(src, dst),
        BlendMode::Saturation => blend_saturation_rgb(src, dst),
        BlendMode::Color => blend_color_rgb(src, dst),
        BlendMode::Luminosity => blend_luminosity_rgb(src, dst),
        _ => panic!("apply_nonseparable_rgb called with separable mode {mode:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference implementations from PDF 32000-1 §11.3.5, in floating point.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to [0, 255] before the cast"
    )]
    fn spec_exclusion(s: u8, d: u8) -> u8 {
        let (cs, cb) = (f64::from(s) / 255.0, f64::from(d) / 255.0);
        (255.0 * (-2.0 * cs).mul_add(cb, cs + cb))
            .round()
            .clamp(0.0, 255.0) as u8
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to [0, 255] before the cast"
    )]
    fn spec_color_burn(s: u8, d: u8) -> u8 {
        let (cs, cb) = (f64::from(s) / 255.0, f64::from(d) / 255.0);
        if cb >= 1.0 {
            return 255;
        }
        if cs <= 0.0 {
            return 0;
        }
        (255.0 * (1.0 - (1.0 - cb).min(cs) / cs))
            .round()
            .clamp(0.0, 255.0) as u8
    }

    /// Exclusion must not saturate its `2·s·d/255` term.
    ///
    /// Routing that term through `div255`, which clamps at 255, capped a quotient
    /// that reaches 510 and inverted the mode across the bright half of the input
    /// space — white over white came out white instead of black.
    #[test]
    fn exclusion_matches_spec_over_the_full_domain() {
        assert_eq!(
            blend_exclusion::<u8>(255, 255),
            0,
            "white over white is black"
        );
        assert_eq!(blend_exclusion::<u8>(0, 0), 0);
        assert_eq!(blend_exclusion::<u8>(255, 0), 255);
        assert_eq!(blend_exclusion::<u8>(0, 255), 255);
        for s in 0..=255u8 {
            for d in 0..=255u8 {
                let got = blend_exclusion::<u8>(s, d);
                let want = spec_exclusion(s, d);
                assert!(
                    got.abs_diff(want) <= 1,
                    "exclusion({s},{d}) = {got}, spec {want}"
                );
            }
        }
    }

    /// A white backdrop survives colour burn at every source value.
    #[test]
    fn color_burn_preserves_a_white_backdrop() {
        for s in 0..=255u8 {
            assert_eq!(
                blend_color_burn::<u8>(s, 255),
                255,
                "color_burn({s},255) must leave a white backdrop white"
            );
        }
        for s in 0..=255u8 {
            for d in 0..=255u8 {
                let got = blend_color_burn::<u8>(s, d);
                let want = spec_color_burn(s, d);
                assert!(
                    got.abs_diff(want) <= 1,
                    "color_burn({s},{d}) = {got}, spec {want}"
                );
            }
        }
    }

    #[test]
    fn multiply_identity() {
        assert_eq!(blend_multiply::<u8>(255, 200), 200);
        assert_eq!(blend_multiply::<u8>(200, 255), 200);
        assert_eq!(blend_multiply::<u8>(0, 200), 0);
        assert_eq!(blend_multiply::<u8>(200, 0), 0);
    }

    #[test]
    fn screen_identity() {
        assert_eq!(blend_screen::<u8>(0, 200), 200);
        assert_eq!(blend_screen::<u8>(200, 0), 200);
        assert_eq!(blend_screen::<u8>(255, 100), 255);
    }

    #[test]
    fn difference_is_abs_diff() {
        for a in (0u8..=255).step_by(13) {
            for b in (0u8..=255).step_by(7) {
                assert_eq!(blend_difference::<u8>(a, b), a.abs_diff(b));
            }
        }
    }

    #[test]
    fn color_dodge_saturates_at_src_255() {
        assert_eq!(blend_color_dodge::<u8>(255, 100), 255);
    }

    /// A black backdrop survives colour dodge at every source value.
    #[test]
    fn color_dodge_preserves_a_black_backdrop() {
        for s in 0..=255u8 {
            assert_eq!(
                blend_color_dodge::<u8>(s, 0),
                0,
                "color_dodge({s},0) must leave a black backdrop black"
            );
        }
    }

    #[test]
    fn color_burn_zeroes_at_src_0() {
        assert_eq!(blend_color_burn::<u8>(0, 100), 0);
    }

    #[test]
    fn hard_light_matches_overlay_swapped() {
        for s in (0u8..=255).step_by(17) {
            for d in (0u8..=255).step_by(11) {
                assert_eq!(
                    blend_hard_light::<u8>(s, d),
                    blend_overlay::<u8>(d, s),
                    "hard_light({s},{d}) should equal overlay({d},{s})"
                );
            }
        }
    }

    #[test]
    fn get_lum_white_is_255() {
        assert_eq!(get_lum(255, 255, 255), 255);
    }

    #[test]
    fn get_lum_black_is_0() {
        assert_eq!(get_lum(0, 0, 0), 0);
    }

    #[test]
    fn get_sat_grey_is_0() {
        assert_eq!(get_sat(128, 128, 128), 0);
    }

    #[test]
    fn luminosity_grey_dst_stays_grey() {
        // Luminosity of red over grey: result should be grey (all channels equal).
        let src = [200u8, 50, 50];
        let dst = [128u8, 128, 128];
        let out = blend_luminosity_rgb::<u8>(src, dst);
        assert_eq!(out[0], out[1]);
        assert_eq!(out[1], out[2]);
    }

    #[test]
    fn apply_separable_multiply_all_channels() {
        let src = [100u8, 200, 50];
        let dst = [255u8, 128, 0];
        let mut out = [0u8; 3];
        apply_separable::<u8>(BlendMode::Multiply, &src, &dst, &mut out);
        assert_eq!(out[0], blend_multiply::<u8>(100, 255));
        assert_eq!(out[1], blend_multiply::<u8>(200, 128));
        assert_eq!(out[2], blend_multiply::<u8>(50, 0));
    }

    #[test]
    #[should_panic(expected = "apply_separable called with non-separable mode")]
    fn apply_separable_panics_on_nonseparable() {
        let mut out = [0u8; 3];
        apply_separable::<u8>(BlendMode::Hue, &[0; 3], &[0; 3], &mut out);
    }

    #[test]
    fn soft_light_midpoint() {
        // For src=128, dst=128: result should be close to 128.
        let r = blend_soft_light::<u8>(128, 128);
        assert!(
            (i32::from(r) - 128).abs() <= 2,
            "soft_light(128,128)={r}, expected ~128"
        );
    }

    #[test]
    fn exclusion_is_screen_minus_extra_mul() {
        // Exclusion(s,d) = s + d - 2×div255(s×d)
        // Screen(s,d)    = s + d - div255(s×d)
        // So exclusion ≤ screen for all inputs.
        for s in (0u8..=255).step_by(23) {
            for d in (0u8..=255).step_by(19) {
                let ex = blend_exclusion::<u8>(s, d);
                let sc = blend_screen::<u8>(s, d);
                assert!(ex <= sc, "exclusion({s},{d})={ex} > screen({s},{d})={sc}");
            }
        }
    }

    #[test]
    fn screen_result_is_always_valid_u8() {
        // Screen(s,d) = s + d - div255(s*d). Mathematically ≤ 255; verify exhaustively.
        for s in 0u8..=255 {
            for d in 0u8..=255 {
                let result = blend_screen::<u8>(s, d);
                // result is u8 so it's always ≤ 255; check it's ≥ max(s,d) isn't true,
                // but it should be ≥ both (Screen is always ≥ each input).
                assert!(
                    result >= s || result >= d,
                    "screen({s},{d})={result} below both inputs"
                );
            }
        }
    }
}
