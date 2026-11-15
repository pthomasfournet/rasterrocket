//! The working space as the source profile of a conversion to an output
//! profile, with the intent's adjustments folded into it.
//!
//! The working space is sRGB, a matrix/TRC profile: a device value maps to
//! the D50 profile connection space (PCS) as `M · lin(v)`, `lin` the
//! channel's tone curve and `M` the matrix whose columns are the
//! colorants; device black is the PCS origin and device white the D50
//! white `W = M · (1, 1, 1)`.  Two adjustments of the PCS are affine maps
//! of that form and stay within it:
//!
//! - ICC-absolute colorimetry (ICC.1:2022 §6.3.2.2) scales each PCS
//!   component by the ratio of the source media white, D50 for the working
//!   space, to the destination's media white `wtpt`: `diag(s) · M · lin(v)`,
//!   the matrix/TRC profile whose colorants are `diag(s) · M`.
//! - Black-point compensation (ISO 18619) maps the source black, the
//!   origin, to the neutral destination black `k · W` and keeps white:
//!   `(1 − k) · X + k · W = M · ((1 − k) · lin(v) + k)`, the profile whose
//!   tone curves are lifted to `(1 − k) · lin + k`.
//!
//! The conversion then evaluates the destination's own PCS → device tables
//! unchanged.

use moxcms::{
    ColorProfile, DataColorSpace, Layout, ProfileVersion, RenderingIntent, ToneReprCurve,
    TransformOptions, Xyzd,
};

use crate::pcs::D50;

/// The luminance `Y` of the perceptual reference medium's black
/// (ICC.1:2022 §6.3.4.4), the black of a version 4 profile's perceptual
/// and saturation tables.
const PERCEPTUAL_BLACK_Y: f64 = 0.003_479;

/// The largest luminance accepted as a destination black: `L* = 50`.
const MAX_BLACK_Y: f64 = 0.184_186;

/// The sRGB tone curve as an ICC parametric curve of type 3,
/// `(g, a, b, c, d)` (ICC.1:2022 Table 68).
const SRGB_CURVE: [f64; 5] = [2.4, 1.0 / 1.055, 0.055 / 1.055, 1.0 / 12.92, 0.040_45];

/// The source profile of a conversion to `dest` under `intent`: the
/// working space, scaled to the destination's media white under absolute
/// colorimetry, its black lifted to `black` (the luminance of the neutral
/// destination black) when given.
pub(super) fn source_profile(
    dest: &ColorProfile,
    intent: RenderingIntent,
    black: Option<f64>,
) -> Result<ColorProfile, moxcms::CmsError> {
    let mut source = ColorProfile::new_srgb();
    // The working space is described by its tags alone: a CICP description
    // would let the transform evaluate the standard sRGB curve instead.
    source.cicp = None;
    if intent == RenderingIntent::AbsoluteColorimetric
        && let Some(white) = dest.media_white_point
    {
        let scale = [D50[0] / white.x, D50[1] / white.y, D50[2] / white.z];
        if scale.iter().all(|s| s.is_finite() && *s > 0.0) {
            crate::pcs::scale_colorants(&mut source, scale);
        }
    }
    if let Some(k) = black.filter(|&k| k > 0.0) {
        let curve = lifted_srgb_curve(k);
        source.red_trc = Some(curve.clone());
        source.green_trc = Some(curve.clone());
        source.blue_trc = Some(curve);
    }
    as_version_2(&source)
}

/// `profile` declared of version 2: the converter scales no black point of
/// a version 2 source, so every compensation is the lift of this module.
fn as_version_2(profile: &ColorProfile) -> Result<ColorProfile, moxcms::CmsError> {
    let mut bytes = profile.encode()?;
    bytes[8] = 2;
    bytes[9] = 0x40;
    ColorProfile::new_from_slice(&bytes)
}

/// The sRGB tone curve lifted to `(1 − k) · lin + k`: the parametric curve
/// of type 4, `(g, a', b', c', d, e, f)` with `a' = a · (1 − k)^(1/g)`,
/// `b' = b · (1 − k)^(1/g)`, `c' = (1 − k) · c` and `e = f = k`.
#[expect(
    clippy::many_single_char_names,
    reason = "g, a … d are the parameter names of ICC.1:2022 Table 68, k the lift"
)]
fn lifted_srgb_curve(k: f64) -> ToneReprCurve {
    let [g, a, b, c, d] = SRGB_CURVE;
    let s = (1.0 - k).powf(1.0 / g);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "an ICC parametric curve stores s15Fixed16 numbers, which f32 holds"
    )]
    let params = [g, a * s, b * s, (1.0 - k) * c, d, k, k].map(|p| p as f32);
    ToneReprCurve::Parametric(params.to_vec())
}

/// Whether `dest` is a version 4 profile.
fn is_v4(dest: &ColorProfile) -> bool {
    dest.version() >= ProfileVersion::V4_0 && dest.version() != ProfileVersion::Unknown
}

/// Whether `dest` converts under `intent` through version 4 perceptual or
/// saturation tables.
fn perceptual_v4_tables(dest: &ColorProfile, intent: RenderingIntent) -> bool {
    matches!(
        intent,
        RenderingIntent::Perceptual | RenderingIntent::Saturation
    ) && is_v4(dest)
        && (dest.lut_b_to_a_perceptual.is_some()
            || dest.lut_b_to_a_saturation.is_some()
            || dest.lut_b_to_a_colorimetric.is_some())
}

/// The luminance of the neutral destination black the working-space black
/// is lifted to for a conversion to `dest` under `intent`, or `None` for
/// none.
///
/// A perceptual or saturation conversion to a version 4 profile always
/// compensates the black point: through its tables to the black of the
/// perceptual reference medium (ICC.1:2022 §6.3.4.4), to a matrix/TRC
/// destination to the destination's own.  To a version 2 profile, and
/// under relative colorimetry, the black is compensated, to the
/// destination's own, when `bpc` asks for it; under absolute colorimetry
/// never.  A destination's own black is [`destination_black`]'s.
///
/// # Errors
///
/// The destination's device → PCS conversion could not be built or run.
pub(super) fn black_lift(
    dest: &ColorProfile,
    intent: RenderingIntent,
    bpc: bool,
) -> Result<Option<f64>, moxcms::CmsError> {
    let own_black = match intent {
        RenderingIntent::Perceptual | RenderingIntent::Saturation => {
            if perceptual_v4_tables(dest, intent) {
                return Ok(Some(PERCEPTUAL_BLACK_Y));
            }
            bpc || is_v4(dest)
        }
        RenderingIntent::RelativeColorimetric => bpc,
        RenderingIntent::AbsoluteColorimetric => false,
    };
    if !own_black {
        return Ok(None);
    }
    destination_black(dest, intent).map(Some)
}

/// The luminance of `dest`'s neutral black under `intent` (ISO 18619 §7):
/// for a destination converting through PCS → device tables, found on the
/// round trip of the neutral `L*` ramp through the device (under `intent`
/// there, relative colorimetry back); otherwise the PCS of the device's
/// darkest colorant (device 0 of an additive space).  Neutral, and at
/// most `L* = 50`.
///
/// On the round trip the black is the darkest colorant's under relative
/// colorimetry when the ramp's midrange is straight (within 4 `L*` of the
/// identity above the bottom fifth of its range); otherwise, and always
/// under the other intents, the vertex of the least-squares quadratic
/// through the ramp's shadow points: those whose output, normalised to
/// the ramp's range, lies in `[0.1, 0.5)` (relative) or `[0.03, 0.25)`.
fn destination_black(
    dest: &ColorProfile,
    intent: RenderingIntent,
) -> Result<f64, moxcms::CmsError> {
    let relative = intent == RenderingIntent::RelativeColorimetric;
    let tables = dest.lut_b_to_a_perceptual.is_some()
        || dest.lut_b_to_a_colorimetric.is_some()
        || dest.lut_b_to_a_saturation.is_some();
    let initial = if relative || !tables {
        darkest_colorant_y(dest, intent)?.clamp(0.0, MAX_BLACK_Y)
    } else {
        0.0
    };
    if !tables {
        return Ok(initial);
    }
    let ramp = round_trip_ramp(dest, intent)?;
    let (min, max) = (ramp[0].1, ramp[ramp.len() - 1].1);
    if min >= max || !min.is_finite() || !max.is_finite() {
        return Ok(0.0);
    }
    let straight = ramp
        .iter()
        .all(|&(l_in, l_out)| l_in <= 0.2f64.mul_add(max - min, min) || (l_in - l_out).abs() < 4.0);
    if relative && straight {
        return Ok(initial);
    }
    let (lo, hi) = if relative { (0.1, 0.5) } else { (0.03, 0.25) };
    let shadows: Vec<(f64, f64)> = ramp
        .iter()
        .map(|&(l_in, l_out)| (l_in, (l_out - min) / (max - min)))
        .filter(|&(_, y)| y >= lo && y < hi)
        .collect();
    if shadows.len() < 3 {
        return Ok(0.0);
    }
    Ok(lab_l_to_y(quadratic_root(&shadows)))
}

/// The neutral `L*` ramp of 256 steps from 0 to 100 through `dest` under
/// `intent` and back under relative colorimetry: `(L* in, L* out)`, the
/// outputs made non-increasing towards the dark end.
fn round_trip_ramp(
    dest: &ColorProfile,
    intent: RenderingIntent,
) -> Result<Vec<(f64, f64)>, moxcms::CmsError> {
    let (layout, channels) = match dest.color_space {
        DataColorSpace::Gray => (Layout::Gray, 1),
        DataColorSpace::Cmyk => (Layout::Rgba, 4),
        _ => (Layout::Rgb, 3),
    };
    let pcs = as_version_2(&pcs_lightness())?;
    let options = |rendering_intent| TransformOptions {
        rendering_intent,
        allow_use_cicp_transfer: false,
        ..TransformOptions::default()
    };
    let to_device = pcs.create_transform_f64(Layout::Rgb, dest, layout, options(intent))?;
    let to_pcs = dest.create_transform_f64(
        layout,
        &pcs,
        Layout::Rgb,
        options(RenderingIntent::RelativeColorimetric),
    )?;
    let l_in: Vec<f64> = (0..256).map(|i| f64::from(i) * 100.0 / 255.0).collect();
    let xyz: Vec<f64> = l_in
        .iter()
        .flat_map(|&l| {
            let y = lab_l_to_y(l);
            [D50[0] * y, y, D50[2] * y].map(lightness_encode)
        })
        .collect();
    let mut device = vec![0.0; l_in.len() * channels];
    to_device.transform(&xyz, &mut device)?;
    let mut back = vec![0.0; xyz.len()];
    to_pcs.transform(&device, &mut back)?;
    let mut l_out: Vec<f64> = back
        .as_chunks::<3>()
        .0
        .iter()
        .map(|px| 100.0 * px[1])
        .collect();
    for i in (1..l_out.len() - 1).rev() {
        l_out[i] = l_out[i].min(l_out[i + 1]);
    }
    Ok(l_in.into_iter().zip(l_out).collect())
}

/// The root of the least-squares quadratic `y = a·x² + b·x + c` through
/// `points`, `(−b + √(b² − 4ac)) / 2a`, within `[0, 50]`; 0 when the
/// curve is nearly linear or has no real root.
#[expect(
    clippy::many_single_char_names,
    reason = "a, b, c are the quadratic's coefficients, d its discriminant, m the normal equations"
)]
fn quadratic_root(points: &[(f64, f64)]) -> f64 {
    let mut m = [[0.0f64; 4]; 3];
    for &(x, y) in points {
        let powers = [x * x, x, 1.0];
        for (row, &p) in m.iter_mut().zip(&powers) {
            for (cell, &q) in row.iter_mut().zip(&powers) {
                *cell = p.mul_add(q, *cell);
            }
            row[3] = p.mul_add(y, row[3]);
        }
    }
    let Some([a, b, c]) = solve3(m) else {
        return 0.0;
    };
    if a.abs() < 1e-10 {
        return 0.0;
    }
    let d = b.mul_add(b, -4.0 * a * c);
    if d <= 0.0 {
        return 0.0;
    }
    ((-b + d.sqrt()) / (2.0 * a)).clamp(0.0, 50.0)
}

/// The solution of the 3 × 3 system of augmented rows `m`, by Gaussian
/// elimination with partial pivoting; `None` when singular.
fn solve3(mut m: [[f64; 4]; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        let pivot = (col..3).max_by(|&i, &j| m[i][col].abs().total_cmp(&m[j][col].abs()))?;
        if m[pivot][col].abs() < f64::MIN_POSITIVE {
            return None;
        }
        m.swap(col, pivot);
        let lead = m[col];
        for (row, r) in m.iter_mut().enumerate() {
            if row != col {
                let f = r[col] / lead[col];
                for (v, &l) in r.iter_mut().zip(&lead).skip(col) {
                    *v = f.mul_add(-l, *v);
                }
            }
        }
    }
    Some([m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2]])
}

/// The relative luminance of CIE lightness `l` (CIE 15:2004 §8.2.1).
fn lab_l_to_y(l: f64) -> f64 {
    let f = (l + 16.0) / 116.0;
    if f > 6.0 / 29.0 {
        f.powi(3)
    } else {
        3.0 * (6.0f64 / 29.0).powi(2) * (f - 4.0 / 29.0)
    }
}

/// The CIE lightness of relative luminance `y`.
fn y_to_lab_l(y: f64) -> f64 {
    let f = if y > (6.0f64 / 29.0).powi(3) {
        y.cbrt()
    } else {
        y / (3.0 * (6.0f64 / 29.0).powi(2)) + 4.0 / 29.0
    };
    116.0f64.mul_add(f, -16.0)
}

/// The PCS luminance `Y` of `dest`'s darkest colorant under `intent`,
/// relative to the D50 white, through [`pcs_lightness`]: device 0 of an
/// additive space; for CMYK, whose darkest ink the profile's separation
/// limits, the PCS black's round trip through the device
/// ([`round_trip_ramp`]).
fn darkest_colorant_y(
    dest: &ColorProfile,
    intent: RenderingIntent,
) -> Result<f64, moxcms::CmsError> {
    if dest.color_space == DataColorSpace::Cmyk {
        return Ok(lab_l_to_y(round_trip_ramp(dest, intent)?[0].1));
    }
    let (layout, black): (Layout, &[f64]) = match dest.color_space {
        DataColorSpace::Gray => (Layout::Gray, &[0.0]),
        _ => (Layout::Rgb, &[0.0, 0.0, 0.0]),
    };
    let xform = dest.create_transform_f64(
        layout,
        &pcs_lightness(),
        Layout::Rgb,
        TransformOptions {
            rendering_intent: intent,
            allow_use_cicp_transfer: false,
            ..TransformOptions::default()
        },
    )?;
    let mut pcs = [0.0; 3];
    xform.transform(black, &mut pcs)?;
    Ok(lab_l_to_y(100.0 * pcs[1]))
}

/// An RGB profile whose device values are the PCS `X`, `Y`, `Z`, each
/// relative to its axis encoded by the CIE lightness function: unit
/// colorants along the axes and the tone curve `t ↦ Y(L* = 100 t)`.  The
/// encoding spaces the device values evenly in lightness, so a conversion
/// sampled on them resolves the shadows as finely as the highlights.
fn pcs_lightness() -> ColorProfile {
    let mut profile = ColorProfile::new_srgb();
    profile.cicp = None;
    profile.red_colorant = Xyzd {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    };
    profile.green_colorant = Xyzd {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };
    profile.blue_colorant = Xyzd {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a luminance in [0, 1] scaled to the u16 range"
    )]
    let table: Vec<u16> = (0..LIGHTNESS_TABLE_LEN)
        .map(|i| {
            let t = f64::from(i) / f64::from(LIGHTNESS_TABLE_LEN - 1);
            (lab_l_to_y(100.0 * t) * 65535.0).round() as u16
        })
        .collect();
    let curve = ToneReprCurve::Lut(table);
    profile.red_trc = Some(curve.clone());
    profile.green_trc = Some(curve.clone());
    profile.blue_trc = Some(curve);
    profile
}

/// Entries of the tone curve of [`pcs_lightness`].
const LIGHTNESS_TABLE_LEN: u16 = 4096;

/// The [`pcs_lightness`] device value of a PCS component `v` relative to
/// its axis.
fn lightness_encode(v: f64) -> f64 {
    y_to_lab_l(v) / 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sRGB decoding of `x`.
    #[expect(
        clippy::many_single_char_names,
        reason = "the parameter names of Table 68"
    )]
    fn srgb(x: f64) -> f64 {
        let [g, a, b, c, d] = SRGB_CURVE;
        if x >= d {
            a.mul_add(x, b).powf(g)
        } else {
            c * x
        }
    }

    /// A parametric curve of type 4 at `x` (ICC.1:2022 Table 68).
    #[expect(
        clippy::many_single_char_names,
        reason = "the parameter names of Table 68"
    )]
    fn type4(curve: &ToneReprCurve, x: f64) -> f64 {
        let ToneReprCurve::Parametric(p) = curve else {
            panic!("a parametric curve")
        };
        let [g, a, b, c, d, e, f] = <[f32; 7]>::try_from(p.as_slice())
            .expect("type 4")
            .map(f64::from);
        if x >= d {
            a.mul_add(x, b).powf(g) + e
        } else {
            c.mul_add(x, f)
        }
    }

    #[test]
    fn the_lifted_curve_is_the_affine_lift_of_srgb() {
        for k in [0.0, 0.003_479, 0.02, 0.1] {
            let lifted = lifted_srgb_curve(k);
            for i in 0..=20 {
                let x = f64::from(i) / 20.0;
                let want = (1.0 - k).mul_add(srgb(x), k);
                let got = type4(&lifted, x);
                assert!((got - want).abs() < 1e-5, "k {k}, x {x}: {got} vs {want}");
            }
        }
    }

    #[test]
    fn the_working_space_black_of_a_matrix_destination_is_its_origin() {
        let dest = ColorProfile::new_srgb();
        let y = darkest_colorant_y(&dest, RenderingIntent::RelativeColorimetric).unwrap();
        assert!(y.abs() < 1e-4, "{y}");
    }

    #[test]
    fn absolute_colorimetry_scales_the_colorants_to_the_media_white() {
        let mut dest = ColorProfile::new_srgb();
        dest.media_white_point = Some(Xyzd {
            x: 0.9,
            y: 0.95,
            z: 0.7,
        });
        let plain = source_profile(&dest, RenderingIntent::RelativeColorimetric, None).unwrap();
        let abs = source_profile(&dest, RenderingIntent::AbsoluteColorimetric, None).unwrap();
        assert!((plain.red_colorant.x - ColorProfile::new_srgb().red_colorant.x).abs() < 1e-4);
        let ratio = abs.green_colorant.z / plain.green_colorant.z;
        assert!((ratio - D50[2] / 0.7).abs() < 1e-4, "{ratio}");
    }
}
