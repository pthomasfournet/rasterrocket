//! Black points in the profile connection space: the black a profile's
//! device values reach, found as ISO 18619 finds it, and the sRGB tone
//! curve lifted to map one black onto another.
//!
//! Lightness and luminance are relative to the D50 white of the PCS; a
//! black is neutral, `k · D50`, and given by its luminance `k`.

use moxcms::{
    ColorProfile, DataColorSpace, Layout, ProfileClass, ProfileVersion, RenderingIntent,
    ToneReprCurve, TransformOptions, Xyzd,
};

use super::D50;

/// The luminance `Y` of the perceptual reference medium's black
/// (ICC.1:2022 §6.3.4.4), the black of a version 4 profile's perceptual
/// and saturation tables.
pub(crate) const PERCEPTUAL_BLACK_Y: f64 = 0.003_479;

/// The largest luminance accepted as a black point: `L* = 50`.
pub(crate) const MAX_BLACK_Y: f64 = 0.184_186;

/// The sRGB tone curve as an ICC parametric curve of type 3,
/// `(g, a, b, c, d)` (ICC.1:2022 Table 68).
pub(crate) const SRGB_CURVE: [f64; 5] = [2.4, 1.0 / 1.055, 0.055 / 1.055, 1.0 / 12.92, 0.040_45];

/// `profile` declared of version 2: the converter scales no black point of
/// a version 2 source, so every compensation is the lift of this module.
pub(crate) fn as_version_2(profile: &ColorProfile) -> Result<ColorProfile, moxcms::CmsError> {
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
pub(crate) fn lifted_srgb_curve(k: f64) -> ToneReprCurve {
    let [g, a, b, c, d] = SRGB_CURVE;
    let s = (1.0 - k).powf(1.0 / g);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "an ICC parametric curve stores s15Fixed16 numbers, which f32 holds"
    )]
    let params = [g, a * s, b * s, (1.0 - k) * c, d, k, k].map(|p| p as f32);
    ToneReprCurve::Parametric(params.to_vec())
}

/// Whether `profile` is a version 4 profile.
pub(crate) fn is_v4(profile: &ColorProfile) -> bool {
    profile.version() >= ProfileVersion::V4_0 && profile.version() != ProfileVersion::Unknown
}

/// The neutral `L*` ramp of 256 steps from 0 to 100 through `dest` under
/// `intent` and back under relative colorimetry: `(L* in, L* out)`, the
/// outputs made non-increasing towards the dark end.
pub(crate) fn round_trip_ramp(
    dest: &ColorProfile,
    intent: RenderingIntent,
) -> Result<Vec<(f64, f64)>, moxcms::CmsError> {
    let l_in: Vec<f64> = (0..256).map(|i| f64::from(i) * 100.0 / 255.0).collect();
    let mut l_out = round_trip(dest, intent, &l_in)?;
    for i in (1..l_out.len() - 1).rev() {
        l_out[i] = l_out[i].min(l_out[i + 1]);
    }
    Ok(l_in.into_iter().zip(l_out).collect())
}

/// The `L*` of each neutral PCS lightness of `l_in` on its round trip
/// through `profile`'s device values: into the device under `intent`, back
/// under relative colorimetry.
fn round_trip(
    profile: &ColorProfile,
    intent: RenderingIntent,
    l_in: &[f64],
) -> Result<Vec<f64>, moxcms::CmsError> {
    let (layout, channels) = layout_of(profile);
    let pcs = as_version_2(&pcs_lightness())?;
    let to_device =
        pcs.create_transform_f64(Layout::Rgb, profile, layout, lightness_options(intent))?;
    let to_pcs = profile.create_transform_f64(
        layout,
        &pcs,
        Layout::Rgb,
        lightness_options(RenderingIntent::RelativeColorimetric),
    )?;
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
    Ok(back
        .as_chunks::<3>()
        .0
        .iter()
        .map(|px| 100.0 * px[1])
        .collect())
}

/// The layout and channel count of `profile`'s device values.
pub(crate) const fn layout_of(profile: &ColorProfile) -> (Layout, usize) {
    match profile.color_space {
        DataColorSpace::Gray => (Layout::Gray, 1),
        DataColorSpace::Cmyk => (Layout::Rgba, 4),
        _ => (Layout::Rgb, 3),
    }
}

/// The options of a conversion to or from [`pcs_lightness`] under
/// `intent`.
fn lightness_options(rendering_intent: RenderingIntent) -> TransformOptions {
    TransformOptions {
        rendering_intent,
        allow_use_cicp_transfer: false,
        ..TransformOptions::default()
    }
}

/// The PCS luminance `Y`, relative to the D50 white, of the device value
/// `device` of `profile` under `intent`.
fn device_y(
    profile: &ColorProfile,
    intent: RenderingIntent,
    device: &[f64],
) -> Result<f64, moxcms::CmsError> {
    let xform = profile.create_transform_f64(
        layout_of(profile).0,
        &pcs_lightness(),
        Layout::Rgb,
        lightness_options(intent),
    )?;
    let mut pcs = [0.0; 3];
    xform.transform(device, &mut pcs)?;
    Ok(lab_l_to_y(100.0 * pcs[1]))
}

/// The relative luminance of CIE lightness `l` (CIE 15:2004 §8.2.1).
pub(crate) fn lab_l_to_y(l: f64) -> f64 {
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
pub(crate) fn darkest_colorant_y(
    dest: &ColorProfile,
    intent: RenderingIntent,
) -> Result<f64, moxcms::CmsError> {
    match dest.color_space {
        DataColorSpace::Cmyk => Ok(lab_l_to_y(round_trip_ramp(dest, intent)?[0].1)),
        DataColorSpace::Gray => device_y(dest, intent, &[0.0]),
        _ => device_y(dest, intent, &[0.0; 3]),
    }
}

/// An RGB profile whose device values are the PCS `X`, `Y`, `Z`, each
/// relative to its axis encoded by the CIE lightness function: unit
/// colorants along the axes and the tone curve `t ↦ Y(L* = 100 t)`.  The
/// encoding spaces the device values evenly in lightness, so a conversion
/// sampled on them resolves the shadows as finely as the highlights.
pub(crate) fn pcs_lightness() -> ColorProfile {
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

/// The luminance of `profile`'s black point as the source of a conversion
/// under `intent` (ISO 18619 §6, as Little CMS detects it), at most
/// [`MAX_BLACK_Y`], else 0:
///
/// - perceptual or saturation tables of a version 4 profile: the black of
///   the perceptual reference medium, [`PERCEPTUAL_BLACK_Y`];
/// - relative colorimetry from a CMYK output profile: the round trip
///   through the device of the PCS black under perceptual, which discounts
///   the ink limit; for a version 4 profile, whose perceptual PCS black is
///   the reference medium's, of that black;
/// - otherwise the darkest device value under `intent`: no colorant of an
///   additive space, every ink of a subtractive one.
///
/// # Errors
///
/// The profile's conversions could not be built or run.
pub(crate) fn source_black(
    profile: &ColorProfile,
    intent: RenderingIntent,
) -> Result<f64, moxcms::CmsError> {
    let tables = profile.lut_a_to_b_perceptual.is_some()
        || profile.lut_a_to_b_colorimetric.is_some()
        || profile.lut_a_to_b_saturation.is_some();
    let y = match intent {
        RenderingIntent::Perceptual | RenderingIntent::Saturation if is_v4(profile) => {
            if tables {
                PERCEPTUAL_BLACK_Y
            } else {
                darkest_colorant_y(profile, RenderingIntent::RelativeColorimetric)?
            }
        }
        RenderingIntent::RelativeColorimetric
            if profile.color_space == DataColorSpace::Cmyk
                && profile.profile_class == ProfileClass::OutputDevice =>
        {
            let black = if is_v4(profile) {
                y_to_lab_l(PERCEPTUAL_BLACK_Y)
            } else {
                0.0
            };
            lab_l_to_y(round_trip(profile, RenderingIntent::Perceptual, &[black])?[0])
        }
        _ => match profile.color_space {
            DataColorSpace::Cmyk => device_y(profile, intent, &[1.0; 4])?,
            DataColorSpace::Gray => device_y(profile, intent, &[0.0])?,
            _ => device_y(profile, intent, &[0.0; 3])?,
        },
    };
    Ok(if (0.0..=MAX_BLACK_Y).contains(&y) {
        y
    } else {
        0.0
    })
}

/// sRGB described by its tags, its tone curves lifted to `(1 − k) · lin +
/// k` ([`lifted_srgb_curve`]): as a destination, it maps the neutral PCS
/// black `k · D50` to sRGB black and keeps white, the black-point
/// compensation of a source whose black is `k`.
pub(crate) fn compensating_srgb(k: f64) -> ColorProfile {
    let mut srgb = ColorProfile::new_srgb();
    srgb.cicp = None;
    let curve = lifted_srgb_curve(k);
    srgb.red_trc = Some(curve.clone());
    srgb.green_trc = Some(curve.clone());
    srgb.blue_trc = Some(curve);
    srgb
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

    /// The perceptual tables of a version 4 profile end at the black of
    /// the perceptual reference medium.
    #[test]
    fn v4_perceptual_tables_have_the_reference_medium_black() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/icc/printer-cmyk-v4.icc"
        );
        let profile = ColorProfile::new_from_slice(&std::fs::read(path).unwrap()).unwrap();
        let y = source_black(&profile, RenderingIntent::Perceptual).unwrap();
        assert!((y - PERCEPTUAL_BLACK_Y).abs() < 1e-12);
        // Relative colorimetry: the reference medium black, L* = 3.14,
        // through the perceptual table, which scales lightness onto the
        // printer's range from L* = 8, and back: L* = 8 + 0.92 · 3.14.
        let y = source_black(&profile, RenderingIntent::RelativeColorimetric).unwrap();
        let l = y_to_lab_l(y);
        assert!((l - 10.9).abs() < 0.5, "L* {l}");
    }

    #[test]
    fn a_matrix_profile_has_a_black_origin() {
        let srgb = ColorProfile::new_srgb();
        for intent in [
            RenderingIntent::RelativeColorimetric,
            RenderingIntent::Perceptual,
        ] {
            let y = source_black(&srgb, intent).unwrap();
            assert!(y.abs() < 1e-4, "{intent:?}: {y}");
        }
    }
}
