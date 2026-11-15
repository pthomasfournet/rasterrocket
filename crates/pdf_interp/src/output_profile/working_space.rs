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

use moxcms::{ColorProfile, RenderingIntent};

use crate::pcs::D50;
use crate::pcs::black::{
    MAX_BLACK_Y, PERCEPTUAL_BLACK_Y, as_version_2, darkest_colorant_y, is_v4, lab_l_to_y,
    lifted_srgb_curve, round_trip_ramp,
};

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

#[cfg(test)]
mod tests {
    use moxcms::Xyzd;

    use super::*;

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
