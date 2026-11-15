//! The ICC profile connection space (PCS) and its ICC-absolute colorimetry.
//!
//! A media-relative PCS value `X` and its ICC-absolute value `X_abs` are
//! related through the media white `w` of the profile they belong to,
//! component by component: `X_abs = (w / D50) · X` (ICC.1:2022 §6.3.2.2).
//! A conversion from a source of media white `w_s` to a destination of
//! media white `w_d` under absolute colorimetry therefore scales the
//! relative PCS by `w_s / w_d`.  When one side is a matrix/TRC profile
//! whose device value maps to the PCS as `M · lin(v)`, that scale is folded
//! into its colorant matrix: the colorants of the source are scaled by
//! `D50 / w_d`, those of the destination by `D50 / w_s`.

use moxcms::{ColorProfile, ProfileClass, ProfileVersion, Xyzd};

pub(crate) mod black;

/// The D50 illuminant of the PCS.
pub(crate) const D50: [f64; 3] = [0.964_2, 1.0, 0.824_9];

/// The largest difference of a media-white component from D50 that is
/// read as D50: the resolution of the s15Fixed16 encoding of `wtpt`,
/// with margin for the rounding of its producer.
const WHITE_TOLERANCE: f64 = 1.0e-4;

/// The per-component scale `D50 / white`, `None` when `white` is D50
/// within [`WHITE_TOLERANCE`] or a component is not a positive finite
/// number.
pub(crate) fn scale_from_white(white: Xyzd) -> Option<[f64; 3]> {
    absolute_scale(d50(), white)
}

/// The per-component scale `source / dest` of the relative PCS under
/// ICC-absolute colorimetry from a profile of media white `source` to one
/// of media white `dest`; `None` when the two whites agree within
/// [`WHITE_TOLERANCE`] or a component is not a positive finite number.
pub(crate) fn absolute_scale(source: Xyzd, dest: Xyzd) -> Option<[f64; 3]> {
    let (source, dest) = ([source.x, source.y, source.z], [dest.x, dest.y, dest.z]);
    if source
        .iter()
        .zip(dest)
        .all(|(s, d)| (s - d).abs() <= WHITE_TOLERANCE)
    {
        return None;
    }
    let scale: [f64; 3] = std::array::from_fn(|i| source[i] / dest[i]);
    scale
        .iter()
        .all(|s| s.is_finite() && *s > 0.0)
        .then_some(scale)
}

/// The D50 illuminant as an XYZ value.
pub(crate) const fn d50() -> Xyzd {
    Xyzd {
        x: D50[0],
        y: D50[1],
        z: D50[2],
    }
}

/// The media white of `profile` that ICC-absolute colorimetry scales by:
/// its `wtpt` tag, except for a version 2 display profile, whose `wtpt`
/// records the display's white before its adaptation to D50 and whose
/// media white is therefore D50; D50 when the tag is absent.
pub(crate) fn media_white(profile: &ColorProfile) -> Xyzd {
    let d50 = d50();
    let v2_display = profile.profile_class == ProfileClass::DisplayDevice
        && profile.version() < ProfileVersion::V4_0;
    if v2_display {
        return d50;
    }
    profile.media_white_point.unwrap_or(d50)
}

/// Scale every colorant of the matrix/TRC `profile` by `scale`, component
/// by component: the profile whose PCS is that of `profile` scaled by
/// `scale`.
pub(crate) fn scale_colorants(profile: &mut ColorProfile, scale: [f64; 3]) {
    for colorant in [
        &mut profile.red_colorant,
        &mut profile.green_colorant,
        &mut profile.blue_colorant,
    ] {
        *colorant = Xyzd {
            x: colorant.x * scale[0],
            y: colorant.y * scale[1],
            z: colorant.z * scale[2],
        };
    }
}

/// sRGB described by its tags alone, its PCS scaled by `scale` (see
/// [`scale_colorants`]).  Without a CICP description the converter
/// evaluates the tags, so the scale takes effect.
pub(crate) fn scaled_srgb(scale: [f64; 3]) -> ColorProfile {
    let mut srgb = ColorProfile::new_srgb();
    srgb.cicp = None;
    scale_colorants(&mut srgb, scale);
    srgb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d50_white_has_no_scale() {
        let d50 = Xyzd {
            x: 0.964_25,
            y: 1.0,
            z: 0.824_88,
        };
        assert_eq!(scale_from_white(d50), None);
    }

    #[test]
    fn paper_white_scales_each_component() {
        let paper = Xyzd {
            x: 0.8,
            y: 0.9,
            z: 0.7,
        };
        let s = scale_from_white(paper).expect("not D50");
        assert!((s[1] - 1.0 / 0.9).abs() < 1e-12);
        assert!((s[0] - D50[0] / 0.8).abs() < 1e-12);
    }

    #[test]
    fn degenerate_white_has_no_scale() {
        let zero = Xyzd {
            x: 0.0,
            y: 1.0,
            z: 0.8,
        };
        assert_eq!(scale_from_white(zero), None);
    }

    /// sRGB with the media white `wtpt` of D65, encoded as version
    /// `major`.
    fn display_profile(major: u8) -> ColorProfile {
        let mut srgb = ColorProfile::new_srgb();
        srgb.media_white_point = Some(Xyzd {
            x: 0.950_5,
            y: 1.0,
            z: 1.089,
        });
        let mut bytes = srgb.encode().expect("encodes");
        bytes[8] = major;
        bytes[9] = if major == 2 { 0x40 } else { 0x30 };
        ColorProfile::new_from_slice(&bytes).expect("parses")
    }

    #[test]
    fn v2_display_media_white_is_d50() {
        let white = media_white(&display_profile(2));
        assert_eq!(scale_from_white(white), None);
        let white = media_white(&display_profile(4));
        assert!(scale_from_white(white).is_some());
    }
}
