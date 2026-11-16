//! The RGB of a transparency group whose colour space is `CalRGB` or an
//! RGB ICC profile that is not sRGB (ISO 32000-2 §11.3.4, §11.6.6).
//!
//! The group is painted on an additive ink canvas
//! ([`Separator::Additive`](super::Separator::Additive)) whose process
//! channels hold the complements of the components of the group's space.
//! A colour painted in the group converts to the working space as on the
//! page and from there into the group's space ([`GroupRgb::entering`]);
//! the group's result converts back as a colour of its space converts on
//! the page ([`GroupRgb::leaving`]).
//!
//! A `CalRGB` space or a matrix/TRC profile converts in both directions
//! through a [`LinearRgb`]: out of the group, its tone curves and its
//! colorants composed with the working space's
//! ([`working_space::composite`]); into it, the working space's curve, the
//! inverse composite and the inverse of its tone curves.  A profile of
//! device ↔ PCS tables converts through two [`GridTable`]s, tabulated from
//! the converter's transforms between the profile and the working space
//! under relative colorimetry.

use std::sync::{Arc, OnceLock};

use color::Sample;
use moxcms::{ColorProfile, Layout, TransformOptions};
use pdf::Document;

use crate::WorkingSpace;
use crate::pcs::grid::GridTable;
use crate::resources::icc_shaper::{Mat3, eval_trc, fetch_profile_bytes, mat_inverse};
use crate::resources::{ColorSpace, RgbGroupSpace};
use crate::working_space::{self, LinearRgb};

/// The number of cells along each axis of a tabulated conversion.
const SPAN: u32 = 64;

/// A conversion of RGB values between two RGB spaces.
enum RgbMap {
    /// Linearisation, matrix and encoding.
    Linear(LinearRgb),
    /// A grid over the source cube.
    Grid(GridTable<3>),
}

impl RgbMap {
    /// The converted value of `rgb`, at its depth.
    fn convert<S: Sample>(&self, rgb: [S; 3]) -> [S; 3] {
        match self {
            Self::Linear(linear) => linear
                .convert(rgb.map(u16::from_sample))
                .map(S::from_sample),
            Self::Grid(grid) => grid.convert(rgb),
        }
    }
}

/// The RGB space of a transparency group's `CalRGB` or ICC profile and
/// its conversions to and from one working space.
pub(crate) struct GroupRgb {
    /// The group's space.
    space: RgbGroupSpace,
    /// The working space converted to and from.
    working: WorkingSpace,
    /// Working space → group space.
    into: RgbMap,
    /// Group space → working space.
    out: RgbMap,
}

impl std::fmt::Debug for GroupRgb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroupRgb")
            .field("space", &self.space)
            .field("working", &self.working)
            .finish_non_exhaustive()
    }
}

/// Two group spaces are one when they are the same space over the same
/// working space.
impl PartialEq for GroupRgb {
    fn eq(&self, other: &Self) -> bool {
        self.space == other.space && self.working == other.working
    }
}

impl Eq for GroupRgb {}

/// The conversions of an ICC group space, by working space, each built on
/// first use.
#[derive(Default)]
struct IccGroupSpaces([OnceLock<Option<Arc<GroupRgb>>>; WorkingSpace::ALL.len()]);

impl GroupRgb {
    /// The conversions of `space`, a `CalRGB` or ICC group space of `doc`,
    /// to and from `working`; those of an ICC space are built once per
    /// document, profile stream and working space.  `None` for a space of
    /// another kind, and for one that does not convert, which is logged.
    pub(crate) fn load(
        doc: &Document,
        space: RgbGroupSpace,
        working: WorkingSpace,
    ) -> Option<Arc<Self>> {
        let build = || {
            Self::new(doc, space, working)
                .map_err(|e| {
                    log::warn!(
                        "rasterrocket-interp: the RGB colour space of a transparency group \
                         does not convert ({e}); the group composites in the working space"
                    );
                })
                .ok()
                .map(Arc::new)
        };
        match space {
            RgbGroupSpace::Icc(id) => {
                let spaces = doc.derived(id, || Some(IccGroupSpaces::default()))?;
                spaces.0[working.index()].get_or_init(build).clone()
            }
            RgbGroupSpace::Cal(_) => build(),
            _ => None,
        }
    }

    /// Build the conversions of `space` to and from `working`.
    fn new(doc: &Document, space: RgbGroupSpace, working: WorkingSpace) -> Result<Self, String> {
        let (into, out) = match space {
            RgbGroupSpace::Cal(cal) => {
                let curves = curves(cal.gamma().map(|g| move |x: f64| Some(x.powf(g))))?;
                linear_maps(curves, &cal.linear_matrix(working))?
            }
            RgbGroupSpace::Icc(id) => {
                let bytes =
                    fetch_profile_bytes(doc, id).ok_or("the profile stream does not decode")?;
                let profile = ColorProfile::new_from_slice(&bytes).map_err(|e| e.to_string())?;
                if profile.color_space != moxcms::DataColorSpace::Rgb {
                    return Err("the profile is not RGB".to_owned());
                }
                if is_matrix_trc(&profile) {
                    matrix_maps(&profile, working)?
                } else {
                    grid_maps(&profile, working)?
                }
            }
            _ => return Err("the space is not CalRGB or ICC-based".to_owned()),
        };
        Ok(Self {
            space,
            working,
            into,
            out,
        })
    }

    /// Whether `space` is the group's space.
    fn is(&self, space: &ColorSpace) -> bool {
        match (space, self.space) {
            (
                ColorSpace::IccBased {
                    n: 3,
                    stream_id: Some(id),
                },
                RgbGroupSpace::Icc(own),
            ) => *id == own,
            (ColorSpace::Cal(cal), RgbGroupSpace::Cal(own)) => **cal == own,
            _ => false,
        }
    }

    /// The working space converted to and from.
    pub(crate) const fn working(&self) -> WorkingSpace {
        self.working
    }

    /// The group-space value of the working-space colour `rgb`, at its
    /// depth.
    pub(crate) fn entering<S: Sample>(&self, rgb: [S; 3]) -> [S; 3] {
        self.into.convert(rgb)
    }

    /// The working-space value of the group-space colour `rgb`, at its
    /// depth.
    pub(crate) fn leaving<S: Sample>(&self, rgb: [S; 3]) -> [S; 3] {
        self.out.convert(rgb)
    }
}

/// Whether `profile` converts by its matrix/TRC model in both directions:
/// it has tone curves and no table to or from the PCS.
const fn is_matrix_trc(profile: &ColorProfile) -> bool {
    profile.lut_a_to_b_perceptual.is_none()
        && profile.lut_a_to_b_colorimetric.is_none()
        && profile.lut_a_to_b_saturation.is_none()
        && profile.lut_b_to_a_perceptual.is_none()
        && profile.lut_b_to_a_colorimetric.is_none()
        && profile.lut_b_to_a_saturation.is_none()
        && profile.red_trc.is_some()
        && profile.green_trc.is_some()
        && profile.blue_trc.is_some()
}

/// The linear value of every 16-bit level of each component under its
/// curve, `f` of the component.
fn curves(f: [impl Fn(f64) -> Option<f64>; 3]) -> Result<[Arc<[f32]>; 3], String> {
    let [r, g, b] = f.map(|f| LinearRgb::table(f).ok_or("a tone curve does not evaluate"));
    Ok([r?, g?, b?])
}

/// The conversions into and out of a space of tone curves `curves` whose
/// linear components convert to the working space's by `in_working`.
fn linear_maps(curves: [Arc<[f32]>; 3], in_working: &Mat3) -> Result<(RgbMap, RgbMap), String> {
    let of_working = mat_inverse(in_working).ok_or("the colorant matrix is singular")?;
    let srgb = working_space::srgb_linear();
    let into = LinearRgb::into_curves(
        [Arc::clone(srgb), Arc::clone(srgb), Arc::clone(srgb)],
        &of_working,
        curves.clone(),
    )
    .ok_or("a tone curve decreases")?;
    let out = LinearRgb::new(curves, in_working);
    Ok((RgbMap::Linear(into), RgbMap::Linear(out)))
}

/// The conversions of the matrix/TRC `profile` into and out of `working`.
fn matrix_maps(profile: &ColorProfile, working: WorkingSpace) -> Result<(RgbMap, RgbMap), String> {
    let column = |c: &moxcms::Xyzd| [c.x, c.y, c.z];
    let cols = [
        column(&profile.red_colorant),
        column(&profile.green_colorant),
        column(&profile.blue_colorant),
    ];
    let colorants = [0, 1, 2].map(|r| [cols[0][r], cols[1][r], cols[2][r]]);
    let trcs = [&profile.red_trc, &profile.green_trc, &profile.blue_trc];
    let curves = curves(trcs.map(|trc| move |x| trc.as_ref().and_then(|t| eval_trc(t, x))))?;
    linear_maps(curves, &working_space::composite(working, &colorants))
}

/// The conversions of the table-based `profile` into and out of
/// `working`, tabulated from the converter's transforms under relative
/// colorimetry.
fn grid_maps(profile: &ColorProfile, working: WorkingSpace) -> Result<(RgbMap, RgbMap), String> {
    let options = TransformOptions {
        rendering_intent: moxcms::RenderingIntent::RelativeColorimetric,
        prefer_fixed_point: false,
        ..TransformOptions::default()
    };
    let space = working.profile();
    let grid = |from: &ColorProfile, to: &ColorProfile| -> Result<RgbMap, String> {
        let xform = from
            .create_transform_f64(Layout::Rgb, to, Layout::Rgb, options)
            .map_err(|e| e.to_string())?;
        GridTable::tabulate(SPAN, 3, |src, dst| xform.transform(src, dst))
            .map(RgbMap::Grid)
            .map_err(|e| e.to_string())
    };
    Ok((grid(&space, profile)?, grid(profile, &space)?))
}

/// The RGB an additive ink canvas holds: of a working space, or of a
/// group's `CalRGB` or ICC space over one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum AdditiveRgb {
    /// RGB of a working space.
    Working(WorkingSpace),
    /// RGB of a group's space, its colours converted from and to its
    /// working space.
    Group(Arc<GroupRgb>),
}

impl AdditiveRgb {
    /// The working space the colours painted convert from and the
    /// canvas's result converts to.
    pub(crate) fn working(&self) -> WorkingSpace {
        match self {
            Self::Working(space) => *space,
            Self::Group(group) => group.working(),
        }
    }

    /// The canvas's value of the colour `comps` of `space`, its components
    /// clamped to `[0, 1]`, when `space` is the group's: a colour of the
    /// space paints its components.  `None` for a colour of another space,
    /// which converts from the working space.
    pub(crate) fn components_of(&self, space: &ColorSpace, comps: &[f64]) -> Option<[f64; 3]> {
        let Self::Group(group) = self else {
            return None;
        };
        let &[r, g, b] = comps else {
            return None;
        };
        group
            .is(space)
            .then(|| [r, g, b].map(|v| if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) }))
    }

    /// The canvas's value of the colour `rgb` of [`Self::working`].
    pub(crate) fn of_working<S: Sample>(&self, rgb: [S; 3]) -> [S; 3] {
        match self {
            Self::Working(_) => rgb,
            Self::Group(group) => group.entering(rgb),
        }
    }

    /// The colour of [`Self::working`] of the canvas's value `rgb`.
    pub(crate) fn in_working<S: Sample>(&self, rgb: [S; 3]) -> [S; 3] {
        match self {
            Self::Working(_) => rgb,
            Self::Group(group) => group.leaving(rgb),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::make_doc_with_objects;

    /// The conversions of the RGB profile `bytes`, stream `(4, 0)`, over
    /// `working`.
    fn icc(bytes: &[u8], working: WorkingSpace) -> Arc<GroupRgb> {
        let mut s = format!("<< /N 3 /Length {} >>\nstream\n", bytes.len()).into_bytes();
        s.extend_from_slice(bytes);
        s.extend_from_slice(b"\nendstream");
        let doc = make_doc_with_objects(&[(4, s)]);
        GroupRgb::load(&doc, RgbGroupSpace::Icc((4, 0)), working).expect("converts")
    }

    fn adobe() -> Vec<u8> {
        ColorProfile::new_adobe_rgb().encode().expect("encodes")
    }

    /// The largest difference over a lattice of 16-bit sRGB colours, in
    /// the group's working space, between a colour and its round trip
    /// into the group's space and out: each lies within the gamut of the
    /// spaces tested.
    fn round_trip_error(group: &GroupRgb) -> u16 {
        let mut worst = 0;
        for r in (0..=65_535u32).step_by(4369) {
            for g in (0..=65_535u32).step_by(4369) {
                for b in (0..=65_535u32).step_by(4369) {
                    let srgb = [r, g, b].map(|v| u16::try_from(v).expect("16 bits"));
                    let px = working_space::from_srgb::<u16, u16>(group.working(), srgb);
                    let back = group.leaving(group.entering(px));
                    let off = px.iter().zip(back).map(|(a, b)| a.abs_diff(b)).max();
                    worst = worst.max(off.unwrap_or(0));
                }
            }
        }
        worst
    }

    /// The working-space value of the Adobe RGB colour `rgb` as the page
    /// converts it, by the converter's floating-point transform.
    fn adobe_on_page(rgb: [f64; 3], working: WorkingSpace) -> [f64; 3] {
        let profile = ColorProfile::new_adobe_rgb();
        let options = TransformOptions {
            rendering_intent: moxcms::RenderingIntent::RelativeColorimetric,
            prefer_fixed_point: false,
            ..TransformOptions::default()
        };
        let xform = profile
            .create_transform_f64(Layout::Rgb, &working.profile(), Layout::Rgb, options)
            .expect("transform");
        let mut out = [0.0; 3];
        xform.transform(&rgb, &mut out).expect("converts");
        out
    }

    #[test]
    fn a_matrix_profile_converts_out_as_the_converter_does() {
        for working in WorkingSpace::ALL {
            let group = icc(&adobe(), working);
            for rgb in [[1.0, 0.0, 0.0], [0.2, 0.6, 0.9], [0.5, 0.5, 0.5]] {
                let got = group.leaving(rgb.map(u16::from_unit));
                let want = adobe_on_page(rgb, working);
                for (g, w) in got.iter().zip(want) {
                    assert!(
                        (g.to_unit() - w).abs() < 2e-3,
                        "{working:?} {rgb:?}: {got:?} vs {want:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_matrix_profile_inverts_its_conversion() {
        for working in WorkingSpace::ALL {
            let group = icc(&adobe(), working);
            let off = round_trip_error(&group);
            assert!(off <= 48, "{working:?}: {off}");
        }
    }

    #[test]
    fn a_cal_rgb_space_inverts_its_conversion() {
        let dict = pdf::parse_object(
            b"<< /WhitePoint [0.9505 1 1.089] /Gamma [2.2 1.8 2.0] \
              /Matrix [0.5767 0.2974 0.0270 0.1856 0.6273 0.0707 0.1882 0.0753 0.9911] >>",
            &mut 0,
        )
        .expect("a dictionary");
        let doc = make_doc_with_objects::<&[u8]>(&[]);
        let cal = crate::resources::colorspace::CalSpace::resolve(&doc, &dict, 3).expect("cal");
        for working in WorkingSpace::ALL {
            let group = GroupRgb::load(&doc, RgbGroupSpace::Cal(cal), working).expect("converts");
            // The space shares sRGB's red and blue primaries to the four
            // digits of its matrix, so a colour on that edge of sRGB lies
            // just outside it and clips: under an 8-bit level.
            let off = round_trip_error(&group);
            assert!(off <= 192, "{working:?}: {off}");
            let gray = group.leaving([65_535u16; 3]);
            assert!(gray.iter().all(|&v| v >= 65_000), "white: {gray:?}");
        }
    }

    #[test]
    fn a_table_profile_converts_through_grids() {
        let path = format!(
            "{}/../../tests/fixtures/icc/printer-rgb-v4.icc",
            env!("CARGO_MANIFEST_DIR")
        );
        let bytes = std::fs::read(&path).expect("the fixture reads");
        let group = icc(&bytes, WorkingSpace::Srgb);
        assert!(matches!(group.into, RgbMap::Grid(_)));
        // A mid-tone colour within the printer's gamut returns near itself.
        for rgb in [[128u8, 128, 128], [160, 110, 90], [90, 140, 120]] {
            let back = group.leaving(group.entering(rgb));
            let off = rgb.iter().zip(back).map(|(a, b)| a.abs_diff(b)).max();
            assert!(off <= Some(3), "{rgb:?} → {back:?}");
        }
    }

    #[test]
    fn a_stream_that_is_not_a_profile_does_not_convert() {
        let doc = make_doc_with_objects(&[(
            4,
            b"<< /N 3 /Length 4 >>\nstream\njunk\nendstream".to_vec(),
        )]);
        assert!(GroupRgb::load(&doc, RgbGroupSpace::Icc((4, 0)), WorkingSpace::Srgb).is_none());
    }

    #[test]
    fn an_additive_working_space_is_the_identity() {
        let space = AdditiveRgb::Working(WorkingSpace::Wide);
        assert_eq!(space.of_working([1u16, 2, 3]), [1, 2, 3]);
        assert_eq!(space.in_working([1u16, 2, 3]), [1, 2, 3]);
        assert_eq!(space.working(), WorkingSpace::Wide);
    }
}
