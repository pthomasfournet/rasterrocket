//! The separation table of a CMYK output profile, evaluated at its nodes.
//!
//! The source of a separation is the working space as a matrix/TRC profile
//! with the intent's adjustments folded in (see [`super::source`]): a node
//! `v` maps to the PCS as `M · lin(v)`, evaluated in floating point.  The
//! profile's PCS → device table of the intent is then evaluated at that PCS
//! value ([`crate::pcs::lut::ProfileLut`]), so each node holds the profile's
//! own separation of its colour, which no intermediate grid resamples.

use moxcms::ColorProfile;

use crate::WorkingSpace;
use crate::pcs::lut::ProfileLut;
use crate::resources::icc_shaper::eval_trc;
use crate::separation::{PROFILE_GRID, ProfileSeparation};

/// The [`ProfileSeparation`] of colours of `space` through `dest` under
/// `intent`, `source` the working space as the conversion's source; `None`
/// when `source` is not a matrix/TRC profile or `dest` has no PCS → CMYK
/// table of the intent.
pub(super) fn tabulate(
    source: &ColorProfile,
    dest: &ColorProfile,
    intent: moxcms::RenderingIntent,
    space: WorkingSpace,
) -> Option<ProfileSeparation> {
    let lut = ProfileLut::new(dest, intent, 4)?;
    let trcs = [&source.red_trc, &source.green_trc, &source.blue_trc];
    let trcs: Vec<&moxcms::ToneReprCurve> =
        trcs.iter().map(|t| t.as_ref()).collect::<Option<_>>()?;
    let m = source.rgb_to_xyz_matrix().v;
    let lin: Vec<Vec<f64>> = trcs
        .iter()
        .map(|trc| {
            (0..PROFILE_GRID)
                .map(|i| {
                    eval_trc(
                        trc,
                        f64::from(ProfileSeparation::node_value(i)) / f64::from(u16::MAX),
                    )
                })
                .collect::<Option<_>>()
        })
        .collect::<Option<_>>()?;
    let mut nodes = Vec::with_capacity(PROFILE_GRID.pow(3));
    let mut ink = [0.0f64; 4];
    for r in 0..PROFILE_GRID {
        for g in 0..PROFILE_GRID {
            for b in 0..PROFILE_GRID {
                let rgb = [lin[0][r], lin[1][g], lin[2][b]];
                let xyz =
                    m.map(|row| row[0].mul_add(rgb[0], row[1].mul_add(rgb[1], row[2] * rgb[2])));
                lut.to_device(xyz, &mut ink);
                nodes.push(ink.map(|v| {
                    #[expect(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "a value of [0, 1] scaled to the u16 range"
                    )]
                    let q = (v * f64::from(u16::MAX)).round() as u16;
                    q
                }));
            }
        }
    }
    ProfileSeparation::from_nodes(nodes, space)
}
