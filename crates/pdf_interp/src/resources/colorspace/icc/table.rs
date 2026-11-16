//! The conversion of a three-channel profile's device → PCS table to a
//! working space, tabulated over the device cube.
//!
//! Each node of a grid of [`SPAN`] cells along each device axis holds the
//! working-space value of its device value: the table evaluated at that
//! value in floating point ([`ProfileLut`], whose CLUT interpolates
//! tetrahedrally), the media-relative PCS adjusted for the conversion's
//! form ([`PcsAdjustment`]), then the working space's PCS → linear matrix
//! and its sRGB curve, each linear component clipped to `[0, 1]`.  A pixel
//! converts by interpolation in the tetrahedron of its grid cell
//! ([`GridTable`]), at 16 bits, so device values of equal components take
//! the nodes of equal components alone.

use moxcms::{ColorProfile, DataColorSpace, RenderingIntent};

use crate::WorkingSpace;
use crate::pcs::D50;
use crate::pcs::grid::GridTable;
use crate::pcs::lut::{Direction, ProfileLut};
use crate::resources::icc_shaper::srgb_encode;

/// The number of cells along each device axis of the grid.
const SPAN: u32 = 32;

/// The adjustment of the media-relative PCS of a conversion's form.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum PcsAdjustment {
    /// The PCS as it is.
    None,
    /// ICC-absolute colorimetry: each component divided by the
    /// [`crate::pcs::scale_from_white`] of the source's media white.
    Absolute([f64; 3]),
    /// Black-point compensation of the neutral source black `k · D50`:
    /// `(X − k · D50) / (1 − k)`, mapping that black to the PCS origin and
    /// keeping the white.
    Compensated(f64),
}

impl PcsAdjustment {
    fn apply(self, xyz: [f64; 3]) -> [f64; 3] {
        match self {
            Self::None => xyz,
            Self::Absolute(scale) => std::array::from_fn(|i| xyz[i] / scale[i]),
            Self::Compensated(k) => std::array::from_fn(|i| k.mul_add(-D50[i], xyz[i]) / (1.0 - k)),
        }
    }
}

/// The conversion of `profile`'s device values to `space` through its
/// device → PCS table under `intent` ([`crate::pcs::lut::table_of`]),
/// adjusted by `adjustment`; `None` when the profile's device space is not
/// of three channels, is a PCS, or the table is missing or malformed.
pub(super) fn tabulate(
    profile: &ColorProfile,
    intent: RenderingIntent,
    adjustment: PcsAdjustment,
    space: WorkingSpace,
) -> Option<GridTable<3>> {
    use DataColorSpace::{Cmy, Color3, Hls, Hsv, Luv, Rgb, YCbr, Yxy};
    if !matches!(
        profile.color_space,
        Rgb | Cmy | Hsv | Hls | YCbr | Yxy | Luv | Color3
    ) {
        return None;
    }
    let lut = ProfileLut::new(profile, intent, Direction::ToPcs, 3)?;
    let to_linear = space.pcs_to_linear();
    GridTable::tabulate(SPAN, 3, |device, out| {
        for (px, o) in device
            .as_chunks::<3>()
            .0
            .iter()
            .zip(out.as_chunks_mut::<3>().0)
        {
            let xyz = adjustment.apply(lut.to_pcs(px));
            *o = to_linear
                .map(|[a, b, c]| srgb_encode(a.mul_add(xyz[0], b.mul_add(xyz[1], c * xyz[2]))));
        }
        Ok::<(), std::convert::Infallible>(())
    })
    .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> ColorProfile {
        let path = format!(
            "{}/../../tests/fixtures/icc/{name}.icc",
            env!("CARGO_MANIFEST_DIR")
        );
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        ColorProfile::new_from_slice(&bytes).expect("parses")
    }

    /// The version 4 test printer maps a neutral device value to a neutral
    /// PCS (the version 2 one holds version 4 Lab read in the legacy
    /// encoding, `a* = b* ≈ 0.5`), and
    /// the tabulation keeps it neutral between nodes, within the 16 levels
    /// of 65 535 by which the working spaces' matrices miss mapping the D50
    /// white to equal components (a multilinear device → PCS table
    /// misses by some 500).
    #[test]
    fn neutral_device_values_convert_to_neutrals() {
        for name in ["printer-rgb-v4"] {
            let profile = fixture(name);
            for intent in [
                RenderingIntent::Perceptual,
                RenderingIntent::RelativeColorimetric,
                RenderingIntent::Saturation,
            ] {
                for space in WorkingSpace::ALL {
                    let table =
                        tabulate(&profile, intent, PcsAdjustment::None, space).expect("tabulates");
                    for v in (0..=u16::MAX).step_by(257 * 3) {
                        let out: [u16; 3] = table.convert([v; 3]);
                        let spread = out.iter().max().unwrap() - out.iter().min().unwrap();
                        assert!(spread <= 16, "{name} {intent:?} {space:?} {v}: {out:?}");
                    }
                }
            }
        }
    }

    /// Compensation maps the source black to the working black and keeps
    /// the white; absolute colorimetry divides the PCS by the white's
    /// scale.
    #[test]
    fn the_adjustments_map_black_and_white() {
        let k = 0.01;
        let black = [k * D50[0], k, k * D50[2]];
        let compensated = PcsAdjustment::Compensated(k);
        assert!(compensated.apply(black).iter().all(|v| v.abs() < 1e-15));
        for (a, b) in compensated.apply(D50).iter().zip(D50) {
            assert!((a - b).abs() < 1e-15);
        }
        assert_eq!(
            PcsAdjustment::Absolute([2.0, 4.0, 0.5]).apply([1.0, 1.0, 1.0]),
            [0.5, 0.25, 2.0]
        );
    }

    #[test]
    fn a_cmyk_or_matrix_profile_does_not_tabulate() {
        let none = |p: &ColorProfile| {
            tabulate(
                p,
                RenderingIntent::Perceptual,
                PcsAdjustment::None,
                WorkingSpace::Srgb,
            )
            .is_none()
        };
        assert!(none(&fixture("printer-cmyk-v4")));
        assert!(none(&ColorProfile::new_srgb()));
    }
}
