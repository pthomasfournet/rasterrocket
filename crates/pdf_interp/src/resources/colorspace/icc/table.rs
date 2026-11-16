//! The conversion of a three- or four-channel profile's device → PCS
//! table to a working space, tabulated over the device cube.
//!
//! Each node of a grid of [`span`] cells along each device axis holds the
//! working-space value of its device value: the table evaluated at that
//! value in floating point ([`ProfileLut`], whose CLUT interpolates
//! tetrahedrally, linearly along the first of four inputs),
//! the media-relative PCS adjusted for the conversion's form
//! ([`PcsAdjustment`]), then the working space's PCS → linear matrix and
//! its sRGB curve, each linear component clipped to `[0, 1]`.  A pixel
//! converts by interpolation in the simplex of its grid cell
//! ([`GridTable`]), at 16 bits, so three-channel device values of equal
//! components take the nodes of equal components alone.

use moxcms::{ColorProfile, DataColorSpace, RenderingIntent};

use crate::WorkingSpace;
use crate::pcs::D50;
use crate::pcs::grid::GridTable;
use crate::pcs::lut::{Direction, ProfileLut};
use crate::resources::icc_shaper::srgb_encode;

/// The number of cells along each device axis of the grid of `n`
/// channels: 32 for three, 16 for four, whose grid of 17⁴ nodes is 2.3
/// times the 33³ of three.
const fn span(n: usize) -> u32 {
    if n == 3 { 32 } else { 16 }
}

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
/// of `N` channels, is a PCS, or the table is missing or malformed.
pub(super) fn tabulate<const N: usize>(
    profile: &ColorProfile,
    intent: RenderingIntent,
    adjustment: PcsAdjustment,
    space: WorkingSpace,
) -> Option<GridTable<N>> {
    use DataColorSpace::{Cmy, Cmyk, Color3, Color4, Hls, Hsv, Luv, Rgb, YCbr, Yxy};
    let channels = match profile.color_space {
        Rgb | Cmy | Hsv | Hls | YCbr | Yxy | Luv | Color3 => 3,
        Cmyk | Color4 => 4,
        _ => 0,
    };
    if channels != N {
        return None;
    }
    let lut = ProfileLut::new(profile, intent, Direction::ToPcs, N)?;
    let to_linear = space.pcs_to_linear();
    GridTable::tabulate(span(N), 3, |device, out| {
        for (px, o) in device
            .as_chunks::<N>()
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
                    let table = tabulate::<3>(&profile, intent, PcsAdjustment::None, space)
                        .expect("tabulates");
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

    /// A profile tabulates for the channel count of its device space
    /// alone, and a matrix profile not at all.
    #[test]
    fn a_profile_tabulates_for_its_own_channel_count() {
        let tabulates = |p: &ColorProfile, n: usize| {
            let args = (
                RenderingIntent::Perceptual,
                PcsAdjustment::None,
                WorkingSpace::Srgb,
            );
            match n {
                3 => tabulate::<3>(p, args.0, args.1, args.2).is_some(),
                _ => tabulate::<4>(p, args.0, args.1, args.2).is_some(),
            }
        };
        let cmyk = fixture("printer-cmyk-v4");
        assert!(tabulates(&cmyk, 4));
        assert!(!tabulates(&cmyk, 3));
        let rgb = fixture("printer-rgb-v4");
        assert!(tabulates(&rgb, 3));
        assert!(!tabulates(&rgb, 4));
        assert!(!tabulates(&ColorProfile::new_srgb(), 3));
    }
}
