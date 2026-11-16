//! The conversion of a three- or four-channel profile's device → PCS
//! table to a working space, tabulated over the device cube.
//!
//! Each node of a grid over the device cube holds the working-space value
//! of its device value: the table evaluated at that value in floating
//! point ([`ProfileLut`], whose CLUT interpolates tetrahedrally, linearly
//! along the first of four inputs), the media-relative PCS adjusted for the
//! conversion's form ([`PcsAdjustment`]), then the working space's PCS →
//! linear matrix and its sRGB curve extended beyond `[0, 1]`
//! ([`srgb_encode_extended`]), held in [`NodeRange::Extended`].  A pixel
//! converts by interpolation in the simplex of its grid cell
//! ([`GridTable`]), at 16 bits, clipped to `[0, 1]` after it, so that a
//! colour leaving the working space's gamut within a cell is clipped
//! where it leaves it; three-channel device values of equal components
//! take the nodes of equal components alone.
//!
//! The table is piecewise linear in its CLUT's cells, which lie between
//! the nodes of its CLUT along the outputs of its input curves.  The
//! grid's cells along each input subdivide the CLUT's ([`spans`]), so that
//! where the input curves are the identity no cell of the grid straddles
//! a node of the CLUT: along an edge of the CLUT the grid reproduces the
//! table, and within a cell of it the grid interpolates a function
//! without kinks.  A four-channel table whose input curves are not the
//! identity is tabulated a second time over the curves' outputs, the
//! curves being the grid's shaper ([`ProfileLut::input_curves`]), where
//! the same holds, and the grid of the two that differs less from the
//! table at a fixed set of device values converts ([`probe_error`]): a
//! curve that flattens, as a gamma curve does near 0, leaves the cells of
//! the shaped grid wide in the device values it spans.  A three-channel
//! grid stays over the device cube.

use moxcms::{ColorProfile, DataColorSpace, RenderingIntent};

use crate::WorkingSpace;
use crate::pcs::D50;
use crate::pcs::grid::{GridTable, NodeRange};
use crate::pcs::lut::{Direction, ProfileLut, lab_encoding};
use crate::resources::icc_shaper::{Mat3, mat_inverse, srgb_decode, srgb_encode_extended};

/// The number of cells along each device axis a grid of `n` channels
/// approaches: 32 for three, 16 for four, whose grid of 17⁴ nodes is 2.3
/// times the 33³ of three.
const fn target_span(n: usize) -> u32 {
    if n == 3 { 32 } else { 16 }
}

/// The number of cells of the grid of `n` channels along each input of a
/// CLUT of `grid[i]` nodes along input `i`: the multiple of the CLUT's
/// cells nearest [`target_span`], at least one, while it is at most twice
/// the target (a four-channel grid of 33⁴ nodes at most); [`target_span`]
/// beyond it, or without a CLUT.
fn spans<const N: usize>(grid: Option<&[usize]>) -> [u32; N] {
    let target = target_span(N);
    std::array::from_fn(|i| {
        let Some(cells) = grid
            .and_then(|g| g.get(i))
            .and_then(|&g| u32::try_from(g - 1).ok())
        else {
            return target;
        };
        let multiple = ((target + cells / 2) / cells).max(1) * cells;
        if multiple <= 2 * target {
            multiple
        } else {
            target
        }
    })
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
    let encode = |pcs: [f64; 3]| {
        let xyz = adjustment.apply(pcs);
        to_linear
            .map(|[a, b, c]| srgb_encode_extended(a.mul_add(xyz[0], b.mul_add(xyz[1], c * xyz[2]))))
    };
    let grid = |shaper: Option<[&[f64]; N]>| {
        let shaped = shaper.is_some();
        GridTable::tabulate_shaped(
            spans::<N>(lut.clut_grid()),
            shaper,
            NodeRange::Extended,
            3,
            |values, out| {
                for (px, o) in values
                    .as_chunks::<N>()
                    .0
                    .iter()
                    .zip(out.as_chunks_mut::<3>().0)
                {
                    *o = encode(if shaped {
                        lut.to_pcs_shaped(px)
                    } else {
                        lut.to_pcs(px)
                    });
                }
                Ok::<(), std::convert::Infallible>(())
            },
        )
        .ok()
    };
    let plain = grid(None)?;
    // Three equal components shape to three equal values only under three
    // equal curves, so a shaped three-channel grid would lose the
    // diagonal of neutrals.
    let Some(curves) = lut.input_curves().filter(|_| N == 4) else {
        return Some(plain);
    };
    let shaped = grid(Some(std::array::from_fn(|i| &*curves[i])))?;
    let exact = |px: &[f64]| encode(lut.to_pcs(px));
    let to_pcs = mat_inverse(&to_linear)?;
    let error = |table: &GridTable<N>| probe_error(table, exact, &to_pcs);
    Some(if error(&shaped) < error(&plain) {
        shaped
    } else {
        plain
    })
}

/// The number of device values at which [`probe_error`] compares a grid
/// with the table.
const PROBES: usize = 4096;

/// The sum of the CIE 1976 colour differences between `table`'s 16-bit
/// values and `exact`'s, working-space values clipped to `[0, 1]` whose
/// linear components `to_pcs` maps to the PCS, at [`PROBES`] device values spread
/// over the device cube by a fixed pseudo-random sequence.
fn probe_error<const N: usize>(
    table: &GridTable<N>,
    exact: impl Fn(&[f64]) -> [f64; 3],
    to_pcs: &Mat3,
) -> f64 {
    let lab = |encoded: [f64; 3]| {
        let linear = encoded.map(srgb_decode);
        let xyz = to_pcs.map(|[a, b, c]| a.mul_add(linear[0], b.mul_add(linear[1], c * linear[2])));
        let [l, a, b] = lab_encoding(xyz);
        [l * 100.0, a * 255.0, b * 255.0]
    };
    let mut state = 0x2545_f491_u32;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        #[expect(clippy::cast_possible_truncation, reason = "the low 16 bits")]
        let v = state as u16;
        v
    };
    (0..PROBES)
        .map(|_| {
            let px: [u16; N] = std::array::from_fn(|_| next());
            let got: [u16; 3] = table.convert(px);
            let got = lab(got.map(|v| f64::from(v) / f64::from(u16::MAX)));
            let want = lab(exact(&px.map(|v| f64::from(v) / f64::from(u16::MAX))));
            got.iter()
                .zip(want)
                .map(|(g, w)| (g - w).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .sum()
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

    /// The version 2 RGB test printer's input tables are gamma-like
    /// curves, yet its grid lies over the device cube, keeping the
    /// diagonal of neutrals.
    #[test]
    fn a_three_channel_grid_is_never_shaped() {
        let profile = fixture("printer-rgb-v2");
        let lut = ProfileLut::new(
            &profile,
            RenderingIntent::RelativeColorimetric,
            Direction::ToPcs,
            3,
        )
        .expect("a table");
        assert!(lut.input_curves().is_some());
        let table = tabulate::<3>(
            &profile,
            RenderingIntent::RelativeColorimetric,
            PcsAdjustment::None,
            WorkingSpace::Srgb,
        )
        .expect("tabulates");
        assert!(!table.is_shaped());
    }

    /// A grid's cells subdivide the CLUT's along each input, as near the
    /// target span as a whole number of them gives, up to twice it.
    #[test]
    fn the_grid_subdivides_the_clut_cells() {
        assert_eq!(spans::<4>(Some(&[11, 9, 16, 17])), [20, 16, 15, 16]);
        assert_eq!(spans::<4>(Some(&[33, 34, 3, 2])), [32, 16, 16, 16]);
        assert_eq!(spans::<4>(None), [16; 4]);
        assert_eq!(spans::<3>(Some(&[17, 11, 33])), [32, 30, 32]);
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
