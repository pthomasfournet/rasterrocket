//! CIE 1976 `L*a*b*` colour space (PDF §8.6.5.4).
//!
//! A colour value `(L*, a*, b*)` has `a*` and `b*` clipped to `Range` and
//! `L*` to `[0, 100]` (components outside the range are adjusted to the
//! nearest valid value), then converts to CIE XYZ relative to the space's
//! `WhitePoint` by the CIE 1976 formulae.  XYZ is adapted from `WhitePoint`
//! to D65 by the Bradford transform and encoded as sRGB, as for `CalRGB`;
//! the two linear steps compose into one matrix at resolve time.
//! `BlackPoint` is not applied.

use pdf::{Document, Object};

use super::cal::{D65, XYZ_D65_TO_LINEAR_SRGB, bradford, numbers};
use crate::resources::icc_shaper::{Mat3, mat_mul, srgb_encode};

/// The default `Range`: `a*` and `b*` in `[-100, 100]`.
const DEFAULT_RANGE: [f64; 4] = [-100.0, 100.0, -100.0, 100.0];

/// A resolved `Lab` space.
#[derive(Debug, Clone)]
pub struct LabSpace {
    /// CIE XYZ of the diffuse white, `Y = 1`.
    white: [f64; 3],
    /// `[amin amax bmin bmax]`.
    range: [f64; 4],
    /// XYZ relative to `white` → linear sRGB.
    to_linear_srgb: Mat3,
}

impl PartialEq for LabSpace {
    fn eq(&self, other: &Self) -> bool {
        let bits = |s: &Self| {
            (
                s.white.map(f64::to_bits),
                s.range.map(f64::to_bits),
                s.to_linear_srgb.map(|row| row.map(f64::to_bits)),
            )
        };
        bits(self) == bits(other)
    }
}

impl Eq for LabSpace {}

impl LabSpace {
    /// Resolve the parameter dictionary `params` of a `[/Lab dict]` array.
    /// A missing or invalid `WhitePoint` (`Y ≠ 1`, or `X`, `Z` not
    /// positive) is D65; a missing or invalid `Range` (not four numbers
    /// with `amin ≤ amax` and `bmin ≤ bmax`) is `[-100 100 -100 100]`.
    pub(super) fn resolve(doc: &Document, params: Option<&Object>) -> Self {
        let resolved;
        let dict = match params {
            Some(Object::Reference(id)) => {
                resolved = doc.get_object(*id).ok();
                resolved.as_ref().and_then(|o| o.as_ref().as_dict())
            }
            Some(other) => other.as_dict(),
            None => None,
        };
        let white = dict
            .and_then(|d| d.get(b"WhitePoint"))
            .and_then(numbers::<3>)
            .filter(|w| {
                #[expect(
                    clippy::float_cmp,
                    reason = "§8.6.5.4 requires Y of WhitePoint to be exactly 1"
                )]
                let valid = w[1] == 1.0 && w[0] > 0.0 && w[2] > 0.0;
                valid
            })
            .unwrap_or(D65);
        let range = dict
            .and_then(|d| d.get(b"Range"))
            .and_then(numbers::<4>)
            .filter(|r| r.iter().all(|v| v.is_finite()) && r[0] <= r[1] && r[2] <= r[3])
            .unwrap_or(DEFAULT_RANGE);
        Self::new(white, range)
    }

    fn new(white: [f64; 3], range: [f64; 4]) -> Self {
        let adapt = bradford(white).unwrap_or_else(|| {
            bradford(D65).expect("the Bradford adaptation of D65 to itself is the identity")
        });
        Self {
            white,
            range,
            to_linear_srgb: mat_mul(&XYZ_D65_TO_LINEAR_SRGB, &adapt),
        }
    }

    /// `[amin amax bmin bmax]`: the valid range of `a*` and `b*`.
    #[must_use]
    pub const fn range(&self) -> [f64; 4] {
        self.range
    }

    /// Convert `(L*, a*, b*)` to sRGB bytes, each component clipped to its
    /// range.  The caller guarantees `comps.len() == 3`.
    pub(super) fn convert(&self, comps: &[f64]) -> [u8; 3] {
        let finite = |v: f64| if v.is_nan() { 0.0 } else { v };
        let l_star = finite(comps[0]).clamp(0.0, 100.0);
        let a_star = finite(comps[1]).clamp(self.range[0], self.range[1]);
        let b_star = finite(comps[2]).clamp(self.range[2], self.range[3]);
        let fy = (l_star + 16.0) / 116.0;
        let fx = a_star / 500.0 + fy;
        let fz = fy - b_star / 200.0;
        let xyz = [
            self.white[0] * f_inv(fx),
            self.white[1] * f_inv(fy),
            self.white[2] * f_inv(fz),
        ];
        let [r, g, b] = self
            .to_linear_srgb
            .map(|row| row[0].mul_add(xyz[0], row[1].mul_add(xyz[1], row[2] * xyz[2])));
        color::convert::rgb_to_bytes(srgb_encode(r), srgb_encode(g), srgb_encode(b))
    }
}

/// Inverse of the CIE Lab `f` function: `δ = 6/29`; `t > δ` cubes, else the
/// linear segment.
fn f_inv(t: f64) -> f64 {
    const DELTA: f64 = 6.0 / 29.0;
    if t > DELTA {
        t * t * t
    } else {
        3.0 * DELTA * DELTA * (t - 4.0 / 29.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::colorspace::{ColorSpace, resolve};
    use crate::test_helpers::empty_doc;

    fn lab(src: &str) -> ColorSpace {
        resolve(
            &empty_doc(),
            &pdf::parse_object(src.as_bytes(), &mut 0).expect("parses"),
        )
    }

    fn rgb(space: &ColorSpace, c: [f64; 3]) -> [u8; 3] {
        space.convert_to_rgb(&empty_doc(), &c)
    }

    const D65_LAB: &str = "[/Lab << /WhitePoint [0.9505 1 1.089] >>]";

    #[test]
    fn black_and_white_map_to_the_srgb_extremes() {
        let space = lab(D65_LAB);
        assert_eq!(rgb(&space, [0.0, 0.0, 0.0]), [0, 0, 0]);
        let white = rgb(&space, [100.0, 0.0, 0.0]);
        assert!(white.iter().all(|&c| c >= 253), "got {white:?}");
    }

    #[test]
    fn neutrals_stay_neutral_under_any_white_point() {
        for src in [D65_LAB, "[/Lab << /WhitePoint [0.9642 1 0.8249] >>]"] {
            let mid = rgb(&lab(src), [50.0, 0.0, 0.0]);
            assert!(
                mid.iter().all(|&c| c.abs_diff(119) <= 1),
                "{src}: got {mid:?}"
            );
        }
    }

    #[test]
    fn white_point_changes_the_chroma_of_a_coloured_value() {
        let d65 = rgb(&lab(D65_LAB), [60.0, 40.0, -30.0]);
        let d50 = rgb(
            &lab("[/Lab << /WhitePoint [0.9642 1 0.8249] >>]"),
            [60.0, 40.0, -30.0],
        );
        assert_ne!(d65, d50);
    }

    #[test]
    fn range_clips_a_and_b() {
        let narrow = lab("[/Lab << /WhitePoint [0.9505 1 1.089] /Range [-50 50 -50 50] >>]");
        assert_eq!(
            rgb(&narrow, [50.0, 100.0, 0.0]),
            rgb(&narrow, [50.0, 50.0, 0.0])
        );
        let ColorSpace::Lab(space) = &narrow else {
            panic!("Lab expected");
        };
        assert_eq!(space.range(), [-50.0, 50.0, -50.0, 50.0]);
        assert_eq!(
            narrow.default_decode(8),
            vec![(0.0, 100.0), (-50.0, 50.0), (-50.0, 50.0)]
        );
    }

    #[test]
    fn invalid_parameters_take_the_defaults() {
        let bare = lab("[/Lab]");
        let defaults = lab("[/Lab << /WhitePoint [0.5 0.5 0.5] /Range [1 0 0 1] >>]");
        assert_eq!(bare, defaults);
        let ColorSpace::Lab(space) = &bare else {
            panic!("Lab expected");
        };
        assert_eq!(space.range(), DEFAULT_RANGE);
        assert_eq!(
            rgb(&bare, [50.0, 0.0, 0.0]),
            rgb(&lab(D65_LAB), [50.0, 0.0, 0.0])
        );
    }
}
