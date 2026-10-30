//! CIE-based `CalGray` and `CalRGB` colour spaces (PDF §8.6.5.2, §8.6.5.3).
//!
//! A colour value converts to CIE XYZ relative to the space's `WhitePoint`:
//!
//! - `CalGray`: `(X, Y, Z) = WhitePoint · A^G`;
//! - `CalRGB`: `(X, Y, Z) = Matrix · (A^GR, B^GG, C^GB)`, with `Matrix`
//!   holding the XYZ of the three components as columns.
//!
//! XYZ is then adapted from `WhitePoint` to D65 by the Bradford transform
//! (relative colorimetric: the space's white maps to sRGB white) and encoded
//! as sRGB.  The three linear steps compose into one matrix at resolve time.
//! `BlackPoint` is not applied.

use pdf::{Dictionary, Document, Object};

use crate::resources::icc_shaper::{Mat3, mat_inverse, mat_mul, srgb_encode};
use crate::resources::obj_to_f64;

/// CIE XYZ of the D65 white, `Y = 1`.
const D65: [f64; 3] = [0.950_47, 1.0, 1.088_83];

/// CIE XYZ (D65) → linear sRGB, IEC 61966-2-1.
const XYZ_D65_TO_LINEAR_SRGB: Mat3 = [
    [3.240_6, -1.537_2, -0.498_6],
    [-0.968_9, 1.875_8, 0.041_5],
    [0.055_7, -0.204_0, 1.057_0],
];

/// Bradford cone-response matrix.
const BRADFORD: Mat3 = [
    [0.895_1, 0.266_4, -0.161_4],
    [-0.750_2, 1.713_5, 0.036_7],
    [0.038_9, -0.068_5, 1.029_6],
];

/// A resolved `CalGray` (`n = 1`) or `CalRGB` (`n = 3`) space.
#[derive(Debug, Clone)]
pub struct CalSpace {
    n: u8,
    /// Per-component exponent; the three entries are equal for `CalGray`.
    gamma: [f64; 3],
    /// Gamma-decoded components → linear sRGB.
    to_linear_srgb: Mat3,
}

impl PartialEq for CalSpace {
    fn eq(&self, other: &Self) -> bool {
        let bits = |s: &Self| {
            (
                s.gamma.map(f64::to_bits),
                s.to_linear_srgb.map(|row| row.map(f64::to_bits)),
            )
        };
        self.n == other.n && bits(self) == bits(other)
    }
}

impl Eq for CalSpace {}

impl CalSpace {
    /// Resolve the parameter dictionary `params` of a `CalGray` (`n = 1`) or
    /// `CalRGB` (`n = 3`) array.  `None` when `WhitePoint` is missing or
    /// invalid (`Y ≠ 1`, or `X`, `Z` not positive), a `Gamma` entry is not a
    /// positive number, or the composed matrix is singular.
    pub(super) fn resolve(doc: &Document, params: &Object, n: u8) -> Option<Self> {
        let resolved;
        let dict = match params {
            Object::Reference(id) => {
                resolved = doc.get_object(*id).ok()?;
                resolved.as_ref().as_dict()?
            }
            other => other.as_dict()?,
        };
        Self::from_dict(dict, n)
    }

    fn from_dict(dict: &Dictionary, n: u8) -> Option<Self> {
        let white = numbers::<3>(dict.get(b"WhitePoint")?)?;
        #[expect(
            clippy::float_cmp,
            reason = "§8.6.5.2 requires Y of WhitePoint to be exactly 1"
        )]
        let valid_white = white[1] == 1.0 && white[0] > 0.0 && white[2] > 0.0;
        if !valid_white {
            log::debug!("colorspace: Cal* WhitePoint {white:?} invalid");
            return None;
        }
        let positive = |g: f64| (g.is_finite() && g > 0.0).then_some(g);
        let (gamma, matrix) = if n == 1 {
            let g = dict
                .get(b"Gamma")
                .map_or(Some(1.0), |o| obj_to_f64(o).and_then(positive))?;
            (
                [g; 3],
                [[white[0], 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, white[2]]],
            )
        } else {
            let gamma = match dict.get(b"Gamma") {
                Some(o) => numbers::<3>(o)?.map(positive),
                None => [Some(1.0); 3],
            };
            let gamma = [gamma[0]?, gamma[1]?, gamma[2]?];
            let matrix = match dict.get(b"Matrix") {
                // [XA YA ZA XB YB ZB XC YC ZC]: one column per component.
                Some(o) => {
                    let m = numbers::<9>(o)?;
                    [[m[0], m[3], m[6]], [m[1], m[4], m[7]], [m[2], m[5], m[8]]]
                }
                None => [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            };
            (gamma, matrix)
        };
        let to_linear_srgb = mat_mul(
            &mat_mul(&XYZ_D65_TO_LINEAR_SRGB, &bradford(white)?),
            &matrix,
        );
        if !to_linear_srgb.iter().flatten().all(|v| v.is_finite()) {
            return None;
        }
        Some(Self {
            n,
            gamma,
            to_linear_srgb,
        })
    }

    /// Component count: 1 for `CalGray`, 3 for `CalRGB`.
    #[must_use]
    pub const fn ncomponents(&self) -> usize {
        self.n as usize
    }

    /// Convert `comps` (each clipped to `[0, 1]`) to sRGB bytes.  The
    /// caller guarantees `comps.len() == self.ncomponents()`.
    pub(super) fn convert(&self, comps: &[f64]) -> [u8; 3] {
        let decode = |i: usize| {
            let c = comps[i.min(comps.len() - 1)];
            let c = if c.is_nan() { 0.0 } else { c.clamp(0.0, 1.0) };
            c.powf(self.gamma[i])
        };
        let abc = [decode(0), decode(1), decode(2)];
        let [r, g, b] = self
            .to_linear_srgb
            .map(|row| row[0].mul_add(abc[0], row[1].mul_add(abc[1], row[2] * abc[2])));
        color::convert::rgb_to_bytes(srgb_encode(r), srgb_encode(g), srgb_encode(b))
    }
}

/// The Bradford adaptation from white `src` to D65; `None` when singular.
fn bradford(src: [f64; 3]) -> Option<Mat3> {
    let cone =
        |w: [f64; 3]| BRADFORD.map(|row| row[0].mul_add(w[0], row[1].mul_add(w[1], row[2] * w[2])));
    let (s, d) = (cone(src), cone(D65));
    let scale = [
        [d[0] / s[0], 0.0, 0.0],
        [0.0, d[1] / s[1], 0.0],
        [0.0, 0.0, d[2] / s[2]],
    ];
    Some(mat_mul(
        &mat_inverse(&BRADFORD)?,
        &mat_mul(&scale, &BRADFORD),
    ))
}

/// Exactly `N` numbers from array `obj`.
fn numbers<const N: usize>(obj: &Object) -> Option<[f64; N]> {
    let arr = obj.as_array()?;
    if arr.len() != N {
        return None;
    }
    let mut out = [0.0; N];
    for (dst, o) in out.iter_mut().zip(arr) {
        *dst = obj_to_f64(o)?;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::colorspace::{ColorSpace, resolve};
    use crate::test_helpers::empty_doc;

    fn cal(family: &str, params: &[(&str, Object)]) -> ColorSpace {
        let mut dict = Dictionary::new();
        for (k, v) in params {
            dict.set(*k, v.clone());
        }
        let arr = Object::Array(vec![
            Object::Name(family.as_bytes().to_vec()),
            Object::Dictionary(dict),
        ]);
        resolve(&empty_doc(), &arr)
    }

    fn reals(v: &[f64]) -> Object {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "test values are exact in f32"
        )]
        Object::Array(v.iter().map(|&x| Object::Real(x as f32)).collect())
    }

    const D65_WP: [f64; 3] = [0.9505, 1.0, 1.089];
    const D50_WP: [f64; 3] = [0.9642, 1.0, 0.8249];
    const SRGB_MATRIX: [f64; 9] = [
        0.4124, 0.2126, 0.0193, 0.3576, 0.7152, 0.1192, 0.1805, 0.0722, 0.9505,
    ];

    fn rgb(space: &ColorSpace, c: [f64; 3]) -> [u8; 3] {
        space.convert_to_rgb(&empty_doc(), &c)
    }

    fn close(got: [u8; 3], want: [u8; 3], tol: u8) {
        assert!(
            got.iter().zip(want).all(|(&g, w)| g.abs_diff(w) <= tol),
            "got {got:?}, want {want:?} ±{tol}"
        );
    }

    #[test]
    fn srgb_parameters_match_the_device_space() {
        let space = cal(
            "CalRGB",
            &[
                ("WhitePoint", reals(&D65_WP)),
                ("Gamma", reals(&[2.2; 3])),
                ("Matrix", reals(&SRGB_MATRIX)),
            ],
        );
        assert_eq!(space.ncomponents(), 3);
        for c in [
            [0.0; 3],
            [1.0; 3],
            [0.5; 3],
            [1.0, 0.0, 0.0],
            [0.2, 0.6, 0.9],
        ] {
            close(
                rgb(&space, c),
                ColorSpace::DeviceRgb.convert_to_rgb(&empty_doc(), &c),
                3,
            );
        }
    }

    #[test]
    fn gamma_applies_to_each_component() {
        // 0.5^1.8 = 0.287 linear → sRGB 0.578 → 147, neutral.
        let space = cal(
            "CalRGB",
            &[
                ("WhitePoint", reals(&D65_WP)),
                ("Gamma", reals(&[1.8; 3])),
                ("Matrix", reals(&SRGB_MATRIX)),
            ],
        );
        close(rgb(&space, [0.5; 3]), [147; 3], 1);
    }

    #[test]
    fn white_point_is_adapted_so_neutrals_stay_neutral() {
        // sRGB primaries adapted to D50 (the IEC 61966-2-1 profile colorants).
        let d50_srgb = [
            0.4361, 0.2225, 0.0139, 0.3851, 0.7169, 0.0971, 0.1431, 0.0606, 0.7141,
        ];
        let space = cal(
            "CalRGB",
            &[
                ("WhitePoint", reals(&D50_WP)),
                ("Gamma", reals(&[2.2; 3])),
                ("Matrix", reals(&d50_srgb)),
            ],
        );
        close(rgb(&space, [1.0; 3]), [255; 3], 1);
        let mid = rgb(&space, [0.5; 3]);
        close(mid, [128; 3], 2);
        assert!(
            mid[0].abs_diff(mid[2]) <= 1,
            "neutral expected, got {mid:?}"
        );
    }

    #[test]
    fn calgray_applies_gamma_relative_to_its_white() {
        let space = cal(
            "CalGray",
            &[("WhitePoint", reals(&D50_WP)), ("Gamma", Object::Real(2.2))],
        );
        assert_eq!(space.ncomponents(), 1);
        close(space.convert_to_rgb(&empty_doc(), &[0.5]), [128; 3], 2);
        // Gamma 1 is linear light: 0.5 → sRGB 0.735 → 188.
        let linear = cal("CalGray", &[("WhitePoint", reals(&D65_WP))]);
        close(linear.convert_to_rgb(&empty_doc(), &[0.5]), [188; 3], 1);
    }

    #[test]
    fn invalid_parameters_fall_back_to_the_device_space() {
        assert_eq!(cal("CalRGB", &[]), ColorSpace::DeviceRgb);
        assert_eq!(
            cal("CalRGB", &[("WhitePoint", reals(&[0.9505, 0.5, 1.089]))]),
            ColorSpace::DeviceRgb
        );
        assert_eq!(
            cal(
                "CalGray",
                &[
                    ("WhitePoint", reals(&D65_WP)),
                    ("Gamma", Object::Integer(0))
                ]
            ),
            ColorSpace::DeviceGray
        );
        assert_eq!(
            cal(
                "CalRGB",
                &[("WhitePoint", reals(&D65_WP)), ("Matrix", reals(&[1.0; 4]))]
            ),
            ColorSpace::DeviceRgb
        );
    }
}
