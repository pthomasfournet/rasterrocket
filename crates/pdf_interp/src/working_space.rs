//! The working space: the RGB space a page's colours are composited in.
//!
//! [`WorkingSpace::Srgb`] is sRGB (IEC 61966-2-1).  [`WorkingSpace::Wide`]
//! has the primaries of ROMM RGB (ISO 22028-2), whose white is the D50 white
//! of the profile connection space (PCS), and the sRGB tone curve.  Both
//! spaces encode a linear component `l` as `encode(l)`, the sRGB curve, and
//! both map the linear `(1, 1, 1)` to the D50 white, so a neutral colour,
//! one of equal components, has the same encoded value in either.
//!
//! A colour converts to the wide space through the PCS: the linear wide
//! value of the D50 XYZ `X` is `M_w⁻¹ · X`, `M_w` the matrix whose columns
//! are the ROMM colorants.  An RGB matrix/TRC source of colorant matrix
//! `M_s` converts through the composite `M_w⁻¹ · M_s`
//! (`wide_composite`); when its rows sum to 1 within
//! `IDENTITY_MATRIX_TOLERANCE`, the source's white is the D50 white within
//! the precision of its encoding, and each row is scaled to sum to 1
//! exactly, so the source's white maps to the working white and its
//! neutrals to neutrals.
//!
//! The 16-bit conversion of a linearised source (`LinearRgb`) looks each
//! component up in a table of the linear value of every 16-bit level, forms
//! the matrix product in `f32`, and encodes the result by linear
//! interpolation in a table of the curve at `ENCODE_STEPS` + 1 points,
//! which is within 0.01 of a 16-bit level of the curve.  The conversion of
//! sRGB values (`from_srgb16`) is the one such conversion every sRGB
//! colour takes, a fill's components quantised to 16 bits first, so a fill
//! and an image of the same sRGB colour convert alike; it returns equal
//! components unchanged.

use std::sync::{Arc, OnceLock};

use color::Sample;
use moxcms::{ColorProfile, Xyzd};

use crate::resources::icc_shaper::{
    IDENTITY_MATRIX_TOLERANCE, Mat3, SRGB_D50_COLORANTS, mat_inverse, mat_mul, pcs_to_linear_srgb,
    srgb_encode,
};

/// The RGB space a page is composited in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum WorkingSpace {
    /// sRGB.
    #[default]
    Srgb,
    /// ROMM RGB primaries, D50 white, the sRGB tone curve.
    Wide,
}

/// The ROMM RGB colorants (ISO 22028-2) as columns; each row sums to the
/// D50 white of the PCS.
pub(crate) const ROMM_COLORANTS: Mat3 = [
    [0.7977, 0.1352, 0.0313],
    [0.2880, 0.7119, 0.0001],
    [0.0, 0.0, 0.8249],
];

/// The number of intervals of the encoding table of [`encode16`].
pub(crate) const ENCODE_STEPS: usize = 1 << 16;

impl WorkingSpace {
    /// Both spaces, in the order of [`Self::index`].
    pub const ALL: [Self; 2] = [Self::Srgb, Self::Wide];

    /// The position of the space in [`Self::ALL`].
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Srgb => 0,
            Self::Wide => 1,
        }
    }

    /// The space as a matrix/TRC profile described by its tags alone: a
    /// CICP description would let a transform evaluate the standard curve
    /// instead of the tags, which the adjustments of a conversion edit.
    #[must_use]
    pub fn profile(self) -> ColorProfile {
        let mut profile = ColorProfile::new_srgb();
        profile.cicp = None;
        if self == Self::Wide {
            let column = |c: usize| Xyzd {
                x: ROMM_COLORANTS[0][c],
                y: ROMM_COLORANTS[1][c],
                z: ROMM_COLORANTS[2][c],
            };
            profile.red_colorant = column(0);
            profile.green_colorant = column(1);
            profile.blue_colorant = column(2);
        }
        profile
    }

    /// The D50 PCS → linear component matrix of the space.
    pub(crate) fn pcs_to_linear(self) -> Mat3 {
        match self {
            Self::Srgb => pcs_to_linear_srgb(),
            Self::Wide => pcs_to_linear_wide(),
        }
    }
}

/// The composite `M_w⁻¹ · colorants` from the linear components of an RGB
/// source of colorant matrix `colorants` to linear wide components, its
/// rows scaled to sum to 1 when they do within
/// [`IDENTITY_MATRIX_TOLERANCE`].
pub(crate) fn wide_composite(colorants: &Mat3) -> Mat3 {
    composite(WorkingSpace::Wide, colorants)
}

/// The composite from the linear components of an RGB source of colorant
/// matrix `colorants` to the linear components of `space`, its rows
/// scaled to sum to 1 when they do within [`IDENTITY_MATRIX_TOLERANCE`].
pub(crate) fn composite(space: WorkingSpace, colorants: &Mat3) -> Mat3 {
    let composite = mat_mul(&space.pcs_to_linear(), colorants);
    let sums = composite.map(|row| row.iter().sum::<f64>());
    if sums
        .iter()
        .all(|s| (s - 1.0).abs() <= IDENTITY_MATRIX_TOLERANCE)
    {
        std::array::from_fn(|r| composite[r].map(|v| v / sums[r]))
    } else {
        composite
    }
}

/// The D50 PCS → linear wide matrix, `M_w⁻¹`.
pub(crate) fn pcs_to_linear_wide() -> Mat3 {
    mat_inverse(&ROMM_COLORANTS).expect("the ROMM colorant matrix is non-singular")
}

/// The sRGB curve at the `ENCODE_STEPS + 1` linear values `i /
/// ENCODE_STEPS`, in 16-bit levels.
fn encode_table() -> &'static [f32] {
    static TABLE: OnceLock<Box<[f32]>> = OnceLock::new();
    TABLE.get_or_init(|| {
        (0..=ENCODE_STEPS)
            .map(|i| {
                #[expect(clippy::cast_precision_loss, reason = "i ≤ 65 536, exact in f64")]
                let lin = i as f64 / ENCODE_STEPS as f64;
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "a 16-bit level needs no more than f32 precision"
                )]
                let level = (srgb_encode(lin) * 65_535.0) as f32;
                level
            })
            .collect()
    })
}

/// The 16-bit level of the linear component `lin`, clamped to `[0, 1]`.
pub(crate) fn encode16(lin: f32) -> u16 {
    let table = encode_table();
    #[expect(clippy::cast_precision_loss, reason = "ENCODE_STEPS is a power of two")]
    let x = lin.clamp(0.0, 1.0) * ENCODE_STEPS as f32;
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "x lies in [0, ENCODE_STEPS]"
    )]
    let i = (x as usize).min(ENCODE_STEPS - 1);
    #[expect(clippy::cast_precision_loss, reason = "i < 65 536, exact in f32")]
    let t = x - i as f32;
    let v = t.mul_add(table[i + 1] - table[i], table[i]);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "an interpolation of values in [0, 65 535]"
    )]
    let level = v.round() as u16;
    level
}

/// The luminance `Y` of the sRGB colour `rgb` (CIE 1931 XYZ, relative to
/// the white of the PCS): the `Y` row of the sRGB colorants, scaled to sum
/// to 1 so that white is 1, applied to its linear components.
pub(crate) fn srgb_luminance<S: Sample>(rgb: [S; 3]) -> S {
    let table = srgb_linear();
    let row = crate::resources::icc_shaper::SRGB_D50_COLORANTS[1];
    let sum: f64 = row.iter().sum();
    let y: f64 = rgb
        .iter()
        .zip(row)
        .map(|(v, w)| f64::from(table[usize::from(u16::from_sample(*v))]) * w)
        .sum();
    S::from_unit(y / sum)
}

/// The linear value of every 16-bit sRGB level.
pub(crate) fn srgb_linear() -> &'static Arc<[f32]> {
    static TABLE: OnceLock<Arc<[f32]>> = OnceLock::new();
    TABLE.get_or_init(|| {
        (0..=u16::MAX)
            .map(|v| {
                let lin = crate::resources::icc_shaper::srgb_decode(f64::from(v) / 65_535.0);
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "a linear-light value needs no more than f32 precision"
                )]
                let lin = lin as f32;
                lin
            })
            .collect()
    })
}

/// The 16-bit conversion of an RGB source to an RGB destination: each
/// component linearised by its table of every 16-bit level, then the
/// composite matrix, then the destination's encoding.
#[derive(Debug, PartialEq)]
pub(crate) struct LinearRgb {
    linear: [Arc<[f32]>; 3],
    matrix: [[f32; 3]; 3],
    encoding: Encoding,
}

/// The encoding of the linear components of a [`LinearRgb`]'s destination.
#[derive(Debug, PartialEq)]
enum Encoding {
    /// The sRGB curve ([`encode16`]).
    Srgb,
    /// The inverse of a curve given, per component, by the linear value of
    /// every 16-bit level, a non-decreasing table: a linear value encodes
    /// as the level at which the table reaches it, interpolated between
    /// the two levels whose values enclose it.
    Inverse([Arc<[f32]>; 3]),
}

impl Encoding {
    /// The 16-bit level of component `c` of the linear value `lin`.
    fn encode(&self, c: usize, lin: f32) -> u16 {
        match self {
            Self::Srgb => encode16(lin),
            Self::Inverse(tables) => invert16(&tables[c], lin),
        }
    }
}

/// The 16-bit level at which `table`, the non-decreasing linear value of
/// every 16-bit level, reaches `lin`, interpolated between the enclosing
/// levels; the first level at or above `lin` when they are equal, and the
/// end levels beyond the table's range.
fn invert16(table: &[f32], lin: f32) -> u16 {
    let above = table.partition_point(|&v| v < lin);
    let Some(&hi) = table.get(above) else {
        return u16::MAX;
    };
    let Some(below) = above.checked_sub(1) else {
        return 0;
    };
    let lo = table[below];
    let t = if hi > lo { (lin - lo) / (hi - lo) } else { 1.0 };
    #[expect(clippy::cast_precision_loss, reason = "a level is below 2^16")]
    let level = (below as f32 + t).round();
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the level lies in [below, below + 1] within [0, 65 535]"
    )]
    let level = level as u16;
    level
}

impl LinearRgb {
    /// The conversion of the linearisations `linear` (65 536 entries each)
    /// followed by the linear → linear-destination `matrix`, into a
    /// destination of the sRGB curve.
    pub(crate) fn new(linear: [Arc<[f32]>; 3], matrix: &Mat3) -> Self {
        Self::with_encoding(linear, matrix, Encoding::Srgb)
    }

    /// The conversion of the linearisations `linear` followed by `matrix`
    /// into a destination whose component `c` has the linear value
    /// `curves[c][v]` at level `v`, each table non-decreasing (see
    /// [`Self::table`]); `None` when a table decreases.
    pub(crate) fn into_curves(
        linear: [Arc<[f32]>; 3],
        matrix: &Mat3,
        curves: [Arc<[f32]>; 3],
    ) -> Option<Self> {
        curves
            .iter()
            .all(|t| t.len() == 1 << 16 && t.is_sorted())
            .then(|| Self::with_encoding(linear, matrix, Encoding::Inverse(curves)))
    }

    fn with_encoding(linear: [Arc<[f32]>; 3], matrix: &Mat3, encoding: Encoding) -> Self {
        debug_assert!(linear.iter().all(|t| t.len() == 1 << 16));
        #[expect(
            clippy::cast_possible_truncation,
            reason = "matrix coefficients need no more than f32 precision"
        )]
        let matrix = matrix.map(|row| row.map(|v| v as f32));
        Self {
            linear,
            matrix,
            encoding,
        }
    }

    /// The linearisation of every 16-bit level through `f`, the linear
    /// value of a normalised component; `None` when `f` fails.
    pub(crate) fn table(f: impl Fn(f64) -> Option<f64>) -> Option<Arc<[f32]>> {
        (0..=u16::MAX)
            .map(|v| {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "a linear-light value needs no more than f32 precision"
                )]
                f(f64::from(v) / 65_535.0).map(|l| l as f32)
            })
            .collect()
    }

    /// Convert one 16-bit pixel.
    pub(crate) fn convert(&self, px: [u16; 3]) -> [u16; 3] {
        let [r, g, b] = std::array::from_fn(|c| self.linear[c][usize::from(px[c])]);
        std::array::from_fn(|c| {
            let row = self.matrix[c];
            let lin = row[2].mul_add(b, row[0].mul_add(r, row[1] * g));
            self.encoding.encode(c, lin)
        })
    }
}

/// The conversion of 16-bit sRGB values to the wide space.
fn srgb_to_wide() -> &'static LinearRgb {
    static CONVERSION: OnceLock<LinearRgb> = OnceLock::new();
    CONVERSION.get_or_init(|| {
        let table = srgb_linear();
        LinearRgb::new(
            [Arc::clone(table), Arc::clone(table), Arc::clone(table)],
            &wide_composite(&SRGB_D50_COLORANTS),
        )
    })
}

/// The wide value of the 16-bit sRGB value `px`; equal components are
/// returned as they are.
pub(crate) fn from_srgb16(px: [u16; 3]) -> [u16; 3] {
    if px[0] == px[1] && px[1] == px[2] {
        return px;
    }
    srgb_to_wide().convert(px)
}

/// The conversion of 16-bit wide values to sRGB, the inverse of
/// [`srgb_to_wide`]; colours outside sRGB clip to its cube.
fn wide_to_srgb() -> &'static LinearRgb {
    static CONVERSION: OnceLock<LinearRgb> = OnceLock::new();
    CONVERSION.get_or_init(|| {
        let table = srgb_linear();
        let to_srgb = mat_inverse(&wide_composite(&SRGB_D50_COLORANTS))
            .expect("the sRGB to wide matrix is non-singular");
        LinearRgb::new(
            [Arc::clone(table), Arc::clone(table), Arc::clone(table)],
            &to_srgb,
        )
    })
}

/// The sRGB value of the 16-bit wide value `px`, clipped to the sRGB
/// cube; equal components are returned as they are.
#[must_use]
pub fn to_srgb16(px: [u16; 3]) -> [u16; 3] {
    if px[0] == px[1] && px[1] == px[2] {
        return px;
    }
    wide_to_srgb().convert(px)
}

/// The sRGB value at the depth of `S` of the colour of `space` value
/// `rgb`, at any depth: `rgb` itself in sRGB; in the wide space
/// [`to_srgb16`] of `rgb` at 16 bits, narrowed to `S`.
pub(crate) fn to_srgb<S: Sample, T: Sample>(space: WorkingSpace, rgb: [T; 3]) -> [S; 3] {
    match space {
        WorkingSpace::Srgb => rgb.map(S::from_sample),
        WorkingSpace::Wide => to_srgb16(rgb.map(u16::from_sample)).map(S::from_sample),
    }
}

/// Convert packed 16-bit sRGB pixels in place to the wide space.  A
/// trailing partial pixel is left untouched.
pub(crate) fn from_srgb16_in_place(samples: &mut [u16]) {
    for px in samples.as_chunks_mut::<3>().0 {
        *px = from_srgb16(*px);
    }
}

/// The value at the depth of `S` of the colour of sRGB value `rgb`, at any
/// depth, in `space`: `rgb` itself in sRGB; in the wide space
/// [`from_srgb16`] of `rgb` at 16 bits, narrowed to `S`.
pub(crate) fn from_srgb<S: Sample, T: Sample>(space: WorkingSpace, rgb: [T; 3]) -> [S; 3] {
    match space {
        WorkingSpace::Srgb => rgb.map(S::from_sample),
        WorkingSpace::Wide => from_srgb16(rgb.map(u16::from_sample)).map(S::from_sample),
    }
}

/// The wide value at the depth of `S` of the linear wide components `lin`.
pub(crate) fn encode_wide<S: Sample>(lin: [f64; 3]) -> [S; 3] {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a linear-light value needs no more than f32 precision"
    )]
    lin.map(|l| S::from_sample(encode16(l as f32)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_luminance_of_white_is_one_and_of_a_mid_gray_its_linear_value() {
        assert_eq!(srgb_luminance([u16::MAX; 3]), u16::MAX);
        assert_eq!(srgb_luminance([255u8; 3]), 255);
        assert_eq!(srgb_luminance([0u8; 3]), 0);
        // sRGB 128 is linear 0.2158.
        assert_eq!(srgb_luminance([128u8; 3]), 55);
        // Green carries most of the luminance, blue the least.
        assert!(srgb_luminance([0u8, 255, 0]) > srgb_luminance([255u8, 0, 0]));
        assert!(srgb_luminance([255u8, 0, 0]) > srgb_luminance([0u8, 0, 255]));
    }

    #[test]
    fn the_romm_white_is_the_pcs_white() {
        for (row, d50) in ROMM_COLORANTS.iter().zip(crate::pcs::D50) {
            assert!((row.iter().sum::<f64>() - d50).abs() < 1e-12);
        }
    }

    #[test]
    fn neutrals_keep_their_value() {
        for v in 0..=u16::MAX {
            assert_eq!(from_srgb16([v; 3]), [v; 3]);
            // Through the matrix, without the shortcut.
            let got = srgb_to_wide().convert([v; 3]);
            assert!(got.iter().all(|&c| c.abs_diff(v) <= 1), "{v}: {got:?}");
        }
    }

    #[test]
    fn the_encoding_follows_the_curve() {
        for i in 0..=4096 {
            let lin = f64::from(i) / 4096.0;
            let want = srgb_encode(lin) * 65_535.0;
            #[expect(clippy::cast_possible_truncation, reason = "test value")]
            let got = f64::from(encode16(lin as f32));
            assert!((got - want).abs() <= 0.51, "{lin}: {got} vs {want}");
        }
    }

    /// The round trip holds within 32 counts of 65 535, an eighth of an
    /// 8-bit level: the wide value is rounded to 16 bits, and a channel
    /// near black converts back through the sRGB curve's slope of 12.92.
    #[test]
    fn srgb_round_trips_through_the_wide_space() {
        for r in (0..=65_535u32).step_by(4369) {
            for g in (0..=65_535u32).step_by(4369) {
                for b in (0..=65_535u32).step_by(4369) {
                    let px = [r, g, b].map(|v| u16::try_from(v).unwrap());
                    let back = to_srgb16(from_srgb16(px));
                    let off = px.iter().zip(back).map(|(a, b)| a.abs_diff(b)).max();
                    assert!(off <= Some(32), "{px:?} → {back:?}");
                }
            }
        }
    }

    #[test]
    fn an_inverse_encoding_finds_the_level_of_a_linear_value() {
        let table = LinearRgb::table(|x| Some(x * x)).expect("evaluates");
        for v in [0u16, 1, 255, 30_000, 65_534, 65_535] {
            assert_eq!(invert16(&table, table[usize::from(v)]), v);
        }
        assert_eq!(invert16(&table, -1.0), 0);
        assert_eq!(invert16(&table, 2.0), u16::MAX);
        // Halfway between two levels' values rounds to the nearer level.
        let mid = f32::midpoint(table[1000], table[1001]);
        assert!([1000, 1001].contains(&invert16(&table, mid)));
    }

    #[test]
    fn a_decreasing_curve_has_no_inverse_encoding() {
        let rising = LinearRgb::table(Some).expect("evaluates");
        let falling = LinearRgb::table(|x| Some(1.0 - x)).expect("evaluates");
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let curves = |t: &Arc<[f32]>| [Arc::clone(t), Arc::clone(t), Arc::clone(t)];
        assert!(LinearRgb::into_curves(curves(&rising), &identity, curves(&falling)).is_none());
        let linear = LinearRgb::into_curves(curves(&rising), &identity, curves(&rising))
            .expect("a rising curve inverts");
        assert_eq!(linear.convert([0, 12_345, 65_535]), [0, 12_345, 65_535]);
    }

    #[test]
    fn a_wide_colour_outside_srgb_clips() {
        let green = to_srgb16([0, 65_535, 0]);
        assert_eq!(green[1], 65_535);
        assert_eq!(green[0], 0, "{green:?}");
    }

    #[test]
    fn a_saturated_srgb_colour_lies_inside_the_wide_cube() {
        let red = from_srgb16([65_535, 0, 0]);
        assert!(red[0] < 65_535 && red[1] > 0 && red[2] > 0, "{red:?}");
    }
}
