//! Shaded fills — gradient patterns and Gouraud-shaded triangles.
//!
//! # Module layout (named for orientation; visibility varies)
//! - `axial`    — [`crate::AxialPattern`]: linear gradient along an axis vector
//! - `radial`   — [`crate::RadialPattern`]: gradient between two circles
//! - `function` — [`crate::FunctionPattern`]: PDF function sampled per pixel
//! - `gouraud`  — `gouraud_triangle_fill`: vertex-coloured triangle scan
//!
//! # Entry point
//!
//! [`shaded_fill`] is a thin wrapper around [`fill::fill`] / [`fill::eo_fill`]
//! that passes the pattern as a [`PipeSrc::Pattern`].  The gradient patterns
//! implement [`Pattern`] and are supplied by the caller.

pub mod axial;
pub mod function;
pub mod gouraud;
pub mod radial;

use crate::bitmap::Bitmap;
use crate::clip::Clip;
use crate::fill;
use crate::path::Path;
use crate::pipe::{Pattern, PipeSrc, PipeState};
use color::{Pixel, Sample};

/// The colours of a gradient at uniformly spaced positions of its
/// normalised parameter `s ∈ [0, 1]`.
///
/// Sample `k` of `n` is the colour at `s = k / (n − 1)`; between samples the
/// colour is the linear interpolation of its two neighbours, channel by
/// channel.  A colour is `N` samples of type `S`: sRGB by default, or the
/// channels of any other pixel type.  Shared by [`axial`] and [`radial`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColorRamp<const N: usize = 3, S: Sample = u8> {
    samples: Box<[[S; N]]>,
}

impl<const N: usize, S: Sample> ColorRamp<N, S> {
    /// The linear ramp from `c0` at `s = 0` to `c1` at `s = 1`.
    #[must_use]
    pub fn two(c0: [S; N], c1: [S; N]) -> Self {
        Self {
            samples: Box::new([c0, c1]),
        }
    }

    /// A ramp through `samples`, uniformly spaced over `s ∈ [0, 1]`.
    ///
    /// Samples that all lie within one level of the linear ramp between the
    /// first and last sample reduce to that two-sample ramp, which colours
    /// every position identically to [`Self::two`].  `None` when fewer than
    /// two samples are given.
    #[must_use]
    pub fn from_samples(samples: Vec<[S; N]>) -> Option<Self> {
        if samples.len() < 2 {
            return None;
        }
        let (first, last) = (samples[0], samples[samples.len() - 1]);
        let linear = Self::two(first, last);
        let segments = samples.len() - 1;
        let is_linear = samples.iter().enumerate().all(|(k, s)| {
            #[expect(
                clippy::cast_precision_loss,
                reason = "sample counts are far below 2^52"
            )]
            let pos = k as f64 / segments as f64;
            let mut lin = [S::default(); N];
            linear.write(pos, &mut lin);
            s.iter()
                .zip(lin)
                .all(|(&a, b)| (a.signed() - b.signed()).abs() <= 1)
        });
        Some(if is_linear {
            linear
        } else {
            Self {
                samples: samples.into_boxed_slice(),
            }
        })
    }

    /// Write the colour at normalised position `s` (clamped to `[0, 1]`;
    /// `NaN` reads as `0`) into `out[0..N]`.
    ///
    /// The position is quantised to `1 / S::LERP_ONE` of a segment (1/256
    /// for 8-bit samples, 1/65 536 for 16-bit ones) and the two neighbouring
    /// samples are blended with [`Sample::lerp`].
    #[inline]
    pub fn write(&self, s: f64, out: &mut [S]) {
        debug_assert_eq!(out.len(), N, "ColorRamp::write: out must be N samples");
        let one = S::LERP_ONE as usize;
        let segments = self.samples.len() - 1;
        #[expect(
            clippy::cast_precision_loss,
            reason = "sample counts are far below 2^52"
        )]
        let scaled = s.clamp(0.0, 1.0) * (one * segments) as f64;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "scaled ∈ [0, LERP_ONE·segments]; NaN saturates to 0"
        )]
        let fixed = scaled as usize;
        let idx = (fixed / one).min(segments - 1);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "fixed − LERP_ONE·idx ≤ LERP_ONE ≤ 2^16"
        )]
        let frac = (fixed - idx * one) as u32;
        let (a, b) = (self.samples[idx], self.samples[idx + 1]);
        for ((o, a), b) in out.iter_mut().zip(a).zip(b) {
            *o = S::lerp(a, b, frac);
        }
    }
}

/// Write the ramp colour of each pixel `x0..=x1` of a span into `out`.
///
/// `position(x)` is the pixel's ramp position, or `None` when the pixel is
/// not painted; such a pixel is written as zeros.  `mark(i, painted)` is
/// called once per pixel with its index in the span.  `out.len()` must be
/// `(x1 - x0 + 1) * N`.
#[inline]
fn fill_ramp_span<const N: usize, S: Sample>(
    ramp: &ColorRamp<N, S>,
    x0: i32,
    x1: i32,
    out: &mut [S],
    mut mark: impl FnMut(usize, bool),
    position: impl Fn(i32) -> Option<f64>,
) {
    for ((i, x), px) in (x0..=x1).enumerate().zip(out.as_chunks_mut::<N>().0) {
        if let Some(s) = position(x) {
            ramp.write(s, px);
            mark(i, true);
        } else {
            px.fill(S::default());
            mark(i, false);
        }
    }
}

/// Fill `path` using a shading pattern as the colour source.
///
/// The path defines the shading's bounding
/// shape; `pattern` supplies per-pixel colour via the [`Pattern`] trait, and
/// pixels it leaves unpainted keep their backdrop.
/// `eo` selects even-odd vs. non-zero winding rule.
/// The caller sets `pipe.a_input` to control fill/stroke opacity.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors shadedFill signature: bitmap+clip+path+pipe+pattern+matrix+flatness+aa+eo"
)]
pub fn shaded_fill<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    path: &Path,
    pipe: &PipeState<'_, P::Sample>,
    pattern: &dyn Pattern<P::Sample>,
    matrix: &[f64; 6],
    flatness: f64,
    vector_antialias: bool,
    eo: bool,
) {
    let src = PipeSrc::Pattern(pattern);
    if eo {
        fill::eo_fill::<P>(
            bitmap,
            clip,
            path,
            pipe,
            &src,
            matrix,
            flatness,
            vector_antialias,
        );
    } else {
        fill::fill::<P>(
            bitmap,
            clip,
            path,
            pipe,
            &src,
            matrix,
            flatness,
            vector_antialias,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitmap::Bitmap;
    use crate::clip::Clip;
    use crate::path::PathBuilder;
    use crate::shading::axial::AxialPattern;
    use crate::testutil::{identity_matrix, rect_path, simple_pipe};
    use color::Rgb8;

    #[test]
    fn shaded_fill_axial_paints_interior() {
        // 8×8 bitmap; fill (1,1)→(6,6) with a left→right gradient black→white.
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(8, 8, 4, false);
        let clip = Clip::new(0.0, 0.0, 7.999, 7.999, false);
        let pipe = simple_pipe();
        let path = rect_path(1.0, 1.0, 6.0, 6.0);

        let pattern = AxialPattern::new(
            ColorRamp::two([0u8, 0, 0], [255u8, 255, 255]),
            1.0,
            3.5,
            6.0,
            3.5,
            false,
            false,
        );

        shaded_fill::<Rgb8>(
            &mut bmp,
            &clip,
            &path,
            &pipe,
            &pattern,
            &identity_matrix(),
            1.0,
            false,
            false,
        );

        let r3 = bmp.row(3);
        assert!(r3[1].r < 60, "x=1 should be near-black (got {})", r3[1].r);
        assert!(r3[5].r > 180, "x=5 should be near-white (got {})", r3[5].r);
    }

    #[test]
    fn shaded_fill_eo_and_nonzero_both_work() {
        // Same rect path, eo=true vs eo=false — single contour, both should fill.
        let mut bmp_nz: Bitmap<Rgb8> = Bitmap::new(8, 8, 4, false);
        let mut bmp_eo: Bitmap<Rgb8> = Bitmap::new(8, 8, 4, false);
        let clip = Clip::new(0.0, 0.0, 7.999, 7.999, false);
        let pipe = simple_pipe();
        let path = rect_path(1.0, 1.0, 6.0, 6.0);
        let pattern = AxialPattern::new(
            ColorRamp::two([200u8, 0, 0], [200u8, 0, 0]),
            0.0,
            0.0,
            1.0,
            0.0,
            true,
            true,
        );
        shaded_fill::<Rgb8>(
            &mut bmp_nz,
            &clip,
            &path,
            &pipe,
            &pattern,
            &identity_matrix(),
            1.0,
            false,
            false,
        );
        shaded_fill::<Rgb8>(
            &mut bmp_eo,
            &clip,
            &path,
            &pipe,
            &pattern,
            &identity_matrix(),
            1.0,
            false,
            true,
        );
        assert_eq!(
            bmp_nz.row(3)[3].r,
            bmp_eo.row(3)[3].r,
            "non-zero and eo must agree for a simple convex path"
        );
    }

    /// Fill the whole 16 × 4 bitmap, pre-painted `[0, 0, 255]`, with an axial
    /// red → green gradient along x from 4 to 11 and return row 1.
    fn axial_over_blue(extend: bool, aa: bool) -> Vec<Rgb8> {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(16, 4, 4, false);
        for px in bmp.data_mut().as_chunks_mut::<3>().0 {
            *px = [0, 0, 255];
        }
        let clip = Clip::new(0.0, 0.0, 15.999, 3.999, aa);
        let pattern = AxialPattern::new(
            ColorRamp::two([255, 0, 0], [0, 255, 0]),
            4.0,
            0.0,
            11.0,
            0.0,
            extend,
            extend,
        );
        shaded_fill::<Rgb8>(
            &mut bmp,
            &clip,
            &rect_path(0.0, 0.0, 16.0, 4.0),
            &simple_pipe(),
            &pattern,
            &identity_matrix(),
            1.0,
            aa,
            false,
        );
        bmp.row(1).to_vec()
    }

    #[test]
    fn axial_without_extend_leaves_the_backdrop_outside_its_axis() {
        for aa in [false, true] {
            let row = axial_over_blue(false, aa);
            for (x, px) in row.iter().enumerate() {
                let rgb = [px.r, px.g, px.b];
                if (4..=11).contains(&x) {
                    assert_eq!(rgb[2], 0, "aa={aa} x={x} painted: {rgb:?}");
                } else {
                    assert_eq!(rgb, [0, 0, 255], "aa={aa} x={x} backdrop: {rgb:?}");
                }
            }
        }
    }

    #[test]
    fn axial_with_extend_paints_beyond_its_axis() {
        let row = axial_over_blue(true, true);
        assert_eq!([row[0].r, row[0].g, row[0].b], [255, 0, 0]);
        assert_eq!([row[15].r, row[15].g, row[15].b], [0, 255, 0]);
    }

    #[test]
    fn shaded_fill_empty_path_is_noop() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(8, 8, 4, false);
        let clip = Clip::new(0.0, 0.0, 7.999, 7.999, false);
        let pipe = simple_pipe();
        let path = PathBuilder::new().build();
        let pattern = AxialPattern::new(
            ColorRamp::two([255u8, 0, 0], [0u8, 255, 0]),
            0.0,
            0.0,
            8.0,
            0.0,
            false,
            false,
        );
        shaded_fill::<Rgb8>(
            &mut bmp,
            &clip,
            &path,
            &pipe,
            &pattern,
            &identity_matrix(),
            1.0,
            false,
            false,
        );
        assert_eq!(bmp.row(4)[4].r, 0, "empty path must not paint");
    }

    fn ramp_at(ramp: &ColorRamp, s: f64) -> [u8; 3] {
        let mut out = [0u8; 3];
        ramp.write(s, &mut out);
        out
    }

    #[test]
    fn ramp_needs_two_samples() {
        assert_eq!(ColorRamp::<3>::from_samples(vec![]), None);
        assert_eq!(ColorRamp::from_samples(vec![[1u8, 2, 3]]), None);
    }

    #[test]
    fn ramp_passes_through_every_sample() {
        // Red → green → blue: not linear between the ends.
        let ramp =
            ColorRamp::from_samples(vec![[255, 0, 0], [0, 255, 0], [0, 0, 255]]).expect("ramp");
        assert_eq!(ramp_at(&ramp, 0.0), [255, 0, 0]);
        assert_eq!(ramp_at(&ramp, 0.5), [0, 255, 0]);
        assert_eq!(ramp_at(&ramp, 1.0), [0, 0, 255]);
        let quarter = ramp_at(&ramp, 0.25);
        assert!(
            quarter[2] == 0 && (126..=129).contains(&quarter[0]),
            "{quarter:?}"
        );
    }

    #[test]
    fn ramp_clamps_out_of_range_and_nan() {
        let ramp = ColorRamp::two([10, 20, 30], [200, 210, 220]);
        assert_eq!(ramp_at(&ramp, -3.0), ramp_at(&ramp, 0.0));
        assert_eq!(ramp_at(&ramp, 7.0), ramp_at(&ramp, 1.0));
        assert_eq!(ramp_at(&ramp, f64::NAN), ramp_at(&ramp, 0.0));
    }

    #[test]
    fn linear_samples_reduce_to_the_two_sample_ramp() {
        // Exact line from (0, 0, 0) to (255, 128, 0), rounded per sample.
        let samples: Vec<[u8; 3]> = (0..=16_u16)
            .map(|k| {
                let g = u8::try_from((128 * k + 8) / 16).expect("≤ 128");
                [u8::try_from(255 * k / 16).expect("≤ 255"), g, 0]
            })
            .collect();
        assert_eq!(
            ColorRamp::from_samples(samples),
            Some(ColorRamp::two([0, 0, 0], [255, 128, 0]))
        );
    }
}
