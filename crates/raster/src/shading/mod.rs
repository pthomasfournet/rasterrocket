//! Shaded fills — gradient patterns and Gouraud-shaded triangles.
//!
//! Replaces `Splash::shadedFill` and `Splash::gouraudTriangleShadedFill`.
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
//! implement [`Pattern`] and are injected by the caller (typically `pdf_bridge`).

pub mod axial;
pub mod function;
pub mod gouraud;
pub mod radial;

use crate::bitmap::Bitmap;
use crate::clip::Clip;
use crate::fill;
use crate::path::Path;
use crate::pipe::{Pattern, PipeSrc, PipeState};
use color::Pixel;
use color::convert::lerp_u8;

/// The colours of a gradient at uniformly spaced positions of its
/// normalised parameter `s ∈ [0, 1]`.
///
/// Sample `k` of `n` is the colour at `s = k / (n − 1)`; between samples the
/// colour is the linear interpolation of its two neighbours.  Shared by
/// [`axial`] and [`radial`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColorRamp {
    samples: Box<[[u8; 3]]>,
}

impl ColorRamp {
    /// The linear ramp from `c0` at `s = 0` to `c1` at `s = 1`.
    #[must_use]
    pub fn two(c0: [u8; 3], c1: [u8; 3]) -> Self {
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
    pub fn from_samples(samples: Vec<[u8; 3]>) -> Option<Self> {
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
            let mut lin = [0u8; 3];
            linear.write(pos, &mut lin);
            s.iter().zip(lin).all(|(&a, b)| a.abs_diff(b) <= 1)
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
    /// `NaN` reads as `0`) into `out[0..3]`.
    ///
    /// The position is quantised to 1/256 of a segment and the two
    /// neighbouring samples are blended with [`lerp_u8`].
    #[inline]
    pub fn write(&self, s: f64, out: &mut [u8]) {
        debug_assert_eq!(out.len(), 3, "ColorRamp::write: out must be 3 bytes");
        let segments = self.samples.len() - 1;
        #[expect(
            clippy::cast_precision_loss,
            reason = "sample counts are far below 2^52"
        )]
        let scaled = s.clamp(0.0, 1.0) * (256 * segments) as f64;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "scaled ∈ [0, 256·segments]; NaN saturates to 0"
        )]
        let fixed = scaled as usize;
        let idx = (fixed / 256).min(segments - 1);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "fixed − 256·idx ∈ [0, 256]"
        )]
        let frac = (fixed - idx * 256) as u32;
        let (a, b) = (self.samples[idx], self.samples[idx + 1]);
        out[0] = lerp_u8(a[0], b[0], frac);
        out[1] = lerp_u8(a[1], b[1], frac);
        out[2] = lerp_u8(a[2], b[2], frac);
    }
}

/// Fill `path` using a shading pattern as the colour source.
///
/// Equivalent to `Splash::shadedFill`.  The path defines the shading's bounding
/// shape; `pattern` supplies per-pixel colour via the [`Pattern`] trait.
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
    pipe: &PipeState<'_>,
    pattern: &dyn Pattern,
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
            0.0,
            1.0,
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
            0.0,
            1.0,
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
            0.0,
            1.0,
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
        assert_eq!(ColorRamp::from_samples(vec![]), None);
        assert_eq!(ColorRamp::from_samples(vec![[1, 2, 3]]), None);
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
