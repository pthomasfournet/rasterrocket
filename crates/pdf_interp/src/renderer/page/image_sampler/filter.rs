//! Choice of the axes a [`SampleGrid`](super::SampleGrid) interpolates.
//!
//! The filter is a function of the draw alone: the image's dimensions, the
//! matrix it is drawn through and its `/Interpolate` flag.  The CPU sampler
//! and the GPU blit kernel read the same [`Filter`].

use super::super::image_reduce::{AXIS_TOLERANCE, REDUCTION_MARGIN};
use crate::renderer::gstate::Ctm;

/// Magnification (device pixels per sample) at or above which an image that
/// does not request interpolation keeps its samples as uniform blocks.
const MAGNIFICATION_LIMIT: f64 = 4.0;

/// Which image axes a [`SampleGrid`](super::SampleGrid) interpolates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::renderer::page) struct Filter {
    /// Interpolate between source columns.
    pub x: bool,
    /// Interpolate between source rows.
    pub y: bool,
}

impl Filter {
    /// Nearest-neighbour sampling on both axes.
    pub(in crate::renderer::page) const NEAREST: Self = Self { x: false, y: false };

    /// The filter for a `width × height` image drawn through `ctm`, whose
    /// dictionary sets `/Interpolate` to `interpolate`.
    ///
    /// An image drawn at or above [`MAGNIFICATION_LIMIT`] on either axis is
    /// interpolated only when it requests it.  Otherwise a rotated or
    /// skewed draw is interpolated on both axes, and a draw whose image axes
    /// map onto the device axes on each axis it magnifies: an axis drawn at
    /// one sample per device pixel or fewer (an image reduced for the draw
    /// is resampled to one per pixel first) keeps its samples.
    pub(in crate::renderer::page) fn for_draw(
        ctm: &Ctm,
        width: u32,
        height: u32,
        interpolate: bool,
    ) -> Self {
        let [a, b, c, d, _, _] = *ctm;
        let mag_x = a.hypot(b) / f64::from(width);
        let mag_y = c.hypot(d) / f64::from(height);
        if !(mag_x.is_finite() && mag_y.is_finite()) {
            return Self::NEAREST;
        }
        if !interpolate && mag_x.max(mag_y) >= MAGNIFICATION_LIMIT {
            return Self::NEAREST;
        }
        if !is_axis_aligned(ctm) {
            return Self { x: true, y: true };
        }
        Self {
            x: magnifies(mag_x),
            y: magnifies(mag_y),
        }
    }

    /// The blit kernel's encoding: [`gpu::blit::BLIT_FILTER_X`] and
    /// [`gpu::blit::BLIT_FILTER_Y`] bits.
    #[cfg(feature = "cache")]
    pub(in crate::renderer::page) const fn bits(self) -> u32 {
        use gpu::blit::{BLIT_FILTER_X, BLIT_FILTER_Y};
        (if self.x { BLIT_FILTER_X } else { 0 }) | (if self.y { BLIT_FILTER_Y } else { 0 })
    }
}

/// `true` when `ctm` maps the image axes onto the device axes, in either
/// order, to within [`AXIS_TOLERANCE`].
fn is_axis_aligned(ctm: &Ctm) -> bool {
    let [a, b, c, d, _, _] = *ctm;
    (b.abs() <= AXIS_TOLERANCE && c.abs() <= AXIS_TOLERANCE)
        || (a.abs() <= AXIS_TOLERANCE && d.abs() <= AXIS_TOLERANCE)
}

/// `true` when an axis drawn at `mag` device pixels per sample magnifies
/// beyond the reduction margin.
fn magnifies(mag: f64) -> bool {
    mag > 1.0 + REDUCTION_MARGIN
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_selection() {
        let both = Filter { x: true, y: true };
        let draw = |ctm: Ctm, interpolate| Filter::for_draw(&ctm, 10, 10, interpolate);
        // One sample per pixel or fewer keeps the samples.
        assert_eq!(
            draw([10.0, 0.0, 0.0, 10.0, 0.0, 0.0], false),
            Filter::NEAREST
        );
        assert_eq!(draw([5.0, 0.0, 0.0, -5.0, 0.0, 5.0], true), Filter::NEAREST);
        // Below the magnification limit, each magnified axis blends.
        assert_eq!(draw([20.0, 0.0, 0.0, 20.0, 0.0, 0.0], false), both);
        assert_eq!(
            draw([20.0, 0.0, 0.0, 10.0, 0.0, 0.0], false),
            Filter { x: true, y: false }
        );
        // A quarter turn magnifying the image's rows only.
        assert_eq!(
            draw([0.0, 10.0, -30.0, 0.0, 0.0, 0.0], false),
            Filter { x: false, y: true }
        );
        // A sub-pixel skew still counts as axis-aligned.
        assert_eq!(
            draw([20.0, 0.4, 0.0, 10.0, 0.0, 0.0], false),
            Filter { x: true, y: false }
        );
        // At the limit, only a flagged image blends.
        assert_eq!(
            draw([40.0, 0.0, 0.0, 10.0, 0.0, 0.0], false),
            Filter::NEAREST
        );
        assert_eq!(
            draw([40.0, 0.0, 0.0, 10.0, 0.0, 0.0], true),
            Filter { x: true, y: false }
        );
        // A rotated draw blends on both axes, at any scale below the limit.
        assert_eq!(draw([8.66, 5.0, -5.0, 8.66, 0.0, 0.0], false), both);
        assert_eq!(
            draw([86.6, 50.0, -50.0, 86.6, 0.0, 0.0], false),
            Filter::NEAREST
        );
        assert_eq!(draw([86.6, 50.0, -50.0, 86.6, 0.0, 0.0], true), both);
    }
}
