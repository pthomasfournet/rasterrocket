//! The placement of an image thinner than a device pixel.
//!
//! The blit paints the device pixels whose centres the image covers (see
//! [`super::image_sampler`]), so an image whose device extent along one of
//! its axes is below one pixel can lie between two pixel centres and paint
//! none.  [`widen`] gives every such axis an extent of one pixel before the
//! draw is resampled and sampled; an axis at least one pixel long is kept.
//!
//! When the image axes map onto the device axes (the classification of
//! [`super::image_reduce::Plan`]: scale, flip and quarter turn, to within
//! [`AXIS_TOLERANCE`]), a short axis spans the one device pixel containing
//! its centre, in its own direction.  That pixel is the one whose centre
//! the axis covers when it covers one, so a draw that already marks a pixel
//! marks the same pixel, and the area-averaging of a reduced axis gives it
//! the mean of the samples along the axis.
//!
//! On a rotated or skewed draw the extent of an axis is the distance
//! between the two edges of the image parallelogram parallel to the other
//! axis.  A short axis is sheared away from the other one until that
//! distance is one pixel, keeping the parallelogram's centre: a band one
//! pixel thick meets a pixel centre in every row and column it crosses.

use super::super::gstate::Ctm;
use super::image_reduce::AXIS_TOLERANCE;

/// `ctm`, an image's matrix into device space with integer pixel edges on
/// both axes, with every axis whose device extent is below one pixel
/// widened to one pixel.  A non-finite or singular matrix is returned as
/// is.
#[expect(
    clippy::many_single_char_names,
    reason = "a–f are the PDF matrix components"
)]
pub(super) fn widen(ctm: &Ctm) -> Ctm {
    let [a, b, c, d, e, f] = *ctm;
    if ctm.iter().any(|v| !v.is_finite()) || a.mul_add(d, -(b * c)) == 0.0 {
        return *ctm;
    }
    let quarter_turn = a.abs() <= AXIS_TOLERANCE && d.abs() <= AXIS_TOLERANCE;
    let upright = b.abs() <= AXIS_TOLERANCE && c.abs() <= AXIS_TOLERANCE;
    if upright && !(quarter_turn && (b * c).abs() > (a * d).abs()) {
        // Columns along device x, rows along device y.
        let (a, e) = widen_aligned(a, e);
        let (d, f) = widen_aligned(d, f);
        [a, b, c, d, e, f]
    } else if quarter_turn {
        // Columns along device y, rows along device x.
        let (b, f) = widen_aligned(b, f);
        let (c, e) = widen_aligned(c, e);
        [a, b, c, d, e, f]
    } else {
        widen_skewed(ctm)
    }
}

/// An axis spanning device `origin .. origin + extent` along a device
/// axis, as `(extent, origin)`: one shorter than a pixel instead spans the
/// pixel containing its centre, in the same direction.
fn widen_aligned(extent: f64, origin: f64) -> (f64, f64) {
    if extent == 0.0 || extent.abs() >= 1.0 {
        return (extent, origin);
    }
    let pixel = extent.mul_add(0.5, origin).floor();
    if extent > 0.0 {
        (1.0, pixel)
    } else {
        (-1.0, pixel + 1.0)
    }
}

/// [`widen`] of a rotated or skewed draw, whose matrix is finite and
/// regular: the column axis `u = (a, b)` and the row axis `v = (c, d)` each
/// sheared, when its extent across the other is below one pixel, by the
/// shortfall along the unit normal of the other on its own side, and the
/// origin moved to keep the centre `(e, f) + (u + v) / 2`.  The row axis is
/// widened first; widening the column axis away from it then leaves the
/// row axis at least one pixel across.
#[expect(
    clippy::many_single_char_names,
    reason = "a–f are the PDF matrix components"
)]
fn widen_skewed(ctm: &Ctm) -> Ctm {
    let [a, b, c, d, e, f] = *ctm;
    let (c2, d2) = widen_across((c, d), (a, b));
    let (a2, b2) = widen_across((a, b), (c2, d2));
    let shift = (a2 - a + c2 - c, b2 - b + d2 - d);
    [
        a2,
        b2,
        c2,
        d2,
        shift.0.mul_add(-0.5, e),
        shift.1.mul_add(-0.5, f),
    ]
}

/// The axis `p`, sheared along the unit normal of the axis `q` on the side
/// of `p` until its extent across `q` is one pixel when it is less.
fn widen_across(p: (f64, f64), q: (f64, f64)) -> (f64, f64) {
    let len = q.0.hypot(q.1);
    // The signed distance of `p`'s end from the line of `q`, along the
    // normal `(-q.1, q.0) / len`.
    let across = q.0.mul_add(p.1, -(q.1 * p.0)) / len;
    if !across.is_finite() || across.abs() >= 1.0 {
        return p;
    }
    let grow = (1.0 - across.abs()).copysign(across) / len;
    (grow.mul_add(-q.1, p.0), grow.mul_add(q.0, p.1))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The distance between the two edges parallel to `q` of the
    /// parallelogram on the axes `p` and `q`.
    fn extent_across(p: (f64, f64), q: (f64, f64)) -> f64 {
        q.0.mul_add(p.1, -(q.1 * p.0)).abs() / q.0.hypot(q.1)
    }

    fn centre(m: &Ctm) -> (f64, f64) {
        (
            m[4] + f64::midpoint(m[0], m[2]),
            m[5] + f64::midpoint(m[1], m[3]),
        )
    }

    #[test]
    fn axes_of_a_pixel_or_more_are_kept() {
        for m in [
            [1.0, 0.0, 0.0, 1.0, 10.6, 5.0],
            [-3.5, 0.0, 0.2, 40.0, 7.25, 1.5],
            [0.0, 2.0, -1.0, 0.0, 3.3, 4.4],
            [30.0, 20.0, -20.0, 30.0, 1.0, 2.0],
        ] {
            assert_eq!(widen(&m), m, "{m:?}");
        }
    }

    #[test]
    fn a_short_upright_axis_spans_the_pixel_containing_its_centre() {
        // x 10.6 .. 11.2: centre 10.9 in pixel 10.
        assert_eq!(
            widen(&[0.6, 0.0, 0.0, 10.0, 10.6, 5.0]),
            [1.0, 0.0, 0.0, 10.0, 10.0, 5.0]
        );
        // Flipped: x 11.2 .. 10.6, the same pixel, still running leftwards.
        assert_eq!(
            widen(&[-0.6, 0.0, 0.0, 10.0, 11.2, 5.0]),
            [-1.0, 0.0, 0.0, 10.0, 11.0, 5.0]
        );
        // Both axes short: y 7.7 .. 7.4 (flipped) has its centre in pixel 7.
        assert_eq!(
            widen(&[0.25, 0.0, 0.0, -0.3, 2.5, 7.7]),
            [1.0, 0.0, 0.0, -1.0, 2.0, 8.0]
        );
    }

    #[test]
    fn a_short_axis_of_a_quarter_turn_spans_the_pixel_containing_its_centre() {
        // Columns along y over 3.9 .. 4.4 (pixel 4), rows along x over
        // 20 .. 12.
        assert_eq!(
            widen(&[0.0, 0.5, -8.0, 0.0, 20.0, 3.9]),
            [0.0, 1.0, -8.0, 0.0, 20.0, 4.0]
        );
        // Every component below the axis tolerance: the larger products
        // decide that the draw is a quarter turn.
        assert_eq!(
            widen(&[0.0, 0.3, -0.4, 0.0, 6.2, 9.1]),
            [0.0, 1.0, -1.0, 0.0, 7.0, 9.0]
        );
    }

    #[test]
    fn singular_and_non_finite_matrices_are_kept() {
        for ctm in [
            [0.0, 0.0, 0.0, 10.0, 1.0, 1.0],
            [0.5, 0.0, 0.0, f64::NAN, 1.0, 1.0],
            [2.0, 2.0, 1.0, 1.0, 0.0, 0.0],
        ] {
            let kept = widen(&ctm);
            assert!(
                kept.iter()
                    .zip(&ctm)
                    .all(|(got, want)| got.to_bits() == want.to_bits()),
                "{ctm:?}"
            );
        }
    }

    #[test]
    fn a_short_skewed_axis_is_sheared_to_one_pixel_about_the_centre() {
        // A 45° band 20 px long and 0.4 px thick.
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let ctm = [20.0 * s, 20.0 * s, -0.4 * s, 0.4 * s, 30.0, 40.0];
        let wide = widen(&ctm);
        let (col, row) = ((wide[0], wide[1]), (wide[2], wide[3]));
        assert!((extent_across(row, col) - 1.0).abs() < 1e-12, "{wide:?}");
        assert_eq!(col, (ctm[0], ctm[1]), "the long axis is kept");
        let (before, after) = (centre(&ctm), centre(&wide));
        assert!((before.0 - after.0).abs() < 1e-12 && (before.1 - after.1).abs() < 1e-12);
    }

    #[test]
    fn two_short_skewed_axes_are_each_at_least_one_pixel_across() {
        let ctm = [0.8, 0.6, 0.1, 0.2, 50.0, 60.0];
        let wide = widen(&ctm);
        let (col, row) = ((wide[0], wide[1]), (wide[2], wide[3]));
        assert!(extent_across(row, col) >= 1.0 - 1e-12, "{wide:?}");
        assert!(extent_across(col, row) >= 1.0 - 1e-12, "{wide:?}");
        let (before, after) = (centre(&ctm), centre(&wide));
        assert!((before.0 - after.0).abs() < 1e-12 && (before.1 - after.1).abs() < 1e-12);
    }
}
