//! The inverse of an image's placement and its Q32 encoding.
//!
//! [`InverseMap`] gives the source coordinate of a device pixel centre as a
//! column term plus a row term in `f64`; the split functions turn a term or
//! a coordinate into the `i32` integer part and `u32` fraction of the
//! module root's layout, clamped so that a coordinate off the image stays
//! off it.

use super::grid::GridView;
use super::{COORD_LIMIT, FRACTION_SCALE};
use crate::renderer::gstate::Ctm;

/// The inverse of an image's placement: the source coordinate `(X, Y)`,
/// in samples, of the centre of each device pixel.
///
/// The CTM `[a b c d e f]` sends image space `(u, v)` to PDF user space:
///
/// ```text
/// x     = a·u + c·v + e
/// y_pdf = b·u + d·v + f
/// ```
///
/// so with `dx_rel = x − e`, `dy_rel = y_pdf − f` and `det = a·d − b·c`,
/// `u = (d·dx_rel − c·dy_rel) / det` and `v = (a·dy_rel − b·dx_rel) / det`.
/// The source coordinate is `X = u·img_w` and `Y = (1 − v)·img_h` (image
/// rows run top-down, PDF `v` bottom-up).  Both are affine in `dx_rel` and
/// `dy_rel`, so each splits into a column term, a function of the device
/// column alone, and a row term, which carries the `img_h` offset of `Y`.
#[derive(Clone, Copy)]
pub(super) struct InverseMap {
    /// `∂X/∂dx_rel = d·img_w / det`.
    kx_col: f64,
    /// `∂Y/∂dx_rel = b·img_h / det`.
    ky_col: f64,
    /// `∂X/∂dy_rel = −c·img_w / det`.
    kx_row: f64,
    /// `∂Y/∂dy_rel = −a·img_h / det`.
    ky_row: f64,
    /// The CTM's translation `(e, f)`.
    origin: (f64, f64),
    /// Page height in pixels, for the bitmap's downward y axis.
    page_h: f64,
    /// The page pixel of the window's pixel `(0, 0)`.
    window: (f64, f64),
    /// Image height in samples.
    img_h: f64,
}

impl InverseMap {
    /// The inverse of `ctm` for an `img_w × img_h` image sampled onto
    /// `view`.  `None` when `ctm` is singular; every
    /// component of `ctm` is finite.
    #[expect(
        clippy::many_single_char_names,
        reason = "PDF CTM components a-f are spec terminology"
    )]
    pub(super) fn new(ctm: &Ctm, view: GridView, img_w: u32, img_h: u32) -> Option<Self> {
        let [a, b, c, d, e, f] = *ctm;
        let det = a.mul_add(d, -(b * c));
        if det.abs() < 1e-12 {
            return None;
        }
        let inv_det = 1.0 / det;
        let w = f64::from(img_w);
        let h = f64::from(img_h);
        Some(Self {
            kx_col: d * inv_det * w,
            ky_col: b * inv_det * h,
            kx_row: -c * inv_det * w,
            ky_row: -a * inv_det * h,
            origin: (e, f),
            page_h: f64::from(view.page_height),
            window: (f64::from(view.origin.0), f64::from(view.origin.1)),
            img_h: h,
        })
    }

    /// The column term `(X, Y)` of the window's column `dx`.
    pub(super) fn col_term(&self, dx: u32) -> (f64, f64) {
        let dx_rel = (f64::from(dx) + self.window.0 + 0.5) - self.origin.0;
        (self.kx_col * dx_rel, self.ky_col * dx_rel)
    }

    /// The row term `(X, Y)` of the window's row `dy`.
    pub(super) fn row_term(&self, dy: u32) -> (f64, f64) {
        let dy_rel = (self.page_h - (f64::from(dy) + self.window.1 + 0.5)) - self.origin.1;
        (
            self.kx_row * dy_rel,
            self.ky_row.mul_add(dy_rel, self.img_h),
        )
    }

    /// The source coordinate of the centre of device pixel `(dx, dy)`.
    pub(super) fn coord(&self, dx: u32, dy: u32) -> (f64, f64) {
        let (cx, cy) = self.col_term(dx);
        let (rx, ry) = self.row_term(dy);
        (cx + rx, cy + ry)
    }
}

/// Append one `[x_hi, x_lo, y_hi, y_lo]` entry.
pub(super) fn push_entry(table: &mut Vec<u32>, x: f64, y: f64) {
    let (x_hi, x_lo) = split_q32(x);
    let (y_hi, y_lo) = split_q32(y);
    table.extend_from_slice(&[x_hi, x_lo, y_hi, y_lo]);
}

/// Whether `v` is a table term whose integer part lies within
/// ±[`COORD_LIMIT`](super::COORD_LIMIT).
pub(super) fn fits_q32(v: f64) -> bool {
    v.abs() <= COORD_LIMIT
}

/// Split a source coordinate along an axis of `n` samples into an integer
/// part and a Q32 fraction, first clamping it to `[-1, n]`: a coordinate off
/// the image stays off it, on the same side, and the integer part fits `i32`.
/// NaN maps to `n`, which the range check rejects.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "value is clamped to [-1, n] with n < 2^31, so the floor fits i32; the fraction is in [0, 2^32)"
)]
pub(super) fn split_bounded(v: f64, n: u32) -> (i32, u32) {
    let upper = f64::from(n.min(i32::MAX.cast_unsigned()));
    let v = if v.is_nan() {
        upper
    } else {
        v.clamp(-1.0, upper)
    };
    let hi = v.floor();
    let lo = ((v - hi) * FRACTION_SCALE) as u32;
    (hi as i32, lo)
}

/// Split a source coordinate into its Q32 integer and fraction words.
///
/// Non-finite input maps to the positive limit, which the range check
/// rejects.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "value is clamped to ±(2^30 - 1) so the floor fits i32; the fraction is in [0, 2^32)"
)]
fn split_q32(v: f64) -> (u32, u32) {
    let v = if v.is_finite() {
        v.clamp(-COORD_LIMIT, COORD_LIMIT)
    } else {
        COORD_LIMIT
    };
    let hi = v.floor();
    // `v - hi` is exact for |v| < 2^52 and lies in [0, 1).
    let lo = ((v - hi) * FRACTION_SCALE) as u32;
    ((hi as i32) as u32, lo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_bounded_keeps_off_image_coordinates_off_the_image() {
        assert_eq!(split_bounded(2.5, 10), (2, 1 << 31));
        assert_eq!(split_bounded(-1e300, 10), (-1, 0));
        assert_eq!(split_bounded(1e300, 10), (10, 0));
        assert_eq!(split_bounded(f64::NAN, 10), (10, 0));
        assert_eq!(split_bounded(f64::NEG_INFINITY, 1 << 30), (-1, 0));
        assert_eq!(split_bounded(3e9, u32::MAX), (i32::MAX, 0));
    }

    #[test]
    fn split_q32_round_trips_sign_and_fraction() {
        let (hi, lo) = split_q32(2.5);
        assert_eq!((hi.cast_signed(), lo), (2, 1 << 31));
        let (hi, lo) = split_q32(-0.25);
        assert_eq!((hi.cast_signed(), lo), (-1, 3 << 30));
        let (hi, lo) = split_q32(f64::INFINITY);
        assert_eq!((hi.cast_signed(), lo), ((1 << 30) - 1, 0));
        let (hi, lo) = split_q32(-1e300);
        assert_eq!((hi.cast_signed(), lo), (1 - (1 << 30), 0));
    }
}
