//! Axial (linear) gradient pattern (PDF §8.7.4.5.3, type 2).
//!
//! The colour at pixel `(x, y)` is read from the ramp at the normalised axis
//! parameter `s` of the pixel's projection onto the axis: `s = 0` at the
//! start point and `s = 1` at the end point.
//!
//! # Points beyond the axis
//!
//! A pixel with `s < 0` takes the start colour when `extend_start` is set and
//! is otherwise not painted; likewise `s > 1` and `extend_end`.  A degenerate
//! (zero-length) axis paints nothing.

use super::{ColorRamp, fill_ramp_span};
use crate::pipe::Pattern;

/// Linear gradient between two points in device space.
///
/// `s = dot(p - p0, axis) / |axis|²`; pixels outside `s ∈ [0, 1]` are
/// painted with the nearer endpoint colour only when that end is extended.
pub struct AxialPattern {
    ramp: ColorRamp,
    ax: f64,
    ay: f64,
    p0x: f64,
    p0y: f64,
    /// `1 / (ax² + ay²)`; zero when the axis is degenerate (zero-length).
    inv_len_sq: f64,
    extend_start: bool,
    extend_end: bool,
}

impl AxialPattern {
    /// Create an axial gradient.
    ///
    /// - `ramp`: colours over the normalised parameter, start point → 0 and
    ///   end point → 1.
    /// - `(p0x, p0y)`, `(p1x, p1y)`: axis endpoints in device pixels.
    /// - `extend_start` / `extend_end`: paint beyond the start / end point.
    #[must_use]
    pub fn new(
        ramp: ColorRamp,
        p0x: f64,
        p0y: f64,
        p1x: f64,
        p1y: f64,
        extend_start: bool,
        extend_end: bool,
    ) -> Self {
        let ax = p1x - p0x;
        let ay = p1y - p0y;
        let len_sq = ax.mul_add(ax, ay * ay);
        let inv_len_sq = if len_sq > 0.0 { 1.0 / len_sq } else { 0.0 };
        Self {
            ramp,
            ax,
            ay,
            p0x,
            p0y,
            inv_len_sq,
            extend_start,
            extend_end,
        }
    }

    /// Ramp position `s ∈ [0, 1]` of pixel `(x, y)`, or `None` when the pixel
    /// is not painted.
    fn position(&self, x: i32, y: i32) -> Option<f64> {
        if self.inv_len_sq == 0.0 {
            return None;
        }
        let dx = f64::from(x) - self.p0x;
        let dy = f64::from(y) - self.p0y;
        let s = dx.mul_add(self.ax, dy * self.ay) * self.inv_len_sq;
        if s < 0.0 {
            self.extend_start.then_some(0.0)
        } else if s > 1.0 {
            self.extend_end.then_some(1.0)
        } else {
            // NaN (non-finite coordinates) is not painted.
            (s >= 0.0).then_some(s)
        }
    }
}

impl Pattern for AxialPattern {
    fn fill_span(&self, y: i32, x0: i32, x1: i32, out: &mut [u8]) {
        fill_ramp_span(&self.ramp, x0, x1, out, |_, _| {}, |x| self.position(x, y));
    }

    fn may_leave_unpainted(&self) -> bool {
        !(self.extend_start && self.extend_end) || self.inv_len_sq == 0.0
    }

    fn fill_span_painted(&self, y: i32, x0: i32, x1: i32, out: &mut [u8], painted: &mut [bool]) {
        fill_ramp_span(
            &self.ramp,
            x0,
            x1,
            out,
            |i, p| painted[i] = p,
            |x| self.position(x, y),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_axial(extend: bool) -> AxialPattern {
        AxialPattern::new(
            ColorRamp::two([0, 0, 0], [255, 255, 255]),
            0.0,
            0.0,
            8.0,
            0.0,
            extend,
            extend,
        )
    }

    fn painted(p: &AxialPattern, y: i32, x0: i32, x1: i32) -> Vec<bool> {
        let n = usize::try_from(x1 - x0 + 1).expect("x1 ≥ x0");
        let mut out = vec![0u8; n * 3];
        let mut mask = vec![false; n];
        p.fill_span_painted(y, x0, x1, &mut out, &mut mask);
        mask
    }

    #[test]
    fn midpoint_is_grey() {
        let p = make_axial(true);
        let mut out = [0u8; 3];
        p.fill_span(0, 4, 4, &mut out);
        assert!(out[0] >= 126 && out[0] <= 129, "mid R={}", out[0]);
    }

    #[test]
    fn start_is_black() {
        let p = make_axial(true);
        let mut out = [0u8; 3];
        p.fill_span(0, 0, 0, &mut out);
        assert_eq!(out[0], 0, "start should be black");
    }

    #[test]
    fn end_is_white() {
        let p = make_axial(true);
        let mut out = [0u8; 3];
        p.fill_span(0, 8, 8, &mut out);
        assert_eq!(out[0], 255, "end should be white");
    }

    #[test]
    fn outside_without_extend_is_not_painted() {
        let p = make_axial(false);
        assert!(p.may_leave_unpainted());
        assert_eq!(
            painted(&p, 0, -2, 10),
            [
                false, false, true, true, true, true, true, true, true, true, true, false, false
            ]
        );
    }

    #[test]
    fn one_sided_extension_paints_only_that_side() {
        let p = AxialPattern::new(
            ColorRamp::two([0, 0, 0], [255, 255, 255]),
            0.0,
            0.0,
            8.0,
            0.0,
            true,
            false,
        );
        let mask = painted(&p, 3, -5, 12);
        assert!(mask[..14].iter().all(|&m| m), "x ≤ 8 painted: {mask:?}");
        assert!(mask[14..].iter().all(|&m| !m), "x > 8 unpainted: {mask:?}");
    }

    #[test]
    fn extension_on_both_ends_paints_everywhere() {
        let p = make_axial(true);
        assert!(!p.may_leave_unpainted());
        assert!(painted(&p, 0, -50, 50).iter().all(|&m| m));
    }

    #[test]
    fn outside_extend_clamps_to_endpoints() {
        let p = make_axial(true);
        let mut out = [0u8; 3];
        p.fill_span(0, -5, -5, &mut out);
        assert_eq!(out[0], 0, "before start with extend should clamp to color0");
        p.fill_span(0, 100, 100, &mut out);
        assert_eq!(out[0], 255, "after end with extend should clamp to color1");
    }

    #[test]
    fn degenerate_axis_paints_nothing() {
        let p = AxialPattern::new(
            ColorRamp::two([255, 0, 0], [0, 255, 0]),
            5.0,
            5.0,
            5.0,
            5.0,
            true,
            true,
        );
        assert!(p.may_leave_unpainted());
        assert_eq!(painted(&p, 5, 4, 6), [false; 3]);
    }

    #[test]
    fn gradient_increases_left_to_right() {
        let p = make_axial(true);
        let mut out = vec![0u8; 5 * 3];
        p.fill_span(0, 0, 4, &mut out);
        assert!(out[0] < out[3], "gradient R should increase left-to-right");
        assert!(out[3] < out[6], "gradient R should increase left-to-right");
    }

    #[test]
    fn reversed_axis_reverses_gradient() {
        // Start point on the right: the start colour lies at x = 4.
        let p = AxialPattern::new(
            ColorRamp::two([255, 0, 0], [0, 0, 255]),
            4.0,
            0.0,
            0.0,
            0.0,
            true,
            true,
        );
        let mut left = [0u8; 3];
        let mut right = [0u8; 3];
        p.fill_span(0, 0, 0, &mut left);
        p.fill_span(0, 4, 4, &mut right);
        assert!(right[0] > 200, "x=4 should be near red (color0)");
        assert!(left[2] > 200, "x=0 should be near blue (color1)");
    }
}
