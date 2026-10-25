//! Axial (linear) gradient pattern (PDF §8.7.4.5.3, type 2).
//!
//! The colour at pixel `(x, y)` is read from the ramp at the normalised axis
//! parameter `s` of the pixel's projection onto the axis: `s = 0` at the
//! start point and `s = 1` at the end point.  The projection is taken in the
//! shading's own space, which [`AxialPattern::in_shading_space`] relates to
//! the device: under a non-conformal transform the device-space lines of
//! constant `s` are not perpendicular to the device-space image of the axis.
//!
//! # Points beyond the axis
//!
//! A pixel with `s < 0` takes the start colour when `extend_start` is set and
//! is otherwise not painted; likewise `s > 1` and `extend_end`.  A degenerate
//! (zero-length) axis paints nothing.

use super::{ColorRamp, fill_ramp_span};
use crate::pipe::Pattern;

/// Linear gradient between two points.
///
/// `s = dot(p - p0, axis) / |axis|²` for the point `p` of the shading space;
/// as `p` is an affine function of the device pixel, so is `s`, held here as
/// `s = gx·x + gy·y + s0`.  Pixels outside `s ∈ [0, 1]` are painted with the
/// nearer endpoint colour only when that end is extended.
pub struct AxialPattern {
    ramp: ColorRamp,
    /// Device-space gradient of `s`.
    gx: f64,
    gy: f64,
    /// `s` at the device origin.
    s0: f64,
    /// Whether the axis is degenerate (zero-length).
    degenerate: bool,
    extend_start: bool,
    extend_end: bool,
}

impl AxialPattern {
    /// Create an axial gradient.
    ///
    /// - `ramp`: colours over the normalised parameter, start point → 0 and
    ///   end point → 1.
    /// - `(p0x, p0y)`, `(p1x, p1y)`: axis endpoints, in device pixels unless
    ///   [`Self::in_shading_space`] says otherwise.
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
        let degenerate = !len_sq.is_normal();
        let inv_len_sq = if degenerate { 0.0 } else { 1.0 / len_sq };
        Self {
            ramp,
            gx: ax * inv_len_sq,
            gy: ay * inv_len_sq,
            s0: -p0x.mul_add(ax, p0y * ay) * inv_len_sq,
            degenerate,
            extend_start,
            extend_end,
        }
    }

    /// Place the axis given to [`Self::new`] in a shading space that the
    /// affine map `device_to_shading` (`[a b c d e f]`, mapping `(x, y)` to
    /// `(a·x + c·y + e, b·x + d·y + f)`) takes device pixels into.
    #[must_use]
    pub fn in_shading_space(mut self, device_to_shading: &[f64; 6]) -> Self {
        let m = device_to_shading;
        let (gx, gy) = (self.gx, self.gy);
        self.gx = m[0].mul_add(gx, m[1] * gy);
        self.gy = m[2].mul_add(gx, m[3] * gy);
        self.s0 = m[4].mul_add(gx, m[5].mul_add(gy, self.s0));
        self
    }

    /// Ramp position `s ∈ [0, 1]` of pixel `(x, y)`, or `None` when the pixel
    /// is not painted.
    fn position(&self, x: i32, y: i32) -> Option<f64> {
        if self.degenerate {
            return None;
        }
        let s = self
            .gx
            .mul_add(f64::from(x), self.gy.mul_add(f64::from(y), self.s0));
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
        !(self.extend_start && self.extend_end) || self.degenerate
    }

    fn fill_span_coverage(&self, y: i32, x0: i32, x1: i32, out: &mut [u8], coverage: &mut [u8]) {
        fill_ramp_span(
            &self.ramp,
            x0,
            x1,
            out,
            |i, p| coverage[i] = u8::from(p) * 255,
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
        let mut coverage = vec![0; n];
        p.fill_span_coverage(y, x0, x1, &mut out, &mut coverage);
        coverage.into_iter().map(|c| c != 0).collect()
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

    /// Ramp position, as a red level, of device pixel `(x, y)`.
    fn level(p: &AxialPattern, x: i32, y: i32) -> u8 {
        let mut out = [0u8; 3];
        p.fill_span(y, x, x, &mut out);
        out[0]
    }

    #[test]
    fn a_shear_keeps_the_lines_of_constant_parameter_in_shading_space() {
        // Axis (0, 0) → (10, 0) in shading space (x − y, y): device points
        // on the line x − y = 5 all sit at s = 0.5.
        let p = AxialPattern::new(
            ColorRamp::two([0, 0, 0], [255, 255, 255]),
            0.0,
            0.0,
            10.0,
            0.0,
            false,
            false,
        )
        .in_shading_space(&[1.0, 0.0, -1.0, 1.0, 0.0, 0.0]);
        let mid = level(&p, 5, 0);
        assert!((126..=129).contains(&mid), "s = 0.5: {mid}");
        assert_eq!(level(&p, 9, 4), mid, "same line, one row up");
        assert_eq!(level(&p, 5, 5), 0, "s = 0 on the line x − y = 0");
    }

    #[test]
    fn a_non_uniform_scale_scales_the_axis() {
        // Device x is twice shading x: the axis 0 → 4 spans device 0 → 8.
        let p = AxialPattern::new(
            ColorRamp::two([0, 0, 0], [255, 255, 255]),
            0.0,
            0.0,
            4.0,
            0.0,
            false,
            false,
        )
        .in_shading_space(&[0.5, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let mid = level(&p, 4, 3);
        assert!((126..=129).contains(&mid), "s = 0.5: {mid}");
        assert_eq!(painted(&p, 0, 8, 9), [true, false]);
    }
}
