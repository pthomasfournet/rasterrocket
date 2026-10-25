//! Radial gradient pattern (PDF §8.7.4.5.4, type 3).
//!
//! The shading is the family of circles `c(s) = c0 + s·(c1 − c0)`,
//! `r(s) = r0 + s·(r1 − r0)` for `s ∈ [0, 1]`, continued to `s < 0` when
//! `extend_start` is set and to `s > 1` when `extend_end` is set.  A pixel
//! takes the ramp colour of the largest such `s` whose circle passes through
//! it with `r(s) ≥ 0`; a pixel on no such circle is not painted.
//!
//! The circles through a point `p` are the roots of
//! `a·s² − 2·b·s + c = 0` with `a = |Δc|² − Δr²`,
//! `b = (p − c0)·Δc + r0·Δr` and `c = |p − c0|² − r0²`.
//!
//! The circles lie in the shading's own space, which
//! [`RadialPattern::in_shading_space`] relates to the device: each device
//! pixel is mapped into that space before the equation is solved, so an
//! affine transform draws the circles as their affine images (ellipses
//! under a non-conformal transform).

use super::{ColorRamp, fill_ramp_span};
use crate::pipe::Pattern;

/// Gradient between two circles (centre `c0`, radius `r0`) and (`c1`, `r1`).
pub struct RadialPattern {
    ramp: ColorRamp,
    c0x: f64,
    c0y: f64,
    /// Δc = c1 − c0.
    dcx: f64,
    dcy: f64,
    r0: f64,
    /// Δr = r1 − r0.
    dr: f64,
    extend_start: bool,
    extend_end: bool,
    /// `|Δc|² − Δr²`, the `s²` coefficient of the circle equation; zero
    /// when the equation is linear in `s`.
    a: f64,
    /// The affine map `[a b c d e f]` from device pixels to the space of
    /// the circles.
    device_to_shading: [f64; 6],
}

impl RadialPattern {
    /// Create a radial gradient.
    ///
    /// - `ramp`: colours over the normalised parameter, start circle → 0 and
    ///   end circle → 1.
    /// - `(c0x, c0y, r0)`: start circle centre and radius (≥ 0).
    /// - `(c1x, c1y, r1)`: end circle centre and radius (≥ 0).
    /// - `extend_start` / `extend_end`: continue the circles beyond the start
    ///   / end circle.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the PDF shading dictionary: ramp, two circles (cx, cy, r each), two extend flags"
    )]
    pub fn new(
        ramp: ColorRamp,
        c0x: f64,
        c0y: f64,
        r0: f64,
        c1x: f64,
        c1y: f64,
        r1: f64,
        extend_start: bool,
        extend_end: bool,
    ) -> Self {
        let dcx = c1x - c0x;
        let dcy = c1y - c0y;
        let dr = r1 - r0;
        let dc_sq = dcx.mul_add(dcx, dcy * dcy);
        let a = dr.mul_add(-dr, dc_sq);
        // A coefficient this small relative to its terms is cancellation
        // noise; the equation is then linear.
        let a = if a.abs() <= 1e-9 * dr.mul_add(dr, dc_sq) {
            0.0
        } else {
            a
        };
        Self {
            ramp,
            c0x,
            c0y,
            dcx,
            dcy,
            r0,
            dr,
            extend_start,
            extend_end,
            a,
            device_to_shading: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        }
    }

    /// Place the circles given to [`Self::new`] in a shading space that the
    /// affine map `device_to_shading` (`[a b c d e f]`, mapping `(x, y)` to
    /// `(a·x + c·y + e, b·x + d·y + f)`) takes device pixels into.
    #[must_use]
    pub const fn in_shading_space(mut self, device_to_shading: &[f64; 6]) -> Self {
        self.device_to_shading = *device_to_shading;
        self
    }

    /// Whether the circle at parameter `s` belongs to the shading.
    fn admits(&self, s: f64) -> bool {
        let in_range = if s < 0.0 {
            self.extend_start
        } else if s > 1.0 {
            self.extend_end
        } else {
            // False for NaN.
            s >= 0.0
        };
        in_range && self.dr.mul_add(s, self.r0) >= 0.0
    }

    /// Ramp position `s ∈ [0, 1]` of pixel `(xi, yi)`, or `None` when no
    /// circle of the shading passes through it.
    fn position(&self, xi: i32, yi: i32) -> Option<f64> {
        let inv = &self.device_to_shading;
        let (px, py) = (f64::from(xi), f64::from(yi));
        let rel_x = inv[0].mul_add(px, inv[2].mul_add(py, inv[4])) - self.c0x;
        let rel_y = inv[1].mul_add(px, inv[3].mul_add(py, inv[5])) - self.c0y;
        let b = self
            .r0
            .mul_add(self.dr, rel_x.mul_add(self.dcx, rel_y * self.dcy));
        let c = self
            .r0
            .mul_add(-self.r0, rel_x.mul_add(rel_x, rel_y * rel_y));

        let roots = if self.a == 0.0 {
            if b == 0.0 {
                return None;
            }
            let s = c / (b + b);
            [s, s]
        } else {
            let disc = b.mul_add(b, -(self.a * c));
            if disc < 0.0 {
                return None;
            }
            let sq = disc.sqrt();
            let (u, v) = ((b + sq) / self.a, (b - sq) / self.a);
            [u.max(v), u.min(v)]
        };
        roots
            .into_iter()
            .find(|&s| self.admits(s))
            .map(|s| s.clamp(0.0, 1.0))
    }
}

impl Pattern for RadialPattern {
    fn fill_span(&self, y: i32, x0: i32, x1: i32, out: &mut [u8]) {
        fill_ramp_span(&self.ramp, x0, x1, out, |_, _| {}, |x| self.position(x, y));
    }

    fn may_leave_unpainted(&self) -> bool {
        true
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

    /// Concentric radial: c0=c1=(4,4), r0=0, r1=4.
    fn make_concentric(extend: bool) -> RadialPattern {
        RadialPattern::new(
            ColorRamp::two([0, 0, 0], [255, 255, 255]),
            4.0,
            4.0,
            0.0,
            4.0,
            4.0,
            4.0,
            extend,
            extend,
        )
    }

    /// Red at the start circle, green at the end circle.
    fn red_to_green(
        c0: (f64, f64, f64),
        c1: (f64, f64, f64),
        extend: (bool, bool),
    ) -> RadialPattern {
        RadialPattern::new(
            ColorRamp::two([255, 0, 0], [0, 255, 0]),
            c0.0,
            c0.1,
            c0.2,
            c1.0,
            c1.1,
            c1.2,
            extend.0,
            extend.1,
        )
    }

    fn colour_at(p: &RadialPattern, x: i32, y: i32) -> Option<[u8; 3]> {
        let mut out = [0u8; 3];
        let mut coverage = [0];
        p.fill_span_coverage(y, x, x, &mut out, &mut coverage);
        (coverage[0] != 0).then_some(out)
    }

    #[test]
    fn centre_is_color0() {
        let c = colour_at(&make_concentric(true), 4, 4).expect("painted");
        assert!(c[0] < 10, "centre should be near-black, got {c:?}");
    }

    #[test]
    fn outer_ring_is_color1() {
        let c = colour_at(&make_concentric(true), 8, 4).expect("painted");
        assert!(c[0] > 240, "outer ring should be near-white, got {c:?}");
    }

    #[test]
    fn annulus_without_extend_paints_only_between_the_circles() {
        let p = red_to_green((4.0, 4.0, 2.0), (4.0, 4.0, 6.0), (false, false));
        let c = colour_at(&p, 6, 4).expect("on the start circle");
        assert!(c[0] > 240 && c[1] < 20, "start circle is red: {c:?}");
        assert_eq!(colour_at(&p, 4, 4), None, "inside the start circle");
        assert_eq!(colour_at(&p, 11, 4), None, "outside the end circle");
    }

    #[test]
    fn extend_clamps_outside_to_endpoints() {
        let c = colour_at(&make_concentric(true), 14, 4).expect("painted");
        assert!(
            c[0] > 240,
            "beyond outer with extend should be color1: {c:?}"
        );
    }

    #[test]
    fn outside_without_extend_is_not_painted() {
        assert_eq!(colour_at(&make_concentric(false), 14, 4), None);
    }

    #[test]
    fn non_concentric_circles_paint_only_the_swept_band() {
        // Start circle (0, 0) r=1, end circle (10, 0) r=4: neither encloses
        // the other, so only the band the circles sweep is painted.
        let p = red_to_green((0.0, 0.0, 1.0), (10.0, 0.0, 4.0), (false, false));
        let start = colour_at(&p, -1, 0).expect("leftmost point of the start circle");
        assert!(start[0] > 240, "start circle is red: {start:?}");
        let end = colour_at(&p, 14, 0).expect("rightmost point of the end circle");
        assert!(end[1] > 240, "end circle is green: {end:?}");
        // (5, 0) lies on the circles at s = 28/91 and s = 78/91; the later
        // circle wins.
        let mid = colour_at(&p, 5, 0).expect("inside the band");
        assert!(
            (25..=45).contains(&mid[0]) && (210..=230).contains(&mid[1]),
            "s = 78/91: {mid:?}"
        );
        assert_eq!(colour_at(&p, -3, 0), None, "behind the start circle");
        assert_eq!(colour_at(&p, 15, 0), None, "beyond the end circle");
        assert_eq!(colour_at(&p, 5, 6), None, "beside the band");
        assert_eq!(colour_at(&p, 0, 100), None, "far off the axis");
    }

    #[test]
    fn extended_start_stops_at_the_cone_apex() {
        // The radius shrinks by 3 per |Δc| = 10, so the extended start is a
        // cone closing at s = −1/3, x = −10/3.
        let p = red_to_green((0.0, 0.0, 1.0), (10.0, 0.0, 4.0), (true, false));
        let c = colour_at(&p, -3, 0).expect("inside the cone");
        assert!(c[0] > 240, "extended start colour: {c:?}");
        assert_eq!(colour_at(&p, -4, 0), None, "beyond the apex");
    }

    #[test]
    fn larger_root_outside_the_range_falls_back_to_the_smaller() {
        // Concentric, radius 6 → 2: distance 4 lies on the circles s = 0.5
        // and s = 2.5, of which only s = 0.5 is in range.  Distance 7 lies
        // on s = −0.25 and s = 3.25, neither in range.
        let p = RadialPattern::new(
            ColorRamp::two([0, 0, 0], [255, 255, 255]),
            0.0,
            0.0,
            6.0,
            0.0,
            0.0,
            2.0,
            false,
            false,
        );
        let c = colour_at(&p, 4, 0).expect("between the circles");
        assert!((120..=135).contains(&c[0]), "s = 0.5: {c:?}");
        assert_eq!(colour_at(&p, 7, 0), None);
    }

    #[test]
    fn linear_equation_when_the_circles_share_a_tangent_point() {
        // |Δc| = Δr: every circle passes through the origin and the equation
        // is linear in s.
        let p = RadialPattern::new(
            ColorRamp::two([0, 0, 0], [255, 255, 255]),
            0.0,
            0.0,
            0.0,
            3.0,
            4.0,
            5.0,
            false,
            false,
        );
        let c = colour_at(&p, 6, 8).expect("end circle, opposite the origin");
        assert!(c[0] > 240, "end colour: {c:?}");
        assert_eq!(colour_at(&p, -3, -4), None, "behind the origin");
    }

    #[test]
    fn a_non_uniform_transform_draws_the_circles_as_ellipses() {
        // Circles of radius up to 4 about the origin, with device x twice
        // shading x: the end circle is the ellipse with half-axes 8 and 4.
        let p = RadialPattern::new(
            ColorRamp::two([0, 0, 0], [255, 255, 255]),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            4.0,
            false,
            false,
        )
        .in_shading_space(&[0.5, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let c = colour_at(&p, 6, 0).expect("inside the ellipse along x");
        assert!((185..=195).contains(&c[0]), "s = 0.75: {c:?}");
        assert!(colour_at(&p, 7, 0).is_some(), "inside along x");
        assert!(colour_at(&p, 0, 3).is_some(), "inside along y");
        assert_eq!(colour_at(&p, 0, 5), None, "outside along y");
        assert_eq!(colour_at(&p, 9, 0), None, "outside along x");
    }

    #[test]
    fn a_shear_moves_the_circle_centres() {
        // Shading space (x − y, y): the circle about the origin of radius 2
        // is the sheared ellipse through device (2, 0) and (2, 2).
        let p = RadialPattern::new(
            ColorRamp::two([0, 0, 0], [255, 255, 255]),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0,
            false,
            false,
        )
        .in_shading_space(&[1.0, 0.0, -1.0, 1.0, 0.0, 0.0]);
        assert!(colour_at(&p, 2, 2).is_some(), "the sheared boundary point");
        assert_eq!(colour_at(&p, -2, 2), None, "its mirror image is outside");
    }
}
