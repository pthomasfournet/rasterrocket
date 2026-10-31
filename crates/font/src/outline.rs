//! Glyph outline → `raster::Path` conversion.
//!
//! [`PathPen`] receives outline commands in font units and appends them to a
//! [`Path`] after mapping every point through an affine transform
//! (`font units → em → device pixels`).  Quadratic segments are promoted to
//! cubics by exact degree elevation:
//!
//! ```text
//! p1 = (p0 + 2·pc) / 3
//! p2 = (2·pc + p3) / 3
//! ```
//!
//! where `pc` is the quadratic control point.

use raster::path::{Path, PathBuilder, PathPoint};
use skrifa::outline::OutlinePen;

/// Affine map `[a, b, c, d, e, f]`: `x' = a·x + c·y + e`, `y' = b·x + d·y + f`.
pub(crate) type Affine = [f64; 6];

/// Outline pen that builds a [`Path`] in transformed coordinates.
///
/// A contour becomes one closed subpath.  A move-to is held back until the
/// first drawing command of its contour, so empty contours and repeated
/// move-tos leave no degenerate one-point subpaths, and a subpath is closed
/// only when at least one segment was drawn into it.
pub(crate) struct PathPen {
    builder: PathBuilder,
    transform: Affine,
    pending_move: Option<PathPoint>,
    open: bool,
}

impl PathPen {
    /// Create a pen applying `transform` to every point it receives.
    #[must_use]
    pub(crate) fn new(transform: Affine) -> Self {
        Self {
            builder: PathBuilder::new(),
            transform,
            pending_move: None,
            open: false,
        }
    }

    /// Finish the outline and return the path.
    #[must_use]
    pub(crate) fn finish(mut self) -> Path {
        self.close_contour();
        self.builder.build()
    }

    fn map(&self, x: f32, y: f32) -> PathPoint {
        let [xx, yx, xy, yy, dx, dy] = self.transform;
        let (x, y) = (f64::from(x), f64::from(y));
        PathPoint::new(
            xx.mul_add(x, xy.mul_add(y, dx)),
            yx.mul_add(x, yy.mul_add(y, dy)),
        )
    }

    /// Emit the held-back move-to, if any, so a segment can follow.
    fn begin_segment(&mut self) -> bool {
        if let Some(p) = self.pending_move.take() {
            self.close_contour();
            // A move-to directly after a closed or fresh path cannot fail:
            // no one-point subpath is ever left open.
            if self.builder.move_to(p.x, p.y).is_err() {
                return false;
            }
            self.open = true;
        }
        self.open
    }

    fn close_contour(&mut self) {
        if self.open {
            // `force = false`: a closing segment is added only when the last
            // point differs from the first.
            let _ = self.builder.close(false);
            self.open = false;
        }
    }
}

impl OutlinePen for PathPen {
    fn move_to(&mut self, x: f32, y: f32) {
        self.close_contour();
        self.pending_move = Some(self.map(x, y));
    }

    fn line_to(&mut self, x: f32, y: f32) {
        let p = self.map(x, y);
        if self.begin_segment() {
            let _ = self.builder.line_to(p.x, p.y);
        }
    }

    fn quad_to(&mut self, cx0: f32, cy0: f32, x: f32, y: f32) {
        let pc = self.map(cx0, cy0);
        let p3 = self.map(x, y);
        if self.begin_segment()
            && let Some(p0) = self.builder.cur_pt()
        {
            let (p1, p2) = quad_to_cubic(p0, pc, p3);
            let _ = self.builder.curve_to(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y);
        }
    }

    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        let p1 = self.map(cx0, cy0);
        let p2 = self.map(cx1, cy1);
        let p3 = self.map(x, y);
        if self.begin_segment() {
            let _ = self.builder.curve_to(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y);
        }
    }

    fn close(&mut self) {
        self.pending_move = None;
        self.close_contour();
    }
}

/// Elevate a quadratic Bézier `(p0, pc, p3)` to a cubic; returns the two
/// cubic control points.  The conversion is exact.
#[inline]
#[must_use]
pub(crate) fn quad_to_cubic(p0: PathPoint, pc: PathPoint, p3: PathPoint) -> (PathPoint, PathPoint) {
    let x1 = p0.x.mul_add(1.0 / 3.0, pc.x * (2.0 / 3.0));
    let y1 = p0.y.mul_add(1.0 / 3.0, pc.y * (2.0 / 3.0));
    let x2 = pc.x.mul_add(2.0 / 3.0, p3.x * (1.0 / 3.0));
    let y2 = pc.y.mul_add(2.0 / 3.0, p3.y * (1.0 / 3.0));
    (PathPoint::new(x1, y1), PathPoint::new(x2, y2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use raster::path::PathFlags;

    fn pt(x: f64, y: f64) -> PathPoint {
        PathPoint::new(x, y)
    }

    const IDENTITY: Affine = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

    #[test]
    fn quad_to_cubic_control_points() {
        // p0=(0,0), pc=(1,1), p3=(2,0): p1 = 2/3·pc, p2 = (2·pc + p3)/3.
        let (p1, p2) = quad_to_cubic(pt(0.0, 0.0), pt(1.0, 1.0), pt(2.0, 0.0));
        assert!((p1.x - 2.0 / 3.0).abs() < 1e-12);
        assert!((p1.y - 2.0 / 3.0).abs() < 1e-12);
        assert!((p2.x - 4.0 / 3.0).abs() < 1e-12);
        assert!((p2.y - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn collinear_quad_stays_collinear() {
        let (p1, p2) = quad_to_cubic(pt(0.0, 0.0), pt(1.5, 0.0), pt(3.0, 0.0));
        assert!(p1.y.abs() < 1e-12 && p2.y.abs() < 1e-12);
        assert!((p1.x - 1.0).abs() < 1e-12);
        assert!((p2.x - 2.0).abs() < 1e-12);
    }

    #[test]
    fn transform_is_applied() {
        let mut pen = PathPen::new([2.0, 0.0, 0.5, 3.0, 10.0, 20.0]);
        pen.move_to(1.0, 1.0);
        pen.line_to(2.0, 1.0);
        pen.close();
        let path = pen.finish();
        // (1,1) → (2·1 + 0.5·1 + 10, 3·1 + 20) = (12.5, 23).
        assert_eq!(path.pts[0], pt(12.5, 23.0));
        assert_eq!(path.pts[1], pt(14.5, 23.0));
    }

    #[test]
    fn contours_become_closed_subpaths() {
        let mut pen = PathPen::new(IDENTITY);
        pen.move_to(0.0, 0.0);
        pen.line_to(1.0, 0.0);
        pen.line_to(1.0, 1.0);
        pen.close();
        pen.move_to(5.0, 5.0);
        pen.line_to(6.0, 5.0);
        pen.line_to(6.0, 6.0);
        // No explicit close: finish() closes the open contour.
        let path = pen.finish();
        let firsts = path.flags.iter().filter(|f| f.is_first()).count();
        assert_eq!(firsts, 2);
        assert!(path.flags[0].contains(PathFlags::CLOSED));
        assert!(path.flags.last().unwrap().contains(PathFlags::CLOSED));
    }

    #[test]
    fn empty_contours_and_repeated_moves_are_dropped() {
        let mut pen = PathPen::new(IDENTITY);
        pen.move_to(0.0, 0.0);
        pen.close();
        pen.move_to(1.0, 1.0);
        pen.move_to(2.0, 2.0);
        pen.line_to(3.0, 2.0);
        pen.close();
        let path = pen.finish();
        assert_eq!(path.pts[0], pt(2.0, 2.0));
        assert_eq!(path.flags.iter().filter(|f| f.is_first()).count(), 1);
    }

    #[test]
    fn quad_is_elevated_from_current_point() {
        let mut pen = PathPen::new(IDENTITY);
        pen.move_to(0.0, 0.0);
        pen.quad_to(1.0, 1.0, 2.0, 0.0);
        pen.close();
        let path = pen.finish();
        assert!(path.flags[1].is_curve() && path.flags[2].is_curve());
        assert!((path.pts[1].x - 2.0 / 3.0).abs() < 1e-6);
        assert_eq!(path.pts[3], pt(2.0, 0.0));
    }
}
