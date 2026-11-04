//! Stroke-adjust hint application.
//!
//! ## What stroke adjustment does
//!
//! PDF stroke adjustment snaps near-axis-aligned path segments to integer
//! pixel boundaries so that adjacent stroked rectangles share a pixel edge
//! rather than leaving a half-pixel gap. This is controlled by the PDF
//! `strokeAdjust` graphics state parameter.
//!
//! Each [`XPathAdjust`] describes three "snap windows" (around x0, xm, x1)
//! with ±0.01 tolerance. Any transformed coordinate that falls **strictly
//! inside** a window is replaced by the corresponding snapped target.
//!
//! ## Open-interval semantics
//!
//! The snap windows are **open** intervals: a coordinate exactly at a boundary
//! value (e.g. `v == adj0 - 0.01`) is **not** snapped. See [`stroke_adjust`].

use crate::types::round_half_up_i32;

/// An axis-aligned stroke-adjust hint, derived from a `StrokeAdjustHint`
/// (see `crate::path::stroke`) after path transformation.
///
/// ## Snap-window fields
///
/// The three snap windows are open intervals; a coordinate `v` is replaced by
/// the target only when `v ∈ (xa, xb)` (strictly inside):
///
/// | Fields | Meaning |
/// |--------|---------|
/// | `x0a`, `x0b`, `x0` | Left snap window: coordinate in `(x0a, x0b)` snaps to `x0`. |
/// | `xma`, `xmb`, `xm` | Mid snap window: coordinate in `(xma, xmb)` snaps to `xm`. |
/// | `x1a`, `x1b`, `x1` | Right snap window: coordinate in `(x1a, x1b)` snaps to `x1`. |
#[derive(Clone, Debug)]
pub(crate) struct XPathAdjust {
    /// First path-point index (inclusive) in the range to adjust.
    pub first_pt: usize,
    /// Last path-point index (inclusive) in the range to adjust.
    pub last_pt: usize,
    /// `true` → adjust the x-coordinate; `false` → adjust the y-coordinate.
    pub vert: bool,
    /// Left snap window lower bound (`adj0 - 0.01`).
    pub x0a: f64,
    /// Left snap window upper bound (`adj0 + 0.01`).
    pub x0b: f64,
    /// Left snap target (rounded `adj0`).
    pub x0: f64,
    /// Mid snap window lower bound (`mid - 0.01`).
    pub xma: f64,
    /// Mid snap window upper bound (`mid + 0.01`).
    pub xmb: f64,
    /// Mid snap target (midpoint of rounded endpoints).
    pub xm: f64,
    /// Right snap window lower bound (`adj1 - 0.01`).
    pub x1a: f64,
    /// Right snap window upper bound (`adj1 + 0.01`).
    pub x1b: f64,
    /// Right snap target (`rounded_adj1 - 0.01`).
    pub x1: f64,
    /// The first of the two points of the edge at `adj1`.  Where the left
    /// and right windows overlap (edges less than 0.02 pixels apart) a
    /// coordinate cannot tell its edge, and the points of this edge take the
    /// right target by index instead.
    pub edge1: usize,
}

impl XPathAdjust {
    /// Construct an adjust record for two endpoint values `adj0 ≤ adj1`,
    /// the edge at `adj1` beginning at point `edge1`.  When both round to the
    /// same pixel boundary, the span is widened to the pixel after it.
    #[must_use]
    pub(crate) fn new(
        first_pt: usize,
        last_pt: usize,
        vert: bool,
        adj0: f64,
        adj1: f64,
        edge1: usize,
    ) -> Self {
        debug_assert!(adj0 <= adj1, "adj0 must be <= adj1");

        let mid = adj0.midpoint(adj1);
        let r0 = f64::from(round_half_up_i32(adj0));
        let mut r1 = f64::from(round_half_up_i32(adj1));
        // Both are exact f64 images of i32 values, so bit equality is exact.
        if r0.to_bits() == r1.to_bits() {
            r1 += 1.0;
        }
        Self {
            first_pt,
            last_pt,
            vert,
            x0a: adj0 - 0.01,
            x0b: adj0 + 0.01,
            x0: r0,
            xma: mid - 0.01,
            xmb: mid + 0.01,
            xm: r0.midpoint(r1),
            x1a: adj1 - 0.01,
            x1b: adj1 + 0.01,
            x1: r1 - 0.01,
            edge1,
        }
    }

    /// Whether the left and right snap windows overlap.
    fn edges_coincide(&self) -> bool {
        self.x0b > self.x1a
    }

    /// Whether point `idx` lies on the edge at `adj1`: one of its two
    /// points, or the first point of the range when the range ends on that
    /// edge, the first and last points of a fill's closed rectangle being
    /// one.  (A stroke's range spans several subpaths and never ends on a
    /// hint edge this way.)
    const fn on_edge1(&self, idx: usize) -> bool {
        let e = self.edge1;
        idx == e || idx == e + 1 || (idx == self.first_pt && self.last_pt == e + 1)
    }
}

/// Apply `adj` to point `idx` of its range, at `(x, y)`: as
/// [`stroke_adjust`], except that where the two edges coincide, the
/// points of each edge take that edge's target.
#[inline]
pub(crate) fn stroke_adjust_point(adj: &XPathAdjust, idx: usize, x: &mut f64, y: &mut f64) {
    if adj.edges_coincide() {
        let v = if adj.vert { x } else { y };
        if *v > adj.x0a && *v < adj.x1b {
            *v = if adj.on_edge1(idx) { adj.x1 } else { adj.x0 };
        }
        return;
    }
    stroke_adjust(adj, x, y);
}

/// Apply a stroke-adjust hint to a single point `(x, y)`.
///
/// If `adj.vert` is true, the x-coordinate is examined; otherwise the
/// y-coordinate. If the relevant coordinate falls **strictly inside** any of
/// the three snap windows (open intervals), it is replaced by the corresponding
/// snapped target.
///
/// ## Open-interval semantics
///
/// Windows are open intervals; boundary values are **not** snapped. For
/// example, a coordinate equal to exactly `adj.x0a` (= `adj0 - 0.01`) is
/// outside the window `(x0a, x0b)` and will not be snapped.
#[inline]
pub(crate) fn stroke_adjust(adj: &XPathAdjust, x: &mut f64, y: &mut f64) {
    let v = if adj.vert { x } else { y };
    if *v > adj.x0a && *v < adj.x0b {
        *v = adj.x0;
    } else if *v > adj.xma && *v < adj.xmb {
        *v = adj.xm;
    } else if *v > adj.x1a && *v < adj.x1b {
        *v = adj.x1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snaps_to_x0() {
        let adj = XPathAdjust::new(0, 1, true, 1.0, 3.0, 1);
        let mut x = 1.005;
        let mut y = 0.0;
        stroke_adjust(&adj, &mut x, &mut y);
        assert!((x - 1.0).abs() < 1e-10, "x={x}");
    }

    #[test]
    fn snaps_to_xm() {
        let adj = XPathAdjust::new(0, 1, true, 1.0, 3.0, 1);
        let mut x = 2.0; // midpoint of [1,3]
        let mut y = 0.0;
        stroke_adjust(&adj, &mut x, &mut y);
        assert!((x - 2.0).abs() < 1e-10, "x={x}"); // xm = (1+3)/2 = 2.0
    }

    #[test]
    fn snaps_to_x1() {
        let adj = XPathAdjust::new(0, 1, true, 1.0, 3.0, 1);
        let mut x = 3.005;
        let mut y = 0.0;
        stroke_adjust(&adj, &mut x, &mut y);
        // x1 = 3.0 - 0.01 = 2.99
        assert!((x - 2.99).abs() < 1e-10, "x={x}");
    }

    #[test]
    fn no_snap_outside_windows() {
        let adj = XPathAdjust::new(0, 1, true, 1.0, 3.0, 1);
        let mut x = 5.0;
        let mut y = 0.0;
        stroke_adjust(&adj, &mut x, &mut y);
        assert!((x - 5.0).abs() < 1e-10);
    }

    #[test]
    fn horizontal_adjusts_y() {
        let adj = XPathAdjust::new(0, 1, false, 2.0, 4.0, 1);
        let mut x = 0.0;
        let mut y = 2.005;
        stroke_adjust(&adj, &mut x, &mut y);
        assert!((y - 2.0).abs() < 1e-10, "y={y}");
        assert!((x - 0.0).abs() < 1e-10); // x unchanged
    }

    #[test]
    fn coinciding_edges_snap_apart_by_point() {
        // A zero-width rectangle's vertical edges: points 1–2 and 3–4, the
        // range 0..=4 closing on the second edge.
        let adj = XPathAdjust::new(0, 4, true, 5.3, 5.3, 3);
        let snapped: Vec<f64> = (0..=4)
            .map(|i| {
                let (mut x, mut y) = (5.3, 0.0);
                stroke_adjust_point(&adj, i, &mut x, &mut y);
                x
            })
            .collect();
        assert_eq!(snapped, [5.99, 5.0, 5.0, 5.99, 5.99]);
        // Apart edges snap by value, as before.
        let apart = XPathAdjust::new(0, 4, true, 1.0, 3.0, 3);
        let (mut x, mut y) = (1.0, 0.0);
        stroke_adjust_point(&apart, 3, &mut x, &mut y);
        assert!((x - 1.0).abs() < 1e-10, "x={x}");
    }

    #[test]
    fn snaps_only_within_open_interval() {
        // A coordinate exactly on the window boundary must NOT be snapped,
        // because the windows are open intervals: v ∈ (x0a, x0b), not [x0a, x0b].
        let adj = XPathAdjust::new(0, 1, true, 1.0, 3.0, 1);
        // x0a = adj0 - 0.01 = 0.99 — exactly on the boundary, must not snap.
        let boundary = adj.x0a;
        let mut x = boundary;
        let mut y = 0.0;
        stroke_adjust(&adj, &mut x, &mut y);
        assert!(
            (x - boundary).abs() < 1e-15,
            "coordinate exactly at x0a boundary must not be snapped, got x={x}"
        );

        // Similarly for x0b = adj0 + 0.01 = 1.01.
        let mut x = adj.x0b;
        let boundary_b = adj.x0b;
        stroke_adjust(&adj, &mut x, &mut y);
        assert!(
            (x - boundary_b).abs() < 1e-15,
            "coordinate exactly at x0b boundary must not be snapped, got x={x}"
        );
    }
}
