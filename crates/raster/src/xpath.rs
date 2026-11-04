//! Flattened, matrix-transformed path edge table.
//!
//! [`XPath`] converts a [`Path`] (in user space) into a sorted sequence of line segments
//! in device space, ready for scan conversion by [`crate::XPathScanner`].
//!
//! ## Typical usage pipeline
//!
//! ```text
//! XPath::new(path, matrix, flatness, close_subpaths)   ← construction
//!     │  internally calls add_segment for every flattened edge
//!     ▼
//! XPath                                                 ← device-space edge table
//!     │  optionally:
//!     ▼
//! xpath.aa_scale()                                      ← scale coords × AA_SIZE
//!     │  must be called AT MOST ONCE, before handing to XPathScanner
//!     ▼
//! XPathScanner                                          ← scan conversion
//! ```
//!
//! ## Key invariants (established by `XPath::add_segment` — internal)
//!
//! - For every non-horizontal segment, `y0 ≤ y1` after construction (swapped
//!   if necessary; [`XPathFlags::FLIPPED`] is set when a swap occurred).
//! - [`XPathFlags::HORIZ`] is set when `y0 == y1`.
//! - [`XPathFlags::VERT`] is set when `x0 == x1`.
//! - `dxdy = (x1-x0)/(y1-y0)` for sloped segments; 0.0 for horizontal/vertical.
//!   Division is safe because the HORIZ early-return guarantees `y0 ≠ y1` for
//!   any segment that reaches the slope computation.
//!
//! ## Affine transform convention
//!
//! ```text
//! x_out = x_in * m[0] + y_in * m[2] + m[4]
//! y_out = x_in * m[1] + y_in * m[3] + m[5]
//! ```
//! (column-vector convention.)

use crate::path::adjust::{XPathAdjust, stroke_adjust_point};
use crate::path::flatten::{CurveData, flatten_curve};
use crate::path::{Path, PathFlags, PathPoint, StrokeAdjustHint};
use crate::types::{AA_SIZE, floor_i32};
use bitflags::bitflags;

// ── XPathFlags ────────────────────────────────────────────────────────────────

bitflags! {
    /// Per-segment flags for [`XPathSeg`].
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
    pub struct XPathFlags: u32 {
        /// Horizontal segment: y0 == y1.
        const HORIZ   = 0x01;
        /// Vertical segment: x0 == x1.
        const VERT    = 0x02;
        /// Segment was flipped (original y0 > y1) to enforce y0 ≤ y1.
        const FLIPPED = 0x04;
        /// First segment of a subpath, in a path whose subpaths are closed.
        const SUBPATH = 0x08;
    }
}

// ── XPathSeg ──────────────────────────────────────────────────────────────────

/// A single line segment in the edge table, in device space.
///
/// # Invariant
///
/// After construction by `XPath::add_segment` (internal), every non-horizontal
/// segment satisfies `y0 ≤ y1`. Horizontal segments (`HORIZ` flag set) are exempt.
#[derive(Clone, Debug)]
pub struct XPathSeg {
    /// X coordinate of the segment start point, in device space.
    pub x0: f64,
    /// Y coordinate of the segment start point, in device space.
    ///
    /// **Invariant**: `y0 ≤ y1` for all non-horizontal segments after
    /// construction. [`XPathFlags::FLIPPED`] is set when the original input had
    /// `y0 > y1` and the endpoints were swapped to enforce this invariant.
    pub y0: f64,
    /// X coordinate of the segment end point, in device space.
    pub x1: f64,
    /// Y coordinate of the segment end point, in device space.
    ///
    /// Always `y0 ≤ y1` for non-horizontal segments after construction.
    pub y1: f64,
    /// Slope `(x1-x0)/(y1-y0)`, stored as both `f64` (for initialising the
    /// fixed-point accumulator) and as a **48.16 fixed-point `i64`** (`dxdy_fp`)
    /// for the per-scanline stepping hot loop.
    ///
    /// Set to `0.0` / `0` for horizontal and vertical segments.
    pub dxdy: f64,
    /// Slope in 48.16 fixed-point: `(dxdy * 65536.0).round()`, saturated.
    ///
    /// Used by the scanner's incremental x-accumulator: `xx_fp += dxdy_fp` per
    /// scanline.  The rounding error is at most 2⁻¹⁷ pixel per scanline, so
    /// under one pixel over the 32 768 rows (×4 under antialiasing) of the
    /// tallest page.
    pub dxdy_fp: i64,
    /// Orientation and flip flags; see [`XPathFlags`] for the set of valid bits.
    pub flags: XPathFlags,
}

impl XPathSeg {
    /// Whether any coordinate is NaN.
    #[must_use]
    pub fn has_nan(&self) -> bool {
        [self.x0, self.y0, self.x1, self.y1]
            .iter()
            .any(|v| v.is_nan())
    }

    /// Whether every coordinate is finite.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        [self.x0, self.y0, self.x1, self.y1]
            .iter()
            .all(|v| v.is_finite())
    }
}

// ── XPath ─────────────────────────────────────────────────────────────────────

/// Matrix-transformed, flattened edge table derived from a [`Path`].
///
/// # Construction pipeline
///
/// 1. Call [`XPath::new`] to build the edge table from a [`Path`].
/// 2. Optionally call [`XPath::aa_scale`] **once** to scale coordinates by
///    [`AA_SIZE`] for supersampled anti-aliasing.
/// 3. Hand the resulting `XPath` to `XPathScanner` for scan conversion.
///
/// Calling `aa_scale` more than once will multiply coordinates by `AA_SIZE`
/// again, producing incorrect results. This is not checked at runtime.
pub struct XPath {
    /// The flattened, transformed edge segments making up this path, in insertion order.
    pub segs: Vec<XPathSeg>,
    /// Whether every subpath is closed, its first segment flagged
    /// [`XPathFlags::SUBPATH`].
    closed: bool,
    /// Lazily allocated (~25 KB) scratch for Bezier subdivision.
    curve_data: Option<CurveData>,
}

impl XPath {
    /// Return a copy of this path with coordinates scaled by [`AA_SIZE`].
    ///
    /// The copy shares no state with `self` (the Bezier scratch is not
    /// carried over) and `self` remains in device space, so this is the
    /// safe form of [`XPath::aa_scale`] for consumers that must keep the
    /// device-space original.
    #[must_use]
    pub(crate) fn aa_scaled_copy(&self) -> Self {
        let mut scaled = Self {
            segs: self.segs.clone(),
            closed: self.closed,
            curve_data: None,
        };
        scaled.aa_scale();
        scaled
    }

    /// Create an empty `XPath` (for tests and internal use).
    #[cfg(test)]
    pub(crate) const fn empty() -> Self {
        Self {
            segs: Vec::new(),
            closed: false,
            curve_data: None,
        }
    }

    /// Build an `XPath` from a [`Path`] by applying `matrix` and flattening curves.
    ///
    /// # Arguments
    ///
    /// - `path`: the source path in user (pre-transform) space.
    /// - `matrix`: a 2-D affine transform `[a, b, c, d, e, f]` mapping user
    ///   space to device space (column-vector convention; see module docs).
    /// - `flatness`: maximum chord deviation (in device pixels) for Bezier
    ///   subdivision. Smaller values produce more accurate curves but more
    ///   segments. Typical range: `0.1`–`1.0`.
    /// - `close_subpaths`: if `true`, an implicit closing segment is added from
    ///   the last point of each subpath back to its first point when they do not
    ///   already coincide.
    ///
    /// # Ordering constraints
    ///
    /// After this call, `segs` is in insertion order (one entry per flattened
    /// edge). No further sorting is performed here; callers that need a sorted
    /// edge table must sort `segs` themselves or use `XPathScanner`.
    ///
    /// Calling [`XPath::aa_scale`] after this method scales all coordinates by
    /// [`AA_SIZE`]. It must be called at most once.
    #[must_use]
    pub fn new(path: &Path, matrix: &[f64; 6], flatness: f64, close_subpaths: bool) -> Self {
        let flatness_sq = flatness * flatness;
        let mut xpath = Self {
            segs: Vec::new(),
            closed: close_subpaths,
            curve_data: None,
        };

        // Transform every path point into device space.
        let tpts: Vec<PathPoint> = path
            .pts
            .iter()
            .map(|p| transform(matrix, p.x, p.y))
            .collect();

        let adjusts = build_adjusts(&path.hints, &tpts);

        // Apply stroke adjustments to the transformed points.
        let mut tpts = tpts;
        for adj in &adjusts {
            // `build_adjusts` keeps only ranges within `tpts`.
            for (idx, pt) in (adj.first_pt..).zip(&mut tpts[adj.first_pt..=adj.last_pt]) {
                stroke_adjust_point(adj, idx, &mut pt.x, &mut pt.y);
            }
        }

        // Walk the path and emit segments; one flattening buffer serves
        // every curve.
        let n = path.pts.len();
        let mut i = 0usize;
        let mut flat_pts = Vec::new();
        let mut budget_spent = false;
        while i < n {
            if path.flags[i].contains(PathFlags::FIRST) {
                // Start of a new subpath.
                let first_seg = xpath.segs.len();
                let sp_x = tpts[i].x;
                let sp_y = tpts[i].y;
                let mut cur_x = sp_x;
                let mut cur_y = sp_y;
                i += 1;
                // The subpath ends at a point flagged `LAST`, a curve whose
                // endpoint is, or the next subpath's first point.
                while i < n && !path.flags[i].contains(PathFlags::FIRST) {
                    if path.flags[i].contains(PathFlags::CURVE) {
                        // Cubic Bezier: consume 3 points (2 control + 1 endpoint).
                        if i + 2 >= n {
                            break;
                        }
                        let p0 = PathPoint::new(cur_x, cur_y);
                        let p1 = tpts[i];
                        let p2 = tpts[i + 1];
                        let p3 = tpts[i + 2];
                        flat_pts.clear();
                        if xpath.segs.len() < MAX_FLATTENED_SEGMENTS {
                            flatten_curve(
                                p0,
                                p1,
                                p2,
                                p3,
                                flatness_sq,
                                &mut flat_pts,
                                &mut xpath.curve_data,
                            );
                        } else {
                            if !budget_spent {
                                log::warn!(
                                    "a path flattens to over {MAX_FLATTENED_SEGMENTS} \
                                     segments; its remaining curves are drawn as chords"
                                );
                                budget_spent = true;
                            }
                            flat_pts.push(p3);
                        }
                        for fp in &flat_pts {
                            xpath.add_segment(cur_x, cur_y, fp.x, fp.y);
                            cur_x = fp.x;
                            cur_y = fp.y;
                        }
                        let is_last = path.flags[i + 2].contains(PathFlags::LAST);
                        i += 3;
                        if is_last {
                            break;
                        }
                    } else {
                        // Line segment.
                        let nx = tpts[i].x;
                        let ny = tpts[i].y;
                        xpath.add_segment(cur_x, cur_y, nx, ny);
                        cur_x = nx;
                        cur_y = ny;
                        let is_last = path.flags[i].contains(PathFlags::LAST);
                        i += 1;
                        if is_last {
                            break;
                        }
                    }
                }
                // A closing segment back to the start whenever the end
                // differs from it at all: every crossing of a closed subpath
                // then has its opposite on each scanline.
                #[expect(
                    clippy::float_cmp,
                    reason = "any difference, however small, leaves a crossing unbalanced"
                )]
                let open = cur_x != sp_x || cur_y != sp_y;
                if close_subpaths && open {
                    xpath.add_segment(cur_x, cur_y, sp_x, sp_y);
                }
                if close_subpaths && let Some(seg) = xpath.segs.get_mut(first_seg) {
                    seg.flags.insert(XPathFlags::SUBPATH);
                }
            } else {
                i += 1;
            }
        }

        xpath
    }

    /// Scale all segment coordinates by [`AA_SIZE`] for supersampled anti-aliasing.
    ///
    /// `dxdy` (the slope) is invariant under uniform scaling and is **not** modified —
    /// a uniform scale cancels out in `(x1-x0)/(y1-y0)`.
    ///
    /// # Ordering constraint
    ///
    /// This method must be called **after** [`XPath::new`] and **at most once**.
    /// Calling it a second time multiplies coordinates by [`AA_SIZE`] again,
    /// which will produce incorrect scan-conversion results. There is no runtime
    /// guard against double-scaling.
    ///
    /// Non-finite coordinates (from an overflowing or degenerate transform)
    /// stay non-finite; the scanner skips segments with a NaN coordinate.
    pub fn aa_scale(&mut self) {
        let s = f64::from(AA_SIZE);
        for seg in &mut self.segs {
            seg.x0 *= s;
            seg.y0 *= s;
            seg.x1 *= s;
            seg.y1 *= s;
        }
    }

    /// Remove the subpaths whose segments all lie in the columns left of
    /// `x_min` or right of `x_max`, or in the rows above `y_min` or below
    /// `y_max` (integer bounds, inclusive, in the path's coordinates).
    ///
    /// Only a path built with `close_subpaths` is changed.  A closed
    /// subpath crosses each scanline as often upwards as downwards, so its
    /// crossings sum to zero on every row: removing one that lies wholly
    /// outside the bounds changes neither the coverage nor the winding
    /// number of any point within them.  A subpath with a non-finite
    /// coordinate is kept, since the scanner skips its NaN segments and its
    /// remaining crossings need not sum to zero.
    pub fn cull_subpaths_outside(&mut self, x_min: i32, x_max: i32, y_min: i32, y_max: i32) {
        if !self.closed {
            return;
        }
        let inside = |(lx, hx, ly, hy): (f64, f64, f64, f64)| {
            floor_i32(lx) >= x_min
                && floor_i32(hx) <= x_max
                && floor_i32(ly) >= y_min
                && floor_i32(hy) <= y_max
        };
        // When the box of the non-NaN coordinates lies within the bounds, so
        // does every subpath that could be removed.
        if inside(bounding_box(&self.segs)) {
            return;
        }
        let outside = |segs: &[XPathSeg]| {
            if segs.is_empty() || !segs.iter().all(XPathSeg::is_finite) {
                return false;
            }
            let (lx, hx, ly, hy) = bounding_box(segs);
            floor_i32(hx) < x_min
                || floor_i32(lx) > x_max
                || floor_i32(hy) < y_min
                || floor_i32(ly) > y_max
        };
        let segs = std::mem::take(&mut self.segs);
        self.segs.reserve(segs.len());
        for subpath in segs.chunk_by(|_, next| !next.flags.contains(XPathFlags::SUBPATH)) {
            if !outside(subpath) {
                self.segs.extend_from_slice(subpath);
            }
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Append one line segment to the edge table, enforcing `y0 ≤ y1` for
    /// non-horizontal segments and computing `dxdy`.
    ///
    /// # y0 ≤ y1 invariant
    ///
    /// If the supplied `y0 > y1`, the endpoints are swapped and
    /// [`XPathFlags::FLIPPED`] is set. Horizontal segments (`y0 == y1`) are
    /// never flipped.
    ///
    /// # Division safety
    ///
    /// `dxdy = (x1-x0)/(y1-y0)` is computed only for segments that are neither
    /// horizontal nor vertical. The HORIZ early-return guarantees `y0 ≠ y1` for
    /// any segment that reaches this computation, so division by zero cannot
    /// occur. A `debug_assert!` enforces this contract in debug builds.
    fn add_segment(&mut self, mut x0: f64, mut y0: f64, mut x1: f64, mut y1: f64) {
        let mut flags = XPathFlags::empty();

        // Exact bit-equality is intentional: checking for axis-aligned segments
        // that were constructed with the same coordinate value.
        if y0.to_bits() == y1.to_bits() {
            // Horizontal segment: y0 == y1, dxdy is undefined; set to 0.0.
            flags.insert(XPathFlags::HORIZ);
            if x0.to_bits() == x1.to_bits() {
                flags.insert(XPathFlags::VERT);
            }
            self.segs.push(XPathSeg {
                x0,
                y0,
                x1,
                y1,
                dxdy: 0.0,
                dxdy_fp: 0,
                flags,
            });
            return; // Horizontal segments are NOT flipped.
        }

        // Non-horizontal: y0 ≠ y1 is guaranteed by the early return above.
        if x0.to_bits() == x1.to_bits() {
            flags.insert(XPathFlags::VERT);
        }

        // Compute slope before the potential swap so that the sign is
        // consistent with the *original* orientation. After the swap below,
        // the stored dxdy is the slope in the y0-ascending direction.
        let dxdy = if flags.contains(XPathFlags::VERT) {
            0.0
        } else {
            // Division is safe: HORIZ guard above guarantees y1 - y0 ≠ 0.0.
            debug_assert_ne!(
                y1.to_bits(),
                y0.to_bits(),
                "add_segment: y0 == y1 must be caught by the HORIZ branch"
            );
            (x1 - x0) / (y1 - y0)
        };

        if y0 > y1 {
            std::mem::swap(&mut x0, &mut x1);
            std::mem::swap(&mut y0, &mut y1);
            flags.insert(XPathFlags::FLIPPED);
        }

        // `as` saturates a float to the integer range.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the conversion saturates; a slope past 2^47 pixels per row is clamped"
        )]
        let dxdy_fp = (dxdy * 65536.0).round() as i64;

        self.segs.push(XPathSeg {
            x0,
            y0,
            x1,
            y1,
            dxdy,
            dxdy_fp,
            flags,
        });
    }
}

// ── Affine transform ──────────────────────────────────────────────────────────

/// Apply a 2-D affine matrix to point `(xi, yi)`, returning the transformed
/// [`PathPoint`] in device space.
///
/// Column-vector convention:
///
/// ```text
/// x_out = xi*m[0] + yi*m[2] + m[4]
/// y_out = xi*m[1] + yi*m[3] + m[5]
/// ```
///
/// Uses `f64::mul_add` for fused multiply-add, giving one rounding error per
/// term rather than two.
#[inline]
#[must_use]
pub const fn transform(m: &[f64; 6], xi: f64, yi: f64) -> PathPoint {
    PathPoint::new(
        xi.mul_add(m[0], yi.mul_add(m[2], m[4])),
        xi.mul_add(m[1], yi.mul_add(m[3], m[5])),
    )
}

// ── Stroke adjust record construction ────────────────────────────────────────

/// Build [`XPathAdjust`] records from path hints and transformed points.
///
/// Only axis-aligned hint pairs (both edges strictly horizontal or both
/// strictly vertical after transformation) are converted to adjust records;
/// skewed pairs are dropped, since snapping applies only to axis-aligned edges.
/// A hint whose indices fall outside `tpts`, whose range is inverted, or whose
/// edge points are not finite is dropped too.
fn build_adjusts(hints: &[StrokeAdjustHint], tpts: &[PathPoint]) -> Vec<XPathAdjust> {
    let mut adjusts = Vec::with_capacity(hints.len());
    for h in hints {
        // Each hint names two consecutive point pairs within its range.
        if h.ctrl0 + 1 >= tpts.len()
            || h.ctrl1 + 1 >= tpts.len()
            || h.last_pt >= tpts.len()
            || h.first_pt > h.last_pt
        {
            continue;
        }
        let p00 = tpts[h.ctrl0];
        let p01 = tpts[h.ctrl0 + 1];
        let p10 = tpts[h.ctrl1];
        let p11 = tpts[h.ctrl1 + 1];
        if ![p00, p01, p10, p11]
            .iter()
            .all(|p| p.x.is_finite() && p.y.is_finite())
        {
            continue;
        }
        // Determine orientation using bit-exact comparison (axis-aligned check).
        let vert = (p00.x.to_bits() == p01.x.to_bits()) && (p10.x.to_bits() == p11.x.to_bits());
        let horiz = (p00.y.to_bits() == p01.y.to_bits()) && (p10.y.to_bits() == p11.y.to_bits());
        if !vert && !horiz {
            continue;
        }
        // Edges of zero length are both: the pair lies apart across the
        // axis on which the edges differ, and is adjusted on that one.
        let vert = vert && !(horiz && (p10.y - p00.y).abs() > (p10.x - p00.x).abs());
        let (c0, c1) = if vert { (p00.x, p10.x) } else { (p00.y, p10.y) };
        // The edge at the larger coordinate takes the right target.
        let edge1 = if c1 >= c0 { h.ctrl1 } else { h.ctrl0 };
        adjusts.push(XPathAdjust::new(
            h.first_pt,
            h.last_pt,
            vert,
            c0.min(c1),
            c0.max(c1),
            edge1,
        ));
    }
    adjusts
}

/// Segments a path's curves are flattened into at most; past it each curve
/// is its chord, so a path's memory stays bounded whatever its curves.
const MAX_FLATTENED_SEGMENTS: usize = 1 << 22;

/// The bounds of no points: every point extends them.
const NO_BOUNDS: (f64, f64, f64, f64) = (
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::INFINITY,
    f64::NEG_INFINITY,
);

/// The smallest box `(x_min, x_max, y_min, y_max)` holding the endpoints of
/// `segs`, NaN coordinates aside; [`NO_BOUNDS`] when there are none.
fn bounding_box(segs: &[XPathSeg]) -> (f64, f64, f64, f64) {
    // `f64::min` and `max` return the other operand of a NaN.
    segs.iter().fold(NO_BOUNDS, |(lx, hx, ly, hy), s| {
        (
            lx.min(s.x0).min(s.x1),
            hx.max(s.x0).max(s.x1),
            ly.min(s.y0).min(s.y1),
            hy.max(s.y0).max(s.y1),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::path::PathBuilder;

    fn identity() -> [f64; 6] {
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    }

    /// The device segments of the closed rectangle `x0,y0`–`x0+w,y1`, fill
    /// adjusted.
    fn adjusted_rect(x0: f64, w: f64, y0: f64, y1: f64) -> Vec<(f64, f64, f64, f64)> {
        let mut b = PathBuilder::new();
        b.move_to(x0, y0).unwrap();
        b.line_to(x0 + w, y0).unwrap();
        b.line_to(x0 + w, y1).unwrap();
        b.line_to(x0, y1).unwrap();
        b.close(true).unwrap();
        let path = b.build().with_fill_adjust_hints().unwrap();
        XPath::new(&path, &identity(), 0.1, true)
            .segs
            .iter()
            .map(|s| (s.x0, s.y0, s.x1, s.y1))
            .collect()
    }

    #[test]
    fn subpaths_ending_in_curves_stay_apart() {
        // Two closed loops drawn with curves alone, 100 px apart: no segment
        // may join them.
        let mut b = PathBuilder::new();
        for x in [0.0, 100.0] {
            b.move_to(x, 0.0).unwrap();
            b.curve_to(x + 10.0, 0.0, x + 10.0, 10.0, x, 10.0).unwrap();
            b.curve_to(x - 10.0, 10.0, x - 10.0, 0.0, x, 0.0).unwrap();
            b.close(false).unwrap();
        }
        let path = b.build();
        let xpath = XPath::new(&path, &identity(), 0.1, true);
        for s in &xpath.segs {
            assert!(
                (s.x0 < 50.0) == (s.x1 < 50.0),
                "segment {:.1},{:.1} -> {:.1},{:.1} joins the loops",
                s.x0,
                s.y0,
                s.x1,
                s.y1
            );
        }
    }

    #[test]
    fn a_zero_width_rectangle_fills_one_pixel_column() {
        // Its vertical edges coincide; they snap apart to the column's two
        // sides, and its zero-length top and bottom edges stay rows.
        for w in [0.0, 0.01] {
            let segs = adjusted_rect(10.2, w, 10.0, 70.0);
            let near = |a: f64, b: f64| (a - b).abs() < 1e-9;
            let xs: Vec<f64> = segs.iter().flat_map(|s| [s.0, s.2]).collect();
            assert!(
                xs.iter().all(|&x| near(x, 10.0) || near(x, 10.99)),
                "{w}: {segs:?}"
            );
            assert!(
                xs.iter().any(|&x| near(x, 10.0)) && xs.iter().any(|&x| near(x, 10.99)),
                "{w}: {segs:?}"
            );
            // No edge runs diagonally.
            assert!(
                segs.iter().all(|s| near(s.0, s.2) || near(s.1, s.3)),
                "{w}: {segs:?}"
            );
        }
    }

    /// The device segments of the closed quadrilateral `pts`, fill adjusted
    /// when `adjust`.
    fn quad_segs(pts: [(f64, f64); 4], adjust: bool) -> Vec<(f64, f64, f64, f64)> {
        let mut b = PathBuilder::new();
        b.move_to(pts[0].0, pts[0].1).unwrap();
        for &(x, y) in &pts[1..] {
            b.line_to(x, y).unwrap();
        }
        b.close(true).unwrap();
        let path = b.build();
        let path = if adjust {
            path.with_fill_adjust_hints().unwrap()
        } else {
            path
        };
        XPath::new(&path, &identity(), 0.1, true)
            .segs
            .iter()
            .map(|s| (s.x0, s.y0, s.x1, s.y1))
            .collect()
    }

    #[test]
    fn a_trapezoid_snaps_its_horizontal_edges_and_keeps_its_slanted_sides() {
        // A one-pixel band whose ends slant by 0.1 px: the horizontal pair
        // snaps to row boundaries as a rectangle's would, and the slanted
        // pair, not axis-aligned, keeps its slant.
        let segs = quad_segs(
            [(10.0, 10.3), (50.0, 10.3), (50.1, 11.3), (10.1, 11.3)],
            true,
        );
        let snapped = |y: f64| {
            [10.0, 10.99, 11.0, 11.99]
                .iter()
                .any(|r| (y - r).abs() < 1e-9)
        };
        assert!(
            segs.iter().all(|s| snapped(s.1) && snapped(s.3)),
            "{segs:?}"
        );
        assert!(
            segs.iter()
                .any(|s| (s.0 - s.2).abs() > 1e-9 && (s.1 - s.3).abs() > 1e-9),
            "a side lost its slant: {segs:?}"
        );
    }

    #[test]
    fn a_quadrilateral_without_axis_aligned_edges_is_not_adjusted() {
        // Every edge slants: no pair qualifies, and the fill is the path's.
        let pts = [(10.0, 10.3), (50.0, 10.4), (50.1, 11.4), (10.1, 11.3)];
        assert_eq!(quad_segs(pts, true), quad_segs(pts, false));
    }

    #[test]
    fn horizontal_segment_not_flipped() {
        let mut xpath = XPath::empty();
        xpath.add_segment(0.0, 5.0, 10.0, 5.0);
        let s = &xpath.segs[0];
        assert!(s.flags.contains(XPathFlags::HORIZ));
        assert!(!s.flags.contains(XPathFlags::FLIPPED));
        assert!((s.y0 - 5.0).abs() < f64::EPSILON);
        assert!((s.y1 - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn downward_segment_flipped() {
        let mut xpath = XPath::empty();
        xpath.add_segment(0.0, 10.0, 0.0, 0.0); // y0 > y1 → flip
        let s = &xpath.segs[0];
        assert!(s.flags.contains(XPathFlags::FLIPPED));
        assert!(s.y0 <= s.y1, "y0={} y1={}", s.y0, s.y1);
    }

    #[test]
    fn aa_scale_multiplies_coords() {
        let mut xpath = XPath::empty();
        xpath.add_segment(1.0, 0.0, 3.0, 2.0);
        let orig_dxdy = xpath.segs[0].dxdy;
        xpath.aa_scale();
        let s = &xpath.segs[0];
        assert!((s.x0 - 4.0).abs() < 1e-10);
        assert!((s.y0 - 0.0).abs() < 1e-10);
        assert!((s.x1 - 12.0).abs() < 1e-10);
        assert!((s.y1 - 8.0).abs() < 1e-10);
        assert!(
            (s.dxdy - orig_dxdy).abs() < 1e-10,
            "dxdy should be unchanged"
        );
    }

    #[test]
    fn vertical_segment_dxdy_zero() {
        let mut xpath = XPath::empty();
        xpath.add_segment(5.0, 0.0, 5.0, 10.0);
        let s = &xpath.segs[0];
        assert!(s.flags.contains(XPathFlags::VERT));
        assert!(s.dxdy.abs() < f64::EPSILON);
    }

    #[test]
    fn triangle_from_path() {
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0).unwrap();
        b.line_to(4.0, 0.0).unwrap();
        b.line_to(2.0, 4.0).unwrap();
        b.close(false).unwrap();
        let path = b.build();
        let xpath = XPath::new(&path, &identity(), 1.0, false);
        // 3 explicit segments + 1 closing → but PathBuilder's close() already
        // adds the closing lineTo, so 3 segments total.
        assert_eq!(xpath.segs.len(), 3);
    }

    /// A degenerate point segment (x0==x1, y0==y1) should set both HORIZ and
    /// VERT flags and not panic.
    #[test]
    fn degenerate_point_segment() {
        let mut xpath = XPath::empty();
        xpath.add_segment(3.0, 7.0, 3.0, 7.0);
        let s = &xpath.segs[0];
        assert!(s.flags.contains(XPathFlags::HORIZ));
        assert!(s.flags.contains(XPathFlags::VERT));
        assert!(!s.flags.contains(XPathFlags::FLIPPED));
        assert_eq!(s.dxdy.to_bits(), 0.0_f64.to_bits());
    }

    /// `dxdy` must equal `(x1-x0)/(y1-y0)` for a sloped segment.
    #[test]
    fn sloped_segment_dxdy() {
        let mut xpath = XPath::empty();
        // y0 < y1, not vertical → dxdy = (6-2)/(5-1) = 1.0
        xpath.add_segment(2.0, 1.0, 6.0, 5.0);
        let s = &xpath.segs[0];
        assert!(!s.flags.contains(XPathFlags::HORIZ));
        assert!(!s.flags.contains(XPathFlags::VERT));
        assert!(!s.flags.contains(XPathFlags::FLIPPED));
        assert!((s.dxdy - 1.0).abs() < f64::EPSILON, "dxdy={}", s.dxdy);
    }

    /// A flipped sloped segment must have the same absolute dxdy value as its
    /// unflipped counterpart, and must satisfy y0 ≤ y1 after construction.
    #[test]
    fn flipped_sloped_segment_dxdy_consistent() {
        let mut xpath = XPath::empty();
        // Supply (x0=6, y0=5) → (x1=2, y1=1): y0 > y1, so it will be flipped.
        xpath.add_segment(6.0, 5.0, 2.0, 1.0);
        let s = &xpath.segs[0];
        assert!(s.flags.contains(XPathFlags::FLIPPED));
        assert!(
            s.y0 <= s.y1,
            "y0 ≤ y1 invariant violated: y0={} y1={}",
            s.y0,
            s.y1
        );
        // dxdy = (x1_orig - x0_orig)/(y1_orig - y0_orig) = (2-6)/(1-5) = 1.0
        assert!((s.dxdy - 1.0).abs() < f64::EPSILON, "dxdy={}", s.dxdy);
    }

    #[test]
    fn dxdy_fp_matches_dxdy_for_slope_one() {
        // Slope = 1.0 → dxdy_fp should be 65536.
        let mut xpath = XPath::empty();
        xpath.add_segment(0.0, 0.0, 4.0, 4.0);
        let s = &xpath.segs[0];
        assert!((s.dxdy - 1.0).abs() < f64::EPSILON, "dxdy={}", s.dxdy);
        assert_eq!(s.dxdy_fp, 65536, "dxdy_fp={}", s.dxdy_fp);
    }

    #[test]
    fn dxdy_fp_matches_dxdy_for_half_slope() {
        // Slope = 0.5 → dxdy_fp should be 32768.
        let mut xpath = XPath::empty();
        xpath.add_segment(0.0, 0.0, 2.0, 4.0);
        let s = &xpath.segs[0];
        assert!((s.dxdy - 0.5).abs() < f64::EPSILON, "dxdy={}", s.dxdy);
        assert_eq!(s.dxdy_fp, 32768, "dxdy_fp={}", s.dxdy_fp);
    }

    #[test]
    fn dxdy_fp_zero_for_horizontal() {
        let mut xpath = XPath::empty();
        xpath.add_segment(0.0, 2.0, 5.0, 2.0);
        let s = &xpath.segs[0];
        assert!(s.flags.contains(XPathFlags::HORIZ));
        assert_eq!(s.dxdy_fp, 0);
    }

    #[test]
    fn dxdy_fp_zero_for_vertical() {
        let mut xpath = XPath::empty();
        xpath.add_segment(3.0, 0.0, 3.0, 5.0);
        let s = &xpath.segs[0];
        assert!(s.flags.contains(XPathFlags::VERT));
        assert_eq!(s.dxdy_fp, 0);
    }

    /// Subpaths: four rectangles wholly outside `[0, 15]²` (left, right,
    /// above, below), one straddling its left edge, a triangle inside, and a
    /// rectangle overlapping the triangle.
    fn culling_test_path() -> Path {
        let mut b = PathBuilder::new();
        let rects = [
            (-30.0, 2.0, -20.0, 12.0),
            (20.0, 2.0, 28.0, 12.0),
            (2.0, -12.0, 12.0, -3.0),
            (2.0, 17.0, 12.0, 25.0),
            (-6.0, 5.0, 4.5, 9.5),
            (6.0, 6.0, 13.0, 14.0),
        ];
        for (x0, y0, x1, y1) in rects {
            b.move_to(x0, y0).unwrap();
            b.line_to(x1, y0).unwrap();
            b.line_to(x1, y1).unwrap();
            b.line_to(x0, y1).unwrap();
            b.close(false).unwrap();
        }
        b.move_to(1.0, 1.0).unwrap();
        b.line_to(14.0, 3.0).unwrap();
        b.line_to(8.0, 12.0).unwrap();
        b.build()
    }

    #[test]
    fn culling_drops_only_subpaths_outside_the_bounds() {
        let path = culling_test_path();
        let full = XPath::new(&path, &identity(), 1.0, true);
        let mut culled = XPath::new(&path, &identity(), 1.0, true);
        culled.cull_subpaths_outside(0, 15, 0, 15);
        assert_eq!(full.segs.len(), 6 * 4 + 3);
        assert_eq!(culled.segs.len(), 2 * 4 + 3);
        for eo in [false, true] {
            let a = crate::XPathScanner::new(&full, eo, 0, 15);
            let b = crate::XPathScanner::new(&culled, eo, 0, 15);
            for y in 0..=15 {
                for x in 0..=15 {
                    assert_eq!(a.test(x, y), b.test(x, y), "eo={eo} ({x}, {y})");
                }
            }
        }
    }

    #[test]
    fn culling_keeps_open_and_non_finite_subpaths() {
        let path = culling_test_path();
        let mut open = XPath::new(&path, &identity(), 1.0, false);
        let n = open.segs.len();
        open.cull_subpaths_outside(0, 15, 0, 15);
        assert_eq!(open.segs.len(), n, "an unclosed path is not culled");

        let mut within = XPath::new(&path, &identity(), 1.0, true);
        let n = within.segs.len();
        within.cull_subpaths_outside(-40, 40, -40, 40);
        assert_eq!(within.segs.len(), n, "nothing lies outside wider bounds");

        let mut b = PathBuilder::new();
        b.move_to(-30.0, 2.0).unwrap();
        b.line_to(-20.0, f64::NAN).unwrap();
        b.line_to(-25.0, 9.0).unwrap();
        let mut nan = XPath::new(&b.build(), &identity(), 1.0, true);
        let n = nan.segs.len();
        nan.cull_subpaths_outside(0, 15, 0, 15);
        assert_eq!(nan.segs.len(), n, "a subpath with a NaN point is kept");
    }
}
