//! Clip region: an axis-aligned rectangle intersected with zero or more
//! arbitrary path clip regions.
//!
//! ## Sharing semantics
//!
//! The path-clip scanners form a persistent singly linked list, newest first:
//! each node owns one [`XPathScanner`] and an [`Arc`] to the list it was
//! intersected onto.  [`Clip::clone_shared`] (a graphics-state save) copies the
//! head pointer, so a save costs O(1) whatever the number of path clips, and
//! an intersection allocates one node shared by no other clip.  A list that
//! reaches [`FOLD_DEPTH`] nodes is replaced by one node holding their
//! intersection, so a list never holds more than `FOLD_DEPTH - 1` nodes.
//! [`XPathScanner`] instances are immutable after construction, so sharing
//! across threads and across `clone_shared` copies is safe: there is no
//! interior mutability in the shared objects.

use std::sync::Arc;

use crate::bitmap::AaBuf;
use crate::scanner::XPathScanner;
use crate::simd;
use crate::types::{AA_SIZE, ceil_i32, floor_i32};
use crate::xpath::XPath;

/// Path clips a [`Clip`] holds at most: intersecting the one that reaches
/// this many replaces the list with a single scanner of their intersection
/// ([`XPathScanner::intersection`]), which every query answers exactly as
/// the list did, so the per-span cost of a clip stays bounded however many
/// paths it intersects.
const FOLD_DEPTH: usize = 8;

// ── ClipResult ────────────────────────────────────────────────────────────────

/// Result of a rectangular or span clip test.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ClipResult {
    /// The entire tested region lies within the clip boundary.
    AllInside,
    /// The entire tested region lies outside the clip boundary.
    AllOutside,
    /// The tested region straddles the clip boundary; per-pixel testing is required.
    Partial,
}

// ── Clip ──────────────────────────────────────────────────────────────────────

/// A clipping region combining an axis-aligned rectangle with an optional
/// stack of arbitrary path clips.
///
/// The effective clip is the intersection of the rectangle and all path clips.
pub struct Clip {
    /// Whether anti-aliasing (supersampling) is enabled; scales coordinates by
    /// [`AA_SIZE`] when testing path clips.
    pub antialias: bool,
    /// Left edge of the clip rectangle in floating-point device space (inclusive).
    pub x_min: f64,
    /// Top edge of the clip rectangle in floating-point device space (inclusive).
    pub y_min: f64,
    /// Right edge of the clip rectangle in floating-point device space (exclusive).
    pub x_max: f64,
    /// Bottom edge of the clip rectangle in floating-point device space (exclusive).
    pub y_max: f64,
    /// Integer pixel column of the left clip edge: `floor(x_min)`.
    pub x_min_i: i32,
    /// Integer pixel row of the top clip edge: `floor(y_min)`.
    pub y_min_i: i32,
    /// Integer pixel column of the right clip edge: `ceil(x_max) - 1`.
    pub x_max_i: i32,
    /// Integer pixel row of the bottom clip edge: `ceil(y_max) - 1`.
    pub y_max_i: i32,
    /// Head of the path-clip list, newest first; `None` for a rectangle-only
    /// clip.
    ///
    /// Shared across [`clone_shared`](Clip::clone_shared) copies via [`Arc`].
    /// [`XPathScanner`] is immutable after construction so no interior-mutability
    /// hazard exists.
    scanners: Option<Arc<ClipNode>>,
}

/// One path clip of a [`Clip`] and the clips it was intersected with.
struct ClipNode {
    scanner: XPathScanner,
    parent: Option<Arc<Self>>,
}

impl Clip {
    /// Create a new clip region from a rectangle.
    #[must_use]
    pub fn new(x0: f64, y0: f64, x1: f64, y1: f64, antialias: bool) -> Self {
        let mut clip = Self {
            antialias,
            x_min: 0.0,
            y_min: 0.0,
            x_max: 0.0,
            y_max: 0.0,
            x_min_i: 0,
            y_min_i: 0,
            x_max_i: 0,
            y_max_i: 0,
            scanners: None,
        };
        clip.set_rect(x0, y0, x1, y1);
        clip
    }

    /// Clone this `Clip`, sharing all path-clip scanners via [`Arc`].
    ///
    /// The cloned value and the original share the same [`XPathScanner`]
    /// instances. Because scanners are immutable after construction there is
    /// no interior-mutability hazard.
    #[must_use]
    pub fn clone_shared(&self) -> Self {
        Self {
            antialias: self.antialias,
            x_min: self.x_min,
            y_min: self.y_min,
            x_max: self.x_max,
            y_max: self.y_max,
            x_min_i: self.x_min_i,
            y_min_i: self.y_min_i,
            x_max_i: self.x_max_i,
            y_max_i: self.y_max_i,
            scanners: self.scanners.clone(),
        }
    }

    /// This clip moved by `dx` columns and `dy` rows of device pixels.  The
    /// path clips' intersection tables shift by whole cells, so a pixel
    /// `(x, y)` passes them exactly when `(x − dx, y − dy)` passes this
    /// clip's; the rectangle's bounds move by `dx` and `dy` in `f64`.
    ///
    /// # Panics
    ///
    /// Panics if the shift in the supersampled AA grid overflows `i32`.
    #[must_use]
    pub fn translated(&self, dx: i32, dy: i32) -> Self {
        let (sx, sy) = if self.antialias {
            (
                dx.checked_mul(AA_SIZE)
                    .expect("Clip::translated: dx * AA_SIZE overflows i32"),
                dy.checked_mul(AA_SIZE)
                    .expect("Clip::translated: dy * AA_SIZE overflows i32"),
            )
        } else {
            (dx, dy)
        };
        let (fx, fy) = (f64::from(dx), f64::from(dy));
        let mut clip = Self {
            antialias: self.antialias,
            x_min: self.x_min + fx,
            y_min: self.y_min + fy,
            x_max: self.x_max + fx,
            y_max: self.y_max + fy,
            x_min_i: 0,
            y_min_i: 0,
            x_max_i: 0,
            y_max_i: 0,
            scanners: None,
        };
        // The list rebuilt with each scanner shifted, oldest first, so the
        // newest stays at its head.
        #[expect(
            clippy::needless_collect,
            reason = "the list iterates newest first and is rebuilt oldest first; its iterator is not double-ended"
        )]
        let shifted: Vec<XPathScanner> = self.scanners().map(|s| s.translated(sx, sy)).collect();
        clip.scanners = shifted.into_iter().rev().fold(None, |parent, scanner| {
            Some(Arc::new(ClipNode { scanner, parent }))
        });
        clip.recompute_int_bounds();
        clip
    }

    /// True when the clip is its rectangle alone, with no path clips.
    #[must_use]
    pub const fn is_rect(&self) -> bool {
        self.scanners.is_none()
    }

    /// Replace the clip rectangle and clear all path clips.
    pub fn reset_to_rect(&mut self, x0: f64, y0: f64, x1: f64, y1: f64) {
        self.set_rect(x0, y0, x1, y1);
        self.scanners = None;
    }

    /// Intersect the clip rectangle with `[x0, y0, x1, y1]`.
    pub fn clip_to_rect(&mut self, x0: f64, y0: f64, x1: f64, y1: f64) {
        let (lx, rx) = (x0.min(x1), x0.max(x1));
        let (ly, ry) = (y0.min(y1), y0.max(y1));
        self.x_min = self.x_min.max(lx);
        self.x_max = self.x_max.min(rx);
        self.y_min = self.y_min.max(ly);
        self.y_max = self.y_max.min(ry);
        self.recompute_int_bounds();
    }

    /// Intersect with an arbitrary path clip.
    ///
    /// If the path resolves to a simple axis-aligned rectangle (4 segments,
    /// axis-aligned), it is reduced to `clip_to_rect`. Otherwise a new
    /// [`XPathScanner`] is pushed onto the scanner stack, and a stack that
    /// reaches [`FOLD_DEPTH`] scanners is folded into one.
    ///
    /// An empty path forces the clip to be empty (nothing passes through).
    ///
    /// # Panics
    ///
    /// Panics if the AA y-range arithmetic overflows `i32`.
    /// In practice `y_max_i` is bounded by the bitmap height (≪ `i32::MAX / AA_SIZE`).
    pub fn clip_to_path(&mut self, xpath: &XPath, eo: bool) {
        if xpath.segs.is_empty() {
            // Force empty: nothing passes.
            self.x_max = self.x_min - 1.0;
            self.y_max = self.y_min - 1.0;
            self.recompute_int_bounds();
            return;
        }
        // Detect axis-aligned rect (4 segments, 2 horiz + 2 vert, forming a closed box).
        if let Some((rx0, ry0, rx1, ry1)) = detect_rect(xpath) {
            self.clip_to_rect(rx0, ry0, rx1, ry1);
            return;
        }
        // General path clip: compute scanline range in (possibly scaled) space.
        let (y_lo, y_hi) = if self.antialias {
            self.aa_row_range()
        } else {
            (self.y_min_i, self.y_max_i)
        };
        // The scanner is queried in AA space when antialiasing (see
        // `aa_coords` and `clip_aa_line`), so it must be built from an
        // AA-scaled copy; `xpath` itself stays in device space, which
        // `detect_rect` above relies on.
        let scanner = if self.antialias {
            XPathScanner::new(&xpath.aa_scaled_copy(), eo, y_lo, y_hi)
        } else {
            XPathScanner::new(xpath, eo, y_lo, y_hi)
        };
        self.scanners = Some(Arc::new(ClipNode {
            scanner,
            parent: self.scanners.take(),
        }));
        if self.scanners().nth(FOLD_DEPTH - 1).is_some() {
            let folded = XPathScanner::intersection(self.scanners());
            self.scanners = Some(Arc::new(ClipNode {
                scanner: folded,
                parent: None,
            }));
        }
    }

    /// The clip's rows in the supersampled AA grid, inclusive:
    /// `[y_min_i × AA_SIZE, (y_max_i + 1) × AA_SIZE − 1]`.
    ///
    /// # Panics
    ///
    /// Panics if the range overflows `i32`; the clip rows are bounded by the
    /// bitmap height, far below `i32::MAX / AA_SIZE`.
    #[must_use]
    pub fn aa_row_range(&self) -> (i32, i32) {
        aa_range(self.y_min_i, self.y_max_i)
    }

    /// The clip's columns in the supersampled AA grid, inclusive:
    /// `[x_min_i × AA_SIZE, (x_max_i + 1) × AA_SIZE − 1]`.
    ///
    /// # Panics
    ///
    /// Panics if the range overflows `i32`; the clip columns are bounded by
    /// the bitmap width, far below `i32::MAX / AA_SIZE`.
    #[must_use]
    pub fn aa_column_range(&self) -> (i32, i32) {
        aa_range(self.x_min_i, self.x_max_i)
    }

    /// The maximal runs `(a, b)`, inclusive and left to right, of the pixels
    /// `x0..=x1` on row `y` that pass [`Self::test`].
    pub fn inside_runs(&self, x0: i32, x1: i32, y: i32) -> impl Iterator<Item = (i32, i32)> + '_ {
        // The next column to examine; always within `x0..=x1`.
        let mut next = (x0 <= x1).then_some(x0);
        std::iter::from_fn(move || {
            let mut x = next?;
            while !self.test(x, y) {
                if x == x1 {
                    next = None;
                    return None;
                }
                x += 1;
            }
            let start = x;
            while x < x1 && self.test(x + 1, y) {
                x += 1;
            }
            next = (x < x1).then(|| x + 1);
            Some((start, x))
        })
    }

    // ── Pixel-level tests ─────────────────────────────────────────────────────

    /// Test whether pixel `(x, y)` is inside the clip region.
    ///
    /// Returns `false` immediately if `(x, y)` is outside the axis-aligned
    /// rectangle; otherwise all path-clip scanners are consulted.
    #[inline]
    #[must_use]
    pub fn test(&self, x: i32, y: i32) -> bool {
        if x < self.x_min_i || x > self.x_max_i || y < self.y_min_i || y > self.y_max_i {
            return false;
        }
        self.test_clip_paths(x, y)
    }

    /// Test a pixel rectangle against the clip region.
    ///
    /// The rectangle is inclusive on both ends: `[left, right] × [top, bottom]`.
    #[must_use]
    pub fn test_rect(&self, left: i32, top: i32, right: i32, bottom: i32) -> ClipResult {
        // Half-open pixel rect: [left, right+1) × [top, bottom+1).
        // Clip rect: [x_min, x_max) × [y_min, y_max).
        if (f64::from(right) + 1.0) <= self.x_min
            || f64::from(left) >= self.x_max
            || (f64::from(bottom) + 1.0) <= self.y_min
            || f64::from(top) >= self.y_max
        {
            return ClipResult::AllOutside;
        }
        if f64::from(left) >= self.x_min
            && (f64::from(right) + 1.0) <= self.x_max
            && f64::from(top) >= self.y_min
            && (f64::from(bottom) + 1.0) <= self.y_max
            && self.scanners.is_none()
        {
            return ClipResult::AllInside;
        }
        ClipResult::Partial
    }

    /// Test whether the span `[x0, x1]` on scanline `y` is fully inside the clip.
    ///
    /// Returns [`ClipResult::AllInside`] only when the span is inside both the
    /// bounding rectangle and every path-clip scanner. Returns
    /// [`ClipResult::AllOutside`] when the span is fully outside the rectangle.
    /// Otherwise returns [`ClipResult::Partial`].
    #[must_use]
    pub fn test_span(&self, x0: i32, x1: i32, y: i32) -> ClipResult {
        let result = self.test_rect(x0, y, x1, y);
        if result != ClipResult::AllInside {
            return result;
        }
        for scanner in self.scanners() {
            let (sx0, sx1, sy) = aa_coords(x0, x1, y, self.antialias);
            if !scanner.test_span(sx0, sx1, sy) {
                return ClipResult::Partial;
            }
        }
        ClipResult::AllInside
    }

    /// Clip an AA buffer row, zeroing bits outside the clip region.
    ///
    /// Each path-clip
    /// scanner intersects its coverage with the sub-samples already in
    /// `aa_buf` over `[*x0, *x1]`, and the output span is then clamped to the
    /// integer clip bounds.
    pub fn clip_aa_line(&self, aa_buf: &mut AaBuf, x0: &mut i32, x1: &mut i32, y: i32) {
        for scanner in self.scanners() {
            scanner.clip_aa_line(aa_buf, *x0, *x1, y);
        }
        *x0 = (*x0).max(self.x_min_i);
        *x1 = (*x1).min(self.x_max_i);
    }

    // ── Private ───────────────────────────────────────────────────────────────

    fn set_rect(&mut self, x0: f64, y0: f64, x1: f64, y1: f64) {
        self.x_min = x0.min(x1);
        self.x_max = x0.max(x1);
        self.y_min = y0.min(y1);
        self.y_max = y0.max(y1);
        self.recompute_int_bounds();
    }

    fn recompute_int_bounds(&mut self) {
        self.x_min_i = floor_i32(self.x_min);
        self.y_min_i = floor_i32(self.y_min);
        self.x_max_i = ceil_i32(self.x_max) - 1;
        self.y_max_i = ceil_i32(self.y_max) - 1;
    }

    fn test_clip_paths(&self, x: i32, y: i32) -> bool {
        let (tx, _, ty) = aa_coords(x, x, y, self.antialias);
        self.scanners().all(|s| s.test(tx, ty))
    }

    /// The path-clip scanners, newest first.  Every consumer intersects their
    /// regions, so the order does not affect any result.
    fn scanners(&self) -> impl Iterator<Item = &XPathScanner> {
        std::iter::successors(self.scanners.as_deref(), |node| node.parent.as_deref())
            .map(|node| &node.scanner)
    }
}

// ── Row coverage ──────────────────────────────────────────────────────────────

/// Per-pixel coverage of a clip over a fixed column span, row by row.
///
/// The span's bounding box is classified once against the clip; rows and
/// then pixels are resolved only as finely as that classification requires.
/// A row the clip edge crosses yields the same 0–255 shape a fill's edge
/// would: with an antialiased clip, every sub-sample of the span is set,
/// intersected with the clip by [`Clip::clip_aa_line`], counted per pixel and
/// mapped through the AA gamma table.  A rectangle-only clip has integer
/// bounds, so its columns come out fully covered or uncovered.  A
/// non-antialiased clip uses the binary pixel test.
pub struct ClipRows<'a> {
    clip: &'a Clip,
    bbox: ClipResult,
    /// Inclusive column span, clamped to non-negative columns.
    x0: i32,
    x1: i32,
    /// Sub-sample scratch, allocated on the first row the clip edge crosses.
    aa_buf: Option<AaBuf>,
    /// Per-column coverage of the current partial row, indexed from `x0`.
    coverage: Vec<u8>,
}

impl<'a> ClipRows<'a> {
    /// Coverage of `clip` over the inclusive rectangle `[x0, x1] × [y0, y1]`.
    /// Columns left of zero and right of the clip rectangle are outside.
    #[must_use]
    pub fn new(clip: &'a Clip, x0: i32, y0: i32, x1: i32, y1: i32) -> Self {
        let x0 = x0.max(0);
        let bbox = if x0 > x1 || y0 > y1 {
            ClipResult::AllOutside
        } else {
            clip.test_rect(x0, y0, x1, y1)
        };
        Self {
            clip,
            bbox,
            x0,
            x1,
            aa_buf: None,
            coverage: Vec::new(),
        }
    }

    /// Classification of the whole rectangle.
    #[must_use]
    pub const fn bbox(&self) -> ClipResult {
        self.bbox
    }

    /// Coverage of row `y` over the span, or `None` when the clip excludes
    /// the entire row.
    pub fn row(&mut self, y: i32) -> Option<RowClip<'_>> {
        let res = match self.bbox {
            ClipResult::Partial => self.clip.test_span(self.x0, self.x1, y),
            other => other,
        };
        match res {
            ClipResult::AllOutside => None,
            ClipResult::AllInside => Some(RowClip::Full),
            ClipResult::Partial => {
                self.partial_coverage(y);
                Some(RowClip::Partial {
                    x0: self.x0,
                    coverage: &self.coverage,
                })
            }
        }
    }

    /// Fill `self.coverage` for a row the clip edge crosses.
    #[expect(
        clippy::cast_sign_loss,
        reason = "0 ≤ x0 ≤ lo ≤ hi, and the clipped span lies within [lo, hi]"
    )]
    fn partial_coverage(&mut self, y: i32) {
        self.coverage.clear();
        // Columns outside the clip rectangle have no coverage, and
        // `RowClip::coverage` reads columns past the vector as 0, so the work
        // and the scratch are bounded by the rectangle, not by the span.
        let (lo, hi) = (
            self.x0.max(self.clip.x_min_i),
            self.x1.min(self.clip.x_max_i),
        );
        if lo > hi {
            return;
        }
        self.coverage.resize((hi - self.x0 + 1) as usize, 0);
        if !self.clip.antialias {
            for (x, cov) in (lo..=hi).zip(&mut self.coverage[(lo - self.x0) as usize..]) {
                if self.clip.test(x, y) {
                    *cov = 255;
                }
            }
            return;
        }
        let aa_size = AA_SIZE as usize;
        let buf = self
            .aa_buf
            .get_or_insert_with(|| AaBuf::new(hi as usize + 1));
        buf.clear();
        for sub_row in 0..buf.height {
            buf.set_span(sub_row, lo as usize * aa_size, (hi as usize + 1) * aa_size);
        }
        let (mut cx0, mut cx1) = (lo, hi);
        self.clip.clip_aa_line(buf, &mut cx0, &mut cx1, y);
        let (cx0, cx1) = (cx0.max(lo), cx1.min(hi));
        if cx0 > cx1 {
            return;
        }
        let rows = [
            buf.row_slice(0),
            buf.row_slice(1),
            buf.row_slice(2),
            buf.row_slice(3),
        ];
        let span = &mut self.coverage[(cx0 - self.x0) as usize..=(cx1 - self.x0) as usize];
        simd::aa_coverage_span(rows, cx0 as usize, span);
        // An uncovered span maps to zeros, which callers skip; the
        // any-covered flag adds nothing here.
        let _any_covered = simd::apply_aa_gamma(span);
    }
}

/// Clip coverage of the pixels on one row of a [`ClipRows`] span.
#[derive(Clone, Copy, Debug)]
pub enum RowClip<'a> {
    /// The whole row lies inside the clip.
    Full,
    /// The clip edge crosses the row; `coverage[i]` is the shape (0–255) of
    /// device column `x0 + i`.
    Partial {
        /// Device column of `coverage[0]`.
        x0: i32,
        /// Per-column shape.
        coverage: &'a [u8],
    },
}

impl RowClip<'_> {
    /// Clip coverage (0 = outside, 255 = inside) of device column `x`.
    #[inline]
    #[must_use]
    pub fn coverage(&self, x: i32) -> u8 {
        match self {
            Self::Full => 255,
            Self::Partial { x0, coverage } => usize::try_from(x - x0)
                .ok()
                .and_then(|i| coverage.get(i))
                .copied()
                .unwrap_or(0),
        }
    }
}

// ── AA coordinate scaling ─────────────────────────────────────────────────────

/// Scale pixel coordinates to the supersampled AA grid when `antialias` is set.
///
/// Returns `(sx0, sx1, sy)` where:
/// - `sx0 = x0 * AA_SIZE` if AA, else `x0`
/// - `sx1 = x1 * AA_SIZE + (AA_SIZE - 1)` if AA, else `x1`
/// - `sy  = y  * AA_SIZE` if AA, else `y`
///
/// The expanded `sx1` covers all supersampled sub-pixels within device pixel `x1`.
///
/// # Panics
///
/// Panics if a scaled coordinate overflows `i32`; pixel coordinates are bounded
/// by the bitmap dimensions, which are far below `i32::MAX / AA_SIZE`.
#[inline]
fn aa_coords(x0: i32, x1: i32, y: i32, antialias: bool) -> (i32, i32, i32) {
    if antialias {
        let sx0 = x0
            .checked_mul(AA_SIZE)
            .expect("aa_coords: x0 * AA_SIZE overflows i32");
        let sx1 = x1
            .checked_mul(AA_SIZE)
            .and_then(|v| v.checked_add(AA_SIZE - 1))
            .expect("aa_coords: x1 * AA_SIZE + (AA_SIZE-1) overflows i32");
        let sy = y
            .checked_mul(AA_SIZE)
            .expect("aa_coords: y * AA_SIZE overflows i32");
        (sx0, sx1, sy)
    } else {
        (x0, x1, y)
    }
}

// ── Rectangle detection ───────────────────────────────────────────────────────

/// Detect whether an `XPath` is an axis-aligned rectangle.
///
/// Returns `Some((x0, y0, x1, y1))` giving the bounding box of the rectangle
/// if the path consists of exactly 4 axis-aligned segments (2 vertical + 2
/// horizontal). Returns `None` for any other path.
fn detect_rect(xpath: &XPath) -> Option<(f64, f64, f64, f64)> {
    use crate::xpath::XPathFlags;
    if xpath.segs.len() != 4 {
        return None;
    }
    let segs = &xpath.segs;
    // Need exactly 2 vertical + 2 horizontal segments.
    let verts = segs
        .iter()
        .filter(|s| s.flags.contains(XPathFlags::VERT))
        .count();
    let horizs = segs
        .iter()
        .filter(|s| s.flags.contains(XPathFlags::HORIZ))
        .count();
    if verts != 2 || horizs != 2 {
        return None;
    }
    // Extract x extents from vertical segments and y extents from horizontal
    // segments by folding directly — no intermediate allocation.
    let vert_xs = segs
        .iter()
        .filter(|s| s.flags.contains(XPathFlags::VERT))
        .flat_map(|s| [s.x0, s.x1]);
    let horiz_ys = segs
        .iter()
        .filter(|s| s.flags.contains(XPathFlags::HORIZ))
        .flat_map(|s| [s.y0, s.y1]);

    let (x0, x1) = vert_xs.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
        (lo.min(v), hi.max(v))
    });
    let (y0, y1) = horiz_ys.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
        (lo.min(v), hi.max(v))
    });

    Some((x0, y0, x1, y1))
}

/// The supersampled AA cells `[lo × AA_SIZE, (hi + 1) × AA_SIZE − 1]` of the
/// pixels `lo..=hi`.
fn aa_range(lo: i32, hi: i32) -> (i32, i32) {
    let lo = lo
        .checked_mul(AA_SIZE)
        .expect("AA range: lower bound × AA_SIZE overflows i32");
    let hi = hi
        .checked_add(1)
        .and_then(|v| v.checked_mul(AA_SIZE))
        .map(|v| v - 1)
        .expect("AA range: (upper bound + 1) × AA_SIZE overflows i32");
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::path::PathBuilder;

    /// An antialiased path clip must select the device-space region the
    /// path covers, not that region scaled down by `AA_SIZE`.
    #[test]
    fn aa_path_clip_selects_device_space_region() {
        let mut clip = Clip::new(0.0, 0.0, 64.0, 64.0, true);

        // Diamond centred on (16, 16): non-axis-aligned, so it stays a
        // scanner instead of reducing to a rectangle.
        let mut builder = PathBuilder::new();
        builder.move_to(16.0, 2.0).expect("move_to");
        builder.line_to(30.0, 16.0).expect("line_to");
        builder.line_to(16.0, 30.0).expect("line_to");
        builder.line_to(2.0, 16.0).expect("line_to");
        builder.close(false).expect("close");
        let path = builder.build();
        let xpath = XPath::new(&path, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 0.1, true);
        clip.clip_to_path(&xpath, false);
        assert_eq!(
            clip.scanners().count(),
            1,
            "diamond must become a scanner clip"
        );

        assert!(clip.test(16, 16), "diamond centre must be inside the clip");
        assert!(clip.test(16, 8), "upper interior point must be inside");
        assert!(clip.test(9, 16), "left interior point must be inside");
        assert!(clip.test(23, 16), "right interior point must be inside");
        assert!(clip.test(16, 24), "lower interior point must be inside");
        assert!(
            !clip.test(3, 3),
            "corner outside the diamond must be clipped"
        );
        assert!(
            !clip.test(4, 4),
            "point inside the AA_SIZE-shrunk ghost of the diamond must be clipped"
        );
    }

    /// Diamond with vertices at `(cx, cy ± r)` and `(cx ± r, cy)`, clipped
    /// into `clip` as a path.
    fn clip_to_diamond(clip: &mut Clip, cx: f64, cy: f64, r: f64) {
        let mut builder = PathBuilder::new();
        builder.move_to(cx, cy - r).expect("move_to");
        builder.line_to(cx + r, cy).expect("line_to");
        builder.line_to(cx, cy + r).expect("line_to");
        builder.line_to(cx - r, cy).expect("line_to");
        builder.close(false).expect("close");
        let xpath = XPath::new(&builder.build(), &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 0.1, true);
        clip.clip_to_path(&xpath, false);
    }

    /// The path of a random star polygon of 5 to 12 vertices around
    /// `(32, 32)`, its radii between 20 and 48 so it is concave and crosses
    /// the clip rectangle's edges; one in three visits its vertices out of
    /// angular order, crossing itself.  Drawn from the generator state `seed`.
    fn random_polygon(seed: &mut u64) -> XPath {
        let mut next = || {
            *seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            #[expect(clippy::cast_precision_loss, reason = "31-bit value")]
            let unit = (*seed >> 33) as f64 / f64::from(1u32 << 31);
            unit
        };
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "0..8"
        )]
        let n = 5 + (next() * 8.0) as usize;
        let crossing = next() < 1.0 / 3.0;
        let mut angles: Vec<f64> = (0..n).map(|_| next() * std::f64::consts::TAU).collect();
        if !crossing {
            angles.sort_by(f64::total_cmp);
        }
        let mut builder = PathBuilder::new();
        for (k, angle) in angles.into_iter().enumerate() {
            let r = next().mul_add(28.0, 20.0);
            let (x, y) = (r.mul_add(angle.cos(), 32.0), r.mul_add(angle.sin(), 32.0));
            if k == 0 {
                builder.move_to(x, y).expect("move_to");
            } else {
                builder.line_to(x, y).expect("line_to");
            }
        }
        builder.close(false).expect("close");
        XPath::new(&builder.build(), &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 0.1, true)
    }

    #[test]
    fn a_folded_clip_answers_as_the_conjunction_of_its_paths() {
        let mut seed = 7;
        // Trials whose intersection is neither empty nor the whole rectangle.
        let mut partly_inside = 0;
        for antialias in [false, true] {
            for eo in [false, true] {
                for _ in 0..12 {
                    let paths: Vec<XPath> = (0..10).map(|_| random_polygon(&mut seed)).collect();
                    let mut folded = Clip::new(0.0, 0.0, 64.0, 64.0, antialias);
                    let singles: Vec<Clip> = paths
                        .iter()
                        .map(|path| {
                            folded.clip_to_path(path, eo);
                            let mut single = Clip::new(0.0, 0.0, 64.0, 64.0, antialias);
                            single.clip_to_path(path, eo);
                            single
                        })
                        .collect();
                    assert!(folded.scanners().count() < FOLD_DEPTH);
                    let inside = (0..64)
                        .flat_map(|y| (0..64).map(move |x| (x, y)))
                        .filter(|&(x, y)| folded.test(x, y))
                        .count();
                    if (1..64 * 64).contains(&inside) {
                        partly_inside += 1;
                    }
                    for y in -2..66 {
                        for x in -2..66 {
                            assert_eq!(
                                folded.test(x, y),
                                singles.iter().all(|c| c.test(x, y)),
                                "test ({x}, {y}), aa {antialias}, eo {eo}"
                            );
                        }
                        for x0 in (-2..66).step_by(3) {
                            for len in [0, 1, 4, 9, 30] {
                                let results: Vec<ClipResult> = singles
                                    .iter()
                                    .map(|c| c.test_span(x0, x0 + len, y))
                                    .collect();
                                let want = if results.iter().all(|&r| r == ClipResult::AllInside) {
                                    ClipResult::AllInside
                                } else if results[0] == ClipResult::AllOutside {
                                    // The rectangle, shared by every clip, excludes the span.
                                    ClipResult::AllOutside
                                } else {
                                    ClipResult::Partial
                                };
                                assert_eq!(
                                    folded.test_span(x0, x0 + len, y),
                                    want,
                                    "test_span {x0}..={} on {y}",
                                    x0 + len
                                );
                            }
                        }
                        if antialias {
                            let full = |buf: &mut AaBuf| {
                                for row in 0..buf.height {
                                    buf.set_span(row, 0, 70 * AA_SIZE as usize);
                                }
                            };
                            let mut got = AaBuf::new(70);
                            full(&mut got);
                            let (mut gx0, mut gx1) = (0, 69);
                            folded.clip_aa_line(&mut got, &mut gx0, &mut gx1, y);
                            let mut want = AaBuf::new(70);
                            full(&mut want);
                            // Every scanner of a clip is given the caller's
                            // columns; only the result is clamped.
                            let (mut wx0, mut wx1) = (0, 69);
                            for single in &singles {
                                let (mut x0, mut x1) = (0, 69);
                                single.clip_aa_line(&mut want, &mut x0, &mut x1, y);
                                (wx0, wx1) = (x0, x1);
                            }
                            assert_eq!((gx0, gx1), (wx0, wx1));
                            for row in 0..AA_SIZE as usize {
                                for x in 0..70 * AA_SIZE as usize {
                                    assert_eq!(
                                        aa_bit(&got, row, x),
                                        aa_bit(&want, row, x),
                                        "clip_aa_line y {y} sub-row {row} sample {x}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        // Of the 48 trials, enough must cross a boundary of the intersection.
        assert!(
            partly_inside >= 20,
            "{partly_inside} of 48 trials had a boundary"
        );
    }

    /// Whether AA sub-sample `x` of sub-row `row` is set.
    fn aa_bit(buf: &AaBuf, row: usize, x: usize) -> bool {
        buf.get_byte(row, x / 8) & (0x80 >> (x % 8)) != 0
    }

    /// `clip_aa_line` intersects the row already in the buffer with the
    /// clip: sub-samples inside the path clip keep their value, those outside
    /// are cleared, and sub-samples that were never set stay unset.
    #[test]
    fn clip_aa_line_intersects_with_the_buffer() {
        let mut clip = Clip::new(0.0, 0.0, 64.0, 64.0, true);
        clip_to_diamond(&mut clip, 16.0, 16.0, 14.0);

        // Row 16 crosses the diamond over device x ≈ 2..30 (AA 8..120).  The
        // buffer holds a fill over AA 40..200 only.
        let mut buf = AaBuf::new(64);
        for row in 0..buf.height {
            buf.set_span(row, 40, 200);
        }
        let (mut x0, mut x1) = (0, 63);
        clip.clip_aa_line(&mut buf, &mut x0, &mut x1, 16);

        for row in 0..buf.height {
            assert!(
                !aa_bit(&buf, row, 20),
                "inside clip, never set: stays clear"
            );
            assert!(aa_bit(&buf, row, 64), "inside clip and fill: kept");
            assert!(!aa_bit(&buf, row, 160), "fill outside the clip: cleared");
        }
    }

    /// Nested path clips intersect: a sub-sample survives only when every
    /// clip covers it.
    #[test]
    fn clip_aa_line_nested_clips_intersect() {
        let mut clip = Clip::new(0.0, 0.0, 64.0, 64.0, true);
        clip_to_diamond(&mut clip, 16.0, 16.0, 14.0);
        clip_to_diamond(&mut clip, 28.0, 16.0, 14.0);
        assert_eq!(clip.scanners().count(), 2);

        let mut buf = AaBuf::new(64);
        for row in 0..buf.height {
            buf.set_span(row, 0, 256);
        }
        let (mut x0, mut x1) = (0, 63);
        clip.clip_aa_line(&mut buf, &mut x0, &mut x1, 16);

        for row in 0..buf.height {
            assert!(!aa_bit(&buf, row, 4 * 8), "only in the first diamond");
            assert!(aa_bit(&buf, row, 4 * 22), "in both diamonds");
            assert!(!aa_bit(&buf, row, 4 * 36), "only in the second diamond");
        }
    }

    /// A translated clip passes and clips exactly what the clip passes and
    /// clips at the shifted coordinates, pixel by pixel and sub-sample by
    /// sub-sample.
    #[test]
    fn translated_clip_tests_the_shifted_pixels() {
        let (dx, dy) = (-9, -5);
        for antialias in [false, true] {
            let mut clip = Clip::new(2.5, 1.0, 60.0, 40.25, antialias);
            clip_to_diamond(&mut clip, 24.0, 18.0, 13.5);
            let moved = clip.translated(dx, dy);
            assert_eq!(
                (moved.x_min_i, moved.y_min_i, moved.x_max_i, moved.y_max_i),
                (
                    clip.x_min_i + dx,
                    clip.y_min_i + dy,
                    clip.x_max_i + dx,
                    clip.y_max_i + dy
                )
            );
            for y in 0..48 {
                for x in 0..64 {
                    assert_eq!(moved.test(x + dx, y + dy), clip.test(x, y), "({x}, {y})");
                }
            }
            if antialias {
                for y in 0..48 {
                    let mut a = AaBuf::new(64);
                    let mut b = AaBuf::new(64);
                    for row in 0..a.height {
                        a.set_span(row, 0, 256);
                        b.set_span(row, 0, 256);
                    }
                    let (mut ax0, mut ax1) = (9, 63);
                    clip.clip_aa_line(&mut a, &mut ax0, &mut ax1, y);
                    let (mut bx0, mut bx1) = (0, 54);
                    moved.clip_aa_line(&mut b, &mut bx0, &mut bx1, y + dy);
                    assert_eq!((ax0 + dx, ax1 + dx), (bx0, bx1), "row {y}");
                    for row in 0..a.height {
                        for sx in 36..256 {
                            assert_eq!(
                                aa_bit(&a, row, sx),
                                aa_bit(&b, row, sx - 36),
                                "row {y}, sub-row {row}, sub-sample {sx}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn new_clip_rect_bounds() {
        let c = Clip::new(1.5, 2.5, 10.5, 8.5, false);
        assert_eq!(c.x_min_i, 1); // floor(1.5) = 1
        assert_eq!(c.y_min_i, 2); // floor(2.5) = 2
        assert_eq!(c.x_max_i, 10); // ceil(10.5) - 1 = 11 - 1 = 10
        assert_eq!(c.y_max_i, 8); // ceil(8.5) - 1 = 9 - 1 = 8
    }

    #[test]
    fn test_inside() {
        let c = Clip::new(0.0, 0.0, 10.0, 10.0, false);
        assert!(c.test(5, 5));
    }

    #[test]
    fn test_outside() {
        let c = Clip::new(0.0, 0.0, 10.0, 10.0, false);
        assert!(!c.test(15, 5));
        assert!(!c.test(5, 15));
    }

    #[test]
    fn clip_to_rect_shrinks() {
        let mut c = Clip::new(0.0, 0.0, 10.0, 10.0, false);
        c.clip_to_rect(2.0, 3.0, 8.0, 7.0);
        assert_eq!(c.x_min_i, 2);
        assert_eq!(c.y_min_i, 3);
    }

    #[test]
    fn test_rect_all_inside() {
        let c = Clip::new(0.0, 0.0, 20.0, 20.0, false);
        assert_eq!(c.test_rect(1, 1, 5, 5), ClipResult::AllInside);
    }

    #[test]
    fn test_rect_all_outside() {
        let c = Clip::new(0.0, 0.0, 10.0, 10.0, false);
        assert_eq!(c.test_rect(15, 15, 20, 20), ClipResult::AllOutside);
    }

    /// A save shares the whole path-clip list (one pointer, not one `Arc`
    /// per clip), and intersecting the copy leaves the original unchanged.
    #[test]
    fn clone_shares_the_path_clip_list() {
        let mut c = Clip::new(0.0, 0.0, 64.0, 64.0, false);
        for i in 0..3 {
            clip_to_diamond(&mut c, 32.0, 32.0, 20.0 + f64::from(i));
        }
        let mut saved = c.clone_shared();
        assert_eq!(saved.x_min_i, c.x_min_i);
        let (head, saved_head) = (c.scanners.as_ref(), saved.scanners.as_ref());
        assert!(head.zip(saved_head).is_some_and(|(a, b)| Arc::ptr_eq(a, b)));
        assert_eq!(head.map(Arc::strong_count), Some(2));

        clip_to_diamond(&mut saved, 10.0, 32.0, 6.0);
        assert_eq!(c.scanners().count(), 3);
        assert_eq!(saved.scanners().count(), 4);
        // (32, 32) is inside the three shared diamonds but outside the
        // fourth, (10, 32) the other way round.
        assert!(c.test(32, 32) && !saved.test(32, 32));
        assert!(!c.test(10, 32) && !saved.test(10, 32));
        assert!(c.test(30, 32) && !saved.test(30, 32));
        assert!(saved.test(14, 32) && c.test(14, 32));
    }

    /// Dropping a long uniquely owned path-clip list does not recurse once
    /// per clip.
    #[test]
    fn a_clip_holds_at_most_fold_depth_scanners() {
        let mut c = Clip::new(0.0, 0.0, 4.0, 4.0, false);
        for _ in 0..10_000 {
            clip_to_diamond(&mut c, 2.0, 2.0, 1.5);
            assert!(c.scanners().count() < FOLD_DEPTH);
        }
        assert!(c.test(2, 2));
    }

    #[test]
    fn aa_coords_non_aa_passthrough() {
        assert_eq!(aa_coords(3, 7, 5, false), (3, 7, 5));
    }

    #[test]
    fn aa_coords_aa_scales() {
        // AA_SIZE = 4
        // sx0 = 3 * 4 = 12
        // sx1 = 7 * 4 + 3 = 31
        // sy  = 5 * 4 = 20
        assert_eq!(aa_coords(3, 7, 5, true), (12, 31, 20));
    }

    /// `inside_runs` yields the maximal runs inside the clip, including runs
    /// touching either end of the span, and nothing for an empty span.
    #[test]
    fn inside_runs_are_maximal_and_ordered() {
        let mut clip = Clip::new(0.0, 0.0, 64.0, 64.0, false);
        clip_to_diamond(&mut clip, 16.0, 16.0, 6.0);
        let runs: Vec<_> = clip.inside_runs(0, 63, 16).collect();
        assert_eq!(runs.len(), 1, "{runs:?}");
        let (a, b) = runs[0];
        assert!((10..=11).contains(&a) && (21..=22).contains(&b), "{runs:?}");
        assert!((a..=b).all(|x| clip.test(x, 16)));
        assert_eq!(clip.inside_runs(a, b, 16).collect::<Vec<_>>(), [(a, b)]);
        assert_eq!(clip.inside_runs(5, 4, 16).count(), 0);
        assert_eq!(clip.inside_runs(0, 5, 16).count(), 0);

        let rect = Clip::new(0.0, 0.0, 8.0, 8.0, false);
        assert_eq!(rect.inside_runs(-3, 20, 2).collect::<Vec<_>>(), [(0, 7)]);
    }

    /// A rectangle reaching `i32::MAX`, a saturated non-finite coordinate,
    /// is classified without overflowing.
    #[test]
    fn test_rect_accepts_i32_max_edges() {
        let c = Clip::new(0.0, 0.0, 10.0, 10.0, false);
        assert_eq!(c.test_rect(2, 2, i32::MAX, i32::MAX), ClipResult::Partial);
        assert_eq!(
            c.test_rect(i32::MAX, 2, i32::MAX, 3),
            ClipResult::AllOutside
        );
    }
}
