//! Queries of a [`Clip`]: whether a pixel, span or rectangle passes it,
//! the runs of a row that pass it, and the scaling of a coverage row by it.
//!
//! A pixel passes when it lies in the rectangle and every aliased region,
//! and the product of the antialiased regions' coverages is at least one
//! half; a coverage row is scaled by each antialiased region's coverage
//! with `div255` and zeroed outside the rectangle and every aliased region.

use color::convert::div255;

use super::{Clip, ClipResult, Region};

impl Clip {
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
        let inside = |r: &Region| match r {
            Region::Aliased(s) => s.test_span(x0, x1, y),
            Region::Area(t) => t.is_full(x0, x1, y),
        };
        if self.regions().all(inside) {
            ClipResult::AllInside
        } else {
            ClipResult::Partial
        }
    }

    /// Scale `out[i]`, a coverage of pixel `(x + i, y)`, by the clip's
    /// coverage of that pixel: 0 outside the rectangle and outside an
    /// aliased path clip, `div255(out · c)` for the coverage `c` of each
    /// antialiased one.
    pub fn mul_row(&self, y: i32, x: i32, out: &mut [u8]) {
        let len = i64::try_from(out.len()).unwrap_or(i64::MAX);
        let end = i64::from(x) + len - 1;
        if y < self.y_min_i || y > self.y_max_i {
            out.fill(0);
            return;
        }
        // Columns outside the rectangle.
        let lo = (i64::from(self.x_min_i) - i64::from(x)).clamp(0, len);
        let hi = (end.min(i64::from(self.x_max_i)) - i64::from(x) + 1).clamp(lo, len);
        let (lo, hi) = (
            usize::try_from(lo).unwrap_or(0),
            usize::try_from(hi).unwrap_or(0),
        );
        out[..lo].fill(0);
        out[hi..].fill(0);
        for region in self.regions() {
            match region {
                Region::Area(t) => t.mul_row(y, x, out),
                Region::Aliased(s) => {
                    for (px, o) in (x..).zip(out.iter_mut()) {
                        if *o != 0 && !s.test(px, y) {
                            *o = 0;
                        }
                    }
                }
            }
        }
    }

    /// Whether `(x, y)` passes every path clip: it lies in each aliased
    /// region, and the product of the antialiased regions' coverages is at
    /// least one half.
    fn test_clip_paths(&self, x: i32, y: i32) -> bool {
        let mut coverage = u8::MAX;
        for region in self.regions() {
            match region {
                Region::Aliased(s) => {
                    if !s.test(x, y) {
                        return false;
                    }
                }
                Region::Area(t) => {
                    coverage = div255(u32::from(coverage) * u32::from(t.coverage(x, y)));
                }
            }
        }
        coverage >= 128
    }
}
