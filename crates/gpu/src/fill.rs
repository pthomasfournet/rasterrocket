//! Scanline-AA fill: the intersection-table input format and the host
//! model of the `aa_fill` kernel.
//!
//! The kernel does not derive geometry.  It consumes the scanline
//! intersection table the CPU rasteriser builds in AA (`AA_SIZE`×)
//! coordinates and walks it with the CPU's span semantics, so its per-pixel
//! sub-sample counts are the CPU's by construction.  [`aa_fill_cpu`] is the
//! operation-for-operation host twin used as the parity oracle for both the
//! CUDA and Vulkan kernels.

use cudarc::driver::DeviceRepr;

/// Sub-samples per pixel per axis.  Fixed at 4 by the kernels' 16-bit
/// coverage masks and by the intersection tables they consume.
pub const AA_SIZE: i32 = 4;

/// One scanline intersection, as consumed by the `aa_fill` kernel.
///
/// Device layout: three consecutive `i32` (`x0`, `x1`, `count`), 12 bytes,
/// matching the `int` triples the kernels index.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct ScanIntersect {
    /// Left AA column of the intersection span, inclusive.
    pub x0: i32,
    /// Right AA column of the intersection span, inclusive (`x0 <= x1`).
    pub x1: i32,
    /// Winding contribution: ±1 for crossing segments, 0 for horizontal ones.
    pub count: i32,
}

// SAFETY: ScanIntersect is repr(C) with three i32 fields and no padding.
unsafe impl bytemuck::Pod for ScanIntersect {}
unsafe impl bytemuck::Zeroable for ScanIntersect {}
// SAFETY: plain i32 fields with repr(C); representable on the device as-is.
unsafe impl DeviceRepr for ScanIntersect {}

/// A scanline intersection table in AA coordinates.
#[derive(Clone, Copy, Debug)]
pub struct AaScanTable<'a> {
    /// Intersections grouped by row, sorted by `x0` within each row.
    pub intersects: &'a [ScanIntersect],
    /// Row offsets: row `r` is `intersects[row_starts[r]..row_starts[r + 1]]`.
    /// Length `n_rows + 1`; the last entry equals `intersects.len()`.
    pub row_starts: &'a [u32],
    /// AA row index of table row 0.
    pub y_min: i32,
}

impl AaScanTable<'_> {
    /// Rows in the table.
    #[must_use]
    pub const fn n_rows(&self) -> usize {
        self.row_starts.len().saturating_sub(1)
    }

    /// Check the structural invariants the kernel relies on without
    /// bounds checks: at least one offset, offsets non-decreasing, and the
    /// last offset within `intersects`.
    ///
    /// # Panics
    ///
    /// Panics with a description of the violated invariant.
    pub fn assert_well_formed(&self) {
        assert!(
            !self.row_starts.is_empty(),
            "AaScanTable: row_starts needs at least the trailing sentinel"
        );
        assert!(
            self.row_starts.windows(2).all(|w| w[0] <= w[1]),
            "AaScanTable: row_starts must be non-decreasing"
        );
        let last = self.row_starts[self.row_starts.len() - 1] as usize;
        assert!(
            last <= self.intersects.len(),
            "AaScanTable: row_starts sentinel {last} exceeds {} intersections",
            self.intersects.len()
        );
    }

    /// The intersections of table row `r`, or an empty slice outside the table.
    #[must_use]
    pub fn row(&self, r: usize) -> &[ScanIntersect] {
        if r + 1 >= self.row_starts.len() {
            return &[];
        }
        &self.intersects[self.row_starts[r] as usize..self.row_starts[r + 1] as usize]
    }
}

/// The pixel rectangle a fill evaluates, in device pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AaFillRect {
    /// Left column, inclusive.
    pub x: i32,
    /// Top row, inclusive.
    pub y: i32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl AaFillRect {
    /// `width × height`, or `None` on overflow.
    #[must_use]
    pub const fn pixel_count(&self) -> Option<usize> {
        (self.width as usize).checked_mul(self.height as usize)
    }
}

/// Sub-sample bitmask of one AA row's coverage within the four AA columns
/// `[ax0, ax0 + AA_SIZE)`.
///
/// The walk is the CPU rasteriser's scanline walk: every intersection covers
/// its own cells `[x0, x1]` whatever its winding count, and a span absorbs
/// the following intersections while they touch it or the running count,
/// masked by `eo_mask`, is inside; a row that ends inside runs to
/// `aa_width`.  Spans are clamped to `[0, aa_width)`; overlapping spans
/// union.  The walk stops at the first intersection starting at or beyond
/// the pixel's last sub-sample, which is sound because the row is sorted by
/// `x0`.
fn row_coverage_mask(row: &[ScanIntersect], aa_width: i32, eo_mask: i32, ax0: i32) -> u32 {
    let mut mask = 0u32;
    let mut count = 0i32;
    let mut i = 0usize;

    while i < row.len() {
        let xx0 = row[i].x0;
        if xx0 >= ax0 + AA_SIZE {
            break;
        }
        let mut xx1 = row[i].x1;
        count = count.wrapping_add(row[i].count);
        i += 1;
        while i < row.len() && (row[i].x0 <= xx1 || (count & eo_mask) != 0) {
            xx1 = xx1.max(row[i].x1);
            count = count.wrapping_add(row[i].count);
            i += 1;
        }
        let span_x1 = if i >= row.len() && (count & eo_mask) != 0 {
            aa_width
        } else {
            xx1.wrapping_add(1)
        };

        let bx0 = xx0.max(0);
        let bx1 = span_x1.min(aa_width);
        if bx0 < bx1 {
            let lo = bx0.max(ax0);
            let hi = bx1.min(ax0 + AA_SIZE);
            if lo < hi {
                mask |= ((1u32 << (hi - lo)) - 1) << (lo - ax0);
            }
        }
    }
    mask
}

/// Sub-sample count (`0..=AA_SIZE²`) of device pixel `(x, y)`.
#[must_use]
pub fn pixel_coverage(table: &AaScanTable<'_>, aa_width: u32, eo: bool, x: i32, y: i32) -> u8 {
    let eo_mask = if eo { 1i32 } else { !0i32 };
    let aa_w = i32::try_from(aa_width).unwrap_or(i32::MAX);
    let ax0 = x * AA_SIZE;
    let ay0 = y * AA_SIZE;
    let mut total = 0u32;
    for yy in 0..AA_SIZE {
        let Ok(r) = usize::try_from(ay0 + yy - table.y_min) else {
            continue;
        };
        total += row_coverage_mask(table.row(r), aa_w, eo_mask, ax0).count_ones();
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "four 4-bit masks contribute at most 16"
    )]
    {
        total as u8
    }
}

/// Host twin of the `aa_fill` kernel.
///
/// Returns `rect.width × rect.height` bytes, row-major, each the sub-sample
/// count `0..=AA_SIZE²` of the corresponding pixel.  `aa_width` is the
/// bitmap width × `AA_SIZE`.
///
/// # Panics
///
/// Panics if `table` violates [`AaScanTable::assert_well_formed`] or the
/// rectangle's pixel count overflows `usize`.
#[must_use]
pub fn aa_fill_cpu(table: &AaScanTable<'_>, rect: AaFillRect, aa_width: u32, eo: bool) -> Vec<u8> {
    table.assert_well_formed();
    let n_pixels = rect.pixel_count().expect("width × height overflows usize");
    let mut out = Vec::with_capacity(n_pixels);
    for py in 0..rect.height {
        let y = rect.y + i32::try_from(py).expect("height fits i32");
        for px in 0..rect.width {
            let x = rect.x + i32::try_from(px).expect("width fits i32");
            out.push(pixel_coverage(table, aa_width, eo, x, y));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{AaFillRect, AaScanTable, ScanIntersect, aa_fill_cpu, pixel_coverage};

    const fn ix(x0: i32, x1: i32, count: i32) -> ScanIntersect {
        ScanIntersect { x0, x1, count }
    }

    /// Two-row table: row 0 covers AA columns 2..=9, row 1 is empty.
    fn two_row_table() -> (Vec<ScanIntersect>, Vec<u32>) {
        (vec![ix(2, 2, -1), ix(9, 9, 1)], vec![0, 2, 2])
    }

    #[test]
    fn empty_table_covers_nothing() {
        let table = AaScanTable {
            intersects: &[],
            row_starts: &[0],
            y_min: 0,
        };
        let rect = AaFillRect {
            x: 0,
            y: 0,
            width: 3,
            height: 2,
        };
        assert_eq!(aa_fill_cpu(&table, rect, 12, false), vec![0; 6]);
    }

    #[test]
    fn span_is_edge_inclusive_on_both_sides() {
        let (intersects, row_starts) = two_row_table();
        let table = AaScanTable {
            intersects: &intersects,
            row_starts: &row_starts,
            y_min: 0,
        };
        // Row 0 is sub-row 0 of pixel row 0: columns 2..=9 are covered, so
        // pixel 0 gets sub-samples {2, 3}, pixel 1 all four, pixel 2 {8, 9}.
        assert_eq!(pixel_coverage(&table, 16, false, 0, 0), 2);
        assert_eq!(pixel_coverage(&table, 16, false, 1, 0), 4);
        assert_eq!(pixel_coverage(&table, 16, false, 2, 0), 2);
        assert_eq!(pixel_coverage(&table, 16, false, 3, 0), 0);
    }

    #[test]
    fn unterminated_span_runs_to_aa_width_and_is_clamped() {
        let intersects = [ix(6, 6, 1)];
        let table = AaScanTable {
            intersects: &intersects,
            row_starts: &[0, 1],
            y_min: 0,
        };
        // Columns 6.. are covered up to aa_width = 10: pixel 1 gets {6, 7},
        // pixel 2 gets {8, 9}, pixel 3 nothing.
        assert_eq!(pixel_coverage(&table, 10, false, 1, 0), 2);
        assert_eq!(pixel_coverage(&table, 10, false, 2, 0), 2);
        assert_eq!(pixel_coverage(&table, 10, false, 3, 0), 0);
    }

    #[test]
    fn zero_count_intersections_cover_their_own_cells() {
        // A count-0 entry (a horizontal edge, or an edge's first scanline)
        // bounds the shape: its own cells are covered, nothing beyond.
        let intersects = [ix(0, 40, 0)];
        let table = AaScanTable {
            intersects: &intersects,
            row_starts: &[0, 1],
            y_min: 0,
        };
        for x in 0..10 {
            assert_eq!(pixel_coverage(&table, 64, false, x, 0), 4, "pixel {x}");
        }
        assert_eq!(pixel_coverage(&table, 64, false, 10, 0), 1, "pixel 10");
        assert_eq!(pixel_coverage(&table, 64, false, 11, 0), 0, "pixel 11");
    }

    #[test]
    fn fill_rule_masks_the_running_count() {
        // Nested squares with the same winding: NZ keeps the centre,
        // EO punches it out.
        let intersects = [ix(0, 0, 1), ix(8, 8, 1), ix(16, 16, -1), ix(24, 24, -1)];
        let table = AaScanTable {
            intersects: &intersects,
            row_starts: &[0, 4],
            y_min: 0,
        };
        // Pixel 3 = columns 12..=15, inside both squares.
        assert_eq!(pixel_coverage(&table, 32, false, 3, 0), 4);
        assert_eq!(pixel_coverage(&table, 32, true, 3, 0), 0);
        // Pixel 1 = columns 4..=7, inside the outer square only.
        assert_eq!(pixel_coverage(&table, 32, false, 1, 0), 4);
        assert_eq!(pixel_coverage(&table, 32, true, 1, 0), 4);
    }

    #[test]
    fn overlapping_spans_union() {
        // Two unmerged entries whose spans overlap: coverage is the union,
        // not the sum.
        let intersects = [ix(0, 5, 1), ix(2, 3, 1), ix(4, 4, -1), ix(7, 7, -1)];
        let table = AaScanTable {
            intersects: &intersects,
            row_starts: &[0, 4],
            y_min: 0,
        };
        assert_eq!(pixel_coverage(&table, 32, false, 0, 0), 4);
        assert_eq!(pixel_coverage(&table, 32, false, 1, 0), 4);
        assert_eq!(pixel_coverage(&table, 32, false, 2, 0), 0);
    }

    #[test]
    fn rows_outside_the_table_contribute_nothing() {
        let intersects = [ix(0, 0, 1), ix(100, 100, -1)];
        let table = AaScanTable {
            intersects: &intersects,
            row_starts: &[0, 2],
            y_min: 6,
        };
        // Pixel row 1 spans AA rows 4..=7; only row 6 is in the table.
        assert_eq!(pixel_coverage(&table, 128, false, 5, 1), 4);
        assert_eq!(pixel_coverage(&table, 128, false, 5, 0), 0);
        assert_eq!(pixel_coverage(&table, 128, false, 5, 2), 0);
    }

    #[test]
    fn rect_output_is_row_major_over_the_rectangle() {
        let (intersects, row_starts) = two_row_table();
        let table = AaScanTable {
            intersects: &intersects,
            row_starts: &row_starts,
            y_min: 0,
        };
        let rect = AaFillRect {
            x: 1,
            y: 0,
            width: 2,
            height: 1,
        };
        assert_eq!(aa_fill_cpu(&table, rect, 16, false), vec![4, 2]);
    }

    #[test]
    #[should_panic(expected = "non-decreasing")]
    fn malformed_row_starts_are_rejected() {
        let table = AaScanTable {
            intersects: &[],
            row_starts: &[1, 0],
            y_min: 0,
        };
        let rect = AaFillRect {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        };
        let _ = aa_fill_cpu(&table, rect, 4, false);
    }
}
