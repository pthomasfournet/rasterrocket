//! Coalescing span iterator over a scanner's intersection row.
//!
//! [`crate::ScanIterator`] yields the covered spans of one scanline as
//! inclusive `(x0, x1)` pixel ranges, under the even-odd or non-zero
//! winding rule.
//!
//! ## Coverage
//!
//! Every intersection covers its own cells `[x0, x1]`, whatever its winding
//! count: a horizontal edge, or a sloped or vertical edge on the scanline
//! where it starts, carries count 0 yet still bounds the shape.  A span
//! starts at an intersection and absorbs the following ones while they
//! overlap it or the running winding count is inside.  The covered cells
//! are therefore the union of the intersections' cells and of the gaps
//! between intersections where the winding count is inside, whatever order
//! intersections with equal `x0` are stored in.
//!
//! ## Winding-rule semantics
//!
//! The iterator keeps the running `count`, the sum of the
//! [`Intersect::count`] values consumed so far:
//!
//! - **Non-zero winding** (`eo = false`): inside when `count != 0`.
//!   `eo_mask = !0` so `count & eo_mask` is non-zero iff `count != 0`.
//! - **Even-odd** (`eo = true`): inside when the crossing count is odd.
//!   `eo_mask = 1` so `count & eo_mask` isolates the low bit (parity).
//!
//! A row whose intersections end inside the shape (unbalanced winding)
//! yields a last span reaching `i32::MAX`.

use super::{Intersect, XPathScanner};

/// An iterator that yields `(x0, x1)` inclusive pixel spans for one scanline.
///
/// Spans are emitted in left-to-right order with no overlap.
///
/// Constructed via [`ScanIterator::new`]. Out-of-range `y` values produce an
/// immediately-exhausted iterator.
pub struct ScanIterator<'a> {
    /// The sorted intersection list for this scanline.
    row: &'a [Intersect],
    /// Index of the next un-consumed entry in `row`.
    idx: usize,
    /// Running winding count: sum of `Intersect::count` for all consumed entries.
    count: i32,
    /// Fill-rule mask.
    ///
    /// `!0` for non-zero winding (`count != 0` → inside); `1` for even-odd
    /// (`count & 1 != 0` → inside). See module-level doc for details.
    eo_mask: i32,
}

impl<'a> ScanIterator<'a> {
    /// Create an iterator for scanline `y` of `scanner`.
    ///
    /// If `y` is outside the scanner's range, the iterator is immediately
    /// exhausted.
    #[must_use]
    pub fn new(scanner: &'a XPathScanner, y: i32) -> Self {
        let row = scanner.row(y);
        let eo_mask = if scanner.eo { 1i32 } else { !0i32 };
        Self {
            row,
            idx: 0,
            count: 0,
            eo_mask,
        }
    }

    const fn inside(&self) -> bool {
        (self.count & self.eo_mask) != 0
    }
}

impl Iterator for ScanIterator<'_> {
    /// `(x0, x1)` — both inclusive pixel coordinates.
    type Item = (i32, i32);

    /// The next covered span: from the next intersection, through every
    /// following one that overlaps it or is reached while inside.
    fn next(&mut self) -> Option<(i32, i32)> {
        let first = self.row.get(self.idx)?;
        let x0 = first.x0;
        let mut x1 = first.x1;
        self.count = self.count.wrapping_add(first.count);
        self.idx += 1;
        while let Some(next) = self.row.get(self.idx) {
            if next.x0 > x1 && !self.inside() {
                break;
            }
            x1 = x1.max(next.x1);
            self.count = self.count.wrapping_add(next.count);
            self.idx += 1;
        }
        if self.idx == self.row.len() && self.inside() {
            x1 = i32::MAX;
        }
        Some((x0, x1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::path::PathBuilder;
    use crate::scanner::XPathScanner;
    use crate::xpath::XPath;

    fn identity() -> [f64; 6] {
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    }

    fn rect_xpath(x0: f64, y0: f64, x1: f64, y1: f64) -> XPath {
        let mut b = PathBuilder::new();
        b.move_to(x0, y0).unwrap();
        b.line_to(x1, y0).unwrap();
        b.line_to(x1, y1).unwrap();
        b.line_to(x0, y1).unwrap();
        b.close(false).unwrap();
        XPath::new(&b.build(), &identity(), 1.0, true)
    }

    #[test]
    fn rect_span_single_row() {
        let xpath = rect_xpath(2.0, 1.0, 6.0, 3.0);
        let scanner = XPathScanner::new(&xpath, false, 0, 10);
        let spans: Vec<_> = ScanIterator::new(&scanner, 2).collect();
        assert!(!spans.is_empty(), "expected at least one span on y=2");
        // The span should cover roughly [2, 5] (inclusive).
        let (x0, x1) = spans[0];
        assert!(x0 <= 2, "x0={x0}");
        assert!(x1 >= 4, "x1={x1}");
    }

    #[test]
    fn out_of_range_y_empty() {
        let xpath = rect_xpath(0.0, 0.0, 4.0, 4.0);
        let scanner = XPathScanner::new(&xpath, false, 0, 10);
        assert!(ScanIterator::new(&scanner, 100).next().is_none());
    }

    #[test]
    fn coalescing_adjacent_nz() {
        // NZ winding: [0,3] count=-1 then [4,7] count=+1.
        // count: 0 → (consume row[0]) -1 → inside; span starts at x0=0.
        // phase2: x1=3, count=-1. Consume row[1]: x1=max(3,7)=7, count=0. Outside.
        // Emit (0, 7). One span total.
        use crate::scanner::Intersect;
        let row = vec![
            Intersect {
                x0: 0,
                x1: 3,
                count: -1,
            },
            Intersect {
                x0: 4,
                x1: 7,
                count: 1,
            },
        ];
        let scanner_stub = XPathScanner {
            eo: false,
            x_min: 0,
            x_max: 7,
            y_min: 0,
            y_max: 0,
            row_start: vec![0, 2],
            intersects: row,
        };
        let spans: Vec<_> = ScanIterator::new(&scanner_stub, 0).collect();
        assert_eq!(spans, vec![(0, 7)]);
    }

    /// An intersection of winding count 0 met outside the shape covers its
    /// own cells, as the antialiased scanline walk covers them.
    #[test]
    fn count_zero_intersection_outside_is_covered() {
        use crate::scanner::Intersect;
        let row = vec![
            Intersect {
                x0: 0,
                x1: 2,
                count: 0,
            },
            Intersect {
                x0: 4,
                x1: 5,
                count: 1,
            },
            Intersect {
                x0: 8,
                x1: 9,
                count: -1,
            },
        ];
        let scanner_stub = XPathScanner {
            eo: false,
            x_min: 0,
            x_max: 9,
            y_min: 0,
            y_max: 0,
            row_start: vec![0, 3],
            intersects: row,
        };
        let spans: Vec<_> = ScanIterator::new(&scanner_stub, 0).collect();
        assert_eq!(spans, vec![(0, 2), (4, 9)]);
    }

    #[test]
    fn two_disjoint_spans_nz() {
        // NZ winding: two separate filled regions.
        // [0,2,-1] → inside; [3,5,+1] → count=0, outside → emit (0,5).
        // Wait, -1+1=0 so this is actually one span [0,5] that ends at row[1].
        // Use a different configuration for genuinely two spans:
        // [0,2,-1], [3,5,+1], [8,10,-1], [11,13,+1]
        // Call 1: consume [0,2,-1] → inside, x0=0. Phase2: x1=2, count=-1.
        //   Consume [3,5,+1]: x1=max(2,5)=5, count=0. Outside. Emit (0,5).
        // Call 2: count=0. Consume [8,10,-1] → inside, x0=8. Phase2: x1=10, count=-1.
        //   Consume [11,13,+1]: x1=max(10,13)=13, count=0. Outside. Emit (8,13).
        use crate::scanner::Intersect;
        let row = vec![
            Intersect {
                x0: 0,
                x1: 2,
                count: -1,
            },
            Intersect {
                x0: 3,
                x1: 5,
                count: 1,
            },
            Intersect {
                x0: 8,
                x1: 10,
                count: -1,
            },
            Intersect {
                x0: 11,
                x1: 13,
                count: 1,
            },
        ];
        let scanner_stub = XPathScanner {
            eo: false,
            x_min: 0,
            x_max: 13,
            y_min: 0,
            y_max: 0,
            row_start: vec![0, 4],
            intersects: row,
        };
        let spans: Vec<_> = ScanIterator::new(&scanner_stub, 0).collect();
        assert_eq!(spans, vec![(0, 5), (8, 13)]);
    }

    #[test]
    fn eo_winding_two_spans() {
        // Even-odd: crossings toggle inside/outside at each entry.
        // All entries have count=1 (EO only looks at parity).
        // [0,2,1]: count 0→1 (odd=inside), x0=0.
        // [3,5,1]: count 1→2 (even=outside), x1=5. Emit (0,5). (spans don't
        //          cleanly separate here since x1 extends through the exit entry)
        //   Actually x1=max(2,5)=5 when consuming [3,5].
        // [8,10,1]: count 2→3 (odd=inside), x0=8.
        // [11,13,1]: count 3→4 (even=outside), x1=max(10,13)=13. Emit (8,13).
        use crate::scanner::Intersect;
        let row = vec![
            Intersect {
                x0: 0,
                x1: 2,
                count: 1,
            },
            Intersect {
                x0: 3,
                x1: 5,
                count: 1,
            },
            Intersect {
                x0: 8,
                x1: 10,
                count: 1,
            },
            Intersect {
                x0: 11,
                x1: 13,
                count: 1,
            },
        ];
        let scanner_stub = XPathScanner {
            eo: true,
            x_min: 0,
            x_max: 13,
            y_min: 0,
            y_max: 0,
            row_start: vec![0, 4],
            intersects: row,
        };
        let spans: Vec<_> = ScanIterator::new(&scanner_stub, 0).collect();
        assert_eq!(spans, vec![(0, 5), (8, 13)]);
    }

    #[test]
    fn already_inside_residue() {
        // Simulate the "already inside" case: count=-1 at the start of next().
        // row has one entry [{5,7,+1}]. count should go to 0 after consuming it.
        // We test this by directly manipulating iterator state.
        use crate::scanner::Intersect;
        let row = vec![Intersect {
            x0: 5,
            x1: 7,
            count: 1,
        }];
        let scanner_stub = XPathScanner {
            eo: false,
            x_min: 5,
            x_max: 7,
            y_min: 0,
            y_max: 0,
            row_start: vec![0, 1],
            intersects: row,
        };
        let mut it = ScanIterator::new(&scanner_stub, 0);
        // Manually set count to -1 to simulate residue from a prior span.
        it.count = -1;
        // With count=-1 (inside), phase 1 immediately uses row[0] as the span
        // start without consuming it; phase 2 consumes it: x1=7, count=0.
        assert_eq!(it.next(), Some((5, 7)));
        assert_eq!(it.next(), None);
    }
}
