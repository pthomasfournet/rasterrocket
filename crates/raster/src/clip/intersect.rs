//! Intersection of a [`Clip`] with a path, and the folding of its path
//! list.
//!
//! A path that is an axis-aligned rectangle narrows the clip rectangle;
//! any other path is pushed onto the list as one region over the clip
//! rectangle, and a list that reaches [`FOLD_DEPTH`] regions is replaced by
//! their conjunction, one region per kind.

use std::sync::Arc;

use super::{Clip, ClipNode, FOLD_DEPTH, Region};
use crate::coverage::{AreaRaster, CoverageTable};
use crate::scanner::XPathScanner;
use crate::types::floor_i32;
use crate::xpath::XPath;

impl Clip {
    /// Intersect with an arbitrary path clip.
    ///
    /// If the path resolves to a simple axis-aligned rectangle (4 segments,
    /// axis-aligned), it is reduced to `clip_to_rect`. Otherwise its region
    /// over the clip rectangle, a coverage table when the clip is
    /// antialiased and a scanner otherwise, is pushed onto the list, and a
    /// list that reaches [`FOLD_DEPTH`] regions is folded.
    ///
    /// An empty path forces the clip to be empty (nothing passes through).
    pub fn clip_to_path(&mut self, xpath: &XPath, eo: bool) {
        self.clip_to_path_as(xpath, eo, self.antialias);
    }

    /// [`clip_to_path`](Self::clip_to_path) with an aliased region whatever
    /// the clip's antialiasing: a pixel passes when the path's closed
    /// outline meets it, so clips of abutting paths leave no pixel of their
    /// shared edge partly covered by both.
    pub fn clip_to_path_aliased(&mut self, xpath: &XPath, eo: bool) {
        self.clip_to_path_as(xpath, eo, false);
    }

    fn clip_to_path_as(&mut self, xpath: &XPath, eo: bool, antialias: bool) {
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
        let region = if antialias {
            Region::Area(self.coverage_of(xpath, eo))
        } else {
            match XPathScanner::new(xpath, eo, self.y_min_i, self.y_max_i) {
                Ok(scanner) => Region::Aliased(scanner),
                Err(e) => {
                    log::warn!("clip path dropped, and the clip made empty: {e}");
                    self.x_max = self.x_min - 1.0;
                    self.y_max = self.y_min - 1.0;
                    self.recompute_int_bounds();
                    return;
                }
            }
        };
        self.scanners = Some(Arc::new(ClipNode {
            region,
            parent: self.scanners.take(),
        }));
        if self.regions().nth(FOLD_DEPTH - 1).is_some() {
            self.fold();
        }
    }

    /// The exact-area coverage of `xpath` over the clip rectangle's pixels.
    fn coverage_of(&self, xpath: &XPath, eo: bool) -> CoverageTable {
        let Some((lx, hx, ly, hy)) = xpath.bounds() else {
            return CoverageTable::empty();
        };
        let mut raster = AreaRaster::new(
            self.x_min_i.max(floor_i32(lx)),
            self.y_min_i.max(floor_i32(ly)),
            self.x_max_i.min(floor_i32(hx)),
            self.y_max_i.min(floor_i32(hy)),
        );
        raster.add_xpath(xpath);
        CoverageTable::from_raster(&mut raster, eo)
    }

    /// Replace the path-clip list with one node per region kind holding
    /// the conjunction of that kind's regions.
    fn fold(&mut self) {
        let aliased: Vec<&XPathScanner> = self
            .regions()
            .filter_map(|r| match r {
                Region::Aliased(s) => Some(s),
                Region::Area(_) => None,
            })
            .collect();
        let tables: Vec<&CoverageTable> = self
            .regions()
            .filter_map(|r| match r {
                Region::Area(t) => Some(t),
                Region::Aliased(_) => None,
            })
            .collect();
        let mut folded = None;
        if !aliased.is_empty() {
            folded = Some(Arc::new(ClipNode {
                region: Region::Aliased(XPathScanner::intersection(aliased)),
                parent: folded,
            }));
        }
        if !tables.is_empty() {
            folded = Some(Arc::new(ClipNode {
                region: Region::Area(CoverageTable::product(tables)),
                parent: folded,
            }));
        }
        self.scanners = folded;
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
