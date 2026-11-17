//! Clip region: an axis-aligned rectangle intersected with zero or more
//! arbitrary path clip regions.
//!
//! ## Sharing semantics
//!
//! The path clips form a persistent singly linked list, newest first: each
//! node owns one region, an [`XPathScanner`] for an aliased clip or a
//! [`CoverageTable`] for an antialiased one, and an [`Arc`] to the list it
//! was intersected onto.  [`Clip::clone_shared`] (a graphics-state save) copies the
//! head pointer, so a save costs O(1) whatever the number of path clips, and
//! an intersection allocates one node shared by no other clip.  A list that
//! reaches `FOLD_DEPTH` nodes is replaced by one node holding their
//! intersection, so a list never holds more than `FOLD_DEPTH - 1` nodes.
//! The regions are immutable after construction, so sharing
//! across threads and across `clone_shared` copies is safe: there is no
//! interior mutability in the shared objects.
//!
//! The submodules hold the intersection with a path and the folding of the
//! list (`intersect`), the pixel, span and rectangle tests and the coverage
//! arithmetic over the regions (`query`), and the row-by-row coverage of a
//! span ([`ClipRows`]).

use std::sync::Arc;

use crate::coverage::CoverageTable;
use crate::scanner::XPathScanner;
use crate::types::{ceil_i32, floor_i32};

mod intersect;
mod query;
mod rows;

pub use rows::{ClipRows, RowClip};

/// Path clips a [`Clip`] holds at most: intersecting the one that reaches
/// this many replaces the list with one region per kind, the intersection
/// of the aliased scanners ([`XPathScanner::intersection`]) and the product
/// of the coverage tables ([`CoverageTable::product`]), which every query
/// answers as the list did, so the per-span cost of a clip stays bounded
/// however many paths it intersects.
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
    /// Whether path clips are antialiased: their regions are then
    /// exact-area coverage tables, otherwise aliased scanners.
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

/// The identity of a path-clip region of a [`Clip`].
///
/// Equal for the same region, distinct for distinct regions alive at once.
/// It is valid while a clip holding the region lives; once none does, a new
/// region may take it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RegionKey(usize);

/// One path clip of a [`Clip`] and the clips it was intersected with.
struct ClipNode {
    region: Region,
    parent: Option<Arc<Self>>,
}

/// The pixels a path clip passes.
enum Region {
    /// An aliased clip: the pixels its scanner covers, each fully.
    Aliased(XPathScanner),
    /// An antialiased clip: each pixel's exact-area coverage.
    Area(CoverageTable),
}

impl Region {
    fn translated(&self, dx: i32, dy: i32) -> Self {
        match self {
            Self::Aliased(s) => Self::Aliased(s.translated(dx, dy)),
            Self::Area(t) => Self::Area(t.translated(dx, dy)),
        }
    }
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
    #[must_use]
    pub fn translated(&self, dx: i32, dy: i32) -> Self {
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
        let shifted: Vec<Region> = self.regions().map(|r| r.translated(dx, dy)).collect();
        clip.scanners = shifted.into_iter().rev().fold(None, |parent, region| {
            Some(Arc::new(ClipNode { region, parent }))
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

    /// The path-clip regions, newest first, each with its identity: the
    /// order in which [`Self::mul_row`] scales a coverage by them.  An
    /// antialiased region is its coverage table, an aliased one `None`.
    pub fn path_regions(&self) -> impl Iterator<Item = (RegionKey, Option<&CoverageTable>)> {
        std::iter::successors(self.scanners.as_deref(), |node| node.parent.as_deref()).map(|node| {
            let table = match &node.region {
                Region::Area(t) => Some(t),
                Region::Aliased(_) => None,
            };
            (RegionKey(std::ptr::from_ref(node) as usize), table)
        })
    }

    /// The identity of the newest path-clip region, `None` for a
    /// rectangle-only clip.  Clips sharing it share every path region.
    #[must_use]
    pub fn path_key(&self) -> Option<RegionKey> {
        self.scanners
            .as_deref()
            .map(|node| RegionKey(std::ptr::from_ref(node) as usize))
    }

    /// The path-clip regions, newest first.  Every consumer intersects them,
    /// so the order does not affect any result.
    fn regions(&self) -> impl Iterator<Item = &Region> {
        std::iter::successors(self.scanners.as_deref(), |node| node.parent.as_deref())
            .map(|node| &node.region)
    }
}

#[cfg(test)]
mod tests;
