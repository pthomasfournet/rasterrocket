//! The graphics-state clip over a [`SampleGrid`], addressed in grid
//! coordinates.

use raster::{Clip, ClipResult, ClipRows, RowClip};

use super::grid::SampleGrid;

/// Admission of a [`SampleGrid`]'s pixels by the graphics-state clip: the
/// clip's [`ClipRows`] over the grid's rectangle, addressed in grid
/// coordinates.
pub(in crate::renderer::page) struct GridClip<'a>(ClipRows<'a>);

impl<'a> GridClip<'a> {
    #[expect(
        clippy::cast_possible_wrap,
        reason = "grid edges are clamped to the page, which fits i32"
    )]
    pub(in crate::renderer::page) fn new(clip: &'a Clip, grid: &SampleGrid) -> Self {
        Self(ClipRows::new(
            clip,
            grid.x0 as i32,
            grid.y0 as i32,
            grid.x1 as i32 - 1,
            grid.y1 as i32 - 1,
        ))
    }

    /// Classification of the whole grid bounding box.
    pub(in crate::renderer::page) const fn bbox(&self) -> ClipResult {
        self.0.bbox()
    }

    /// Coverage for row `dy` of the grid, or `None` when the clip excludes
    /// the entire row.
    #[expect(
        clippy::cast_possible_wrap,
        reason = "dy < grid.y1 ≤ page height, which fits i32"
    )]
    pub(in crate::renderer::page) fn row(&mut self, dy: u32) -> Option<GridRow<'_>> {
        self.0.row(dy as i32).map(GridRow)
    }
}

/// Clip coverage of the pixels on one grid row.
pub(in crate::renderer::page) struct GridRow<'a>(RowClip<'a>);

impl GridRow<'_> {
    /// Clip coverage (0 = outside, 255 = inside) of device column `dx`.
    #[inline]
    #[expect(
        clippy::cast_possible_wrap,
        reason = "dx < grid.x1 ≤ page width, which fits i32"
    )]
    pub(in crate::renderer::page) fn coverage(&self, dx: u32) -> u8 {
        self.0.coverage(dx as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Filter, GridView};
    use super::*;

    /// A clip rectangle covering the left half of a 1:1 grid: the bbox is
    /// partial, every row is partial, and coverage splits at the integer clip
    /// edge.
    #[test]
    fn grid_clip_rect_coverage_is_binary() {
        let ctm = [10.0, 0.0, 0.0, 10.0, 0.0, 0.0];
        let grid =
            SampleGrid::new(&ctm, GridView::whole(10, 10), 10, 10, Filter::NEAREST).expect("grid");
        let clip = Clip::new(0.0, 0.0, 5.0, 10.0, true);
        let mut gc = GridClip::new(&clip, &grid);
        assert_eq!(gc.bbox(), ClipResult::Partial);
        let row = gc.row(3).expect("row 3 intersects the clip");
        assert_eq!(row.coverage(0), 255);
        assert_eq!(row.coverage(4), 255);
        assert_eq!(row.coverage(5), 0);
        assert_eq!(row.coverage(9), 0);
    }

    /// A clip rectangle above the grid: the bbox is outside and no row is
    /// admitted; one enclosing the grid: every row admits every column.
    #[test]
    fn grid_clip_bbox_extremes() {
        let ctm = [10.0, 0.0, 0.0, 10.0, 0.0, 0.0];
        let grid =
            SampleGrid::new(&ctm, GridView::whole(10, 20), 10, 10, Filter::NEAREST).expect("grid");
        assert_eq!((grid.y0, grid.y1), (10, 20));

        let above = Clip::new(0.0, 0.0, 10.0, 10.0, true);
        let mut gc = GridClip::new(&above, &grid);
        assert_eq!(gc.bbox(), ClipResult::AllOutside);
        assert!(gc.row(10).is_none());

        let enclosing = Clip::new(0.0, 0.0, 10.0, 20.0, true);
        let mut gc = GridClip::new(&enclosing, &grid);
        assert_eq!(gc.bbox(), ClipResult::AllInside);
        let row = gc.row(15).expect("row inside the clip");
        assert!((0..10).all(|dx| row.coverage(dx) == 255));
    }
}
