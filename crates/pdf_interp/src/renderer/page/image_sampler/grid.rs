//! The per-image sampling grid: a draw's device-space bounding box and the
//! map from each of its pixels to a source coordinate.
//!
//! The map is the pair of separable Q32 tables the module root describes,
//! or, for a draw whose table terms leave the Q32 range, the inverse matrix
//! evaluated per pixel.  Either way [`SampleGrid::sample`] yields the taps of
//! the samples under each pixel centre, or `None` off the image.

use super::ENTRY_WORDS;
use super::filter::Filter;
use super::inverse::{InverseMap, fits_q32, push_entry, split_bounded};
use super::taps::{AxisTaps, SeparableTaps, Taps, axis_taps, on_axis};
use crate::renderer::gstate::{Ctm, ctm_transform};

/// The bitmap an image is sampled onto: a `width × height` window whose
/// pixel `(0, 0)` is pixel `origin` of a page `page_height` pixels tall,
/// whose device space the draw matrices map to.  Source coordinates are
/// computed for the page's pixels and the window takes them, so a window
/// samples the pixels the whole page samples there.
#[derive(Clone, Copy, Debug)]
pub(in crate::renderer::page) struct GridView {
    pub origin: (u32, u32),
    pub width: u32,
    pub height: u32,
    pub page_height: u32,
}

#[cfg(test)]
impl GridView {
    /// A whole `width × height` page.
    pub(super) const fn whole(width: u32, height: u32) -> Self {
        Self {
            origin: (0, 0),
            width,
            height,
            page_height: height,
        }
    }
}

/// Per-image sampling grid: the device-space bounding box plus the column
/// and row tables that map every pixel inside it to a source coordinate.
pub(in crate::renderer::page) struct SampleGrid {
    /// Inclusive left edge of the bounding box, page pixels.
    pub x0: u32,
    /// Inclusive top edge.
    pub y0: u32,
    /// Exclusive right edge.
    pub x1: u32,
    /// Exclusive bottom edge.
    pub y1: u32,
    /// Source image width in pixels.
    img_w: u32,
    /// Source image height in pixels.
    img_h: u32,
    /// Axes interpolated between samples.
    filter: Filter,
    /// How a device pixel maps to its source coordinate.
    map: GridMap,
}

/// The inverse image map of a [`SampleGrid`].
enum GridMap {
    /// Separable Q32 tables whose every term lies within
    /// ±[`COORD_LIMIT`](super::COORD_LIMIT).
    Tables {
        /// `x1 - x0` entries of [`ENTRY_WORDS`](super::ENTRY_WORDS) words each.
        cols: Vec<u32>,
        /// `y1 - y0` entries of [`ENTRY_WORDS`](super::ENTRY_WORDS) words each.
        rows: Vec<u32>,
    },
    /// Per-pixel `f64` evaluation of the inverse matrix, for a draw whose
    /// table terms would exceed ±[`COORD_LIMIT`](super::COORD_LIMIT).
    Direct(InverseMap),
}

impl SampleGrid {
    /// Build the grid for an image of `img_w × img_h` pixels drawn through
    /// `ctm`, a matrix into the page's device space (y up), onto `view`,
    /// whose y axis points down, interpolating the axes `filter` selects.
    ///
    /// Returns `None` when nothing can be drawn: a non-finite or singular
    /// CTM, or a bounding box that misses the page entirely.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "bbox edges are clamped to [0, page dim] in i64 before the u32 cast"
    )]
    pub(in crate::renderer::page) fn new(
        ctm: &Ctm,
        view: GridView,
        img_w: u32,
        img_h: u32,
        filter: Filter,
    ) -> Option<Self> {
        if img_w == 0 || img_h == 0 {
            return None;
        }
        let page_h_f = f64::from(view.page_height);

        // Bounding box of the unit square's image, y-flipped into bitmap
        // space and clamped to the page.  `floor`/`ceil` of the corner
        // extremes covers every pixel whose area intersects the image, so
        // the centre-sampling test inside the loop never misses a pixel.
        let corners = [
            ctm_transform(ctm, 0.0, 0.0),
            ctm_transform(ctm, 1.0, 0.0),
            ctm_transform(ctm, 0.0, 1.0),
            ctm_transform(ctm, 1.0, 1.0),
        ];
        if corners
            .iter()
            .any(|(x, y)| !x.is_finite() || !y.is_finite())
        {
            return None;
        }
        let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
        for &(x, y) in &corners {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            let y_dev = page_h_f - y;
            min_y = min_y.min(y_dev);
            max_y = max_y.max(y_dev);
        }
        // The bounding box on the page, moved into the window and clamped
        // to it.
        let clamp = |v: f64, at: u32, dim: u32| {
            ((v as i64).saturating_sub(i64::from(at))).clamp(0, i64::from(dim)) as u32
        };
        let x0 = clamp(min_x.floor(), view.origin.0, view.width);
        let x1 = clamp(max_x.ceil(), view.origin.0, view.width);
        let y0 = clamp(min_y.floor(), view.origin.1, view.height);
        let y1 = clamp(max_y.ceil(), view.origin.1, view.height);
        if x0 >= x1 || y0 >= y1 {
            return None;
        }

        // Every corner is finite, so every CTM component is.
        let inverse = InverseMap::new(ctm, view, img_w, img_h)?;

        // Each term is affine in `dx_rel` or `dy_rel`, so its extremes over
        // the bounding box lie at the box's first and last pixel centres.
        let representable = [
            inverse.col_term(x0),
            inverse.col_term(x1 - 1),
            inverse.row_term(y0),
            inverse.row_term(y1 - 1),
        ]
        .iter()
        .all(|&(x, y)| fits_q32(x) && fits_q32(y));

        let map = if representable {
            let mut cols = Vec::with_capacity((x1 - x0) as usize * ENTRY_WORDS);
            for dx in x0..x1 {
                let (x, y) = inverse.col_term(dx);
                push_entry(&mut cols, x, y);
            }
            let mut rows = Vec::with_capacity((y1 - y0) as usize * ENTRY_WORDS);
            for dy in y0..y1 {
                let (x, y) = inverse.row_term(dy);
                push_entry(&mut rows, x, y);
            }
            GridMap::Tables { cols, rows }
        } else {
            GridMap::Direct(inverse)
        };

        Some(Self {
            x0,
            y0,
            x1,
            y1,
            img_w,
            img_h,
            filter,
            map,
        })
    }

    /// Axes interpolated between samples.
    pub(in crate::renderer::page) const fn filter(&self) -> Filter {
        self.filter
    }

    /// The width and height in samples of the image the grid samples.
    #[cfg(device_pages)]
    pub(in crate::renderer::page) const fn source_size(&self) -> (u32, u32) {
        (self.img_w, self.img_h)
    }

    /// The column table, `(x1 - x0) * ENTRY_WORDS` words, and the row
    /// table, `(y1 - y0) * ENTRY_WORDS` words; `None` when the draw's terms
    /// exceed the Q32 table range and the grid evaluates the inverse matrix
    /// per pixel instead.
    #[cfg(any(
        feature = "cache",
        feature = "vulkan",
        all(windows, feature = "d3d12"),
        test
    ))]
    pub(in crate::renderer::page) fn tables(&self) -> Option<(&[u32], &[u32])> {
        match &self.map {
            GridMap::Tables { cols, rows } => Some((cols, rows)),
            GridMap::Direct(_) => None,
        }
    }

    /// Source taps for device pixel `(dx, dy)`, or `None` when the pixel
    /// centre falls outside the image.
    ///
    /// `dx`/`dy` must lie inside the bounding box.
    #[inline]
    pub(in crate::renderer::page) fn sample(&self, dx: u32, dy: u32) -> Option<Taps> {
        let (x, y) = self.coord(dx, dy);
        (on_axis(x.0, self.img_w) && on_axis(y.0, self.img_h)).then(|| self.taps(x, y))
    }

    /// Per-column taps when the grid is separable: every row entry holds
    /// the same x words and every column entry the same y words, so the
    /// source x of a pixel is a function of its column and its source y of
    /// its row: an axis-aligned draw without a quarter turn.  `None`
    /// otherwise, and for a draw mapped per pixel (its terms past the
    /// tables' range).
    pub(in crate::renderer::page) fn separable(&self) -> Option<SeparableTaps> {
        let uniform = |table: &[u32], word: usize| {
            let first = &table[word..word + 2];
            table
                .as_chunks::<ENTRY_WORDS>()
                .0
                .iter()
                .all(|e| &e[word..word + 2] == first)
        };
        // A draw mapped per pixel is never taken as separable.
        let GridMap::Tables { cols, rows } = &self.map else {
            return None;
        };
        if !uniform(rows, 0) || !uniform(cols, 2) {
            return None;
        }
        let cols = (self.x0..self.x1)
            .map(|dx| {
                let (x, _) = self.coord(dx, self.y0);
                on_axis(x.0, self.img_w).then(|| axis_taps(x.0, x.1, self.img_w, self.filter.x))
            })
            .collect();
        Some(SeparableTaps { cols })
    }

    /// The y taps of bounding-box row `dy` of a separable grid (see
    /// [`Self::separable`]), or `None` when its source y lies off the image.
    pub(in crate::renderer::page) fn row_taps(&self, dy: u32) -> Option<AxisTaps> {
        let (_, y) = self.coord(self.x0, dy);
        on_axis(y.0, self.img_h).then(|| axis_taps(y.0, y.1, self.img_h, self.filter.y))
    }

    /// Source taps for device pixel `(dx, dy)`, clamped onto the image.
    ///
    /// A device pixel outside the bounding box is first clamped onto it.
    #[inline]
    pub(in crate::renderer::page) fn sample_clamped(&self, dx: u32, dy: u32) -> Taps {
        let (x, y) = self.coord(
            dx.clamp(self.x0, self.x1 - 1),
            dy.clamp(self.y0, self.y1 - 1),
        );
        self.taps(x, y)
    }

    /// Taps at source coordinate `(x, y)`, each an integer part and a
    /// fraction, clamped onto the image.
    #[inline]
    fn taps(&self, x: (i32, u32), y: (i32, u32)) -> Taps {
        Taps::from_axes(
            axis_taps(x.0, x.1, self.img_w, self.filter.x),
            axis_taps(y.0, y.1, self.img_h, self.filter.y),
        )
    }

    /// Unbounded source coordinate of device pixel `(dx, dy)`, as an
    /// integer part and a Q32 fraction per axis: the column entry plus the
    /// row entry, with the fraction carry.
    #[inline]
    #[expect(
        clippy::cast_possible_wrap,
        reason = "hi words carry an i32 bit pattern by construction"
    )]
    fn coord(&self, dx: u32, dy: u32) -> ((i32, u32), (i32, u32)) {
        let (cols, rows) = match &self.map {
            GridMap::Tables { cols, rows } => (cols, rows),
            GridMap::Direct(inverse) => {
                let (x, y) = inverse.coord(dx, dy);
                return (split_bounded(x, self.img_w), split_bounded(y, self.img_h));
            }
        };
        let c = (dx - self.x0) as usize * ENTRY_WORDS;
        let r = (dy - self.y0) as usize * ENTRY_WORDS;
        let col = &cols[c..c + ENTRY_WORDS];
        let row = &rows[r..r + ENTRY_WORDS];
        let (frac_x, carry_x) = col[1].overflowing_add(row[1]);
        let x = (col[0] as i32)
            .wrapping_add(row[0] as i32)
            .wrapping_add(i32::from(carry_x));
        let (frac_y, carry_y) = col[3].overflowing_add(row[3]);
        let y = (col[2] as i32)
            .wrapping_add(row[2] as i32)
            .wrapping_add(i32::from(carry_y));
        ((x, frac_x), (y, frac_y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference sampler: exact f64 inverse map at the pixel centre.
    #[expect(
        clippy::many_single_char_names,
        clippy::similar_names,
        clippy::suboptimal_flops,
        reason = "test oracle: spelled as the textbook inverse-CTM formula, not for speed"
    )]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the range check proves x and y lie in [0, img dim) before the u32 cast"
    )]
    fn reference(
        ctm: &Ctm,
        page_h: u32,
        img_w: u32,
        img_h: u32,
        dx: u32,
        dy: u32,
    ) -> Option<(u32, u32)> {
        let [a, b, c, d, e, f] = *ctm;
        let det = a * d - b * c;
        let dx_rel = f64::from(dx) + 0.5 - e;
        let dy_rel = f64::from(page_h) - (f64::from(dy) + 0.5) - f;
        let u = (d * dx_rel - c * dy_rel) / det;
        let v = (-b * dx_rel + a * dy_rel) / det;
        let x = u * f64::from(img_w);
        let y = (1.0 - v) * f64::from(img_h);
        if x < 0.0 || y < 0.0 || x >= f64::from(img_w) || y >= f64::from(img_h) {
            return None;
        }
        Some((x as u32, y as u32))
    }

    fn assert_matches_reference(ctm: &Ctm, page: (u32, u32), img: (u32, u32)) {
        let grid = SampleGrid::new(
            ctm,
            GridView::whole(page.0, page.1),
            img.0,
            img.1,
            Filter::NEAREST,
        )
        .expect("grid");
        let mut inside = 0;
        for dy in grid.y0..grid.y1 {
            for dx in grid.x0..grid.x1 {
                let got = grid.sample(dx, dy);
                let want =
                    reference(ctm, page.1, img.0, img.1, dx, dy).map(|(x, y)| Taps::point(x, y));
                assert_eq!(got, want, "pixel ({dx}, {dy})");
                inside += usize::from(got.is_some());
            }
        }
        assert!(inside > 0, "degenerate test geometry");
    }

    #[test]
    fn axis_aligned_fractional_placement_matches_reference() {
        // Fractional translation: the first bbox column starts before the
        // image edge, which the exact map must not snap to.
        let ctm = [1092.43, 0.0, 0.0, -1650.0, 91.285, 1650.0];
        assert_matches_reference(&ctm, (1275, 1650), (1844, 2828));
    }

    #[test]
    fn rotated_placement_matches_reference_and_rejects_outside_unit_square() {
        let ctm = [120.5, 30.25, -20.75, 150.5, 50.33, 40.17];
        let grid = SampleGrid::new(&ctm, GridView::whole(300, 300), 50, 80, Filter::NEAREST)
            .expect("grid");
        let outside = (grid.y0..grid.y1)
            .flat_map(|dy| (grid.x0..grid.x1).map(move |dx| (dx, dy)))
            .filter(|&(dx, dy)| grid.sample(dx, dy).is_none())
            .count();
        assert!(
            outside > 0,
            "a rotated image must leave bbox corners untouched"
        );
        assert_matches_reference(&ctm, (300, 300), (50, 80));
    }

    #[test]
    fn flipped_and_upscaled_placement_matches_reference() {
        // Negative `a` mirrors horizontally; a 3×3 image over 200 px
        // exercises repeated source columns.
        let ctm = [-200.0, 0.0, 0.0, -100.0, 250.0, 150.0];
        assert_matches_reference(&ctm, (300, 200), (3, 3));
    }

    #[test]
    fn exact_integer_scale_is_a_pure_copy() {
        // 1:1 placement at an integer offset: the unit square lands on
        // user-space x 10..50, y 20..50, i.e. device rows 50..80 on a
        // 100 px page, and device pixel (10+k, 50+j) reads source (k, j).
        let ctm = [40.0, 0.0, 0.0, 30.0, 10.0, 20.0];
        let grid = SampleGrid::new(&ctm, GridView::whole(100, 100), 40, 30, Filter::NEAREST)
            .expect("grid");
        assert_eq!((grid.x0, grid.y0, grid.x1, grid.y1), (10, 50, 50, 80));
        for dy in 50..80 {
            for dx in 10..50 {
                assert_eq!(grid.sample(dx, dy), Some(Taps::point(dx - 10, dy - 50)));
            }
        }
    }

    #[test]
    fn partially_off_page_bbox_is_clamped() {
        let ctm = [100.0, 0.0, 0.0, -100.0, -30.0, 120.0];
        let grid =
            SampleGrid::new(&ctm, GridView::whole(50, 50), 10, 10, Filter::NEAREST).expect("grid");
        assert_eq!((grid.x0, grid.x1), (0, 50));
        assert_eq!((grid.y0, grid.y1), (0, 30));
        let (cols, rows) = grid.tables().expect("tables");
        assert_eq!(cols.len(), 50 * ENTRY_WORDS);
        assert_eq!(rows.len(), 30 * ENTRY_WORDS);
    }

    #[test]
    fn degenerate_inputs_yield_no_grid() {
        let singular = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        assert!(
            SampleGrid::new(&singular, GridView::whole(10, 10), 4, 4, Filter::NEAREST).is_none()
        );
        let non_finite = [f64::NAN, 0.0, 0.0, 1.0, 0.0, 0.0];
        assert!(
            SampleGrid::new(&non_finite, GridView::whole(10, 10), 4, 4, Filter::NEAREST).is_none()
        );
        let off_page = [10.0, 0.0, 0.0, -10.0, 500.0, 500.0];
        assert!(
            SampleGrid::new(&off_page, GridView::whole(10, 10), 4, 4, Filter::NEAREST).is_none()
        );
        let empty_image = [10.0, 0.0, 0.0, -10.0, 0.0, 10.0];
        assert!(
            SampleGrid::new(&empty_image, GridView::whole(10, 10), 0, 4, Filter::NEAREST).is_none()
        );
    }

    /// A one-row image of 2^20 samples drawn as a half-pixel-thick sliver
    /// along a page diagonal: across the 1000-pixel bounding box each column
    /// and row term spans about 1.5e9 samples, past the Q32 table range, while
    /// their sum stays on the image along the sliver.  Every pixel must still
    /// match the exact inverse map — in particular the far corners, where a
    /// positive and a negative term both exceed the range.
    #[test]
    fn sliver_whose_table_terms_exceed_q32_matches_reference() {
        let r = std::f64::consts::FRAC_1_SQRT_2;
        let (thickness, length) = (0.5, 1350.0);
        let ctm = [
            thickness * r,
            -thickness * r,
            length * r,
            length * r,
            20.0,
            20.0,
        ];
        let img = (1 << 20, 1);
        let grid = SampleGrid::new(
            &ctm,
            GridView::whole(1000, 1000),
            img.0,
            img.1,
            Filter::NEAREST,
        )
        .expect("grid");
        assert!(grid.tables().is_none(), "terms must exceed the table range");
        assert_matches_reference(&ctm, (1000, 1000), img);
    }

    /// A draw whose terms fit the table range keeps its tables.
    #[test]
    fn ordinary_draw_builds_tables() {
        let ctm = [120.5, 30.25, -20.75, 150.5, 50.33, 40.17];
        let grid = SampleGrid::new(&ctm, GridView::whole(300, 300), 50, 80, Filter::NEAREST)
            .expect("grid");
        assert!(grid.tables().is_some());
    }

    #[test]
    fn rotated_draw_is_not_separable() {
        let grid = SampleGrid::new(
            &[60.0, 25.0, -25.0, 60.0, 50.0, 40.0],
            GridView::whole(200, 160),
            40,
            40,
            Filter { x: true, y: true },
        )
        .unwrap();
        assert!(grid.separable().is_none());
    }
}
