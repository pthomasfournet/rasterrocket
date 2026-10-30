//! The fill-and-stroke operators `B`, `B*`, `b` and `b*` as one object.
//!
//! PDF §11.7.4.4: the fill and the stroke of one path operator form a
//! single object whose stroke knocks out the fill — where the stroke covers
//! a pixel it composites with the backdrop as it was before the fill, not
//! over the fill.  With stroke coverage `s` of a pixel, `F` the pixel with
//! the fill painted and `K` the backdrop with the stroke's paint applied at
//! full coverage, the result is `F + (K − F) × s`.  Only a stroke with
//! alpha below 1 takes this composition; an opaque stroke is painted over
//! the fill, which under the `Normal` blend mode gives the same pixels
//! wherever the stroke covers fully.

use color::Rgb8;
use raster::Bitmap;
use raster::path::Path;
use raster::types::BlendMode;

use super::PageRenderer;
use super::pixel_rect::PixelRect;
use super::shading_paint;
use crate::renderer::color::RasterColor;

impl PageRenderer<'_> {
    /// Fill `path` (by the even-odd rule when `even_odd`) and stroke it as
    /// one object.
    pub(super) fn fill_and_stroke(&mut self, path: &Path, even_odd: bool) {
        let knockout = self.gstate.current().stroke_alpha < u8::MAX;
        let line_width = self.min_stroke_width();
        let rect = self
            .stroke_bounds(path, line_width)
            .filter(|_| knockout)
            .and_then(|r| {
                r.intersect(PixelRect::of_clip(
                    &self.gstate.current().clip,
                    self.width,
                    self.height,
                )?)
            });
        let Some(rect) = rect else {
            self.fill_path(path, even_odd);
            self.stroke_path(path);
            return;
        };
        let coverage = self.stroke_coverage(path, line_width, rect);
        let (x0, y0) = (f64::from(rect.x0), f64::from(rect.y0));
        let (x1, y1) = (f64::from(rect.x1), f64::from(rect.y1));
        let area = shading_paint::polygon(&[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]);
        self.paint_knocked_out(
            rect,
            &coverage,
            |r| r.fill_path(path, even_odd),
            |r| r.fill_with_stroke_paint(&area),
        );
    }

    /// The coverage (0–255) of the stroke of `path`, `line_width` wide,
    /// over each pixel of `rect` in row-major order, within the clip.
    fn stroke_coverage(&mut self, path: &Path, line_width: f64, rect: PixelRect) -> Vec<u8> {
        self.gstate.save();
        {
            let gs = self.gstate.current_mut();
            gs.soft_mask = None;
            gs.stroke_alpha = u8::MAX;
            gs.blend_mode = BlendMode::Normal;
            gs.stroke_pattern = None;
            gs.stroke_color = RasterColor::from_bytes([u8::MAX; 3]);
        }
        let mut canvas = Bitmap::<Rgb8>::new(self.width, self.height, 1, false);
        std::mem::swap(&mut self.bitmap, &mut canvas);
        self.stroke_within(path, line_width, Some(rect));
        std::mem::swap(&mut self.bitmap, &mut canvas);
        self.gstate.restore();

        let (cx0, cx1) = (rect.x0 as usize * 3, rect.x1 as usize * 3);
        (rect.y0..rect.y1)
            .flat_map(|y| canvas.row_bytes(y)[cx0..cx1].iter().step_by(3).copied())
            .collect()
    }

    /// Fill `area` with the stroke's colour or pattern and alpha.
    fn fill_with_stroke_paint(&mut self, area: &Path) {
        self.gstate.save();
        {
            let gs = self.gstate.current_mut();
            gs.fill_color = gs.stroke_color.clone();
            gs.fill_pattern.clone_from(&gs.stroke_pattern);
            gs.fill_pattern_components
                .clone_from(&gs.stroke_pattern_components);
            gs.fill_color_space = gs.stroke_color_space.clone();
            gs.fill_alpha = gs.stroke_alpha;
        }
        self.fill_path(area, false);
        self.gstate.restore();
    }
}
