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
use raster::pipe::PipeSrc;
use raster::stroke::stroke;
use raster::types::BlendMode;

use super::PageRenderer;
use super::canvas::CanvasKind;
use super::paint::PaintParams;
use super::pixel_rect::PixelRect;
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
            .and_then(|r| self.within_clip(r));
        let Some(rect) = rect else {
            self.fill_path(path, even_odd);
            self.stroke_path(path);
            return;
        };
        let coverage = match self.stroke_coverage(path, line_width, rect) {
            Ok(coverage) => coverage,
            Err(e) => {
                self.abort_allocation(e);
                return;
            }
        };
        let area = rect.to_path();
        self.paint_knocked_out(
            rect,
            &coverage,
            |r| r.fill_path(path, even_odd),
            |r| r.fill_with_stroke_paint(&area),
        );
    }

    /// The coverage (0–255) of the stroke of `path`, `line_width` wide,
    /// over each pixel of `rect` in row-major order, within the clip.
    ///
    /// The stroke is drawn opaque white on a black RGB bitmap of `rect`'s
    /// size, the device matrix and the clip moved by `rect`'s origin, so
    /// the red channel is the coverage whatever the stroke's colour space,
    /// pattern, transfer function or soft mask.  The error names the
    /// bitmap when the allocator refuses it, or when `rect`'s origin lies
    /// off the `i32` range a clip translates by.
    pub(super) fn stroke_coverage(
        &self,
        path: &Path,
        line_width: f64,
        rect: PixelRect,
    ) -> Result<Vec<u8>, raster::BitmapAllocError> {
        let (width, height) = (rect.width(), rect.height());
        let mut bitmap = Bitmap::<Rgb8>::try_new_filled(width, height, 1, false, 0)?;
        let mut shape = self.stroke_shape(path, line_width);
        let (x0, y0) = (rect.x0, rect.y0);
        shape.matrix[4] -= f64::from(x0);
        shape.matrix[5] -= f64::from(y0);
        // A rectangle within the clip lies within the page, whose
        // coordinates fit i32.
        debug_assert!(
            i32::try_from(x0).is_ok() && i32::try_from(y0).is_ok(),
            "stroke_coverage: the rectangle's origin ({x0}, {y0}) lies off the i32 range"
        );
        let (Ok(dx), Ok(dy)) = (i32::try_from(x0), i32::try_from(y0)) else {
            log::warn!(
                "rasterrocket-interp: knockout stroke rectangle at ({x0}, {y0}) lies off the i32 range"
            );
            return Err(raster::BitmapAllocError { width, height });
        };
        let mut clip = self.gstate.current().clip.translated(-dx, -dy);
        clip.clip_to_rect(0.0, 0.0, f64::from(width), f64::from(height));

        let white = RasterColor::from_bytes([u8::MAX; 3]);
        let paint = PaintParams {
            alpha: u8::MAX,
            blend_mode: BlendMode::Normal,
            overprint: None,
            backdrop: None,
        };
        let pipe = paint.pipe();
        let src = PipeSrc::Solid(white.bytes_for(CanvasKind::Rgb.mode()));
        stroke(
            &mut bitmap,
            &clip,
            &shape.path,
            &pipe,
            &src,
            &shape.matrix,
            &shape.params(),
        );

        let mut coverage = Vec::new();
        coverage
            .try_reserve_exact(width as usize * height as usize)
            .map_err(|_| raster::BitmapAllocError { width, height })?;
        let n = width as usize * 3;
        for y in 0..height {
            coverage.extend(bitmap.row_bytes(y)[..n].iter().step_by(3));
        }
        Ok(coverage)
    }

    /// Fill `area` with the stroke's colour or pattern, alpha and
    /// overprint.
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
            gs.overprint_fill = gs.overprint_stroke;
        }
        self.fill_path(area, false);
        self.gstate.restore();
    }
}
