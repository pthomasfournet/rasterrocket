//! The glyphs of a shown string in device space: collected as cached
//! bitmaps and outlines while the string is laid out, then filled, stroked
//! and added to the text object's clip as its rendering mode requires.

use super::view::page_to_window;
use std::sync::Arc;

use super::super::color::RasterColor;
use super::PageRenderer;
use super::canvas::with_canvas;
use super::canvas::{CanvasPipe, CanvasSource};
#[cfg(any(feature = "vulkan", all(windows, feature = "d3d12")))]
use super::device_page::PlacedGlyph;
use super::paint::PaintParams;
use super::pixel_rect::PixelRect;
use super::text_ops::{GlyphRecord, glyph_placement};
use super::text_shaping::Drawn;
use super::text_stroke::GlyphStroke;
use crate::renderer::font_cache::DocumentFonts;
use font::cache::GlyphCache;
use font::key::FaceId;
use raster::Clip;
use raster::glyph::{GlyphBitmap, fill_glyph};
use raster::path::Path;
use raster::types::BlendMode;

/// The painting operations of a text rendering mode (PDF §9.3.6, Table 106).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TextModeOps {
    /// Fill the glyphs with the non-stroking colour.
    pub fill: bool,
    /// Stroke the glyph outlines with the stroking colour and parameters.
    pub stroke: bool,
    /// Add the glyph outlines to the clipping path.
    pub clip: bool,
}

impl TextModeOps {
    /// The operations of `Tr` operand `mode`; a value outside `0..=7` fills.
    pub(super) const fn of(mode: i32) -> Self {
        let (fill, stroke, clip) = match mode {
            1 => (false, true, false),
            2 => (true, true, false),
            3 => (false, false, false),
            4 => (true, false, true),
            5 => (false, true, true),
            6 => (true, true, true),
            7 => (false, false, true),
            _ => (true, false, false),
        };
        Self { fill, stroke, clip }
    }
}

/// The top-left and bottom-right corners of the bitmaps of `records`, as
/// [`fill_glyph`] places them: a bitmap's left edge lies `x_off` pixels left
/// of the pen and its top edge `y_off` pixels above it.
fn glyph_corners(records: &[GlyphRecord]) -> impl Iterator<Item = (f64, f64)> + '_ {
    records.iter().flat_map(|rec| {
        let x = f64::from(rec.pen_x) - f64::from(rec.bitmap.x_off);
        let y = f64::from(rec.pen_y) - f64::from(rec.bitmap.y_off);
        [
            (x, y),
            (
                x + f64::from(rec.bitmap.width),
                y + f64::from(rec.bitmap.height),
            ),
        ]
    })
}

/// The glyphs of a shown string as its rendering mode paints them: the
/// cached bitmaps of the fill, the cached coverage of the stroke, and the
/// device-space outlines of the glyphs that are clipped to, stroked as a
/// path or filled as a path.
pub(super) struct GlyphSink {
    /// The page's device pixel at the window's pixel `(0, 0)`.
    window_origin: (u32, u32),
    mode: TextModeOps,
    glyph_cache: GlyphCache,
    document_fonts: Arc<DocumentFonts>,
    /// The fill paints through filled outlines: every glyph when the fill
    /// colour is a pattern, else the glyphs of faces too large for glyph
    /// bitmaps.
    fill_as_path: bool,
    /// The fill colour is a pattern, resolved once over the union of the
    /// glyph outlines.
    pattern_fill: bool,
    /// The stroke as per-glyph cached coverage, when the graphics state
    /// admits it, with its cache key.
    stroke: Option<(GlyphStroke, [u64; 9])>,
    /// The cache identifier of each stroked face, by face and selector kind.
    stroked_ids: Vec<((FaceId, bool), FaceId)>,
    records: Vec<GlyphRecord>,
    stroke_records: Vec<GlyphRecord>,
    clip_outline: Path,
    stroke_outline: Path,
    fill_paths: Vec<Path>,
}

impl GlyphSink {
    /// An empty sink for glyphs painted by `mode`; `fill_as_path` and
    /// `pattern_fill` as the fields, `stroke` the cached stroke when the
    /// mode strokes.
    pub(super) fn new(
        window_origin: (u32, u32),
        mode: TextModeOps,
        glyph_cache: GlyphCache,
        document_fonts: Arc<DocumentFonts>,
        (fill_as_path, pattern_fill): (bool, bool),
        stroke: Option<GlyphStroke>,
        capacity: usize,
    ) -> Self {
        Self {
            window_origin,
            mode,
            glyph_cache,
            document_fonts,
            fill_as_path: mode.fill && fill_as_path,
            pattern_fill,
            stroke: stroke.map(|stroke| {
                let key = stroke.key();
                (stroke, key)
            }),
            stroked_ids: Vec::new(),
            records: Vec::with_capacity(capacity),
            stroke_records: Vec::new(),
            clip_outline: Path::new(),
            stroke_outline: Path::new(),
            fill_paths: Vec::new(),
        }
    }

    /// Add the glyph `drawn` at the exact pen position `pen` in the page's
    /// device space,
    /// condensed by `x_scale` to the document's advance: its bitmap when
    /// filling with bitmaps, its cached stroke coverage when stroking, and
    /// its outline when clipping, stroking without cached coverage, filling
    /// as a path or filling a glyph that has no bitmap.  `code` is the shown code that selects a glyph
    /// whose selector is no glyph index.
    ///
    /// The selector of a simple font's glyph (`by_gid == false`) is its
    /// code, resolved through the face's `code_to_gid` table or Unicode
    /// charmap (`Differences`, the standard 14).  That of a composite
    /// font's glyph, or of a fallback face's, is the glyph index the
    /// encoding `CMap` and `CIDToGIDMap` resolve to; re-resolving it through
    /// the charmap of a CID-keyed CFF subset, absent or unrelated, would
    /// select `.notdef`.
    pub(super) fn push(
        &mut self,
        (face, gid, by_gid): Drawn<'_>,
        code: u32,
        pen: (f64, f64),
        x_scale: f64,
    ) {
        let (dx, dy) = page_to_window(self.window_origin, pen);
        let (pen_x, x_frac, pen_y) = glyph_placement(dx, dy);
        let fill_as_bitmap = self.mode.fill && !self.fill_as_path;
        let bmp = if fill_as_bitmap {
            if by_gid {
                face.make_glyph_by_gid_cached_fitted(&self.glyph_cache, gid, x_frac, x_scale)
            } else {
                face.make_glyph_cached_fitted(&self.glyph_cache, gid, x_frac, x_scale)
            }
        } else {
            None
        };
        // A glyph the fill has no bitmap of, one past the bitmap budget of
        // its face, fills as its outline.
        let fill_outline = self.fill_as_path || (fill_as_bitmap && bmp.is_none());
        if let Some(bitmap) = bmp {
            self.records.push(GlyphRecord {
                pen_x,
                pen_y,
                bitmap,
            });
        }
        // The glyph outline relative to the pen, in device pixels with Trm
        // applied, y up, condensed by `scale`.
        let outline_at = |scale: f64| {
            if by_gid {
                face.glyph_path_by_gid_fitted(gid, scale)
            } else {
                face.glyph_path_fitted(code, scale)
            }
        };
        let stroked = self.stroke.as_ref().and_then(|(stroke, key)| {
            let id = if let Some(&(_, id)) = self
                .stroked_ids
                .iter()
                .find(|(k, _)| *k == (face.id, by_gid))
            {
                id
            } else {
                let id = self.document_fonts.stroked_face_id(face.id, by_gid, *key);
                self.stroked_ids.push(((face.id, by_gid), id));
                id
            };
            let selector = if by_gid { gid } else { code };
            face.cached_condensed(
                &self.glyph_cache,
                id,
                selector,
                true,
                x_frac,
                x_scale,
                |scale| stroke.coverage(&outline_at(scale)?, x_frac),
            )
        });
        // A glyph whose stroke has no cached coverage strokes with the
        // outline of the run.
        let stroke_path = self.mode.stroke && stroked.is_none();
        if let Some(bitmap) = stroked {
            self.stroke_records.push(GlyphRecord {
                pen_x,
                pen_y,
                bitmap,
            });
        }
        if !(self.mode.clip || stroke_path || fill_outline) {
            return;
        }
        // The glyph outline in the page's device space, translated by the
        // exact x and the bitmap's row, y flipped (device y is
        // positive-down), then moved into the window.
        let Some(mut path) = outline_at(x_scale) else {
            return;
        };
        let row = f64::from(pen_y) + f64::from(self.window_origin.1);
        for pt in &mut path.pts {
            (pt.x, pt.y) = page_to_window(self.window_origin, (pt.x + pen.0, row - pt.y));
        }
        if self.mode.clip {
            self.clip_outline.append(&path);
        }
        if stroke_path {
            self.stroke_outline.append(&path);
        }
        if fill_outline {
            self.fill_paths.push(path);
        }
    }
}

impl GlyphSink {
    /// Whether the fill paints through filled outlines rather than glyph
    /// bitmaps.
    pub(super) const fn fills_as_path(&self) -> bool {
        self.fill_as_path
    }
}

impl PageRenderer<'_> {
    /// Paint the glyphs of `sink`: fill them, then stroke them, then add
    /// their outlines to the text object's clip (modes 4–7), which `ET`
    /// applies.
    pub(super) fn paint_glyphs(&mut self, sink: &GlyphSink) {
        self.fill_glyphs(sink);
        // The cached strokes are one opaque object in the normal blend mode
        // under the soft mask in effect.
        if !sink.stroke_records.is_empty() && !self.gstate.current().stroke_paints_nothing() {
            let stroke_color = self.gstate.current().painted_stroke_color();
            let clip = self.gstate.current().clip.clone_shared();
            let paint = self.solid_paint(true, &stroke_color, u8::MAX, BlendMode::Normal);
            let region = PixelRect::around(
                glyph_corners(&sink.stroke_records),
                1.0,
                self.width,
                self.height,
            );
            self.paint_object(region, u8::MAX, |r| {
                r.blit_glyph_records(&sink.stroke_records, &clip, &paint, (&stroke_color, true));
            });
        }
        // The stroke parameters are user-space quantities under the CTM, as
        // for any stroked path (PDF §8.4.3.2); the text matrix and font size
        // scale the outline but not the line width.
        if sink.mode.stroke && !sink.stroke_outline.pts.is_empty() {
            self.stroke_text_outline(&sink.stroke_outline);
        }
        if sink.mode.clip {
            self.text_clip
                .get_or_insert_with(Path::new)
                .append(&sink.clip_outline);
        }
    }

    /// Fill the glyphs of `sink`, as bitmaps or filled outlines, as one
    /// object under the soft mask in effect.
    fn fill_glyphs(&mut self, sink: &GlyphSink) {
        if (sink.records.is_empty() && sink.fill_paths.is_empty())
            || self.gstate.current().fill_paints_nothing()
        {
            return;
        }
        let fill_color = self.gstate.current().painted_fill_color();
        let clip = self.gstate.current().clip.clone_shared();
        let gs = self.gstate.current();
        let paint = self.solid_paint(false, &fill_color, gs.fill_alpha, gs.blend_mode);
        // The outline fills lie within one pixel of their points.
        let outline_points = sink
            .fill_paths
            .iter()
            .flat_map(|path| path.pts.iter().map(|p| (p.x, p.y)));
        let region = PixelRect::around(
            glyph_corners(&sink.records).chain(outline_points),
            1.0,
            self.width,
            self.height,
        );
        let alpha = self.gstate.current().fill_alpha;
        self.paint_object(region, alpha, |r| {
            r.blit_glyph_records(&sink.records, &clip, &paint, (&fill_color, false));
            if sink.pattern_fill {
                let mut union = Path::new();
                for path in &sink.fill_paths {
                    union.append(path);
                }
                if !union.pts.is_empty() {
                    r.fill_path_area(&union, false);
                }
            } else {
                for path in &sink.fill_paths {
                    r.fill_path_area(path, false);
                }
            }
        });
    }

    /// Composite the glyph bitmaps of `records` through `clip` in the solid
    /// colour `colour` with the parameters `paint`: the stroke colour when
    /// `stroking`, else the fill colour.
    fn blit_glyph_records(
        &mut self,
        records: &[GlyphRecord],
        clip: &Clip,
        paint: &PaintParams,
        (colour, stroking): (&RasterColor, bool),
    ) {
        #[cfg(any(feature = "vulkan", all(windows, feature = "d3d12")))]
        if self.device_recording() {
            let glyphs = records.iter().map(|rec| PlacedGlyph {
                pen_x: rec.pen_x,
                pen_y: rec.pen_y,
                bitmap: &rec.bitmap,
            });
            self.record_device_glyphs(glyphs, stroking, paint.alpha);
            return;
        }
        let pipe = CanvasPipe::of(paint);
        let separator = if stroking {
            self.stroke_separator()
        } else {
            self.fill_separator()
        };
        let source = CanvasSource::Solid(colour, &separator);
        for rec in records {
            #[expect(
                clippy::cast_possible_wrap,
                reason = "bitmap sides are bounded by the glyph rasteriser and, for a \
                          cached stroke, by its 2^18-pixel area, far below i32::MAX"
            )]
            let glyph = GlyphBitmap {
                data: &rec.bitmap.data,
                x: rec.bitmap.x_off,
                y: rec.bitmap.y_off,
                w: rec.bitmap.width as i32,
                h: rec.bitmap.height as i32,
                aa: rec.bitmap.aa,
            };
            // Whether the glyph lay inside the clip matters only to a
            // clipping render mode, whose clip is built from the outlines.
            with_canvas!(&mut self.canvas, |b| {
                let pipe = pipe.at();
                source.paint(b, |b, src| {
                    let _ = fill_glyph(b, clip, &pipe, src, rec.pen_x, rec.pen_y, &glyph);
                });
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TextModeOps;

    #[test]
    fn text_mode_ops_follow_the_mode_table() {
        let ops = |m| {
            let o = TextModeOps::of(m);
            (o.fill, o.stroke, o.clip)
        };
        assert_eq!(ops(0), (true, false, false));
        assert_eq!(ops(1), (false, true, false));
        assert_eq!(ops(2), (true, true, false));
        assert_eq!(ops(3), (false, false, false));
        assert_eq!(ops(4), (true, false, true));
        assert_eq!(ops(5), (false, true, true));
        assert_eq!(ops(6), (true, true, true));
        assert_eq!(ops(7), (false, false, true));
        assert_eq!(ops(8), ops(0), "out of range fills");
        assert_eq!(ops(-1), ops(0), "out of range fills");
    }
}
