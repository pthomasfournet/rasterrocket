//! Type 3 text: `CharProc` content streams run per glyph, and the coverage
//! cache of shape-only (`d1`) glyphs.
//!
//! A `d1` glyph is a stencil painted in the text fill colour (PDF §9.6.5),
//! so its appearance at one glyph-to-device matrix and sub-pixel phase is
//! a coverage bitmap.  The bitmap is rendered once by running the
//! `CharProc` in a child renderer onto a transparent bitmap, whose alpha
//! plane records the coverage, and is kept in the document's glyph cache;
//! every later occurrence composites the bitmap in the fill colour as a
//! font glyph is.

use std::sync::Arc;

use color::Rgb8;
use font::bitmap::GlyphBitmap as CachedGlyph;
use font::key::GlyphKey;
use raster::bitmap::Bitmap;
use raster::glyph::{GlyphBitmap, fill_glyph};
use raster::pipe::PipeSrc;
use raster::types::BlendMode;

use super::super::gstate::ctm_multiply;
use super::super::text::TextState;
use super::MAX_FORM_DEPTH;
use super::PageRenderer;
use super::text_ops::{glyph_advance, translate_text_x};
use crate::content::Operator;
use crate::resources::ResourceKey;
use crate::resources::font::{Type3Data, Type3Glyph};

/// Largest device area, in pixels, of a cached Type 3 glyph; a larger
/// glyph runs its `CharProc` directly.
const MAX_CACHED_AREA: u32 = 512 * 512;

/// Positions per pixel, horizontally and vertically, at which a cached
/// glyph's origin is placed; a glyph is rendered once per phase pair.
const PHASES: u32 = 16;

/// Blank pixels kept around a glyph's declared bounding box in its
/// coverage bitmap.  Content reaching the outermost of them may extend
/// past the box, and such a glyph is not cached.
const MARGIN: i32 = 2;

impl PageRenderer<'_> {
    /// Execute Type 3 `CharProcs` for all character codes in `bytes`.
    ///
    /// Each `CharProc` is a PDF content stream that paints the glyph using the
    /// full graphics pipeline.  This method sets up the CTM for each glyph
    /// position and executes the stream, then advances the text matrix.
    /// A `d1` glyph of an indirect font is drawn from its cached coverage
    /// when its fill allows ([`Self::paint_type3_cached`]).
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors show_text() parameter set; all values are needed for glyph placement"
    )]
    pub(super) fn show_text_type3(
        &mut self,
        bytes: &[u8],
        t3: &Type3Data,
        font_key: &ResourceKey,
        font_size: f64,
        ctm: [f64; 6],
        mut tm: [f64; 6],
        char_spacing: f64,
        word_spacing: f64,
        horiz_scaling: f64,
        rise: f64,
        do_paint: bool,
    ) {
        // The CharProcs resolve their named resources in the font's own
        // `/Resources` when it has them, else in the resources in effect.
        let parent_resources = t3.resources.map(|id| {
            let child = self.resources.for_context(id);
            std::mem::replace(&mut self.resources, child)
        });
        // The CharProc's glyph space maps to text space through
        // `FontMatrix × [Tfs×Th 0 0 Tfs 0 Ts]` (PDF §9.4.4 with §9.6.5's
        // `FontMatrix`).
        let text_space = [font_size * horiz_scaling, 0.0, 0.0, font_size, 0.0, rise];
        let glyph_to_text = ctm_multiply(&t3.font_matrix, &text_space);
        for &byte in bytes {
            // A code without a CharProc paints nothing but advances by its
            // width (PDF §9.4.4).
            if let Some(glyph) = t3.glyph(byte)
                && do_paint
            {
                let charproc_ctm = ctm_multiply(&ctm_multiply(&glyph_to_text, &tm), &ctm);
                if !self.paint_type3_cached(font_key, byte, glyph, charproc_ctm) {
                    self.paint_type3_glyph(byte, glyph, charproc_ctm);
                }
            }

            // Advance the text matrix by the glyph width (PDF §9.4.4).
            // Widths are in glyph space; `FontMatrix[0]` maps them to text
            // space.
            let width = f64::from(t3.width(byte)) * t3.font_matrix[0];
            let extra = if byte == b' ' { word_spacing } else { 0.0 };
            let tx_adv = glyph_advance(width, font_size, char_spacing, extra, horiz_scaling);
            tm = translate_text_x(tm, tx_adv);
        }
        if let Some(parent) = parent_resources {
            self.resources = parent;
        }

        // Write updated text matrix back.
        self.gstate.current_mut().text.text_matrix = tm;
    }

    /// Execute the `CharProc` `glyph` of code `byte` under the
    /// glyph-to-device matrix `charproc_ctm`, in a graphics state saved
    /// around it.
    fn paint_type3_glyph(&mut self, byte: u8, glyph: &Type3Glyph, charproc_ctm: [f64; 6]) {
        if self.form_depth >= MAX_FORM_DEPTH {
            log::warn!(
                "rasterrocket-interp: Type 3 CharProc depth {MAX_FORM_DEPTH} exceeded — \
                 glyph 0x{byte:02X} not painted (text position still advances)"
            );
            return;
        }
        self.gstate.save();
        self.form_depth += 1;
        self.gstate.current_mut().ctm = charproc_ctm;
        // Reset text state inside charproc (PDF §9.6.5 — charproc is a
        // fresh graphics state, not a text object).
        self.gstate.current_mut().text = TextState::default();

        self.execute_content_stream(&glyph.ops);

        self.form_depth -= 1;
        self.gstate.restore();
    }

    /// Draw the `d1` glyph `glyph` of code `byte` of the font `font_key`
    /// under `charproc_ctm` from its cached coverage, rendering the
    /// coverage on first use.  `false`, with nothing painted, when the
    /// glyph must run its `CharProc` instead: a `d0` glyph, a font
    /// dictionary that is not an indirect object, a fill other than an
    /// opaque colour in the normal blend mode, a degenerate or oversized
    /// bounding box, or content that leaves the declared bounding box.
    fn paint_type3_cached(
        &mut self,
        font_key: &ResourceKey,
        byte: u8,
        glyph: &Type3Glyph,
        charproc_ctm: [f64; 6],
    ) -> bool {
        let Some(&Operator::Type3GlyphWidthBBox(_, _, llx, lly, urx, ury)) = glyph.ops.first()
        else {
            return false;
        };
        let ResourceKey::Object(font_id) = font_key else {
            return false;
        };
        let gs = self.gstate.current();
        if gs.fill_pattern.is_some() || gs.fill_alpha != 255 || gs.blend_mode != BlendMode::Normal {
            return false;
        }
        if self.form_depth >= MAX_FORM_DEPTH {
            return false;
        }
        let [m0, m1, m2, m3, tx, ty] = charproc_ctm;
        let matrix = [m0, m1, m2, m3];
        let Some(frame) = GlyphFrame::new(
            matrix,
            [llx, lly, urx, ury],
            tx,
            f64::from(self.height) - ty,
        ) else {
            return false;
        };
        let face_id = self.font_cache.type3_face_id(*font_id, matrix);
        let key = GlyphKey::new(
            face_id,
            u32::from(byte) | (u32::from(frame.x_frac) << 8) | (u32::from(frame.y_frac) << 16),
            1,
            0,
            self.vector_antialias,
        );
        let cache = self.font_cache.glyph_cache().clone();
        let Some(coverage) =
            cache.get_or_render(key, || Some(self.render_type3_coverage(glyph, &frame)))
        else {
            return false;
        };
        if coverage.width == 0 {
            return false;
        }
        if self.gstate.current().fill_color_space.paints_nothing() {
            return true;
        }
        let fill = self.gstate.current().painted_fill_color().to_vec();
        let clip = self.gstate.current().clip.clone_shared();
        let pipe = Self::make_pipe(255, BlendMode::Normal, None);
        #[expect(
            clippy::cast_possible_wrap,
            reason = "a cached glyph is at most MAX_CACHED_AREA pixels; its sides fit i32"
        )]
        let bitmap = GlyphBitmap {
            data: &coverage.data,
            x: coverage.x_off,
            y: coverage.y_off,
            w: coverage.width as i32,
            h: coverage.height as i32,
            aa: coverage.aa,
        };
        let _ = fill_glyph::<Rgb8>(
            &mut self.bitmap,
            &clip,
            &pipe,
            &PipeSrc::Solid(&fill),
            frame.pen_x,
            frame.pen_y,
            &bitmap,
        );
        true
    }

    /// Render the coverage of the `d1` glyph `glyph` placed by `frame`:
    /// its `CharProc` runs in a child renderer onto a transparent bitmap
    /// covering the declared bounding box and a margin, whose alpha plane
    /// is the coverage.  A bitmap of width 0 marks a glyph whose content
    /// reaches the margin's outer edge and so may extend past the box.
    fn render_type3_coverage(&mut self, glyph: &Type3Glyph, frame: &GlyphFrame) -> CachedGlyph {
        let (width, height) = (frame.width, frame.height);
        let doc = self.resources.doc();
        let context = self.resources.resource_context_id();
        let mut child = PageRenderer::new(width, height, doc, context);
        child.set_document_fonts(Arc::clone(self.font_cache.document_fonts()));
        child.set_antialias(self.text_antialias, self.vector_antialias);
        child.bitmap = Bitmap::new(width, height, 1, true);
        child.pattern_depth = self.pattern_depth;
        child.form_depth = self.form_depth + 1;
        child.adopt_parent_budget(self);
        child.share_memos(self);
        child.set_optional_content(Arc::clone(self.optional_content()));
        child.resources = self.resources.inherited();
        child.gstate.current_mut().ctm = frame.child_ctm;
        child.execute(&glyph.ops);
        child.fold_child_into(self);
        let bitmap = child.finish().0;
        let data = bitmap.alpha_plane().map(<[u8]>::to_vec).unwrap_or_default();
        let (w, h) = (width as usize, height as usize);
        let touches_edge = data.len() != w * h
            || data[..w].iter().any(|&v| v != 0)
            || data[w * (h - 1)..].iter().any(|&v| v != 0)
            || data
                .chunks_exact(w)
                .any(|row| row[0] != 0 || row[w - 1] != 0);
        if touches_edge {
            return CachedGlyph {
                x_off: 0,
                y_off: 0,
                width: 0,
                height: 0,
                aa: true,
                data: Vec::new(),
            };
        }
        CachedGlyph {
            x_off: frame.x_off,
            y_off: frame.y_off,
            width,
            height,
            aa: true,
            data,
        }
    }
}

/// Where a cached Type 3 glyph's coverage bitmap lies: the pen pixel, the
/// phase of the glyph origin in [`PHASES`] steps per pixel, the bitmap's size and offset
/// from the pen, and the CTM that places the glyph in the bitmap.
struct GlyphFrame {
    pen_x: i32,
    pen_y: i32,
    x_frac: u8,
    y_frac: u8,
    width: u32,
    height: u32,
    /// Offset of the pen from the bitmap's left edge (`fill_glyph`'s `x`).
    x_off: i32,
    /// Offset of the pen from the bitmap's top edge (`fill_glyph`'s `y`).
    y_off: i32,
    /// Glyph space to the child bitmap's y-up device space.
    child_ctm: [f64; 6],
}

impl GlyphFrame {
    /// The frame of a glyph with glyph-to-device matrix `m` (2×2 part),
    /// declared bounding box `bbox` in glyph space, and origin at device
    /// pixel position `(x, y)` (y down).  `None` for a degenerate or
    /// non-finite box, or one larger than [`MAX_CACHED_AREA`].
    fn new(m: [f64; 4], bbox: [f64; 4], x: f64, y: f64) -> Option<Self> {
        let [llx, lly, urx, ury] = bbox;
        let finite = bbox.iter().chain([&x, &y]).all(|v| v.is_finite());
        if !finite || urx <= llx || ury <= lly {
            return None;
        }
        let steps = f64::from(PHASES);
        // The origin snaps to the nearest phase, so a cached glyph lies
        // within half a phase of its exact position in either direction.
        let (x, y) = ((x * steps).round() / steps, (y * steps).round() / steps);
        let (column, row) = (x.floor(), y.floor());
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a snapped position's fraction in steps is an integer in [0, PHASES)"
        )]
        let (x_frac, y_frac) = (
            ((x - column) * steps).round() as u8,
            ((y - row) * steps).round() as u8,
        );
        let (fx, fy) = (f64::from(x_frac) / steps, f64::from(y_frac) / steps);
        // Device extents of the box relative to the origin: x right, y down.
        let corners = [(llx, lly), (llx, ury), (urx, lly), (urx, ury)]
            .map(|(u, v)| (m[0].mul_add(u, m[2] * v), -m[1].mul_add(u, m[3] * v)));
        let x_min = corners.iter().map(|p| p.0).fold(f64::INFINITY, f64::min) + fx;
        let x_max = corners
            .iter()
            .map(|p| p.0)
            .fold(f64::NEG_INFINITY, f64::max)
            + fx;
        let y_min = corners.iter().map(|p| p.1).fold(f64::INFINITY, f64::min) + fy;
        let y_max = corners
            .iter()
            .map(|p| p.1)
            .fold(f64::NEG_INFINITY, f64::max)
            + fy;
        let span = (x_max - x_min).max(y_max - y_min);
        if !span.is_finite() || span > f64::from(MAX_CACHED_AREA) {
            return None;
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "extents are bounded by MAX_CACHED_AREA above"
        )]
        let (x0, x1, y0, y1) = (
            x_min.floor() as i32 - MARGIN,
            x_max.ceil() as i32 + MARGIN,
            y_min.floor() as i32 - MARGIN,
            y_max.ceil() as i32 + MARGIN,
        );
        let (width, height) = (x1 - x0, y1 - y0);
        let area = i64::from(width) * i64::from(height);
        if area > i64::from(MAX_CACHED_AREA) {
            return None;
        }
        let (width, height) = (u32::try_from(width).ok()?, u32::try_from(height).ok()?);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "device coordinates of a page fit i32"
        )]
        let (pen_x, pen_y) = (column as i32, row as i32);
        let origin_x = fx - f64::from(x0);
        let origin_y_down = fy - f64::from(y0);
        Some(Self {
            pen_x,
            pen_y,
            x_frac,
            y_frac,
            width,
            height,
            x_off: -x0,
            y_off: -y0,
            child_ctm: [
                m[0],
                m[1],
                m[2],
                m[3],
                origin_x,
                f64::from(height) - origin_y_down,
            ],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cached `d1` glyph of a font without resources of its own resolves
    /// names in the resources in effect, even when those are a dictionary
    /// given directly rather than an object's `/Resources`.
    #[test]
    fn a_cached_glyph_resolves_names_in_the_resources_in_effect() {
        let charproc = b"1 0 0 0 1 1 d1 /M Do";
        let doc = crate::test_helpers::make_doc_with_objects(&[
            (4, b"<< /Type /Page >>".to_vec()),
            (
                5,
                b"<< /Type /Font /Subtype /Type3 /FontBBox [0 0 1 1] \
                  /FontMatrix [1 0 0 1 0 0] /CharProcs << /a 6 0 R >> \
                  /Encoding << /Differences [65 /a] >> /FirstChar 65 /LastChar 65 \
                  /Widths [1] >>"
                    .to_vec(),
            ),
            (
                6,
                format!(
                    "<< /Length {} >>\nstream\n{}\nendstream",
                    charproc.len(),
                    String::from_utf8_lossy(charproc)
                )
                .into_bytes(),
            ),
            (
                7,
                b"<< /Type /XObject /Subtype /Image /Width 1 /Height 1 /ImageMask true \
                  /BitsPerComponent 1 /Length 1 >>\nstream\n\x00\nendstream"
                    .to_vec(),
            ),
        ]);
        let resources = pdf::parse_object(
            b"<< /Font << /T 5 0 R >> /XObject << /M 7 0 R >> >>",
            &mut 0,
        )
        .unwrap();
        let mut renderer = PageRenderer::new(32, 32, &doc, (4, 0));
        renderer.resources = renderer
            .resources
            .with_resources((0, 0), resources.as_dict().unwrap());
        renderer.execute(&crate::content::parse(b"BT /T 16 Tf 4 4 Td (A) Tj ET"));
        // The glyph covers device x 4..20, rows 12..28.
        let row = renderer.bitmap.row_bytes(20);
        assert_eq!(&row[12 * 3..12 * 3 + 3], &[0, 0, 0], "glyph painted");
    }

    /// A unit-square glyph at 20 px places its bitmap so that the origin
    /// sits at the pen plus its phase, and the box plus margin is covered.
    #[test]
    fn frame_places_the_box_around_the_origin() {
        let frame = GlyphFrame::new([20.0, 0.0, 0.0, 20.0], [0.0, 0.0, 1.0, 1.0], 10.5, 30.25)
            .expect("frame");
        assert_eq!((frame.pen_x, frame.pen_y), (10, 30));
        assert_eq!((frame.x_frac, frame.y_frac), (8, 4));
        // x: [0.5, 20.5] → [-2, 23]; y down: [-19.75, 0.25] → [-22, 3].
        assert_eq!((frame.width, frame.height), (25, 25));
        assert_eq!((frame.x_off, frame.y_off), (2, 22));
        // The origin lands 0.5 px right of the left margin column 2 and,
        // y up in the child, 3 - 0.25 px above the bottom.
        assert_eq!(frame.child_ctm[4], 2.5);
        assert_eq!(frame.child_ctm[5], 25.0 - 22.25);
    }

    /// The origin snaps to the nearest phase: just below a pixel boundary
    /// it carries into the next pixel at phase 0, and a fraction rounds
    /// down as well as up.
    #[test]
    fn frame_rounds_the_origin_to_the_nearest_phase() {
        let unit = [20.0, 0.0, 0.0, 20.0];
        let bbox = [0.0, 0.0, 1.0, 1.0];
        let frame = GlyphFrame::new(unit, bbox, 10.99, 5.02).expect("frame");
        assert_eq!((frame.pen_x, frame.x_frac), (11, 0));
        assert_eq!((frame.pen_y, frame.y_frac), (5, 0));
        let frame = GlyphFrame::new(unit, bbox, 10.53, 5.47).expect("frame");
        assert_eq!((frame.pen_x, frame.x_frac), (10, 8));
        assert_eq!((frame.pen_y, frame.y_frac), (5, 8));
    }

    #[test]
    fn degenerate_or_huge_boxes_have_no_frame() {
        assert!(GlyphFrame::new([20.0, 0.0, 0.0, 20.0], [0.0, 0.0, 0.0, 0.0], 0.0, 0.0).is_none());
        assert!(
            GlyphFrame::new([2000.0, 0.0, 0.0, 2000.0], [0.0, 0.0, 1.0, 1.0], 0.0, 0.0).is_none()
        );
    }
}
