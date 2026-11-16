//! Text rendering: `show_text`, and the reading of a shown string's codes
//! for their layout (see [`super::text_layout`]).

use std::sync::Arc;

use super::PageRenderer;
use super::text_census::GlyphCensus;
use super::text_context::{Following, NextString, TextTail, neighbours as neighbours_at};
use super::text_glyphs::{GlyphSink, TextModeOps};
use super::text_layout::{
    ChainExtent, Continuation, Layout, ShapingContext, TextFrame, code_advances, drawn_by,
    leading_chain,
};
use super::text_ops::{is_singular, text_rendering_matrix, translate_text_x};
use super::text_shaping::{
    CodeText, Drawn, ShapedRun, last_texts, shaped_runs, shapes, takes_context,
};
use super::text_stroke::GlyphStroke;
use crate::renderer::font_cache::{ActiveFont, FontCache, PageFace};
use crate::renderer::text::{TextMatrix, TextState};
use crate::resources::ResourceKey;
use crate::resources::cmap::DecodedCode;
use crate::resources::font::{CidEncoding, Type3Data};

/// The strings after a string that are read ahead for the extent of the
/// chain its last run starts or continues: the runs of a word split
/// between more strings are fitted by the extent of its first ones.
const CHAIN_STRINGS: usize = 16;

/// A shown string read for layout: its frame and text rendering matrix,
/// its codes, their advances and, when it is shaped, their texts, whether
/// its glyphs are fitted to its advances (see [`Layout::fit`]) and the
/// shaping context of its runs.
struct ReadString {
    frame: TextFrame,
    trm: [f64; 4],
    decoded: Vec<DecodedCode>,
    advances: Vec<f64>,
    texts: Vec<CodeText>,
    shaped: bool,
    fit: bool,
    context: ShapingContext,
}

impl ReadString {
    /// The face that draws the string: `None` when its text rendering
    /// matrix is singular, mapping every glyph to zero area, or the face
    /// cannot load (below one device pixel, an unreadable program).
    fn face<'c>(
        &self,
        cache: &'c mut FontCache,
        font: &ActiveFont,
        name: &[u8],
    ) -> Option<&'c PageFace> {
        if is_singular(&self.trm) {
            None
        } else {
            cache.get_or_load(font, name, self.trm)
        }
    }

    /// The drawing glyph of each code drawn by `face` of a font encoded by
    /// `cid_enc`, and the shaped runs among them.
    fn runs<'f>(
        &self,
        face: Option<&'f PageFace>,
        cid_enc: Option<&CidEncoding>,
    ) -> (Vec<Option<Drawn<'f>>>, Vec<ShapedRun<'f>>) {
        let drawn = drawn_by(face, cid_enc, &self.decoded);
        let runs = match face {
            Some(face) if self.shaped => {
                shaped_runs(face, &self.texts, &drawn, &self.context.neighbours)
            }
            _ => Vec::new(),
        };
        (drawn, runs)
    }

    /// The end of the string at the text matrix `end`: see [`TextTail`].
    fn tail(&self, end: TextMatrix, continuation: Option<Continuation>) -> TextTail {
        TextTail {
            texts: last_texts(&self.context.neighbours.before, &self.texts),
            end,
            rise: self.frame.rise,
            em: self.frame.em(),
            continuation,
        }
    }
}

impl PageRenderer<'_> {
    /// Render a byte string using the current font, colour, and text matrix;
    /// `following` is what follows it in its content stream, from which the
    /// string shown next is read for shaping context.
    ///
    /// The glyphs are laid out while the font cache is borrowed and painted
    /// from the [`GlyphSink`] afterwards.
    pub(super) fn show_text(&mut self, bytes: &[u8], following: Following<'_, '_>) {
        if bytes.is_empty() {
            return;
        }
        self.diag.has_vector_text = true;
        // The end of the string shown before, replaced by this string's when
        // it is shaped and dropped otherwise.
        let tail = self.text_tail.take();
        // Mode 3 paints nothing and clips nothing, but the pen still
        // advances by every glyph's width (PDF §9.4.4): the run is laid out,
        // rasterising nothing it will not use.
        let mode = TextModeOps::of(self.gstate.current().text.render_mode);
        let Some(font) = self.gstate.current().text.font.clone() else {
            log::debug!(
                "rasterrocket-interp: no font dict for /{}",
                String::from_utf8_lossy(&self.gstate.current().text.font_name)
            );
            // Without a font the codes are not delimited: each byte counts
            // as a glyph.
            let mut census = self.glyph_census(mode, false);
            if let Some(census) = census.as_mut() {
                census.fontless(bytes.len());
            }
            self.record_glyphs(census);
            if let Some(text) = self.text_trace_line(None, mode, false) {
                self.trace_text(&text, &format!("fontless n={}", bytes.len()));
            }
            return;
        };
        if let Some(t3) = &font.descriptor.type3 {
            self.show_type3_string(bytes, t3, &font.key, mode);
            return;
        }
        let glyph_stroke = if mode.stroke {
            GlyphStroke::new(
                self.gstate.current(),
                &self.device_matrix(),
                self.vector_antialias,
            )
        } else {
            None
        };
        let mut census = self.glyph_census(mode, false);
        let mut glyph_trace = self.tracing().then(String::new);
        let (sink, tm, new_tail) = self.lay_out(
            &font,
            bytes,
            following,
            tail,
            (mode, glyph_stroke),
            (census.as_mut(), glyph_trace.as_mut()),
        );
        self.record_glyphs(census);
        if let Some(glyphs) = glyph_trace
            && let Some(text) = self.text_trace_line(Some(&font.key), mode, sink.fills_as_path())
        {
            self.trace_text(&text, &glyphs);
        }
        self.paint_glyphs(&sink);
        self.gstate.current_mut().text.text_matrix = tm;
        self.text_tail = new_tail;
    }

    /// Show `bytes` with the Type 3 font of glyph descriptions `t3` and
    /// resource key `font_key` in the mode of operations `mode`.
    ///
    /// Type 3 glyphs are content streams executed at the current pen
    /// position, painting directly into the page.  A glyph description
    /// paints with its own operators and has no outline to fill, stroke or
    /// clip with (PDF §9.6.5): every mode but the invisible one executes the
    /// glyph description unchanged, and none adds to the clip.  The glyphs
    /// are one object under the soft mask in effect.
    fn show_type3_string(
        &mut self,
        bytes: &[u8],
        t3: &Type3Data,
        font_key: &ResourceKey,
        mode: TextModeOps,
    ) {
        let paints = mode.fill || mode.stroke || mode.clip;
        let mut census = self.glyph_census(mode, true);
        let text_trace = self.text_trace_line(Some(font_key), mode, false);
        let mut glyph_trace = text_trace.as_ref().map(|_| String::new());
        let gs = self.gstate.current();
        let ts = &gs.text;
        let (ctm, tm) = (gs.ctm, ts.text_matrix);
        let (font_size, char_spacing, word_spacing) =
            (ts.font_size, ts.char_spacing, ts.word_spacing);
        let (horiz_scaling, rise) = (ts.horiz_scaling / 100.0, ts.rise);
        let region = self.clip_rect();
        let alpha = gs.fill_alpha;
        self.paint_object(region, alpha, |r| {
            r.show_text_type3(
                bytes,
                t3,
                font_key,
                font_size,
                ctm,
                tm,
                char_spacing,
                word_spacing,
                horiz_scaling,
                rise,
                paints,
                (census.as_mut(), glyph_trace.as_mut()),
            );
        });
        self.record_glyphs(census);
        if let (Some(text), Some(glyphs)) = (text_trace, glyph_trace) {
            self.trace_text(&text, &glyphs);
        }
    }

    /// Lay out the glyphs of `bytes`, shown with `font` in the current text
    /// state, after the string whose end is `tail` and before `following`,
    /// into a sink for `mode` and the cached stroke `glyph_stroke`, counting
    /// each code's glyph in `census` and printing it to `glyph_trace`.
    /// Returns the sink, the text matrix after the string and, when the
    /// string is shaped, its end.
    fn lay_out(
        &mut self,
        font: &ActiveFont,
        bytes: &[u8],
        following: Following<'_, '_>,
        tail: Option<TextTail>,
        (mode, glyph_stroke): (TextModeOps, Option<GlyphStroke>),
        counted: (Option<&mut GlyphCensus>, Option<&mut String>),
    ) -> (GlyphSink, TextMatrix, Option<TextTail>) {
        let ts = self.gstate.current().text.clone();
        let pattern_fill = mode.fill && self.gstate.current().fill_pattern.is_some();
        let document_fonts = Arc::clone(self.font_cache.document_fonts());
        // The glyph cache is shared, not copied, by the clone.
        let glyph_cache = self.font_cache.glyph_cache().clone();
        let read = self.read_string(&ts, font, bytes, following, tail, 0);
        let descriptor = &*font.descriptor;
        let cid_enc = descriptor.cid_encoding.as_ref();
        let face = read.face(&mut self.font_cache, font, &ts.font_name);
        let fill_as_path = face.is_some_and(|f| pattern_fill || f.prefers_path_fill());
        let mut sink = GlyphSink::new(
            self.window_origin,
            mode,
            glyph_cache,
            document_fonts,
            (fill_as_path, pattern_fill),
            glyph_stroke,
            bytes.len(),
        );
        let (drawn, runs) = read.runs(face, cid_enc);
        let layout = Layout {
            descriptor,
            vertical: cid_enc.filter(|enc| enc.vertical),
            decoded: &read.decoded,
            drawn: &drawn,
            advances: &read.advances,
            frame: read.frame,
            fit: read.fit,
        };
        let (end, continuation) =
            layout.place(ts.text_matrix, &runs, &read.context, &mut sink, counted);
        let tail = read.shaped.then(|| read.tail(end, continuation));
        (sink, end, tail)
    }

    /// Read `bytes`, shown with `font` in the text state `ts`, after the
    /// string whose end is `tail` and before `following`, for its layout;
    /// `depth` is the number of strings it follows the string being shown
    /// by, when it is read ahead.
    ///
    /// The 2×2 text rendering matrix in device pixels (PDF §9.4.4) is the
    /// same for every glyph of the string.  A string whose face does not
    /// load draws no glyph and the pen still advances by the document's
    /// widths.  Runs are shaped in horizontal writing only, between the
    /// strings shown next to them on their baseline.
    fn read_string(
        &mut self,
        ts: &TextState,
        font: &ActiveFont,
        bytes: &[u8],
        following: Following<'_, '_>,
        tail: Option<TextTail>,
        depth: usize,
    ) -> ReadString {
        // `Tz` is a percentage; `Tz 0` collapses the glyphs and the
        // horizontal advances to zero width.
        let frame = TextFrame {
            ctm: self.gstate.current().ctm,
            page_h: self.page_height,
            font_size: ts.font_size,
            horiz_scaling: ts.horiz_scaling / 100.0,
            rise: ts.rise,
        };
        let tm = ts.text_matrix;
        let descriptor = &*font.descriptor;
        let cid_enc = descriptor.cid_encoding.as_ref();
        let trm = text_rendering_matrix(&tm, &frame.ctm, frame.font_size, frame.horiz_scaling);
        let face = if is_singular(&trm) {
            None
        } else {
            self.font_cache
                .get_or_load_showing(font, &ts.font_name, trm, bytes)
        };
        let decoded: Vec<DecodedCode> = CidEncoding::shown_codes(cid_enc, bytes).collect();
        let drawn = drawn_by(face, cid_enc, &decoded);
        let spacing = (ts.char_spacing, ts.word_spacing);
        let advances = code_advances(descriptor, face, &decoded, &drawn, &frame, spacing);
        let shaped = face.is_some_and(|f| shapes(descriptor, f));
        let fit = face.is_some_and(|f| f.substitute)
            && !descriptor.substitute.standard
            && !cid_enc.is_some_and(|enc| enc.vertical);
        let texts: Vec<CodeText> = match face {
            Some(face) if shaped => decoded.iter().map(|d| face.text(d.code, d.value)).collect(),
            _ => Vec::new(),
        };
        let mut read = ReadString {
            frame,
            trm,
            decoded,
            advances,
            texts,
            shaped,
            fit,
            context: ShapingContext::default(),
        };
        if !shaped {
            return read;
        }
        let em = frame.em();
        if let Some(tail) =
            tail.filter(|tail| neighbours_at(&tail.end, &tm, frame.rise - tail.rise, tail.em))
        {
            read.context.neighbours.before = tail.texts;
            read.context.continuation = tail.continuation.map(|c| Continuation { at: 0, ..c });
        }
        if read.texts.last().is_some_and(|&text| takes_context(text)) {
            let mut end = ts.clone();
            end.text_matrix = read
                .advances
                .iter()
                .fold(tm, |tm, &advance| translate_text_x(tm, advance));
            let end_tm = end.text_matrix;
            if let Some(next) = self.next_string(end, em, following) {
                read.context.neighbours.after.clone_from(&next.texts);
                read.context.next_start = next.start;
                if fit && depth < CHAIN_STRINGS {
                    read.context.ahead =
                        self.chain_ahead(&next, read.tail(end_tm, None), depth + 1);
                }
            }
        }
        read
    }

    /// The extent of the runs of the string `next`, shown after the string
    /// whose end is `tail`, and of the strings after it, that continue a
    /// chain from the string before (see [`super::text_layout`]); `None`
    /// when `next` is not fitted or its first run continues no chain.
    /// `depth` is as for [`Self::read_string`].
    fn chain_ahead(
        &mut self,
        next: &NextString<'_, '_>,
        tail: TextTail,
        depth: usize,
    ) -> Option<ChainExtent> {
        let font = next.state.font.clone()?;
        let read = self.read_string(
            &next.state,
            &font,
            next.bytes,
            next.following,
            Some(tail),
            depth,
        );
        if !read.fit {
            return None;
        }
        let face = read.face(&mut self.font_cache, &font, &next.state.font_name);
        let (_, runs) = read.runs(face, font.descriptor.cid_encoding.as_ref());
        let (extent, past) =
            leading_chain(&runs, &read.advances, read.frame.em(), read.decoded.len())?;
        Some(extent.then(read.context.ahead.filter(|_| past)))
    }

    /// End a text object (`ET`): intersect the clip with the union of the
    /// glyph outlines that the object's text shown in a clipping render
    /// mode added, under the non-zero winding rule (PDF §9.3.6).  Glyphs
    /// painted within the object are not clipped by one another.  Text
    /// shown in a clipping mode that added no outline leaves an empty clip.
    pub(super) fn end_text(&mut self) {
        if let Some(outline) = self.text_clip.take() {
            self.clip_path_into_gstate(&outline, false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PageRenderer;

    /// Run `content` on a 64 × 32 px page whose `/F1` is Helvetica and
    /// return the number of pixels darker than mid-grey.  The graphics
    /// state `/Hidden` sets a soft mask that is zero over the whole page.
    fn dark_pixels(content: &[u8]) -> usize {
        let mask = "0 g 0 0 64 32 re f";
        let doc = crate::test_helpers::make_doc_with_objects(&[
            (
                4,
                "<< /Type /Page /Resources << /Font << /F1 5 0 R >> \
                 /ExtGState << /Hidden << /SMask << /S /Luminosity /G 6 0 R >> >> >> >> >>"
                    .to_owned(),
            ),
            (
                5,
                "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_owned(),
            ),
            (
                6,
                format!(
                    "<< /Type /XObject /Subtype /Form /BBox [0 0 64 32] \
                     /Group << /S /Transparency /CS /DeviceGray >> /Length {} >>\n\
                     stream\n{mask}\nendstream",
                    mask.len()
                ),
            ),
        ]);
        let mut renderer = PageRenderer::new(64, 32, &doc, (4, 0));
        renderer.execute(&crate::content::parse(content));
        (0..32)
            .flat_map(|y| renderer.rgb_canvas().row_bytes(y).to_vec())
            .filter(|&v| v < 128)
            .count()
    }

    /// Glyphs painted under a soft mask are masked over their whole
    /// extent, above the baseline as below it.
    #[test]
    fn a_soft_mask_masks_whole_glyphs() {
        let shown = b"BT /F1 24 Tf 4 8 Td (AB) Tj ET";
        assert!(dark_pixels(shown) > 40, "the glyphs paint");
        for mode in [0, 2] {
            let content = format!("/Hidden gs BT /F1 24 Tf {mode} Tr 4 8 Td (AB) Tj ET");
            assert_eq!(dark_pixels(content.as_bytes()), 0, "mode {mode}");
        }
    }

    /// A stroke drawn from cached per-glyph coverage covers what the
    /// stroke of the run's outline covers, to within the quarter-pixel
    /// placement of the glyphs.  A dash cycle far below a pixel strokes
    /// solid but keeps the stroke off the cache.
    #[test]
    fn a_cached_glyph_stroke_matches_the_outline_stroke() {
        let run = |dash: &str| {
            let content = format!("{dash} 0.8 w BT /F1 24 Tf 1 Tr 4 8 Td (AB) Tj ET");
            dark_pixels(content.as_bytes())
        };
        let (cached, outline) = (run("[] 0 d"), run("[0.001] 0 d"));
        assert!(outline > 40, "the outline stroke paints: {outline}");
        assert!(
            cached.abs_diff(outline) * 20 <= outline,
            "cached {cached} vs outline {outline}"
        );
    }

    /// The glyphs of every text-showing operator of a text object in a
    /// clipping mode clip together at `ET`: two runs clip to their union,
    /// not to the intersection of two disjoint glyph sets.
    #[test]
    fn a_text_object_clips_to_the_union_of_its_runs() {
        let one_run = dark_pixels(b"BT /F1 24 Tf 7 Tr 4 8 Td (AB) Tj ET 0 0 64 32 re f");
        let two_runs = dark_pixels(b"BT /F1 24 Tf 7 Tr 4 8 Td [(A) -10 (B)] TJ ET 0 0 64 32 re f");
        assert!(one_run > 40, "the clip admits the glyphs: {one_run}");
        assert!(
            two_runs > one_run / 2,
            "both runs admit paint: {two_runs} vs {one_run}"
        );
    }

    /// A fill-and-clip run is not clipped by the run before it in the same
    /// text object.
    #[test]
    fn a_run_is_not_clipped_by_an_earlier_run_of_its_object() {
        let first = dark_pixels(b"BT /F1 24 Tf 4 Tr 4 8 Td (A) Tj ET");
        let both = dark_pixels(b"BT /F1 24 Tf 4 Tr 4 8 Td (A) Tj (B) Tj ET");
        assert!(first > 20, "{first}");
        assert!(
            both > first + 20,
            "the second run paints: {both} vs {first}"
        );
    }

    /// `Tz 0` collapses the glyphs to zero width: the run paints nothing.
    #[test]
    fn zero_horizontal_scaling_paints_nothing() {
        assert!(dark_pixels(b"BT /F1 24 Tf 4 8 Td (AB) Tj ET") > 40);
        assert_eq!(dark_pixels(b"BT /F1 24 Tf 0 Tz 4 8 Td (AB) Tj ET"), 0);
    }

    /// A run in a face too small to load still advances the pen by the
    /// document's widths.
    #[test]
    fn a_run_below_one_pixel_advances_the_pen() {
        let doc = crate::test_helpers::make_doc_with_objects(&[
            (
                4,
                "<< /Type /Page /Resources << /Font << /F1 5 0 R >> >> >>",
            ),
            (
                5,
                "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica \
                 /FirstChar 65 /LastChar 66 /Widths [600 400] >>",
            ),
        ]);
        let mut renderer = PageRenderer::new(64, 32, &doc, (4, 0));
        renderer.execute(&crate::content::parse(b"BT /F1 0.5 Tf (AB) Tj"));
        let tm = renderer.gstate.current().text.text_matrix;
        assert!((tm[4] - 0.5).abs() < 1e-9, "{tm:?}");
    }

    /// Clipping-mode text that adds no outline leaves an empty clip.
    #[test]
    fn clipping_text_without_outlines_empties_the_clip() {
        assert_eq!(
            dark_pixels(b"BT /F1 24 Tf 7 Tr 4 8 Td ( ) Tj ET 0 0 64 32 re f"),
            0
        );
        assert_eq!(dark_pixels(b"0 0 64 32 re f").min(1), 1);
    }
}
