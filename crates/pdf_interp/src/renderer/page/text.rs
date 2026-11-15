//! Text rendering: the layout of a shown string's glyphs, by code or by
//! shaped run, and `show_text`.

use std::sync::Arc;

use super::PageRenderer;
use super::text_census::GlyphCensus;
use super::text_context::{Following, TextTail, neighbours as neighbours_at};
use super::text_glyphs::{GlyphSink, TextModeOps};
use super::text_ops::{
    fit_to_advance, glyph_advance, is_singular, text_rendering_matrix, text_to_device,
    translate_text_x, translate_text_y,
};
use super::text_shaping::{
    CONTEXT_CODES, CodeText, Drawn, Neighbours, ShapedRun, shaped_runs, shapes, takes_context,
};
use super::text_stroke::GlyphStroke;
use crate::renderer::font_cache::{ActiveFont, PageFace};
use crate::renderer::text::TextMatrix;
use crate::resources::ResourceKey;
use crate::resources::cmap::DecodedCode;
use crate::resources::font::{CidEncoding, FontDescriptor, Type3Data};

/// The drawing glyph of each of the codes `decoded` shown with `face`, of a
/// composite font when `cid_enc` is its encoding (see
/// [`PageFace::glyph_for`]).
fn drawn_by<'f>(
    face: Option<&'f PageFace>,
    cid_enc: Option<&CidEncoding>,
    decoded: &[DecodedCode],
) -> Vec<Option<Drawn<'f>>> {
    decoded
        .iter()
        .map(|d| face.and_then(|face| face.glyph_for(cid_enc, d.code, d.value)))
        .collect()
}

/// The advance of the drawn glyph `(face, selector, by_gid)` in its face,
/// in em.
fn drawn_advance((face, selector, by_gid): Drawn<'_>) -> Option<f64> {
    if by_gid {
        face.glyph_advance_by_gid(selector)
    } else {
        face.glyph_advance(selector)
    }
}

/// `advance` in thousandths of the em, zero when unknown.
fn thousandths(advance: Option<f64>) -> i32 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "advances are a few em at most; thousandths fit i32"
    )]
    let width = advance.map_or(0.0, |w| (w * 1000.0).round()) as i32;
    width
}

/// The document's advance of the shown code `d` in thousandths of the em,
/// `None` for a simple font without a width for it.
fn document_width(descriptor: &FontDescriptor, d: &DecodedCode) -> Option<i32> {
    descriptor.cid_encoding.as_ref().map_or_else(
        || descriptor.width_for_code(d.code),
        |enc| Some(enc.width_for_cid(d.value)),
    )
}

/// What maps a text-space position of a string's glyphs to the page's
/// device space: the CTM and the page height, and the text state's font
/// size, horizontal scaling (a fraction) and rise.
#[derive(Clone, Copy)]
struct TextFrame {
    ctm: [f64; 6],
    page_h: u32,
    font_size: f64,
    horiz_scaling: f64,
    rise: f64,
}

impl TextFrame {
    /// The em of the horizontal advances, `Tfs × Th`, in text space.
    fn em(&self) -> f64 {
        self.font_size * self.horiz_scaling
    }

    /// The page's device position of the text-space point `(tx, ty)` of
    /// text matrix `tm`.
    fn page(&self, tm: &TextMatrix, tx: f64, ty: f64) -> (f64, f64) {
        text_to_device(&self.ctm, tm, tx, ty, self.page_h)
    }
}

/// The advance of each of the codes `decoded`, drawn as `drawn` by `face`
/// of the font `descriptor`, under `frame` and the character and word
/// spacing `spacing` (PDF §9.4.4): horizontally `(w0 × Tfs + Tc + Tw) ×
/// Th`; vertically `w1 × Tfs + Tc + Tw`, along the text matrix's y axis.
/// Word spacing applies to the byte 32 when it is a one-byte code, never
/// to a multi-byte code (PDF §9.3.3).  A simple font's code without a
/// width advances by its drawn glyph's (PDF §9.2.4), and the pen advances
/// for a glyph that draws nothing.
fn code_advances(
    descriptor: &FontDescriptor,
    face: Option<&PageFace>,
    decoded: &[DecodedCode],
    drawn: &[Option<Drawn<'_>>],
    frame: &TextFrame,
    (char_spacing, word_spacing): (f64, f64),
) -> Vec<f64> {
    let vertical_enc = descriptor.cid_encoding.as_ref().filter(|enc| enc.vertical);
    decoded
        .iter()
        .zip(drawn)
        .map(|(d, &drawn)| {
            let extra = if d.code == 32 && d.len == 1 {
                word_spacing
            } else {
                0.0
            };
            if let Some(enc) = vertical_enc {
                let w1 = f64::from(enc.vertical_metrics_for_cid(d.value).w1y) / 1000.0;
                return glyph_advance(w1, frame.font_size, char_spacing, extra, 1.0);
            }
            let advance_glyph = document_width(descriptor, d).map_or_else(
                || {
                    drawn
                        .map_or_else(
                            || face.and_then(|face| face.glyph_advance(d.code)),
                            drawn_advance,
                        )
                        .unwrap_or(0.0)
                },
                |w| f64::from(w) / 1000.0,
            );
            glyph_advance(
                advance_glyph,
                frame.font_size,
                char_spacing,
                extra,
                frame.horiz_scaling,
            )
        })
        .collect()
}

/// The characters of the strings shown next to a string, and where the
/// glyphs of the runs continuing into them lie.
struct ShapingContext {
    neighbours: Neighbours,
    /// The index of the code a run continuing the run before starts at,
    /// and how far past that code's pen position the glyphs of the run
    /// before end: from the string before at the start.
    drift: (usize, f64),
    /// How far past the string's end the string after starts.
    next_start: f64,
}

impl Default for ShapingContext {
    fn default() -> Self {
        Self {
            neighbours: Neighbours::default(),
            drift: (usize::MAX, 0.0),
            next_start: 0.0,
        }
    }
}

/// A shown string laid out from its codes: their drawing glyphs and
/// advances, and the frame that maps their glyphs to the window.
struct Layout<'a, 'f> {
    descriptor: &'a FontDescriptor,
    /// The encoding of a composite font that writes vertically
    /// (PDF §9.7.4.3).
    vertical: Option<&'a CidEncoding>,
    decoded: &'a [DecodedCode],
    drawn: &'a [Option<Drawn<'f>>],
    advances: &'a [f64],
    frame: TextFrame,
    /// A substitute's glyphs are fitted to the advances the document
    /// specifies: condensed when wider, centred when narrower.  An
    /// embedded program's glyphs are the document's own, and a standard
    /// family's substitute already has its advances.
    fit: bool,
}

impl Layout<'_, '_> {
    /// Add the glyphs of the string to `sink`, each code's at its pen
    /// position from the text matrix `tm` and each of `runs` from its first
    /// code's, in `context`, and count each code's glyph at its pen
    /// position in `census`.  Returns the text matrix after the last
    /// advance and the drift of a run continuing into the string after
    /// (see [`ShapingContext::drift`]).
    fn place(
        &self,
        mut tm: TextMatrix,
        runs: &[ShapedRun<'_>],
        context: &ShapingContext,
        sink: &mut GlyphSink,
        mut census: Option<&mut GlyphCensus>,
    ) -> (TextMatrix, (usize, f64)) {
        let mut drift = context.drift;
        let mut runs = runs.iter().peekable();
        // The text matrix at the start of the run being shown, and the
        // run's advance so far in text space.
        let mut run_origin = tm;
        let mut run_width = 0.0;
        for (i, (d, &drawn)) in self.decoded.iter().zip(self.drawn).enumerate() {
            if let Some(census) = census.as_deref_mut() {
                census.glyph(&tm, drawn.is_some());
            }
            while runs.next_if(|run| run.end <= i).is_some() {}
            let run = runs.peek().filter(|run| run.start <= i);
            if run.is_some_and(|run| run.start == i) {
                run_origin = tm;
                run_width = 0.0;
            }
            if run.is_none()
                && let Some(drawn) = drawn
            {
                self.place_code(d, drawn, &tm, sink);
            }
            let advance = self.advances[i];
            if self.vertical.is_some() {
                debug_assert!(run.is_none(), "vertical text is not shaped");
                tm = translate_text_y(tm, advance);
                continue;
            }
            tm = translate_text_x(tm, advance);
            let Some(run) = run else { continue };
            run_width += advance;
            if i + 1 == run.end {
                self.place_run(
                    run,
                    &run_origin,
                    run_width,
                    context.next_start,
                    &mut drift,
                    sink,
                );
            }
        }
        (tm, drift)
    }

    /// Add the glyph `drawn` of the code `d`, drawn on its own at the text
    /// matrix `tm`, to `sink`.
    fn place_code(&self, d: &DecodedCode, drawn: Drawn<'_>, tm: &TextMatrix, sink: &mut GlyphSink) {
        let f = &self.frame;
        let (x_scale, centre) = match document_width(self.descriptor, d) {
            Some(width) if self.fit => fit_to_advance(width, thousandths(drawn_advance(drawn))),
            _ => (1.0, 0.0),
        };
        // In vertical writing mode the glyph's vertical origin sits at the
        // pen: its horizontal origin is displaced by the position vector
        // −v, in glyph units scaled by the font size (PDF §9.7.4.3).
        let (tx, ty) = self.vertical.map_or_else(
            || (centre / 1000.0 * f.font_size * f.horiz_scaling, f.rise),
            |enc| {
                let v = enc.vertical_metrics_for_cid(d.value);
                let tx = -f64::from(v.vx) / 1000.0 * f.font_size * f.horiz_scaling;
                (tx, (-f64::from(v.vy) / 1000.0).mul_add(f.font_size, f.rise))
            },
        );
        sink.push(drawn, d.code, f.page(tm, tx, ty), x_scale);
    }

    /// Add the glyphs of `run`, whose codes start at the text matrix
    /// `origin` and advance `width` in text space, to `sink`.  A run's
    /// glyphs are fitted to the advance of its codes as one glyph is to its
    /// code's: condensed when wider and centred when narrower, under `fit`;
    /// else at their own advances from the run's start.  `drift` is the
    /// drift of a run continuing from the run before, replaced by this
    /// run's when it continues into the code after; `next_start` is how far
    /// past the string's end the string after starts.
    fn place_run(
        &self,
        run: &ShapedRun<'_>,
        origin: &TextMatrix,
        width: f64,
        next_start: f64,
        drift: &mut (usize, f64),
        sink: &mut GlyphSink,
    ) {
        let f = &self.frame;
        let em = f.em();
        // Marks drawn over the first letter of the string after, at their
        // displacements from its origin.
        if run.over_next {
            for g in &run.glyphs {
                let tx = g.offset.0.mul_add(em, width + next_start);
                let ty = g.offset.1.mul_add(f.font_size, f.rise);
                sink.push(
                    (run.face, g.selector, true),
                    g.selector,
                    f.page(origin, tx, ty),
                    1.0,
                );
            }
            return;
        }
        let natural: f64 = run.glyphs.iter().map(|g| g.advance).sum::<f64>() * em;
        let fitted = self.fit && width > 0.0 && natural > 0.0;
        let scale = if fitted {
            (width / natural).min(1.0)
        } else {
            1.0
        };
        // A run that continues the run before it starts where that run's
        // glyphs end, offset from its own start as they are from that run's
        // end; one that continues into the code after it starts at its own
        // start.
        let mut pen = match run.joins {
            (true, _) if drift.0 == run.start => drift.1,
            (true, _) | (false, true) => 0.0,
            _ if fitted => natural.mul_add(-scale, width) / 2.0,
            _ => 0.0,
        };
        for g in &run.glyphs {
            let tx = (g.offset.0 * em).mul_add(scale, pen);
            let ty = g.offset.1.mul_add(f.font_size, f.rise);
            sink.push(
                (run.face, g.selector, true),
                g.selector,
                f.page(origin, tx, ty),
                scale,
            );
            pen = (g.advance * em).mul_add(scale, pen);
        }
        if run.joins.1 {
            *drift = (run.end, pen - width);
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
        let (sink, tm, new_tail) = self.lay_out(
            &font,
            bytes,
            following,
            tail,
            (mode, glyph_stroke),
            census.as_mut(),
        );
        self.record_glyphs(census);
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
                census.as_mut(),
            );
        });
        self.record_glyphs(census);
    }

    /// Lay out the glyphs of `bytes`, shown with `font` in the current text
    /// state, after the string whose end is `tail` and before `following`,
    /// into a sink for `mode` and the cached stroke `glyph_stroke`, counting
    /// each code's glyph in `census`.  Returns the sink, the text matrix
    /// after the string and, when the string is shaped, its end.
    fn lay_out(
        &mut self,
        font: &ActiveFont,
        bytes: &[u8],
        following: Following<'_, '_>,
        tail: Option<TextTail>,
        (mode, glyph_stroke): (TextModeOps, Option<GlyphStroke>),
        census: Option<&mut GlyphCensus>,
    ) -> (GlyphSink, TextMatrix, Option<TextTail>) {
        let gs = self.gstate.current();
        let ts = &gs.text;
        // `Tz` is a percentage; `Tz 0` collapses the glyphs and the
        // horizontal advances to zero width.
        let frame = TextFrame {
            ctm: gs.ctm,
            page_h: self.page_height,
            font_size: ts.font_size,
            horiz_scaling: ts.horiz_scaling / 100.0,
            rise: ts.rise,
        };
        let (spacing, tm) = ((ts.char_spacing, ts.word_spacing), ts.text_matrix);
        let pattern_fill = mode.fill && gs.fill_pattern.is_some();
        let descriptor = &*font.descriptor;
        let cid_enc = descriptor.cid_encoding.as_ref();
        let document_fonts = Arc::clone(self.font_cache.document_fonts());
        // The glyph cache is shared, not copied, by the clone.
        let glyph_cache = self.font_cache.glyph_cache().clone();
        // The 2×2 text rendering matrix in device pixels (PDF §9.4.4) is
        // the same for every glyph of the string.  A singular one maps
        // every glyph to zero area, and a face that cannot load (below one
        // device pixel, an unreadable program) draws nothing; either way
        // the string draws no glyph and the pen still advances by the
        // document's widths.
        let trm = text_rendering_matrix(&tm, &frame.ctm, frame.font_size, frame.horiz_scaling);
        let mut face = if is_singular(&trm) {
            None
        } else {
            let font_name = &self.gstate.current().text.font_name;
            self.font_cache
                .get_or_load_showing(font, font_name, trm, bytes)
        };
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
        let decoded: Vec<DecodedCode> = CidEncoding::shown_codes(cid_enc, bytes).collect();
        let mut drawn = drawn_by(face, cid_enc, &decoded);
        let advances = code_advances(descriptor, face, &decoded, &drawn, &frame, spacing);

        // Runs are shaped in horizontal writing only, between the strings
        // shown next to them on their baseline.
        let shaped = face.is_some_and(|f| shapes(descriptor, f));
        let texts: Vec<CodeText> = match face {
            Some(face) if shaped => decoded.iter().map(|d| face.text(d.code, d.value)).collect(),
            _ => Vec::new(),
        };
        let em = frame.em();
        let mut context = ShapingContext::default();
        if shaped {
            if let Some(tail) =
                tail.filter(|tail| neighbours_at(&tail.end, &tm, frame.rise - tail.rise, tail.em))
            {
                context.neighbours.before = tail.texts;
                context.drift = (0, tail.drift);
            }
            if texts.last().is_some_and(|&text| takes_context(text)) {
                let end = advances
                    .iter()
                    .fold(tm, |tm, &advance| translate_text_x(tm, advance));
                if let Some(next) = self.next_string(end, em, following) {
                    context.neighbours.after = next.texts;
                    context.next_start = next.start;
                }
                // Reading ahead may load faces: the face is borrowed anew.
                let font_name = &self.gstate.current().text.font_name;
                face = self.font_cache.get_or_load(font, font_name, trm);
                drawn = drawn_by(face, cid_enc, &decoded);
            }
        }
        let runs = match face {
            Some(face) if shaped => shaped_runs(face, &texts, &drawn, &context.neighbours),
            _ => Vec::new(),
        };
        let vertical = cid_enc.filter(|enc| enc.vertical);
        let layout = Layout {
            descriptor,
            vertical,
            decoded: &decoded,
            drawn: &drawn,
            advances: &advances,
            frame,
            fit: face.is_some_and(|f| f.substitute)
                && !descriptor.substitute.standard
                && vertical.is_none(),
        };
        let (end, drift) = layout.place(tm, &runs, &context, &mut sink, census);
        let tail = shaped.then(|| {
            let mut texts: Vec<CodeText> =
                context.neighbours.before.into_iter().chain(texts).collect();
            let _ = texts.drain(..texts.len().saturating_sub(CONTEXT_CODES));
            TextTail {
                texts,
                end,
                rise: frame.rise,
                em,
                drift: if drift.0 == decoded.len() {
                    drift.1
                } else {
                    0.0
                },
            }
        });
        (sink, end, tail)
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
