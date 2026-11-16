//! The layout of a shown string's glyphs: each code's at its pen position
//! and each shaped run's from its first code's, a substitute's glyphs
//! fitted to the advances the document specifies.
//!
//! Shaped runs that join one another form a chain: a run and the run that
//! starts at its end and takes context from it, in one string (a run
//! drawn by another face) or across neighbouring strings (see
//! [`super::text_context`]).  The glyphs of each run of a chain after the
//! first start where those of the run before end, displaced by the shift
//! of their string's start, and under a fit the chain is fitted as one
//! run: condensed by one scale to the advances of all its codes when its
//! glyphs are wider, centred in them when narrower.  A word split between
//! strings or faces is thereby drawn as the run of the whole word.

use super::text_census::GlyphCensus;
use super::text_glyphs::GlyphSink;
use super::text_ops::{
    fit_to_advance, glyph_advance, text_to_device, translate_text_x, translate_text_y,
};
use super::text_shaping::{Drawn, Neighbours, ShapedRun};
use super::trace;
use crate::renderer::font_cache::PageFace;
use crate::renderer::text::TextMatrix;
use crate::resources::cmap::DecodedCode;
use crate::resources::font::{CidEncoding, FontDescriptor};

/// The drawing glyph of each of the codes `decoded` shown with `face`, of a
/// composite font when `cid_enc` is its encoding (see
/// [`PageFace::glyph_for`]).
pub(super) fn drawn_by<'f>(
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
pub(super) struct TextFrame {
    pub ctm: [f64; 6],
    pub page_h: u32,
    pub font_size: f64,
    pub horiz_scaling: f64,
    pub rise: f64,
}

impl TextFrame {
    /// The em of the horizontal advances, `Tfs × Th`, in text space.
    pub(super) fn em(&self) -> f64 {
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
pub(super) fn code_advances(
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

/// Where the glyphs of a chain continue after one of its runs: `at`, the
/// index of the code its next run starts at (one past the string's last
/// code for a chain that continues into the string after), how far along
/// x in text space past that code's pen position its glyphs end, and the
/// scale of its glyphs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Continuation {
    pub at: usize,
    pub drift: f64,
    pub scale: f64,
}

/// The advance of the codes of runs of a chain and the natural advance of
/// their glyphs, along x in text space.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct ChainExtent {
    pub width: f64,
    pub natural: f64,
}

impl ChainExtent {
    /// The extent of `runs`, whose codes advance by `advances` (indexed as
    /// the runs' codes) and whose glyph advances are in em of `em`.
    pub(super) fn of(runs: &[ShapedRun<'_>], advances: &[f64], em: f64) -> Self {
        runs.iter().fold(Self::default(), |sum, run| Self {
            width: sum.width + advances[run.start..run.end].iter().sum::<f64>(),
            natural: run
                .glyphs
                .iter()
                .map(|g| g.advance)
                .sum::<f64>()
                .mul_add(em, sum.natural),
        })
    }

    /// The extent of `self` and then `other`, when it is known.
    pub(super) fn then(self, other: Option<Self>) -> Self {
        other.map_or(self, |other| Self {
            width: self.width + other.width,
            natural: self.natural + other.natural,
        })
    }

    /// The scale of a chain of this extent and the offset of its first
    /// glyph from its first code's pen: under `fit`, condensed to its
    /// codes' advances when wider and centred in them when narrower; else
    /// at its own advances from its start.
    fn fitted(self, fit: bool) -> (f64, f64) {
        if fit && self.width > 0.0 && self.natural > 0.0 {
            let scale = (self.width / self.natural).min(1.0);
            (scale, self.natural.mul_add(-scale, self.width) / 2.0)
        } else {
            (1.0, 0.0)
        }
    }
}

/// Whether the glyphs of the run after `run` can continue those of `run`:
/// it joins the code after it.
const fn continues_forward(run: &ShapedRun<'_>) -> bool {
    run.joins.1
}

/// Whether `run` continues a chain whose next run starts at code `at`.
const fn continues(run: &ShapedRun<'_>, at: usize) -> bool {
    run.start == at && run.joins.0
}

/// The end of the chain that starts with `runs[first]` in a string of
/// `len` codes: the index past its last run in `runs`, and whether it
/// continues into the string after.
pub(super) fn chain_end(runs: &[ShapedRun<'_>], first: usize, len: usize) -> (usize, bool) {
    let mut last = first;
    while continues_forward(&runs[last])
        && let Some(next) = runs.get(last + 1)
        && continues(next, runs[last].end)
    {
        last += 1;
    }
    let past = continues_forward(&runs[last]) && runs[last].end == len;
    (last + 1, past)
}

/// The extent of the runs of a string of `len` codes advancing by
/// `advances` that continue a chain from the string before, with glyph
/// advances in em of `em`, and whether they continue into the string
/// after; `None` when its first run does not continue the chain.
pub(super) fn leading_chain(
    runs: &[ShapedRun<'_>],
    advances: &[f64],
    em: f64,
    len: usize,
) -> Option<(ChainExtent, bool)> {
    if !runs.first().is_some_and(|run| continues(run, 0)) {
        return None;
    }
    let (end, past) = chain_end(runs, 0, len);
    Some((ChainExtent::of(&runs[..end], advances, em), past))
}

/// The characters of the strings shown next to a string, and the chain
/// its runs continue and continue into.
#[derive(Default)]
pub(super) struct ShapingContext {
    pub neighbours: Neighbours,
    /// The chain of the string before, continuing at the string's first
    /// code.
    pub continuation: Option<Continuation>,
    /// How far past the string's end the string after starts.
    pub next_start: f64,
    /// The extent of the runs of the strings after that continue a chain
    /// into which the string's last run continues; `None` when they are
    /// not fitted or not known.
    pub ahead: Option<ChainExtent>,
}

/// A shown string laid out from its codes: their drawing glyphs and
/// advances, and the frame that maps their glyphs to the window.
pub(super) struct Layout<'a, 'f> {
    pub descriptor: &'a FontDescriptor,
    /// The encoding of a composite font that writes vertically
    /// (PDF §9.7.4.3).
    pub vertical: Option<&'a CidEncoding>,
    pub decoded: &'a [DecodedCode],
    pub drawn: &'a [Option<Drawn<'f>>],
    pub advances: &'a [f64],
    pub frame: TextFrame,
    /// A substitute's glyphs are fitted to the advances the document
    /// specifies: condensed when wider, centred when narrower.  An
    /// embedded program's glyphs are the document's own, and a standard
    /// family's substitute already has its advances.
    pub fit: bool,
}

impl Layout<'_, '_> {
    /// Add the glyphs of the string to `sink`, each code's at its pen
    /// position from the text matrix `tm` and each of `runs` from its first
    /// code's, in `context`, and count each code's glyph at its pen
    /// position in `census` and print it to `glyph_trace`.  Returns the
    /// text matrix after the last advance and the chain that continues
    /// into the string after.
    pub(super) fn place(
        &self,
        mut tm: TextMatrix,
        runs: &[ShapedRun<'_>],
        context: &ShapingContext,
        sink: &mut GlyphSink,
        (mut census, mut glyph_trace): (Option<&mut GlyphCensus>, Option<&mut String>),
    ) -> (TextMatrix, Option<Continuation>) {
        let mut continuation = context.continuation;
        // The run being shown or the next, the text matrix at its start,
        // and its advance so far in text space.
        let mut current = 0;
        let mut run_origin = tm;
        let mut run_width = 0.0;
        for (i, (d, &drawn)) in self.decoded.iter().zip(self.drawn).enumerate() {
            if let Some(census) = census.as_deref_mut() {
                census.glyph(&tm, drawn.is_some());
            }
            if let Some(out) = glyph_trace.as_deref_mut() {
                let f = &self.frame;
                trace::push_glyph(
                    out,
                    d.code,
                    drawn.map(|(_, gid, by_gid)| (gid, by_gid)),
                    f.page(&tm, 0.0, f.rise),
                );
            }
            while runs.get(current).is_some_and(|run| run.end <= i) {
                current += 1;
            }
            let run = runs.get(current).filter(|run| run.start <= i);
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
                    (runs, current),
                    &run_origin,
                    run_width,
                    context,
                    &mut continuation,
                    sink,
                );
            }
        }
        let end = self.decoded.len();
        (tm, continuation.filter(|c| c.at == end))
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

    /// Add the glyphs of `runs[k]`, whose codes start at the text matrix
    /// `origin` and advance `width` in text space, to `sink`, in `context`.
    /// A run that continues the chain `continuation` starts where its
    /// glyphs end, at its scale; one that starts a chain is fitted with the
    /// rest of the chain (see the module documentation).  `continuation`
    /// is replaced by where the run's chain continues after it.
    fn place_run(
        &self,
        (runs, k): (&[ShapedRun<'_>], usize),
        origin: &TextMatrix,
        width: f64,
        context: &ShapingContext,
        continuation: &mut Option<Continuation>,
        sink: &mut GlyphSink,
    ) {
        let run = &runs[k];
        let f = &self.frame;
        let em = f.em();
        // Marks drawn over the first letter of the string after, at their
        // displacements from its origin.
        if run.over_next {
            for g in &run.glyphs {
                let tx = g.offset.0.mul_add(em, width + context.next_start);
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
        let (scale, mut pen) = continuation.filter(|c| continues(run, c.at)).map_or_else(
            || {
                let (end, past) = chain_end(runs, k, self.decoded.len());
                ChainExtent::of(&runs[k..end], self.advances, em)
                    .then(context.ahead.filter(|_| past))
                    .fitted(self.fit)
            },
            |c| (if self.fit { c.scale } else { 1.0 }, c.drift),
        );
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
        *continuation = continues_forward(run).then_some(Continuation {
            at: run.end,
            drift: pen - width,
            scale,
        });
    }
}
