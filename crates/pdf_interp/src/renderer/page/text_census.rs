//! The census of shown glyphs: each glyph a text-showing operator shows is
//! counted under its text rendering mode when it can mark the page, else
//! under the reason it cannot (see [`GlyphCounts`]).

use super::PageRenderer;
use super::diagnostics::GlyphCounts;
use super::text_glyphs::TextModeOps;
use super::text_ops::{is_singular, text_rendering_matrix, text_to_device};
use crate::renderer::font_cache::trm_pixel_size_valid;

/// The counter of [`GlyphCounts`] a glyph of a text rendering mode goes to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    Fill,
    Stroke,
    FillStroke,
    Invisible,
    ClipPainted,
    ClipOnly,
}

impl Mode {
    /// The counter of the mode of operations `ops`.
    const fn of(ops: TextModeOps) -> Self {
        match (ops.fill, ops.stroke, ops.clip) {
            (true, false, false) => Self::Fill,
            (false, true, false) => Self::Stroke,
            (true, true, false) => Self::FillStroke,
            (false, false, false) => Self::Invisible,
            (false, false, true) => Self::ClipOnly,
            (_, _, true) => Self::ClipPainted,
        }
    }

    /// Whether a glyph of the mode is painted.
    const fn paints(self) -> bool {
        !matches!(self, Self::Invisible | Self::ClipOnly)
    }

    /// The counter of the mode in `counts`.
    const fn counter(self, counts: &mut GlyphCounts) -> &mut u32 {
        match self {
            Self::Fill => &mut counts.fill,
            Self::Stroke => &mut counts.stroke,
            Self::FillStroke => &mut counts.fill_stroke,
            Self::Invisible => &mut counts.invisible,
            Self::ClipPainted => &mut counts.clip_painted,
            Self::ClipOnly => &mut counts.clip_only,
        }
    }
}

/// How the glyphs of one shown string are counted.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    /// Each under its mode, which paints nothing.
    Unpainted(Mode),
    /// Each under [`GlyphCounts::zero_size`].
    ZeroSize,
    /// Each under [`GlyphCounts::unrendered`] without a glyph, under
    /// [`GlyphCounts::off_page`] wholly off the page, else under
    /// [`GlyphCounts::transparent`] when `transparent`, else under `mode`.
    Painted { mode: Mode, transparent: bool },
}

/// Whether a renderer adds the glyphs it shows to its page's
/// [`GlyphCounts`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Census {
    /// Shown glyphs are counted.
    Counted,
    /// Shown glyphs are not counted: the renderer of a pattern cell or
    /// Type 3 glyph, a window after a page's first, or content that marks
    /// the page only through another object.
    Uncounted,
}

/// The census of the glyphs of one shown string: where its glyphs lie
/// on the page and how they are counted.
pub(super) struct GlyphCensus {
    /// The CTM the string is shown under.
    ctm: [f64; 6],
    /// The page's size in device pixels.
    page: (u32, u32),
    /// The em in text space, horizontally (`Tfs × Th`) and vertically
    /// (`Tfs`), and the text rise.
    em: (f64, f64),
    rise: f64,
    class: Class,
    counts: GlyphCounts,
}

impl GlyphCensus {
    /// Count the glyph at the pen of text matrix `tm`; `drawn` when the
    /// font has a glyph for its code.
    pub(super) fn glyph(&mut self, tm: &[f64; 6], drawn: bool) {
        let counter = match self.class {
            Class::ZeroSize => &mut self.counts.zero_size,
            Class::Painted { .. } if !drawn => &mut self.counts.unrendered,
            Class::Painted { .. } if !self.meets_page(tm) => &mut self.counts.off_page,
            Class::Painted {
                transparent: true, ..
            } => &mut self.counts.transparent,
            Class::Unpainted(mode) | Class::Painted { mode, .. } => mode.counter(&mut self.counts),
        };
        *counter = counter.saturating_add(1);
    }

    /// Count `n` glyphs shown without a font: under the mode when it
    /// paints nothing, else as unrendered, whatever their size.
    pub(super) fn fontless(&mut self, n: usize) {
        let counter = match self.class {
            Class::Unpainted(mode) => mode.counter(&mut self.counts),
            Class::ZeroSize | Class::Painted { .. } => &mut self.counts.unrendered,
        };
        *counter = counter.saturating_add(u32::try_from(n).unwrap_or(u32::MAX));
    }

    /// Whether the em square of the glyph at the pen of `tm`, widened by
    /// one em on each side, meets the page.  A box with a non-finite
    /// corner on every side does not.
    fn meets_page(&self, tm: &[f64; 6]) -> bool {
        let (ex, ey) = self.em;
        let (mut x0, mut y0, mut x1, mut y1) = (
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        );
        for (u, v) in [(-1.0_f64, -1.0_f64), (2.0, -1.0), (-1.0, 2.0), (2.0, 2.0)] {
            let (x, y) =
                text_to_device(&self.ctm, tm, u * ex, v.mul_add(ey, self.rise), self.page.1);
            (x0, x1) = (x0.min(x), x1.max(x));
            (y0, y1) = (y0.min(y), y1.max(y));
        }
        let (w, h) = (f64::from(self.page.0), f64::from(self.page.1));
        x1 > 0.0 && x0 < w && y1 > 0.0 && y0 < h
    }
}

impl PageRenderer<'_> {
    /// The census of a string shown in the current text state with the
    /// operations `ops`, of a Type 3 font when `type3`; `None` when the
    /// renderer does not count glyphs.
    ///
    /// A Type 3 glyph's description paints in every mode but 3, under the
    /// fill opacity, and at any size its matrix does not collapse.
    pub(super) fn glyph_census(&self, ops: TextModeOps, type3: bool) -> Option<GlyphCensus> {
        if self.census == Census::Uncounted {
            return None;
        }
        let gs = self.gstate.current();
        let ts = &gs.text;
        let horiz_scaling = ts.horiz_scaling / 100.0;
        let mut mode = Mode::of(ops);
        if type3 && mode == Mode::ClipOnly {
            mode = Mode::ClipPainted;
        }
        let trm = text_rendering_matrix(&ts.text_matrix, &gs.ctm, ts.font_size, horiz_scaling);
        let class = if !mode.paints() {
            Class::Unpainted(mode)
        } else if is_singular(&trm) || (!type3 && !trm_pixel_size_valid(trm)) {
            Class::ZeroSize
        } else {
            let marks = |alpha: u8, nothing: bool| alpha > 0 && !nothing;
            let fill = marks(
                gs.fill_alpha,
                !type3 && gs.fill_color_space.paints_nothing(),
            );
            let stroke = marks(gs.stroke_alpha, gs.stroke_color_space.paints_nothing());
            let transparent = if type3 {
                !fill
            } else {
                !(ops.fill && fill || ops.stroke && stroke)
            };
            Class::Painted { mode, transparent }
        };
        Some(GlyphCensus {
            ctm: gs.ctm,
            page: (self.page_width, self.page_height),
            em: (ts.font_size * horiz_scaling, ts.font_size),
            rise: ts.rise,
            class,
            counts: GlyphCounts::default(),
        })
    }

    /// Add the glyphs `census` counted to the page's.
    pub(super) const fn record_glyphs(&mut self, census: Option<GlyphCensus>) {
        if let Some(census) = census {
            self.diag.glyphs.add(&census.counts);
        }
    }

    /// Run `f` with glyphs not counted: content that marks the page only
    /// through another object, a soft mask's group or a Type 3 glyph's
    /// description.
    pub(super) fn without_glyph_census<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        let census = std::mem::replace(&mut self.census, Census::Uncounted);
        let out = f(self);
        self.census = census;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::super::PageRenderer;
    use super::GlyphCounts;

    /// The glyph counts of `content` run on a 64 × 32 px page whose `/F1`
    /// is Helvetica.  The graphics state `/Clear` sets the fill and stroke
    /// opacities to 0, `/Masked` sets a soft mask whose group shows two
    /// glyphs, and `/None` is a Separation space of colorant `None`.
    fn counts(content: &[u8]) -> GlyphCounts {
        let mask = "BT /F1 12 Tf 2 2 Td (MM) Tj ET";
        let doc = crate::test_helpers::make_doc_with_objects(&[
            (
                4,
                "<< /Type /Page /Resources << /Font << /F1 5 0 R >> \
                 /ColorSpace << /None [/Separation /None /DeviceGray << /FunctionType 2 \
                 /Domain [0 1] /C0 [1] /C1 [0] /N 1 >>] >> \
                 /ExtGState << /Clear << /ca 0 /CA 0 >> \
                 /Masked << /SMask << /S /Luminosity /G 6 0 R >> >> >> >> >>"
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
                     /Resources << /Font << /F1 5 0 R >> >> \
                     /Group << /S /Transparency /CS /DeviceGray >> /Length {} >>\n\
                     stream\n{mask}\nendstream",
                    mask.len()
                ),
            ),
        ]);
        let mut renderer = PageRenderer::new(64, 32, &doc, (4, 0));
        renderer.execute(&crate::content::parse(content));
        renderer.finish().1.glyphs
    }

    /// Glyphs are counted under the counter of their mode, one per code.
    #[test]
    fn each_mode_has_its_counter() {
        let shown =
            |mode: u8| counts(format!("BT /F1 12 Tf {mode} Tr 4 8 Td (AB) Tj ET").as_bytes());
        let by_mode = [
            GlyphCounts {
                fill: 2,
                ..GlyphCounts::default()
            },
            GlyphCounts {
                stroke: 2,
                ..GlyphCounts::default()
            },
            GlyphCounts {
                fill_stroke: 2,
                ..GlyphCounts::default()
            },
            GlyphCounts {
                invisible: 2,
                ..GlyphCounts::default()
            },
            GlyphCounts {
                clip_painted: 2,
                ..GlyphCounts::default()
            },
            GlyphCounts {
                clip_painted: 2,
                ..GlyphCounts::default()
            },
            GlyphCounts {
                clip_painted: 2,
                ..GlyphCounts::default()
            },
            GlyphCounts {
                clip_only: 2,
                ..GlyphCounts::default()
            },
        ];
        for (mode, want) in (0..).zip(by_mode) {
            assert_eq!(shown(mode), want, "mode {mode}");
        }
        let visible = [2, 2, 2, 0, 2, 2, 2, 0];
        for (mode, want) in (0..).zip(visible) {
            assert_eq!(shown(mode).visible(), want, "mode {mode}");
        }
    }

    /// A page of visible text and an invisible layer over it counts both.
    #[test]
    fn a_mixed_page_counts_each_layer() {
        let c = counts(b"BT /F1 12 Tf 4 8 Td (Title) Tj 3 Tr 0 10 Td [(Te) 20 (xt)] TJ ET");
        assert_eq!((c.fill, c.invisible, c.visible()), (5, 4, 5), "{c:?}");
    }

    /// A glyph wholly outside the page box cannot mark it; one straddling
    /// its edge can.
    #[test]
    fn a_glyph_off_the_page_is_not_visible() {
        let c = counts(b"BT /F1 12 Tf 200 8 Td (AB) Tj ET BT /F1 12 Tf 4 -100 Td (C) Tj ET");
        assert_eq!((c.off_page, c.visible()), (3, 0), "{c:?}");
        let edge = counts(b"BT /F1 12 Tf 58 8 Td (A) Tj ET");
        assert_eq!((edge.off_page, edge.fill), (0, 1), "{edge:?}");
        let rotated = counts(b"q 0 1 -1 0 0 0 cm BT /F1 12 Tf 4 40 Td (A) Tj ET Q");
        assert_eq!(rotated.off_page, 1, "{rotated:?}");
    }

    /// A glyph of zero size, or below a pixel, draws nothing.
    #[test]
    fn a_glyph_without_extent_is_not_visible() {
        for content in [
            &b"BT /F1 0 Tf 4 8 Td (AB) Tj ET"[..],
            b"BT /F1 12 Tf 0 Tz 4 8 Td (AB) Tj ET",
            b"BT /F1 0.5 Tf 4 8 Td (AB) Tj ET",
            b"q 0 0 0 0 0 0 cm BT /F1 12 Tf (AB) Tj ET Q",
        ] {
            let c = counts(content);
            assert_eq!((c.zero_size, c.visible()), (2, 0), "{c:?}");
        }
        assert_eq!(counts(b"BT /F1 0 Tf 3 Tr (AB) Tj ET").invisible, 2);
    }

    /// A paint of opacity 0, or in a space of colorant `None`, marks
    /// nothing; a mode with another paint that marks is visible.
    #[test]
    fn a_transparent_glyph_is_not_visible() {
        let c = counts(b"/Clear gs BT /F1 12 Tf 4 8 Td (AB) Tj ET");
        assert_eq!((c.transparent, c.visible()), (2, 0), "{c:?}");
        let c = counts(b"/None cs 1 scn BT /F1 12 Tf 4 8 Td (AB) Tj ET");
        assert_eq!((c.transparent, c.visible()), (2, 0), "{c:?}");
        let c = counts(b"/None cs 1 scn BT /F1 12 Tf 2 Tr 4 8 Td (AB) Tj ET");
        assert_eq!(c.fill_stroke, 2, "{c:?}");
    }

    /// Codes shown before a font is selected render nothing.
    #[test]
    fn a_string_without_a_font_is_unrendered() {
        let c = counts(b"BT 4 8 Td (AB) Tj ET");
        assert_eq!((c.unrendered, c.visible()), (2, 0), "{c:?}");
    }

    /// The text of a soft mask's group marks the page only through the
    /// mask and is not counted.
    #[test]
    fn the_text_of_a_soft_mask_is_not_counted() {
        let c = counts(b"/Masked gs BT /F1 12 Tf 4 8 Td (A) Tj ET");
        assert_eq!(
            c,
            GlyphCounts {
                fill: 1,
                ..GlyphCounts::default()
            }
        );
    }
}
