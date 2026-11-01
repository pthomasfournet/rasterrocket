//! Appearances synthesised for free text annotations without a normal
//! appearance (ISO 32000-2 §12.5.6.6).
//!
//! The text box is `/Rect` moved in by `/RD`.  It is filled with `/C` and
//! outlined with the border of `/BS` (or `/Border`) in the text colour of
//! `/DA`.  The text, the rich text string `/RC` with its styles or else
//! `/Contents`, is set in the `/DA` font, size and colour as modified by
//! the default style string `/DS` and the styles of `/RC`, wrapped to the
//! box less a margin of twice the border width, aligned by `/Q` (or a
//! `text-align` style) and run from the top; it is clipped to that area.
//! `/Rotate`, a multiple of 90 degrees, turns the box and its text
//! counter-clockwise within the text box, its sides swapped for a quarter
//! turn.  A callout (`/CL`, two or three points) is stroked from its first
//! point, which carries the `/LE` line ending.  The `/DA` font comes from
//! the form's default resources; when they lack it, a standard face name
//! selects that face and any other name standard Helvetica.

use std::fmt::Write as _;

use pdf::{Dictionary, Document, Object};
use raster::types::BlendMode;

use super::content::{
    Border, cm, colour, colour_op, inset, n, numbers, opacity, rectangle, string, text_or_stream,
};
use super::line_ending::LineEnding;
use super::rich::{self, Base};
use super::text::{DefaultAppearance, decode_text_string};
use super::typeset::{AUTO_MULTILINE_SIZE, Paragraph, Style, Typesetter, write_block};
use super::widget::AcroForm;
use super::{IDENTITY, Synthesized};
use crate::resources::dict_ext::DictExt;

/// The default appearance of a free text annotation that has none and
/// whose form has none.
const DEFAULT_DA: &[u8] = b"/Helv 10 Tf 0 g";

/// A free text annotation, read from its dictionary.
pub(super) struct FreeText {
    da: DefaultAppearance,
    /// The styled text.
    paragraphs: Vec<Paragraph>,
    /// The fill colour operator of `/C`.
    fill: Option<String>,
    /// The fill colour operator of `/IC`, for a closed line ending.
    interior: Option<String>,
    border: Border,
    /// The text box: `/Rect` moved in by `/RD`.
    text_box: [f64; 4],
    /// `/Rotate`: 0, 90, 180 or 270 degrees counter-clockwise.
    rotation: u16,
    /// The normalised `/Rect`.
    rect: [f64; 4],
    /// The callout points.
    callout: Vec<(f64, f64)>,
    ending: Option<LineEnding>,
    opacity: f64,
}

impl FreeText {
    /// The free text annotation `annot` with the normalised rectangle
    /// `rect`, in the document whose interactive form is `form`.
    pub(super) fn of(
        doc: &Document,
        annot: &Dictionary,
        rect: [f64; 4],
        form: Option<&AcroForm>,
    ) -> Self {
        let da = string(doc, annot.get(b"DA"))
            .or_else(|| {
                form.and_then(AcroForm::default_appearance)
                    .map(<[u8]>::to_vec)
            })
            .unwrap_or_else(|| DEFAULT_DA.to_vec());
        let callout: Vec<(f64, f64)> = numbers(doc, annot.get(b"CL"))
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&p| p.into())
            .collect();
        let ending = match annot.get(b"LE") {
            Some(Object::Name(name)) => LineEnding::from_name(name),
            other => super::content::resolve_array(doc, other)
                .first()
                .and_then(Object::as_name)
                .and_then(LineEnding::from_name),
        };
        let da = DefaultAppearance::parse(&da);
        let style = Style::of(
            &da,
            if da.size > 0.0 {
                da.size
            } else {
                AUTO_MULTILINE_SIZE
            },
        );
        let base = Base {
            style: &style,
            quadding: annot.get_i64(b"Q").unwrap_or(0),
        };
        let text = |key: &[u8]| text_or_stream(doc, annot.get(key)).map(|t| decode_text_string(&t));
        let ds = text(b"DS");
        let paragraphs = text(b"RC")
            .map(|rc| rich::paragraphs(&rc, ds.as_deref(), &base))
            .filter(|p| has_text(p))
            .unwrap_or_else(|| {
                let (style, quadding) = rich::default_style(ds.as_deref(), &base);
                Paragraph::plain(&text(b"Contents").unwrap_or_default(), &style, quadding)
            });
        Self {
            paragraphs,
            fill: colour(doc, annot.get(b"C"), false),
            interior: colour(doc, annot.get(b"IC"), false),
            border: Border::of(doc, annot),
            text_box: inset(rect, &numbers(doc, annot.get(b"RD"))),
            rotation: annot.get_i64(b"Rotate").map_or(0, super::quarter_turns),
            rect,
            callout: if matches!(callout.len(), 2 | 3) {
                callout
            } else {
                Vec::new()
            },
            ending,
            opacity: opacity(doc, annot),
            da,
        }
    }

    /// The resource name of the font the text is set in.
    pub(super) fn font(&self) -> &[u8] {
        &self.da.font
    }

    /// The appearance, its text set by `ts`.
    pub(super) fn appearance(&self, ts: &mut dyn Typesetter) -> Synthesized {
        let mut s = String::new();
        let bw = self.border.width;
        let stroke = self
            .da
            .colour()
            .and_then(|c| colour_op(&c, true))
            .unwrap_or_else(|| "0 G".to_owned());
        let mut reach = 0.0f64;
        if bw > 0.0
            && let [first, second, rest @ ..] = self.callout.as_slice()
        {
            let _ = writeln!(s, "q\n{stroke}");
            self.border.write(&mut s);
            let _ = writeln!(s, "{} {} m", n(first.0), n(first.1));
            for (x, y) in std::iter::once(second).chain(rest) {
                let _ = writeln!(s, "{} {} l", n(*x), n(*y));
            }
            s.push_str("S\n");
            if let Some(ending) = self.ending {
                reach = ending.draw(&mut s, *first, *second, bw, self.interior.as_deref());
            }
            s.push_str("Q\n");
        }
        let (text_box, matrix) = super::rotated_box(self.text_box, self.rotation);
        let _ = writeln!(s, "q {}", cm(matrix));
        let frame = inset(text_box, &[bw / 2.0; 4]);
        if let Some(fill) = &self.fill {
            let _ = writeln!(s, "{fill}");
            rectangle(&mut s, frame);
            s.push_str("f\n");
        }
        if bw > 0.0 {
            let _ = writeln!(s, "{stroke}");
            self.border.write(&mut s);
            rectangle(&mut s, frame);
            s.push_str("S\n");
        }
        let area = inset(text_box, &[2.0 * bw; 4]);
        if has_text(&self.paragraphs) {
            s.push_str("q\n");
            rectangle(&mut s, area);
            s.push_str("W n\n");
            write_block(&mut s, &self.paragraphs, area, ts);
            s.push_str("Q\n");
        }
        s.push_str("Q\n");
        Synthesized {
            content: s.into_bytes(),
            bbox: inset(self.rect, &[-(bw + reach); 4]),
            opacity: self.opacity,
            blend: BlendMode::Normal,
            matrix: IDENTITY,
        }
    }
}

/// Whether any paragraph of `paragraphs` holds text.
fn has_text(paragraphs: &[Paragraph]) -> bool {
    paragraphs
        .iter()
        .any(|p| p.spans.iter().any(|(_, t)| !t.is_empty()))
}

#[cfg(test)]
mod tests {
    use super::FreeText;
    use crate::renderer::page::annotations::typeset::WinAnsiTypesetter;
    use crate::test_helpers::make_doc_with_object;

    /// The appearance content of the free text annotation `annot`, set by
    /// [`WinAnsiTypesetter`].
    fn content(annot: &str) -> String {
        let doc = make_doc_with_object(annot);
        let dict = doc.get_dictionary((2, 0)).expect("a dictionary");
        let rect = super::super::read_rect(&dict).expect("a rect");
        let ft = FreeText::of(&doc, &dict, rect, None);
        String::from_utf8(ft.appearance(&mut WinAnsiTypesetter).content).expect("ASCII")
    }

    #[test]
    fn the_box_is_filled_with_c_and_outlined_in_the_text_colour() {
        let c = content(
            "<< /Subtype /FreeText /Rect [100 50 150 75] /C [1 0 0] /Contents (Hello!) \
             /DA (0 0 1 rg /F1 8 Tf) >>",
        );
        assert!(c.starts_with("q 1 0 0 1 100 50 cm\n"), "{c}");
        assert!(c.contains("1 0 0 rg\n0.5 0.5 49 24 re\nf"), "{c}");
        assert!(c.contains("0 0 1 RG\n1 w\n0.5 0.5 49 24 re\nS"), "{c}");
        // The first baseline is one font size below the top less the margin.
        assert!(
            c.contains("BT /F1 8 Tf 0 0 1 rg\n2 15 Td <48656C6C6F21> Tj ET"),
            "{c}"
        );
    }

    #[test]
    fn rotate_turns_the_box_and_its_text_within_the_text_box() {
        let c = content(
            "<< /Subtype /FreeText /Rect [0 0 50 100] /Rotate 90 /DA (/Helv 10 Tf 0 g) \
             /Contents (x) >>",
        );
        assert!(c.starts_with("q 0 1 -1 0 50 0 cm\n"), "{c}");
        assert!(c.contains("0.5 0.5 99 49 re\nS"), "{c}");
        let c = content(
            "<< /Subtype /FreeText /Rect [0 0 50 100] /Rotate 45 /DA (/Helv 10 Tf 0 g) \
             /Contents (x) >>",
        );
        assert!(c.starts_with("q 1 0 0 1 0 0 cm\n"), "{c}");
    }

    #[test]
    fn rich_text_and_the_default_style_set_the_styles_of_the_text() {
        let c = content(
            "<< /Subtype /FreeText /Rect [0 0 200 100] /DA (/Helv 10 Tf 0 g) /Contents (plain) \
             /DS (font: 8pt Times; color:#0000FF) \
             /RC (<p>a <b style=\"font-size:12pt\">b</b></p>) >>",
        );
        assert!(c.contains("BT /TiRo 8 Tf 0 g 0 0 1 rg\n"), "{c}");
        assert!(c.contains("/TiBo 12 Tf 0 g 0 0 1 rg <62> Tj"), "{c}");
        assert!(!c.contains("<706C61696E>"), "the rich text is shown: {c}");
        let c = content(
            "<< /Subtype /FreeText /Rect [0 0 200 100] /DA (/Helv 10 Tf 0 g) /Contents (plain) \
             /DS (text-align:center; font-weight:bold) >>",
        );
        assert!(c.contains("BT /HeBo 10 Tf 0 g\n"), "{c}");
    }

    #[test]
    fn rich_text_stands_in_for_missing_contents() {
        let c = content(
            "<< /Subtype /FreeText /Rect [0 0 200 100] /DA (/Helv 10 Tf 0 g) \
             /RC (<body><p>One</p><p>Two</p></body>) >>",
        );
        assert!(
            c.contains("<4F6E65> Tj") && c.contains("<54776F> Tj"),
            "{c}"
        );
    }

    #[test]
    fn a_callout_is_stroked_from_its_first_point_with_its_ending() {
        let c = content(
            "<< /Subtype /FreeText /IT /FreeTextCallout /Rect [0 0 200 100] /RD [100 0 0 0] \
             /CL [10 10 50 50 100 50] /LE /OpenArrow /DA (/Helv 10 Tf 0 g) /Contents (x) >>",
        );
        assert!(c.contains("10 10 m\n50 50 l\n100 50 l\nS"), "{c}");
        assert!(
            c.contains("q 1 0 0 1 100 0 cm\n") && c.contains("0.5 0.5 99 99 re"),
            "the box is inset by /RD: {c}"
        );
        assert!(c.contains("cm\n-6 3.4641 m"), "{c}");
    }
}
