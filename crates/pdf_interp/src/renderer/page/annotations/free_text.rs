//! Appearances synthesised for free text annotations without a normal
//! appearance (ISO 32000-2 §12.5.6.6).
//!
//! The text box is `/Rect` moved in by `/RD`.  It is filled with `/C` and
//! outlined with the border of `/BS` (or `/Border`) in the text colour of
//! `/DA`.  The text, `/Contents` or else the plain text of the rich text
//! string `/RC`, is set in the `/DA` font, size and colour, wrapped to the
//! box less a margin of twice the border width, aligned by `/Q` and run
//! from the top; it is clipped to that area.  A callout (`/CL`, two or
//! three points) is stroked from its first point, which carries the `/LE`
//! line ending.  The `/DA` font comes from the form's default resources,
//! standard Helvetica when they lack it.

use std::fmt::Write as _;

use pdf::{Dictionary, Document, Object};
use raster::types::BlendMode;

use super::content::{
    Border, colour, colour_op, inset, n, number, numbers, rectangle, string, text_or_stream,
};
use super::line_ending::LineEnding;
use super::text::{DefaultAppearance, decode_text_string, plain_text};
use super::typeset::{AUTO_MULTILINE_SIZE, TextBlock, Typesetter};
use super::widget::AcroForm;
use super::{IDENTITY, Synthesized};
use crate::resources::dict_ext::DictExt;

/// The default appearance of a free text annotation that has none and
/// whose form has none.
const DEFAULT_DA: &[u8] = b"/Helv 10 Tf 0 g";

/// A free text annotation, read from its dictionary.
pub(super) struct FreeText {
    da: DefaultAppearance,
    quadding: i64,
    /// The text, as Unicode.
    text: String,
    /// The fill colour operator of `/C`.
    fill: Option<String>,
    /// The fill colour operator of `/IC`, for a closed line ending.
    interior: Option<String>,
    border: Border,
    /// The text box: `/Rect` moved in by `/RD`.
    text_box: [f64; 4],
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
        let text = string(doc, annot.get(b"Contents"))
            .map(|c| decode_text_string(&c))
            .filter(|t| !t.is_empty())
            .or_else(|| {
                let rc = text_or_stream(doc, annot.get(b"RC"))?;
                Some(plain_text(&decode_text_string(&rc)))
            })
            .unwrap_or_default();
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
        Self {
            da: DefaultAppearance::parse(&da),
            quadding: annot.get_i64(b"Q").unwrap_or(0),
            text,
            fill: colour(doc, annot.get(b"C"), false),
            interior: colour(doc, annot.get(b"IC"), false),
            border: Border::of(doc, annot),
            text_box: inset(rect, &numbers(doc, annot.get(b"RD"))),
            rect,
            callout: if matches!(callout.len(), 2 | 3) {
                callout
            } else {
                Vec::new()
            },
            ending,
            opacity: number(doc, annot.get(b"CA")).unwrap_or(1.0).clamp(0.0, 1.0),
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
        let frame = inset(self.text_box, &[bw / 2.0; 4]);
        let stroke = self
            .da
            .colour()
            .and_then(|c| colour_op(&c, true))
            .unwrap_or_else(|| "0 G".to_owned());
        s.push_str("q\n");
        if let Some(fill) = &self.fill {
            let _ = writeln!(s, "{fill}");
            rectangle(&mut s, frame);
            s.push_str("f\n");
        }
        let mut reach = 0.0f64;
        if bw > 0.0 {
            let _ = writeln!(s, "{stroke}");
            self.border.write(&mut s);
            rectangle(&mut s, frame);
            s.push_str("S\n");
            if let [first, second, rest @ ..] = self.callout.as_slice() {
                let _ = writeln!(s, "{} {} m", n(first.0), n(first.1));
                for (x, y) in std::iter::once(second).chain(rest) {
                    let _ = writeln!(s, "{} {} l", n(*x), n(*y));
                }
                s.push_str("S\n");
                if let Some(ending) = self.ending {
                    let size = LineEnding::size(bw, (first.0 - second.0).hypot(first.1 - second.1));
                    ending.write(&mut s, *first, *second, size, self.interior.as_deref());
                    reach = size;
                }
            }
        }
        s.push_str("Q\n");
        let area = inset(self.text_box, &[2.0 * bw; 4]);
        if !self.text.is_empty() {
            s.push_str("q\n");
            rectangle(&mut s, area);
            s.push_str("W n\n");
            let block = TextBlock {
                da: &self.da,
                size: if self.da.size > 0.0 {
                    self.da.size
                } else {
                    AUTO_MULTILINE_SIZE
                },
                quadding: self.quadding,
                area,
            };
            block.write(&mut s, &self.text, ts);
            s.push_str("Q\n");
        }
        Synthesized {
            content: s.into_bytes(),
            bbox: inset(self.rect, &[-(bw + reach); 4]),
            opacity: self.opacity,
            blend: BlendMode::Normal,
            matrix: IDENTITY,
        }
    }
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
        assert!(c.contains("1 0 0 rg\n100.5 50.5 49 24 re\nf"), "{c}");
        assert!(c.contains("0 0 1 RG\n1 w\n100.5 50.5 49 24 re\nS"), "{c}");
        // The first baseline is one font size below the top less the margin.
        assert!(
            c.contains("BT /F1 8 Tf 0 0 1 rg\n102 65 Td <48656C6C6F21> Tj ET"),
            "{c}"
        );
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
            c.contains("100.5 0.5 99 99 re"),
            "the box is inset by /RD: {c}"
        );
        assert!(c.contains("cm\n-6 3.4641 m"), "{c}");
    }
}
