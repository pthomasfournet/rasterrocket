//! Pop-up annotations (ISO 32000-2 §12.5.6.14).
//!
//! A pop-up window shows the text of its parent markup annotation and is
//! displayed only while open: a pop-up whose `/Open` is not true draws
//! nothing, whatever appearance it carries.  An open pop-up without a
//! normal appearance is drawn as a window filled yellow and outlined
//! in black, holding the parent's title `/T` and, on the following lines,
//! its text (`/Contents`, else the plain text of `/RC`), in black 12 pt
//! Helvetica from the window's upper-left corner.

use pdf::{Dictionary, Document};
use raster::types::BlendMode;

use super::content::{inset, rectangle, string};
use super::text::{DefaultAppearance, decode_text_string, markup_text};
use super::typeset::{TextBlock, Typesetter};
use super::{IDENTITY, Synthesized};
use crate::resources::dict_ext::DictExt;

/// The resource name of the window's font, standard Helvetica unless the
/// form's default resources define the name.
pub(super) const FONT: &[u8] = b"Helvetica";

/// Distance of the text from the window's edges.
const MARGIN: f64 = 3.0;

/// Whether the pop-up `annot` is open.
pub(super) fn is_open(annot: &Dictionary) -> bool {
    annot.get_bool(b"Open") == Some(true)
}

/// The parent annotation of the pop-up `annot`, the markup annotation
/// whose text it shows.
pub(super) fn parent(doc: &Document, annot: &Dictionary) -> Option<Dictionary> {
    crate::resources::resolve_dict(doc, annot.get(b"Parent")?)
}

/// The appearance of the open pop-up whose window is `rect` and whose
/// parent is `parent`, its text set by `ts` in [`FONT`].
pub(super) fn appearance(
    doc: &Document,
    parent: Option<&Dictionary>,
    rect: [f64; 4],
    ts: &mut dyn Typesetter,
) -> Synthesized {
    let mut s = String::from("1 1 0 rg 0 G 1 w\n");
    rectangle(&mut s, inset(rect, &[0.5; 4]));
    s.push_str("b\n");
    let contents = parent.map(|p| markup_text(doc, p)).unwrap_or_default();
    let title = parent
        .and_then(|p| string(doc, p.get(b"T")))
        .map(|t| decode_text_string(&t))
        .unwrap_or_default();
    let text = match (title.is_empty(), contents.is_empty()) {
        (false, false) => format!("{title}\n{contents}"),
        (false, true) => title,
        _ => contents,
    };
    if !text.is_empty() {
        let area = inset(rect, &[MARGIN; 4]);
        s.push_str("q\n");
        rectangle(&mut s, area);
        s.push_str("W n\n");
        let da = DefaultAppearance {
            font: FONT.to_vec(),
            size: 12.0,
            rest: "0 g".to_owned(),
        };
        let block = TextBlock {
            da: &da,
            size: da.size,
            quadding: 0,
            area,
        };
        block.write(&mut s, &text, ts);
        s.push_str("Q\n");
    }
    Synthesized {
        content: s.into_bytes(),
        bbox: rect,
        opacity: 1.0,
        blend: BlendMode::Normal,
        matrix: IDENTITY,
    }
}

#[cfg(test)]
mod tests {
    use super::appearance;
    use crate::renderer::page::annotations::typeset::WinAnsiTypesetter;
    use crate::test_helpers::make_doc_with_objects;

    #[test]
    fn the_window_shows_the_title_and_a_rich_text_stream() {
        let rc = "<body><p>Note</p></body>";
        let doc = make_doc_with_objects(&[
            (4, "<< /Subtype /Text /T (Ann) /RC 5 0 R >>".to_owned()),
            (
                5,
                format!("<< /Length {} >>\nstream\n{rc}\nendstream", rc.len()),
            ),
        ]);
        let parent = doc.get_dictionary((4, 0)).expect("a dictionary");
        let content = appearance(
            &doc,
            Some(&parent),
            [0.0, 0.0, 200.0, 100.0],
            &mut WinAnsiTypesetter,
        )
        .content;
        let content = String::from_utf8(content).expect("ASCII");
        assert!(content.contains("<416E6E> Tj"), "the title: {content}");
        assert!(
            content.contains("<4E6F7465> Tj"),
            "the rich text: {content}"
        );
    }
}
