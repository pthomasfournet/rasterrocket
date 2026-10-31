//! Pop-up annotations (ISO 32000-2 §12.5.6.14).
//!
//! A pop-up window shows the text of its parent markup annotation and is
//! displayed only while open: a pop-up whose `/Open` is not true draws
//! nothing, whatever appearance it carries.  An open pop-up without a
//! normal appearance is drawn as a window filled yellow and outlined
//! in black, holding the parent's title `/T` and, on the following lines,
//! its `/Contents` (else the plain text of its `/RC`), in black 12 pt
//! Helvetica from the window's upper-left corner.

use pdf::{Dictionary, Document};
use raster::types::BlendMode;

use super::Synthesized;
use super::content::{inset, rectangle, string};
use super::text::{DefaultAppearance, TextBlock, decode_text_string, encode_win_ansi, plain_text};
use crate::resources::dict_ext::DictExt;

/// The resource name of the window's font, standard Helvetica.
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
/// parent is `parent`; `width` measures a byte string in [`FONT`].
pub(super) fn appearance(
    doc: &Document,
    parent: Option<&Dictionary>,
    rect: [f64; 4],
    mut width: &mut dyn FnMut(&[u8], &[u8]) -> f64,
) -> Synthesized {
    let mut s = String::from("1 1 0 rg 0 G 1 w\n");
    rectangle(&mut s, inset(rect, &[0.5; 4]));
    s.push_str("b\n");
    let text_of = |key: &[u8]| {
        parent
            .and_then(|p| string(doc, p.get(key)))
            .map(|t| decode_text_string(&t))
            .unwrap_or_default()
    };
    let mut contents = text_of(b"Contents");
    if contents.is_empty() {
        contents = plain_text(&text_of(b"RC"));
    }
    let title = text_of(b"T");
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
        block.write(&mut s, &encode_win_ansi(&text), &mut width);
        s.push_str("Q\n");
    }
    Synthesized {
        content: s.into_bytes(),
        bbox: rect,
        opacity: 1.0,
        blend: BlendMode::Normal,
    }
}
