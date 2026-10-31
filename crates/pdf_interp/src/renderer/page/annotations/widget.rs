//! Appearances synthesised for interactive form field widgets (ISO 32000-2
//! §12.7): text fields, choice fields and buttons, drawn from the field
//! value, the default appearance string `/DA`, the appearance
//! characteristics `/MK` and the form's default resources `/DR`.
//!
//! A widget is synthesised when it has no normal appearance, and a text or
//! choice field also when the form sets `/NeedAppearances`, since its
//! stored appearance may not show its value.  The content is written in a
//! box of the widget's size, its sides swapped when `/MK /R` turns it a
//! quarter turn, which the form matrix turns and places on `/Rect`.  It is
//! drawn with the resources of [`resources`]: `/DR` with the `/DA` font,
//! and `ZaDb` for check marks, added as standard fonts when `/DR` lacks
//! them.

use std::collections::HashSet;
use std::fmt::Write as _;

use pdf::{Dictionary, Document, Object, ObjectId};

use super::content::{
    self, circle, colour_op, n, number, numbers, rectangle, resolve_array, string, text_or_stream,
};
use super::text::{self, CHECK_FONT, DefaultAppearance, decode_text_string, plain_text};
use super::typeset::{
    AUTO_MULTILINE_SIZE, LEADING, TextBlock, Typesetter, aligned_x, hex, show_line,
};
use crate::resources::dict_ext::DictExt;

/// Field flags (§12.7.4, Tables 226, 227, 229, 231).
const MULTILINE: i64 = 1 << 12;
const PASSWORD: i64 = 1 << 13;
const RADIO: i64 = 1 << 15;
const PUSH_BUTTON: i64 = 1 << 16;
const COMBO: i64 = 1 << 17;
const COMB: i64 = 1 << 24;
const RICH_TEXT: i64 = 1 << 25;

/// Deepest `/Parent` chain followed for inherited field attributes.
const MAX_FIELD_DEPTH: usize = 32;

/// The document's interactive form dictionary (§12.7.3).
pub(super) struct AcroForm {
    need_appearances: bool,
    da: Option<Vec<u8>>,
    q: Option<i64>,
    dr: Option<Dictionary>,
}

impl AcroForm {
    /// The catalog's `/AcroForm`, or `None` when it has none.
    pub(super) fn of(doc: &Document) -> Option<Self> {
        let catalog = doc.catalog().ok()?;
        let form = crate::resources::resolve_dict(doc, catalog.get(b"AcroForm")?)?;
        Some(Self {
            need_appearances: form.get_bool(b"NeedAppearances") == Some(true),
            da: string(doc, form.get(b"DA")),
            q: form.get_i64(b"Q"),
            dr: form
                .get(b"DR")
                .and_then(|o| crate::resources::resolve_dict(doc, o)),
        })
    }

    /// The form's default appearance string `/DA`.
    pub(super) fn default_appearance(&self) -> Option<&[u8]> {
        self.da.as_deref()
    }

    /// The form's default resources `/DR`.
    pub(super) const fn default_resources(&self) -> Option<&Dictionary> {
        self.dr.as_ref()
    }
}

/// The kind of a field (`/FT`, §12.7.4).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Text,
    Choice,
    Button,
}

/// The attributes of the field a widget belongs to, with inheritance
/// through `/Parent` resolved.
pub(super) struct Field {
    kind: Kind,
    flags: i64,
    value: Option<Object>,
    /// The plain text of the rich text value `/RV` of a text field.
    rich: Option<String>,
    /// Whether a button is in an on state.
    on: bool,
    da: DefaultAppearance,
    quadding: i64,
    max_len: Option<i64>,
    options: Vec<(Vec<u8>, Vec<u8>)>,
    top_index: usize,
    /// The indices `/I` of the selected options.
    selected: HashSet<usize>,
    background: Option<Vec<f64>>,
    border_colour: Option<Vec<f64>>,
    caption: Option<Vec<u8>>,
    border_width: f64,
    /// The `/MK /R` rotation: 0, 90, 180 or 270 degrees counter-clockwise.
    rotation: u16,
}

impl Field {
    /// The field of the widget `annot` when its appearance is synthesised:
    /// it has no normal appearance (`has_normal` false), or it is a text or
    /// choice field and the form sets `/NeedAppearances`.
    pub(super) fn to_synthesise(
        doc: &Document,
        annot: &Dictionary,
        form: &AcroForm,
        has_normal: bool,
    ) -> Option<Self> {
        let chain = field_chain(doc, annot);
        let inherited = |key: &[u8]| inherited(doc, &chain, key);
        let kind = match inherited(b"FT")?.as_name()? {
            b"Tx" => Kind::Text,
            b"Ch" => Kind::Choice,
            b"Btn" => Kind::Button,
            _ => return None,
        };
        if has_normal && (kind == Kind::Button || !form.need_appearances) {
            return None;
        }
        let flags = inherited(b"Ff").and_then(|o| o.as_i64()).unwrap_or(0);
        let value = inherited(b"V");
        let state = annot
            .get_name(b"AS")
            .map(<[u8]>::to_vec)
            .or_else(|| value.as_ref().and_then(Object::as_name).map(<[u8]>::to_vec));
        let da = string(doc, inherited(b"DA").as_ref())
            .or_else(|| form.da.clone())
            .unwrap_or_else(|| b"/Helv 0 Tf 0 g".to_vec());
        let da = DefaultAppearance::parse(&da);
        let mk = annot
            .get(b"MK")
            .and_then(|o| crate::resources::resolve_dict(doc, o));
        let colour = |key: &[u8]| {
            mk.as_ref()
                .map(|mk| numbers(doc, mk.get(key)))
                .filter(|c| matches!(c.len(), 1 | 3 | 4))
        };
        let bs = annot
            .get(b"BS")
            .and_then(|o| crate::resources::resolve_dict(doc, o));
        let border_width = bs
            .as_ref()
            .and_then(|bs| number(doc, bs.get(b"W")))
            .unwrap_or(1.0)
            .max(0.0);
        let options = resolve_array(doc, inherited(b"Opt").as_ref())
            .iter()
            .filter_map(|o| option(doc, o))
            .collect();
        Some(Self {
            kind,
            flags,
            on: state.is_some_and(|s| s != b"Off"),
            rich: (kind == Kind::Text)
                .then(|| text_or_stream(doc, inherited(b"RV").as_ref()))
                .flatten()
                .map(|rv| plain_text(&decode_text_string(&rv))),
            value,
            da,
            quadding: annot
                .get_i64(b"Q")
                .or_else(|| inherited(b"Q").and_then(|o| o.as_i64()))
                .or(form.q)
                .unwrap_or(0),
            max_len: inherited(b"MaxLen").and_then(|o| o.as_i64()),
            options,
            top_index: inherited(b"TI")
                .and_then(|o| o.as_i64())
                .and_then(|i| usize::try_from(i).ok())
                .unwrap_or(0),
            selected: numbers(doc, inherited(b"I").as_ref())
                .iter()
                .filter(|i| **i >= 0.0)
                .map(|&i| {
                    #[expect(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "a finite non-negative index; `as` saturates, and an index past the options selects none"
                    )]
                    let i = i as usize;
                    i
                })
                .collect(),
            background: colour(b"BG"),
            border_colour: colour(b"BC"),
            caption: mk.as_ref().and_then(|mk| string(doc, mk.get(b"CA"))),
            border_width,
            rotation: mk
                .as_ref()
                .and_then(|mk| mk.get_i64(b"R"))
                .map_or(0, quarter_turns),
        })
    }

    /// The `/MK /R` rotation of the widget's appearance, in degrees
    /// counter-clockwise: 0, 90, 180 or 270.
    pub(super) const fn rotation(&self) -> u16 {
        self.rotation
    }
}

/// The rotation `degrees`, which `/MK /R` requires to be a multiple of 90,
/// reduced to 0, 90, 180 or 270; 0 for another value.
const fn quarter_turns(degrees: i64) -> u16 {
    match degrees.rem_euclid(360) {
        90 => 90,
        180 => 180,
        270 => 270,
        _ => 0,
    }
}

/// The resources a field's appearance is drawn with: [`text::resources`]
/// over the form's `/DR` with the field's `/DA` font.
pub(super) fn resources(doc: &Document, form: &AcroForm, field: &Field) -> Dictionary {
    text::resources(doc, form.dr.as_ref(), &[&field.da.font])
}

/// The appearance content of `field` in `rect`, its text set by `ts`.
pub(super) fn content(
    doc: &Document,
    field: &Field,
    rect: [f64; 4],
    ts: &mut dyn Typesetter,
) -> Vec<u8> {
    let mut s = String::new();
    let radio = field.kind == Kind::Button && field.flags & (RADIO | PUSH_BUTTON) == RADIO;
    frame(&mut s, field, rect, radio);
    let bw = if field.border_colour.is_some() {
        field.border_width
    } else {
        0.0
    };
    let inner = inset(rect, bw);
    let _ = writeln!(
        s,
        "q {} {} {} {} re W n",
        n(inner[0]),
        n(inner[1]),
        n(inner[2] - inner[0]),
        n(inner[3] - inner[1])
    );
    let text_area = inset(inner, bw.max(1.0));
    match field.kind {
        Kind::Choice if field.flags & COMBO == 0 => list(&mut s, field, text_area, ts),
        Kind::Text | Kind::Choice => {
            let text = display_text(doc, field);
            let text = if field.flags & PASSWORD != 0 {
                "*".repeat(text.chars().count())
            } else {
                text
            };
            if field.flags & MULTILINE != 0 && field.kind == Kind::Text {
                multiline(&mut s, field, text_area, &text, ts);
            } else if field.flags & COMB != 0 && field.max_len.is_some_and(|m| m > 0) {
                comb(&mut s, field, text_area, &text, ts);
            } else {
                single_line(&mut s, field, field.quadding, text_area, &text, ts);
            }
        }
        Kind::Button if field.flags & PUSH_BUTTON != 0 => {
            if let Some(caption) = &field.caption {
                let text = decode_text_string(caption);
                single_line(&mut s, field, 1, text_area, &text, ts);
            }
        }
        Kind::Button => {
            if field.on {
                check_mark(&mut s, field, inner, radio, ts);
            }
        }
    }
    s.push_str("Q\n");
    s.into_bytes()
}

/// The background `/MK /BG` and border `/MK /BC` of the widget, a circle
/// for a radio button.
fn frame(s: &mut String, field: &Field, rect: [f64; 4], radio: bool) {
    let shape = |s: &mut String, r: [f64; 4]| {
        if radio {
            circle(s, r);
        } else {
            rectangle(s, r);
        }
    };
    if let Some(bg) = &field.background {
        let _ = writeln!(s, "{}", colour_op(bg, false).unwrap_or_default());
        shape(s, rect);
        s.push_str("f\n");
    }
    if let Some(bc) = &field.border_colour
        && field.border_width > 0.0
    {
        let _ = writeln!(
            s,
            "{}\n{} w",
            colour_op(bc, true).unwrap_or_default(),
            n(field.border_width)
        );
        shape(s, inset(rect, field.border_width / 2.0));
        s.push_str("S\n");
    }
}

/// The text a text field or combo box shows: its value, a text string or
/// stream, or for a choice the display text of the option whose export
/// value it is.  A text field shows the plain text of its rich text value
/// `/RV` instead when it is a rich text field or its value is empty; its
/// paragraphs are lines of a multiline field and are joined by spaces
/// otherwise (§12.7.4.3).
fn display_text(doc: &Document, field: &Field) -> String {
    let raw = match &field.value {
        Some(Object::String(s, _)) => s.clone(),
        Some(Object::Name(n)) => n.clone(),
        Some(Object::Array(a)) => a
            .first()
            .and_then(|o| string(doc, Some(o)))
            .unwrap_or_default(),
        Some(o @ Object::Reference(_)) => text_or_stream(doc, Some(o)).unwrap_or_default(),
        _ => Vec::new(),
    };
    if let Some(rich) = &field.rich
        && (field.flags & RICH_TEXT != 0 || raw.is_empty())
    {
        return if field.flags & MULTILINE != 0 {
            rich.clone()
        } else {
            rich.replace('\n', " ")
        };
    }
    let raw = field
        .options
        .iter()
        .find(|(export, _)| *export == raw)
        .map_or(raw, |(_, display)| display.clone());
    decode_text_string(&raw)
}

/// The font size of a single line of `text` in `area`: the `/DA` size, or
/// when it asks for automatic sizing, the size that fits the area's height
/// and width.
fn line_size(field: &Field, area: [f64; 4], text_em: f64) -> f64 {
    if field.da.size > 0.0 {
        return field.da.size;
    }
    let by_height = (area[3] - area[1]) / LEADING;
    let by_width = if text_em > 0.0 {
        (area[2] - area[0]) / text_em
    } else {
        by_height
    };
    by_height.min(by_width).max(1.0)
}

/// One line of `text` in the field's font, aligned in `area` by `quadding`
/// and centred vertically.
fn single_line(
    s: &mut String,
    field: &Field,
    quadding: i64,
    area: [f64; 4],
    text: &str,
    ts: &mut dyn Typesetter,
) {
    if text.is_empty() {
        return;
    }
    let font = &field.da.font;
    let em = ts.width(font, text);
    let size = line_size(field, area, em);
    let x = aligned_x(quadding, area, em * size);
    // The baseline sits so that the body of the line, taken as 0.7 of the
    // font size above it, is centred.
    let y = 0.35f64.mul_add(-size, f64::midpoint(area[1], area[3]));
    show_line(s, ts, &field.da, font, size, (x, y), text);
}

/// Multi-line text: lines broken at line ends and wrapped at spaces to the
/// area's width, from its top.
fn multiline(s: &mut String, field: &Field, area: [f64; 4], text: &str, ts: &mut dyn Typesetter) {
    let size = if field.da.size > 0.0 {
        field.da.size
    } else {
        AUTO_MULTILINE_SIZE
    };
    let block = TextBlock {
        da: &field.da,
        size,
        quadding: field.quadding,
        area,
    };
    block.write(s, text, ts);
}

/// A comb field: `/MaxLen` equal cells, one character centred in each.
fn comb(s: &mut String, field: &Field, area: [f64; 4], text: &str, ts: &mut dyn Typesetter) {
    let font = &field.da.font;
    let cells = field.max_len.unwrap_or(1).max(1);
    #[expect(
        clippy::cast_precision_loss,
        reason = "a field holds far fewer than 2^52 characters"
    )]
    let cell = (area[2] - area[0]) / cells as f64;
    let size = line_size(
        field,
        [area[0], area[1], area[0] + cell, area[3]],
        ts.width(font, "W"),
    );
    let y = 0.35f64.mul_add(-size, f64::midpoint(area[1], area[3]));
    let mut buf = [0u8; 4];
    for (i, ch) in text
        .chars()
        .take(usize::try_from(cells).unwrap_or(usize::MAX))
        .enumerate()
    {
        let ch = ch.encode_utf8(&mut buf);
        #[expect(
            clippy::cast_precision_loss,
            reason = "a field holds far fewer than 2^52 characters"
        )]
        let left = cell.mul_add(i as f64, area[0]);
        let x = aligned_x(
            1,
            [left, area[1], left + cell, area[3]],
            ts.width(font, ch) * size,
        );
        show_line(s, ts, &field.da, font, size, (x, y), ch);
    }
}

/// A list box: the options from `/TI` down, selected ones highlighted.
fn list(s: &mut String, field: &Field, area: [f64; 4], ts: &mut dyn Typesetter) {
    let font = &field.da.font;
    let size = if field.da.size > 0.0 {
        field.da.size
    } else {
        AUTO_MULTILINE_SIZE
    };
    let row = size * LEADING;
    let mut top = area[3];
    for (i, (_, display)) in field.options.iter().enumerate().skip(field.top_index) {
        if top <= area[1] {
            break;
        }
        if field.selected.contains(&i) {
            let _ = writeln!(
                s,
                "0.6 0.75 0.95 rg {} {} {} {} re f",
                n(area[0]),
                n(top - row),
                n(area[2] - area[0]),
                n(row)
            );
        }
        let text = decode_text_string(display);
        let x = aligned_x(field.quadding, area, ts.width(font, &text) * size);
        show_line(s, ts, &field.da, font, size, (x, top - size), &text);
        top -= row;
    }
}

/// The mark of a checked check box or radio button: the `/MK /CA`
/// character, by default a check (`4`) or a bullet (`l`), in `ZapfDingbats`,
/// centred in `inner`.
fn check_mark(
    s: &mut String,
    field: &Field,
    inner: [f64; 4],
    radio: bool,
    ts: &mut dyn Typesetter,
) {
    let glyph = field
        .caption
        .as_ref()
        .and_then(|c| c.first().copied())
        .unwrap_or(if radio { b'l' } else { b'4' });
    let shorter_side = (inner[2] - inner[0]).min(inner[3] - inner[1]);
    let size = if field.da.size > 0.0 {
        field.da.size
    } else {
        shorter_side * 0.8
    };
    let em = ts.advance(CHECK_FONT, &[glyph]);
    let x = aligned_x(1, inner, em * size);
    let y = 0.35f64.mul_add(-size, f64::midpoint(inner[1], inner[3]));
    field.da.begin_text(s, CHECK_FONT, size);
    let _ = writeln!(s, "{} {} Td {} Tj ET", n(x), n(y), hex(&[glyph]));
}

/// The widget, then each field above it through `/Parent`, as resolved
/// dictionaries: the order in which an inherited attribute is looked up.
pub(super) fn field_chain(doc: &Document, annot: &Dictionary) -> Vec<Dictionary> {
    let mut chain = vec![annot.clone()];
    let mut seen: Vec<ObjectId> = Vec::new();
    while chain.len() < MAX_FIELD_DEPTH {
        let Some(Object::Reference(id)) = chain.last().and_then(|d| d.get(b"Parent")).cloned()
        else {
            break;
        };
        if seen.contains(&id) {
            break;
        }
        seen.push(id);
        let Ok(parent) = doc.get_dictionary(id) else {
            break;
        };
        chain.push((*parent).clone());
    }
    chain
}

/// The inheritable field attribute `key` (§12.7.4.1): the entry of the
/// first dictionary of `chain` (from [`field_chain`]) that has it, with an
/// indirect reference followed unless it refers to a stream, which stays a
/// reference for [`text_or_stream`].
pub(super) fn inherited(doc: &Document, chain: &[Dictionary], key: &[u8]) -> Option<Object> {
    let value = chain.iter().find_map(|d| d.get(key))?;
    if let Object::Reference(id) = value
        && let Ok(referent) = doc.get_object(*id)
        && !matches!(referent.as_ref(), Object::Stream(_))
    {
        return Some(referent.as_ref().clone());
    }
    Some(value.clone())
}

/// An `/Opt` entry: `(export value, display text)`, both the string itself
/// for a plain string.
fn option(doc: &Document, entry: &Object) -> Option<(Vec<u8>, Vec<u8>)> {
    if let Some(s) = string(doc, Some(entry)) {
        return Some((s.clone(), s));
    }
    let pair = resolve_array(doc, Some(entry));
    let export = string(doc, pair.first())?;
    let display = string(doc, pair.get(1)).unwrap_or_else(|| export.clone());
    Some((export, display))
}

/// `r` moved in by `d` on every side, collapsing rather than inverting.
fn inset(r: [f64; 4], d: f64) -> [f64; 4] {
    content::inset(r, &[d; 4])
}
