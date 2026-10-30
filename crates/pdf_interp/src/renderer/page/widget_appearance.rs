//! Appearances synthesised for interactive form field widgets (ISO 32000-2
//! §12.7): text fields, choice fields and buttons, drawn from the field
//! value, the default appearance string `/DA`, the appearance
//! characteristics `/MK` and the form's default resources `/DR`.
//!
//! A widget is synthesised when it has no normal appearance, and a text or
//! choice field also when the form sets `/NeedAppearances`, since its
//! stored appearance may not show its value.  The content is written in
//! page user space and drawn with the resources of [`resources`]: `/DR`
//! with the `/DA` font, and `ZaDb` for check marks, added as standard
//! fonts when `/DR` lacks them.

use std::fmt::Write as _;

use pdf::{Dictionary, Document, Object, ObjectId};

use crate::resources::dict_ext::DictExt;

/// Field flags (§12.7.4, Tables 226, 227, 229, 231).
const MULTILINE: i64 = 1 << 12;
const PASSWORD: i64 = 1 << 13;
const RADIO: i64 = 1 << 15;
const PUSH_BUTTON: i64 = 1 << 16;
const COMBO: i64 = 1 << 17;
const COMB: i64 = 1 << 24;

/// The resource name of the font check marks are drawn in.
const CHECK_FONT: &[u8] = b"ZaDb";

/// Deepest `/Parent` chain followed for inherited field attributes.
const MAX_FIELD_DEPTH: usize = 32;

/// Line spacing of multi-line text, in units of the font size.
const LEADING: f64 = 1.15;

/// Font size of multi-line text whose `/DA` asks for automatic sizing.
const AUTO_MULTILINE_SIZE: f64 = 12.0;

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
    /// Whether a button is in an on state.
    on: bool,
    font: Vec<u8>,
    /// `/DA` with its `Tf` operator removed; the colour operators remain.
    da_rest: String,
    /// The `/DA` font size; 0 asks for automatic sizing.
    size: f64,
    quadding: i64,
    max_len: Option<i64>,
    options: Vec<(Vec<u8>, Vec<u8>)>,
    top_index: usize,
    selected: Vec<usize>,
    background: Option<Vec<f64>>,
    border_colour: Option<Vec<f64>>,
    caption: Option<Vec<u8>>,
    border_width: f64,
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
        let inherited = |key: &[u8]| chain.iter().find_map(|d| d.get(key).cloned());
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
        let (font, size, da_rest) = split_da(&da);
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
            value,
            font,
            da_rest,
            size,
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
                        reason = "a non-negative option index"
                    )]
                    let i = i as usize;
                    i
                })
                .collect(),
            background: colour(b"BG"),
            border_colour: colour(b"BC"),
            caption: mk.as_ref().and_then(|mk| string(doc, mk.get(b"CA"))),
            border_width,
        })
    }
}

/// The resources a field's appearance is drawn with: the form's `/DR`, with
/// `ZaDb` as standard `ZapfDingbats` and the field's font as standard
/// Helvetica when it lacks them.
pub(super) fn resources(doc: &Document, form: &AcroForm, field: &Field) -> Dictionary {
    let mut resources = form.dr.clone().unwrap_or_default();
    let mut fonts = resources
        .get(b"Font")
        .and_then(|o| crate::resources::resolve_dict(doc, o))
        .unwrap_or_default();
    for (name, base_font) in [
        (CHECK_FONT, &b"ZapfDingbats"[..]),
        (field.font.as_slice(), b"Helvetica"),
    ] {
        if fonts.get(name).is_none() {
            let mut font = Dictionary::new();
            font.set(b"Type", Object::Name(b"Font".to_vec()));
            font.set(b"Subtype", Object::Name(b"Type1".to_vec()));
            font.set(b"BaseFont", Object::Name(base_font.to_vec()));
            if base_font == b"Helvetica" {
                font.set(b"Encoding", Object::Name(b"WinAnsiEncoding".to_vec()));
            }
            fonts.set(name, Object::Dictionary(font));
        }
    }
    resources.set(b"Font", Object::Dictionary(fonts));
    resources
}

/// The appearance content of `field` in `rect`; `width` measures a byte
/// string in the field's font, in text space units per unit font size.
pub(super) fn content(
    doc: &Document,
    field: &Field,
    rect: [f64; 4],
    mut width: impl FnMut(&[u8], &[u8]) -> f64,
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
        Kind::Choice if field.flags & COMBO == 0 => list(&mut s, field, text_area, &mut width),
        Kind::Text | Kind::Choice => {
            let text = display_text(doc, field);
            let text: Vec<u8> = if field.flags & PASSWORD != 0 {
                vec![b'*'; text.len()]
            } else {
                text
            };
            if field.flags & MULTILINE != 0 && field.kind == Kind::Text {
                multiline(&mut s, field, text_area, &text, &mut width);
            } else if field.flags & COMB != 0 && field.max_len.is_some_and(|m| m > 0) {
                comb(&mut s, field, text_area, &text, &mut width);
            } else {
                let line = Line {
                    font: &field.font,
                    quadding: field.quadding,
                    area: text_area,
                };
                single_line(&mut s, field, &line, &text, &mut width);
            }
        }
        Kind::Button if field.flags & PUSH_BUTTON != 0 => {
            if let Some(caption) = &field.caption {
                let text = encode_win_ansi(&decode_text_string(caption));
                let line = Line {
                    font: &field.font,
                    quadding: 1,
                    area: text_area,
                };
                single_line(&mut s, field, &line, &text, &mut width);
            }
        }
        Kind::Button => {
            if field.on {
                check_mark(&mut s, field, inner, radio, &mut width);
            }
        }
    }
    s.push_str("Q\n");
    s.into_bytes()
}

/// Where a single line of text goes: its font, its quadding and the area it
/// is fitted and aligned in.
struct Line<'a> {
    font: &'a [u8],
    quadding: i64,
    area: [f64; 4],
}

/// The background `/MK /BG` and border `/MK /BC` of the widget, a circle
/// for a radio button.
fn frame(s: &mut String, field: &Field, rect: [f64; 4], radio: bool) {
    let shape = |s: &mut String, r: [f64; 4]| {
        if radio {
            circle(s, r);
        } else {
            let _ = writeln!(
                s,
                "{} {} {} {} re",
                n(r[0]),
                n(r[1]),
                n(r[2] - r[0]),
                n(r[3] - r[1])
            );
        }
    };
    if let Some(bg) = &field.background {
        let _ = writeln!(s, "{}", colour_op(bg, false));
        shape(s, rect);
        s.push_str("f\n");
    }
    if let Some(bc) = &field.border_colour
        && field.border_width > 0.0
    {
        let _ = writeln!(s, "{}\n{} w", colour_op(bc, true), n(field.border_width));
        shape(s, inset(rect, field.border_width / 2.0));
        s.push_str("S\n");
    }
}

/// The text a text field or combo box shows: its value, or for a choice
/// the display text of the option whose export value it is.
fn display_text(doc: &Document, field: &Field) -> Vec<u8> {
    let raw = match &field.value {
        Some(Object::String(s, _)) => s.clone(),
        Some(Object::Name(n)) => n.clone(),
        Some(Object::Array(a)) => a
            .first()
            .and_then(|o| string(doc, Some(o)))
            .unwrap_or_default(),
        Some(o @ Object::Reference(_)) => string(doc, Some(o)).unwrap_or_default(),
        _ => Vec::new(),
    };
    let raw = field
        .options
        .iter()
        .find(|(export, _)| *export == raw)
        .map_or(raw, |(_, display)| display.clone());
    encode_win_ansi(&decode_text_string(&raw))
}

/// The font size of a single line of `text` in `area`: the `/DA` size, or
/// when it asks for automatic sizing, the size that fits the area's height
/// and width.
fn line_size(field: &Field, area: [f64; 4], text_em: f64) -> f64 {
    if field.size > 0.0 {
        return field.size;
    }
    let by_height = (area[3] - area[1]) / LEADING;
    let by_width = if text_em > 0.0 {
        (area[2] - area[0]) / text_em
    } else {
        by_height
    };
    by_height.min(by_width).max(1.0)
}

/// Begin a text object in `font` at `size` with the `/DA` colours.
fn begin_text(s: &mut String, field: &Field, font: &[u8], size: f64) {
    let _ = writeln!(
        s,
        "BT /{} {} Tf {}",
        String::from_utf8_lossy(font),
        n(size),
        field.da_rest
    );
}

/// One line of `text`, aligned and centred vertically as `line` places it.
fn single_line(
    s: &mut String,
    field: &Field,
    line: &Line<'_>,
    text: &[u8],
    width: &mut impl FnMut(&[u8], &[u8]) -> f64,
) {
    if text.is_empty() {
        return;
    }
    let em = width(line.font, text);
    let size = line_size(field, line.area, em);
    let x = aligned_x(line.quadding, line.area, em * size);
    // The baseline sits so that the body of the line, taken as 0.7 of the
    // font size above it, is centred.
    let y = 0.35f64.mul_add(-size, f64::midpoint(line.area[1], line.area[3]));
    begin_text(s, field, line.font, size);
    let _ = writeln!(s, "{} {} Td {} Tj ET", n(x), n(y), hex(text));
}

/// The left edge of a line `line_width` wide in `area` for quadding `q`.
fn aligned_x(q: i64, area: [f64; 4], line_width: f64) -> f64 {
    match q {
        1 => f64::midpoint(area[0], area[2]) - line_width / 2.0,
        2 => area[2] - line_width,
        _ => area[0],
    }
}

/// Multi-line text: lines broken at line ends and wrapped at spaces to the
/// area's width, from its top.
fn multiline(
    s: &mut String,
    field: &Field,
    area: [f64; 4],
    text: &[u8],
    width: &mut impl FnMut(&[u8], &[u8]) -> f64,
) {
    let size = if field.size > 0.0 {
        field.size
    } else {
        AUTO_MULTILINE_SIZE
    };
    let max = (area[2] - area[0]) / size;
    let mut lines: Vec<Vec<u8>> = Vec::new();
    for paragraph in text.split(|&b| b == b'\n' || b == b'\r') {
        let mut line: Vec<u8> = Vec::new();
        for word in paragraph.split(|&b| b == b' ') {
            let mut candidate = line.clone();
            if !candidate.is_empty() {
                candidate.push(b' ');
            }
            candidate.extend_from_slice(word);
            if !line.is_empty() && width(&field.font, &candidate) > max {
                lines.push(std::mem::take(&mut line));
                line.extend_from_slice(word);
            } else {
                line = candidate;
            }
        }
        lines.push(line);
    }
    let mut y = area[3] - size;
    for line in lines {
        if y < area[1] - size {
            break;
        }
        if !line.is_empty() {
            let x = aligned_x(field.quadding, area, width(&field.font, &line) * size);
            begin_text(s, field, &field.font, size);
            let _ = writeln!(s, "{} {} Td {} Tj ET", n(x), n(y), hex(&line));
        }
        y = size.mul_add(-LEADING, y);
    }
}

/// A comb field: `/MaxLen` equal cells, one character centred in each.
fn comb(
    s: &mut String,
    field: &Field,
    area: [f64; 4],
    text: &[u8],
    width: &mut impl FnMut(&[u8], &[u8]) -> f64,
) {
    let cells = field.max_len.unwrap_or(1).max(1);
    #[expect(
        clippy::cast_precision_loss,
        reason = "a field holds far fewer than 2^52 characters"
    )]
    let cell = (area[2] - area[0]) / cells as f64;
    let size = line_size(
        field,
        [area[0], area[1], area[0] + cell, area[3]],
        width(&field.font, b"W"),
    );
    let y = 0.35f64.mul_add(-size, f64::midpoint(area[1], area[3]));
    for (i, ch) in text
        .iter()
        .take(usize::try_from(cells).unwrap_or(usize::MAX))
        .enumerate()
    {
        let ch = [*ch];
        #[expect(
            clippy::cast_precision_loss,
            reason = "a field holds far fewer than 2^52 characters"
        )]
        let left = cell.mul_add(i as f64, area[0]);
        let x = aligned_x(
            1,
            [left, area[1], left + cell, area[3]],
            width(&field.font, &ch) * size,
        );
        begin_text(s, field, &field.font, size);
        let _ = writeln!(s, "{} {} Td {} Tj ET", n(x), n(y), hex(&ch));
    }
}

/// A list box: the options from `/TI` down, selected ones highlighted.
fn list(
    s: &mut String,
    field: &Field,
    area: [f64; 4],
    width: &mut impl FnMut(&[u8], &[u8]) -> f64,
) {
    let size = if field.size > 0.0 {
        field.size
    } else {
        AUTO_MULTILINE_SIZE
    };
    let row = size * LEADING;
    let mut top = area[3];
    for (i, (_, display)) in field.options.iter().enumerate().skip(field.top_index) {
        if top - row < area[1] - row {
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
        let text = encode_win_ansi(&decode_text_string(display));
        let x = aligned_x(field.quadding, area, width(&field.font, &text) * size);
        begin_text(s, field, &field.font, size);
        let _ = writeln!(s, "{} {} Td {} Tj ET", n(x), n(top - size), hex(&text));
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
    width: &mut impl FnMut(&[u8], &[u8]) -> f64,
) {
    let glyph = field
        .caption
        .as_ref()
        .and_then(|c| c.first().copied())
        .unwrap_or(if radio { b'l' } else { b'4' });
    let shorter_side = (inner[2] - inner[0]).min(inner[3] - inner[1]);
    let size = if field.size > 0.0 {
        field.size
    } else {
        shorter_side * 0.8
    };
    let em = width(CHECK_FONT, &[glyph]);
    let x = aligned_x(1, inner, em * size);
    let y = 0.35f64.mul_add(-size, f64::midpoint(inner[1], inner[3]));
    begin_text(s, field, CHECK_FONT, size);
    let _ = writeln!(s, "{} {} Td {} Tj ET", n(x), n(y), hex(&[glyph]));
}

/// A circle inscribed in `r`.
fn circle(s: &mut String, r: [f64; 4]) {
    const KAPPA: f64 = 0.552_284_75;
    let (cx, cy) = (f64::midpoint(r[0], r[2]), f64::midpoint(r[1], r[3]));
    let radius = ((r[2] - r[0]).min(r[3] - r[1]) / 2.0).max(0.0);
    let k = radius * KAPPA;
    let _ = writeln!(s, "{} {} m", n(cx), n(cy + radius));
    let arcs = [
        [cx + k, cy + radius, cx + radius, cy + k, cx + radius, cy],
        [cx + radius, cy - k, cx + k, cy - radius, cx, cy - radius],
        [cx - k, cy - radius, cx - radius, cy - k, cx - radius, cy],
        [cx - radius, cy + k, cx - k, cy + radius, cx, cy + radius],
    ];
    for a in arcs {
        let _ = writeln!(
            s,
            "{} {} {} {} {} {} c",
            n(a[0]),
            n(a[1]),
            n(a[2]),
            n(a[3]),
            n(a[4]),
            n(a[5])
        );
    }
    s.push_str("h\n");
}

/// The colour operator for 1, 3 or 4 components.
fn colour_op(c: &[f64], stroke: bool) -> String {
    let op = match (c.len(), stroke) {
        (1, false) => "g",
        (1, true) => "G",
        (3, false) => "rg",
        (3, true) => "RG",
        (_, false) => "k",
        (_, true) => "K",
    };
    let comps: Vec<String> = c.iter().map(|&v| n(v)).collect();
    format!("{} {op}", comps.join(" "))
}

/// Split a default appearance string into its `Tf` font name and size and
/// the remaining operators.  Without a `Tf`, the font is `Helv` at
/// automatic size.
fn split_da(da: &[u8]) -> (Vec<u8>, f64, String) {
    let text = String::from_utf8_lossy(da);
    let tokens: Vec<&str> = text.split_ascii_whitespace().collect();
    let Some(tf) = tokens.iter().rposition(|t| *t == "Tf") else {
        return (b"Helv".to_vec(), 0.0, tokens.join(" "));
    };
    let font = tf
        .checked_sub(2)
        .and_then(|i| tokens[i].strip_prefix('/'))
        .map_or_else(|| b"Helv".to_vec(), |f| f.as_bytes().to_vec());
    let size = tf
        .checked_sub(1)
        .and_then(|i| tokens[i].parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .unwrap_or(0.0)
        .max(0.0);
    let start = tf.saturating_sub(2);
    let rest: Vec<&str> = tokens[..start]
        .iter()
        .chain(&tokens[tf + 1..])
        .copied()
        .collect();
    (font, size, rest.join(" "))
}

/// Decode a PDF text string (§7.9.2.2): UTF-16BE or UTF-8 with a byte order
/// mark, otherwise `PDFDocEncoding`.
fn decode_text_string(bytes: &[u8]) -> String {
    if let Some(utf16) = bytes.strip_prefix(&[0xFE, 0xFF]) {
        let units: Vec<u16> = utf16
            .as_chunks::<2>()
            .0
            .iter()
            .map(|u| u16::from_be_bytes(*u))
            .collect();
        return String::from_utf16_lossy(&units);
    }
    if let Some(utf8) = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]) {
        return String::from_utf8_lossy(utf8).into_owned();
    }
    bytes.iter().map(|&b| pdf_doc_char(b)).collect()
}

/// The character of a `PDFDocEncoding` byte (Annex D.2).
fn pdf_doc_char(b: u8) -> char {
    const HIGH: [u16; 32] = [
        0x2022, 0x2020, 0x2021, 0x2026, 0x2014, 0x2013, 0x0192, 0x2044, 0x2039, 0x203A, 0x2212,
        0x2030, 0x201E, 0x201C, 0x201D, 0x2018, 0x2019, 0x201A, 0x2122, 0xFB01, 0xFB02, 0x0141,
        0x0152, 0x0160, 0x0178, 0x017D, 0x0131, 0x0142, 0x0153, 0x0161, 0x017E, 0xFFFD,
    ];
    match b {
        0x80..=0x9F => char::from_u32(u32::from(HIGH[usize::from(b - 0x80)])).unwrap_or('?'),
        0xA0 => '\u{20AC}',
        _ => char::from(b),
    }
}

/// Encode `text` in `WinAnsiEncoding` (Annex D.2), `?` for a character it
/// lacks.
fn encode_win_ansi(text: &str) -> Vec<u8> {
    const HIGH: [(char, u8); 27] = [
        ('\u{20AC}', 0x80),
        ('\u{201A}', 0x82),
        ('\u{0192}', 0x83),
        ('\u{201E}', 0x84),
        ('\u{2026}', 0x85),
        ('\u{2020}', 0x86),
        ('\u{2021}', 0x87),
        ('\u{02C6}', 0x88),
        ('\u{2030}', 0x89),
        ('\u{0160}', 0x8A),
        ('\u{2039}', 0x8B),
        ('\u{0152}', 0x8C),
        ('\u{017D}', 0x8E),
        ('\u{2018}', 0x91),
        ('\u{2019}', 0x92),
        ('\u{201C}', 0x93),
        ('\u{201D}', 0x94),
        ('\u{2022}', 0x95),
        ('\u{2013}', 0x96),
        ('\u{2014}', 0x97),
        ('\u{02DC}', 0x98),
        ('\u{2122}', 0x99),
        ('\u{0161}', 0x9A),
        ('\u{203A}', 0x9B),
        ('\u{0153}', 0x9C),
        ('\u{017E}', 0x9E),
        ('\u{0178}', 0x9F),
    ];
    text.chars()
        .map(|c| match u32::from(c) {
            v @ (0x0A | 0x0D | 0x20..=0x7E | 0xA0..=0xFF) => u8::try_from(v).unwrap_or(b'?'),
            _ => HIGH.iter().find(|(h, _)| *h == c).map_or(b'?', |(_, b)| *b),
        })
        .collect()
}

/// A content-stream hexadecimal string of `bytes`.
fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2 + 2);
    s.push('<');
    for b in bytes {
        let _ = write!(s, "{b:02X}");
    }
    s.push('>');
    s
}

/// The widget, then each field above it through `/Parent`, as resolved
/// dictionaries: the order in which an inherited attribute is looked up.
fn field_chain(doc: &Document, annot: &Dictionary) -> Vec<Dictionary> {
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

/// `obj` as a string's bytes, following an indirect reference.
fn string(doc: &Document, obj: Option<&Object>) -> Option<Vec<u8>> {
    match obj? {
        Object::String(s, _) => Some(s.clone()),
        Object::Reference(id) => match doc.get_object(*id).ok()?.as_ref() {
            Object::String(s, _) => Some(s.clone()),
            _ => None,
        },
        _ => None,
    }
}

/// The array `obj`, following an indirect reference; empty otherwise.
fn resolve_array(doc: &Document, obj: Option<&Object>) -> Vec<Object> {
    match obj {
        Some(Object::Array(a)) => a.clone(),
        Some(Object::Reference(id)) => doc
            .get_object(*id)
            .ok()
            .and_then(|o| o.as_array().map(<[Object]>::to_vec))
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

/// The numbers of the array `obj`.
fn numbers(doc: &Document, obj: Option<&Object>) -> Vec<f64> {
    resolve_array(doc, obj)
        .iter()
        .filter_map(|o| number(doc, Some(o)))
        .collect()
}

/// `obj` as a finite number, following an indirect reference.
fn number(doc: &Document, obj: Option<&Object>) -> Option<f64> {
    match obj? {
        Object::Reference(id) => {
            let referent = doc.get_object(*id).ok()?;
            crate::resources::obj_to_f64(&referent)
        }
        o => crate::resources::obj_to_f64(o),
    }
    .filter(|v| v.is_finite())
}

/// `r` moved in by `d` on every side, collapsing rather than inverting.
fn inset(r: [f64; 4], d: f64) -> [f64; 4] {
    let x0 = r[0] + d;
    let y0 = r[1] + d;
    [x0, y0, (r[2] - d).max(x0), (r[3] - d).max(y0)]
}

/// `v` for a content stream: at most four decimals, no trailing zeros.
fn n(v: f64) -> String {
    let s = format!("{v:.4}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" {
        "0".to_owned()
    } else {
        s.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::{decode_text_string, encode_win_ansi, split_da};

    #[test]
    fn the_default_appearance_splits_into_font_size_and_colours() {
        let (font, size, rest) = split_da(b"/Helv 14 Tf 0 g");
        assert_eq!(
            (font.as_slice(), size, rest.as_str()),
            (&b"Helv"[..], 14.0, "0 g")
        );
        let (font, size, rest) = split_da(b"0 0 1 rg /F1 0 Tf");
        assert_eq!(
            (font.as_slice(), size, rest.as_str()),
            (&b"F1"[..], 0.0, "0 0 1 rg")
        );
        let (font, size, _) = split_da(b"0 g");
        assert_eq!((font.as_slice(), size), (&b"Helv"[..], 0.0));
    }

    #[test]
    fn text_strings_decode_from_utf16_and_pdfdoc() {
        assert_eq!(decode_text_string(b"\xFE\xFF\x00H\x00\xE9"), "H\u{e9}");
        assert_eq!(decode_text_string(b"a\x80b"), "a\u{2022}b");
        assert_eq!(decode_text_string(b"caf\xE9"), "caf\u{e9}");
    }

    #[test]
    fn win_ansi_encodes_latin1_and_the_high_block() {
        assert_eq!(
            encode_win_ansi("caf\u{e9} \u{20AC}\u{2014}"),
            b"caf\xE9 \x80\x97"
        );
        assert_eq!(encode_win_ansi("\u{4E2D}"), b"?");
    }
}
