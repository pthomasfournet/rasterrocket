//! Rich text strings (ISO 32000-2 §12.7.4.3.4) and default style strings
//! (`/DS`, §12.7.4.3.5), read into styled paragraphs.
//!
//! A rich text string is an XHTML fragment.  Its paragraphs are the
//! contents of `p`, `div` and `li` elements and the text between `br`
//! elements; `b` and `strong` set bold, `i` and `em` italic, and the
//! `style` attribute of any element holds CSS declarations.  A default
//! style string holds the declarations applied to the whole text.  The
//! declarations read are `font` (style, weight, size and family),
//! `font-size`, `font-weight`, `font-style`, `font-family`, `color` and
//! `text-align`; others are ignored.
//!
//! A family is reduced to sans-serif, serif or monospace and set in the
//! standard Helvetica, Times or Courier face of the requested weight and
//! style, through the resource names of [`STANDARD_FONTS`].  Text in the
//! family, weight and style of the default appearance keeps its font.
//! White space collapses to single spaces, as in HTML.

use super::content::colour_op;
use super::text::decode_entities;
use super::typeset::{Paragraph, Style};

/// The resource names of the standard faces, by family (sans-serif, serif,
/// monospace) and then regular, bold, italic and bold italic, with their
/// base fonts.
pub(super) const STANDARD_FONTS: [[(&[u8], &[u8]); 4]; 3] = [
    [
        (b"Helv", b"Helvetica"),
        (b"HeBo", b"Helvetica-Bold"),
        (b"HeOb", b"Helvetica-Oblique"),
        (b"HeBO", b"Helvetica-BoldOblique"),
    ],
    [
        (b"TiRo", b"Times-Roman"),
        (b"TiBo", b"Times-Bold"),
        (b"TiIt", b"Times-Italic"),
        (b"TiBI", b"Times-BoldItalic"),
    ],
    [
        (b"Cour", b"Courier"),
        (b"CoBo", b"Courier-Bold"),
        (b"CoOb", b"Courier-Oblique"),
        (b"CoBO", b"Courier-BoldOblique"),
    ],
];

/// The base font of the standard resource name `name`.
pub(super) fn standard_base_font(name: &[u8]) -> Option<&'static [u8]> {
    STANDARD_FONTS
        .iter()
        .flatten()
        .find(|(n, _)| *n == name)
        .map(|(_, base)| *base)
}

/// Largest font size a declaration may set.
const MAX_SIZE: f64 = 1000.0;

/// A family, as an index into [`STANDARD_FONTS`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    Sans = 0,
    Serif = 1,
    Mono = 2,
}

/// The computed style of an element.
#[derive(Clone, Debug, PartialEq)]
struct Computed {
    family: Family,
    bold: bool,
    italic: bool,
    size: f64,
    /// The fill colour operator; `None` for the default appearance's.
    colour: Option<String>,
    quadding: i64,
}

/// The base style of styled text: the default appearance's font, size and
/// operators, and the alignment of the text.
pub(super) struct Base<'a> {
    pub style: &'a Style,
    pub quadding: i64,
}

impl Base<'_> {
    /// The computed style of the default appearance.
    fn computed(&self) -> Computed {
        let (family, bold, italic) = STANDARD_FONTS
            .iter()
            .enumerate()
            .find_map(|(f, faces)| {
                let face = faces.iter().position(|(n, _)| *n == self.style.font)?;
                Some((family_at(f), face & 1 != 0, face & 2 != 0))
            })
            .unwrap_or((Family::Sans, false, false));
        Computed {
            family,
            bold,
            italic,
            size: self.style.size,
            colour: None,
            quadding: self.quadding,
        }
    }

    /// The style of text computed as `c`.
    fn style(&self, c: &Computed, base: &Computed) -> Style {
        let same_face = (c.family, c.bold, c.italic) == (base.family, base.bold, base.italic);
        let font = if same_face {
            self.style.font.clone()
        } else {
            let face = usize::from(c.bold) | usize::from(c.italic) << 1;
            STANDARD_FONTS[c.family as usize][face].0.to_vec()
        };
        let paint = c.colour.as_ref().map_or_else(
            || self.style.paint.clone(),
            |colour| {
                if self.style.paint.is_empty() {
                    colour.clone()
                } else {
                    format!("{} {colour}", self.style.paint)
                }
            },
        );
        Style {
            font,
            size: c.size,
            paint,
        }
    }
}

const fn family_at(index: usize) -> Family {
    match index {
        1 => Family::Serif,
        2 => Family::Mono,
        _ => Family::Sans,
    }
}

/// The style and alignment of plain text over `base` with the default
/// style string `ds` applied.
pub(super) fn default_style(ds: Option<&str>, base: &Base<'_>) -> (Style, i64) {
    let root = base.computed();
    let top = with_default_style(&root, ds);
    (base.style(&top, &root), top.quadding)
}

/// `root` with the default style string `ds` applied.
fn with_default_style(root: &Computed, ds: Option<&str>) -> Computed {
    let mut top = root.clone();
    if let Some(ds) = ds {
        apply(&mut top, ds);
    }
    top
}

/// The paragraphs of the rich text string `rich` over `base`, with the
/// default style string `ds` applied first.  Elements nested deeper than
/// [`MAX_DEPTH`] take the style of their deepest enclosing element.
pub(super) fn paragraphs(rich: &str, ds: Option<&str>, base: &Base<'_>) -> Vec<Paragraph> {
    let root = base.computed();
    let top = with_default_style(&root, ds);
    let mut stack: Vec<(String, Computed)> = Vec::new();
    let mut out = vec![Paragraph {
        quadding: top.quadding,
        style: base.style(&top, &root),
        spans: Vec::new(),
    }];
    // The style of a collapsed run of white space not yet written: the
    // style it began in.
    let mut pending_space: Option<Style> = None;
    // A `p`, `div` or `li` element has ended: text that follows begins a
    // paragraph.
    let mut block_ended = false;
    let mut rest = rich;
    while !rest.is_empty() {
        let current = stack.last().map_or(&top, |(_, c)| c);
        if let Some(after) = rest.strip_prefix('<') {
            let (tag, tail) = after.split_once('>').unwrap_or((after, ""));
            rest = tail;
            let Some(element) = Tag::parse(tag) else {
                continue;
            };
            let block = matches!(element.name.as_str(), "p" | "div" | "li");
            if element.closing {
                if let Some(at) = stack.iter().rposition(|(n, _)| *n == element.name) {
                    stack.truncate(at);
                }
                block_ended |= block;
                continue;
            }
            let mut computed = current.clone();
            match element.name.as_str() {
                "b" | "strong" => computed.bold = true,
                "i" | "em" => computed.italic = true,
                _ => {}
            }
            if let Some(style) = element.style {
                apply(&mut computed, &style);
            }
            let paragraph = out.last_mut().expect("at least one paragraph");
            if element.name == "br" || (block && !paragraph.spans.is_empty()) {
                let quadding = if block {
                    computed.quadding
                } else {
                    paragraph.quadding
                };
                out.push(Paragraph {
                    quadding,
                    style: base.style(&computed, &root),
                    spans: Vec::new(),
                });
                pending_space = None;
                block_ended = false;
            } else if block {
                paragraph.quadding = computed.quadding;
                paragraph.style = base.style(&computed, &root);
                block_ended = false;
            }
            if element.name != "br" && !element.self_closing && stack.len() < MAX_DEPTH {
                stack.push((element.name, computed));
            }
            continue;
        }
        let end = rest.find('<').unwrap_or(rest.len());
        let style = base.style(current, &root);
        for c in decode_entities(&rest[..end]).chars() {
            let paragraph = out.last_mut().expect("at least one paragraph");
            if c.is_whitespace() && c != '\u{a0}' {
                if !paragraph.spans.is_empty() && pending_space.is_none() {
                    pending_space = Some(style.clone());
                }
                continue;
            }
            if block_ended && !paragraph.spans.is_empty() {
                out.push(Paragraph {
                    quadding: current.quadding,
                    style: style.clone(),
                    spans: Vec::new(),
                });
                pending_space = None;
            }
            block_ended = false;
            let paragraph = out.last_mut().expect("at least one paragraph");
            if let Some(space) = pending_space.take() {
                push_text(paragraph, &space, ' ');
            }
            push_text(paragraph, &style, c);
        }
        rest = &rest[end..];
    }
    out
}

/// Greatest depth of nested elements whose styles are kept.
const MAX_DEPTH: usize = 256;

/// Append `c` in `style` to `paragraph`, to its last span when that is
/// in the same style.
fn push_text(paragraph: &mut Paragraph, style: &Style, c: char) {
    match paragraph.spans.last_mut() {
        Some((last, text)) if last == style => text.push(c),
        _ => paragraph.spans.push((style.clone(), c.to_string())),
    }
}

/// A start or end tag.
struct Tag {
    /// The local name, lower case.
    name: String,
    closing: bool,
    self_closing: bool,
    /// The value of the `style` attribute.
    style: Option<String>,
}

impl Tag {
    /// The tag whose text between `<` and `>` is `tag`; `None` for a
    /// comment, declaration or processing instruction.
    fn parse(tag: &str) -> Option<Self> {
        if tag.starts_with(['!', '?']) {
            return None;
        }
        let closing = tag.starts_with('/');
        let body = tag.trim_start_matches('/');
        let self_closing = body.ends_with('/');
        let body = body.trim_end_matches('/');
        let (name, attributes) = body
            .split_once(|c: char| c.is_ascii_whitespace())
            .unwrap_or((body, ""));
        let name = name.rsplit(':').next().unwrap_or(name).to_ascii_lowercase();
        Some(Self {
            name,
            closing,
            self_closing,
            style: attribute(attributes, "style").map(decode_entities),
        })
    }
}

/// The value of the attribute `name` (its local name, any case) in
/// `attributes`: quoted with `"` or `'`, or unquoted up to white space.
/// An attribute without a value is skipped.
fn attribute<'a>(attributes: &'a str, name: &str) -> Option<&'a str> {
    let mut rest = attributes.trim_start();
    while !rest.is_empty() {
        let key_end = rest
            .find(|c: char| c == '=' || c.is_ascii_whitespace())
            .unwrap_or(rest.len());
        let key = &rest[..key_end];
        rest = rest[key_end..].trim_start();
        let Some(after) = rest.strip_prefix('=') else {
            continue;
        };
        let after = after.trim_start();
        let (value, tail) = match after.chars().next() {
            Some(quote @ ('"' | '\'')) => {
                let quoted = &after[1..];
                quoted.split_once(quote).unwrap_or((quoted, ""))
            }
            _ => after
                .split_once(|c: char| c.is_ascii_whitespace())
                .unwrap_or((after, "")),
        };
        rest = tail.trim_start();
        if key
            .rsplit(':')
            .next()
            .unwrap_or(key)
            .eq_ignore_ascii_case(name)
        {
            return Some(value);
        }
    }
    None
}

/// Apply the CSS declarations `css` to `c`.
fn apply(c: &mut Computed, css: &str) {
    for declaration in css.split(';') {
        let Some((property, value)) = declaration.split_once(':') else {
            continue;
        };
        let value = value.trim();
        match property.trim().to_ascii_lowercase().as_str() {
            "font" => font_shorthand(c, value),
            "font-size" => {
                if let Some(size) = size(value, c.size) {
                    c.size = size;
                }
            }
            "font-weight" => c.bold = bold(value).unwrap_or(c.bold),
            "font-style" => c.italic = italic(value).unwrap_or(c.italic),
            "font-family" => c.family = family(value),
            "color" => {
                if let Some(colour) = colour(value) {
                    c.colour = Some(colour);
                }
            }
            "text-align" => {
                c.quadding = match value.to_ascii_lowercase().as_str() {
                    "center" => 1,
                    "right" => 2,
                    "left" | "justify" | "start" => 0,
                    _ => c.quadding,
                };
            }
            _ => {}
        }
    }
}

/// Apply the `font` shorthand `value`: optional style and weight keywords,
/// then a size (optionally `/line-height`) and the family list.
fn font_shorthand(c: &mut Computed, value: &str) {
    let mut tokens = value.split_ascii_whitespace();
    for token in tokens.by_ref() {
        if let Some(i) = italic(token) {
            c.italic = i;
        } else if let Some(b) = bold(token) {
            c.bold = b;
        } else if let Some(sz) = size(token.split('/').next().unwrap_or(token), c.size) {
            c.size = sz;
            break;
        }
    }
    let rest: Vec<&str> = tokens.collect();
    if !rest.is_empty() {
        c.family = family(&rest.join(" "));
    }
}

/// The size `value` relative to the parent's `parent`: points, pixels (at
/// 96 per inch), ems or a percentage; `None` when it is not a positive
/// size.
fn size(value: &str, parent: f64) -> Option<f64> {
    let value = value.trim().to_ascii_lowercase();
    let (number, scale) = [
        ("pt", 1.0),
        ("px", 0.75),
        ("em", parent),
        ("%", parent / 100.0),
        ("", 1.0),
    ]
    .into_iter()
    .find_map(|(unit, scale)| Some((value.strip_suffix(unit)?, scale)))?;
    let v = number.trim().parse::<f64>().ok()? * scale;
    (v.is_finite() && v > 0.0).then_some(v.min(MAX_SIZE))
}

/// Whether the weight `value` is bold; `None` when it is no weight.
fn bold(value: &str) -> Option<bool> {
    match value.to_ascii_lowercase().as_str() {
        "bold" | "bolder" => Some(true),
        "normal" | "lighter" => Some(false),
        v => v
            .parse::<u16>()
            .ok()
            .filter(|w| (1..=1000).contains(w))
            .map(|w| w >= 600),
    }
}

/// Whether the style `value` is italic; `None` when it is no style.
fn italic(value: &str) -> Option<bool> {
    match value.to_ascii_lowercase().as_str() {
        "italic" | "oblique" => Some(true),
        "normal" => Some(false),
        _ => None,
    }
}

/// The family of the first entry of the family list `value` that names a
/// serif or monospace face or a sans-serif one; sans-serif otherwise.
fn family(value: &str) -> Family {
    for name in value.split(',') {
        let name = name.trim().trim_matches(['"', '\'']).to_ascii_lowercase();
        if name.contains("mono") || name.contains("courier") || name.contains("consolas") {
            return Family::Mono;
        }
        if name.contains("sans") || name.contains("helvetica") || name.contains("arial") {
            return Family::Sans;
        }
        if name.contains("serif") || name.contains("times") || name.contains("georgia") {
            return Family::Serif;
        }
    }
    Family::Sans
}

/// The fill colour operator of the CSS colour `value`: `#rgb`, `#rrggbb`,
/// `rgb(r, g, b)` in integers or percentages, or one of the sixteen basic
/// colour keywords.
fn colour(value: &str) -> Option<String> {
    let value = value.trim().to_ascii_lowercase();
    let rgb: [f64; 3] = if let Some(hex) = value.strip_prefix('#') {
        let digits: Vec<u8> = hex
            .chars()
            .map(|c| c.to_digit(16).and_then(|d| u8::try_from(d).ok()))
            .collect::<Option<_>>()?;
        match *digits.as_slice() {
            [r, g, b] => [r, g, b].map(|d| f64::from(d * 17) / 255.0),
            [r1, r0, g1, g0, b1, b0] => {
                [(r1, r0), (g1, g0), (b1, b0)].map(|(h, l)| f64::from(h * 16 + l) / 255.0)
            }
            _ => return None,
        }
    } else if let Some(args) = value.strip_prefix("rgb(").and_then(|v| v.strip_suffix(')')) {
        let parts: Vec<f64> = args
            .split(',')
            .map(|p| {
                let p = p.trim();
                p.strip_suffix('%').map_or_else(
                    || p.parse::<f64>().ok().map(|v| v / 255.0),
                    |pct| pct.trim().parse::<f64>().ok().map(|v| v / 100.0),
                )
            })
            .collect::<Option<_>>()?;
        let [r, g, b] = *parts.as_slice() else {
            return None;
        };
        [r, g, b].map(|v| {
            if v.is_finite() {
                v.clamp(0.0, 1.0)
            } else {
                0.0
            }
        })
    } else {
        let hex: u32 = match value.as_str() {
            "black" => 0x00_0000,
            "silver" => 0xC0_C0C0,
            "gray" | "grey" => 0x80_8080,
            "white" => 0xFF_FFFF,
            "maroon" => 0x80_0000,
            "red" => 0xFF_0000,
            "purple" => 0x80_0080,
            "fuchsia" | "magenta" => 0xFF_00FF,
            "green" => 0x00_8000,
            "lime" => 0x00_FF00,
            "olive" => 0x80_8000,
            "yellow" => 0xFF_FF00,
            "navy" => 0x00_0080,
            "blue" => 0x00_00FF,
            "teal" => 0x00_8080,
            "aqua" | "cyan" => 0x00_FFFF,
            _ => return None,
        };
        hex.to_be_bytes()[1..]
            .iter()
            .map(|&c| f64::from(c) / 255.0)
            .collect::<Vec<_>>()
            .try_into()
            .ok()?
    };
    colour_op(&rgb, false)
}

#[cfg(test)]
mod tests {
    use super::{Base, paragraphs, standard_base_font};
    use crate::renderer::page::annotations::typeset::Style;

    fn base_style() -> Style {
        Style {
            font: b"Helv".to_vec(),
            size: 10.0,
            paint: "0 g".to_owned(),
        }
    }

    /// A span as (font, size, paint, text).
    type Span = (String, f64, String, String);

    /// The spans of `rich` over 10 pt black Helvetica, left aligned, with
    /// each paragraph's quadding.
    fn spans(rich: &str, ds: Option<&str>) -> Vec<(i64, Vec<Span>)> {
        let style = base_style();
        let base = Base {
            style: &style,
            quadding: 0,
        };
        paragraphs(rich, ds, &base)
            .into_iter()
            .map(|p| {
                (
                    p.quadding,
                    p.spans
                        .into_iter()
                        .map(|(s, t)| {
                            (
                                String::from_utf8(s.font).expect("ASCII"),
                                s.size,
                                s.paint,
                                t,
                            )
                        })
                        .collect(),
                )
            })
            .collect()
    }

    #[test]
    fn bold_italic_size_and_colour_select_faces_and_operators() {
        let p = spans(
            "<body><p>plain <b>bold <i>both</i></b> \
             <span style=\"font-size:14pt;color:#FF0000\">big red</span></p></body>",
            None,
        );
        assert_eq!(p.len(), 1);
        let s = &p[0].1;
        assert_eq!(s[0], ("Helv".into(), 10.0, "0 g".into(), "plain ".into()));
        assert_eq!(s[1], ("HeBo".into(), 10.0, "0 g".into(), "bold ".into()));
        assert_eq!(s[2], ("HeBO".into(), 10.0, "0 g".into(), "both".into()));
        assert_eq!(s[3], ("Helv".into(), 10.0, "0 g".into(), " ".into()));
        assert_eq!(
            s[4],
            ("Helv".into(), 14.0, "0 g 1 0 0 rg".into(), "big red".into())
        );
    }

    #[test]
    fn paragraphs_break_at_blocks_and_take_their_alignment() {
        let p = spans(
            "<p style='text-align:center'>one</p><p>two<br/>three</p>",
            Some("text-align:right"),
        );
        let texts: Vec<(i64, &str)> = p.iter().map(|(q, s)| (*q, s[0].3.as_str())).collect();
        assert_eq!(texts, [(1, "one"), (2, "two"), (2, "three")]);
    }

    #[test]
    fn the_default_style_and_font_shorthand_set_family_and_size() {
        let p = spans(
            "x <span style=\"font: italic bold 12pt/14pt 'Times New Roman', serif\">y</span>",
            Some("font-family: Courier; font-size: 8pt"),
        );
        let s = &p[0].1;
        assert_eq!((s[0].0.as_str(), s[0].1), ("Cour", 8.0));
        assert_eq!((s[1].0.as_str(), s[1].1), ("TiBI", 12.0));
    }

    #[test]
    fn unusable_declarations_are_ignored() {
        let p = spans(
            "<span style='font-size:-3pt;color:#12;font-weight:normal;bogus'>t</span>",
            Some("font-size: 1e999pt"),
        );
        assert_eq!(p[0].1[0], ("Helv".into(), 10.0, "0 g".into(), "t".into()));
    }

    #[test]
    fn units_and_named_colours() {
        let p = spans(
            "<span style='font-size:16px;color:navy'>a</span>\
             <span style='font-size:150%;color:rgb(0,50%,255)'>b</span>",
            None,
        );
        assert_eq!(p[0].1[0].1, 12.0);
        assert_eq!(p[0].1[0].2, "0 g 0 0 0.502 rg");
        assert_eq!(p[0].1[1].1, 15.0);
        assert_eq!(p[0].1[1].2, "0 g 0 0.5 1 rg");
    }

    #[test]
    fn malformed_markup_neither_panics_nor_loses_text() {
        let texts = |rich: &str| -> Vec<String> {
            spans(rich, None)
                .into_iter()
                .map(|(_, s)| s.into_iter().map(|(.., t)| t).collect())
                .collect()
        };
        // A tag left open at the end, even after a multibyte character.
        assert_eq!(texts("a<é"), ["a"]);
        assert_eq!(texts("a<"), ["a"]);
        // Unquoted and valueless attributes do not hide the style.
        let p = spans("<span hidden style=color:red>r</span>", None);
        assert_eq!(p[0].1[0].2, "0 g 1 0 0 rg");
        // Text after a closed block begins a paragraph.
        assert_eq!(texts("<p>a</p>b"), ["a", "b"]);
        // Deep nesting is bounded and keeps the text.
        let deep = format!("{}x{}", "<b>".repeat(10_000), "</b>".repeat(10_000));
        assert_eq!(texts(&deep), ["x"]);
    }

    #[test]
    fn standard_names_map_to_their_base_fonts() {
        assert_eq!(standard_base_font(b"TiBI"), Some(&b"Times-BoldItalic"[..]));
        assert_eq!(standard_base_font(b"F1"), None);
    }
}
