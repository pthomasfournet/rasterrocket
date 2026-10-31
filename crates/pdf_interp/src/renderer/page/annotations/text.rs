//! Text in synthesised appearances: the default appearance string `/DA`
//! (ISO 32000-2 §12.7.4.3), the resources it selects a font from, text
//! strings (§7.9.2.2) and rich text strings, their encoding for a simple
//! font, and the layout of lines in a box.

use std::fmt::Write as _;

use pdf::{Dictionary, Document, Object};

use super::content::n;

/// A default appearance string split into its `Tf` font and size and the
/// remaining operators (the text colour).
#[derive(Clone, Debug, PartialEq)]
pub(super) struct DefaultAppearance {
    /// The font resource name; `Helv` when the string has no `Tf`.
    pub font: Vec<u8>,
    /// The font size; 0 asks for automatic sizing.
    pub size: f64,
    /// The operators other than `Tf`, space-separated.
    pub rest: String,
}

impl DefaultAppearance {
    /// Split the default appearance string `da`.
    pub(super) fn parse(da: &[u8]) -> Self {
        let text = String::from_utf8_lossy(da);
        let tokens: Vec<&str> = text.split_ascii_whitespace().collect();
        let Some(tf) = tokens.iter().rposition(|t| *t == "Tf") else {
            return Self {
                font: b"Helv".to_vec(),
                size: 0.0,
                rest: tokens.join(" "),
            };
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
        Self {
            font,
            size,
            rest: rest.join(" "),
        }
    }

    /// Begin a text object in `font` at `size` with these colours.
    pub(super) fn begin_text(&self, s: &mut String, font: &[u8], size: f64) {
        let _ = writeln!(
            s,
            "BT /{} {} Tf {}",
            String::from_utf8_lossy(font),
            n(size),
            self.rest
        );
    }

    /// The components of the last `g`, `rg` or `k` operator, the colour
    /// text is filled with; `None` when there is none.
    pub(super) fn colour(&self) -> Option<Vec<f64>> {
        let tokens: Vec<&str> = self.rest.split_ascii_whitespace().collect();
        let (i, count) = tokens
            .iter()
            .enumerate()
            .rev()
            .find_map(|(i, t)| match *t {
                "g" => Some((i, 1)),
                "rg" => Some((i, 3)),
                "k" => Some((i, 4)),
                _ => None,
            })?;
        tokens[i.checked_sub(count)?..i]
            .iter()
            .map(|t| t.parse::<f64>().ok().filter(|v| v.is_finite()))
            .collect()
    }
}

/// The resources text is drawn with: `dr`, a copy of the form's default
/// resources, with each `(name, base font)` of `standard` added to its
/// fonts as that standard font when it lacks the name.  Helvetica is given
/// `WinAnsiEncoding`.
pub(super) fn resources(
    doc: &Document,
    dr: Option<&Dictionary>,
    standard: &[(&[u8], &[u8])],
) -> Dictionary {
    let mut resources = dr.cloned().unwrap_or_default();
    let mut fonts = resources
        .get(b"Font")
        .and_then(|o| crate::resources::resolve_dict(doc, o))
        .unwrap_or_default();
    for &(name, base_font) in standard {
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
    super::encoder::add_fallback_fonts(&mut fonts);
    resources.set(b"Font", Object::Dictionary(fonts));
    resources
}

/// The plain text of a rich text string (§12.7.4.3.4), an XHTML fragment:
/// markup removed, character references resolved, white space collapsed,
/// and a line break at each paragraph (`p`, `div`, `li`) and `br`.
pub(super) fn plain_text(rich: &str) -> String {
    let mut out = String::new();
    let mut rest = rich;
    let mut pending_space = false;
    while !rest.is_empty() {
        if let Some(after) = rest.strip_prefix('<') {
            let end = after.find('>').map_or(after.len(), |e| e + 1);
            let tag = &after[..end.saturating_sub(1)];
            rest = &after[end..];
            let name = tag
                .trim_start_matches('/')
                .split(|c: char| c.is_ascii_whitespace() || c == '/')
                .next()
                .unwrap_or("");
            let name = name.rsplit(':').next().unwrap_or(name).to_ascii_lowercase();
            let opening = !tag.starts_with('/');
            let breaks = match name.as_str() {
                "br" => true,
                "p" | "div" | "li" => opening && !out.is_empty(),
                _ => false,
            };
            if breaks {
                if name == "br" || !out.ends_with('\n') {
                    out.push('\n');
                }
                pending_space = false;
            }
            continue;
        }
        let end = rest.find('<').unwrap_or(rest.len());
        for c in decode_entities(&rest[..end]).chars() {
            if c.is_whitespace() && c != '\u{a0}' {
                pending_space = !out.is_empty() && !out.ends_with('\n');
            } else {
                if pending_space {
                    out.push(' ');
                    pending_space = false;
                }
                out.push(c);
            }
        }
        rest = &rest[end..];
    }
    out
}

/// Greatest offset of the `;` of a reference from its `&` that
/// [`decode_entities`] resolves; `&#x10FFFF;` needs 9.
const MAX_SEMICOLON_OFFSET: usize = 10;

/// `s` with the XML character references and the predefined entities
/// resolved; an unknown reference is kept as written.
fn decode_entities(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        rest = &rest[amp..];
        let Some(semi) = rest
            .bytes()
            .take(MAX_SEMICOLON_OFFSET + 1)
            .position(|b| b == b';')
        else {
            out.push('&');
            rest = &rest[1..];
            continue;
        };
        let entity = &rest[1..semi];
        let ch = match entity {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            "nbsp" => Some('\u{a0}'),
            _ => entity
                .strip_prefix("#x")
                .or_else(|| entity.strip_prefix("#X"))
                .map_or_else(
                    || entity.strip_prefix('#').and_then(|d| d.parse().ok()),
                    |h| u32::from_str_radix(h, 16).ok(),
                )
                .and_then(char::from_u32),
        };
        if let Some(c) = ch {
            out.push(c);
            rest = &rest[semi + 1..];
        } else {
            out.push('&');
            rest = &rest[1..];
        }
    }
    out.push_str(rest);
    out
}

/// Decode a PDF text string (§7.9.2.2): UTF-16BE or UTF-8 with a byte order
/// mark, otherwise `PDFDocEncoding`.
pub(super) fn decode_text_string(bytes: &[u8]) -> String {
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

#[cfg(test)]
mod tests {
    use super::{DefaultAppearance, decode_text_string, plain_text};

    #[test]
    fn the_default_appearance_splits_into_font_size_and_colours() {
        let da = DefaultAppearance::parse(b"/Helv 14 Tf 0 g");
        assert_eq!(
            (da.font.as_slice(), da.size, da.rest.as_str()),
            (&b"Helv"[..], 14.0, "0 g")
        );
        let da = DefaultAppearance::parse(b"0 0 1 rg /F1 0 Tf");
        assert_eq!(
            (da.font.as_slice(), da.size, da.rest.as_str()),
            (&b"F1"[..], 0.0, "0 0 1 rg")
        );
        let da = DefaultAppearance::parse(b"0 g");
        assert_eq!((da.font.as_slice(), da.size), (&b"Helv"[..], 0.0));
    }

    #[test]
    fn the_text_colour_is_the_last_fill_colour_operator() {
        let colour = |da: &[u8]| DefaultAppearance::parse(da).colour();
        assert_eq!(colour(b"0 0 1 rg /F1 8 Tf"), Some(vec![0.0, 0.0, 1.0]));
        assert_eq!(colour(b"/F1 8 Tf 0.5 g"), Some(vec![0.5]));
        assert_eq!(
            colour(b"1 g 0 0 0 1 k /F 0 Tf"),
            Some(vec![0.0, 0.0, 0.0, 1.0])
        );
        assert_eq!(colour(b"/F1 8 Tf"), None);
        assert_eq!(colour(b"rg"), None);
    }

    #[test]
    fn rich_text_reduces_to_paragraphs_of_plain_text() {
        let rc = "<?xml version=\"1.0\"?><body xmlns=\"http://www.w3.org/1999/xhtml\">\
                  <p dir=\"ltr\">Rich <span style=\"font-weight:bold\">bold</span>\n  \
                  &amp; &#x41;&#66;</p><p>second<br/>third</p></body>";
        assert_eq!(plain_text(rc), "Rich bold & AB\nsecond\nthird");
        assert_eq!(plain_text("plain"), "plain");
    }

    #[test]
    fn text_strings_decode_from_utf16_and_pdfdoc() {
        assert_eq!(decode_text_string(b"\xFE\xFF\x00H\x00\xE9"), "H\u{e9}");
        assert_eq!(decode_text_string(b"a\x80b"), "a\u{2022}b");
        assert_eq!(decode_text_string(b"caf\xE9"), "caf\u{e9}");
    }
}
