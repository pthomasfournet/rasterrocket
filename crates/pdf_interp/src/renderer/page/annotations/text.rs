//! Text in synthesised appearances: the default appearance string `/DA`
//! (ISO 32000-2 §12.7.4.3), text strings (§7.9.2.2), their encoding for a
//! simple font, and the layout of lines in a box.

use std::fmt::Write as _;

use super::content::n;

/// Line spacing of laid-out text, in units of the font size.
pub(super) const LEADING: f64 = 1.15;

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

/// Encode `text` in `WinAnsiEncoding` (Annex D.2), `?` for a character it
/// lacks.
pub(super) fn encode_win_ansi(text: &str) -> Vec<u8> {
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
pub(super) fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2 + 2);
    s.push('<');
    for b in bytes {
        let _ = write!(s, "{b:02X}");
    }
    s.push('>');
    s
}

/// The left edge of a line `line_width` wide in `area` for quadding `q`:
/// 0 left, 1 centred, 2 right.
pub(super) fn aligned_x(q: i64, area: [f64; 4], line_width: f64) -> f64 {
    match q {
        1 => f64::midpoint(area[0], area[2]) - line_width / 2.0,
        2 => area[2] - line_width,
        _ => area[0],
    }
}

/// A block of lines in a box: the default appearance's font and colours
/// at `size`, aligned by `quadding`, from the top of `area` down.
pub(super) struct TextBlock<'a> {
    pub da: &'a DefaultAppearance,
    pub size: f64,
    pub quadding: i64,
    pub area: [f64; 4],
}

impl TextBlock<'_> {
    /// Append `text`, broken at line ends and wrapped at spaces to the
    /// area's width; `width` measures a byte string in a font resource, in
    /// text space units per unit font size.  The first baseline is one font
    /// size below the top; lines follow at [`LEADING`] until one falls a
    /// full line below the bottom.
    pub(super) fn write(
        &self,
        s: &mut String,
        text: &[u8],
        width: &mut impl FnMut(&[u8], &[u8]) -> f64,
    ) {
        let (font, size, area) = (&self.da.font, self.size, self.area);
        let max = (area[2] - area[0]) / size;
        let lines = wrap(text, max, |line| width(font, line));
        let mut y = area[3] - size;
        for line in lines {
            if y < area[1] - size {
                break;
            }
            if !line.is_empty() {
                let x = aligned_x(self.quadding, area, width(font, &line) * size);
                self.da.begin_text(s, font, size);
                let _ = writeln!(s, "{} {} Td {} Tj ET", n(x), n(y), hex(&line));
            }
            y = size.mul_add(-LEADING, y);
        }
    }
}

/// `text` broken at line ends and wrapped at spaces so that no line wider
/// than one word exceeds `max`; `width` measures a line.
pub(super) fn wrap(text: &[u8], max: f64, mut width: impl FnMut(&[u8]) -> f64) -> Vec<Vec<u8>> {
    let mut lines: Vec<Vec<u8>> = Vec::new();
    for paragraph in split_paragraphs(text) {
        let mut line: Vec<u8> = Vec::new();
        for word in paragraph.split(|&b| b == b' ') {
            let mut candidate = line.clone();
            if !candidate.is_empty() {
                candidate.push(b' ');
            }
            candidate.extend_from_slice(word);
            if !line.is_empty() && width(&candidate) > max {
                lines.push(std::mem::take(&mut line));
                line.extend_from_slice(word);
            } else {
                line = candidate;
            }
        }
        lines.push(line);
    }
    lines
}

/// The paragraphs of `text`, separated by CR, LF or CR LF.
fn split_paragraphs(text: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < text.len() {
        match text[i] {
            b'\r' => {
                out.push(&text[start..i]);
                i += if text.get(i + 1) == Some(&b'\n') {
                    2
                } else {
                    1
                };
                start = i;
            }
            b'\n' => {
                out.push(&text[start..i]);
                i += 1;
                start = i;
            }
            _ => i += 1,
        }
    }
    out.push(&text[start..]);
    out
}

#[cfg(test)]
mod tests {
    use super::{DefaultAppearance, decode_text_string, encode_win_ansi, wrap};

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

    #[test]
    fn wrapping_breaks_at_line_ends_and_spaces() {
        // One unit per byte.
        let width = |b: &[u8]| f64::from(u32::try_from(b.len()).unwrap_or(u32::MAX));
        let lines = wrap(b"aa bb cc\r\ndd\ree", 5.0, width);
        assert_eq!(lines, [&b"aa bb"[..], b"cc", b"dd", b"ee"]);
        let lines = wrap(b"longword x", 3.0, width);
        assert_eq!(lines, [&b"longword"[..], b"x"]);
    }
}
