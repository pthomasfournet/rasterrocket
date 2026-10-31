//! Setting Unicode text in synthesised appearances: runs of character codes
//! in font resources, their advances, and the layout of lines in a box.
//!
//! Text is laid out as Unicode and encoded only when shown, through a
//! [`Typesetter`]: a character the requested font cannot encode is set in
//! a fallback font, so one line may switch fonts between runs.

use std::fmt::Write as _;

use super::content::n;
use super::text::DefaultAppearance;

/// Line spacing of laid-out text, in units of the font size.
pub(super) const LEADING: f64 = 1.15;

/// Font size of multi-line text whose `/DA` asks for automatic sizing.
pub(super) const AUTO_MULTILINE_SIZE: f64 = 12.0;

/// A run of text in one font resource: the resource name and the character
/// codes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Run {
    pub font: Vec<u8>,
    pub codes: Vec<u8>,
}

/// The fonts that set text in a synthesised appearance.
pub(super) trait Typesetter {
    /// `text` encoded for the font resource `font`: runs in `font`, and
    /// runs in fallback fonts for the characters `font` cannot encode.
    fn runs(&mut self, font: &[u8], text: &str) -> Vec<Run>;

    /// The advance of `codes` in the font resource `font`, in text space
    /// units per unit font size.
    fn advance(&mut self, font: &[u8], codes: &[u8]) -> f64;

    /// The advance of `text` set in `font`.
    fn width(&mut self, font: &[u8], text: &str) -> f64 {
        self.runs(font, text)
            .iter()
            .map(|run| self.advance(&run.font, &run.codes))
            .sum()
    }

    /// Append `text` set in `font` at `size` to a text object whose
    /// current font is `font`: a `Tj` per run, a `Tf` before each run in
    /// another font.  The current font is `font` again afterwards.
    fn show(&mut self, s: &mut String, font: &[u8], size: f64, text: &str) {
        let mut current = font.to_vec();
        for run in self.runs(font, text) {
            if run.font != current {
                let _ = write!(s, "{} {} Tf ", name(&run.font), n(size));
                current.clone_from(&run.font);
            }
            let _ = write!(s, "{} Tj ", hex(&run.codes));
        }
        if current != font {
            let _ = write!(s, "{} {} Tf ", name(font), n(size));
        }
    }
}

/// Append one line of `text` in a new text object: in the default
/// appearance's colours, set in `font` at `size`, its baseline starting at
/// `(x, y)`.
pub(super) fn show_line(
    s: &mut String,
    ts: &mut dyn Typesetter,
    da: &DefaultAppearance,
    font: &[u8],
    size: f64,
    (x, y): (f64, f64),
    text: &str,
) {
    da.begin_text(s, font, size);
    let _ = write!(s, "{} {} Td ", n(x), n(y));
    ts.show(s, font, size, text);
    s.push_str("ET\n");
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

/// A content-stream name object `/…` of `bytes`, writing as `#xx` every
/// byte outside `!`..=`~`, every delimiter and `#` itself (PDF §7.3.5), so
/// that the content-stream tokenizer reads back exactly `bytes`.
pub(super) fn name(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() + 1);
    s.push('/');
    for &b in bytes {
        if b.is_ascii_graphic() && !b"()<>[]{}/%#".contains(&b) {
            s.push(char::from(b));
        } else {
            let _ = write!(s, "#{b:02X}");
        }
    }
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
    /// area's width.  The first baseline is one font size below the top;
    /// lines follow at [`LEADING`] until one falls a full line below the
    /// bottom.
    pub(super) fn write(&self, s: &mut String, text: &str, ts: &mut dyn Typesetter) {
        let (font, size, area) = (&self.da.font, self.size, self.area);
        let max = (area[2] - area[0]) / size;
        let lines = wrap(text, max, |line| ts.width(font, line));
        let mut y = area[3] - size;
        for line in lines {
            if y < area[1] - size {
                break;
            }
            if !line.is_empty() {
                let x = aligned_x(self.quadding, area, ts.width(font, &line) * size);
                show_line(s, ts, self.da, font, size, (x, y), &line);
            }
            y = size.mul_add(-LEADING, y);
        }
    }
}

/// `text` broken at line ends (CR, LF or CR LF) and wrapped at spaces so
/// that no line wider than one word exceeds `max`.  `width` measures a
/// word and a space; a line is as wide as its words and the spaces between
/// them, so each word is measured once.
pub(super) fn wrap(text: &str, max: f64, mut width: impl FnMut(&str) -> f64) -> Vec<String> {
    let space = width(" ");
    let mut lines = Vec::new();
    for paragraph in text.split("\r\n").flat_map(|p| p.split(['\r', '\n'])) {
        let mut line = String::new();
        let mut line_width = 0.0;
        for word in paragraph.split(' ') {
            let word_width = width(word);
            if line.is_empty() {
                line.push_str(word);
                line_width = word_width;
            } else if line_width + space + word_width > max {
                lines.push(std::mem::replace(&mut line, word.to_owned()));
                line_width = word_width;
            } else {
                line.push(' ');
                line.push_str(word);
                line_width += space + word_width;
            }
        }
        lines.push(line);
    }
    lines
}

/// A typesetter for tests: every character in `WinAnsiEncoding` (`?` for
/// one it lacks) in the requested font, half an em wide.
#[cfg(test)]
pub(super) struct WinAnsiTypesetter;

#[cfg(test)]
impl Typesetter for WinAnsiTypesetter {
    fn runs(&mut self, font: &[u8], text: &str) -> Vec<Run> {
        vec![Run {
            font: font.to_vec(),
            codes: encode_win_ansi(text),
        }]
    }

    fn advance(&mut self, _font: &[u8], codes: &[u8]) -> f64 {
        f64::from(u32::try_from(codes.len()).unwrap_or(u32::MAX)) / 2.0
    }
}

/// Encode `text` in `WinAnsiEncoding` (Annex D.2), `?` for a character it
/// lacks.
#[cfg(test)]
fn encode_win_ansi(text: &str) -> Vec<u8> {
    text.chars()
        .map(|c| match u32::from(c) {
            v @ (0x20..=0x7E | 0xA0..=0xFF) => u8::try_from(v).unwrap_or(b'?'),
            0x2014 => 0x97,
            _ => b'?',
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{Run, Typesetter, WinAnsiTypesetter, name, wrap};

    #[test]
    fn a_written_name_reads_back_as_the_same_bytes() {
        let raw = b"Helv Bold/#(1)\x80";
        let written = name(raw);
        assert_eq!(written, "/Helv#20Bold#2F#23#281#29#80");
        let tokens: Vec<_> = crate::content::Tokenizer::new(written.as_bytes()).collect();
        assert_eq!(
            tokens,
            [crate::content::tokenizer::Token::Name(
                raw.as_slice().into()
            )]
        );
    }

    #[test]
    fn wrapping_breaks_at_line_ends_and_spaces() {
        let width = |t: &str| f64::from(u32::try_from(t.chars().count()).unwrap_or(0));
        let lines = wrap("aa bb cc\r\ndd\ree", 5.0, width);
        assert_eq!(lines, ["aa bb", "cc", "dd", "ee"]);
        let lines = wrap("longword x", 3.0, width);
        assert_eq!(lines, ["longword", "x"]);
        let lines = wrap("日本 語", 2.0, width);
        assert_eq!(lines, ["日本", "語"]);
        let lines = wrap("a  b", 10.0, width);
        assert_eq!(lines, ["a  b"], "runs of spaces are kept");
    }

    #[test]
    fn wrapping_measures_each_word_once() {
        let text = "w ".repeat(1000);
        let mut measured = 0usize;
        let lines = wrap(&text, f64::INFINITY, |t| {
            measured += t.len();
            1.0
        });
        assert_eq!(lines.len(), 1);
        assert!(measured <= 2 * text.len(), "{measured} bytes measured");
    }

    /// Sets digits in a second font.
    struct TwoFonts;

    impl Typesetter for TwoFonts {
        fn runs(&mut self, font: &[u8], text: &str) -> Vec<Run> {
            let mut runs: Vec<Run> = Vec::new();
            for c in text.chars() {
                let f = if c.is_ascii_digit() {
                    b"D".to_vec()
                } else {
                    font.to_vec()
                };
                match runs.last_mut() {
                    Some(run) if run.font == f => run.codes.push(c as u8),
                    _ => runs.push(Run {
                        font: f,
                        codes: vec![c as u8],
                    }),
                }
            }
            runs
        }

        fn advance(&mut self, font: &[u8], codes: &[u8]) -> f64 {
            let per = if font == b"D" { 1.0 } else { 0.5 };
            per * f64::from(u32::try_from(codes.len()).unwrap_or(0))
        }
    }

    #[test]
    fn showing_switches_fonts_between_runs_and_back() {
        let mut s = String::new();
        TwoFonts.show(&mut s, b"F", 10.0, "ab12c");
        assert_eq!(s, "<6162> Tj /D 10 Tf <3132> Tj /F 10 Tf <63> Tj ");
        assert!((TwoFonts.width(b"F", "ab12c") - 3.5).abs() < 1e-12);
        let mut s = String::new();
        WinAnsiTypesetter.show(&mut s, b"F", 10.0, "x\u{2014}");
        assert_eq!(s, "<7897> Tj ");
    }
}
