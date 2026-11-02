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
    match visual_order(text, right_to_left(text)) {
        Some(visual) => {
            let visual: String = visual.into_iter().map(|(_, c)| c).collect();
            ts.show(s, font, size, &visual);
        }
        None => ts.show(s, font, size, text),
    }
    s.push_str("ET\n");
}

/// Whether a paragraph of `text` is right to left: its first strong
/// character is (UAX #9 rules P2 and P3).  Text without one is left to
/// right.
pub(super) fn right_to_left(text: &str) -> bool {
    unicode_bidi::get_base_direction(text) == unicode_bidi::Direction::Rtl
}

/// The characters of the line `text` in visual order (UAX #9 rules L1 to
/// L4) in a paragraph that is right to left when `rtl`, each with the byte
/// offset of the character it shows; a character at a right-to-left level
/// shows its mirrored form when it has one.  `None` when the visual order
/// is the logical order.
fn visual_order(text: &str, rtl: bool) -> Option<Vec<(usize, char)>> {
    use unicode_bidi::{Level, ParagraphBidiInfo};
    let info = ParagraphBidiInfo::new(text, Some(if rtl { Level::rtl() } else { Level::ltr() }));
    if !info.has_rtl() {
        return None;
    }
    let (levels, runs) = info.visual_runs(0..text.len());
    let mut visual = Vec::with_capacity(text.len());
    for run in runs {
        let chars = text[run.clone()]
            .char_indices()
            .map(|(i, c)| (run.start + i, c));
        if levels[run.start].is_rtl() {
            visual.extend(
                chars
                    .rev()
                    .map(|(i, c)| (i, unicode_bidi_mirroring::get_mirrored(c).unwrap_or(c))),
            );
        } else {
            visual.extend(chars);
        }
    }
    Some(visual)
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

/// The font, size and colour a span of text is set in.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Style {
    /// The font resource name.
    pub font: Vec<u8>,
    pub size: f64,
    /// The operators other than `Tf` that apply to the text, such as its
    /// fill colour; may be empty.
    pub paint: String,
}

impl Style {
    /// The font, size and remaining operators of the default appearance
    /// `da`, at `size`.
    pub(super) fn of(da: &DefaultAppearance, size: f64) -> Self {
        Self {
            font: da.font.clone(),
            size,
            paint: da.rest.clone(),
        }
    }

    /// Append the operators selecting this style inside a text object.
    fn select(&self, s: &mut String) {
        let _ = write!(s, "{} {} Tf", name(&self.font), n(self.size));
        if !self.paint.is_empty() {
            let _ = write!(s, " {}", self.paint);
        }
    }
}

/// A paragraph: spans of styled text, aligned by `quadding` (0 left,
/// 1 centred, 2 right).  An empty paragraph is a blank line as high as
/// `style` sets it.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Paragraph {
    pub quadding: i64,
    pub style: Style,
    pub spans: Vec<(Style, String)>,
}

impl Paragraph {
    /// `text` broken into paragraphs at line ends (CR, LF or CR LF), each
    /// one span in `style` aligned by `quadding`.
    pub(super) fn plain(text: &str, style: &Style, quadding: i64) -> Vec<Self> {
        lines(text)
            .map(|line| Self {
                quadding,
                style: style.clone(),
                spans: vec![(style.clone(), line.to_owned())],
            })
            .collect()
    }

    /// The paragraph's text without its styles.
    pub(super) fn text(&self) -> String {
        self.spans.iter().map(|(_, t)| t.as_str()).collect()
    }
}

/// A line laid out from a paragraph: runs of text in one style, its width,
/// and the largest font size on it.
struct LaidLine<'a> {
    runs: Vec<(&'a Style, String)>,
    width: f64,
    size: f64,
}

impl<'a> LaidLine<'a> {
    const fn new(size: f64) -> Self {
        Self {
            runs: Vec::new(),
            width: 0.0,
            size,
        }
    }

    /// Append `text` in `style`, `width` wide, to the line.
    fn push(&mut self, style: &'a Style, text: &str, width: f64) {
        match self.runs.last_mut() {
            Some((last, run)) if *last == style => run.push_str(text),
            _ => self.runs.push((style, text.to_owned())),
        }
        self.width += width;
        self.size = self.size.max(style.size);
    }

    const fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }
}

/// The lines of `paragraph` wrapped at spaces so that no line wider than
/// one word exceeds `max`.  A word is a maximal run of characters without
/// a space, possibly in several styles; a space takes the style it was
/// written in.  The first line is as high as the paragraph's style even
/// when empty.
fn lay_out<'a>(paragraph: &'a Paragraph, max: f64, ts: &mut dyn Typesetter) -> Vec<LaidLine<'a>> {
    let mut wrapper = Wrapper {
        lines: vec![LaidLine::new(paragraph.style.size)],
        word: Vec::new(),
        word_width: 0.0,
        max,
    };
    // The space before the pending word: its style and width.
    let mut space: Option<(&Style, f64)> = None;
    for (style, text) in &paragraph.spans {
        for (i, piece) in text.split(' ').enumerate() {
            if i > 0 {
                wrapper.place_word(space);
                space = Some((style, ts.width(&style.font, " ") * style.size));
            }
            if !piece.is_empty() {
                let w = ts.width(&style.font, piece) * style.size;
                wrapper.word.push((style, piece, w));
                wrapper.word_width += w;
            }
        }
    }
    wrapper.place_word(space);
    wrapper.lines
}

/// The state of [`lay_out`]: the lines so far, and the pieces and width of
/// the word being read.
struct Wrapper<'a> {
    lines: Vec<LaidLine<'a>>,
    word: Vec<(&'a Style, &'a str, f64)>,
    word_width: f64,
    max: f64,
}

impl<'a> Wrapper<'a> {
    /// Place the pending word, preceded by `space` unless it begins a
    /// line: on the last line when it fits or the line is empty, else on a
    /// new line.
    fn place_word(&mut self, space: Option<(&'a Style, f64)>) {
        let line = self.lines.last_mut().expect("at least one line");
        let gap = space.filter(|_| !line.is_empty());
        let gap_width = gap.map_or(0.0, |(_, w)| w);
        if !line.is_empty() && line.width + gap_width + self.word_width > self.max {
            self.lines.push(LaidLine::new(0.0));
        } else if let Some((style, w)) = gap {
            line.push(style, " ", w);
        }
        let line = self.lines.last_mut().expect("at least one line");
        for (style, piece, w) in self.word.drain(..) {
            line.push(style, piece, w);
        }
        self.word_width = 0.0;
    }
}

/// `runs`, the styled text of a line in logical order, in visual order
/// (see [`visual_order`]) in a paragraph that is right to left when `rtl`.
fn visual_runs<'a>(runs: &[(&'a Style, String)], rtl: bool) -> Vec<(&'a Style, String)> {
    let text: String = runs.iter().map(|(_, t)| t.as_str()).collect();
    let Some(visual) = visual_order(&text, rtl) else {
        return runs.to_vec();
    };
    // The byte offset in `text` at which each run ends.
    let ends: Vec<usize> = runs
        .iter()
        .scan(0, |end, (_, t)| {
            *end += t.len();
            Some(*end)
        })
        .collect();
    let mut out: Vec<(&Style, String)> = Vec::new();
    for (offset, c) in visual {
        let style = runs[ends.partition_point(|&end| end <= offset)].0;
        match out.last_mut() {
            Some((last, t)) if std::ptr::eq(*last, style) => t.push(c),
            _ => out.push((style, c.to_string())),
        }
    }
    out
}

/// Append the runs of `line` in a new text object, its baseline starting
/// at `(x, y)`, in visual order in a paragraph that is right to left when
/// `rtl`.
fn show_runs(
    s: &mut String,
    ts: &mut dyn Typesetter,
    runs: &[(&Style, String)],
    (x, y): (f64, f64),
    rtl: bool,
) {
    let mut first = true;
    for (style, text) in &visual_runs(runs, rtl) {
        if first {
            s.push_str("BT ");
            style.select(s);
            let _ = write!(s, "\n{} {} Td ", n(x), n(y));
            first = false;
        } else {
            style.select(s);
            s.push(' ');
        }
        ts.show(s, &style.font, style.size, text);
    }
    if !first {
        s.push_str("ET\n");
    }
}

/// Append `paragraphs` in `area`, each wrapped at spaces to the area's
/// width and aligned by its quadding.  The first baseline is the first
/// line's largest font size below the top, and each following one
/// [`LEADING`] times its own line's largest size below the one before;
/// lines stop when one falls a full line below the bottom.
pub(super) fn write_block(
    s: &mut String,
    paragraphs: &[Paragraph],
    area: [f64; 4],
    ts: &mut dyn Typesetter,
) {
    let max = area[2] - area[0];
    let mut y = area[3];
    let mut first = true;
    for paragraph in paragraphs {
        let rtl = right_to_left(&paragraph.text());
        for line in lay_out(paragraph, max, ts) {
            y -= if first {
                line.size
            } else {
                LEADING * line.size
            };
            first = false;
            if y < area[1] - line.size {
                return;
            }
            if !line.is_empty() {
                let x = aligned_x(paragraph.quadding, area, line.width);
                show_runs(s, ts, &line.runs, (x, y), rtl);
            }
        }
    }
}

/// Append `paragraphs` as one line, the paragraphs joined by spaces and
/// not wrapped, aligned in `area` by `quadding` and centred vertically:
/// the body of the line, taken as 0.7 of its largest font size above the
/// baseline, is centred.
pub(super) fn write_single_line(
    s: &mut String,
    paragraphs: &[Paragraph],
    quadding: i64,
    area: [f64; 4],
    ts: &mut dyn Typesetter,
) {
    let joined = Paragraph {
        quadding,
        style: paragraphs.first().map_or_else(
            || Style {
                font: Vec::new(),
                size: 0.0,
                paint: String::new(),
            },
            |p| p.style.clone(),
        ),
        spans: paragraphs
            .iter()
            .enumerate()
            .flat_map(|(i, p)| {
                let gap = (i > 0).then(|| (p.style.clone(), " ".to_owned()));
                gap.into_iter().chain(p.spans.iter().cloned())
            })
            .collect(),
    };
    let lines = lay_out(&joined, f64::INFINITY, ts);
    let Some(line) = lines.first().filter(|l| !l.is_empty()) else {
        return;
    };
    let x = aligned_x(quadding, area, line.width);
    let y = 0.35f64.mul_add(-line.size, f64::midpoint(area[1], area[3]));
    show_runs(s, ts, &line.runs, (x, y), right_to_left(&joined.text()));
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
    /// area's width, as [`write_block`] lays it out.
    pub(super) fn write(&self, s: &mut String, text: &str, ts: &mut dyn Typesetter) {
        let style = Style::of(self.da, self.size);
        write_block(
            s,
            &Paragraph::plain(text, &style, self.quadding),
            self.area,
            ts,
        );
    }
}

/// The lines of `text`, broken at line ends: CR, LF or CR LF.
pub(super) fn lines(text: &str) -> impl Iterator<Item = &str> {
    text.split("\r\n").flat_map(|p| p.split(['\r', '\n']))
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
    use super::{
        Paragraph, Run, Style, Typesetter, WinAnsiTypesetter, name, right_to_left, visual_order,
        visual_runs, write_block,
    };

    /// The visual order of `text` as a string.
    fn visual(text: &str, rtl: bool) -> Option<String> {
        visual_order(text, rtl).map(|v| v.into_iter().map(|(_, c)| c).collect())
    }

    #[test]
    fn left_to_right_text_keeps_its_order() {
        assert_eq!(visual("abc def", false), None);
        assert!(!right_to_left("abc \u{05D0}"));
        assert!(right_to_left("12 \u{05D0}bc"), "digits are not strong");
    }

    #[test]
    fn right_to_left_text_is_reversed_and_numbers_are_not() {
        assert_eq!(
            visual("abc \u{05D0}\u{05D1}\u{05D2}", false).as_deref(),
            Some("abc \u{05D2}\u{05D1}\u{05D0}")
        );
        assert_eq!(
            visual("\u{05D0}\u{05D1} 12", true).as_deref(),
            Some("12 \u{05D1}\u{05D0}")
        );
    }

    #[test]
    fn brackets_at_a_right_to_left_level_are_mirrored() {
        assert_eq!(
            visual("\u{05D0}(\u{05D1})", true).as_deref(),
            Some("(\u{05D1})\u{05D0}")
        );
    }

    #[test]
    fn styles_follow_their_characters_into_visual_order() {
        let (a, b) = (style(b"A", 10.0), style(b"B", 10.0));
        let mixed = [(&a, "ab ".to_owned()), (&b, "\u{05D0}\u{05D1}".to_owned())];
        assert_eq!(
            visual_runs(&mixed, false),
            [(&a, "ab ".to_owned()), (&b, "\u{05D1}\u{05D0}".to_owned())]
        );
        let hebrew = [
            (&a, "\u{05D0}\u{05D1}".to_owned()),
            (&b, "\u{05D2}".to_owned()),
        ];
        assert_eq!(
            visual_runs(&hebrew, true),
            [
                (&b, "\u{05D2}".to_owned()),
                (&a, "\u{05D1}\u{05D0}".to_owned())
            ]
        );
    }

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

    fn style(font: &[u8], size: f64) -> Style {
        Style {
            font: font.to_vec(),
            size,
            paint: String::new(),
        }
    }

    /// The text shown by each text object of `content`, decoded from the
    /// hexadecimal strings of [`WinAnsiTypesetter`], and its `Td` origin.
    fn shown(content: &str) -> Vec<(String, String)> {
        content
            .split("BT ")
            .skip(1)
            .map(|object| {
                let td = object
                    .split_once('\n')
                    .and_then(|(_, rest)| rest.split_once(" Td"))
                    .map_or_else(String::new, |(xy, _)| xy.to_owned());
                let text = object
                    .split('<')
                    .skip(1)
                    .filter_map(|h| h.split_once('>'))
                    .flat_map(|(h, _)| {
                        h.as_bytes()
                            .chunks(2)
                            .map(|b| u8::from_str_radix(std::str::from_utf8(b).unwrap_or(""), 16))
                            .collect::<Result<Vec<_>, _>>()
                            .unwrap_or_default()
                    })
                    .map(char::from)
                    .collect();
                (text, td)
            })
            .collect()
    }

    /// The lines `write_block` sets `paragraphs` in, `width` wide.
    fn block(paragraphs: &[Paragraph], width: f64) -> Vec<(String, String)> {
        let mut s = String::new();
        write_block(
            &mut s,
            paragraphs,
            [0.0, -100.0, width, 0.0],
            &mut WinAnsiTypesetter,
        );
        shown(&s)
    }

    #[test]
    fn plain_text_breaks_at_line_ends_and_wraps_at_spaces() {
        // Each character is half an em wide; at 1 pt, "aa bb" is 2.5 wide.
        let text = |t: &str, width: f64| -> Vec<String> {
            block(&Paragraph::plain(t, &style(b"F", 1.0), 0), width)
                .into_iter()
                .map(|(t, _)| t)
                .collect()
        };
        assert_eq!(text("aa bb cc\r\ndd\ree", 2.5), ["aa bb", "cc", "dd", "ee"]);
        assert_eq!(text("longword x", 1.5), ["longword", "x"]);
        assert_eq!(text("a  b", 10.0), ["a  b"], "runs of spaces are kept");
    }

    #[test]
    fn a_word_in_two_styles_stays_whole_and_the_largest_size_sets_the_line() {
        let small = style(b"F", 1.0);
        let large = style(b"F", 4.0);
        let paragraph = Paragraph {
            quadding: 0,
            style: small.clone(),
            spans: vec![
                (small.clone(), "x bo".to_owned()),
                (large, "ld".to_owned()),
                (small, " y".to_owned()),
            ],
        };
        // "bo" is 1 wide and "ld" at 4 pt is 4: the word "bold", 5 wide,
        // does not fit after "x " in 5 and is not broken after "bo".
        let lines = block(&[paragraph], 5.0);
        let texts: Vec<&str> = lines.iter().map(|(t, _)| t.as_str()).collect();
        assert_eq!(texts, ["x", "bold", "y"]);
        // Each baseline is 1.15 times its own line's largest size below
        // the one before; the first is that size below the top.
        let origins: Vec<&str> = lines.iter().map(|(_, o)| o.as_str()).collect();
        assert_eq!(origins, ["0 -1", "0 -5.6", "0 -6.75"]);
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
