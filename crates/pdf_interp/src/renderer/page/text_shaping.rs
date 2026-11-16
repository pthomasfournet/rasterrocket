//! Shaping of text drawn by substitute faces.
//!
//! A font the document does not embed is drawn by an installed or bundled
//! face, one glyph per code.  For scripts whose letters change form with
//! their neighbours or are reordered (Arabic joining, Indic conjuncts and
//! vowel signs, mark placement), the codes of a run in such a script are
//! instead shaped as one text with the drawing face's layout tables.
//!
//! A run is a maximal sequence of shown codes drawn by the same face whose
//! characters belong to one script, with the combining marks and joiners
//! among them, before them included.  It is shaped when it holds at least
//! two codes or a code next to it, in its string or a neighbouring one,
//! could itself be part of a run.  A code's text is
//! the one or two characters its font maps it to.  Show strings carry
//! right-to-left text in visual order and left-to-right text in logical
//! order, so a right-to-left run's codes are reversed into logical order
//! before shaping, each code's own characters kept in theirs; the shaped
//! glyphs come back in visual order.
//!
//! A run never crosses a show string, but the characters around it are
//! its shaping context: those of the neighbouring codes of its string and,
//! at the string's ends, those of the strings shown next to it on its
//! baseline (see [`Neighbours`]).  A word split between two strings is
//! thereby joined across the split; its parts are drawn as one chain (see
//! [`super::text_layout`]).
//!
//! Right-to-left visual order stores a combining mark before its letter.
//! The marks that end a right-to-left string whose neighbour after starts
//! with a right-to-left letter of the same script belong to that letter:
//! they are shaped with it as a run of their own, drawn over the letter
//! where the string after draws it.

use font::face::FontFace;
use font::program::{ShapeError, ShapedGlyph};
use unicode_bidi::{BidiClass, bidi_class};

use crate::renderer::font_cache::PageFace;
use crate::resources::font::FontDescriptor;

/// The text of a shown code: the one or two characters its font maps it
/// to, `None` when unknown.
pub(super) type CodeText = Option<(char, Option<char>)>;

/// The codes read as context on each side of a run: as many as the shaper
/// reads characters.
pub(super) const CONTEXT_CODES: usize = 5;

/// The texts of the codes shown next to a string on its baseline, in
/// storage order: `before` ends with the code shown just before the
/// string, `after` starts with the code shown just after it.  Each holds
/// at most [`CONTEXT_CODES`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct Neighbours {
    pub before: Vec<CodeText>,
    pub after: Vec<CodeText>,
}

/// The drawing glyph of a shown code: its face, its selector there and
/// whether the selector is a glyph index (see [`PageFace::glyph_for`]).
pub(super) type Drawn<'f> = (&'f FontFace, u32, bool);

/// A run of shown codes `start..end` drawn as `glyphs` from `face`, which
/// continues the code before it when `joins.0` and the code after it when
/// `joins.1`: codes of its string drawn by another face, or of the strings
/// shown next to it, that can be part of a run.
///
/// When `over_next`, the codes are the marks that end their string and
/// `glyphs` the marks alone, each displaced from the origin of the first
/// letter of the string after, which they belong to, and without advance.
pub(super) struct ShapedRun<'f> {
    pub start: usize,
    pub end: usize,
    pub face: &'f FontFace,
    pub glyphs: Vec<ShapedGlyph>,
    pub joins: (bool, bool),
    pub over_next: bool,
}

/// How a character takes part in a shaped run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class {
    /// A letter of a script that needs shaping, written right to left
    /// (`true`) or left to right, the script named by the first code point
    /// of its main block.
    Shaped { rtl: bool, script: u32 },
    /// A combining mark, variation selector or joiner: part of the run
    /// around it.
    Inherited,
}

/// The [`Class`] of `c`; `None` for a character of a script drawn glyph
/// by glyph, which ends a run.
///
/// The right-to-left scripts are Hebrew, Arabic, Syriac, Thaana, N'Ko,
/// Samaritan, Mandaic, Hanifi Rohingya and Adlam; the left-to-right ones
/// the Indic scripts, Sinhala, Thai, Lao, Tibetan, Myanmar, Khmer,
/// Mongolian, the Brahmic scripts of Southeast Asia and those of the
/// Supplementary Multilingual Plane.  A script's supplement blocks share
/// its name.  Arabic presentation forms are already shaped and are
/// excluded.
const fn class(c: char) -> Option<Class> {
    const fn rtl(script: u32) -> Class {
        Class::Shaped { rtl: true, script }
    }
    const fn ltr(script: u32) -> Class {
        Class::Shaped { rtl: false, script }
    }
    let u = c as u32;
    Some(match u {
        0x0300..=0x036F
        | 0x1AB0..=0x1AFF
        | 0x1CD0..=0x1CFF
        | 0x1DC0..=0x1DFF
        | 0x200C
        | 0x200D
        | 0x20D0..=0x20FF
        | 0xFE00..=0xFE0F
        | 0xFE20..=0xFE2F
        | 0xE0100..=0xE01EF => Class::Inherited,
        0x0590..=0x05FF => rtl(0x0590),
        0x0600..=0x06FF | 0x0750..=0x077F | 0x0870..=0x08FF | 0x10EC0..=0x10EFF => rtl(0x0600),
        0x0700..=0x074F | 0x0860..=0x086F => rtl(0x0700),
        0x0780..=0x07BF => rtl(0x0780),
        0x07C0..=0x07FF => rtl(0x07C0),
        0x0800..=0x083F => rtl(0x0800),
        0x0840..=0x085F => rtl(0x0840),
        0x10D00..=0x10D3F => rtl(0x10D00),
        0x1E900..=0x1E95F => rtl(0x1E900),
        0x0900..=0x097F | 0xA8E0..=0xA8FF => ltr(0x0900),
        // One script per 128-code-point block: Bengali to Lao, and the
        // Brahmic scripts of the Supplementary Multilingual Plane.
        0x0980..=0x0EFF | 0x11000..=0x11FFF => ltr(u & !0x7F),
        0x0F00..=0x0FFF => ltr(0x0F00),
        0x1000..=0x109F | 0xA9E0..=0xA9FF | 0xAA60..=0xAA7F => ltr(0x1000),
        0x1780..=0x17FF => ltr(0x1780),
        0x1800..=0x18AF => ltr(0x1800),
        0x1A20..=0x1AAF => ltr(0x1A20),
        0x1B00..=0x1B7F => ltr(0x1B00),
        0x1B80..=0x1BBF => ltr(0x1B80),
        0x1BC0..=0x1BFF => ltr(0x1BC0),
        0x1C00..=0x1C4F => ltr(0x1C00),
        0xA980..=0xA9DF => ltr(0xA980),
        0xAA00..=0xAA5F => ltr(0xAA00),
        0xAA80..=0xAADF => ltr(0xAA80),
        0xABC0..=0xABFF => ltr(0xABC0),
        _ => return None,
    })
}

/// Whether text shown with the font `descriptor` can be shaped: the font
/// is no Type 3 font and writes horizontally.
pub(super) fn font_shapes(descriptor: &FontDescriptor) -> bool {
    descriptor.type3.is_none()
        && !descriptor
            .cid_encoding
            .as_ref()
            .is_some_and(|enc| enc.vertical)
}

/// Whether text shown with the font `descriptor` and drawn by `face` is
/// shaped: the font can be shaped and `face` is a substitute.
pub(super) fn shapes(descriptor: &FontDescriptor, face: &PageFace) -> bool {
    font_shapes(descriptor) && face.substitute
}

/// Whether a code of text `text` can be part of a run: a letter of a
/// script that needs shaping, a combining mark or a joiner.
pub(super) fn takes_context(text: CodeText) -> bool {
    text.is_some_and(|(first, _)| class(first).is_some())
}

/// The texts of the last codes, at most [`CONTEXT_CODES`], of a string of
/// codes of the texts `texts` shown after codes of the texts `before`.
pub(super) fn last_texts(before: &[CodeText], texts: &[CodeText]) -> Vec<CodeText> {
    let all = before.len() + texts.len();
    let skip = all.saturating_sub(CONTEXT_CODES);
    before.iter().chain(texts).skip(skip).copied().collect()
}

/// Whether a code of text `text` is a combining mark: its first character
/// is a nonspacing mark (bidirectional class NSM).
fn is_mark(text: CodeText) -> bool {
    text.is_some_and(|(first, _)| bidi_class(first) == BidiClass::NSM)
}

/// Whether a code of text `text` is a right-to-left letter: its first
/// character has bidirectional class R or AL.
fn is_rtl_letter(text: CodeText) -> bool {
    text.is_some_and(|(first, _)| matches!(bidi_class(first), BidiClass::R | BidiClass::AL))
}

/// The shaped runs of a shown string whose codes have the texts `texts`
/// and the drawing glyphs `drawn`, for a string shown with `face` between
/// `neighbours`.  Empty unless `face` is a substitute that knows its
/// codes' characters.
///
/// A run is left glyph by glyph when its face is not an sfnt, when a
/// shaped glyph has no selector, when shaping yields `.notdef`, which a
/// per-code fallback may still draw, or when the shaper reads no font from
/// the program, refuses the run or exceeds its limits.  The last two are
/// logged at `warn`, the others but the first at `debug`; a program the
/// shaper reads no font from is reported at `warn` once by the font crate.
pub(super) fn shaped_runs<'f>(
    face: &PageFace,
    texts: &[CodeText],
    drawn: &[Option<Drawn<'f>>],
    neighbours: &Neighbours,
) -> Vec<ShapedRun<'f>> {
    debug_assert_eq!(texts.len(), drawn.len());
    if !face.substitute {
        return Vec::new();
    }
    let classes: Vec<Option<Class>> = texts
        .iter()
        .map(|text| text.and_then(|(first, _)| class(first)))
        .collect();
    let mut runs = Vec::new();
    let mut start = 0;
    while start < texts.len() {
        let Some((run_face, ..)) = drawn[start] else {
            start += 1;
            continue;
        };
        let same_face = |i: usize| drawn[i].is_some_and(|(f, ..)| std::ptr::eq(f, run_face));
        // Leading marks join the script that follows them.
        let first_letter = (start..texts.len())
            .take_while(|&i| same_face(i))
            .find(|&i| classes[i] != Some(Class::Inherited));
        let Some(script @ Class::Shaped { rtl, .. }) = first_letter.and_then(|i| classes[i]) else {
            start = first_letter.map_or(start + 1, |i| i.max(start + 1));
            continue;
        };
        let after_first = first_letter.map_or(start, |i| i + 1);
        let end = (after_first..texts.len())
            .find(|&i| {
                !same_face(i)
                    || !matches!(classes[i], Some(Class::Inherited)) && classes[i] != Some(script)
            })
            .unwrap_or(texts.len());
        // The known texts of the codes on each side, nearest first.
        let known = |codes: &mut dyn Iterator<Item = &CodeText>| -> Vec<(char, Option<char>)> {
            codes.take(CONTEXT_CODES).map_while(|&text| text).collect()
        };
        // The marks that end a right-to-left string belong to the letter
        // of the run's script the neighbour after starts with; the run ends
        // before them.
        let next_letter = neighbours
            .after
            .first()
            .copied()
            .flatten()
            .filter(|&letter| {
                rtl && end == texts.len()
                    && is_rtl_letter(Some(letter))
                    && class(letter.0) == Some(script)
            });
        let marks = next_letter.map_or(0, |_| {
            texts[start..end]
                .iter()
                .rev()
                .take_while(|&&text| is_mark(text))
                .count()
        });
        let (end, all) = (end - marks, end);
        let before = known(
            &mut texts[..start]
                .iter()
                .rev()
                .chain(neighbours.before.iter().rev()),
        );
        let after = known(&mut texts[end..].iter().chain(&neighbours.after));
        // Whether the code just before and the code just after the run, of
        // its string (drawn by another face) or of a neighbouring one, can
        // be part of a run; a lone code takes a contextual form only then.
        let in_context = |code: Option<&CodeText>| code.is_some_and(|&t| takes_context(t));
        let joins = (
            in_context(texts[..start].last().or_else(|| neighbours.before.last())),
            in_context(texts.get(end).or_else(|| neighbours.after.first())),
        );
        if end > start
            && (end - start >= 2 || joins.0 || joins.1)
            && let Some(glyphs) = shape(run_face, &texts[start..end], &before, &after, rtl)
        {
            runs.push(ShapedRun {
                start,
                end,
                face: run_face,
                glyphs,
                joins,
                over_next: false,
            });
        }
        if let Some(letter) = next_letter.filter(|_| marks > 0) {
            let before = known(
                &mut texts[..end]
                    .iter()
                    .rev()
                    .chain(neighbours.before.iter().rev()),
            );
            let after = known(&mut neighbours.after.iter().skip(1));
            if let Some(glyphs) = marks_over(run_face, &texts[end..all], letter, &before, &after) {
                runs.push(ShapedRun {
                    start: end,
                    end: all,
                    face: run_face,
                    glyphs,
                    joins: (false, false),
                    over_next: true,
                });
            }
        }
        start = all;
    }
    runs
}

/// The glyphs of the marks whose texts are `marks`, stored before the
/// letter of text `letter` that the string after starts with, shaped with
/// that letter by `face` between codes of the texts `before` and `after`,
/// each listed nearest first: the marks alone, each displaced from the
/// letter's origin, without advance.  `None` when the marks and the letter
/// are not shaped or the letter is not the one glyph with an advance.
fn marks_over(
    face: &FontFace,
    marks: &[CodeText],
    letter: (char, Option<char>),
    before: &[(char, Option<char>)],
    after: &[(char, Option<char>)],
) -> Option<Vec<ShapedGlyph>> {
    let mut texts = marks.to_vec();
    texts.push(Some(letter));
    let glyphs = shape(face, &texts, before, after, true)?;
    let mut pen = 0.0;
    let placed: Vec<(f64, &ShapedGlyph)> = glyphs
        .iter()
        .map(|g| {
            let x = pen + g.offset.0;
            pen += g.advance;
            (x, g)
        })
        .collect();
    let mut letters = placed.iter().filter(|(_, g)| g.advance.abs() > 1e-6);
    let &(origin, _) = letters.next()?;
    if letters.next().is_some() {
        return None;
    }
    Some(
        placed
            .iter()
            .filter(|(_, g)| g.advance.abs() <= 1e-6)
            .map(|&(x, g)| ShapedGlyph {
                selector: g.selector,
                advance: 0.0,
                offset: (x - origin, g.offset.1),
            })
            .collect(),
    )
}

/// The glyphs of the codes whose texts are `texts`, stored between codes
/// of the texts `before` and `after`, each listed nearest first, shaped
/// with `run_face`; `None` when a code's text is unknown or the face
/// cannot shape them.
fn shape(
    run_face: &FontFace,
    texts: &[CodeText],
    before: &[(char, Option<char>)],
    after: &[(char, Option<char>)],
    rtl: bool,
) -> Option<Vec<ShapedGlyph>> {
    let joined = |texts: &mut dyn Iterator<Item = &(char, Option<char>)>| {
        texts
            .flat_map(|&(first, second)| std::iter::once(first).chain(second))
            .collect::<String>()
    };
    let texts = texts.iter().copied().collect::<Option<Vec<_>>>()?;
    // Right-to-left codes are stored in visual order: the logical text
    // reads them in reverse, so the codes stored after the run precede it.
    let (text, pre, post) = if rtl {
        (
            joined(&mut texts.iter().rev()),
            joined(&mut after.iter().rev()),
            joined(&mut before.iter()),
        )
    } else {
        (
            joined(&mut texts.iter()),
            joined(&mut before.iter().rev()),
            joined(&mut after.iter()),
        )
    };
    match run_face.program().shape_in_context(&text, &pre, &post, rtl) {
        Ok(glyphs) if glyphs.iter().all(|g| g.selector != 0) => Some(glyphs),
        Ok(_) => {
            log::debug!(
                "shaping {} yields .notdef; drawn glyph by glyph",
                Excerpt(&text)
            );
            None
        }
        Err(ShapeError::NotSfnt) => None,
        Err(e @ (ShapeError::NoSelector(_) | ShapeError::NoShaperFont)) => {
            log::debug!("shaping {}: {e}; drawn glyph by glyph", Excerpt(&text));
            None
        }
        Err(e @ (ShapeError::Refused(_) | ShapeError::Exhausted)) => {
            log::warn!("shaping {}: {e}; drawn glyph by glyph", Excerpt(&text));
            None
        }
    }
}

/// A run's text as a log message shows it: its first [`Excerpt::CHARS`]
/// characters, quoted, then the number of characters when there are more.
struct Excerpt<'a>(&'a str);

impl Excerpt<'_> {
    /// The characters shown.
    const CHARS: usize = 32;
}

impl std::fmt::Display for Excerpt<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0.char_indices().nth(Self::CHARS) {
            None => write!(f, "{:?}", self.0),
            Some((cut, _)) => write!(
                f,
                "{:?}… ({} characters)",
                &self.0[..cut],
                self.0.chars().count()
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Class, Excerpt, class};

    #[test]
    fn an_excerpt_cuts_a_long_run() {
        assert_eq!(
            Excerpt("\u{0628}\u{064A}").to_string(),
            "\"\u{0628}\u{064A}\""
        );
        let exact = "a".repeat(Excerpt::CHARS);
        assert_eq!(Excerpt(&exact).to_string(), format!("{exact:?}"));
        let long = "\u{0628}".repeat(Excerpt::CHARS + 8);
        let shown = "\u{0628}".repeat(Excerpt::CHARS);
        assert_eq!(
            Excerpt(&long).to_string(),
            format!("{shown:?}… ({} characters)", Excerpt::CHARS + 8)
        );
    }

    #[test]
    fn scripts_take_their_direction_and_marks_inherit() {
        let rtl = |script| Some(Class::Shaped { rtl: true, script });
        let ltr = |script| Some(Class::Shaped { rtl: false, script });
        assert_eq!(class('\u{0628}'), rtl(0x0600));
        assert_eq!(
            class('\u{08A0}'),
            rtl(0x0600),
            "Arabic Extended-A is Arabic"
        );
        assert_eq!(class('\u{05D0}'), rtl(0x0590));
        assert_eq!(class('\u{1E900}'), rtl(0x1E900), "Adlam");
        assert_eq!(class('\u{0915}'), ltr(0x0900));
        assert_eq!(class('\u{A8F2}'), ltr(0x0900), "Devanagari Extended");
        assert_eq!(class('\u{0995}'), ltr(0x0980), "Bengali is its own script");
        assert_eq!(class('\u{0E01}'), ltr(0x0E00));
        assert_eq!(class('\u{1CD0}'), Some(Class::Inherited), "Vedic tone mark");
        assert_eq!(
            class('\u{FE0F}'),
            Some(Class::Inherited),
            "variation selector"
        );
        assert_eq!(class('\u{0301}'), Some(Class::Inherited));
        assert_eq!(class('\u{200D}'), Some(Class::Inherited));
        assert_eq!(class('a'), None);
        assert_eq!(class(' '), None);
        assert_eq!(
            class('\u{FEFB}'),
            None,
            "presentation forms are already shaped"
        );
    }
}
