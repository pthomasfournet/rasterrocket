//! Shaping of text drawn by substitute faces.
//!
//! A font the document does not embed is drawn by an installed or bundled
//! face, one glyph per code.  For scripts whose letters change form with
//! their neighbours or are reordered (Arabic joining, Indic conjuncts and
//! vowel signs, mark placement), the codes of a run in such a script are
//! instead shaped as one text with the drawing face's layout tables.
//!
//! A run is a maximal sequence of at least two shown codes drawn by the
//! same face whose characters belong to one script, with the combining
//! marks and joiners among them, before them included.  A code's text is
//! the one or two characters its font maps it to.  Show strings carry
//! right-to-left text in visual order and left-to-right text in logical
//! order, so a right-to-left run's codes are reversed into logical order
//! before shaping, each code's own characters kept in theirs; the shaped
//! glyphs come back in visual order.  A run never crosses a show string,
//! so a word split between two strings is shaped as two runs.

use font::face::FontFace;
use font::program::{ShapeError, ShapedGlyph};

use crate::renderer::font_cache::PageFace;
use crate::resources::cmap::DecodedCode;

/// The drawing glyph of a shown code: its face, its selector there and
/// whether the selector is a glyph index (see [`PageFace::glyph_for`]).
pub(super) type Drawn<'f> = (&'f FontFace, u32, bool);

/// A run of shown codes `start..end` drawn as `glyphs` from `face`.
pub(super) struct ShapedRun<'f> {
    pub start: usize,
    pub end: usize,
    pub face: &'f FontFace,
    pub glyphs: Vec<ShapedGlyph>,
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

/// The shaped runs of a shown string whose codes are `codes` and whose
/// per-code drawing glyphs are `drawn`, for a string shown with `face`.
/// Empty unless `face` is a substitute that knows its codes' characters.
///
/// A run is left glyph by glyph when its face is not an sfnt, when a
/// shaped glyph has no selector, when shaping yields `.notdef`, which a
/// per-code fallback may still draw, or when the shaper reads no font from
/// the program, refuses the run or exceeds its limits.  The middle two are
/// logged at `debug`, the last three at `warn`.
pub(super) fn shaped_runs<'f>(
    face: &PageFace,
    codes: &[DecodedCode],
    drawn: &[Option<Drawn<'f>>],
) -> Vec<ShapedRun<'f>> {
    debug_assert_eq!(codes.len(), drawn.len());
    if !face.substitute {
        return Vec::new();
    }
    let texts: Vec<Option<(char, Option<char>)>> =
        codes.iter().map(|d| face.text(d.code, d.value)).collect();
    let classes: Vec<Option<Class>> = texts
        .iter()
        .map(|text| text.and_then(|(first, _)| class(first)))
        .collect();
    let mut runs = Vec::new();
    let mut start = 0;
    while start < codes.len() {
        let Some((run_face, ..)) = drawn[start] else {
            start += 1;
            continue;
        };
        let same_face = |i: usize| drawn[i].is_some_and(|(f, ..)| std::ptr::eq(f, run_face));
        // Leading marks join the script that follows them.
        let first_letter = (start..codes.len())
            .take_while(|&i| same_face(i))
            .find(|&i| classes[i] != Some(Class::Inherited));
        let Some(script @ Class::Shaped { rtl, .. }) = first_letter.and_then(|i| classes[i]) else {
            start = first_letter.map_or(start + 1, |i| i.max(start + 1));
            continue;
        };
        let after_first = first_letter.map_or(start, |i| i + 1);
        let end = (after_first..codes.len())
            .find(|&i| {
                !same_face(i)
                    || !matches!(classes[i], Some(Class::Inherited)) && classes[i] != Some(script)
            })
            .unwrap_or(codes.len());
        if end - start >= 2
            && let Some(glyphs) = shape(run_face, &texts[start..end], rtl)
        {
            runs.push(ShapedRun {
                start,
                end,
                face: run_face,
                glyphs,
            });
        }
        start = end;
    }
    runs
}

/// The glyphs of the codes whose texts are `texts`, shown in visual order,
/// shaped with `run_face`; `None` when a code's text is unknown or the
/// face cannot shape them.
fn shape(
    run_face: &FontFace,
    texts: &[Option<(char, Option<char>)>],
    rtl: bool,
) -> Option<Vec<ShapedGlyph>> {
    let in_logical_order = |texts: &mut dyn Iterator<Item = &Option<(char, Option<char>)>>| {
        texts
            .map(|&text| text.map(|(first, second)| std::iter::once(first).chain(second)))
            .collect::<Option<Vec<_>>>()
            .map(|texts| texts.into_iter().flatten().collect::<String>())
    };
    let text = if rtl {
        in_logical_order(&mut texts.iter().rev())
    } else {
        in_logical_order(&mut texts.iter())
    }?;
    match run_face.program().shape(&text, rtl) {
        Ok(glyphs) if glyphs.iter().all(|g| g.selector != 0) => Some(glyphs),
        Ok(_) => {
            log::debug!("shaping {text:?} yields .notdef; drawn glyph by glyph");
            None
        }
        Err(ShapeError::NotSfnt) => None,
        Err(e @ ShapeError::NoSelector(_)) => {
            log::debug!("shaping {text:?}: {e}; drawn glyph by glyph");
            None
        }
        Err(e @ (ShapeError::NoShaperFont | ShapeError::Refused(_) | ShapeError::Exhausted)) => {
            log::warn!("shaping {text:?}: {e}; drawn glyph by glyph");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Class, class};

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
