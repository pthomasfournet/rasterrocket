//! Shaping of text drawn by substitute faces.
//!
//! A font the document does not embed is drawn by an installed or bundled
//! face, one glyph per code.  For scripts whose letters change form with
//! their neighbours or are reordered (Arabic joining, Indic conjuncts and
//! vowel signs, mark placement), the codes of a run in such a script are
//! instead shaped as one text with the drawing face's layout tables.
//!
//! A run is a maximal sequence of at least two shown codes drawn by the
//! same face whose characters belong to scripts of one direction, with the
//! combining marks and joiners among them.  Show strings carry text in
//! visual order, so a right-to-left run is reversed into logical order
//! before shaping; the shaped glyphs come back in visual order.

use font::face::FontFace;
use font::program::ShapedGlyph;

use crate::renderer::font_cache::PageFace;

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
    /// (`true`) or left to right.
    Shaped { rtl: bool },
    /// A combining mark or a joiner: part of the run around it.
    Inherited,
}

/// The [`Class`] of `c`; `None` for a character of a script drawn glyph
/// by glyph, which ends a run.
///
/// The right-to-left scripts are Hebrew, Arabic, Syriac, Thaana, N'Ko,
/// Samaritan and Mandaic; the left-to-right ones the Indic scripts,
/// Sinhala, Thai, Lao, Tibetan, Myanmar, Khmer and Mongolian.  Arabic
/// presentation forms are already shaped and are excluded.
const fn class(c: char) -> Option<Class> {
    match c as u32 {
        0x0300..=0x036F
        | 0x1AB0..=0x1AFF
        | 0x1DC0..=0x1DFF
        | 0x200C
        | 0x200D
        | 0x20D0..=0x20FF
        | 0xFE20..=0xFE2F => Some(Class::Inherited),
        0x0590..=0x08FF => Some(Class::Shaped { rtl: true }),
        0x0900..=0x0FFF | 0x1000..=0x109F | 0x1780..=0x18AF | 0xA9E0..=0xA9FF | 0xAA60..=0xAA7F => {
            Some(Class::Shaped { rtl: false })
        }
        _ => None,
    }
}

/// The shaped runs of a shown string whose codes are `codes` (code, CID)
/// and whose per-code drawing glyphs are `drawn`, for a string shown with
/// `face`.  Empty unless `face` is a substitute that knows its codes'
/// characters.
///
/// A run is left glyph by glyph when its face is not an sfnt or when
/// shaping yields `.notdef`, which a per-code fallback may still draw.
pub(super) fn shaped_runs<'f>(
    face: &PageFace,
    codes: &[(u32, u32)],
    drawn: &[Option<Drawn<'f>>],
) -> Vec<ShapedRun<'f>> {
    debug_assert_eq!(codes.len(), drawn.len());
    if !face.substitute {
        return Vec::new();
    }
    let classes: Vec<Option<Class>> = codes
        .iter()
        .map(|&(code, cid)| face.unicode(code, cid).and_then(class))
        .collect();
    let mut runs = Vec::new();
    let mut start = 0;
    while start < codes.len() {
        let (Some(Class::Shaped { rtl }), Some((run_face, ..))) = (classes[start], drawn[start])
        else {
            start += 1;
            continue;
        };
        let same_face = |i: usize| drawn[i].is_some_and(|(f, ..)| std::ptr::eq(f, run_face));
        let end = (start + 1..codes.len())
            .find(|&i| {
                !same_face(i)
                    || !matches!(classes[i], Some(Class::Inherited))
                        && classes[i] != Some(Class::Shaped { rtl })
            })
            .unwrap_or(codes.len());
        if end - start >= 2
            && let Some(glyphs) = shape(face, run_face, &codes[start..end], rtl)
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

/// The glyphs of the codes `codes`, shown in visual order, shaped with
/// `run_face`; `None` when the face cannot shape them or yields `.notdef`.
fn shape(
    face: &PageFace,
    run_face: &FontFace,
    codes: &[(u32, u32)],
    rtl: bool,
) -> Option<Vec<ShapedGlyph>> {
    let mut chars: Vec<char> = codes
        .iter()
        .map(|&(code, cid)| face.unicode(code, cid))
        .collect::<Option<_>>()?;
    if rtl {
        chars.reverse();
    }
    let text: String = chars.into_iter().collect();
    run_face
        .program()
        .shape(&text, rtl)
        .filter(|glyphs| !glyphs.is_empty() && glyphs.iter().all(|g| g.selector != 0))
}

#[cfg(test)]
mod tests {
    use super::{Class, class};

    #[test]
    fn scripts_take_their_direction_and_marks_inherit() {
        assert_eq!(class('\u{0628}'), Some(Class::Shaped { rtl: true }));
        assert_eq!(class('\u{05D0}'), Some(Class::Shaped { rtl: true }));
        assert_eq!(class('\u{0915}'), Some(Class::Shaped { rtl: false }));
        assert_eq!(class('\u{0E01}'), Some(Class::Shaped { rtl: false }));
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
