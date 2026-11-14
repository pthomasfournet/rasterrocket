//! The strings shown next to a string on its baseline, whose characters
//! are the shaping context of its ends (see [`super::text_shaping`]).
//!
//! Two strings are neighbours when the second is shown after the first in
//! one content stream, only operators that change no more than the text
//! state, the colour or the marked-content nesting come between them
//! ([`keeps_text_context`]), and the second starts where the first ends:
//! on its baseline, at most [`MAX_GAP`] em past its end and at most
//! [`MAX_OVERLAP`] em before it.  A `TJ` adjustment between two strings of
//! an array moves the second within these bounds or out of them.
//!
//! The string before is known when a string is shown: its end is kept in
//! a [`TextTail`] while neighbouring operators follow it.  The string after
//! is read ahead from the operators that follow ([`Following`]), with the
//! text state they set.

use super::PageRenderer;
use super::text_ops::{apply_kern, is_singular, text_rendering_matrix};
use super::text_shaping::{CONTEXT_CODES, CodeText, font_shapes, shapes};
use crate::content::{Operator, TextArrayElement};
use crate::renderer::text::{TextMatrix, TextState};
use crate::resources::font::CidEncoding;

/// The largest gap between neighbouring strings, in em of the first: a
/// wider one separates words.
const MAX_GAP: f64 = 0.1;

/// The largest overlap of neighbouring strings, in em of the first.
const MAX_OVERLAP: f64 = 0.5;

/// The end of a shown string: the texts of its last codes in storage
/// order, at most [`CONTEXT_CODES`], the text matrix after its last
/// advance, its em, `Tfs × Th`, in text space, and the distance along x
/// in text space from that end to where the glyphs of its last run end
/// when that run continues into the string after, else zero.
#[derive(Clone, Debug)]
pub(super) struct TextTail {
    pub texts: Vec<CodeText>,
    pub end: TextMatrix,
    pub em: f64,
    pub drift: f64,
}

/// What follows a shown string in its content stream: the rest of its
/// `TJ` array, then the operators after the one that shows it.
#[derive(Clone, Copy)]
pub(super) struct Following<'a, 'o> {
    pub elements: &'a [TextArrayElement<'o>],
    pub ops: &'a [Operator<'o>],
}

impl<'a, 'o> Following<'a, 'o> {
    /// The operators `ops`, after a string shown outside a `TJ` array.
    pub(super) const fn ops(ops: &'a [Operator<'o>]) -> Self {
        Self { elements: &[], ops }
    }
}

/// Whether `op` keeps the strings around it neighbours: it sets the text
/// state, positions text, sets a colour or opens or closes a marked-content
/// sequence that is no optional content, or it shows text.
pub(super) const fn keeps_text_context(op: &Operator) -> bool {
    op.sets_colour()
        || matches!(
            op,
            Operator::SetFont { .. }
                | Operator::SetCharSpacing(_)
                | Operator::SetWordSpacing(_)
                | Operator::SetHorizScaling(_)
                | Operator::SetLeading(_)
                | Operator::SetTextRise(_)
                | Operator::SetTextRenderMode(_)
                | Operator::TextMove(..)
                | Operator::TextMoveSetLeading(..)
                | Operator::SetTextMatrix(_)
                | Operator::NextLine
                | Operator::ShowText(_)
                | Operator::ShowTextArray(_)
                | Operator::MoveNextLineShow(_)
                | Operator::MoveNextLineShowSpaced { .. }
                | Operator::BeginMarkedContent
                | Operator::EndMarkedContent
                | Operator::MarkedContentPoint
        )
}

/// Whether a string whose text matrix is `start` neighbours a string that
/// ends at `end` with em `em` (see [`offset`]).
pub(super) fn neighbours(end: &TextMatrix, start: &TextMatrix, em: f64) -> bool {
    offset(end, start, em).is_some()
}

/// How far along x in text space past `end` a string whose text matrix is
/// `start` starts, when it neighbours a string that ends at `end` with em
/// `em`: the two matrices share their 2×2 part and `start`'s origin lies
/// on the baseline of `end`, from [`MAX_OVERLAP`] em before it to
/// [`MAX_GAP`] em after it.
#[expect(clippy::many_single_char_names, reason = "PDF matrix components")]
fn offset(end: &TextMatrix, start: &TextMatrix, em: f64) -> Option<f64> {
    let [a, b, c, d, e, f] = *end;
    let scale = a.abs().max(b.abs()).max(c.abs()).max(d.abs());
    let det = a.mul_add(d, -(b * c));
    if !(em > 0.0 && det != 0.0 && det.is_finite()) {
        return None;
    }
    if (0..4).any(|i| (end[i] - start[i]).abs() > scale * 1e-9) {
        return None;
    }
    // The displacement in text space: (dx, dy) = tx × (a, b) + ty × (c, d).
    let (dx, dy) = (start[4] - e, start[5] - f);
    let tx = dx.mul_add(d, -(dy * c)) / det;
    let ty = dy.mul_add(a, -(dx * b)) / det;
    ((-MAX_OVERLAP * em..=MAX_GAP * em).contains(&tx) && ty.abs() <= em * 1e-3).then_some(tx)
}

/// The start of the string shown after a string on its baseline: the
/// texts of its first codes, at most [`CONTEXT_CODES`], and how far along
/// x in text space past the end of the string before it starts.
#[derive(Clone, Debug, Default)]
pub(super) struct NextString {
    pub texts: Vec<CodeText>,
    pub start: f64,
}

impl PageRenderer<'_> {
    /// The start of the string shown after a string that ends at `end`
    /// with em `em`, when it is a neighbour (see the module documentation)
    /// drawn by a substitute face in horizontal writing.
    pub(super) fn next_string(
        &mut self,
        end: TextMatrix,
        em: f64,
        following: Following<'_, '_>,
    ) -> Option<NextString> {
        let mut ts = self.gstate.current().text.clone();
        ts.text_matrix = end;
        if let Some(bytes) = first_string(&mut ts, following.elements) {
            return self.next_string_at(&ts, bytes, end, em);
        }
        for op in following.ops {
            if ts.apply_state_op(op, |name| self.font_cache.select(&self.resources, name)) {
                continue;
            }
            let bytes = if let Some(bytes) = ts.begin_show(op) {
                bytes
            } else if let Operator::ShowTextArray(elements) = op {
                match first_string(&mut ts, elements) {
                    Some(bytes) => bytes,
                    None => continue,
                }
            } else if keeps_text_context(op) {
                continue;
            } else {
                return None;
            };
            if !bytes.is_empty() {
                return self.next_string_at(&ts, bytes, end, em);
            }
        }
        None
    }

    /// The start of `bytes` shown in the text state `ts` after a string
    /// that ends at `end` with em `em`; `None` unless the two strings are
    /// neighbours and `bytes` is drawn by a substitute face in horizontal
    /// writing.
    fn next_string_at(
        &mut self,
        ts: &TextState,
        bytes: &[u8],
        end: TextMatrix,
        em: f64,
    ) -> Option<NextString> {
        let font = ts.font.as_ref()?;
        if !font_shapes(&font.descriptor) {
            return None;
        }
        let start = offset(&end, &ts.text_matrix, em)?;
        let ctm = self.gstate.current().ctm;
        let trm = text_rendering_matrix(
            &ts.text_matrix,
            &ctm,
            ts.font_size,
            ts.horiz_scaling / 100.0,
        );
        if is_singular(&trm) {
            return None;
        }
        let face = self.font_cache.get_or_load(font, &ts.font_name, trm)?;
        if !shapes(&font.descriptor, face) {
            return None;
        }
        Some(NextString {
            texts: CidEncoding::shown_codes(font.descriptor.cid_encoding.as_ref(), bytes)
                .take(CONTEXT_CODES)
                .map(|d| face.text(d.code, d.value))
                .collect(),
            start,
        })
    }
}

/// The first non-empty string of the `TJ` array `elements`, with `ts`
/// moved by the adjustments before it.
fn first_string<'e>(ts: &mut TextState, elements: &'e [TextArrayElement<'_>]) -> Option<&'e [u8]> {
    for element in elements {
        match element {
            TextArrayElement::Text(bytes) if !bytes.is_empty() => return Some(bytes),
            TextArrayElement::Text(_) => {}
            TextArrayElement::Offset(kern) => apply_kern(ts, *kern),
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::neighbours;

    const IDENTITY: [f64; 6] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

    fn at(mut tm: [f64; 6], x: f64, y: f64) -> [f64; 6] {
        tm[4] = x;
        tm[5] = y;
        tm
    }

    #[test]
    fn neighbours_share_a_baseline_without_a_word_gap() {
        let end = at(IDENTITY, 100.0, 50.0);
        let em = 10.0;
        assert!(neighbours(&end, &end, em), "a string shown at the end");
        assert!(neighbours(&end, &at(IDENTITY, 100.9, 50.0), em));
        assert!(
            neighbours(&end, &at(IDENTITY, 96.0, 50.0), em),
            "an overlap"
        );
        assert!(
            !neighbours(&end, &at(IDENTITY, 101.5, 50.0), em),
            "a word gap"
        );
        assert!(!neighbours(&end, &at(IDENTITY, 94.0, 50.0), em));
        assert!(
            !neighbours(&end, &at(IDENTITY, 100.0, 38.0), em),
            "a new line"
        );
        let rotated = [0.0, 1.0, -1.0, 0.0, 0.0, 0.0];
        assert!(!neighbours(&end, &at(rotated, 100.0, 50.0), em));
        assert!(!neighbours(&end, &end, 0.0), "no em");
    }

    #[test]
    fn the_gap_is_measured_along_the_baseline() {
        // A baseline rotated a quarter turn: text space x is user space y.
        let rotated = [0.0, 2.0, -2.0, 0.0, 10.0, 10.0];
        let em = 1.0;
        assert!(neighbours(&rotated, &at(rotated, 10.0, 10.1), em));
        assert!(!neighbours(&rotated, &at(rotated, 10.0, 10.4), em));
        assert!(
            !neighbours(&rotated, &at(rotated, 10.1, 10.0), em),
            "off the baseline"
        );
    }
}
