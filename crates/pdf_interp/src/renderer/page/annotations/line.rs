//! Appearances synthesised for line annotations without a normal
//! appearance (ISO 32000-2 §12.5.6.7).
//!
//! The appearance is written in the frame of the line: its origin is the
//! first point of `/L`, its x axis runs to the second point, `length` away,
//! and its y axis is the x axis turned a quarter turn counter-clockwise.
//! The form matrix maps that frame onto the page.
//!
//! With a non-zero leader line length `/LL`, the line is drawn displaced
//! along the y axis, to the side of the sign of `/LL`, by `/LLO + |/LL|`,
//! and a leader line is drawn at each end from `/LLO` to
//! `/LLO + |/LL| + /LLE` on that side: the leader line offset `/LLO` is
//! the gap left at the measured points, the extension `/LLE` the part of
//! the leader line beyond the line.  The line is stroked with `/C` (default
//! black) and its ends carry the `/LE` line endings, closed ones filled
//! with `/IC`.
//!
//! With `/Cap` true, the text of the annotation is shown as a caption in
//! black Helvetica, centred on the line and moved by `/CO` (along the line,
//! then perpendicular to it).  An `/Inline` caption (`/CP`, the default) is
//! centred vertically on the line, which is cut around it; a `/Top` caption
//! sits on the line.  An inline caption longer than the line is shown on
//! top.

use std::fmt::Write as _;

use pdf::{Dictionary, Document};
use raster::types::BlendMode;

use super::super::super::gstate::Ctm;
use super::Synthesized;
use super::content::{Border, colour, n, number, numbers, opacity};
use super::line_ending::LineEnding;
use super::text::{DefaultAppearance, markup_text};
use super::typeset::{LEADING, Typesetter, show_line, wrap};
use crate::resources::dict_ext::DictExt;

/// The resource name of the caption font, standard Helvetica unless the
/// form's default resources define the name.
pub(super) const FONT: &[u8] = b"Helv";

/// Font size of a caption.
const CAPTION_SIZE: f64 = 9.0;

/// Length of line left out on each side of an inline caption.
const CAPTION_GAP: f64 = 2.0;

/// Cap height and descent of Helvetica, in units of the font size: an
/// inline caption's baseline lies half a cap height below the line, a top
/// caption's a descent above it.
const CAP_HEIGHT: f64 = 0.718;
const DESCENT: f64 = 0.207;

/// The extent of a caption line below and above its baseline, in units of
/// the font size, enclosing the glyphs of the standard and fallback fonts.
const BELOW_BASELINE: f64 = 0.3;
const ABOVE_BASELINE: f64 = 1.0;

/// A caption: the text and where it is shown.
struct Caption {
    text: String,
    top: bool,
    /// `/CO`: the offset along the line and perpendicular to it.
    shift: (f64, f64),
}

/// A line annotation, read from its dictionary.
pub(super) struct Line {
    /// The matrix from the frame of the line to page user space.
    frame: Ctm,
    /// The distance between the points of `/L`.
    length: f64,
    /// The y coordinate of the line.
    offset: f64,
    /// The y coordinates where the leader lines start and end; `None`
    /// without leader lines.
    leaders: Option<(f64, f64)>,
    border: Border,
    /// The stroke colour operator; `None` when the line is not stroked.
    stroke: Option<String>,
    /// The fill colour operator of `/IC`, for closed line endings.
    interior: Option<String>,
    endings: [Option<LineEnding>; 2],
    caption: Option<Caption>,
    opacity: f64,
}

impl Line {
    /// The line annotation `annot`; `None` when `/L` does not hold two
    /// points.
    pub(super) fn of(doc: &Document, annot: &Dictionary) -> Option<Self> {
        let &[x0, y0, x1, y1, ..] = numbers(doc, annot.get(b"L")).as_slice() else {
            return None;
        };
        let length = (x1 - x0).hypot(y1 - y0);
        let (cos, sin) = if length > 0.0 {
            ((x1 - x0) / length, (y1 - y0) / length)
        } else {
            (1.0, 0.0)
        };
        let leader = number(doc, annot.get(b"LL")).unwrap_or(0.0);
        let (offset, leaders) = if leader == 0.0 {
            (0.0, None)
        } else {
            let side = leader.signum();
            let gap = number(doc, annot.get(b"LLO")).unwrap_or(0.0).max(0.0);
            let extension = number(doc, annot.get(b"LLE")).unwrap_or(0.0).max(0.0);
            let line = gap + leader.abs();
            (side * line, Some((side * gap, side * (line + extension))))
        };
        let border = Border::of(doc, annot);
        let stroke = (border.width > 0.0)
            .then(|| colour(doc, annot.get(b"C"), true).unwrap_or_else(|| "0 G".to_owned()));
        let caption = (annot.get_bool(b"Cap") == Some(true))
            .then(|| markup_text(doc, annot))
            .filter(|text| !text.trim().is_empty())
            .map(|text| {
                let shift = numbers(doc, annot.get(b"CO"));
                Caption {
                    text,
                    top: annot.get_name(b"CP") == Some(b"Top"),
                    shift: (
                        shift.first().copied().unwrap_or(0.0),
                        shift.get(1).copied().unwrap_or(0.0),
                    ),
                }
            });
        Some(Self {
            frame: [cos, sin, -sin, cos, x0, y0],
            length,
            offset,
            leaders,
            border,
            stroke,
            interior: colour(doc, annot.get(b"IC"), false),
            endings: LineEnding::pair(doc, annot.get(b"LE")),
            caption,
            opacity: opacity(doc, annot),
        })
    }

    /// Whether the appearance shows a caption, set in [`FONT`].
    pub(super) const fn has_caption(&self) -> bool {
        self.caption.is_some()
    }

    /// The appearance, its caption set by `ts`; `None` when it draws
    /// nothing.  Without `ts` the caption is left out.
    pub(super) fn appearance(&self, ts: Option<&mut dyn Typesetter>) -> Option<Synthesized> {
        let caption = self.caption.as_ref().zip(ts);
        if self.stroke.is_none() && caption.is_none() {
            return None;
        }
        let mut s = String::new();
        let mut bounds = Bounds::new();
        let gap = if let Some((caption, ts)) = caption {
            self.write_caption(&mut s, &mut bounds, caption, ts)
        } else {
            None
        };
        if let Some(stroke) = &self.stroke {
            self.write_lines(&mut s, &mut bounds, stroke, gap);
        }
        Some(Synthesized {
            content: s.into_bytes(),
            bbox: bounds.rect(),
            opacity: self.opacity,
            blend: BlendMode::Normal,
            matrix: self.frame,
        })
    }

    /// Append the line, cut over `gap` when given, its endings and the
    /// leader lines, stroked with `stroke`.
    fn write_lines(
        &self,
        s: &mut String,
        bounds: &mut Bounds,
        stroke: &str,
        gap: Option<(f64, f64)>,
    ) {
        let (y, end) = (self.offset, self.length);
        let width = self.border.width;
        s.push_str("q\n");
        self.border.write(s);
        let _ = writeln!(s, "1 J 1 j {stroke}");
        let segments = match gap {
            Some((a, b)) => vec![(0.0, a), (b, end)],
            None => vec![(0.0, end)],
        };
        for (a, b) in segments.into_iter().filter(|(a, b)| a < b || end == 0.0) {
            let _ = writeln!(s, "{} {} m {} {} l", n(a), n(y), n(b), n(y));
        }
        if let Some((from, to)) = self.leaders {
            for x in [0.0, end] {
                let _ = writeln!(s, "{} {} m {} {} l", n(x), n(from), n(x), n(to));
            }
            bounds.include(0.0, from);
            bounds.include(end, to);
        }
        s.push_str("S\n");
        let mut reach = width;
        for (ending, tip, from) in [
            (self.endings[0], (0.0, y), (end, y)),
            (self.endings[1], (end, y), (0.0, y)),
        ] {
            if let Some(ending) = ending {
                let size = ending.draw(s, tip, from, width, self.interior.as_deref());
                reach = reach.max(width + size);
            }
        }
        s.push_str("Q\n");
        bounds.include(0.0, y);
        bounds.include(end, y);
        bounds.grow(reach);
    }

    /// Append `caption` set by `ts`; returns the span of the line it
    /// covers when it is inline.
    fn write_caption(
        &self,
        s: &mut String,
        bounds: &mut Bounds,
        caption: &Caption,
        ts: &mut dyn Typesetter,
    ) -> Option<(f64, f64)> {
        let size = CAPTION_SIZE;
        let lines = wrap(&caption.text, f64::INFINITY, |_| 0.0);
        let widths: Vec<f64> = lines.iter().map(|l| ts.width(FONT, l) * size).collect();
        let block = widths.iter().copied().fold(0.0, f64::max);
        let inline = !caption.top && CAPTION_GAP.mul_add(2.0, block) <= self.length;
        let leading = LEADING * size;
        // The distance from the first baseline down to the last.
        let below: f64 = lines.iter().skip(1).map(|_| leading).sum();
        let (dx, dy) = caption.shift;
        let last_baseline = dy
            + self.offset
            + if inline {
                below / 2.0 - CAP_HEIGHT * size / 2.0
            } else {
                DESCENT * size
            };
        let da = DefaultAppearance {
            font: FONT.to_vec(),
            size,
            rest: "0 g".to_owned(),
        };
        let centre = self.length / 2.0 + dx;
        let mut baseline = last_baseline + below;
        for (line, width) in lines.iter().zip(&widths) {
            if !line.is_empty() {
                show_line(
                    s,
                    ts,
                    &da,
                    FONT,
                    size,
                    (centre - width / 2.0, baseline),
                    line,
                );
            }
            baseline -= leading;
        }
        let (left, right) = (centre - block / 2.0, centre + block / 2.0);
        bounds.include(left, BELOW_BASELINE.mul_add(-size, last_baseline));
        bounds.include(right, ABOVE_BASELINE.mul_add(size, last_baseline + below));
        inline.then(|| {
            (
                (left - CAPTION_GAP).max(0.0),
                (right + CAPTION_GAP).min(self.length),
            )
        })
    }
}

/// The bounding box of the content drawn in the frame of the line.
struct Bounds([f64; 4]);

impl Bounds {
    const fn new() -> Self {
        Self([
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ])
    }

    const fn include(&mut self, x: f64, y: f64) {
        let [x0, y0, x1, y1] = &mut self.0;
        *x0 = x0.min(x);
        *y0 = y0.min(y);
        *x1 = x1.max(x);
        *y1 = y1.max(y);
    }

    fn grow(&mut self, by: f64) {
        let [x0, y0, x1, y1] = self.0;
        self.include(x0 - by, y0 - by);
        self.include(x1 + by, y1 + by);
    }

    /// The box; empty at the origin when nothing was included.
    fn rect(&self) -> [f64; 4] {
        if self.0[0] <= self.0[2] {
            self.0
        } else {
            [0.0; 4]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Line;
    use crate::renderer::page::annotations::Synthesized;
    use crate::renderer::page::annotations::typeset::WinAnsiTypesetter;
    use crate::test_helpers::make_doc_with_object;

    /// The appearance of the line annotation `annot`, its caption set by
    /// [`WinAnsiTypesetter`], and its content.
    fn appearance(annot: &str) -> Option<(Synthesized, String)> {
        let doc = make_doc_with_object(annot);
        let dict = doc.get_dictionary((2, 0)).expect("a dictionary");
        let a = Line::of(&doc, &dict)?.appearance(Some(&mut WinAnsiTypesetter))?;
        let content = String::from_utf8(a.content.clone()).expect("ASCII");
        Some((a, content))
    }

    #[test]
    fn leader_lines_run_from_the_offset_past_the_displaced_line() {
        let (a, c) =
            appearance("<< /Subtype /Line /Rect [0 0 1 1] /L [0 0 100 0] /LL 10 /LLO 2 /LLE 3 >>")
                .expect("an appearance");
        assert_eq!(a.matrix, [1.0, 0.0, -0.0, 1.0, 0.0, 0.0]);
        assert!(c.contains("0 12 m 100 12 l"), "{c}");
        assert!(
            c.contains("0 2 m 0 15 l") && c.contains("100 2 m 100 15 l"),
            "{c}"
        );
        assert_eq!(a.bbox, [-1.0, 1.0, 101.0, 16.0]);
    }

    #[test]
    fn a_negative_leader_length_puts_the_line_on_the_other_side() {
        let (_, c) = appearance("<< /Subtype /Line /Rect [0 0 1 1] /L [0 0 100 0] /LL -10 >>")
            .expect("an appearance");
        assert!(c.contains("0 -10 m 100 -10 l"), "{c}");
        assert!(c.contains("0 0 m 0 -10 l"), "{c}");
    }

    #[test]
    fn the_frame_runs_along_the_line() {
        let (a, c) = appearance("<< /Subtype /Line /Rect [0 0 1 1] /L [10 10 10 50] >>")
            .expect("an appearance");
        assert_eq!(a.matrix, [0.0, 1.0, -1.0, 0.0, 10.0, 10.0]);
        assert!(c.contains("0 0 m 40 0 l"), "{c}");
    }

    #[test]
    fn an_inline_caption_is_centred_on_the_line_which_is_cut_around_it() {
        // Two characters half an em wide at 9 points: 9 wide, centred at 50.
        let (_, c) = appearance(
            "<< /Subtype /Line /Rect [0 0 1 1] /L [0 0 100 0] /Cap true /Contents (ab) >>",
        )
        .expect("an appearance");
        assert!(
            c.contains("0 0 m 43.5 0 l") && c.contains("56.5 0 m 100 0 l"),
            "{c}"
        );
        assert!(c.contains("45.5 -3.231 Td"), "{c}");
        assert!(c.contains("BT /Helv 9 Tf 0 g"), "{c}");
    }

    #[test]
    fn a_caption_too_long_for_the_line_sits_on_top_with_its_offset() {
        let (_, c) = appearance(
            "<< /Subtype /Line /Rect [0 0 1 1] /L [0 0 10 0] /Cap true /Contents (abcd) \
             /CO [5 2] >>",
        )
        .expect("an appearance");
        assert!(c.contains("0 0 m 10 0 l"), "{c}");
        assert!(c.contains("1 3.863 Td"), "{c}");
    }

    #[test]
    fn caption_lines_stack_upwards_from_the_last() {
        let (_, c) = appearance(
            "<< /Subtype /Line /Rect [0 0 1 1] /L [0 0 100 0] /Cap true /CP /Top \
             /Contents (ab\\rcd) >>",
        )
        .expect("an appearance");
        let first = c.find("45.5 12.213 Td").expect("first line one leading up");
        let second = c.find("45.5 1.863 Td").expect("last line on the line");
        assert!(first < second, "{c}");
    }

    #[test]
    fn an_unstroked_line_without_caption_has_no_appearance() {
        assert!(
            appearance("<< /Subtype /Line /Rect [0 0 1 1] /L [0 0 9 9] /BS << /W 0 >> >>")
                .is_none()
        );
        assert!(
            appearance(
                "<< /Subtype /Line /Rect [0 0 1 1] /L [0 0 9 9] /BS << /W 0 >> /Cap true >>"
            )
            .is_none()
        );
        assert!(appearance("<< /Subtype /Line /Rect [0 0 1 1] /L [0 0] >>").is_none());
    }

    #[test]
    fn the_box_reaches_over_the_line_endings() {
        let (a, _) = appearance(
            "<< /Subtype /Line /Rect [0 0 1 1] /L [0 0 100 0] /LE [/ClosedArrow /None] >>",
        )
        .expect("an appearance");
        assert_eq!(a.bbox, [-7.0, -7.0, 107.0, 7.0]);
    }
}
