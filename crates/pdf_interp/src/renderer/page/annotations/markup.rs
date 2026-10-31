//! Appearance streams synthesised for annotations that carry no normal
//! appearance (`/AP /N`), from the geometry and colours in the annotation
//! dictionary (ISO 32000-2 §12.5.6).
//!
//! The generated content is in page user space: it is drawn with the
//! identity form matrix, not fitted to `/Rect`.  [`markup`] covers the
//! subtypes whose appearance follows from their geometry alone: `Square`,
//! `Circle`, `Line`, `Polygon`, `PolyLine`, `Ink`, the text markups
//! `Highlight`, `Underline`, `StrikeOut` and `Squiggly`, and the `Text` note
//! icon.

use std::fmt::Write as _;

use pdf::{Dictionary, Document, Object};
use raster::types::BlendMode;

use super::content::{
    Border, colour, ellipse, inset, n, number, numbers, rectangle, resolve_array,
};
use super::line_ending::LineEnding;
use super::{IDENTITY, Synthesized};
use crate::resources::dict_ext::DictExt;

/// Side of the square `Text` annotation icon, in points.
const NOTE_SIZE: f64 = 20.0;

/// Width of the lines drawn for `Underline`, `StrikeOut` and `Squiggly`.
const MARKUP_LINE_WIDTH: f64 = 1.0;

/// Period and amplitude of the `Squiggly` zigzag.
const SQUIGGLE_STEP: f64 = 2.0;

/// Most zigzag steps of one `Squiggly` quadrilateral: those spanning the
/// largest page side, 14 400 units (Annex C.2).  A wider quadrilateral
/// reaches past any page, and its steps there would only lengthen the
/// content stream.
const MAX_SQUIGGLE_STEPS: f64 = 14_400.0 / SQUIGGLE_STEP;

/// The appearance of the markup annotation `annot` with the normalised
/// rectangle `rect`, or `None` when its subtype has no geometric appearance
/// or the geometry it needs is absent.
pub(super) fn markup(doc: &Document, annot: &Dictionary, rect: [f64; 4]) -> Option<Synthesized> {
    let subtype = annot.get_name(b"Subtype")?;
    let (content, bbox, blend) = match subtype {
        b"Square" | b"Circle" => (
            closed_shape(doc, annot, subtype, rect)?,
            rect,
            BlendMode::Normal,
        ),
        b"Line" | b"PolyLine" | b"Polygon" | b"Ink" => {
            let (content, width) = polylines(doc, annot, subtype)?;
            (content, inset(rect, &[-width; 4]), BlendMode::Normal)
        }
        b"Highlight" | b"Underline" | b"StrikeOut" | b"Squiggly" => {
            text_markup(doc, annot, subtype, rect)?
        }
        b"Text" => {
            let fill = colour_or(doc, annot.get(b"C"), false, "1 1 0 rg")?;
            let icon = [rect[0], rect[3] - NOTE_SIZE, rect[0] + NOTE_SIZE, rect[3]];
            let mut s = String::new();
            note_icon(&mut s, &fill, icon);
            (s, icon, BlendMode::Normal)
        }
        _ => return None,
    };
    Some(Synthesized {
        content: content.into_bytes(),
        bbox,
        opacity: number(doc, annot.get(b"CA")).unwrap_or(1.0).clamp(0.0, 1.0),
        blend,
        matrix: IDENTITY,
    })
}

/// A `Square` or `Circle` (§12.5.6.8): the rectangle, or the ellipse
/// inscribed in it, inside `/RD` and half the border width, filled with
/// `/IC` and stroked with `/C` (default black).
fn closed_shape(
    doc: &Document,
    annot: &Dictionary,
    subtype: &[u8],
    rect: [f64; 4],
) -> Option<String> {
    let border = Border::of(doc, annot);
    let fill = colour(doc, annot.get(b"IC"), false);
    let stroke = (border.width > 0.0)
        .then(|| colour_or_black(doc, annot.get(b"C"), true))
        .flatten();
    if fill.is_none() && stroke.is_none() {
        return None;
    }
    let mut s = String::new();
    let mut r = inset(rect, &numbers(doc, annot.get(b"RD")));
    if stroke.is_some() {
        let half = border.width / 2.0;
        r = inset(r, &[half; 4]);
        border.write(&mut s);
    }
    write_colours(&mut s, fill.as_deref(), stroke.as_deref());
    if subtype == b"Square" {
        rectangle(&mut s, r);
    } else {
        ellipse(&mut s, r);
    }
    s.push_str(paint_op(fill.is_some(), stroke.is_some()));
    Some(s)
}

/// A `Line` (`/L`), `PolyLine` or `Polygon` (`/Vertices`), or `Ink`
/// (`/InkList`) (§12.5.6.7, §12.5.6.9, §12.5.6.13): the paths stroked with
/// `/C` (default black) with round joins and caps, a polygon closed and
/// filled with `/IC`.  The ends of a stroked `Line` or `PolyLine` carry
/// the `/LE` line endings, closed ones filled with `/IC`.  Also returns how
/// far the drawing may reach outside `/Rect`: the border width and the
/// largest ending.
fn polylines(doc: &Document, annot: &Dictionary, subtype: &[u8]) -> Option<(String, f64)> {
    let border = Border::of(doc, annot);
    let stroke = colour_or_black(doc, annot.get(b"C"), true).filter(|_| border.width > 0.0);
    let closed = subtype == b"Polygon";
    let fill = closed
        .then(|| colour(doc, annot.get(b"IC"), false))
        .flatten();
    if fill.is_none() && stroke.is_none() {
        return None;
    }
    let paths: Vec<Vec<f64>> = match subtype {
        b"Line" => vec![numbers(doc, annot.get(b"L"))],
        b"Ink" => resolve_array(doc, annot.get(b"InkList"))
            .iter()
            .map(|p| numbers(doc, Some(p)))
            .collect(),
        _ => vec![numbers(doc, annot.get(b"Vertices"))],
    };
    let mut s = String::new();
    if stroke.is_some() {
        border.write(&mut s);
        s.push_str("1 J 1 j\n");
    }
    write_colours(&mut s, fill.as_deref(), stroke.as_deref());
    let mut drawn = false;
    for path in &paths {
        let (points, _) = path.as_chunks::<2>();
        let Some(([x0, y0], following)) = points.split_first() else {
            continue;
        };
        let _ = writeln!(s, "{} {} m", n(*x0), n(*y0));
        for [x, y] in following {
            let _ = writeln!(s, "{} {} l", n(*x), n(*y));
        }
        if closed {
            s.push_str("h\n");
        }
        drawn = true;
    }
    if !drawn {
        return None;
    }
    s.push_str(paint_op(fill.is_some(), stroke.is_some()));
    let mut reach = border.width;
    if stroke.is_some()
        && matches!(subtype, b"Line" | b"PolyLine")
        && let Some(path) = paths.first()
    {
        let interior = colour(doc, annot.get(b"IC"), false);
        let endings = line_endings(doc, annot.get(b"LE"));
        let (points, _) = path.as_chunks::<2>();
        if let [[x0, y0], [x1, y1], ..] = points
            && let Some(ending) = endings[0]
        {
            let size = LineEnding::size(border.width, (x1 - x0).hypot(y1 - y0));
            ending.write(&mut s, (*x0, *y0), (*x1, *y1), size, interior.as_deref());
            reach = reach.max(border.width + size);
        }
        if let [.., [x0, y0], [x1, y1]] = points
            && let Some(ending) = endings[1]
        {
            let size = LineEnding::size(border.width, (x1 - x0).hypot(y1 - y0));
            ending.write(&mut s, (*x1, *y1), (*x0, *y0), size, interior.as_deref());
            reach = reach.max(border.width + size);
        }
    }
    Some((s, reach))
}

/// The start and end line endings of the `/LE` array `obj`.
fn line_endings(doc: &Document, obj: Option<&Object>) -> [Option<LineEnding>; 2] {
    let names = resolve_array(doc, obj);
    let ending = |i: usize| {
        names
            .get(i)
            .and_then(Object::as_name)
            .and_then(LineEnding::from_name)
    };
    [ending(0), ending(1)]
}

/// A text markup over the boxes of `/QuadPoints` (§12.5.6.10): `Highlight`
/// fills them with `/C` (default yellow) under `Multiply`; `Underline`,
/// `StrikeOut` and `Squiggly` stroke a line in `/C` (default black).
/// Returns the content, the region it paints and its blend mode.
fn text_markup(
    doc: &Document,
    annot: &Dictionary,
    subtype: &[u8],
    rect: [f64; 4],
) -> Option<(String, [f64; 4], BlendMode)> {
    let quads = quad_boxes(doc, annot);
    if quads.is_empty() {
        return None;
    }
    let mut s = String::new();
    let blend = if subtype == b"Highlight" {
        let fill = colour_or(doc, annot.get(b"C"), false, "1 1 0 rg")?;
        let _ = writeln!(s, "{fill}");
        for q in &quads {
            rectangle(&mut s, *q);
        }
        s.push_str("f\n");
        BlendMode::Multiply
    } else {
        let stroke = colour_or_black(doc, annot.get(b"C"), true)?;
        let _ = writeln!(s, "{stroke}\n{} w", n(MARKUP_LINE_WIDTH));
        for q in &quads {
            markup_line(&mut s, subtype, *q);
        }
        s.push_str("S\n");
        BlendMode::Normal
    };
    let bbox = quads.iter().fold(rect, |b, q| {
        [
            b[0].min(q[0]),
            b[1].min(q[1]),
            b[2].max(q[2]),
            b[3].max(q[3]),
        ]
    });
    Some((s, inset(bbox, &[-SQUIGGLE_STEP; 4]), blend))
}

/// [`colour`], with `default` when the entry is absent; `None` when it is
/// present and transparent.
fn colour_or(doc: &Document, obj: Option<&Object>, stroke: bool, default: &str) -> Option<String> {
    if obj.is_none() {
        return Some(default.to_owned());
    }
    colour(doc, obj, stroke)
}

/// [`colour_or`] defaulting to black.
fn colour_or_black(doc: &Document, obj: Option<&Object>, stroke: bool) -> Option<String> {
    colour_or(doc, obj, stroke, if stroke { "0 G" } else { "0 g" })
}

fn write_colours(s: &mut String, fill: Option<&str>, stroke: Option<&str>) {
    for op in [fill, stroke].into_iter().flatten() {
        s.push_str(op);
        s.push('\n');
    }
}

/// The painting operator for a closed path that is filled and/or stroked.
const fn paint_op(fill: bool, stroke: bool) -> &'static str {
    match (fill, stroke) {
        (true, true) => "B\n",
        (true, false) => "f\n",
        _ => "S\n",
    }
}

/// The line of a text markup over the quadrilateral bounds `q`: along its
/// bottom for `Underline`, through its middle for `StrikeOut`, a zigzag
/// along its bottom for `Squiggly`.
fn markup_line(s: &mut String, subtype: &[u8], q: [f64; 4]) {
    if subtype == b"Squiggly" {
        let (bottom, top) = (q[1], q[1] + SQUIGGLE_STEP);
        let _ = writeln!(s, "{} {} m", n(q[0]), n(top));
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "clamped to [0, MAX_SQUIGGLE_STEPS], an integer below u32::MAX"
        )]
        let steps = ((q[2] - q[0]) / SQUIGGLE_STEP)
            .floor()
            .clamp(0.0, MAX_SQUIGGLE_STEPS) as u32;
        for i in 1..=steps {
            let x = SQUIGGLE_STEP.mul_add(f64::from(i), q[0]);
            let _ = writeln!(s, "{} {} l", n(x), n(if i % 2 == 0 { top } else { bottom }));
        }
        return;
    }
    let y = if subtype == b"StrikeOut" {
        f64::midpoint(q[1], q[3])
    } else {
        q[1] + MARKUP_LINE_WIDTH
    };
    let _ = writeln!(s, "{} {} m {} {} l", n(q[0]), n(y), n(q[2]), n(y));
}

/// A note icon in `r`: a speech bubble filled with `fill`, outlined in
/// black, with three text lines.
fn note_icon(s: &mut String, fill: &str, r: [f64; 4]) {
    // The bubble body, inset by half the line width, above a 4 pt tail.
    let (left, bottom, right, top) = (r[0] + 0.5, r[1] + 4.5, r[2] - 0.5, r[3] - 0.5);
    let tip = r[1] + 0.5;
    let _ = writeln!(s, "{fill}\n0 G\n1 w\n1 j");
    let outline = [
        (left, top),
        (right, top),
        (right, bottom),
        (left + 8.0, bottom),
        (left + 4.0, tip),
        (left + 4.0, bottom),
    ];
    let _ = writeln!(s, "{} {} m", n(left), n(bottom));
    for (x, y) in outline {
        let _ = writeln!(s, "{} {} l", n(x), n(y));
    }
    s.push_str("h B\n");
    let step = (top - bottom) / 4.0;
    for i in 1..=3 {
        let y = step.mul_add(-f64::from(i), top);
        let _ = writeln!(
            s,
            "{} {} m {} {} l",
            n(left + 3.0),
            n(y),
            n(right - 3.0),
            n(y)
        );
    }
    s.push_str("S\n");
}

/// The bounds of each quadrilateral of `/QuadPoints` (§12.5.6.10).
fn quad_boxes(doc: &Document, annot: &Dictionary) -> Vec<[f64; 4]> {
    numbers(doc, annot.get(b"QuadPoints"))
        .as_chunks::<8>()
        .0
        .iter()
        .map(|q| {
            let xs = [q[0], q[2], q[4], q[6]];
            let ys = [q[1], q[3], q[5], q[7]];
            [
                xs.iter().copied().fold(f64::INFINITY, f64::min),
                ys.iter().copied().fold(f64::INFINITY, f64::min),
                xs.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                ys.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            ]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::markup;
    use crate::test_helpers::make_doc_with_object;

    /// The synthesised content of the annotation dictionary `annot`.
    fn content(annot: &str) -> Option<String> {
        let doc = make_doc_with_object(annot);
        let dict = doc.get_dictionary((2, 0)).expect("a dictionary");
        let rect = super::super::read_rect(&dict).expect("a rect");
        markup(&doc, &dict, rect).map(|s| String::from_utf8(s.content).expect("ASCII"))
    }

    #[test]
    fn a_square_is_inset_by_half_its_border_and_filled_and_stroked() {
        let c = content(
            "<< /Subtype /Square /Rect [10 10 60 60] /C [1 0 0] /IC [0 0 1] /BS << /W 4 >> >>",
        )
        .expect("an appearance");
        assert!(c.contains("4 w"), "{c}");
        assert!(c.contains("0 0 1 rg") && c.contains("1 0 0 RG"), "{c}");
        assert!(c.contains("12 12 46 46 re"), "{c}");
        assert!(c.trim_end().ends_with('B'), "{c}");
    }

    #[test]
    fn a_square_with_a_transparent_border_and_no_interior_has_no_appearance() {
        assert!(content("<< /Subtype /Square /Rect [10 10 60 60] /C [] >>").is_none());
    }

    #[test]
    fn a_dashed_border_sets_the_dash_pattern() {
        let c = content("<< /Subtype /Circle /Rect [0 0 20 20] /BS << /S /D /D [2 1] >> >>")
            .expect("an appearance");
        assert!(c.contains("[2 1] 0 d"), "{c}");
        assert!(c.contains(" c\n"), "{c}");
    }

    #[test]
    fn a_squiggle_stops_at_the_largest_page_width() {
        let c = content(
            "<< /Subtype /Squiggly /Rect [0 0 10 10] \
             /QuadPoints [0 10 1000000000000 10 0 0 1000000000000 0] >>",
        )
        .expect("an appearance");
        assert_eq!(c.matches(" l\n").count(), 7_200, "one line per step");
    }

    #[test]
    fn unknown_subtypes_have_no_synthesised_appearance() {
        assert!(content("<< /Subtype /Link /Rect [0 0 20 20] >>").is_none());
    }
}
