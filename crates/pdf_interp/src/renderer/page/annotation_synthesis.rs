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

use crate::resources::dict_ext::DictExt;

/// A synthesised appearance.
pub(super) struct Synthesized {
    /// The content stream, in page user space.
    pub content: Vec<u8>,
    /// The region the content may paint, in page user space.
    pub bbox: [f64; 4],
    /// The constant opacity `/CA` applied to stroking and filling.
    pub opacity: f64,
    /// The blend mode the content is painted with.
    pub blend: BlendMode,
}

/// Side of the square `Text` annotation icon, in points.
const NOTE_SIZE: f64 = 20.0;

/// Width of the lines drawn for `Underline`, `StrikeOut` and `Squiggly`.
const MARKUP_LINE_WIDTH: f64 = 1.0;

/// Period and amplitude of the `Squiggly` zigzag.
const SQUIGGLE_STEP: f64 = 2.0;

/// The ratio of a cubic Bézier control-point distance to the radius that
/// approximates a quarter circle: `4 (√2 − 1) / 3`.
const KAPPA: f64 = 0.552_284_75;

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
        let _ = writeln!(
            s,
            "{} {} {} {} re",
            n(r[0]),
            n(r[1]),
            n(r[2] - r[0]),
            n(r[3] - r[1])
        );
    } else {
        ellipse(&mut s, r);
    }
    s.push_str(paint_op(fill.is_some(), stroke.is_some()));
    Some(s)
}

/// A `Line` (`/L`), `PolyLine` or `Polygon` (`/Vertices`), or `Ink`
/// (`/InkList`) (§12.5.6.7, §12.5.6.9, §12.5.6.13): the paths stroked with
/// `/C` (default black) with round joins and caps, a polygon closed and
/// filled with `/IC`.  Also returns the border width.
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
    Some((s, border.width))
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
            let _ = writeln!(
                s,
                "{} {} {} {} re",
                n(q[0]),
                n(q[1]),
                n(q[2] - q[0]),
                n(q[3] - q[1])
            );
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

/// The border of an annotation (§12.5.4): its width from `/BS /W`, else the
/// third element of `/Border`, else 1; and its dash array from `/BS /D` when
/// `/BS /S` is `/D`, else the fourth element of `/Border`.
struct Border {
    width: f64,
    dash: Vec<f64>,
}

impl Border {
    fn of(doc: &Document, annot: &Dictionary) -> Self {
        let bs = annot
            .get(b"BS")
            .and_then(|o| crate::resources::resolve_dict(doc, o));
        let border = resolve_array(doc, annot.get(b"Border"));
        let width = bs
            .as_ref()
            .and_then(|bs| number(doc, bs.get(b"W")))
            .or_else(|| border.get(2).and_then(|w| number(doc, Some(w))))
            .unwrap_or(1.0)
            .max(0.0);
        let dash = match &bs {
            Some(bs) if bs.get_name(b"S") == Some(b"D") => {
                let d = numbers(doc, bs.get(b"D"));
                if d.is_empty() { vec![3.0] } else { d }
            }
            Some(_) => Vec::new(),
            None => border
                .get(3)
                .map(|d| numbers(doc, Some(d)))
                .unwrap_or_default(),
        };
        Self { width, dash }
    }

    /// Write the line width and, when dashed, the dash pattern.
    fn write(&self, s: &mut String) {
        let _ = writeln!(s, "{} w", n(self.width));
        if self.dash.iter().any(|&d| d > 0.0) {
            let dash: Vec<String> = self.dash.iter().take(10).map(|&d| n(d)).collect();
            let _ = writeln!(s, "[{}] 0 d", dash.join(" "));
        }
    }
}

/// The colour operator for the colour array `obj` (§12.5.2 `/C`, `/IC`):
/// `g`, `rg` or `k` for 1, 3 or 4 components, upper-case when `stroke`.
/// `None` for an empty array (transparent), an absent entry or another
/// component count.
fn colour(doc: &Document, obj: Option<&Object>, stroke: bool) -> Option<String> {
    let c = numbers(doc, obj);
    let op = match (c.len(), stroke) {
        (1, false) => "g",
        (1, true) => "G",
        (3, false) => "rg",
        (3, true) => "RG",
        (4, false) => "k",
        (4, true) => "K",
        _ => return None,
    };
    let comps: Vec<String> = c.iter().map(|&v| n(v)).collect();
    Some(format!("{} {op}", comps.join(" ")))
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

/// An ellipse inscribed in `r`, as four Bézier quarter arcs.
fn ellipse(s: &mut String, r: [f64; 4]) {
    let (cx, cy) = (f64::midpoint(r[0], r[2]), f64::midpoint(r[1], r[3]));
    let (rx, ry) = ((r[2] - r[0]) / 2.0, (r[3] - r[1]) / 2.0);
    let (kx, ky) = (rx * KAPPA, ry * KAPPA);
    let _ = writeln!(s, "{} {} m", n(cx), n(r[3]));
    let arcs = [
        [cx + kx, r[3], r[2], cy + ky, r[2], cy],
        [r[2], cy - ky, cx + kx, r[1], cx, r[1]],
        [cx - kx, r[1], r[0], cy - ky, r[0], cy],
        [r[0], cy + ky, cx - kx, r[3], cx, r[3]],
    ];
    for a in arcs {
        let _ = writeln!(
            s,
            "{} {} {} {} {} {} c",
            n(a[0]),
            n(a[1]),
            n(a[2]),
            n(a[3]),
            n(a[4]),
            n(a[5])
        );
    }
    s.push_str("h\n");
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
            reason = "a quad is at most a page wide, far below u32::MAX steps"
        )]
        let steps = ((q[2] - q[0]) / SQUIGGLE_STEP).floor().max(0.0) as u32;
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

/// `r` moved in by `d` = `[left, bottom, right, top]`; a missing or short
/// `d` insets nothing.  A side inset past the opposite one collapses to it.
fn inset(r: [f64; 4], d: &[f64]) -> [f64; 4] {
    let [left, bottom, right, top] = match d {
        [left, bottom, right, top, ..] => [*left, *bottom, *right, *top],
        _ => [0.0; 4],
    };
    let x0 = r[0] + left;
    let y0 = r[1] + bottom;
    [x0, y0, (r[2] - right).max(x0), (r[3] - top).max(y0)]
}

/// The array `obj`, following an indirect reference; empty otherwise.
fn resolve_array(doc: &Document, obj: Option<&Object>) -> Vec<Object> {
    match obj {
        Some(Object::Array(a)) => a.clone(),
        Some(Object::Reference(id)) => doc
            .get_object(*id)
            .ok()
            .and_then(|o| o.as_array().map(<[Object]>::to_vec))
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

/// The numbers of the array `obj` (non-numbers skipped).
fn numbers(doc: &Document, obj: Option<&Object>) -> Vec<f64> {
    resolve_array(doc, obj)
        .iter()
        .filter_map(|o| number(doc, Some(o)))
        .collect()
}

/// `obj` as a number, following an indirect reference.
fn number(doc: &Document, obj: Option<&Object>) -> Option<f64> {
    match obj? {
        Object::Reference(id) => {
            let referent = doc.get_object(*id).ok()?;
            crate::resources::obj_to_f64(&referent)
        }
        o => crate::resources::obj_to_f64(o),
    }
    .filter(|v| v.is_finite())
}

/// `v` formatted for a content stream: at most four decimals, no trailing
/// zeros.
fn n(v: f64) -> String {
    let s = format!("{v:.4}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" {
        "0".to_owned()
    } else {
        s.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::{inset, markup, n};
    use crate::test_helpers::make_doc_with_object;

    /// The synthesised content of the annotation dictionary `annot`.
    fn content(annot: &str) -> Option<String> {
        let doc = make_doc_with_object(annot);
        let dict = doc.get_dictionary((2, 0)).expect("a dictionary");
        let rect = super::super::annotations::read_rect(&dict).expect("a rect");
        markup(&doc, &dict, rect).map(|s| String::from_utf8(s.content).expect("ASCII"))
    }

    #[test]
    fn numbers_are_written_compactly() {
        assert_eq!(n(1.0), "1");
        assert_eq!(n(0.5), "0.5");
        assert_eq!(n(-0.000_01), "0");
        assert_eq!(n(2.123_456), "2.1235");
    }

    #[test]
    fn inset_collapses_rather_than_inverts() {
        assert_eq!(
            inset([0.0, 0.0, 10.0, 10.0], &[1.0, 2.0, 3.0, 4.0]),
            [1.0, 2.0, 7.0, 6.0]
        );
        assert_eq!(
            inset([0.0, 0.0, 10.0, 10.0], &[8.0, 0.0, 8.0, 0.0]),
            [8.0, 0.0, 8.0, 10.0]
        );
        assert_eq!(inset([0.0, 0.0, 10.0, 10.0], &[]), [0.0, 0.0, 10.0, 10.0]);
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
    fn unknown_subtypes_have_no_synthesised_appearance() {
        assert!(content("<< /Subtype /Link /Rect [0 0 20 20] >>").is_none());
    }
}
