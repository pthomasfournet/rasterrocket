//! Object access and content-stream fragments shared by the synthesised
//! appearances: numbers, strings and arrays read through indirect
//! references, and the operators that write numbers, colours, rectangles
//! and ellipses.

use std::fmt::Write as _;

use pdf::{Dictionary, Document, Object};

use crate::resources::dict_ext::DictExt;

/// The ratio of a cubic Bézier control-point distance to the radius that
/// approximates a quarter circle: `4 (√2 − 1) / 3`.
const KAPPA: f64 = 0.552_284_75;

pub(super) use crate::resources::resolve_array;

/// The numbers of the array `obj` (non-numbers skipped).
pub(super) fn numbers(doc: &Document, obj: Option<&Object>) -> Vec<f64> {
    resolve_array(doc, obj)
        .iter()
        .filter_map(|o| number(doc, Some(o)))
        .collect()
}

/// `obj` as a finite number, following an indirect reference.
pub(super) fn number(doc: &Document, obj: Option<&Object>) -> Option<f64> {
    match obj? {
        Object::Reference(id) => {
            let referent = doc.get_object(*id).ok()?;
            crate::resources::obj_to_f64(&referent)
        }
        o => crate::resources::obj_to_f64(o),
    }
    .filter(|v| v.is_finite())
}

/// `obj` as a string's bytes, following an indirect reference.
pub(super) fn string(doc: &Document, obj: Option<&Object>) -> Option<Vec<u8>> {
    match obj? {
        Object::String(s, _) => Some(s.clone()),
        Object::Reference(id) => match doc.get_object(*id).ok()?.as_ref() {
            Object::String(s, _) => Some(s.clone()),
            _ => None,
        },
        _ => None,
    }
}

/// The text string `obj`, or the content of the text stream it refers to
/// (§7.9.3), following an indirect reference.
pub(super) fn text_or_stream(doc: &Document, obj: Option<&Object>) -> Option<Vec<u8>> {
    if let Some(s) = string(doc, obj) {
        return Some(s);
    }
    let Some(Object::Reference(id)) = obj else {
        return None;
    };
    let object = doc.get_object(*id).ok()?;
    let Object::Stream(stream) = object.as_ref() else {
        return None;
    };
    stream.decompressed_content().ok()
}

/// The constant opacity `/CA` of the annotation `annot` (§12.5.2), clamped
/// to [0, 1]; 1 when absent.
pub(super) fn opacity(doc: &Document, annot: &Dictionary) -> f64 {
    number(doc, annot.get(b"CA")).unwrap_or(1.0).clamp(0.0, 1.0)
}

/// `v` formatted for a content stream: at most four decimals, no trailing
/// zeros.  A non-finite `v`, which no content-stream number expresses, is
/// written as 0.
pub(super) fn n(v: f64) -> String {
    if !v.is_finite() {
        return "0".to_owned();
    }
    let s = format!("{v:.4}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" {
        "0".to_owned()
    } else {
        s.to_owned()
    }
}

/// The colour operator setting the components `c`: `g`, `rg` or `k` for
/// 1, 3 or 4 components, upper-case when `stroke`; `None` for another
/// component count.
pub(super) fn colour_op(c: &[f64], stroke: bool) -> Option<String> {
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

/// Append the rectangle `r` = `[x0 y0 x1 y1]` as an `re` operator.
pub(super) fn rectangle(s: &mut String, r: [f64; 4]) {
    let _ = writeln!(
        s,
        "{} {} {} {} re",
        n(r[0]),
        n(r[1]),
        n(r[2] - r[0]),
        n(r[3] - r[1])
    );
}

/// Append the ellipse inscribed in `r` as a closed path of four Bézier
/// quarter arcs.
pub(super) fn ellipse(s: &mut String, r: [f64; 4]) {
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

/// Append the circle of the largest radius inscribed in `r`, centred in it.
pub(super) fn circle(s: &mut String, r: [f64; 4]) {
    let (cx, cy) = (f64::midpoint(r[0], r[2]), f64::midpoint(r[1], r[3]));
    let radius = ((r[2] - r[0]).min(r[3] - r[1]) / 2.0).max(0.0);
    ellipse(s, [cx - radius, cy - radius, cx + radius, cy + radius]);
}

/// `r` moved in by `d` = `[left, bottom, right, top]`; a missing or short
/// `d` insets nothing.  A side inset past the opposite one collapses to it.
pub(super) fn inset(r: [f64; 4], d: &[f64]) -> [f64; 4] {
    let [left, bottom, right, top] = match d {
        [left, bottom, right, top, ..] => [*left, *bottom, *right, *top],
        _ => [0.0; 4],
    };
    let x0 = r[0] + left;
    let y0 = r[1] + bottom;
    [x0, y0, (r[2] - right).max(x0), (r[3] - top).max(y0)]
}

/// The border of an annotation (§12.5.4): its width from `/BS /W`, else the
/// third element of `/Border`, else 1; and its dash array from `/BS /D` when
/// `/BS /S` is `/D`, else the fourth element of `/Border`.
pub(super) struct Border {
    pub width: f64,
    pub dash: Vec<f64>,
}

impl Border {
    pub(super) fn of(doc: &Document, annot: &Dictionary) -> Self {
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
    pub(super) fn write(&self, s: &mut String) {
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
pub(super) fn colour(doc: &Document, obj: Option<&Object>, stroke: bool) -> Option<String> {
    colour_op(&numbers(doc, obj), stroke)
}

#[cfg(test)]
mod tests {
    use super::{colour_op, inset, n};

    #[test]
    fn numbers_are_written_compactly() {
        assert_eq!(n(1.0), "1");
        assert_eq!(n(0.5), "0.5");
        assert_eq!(n(-0.000_01), "0");
        assert_eq!(n(2.123_456), "2.1235");
        assert_eq!(n(f64::INFINITY), "0");
        assert_eq!(n(f64::NAN), "0");
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
    fn colour_operators_follow_the_component_count() {
        assert_eq!(colour_op(&[0.5], false).as_deref(), Some("0.5 g"));
        assert_eq!(
            colour_op(&[1.0, 0.0, 0.0], true).as_deref(),
            Some("1 0 0 RG")
        );
        assert_eq!(colour_op(&[0.0; 4], false).as_deref(), Some("0 0 0 0 k"));
        assert_eq!(colour_op(&[0.0; 2], false), None);
    }
}
