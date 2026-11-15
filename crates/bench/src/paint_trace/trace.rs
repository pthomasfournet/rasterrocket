//! Reading a paint trace back: its operations, their nesting, and the
//! device-space rectangle each can paint.

use std::collections::HashMap;

/// A rectangle `[x0, y0, x1, y1]` of device space.
pub type Rect = [f64; 4];

/// One line of a trace.
pub struct Op {
    /// The line's number in the trace, from 1.
    pub line: usize,
    /// The line, without its indentation.
    pub text: String,
    /// Index of the operation that encloses this one (a form, soft mask
    /// or text object), if any.
    pub parent: Option<usize>,
    /// Whether the operation lies in a soft mask's group: it paints the
    /// mask, not the page.
    pub in_mask: bool,
    /// The device rectangle the operation can paint, within its clip;
    /// `None` for a line that paints nothing itself.
    pub extent: Option<Rect>,
    /// For a stroke whose points the trace prints: the segments
    /// `[x0, y0, x1, y1]` of its path (a curve by its control polygon,
    /// which contains it) and the stroke's reach past them.  The stroke
    /// paints within its extent only that near its segments.
    pub outline: Option<(Vec<Rect>, f64)>,
}

impl Op {
    /// Whether the operation can paint the pixel whose centre is `(x, y)`.
    pub fn covers(&self, x: f64, y: f64) -> bool {
        let Some(e) = self.extent else {
            return false;
        };
        if x < e[0] || x > e[2] || y < e[1] || y > e[3] {
            return false;
        }
        self.outline.as_ref().is_none_or(|(segments, reach)| {
            segments
                .iter()
                .any(|s| segment_distance(*s, x, y) <= *reach)
        })
    }
}

/// The distance from `(x, y)` to the segment `[x0, y0, x1, y1]`.
fn segment_distance([x0, y0, x1, y1]: Rect, x: f64, y: f64) -> f64 {
    let (dx, dy) = (x1 - x0, y1 - y0);
    let len2 = dx.mul_add(dx, dy * dy);
    let t = if len2 > 0.0 {
        ((x - x0).mul_add(dx, (y - y0) * dy) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (x - t.mul_add(dx, x0)).hypot(y - t.mul_add(dy, y0))
}

/// The segments of the `pts=` points of `line`: each subpath's points
/// joined in order, a closed one back to its first.
fn segments(line: &str) -> Option<Vec<Rect>> {
    let pts = line.split(" pts=").nth(1)?;
    let mut out = Vec::new();
    let mut start: Option<(f64, f64)> = None;
    let mut last: Option<(f64, f64)> = None;
    for token in pts.split(' ') {
        let mut chars = token.chars();
        match chars.next() {
            Some('z') => {
                if let (Some(a), Some(b)) = (last, start) {
                    out.push([a.0, a.1, b.0, b.1]);
                }
            }
            Some(op @ ('m' | 'l' | 'c')) => {
                let &[x, y] = numbers(chars.as_str()).as_slice() else {
                    break;
                };
                if op == 'm' {
                    start = Some((x, y));
                } else if let Some(a) = last {
                    out.push([a.0, a.1, x, y]);
                }
                last = Some((x, y));
            }
            _ => break,
        }
    }
    Some(out)
}

/// A trace's page and its operations.
pub struct Trace {
    /// The page's size in device pixels.
    pub size: (u32, u32),
    pub ops: Vec<Op>,
}

/// The value of the attribute `key=` of `line`, the first if repeated.
pub fn attr<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    line.split(' ')
        .find_map(|token| token.strip_prefix(key)?.strip_prefix('='))
}

/// The numbers of the comma-separated list `list`.
fn numbers(list: &str) -> Vec<f64> {
    list.trim_matches(|c| c == '[' || c == ']')
        .split(',')
        .filter_map(|v| v.parse().ok())
        .collect()
}

/// The rectangle of the attribute `bbox=` of `line`.
fn bbox(line: &str) -> Option<Rect> {
    match numbers(attr(line, "bbox")?).as_slice() {
        &[x0, y0, x1, y1] => Some([x0, y0, x1, y1]),
        _ => None,
    }
}

/// The bounding box of the `x,y` points of `tokens`.
fn points_box<'a>(tokens: impl Iterator<Item = &'a str>) -> Option<Rect> {
    let mut rect: Option<Rect> = None;
    for token in tokens {
        if let &[x, y] = numbers(token).as_slice() {
            rect = Some(rect.map_or([x, y, x, y], |[x0, y0, x1, y1]| {
                [x0.min(x), y0.min(y), x1.max(x), y1.max(y)]
            }));
        }
    }
    rect
}

/// The intersection of `a` and `b`.
pub fn intersect(a: Rect, b: Rect) -> Rect {
    [
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    ]
}

/// `r` grown by `m` on every side.
fn grow(r: Rect, m: f64) -> Rect {
    [r[0] - m, r[1] - m, r[2] + m, r[3] + m]
}

/// `√|det|` of the 2 × 2 part of the matrix attribute `key=[a,b,c,d,…]`
/// of `line`; 1 when absent.
fn matrix_scale(line: &str, key: &str) -> f64 {
    match attr(line, key).map(numbers).as_deref() {
        Some(&[a, b, c, d, ..]) => a.mul_add(d, -(b * c)).abs().sqrt(),
        _ => 1.0,
    }
}

/// The em of the glyphs of the text line `text`, in device pixels.
fn em(text: &str) -> f64 {
    let size: f64 = attr(text, "size")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    size * matrix_scale(text, "tm") * matrix_scale(text, "ctm")
}

/// The device width of the stroke of the line `line`.
fn stroke_width(line: &str) -> f64 {
    let stroke = line.split(" | ").nth(1).unwrap_or(line);
    attr(stroke, "dlw")
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1.0)
        .max(1.0)
}

/// How far past its path the stroke of the line `line` can reach in
/// device pixels: half its width, times the miter limit for mitred joins
/// or √2 for the corners of caps and other joins, and one pixel of
/// antialiasing.
fn stroke_reach(line: &str) -> f64 {
    let stroke = line.split(" | ").nth(1).unwrap_or(line);
    let miter: f64 = attr(stroke, "ml")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let join = if attr(stroke, "join") == Some("Miter") {
        miter.max(std::f64::consts::SQRT_2)
    } else {
        std::f64::consts::SQRT_2
    };
    (stroke_width(line) * join).mul_add(0.5, 1.0)
}

/// How far from its path's segments a stroke of the line `line` paints,
/// away from its joins: half its width and a pixel of antialiasing.
fn stroke_band(line: &str) -> f64 {
    stroke_width(line).mul_add(0.5, 1.0)
}

/// The operations of `text`, a paint trace.
pub fn parse(text: &str) -> Trace {
    let mut size = (0, 0);
    let mut ops: Vec<Op> = Vec::new();
    let mut clips: HashMap<String, Rect> = HashMap::new();
    // The open containers: (depth of their children, index).
    let mut open: Vec<(usize, usize)> = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let depth = (raw.len() - raw.trim_start_matches(' ').len()) / 2;
        let line = raw.trim_start();
        if let Some(page) = line.strip_prefix("page ")
            && let Some((w, h)) = attr(page, "px").and_then(|px| px.split_once('x'))
        {
            size = (w.parse().unwrap_or(0), h.parse().unwrap_or(0));
        }
        while open.last().is_some_and(|&(d, _)| d > depth) {
            open.pop();
        }
        let parent = open.last().map(|&(_, at)| at);
        let page_rect = [0.0, 0.0, f64::from(size.0), f64::from(size.1)];
        let kind = line.split(' ').next().unwrap_or("");
        let in_mask = parent.is_some_and(|p| ops[p].in_mask || ops[p].text.starts_with("smask "));
        let clip_of = |line: &str| {
            attr(line, "clip")
                .and_then(|name| clips.get(name).copied())
                .unwrap_or(page_rect)
        };
        let extent = match kind {
            "clip" => {
                let mut words = line.split(' ');
                let name = words.nth(1).unwrap_or("");
                if let Some((own, parent)) = name.split_once('<') {
                    let body = if line.contains(" rect ") {
                        points_box(line.split(" rect ").nth(1).unwrap_or("").split(' '))
                    } else {
                        bbox(line)
                    };
                    let parent_rect = clips.get(parent).copied().unwrap_or(page_rect);
                    let rect = body.map_or(parent_rect, |b| intersect(parent_rect, b));
                    clips.insert(own.to_owned(), rect);
                }
                None
            }
            "fill" | "image" => bbox(line).map(|b| intersect(grow(b, 1.0), clip_of(line))),
            "stroke" | "fillstroke" => {
                bbox(line).map(|b| intersect(grow(b, stroke_reach(line)), clip_of(line)))
            }
            "sh" => Some(clip_of(line)),
            "glyph" => parent.and_then(|p| {
                let text = &ops[p].text;
                let em = em(text);
                let pen = attr(line, "pen").map(numbers)?;
                let &[x, y] = pen.as_slice() else {
                    return None;
                };
                let glyph = [x - 0.3 * em, y - 1.2 * em, x + 1.3 * em, y + 0.4 * em];
                Some(intersect(glyph, clip_of(text)))
            }),
            _ => None,
        };
        // A fill and stroke paints the area its path encloses as well.
        let outline = (kind == "stroke")
            .then(|| segments(line))
            .flatten()
            .map(|segments| (segments, stroke_band(line)));
        let index = ops.len();
        ops.push(Op {
            line: i + 1,
            text: line.to_owned(),
            parent,
            in_mask,
            extent,
            outline,
        });
        if matches!(kind, "form" | "smask" | "text") && !line.ends_with(" end") {
            open.push((depth + 1, index));
        }
    }
    Trace { size, ops }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRACE: &str = "rasterrocket-paint-trace 1\n\
        page px=100x100 scale=1\n\
        clip a<0000000000000000 nz n=5 sub=1 cub=0 bbox=10,10,50,50 h=1\n\
        fill nz n=5 sub=1 cub=0 bbox=0,0,100,100 h=2 paint=DeviceGray(0) ca=255 clip=a\n\
        smask m (5 0 R) luminosity bc=[] ctm=[1,0,0,1,0,0]\n\
        \x20 fill nz n=5 sub=1 cub=0 bbox=0,0,9,9 h=3 paint=DeviceGray(1) ca=255 clip=0\n\
        smask end\n\
        text /F1(5 0 R) size=10 Tr=0 tm=[1,0,0,1,0,0] ctm=[1,0,0,1,0,0] clip=0\n\
        \x20 glyph c=41 sel=65 pen=20,80\n";

    #[test]
    fn extents_lie_within_their_clips() {
        let trace = parse(TRACE);
        assert_eq!(trace.size, (100, 100));
        let fill = &trace.ops[3];
        assert_eq!(fill.extent, Some([10.0, 10.0, 50.0, 50.0]));
        assert!(!fill.in_mask);
    }

    #[test]
    fn nesting_follows_indentation() {
        let trace = parse(TRACE);
        assert_eq!(trace.ops[5].parent, Some(4));
        assert!(trace.ops[5].in_mask);
        assert_eq!(trace.ops[6].parent, None);
        let glyph = &trace.ops[8];
        assert_eq!(glyph.parent, Some(7));
        assert_eq!(glyph.extent, Some([17.0, 68.0, 33.0, 84.0]));
    }

    #[test]
    fn a_stroke_covers_only_the_pixels_near_its_path() {
        let trace = parse(
            "page px=100x100 scale=1\n\
             stroke n=5 sub=1 cub=0 bbox=10,10,90,90 h=1 pts=m10,10 l90,10 l90,90 l10,90 \
             l10,10 z lw=2 dlw=2 cap=Butt join=Bevel ml=10 | paint=DeviceGray(0) CA=255 clip=0\n",
        );
        let stroke = &trace.ops[1];
        assert!(stroke.covers(50.5, 10.5));
        assert!(stroke.covers(89.5, 50.5));
        assert!(!stroke.covers(50.5, 50.5));
    }
}
