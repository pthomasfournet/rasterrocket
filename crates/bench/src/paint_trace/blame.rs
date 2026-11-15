//! Attributing the pixels at which two renders differ to the operations
//! of a trace.

use std::collections::HashMap;

use crate::pnm::Image;
use crate::trace::{Rect, Trace};

/// The most operations reported.
const SHOWN: usize = 30;

/// The longest part of a line printed.
const LINE_CHARS: usize = 160;

/// The half-open pixel range `[lo, hi)` a rectangle's edges `lo..hi`
/// cover, within `0..limit`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the values are clamped to 0..=limit first"
)]
fn span(lo: f64, hi: f64, limit: usize) -> (usize, usize) {
    #[expect(clippy::cast_precision_loss, reason = "image sides are far below 2^52")]
    let max = limit as f64;
    (
        lo.floor().clamp(0.0, max) as usize,
        hi.ceil().clamp(0.0, max) as usize,
    )
}

/// The device coordinate of the centre of pixel `i`.
#[expect(clippy::cast_precision_loss, reason = "image sides are far below 2^52")]
fn centre(i: usize) -> f64 {
    i as f64 + 0.5
}

/// `line` cut to [`LINE_CHARS`] characters.
fn shorten(line: &str) -> String {
    let mut out: String = line.chars().take(LINE_CHARS).collect();
    if line.chars().count() > LINE_CHARS {
        out.push('…');
    }
    out
}

/// Print the operations of `trace` that cover the pixels at which
/// `before` and `after` differ, most pixels first; `Ok(true)` when the
/// renders are identical.  Each differing pixel goes to the last
/// operation in paint order whose extent holds it; an operation in a soft
/// mask's group paints the mask, so its pixels go to the operations the
/// mask applies to instead.
pub fn run(trace: &Trace, before: &Image, after: &Image) -> Result<bool, String> {
    let dims = |i: &Image| (i.width, i.height, i.channels);
    if dims(before) != dims(after) {
        return Err(format!(
            "the renders differ in shape: {:?} vs {:?}",
            dims(before),
            dims(after)
        ));
    }
    let (w, h) = (before.width, before.height);
    let side = |v: u32| usize::try_from(v).unwrap_or(usize::MAX);
    let page = (side(trace.size.0), side(trace.size.1));
    if page != (w, h) {
        eprintln!(
            "paint_trace: the trace's page is {}x{} px and the renders {w}x{h}: render at the \
             trace's resolution for the extents to line up",
            page.0, page.1
        );
    }
    let mut differ = vec![false; w * h];
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
    let mut count = 0_usize;
    for y in 0..h {
        for x in 0..w {
            if before.pixel(x, y) != after.pixel(x, y) {
                differ[y * w + x] = true;
                (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
                count += 1;
            }
        }
    }
    if count == 0 {
        println!("the renders are identical");
        return Ok(true);
    }
    // The last operation covering each pixel of the differing area.
    let (aw, ah) = (x1 - x0, y1 - y0);
    let mut owner: Vec<Option<usize>> = vec![None; aw * ah];
    for (index, op) in trace.ops.iter().enumerate() {
        let Some(rect): Option<Rect> = op.extent.filter(|_| !op.in_mask) else {
            continue;
        };
        let (rx0, rx1) = span(rect[0], rect[2], w);
        let (ry0, ry1) = span(rect[1], rect[3], h);
        for y in ry0.max(y0)..ry1.min(y1) {
            for x in rx0.max(x0)..rx1.min(x1) {
                if op.covers(centre(x), centre(y)) {
                    owner[(y - y0) * aw + (x - x0)] = Some(index);
                }
            }
        }
    }
    let mut counts: HashMap<usize, usize> = HashMap::new();
    let mut unattributed = 0_usize;
    for y in y0..y1 {
        for x in x0..x1 {
            if !differ[y * w + x] {
                continue;
            }
            match owner[(y - y0) * aw + (x - x0)] {
                Some(index) => *counts.entry(index).or_default() += 1,
                None => unattributed += 1,
            }
        }
    }
    let mut tally: Vec<(usize, usize)> = counts.into_iter().collect();
    tally.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    println!(
        "{count} pixels differ, within x {x0}..{x1}, y {y0}..{y1}; {} operations cover them",
        tally.len()
    );
    for &(index, n) in tally.iter().take(SHOWN) {
        let op = &trace.ops[index];
        println!("{n:>8}  L{:<6} {}", op.line, shorten(&op.text));
        let mut parent = op.parent;
        while let Some(p) = parent {
            let enclosing = &trace.ops[p];
            println!(
                "{:>8}  in L{:<3} {}",
                "",
                enclosing.line,
                shorten(&enclosing.text)
            );
            parent = enclosing.parent;
        }
    }
    if tally.len() > SHOWN {
        println!("  … and {} more operations", tally.len() - SHOWN);
    }
    if unattributed > 0 {
        println!("{unattributed:>8}  pixels no operation's extent covers");
    }
    Ok(false)
}
