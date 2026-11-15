//! Connected regions of a pixel grid.

/// Iterative 4-connected flood fill from `start`; `visit` claims a pixel and
/// reports whether it belongs to the region.  Returns the region size.
pub(crate) fn flood(
    start: usize,
    w: usize,
    h: usize,
    stack: &mut Vec<usize>,
    mut visit: impl FnMut(usize) -> bool,
) -> usize {
    stack.clear();
    if !visit(start) {
        return 0;
    }
    stack.push(start);
    let mut size = 0;
    while let Some(i) = stack.pop() {
        size += 1;
        let (x, y) = (i % w, i / w);
        let mut push = |j: usize, stack: &mut Vec<usize>| {
            if visit(j) {
                stack.push(j);
            }
        };
        if x > 0 {
            push(i - 1, stack);
        }
        if x + 1 < w {
            push(i + 1, stack);
        }
        if y > 0 {
            push(i - w, stack);
        }
        if y + 1 < h {
            push(i + w, stack);
        }
    }
    size
}

/// The indices of the pixels on the border of a `w × h` grid (`w, h ≥ 1`);
/// corner and single-row or single-column pixels repeat.
pub(crate) fn border_indices(w: usize, h: usize) -> impl Iterator<Item = usize> {
    let rows = (0..w).flat_map(move |x| [x, (h - 1) * w + x]);
    let cols = (0..h).flat_map(move |y| [y * w, y * w + w - 1]);
    rows.chain(cols)
}
