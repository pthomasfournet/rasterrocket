//! Exact-area coverage of line outlines.
//!
//! [`AreaRaster`] scan-converts closed outlines made of line segments into
//! per-pixel coverage: the fraction of each pixel's area inside the outline
//! under the non-zero or the even-odd rule, as a byte `round(255 · area)`.
//!
//! ## Arithmetic
//!
//! Coordinates are fixed point with [`SUBPIXEL_BITS`] fraction bits, taken
//! as `floor(v · 256)` of the device coordinate and then made relative to
//! the raster window, so moving an outline and its window by whole pixels
//! moves the coverage exactly.  Every later step is integer.
//!
//! Each segment, clipped to the window, is walked cell by cell.  In a cell
//! it deposits its `cover`, the signed height it spans there, and its
//! `area`, `cover · (fx_enter + fx_exit)` with `fx` its offset from the
//! cell's left side: twice the signed area between the segment and that
//! side.  The covers of a segment telescope, so the covers it deposits in a
//! row sum exactly to the height it spans in that row.  Along a row, the
//! running sum of covers to the left of a pixel plus the pixel's own cells
//! gives `v = 512 · Σcover − Σarea`, the winding-weighted area of the pixel
//! in units of 2⁻¹⁷ px²: an integer winding contributes exactly
//! [`FULL_AREA`] per pixel.
//!
//! The fill rule maps `v` to an area: non-zero takes `min(|v|, FULL_AREA)`,
//! even-odd folds `v` modulo two windings.  Both are exact where the
//! outline's edges do not cross one another inside a pixel; where several
//! edges share a pixel their signed areas add before the rule applies.
//!
//! ## Window clipping
//!
//! A segment is walked over the window's rows only, its position on each row
//! and cell boundary computed from its own endpoints, so a pixel's coverage
//! does not depend on the window.  The cells left of the window add their
//! cover to every pixel of the row inside it and are deposited as one cell
//! at its left side with no area; the cells right of it affect no pixel in
//! it and are dropped.

mod outline;
mod table;

pub use outline::AreaOutline;
pub(crate) use outline::scale;
pub use table::CoverageTable;

use crate::xpath::{XPath, XPathFlags};

/// Fraction bits of a fixed-point coordinate.
pub const SUBPIXEL_BITS: u32 = 8;

/// One pixel in fixed point.
const ONE: i64 = 1 << SUBPIXEL_BITS;

/// The accumulated area of a fully covered pixel: `2 · ONE²`.
pub const FULL_AREA: i64 = 2 * ONE * ONE;

/// Fixed-point coordinates are clamped to this magnitude, which keeps the
/// product of two coordinate differences inside `i128` and the per-step
/// increments of the walk inside `i64`.
const COORD_LIMIT: i64 = 1 << 52;

/// The coverage byte of an accumulated area `a ∈ [0, FULL_AREA]`, rounded
/// to nearest with halves up.
#[inline]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a ∈ [0, FULL_AREA] maps into [0, 255]"
)]
const fn area_byte(a: i64) -> u8 {
    ((a * 255 + FULL_AREA / 2) / FULL_AREA) as u8
}

/// The coverage byte of the accumulated value `v` under the fill rule.
#[inline]
const fn rule_byte(v: i64, eo: bool) -> u8 {
    let a = if eo {
        let m = v.rem_euclid(2 * FULL_AREA);
        if m > FULL_AREA { 2 * FULL_AREA - m } else { m }
    } else {
        let m = v.abs();
        if m > FULL_AREA { FULL_AREA } else { m }
    };
    area_byte(a)
}

/// The fixed-point image of the device coordinate `v`, floored and clamped
/// to `±COORD_LIMIT` (NaN maps to 0).
#[inline]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the float-to-int conversion saturates; the clamp bounds the result"
)]
fn fixed(v: f64) -> i64 {
    ((v * 256.0).floor() as i64).clamp(-COORD_LIMIT, COORD_LIMIT)
}

/// [`COORD_LIMIT`] in device pixels, `2^44`.
const LIMIT_PX: f64 = 17_592_186_044_416.0;

/// The fixed-point segments the device segment `a → b` deposits, passed to
/// `emit` as `[ax, ay, bx, by]`; no NaN coordinate is allowed.
///
/// A segment within `±LIMIT_PX` on both axes is one segment, converted
/// coordinate by coordinate.  A longer one is cut where it crosses the
/// limits, so each piece keeps the segment's slope: the parts beyond
/// `|y| = LIMIT_PX` lie in rows outside every window and are dropped, and
/// the parts beyond `|x| = LIMIT_PX` are moved onto that line, which keeps
/// their rows and the cover they add to every pixel between the limits.
/// A segment with an infinite coordinate or extent has no slope to keep,
/// and one too long for its cuts to be resolved has none that can be kept;
/// both are clamped coordinate by coordinate.
fn fixed_segments(a: (f64, f64), b: (f64, f64), mut emit: impl FnMut([i64; 4])) {
    let within = |v: f64| v.abs() <= LIMIT_PX;
    let clamped = |a: (f64, f64), b: (f64, f64)| [fixed(a.0), fixed(a.1), fixed(b.0), fixed(b.1)];
    if within(a.0) && within(a.1) && within(b.0) && within(b.1) {
        emit(clamped(a, b));
        return;
    }
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    if ![dx, dy].iter().all(|v| v.is_finite()) {
        emit(clamped(a, b));
        return;
    }
    let at = |t: f64| (dx.mul_add(t, a.0), dy.mul_add(t, a.1));
    // The parameter interval of the rows within the limits.
    let (lo, hi) = if dy == 0.0 {
        if !within(a.1) {
            return;
        }
        (0.0, 1.0)
    } else {
        let (t0, t1) = ((-LIMIT_PX - a.1) / dy, (LIMIT_PX - a.1) / dy);
        (t0.min(t1).max(0.0), t0.max(t1).min(1.0))
    };
    if lo >= hi {
        // An interval that rounds to nothing while the segment's rows meet
        // the limits belongs to a segment too long for its cuts to be
        // resolved; it is clamped, which keeps the outline closed.
        if a.1.min(b.1) <= LIMIT_PX && a.1.max(b.1) >= -LIMIT_PX {
            emit(clamped(a, b));
        }
        return;
    }
    // Cuts at the columns `x = ±LIMIT_PX` inside that interval.
    let mut cuts = [lo, hi, hi, hi];
    let mut n = 1;
    if dx != 0.0 {
        let (t0, t1) = ((-LIMIT_PX - a.0) / dx, (LIMIT_PX - a.0) / dx);
        for t in [t0.min(t1), t0.max(t1)] {
            if lo < t && t < hi {
                cuts[n] = t;
                n += 1;
            }
        }
    }
    cuts[n] = hi;
    #[expect(
        clippy::float_cmp,
        reason = "an uncut end has the parameter 0 or 1 exactly, and keeps its endpoint"
    )]
    let point = |t: f64| {
        if t == 0.0 {
            a
        } else if t == 1.0 {
            b
        } else {
            at(t)
        }
    };
    for pair in cuts[..=n].windows(2) {
        emit(clamped(point(pair[0]), point(pair[1])));
    }
}

/// Bytes per block in which a coverage row is searched for runs of one
/// value.
pub(crate) const BLOCK: usize = 16;

/// The number of leading bytes of `bytes` equal to `value`.
pub(crate) fn run_len(bytes: &[u8], value: u8) -> usize {
    let (blocks, _) = bytes.as_chunks::<BLOCK>();
    let whole = blocks.iter().take_while(|b| is_uniform(b, value)).count() * BLOCK;
    whole + bytes[whole..].iter().take_while(|&&c| c == value).count()
}

/// Whether every byte of `block` equals `value`.
const fn is_uniform(block: &[u8; BLOCK], value: u8) -> bool {
    u128::from_ne_bytes(*block) == u128::from_ne_bytes([value; BLOCK])
}

/// Whether `block` holds [`BLOCK`] bytes, all equal to `value`.
pub(crate) fn uniform(block: &[u8], value: u8) -> bool {
    <&[u8; BLOCK]>::try_from(block).is_ok_and(|b| is_uniform(b, value))
}

/// One cell deposit.
#[derive(Clone, Copy, Debug)]
struct Cell {
    x: i32,
    y: i32,
    cover: i32,
    area: i32,
}

/// Exact-area rasteriser over an inclusive pixel window.
///
/// Outlines are added with [`add_line`](Self::add_line) or
/// [`add_xpath`](Self::add_xpath); [`rows`](Self::rows) then yields the
/// coverage of each row that any segment reaches.  Every subpath must be
/// closed: an open outline leaves a winding that does not return to zero.
pub struct AreaRaster {
    /// Window origin, in pixels.
    x0: i32,
    y0: i32,
    /// Window size, in pixels.
    width: i32,
    height: i32,
    /// Deposits, in walk order.  Consecutive deposits into one cell are
    /// merged into one entry while the sums fit `i32`.
    cells: Vec<Cell>,
    /// Scratch: the deposits sorted by row, then column.
    sorted: Vec<Cell>,
    /// Scratch: per-row offsets into `sorted`.
    row_start: Vec<u32>,
    /// Scratch: one row's coverage bytes.
    line: Vec<u8>,
}

impl AreaRaster {
    /// A rasteriser over the pixels `[x0, x1] × [y0, y1]`; an inverted range
    /// gives an empty window.
    #[must_use]
    pub fn new(x0: i32, y0: i32, x1: i32, y1: i32) -> Self {
        let span = |a: i32, b: i32| {
            i32::try_from((i64::from(b) - i64::from(a) + 1).max(0)).unwrap_or(i32::MAX)
        };
        Self {
            x0,
            y0,
            width: span(x0, x1),
            height: span(y0, y1),
            cells: Vec::new(),
            sorted: Vec::new(),
            row_start: Vec::new(),
            line: Vec::new(),
        }
    }

    /// Whether the window holds no pixel.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// Add every segment of `xpath` (device space), in its original
    /// direction.
    pub fn add_xpath(&mut self, xpath: &XPath) {
        for seg in &xpath.segs {
            if seg.flags.contains(XPathFlags::FLIPPED) {
                self.add_line(seg.x1, seg.y1, seg.x0, seg.y0);
            } else {
                self.add_line(seg.x0, seg.y0, seg.x1, seg.y1);
            }
        }
    }

    /// Add the segment `(x0, y0) → (x1, y1)` (device space), cut at the
    /// fixed-point limits as [`fixed_segments`] cuts it.  Segments with a
    /// NaN coordinate are skipped.
    pub fn add_line(&mut self, x0: f64, y0: f64, x1: f64, y1: f64) {
        if [x0, y0, x1, y1].iter().any(|v| v.is_nan()) {
            return;
        }
        fixed_segments((x0, y0), (x1, y1), |[ax, ay, bx, by]| {
            self.add_fixed(ax, ay, bx, by);
        });
    }

    /// Add the segment `(ax, ay) → (bx, by)`, device coordinates in fixed
    /// point as [`fixed_segments`] makes them.
    pub(crate) fn add_fixed(&mut self, ax: i64, ay: i64, bx: i64, by: i64) {
        if self.is_empty() {
            return;
        }
        let ox = i64::from(self.x0) * ONE;
        let oy = i64::from(self.y0) * ONE;
        self.walk(ax - ox, ay - oy, bx - ox, by - oy);
    }

    /// Deposit the cells the window-relative segment `(x1, y1) → (x2, y2)`
    /// crosses within the window's rows.
    ///
    /// The segment's position on each row boundary is computed from its
    /// endpoints alone, never from a clipped copy, so the cells of a pixel
    /// do not depend on the window.
    fn walk(&mut self, x1: i64, y1: i64, x2: i64, y2: i64) {
        if y1 == y2 {
            return;
        }
        // Walk downwards; `sign` restores the segment's direction.
        let (sign, (xa, ya), (xb, yb)) = if y1 < y2 {
            (1, (x1, y1), (x2, y2))
        } else {
            (-1, (x2, y2), (x1, y1))
        };
        let h = i64::from(self.height);
        if yb <= 0 || ya >= h * ONE {
            return;
        }
        let first_row = (ya >> SUBPIXEL_BITS).max(0);
        let last_row = ((yb - 1) >> SUBPIXEL_BITS).min(h - 1);
        let (dx, dy) = (xb - xa, yb - ya);
        // x on the boundary at the top of each row from `first_row` on.
        let top = first_row * ONE;
        let mut boundary = Dda::new(xa, (top - ya, dx), dy, ONE * dx);
        for row in first_row..=last_row {
            let row_top = row * ONE;
            let (x_top, fy_top) = if ya >= row_top {
                (xa, ya - row_top)
            } else {
                (boundary.value(), 0)
            };
            boundary.advance();
            let (x_bot, fy_bot) = if yb <= row_top + ONE {
                (xb, yb - row_top)
            } else {
                (boundary.value(), ONE)
            };
            if x_top <= x_bot {
                self.hline(row, (x_top, fy_top), (x_bot, fy_bot), sign);
            } else {
                self.hline(row, (x_bot, fy_bot), (x_top, fy_top), -sign);
            }
        }
    }

    /// Deposit the cells of the part of a segment within row `ey`, from
    /// `left` to `right` (`left.0 ≤ right.0`), each `(x, fy)` with `fy`
    /// measured from the row's top.  A cell's cover is `sign` times the rise
    /// of `fy` across it from left to right.
    fn hline(&mut self, ey: i64, left: (i64, i64), right: (i64, i64), sign: i64) {
        let ((xl, yl), (xr, yr)) = (left, right);
        if yl == yr {
            return;
        }
        let w = i64::from(self.width);
        if xl == xr {
            let ex = xl >> SUBPIXEL_BITS;
            let cover = sign * (yr - yl);
            let fx = xl - ex * ONE;
            if ex < 0 {
                self.deposit(0, ey, cover, 0);
            } else {
                self.deposit(ex, ey, cover, cover * 2 * fx);
            }
            return;
        }
        let first_cell = xl >> SUBPIXEL_BITS;
        let end_cell = (xr - 1) >> SUBPIXEL_BITS;
        let last_cell = end_cell.min(w - 1);
        if first_cell == end_cell && first_cell >= 0 {
            let cover = sign * (yr - yl);
            let left = first_cell * ONE;
            self.deposit(first_cell, ey, cover, cover * (xl + xr - 2 * left));
            return;
        }
        let (dx, dy) = (xr - xl, yr - yl);
        // fy on each cell boundary, from the left side of the first cell in
        // the window on.
        let start = first_cell.max(0);
        let mut boundary = Dda::new(yl, (start * ONE - xl, dy), dx, ONE * dy);
        if first_cell < 0 {
            // The cells left of the window add their cover to every pixel of
            // the row inside it: one deposit at its left side, no area.
            let y0 = if xr <= 0 { yr } else { boundary.value() };
            self.deposit(0, ey, sign * (y0 - yl), 0);
        }
        for cell in start..=last_cell {
            let cell_left = cell * ONE;
            let (x_in, y_in) = if xl >= cell_left {
                (xl, yl)
            } else {
                (cell_left, boundary.value())
            };
            boundary.advance();
            let (x_out, y_out) = if xr <= cell_left + ONE {
                (xr, yr)
            } else {
                (cell_left + ONE, boundary.value())
            };
            let cover = sign * (y_out - y_in);
            self.deposit(cell, ey, cover, cover * (x_in + x_out - 2 * cell_left));
        }
    }

    /// Add `cover` and `area` to cell `(ex, ey)`, merging with the previous
    /// deposit when it is the same cell.  Cells outside the window are
    /// dropped: none lies left of it, and those right of it affect no pixel.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "ex < width and ey < height fit i32; one deposit has |cover| ≤ ONE and \
                  |area| ≤ 2·ONE²"
    )]
    fn deposit(&mut self, ex: i64, ey: i64, cover: i64, area: i64) {
        if cover == 0 && area == 0 {
            return;
        }
        if ex < 0 || ex >= i64::from(self.width) || ey < 0 || ey >= i64::from(self.height) {
            return;
        }
        let (x, y) = (ex as i32, ey as i32);
        let (cover, area) = (cover as i32, area as i32);
        if let Some(last) = self.cells.last_mut()
            && last.x == x
            && last.y == y
            && let (Some(c), Some(a)) = (last.cover.checked_add(cover), last.area.checked_add(area))
        {
            last.cover = c;
            last.area = a;
            return;
        }
        self.cells.push(Cell { x, y, cover, area });
    }

    /// Call `emit(y, x, coverage)` for every row a deposit reaches, in
    /// ascending `y`: `coverage[i]` is the coverage of device pixel
    /// `(x + i, y)` under the even-odd rule when `eo`, else non-zero.  The
    /// span starts at the row's leftmost deposit and ends at its last
    /// covered pixel within the window; pixels outside it are uncovered.
    /// A span may hold uncovered pixels and may be empty.
    ///
    /// The deposits are consumed: the rasteriser is empty afterwards.
    pub fn rows(&mut self, eo: bool, mut emit: impl FnMut(i32, i32, &[u8])) {
        self.rows_mut(eo, |y, x, coverage| emit(y, x, coverage));
    }

    /// [`Self::rows`], each row's coverage lent mutably: `emit` may change
    /// it in place, and the change is not seen by later rows.
    #[expect(
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        reason = "cell coordinates lie in [0, width) × [0, height), both within i32"
    )]
    pub fn rows_mut(&mut self, eo: bool, mut emit: impl FnMut(i32, i32, &mut [u8])) {
        if self.cells.is_empty() {
            return;
        }
        // Counting sort by row, then sort each row by column.
        let n_rows = self.height as usize;
        self.row_start.clear();
        self.row_start.resize(n_rows + 1, 0);
        for c in &self.cells {
            self.row_start[c.y as usize + 1] += 1;
        }
        for i in 0..n_rows {
            self.row_start[i + 1] += self.row_start[i];
        }
        self.sorted.clear();
        self.sorted.resize(
            self.cells.len(),
            Cell {
                x: 0,
                y: 0,
                cover: 0,
                area: 0,
            },
        );
        {
            let mut cursor: Vec<u32> = self.row_start[..n_rows].to_vec();
            for c in &self.cells {
                let slot = &mut cursor[c.y as usize];
                self.sorted[*slot as usize] = *c;
                *slot += 1;
            }
        }
        self.cells.clear();
        let width = self.width as usize;
        for row in 0..n_rows {
            let (s, e) = (
                self.row_start[row] as usize,
                self.row_start[row + 1] as usize,
            );
            if s == e {
                continue;
            }
            let cells = &mut self.sorted[s..e];
            cells.sort_unstable_by_key(|c| c.x);
            let x_first = cells[0].x as usize;
            self.line.clear();
            let mut cover = 0i64;
            let mut i = 0;
            while i < cells.len() {
                let x = cells[i].x as usize;
                // Pixels between the previous cell and this one take the
                // running winding alone.
                let run = rule_byte(cover * 2 * ONE, eo);
                self.line.resize(x - x_first, run);
                let mut area = 0i64;
                while i < cells.len() && cells[i].x as usize == x {
                    cover += i64::from(cells[i].cover);
                    area += i64::from(cells[i].area);
                    i += 1;
                }
                self.line.push(rule_byte(cover * 2 * ONE - area, eo));
            }
            let tail = rule_byte(cover * 2 * ONE, eo);
            if tail != 0 {
                self.line.resize(width - x_first, tail);
            }
            let y = self.y0 + row as i32;
            emit(y, self.x0 + x_first as i32, &mut self.line);
        }
    }
}

/// The values `base + ⌊n_k / d⌋` of a linear function at evenly spaced
/// points, `n_k = n_0 + k · step`, stepped without division.
struct Dda {
    q: i64,
    r: i64,
    lift: i64,
    rem: i64,
    d: i64,
}

impl Dda {
    /// The stepper at `n_0 = factors.0 · factors.1` for the divisor
    /// `divisor > 0`.
    fn new(base: i64, factors: (i64, i64), divisor: i64, step: i64) -> Self {
        let (quotient, remainder) = factors.0.checked_mul(factors.1).map_or_else(
            || {
                let n0 = i128::from(factors.0) * i128::from(factors.1);
                let wide = i128::from(divisor);
                let quotient = i64::try_from(n0.div_euclid(wide)).unwrap_or(if n0 < 0 {
                    -COORD_LIMIT
                } else {
                    COORD_LIMIT
                });
                (quotient, i64::try_from(n0.rem_euclid(wide)).unwrap_or(0))
            },
            |n0| (n0.div_euclid(divisor), n0.rem_euclid(divisor)),
        );
        Self {
            q: base.saturating_add(quotient),
            r: remainder,
            lift: step.div_euclid(divisor),
            rem: step.rem_euclid(divisor),
            d: divisor,
        }
    }

    /// The value at the current point.
    const fn value(&self) -> i64 {
        self.q
    }

    /// Move to the next point.
    const fn advance(&mut self) {
        self.q = self.q.saturating_add(self.lift);
        self.r += self.rem;
        if self.r >= self.d {
            self.r -= self.d;
            self.q = self.q.saturating_add(1);
        }
    }
}

#[cfg(test)]
mod tests;
