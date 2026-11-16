//! Outlines rasterised on the device: the exact-area coverage of a closed
//! outline, evaluated tile by tile.
//!
//! The coverage of a pixel is that of the CPU's exact-area rasteriser over
//! the command's window.  That rasteriser's coverage of a pixel does not
//! depend on the window it runs over: the position of a segment on every
//! row and cell boundary is computed from the segment's endpoints alone,
//! the cells left of the window fold into one deposit of their cover, with
//! no area, at the window's left column, and the cells right of it touch
//! no pixel in it.  A tile taken as the window therefore yields the CPU's
//! coverage of every pixel in it, and the command's window masks the rest.
//!
//! The host lists, for every tile a command's window meets, the segments
//! that may deposit a cell in the tile ([`bin_path`]), and the tile's
//! backdrop: per pixel row, the summed cover of the segments that lie
//! wholly left of it within the tile's rows, which fold into its left
//! column.  The device then walks only the listed segments.
//!
//! Every quantity stays exact in the integer widths of the kernel: segment
//! coordinates are below [`PATH_COORD_LIMIT`] in magnitude, so the
//! products of two coordinate differences fit `i64`; a cell sum adds at
//! most [`PATH_TILE_SEGMENTS`] deposits of area at most `2^17`, so it fits
//! `i32`; a backdrop adds the cover of at most [`PATH_SEGMENTS`] segments,
//! at most 256 each.

use super::{PAGE_TILE, PageCommandsError, PixelBounds};

/// Fraction bits of a fixed-point coordinate.
const SUBPIXEL_BITS: u32 = 8;

/// One pixel in fixed point.
const ONE: i64 = 1 << SUBPIXEL_BITS;

/// One tile in fixed point.
const TILE_FIXED: i64 = ONE * PAGE_TILE as i64;

/// The accumulated area of a fully covered pixel.
const FULL_AREA: i64 = 2 * ONE * ONE;

/// Pixels of a tile.
pub(crate) const TILE_PIXELS: usize = (PAGE_TILE * PAGE_TILE) as usize;

/// Coordinates of a device outline lie strictly inside
/// `±PATH_COORD_LIMIT`, fixed point.
pub const PATH_COORD_LIMIT: i64 = 1 << 30;

/// The most segments one tile of an outline may list.
pub const PATH_TILE_SEGMENTS: usize = (1 << 14) - 1;

/// The most segments one outline may hold.
pub const PATH_SEGMENTS: usize = 1 << 22;

/// A closed outline for the device raster.
///
/// The segments are device coordinates in the exact-area rasteriser's
/// fixed point (`floor(256 · v)`), `[x0, y0, x1, y1]` in the direction the
/// outline runs; the coverage is that rasteriser's over `window` under the
/// even-odd rule when `even_odd`, else non-zero, scaled by `weight / 255`.
#[derive(Clone, Copy, Debug)]
pub struct AreaPath<'a> {
    /// The segments, fixed point.
    pub segs: &'a [[i64; 4]],
    /// The pixels the coverage is evaluated over.
    pub window: PixelBounds,
    /// Even-odd rule, else non-zero.
    pub even_odd: bool,
    /// Coverage weight out of 255.
    pub weight: u8,
}

/// One tile of a binned outline: the tile's index, its segments
/// `lists[list_start..][..list_len]` (indices into the outline's
/// segments), and its backdrop `backdrops[backdrop..][..PAGE_TILE]`, or
/// [`NO_BACKDROP`]; and whether it hides what lies below it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PathTile {
    pub tile: u32,
    pub list_start: u32,
    pub list_len: u32,
    pub backdrop: u32,
    /// Whether the tile hides the items before it
    /// ([`super::PageCommands::bin`]).
    pub occludes: bool,
}

/// The backdrop index of a tile whose backdrop is zero on every row.
pub(crate) const NO_BACKDROP: u32 = u32::MAX;

/// An outline binned into the tiles of a page: the tiles that may hold
/// coverage, in raster order, with indices relative to this binning.
#[derive(Debug, Default)]
pub(crate) struct BinnedPath {
    pub tiles: Vec<PathTile>,
    /// Segment indices, relative to the outline.
    pub lists: Vec<u32>,
    /// `PAGE_TILE` covers per backdrop.
    pub backdrops: Vec<i32>,
}

/// What decides whether a tile of an outline hides the items below it.
struct Occlusion {
    /// The command paints at full alpha and weight.
    opaque: bool,
    even_odd: bool,
    /// The outline's window, within the page.
    window: PixelBounds,
    /// The page's pixels.
    page: PixelBounds,
}

impl Occlusion {
    /// Whether the pixels of tile `(col, band)` that lie on the page lie
    /// in the window.
    fn holds(&self, col: i64, band: i64) -> bool {
        let tile = i64::from(PAGE_TILE);
        let clamp = |v: i64| i32::try_from(v).unwrap_or(i32::MAX);
        let on_page = PixelBounds {
            x0: clamp(col * tile),
            y0: clamp(band * tile),
            x1: clamp((col + 1) * tile - 1).min(self.page.x1),
            y1: clamp((band + 1) * tile - 1).min(self.page.y1),
        };
        on_page.intersect(self.window) == Some(on_page)
    }

    /// Whether a tile of band `band` listing no segment, with backdrop
    /// `backdrop`, covers its rows on the page fully.
    fn full(&self, band: i64, backdrop: &[i32; PAGE_TILE as usize]) -> bool {
        let rows = usize::try_from(
            (i64::from(self.page.y1) - band * i64::from(PAGE_TILE) + 1).min(i64::from(PAGE_TILE)),
        )
        .unwrap_or(0);
        self.opaque && backdrop[..rows].iter().all(|&b| full_row(b, self.even_odd))
    }
}

impl BinnedPath {
    /// Append the tiles of band `band`, columns `col0..`: column `k` lists
    /// the segments `lists[k]` and has the backdrop summed from
    /// `deltas[..=k]`, which changes only at the columns `changed[k]`.  A
    /// tile with neither is left out.
    fn push_band(
        &mut self,
        (lists, deltas, changed): (&[Vec<u32>], &[[i32; PAGE_TILE as usize]], &[bool]),
        (band, col0, tiles_x): (i64, i64, u32),
        occlusion: &Occlusion,
    ) -> Result<(), PageCommandsError> {
        let too_large = |_| PageCommandsError::TooLarge;
        self.tiles.try_reserve(lists.len())?;
        self.lists.try_reserve(lists.iter().map(Vec::len).sum())?;
        let mut backdrop = [0i32; PAGE_TILE as usize];
        // The current backdrop: whether it is non-zero, its index once
        // stored, and whether it covers a tile's rows fully.
        let (mut shaded, mut stored, mut full) = (false, None, false);
        for (k, (list, col)) in lists.iter().zip(col0..).enumerate() {
            if changed[k] {
                for (b, d) in backdrop.iter_mut().zip(&deltas[k]) {
                    *b += d;
                }
                shaded = backdrop.iter().any(|&b| b != 0);
                stored = None;
                full = shaded && occlusion.full(band, &backdrop);
            }
            if list.is_empty() && !shaded {
                continue;
            }
            if list.len() > PATH_TILE_SEGMENTS {
                return Err(PageCommandsError::Unrepresentable);
            }
            let list_start = u32::try_from(self.lists.len()).map_err(too_large)?;
            self.lists.extend_from_slice(list);
            let backdrop_index = match stored {
                _ if !shaded => NO_BACKDROP,
                Some(at) => at,
                None => {
                    self.backdrops.try_reserve(PAGE_TILE as usize)?;
                    let at = u32::try_from(self.backdrops.len() / PAGE_TILE as usize)
                        .map_err(too_large)?;
                    self.backdrops.extend_from_slice(&backdrop);
                    stored = Some(at);
                    at
                }
            };
            self.tiles.push(PathTile {
                tile: u32::try_from(band * i64::from(tiles_x) + col).map_err(too_large)?,
                list_start,
                list_len: u32::try_from(list.len()).map_err(too_large)?,
                backdrop: backdrop_index,
                occludes: list.is_empty() && full && occlusion.holds(col, band),
            });
        }
        Ok(())
    }
}

/// `⌊a / b⌋` for `b > 0`.
const fn floor_div(a: i64, b: i64) -> i64 {
    a.div_euclid(b)
}

/// The x coordinate of segment `(xa, ya) → (xb, yb)`, `ya < yb`, at height
/// `y ∈ [ya, yb]`: an endpoint's own x at its height, else the rasteriser's
/// boundary position `xa + ⌊(y − ya) · (xb − xa) / (yb − ya)⌋`.
const fn x_at(xa: i64, ya: i64, xb: i64, yb: i64, y: i64) -> i64 {
    if y == ya {
        xa
    } else if y == yb {
        xb
    } else {
        xa + floor_div((y - ya) * (xb - xa), yb - ya)
    }
}

/// `seg` ordered top to bottom, with the sign that restores its
/// direction; `None` for a horizontal segment, which deposits nothing.
const fn downward(seg: [i64; 4]) -> Option<(i64, [i64; 4])> {
    let [x0, y0, x1, y1] = seg;
    if y0 < y1 {
        Some((1, seg))
    } else if y0 > y1 {
        Some((-1, [x1, y1, x0, y0]))
    } else {
        None
    }
}

/// Bin `path` over the tiles of the page `page`, `tiles_x` tiles wide;
/// its window lies on the page, and its command paints at full alpha when
/// `opaque`.
///
/// # Errors
///
/// [`PageCommandsError::Unrepresentable`] when a coordinate, the number of
/// segments or the segments of one tile exceed the device raster's
/// integer range; [`PageCommandsError::Alloc`] when memory cannot be
/// reserved.
pub(crate) fn bin_path(
    path: &AreaPath<'_>,
    page: PixelBounds,
    tiles_x: u32,
    opaque: bool,
) -> Result<BinnedPath, PageCommandsError> {
    let in_range = |v: i64| v.abs() < PATH_COORD_LIMIT;
    if path.segs.len() > PATH_SEGMENTS || !path.segs.iter().flatten().copied().all(in_range) {
        return Err(PageCommandsError::Unrepresentable);
    }
    let w = path.window;
    let tile = |v: i32| i64::from(v) / i64::from(PAGE_TILE);
    let (tc0, tc1, tb0, tb1) = (tile(w.x0), tile(w.x1), tile(w.y0), tile(w.y1));
    let n_cols = usize::try_from(tc1 - tc0 + 1).map_err(|_| PageCommandsError::TooLarge)?;
    // Fixed-point rows of the window: [wy0, wy1).
    let (wy0, wy1) = (i64::from(w.y0) * ONE, (i64::from(w.y1) + 1) * ONE);
    let mut out = BinnedPath::default();
    let mut lists: Vec<Vec<u32>> = vec![Vec::new(); n_cols];
    let mut deltas = vec![[0i32; PAGE_TILE as usize]; n_cols];
    let mut changed = vec![false; n_cols];
    let occlusion = Occlusion {
        opaque: opaque && path.weight == u8::MAX,
        even_odd: path.even_odd,
        window: w,
        page,
    };
    // The segments meeting each band, found once per segment.
    let mut by_band: Vec<Vec<u32>> =
        vec![Vec::new(); usize::try_from(tb1 - tb0 + 1).map_err(|_| PageCommandsError::TooLarge)?];
    for (i, &seg) in path.segs.iter().enumerate() {
        let Some((_, [_, ya, _, yb])) = downward(seg) else {
            continue;
        };
        let (top, bottom) = (ya.max(wy0), yb.min(wy1));
        if top >= bottom {
            continue;
        }
        let first = (top >> SUBPIXEL_BITS) / i64::from(PAGE_TILE);
        let last = ((bottom - 1) >> SUBPIXEL_BITS) / i64::from(PAGE_TILE);
        let index = u32::try_from(i).map_err(|_| PageCommandsError::Unrepresentable)?;
        for band in first..=last {
            by_band[usize::try_from(band - tb0).unwrap_or(0)].push(index);
        }
    }
    for band in tb0..=tb1 {
        let band_top = (band * TILE_FIXED).max(wy0);
        let band_bottom = ((band + 1) * TILE_FIXED).min(wy1);
        for l in &mut lists {
            l.clear();
        }
        for (d, c) in deltas.iter_mut().zip(&mut changed) {
            if *c {
                *d = [0; PAGE_TILE as usize];
                *c = false;
            }
        }
        for &i in &by_band[usize::try_from(band - tb0).unwrap_or(0)] {
            let Some((sign, [xa, ya, xb, yb])) = downward(path.segs[i as usize]) else {
                continue;
            };
            let (top, bottom) = (ya.max(band_top), yb.min(band_bottom));
            let (xt, xbot) = (x_at(xa, ya, xb, yb, top), x_at(xa, ya, xb, yb, bottom));
            let (lo, hi) = (xt.min(xbot), xt.max(xbot));
            // Tiles from column `left` on lie wholly right of the segment
            // within the band: it folds into their backdrop.
            let left = (hi + TILE_FIXED - 1).div_euclid(TILE_FIXED);
            let first = (lo >> SUBPIXEL_BITS).div_euclid(i64::from(PAGE_TILE));
            for col in first.max(tc0)..=(left - 1).min(tc1) {
                lists[usize::try_from(col - tc0).unwrap_or(0)].push(i);
            }
            if left <= tc1 {
                let at = usize::try_from(left.max(tc0) - tc0).unwrap_or(0);
                changed[at] = true;
                let slot = &mut deltas[at];
                let row0 = band * i64::from(PAGE_TILE);
                for (r, d) in (0i64..).zip(slot.iter_mut()) {
                    let row_top = (row0 + r) * ONE;
                    let h = yb.min(row_top + ONE).min(bottom) - ya.max(row_top).max(top);
                    if h > 0 {
                        *d += i32::try_from(sign * h).unwrap_or(0);
                    }
                }
            }
        }
        out.push_band(
            (&lists, &deltas, &changed),
            (band, tc0, tiles_x),
            &occlusion,
        )?;
    }
    Ok(out)
}

/// The values `base + ⌊(n0 + k · step) / d⌋`, `k = 0, 1, …`, stepped
/// without division; `d > 0`.
struct Dda {
    q: i64,
    r: i64,
    lift: i64,
    rem: i64,
    d: i64,
}

impl Dda {
    const fn new(base: i64, n0: i64, d: i64, step: i64) -> Self {
        Self {
            q: base + n0.div_euclid(d),
            r: n0.rem_euclid(d),
            lift: step.div_euclid(d),
            rem: step.rem_euclid(d),
            d,
        }
    }

    const fn advance(&mut self) {
        self.q += self.lift;
        self.r += self.rem;
        if self.r >= self.d {
            self.r -= self.d;
            self.q += 1;
        }
    }
}

/// The cells of one tile: per pixel row and column, the summed cover and
/// area of the deposits there, row-major.
pub(crate) struct TileCells {
    pub cover: [i32; TILE_PIXELS],
    pub area: [i32; TILE_PIXELS],
}

impl TileCells {
    pub(crate) const fn new() -> Self {
        Self {
            cover: [0; TILE_PIXELS],
            area: [0; TILE_PIXELS],
        }
    }

    /// Add the backdrop `covers`, one per row, to the tile's left column.
    pub(crate) fn add_backdrop(&mut self, covers: &[i32]) {
        for (r, &c) in covers.iter().enumerate().take(PAGE_TILE as usize) {
            self.cover[r * PAGE_TILE as usize] += c;
        }
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "0 <= ex, ey < PAGE_TILE; a deposit's cover and area fit i32"
    )]
    const fn deposit(&mut self, ex: i64, ey: i64, cover: i64, area: i64) {
        let at = ey as usize * PAGE_TILE as usize + ex as usize;
        self.cover[at] += cover as i32;
        self.area[at] += area as i32;
    }

    /// Deposit pixel row `row` of segment `seg`, fixed point relative to
    /// the tile's top-left corner, as the CPU's rasteriser deposits it over
    /// the tile taken as its window; rows outside the tile deposit nothing.
    pub(crate) fn add_row(&mut self, seg: [i64; 4], row: i64) {
        let Some((sign, [xa, ya, xb, yb])) = downward(seg) else {
            return;
        };
        if row < 0 || row >= i64::from(PAGE_TILE) {
            return;
        }
        let row_top = row * ONE;
        if yb <= row_top || ya >= row_top + ONE {
            return;
        }
        let (x_top, fy_top) = if ya >= row_top {
            (xa, ya - row_top)
        } else {
            (x_at(xa, ya, xb, yb, row_top), 0)
        };
        let (x_bot, fy_bot) = if yb <= row_top + ONE {
            (xb, yb - row_top)
        } else {
            (x_at(xa, ya, xb, yb, row_top + ONE), ONE)
        };
        if x_top <= x_bot {
            self.hline(row, (x_top, fy_top), (x_bot, fy_bot), sign);
        } else {
            self.hline(row, (x_bot, fy_bot), (x_top, fy_top), -sign);
        }
    }

    /// The rasteriser's `hline` over a window `PAGE_TILE` cells wide.
    fn hline(&mut self, ey: i64, (xl, yl): (i64, i64), (xr, yr): (i64, i64), sign: i64) {
        if yl == yr {
            return;
        }
        let w = i64::from(PAGE_TILE);
        if xl == xr {
            let ex = xl >> SUBPIXEL_BITS;
            let cover = sign * (yr - yl);
            if ex < 0 {
                self.deposit(0, ey, cover, 0);
            } else if ex < w {
                self.deposit(ex, ey, cover, cover * 2 * (xl - ex * ONE));
            }
            return;
        }
        let first_cell = xl >> SUBPIXEL_BITS;
        let end_cell = (xr - 1) >> SUBPIXEL_BITS;
        let last_cell = end_cell.min(w - 1);
        if first_cell == end_cell && first_cell >= 0 {
            if first_cell < w {
                let cover = sign * (yr - yl);
                self.deposit(
                    first_cell,
                    ey,
                    cover,
                    cover * (xl + xr - 2 * first_cell * ONE),
                );
            }
            return;
        }
        let (dx, dy) = (xr - xl, yr - yl);
        let start = first_cell.max(0);
        let mut boundary = Dda::new(yl, (start * ONE - xl) * dy, dx, ONE * dy);
        if first_cell < 0 {
            let y0 = if xr <= 0 { yr } else { boundary.q };
            self.deposit(0, ey, sign * (y0 - yl), 0);
        }
        let mut cell = start;
        while cell <= last_cell {
            let cell_left = cell * ONE;
            let (x_in, y_in) = if xl >= cell_left {
                (xl, yl)
            } else {
                (cell_left, boundary.q)
            };
            boundary.advance();
            let (x_out, y_out) = if xr <= cell_left + ONE {
                (xr, yr)
            } else {
                (cell_left + ONE, boundary.q)
            };
            let cover = sign * (y_out - y_in);
            self.deposit(cell, ey, cover, cover * (x_in + x_out - 2 * cell_left));
            cell += 1;
        }
    }

    /// The coverage byte of every pixel of the tile under the even-odd rule
    /// when `even_odd`, else non-zero, scaled by `weight / 255`.
    pub(crate) fn coverage(&self, even_odd: bool, weight: u8) -> [u8; TILE_PIXELS] {
        let mut out = [0u8; TILE_PIXELS];
        for r in 0..PAGE_TILE as usize {
            let mut acc = 0i64;
            for c in 0..PAGE_TILE as usize {
                let at = r * PAGE_TILE as usize + c;
                acc += i64::from(self.cover[at]);
                let v = acc * 2 * ONE - i64::from(self.area[at]);
                out[at] = weighted(rule_byte(v, even_odd), weight);
            }
        }
        out
    }
}

/// Whether a pixel row whose cells hold no deposit and whose backdrop is
/// `cover` is covered fully under the even-odd rule when `even_odd`, else
/// non-zero.
pub(crate) const fn full_row(cover: i32, even_odd: bool) -> bool {
    rule_byte(cover as i64 * 2 * ONE, even_odd) == u8::MAX
}

/// The rasteriser's coverage byte of the accumulated value `v`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the area lies in [0, FULL_AREA], which maps into [0, 255]"
)]
const fn rule_byte(v: i64, even_odd: bool) -> u8 {
    let a = if even_odd {
        let m = v.rem_euclid(2 * FULL_AREA);
        if m > FULL_AREA { 2 * FULL_AREA - m } else { m }
    } else {
        let m = v.abs();
        if m > FULL_AREA { FULL_AREA } else { m }
    };
    ((a * 255 + FULL_AREA / 2) / FULL_AREA) as u8
}

/// `c · weight / 255`, rounded to nearest; `c` when `weight` is 255.
#[expect(clippy::cast_possible_truncation, reason = "the result is at most 255")]
const fn weighted(c: u8, weight: u8) -> u8 {
    if weight == u8::MAX {
        return c;
    }
    let t = c as u32 * weight as u32 + 0x80;
    let v = (t + (t >> 8)) >> 8;
    (if v > 255 { 255 } else { v }) as u8
}

#[cfg(test)]
mod tests;
