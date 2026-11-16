//! The page raster on the host: the device kernel's algorithm, tile by
//! tile, in the same integer arithmetic.  It is the oracle the device is
//! tested against and the model a twin of the kernel on another device
//! API reproduces.

use super::path::{NO_BACKDROP, TILE_PIXELS, TileCells};
use super::{KIND_PATH, PAGE_TILE, PageCommands, PageCommandsError};

/// `x / M` rounded to nearest for `x ≤ M²`, `M = 2^bits − 1`.
const fn div_max(x: u32, bits: u32) -> u32 {
    let t = x + (1 << (bits - 1));
    let v = (t + (t >> bits)) >> bits;
    let max = (1 << bits) - 1;
    if v > max { max } else { v }
}

/// Coverage of pixel `x` in spans `[x0, x1, offset]` of one row, sorted.
fn row_coverage(spans: &[[i32; 3]], bytes: &[u8], x: i32) -> u8 {
    let i = spans.partition_point(|s| s[1] < x);
    match spans.get(i) {
        Some(&[x0, _, offset]) if x0 <= x => {
            if offset < 0 {
                u8::MAX
            } else {
                bytes[usize::try_from(offset + x - x0).unwrap_or(0)]
            }
        }
        _ => 0,
    }
}

/// The coverage of the pixels of tile `(tx, ty)`, row-major, by the item
/// `[command, list_start, list_len, backdrop]` of a binning of `page`.
#[expect(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "pixel coordinates of a page within the push limit fit i32"
)]
fn item_coverage(page: &PageCommands, item: [u32; 4], tx: usize, ty: usize) -> [u8; TILE_PIXELS] {
    let [cmd, list_start, list_len, backdrop] = item;
    let c = &page.cmds[cmd as usize];
    let tile = PAGE_TILE as usize;
    if c.kind == KIND_PATH {
        let mut cells = TileCells::new();
        if backdrop != NO_BACKDROP {
            cells.add_backdrop(&page.backdrops[backdrop as usize * tile..][..tile]);
        }
        let (ox, oy) = ((tx * tile) as i64 * 256, (ty * tile) as i64 * 256);
        for &s in &page.seg_lists[list_start as usize..][..list_len as usize] {
            let [x0, y0, x1, y1] = page.segs[s as usize].map(i64::from);
            let rel = [x0 - ox, y0 - oy, x1 - ox, y1 - oy];
            for row in 0..PAGE_TILE {
                cells.add_row(rel, i64::from(row));
            }
        }
        return cells.coverage(c.flags & 1 != 0, ((c.flags >> 8) & 0xFF) as u8);
    }
    let mut coverage = [0u8; TILE_PIXELS];
    for (i, v) in coverage.iter_mut().enumerate() {
        let (x, y) = ((tx * tile + i % tile) as i32, (ty * tile + i / tile) as i32);
        let row = y - c.table_y0;
        if row < 0 || row as u32 >= c.n_rows {
            continue;
        }
        let r = c.row_base as usize + row as usize;
        let spans = &page.spans[page.row_starts[r] as usize..page.row_starts[r + 1] as usize];
        *v = row_coverage(spans, &page.bytes, x);
    }
    coverage
}

/// The pixels of `page`, RGB samples of its depth, row-major without
/// padding, as the device raster renders them.
///
/// # Errors
///
/// As [`PageCommands::bin`].
#[expect(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    reason = "pixel coordinates of a page within the push limit fit i32; a composited sample \
              fits its depth"
)]
pub fn render_page(page: &PageCommands) -> Result<Vec<u16>, PageCommandsError> {
    let binned = page.bin()?;
    let (width, height) = (page.width() as usize, page.height() as usize);
    let bits = u32::from(page.depth().bits());
    let max_sample = (1u32 << bits) - 1;
    let shape_scale = max_sample / 255;
    let mut out: Vec<u16> = (0..width * height).flat_map(|_| page.paper()).collect();
    let tile = PAGE_TILE as usize;
    for ty in 0..binned.tiles_y as usize {
        for tx in 0..binned.tiles_x as usize {
            let t = ty * binned.tiles_x as usize + tx;
            for &item in &binned.items[binned.offsets[t] as usize..binned.offsets[t + 1] as usize] {
                let c = &page.cmds[item[0] as usize];
                let src = [c.rg & 0xFFFF, c.rg >> 16, c.b];
                for (i, &cov) in item_coverage(page, item, tx, ty).iter().enumerate() {
                    let (px, py) = (tx * tile + i % tile, ty * tile + i / tile);
                    let inside = px < width
                        && py < height
                        && (c.x0..=c.x1).contains(&(px as i32))
                        && (c.y0..=c.y1).contains(&(py as i32));
                    if !inside || cov == 0 {
                        continue;
                    }
                    let alpha = div_max(c.a_input * (u32::from(cov) * shape_scale), bits);
                    let o = (py * width + px) * 3;
                    for (d, s) in out[o..o + 3].iter_mut().zip(src) {
                        *d = div_max((max_sample - alpha) * u32::from(*d) + alpha * s, bits) as u16;
                    }
                }
            }
        }
    }
    Ok(out)
}
