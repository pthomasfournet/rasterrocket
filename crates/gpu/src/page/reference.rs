//! The page raster on the host: the device kernel's algorithm, tile by
//! tile, in the same integer arithmetic.  It is the oracle the device is
//! tested against and the model a twin of the kernel on another device
//! API reproduces.

use super::image::{DIRECT, ImageModel};
use super::path::{LIST_LEN, NO_BACKDROP, TILE_PIXELS, TileCells, UNIFORM_BACKDROP};
use super::{
    KIND_IMAGE, KIND_MASK, KIND_PATH, PAGE_TILE, PageCmd, PageCommands, PageCommandsError,
};

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
/// `[command, list, len, backdrop]` of a binning of `page`
/// ([`super::TileBins::items`]).
#[expect(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "pixel coordinates of a page within the push limit fit i32"
)]
fn item_coverage(page: &PageCommands, item: [u32; 4], tx: usize, ty: usize) -> [u8; TILE_PIXELS] {
    let [cmd, list, len, backdrop] = item;
    let c = &page.cmds[cmd as usize];
    let tile = PAGE_TILE as usize;
    if c.kind == KIND_PATH {
        let mut cells = TileCells::new();
        if len & UNIFORM_BACKDROP != 0 {
            cells.add_backdrop(&[backdrop as i32; PAGE_TILE as usize]);
        } else if backdrop != NO_BACKDROP {
            cells.add_backdrop(&page.backdrops[backdrop as usize * tile..][..tile]);
        }
        let (ox, oy) = ((tx * tile) as i64 * 256, (ty * tile) as i64 * 256);
        let segments = match len & LIST_LEN {
            1 => std::slice::from_ref(&list),
            n => &page.seg_lists[list as usize..][..n as usize],
        };
        for &s in segments {
            let [x0, y0, x1, y1] = page.segs[s as usize].map(i64::from);
            let rel = [x0 - ox, y0 - oy, x1 - ox, y1 - oy];
            for row in 0..PAGE_TILE {
                cells.add_row(rel, i64::from(row));
            }
        }
        return cells.coverage(c.flags & 1 != 0, ((c.flags >> 8) & 0xFF) as u8);
    }
    let mut coverage = [0u8; TILE_PIXELS];
    if c.kind == KIND_MASK {
        let left = c.flags as i32;
        for (i, v) in coverage.iter_mut().enumerate() {
            let (x, y) = ((tx * tile + i % tile) as i32, (ty * tile + i / tile) as i32);
            if (c.x0..=c.x1).contains(&x) && (c.y0..=c.y1).contains(&y) {
                let at = c.row_base as usize
                    + (y - c.table_y0) as usize * c.n_rows as usize
                    + (x - left) as usize;
                *v = page.bytes[at];
            }
        }
        return coverage;
    }
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

/// Composite `src` with shape `shape > 0`, samples of `bits` bits, over
/// the pixel `dst` as the antialiasing pipe at constant alpha `a_input`.
fn composite_pipe(dst: &mut [u16], src: [u32; 3], shape: u32, a_input: u32, bits: u32) {
    let max = (1u32 << bits) - 1;
    let alpha = div_max(a_input * shape, bits);
    for (d, s) in dst.iter_mut().zip(src) {
        *d = u16::try_from(div_max((max - alpha) * u32::from(*d) + alpha * s, bits))
            .expect("a composite is a sample");
    }
}

/// Composite `src` with shape `shape > 0`, samples of `bits` bits, over
/// the pixel `dst` directly: a full shape writes `src`, any other blends
/// with the quotient truncated.
fn composite_direct(dst: &mut [u16], src: [u32; 3], shape: u32, bits: u32) {
    let max = (1u32 << bits) - 1;
    for (d, s) in dst.iter_mut().zip(src) {
        let v = if shape == max {
            s
        } else {
            (s * shape + u32::from(*d) * (max - shape)) / max
        };
        *d = u16::try_from(v).expect("a composite is a sample");
    }
}

/// Paint the pixels of tile `(tx, ty)` of `out`, a `width × height` page,
/// that the image command `c` covers.
#[expect(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    reason = "pixel coordinates of a page within the push limit fit i32"
)]
fn paint_image(
    model: &ImageModel<'_>,
    c: &PageCmd,
    (tx, ty): (usize, usize),
    (width, height): (usize, usize),
    out: &mut [u16],
) {
    let tile = PAGE_TILE as usize;
    for i in 0..TILE_PIXELS {
        let (px, py) = (tx * tile + i % tile, ty * tile + i / tile);
        let (x, y) = (px as i32, py as i32);
        if px >= width || py >= height || !(c.x0..=c.x1).contains(&x) || !(c.y0..=c.y1).contains(&y)
        {
            continue;
        }
        let Some((src, shape)) = model.pixel(c, x, y) else {
            continue;
        };
        let o = (py * width + px) * 3;
        if c.flags & DIRECT == 0 {
            composite_pipe(&mut out[o..o + 3], src, shape, c.a_input, model.bits);
        } else {
            composite_direct(&mut out[o..o + 3], src, shape, model.bits);
        }
    }
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
    reason = "pixel coordinates of a page within the push limit fit i32"
)]
pub fn render_page(page: &PageCommands) -> Result<Vec<u16>, PageCommandsError> {
    let binned = page.bin()?;
    let (width, height) = (page.width() as usize, page.height() as usize);
    let bits = u32::from(page.depth().bits());
    let shape_scale = ((1u32 << bits) - 1) / 255;
    let model = ImageModel {
        words: &page.image_words,
        bytes: &page.bytes,
        bits,
    };
    let mut out: Vec<u16> = (0..width * height).flat_map(|_| page.paper()).collect();
    let tile = PAGE_TILE as usize;
    for ty in 0..binned.tiles_y as usize {
        for tx in 0..binned.tiles_x as usize {
            let t = ty * binned.tiles_x as usize + tx;
            for &item in &binned.items[binned.offsets[t] as usize..binned.offsets[t + 1] as usize] {
                let c = &page.cmds[item[0] as usize];
                if c.kind == KIND_IMAGE {
                    paint_image(&model, c, (tx, ty), (width, height), &mut out);
                    continue;
                }
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
                    let o = (py * width + px) * 3;
                    let shape = u32::from(cov) * shape_scale;
                    composite_pipe(&mut out[o..o + 3], src, shape, c.a_input, bits);
                }
            }
        }
    }
    Ok(out)
}
