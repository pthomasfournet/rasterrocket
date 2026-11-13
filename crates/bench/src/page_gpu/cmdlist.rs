//! A page as a GPU command list, and its CPU render through the raster
//! crate.
//!
//! Each star becomes one `KIND_AA` command carrying the CPU scanner's
//! intersection table (`raster::fill::aa_scanner`, the geometry stage of the
//! CPU's antialiased fill), its colour and its pixel bounds.  The page's
//! paper is a `KIND_SIMPLE` command over the whole page, first in order.

use bytemuck::{Pod, Zeroable};
use color::Rgb8;
use raster::types::{AA_SIZE, BlendMode};
use raster::{Bitmap, Clip, PipeSrc, PipeState, TransferSet};

use crate::stars::{Star, device_path};

/// Side of a binning tile in pixels; equals the fine-raster workgroup side.
pub const TILE: u32 = 16;
/// Every pixel of the bounds takes the colour.
pub const KIND_SIMPLE: u32 = 0;
/// Scanline-AA fill from an intersection table.
pub const KIND_AA: u32 = 1;

/// Path flatness the interpreter fills with by default.
pub const FLATNESS: f64 = 0.1;
/// Identity: the device paths are already in device space.
pub const DEVICE_MATRIX: [f64; 6] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

/// One command; layout matches `Cmd` in `page_raster.slang`.
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
#[repr(C)]
pub struct Cmd {
    pub kind: u32,
    pub eo: u32,
    pub rgb: u32,
    pub a_input: u32,
    pub table_y0: i32,
    pub n_rows: u32,
    pub row_base: u32,
    pub pad: u32,
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

/// A page's command list and side buffers.
#[derive(Default)]
pub struct CmdList {
    pub width: u32,
    pub height: u32,
    pub cmds: Vec<Cmd>,
    /// Intersections as `[x0, x1, count]` triples, all tables concatenated.
    pub intersects: Vec<i32>,
    /// Absolute offsets into the triples; each table owns `n_rows + 1`.
    pub row_starts: Vec<u32>,
    /// Sum over commands of the tiles their bounds touch: the length of
    /// the binned tile lists.
    pub tile_refs: u32,
}

const fn pack_rgb([r, g, b]: [u8; 3]) -> u32 {
    r as u32 | (g as u32) << 8 | (b as u32) << 16
}

/// The page clip the interpreter starts a page with.
pub fn page_clip(width: u32, height: u32) -> Clip {
    Clip::new(0.0, 0.0, f64::from(width), f64::from(height), true)
}

impl CmdList {
    /// Clear for a page of `width × height` pixels and append its paper.
    pub fn begin(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.cmds.clear();
        self.intersects.clear();
        self.row_starts.clear();
        self.tile_refs = 0;
        let (w, h) = (to_i32(width), to_i32(height));
        self.push(Cmd {
            kind: KIND_SIMPLE,
            rgb: pack_rgb([255; 3]),
            a_input: 255,
            x0: 0,
            y0: 0,
            x1: w - 1,
            y1: h - 1,
            ..Cmd::default()
        });
    }

    fn push(&mut self, cmd: Cmd) {
        let tiles = |lo: i32, hi: i32| {
            u32::try_from(hi / to_i32(TILE) - lo / to_i32(TILE) + 1).expect("lo <= hi")
        };
        self.tile_refs += tiles(cmd.x0, cmd.x1) * tiles(cmd.y0, cmd.y1);
        self.cmds.push(cmd);
    }

    /// Append the non-zero fill of `star` (device path at `scale`).
    ///
    /// The pixel bounds are the scanner's AA bounds divided by `AA_SIZE` and
    /// clamped to the page.  A row whose winding does not return to zero
    /// paints to the page's right edge, so such a table's bounds span the
    /// page's width.  A star that scan-converts to nothing, or lies off
    /// the page, appends nothing.
    pub fn push_star(&mut self, star: &Star, scale: f64, clip: &Clip) {
        let path = device_path(star, scale, self.height);
        let hinted = path.with_fill_adjust_hints();
        let path = hinted.as_ref().unwrap_or(&path);
        let Some(scanner) = raster::fill::aa_scanner(path, &DEVICE_MATRIX, FLATNESS, clip, false)
        else {
            return;
        };
        let (w, h) = (to_i32(self.width), to_i32(self.height));
        let x0 = (scanner.x_min / AA_SIZE).max(0);
        let mut x1 = (scanner.x_max / AA_SIZE).min(w - 1);
        let y0 = (scanner.y_min / AA_SIZE).max(0);
        let y1 = (scanner.y_max / AA_SIZE).min(h - 1);
        if x0 > x1 || y0 > y1 {
            return;
        }
        let starts = scanner.row_starts();
        let ix = scanner.intersects();
        let balanced = starts.windows(2).all(|r| {
            ix[r[0] as usize..r[1] as usize]
                .iter()
                .fold(0i32, |acc, i| acc.wrapping_add(i.count))
                == 0
        });
        if !balanced {
            x1 = w - 1;
        }
        let base = u32::try_from(self.intersects.len() / 3).expect("table fits u32");
        let row_base = u32::try_from(self.row_starts.len()).expect("table fits u32");
        self.intersects.reserve(ix.len() * 3);
        for i in ix {
            self.intersects.extend_from_slice(&[i.x0, i.x1, i.count]);
        }
        self.row_starts.extend(starts.iter().map(|&s| s + base));
        self.push(Cmd {
            kind: KIND_AA,
            eo: 0,
            rgb: pack_rgb(star.rgb),
            a_input: 255,
            table_y0: scanner.y_min,
            n_rows: u32::try_from(starts.len() - 1).expect("rows fit u32"),
            row_base,
            pad: 0,
            x0,
            y0,
            x1,
            y1,
        });
    }

    /// The whole page's command list.
    pub fn build(&mut self, stars: &[Star], scale: f64, width: u32, height: u32) {
        self.begin(width, height);
        let clip = page_clip(width, height);
        for star in stars {
            self.push_star(star, scale, &clip);
        }
    }

    /// Bytes of the command list and its side buffers.
    pub fn bytes(&self) -> usize {
        size_of_val(self.cmds.as_slice())
            + size_of_val(self.intersects.as_slice())
            + size_of_val(self.row_starts.as_slice())
    }
}

fn to_i32(v: u32) -> i32 {
    i32::try_from(v).expect("page dimension fits i32")
}

/// The page rendered by the raster crate's antialiased fill, as the
/// interpreter fills an opaque solid path: white paper, `Normal` blend,
/// `a_input` 255, identity transfer, page clip.
pub fn render_cpu(stars: &[Star], scale: f64, width: u32, height: u32) -> Bitmap<Rgb8> {
    let mut bmp = Bitmap::<Rgb8>::new_filled(width, height, 1, false, u8::MAX);
    let clip = page_clip(width, height);
    let pipe = PipeState {
        blend_mode: BlendMode::Normal,
        a_input: 255,
        transfer: TransferSet::identity_rgb(),
        backdrop: None,
        overprint: None,
    };
    for star in stars {
        let path = device_path(star, scale, height);
        let hinted = path.with_fill_adjust_hints();
        let path = hinted.as_ref().unwrap_or(&path);
        raster::fill(
            &mut bmp,
            &clip,
            path,
            &pipe,
            &PipeSrc::Solid(&star.rgb),
            &DEVICE_MATRIX,
            FLATNESS,
            true,
        );
    }
    bmp
}

/// The bitmap's pixels without row padding.
pub fn packed_rows(bmp: &Bitmap<Rgb8>) -> Vec<u8> {
    let row = bmp.width as usize * 3;
    bmp.data()
        .chunks(bmp.stride)
        .take(bmp.height as usize)
        .flat_map(|r| &r[..row])
        .copied()
        .collect()
}
