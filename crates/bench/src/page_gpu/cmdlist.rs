//! A page as a GPU command list, and its CPU render through the raster
//! crate.
//!
//! Each star becomes one command per coverage table of its antialiased
//! fill (`raster::fill::aa_fill_coverage`, the geometry stage of the CPU's
//! exact-area fill), carrying its colour and the table's pixel bounds on
//! the page.  The paper is the page's initial colour, not a command.

use bytemuck::{Pod, Zeroable};
use color::Rgb8;
use raster::types::BlendMode;
use raster::{Bitmap, Clip, PipeSrc, PipeState, TransferSet};

use crate::stars::{Star, device_path};

/// Side of a binning tile in pixels; equals the fine-raster workgroup side.
pub const TILE: u32 = 16;
/// The page's colour before the first command: the white of `render_cpu`.
pub const PAPER: [u8; 3] = [u8::MAX; 3];

/// Path flatness the interpreter fills with by default.
pub const FLATNESS: f64 = 0.1;
/// Identity: the device paths are already in device space.
pub const DEVICE_MATRIX: [f64; 6] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

/// One command; layout matches `Cmd` in `page_raster.slang`.
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
#[repr(C)]
pub struct Cmd {
    pub kind: u32,
    pub reserved: u32,
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
    /// Spans as `[x0, x1, offset]` triples, all tables concatenated;
    /// `offset` is −1 for a fully covered span, else an index into
    /// `coverage`.
    pub spans: Vec<i32>,
    /// The partial spans' coverage bytes, all tables concatenated.
    pub coverage: Vec<u8>,
    /// Absolute offsets into the triples; each table owns `n_rows + 1`.
    pub row_starts: Vec<u32>,
    /// Sum over commands of the tiles their bounds touch: the length of
    /// the binned tile lists.
    pub tile_refs: u32,
}

/// `[r, g, b]` as the kernel's packed `r | g << 8 | b << 16`.
pub const fn pack_rgb([r, g, b]: [u8; 3]) -> u32 {
    r as u32 | (g as u32) << 8 | (b as u32) << 16
}

/// The page clip the interpreter starts a page with.
pub fn page_clip(width: u32, height: u32) -> Clip {
    Clip::new(0.0, 0.0, f64::from(width), f64::from(height), true)
}

impl CmdList {
    /// Clear for a page of `width × height` pixels.
    pub fn begin(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.cmds.clear();
        self.spans.clear();
        self.coverage.clear();
        self.row_starts.clear();
        self.tile_refs = 0;
    }

    /// Append the non-zero fill of `star` (device path at `scale`): one
    /// command per coverage table, in compositing order.
    ///
    /// A command's pixel bounds are its table's bounds clamped to the
    /// page.  A table that covers no pixel of the page appends nothing.
    pub fn push_star(&mut self, star: &Star, scale: f64, clip: &Clip) {
        let path = device_path(star, scale, self.height);
        let hinted = path.with_fill_adjust_hints();
        let path = hinted.as_ref().unwrap_or(&path);
        let tables = raster::fill::aa_fill_coverage(
            path,
            &DEVICE_MATRIX,
            FLATNESS,
            clip,
            self.width,
            self.height,
            false,
            true,
        );
        let (w, h) = (to_i32(self.width), to_i32(self.height));
        for table in &tables {
            let Some((bx0, by0, bx1, by1)) = table.bounds() else {
                continue;
            };
            let (x0, y0) = (bx0.max(0), by0.max(0));
            let (x1, y1) = (bx1.min(w - 1), by1.min(h - 1));
            if x0 > x1 || y0 > y1 {
                continue;
            }
            let span_base = u32::try_from(self.spans.len() / 3).expect("spans fit u32");
            let byte_base = i32::try_from(self.coverage.len()).expect("coverage fits i32");
            let row_base = u32::try_from(self.row_starts.len()).expect("rows fit u32");
            let starts = table.row_starts();
            for [sx0, sx1, offset] in table.span_triples() {
                let offset = if offset < 0 { -1 } else { offset + byte_base };
                self.spans.extend_from_slice(&[sx0, sx1, offset]);
            }
            self.coverage.extend_from_slice(table.bytes());
            self.row_starts
                .extend(starts.iter().map(|&s| s + span_base));
            self.push(Cmd {
                rgb: pack_rgb(star.rgb),
                a_input: 255,
                table_y0: table.y_min(),
                n_rows: u32::try_from(starts.len() - 1).expect("rows fit u32"),
                row_base,
                x0,
                y0,
                x1,
                y1,
                ..Cmd::default()
            });
        }
    }

    fn push(&mut self, cmd: Cmd) {
        let tiles = |lo: i32, hi: i32| {
            u32::try_from(hi / to_i32(TILE) - lo / to_i32(TILE) + 1).expect("lo <= hi")
        };
        self.tile_refs += tiles(cmd.x0, cmd.x1) * tiles(cmd.y0, cmd.y1);
        self.cmds.push(cmd);
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
            + size_of_val(self.spans.as_slice())
            + self.coverage.len().div_ceil(4) * 4
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
