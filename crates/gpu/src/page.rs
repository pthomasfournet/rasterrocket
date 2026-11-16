//! A page as a list of paint commands for the device page raster.
//!
//! Each command paints one colour at a constant alpha over an opaque RGB
//! page of 8 or 16 bits per sample, limited to an inclusive pixel
//! rectangle, with per-pixel coverage either taken from a coverage table
//! the CPU rasteriser computed (see [`CoverageRows`]), computed on the
//! device from a closed outline by the CPU rasteriser's integer arithmetic
//! (see [`AreaPath`]), or read from a coverage mask the page holds once
//! and any number of commands place (see [`MaskId`]).  Commands paint in
//! the order they were pushed.  The tables, outlines and masks of every
//! command are copied into shared flat arrays.
//!
//! The device raster bins the commands into [`PAGE_TILE`]-pixel square
//! tiles ([`PageCommands::bin`]) and walks each tile's commands in order;
//! the per-pixel arithmetic is documented in `kernels/page_raster.slang`.
//!
//! A page's device inputs, binning included, are bounded by
//! [`PAGE_COMMANDS_MAX_BYTES`]: a push that would exceed it fails and
//! leaves the page unchanged.

use std::collections::TryReserveError;
use std::fmt;

use bytemuck::{Pod, Zeroable};
use color::{Sample, SampleDepth};

mod bins;
mod budget;
mod mask;
mod path;
#[doc(hidden)]
pub mod reference;
mod runs;
mod table;
#[cfg(test)]
pub(crate) mod test_support;

pub use bins::TileBins;
use budget::{Counts, index};
pub use mask::{MaskBits, MaskId};
pub use path::{AreaPath, PATH_COORD_LIMIT, PATH_SEGMENTS, PATH_TILE_SEGMENTS};
use path::{Backdrop, PathTile, bin_path};
pub use table::CoverageRows;

/// Side of a binning tile in pixels; equals the raster workgroup's side.
pub const PAGE_TILE: u32 = 16;

/// The most bytes the device inputs of a page may take
/// ([`PageCommands::bytes`]).
///
/// It is 2^27, the least `maxStorageBufferRange` a Vulkan device may
/// report: every input of a page within it binds on every device, and
/// every element index into an input fits `u32`.
pub const PAGE_COMMANDS_MAX_BYTES: usize = 1 << 27;

/// [`PageCmd::kind`] of a command whose coverage is a table.
const KIND_TABLE: u32 = 0;

/// [`PageCmd::kind`] of a command whose coverage is an outline's.
const KIND_PATH: u32 = 1;

/// [`PageCmd::kind`] of a command whose coverage is a mask's.
const KIND_MASK: u32 = 2;

/// One command as the kernel reads it; layout of `Cmd` in
/// `kernels/page_raster.slang`.  An outline's command holds its rule in
/// bit 0 of `flags` and its weight in bits 8–15, and its tiles in
/// `path_tiles[row_base..][..n_rows]`.  A mask's command holds the mask's
/// left column in `flags` (two's complement), its top row in `table_y0`,
/// its width in `n_rows` and the offset of its bytes in `row_base`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
#[repr(C)]
pub(crate) struct PageCmd {
    kind: u32,
    flags: u32,
    rg: u32,
    a_input: u32,
    table_y0: i32,
    n_rows: u32,
    row_base: u32,
    b: u32,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
}

/// An inclusive rectangle of device pixels: `x0..=x1` × `y0..=y1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PixelBounds {
    /// Leftmost column.
    pub x0: i32,
    /// Top row.
    pub y0: i32,
    /// Rightmost column.
    pub x1: i32,
    /// Bottom row.
    pub y1: i32,
}

impl PixelBounds {
    /// The rectangle's intersection with `other`, or `None` when they are
    /// disjoint.
    #[must_use]
    pub fn intersect(self, other: Self) -> Option<Self> {
        let r = Self {
            x0: self.x0.max(other.x0),
            y0: self.y0.max(other.y0),
            x1: self.x1.min(other.x1),
            y1: self.y1.min(other.y1),
        };
        (r.x0 <= r.x1 && r.y0 <= r.y1).then_some(r)
    }

    /// The number of [`PAGE_TILE`] tiles the rectangle touches, for
    /// `0 <= x0 <= x1` and `0 <= y0 <= y1`.
    pub(super) fn tiles(self) -> usize {
        let tile = |v: i32| v.unsigned_abs() as usize / PAGE_TILE as usize;
        (tile(self.x1) - tile(self.x0) + 1) * (tile(self.y1) - tile(self.y0) + 1)
    }
}

/// Why a command could not be added to a page, or a page binned.  The page
/// is unchanged then.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PageCommandsError {
    /// The page's device inputs would exceed [`PAGE_COMMANDS_MAX_BYTES`],
    /// or the page is too large for the raster's 32-bit coordinates.
    TooLarge,
    /// An outline exceeds the integer range of the device raster
    /// ([`PATH_COORD_LIMIT`], [`PATH_SEGMENTS`], [`PATH_TILE_SEGMENTS`]);
    /// its coverage can be pushed as a table instead.
    Unrepresentable,
    /// Host memory for the inputs could not be reserved.
    Alloc(TryReserveError),
}

impl fmt::Display for PageCommandsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLarge => write!(
                f,
                "the page's paint commands exceed {PAGE_COMMANDS_MAX_BYTES} bytes or its \
                 coordinates exceed 32 bits"
            ),
            Self::Unrepresentable => {
                write!(f, "an outline exceeds the device raster's integer range")
            }
            Self::Alloc(e) => write!(f, "the page's paint commands cannot be allocated: {e}"),
        }
    }
}

impl std::error::Error for PageCommandsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::TooLarge | Self::Unrepresentable => None,
            Self::Alloc(e) => Some(e),
        }
    }
}

impl From<TryReserveError> for PageCommandsError {
    fn from(e: TryReserveError) -> Self {
        Self::Alloc(e)
    }
}

/// A page's paint commands and the coverage tables they read.
///
/// Colours, alphas and the paper are samples of the page's
/// [`SampleDepth`]: at most 255 on an 8-bit page, 65535 on a 16-bit one.
#[derive(Debug, Clone)]
pub struct PageCommands {
    width: u32,
    height: u32,
    depth: SampleDepth,
    paper: [u16; 3],
    cmds: Vec<PageCmd>,
    /// Every command's spans, `[x0, x1, offset]`, `offset` into `bytes`.
    spans: Vec<[i32; 3]>,
    /// Every command's coverage bytes.
    bytes: Vec<u8>,
    row_starts: Vec<u32>,
    /// Every outline's segments, fixed point.
    segs: Vec<[i32; 4]>,
    /// Every outline tile's segments, indices into `segs`.
    seg_lists: Vec<u32>,
    /// Every outline tile's backdrop, [`PAGE_TILE`] covers each.
    backdrops: Vec<i32>,
    /// Every outline's tiles.
    path_tiles: Vec<PathTile>,
    /// Per tile, the items [`Self::bin`] lists in it: those from the last
    /// that hides the items before it on.  Empty until the first push.
    live: Vec<u32>,
    /// The sum of `live`, the length of [`TileBins::items`].
    tile_refs: usize,
}

/// The layout of [`PageCommands::table_parts`]: the index of the first
/// word of each array after the spans.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TableBases {
    /// Coverage bytes, four to a word.
    pub coverage: u32,
    /// Segments, four words each.
    pub segs: u32,
    /// Segment lists.
    pub seg_lists: u32,
    /// Backdrops, [`PAGE_TILE`] words each.
    pub backdrops: u32,
}

/// A sample type of the page raster's output: `u8` for an 8-bit page,
/// `u16` for a 16-bit one.
pub trait PageSample: Sample + sealed::Sealed {
    /// `dst` from `src`, the same samples little-endian, each of the bytes
    /// of a sample of its depth ([`SampleDepth::bytes`]).
    fn copy_from_le(dst: &mut [Self], src: &[u8]);
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
}

impl PageSample for u8 {
    fn copy_from_le(dst: &mut [Self], src: &[u8]) {
        dst.copy_from_slice(src);
    }
}

impl PageSample for u16 {
    fn copy_from_le(dst: &mut [Self], src: &[u8]) {
        for (d, s) in dst.iter_mut().zip(src.as_chunks::<2>().0) {
            *d = Self::from_le_bytes(*s);
        }
    }
}

/// `[r, g, b]` as the kernel's words `r | g << 16` and `b`.
pub(crate) fn pack_rgb([r, g, b]: [u16; 3]) -> [u32; 2] {
    [u32::from(r) | u32::from(g) << 16, u32::from(b)]
}

impl PageCommands {
    /// An empty page of `width × height` pixels of `depth`, every pixel
    /// `paper`.
    ///
    /// # Panics
    ///
    /// When a sample of `paper` exceeds the largest sample of `depth`.
    #[must_use]
    pub fn new(width: u32, height: u32, depth: SampleDepth, paper: [u16; 3]) -> Self {
        assert!(
            paper.iter().all(|&v| v <= depth.max_sample()),
            "PageCommands::new: paper {paper:?} exceeds {} bits",
            depth.bits()
        );
        Self {
            width,
            height,
            depth,
            paper,
            cmds: Vec::new(),
            spans: Vec::new(),
            bytes: Vec::new(),
            row_starts: Vec::new(),
            segs: Vec::new(),
            seg_lists: Vec::new(),
            backdrops: Vec::new(),
            path_tiles: Vec::new(),
            live: Vec::new(),
            tile_refs: 0,
        }
    }

    /// Page width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Page height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// The sample depth of the page's pixels.
    #[must_use]
    pub const fn depth(&self) -> SampleDepth {
        self.depth
    }

    /// The colour of every pixel before the first command.
    #[must_use]
    pub const fn paper(&self) -> [u16; 3] {
        self.paper
    }

    /// Number of commands.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.cmds.len()
    }

    /// Whether the page has no command.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.cmds.is_empty()
    }

    /// Number of tiles of the page's binning grid.
    pub(super) const fn tile_count(&self) -> usize {
        self.width.div_ceil(PAGE_TILE) as usize * self.height.div_ceil(PAGE_TILE) as usize
    }

    /// The page's pixels as bounds, or `None` for a page without pixels.
    pub(super) fn page_bounds(&self) -> Result<Option<PixelBounds>, PageCommandsError> {
        let extent = |v: u32| i32::try_from(v).map_err(|_| PageCommandsError::TooLarge);
        let (w, h) = (extent(self.width)?, extent(self.height)?);
        Ok((w > 0 && h > 0).then_some(PixelBounds {
            x0: 0,
            y0: 0,
            x1: w - 1,
            y1: h - 1,
        }))
    }

    /// Paint `rgb` at constant alpha `a_input` with the coverage of `path`
    /// as the shape, over the pixels of its window that lie on the page.
    /// Nothing is pushed when none does.
    ///
    /// # Errors
    ///
    /// As [`Self::push_coverage`]; [`PageCommandsError::Unrepresentable`]
    /// when the outline exceeds the device raster's integer range.  The
    /// page is unchanged then.
    ///
    /// # Panics
    ///
    /// When a sample of `rgb` or `a_input` exceeds the page's depth.
    pub fn push_area_path(
        &mut self,
        path: &AreaPath<'_>,
        rgb: [u16; 3],
        a_input: u16,
    ) -> Result<(), PageCommandsError> {
        let max = self.depth.max_sample();
        assert!(
            rgb.iter().all(|&v| v <= max) && a_input <= max,
            "push_area_path: colour {rgb:?} or alpha {a_input} exceeds {} bits",
            self.depth.bits()
        );
        let Some(page) = self.page_bounds()? else {
            return Ok(());
        };
        let Some(window) = page.intersect(path.window) else {
            return Ok(());
        };
        let binned = bin_path(
            &AreaPath { window, ..*path },
            page,
            self.width.div_ceil(PAGE_TILE),
            a_input == max,
        )?;
        if binned.tiles.is_empty() {
            return Ok(());
        }
        let hidden_refs = if self.live.is_empty() {
            0
        } else {
            binned
                .tiles
                .iter()
                .filter(|t| t.occludes)
                .map(|t| self.live[t.tile as usize] as usize)
                .sum()
        };
        let more = Counts {
            cmds: 1,
            tile_refs: binned.tiles.len(),
            hidden_refs,
            segs: path.segs.len(),
            seg_lists: binned.lists.len(),
            backdrops: binned.backdrops.len(),
            ..Counts::default()
        };
        if self.bytes_with(more) > PAGE_COMMANDS_MAX_BYTES {
            return Err(PageCommandsError::TooLarge);
        }
        let seg_base = index(self.segs.len())?;
        let list_base = index(self.seg_lists.len())?;
        let backdrop_base = index(self.backdrops.len() / PAGE_TILE as usize)?;
        let tile_base = index(self.path_tiles.len())?;
        let n_tiles = index(binned.tiles.len())?;
        self.cmds.try_reserve(1)?;
        self.segs.try_reserve(path.segs.len())?;
        self.seg_lists.try_reserve(binned.lists.len())?;
        self.backdrops.try_reserve(binned.backdrops.len())?;
        self.path_tiles.try_reserve(binned.tiles.len())?;
        self.reserve_live()?;
        // `bin_path` bounds every coordinate by `PATH_COORD_LIMIT`.
        self.segs.extend(
            path.segs
                .iter()
                .map(|s| s.map(|v| i32::try_from(v).unwrap_or_default())),
        );
        self.seg_lists
            .extend(binned.lists.iter().map(|&i| i + seg_base));
        self.backdrops.extend_from_slice(&binned.backdrops);
        self.path_tiles
            .extend(binned.tiles.iter().map(|t| PathTile {
                list: t.list + if t.list_len == 1 { seg_base } else { list_base },
                backdrop: match t.backdrop {
                    Backdrop::Stored(i) => Backdrop::Stored(i + backdrop_base),
                    other => other,
                },
                ..*t
            }));
        for t in &binned.tiles {
            let live = &mut self.live[t.tile as usize];
            *live = if t.occludes { 1 } else { *live + 1 };
        }
        self.tile_refs = self.tile_refs + binned.tiles.len() - hidden_refs;
        let [rg, b] = pack_rgb(rgb);
        self.cmds.push(PageCmd {
            kind: KIND_PATH,
            flags: u32::from(path.even_odd) | u32::from(path.weight) << 8,
            rg,
            b,
            a_input: u32::from(a_input),
            n_rows: n_tiles,
            row_base: tile_base,
            x0: window.x0,
            y0: window.y0,
            x1: window.x1,
            y1: window.y1,
            ..PageCmd::default()
        });
        Ok(())
    }

    /// The commands as the kernel reads them.
    pub(crate) fn cmds(&self) -> &[PageCmd] {
        &self.cmds
    }

    /// The table input as consecutive byte slices, words little-endian:
    /// the spans, three words each; the coverage bytes, four to a word,
    /// zero-padded to a whole word; the outlines' segments, four words
    /// each; the outline tiles' segment lists; and their backdrops.  With
    /// the index of the first word of each array after the spans.
    pub(crate) fn table_parts(&self) -> ([&[u8]; 6], TableBases) {
        const PAD: [u8; 3] = [0; 3];
        let pad = &PAD[..self.bytes.len().next_multiple_of(4) - self.bytes.len()];
        let parts: [&[u8]; 6] = [
            bytemuck::cast_slice(&self.spans),
            &self.bytes,
            pad,
            bytemuck::cast_slice(&self.segs),
            bytemuck::cast_slice(&self.seg_lists),
            bytemuck::cast_slice(&self.backdrops),
        ];
        let mut words = [0u32; 6];
        let mut at = 0usize;
        for (w, part) in words.iter_mut().zip(parts) {
            *w = u32::try_from(at / 4).expect("the push limit bounds the words");
            at += part.len();
        }
        (
            parts,
            TableBases {
                coverage: words[1],
                segs: words[3],
                seg_lists: words[4],
                backdrops: words[5],
            },
        )
    }

    /// Whether a command takes its coverage from an outline.
    #[must_use]
    pub const fn has_paths(&self) -> bool {
        !self.path_tiles.is_empty()
    }

    pub(crate) fn row_starts(&self) -> &[u32] {
        &self.row_starts
    }

    #[cfg(test)]
    pub(crate) fn spans(&self) -> &[[i32; 3]] {
        &self.spans
    }
}

#[cfg(test)]
mod tests;
