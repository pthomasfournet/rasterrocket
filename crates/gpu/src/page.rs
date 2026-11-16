//! A page as a list of paint commands for the device page raster.
//!
//! Each command paints one colour at a constant alpha over an opaque RGB
//! page of 8 or 16 bits per sample, limited to an inclusive pixel
//! rectangle, with per-pixel coverage either taken from a coverage table
//! the CPU rasteriser computed (see [`CoverageRows`]) or computed on the
//! device from a closed outline by the CPU rasteriser's integer arithmetic
//! (see [`AreaPath`]).  Commands paint in the order they were pushed.  The
//! tables and outlines of every command are copied into shared flat
//! arrays.
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

mod path;
#[doc(hidden)]
pub mod reference;
mod runs;
#[cfg(test)]
pub(crate) mod test_support;

pub use path::{AreaPath, PATH_COORD_LIMIT, PATH_SEGMENTS, PATH_TILE_SEGMENTS};
use path::{NO_BACKDROP, PathTile, bin_path};

/// A region's coverage, row by row, as the page raster reads it.
///
/// The pixels `x0..=x1` of a span `[x0, x1, offset]` on row `y_min + i` are
/// covered fully when `offset` is −1, else by `bytes[offset + (x − x0)]`;
/// the spans of row `y_min + i` are `spans[row_starts[i]..row_starts[i +
/// 1]]`.  Pixels in no span are uncovered.
#[derive(Clone, Copy, Debug)]
pub struct CoverageRows<'a> {
    /// The row of `row_starts[0]`.
    pub y_min: i32,
    /// `n_rows + 1` offsets into `spans`.
    pub row_starts: &'a [u32],
    /// The spans, sorted by `x0` within each row.
    pub spans: &'a [[i32; 3]],
    /// The coverage bytes of the partial spans.
    pub bytes: &'a [u8],
}

impl CoverageRows<'_> {
    /// Rows the table holds.
    #[must_use]
    pub const fn n_rows(&self) -> usize {
        self.row_starts.len().saturating_sub(1)
    }

    /// Whether every offset lies within its array: `row_starts` ascends
    /// from 0 to at most `spans.len()`, and each partial span's bytes lie
    /// in `bytes`.
    #[must_use]
    pub fn is_well_formed(&self) -> bool {
        let rows_ok = self.row_starts.first().is_none_or(|&s| s == 0)
            && self.row_starts.windows(2).all(|w| w[0] <= w[1])
            && self
                .row_starts
                .last()
                .is_none_or(|&e| e as usize <= self.spans.len());
        let span_ok = |&[x0, x1, offset]: &[i32; 3]| {
            let len = usize::try_from(i64::from(x1) - i64::from(x0) + 1).ok();
            x0 <= x1
                && (offset == -1
                    || usize::try_from(offset)
                        .ok()
                        .zip(len)
                        .and_then(|(o, len)| o.checked_add(len))
                        .is_some_and(|end| end <= self.bytes.len()))
        };
        rows_ok && self.spans.iter().all(span_ok)
    }
}

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

/// One command as the kernel reads it; layout of `Cmd` in
/// `kernels/page_raster.slang`.  An outline's command holds its rule in
/// bit 0 of `flags` and its weight in bits 8–15, and its tiles in
/// `path_tiles[row_base..][..n_rows]`.
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
    fn tiles(self) -> usize {
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

/// The tile lists of a page: tile `t` (row-major over `tiles_x` columns)
/// paints `items[offsets[t]..offsets[t + 1]]` in order.
///
/// An item is `[command, list_start, list_len, backdrop]`: the command's
/// index, and for an outline's command the tile's segments
/// `segment_lists[list_start..][..list_len]` and backdrop
/// (`u32::MAX` when it is zero), as [`PageCommands::table_parts`] lays
/// them out; a table's command has `[command, 0, 0, u32::MAX]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileBins {
    /// Tile columns.
    pub tiles_x: u32,
    /// Tile rows.
    pub tiles_y: u32,
    /// `tiles_x × tiles_y + 1` offsets into `items`.
    pub offsets: Vec<u32>,
    /// The items, grouped by tile, in paint order within a tile.
    pub items: Vec<[u32; 4]>,
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
    /// Tile references the commands make, the length of
    /// [`TileBins::items`].
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

/// `v` as a `u32` element index.
fn index(v: usize) -> Result<u32, PageCommandsError> {
    u32::try_from(v).map_err(|_| PageCommandsError::TooLarge)
}

/// Counts of the elements of a page's device inputs.
#[derive(Clone, Copy, Default)]
struct Counts {
    cmds: usize,
    spans: usize,
    coverage: usize,
    row_starts: usize,
    /// Tile references, four words each.
    tile_refs: usize,
    segs: usize,
    seg_lists: usize,
    backdrops: usize,
}

impl Counts {
    /// The bytes of device inputs holding these elements and `tiles + 1`
    /// tile offsets, saturating.  The coverage bytes are packed four to a
    /// word.
    const fn bytes(self, tiles: usize) -> usize {
        let words = (self.coverage.saturating_add(3) / 4)
            .saturating_add(self.row_starts)
            .saturating_add(tiles.saturating_add(1))
            .saturating_add(self.tile_refs.saturating_mul(4))
            .saturating_add(self.segs.saturating_mul(4))
            .saturating_add(self.seg_lists)
            .saturating_add(self.backdrops);
        size_of::<PageCmd>()
            .saturating_mul(self.cmds)
            .saturating_add(size_of::<[i32; 3]>().saturating_mul(self.spans))
            .saturating_add(size_of::<u32>().saturating_mul(words))
    }
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
    const fn tile_count(&self) -> usize {
        self.width.div_ceil(PAGE_TILE) as usize * self.height.div_ceil(PAGE_TILE) as usize
    }

    /// Bytes of the page's device inputs: the commands, their tables and
    /// their tile lists.  At most [`PAGE_COMMANDS_MAX_BYTES`].
    #[must_use]
    pub const fn bytes(&self) -> usize {
        self.bytes_with(Counts {
            cmds: 0,
            spans: 0,
            coverage: 0,
            row_starts: 0,
            tile_refs: 0,
            segs: 0,
            seg_lists: 0,
            backdrops: 0,
        })
    }

    /// Bytes the page's device inputs would take with `more` more elements.
    const fn bytes_with(&self, more: Counts) -> usize {
        Counts {
            cmds: self.cmds.len().saturating_add(more.cmds),
            spans: self.spans.len().saturating_add(more.spans),
            coverage: self.bytes.len().saturating_add(more.coverage),
            row_starts: self.row_starts.len().saturating_add(more.row_starts),
            tile_refs: self.tile_refs.saturating_add(more.tile_refs),
            segs: self.segs.len().saturating_add(more.segs),
            seg_lists: self.seg_lists.len().saturating_add(more.seg_lists),
            backdrops: self.backdrops.len().saturating_add(more.backdrops),
        }
        .bytes(self.tile_count())
    }

    /// The page's pixels as bounds, or `None` for a page without pixels.
    fn page_bounds(&self) -> Result<Option<PixelBounds>, PageCommandsError> {
        let extent = |v: u32| i32::try_from(v).map_err(|_| PageCommandsError::TooLarge);
        let (w, h) = (extent(self.width)?, extent(self.height)?);
        Ok((w > 0 && h > 0).then_some(PixelBounds {
            x0: 0,
            y0: 0,
            x1: w - 1,
            y1: h - 1,
        }))
    }

    /// Paint `rgb` at constant alpha `a_input` with the coverage `table`
    /// gives, as the shape, over the pixels of `bounds` that lie on the
    /// page.  Nothing is pushed when none does.
    ///
    /// # Errors
    ///
    /// [`PageCommandsError`] when the command would take the page past
    /// [`PAGE_COMMANDS_MAX_BYTES`] or its memory cannot be reserved; the
    /// page is unchanged then.
    ///
    /// # Panics
    ///
    /// When the table is malformed ([`CoverageRows::is_well_formed`]), or a
    /// sample of `rgb` or `a_input` exceeds the page's depth.
    pub fn push_coverage(
        &mut self,
        table: CoverageRows<'_>,
        bounds: PixelBounds,
        rgb: [u16; 3],
        a_input: u16,
    ) -> Result<(), PageCommandsError> {
        assert!(
            table.is_well_formed(),
            "push_coverage: malformed coverage table"
        );
        let max = self.depth.max_sample();
        assert!(
            rgb.iter().all(|&v| v <= max) && a_input <= max,
            "push_coverage: colour {rgb:?} or alpha {a_input} exceeds {} bits",
            self.depth.bits()
        );
        let Some(bounds) = self.page_bounds()?.and_then(|p| p.intersect(bounds)) else {
            return Ok(());
        };
        let tile_refs = bounds.tiles();
        let (n_spans, n_bytes, n_rows) =
            (table.spans.len(), table.bytes.len(), table.row_starts.len());
        let more = Counts {
            cmds: 1,
            spans: n_spans,
            coverage: n_bytes,
            row_starts: n_rows,
            tile_refs,
            ..Counts::default()
        };
        if self.bytes_with(more) > PAGE_COMMANDS_MAX_BYTES {
            return Err(PageCommandsError::TooLarge);
        }
        let span_base = index(self.spans.len())?;
        index(self.spans.len() + n_spans).map(drop)?;
        i32::try_from(self.bytes.len() + n_bytes)
            .map(drop)
            .map_err(|_| PageCommandsError::TooLarge)?;
        let byte_base = i32::try_from(self.bytes.len()).map_err(|_| PageCommandsError::TooLarge)?;
        let row_base = index(self.row_starts.len())?;
        let n_table_rows = index(table.n_rows())?;
        self.cmds.try_reserve(1)?;
        self.spans.try_reserve(n_spans)?;
        self.bytes.try_reserve(n_bytes)?;
        self.row_starts.try_reserve(n_rows)?;
        self.spans.extend(
            table.spans.iter().map(|&[x0, x1, offset]| {
                [x0, x1, if offset < 0 { -1 } else { offset + byte_base }]
            }),
        );
        self.bytes.extend_from_slice(table.bytes);
        self.row_starts
            .extend(table.row_starts.iter().map(|&s| s + span_base));
        self.tile_refs += tile_refs;
        let [rg, b] = pack_rgb(rgb);
        self.cmds.push(PageCmd {
            kind: KIND_TABLE,
            rg,
            b,
            a_input: u32::from(a_input),
            table_y0: table.y_min,
            n_rows: n_table_rows,
            row_base,
            x0: bounds.x0,
            y0: bounds.y0,
            x1: bounds.x1,
            y1: bounds.y1,
            ..PageCmd::default()
        });
        Ok(())
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
        let more = Counts {
            cmds: 1,
            tile_refs: binned.tiles.len(),
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
                list_start: t.list_start + list_base,
                backdrop: if t.backdrop == NO_BACKDROP {
                    NO_BACKDROP
                } else {
                    t.backdrop + backdrop_base
                },
                ..*t
            }));
        self.tile_refs += binned.tiles.len();
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

    /// Call `f(tile, item, occludes)` for every tile each command paints,
    /// in paint order, with `item` as in [`TileBins::items`] and
    /// `occludes` whether it hides the items before it in the tile.
    fn for_each_item(&self, mut f: impl FnMut(usize, [u32; 4], bool)) {
        let tiles_x = self.width.div_ceil(PAGE_TILE) as usize;
        let tile = |v: i32| usize::try_from(v).unwrap_or(0) / PAGE_TILE as usize;
        for (c, i) in self.cmds.iter().zip(0u32..) {
            if c.kind == KIND_PATH {
                for t in &self.path_tiles[c.row_base as usize..][..c.n_rows as usize] {
                    let item = [i, t.list_start, t.list_len, t.backdrop];
                    f(t.tile as usize, item, t.occludes);
                }
                continue;
            }
            for ty in tile(c.y0)..=tile(c.y1) {
                for tx in tile(c.x0)..=tile(c.x1) {
                    f(ty * tiles_x + tx, [i, 0, 0, NO_BACKDROP], false);
                }
            }
        }
    }

    /// The commands' tile lists, built on the host: each command is listed,
    /// in paint order, in every tile its bounds touch, and an outline's
    /// command in every tile it may cover; the items a later item of the
    /// tile hides are left out: an outline at full alpha and weight that
    /// lists no segment in the tile and covers every pixel of it on the
    /// page fully paints them in its colour, whatever lay below.
    ///
    /// # Errors
    ///
    /// [`PageCommandsError`] when the lists' memory cannot be reserved, or
    /// they outnumber `u32` indices (which the push limit precludes).
    pub fn bin(&self) -> Result<TileBins, PageCommandsError> {
        let tiles_x = self.width.div_ceil(PAGE_TILE);
        let tiles_y = self.height.div_ceil(PAGE_TILE);
        let n_tiles = self.tile_count();
        index(self.tile_refs).map(drop)?;
        // Per tile: the items seen so far, and the index of the last one
        // that hides those before it.
        let mut seen = Vec::new();
        seen.try_reserve_exact(n_tiles)?;
        seen.resize(n_tiles, 0u32);
        let mut first = seen.clone();
        self.for_each_item(|t, _, occludes| {
            if occludes {
                first[t] = seen[t];
            }
            seen[t] += 1;
        });
        let mut offsets = Vec::new();
        offsets.try_reserve_exact(n_tiles + 1)?;
        offsets.resize(n_tiles + 1, 0u32);
        for t in 0..n_tiles {
            offsets[t + 1] = offsets[t] + seen[t] - first[t];
        }
        let n_items = offsets[n_tiles] as usize;
        let mut cursor = offsets.clone();
        let mut items = Vec::new();
        items.try_reserve_exact(n_items)?;
        items.resize(n_items, [0u32; 4]);
        seen.fill(0);
        self.for_each_item(|t, item, _| {
            if seen[t] >= first[t] {
                items[cursor[t] as usize] = item;
                cursor[t] += 1;
            }
            seen[t] += 1;
        });
        Ok(TileBins {
            tiles_x,
            tiles_y,
            offsets,
            items,
        })
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
