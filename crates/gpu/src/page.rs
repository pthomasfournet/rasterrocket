//! A page as a list of paint commands for the device page raster.
//!
//! Each command paints one colour at a constant alpha over an opaque RGB
//! page, with per-pixel coverage taken from a coverage table the CPU
//! rasteriser computed (see [`CoverageRows`]) and limited
//! to an inclusive pixel rectangle.  Commands paint in the order they were
//! pushed.  The table of every command is copied into one shared pair of
//! buffers, so a page is three flat arrays plus its size and paper colour.
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

/// One command as the kernel reads it; layout of `Cmd` in
/// `kernels/page_raster.slang`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
#[repr(C)]
pub(crate) struct PageCmd {
    kind: u32,
    reserved: u32,
    rgb: u32,
    a_input: u32,
    table_y0: i32,
    n_rows: u32,
    row_base: u32,
    pad: u32,
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
            Self::Alloc(e) => write!(f, "the page's paint commands cannot be allocated: {e}"),
        }
    }
}

impl std::error::Error for PageCommandsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::TooLarge => None,
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
/// paints `cmds[offsets[t]..offsets[t + 1]]`, command indices in paint
/// order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileBins {
    /// Tile columns.
    pub tiles_x: u32,
    /// Tile rows.
    pub tiles_y: u32,
    /// `tiles_x × tiles_y + 1` offsets into `cmds`.
    pub offsets: Vec<u32>,
    /// Command indices, grouped by tile.
    pub cmds: Vec<u32>,
}

/// A page's paint commands and the coverage tables they read.
#[derive(Debug, Clone)]
pub struct PageCommands {
    width: u32,
    height: u32,
    paper: [u8; 3],
    cmds: Vec<PageCmd>,
    /// Every command's spans, `[x0, x1, offset]`, `offset` into `bytes`.
    spans: Vec<[i32; 3]>,
    /// Every command's coverage bytes.
    bytes: Vec<u8>,
    row_starts: Vec<u32>,
    /// Tile references the commands' bounds make, the length of
    /// [`TileBins::cmds`].
    tile_refs: usize,
}

/// `[r, g, b]` as the kernel's packed `r | g << 8 | b << 16`.
pub(crate) fn pack_rgb([r, g, b]: [u8; 3]) -> u32 {
    u32::from(r) | u32::from(g) << 8 | u32::from(b) << 16
}

/// `v` as a `u32` element index.
fn index(v: usize) -> Result<u32, PageCommandsError> {
    u32::try_from(v).map_err(|_| PageCommandsError::TooLarge)
}

/// The bytes of device inputs holding `cmds` commands, `spans` spans,
/// `coverage` coverage bytes, `row_starts` row offsets and `tile_words`
/// binning words, saturating.  The coverage bytes are packed four to a
/// word after the spans.
const fn input_bytes(
    cmds: usize,
    spans: usize,
    coverage: usize,
    row_starts: usize,
    tile_words: usize,
) -> usize {
    size_of::<PageCmd>()
        .saturating_mul(cmds)
        .saturating_add(size_of::<[i32; 3]>().saturating_mul(spans))
        .saturating_add(coverage.saturating_add(3) / 4 * 4)
        .saturating_add(size_of::<u32>().saturating_mul(row_starts.saturating_add(tile_words)))
}

impl PageCommands {
    /// An empty page of `width × height` pixels, every pixel `paper`.
    #[must_use]
    pub const fn new(width: u32, height: u32, paper: [u8; 3]) -> Self {
        Self {
            width,
            height,
            paper,
            cmds: Vec::new(),
            spans: Vec::new(),
            bytes: Vec::new(),
            row_starts: Vec::new(),
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

    /// The colour of every pixel before the first command.
    #[must_use]
    pub const fn paper(&self) -> [u8; 3] {
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
        self.bytes_with(0, 0, 0, 0, 0)
    }

    /// Bytes the page's device inputs would take with `cmds` more commands
    /// holding `spans` more spans, `coverage` more coverage bytes,
    /// `row_starts` more row offsets and `tile_refs` more tile references.
    const fn bytes_with(
        &self,
        cmds: usize,
        spans: usize,
        coverage: usize,
        row_starts: usize,
        tile_refs: usize,
    ) -> usize {
        input_bytes(
            self.cmds.len().saturating_add(cmds),
            self.spans.len().saturating_add(spans),
            self.bytes.len().saturating_add(coverage),
            self.row_starts.len().saturating_add(row_starts),
            (self.tile_count() + 1)
                .saturating_add(self.tile_refs)
                .saturating_add(tile_refs),
        )
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
    /// When the table is malformed ([`CoverageRows::is_well_formed`]).
    pub fn push_coverage(
        &mut self,
        table: CoverageRows<'_>,
        bounds: PixelBounds,
        rgb: [u8; 3],
        a_input: u8,
    ) -> Result<(), PageCommandsError> {
        assert!(
            table.is_well_formed(),
            "push_coverage: malformed coverage table"
        );
        let Some(bounds) = self.page_bounds()?.and_then(|p| p.intersect(bounds)) else {
            return Ok(());
        };
        let tile_refs = bounds.tiles();
        let (n_spans, n_bytes, n_rows) =
            (table.spans.len(), table.bytes.len(), table.row_starts.len());
        if self.bytes_with(1, n_spans, n_bytes, n_rows, tile_refs) > PAGE_COMMANDS_MAX_BYTES {
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
        self.cmds.push(PageCmd {
            rgb: pack_rgb(rgb),
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

    /// Paint `rgb` opaquely over the pixel runs `runs`, each `(y, x0, x1)`
    /// with `x0 ≤ x1` inclusive: every pixel of a run that lies on the page
    /// takes `rgb`, however many runs cover it.  The runs are clipped to
    /// the page, sorted and merged in place, so the slice's order and
    /// contents are unspecified afterwards.
    ///
    /// # Errors
    ///
    /// As [`Self::push_coverage`].
    pub fn push_opaque_runs(
        &mut self,
        runs: &mut [(i32, i32, i32)],
        rgb: [u8; 3],
    ) -> Result<(), PageCommandsError> {
        let Some(page) = self.page_bounds()? else {
            return Ok(());
        };
        let runs = merge_runs(runs, page);
        let (Some(&(y_first, ..)), Some(&(y_last, ..))) = (runs.first(), runs.last()) else {
            return Ok(());
        };
        let n_rows = (y_last - y_first + 1).unsigned_abs() as usize;
        let bounds = runs.iter().fold(
            PixelBounds {
                x0: i32::MAX,
                y0: y_first,
                x1: i32::MIN,
                y1: y_last,
            },
            |b, &(_, x0, x1)| PixelBounds {
                x0: b.x0.min(x0),
                x1: b.x1.max(x1),
                ..b
            },
        );
        if self.bytes_with(1, runs.len(), 0, n_rows + 1, bounds.tiles()) > PAGE_COMMANDS_MAX_BYTES {
            return Err(PageCommandsError::TooLarge);
        }
        let mut spans = Vec::new();
        spans.try_reserve_exact(runs.len())?;
        let mut row_starts = Vec::new();
        row_starts.try_reserve_exact(n_rows + 1)?;
        row_starts.push(0u32);
        let mut rest = runs.iter().peekable();
        for y in y_first..=y_last {
            while let Some(&(_, x0, x1)) = rest.next_if(|r| r.0 == y) {
                spans.push([x0, x1, -1]);
            }
            row_starts.push(index(spans.len())?);
        }
        let table = CoverageRows {
            y_min: y_first,
            row_starts: &row_starts,
            spans: &spans,
            bytes: &[],
        };
        self.push_coverage(table, bounds, rgb, u8::MAX)
    }

    /// The commands' tile lists, built on the host: each command is listed,
    /// in paint order, in every tile its bounds touch.
    ///
    /// # Errors
    ///
    /// [`PageCommandsError`] when the lists' memory cannot be reserved, or
    /// they outnumber `u32` indices (which the push limit precludes).
    pub fn bin(&self) -> Result<TileBins, PageCommandsError> {
        let tiles_x = self.width.div_ceil(PAGE_TILE);
        let tiles_y = self.height.div_ceil(PAGE_TILE);
        let n_tiles = self.tile_count();
        let tile = |v: i32| usize::try_from(v).unwrap_or(0) / PAGE_TILE as usize;
        let tiles_of = |c: &PageCmd| {
            let (tx0, tx1) = (tile(c.x0), tile(c.x1));
            (tile(c.y0)..=tile(c.y1))
                .flat_map(move |ty| (tx0..=tx1).map(move |tx| ty * tiles_x as usize + tx))
        };
        index(self.tile_refs).map(drop)?;
        let mut offsets = Vec::new();
        offsets.try_reserve_exact(n_tiles + 1)?;
        offsets.resize(n_tiles + 1, 0u32);
        for c in &self.cmds {
            for t in tiles_of(c) {
                offsets[t + 1] += 1;
            }
        }
        // Every partial sum is at most `tile_refs`, which fits `u32`.
        for t in 0..n_tiles {
            offsets[t + 1] += offsets[t];
        }
        let mut cursor = offsets.clone();
        let mut cmds = Vec::new();
        cmds.try_reserve_exact(self.tile_refs)?;
        cmds.resize(self.tile_refs, 0u32);
        for (i, c) in self.cmds.iter().enumerate() {
            let i = index(i)?;
            for t in tiles_of(c) {
                cmds[cursor[t] as usize] = i;
                cursor[t] += 1;
            }
        }
        Ok(TileBins {
            tiles_x,
            tiles_y,
            offsets,
            cmds,
        })
    }

    /// The commands as the kernel reads them.
    pub(crate) fn cmds(&self) -> &[PageCmd] {
        &self.cmds
    }

    /// The spans as words, three each, then the coverage bytes packed four
    /// to a word, little-endian; and the index of the first coverage word.
    pub(crate) fn table_words(&self) -> (Vec<u32>, u32) {
        let mut words: Vec<u32> = self
            .spans
            .iter()
            .flat_map(|s| s.map(i32::cast_unsigned))
            .collect();
        let base = u32::try_from(words.len()).expect("the push limit bounds the spans");
        words.extend(self.bytes.chunks(4).map(|c| {
            let mut w = [0u8; 4];
            w[..c.len()].copy_from_slice(c);
            u32::from_le_bytes(w)
        }));
        (words, base)
    }

    pub(crate) fn row_starts(&self) -> &[u32] {
        &self.row_starts
    }

    #[cfg(test)]
    pub(crate) fn spans(&self) -> &[[i32; 3]] {
        &self.spans
    }
}

/// The runs of `runs` that meet `page`, clipped to it, sorted by row and
/// column, with each run that starts within the previous run of its row
/// merged into it; compacted to the front of `runs`.
fn merge_runs(runs: &mut [(i32, i32, i32)], page: PixelBounds) -> &[(i32, i32, i32)] {
    let mut kept = 0;
    for i in 0..runs.len() {
        let (y, x0, x1) = runs[i];
        let (x0, x1) = (x0.max(page.x0), x1.min(page.x1));
        if (page.y0..=page.y1).contains(&y) && x0 <= x1 {
            runs[kept] = (y, x0, x1);
            kept += 1;
        }
    }
    let runs = &mut runs[..kept];
    runs.sort_unstable();
    let mut merged: usize = 0;
    for i in 0..runs.len() {
        let run = runs[i];
        if let Some(m) = merged.checked_sub(1)
            && runs[m].0 == run.0
            && run.1 <= runs[m].2
        {
            runs[m].2 = runs[m].2.max(run.2);
        } else {
            runs[merged] = run;
            merged += 1;
        }
    }
    &runs[..merged]
}

#[cfg(test)]
mod tests {
    use super::*;

    const EMPTY: CoverageRows<'static> = CoverageRows {
        y_min: 0,
        row_starts: &[0],
        spans: &[],
        bytes: &[],
    };

    #[test]
    fn bins_list_each_command_in_its_tiles_in_paint_order() {
        let mut page = PageCommands::new(40, 20, [255; 3]);
        let b = |x0, y0, x1, y1| PixelBounds { x0, y0, x1, y1 };
        for bounds in [b(0, 0, 39, 19), b(20, 2, 25, 3), b(5, 17, 100, 100)] {
            page.push_coverage(EMPTY, bounds, [0; 3], 255)
                .expect("push");
        }
        let bins = page.bin().expect("bin");
        assert_eq!(bins.cmds.len(), 6 + 1 + 3);
        assert_eq!((bins.tiles_x, bins.tiles_y), (3, 2));
        let tile = |t: usize| &bins.cmds[bins.offsets[t] as usize..bins.offsets[t + 1] as usize];
        assert_eq!(tile(0), [0]);
        assert_eq!(tile(1), [0, 1]);
        assert_eq!(tile(2), [0]);
        assert_eq!(tile(3), [0, 2]);
        assert_eq!(tile(4), [0, 2]);
        assert_eq!(tile(5), [0, 2]);
    }

    #[test]
    fn bounds_off_the_page_push_nothing() {
        let mut page = PageCommands::new(10, 10, [255; 3]);
        let bounds = PixelBounds {
            x0: 10,
            y0: 0,
            x1: 20,
            y1: 5,
        };
        page.push_coverage(EMPTY, bounds, [0; 3], 255)
            .expect("push");
        assert!(page.is_empty());
    }

    #[test]
    fn tables_are_rebased_onto_the_page() {
        let mut page = PageCommands::new(16, 16, [255; 3]);
        let table = CoverageRows {
            y_min: 3,
            row_starts: &[0, 2, 3],
            spans: &[[1, 2, 0], [4, 9, -1], [2, 2, 2]],
            bytes: &[10, 20, 30],
        };
        let bounds = PixelBounds {
            x0: 0,
            y0: 0,
            x1: 15,
            y1: 15,
        };
        page.push_coverage(table, bounds, [0; 3], 255)
            .expect("push");
        page.push_coverage(table, bounds, [0; 3], 255)
            .expect("push");
        assert_eq!(page.row_starts(), [0, 2, 3, 3, 5, 6]);
        assert_eq!(page.spans()[3..], [[1, 2, 3], [4, 9, -1], [2, 2, 5]]);
        let (words, base) = page.table_words();
        assert_eq!(base, 18);
        assert_eq!(
            words[18..],
            [
                u32::from_le_bytes([10, 20, 30, 10]),
                u32::from_le_bytes([20, 30, 0, 0])
            ]
        );
    }

    #[test]
    #[should_panic(expected = "malformed")]
    fn a_span_past_its_bytes_is_refused() {
        let mut page = PageCommands::new(16, 16, [255; 3]);
        let table = CoverageRows {
            y_min: 0,
            row_starts: &[0, 1],
            spans: &[[0, 3, 1]],
            bytes: &[1, 2, 3, 4],
        };
        let bounds = PixelBounds {
            x0: 0,
            y0: 0,
            x1: 3,
            y1: 0,
        };
        let _ = page.push_coverage(table, bounds, [0; 3], 255);
    }

    #[test]
    fn opaque_runs_cover_whole_pixels_and_merge_overlaps() {
        let mut page = PageCommands::new(32, 32, [255; 3]);
        let mut runs = vec![(3, 7, 9), (2, 1, 4), (2, 3, 6), (2, 10, 10)];
        page.push_opaque_runs(&mut runs, [1, 2, 3]).expect("push");
        assert_eq!(page.len(), 1);
        let c = page.cmds()[0];
        assert_eq!((c.x0, c.y0, c.x1, c.y1), (1, 2, 10, 3));
        assert_eq!((c.table_y0, c.n_rows, c.a_input), (2, 2, 255));
        assert_eq!(page.row_starts(), [0, 2, 3]);
        assert_eq!(page.spans(), [[1, 6, -1], [10, 10, -1], [7, 9, -1]]);
    }

    #[test]
    fn opaque_runs_are_clipped_to_the_page() {
        let mut page = PageCommands::new(32, 8, [255; 3]);
        let mut runs = vec![
            (2, -1_000_000_000, 3),
            (2, 30, 1_000_000_000),
            (-5, 0, 4),
            (9, 0, 4),
            (3, 40, 50),
            (4, i32::MIN, i32::MAX),
        ];
        page.push_opaque_runs(&mut runs, [0; 3]).expect("push");
        let c = page.cmds()[0];
        assert_eq!((c.x0, c.y0, c.x1, c.y1), (0, 2, 31, 4));
        assert_eq!(page.row_starts(), [0, 2, 2, 3]);
        assert_eq!(page.spans(), [[0, 3, -1], [30, 31, -1], [0, 31, -1]]);

        let mut off_page = vec![(-1, 0, 5), (2, 32, 40)];
        page.push_opaque_runs(&mut off_page, [0; 3]).expect("push");
        assert_eq!(page.len(), 1);
    }

    #[test]
    fn a_push_past_the_byte_limit_fails_and_leaves_the_page_unchanged() {
        let n = PAGE_COMMANDS_MAX_BYTES / size_of::<[i32; 3]>();
        let spans = vec![[0, 0, -1]; n];
        let row_starts = [0, u32::try_from(n).expect("small")];
        let table = CoverageRows {
            y_min: 0,
            row_starts: &row_starts,
            spans: &spans,
            bytes: &[],
        };
        let bounds = PixelBounds {
            x0: 0,
            y0: 0,
            x1: 0,
            y1: 0,
        };
        let mut page = PageCommands::new(16, 16, [255; 3]);
        let before = page.bytes();
        assert_eq!(
            page.push_coverage(table, bounds, [0; 3], 255),
            Err(PageCommandsError::TooLarge)
        );
        assert!(page.is_empty());
        assert_eq!(page.bytes(), before);
        let fits = CoverageRows {
            spans: &spans[..n / 2],
            row_starts: &[0, u32::try_from(n / 2).expect("small")],
            ..table
        };
        page.push_coverage(fits, bounds, [0; 3], 255)
            .expect("half the limit fits");
        assert!(page.bytes() <= PAGE_COMMANDS_MAX_BYTES);
    }
}
