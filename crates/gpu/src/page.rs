//! A page as a list of paint commands for the device page raster.
//!
//! Each command paints one colour at a constant alpha over an opaque RGB
//! page, with per-pixel coverage taken from a scanline intersection table
//! in AA coordinates (the CPU scanner's, see [`AaScanTable`]) and limited
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

use crate::fill::{AA_SIZE, AaScanTable, ScanIntersect};

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
    eo: u32,
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
    /// or the page is too large for the raster's 32-bit AA coordinates.
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
                 AA coordinates exceed 32 bits"
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

/// A page's paint commands and the intersection tables they read.
#[derive(Debug, Clone)]
pub struct PageCommands {
    width: u32,
    height: u32,
    paper: [u8; 3],
    cmds: Vec<PageCmd>,
    intersects: Vec<ScanIntersect>,
    row_starts: Vec<u32>,
    /// Tile references the commands' bounds make, the length of
    /// [`TileBins::cmds`].
    tile_refs: usize,
}

/// `[r, g, b]` as the kernel's packed `r | g << 8 | b << 16`.
pub(crate) fn pack_rgb([r, g, b]: [u8; 3]) -> u32 {
    u32::from(r) | u32::from(g) << 8 | u32::from(b) << 16
}

/// [`AA_SIZE`] as a count.
const AA: usize = AA_SIZE.unsigned_abs() as usize;

/// `v` as a `u32` element index.
fn index(v: usize) -> Result<u32, PageCommandsError> {
    u32::try_from(v).map_err(|_| PageCommandsError::TooLarge)
}

/// The bytes of device inputs holding `cmds` commands, `intersects`
/// intersections, `row_starts` row offsets and `tile_words` binning words,
/// saturating.
const fn input_bytes(
    cmds: usize,
    intersects: usize,
    row_starts: usize,
    tile_words: usize,
) -> usize {
    size_of::<PageCmd>()
        .saturating_mul(cmds)
        .saturating_add(size_of::<ScanIntersect>().saturating_mul(intersects))
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
            intersects: Vec::new(),
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
        self.bytes_with(0, 0, 0, 0)
    }

    /// Bytes the page's device inputs would take with `cmds` more commands
    /// holding `intersects` more intersections, `row_starts` more row
    /// offsets and `tile_refs` more tile references.
    const fn bytes_with(
        &self,
        cmds: usize,
        intersects: usize,
        row_starts: usize,
        tile_refs: usize,
    ) -> usize {
        input_bytes(
            self.cmds.len().saturating_add(cmds),
            self.intersects.len().saturating_add(intersects),
            self.row_starts.len().saturating_add(row_starts),
            (self.tile_count() + 1)
                .saturating_add(self.tile_refs)
                .saturating_add(tile_refs),
        )
    }

    /// The page's pixels, as an inclusive rectangle; `None` for an empty
    /// page.
    ///
    /// # Errors
    ///
    /// [`PageCommandsError::TooLarge`] when an AA coordinate of the page
    /// (a pixel coordinate times [`AA_SIZE`]) does not fit `i32`.
    fn page_bounds(&self) -> Result<Option<PixelBounds>, PageCommandsError> {
        let aa_extent = |v: u32| {
            i32::try_from(v)
                .ok()
                .filter(|v| v.checked_mul(AA_SIZE).is_some())
                .ok_or(PageCommandsError::TooLarge)
        };
        let (w, h) = (aa_extent(self.width)?, aa_extent(self.height)?);
        Ok((w > 0 && h > 0).then_some(PixelBounds {
            x0: 0,
            y0: 0,
            x1: w - 1,
            y1: h - 1,
        }))
    }

    /// Paint `rgb` at constant alpha `a_input` with the coverage `table`
    /// gives under the even-odd (`even_odd`) or non-zero rule, over the
    /// pixels of `bounds` that lie on the page.  Nothing is pushed when
    /// none does.
    ///
    /// A pixel's coverage is its count of covered sub-samples in the
    /// table's [`AA_SIZE`] rows and columns, mapped through the AA gamma
    /// table.
    ///
    /// # Errors
    ///
    /// [`PageCommandsError`] when the command would take the page past
    /// [`PAGE_COMMANDS_MAX_BYTES`] or its memory cannot be reserved; the
    /// page is unchanged then.
    ///
    /// # Panics
    ///
    /// When the table is malformed ([`AaScanTable::assert_well_formed`]).
    pub fn push_coverage(
        &mut self,
        table: AaScanTable<'_>,
        even_odd: bool,
        bounds: PixelBounds,
        rgb: [u8; 3],
        a_input: u8,
    ) -> Result<(), PageCommandsError> {
        table.assert_well_formed();
        let Some(bounds) = self.page_bounds()?.and_then(|p| p.intersect(bounds)) else {
            return Ok(());
        };
        let tile_refs = bounds.tiles();
        let (n_ix, n_rows) = (table.intersects.len(), table.row_starts.len());
        if self.bytes_with(1, n_ix, n_rows, tile_refs) > PAGE_COMMANDS_MAX_BYTES {
            return Err(PageCommandsError::TooLarge);
        }
        let base = index(self.intersects.len())?;
        // Every row start is at most the table's length (`assert_well_formed`),
        // so it is at most this total once offset by `base`.
        index(self.intersects.len() + n_ix).map(drop)?;
        let row_base = index(self.row_starts.len())?;
        let n_table_rows = index(table.n_rows())?;
        self.cmds.try_reserve(1)?;
        self.intersects.try_reserve(n_ix)?;
        self.row_starts.try_reserve(n_rows)?;
        self.intersects.extend_from_slice(table.intersects);
        self.row_starts
            .extend(table.row_starts.iter().map(|&s| s + base));
        self.tile_refs += tile_refs;
        self.cmds.push(PageCmd {
            eo: u32::from(even_odd),
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
        let aa_rows = (y_last - y_first + 1).unsigned_abs() as usize * AA;
        let n_ix = runs.len() * AA;
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
        if self.bytes_with(1, n_ix, aa_rows + 1, bounds.tiles()) > PAGE_COMMANDS_MAX_BYTES {
            return Err(PageCommandsError::TooLarge);
        }
        let mut intersects = Vec::new();
        intersects.try_reserve_exact(n_ix)?;
        let mut row_starts = Vec::new();
        row_starts.try_reserve_exact(aa_rows + 1)?;
        row_starts.push(0u32);
        let mut rest = runs.iter().peekable();
        for y in y_first..=y_last {
            let row_start = intersects.len();
            while let Some(&(_, x0, x1)) = rest.next_if(|r| r.0 == y) {
                // Each pixel span covers all of its AA columns; a zero
                // winding count keeps the spans disjoint.  The page bounds
                // keep the AA coordinates within `i32`.
                intersects.push(ScanIntersect {
                    x0: x0 * AA_SIZE,
                    x1: x1 * AA_SIZE + AA_SIZE - 1,
                    count: 0,
                });
            }
            let row_end = intersects.len();
            row_starts.push(index(row_end)?);
            for _ in 1..AA_SIZE {
                intersects.extend_from_within(row_start..row_end);
                row_starts.push(index(intersects.len())?);
            }
        }
        let table = AaScanTable {
            intersects: &intersects,
            row_starts: &row_starts,
            y_min: y_first * AA_SIZE,
        };
        self.push_coverage(table, false, bounds, rgb, u8::MAX)
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

    /// Every command's intersections, concatenated.
    pub(crate) fn intersects(&self) -> &[ScanIntersect] {
        &self.intersects
    }

    /// Absolute row offsets into [`Self::intersects`].
    pub(crate) fn row_starts(&self) -> &[u32] {
        &self.row_starts
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

    fn rect_table(intersects: &[ScanIntersect], row_starts: &[u32]) -> AaScanTable<'static> {
        AaScanTable {
            intersects: Vec::leak(intersects.to_vec()),
            row_starts: Vec::leak(row_starts.to_vec()),
            y_min: 0,
        }
    }

    #[test]
    fn bins_list_each_command_in_its_tiles_in_paint_order() {
        let mut page = PageCommands::new(40, 20, [255; 3]);
        let table = rect_table(&[], &[0]);
        let b = |x0, y0, x1, y1| PixelBounds { x0, y0, x1, y1 };
        for bounds in [b(0, 0, 39, 19), b(20, 2, 25, 3), b(5, 17, 100, 100)] {
            page.push_coverage(table, false, bounds, [0; 3], 255)
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
        let table = rect_table(&[], &[0]);
        let bounds = PixelBounds {
            x0: 10,
            y0: 0,
            x1: 20,
            y1: 5,
        };
        page.push_coverage(table, false, bounds, [0; 3], 255)
            .expect("push");
        assert!(page.is_empty());
    }

    #[test]
    fn opaque_runs_cover_whole_pixels_and_merge_overlaps() {
        let mut page = PageCommands::new(32, 32, [255; 3]);
        let mut runs = vec![(3, 7, 9), (2, 1, 4), (2, 3, 6), (2, 10, 10)];
        page.push_opaque_runs(&mut runs, [1, 2, 3]).expect("push");
        assert_eq!(page.len(), 1);
        let c = page.cmds()[0];
        assert_eq!((c.x0, c.y0, c.x1, c.y1), (1, 2, 10, 3));
        assert_eq!((c.table_y0, c.n_rows, c.a_input), (8, 8, 255));
        let rows = page.row_starts();
        let row = |r: usize| &page.intersects()[rows[r] as usize..rows[r + 1] as usize];
        let span = |x0, x1| ScanIntersect { x0, x1, count: 0 };
        for r in 0..4 {
            assert_eq!(row(r), [span(4, 27), span(40, 43)]);
            assert_eq!(row(r + 4), [span(28, 39)]);
        }
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
        let rows = page.row_starts();
        let row = |r: usize| &page.intersects()[rows[r] as usize..rows[r + 1] as usize];
        let span = |x0, x1| ScanIntersect { x0, x1, count: 0 };
        assert_eq!(row(0), [span(0, 15), span(120, 127)]);
        assert_eq!(row(4), []);
        assert_eq!(row(8), [span(0, 127)]);
        assert_eq!(rows.len(), 3 * 4 + 1);

        let mut off_page = vec![(-1, 0, 5), (2, 32, 40)];
        page.push_opaque_runs(&mut off_page, [0; 3]).expect("push");
        assert_eq!(page.len(), 1);
    }

    #[test]
    fn a_push_past_the_byte_limit_fails_and_leaves_the_page_unchanged() {
        let n = PAGE_COMMANDS_MAX_BYTES / size_of::<ScanIntersect>();
        let intersects = vec![ScanIntersect::default(); n];
        let row_starts = [0, u32::try_from(n).expect("small")];
        let table = AaScanTable {
            intersects: &intersects,
            row_starts: &row_starts,
            y_min: 0,
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
            page.push_coverage(table, false, bounds, [0; 3], 255),
            Err(PageCommandsError::TooLarge)
        );
        assert!(page.is_empty());
        assert_eq!(page.bytes(), before);
        let fits = AaScanTable {
            intersects: &intersects[..n / 2],
            row_starts: &[0, u32::try_from(n / 2).expect("small")],
            y_min: 0,
        };
        page.push_coverage(fits, false, bounds, [0; 3], 255)
            .expect("half the limit fits");
        assert!(page.bytes() <= PAGE_COMMANDS_MAX_BYTES);
    }

    #[test]
    fn a_page_beyond_32_bit_aa_coordinates_takes_no_command() {
        let mut page = PageCommands::new(u32::MAX / 4, 1, [255; 3]);
        assert_eq!(
            page.push_opaque_runs(&mut [(0, 0, 0)], [0; 3]),
            Err(PageCommandsError::TooLarge)
        );
        assert!(page.is_empty());
    }
}
