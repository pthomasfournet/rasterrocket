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

use bytemuck::{Pod, Zeroable};

use crate::fill::{AA_SIZE, AaScanTable, ScanIntersect};

/// Side of a binning tile in pixels; equals the raster workgroup's side.
pub const PAGE_TILE: u32 = 16;

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
}

/// `[r, g, b]` as the kernel's packed `r | g << 8 | b << 16`.
pub(crate) fn pack_rgb([r, g, b]: [u8; 3]) -> u32 {
    u32::from(r) | u32::from(g) << 8 | u32::from(b) << 16
}

/// `v` as `u32`, for counts bounded by the host memory a page holds.
fn index(v: usize) -> u32 {
    u32::try_from(v).expect("page command list exceeds u32 indices")
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

    /// Bytes of the commands and their tables.
    #[must_use]
    pub const fn bytes(&self) -> usize {
        size_of::<PageCmd>() * self.cmds.len()
            + size_of::<ScanIntersect>() * self.intersects.len()
            + size_of::<u32>() * self.row_starts.len()
    }

    /// The page's pixels, as an inclusive rectangle; `None` for an empty
    /// page.
    fn page_bounds(&self) -> Option<PixelBounds> {
        let w = i32::try_from(self.width).ok()?;
        let h = i32::try_from(self.height).ok()?;
        (w > 0 && h > 0).then_some(PixelBounds {
            x0: 0,
            y0: 0,
            x1: w - 1,
            y1: h - 1,
        })
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
    ) {
        table.assert_well_formed();
        let Some(bounds) = self.page_bounds().and_then(|p| p.intersect(bounds)) else {
            return;
        };
        let base = index(self.intersects.len());
        let row_base = index(self.row_starts.len());
        self.intersects.extend_from_slice(table.intersects);
        self.row_starts
            .extend(table.row_starts.iter().map(|&s| s + base));
        self.cmds.push(PageCmd {
            eo: u32::from(even_odd),
            rgb: pack_rgb(rgb),
            a_input: u32::from(a_input),
            table_y0: table.y_min,
            n_rows: index(table.n_rows()),
            row_base,
            x0: bounds.x0,
            y0: bounds.y0,
            x1: bounds.x1,
            y1: bounds.y1,
            ..PageCmd::default()
        });
    }

    /// Paint `rgb` opaquely over the pixel runs `runs`, each `(y, x0, x1)`
    /// with `x0 ≤ x1` inclusive, clipped to the page: every pixel of a run
    /// takes `rgb`, however many runs cover it.  The runs are sorted in
    /// place.
    pub fn push_opaque_runs(&mut self, runs: &mut [(i32, i32, i32)], rgb: [u8; 3]) {
        runs.sort_unstable();
        let Some(&(y_first, ..)) = runs.first() else {
            return;
        };
        let y_last = runs[runs.len() - 1].0;
        let (mut x_min, mut x_max) = (i32::MAX, i32::MIN);
        let mut row_spans: Vec<ScanIntersect> = Vec::new();
        let mut intersects = Vec::new();
        let mut row_starts = vec![0u32];
        let mut rest = runs.iter().peekable();
        for y in y_first..=y_last {
            row_spans.clear();
            while let Some(&(_, x0, x1)) = rest.next_if(|r| r.0 == y) {
                x_min = x_min.min(x0);
                x_max = x_max.max(x1);
                match row_spans.last_mut() {
                    Some(last) if x0 <= last.x1 => last.x1 = last.x1.max(x1),
                    _ => row_spans.push(ScanIntersect { x0, x1, count: 0 }),
                }
            }
            // Each pixel span covers its AA_SIZE sub-rows over all of its
            // AA columns; a zero winding count keeps the spans disjoint.
            for _ in 0..AA_SIZE {
                intersects.extend(row_spans.iter().map(|s| ScanIntersect {
                    x0: s.x0 * AA_SIZE,
                    x1: s.x1 * AA_SIZE + AA_SIZE - 1,
                    count: 0,
                }));
                row_starts.push(index(intersects.len()));
            }
        }
        let table = AaScanTable {
            intersects: &intersects,
            row_starts: &row_starts,
            y_min: y_first * AA_SIZE,
        };
        let bounds = PixelBounds {
            x0: x_min,
            y0: y_first,
            x1: x_max,
            y1: y_last,
        };
        self.push_coverage(table, false, bounds, rgb, u8::MAX);
    }

    /// The commands' tile lists, built on the host: each command is listed,
    /// in paint order, in every tile its bounds touch.
    ///
    /// # Panics
    ///
    /// When the tile references outnumber `u32` indices.
    #[must_use]
    pub fn bin(&self) -> TileBins {
        let tiles_x = self.width.div_ceil(PAGE_TILE);
        let tiles_y = self.height.div_ceil(PAGE_TILE);
        let n_tiles = tiles_x as usize * tiles_y as usize;
        let tile = |v: i32| usize::try_from(v).unwrap_or(0) / PAGE_TILE as usize;
        let tiles_of = |c: &PageCmd| {
            let (tx0, tx1) = (tile(c.x0), tile(c.x1));
            (tile(c.y0)..=tile(c.y1))
                .flat_map(move |ty| (tx0..=tx1).map(move |tx| ty * tiles_x as usize + tx))
        };
        let mut offsets = vec![0u32; n_tiles + 1];
        for c in &self.cmds {
            for t in tiles_of(c) {
                offsets[t + 1] += 1;
            }
        }
        for t in 0..n_tiles {
            offsets[t + 1] += offsets[t];
        }
        let mut cursor = offsets.clone();
        let mut cmds = vec![0u32; offsets[n_tiles] as usize];
        for (i, c) in self.cmds.iter().enumerate() {
            for t in tiles_of(c) {
                cmds[cursor[t] as usize] = index(i);
                cursor[t] += 1;
            }
        }
        TileBins {
            tiles_x,
            tiles_y,
            offsets,
            cmds,
        }
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
        page.push_coverage(table, false, b(0, 0, 39, 19), [0; 3], 255);
        page.push_coverage(table, false, b(20, 2, 25, 3), [0; 3], 255);
        page.push_coverage(table, false, b(5, 17, 100, 100), [0; 3], 255);
        let bins = page.bin();
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
        page.push_coverage(table, false, bounds, [0; 3], 255);
        assert!(page.is_empty());
    }

    #[test]
    fn opaque_runs_cover_whole_pixels_and_merge_overlaps() {
        let mut page = PageCommands::new(32, 32, [255; 3]);
        let mut runs = vec![(3, 7, 9), (2, 1, 4), (2, 3, 6), (2, 10, 10)];
        page.push_opaque_runs(&mut runs, [1, 2, 3]);
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
}
