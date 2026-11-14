//! Per-row run-length coverage of a region.

use color::convert::div255;

use super::AreaRaster;

/// `Span::bytes` of a fully covered span.
const FULL: u32 = u32::MAX;

/// A maximal run of covered pixels on one row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Span {
    /// First pixel, inclusive.
    x0: i32,
    /// Last pixel, inclusive.
    x1: i32,
    /// [`FULL`] for coverage 255 throughout; otherwise the offset of the
    /// span's `x1 − x0 + 1` coverage bytes in [`CoverageTable::bytes`].
    bytes: u32,
}

/// The coverage of a region, row by row: on each row a sorted list of
/// disjoint spans, each fully covered or carrying one byte per pixel.
/// Pixels in no span are uncovered.
///
/// The representation is canonical: no stored byte is 0 or 255, a run of
/// coverage 255 is one fully covered span, and fully covered spans never
/// touch.  The table is immutable once built.
#[derive(Clone, Debug)]
pub struct CoverageTable {
    /// First row.
    y_min: i32,
    /// `row_start[i]..row_start[i + 1]` are the spans of row `y_min + i`.
    row_start: Vec<u32>,
    spans: Vec<Span>,
    bytes: Vec<u8>,
}

impl CoverageTable {
    /// The table of no pixel.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            y_min: 0,
            row_start: Vec::new(),
            spans: Vec::new(),
            bytes: Vec::new(),
        }
    }

    /// The coverage `raster` yields under the even-odd rule when `eo`,
    /// else non-zero; the raster's deposits are consumed.
    #[must_use]
    pub fn from_raster(raster: &mut AreaRaster, eo: bool) -> Self {
        let mut table = Self::empty();
        raster.rows(eo, |y, x, coverage| table.push_row(y, x, coverage));
        table
    }

    /// Whether the table covers no pixel.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Number of rows from the first to the last stored one.
    const fn n_rows(&self) -> usize {
        self.row_start.len().saturating_sub(1)
    }

    /// Append the coverage `coverage` of pixels `x..` on row `y`, below every
    /// row already stored.
    pub(crate) fn push_row(&mut self, y: i32, x: i32, coverage: &[u8]) {
        if self.row_start.is_empty() {
            self.y_min = y;
            self.row_start.push(0);
        }
        let row = usize::try_from(i64::from(y) - i64::from(self.y_min)).unwrap_or(usize::MAX);
        debug_assert!(row >= self.n_rows(), "rows are pushed in ascending order");
        let end = self.end_offset();
        while self.n_rows() < row {
            self.row_start.push(end);
        }
        let mut i = 0;
        while i < coverage.len() {
            let v = coverage[i];
            let start = i;
            if v == 0 {
                while i < coverage.len() && coverage[i] == 0 {
                    i += 1;
                }
                continue;
            }
            let full = v == u8::MAX;
            while i < coverage.len() && (coverage[i] == u8::MAX) == full && coverage[i] != 0 {
                i += 1;
            }
            let x0 = x.saturating_add(i32::try_from(start).unwrap_or(i32::MAX));
            let x1 = x.saturating_add(i32::try_from(i - 1).unwrap_or(i32::MAX));
            let bytes = if full {
                FULL
            } else {
                let at = u32::try_from(self.bytes.len()).expect("coverage table exceeds 4 GiB");
                self.bytes.extend_from_slice(&coverage[start..i]);
                at
            };
            self.spans.push(Span { x0, x1, bytes });
        }
        self.row_start.push(self.end_offset());
    }

    fn end_offset(&self) -> u32 {
        u32::try_from(self.spans.len()).expect("coverage table exceeds u32::MAX spans")
    }

    /// The first row the table stores; `row_starts` counts rows from it.
    #[must_use]
    pub const fn y_min(&self) -> i32 {
        self.y_min
    }

    /// Span offsets: the spans of row `y_min + i` are
    /// `spans[row_starts[i]..row_starts[i + 1]]`; empty for an empty table.
    #[must_use]
    pub fn row_starts(&self) -> &[u32] {
        &self.row_start
    }

    /// Every span as `[x0, x1, offset]`, in row order: pixels `x0..=x1`
    /// covered fully when `offset` is −1, else by the bytes
    /// `bytes()[offset..][..x1 − x0 + 1]`.
    pub fn span_triples(&self) -> impl Iterator<Item = [i32; 3]> + '_ {
        self.spans.iter().map(|s| {
            let offset = if s.bytes == FULL {
                -1
            } else {
                i32::try_from(s.bytes).unwrap_or(i32::MAX)
            };
            [s.x0, s.x1, offset]
        })
    }

    /// The partial spans' coverage bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The pixels the table covers, as `(x0, y0, x1, y1)` inclusive; `None`
    /// when it covers none.
    #[must_use]
    pub fn bounds(&self) -> Option<(i32, i32, i32, i32)> {
        let (y0, y1) = self.row_range()?;
        let x0 = self.spans.iter().map(|s| s.x0).min()?;
        let x1 = self.spans.iter().map(|s| s.x1).max()?;
        Some((x0, y0, x1, y1))
    }

    /// The spans of row `y`.
    fn row(&self, y: i32) -> &[Span] {
        let Ok(i) = usize::try_from(i64::from(y) - i64::from(self.y_min)) else {
            return &[];
        };
        if i >= self.n_rows() {
            return &[];
        }
        &self.spans[self.row_start[i] as usize..self.row_start[i + 1] as usize]
    }

    /// The coverage bytes of the partial span `s`.
    fn span_bytes(&self, s: &Span) -> &[u8] {
        let at = s.bytes as usize;
        let len = usize::try_from(i64::from(s.x1) - i64::from(s.x0) + 1).unwrap_or(0);
        &self.bytes[at..at + len]
    }

    /// Coverage (0–255) of pixel `(x, y)`.
    #[must_use]
    pub fn coverage(&self, x: i32, y: i32) -> u8 {
        let row = self.row(y);
        let i = row.partition_point(|s| s.x1 < x);
        match row.get(i) {
            Some(s) if s.x0 <= x => {
                if s.bytes == FULL {
                    u8::MAX
                } else {
                    let off = usize::try_from(i64::from(x) - i64::from(s.x0)).unwrap_or(0);
                    self.span_bytes(s)[off]
                }
            }
            _ => 0,
        }
    }

    /// Whether every pixel of `[x0, x1]` on row `y` is fully covered.
    #[must_use]
    pub fn is_full(&self, x0: i32, x1: i32, y: i32) -> bool {
        let row = self.row(y);
        let i = row.partition_point(|s| s.x1 < x0);
        row.get(i)
            .is_some_and(|s| s.bytes == FULL && s.x0 <= x0 && x1 <= s.x1)
    }

    /// Scale `out[i]`, the coverage of pixel `(x + i, y)`, by this table's
    /// coverage of that pixel: `div255(out · coverage)`, 0 outside it.
    pub fn mul_row(&self, y: i32, x: i32, out: &mut [u8]) {
        let end = i64::from(x) + i64::try_from(out.len()).unwrap_or(i64::MAX);
        let mut cursor = 0usize;
        let row = self.row(y);
        let first = row.partition_point(|s| s.x1 < x);
        for s in &row[first..] {
            if i64::from(s.x0) >= end {
                break;
            }
            let lo = usize::try_from((i64::from(s.x0) - i64::from(x)).max(0)).unwrap_or(0);
            let hi = usize::try_from((i64::from(s.x1) + 1).min(end) - i64::from(x)).unwrap_or(0);
            out[cursor..lo].fill(0);
            if s.bytes != FULL {
                let skip = usize::try_from((i64::from(x) - i64::from(s.x0)).max(0)).unwrap_or(0);
                for (o, &c) in out[lo..hi].iter_mut().zip(&self.span_bytes(s)[skip..]) {
                    *o = div255(u32::from(*o) * u32::from(c));
                }
            }
            cursor = hi;
        }
        out[cursor..].fill(0);
    }

    /// This table moved by `dx` columns and `dy` rows; coordinates saturate
    /// at the `i32` range.
    #[must_use]
    pub fn translated(&self, dx: i32, dy: i32) -> Self {
        Self {
            y_min: self.y_min.saturating_add(dy),
            row_start: self.row_start.clone(),
            spans: self
                .spans
                .iter()
                .map(|s| Span {
                    x0: s.x0.saturating_add(dx),
                    x1: s.x1.saturating_add(dx),
                    bytes: s.bytes,
                })
                .collect(),
            bytes: self.bytes.clone(),
        }
    }

    /// Rows `[y_min, y_max]` the table stores, or `None` when it is empty.
    fn row_range(&self) -> Option<(i32, i32)> {
        let n = i32::try_from(self.n_rows()).ok()?;
        (n > 0).then(|| (self.y_min, self.y_min.saturating_add(n - 1)))
    }

    /// Columns `[x_min, x_max]` of row `y`'s spans, or `None` for an empty row.
    fn row_extent(&self, y: i32) -> Option<(i32, i32)> {
        let row = self.row(y);
        Some((row.first()?.x0, row.last()?.x1))
    }

    /// The pointwise product of the tables' coverages, `div255` after each
    /// factor in iteration order; no table gives an empty one.
    #[must_use]
    pub fn product<'a>(tables: impl IntoIterator<Item = &'a Self>) -> Self {
        let tables: Vec<&Self> = tables.into_iter().collect();
        let (Some(y_min), Some(y_max)) = (
            tables
                .iter()
                .map(|t| t.row_range().map(|r| r.0))
                .max()
                .flatten(),
            tables
                .iter()
                .map(|t| t.row_range().map(|r| r.1))
                .min()
                .flatten(),
        ) else {
            return Self::empty();
        };
        if tables.iter().any(|t| t.is_empty()) {
            return Self::empty();
        }
        let mut out = Self::empty();
        let mut line = Vec::new();
        for y in y_min..=y_max {
            let extent = tables.iter().try_fold((i32::MIN, i32::MAX), |(lo, hi), t| {
                let (a, b) = t.row_extent(y)?;
                Some((lo.max(a), hi.min(b)))
            });
            let Some((lo, hi)) = extent.filter(|&(lo, hi)| lo <= hi) else {
                continue;
            };
            line.clear();
            line.resize(
                usize::try_from(i64::from(hi) - i64::from(lo) + 1).unwrap_or(0),
                u8::MAX,
            );
            for t in &tables {
                t.mul_row(y, lo, &mut line);
            }
            out.push_row(y, lo, &line);
        }
        out
    }
}
