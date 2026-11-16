//! Coverage tables: commands whose per-pixel coverage the CPU rasteriser
//! computed, row by row.

use super::{
    Counts, KIND_TABLE, PAGE_COMMANDS_MAX_BYTES, PageCmd, PageCommands, PageCommandsError,
    PixelBounds, index, pack_rgb,
};

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

impl PageCommands {
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
        self.reserve_live()?;
        self.spans.extend(
            table.spans.iter().map(|&[x0, x1, offset]| {
                [x0, x1, if offset < 0 { -1 } else { offset + byte_base }]
            }),
        );
        self.bytes.extend_from_slice(table.bytes);
        self.row_starts
            .extend(table.row_starts.iter().map(|&s| s + span_base));
        self.count_rect(bounds);
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
}
