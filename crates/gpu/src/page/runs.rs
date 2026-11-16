//! Opaque pixel runs as table commands.

use super::{
    Counts, CoverageRows, PAGE_COMMANDS_MAX_BYTES, PageCommands, PageCommandsError, PixelBounds,
    index,
};

impl PageCommands {
    /// Paint `rgb` opaquely over the pixel runs `runs`, each `(y, x0, x1)`
    /// with `x0 ≤ x1` inclusive: every pixel of a run that lies on the page
    /// takes `rgb`, however many runs cover it.  The runs are clipped to
    /// the page, sorted and merged in place, so the slice's order and
    /// contents are unspecified afterwards.
    ///
    /// # Errors
    ///
    /// As [`Self::push_coverage`].
    ///
    /// # Panics
    ///
    /// When a sample of `rgb` exceeds the page's depth.
    pub fn push_opaque_runs(
        &mut self,
        runs: &mut [(i32, i32, i32)],
        rgb: [u16; 3],
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
        let more = Counts {
            cmds: 1,
            spans: runs.len(),
            row_starts: n_rows + 1,
            tile_refs: bounds.tiles(),
            ..Counts::default()
        };
        if self.bytes_with(more) > PAGE_COMMANDS_MAX_BYTES {
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
        self.push_coverage(table, bounds, rgb, self.depth.max_sample())
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
