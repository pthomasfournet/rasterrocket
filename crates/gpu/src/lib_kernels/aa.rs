//! Scanline-AA fill kernel dispatch.

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::GpuCtx;
use crate::fill::{AaFillRect, AaScanTable};

/// Threads per block edge for the `aa_fill` launch (`BLOCK_W` / `BLOCK_H`
/// in `aa_fill.cu`).
const BLOCK_EDGE: u32 = 16;

impl GpuCtx {
    /// Compute per-pixel AA sub-sample counts for `rect` from a scanline
    /// intersection table.
    ///
    /// The output is `rect.width × rect.height` bytes, row-major, each the
    /// count `0..=AA_SIZE²` of the pixel's covered sub-samples — the same
    /// values [`crate::aa_fill_cpu`] produces.  `aa_width` is the bitmap
    /// width × `AA_SIZE`.
    ///
    /// The caller decides whether the fill is worth a GPU round trip
    /// (see [`crate::GPU_AA_FILL_THRESHOLD`]); this always dispatches.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU data transfer or the kernel launch fails.
    ///
    /// # Panics
    ///
    /// Panics if `table` is malformed ([`AaScanTable::assert_well_formed`]),
    /// the table has more than `u32::MAX` rows, or `rect`'s pixel count
    /// exceeds `u32::MAX`.
    pub fn aa_fill(
        &self,
        table: &AaScanTable<'_>,
        rect: AaFillRect,
        aa_width: u32,
        eo: bool,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let mut counts = Vec::new();
        self.aa_fill_with(table, rect, aa_width, eo, |_, band| {
            counts.extend_from_slice(band);
        })?;
        Ok(counts)
    }

    /// [`Self::aa_fill`], handing the counts to `f` in bands of whole rows
    /// as they arrive on the host: `f(first_row, band)` in row order, where
    /// `first_row` is relative to `rect` and `band` holds the band's rows
    /// contiguously, `rect.width` counts each.  `band` may be modified.
    /// The copy of the next band runs while `f` processes the current one.
    ///
    /// # Errors
    ///
    /// As [`Self::aa_fill`]; bands already passed to `f` stay consumed.
    ///
    /// # Panics
    ///
    /// As [`Self::aa_fill`].
    pub fn aa_fill_with<F>(
        &self,
        table: &AaScanTable<'_>,
        rect: AaFillRect,
        aa_width: u32,
        eo: bool,
        mut f: F,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnMut(u32, &mut [u8]),
    {
        table.assert_well_formed();
        let n_pixels = rect.pixel_count().expect("width × height overflows usize");
        assert!(
            u32::try_from(n_pixels).is_ok(),
            "width × height = {n_pixels} pixels exceeds the kernel's 32-bit index range"
        );
        if n_pixels == 0 {
            return Ok(());
        }
        if table.intersects.is_empty() {
            // No intersections means no spans: every count is zero.
            f(0, &mut vec![0u8; n_pixels]);
            return Ok(());
        }
        let n_rows = u32::try_from(table.n_rows()).expect("row count exceeds u32::MAX");

        let stream = self.stream();
        let d_intersects = stream.clone_htod(bytemuck::cast_slice(table.intersects))?;
        let d_row_starts = stream.clone_htod(bytemuck::cast_slice::<u32, u8>(table.row_starts))?;
        // Device-side zero alloc — the kernel overwrites every in-bounds
        // pixel, so nothing is gained by shipping a zero Vec over PCIe.
        let d_coverage = stream.alloc_zeros::<u8>(n_pixels)?;

        self.launch_aa_fill_async(
            &d_intersects,
            &d_row_starts,
            n_rows,
            table.y_min,
            rect,
            aa_width,
            eo,
            &d_coverage,
        )?;

        crate::staging::download_banded(
            &stream,
            &d_coverage.as_view(),
            rect.width as usize,
            |row, band| f(u32::try_from(row).expect("row < rect.height"), band),
        )?;
        Ok(())
    }

    /// Async kernel launch for the scanline-AA fill kernel.
    ///
    /// Caller is responsible for stream ordering and any final D→H
    /// download. This function does **not** call `synchronize` and does
    /// **not** touch host memory.
    ///
    /// `d_intersects` holds `i32` triples, `d_row_starts` holds
    /// `n_rows + 1` `u32` offsets into it, and `d_coverage` must hold
    /// `rect.width × rect.height` bytes; the kernel overwrites every
    /// in-bounds pixel, so no pre-fill is required.
    ///
    /// # Errors
    ///
    /// Returns the underlying CUDA error if the launch fails.
    #[expect(
        clippy::too_many_arguments,
        reason = "all args mirror the PTX kernel signature; grouping would obscure the mapping"
    )]
    #[expect(
        unused_results,
        reason = "cudarc LaunchArgs::arg returns &mut Self for chaining; chain output is intentionally discarded"
    )]
    pub(crate) fn launch_aa_fill_async(
        &self,
        d_intersects: &CudaSlice<u8>,
        d_row_starts: &CudaSlice<u8>,
        n_rows: u32,
        table_y0: i32,
        rect: AaFillRect,
        aa_width: u32,
        eo: bool,
        d_coverage: &CudaSlice<u8>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = LaunchConfig {
            grid_dim: (
                rect.width.div_ceil(BLOCK_EDGE),
                rect.height.div_ceil(BLOCK_EDGE),
                1,
            ),
            block_dim: (BLOCK_EDGE, BLOCK_EDGE, 1),
            shared_mem_bytes: 0,
        };

        let eo_int: i32 = i32::from(eo);
        let stream = self.stream();
        let mut builder = stream.launch_builder(&self.kernels.aa_fill);
        builder
            .arg(d_intersects)
            .arg(d_row_starts)
            .arg(&n_rows)
            .arg(&table_y0)
            .arg(&rect.x)
            .arg(&rect.y)
            .arg(&rect.width)
            .arg(&rect.height)
            .arg(&aa_width)
            .arg(&eo_int)
            .arg(d_coverage);
        // SAFETY: kernel arguments match the PTX signature; bounds verified
        // by the caller.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }
}
