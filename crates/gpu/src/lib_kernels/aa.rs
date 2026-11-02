//! Scanline-AA fill kernel dispatch.

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::GpuCtx;
use crate::fill::AaFillRect;

/// Threads per block edge for the `aa_fill` launch (`BLOCK_W` / `BLOCK_H`
/// in `aa_fill.cu`).
const BLOCK_EDGE: u32 = 16;

impl GpuCtx {
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
