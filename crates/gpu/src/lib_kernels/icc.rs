//! ICC CMYK→RGB kernel dispatch.

use cudarc::driver::{CudaSlice, PushKernelArg};

use color::grid::CmykGrid;

use crate::backend::params::MAX_KERNEL_PIXELS;
use crate::{GPU_ICC_CLUT_THRESHOLD, GpuCtx, launch_cfg};

/// Check that `cmyk` holds whole 4-byte pixels.
fn check_cmyk_len(cmyk: &[u8]) -> Result<(), String> {
    if cmyk.len().is_multiple_of(4) {
        Ok(())
    } else {
        Err(format!(
            "CMYK buffer length {} is not a multiple of 4",
            cmyk.len()
        ))
    }
}

impl GpuCtx {
    /// Convert CMYK pixels to RGB, on the device when worth it.
    ///
    /// `cmyk` is interleaved CMYK, 4 bytes per pixel (PDF convention: 0 = no ink,
    /// 255 = full ink).  Returns interleaved RGB, 3 bytes per pixel.
    ///
    /// Two dispatch paths:
    /// - `grid` is `None` — the profile-free process-ink model
    ///   ([`color::cmyk`]), always evaluated on the CPU.
    /// - `grid` is `Some` — the conversion through the grid
    ///   ([`CmykGrid::convert`]), on the device from
    ///   [`GPU_ICC_CLUT_THRESHOLD`] pixels, with the same result.
    ///
    /// # Errors
    ///
    /// Returns an error if `cmyk.len()` is not a multiple of 4, or if GPU
    /// data transfer or kernel launch fails.
    pub fn icc_cmyk_to_rgb(
        &self,
        cmyk: &[u8],
        grid: Option<&CmykGrid>,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        check_cmyk_len(cmyk)?;
        let n = cmyk.len() / 4;
        match grid {
            // The CPU SIMD path always beats GPU on this machine for the
            // profile-free conversion: threshold_bench showed the PCIe
            // round-trip cost exceeds the compute cost at all measured
            // sizes (256–4M pixels).
            None => Ok(color::cmyk::cmyk_to_rgb_vec(cmyk)),
            Some(grid) if n < GPU_ICC_CLUT_THRESHOLD => Ok(grid.convert(cmyk)),
            Some(_) => self.icc_cmyk_to_rgb_gpu(cmyk, grid),
        }
    }

    /// Unconditional GPU dispatch for CMYK→RGB (skips threshold check).
    ///
    /// Use this when the caller has already decided GPU is appropriate
    /// (e.g. benchmarking or when the pixel count is known to be large).
    ///
    /// # Errors
    ///
    /// Returns an error if `cmyk.len()` is not a multiple of 4, if the pixel
    /// count exceeds [`MAX_KERNEL_PIXELS`], or if GPU data transfer or
    /// kernel launch fails.
    pub fn icc_cmyk_to_rgb_gpu(
        &self,
        cmyk: &[u8],
        grid: Option<&CmykGrid>,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        check_cmyk_len(cmyk)?;
        let n = cmyk.len() / 4;
        if n == 0 {
            // Match the sibling wrappers: zero pixels is a no-op, not a
            // zero-size device-allocation error.
            return Ok(Vec::new());
        }
        let n_u32 = u32::try_from(n)
            .ok()
            .filter(|&n| n <= MAX_KERNEL_PIXELS)
            .ok_or_else(|| {
                format!("{n} CMYK pixels exceed the kernel limit {MAX_KERNEL_PIXELS}")
            })?;
        let stream = self.stream();

        let d_cmyk = stream.clone_htod(cmyk)?;
        // Device-side zero alloc — both kernels write every output pixel,
        // so nothing is gained by shipping a zero Vec over PCIe.
        let mut d_rgb = stream.alloc_zeros::<u8>(n * 3)?;

        match grid {
            None => {
                let d_corners = stream.clone_htod(&color::cmyk::CORNER_TABLE)?;
                self.launch_icc_matrix_async(&d_cmyk, &mut d_rgb, &d_corners, n_u32)?;
            }
            Some(grid) => {
                let bytes = grid.as_bytes();
                let n_nodes = CmykGrid::node_count(bytes.len()).ok_or("a grid of no nodes")?;
                let d_grid = stream.clone_htod(bytes)?;
                self.launch_icc_clut_async(&d_cmyk, &mut d_rgb, &d_grid, n_nodes, n_u32)?;
            }
        }

        stream.synchronize()?;
        let rgb = stream.clone_dtoh(&d_rgb)?;
        Ok(rgb)
    }

    /// Async kernel launch for the profile-free CMYK→RGB kernel.
    ///
    /// `d_corners` holds [`color::cmyk::CORNER_TABLE`]; the kernel output is
    /// byte-identical to [`color::cmyk::cmyk_to_rgb_row`].  Caller is
    /// responsible for stream ordering and any final D→H download.  The
    /// public dispatcher always routes this conversion to the CPU; the
    /// launch exists for benchmarking and parity tests.
    ///
    /// # Errors
    ///
    /// Returns the underlying CUDA error if the launch fails.
    #[expect(
        unused_results,
        reason = "cudarc LaunchArgs::arg returns &mut Self for chaining; chain output is intentionally discarded"
    )]
    pub(crate) fn launch_icc_matrix_async(
        &self,
        d_cmyk: &CudaSlice<u8>,
        d_rgb: &mut CudaSlice<u8>,
        d_corners: &CudaSlice<u8>,
        n_pixels: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = launch_cfg(n_pixels as usize);
        let stream = self.stream();
        let mut builder = stream.launch_builder(&self.kernels.icc_cmyk_matrix);
        builder.arg(d_cmyk).arg(d_rgb).arg(d_corners).arg(&n_pixels);
        // SAFETY: 4 args match icc_cmyk_matrix PTX signature exactly.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// Async kernel launch for the ICC CMYK→RGB grid kernel: `d_clut`
    /// holds the bytes of a grid of `n_nodes` nodes
    /// ([`CmykGrid::as_bytes`]); the kernel reads no node beyond them.
    ///
    /// Caller is responsible for stream ordering and any final D→H download.
    ///
    /// # Errors
    ///
    /// Returns the underlying CUDA error if the launch fails.
    #[expect(
        unused_results,
        reason = "cudarc LaunchArgs::arg returns &mut Self for chaining; chain output is intentionally discarded"
    )]
    pub(crate) fn launch_icc_clut_async(
        &self,
        d_cmyk: &CudaSlice<u8>,
        d_rgb: &mut CudaSlice<u8>,
        d_clut: &CudaSlice<u8>,
        n_nodes: u32,
        n_pixels: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = launch_cfg(n_pixels as usize);
        let stream = self.stream();
        let mut builder = stream.launch_builder(&self.kernels.icc_cmyk_clut);
        builder
            .arg(d_cmyk)
            .arg(d_rgb)
            .arg(d_clut)
            .arg(&n_nodes)
            .arg(&n_pixels);
        // SAFETY: 5 args match icc_cmyk_clut PTX signature exactly.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }
}

#[cfg(all(test, feature = "gpu-validation"))]
mod tests {
    use crate::GpuCtx;

    /// A buffer of no whole pixels is an error, not a panic or a device
    /// read past it.
    #[test]
    fn icc_rejects_malformed_input_with_errors() {
        let gpu = GpuCtx::init().expect("gpu init");
        let grid = crate::backend::random_grid(1, [16; 4], color::grid::NodeRange::Unit);
        for grid in [None, Some(&grid)] {
            let e = gpu.icc_cmyk_to_rgb(&[0; 3], grid).unwrap_err().to_string();
            assert!(e.contains("multiple of 4"), "{e}");
            let e = gpu
                .icc_cmyk_to_rgb_gpu(&[0; 3], grid)
                .unwrap_err()
                .to_string();
            assert!(e.contains("multiple of 4"), "{e}");
        }
    }

    /// The CUDA profile-free kernel is byte-identical to the CPU conversion
    /// over every (ink, K) byte pair plus pseudo-random pixels.
    #[test]
    fn icc_matrix_kernel_matches_cpu() {
        let gpu = GpuCtx::init().expect("gpu init");
        let mut cmyk: Vec<u8> = (0u32..65_536)
            .flat_map(|i| {
                let [ink, k, ..] = i.to_le_bytes();
                [ink, 255 - ink, ink.wrapping_mul(167), k]
            })
            .collect();
        let mut s = 0x2468_ace0_u32;
        cmyk.extend((0..4 * 100_003).map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            s.to_be_bytes()[0]
        }));
        let got = gpu.icc_cmyk_to_rgb_gpu(&cmyk, None).expect("matrix kernel");
        assert_eq!(got, color::cmyk::cmyk_to_rgb_vec(&cmyk));
    }
}
