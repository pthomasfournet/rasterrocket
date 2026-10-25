//! ICC CMYK→RGB kernel dispatch.

use cudarc::driver::{CudaSlice, PushKernelArg};

use crate::{GPU_ICC_CLUT_THRESHOLD, GpuCtx, cmyk::icc_cmyk_to_rgb_cpu, launch_cfg};

impl GpuCtx {
    /// Convert CMYK pixels to RGB using a GPU kernel.
    ///
    /// `cmyk` is interleaved CMYK, 4 bytes per pixel (PDF convention: 0 = no ink,
    /// 255 = full ink).  Returns interleaved RGB, 3 bytes per pixel.
    ///
    /// Two dispatch paths:
    /// - `clut` is `None` — the profile-free process-ink model
    ///   ([`color::cmyk`]), always evaluated on the CPU.
    /// - `clut` is `Some((table, grid_n))` — uses the 4D quadrilinear CLUT kernel.
    ///   `table` must be `grid_n^4 * 3` bytes, ordered
    ///   `(k * G³ + c * G² + m * G + y) * 3` (RGB output values, u8).
    ///   `grid_n` is typically 17 (83 521 nodes) or 33 (1 185 921 nodes).
    ///
    /// Falls back to [`icc_cmyk_to_rgb_cpu`] when `n_pixels < GPU_ICC_CLUT_THRESHOLD`
    /// or `cmyk` is empty.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU data transfer or kernel launch fails.
    ///
    /// # Panics
    ///
    /// Panics if `cmyk.len()` is not a multiple of 4, or if `clut` is `Some` and
    /// `table.len() != grid_n^4 * 3`.
    pub fn icc_cmyk_to_rgb(
        &self,
        cmyk: &[u8],
        clut: Option<(&[u8], u32)>,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        assert!(
            cmyk.len().is_multiple_of(4),
            "cmyk.len() must be a multiple of 4 (got {})",
            cmyk.len()
        );

        // Early-out before any CLUT validation: empty input always produces empty output.
        let n = cmyk.len() / 4;
        if n == 0 {
            return Ok(Vec::new());
        }

        if let Some((table, grid_n)) = clut {
            // grid_n ≤ 255 is enforced by the baking API; checked_pow guards future misuse.
            let expected = (grid_n as usize)
                .checked_pow(4)
                .and_then(|n| n.checked_mul(3))
                .unwrap_or_else(|| {
                    panic!("grid_n({grid_n})^4*3 overflows usize — grid_n must be ≤ 255")
                });
            assert_eq!(
                table.len(),
                expected,
                "CLUT table length {got} ≠ grid_n({grid_n})^4*3={expected}",
                got = table.len(),
            );
        }
        // Profile-free path (clut=None): the CPU SIMD path always beats GPU on this machine —
        // threshold_bench showed the PCIe round-trip cost exceeds the compute cost
        // at all measured sizes (256–4M pixels).  Always use the CPU path here.
        if clut.is_none() {
            return Ok(icc_cmyk_to_rgb_cpu(cmyk, None));
        }
        if n < GPU_ICC_CLUT_THRESHOLD {
            return Ok(icc_cmyk_to_rgb_cpu(cmyk, clut));
        }

        self.icc_cmyk_to_rgb_gpu(cmyk, clut)
    }

    /// Unconditional GPU dispatch for CMYK→RGB (skips threshold check).
    ///
    /// Use this when the caller has already decided GPU is appropriate
    /// (e.g. benchmarking or when the pixel count is known to be large).
    ///
    /// # Errors
    ///
    /// Returns an error if GPU data transfer or kernel launch fails.
    ///
    /// # Panics
    ///
    /// Panics if `cmyk.len()` is not a multiple of 4 or if the pixel count
    /// overflows `u32::MAX`.
    pub fn icc_cmyk_to_rgb_gpu(
        &self,
        cmyk: &[u8],
        clut: Option<(&[u8], u32)>,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        assert!(
            cmyk.len().is_multiple_of(4),
            "cmyk.len() must be a multiple of 4 (got {})",
            cmyk.len()
        );
        let n = cmyk.len() / 4;
        if n == 0 {
            // Match the sibling wrappers: zero pixels is a no-op, not a
            // zero-size device-allocation error.
            return Ok(Vec::new());
        }
        let n_u32 = u32::try_from(n).expect("pixel count exceeds u32::MAX");
        let stream = self.stream();

        let d_cmyk = stream.clone_htod(cmyk)?;
        // Device-side zero alloc — both kernels write every output pixel,
        // so nothing is gained by shipping a zero Vec over PCIe.
        let d_rgb = stream.alloc_zeros::<u8>(n * 3)?;

        match clut {
            None => {
                let d_corners = stream.clone_htod(&color::cmyk::CORNER_TABLE)?;
                self.launch_icc_matrix_async(&d_cmyk, &d_rgb, &d_corners, n_u32)?;
            }
            Some((table, grid_n)) => {
                let d_clut = stream.clone_htod(table)?;
                self.launch_icc_clut_async(&d_cmyk, &d_rgb, &d_clut, grid_n, n_u32)?;
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
        d_rgb: &CudaSlice<u8>,
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

    /// Async kernel launch for the ICC CMYK→RGB 4D CLUT kernel.
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
        d_rgb: &CudaSlice<u8>,
        d_clut: &CudaSlice<u8>,
        grid_n: u32,
        n_pixels: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = launch_cfg(n_pixels as usize);
        let stream = self.stream();
        let mut builder = stream.launch_builder(&self.kernels.icc_cmyk_clut);
        builder
            .arg(d_cmyk)
            .arg(d_rgb)
            .arg(d_clut)
            .arg(&grid_n)
            .arg(&n_pixels);
        // SAFETY: 5 args match icc_cmyk_clut PTX signature exactly.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }
}

#[cfg(all(test, feature = "gpu-validation"))]
mod tests {
    use crate::GpuCtx;

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
