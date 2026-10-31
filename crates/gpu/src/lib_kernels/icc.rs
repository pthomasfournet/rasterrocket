//! ICC CMYK→RGB kernel dispatch.

use cudarc::driver::{CudaSlice, PushKernelArg};

use crate::backend::params::MAX_KERNEL_PIXELS;
use crate::cmyk::{check_clut, icc_cmyk_to_rgb_cpu};
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
    /// Returns an error if `cmyk.len()` is not a multiple of 4, if `clut` is
    /// `Some` with `grid_n` outside `[2, 255]` or `table.len() != grid_n⁴ × 3`,
    /// or if GPU data transfer or kernel launch fails.
    pub fn icc_cmyk_to_rgb(
        &self,
        cmyk: &[u8],
        clut: Option<(&[u8], u32)>,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        check_cmyk_len(cmyk)?;
        // Early-out before any CLUT validation: empty input always produces empty output.
        let n = cmyk.len() / 4;
        if n == 0 {
            return Ok(Vec::new());
        }
        let Some((table, grid_n)) = clut else {
            // Profile-free path: the CPU SIMD path always beats GPU on this machine —
            // threshold_bench showed the PCIe round-trip cost exceeds the compute cost
            // at all measured sizes (256–4M pixels).
            return Ok(icc_cmyk_to_rgb_cpu(cmyk, None));
        };
        check_clut(table, grid_n)?;
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
    /// Returns an error if `cmyk.len()` is not a multiple of 4, if the pixel
    /// count exceeds [`MAX_KERNEL_PIXELS`], if `clut` is `Some` and fails the
    /// [`icc_cmyk_to_rgb`](Self::icc_cmyk_to_rgb) table check or exceeds
    /// `i32::MAX` bytes (the kernel's 32-bit node index), or if GPU data
    /// transfer or kernel launch fails.
    pub fn icc_cmyk_to_rgb_gpu(
        &self,
        cmyk: &[u8],
        clut: Option<(&[u8], u32)>,
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
        if let Some((table, grid_n)) = clut {
            check_clut(table, grid_n)?;
            if i32::try_from(table.len()).is_err() {
                return Err(format!(
                    "CLUT of {} bytes exceeds the kernel's 32-bit node index",
                    table.len()
                )
                .into());
            }
        }
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

    /// Malformed buffers and tables are errors, not panics or device reads
    /// past a buffer.
    #[test]
    fn icc_rejects_malformed_input_with_errors() {
        let gpu = GpuCtx::init().expect("gpu init");
        let table = vec![0u8; 17usize.pow(4) * 3];
        for (cmyk, clut, what) in [
            (&[0u8; 3][..], None, "multiple of 4"),
            (&[0u8; 4][..], Some((&table[..], 33)), "table length"),
            (&[0u8; 4][..], Some((&[][..], 0)), "grid_n"),
        ] {
            let e = gpu.icc_cmyk_to_rgb(cmyk, clut).unwrap_err().to_string();
            assert!(e.contains(what), "{e}");
            let e = gpu.icc_cmyk_to_rgb_gpu(cmyk, clut).unwrap_err().to_string();
            assert!(e.contains(what), "{e}");
        }
    }

    /// The CUDA CLUT kernel is byte-identical to the CPU evaluator on a
    /// pseudo-random 17-node table, at the lattice nodes and between them.
    #[test]
    fn icc_clut_kernel_matches_cpu() {
        let gpu = GpuCtx::init().expect("gpu init");
        let mut s = 0x0bad_5eed_u32;
        let mut next = move || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            s.to_be_bytes()[0]
        };
        let table: Vec<u8> = (0..17usize.pow(4) * 3).map(|_| next()).collect();
        let mut cmyk: Vec<u8> = (0u32..=255)
            .flat_map(|v| {
                let v = u8::try_from(v).expect("byte");
                [v, v, v, v]
            })
            .collect();
        cmyk.extend((0..4 * 200_000).map(|_| next()));
        let got = gpu
            .icc_cmyk_to_rgb_gpu(&cmyk, Some((&table, 17)))
            .expect("clut kernel");
        assert_eq!(got, crate::icc_cmyk_to_rgb_cpu(&cmyk, Some((&table, 17))));
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
