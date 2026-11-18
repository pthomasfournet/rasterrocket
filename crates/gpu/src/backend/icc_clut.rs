//! The ICC CMYK → RGB conversion through a grid ([`CmykGrid`]) as one
//! backend page, on either backend.
//!
//! The kernels interpolate the grid in integer arithmetic, the
//! arithmetic of [`CmykGrid::convert`] and [`CmykGrid::convert16`], so a
//! device's conversion is the host's value for value, at 8 or 16 bits; [`icc_clut_matches_cpu`] checks a device against
//! it.

use color::grid::{CmykGrid, NodeRange};

use super::layout::{device_pixel_words, unpad_pixels};
use super::params::{IccClutParams, MAX_KERNEL_PIXELS};
use super::staged::{
    Direction, Staged, ensure, host_view, record_download, record_upload, stage, stage_exact,
};
use super::{BackendError, GpuBackend, Result};

/// Reusable host and device buffers for [`icc_clut`].
///
/// The CMYK input and the RGB output are grown to the largest conversion
/// seen; the grid is held at exactly its length, which the kernel reads
/// its node count from.  Lent by the backend's pool
/// ([`GpuBackend::scratch`]).
pub struct IccClutScratch<B: GpuBackend> {
    cmyk: Option<Staged<B>>,
    clut: Option<Staged<B>>,
    rgb: Option<Staged<B>>,
}

/// An empty scratch; the first conversion allocates.
impl<B: GpuBackend> Default for IccClutScratch<B> {
    fn default() -> Self {
        Self {
            cmyk: None,
            clut: None,
            rgb: None,
        }
    }
}

/// The RGB conversion of the interleaved CMYK pixels `cmyk` through
/// `grid`, computed on `backend`: [`CmykGrid::convert`] byte for byte.
///
/// The pixels and the grid are written into the host buffers of a scratch
/// lent by `backend`'s pool, and their uploads, the kernel and the download
/// of the result are recorded into one page ([`GpuBackend::run_page`]).  On
/// a device whose buffers the host reaches in place there are no copies.
/// Zero pixels yield an empty result without a device round trip.  The
/// caller decides whether the conversion is worth one (see
/// [`crate::GPU_ICC_CLUT_THRESHOLD`]).
///
/// # Errors
///
/// [`BackendError::InvariantViolation`] when `cmyk` does not hold whole
/// pixels or the pixel count violates the [`IccClutParams`] invariants;
/// the backend's error when an allocation, a transfer or the kernel fails.
pub fn icc_clut<B: GpuBackend>(backend: &B, cmyk: &[u8], grid: &CmykGrid) -> Result<Vec<u8>> {
    convert(backend, cmyk, grid, false)
}

/// [`icc_clut`] at 16 bits: [`CmykGrid::convert16`] value for value.
///
/// # Errors
///
/// As [`icc_clut`].
pub fn icc_clut16<B: GpuBackend>(backend: &B, cmyk: &[u8], grid: &CmykGrid) -> Result<Vec<u16>> {
    let bytes = convert(backend, cmyk, grid, true)?;
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&b| u16::from_le_bytes(b))
        .collect())
}

/// [`icc_clut`], the output of 16-bit values, two bytes each, low first,
/// when `wide`: the device's output words, unpadded.
fn convert<B: GpuBackend>(
    backend: &B,
    cmyk: &[u8],
    grid: &CmykGrid,
    wide: bool,
) -> Result<Vec<u8>> {
    const KIND: &str = "IccClutInvariantViolation";
    if !cmyk.len().is_multiple_of(4) {
        return Err(BackendError::invariant(
            KIND,
            format!(
                "cmyk buffer length {} bytes is not a multiple of 4",
                cmyk.len()
            ),
        ));
    }
    let n = cmyk.len() / 4;
    if n == 0 {
        return Ok(Vec::new());
    }
    let n_pixels = u32::try_from(n)
        .ok()
        .filter(|&n| n <= MAX_KERNEL_PIXELS)
        .ok_or_else(|| {
            BackendError::invariant(
                KIND,
                format!(
                    "n_pixels {n} exceeds MAX_KERNEL_PIXELS {MAX_KERNEL_PIXELS} \
                     (32-bit byte addressing)"
                ),
            )
        })?;
    let table = grid.as_bytes();
    let sample_bytes = if wide { 2 } else { 1 };
    let rgb_len = n * 3 * sample_bytes;
    let device_len = n * 4 * device_pixel_words(sample_bytes);
    backend.scratch().icc_clut.with(|scratch| {
        let IccClutScratch {
            cmyk: cmyk_slot,
            clut: clut_slot,
            rgb: rgb_slot,
        } = scratch;
        let d_cmyk = stage(backend, cmyk_slot, cmyk)?;
        let d_clut = stage_exact(backend, clut_slot, table)?;
        let d_rgb = ensure(backend, rgb_slot, device_len, Direction::ToHost)?;
        // SAFETY (the recorded copies and the read below): the scratch is
        // borrowed mutably for this whole call and `run_page` returns only
        // once its page has completed or, on failure, been waited on.
        backend.run_page(|backend| {
            unsafe { record_upload(backend, d_cmyk, cmyk.len()) }?;
            unsafe { record_upload(backend, d_clut, table.len()) }?;
            backend.record_icc_clut(IccClutParams {
                cmyk: &d_cmyk.device,
                rgb: &mut d_rgb.device,
                clut: &d_clut.device,
                n_pixels,
                wide,
            })?;
            unsafe { record_download(backend, d_rgb, device_len) }
        })?;
        let device = unsafe { host_view(backend, d_rgb) }?;
        let mut rgb = vec![0; rgb_len];
        unpad_pixels(&device[..device_len], &mut rgb, sample_bytes);
        Ok(rgb)
    })
}

/// A pseudo-random sequence of 32-bit values from `seed`.
fn lcg(seed: u32) -> impl FnMut() -> u32 {
    let mut state = seed;
    move || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        state
    }
}

/// A grid of `spans[i]` cells along input `i` in `range`, of pseudo-random
/// nodes and pseudo-random cells of each sample, from `seed`.
///
/// Every order of the positions, every weight and, in the extended range,
/// both clamps occur.
///
/// # Panics
///
/// For a span of 0.
#[must_use]
pub fn random_grid(seed: u32, spans: [u32; 4], range: NodeRange) -> CmykGrid {
    let mut next = lcg(seed);
    let mut cells = [[0u32; 256]; 4];
    for (axis, span) in cells.iter_mut().zip(spans) {
        for cell in axis.iter_mut() {
            let index = (next() >> 16) % span;
            *cell = (index << 16) | (next() >> 16);
        }
    }
    let mut strides = [1u32; 4];
    for i in (0..3).rev() {
        strides[i] = strides[i + 1] * (spans[i + 1] + 1);
    }
    let count = strides[0] * (spans[0] + 1);
    let mut value = || u16::try_from(next() >> 16).expect("the high 16 bits");
    let nodes: Vec<[u16; 4]> = (0..count).map(|_| [value(), value(), value(), 0]).collect();
    CmykGrid::new(&cells, strides, range, &nodes).expect("the cells lie within the grid")
}

/// Every grey level and 4 096 pseudo-random pixels.
fn probe_pixels() -> Vec<u8> {
    let mut next = lcg(0x0bad_5eed);
    let mut cmyk: Vec<u8> = (0..=u8::MAX).flat_map(|v| [v; 4]).collect();
    cmyk.extend((0..4 * 4096).map(|_| next().to_be_bytes()[0]));
    cmyk
}

/// Whether [`icc_clut`] and [`icc_clut16`] on `backend` reproduce
/// [`CmykGrid::convert`] and [`CmykGrid::convert16`] value for value.
///
/// The check converts the `probe_pixels` through a [`random_grid`] of
/// uneven spans in the extended range, whose outputs clamp at both ends.
/// The interpolation is in integers, so a device differs only where its
/// storage, integer arithmetic or kernel compilation is faulty.
///
/// # Errors
///
/// As [`icc_clut`].
pub fn icc_clut_matches_cpu<B: GpuBackend>(backend: &B) -> Result<bool> {
    let grid = random_grid(0x0dd_ba11, [7, 4, 5, 3], NodeRange::Extended);
    let cmyk = probe_pixels();
    Ok(icc_clut(backend, &cmyk, &grid)? == grid.convert(&cmyk)
        && icc_clut16(backend, &cmyk, &grid)? == grid.convert16(&cmyk))
}

/// The grid shapes the device evaluators are checked on: one cell, uneven
/// spans, the 17⁴ and 33⁴ node grids of the converter's CMYK tables.
#[cfg(all(test, any(feature = "gpu-validation", feature = "vulkan")))]
const EXACTNESS_SPANS: [[u32; 4]; 4] = [[1; 4], [20, 15, 16, 7], [16; 4], [32; 4]];

/// Converts, on a grid of each of the [`EXACTNESS_SPANS`] in each range,
/// every grey level and 65 536 pseudo-random pixels with `device`, and
/// asserts equality with `host`, value for value.
#[cfg(all(test, any(feature = "gpu-validation", feature = "vulkan")))]
fn assert_exact_across_grids<T: PartialEq>(
    label: &str,
    host: impl Fn(&CmykGrid, &[u8]) -> Vec<T>,
    device: impl Fn(&[u8], &CmykGrid) -> Vec<T>,
) {
    let mut next = lcg(0x1234_5678);
    let mut cmyk: Vec<u8> = (0..=u8::MAX).flat_map(|v| [v; 4]).collect();
    cmyk.extend((0..65_536 * 4).map(|_| next().to_be_bytes()[0]));
    for (seed, spans) in (1u32..).zip(EXACTNESS_SPANS) {
        for range in [NodeRange::Unit, NodeRange::Extended] {
            let grid = random_grid(seed, spans, range);
            let cpu = host(&grid, &cmyk);
            let device = device(&cmyk, &grid);
            assert_eq!(cpu.len(), device.len(), "{label}: lengths differ");
            let differing = cpu.iter().zip(&device).filter(|(a, b)| a != b).count();
            assert_eq!(
                differing, 0,
                "{label}: spans {spans:?} {range:?}: {differing} values differ from the host"
            );
        }
    }
}

/// Conversions reuse one pooled scratch, whose grid buffer follows the
/// grid's length down as well as up.
#[cfg(all(test, any(feature = "gpu-validation", feature = "vulkan")))]
fn check_scratch_is_pooled<B: GpuBackend>(backend: &B) {
    let cmyk: Vec<u8> = (0..=u8::MAX)
        .flat_map(|v| [v, 255 - v, v / 2, v / 3])
        .collect();
    for spans in [[32; 4], [16; 4], [1; 4]] {
        let grid = random_grid(7, spans, NodeRange::Unit);
        let device = icc_clut(backend, &cmyk, &grid).expect("icc_clut");
        assert_eq!(device, grid.convert(&cmyk));
        assert_eq!(
            backend.scratch().icc_clut.idle_len(),
            1,
            "the scratch is returned"
        );
        let scratch = backend.scratch().icc_clut.take_idle();
        let clut = scratch.clut.as_ref().expect("the grid slot is kept");
        assert_eq!(
            backend.device_buffer_len(&clut.device),
            grid.as_bytes().len()
        );
        backend.scratch().icc_clut.with(|s| *s = scratch);
    }
}

#[cfg(all(test, feature = "vulkan"))]
mod vulkan_tests {
    use super::*;

    /// On a device that passes the exactness probe, the Vulkan kernel is
    /// byte-identical to the host on every grid.
    #[test]
    fn vulkan_icc_clut_exact_across_grids() {
        let Some(backend) = crate::backend::vulkan::test_backend() else {
            return;
        };
        assert!(
            backend.icc_clut_is_exact(),
            "the device fails the exactness probe"
        );
        assert_exact_across_grids("vulkan", CmykGrid::convert, |cmyk, grid| {
            icc_clut(&backend, cmyk, grid).expect("icc_clut")
        });
        assert_exact_across_grids("vulkan 16-bit", CmykGrid::convert16, |cmyk, grid| {
            icc_clut16(&backend, cmyk, grid).expect("icc_clut16")
        });
    }

    #[test]
    fn vulkan_icc_clut_scratch_is_pooled() {
        let Some(backend) = crate::backend::vulkan::test_backend() else {
            return;
        };
        check_scratch_is_pooled(&backend);
    }
}

#[cfg(all(test, feature = "gpu-validation"))]
mod tests {
    use super::*;

    /// The CUDA backend's page is byte-identical to the host.
    #[test]
    fn cuda_icc_clut_matches_cpu() {
        let Ok(backend) = crate::backend::cuda::CudaBackend::new() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        assert!(icc_clut_matches_cpu(&backend).expect("icc_clut"));
    }

    /// The CUDA backend's page and the context's direct launch are
    /// byte-identical to the host on every grid.
    #[test]
    fn cuda_icc_clut_exact_across_grids() {
        let Ok(backend) = crate::backend::cuda::CudaBackend::new() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        assert_exact_across_grids("cuda backend", CmykGrid::convert, |cmyk, grid| {
            icc_clut(&backend, cmyk, grid).expect("icc_clut")
        });
        assert_exact_across_grids("cuda backend 16-bit", CmykGrid::convert16, |cmyk, grid| {
            icc_clut16(&backend, cmyk, grid).expect("icc_clut16")
        });
        let gpu = crate::GpuCtx::init().expect("gpu init");
        assert_exact_across_grids("cuda context", CmykGrid::convert, |cmyk, grid| {
            gpu.icc_cmyk_to_rgb_gpu(cmyk, Some(grid))
                .expect("icc_cmyk_to_rgb_gpu")
        });
    }

    #[test]
    fn cuda_icc_clut_scratch_is_pooled() {
        let Ok(backend) = crate::backend::cuda::CudaBackend::new() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        check_scratch_is_pooled(&backend);
    }

    /// Malformed input is an invariant error, and zero pixels are no work.
    #[test]
    fn malformed_input_is_rejected() {
        let Ok(backend) = crate::backend::cuda::CudaBackend::new() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let grid = random_grid(3, [16; 4], NodeRange::Unit);
        let invariant =
            |r: Result<Vec<u8>>| matches!(r, Err(BackendError::InvariantViolation { .. }));
        assert!(invariant(icc_clut(&backend, &[0; 3], &grid)));
        assert_eq!(
            icc_clut(&backend, &[], &grid).expect("empty"),
            Vec::<u8>::new()
        );
    }
}
