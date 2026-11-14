//! The ICC CMYK → RGB CLUT transform as one backend page, on either
//! backend.

use super::params::{IccClutParams, MAX_KERNEL_PIXELS, grid_n_from_clut_len};
use super::{BackendError, GpuBackend, Result};

/// A device buffer freed on drop.
///
/// Every page that references the buffer has completed before the guard
/// drops: [`GpuBackend::run_page`] returns only once its page has
/// completed or been abandoned, and the transfers are synchronous.
struct Owned<'a, B: GpuBackend> {
    backend: &'a B,
    buf: Option<B::DeviceBuffer>,
}

impl<'a, B: GpuBackend> Owned<'a, B> {
    const fn new(backend: &'a B, buf: B::DeviceBuffer) -> Self {
        Self {
            backend,
            buf: Some(buf),
        }
    }

    fn get(&self) -> Result<&B::DeviceBuffer> {
        self.buf
            .as_ref()
            .ok_or_else(|| BackendError::msg("icc_clut: device buffer already freed"))
    }
}

impl<B: GpuBackend> Drop for Owned<'_, B> {
    fn drop(&mut self) {
        if let Some(buf) = self.buf.take() {
            self.backend.free_device(buf);
        }
    }
}

/// The RGB conversion of the interleaved CMYK pixels `cmyk` through the
/// CLUT `table`, computed on `backend`.
///
/// `table` holds `grid_n⁴` RGB nodes indexed `(k·G³ + c·G² + m·G + y) · 3`
/// for some `grid_n` in `[2, 255]`; the result holds three bytes per pixel.
/// The pixels and the table are uploaded, the kernel runs as one page
/// ([`GpuBackend::run_page`]) and the result is downloaded.  Zero pixels
/// yield an empty result without a device round trip.  The caller decides
/// whether the conversion is worth one (see
/// [`crate::GPU_ICC_CLUT_THRESHOLD`]).
///
/// # Errors
///
/// [`BackendError::InvariantViolation`] when `cmyk` does not hold whole
/// pixels, or when the pixel count or `table` violates the
/// [`IccClutParams`] invariants; the backend's error when an allocation, a
/// transfer or the kernel fails.
pub fn icc_clut<B: GpuBackend>(backend: &B, cmyk: &[u8], table: &[u8]) -> Result<Vec<u8>> {
    const KIND: &str = "IccClutInvariantViolation";
    let invariant = |detail| BackendError::InvariantViolation { kind: KIND, detail };
    if !cmyk.len().is_multiple_of(4) {
        return Err(invariant("cmyk buffer length is not a multiple of 4"));
    }
    let n = cmyk.len() / 4;
    if n == 0 {
        return Ok(Vec::new());
    }
    let n_pixels = u32::try_from(n)
        .ok()
        .filter(|&n| n <= MAX_KERNEL_PIXELS)
        .ok_or_else(|| invariant("n_pixels exceeds MAX_KERNEL_PIXELS (32-bit byte addressing)"))?;
    if i32::try_from(table.len()).is_err() || grid_n_from_clut_len(table.len()).is_none() {
        return Err(invariant(
            "clut length is not grid_n^4 × 3 for grid_n in [2, 255] within i32::MAX bytes",
        ));
    }

    let d_cmyk = Owned::new(backend, backend.alloc_device(cmyk.len())?);
    let d_clut = Owned::new(backend, backend.alloc_device(table.len())?);
    let d_rgb = Owned::new(backend, backend.alloc_device_readback(n * 3)?);
    backend.upload(d_cmyk.get()?, cmyk)?;
    backend.upload(d_clut.get()?, table)?;
    backend.run_page(|backend| {
        backend.record_icc_clut(IccClutParams {
            cmyk: d_cmyk.get()?,
            rgb: d_rgb.get()?,
            clut: d_clut.get()?,
            n_pixels,
        })
    })?;
    let mut rgb = vec![0; n * 3];
    backend.download(d_rgb.get()?, &mut rgb)?;
    Ok(rgb)
}

/// Nodes per axis of the [`probe_table`].
const PROBE_GRID_N: u32 = 17;

/// Pixels whose conversion through the [`probe_table`] changes by one
/// level in some channel when every interpolation step rounds its product
/// and its sum separately instead of once.
const PROBE_PIXELS: [[u8; 4]; 24] = [
    [164, 63, 41, 88],
    [63, 146, 97, 80],
    [157, 211, 147, 215],
    [106, 136, 199, 30],
    [50, 197, 115, 104],
    [177, 27, 236, 121],
    [248, 110, 241, 223],
    [206, 188, 174, 185],
    [80, 229, 183, 247],
    [48, 38, 103, 39],
    [224, 14, 204, 14],
    [52, 203, 212, 16],
    [233, 148, 28, 247],
    [28, 69, 33, 244],
    [111, 172, 196, 236],
    [71, 158, 38, 50],
    [43, 46, 37, 108],
    [207, 32, 243, 72],
    [55, 226, 254, 231],
    [96, 204, 10, 42],
    [63, 149, 45, 118],
    [78, 2, 154, 29],
    [158, 246, 9, 59],
    [187, 18, 80, 52],
];

/// The pseudo-random `PROBE_GRID_N`⁴ table the [`PROBE_PIXELS`] are
/// chosen for: neighbouring nodes differ arbitrarily, so every
/// interpolation step carries a non-trivial difference.
fn probe_table() -> Vec<u8> {
    let mut state = 0x0bad_5eed_u32;
    (0..(PROBE_GRID_N as usize).pow(4) * 3)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state.to_be_bytes()[0]
        })
        .collect()
}

/// Whether [`icc_clut`] on `backend` reproduces
/// [`crate::icc_cmyk_to_rgb_cpu`] byte for byte.
///
/// Converts every grey level and the [`PROBE_PIXELS`] through the
/// [`probe_table`] on the device and on the CPU.  A device whose
/// interpolation rounds twice per step differs at every probe pixel.
///
/// # Errors
///
/// As [`icc_clut`].
pub fn icc_clut_matches_cpu<B: GpuBackend>(backend: &B) -> Result<bool> {
    let table = probe_table();
    let mut cmyk: Vec<u8> = (0..=u8::MAX).flat_map(|v| [v; 4]).collect();
    cmyk.extend(PROBE_PIXELS.as_flattened());
    let device = icc_clut(backend, &cmyk, &table)?;
    Ok(device == crate::icc_cmyk_to_rgb_cpu(&cmyk, Some((&table, PROBE_GRID_N))))
}

#[cfg(test)]
mod probe_tests {
    use super::{PROBE_GRID_N, PROBE_PIXELS, probe_table};

    /// The CPU evaluator's interpolation with the product and the sum of
    /// each step rounded separately.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        clippy::many_single_char_names,
        reason = "lattice coordinates lie in [0, 16], outputs are clamped to a byte, and \
                  c/m/y/k/g name the channels and the grid as in the evaluator"
    )]
    fn eval_twice_rounded(table: &[u8], px: [u8; 4]) -> [u8; 3] {
        let g = PROBE_GRID_N as usize;
        let scale = (PROBE_GRID_N - 1) as f32 / 255.0;
        let axis = |v: u8| {
            let f = f32::from(v) * scale;
            let i0 = (f as usize).min(g - 1);
            (i0, (i0 + 1).min(g - 1), f - i0 as f32)
        };
        let node = |c: usize, m: usize, y: usize, k: usize| -> [f32; 3] {
            let i = (((k * g + c) * g + m) * g + y) * 3;
            std::array::from_fn(|ch| f32::from(table[i + ch]))
        };
        let lerp = |a: [f32; 3], b: [f32; 3], t: f32| -> [f32; 3] {
            std::array::from_fn(|ch| std::hint::black_box(t * (b[ch] - a[ch])) + a[ch])
        };
        let [c, m, y, k] = px.map(axis);
        let pick = |a: (usize, usize, f32), hi: bool| if hi { a.1 } else { a.0 };
        let face = |kk: bool| {
            let ym = |cc: bool, mm: bool| {
                let at = |yy| node(pick(c, cc), pick(m, mm), pick(y, yy), pick(k, kk));
                lerp(at(false), at(true), y.2)
            };
            let c0 = lerp(ym(false, false), ym(false, true), m.2);
            let c1 = lerp(ym(true, false), ym(true, true), m.2);
            lerp(c0, c1, c.2)
        };
        lerp(face(false), face(true), k.2).map(|v| v.clamp(0.0, 255.0).round() as u8)
    }

    /// Every probe pixel tells single from double rounding apart.
    #[test]
    fn probe_pixels_detect_double_rounding() {
        let table = probe_table();
        for px in PROBE_PIXELS {
            let once = crate::icc_cmyk_to_rgb_cpu(&px, Some((&table, PROBE_GRID_N)));
            assert_ne!(once, eval_twice_rounded(&table, px), "{px:?}");
        }
    }
}

/// Grid sizes the device evaluators are checked on.  For several of them
/// (`grid_n` 4, 8, 13, 16, 25) a reciprocal-based device division moves
/// `(grid_n − 1) / 255` by one ULP away from the correctly rounded value.
#[cfg(test)]
const EXACTNESS_GRIDS: [u32; 9] = [2, 3, 4, 8, 13, 16, 17, 25, 33];

/// Converts, on each of the [`EXACTNESS_GRIDS`], every grey level and
/// 65 536 pseudo-random pixels through a pseudo-random table with
/// `convert(cmyk, table, grid_n)`, and asserts byte equality with
/// [`crate::icc_cmyk_to_rgb_cpu`].
#[cfg(test)]
fn assert_exact_across_grids(label: &str, convert: impl Fn(&[u8], &[u8], u32) -> Vec<u8>) {
    let lcg = |seed: u32, n: usize| {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                state.to_be_bytes()[0]
            })
            .collect::<Vec<u8>>()
    };
    let mut cmyk: Vec<u8> = (0..=u8::MAX).flat_map(|v| [v; 4]).collect();
    cmyk.extend(lcg(0x1234_5678, 65_536 * 4));
    for grid_n in EXACTNESS_GRIDS {
        let table = lcg(0x0dd_ba11 ^ grid_n, (grid_n as usize).pow(4) * 3);
        let cpu = crate::icc_cmyk_to_rgb_cpu(&cmyk, Some((&table, grid_n)));
        let device = convert(&cmyk, &table, grid_n);
        let differing = cpu.iter().zip(&device).filter(|(a, b)| a != b).count();
        assert_eq!(
            differing, 0,
            "{label}: grid_n {grid_n}: {differing} bytes differ from the CPU evaluator"
        );
    }
}

#[cfg(all(test, feature = "vulkan"))]
mod vulkan_tests {
    use super::*;

    /// On a device that passes the exactness probe, the Vulkan kernel is
    /// byte-identical to the CPU evaluator on every grid size.
    #[test]
    fn vulkan_icc_clut_exact_across_grids() {
        let Some(backend) = crate::backend::vulkan::test_backend() else {
            return;
        };
        if !backend.icc_clut_is_exact() {
            eprintln!("skipping: the device fails the exactness probe");
            return;
        }
        assert_exact_across_grids("vulkan", |cmyk, table, _| {
            icc_clut(&backend, cmyk, table).expect("icc_clut")
        });
    }
}

#[cfg(all(test, feature = "gpu-validation"))]
mod tests {
    use super::*;

    /// The CUDA backend's page is byte-identical to the CPU evaluator.
    #[test]
    fn cuda_icc_clut_matches_cpu() {
        let Ok(backend) = crate::backend::cuda::CudaBackend::new() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        assert!(icc_clut_matches_cpu(&backend).expect("icc_clut"));
    }

    /// The CUDA backend's page and the context's direct launch are
    /// byte-identical to the CPU evaluator on every grid size.
    #[test]
    fn cuda_icc_clut_exact_across_grids() {
        let Ok(backend) = crate::backend::cuda::CudaBackend::new() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        assert_exact_across_grids("cuda backend", |cmyk, table, _| {
            icc_clut(&backend, cmyk, table).expect("icc_clut")
        });
        let gpu = crate::GpuCtx::init().expect("gpu init");
        assert_exact_across_grids("cuda context", |cmyk, table, grid_n| {
            gpu.icc_cmyk_to_rgb_gpu(cmyk, Some((table, grid_n)))
                .expect("icc_cmyk_to_rgb_gpu")
        });
    }

    /// Malformed input is an invariant error, and zero pixels are no work.
    #[test]
    fn malformed_input_is_rejected() {
        let Ok(backend) = crate::backend::cuda::CudaBackend::new() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let table = vec![0u8; 17usize.pow(4) * 3];
        let invariant =
            |r: Result<Vec<u8>>| matches!(r, Err(BackendError::InvariantViolation { .. }));
        assert!(invariant(icc_clut(&backend, &[0; 3], &table)));
        assert!(invariant(icc_clut(
            &backend,
            &[0; 4],
            &table[..table.len() - 3]
        )));
        assert!(invariant(icc_clut(&backend, &[0; 4], &[])));
        assert_eq!(
            icc_clut(&backend, &[], &table).expect("empty"),
            Vec::<u8>::new()
        );
    }
}
