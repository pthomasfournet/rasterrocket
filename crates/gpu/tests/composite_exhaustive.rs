//! `composite_rgba8` and `apply_soft_mask` against their CPU references over
//! every alpha pair, on each backend built in.
//!
//! The CUDA module needs `gpu-validation` (a CUDA device); the Vulkan module
//! needs `vulkan`:
//!   `cargo test -p rasterrocket-gpu --features gpu-validation,vulkan --test composite_exhaustive`

#![cfg(any(feature = "gpu-validation", feature = "vulkan"))]

use rasterrocket_gpu::backend::GpuBackend;
use rasterrocket_gpu::backend::params::{CompositeParams, SoftMaskParams};

/// One pixel per (alpha, other) pair: `alpha` in the high byte of the index.
const PAIRS: usize = 256 * 256;

/// Colour bytes from a 32-bit LCG; deterministic across runs.
fn colours(seed: u32) -> impl FnMut() -> u8 {
    let mut state = seed;
    move || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        state.to_be_bytes()[0]
    }
}

/// Source and destination pixels covering every (`a_src`, `a_dst`) pair.
///
/// Every 7th pixel uses white on white, where the blend's floor truncation
/// overshoots 255 and the clamp decides the result; the rest use LCG colours.
fn composite_inputs() -> (Vec<u8>, Vec<u8>) {
    let mut next = colours(0xdead_beef);
    let mut src = Vec::with_capacity(PAIRS * 4);
    let mut dst = Vec::with_capacity(PAIRS * 4);
    for (a_src, a_dst) in (0..=255u8).flat_map(|a| (0..=255u8).map(move |b| (a, b))) {
        let (s, d) = if src.len() % 28 == 0 {
            ([255; 3], [255; 3])
        } else {
            ([next(), next(), next()], [next(), next(), next()])
        };
        src.extend_from_slice(&[s[0], s[1], s[2], a_src]);
        dst.extend_from_slice(&[d[0], d[1], d[2], a_dst]);
    }
    (src, dst)
}

/// Pixels and mask covering every (alpha, mask) pair.
fn soft_mask_inputs() -> (Vec<u8>, Vec<u8>) {
    let mut next = colours(0x0bad_cafe);
    let mut pixels = Vec::with_capacity(PAIRS * 4);
    let mut mask = Vec::with_capacity(PAIRS);
    for (alpha, m) in (0..=255u8).flat_map(|a| (0..=255u8).map(move |b| (a, b))) {
        pixels.extend_from_slice(&[next(), next(), next(), alpha]);
        mask.push(m);
    }
    (pixels, mask)
}

fn n_pixels(len: usize) -> u32 {
    u32::try_from(len).expect("pixel count fits u32")
}

fn upload<B: GpuBackend>(backend: &B, bytes: &[u8]) -> B::DeviceBuffer {
    let buf = backend.alloc_device(bytes.len()).expect("alloc_device");
    backend.upload(&buf, bytes).expect("upload");
    buf
}

fn run_page<B: GpuBackend>(backend: &B, record: impl FnOnce(&B)) {
    backend.begin_page().expect("begin_page");
    record(backend);
    let fence = backend.submit_page().expect("submit_page");
    backend.wait_page(fence).expect("wait_page");
}

fn download<B: GpuBackend>(backend: &B, buf: &B::DeviceBuffer, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    backend.download(buf, &mut out).expect("download");
    out
}

/// Index of the first differing pixel, with both values, for the message.
fn first_mismatch<'a>(device: &'a [u8], cpu: &'a [u8]) -> Option<(usize, &'a [u8], &'a [u8])> {
    device
        .chunks(4)
        .zip(cpu.chunks(4))
        .position(|(d, c)| d != c)
        .map(|i| (i, &device[i * 4..i * 4 + 4], &cpu[i * 4..i * 4 + 4]))
}

fn composite_matches_cpu<B: GpuBackend>(backend: &B) {
    let (src, dst) = composite_inputs();
    let d_src = upload(backend, &src);
    let d_dst = upload(backend, &dst);
    run_page(backend, |b| {
        b.record_composite(CompositeParams {
            src: &d_src,
            dst: &d_dst,
            n_pixels: n_pixels(PAIRS),
        })
        .expect("record_composite");
    });
    let device = download(backend, &d_dst, dst.len());

    let mut cpu = dst;
    rasterrocket_gpu::composite_rgba8_cpu(&src, &mut cpu);
    if let Some((i, d, c)) = first_mismatch(&device, &cpu) {
        panic!(
            "composite pixel {i} (a_src {}, a_dst {}): device {d:?}, CPU {c:?}",
            i / 256,
            i % 256
        );
    }
}

fn soft_mask_matches_cpu<B: GpuBackend>(backend: &B) {
    let (pixels, mask) = soft_mask_inputs();
    let d_pixels = upload(backend, &pixels);
    let d_mask = upload(backend, &mask);
    run_page(backend, |b| {
        b.record_apply_soft_mask(SoftMaskParams {
            pixels: &d_pixels,
            mask: &d_mask,
            n_pixels: n_pixels(PAIRS),
        })
        .expect("record_apply_soft_mask");
    });
    let device = download(backend, &d_pixels, pixels.len());

    let mut cpu = pixels;
    rasterrocket_gpu::apply_soft_mask_cpu(&mut cpu, &mask);
    if let Some((i, d, c)) = first_mismatch(&device, &cpu) {
        panic!(
            "soft mask pixel {i} (alpha {}, mask {}): device {d:?}, CPU {c:?}",
            i / 256,
            i % 256
        );
    }
}

#[cfg(feature = "gpu-validation")]
mod cuda {
    use rasterrocket_gpu::backend::cuda::CudaBackend;

    fn backend() -> CudaBackend {
        CudaBackend::new().expect("CudaBackend::new: no CUDA device or driver")
    }

    #[test]
    fn composite_matches_cpu_for_every_alpha_pair() {
        super::composite_matches_cpu(&backend());
    }

    #[test]
    fn soft_mask_matches_cpu_for_every_alpha_and_mask() {
        super::soft_mask_matches_cpu(&backend());
    }
}

#[cfg(feature = "vulkan")]
mod vulkan {
    use rasterrocket_gpu::backend::vulkan::VulkanBackend;

    fn backend() -> VulkanBackend {
        VulkanBackend::new().expect("VulkanBackend::new: no Vulkan device")
    }

    #[test]
    fn composite_matches_cpu_for_every_alpha_pair() {
        super::composite_matches_cpu(&backend());
    }

    #[test]
    fn soft_mask_matches_cpu_for_every_alpha_and_mask() {
        super::soft_mask_matches_cpu(&backend());
    }
}
