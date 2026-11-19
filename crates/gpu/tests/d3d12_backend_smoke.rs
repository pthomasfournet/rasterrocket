//! Smoke test: `D3d12Backend` implements `GpuBackend` on the adapter
//! `PDF_RASTER_D3D12_DEVICE` names (`warp`, a device selector, or the
//! automatic choice when unset).
//!
//! Run with:
//!   `cargo test -p rasterrocket-gpu --features d3d12 --test d3d12_backend_smoke`
//!
//! `PDF_RASTER_REQUIRE_D3D12=1` turns a skip for want of an adapter into
//! a failure; `PDF_RASTER_D3D12_DEBUG=1` enables the debug layer, whose
//! messages go to the debugger output.
//!
//! Every test but the one that makes a backend of its own shares one
//! backend and runs under [`SERIAL`]: the backend has one recording state,
//! and the descriptor test fills a page's whole descriptor budget.

#![cfg(all(feature = "d3d12", windows))]

#[path = "common/d3d12.rs"]
mod common_d3d12;

use std::sync::{Mutex, MutexGuard, OnceLock};

use rasterrocket_gpu::backend::d3d12::D3d12Backend;
use rasterrocket_gpu::backend::params::IccClutParams;
use rasterrocket_gpu::backend::{BackendError, GpuBackend};

/// The backend every shared test uses; `None` when no adapter opens.
static SHARED: OnceLock<Option<D3d12Backend>> = OnceLock::new();

/// Serialises the shared tests.
static SERIAL: Mutex<()> = Mutex::new(());

/// The shared backend and the guard that serialises its tests, or `None`
/// (the test skips) when no adapter opens.
fn shared() -> Option<(&'static D3d12Backend, MutexGuard<'static, ()>)> {
    let serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    SHARED
        .get_or_init(common_d3d12::test_backend)
        .as_ref()
        .map(|b| (b, serial))
}

/// `n` bytes of a pattern that differs between neighbouring bytes and
/// between buffers of different `seed`.
fn pattern(n: usize, seed: u8) -> Vec<u8> {
    (0..n)
        .map(|i| u8::try_from((i * 7 + usize::from(seed)) % 251).expect("below 251"))
        .collect()
}

#[test]
fn d3d12_backend_is_named_and_reports_a_budget() {
    let Some(backend) = common_d3d12::test_backend() else {
        return;
    };
    assert_eq!(D3d12Backend::NAME, "d3d12");
    let budget = backend.detect_vram_budget().expect("detect_vram_budget");
    assert!(budget.usable_bytes <= budget.total_bytes, "{budget:?}");
    assert!(budget.total_bytes > 0, "{budget:?}");
    assert!(!backend.device_memory_is_host_mapped());
}

#[test]
fn upload_download_round_trips() {
    let Some((backend, _serial)) = shared() else {
        return;
    };
    for len in [1, 3, 4, (1 << 20) + 3] {
        let data = pattern(len, 1);
        let mut buf = backend.alloc_device(len).expect("alloc_device");
        assert_eq!(backend.device_buffer_len(&buf), len);
        backend.upload(&mut buf, &data).expect("upload");
        let mut out = vec![0xEE; len];
        backend.download(&buf, &mut out).expect("download");
        assert!(out == data, "round trip of {len} bytes differs");
        backend.free_device(buf);
    }
}

#[test]
fn alloc_device_zeroed_reads_zeros() {
    let Some((backend, _serial)) = shared() else {
        return;
    };
    let buf = backend
        .alloc_device_zeroed(1 << 16)
        .expect("alloc_device_zeroed");
    let mut out = vec![0xAA; 1 << 16];
    backend.download(&buf, &mut out).expect("download");
    assert!(
        out.iter().all(|&b| b == 0),
        "a zeroed buffer reads non-zero"
    );
    backend.free_device(buf);
    let err = backend
        .alloc_device(0)
        .expect_err("a zero-size allocation is refused");
    assert!(
        matches!(
            err,
            BackendError::ZeroSizeAlloc {
                what: "alloc_device"
            }
        ),
        "{err:?}"
    );
}

#[test]
fn record_zero_buffer_zeroes_a_written_buffer_in_a_page() {
    const LEN: usize = 4096;
    let Some((backend, _serial)) = shared() else {
        return;
    };
    let mut buf = backend.alloc_device(LEN).expect("alloc_device");
    backend
        .upload(&mut buf, &[0xCD; LEN])
        .expect("upload sentinel");
    backend
        .run_page(|b| b.record_zero_buffer(&mut buf, 1024))
        .expect("prefix fill");
    let mut out = vec![0xAA; LEN];
    backend.download(&buf, &mut out).expect("download");
    assert!(out[..1024].iter().all(|&b| b == 0), "the prefix is zero");
    assert!(out[1024..].iter().all(|&b| b == 0xCD), "the rest is intact");

    backend
        .run_page(|b| b.record_zero_buffer(&mut buf, LEN))
        .expect("whole fill");
    backend.download(&buf, &mut out).expect("download");
    assert!(out.iter().all(|&b| b == 0), "the whole buffer is zero");

    let err = backend
        .run_page(|b| b.record_zero_buffer(&mut buf, 6))
        .expect_err("a fill of 6 bytes is refused");
    assert!(
        matches!(
            err,
            BackendError::UnalignedFill {
                size: 6,
                required_alignment: 4
            }
        ),
        "{err:?}"
    );
    let err = backend
        .run_page(|b| b.record_zero_buffer(&mut buf, LEN + 4))
        .expect_err("a fill past the buffer is refused");
    assert!(err.to_string().contains("exceeds"), "{err}");
    backend.free_device(buf);
}

#[test]
fn host_to_device_to_host_copies_in_one_page() {
    const LEN: usize = 64 * 1024 + 3;
    let Some((backend, _serial)) = shared() else {
        return;
    };
    let data = pattern(LEN, 9);
    let mut src = backend.alloc_host_pinned(LEN).expect("alloc_host_pinned");
    backend
        .host_bytes_mut(&mut src)
        .expect("host_bytes_mut")
        .copy_from_slice(&data);
    let mut dev = backend.alloc_device(LEN).expect("alloc_device");
    let mut dst = backend
        .alloc_host_readback(LEN)
        .expect("alloc_host_readback");
    backend
        .run_page(|b| {
            // SAFETY: `run_page` returns once the page has completed;
            // `src` and `dst` are touched only after it.
            unsafe { b.record_copy_to_device(&src, &mut dev, LEN) }?;
            b.record_zero_buffer(&mut dev, 4)?;
            // SAFETY: as above.
            unsafe { b.record_copy_to_host(&dev, &mut dst, LEN) }
        })
        .expect("page");
    let out = backend.host_bytes(&dst).expect("host_bytes");
    assert_eq!(&out[..4], &[0; 4], "the fill after the copy is lost");
    assert!(out[4..] == data[4..], "the copies lost bytes");
    backend.free_device(dev);
    backend.free_host_pinned(src);
    backend.free_host_pinned(dst);
}

#[test]
fn a_failing_page_leaves_the_backend_usable() {
    let Some((backend, _serial)) = shared() else {
        return;
    };
    let mut buf = backend.alloc_device(16).expect("alloc_device");
    backend.upload(&mut buf, &[7; 16]).expect("upload");
    let err = backend
        .run_page(|b| {
            b.record_zero_buffer(&mut buf, 16)?;
            Err(BackendError::msg("the closure fails"))
        })
        .expect_err("the page fails");
    assert_eq!(err.to_string(), "the closure fails");
    let mut out = [0; 16];
    backend.download(&buf, &mut out).expect("download");
    assert_eq!(out, [7; 16], "the abandoned page ran");
    backend
        .run_page(|b| b.record_zero_buffer(&mut buf, 16))
        .expect("the next page runs");
    backend.download(&buf, &mut out).expect("download");
    assert_eq!(out, [0; 16]);
    backend.free_device(buf);
}

#[test]
fn the_65th_dispatch_of_a_page_exhausts_the_descriptors() {
    const PER_PAGE_MAX: u32 = 64;
    const N_PIXELS: u32 = 4;
    const CMYK_BYTES: usize = (N_PIXELS as usize) * 4;
    const RGB_BYTES: usize = (N_PIXELS as usize) * 4;
    // A grid of one cell along each input: 2⁴ nodes.
    const CLUT_BYTES: usize = (color::grid::CmykGrid::NODES + 2 * 16) * 4;
    let Some((backend, _serial)) = shared() else {
        return;
    };
    let cmyk = backend.alloc_device_zeroed(CMYK_BYTES).expect("cmyk");
    let mut rgb = backend.alloc_device_zeroed(RGB_BYTES).expect("rgb");
    let clut = backend.alloc_device_zeroed(CLUT_BYTES).expect("clut");
    let err = backend
        .run_page(|b| {
            for i in 0..=PER_PAGE_MAX {
                b.record_icc_clut(IccClutParams {
                    cmyk: &cmyk,
                    rgb: &mut rgb,
                    clut: &clut,
                    n_pixels: N_PIXELS,
                    wide: false,
                })
                .map_err(|e| {
                    assert_eq!(i, PER_PAGE_MAX, "dispatch {i} failed: {e}");
                    e
                })?;
            }
            Ok(())
        })
        .expect_err("the 65th dispatch is refused");
    let BackendError::DescriptorPoolExhausted { allocated, max } = err else {
        panic!("expected DescriptorPoolExhausted, got {err:?}");
    };
    assert_eq!((allocated, max), (PER_PAGE_MAX, PER_PAGE_MAX));
    backend
        .run_page(|b| {
            b.record_icc_clut(IccClutParams {
                cmyk: &cmyk,
                rgb: &mut rgb,
                clut: &clut,
                n_pixels: N_PIXELS,
                wide: false,
            })
        })
        .expect("the next page has its descriptors back");
    backend.free_device(cmyk);
    backend.free_device(rgb);
    backend.free_device(clut);
}

#[test]
fn free_device_after_wait_page_is_clean() {
    let Some((backend, _serial)) = shared() else {
        return;
    };
    let mut buf = backend.alloc_device(256).expect("alloc_device");
    backend.begin_page().expect("begin_page");
    backend.record_zero_buffer(&mut buf, 256).expect("fill");
    let fence = backend.submit_page().expect("submit_page");
    let again = fence.clone();
    backend.wait_page(fence).expect("wait_page");
    backend.free_device(buf);
    backend
        .run_page(|_| Ok(()))
        .expect("an empty page after the free");
    drop(again);
}

/// `wait_page` waits for the page in flight even when handed an earlier
/// page's fence, which has long been reached: the page's read-back is
/// complete when it returns.
#[test]
fn wait_page_with_a_stale_fence_waits_for_the_page_in_flight() {
    const LEN: usize = 64 << 20;
    let Some((backend, _serial)) = shared() else {
        return;
    };
    backend.begin_page().expect("begin_page");
    let stale = backend.submit_page().expect("submit_page");
    backend.wait_page(stale.clone()).expect("wait_page");

    let mut buf = backend.alloc_device(LEN).expect("alloc_device");
    backend.upload(&mut buf, &[0x5A; 16]).expect("upload");
    let mut out = backend.alloc_host_readback(LEN).expect("readback");
    backend
        .host_bytes_mut(&mut out)
        .expect("host_bytes_mut")
        .fill(0xEE);
    backend.begin_page().expect("begin_page");
    backend.record_zero_buffer(&mut buf, LEN).expect("fill");
    // SAFETY: `out` is read only after `wait_page` below.
    unsafe { backend.record_copy_to_host(&buf, &mut out, LEN) }.expect("copy");
    let own = backend.submit_page().expect("submit_page");
    backend
        .wait_page(stale)
        .expect("wait_page with an earlier fence");
    let bytes = backend.host_bytes(&out).expect("host_bytes");
    assert!(
        bytes.iter().all(|&b| b == 0),
        "the page had not completed when wait_page returned"
    );
    drop(own);
    backend.free_device(buf);
    backend.free_host_pinned(out);
}
