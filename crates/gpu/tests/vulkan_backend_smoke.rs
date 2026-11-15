//! Smoke test: `VulkanBackend` implements `GpuBackend` on whatever ICD
//! the system has.  Skipped if the `vulkan` feature isn't built.
//!
//! Run with:
//!   `cargo test -p rasterrocket-gpu --features vulkan --test vulkan_backend_smoke`
//!
//! This test only requires *any* working Vulkan ICD — Mesa lavapipe (the
//! CPU software ICD) counts.  CI without a discrete GPU can run it via
//! `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json`.
//!
//! ## Backend lifecycle
//!
//! Most tests use [`shared_backend`], which returns the per-process
//! `Arc<VulkanBackend>` (lazy-initialised on first call) **paired with a
//! mutex guard** that serialises against any other shared-backend test.
//! The serialisation is required because:
//!   - `PageRecorder`'s state machine forbids two `begin_page` calls
//!     without an intervening `submit_page`; the recorder mutex would
//!     reject the racing call with `InvalidRecorderState` instead of
//!     interleaving the work, so parallel tests would flake.
//!   - The descriptor-pool exhaustion test allocates the entire per-page
//!     budget; a parallel `begin_page` would see a full pool and fail.
//!
//! Sharing the backend matches the *production* lifecycle (the renderer
//! holds one `VulkanBackend` for the whole process), so subtle leaks or
//! stale-state bugs that per-test reset would mask now surface here.
//!
//! Tests that need a backend of their own build it: cold init
//! ([`vulkan_backend_initialises`]), concurrent creation and teardown, a
//! page left broken by a panic, and the integrated GPU
//! ([`integrated_backend`]).

#![cfg(feature = "vulkan")]

mod common;

use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use rasterrocket_gpu::backend::vulkan::VulkanBackend;
use rasterrocket_gpu::backend::{BackendError, GpuBackend};

/// Per-process `VulkanBackend`, lazy-initialised on first access.
///
/// Created via `OnceLock` so cold init (`vkCreateInstance` /
/// `vkCreateDevice` / pipeline-cache build) runs once for the whole test
/// binary instead of once per test.
static SHARED: OnceLock<Arc<VulkanBackend>> = OnceLock::new();

/// Mutex guarding shared-backend tests against concurrent execution.
///
/// Cargo's test runner is parallel by default; this lock makes
/// shared-backend tests sequential among themselves while still letting
/// them run in parallel with any test (e.g. [`vulkan_backend_initialises`])
/// that doesn't take the lock.
static SERIAL: Mutex<()> = Mutex::new(());

/// Acquire the shared `VulkanBackend` together with a serialisation
/// guard. Hold the returned tuple's second value (`_serial`) until the
/// test ends — dropping it releases the next shared-backend test.
fn shared_backend() -> (Arc<VulkanBackend>, MutexGuard<'static, ()>) {
    // Lock first, *then* lazily initialise: the OnceLock initialiser may
    // perform Vulkan FFI under the hood, and we want the same
    // serialisation discipline to cover initialisation as covers normal
    // use. Mutex poisoning would only happen if a previous test panicked
    // mid-critical-section; rethrow by panicking — the test infra will
    // report it on the next test that runs.
    let serial = SERIAL.lock().expect("SERIAL poisoned by a prior panic");
    let backend = SHARED
        .get_or_init(|| Arc::new(crate::common::vulkan_backend().expect("VulkanBackend::new")))
        .clone();
    (backend, serial)
}

#[test]
fn vulkan_backend_initialises() {
    // Intentionally NOT shared_backend(): this test verifies cold init,
    // so it must mint its own VulkanBackend each run.
    let backend =
        VulkanBackend::new().expect("VulkanBackend::new failed — is a Vulkan ICD available?");
    let budget = backend
        .detect_vram_budget()
        .expect("detect_vram_budget failed — driver query returned an error");
    // total_bytes can legitimately be 0 on a software ICD that exposes
    // no DEVICE_LOCAL heap (lavapipe falls back to a HOST_VISIBLE heap).
    // Just check the invariant holds.
    assert!(
        budget.usable_bytes <= budget.total_bytes,
        "VramBudget invariant violated: usable {} > total {}",
        budget.usable_bytes,
        budget.total_bytes
    );
}

#[test]
fn vulkan_backend_alloc_free_round_trip() {
    let (backend, _serial) = shared_backend();
    let buf = backend.alloc_device(4096).expect("alloc_device(4096)");
    assert_eq!(buf.size(), 4096);
    backend.free_device(buf);
}

#[test]
fn vulkan_backend_alloc_zero_size_rejected() {
    let (backend, _serial) = shared_backend();
    let err = backend
        .alloc_device(0)
        .expect_err("alloc_device(0) must be rejected");
    assert!(
        matches!(
            err,
            BackendError::ZeroSizeAlloc {
                what: "alloc_device"
            }
        ),
        "expected ZeroSizeAlloc {{ what: \"alloc_device\" }}, got: {err:?}",
    );
}

/// Allocate a zeroed device buffer, download into a sentinel-filled
/// host vec, assert every byte is zero, free.  `tag` is included in
/// failure messages so concurrent / parameterised callers can
/// pinpoint which invocation broke.
fn assert_zeroed_round_trip(backend: &VulkanBackend, size: usize, tag: &str) {
    let buf = backend
        .alloc_device_zeroed(size)
        .unwrap_or_else(|e| panic!("{tag}: alloc_device_zeroed({size}) failed: {e}"));
    let mut readback = vec![0xAAu8; size];
    backend
        .download(&buf, &mut readback)
        .unwrap_or_else(|e| panic!("{tag}: download failed: {e}"));
    if let Some(bad) = readback.iter().position(|&b| b != 0) {
        panic!(
            "{tag}: alloc_device_zeroed({size}) leaked non-zero byte 0x{:02x} at index {bad}",
            readback[bad]
        );
    }
    backend.free_device(buf);
}

#[test]
fn vulkan_backend_alloc_device_zeroed_returns_zero_bytes() {
    let (backend, _serial) = shared_backend();
    assert_zeroed_round_trip(&backend, 4096, "single");
}

#[test]
fn vulkan_backend_alloc_device_zeroed_concurrent() {
    // Regression: VkCommandPool + VkQueue both require external sync
    // (Vulkan "Threading Behavior" table).  Without the locks on
    // DeviceCtx + TransferContext, concurrent callers race in
    // run_one_shot — either driver crash or non-zero readbacks.
    use std::thread;

    const THREADS: usize = 4;
    const ITERS_PER_THREAD: usize = 8;

    let (backend, _serial) = shared_backend();
    let mut handles = Vec::with_capacity(THREADS);
    for thread_idx in 0..THREADS {
        let backend = Arc::clone(&backend);
        handles.push(thread::spawn(move || {
            for iter in 0..ITERS_PER_THREAD {
                assert_zeroed_round_trip(
                    &backend,
                    4096,
                    &format!("thread {thread_idx} iter {iter}"),
                );
            }
        }));
    }
    for h in handles {
        h.join().expect("worker thread panicked");
    }
}

#[test]
fn vulkan_backend_descriptor_pool_exhausts_at_max_sets() {
    // Recorder caps descriptor sets per page at MAX_DESC_SETS_PER_PAGE (64).
    // Verify the boundary: 64 record_icc_clut calls succeed, the 65th
    // returns Err with DescriptorPoolExhausted.  Pre-existing guard
    // had zero coverage; a refactor that reordered guard / allocate /
    // increment could silently regress.
    use rasterrocket_gpu::backend::params::IccClutParams;

    // Keep in sync with `MAX_DESC_SETS_PER_PAGE` in
    // crates/gpu/src/backend/vulkan/recorder.rs (private constant).
    // The let-else below destructures `BackendError::DescriptorPoolExhausted`
    // and asserts the production `max` field equals this — a production bump
    // without updating PER_PAGE_MAX fails loudly instead of silently
    // passing through the loop.
    const PER_PAGE_MAX: u32 = 64;
    const N_PIXELS: u32 = 4;
    const BUF_BYTES: usize = (N_PIXELS as usize) * 4;
    // A 2-node grid: 2⁴ nodes × 3 bytes.
    const CLUT_BYTES: usize = 16 * 3;

    let (backend, _serial) = shared_backend();
    let cmyk = backend.alloc_device_zeroed(BUF_BYTES).expect("alloc cmyk");
    let mut rgb = backend.alloc_device_zeroed(BUF_BYTES).expect("alloc rgb");
    let clut = backend.alloc_device_zeroed(CLUT_BYTES).expect("alloc clut");

    backend.begin_page().expect("begin_page");
    for i in 0..PER_PAGE_MAX {
        backend
            .record_icc_clut(IccClutParams {
                cmyk: &cmyk,
                rgb: &mut rgb,
                clut: &clut,
                n_pixels: N_PIXELS,
            })
            .unwrap_or_else(|e| panic!("record_icc_clut #{i} of {PER_PAGE_MAX} failed: {e}"));
    }
    let err = backend
        .record_icc_clut(IccClutParams {
            cmyk: &cmyk,
            rgb: &mut rgb,
            clut: &clut,
            n_pixels: N_PIXELS,
        })
        .expect_err("record_icc_clut past the cap must be rejected");
    let BackendError::DescriptorPoolExhausted { allocated, max } = err else {
        panic!("expected DescriptorPoolExhausted, got: {err:?}");
    };
    assert_eq!(
        allocated, PER_PAGE_MAX,
        "exhaustion should report the cap as the count already in flight",
    );
    assert_eq!(
        max, PER_PAGE_MAX,
        "test PER_PAGE_MAX is out of sync with production MAX_DESC_SETS_PER_PAGE",
    );

    // Recorder state stayed Recording (the rejected call exited before
    // touching the cmd buffer); finish the page cleanly so VulkanBackend's
    // Drop doesn't see in-flight work on device_wait_idle.
    let fence = backend.submit_page().expect("submit_page");
    backend.wait_page(fence).expect("wait_page");

    backend.free_device(cmyk);
    backend.free_device(rgb);
    backend.free_device(clut);
}

#[test]
fn vulkan_backend_record_zero_buffer_zeros_dirty_alloc() {
    // alloc_device returns a buffer with undefined contents; upload a
    // sentinel pattern, then record_zero_buffer inside a page submission,
    // then verify the readback is all-zero.
    //
    // This is the production-shape path the audit asked for: zero-fill
    // folded into the per-page command buffer instead of riding its own
    // submit + vkQueueWaitIdle in alloc_device_zeroed.
    const SIZE: usize = 4096;
    let (backend, _serial) = shared_backend();
    let buf = backend.alloc_device(SIZE).expect("alloc_device");
    let sentinel = vec![0xCDu8; SIZE];
    backend.upload(&buf, &sentinel).expect("upload sentinel");

    backend.begin_page().expect("begin_page");
    backend
        .record_zero_buffer(&buf, SIZE)
        .expect("record_zero_buffer");
    let fence = backend.submit_page().expect("submit_page");
    backend.wait_page(fence).expect("wait_page");

    let mut readback = vec![0xAAu8; SIZE];
    backend.download(&buf, &mut readback).expect("download");
    if let Some(bad) = readback.iter().position(|&b| b != 0) {
        panic!(
            "record_zero_buffer left non-zero byte 0x{:02x} at index {bad}",
            readback[bad]
        );
    }
    backend.free_device(buf);
}

/// A zero-fill of a prefix leaves the rest of the buffer intact, and a
/// prefix longer than the buffer is rejected.
#[test]
fn vulkan_backend_record_zero_buffer_fills_only_the_prefix() {
    const SIZE: usize = 4096;
    let (backend, _serial) = shared_backend();
    let buf = backend.alloc_device(SIZE).expect("alloc_device");
    backend
        .upload(&buf, &[0xCD; SIZE])
        .expect("upload sentinel");
    backend
        .run_page(|b| b.record_zero_buffer(&buf, 1024))
        .expect("prefix fill");
    let mut readback = vec![0xAA; SIZE];
    backend.download(&buf, &mut readback).expect("download");
    assert!(
        readback[..1024].iter().all(|&b| b == 0),
        "the prefix is zero"
    );
    assert!(
        readback[1024..].iter().all(|&b| b == 0xCD),
        "the rest is intact"
    );
    let err = backend
        .run_page(|b| b.record_zero_buffer(&buf, SIZE + 4))
        .expect_err("a fill past the buffer is rejected");
    assert!(err.to_string().contains("exceeds"), "{err}");
    backend.free_device(buf);
}

#[test]
fn vulkan_backend_record_zero_buffer_rejects_outside_recording_state() {
    // record_zero_buffer is a state-machine op: legal only between
    // begin_page and submit_page.  Calling it from Idle must error,
    // not record into a stale command buffer.
    let (backend, _serial) = shared_backend();
    let buf = backend.alloc_device(64).expect("alloc_device");
    let err = backend
        .record_zero_buffer(&buf, 64)
        .expect_err("record_zero_buffer outside Recording must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("expected Recording"),
        "expected state-machine error, got: {msg}"
    );
    backend.free_device(buf);
}

#[test]
fn vulkan_backend_record_zero_buffer_rejects_unaligned_size() {
    // vkCmdFillBuffer requires 4-byte-aligned size; record_zero_buffer
    // surfaces this loudly via BackendError::UnalignedFill instead of
    // silently leaving 1–3 trailing bytes non-zero.
    let (backend, _serial) = shared_backend();
    let buf = backend.alloc_device(17).expect("alloc_device(17)");
    backend.begin_page().expect("begin_page");
    let err = backend
        .record_zero_buffer(&buf, 17)
        .expect_err("record_zero_buffer on 17-byte buf must be rejected");
    assert!(
        matches!(
            err,
            BackendError::UnalignedFill {
                size: 17,
                required_alignment: 4,
            }
        ),
        "expected UnalignedFill{{17,4}}, got: {err:?}"
    );
    // Finish the page cleanly so VulkanBackend::Drop doesn't see
    // in-flight work — the recorder stays in Recording state after a
    // rejected call (neither the state nor the size check transitions
    // anything).  submit then wait drains the empty cmd buffer.
    let fence = backend.submit_page().expect("submit_page");
    backend.wait_page(fence).expect("wait_page");
    backend.free_device(buf);
}

#[test]
fn vulkan_backend_alloc_device_zeroed_rejects_unaligned_size() {
    // vkCmdFillBuffer requires 4-byte-aligned size; alloc_device_zeroed
    // must surface this loudly rather than silently downgrading.
    let (backend, _serial) = shared_backend();
    let err = backend
        .alloc_device_zeroed(17)
        .expect_err("alloc_device_zeroed(17) must be rejected (size not multiple of 4)");
    assert!(
        matches!(
            err,
            BackendError::UnalignedFill {
                size: 17,
                required_alignment: 4
            }
        ),
        "expected UnalignedFill {{ size: 17, required_alignment: 4 }}, got: {err:?}",
    );
}

#[test]
fn vulkan_backend_host_buffer_round_trip() {
    let (backend, _serial) = shared_backend();
    let mut buf = backend
        .alloc_host_pinned(256)
        .expect("alloc_host_pinned(256)");
    assert_eq!(buf.size(), 256);
    let slice = buf.as_mut_slice();
    for (i, b) in slice.iter_mut().enumerate() {
        *b = u8::try_from(i & 0xff).expect("masked to 0..=255");
    }
    let read_back = buf.as_slice();
    for (i, &b) in read_back.iter().enumerate() {
        assert_eq!(
            b,
            u8::try_from(i & 0xff).expect("masked to 0..=255"),
            "byte {i} mismatch"
        );
    }
    backend.free_host_pinned(buf);
}

/// The first integrated GPU: the default device when it is one, else a
/// device matched by a driver name that only integrated parts carry.
///
/// `None` (the caller skips) only when no device matches either name; any
/// other failure, an ambiguous name included, fails the test, so that a
/// broken device selection does not pass as a box without an iGPU.
fn integrated_backend() -> Option<VulkanBackend> {
    let default = VulkanBackend::new().expect("VulkanBackend::new");
    if default.is_integrated() {
        return Some(default);
    }
    drop(default);
    ["radv", "intel"].iter().find_map(|name| {
        let selector = name.parse().expect("non-empty selector");
        match VulkanBackend::with_selector(Some(&selector)) {
            Ok(backend) => backend.is_integrated().then_some(backend),
            Err(BackendError::DeviceSelection(msg))
                if msg.contains("no device name contains it") =>
            {
                None
            }
            Err(e) => panic!("selecting the {name} device failed: {e}"),
        }
    })
}

/// On shared memory an upload writes the device buffer directly, so it must
/// first wait for every submitted page: a page that reads the buffer and is
/// still in flight sees the old bytes, the next page the new ones.
#[test]
fn shared_memory_upload_waits_for_the_page_reading_the_buffer() {
    const LEN: usize = 1 << 20;
    let Some(backend) = integrated_backend() else {
        eprintln!("skipping: no integrated GPU");
        return;
    };
    let buf = backend.alloc_device(LEN).expect("alloc_device");
    backend
        .upload(&buf, &vec![0x11; LEN])
        .expect("first upload");
    let mut first = backend.alloc_host_readback(LEN).expect("readback");
    let mut second = backend.alloc_host_readback(LEN).expect("readback");

    backend.begin_page().expect("begin_page");
    // SAFETY: `first` is read only after `wait_page` below.
    unsafe { backend.record_copy_to_host(&buf, &mut first, LEN) }.expect("record copy");
    let fence = backend.submit_page().expect("submit_page");
    let before = backend.submits();
    backend
        .upload(&buf, &vec![0x22; LEN])
        .expect("second upload");
    assert_eq!(
        backend.submits(),
        before,
        "an upload to shared memory submits nothing"
    );
    backend.wait_page(fence).expect("wait_page");
    assert!(
        first.as_slice().iter().all(|&b| b == 0x11),
        "the in-flight page saw the new bytes"
    );

    backend
        // SAFETY: `run_page` waits for the page before `second` is read.
        .run_page(|b| unsafe { b.record_copy_to_host(&buf, &mut second, LEN) })
        .expect("second page");
    assert!(
        second.as_slice().iter().all(|&b| b == 0x22),
        "the next page missed the upload"
    );
}

/// Backends created and dropped on several threads at once: the loader is
/// shared by the process, so one thread's teardown never unloads it under
/// another thread's instance creation.
#[test]
fn backends_are_created_and_dropped_on_several_threads() {
    std::thread::scope(|scope| {
        for _ in 0..4 {
            // The scope joins every thread, propagating a panic.
            let _ = scope.spawn(|| {
                for _ in 0..8 {
                    drop(VulkanBackend::new().expect("VulkanBackend::new"));
                }
            });
        }
    });
}

/// A copy recorded after a zero-fill in one page reads the zeros: the fill
/// runs in the `CLEAR` stage, which the copy's barrier must cover.
#[test]
fn a_copy_after_a_zero_fill_in_one_page_reads_zeros() {
    const LEN: usize = 1 << 16;
    let (backend, _serial) = shared_backend();
    let buf = backend.alloc_device(LEN).expect("alloc_device");
    backend
        .upload(&buf, &vec![0xA5; LEN])
        .expect("dirty upload");
    let mut out = backend.alloc_host_readback(LEN).expect("readback");
    backend
        .run_page(|b| {
            b.record_zero_buffer(&buf, LEN)?;
            // SAFETY: `run_page` waits for the page before `out` is read.
            unsafe { b.record_copy_to_host(&buf, &mut out, LEN) }
        })
        .expect("page");
    assert!(
        out.as_slice().iter().all(|&b| b == 0),
        "the copy read bytes the fill had not cleared"
    );
}

/// A panic inside a `run_page` closure abandons the page: the next page
/// starts from `Idle` instead of finding the recorder still `Recording`.
#[test]
fn a_panicking_page_leaves_the_backend_usable() {
    let backend = crate::common::vulkan_backend().expect("VulkanBackend::new");
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = backend.run_page(|_| -> Result<(), BackendError> { panic!("page closure") });
    }));
    assert!(panicked.is_err(), "the closure's panic propagates");
    backend
        .run_page(|_| Ok(()))
        .expect("a page after the panicking one runs");
}

/// An oversize upload or download is refused with the same error whether
/// the buffer is reached through staging (discrete) or in place (shared
/// memory).
#[test]
fn oversize_transfers_fail_alike_on_every_device() {
    fn errors(backend: &VulkanBackend) -> (String, String) {
        let buf = backend.alloc_device(4).expect("alloc_device");
        let up = backend.upload(&buf, &[0; 8]).expect_err("oversize upload");
        let down = backend
            .download(&buf, &mut [0; 8])
            .expect_err("oversize download");
        (up.to_string(), down.to_string())
    }
    let expected = (
        "upload: copy of 8 bytes exceeds the destination buffer (4 bytes)".to_owned(),
        "download: copy of 8 bytes exceeds the source buffer (4 bytes)".to_owned(),
    );
    {
        let (backend, _serial) = shared_backend();
        assert_eq!(errors(&backend), expected);
    }
    if let Some(backend) = integrated_backend() {
        assert_eq!(errors(&backend), expected);
    }
}
