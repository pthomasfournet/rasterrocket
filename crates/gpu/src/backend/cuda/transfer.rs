//! CUDA host↔device transfer helpers.
//!
//! `upload` / `download_async` are implemented here so the trait
//! impl in `mod.rs` stays a thin dispatch table. Both queue on the
//! calling thread's stream alongside its compute work, so a later
//! `record_*` kernel launch from the same thread observes the transfer
//! without an explicit barrier.
//!
//! Drops down to cudarc's raw `result::memcpy_*_async` entry points
//! rather than the safe `CudaStream::memcpy_htod` wrapper because the
//! safe wrapper takes `&mut CudaSlice<u8>` (Rust borrow semantics) but
//! our trait API gives us `&Self::DeviceBuffer` (shared reference,
//! stream-ordered exclusivity). Matches the in-tree pattern of
//! `PageRecorder::record_zero_buffer`, which does the same drop-down
//! for `cuMemsetD8Async`.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, DevicePtr};

use crate::GpuCtx;
use crate::backend::{BackendError, DownloadInner, Result};

use super::be;
use super::page_recorder::{PageFence, PageRecorder};

/// Verify `src_len ≤ dst_capacity` and emit a uniform error message
/// with the operation label for diagnostic clarity.
///
/// Returns the validated length as `usize` for downstream use. The
/// length-mismatch case is the most common cause of an unsigned-int
/// underflow in transfer code, so this stays a hard check rather than
/// a debug-assert.
fn check_transfer_len(src_len: usize, dst_capacity: usize, op_label: &'static str) -> Result<()> {
    if src_len > dst_capacity {
        return Err(BackendError::msg(format!(
            "{op_label}: source length ({src_len} bytes) exceeds destination \
             capacity ({dst_capacity} bytes)"
        )));
    }
    Ok(())
}

/// Copy `src` into `dst` on the calling thread's stream and return
/// once the copy has completed.
///
/// `cuMemcpyHtoDAsync` reads a page-locked source by DMA after the call
/// returns, and the driver does not guarantee that a pageable source is
/// staged before return either. Waiting on an event recorded after the
/// copy is therefore what ends the read of `src` within its borrow.
/// The event also orders every earlier item on the stream, so the call
/// returns after previously queued kernels as well.
///
/// Empty `src` is a no-op, since cudarc rejects zero-byte transfers.
pub(super) fn upload(ctx: &Arc<GpuCtx>, dst: &CudaSlice<u8>, src: &[u8]) -> Result<()> {
    check_transfer_len(src.len(), dst.len(), "upload")?;
    if src.is_empty() {
        return Ok(());
    }
    let stream = ctx.stream();
    let (dptr, _sync) = dst.device_ptr(&stream);
    // SAFETY: `dptr` addresses `dst` on `stream` and `src.len()` does
    // not exceed `dst.len()` (checked above). The fence wait below
    // returns only after the copy has finished reading `src`, which is
    // borrowed for the whole call.
    unsafe {
        cudarc::driver::result::memcpy_htod_async(dptr, src, stream.cu_stream()).map_err(be)?;
    }
    PageRecorder::record_fence(ctx)?.synchronize()
}

/// Initiate an asynchronous device→host download on the calling
/// thread's stream. Returns a download inner that holds the fence; the
/// caller wraps it in a `DownloadHandle` and either waits on it via
/// `wait_download` or lets it block-and-discard on Drop.
///
/// The dtoh copy writes directly into `dst` (no staging buffer); the
/// caller-provided `&mut [u8]` lifetime is held by the
/// `DownloadHandle` for the duration of the in-flight DMA.
pub(super) fn download_async(
    ctx: &Arc<GpuCtx>,
    src: &CudaSlice<u8>,
    dst: &mut [u8],
) -> Result<(CudaDownloadInner, PageFence)> {
    check_transfer_len(dst.len(), src.len(), "download_async")?;
    // Empty `dst` path: skip the cudarc call (which would reject zero
    // bytes), but still record a fence so the trait's "callable, then
    // wait_download or wait on fence()" contract holds uniformly.
    if !dst.is_empty() {
        let stream = ctx.stream();
        let (dptr, _sync) = src.device_ptr(&stream);
        // Safety: `dptr` from cudarc for `src` on `stream`; `dst` is
        // borrowed by the returned `DownloadHandle` for the in-flight
        // DMA's lifetime via the trait's PhantomData; cudarc's async
        // dtoh takes the host pointer at call time and the driver
        // retains the address (not the slice metadata) until the
        // stream catches up.
        unsafe {
            cudarc::driver::result::memcpy_dtoh_async(dst, dptr, stream.cu_stream()).map_err(be)?;
        }
    }
    let fence = PageRecorder::record_fence(ctx)?;
    Ok((CudaDownloadInner::new(fence.clone()), fence))
}

/// Backend-internal download-completion state for CUDA.
///
/// Wraps the post-copy `PageFence`. `finish()` blocks on the fence;
/// after a successful wait the inner becomes a no-op so `Drop` (which
/// also calls `finish()`) doesn't re-block.
pub(super) struct CudaDownloadInner {
    /// `None` after `finish()` returns Ok — disarms the Drop path.
    fence: Option<PageFence>,
}

impl CudaDownloadInner {
    const fn new(fence: PageFence) -> Self {
        Self { fence: Some(fence) }
    }
}

impl DownloadInner for CudaDownloadInner {
    fn finish(&mut self) -> Result<()> {
        if let Some(fence) = self.fence.take() {
            fence.synchronize()?;
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "gpu-validation"))]
mod tests {
    use crate::backend::cuda::CudaBackend;
    use crate::backend::{DownloadHandle, GpuBackend};

    fn try_backend() -> Option<CudaBackend> {
        CudaBackend::new().ok()
    }

    #[test]
    fn upload_then_download_round_trip() {
        let Some(b) = try_backend() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let payload: Vec<u8> = (0u8..=255).collect();
        let dst = b.alloc_device(payload.len()).expect("alloc");
        b.upload(&dst, &payload).expect("upload");

        let mut readback = vec![0u8; payload.len()];
        let handle = b.download_async(&dst, &mut readback).expect("download");
        b.wait_download(handle).expect("wait download");
        assert_eq!(readback, payload);
    }

    #[test]
    fn upload_rejects_oversize_src() {
        let Some(b) = try_backend() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let dst = b.alloc_device(4).expect("alloc");
        let err = b.upload(&dst, &[0u8; 8]).expect_err("oversize must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("upload") && msg.contains("exceeds"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn download_rejects_oversize_dst() {
        let Some(b) = try_backend() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let src = b.alloc_device(4).expect("alloc");
        let mut dst = vec![0u8; 8];
        // DownloadHandle is not Debug; expect_err would need it. Match
        // manually instead.
        match b.download_async(&src, &mut dst) {
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("download_async") && msg.contains("exceeds"),
                    "unexpected error: {msg}"
                );
            }
            Ok(_) => panic!("oversize download must fail"),
        }
    }

    #[test]
    fn empty_upload_is_a_no_op() {
        let Some(b) = try_backend() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        // A zero-byte device buffer cannot be allocated; the empty source
        // is checked against a non-empty destination instead.
        let dst = b.alloc_device(4).expect("alloc");
        b.upload(&dst, &[]).expect("empty upload ok");
    }

    /// A page-locked source is read by DMA after `cuMemcpyHtoDAsync`
    /// returns; `upload` must not return before that read has finished.
    #[test]
    fn pinned_source_is_reusable_when_upload_returns() {
        const LEN: usize = 16 << 20;
        let Some(b) = try_backend() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let mut pinned = b.alloc_host_pinned(LEN).expect("pinned");
        let host = pinned.as_mut_slice().expect("pinned slice");
        host.fill(0x5A);
        let dst = b.alloc_device(LEN).expect("alloc");
        b.upload(&dst, host).expect("upload");
        host.fill(0xC3);
        let mut readback = vec![0u8; LEN];
        let handle = b.download_async(&dst, &mut readback).expect("download");
        b.wait_download(handle).expect("wait download");
        assert!(readback.iter().all(|&v| v == 0x5A));
    }

    #[test]
    fn drop_without_wait_does_not_panic() {
        let Some(b) = try_backend() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let dst = b.alloc_device(64).expect("alloc");
        b.upload(&dst, &[0xAB; 64]).expect("setup");
        let mut readback = vec![0u8; 64];
        let handle: DownloadHandle<'_, CudaBackend> =
            b.download_async(&dst, &mut readback).expect("download");
        // Drop without calling wait_download — CudaDownloadInner::finish
        // is idempotent (fence.take() arms it for exactly one
        // synchronize()), so Drop's block-and-discard runs cleanly.
        drop(handle);
    }
}
