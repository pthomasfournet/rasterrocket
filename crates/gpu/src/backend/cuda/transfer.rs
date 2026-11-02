//! CUDA host↔device transfer helpers.
//!
//! `upload` / `download` are implemented here so the trait impl in
//! `mod.rs` stays a thin dispatch table.  Both queue the copy on the
//! calling thread's stream and synchronise the stream before returning:
//! an asynchronous copy reads (or writes) a page-locked host buffer by
//! DMA after the call that queued it returns, and the driver does not
//! guarantee that a pageable buffer is staged before return either, so
//! the wait is what keeps every access to the host slice within its
//! borrow.  The wait also covers the work queued before the copy.
//!
//! Drops down to cudarc's raw `result::memcpy_*_async` entry points
//! rather than the safe `CudaStream::memcpy_htod` wrapper because the
//! safe wrapper takes `&mut CudaSlice<u8>` (Rust borrow semantics) but
//! our trait API gives us `&Self::DeviceBuffer` (shared reference,
//! stream-ordered exclusivity). Matches the in-tree pattern of
//! `PageRecorder::record_zero_buffer`, which does the same drop-down
//! for `cuMemsetD8Async`.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DriverError};

use crate::GpuCtx;
use crate::backend::{BackendError, Result};

use super::be;

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

/// Copy `src` into `dst` and return once the copy has completed.  Empty
/// `src` is a no-op, since cudarc rejects zero-byte transfers.
pub(super) fn upload(ctx: &Arc<GpuCtx>, dst: &CudaSlice<u8>, src: &[u8]) -> Result<()> {
    check_transfer_len(src.len(), dst.len(), "upload")?;
    if src.is_empty() {
        return Ok(());
    }
    let stream = ctx.stream();
    let (dptr, _sync) = dst.device_ptr(&stream);
    // SAFETY: `dptr` addresses `dst` on `stream` and `src.len()` does
    // not exceed `dst.len()` (checked above).  `src` is borrowed for the
    // whole call, and `finish_copy` returns only once the stream, and
    // with it the copy, has completed.
    let queued =
        unsafe { cudarc::driver::result::memcpy_htod_async(dptr, src, stream.cu_stream()) };
    finish_copy(&stream, queued)
}

/// Copy the first `dst.len()` bytes of `src` into `dst` and return once
/// the copy has completed.  Empty `dst` is a no-op.
pub(super) fn download(ctx: &Arc<GpuCtx>, src: &CudaSlice<u8>, dst: &mut [u8]) -> Result<()> {
    check_transfer_len(dst.len(), src.len(), "download")?;
    if dst.is_empty() {
        return Ok(());
    }
    let stream = ctx.stream();
    let (dptr, _sync) = src.device_ptr(&stream);
    // SAFETY: `dptr` addresses `src` on `stream` and `dst.len()` does
    // not exceed `src.len()` (checked above).  `dst` is borrowed for the
    // whole call, and `finish_copy` returns only once the copy has
    // completed.
    let queued =
        unsafe { cudarc::driver::result::memcpy_dtoh_async(dst, dptr, stream.cu_stream()) };
    finish_copy(&stream, queued)
}

/// Wait for `stream`, on which a copy was queued with result `queued`.
///
/// The stream is synchronised even when queueing reported an error, so
/// that no copy the driver did accept outlives the host borrow; the
/// queueing error takes precedence over a synchronisation error.
fn finish_copy(stream: &CudaStream, queued: std::result::Result<(), DriverError>) -> Result<()> {
    let waited = stream.synchronize();
    queued.map_err(be)?;
    waited.map_err(be)
}

#[cfg(all(test, feature = "gpu-validation"))]
mod tests {
    use crate::backend::GpuBackend;
    use crate::backend::cuda::CudaBackend;

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
        b.download(&dst, &mut readback).expect("download");
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
            msg.starts_with("upload:") && msg.contains("exceeds"),
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
        let msg = b
            .download(&src, &mut dst)
            .expect_err("oversize must fail")
            .to_string();
        assert!(
            msg.starts_with("download:") && msg.contains("exceeds"),
            "{msg}"
        );
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
        b.download(&dst, &mut readback).expect("download");
        assert!(readback.iter().all(|&v| v == 0x5A));
    }
}
