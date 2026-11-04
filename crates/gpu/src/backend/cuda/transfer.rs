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

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DriverError, HostSlice, PinnedHostSlice};

use crate::GpuCtx;
use crate::backend::{Result, copy_len, recorded_copy_len};

use super::be;

/// Copy `src` into `dst` and return once the copy has completed.  Empty
/// `src` is a no-op, since cudarc rejects zero-byte transfers.
pub(super) fn upload(ctx: &Arc<GpuCtx>, dst: &CudaSlice<u8>, src: &[u8]) -> Result<()> {
    copy_len(src.len(), dst.len() as u64, "upload", "destination").map(drop)?;
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
    copy_len(dst.len(), src.len() as u64, "download", "source").map(drop)?;
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

/// Queue a copy of `src[..len]` into the start of `dst` on the calling
/// thread's stream without waiting for it.  A zero `len` is a no-op.
///
/// The copy reads `src` after this call returns.  `src`'s event is
/// recorded on the stream after the copy, so [`PinnedHostSlice::as_mut_slice`]
/// and the drop of `src` wait for the copy instead of racing it.
pub(super) fn queue_htod(
    ctx: &Arc<GpuCtx>,
    src: &PinnedHostSlice<u8>,
    len: usize,
    dst: &CudaSlice<u8>,
) -> Result<()> {
    if recorded_copy_len(
        len,
        src.len() as u64,
        dst.len() as u64,
        "record_copy_to_device",
    )?
    .is_none()
    {
        return Ok(());
    }
    let stream = ctx.stream();
    let (dptr, _dst_sync) = dst.device_ptr(&stream);
    // SAFETY: the slice is used only by the copy queued on `stream` below,
    // and its guard, dropped after the copy is queued, records `src`'s event
    // behind the copy.
    let (host, _src_sync) = unsafe { src.stream_synced_slice(&stream) };
    // SAFETY: `dptr` addresses `dst` on `stream`; `len` fits both buffers
    // (checked above); `host` stays allocated and unwritten until the event
    // recorded behind the copy has completed.
    unsafe { cudarc::driver::result::memcpy_htod_async(dptr, &host[..len], stream.cu_stream()) }
        .map_err(be)
}

/// Queue a copy of the first `len` bytes of `src` into `dst` on the calling
/// thread's stream without waiting for it.  A zero `len` is a no-op.
///
/// The copy writes `dst` after this call returns.  `dst`'s event is
/// recorded on the stream after the copy, so [`PinnedHostSlice::as_slice`],
/// [`PinnedHostSlice::as_mut_slice`] and the drop of `dst` wait for the copy
/// instead of racing it.
pub(super) fn queue_dtoh(
    ctx: &Arc<GpuCtx>,
    src: &CudaSlice<u8>,
    dst: &mut PinnedHostSlice<u8>,
    len: usize,
) -> Result<()> {
    if recorded_copy_len(
        len,
        src.len() as u64,
        dst.len() as u64,
        "record_copy_to_host",
    )?
    .is_none()
    {
        return Ok(());
    }
    let stream = ctx.stream();
    let (dptr, _src_sync) = src.device_ptr(&stream);
    // SAFETY: as in `queue_htod`.
    let (host, _dst_sync) = unsafe { dst.stream_synced_mut_slice(&stream) };
    // SAFETY: `dptr` addresses `src` on `stream`; `len` fits both buffers
    // (checked above); `host` is accessed again only once the event
    // recorded behind the copy has completed.
    unsafe { cudarc::driver::result::memcpy_dtoh_async(&mut host[..len], dptr, stream.cu_stream()) }
        .map_err(be)
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
        let host = pinned.as_mut_slice().expect("host view");
        host.fill(0x5A);
        let dst = b.alloc_device(LEN).expect("alloc");
        b.upload(&dst, host).expect("upload");
        host.fill(0xC3);
        let mut readback = vec![0u8; LEN];
        b.download(&dst, &mut readback).expect("download");
        assert!(readback.iter().all(|&v| v == 0x5A));
    }

    /// A host view taken while a recorded copy into or out of the buffer
    /// is in flight waits for that copy.
    #[test]
    fn host_access_waits_for_a_recorded_copy() {
        const LEN: usize = 64 << 20;
        let Some(b) = try_backend() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let device = b.alloc_device(LEN).expect("alloc");
        b.upload(&device, &vec![0x5A; LEN]).expect("upload");
        let mut readback = b.alloc_host_readback(LEN).expect("readback");
        b.begin_page().expect("begin_page");
        // SAFETY: the buffer's event orders the host view below after the
        // copy; the page is waited on before either buffer is dropped.
        unsafe { b.record_copy_to_host(&device, &mut readback, LEN) }.expect("record");
        let seen = b.host_bytes(&readback).expect("host view");
        assert!(seen.iter().all(|&v| v == 0x5A), "the view raced the copy");
        let fence = b.submit_page().expect("submit_page");
        b.wait_page(fence).expect("wait_page");

        let mut staged = b.alloc_host_pinned(LEN).expect("pinned");
        b.host_bytes_mut(&mut staged).expect("host view").fill(0xC3);
        b.begin_page().expect("begin_page");
        // SAFETY: as above.
        unsafe { b.record_copy_to_device(&staged, &device, LEN) }.expect("record");
        b.host_bytes_mut(&mut staged).expect("host view").fill(0x11);
        let fence = b.submit_page().expect("submit_page");
        b.wait_page(fence).expect("wait_page");
        let mut out = vec![0u8; LEN];
        b.download(&device, &mut out).expect("download");
        assert!(
            out.iter().all(|&v| v == 0xC3),
            "a host write raced the copy"
        );
    }
}
