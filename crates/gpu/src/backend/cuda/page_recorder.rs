//! CUDA page recorder — buffers `record_*` calls until `submit_page`.
//!
//! All `record_*` methods route through the per-kernel
//! `launch_<name>_async` helpers in `lib_kernels::*`, which queue the
//! kernel onto the calling thread's stream ([`GpuCtx::stream`]) **without**
//! synchronising or downloading.  `submit_page` records a `CudaEvent` on
//! that stream; `wait_page` blocks the host on it.  Callers therefore stall
//! once per page rather than once per kernel.  A fence covers only the work
//! its recording thread queued, so a page's `record_*` calls and its
//! `submit_page` must come from one thread.
//!
//! ## Concurrency note
//!
//! cudarc records a write event on a kernel argument only when the
//! argument is a `&mut CudaSlice`; a `&CudaSlice` argument records a read.
//! Every buffer a kernel writes is therefore passed by `&mut`, through the
//! `&mut` fields of `params::*Params`, and every buffer it only reads by
//! `&`: a launch on another stream waits for the last recorded write of
//! each buffer it reads, and for the last recorded read and write of each
//! buffer it writes.

#![expect(
    clippy::needless_pass_by_value,
    reason = "record_* methods take params by value to mirror the GpuBackend trait shape; the trait could take &Params, but every backend would still want to destructure, so the by-value form is canonical"
)]

use std::collections::HashSet;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::ThreadId;

use cudarc::driver::{CudaEvent, CudaSlice, DevicePtrMut};

use super::be;
use crate::GpuCtx;
use crate::backend::{BackendError, Result, params};
#[cfg(feature = "cache")]
use crate::blit::BlitBbox;

pub(super) struct PageRecorder {
    pub(super) ctx: Arc<GpuCtx>,
    /// Threads between `begin_page` and `submit_page` (or
    /// `abandon_page`): a page is recorded on its thread's stream, so the
    /// recording state is per thread.
    recording: Mutex<HashSet<ThreadId>>,
}

/// A synchronisation token returned by [`PageRecorder::submit_page`].
///
/// Wraps a [`CudaEvent`] recorded on the submitting thread's stream.
/// `wait_page` calls `event.synchronize()`, which blocks the host until
/// every prior kernel queued on that stream has completed.
///
/// `Arc<CudaEvent>` so the trait's `PageFence: Clone` bound holds —
/// the cache stashes one clone per cached entry to gate
/// `free_device`, while the renderer holds another to wait on the
/// per-page submission.  CUDA events are pure synchronisation
/// primitives (no payload, just the recorded stream-position),
/// safe to share across threads.
#[derive(Debug, Clone)]
pub struct PageFence(Arc<CudaEvent>);

impl PageRecorder {
    pub(super) fn new(ctx: Arc<GpuCtx>) -> Self {
        Self {
            ctx,
            recording: Mutex::new(HashSet::new()),
        }
    }

    fn recording(&self) -> MutexGuard<'_, HashSet<ThreadId>> {
        self.recording
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Begin a page on the calling thread.  Recording queues onto the
    /// thread's stream, so only the thread's state changes.
    ///
    /// # Errors
    /// The thread is already recording a page.
    pub(super) fn begin_page(&self) -> Result<()> {
        if self.recording().insert(std::thread::current().id()) {
            Ok(())
        } else {
            Err(BackendError::msg(
                "begin_page called while this thread records a page; expected Idle",
            ))
        }
    }

    /// Fail unless the calling thread is recording a page; `op` names the
    /// rejected call.
    pub(super) fn check_recording(&self, op: &'static str) -> Result<()> {
        if self.recording().contains(&std::thread::current().id()) {
            Ok(())
        } else {
            Err(BackendError::msg(format!(
                "{op} called outside a page on this thread; expected Recording"
            )))
        }
    }

    /// Leave the calling thread's page, recorded or not.
    pub(super) fn end_page(&self) {
        let _ = self.recording().remove(&std::thread::current().id());
    }

    /// Record a stream event for the current page's queued work.
    ///
    /// All prior `record_*` launches on this stream are ordered before
    /// the event.  The host can then block on the event with
    /// `wait_page` instead of synchronising the whole stream.
    ///
    /// # Errors
    /// The thread records no page, or the event cannot be recorded; the
    /// page is left in either case.
    pub(super) fn submit_page(&self) -> Result<PageFence> {
        self.check_recording("submit_page")?;
        self.end_page();
        let event = self.ctx.stream().record_event(None).map_err(be)?;
        Ok(PageFence(Arc::new(event)))
    }

    /// Block until the work covered by `fence` has completed.
    #[expect(
        clippy::unused_self,
        reason = "self-arg kept for symmetry with the trait method and forward-compat with future per-recorder fence pools"
    )]
    pub(super) fn wait_page(&self, fence: PageFence) -> Result<()> {
        fence.0.synchronize().map_err(be)
    }

    /// Callers run `params.validate(&backend)` before this in the
    /// `CudaBackend::record_blit_image` wrapper.
    pub(super) fn record_blit_image(
        &self,
        p: params::BlitParams<'_, super::CudaBackend>,
    ) -> Result<()> {
        Self::record_blit_image_inner(&self.ctx, p)
    }

    #[cfg(feature = "cache")]
    fn record_blit_image_inner(
        ctx: &Arc<GpuCtx>,
        p: params::BlitParams<'_, super::CudaBackend>,
    ) -> Result<()> {
        let bbox = BlitBbox {
            x0: p.bbox[0],
            y0: p.bbox[1],
            x1: p.bbox[2],
            y1: p.bbox[3],
        };
        // Trait passes layout as u32; kernel takes i32. validate() has
        // already constrained the value to {0, 1}, so the cast is exact.
        // Keeping the explicit cast (rather than `as i32`) so a future
        // widening of valid layouts surfaces at this exact line.
        let layout_code =
            i32::try_from(p.src_layout).expect("validate() proved src_layout is 0 or 1, fits i32");

        ctx.launch_blit_image_async(
            p.src,
            (p.src_w, p.src_h),
            layout_code,
            p.dst,
            (p.dst_w, p.dst_h),
            bbox,
            p.cols,
            p.rows,
            p.filter,
        )
        .map_err(be)
    }

    #[cfg(not(feature = "cache"))]
    fn record_blit_image_inner(
        _ctx: &Arc<GpuCtx>,
        _p: params::BlitParams<'_, super::CudaBackend>,
    ) -> Result<()> {
        Err(crate::backend::BackendError::msg(
            "blit_image requires the gpu crate's `cache` feature",
        ))
    }

    /// `grid_n` is the value `IccClutParams::validate` derived from the
    /// CLUT length; the `CudaBackend::record_icc_clut` wrapper runs it.
    pub(super) fn record_icc_clut(
        &self,
        p: params::IccClutParams<'_, super::CudaBackend>,
        grid_n: u32,
    ) -> Result<()> {
        self.ctx
            .launch_icc_clut_async(p.cmyk, p.rgb, p.clut, grid_n, p.n_pixels)
            .map_err(be)
    }

    /// Record an async zero-fill of `buf[..len]` on the calling thread's stream via
    /// `cuMemsetD8Async`.  The fill waits for the buffer's recorded reads
    /// and writes and records a write, so later work on any stream observes
    /// zeros; on the same stream, stream order alone provides that.
    pub(super) fn record_zero_buffer(&self, buf: &mut CudaSlice<u8>, len: usize) -> Result<()> {
        crate::backend::copy_len(
            len,
            buf.num_bytes() as u64,
            "record_zero_buffer",
            "destination",
        )
        .map(drop)?;
        if len == 0 {
            return Ok(());
        }
        let stream = self.ctx.stream();
        let (dptr, _sync) = buf.device_ptr_mut(&stream);
        // Safety: dptr was returned by cudarc for `buf` on `stream`;
        // `len` is at most the buffer's size (checked above); the stream owns the
        // ordering against subsequent kernel launches.  Mirrors the
        // pattern in cudarc's safe `memset_zeros` which calls the same
        // unsafe entry point.
        unsafe { cudarc::driver::result::memset_d8_async(dptr, 0, len, stream.cu_stream()) }
            .map_err(be)?;
        Ok(())
    }
}

#[cfg(all(test, feature = "gpu-validation"))]
mod tests {
    use crate::backend::cuda::CudaBackend;
    use crate::backend::{BackendError, GpuBackend};

    #[test]
    fn page_calls_outside_a_page_are_rejected() {
        let Ok(b) = CudaBackend::new() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let mut device = b.alloc_device(64).expect("alloc");
        let host = b.alloc_host_pinned(64).expect("pinned");
        let outside = |r: Result<(), BackendError>| {
            let msg = r.expect_err("accepted outside a page").to_string();
            assert!(msg.contains("expected Recording"), "{msg}");
        };
        outside(b.record_zero_buffer(&mut device, 64));
        // SAFETY: the copy is rejected before anything is queued.
        outside(unsafe { b.record_copy_to_device(&host, &mut device, 64) });
        outside(b.submit_page().map(drop));

        b.begin_page().expect("begin_page");
        let msg = b.begin_page().expect_err("nested page").to_string();
        assert!(msg.contains("expected Idle"), "{msg}");
        b.record_zero_buffer(&mut device, 64)
            .expect("inside the page");
        let fence = b.submit_page().expect("submit_page");
        b.wait_page(fence).expect("wait_page");
        outside(b.record_zero_buffer(&mut device, 64));

        let failed = b.run_page(|_| Err(BackendError::msg("injected")));
        assert!(failed.is_err());
        b.run_page(|b| b.record_zero_buffer(&mut device, 64))
            .expect("a failed page leaves the thread idle");
    }

    #[test]
    fn the_recording_state_is_per_thread() {
        let Ok(b) = CudaBackend::new() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        b.begin_page().expect("begin_page");
        std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    let mut device = b.alloc_device(64).expect("alloc");
                    b.run_page(|b| b.record_zero_buffer(&mut device, 64))
                        .expect("another thread's page is independent");
                })
                .join()
                .expect("page thread");
        });
        let fence = b.submit_page().expect("submit_page");
        b.wait_page(fence).expect("wait_page");
    }
}
