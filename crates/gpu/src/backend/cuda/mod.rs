//! CUDA backend implementing `GpuBackend`.
//!
//! Wraps the existing `GpuCtx` and per-kernel `lib_kernels::*` functions.
//! Per-page batching is recorded into a `PageRecorder`; see `page_recorder.rs`.

mod page_recorder;
mod pinned;
mod transfer;

pub use pinned::PinnedBuffer;

use std::sync::Arc;

use cudarc::driver::CudaSlice;

use crate::GpuCtx;
use crate::backend::{
    BackendError, GpuBackend, Result, VramBudget, check_copy_len, params, reject_zero_size,
};

/// Convert any `Display` error to a `BackendError`.
///
/// `GpuCtx::init` returns `Box<dyn Error>` (not `Send + Sync`), so we
/// stringify and route through [`BackendError::msg`].  The source chain
/// is lost but the message survives — sufficient for the init paths
/// this is used on (driver missing, device missing, kernel-load failure).
pub(super) fn be(e: impl std::fmt::Display) -> BackendError {
    BackendError::msg(e.to_string())
}

/// A CUDA backend that wraps `GpuCtx`.
pub struct CudaBackend {
    ctx: Arc<GpuCtx>,
    recorder: page_recorder::PageRecorder,
}

impl CudaBackend {
    /// Initialise CUDA device 0 and compile the embedded kernels.
    ///
    /// # Errors
    /// Returns `BackendError` if no CUDA device is present or kernel load fails.
    pub fn new() -> Result<Self> {
        Ok(Self::from_ctx(Arc::new(GpuCtx::init().map_err(be)?)))
    }

    /// A backend over an initialised context, sharing its kernels and its
    /// per-thread streams.
    #[must_use]
    pub fn from_ctx(ctx: Arc<GpuCtx>) -> Self {
        Self {
            recorder: page_recorder::PageRecorder::new(Arc::clone(&ctx)),
            ctx,
        }
    }
}

impl crate::backend::sealed::Sealed for CudaBackend {}

impl GpuBackend for CudaBackend {
    const NAME: &'static str = "cuda";
    type DeviceBuffer = CudaSlice<u8>;
    type HostBuffer = PinnedBuffer;
    type PageFence = page_recorder::PageFence;

    fn alloc_device(&self, size: usize) -> Result<Self::DeviceBuffer> {
        reject_zero_size(size, "alloc_device")?;
        // cudarc 0.19 does expose an unzeroed path (the unsafe
        // stream-ordered `CudaStream::alloc`), but we deliberately
        // over-deliver the trait contract (which makes no promise
        // about contents) with `alloc_zeros`: the memset is device-side
        // and cheap, and a caller bug that reads before writing sees
        // deterministic zeros instead of nondeterministic garbage.
        // Callers that rely on zeroing must still use
        // `alloc_device_zeroed` — the contract is the weaker one.
        self.ctx.stream().alloc_zeros::<u8>(size).map_err(be)
    }

    fn alloc_device_zeroed(&self, size: usize) -> Result<Self::DeviceBuffer> {
        reject_zero_size(size, "alloc_device_zeroed")?;
        // The memset is queued on the calling thread's stream, ahead of
        // that thread's later kernels and copies.
        self.ctx.stream().alloc_zeros::<u8>(size).map_err(be)
    }

    fn device_buffer_len(&self, buf: &Self::DeviceBuffer) -> usize {
        buf.len()
    }

    fn free_device(&self, _buf: Self::DeviceBuffer) {
        // cudarc's `CudaSlice::Drop` calls `cuMemFreeAsync` on the
        // owning stream when the buffer was allocated via a stream
        // (`stream.alloc_zeros` etc.), gating the free on prior stream
        // work.  Nothing to do here — Drop runs as the value moves out
        // of scope.
    }

    fn alloc_host_pinned(&self, size: usize) -> Result<Self::HostBuffer> {
        reject_zero_size(size, "alloc_host_pinned")?;
        PinnedBuffer::new(&self.ctx.cuda, size)
    }

    fn alloc_host_readback(&self, size: usize) -> Result<Self::HostBuffer> {
        reject_zero_size(size, "alloc_host_readback")?;
        PinnedBuffer::new(&self.ctx.cuda, size)
    }

    fn free_host_pinned(&self, _buf: Self::HostBuffer) {
        // cudarc frees the pinned allocation on Drop.
    }

    fn begin_page(&self) -> Result<()> {
        self.recorder.begin_page()
    }

    fn record_blit_image(&self, params: params::BlitParams<'_, Self>) -> Result<()> {
        params.validate(self)?;
        self.recorder.record_blit_image(params)
    }

    fn record_aa_fill(&self, params: params::AaFillParams<'_, Self>) -> Result<()> {
        params.validate(self)?;
        self.recorder.record_aa_fill(params)
    }

    fn record_icc_clut(&self, params: params::IccClutParams<'_, Self>) -> Result<()> {
        let grid_n = params.validate(self)?;
        self.recorder.record_icc_clut(params, grid_n)
    }

    fn record_composite(&self, params: params::CompositeParams<'_, Self>) -> Result<()> {
        params.validate(self)?;
        self.recorder.record_composite(params)
    }

    fn record_apply_soft_mask(&self, params: params::SoftMaskParams<'_, Self>) -> Result<()> {
        params.validate(self)?;
        self.recorder.record_apply_soft_mask(params)
    }

    #[cfg(feature = "gpu-jpeg-huffman")]
    fn record_scan(&self, params: params::ScanParams<'_, Self>) -> Result<()> {
        params.validate(self)?;
        self.recorder.record_scan(params)
    }

    #[cfg(not(feature = "gpu-jpeg-huffman"))]
    fn record_scan(&self, _params: params::ScanParams<'_, Self>) -> Result<()> {
        // The kernel only ships when the gpu-jpeg-huffman feature is
        // enabled at build time; outside that feature the trait
        // method exists for ABI uniformity but rejects calls loudly.
        Err(BackendError::msg(
            "CudaBackend::record_scan: gpu-jpeg-huffman feature is not enabled",
        ))
    }

    #[cfg(feature = "gpu-jpeg-huffman")]
    fn record_huffman(&self, params: params::HuffmanParams<'_, Self>) -> Result<()> {
        params.validate(self)?;
        self.recorder.record_huffman(params)
    }

    #[cfg(not(feature = "gpu-jpeg-huffman"))]
    fn record_huffman(&self, _params: params::HuffmanParams<'_, Self>) -> Result<()> {
        Err(BackendError::msg(
            "CudaBackend::record_huffman: gpu-jpeg-huffman feature is not enabled",
        ))
    }

    #[cfg(feature = "gpu-jpeg-huffman")]
    fn record_idct(&self, params: params::IdctParams<'_, Self>) -> Result<()> {
        params.validate(self)?;
        self.recorder.record_idct(params)
    }

    #[cfg(not(feature = "gpu-jpeg-huffman"))]
    fn record_idct(&self, _params: params::IdctParams<'_, Self>) -> Result<()> {
        Err(BackendError::msg(
            "CudaBackend::record_idct: gpu-jpeg-huffman feature is not enabled",
        ))
    }

    fn record_zero_buffer(&self, buf: &Self::DeviceBuffer) -> Result<()> {
        self.recorder.record_zero_buffer(buf)
    }

    unsafe fn record_copy_to_device(
        &self,
        src: &Self::HostBuffer,
        dst: &Self::DeviceBuffer,
        len: usize,
    ) -> Result<()> {
        check_copy_len(len, src.len(), dst.len(), "record_copy_to_device")?;
        transfer::queue_htod(&self.ctx, &src.as_slice()[..len], dst)
    }

    unsafe fn record_copy_to_host(
        &self,
        src: &Self::DeviceBuffer,
        dst: &mut Self::HostBuffer,
        len: usize,
    ) -> Result<()> {
        check_copy_len(len, src.len(), dst.len(), "record_copy_to_host")?;
        transfer::queue_dtoh(&self.ctx, src, &mut dst.as_mut_slice()[..len])
    }

    fn device_memory_is_host_mapped(&self) -> bool {
        false
    }

    unsafe fn device_bytes<'a>(&self, _buf: &'a Self::DeviceBuffer) -> Option<&'a [u8]> {
        None
    }

    unsafe fn device_bytes_mut<'a>(
        &self,
        _buf: &'a mut Self::DeviceBuffer,
    ) -> Option<&'a mut [u8]> {
        None
    }

    fn host_buffer_len(&self, buf: &Self::HostBuffer) -> usize {
        buf.len()
    }

    fn host_bytes<'a>(&self, buf: &'a Self::HostBuffer) -> &'a [u8] {
        buf.as_slice()
    }

    fn host_bytes_mut<'a>(&self, buf: &'a mut Self::HostBuffer) -> &'a mut [u8] {
        buf.as_mut_slice()
    }

    /// As the trait's default, except that a failed page synchronises the
    /// calling thread's stream before returning: copies and kernels queued
    /// before the failure may still read or write host and device buffers
    /// the caller frees once the error is returned.
    fn run_page<F>(&self, record: F) -> Result<()>
    where
        F: FnOnce(&Self) -> Result<()>,
    {
        let result = self.begin_page().and_then(|()| {
            record(self)?;
            let fence = self.submit_page()?;
            self.wait_page(fence)
        });
        if result.is_err()
            && let Err(e) = self.ctx.stream().synchronize()
        {
            log::warn!("abandoning a failed page: synchronising its stream failed: {e}");
        }
        result
    }

    fn submit_page(&self) -> Result<Self::PageFence> {
        self.recorder.submit_page()
    }

    fn wait_page(&self, fence: Self::PageFence) -> Result<()> {
        self.recorder.wait_page(fence)
    }

    fn upload(&self, dst: &Self::DeviceBuffer, src: &[u8]) -> Result<()> {
        transfer::upload(&self.ctx, dst, src)
    }

    fn download(&self, src: &Self::DeviceBuffer, dst: &mut [u8]) -> Result<()> {
        transfer::download(&self.ctx, src, dst)
    }

    fn detect_vram_budget(&self) -> Result<VramBudget> {
        let cuda_ctx = &self.ctx.cuda;
        let (free, total) = cuda_ctx.mem_get_info().map_err(be)?;
        // free and total are usize (bytes); `free * 3 / 4` keeps arithmetic in
        // integer space — no f64 cast, no precision loss, no sign ambiguity.
        let usable_bytes = (free as u64).saturating_mul(3) / 4;
        // VramBudget::new asserts usable <= total. cuMemGetInfo's free is always
        // <= total, and 3/4 of free is <= free, so the invariant holds.
        Ok(VramBudget::new(total as u64, usable_bytes))
    }
}
