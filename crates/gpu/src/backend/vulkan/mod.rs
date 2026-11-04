//! Vulkan compute backend implementing `GpuBackend`.
//!
//! Mirrors the CUDA backend's shape: instance → device → queue at init
//! time, then `begin_page → record_* → submit_page → wait_page` for each
//! rendered page.  All five kernel families dispatch through `record_*`
//! into a per-page command buffer; submission uses a timeline semaphore
//! so the host waits exactly once per page.
//!
//! Module split (all private; named here for orientation only):
//! - `barrier` — global memory barriers by stage/access scope.
//! - `device` — loader, physical-device pick, logical device + queue.
//! - `error` — `vk::Result → BackendError` adaptor.
//! - `memory` — slab sub-allocator, host-visible staging pool.
//! - `pipeline` — descriptor set layouts + pipeline cache (lazy SPIR-V compile).
//! - `select` — device selector parsing and physical-device ranking.
//! - `recorder` — per-page command buffer + timeline-semaphore fence.
//! - `transfer` — synchronous staged upload and download.

mod barrier;
mod device;
mod error;
mod memory;
mod pipeline;
mod recorder;
mod select;
mod transfer;

pub use memory::{DeviceBuffer, HostBuffer};
pub use recorder::PageFence;
pub use select::DeviceSelector;

use std::sync::Arc;

use crate::backend::{GpuBackend, Result, VramBudget, params, reject_zero_size};

/// Vulkan compute backend.
///
/// Wraps an Arc'd device context plus a slab allocator and pipeline cache.
/// The trait surface is identical to the CUDA backend; switching is a
/// one-line change at the renderer's `RasterSession` constructor.
pub struct VulkanBackend {
    // Field order = drop order: children that record into command buffers
    // / hold pool handles must drop before the device they were created on.
    transfer: transfer::TransferContext,
    recorder: recorder::PageRecorder,
    /// Held for the whole of a [`GpuBackend::run_page`]: the recorder has
    /// one recording state, so pages from several threads run one at a
    /// time.
    page: std::sync::Mutex<()>,
    #[expect(
        dead_code,
        reason = "kept for explicit drop ordering; readers go via recorder's Arc clone"
    )]
    pipelines: Arc<pipeline::PipelineCache>,
    memory: memory::SlabAllocator,
    device: Arc<device::DeviceCtx>,
}

impl VulkanBackend {
    /// Initialise the Vulkan loader, instance, device, and kernel pipelines
    /// on the automatically ranked physical device.
    ///
    /// # Errors
    /// Returns `BackendError` if no suitable Vulkan 1.3+ device is present,
    /// any required feature is missing, kernel SPIR-V fails to load, or any
    /// driver call returns a non-success status.
    pub fn new() -> Result<Self> {
        Self::with_selector(None)
    }

    /// As [`Self::new`], on the device named by `selector` when given.
    ///
    /// # Errors
    /// As [`Self::new`]; additionally [`BackendError::DeviceSelection`]
    /// when `selector` matches no device, several devices, or a device
    /// that is not eligible.
    ///
    /// [`BackendError::DeviceSelection`]: crate::backend::BackendError::DeviceSelection
    pub fn with_selector(selector: Option<&DeviceSelector>) -> Result<Self> {
        let device = device::init(selector)?;
        let memory = memory::SlabAllocator::new(device.clone())?;
        let pipelines = pipeline::PipelineCache::new(device.clone())?;
        let recorder = recorder::PageRecorder::new(device.clone(), pipelines.clone())?;
        let transfer = transfer::TransferContext::new(device.clone())?;
        Ok(Self {
            transfer,
            recorder,
            page: std::sync::Mutex::new(()),
            pipelines,
            memory,
            device,
        })
    }

    /// Check that `selector` names exactly one device that can host the
    /// backend, without opening it: an instance is created and destroyed,
    /// no logical device or pipeline is built.
    ///
    /// # Errors
    /// [`BackendError::DeviceSelection`] when it does not resolve; the
    /// loader's or instance's error when Vulkan is unavailable.
    ///
    /// [`BackendError::DeviceSelection`]: crate::backend::BackendError::DeviceSelection
    pub fn check_selector(selector: &DeviceSelector) -> Result<()> {
        device::check_selector(selector)
    }

    /// Whether the device is an integrated GPU, whose device buffers are
    /// host memory the host writes and reads directly.
    #[must_use]
    pub fn is_integrated(&self) -> bool {
        self.device.integrated
    }

    /// Queue submissions this backend has made so far: one per page, one
    /// per synchronous transfer or zero-fill.
    #[must_use]
    pub fn submits(&self) -> u64 {
        self.device.submits()
    }
}

/// `len` as a Vulkan copy size, after checking that it fits `capacity`, the
/// size of the buffer that is the copy's `side` (`"source"` or
/// `"destination"`); `op` names the operation in the error.
fn copy_len(len: usize, capacity: u64, op: &'static str, side: &'static str) -> Result<u64> {
    u64::try_from(len)
        .ok()
        .filter(|&n| n <= capacity)
        .ok_or_else(|| {
            crate::backend::BackendError::msg(format!(
                "{op}: copy of {len} bytes exceeds the {side} buffer ({capacity} bytes)"
            ))
        })
}

/// The size of a recorded copy of `len` bytes from a buffer of `src` bytes
/// to one of `dst` bytes, `None` for an empty copy.
fn recorded_copy_len(len: usize, src: u64, dst: u64, op: &'static str) -> Result<Option<u64>> {
    copy_len(len, src, op, "source").map(drop)?;
    let n = copy_len(len, dst, op, "destination")?;
    Ok((n > 0).then_some(n))
}

/// Abandons the recorder's page when dropped armed: on the error path of
/// [`GpuBackend::run_page`] and when its closure unwinds.
struct AbortPageOnExit<'a> {
    recorder: &'a recorder::PageRecorder,
    armed: bool,
}

impl Drop for AbortPageOnExit<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.recorder.abort_page();
        }
    }
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        // Wait for all in-flight work before any handle in any submodule
        // tears down; the per-module Drop impls assume nothing is using
        // their resources.  Failure here is logged and ignored — we're
        // already on the destruction path.
        // Safety: sole owner; called exactly once.
        unsafe {
            if let Err(e) = self.device.device.device_wait_idle() {
                log::warn!("vkDeviceWaitIdle failed during VulkanBackend::drop: {e:?}");
            }
        }
    }
}

impl crate::backend::sealed::Sealed for VulkanBackend {}

impl GpuBackend for VulkanBackend {
    const NAME: &'static str = "vulkan";
    type DeviceBuffer = memory::DeviceBuffer;
    type HostBuffer = memory::HostBuffer;
    type PageFence = recorder::PageFence;

    fn alloc_device(&self, size: usize) -> Result<Self::DeviceBuffer> {
        reject_zero_size(size, "alloc_device")?;
        self.memory.alloc_device(size, memory::HostAccess::Write)
    }

    fn alloc_device_readback(&self, size: usize) -> Result<Self::DeviceBuffer> {
        reject_zero_size(size, "alloc_device_readback")?;
        self.memory.alloc_device(size, memory::HostAccess::ReadBack)
    }

    fn alloc_device_zeroed(&self, size: usize) -> Result<Self::DeviceBuffer> {
        reject_zero_size(size, "alloc_device_zeroed")?;
        let mut buf = self.memory.alloc_device(size, memory::HostAccess::Write)?;
        let byte_len = buf.size();
        // A fresh buffer is referenced by no page: a mapped one is zeroed in
        // place, under the same size rule as the device fill so that a size
        // is accepted or refused alike on every device.
        if let Some(bytes) = buf.mapped_mut() {
            memory::validate_fill_size(byte_len).map(drop)?;
            bytes.fill(0);
            return Ok(buf);
        }
        // vkCmdFillBuffer requires 4-byte-aligned size; every realistic
        // caller (RGBA8 pages, u32 indices) is naturally aligned.
        // Mis-aligned sizes fail loudly rather than silently downgrading
        // to a 33 MB host-vec staging path.
        //
        // Allocation-time zero-fill rides `run_one_shot`, which holds
        // both `TransferContext::cmd_pool` and `DeviceCtx::compute_queue`
        // for the full submit + `vkQueueWaitIdle` (~1–2 ms for a 4K RGBA8
        // page). Concurrent renderer threads serialise here.
        //
        // The cheaper path is [`GpuBackend::record_zero_buffer`], which
        // folds the fill into the page's own command buffer between
        // `begin_page` and `submit_page` — zero extra submits, no
        // transfer-queue lock. Callers that already hold an active page
        // recording should prefer it. This method exists for paths that
        // need the buffer zeroed *outside* a page (e.g., one-shot setup,
        // tests).
        self.transfer.fill_zero(&buf)?;
        Ok(buf)
    }

    fn device_buffer_len(&self, buf: &Self::DeviceBuffer) -> usize {
        // Vulkan stores sizes as u64 (vk::DeviceSize); the trait
        // returns usize.  rasterrocket targets 64-bit only, so the
        // conversion is total — failing loudly is correct because
        // a saturating fallback (e.g. usize::MAX) would silently
        // poison cache size accounting.
        usize::try_from(buf.size()).expect("DeviceBuffer size fits usize on 64-bit targets")
    }

    fn free_device(&self, buf: Self::DeviceBuffer) {
        self.memory.free_device(buf);
    }

    fn alloc_host_pinned(&self, size: usize) -> Result<Self::HostBuffer> {
        reject_zero_size(size, "alloc_host_pinned")?;
        self.memory.alloc_host(size)
    }

    fn alloc_host_readback(&self, size: usize) -> Result<Self::HostBuffer> {
        reject_zero_size(size, "alloc_host_readback")?;
        self.memory.alloc_host_readback(size)
    }

    fn free_host_pinned(&self, buf: Self::HostBuffer) {
        self.memory.free_host(buf);
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
        Err(crate::backend::BackendError::msg(
            "VulkanBackend::record_scan: gpu-jpeg-huffman feature is not enabled",
        ))
    }

    #[cfg(feature = "gpu-jpeg-huffman")]
    fn record_huffman(&self, params: params::HuffmanParams<'_, Self>) -> Result<()> {
        params.validate(self)?;
        self.recorder.record_huffman(params)
    }

    #[cfg(not(feature = "gpu-jpeg-huffman"))]
    fn record_huffman(&self, _params: params::HuffmanParams<'_, Self>) -> Result<()> {
        Err(crate::backend::BackendError::msg(
            "VulkanBackend::record_huffman: gpu-jpeg-huffman feature is not enabled",
        ))
    }

    #[cfg(feature = "gpu-jpeg-huffman")]
    fn record_idct(&self, params: params::IdctParams<'_, Self>) -> Result<()> {
        params.validate(self)?;
        self.recorder.record_idct(params)
    }

    #[cfg(not(feature = "gpu-jpeg-huffman"))]
    fn record_idct(&self, _params: params::IdctParams<'_, Self>) -> Result<()> {
        Err(crate::backend::BackendError::msg(
            "VulkanBackend::record_idct: gpu-jpeg-huffman feature is not enabled",
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
        recorded_copy_len(len, src.size(), dst.size(), "record_copy_to_device")?
            .map_or(Ok(()), |n| {
                self.recorder.record_copy(src.handle(), dst.handle(), n)
            })
    }

    unsafe fn record_copy_to_host(
        &self,
        src: &Self::DeviceBuffer,
        dst: &mut Self::HostBuffer,
        len: usize,
    ) -> Result<()> {
        recorded_copy_len(len, src.size(), dst.size(), "record_copy_to_host")?.map_or(Ok(()), |n| {
            self.recorder.record_copy(src.handle(), dst.handle(), n)
        })
    }

    fn device_memory_is_host_mapped(&self) -> bool {
        self.device.integrated
    }

    unsafe fn device_bytes<'a>(&self, buf: &'a Self::DeviceBuffer) -> Option<&'a [u8]> {
        buf.mapped()
    }

    unsafe fn device_bytes_mut<'a>(&self, buf: &'a mut Self::DeviceBuffer) -> Option<&'a mut [u8]> {
        buf.mapped_mut()
    }

    fn host_buffer_len(&self, buf: &Self::HostBuffer) -> usize {
        buf.as_slice().len()
    }

    fn host_bytes<'a>(&self, buf: &'a Self::HostBuffer) -> &'a [u8] {
        buf.as_slice()
    }

    fn host_bytes_mut<'a>(&self, buf: &'a mut Self::HostBuffer) -> &'a mut [u8] {
        buf.as_mut_slice()
    }

    fn submit_page(&self) -> Result<Self::PageFence> {
        self.recorder.submit_page()
    }

    fn wait_page(&self, fence: Self::PageFence) -> Result<()> {
        self.recorder.wait_page(fence)
    }

    fn run_page<F>(&self, record: F) -> Result<()>
    where
        F: FnOnce(&Self) -> Result<()>,
    {
        let _page = self
            .page
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut abort = AbortPageOnExit {
            recorder: &self.recorder,
            armed: true,
        };
        let result = self.begin_page().and_then(|()| {
            record(self)?;
            let fence = self.submit_page()?;
            self.wait_page(fence)
        });
        abort.armed = result.is_err();
        result
    }

    /// A mapped buffer (integrated GPU) is written in place under
    /// [`recorder::PageRecorder::with_quiescent`]: earlier pages read the old
    /// bytes, later ones the new, as with the staged copy.
    fn upload(&self, dst: &Self::DeviceBuffer, src: &[u8]) -> Result<()> {
        if !dst.is_mapped() {
            return self.transfer.upload(&self.memory, dst, src);
        }
        if src.is_empty() {
            return Ok(());
        }
        self.recorder.with_quiescent(|| dst.write_mapped(src))?
    }

    /// A mapped buffer (integrated GPU) is read in place under
    /// [`recorder::PageRecorder::with_quiescent`]: no page writes it while
    /// the host reads.
    fn download(&self, src: &Self::DeviceBuffer, dst: &mut [u8]) -> Result<()> {
        if !src.is_mapped() {
            return self.transfer.download(&self.memory, src, dst);
        }
        if dst.is_empty() {
            return Ok(());
        }
        let len = copy_len(dst.len(), src.size(), "download", "source")?;
        self.recorder.with_quiescent(|| {
            let bytes = src
                .mapped()
                .and_then(|b| b.get(..dst.len()))
                .ok_or_else(|| memory::not_mapped("download", len))?;
            dst.copy_from_slice(bytes);
            Ok(())
        })?
    }

    fn detect_vram_budget(&self) -> Result<VramBudget> {
        let (used, budget) = device::query_memory_budget(&self.device)?;
        let usable = budget.saturating_sub(used).saturating_mul(3) / 4;
        let total = self.device.vram_total;
        // `total` and `budget` are sums over the same heaps (those device
        // buffers are placed in, `DeviceCtx::buffer_heaps`), so usable <=
        // total; the clamp covers a driver reporting a budget above the
        // heap size.
        Ok(VramBudget::new(total, usable.min(total)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk;

    /// The default device, or `None` (the test is skipped) when no Vulkan
    /// device is present.
    fn backend() -> Option<VulkanBackend> {
        match VulkanBackend::new() {
            Ok(b) => Some(b),
            Err(e) => {
                eprintln!("skipping: no Vulkan backend: {e}");
                None
            }
        }
    }

    /// Whether `device` offers a host-visible, host-coherent, host-cached
    /// memory type.
    fn offers_host_cached(device: &device::DeviceCtx) -> bool {
        let wanted = vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT
            | vk::MemoryPropertyFlags::HOST_CACHED;
        device.mem_props.memory_types[..device.mem_props.memory_type_count as usize]
            .iter()
            .any(|t| t.property_flags.contains(wanted))
    }

    #[test]
    fn vulkan_quiescent_host_access_excludes_submission() {
        let Some(backend) = backend() else { return };
        let blocked = backend
            .recorder
            .with_quiescent(|| backend.recorder.submission_blocked())
            .expect("with_quiescent");
        assert!(blocked, "a page could be submitted during a host access");
        assert!(!backend.recorder.submission_blocked());
    }

    /// The first integrated GPU among the devices named by a driver that
    /// only integrated parts carry, or `None` (the test is skipped).
    fn integrated() -> Option<VulkanBackend> {
        let found = ["radv", "intel"].iter().find_map(|name| {
            let selector = name.parse().expect("non-empty selector");
            VulkanBackend::with_selector(Some(&selector))
                .ok()
                .filter(VulkanBackend::is_integrated)
        });
        if found.is_none() {
            eprintln!("skipping: no integrated GPU");
        }
        found
    }

    #[test]
    fn vulkan_shared_memory_readback_buffers_are_host_cached() {
        let Some(backend) = integrated() else { return };
        let written = backend.alloc_device(4096).expect("alloc_device");
        let read = backend
            .alloc_device_readback(4096)
            .expect("alloc_device_readback");
        assert!(written.is_mapped() && read.is_mapped());
        if offers_host_cached(&backend.device) {
            assert!(
                read.memory_properties()
                    .contains(vk::MemoryPropertyFlags::HOST_CACHED),
                "read-back buffer is in uncached memory: {:?}",
                read.memory_properties()
            );
        }
    }

    #[test]
    fn vulkan_staged_download_reads_host_cached_memory() {
        let Some(backend) = backend() else { return };
        if backend.is_integrated() || !offers_host_cached(&backend.device) {
            eprintln!("skipping: the download is not staged or no cached type exists");
            return;
        }
        let buf = backend
            .alloc_device_zeroed(4096)
            .expect("alloc_device_zeroed");
        let mut out = vec![1u8; 4096];
        backend.download(&buf, &mut out).expect("download");
        let flags = backend
            .transfer
            .download_staging_properties()
            .expect("the download allocated its staging buffer");
        assert!(
            flags.contains(vk::MemoryPropertyFlags::HOST_CACHED),
            "download staging is in uncached memory: {flags:?}"
        );
    }
}
