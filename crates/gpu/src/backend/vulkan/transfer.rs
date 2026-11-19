//! Host↔device transfer helpers.
//!
//! Synchronous upload/download via reusable `HOST_VISIBLE` staging
//! buffers, one per direction (each grown to the largest transfer seen so
//! far), and a one-time-submit command buffer on the compute queue whose
//! completion is waited for on the context's own fence.

#![expect(
    clippy::significant_drop_tightening,
    reason = "each staging Mutex<Option<HostBuffer>> is held across Vulkan calls intentionally — releasing it between the host-side write and the GPU-side wait would let a second upload observe the same staging buffer mid-DMA"
)]

use std::sync::{Arc, Mutex};

use ash::vk;

use crate::backend::{BackendError, Result};

use super::barrier;
use super::device::DeviceCtx;
use super::error::vk_err;
use super::memory::{DeviceBuffer, FillAction, HostBuffer, SlabAllocator, validate_fill_size};

pub(super) struct TransferContext {
    device: Arc<DeviceCtx>,
    /// Reusable command pool + cached primary command buffer for
    /// transient one-shot transfers.  A separate pool from the
    /// recorder's so resets don't interleave; matches the canonical
    /// "one pool per logical stream of submissions" pattern.
    ///
    /// `VkCommandPool` requires external sync (spec "Threading
    /// Behavior" table; VUID-vkResetCommandPool-commandPool-00040
    /// forbids reset while any buffer from the pool is recording or
    /// pending) — the `Mutex` is the only access path.  The cached
    /// `vk::CommandBuffer` is reused across calls: pool reset
    /// transitions it back to Initial state without freeing it,
    /// avoiding a `vkAllocateCommandBuffers` round-trip per transfer.
    cmd_pool: Mutex<TransferPool>,
    /// Staging buffer of [`Self::upload`], in memory the host writes
    /// ([`SlabAllocator::alloc_host`]), grown to the largest upload so far.
    /// `None` until the first upload.  The `Mutex` is held from the host
    /// write to the end of the device copy, so a concurrent upload cannot
    /// overwrite the bytes being copied.
    upload_staging: Mutex<Option<HostBuffer>>,
    /// Staging buffer of [`Self::download`], in host-cached memory where
    /// the device offers it ([`SlabAllocator::alloc_host_readback`]), under
    /// the same growth and locking rule as `upload_staging`.
    download_staging: Mutex<Option<HostBuffer>>,
}

/// Pool, cached cmd buffer and the fence its submissions signal, guarded
/// together by `cmd_pool`'s mutex.
struct TransferPool {
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
}

impl TransferContext {
    pub(super) fn new(device: Arc<DeviceCtx>) -> Result<Self> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(device.compute_queue_family);
        // Safety: device is live; create_command_pool returns Result on failure.
        let pool = unsafe { device.device.create_command_pool(&pool_info, None) }
            .map_err(vk_err("vkCreateCommandPool (transfer)"))?;

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // Safety: pool is live; allocate_command_buffers returns Result on failure.
        let buffers = unsafe { device.device.allocate_command_buffers(&alloc_info) }
            .map_err(vk_err("vkAllocateCommandBuffers (transfer init)"))
            .inspect_err(|_| unsafe { device.device.destroy_command_pool(pool, None) })?;
        let cmd = buffers[0];
        // Safety: device is live; create_fence returns Result on failure.
        let fence = unsafe {
            device
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .map_err(vk_err("vkCreateFence (transfer)"))
        .inspect_err(|_| unsafe { device.device.destroy_command_pool(pool, None) })?;

        Ok(Self {
            device,
            cmd_pool: Mutex::new(TransferPool { pool, cmd, fence }),
            upload_staging: Mutex::new(None),
            download_staging: Mutex::new(None),
        })
    }

    /// Copy `src` into `dst[0..src.len()]` and `zeros` zero bytes after it
    /// through the staging buffer and block until the copy has completed; backs the trait's
    /// `upload`.
    pub(super) fn upload(
        &self,
        allocator: &SlabAllocator,
        dst: &DeviceBuffer,
        src: &[u8],
        zeros: usize,
    ) -> Result<()> {
        let len = src.len().saturating_add(zeros);
        if len == 0 {
            return Ok(());
        }
        let total = super::copy_len(len, dst.size(), "upload", "destination")?;
        // Hold the staging slot for the entire operation: write the
        // payload, kick off the GPU copy, wait for completion.  Releasing
        // the lock between write and wait would let a second upload
        // observe the same staging buffer mid-DMA.
        let mut staging_slot = self
            .upload_staging
            .lock()
            .expect("transfer staging mutex poisoned");
        let staging = ensure_staging(&mut staging_slot, len, |n| allocator.alloc_host(n))?;
        let (payload, pad) = staging.as_mut_slice()[..len].split_at_mut(src.len());
        payload.copy_from_slice(src);
        pad.fill(0);
        let staging_handle = staging.handle();
        let dst_handle = dst.handle();
        self.copy_via_staging(staging_handle, dst_handle, total)
    }

    /// Synchronous download: copy `src[0..dst.len()]` into `dst`, block
    /// until complete.
    pub(super) fn download(
        &self,
        allocator: &SlabAllocator,
        src: &DeviceBuffer,
        dst: &mut [u8],
    ) -> Result<()> {
        if dst.is_empty() {
            return Ok(());
        }
        let len = super::copy_len(dst.len(), src.size(), "download", "source")?;
        let mut staging_slot = self
            .download_staging
            .lock()
            .expect("transfer staging mutex poisoned");
        let staging = ensure_staging(&mut staging_slot, dst.len(), |n| {
            allocator.alloc_host_readback(n)
        })?;
        let staging_handle = staging.handle();
        let src_handle = src.handle();
        self.copy_via_staging(src_handle, staging_handle, len)?;
        // The copy has completed (the transfer fence) and its writes are
        // visible to the host (`run_one_shot`'s closing barrier); the
        // memory is HOST_COHERENT, so no invalidate is needed.
        dst.copy_from_slice(&staging.as_slice()[..dst.len()]);
        Ok(())
    }

    /// Submit a single `vkCmdCopyBuffer(src→dst, len)` and wait for it
    /// to complete.  Caller chooses which side is the staging buffer.
    fn copy_via_staging(&self, src: vk::Buffer, dst: vk::Buffer, len: u64) -> Result<()> {
        self.run_one_shot(|cmd| {
            let region = [vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(len)];
            // Safety: cmd is in Recording state per run_one_shot's contract.
            unsafe {
                self.device.device.cmd_copy_buffer(cmd, src, dst, &region);
            }
        })
    }

    /// Zero-fill `dst[0..dst.size()]` on the GPU via `vkCmdFillBuffer`.
    ///
    /// `vkCmdFillBuffer` requires `dst.size()` to be a multiple of 4 —
    /// non-multiple-of-4 sizes return an error rather than silently
    /// rounding down (which would leave the trailing 1–3 bytes
    /// non-zero).
    pub(super) fn fill_zero(&self, dst: &DeviceBuffer) -> Result<()> {
        let size = match validate_fill_size(dst.size())? {
            FillAction::Skip => return Ok(()),
            FillAction::Fill(s) => s,
        };
        let dst_handle = dst.handle();
        self.run_one_shot(|cmd| {
            // Safety: run_one_shot has cmd in Recording state; dst_handle
            // was created with TRANSFER_DST usage (DEVICE_USAGE in
            // memory.rs); size is a checked multiple of 4.
            unsafe {
                self.device
                    .device
                    .cmd_fill_buffer(cmd, dst_handle, 0, size, 0);
            }
        })
    }

    /// Reset the pool, record `f` and a closing barrier into the cached
    /// cmd buffer, submit it signalling the transfer fence, and wait on the
    /// fence.  Holds the `cmd_pool` mutex for the whole body so the buffer
    /// and the fence can't be reused before the wait returns (cmd buffers in
    /// Pending state must not be reset).  The queue itself is locked only
    /// for the submit.
    fn run_one_shot<F: FnOnce(vk::CommandBuffer)>(&self, f: F) -> Result<()> {
        let pool = self.cmd_pool.lock().expect("transfer cmd_pool poisoned");
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // Safety: pool mutex held throughout; cmd buffer was allocated
        // from this pool at TransferContext::new and is reused across
        // calls.  reset_command_pool transitions it back to Initial.
        unsafe {
            self.device
                .device
                .reset_command_pool(pool.pool, vk::CommandPoolResetFlags::empty())
                .map_err(vk_err("vkResetCommandPool (transfer)"))?;
            self.device
                .device
                .begin_command_buffer(pool.cmd, &begin_info)
                .map_err(vk_err("vkBeginCommandBuffer (transfer)"))?;
        }

        f(pool.cmd);

        // A wait on the queue makes the transfer's writes available to the
        // device domain only: the barrier makes them visible to host reads
        // of a staging buffer and to the dispatches of later submissions.
        // Safety: cmd is in Recording state; pool mutex still held.
        unsafe {
            barrier::global(
                &self.device.device,
                pool.cmd,
                barrier::TRANSFER_WRITES,
                barrier::LATER_SUBMISSIONS_AND_HOST,
            );
            self.device
                .device
                .end_command_buffer(pool.cmd)
                .map_err(vk_err("vkEndCommandBuffer (transfer)"))?;
        }

        let submit_buffers = [pool.cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&submit_buffers);
        let fences = [pool.fence];
        // Safety: the command buffer was ended above; the fence is used by
        // this context's submissions only, each waited for under the pool
        // mutex before the next resets it, so it is not in use here.
        unsafe {
            self.device
                .device
                .reset_fences(&fences)
                .map_err(vk_err("vkResetFences (transfer)"))?;
            self.device
                .submit(&submit, pool.fence, "vkQueueSubmit (transfer)")?;
            self.device
                .device
                .wait_for_fences(&fences, true, u64::MAX)
                .map_err(vk_err("vkWaitForFences (transfer)"))
        }
    }
}

#[cfg(test)]
impl TransferContext {
    /// Memory property flags of the download staging buffer, once a
    /// download has allocated it.
    pub(super) fn download_staging_properties(&self) -> Option<vk::MemoryPropertyFlags> {
        self.download_staging
            .lock()
            .expect("transfer staging mutex poisoned")
            .as_ref()
            .map(HostBuffer::memory_properties)
    }
}

/// The staging buffer in `slot`, replaced first by one from `alloc` when it
/// holds fewer than `min_size` bytes, at the size
/// [`crate::backend::staged::grown_size`] gives.
fn ensure_staging(
    slot: &mut Option<HostBuffer>,
    min_size: usize,
    alloc: impl FnOnce(usize) -> Result<HostBuffer>,
) -> Result<&mut HostBuffer> {
    let current_capacity = slot.as_ref().map_or(Ok(0), |b| {
        usize::try_from(b.size()).map_err(|_| BackendError::msg("staging size exceeds usize"))
    })?;
    if min_size > current_capacity {
        let grown = crate::backend::staged::grown_size(current_capacity, min_size);
        // Drop the old buffer first so its VkBuffer + allocation are
        // released before we ask for the new one.
        *slot = None;
        return Ok(slot.insert(alloc(grown)?));
    }
    slot.as_mut()
        .ok_or_else(|| BackendError::msg("staging buffer requested for an empty transfer"))
}

impl Drop for TransferContext {
    fn drop(&mut self) {
        // VulkanBackend::drop calls device_wait_idle before module Drops
        // run, so the cached cmd buffer is not in flight.  Destroying
        // the pool also frees its allocated buffers.
        let pool = self.cmd_pool.lock().expect("transfer cmd_pool poisoned");
        // Safety: pool is owned by us; no in-flight references.
        unsafe {
            self.device.device.destroy_fence(pool.fence, None);
            self.device.device.destroy_command_pool(pool.pool, None);
        }
    }
}
