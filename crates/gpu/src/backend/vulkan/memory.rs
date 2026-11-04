//! Vulkan memory allocator wrapping `gpu-allocator`.
//!
//! `gpu-allocator` (pure Rust, used by `wgpu`) handles slab sub-allocation,
//! memory-type selection, and dedicated-allocation cut-off internally.
//! This module wraps it in the buffer types the `GpuBackend` trait
//! expects: opaque `DeviceBuffer` / `HostBuffer` handles whose `Drop`
//! returns the underlying allocation to the slab.
//!
//! Buffer usage flags are uniform across kernels: every device buffer is
//! a generic storage buffer that can also be a transfer source/dest, and
//! has its device-address queried (used by the future BDA path —
//! today's descriptor-set path doesn't read it).
//!
//! ## Drop and fence interaction
//!
//! `DeviceBuffer::drop` calls `vkDestroyBuffer` and frees the gpu-allocator
//! `Allocation` immediately.  Callers must therefore ensure no submitted
//! command buffer still references the buffer — the trait's per-page
//! `wait_page` is the canonical safe point.  Between `submit_page` and
//! `wait_page`, the caller still holds the `DeviceBuffer` (we hand back
//! a fence, not a buffer), so this is a documentation issue, not a leak.

use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{
    Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc,
};

use crate::backend::{BackendError, Result};

use super::device::DeviceCtx;
use super::error::vk_err;

/// Buffer usage shared by every device buffer in this backend.
///
/// `STORAGE_BUFFER` covers every read/write the compute kernels do.
/// `TRANSFER_SRC` + `TRANSFER_DST` lets `upload` and any
/// host-readback path use the same buffer without a re-allocation.
/// `SHADER_DEVICE_ADDRESS` is required for `vkGetBufferDeviceAddress`.
const DEVICE_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
        | vk::BufferUsageFlags::TRANSFER_SRC.as_raw()
        | vk::BufferUsageFlags::TRANSFER_DST.as_raw()
        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS.as_raw(),
);

/// Host-staging usage — host buffers are pure transfer sources/dests;
/// they aren't bound to descriptors, so no `STORAGE_BUFFER` and no BDA.
const HOST_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::TRANSFER_SRC.as_raw() | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
);

/// A device-resident storage buffer.
///
/// Owns the `VkBuffer` and the gpu-allocator `Allocation` that backs it.
/// `Drop` frees both.
pub struct DeviceBuffer {
    pub(super) buffer: vk::Buffer,
    pub(super) size: u64,
    pub(super) device_address: vk::DeviceAddress,
    /// Held inside an `Option` so `Drop` can take ownership and hand the
    /// allocation back to the slab; gpu-allocator's `free` consumes by
    /// value.  Always `Some` between construction and drop.
    allocation: Option<Allocation>,
    parent: Arc<Inner>,
}

impl std::fmt::Debug for DeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("buffer", &self.buffer)
            .field("size", &self.size)
            .field("device_address", &self.device_address)
            .finish_non_exhaustive()
    }
}

impl DeviceBuffer {
    /// Underlying `VkBuffer` handle.
    #[must_use]
    pub const fn handle(&self) -> vk::Buffer {
        self.buffer
    }
    /// Buffer size in bytes (as requested at allocation time, before
    /// alignment padding).
    #[must_use]
    pub const fn size(&self) -> u64 {
        self.size
    }
    /// Whether the buffer's memory is host-mapped (device buffers of an
    /// integrated GPU).
    #[must_use]
    pub fn is_mapped(&self) -> bool {
        self.allocation
            .as_ref()
            .is_some_and(|a| a.mapped_ptr().is_some())
    }

    /// The buffer's bytes in place, when its memory is host-mapped (device
    /// buffers of an integrated GPU); `None` otherwise.
    ///
    /// The bytes reflect device writes only once the fence of the page that
    /// made them has been waited on.
    #[must_use]
    pub(super) fn mapped(&self) -> Option<&[u8]> {
        let len = usize::try_from(self.size).ok()?;
        self.allocation.as_ref()?.mapped_slice()?.get(..len)
    }

    /// The buffer's bytes in place, writable; see [`Self::mapped`].  Writes
    /// are visible to device work submitted afterwards.
    pub(super) fn mapped_mut(&mut self) -> Option<&mut [u8]> {
        let len = usize::try_from(self.size).ok()?;
        self.allocation.as_mut()?.mapped_slice_mut()?.get_mut(..len)
    }

    /// Copy `src` into the start of the mapped bytes through a shared
    /// reference.
    ///
    /// The trait's `upload` takes the buffer by shared reference, as the
    /// staged path, which writes it by device copy, does.  The caller runs
    /// the write under `PageRecorder::with_quiescent`, so no device access
    /// races it and every other host access through `upload` / `download`
    /// is serialised against it; views from the trait's `device_bytes` are
    /// excluded by its safety contract.
    ///
    /// # Errors
    /// When `src` is longer than the buffer, or the buffer is not mapped.
    pub(super) fn write_mapped(&self, src: &[u8]) -> Result<()> {
        let len = super::copy_len(src.len(), self.size, "upload", "destination")?;
        let dst = self
            .allocation
            .as_ref()
            .and_then(Allocation::mapped_ptr)
            .ok_or_else(|| not_mapped("upload", len))?
            .cast::<u8>();
        // SAFETY: `dst` addresses the allocation's mapping, which holds at
        // least `self.size >= src.len()` bytes and which the device does
        // not access (see above); the source is host memory distinct from
        // the mapping.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_ptr(), src.len()) };
        Ok(())
    }

    /// Buffer device address — non-zero only when the device opted into
    /// `bufferDeviceAddress`.  Reserved for the future BDA push-constant
    /// path; today's descriptor-set bindings don't read it.
    #[must_use]
    pub const fn device_address(&self) -> vk::DeviceAddress {
        self.device_address
    }
}

#[cfg(test)]
impl DeviceBuffer {
    /// Property flags of the memory type backing the buffer.
    pub(crate) fn memory_properties(&self) -> vk::MemoryPropertyFlags {
        self.allocation.as_ref().map_or_else(
            vk::MemoryPropertyFlags::empty,
            Allocation::memory_properties,
        )
    }
}

/// The error of a host access in place to a device buffer that is not
/// host-mapped.
pub(super) fn not_mapped(op: &'static str, len: u64) -> BackendError {
    BackendError::msg(format!(
        "{op}: in-place access of {len} bytes to a device buffer that is not host-mapped"
    ))
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        let alloc = self.allocation.take().expect("DeviceBuffer dropped twice");
        free_buffer(&self.parent, self.buffer, alloc, "DeviceBuffer");
    }
}

/// Outcome of validating a `vkCmdFillBuffer` size against the Vulkan
/// spec's 4-byte-multiple requirement.
///
/// `Skip` is the explicit "no-op" signal — callers can short-circuit
/// without issuing the command at all. `Fill(size)` carries the
/// validated size for the caller to pass to `cmd_fill_buffer`. The
/// `Result` wrapper carries the misaligned-size error so callers do
/// `let size = validate_fill_size(buf.size())?;` and pattern-match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FillAction {
    /// Buffer has zero bytes; nothing to fill.  Returned for
    /// `size == 0` rather than `Fill(0)` so callers can't accidentally
    /// issue a zero-length `vkCmdFillBuffer` (which the spec rejects
    /// anyway, but explicit is cheaper than a driver round-trip).
    Skip,
    /// Validated size in bytes; safe to pass to `vkCmdFillBuffer`.
    Fill(u64),
}

/// Validate a buffer size against `vkCmdFillBuffer`'s 4-byte-multiple
/// rule, surfacing misalignment as a typed error rather than letting
/// the driver silently round down (which would leave 1–3 trailing
/// bytes non-zero).
///
/// Returns `Ok(FillAction::Skip)` for zero-size, `Ok(FillAction::Fill)`
/// for valid sizes, and `Err(BackendError::UnalignedFill)` otherwise.
pub(super) const fn validate_fill_size(size: u64) -> Result<FillAction> {
    if size == 0 {
        return Ok(FillAction::Skip);
    }
    if !size.is_multiple_of(4) {
        return Err(BackendError::UnalignedFill {
            size,
            required_alignment: 4,
        });
    }
    Ok(FillAction::Fill(size))
}

/// A host-visible (CPU-mappable) buffer.
///
/// `HOST_VISIBLE | HOST_COHERENT` so writes from the CPU are immediately
/// visible to the GPU without an explicit `vkFlushMappedMemoryRanges`.
pub struct HostBuffer {
    pub(super) buffer: vk::Buffer,
    pub(super) size: u64,
    /// Persistent map: gpu-allocator returns a stable `*mut u8` for the
    /// allocation's lifetime when we request a CpuToGpu/GpuToCpu mapping.
    mapped_ptr: *mut u8,
    allocation: Option<Allocation>,
    parent: Arc<Inner>,
}

// SAFETY: HostBuffer owns the mapping for its lifetime and gpu-allocator
// holds the underlying VkDeviceMemory; the raw pointer is just a cached
// copy of the persistent map.  Sending across threads is safe as long as
// only one thread accesses the slice at a time, which is enforced by the
// &self / &mut self borrow checker.
unsafe impl Send for HostBuffer {}
unsafe impl Sync for HostBuffer {}

impl std::fmt::Debug for HostBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostBuffer")
            .field("buffer", &self.buffer)
            .field("size", &self.size)
            .field("mapped_ptr", &self.mapped_ptr)
            .finish_non_exhaustive()
    }
}

impl HostBuffer {
    /// Underlying `VkBuffer` handle.
    #[must_use]
    pub const fn handle(&self) -> vk::Buffer {
        self.buffer
    }
    /// Size in bytes.
    #[must_use]
    pub const fn size(&self) -> u64 {
        self.size
    }
    /// Read-only view of the persistently-mapped contents.
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "self.size was set from a `usize` at alloc time via `u64::try_from(usize)`, so the round-trip back to usize is exact on every host"
    )]
    pub const fn as_slice(&self) -> &[u8] {
        // Safety: gpu-allocator returns a valid `*mut u8` that stays
        // mapped for the allocation's lifetime; the slice lifetime is
        // tied to `&self` so the pointer is dereferenceable.
        let len = self.size as usize;
        unsafe { std::slice::from_raw_parts(self.mapped_ptr, len) }
    }
    /// Mutable view of the persistently-mapped contents.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "see as_slice — usize→u64→usize round-trip is exact"
    )]
    pub const fn as_mut_slice(&mut self) -> &mut [u8] {
        let len = self.size as usize;
        // Safety: same as `as_slice`, with `&mut self` enforcing
        // exclusive access at the Rust-borrow level.
        unsafe { std::slice::from_raw_parts_mut(self.mapped_ptr, len) }
    }
}

#[cfg(test)]
impl HostBuffer {
    /// Property flags of the memory type backing the buffer.
    pub(super) fn memory_properties(&self) -> vk::MemoryPropertyFlags {
        self.allocation.as_ref().map_or_else(
            vk::MemoryPropertyFlags::empty,
            Allocation::memory_properties,
        )
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        let alloc = self.allocation.take().expect("HostBuffer dropped twice");
        free_buffer(&self.parent, self.buffer, alloc, "HostBuffer");
    }
}

/// Free a buffer + allocation pair.  Used by both `DeviceBuffer::drop`
/// and `HostBuffer::drop`; logs but does not propagate the gpu-allocator
/// free error since we're already on the destruction path.
fn free_buffer(parent: &Arc<Inner>, buffer: vk::Buffer, alloc: Allocation, kind: &str) {
    let mut allocator = parent.allocator.lock().expect("allocator mutex poisoned");
    if let Err(e) = allocator.free(alloc) {
        log::warn!("{kind} free returned: {e}");
    }
    drop(allocator);
    // Safety: `buffer` was created via `vkCreateBuffer` on `parent.device`;
    // we have exclusive ownership (the wrapping struct just took it via
    // `Drop::drop`); per the trait contract the caller has waited on any
    // PageFence that referenced it.
    unsafe {
        parent.device.device.destroy_buffer(buffer, None);
    }
}

/// Shared state held by every `DeviceBuffer` / `HostBuffer` so they can
/// reach back to the allocator and the device on Drop.
struct Inner {
    device: Arc<DeviceCtx>,
    /// gpu-allocator is `Send` but not `Sync`; wrap in a `Mutex` for the
    /// allocate/free hot path.  Contention is negligible — we make ~tens
    /// of allocations per page.
    allocator: Mutex<Allocator>,
}

/// How the host accesses a device buffer, which selects its memory on a
/// device whose buffers are host-mapped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum HostAccess {
    /// The host writes the buffer, or does not access it.
    Write,
    /// Device work writes the buffer and the host reads it back.
    ReadBack,
}

/// Slab sub-allocator front-end.
pub(super) struct SlabAllocator {
    inner: Arc<Inner>,
}

impl SlabAllocator {
    /// Build a `gpu-allocator` `Allocator` for the given device.
    ///
    /// # Errors
    /// Returns `BackendError` if `gpu-allocator` rejects the create
    /// description (typically only on a malformed instance/device pair,
    /// which we don't construct here, so this is unreachable in practice
    /// but propagated cleanly anyway).
    pub(super) fn new(device: Arc<DeviceCtx>) -> Result<Self> {
        // AllocatorDebugSettings is #[non_exhaustive] — build via the
        // `..Default::default()` style with a Default-constructed base
        // we then mutate.
        let mut debug_settings = gpu_allocator::AllocatorDebugSettings::default();
        debug_settings.log_memory_information = cfg!(feature = "gpu-validation");
        debug_settings.log_leaks_on_shutdown = cfg!(feature = "gpu-validation");

        let create_desc = AllocatorCreateDesc {
            instance: device.instance.clone(),
            device: device.device.clone(),
            physical_device: device.phys,
            debug_settings,
            buffer_device_address: true,
            allocation_sizes: gpu_allocator::AllocationSizes::default(),
        };
        let allocator = Allocator::new(&create_desc)
            .map_err(|e| BackendError::msg(format!("gpu_allocator::Allocator::new: {e}")))?;
        Ok(Self {
            inner: Arc::new(Inner {
                device,
                allocator: Mutex::new(allocator),
            }),
        })
    }

    /// Allocate a storage buffer of `size` bytes the host accesses as
    /// `access` says.
    ///
    /// A discrete GPU keeps it in `DEVICE_LOCAL` memory, which the host does
    /// not map: host access to its `HOST_VISIBLE` device memory crosses the
    /// bus uncached.  On an integrated GPU device memory is host memory, so
    /// the buffer is `HOST_VISIBLE` | `HOST_COHERENT` and persistently mapped
    /// (see [`DeviceBuffer::mapped`]), and the host writes and reads it in
    /// place instead of through a staging copy: [`HostAccess::Write`] in
    /// `DEVICE_LOCAL` memory where offered, which the host may map
    /// write-combined, [`HostAccess::ReadBack`] in `HOST_CACHED` memory
    /// where offered.
    pub(super) fn alloc_device(&self, size: usize, access: HostAccess) -> Result<DeviceBuffer> {
        let (location, name) = match (self.inner.device.integrated, access) {
            (false, _) => (MemoryLocation::GpuOnly, "rasterrocket device buffer"),
            (true, HostAccess::Write) => (MemoryLocation::CpuToGpu, "rasterrocket device buffer"),
            (true, HostAccess::ReadBack) => (
                MemoryLocation::GpuToCpu,
                "rasterrocket device readback buffer",
            ),
        };
        let (buffer, allocation) = self.create_and_bind(size, DEVICE_USAGE, location, name)?;
        let info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        // Safety: `buffer` is a freshly created handle bound to memory.
        let device_address = unsafe { self.inner.device.device.get_buffer_device_address(&info) };
        Ok(DeviceBuffer {
            buffer,
            size: u64::try_from(size).expect("size fits u64"),
            device_address,
            allocation: Some(allocation),
            parent: self.inner.clone(),
        })
    }

    /// Allocate a `HOST_VISIBLE` | `HOST_COHERENT` staging buffer of `size`
    /// bytes for host writes the device reads.  Persistently mapped — caller
    /// can read/write via [`HostBuffer::as_mut_slice`].
    pub(super) fn alloc_host(&self, size: usize) -> Result<HostBuffer> {
        self.alloc_mapped(
            size,
            MemoryLocation::CpuToGpu,
            "rasterrocket host staging buffer",
        )
    }

    /// Allocate a `HOST_VISIBLE` | `HOST_COHERENT` buffer of `size` bytes for
    /// device writes the host reads, in host-cached memory where the device
    /// offers it: reads from uncached memory cost one bus transaction each.
    pub(super) fn alloc_host_readback(&self, size: usize) -> Result<HostBuffer> {
        self.alloc_mapped(
            size,
            MemoryLocation::GpuToCpu,
            "rasterrocket host readback buffer",
        )
    }

    /// A persistently mapped host buffer of `size` bytes at `location`.
    fn alloc_mapped(
        &self,
        size: usize,
        location: MemoryLocation,
        name: &'static str,
    ) -> Result<HostBuffer> {
        let (buffer, allocation) = self.create_and_bind(size, HOST_USAGE, location, name)?;
        // gpu-allocator persistently maps host-visible allocations.  If the
        // pointer is missing, free what we have and report.
        let mapped_ptr = if let Some(nn) = allocation.mapped_ptr() {
            nn.as_ptr().cast::<u8>()
        } else {
            free_buffer(&self.inner, buffer, allocation, "host buffer rollback");
            return Err(BackendError::msg(
                "host buffer allocation has no mapped pointer (gpu-allocator returned None)",
            ));
        };
        Ok(HostBuffer {
            buffer,
            size: u64::try_from(size).expect("size fits u64"),
            mapped_ptr,
            allocation: Some(allocation),
            parent: self.inner.clone(),
        })
    }

    /// Free a device buffer.  Drop runs the actual free; the explicit
    /// method exists to match the trait shape (`CudaBackend::free_device`
    /// is similarly empty).
    #[expect(
        clippy::unused_self,
        reason = "shape-only impl; mirrors CudaBackend::free_device"
    )]
    pub(super) fn free_device(&self, _buf: DeviceBuffer) {}

    /// Free a host buffer.  Drop runs the actual free.
    #[expect(
        clippy::unused_self,
        reason = "shape-only impl; mirrors CudaBackend::free_host_pinned"
    )]
    pub(super) fn free_host(&self, _buf: HostBuffer) {}

    /// Create a `VkBuffer`, allocate memory for it, and bind them
    /// together.  Common to both device and host alloc paths.  On error
    /// in any step, rolls back cleanly so neither resource leaks.
    fn create_and_bind(
        &self,
        size: usize,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        name: &'static str,
    ) -> Result<(vk::Buffer, Allocation)> {
        let create_info = vk::BufferCreateInfo::default()
            .size(u64::try_from(size).expect("size fits u64"))
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        // Safety: create_info outlives this call.
        let buffer = unsafe { self.inner.device.device.create_buffer(&create_info, None) }
            .map_err(vk_err("vkCreateBuffer"))?;

        // Safety: `buffer` is freshly created and unbound.
        let requirements = unsafe {
            self.inner
                .device
                .device
                .get_buffer_memory_requirements(buffer)
        };

        let alloc_desc = AllocationCreateDesc {
            name,
            requirements,
            location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };

        let allocation = {
            let mut allocator = self
                .inner
                .allocator
                .lock()
                .expect("allocator mutex poisoned");
            match allocator.allocate(&alloc_desc) {
                Ok(a) => a,
                Err(e) => {
                    drop(allocator);
                    // Safety: buffer was freshly created above; no other handle exists.
                    unsafe {
                        self.inner.device.device.destroy_buffer(buffer, None);
                    }
                    return Err(BackendError::msg(format!("gpu_allocator::allocate: {e}")));
                }
            }
        };

        log::debug!(
            "vulkan: {name} of {size} bytes ({location:?}) in memory {:?}",
            allocation.memory_properties()
        );

        // Safety: buffer + allocation belong to the same device; the
        // allocation's memory + offset are valid per gpu-allocator's docs.
        let bind_result = unsafe {
            self.inner.device.device.bind_buffer_memory(
                buffer,
                allocation.memory(),
                allocation.offset(),
            )
        };
        if let Err(code) = bind_result {
            free_buffer(&self.inner, buffer, allocation, "create_and_bind rollback");
            return Err(BackendError::msg(format!(
                "vkBindBufferMemory failed: {code:?}"
            )));
        }

        Ok((buffer, allocation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_fill_size_skips_zero() {
        assert_eq!(validate_fill_size(0).unwrap(), FillAction::Skip);
    }

    #[test]
    fn validate_fill_size_accepts_aligned() {
        assert_eq!(validate_fill_size(4).unwrap(), FillAction::Fill(4));
        assert_eq!(validate_fill_size(4096).unwrap(), FillAction::Fill(4096));
        // u64::MAX is not a multiple of 4 (ends 0b11), so use a large
        // explicit aligned value instead.
        let big = (1u64 << 62) - 4;
        assert_eq!(validate_fill_size(big).unwrap(), FillAction::Fill(big));
    }

    #[test]
    fn validate_fill_size_rejects_unaligned() {
        for size in [1u64, 2, 3, 5, 6, 7, 17, 4095, 4097] {
            let err = validate_fill_size(size).unwrap_err();
            assert!(
                matches!(
                    err,
                    BackendError::UnalignedFill {
                        size: s,
                        required_alignment: 4,
                    } if s == size
                ),
                "expected UnalignedFill{{{size},4}}, got: {err:?}"
            );
        }
    }
}
