//! GPU backend abstraction layer.
//!
//! The `GpuBackend` trait is the single seam between the renderer and any
//! concrete GPU implementation. Implementations are added as sibling modules
//! (`backend::cuda`, `backend::vulkan`).
//!
//! Per-page semantics: callers use `begin_page` → `record_*` → `submit_page`
//! → `wait_page` to batch a page's GPU work into a single submission, so the
//! host blocks once per page instead of once per kernel.
//!
//! Pointer-passing ABI: device buffers are passed by reference into
//! `record_*` calls; the backend resolves them to native pointers (CUDA) or
//! buffer-device-addresses (Vulkan) at record time.

#[cfg(feature = "cache")]
mod blit_page;
#[cfg(all(feature = "d3d12", any(test, windows)))]
mod dxil;
mod error;
#[doc(hidden)]
pub mod layout;
mod scratch;
mod staged;
#[cfg(feature = "cache")]
pub use blit_page::{BlitScratch, BlitSource, blit_image, blit_image_box};
pub use scratch::{DeviceScratch, ScratchPool};
mod icc_clut;
#[cfg(any(feature = "vulkan", feature = "d3d12"))]
#[cfg_attr(
    not(any(feature = "vulkan", all(windows, feature = "d3d12"), test)),
    expect(dead_code, reason = "only the Vulkan and D3D12 backends consume it")
)]
mod kernel_status;
pub use icc_clut::{IccClutScratch, icc_clut, icc_clut_matches_cpu, icc_clut16, random_grid};
pub mod cuda;
#[cfg(feature = "d3d12")]
pub mod d3d12;
#[cfg(any(feature = "vulkan", feature = "d3d12"))]
pub mod page_device;
#[cfg(any(feature = "vulkan", feature = "d3d12"))]
#[cfg_attr(
    not(any(feature = "vulkan", all(windows, feature = "d3d12"))),
    expect(dead_code, reason = "only the Vulkan and D3D12 backends consume it")
)]
mod page_job;
pub mod params;
#[cfg(any(feature = "vulkan", feature = "d3d12"))]
pub mod select;
#[cfg(feature = "vulkan")]
pub mod vulkan;

#[cfg(any(feature = "vulkan", feature = "d3d12"))]
pub use error::DEVICE_LOST_CAUSE;
pub use error::{BackendError, Result, SelectionFailure};

/// Reject zero-size allocation requests with a clear message.
///
/// Both Vulkan (`vkAllocateMemory` rejects 0) and CUDA (`cuMemAlloc(0)` returns
/// `CUDA_ERROR_INVALID_VALUE`) refuse zero-size allocations; pre-checking gives
/// a uniform, callable-side error rather than a driver-specific status.
pub(crate) const fn reject_zero_size(size: usize, what: &'static str) -> Result<()> {
    if size == 0 {
        return Err(BackendError::ZeroSizeAlloc { what });
    }
    Ok(())
}

/// `len` as a device copy size, after checking that it fits `capacity`, the
/// byte size of the buffer that is the copy's `side` (`"source"` or
/// `"destination"`); `op` names the operation in the error.
pub(crate) fn copy_len(
    len: usize,
    capacity: u64,
    op: &'static str,
    side: &'static str,
) -> Result<u64> {
    u64::try_from(len)
        .ok()
        .filter(|&n| n <= capacity)
        .ok_or_else(|| {
            BackendError::msg(format!(
                "{op}: copy of {len} bytes exceeds the {side} buffer ({capacity} bytes)"
            ))
        })
}

/// Check a dispatch of `kernel` over `groups` thread groups against the
/// device's group count per dimension, `max`, named `limit`.
///
/// # Errors
/// [`BackendError::DeviceLimit`] in thread groups, for the first dimension
/// above its maximum.
#[cfg(any(feature = "vulkan", feature = "d3d12"))]
#[cfg_attr(
    not(any(feature = "vulkan", all(windows, feature = "d3d12"), test)),
    expect(dead_code, reason = "only the Vulkan and D3D12 backends consume it")
)]
pub(crate) fn check_dispatch_groups(
    kernel: &'static str,
    groups: (u32, u32, u32),
    max: [u32; 3],
    limit: &'static str,
) -> Result<()> {
    let need: [u32; 3] = groups.into();
    match need.iter().zip(max).find(|&(&n, m)| n > m) {
        Some((&need, max)) => Err(BackendError::DeviceLimit {
            what: kernel,
            need: need.into(),
            limit,
            max: max.into(),
            unit: "thread groups",
        }),
        None => Ok(()),
    }
}

/// The size of a recorded copy of `len` bytes from a buffer of `src` bytes
/// to one of `dst` bytes, `None` for an empty copy.
pub(crate) fn recorded_copy_len(
    len: usize,
    src: u64,
    dst: u64,
    op: &'static str,
) -> Result<Option<u64>> {
    copy_len(len, src, op, "source").map(drop)?;
    let n = copy_len(len, dst, op, "destination")?;
    Ok((n > 0).then_some(n))
}

/// Live VRAM budget snapshot returned by `GpuBackend::detect_vram_budget`.
///
/// Construct via [`VramBudget::new`] to enforce the `usable_bytes <= total_bytes`
/// invariant; the public fields are kept for convenient pattern-matching but
/// hand-construction skips the assertion.
#[derive(Debug, Clone, Copy)]
pub struct VramBudget {
    /// Total VRAM on the device in bytes.
    pub total_bytes: u64,
    /// VRAM that allocations may safely consume in this process.
    ///
    /// Backends may apply a safety margin (e.g., a fraction of `free` reported
    /// by the driver) to account for fragmentation and other processes'
    /// allocations. Always `<= total_bytes`.
    pub usable_bytes: u64,
}

impl VramBudget {
    /// Construct a `VramBudget`, asserting `usable <= total`.
    ///
    /// # Panics
    /// Panics if `usable > total` — backends must apply their safety margin
    /// before calling this constructor, not let the caller derive a value that
    /// could exceed total VRAM.
    #[must_use]
    pub const fn new(total_bytes: u64, usable_bytes: u64) -> Self {
        assert!(
            usable_bytes <= total_bytes,
            "VramBudget invariant violated: usable_bytes must not exceed total_bytes"
        );
        Self {
            total_bytes,
            usable_bytes,
        }
    }
}

/// Abstraction over a concrete GPU implementation.
///
/// # Per-page state machine
///
/// `record_*` methods accumulate work into a per-backend command list. Callers
/// must follow the order:
///
/// ```text
/// begin_page() → [record_*()…] → submit_page() → wait_page(fence)
/// ```
///
/// Calling `record_*` outside a `begin_page` / `submit_page` pair, calling
/// `begin_page` while a page is being recorded, or calling `submit_page`
/// without a page is rejected with an error.  CUDA keeps that state per
/// thread, since a page is recorded on its thread's stream; Vulkan keeps
/// one state for the backend. [`GpuBackend::run_page`] runs a whole page
/// and is safe from several threads; the individual calls require external
/// synchronisation even though the trait is `Send + Sync`.
///
/// Every operation takes a device buffer it writes by `&mut` and one it
/// only reads by `&`, so a backend that orders work across queues by
/// recorded reads and writes sees every write.
///
/// # Stability — closed implementation set
///
/// The trait is `pub` for callers (`rasterrocket`, `rasterrocket-interp`) but the
/// implementation set is closed to in-tree backends (`CudaBackend`,
/// `VulkanBackend`): its supertrait `Sealed` is private to this crate, so
/// no other crate can implement it.  The trait surface co-evolves with
/// the in-tree backends and carries no stability commitment for external
/// implementors.
pub trait GpuBackend: Send + Sync + sealed::Sealed {
    /// An opaque device-resident buffer handle.
    type DeviceBuffer: Send + Sync;
    /// An opaque host-pinned buffer handle.
    type HostBuffer: Send + Sync;
    /// A synchronisation primitive returned by `submit_page`.
    ///
    /// `Clone` is required so callers can stash a fence inside a long-
    /// lived structure (e.g. `DeviceImageCache` keeps one per cached
    /// image to gate `free_device` until the last in-flight DMA
    /// completes) while a separate render path also waits on it.
    /// CUDA wraps `Arc<cudaEvent_t>`; Vulkan wraps `Arc<VkFence>`.
    type PageFence: Send + Sync + Clone;

    /// The backend's name in diagnostics: `"cuda"`, `"vulkan"`, `"d3d12"`.
    const NAME: &'static str;

    /// Allocate `size` bytes of device memory.
    ///
    /// `size` must be greater than zero; passing `0` returns a `BackendError`
    /// because both Vulkan (`vkAllocateMemory` rejects 0) and CUDA
    /// (`cuMemAlloc(0)` returns `CUDA_ERROR_INVALID_VALUE`) fail this call.
    /// Implementations should pre-check rather than fall through to the driver.
    ///
    /// # Errors
    /// Returns `BackendError` if `size == 0`, or if the device allocation fails
    /// (OOM or driver error).
    fn alloc_device(&self, size: usize) -> Result<Self::DeviceBuffer>;

    /// Allocate `size` bytes of device memory and zero-initialise them.
    ///
    /// Device work recorded and downloads issued after the call read
    /// zeros: CUDA queues the fill on the calling thread's stream, ahead of
    /// that thread's later work; Vulkan waits for a one-shot fill
    /// submission before returning.
    ///
    /// `DevicePageBuffer` relies on this: every page allocates a fresh
    /// zeroed RGBA8 buffer that the blit kernel only writes touched
    /// pixels into; reading `(0,0,0,0)` from un-touched pixels is the
    /// "transparent" signal for the host-side composite.
    ///
    /// # Errors
    /// Returns `BackendError` if `size == 0`, or if the allocation /
    /// zero-fill fails.
    fn alloc_device_zeroed(&self, size: usize) -> Result<Self::DeviceBuffer>;

    /// Allocate `size` bytes of device memory that device work writes and
    /// the host reads back ([`Self::download`], [`Self::device_bytes`]).
    ///
    /// On a device whose buffers are host-mapped
    /// ([`Self::device_memory_is_host_mapped`]) the buffer is placed in
    /// host-cached memory where the device offers it, whereas
    /// [`Self::alloc_device`] may return memory the host reads uncached, one
    /// bus transaction per access.  Elsewhere the two are the same.  Same
    /// zero-size constraints as `alloc_device`; contents are unspecified.
    ///
    /// # Errors
    /// Returns `BackendError` if `size == 0`, or if the device allocation fails.
    fn alloc_device_readback(&self, size: usize) -> Result<Self::DeviceBuffer>;

    /// Byte length of a device buffer previously returned by
    /// `alloc_device`, `alloc_device_zeroed` or `alloc_device_readback`.
    ///
    /// Exposed so generic callers (e.g. `DeviceImageCache`) can size
    /// transfers without holding a parallel length field per entry.
    fn device_buffer_len(&self, buf: &Self::DeviceBuffer) -> usize;

    /// Free a device buffer previously returned by `alloc_device`,
    /// `alloc_device_zeroed` or `alloc_device_readback`.
    ///
    /// **Deferred-free contract:** the caller must ensure no
    /// outstanding `PageFence` (from `submit_page`) still references this
    /// buffer. CUDA's stream-recorded events make a same-stream free
    /// always safe; Vulkan has no equivalent and `vkDestroyBuffer`
    /// while a copy is in-flight is undefined behaviour. Calling
    /// `wait_page` on every fence the buffer
    /// participated in before `free_device` is always safe; the
    /// backend may otherwise treat such a call as undefined behaviour
    /// or a panic.
    fn free_device(&self, buf: Self::DeviceBuffer);

    /// Allocate `size` bytes of host-pinned (DMA-accessible) memory for host
    /// writes the device reads.
    ///
    /// `size` must be greater than zero; the same zero-size constraints as
    /// `alloc_device` apply.
    ///
    /// # Errors
    /// Returns `BackendError` if `size == 0`, or if the pinned allocation fails
    /// (OOM or driver error).
    fn alloc_host_pinned(&self, size: usize) -> Result<Self::HostBuffer>;
    /// Allocate `size` bytes of host-pinned memory for device writes the
    /// host reads back.
    ///
    /// [`Self::alloc_host_pinned`] may return memory the host reads
    /// uncached, one bus transaction per access (Vulkan's host-visible
    /// upload heaps); this allocation is host-cached where the device
    /// offers it.  Same zero-size constraints as `alloc_device`.
    ///
    /// # Errors
    /// Returns `BackendError` if `size == 0`, or if the pinned allocation fails.
    fn alloc_host_readback(&self, size: usize) -> Result<Self::HostBuffer>;

    /// Free a host-pinned buffer previously returned by `alloc_host_pinned`
    /// or `alloc_host_readback`.
    ///
    /// Same deferred-free contract as `free_device`: the caller must
    /// ensure no outstanding `PageFence` (from
    /// `submit_page`) still
    /// references this buffer.
    fn free_host_pinned(&self, buf: Self::HostBuffer);

    /// Begin accumulating GPU work for a new page.
    ///
    /// # Errors
    /// Returns `BackendError` if the backend cannot initialise the per-page
    /// command state (e.g. stream creation failure).
    fn begin_page(&self) -> Result<()>;
    /// Record an image blit operation into the current page's command list.
    ///
    /// # Errors
    /// Returns `BackendError` if the operation cannot be recorded.
    fn record_blit_image(&self, params: params::BlitParams<'_, Self>) -> Result<()>;
    /// Record an ICC CLUT colour-transform into the current page's command list.
    ///
    /// # Errors
    /// Returns `BackendError` if the operation cannot be recorded.
    fn record_icc_clut(&self, params: params::IccClutParams<'_, Self>) -> Result<()>;

    /// Record a zero-fill of the first `len` bytes of `buf` into the current
    /// page's command list.
    ///
    /// Folds what [`Self::alloc_device_zeroed`] does as a separate one-shot
    /// submission into the page's own command buffer / stream: zero extra
    /// submits, no transfer-pool lock acquisition, no host wait in the
    /// alloc path.  Callers that have an active `begin_page` (e.g.
    /// the per-page composite buffer) should prefer this over
    /// `alloc_device_zeroed` to avoid serialising concurrent renderer
    /// threads on the Vulkan transfer pool.
    ///
    /// The fill becomes visible by the time the page's `wait_page`
    /// returns.
    ///
    /// # Errors
    /// Returns `BackendError` if the operation cannot be recorded.
    /// `len` beyond the buffer's length is an error.  Vulkan's
    /// `vkCmdFillBuffer` requires the size to be a multiple of 4; non-aligned
    /// sizes return [`BackendError::UnalignedFill`].
    fn record_zero_buffer(&self, buf: &mut Self::DeviceBuffer, len: usize) -> Result<()>;
    /// Record a copy of `src[..len]` into `dst[..len]` into the current
    /// page's command list.
    ///
    /// The copy runs in page order: device work recorded after it reads the
    /// copied bytes.  The device reads `src` asynchronously, at any time
    /// from this call until the page's fence has been waited on (a CUDA
    /// stream may start the copy before `submit_page`).  A zero `len`
    /// records nothing.
    ///
    /// # Safety
    /// From this call until the fence of the page that records it has been
    /// waited on, `src` is neither written, through [`Self::host_bytes_mut`]
    /// or otherwise, nor freed.  A page abandoned by an error of
    /// [`Self::run_page`] has been waited on when `run_page` returns.
    ///
    /// # Errors
    /// Returns `BackendError` if no page is being recorded, if `len`
    /// exceeds either buffer, or if the copy cannot be recorded.
    unsafe fn record_copy_to_device(
        &self,
        src: &Self::HostBuffer,
        dst: &mut Self::DeviceBuffer,
        len: usize,
    ) -> Result<()>;

    /// Record a copy of `src[..len]` into `dst[..len]` into the current
    /// page's command list.
    ///
    /// The copy runs in page order, after the device work recorded before
    /// it, and writes `dst` asynchronously.  A zero `len` records nothing.
    ///
    /// # Safety
    /// From this call until the fence of the page that records it has been
    /// waited on, `dst` is neither read, written nor freed.  A page
    /// abandoned by an error of [`Self::run_page`] has been waited on when
    /// `run_page` returns.
    ///
    /// # Errors
    /// Returns `BackendError` if no page is being recorded, if `len`
    /// exceeds either buffer, or if the copy cannot be recorded.
    unsafe fn record_copy_to_host(
        &self,
        src: &Self::DeviceBuffer,
        dst: &mut Self::HostBuffer,
        len: usize,
    ) -> Result<()>;

    /// Whether device buffers live in host memory the host reaches in place
    /// ([`Self::device_bytes`]): true on an integrated GPU whose backend
    /// maps them.
    fn device_memory_is_host_mapped(&self) -> bool;

    /// The bytes of a device buffer in place, when the device shares host
    /// memory ([`Self::device_memory_is_host_mapped`]); `None` when they are
    /// reachable only through copies.
    ///
    /// # Safety
    /// While the returned view lives, no device work that writes `buf` is
    /// recorded-and-submitted without its fence having been waited on, and
    /// no other thread writes `buf` ([`Self::upload`],
    /// [`Self::device_bytes_mut`]).
    unsafe fn device_bytes<'a>(&self, buf: &'a Self::DeviceBuffer) -> Option<&'a [u8]>;

    /// The writable bytes of a device buffer in place; see
    /// [`Self::device_bytes`].  Writes are seen by device work submitted
    /// afterwards.
    ///
    /// # Safety
    /// While the returned view lives, no submitted device work whose fence
    /// has not been waited on accesses `buf`, and no other thread reads it
    /// ([`Self::download`]).
    unsafe fn device_bytes_mut<'a>(&self, buf: &'a mut Self::DeviceBuffer) -> Option<&'a mut [u8]>;

    /// Byte length of a host buffer returned by
    /// [`Self::alloc_host_pinned`] or [`Self::alloc_host_readback`].
    fn host_buffer_len(&self, buf: &Self::HostBuffer) -> usize;

    /// The host view of `buf`'s bytes.
    ///
    /// The view reflects device writes only after the fence of the page
    /// that recorded them has been waited on.  A backend that tracks the
    /// copies recorded against `buf` waits for them first.
    ///
    /// # Errors
    /// Returns `BackendError` if waiting for such a copy fails.
    fn host_bytes<'a>(&self, buf: &'a Self::HostBuffer) -> Result<&'a [u8]>;

    /// The writable host view of `buf`'s bytes; see [`Self::host_bytes`].
    ///
    /// # Errors
    /// As [`Self::host_bytes`].
    fn host_bytes_mut<'a>(&self, buf: &'a mut Self::HostBuffer) -> Result<&'a mut [u8]>;

    /// Submit all recorded work for the current page; returns a fence to wait on.
    ///
    /// # Errors
    /// Returns `BackendError` if the submission fails.
    fn submit_page(&self) -> Result<Self::PageFence>;
    /// Block the calling thread until the submitted page work is complete.
    ///
    /// # Errors
    /// Returns `BackendError` if the fence wait fails or the GPU raised an error.
    fn wait_page(&self, fence: Self::PageFence) -> Result<()>;

    /// Abandon the page the calling thread has begun or submitted, if any.
    ///
    /// Returns once no work that page queued is still running, with the
    /// backend ready for a new [`Self::begin_page`]; buffers the page
    /// referenced may then be accessed and freed.  A failure to wait is
    /// logged at `warn`, the page is abandoned regardless.  Requires the
    /// same external synchronisation as the individual page calls.
    fn abandon_page(&self);

    /// Run one page: `begin_page`, the work `record` records, `submit_page`
    /// and `wait_page`, returning once the device has finished it.
    ///
    /// This is the page lifecycle for a backend shared between threads: a
    /// backend with one recording state serialises whole pages, so a
    /// second thread's page waits instead of failing.  A failure at any
    /// step, or an unwinding `record`, abandons the page
    /// ([`Self::abandon_page`]) before `run_page` returns or unwinds, so
    /// no work of the page is still running then and one failed page does
    /// not fail every later one.
    ///
    /// # Errors
    /// The first error of any step.
    fn run_page<F>(&self, record: F) -> Result<()>
    where
        F: FnOnce(&Self) -> Result<()>,
        Self: Sized,
    {
        run_page_steps(self, record)
    }

    /// Copy `src` into `dst[..src.len()]`.
    ///
    /// Returns once the copy has completed: `src` may be overwritten or
    /// freed as soon as the call returns, whatever kind of host memory
    /// backs it, including memory from [`Self::alloc_host_pinned`]. Device
    /// work recorded afterwards reads the copied bytes.
    ///
    /// `src.len()` must not exceed the device-side capacity of `dst`;
    /// an oversize source is rejected, never truncated. An empty `src`
    /// is a no-op.
    ///
    /// # Errors
    /// Returns `BackendError` if the copy fails, or if `src.len()`
    /// exceeds `dst`'s device capacity.
    fn upload(&self, dst: &mut Self::DeviceBuffer, src: &[u8]) -> Result<()>;

    /// Copy `dst.len()` bytes from the start of `src` into `dst`.
    ///
    /// Returns once the copy has completed and `dst` holds the device
    /// bytes; no transfer outlives the borrow of `dst`.  An empty `dst` is
    /// a no-op.
    ///
    /// # Errors
    /// Returns `BackendError` if the copy fails, or if `dst.len()`
    /// exceeds `src`'s device length.
    fn download(&self, src: &Self::DeviceBuffer, dst: &mut [u8]) -> Result<()>;

    /// The scratch buffers of the operations this backend runs ([`icc_clut()`]
    /// and, with the `cache` feature, `blit_image`).
    fn scratch(&self) -> &DeviceScratch<Self>
    where
        Self: Sized;

    /// Query the current VRAM budget from the driver.
    ///
    /// # Errors
    /// Returns `BackendError` if the driver query fails.
    fn detect_vram_budget(&self) -> Result<VramBudget>;
}

/// `begin_page`, `record`, `submit_page` and `wait_page` on `backend`, with
/// [`GpuBackend::abandon_page`] run when a step fails or `record` unwinds:
/// the body of [`GpuBackend::run_page`] for every backend.
pub(crate) fn run_page_steps<B, F>(backend: &B, record: F) -> Result<()>
where
    B: GpuBackend,
    F: FnOnce(&B) -> Result<()>,
{
    /// Abandons the page when dropped armed.
    struct AbandonOnExit<'a, B: GpuBackend> {
        backend: &'a B,
        armed: bool,
    }

    impl<B: GpuBackend> Drop for AbandonOnExit<'_, B> {
        fn drop(&mut self) {
            if self.armed {
                self.backend.abandon_page();
            }
        }
    }

    let mut abandon = AbandonOnExit {
        backend,
        armed: true,
    };
    let result = backend.begin_page().and_then(|()| {
        record(backend)?;
        let fence = backend.submit_page()?;
        backend.wait_page(fence)
    });
    abandon.armed = result.is_err();
    result
}

/// The supertrait that closes [`GpuBackend`] to the backends of this
/// crate.
mod sealed {
    /// Implemented by the in-tree backends only.
    pub trait Sealed {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vram_budget_new_accepts_usable_le_total() {
        let b = VramBudget::new(100, 75);
        assert_eq!(b.total_bytes, 100);
        assert_eq!(b.usable_bytes, 75);
    }

    #[test]
    fn vram_budget_new_accepts_equal() {
        let b = VramBudget::new(100, 100);
        assert_eq!(b.total_bytes, 100);
        assert_eq!(b.usable_bytes, 100);
    }

    #[test]
    fn vram_budget_new_accepts_zero() {
        // Degenerate but valid: a CPU-only backend may report no VRAM at all.
        let b = VramBudget::new(0, 0);
        assert_eq!(b.total_bytes, 0);
        assert_eq!(b.usable_bytes, 0);
    }

    #[test]
    #[should_panic(expected = "VramBudget invariant violated")]
    fn vram_budget_new_panics_when_usable_exceeds_total() {
        let _ = VramBudget::new(100, 101);
    }

    #[test]
    fn reject_zero_size_passes_nonzero() {
        assert!(reject_zero_size(1, "test").is_ok());
        assert!(reject_zero_size(usize::MAX, "test").is_ok());
    }

    #[test]
    fn reject_zero_size_returns_error_for_zero() {
        let err = reject_zero_size(0, "alloc_device").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("alloc_device"), "missing context: {msg}");
        assert!(msg.contains("size = 0"), "missing diagnostic: {msg}");
    }

    /// A dispatch above the group count of any dimension is a device
    /// limit, in thread groups, which keeps the device.
    #[cfg(any(feature = "vulkan", feature = "d3d12"))]
    #[test]
    fn a_dispatch_above_the_group_count_is_a_device_limit() {
        let max = [65_535, 65_535, 1];
        check_dispatch_groups("k", (65_535, 65_535, 1), max, "L").expect("at the limit");
        for groups in [(65_536, 1, 1), (1, 65_536, 1), (1, 1, 2)] {
            let err = check_dispatch_groups("k", groups, max, "L").expect_err("above");
            assert!(
                matches!(
                    err,
                    BackendError::DeviceLimit { what: "k", limit: "L", unit: "thread groups", need, max }
                        if need == max + 1
                ),
                "{err:?}"
            );
            assert!(err.to_string().contains("thread groups"), "{err}");
        }
    }

    #[test]
    fn backend_error_msg_displays_verbatim() {
        let err = BackendError::msg("widget broke");
        assert_eq!(err.to_string(), "widget broke");
    }

    #[test]
    fn backend_error_msg_accepts_string_and_str() {
        let _from_str = BackendError::msg("static");
        let _from_string = BackendError::msg(String::from("owned"));
        let _from_format = BackendError::msg(format!("formatted {}", 42));
    }
}
