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

mod aa_fill;
pub use aa_fill::{AaFillScratch, aa_fill_page};
pub mod cuda;
pub mod params;
#[cfg(feature = "vulkan")]
pub mod vulkan;

use std::error::Error;
use std::fmt;

/// A backend-agnostic error type.
///
/// Variants exist for conditions that tests assert on (`matches!`) or
/// that benefit from typed log clarity. The long tail of errors goes
/// through [`BackendError::Other`] — Vulkan FFI failures, allocator
/// errors, init failures, numeric overflow, and free-form messages
/// constructed via [`BackendError::msg`].
///
/// `Display` is hand-rolled per variant; `Error::source` returns
/// `Some(&inner)` only for `Other` (typed variants are themselves the
/// source).
#[derive(Debug)]
pub enum BackendError {
    /// `alloc_*` was called with `size = 0`. `what` names the allocator
    /// (`"alloc_device"`, `"alloc_host_pinned"`, `"alloc_device_zeroed"`).
    ZeroSizeAlloc {
        /// Name of the rejecting allocator entry point, for diagnostic context.
        what: &'static str,
    },
    /// `vkCmdFillBuffer` requires a 4-byte-aligned size; `alloc_device_zeroed`
    /// (and other fill paths) surface this loudly rather than silently
    /// downgrading and leaving 1–3 trailing bytes non-zero.
    UnalignedFill {
        /// The offending size in bytes.
        size: u64,
        /// The backend's required alignment (always `4` for `vkCmdFillBuffer`).
        required_alignment: u64,
    },
    /// Recorder cap on descriptor sets per page was reached. `max` is the
    /// production limit (`MAX_DESC_SETS_PER_PAGE`).
    DescriptorPoolExhausted {
        /// Number of descriptor sets already allocated this page.
        allocated: u32,
        /// The backend's per-page cap.
        max: u32,
    },
    /// A `*Params::validate` invariant failed. `kind` names which
    /// invariant family (`"BlitInvariantViolation"`, …); `detail` is
    /// the human-readable "why".
    InvariantViolation {
        /// Stable name of the invariant family for `matches!` ergonomics.
        kind: &'static str,
        /// Human-readable description of the specific violation.
        detail: &'static str,
    },
    /// An explicit device selector did not resolve to exactly one usable
    /// device.  The message names the selector, the reason, and every
    /// enumerated device.  Distinct from [`BackendError::Other`] so callers
    /// can treat a configuration mistake as fatal even where an unavailable
    /// backend would fall back silently.
    DeviceSelection(String),
    /// Catch-all for any other backend error: Vulkan FFI failures,
    /// allocator errors, init failures, numeric overflow, free-form
    /// `BackendError::msg(...)`. `source()` returns the inner error.
    Other(Box<dyn Error + Send + Sync + 'static>),
}

impl BackendError {
    /// Wrap an arbitrary error as a `BackendError::Other`.
    pub fn new<E: Error + Send + Sync + 'static>(e: E) -> Self {
        Self::Other(Box::new(e))
    }

    /// Build a `BackendError::Other` from a free-form message.
    ///
    /// Convenience for callsites that don't have an underlying [`Error`]
    /// to wrap and don't fit any typed variant. Routes through `Other`,
    /// so `source()` returns the carrier.
    #[must_use]
    pub fn msg(message: impl Into<String>) -> Self {
        Self::Other(Box::new(MsgError(message.into())))
    }
}

/// Internal carrier used by [`BackendError::msg`] so the inner type
/// satisfies `Error + Send + Sync + 'static`.
#[derive(Debug)]
struct MsgError(String);

impl fmt::Display for MsgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for MsgError {}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSizeAlloc { what } => {
                write!(f, "{what} called with size = 0; backends require size > 0")
            }
            Self::UnalignedFill {
                size,
                required_alignment,
            } => write!(
                f,
                "fill size {size} is not a multiple of {required_alignment}"
            ),
            Self::DescriptorPoolExhausted { allocated, max } => write!(
                f,
                "descriptor pool exhausted: {allocated} sets allocated this page (max {max})"
            ),
            Self::InvariantViolation { kind, detail } => write!(f, "{kind}: {detail}"),
            Self::DeviceSelection(msg) => f.write_str(msg),
            Self::Other(e) => fmt::Display::fmt(e, f),
        }
    }
}

impl Error for BackendError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Other(e) => Some(&**e),
            _ => None,
        }
    }
}

/// Convenience alias for `Result<T, BackendError>`.
pub type Result<T> = std::result::Result<T, BackendError>;

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

/// Check a recorded copy of `len` bytes against the lengths of its source
/// and destination buffers.
pub(crate) fn check_copy_len(
    len: usize,
    src_len: usize,
    dst_len: usize,
    what: &'static str,
) -> Result<()> {
    if len > src_len || len > dst_len {
        return Err(BackendError::msg(format!(
            "{what}: copy of {len} bytes exceeds the source ({src_len} bytes) or the \
             destination ({dst_len} bytes)"
        )));
    }
    Ok(())
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
/// Calling `record_*` outside a `begin_page` / `submit_page` pair, or calling
/// `submit_page` twice without an intervening `begin_page`, is implementation-
/// defined behaviour: backends may panic, return an error, or silently produce
/// incorrect results. [`GpuBackend::run_page`] runs a whole page and is safe
/// from several threads; the individual calls require external
/// synchronisation even though the trait is `Send + Sync`.
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

    /// The backend's name in diagnostics: `"cuda"`, `"vulkan"`.
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

    /// Byte length of a device buffer previously returned by
    /// `alloc_device` / `alloc_device_zeroed`.
    ///
    /// Exposed so generic callers (e.g. `DeviceImageCache`) can size
    /// transfers without holding a parallel length field per entry.
    fn device_buffer_len(&self, buf: &Self::DeviceBuffer) -> usize;

    /// Free a device buffer previously returned by `alloc_device` or
    /// `alloc_device_zeroed`.
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
    /// Record an antialiased fill operation into the current page's command list.
    ///
    /// # Errors
    /// Returns `BackendError` if the operation cannot be recorded.
    fn record_aa_fill(&self, params: params::AaFillParams<'_, Self>) -> Result<()>;
    /// Record an ICC CLUT colour-transform into the current page's command list.
    ///
    /// # Errors
    /// Returns `BackendError` if the operation cannot be recorded.
    fn record_icc_clut(&self, params: params::IccClutParams<'_, Self>) -> Result<()>;
    /// Record a Porter-Duff source-over composite into the current page's command list.
    ///
    /// # Errors
    /// Returns `BackendError` if the operation cannot be recorded.
    fn record_composite(&self, params: params::CompositeParams<'_, Self>) -> Result<()>;
    /// Record a soft-mask application into the current page's command list.
    ///
    /// # Errors
    /// Returns `BackendError` if the operation cannot be recorded.
    fn record_apply_soft_mask(&self, params: params::SoftMaskParams<'_, Self>) -> Result<()>;

    /// Record one phase of the Blelloch exclusive scan kernel.
    ///
    /// The scan is built from three dispatches (selected by
    /// `params.phase`); callers must invoke all three in order
    /// (`PerWorkgroup` → `BlockSums` → `ScatterBlockSums`) for the
    /// scan to complete. All three share `params.data` /
    /// `params.block_sums` / `params.len_elems`, so the caller builds
    /// one `ScanParams` and updates only `phase` between dispatches.
    ///
    /// The shared shape (one trait method, phase enum in the params)
    /// avoids multiplying `record_*` trait methods 1-per-kernel-phase.
    ///
    /// # Errors
    /// Returns `BackendError` if the operation cannot be recorded or
    /// if `params.validate(self)` fails.
    fn record_scan(&self, params: params::ScanParams<'_, Self>) -> Result<()>;

    /// Record one phase of the parallel-Huffman JPEG decoder.
    ///
    /// One thread per subsequence walks the entropy-coded stream and
    /// writes a `(p, n, c, z)` tuple into `s_info[seq_idx]`. Phase 3
    /// of the Weißenberger algorithm is the Blelloch scan (see
    /// [`Self::record_scan`]); other phases are dispatched by
    /// selecting `params.phase`.
    ///
    /// # Errors
    /// Returns `BackendError` if the operation cannot be recorded or
    /// if `params.validate(self)` fails.
    fn record_huffman(&self, params: params::HuffmanParams<'_, Self>) -> Result<()>;
    /// Record IDCT + dequant + colour-conversion (Phase 5) into the current
    /// page's command list.
    ///
    /// Dispatches one workgroup per 8×8 block per component.  The kernel
    /// performs zigzag inverse, dequantisation, 2-D LLM IDCT, YCbCr→RGB
    /// colour conversion (T.871/JFIF full-range), and RGBA8 packing.
    ///
    /// # Errors
    /// Returns `BackendError` if the operation cannot be recorded or if
    /// `params.validate(self)` fails.
    fn record_idct(&self, params: params::IdctParams<'_, Self>) -> Result<()>;
    /// Record a zero-fill of `buf` into the current page's command list.
    ///
    /// Folds what [`Self::alloc_device_zeroed`] does as a separate one-shot
    /// submission into the page's own command buffer / stream: zero extra
    /// submits, no transfer-queue lock acquisition, no `vkQueueWaitIdle`
    /// in the alloc path.  Callers that have an active `begin_page` (e.g.
    /// the per-page composite buffer) should prefer this over
    /// `alloc_device_zeroed` to avoid serialising concurrent renderer
    /// threads on the Vulkan transfer pool.
    ///
    /// The fill becomes visible by the time the page's `wait_page`
    /// returns.
    ///
    /// # Errors
    /// Returns `BackendError` if the operation cannot be recorded.
    /// Vulkan's `vkCmdFillBuffer` requires the buffer size to be a multiple
    /// of 4; non-aligned sizes return [`BackendError::UnalignedFill`].
    fn record_zero_buffer(&self, buf: &Self::DeviceBuffer) -> Result<()>;
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
        dst: &Self::DeviceBuffer,
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
    /// [`Self::alloc_host_pinned`].
    fn host_buffer_len(&self, buf: &Self::HostBuffer) -> usize;

    /// The host view of `buf`'s bytes.
    ///
    /// The view reflects device writes only after the fence of the page
    /// that recorded them has been waited on.
    fn host_bytes<'a>(&self, buf: &'a Self::HostBuffer) -> &'a [u8];

    /// The writable host view of `buf`'s bytes; see [`Self::host_bytes`].
    fn host_bytes_mut<'a>(&self, buf: &'a mut Self::HostBuffer) -> &'a mut [u8];

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

    /// Run one page: `begin_page`, the work `record` records, `submit_page`
    /// and `wait_page`, returning once the device has finished it.
    ///
    /// This is the page lifecycle for a backend shared between threads: a
    /// backend with one recording state serialises whole pages, so a
    /// second thread's page waits instead of failing, and a failure at any
    /// step abandons the page and returns the backend to idle, so one
    /// failed page does not fail every later one.
    ///
    /// # Errors
    /// The first error of any step.
    fn run_page<F>(&self, record: F) -> Result<()>
    where
        F: FnOnce(&Self) -> Result<()>,
        Self: Sized,
    {
        self.begin_page()?;
        record(self)?;
        let fence = self.submit_page()?;
        self.wait_page(fence)
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
    fn upload(&self, dst: &Self::DeviceBuffer, src: &[u8]) -> Result<()>;

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

    /// Query the current VRAM budget from the driver.
    ///
    /// # Errors
    /// Returns `BackendError` if the driver query fails.
    fn detect_vram_budget(&self) -> Result<VramBudget>;
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
