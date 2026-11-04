//! Page-locked, host-cached memory.
//!
//! The allocation never requests write-combining: host reads from
//! write-combined memory are uncached, one bus transaction per access, and
//! host-to-device copies of a few kilobytes from it measured several times
//! slower than from cached page-locked memory, so it serves neither
//! direction of the kernels' transfers better.

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

use cudarc::driver::{CudaContext, result};

use super::be;
use crate::backend::Result;

/// `len` bytes of page-locked, host-cached memory, zero-initialised, freed
/// on drop.
pub struct PinnedBuffer {
    ptr: NonNull<u8>,
    len: usize,
    ctx: Arc<CudaContext>,
}

// SAFETY: the buffer owns its allocation; shared access is read-only through
// `&self` and mutation requires `&mut self`, so the usual borrow rules
// exclude data races between host threads.
unsafe impl Send for PinnedBuffer {}
// SAFETY: as for `Send`.
unsafe impl Sync for PinnedBuffer {}

impl std::fmt::Debug for PinnedBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedBuffer")
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl PinnedBuffer {
    /// Allocate `len > 0` bytes in `ctx` and zero them.
    pub(super) fn new(ctx: &Arc<CudaContext>, len: usize) -> Result<Self> {
        ctx.bind_to_thread().map_err(be)?;
        // SAFETY: the context is current on this thread; the returned
        // pointer is checked for null before use.  Flags 0: page-locked,
        // cached, not portable across contexts.
        let raw = unsafe { result::malloc_host(len, 0) }.map_err(be)?;
        let ptr = NonNull::new(raw.cast::<u8>())
            .ok_or_else(|| crate::backend::BackendError::msg("cuMemHostAlloc returned null"))?;
        // SAFETY: `ptr` addresses `len` writable bytes just allocated.
        unsafe { ptr.as_ptr().write_bytes(0, len) };
        Ok(Self {
            ptr,
            len,
            ctx: Arc::clone(ctx),
        })
    }

    /// Length in bytes.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer holds no bytes; never true for an allocated buffer.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The buffer's bytes.
    #[must_use]
    pub const fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` addresses `len` initialised bytes owned by `self`.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// The buffer's bytes, writable.
    pub const fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as `as_slice`, with `&mut self` excluding other views.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for PinnedBuffer {
    /// A failure cannot be returned from `drop`; it is logged at `warn`, as
    /// a leaked page-locked allocation or an unusable context is.
    fn drop(&mut self) {
        if let Err(e) = self.ctx.bind_to_thread() {
            log::warn!("PinnedBuffer drop: binding the CUDA context failed: {e}");
        }
        // SAFETY: `ptr` was returned by `cuMemHostAlloc` and is freed once.
        if let Err(e) = unsafe { result::free_host(self.ptr.as_ptr().cast::<c_void>()) } {
            log::warn!(
                "PinnedBuffer drop: cuMemFreeHost of {} bytes failed: {e}",
                self.len
            );
        }
    }
}
