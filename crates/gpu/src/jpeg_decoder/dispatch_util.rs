//! Shared helpers for one-shot test/oracle dispatchers in this
//! module.
//!
//! The dispatchers (`scan::dispatch_blelloch_scan`,
//! `huffman::dispatch_phase1_intra_sync`, etc.) follow the same
//! shape: alloc several device buffers → upload → record → submit
//! → wait → download → free.
//!
//! ## Why a guard if the backend's `free_device` is shape-only
//!
//! Both the CUDA and Vulkan backends' `free_device` impls are
//! currently `Self::DeviceBuffer`-take-only with empty bodies —
//! the real free happens in the buffer's own `Drop` (cudarc's
//! `CudaSlice::Drop` calls `cuMemFreeAsync`; Vulkan's
//! `DeviceBuffer::Drop` returns the allocation to gpu-allocator).
//! So a missing `free_device` on an error path does **not** leak
//! memory today.
//!
//! The guard still earns its keep because:
//!
//! - **Structural clarity.** The alloc-frees-on-error contract is
//!   visible at the call site rather than implicit in the buffer
//!   type's `Drop`.
//! - **Future-proofing.** Any backend that ever moves real work
//!   into `free_device` (e.g., a pool-returner, a debug-leak
//!   tracker, or a `record_zero_buffer`-on-free policy) doesn't
//!   leak when an upstream `?` skips the trailing `free_device`.
//! - **One source of ownership.** Without the guard, the only thing
//!   keeping the buffer alive is its local binding; with the guard,
//!   ownership is explicitly handed back via `.take()` on the happy
//!   path. The two states are distinguishable.
//!
//! `DeviceBufferGuard` wraps a `B::DeviceBuffer` and, on Drop with
//! `Some(...)`, hands it to `backend.free_device`. `.take()` empties
//! the Option so Drop is a no-op; the caller then owns the raw
//! buffer and frees it itself.

use crate::backend::{GpuBackend, Result};

/// RAII guard around an allocated `B::DeviceBuffer`. On Drop the
/// buffer is handed to the backend via `free_device` — unless
/// `take()` has already been called, in which case Drop is a
/// no-op and the caller owns the raw buffer.
///
/// The guard borrows the backend immutably so multiple guards can
/// coexist (each dispatcher allocates several buffers concurrently).
///
/// `#[must_use]` because a freshly-allocated and then immediately
/// dropped guard is always a bug — the device alloc work was wasted.
#[must_use = "DeviceBufferGuard holds a freshly-allocated device buffer; \
              dropping it without using it wastes the alloc + drives an \
              immediate device-side free"]
pub(super) struct DeviceBufferGuard<'a, B: GpuBackend + ?Sized> {
    backend: &'a B,
    buf: Option<B::DeviceBuffer>,
}

impl<'a, B: GpuBackend + ?Sized> DeviceBufferGuard<'a, B> {
    /// Allocate `size` bytes of (uninitialised) device memory and
    /// wrap the result in a Drop-frees guard.
    ///
    /// # Errors
    /// Propagates whatever `backend.alloc_device` returns.
    pub(super) fn alloc(backend: &'a B, size: usize) -> Result<Self> {
        Ok(Self {
            backend,
            buf: Some(backend.alloc_device(size)?),
        })
    }

    /// Allocate `size` bytes of zeroed device memory and wrap the
    /// result in a Drop-frees guard.
    ///
    /// # Errors
    /// Propagates whatever `backend.alloc_device_zeroed` returns.
    //
    pub(super) fn alloc_zeroed(backend: &'a B, size: usize) -> Result<Self> {
        Ok(Self {
            backend,
            buf: Some(backend.alloc_device_zeroed(size)?),
        })
    }

    /// Allocate `size` bytes of device memory that the host reads back
    /// ([`GpuBackend::alloc_device_readback`]); contents unspecified.
    ///
    /// # Errors
    /// Propagates whatever `backend.alloc_device_readback` returns.
    pub(super) fn alloc_readback(backend: &'a B, size: usize) -> Result<Self> {
        Ok(Self {
            backend,
            buf: Some(backend.alloc_device_readback(size)?),
        })
    }

    /// Allocate `size` bytes of zeroed device memory that the host reads
    /// back.  On a device whose buffers the host reaches in place the
    /// read-back buffer is zeroed in place; elsewhere it is the memory of
    /// [`Self::alloc_zeroed`].
    ///
    /// # Errors
    /// Propagates the allocation's error; fails if a host-mapped device
    /// returns a buffer the host cannot reach.
    pub(super) fn alloc_readback_zeroed(backend: &'a B, size: usize) -> Result<Self> {
        if !backend.device_memory_is_host_mapped() {
            return Self::alloc_zeroed(backend, size);
        }
        let mut buf = backend.alloc_device_readback(size)?;
        // SAFETY: the buffer was just allocated: no device work references
        // it and no other thread holds it.
        let zeroed = unsafe { backend.device_bytes_mut(&mut buf) }.map(|bytes| bytes.fill(0));
        let guard = Self {
            backend,
            buf: Some(buf),
        };
        zeroed.ok_or_else(|| {
            crate::backend::BackendError::msg(
                "alloc_readback_zeroed: a host-mapped device returned an unmapped buffer",
            )
        })?;
        Ok(guard)
    }

    /// Borrow the underlying buffer for recording / uploads. Does
    /// not transfer ownership; the guard still frees on Drop.
    pub(super) const fn as_ref(&self) -> &B::DeviceBuffer {
        match &self.buf {
            Some(b) => b,
            None => panic!("DeviceBufferGuard::as_ref after take"),
        }
    }

    /// Transfer the buffer to the caller. Drop becomes a no-op;
    /// the caller is now responsible for freeing the buffer (e.g.,
    /// via `backend.free_device`).
    ///
    /// Used at the end of a dispatcher's happy path so the explicit
    /// `free_device` calls keep their existing single-source-of-
    /// ownership shape (rather than relying on Drop ordering).
    pub(super) fn take(mut self) -> B::DeviceBuffer {
        self.buf
            .take()
            .expect("DeviceBufferGuard::take called twice")
    }
}

impl<B: GpuBackend + ?Sized> Drop for DeviceBufferGuard<'_, B> {
    fn drop(&mut self) {
        if let Some(buf) = self.buf.take() {
            self.backend.free_device(buf);
        }
    }
}

#[cfg(all(test, feature = "vulkan"))]
mod tests {
    use super::DeviceBufferGuard;
    use crate::backend::vulkan::VulkanBackend;
    use crate::backend::{BackendError, GpuBackend};

    #[test]
    fn zeroed_readback_guard_uses_readback_memory_on_an_integrated_gpu() {
        let selector = "radv".parse().expect("non-empty selector");
        let backend = match VulkanBackend::with_selector(Some(&selector)) {
            Ok(b) if b.is_integrated() => b,
            Ok(_) => {
                eprintln!("skipping: the radv device is not integrated");
                return;
            }
            Err(BackendError::DeviceSelection(msg))
                if msg.contains("no device name contains it") =>
            {
                eprintln!("skipping: no radv device");
                return;
            }
            Err(e) => panic!("selecting the radv device failed: {e}"),
        };
        let guard = DeviceBufferGuard::alloc_readback_zeroed(&backend, 4096).expect("alloc");
        let reference = backend.alloc_device_readback(4096).expect("readback");
        assert_eq!(
            guard.as_ref().memory_properties(),
            reference.memory_properties(),
            "the zeroed read-back buffer is not in read-back memory"
        );
        let mut out = vec![1u8; 4096];
        backend
            .download(guard.as_ref(), &mut out)
            .expect("download");
        assert!(out.iter().all(|&b| b == 0), "the buffer is not zeroed");
    }
}
