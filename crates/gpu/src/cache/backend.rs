//! The device operations the image cache performs, per backend.
//!
//! [`CacheBackend`] extends [`GpuBackend`] with the transfers the cache
//! issues outside any page: an upload that creates a slab, a download of a
//! slab, and the moves between a slab and its host-tier copy.  The cache's
//! slabs are the backend's [`GpuBackend::DeviceBuffer`]s, which free their
//! memory on drop on every backend.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, HostSlice, PinnedHostSlice};

use super::CacheError;
use crate::backend::GpuBackend;
use crate::backend::cuda::CudaBackend;
use crate::backend::layout::word_padded;

/// A backend the device image cache keeps its slabs on.
///
/// Implemented by the in-tree backends only ([`GpuBackend`] is sealed).
/// Every transfer is ordered before any device read issued after the
/// method returns: it has completed (Vulkan, and every CUDA download), or
/// it is a CUDA upload whose write event every cudarc launch that reads the
/// slab waits on.  A slab returned by [`Self::upload_slab`] or
/// [`Self::promote`] therefore holds its bytes for every such read, and a
/// host copy returned by [`Self::demote`] holds them once the source slab
/// is dropped.
pub trait CacheBackend: GpuBackend {
    /// The handle the cache issues its transfers through.
    type Device: Send + Sync;
    /// Host memory holding a demoted slab's bytes in the host tier.
    type HostSlab: Send + Sync;

    /// A new slab holding `pixels`, then up to three zero bytes of padding
    /// to the next multiple of 4 ([`word_padded`]), as a blit source requires
    /// ([`crate::backend::params::BlitParams::src`]); `pixels` is not
    /// empty.
    ///
    /// # Errors
    /// The backend's error when the allocation or the copy fails.
    fn upload_slab(device: &Self::Device, pixels: &[u8]) -> Result<Self::DeviceBuffer, CacheError>;

    /// Copy the first `out.len()` bytes of `slab` into `out`.
    ///
    /// # Errors
    /// The backend's error when the copy fails.
    fn download_slab(
        device: &Self::Device,
        slab: &Self::DeviceBuffer,
        out: &mut [u8],
    ) -> Result<(), CacheError>;

    /// Byte length of `slab`.
    fn slab_len(device: &Self::Device, slab: &Self::DeviceBuffer) -> usize;

    /// An uninitialised-content host slab of `bytes > 0` bytes, to be
    /// filled through [`Self::host_slab_mut`] before it is read.
    ///
    /// # Errors
    /// The backend's error when the allocation fails.
    fn alloc_host_slab(device: &Self::Device, bytes: usize) -> Result<Self::HostSlab, CacheError>;

    /// The writable bytes of `slab`.
    ///
    /// # Errors
    /// The backend's error when the host view is unavailable.
    fn host_slab_mut(slab: &mut Self::HostSlab) -> Result<&mut [u8], CacheError>;

    /// Byte length of `slab`.
    fn host_slab_len(slab: &Self::HostSlab) -> usize;

    /// A host copy of the first `bytes` bytes of `slab`.
    ///
    /// # Errors
    /// The backend's error when the allocation or the copy fails.
    fn demote(
        device: &Self::Device,
        slab: &Self::DeviceBuffer,
        bytes: usize,
    ) -> Result<Self::HostSlab, CacheError>;

    /// A new slab holding the bytes of `host`, padded as
    /// [`Self::upload_slab`] pads.
    ///
    /// # Errors
    /// The backend's error when the allocation or the copy fails.
    fn promote(
        device: &Self::Device,
        host: &Self::HostSlab,
    ) -> Result<Self::DeviceBuffer, CacheError>;
}

/// The CUDA cache issues its transfers on a stream of its own; its host
/// tier holds write-combined page-locked memory, which the device reads and
/// writes by DMA and the host only fills from disk.
///
/// Uploads are stream-ordered rather than waited on: the copy records the
/// slab's write event, and a kernel that reads the slab through cudarc on
/// any stream of the context waits on that event.
impl CacheBackend for CudaBackend {
    type Device = Arc<CudaStream>;
    type HostSlab = PinnedHostSlice<u8>;

    fn upload_slab(device: &Self::Device, pixels: &[u8]) -> Result<Self::DeviceBuffer, CacheError> {
        upload_padded(device, pixels)
    }

    fn download_slab(
        device: &Self::Device,
        slab: &Self::DeviceBuffer,
        out: &mut [u8],
    ) -> Result<(), CacheError> {
        if out.is_empty() {
            return Ok(());
        }
        let view = slab.try_slice(..out.len()).ok_or(CacheError::Backend(
            crate::backend::BackendError::msg("cache download longer than the slab"),
        ))?;
        device.memcpy_dtoh(&view, out).map_err(CacheError::Cuda)?;
        device.synchronize().map_err(CacheError::Cuda)
    }

    fn slab_len(_device: &Self::Device, slab: &Self::DeviceBuffer) -> usize {
        slab.len()
    }

    fn alloc_host_slab(device: &Self::Device, bytes: usize) -> Result<Self::HostSlab, CacheError> {
        // SAFETY: the slab's bytes are written through `host_slab_mut` or
        // a device-to-host copy before any read (trait contract).
        unsafe { device.context().alloc_pinned::<u8>(bytes) }.map_err(CacheError::Cuda)
    }

    fn host_slab_mut(slab: &mut Self::HostSlab) -> Result<&mut [u8], CacheError> {
        slab.as_mut_slice().map_err(CacheError::Cuda)
    }

    fn host_slab_len(slab: &Self::HostSlab) -> usize {
        slab.num_bytes()
    }

    /// The copy runs on the cache's stream, which is synchronised before
    /// returning, so the source slab may be dropped afterwards.
    fn demote(
        device: &Self::Device,
        slab: &Self::DeviceBuffer,
        bytes: usize,
    ) -> Result<Self::HostSlab, CacheError> {
        let mut pinned = Self::alloc_host_slab(device, bytes)?;
        let view = slab.try_slice(..bytes).ok_or(CacheError::Backend(
            crate::backend::BackendError::msg("cache demotion longer than the slab"),
        ))?;
        let dst = pinned.as_mut_slice().map_err(CacheError::Cuda)?;
        device.memcpy_dtoh(&view, dst).map_err(CacheError::Cuda)?;
        device.synchronize().map_err(CacheError::Cuda)?;
        Ok(pinned)
    }

    /// The `PinnedHostSlice` is passed directly so that cudarc records the
    /// copy against the slice's event: dropping the host entry while the
    /// copy is in flight then waits for it instead of freeing the source
    /// mid-copy.
    fn promote(
        device: &Self::Device,
        host: &Self::HostSlab,
    ) -> Result<Self::DeviceBuffer, CacheError> {
        upload_padded(device, host)
    }
}

/// A zeroed slab of `src.len()` bytes rounded up to a multiple of 4
/// ([`word_padded`]) whose first `src.len()` bytes are a stream-ordered
/// copy of `src`.
///
/// cudarc records the copy against `src`, so a page-locked `src` dropped
/// while the copy is in flight waits for it.
fn upload_padded<Src: HostSlice<u8> + ?Sized>(
    device: &Arc<CudaStream>,
    src: &Src,
) -> Result<CudaSlice<u8>, CacheError> {
    let len = src.len();
    let mut slab = device
        .alloc_zeros::<u8>(word_padded(len))
        .map_err(CacheError::Cuda)?;
    device
        .memcpy_htod(src, &mut slab.slice_mut(..len))
        .map_err(CacheError::Cuda)?;
    Ok(slab)
}

/// The Vulkan cache issues its transfers through the backend's synchronous
/// [`GpuBackend::upload`] and [`GpuBackend::download`], which stage through
/// the backend's own mapped buffers (or write the slab in place on a device
/// whose buffers are host memory); its host tier therefore holds ordinary
/// host memory.
#[cfg(feature = "vulkan")]
impl CacheBackend for crate::backend::vulkan::VulkanBackend {
    type Device = Arc<Self>;
    type HostSlab = Box<[u8]>;

    fn upload_slab(device: &Self::Device, pixels: &[u8]) -> Result<Self::DeviceBuffer, CacheError> {
        let mut slab = device.alloc_device(word_padded(pixels.len()))?;
        device.upload_word_padded(&mut slab, pixels)?;
        Ok(slab)
    }

    fn download_slab(
        device: &Self::Device,
        slab: &Self::DeviceBuffer,
        out: &mut [u8],
    ) -> Result<(), CacheError> {
        Ok(device.download(slab, out)?)
    }

    fn slab_len(device: &Self::Device, slab: &Self::DeviceBuffer) -> usize {
        device.device_buffer_len(slab)
    }

    fn alloc_host_slab(_device: &Self::Device, bytes: usize) -> Result<Self::HostSlab, CacheError> {
        let mut slab = Vec::new();
        slab.try_reserve_exact(bytes)
            .map_err(|_| CacheError::HostAlloc { bytes })?;
        slab.resize(bytes, 0);
        Ok(slab.into_boxed_slice())
    }

    fn host_slab_mut(slab: &mut Self::HostSlab) -> Result<&mut [u8], CacheError> {
        Ok(slab)
    }

    fn host_slab_len(slab: &Self::HostSlab) -> usize {
        slab.len()
    }

    fn demote(
        device: &Self::Device,
        slab: &Self::DeviceBuffer,
        bytes: usize,
    ) -> Result<Self::HostSlab, CacheError> {
        let mut host = Self::alloc_host_slab(device, bytes)?;
        Self::download_slab(device, slab, &mut host)?;
        Ok(host)
    }

    fn promote(
        device: &Self::Device,
        host: &Self::HostSlab,
    ) -> Result<Self::DeviceBuffer, CacheError> {
        Self::upload_slab(device, host)
    }
}
