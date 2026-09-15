//! Device-resident page buffer.
//!
//! `DevicePageBuffer` is the GPU-side composition target for one page.
//! The blit kernel writes transformed image pixels into it with
//! `alpha=255`; pixels it does not touch keep their zero initialisation.
//! The renderer downloads the rows a blit touched via
//! [`DevicePageBuffer::download_rows`], source-over composites them onto
//! its host bitmap, and clears them again with
//! [`DevicePageBuffer::zero_rows`] so the next blit's alpha channel
//! reflects only its own writes.
//!
//! # Layout
//!
//! Row-major RGBA8 (4 bytes/pixel).  Width and height match the page
//! resolution.  Zero-initialised on creation so any pixel the kernel
//! doesn't write reads back as `(0, 0, 0, 0)` — fully transparent, and
//! skipped by the host-side composite.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DriverError};

/// Device-side composition target for one rendered page.
///
/// Drop releases the device memory; the buffer doesn't need to outlive
/// the page render.
///
/// # Backend abstraction
///
/// `DevicePageBuffer` is intentionally **not** generic over
/// [`crate::backend::GpuBackend`].  The internals call
/// `CudaStream::alloc_zeros`, `memcpy_dtoh`, `memset_zeros`, and
/// `synchronize` directly — the `GpuBackend` trait abstracts away streams
/// (Vulkan has command buffers + fences, not streams) and does not yet
/// expose a sub-range download / zero-fill equivalent.  Generifying would
/// either require those trait methods (not yet designed) or a
/// `PhantomData` type parameter that adds noise without buying abstraction.
///
/// Same applies to [`crate::cache::DeviceImageCache`], which is more
/// deeply CUDA-coupled (`clone_htod`, `memcpy_dtoh`, pinned-pool
/// promotion).  Generification of both is tracked as follow-up after
/// the Vulkan backend's transfer / upload surface settles.
pub struct DevicePageBuffer {
    /// Width × height × 4 bytes, row-major RGBA8.
    pub rgba: CudaSlice<u8>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// CUDA stream the buffer's allocations and writes are bound to.
    /// Stored so the row download and clear use the same stream and so
    /// a caller can synchronise it before relying on completed writes.
    stream: Arc<CudaStream>,
}

impl DevicePageBuffer {
    /// Allocate and zero a fresh page buffer of `width × height` pixels.
    ///
    /// Zero-initialisation matters: the blit kernel doesn't write every
    /// pixel — only the ones that fall inside an image's transformed
    /// extent — and the host-side composite reads alpha from un-written
    /// pixels to leave the CPU-rasterised content alone.
    ///
    /// # Errors
    /// Returns the underlying [`DriverError`] if device allocation fails
    /// (typically VRAM exhaustion).
    pub fn new(stream: Arc<CudaStream>, width: u32, height: u32) -> Result<Self, DriverError> {
        let len = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(crate::RGBA_BPP))
            .ok_or(DriverError(
                cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE,
            ))?;
        let rgba = stream.alloc_zeros::<u8>(len)?;
        Ok(Self {
            rgba,
            width,
            height,
            stream,
        })
    }

    /// Width × height × 4 bytes — the size of the device allocation
    /// and the host buffer [`Self::download`] writes into.
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        (self.width as usize) * (self.height as usize) * crate::RGBA_BPP
    }

    /// Bytes per row: `width × 4`.
    #[must_use]
    pub const fn row_bytes(&self) -> usize {
        (self.width as usize) * crate::RGBA_BPP
    }

    /// The CUDA stream this buffer is bound to.
    ///
    /// [`Self::download_rows`] synchronises this stream, so kernels
    /// writing or reading [`Self::rgba`] from a *different* stream MUST
    /// first `cudaStreamWaitEvent` against an event recorded on this
    /// stream — otherwise the download may DMA stale bytes.
    /// Same-stream consumption is correct without any explicit sync
    /// because cudarc serialises operations on a single stream.
    #[must_use]
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    /// Byte range of rows `[y0, y1)`; `Err` if the range is inverted or
    /// runs past the buffer.
    const fn row_range(&self, y0: u32, y1: u32) -> Result<std::ops::Range<usize>, DriverError> {
        if y0 > y1 || y1 > self.height {
            return Err(DriverError(
                cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE,
            ));
        }
        let row = self.row_bytes();
        Ok(y0 as usize * row..y1 as usize * row)
    }

    /// Copy rows `[y0, y1)` to a host `Vec<u8>` of `(y1 - y0) × width × 4`
    /// bytes.
    ///
    /// Synchronises the stream first so all pending kernel writes
    /// (typically a sequence of blit-kernel launches) complete before
    /// the D→H copy is enqueued, then synchronises again after the
    /// copy so the bytes are observed-stable on host before returning.
    /// This matches the sync-then-copy ordering used by every other
    /// dispatcher in [`crate::GpuCtx`].
    ///
    /// # Errors
    /// Returns [`DriverError`] for an invalid row range, or if either sync
    /// or the D→H copy fails.
    pub fn download_rows(&self, y0: u32, y1: u32) -> Result<Vec<u8>, DriverError> {
        let range = self.row_range(y0, y1)?;
        let mut host = vec![0u8; range.len()];
        if host.is_empty() {
            return Ok(host);
        }
        let view = self.rgba.slice(range);
        self.stream.synchronize()?;
        self.stream.memcpy_dtoh(&view, &mut host)?;
        self.stream.synchronize()?;
        Ok(host)
    }

    /// Copy the whole buffer to a host `Vec<u8>` of [`Self::byte_len`] bytes.
    ///
    /// # Errors
    /// Same as [`Self::download_rows`].
    pub fn download(&self) -> Result<Vec<u8>, DriverError> {
        self.download_rows(0, self.height)
    }

    /// Zero rows `[y0, y1)`.  Stream-ordered: runs after any kernel
    /// already queued on the buffer's stream and before anything queued
    /// later.
    ///
    /// # Errors
    /// Returns [`DriverError`] for an invalid row range or if the memset
    /// fails to enqueue.
    pub fn zero_rows(&mut self, y0: u32, y1: u32) -> Result<(), DriverError> {
        let range = self.row_range(y0, y1)?;
        if range.is_empty() {
            return Ok(());
        }
        let mut view = self.rgba.slice_mut(range);
        self.stream.memset_zeros(&mut view)
    }
}

impl std::fmt::Debug for DevicePageBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DevicePageBuffer")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("byte_len", &self.byte_len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "gpu-validation")]
    use super::*;

    #[test]
    fn rgba_bpp_is_four() {
        assert_eq!(crate::RGBA_BPP, 4);
    }

    #[cfg(feature = "gpu-validation")]
    #[test]
    fn alloc_and_download_zero_buffer() {
        let ctx = cudarc::driver::CudaContext::new(0).expect("CUDA device 0");
        let stream = ctx.default_stream();
        let buf = DevicePageBuffer::new(stream, 32, 16).expect("alloc");
        assert_eq!(buf.width, 32);
        assert_eq!(buf.height, 16);
        assert_eq!(buf.byte_len(), 32 * 16 * 4);
        let host = buf.download().expect("download");
        assert_eq!(host.len(), 32 * 16 * 4);
        assert!(host.iter().all(|&b| b == 0), "alloc_zeros must zero-fill");
    }

    #[cfg(feature = "gpu-validation")]
    #[test]
    fn row_ranges_are_validated_and_sized() {
        let ctx = cudarc::driver::CudaContext::new(0).expect("CUDA device 0");
        let stream = ctx.default_stream();
        let mut buf = DevicePageBuffer::new(stream, 8, 4).expect("alloc");
        assert_eq!(buf.download_rows(1, 3).expect("rows").len(), 2 * 8 * 4);
        assert!(buf.download_rows(2, 2).expect("empty").is_empty());
        assert!(buf.download_rows(3, 2).is_err(), "inverted range");
        assert!(buf.download_rows(0, 5).is_err(), "past the buffer");
        assert!(buf.zero_rows(0, 5).is_err(), "past the buffer");
        buf.zero_rows(0, 4).expect("zero");
        assert!(buf.download().expect("download").iter().all(|&b| b == 0));
    }

    #[cfg(feature = "gpu-validation")]
    #[test]
    fn alloc_rejects_overflow_dimensions() {
        let ctx = cudarc::driver::CudaContext::new(0).expect("CUDA device 0");
        let stream = ctx.default_stream();
        // u32::MAX × u32::MAX × 4 overflows usize.
        let err = DevicePageBuffer::new(stream, u32::MAX, u32::MAX).unwrap_err();
        // Don't lock to a specific CUresult variant; just verify we
        // surfaced an error rather than panicking on overflow.
        let _ = format!("{err:?}");
    }
}
