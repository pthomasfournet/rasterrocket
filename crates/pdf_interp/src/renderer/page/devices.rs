//! GPU decoders, kernels and the device image cache attached to a renderer.

#[cfg(any(feature = "gpu-aa", feature = "gpu-icc", feature = "cache"))]
use std::sync::Arc;

#[cfg(any(feature = "gpu-aa", feature = "gpu-icc", feature = "cache"))]
use gpu::GpuCtx;
#[cfg(feature = "vaapi")]
use gpu::JpegQueueHandle;
#[cfg(feature = "gpu-jpeg-huffman")]
use gpu::backend::cuda::CudaBackend;
#[cfg(feature = "vulkan")]
use gpu::backend::vulkan::VulkanBackend;
#[cfg(feature = "cache")]
use gpu::cache::{DeviceImageCache, DevicePageBuffer, DocId};
#[cfg(feature = "gpu-jpeg-huffman")]
use gpu::jpeg_decoder::JpegGpuDecoder;
#[cfg(feature = "nvjpeg")]
use gpu::nvjpeg::NvJpegDecoder;
#[cfg(feature = "nvjpeg2k")]
use gpu::nvjpeg2k::NvJpeg2kDecoder;

use super::PageRenderer;

/// Renderer-side device-image-cache state bundled together so the "cache
/// wired in" condition is a single `Option<CacheState>` rather than three
/// implicit invariants on separate fields.
///
/// Without this bundling, `doc_id: DocId` had to default to
/// `DocId([0u8; 32])` — a valid hash that would collide with anyone
/// using the all-zeros doc id, a sharp foot-gun.
#[cfg(feature = "cache")]
pub(super) struct CacheState {
    /// Device-resident image cache.  Shared across pages so
    /// content-hash dedup spans the whole render session.
    pub(super) cache: Arc<DeviceImageCache>,
    /// Stable identifier for the source PDF — combined with the
    /// per-image PDF object id, forms the cache's `(DocId, ObjId)`
    /// alias key for fast same-document lookups.
    pub(super) doc_id: DocId,
    /// Lazy-allocated device-resident page buffer.  `None` until the
    /// first GPU image blit allocates it.  Between blits every pixel is
    /// zero (alpha 0): each blit composites its rows onto `bitmap` and
    /// clears them before returning.
    pub(super) page_buffer: Option<DevicePageBuffer>,
}

impl PageRenderer<'_> {
    /// Attach a GPU parallel-Huffman JPEG decoder to this renderer.
    #[cfg(feature = "gpu-jpeg-huffman")]
    pub fn set_jpeg_gpu(&mut self, dec: Option<JpegGpuDecoder<CudaBackend>>) {
        self.jpeg_gpu = dec;
    }

    /// Detach and return the GPU parallel-Huffman JPEG decoder for reuse.
    #[cfg(feature = "gpu-jpeg-huffman")]
    pub const fn take_jpeg_gpu(&mut self) -> Option<JpegGpuDecoder<CudaBackend>> {
        self.jpeg_gpu.take()
    }

    /// Attach a Vulkan parallel-Huffman JPEG decoder to this renderer.
    ///
    /// Mutually exclusive with [`set_jpeg_gpu`] — call one or the other per
    /// render, never both.
    #[cfg(all(feature = "gpu-jpeg-huffman", feature = "vulkan"))]
    pub fn set_jpeg_vk(&mut self, dec: Option<JpegGpuDecoder<VulkanBackend>>) {
        self.jpeg_vk = dec;
    }

    /// Detach and return the Vulkan parallel-Huffman JPEG decoder for reuse.
    #[cfg(all(feature = "gpu-jpeg-huffman", feature = "vulkan"))]
    pub const fn take_jpeg_vk(&mut self) -> Option<JpegGpuDecoder<VulkanBackend>> {
        self.jpeg_vk.take()
    }

    /// [`crate::resources::image::GPU_JPEG_THRESHOLD_PX`] are decoded on the
    /// GPU via nvJPEG rather than the CPU JPEG decoder.
    ///
    /// Calling this with `None` detaches any existing decoder (reverts to CPU).
    #[cfg(feature = "nvjpeg")]
    pub fn set_nvjpeg(&mut self, dec: Option<NvJpegDecoder>) {
        self.nvjpeg = dec;
    }

    /// Attach a VA-API JPEG decode queue handle to this renderer.
    ///
    /// The handle is a cheaply-cloneable sender to a dedicated OS worker thread
    /// that owns the `VapiJpegDecoder`.  `DCTDecode` baseline JPEG streams with
    /// pixel area ≥ [`crate::resources::image::GPU_JPEG_THRESHOLD_PX`] are
    /// submitted to the queue instead of being decoded on the calling thread,
    /// eliminating Mesa driver contention across Rayon workers.
    ///
    /// The handle is dropped when the renderer drops — no explicit reclaim step
    /// is needed because the queue worker lives in `RasterSession`, which always
    /// outlives the renderer.
    #[cfg(feature = "vaapi")]
    pub fn set_vaapi_queue(&mut self, handle: JpegQueueHandle) {
        self.vaapi_jpeg_queue = Some(handle);
    }

    /// Attach a GPU JPEG 2000 decoder to this renderer.
    ///
    /// When set, `JPXDecode` image streams with pixel area ≥
    /// [`crate::resources::image::GPU_JPEG2K_THRESHOLD_PX`] are decoded on the
    /// GPU via nvJPEG2000 rather than the CPU JPEG 2000 decoder.
    ///
    /// Call with `None` to revert to CPU-only JPEG 2000 decode.
    #[cfg(feature = "nvjpeg2k")]
    pub fn set_nvjpeg2k(&mut self, dec: Option<NvJpeg2kDecoder>) {
        self.nvjpeg2k = dec;
    }

    /// Detach and return the nvJPEG decoder so the caller can reuse it.
    ///
    /// Returns `None` if no decoder was attached (e.g. GPU init failed).
    /// Used by the CLI to return the decoder to its thread-local slot
    /// after each page render so it survives across pages.
    #[cfg(feature = "nvjpeg")]
    pub const fn take_nvjpeg(&mut self) -> Option<NvJpegDecoder> {
        self.nvjpeg.take()
    }

    /// Detach and return the nvJPEG2000 decoder so the caller can reuse it.
    ///
    /// Returns `None` if no decoder was attached (e.g. GPU init failed).
    /// Used by the CLI to return the decoder to its thread-local slot
    /// after each page render so it survives across pages.
    #[cfg(feature = "nvjpeg2k")]
    pub const fn take_nvjpeg2k(&mut self) -> Option<NvJpeg2kDecoder> {
        self.nvjpeg2k.take()
    }

    /// Attach a GPU context for antialiased fill dispatch.
    ///
    /// When set, solid-colour fills under a rectangular clip whose pixel
    /// bounding-box area reaches the dispatch threshold (dormant by default;
    /// see [`gpu::GPU_AA_FILL_THRESHOLD`]) have their
    /// coverage evaluated on the GPU.  The device walks the same 4× scanline
    /// intersection table the CPU fill builds and the counts are composited
    /// through the CPU pipeline, so the rendered bytes are identical to the
    /// CPU-only render.
    ///
    /// Call with `None` to revert to CPU-only AA.
    #[cfg(any(feature = "gpu-aa", feature = "gpu-icc", feature = "cache"))]
    pub fn set_gpu_ctx(&mut self, ctx: Option<Arc<GpuCtx>>) {
        self.gpu_ctx = ctx;
    }

    /// Attach a Vulkan compute backend for kernel dispatch.
    ///
    /// When set, GPU-eligible AA fill and ICC CMYK→RGB calls
    /// are routed through the [`gpu::backend::GpuBackend`] trait surface
    /// (`alloc → upload → record_* → submit → wait → download`) on the
    /// Vulkan recorder rather than the CUDA `GpuCtx`.  The CUDA path is
    /// not removed; it remains live for sessions opened with
    /// `BackendPolicy::Auto` / `BackendPolicy::ForceCuda` (defined in `rasterrocket`).
    ///
    /// The device-resident image cache (`DeviceImageCache`,
    /// `DevicePageBuffer`) is CUDA-only — under Vulkan the renderer
    /// runs uncached (CPU-resident image decode + blit).
    #[cfg(feature = "vulkan")]
    pub fn set_vk_backend(&mut self, backend: Option<Arc<VulkanBackend>>) {
        self.vk_backend = backend;
    }

    /// Attach a device-resident image cache.
    ///
    /// When set, JPEG image decode goes through the cache: a content-
    /// hash hit returns an `Arc<CachedDeviceImage>` with no decode
    /// work; a miss decodes on CPU, uploads to VRAM, caches, and
    /// returns the same handle.  The renderer dispatches `Gpu`-variant
    /// images to a CUDA blit kernel that writes into a per-page
    /// `DevicePageBuffer` and composites the result onto `bitmap` in
    /// content-stream order, so the page renders byte-identically to
    /// the CPU-only path.
    ///
    /// `doc_id` should be a stable identifier for the source PDF
    /// (typically `BLAKE3(pdf_bytes)` or similar).  It's combined
    /// with the per-image PDF object number to form the cache's
    /// secondary alias key for fast same-document lookups.
    ///
    /// Call with `None` to revert to CPU-only image decode.
    ///
    /// **Caller contract:** call before any rendering operators
    /// execute on this renderer; today no caller wires or unwires a
    /// cache mid-page.
    #[cfg(feature = "cache")]
    pub fn set_image_cache(&mut self, cache: Option<Arc<DeviceImageCache>>, doc_id: DocId) {
        self.cache_state = cache.map(|cache| CacheState {
            cache,
            doc_id,
            page_buffer: None,
        });
    }
}
