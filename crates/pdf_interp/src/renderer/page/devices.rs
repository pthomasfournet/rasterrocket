//! GPU decoders, kernels and the device image cache attached to a renderer.

#[cfg(any(feature = "gpu-icc", feature = "cache", feature = "vulkan"))]
use std::sync::Arc;

#[cfg(any(feature = "gpu-icc", feature = "cache"))]
use gpu::GpuCtx;
#[cfg(feature = "vaapi")]
use gpu::JpegQueueHandle;
#[cfg(feature = "vulkan")]
use gpu::backend::vulkan::VulkanBackend;
#[cfg(feature = "cache")]
use gpu::cache::{DevicePageBuffer, DocId, ImageCache};
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
    /// Device-resident image cache, on the session's backend.  Shared
    /// across pages so content-hash dedup spans the whole render session.
    pub(super) cache: ImageCache,
    /// Stable identifier for the source PDF — combined with the
    /// per-image PDF object id, forms the cache's `(DocId, ObjId)`
    /// alias key for fast same-document lookups.
    pub(super) doc_id: DocId,
    /// Lazy-allocated device-resident page buffer, the target of a CUDA
    /// image blit.  `None` until the first such blit allocates it.  Between
    /// blits every pixel is zero (alpha 0): each blit composites its rows
    /// onto `bitmap` and clears them before returning.
    pub(super) page_buffer: Option<DevicePageBuffer>,
}

impl PageRenderer<'_> {
    /// Attach an nvJPEG decoder to this renderer.  `DCTDecode` streams with
    /// pixel area ≥ [`crate::resources::image::GPU_JPEG_THRESHOLD_PX`] are
    /// decoded on the GPU via nvJPEG rather than the CPU JPEG decoder.
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
    /// [`crate::resources::image::thresholds::GPU_JPEG2K_ATTACHED_THRESHOLD_PX`]
    /// (`PDF_RASTER_JPX_THRESHOLD` overrides it) are decoded on the GPU via
    /// nvJPEG2000 rather than the CPU JPEG 2000 decoder; with a source set
    /// (see [`Self::set_nvjpeg2k_source`]) the source's threshold applies.
    ///
    /// Call with `None` to revert to CPU-only JPEG 2000 decode.
    #[cfg(feature = "nvjpeg2k")]
    pub fn set_nvjpeg2k(&mut self, dec: Option<NvJpeg2kDecoder>) {
        self.nvjpeg2k = dec;
    }

    /// Obtain the GPU JPEG 2000 decoder from `source` on demand.  With a
    /// source set, an image `XObject` is offered the decoder only when it
    /// is one the decoder takes (`JPXDecode`, at least
    /// [`crate::resources::image::thresholds::GPU_JPEG2K_THRESHOLD_PX`] pixels, which
    /// admits none unless `PDF_RASTER_JPX_THRESHOLD` sets an area) and the
    /// draw does not reduce it to half its size or less on both axes (such
    /// a draw decodes on the CPU at a reduced resolution level); the first
    /// such image whose codestream the device decodes obtains the decoder
    /// from `source` when none is attached.  `source` is called at most
    /// once per renderer; its `None` leaves JPEG 2000 decode on the CPU.
    #[cfg(feature = "nvjpeg2k")]
    pub fn set_nvjpeg2k_source(&mut self, source: Option<fn() -> Option<NvJpeg2kDecoder>>) {
        self.nvjpeg2k_supply = source.map_or(super::Nvjpeg2kSupply::Attached, |source| {
            super::Nvjpeg2kSupply::OnDemand(Some(source))
        });
    }

    /// Whether the decode of image `XObject` `name` drawn at `draw_size`
    /// is offered the GPU JPEG 2000 decoder, attached or from a source
    /// (see [`Self::set_nvjpeg2k_source`]): a `JPXDecode` image of at
    /// least the supply's threshold whose draw the CPU decodes at full
    /// resolution, the device decoding only at full resolution.
    #[cfg(feature = "nvjpeg2k")]
    pub(super) fn nvjpeg2k_offered(&self, name: &[u8], draw_size: Option<(u32, u32)>) -> bool {
        self.resources.xobject_takes_device_jpx(
            name,
            draw_size,
            self.nvjpeg2k_supply.jpx_threshold(),
        )
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

    /// Attach a CUDA context for ICC CMYK→RGB conversion and, with the
    /// `cache` feature, image blits.
    ///
    /// Call with `None` to detach it.
    #[cfg(any(feature = "gpu-icc", feature = "cache"))]
    pub fn set_gpu_ctx(&mut self, ctx: Option<Arc<GpuCtx>>) {
        self.gpu_ctx = ctx;
    }

    /// Attach a Vulkan compute backend for kernel dispatch.
    ///
    /// When set, ICC CMYK→RGB conversions run on it, ahead of any CUDA
    /// context.  Image blits run on the backend of
    /// the attached image cache (`Self::set_image_cache`).
    #[cfg(feature = "vulkan")]
    pub fn set_vk_backend(&mut self, backend: Option<Arc<VulkanBackend>>) {
        self.vk_backend = backend;
    }

    /// Attach a device-resident image cache.
    ///
    /// When set, JPEG image decode goes through the cache: a content-
    /// hash hit returns a `CachedImage` with no decode work; a miss
    /// decodes on CPU, uploads to VRAM, caches, and returns the same
    /// handle.  The renderer dispatches `Gpu`-variant images to the blit
    /// kernel of the cache's backend (CUDA or Vulkan), downloads the
    /// pixels it wrote and composites them onto `bitmap` in content-stream
    /// order, so the page renders byte-identically to the CPU-only path.
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
    pub fn set_image_cache(&mut self, cache: Option<ImageCache>, doc_id: DocId) {
        self.cache_state = cache.map(|cache| CacheState {
            cache,
            doc_id,
            page_buffer: None,
        });
    }
}
