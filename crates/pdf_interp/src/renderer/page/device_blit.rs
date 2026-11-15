//! Drawing a device-resident image through the blit kernel of the image
//! cache's backend, and reading one back to the host where the kernel
//! cannot draw it.

#![cfg(feature = "cache")]

use gpu::GpuCtx;
use gpu::blit::BlitBbox;
use gpu::cache::{CachedDeviceImage, CachedImage, DevicePageBuffer};
use raster::pipe::ChannelMask;

use super::super::color::PROCESS_CHANNELS;
use super::PageRenderer;
use super::canvas::{CanvasKind, with_canvas};
use super::devices::CacheState;
use super::image_compositor::{ImageCompositor, RgbaRows, composite_rgba_rows};
use super::image_sampler::{GridClip, SampleGrid};

/// What a device blit left on the bitmap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BlitOutcome {
    /// Every pixel of the image is composited.
    Drawn,
    /// Nothing is composited; the image may still be drawn another way.
    NotStarted,
    /// Some bands are composited and the rest are lost.
    Partial,
}

/// One blit: the page pixels of `bbox`, sampled through the `cols` and
/// `rows` tables of the draw's [`SampleGrid`] with the `filter` bits.
struct BlitJob<'a> {
    bbox: BlitBbox,
    cols: &'a [u32],
    rows: &'a [u32],
    filter: u32,
}

impl PageRenderer<'_> {
    /// A host-resident copy of `img` when its samples are device-resident
    /// and `grid` has no tables, which are all the blit kernel reads;
    /// `None` otherwise, or when the copy fails.
    pub(super) fn host_copy_for_table_less_grid(
        &self,
        img: &crate::resources::ImageDescriptor,
        grid: &SampleGrid,
    ) -> Option<crate::resources::ImageDescriptor> {
        grid.tables()
            .is_none()
            .then(|| self.host_copy(img, "a draw beyond the blit tables"))
            .flatten()
    }

    /// A host-resident copy of `img` when its samples are device-resident,
    /// read back from the image cache for `purpose`; `None` when they are
    /// host-resident or the read-back fails.
    pub(super) fn host_copy(
        &self,
        img: &crate::resources::ImageDescriptor,
        purpose: &str,
    ) -> Option<crate::resources::ImageDescriptor> {
        use crate::resources::image::ImageData;

        let ImageData::Gpu(cached) = &img.data else {
            return None;
        };
        let cache = &self.cache_state.as_ref()?.cache;
        let pixels = cache
            .download(cached)
            .map_err(|e| {
                log::warn!(
                    "rasterrocket-interp: reading a cached image back for {purpose} failed: {e}"
                );
            })
            .ok()?;
        Some(crate::resources::ImageDescriptor {
            width: img.width,
            height: img.height,
            color_space: img.color_space,
            data: ImageData::Cpu(pixels),
            smask: None,
            filter: img.filter,
        })
    }

    /// GPU image blit: sample `cached` through `grid` on the device of the
    /// image cache's backend, then download the pixels the kernel wrote
    /// and composite them onto `bitmap`, writing the channels `overprint`
    /// (every channel for `None`).  [`BlitOutcome::Drawn`] once the pixels
    /// are on the host bitmap; [`BlitOutcome::NotStarted`] when a
    /// prerequisite is missing or fails before anything is composited (no
    /// CUDA context for a CUDA image, a target allocation, the kernel
    /// dispatch, a Vulkan download); [`BlitOutcome::Partial`] when a CUDA
    /// band download fails after earlier bands were composited.
    ///
    /// Compositing before returning keeps content-stream order: anything
    /// painted after the image lands on top of it, and an image inside a
    /// transparency group lands in the group's bitmap.
    pub(super) fn try_gpu_blit_image(
        &mut self,
        cached: &CachedImage,
        grid: &SampleGrid,
        grid_clip: &mut GridClip<'_>,
        overprint: Option<ChannelMask>,
    ) -> BlitOutcome {
        // Grid edges are clamped to the page, so they fit i32 whenever the
        // page does; a page wider than i32::MAX pixels cannot exist.
        let (Ok(x0), Ok(y0), Ok(x1), Ok(y1)) = (
            i32::try_from(grid.x0),
            i32::try_from(grid.y0),
            i32::try_from(grid.x1),
            i32::try_from(grid.y1),
        ) else {
            return BlitOutcome::NotStarted;
        };
        let Some((cols, rows)) = grid.tables() else {
            return BlitOutcome::NotStarted;
        };
        let job = BlitJob {
            bbox: BlitBbox { x0, y0, x1, y1 },
            cols,
            rows,
            filter: grid.filter().bits(),
        };
        let (paint, transfer) = {
            let gs = self.gstate.current();
            (
                self.paint_params(gs.fill_alpha, gs.blend_mode, overprint, PROCESS_CHANNELS),
                gs.transfer.clone(),
            )
        };
        let page = (self.width, self.height);
        let Some(cache_state) = self.cache_state.as_mut() else {
            // Shouldn't happen: ImageData::Gpu only originates from a
            // wired-in cache.  Defensive check; cheap.
            return BlitOutcome::NotStarted;
        };
        // The kernel's samples are sRGB: on an RGB canvas they map through
        // the transfer function here, on an ink canvas the compositor
        // converts them to ink and maps that, as the CPU blit does.
        let kind = self.canvas.kind();
        let rgb_transfer = transfer.as_deref().filter(|_| kind == CanvasKind::Rgb);
        let gpu_ctx = self.gpu_ctx.as_deref();
        with_canvas!(&mut self.canvas, |bitmap| {
            let mut compositor = ImageCompositor::new(bitmap, &paint);
            match kind {
                CanvasKind::Rgb => {}
                CanvasKind::Rgb16 => compositor.wide_transfer = transfer.as_deref(),
                CanvasKind::Ink | CanvasKind::Ink16 => compositor.transfer = transfer.as_deref(),
            }
            let composite = |rows: RgbaRows<'_>| {
                composite_rgba_rows(bitmap, rows, grid, grid_clip, compositor, rgb_transfer);
            };
            match cached {
                CachedImage::Cuda(image) => {
                    blit_on_cuda(gpu_ctx, cache_state, image, page, &job, composite)
                }
                #[cfg(feature = "vulkan")]
                CachedImage::Vulkan(image) => blit_on_vulkan(cache_state, image, &job, composite),
            }
        })
    }
}

/// [`PageRenderer::try_gpu_blit_image`] of a CUDA image onto a page of
/// `page` pixels: the kernel writes the page buffer, whose touched rows are
/// downloaded in bands, passed to `composite` and cleared again.
///
/// The blit kernel and the page buffer's row download and clear all run on
/// the rendering thread's stream, so they execute in order.  The kernel
/// writes the page buffer through a shared device pointer, which records no
/// cross-stream write event; the buffer is therefore bound to the stream of
/// the thread that renders the page.  `image` was uploaded on the cache's
/// stream; the launch waits on its write event.
fn blit_on_cuda(
    gpu_ctx: Option<&GpuCtx>,
    state: &mut CacheState,
    image: &CachedDeviceImage,
    (width, height): (u32, u32),
    job: &BlitJob<'_>,
    mut composite: impl FnMut(RgbaRows<'_>),
) -> BlitOutcome {
    let Some(gpu_ctx) = gpu_ctx else {
        log::debug!("blit_image: GPU image blit skipped — no GpuCtx attached");
        return BlitOutcome::NotStarted;
    };
    // Lazy-allocate the per-page device buffer on first GPU
    // image; subsequent images on the same page reuse it.
    if state.page_buffer.is_none() {
        match DevicePageBuffer::new(gpu_ctx.stream(), width, height) {
            Ok(buf) => state.page_buffer = Some(buf),
            Err(e) => {
                log::warn!("blit_image: DevicePageBuffer alloc failed: {e}");
                return BlitOutcome::NotStarted;
            }
        }
    }
    let Some(buf) = state.page_buffer.as_mut() else {
        return BlitOutcome::NotStarted;
    };
    if let Err(e) =
        gpu_ctx.blit_image_to_buffer(image, buf, job.bbox, job.cols, job.rows, job.filter)
    {
        log::warn!("blit_image: GPU kernel dispatch failed: {e}");
        return BlitOutcome::NotStarted;
    }
    let (y0, y1) = (job.bbox.y0.unsigned_abs(), job.bbox.y1.unsigned_abs());
    let downloaded = buf.download_rows_with(y0, y1, |y, bytes| {
        composite(RgbaRows {
            bytes,
            x0: 0,
            y0: y,
            width,
        });
    });
    if let Err(e) = downloaded {
        log::warn!("blit_image: device page buffer download failed: {e}");
        // The buffer now holds this image's pixels with alpha set;
        // drop it so a later blit starts from a zeroed buffer.  Bands
        // composited before the failure stay on the bitmap; the image
        // is reported as partially drawn.
        state.page_buffer = None;
        return BlitOutcome::Partial;
    }
    // Clear the touched rows so the next blit's alpha test sees only
    // its own writes.  On failure drop the buffer for the same reason
    // as above; the composite already succeeded.
    if let Err(e) = buf.zero_rows(y0, y1) {
        log::warn!("blit_image: device page buffer clear failed: {e}");
        state.page_buffer = None;
    }
    BlitOutcome::Drawn
}

/// [`PageRenderer::try_gpu_blit_image`] of a Vulkan image: the table
/// uploads, the zero-fill of a target the size of the bounding box, the
/// kernel and the target's download run as one page on the cache's
/// backend ([`gpu::backend::blit_image`], with buffers pooled on the
/// backend), and the box is passed to `composite` whole.
#[cfg(feature = "vulkan")]
fn blit_on_vulkan(
    state: &CacheState,
    image: &CachedDeviceImage<gpu::backend::vulkan::VulkanBackend>,
    job: &BlitJob<'_>,
    mut composite: impl FnMut(RgbaRows<'_>),
) -> BlitOutcome {
    let gpu::cache::ImageCache::Vulkan(cache) = &state.cache else {
        log::warn!("blit_image: a Vulkan image reached a cache of another backend");
        return BlitOutcome::NotStarted;
    };
    let Some(layout) = image.layout.blit_layout() else {
        return BlitOutcome::NotStarted;
    };
    let src = gpu::backend::BlitSource {
        pixels: &image.device_ptr,
        width: image.width,
        height: image.height,
        layout,
    };
    let BlitBbox { x0, y0, x1, y1 } = job.bbox;
    let drawn = gpu::backend::blit_image(
        &**cache.device(),
        &src,
        [x0, y0, x1, y1],
        job.cols,
        job.rows,
        job.filter,
        |bytes| {
            composite(RgbaRows {
                bytes,
                x0: x0.unsigned_abs(),
                y0: y0.unsigned_abs(),
                width: job.bbox.width(),
            });
        },
    );
    match drawn {
        Ok(()) => BlitOutcome::Drawn,
        Err(e) => {
            log::warn!("blit_image: Vulkan image blit failed: {e}");
            BlitOutcome::NotStarted
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use gpu::cache::{
        ContentHash, DeviceImageCache, DocId, HostBudget, ImageCache, ImageLayout, InsertRequest,
        ObjId, VramBudget,
    };

    use super::PageRenderer;
    use crate::resources::image::ImageData;
    use crate::resources::{DecodedImage, ImageColorSpace, ImageDescriptor, ImageFilter};

    /// An 8 × 8 RGB gradient drawn through `data` on a 16 × 16 page whose
    /// image cache is `cache`: the canvas bytes and the image failures.
    fn draw(cache: Option<ImageCache>, data: ImageData) -> (Vec<u8>, usize) {
        let doc = crate::test_helpers::empty_doc();
        let mut r = PageRenderer::new(16, 16, &doc, (4, 0));
        r.set_image_cache(cache, DocId([1; 32]));
        r.execute(&crate::content::parse(b"16 0 0 16 0 0 cm"));
        let decoded = DecodedImage {
            image: ImageDescriptor {
                width: 8,
                height: 8,
                color_space: ImageColorSpace::Rgb,
                data,
                smask: None,
                filter: ImageFilter::Dct,
            },
            source_width: 8,
            interpolate: false,
            own_soft_mask: false,
            partial: None,
            tints: None,
            kept_tints: false,
        };
        r.blit_image(&decoded, Some(b"Im0"));
        (r.rgb_canvas().data().to_vec(), r.diag.image_failures.len())
    }

    /// A CUDA image the blit cannot draw (the page has no CUDA context)
    /// is read back and drawn on the CPU, byte-identically to the host
    /// image.
    #[test]
    fn a_blit_that_does_not_start_falls_back_to_the_host_copy() {
        let Ok(ctx) = gpu::GpuCtx::init() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let pixels: Vec<u8> = (0..8 * 8 * 3)
            .map(|i| u8::try_from(i % 251).unwrap())
            .collect();
        let cache = Arc::new(DeviceImageCache::new(
            ctx.stream(),
            VramBudget {
                vram_bytes: 1 << 20,
            },
            HostBudget { host_bytes: 0 },
        ));
        let cached = ImageCache::Cuda(Arc::clone(&cache))
            .insert(InsertRequest {
                doc: DocId([1; 32]),
                obj: ObjId(5),
                hash: ContentHash([7; 32]),
                width: 8,
                height: 8,
                layout: ImageLayout::Rgb,
                pixels: &pixels,
            })
            .expect("insert");
        let (host, host_failures) = draw(None, ImageData::Cpu(pixels));
        let (device, device_failures) = draw(Some(ImageCache::Cuda(cache)), ImageData::Gpu(cached));
        assert_eq!((host_failures, device_failures), (0, 0));
        assert!(host.iter().any(|&b| b != 0xff), "the image paints");
        assert_eq!(host, device);
    }
}
