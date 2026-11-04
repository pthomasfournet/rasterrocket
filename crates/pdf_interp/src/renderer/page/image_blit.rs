//! Drawing a decoded image or stencil mask onto the page through the clip,
//! on the CPU or through the device blit kernel.

#[cfg(feature = "cache")]
use std::sync::Arc;

use color::convert::div255;
use raster::ClipResult;

use super::super::gstate::Ctm;
use super::PageRenderer;
use super::canvas::{CanvasKind, blend_u8, with_canvas};
use super::image_compositor::ImageCompositor;
#[cfg(feature = "cache")]
use super::image_compositor::composite_rgba_rows;
use super::image_reduce::{self, DeviceWindow};
use super::image_sampler::{BlendedRows, Filter, GridClip, MaskSampler, SampleGrid, sample_bytes};
use super::pixel_rect::PixelRect;
use crate::resources::{ImageColorSpace, ImageDescriptor, SoftMask};
#[cfg(feature = "cache")]
use gpu::cache::DevicePageBuffer;

impl PageRenderer<'_> {
    /// Count the drawn image in the page diagnostics (presence, filter) and
    /// raise the source-resolution hint to that of its `source_width`
    /// samples across the page (see [`super::diagnostics::source_ppi`]).
    fn record_image_diagnostics(&mut self, img: &ImageDescriptor, source_width: u32) {
        self.diag.has_images = true;
        let filter_idx = img.filter as usize;
        self.filter_counts[filter_idx] = self.filter_counts[filter_idx].saturating_add(1);
        if let Some(ppi) = super::diagnostics::source_ppi(source_width, self.page_width_pts) {
            self.diag.source_ppi_hint =
                Some(self.diag.source_ppi_hint.map_or(ppi, |prev| prev.max(ppi)));
        }
    }

    /// Blit a decoded image `XObject` onto the bitmap using the current CTM.
    ///
    /// The CTM maps the unit square (image space) onto the page; every
    /// device pixel inside the image's bounding box samples the image at its
    /// centre, through a [`SampleGrid`] shared by the CPU sampler below and
    /// the GPU blit kernel: the nearest sample, or a bilinear blend along
    /// the axes [`Filter::for_draw`] selects for a magnified or rotated
    /// draw.  A host-resident image or stencil drawn below its resolution
    /// is first area-averaged onto the device pixels (`image_reduce`).  A
    /// mask is resampled the same way and sampled through a [`MaskSampler`]
    /// at its own resolution.
    ///
    /// `decoded.source_width` is the stream's own pixel width, which the
    /// image's width falls below when it was decoded at reduced resolution;
    /// it feeds the `source_ppi_hint` diagnostic.
    pub(super) fn blit_image(
        &mut self,
        decoded: &crate::resources::DecodedImage,
        name: Option<&[u8]>,
    ) {
        let img = &decoded.image;
        let source_width = decoded.source_width;
        // Degenerate image — nothing to blit.
        if img.width == 0 || img.height == 0 {
            return;
        }
        // A shape-only (`d1`) glyph description is a stencil in the text
        // fill colour (PDF §9.6.5): an image mask paints in that colour, and
        // an image carrying its own colours, which the description may not
        // contain, paints nothing.
        if self.gstate.current().colour_locked && img.color_space != ImageColorSpace::Mask {
            return;
        }
        // A stencil mask paints the fill colour, which may paint nothing.
        if img.color_space == ImageColorSpace::Mask && self.gstate.current().fill_paints_nothing() {
            return;
        }
        // PDF §8.9.6: stencil masks (/ImageMask true → ImageColorSpace::Mask)
        // do not carry an /SMask.  The Mask arm below intentionally skips
        // the smask gate that the Rgb / Gray arms apply; this assert pins
        // the spec invariant so a future decoder change that sets `smask`
        // on a Mask descriptor surfaces in debug builds.
        debug_assert!(
            img.smask.is_none() || !matches!(img.color_space, ImageColorSpace::Mask),
            "PDF §8.9.6: image masks must not carry SMask",
        );

        self.record_image_diagnostics(img, source_width);

        // An image or mask drawn below its resolution is area-averaged
        // first, so that point sampling neither drops thin features nor
        // turns them into whole pixels.  The image and its mask resample
        // separately, each on its own grid, and each comes with the matrix
        // that draws the resampled samples.
        let page_ctm = self.window_ctm(&self.gstate.current().ctm);
        let window = self.paintable_window();
        let reduced_mask = img
            .smask
            .as_ref()
            .and_then(|m| image_reduce::reduce_mask_for_draw(m, &page_ctm, &window));
        let (mask_src, mask_ctm) = match &reduced_mask {
            Some((m, c)) => (Some(m), *c),
            None => (img.smask.as_ref(), page_ctm),
        };
        // A device-resident image is sampled at full resolution by the
        // blit kernel.  A draw that reduces it takes a host copy so that it
        // is area-averaged as a host-resident image is.
        #[cfg(feature = "cache")]
        let host_copy = self.host_copy_for_reduced_draw(img, &page_ctm, &window);
        #[cfg(feature = "cache")]
        let img = host_copy.as_ref().unwrap_or(img);
        let reduced = image_reduce::reduce_for_draw(img, &page_ctm, &window);
        let (img, ctm) = match &reduced {
            Some((i, c)) => (i, *c),
            None => (img, page_ctm),
        };
        let filter = Filter::for_draw(&ctm, img.width, img.height, decoded.interpolate);
        let Some(grid) =
            SampleGrid::new(&ctm, self.width, self.height, img.width, img.height, filter)
        else {
            log::debug!(
                "rasterrocket-interp: blit_image: image maps to no page pixels (CTM {ctm:?}) — skipping"
            );
            return;
        };
        #[cfg(feature = "cache")]
        let table_less_copy = self.host_copy_for_table_less_grid(img, &grid);
        #[cfg(feature = "cache")]
        let img = table_less_copy.as_ref().unwrap_or(img);

        // The grid covers the image's bounding box on the page; the clip
        // region decides which of those pixels the image may paint.
        let clip = self.gstate.current().clip.clone_shared();
        let mut grid_clip = GridClip::new(&clip, &grid);
        if grid_clip.bbox() == ClipResult::AllOutside {
            return;
        }

        if matches!(img.color_space, ImageColorSpace::Mask)
            && self.gstate.current().fill_pattern.is_some()
        {
            match self.fill_stencil_with_pattern(img, &grid) {
                Ok(true) => return,
                Ok(false) => {}
                Err(reason) => {
                    self.record_image_failure(name, reason);
                    return;
                }
            }
        }

        // The image is one object under the soft mask in effect, unless its
        // own `/SMask` replaces that mask.
        let paint = |r: &mut Self| {
            if let Err(reason) = r.composite_image(
                img,
                &ctm,
                mask_src.map(|m| (m, mask_ctm)),
                &grid,
                &mut grid_clip,
            ) {
                r.record_image_failure(name, reason);
            }
        };
        let region = PixelRect::new(grid.x0, grid.y0, grid.x1, grid.y1);
        let alpha = self.gstate.current().fill_alpha;
        if decoded.own_soft_mask {
            self.paint_knocking_out(region, alpha, paint);
        } else {
            self.paint_object(region, alpha, paint);
        }
    }

    /// The device pixels an image draw can paint: the bitmap within the
    /// clip rectangle, which bounds every path clip.
    fn paintable_window(&self) -> DeviceWindow {
        let clip = &self.gstate.current().clip;
        DeviceWindow::of_bitmap_rect(
            self.width,
            self.height,
            [
                i64::from(clip.x_min_i),
                i64::from(clip.y_min_i),
                i64::from(clip.x_max_i) + 1,
                i64::from(clip.y_max_i) + 1,
            ],
        )
    }

    /// Paint the fill pattern through the stencil mask `img` sampled
    /// through `grid` (PDF §8.9.6.2: the stencil's set samples are painted
    /// with the current fill colour, here a pattern).  The pattern fills the
    /// grid's rectangle, scaled at each pixel by the stencil coverage, so a
    /// tiling or shading pattern of any kind — with its `Background`, `BBox`
    /// and `ExtGState` — paints exactly as it does through a path fill.
    ///
    /// Returns `Ok(false)`, painting nothing, when the stencil is not host
    /// resident.
    ///
    /// # Errors
    /// The reason, when the stencil's samples cannot hold its dimensions.
    pub(super) fn fill_stencil_with_pattern(
        &mut self,
        img: &crate::resources::ImageDescriptor,
        grid: &SampleGrid,
    ) -> Result<bool, String> {
        let Some(bytes) = img.data.as_cpu() else {
            return Ok(false);
        };
        verify_sample_len(img, bytes.len())?;
        let Some(rect) = PixelRect::new(grid.x0, grid.y0, grid.x1, grid.y1) else {
            return Ok(true);
        };
        let img_w = img.width as usize;
        let coverage: Vec<u8> = (rect.y0..rect.y1)
            .flat_map(|dy| (rect.x0..rect.x1).map(move |dx| (dx, dy)))
            .map(|(dx, dy)| {
                grid.sample(dx, dy).map_or(0, |taps| {
                    u8::MAX - taps.blend_px(|x, y| sample_bytes::<1>(bytes, img_w, 1, x, y))[0]
                })
            })
            .collect();
        let area = rect.to_path();
        self.paint_through_coverage(rect, &coverage, |r| r.fill_path(&area, false));
        Ok(true)
    }

    /// A host-resident copy of `img` when its samples are device-resident
    /// and drawing them through `ctm` reduces the image, so that it is
    /// area-averaged as a host-resident image is.  `None` otherwise, or
    /// when the copy fails; the device samples are then drawn as they are.
    #[cfg(feature = "cache")]
    pub(super) fn host_copy_for_reduced_draw(
        &self,
        img: &crate::resources::ImageDescriptor,
        ctm: &Ctm,
        window: &DeviceWindow,
    ) -> Option<crate::resources::ImageDescriptor> {
        if !image_reduce::draw_reduces(ctm, img.width, img.height, window) {
            return None;
        }
        self.host_copy(img, "a reduced draw")
    }

    /// A host-resident copy of `img` when its samples are device-resident
    /// and `grid` has no tables, which are all the blit kernel reads;
    /// `None` otherwise, or when the copy fails.
    #[cfg(feature = "cache")]
    fn host_copy_for_table_less_grid(
        &self,
        img: &crate::resources::ImageDescriptor,
        grid: &SampleGrid,
    ) -> Option<crate::resources::ImageDescriptor> {
        grid.tables()
            .is_none()
            .then(|| self.host_copy(img, "a draw beyond the blit tables"))
            .flatten()
    }

    /// `decoded` with its samples host-resident when they are
    /// device-resident and the canvas holds ink, which the blit kernel
    /// cannot write; unchanged otherwise, or when the copy fails.  Applied
    /// before the image enters the image memo, so the samples are read
    /// back once per page however often the image is drawn.
    #[cfg(feature = "cache")]
    pub(super) fn host_resident_for_ink(
        &self,
        mut decoded: crate::resources::DecodedImage,
    ) -> crate::resources::DecodedImage {
        if self.canvas.kind() == CanvasKind::Ink
            && let Some(host) = self.host_copy(&decoded.image, "an ink canvas")
        {
            decoded.image = host;
        }
        decoded
    }

    /// A host-resident copy of `img` when its samples are device-resident,
    /// read back from the image cache for `purpose`; `None` when they are
    /// host-resident or the read-back fails.
    #[cfg(feature = "cache")]
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

    /// Composite `img`, sampled through `grid` under `ctm`, onto the bitmap
    /// within `grid_clip`; `mask` is its soft mask with the matrix that
    /// draws it.  `Err` carries the reason a decoded image cannot be drawn,
    /// for the caller to record against the image's name.
    #[expect(
        clippy::too_many_lines,
        reason = "GPU dispatch, decoder-contract checks, the mask sampler and the per-pixel loop share the grid and CTM state"
    )]
    pub(super) fn composite_image(
        &mut self,
        img: &crate::resources::ImageDescriptor,
        ctm: &Ctm,
        mask: Option<(&SoftMask, Ctm)>,
        grid: &SampleGrid,
        grid_clip: &mut GridClip<'_>,
    ) -> Result<(), String> {
        // GPU image-blit path: device-resident pixels sampled by the CUDA
        // kernel through the same grid, composited onto `bitmap` before
        // this call returns so later operators paint over the image in
        // content-stream order.
        //
        // The kernel composites opaquely and takes no mask: the decoder
        // keeps every masked image host-resident, so masked images of any
        // resolution take the CPU path below.
        #[cfg(feature = "cache")]
        if let crate::resources::image::ImageData::Gpu(cached) = &img.data {
            debug_assert!(img.smask.is_none(), "device-resident images carry no mask");
            if self.canvas.kind() == CanvasKind::Ink {
                return Err(
                    "device-resident image could not be read back for the ink canvas".to_owned(),
                );
            }
            if self.try_gpu_blit_image(cached, grid, grid_clip) {
                return Ok(());
            }
            // Promotion or kernel dispatch failed and there are no host
            // bytes to fall back to.  The image decoded successfully but
            // cannot be rendered: this is the same silent-blank class the
            // `ImageResolution` contract eradicates upstream, so record it
            // as an image failure rather than dropping the image with only a
            // log line.
            return Err("GPU blit failed, no host fallback".to_owned());
        }

        // CPU path: needs host-resident pixels.
        let Some(img_bytes) = img.data.as_cpu() else {
            // Decoded pixels exist but are device-resident with no GPU
            // dispatch available to blit them — unrenderable, not absent.
            return Err("pixels device-resident but GPU dispatch unavailable".to_owned());
        };
        verify_sample_len(img, img_bytes.len())?;
        let bpp = img.color_space.bytes_per_pixel();

        let painted_fill = self.gstate.current().painted_fill_color();
        let fill_color = painted_fill.bytes();
        let mask = match mask {
            Some((m, mask_ctm)) => {
                let Some(sampler) = MaskSampler::new(
                    m,
                    &mask_ctm,
                    ctm,
                    grid.filter(),
                    self.width,
                    self.height,
                    img.width,
                    img.height,
                ) else {
                    return Ok(());
                };
                Some(sampler)
            }
            None => None,
        };
        let img_w = img.width as usize;
        // A stencil paints the fill colour under the fill's overprint; an
        // image's samples write every channel.
        let stencil = img.color_space == ImageColorSpace::Mask;
        let (fill_alpha, blend_mode) = {
            let gs = self.gstate.current();
            (gs.fill_alpha, gs.blend_mode)
        };
        let paint = if stencil {
            self.solid_paint(false, &painted_fill, fill_alpha, blend_mode)
        } else {
            self.paint_params(fill_alpha, blend_mode, None)
        };
        // Taps lie on the image, so every sample is inside the buffer by
        // the length check above.
        let gray = |x, y| sample_bytes::<1>(img_bytes, img_w, bpp, x, y);
        let rgb = |x, y| sample_bytes::<3>(img_bytes, img_w, bpp, x, y);
        let gs_transfer = self.gstate.current().transfer.clone();
        let transfer = |px: [u8; 3]| gs_transfer.as_ref().map_or(px, |t| t.apply(px));
        let mut compositor = with_canvas!(&self.canvas, |b| ImageCompositor::new(b, &paint));
        // On an ink canvas a stencil paints the fill colour as its ink.
        let kind = self.canvas.kind();
        if stencil && kind == CanvasKind::Ink {
            compositor.solid = Some(painted_fill.bytes_for(kind.mode()));
        }

        // Each row's samples and their coverage are gathered first and
        // composited as one row (see `ImageCompositor::paint_row`).
        let span = (grid.x1 - grid.x0) as usize;
        let mut colours = vec![0u8; span * 3];
        let mut shape = vec![0u8; span];
        // A separable draw's taps are built once per column and once per
        // row instead of once per pixel.  Without a soft mask, which reads
        // each pixel's taps, its samples are also blended a row at a time.
        let separable = grid.separable();
        let mut row_blend =
            separable
                .as_ref()
                .filter(|_| mask.is_none())
                .map(|cols| match img.color_space {
                    ImageColorSpace::Rgb => RowBlend::Rgb(BlendedRows::new(cols)),
                    ImageColorSpace::Gray | ImageColorSpace::Mask => {
                        RowBlend::Gray(BlendedRows::new(cols))
                    }
                });
        for dy in grid.y0..grid.y1 {
            let Some(row_clip) = grid_clip.row(dy) else {
                continue;
            };
            // `Some((columns, row taps))` for a separable grid; a row whose
            // source y lies off the image admits no pixel.
            let separable_row = separable.as_ref().map(|cols| (cols, grid.row_taps(dy)));
            let samples = match (row_blend.as_mut(), separable_row) {
                (Some(RowBlend::Rgb(rows)), Some((_, Some(row)))) => {
                    RowSamples::Rgb(rows.blend_row(row, rgb))
                }
                (Some(RowBlend::Gray(rows)), Some((_, Some(row)))) => {
                    RowSamples::Gray(rows.blend_row(row, gray))
                }
                _ => RowSamples::PerPixel,
            };
            shape.fill(0);
            for (((col, dx), out), shape) in (0..)
                .zip(grid.x0..grid.x1)
                .zip(colours.as_chunks_mut::<3>().0)
                .zip(shape.iter_mut())
            {
                let coverage = row_clip.coverage(dx);
                if coverage == 0 {
                    continue;
                }
                let taps = match separable_row {
                    Some((cols, row)) => row.and_then(|row| cols.taps(col, row)),
                    None => grid.sample(dx, dy),
                };
                let Some(taps) = taps else {
                    continue;
                };
                let gray_at = || match samples {
                    RowSamples::Gray(row) => row.get(col).map_or(u8::MAX, |p| p[0]),
                    _ => taps.blend_px(gray)[0],
                };
                // Stencil masks paint the fill colour with coverage
                // `255 − sample` and carry no soft mask (asserted at entry).
                let alpha = match img.color_space {
                    ImageColorSpace::Mask => {
                        div255(u32::from(u8::MAX - gray_at()) * u32::from(coverage))
                    }
                    ImageColorSpace::Rgb | ImageColorSpace::Gray => div255(
                        u32::from(mask.as_ref().map_or(u8::MAX, |m| m.alpha(dx, dy, &taps)))
                            * u32::from(coverage),
                    ),
                };
                if alpha == 0 {
                    continue;
                }
                // Image samples map through the transfer function; the
                // stencil colour is already the painted fill colour.
                let src = match img.color_space {
                    ImageColorSpace::Rgb => transfer(match samples {
                        RowSamples::Rgb(row) => row.get(col).copied().unwrap_or([u8::MAX; 3]),
                        _ => taps.blend_px(rgb),
                    }),
                    ImageColorSpace::Gray => transfer([gray_at(); 3]),
                    ImageColorSpace::Mask => fill_color,
                };
                *out = src;
                *shape = alpha;
            }
            with_canvas!(&mut self.canvas, |b| compositor
                .paint_row(b, grid.x0, dy, &colours, &shape));
        }
        Ok(())
    }

    /// GPU image blit: sample `cached` through `grid` on the device, then
    /// download the touched rows of the page buffer and composite them onto
    /// `bitmap`.  Returns `true` once the pixels are on the host bitmap;
    /// `false` if any prerequisite is missing (no `GpuCtx`, page-buffer
    /// alloc fails, kernel dispatch or download errors), in which case the
    /// bitmap is untouched.
    ///
    /// Compositing before returning keeps content-stream order: anything
    /// painted after the image lands on top of it, and an image inside a
    /// transparency group lands in the group's bitmap.
    #[cfg(feature = "cache")]
    pub(super) fn try_gpu_blit_image(
        &mut self,
        cached: &Arc<gpu::cache::CachedDeviceImage>,
        grid: &SampleGrid,
        grid_clip: &mut GridClip<'_>,
    ) -> bool {
        use gpu::blit::BlitBbox;

        // The kernel's samples are sRGB, composited onto an RGB canvas only.
        if self.canvas.kind() != CanvasKind::Rgb {
            return false;
        }
        let paint = {
            let gs = self.gstate.current();
            self.paint_params(gs.fill_alpha, gs.blend_mode, None)
        };
        let Some(gpu_ctx) = self.gpu_ctx.as_deref() else {
            log::debug!("blit_image: GPU image blit skipped — no GpuCtx attached");
            return false;
        };
        let Some(cache_state) = self.cache_state.as_mut() else {
            // Shouldn't happen: ImageData::Gpu only originates from a
            // wired-in cache.  Defensive check; cheap.
            return false;
        };
        // Stream ordering: the blit kernel and the page buffer's row
        // download and clear all run on this thread's stream, so they
        // execute in order.  The kernel writes the page buffer through a
        // shared device pointer, which records no cross-stream write event;
        // the buffer is therefore bound to the stream of the thread that
        // renders the page.  `cached` was uploaded on the cache's stream;
        // the launch waits on its write event.

        // Grid edges are clamped to the page, so they fit i32 whenever the
        // page does; a page wider than i32::MAX pixels cannot exist.
        let (Ok(x0), Ok(y0), Ok(x1), Ok(y1)) = (
            i32::try_from(grid.x0),
            i32::try_from(grid.y0),
            i32::try_from(grid.x1),
            i32::try_from(grid.y1),
        ) else {
            return false;
        };
        let bbox = BlitBbox { x0, y0, x1, y1 };
        let Some((cols, rows)) = grid.tables() else {
            return false;
        };

        // Lazy-allocate the per-page device buffer on first GPU
        // image; subsequent images on the same page reuse it.
        if cache_state.page_buffer.is_none() {
            match DevicePageBuffer::new(gpu_ctx.stream(), self.width, self.height) {
                Ok(buf) => cache_state.page_buffer = Some(buf),
                Err(e) => {
                    log::warn!("blit_image: DevicePageBuffer alloc failed: {e}");
                    return false;
                }
            }
        }
        let Some(buf) = cache_state.page_buffer.as_mut() else {
            return false;
        };

        if let Err(e) =
            gpu_ctx.blit_image_to_buffer(cached, buf, bbox, cols, rows, grid.filter().bits())
        {
            log::warn!("blit_image: GPU kernel dispatch failed: {e}");
            return false;
        }
        let super::canvas::PageCanvas::Rgb(bitmap) = &mut self.canvas else {
            return false;
        };
        let compositor = ImageCompositor::new(bitmap, &paint);
        let transfer = self.gstate.current().transfer.clone();
        let downloaded = buf.download_rows_with(grid.y0, grid.y1, |y, rows| {
            composite_rgba_rows(
                bitmap,
                rows,
                y,
                grid,
                grid_clip,
                compositor,
                transfer.as_deref(),
            );
        });
        if let Err(e) = downloaded {
            log::warn!("blit_image: device page buffer download failed: {e}");
            // The buffer now holds this image's pixels with alpha set;
            // drop it so a later blit starts from a zeroed buffer.  Bands
            // composited before the failure stay on the bitmap; the image
            // is still reported as not drawn.
            cache_state.page_buffer = None;
            return false;
        }
        // Clear the touched rows so the next blit's alpha test sees only
        // its own writes.  On failure drop the buffer for the same reason
        // as above; the composite already succeeded.
        if let Err(e) = buf.zero_rows(grid.y0, grid.y1) {
            log::warn!("blit_image: device page buffer clear failed: {e}");
            cache_state.page_buffer = None;
        }
        true
    }
}

/// The per-row sample cache of a separable image draw, by channel count.
enum RowBlend<'a> {
    Rgb(BlendedRows<'a, 3>),
    Gray(BlendedRows<'a, 1>),
}

/// The samples of one device row: blended a row at a time, or left to each
/// pixel's own taps.
#[derive(Clone, Copy)]
enum RowSamples<'a> {
    Rgb(&'a [[u8; 3]]),
    Gray(&'a [[u8; 1]]),
    PerPixel,
}

/// Checks the decoder contract that `len` bytes hold `img`'s samples, so
/// that a truncated buffer (a stream declaring larger dimensions than its
/// data) fails the image instead of painting a partly blank one.
///
/// # Errors
/// The reason, when the dimensions overflow `usize` or `len` is short.
fn verify_sample_len(img: &crate::resources::ImageDescriptor, len: usize) -> Result<(), String> {
    let bpp = img.color_space.bytes_per_pixel();
    match check_image_bytes_len(img.width, img.height, bpp, len) {
        Ok(_) => Ok(()),
        Err(ImageLenError::DimensionOverflow) => Err(format!(
            "dimensions overflow usize ({}×{}×{bpp} bpp)",
            img.width, img.height,
        )),
        Err(ImageLenError::ShortBuffer { expected }) => Err(format!(
            "decoder produced short buffer ({len} bytes, expected {expected})"
        )),
    }
}

/// Paint `src` onto the RGB triple `dst` with coverage `alpha`; an opaque
/// sample copies, anything else blends per channel.
#[inline]
pub(super) fn blend_rgb(dst: &mut [u8], src: [u8; 3], alpha: u8) {
    if alpha == 255 {
        dst.copy_from_slice(&src);
    } else {
        // Each output channel blends against its own existing value —
        // the destination is RGB and channels may differ.
        dst[0] = blend_u8(src[0], dst[0], alpha);
        dst[1] = blend_u8(src[1], dst[1], alpha);
        dst[2] = blend_u8(src[2], dst[2], alpha);
    }
}

/// Reason `blit_image`'s decoder-contract check rejected an image.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum ImageLenError {
    /// `width × height × bpp` overflows `usize`.
    DimensionOverflow,
    /// Buffer is shorter than the declared dimensions require.
    ShortBuffer { expected: usize },
}

/// Validates the decoder contract `actual_len >= width × height × bpp` and
/// returns the expected length on success.  Pulled out of `blit_image` so the
/// two guard branches (overflow, short buffer) are unit-testable without
/// constructing a `PageRenderer`.
pub(super) fn check_image_bytes_len(
    width: u32,
    height: u32,
    bpp: usize,
    actual_len: usize,
) -> Result<usize, ImageLenError> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(bpp))
        .ok_or(ImageLenError::DimensionOverflow)?;
    if actual_len < expected {
        return Err(ImageLenError::ShortBuffer { expected });
    }
    Ok(expected)
}
