//! Drawing a decoded image or stencil mask onto the page through the clip,
//! on the CPU or through the device blit kernel.

#[cfg(feature = "cache")]
use std::sync::Arc;

use color::Rgb8;
use color::convert::div255;
use raster::{
    Bitmap, ClipResult,
    pipe::{PipeSrc, PipeState, RowPattern},
    transparency::Backdrop,
    types::BlendMode,
};

use super::super::gstate::Ctm;
use super::PageRenderer;
use super::image_reduce;
use super::image_sampler::{Filter, GridClip, MaskSampler, SampleGrid};
use super::pixel_rect::PixelRect;
use crate::resources::{ImageColorSpace, SoftMask};
#[cfg(feature = "cache")]
use gpu::cache::DevicePageBuffer;

impl PageRenderer<'_> {
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

        // Track diagnostics: image presence and per-filter counts.
        self.diag.has_images = true;
        let filter_idx = img.filter as usize;
        self.filter_counts[filter_idx] = self.filter_counts[filter_idx].saturating_add(1);

        // Update source_ppi_hint: take the widest image seen (best resolution proxy).
        // ppi = (image_px_width / page_pts_width) × 72.
        let page_pts_w = f64::from(self.page_width) / {
            // For unrotated pages CTM[0] carries the scale; for 90°/270° rotations
            // CTM[0] is 0 and CTM[1] carries it.  Take the larger of the two to
            // handle all axis-aligned cases.  Falls back to 1.0 for a degenerate CTM
            // (page_pts_w would be width in pixels — wrong, but finite).
            let ctm = self.gstate.current().ctm;
            let ctm_scale = ctm[0].abs().max(ctm[1].abs());
            if ctm_scale > 0.0 { ctm_scale } else { 1.0 }
        };
        #[expect(
            clippy::cast_possible_truncation,
            reason = "src_ppi is finite and positive, checked below; f64→f32 is safe for PPI values in practice (well under f32::MAX)"
        )]
        let src_ppi = ((f64::from(source_width) / page_pts_w) * 72.0) as f32;
        if src_ppi.is_finite() && src_ppi > 0.0 {
            self.diag.source_ppi_hint = Some(match self.diag.source_ppi_hint {
                Some(prev) if prev >= src_ppi => prev,
                _ => src_ppi,
            });
        }

        // An image or mask drawn below its resolution is area-averaged
        // first, so that point sampling neither drops thin features nor
        // turns them into whole pixels.  The image and its mask resample
        // separately, each on its own grid, and each comes with the matrix
        // that draws the resampled samples.
        let page_ctm = self.gstate.current().ctm;
        let reduced_mask = img
            .smask
            .as_ref()
            .and_then(|m| image_reduce::reduce_mask_for_draw(m, &page_ctm));
        let (mask_src, mask_ctm) = match &reduced_mask {
            Some((m, c)) => (Some(m), *c),
            None => (img.smask.as_ref(), page_ctm),
        };
        // A device-resident image is sampled at full resolution by the
        // blit kernel; a draw that reduces it takes a host copy so that it
        // is area-averaged as a host-resident image is.
        #[cfg(feature = "cache")]
        let host_copy = self.host_copy_for_reduced_draw(img, &page_ctm);
        #[cfg(feature = "cache")]
        let img = host_copy.as_ref().unwrap_or(img);
        let reduced = image_reduce::reduce_for_draw(img, &page_ctm);
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

        // The grid covers the image's bounding box on the page; the clip
        // region decides which of those pixels the image may paint.
        let clip = self.gstate.current().clip.clone_shared();
        let mut grid_clip = GridClip::new(&clip, &grid);
        if grid_clip.bbox() == ClipResult::AllOutside {
            return;
        }

        if matches!(img.color_space, ImageColorSpace::Mask)
            && self.gstate.current().fill_pattern.is_some()
            && self.fill_stencil_with_pattern(img, &grid)
        {
            return;
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

    /// Paint the fill pattern through the stencil mask `img` sampled
    /// through `grid` (PDF §8.9.6.2: the stencil's set samples are painted
    /// with the current fill colour, here a pattern).  The pattern fills the
    /// grid's rectangle, scaled at each pixel by the stencil coverage, so a
    /// tiling or shading pattern of any kind — with its `Background`, `BBox`
    /// and `ExtGState` — paints exactly as it does through a path fill.
    ///
    /// Returns `false`, painting nothing, when the stencil is not host
    /// resident.
    pub(super) fn fill_stencil_with_pattern(
        &mut self,
        img: &crate::resources::ImageDescriptor,
        grid: &SampleGrid,
    ) -> bool {
        let Some(bytes) = img.data.as_cpu() else {
            return false;
        };
        let Some(rect) = PixelRect::new(grid.x0, grid.y0, grid.x1, grid.y1) else {
            return true;
        };
        let img_w = img.width as usize;
        let sample = |x: u32, y: u32| {
            bytes
                .get(y as usize * img_w + x as usize)
                .copied()
                .unwrap_or(u8::MAX)
        };
        let coverage: Vec<u8> = (rect.y0..rect.y1)
            .flat_map(|dy| (rect.x0..rect.x1).map(move |dx| (dx, dy)))
            .map(|(dx, dy)| {
                grid.sample(dx, dy)
                    .map_or(0, |taps| u8::MAX - taps.blend(sample))
            })
            .collect();
        let area = rect.to_path();
        self.paint_through_coverage(rect, &coverage, |r| r.fill_path(&area, false));
        true
    }

    /// A host-resident copy of `img` when its samples are device-resident
    /// and drawing them through `ctm` reduces the image; `None` otherwise,
    /// or when the copy fails, in which case the device samples are drawn
    /// as they are.
    #[cfg(feature = "cache")]
    pub(super) fn host_copy_for_reduced_draw(
        &self,
        img: &crate::resources::ImageDescriptor,
        ctm: &Ctm,
    ) -> Option<crate::resources::ImageDescriptor> {
        use crate::resources::image::ImageData;

        let ImageData::Gpu(cached) = &img.data else {
            return None;
        };
        if !image_reduce::draw_reduces(ctm, img.width, img.height) {
            return None;
        }
        let cache = &self.cache_state.as_ref()?.cache;
        let pixels = cache
            .download(cached)
            .map_err(|e| {
                log::warn!(
                    "rasterrocket-interp: reading a cached image back for a reduced draw failed: {e}"
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
        // Decoder contract: bytes.len() == width × height × bpp.  Promote
        // the contract to a release-mode check so a truncated buffer (an
        // adversarial PDF reporting larger dims than the decoder produced)
        // skips the image cleanly instead of indexing past the buffer in
        // the inner loop.
        let bpp = img.color_space.bytes_per_pixel();
        match check_image_bytes_len(img.width, img.height, bpp, img_bytes.len()) {
            Ok(_) => {}
            Err(ImageLenError::DimensionOverflow) => {
                // A decoder-contract violation on an image the codec
                // reported as decoded — surface, never silently blank.
                let msg = format!(
                    "dimensions overflow usize ({}×{}×{bpp} bpp)",
                    img.width, img.height,
                );
                return Err(msg);
            }
            Err(ImageLenError::ShortBuffer { expected }) => {
                let msg = format!(
                    "decoder produced short buffer ({} bytes, expected {expected})",
                    img_bytes.len(),
                );
                return Err(msg);
            }
        }

        let fill_color = self.gstate.current().painted_fill_color();
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
        let backdrop = self.blend_backdrop(self.gstate.current().blend_mode);
        // Channel `ch` of sample `(x, y)`.  Taps lie on the image, so the
        // index is inside the buffer by the length check above; the `get`
        // keeps the loop panic-free regardless.
        let channel = |x: u32, y: u32, ch: usize| {
            img_bytes
                .get((y as usize * img_w + x as usize) * bpp + ch)
                .copied()
                .unwrap_or(u8::MAX)
        };
        let gs_transfer = self.gstate.current().transfer.clone();
        let transfer = |px: [u8; 3]| gs_transfer.as_ref().map_or(px, |t| t.apply(px));
        let compositor = ImageCompositor::new(
            &self.bitmap,
            self.gstate.current().fill_alpha,
            self.gstate.current().blend_mode,
            backdrop.as_deref(),
        );

        // Each row's samples and their coverage are gathered first and
        // composited as one row (see `ImageCompositor::paint_row`).
        let span = (grid.x1 - grid.x0) as usize;
        let mut colours = vec![0u8; span * 3];
        let mut shape = vec![0u8; span];
        for dy in grid.y0..grid.y1 {
            let Some(row_clip) = grid_clip.row(dy) else {
                continue;
            };
            shape.fill(0);
            for ((dx, out), shape) in (grid.x0..grid.x1)
                .zip(colours.as_chunks_mut::<3>().0)
                .zip(shape.iter_mut())
            {
                let coverage = row_clip.coverage(dx);
                if coverage == 0 {
                    continue;
                }
                let Some(taps) = grid.sample(dx, dy) else {
                    continue;
                };
                // Stencil masks paint the fill colour with coverage
                // `255 − sample` and carry no soft mask (asserted at entry).
                let alpha = match img.color_space {
                    ImageColorSpace::Mask => div255(
                        u32::from(u8::MAX - taps.blend(|x, y| channel(x, y, 0)))
                            * u32::from(coverage),
                    ),
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
                    ImageColorSpace::Rgb => {
                        transfer([0, 1, 2].map(|ch| taps.blend(|x, y| channel(x, y, ch))))
                    }
                    ImageColorSpace::Gray => transfer([taps.blend(|x, y| channel(x, y, 0)); 3]),
                    ImageColorSpace::Mask => fill_color,
                };
                *out = src;
                *shape = alpha;
            }
            let (row, alpha_row) = self.bitmap.row_and_alpha_mut(dy);
            compositor.paint_row(row, alpha_row, grid.x0, dy, &colours, &shape);
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

        let backdrop = self.blend_backdrop(self.gstate.current().blend_mode);
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

        if let Err(e) = gpu_ctx.blit_image_to_buffer(
            cached,
            buf,
            bbox,
            grid.cols(),
            grid.rows(),
            grid.filter().bits(),
        ) {
            log::warn!("blit_image: GPU kernel dispatch failed: {e}");
            return false;
        }
        let compositor = ImageCompositor::new(
            &self.bitmap,
            self.gstate.current().fill_alpha,
            self.gstate.current().blend_mode,
            backdrop.as_deref(),
        );
        let transfer = self.gstate.current().transfer.clone();
        let bitmap = &mut self.bitmap;
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

/// Source-over whole rows of a downloaded RGBA8 page buffer, the first of
/// which is page row `y0`, onto `bitmap`, restricted to the grid's columns
/// and to the pixels `grid_clip` covers.  Alpha is `255` where the blit
/// kernel wrote a sample and `0` elsewhere, so the composite copies each
/// written sample, blended by the clip coverage on rows the clip edge
/// crosses.  Samples map through `transfer` when one is in effect.
#[cfg(feature = "cache")]
fn composite_rgba_rows(
    bitmap: &mut Bitmap<Rgb8>,
    rgba: &[u8],
    y0: u32,
    grid: &SampleGrid,
    grid_clip: &mut GridClip<'_>,
    compositor: ImageCompositor<'_>,
    transfer: Option<&crate::resources::transfer::Transfer>,
) {
    use gpu::cache::RGBA_BPP;

    let width = bitmap.width as usize;
    let src_stride = width * RGBA_BPP;
    let span = (grid.x1 - grid.x0) as usize;
    let mut colours = vec![0u8; span * 3];
    let mut shape = vec![0u8; span];
    for (src_row, dy) in rgba.chunks_exact(src_stride.max(1)).zip(y0..) {
        let Some(row_clip) = grid_clip.row(dy) else {
            continue;
        };
        let samples = &src_row[grid.x0 as usize * RGBA_BPP..grid.x1 as usize * RGBA_BPP];
        for (((dx, s), out), shape) in (grid.x0..grid.x1)
            .zip(samples.as_chunks::<RGBA_BPP>().0)
            .zip(colours.as_chunks_mut::<3>().0)
            .zip(shape.iter_mut())
        {
            let rgb = [s[0], s[1], s[2]];
            *out = transfer.map_or(rgb, |t| t.apply(rgb));
            *shape = if s[3] == 0 { 0 } else { row_clip.coverage(dx) };
        }
        let (row, alpha_row) = bitmap.row_and_alpha_mut(dy);
        compositor.paint_row(row, alpha_row, grid.x0, dy, &colours, &shape);
    }
}

/// Composites image samples onto a page or group bitmap.
///
/// An image paints with the nonstroking constant alpha and the blend mode in
/// effect (ISO 32000-1 §11.6.4.4, §11.3.5), and inside a transparency group
/// its shape accumulates into the group's alpha plane like any other object.
/// When none of these apply (no alpha plane, `ca` = 1, `Normal`), a sample is
/// blended onto the destination by its coverage alone; otherwise every sample
/// runs through the raster pipe that fills use.
#[derive(Clone, Copy)]
pub(super) struct ImageCompositor<'a> {
    /// `None` selects the direct coverage blend.
    pub(super) pipe: Option<PipeState<'a>>,
}

impl<'a> ImageCompositor<'a> {
    pub(super) fn new(
        bitmap: &Bitmap<Rgb8>,
        fill_alpha: u8,
        blend_mode: BlendMode,
        backdrop: Option<&'a Backdrop>,
    ) -> Self {
        let direct =
            bitmap.alpha_plane().is_none() && fill_alpha == 255 && blend_mode == BlendMode::Normal;
        Self {
            pipe: (!direct).then(|| PageRenderer::make_pipe(fill_alpha, blend_mode, backdrop)),
        }
    }

    /// Paint the samples of row `y` from column `x0`: `colours` holds the
    /// RGB of each sample and `shape` its coverage, `0` where the sample
    /// paints nothing.  `row` is the row's pixel bytes and `alpha_row` its
    /// alpha plane row, if the bitmap has one.  Columns past the row's end
    /// paint nothing.
    ///
    /// Through the pipe, each run of painted samples composites as one span
    /// whose shape is the coverage: the pipe takes the source alpha as
    /// `div255(a_input × shape)`, the alpha a sample painted alone with
    /// that constant alpha would have.
    pub(super) fn paint_row(
        &self,
        row: &mut [u8],
        mut alpha_row: Option<&mut [u8]>,
        x0: u32,
        y: u32,
        colours: &[u8],
        shape: &[u8],
    ) {
        let x0 = x0 as usize;
        let columns = (row.len() / 3).saturating_sub(x0).min(shape.len());
        let (colours, shape) = (&colours[..columns * 3], &shape[..columns]);
        let Some(pipe) = &self.pipe else {
            let dst = &mut row[x0 * 3..(x0 + columns) * 3];
            for ((dst, src), &coverage) in dst
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(colours.as_chunks::<3>().0)
                .zip(shape)
            {
                if coverage != 0 {
                    blend_rgb(dst, *src, coverage);
                }
            }
            return;
        };
        let mut start = 0;
        while let Some(offset) = shape[start..].iter().position(|&c| c != 0) {
            let a = start + offset;
            let b = shape[a..]
                .iter()
                .position(|&c| c == 0)
                .map_or(columns, |n| a + n);
            let (c0, c1) = (x0 + a, x0 + b);
            #[expect(
                clippy::cast_possible_wrap,
                clippy::cast_possible_truncation,
                reason = "columns and rows lie within the bitmap, whose dimensions are below 2^31"
            )]
            raster::pipe::render_span::<Rgb8>(
                pipe,
                &PipeSrc::Pattern(&RowPattern {
                    data: &colours[a * 3..b * 3],
                }),
                &mut row[c0 * 3..c1 * 3],
                alpha_row.as_deref_mut().map(|al| &mut al[c0..c1]),
                Some(&shape[a..b]),
                c0 as i32,
                (c1 - 1) as i32,
                y as i32,
            );
            start = b;
        }
    }
}

/// Composite `bitmap`, a page group with an alpha plane, onto opaque white
/// in place: each pixel becomes `c·α + 255·(1 − α)`, and the alpha plane is
/// dropped.  A bitmap without an alpha plane is left as it is.
pub(super) fn composite_onto_white(bitmap: &mut Bitmap<Rgb8>) {
    if !bitmap.has_alpha() {
        return;
    }
    for y in 0..bitmap.height {
        let (row, Some(alpha_row)) = bitmap.row_and_alpha_mut(y) else {
            return;
        };
        for (px, &a) in row.as_chunks_mut::<3>().0.iter_mut().zip(&*alpha_row) {
            if a == u8::MAX {
                continue;
            }
            for c in px {
                *c = blend_u8(*c, u8::MAX, u16::from(a));
            }
        }
    }
    bitmap.drop_alpha_plane();
}

/// Paint `src` onto the RGB triple `dst` with coverage `alpha`; an opaque
/// sample copies, anything else blends per channel.
#[inline]
pub(super) fn blend_rgb(dst: &mut [u8], src: [u8; 3], alpha: u8) {
    if alpha == 255 {
        dst.copy_from_slice(&src);
    } else {
        let a = u16::from(alpha);
        // Each output channel blends against its own existing value —
        // the destination is RGB and channels may differ.
        dst[0] = blend_u8(src[0], dst[0], a);
        dst[1] = blend_u8(src[1], dst[1], a);
        dst[2] = blend_u8(src[2], dst[2], a);
    }
}

/// Porter-Duff source-over blend for a single 8-bit channel.
///
/// Formula: `(src * alpha + dst * (255 - alpha)) / 255`.
///
/// Maximum intermediate value: `255 * 255 = 65025 < u16::MAX`, so no overflow
/// is possible regardless of the alpha split.
#[expect(
    clippy::cast_possible_truncation,
    reason = "result is (x*a + y*ia)/255 where a+ia=255; max value is 255*255/255=255, always fits u8"
)]
pub(super) fn blend_u8(src: u8, dst: u8, alpha: u16) -> u8 {
    let ia = 255_u16 - alpha;
    ((u16::from(src) * alpha + u16::from(dst) * ia) / 255) as u8
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
