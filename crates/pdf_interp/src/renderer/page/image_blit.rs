//! Drawing a decoded image or stencil mask onto the page through the clip,
//! on the CPU or through the device blit kernel.

use color::convert::div255;
use raster::ClipResult;
use raster::pipe::ChannelMask;

use super::super::color::PROCESS_CHANNELS;
use super::super::gstate::Ctm;
use super::PageRenderer;
use super::canvas::{CanvasKind, blend_u8, with_canvas};
use super::image_compositor::ImageCompositor;
use super::image_reduce::{self, DeviceWindow};
use super::image_sampler::{BlendedRows, Filter, GridClip, MaskSampler, SampleGrid, sample_bytes};
use super::pixel_rect::PixelRect;
use crate::resources::{ImageColorSpace, ImageDescriptor, SoftMask};

impl PageRenderer<'_> {
    /// Whether image decodes keep their colorant tints: on an ink canvas,
    /// which draws an image in a `Separation` or `DeviceN` space as the
    /// ink of its tints.
    pub(super) fn keeps_tints(&self) -> bool {
        self.canvas.kind() == CanvasKind::Ink
    }

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
        let plan = image_reduce::Plan::new(&page_ctm, img.width, img.height, &window);
        // A device-resident image is sampled at full resolution by the
        // blit kernel.  A draw that reduces it takes a host copy so that it
        // is area-averaged as a host-resident image is.
        #[cfg(feature = "cache")]
        let host_copy = plan
            .as_ref()
            .and_then(|_| self.host_copy(img, "a reduced draw"));
        #[cfg(feature = "cache")]
        let img = host_copy.as_ref().unwrap_or(img);
        let reduced = plan
            .as_ref()
            .and_then(|plan| image_reduce::reduce_for_draw(img, plan));
        let (img, ctm) = match &reduced {
            Some((i, c)) => (i, *c),
            None => (img, page_ctm),
        };
        // On an ink canvas an image in a `Separation` or `DeviceN` space
        // draws the ink of its tints, resampled as its colours are.
        let ink = self
            .ink_samples(decoded, plan.as_ref())
            .filter(|(ink, _)| ink.len() == img.width as usize * img.height as usize * 8);
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
            let mask = mask_src.map(|m| (m, mask_ctm));
            let drawn = match &ink {
                Some((samples, paints)) => {
                    r.composite_ink_image(samples, *paints, img, &ctm, mask, &grid, &mut grid_clip);
                    Ok(())
                }
                None => r.composite_image(img, &ctm, mask, &grid, &mut grid_clip),
            };
            if let Err(reason) = drawn {
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

    /// The ink of each sample of the colorant tints of `decoded` on the ink
    /// canvas, resampled through `plan` when the draw reduces the image,
    /// eight bytes per sample, with the channels the image paints.  `None`
    /// on an RGB canvas, for an image without tints, and for one whose
    /// spot colorants find no slot: such an image draws as its sRGB
    /// samples.
    fn ink_samples(
        &self,
        decoded: &crate::resources::DecodedImage,
        plan: Option<&image_reduce::Plan>,
    ) -> Option<(Vec<u8>, ChannelMask)> {
        if self.canvas.kind() != CanvasKind::Ink || self.gstate.current().colour_locked {
            return None;
        }
        let tints = decoded.tints.as_ref()?;
        let colorants = self
            .ink_map
            .borrow_mut()
            .plan(self.resources.doc(), &tints.space);
        let n = tints.space.ncomponents();
        let resampled;
        let data = match plan {
            Some(plan) => {
                resampled = plan.resample(&tints.data, n)?;
                &resampled
            }
            None => &tints.data,
        };
        let mut ink = Vec::new();
        ink.try_reserve_exact(data.len() / n * 8).ok()?;
        for px in data.chunks_exact(n) {
            ink.extend_from_slice(&colorants.ink_of_tints(px)?);
        }
        Some((ink, colorants.paints()))
    }

    /// Composite the ink samples `ink`, eight bytes per sample on the grid
    /// of `img`, which it was resampled with, sampled through `grid` under
    /// `ctm`, onto the ink canvas within `grid_clip`, writing the channels
    /// `paints` under overprint; `mask` is the image's soft mask with the
    /// matrix that draws it.
    #[expect(
        clippy::too_many_arguments,
        reason = "the samples, their channels and the draw state composite_image takes"
    )]
    fn composite_ink_image(
        &mut self,
        ink: &[u8],
        paints: ChannelMask,
        img: &crate::resources::ImageDescriptor,
        ctm: &Ctm,
        mask: Option<(&SoftMask, Ctm)>,
        grid: &SampleGrid,
        grid_clip: &mut GridClip<'_>,
    ) {
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
                    return;
                };
                Some(sampler)
            }
            None => None,
        };
        let (fill_alpha, blend_mode, transfer) = {
            let gs = self.gstate.current();
            (gs.fill_alpha, gs.blend_mode, gs.transfer.clone())
        };
        let overprint = self.overprint_channels(false, paints);
        let paint = self.paint_params(fill_alpha, blend_mode, overprint, paints);
        let mut compositor = with_canvas!(&self.canvas, |b| ImageCompositor::new(b, &paint));
        compositor.transfer = transfer.as_deref();
        compositor.ink_samples = Some(paints);
        let img_w = img.width as usize;
        let at = |x, y| sample_bytes::<8>(ink, img_w, 8, x, y);
        let span = (grid.x1 - grid.x0) as usize;
        let mut colours = vec![0u8; span * 8];
        let mut shape = vec![0u8; span];
        for dy in grid.y0..grid.y1 {
            let Some(row_clip) = grid_clip.row(dy) else {
                continue;
            };
            shape.fill(0);
            for ((dx, out), shape) in (grid.x0..grid.x1)
                .zip(colours.as_chunks_mut::<8>().0)
                .zip(shape.iter_mut())
            {
                let coverage = row_clip.coverage(dx);
                if coverage == 0 {
                    continue;
                }
                let Some(taps) = grid.sample(dx, dy) else {
                    continue;
                };
                let alpha = div255(
                    u32::from(mask.as_ref().map_or(u8::MAX, |m| m.alpha(dx, dy, &taps)))
                        * u32::from(coverage),
                );
                if alpha == 0 {
                    continue;
                }
                *out = taps.blend_px(at);
                *shape = alpha;
            }
            with_canvas!(&mut self.canvas, |b| compositor
                .paint_row(b, grid.x0, dy, &colours, &shape));
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
        // GPU image-blit path: device-resident pixels sampled by the blit
        // kernel of the cache's backend through the same grid, composited onto `bitmap` before
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
                self.diag.device_image_blits = self.diag.device_image_blits.saturating_add(1);
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
            self.paint_params(fill_alpha, blend_mode, None, PROCESS_CHANNELS)
        };
        // Taps lie on the image, so every sample is inside the buffer by
        // the length check above.
        let gray = |x, y| sample_bytes::<1>(img_bytes, img_w, bpp, x, y);
        let rgb = |x, y| sample_bytes::<3>(img_bytes, img_w, bpp, x, y);
        // On an RGB canvas the samples map through the transfer function
        // here; on an ink canvas the compositor maps their ink.
        let gs_transfer = self.gstate.current().transfer.clone();
        let kind = self.canvas.kind();
        let rgb_transfer = gs_transfer.as_deref().filter(|_| kind == CanvasKind::Rgb);
        let transfer = |px: [u8; 3]| rgb_transfer.map_or(px, |t| t.apply(px));
        let mut compositor = with_canvas!(&self.canvas, |b| ImageCompositor::new(b, &paint));
        // On an ink canvas a stencil paints the fill colour as its ink.
        if kind == CanvasKind::Ink {
            if stencil {
                compositor.solid = Some(painted_fill.bytes_for(kind.mode()));
            }
            compositor.transfer = gs_transfer.as_deref();
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
