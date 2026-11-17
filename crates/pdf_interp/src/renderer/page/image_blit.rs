//! Drawing a decoded image or stencil mask onto the page through the clip,
//! on the CPU or through the device blit kernel.

use color::Sample;
use raster::ClipResult;
use raster::pipe::ChannelMask;

use super::super::color::{ColorantPlan, PROCESS_CHANNELS};
use super::super::gstate::Ctm;
use super::PageRenderer;
use super::canvas::{CanvasKind, blend, with_canvas};
use super::depth::CanvasSample;
use super::image_compositor::ImageCompositor;
use super::image_hairline;
use super::image_reduce::{self, DeviceWindow};
use super::image_sampler::{
    BlendedRows, Filter, GridClip, GridView, MaskSampler, SampleGrid, sample_bytes,
};
use super::pixel_rect::PixelRect;
use crate::resources::image::KeptTints;
use crate::resources::{ImageColorSpace, ImageDescriptor, SoftMask, TintData};
use crate::separation::Separator;

impl PageRenderer<'_> {
    /// The tints image decodes keep: on an ink canvas, which draws an
    /// image in a `Separation` or `DeviceN` space as the ink of its tints,
    /// those of the images of ink, and on the canvas of a group of a
    /// `CalRGB` or ICC RGB space, which draws an image of that space as
    /// its components, those of the RGB images as well.
    pub(super) const fn keeps_tints(&self) -> KeptTints {
        if !self.canvas.kind().is_ink() {
            KeptTints::None
        } else if self.ink.separator.has_own_space() {
            KeptTints::InkAndRgb
        } else {
            KeptTints::Ink
        }
    }

    /// Count the drawn image in the page diagnostics (presence, filter) and
    /// raise the source-resolution hint to that of its `source_width`
    /// samples across the page (see [`super::diagnostics::source_ppi`]).
    fn record_image_diagnostics(&mut self, img: &ImageDescriptor, source_width: u32) {
        self.diag.has_images = true;
        let filter_idx = img.filter as usize;
        self.filter_counts[filter_idx] = self.filter_counts[filter_idx].saturating_add(1);
        if let Some(ppi) = super::diagnostics::source_ppi(source_width, self.unrotated_width_pts) {
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
        //
        // The resampling and the sampling are computed on the whole page's
        // device grid, from the CTM and the paintable pixels there, and the
        // window takes the pixels it covers, so a window draws as the page
        // does.  An axis shorter than a device pixel is first widened to
        // one pixel (`image_hairline`), so that the image meets a pixel
        // centre.
        let page_ctm = image_hairline::widen(&self.gstate.current().ctm);
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
        // The spot colorants of an image's tints take their slots when it
        // is drawn, whether or not it reaches a pixel.
        let colorants = self.tint_plan(decoded);
        let filter = Filter::for_draw(&ctm, img.width, img.height, decoded.interpolate);
        let Some(grid) = SampleGrid::new(&ctm, self.grid_view(), img.width, img.height, filter)
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

        let source = self.image_source(decoded, colorants.as_ref(), plan.as_ref(), img, name);
        let separator = self.image_separator(decoded, img);

        // The image is one object under the soft mask in effect, unless its
        // own `/SMask` replaces that mask.
        let paint = |r: &mut Self| {
            let mask = mask_src.map(|m| (m, mask_ctm));
            let drawn = match &source {
                ImageSource::Ink(ink) => {
                    r.composite_ink_image(ink, img, &ctm, mask, &grid, &mut grid_clip);
                    Ok(())
                }
                ImageSource::Samples(overprint) => r.composite_image(
                    img,
                    &ctm,
                    mask,
                    &grid,
                    &mut grid_clip,
                    (*overprint, &separator),
                ),
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

    /// How `img`, the samples of `decoded` drawn under the resampling
    /// `plan`, draws under `colorants`, the plan of its tints.  On an ink
    /// canvas an image in a `Separation` or `DeviceN` space draws the ink
    /// of its tints, resampled as its colours are, and on the canvas of a
    /// group of a `CalRGB` or ICC RGB space an image of that space the
    /// complements of its components.  One whose spot
    /// colorants found no slot draws its sRGB samples as process ink in
    /// the process channels, as a colour of its space does; any other
    /// image draws its sRGB samples in every channel.
    fn image_source(
        &self,
        decoded: &crate::resources::DecodedImage,
        colorants: Option<&ColorantPlan>,
        plan: Option<&image_reduce::Plan>,
        img: &ImageDescriptor,
        name: Option<&[u8]>,
    ) -> ImageSource {
        match colorants {
            Some(
                colorants @ (ColorantPlan::Channels(_)
                | ColorantPlan::Linked(_)
                | ColorantPlan::Complements),
            ) => {
                let samples = img.width as usize * img.height as usize;
                if let Some(ink) =
                    ink_samples(decoded, colorants, plan).filter(|ink| ink.len() == samples * 8)
                {
                    return ImageSource::Ink(ink);
                }
                log::debug!(
                    "rasterrocket-interp: blit_image: the tints of image {:?} give no ink \
                     for its {}×{} samples; it draws as its sRGB samples",
                    name.map(String::from_utf8_lossy),
                    img.width,
                    img.height,
                );
                ImageSource::Samples(None)
            }
            Some(ColorantPlan::Process(_)) => {
                ImageSource::Samples(self.overprint_channels(false, PROCESS_CHANNELS))
            }
            None => ImageSource::Samples(None),
        }
    }

    /// The separator of `img`, the samples of `decoded`: a stencil paints
    /// the fill colour; an image's samples separate as grays or as colours,
    /// under the intent they were converted under.
    fn image_separator(
        &self,
        decoded: &crate::resources::DecodedImage,
        img: &ImageDescriptor,
    ) -> Separator {
        if img.color_space == ImageColorSpace::Mask {
            return self.fill_separator();
        }
        self.separator_for(
            img.color_space == ImageColorSpace::Gray,
            Some(decoded.intent),
        )
    }

    /// The plan by which the colorant tints of `decoded` paint on the ink
    /// canvas (see `InkMap::plan`), assigning their spot colorants slots.
    /// `None` on an RGB canvas, for an image without tints or whose space
    /// paints its sRGB samples, and in a shape-only glyph.
    fn tint_plan(&self, decoded: &crate::resources::DecodedImage) -> Option<ColorantPlan> {
        if !self.canvas.kind().is_ink() || self.gstate.current().colour_locked {
            return None;
        }
        let tints = decoded.tints.as_ref()?;
        let separator = self.separator_for(false, Some(decoded.intent));
        self.ink_map
            .borrow_mut()
            .plan(self.resources.doc(), &tints.space, &separator)
    }

    /// Composite the ink samples `ink`, on the grid of `img`, which they
    /// were resampled with, sampled through `grid` under `ctm`, onto the
    /// ink canvas within `grid_clip`, writing their channels under
    /// overprint; `mask` is the image's soft mask with the matrix that
    /// draws it.  The samples are blended at their own depth and their
    /// shape taken at the canvas's (see [`Self::composite_host`]).
    fn composite_ink_image(
        &mut self,
        ink: &InkSamples,
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
                    self.grid_view(),
                    img.width,
                    img.height,
                ) else {
                    return;
                };
                Some(sampler)
            }
            None => None,
        };
        let mask = mask.as_ref();
        let deep = self.canvas.kind().depth() == color::SampleDepth::Sixteen;
        let paints = ink.paints;
        match (&ink.data, deep) {
            (InkData::U8(d), false) => {
                self.composite_ink_host::<_, u8>(d, paints, img, mask, grid, grid_clip);
            }
            (InkData::U8(d), true) => {
                self.composite_ink_host::<_, u16>(d, paints, img, mask, grid, grid_clip);
            }
            (InkData::U16(d), false) => {
                self.composite_ink_host::<_, u8>(d, paints, img, mask, grid, grid_clip);
            }
            (InkData::U16(d), true) => {
                self.composite_ink_host::<_, u16>(d, paints, img, mask, grid, grid_clip);
            }
        }
    }

    /// [`Self::composite_ink_image`] of the ink `data`, eight samples of
    /// type `T` a pixel painting the channels `paints`, its alpha as
    /// samples of type `A`.
    fn composite_ink_host<T: CanvasSample, A: CanvasSample>(
        &mut self,
        data: &[T],
        paints: ChannelMask,
        img: &crate::resources::ImageDescriptor,
        mask: Option<&MaskSampler<'_>>,
        grid: &SampleGrid,
        grid_clip: &mut GridClip<'_>,
    ) {
        let (fill_alpha, blend_mode, transfer) = {
            let gs = self.gstate.current();
            (gs.fill_alpha, gs.blend_mode, gs.transfer.clone())
        };
        let overprint = self.overprint_channels(false, paints);
        let paint = self.paint_params(fill_alpha, blend_mode, overprint, paints);
        let separator = self.separator(false);
        let mut compositor = with_canvas!(&self.canvas, |b| ImageCompositor::new(
            b, &paint, &separator
        ));
        compositor.transfer = transfer.as_deref();
        compositor.ink_samples = Some(paints);
        let img_w = img.width as usize;
        let at = |x, y| sample_bytes::<T, 8>(data, img_w, 8, x, y);
        let span = (grid.x1 - grid.x0) as usize;
        let mut colours = vec![T::ZERO; span * 8];
        let mut shape = vec![A::ZERO; span];
        for dy in grid.y0..grid.y1 {
            let Some(row_clip) = grid_clip.row(dy) else {
                continue;
            };
            shape.fill(A::ZERO);
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
                let alpha = A::div_max(
                    mask.map_or(A::MAX, |m| m.alpha_at(dx, dy, &taps)).wide()
                        * A::from_u8(coverage).wide(),
                );
                if alpha == A::ZERO {
                    continue;
                }
                *out = taps.blend_px(at);
                *shape = alpha;
            }
            with_canvas!(&mut self.canvas, |b| compositor
                .paint_row(b, grid.x0, dy, &colours, &shape));
        }
    }

    /// The bitmap images are sampled onto: this renderer's window of the
    /// page.
    pub(super) const fn grid_view(&self) -> GridView {
        GridView {
            origin: self.window_origin,
            width: self.width,
            height: self.height,
            page_height: self.page_height,
        }
    }

    /// The device pixels an image draw can paint, on the whole page's
    /// device grid: the bitmap within the clip rectangle, which bounds
    /// every path clip.
    fn paintable_window(&self) -> DeviceWindow {
        let clip = &self.gstate.current().clip;
        let (ox, oy) = (
            i64::from(self.window_origin.0),
            i64::from(self.window_origin.1),
        );
        let x = |v: i32| i64::from(v).clamp(0, i64::from(self.width)) + ox;
        let y = |v: i32| i64::from(v).clamp(0, i64::from(self.height)) + oy;
        DeviceWindow::of_bitmap_rect(
            self.page_width,
            self.page_height,
            [
                x(clip.x_min_i),
                y(clip.y_min_i),
                x(clip.x_max_i.saturating_add(1)),
                y(clip.y_max_i.saturating_add(1)),
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
                    u8::MAX - taps.blend_px(|x, y| sample_bytes::<_, 1>(bytes, img_w, 1, x, y))[0]
                })
            })
            .collect();
        let area = rect.to_path();
        self.paint_through_coverage(rect, &coverage, |r| r.fill_path(&area, false));
        Ok(true)
    }

    /// Composite `img`, sampled through `grid` under `ctm`, onto the bitmap
    /// within `grid_clip`; `mask` is its soft mask with the matrix that
    /// draws it.  An image's samples write the channels `overprint`, every
    /// channel for `None`; a stencil writes those of the fill colour.
    /// Colours not given as ink are separated by `separator` on an ink
    /// canvas.
    /// `Err` carries the reason a decoded image cannot be drawn, for the
    /// caller to record against the image's name.
    pub(super) fn composite_image(
        &mut self,
        img: &crate::resources::ImageDescriptor,
        ctm: &Ctm,
        mask: Option<(&SoftMask, Ctm)>,
        grid: &SampleGrid,
        grid_clip: &mut GridClip<'_>,
        (overprint, separator): (Option<ChannelMask>, &Separator),
    ) -> Result<(), String> {
        // A page recorded for the device raster takes the host copy of
        // device-resident samples.
        #[cfg(all(feature = "cache", feature = "vulkan"))]
        if self.device_recording() && matches!(img.data, crate::resources::image::ImageData::Gpu(_))
        {
            let Some(host) = self.host_copy(img, "a recorded page") else {
                self.refuse_device("device-resident image");
                return Ok(());
            };
            return self.composite_image(&host, ctm, mask, grid, grid_clip, (overprint, separator));
        }
        // Device-resident samples are drawn by the device blit, or through
        // the CPU path below from their host copy.
        #[cfg(feature = "cache")]
        if let crate::resources::image::ImageData::Gpu(cached) = &img.data {
            let host =
                self.draw_device_image(img, cached, grid, grid_clip, (overprint, separator))?;
            return host.map_or(Ok(()), |host| {
                self.composite_image(&host, ctm, mask, grid, grid_clip, (overprint, separator))
            });
        }

        // CPU path: needs host-resident pixels.
        let wide = img.data.as_cpu16();
        let bytes = img.data.as_cpu();
        let len = match (wide, bytes) {
            (Some(wide), _) => wide.len(),
            (None, Some(bytes)) => bytes.len(),
            // Decoded pixels exist but are device-resident with no GPU
            // dispatch available to blit them — unrenderable, not absent.
            (None, None) => {
                return Err("pixels device-resident but GPU dispatch unavailable".to_owned());
            }
        };
        verify_sample_len(img, len)?;
        let mask = match mask {
            Some((m, mask_ctm)) => {
                let Some(sampler) = MaskSampler::new(
                    m,
                    &mask_ctm,
                    ctm,
                    grid.filter(),
                    self.grid_view(),
                    img.width,
                    img.height,
                ) else {
                    return Ok(());
                };
                Some(sampler)
            }
            None => None,
        };
        #[cfg(feature = "vulkan")]
        if self.device_recording() {
            self.record_device_image(img, (wide, bytes), mask.as_ref(), grid);
            return Ok(());
        }
        // The shape is taken at the canvas's depth, so a 16-bit mask keeps
        // its levels whatever the depth of the image's samples.
        let deep = self.canvas.kind().depth() == color::SampleDepth::Sixteen;
        let mask = mask.as_ref();
        match (wide, bytes, deep) {
            (Some(s), _, true) => {
                self.composite_host::<_, u16>(
                    img,
                    s,
                    mask,
                    grid,
                    grid_clip,
                    (overprint, separator),
                );
            }
            (Some(s), _, false) => {
                self.composite_host::<_, u8>(img, s, mask, grid, grid_clip, (overprint, separator));
            }
            (None, Some(s), true) => {
                self.composite_host::<_, u16>(
                    img,
                    s,
                    mask,
                    grid,
                    grid_clip,
                    (overprint, separator),
                );
            }
            (None, Some(s), false) => {
                self.composite_host::<_, u8>(img, s, mask, grid, grid_clip, (overprint, separator));
            }
            (None, None, _) => {}
        }
        Ok(())
    }

    /// Composite the host-resident `samples` of `img`, of type `T`, as
    /// [`Self::composite_image`] does: the sampling and the transfer
    /// function of an 8-bit RGB canvas at the depth of the samples, the
    /// alpha as samples of type `A`, the compositing at the canvas's depth
    /// (see [`ImageCompositor::paint_row`]).
    #[expect(
        clippy::too_many_lines,
        reason = "the compositor set-up, the mask sampler and the per-pixel loop share the grid and the sample buffers"
    )]
    fn composite_host<T: CanvasSample, A: CanvasSample>(
        &mut self,
        img: &crate::resources::ImageDescriptor,
        samples_of: &[T],
        mask: Option<&MaskSampler<'_>>,
        grid: &SampleGrid,
        grid_clip: &mut GridClip<'_>,
        (overprint, separator): (Option<ChannelMask>, &Separator),
    ) {
        let bpp = img.color_space.bytes_per_pixel();
        let painted_fill = self.gstate.current().painted_fill_color();
        let fill_color = painted_fill.bytes().map(T::from_u8);
        let img_w = img.width as usize;
        // A stencil paints the fill colour under the fill's overprint; an
        // image's samples write the channels `overprint`.
        let stencil = img.color_space == ImageColorSpace::Mask;
        let (fill_alpha, blend_mode) = {
            let gs = self.gstate.current();
            (gs.fill_alpha, gs.blend_mode)
        };
        let paint = if stencil {
            self.solid_paint(false, &painted_fill, fill_alpha, blend_mode)
        } else {
            self.paint_params(fill_alpha, blend_mode, overprint, PROCESS_CHANNELS)
        };
        // Taps lie on the image, so every sample is inside the buffer by
        // the length check above.
        let gray = |x, y| sample_bytes::<T, 1>(samples_of, img_w, bpp, x, y);
        let rgb = |x, y| sample_bytes::<T, 3>(samples_of, img_w, bpp, x, y);
        // On an RGB canvas the samples map through the transfer function
        // here; on an ink canvas the compositor maps their ink.
        let gs_transfer = self.gstate.current().transfer.clone();
        let kind = self.canvas.kind();
        let rgb_transfer = gs_transfer.as_deref().filter(|_| kind == CanvasKind::Rgb);
        let transfer = |px: [T; 3]| {
            rgb_transfer.map_or(px, |t| {
                let tables = T::rgb_tables(t);
                std::array::from_fn(|c| T::lookup(tables[c], px[c]))
            })
        };
        let mut compositor =
            with_canvas!(&self.canvas, |b| ImageCompositor::new(b, &paint, separator));
        // On an ink canvas a stencil paints the fill colour as its ink, on
        // a 16-bit canvas as its 16-bit colour; there the samples map
        // through the 16-bit tables once widened.
        match kind {
            CanvasKind::Rgb => {}
            CanvasKind::Rgb16 => {
                if stencil {
                    compositor.solid = Some(&painted_fill);
                }
                compositor.wide_transfer = gs_transfer.as_deref();
            }
            CanvasKind::Ink | CanvasKind::Ink16 => {
                if stencil {
                    compositor.solid = Some(&painted_fill);
                }
                compositor.transfer = gs_transfer.as_deref();
            }
        }

        // Each row's samples and their coverage are gathered first and
        // composited as one row (see `ImageCompositor::paint_row`).
        let span = (grid.x1 - grid.x0) as usize;
        let mut colours = vec![T::ZERO; span * 3];
        let mut shape = vec![A::ZERO; span];
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
            shape.fill(A::ZERO);
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
                    RowSamples::Gray(row) => row.get(col).map_or(T::MAX, |p| p[0]),
                    _ => taps.blend_px(gray)[0],
                };
                // Stencil masks paint the fill colour with coverage
                // `MAX − sample` and carry no soft mask (asserted at entry).
                let coverage = A::from_u8(coverage).wide();
                let alpha = match img.color_space {
                    ImageColorSpace::Mask => {
                        A::div_max((A::max_wide() - A::from_sample(gray_at()).wide()) * coverage)
                    }
                    ImageColorSpace::Rgb | ImageColorSpace::Gray => A::div_max(
                        mask.map_or(A::MAX, |m| m.alpha_at(dx, dy, &taps)).wide() * coverage,
                    ),
                };
                if alpha == A::ZERO {
                    continue;
                }
                // Image samples map through the transfer function; the
                // stencil colour is already the painted fill colour.
                let src = match img.color_space {
                    ImageColorSpace::Rgb => transfer(match samples {
                        RowSamples::Rgb(row) => row.get(col).copied().unwrap_or([T::MAX; 3]),
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
    }
}

/// The per-row sample cache of a separable image draw, by channel count.
enum RowBlend<'a, T: Sample> {
    Rgb(BlendedRows<'a, 3, T>),
    Gray(BlendedRows<'a, 1, T>),
}

/// The samples of one device row: blended a row at a time, or left to each
/// pixel's own taps.
#[derive(Clone, Copy)]
enum RowSamples<'a, T> {
    Rgb(&'a [[T; 3]]),
    Gray(&'a [[T; 1]]),
    PerPixel,
}

/// How an image draws (see [`PageRenderer::image_source`]).
enum ImageSource {
    /// The ink of its tints.
    Ink(InkSamples),
    /// Its sRGB samples, writing the channels given, every channel for
    /// `None`.
    Samples(Option<ChannelMask>),
}

/// The ink of an image's samples, eight samples a pixel, and the channels
/// it paints.
struct InkSamples {
    /// The ink, at the depth of the image's tints.
    data: InkData,
    /// The channels the image paints.
    paints: ChannelMask,
}

impl InkSamples {
    /// The number of ink samples.
    const fn len(&self) -> usize {
        match &self.data {
            InkData::U8(d) => d.len(),
            InkData::U16(d) => d.len(),
        }
    }
}

/// Ink samples of either depth.
enum InkData {
    /// 8-bit ink.
    U8(Vec<u8>),
    /// 16-bit ink.
    U16(Vec<u16>),
}

/// The ink of each sample of the colorant tints of `decoded` under
/// `colorants`, their plan, resampled through `plan` when the draw reduces
/// the image, at the depth of the tints.  `None` for an image without
/// tints, under [`ColorantPlan::Process`], and when the samples cannot be
/// resampled or held.
fn ink_samples(
    decoded: &crate::resources::DecodedImage,
    colorants: &ColorantPlan,
    plan: Option<&image_reduce::Plan>,
) -> Option<InkSamples> {
    let tints = decoded.tints.as_ref()?;
    let n = tints.space.ncomponents();
    let data = match &tints.data {
        TintData::U8(t) => InkData::U8(tint_ink(t, n, colorants, plan)?),
        TintData::U16(t) => InkData::U16(tint_ink(t, n, colorants, plan)?),
    };
    Some(InkSamples {
        data,
        paints: colorants.paints(),
    })
}

/// The ink of `tints`, `n` a pixel, under `colorants`, resampled through
/// `plan` first when there is one.
fn tint_ink<S: color::Sample>(
    tints: &[S],
    n: usize,
    colorants: &ColorantPlan,
    plan: Option<&image_reduce::Plan>,
) -> Option<Vec<S>> {
    let resampled;
    let data = match plan {
        Some(plan) => {
            resampled = plan.resample(tints, n)?;
            &resampled
        }
        None => tints,
    };
    let mut ink = Vec::new();
    ink.try_reserve_exact(data.len() / n * 8).ok()?;
    for px in data.chunks_exact(n) {
        ink.extend_from_slice(&colorants.ink_of_tints(px)?);
    }
    Some(ink)
}

/// Checks the decoder contract that `len` samples, of 8 or 16 bits, hold
/// `img`'s pixels, so that a truncated buffer (a stream declaring larger
/// dimensions than its data) fails the image instead of painting a partly
/// blank one.
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
            "decoder produced short buffer ({len} samples, expected {expected})"
        )),
    }
}

/// Paint `src` onto the RGB triple `dst` with coverage `alpha`; an opaque
/// sample copies, anything else blends per channel.
#[inline]
pub(super) fn blend_rgb<S: Sample>(dst: &mut [S], src: &[S], alpha: S) {
    if alpha == S::MAX {
        dst.copy_from_slice(&src[..3]);
    } else {
        // Each output channel blends against its own existing value —
        // the destination is RGB and channels may differ.
        dst[0] = blend(src[0], dst[0], alpha);
        dst[1] = blend(src[1], dst[1], alpha);
        dst[2] = blend(src[2], dst[2], alpha);
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
