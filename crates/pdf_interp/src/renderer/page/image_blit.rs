//! Drawing a decoded image or stencil mask onto the page through the clip,
//! on the CPU or through the device blit kernel.
//!
//! The draw resamples an image drawn below its resolution, builds its
//! [`SampleGrid`] and clip, and hands the samples to one of two
//! composites: the ink of an image's colorant tints on an ink canvas
//! (`ink`), or its sRGB samples, or a stencil's fill colour, on any canvas
//! (`composite`).

use raster::ClipResult;
use raster::pipe::ChannelMask;

use super::super::color::{ColorantPlan, PROCESS_CHANNELS};
use super::PageRenderer;
use super::image_hairline;
use super::image_reduce::{self, DeviceWindow};
use super::image_sampler::{Filter, GridClip, GridView, SampleGrid};
use super::pixel_rect::PixelRect;
use crate::resources::image::KeptTints;
use crate::resources::{ImageColorSpace, ImageDescriptor};
use crate::separation::Separator;
use ink::{InkSamples, ink_samples};

mod composite;
mod ink;

pub(super) use composite::blend_rgb;
#[cfg(test)]
pub(super) use composite::{ImageLenError, check_image_bytes_len};

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
}

/// How an image draws (see [`PageRenderer::image_source`]).
enum ImageSource {
    /// The ink of its tints.
    Ink(InkSamples),
    /// Its sRGB samples, writing the channels given, every channel for
    /// `None`.
    Samples(Option<ChannelMask>),
}
