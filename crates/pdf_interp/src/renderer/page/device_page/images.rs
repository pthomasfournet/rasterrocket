//! Images recorded for the whole-page GPU raster.
//!
//! An image the CPU composites from host-resident samples through a
//! [`SampleGrid`] with tables is recorded as an image command: its
//! samples, already decoded, colour-converted and resampled for the draw,
//! are held by the page, and the command carries the grid's tables, filter
//! and rectangle, its soft mask's samples read through the image's taps or
//! through the mask's own grid, as [`MaskSampler`] reads them, and the
//! composite `ImageCompositor` takes on an opaque RGB canvas: directly at
//! full constant alpha, else through the pipe.  The device samples each
//! pixel with the CPU's integer arithmetic, so it paints the samples, the
//! shapes and the composites the CPU does.
//!
//! The pixels the command may paint are the grid's within the clip
//! rectangle's integer bounds: a rectangle clip's row coverage is 255 or 0
//! by column there, and an alpha taken by `div_max(alpha · 255)` at 8 bits
//! or `div_max(alpha · 65535)` at 16 is itself.

use color::{Sample, SampleDepth};
use gpu::page::{ImageDraw, ImageLayout, ImageMask, PixelBounds, SampleMap, Samples};
use raster::types::BlendMode;

use super::super::PageRenderer;
use super::super::image_sampler::{MaskSampler, SampleGrid};
use super::command_refusal;
use crate::resources::{ImageColorSpace, ImageDescriptor};

/// The tables of `grid` as the page raster reads them; `None` for a grid
/// mapped per pixel.
#[expect(
    clippy::cast_possible_wrap,
    reason = "grid edges are clamped to the page, which fits i32"
)]
fn sample_map(grid: &SampleGrid) -> Option<SampleMap<'_>> {
    let (cols, rows) = grid.tables()?;
    let filter = grid.filter();
    Some(SampleMap {
        x0: grid.x0 as i32,
        y0: grid.y0 as i32,
        x1: grid.x1 as i32,
        y1: grid.y1 as i32,
        cols,
        rows,
        filter_x: filter.x,
        filter_y: filter.y,
    })
}

impl PageRenderer<'_> {
    /// Record the draw that `composite_host` would composite: `img`, whose
    /// host-resident samples are `wide` (16-bit) or `bytes`, sampled
    /// through `grid` under the soft mask `mask`, with the constant alpha
    /// and clip in effect.
    pub(in crate::renderer::page) fn record_device_image(
        &mut self,
        img: &ImageDescriptor,
        (wide, bytes): (Option<&[u16]>, Option<&[u8]>),
        mask: Option<&MaskSampler<'_>>,
        grid: &SampleGrid,
    ) {
        if let Err(reason) = self.try_record_device_image(img, (wide, bytes), mask, grid) {
            self.refuse_device(reason);
        }
    }

    /// [`Self::record_device_image`], or the reason the draw cannot be
    /// recorded.
    fn try_record_device_image(
        &mut self,
        img: &ImageDescriptor,
        (wide, bytes): (Option<&[u16]>, Option<&[u8]>),
        mask: Option<&MaskSampler<'_>>,
        grid: &SampleGrid,
    ) -> Result<(), &'static str> {
        let stencil = img.color_space == ImageColorSpace::Mask;
        let (fill_alpha, blend_mode, transfer, gs_clip, clip) = {
            let gs = self.gstate.current();
            (
                gs.fill_alpha,
                gs.blend_mode,
                gs.transfer.is_some(),
                gs.clip.clone_shared(),
                PixelBounds {
                    x0: gs.clip.x_min_i,
                    y0: gs.clip.y_min_i,
                    x1: gs.clip.x_max_i,
                    y1: gs.clip.y_max_i,
                },
            )
        };
        // A stencil paints the fill colour, already mapped through the
        // transfer function; an image's samples map through it.
        let colour = if stencil {
            self.solid_device_paint_at(false, fill_alpha)?.rgb
        } else {
            if blend_mode != BlendMode::Normal {
                return Err("blend mode");
            }
            if transfer {
                return Err("image transfer function");
            }
            [0; 3]
        };
        let depth = self.canvas.kind().depth();
        let deep = depth == SampleDepth::Sixteen;
        let a_input = match depth {
            SampleDepth::Eight => u16::from(fill_alpha),
            SampleDepth::Sixteen => u16::from_u8(fill_alpha),
        };
        let map = sample_map(grid).ok_or("image sampling range")?;
        let mask_map = match mask.and_then(|m| m.grid.as_ref()) {
            Some(g) => Some(sample_map(g).ok_or("image sampling range")?),
            None => None,
        };
        let channels = img.color_space.bytes_per_pixel();
        let count = img.width as usize * img.height as usize * channels;
        let samples = match (wide, bytes) {
            (Some(_), _) if !deep => return Err("16-bit samples on an 8-bit canvas"),
            (Some(s), _) => Samples::Sixteen(&s[..count]),
            (None, Some(s)) => Samples::Eight(&s[..count]),
            (None, None) => return Ok(()),
        };
        let layout = match img.color_space {
            ImageColorSpace::Rgb => ImageLayout::Rgb,
            ImageColorSpace::Gray => ImageLayout::Gray,
            ImageColorSpace::Mask => ImageLayout::Stencil,
        };
        let channels = u32::try_from(channels).map_err(|_| "image sampling range")?;
        let Some(recording) = self.device_page.as_mut() else {
            return Ok(());
        };
        recording.use_clip(&gs_clip)?;
        let refusal = |e| command_refusal(&e);
        let samples = recording
            .commands
            .add_samples(samples, img.width, img.height, channels)
            .map_err(refusal)?;
        let mask = match mask {
            None => ImageMask::None,
            Some(soft) => {
                let (mask_w, mask_h) = soft
                    .grid
                    .as_ref()
                    .map_or((img.width, img.height), SampleGrid::source_size);
                let alpha = soft
                    .alpha16
                    .filter(|_| deep)
                    .map_or(Samples::Eight(soft.alpha), Samples::Sixteen);
                let held = recording
                    .commands
                    .add_samples(alpha, mask_w, mask_h, 1)
                    .map_err(refusal)?;
                mask_map.map_or(ImageMask::Taps(held), |map| ImageMask::Own(held, map))
            }
        };
        recording
            .commands
            .push_image(&ImageDraw {
                samples,
                layout,
                map,
                mask,
                bounds: clip,
                colour,
                a_input,
                direct: fill_alpha == u8::MAX,
            })
            .map_err(refusal)
    }
}
