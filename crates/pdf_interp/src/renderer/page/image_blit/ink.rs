//! Drawing an image as the ink of its colorant tints, on an ink canvas.
//!
//! The ink is computed per source sample, eight samples a pixel at the
//! depth of the tints, resampled as the image's colours are when the draw
//! reduces it, and sampled through the image's grid; each row is
//! composited with the shape taken at the canvas's depth.

use raster::pipe::ChannelMask;

use super::super::PageRenderer;
use super::super::canvas::with_canvas;
use super::super::depth::CanvasSample;
use super::super::image_compositor::ImageCompositor;
use super::super::image_reduce;
use super::super::image_sampler::{GridClip, MaskSampler, SampleGrid, sample_bytes};
use crate::renderer::color::ColorantPlan;
use crate::renderer::gstate::Ctm;
use crate::resources::{SoftMask, TintData};

impl PageRenderer<'_> {
    /// Composite the ink samples `ink`, on the grid of `img`, which they
    /// were resampled with, sampled through `grid` under `ctm`, onto the
    /// ink canvas within `grid_clip`, writing their channels under
    /// overprint; `mask` is the image's soft mask with the matrix that
    /// draws it.  The samples are blended at their own depth and their
    /// shape taken at the canvas's (see [`Self::composite_host`]).
    pub(super) fn composite_ink_image(
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
}

/// The ink of an image's samples, eight samples a pixel, and the channels
/// it paints.
pub(super) struct InkSamples {
    /// The ink, at the depth of the image's tints.
    data: InkData,
    /// The channels the image paints.
    paints: ChannelMask,
}

impl InkSamples {
    /// The number of ink samples.
    pub(super) const fn len(&self) -> usize {
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
pub(super) fn ink_samples(
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
