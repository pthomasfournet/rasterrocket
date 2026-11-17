//! Compositing an image's sRGB samples, or a stencil's fill colour, onto
//! the canvas through the image's sampling grid and the clip.
//!
//! Host-resident samples composite on the CPU a device row at a time;
//! device-resident ones go to the device blit or, failing it, through
//! their host copy.  A sample buffer is checked against the image's
//! dimensions before any pixel is read.

use color::Sample;
use raster::pipe::ChannelMask;

use super::super::PageRenderer;
use super::super::canvas::{CanvasKind, blend, with_canvas};
use super::super::depth::CanvasSample;
use super::super::image_compositor::ImageCompositor;
use super::super::image_sampler::{BlendedRows, GridClip, MaskSampler, SampleGrid, sample_bytes};
use super::super::pixel_rect::PixelRect;
use crate::renderer::color::PROCESS_CHANNELS;
use crate::renderer::gstate::Ctm;
use crate::resources::{ImageColorSpace, SoftMask};
use crate::separation::Separator;

impl PageRenderer<'_> {
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
pub(in crate::renderer::page) fn blend_rgb<S: Sample>(dst: &mut [S], src: &[S], alpha: S) {
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
pub(in crate::renderer::page) enum ImageLenError {
    /// `width × height × bpp` overflows `usize`.
    DimensionOverflow,
    /// Buffer is shorter than the declared dimensions require.
    ShortBuffer { expected: usize },
}

/// Validates the decoder contract `actual_len >= width × height × bpp` and
/// returns the expected length on success.  Pulled out of `blit_image` so the
/// two guard branches (overflow, short buffer) are unit-testable without
/// constructing a `PageRenderer`.
pub(in crate::renderer::page) fn check_image_bytes_len(
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
