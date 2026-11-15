//! Compositing image samples onto a page or group bitmap, directly or
//! through the raster pipe.

use color::{Pixel, PixelMode, Sample};
use raster::{
    Bitmap,
    pipe::{ChannelMask, Pattern, PipeSrc, RowPattern},
    types::BlendMode,
};

use super::super::color::RasterColor;
use super::canvas::{InkPattern, TransferredInk};
use super::depth::CanvasSample;
use super::image_blit::blend_rgb;
#[cfg(feature = "cache")]
use super::image_sampler::{GridClip, SampleGrid};
use super::paint::PaintParams;
use crate::resources::transfer::Transfer;

/// Whole rows of RGBA8 pixels downloaded from a blit target: `width`
/// pixels per row, the first pixel of the first row at page position
/// `(x0, y0)`.
#[cfg(feature = "cache")]
#[derive(Clone, Copy)]
pub(super) struct RgbaRows<'a> {
    pub(super) bytes: &'a [u8],
    pub(super) x0: u32,
    pub(super) y0: u32,
    pub(super) width: u32,
}

/// Composite `rows` onto `bitmap`, restricted to the grid's columns,
/// which the rows span, and to the pixels `grid_clip` covers.  Alpha is
/// `255` where the blit kernel wrote a sample and `0` elsewhere, so the
/// composite copies each written sample, blended by the clip coverage on
/// rows the clip edge crosses.  Samples map through the RGB tables of
/// `transfer`, if any, before `compositor` paints them.
#[cfg(feature = "cache")]
pub(super) fn composite_rgba_rows<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    rows: RgbaRows<'_>,
    grid: &SampleGrid,
    grid_clip: &mut GridClip<'_>,
    compositor: ImageCompositor<'_>,
    transfer: Option<&Transfer>,
) where
    P::Sample: CanvasSample,
{
    use gpu::cache::RGBA_BPP;

    debug_assert!(
        rows.x0 <= grid.x0 && grid.x1 - rows.x0 <= rows.width,
        "blit rows do not span the grid's columns"
    );
    let src_stride = rows.width as usize * RGBA_BPP;
    let (c0, c1) = ((grid.x0 - rows.x0) as usize, (grid.x1 - rows.x0) as usize);
    let span = (grid.x1 - grid.x0) as usize;
    let mut colours = vec![0u8; span * 3];
    let mut shape = vec![0u8; span];
    for (src_row, dy) in rows.bytes.chunks_exact(src_stride.max(1)).zip(rows.y0..) {
        let Some(row_clip) = grid_clip.row(dy) else {
            continue;
        };
        let samples = &src_row[c0 * RGBA_BPP..c1 * RGBA_BPP];
        for (((dx, s), out), shape) in (grid.x0..grid.x1)
            .zip(samples.as_chunks::<RGBA_BPP>().0)
            .zip(colours.as_chunks_mut::<3>().0)
            .zip(shape.iter_mut())
        {
            let rgb = [s[0], s[1], s[2]];
            *out = transfer.map_or(rgb, |t| t.apply(rgb));
            *shape = if s[3] == 0 { 0 } else { row_clip.coverage(dx) };
        }
        compositor.paint_row(bitmap, grid.x0, dy, &colours, &shape);
    }
}

/// Composites image samples onto a page or group bitmap.
///
/// An image paints with the nonstroking constant alpha and the blend mode in
/// effect (ISO 32000-1 §11.6.4.4, §11.3.5), and inside a transparency group
/// its shape accumulates into the group's alpha plane like any other object.
/// When none of these apply (an RGB bitmap without an alpha plane, `ca` = 1,
/// `Normal`), a sample is blended onto the destination by its coverage
/// alone; otherwise every sample runs through the raster pipe that fills
/// use, the samples converted to ink on an ink bitmap.  The samples and
/// their coverage are 8-bit; on a 16-bit bitmap they are widened exactly
/// and the samples then mapped through the 16-bit transfer tables.
#[derive(Clone, Copy)]
pub(super) struct ImageCompositor<'a> {
    /// The parameters of the pipe; `None` selects the direct coverage
    /// blend.
    pub(super) paint: Option<&'a PaintParams>,
    /// A colour painted in place of the samples' colours; `None` paints the
    /// samples.  Set on an ink bitmap with a pipe, and on a 16-bit bitmap.
    pub(super) solid: Option<&'a RasterColor>,
    /// The transfer function the samples' ink maps through on an ink
    /// bitmap; on an RGB bitmap the samples are mapped beforehand.
    pub(super) transfer: Option<&'a Transfer>,
    /// The transfer function the samples map through on a 16-bit bitmap,
    /// once widened.
    pub(super) wide_transfer: Option<&'a Transfer>,
    /// The channels that carry the samples' colorants when the samples
    /// are ink, eight bytes each, for an ink bitmap; `None` when they are
    /// RGB, three bytes each.  Set only with a pipe.
    pub(super) ink_samples: Option<ChannelMask>,
}

impl<'a> ImageCompositor<'a> {
    /// The compositor of samples painted onto `bitmap` with the
    /// parameters `paint`.
    pub(super) fn new<P: Pixel>(bitmap: &Bitmap<P>, paint: &'a PaintParams) -> Self {
        let direct = matches!(P::MODE, PixelMode::Rgb8 | PixelMode::Rgb16)
            && bitmap.alpha_plane().is_none()
            && paint.alpha == u8::MAX
            && paint.blend_mode == BlendMode::Normal;
        Self {
            paint: (!direct).then_some(paint),
            solid: None,
            transfer: None,
            wide_transfer: None,
            ink_samples: None,
        }
    }

    /// Paint the samples of row `y` of `bitmap` from column `x0`: `colours`
    /// holds the RGB of each sample, or its ink under
    /// [`Self::ink_samples`], and `shape` its coverage, `0` where the
    /// sample paints nothing.  Columns past the row's end paint nothing.
    ///
    /// Through the pipe, each run of painted samples composites as one span
    /// whose shape is the coverage: the pipe takes the source alpha as
    /// `div_max(a_input × shape)`, the alpha a sample painted alone with
    /// that constant alpha would have.
    pub(super) fn paint_row<P: Pixel>(
        &self,
        bitmap: &mut Bitmap<P>,
        x0: u32,
        y: u32,
        colours: &[u8],
        shape: &[u8],
    ) where
        P::Sample: CanvasSample,
    {
        if let (Some(colours), Some(shape)) = (
            P::Sample::from_u8_slice(colours),
            P::Sample::from_u8_slice(shape),
        ) {
            self.paint_samples(bitmap, x0, y, colours, shape);
            return;
        }
        let tables = self.wide_transfer.map(P::Sample::rgb_tables);
        let wide_colours: Vec<P::Sample> = colours
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let v = P::Sample::from_u8(v);
                tables.map_or(v, |t| P::Sample::lookup(t[i % 3], v))
            })
            .collect();
        let wide_shape: Vec<P::Sample> = shape.iter().map(|&c| P::Sample::from_u8(c)).collect();
        self.paint_samples(bitmap, x0, y, &wide_colours, &wide_shape);
    }

    /// [`Self::paint_row`] of samples and coverage of the bitmap's depth.
    fn paint_samples<P: Pixel>(
        &self,
        bitmap: &mut Bitmap<P>,
        x0: u32,
        y: u32,
        colours: &[P::Sample],
        shape: &[P::Sample],
    ) where
        P::Sample: CanvasSample,
    {
        let n = P::NCOMPS;
        let sample_n = if self.ink_samples.is_some() { 8 } else { 3 };
        let zero = P::Sample::default();
        let solid = self.solid.map(|c| P::Sample::solid(c, P::MODE));
        let (row, mut alpha_row) = bitmap.row_and_alpha_mut(y);
        let x0 = x0 as usize;
        let columns = (row.len() / n).saturating_sub(x0).min(shape.len());
        let (colours, shape) = (&colours[..columns * sample_n], &shape[..columns]);
        let Some(paint) = self.paint else {
            debug_assert_ne!(P::MODE, PixelMode::DeviceN8, "the direct blend writes RGB");
            let dst = &mut row[x0 * 3..(x0 + columns) * 3];
            for ((dst, src), &coverage) in dst
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(colours.as_chunks::<3>().0)
                .zip(shape)
            {
                if coverage != zero {
                    blend_rgb(dst, solid.unwrap_or(src), coverage);
                }
            }
            return;
        };
        let pipe = paint.pipe::<P::Sample>();
        let mut start = 0;
        while let Some(offset) = shape[start..].iter().position(|&c| c != zero) {
            let a = start + offset;
            let b = shape[a..]
                .iter()
                .position(|&c| c == zero)
                .map_or(columns, |n| a + n);
            let (c0, c1) = (x0 + a, x0 + b);
            let data = &colours[a * sample_n..b * sample_n];
            let samples = RowPattern { data };
            // Ink is 8-bit: an ink bitmap's samples are bytes.
            let bytes = P::Sample::as_u8(data).map(|data| RowPattern { data });
            let ink = bytes
                .as_ref()
                .map(|row| InkPattern::new(row, self.transfer));
            let native = bytes
                .as_ref()
                .zip(self.ink_samples)
                .map(|(row, channels)| TransferredInk::new(row, self.transfer, channels));
            let src = match (solid, &native, &ink) {
                (Some(solid), _, _) => Some(PipeSrc::Solid(solid)),
                (None, Some(native), _) => P::Sample::pattern_u8(native).map(PipeSrc::Pattern),
                (None, None, Some(ink)) if P::MODE == PixelMode::DeviceN8 => {
                    P::Sample::pattern_u8(ink).map(PipeSrc::Pattern)
                }
                (None, _, _) => Some(PipeSrc::Pattern(&samples as &dyn Pattern<P::Sample>)),
            };
            if let Some(src) = src {
                #[expect(
                    clippy::cast_possible_wrap,
                    clippy::cast_possible_truncation,
                    reason = "columns and rows lie within the bitmap, whose dimensions are below 2^31"
                )]
                raster::pipe::render_span::<P>(
                    &pipe,
                    &src,
                    &mut row[c0 * n..c1 * n],
                    alpha_row.as_deref_mut().map(|al| &mut al[c0..c1]),
                    Some(&shape[a..b]),
                    c0 as i32,
                    (c1 - 1) as i32,
                    y as i32,
                );
            }
            start = b;
        }
    }
}
