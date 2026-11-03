//! Compositing image samples onto a page or group bitmap, directly or
//! through the raster pipe.

use color::{Pixel, PixelMode};
use raster::{
    Bitmap,
    pipe::{Pattern, PipeSrc, PipeState, RowPattern},
    types::BlendMode,
};

use super::canvas::InkPattern;
use super::image_blit::blend_rgb;
#[cfg(feature = "cache")]
use super::image_sampler::{GridClip, SampleGrid};
use super::paint::PaintParams;

/// Source-over whole rows of a downloaded RGBA8 page buffer, the first of
/// which is page row `y0`, onto `bitmap`, restricted to the grid's columns
/// and to the pixels `grid_clip` covers.  Alpha is `255` where the blit
/// kernel wrote a sample and `0` elsewhere, so the composite copies each
/// written sample, blended by the clip coverage on rows the clip edge
/// crosses.  Samples map through `transfer` when one is in effect.
#[cfg(feature = "cache")]
pub(super) fn composite_rgba_rows(
    bitmap: &mut Bitmap<color::Rgb8>,
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
/// use, the samples converted to ink on an ink bitmap.
#[derive(Clone, Copy)]
pub(super) struct ImageCompositor<'a> {
    /// `None` selects the direct coverage blend.
    pub(super) pipe: Option<PipeState<'a>>,
    /// A colour, in the destination's pixel bytes, painted in place of the
    /// samples' colours; `None` paints the samples.  Set only with a pipe.
    pub(super) solid: Option<&'a [u8]>,
}

impl<'a> ImageCompositor<'a> {
    /// The compositor of samples painted onto `bitmap` with the
    /// parameters `paint`.
    pub(super) fn new<P: Pixel>(bitmap: &Bitmap<P>, paint: &'a PaintParams) -> Self {
        let direct = P::MODE == PixelMode::Rgb8
            && bitmap.alpha_plane().is_none()
            && paint.alpha == u8::MAX
            && paint.blend_mode == BlendMode::Normal;
        Self {
            pipe: (!direct).then(|| paint.pipe()),
            solid: None,
        }
    }

    /// Paint the samples of row `y` of `bitmap` from column `x0`: `colours`
    /// holds the RGB of each sample and `shape` its coverage, `0` where the
    /// sample paints nothing.  Columns past the row's end paint nothing.
    ///
    /// Through the pipe, each run of painted samples composites as one span
    /// whose shape is the coverage: the pipe takes the source alpha as
    /// `div255(a_input × shape)`, the alpha a sample painted alone with
    /// that constant alpha would have.
    pub(super) fn paint_row<P: Pixel>(
        &self,
        bitmap: &mut Bitmap<P>,
        x0: u32,
        y: u32,
        colours: &[u8],
        shape: &[u8],
    ) {
        let bpp = P::BYTES;
        let (row, mut alpha_row) = bitmap.row_and_alpha_mut(y);
        let x0 = x0 as usize;
        let columns = (row.len() / bpp).saturating_sub(x0).min(shape.len());
        let (colours, shape) = (&colours[..columns * 3], &shape[..columns]);
        let Some(pipe) = &self.pipe else {
            debug_assert_eq!(P::MODE, PixelMode::Rgb8, "the direct blend writes RGB");
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
            let samples = RowPattern {
                data: &colours[a * 3..b * 3],
            };
            let ink = InkPattern::new(&samples, None);
            let src = match self.solid {
                Some(solid) => PipeSrc::Solid(solid),
                None if P::MODE == PixelMode::DeviceN8 => PipeSrc::Pattern(&ink),
                None => PipeSrc::Pattern(&samples as &dyn Pattern),
            };
            #[expect(
                clippy::cast_possible_wrap,
                clippy::cast_possible_truncation,
                reason = "columns and rows lie within the bitmap, whose dimensions are below 2^31"
            )]
            raster::pipe::render_span::<P>(
                pipe,
                &src,
                &mut row[c0 * bpp..c1 * bpp],
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
