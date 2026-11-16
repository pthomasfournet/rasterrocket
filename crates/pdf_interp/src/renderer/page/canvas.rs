//! The page canvas: the bitmap a page is painted on, of RGB or ink pixels.
//!
//! An RGB canvas holds sRGB samples of 8 or 16 bits, full intensity being
//! paper white.  An ink canvas holds [`DeviceN8`] or [`DeviceN16`] pixels,
//! the amounts `[C, M, Y, K, S0, S1, S2, S3]` of four process and four
//! spot inks, `0` being no ink, so that paper white is all zeros; it is
//! converted to RGB or CMYK of its depth when the page is finished (see
//! [`super::finished`]).
//!
//! The raster primitives are generic over the pixel type.  [`with_canvas!`]
//! dispatches one call to the bitmap of any kind, so a paint is written
//! once; its pipe and source are taken at the canvas's depth (see
//! [`CanvasPipe`], [`CanvasPattern::at`]).  The sources the renderer builds
//! in sRGB — shadings, tiling cells, image samples, mesh triangles — reach
//! an ink canvas through [`InkPattern`], which converts each span to ink.

use color::{DeviceN8, DeviceN16, Pixel, PixelMode, Rgb8, Rgb16, Sample, SampleDepth};
use raster::clip::Clip;
use raster::pipe::{ChannelMask, Pattern, PipeSrc, PipeState, RowPattern, render_bitmap_span};
use raster::shading::gouraud::{GouraudVertex, gouraud_triangle_fill, gouraud_triangle_spans};
use raster::transparency::{GroupBitmap, GroupParams, begin_group, paint_group};
use raster::{Bitmap, BitmapBuffers};

use super::super::color::{PROCESS_CHANNELS, no_spots};
use super::depth::{CanvasBackdrop, CanvasSample, SampleVec, SourcePattern, Widened};
use super::paint::PaintParams;
use super::pixel_rect::PixelRect;
use super::{CanvasBuffers, PageRenderer};
use crate::resources::transfer::Transfer;
use crate::separation::Separator;

/// Run `$body` with `$b` bound to the bitmap of `$canvas`, of any pixel
/// type: `with_canvas!(&mut self.canvas, |b| fill(b, ..))` binds `b` as
/// `&mut Bitmap<Rgb8>`, `&mut Bitmap<Rgb16>`, `&mut Bitmap<DeviceN8>` and
/// `&mut Bitmap<DeviceN16>` in turn, so `$body` is written once over the
/// pixel type.
macro_rules! with_canvas {
    ($canvas:expr, |$b:ident| $body:expr) => {
        match $canvas {
            $crate::renderer::page::canvas::PageCanvas::Rgb($b) => $body,
            $crate::renderer::page::canvas::PageCanvas::Rgb16($b) => $body,
            $crate::renderer::page::canvas::PageCanvas::Ink($b) => $body,
            $crate::renderer::page::canvas::PageCanvas::Ink16($b) => $body,
        }
    };
}
pub(super) use with_canvas;

/// The pixel type of a [`PageCanvas`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum CanvasKind {
    /// sRGB, three bytes per pixel.
    Rgb,
    /// sRGB, three 16-bit samples per pixel.
    Rgb16,
    /// Process and spot ink amounts, eight bytes per pixel.
    Ink,
    /// Process and spot ink amounts, eight 16-bit samples per pixel.
    Ink16,
}

impl CanvasKind {
    /// The pixel mode of the canvas.
    pub(super) const fn mode(self) -> PixelMode {
        match self {
            Self::Rgb => PixelMode::Rgb8,
            Self::Rgb16 => PixelMode::Rgb16,
            Self::Ink => PixelMode::DeviceN8,
            Self::Ink16 => PixelMode::DeviceN16,
        }
    }

    /// The sRGB canvas of samples of `depth`.
    pub(super) const fn rgb(depth: SampleDepth) -> Self {
        match depth {
            SampleDepth::Eight => Self::Rgb,
            SampleDepth::Sixteen => Self::Rgb16,
        }
    }

    /// The ink canvas of samples of `depth`.
    pub(super) const fn ink(depth: SampleDepth) -> Self {
        match depth {
            SampleDepth::Eight => Self::Ink,
            SampleDepth::Sixteen => Self::Ink16,
        }
    }

    /// Whether the canvas holds ink.
    pub(super) const fn is_ink(self) -> bool {
        matches!(self, Self::Ink | Self::Ink16)
    }

    /// The depth of the canvas's samples.
    pub(super) const fn depth(self) -> SampleDepth {
        match self {
            Self::Rgb | Self::Ink => SampleDepth::Eight,
            Self::Rgb16 | Self::Ink16 => SampleDepth::Sixteen,
        }
    }
}

/// The bitmap a page, a transparency group or a soft mask is painted on.
pub(super) enum PageCanvas {
    /// sRGB pixels.
    Rgb(Bitmap<Rgb8>),
    /// sRGB pixels of 16-bit samples.
    Rgb16(Bitmap<Rgb16>),
    /// Ink pixels.
    Ink(Bitmap<DeviceN8>),
    /// Ink pixels of 16-bit samples.
    Ink16(Bitmap<DeviceN16>),
}

/// The value of every channel of paper white in pixels of type `P`.
fn paper<P: Pixel>() -> P::Sample {
    if matches!(P::MODE, PixelMode::DeviceN8 | PixelMode::DeviceN16) {
        P::Sample::default()
    } else {
        P::Sample::MAX
    }
}

/// The sample every channel of a fresh bitmap of pixels `P` holds: paper
/// white for an opaque bitmap, 0 (transparent) otherwise.
fn initial_fill<P: Pixel>(alpha: bool) -> P::Sample {
    if alpha {
        P::Sample::default()
    } else {
        paper::<P>()
    }
}

/// Fill one Gouraud-shaded triangle `tri` of vertices of `N` samples of
/// type `V` onto `b` through `pipe`, clipped to `clip`: each span of the
/// interpolated vertex colours is handed to `span`, which runs its second
/// argument on the span as the canvas takes it.
fn gouraud_spans<P: Pixel, const N: usize, V: Sample>(
    b: &mut Bitmap<P>,
    origin: (i32, i32),
    clip: &Clip,
    pipe: &CanvasPipe<'_>,
    tri: [GouraudVertex<N, V>; 3],
    span: impl Fn(&dyn Pattern<V>, &mut dyn FnMut(&dyn Pattern<P::Sample>)),
) where
    P::Sample: CanvasSample,
{
    let (width, height) = (b.width, b.height);
    let pipe = pipe.at();
    gouraud_triangle_spans(width, height, origin, clip, tri, |data, x0, x1, y| {
        span(&RowPattern { data }, &mut |source| {
            render_bitmap_span::<P>(b, &pipe, &PipeSrc::Pattern(source), None, x0, x1, y);
        });
    });
}

impl PageCanvas {
    /// [`Self::try_new`] for a canvas the allocator grants.
    ///
    /// # Panics
    ///
    /// When the allocator refuses the canvas.
    #[cfg(test)]
    pub(super) fn new(kind: CanvasKind, width: u32, height: u32, alpha: bool) -> Self {
        Self::try_new(kind, width, height, alpha).expect("a test canvas is allocated")
    }

    /// A `width × height` canvas of `kind`: transparent, with an alpha
    /// plane, when `alpha`; opaque paper white otherwise; or the error for
    /// buffers the allocator refuses.
    pub(super) fn try_new(
        kind: CanvasKind,
        width: u32,
        height: u32,
        alpha: bool,
    ) -> Result<Self, raster::BitmapAllocError> {
        Self::try_new_in(kind, width, height, alpha, &mut CanvasBuffers::default())
    }

    /// [`Self::try_new`] built in the buffers of `buffers` for `kind`
    /// ([`Bitmap::try_new_filled_in`]), which are taken from it.
    pub(super) fn try_new_in(
        kind: CanvasKind,
        width: u32,
        height: u32,
        alpha: bool,
        buffers: &mut CanvasBuffers,
    ) -> Result<Self, raster::BitmapAllocError> {
        fn bitmap<P: Pixel>(
            held: &mut BitmapBuffers<P::Sample>,
            width: u32,
            height: u32,
            alpha: bool,
        ) -> Result<Bitmap<P>, raster::BitmapAllocError> {
            let fill = initial_fill::<P>(alpha);
            Bitmap::try_new_filled_in(held.take_for(alpha), width, height, 1, alpha, fill)
        }
        Ok(match kind {
            CanvasKind::Rgb => Self::Rgb(bitmap(&mut buffers.rgb, width, height, alpha)?),
            CanvasKind::Rgb16 => Self::Rgb16(bitmap(&mut buffers.rgb16, width, height, alpha)?),
            CanvasKind::Ink => Self::Ink(bitmap(&mut buffers.ink, width, height, alpha)?),
            CanvasKind::Ink16 => Self::Ink16(bitmap(&mut buffers.ink16, width, height, alpha)?),
        })
    }

    /// A `width × height` opaque canvas of `kind` whose samples are
    /// unspecified ([`Bitmap::try_new_unpainted_in`]), built in the buffers
    /// of `buffers` for `kind`; [`Self::paint_paper`] makes it the canvas
    /// [`Self::try_new_in`] builds.
    pub(super) fn try_new_unpainted_in(
        kind: CanvasKind,
        width: u32,
        height: u32,
        buffers: &mut CanvasBuffers,
    ) -> Result<Self, raster::BitmapAllocError> {
        fn bitmap<P: Pixel>(
            held: &mut BitmapBuffers<P::Sample>,
            width: u32,
            height: u32,
        ) -> Result<Bitmap<P>, raster::BitmapAllocError> {
            Bitmap::try_new_unpainted_in(held.take_for(false), width, height, 1)
        }
        Ok(match kind {
            CanvasKind::Rgb => Self::Rgb(bitmap(&mut buffers.rgb, width, height)?),
            CanvasKind::Rgb16 => Self::Rgb16(bitmap(&mut buffers.rgb16, width, height)?),
            CanvasKind::Ink => Self::Ink(bitmap(&mut buffers.ink, width, height)?),
            CanvasKind::Ink16 => Self::Ink16(bitmap(&mut buffers.ink16, width, height)?),
        })
    }

    /// Paint every sample of the opaque canvas paper white.
    pub(super) fn paint_paper(&mut self) {
        fn fill<P: Pixel>(bitmap: &mut Bitmap<P>) {
            bitmap.data_mut().fill(paper::<P>());
        }
        with_canvas!(self, |b| fill(b));
    }

    /// The pixel type of the canvas.
    pub(super) const fn kind(&self) -> CanvasKind {
        match self {
            Self::Rgb(_) => CanvasKind::Rgb,
            Self::Rgb16(_) => CanvasKind::Rgb16,
            Self::Ink(_) => CanvasKind::Ink,
            Self::Ink16(_) => CanvasKind::Ink16,
        }
    }

    /// `(width, height)` in pixels.
    pub(super) const fn dims(&self) -> (u32, u32) {
        with_canvas!(self, |b| (b.width, b.height))
    }

    /// Whether the canvas has an alpha plane.
    pub(super) const fn has_alpha(&self) -> bool {
        with_canvas!(self, |b| b.has_alpha())
    }

    /// Composite the canvas, a page group with an alpha plane, onto opaque
    /// paper in place (see [`composite_onto_white`]), the alpha plane
    /// removed into the buffers of `buffers` for its kind.
    pub(super) fn composite_onto_white(&mut self, buffers: &mut CanvasBuffers) {
        match self {
            Self::Rgb(b) => {
                if let Some(alpha) = composite_onto_white(b) {
                    buffers.rgb.alpha = alpha;
                }
            }
            Self::Rgb16(b) => {
                if let Some(alpha) = composite_onto_white(b) {
                    buffers.rgb16.alpha = alpha;
                }
            }
            Self::Ink(b) => {
                if let Some(alpha) = composite_onto_white(b) {
                    buffers.ink.alpha = alpha;
                }
            }
            Self::Ink16(b) => {
                if let Some(alpha) = composite_onto_white(b) {
                    buffers.ink16.alpha = alpha;
                }
            }
        }
    }

    /// Make the pixels of `region` transparent.
    pub(super) fn clear(&mut self, region: PixelRect) {
        with_canvas!(self, |b| clear(b, region));
    }

    /// Fill one Gouraud-shaded triangle of 8-bit sRGB vertices through
    /// `pipe`, clipped to `clip`.  The interpolated sRGB spans reach an ink
    /// canvas converted to ink by `separator` and mapped through
    /// `transfer`, and a 16-bit canvas widened; on an RGB canvas `pipe`
    /// carries the transfer function.
    pub(super) fn gouraud_triangle(
        &mut self,
        origin: (i32, i32),
        clip: &Clip,
        pipe: &CanvasPipe<'_>,
        (transfer, separator): (Option<&Transfer>, &Separator),
        tri: [GouraudVertex; 3],
    ) {
        match self {
            Self::Rgb(b) => gouraud_triangle_fill(b, origin, clip, &pipe.at(), tri),
            Self::Rgb16(b) => gouraud_spans(b, origin, clip, pipe, tri, |row, paint| {
                paint(SourcePattern::<u16>::of_u8(row).as_dyn());
            }),
            Self::Ink(b) => gouraud_spans(b, origin, clip, pipe, tri, |row, paint| {
                paint(&InkPattern::new(row, transfer, separator));
            }),
            Self::Ink16(b) => gouraud_spans(b, origin, clip, pipe, tri, |row, paint| {
                paint(&Widened::new(&InkPattern::new(row, transfer, separator)));
            }),
        }
    }

    /// Fill one Gouraud-shaded triangle of 16-bit sRGB vertices through
    /// `pipe`, clipped to `clip`; on an ink canvas the spans convert to ink
    /// at 16 bits by `separator` and map through `transfer`.  Nothing is
    /// painted on an 8-bit canvas, whose shadings are resolved at 8 bits.
    pub(super) fn gouraud_triangle16(
        &mut self,
        origin: (i32, i32),
        clip: &Clip,
        pipe: &CanvasPipe<'_>,
        (transfer, separator): (Option<&Transfer>, &Separator),
        tri: [GouraudVertex<3, u16>; 3],
    ) {
        match self {
            Self::Rgb16(b) => gouraud_triangle_fill(b, origin, clip, &pipe.at(), tri),
            Self::Ink16(b) => gouraud_spans(b, origin, clip, pipe, tri, |row, paint| {
                paint(&InkPattern::new(row, transfer, separator));
            }),
            Self::Rgb(_) | Self::Ink(_) => {
                debug_assert!(false, "a 16-bit triangle on an 8-bit canvas");
            }
        }
    }

    /// Fill one Gouraud-shaded triangle of ink vertices through `pipe`,
    /// clipped to `clip`, the ink of the channels `channels` mapped through
    /// `transfer`.  Nothing is painted on an RGB canvas, which takes no
    /// ink.
    pub(super) fn gouraud_ink_triangle(
        &mut self,
        origin: (i32, i32),
        clip: &Clip,
        pipe: &CanvasPipe<'_>,
        transfer: Option<&Transfer>,
        channels: ChannelMask,
        tri: [GouraudVertex<8>; 3],
    ) {
        match self {
            Self::Ink(b) => gouraud_spans(b, origin, clip, pipe, tri, |row, paint| {
                paint(&TransferredInk::new(row, transfer, channels));
            }),
            Self::Ink16(b) => gouraud_spans(b, origin, clip, pipe, tri, |row, paint| {
                paint(&TransferredInk::new(&Widened::new(row), transfer, channels));
            }),
            Self::Rgb(_) | Self::Rgb16(_) => {
                debug_assert!(false, "an ink triangle on an RGB canvas");
            }
        }
    }

    /// Fill one Gouraud-shaded triangle of 16-bit ink vertices through
    /// `pipe`, clipped to `clip`, the ink of the channels `channels` mapped
    /// through `transfer`.  Nothing is painted on a canvas other than a
    /// 16-bit ink canvas, which alone is given them.
    pub(super) fn gouraud_ink_triangle16(
        &mut self,
        origin: (i32, i32),
        clip: &Clip,
        pipe: &CanvasPipe<'_>,
        transfer: Option<&Transfer>,
        channels: ChannelMask,
        tri: [GouraudVertex<8, u16>; 3],
    ) {
        let Self::Ink16(b) = self else {
            debug_assert!(false, "a 16-bit ink triangle on another canvas");
            return;
        };
        gouraud_spans(b, origin, clip, pipe, tri, |row, paint| {
            paint(&TransferredInk::new(row, transfer, channels));
        });
    }

    /// Open a transparency group over this canvas on `bitmap`, a
    /// transparent canvas of the same kind and size (see
    /// [`raster::transparency::begin_group`]).
    ///
    /// # Panics
    ///
    /// Panics if `bitmap` is of another kind.
    pub(super) fn begin_group(
        &self,
        backdrop: Option<&CanvasBackdrop>,
        clip: &Clip,
        params: GroupParams,
        bitmap: Self,
    ) -> CanvasGroup {
        match (self, bitmap) {
            (Self::Rgb(p), Self::Rgb(b)) => CanvasGroup::Rgb(begin_group(
                p,
                backdrop.and_then(u8::backdrop),
                clip,
                params,
                b,
            )),
            (Self::Rgb16(p), Self::Rgb16(b)) => CanvasGroup::Rgb16(begin_group(
                p,
                backdrop.and_then(u16::backdrop),
                clip,
                params,
                b,
            )),
            (Self::Ink(p), Self::Ink(b)) => CanvasGroup::Ink(begin_group(
                p,
                backdrop.and_then(u8::backdrop),
                clip,
                params,
                b,
            )),
            (Self::Ink16(p), Self::Ink16(b)) => CanvasGroup::Ink16(begin_group(
                p,
                backdrop.and_then(u16::backdrop),
                clip,
                params,
                b,
            )),
            _ => panic!("begin_group: the group canvas differs in kind from its parent"),
        }
    }

    /// Exchange this canvas's bitmap with the bitmap of `group`.
    ///
    /// # Panics
    ///
    /// Panics if `group` is of another kind.
    pub(super) fn swap_group(&mut self, group: &mut CanvasGroup) {
        match (self, group) {
            (Self::Rgb(c), CanvasGroup::Rgb(g)) => std::mem::swap(c, &mut g.bitmap),
            (Self::Rgb16(c), CanvasGroup::Rgb16(g)) => std::mem::swap(c, &mut g.bitmap),
            (Self::Ink(c), CanvasGroup::Ink(g)) => std::mem::swap(c, &mut g.bitmap),
            (Self::Ink16(c), CanvasGroup::Ink16(g)) => std::mem::swap(c, &mut g.bitmap),
            _ => panic!("swap_group: the group differs in kind from the canvas"),
        }
    }

    /// Composite the finished `group` onto this canvas through `pipe` and
    /// return its bitmap, cleared (see [`raster::transparency::paint_group`]).
    /// The clip saved when the group was opened is dropped: the caller
    /// restores its own graphics state.
    ///
    /// # Panics
    ///
    /// Panics if `group` is of another kind.
    pub(super) fn paint_group(&mut self, group: CanvasGroup, pipe: &CanvasPipe<'_>) -> Self {
        match (self, group) {
            (Self::Rgb(c), CanvasGroup::Rgb(g)) => Self::Rgb(paint_group(c, g, &pipe.at()).1),
            (Self::Rgb16(c), CanvasGroup::Rgb16(g)) => Self::Rgb16(paint_group(c, g, &pipe.at()).1),
            (Self::Ink(c), CanvasGroup::Ink(g)) => Self::Ink(paint_group(c, g, &pipe.at()).1),
            (Self::Ink16(c), CanvasGroup::Ink16(g)) => Self::Ink16(paint_group(c, g, &pipe.at()).1),
            _ => panic!("paint_group: the group differs in kind from the canvas"),
        }
    }
}

/// An open transparency group of any pixel type.
pub(super) enum CanvasGroup {
    /// A group over an RGB canvas.
    Rgb(GroupBitmap<Rgb8>),
    /// A group over a 16-bit RGB canvas.
    Rgb16(GroupBitmap<Rgb16>),
    /// A group over an ink canvas.
    Ink(GroupBitmap<DeviceN8>),
    /// A group over a 16-bit ink canvas.
    Ink16(GroupBitmap<DeviceN16>),
}

impl CanvasGroup {
    /// The group region as half-open pixel ranges `(x0, y0, x1, y1)`.
    pub(super) const fn region(&self) -> (u32, u32, u32, u32) {
        match self {
            Self::Rgb(g) => g.region(),
            Self::Rgb16(g) => g.region(),
            Self::Ink(g) => g.region(),
            Self::Ink16(g) => g.region(),
        }
    }

    /// Take the backdrop of a non-isolated group.
    pub(super) fn take_backdrop(&mut self) -> Option<CanvasBackdrop> {
        match self {
            Self::Rgb(g) => g.backdrop.take().map(u8::into_backdrop),
            Self::Rgb16(g) => g.backdrop.take().map(u16::into_backdrop),
            Self::Ink(g) => g.backdrop.take().map(u8::into_backdrop),
            Self::Ink16(g) => g.backdrop.take().map(u16::into_backdrop),
        }
    }
}

/// The compositing state of a paint as a canvas of any depth takes it: the
/// paint's parameters, and the transfer function the pipe applies to a
/// source that is not mapped beforehand (see [`CanvasPattern::pipe_transfer`]).
#[derive(Clone, Copy)]
pub(super) struct CanvasPipe<'a> {
    /// The alpha, blend mode, channels and backdrop of the paint.
    pub(super) paint: &'a PaintParams,
    /// The transfer function the pipe applies, if any.
    pub(super) transfer: Option<&'a Transfer>,
}

impl<'a> CanvasPipe<'a> {
    /// The pipe of `paint`, with the identity transfer.
    pub(super) const fn of(paint: &'a PaintParams) -> Self {
        Self {
            paint,
            transfer: None,
        }
    }

    /// The pipe for samples of type `S`.
    pub(super) fn at<S: CanvasSample>(&self) -> PipeState<'a, S> {
        self.paint.pipe_with(self.transfer)
    }
}

/// Composite `bitmap`, a page group with an alpha plane, onto opaque paper
/// in place: each channel becomes `c·α + w·(1 − α)`, `w` being paper white,
/// and the alpha plane is removed and returned.  A bitmap without an alpha
/// plane is left as it is, and the result is `None`.
pub(super) fn composite_onto_white<P: Pixel>(bitmap: &mut Bitmap<P>) -> Option<Vec<P::Sample>> {
    let white = paper::<P>();
    let n = bitmap.width as usize * P::NCOMPS;
    for y in 0..bitmap.height {
        let (row, Some(alpha_row)) = bitmap.row_and_alpha_mut(y) else {
            return None;
        };
        for (px, &a) in row[..n].chunks_exact_mut(P::NCOMPS).zip(&*alpha_row) {
            if a == P::Sample::MAX {
                continue;
            }
            for c in px {
                *c = blend(*c, white, a);
            }
        }
    }
    bitmap.take_alpha_plane()
}

/// Set the colour of every pixel of `bitmap` whose alpha is 0 to 0.  A
/// bitmap without an alpha plane is left as it is.
pub(super) fn zero_transparent_colour<P: Pixel>(bitmap: &mut Bitmap<P>) {
    let n = bitmap.width as usize * P::NCOMPS;
    for y in 0..bitmap.height {
        let (row, Some(alpha_row)) = bitmap.row_and_alpha_mut(y) else {
            return;
        };
        for (px, &a) in row[..n].chunks_exact_mut(P::NCOMPS).zip(&*alpha_row) {
            if a == P::Sample::default() {
                px.fill(P::Sample::default());
            }
        }
    }
}

/// Porter-Duff source-over blend for a single channel:
/// `(src × alpha + dst × (MAX − alpha)) / MAX`, the quotient truncated.
///
/// The two weights sum to `MAX`, so the sum of products is at most `MAX²`
/// and the quotient at most `MAX`.
pub(super) fn blend<S: Sample>(src: S, dst: S, alpha: S) -> S {
    let ia = S::MAX.wide() - alpha.wide();
    S::from_wide((src.wide() * alpha.wide() + dst.wide() * ia) / S::max_wide())
}

/// Make the pixels of `region` transparent: colour and alpha zero.
pub(super) fn clear<P: Pixel>(bitmap: &mut Bitmap<P>, region: PixelRect) {
    let PixelRect { x0, y0, x1, y1 } = region;
    let n = P::NCOMPS;
    for y in y0..y1 {
        let (row, alpha) = bitmap.row_and_alpha_mut(y);
        row[x0 as usize * n..x1 as usize * n].fill(P::Sample::default());
        if let Some(alpha) = alpha {
            alpha[x0 as usize..x1 as usize].fill(P::Sample::default());
        }
    }
}

/// The pixels of a rectangle of a canvas, captured before an object is
/// painted.  Restored, blended back or merged onto the canvas it was taken
/// from.
pub(super) struct Snapshot {
    pub(super) x0: u32,
    pub(super) y0: u32,
    pub(super) x1: u32,
    /// Samples per pixel of the canvas.
    pub(super) ncomps: usize,
    /// Colour samples, `ncomps` per pixel, row-major over the rectangle.
    pub(super) color: SampleVec,
    /// Alpha per pixel, or `None` for a bitmap without an alpha plane.
    pub(super) alpha: Option<SampleVec>,
}

impl Snapshot {
    /// Capture `region` of `bitmap`, or the error for a buffer of the
    /// region's size when the allocator refuses one.
    pub(super) fn try_take<P: Pixel>(
        bitmap: &Bitmap<P>,
        region: PixelRect,
    ) -> Result<Self, raster::BitmapAllocError>
    where
        P::Sample: CanvasSample,
    {
        let PixelRect { x0, y0, x1, y1 } = region;
        let refused = |()| raster::BitmapAllocError {
            width: region.width(),
            height: region.height(),
        };
        let ncomps = P::NCOMPS;
        let (cx0, cx1) = (x0 as usize * ncomps, x1 as usize * ncomps);
        let (ax0, ax1) = (x0 as usize, x1 as usize);
        let rows = (y1 - y0) as usize;
        let buffer = |samples: Option<usize>| -> Result<Vec<P::Sample>, ()> {
            let mut v = Vec::new();
            v.try_reserve_exact(samples.ok_or(())?).map_err(|_| ())?;
            Ok(v)
        };
        let mut color = buffer((cx1 - cx0).checked_mul(rows)).map_err(refused)?;
        let mut alpha = if bitmap.has_alpha() {
            Some(buffer((ax1 - ax0).checked_mul(rows)).map_err(refused)?)
        } else {
            None
        };
        for y in y0..y1 {
            color.extend_from_slice(&bitmap.row_samples(y)[cx0..cx1]);
            if let (Some(a), Some(row)) = (alpha.as_mut(), bitmap.alpha_row(y)) {
                a.extend_from_slice(&row[ax0..ax1]);
            }
        }
        Ok(Self {
            x0,
            y0,
            x1,
            ncomps,
            color: P::Sample::into_samples(color),
            alpha: alpha.map(P::Sample::into_samples),
        })
    }

    /// Pixels per row of the rectangle.
    pub(super) const fn span(&self) -> usize {
        (self.x1 - self.x0) as usize
    }

    /// The colour and alpha samples, of the type of a canvas the snapshot
    /// was taken from; `None` for a canvas of another depth.
    pub(super) fn samples<S: CanvasSample>(&self) -> Option<(&[S], Option<&[S]>)> {
        let color = S::samples(&self.color)?;
        let alpha = match &self.alpha {
            Some(a) => Some(S::samples(a)?),
            None => None,
        };
        Some((color, alpha))
    }

    /// Write the snapshot back over its rectangle of `bitmap`.
    fn restore<P: Pixel>(self, bitmap: &mut Bitmap<P>)
    where
        P::Sample: CanvasSample,
    {
        debug_assert_eq!(
            self.ncomps,
            P::NCOMPS,
            "a snapshot restores onto its own canvas"
        );
        let Some((color, alpha)) = self.samples::<P::Sample>() else {
            debug_assert!(false, "a snapshot restores onto a canvas of its depth");
            return;
        };
        let (span, n) = (self.span(), self.ncomps);
        let (cx0, ax0) = (self.x0 as usize * n, self.x0 as usize);
        for (i, prior) in color.chunks_exact(span * n).enumerate() {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "i counts rows of the snapshot, which lie below the bitmap's u32 height"
            )]
            let y = self.y0 + i as u32;
            let (row, alpha_row) = bitmap.row_and_alpha_mut(y);
            row[cx0..cx0 + span * n].copy_from_slice(prior);
            if let (Some(prior_a), Some(alpha_row)) = (alpha, alpha_row) {
                alpha_row[ax0..ax0 + span].copy_from_slice(&prior_a[i * span..(i + 1) * span]);
            }
        }
    }
}

impl PageRenderer<'_> {
    /// Capture `region` of the canvas, or `None`, with the render stopped
    /// by `RenderAbort::AllocationFailed`, when the allocator refuses the
    /// snapshot's buffers.
    pub(super) fn take_region(&mut self, region: PixelRect) -> Option<Snapshot> {
        #[cfg(test)]
        if region.area() > self.max_snapshot_px {
            self.abort_allocation(raster::BitmapAllocError {
                width: region.width(),
                height: region.height(),
            });
            return None;
        }
        with_canvas!(&self.canvas, |b| Snapshot::try_take(b, region))
            .map_err(|e| self.abort_allocation(e))
            .ok()
    }

    /// Write `snapshot`, taken from the canvas, back over its rectangle.
    pub(super) fn restore_region(&mut self, snapshot: Snapshot) {
        with_canvas!(&mut self.canvas, |b| snapshot.restore(b));
    }

    /// The canvas as an RGB bitmap.
    ///
    /// # Panics
    ///
    /// Panics on a canvas of another kind.
    #[cfg(test)]
    pub(super) fn rgb_canvas(&self) -> &Bitmap<Rgb8> {
        match &self.canvas {
            PageCanvas::Rgb(b) => b,
            PageCanvas::Rgb16(_) | PageCanvas::Ink(_) | PageCanvas::Ink16(_) => {
                panic!("rgb_canvas: the canvas is not 8-bit RGB")
            }
        }
    }
}

/// A [`Pattern`] of sRGB samples of type `S` served as ink of that type:
/// each pixel the inner pattern yields becomes the ink a [`Separator`]
/// gives it, with no spot ink, its process ink mapped through the ink
/// tables of the transfer function, if any.  The pattern only converts;
/// coverage and the static-colour hint are the inner pattern's.
pub(super) struct InkPattern<'a, S: Sample = u8> {
    pattern: &'a dyn Pattern<S>,
    transfer: Option<&'a Transfer>,
    separator: &'a Separator,
}

impl<'a, S: CanvasSample> InkPattern<'a, S> {
    /// `pattern` served as ink by `separator`, its colours mapped through
    /// `transfer`.
    pub(super) const fn new(
        pattern: &'a dyn Pattern<S>,
        transfer: Option<&'a Transfer>,
        separator: &'a Separator,
    ) -> Self {
        Self {
            pattern,
            transfer,
            separator,
        }
    }

    /// Fill `out` with ink by running `fill` on an sRGB span of the same
    /// pixel count, drawn from [`CanvasSample::with_rgb_span`]: the inner
    /// pattern yields sRGB, never ink, so an `InkPattern` never wraps
    /// another and the span is never borrowed twice.
    fn convert(&self, out: &mut [S], fill: impl FnOnce(&mut [S])) {
        S::with_rgb_span(out.len() / 8 * 3, |rgb| {
            fill(rgb);
            let mut last: Option<([S; 3], [S; 8])> = None;
            for (src, dst) in rgb
                .as_chunks::<3>()
                .0
                .iter()
                .zip(out.as_chunks_mut::<8>().0)
            {
                *dst = match last {
                    Some((seen, ink)) if seen == *src => ink,
                    _ => {
                        let ink = no_spots(self.separator.ink(*src));
                        let ink = self
                            .transfer
                            .map_or(ink, |t| S::map_ink(t, ink, PROCESS_CHANNELS));
                        last = Some((*src, ink));
                        ink
                    }
                };
            }
        });
    }
}

impl<S: CanvasSample> Pattern<S> for InkPattern<'_, S> {
    fn fill_span(&self, y: i32, x0: i32, x1: i32, out: &mut [S]) {
        self.convert(out, |rgb| self.pattern.fill_span(y, x0, x1, rgb));
    }

    fn is_static_color(&self) -> bool {
        self.pattern.is_static_color()
    }

    fn may_leave_unpainted(&self) -> bool {
        self.pattern.may_leave_unpainted()
    }

    fn fill_span_coverage(&self, y: i32, x0: i32, x1: i32, out: &mut [S], coverage: &mut [S]) {
        self.convert(out, |rgb| {
            self.pattern.fill_span_coverage(y, x0, x1, rgb, coverage);
        });
    }
}

/// A [`Pattern`] of ink whose pixels map, in the channels that carry the
/// pattern's colorants, through the ink tables of a transfer function, if
/// any.  Coverage and the static-colour hint are the inner pattern's.
pub(super) struct TransferredInk<'a, S: Sample = u8> {
    pattern: &'a dyn Pattern<S>,
    transfer: Option<&'a Transfer>,
    channels: ChannelMask,
}

impl<'a, S: CanvasSample> TransferredInk<'a, S> {
    /// `pattern`, of ink in the channels `channels`, mapped through
    /// `transfer`.
    pub(super) const fn new(
        pattern: &'a dyn Pattern<S>,
        transfer: Option<&'a Transfer>,
        channels: ChannelMask,
    ) -> Self {
        Self {
            pattern,
            transfer,
            channels,
        }
    }

    /// Map the ink pixels of `out` through the transfer function.
    fn map(&self, out: &mut [S]) {
        if let Some(t) = self.transfer {
            for px in out.as_chunks_mut::<8>().0 {
                *px = S::map_ink(t, *px, self.channels);
            }
        }
    }
}

impl<S: CanvasSample> Pattern<S> for TransferredInk<'_, S> {
    fn fill_span(&self, y: i32, x0: i32, x1: i32, out: &mut [S]) {
        self.pattern.fill_span(y, x0, x1, out);
        self.map(out);
    }

    fn is_static_color(&self) -> bool {
        self.pattern.is_static_color()
    }

    fn may_leave_unpainted(&self) -> bool {
        self.pattern.may_leave_unpainted()
    }

    fn fill_span_coverage(&self, y: i32, x0: i32, x1: i32, out: &mut [S], coverage: &mut [S]) {
        self.pattern.fill_span_coverage(y, x0, x1, out, coverage);
        self.map(out);
    }
}

/// A pattern as a canvas of a given kind takes it.  A pattern of sRGB is
/// taken as it is on an RGB canvas, where the pipe applies the transfer
/// function, widened on a 16-bit canvas, and through [`InkPattern`], which
/// applies it to the ink, on an ink canvas; a pattern of ink is taken on an
/// ink canvas through [`TransferredInk`].  An 8-bit pattern reaches a
/// 16-bit ink canvas as its 8-bit ink widened.
pub(super) enum CanvasPattern<'a> {
    /// A pattern of 8-bit sRGB.
    Rgb(&'a dyn Pattern),
    /// A pattern of 16-bit sRGB, for a 16-bit RGB canvas.
    Rgb16(&'a dyn Pattern<u16>),
    /// The pattern converted to ink.
    Ink(InkPattern<'a>),
    /// A pattern of 16-bit sRGB converted to 16-bit ink, for a 16-bit ink
    /// canvas.
    Ink16(InkPattern<'a, u16>),
    /// A pattern of ink.
    Native(TransferredInk<'a>),
    /// A pattern of 16-bit ink, for a 16-bit ink canvas.
    Native16(TransferredInk<'a, u16>),
}

impl<'a> CanvasPattern<'a> {
    /// `pattern`, of 8-bit sRGB, for a canvas of `kind`, its colours
    /// converted to ink by `separator` and mapped through `transfer` on an
    /// ink canvas.
    pub(super) const fn new(
        kind: CanvasKind,
        pattern: &'a dyn Pattern,
        (transfer, separator): (Option<&'a Transfer>, &'a Separator),
    ) -> Self {
        match kind {
            CanvasKind::Rgb | CanvasKind::Rgb16 => Self::Rgb(pattern),
            CanvasKind::Ink | CanvasKind::Ink16 => {
                Self::Ink(InkPattern::new(pattern, transfer, separator))
            }
        }
    }

    /// `pattern`, of 16-bit sRGB, for a 16-bit canvas of `kind`, its
    /// colours converted to ink at 16 bits by `separator` and mapped
    /// through `transfer` on an ink canvas.
    pub(super) const fn wide(
        kind: CanvasKind,
        pattern: &'a dyn Pattern<u16>,
        (transfer, separator): (Option<&'a Transfer>, &'a Separator),
    ) -> Self {
        if kind.is_ink() {
            Self::Ink16(InkPattern::new(pattern, transfer, separator))
        } else {
            Self::Rgb16(pattern)
        }
    }

    /// `pattern`, of ink in the channels `channels`, for an ink canvas,
    /// its ink mapped through `transfer`.
    pub(super) const fn ink(
        pattern: &'a dyn Pattern,
        transfer: Option<&'a Transfer>,
        channels: ChannelMask,
    ) -> Self {
        Self::Native(TransferredInk::new(pattern, transfer, channels))
    }

    /// `pattern`, of 16-bit ink in the channels `channels`, for a 16-bit
    /// ink canvas, its ink mapped through `transfer`.
    pub(super) const fn ink16(
        pattern: &'a dyn Pattern<u16>,
        transfer: Option<&'a Transfer>,
        channels: ChannelMask,
    ) -> Self {
        Self::Native16(TransferredInk::new(pattern, transfer, channels))
    }

    /// The transfer function `pipe` applies for a pattern on a canvas of
    /// `kind`: `transfer` on an RGB canvas; on an ink canvas the pattern
    /// itself applies it (see [`InkPattern`]), so none.
    pub(super) fn pipe_transfer(
        kind: CanvasKind,
        transfer: Option<&Transfer>,
    ) -> Option<&Transfer> {
        transfer.filter(|_| !kind.is_ink())
    }

    /// The pattern as a canvas of samples of type `S` takes it; `None` for
    /// a pattern of 16-bit samples on an 8-bit canvas, which is never made
    /// for one.
    pub(super) fn at<S: CanvasSample>(&self) -> Option<SourcePattern<'_, S>> {
        match self {
            Self::Rgb(p) => Some(SourcePattern::of_u8(*p)),
            Self::Rgb16(p) => S::pattern_u16(*p).map(SourcePattern::Native),
            Self::Ink16(p) => S::pattern_u16(p).map(SourcePattern::Native),
            Self::Ink(p) => Some(SourcePattern::of_u8(p)),
            Self::Native(p) => Some(SourcePattern::of_u8(p)),
            Self::Native16(p) => S::pattern_u16(p).map(SourcePattern::Native),
        }
    }
}

/// What a paint composites, as a canvas of any depth takes it: a solid
/// colour or a pattern.
#[derive(Clone, Copy)]
pub(super) enum CanvasSource<'a> {
    /// A solid colour, already mapped through the transfer function, and
    /// the separator a derived ink of it is converted by.
    Solid(&'a super::super::color::RasterColor, &'a Separator),
    /// A pattern.
    Pattern(&'a CanvasPattern<'a>),
}

impl CanvasSource<'_> {
    /// Run `paint` on `bitmap` with the source as the pipe takes it on a
    /// canvas of pixels `P`; nothing runs for a pattern the canvas cannot
    /// take (see [`CanvasPattern::at`]).
    pub(super) fn paint<P: Pixel>(
        self,
        bitmap: &mut Bitmap<P>,
        paint: impl FnOnce(&mut Bitmap<P>, &PipeSrc<'_, P::Sample>),
    ) where
        P::Sample: CanvasSample,
    {
        match self {
            Self::Solid(colour, separator) => {
                let samples = P::Sample::solid(colour, P::MODE, separator);
                paint(bitmap, &PipeSrc::Solid(samples));
            }
            Self::Pattern(pattern) => {
                if let Some(source) = pattern.at::<P::Sample>() {
                    paint(bitmap, &PipeSrc::Pattern(source.as_dyn()));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::page::FinishedCanvas;
    use crate::renderer::page::ink_map::SpotTables;
    use color::cmyk::rgb_to_ink;

    /// The 8-bit RGB bitmap of a finished canvas.
    fn rgb8(canvas: PageCanvas) -> Bitmap<Rgb8> {
        let spots = SpotTables {
            narrow: [None; 4],
            wide: [None; 4],
        };
        match canvas.into_finished(&spots, &crate::renderer::page::finished::InkFinish::SRGB) {
            FinishedCanvas::Rgb8(b) => b,
            other => panic!("an 8-bit RGB canvas, not {other:?}"),
        }
    }

    #[test]
    fn a_fresh_canvas_is_paper_white_or_transparent() {
        let rgb = rgb8(PageCanvas::new(CanvasKind::Rgb, 2, 1, false));
        assert!(rgb.data().iter().all(|&v| v == u8::MAX));
        let PageCanvas::Rgb16(wide) = PageCanvas::new(CanvasKind::Rgb16, 2, 1, false) else {
            panic!("a 16-bit canvas");
        };
        assert!(wide.data().iter().all(|&v| v == u16::MAX));
        let ink = PageCanvas::new(CanvasKind::Ink, 2, 1, false);
        let PageCanvas::Ink(bitmap) = &ink else {
            panic!("an ink canvas");
        };
        assert!(bitmap.data().iter().all(|&v| v == 0), "white is no ink");
        assert!(rgb8(ink).data().iter().all(|&v| v == u8::MAX));
        let PageCanvas::Ink(group) = PageCanvas::new(CanvasKind::Ink, 2, 1, true) else {
            panic!("an ink canvas");
        };
        assert_eq!(group.alpha_plane(), Some([0, 0].as_slice()));
    }

    #[test]
    fn a_fallible_canvas_matches_and_reports_refusal() {
        for (kind, alpha) in [
            (CanvasKind::Rgb, false),
            (CanvasKind::Rgb, true),
            (CanvasKind::Rgb16, false),
            (CanvasKind::Rgb16, true),
            (CanvasKind::Ink, false),
            (CanvasKind::Ink, true),
        ] {
            let a = PageCanvas::new(kind, 3, 2, alpha);
            let b = PageCanvas::try_new(kind, 3, 2, alpha).unwrap();
            assert_eq!(a.kind(), b.kind());
            assert_eq!(a.has_alpha(), b.has_alpha());
            let bytes = |c: &PageCanvas| with_canvas!(c, |b| b.row_bytes(0).to_vec());
            assert_eq!(bytes(&a), bytes(&b), "{kind:?} alpha {alpha}");
        }
        // 36 PB of ink: representable, but no allocator grants it.
        assert!(PageCanvas::try_new(CanvasKind::Ink, u32::MAX, 1 << 20, false).is_err());
    }

    #[test]
    fn an_ink_page_group_composites_onto_paper() {
        let mut canvas = PageCanvas::new(CanvasKind::Ink, 1, 1, true);
        let PageCanvas::Ink(bitmap) = &mut canvas else {
            panic!("an ink canvas");
        };
        bitmap.row_bytes_mut(0)[3] = 200;
        bitmap.alpha_plane_mut().unwrap()[0] = 128;
        let mut buffers = CanvasBuffers::default();
        canvas.composite_onto_white(&mut buffers);
        assert_eq!(buffers.ink.alpha, vec![128]);
        assert!(!canvas.has_alpha());
        let PageCanvas::Ink(bitmap) = &canvas else {
            panic!("an ink canvas");
        };
        assert_eq!(bitmap.row_bytes(0), &[0, 0, 0, 100, 0, 0, 0, 0]);
    }

    /// A pattern of sRGB white and black becomes no ink and the ink of
    /// black: solid black with C, M and Y on top.
    #[test]
    fn an_ink_pattern_converts_each_pixel() {
        let rgb = [255, 255, 255, 0, 0, 0, 0, 0, 0];
        let row = RowPattern { data: &rgb };
        let mut out = [9u8; 24];
        InkPattern::new(&row, None, &Separator::Model).fill_span(0, 0, 2, &mut out);
        let [c, m, y, k] = rgb_to_ink([0; 3]);
        assert_eq!(k, 255);
        let black = [c, m, y, k, 0, 0, 0, 0];
        assert_eq!(out[..8], [0; 8]);
        assert_eq!(out[8..16], black);
        assert_eq!(out[16..], black);
    }
}
