//! The page canvas: the bitmap a page is painted on, of RGB or ink pixels.
//!
//! An RGB canvas holds sRGB bytes, `0xFF` being paper white.  An ink canvas
//! holds [`DeviceN8`] pixels, the amounts `[C, M, Y, K, S0, S1, S2, S3]` of
//! four process and four spot inks, `0` being no ink, so that paper white
//! is all zeros; it is converted to sRGB when the page is finished.
//!
//! The raster primitives are generic over the pixel type.  [`with_canvas!`]
//! dispatches one call to the bitmap of either kind, so a paint is written
//! once.  The sources the renderer builds in sRGB — shadings, tiling cells,
//! image samples, mesh triangles — reach an ink canvas through
//! [`InkPattern`], which converts each span to ink.

use std::cell::RefCell;

use color::cmyk::{SpotLut, ink_to_rgb_row, rgb_to_ink};
use color::{DeviceN8, Pixel, PixelMode, Rgb8};
use raster::Bitmap;
use raster::clip::Clip;
use raster::pipe::{Pattern, PipeSrc, PipeState, RowPattern, TransferSet, render_bitmap_span};
use raster::shading::gouraud::{GouraudVertex, gouraud_triangle_fill, gouraud_triangle_spans};
use raster::transparency::{Backdrop, GroupBitmap, GroupParams, begin_group, paint_group};

use super::PageRenderer;
use super::pixel_rect::PixelRect;
use crate::resources::transfer::Transfer;

/// Run `$body` with `$b` bound to the bitmap of `$canvas`, of either pixel
/// type: `with_canvas!(&mut self.canvas, |b| fill(b, ..))` binds `b` as
/// `&mut Bitmap<Rgb8>` in one arm and `&mut Bitmap<DeviceN8>` in the other.
macro_rules! with_canvas {
    ($canvas:expr, |$b:ident| $body:expr) => {
        match $canvas {
            $crate::renderer::page::canvas::PageCanvas::Rgb($b) => $body,
            $crate::renderer::page::canvas::PageCanvas::Ink($b) => $body,
        }
    };
}
pub(super) use with_canvas;

/// The pixel type of a [`PageCanvas`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CanvasKind {
    /// sRGB, three bytes per pixel.
    Rgb,
    /// Process and spot ink amounts, eight bytes per pixel.
    Ink,
}

impl CanvasKind {
    /// The pixel mode of the canvas.
    pub(super) const fn mode(self) -> PixelMode {
        match self {
            Self::Rgb => PixelMode::Rgb8,
            Self::Ink => PixelMode::DeviceN8,
        }
    }
}

/// The bitmap a page, a transparency group or a soft mask is painted on.
pub(super) enum PageCanvas {
    /// sRGB pixels.
    Rgb(Bitmap<Rgb8>),
    /// Ink pixels.
    Ink(Bitmap<DeviceN8>),
}

/// The value of every channel of paper white in pixels of type `P`.
fn paper<P: Pixel>() -> u8 {
    if P::MODE == PixelMode::DeviceN8 {
        0
    } else {
        u8::MAX
    }
}

impl PageCanvas {
    /// A `width × height` canvas of `kind`: transparent, with an alpha
    /// plane, when `alpha`; opaque paper white otherwise.
    pub(super) fn new(kind: CanvasKind, width: u32, height: u32, alpha: bool) -> Self {
        match kind {
            CanvasKind::Rgb if alpha => Self::Rgb(Bitmap::new(width, height, 1, true)),
            CanvasKind::Rgb => Self::Rgb(Bitmap::new_filled(width, height, 1, false, u8::MAX)),
            CanvasKind::Ink => Self::Ink(Bitmap::new(width, height, 1, alpha)),
        }
    }

    /// [`Self::new`] that reports buffers the allocator refuses as an
    /// error.
    pub(super) fn try_new(
        kind: CanvasKind,
        width: u32,
        height: u32,
        alpha: bool,
    ) -> Result<Self, raster::BitmapAllocError> {
        Ok(match kind {
            CanvasKind::Rgb if alpha => {
                Self::Rgb(Bitmap::try_new_filled(width, height, 1, true, 0)?)
            }
            CanvasKind::Rgb => Self::Rgb(Bitmap::try_new_filled(width, height, 1, false, u8::MAX)?),
            CanvasKind::Ink => Self::Ink(Bitmap::try_new_filled(width, height, 1, alpha, 0)?),
        })
    }

    /// The pixel type of the canvas.
    pub(super) const fn kind(&self) -> CanvasKind {
        match self {
            Self::Rgb(_) => CanvasKind::Rgb,
            Self::Ink(_) => CanvasKind::Ink,
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

    /// The canvas as sRGB pixels, keeping its alpha plane.  Ink converts
    /// through [`ink_to_rgb_row`] with the spot colourants `spots`.
    pub(super) fn into_rgb(self, spots: &[Option<&SpotLut>; 4]) -> Bitmap<Rgb8> {
        let ink = match self {
            Self::Rgb(bitmap) => return bitmap,
            Self::Ink(ink) => ink,
        };
        let mut rgb = Bitmap::<Rgb8>::new(ink.width, ink.height, 1, ink.has_alpha());
        let n = ink.width as usize;
        for y in 0..ink.height {
            ink_to_rgb_row(
                &ink.row_bytes(y)[..n * 8],
                spots,
                &mut rgb.row_bytes_mut(y)[..n * 3],
            );
        }
        if let (Some(src), Some(dst)) = (ink.alpha_plane(), rgb.alpha_plane_mut()) {
            dst.copy_from_slice(src);
        }
        rgb
    }

    /// Composite the canvas, a page group with an alpha plane, onto opaque
    /// paper in place (see [`composite_onto_white`]).
    pub(super) fn composite_onto_white(&mut self) {
        with_canvas!(self, |b| composite_onto_white(b));
    }

    /// Make the pixels of `region` transparent.
    pub(super) fn clear(&mut self, region: PixelRect) {
        with_canvas!(self, |b| clear(b, region));
    }

    /// Fill one Gouraud-shaded triangle through `pipe`, clipped to `clip`.
    /// The interpolated sRGB spans reach an ink canvas mapped through
    /// `transfer` and converted to ink; on an RGB canvas `pipe` carries the
    /// transfer function.
    pub(super) fn gouraud_triangle(
        &mut self,
        clip: &Clip,
        pipe: &PipeState<'_>,
        transfer: Option<&Transfer>,
        tri: [GouraudVertex; 3],
    ) {
        match self {
            Self::Rgb(b) => gouraud_triangle_fill(b, clip, pipe, tri),
            Self::Ink(b) => {
                let (width, height) = (b.width, b.height);
                gouraud_triangle_spans(width, height, clip, tri, |rgb, x0, x1, y| {
                    let row = RowPattern { data: rgb };
                    render_bitmap_span::<DeviceN8>(
                        b,
                        pipe,
                        &PipeSrc::Pattern(&InkPattern::new(&row, transfer)),
                        None,
                        x0,
                        x1,
                        y,
                    );
                });
            }
        }
    }

    /// Open a transparency group over this canvas on `bitmap`, a
    /// transparent canvas of the same kind and size (see
    /// [`raster::transparency::begin_group`]).
    ///
    /// # Panics
    ///
    /// Panics if `bitmap` is of the other kind.
    pub(super) fn begin_group(
        &self,
        backdrop: Option<&Backdrop>,
        clip: &Clip,
        params: GroupParams,
        bitmap: Self,
    ) -> CanvasGroup {
        match (self, bitmap) {
            (Self::Rgb(p), Self::Rgb(b)) => {
                CanvasGroup::Rgb(begin_group(p, backdrop, clip, params, b))
            }
            (Self::Ink(p), Self::Ink(b)) => {
                CanvasGroup::Ink(begin_group(p, backdrop, clip, params, b))
            }
            _ => panic!("begin_group: the group canvas differs in kind from its parent"),
        }
    }

    /// Exchange this canvas's bitmap with the bitmap of `group`.
    ///
    /// # Panics
    ///
    /// Panics if `group` is of the other kind.
    pub(super) fn swap_group(&mut self, group: &mut CanvasGroup) {
        match (self, group) {
            (Self::Rgb(c), CanvasGroup::Rgb(g)) => std::mem::swap(c, &mut g.bitmap),
            (Self::Ink(c), CanvasGroup::Ink(g)) => std::mem::swap(c, &mut g.bitmap),
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
    /// Panics if `group` is of the other kind.
    pub(super) fn paint_group(&mut self, group: CanvasGroup, pipe: &PipeState<'_>) -> Self {
        match (self, group) {
            (Self::Rgb(c), CanvasGroup::Rgb(g)) => Self::Rgb(paint_group(c, g, pipe).1),
            (Self::Ink(c), CanvasGroup::Ink(g)) => Self::Ink(paint_group(c, g, pipe).1),
            _ => panic!("paint_group: the group differs in kind from the canvas"),
        }
    }
}

/// An open transparency group of either pixel type.
pub(super) enum CanvasGroup {
    /// A group over an RGB canvas.
    Rgb(GroupBitmap<Rgb8>),
    /// A group over an ink canvas.
    Ink(GroupBitmap<DeviceN8>),
}

impl CanvasGroup {
    /// The group region as half-open pixel ranges `(x0, y0, x1, y1)`.
    pub(super) const fn region(&self) -> (u32, u32, u32, u32) {
        match self {
            Self::Rgb(g) => g.region(),
            Self::Ink(g) => g.region(),
        }
    }

    /// Take the backdrop of a non-isolated group.
    pub(super) const fn take_backdrop(&mut self) -> Option<Backdrop> {
        match self {
            Self::Rgb(g) => g.backdrop.take(),
            Self::Ink(g) => g.backdrop.take(),
        }
    }
}

/// Composite `bitmap`, a page group with an alpha plane, onto opaque paper
/// in place: each channel becomes `c·α + w·(1 − α)`, `w` being paper white,
/// and the alpha plane is dropped.  A bitmap without an alpha plane is left
/// as it is.
pub(super) fn composite_onto_white<P: Pixel>(bitmap: &mut Bitmap<P>) {
    if !bitmap.has_alpha() {
        return;
    }
    let white = paper::<P>();
    let n = bitmap.width as usize * P::BYTES;
    for y in 0..bitmap.height {
        let (row, Some(alpha_row)) = bitmap.row_and_alpha_mut(y) else {
            return;
        };
        for (px, &a) in row[..n].chunks_exact_mut(P::BYTES).zip(&*alpha_row) {
            if a == u8::MAX {
                continue;
            }
            for c in px {
                *c = blend_u8(*c, white, a);
            }
        }
    }
    bitmap.drop_alpha_plane();
}

/// Porter-Duff source-over blend for a single 8-bit channel:
/// `(src × alpha + dst × (255 − alpha)) / 255`.
///
/// The two weights sum to 255, so the sum of products is at most
/// `255 × 255 = 65025`, within `u16`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the sum of products is at most 255 × 255, so the quotient is at most 255"
)]
pub(super) fn blend_u8(src: u8, dst: u8, alpha: u8) -> u8 {
    let (a, ia) = (u16::from(alpha), u16::from(u8::MAX - alpha));
    ((u16::from(src) * a + u16::from(dst) * ia) / 255) as u8
}

/// Make the pixels of `region` transparent: colour and alpha zero.
pub(super) fn clear<P: Pixel>(bitmap: &mut Bitmap<P>, region: PixelRect) {
    let PixelRect { x0, y0, x1, y1 } = region;
    let n = P::BYTES;
    for y in y0..y1 {
        let (row, alpha) = bitmap.row_and_alpha_mut(y);
        row[x0 as usize * n..x1 as usize * n].fill(0);
        if let Some(alpha) = alpha {
            alpha[x0 as usize..x1 as usize].fill(0);
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
    /// Bytes per pixel of the canvas.
    pub(super) bpp: usize,
    /// Colour bytes, `bpp` per pixel, row-major over the rectangle.
    pub(super) color: Vec<u8>,
    /// Alpha per pixel, or `None` for a bitmap without an alpha plane.
    pub(super) alpha: Option<Vec<u8>>,
}

impl Snapshot {
    /// Capture `region` of `bitmap`.
    pub(super) fn take<P: Pixel>(bitmap: &Bitmap<P>, region: PixelRect) -> Self {
        let PixelRect { x0, y0, x1, y1 } = region;
        let bpp = P::BYTES;
        let (cx0, cx1) = (x0 as usize * bpp, x1 as usize * bpp);
        let (ax0, ax1) = (x0 as usize, x1 as usize);
        let mut color = Vec::with_capacity((cx1 - cx0) * (y1 - y0) as usize);
        let mut alpha = bitmap
            .has_alpha()
            .then(|| Vec::with_capacity((ax1 - ax0) * (y1 - y0) as usize));
        for y in y0..y1 {
            color.extend_from_slice(&bitmap.row_bytes(y)[cx0..cx1]);
            if let (Some(a), Some(row)) = (alpha.as_mut(), bitmap.alpha_row(y)) {
                a.extend_from_slice(&row[ax0..ax1]);
            }
        }
        Self {
            x0,
            y0,
            x1,
            bpp,
            color,
            alpha,
        }
    }

    /// Pixels per row of the rectangle.
    pub(super) const fn span(&self) -> usize {
        (self.x1 - self.x0) as usize
    }

    /// Write the snapshot back over its rectangle of `bitmap`.
    fn restore<P: Pixel>(self, bitmap: &mut Bitmap<P>) {
        debug_assert_eq!(
            self.bpp,
            P::BYTES,
            "a snapshot restores onto its own canvas"
        );
        let span = self.span();
        let (cx0, ax0) = (self.x0 as usize * self.bpp, self.x0 as usize);
        for (i, prior) in self.color.chunks_exact(span * self.bpp).enumerate() {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "i counts rows of the snapshot, which lie below the bitmap's u32 height"
            )]
            let y = self.y0 + i as u32;
            let (row, alpha_row) = bitmap.row_and_alpha_mut(y);
            row[cx0..cx0 + span * self.bpp].copy_from_slice(prior);
            if let (Some(prior_a), Some(alpha_row)) = (self.alpha.as_ref(), alpha_row) {
                alpha_row[ax0..ax0 + span].copy_from_slice(&prior_a[i * span..(i + 1) * span]);
            }
        }
    }
}

impl PageRenderer<'_> {
    /// Capture `region` of the canvas.
    pub(super) fn take_region(&self, region: PixelRect) -> Snapshot {
        with_canvas!(&self.canvas, |b| Snapshot::take(b, region))
    }

    /// Write `snapshot`, taken from the canvas, back over its rectangle.
    pub(super) fn restore_region(&mut self, snapshot: Snapshot) {
        with_canvas!(&mut self.canvas, |b| snapshot.restore(b));
    }

    /// The canvas as an RGB bitmap.
    ///
    /// # Panics
    ///
    /// Panics on an ink canvas.
    #[cfg(test)]
    pub(super) fn rgb_canvas(&self) -> &Bitmap<Rgb8> {
        match &self.canvas {
            PageCanvas::Rgb(b) => b,
            PageCanvas::Ink(_) => panic!("rgb_canvas: the canvas holds ink"),
        }
    }
}

thread_local! {
    /// The sRGB span an [`InkPattern`] converts.  Separate from the pipe's
    /// own span buffers, which are held while a pattern fills a span.  It
    /// is borrowed while the inner pattern fills it, so an `InkPattern`
    /// never wraps another: the inner pattern yields sRGB, never ink.
    /// Grow-never-shrink.
    static RGB_SPAN: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// A [`Pattern`] of sRGB bytes served as ink: each pixel the inner pattern
/// yields is mapped through the transfer function, if any, and becomes the
/// ink [`rgb_to_ink`] gives it, with no spot ink.  The pattern only
/// converts; coverage and the static-colour hint are the inner pattern's.
pub(super) struct InkPattern<'a> {
    pattern: &'a dyn Pattern,
    transfer: Option<&'a Transfer>,
}

impl<'a> InkPattern<'a> {
    /// `pattern` served as ink, its colours mapped through `transfer`.
    pub(super) fn new(pattern: &'a dyn Pattern, transfer: Option<&'a Transfer>) -> Self {
        Self { pattern, transfer }
    }

    /// Fill `out` with ink by running `fill` on an sRGB span of the same
    /// pixel count.
    fn convert(&self, out: &mut [u8], fill: impl FnOnce(&mut [u8])) {
        RGB_SPAN.with_borrow_mut(|rgb| {
            rgb.resize(out.len() / 8 * 3, 0);
            fill(rgb);
            let mut last: Option<([u8; 3], [u8; 8])> = None;
            for (src, dst) in rgb
                .as_chunks::<3>()
                .0
                .iter()
                .zip(out.as_chunks_mut::<8>().0)
            {
                *dst = match last {
                    Some((seen, ink)) if seen == *src => ink,
                    _ => {
                        let mapped = self.transfer.map_or(*src, |t| t.apply(*src));
                        let [c, m, y, k] = rgb_to_ink(mapped);
                        let ink = [c, m, y, k, 0, 0, 0, 0];
                        last = Some((*src, ink));
                        ink
                    }
                };
            }
        });
    }
}

impl Pattern for InkPattern<'_> {
    fn fill_span(&self, y: i32, x0: i32, x1: i32, out: &mut [u8]) {
        self.convert(out, |rgb| self.pattern.fill_span(y, x0, x1, rgb));
    }

    fn is_static_color(&self) -> bool {
        self.pattern.is_static_color()
    }

    fn may_leave_unpainted(&self) -> bool {
        self.pattern.may_leave_unpainted()
    }

    fn fill_span_coverage(&self, y: i32, x0: i32, x1: i32, out: &mut [u8], coverage: &mut [u8]) {
        self.convert(out, |rgb| {
            self.pattern.fill_span_coverage(y, x0, x1, rgb, coverage);
        });
    }
}

/// A pattern of sRGB bytes as a canvas of a given kind takes it: as it is
/// on an RGB canvas, where the pipe applies the transfer function, and
/// through [`InkPattern`], which applies it, on an ink canvas.
pub(super) enum CanvasPattern<'a> {
    /// The pattern itself.
    Rgb(&'a dyn Pattern),
    /// The pattern converted to ink.
    Ink(InkPattern<'a>),
}

impl<'a> CanvasPattern<'a> {
    /// `pattern` for a canvas of `kind`, its colours mapped through
    /// `transfer` on an ink canvas.
    pub(super) fn new(
        kind: CanvasKind,
        pattern: &'a dyn Pattern,
        transfer: Option<&'a Transfer>,
    ) -> Self {
        match kind {
            CanvasKind::Rgb => Self::Rgb(pattern),
            CanvasKind::Ink => Self::Ink(InkPattern::new(pattern, transfer)),
        }
    }

    /// The transfer set `pipe` applies for a pattern on a canvas of `kind`:
    /// the tables of `transfer` on an RGB canvas; on an ink canvas the
    /// pattern itself applies them (see [`InkPattern`]), so none.
    pub(super) fn pipe_transfer(
        kind: CanvasKind,
        transfer: Option<&Transfer>,
    ) -> Option<TransferSet<'_>> {
        transfer
            .filter(|_| kind == CanvasKind::Rgb)
            .map(PageRenderer::transfer_set)
    }

    /// The pattern to hand to the pipe.
    pub(super) fn as_dyn(&self) -> &dyn Pattern {
        match self {
            Self::Rgb(p) => *p,
            Self::Ink(p) => p,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_canvas_is_paper_white_or_transparent() {
        let rgb = PageCanvas::new(CanvasKind::Rgb, 2, 1, false).into_rgb(&[None; 4]);
        assert!(rgb.data().iter().all(|&v| v == u8::MAX));
        let ink = PageCanvas::new(CanvasKind::Ink, 2, 1, false);
        let PageCanvas::Ink(bitmap) = &ink else {
            panic!("an ink canvas");
        };
        assert!(bitmap.data().iter().all(|&v| v == 0), "white is no ink");
        assert!(
            ink.into_rgb(&[None; 4])
                .data()
                .iter()
                .all(|&v| v == u8::MAX)
        );
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
            (CanvasKind::Ink, false),
            (CanvasKind::Ink, true),
        ] {
            let a = PageCanvas::new(kind, 3, 2, alpha);
            let b = PageCanvas::try_new(kind, 3, 2, alpha).unwrap();
            assert_eq!(a.kind(), b.kind());
            assert_eq!(a.has_alpha(), b.has_alpha());
            let bytes = |c: &PageCanvas| with_canvas!(c, |b| b.data().to_vec());
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
        canvas.composite_onto_white();
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
        InkPattern::new(&row, None).fill_span(0, 0, 2, &mut out);
        let [c, m, y, k] = rgb_to_ink([0; 3]);
        assert_eq!(k, 255);
        let black = [c, m, y, k, 0, 0, 0, 0];
        assert_eq!(out[..8], [0; 8]);
        assert_eq!(out[8..16], black);
        assert_eq!(out[16..], black);
    }
}
