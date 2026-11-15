//! Soft masks set through `ExtGState` `SMask` (PDF §11.6.5.2).
//!
//! `gs` resolves the mask group to one value per device pixel: the group is
//! rendered onto a transparent bitmap with the CTM in effect at `gs`, then
//! reduced to its luminosity over the backdrop colour `BC` (`/Luminosity`)
//! or to its alpha (`/Alpha`), and mapped through the transfer function.
//!
//! An object painted under a mask is painted normally, then each pixel it
//! touched is moved back toward its prior value: with mask value `m`, the
//! result is `prior + (painted − prior) × m / MAX`, taken on
//! alpha-premultiplied colour when the bitmap carries alpha.  Every
//! compositing operation is linear in its source alpha in that space, so the
//! result equals painting the object with its alpha scaled by `m`, whatever
//! the paint path (fill, stroke, glyph, image, shading, group).  The prior
//! pixels are captured over the object's own rectangle within the clip,
//! which each painting primitive knows.

use std::borrow::Cow;
use std::sync::Arc;

use color::{Pixel, Sample, Wide};
use raster::Bitmap;
use raster::types::BlendMode;

use super::PageRenderer;
use super::canvas::{CanvasKind, PageCanvas, Snapshot, with_canvas};
use super::depth::{CanvasSample, SampleVec};
use super::pixel_rect::PixelRect;
use crate::renderer::color::{RasterColor, Srgb};
use crate::renderer::gstate::SoftMaskPlane;
use crate::resources::shading::function::Function;
use crate::resources::transfer::tabulate;
use crate::resources::{SoftMaskParams, colorspace, resolve_dict};

impl PageRenderer<'_> {
    /// Apply an `ExtGState` `SMask` entry to the current graphics state.
    pub(super) fn set_soft_mask(&mut self, params: &SoftMaskParams) {
        let plane = match params {
            SoftMaskParams::None => None,
            SoftMaskParams::Group {
                luminosity,
                group,
                backdrop,
                transfer,
            } => {
                let plane = self.render_soft_mask(*luminosity, *group, backdrop, transfer.as_ref());
                if plane.is_none() && self.budget_exceeded.is_none() {
                    log::warn!(
                        "rasterrocket-interp: soft mask group {group:?} unreadable — painting unmasked"
                    );
                }
                plane.map(Arc::new)
            }
        };
        self.gstate.current_mut().soft_mask = plane;
    }

    /// Render the mask group `group` and reduce it to mask values.  A mask
    /// canvas the allocator refuses stops the render.
    fn render_soft_mask(
        &mut self,
        luminosity: bool,
        group: pdf::ObjectId,
        backdrop: &[f64],
        transfer: Option<&pdf::Object>,
    ) -> Option<SoftMaskPlane> {
        let form = self.resources.form_from_stream_id(group)?;
        // A mask group yields mask values, not printed colorants, so it is
        // painted in sRGB whatever the page's canvas: its luminosity is
        // that of its colours (ISO 32000-1 §11.6.5.2), not of the inks
        // that print them, which reproduce a colour only within the
        // process gamut.  It is painted at the depth of the page.
        let depth = self.canvas.kind().depth();
        let mask = self.try_canvas(CanvasKind::rgb(depth), self.width, self.height)?;

        // The mask group is painted with the graphics state in effect at
        // `gs`, except that the soft mask is `None`, the alpha constants are
        // 1 and the blend mode is `Normal`.  It is composited against an
        // opaque backdrop of `BC` (ISO 32000-1 §11.6.5.2), so it sees
        // nothing of the enclosing group: neither its backdrop, whose
        // pixels are of the page canvas's kind, nor its knockout.
        self.gstate.save();
        {
            let gs = self.gstate.current_mut();
            gs.soft_mask = None;
            gs.fill_alpha = 255;
            gs.stroke_alpha = 255;
            gs.blend_mode = BlendMode::Normal;
        }
        let enclosing = std::mem::take(&mut self.group);
        let page = std::mem::replace(&mut self.canvas, mask);
        self.do_form_xobject(&form);
        let canvas = std::mem::replace(&mut self.canvas, page);
        self.group = enclosing;
        self.gstate.restore();

        let function = transfer.map(|f| Function::load(self.resources.doc(), f));
        let reduce = if luminosity {
            Reduce::Luminosity(self.mask_backdrop_rgb(group, backdrop))
        } else {
            Reduce::Alpha
        };
        let values = match canvas {
            PageCanvas::Rgb(b) => mask_values(&b, reduce, function.as_deref()),
            PageCanvas::Rgb16(b) => mask_values(&b, reduce, function.as_deref()),
            PageCanvas::Ink(_) | PageCanvas::Ink16(_) => {
                debug_assert!(false, "a soft mask group is painted on an RGB canvas");
                return None;
            }
        };
        let values = match values {
            Ok(values) => values?,
            Err(e) => {
                self.abort_allocation(e);
                return None;
            }
        };
        Some(SoftMaskPlane {
            width: self.width,
            height: self.height,
            values,
        })
    }

    /// The backdrop colour `BC` of the mask group `group` as sRGB at the
    /// page's depth: `BC` is given in the group's colour space (`/Group
    /// /CS`, PDF §11.6.5.2).  A group without a space of `BC`'s component
    /// count reads `BC` by its count (see [`backdrop_rgb`]).
    fn mask_backdrop_rgb(&self, group: pdf::ObjectId, bc: &[f64]) -> Srgb {
        let doc = self.resources.doc();
        let space = doc.get_object(group).ok().and_then(|obj| {
            let group_dict = obj.as_stream()?.dict.get(b"Group")?;
            let cs = resolve_dict(doc, group_dict)?.get(b"CS")?.clone();
            Some(colorspace::resolve(doc, &cs))
        });
        match space {
            Some(space) if !bc.is_empty() && space.ncomponents() == bc.len() => {
                space.convert_to_srgb(doc, bc, self.canvas.kind().depth())
            }
            _ => backdrop_rgb(bc),
        }
    }

    /// Run `paint` — the painting of one object of constant alpha `alpha`
    /// within `region`, the pixels it can touch — under the soft mask in
    /// effect, if any, and knocking out the group's earlier content within
    /// the object's shape when the group being painted into is a knockout
    /// group (see [`Self::paint_knocking_out`]).
    ///
    /// The mask is removed from the graphics state while `paint` runs, so
    /// that painting it performs through other primitives (a Type 3 glyph's
    /// content, a group's compositing) is not masked a second time.
    pub(super) fn paint_object(
        &mut self,
        region: Option<PixelRect>,
        alpha: u8,
        paint: impl FnOnce(&mut Self),
    ) {
        #[cfg(feature = "vulkan")]
        if self.device_recording() && self.gstate.current().soft_mask.is_some() {
            self.refuse_device("soft mask");
            return;
        }
        self.paint_knocking_out(region, alpha, |r| r.paint_masked(region, paint));
    }

    /// Run `paint` — the painting of one object within `region` — under the
    /// soft mask in effect, if any.
    fn paint_masked(&mut self, region: Option<PixelRect>, paint: impl FnOnce(&mut Self)) {
        let Some(mask) = self.gstate.current_mut().soft_mask.take() else {
            paint(self);
            return;
        };
        self.paint_through_soft_mask(&mask, region, paint);
        self.gstate.current_mut().soft_mask = Some(mask);
    }

    /// Run `paint` — the painting of one object within `region`, the
    /// pixels it can touch — under `mask`.  A mask of the wrong size, or a
    /// region outside the clip, leaves the painting unmasked; a snapshot of
    /// the region the allocator refuses stops the render, `paint` unrun.
    pub(super) fn paint_through_soft_mask(
        &mut self,
        mask: &SoftMaskPlane,
        region: Option<PixelRect>,
        paint: impl FnOnce(&mut Self),
    ) {
        let region = region
            .filter(|_| mask.width == self.width && mask.height == self.height)
            .and_then(|r| self.within_clip(r));
        let Some(region) = region else {
            paint(self);
            return;
        };
        let view = MaskView {
            values: MaskValues::Plane(&mask.values),
            width: mask.width as usize,
            x0: 0,
            y0: 0,
        };
        let Some(before) = self.take_region(region) else {
            return;
        };
        paint(self);
        with_canvas!(&mut self.canvas, |b| before.blend_back(b, &view));
    }

    /// Run `paint` with what it paints scaled by `coverage`, one value per
    /// pixel of `rect` in row-major order; pixels outside `rect` are left
    /// unchanged.  `coverage` must hold `rect.area()` values.  A snapshot
    /// the allocator refuses stops the render, `paint` unrun.
    pub(super) fn paint_through_coverage(
        &mut self,
        rect: PixelRect,
        coverage: &[u8],
        paint: impl FnOnce(&mut Self),
    ) {
        debug_assert_eq!(
            coverage.len() as u64,
            rect.area(),
            "one coverage value per pixel of the rectangle"
        );
        let region = self.within_clip(rect);
        let Some(region) = region else {
            paint(self);
            return;
        };
        let view = MaskView {
            values: MaskValues::Coverage(coverage),
            width: rect.width() as usize,
            x0: rect.x0,
            y0: rect.y0,
        };
        let Some(before) = self.take_region(region) else {
            return;
        };
        paint(self);
        with_canvas!(&mut self.canvas, |b| before.blend_back(b, &view));
    }
}

impl PageRenderer<'_> {
    /// Run `under` and then `over` with `over` knocking out `under` within
    /// `rect`: each pixel there ends as `U + (O − U) × coverage / 255`,
    /// where `U` is the pixel after `under` and `O` the pixel `over` paints
    /// onto the backdrop as it was before `under`.  `coverage` holds one
    /// value per pixel of `rect` in row-major order; outside `rect`, only
    /// `under` paints.
    pub(super) fn paint_knocked_out(
        &mut self,
        rect: PixelRect,
        coverage: &[u8],
        under: impl FnOnce(&mut Self),
        over: impl FnOnce(&mut Self),
    ) {
        debug_assert_eq!(
            coverage.len() as u64,
            rect.area(),
            "one coverage value per pixel of the rectangle"
        );
        let region = self.within_clip(rect);
        let Some(region) = region else {
            under(self);
            return;
        };
        let view = MaskView {
            values: MaskValues::Coverage(coverage),
            width: rect.width() as usize,
            x0: rect.x0,
            y0: rect.y0,
        };
        let Some(backdrop) = self.take_region(region) else {
            return;
        };
        under(self);
        let Some(painted) = self.take_region(region) else {
            return;
        };
        self.restore_region(backdrop);
        over(self);
        with_canvas!(&mut self.canvas, |b| painted.blend_back(b, &view));
    }
}

/// The values of a [`MaskView`]: a soft mask's plane, of the canvas's
/// depth, or 8-bit coverage, widened for a canvas of another depth.
#[derive(Clone, Copy)]
enum MaskValues<'a> {
    /// A soft mask's values.
    Plane(&'a SampleVec),
    /// Coverage from the rasteriser.
    Coverage(&'a [u8]),
}

impl<'a> MaskValues<'a> {
    /// The values as samples of type `S`; `None` for a plane of another
    /// depth, which a canvas is never given.
    fn at<S: CanvasSample>(self) -> Option<Cow<'a, [S]>> {
        match self {
            Self::Plane(values) => S::samples(values).map(Cow::Borrowed),
            Self::Coverage(bytes) => Some(S::from_u8_slice(bytes).map_or_else(
                || Cow::Owned(bytes.iter().map(|&v| S::from_u8(v)).collect()),
                Cow::Borrowed,
            )),
        }
    }
}

/// Mask values over a rectangle of the bitmap whose top-left pixel is
/// `(x0, y0)`, `width` values per row.
struct MaskView<'a> {
    values: MaskValues<'a>,
    width: usize,
    x0: u32,
    y0: u32,
}

impl Snapshot {
    /// Replace each pixel of the rectangle by its interpolation between the
    /// snapshot and its current value, weighted by the mask.
    /// The snapshot lies within the mask's rectangle.
    fn blend_back<P: Pixel>(self, bitmap: &mut Bitmap<P>, mask: &MaskView<'_>)
    where
        P::Sample: CanvasSample,
    {
        debug_assert_eq!(
            self.ncomps,
            P::NCOMPS,
            "a snapshot blends onto its own canvas"
        );
        let (Some((colour, alpha)), Some(values)) =
            (self.samples::<P::Sample>(), mask.values.at::<P::Sample>())
        else {
            debug_assert!(false, "a snapshot and its mask are of the canvas's depth");
            return;
        };
        let (span, n) = (self.span(), P::NCOMPS);
        let (cx0, ax0) = (self.x0 as usize * n, self.x0 as usize);
        let mx0 = (self.x0 - mask.x0) as usize;
        for (i, prior) in colour.chunks_exact(span * n).enumerate() {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "i counts rows of the snapshot, which lie below the bitmap's u32 height"
            )]
            let y = self.y0 + i as u32;
            let m_off = (y - mask.y0) as usize * mask.width + mx0;
            let m = &values[m_off..m_off + span];
            let (row, alpha_row) = bitmap.row_and_alpha_mut(y);
            let row = &mut row[cx0..cx0 + span * n];
            match (alpha, alpha_row) {
                (Some(prior_a), Some(alpha_row)) => blend_back_premultiplied(
                    n,
                    prior,
                    &prior_a[i * span..(i + 1) * span],
                    row,
                    &mut alpha_row[ax0..ax0 + span],
                    m,
                ),
                _ => blend_back_opaque(n, prior, row, m),
            }
        }
    }
}

/// `painted ← prior + (painted − prior) × m / MAX` per channel, rounded,
/// for a bitmap without alpha whose pixels are `n` samples.
fn blend_back_opaque<S: Sample>(n: usize, prior: &[S], painted: &mut [S], mask: &[S]) {
    for ((p, c), &m) in prior
        .chunks_exact(n)
        .zip(painted.chunks_exact_mut(n))
        .zip(mask)
    {
        if m == S::MAX {
            continue;
        }
        let (w, iw) = (m.wide(), S::max_wide() - m.wide());
        for (&b, a) in p.iter().zip(c.iter_mut()) {
            *a = S::div_max(a.wide() * w + b.wide() * iw);
        }
    }
}

/// The interpolation of [`blend_back_opaque`] on alpha-premultiplied
/// colour, for a bitmap that stores straight colour beside an alpha plane.
fn blend_back_premultiplied<S: Sample>(
    n: usize,
    prior: &[S],
    prior_a: &[S],
    painted: &mut [S],
    painted_a: &mut [S],
    mask: &[S],
) {
    let two = <S::Wide as Wide>::of(2);
    for ((((p, &pa), c), ca), &m) in prior
        .chunks_exact(n)
        .zip(prior_a)
        .zip(painted.chunks_exact_mut(n))
        .zip(painted_a.iter_mut())
        .zip(mask)
    {
        if m == S::MAX {
            continue;
        }
        // Premultiplied numerators, scaled by MAX².
        let wa = ca.wide() * m.wide();
        let wb = pa.wide() * (S::max_wide() - m.wide());
        let alpha_scaled = wa + wb;
        if alpha_scaled == <S::Wide as Wide>::ZERO {
            p.iter().zip(c.iter_mut()).for_each(|(&b, x)| *x = b);
            *ca = S::default();
            continue;
        }
        for (&b, x) in p.iter().zip(c.iter_mut()) {
            let num = x.wide() * wa + b.wide() * wb;
            // A weighted mean of samples is at most MAX.
            *x = S::from_wide((num + alpha_scaled / two) / alpha_scaled);
        }
        *ca = S::div_max(alpha_scaled);
    }
}

/// Backdrop colour `BC` as sRGB; black when absent.
///
/// The component count selects the space: 1 grey, 3 RGB, 4 CMYK.
fn backdrop_rgb(bc: &[f64]) -> Srgb {
    let c = match *bc {
        [g] => RasterColor::gray(g),
        [r, g, b] => RasterColor::rgb(r, g, b),
        [c, m, y, k] => RasterColor::cmyk(c, m, y, k),
        _ => RasterColor::gray(0.0),
    };
    Srgb {
        bytes: c.bytes(),
        wide: *c.rgb16(),
    }
}

/// An empty buffer with room for the `width × height` values of a mask
/// plane.
fn mask_plane<S: Sample>(width: u32, height: u32) -> Result<Vec<S>, raster::BitmapAllocError> {
    let error = raster::BitmapAllocError { width, height };
    let len = (width as usize).checked_mul(height as usize).ok_or(error)?;
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|_| error)?;
    Ok(values)
}

/// How a mask group's canvas reduces to mask values.
#[derive(Clone, Copy)]
enum Reduce {
    /// The luminosity of the group over the opaque backdrop colour.
    Luminosity(Srgb),
    /// The group's alpha.
    Alpha,
}

/// The mask values of `group`, the mask group's canvas, reduced as
/// `reduce` gives and mapped through the transfer function `function`, if
/// any, tabulated at the canvas's depth.  `Ok(None)` for an alpha mask of
/// a canvas without an alpha plane; the error names a plane the allocator
/// refuses.
fn mask_values<P: Pixel>(
    group: &Bitmap<P>,
    reduce: Reduce,
    function: Option<&Function>,
) -> Result<Option<SampleVec>, raster::BitmapAllocError>
where
    P::Sample: CanvasSample,
{
    let mut values = mask_plane::<P::Sample>(group.width, group.height)?;
    match reduce {
        Reduce::Luminosity(bc) => luminosity_values(group, bc.at(), &mut values),
        Reduce::Alpha => match group.alpha_plane() {
            Some(alpha) => values.extend_from_slice(alpha),
            None => return Ok(None),
        },
    }
    if let Some(function) = function {
        let mut lut = vec![P::Sample::default(); P::Sample::identity_table().as_ref().len()];
        if tabulate(function, &mut lut).is_some() {
            for v in &mut values {
                *v = lut[sample_index(*v)];
            }
        }
    }
    Ok(Some(P::Sample::into_samples(values)))
}

/// The index of `v` in a table of one entry per sample value.
fn sample_index<S: Sample>(v: S) -> usize {
    usize::try_from(v.signed()).unwrap_or(0)
}

/// Append to `out` the luminosity mask values: each pixel of `group`
/// composited over the opaque backdrop `bc`, reduced to its luminosity by
/// [`color::convert::luminosity_at`] (PDF §11.3.5.3).
fn luminosity_values<P: Pixel>(group: &Bitmap<P>, bc: [P::Sample; 3], out: &mut Vec<P::Sample>) {
    let w = group.width as usize;
    for y in 0..group.height {
        let row = group.row_samples(y);
        let alpha = group.alpha_row(y);
        for x in 0..w {
            let a = alpha.map_or(P::Sample::MAX, |r| r[x]).wide();
            let ia = P::Sample::max_wide() - a;
            let px = &row[x * 3..x * 3 + 3];
            let ch = |i: usize| P::Sample::div_max(px[i].wide() * a + bc[i].wide() * ia);
            out.push(color::convert::luminosity_at(ch(0), ch(1), ch(2)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use color::Rgb8;

    #[test]
    fn opaque_blend_back_scales_the_change_by_the_mask() {
        let prior = [200u8, 100, 0, 200, 100, 0, 200, 100, 0];
        let mut painted = [0u8, 0, 255, 0, 0, 255, 0, 0, 255];
        blend_back_opaque(3, &prior, &mut painted, &[0, 255, 128]);
        assert_eq!(painted, [200, 100, 0, 0, 0, 255, 100, 50, 128]);
    }

    #[test]
    fn premultiplied_blend_back_matches_painting_at_reduced_alpha() {
        // Transparent backdrop; black painted at full alpha; mask 25 %.
        // Painting black at alpha 64 gives black at alpha 64.
        let mut c = [0u8, 0, 0];
        let mut a = [255u8];
        blend_back_premultiplied(3, &[255, 255, 255], &[0], &mut c, &mut a, &[64]);
        assert_eq!((c, a), ([0, 0, 0], [64]));

        // Opaque white backdrop; black painted; mask 50 %: mid-grey, opaque.
        let mut c = [0u8, 0, 0];
        let mut a = [255u8];
        blend_back_premultiplied(3, &[255, 255, 255], &[255], &mut c, &mut a, &[128]);
        assert_eq!((c, a), ([127, 127, 127], [255]));

        // Mask zero restores the prior pixel, including a transparent one.
        let mut c = [9u8, 9, 9];
        let mut a = [255u8];
        blend_back_premultiplied(3, &[1, 2, 3], &[0], &mut c, &mut a, &[0]);
        assert_eq!((c, a), ([1, 2, 3], [0]));
    }

    #[test]
    fn luminosity_composites_over_the_backdrop() {
        let mut bmp = Bitmap::<Rgb8>::new(3, 1, 1, true);
        bmp.row_bytes_mut(0)
            .copy_from_slice(&[255, 255, 255, 255, 0, 0, 0, 0, 0]);
        bmp.alpha_plane_mut()
            .unwrap()
            .copy_from_slice(&[255, 255, 0]);
        let values = |bc| {
            let mut out = mask_plane::<u8>(3, 1).unwrap();
            luminosity_values(&bmp, bc, &mut out);
            out
        };
        // White, pure red, and a transparent pixel showing the backdrop.
        assert_eq!(values([0, 0, 0]), [255, 77, 0]);
        assert_eq!(values([255, 255, 255]), [255, 77, 255]);
    }

    #[test]
    fn a_mask_plane_beyond_the_address_space_is_an_error() {
        assert!(mask_plane::<u8>(4, 2).unwrap().capacity() >= 8);
        let refused = mask_plane::<u8>(u32::MAX, u32::MAX).unwrap_err();
        assert_eq!((refused.width, refused.height), (u32::MAX, u32::MAX));
    }

    /// `BC` is read in the mask group's colour space: Lab white is white,
    /// where read by its component count it would be RGB (100, 0, 0).
    #[test]
    fn the_backdrop_is_read_in_the_group_colour_space() {
        let doc = crate::test_helpers::make_doc_with_objects(&[
            (4, b"<< /Type /Page >>".to_vec()),
            (
                5,
                b"<< /Type /XObject /Subtype /Form /BBox [0 0 1 1] /Group << /S /Transparency \
                  /CS [/Lab << /WhitePoint [0.9505 1 1.089] >>] >> /Length 0 >>\nstream\n\nendstream"
                    .to_vec(),
            ),
        ]);
        let renderer = PageRenderer::new(1, 1, &doc, (4, 0));
        let white = renderer.mask_backdrop_rgb((5, 0), &[100.0, 0.0, 0.0]).bytes;
        assert!(white.iter().all(|&v| v >= 250), "{white:?}");
        // A count that does not match the group's space falls back to it.
        assert_eq!(
            renderer.mask_backdrop_rgb((5, 0), &[1.0]).bytes,
            [255, 255, 255]
        );
        assert_eq!(
            renderer.mask_backdrop_rgb((9, 0), &[1.0, 0.0, 0.0]).bytes,
            [255, 0, 0]
        );
    }

    #[test]
    fn backdrop_component_count_selects_the_space() {
        assert_eq!(backdrop_rgb(&[]).bytes, [0, 0, 0]);
        assert_eq!(backdrop_rgb(&[1.0]).bytes, [255, 255, 255]);
        assert_eq!(backdrop_rgb(&[1.0, 0.0, 0.0]).bytes, [255, 0, 0]);
        assert_eq!(backdrop_rgb(&[1.0, 0.0, 0.0]).wide, [u16::MAX, 0, 0]);
    }
}
