//! Soft masks set through `ExtGState` `SMask` (PDF §11.6.5.2).
//!
//! `gs` resolves the mask group to one value per device pixel: the group is
//! rendered onto a transparent bitmap with the CTM in effect at `gs`, then
//! reduced to its luminosity over the backdrop colour `BC` (`/Luminosity`)
//! or to its alpha (`/Alpha`), and mapped through the transfer function.
//!
//! An object painted under a mask is painted normally, then each pixel it
//! touched is moved back toward its prior value: with mask value `m`, the
//! result is `prior + (painted − prior) × m / 255`, taken on
//! alpha-premultiplied colour when the bitmap carries alpha.  Every
//! compositing operation is linear in its source alpha in that space, so the
//! result equals painting the object with its alpha scaled by `m`, whatever
//! the paint path (fill, stroke, glyph, image, shading, group).  The prior
//! pixels are captured over the object's own rectangle within the clip,
//! which each painting primitive knows.

use std::sync::Arc;

use color::Rgb8;
use raster::Bitmap;
use raster::types::BlendMode;

use super::PageRenderer;
use super::pixel_rect::PixelRect;
use crate::renderer::color::RasterColor;
use crate::renderer::gstate::SoftMaskPlane;
use crate::resources::transfer::function_lut;
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
                if plane.is_none() {
                    log::warn!(
                        "rasterrocket-interp: soft mask group {group:?} unreadable — painting unmasked"
                    );
                }
                plane.map(Arc::new)
            }
        };
        self.gstate.current_mut().soft_mask = plane;
    }

    /// Render the mask group `group` and reduce it to mask values.
    fn render_soft_mask(
        &mut self,
        luminosity: bool,
        group: pdf::ObjectId,
        backdrop: &[f64],
        transfer: Option<&pdf::Object>,
    ) -> Option<SoftMaskPlane> {
        let form = self.resources.form_from_stream_id(group)?;

        // The mask group is painted with the graphics state in effect at
        // `gs`, except that the soft mask is `None`, the alpha constants are
        // 1 and the blend mode is `Normal`.  The canvas is a group of its
        // own, not the knockout group that may be in effect.
        self.gstate.save();
        {
            let gs = self.gstate.current_mut();
            gs.soft_mask = None;
            gs.fill_alpha = 255;
            gs.stroke_alpha = 255;
            gs.blend_mode = BlendMode::Normal;
        }
        let knockout = std::mem::replace(&mut self.group.knockout, false);
        let mut canvas = Bitmap::<Rgb8>::new(self.width, self.height, 1, true);
        std::mem::swap(&mut self.bitmap, &mut canvas);
        self.do_form_xobject(&form);
        std::mem::swap(&mut self.bitmap, &mut canvas);
        self.group.knockout = knockout;
        self.gstate.restore();

        let lut = transfer.and_then(|f| function_lut(self.resources.doc(), f));
        let values = if luminosity {
            luminosity_values(&canvas, self.mask_backdrop_rgb(group, backdrop))
        } else {
            canvas.alpha_plane()?.to_vec()
        };
        let values = match lut {
            Some(lut) => values.into_iter().map(|v| lut[usize::from(v)]).collect(),
            None => values,
        };
        Some(SoftMaskPlane {
            width: self.width,
            height: self.height,
            values,
        })
    }

    /// The backdrop colour `BC` of the mask group `group` as sRGB bytes:
    /// `BC` is given in the group's colour space (`/Group /CS`, PDF
    /// §11.6.5.2).  A group without a space of `BC`'s component count reads
    /// `BC` by its count (see [`backdrop_rgb`]).
    fn mask_backdrop_rgb(&self, group: pdf::ObjectId, bc: &[f64]) -> [u8; 3] {
        let doc = self.resources.doc();
        let space = doc.get_object(group).ok().and_then(|obj| {
            let group_dict = obj.as_stream()?.dict.get(b"Group")?;
            let cs = resolve_dict(doc, group_dict)?.get(b"CS")?.clone();
            Some(colorspace::resolve(doc, &cs))
        });
        match space {
            Some(space) if !bc.is_empty() && space.ncomponents() == bc.len() => {
                space.convert_to_rgb(doc, bc)
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
    pub(super) fn paint_object<R>(
        &mut self,
        region: Option<PixelRect>,
        alpha: u8,
        paint: impl FnOnce(&mut Self) -> R,
    ) -> R {
        self.paint_knocking_out(region, alpha, |r| r.paint_masked(region, paint))
    }

    /// Run `paint` — the painting of one object within `region` — under the
    /// soft mask in effect, if any.
    fn paint_masked<R>(
        &mut self,
        region: Option<PixelRect>,
        paint: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let Some(mask) = self.gstate.current_mut().soft_mask.take() else {
            return paint(self);
        };
        let result = self.paint_through_soft_mask(&mask, region, paint);
        self.gstate.current_mut().soft_mask = Some(mask);
        result
    }

    /// Run `paint` — the painting of one object within `region`, the
    /// pixels it can touch — under `mask`.  A mask of the wrong size, or a
    /// region outside the clip, leaves the painting unmasked.
    pub(super) fn paint_through_soft_mask<R>(
        &mut self,
        mask: &SoftMaskPlane,
        region: Option<PixelRect>,
        paint: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let region = region
            .filter(|_| mask.width == self.width && mask.height == self.height)
            .and_then(|r| {
                r.intersect(PixelRect::of_clip(
                    &self.gstate.current().clip,
                    self.width,
                    self.height,
                )?)
            });
        let Some(region) = region else {
            return paint(self);
        };
        let view = MaskView {
            values: &mask.values,
            width: mask.width as usize,
            x0: 0,
            y0: 0,
        };
        let before = Snapshot::take(&self.bitmap, region);
        let result = paint(self);
        before.blend_back(&mut self.bitmap, &view);
        result
    }

    /// Run `paint` with what it paints scaled by `coverage`, one value per
    /// pixel of `rect` in row-major order; pixels outside `rect` are left
    /// unchanged.  `coverage` must hold `rect.area()` values.
    pub(super) fn paint_through_coverage<R>(
        &mut self,
        rect: PixelRect,
        coverage: &[u8],
        paint: impl FnOnce(&mut Self) -> R,
    ) -> R {
        debug_assert_eq!(
            coverage.len() as u64,
            rect.area(),
            "one coverage value per pixel of the rectangle"
        );
        let region = PixelRect::of_clip(&self.gstate.current().clip, self.width, self.height)
            .and_then(|clip| clip.intersect(rect));
        let Some(region) = region else {
            return paint(self);
        };
        let view = MaskView {
            values: coverage,
            width: rect.width() as usize,
            x0: rect.x0,
            y0: rect.y0,
        };
        let before = Snapshot::take(&self.bitmap, region);
        let result = paint(self);
        before.blend_back(&mut self.bitmap, &view);
        result
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
        let region = PixelRect::of_clip(&self.gstate.current().clip, self.width, self.height)
            .and_then(|clip| clip.intersect(rect));
        let Some(region) = region else {
            under(self);
            return;
        };
        let view = MaskView {
            values: coverage,
            width: rect.width() as usize,
            x0: rect.x0,
            y0: rect.y0,
        };
        let backdrop = Snapshot::take(&self.bitmap, region);
        under(self);
        let painted = Snapshot::take(&self.bitmap, region);
        backdrop.restore(&mut self.bitmap);
        over(self);
        painted.blend_back(&mut self.bitmap, &view);
    }
}

/// Mask values over a rectangle of the bitmap whose top-left pixel is
/// `(x0, y0)`, `width` values per row.
struct MaskView<'a> {
    values: &'a [u8],
    width: usize,
    x0: u32,
    y0: u32,
}

/// The pixels of a rectangle of the bitmap, captured before an object is
/// painted.
pub(super) struct Snapshot {
    pub(super) x0: u32,
    pub(super) y0: u32,
    pub(super) x1: u32,
    /// Colour bytes, three per pixel, row-major over the rectangle.
    pub(super) color: Vec<u8>,
    /// Alpha per pixel, or `None` for a bitmap without an alpha plane.
    pub(super) alpha: Option<Vec<u8>>,
}

impl Snapshot {
    pub(super) fn take(bitmap: &Bitmap<Rgb8>, region: PixelRect) -> Self {
        let PixelRect { x0, y0, x1, y1 } = region;
        let (cx0, cx1) = (x0 as usize * 3, x1 as usize * 3);
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
            color,
            alpha,
        }
    }

    /// Write the snapshot back over its rectangle.
    fn restore(self, bitmap: &mut Bitmap<Rgb8>) {
        let span = (self.x1 - self.x0) as usize;
        let (cx0, ax0) = (self.x0 as usize * 3, self.x0 as usize);
        for (i, prior) in self.color.chunks_exact(span * 3).enumerate() {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "i counts rows of the snapshot, which lie below the bitmap's u32 height"
            )]
            let y = self.y0 + i as u32;
            let (row, alpha_row) = bitmap.row_and_alpha_mut(y);
            row[cx0..cx0 + span * 3].copy_from_slice(prior);
            if let (Some(prior_a), Some(alpha_row)) = (self.alpha.as_ref(), alpha_row) {
                alpha_row[ax0..ax0 + span].copy_from_slice(&prior_a[i * span..(i + 1) * span]);
            }
        }
    }

    /// Replace each pixel of the rectangle by its interpolation between the
    /// snapshot and its current value, weighted by the mask.
    /// The snapshot lies within the mask's rectangle.
    fn blend_back(self, bitmap: &mut Bitmap<Rgb8>, mask: &MaskView<'_>) {
        let span = (self.x1 - self.x0) as usize;
        let (cx0, ax0) = (self.x0 as usize * 3, self.x0 as usize);
        let mx0 = (self.x0 - mask.x0) as usize;
        for (i, prior) in self.color.chunks_exact(span * 3).enumerate() {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "i counts rows of the snapshot, which lie below the bitmap's u32 height"
            )]
            let y = self.y0 + i as u32;
            let m_off = (y - mask.y0) as usize * mask.width + mx0;
            let m = &mask.values[m_off..m_off + span];
            let (row, alpha_row) = bitmap.row_and_alpha_mut(y);
            let row = &mut row[cx0..cx0 + span * 3];
            match (self.alpha.as_ref(), alpha_row) {
                (Some(prior_a), Some(alpha_row)) => blend_back_premultiplied(
                    prior,
                    &prior_a[i * span..(i + 1) * span],
                    row,
                    &mut alpha_row[ax0..ax0 + span],
                    m,
                ),
                _ => blend_back_opaque(prior, row, m),
            }
        }
    }
}

/// `painted ← prior + (painted − prior) × m / 255` per channel, for a
/// bitmap without alpha.
fn blend_back_opaque(prior: &[u8], painted: &mut [u8], mask: &[u8]) {
    for ((p, c), &m) in prior
        .as_chunks::<3>()
        .0
        .iter()
        .zip(painted.as_chunks_mut::<3>().0)
        .zip(mask)
    {
        if m == 255 {
            continue;
        }
        let m = u32::from(m);
        for (&b, a) in p.iter().zip(c.iter_mut()) {
            *a = lerp_u8(b, *a, m);
        }
    }
}

/// The interpolation of [`blend_back_opaque`] on alpha-premultiplied
/// colour, for a bitmap that stores straight colour beside an alpha plane.
fn blend_back_premultiplied(
    prior: &[u8],
    prior_a: &[u8],
    painted: &mut [u8],
    painted_a: &mut [u8],
    mask: &[u8],
) {
    for ((((p, &pa), c), ca), &m) in prior
        .as_chunks::<3>()
        .0
        .iter()
        .zip(prior_a)
        .zip(painted.as_chunks_mut::<3>().0)
        .zip(painted_a.iter_mut())
        .zip(mask)
    {
        if m == 255 {
            continue;
        }
        let m = u32::from(m);
        let (pa, a) = (u32::from(pa), u32::from(*ca));
        // Premultiplied numerators, scaled by 255 · 255.
        let wa = a * m;
        let wb = pa * (255 - m);
        let alpha_scaled = wa + wb;
        if alpha_scaled == 0 {
            p.iter().zip(c.iter_mut()).for_each(|(&b, x)| *x = b);
            *ca = 0;
            continue;
        }
        for (&b, x) in p.iter().zip(c.iter_mut()) {
            let num = u32::from(*x) * wa + u32::from(b) * wb;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "a weighted mean of u8 values is at most 255"
            )]
            let v = ((num + alpha_scaled / 2) / alpha_scaled) as u8;
            *x = v;
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "alpha_scaled ≤ 255 · 255, so the rounded quotient is at most 255"
        )]
        let out_a = ((alpha_scaled + 127) / 255) as u8;
        *ca = out_a;
    }
}

/// `prior + (painted − prior) × weight / 255`, rounded.
fn lerp_u8(prior: u8, painted: u8, weight: u32) -> u8 {
    let sum = u32::from(painted) * weight + u32::from(prior) * (255 - weight);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a weighted mean of u8 values is at most 255"
    )]
    let mean = ((sum + 127) / 255) as u8;
    mean
}

/// Backdrop colour `BC` as sRGB bytes; black when absent.
///
/// The component count selects the space: 1 grey, 3 RGB, 4 CMYK.
fn backdrop_rgb(bc: &[f64]) -> [u8; 3] {
    let c = match *bc {
        [g] => RasterColor::gray(g),
        [r, g, b] => RasterColor::rgb(r, g, b),
        [c, m, y, k] => RasterColor::cmyk(c, m, y, k),
        _ => RasterColor::gray(0.0),
    };
    let s = c.as_slice();
    [s[0], s[1], s[2]]
}

/// Luminosity mask values: each pixel of `group` composited over the opaque
/// backdrop `bc`, reduced to its luminosity by [`color::convert::luminosity`]
/// (PDF §11.3.5.3).
fn luminosity_values(group: &Bitmap<Rgb8>, bc: [u8; 3]) -> Vec<u8> {
    let (w, h) = (group.width as usize, group.height as usize);
    let mut out = Vec::with_capacity(w * h);
    for y in 0..group.height {
        let row = group.row_bytes(y);
        let alpha = group.alpha_row(y);
        for x in 0..w {
            let a = alpha.map_or(255, |r| u32::from(r[x]));
            let px = &row[x * 3..x * 3 + 3];
            #[expect(
                clippy::cast_possible_truncation,
                reason = "a weighted mean of two u8 values is at most 255"
            )]
            let ch = |i: usize| {
                ((u32::from(px[i]) * a + u32::from(bc[i]) * (255 - a) + 127) / 255) as u8
            };
            out.push(color::convert::luminosity(ch(0), ch(1), ch(2)));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opaque_blend_back_scales_the_change_by_the_mask() {
        let prior = [200, 100, 0, 200, 100, 0, 200, 100, 0];
        let mut painted = [0, 0, 255, 0, 0, 255, 0, 0, 255];
        blend_back_opaque(&prior, &mut painted, &[0, 255, 128]);
        assert_eq!(painted, [200, 100, 0, 0, 0, 255, 100, 50, 128]);
    }

    #[test]
    fn premultiplied_blend_back_matches_painting_at_reduced_alpha() {
        // Transparent backdrop; black painted at full alpha; mask 25 %.
        // Painting black at alpha 64 gives black at alpha 64.
        let mut c = [0, 0, 0];
        let mut a = [255];
        blend_back_premultiplied(&[255, 255, 255], &[0], &mut c, &mut a, &[64]);
        assert_eq!((c, a), ([0, 0, 0], [64]));

        // Opaque white backdrop; black painted; mask 50 %: mid-grey, opaque.
        let mut c = [0, 0, 0];
        let mut a = [255];
        blend_back_premultiplied(&[255, 255, 255], &[255], &mut c, &mut a, &[128]);
        assert_eq!((c, a), ([127, 127, 127], [255]));

        // Mask zero restores the prior pixel, including a transparent one.
        let mut c = [9, 9, 9];
        let mut a = [255];
        blend_back_premultiplied(&[1, 2, 3], &[0], &mut c, &mut a, &[0]);
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
        // White, pure red, and a transparent pixel showing the backdrop.
        assert_eq!(luminosity_values(&bmp, [0, 0, 0]), [255, 77, 0]);
        assert_eq!(luminosity_values(&bmp, [255, 255, 255]), [255, 77, 255]);
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
        let white = renderer.mask_backdrop_rgb((5, 0), &[100.0, 0.0, 0.0]);
        assert!(white.iter().all(|&v| v >= 250), "{white:?}");
        // A count that does not match the group's space falls back to it.
        assert_eq!(renderer.mask_backdrop_rgb((5, 0), &[1.0]), [255, 255, 255]);
        assert_eq!(
            renderer.mask_backdrop_rgb((9, 0), &[1.0, 0.0, 0.0]),
            [255, 0, 0]
        );
    }

    #[test]
    fn backdrop_component_count_selects_the_space() {
        assert_eq!(backdrop_rgb(&[]), [0, 0, 0]);
        assert_eq!(backdrop_rgb(&[1.0]), [255, 255, 255]);
        assert_eq!(backdrop_rgb(&[1.0, 0.0, 0.0]), [255, 0, 0]);
    }
}
