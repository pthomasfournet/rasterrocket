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
//! the paint path (fill, stroke, glyph, image, shading, group).

use std::sync::Arc;

use color::Rgb8;
use raster::Bitmap;
use raster::types::BlendMode;

use super::PageRenderer;
use crate::renderer::color::RasterColor;
use crate::renderer::gstate::SoftMaskPlane;
use crate::resources::SoftMaskParams;
use crate::resources::shading::function::eval_function;

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
        // 1 and the blend mode is `Normal`.
        self.gstate.save();
        {
            let gs = self.gstate.current_mut();
            gs.soft_mask = None;
            gs.fill_alpha = 255;
            gs.stroke_alpha = 255;
            gs.blend_mode = BlendMode::Normal;
        }
        let mut canvas = Bitmap::<Rgb8>::new(self.width, self.height, 1, true);
        std::mem::swap(&mut self.bitmap, &mut canvas);
        self.do_form_xobject(&form);
        std::mem::swap(&mut self.bitmap, &mut canvas);
        self.gstate.restore();

        let lut = transfer.and_then(|f| transfer_lut(self.resources.doc(), f));
        let values = if luminosity {
            let bc = backdrop_rgb(backdrop);
            luminosity_values(&canvas, bc)
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

    /// Run `paint` — one painting operator — under `mask`.
    ///
    /// The soft mask is removed from the graphics state while `paint` runs,
    /// so that operators nested inside it (a form's content, a Type 3
    /// glyph) are not masked a second time.
    pub(super) fn paint_through_soft_mask<R>(
        &mut self,
        mask: &SoftMaskPlane,
        paint: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let region = (mask.width == self.width && mask.height == self.height)
            .then(|| clip_region(&self.gstate.current().clip, self.width, self.height))
            .flatten();
        let Some((x0, y0, x1, y1)) = region else {
            return paint(self);
        };
        let before = Snapshot::take(&self.bitmap, x0, y0, x1, y1);
        let result = paint(self);
        before.blend_back(&mut self.bitmap, mask);
        result
    }
}

/// The pixel rectangle `(x0, y0, x1, y1)`, half-open, that the clip admits
/// within a `width × height` bitmap; `None` when it is empty.
fn clip_region(clip: &raster::Clip, width: u32, height: u32) -> Option<(u32, u32, u32, u32)> {
    let bound = |v: i64, len: u32| u32::try_from(v.clamp(0, i64::from(len))).ok();
    let x0 = bound(i64::from(clip.x_min_i), width)?;
    let y0 = bound(i64::from(clip.y_min_i), height)?;
    let x1 = bound(i64::from(clip.x_max_i) + 1, width)?;
    let y1 = bound(i64::from(clip.y_max_i) + 1, height)?;
    (x0 < x1 && y0 < y1).then_some((x0, y0, x1, y1))
}

/// The pixels of a rectangle of the bitmap, captured before an object is
/// painted.
struct Snapshot {
    x0: u32,
    y0: u32,
    x1: u32,
    color: Vec<u8>,
    alpha: Option<Vec<u8>>,
}

impl Snapshot {
    fn take(bitmap: &Bitmap<Rgb8>, x0: u32, y0: u32, x1: u32, y1: u32) -> Self {
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

    /// Replace each pixel of the rectangle by its interpolation between the
    /// snapshot and its current value, weighted by the mask.
    fn blend_back(self, bitmap: &mut Bitmap<Rgb8>, mask: &SoftMaskPlane) {
        let span = (self.x1 - self.x0) as usize;
        let (cx0, ax0) = (self.x0 as usize * 3, self.x0 as usize);
        let width = mask.width as usize;
        for (i, prior) in self.color.chunks_exact(span * 3).enumerate() {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "i counts rows of the snapshot, which lie below the bitmap's u32 height"
            )]
            let y = self.y0 + i as u32;
            let m_off = y as usize * width + ax0;
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
/// backdrop `bc`, reduced to `0.30 R + 0.59 G + 0.11 B` (PDF §11.3.5.3).
fn luminosity_values(group: &Bitmap<Rgb8>, bc: [u8; 3]) -> Vec<u8> {
    let (w, h) = (group.width as usize, group.height as usize);
    let mut out = Vec::with_capacity(w * h);
    for y in 0..group.height {
        let row = group.row_bytes(y);
        let alpha = group.alpha_row(y);
        for x in 0..w {
            let a = alpha.map_or(255, |r| u32::from(r[x]));
            let px = &row[x * 3..x * 3 + 3];
            let ch = |i: usize| (u32::from(px[i]) * a + u32::from(bc[i]) * (255 - a) + 127) / 255;
            let lum = (30 * ch(0) + 59 * ch(1) + 11 * ch(2) + 50) / 100;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "a weighted mean of u8 values is at most 255"
            )]
            out.push(lum as u8);
        }
    }
    out
}

/// Tabulate a transfer function over the 256 mask values.
fn transfer_lut(doc: &pdf::Document, f: &pdf::Object) -> Option<[u8; 256]> {
    let mut lut = [0u8; 256];
    for (i, v) in lut.iter_mut().enumerate() {
        let t = f64::from(u8::try_from(i).ok()?) / 255.0;
        let y = eval_function(doc, f, t, 1)?.first().copied()?;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "clamped to [0, 1] and scaled to [0, 255]"
        )]
        let b = (y.clamp(0.0, 1.0) * 255.0).round() as u8;
        *v = b;
    }
    Some(lut)
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

    #[test]
    fn backdrop_component_count_selects_the_space() {
        assert_eq!(backdrop_rgb(&[]), [0, 0, 0]);
        assert_eq!(backdrop_rgb(&[1.0]), [255, 255, 255]);
        assert_eq!(backdrop_rgb(&[1.0, 0.0, 0.0]), [255, 0, 0]);
    }
}
