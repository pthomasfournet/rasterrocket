//! Knockout groups (ISO 32000-2 §11.4.6.2): each object of the group
//! composites with the group's initial backdrop, and within its shape
//! replaces what the objects before it painted.
//!
//! A group bitmap starts transparent, so in the group's own terms the initial
//! backdrop is the transparent pixel; a blend mode inside a non-isolated group
//! consults the backdrop under the group through the pipe, which does not
//! change over the group's life.  An object is therefore painted onto its
//! region cleared to transparent, and the result is merged with the group as
//! it was before.  With shape `f` and opacity `α` of the object and `α_g` the
//! group's alpha before it, the group's alpha after it is
//! `(1 − f)·α_g + f·α`, and its colour the same interpolation on
//! alpha-premultiplied colour.  Coverage folds into the painted alpha, which
//! is `f·α`, so the merge is `painted + (1 − f)·prior` on premultiplied values,
//! with `f` recovered as the painted alpha over the object's constant alpha.
//! A soft mask, the unpainted cells of a pattern and the mask of an image
//! thereby shape the object.

use super::PageRenderer;
use super::pixel_rect::PixelRect;
use super::soft_mask::Snapshot;
use color::Rgb8;
use raster::Bitmap;

impl PageRenderer<'_> {
    /// Run `paint`, the painting of one object of constant alpha `alpha`
    /// within `region`, knocking out the group's earlier content within the
    /// object's shape when the group being painted into is a knockout group.
    /// Outside one, or on a bitmap without an alpha plane, `paint` runs as
    /// it is.
    pub(super) fn paint_knocking_out<R>(
        &mut self,
        region: Option<PixelRect>,
        alpha: u8,
        paint: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let region = region
            .filter(|_| self.group.knockout && alpha > 0 && self.bitmap.has_alpha())
            .and_then(|r| self.within_clip(r));
        let Some(region) = region else {
            return paint(self);
        };
        let prior = Snapshot::take(&self.bitmap, region);
        clear(&mut self.bitmap, region);
        let result = paint(self);
        prior.merge_under(&mut self.bitmap, alpha);
        result
    }
}

/// Make the pixels of `region` transparent.
fn clear(bitmap: &mut Bitmap<Rgb8>, region: PixelRect) {
    let PixelRect { x0, y0, x1, y1 } = region;
    for y in y0..y1 {
        let (row, alpha) = bitmap.row_and_alpha_mut(y);
        row[x0 as usize * 3..x1 as usize * 3].fill(0);
        if let Some(alpha) = alpha {
            alpha[x0 as usize..x1 as usize].fill(0);
        }
    }
}

impl Snapshot {
    /// Merge the snapshot, the group before an object of constant alpha
    /// `alpha`, under the object as painted onto the cleared region: each
    /// pixel becomes `painted + (1 − f)·prior` on premultiplied colour and
    /// alpha, `f` being the painted alpha over `alpha`.
    fn merge_under(self, bitmap: &mut Bitmap<Rgb8>, alpha: u8) {
        let Some(prior_a) = self.alpha.as_ref() else {
            return;
        };
        let span = (self.x1 - self.x0) as usize;
        let (cx0, ax0) = (self.x0 as usize * 3, self.x0 as usize);
        let alpha = u32::from(alpha);
        for (i, prior) in self.color.chunks_exact(span * 3).enumerate() {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "i counts rows of the snapshot, which lie below the bitmap's u32 height"
            )]
            let y = self.y0 + i as u32;
            let (row, Some(alpha_row)) = bitmap.row_and_alpha_mut(y) else {
                return;
            };
            let painted = &mut row[cx0..cx0 + span * 3];
            let painted_a = &mut alpha_row[ax0..ax0 + span];
            for (((p, &pa), c), ca) in prior
                .as_chunks::<3>()
                .0
                .iter()
                .zip(&prior_a[i * span..(i + 1) * span])
                .zip(painted.as_chunks_mut::<3>().0)
                .zip(painted_a.iter_mut())
            {
                if pa == 0 {
                    continue;
                }
                let a = u32::from(*ca);
                // The object's shape, from its painted alpha `f·alpha`.
                let shape = ((a * 255 + alpha / 2) / alpha).min(255);
                if shape == 255 {
                    continue;
                }
                let keep = (255 - shape) * u32::from(pa);
                // Premultiplied numerators, scaled by 255 · 255.
                let out_a = a * 255 + keep;
                for (&b, x) in p.iter().zip(c.iter_mut()) {
                    let num = u32::from(*x) * a * 255 + u32::from(b) * keep;
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "a weighted mean of u8 values is at most 255"
                    )]
                    let v = ((num + out_a / 2) / out_a) as u8;
                    *x = v;
                }
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "out_a ≤ 255 · 255, so the rounded quotient is at most 255"
                )]
                let merged = ((out_a + 127) / 255) as u8;
                *ca = merged;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2 × 1 bitmap with alpha: pixel 0 holds `prior`, then the region is
    /// cleared and `painted` written; the merge is checked against the
    /// knockout formula.
    fn merge(prior: ([u8; 3], u8), painted: ([u8; 3], u8), alpha: u8) -> ([u8; 3], u8) {
        let mut bmp = Bitmap::<Rgb8>::new(1, 1, 1, true);
        bmp.row_bytes_mut(0).copy_from_slice(&prior.0);
        bmp.alpha_plane_mut().unwrap()[0] = prior.1;
        let region = PixelRect::new(0, 0, 1, 1).unwrap();
        let snapshot = Snapshot::take(&bmp, region);
        clear(&mut bmp, region);
        bmp.row_bytes_mut(0).copy_from_slice(&painted.0);
        bmp.alpha_plane_mut().unwrap()[0] = painted.1;
        snapshot.merge_under(&mut bmp, alpha);
        let c = bmp.row_bytes(0);
        ([c[0], c[1], c[2]], bmp.alpha_plane().unwrap()[0])
    }

    #[test]
    fn a_fully_shaped_object_replaces_the_prior_pixel() {
        assert_eq!(
            merge(([0, 0, 0], 255), ([255, 255, 255], 128), 128),
            ([255, 255, 255], 128),
            "half-alpha white at full shape over opaque black"
        );
    }

    #[test]
    fn an_unpainted_pixel_keeps_the_prior_pixel() {
        assert_eq!(
            merge(([10, 20, 30], 200), ([0, 0, 0], 0), 255),
            ([10, 20, 30], 200)
        );
    }

    #[test]
    fn a_half_shaped_object_interpolates_with_the_prior_pixel() {
        // Opaque white at half coverage over opaque black: mid-grey, opaque.
        assert_eq!(
            merge(([0, 0, 0], 255), ([255, 255, 255], 128), 255),
            ([128, 128, 128], 255)
        );
        // Half-alpha white at half coverage (painted alpha 64) over opaque
        // black: ½ black + ½ (white at α ½) = premultiplied 64, alpha ¾.
        let (c, a) = merge(([0, 0, 0], 255), ([255, 255, 255], 64), 128);
        assert_eq!(a, 191);
        assert!(c.iter().all(|&v| (84..=86).contains(&v)), "{c:?}");
    }
}
