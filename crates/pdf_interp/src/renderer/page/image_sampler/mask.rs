//! Soft-mask alpha under a sampled image.
//!
//! The mask is resolved at device resolution: each device pixel reads the
//! mask around its own centre, through the image's taps when the mask shares
//! the image's grid and filter, and through a grid of its own otherwise.

use color::{Sample, SampleDepth};

use super::filter::Filter;
use super::grid::{GridView, SampleGrid};
use super::taps::{Taps, sample_bytes};
use crate::renderer::gstate::Ctm;
use crate::resources::SoftMask;

/// Alpha lookup for a masked image blit.
///
/// A mask on the image's own grid — the same dimensions drawn through the
/// same matrix — is read through the image's own taps.  Any other mask is
/// sampled through a [`SampleGrid`] of its own dimensions and matrix on the
/// same page, with the filter [`Filter::for_draw`] selects for it: each
/// device pixel reads the mask around its centre, so the mask is resolved
/// at device resolution whatever resolution the image was decoded or
/// resampled at.  Where the image admits a pixel centre that rounding
/// places just off the mask, the nearest mask pixel is used.
pub(in crate::renderer::page) struct MaskSampler<'a> {
    pub(in crate::renderer::page) alpha: &'a [u8],
    /// The mask's 16-bit alpha, when it was decoded at 16 bits.
    pub(in crate::renderer::page) alpha16: Option<&'a [u16]>,
    width: usize,
    /// `None` when the mask shares the image's grid.
    pub(in crate::renderer::page) grid: Option<SampleGrid>,
}

impl<'a> MaskSampler<'a> {
    /// Sampler for `mask` drawn through `mask_ctm`, over an image of
    /// `img_w × img_h` pixels drawn through `img_ctm` with `img_filter`
    /// onto `view`.  The mask is interpolated as its
    /// own `/Interpolate` asks; it takes the image's taps when it is placed
    /// as the image and filtered alike.
    ///
    /// Returns `None` only where [`SampleGrid::new`] does for the mask: a
    /// degenerate matrix or a bounding box off the page.
    #[expect(
        clippy::float_cmp,
        reason = "the mask shares the image grid only when both are drawn through the identical matrix"
    )]
    pub(in crate::renderer::page) fn new(
        mask: &'a SoftMask,
        mask_ctm: &Ctm,
        img_ctm: &Ctm,
        img_filter: Filter,
        view: GridView,
        img_w: u32,
        img_h: u32,
    ) -> Option<Self> {
        let (w, h) = (mask.width(), mask.height());
        let filter = Filter::for_draw(mask_ctm, w, h, mask.interpolate());
        let placed_as_image = w == img_w && h == img_h && mask_ctm == img_ctm;
        let grid = if placed_as_image && filter == img_filter {
            None
        } else {
            Some(SampleGrid::new(mask_ctm, view, w, h, filter)?)
        };
        Some(Self {
            alpha: mask.alpha(),
            alpha16: mask.alpha16(),
            width: mask.width() as usize,
            grid,
        })
    }

    /// Alpha of device pixel `(dx, dy)`, whose image taps are `image`, as
    /// a sample of type `S`: from the 16-bit alpha of a mask that has one
    /// for a 16-bit sample, else the 8-bit alpha on the sample's scale.
    ///
    /// `dx`/`dy` must lie inside the bounding box.
    #[inline]
    pub(in crate::renderer::page) fn alpha_at<S: Sample>(
        &self,
        dx: u32,
        dy: u32,
        image: &Taps,
    ) -> S {
        let taps = self
            .grid
            .as_ref()
            .map_or(*image, |g| g.sample_clamped(dx, dy));
        if let Some(alpha16) = self.alpha16.filter(|_| S::DEPTH == SampleDepth::Sixteen) {
            let [alpha] =
                taps.blend_px(|x, y| sample_bytes::<u16, 1>(alpha16, self.width, 1, x, y));
            return S::from_sample(alpha);
        }
        let [alpha] = taps.blend_px(|x, y| sample_bytes::<u8, 1>(self.alpha, self.width, 1, x, y));
        S::from_u8(alpha)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mask at four times the image's resolution is resolved per device
    /// pixel: under a 1:1 device mapping of the mask, each device pixel reads
    /// its own mask pixel, while the image repeats each sample four times.
    #[test]
    fn mask_sampler_resolves_a_finer_mask_per_device_pixel() {
        let ctm = [8.0, 0.0, 0.0, 8.0, 0.0, 0.0];
        let (img_w, img_h) = (2, 2);
        let alpha: Vec<u8> = (0..64).collect();
        let mask = SoftMask::new(8, 8, alpha).expect("8×8 mask");
        let grid = SampleGrid::new(&ctm, GridView::whole(8, 8), img_w, img_h, Filter::NEAREST)
            .expect("grid");
        let sampler = MaskSampler::new(
            &mask,
            &ctm,
            &ctm,
            Filter::NEAREST,
            GridView::whole(8, 8),
            img_w,
            img_h,
        )
        .expect("sampler");
        for dy in grid.y0..grid.y1 {
            for dx in grid.x0..grid.x1 {
                let px = grid.sample(dx, dy).expect("inside");
                assert_eq!(px, Taps::point(dx / 4, dy / 4));
                let want = u8::try_from(dy * 8 + dx).unwrap();
                assert_eq!(
                    sampler.alpha_at::<u8>(dx, dy, &px),
                    want,
                    "pixel ({dx}, {dy})"
                );
            }
        }
    }

    /// A mask on the image's grid is indexed by the image sample itself.
    #[test]
    fn mask_sampler_on_the_image_grid_reuses_the_image_sample() {
        let ctm = [40.0, 0.0, 0.0, 30.0, 10.0, 20.0];
        let mask = SoftMask::new(
            40,
            30,
            (0..1200u32)
                .map(|i| u8::try_from(i % 251).unwrap())
                .collect(),
        )
        .expect("40×30 mask");
        let filter = Filter::for_draw(&ctm, 40, 30, false);
        let sampler =
            MaskSampler::new(&mask, &ctm, &ctm, filter, GridView::whole(100, 100), 40, 30)
                .expect("sampler");
        assert!(sampler.grid.is_none());
        assert_eq!(
            sampler.alpha_at::<u8>(0, 0, &Taps::point(3, 2)),
            mask.alpha()[2 * 40 + 3]
        );
    }

    /// A 16-bit sample of a mask with a 16-bit alpha is that alpha; of a
    /// mask without one, its 8-bit alpha widened; an 8-bit sample is the
    /// 8-bit alpha either way.
    #[test]
    fn mask_sampler_reads_the_alpha_of_the_sample_depth() {
        let ctm = [40.0, 0.0, 0.0, 30.0, 10.0, 20.0];
        let filter = Filter::for_draw(&ctm, 40, 30, false);
        let wide: Vec<u16> = (0..1200u32)
            .map(|i| u16::try_from(i * 53 % 65_536).unwrap())
            .collect();
        let deep = SoftMask::new16(40, 30, wide.clone()).expect("40×30 mask");
        let narrow = SoftMask::new(40, 30, deep.alpha().to_vec()).expect("40×30 mask");
        let at = Taps::point(3, 2);
        let i = 2 * 40 + 3;
        let sampler = |mask| {
            MaskSampler::new(mask, &ctm, &ctm, filter, GridView::whole(100, 100), 40, 30)
                .expect("sampler")
        };
        assert_eq!(sampler(&deep).alpha_at::<u16>(0, 0, &at), wide[i]);
        assert_eq!(sampler(&deep).alpha_at::<u8>(0, 0, &at), deep.alpha()[i]);
        assert_eq!(
            sampler(&narrow).alpha_at::<u16>(0, 0, &at),
            u16::from_u8(narrow.alpha()[i])
        );
    }
}
