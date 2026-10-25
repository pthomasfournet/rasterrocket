//! Area-averaging reduction of an image drawn much smaller than its
//! resolution.
//!
//! The blit samples one source pixel per device pixel (see
//! [`super::image_sampler`]).  When an image is drawn at less than half its
//! resolution, that point sampling skips most source pixels and features
//! narrower than the sampling step — thin strokes of lettering, fine rules —
//! break up or vanish, or, for a bilevel stencil, land as whole dark pixels.
//! Reducing the image first by the integer factor `floor(source / drawn)` on
//! each axis, each output pixel the mean of its `fx × fy` source block,
//! leaves the blit at most a 2:1 step, so every source pixel contributes to
//! the page in proportion to its area.

use crate::resources::SoftMask;
use crate::resources::image::{ImageData, ImageDescriptor, ImageFilter};

/// Reduce `img` for drawing at `draw` device pixels (its extent along the
/// image's own x and y axes).
///
/// Returns `None` — draw `img` as is — when neither axis is reduced by a
/// factor of at least 2 or when the samples are not host-resident.  A
/// stencil mask reduces like a grey image: its samples are 0 (paint) and
/// 255 (transparent), so each block mean is 255 minus the block's paint
/// coverage.  The result carries no mask: the caller reduces the original's
/// mask separately with [`reduce_mask_for_draw`], on the mask's own grid.
pub(super) fn reduce_for_draw(img: &ImageDescriptor, draw: (u32, u32)) -> Option<ImageDescriptor> {
    let bpp = img.color_space.bytes_per_pixel();
    let (fx, fy) = factors(img.width, img.height, draw)?;
    let src = img.data.as_cpu()?;
    let (width, height, out) = block_mean(src, img.width, img.height, bpp, fx, fy)?;
    Some(ImageDescriptor {
        width,
        height,
        color_space: img.color_space,
        data: ImageData::Cpu(out),
        smask: None,
        filter: ImageFilter::Raw,
    })
}

/// Reduce `mask` for drawing at `draw` device pixels, as
/// [`reduce_for_draw`] does an image: each reduced value is the mean alpha
/// of its block, so a bilevel mask becomes the fraction of each block it
/// admits.
pub(super) fn reduce_mask_for_draw(mask: &SoftMask, draw: (u32, u32)) -> Option<SoftMask> {
    let (fx, fy) = factors(mask.width(), mask.height(), draw)?;
    let (width, height, out) = block_mean(mask.alpha(), mask.width(), mask.height(), 1, fx, fy)?;
    SoftMask::new(width, height, out)
}

/// The integer reduction factors `floor(source / drawn)` per axis, or
/// `None` when neither reaches 2.
fn factors(width: u32, height: u32, draw: (u32, u32)) -> Option<(u32, u32)> {
    let fx = (width / draw.0.max(1)).max(1);
    let fy = (height / draw.1.max(1)).max(1);
    (fx >= 2 || fy >= 2).then_some((fx, fy))
}

/// Replace each `fx × fy` block of a `width × height` image of `bpp`-byte
/// pixels by its rounded per-channel mean; partial blocks at the right and
/// bottom edges average the pixels they hold.
///
/// Returns the reduced width, height and samples, or `None` when `src` does
/// not hold exactly `width × height × bpp` bytes.
fn block_mean(
    src: &[u8],
    width: u32,
    height: u32,
    bpp: usize,
    fx: u32,
    fy: u32,
) -> Option<(u32, u32, Vec<u8>)> {
    let (w, h) = (width as usize, height as usize);
    if src.len() != w.checked_mul(h)?.checked_mul(bpp)? {
        return None;
    }
    let (fx, fy) = (fx as usize, fy as usize);
    let out_w = w.div_ceil(fx);
    let out_h = h.div_ceil(fy);

    let mut out = Vec::with_capacity(out_w.checked_mul(out_h)?.checked_mul(bpp)?);
    let mut column_sums = vec![0u32; w * bpp];
    for y0 in (0..h).step_by(fy) {
        let rows = fy.min(h - y0);
        column_sums.fill(0);
        for row in src.chunks_exact(w * bpp).skip(y0).take(rows) {
            for (sum, &v) in column_sums.iter_mut().zip(row) {
                *sum += u32::from(v);
            }
        }
        for x0 in (0..w).step_by(fx) {
            let cols = fx.min(w - x0);
            #[expect(
                clippy::cast_possible_truncation,
                reason = "a block holds at most 65 536² pixels; far below u32::MAX"
            )]
            let count = (cols * rows) as u32;
            for ch in 0..bpp {
                let sum: u32 = (x0..x0 + cols).map(|x| column_sums[x * bpp + ch]).sum();
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "the rounded mean of u8 samples is at most 255"
                )]
                out.push(((sum + count / 2) / count) as u8);
            }
        }
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "out_w ≤ width and out_h ≤ height, which are u32"
    )]
    Some((out_w as u32, out_h as u32, out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::image::ImageColorSpace;

    fn gray(width: u32, height: u32, data: Vec<u8>) -> ImageDescriptor {
        ImageDescriptor {
            width,
            height,
            color_space: ImageColorSpace::Gray,
            data: ImageData::Cpu(data),
            smask: None,
            filter: ImageFilter::Raw,
        }
    }

    /// A one-pixel-wide black stroke in every fourth column survives a 4:1
    /// reduction as a uniform quarter-gray, where point sampling at that
    /// step lands on the stroke either always or never.
    #[test]
    fn thin_stroke_contributes_to_every_block() {
        let (w, h) = (16u32, 4u32);
        let data = (0..h)
            .flat_map(|_| (0..w).map(|x| if x % 4 == 1 { 0 } else { 255 }))
            .collect();
        let out = reduce_for_draw(&gray(w, h, data), (4, 1)).expect("reduced");
        assert_eq!((out.width, out.height), (4, 1));
        assert_eq!(out.data.as_cpu().unwrap(), &[191, 191, 191, 191]);
    }

    #[test]
    fn rgb_blocks_average_per_channel_with_partial_edges() {
        // 5×1 RGB drawn 2 px wide: factor 2, blocks {0, 1}, {2, 3} and the
        // partial {4}.
        let img = ImageDescriptor {
            color_space: ImageColorSpace::Rgb,
            ..gray(
                5,
                1,
                vec![255, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 10, 20, 30],
            )
        };
        let out = reduce_for_draw(&img, (2, 1)).expect("reduced");
        assert_eq!((out.width, out.height), (3, 1));
        assert_eq!(
            out.data.as_cpu().unwrap(),
            &[128, 0, 128, 0, 0, 0, 10, 20, 30]
        );
    }

    #[test]
    fn no_reduction_below_two_to_one() {
        assert!(reduce_for_draw(&gray(10, 10, vec![0; 100]), (6, 6)).is_none());
        let mask = SoftMask::new(10, 10, vec![0; 100]).unwrap();
        assert!(reduce_mask_for_draw(&mask, (6, 6)).is_none());
    }

    /// A stencil's reduced sample is 255 minus the paint coverage of its
    /// block: one painted sample in four leaves 191.
    #[test]
    fn stencil_blocks_reduce_to_their_paint_coverage() {
        let stencil = ImageDescriptor {
            color_space: ImageColorSpace::Mask,
            ..gray(4, 2, vec![0, 255, 0, 0, 255, 255, 0, 0])
        };
        let out = reduce_for_draw(&stencil, (2, 1)).expect("reduced");
        assert_eq!(out.color_space, ImageColorSpace::Mask);
        assert_eq!((out.width, out.height), (2, 1));
        assert_eq!(out.data.as_cpu().unwrap(), &[191, 0]);
    }

    /// A bilevel mask reduces to the fraction of each block it admits.
    #[test]
    fn mask_blocks_reduce_to_their_mean_alpha() {
        let mask = SoftMask::new(6, 3, [[255, 0, 0, 255, 255, 255]; 3].concat()).unwrap();
        let out = reduce_mask_for_draw(&mask, (2, 1)).expect("reduced");
        assert_eq!((out.width(), out.height()), (2, 1));
        assert_eq!(out.alpha(), &[85, 255]);
    }
}
