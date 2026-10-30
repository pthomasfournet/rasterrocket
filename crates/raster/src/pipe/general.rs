//! General pipe: soft mask, non-normal blend modes, non-isolated/knockout groups.
//!
//! This is the "everything else" path, matching `Splash::pipeRun` (the general
//! case, not one of the specialised `pipeRunSimple*` or `pipeRunAA*` variants).
//!
//! # Compositing formula (PDF spec §11.3)
//!
//! Given source alpha `a_src`, destination alpha `a_dst`, and source/dest colours:
//!
//! ```text
//! a_result = a_src + a_dst - div255(a_src * a_dst)           (isolated, non-knockout)
//! c_result = ((a_result - a_src) * c_dst + a_src * blend(c_src, c_dst)) / a_result
//! ```
//!
//! For blend mode `Normal`, `blend(c_src, c_dst) = c_src` so the formula reduces
//! to the standard Porter-Duff over.
//!
//! Inside a non-isolated group (`pipe.backdrop` set) the destination holds
//! the group's own colour and alpha, and the blend function sees the group's
//! content over the backdrop (ISO 32000-2 §11.4.7, with the backdrop
//! contribution removed from the stored result):
//!
//! ```text
//! a_union  = a_dst + a_0 - div255(a_dst * a_0)
//! c_union  = (a_dst * c_dst + (255 - a_dst) * a_0 * c_0 / 255) / a_union
//! c_result = ((a_result - a_src) * c_dst
//!             + a_src * ((255 - a_union) * c_src + a_union * blend(c_src, c_union)) / 255)
//!            / a_result
//! ```

use std::cell::RefCell;

use crate::pipe::{self, PipeSrc, PipeState, blend};
use crate::types::BlendMode;
use color::Pixel;
use color::convert::div255;

pub(crate) const MAX_COMPS: usize = 8; // DeviceN8: 4 CMYK + 4 spot = 8 bytes

// Per-thread scratch buffer for pattern spans — grow-never-shrink.
thread_local! {
    static PAT_BUF: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// General-purpose compositing span function.
///
/// Handles soft mask, blend modes, non-isolated groups, knockout, and overprint.
/// Slower than `simple` or `aa` but covers every case.
///
/// # Preconditions (checked with `debug_assert!`)
///
/// - `x1 >= x0`
/// - `dst_pixels.len() == (x1 - x0 + 1) * P::BYTES`
/// - If `shape` is `Some`, `shape.len() == x1 - x0 + 1`
/// - If `pipe.soft_mask` is `Some`, `soft_mask.len() == x1 - x0 + 1`
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors C++ SplashPipe API; all parameters are necessary"
)]
pub(crate) fn render_span_general<P: Pixel>(
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    dst_pixels: &mut [u8],
    dst_alpha: Option<&mut [u8]>,
    shape: Option<&[u8]>,
    x0: i32,
    x1: i32,
    y: i32,
) {
    debug_assert!(x1 >= x0, "render_span_general: x1={x1} < x0={x0}");
    #[expect(
        clippy::cast_sign_loss,
        reason = "x1 >= x0 is asserted above, so x1 - x0 + 1 >= 1 > 0"
    )]
    let count = (x1 - x0 + 1) as usize;
    let ncomps = P::BYTES;

    debug_assert_eq!(dst_pixels.len(), count * ncomps);
    if let Some(sh) = shape {
        debug_assert_eq!(sh.len(), count);
    }
    if let Some(sm) = pipe.soft_mask {
        debug_assert_eq!(sm.len(), count, "soft_mask length must equal span count");
    }

    let a_input = u32::from(pipe.a_input);
    let is_nonseparable = matches!(
        pipe.blend_mode,
        BlendMode::Hue | BlendMode::Saturation | BlendMode::Color | BlendMode::Luminosity
    );
    let is_cmyk_like = ncomps == 4 || ncomps == 8;

    // Convenience: get optional slice at index i.
    let shape_at = |i: usize| shape.map_or(0xFFu8, |s| s[i]);
    let soft_mask_at = |i: usize| pipe.soft_mask.map_or(0xFFu8, |s| s[i]);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "i < count = x1 − x0 + 1, which fits in i32"
    )]
    let backdrop_at = |i: usize| {
        pipe.backdrop.as_ref().map(|b| {
            b.pixel(x0 + i as i32, y, ncomps)
                .unwrap_or((&[0; MAX_COMPS][..ncomps], 0))
        })
    };

    match src {
        PipeSrc::Solid(color) => {
            debug_assert_eq!(color.len(), ncomps);
            render_span_general_inner(
                pipe,
                |_i| color,
                dst_pixels,
                dst_alpha,
                shape,
                count,
                ncomps,
                a_input,
                is_nonseparable,
                is_cmyk_like,
                &shape_at,
                &soft_mask_at,
                &backdrop_at,
            );
        }
        PipeSrc::Pattern(pat) => {
            PAT_BUF.with(|cell| {
                let mut buf = cell.borrow_mut();
                buf.resize(count * ncomps, 0);
                pat.fill_span(y, x0, x1, &mut buf[..count * ncomps]);
                render_span_general_inner(
                    pipe,
                    |i| &buf[i * ncomps..(i + 1) * ncomps],
                    dst_pixels,
                    dst_alpha,
                    shape,
                    count,
                    ncomps,
                    a_input,
                    is_nonseparable,
                    is_cmyk_like,
                    &shape_at,
                    &soft_mask_at,
                    &backdrop_at,
                );
            });
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "all params necessary; closure eliminates solid/pattern duplication"
)]
#[expect(
    clippy::too_many_lines,
    reason = "compositing formula has many branches that cannot be meaningfully split"
)]
#[expect(
    clippy::single_match_else,
    reason = "both Some and None arms have substantial independent logic; if-let would be less clear"
)]
fn render_span_general_inner<'src, 'bd>(
    pipe: &PipeState<'_>,
    src_px_at: impl Fn(usize) -> &'src [u8],
    dst_pixels: &mut [u8],
    dst_alpha: Option<&mut [u8]>,
    shape: Option<&[u8]>,
    count: usize,
    ncomps: usize,
    a_input: u32,
    is_nonseparable: bool,
    is_cmyk_like: bool,
    shape_at: &dyn Fn(usize) -> u8,
    soft_mask_at: &dyn Fn(usize) -> u8,
    backdrop_at: &dyn Fn(usize) -> Option<(&'bd [u8], u8)>,
) {
    let has_soft_mask = pipe.soft_mask.is_some();
    let has_shape = shape.is_some();

    match dst_alpha {
        Some(dst_alpha) => {
            debug_assert_eq!(dst_alpha.len(), count);
            for i in 0..count {
                let src_px = src_px_at(i);
                let dst_px = &mut dst_pixels[i * ncomps..(i + 1) * ncomps];
                let shape_v = u32::from(shape_at(i));
                let soft_v = u32::from(soft_mask_at(i));

                // Source alpha (PDF spec §11.3.6 eq 11.1).
                let a_src = compute_a_src(a_input, soft_v, shape_v, has_soft_mask, has_shape);

                // Knockout: clear destination alpha before the colour formula so that
                // a_dst_eff (read on line below) reflects the zeroed value.
                // This applies to both isolated and non-isolated knockout groups.
                if pipe.knockout && shape_v >= u32::from(pipe.knockout_opacity) {
                    dst_alpha[i] = 0;
                }

                let c_src: &[u8] = src_px;
                let a_dst_eff = u32::from(dst_alpha[i]); // may have been cleared by knockout.

                // The pixel the blend function sees and the alpha weighting
                // it: the destination, or in a non-isolated group the
                // group's content over the backdrop (its backdrop alone
                // under knockout).
                let mut c_union: [u8; MAX_COMPS] = [0; MAX_COMPS];
                let (c_blend_dst, alpha_im1): (&[u8], u32) = match backdrop_at(i) {
                    Some((c_0, a_0)) if pipe.knockout => (c_0, u32::from(a_0)),
                    Some((c_0, a_0)) => {
                        let a_union = union_colour(
                            dst_px,
                            a_dst_eff,
                            c_0,
                            u32::from(a_0),
                            &mut c_union[..ncomps],
                        );
                        (&c_union[..ncomps], a_union)
                    }
                    None => (&*dst_px, a_dst_eff),
                };

                // Blend function.
                let mut c_blend: [u8; MAX_COMPS] = [0; MAX_COMPS];
                if pipe.blend_mode != BlendMode::Normal {
                    apply_blend_fn(
                        pipe.blend_mode,
                        c_src,
                        c_blend_dst,
                        &mut c_blend[..ncomps],
                        is_cmyk_like,
                        is_nonseparable,
                    );
                }

                // Result alpha.
                let a_result = compute_a_result(a_src, a_dst_eff, shape_v, pipe.knockout);
                let alpha_i = a_result;

                // Result colour.
                if a_result == 0 {
                    dst_px.fill(0);
                } else {
                    // alpha_i >= a_result > 0: safe to divide.
                    debug_assert!(alpha_i > 0, "alpha_i must be > 0 when a_result > 0");
                    for j in 0..ncomps {
                        let c_src_j = u32::from(c_src[j]);
                        let c_dst_j = u32::from(dst_px[j]);
                        let c_b_j = u32::from(c_blend[j]);

                        let c = if pipe.blend_mode == BlendMode::Normal {
                            // No blend function: standard Porter-Duff.
                            ((alpha_i - a_src) * c_dst_j + a_src * c_src_j) / alpha_i
                        } else {
                            // With blend function.
                            ((alpha_i - a_src) * c_dst_j
                                + a_src * ((255 - alpha_im1) * c_src_j + alpha_im1 * c_b_j) / 255)
                                / alpha_i
                        };
                        #[expect(
                            clippy::cast_possible_truncation,
                            reason = "c is a weighted average of values ≤ 255, so c ≤ 255"
                        )]
                        {
                            dst_px[j] = c as u8;
                        }
                    }
                    finish_pixel(pipe, dst_px, src_px, ncomps);
                }

                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "a_result is clamped to ≤ 255 in compute_a_result"
                )]
                {
                    dst_alpha[i] = a_result as u8;
                }
            }
        }
        None => {
            // No separate alpha plane: aDest = 0xFF implicitly.
            // Simplifies: aResult = aSrc + 255 - div255(aSrc * 255) = 255, alpha_i = 255.
            // Non-isolated and knockout modes require a dst_alpha plane to carry the
            // group alpha; those states are meaningless without one.
            debug_assert!(
                pipe.backdrop.is_none() && !pipe.knockout,
                "a group backdrop and knockout require dst_alpha; None arm uses implicit a_dst=255"
            );
            for i in 0..count {
                let src_px = src_px_at(i);
                let dst_px = &mut dst_pixels[i * ncomps..(i + 1) * ncomps];
                let shape_v = u32::from(shape_at(i));
                let soft_v = u32::from(soft_mask_at(i));

                let a_src = compute_a_src(a_input, soft_v, shape_v, has_soft_mask, has_shape);
                // a_src is derived from u8 inputs via div255, so always ≤ 255.
                // The assert catches any future change that widens the input path.
                debug_assert!(a_src <= 255, "a_src={a_src} out of [0, 255]");

                let mut c_blend: [u8; MAX_COMPS] = [0; MAX_COMPS];
                if pipe.blend_mode != BlendMode::Normal {
                    apply_blend_fn(
                        pipe.blend_mode,
                        src_px,
                        dst_px,
                        &mut c_blend[..ncomps],
                        is_cmyk_like,
                        is_nonseparable,
                    );
                }

                for j in 0..ncomps {
                    let c_src_j = u32::from(src_px[j]);
                    let c_dst_j = u32::from(dst_px[j]);
                    let c_b_j = u32::from(c_blend[j]);

                    // With implicit a_dst=255: alpha_i=255, alpha_im1=255.
                    // General formula simplifies to div255((255-a_src)*c_dst + a_src*c_b).
                    // 255 - a_src: safe because a_src ≤ 255 (asserted above).
                    let c = if pipe.blend_mode == BlendMode::Normal {
                        u32::from(div255((255 - a_src) * c_dst_j + a_src * c_src_j))
                    } else {
                        u32::from(div255((255 - a_src) * c_dst_j + a_src * c_b_j))
                    };
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "c is a weighted average of values ≤ 255, so c ≤ 255"
                    )]
                    {
                        dst_px[j] = c as u8;
                    }
                }
                finish_pixel(pipe, dst_px, src_px, ncomps);
            }
        }
    }
}

/// Compute the effective source alpha for one pixel (PDF spec §11.3.6 eq 11.1).
///
/// Combines `a_input` with the soft mask and/or shape coverage according to the
/// rules: soft mask and shape are multiplied together via `div255`; if either is
/// absent its default is 1.0 (== 0xFF).
#[inline]
fn compute_a_src(
    a_input: u32,
    soft_v: u32,
    shape_v: u32,
    has_soft_mask: bool,
    has_shape: bool,
) -> u32 {
    if has_soft_mask {
        if has_shape {
            u32::from(div255(u32::from(div255(a_input * soft_v)) * shape_v))
        } else {
            u32::from(div255(a_input * soft_v))
        }
    } else if has_shape {
        u32::from(div255(a_input * shape_v))
    } else {
        a_input
    }
}

/// The result alpha of a source of alpha `a_src` and shape `shape` over a
/// destination of alpha `a_dst`: their union, or under knockout the source
/// over the destination outside the shape.
fn compute_a_result(a_src: u32, a_dst: u32, shape: u32, knockout: bool) -> u32 {
    let a_result = if knockout {
        a_src + u32::from(div255(a_dst * (255 - shape)))
    } else {
        a_src + a_dst - u32::from(div255(a_src * a_dst))
    };
    a_result.min(255)
}

/// Write into `out` the colour of a group's content (`c_dst`, alpha `a_dst`)
/// composited over its backdrop (`c_0`, alpha `a_0`) and return the alpha
/// of that union.  Black where the union is transparent.
pub(crate) fn union_colour(c_dst: &[u8], a_dst: u32, c_0: &[u8], a_0: u32, out: &mut [u8]) -> u32 {
    let a_union = (a_dst + a_0 - u32::from(div255(a_dst * a_0))).min(255);
    if a_union == 0 {
        out.fill(0);
        return 0;
    }
    for ((o, &c_d), &c_b) in out.iter_mut().zip(c_dst).zip(c_0) {
        let num = a_dst * u32::from(c_d) * 255 + (255 - a_dst) * a_0 * u32::from(c_b);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a weighted mean of u8 values is at most 255"
        )]
        {
            *o = (num / (255 * a_union)) as u8;
        }
    }
    a_union
}

/// Apply the blend function and write into `c_blend`.
fn apply_blend_fn(
    mode: BlendMode,
    src: &[u8],
    dst: &[u8],
    c_blend: &mut [u8],
    is_cmyk_like: bool,
    is_nonseparable: bool,
) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len(), c_blend.len());
    let ncomps = src.len();

    if is_cmyk_like {
        // Subtractive complement: invert all channels, blend in additive space, re-invert.
        // Fill all ncomps so spot channels (j >= 4) are correctly inverted.
        let mut src2 = [0u8; MAX_COMPS];
        let mut dst2 = [0u8; MAX_COMPS];
        for j in 0..ncomps {
            src2[j] = 255 - src[j];
            dst2[j] = 255 - dst[j];
        }

        if is_nonseparable {
            let s3 = [src2[0], src2[1], src2[2]];
            let d3 = [dst2[0], dst2[1], dst2[2]];
            let r3 = blend::apply_nonseparable_rgb(mode, s3, d3);
            c_blend[0] = 255 - r3[0];
            c_blend[1] = 255 - r3[1];
            c_blend[2] = 255 - r3[2];
            // K/spot channel: for Luminosity, use src K; for others use dst K (PDF §11.3.5).
            if ncomps >= 4 {
                c_blend[3] = 255
                    - (if mode == BlendMode::Luminosity {
                        src2[3]
                    } else {
                        dst2[3]
                    });
            }
            for j in 4..ncomps {
                // Spot channels pass through dst (same rule as K for non-Luminosity).
                c_blend[j] = 255 - dst2[j];
            }
        } else {
            blend::apply_separable(
                mode,
                &src2[..ncomps.min(4)],
                &dst2[..ncomps.min(4)],
                &mut c_blend[..ncomps.min(4)],
            );
            for v in &mut c_blend[..ncomps.min(4)] {
                *v = 255 - *v;
            }
            // Spot channels (j >= 4): not blended, pass dst through unchanged.
            c_blend[4..ncomps].copy_from_slice(&dst[4..ncomps]);
        }
    } else if is_nonseparable {
        // RGB/Gray additive space.
        let n = ncomps.min(3);
        let mut s3 = [0u8; 3];
        let mut d3 = [0u8; 3];
        s3[..n].copy_from_slice(&src[..n]);
        d3[..n].copy_from_slice(&dst[..n]);
        // Mono: replicate the single channel to all three.
        if ncomps == 1 {
            s3[1] = s3[0];
            s3[2] = s3[0];
            d3[1] = d3[0];
            d3[2] = d3[0];
        }
        let r3 = blend::apply_nonseparable_rgb(mode, s3, d3);
        c_blend[..n].copy_from_slice(&r3[..n]);
    } else {
        blend::apply_separable(mode, src, dst, c_blend);
    }
}

/// Apply transfer and conditional overprint at the end of each pixel's colour computation.
///
/// Called after the colour formula writes all channels of `dst_px` but before
/// the alpha plane is updated.  Order matters: transfer must precede overprint
/// so the transfer LUT sees the blended colour, not the restored source.
#[inline]
fn finish_pixel(pipe: &PipeState<'_>, dst_px: &mut [u8], src_px: &[u8], ncomps: usize) {
    pipe::apply_transfer_in_place(pipe, dst_px);
    if pipe.overprint_mask != 0xFFFF_FFFF {
        apply_overprint(pipe, dst_px, src_px, ncomps);
    }
}

/// Apply overprint: for channels where the bit in `overprint_mask` is 0,
/// the channel is not painted.
///
/// Only `overprint_additive = true` is implemented.  Replace-mode overprint
/// requires the caller to preserve the pre-blend destination bytes, which the
/// current call structure does not support.
///
/// # Panics
///
/// Panics if `pipe.overprint_additive` is `false`; replace-mode overprint is
/// not yet implemented.
fn apply_overprint(pipe: &PipeState<'_>, dst_px: &mut [u8], src_px: &[u8], ncomps: usize) {
    if pipe.overprint_additive {
        for j in 0..ncomps {
            // Channels whose bit is 0 in the mask are not painted; dst already holds
            // the correct value, so skip them.
            if pipe.overprint_mask & (1 << j) == 0 {
                continue;
            }
            // Additive overprint: accumulate into the destination, clamped to 255.
            dst_px[j] = (u16::from(dst_px[j]) + u16::from(src_px[j])).min(255) as u8;
        }
    } else {
        // Replace overprint: channels not in mask should keep the original dst value,
        // but the pre-blend destination has already been overwritten.  Restoring it
        // here would require the caller to pass the pre-blend dst separately.
        // Panic loudly (in both debug and release) rather than silently producing
        // wrong output.  Callers must either set overprint_additive=true or pass the
        // original dst bytes before calling apply_overprint.
        panic!(
            "general pipe: replace overprint (mask={:#010x}) is not yet implemented; \
             use overprint_additive=true or preserve pre-blend dst in the caller",
            pipe.overprint_mask,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipe::GroupBackdrop;
    use crate::testutil::make_pipe;

    #[test]
    fn compute_a_src_no_mask_no_shape_returns_a_input() {
        assert_eq!(compute_a_src(200, 0xFF, 0xFF, false, false), 200);
    }

    #[test]
    fn compute_a_src_shape_zero_gives_zero() {
        assert_eq!(compute_a_src(255, 0xFF, 0, false, true), 0);
    }

    #[test]
    fn compute_a_src_soft_mask_scales_alpha() {
        let result = compute_a_src(255, 128, 0xFF, true, false);
        let expected = u32::from(div255(255 * 128));
        assert_eq!(result, expected);
    }

    #[test]
    fn compute_a_src_soft_and_shape_combines_both() {
        let result = compute_a_src(255, 128, 128, true, true);
        let expected = u32::from(div255(u32::from(div255(255 * 128)) * 128));
        assert_eq!(result, expected);
    }
    use crate::pipe::{PipeSrc, PipeState};
    use crate::state::TransferSet;
    use color::{Gray8, Rgb8, TransferLut};

    fn normal_pipe(a: u8) -> PipeState<'static> {
        PipeState {
            blend_mode: BlendMode::Normal,
            a_input: a,
            overprint_mask: 0xFFFF_FFFF,
            overprint_additive: false,
            transfer: TransferSet::identity_rgb(),
            soft_mask: None,
            backdrop: None,
            knockout: false,
            knockout_opacity: 255,
        }
    }

    #[test]
    fn opaque_src_over_any_dst_gives_src() {
        let pipe = normal_pipe(255);
        let src_color = [200u8, 100, 50];
        let src = PipeSrc::Solid(&src_color);

        let mut dst = vec![10u8, 20, 30];
        let mut alpha = vec![128u8];

        render_span_general::<Rgb8>(&pipe, &src, &mut dst, Some(&mut alpha), None, 0, 0, 0);

        // a_src = 255, a_result = 255; c = (0 * c_dst + 255 * c_src) / 255 = c_src.
        assert_eq!(&dst, &[200, 100, 50]);
        assert_eq!(alpha[0], 255);
    }

    #[test]
    fn transparent_src_leaves_dst_unchanged() {
        let pipe = normal_pipe(0);
        let src = PipeSrc::Solid(&[255u8, 255, 255]);

        let mut dst = vec![10u8, 20, 30];
        let mut alpha = vec![200u8];

        render_span_general::<Rgb8>(&pipe, &src, &mut dst, Some(&mut alpha), None, 0, 0, 0);

        // a_src = 0; a_result = 0 + 200 - 0 = 200.
        // c = (200 * c_dst + 0) / 200 = c_dst.
        assert_eq!(&dst, &[10, 20, 30]);
        assert_eq!(alpha[0], 200);
    }

    #[test]
    fn blend_multiply_with_dst() {
        let mut pipe = normal_pipe(255);
        pipe.blend_mode = BlendMode::Multiply;

        // src = 128 (grey), dst = 200.
        let src = PipeSrc::Solid(&[128u8, 128, 128]);
        let mut dst = vec![200u8, 200, 200];
        let mut alpha = vec![255u8];

        render_span_general::<Rgb8>(&pipe, &src, &mut dst, Some(&mut alpha), None, 0, 0, 0);

        // With a_src=255, a_dst=255: a_result=255, alpha_i=255, alpha_im1=255.
        // c = ((255-255)*200 + 255*((255-255)*128 + 255*Multiply(128,200)/255)) / 255
        //   = Multiply(128, 200)
        //   = div255(128 * 200) ≈ 100.
        let v = dst[0];
        assert!((i32::from(v) - 100).abs() <= 1, "expected ~100, got {v}");
    }

    #[test]
    fn a_result_is_the_union_of_source_and_destination() {
        // a_src=128, a_dst=200: 128 + 200 - div255(128 * 200) ≈ 228.
        let ar = compute_a_result(128, 200, 255, false);
        assert!((226..=230).contains(&ar), "a_result={ar}");
        // Knockout: the source replaces the destination within its shape.
        assert_eq!(compute_a_result(128, 200, 255, true), 128);
    }

    #[test]
    fn union_colour_composites_the_group_over_its_backdrop() {
        let mut out = [0u8; 3];
        // Nothing painted yet: the backdrop itself.
        assert_eq!(
            union_colour(&[9, 9, 9], 0, &[10, 20, 30], 255, &mut out),
            255
        );
        assert_eq!(out, [10, 20, 30]);
        // Opaque content hides the backdrop.
        assert_eq!(
            union_colour(&[1, 2, 3], 255, &[10, 20, 30], 255, &mut out),
            255
        );
        assert_eq!(out, [1, 2, 3]);
        // Half-covered white over opaque black: mid-grey.
        assert_eq!(union_colour(&[255; 3], 128, &[0; 3], 255, &mut out), 255);
        assert_eq!(out, [128, 128, 128]);
        // A transparent backdrop contributes nothing.
        assert_eq!(union_colour(&[255; 3], 128, &[0; 3], 0, &mut out), 128);
        assert_eq!(out, [255, 255, 255]);
    }

    /// Inside a non-isolated group, a blend mode sees the backdrop: blue
    /// multiplied into an empty group over opaque mid-grey stores
    /// (0, 0, 128) at full alpha, where an isolated group stores blue.
    #[test]
    fn blend_inside_a_non_isolated_group_sees_the_backdrop() {
        let backdrop = [128u8, 128, 128];
        let mut pipe = make_pipe(255, BlendMode::Multiply);
        pipe.backdrop = Some(GroupBackdrop {
            colour: &backdrop,
            alpha: None,
            x0: 0,
            y0: 0,
            width: 1,
            height: 1,
        });
        let mut dst = [0u8; 3];
        let mut alpha = [0u8];
        render_span_general::<Rgb8>(
            &pipe,
            &PipeSrc::Solid(&[0, 0, 255]),
            &mut dst,
            Some(&mut alpha),
            None,
            0,
            0,
            0,
        );
        assert_eq!((dst, alpha), ([0, 0, 128], [255]));

        // Outside the copied rectangle the backdrop is transparent: no blend.
        let mut dst = [0u8; 3];
        let mut alpha = [0u8];
        render_span_general::<Rgb8>(
            &pipe,
            &PipeSrc::Solid(&[0, 0, 255]),
            &mut dst,
            Some(&mut alpha),
            None,
            5,
            5,
            0,
        );
        assert_eq!((dst, alpha), ([0, 0, 255], [255]));
    }

    #[test]
    fn soft_mask_modulates_alpha() {
        // soft_mask[0] = 128 → a_src = div255(255 * 128) ≈ 128.
        let soft = vec![128u8];
        let mut dst = vec![0u8; 3];
        let mut alpha = vec![0u8];

        // We need a pipe with a soft_mask reference.
        // Since PipeState has a 'bmp lifetime, we store soft_mask as a slice reference.
        let pipe = PipeState {
            blend_mode: BlendMode::Normal,
            a_input: 255,
            overprint_mask: 0xFFFF_FFFF,
            overprint_additive: false,
            transfer: TransferSet::identity_rgb(),
            soft_mask: Some(soft.as_slice()),
            backdrop: None,
            knockout: false,
            knockout_opacity: 255,
        };

        let src = PipeSrc::Solid(&[255u8, 255, 255]);
        render_span_general::<Rgb8>(&pipe, &src, &mut dst, Some(&mut alpha), None, 0, 0, 0);

        // a_src ≈ 128; a_dst = 0; a_result ≈ 128.
        // c = (0 * 0 + 128 * 255) / 128 = 255.
        assert_eq!(dst[0], 255);
        assert!((i32::from(alpha[0]) - 128).abs() <= 2, "alpha={}", alpha[0]);
    }

    #[test]
    fn gray_transfer_lut_applied_correctly() {
        // Regression: the old apply_transfer_channel used rgb[0] for channel 0
        // regardless of pixel mode, so gray transfer was silently wrong.
        // Build an inverting gray LUT and a pass-through rgb LUT; the general
        // pipe over a Gray8 pixel must invert the output via the gray table.
        static DN: [[u8; 256]; 8] = [TransferLut::IDENTITY.0; 8];
        let id = TransferLut::IDENTITY.as_array();
        let inv = TransferLut::INVERTED.as_array();
        let transfer = TransferSet::new([id; 3], inv, [id; 4], &DN); // inverting gray transfer
        let pipe = PipeState {
            blend_mode: BlendMode::Normal,
            a_input: 255,
            overprint_mask: 0xFFFF_FFFF,
            overprint_additive: false,
            transfer,
            soft_mask: None,
            backdrop: None,
            knockout: false,
            knockout_opacity: 255,
        };

        // Opaque gray source value 100; after inverting transfer: 155.
        let src_color = [100u8];
        let src = PipeSrc::Solid(&src_color);
        let mut dst = vec![0u8; 1];
        let mut alpha = vec![0u8; 1];
        render_span_general::<Gray8>(&pipe, &src, &mut dst, Some(&mut alpha), None, 0, 0, 0);
        assert_eq!(dst[0], 155, "gray transfer must use gray LUT, not rgb[0]");
    }
}
