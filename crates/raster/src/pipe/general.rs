//! General pipe: non-normal blend modes and non-isolated groups.
//!
//! The path for every span the `simple` and `aa` pipes do not take.
//!
//! # Compositing formula (PDF spec §11.3)
//!
//! Given source alpha `a_src`, destination alpha `a_dst`, source/dest
//! colours and `M = S::MAX`:
//!
//! ```text
//! a_result = a_src + a_dst - div_max(a_src * a_dst)
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
//! a_union  = a_dst + a_0 - div_max(a_dst * a_0)
//! c_union  = (a_dst * c_dst + (M - a_dst) * a_0 * c_0 / M) / a_union
//! c_result = ((a_result - a_src) * c_dst
//!             + a_src * ((M - a_union) * c_src + a_union * blend(c_src, c_union)) / M)
//!            / a_result
//! ```

use crate::pipe::{self, PipeSrc, PipeState, Scratch, blend};
use crate::types::BlendMode;
use color::{Pixel, Sample, Wide};

pub(crate) const MAX_COMPS: usize = 8; // DeviceN: 4 CMYK + 4 spot channels

// Per-thread scratch buffer for pattern spans — grow-never-shrink.
thread_local! {
    static PAT_BUF: Scratch = const { Scratch::new() };
}

/// General-purpose compositing span function.
///
/// Handles blend modes and non-isolated groups.
/// Slower than `simple` or `aa` but covers every case.
///
/// # Preconditions (checked with `debug_assert!`)
///
/// - `x1 >= x0`
/// - `dst_pixels.len() == (x1 - x0 + 1) * P::NCOMPS`
/// - If `shape` is `Some`, `shape.len() == x1 - x0 + 1`
#[expect(
    clippy::too_many_arguments,
    reason = "the destination row, its alpha, the shape and the span position are independent inputs"
)]
pub(crate) fn render_span_general<P: Pixel>(
    pipe: &PipeState<'_, P::Sample>,
    src: &PipeSrc<'_, P::Sample>,
    dst_pixels: &mut [P::Sample],
    dst_alpha: Option<&mut [P::Sample]>,
    shape: Option<&[P::Sample]>,
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
    let ncomps = P::NCOMPS;

    debug_assert_eq!(dst_pixels.len(), count * ncomps);
    if let Some(sh) = shape {
        debug_assert_eq!(sh.len(), count);
    }

    let a_input = pipe.a_input.wide();
    let is_nonseparable = matches!(
        pipe.blend_mode,
        BlendMode::Hue | BlendMode::Saturation | BlendMode::Color | BlendMode::Luminosity
    );
    let is_cmyk_like = ncomps == 4 || ncomps == 8;

    // Convenience: get optional slice at index i.
    let shape_at = |i: usize| shape.map_or(P::Sample::MAX, |s| s[i]);
    let zero = [P::Sample::default(); MAX_COMPS];
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "i < count = x1 − x0 + 1, which fits in i32"
    )]
    let backdrop_at = |i: usize| {
        pipe.backdrop.as_ref().map(|b| {
            b.pixel(x0 + i as i32, y, ncomps)
                .unwrap_or_else(|| (&zero[..ncomps], P::Sample::default()))
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
                &backdrop_at,
            );
        }
        PipeSrc::Pattern(pat) => {
            PAT_BUF.with(|scratch| {
                scratch.with::<P::Sample, ()>(count * ncomps, |buf| {
                    pat.fill_span(y, x0, x1, buf);
                    let buf = &*buf;
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
                        &backdrop_at,
                    );
                });
            });
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "all params necessary; closure eliminates solid/pattern duplication"
)]
#[expect(
    clippy::single_match_else,
    reason = "both Some and None arms have substantial independent logic; if-let would be less clear"
)]
fn render_span_general_inner<'src, 'bd, S: Sample>(
    pipe: &PipeState<'_, S>,
    src_px_at: impl Fn(usize) -> &'src [S],
    dst_pixels: &mut [S],
    dst_alpha: Option<&mut [S]>,
    shape: Option<&[S]>,
    count: usize,
    ncomps: usize,
    a_input: S::Wide,
    is_nonseparable: bool,
    is_cmyk_like: bool,
    shape_at: &dyn Fn(usize) -> S,
    backdrop_at: &dyn Fn(usize) -> Option<(&'bd [S], S)>,
) {
    let has_shape = shape.is_some();
    let max = S::max_wide();

    match dst_alpha {
        Some(dst_alpha) => {
            debug_assert_eq!(dst_alpha.len(), count);
            for i in 0..count {
                let src_px = src_px_at(i);
                let dst_px = &mut dst_pixels[i * ncomps..(i + 1) * ncomps];
                let shape_v = shape_at(i).wide();

                // Source alpha (PDF spec §11.3.6 eq 11.1).
                let a_src = compute_a_src::<S>(a_input, shape_v, has_shape);

                let c_src: &[S] = src_px;
                let a_dst_eff = dst_alpha[i].wide();

                // The pixel the blend function sees and the alpha weighting
                // it: the destination, or in a non-isolated group the
                // group's content over the backdrop.  `Normal` reads
                // neither, so the backdrop is consulted only for other modes.
                let mut c_union = [S::default(); MAX_COMPS];
                let backdrop = (pipe.blend_mode != BlendMode::Normal)
                    .then(|| backdrop_at(i))
                    .flatten();
                let (c_blend_dst, alpha_im1): (&[S], S::Wide) = match backdrop {
                    Some((c_0, a_0)) => {
                        let a_union = union_colour(
                            dst_px,
                            a_dst_eff,
                            c_0,
                            a_0.wide(),
                            &mut c_union[..ncomps],
                        );
                        (&c_union[..ncomps], a_union)
                    }
                    None => (&*dst_px, a_dst_eff),
                };

                // Blend function.
                let mut c_blend = [S::default(); MAX_COMPS];
                if pipe.blend_mode != BlendMode::Normal {
                    blend_in_space(
                        pipe,
                        c_src,
                        c_blend_dst,
                        &mut c_blend[..ncomps],
                        is_cmyk_like,
                        is_nonseparable,
                    );
                }

                // Result alpha.
                let a_result = compute_a_result::<S>(a_src, a_dst_eff);
                let alpha_i = a_result;

                // Result colour.
                if a_result == S::Wide::ZERO {
                    dst_px.fill(S::default());
                } else {
                    for j in 0..ncomps {
                        let c_src_j = c_src[j].wide();
                        let c_dst_j = dst_px[j].wide();
                        let c_b_j = c_blend[j].wide();

                        // A weighted mean of samples, at most M.
                        let c = if pipe.blend_mode == BlendMode::Normal {
                            // No blend function: standard Porter-Duff.
                            ((alpha_i - a_src) * c_dst_j + a_src * c_src_j) / alpha_i
                        } else {
                            // With blend function.
                            ((alpha_i - a_src) * c_dst_j
                                + a_src * ((max - alpha_im1) * c_src_j + alpha_im1 * c_b_j) / max)
                                / alpha_i
                        };
                        dst_px[j] = S::from_wide(c);
                    }
                    pipe::apply_transfer_in_place(pipe, dst_px);
                }

                // a_result is clamped to ≤ M in compute_a_result.
                dst_alpha[i] = S::from_wide(a_result);
            }
        }
        None => {
            // No separate alpha plane: aDest = M implicitly.
            // Simplifies: aResult = aSrc + M - div_max(aSrc * M) = M, alpha_i = M.
            // A non-isolated group requires a dst_alpha plane to carry the
            // group alpha.
            debug_assert!(
                pipe.backdrop.is_none(),
                "a group backdrop requires dst_alpha; None arm uses implicit a_dst=M"
            );
            for i in 0..count {
                let src_px = src_px_at(i);
                let dst_px = &mut dst_pixels[i * ncomps..(i + 1) * ncomps];
                let shape_v = shape_at(i).wide();

                let a_src = compute_a_src::<S>(a_input, shape_v, has_shape);
                // a_src is derived from samples via div_max, so always ≤ M.
                debug_assert!(a_src <= max, "a_src={a_src:?} out of [0, M]");

                let mut c_blend = [S::default(); MAX_COMPS];
                if pipe.blend_mode != BlendMode::Normal {
                    blend_in_space(
                        pipe,
                        src_px,
                        dst_px,
                        &mut c_blend[..ncomps],
                        is_cmyk_like,
                        is_nonseparable,
                    );
                }

                for j in 0..ncomps {
                    let c_src_j = src_px[j].wide();
                    let c_dst_j = dst_px[j].wide();
                    let c_b_j = c_blend[j].wide();

                    // With implicit a_dst=M: alpha_i=M, alpha_im1=M.
                    // General formula simplifies to div_max((M-a_src)*c_dst + a_src*c_b).
                    dst_px[j] = if pipe.blend_mode == BlendMode::Normal {
                        S::div_max((max - a_src) * c_dst_j + a_src * c_src_j)
                    } else {
                        S::div_max((max - a_src) * c_dst_j + a_src * c_b_j)
                    };
                }
                pipe::apply_transfer_in_place(pipe, dst_px);
            }
        }
    }
}

/// Compute the effective source alpha for one pixel (PDF spec §11.3.6 eq 11.1):
/// `a_input` scaled by the shape, or `a_input` itself without a shape.
#[inline]
fn compute_a_src<S: Sample>(a_input: S::Wide, shape_v: S::Wide, has_shape: bool) -> S::Wide {
    if has_shape {
        S::div_max(a_input * shape_v).wide()
    } else {
        a_input
    }
}

/// The result alpha of a source of alpha `a_src` over a destination of
/// alpha `a_dst`: their union.
pub(crate) fn compute_a_result<S: Sample>(a_src: S::Wide, a_dst: S::Wide) -> S::Wide {
    (a_src + a_dst - S::div_max(a_src * a_dst).wide()).min(S::max_wide())
}

/// Write into `out` the colour of a group's content (`c_dst`, alpha `a_dst`)
/// composited over its backdrop (`c_0`, alpha `a_0`) and return the alpha
/// of that union.  Black where the union is transparent.
pub(crate) fn union_colour<S: Sample>(
    c_dst: &[S],
    a_dst: S::Wide,
    c_0: &[S],
    a_0: S::Wide,
    out: &mut [S],
) -> S::Wide {
    let a_union = compute_a_result::<S>(a_dst, a_0);
    if a_union == S::Wide::ZERO {
        out.fill(S::default());
        return a_union;
    }
    // `a_union` is the union rounded to nearest, which can lie below the
    // exact union `a_dst + a_0·(M − a_dst)/M`; the quotient of a mean of
    // values near `M` can then exceed `M`, and is clamped to it.
    let max = S::max_wide();
    for ((o, &c_d), &c_b) in out.iter_mut().zip(c_dst).zip(c_0) {
        let num = a_dst * c_d.wide() * max + (max - a_dst) * a_0 * c_b.wide();
        *o = S::from_wide((num / (max * a_union)).min(max));
    }
    a_union
}

/// Apply the blend function of `pipe` to `src` over `dst` and write into
/// `c_blend`: in the components of the pixels, or in gray when `pipe`
/// blends in a gray space ([`pipe::GrayBlend`]), the result written as a
/// pixel of that gray.
fn blend_in_space<S: Sample>(
    pipe: &PipeState<'_, S>,
    src: &[S],
    dst: &[S],
    c_blend: &mut [S],
    is_cmyk_like: bool,
    is_nonseparable: bool,
) {
    let Some(gray) = pipe.gray else {
        apply_blend_fn(
            pipe.blend_mode,
            src,
            dst,
            c_blend,
            is_cmyk_like,
            is_nonseparable,
        );
        return;
    };
    let mut blended = [S::default()];
    apply_blend_fn(
        pipe.blend_mode,
        &[gray.gray(src)],
        &[gray.gray(dst)],
        &mut blended,
        false,
        is_nonseparable,
    );
    gray.neutral(blended[0], c_blend);
}

/// Whether the separable blend mode `mode` is white-preserving:
/// `B(1, 1) = 1` in additive values.  Every separable mode is but
/// `Difference` and `Exclusion`.
const fn white_preserving(mode: BlendMode) -> bool {
    !matches!(mode, BlendMode::Difference | BlendMode::Exclusion)
}

/// Apply the blend function and write into `c_blend`.
fn apply_blend_fn<S: Sample>(
    mode: BlendMode,
    src: &[S],
    dst: &[S],
    c_blend: &mut [S],
    is_cmyk_like: bool,
    is_nonseparable: bool,
) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len(), c_blend.len());
    let ncomps = src.len();
    let inv = |v: S| S::from_wide(S::max_wide() - v.wide());

    if is_cmyk_like {
        // Subtractive complement: invert all channels, blend in additive space, re-invert.
        // Fill all ncomps so spot channels (j >= 4) are correctly inverted.
        let mut src2 = [S::default(); MAX_COMPS];
        let mut dst2 = [S::default(); MAX_COMPS];
        for j in 0..ncomps {
            src2[j] = inv(src[j]);
            dst2[j] = inv(dst[j]);
        }

        if is_nonseparable {
            let s3 = [src2[0], src2[1], src2[2]];
            let d3 = [dst2[0], dst2[1], dst2[2]];
            let r3 = blend::apply_nonseparable_rgb(mode, s3, d3);
            c_blend[0] = inv(r3[0]);
            c_blend[1] = inv(r3[1]);
            c_blend[2] = inv(r3[2]);
            // K: the source's for Luminosity, the backdrop's otherwise (PDF §11.3.5).
            if ncomps >= 4 {
                c_blend[3] = inv(if mode == BlendMode::Luminosity {
                    src2[3]
                } else {
                    dst2[3]
                });
            }
            // Spot channels take `Normal` under a non-separable mode.
            if ncomps > 4 {
                c_blend[4..ncomps].copy_from_slice(&src[4..ncomps]);
            }
        } else {
            // Spot channels are blended by a white-preserving separable
            // mode and take `Normal` under any other (ISO 32000-1
            // §11.7.4.2).
            let blended = if white_preserving(mode) {
                ncomps
            } else {
                ncomps.min(4)
            };
            blend::apply_separable(
                mode,
                &src2[..blended],
                &dst2[..blended],
                &mut c_blend[..blended],
            );
            for v in &mut c_blend[..blended] {
                *v = inv(*v);
            }
            c_blend[blended..ncomps].copy_from_slice(&src[blended..ncomps]);
        }
    } else if is_nonseparable {
        // RGB/Gray additive space.
        let n = ncomps.min(3);
        let mut s3 = [S::default(); 3];
        let mut d3 = [S::default(); 3];
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipe::GroupBackdrop;
    use crate::testutil::make_pipe;
    use color::convert::div255;

    #[test]
    fn compute_a_src_without_shape_returns_a_input() {
        assert_eq!(compute_a_src::<u8>(200, 0xFF, false), 200);
    }

    #[test]
    fn compute_a_src_shape_zero_gives_zero() {
        assert_eq!(compute_a_src::<u8>(255, 0, true), 0);
    }

    #[test]
    fn compute_a_src_shape_scales_alpha() {
        assert_eq!(
            compute_a_src::<u8>(255, 128, true),
            u32::from(div255(255 * 128))
        );
    }
    use crate::pipe::TransferSet;
    use crate::pipe::{PipeSrc, PipeState};
    use color::{Gray8, Rgb8, TransferLut};

    fn normal_pipe(a: u8) -> PipeState<'static> {
        PipeState {
            blend_mode: BlendMode::Normal,
            a_input: a,
            transfer: TransferSet::identity_rgb(),
            backdrop: None,
            overprint: None,
            gray: None,
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

    /// The gray of an RGB pixel by integer weights summing to 100.
    #[derive(Debug)]
    struct Luma;

    impl crate::pipe::GrayBlend<u8> for Luma {
        fn gray(&self, px: &[u8]) -> u8 {
            let w = |i: usize, k: u32| u32::from(px[i]) * k;
            u8::try_from((w(0, 30) + w(1, 59) + w(2, 11) + 50) / 100).expect("a mean of samples")
        }
        fn neutral(&self, gray: u8, px: &mut [u8]) {
            px.fill(gray);
        }
    }

    /// In a gray blending space the blend function sees the gray of the
    /// source and of the destination: blue multiplied onto red gives the
    /// product of their grays, where blending the components gives black;
    /// `Difference` gives the difference of their grays.
    #[test]
    fn a_gray_blending_space_blends_the_grays() {
        let luma = Luma;
        for (mode, want) in [(BlendMode::Multiply, 8u8), (BlendMode::Difference, 49)] {
            let pipe = PipeState {
                blend_mode: mode,
                gray: Some(&luma),
                ..normal_pipe(255)
            };
            let src = PipeSrc::Solid(&[0u8, 0, 255]);
            let mut dst = vec![255u8, 0, 0];
            let mut alpha = vec![255u8];
            render_span_general::<Rgb8>(&pipe, &src, &mut dst, Some(&mut alpha), None, 0, 0, 0);
            assert_eq!(dst, [want; 3], "{mode:?}");
        }
    }

    #[test]
    fn a_result_is_the_union_of_source_and_destination() {
        // a_src=128, a_dst=200: 128 + 200 - div255(128 * 200) ≈ 228.
        let ar = compute_a_result::<u8>(128, 200);
        assert!((226..=230).contains(&ar), "a_result={ar}");
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

    /// The union of white over white is white, within the rounding of the
    /// union alpha (at most six levels, at the smallest alphas), for every
    /// pair of alphas: the quotient by the rounded union alpha exceeds 255
    /// for some pairs and must not wrap.
    #[test]
    fn union_of_white_over_white_is_white() {
        let mut out = [0u8; 1];
        for a_dst in 0..=255 {
            for a_0 in 0..=255 {
                if union_colour::<u8>(&[255], a_dst, &[255], a_0, &mut out) != 0 {
                    assert!(out[0] >= 249, "a_dst={a_dst} a_0={a_0}: {}", out[0]);
                }
            }
        }
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
    fn gray_transfer_lut_applied_correctly() {
        // Regression: the old apply_transfer_channel used rgb[0] for channel 0
        // regardless of pixel mode, so gray transfer was silently wrong.
        // Build an inverting gray LUT and a pass-through rgb LUT; the general
        // pipe over a Gray8 pixel must invert the output via the gray table.
        let id = TransferLut::IDENTITY.as_array();
        let inv = TransferLut::INVERTED.as_array();
        let transfer = TransferSet::new([id; 3], inv, [id; 4], [id; 8]); // inverting gray transfer
        let pipe = PipeState {
            blend_mode: BlendMode::Normal,
            a_input: 255,
            transfer,
            backdrop: None,
            overprint: None,
            gray: None,
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
