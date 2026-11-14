//! AA pipe: shape present, `BlendMode::Normal`, no soft mask, isolated group.
//!
//! For each pixel, with `M = S::MAX`:
//! 1. `a_src = div_max(a_input * shape)` — scale source alpha by the shape.
//! 2. If `a_src == M`: direct write (no read-back needed).
//! 3. If `a_src == 0` and `a_dst == 0`: write zeros.
//! 4. Otherwise: `a_result = a_src + a_dst - div_max(a_src * a_dst)`.
//!    `c_result = ((a_result - a_src) * c_dst + a_src * c_src) / a_result`.
//!    Then apply transfer LUT.

use crate::pipe::{self, PipeSrc, PipeState, Scratch};
use crate::simd::composite_aa_rgb8_opaque;
use crate::types::BlendMode;
use color::{Pixel, Sample, Wide};

// Per-thread scratch buffer for pattern spans — grow-never-shrink, zero per-span alloc.
thread_local! {
    static PAT_BUF: Scratch = const { Scratch::new() };
}

/// Composite a span with a per-pixel shape (AA coverage or group alpha).
///
/// `shape[i]` is the shape for pixel `x0 + i`.  Length must equal
/// `x1 - x0 + 1`.
///
/// # Preconditions (checked in `render_span`)
///
/// - `pipe.use_aa_path()` — no soft mask, `BlendMode::Normal`, no group correction.
/// - `dst_pixels.len() == count * P::NCOMPS`.
/// - `shape.len() == count`.
/// - `P::NCOMPS > 0`.
#[expect(
    clippy::too_many_arguments,
    reason = "the destination row, its alpha, the shape and the span position are independent inputs"
)]
pub(crate) fn render_span_aa<P: Pixel>(
    pipe: &PipeState<'_, P::Sample>,
    src: &PipeSrc<'_, P::Sample>,
    dst_pixels: &mut [P::Sample],
    dst_alpha: Option<&mut [P::Sample]>,
    shape: &[P::Sample],
    x0: i32,
    x1: i32,
    y: i32,
) {
    debug_assert_eq!(pipe.blend_mode, BlendMode::Normal);

    #[expect(
        clippy::cast_sign_loss,
        reason = "x1 >= x0 is a precondition, so x1 - x0 + 1 >= 1 > 0"
    )]
    let count = (x1 - x0 + 1) as usize;
    let ncomps = P::NCOMPS;

    debug_assert_eq!(shape.len(), count, "shape length must equal pixel count");
    debug_assert_eq!(dst_pixels.len(), count * ncomps);

    let a_input = pipe.a_input.wide();

    match src {
        PipeSrc::Solid(color) => {
            debug_assert_eq!(color.len(), ncomps);

            // Fast path: solid 8-bit RGB source, no alpha plane, identity transfer.
            if dst_alpha.is_none() && ncomps == 3 && pipe.transfer.is_identity_rgb() {
                let shape8 = P::Sample::as_u8(shape);
                let color8 = P::Sample::as_u8(color);
                let a8 = P::Sample::as_u8(std::slice::from_ref(&pipe.a_input));
                if let (Some(dst), Some(shape), Some(&[r, g, b]), Some(&[a])) =
                    (P::Sample::as_u8_mut(dst_pixels), shape8, color8, a8)
                {
                    composite_aa_rgb8_opaque(dst, [r, g, b], a, shape);
                    return;
                }
            }

            // General solid path: read colour directly — no allocation.
            render_span_aa_inner(
                pipe,
                |_i| color,
                dst_pixels,
                dst_alpha,
                shape,
                count,
                ncomps,
                a_input,
            );
        }
        PipeSrc::Pattern(pat) => {
            PAT_BUF.with(|scratch| {
                scratch.with::<P::Sample, ()>(count * ncomps, |buf| {
                    pat.fill_span(y, x0, x1, buf);
                    let buf = &*buf;
                    render_span_aa_inner(
                        pipe,
                        |i| &buf[i * ncomps..(i + 1) * ncomps],
                        dst_pixels,
                        dst_alpha,
                        shape,
                        count,
                        ncomps,
                        a_input,
                    );
                });
            });
        }
    }
}

/// Inner AA compositing loop.
///
/// `src_px_at(i)` returns the `ncomps` samples of the source pixel at
/// index `i`.  For solid sources this is always the same slice; for patterns it
/// indexes into the pre-filled scratch buffer.  Using a closure rather than a
/// `bool` flag keeps a single code path and lets the compiler inline both variants.
#[inline]
#[expect(
    clippy::too_many_arguments,
    reason = "all params necessary; closure eliminates the solid/pattern duplication"
)]
fn render_span_aa_inner<'src, S: Sample>(
    pipe: &PipeState<'_, S>,
    src_px_at: impl Fn(usize) -> &'src [S],
    dst_pixels: &mut [S],
    dst_alpha: Option<&mut [S]>,
    shape: &[S],
    count: usize,
    ncomps: usize,
    a_input: S::Wide,
) {
    let max = S::max_wide();
    let zero = S::Wide::ZERO;
    match dst_alpha {
        Some(dst_alpha) => {
            debug_assert_eq!(dst_alpha.len(), count);
            for i in 0..count {
                let shape_v = shape[i].wide();
                let a_src = S::div_max(a_input * shape_v).wide();
                let a_dst = dst_alpha[i].wide();

                let (a_result, fully_opaque_src) = if a_src == max {
                    (max, true)
                } else if a_src == zero && a_dst == zero {
                    // Transparent src over transparent dst: zero and skip.
                    let base = i * ncomps;
                    dst_pixels[base..base + ncomps].fill(S::default());
                    dst_alpha[i] = S::default();
                    continue;
                } else {
                    let ar = a_src + a_dst - S::div_max(a_src * a_dst).wide();
                    (ar, false)
                };

                let base = i * ncomps;
                let src_px = src_px_at(i);
                let dst_px = &mut dst_pixels[base..base + ncomps];

                if fully_opaque_src {
                    // Full coverage: transfer src directly, no blending needed.
                    pipe::apply_transfer_pixel(pipe, src_px, dst_px);
                } else {
                    // Partial coverage: Porter-Duff over, then apply transfer.
                    for j in 0..ncomps {
                        let c_src = src_px[j].wide();
                        let c_dst = dst_px[j].wide();
                        // ((a_result - a_src) * c_dst + a_src * c_src) / a_result:
                        // a weighted mean of samples, at most M.
                        dst_px[j] =
                            S::from_wide(((a_result - a_src) * c_dst + a_src * c_src) / a_result);
                    }
                    pipe::apply_transfer_in_place(pipe, dst_px);
                }
                // a_result = a_src + a_dst - div_max(a_src·a_dst) ≤ M.
                dst_alpha[i] = S::from_wide(a_result);
            }
        }
        None => {
            // No separate alpha plane: a_dst is implicitly M, a_result = M.
            // Formula simplifies to: c = div_max((M - a_src) * c_dst + a_src * c_src).
            for (i, &sh) in shape.iter().enumerate() {
                let a_src = S::div_max(a_input * sh.wide()).wide();
                let base = i * ncomps;
                let src_px = src_px_at(i);
                let dst_px = &mut dst_pixels[base..base + ncomps];
                for j in 0..ncomps {
                    dst_px[j] =
                        S::div_max((max - a_src) * dst_px[j].wide() + a_src * src_px[j].wide());
                }

                pipe::apply_transfer_in_place(pipe, dst_px);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipe::PipeSrc;
    use crate::pipe::TransferSet;
    use color::{Rgb8, TransferLut};

    fn aa_pipe() -> PipeState<'static> {
        PipeState {
            blend_mode: BlendMode::Normal,
            a_input: 255,
            transfer: TransferSet::identity_rgb(),
            backdrop: None,
            overprint: None,
        }
    }

    #[test]
    fn full_coverage_writes_src() {
        let pipe = aa_pipe();
        let color = [200u8, 100, 50];
        let src = PipeSrc::Solid(&color);
        let shape = [255u8, 255];

        let mut dst = vec![50u8; 6]; // two pixels, initially different from src
        let mut alpha = vec![128u8; 2];

        render_span_aa::<Rgb8>(&pipe, &src, &mut dst, Some(&mut alpha), &shape, 0, 1, 0);

        assert_eq!(&dst[0..3], &[200, 100, 50]);
        assert_eq!(&dst[3..6], &[200, 100, 50]);
        assert_eq!(alpha[0], 255);
        assert_eq!(alpha[1], 255);
    }

    #[test]
    fn zero_coverage_over_transparent_zeroes_output() {
        let pipe = aa_pipe();
        let color = [255u8, 255, 255];
        let src = PipeSrc::Solid(&color);
        let shape = [0u8];

        let mut dst = vec![0u8; 3];
        let mut alpha = vec![0u8; 1]; // dest also transparent

        render_span_aa::<Rgb8>(&pipe, &src, &mut dst, Some(&mut alpha), &shape, 0, 0, 0);

        assert_eq!(dst[0], 0);
        assert_eq!(alpha[0], 0);
    }

    #[test]
    fn half_coverage_blends_correctly() {
        let pipe = aa_pipe();
        // src = white (255,255,255), dst = black (0,0,0), shape ≈ 128 ≈ 50%.
        let color = [255u8, 255, 255];
        let src = PipeSrc::Solid(&color);
        let shape = [128u8];

        let mut dst = vec![0u8; 3];
        let mut alpha = vec![255u8; 1]; // fully opaque destination

        render_span_aa::<Rgb8>(&pipe, &src, &mut dst, Some(&mut alpha), &shape, 0, 0, 0);

        // a_src = div255(255 * 128) ≈ 128.
        // a_result = 128 + 255 - div255(128 * 255) ≈ 255.
        // c = ((255 - 128) * 0 + 128 * 255) / 255 ≈ 128.
        let v = dst[0];
        assert!((125..=131).contains(&v), "expected ~128, got {v}");
        assert_eq!(alpha[0], 255);
    }

    #[test]
    fn no_alpha_plane_uses_opaque_dst() {
        let pipe = aa_pipe();
        let color = [200u8, 100, 50];
        let src = PipeSrc::Solid(&color);
        let shape = [128u8];

        let mut dst = vec![0u8; 3]; // black dst

        render_span_aa::<Rgb8>(&pipe, &src, &mut dst, None, &shape, 0, 0, 0);

        // With implicit a_dst=255: result should be a blend.
        // Expected: div255((255 - 128) * 0 + 128 * 200) ≈ 100.
        let v = dst[0];
        assert!((95..=105).contains(&v), "expected ~100, got {v}");
    }

    /// `TransferSet::is_identity_rgb()` gates a SIMD-friendly fast path
    /// (`composite_aa_rgb8_opaque`) that intentionally skips transfer-LUT
    /// application. If the predicate mis-reports `true` for a non-identity
    /// LUT (cargo-mutants whole-body → `true` survives without this test),
    /// the fast path runs and silently drops the transfer.
    ///
    /// Construct a non-identity LUT (channel-inverting), run `render_span_aa`,
    /// and require the inversion to be visible — only the general path
    /// applies it.
    #[test]
    fn non_identity_transfer_must_use_general_path() {
        // Inverting RGB transfer + identity gray/cmyk/device_n; the inverting
        // table is what makes this test's transfer set non-identity.
        let id = TransferLut::IDENTITY.as_array();
        let inv = TransferLut::INVERTED.as_array();

        let pipe = PipeState {
            blend_mode: BlendMode::Normal,
            a_input: 255,
            transfer: TransferSet::new([inv; 3], id, [id; 4], [id; 8]),
            backdrop: None,
            overprint: None,
        };
        assert!(
            !pipe.transfer.is_identity_rgb(),
            "test prerequisite: inverting LUT must not register as identity"
        );

        let color = [200u8, 100, 50];
        let src = PipeSrc::Solid(&color);
        let shape = [255u8; 4]; // full coverage → general path writes src, then applies transfer
        let mut dst = vec![0u8; 12];

        render_span_aa::<Rgb8>(&pipe, &src, &mut dst, None, &shape, 0, 3, 0);

        // General path: full coverage → `apply_transfer_pixel` runs and
        // emits `255 - src`. Fast path would emit `src` unchanged.
        for px in 0..4 {
            assert_eq!(
                &dst[px * 3..px * 3 + 3],
                &[55, 155, 205],
                "pixel {px}: transfer LUT must invert each channel; \
                 if the fast-path gate mis-fired, dst would be [200, 100, 50]"
            );
        }
    }

    /// `TransferSet::is_identity_rgb()` gates the fast path; when it returns
    /// `true`, `composite_aa_rgb8_opaque` runs.  The prerequisite assertion
    /// catches a predicate that reports `false` for the identity LUT.  The
    /// two paths produce identical bytes (see the cross-path fixture below),
    /// so the pinned bytes constrain the composite, not the path taken.
    ///
    /// The bytes of a span crossing the lane boundary are pinned against an
    /// inline copy of the fast path's algebra.
    #[test]
    fn identity_transfer_takes_fast_path_with_pinned_bytes() {
        let pipe = aa_pipe();
        assert!(
            pipe.transfer.is_identity_rgb(),
            "test prerequisite: aa_pipe() must register as identity"
        );

        // 17 pixels: crosses the LANE=16 boundary, exercising both the
        // chunked path and the scalar tail.
        let color = [200u8, 100, 50];
        let src = PipeSrc::Solid(&color);
        let shape: Vec<u8> = (0u8..17).map(|i| i.wrapping_mul(17)).collect();
        let initial: Vec<u8> = (0u8..51).map(|i| i.wrapping_mul(13)).collect();
        let mut dst_fast = initial.clone();

        render_span_aa::<Rgb8>(&pipe, &src, &mut dst_fast, None, &shape, 0, 16, 0);

        // Compute the reference via the fast path's formula, with
        // div255(v) = v / 255 rounded to nearest applied at both stages:
        //   a_src   = div255(a_input * shape[i])
        //   c_out_j = div255((255 - a_src) * c_dst[j] + a_src * src[j])
        let div255 = |v: u16| u16::from(color::convert::div255(u32::from(v)));
        let a_in = 255u16;
        let mut expected = initial;
        for (i, &sh) in shape.iter().enumerate() {
            let a_src = div255(a_in * u16::from(sh));
            let inv = 255 - a_src;
            let b = i * 3;
            for (j, sc) in color.iter().enumerate() {
                let v = div255(inv * u16::from(expected[b + j]) + a_src * u16::from(*sc));
                // `v` ≤ 255 on this operand range (max intermediate 65 407).
                expected[b + j] = u8::try_from(v).expect("fast-path div255 result must fit in u8");
            }
        }
        assert_eq!(
            dst_fast, expected,
            "identity-LUT path must produce the exact-div255 fast-path bytes"
        );
    }

    // ── Cross-path byte-equality fixture ─────────────────────────────────────
    //
    // The fast path (`composite_aa_rgb8_opaque`) and the general no-alpha
    // path (`render_span_aa_inner`) both apply `v / 255` rounded to nearest
    // (`color::convert::div255`) twice each, so their output must be
    // byte-identical on every gate-eligible input. Output bytes must not
    // depend on which path the transfer-identity gate picks:
    // an approximate div255 in either tier composes to a 2-LSB divergence
    // (e.g. coverage 187, a_input 3, c_dst 0, c_src 171), making rendered
    // pixels depend on an unrelated graphics-state entry.
    //
    // Strategy:
    //   1. Build a deterministic corpus of (color, a_input, shape, dst)
    //      tuples that span the gate-firing domain, including small odd
    //      a_input values where an approximate a_src error would compound.
    //   2. Run the fast path via `render_span_aa::<Rgb8>` with
    //      `dst_alpha = None` and an identity transfer (so the gate fires).
    //   3. Compute the general-path reference using the exact
    //      `color::convert::div255` — same algebra as the `None` arm of
    //      `render_span_aa_inner`, with the identity transfer step elided.
    //   4. Assert `fast[i] == reference[i]` for every byte.

    /// Span lengths the corpus iterates over.  Covers:
    /// * `7`  — pure scalar tail (count < LANE), no chunked branch.
    /// * `16` — exactly one LANE chunk, zero tail.
    /// * `17` — one LANE chunk + 1-byte tail (chunk-tail boundary).
    /// * `23` — one LANE chunk + 7-byte tail (typical mixed-mode width).
    const CROSS_PATH_SPAN_LENGTHS: [usize; 4] = [7, 16, 17, 23];

    // ── Corpus pattern generators (free `fn` items, no allocation) ────────────

    fn shape_full(_: usize) -> u8 {
        255
    }
    fn shape_zero(_: usize) -> u8 {
        0
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "i ∈ [0, max span = 23) * 11 ≤ 242 fits u8"
    )]
    fn shape_ramp(i: usize) -> u8 {
        (i * 11) as u8
    }
    fn shape_alt(i: usize) -> u8 {
        if i.is_multiple_of(2) { 255 } else { 64 }
    }

    fn dst_black(_: usize) -> u8 {
        0
    }
    fn dst_white(_: usize) -> u8 {
        255
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "(i * 7) % 256 fits u8 by construction"
    )]
    fn dst_ramp(i: usize) -> u8 {
        ((i * 7) % 256) as u8
    }
    fn dst_alt(i: usize) -> u8 {
        if i.is_multiple_of(2) { 0 } else { 200 }
    }

    /// Run the gate-eligible (no alpha plane, identity transfer) fast path
    /// over `initial`, returning the result. The test's "ground truth" leg
    /// is `run_exact_reference`.
    fn run_fast_path(color: [u8; 3], a_input: u8, shape: &[u8], initial: &[u8]) -> Vec<u8> {
        let mut dst = initial.to_vec();
        let mut pipe = aa_pipe();
        pipe.a_input = a_input;
        let src = PipeSrc::Solid(color.as_slice());
        let count = shape.len();
        assert!(count >= 1, "render_span_aa requires count >= 1");
        let x1: i32 = i32::try_from(count - 1).expect(
            "CROSS_PATH_SPAN_LENGTHS values must fit in i32 for render_span_aa's x0..=x1 API",
        );
        render_span_aa::<Rgb8>(&pipe, &src, &mut dst, None, shape, 0, x1, 0);
        dst
    }

    /// Inline algebra of the general no-alpha path (`aa.rs:192-207`) under
    /// an identity transfer — `apply_transfer_in_place` is a no-op so the
    /// loop reduces to per-channel
    /// `div255((255 - a_src) * c_dst + a_src * c_src)` using the exact
    /// `color::convert::div255`.
    fn run_exact_reference(color: [u8; 3], a_input: u8, shape: &[u8], initial: &[u8]) -> Vec<u8> {
        use color::convert::div255 as exact_div255;
        let mut dst = initial.to_vec();
        let a_in_u32 = u32::from(a_input);
        for (i, &sh) in shape.iter().enumerate() {
            let a_src = u32::from(exact_div255(a_in_u32 * u32::from(sh)));
            let base = i * 3;
            for j in 0..3 {
                let c_dst = u32::from(dst[base + j]);
                let c_src = u32::from(color[j]);
                dst[base + j] = exact_div255((255 - a_src) * c_dst + a_src * c_src);
            }
        }
        dst
    }

    /// Per-byte equality assertion with a diagnostic locating the input.
    fn assert_bytes_equal(
        fast: &[u8],
        exact: &[u8],
        color: [u8; 3],
        a_input: u8,
        shape: &[u8],
        initial: &[u8],
    ) {
        for (i, (&f, &r)) in fast.iter().zip(exact.iter()).enumerate() {
            assert_eq!(
                f,
                r,
                "byte {i}: fast={f}, exact={r}; \
                 colour={color:?}, a_input={a_input}, \
                 shape[{i_px}]={sh}, initial[{i}]={init}",
                i_px = i / 3,
                sh = shape[i / 3],
                init = initial[i],
            );
        }
    }

    #[test]
    fn fast_path_matches_general_div255_exactly() {
        let pipe = aa_pipe();
        assert!(
            pipe.transfer.is_identity_rgb(),
            "test prerequisite: aa_pipe() must register as identity so the gate fires"
        );

        // Corpus: cross-product of representative source colours, alpha
        // inputs, shape patterns, initial destinations, and span lengths.
        // Small odd a_input values (3, 5) are where an approximate a_src
        // rounding error would compound with the per-channel division.
        let colours: [[u8; 3]; 6] = [
            [0, 0, 0],       // black
            [255, 255, 255], // white
            [200, 100, 50],  // mid-saturated warm
            [1, 254, 127],   // off-by-one boundaries
            [128, 128, 128], // 50%-grey
            [171, 171, 171], // known 2-LSB repro colour under a_input=3
        ];
        let a_inputs: [u8; 6] = [0, 1, 3, 5, 128, 255];
        let shape_patterns: [fn(usize) -> u8; 4] = [shape_full, shape_zero, shape_ramp, shape_alt];
        let dst_patterns: [fn(usize) -> u8; 4] = [dst_black, dst_white, dst_ramp, dst_alt];

        for &count in &CROSS_PATH_SPAN_LENGTHS {
            for &color in &colours {
                for &a_input in &a_inputs {
                    for &sh_fn in &shape_patterns {
                        for &dst_fn in &dst_patterns {
                            let shape: Vec<u8> = (0..count).map(sh_fn).collect();
                            let initial: Vec<u8> = (0..count * 3).map(dst_fn).collect();
                            let fast = run_fast_path(color, a_input, &shape, &initial);
                            let exact = run_exact_reference(color, a_input, &shape, &initial);
                            assert_bytes_equal(&fast, &exact, color, a_input, &shape, &initial);
                        }
                    }
                }
            }
        }
    }
}
