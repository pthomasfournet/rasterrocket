//! The SIMD row kernels of the bilinear rotation, one per architecture,
//! each generic over the sample depth and bit-exact with the scalar kernel
//! (`bilinear` module) at 8 and 16 bits.
//!
//! - **x86-64 (AVX-512 F+BW+VL, runtime-detected)**: [`rotate_row_avx512`]
//!   processes 16 output pixels per iteration end to end: FMA coordinate
//!   stepping, a lane mask for the bounds test, two dword gathers for the
//!   2×2 footprints, FMA bilinear blend, and a white fill merged in for
//!   out-of-bounds lanes.
//! - **aarch64**: [`rotate_row_neon`] processes 8 output pixels per
//!   iteration as two 4-lane quads: fused coordinate stepping, floor, an
//!   unsigned bounds mask, fused bilinear blend and a white merge are
//!   lane-wise; only the 2×2 footprint fetch is per lane (NEON has no
//!   gather).
//! - **other**: the scalar per-pixel loop.  Its per-pixel `floor`, early
//!   return and gathered 2×2 reads keep LLVM from auto-vectorising it.
//!
//! The kernels serve [`DeskewPixel`] samples only: `u8` and `u16`, whose
//! width selects the gather extraction and the narrowing store.

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use color::Sample;
use raster::Bitmap;

use super::DeskewPixel;
#[cfg(target_arch = "aarch64")]
use super::bilinear::footprint;
use super::bilinear::{RowParams, rotate_span_scalar};

/// NEON inner loop: 8 output pixels per iteration as two 4-lane quads,
/// bit-exact with [`rotate_span_scalar`].
///
/// Coordinates are stepped with fused multiply-add exactly as the scalar
/// path does (`cos·ox + sx_base`), floored with `vcvtmq_s32_f32`, and
/// bounds-tested as unsigned integers so a negative origin compares above
/// the maximum.  NEON has no gather, so each in-bounds lane fetches its 2×2
/// footprint through [`footprint`]; out-of-bounds lanes fetch at origin
/// `(0, 0)` and are replaced by `Sample::MAX` afterwards.  The bilinear
/// blend is the scalar `mul_add` chain lane-wise (`vfmaq_f32`),
/// `vcvtaq_u32_f32` rounds half away from zero as `f32::round` does, and
/// saturating narrows to the sample width mirror the clamp of
/// `Sample::from_signed`.  A quad whose lanes are all out of bounds skips
/// the fetch, and the final `dst_row.len() % 8` pixels take the scalar
/// path.
///
/// # Safety
///
/// NEON is mandatory on all ARMv8-A targets; no runtime check is needed.
#[cfg(target_arch = "aarch64")]
#[expect(
    unsafe_code,
    reason = "NEON intrinsics for the aarch64 rotate row kernel; mandatory on ARMv8-A"
)]
#[expect(
    clippy::similar_names,
    reason = "sx/sy and x0/y0 are paired coordinate vectors; renaming would obscure the symmetry"
)]
#[target_feature(enable = "neon")]
pub(super) unsafe fn rotate_row_neon<P: DeskewPixel>(
    dst_row: &mut [P::Sample],
    src: &Bitmap<P>,
    p: &RowParams,
) {
    use std::arch::aarch64::{
        uint32x4_t, vaddq_f32, vandq_u32, vbslq_u32, vcleq_u32, vcombine_u16, vcvtaq_u32_f32,
        vcvtmq_s32_f32, vcvtq_f32_s32, vdupq_n_f32, vdupq_n_u32, vfmaq_f32, vgetq_lane_u32,
        vld1q_f32, vmaxvq_u32, vqmovn_u16, vqmovn_u32, vreinterpretq_u32_s32, vst1_u8, vst1q_u16,
        vsubq_f32,
    };

    // sx_max / sy_max ≥ 0 because the source is at least 1×1.
    #[expect(
        clippy::cast_sign_loss,
        reason = "sx_max and sy_max are w-1 and h-1 with w, h ≥ 1"
    )]
    let (v_sx_max, v_sy_max) = (vdupq_n_u32(p.sx_max as u32), vdupq_n_u32(p.sy_max as u32));
    let v_cos = vdupq_n_f32(p.cos_a);
    let v_sin = vdupq_n_f32(p.sin_a);
    let v_sx_base = vdupq_n_f32(p.sx_base);
    let v_sy_base = vdupq_n_f32(p.sy_base);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "Sample::MAX is at most 65535"
    )]
    let v_white = vdupq_n_u32(P::Sample::MAX.signed() as u32);
    // SAFETY: four f32 values, the extent of one vld1q_f32.
    let lane = unsafe { vld1q_f32([0.0f32, 1.0, 2.0, 3.0].as_ptr()) };

    // Output pixels for columns `ox .. ox + 4`, white where out of bounds.
    let quad = |ox: usize| -> uint32x4_t {
        // ox ≤ DESKEW_MAX_SIDE = 32768; exact in f32.
        #[expect(
            clippy::cast_precision_loss,
            reason = "ox ≤ DESKEW_MAX_SIDE = 32768; exact in f32"
        )]
        let dx = vaddq_f32(vdupq_n_f32(ox as f32), lane);
        let sx = vfmaq_f32(v_sx_base, v_cos, dx);
        let sy = vfmaq_f32(v_sy_base, v_sin, dx);
        let x0 = vcvtmq_s32_f32(sx);
        let y0 = vcvtmq_s32_f32(sy);
        let in_bounds = vandq_u32(
            vcleq_u32(vreinterpretq_u32_s32(x0), v_sx_max),
            vcleq_u32(vreinterpretq_u32_s32(y0), v_sy_max),
        );
        if vmaxvq_u32(in_bounds) == 0 {
            return v_white;
        }
        let fx = vsubq_f32(sx, vcvtq_f32_s32(x0));
        let fy = vsubq_f32(sy, vcvtq_f32_s32(y0));

        // Out-of-bounds lanes are zeroed so every origin is a valid index.
        let x0 = vandq_u32(vreinterpretq_u32_s32(x0), in_bounds);
        let y0 = vandq_u32(vreinterpretq_u32_s32(y0), in_bounds);
        let origin = |x: u32, y: u32| footprint(src, x as usize, y as usize);
        let f = [
            origin(vgetq_lane_u32::<0>(x0), vgetq_lane_u32::<0>(y0)),
            origin(vgetq_lane_u32::<1>(x0), vgetq_lane_u32::<1>(y0)),
            origin(vgetq_lane_u32::<2>(x0), vgetq_lane_u32::<2>(y0)),
            origin(vgetq_lane_u32::<3>(x0), vgetq_lane_u32::<3>(y0)),
        ];
        let corner = |c: usize| {
            let v = [f[0][c], f[1][c], f[2][c], f[3][c]].map(P::to_f32);
            // SAFETY: four f32 values, the extent of one vld1q_f32.
            unsafe { vld1q_f32(v.as_ptr()) }
        };
        let (p00, p10, p01, p11) = (corner(0), corner(1), corner(2), corner(3));

        let top = vfmaq_f32(p00, vsubq_f32(p10, p00), fx);
        let bot = vfmaq_f32(p01, vsubq_f32(p11, p01), fx);
        let val = vfmaq_f32(top, vsubq_f32(bot, top), fy);
        vbslq_u32(in_bounds, vcvtaq_u32_f32(val), v_white)
    };

    let (groups, tail) = dst_row.as_chunks_mut::<8>();
    let tail_start = groups.len() * 8;
    for (g, out) in groups.iter_mut().enumerate() {
        let ox = g * 8;
        let px = vcombine_u16(vqmovn_u32(quad(ox)), vqmovn_u32(quad(ox + 4)));
        // SAFETY: `out` is eight samples: eight bytes, the extent of one
        // vst1_u8, for `u8`; sixteen, the extent of one vst1q_u16, for `u16`.
        unsafe {
            if P::Sample::BITS == 8 {
                vst1_u8(out.as_mut_ptr().cast(), vqmovn_u16(px));
            } else {
                vst1q_u16(out.as_mut_ptr().cast(), px);
            }
        }
    }
    rotate_span_scalar(tail, tail_start, src, p);
}

/// AVX-512 inner loop: 16 output pixels per iteration, bit-exact with
/// [`rotate_span_scalar`].
///
/// Source coordinates are stepped with FMA exactly as the scalar path does
/// (`cos·ox + sx_base`), floored with a round-toward-−∞ conversion, and
/// bounds-tested as integers into a lane mask.  Each in-bounds lane fetches
/// its 2×2 footprint with two dword gathers at byte offsets, row `y0` and
/// row `y1` at column `x0`, whose next sample is the `x0 + 1` neighbour; on
/// the last column that sample is replaced by the `x0` sample, mirroring
/// the edge clamp of `footprint`.  Gather offsets are clamped to
/// `len − 4` bytes and the wanted sample is shifted down, so no dword read
/// crosses the end of the pixel buffer.  Out-of-bounds lanes gather nothing
/// and are merged to `Sample::MAX` before the store.  `round()` is
/// reproduced as `trunc + (frac ≥ 0.5)`, which equals
/// round-half-away-from-zero on the non-negative bilinear result.  A
/// buffer shorter than one dword, or longer than `i32` byte offsets reach,
/// takes the scalar path.
///
/// # Safety
///
/// Caller must ensure `avx512f`, `avx512bw` and `avx512vl` are available.
#[cfg(target_arch = "x86_64")]
#[expect(
    unsafe_code,
    reason = "AVX-512 intrinsics for the x86-64 rotate row kernel; runtime-detected by the caller"
)]
#[expect(
    clippy::too_many_lines,
    reason = "one SIMD pipeline per row; splitting would scatter the lane bookkeeping"
)]
#[expect(
    clippy::similar_names,
    reason = "v_sx_*/v_sy_* are paired coordinate vectors; renaming would obscure the symmetry"
)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub(super) unsafe fn rotate_row_avx512<P: DeskewPixel>(
    dst_row: &mut [P::Sample],
    src: &Bitmap<P>,
    p: &RowParams,
) {
    use std::arch::x86_64::{_MM_FROUND_NO_EXC, _MM_FROUND_TO_NEG_INF, _MM_FROUND_TO_ZERO};
    const FLOOR: i32 = _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC;
    const TRUNC: i32 = _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC;
    use std::arch::x86_64::{
        __m512i, __mmask16, _CMP_GE_OQ, _mm_mask_storeu_epi8, _mm256_mask_storeu_epi16,
        _mm512_add_epi32, _mm512_add_ps, _mm512_and_si512, _mm512_cmp_ps_mask,
        _mm512_cmpeq_epi32_mask, _mm512_cmpge_epi32_mask, _mm512_cmple_epi32_mask,
        _mm512_cvt_roundps_epi32, _mm512_cvtepi32_epi8, _mm512_cvtepi32_epi16, _mm512_cvtepi32_ps,
        _mm512_cvttps_epi32, _mm512_fmadd_ps, _mm512_mask_add_ps, _mm512_mask_i32gather_epi32,
        _mm512_mask_mov_epi32, _mm512_maskz_mov_epi32, _mm512_min_epi32, _mm512_mullo_epi32,
        _mm512_roundscale_ps, _mm512_set1_epi32, _mm512_set1_ps, _mm512_setr_ps,
        _mm512_setzero_si512, _mm512_slli_epi32, _mm512_srlv_epi32, _mm512_sub_epi32,
        _mm512_sub_ps,
    };

    let data = src.data();
    let len = size_of_val(data);
    let Ok(len_i32) = i32::try_from(len) else {
        rotate_span_scalar(dst_row, 0, src, p);
        return;
    };
    if len < 4 {
        // Too small for a dword gather to stay inside the buffer.
        rotate_span_scalar(dst_row, 0, src, p);
        return;
    }
    let base = data.as_ptr().cast::<u8>();
    let w = dst_row.len();
    let out = dst_row.as_mut_ptr();
    let size = size_of::<P::Sample>();

    // The samples of the low 16 lanes of `v`, which are at most
    // `Sample::MAX`, at `out + ox` for the lanes of `k`.
    let store = |ox: usize, k: __mmask16, v: __m512i| {
        // SAFETY: the caller's lanes `k` lie in `ox .. w`, inside `dst_row`;
        // each masked store writes those lanes' samples only.
        unsafe {
            if P::Sample::BITS == 8 {
                _mm_mask_storeu_epi8(out.add(ox).cast(), k, _mm512_cvtepi32_epi8(v));
            } else {
                _mm256_mask_storeu_epi16(out.add(ox).cast(), k, _mm512_cvtepi32_epi16(v));
            }
        }
    };

    let lane = _mm512_setr_ps(
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
    );
    let v_cos = _mm512_set1_ps(p.cos_a);
    let v_sin = _mm512_set1_ps(p.sin_a);
    let v_sx_base = _mm512_set1_ps(p.sx_base);
    let v_sy_base = _mm512_set1_ps(p.sy_base);
    let v_sx_max = _mm512_set1_epi32(p.sx_max);
    let v_sy_max = _mm512_set1_epi32(p.sy_max);
    let v_zero = _mm512_setzero_si512();
    let v_one = _mm512_set1_epi32(1);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "Sample::BITS is 8 or 16 and Sample::MAX at most 65535"
    )]
    let (v_bits, v_sample) = (
        _mm512_set1_epi32(P::Sample::BITS as i32),
        _mm512_set1_epi32(P::Sample::MAX.signed() as i32),
    );
    let v_half = _mm512_set1_ps(0.5);
    let v_one_f = _mm512_set1_ps(1.0);
    // The row stride in bytes is at most len ≤ i32::MAX; a sample is 1 or 2 bytes.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "stride · size ≤ len ≤ i32::MAX and size ≤ 2"
    )]
    let (v_stride, v_size, v_len_m4) = (
        _mm512_set1_epi32((src.stride * size) as i32),
        _mm512_set1_epi32(size as i32),
        _mm512_set1_epi32(len_i32 - 4),
    );

    let mut ox = 0usize;
    while ox < w {
        let take = (w - ox).min(16);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "take ≤ 16 → mask fits in 16 bits"
        )]
        let k: __mmask16 = ((1u32 << take) - 1) as __mmask16;

        // sx = cos·dx + sx_base, sy = sin·dx + sy_base, dx = ox + lane.
        // ox ≤ DESKEW_MAX_SIDE = 32768; exact in f32.
        #[expect(
            clippy::cast_precision_loss,
            reason = "ox ≤ DESKEW_MAX_SIDE = 32768; exact in f32"
        )]
        let dx = _mm512_add_ps(_mm512_set1_ps(ox as f32), lane);
        let sx = _mm512_fmadd_ps(v_cos, dx, v_sx_base);
        let sy = _mm512_fmadd_ps(v_sin, dx, v_sy_base);
        let x0 = _mm512_cvt_roundps_epi32::<FLOOR>(sx);
        let y0 = _mm512_cvt_roundps_epi32::<FLOOR>(sy);

        let in_bounds = k
            & _mm512_cmpge_epi32_mask(x0, v_zero)
            & _mm512_cmpge_epi32_mask(y0, v_zero)
            & _mm512_cmple_epi32_mask(x0, v_sx_max)
            & _mm512_cmple_epi32_mask(y0, v_sy_max);
        if in_bounds == 0 {
            store(ox, k, v_sample);
            ox += 16;
            continue;
        }

        let fx = _mm512_sub_ps(sx, _mm512_cvtepi32_ps(x0));
        let fy = _mm512_sub_ps(sy, _mm512_cvtepi32_ps(y0));

        // Out-of-bounds lanes are zeroed so every offset below is a valid
        // index even though the gather skips them.
        let x0 = _mm512_maskz_mov_epi32(in_bounds, x0);
        let y0 = _mm512_maskz_mov_epi32(in_bounds, y0);
        let y1 = _mm512_min_epi32(_mm512_add_epi32(y0, v_one), v_sy_max);
        let last_col = _mm512_cmpeq_epi32_mask(x0, v_sx_max);

        // Byte offsets of the origins: y · stride · size + x · size < len.
        let x0_bytes = _mm512_mullo_epi32(x0, v_size);
        let off0 = _mm512_add_epi32(_mm512_mullo_epi32(y0, v_stride), x0_bytes);
        let off1 = _mm512_add_epi32(_mm512_mullo_epi32(y1, v_stride), x0_bytes);
        let off0_c = _mm512_min_epi32(off0, v_len_m4);
        let off1_c = _mm512_min_epi32(off1, v_len_m4);
        let sh0 = _mm512_slli_epi32::<3>(_mm512_sub_epi32(off0, off0_c));
        let sh1 = _mm512_slli_epi32::<3>(_mm512_sub_epi32(off1, off1_c));

        // SAFETY: every active lane's offset is ≤ len − 4, so each dword read
        // lies within `data`; inactive lanes are not read.
        let (d0, d1) = unsafe {
            (
                _mm512_mask_i32gather_epi32::<1>(v_zero, in_bounds, off0_c, base.cast()),
                _mm512_mask_i32gather_epi32::<1>(v_zero, in_bounds, off1_c, base.cast()),
            )
        };
        // A shift of 32 or more bits, past the dword, gives 0; such a lane
        // is on the last column, whose neighbour is replaced below.
        let p00 = _mm512_and_si512(_mm512_srlv_epi32(d0, sh0), v_sample);
        let p01 = _mm512_and_si512(_mm512_srlv_epi32(d1, sh1), v_sample);
        let p10 = _mm512_mask_mov_epi32(
            _mm512_and_si512(
                _mm512_srlv_epi32(d0, _mm512_add_epi32(sh0, v_bits)),
                v_sample,
            ),
            last_col,
            p00,
        );
        let p11 = _mm512_mask_mov_epi32(
            _mm512_and_si512(
                _mm512_srlv_epi32(d1, _mm512_add_epi32(sh1, v_bits)),
                v_sample,
            ),
            last_col,
            p01,
        );

        let (f00, f10, f01, f11) = (
            _mm512_cvtepi32_ps(p00),
            _mm512_cvtepi32_ps(p10),
            _mm512_cvtepi32_ps(p01),
            _mm512_cvtepi32_ps(p11),
        );
        let top = _mm512_fmadd_ps(_mm512_sub_ps(f10, f00), fx, f00);
        let bot = _mm512_fmadd_ps(_mm512_sub_ps(f11, f01), fx, f01);
        let val = _mm512_fmadd_ps(_mm512_sub_ps(bot, top), fy, top);

        let trunc = _mm512_roundscale_ps::<TRUNC>(val);
        let round_up = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(_mm512_sub_ps(val, trunc), v_half);
        let rounded = _mm512_cvttps_epi32(_mm512_mask_add_ps(trunc, round_up, trunc, v_one_f));
        store(ox, k, _mm512_mask_mov_epi32(v_sample, in_bounds, rounded));
        ox += 16;
    }
}

/// One output row through the fastest kernel of this machine.
#[cfg(target_arch = "aarch64")]
#[expect(
    unsafe_code,
    reason = "NEON intrinsics for the aarch64 rotate row kernel; mandatory on ARMv8-A"
)]
#[inline]
pub(super) fn dispatch_rotate_row<P: DeskewPixel>(
    dst_row: &mut [P::Sample],
    src: &Bitmap<P>,
    p: &RowParams,
) {
    // SAFETY: NEON mandatory on all ARMv8-A targets.
    unsafe { rotate_row_neon(dst_row, src, p) }
}

/// One output row through the fastest kernel of this machine.
#[cfg(target_arch = "x86_64")]
#[expect(
    unsafe_code,
    reason = "AVX-512 intrinsics for the x86-64 rotate row kernel; runtime-detected here"
)]
#[inline]
pub(super) fn dispatch_rotate_row<P: DeskewPixel>(
    dst_row: &mut [P::Sample],
    src: &Bitmap<P>,
    p: &RowParams,
) {
    if has_avx512() {
        // SAFETY: all three features confirmed present.
        unsafe { rotate_row_avx512(dst_row, src, p) }
    } else {
        rotate_span_scalar(dst_row, 0, src, p);
    }
}

/// One output row through the scalar kernel, on targets without a SIMD tier.
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline]
pub(super) fn dispatch_rotate_row<P: DeskewPixel>(
    dst_row: &mut [P::Sample],
    src: &Bitmap<P>,
    p: &RowParams,
) {
    rotate_span_scalar(dst_row, 0, src, p);
}

/// Whether the CPU has the features of [`rotate_row_avx512`].
#[cfg(target_arch = "x86_64")]
fn has_avx512() -> bool {
    is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vl")
}

#[cfg(test)]
mod tests {
    use color::{Gray8, Gray16, Sample};
    use raster::Bitmap;

    use super::super::DeskewPixel;
    use super::super::bilinear::{RowParams, rotate_cpu_with, rotate_span_scalar};
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    use super::*;

    /// Source sizes for the parity tests: 1–15-pixel tails after the 8- and
    /// 16-wide groups, a 4×4 image (gather clamp active on every row), a
    /// 3-row strip, 1-pixel strips and a 1×1 image (shorter than a dword
    /// at 8 bits).
    const PARITY_SIZES: [(u32, u32); 11] = [
        (33, 16),
        (32, 32),
        (30, 8),
        (7, 5),
        (100, 3),
        (4, 4),
        (257, 129),
        (1, 1),
        (3, 1),
        (1, 3),
        (2, 1),
    ];

    /// Angles for the parity tests: the identity, edge-boundary lanes,
    /// mixed in/out-of-bounds groups, exact-half bilinear fractions (180°)
    /// and negative skew.
    const PARITY_ANGLES: [f32; 9] = [0.0, 0.1, 1.5, 5.0, 45.0, 89.0, 180.0, -3.3, 12.34];

    /// A `w × h` source whose samples cover the whole range of `P`, with
    /// row padding `pad`, so the last sample of the buffer is padding for
    /// `pad > 1`.
    fn pattern<P: DeskewPixel>(w: u32, h: u32, pad: usize) -> Bitmap<P> {
        let mut src = Bitmap::<P>::new(w, h, pad, false);
        let modulus = P::Sample::MAX.signed() + 1;
        for y in 0..h {
            for (x, s) in src.row_samples_mut(y).iter_mut().enumerate() {
                #[expect(clippy::cast_possible_wrap, reason = "x < 2^9, far inside i64")]
                let v = (x as i64 * 7919 + i64::from(y) * 104_729) % modulus;
                *s = P::Sample::from_signed(v);
            }
        }
        src
    }

    /// Asserts that `row_fn` reproduces the scalar kernel sample for sample
    /// over [`PARITY_SIZES`] × [`PARITY_ANGLES`], unpadded and padded.
    fn assert_rows_match_scalar<P: DeskewPixel>(
        kernel: &str,
        row_fn: impl Fn(&mut [P::Sample], &Bitmap<P>, &RowParams) + Copy,
    ) {
        for pad in [1, 8] {
            for &(w, h) in &PARITY_SIZES {
                let src = pattern::<P>(w, h, pad);
                for &angle in &PARITY_ANGLES {
                    let scalar = rotate_cpu_with(&src, angle, |row, s, p| {
                        rotate_span_scalar(row, 0, s, p);
                    });
                    let simd = rotate_cpu_with(&src, angle, row_fn);
                    assert_eq!(
                        scalar.data(),
                        simd.data(),
                        "{kernel} {} bits: w={w} h={h} pad={pad} angle={angle}",
                        P::Sample::BITS
                    );
                }
            }
        }
    }

    /// The NEON row kernel vs the scalar reference, at 8 and 16 bits.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_rotate_rows_match_scalar_reference() {
        fn check<P: DeskewPixel>() {
            #[expect(
                unsafe_code,
                reason = "direct NEON kernel invocation under test; NEON is mandatory on ARMv8-A"
            )]
            assert_rows_match_scalar::<P>("neon", |row, s, p| {
                // SAFETY: NEON mandatory on all ARMv8-A targets.
                unsafe { rotate_row_neon(row, s, p) }
            });
        }
        check::<Gray8>();
        check::<Gray16>();
    }

    /// The AVX-512 row kernel vs the scalar reference, at 8 and 16 bits.
    /// Runtime-skipped without AVX-512.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_rotate_rows_match_scalar_reference() {
        fn check<P: DeskewPixel>() {
            #[expect(
                unsafe_code,
                reason = "direct AVX-512 kernel invocation under test; features confirmed by the caller"
            )]
            assert_rows_match_scalar::<P>("avx512", |row, s, p| {
                // SAFETY: avx512f + avx512bw + avx512vl confirmed present by the caller.
                unsafe { rotate_row_avx512(row, s, p) }
            });
        }
        if !has_avx512() {
            return;
        }
        check::<Gray8>();
        check::<Gray16>();
    }

    /// The tier this machine dispatches to vs the scalar reference.
    #[test]
    fn dispatched_rotate_rows_match_scalar_reference() {
        assert_rows_match_scalar::<Gray8>("dispatch", super::dispatch_rotate_row);
        assert_rows_match_scalar::<Gray16>("dispatch", super::dispatch_rotate_row);
    }

    /// Timing smoke test of the 16-bit rotation on 2900×2900 at 1.5°,
    /// through the dispatched tier and the scalar loop.  Always passes; run
    /// with `--ignored --nocapture` in a release build to see the times.
    #[test]
    #[ignore = "manual perf smoke test; run with `--ignored --nocapture` for timing"]
    fn rotate_gray16_8mp_timing() {
        let src = pattern::<Gray16>(2900, 2900, 1);
        let time = |name: &str, f: &dyn Fn() -> Bitmap<Gray16>| {
            let _ = f();
            let t0 = std::time::Instant::now();
            for _ in 0..10 {
                drop(std::hint::black_box(f()));
            }
            println!(
                "{name}: {:.2} ms",
                t0.elapsed().as_secs_f64() * 1000.0 / 10.0
            );
        };
        time("dispatched", &|| {
            rotate_cpu_with(&src, 1.5, super::dispatch_rotate_row)
        });
        time("scalar", &|| {
            rotate_cpu_with(&src, 1.5, |row, s, p| rotate_span_scalar(row, 0, s, p))
        });
    }
}
