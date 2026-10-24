//! Image rotation for deskew correction.
//!
//! Rotates an 8-bit grayscale bitmap by a given angle (in degrees, CW positive)
//! using bilinear interpolation.  Background pixels introduced by the rotation
//! are set to 255 (white — scanner background convention).
//!
//! # Implementation
//!
//! **CPU path** (always available): per-output-pixel inverse mapping with
//! bilinear interpolation.  The inner loop is written in the per-arch dispatch
//! pattern used throughout the codebase:
//!
//! - **x86-64 (AVX-512 F+BW+VL, runtime-detected)**: `rotate_row_avx512`
//!   processes 16 output pixels per iteration end to end — FMA coordinate
//!   stepping, a lane mask for the bounds test, two dword gathers for the
//!   2×2 footprints, FMA bilinear blend, and a white fill merged in for
//!   out-of-bounds lanes.  Bit-exact with the scalar loop.
//! - **aarch64**: `rotate_row_neon` processes 8 output pixels per iteration
//!   as two 4-lane quads — fused coordinate stepping, floor, an unsigned
//!   bounds mask, fused bilinear blend and a white merge are lane-wise; only
//!   the 2×2 footprint fetch is per lane (NEON has no gather).  Bit-exact
//!   with the scalar loop.
//! - **other**: scalar per-pixel loop.  Its per-pixel `floor`, early return
//!   and gathered 2×2 reads keep LLVM from auto-vectorising it.
//!
//! **GPU path** (`gpu-deskew` feature, optional): CUDA NPP `nppiRotate_8u_C1R_Ctx`
//! with hardware bilinear via texture units.  ~0.3–0.5 ms for a 2550×3300 image
//! on RTX 5070.  Falls back to CPU silently when no CUDA device is available.

use color::Gray8;
use raster::Bitmap;

use super::DeskewError;

/// Rotate `img` in-place by `angle_deg` degrees (clockwise positive).
///
/// The image is replaced with a new bitmap of the same dimensions containing
/// the rotated content.  Out-of-bounds source pixels are filled with 255 (white).
///
/// When the `gpu-deskew` feature is enabled, attempts GPU rotation first and
/// falls back to the CPU bilinear path on any GPU error.
///
/// # Errors
///
/// Currently always returns `Ok(())` — the CPU path cannot fail beyond OOM
/// (which panics).  The return type is `Result` to accommodate future GPU-only
/// configurations where CPU fallback may not be available.
#[expect(
    clippy::unnecessary_wraps,
    reason = "Result kept for future GPU-only configurations where CPU fallback may not be available"
)]
pub fn rotate_inplace(img: &mut Bitmap<Gray8>, angle_deg: f32) -> Result<(), DeskewError> {
    #[cfg(feature = "gpu-deskew")]
    match rotate_gpu(img, angle_deg) {
        Ok(rotated) => {
            *img = rotated;
            return Ok(());
        }
        Err(e) => {
            log::warn!("rotate_inplace: GPU rotation failed ({e}); falling back to CPU");
        }
    }

    *img = rotate_cpu(img, angle_deg);
    Ok(())
}

// ── Shared helpers ────────────────────────────────────────────────────────────

/// Parameters shared between the scalar and SIMD row kernels.
///
/// Passed by value to avoid threading individual scalars through dispatch helpers.
struct RowParams {
    cos_a: f32,
    sin_a: f32,
    sx_base: f32,
    sy_base: f32,
    sx_max: i32,
    sy_max: i32,
}

/// The 2×2 bilinear footprint `[p00, p10, p01, p11]` at origin `(x0, y0)`.
///
/// `x0` and `y0` must already be bounds-checked (`0 ≤ x0 ≤ sx_max = w-1`,
/// `0 ≤ y0 ≤ sy_max = h-1`). On the last column/row the footprint is clamped
/// to the edge, so the missing neighbour duplicates the edge pixel and the
/// interpolation degrades gracefully to nearest-neighbour there.
#[inline]
fn footprint(src: &Bitmap<Gray8>, x0: usize, y0: usize) -> [u8; 4] {
    let x1 = (x0 + 1).min(src.width as usize - 1);
    let y1 = (y0 + 1).min(src.height as usize - 1);
    // y0 ≤ sy_max = h-1 ≤ 32767 and y1 ≤ h-1 — safe casts.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "y0 ≤ h-1 ≤ 32767 and y1 ≤ h-1; both fit in u32"
    )]
    let (r0, r1) = (src.row_bytes(y0 as u32), src.row_bytes(y1 as u32));
    [r0[x0], r0[x1], r1[x0], r1[x1]]
}

/// Bilinear interpolation at source `(sx, sy)` given pre-floored origin `(x0, y0)`
/// and fractions `(fx, fy)`; the origin contract is that of [`footprint`].
#[inline]
fn bilinear_sample(src: &Bitmap<Gray8>, x0: usize, y0: usize, fx: f32, fy: f32) -> u8 {
    let [p00, p10, p01, p11] = footprint(src, x0, y0).map(f32::from);

    let top = (p10 - p00).mul_add(fx, p00);
    let bot = (p11 - p01).mul_add(fx, p01);
    let val = (bot - top).mul_add(fy, top);

    // Bilinear of values in [0, 255] stays in [0, 255].
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "bilinear of [0,255] inputs stays in [0,255]; rounding keeps it non-negative"
    )]
    {
        val.round() as u8
    }
}

/// Compute a single output pixel at column `ox`.
///
/// Returns 255 (white) when the source coordinates are out of the valid
/// bilinear-sampling range.
#[inline]
fn rotate_pixel_scalar(src: &Bitmap<Gray8>, ox: f32, p: &RowParams) -> u8 {
    let sx = p.cos_a.mul_add(ox, p.sx_base);
    let sy = p.sin_a.mul_add(ox, p.sy_base);

    // floor() → i32: coordinates bounded by ≤ 32768·√2 ≈ 46341 — fits in i32.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "image coordinates bounded by rotated extent ≤ 32768 * sqrt(2) ≈ 46341; fits in i32"
    )]
    let x0 = sx.floor() as i32;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "image coordinates bounded by rotated extent ≤ 32768 * sqrt(2) ≈ 46341; fits in i32"
    )]
    let y0 = sy.floor() as i32;

    if x0 < 0 || y0 < 0 || x0 > p.sx_max || y0 > p.sy_max {
        return 255;
    }

    // x0 ≥ 0, y0 ≥ 0 verified above.
    #[expect(
        clippy::cast_sign_loss,
        reason = "x0 ≥ 0 and y0 ≥ 0 verified by the guard above"
    )]
    let (x0u, y0u) = (x0 as usize, y0 as usize);
    // x0 ≤ 46341 ≤ 2^23 — exactly representable in f32; subtraction is exact.
    #[expect(
        clippy::cast_precision_loss,
        reason = "x0 ≤ 46341 ≤ 2^23 — exactly representable in f32"
    )]
    let fx = sx - x0 as f32;
    #[expect(
        clippy::cast_precision_loss,
        reason = "y0 ≤ 46341 ≤ 2^23 — exactly representable in f32"
    )]
    let fy = sy - y0 as f32;

    bilinear_sample(src, x0u, y0u, fx, fy)
}

// ── Per-arch row implementations ──────────────────────────────────────────────

/// NEON inner loop: 8 output pixels per iteration as two 4-lane quads,
/// bit-exact with [`rotate_row_scalar`].
///
/// Coordinates are stepped with fused multiply-add exactly as the scalar
/// path does (`cos·ox + sx_base`), floored with `vcvtmq_s32_f32`, and
/// bounds-tested as unsigned integers so a negative origin compares above
/// the maximum.  NEON has no gather, so each in-bounds lane fetches its 2×2
/// footprint through [`footprint`]; out-of-bounds lanes fetch at origin
/// `(0, 0)` and are replaced by 255 afterwards.  The bilinear blend is the
/// scalar `mul_add` chain lane-wise (`vfmaq_f32`), `vcvtaq_u32_f32` rounds
/// half away from zero as `f32::round` does, and saturating narrows mirror
/// the `as u8` cast.  A quad whose lanes are all out of bounds skips the
/// fetch, and the final `dst_row.len() % 8` pixels take the scalar path.
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
unsafe fn rotate_row_neon(dst_row: &mut [u8], src: &Bitmap<Gray8>, p: &RowParams) {
    use std::arch::aarch64::{
        float32x4_t, uint32x4_t, vaddq_f32, vandq_u32, vbslq_u32, vcleq_u32, vcombine_u16,
        vcvtaq_u32_f32, vcvtmq_s32_f32, vcvtq_f32_s32, vcvtq_f32_u32, vdupq_n_f32, vdupq_n_u32,
        vfmaq_f32, vgetq_lane_u32, vld1q_f32, vld1q_u32, vmaxvq_u32, vqmovn_u16, vqmovn_u32,
        vreinterpretq_u32_s32, vst1_u8, vsubq_f32,
    };

    /// Lane-wise `[a0, a1, a2, a3]` as `f32` from four `u8` values.
    #[inline]
    #[target_feature(enable = "neon")]
    fn widen(a: [u8; 4]) -> float32x4_t {
        let a = a.map(u32::from);
        // SAFETY: `a` is four u32 values, the extent of one vld1q_u32.
        vcvtq_f32_u32(unsafe { vld1q_u32(a.as_ptr()) })
    }

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
    let v_white = vdupq_n_u32(255);
    // SAFETY: four f32 values, the extent of one vld1q_f32.
    let lane = unsafe { vld1q_f32([0.0f32, 1.0, 2.0, 3.0].as_ptr()) };

    // Output pixels for columns `ox .. ox + 4`, white where out of bounds.
    let quad = |ox: usize| -> uint32x4_t {
        // ox ≤ MAX_PX_DIMENSION = 32768; exact in f32.
        #[expect(
            clippy::cast_precision_loss,
            reason = "ox ≤ MAX_PX_DIMENSION = 32768; exact in f32"
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
        let corner = |c: usize| widen([f[0][c], f[1][c], f[2][c], f[3][c]]);
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
        let px = vqmovn_u16(vcombine_u16(vqmovn_u32(quad(ox)), vqmovn_u32(quad(ox + 4))));
        // SAFETY: `out` is eight bytes, the extent of one vst1_u8.
        unsafe { vst1_u8(out.as_mut_ptr(), px) };
    }
    rotate_span_scalar(tail, tail_start, src, p);
}

/// AVX-512 inner loop: 16 output pixels per iteration, bit-exact with
/// [`rotate_row_scalar`].
///
/// Source coordinates are stepped with FMA exactly as the scalar path does
/// (`cos·ox + sx_base`), floored with a round-toward-−∞ conversion, and
/// bounds-tested as integers into a lane mask.  Each in-bounds lane fetches
/// its 2×2 footprint with two dword gathers — row `y0` and row `y1` at column
/// `x0` — whose second byte is the `x0 + 1` neighbour; on the last column that
/// byte is replaced by the `x0` sample, mirroring `bilinear_sample`'s edge
/// clamp.  Gather offsets are clamped to `len − 4` and the wanted byte is
/// shifted down, so no dword read crosses the end of the pixel buffer.
/// Out-of-bounds lanes gather nothing and are merged to 255 before the
/// store.  `round()` is reproduced as `trunc + (frac ≥ 0.5)`, which equals
/// round-half-away-from-zero on the non-negative bilinear result.
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
unsafe fn rotate_row_avx512(dst_row: &mut [u8], src: &Bitmap<Gray8>, p: &RowParams) {
    use std::arch::x86_64::{_MM_FROUND_NO_EXC, _MM_FROUND_TO_NEG_INF, _MM_FROUND_TO_ZERO};
    const FLOOR: i32 = _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC;
    const TRUNC: i32 = _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC;
    use std::arch::x86_64::{
        __mmask16, _CMP_GE_OQ, _mm_mask_storeu_epi8, _mm_set1_epi8, _mm_storeu_si128,
        _mm512_add_epi32, _mm512_add_ps, _mm512_and_si512, _mm512_cmp_ps_mask,
        _mm512_cmpeq_epi32_mask, _mm512_cmpge_epi32_mask, _mm512_cmple_epi32_mask,
        _mm512_cvt_roundps_epi32, _mm512_cvtepi32_epi8, _mm512_cvtepi32_ps, _mm512_cvttps_epi32,
        _mm512_fmadd_ps, _mm512_mask_add_ps, _mm512_mask_i32gather_epi32, _mm512_mask_mov_epi32,
        _mm512_maskz_mov_epi32, _mm512_min_epi32, _mm512_mullo_epi32, _mm512_roundscale_ps,
        _mm512_set1_epi32, _mm512_set1_ps, _mm512_setr_ps, _mm512_setzero_si512, _mm512_slli_epi32,
        _mm512_srlv_epi32, _mm512_sub_epi32, _mm512_sub_ps,
    };

    let data = src.data();
    let len = data.len();
    if len < 4 {
        // Too small for a dword gather to stay inside the buffer.
        rotate_row_scalar(dst_row, src, p);
        return;
    }
    let base = data.as_ptr();
    let w = dst_row.len();
    let out = dst_row.as_mut_ptr();

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
    let v_eight = _mm512_set1_epi32(8);
    let v_byte = _mm512_set1_epi32(0xFF);
    let v_white = _mm512_set1_epi32(255);
    let v_half = _mm512_set1_ps(0.5);
    let v_one_f = _mm512_set1_ps(1.0);
    // stride ≤ MAX_PX_DIMENSION rounded up to the row pad; len ≤ 2^30: both fit in i32.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "stride and len are bounded by MAX_PX_DIMENSION² ≤ 2^30"
    )]
    let (v_stride, v_len_m4) = (
        _mm512_set1_epi32(src.stride as i32),
        _mm512_set1_epi32((len - 4) as i32),
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
        // ox ≤ MAX_PX_DIMENSION = 32768; exact in f32.
        #[expect(
            clippy::cast_precision_loss,
            reason = "ox ≤ MAX_PX_DIMENSION = 32768; exact in f32"
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
            // SAFETY: ox + take ≤ w = dst_row.len(); the masked store writes `take` bytes.
            unsafe {
                if take == 16 {
                    _mm_storeu_si128(out.add(ox).cast(), _mm_set1_epi8(-1));
                } else {
                    _mm_mask_storeu_epi8(out.add(ox).cast(), k, _mm_set1_epi8(-1));
                }
            }
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

        let off0 = _mm512_add_epi32(_mm512_mullo_epi32(y0, v_stride), x0);
        let off1 = _mm512_add_epi32(_mm512_mullo_epi32(y1, v_stride), x0);
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
        let p00 = _mm512_and_si512(_mm512_srlv_epi32(d0, sh0), v_byte);
        let p01 = _mm512_and_si512(_mm512_srlv_epi32(d1, sh1), v_byte);
        let p10 = _mm512_mask_mov_epi32(
            _mm512_and_si512(
                _mm512_srlv_epi32(d0, _mm512_add_epi32(sh0, v_eight)),
                v_byte,
            ),
            last_col,
            p00,
        );
        let p11 = _mm512_mask_mov_epi32(
            _mm512_and_si512(
                _mm512_srlv_epi32(d1, _mm512_add_epi32(sh1, v_eight)),
                v_byte,
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
        let pixels = _mm512_cvtepi32_epi8(_mm512_mask_mov_epi32(v_white, in_bounds, rounded));

        // SAFETY: ox + take ≤ w = dst_row.len(); the masked store writes `take` bytes.
        unsafe {
            if take == 16 {
                _mm_storeu_si128(out.add(ox).cast(), pixels);
            } else {
                _mm_mask_storeu_epi8(out.add(ox).cast(), k, pixels);
            }
        }
        ox += 16;
    }
}

/// Scalar per-pixel row implementation.
///
/// Targets without a SIMD tier use it directly, the AVX-512 tier falls back
/// to it, and it is the reference the per-row parity tests compare against.
#[cfg(any(test, not(target_arch = "aarch64")))]
fn rotate_row_scalar(dst_row: &mut [u8], src: &Bitmap<Gray8>, p: &RowParams) {
    rotate_span_scalar(dst_row, 0, src, p);
}

/// Scalar per-pixel loop over output columns `first .. first + span.len()`.
fn rotate_span_scalar(span: &mut [u8], first: usize, src: &Bitmap<Gray8>, p: &RowParams) {
    for (i, dst_px) in span.iter_mut().enumerate() {
        #[expect(
            clippy::cast_precision_loss,
            reason = "column ≤ MAX_PX_DIMENSION = 32768; exact in f32"
        )]
        {
            *dst_px = rotate_pixel_scalar(src, (first + i) as f32, p);
        }
    }
}

// ── Per-arch dispatch ─────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[expect(
    unsafe_code,
    reason = "NEON intrinsics for the aarch64 rotate row kernel; mandatory on ARMv8-A"
)]
#[inline]
fn dispatch_rotate_row(dst_row: &mut [u8], src: &Bitmap<Gray8>, p: &RowParams) {
    // SAFETY: NEON mandatory on all ARMv8-A targets.
    unsafe { rotate_row_neon(dst_row, src, p) }
}

#[cfg(target_arch = "x86_64")]
#[expect(
    unsafe_code,
    reason = "AVX-512 intrinsics for the x86-64 rotate row kernel; runtime-detected here"
)]
#[inline]
fn dispatch_rotate_row(dst_row: &mut [u8], src: &Bitmap<Gray8>, p: &RowParams) {
    if is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vl")
    {
        // SAFETY: all three features confirmed present.
        unsafe { rotate_row_avx512(dst_row, src, p) }
    } else {
        rotate_row_scalar(dst_row, src, p);
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline]
fn dispatch_rotate_row(dst_row: &mut [u8], src: &Bitmap<Gray8>, p: &RowParams) {
    rotate_row_scalar(dst_row, src, p);
}

// ── Public CPU path ───────────────────────────────────────────────────────────

/// CPU bilinear rotation (clockwise-positive).
///
/// Uses inverse mapping: for each output pixel `(ox, oy)`, compute the
/// corresponding source coordinates `(sx, sy)` by applying a CW rotation
/// matrix centred on the image centre, then bilinear-interpolate from the
/// four surrounding source pixels.
///
/// Out-of-bounds source pixels are filled with 255 (white).  The last valid
/// sample origin is (w-1, h-1); on that boundary `bilinear_sample` clamps
/// the 2×2 footprint to the edge.
pub fn rotate_cpu(src: &Bitmap<Gray8>, angle_deg: f32) -> Bitmap<Gray8> {
    rotate_cpu_with(src, angle_deg, dispatch_rotate_row)
}

/// Row-loop driver shared by [`rotate_cpu`] and the per-row parity tests:
/// computes the per-row inverse-map parameters and hands each output row to
/// `row_fn`.
#[expect(
    clippy::similar_names,
    reason = "sx_*/sy_* are paired coordinate variables; renaming would obscure the symmetry"
)]
fn rotate_cpu_with(
    src: &Bitmap<Gray8>,
    angle_deg: f32,
    row_fn: impl Fn(&mut [u8], &Bitmap<Gray8>, &RowParams),
) -> Bitmap<Gray8> {
    let w = src.width as usize;
    let h = src.height as usize;

    let mut dst = Bitmap::<Gray8>::new(src.width, src.height, 1, false);

    let rad = angle_deg.to_radians();
    let cos_a = rad.cos();
    let sin_a = rad.sin();

    // Rotation centre: image centre (may be fractional).
    // w, h ≤ MAX_PX_DIMENSION = 32768; exact in f32 (24-bit mantissa covers integers to 16 M).
    #[expect(
        clippy::cast_precision_loss,
        reason = "image dimensions ≤ MAX_PX_DIMENSION = 32768; exact in f32"
    )]
    let cx = (w as f32 - 1.0) * 0.5;
    #[expect(
        clippy::cast_precision_loss,
        reason = "image dimensions ≤ MAX_PX_DIMENSION = 32768; exact in f32"
    )]
    let cy = (h as f32 - 1.0) * 0.5;

    // Valid source coordinate range: x ∈ [0, w-1], y ∈ [0, h-1].
    // `bilinear_sample` clamps its 2×2 footprint to the edge, so the last
    // column/row are sampleable rather than white-filled.
    // w, h ≤ 32768 — safe to cast to i32 (fits in i32 range).
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "image dimensions ≤ MAX_PX_DIMENSION = 32768; fits in i32"
    )]
    let sx_max = (w as i32) - 1;
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "image dimensions ≤ MAX_PX_DIMENSION = 32768; fits in i32"
    )]
    let sy_max = (h as i32) - 1;

    for oy in 0..h {
        // oy < h ≤ 32768; safe to cast to u32.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "oy < h ≤ MAX_PX_DIMENSION = 32768; fits in u32"
        )]
        let dst_row = dst.row_bytes_mut(oy as u32);

        // Pre-compute y-dependent terms outside the x loop.
        // oy ≤ 32768; exact in f32.
        #[expect(
            clippy::cast_precision_loss,
            reason = "oy ≤ MAX_PX_DIMENSION = 32768; exact in f32"
        )]
        let dy = oy as f32 - cy;
        let sx_base = cos_a.mul_add(-cx, -(sin_a * dy)) + cx;
        let sy_base = sin_a.mul_add(-cx, cos_a * dy) + cy;

        let params = RowParams {
            cos_a,
            sin_a,
            sx_base,
            sy_base,
            sx_max,
            sy_max,
        };
        row_fn(dst_row, src, &params);
    }

    dst
}

/// GPU rotation via CUDA NPP (`nppiRotate_8u_C1R_Ctx`).
#[cfg(feature = "gpu-deskew")]
fn rotate_gpu(src: &Bitmap<Gray8>, angle_deg: f32) -> Result<Bitmap<Gray8>, DeskewError> {
    let pixels =
        gpu::npp_rotate::rotate_gray8(src.data(), src.stride, src.width, src.height, angle_deg)
            .map_err(|e| DeskewError(e.to_string()))?;

    // Reconstruct a tightly-packed Bitmap from the returned pixel Vec.
    // rotate_gray8 returns width*height bytes with no stride padding.
    let mut dst = Bitmap::<Gray8>::new(src.width, src.height, 1, false);
    dst.data_mut().copy_from_slice(&pixels);
    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rotating by 0° is a no-op (pixels should be unchanged).
    #[test]
    fn rotate_zero_is_identity() {
        let mut src = Bitmap::<Gray8>::new(64, 64, 1, false);
        // Write a checkerboard pattern.
        for y in 0..64u32 {
            let row = src.row_bytes_mut(y);
            for (x, px) in row.iter_mut().enumerate() {
                *px = if (x + y as usize).is_multiple_of(2) {
                    0
                } else {
                    128
                };
            }
        }
        let rotated = rotate_cpu(&src, 0.0);
        // Centre pixels should be identical; edges may differ (white fill vs original).
        let r_src = &src.row_bytes(32)[1..63];
        let r_dst = &rotated.row_bytes(32)[1..63];
        assert_eq!(r_src, r_dst, "centre row should be unchanged at 0°");

        // Pin the OOB-guard strict-inequality boundaries: at 0° the inverse
        // map is identity, so output `(0, y)` samples source `x0 = 0` and
        // output `(sx_max, y)` samples `x0 = sx_max = w - 1 = 63`.  If the
        // guard `<` → `<=`, x0=0 wrongly becomes OOB → 255.  If `>` → `>=`,
        // x0=63 wrongly becomes OOB → 255.
        assert_eq!(
            rotated.row_bytes(32)[0],
            src.row_bytes(32)[0],
            "x0 = 0 must be in-bounds (guards `x0 < 0`, not `x0 <= 0`)"
        );
        assert_eq!(
            rotated.row_bytes(32)[63],
            src.row_bytes(32)[63],
            "x0 = sx_max must be in-bounds (guards `x0 > sx_max`, not `x0 >= sx_max`)"
        );
        // And the y-axis boundaries: row 0 and row 63.
        assert_eq!(
            rotated.row_bytes(0)[32],
            src.row_bytes(0)[32],
            "y0 = 0 must be in-bounds (guards `y0 < 0`, not `y0 <= 0`)"
        );
        assert_eq!(
            rotated.row_bytes(63)[32],
            src.row_bytes(63)[32],
            "y0 = sy_max must be in-bounds (guards `y0 > sy_max`, not `y0 >= sy_max`)"
        );
    }

    /// At 0° the inverse map is the identity, so every pixel — including
    /// the last row and column — must survive: edge-clamped bilinear
    /// sampling degrades to nearest-neighbour exactly on the boundary
    /// instead of white-filling it.
    #[test]
    fn rotate_zero_preserves_last_row_and_column() {
        let mut src = Bitmap::<Gray8>::new(64, 64, 1, false);
        for y in 0..64u32 {
            let row = src.row_bytes_mut(y);
            for (x, px) in row.iter_mut().enumerate() {
                *px = if (x + y as usize).is_multiple_of(2) {
                    0
                } else {
                    128
                };
            }
        }
        let rotated = rotate_cpu(&src, 0.0);
        for y in 0..64u32 {
            assert_eq!(
                src.row_bytes(y),
                rotated.row_bytes(y),
                "row {y} must be unchanged at 0°"
            );
        }
    }

    /// Source sizes for the SIMD row-kernel parity tests: 1–15-pixel tails
    /// after the 8- and 16-wide groups, a 4×4 image (16-byte buffer, dword
    /// gather clamp active on every row), a 3-row strip, and a 1×1 image.
    const PARITY_SIZES: [(u32, u32); 8] = [
        (33, 16),
        (32, 32),
        (30, 8),
        (7, 5),
        (100, 3),
        (4, 4),
        (257, 129),
        (1, 1),
    ];

    /// Angles for the parity tests: the identity, edge-boundary lanes,
    /// mixed in/out-of-bounds groups, exact-half bilinear fractions (180°)
    /// and negative skew.
    const PARITY_ANGLES: [f32; 9] = [0.0, 0.1, 1.5, 5.0, 45.0, 89.0, 180.0, -3.3, 12.34];

    /// Asserts that `row_fn` reproduces [`rotate_row_scalar`] byte-for-byte
    /// over [`PARITY_SIZES`] × [`PARITY_ANGLES`].
    fn assert_rows_match_scalar(
        kernel: &str,
        row_fn: impl Fn(&mut [u8], &Bitmap<Gray8>, &RowParams) + Copy,
    ) {
        for &(w, h) in &PARITY_SIZES {
            let mut src = Bitmap::<Gray8>::new(w, h, 1, false);
            for y in 0..h {
                let row = src.row_bytes_mut(y);
                for (x, px) in row.iter_mut().enumerate() {
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "mod-256 result fits in u8 by construction"
                    )]
                    {
                        *px = ((x * 31 + y as usize * 97) % 256) as u8;
                    }
                }
            }
            for &angle in &PARITY_ANGLES {
                let scalar = rotate_cpu_with(&src, angle, rotate_row_scalar);
                let simd = rotate_cpu_with(&src, angle, row_fn);
                for y in 0..h {
                    assert_eq!(
                        scalar.row_bytes(y),
                        simd.row_bytes(y),
                        "{kernel}: w={w} h={h} angle={angle} row {y}"
                    );
                }
            }
        }
    }

    /// NEON row kernel vs the scalar reference, byte-for-byte.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_rotate_rows_match_scalar_reference() {
        #[expect(
            unsafe_code,
            reason = "direct NEON kernel invocation under test; NEON is mandatory on ARMv8-A"
        )]
        assert_rows_match_scalar("neon", |row, s, p| {
            // SAFETY: NEON mandatory on all ARMv8-A targets.
            unsafe { rotate_row_neon(row, s, p) }
        });
    }

    /// AVX-512 row kernel vs the scalar reference, byte-for-byte.
    /// Runtime-skipped without AVX-512.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_rotate_rows_match_scalar_reference() {
        if !is_x86_feature_detected!("avx512f")
            || !is_x86_feature_detected!("avx512bw")
            || !is_x86_feature_detected!("avx512vl")
        {
            return;
        }
        #[expect(
            unsafe_code,
            reason = "direct AVX-512 kernel invocation under test; features confirmed above"
        )]
        assert_rows_match_scalar("avx512", |row, s, p| {
            // SAFETY: avx512f + avx512bw + avx512vl confirmed present above.
            unsafe { rotate_row_avx512(row, s, p) }
        });
    }

    /// A source buffer shorter than one dword cannot be gathered from; the
    /// kernel must hand such rows to the scalar path rather than read past
    /// the buffer.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_rotate_tiny_source_matches_scalar() {
        if !is_x86_feature_detected!("avx512f")
            || !is_x86_feature_detected!("avx512bw")
            || !is_x86_feature_detected!("avx512vl")
        {
            return;
        }
        for &(w, h) in &[(1u32, 1u32), (3, 1), (1, 3)] {
            let mut src = Bitmap::<Gray8>::new(w, h, 1, false);
            for y in 0..h {
                src.row_bytes_mut(y).fill(77);
            }
            let scalar = rotate_cpu_with(&src, 7.0, rotate_row_scalar);
            #[expect(
                unsafe_code,
                reason = "direct AVX-512 kernel invocation under test; features confirmed above"
            )]
            let simd = rotate_cpu_with(&src, 7.0, |row, s, p| {
                // SAFETY: avx512f + avx512bw + avx512vl confirmed present above.
                unsafe { rotate_row_avx512(row, s, p) }
            });
            for y in 0..h {
                assert_eq!(
                    scalar.row_bytes(y),
                    simd.row_bytes(y),
                    "w={w} h={h} row {y}"
                );
            }
        }
    }

    /// `rotate_inplace` at 0° does not panic.
    #[test]
    fn rotate_inplace_zero_no_panic() {
        let mut img = Bitmap::<Gray8>::new(32, 32, 1, false);
        rotate_inplace(&mut img, 0.0).unwrap();
    }

    /// `rotate_inplace` must actually mutate `*img`, not return Ok and leave
    /// the input untouched.  Pins the `rotate_inplace → Ok(())` mutant by
    /// asserting visible pixel content changes after a non-trivial rotation.
    #[test]
    fn rotate_inplace_modifies_pixels() {
        // 32×32 image with a single white-stripe column on the left half so
        // any non-no-op rotation visibly redistributes the white.
        let mut img = Bitmap::<Gray8>::new(32, 32, 1, false);
        for y in 0..32u32 {
            let row = img.row_bytes_mut(y);
            row[5] = 255; // vertical white line at x=5
        }
        let before_col5 = (0..32u32).map(|y| img.row_bytes(y)[5]).collect::<Vec<_>>();
        // 45° rotation breaks the vertical line into diagonal pieces.
        rotate_inplace(&mut img, 45.0).unwrap();
        let after_col5 = (0..32u32).map(|y| img.row_bytes(y)[5]).collect::<Vec<_>>();
        assert_ne!(
            before_col5, after_col5,
            "rotate_inplace at 45° must mutate the bitmap, not no-op"
        );
    }

    /// The centre pixel is the rotation pivot — at any angle it maps to
    /// itself, so the rotated image's centre must equal the source's centre
    /// (within bilinear-rounding tolerance).  This pins the centre-pivot
    /// shift formulae (`cx`, `cy` and the offset terms in `sx_base`/
    /// `sy_base`): any operator swap that mis-locates the pivot would shift
    /// the centre pixel away from its source value.
    ///
    /// Uses a centre-asymmetric pattern (single bright dot at the centre,
    /// black elsewhere) so any pivot drift onto a neighbouring black pixel
    /// changes the centre value dramatically.  A smooth gradient would let
    /// pivot-off-by-one mutations slip through because the bilinear samples
    /// of an off-by-one pivot still approximate the same gradient value.
    ///
    /// Tests three angles (12°, 31.7°, 89°) so a mutation that happens to
    /// preserve the pivot at one specific angle still fails at another.
    #[test]
    fn rotate_cpu_centre_pixel_is_rotation_fixed_point() {
        // Mid-grey dot at the centre (not 255) so the OOB-white fallback
        // (255) on a coordinate corruption is visibly different from a true
        // centre sample — catches mutations that produce NaN source
        // coordinates and silently route through the OOB guard.
        const DOT: u8 = 128;
        // Odd dimensions so cx, cy land on integer pixel centres (16.0).
        let mut src = Bitmap::<Gray8>::new(33, 33, 1, false);
        src.row_bytes_mut(16)[16] = DOT;

        for angle in [12.0_f32, 31.7, 89.0] {
            let rotated = rotate_cpu(&src, angle);
            let dst_centre = rotated.row_bytes(16)[16];
            // True bilinear sample at the rotated centre averages the DOT
            // pixel with some-fraction-of-zero neighbours, landing in
            // [~80, 128] depending on angle.  An off-by-one pivot would
            // sample 4 zero pixels (~0); a NaN-coordinate mutation would
            // OOB-fallback to 255.  Bracket both failure modes.
            assert!(
                (50..=200).contains(&dst_centre),
                "centre pixel must be a true bilinear sample of the centre dot at {angle}°: \
                 got dst={dst_centre} (expected ~50..=200); \
                 ~0 = pivot drift, 255 = NaN/OOB"
            );
        }
    }

    /// `rotate_cpu(rotate_cpu(S, 180), 180)` must equal S on the interior,
    /// modulo small bilinear-rounding error.  Catches any inner-loop
    /// arithmetic mutation that doesn't self-invert (e.g. `*` → `+` would
    /// break the double-rotation identity).
    #[test]
    fn rotate_cpu_180_then_180_round_trips() {
        let mut src = Bitmap::<Gray8>::new(32, 32, 1, false);
        for y in 0..32u32 {
            let row = src.row_bytes_mut(y);
            for (x, px) in row.iter_mut().enumerate() {
                // (x * 7 + y * 11) % 256 keeps the cast in [0, 255].
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "modulo 256 keeps the value in [0, u8::MAX]"
                )]
                {
                    *px = ((x * 7 + y as usize * 11) % 256) as u8;
                }
            }
        }

        let once = rotate_cpu(&src, 180.0);
        let twice = rotate_cpu(&once, 180.0);

        // Interior pixels only — borders accumulate two white-fill rounds.
        // Bilinear sampling at 180° hits half-pixel coordinates (cx/cy are
        // .5 in odd-sized images, 31.5 here), so a ±2 tolerance absorbs the
        // double-round error.
        let mut max_diff = 0i32;
        for y in 4..28u32 {
            let sr = &src.row_bytes(y)[4..28];
            let tr = &twice.row_bytes(y)[4..28];
            for (&a, &b) in sr.iter().zip(tr.iter()) {
                max_diff = max_diff.max((i32::from(a) - i32::from(b)).abs());
            }
        }
        assert!(
            max_diff <= 2,
            "180°×2 round-trip diverges from identity: max diff {max_diff} > 2"
        );
    }

    /// A 1-pixel-wide source rotated by a small angle must have white-fill
    /// (255) at the diagonal corners where the inverse mapping falls
    /// outside the source.  Pins the OOB-guard bounds checks
    /// (`x0 < 0 || y0 < 0 || x0 > sx_max || y0 > sy_max`) by exercising
    /// pixels that depend on the strict `>` / `<` comparison — if any
    /// boundary flips to `>=` / `<=`, an in-range pixel turns white or
    /// vice versa.
    #[test]
    fn rotate_cpu_oob_guard_fills_diagonal_with_white() {
        // 8×8 source, fully black (all zeros).  After a 5° rotation, the
        // four corner regions of the output map to OOB source coordinates
        // and get filled with white.
        let src = Bitmap::<Gray8>::new(8, 8, 1, false);
        let rotated = rotate_cpu(&src, 5.0);

        // Corner pixels of the rotated 8×8, with cx = cy = 3.5 and the
        // valid sample range [0, 7] × [0, 7]: at 5° the CW inverse map
        // sends (0,0) to (0.32, -0.29) and (0,7) to (-0.29, 7.29) — a
        // negative coordinate makes both genuinely OOB, so they white-fill.
        // (7,0) maps to (7.29, 0.32) and (7,7) to (6.68, 7.29): both floor
        // into the last valid column/row, where the clamped bilinear
        // footprint samples the (black) edge pixels.
        assert_eq!(
            rotated.row_bytes(0)[0],
            255,
            "(0,0) corner must be white-filled (sy < 0 at 5°)"
        );
        assert_eq!(
            rotated.row_bytes(7)[0],
            255,
            "(0,7) corner must be white-filled (sx < 0 at 5°)"
        );
        assert_eq!(
            rotated.row_bytes(0)[7],
            0,
            "(7,0) corner floors into the last column and must sample black"
        );
        assert_eq!(
            rotated.row_bytes(7)[7],
            0,
            "(7,7) corner floors into the last row and must sample black"
        );

        // Centre pixel: maps to (3.5, 3.5) which IS in-bounds; sample of
        // black source = 0.  If the bounds-check inverted to <= / >=, the
        // centre would also be white.
        assert_eq!(
            rotated.row_bytes(4)[4],
            0,
            "centre pixel must sample in-bounds black source, not OOB white"
        );
    }

    /// Rotating 360° should return approximately the original image.
    #[test]
    fn rotate_full_circle_near_identity() {
        let mut src = Bitmap::<Gray8>::new(64, 64, 1, false);
        for y in 0..64u32 {
            let row = src.row_bytes_mut(y);
            for (x, px) in row.iter_mut().enumerate() {
                // x ≤ 63 so x * 4 ≤ 252; well below u8::MAX, no truncation.
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "x ≤ 63 so x * 4 ≤ 252 < 256; fits u8 trivially"
                )]
                {
                    *px = (x * 4) as u8;
                }
            }
        }
        let rotated = rotate_cpu(&src, 360.0);
        // Bilinear + floating-point: allow ≤2 absolute difference at any pixel.
        let mut max_diff = 0i32;
        for y in 4..60u32 {
            let sr = &src.row_bytes(y)[4..60];
            let dr = &rotated.row_bytes(y)[4..60];
            for (&a, &b) in sr.iter().zip(dr.iter()) {
                max_diff = max_diff.max((i32::from(a) - i32::from(b)).abs());
            }
        }
        assert!(max_diff <= 2, "360° rotation max diff {max_diff} > 2");
    }

    /// GPU and CPU bilinear rotation agree to within 2 grey levels at any
    /// interior pixel for a typical deskew angle (2°).
    ///
    /// Gated on `gpu-validation` so it only runs when a CUDA device is
    /// available (same gate as the other GPU-vs-CPU parity tests).
    #[test]
    #[cfg(all(feature = "gpu-deskew", feature = "gpu-validation"))]
    fn gpu_vs_cpu_rotation_parity() {
        let mut src = Bitmap::<Gray8>::new(128, 128, 1, false);
        for y in 0..128u32 {
            let row = src.row_bytes_mut(y);
            for x in 0..128usize {
                row[x] = ((x * 2 + y as usize * 3) % 256) as u8;
            }
        }

        let angle = 2.0_f32;
        let cpu = rotate_cpu(&src, angle);
        let gpu = rotate_gpu(&src, angle).expect("GPU rotate failed");

        let mut max_diff = 0i32;
        // Skip a 4-pixel border where bilinear vs NPP edge treatment may differ.
        for y in 4..124u32 {
            let cr = &cpu.row_bytes(y)[4..124];
            let gr = &gpu.row_bytes(y)[4..124];
            for (&a, &b) in cr.iter().zip(gr.iter()) {
                max_diff = max_diff.max((a as i32 - b as i32).abs());
            }
        }
        assert!(
            max_diff <= 2,
            "GPU vs CPU rotation max diff {max_diff} > 2 at 2°"
        );
    }

    #[test]
    #[cfg(all(feature = "gpu-deskew", feature = "gpu-validation"))]
    fn gpu_rotate_zero_is_near_identity() {
        let mut src = Bitmap::<Gray8>::new(64, 64, 1, false);
        for y in 0..64u32 {
            let row = src.row_bytes_mut(y);
            for x in 0..64usize {
                row[x] = ((x * 4 + y as usize * 2) % 256) as u8;
            }
        }
        let gpu = rotate_gpu(&src, 0.0).expect("GPU rotate 0° failed");
        let mut max_diff = 0i32;
        for y in 4..60u32 {
            let sr = &src.row_bytes(y)[4..60];
            let gr = &gpu.row_bytes(y)[4..60];
            for (&a, &b) in sr.iter().zip(gr.iter()) {
                max_diff = max_diff.max((a as i32 - b as i32).abs());
            }
        }
        assert!(max_diff <= 2, "GPU 0° rotation max diff {max_diff} > 2");
    }

    /// Structural shape test for `rotate_gpu` — runs without a CUDA device.
    ///
    /// On a no-GPU machine, `rotate_gpu` returns `Err`; on a GPU machine it
    /// returns `Ok` with a bitmap matching the source dimensions.  This test
    /// accepts both outcomes but verifies the dimensions when the call
    /// succeeds.
    ///
    /// Pins the four whole-function-body replacement mutants
    /// (`Ok(Bitmap::new())`, `Ok(Bitmap::from_iter(...))`, etc.) that
    /// otherwise survive: each substitutes an empty / default Bitmap with
    /// dimensions ≠ source, which the size assertion catches even without
    /// a CUDA device. Pixel-content parity is covered by
    /// `gpu_vs_cpu_rotation_parity` under the `gpu-validation` gate.
    #[test]
    #[cfg(feature = "gpu-deskew")]
    fn rotate_gpu_preserves_dimensions_or_errors() {
        let mut src = Bitmap::<Gray8>::new(64, 64, 1, false);
        for y in 0..64u32 {
            let row = src.row_bytes_mut(y);
            for (x, px) in row.iter_mut().enumerate() {
                // x + y < 128, fits u8 trivially; bytemuck-equivalent cast.
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "x ≤ 63 and y ≤ 63 so x + y ≤ 126; well below u8::MAX"
                )]
                {
                    *px = (x + y as usize) as u8;
                }
            }
        }
        // No CUDA device → Err is the expected outcome; only the Ok arm needs
        // assertions (and is where cargo-mutants exercises rotate_gpu).
        if let Ok(out) = rotate_gpu(&src, 1.0) {
            assert_eq!(
                out.width, src.width,
                "rotate_gpu must return a bitmap with the source width"
            );
            assert_eq!(
                out.height, src.height,
                "rotate_gpu must return a bitmap with the source height"
            );
            assert_eq!(
                out.data().len(),
                (src.width as usize) * (src.height as usize),
                "rotate_gpu output buffer must have width*height bytes (tightly packed)"
            );
        }
    }

    /// Timing smoke-test: `rotate_cpu` on an 8.4 MP image (2900×2900).  The
    /// test always passes — the time is printed for manual inspection.  Run
    /// with `--nocapture` to see it.
    ///
    /// A4 at 300 DPI ≈ 2480×3508 (8.7 MP); 2900×2900 is a round-number proxy.
    /// Ryzen 9 9900X3D, release build, 1.5°: ~8.5 ms through the AVX-512
    /// tier, ~31 ms through the scalar loop.
    #[test]
    #[ignore = "manual perf smoke test; run with `--ignored --nocapture` for timing"]
    fn rotate_cpu_8mp_timing() {
        let w = 2900u32;
        let h = 2900u32;
        let mut src = Bitmap::<Gray8>::new(w, h, 1, false);
        for y in 0..h {
            let row = src.row_bytes_mut(y);
            for (x, px) in row.iter_mut().enumerate() {
                // % 256 keeps the value in [0, 255]; cast is exact.
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "modulo 256 keeps the value in [0, u8::MAX]; cast is exact"
                )]
                {
                    *px = ((x + y as usize * 3) % 256) as u8;
                }
            }
        }

        let t0 = std::time::Instant::now();
        let _ = rotate_cpu(&src, 1.5);
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        println!(
            "rotate_cpu 2900×2900 ({:.1} MP): {elapsed_ms:.2} ms",
            f64::from(w) * f64::from(h) / 1_000_000.0,
        );
        // Soft threshold: warn rather than fail so CI on slow machines doesn't flake.
        if elapsed_ms > 20.0 {
            eprintln!(
                "WARN: rotate_cpu exceeded 20 ms ({elapsed_ms:.2} ms) — \
                 check which row tier was dispatched on this machine"
            );
        }
    }
}
