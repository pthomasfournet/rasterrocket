//! SIMD tiers of [`super::cmyk_to_rgb_row`].
//!
//! Every tier evaluates the fixed-point multilinear interpolation of the
//! parent module lane-wise on `i32` lanes, one pixel per lane, with the
//! same operations in the same order as [`super::cmyk_to_rgb`]; integer
//! arithmetic makes the tiers byte-identical to the scalar reference.  The
//! M, C and K stages are shared through [`channel_fn`]; the Y stage, the
//! pixel deinterleave and the RGB packing are tier-specific.

#![expect(
    unsafe_code,
    reason = "SIMD loads, stores and runtime-dispatched target-feature calls require unsafe"
)]

use super::{Y_STAGES, YStage, convert_scalar};

/// Define a `#[target_feature]` function that interpolates one output
/// channel from the weight vectors `[c, m, y_operand, k]`.
///
/// `$ystage(stage, j, y_operand)` evaluates the Y stage of corner pair `j`
/// (`base + step·w_y` in the scalar reference); `y_operand` is whatever
/// form of `w_y` the tier's Y stage consumes.  The M, C and K stages mirror
/// `super::channel` operation for operation: `(a << 8) + (b − a)·w`,
/// rounding with `+128 >> 8` after M and C and `+32768 >> 16` after K.
/// Every intermediate is non-negative, so logical and arithmetic right
/// shifts agree.
macro_rules! channel_fn {
    (
        $name:ident, $feature:literal, $v:ty, ystage: $ystage:ident,
        splat: $splat:ident, add: $add:ident, sub: $sub:ident, mul: $mul:ident,
        shl: $shl:ident, shr: $shr:ident $(,)?
    ) => {
        #[inline]
        #[target_feature(enable = $feature)]
        fn $name(stage: &YStage, w: [$v; 4]) -> $v {
            let [wc, wm, wy, wk] = w;
            let lerp = |a: $v, b: $v, t: $v| $add($shl::<8>(a), $mul($sub(b, a), t));
            let round8 = |v: $v| $shr::<8>($add(v, $splat(128)));
            let y: [$v; 8] = std::array::from_fn(|j| $ystage(stage, j, wy));
            let m0 = round8(lerp(y[0], y[1], wm));
            let m1 = round8(lerp(y[2], y[3], wm));
            let m2 = round8(lerp(y[4], y[5], wm));
            let m3 = round8(lerp(y[6], y[7], wm));
            let c0 = round8(lerp(m0, m1, wc));
            let c1 = round8(lerp(m2, m3, wc));
            $shr::<16>($add(lerp(c0, c1, wk), $splat(1 << 15)))
        }
    };
}

// ── x86-64 ────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
pub(super) mod x86 {
    use std::arch::x86_64::{
        __m256i, __m512i, _mm_storel_epi64, _mm_storeu_si128, _mm256_add_epi32, _mm256_and_si256,
        _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_madd_epi16,
        _mm256_mullo_epi32, _mm256_or_si256, _mm256_permutevar8x32_epi32, _mm256_set1_epi32,
        _mm256_setr_epi8, _mm256_setr_epi32, _mm256_shuffle_epi8, _mm256_slli_epi32,
        _mm256_srli_epi32, _mm256_sub_epi32, _mm512_add_epi32, _mm512_and_si512,
        _mm512_loadu_si512, _mm512_madd_epi16, _mm512_mask_storeu_epi8, _mm512_mullo_epi32,
        _mm512_or_si512, _mm512_permutexvar_epi8, _mm512_set1_epi32, _mm512_slli_epi32,
        _mm512_srli_epi32, _mm512_sub_epi32,
    };

    use super::{Y_STAGES, YStage};

    // The x86 Y stage is one vpmaddwd per corner pair: each 32-bit lane of
    // the weight operand holds `256 − w_y` (low half) and `w_y` (high
    // half), and the broadcast constant holds the two corners as
    // `YStage::pair`, so the multiply-add yields `base + step·w_y` exactly.

    /// `(256 − w_y) | w_y << 16` per lane: the Y-stage weight operand.
    #[inline]
    #[target_feature(enable = "avx512f")]
    fn y_pair_avx512(wy: __m512i) -> __m512i {
        _mm512_or_si512(
            _mm512_sub_epi32(_mm512_set1_epi32(256), wy),
            _mm512_slli_epi32::<16>(wy),
        )
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    fn y_stage_avx512(stage: &YStage, j: usize, pair: __m512i) -> __m512i {
        _mm512_madd_epi16(pair, _mm512_set1_epi32(stage.pair[j]))
    }

    /// `(256 − w_y) | w_y << 16` per lane: the Y-stage weight operand.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn y_pair_avx2(wy: __m256i) -> __m256i {
        _mm256_or_si256(
            _mm256_sub_epi32(_mm256_set1_epi32(256), wy),
            _mm256_slli_epi32::<16>(wy),
        )
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn y_stage_avx2(stage: &YStage, j: usize, pair: __m256i) -> __m256i {
        _mm256_madd_epi16(pair, _mm256_set1_epi32(stage.pair[j]))
    }

    channel_fn!(
        channel_avx512, "avx512f,avx512bw", __m512i, ystage: y_stage_avx512,
        splat: _mm512_set1_epi32, add: _mm512_add_epi32, sub: _mm512_sub_epi32,
        mul: _mm512_mullo_epi32, shl: _mm512_slli_epi32, shr: _mm512_srli_epi32,
    );

    channel_fn!(
        channel_avx2, "avx2", __m256i, ystage: y_stage_avx2,
        splat: _mm256_set1_epi32, add: _mm256_add_epi32, sub: _mm256_sub_epi32,
        mul: _mm256_mullo_epi32, shl: _mm256_slli_epi32, shr: _mm256_srli_epi32,
    );

    /// vpermb table: RGB byte `i < 48` reads `RGBx` byte `(i / 3)·4 + i % 3`.
    #[expect(clippy::cast_possible_truncation, reason = "entries < 64")]
    const RGBX_TO_RGB: [u8; 64] = {
        let mut table = [0u8; 64];
        let mut i = 0;
        while i < 48 {
            table[i] = ((i / 3) * 4 + i % 3) as u8;
            i += 1;
        }
        table
    };

    /// Convert 16 CMYK pixels to RGB.
    ///
    /// Each 32-bit lane holds one pixel's `[C M Y K]` bytes; masks and
    /// shifts split it into four weight vectors, and the three channel
    /// results are packed as `RGBx` lanes, compacted with vpermb and
    /// stored with a 48-byte mask.
    ///
    /// # Safety
    ///
    /// Caller must ensure `avx512f`, `avx512bw` and `avx512vbmi` are available.
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
    pub(in crate::cmyk) unsafe fn block_avx512(cmyk: &[u8; 64], rgb: &mut [u8; 48]) {
        // SAFETY: `cmyk` is exactly 64 bytes; RGBX_TO_RGB is exactly 64 bytes.
        let (px, gather) = unsafe {
            (
                _mm512_loadu_si512(cmyk.as_ptr().cast()),
                _mm512_loadu_si512(RGBX_TO_RGB.as_ptr().cast()),
            )
        };
        let byte = _mm512_set1_epi32(0xFF);
        let weight = |x: __m512i| _mm512_add_epi32(x, _mm512_srli_epi32::<7>(x));
        let w = [
            weight(_mm512_and_si512(px, byte)),
            weight(_mm512_and_si512(_mm512_srli_epi32::<8>(px), byte)),
            y_pair_avx512(weight(_mm512_and_si512(_mm512_srli_epi32::<16>(px), byte))),
            weight(_mm512_srli_epi32::<24>(px)),
        ];
        let r = channel_avx512(&Y_STAGES[0], w);
        let g = channel_avx512(&Y_STAGES[1], w);
        let b = channel_avx512(&Y_STAGES[2], w);
        let rgbx = _mm512_or_si512(
            _mm512_or_si512(r, _mm512_slli_epi32::<8>(g)),
            _mm512_slli_epi32::<16>(b),
        );
        let packed = _mm512_permutexvar_epi8(gather, rgbx);
        // SAFETY: the mask selects bytes 0..48, exactly the extent of `rgb`.
        unsafe { _mm512_mask_storeu_epi8(rgb.as_mut_ptr().cast(), (1u64 << 48) - 1, packed) };
    }

    /// Convert 8 CMYK pixels to RGB.
    ///
    /// Same lane layout as [`block_avx512`]; the `RGBx` lanes are compacted
    /// per 128-bit half with vpshufb and the two 12-byte halves joined with
    /// vpermd before a 16 + 8 byte store.
    ///
    /// # Safety
    ///
    /// Caller must ensure `avx2` is available.
    #[target_feature(enable = "avx2")]
    pub(in crate::cmyk) unsafe fn block_avx2(cmyk: &[u8; 32], rgb: &mut [u8; 24]) {
        // SAFETY: `cmyk` is exactly 32 bytes.
        let px = unsafe { _mm256_loadu_si256(cmyk.as_ptr().cast()) };
        let byte = _mm256_set1_epi32(0xFF);
        let weight = |x: __m256i| _mm256_add_epi32(x, _mm256_srli_epi32::<7>(x));
        let w = [
            weight(_mm256_and_si256(px, byte)),
            weight(_mm256_and_si256(_mm256_srli_epi32::<8>(px), byte)),
            y_pair_avx2(weight(_mm256_and_si256(_mm256_srli_epi32::<16>(px), byte))),
            weight(_mm256_srli_epi32::<24>(px)),
        ];
        let r = channel_avx2(&Y_STAGES[0], w);
        let g = channel_avx2(&Y_STAGES[1], w);
        let b = channel_avx2(&Y_STAGES[2], w);
        let rgbx = _mm256_or_si256(
            _mm256_or_si256(r, _mm256_slli_epi32::<8>(g)),
            _mm256_slli_epi32::<16>(b),
        );
        let compact = _mm256_shuffle_epi8(
            rgbx,
            _mm256_setr_epi8(
                0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1, //
                0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1,
            ),
        );
        let joined =
            _mm256_permutevar8x32_epi32(compact, _mm256_setr_epi32(0, 1, 2, 4, 5, 6, 3, 7));
        // SAFETY: bytes 0..16 and 16..24 lie inside the 24-byte `rgb`.
        unsafe {
            _mm_storeu_si128(rgb.as_mut_ptr().cast(), _mm256_castsi256_si128(joined));
            _mm_storel_epi64(
                rgb.as_mut_ptr().add(16).cast(),
                _mm256_extracti128_si256::<1>(joined),
            );
        }
    }
}

/// x86-64: runtime-detect AVX-512 → AVX2 → scalar.
///
/// Runtime detection keeps one binary correct on every x86-64 CPU; a
/// compile-time `cfg(target_feature)` gate would remove the fallbacks from
/// `target-cpu=native` builds.
#[cfg(target_arch = "x86_64")]
pub(super) fn convert(cmyk: &[u8], rgb: &mut [u8]) {
    if is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vbmi")
    {
        // 64 input bytes ↔ 16 pixels ↔ 48 output bytes: the two chunk
        // iterators run in lockstep and their remainders hold the same pixels.
        let (blocks, tail) = cmyk.as_chunks::<64>();
        let (out_blocks, out_tail) = rgb.as_chunks_mut::<48>();
        for (block, out) in blocks.iter().zip(out_blocks) {
            // SAFETY: avx512f + avx512bw + avx512vbmi detected above.
            unsafe { x86::block_avx512(block, out) };
        }
        convert_scalar(tail, out_tail);
    } else if is_x86_feature_detected!("avx2") {
        let (blocks, tail) = cmyk.as_chunks::<32>();
        let (out_blocks, out_tail) = rgb.as_chunks_mut::<24>();
        for (block, out) in blocks.iter().zip(out_blocks) {
            // SAFETY: avx2 detected above.
            unsafe { x86::block_avx2(block, out) };
        }
        convert_scalar(tail, out_tail);
    } else {
        convert_scalar(cmyk, rgb);
    }
}

// ── aarch64 ───────────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
pub(super) mod arm {
    use std::arch::aarch64::{
        int32x4_t, uint8x16_t, uint8x16x3_t, vaddq_s32, vcombine_u8, vcombine_u16, vdupq_n_s32,
        vget_high_u8, vget_high_u16, vget_low_u8, vget_low_u16, vld4q_u8, vmlaq_s32, vmovl_u8,
        vmovl_u16, vmovn_u16, vmovn_u32, vmulq_s32, vreinterpretq_s32_u32, vreinterpretq_u32_s32,
        vshlq_n_s32, vshrq_n_s32, vst3q_u8, vsubq_s32,
    };

    use super::{Y_STAGES, YStage};

    /// Y stage `base + step·w_y` as one multiply-accumulate.
    #[inline]
    #[target_feature(enable = "neon")]
    fn y_stage_neon(stage: &YStage, j: usize, wy: int32x4_t) -> int32x4_t {
        vmlaq_s32(vdupq_n_s32(stage.base[j]), vdupq_n_s32(stage.step[j]), wy)
    }

    channel_fn!(
        channel_neon, "neon", int32x4_t, ystage: y_stage_neon,
        splat: vdupq_n_s32, add: vaddq_s32, sub: vsubq_s32,
        mul: vmulq_s32, shl: vshlq_n_s32, shr: vshrq_n_s32,
    );

    /// Widen 16 ink bytes to four vectors of `[0, 256]` weights.
    #[inline]
    #[target_feature(enable = "neon")]
    fn weights(x: uint8x16_t) -> [int32x4_t; 4] {
        let lo = vmovl_u8(vget_low_u8(x));
        let hi = vmovl_u8(vget_high_u8(x));
        [
            vmovl_u16(vget_low_u16(lo)),
            vmovl_u16(vget_high_u16(lo)),
            vmovl_u16(vget_low_u16(hi)),
            vmovl_u16(vget_high_u16(hi)),
        ]
        .map(|v| {
            let v = vreinterpretq_s32_u32(v);
            vaddq_s32(v, vshrq_n_s32::<7>(v))
        })
    }

    /// Narrow four vectors of channel bytes (one per `i32` lane) to 16 bytes.
    #[inline]
    #[target_feature(enable = "neon")]
    fn narrow(quarters: [int32x4_t; 4]) -> uint8x16_t {
        let [q0, q1, q2, q3] = quarters.map(|v| vmovn_u32(vreinterpretq_u32_s32(v)));
        vcombine_u8(
            vmovn_u16(vcombine_u16(q0, q1)),
            vmovn_u16(vcombine_u16(q2, q3)),
        )
    }

    /// Convert 16 CMYK pixels to RGB.
    ///
    /// `vld4q_u8` deinterleaves the channels, each widened to four `i32`
    /// vectors; `vst3q_u8` re-interleaves the narrowed results.
    ///
    /// # Safety
    ///
    /// Caller must ensure `neon` is available (mandatory on all ARMv8-A targets).
    #[target_feature(enable = "neon")]
    pub(in crate::cmyk) unsafe fn block_neon(cmyk: &[u8; 64], rgb: &mut [u8; 48]) {
        // SAFETY: `cmyk` is exactly 64 bytes, the extent of one vld4q_u8.
        let px = unsafe { vld4q_u8(cmyk.as_ptr()) };
        let (wc, wm, wy, wk) = (weights(px.0), weights(px.1), weights(px.2), weights(px.3));
        let quarter = |ch: usize| -> [int32x4_t; 4] {
            std::array::from_fn(|q| channel_neon(&Y_STAGES[ch], [wc[q], wm[q], wy[q], wk[q]]))
        };
        let out = uint8x16x3_t(narrow(quarter(0)), narrow(quarter(1)), narrow(quarter(2)));
        // SAFETY: `rgb` is exactly 48 bytes, the extent of one vst3q_u8.
        unsafe { vst3q_u8(rgb.as_mut_ptr(), out) };
    }
}

/// aarch64: NEON is mandatory on all ARMv8-A targets; no runtime detection.
#[cfg(target_arch = "aarch64")]
pub(super) fn convert(cmyk: &[u8], rgb: &mut [u8]) {
    let (blocks, tail) = cmyk.as_chunks::<64>();
    let (out_blocks, out_tail) = rgb.as_chunks_mut::<48>();
    for (block, out) in blocks.iter().zip(out_blocks) {
        // SAFETY: NEON is mandatory on aarch64.
        unsafe { arm::block_neon(block, out) };
    }
    convert_scalar(tail, out_tail);
}
