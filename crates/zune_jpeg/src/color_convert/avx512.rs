//! YCbCr to RGB on AVX-512 (BW and VBMI), 32 pixels at a time.
//!
//! The arithmetic is that of the AVX2 conversion (`avx.rs`) on 512-bit
//! vectors, lane for lane, so the two produce the same bytes as the scalar
//! conversion.  The three channels are packed to bytes and interleaved by
//! two byte permutations (`vpermt2b`).

#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#![cfg(feature = "x86")]

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::color_convert::avx::{B_AVX_COEF, G_COEF_AVX_COEF, R_AVX_COEF};
use crate::color_convert::scalar::YUV_RND;

/// The source byte of each output byte of 32 RGB pixels, for
/// `_mm512_permutex2var_epi8` over `rg` (red and green packed by
/// `_mm512_packus_epi16`: in 128-bit lane `l`, red of pixels `8l .. 8l + 8`
/// then their green) and `b` (blue packed against zeros, bit 6 of the
/// index): output byte `j` is channel `j % 3` of pixel `j / 3`.  Bytes 96
/// and above are not stored.
const fn rgb_index(first: usize) -> [u8; 64] {
    let mut index = [0_u8; 64];
    let mut i = 0;
    while i < 64 {
        let j = first + i;
        if j < 96 {
            let (pixel, channel) = (j / 3, j % 3);
            let byte = 16 * (pixel / 8) + pixel % 8;
            index[i] = match channel {
                0 => byte,
                1 => byte + 8,
                _ => 64 + byte,
            } as u8;
        }
        i += 1;
    }
    index
}

static RGB_INDEX: [[u8; 64]; 2] = [rgb_index(0), rgb_index(64)];

/// Convert 32 YCbCr pixels to 96 bytes of RGB at the start of `out`.
///
/// # Safety
/// The CPU must support AVX-512 F, BW and VBMI.
///
/// # Panics
/// When `out` is shorter than 96 bytes.
#[inline(always)]
pub unsafe fn ycbcr_to_rgb_avx512(y: &[i16; 32], cb: &[i16; 32], cr: &[i16; 32], out: &mut [u8]) {
    let out: &mut [u8; 96] = out
        .get_mut(..96)
        .expect("Slice to small cannot write")
        .try_into()
        .unwrap();
    // SAFETY: the caller guarantees AVX-512 F, BW and VBMI.
    unsafe { ycbcr_to_rgb_avx512_inner(y, cb, cr, out) }
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
unsafe fn ycbcr_to_rgb_avx512_inner(
    y: &[i16; 32],
    cb: &[i16; 32],
    cr: &[i16; 32],
    out: &mut [u8; 96],
) {
    let y_c = _mm512_loadu_si512(y.as_ptr().cast());
    let cb_c = _mm512_loadu_si512(cb.as_ptr().cast());
    let cr_c = _mm512_loadu_si512(cr.as_ptr().cast());

    let cr_coeff = _mm512_set1_epi32(R_AVX_COEF);
    let cb_coeff = _mm512_set1_epi32(B_AVX_COEF);
    let cg_coeff = _mm512_set1_epi32(G_COEF_AVX_COEF);
    let v_rnd = _mm512_set1_epi32(YUV_RND);
    let uv_bias = _mm512_set1_epi16(128);
    let zero = _mm512_setzero_si512();

    // Cr in the low and Cb in the high byte of each 16-bit lane, widened
    // to (Cr, Cb) pairs of 16-bit lanes for `madd`.
    let u_v_8 = _mm512_or_si512(_mm512_slli_epi16::<8>(cb_c), cr_c);
    let u_v_lo = _mm512_sub_epi16(_mm512_unpacklo_epi8(u_v_8, zero), uv_bias);
    let u_v_hi = _mm512_sub_epi16(_mm512_unpackhi_epi8(u_v_8, zero), uv_bias);
    // The fractional part of each coefficient, rounded half up; the whole
    // part is applied as additions.
    let fraction = |coefficients| {
        let low = _mm512_add_epi32(_mm512_madd_epi16(u_v_lo, coefficients), v_rnd);
        let high = _mm512_add_epi32(_mm512_madd_epi16(u_v_hi, coefficients), v_rnd);
        _mm512_packs_epi32(_mm512_srai_epi32::<16>(low), _mm512_srai_epi32::<16>(high))
    };
    let cb = _mm512_sub_epi16(cb_c, uv_bias);
    let cr = _mm512_sub_epi16(cr_c, uv_bias);
    let r = _mm512_add_epi16(_mm512_add_epi16(y_c, cr), fraction(cr_coeff));
    let g = _mm512_add_epi16(_mm512_sub_epi16(y_c, cr), fraction(cg_coeff));
    let b = _mm512_add_epi16(
        _mm512_add_epi16(y_c, _mm512_slli_epi16::<1>(cb)),
        fraction(cb_coeff),
    );

    // Clamped to bytes by unsigned saturation.
    let rg = _mm512_packus_epi16(r, g);
    let b = _mm512_packus_epi16(b, zero);
    let index0 = _mm512_loadu_si512(RGB_INDEX[0].as_ptr().cast());
    let index1 = _mm512_loadu_si512(RGB_INDEX[1].as_ptr().cast());
    let rgb0 = _mm512_permutex2var_epi8(rg, index0, b);
    let rgb1 = _mm512_permutex2var_epi8(rg, index1, b);
    _mm512_storeu_si512(out.as_mut_ptr().cast(), rgb0);
    _mm256_storeu_si256(out[64..].as_mut_ptr().cast(), _mm512_castsi512_si256(rgb1));
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::color_convert::avx::ycbcr_to_rgb_avx2;
    use crate::color_convert::scalar::ycbcr_to_rgb_inner_16_scalar;

    fn supported() -> bool {
        is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vbmi")
    }

    /// Every Cb/Cr pair, at luma values across the range and past it, with
    /// Cb and Cr varying across the 32 pixels: the AVX-512 conversion of 32
    /// pixels equals the scalar and the AVX2 conversions of their two
    /// halves, and writes exactly 96 bytes.
    #[test]
    fn matches_scalar_and_avx2() {
        if !supported() {
            return;
        }
        let avx2 = is_x86_feature_detected!("avx2");
        let ys: [[i16; 32]; 3] = [
            core::array::from_fn(|i| (i * 8) as i16),
            core::array::from_fn(|i| 255 - (i * 5) as i16),
            core::array::from_fn(|i| [-40, 0, 128, 255, 300][i % 5]),
        ];
        for y in &ys {
            for cb0 in 0_i16..=255 {
                for cr0 in 0_i16..=255 {
                    let cb: [i16; 32] = core::array::from_fn(|i| (cb0 + i as i16 * 37) & 255);
                    let cr: [i16; 32] = core::array::from_fn(|i| (cr0 + i as i16 * 101) & 255);
                    let mut got = [0xa5_u8; 100];
                    // SAFETY: `supported` checked AVX-512 F, BW and VBMI.
                    unsafe { ycbcr_to_rgb_avx512(y, &cb, &cr, &mut got) };
                    assert_eq!(got[96..], [0xa5; 4]);
                    let mut want = [0_u8; 128];
                    for half in 0..2 {
                        let part = |v: &[i16; 32]| -> [i16; 16] {
                            v[16 * half..][..16].try_into().unwrap()
                        };
                        let (y, cb, cr) = (part(y), part(&cb), part(&cr));
                        let mut scalar = [0_u8; 64];
                        ycbcr_to_rgb_inner_16_scalar::<false>(&y, &cb, &cr, &mut scalar, &mut 0);
                        want[48 * half..][..48].copy_from_slice(&scalar[..48]);
                        if avx2 {
                            let mut vector = [0_u8; 64];
                            ycbcr_to_rgb_avx2(&y, &cb, &cr, &mut vector, &mut 0);
                            assert_eq!(vector[..48], scalar[..48], "AVX2, cb {cb0} cr {cr0}");
                        }
                    }
                    assert_eq!(got[..96], want[..96], "y {y:?}, cb {cb0} cr {cr0}");
                }
            }
        }
    }
}
