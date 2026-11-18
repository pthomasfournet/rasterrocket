//! The triangle up-sampling filters on AVX-512 (F and BW), 32 samples at a
//! time.
//!
//! Each output sample is computed as in the AVX2 filters (`avx2.rs`), from
//! the same three inputs in 16-bit lanes; rows shorter than one step of 32
//! take the AVX2 filters.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::avx2;

/// [`avx2::upsample_horizontal_avx2`], 32 input samples at a time.
///
/// # Safety
/// The CPU must support AVX-512 F and BW.
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn upsample_horizontal_avx512(
    input: &[i16],
    in_near: &[i16],
    in_far: &[i16],
    scratch: &mut [i16],
    output: &mut [i16],
) {
    assert_eq!(input.len() * 2, output.len());
    let len = input.len();
    if len < 34 {
        return avx2::upsample_horizontal_avx2(input, in_near, in_far, scratch, output);
    }

    // First two samples.
    output[0] = input[0];
    output[1] = (input[0] * 3 + input[1] + 2) >> 2;

    let v_three = _mm512_set1_epi16(3);
    let v_two = _mm512_set1_epi16(2);
    // Even and odd outputs interleaved per 128-bit lane by the unpacks; in
    // 64-bit units, the first 32 outputs are lanes 0 and 1 of each, the
    // last 32 lanes 2 and 3.
    let first = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    let second = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);

    let upsample32 = |input: &[i16; 34], output: &mut [i16; 64]| {
        // SAFETY: the input holds 34 samples, the three loads 32 each from
        // offsets 0, 1 and 2.
        let (v_prev, v_curr, v_next) = unsafe {
            (
                _mm512_loadu_si512(input.as_ptr().cast()),
                _mm512_loadu_si512(input.as_ptr().add(1).cast()),
                _mm512_loadu_si512(input.as_ptr().add(2).cast()),
            )
        };
        let v_common = _mm512_add_epi16(_mm512_mullo_epi16(v_curr, v_three), v_two);
        let v_even = _mm512_srai_epi16::<2>(_mm512_add_epi16(v_common, v_prev));
        let v_odd = _mm512_srai_epi16::<2>(_mm512_add_epi16(v_common, v_next));
        let lo = _mm512_unpacklo_epi16(v_even, v_odd);
        let hi = _mm512_unpackhi_epi16(v_even, v_odd);
        // SAFETY: the output holds 64 samples, the two stores 32 each.
        unsafe {
            _mm512_storeu_si512(
                output.as_mut_ptr().cast(),
                _mm512_permutex2var_epi64(lo, first, hi),
            );
            _mm512_storeu_si512(
                output.as_mut_ptr().add(32).cast(),
                _mm512_permutex2var_epi64(lo, second, hi),
            );
        }
    };

    for (input, output) in input
        .windows(34)
        .step_by(32)
        .zip(output[2..].as_chunks_mut::<64>().0.iter_mut())
    {
        upsample32(input.try_into().unwrap(), output);
    }
    // The remainder, overlapping samples already made with the same values.
    if let Some(rest_input) = input.last_chunk::<34>() {
        let end = output.len() - 2;
        if let Some(rest_output) = output[..end].last_chunk_mut::<64>() {
            upsample32(rest_input, rest_output);
        }
    }

    // Last two samples.
    output[output.len() - 2] = (3 * input[len - 1] + input[len - 2] + 2) >> 2;
    output[output.len() - 1] = input[len - 1];
}

/// [`avx2::upsample_vertical_avx2`], 32 samples at a time.
///
/// # Safety
/// The CPU must support AVX-512 F and BW.
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn upsample_vertical_avx512(
    input: &[i16],
    in_near: &[i16],
    in_far: &[i16],
    scratch: &mut [i16],
    output: &mut [i16],
) {
    assert_eq!(input.len() * 2, output.len());
    assert_eq!(in_near.len(), input.len());
    assert_eq!(in_far.len(), input.len());
    if input.len() < 32 {
        return avx2::upsample_vertical_avx2(input, in_near, in_far, scratch, output);
    }

    let middle = output.len() / 2;
    let (out_top, out_bottom) = output.split_at_mut(middle);
    let v_three = _mm512_set1_epi16(3);
    let v_two = _mm512_set1_epi16(2);

    let upsample32 = |input: &[i16; 32],
                      in_near: &[i16; 32],
                      in_far: &[i16; 32],
                      out_top: &mut [i16; 32],
                      out_bottom: &mut [i16; 32]| {
        // SAFETY: every input holds 32 samples.
        let (v_in, v_near, v_far) = unsafe {
            (
                _mm512_loadu_si512(input.as_ptr().cast()),
                _mm512_loadu_si512(in_near.as_ptr().cast()),
                _mm512_loadu_si512(in_far.as_ptr().cast()),
            )
        };
        let v_common = _mm512_add_epi16(_mm512_mullo_epi16(v_in, v_three), v_two);
        let v_top = _mm512_srai_epi16::<2>(_mm512_add_epi16(v_common, v_near));
        let v_bottom = _mm512_srai_epi16::<2>(_mm512_add_epi16(v_common, v_far));
        // SAFETY: both outputs hold 32 samples.
        unsafe {
            _mm512_storeu_si512(out_top.as_mut_ptr().cast(), v_top);
            _mm512_storeu_si512(out_bottom.as_mut_ptr().cast(), v_bottom);
        }
    };

    let chunks = input
        .as_chunks::<32>()
        .0
        .iter()
        .zip(in_near.as_chunks::<32>().0.iter())
        .zip(in_far.as_chunks::<32>().0.iter())
        .zip(out_top.as_chunks_mut::<32>().0.iter_mut())
        .zip(out_bottom.as_chunks_mut::<32>().0.iter_mut());
    for ((((input, in_near), in_far), out_top), out_bottom) in chunks {
        upsample32(input, in_near, in_far, out_top, out_bottom);
    }
    // The remainder, overlapping samples already made with the same values.
    if let (Some(rest), Some(rest_near), Some(rest_far), Some(rest_top), Some(rest_bottom)) = (
        input.last_chunk::<32>(),
        in_near.last_chunk::<32>(),
        in_far.last_chunk::<32>(),
        out_top.last_chunk_mut::<32>(),
        out_bottom.last_chunk_mut::<32>(),
    ) {
        upsample32(rest, rest_near, rest_far, rest_top, rest_bottom);
    }
}

/// [`avx2::upsample_hv_avx2`] through the AVX-512 filters.
///
/// # Safety
/// The CPU must support AVX-512 F and BW.
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn upsample_hv_avx512(
    input: &[i16],
    in_near: &[i16],
    in_far: &[i16],
    scratch_space: &mut [i16],
    output: &mut [i16],
) {
    assert_eq!(input.len() * 4, output.len());
    assert!(input.len() * 2 <= scratch_space.len());
    let scratch_space = &mut scratch_space[..input.len() * 2];

    upsample_vertical_avx512(input, in_near, in_far, &mut [], scratch_space);

    let scratch_half = scratch_space.len() / 2;
    let output_half = output.len() / 2;
    let (scratch_top, scratch_bottom) = scratch_space.split_at_mut(scratch_half);
    let (out_top, out_bottom) = output.split_at_mut(output_half);

    let mut t = [0];
    upsample_horizontal_avx512(scratch_top, &[], &[], &mut t, out_top);
    upsample_horizontal_avx512(scratch_bottom, &[], &[], &mut t, out_bottom);
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::super::scalar;
    use super::*;

    fn supported() -> bool {
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
    }

    /// `n` pseudo-random samples in 0–255.
    fn samples(n: usize, seed: u32) -> Vec<i16> {
        let mut state = seed.wrapping_mul(0x9E37_79B9) | 1;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state % 256) as i16
            })
            .collect()
    }

    /// Every filter equals the scalar and AVX2 ones over every width from 3
    /// to 300 samples.
    #[test]
    fn matches_scalar_and_avx2() {
        if !supported() {
            return;
        }
        for width in 3..=300 {
            let input = samples(width, width as u32);
            let near = samples(width, width as u32 + 1000);
            let far = samples(width, width as u32 + 2000);

            let mut want = vec![0_i16; width * 2];
            let mut got = vec![0_i16; width * 2];
            let mut vector = vec![0_i16; width * 2];
            scalar::upsample_horizontal(&input, &[], &[], &mut [0], &mut want);
            unsafe {
                upsample_horizontal_avx512(&input, &[], &[], &mut [0], &mut got);
                avx2::upsample_horizontal_avx2(&input, &[], &[], &mut [0], &mut vector);
            }
            assert_eq!(got, want, "horizontal, width {width}");
            assert_eq!(vector, want, "horizontal AVX2, width {width}");

            scalar::upsample_vertical(&input, &near, &far, &mut [], &mut want);
            unsafe {
                upsample_vertical_avx512(&input, &near, &far, &mut [], &mut got);
                avx2::upsample_vertical_avx2(&input, &near, &far, &mut [], &mut vector);
            }
            assert_eq!(got, want, "vertical, width {width}");
            assert_eq!(vector, want, "vertical AVX2, width {width}");

            let mut want = vec![0_i16; width * 4];
            let mut got = vec![0_i16; width * 4];
            let mut vector = vec![0_i16; width * 4];
            let mut scratch = vec![0_i16; width * 2];
            scalar::upsample_hv(&input, &near, &far, &mut scratch, &mut want);
            unsafe {
                upsample_hv_avx512(&input, &near, &far, &mut scratch, &mut got);
                avx2::upsample_hv_avx2(&input, &near, &far, &mut scratch, &mut vector);
            }
            assert_eq!(got, want, "hv, width {width}");
            assert_eq!(vector, want, "hv AVX2, width {width}");
        }
    }
}
