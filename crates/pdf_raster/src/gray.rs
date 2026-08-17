//! RGB → grayscale conversion (BT.709 luma, 15-bit fixed point).
//!
//! The luma definition shared by every tier is
//!
//! ```text
//! gray = (6966·r + 23436·g + 2366·b + 16384) >> 15
//! ```
//!
//! — BT.709 coefficients quantised to 1/32768 with round-half-up. The
//! coefficient sum is exactly 32768, so `r == g == b` maps to itself at
//! every level. The power-of-two denominator is what lets the SIMD tier
//! be byte-exact with the scalar reference: `_mm_madd_epi16` produces
//! the dot product directly and a single shift replaces the decimal
//! divide (the parity test is exhaustive over all 2²⁴ inputs).

use color::{Gray8, Rgb8};
use raster::Bitmap;

/// BT.709 luma coefficients in 1.15 fixed point; they sum to exactly
/// `1 << LUMA_SHIFT`.
const LUMA_R: u32 = 6966;
const LUMA_G: u32 = 23436;
const LUMA_B: u32 = 2366;
/// Round-half-up bias (`0.5` in 1.15 fixed point).
const LUMA_BIAS: u32 = 16384;
const LUMA_SHIFT: u32 = 15;

/// Convert an RGB bitmap to grayscale using BT.709 luminance coefficients.
#[must_use]
pub fn rgb_to_gray(src: &Bitmap<Rgb8>) -> Bitmap<Gray8> {
    let mut dst = Bitmap::<Gray8>::new(src.width, src.height, 1, false);
    let w = src.width as usize;
    for y in 0..src.height {
        let src_row = &src.row_bytes(y)[..w * 3];
        let dst_row = &mut dst.row_bytes_mut(y)[..w];
        convert_row(src_row, dst_row);
    }
    dst
}

/// Per-row dispatch.
///
/// The explicit kernel is 128-bit; when the build statically enables
/// AVX2 or wider (`-C target-cpu=native` on any modern x86), LLVM
/// autovectorises `convert_row_scalar` to 256/512-bit code that is
/// faster than the kernel, so the kernel dispatches only on builds
/// without static AVX2 — where the autovectoriser is limited to SSE2
/// and the runtime-detected SSE4.1 kernel wins.
fn convert_row(src_row: &[u8], dst_row: &mut [u8]) {
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    if std::arch::is_x86_feature_detected!("sse4.1") {
        // SAFETY: guarded by the sse4.1 runtime check; sse4.1 implies
        // ssse3 on every shipped x86-64 CPU.
        #[expect(unsafe_code, reason = "SIMD tier behind runtime feature detection")]
        unsafe {
            convert_row_sse41(src_row, dst_row);
        }
        return;
    }
    convert_row_scalar(src_row, dst_row);
}

/// Scalar reference: one pixel at a time.
fn convert_row_scalar(src_row: &[u8], dst_row: &mut [u8]) {
    for (dst_px, rgb) in dst_row.iter_mut().zip(src_row.chunks_exact(3)) {
        let (r, g, b) = (u32::from(rgb[0]), u32::from(rgb[1]), u32::from(rgb[2]));
        #[expect(
            clippy::cast_possible_truncation,
            reason = "sum ≤ 255 by the coefficient-sum identity"
        )]
        {
            *dst_px = ((LUMA_R * r + LUMA_G * g + LUMA_B * b + LUMA_BIAS) >> LUMA_SHIFT) as u8;
        }
    }
}

/// SSE4.1 tier: 16 pixels (48 source bytes) per iteration.
///
/// Deinterleaves each channel with three `pshufb` masks per channel,
/// pairs the channels as `[r, g]` / `[b, 1]` 16-bit lanes, and reduces
/// with `_mm_madd_epi16` against `[LUMA_R, LUMA_G]` / `[LUMA_B,
/// LUMA_BIAS]` — the `1` in the data lane turns the bias coefficient
/// into the round-half-up addend. All intermediates fit their lanes:
/// coefficients < 2¹⁵ as i16, per-pair dot products ≤ 32768·255 +
/// 16384 < 2³¹ as i32.
///
/// # Safety
///
/// Caller must ensure the CPU supports SSE4.1 (and therefore SSSE3).
#[cfg(target_arch = "x86_64")]
#[cfg_attr(
    all(target_feature = "avx2", not(test)),
    expect(
        dead_code,
        reason = "dispatch prefers the autovectorised scalar on static-AVX2 builds; \
                  the kernel serves baseline builds and the parity tests"
    )
)]
#[expect(
    unsafe_code,
    reason = "x86 intrinsics for the RGB deinterleave + madd luma kernel"
)]
#[expect(
    clippy::many_single_char_names,
    reason = "r/g/b are the domain names for the colour channels"
)]
#[expect(
    clippy::too_many_lines,
    reason = "shuffle-mask tables and the block loop form one coherent kernel"
)]
#[target_feature(enable = "ssse3", enable = "sse4.1")]
unsafe fn convert_row_sse41(src_row: &[u8], dst_row: &mut [u8]) {
    use std::arch::x86_64::{
        __m128i, _mm_add_epi32, _mm_loadu_si128, _mm_madd_epi16, _mm_or_si128, _mm_packus_epi16,
        _mm_packus_epi32, _mm_set1_epi8, _mm_set1_epi32, _mm_setzero_si128, _mm_shuffle_epi8,
        _mm_srli_epi32, _mm_storeu_si128, _mm_unpackhi_epi8, _mm_unpacklo_epi8,
    };

    /// Build a `pshufb` mask selecting `idx` (0x80 = zero) per output byte.
    const fn mask(idx: [i16; 16]) -> [u8; 16] {
        let mut out = [0u8; 16];
        let mut i = 0;
        while i < 16 {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "entries are 0..=15 or the 0x80 zero sentinel (-1)"
            )]
            {
                out[i] = if idx[i] < 0 { 0x80 } else { idx[i] as u8 };
            }
            i += 1;
        }
        out
    }

    // Channel c of pixel p lives at source byte 3·p + c. Per 48-byte
    // block (three 16-byte loads v0/v1/v2), each channel's 16 bytes are
    // assembled from three shuffles OR'd together.
    const R0: [u8; 16] = mask([0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]);
    const R1: [u8; 16] = mask([-1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1, -1, -1, -1]);
    const R2: [u8; 16] = mask([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4, 7, 10, 13]);
    const G0: [u8; 16] = mask([1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]);
    const G1: [u8; 16] = mask([-1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1]);
    const G2: [u8; 16] = mask([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14]);
    const B0: [u8; 16] = mask([2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]);
    const B1: [u8; 16] = mask([-1, -1, -1, -1, -1, 1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1]);
    const B2: [u8; 16] = mask([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15]);

    #[inline]
    fn m128(bytes: &[u8; 16]) -> __m128i {
        // SAFETY: loadu has no alignment requirement; the array is 16 bytes.
        #[expect(unsafe_code, reason = "unaligned 16-byte constant load")]
        unsafe {
            _mm_loadu_si128(bytes.as_ptr().cast())
        }
    }

    let px_count = dst_row.len().min(src_row.len() / 3);
    let blocks = px_count / 16;

    #[expect(
        clippy::cast_possible_wrap,
        reason = "packed coefficient constants are < 2^31"
    )]
    let (rg_coef, b1_coef) = (
        _mm_set1_epi32(((LUMA_G << 16) | LUMA_R) as i32),
        _mm_set1_epi32(((LUMA_BIAS << 16) | LUMA_B) as i32),
    );
    let ones = _mm_set1_epi8(1);
    let zero = _mm_setzero_si128();

    for blk in 0..blocks {
        let s = blk * 48;
        // SAFETY: blk < px_count/16 ⇒ s + 48 ≤ px_count·3 ≤ src_row.len().
        let (v0, v1, v2) = unsafe {
            (
                _mm_loadu_si128(src_row.as_ptr().add(s).cast()),
                _mm_loadu_si128(src_row.as_ptr().add(s + 16).cast()),
                _mm_loadu_si128(src_row.as_ptr().add(s + 32).cast()),
            )
        };

        let r = _mm_or_si128(
            _mm_or_si128(
                _mm_shuffle_epi8(v0, m128(&R0)),
                _mm_shuffle_epi8(v1, m128(&R1)),
            ),
            _mm_shuffle_epi8(v2, m128(&R2)),
        );
        let g = _mm_or_si128(
            _mm_or_si128(
                _mm_shuffle_epi8(v0, m128(&G0)),
                _mm_shuffle_epi8(v1, m128(&G1)),
            ),
            _mm_shuffle_epi8(v2, m128(&G2)),
        );
        let b = _mm_or_si128(
            _mm_or_si128(
                _mm_shuffle_epi8(v0, m128(&B0)),
                _mm_shuffle_epi8(v1, m128(&B1)),
            ),
            _mm_shuffle_epi8(v2, m128(&B2)),
        );

        // Interleave to [r, g] / [b, 1] byte pairs, then zero-extend to
        // the i16 lane layout `_mm_madd_epi16` reduces pairwise.
        let rg_lo = _mm_unpacklo_epi8(r, g);
        let rg_hi = _mm_unpackhi_epi8(r, g);
        let b1_lo = _mm_unpacklo_epi8(b, ones);
        let b1_hi = _mm_unpackhi_epi8(b, ones);

        let q = |rg_bytes: __m128i, b1_bytes: __m128i| -> __m128i {
            let sum = _mm_add_epi32(
                _mm_madd_epi16(_mm_unpacklo_epi8(rg_bytes, zero), rg_coef),
                _mm_madd_epi16(_mm_unpacklo_epi8(b1_bytes, zero), b1_coef),
            );
            let sum_hi = _mm_add_epi32(
                _mm_madd_epi16(_mm_unpackhi_epi8(rg_bytes, zero), rg_coef),
                _mm_madd_epi16(_mm_unpackhi_epi8(b1_bytes, zero), b1_coef),
            );
            _mm_packus_epi32(
                _mm_srli_epi32(sum, LUMA_SHIFT.cast_signed()),
                _mm_srli_epi32(sum_hi, LUMA_SHIFT.cast_signed()),
            )
        };
        let gray16_lo = q(rg_lo, b1_lo); // pixels 0..8 as u16
        let gray16_hi = q(rg_hi, b1_hi); // pixels 8..16 as u16
        let gray = _mm_packus_epi16(gray16_lo, gray16_hi);

        // SAFETY: blk < px_count/16 ⇒ blk·16 + 16 ≤ px_count ≤ dst_row.len().
        unsafe {
            _mm_storeu_si128(dst_row.as_mut_ptr().add(blk * 16).cast(), gray);
        }
    }

    let done = blocks * 16;
    convert_row_scalar(&src_row[done * 3..], &mut dst_row[done..]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bitmap_from_pixels(pixels: &[[u8; 3]], w: u32, h: u32) -> Bitmap<Rgb8> {
        assert_eq!(pixels.len(), (w * h) as usize);
        let mut bmp = Bitmap::<Rgb8>::new(w, h, 3, false);
        for y in 0..h {
            let row = &mut bmp.row_bytes_mut(y)[..(w * 3) as usize];
            for x in 0..w as usize {
                row[x * 3..x * 3 + 3].copy_from_slice(&pixels[y as usize * w as usize + x]);
            }
        }
        bmp
    }

    #[test]
    fn luma_uses_15_bit_fixed_point_rounding() {
        // Pin the exact fixed-point definition
        // `(6966·r + 23436·g + 2366·b + 16384) >> 15` on inputs where it
        // differs (by exactly one level) from the old decimal
        // `(2126·r + 7152·g + 722·b + 5000) / 10000` — a regression to
        // the decimal divide flips these.
        let cases: [([u8; 3], u8); 3] = [
            ([0, 5, 179], 17),
            ([0, 13, 169], 22),
            ([255, 255, 255], 255),
        ];
        let pixels: Vec<[u8; 3]> = cases.iter().map(|&(px, _)| px).collect();
        let bmp = bitmap_from_pixels(&pixels, 3, 1);
        let gray = rgb_to_gray(&bmp);
        for (i, &(px, want)) in cases.iter().enumerate() {
            assert_eq!(
                gray.row_bytes(0)[i],
                want,
                "luma({px:?}) must be {want} under 15-bit fixed point"
            );
        }
    }

    #[test]
    fn gray_input_maps_to_itself() {
        // The coefficient sum is exactly one, so r == g == b must map to
        // itself at every level.
        let pixels: Vec<[u8; 3]> = (0..=255u8).map(|v| [v, v, v]).collect();
        let bmp = bitmap_from_pixels(&pixels, 256, 1);
        let gray = rgb_to_gray(&bmp);
        for (v, &px) in (0..=255u8).zip(&gray.row_bytes(0)[..256]) {
            assert_eq!(px, v, "gray input {v} must map to itself");
        }
    }

    /// The SIMD tier must agree with the scalar reference on every one of
    /// the 2²⁴ possible RGB inputs — exhaustive, so any rounding drift in
    /// the kernel is unreachable rather than merely unlikely.
    #[test]
    fn simd_matches_scalar_exhaustively() {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::arch::is_x86_feature_detected!("sse4.1") {
                eprintln!("skipping: no SSE4.1");
                return;
            }
            let mut src = vec![0u8; 65536 * 3];
            let mut scalar = vec![0u8; 65536];
            let mut simd = vec![0u8; 65536];
            for r in 0..=255u8 {
                for g in 0..=255u16 {
                    for b in 0..=255u16 {
                        let i = (usize::from(g) * 256 + usize::from(b)) * 3;
                        #[expect(clippy::cast_possible_truncation, reason = "g, b iterate 0..=255")]
                        {
                            src[i] = r;
                            src[i + 1] = g as u8;
                            src[i + 2] = b as u8;
                        }
                    }
                }
                convert_row_scalar(&src, &mut scalar);
                // SAFETY: SSE4.1 presence checked above.
                #[expect(unsafe_code, reason = "runtime-detected SIMD under test")]
                unsafe {
                    convert_row_sse41(&src, &mut simd);
                }
                assert_eq!(scalar, simd, "SIMD diverges from scalar at r={r}");
            }
        }
    }

    /// Row lengths that are not a multiple of the 16-pixel SIMD block
    /// must be finished by the scalar tail without over- or under-run.
    #[test]
    fn ragged_row_lengths_convert_fully() {
        for w in [1u32, 3, 15, 16, 17, 31, 33, 100] {
            let pixels: Vec<[u8; 3]> = (0..w)
                .map(|i| {
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "test pattern bytes wrap intentionally"
                    )]
                    {
                        [(i * 7) as u8, (i * 13) as u8, (i * 29) as u8]
                    }
                })
                .collect();
            let bmp = bitmap_from_pixels(&pixels, w, 1);
            let gray = rgb_to_gray(&bmp);
            let mut want = vec![0u8; w as usize];
            let flat: Vec<u8> = pixels.iter().flatten().copied().collect();
            convert_row_scalar(&flat, &mut want);
            assert_eq!(
                &gray.row_bytes(0)[..w as usize],
                &want[..],
                "width {w} must convert fully"
            );
        }
    }
}
