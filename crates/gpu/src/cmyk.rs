//! CMYK→RGB CPU conversion paths (AVX-512 VBMI, NEON, and scalar fallback).

use color::convert::cmyk_to_rgb_reflectance;

// ── AVX-512 VBMI CMYK→RGB ─────────────────────────────────────────────────────
//
// Subtractive complement R = ⌊((255−C)(255−K) + 127) / 255⌋ (same for G/M and
// B/Y), 32 pixels per call.
//
// The 128 input bytes are widened to four vectors of 32 u16 lanes that keep
// the interleaved [C M Y K] order.  Every lane is complemented (255 − v) and
// the complemented K of each pixel is broadcast over the pixel's quad with
// vpermw (lane j reads lane j | 3), so a single multiply yields
// [R' G' B' K'·K'] per pixel; the fourth lane is discarded.  After the exact
// ⌊(x+127)/255⌋ identity the lanes narrow to bytes as RGBx quads and two
// vpermt2b passes gather the 96 RGB bytes out of the 128-byte RGBx sequence.
//
// Division: exact ⌊(x+127)/255⌋ = (n + (n>>8) + 1) >> 8, n = x+127.
// Valid for n ∈ [0, 65152] (max n = 255²+127 = 65152 < 65280 = 255×256).

/// vpermw table: lane `j` reads lane `j | 3`, the K slot of `j`'s pixel.
#[cfg(target_arch = "x86_64")]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "entries < 32"
)]
const K_LANE_OF_QUAD: [i16; 32] = {
    let mut table = [0i16; 32];
    let mut j = 0;
    while j < 32 {
        table[j] = (j | 3) as i16;
        j += 1;
    }
    table
};

/// vpermt2b table selecting RGB bytes `first..first + 64` out of the two
/// 64-byte `RGBx` vectors: RGB byte `b` lives at `RGBx` byte
/// `(b / 3) * 4 + b % 3`.  Entries for `b ≥ 96` fall outside the 96 stored
/// bytes and are never read.
#[cfg(target_arch = "x86_64")]
#[expect(clippy::cast_possible_truncation, reason = "entries < 128")]
const fn rgbx_to_rgb(first: usize) -> [u8; 64] {
    let mut table = [0u8; 64];
    let mut i = 0;
    while i < 64 {
        let b = first + i;
        table[i] = if b < 96 {
            ((b / 3) * 4 + b % 3) as u8
        } else {
            0
        };
        i += 1;
    }
    table
}

#[cfg(target_arch = "x86_64")]
const RGBX_TO_RGB_LO: [u8; 64] = rgbx_to_rgb(0);
#[cfg(target_arch = "x86_64")]
const RGBX_TO_RGB_HI: [u8; 64] = rgbx_to_rgb(64);

/// Convert 32 CMYK pixels to RGB using AVX-512 u16 arithmetic.
///
/// # Safety
///
/// Caller must ensure `avx512bw` and `avx512vbmi` are available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vbmi")]
unsafe fn cmyk_to_rgb_avx512(cmyk: &[u8; 128], rgb: &mut [u8; 96]) {
    use std::arch::x86_64::{
        __m256i, _mm256_loadu_si256, _mm256_setzero_si256, _mm256_storeu_si256, _mm512_add_epi16,
        _mm512_castsi256_si512, _mm512_castsi512_si256, _mm512_cvtepi16_epi8, _mm512_cvtepu8_epi16,
        _mm512_inserti64x4, _mm512_loadu_si512, _mm512_mullo_epi16, _mm512_permutex2var_epi8,
        _mm512_permutexvar_epi16, _mm512_set1_epi16, _mm512_srli_epi16, _mm512_storeu_si512,
        _mm512_sub_epi16,
    };

    let v255 = _mm512_set1_epi16(255);
    let v127 = _mm512_set1_epi16(127);
    let v1 = _mm512_set1_epi16(1);
    // SAFETY: the three tables are exactly 64 bytes each.
    let (k_lane, gather_lo, gather_hi) = unsafe {
        (
            _mm512_loadu_si512(K_LANE_OF_QUAD.as_ptr().cast()),
            _mm512_loadu_si512(RGBX_TO_RGB_LO.as_ptr().cast()),
            _mm512_loadu_si512(RGBX_TO_RGB_HI.as_ptr().cast()),
        )
    };

    // Four quads of eight pixels: [C M Y K] × 8 per 32 lanes.
    let mut rgbx: [__m256i; 4] = [_mm256_setzero_si256(); 4];
    for (q, out) in rgbx.iter_mut().enumerate() {
        // SAFETY: q * 32 + 32 ≤ 128.
        let v =
            unsafe { _mm512_cvtepu8_epi16(_mm256_loadu_si256(cmyk.as_ptr().add(q * 32).cast())) };
        let inv = _mm512_sub_epi16(v255, v);
        let inv_k = _mm512_permutexvar_epi16(k_lane, inv);
        // prod ≤ 255² = 65025; n = prod + 127 ≤ 65152.
        let n = _mm512_add_epi16(_mm512_mullo_epi16(inv, inv_k), v127);
        let r = _mm512_srli_epi16::<8>(_mm512_add_epi16(
            _mm512_add_epi16(n, _mm512_srli_epi16::<8>(n)),
            v1,
        ));
        // Values ≤ 255: truncation to bytes is exact.
        *out = _mm512_cvtepi16_epi8(r);
    }

    let lo = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(rgbx[0]), rgbx[1]);
    let hi = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(rgbx[2]), rgbx[3]);
    // SAFETY: rgb is exactly 96 bytes: 64 at offset 0, 32 at offset 64.
    unsafe {
        _mm512_storeu_si512(
            rgb.as_mut_ptr().cast(),
            _mm512_permutex2var_epi8(lo, gather_lo, hi),
        );
        _mm256_storeu_si256(
            rgb.as_mut_ptr().add(64).cast(),
            _mm512_castsi512_si256(_mm512_permutex2var_epi8(lo, gather_hi, hi)),
        );
    }
}

// ── ARM NEON CMYK→RGB ─────────────────────────────────────────────────────────
//
// Subtractive complement R = ⌊((255−C)(255−K) + 127) / 255⌋ (same for G/M and
// B/Y), 16 pixels per call.
//
// vld4q_u8 deinterleaves the 64 input bytes into one 16-lane register per
// channel and vst3q_u8 re-interleaves the three results into 48 RGB bytes, so
// no lane is moved by scalar code.  Each complemented channel is multiplied
// by the complemented K with vmull_u8 / vmull_high_u8 (u8 × u8 → u16, 8 lanes
// per half).
//
// Division: ⌊(x + 127) / 255⌋ = (t + 128) >> 8 with t = x + ((x + 128) >> 8),
// exact for every x ∈ [0, 255²]; t ≤ 65279, so neither sum overflows u16.
// vrsraq_n_u16 forms t and vrshrn_n_u16 / vrshrn_high_n_u16 apply the final
// rounding shift while narrowing to bytes.

/// Convert 16 CMYK pixels to RGB using NEON register deinterleave.
///
/// # Safety
///
/// Caller must ensure `neon` is available (mandatory on all ARMv8-A targets).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn cmyk_to_rgb_neon(cmyk: &[u8; 64], rgb: &mut [u8; 48]) {
    use std::arch::aarch64::{
        uint8x16_t, uint8x16x3_t, uint16x8_t, vget_low_u8, vld4q_u8, vmull_high_u8, vmull_u8,
        vmvnq_u8, vrshrn_high_n_u16, vrshrn_n_u16, vrsraq_n_u16, vst3q_u8,
    };

    /// `⌊((255 − ch)(255 − k) + 127) / 255⌋` for 16 lanes; `inv_k = 255 − k`.
    #[inline]
    #[target_feature(enable = "neon")]
    fn blend(ch: uint8x16_t, inv_k: uint8x16_t) -> uint8x16_t {
        let inv = vmvnq_u8(ch);
        let lo: uint16x8_t = vmull_u8(vget_low_u8(inv), vget_low_u8(inv_k));
        let hi: uint16x8_t = vmull_high_u8(inv, inv_k);
        // t = x + ((x + 128) >> 8), then (t + 128) >> 8 narrowed to bytes.
        let lo = vrsraq_n_u16::<8>(lo, lo);
        let hi = vrsraq_n_u16::<8>(hi, hi);
        vrshrn_high_n_u16::<8>(vrshrn_n_u16::<8>(lo), hi)
    }

    // SAFETY: `cmyk` is exactly 64 bytes, the extent of one vld4q_u8.
    let px = unsafe { vld4q_u8(cmyk.as_ptr()) };
    let inv_k = vmvnq_u8(px.3);
    let out = uint8x16x3_t(blend(px.0, inv_k), blend(px.1, inv_k), blend(px.2, inv_k));
    // SAFETY: `rgb` is exactly 48 bytes, the extent of one vst3q_u8.
    unsafe { vst3q_u8(rgb.as_mut_ptr(), out) };
}

// ── Scalar CMYK→RGB ───────────────────────────────────────────────────────────

/// Per-pixel loop: the tail of every SIMD tier and the whole path on targets
/// without one.  LLVM auto-vectorises the loop on x86-64; on aarch64 it
/// compiles to a scalar byte loop.
#[inline]
fn cmyk_to_rgb_scalar(cmyk: &[u8], rgb: &mut [u8]) {
    for (src, dst) in cmyk
        .as_chunks::<4>()
        .0
        .iter()
        .zip(rgb.as_chunks_mut::<3>().0)
    {
        let (r, g, b) = cmyk_to_rgb_reflectance(src[0], src[1], src[2], src[3]);
        dst[0] = r;
        dst[1] = g;
        dst[2] = b;
    }
}

// ── Per-arch dispatch ─────────────────────────────────────────────────────────

/// x86-64: runtime-detect AVX-512 VBMI → scalar.
///
/// Uses `is_x86_feature_detected!` so the same binary works on VBMI machines
/// (Zen 4+, Ice Lake+) and everything older.  A compile-time
/// `#[cfg(target_feature)]` gate would compile out the fallback on native
/// builds, causing SIGILL on machines that lack the compile-time feature.
#[cfg(target_arch = "x86_64")]
#[inline]
fn dispatch_cmyk_matrix(cmyk: &[u8], rgb: &mut [u8]) {
    if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vbmi") {
        // 128 input bytes ↔ 32 pixels ↔ 96 output bytes, so the two chunk
        // iterators run in lockstep and their remainders describe the same
        // tail pixels.
        let (blocks, tail) = cmyk.as_chunks::<128>();
        let (out_blocks, out_tail) = rgb.as_chunks_mut::<96>();
        for (block, out) in blocks.iter().zip(out_blocks) {
            // SAFETY: avx512bw + avx512vbmi confirmed by is_x86_feature_detected! above.
            unsafe { cmyk_to_rgb_avx512(block, out) };
        }
        cmyk_to_rgb_scalar(tail, out_tail);
    } else {
        cmyk_to_rgb_scalar(cmyk, rgb);
    }
}

/// aarch64: NEON is mandatory on all ARMv8-A targets; no runtime detection needed.
#[cfg(target_arch = "aarch64")]
#[inline]
fn dispatch_cmyk_matrix(cmyk: &[u8], rgb: &mut [u8]) {
    // 64 input bytes ↔ 16 pixels ↔ 48 output bytes, so the two chunk
    // iterators run in lockstep and their remainders describe the same
    // tail pixels.
    let (blocks, tail) = cmyk.as_chunks::<64>();
    let (out_blocks, out_tail) = rgb.as_chunks_mut::<48>();
    for (block, out) in blocks.iter().zip(out_blocks) {
        // SAFETY: NEON is mandatory on aarch64.
        unsafe { cmyk_to_rgb_neon(block, out) };
    }
    cmyk_to_rgb_scalar(tail, out_tail);
}

/// Generic fallback for targets without SIMD specialisation.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
fn dispatch_cmyk_matrix(cmyk: &[u8], rgb: &mut [u8]) {
    cmyk_to_rgb_scalar(cmyk, rgb);
}

// ── Public API ────────────────────────────────────────────────────────────────

/// CPU fallback for [`crate::GpuCtx::icc_cmyk_to_rgb`].
///
/// When `clut` is `None`, applies the subtractive complement formula:
///   `R = (255−C)*(255−K)/255` (rounded), same for G/M and B/Y.
///
/// When `clut` is `Some((table, grid_n))`, evaluates the 4D CLUT using
/// quadrilinear interpolation — the same algorithm as the GPU kernel.
///
/// The `clut = None` path uses the best SIMD available at runtime:
/// - x86-64: AVX-512 VBMI (32 px/iter) → scalar
/// - aarch64: NEON (16 px/iter) — mandatory on all ARMv8-A targets
/// - other: scalar per-pixel loop
///
/// # Panics
///
/// Panics if `clut` is `Some((table, grid_n))` and either `grid_n < 2`
/// (fewer than 2 nodes per axis is degenerate and unusable for
/// interpolation) or `table.len() != grid_n⁴ × 3` (the table and grid size
/// disagree — reading through the mismatch would index out of bounds for a
/// short table and silently sample garbage for a long one).
#[must_use]
#[expect(
    clippy::too_many_lines,
    reason = "one function per CLUT/matrix branch with the interpolation inline — \
              splitting would scatter the shared index arithmetic"
)]
pub fn icc_cmyk_to_rgb_cpu(cmyk: &[u8], clut: Option<(&[u8], u32)>) -> Vec<u8> {
    let n = cmyk.len() / 4;
    let mut rgb = vec![0u8; n * 3];

    match clut {
        None => {
            dispatch_cmyk_matrix(cmyk, &mut rgb);
        }
        Some((table, grid_n)) => {
            // grid_n = 0 would cause underflow in `grid_n - 1` and panic via
            // `g - 1` index arithmetic below.  Public function: guard explicitly.
            assert!(
                grid_n >= 2,
                "icc_cmyk_to_rgb_cpu: grid_n must be ≥ 2, got {grid_n}"
            );
            let expected_len = (grid_n as usize)
                .checked_pow(4)
                .and_then(|nodes| nodes.checked_mul(3))
                .unwrap_or_else(|| {
                    panic!("grid_n({grid_n})^4*3 overflows usize — grid_n must be ≤ 255")
                });
            assert_eq!(
                table.len(),
                expected_len,
                "CLUT table length {got} ≠ grid_n({grid_n})^4*3={expected_len}",
                got = table.len(),
            );
            let g = grid_n as usize; // grid_n ≤ 255 from caller validation
            let g2 = g * g;
            let g3 = g2 * g;
            // grid_n ≤ 255 → (grid_n - 1) ≤ 254, exact in f32 (needs ≤ 8 mantissa bits).
            #[expect(
                clippy::cast_precision_loss,
                reason = "grid_n ≤ 255, fits exactly in f32 (8 bits < 23-bit mantissa)"
            )]
            let g1 = (grid_n - 1) as f32;
            let scale = g1 / 255.0;
            for (src, dst) in cmyk
                .as_chunks::<4>()
                .0
                .iter()
                .zip(rgb.as_chunks_mut::<3>().0)
            {
                let fc = f32::from(src[0]) * scale;
                let fm = f32::from(src[1]) * scale;
                let fy = f32::from(src[2]) * scale;
                let fk = f32::from(src[3]) * scale;

                // fc ∈ [0.0, g1] ⊂ [0.0, 254.0]; floor is non-negative and ≤ 254.
                // The sign-loss and truncation lints fire because `as usize` is UB
                // for negative or >usize::MAX floats; here neither can happen.
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "fc/fm/fy/fk are in [0.0, 254.0]; floor is exact and non-negative"
                )]
                let (ic0, im0, iy0, ik0) = (
                    (fc as usize).min(g - 1),
                    (fm as usize).min(g - 1),
                    (fy as usize).min(g - 1),
                    (fk as usize).min(g - 1),
                );
                let ic1 = (ic0 + 1).min(g - 1);
                let im1 = (im0 + 1).min(g - 1);
                let iy1 = (iy0 + 1).min(g - 1);
                let ik1 = (ik0 + 1).min(g - 1);

                // Fractional weights: difference between float position and floored index.
                // ic0 ≤ 254 → ic0 as f32 is exact (fits in 8 mantissa bits).
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "ic0/im0/iy0/ik0 ≤ 254, exact in f32"
                )]
                let (wc, wm, wy, wk) = (
                    fc - ic0 as f32,
                    fm - im0 as f32,
                    fy - iy0 as f32,
                    fk - ik0 as f32,
                );

                let node = |ci: usize, mi: usize, yi: usize, ki: usize| -> [f32; 3] {
                    let idx = (ki * g3 + ci * g2 + mi * g + yi) * 3;
                    [
                        f32::from(table[idx]),
                        f32::from(table[idx + 1]),
                        f32::from(table[idx + 2]),
                    ]
                };

                let lerp = |a: f32, b: f32, t: f32| t.mul_add(b - a, a);
                let lerp3 = |a: [f32; 3], b: [f32; 3], t: f32| -> [f32; 3] {
                    [
                        lerp(a[0], b[0], t),
                        lerp(a[1], b[1], t),
                        lerp(a[2], b[2], t),
                    ]
                };

                // K=0 face: lerp Y → M → C
                let c0m0k0 = lerp3(node(ic0, im0, iy0, ik0), node(ic0, im0, iy1, ik0), wy);
                let c0m1k0 = lerp3(node(ic0, im1, iy0, ik0), node(ic0, im1, iy1, ik0), wy);
                let c1m0k0 = lerp3(node(ic1, im0, iy0, ik0), node(ic1, im0, iy1, ik0), wy);
                let c1m1k0 = lerp3(node(ic1, im1, iy0, ik0), node(ic1, im1, iy1, ik0), wy);
                let rk0 = lerp3(lerp3(c0m0k0, c0m1k0, wm), lerp3(c1m0k0, c1m1k0, wm), wc);

                // K=1 face: lerp Y → M → C
                let c0m0k1 = lerp3(node(ic0, im0, iy0, ik1), node(ic0, im0, iy1, ik1), wy);
                let c0m1k1 = lerp3(node(ic0, im1, iy0, ik1), node(ic0, im1, iy1, ik1), wy);
                let c1m0k1 = lerp3(node(ic1, im0, iy0, ik1), node(ic1, im0, iy1, ik1), wy);
                let c1m1k1 = lerp3(node(ic1, im1, iy0, ik1), node(ic1, im1, iy1, ik1), wy);
                let rk1 = lerp3(lerp3(c0m0k1, c0m1k1, wm), lerp3(c1m0k1, c1m1k1, wm), wc);

                let out = lerp3(rk0, rk1, wk);
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "clamped to [0,255] before cast"
                )]
                {
                    dst[0] = out[0].clamp(0.0, 255.0).round() as u8;
                    dst[1] = out[1].clamp(0.0, 255.0).round() as u8;
                    dst[2] = out[2].clamp(0.0, 255.0).round() as u8;
                }
            }
        }
    }

    rgb
}

#[cfg(test)]
mod tests {
    use super::icc_cmyk_to_rgb_cpu;

    /// Scalar reference: `cmyk_to_rgb_reflectance` per pixel.
    fn reference(cmyk: &[u8]) -> Vec<u8> {
        cmyk.as_chunks::<4>()
            .0
            .iter()
            .flat_map(|px| {
                let (r, g, b) = color::convert::cmyk_to_rgb_reflectance(px[0], px[1], px[2], px[3]);
                <[u8; 3]>::from((r, g, b))
            })
            .collect()
    }

    #[test]
    fn icc_cmyk_matrix_white() {
        let cmyk = [0u8, 0, 0, 0];
        let rgb = icc_cmyk_to_rgb_cpu(&cmyk, None);
        assert_eq!(rgb, [255, 255, 255]);
    }

    #[test]
    fn icc_cmyk_matrix_black() {
        let cmyk = [0u8, 0, 0, 255];
        let rgb = icc_cmyk_to_rgb_cpu(&cmyk, None);
        assert_eq!(rgb, [0, 0, 0]);
    }

    #[test]
    fn icc_cmyk_matrix_pure_cyan() {
        // C=255, M=Y=K=0 → R=0, G=255, B=255
        let cmyk = [255u8, 0, 0, 0];
        let rgb = icc_cmyk_to_rgb_cpu(&cmyk, None);
        assert_eq!(rgb, [0, 255, 255]);
    }

    #[test]
    fn icc_cmyk_matrix_multi_pixel() {
        let cmyk = [0u8, 0, 0, 0, 0, 0, 0, 255];
        let rgb = icc_cmyk_to_rgb_cpu(&cmyk, None);
        assert_eq!(&rgb[0..3], &[255, 255, 255]);
        assert_eq!(&rgb[3..6], &[0, 0, 0]);
    }

    /// Parity: the active SIMD path must match `cmyk_to_rgb_reflectance` byte-for-byte.
    ///
    /// 40 pixels: on x86-64 with AVX-512 VBMI that is one 32-pixel block plus
    /// an 8-pixel scalar tail; on aarch64 two 16-pixel NEON blocks plus an
    /// 8-pixel scalar tail.  On other targets
    /// both sides go scalar and the test is a tautology (still passes).
    #[test]
    fn icc_cmyk_matrix_simd_vs_scalar() {
        #[rustfmt::skip]
        let mut cmyk: Vec<u8> = vec![
            // white, black, cyan, magenta
              0,   0,   0,   0,
              0,   0,   0, 255,
            255,   0,   0,   0,
              0, 255,   0,   0,
            // yellow, key-only mid, all-max, all-mid
              0,   0, 255,   0,
              0,   0,   0, 128,
            255, 255, 255, 255,
            128, 128, 128, 128,
            // mid-range sweep
             64,  32,  16,   8,
            200, 100,  50,  25,
             10,  20,  30,  40,
             50,  60,  70,  80,
             90, 100, 110, 120,
            130, 140, 150, 160,
            170, 180, 190, 200,
            210, 220, 230, 240,
        ];
        // Second half: the same 16 pixels with every channel inverted, then
        // the first eight pixels again as the tail.
        let inverted: Vec<u8> = cmyk.iter().map(|&v| 255 - v).collect();
        cmyk.extend_from_slice(&inverted);
        let tail = cmyk[..32].to_vec();
        cmyk.extend_from_slice(&tail);
        assert_eq!(cmyk.len(), 160, "test vector must be exactly 40 pixels");

        let scalar_rgb = reference(&cmyk);
        let simd_rgb = icc_cmyk_to_rgb_cpu(&cmyk, None);

        for (i, (s, a)) in scalar_rgb.iter().zip(simd_rgb.iter()).enumerate() {
            assert_eq!(
                s,
                a,
                "RGB byte {i} (pixel {}, channel {}): scalar={s} simd={a}",
                i / 3,
                i % 3,
            );
        }
    }

    /// Non-multiple-of-chunk-size: exercises the scalar tail path on all SIMD tiers.
    #[test]
    fn icc_cmyk_matrix_tail_pixels() {
        // 5 pixels: AVX-512 processes 0 full blocks (< 32), NEON processes 0 (< 16).
        // All 5 pixels fall into the scalar tail on every arch.
        let cmyk: Vec<u8> = (0u8..5)
            .flat_map(|i| {
                let v = i.wrapping_mul(50);
                [
                    v,
                    v.wrapping_add(10),
                    v.wrapping_add(20),
                    v.wrapping_add(30),
                ]
            })
            .collect();
        assert_eq!(icc_cmyk_to_rgb_cpu(&cmyk, None), reference(&cmyk));
    }

    /// 16 pixels exactly — one full NEON block, zero AVX-512 blocks.
    #[test]
    fn icc_cmyk_matrix_sixteen_pixels() {
        let cmyk: Vec<u8> = (0u8..16)
            .flat_map(|i| {
                [
                    i.wrapping_mul(30),
                    i.wrapping_mul(20),
                    i.wrapping_mul(10),
                    i.wrapping_mul(5),
                ]
            })
            .collect();
        assert_eq!(cmyk.len(), 64);
        assert_eq!(icc_cmyk_to_rgb_cpu(&cmyk, None), reference(&cmyk));
    }

    /// NEON kernel vs the scalar reference over every (ink, K) byte pair.
    ///
    /// Pixel `i` carries ink `i & 0xFF` on C, its complement on M, a
    /// permuted value on Y and K `i >> 8`, so each channel's product
    /// covers all 256 × 256 operand pairs across the 4096 blocks.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn icc_cmyk_neon_matches_scalar_exhaustive() {
        let cmyk: Vec<u8> = (0u32..65_536)
            .flat_map(|i| {
                let ink = i.to_le_bytes()[0];
                let k = i.to_le_bytes()[1];
                [ink, 255 - ink, ink.wrapping_mul(167), k]
            })
            .collect();
        let expected = reference(&cmyk);
        let mut got = vec![0u8; expected.len()];
        for (block, out) in cmyk
            .as_chunks::<64>()
            .0
            .iter()
            .zip(got.as_chunks_mut::<48>().0)
        {
            // SAFETY: NEON is mandatory on aarch64.
            unsafe { super::cmyk_to_rgb_neon(block, out) };
        }
        assert_eq!(got, expected);
    }

    /// AVX-512-specific: direct call with 200 pseudorandom 32-pixel blocks.
    ///
    /// The dispatcher-driven tests exercise only the highest tier the
    /// running machine reports, so on a machine without VBMI the kernel is
    /// never executed; this direct call is its only coverage there is.
    /// Runtime-skipped on machines without AVX-512 VBMI.  Many random blocks
    /// rather than one fixed vector, so a per-lane table error cannot slip
    /// through on a lucky input.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn icc_cmyk_avx512_matches_scalar() {
        if !is_x86_feature_detected!("avx512bw") || !is_x86_feature_detected!("avx512vbmi") {
            return;
        }
        let mut rng = 0x1357_9bdf_u32;
        let mut next = move || {
            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (rng >> 24) as u8
        };
        for block_i in 0..200 {
            let cmyk: [u8; 128] = std::array::from_fn(|_| next());
            let expected = reference(&cmyk);
            let mut got = [0u8; 96];
            // SAFETY: avx512bw + avx512vbmi confirmed present above.
            unsafe { super::cmyk_to_rgb_avx512(&cmyk, &mut got) };
            assert_eq!(
                got.as_slice(),
                expected.as_slice(),
                "AVX-512 CMYK→RGB mismatch on block {block_i}"
            );
        }
    }

    /// A (table, `grid_n`) pair whose lengths disagree must fail at the API
    /// boundary with a diagnostic naming both, not deep inside the
    /// interpolation closure with a bare slice index.
    #[test]
    #[should_panic(expected = "CLUT table length")]
    fn icc_cmyk_clut_rejects_mismatched_table_length() {
        let table = vec![0u8; 17usize.pow(4) * 3];
        let _ = icc_cmyk_to_rgb_cpu(&[0, 0, 0, 255], Some((&table, 33)));
    }

    #[test]
    fn icc_cmyk_clut_identity_corners() {
        // 2^4 = 16-node CLUT where output = matrix formula at corners.
        const G: u8 = 2;
        let g_usize = G as usize;
        let mut table = vec![0u8; g_usize * g_usize * g_usize * g_usize * 3];
        for ki in 0..G {
            for ci in 0..G {
                for mi in 0..G {
                    for yi in 0..G {
                        let idx = (usize::from(ki) * g_usize * g_usize * g_usize
                            + usize::from(ci) * g_usize * g_usize
                            + usize::from(mi) * g_usize
                            + usize::from(yi))
                            * 3;
                        // ki/ci/mi/yi ∈ {0, 1}; * 255 ∈ {0, 255} — fits in u8.
                        let c = ci * 255;
                        let m = mi * 255;
                        let y = yi * 255;
                        let k = ki * 255;
                        let inv_k = u32::from(255 - k);
                        // Each `((255 - x) * inv_k) / 255` ∈ [0, 255]; `try_into`
                        // makes the bound explicit so the lint stays satisfied.
                        table[idx] = u8::try_from((u32::from(255 - c) * inv_k) / 255)
                            .expect("by construction <=255");
                        table[idx + 1] = u8::try_from((u32::from(255 - m) * inv_k) / 255)
                            .expect("by construction <=255");
                        table[idx + 2] = u8::try_from((u32::from(255 - y) * inv_k) / 255)
                            .expect("by construction <=255");
                    }
                }
            }
        }
        let rgb = icc_cmyk_to_rgb_cpu(&[0u8, 0, 0, 0], Some((&table, 2)));
        assert_eq!(rgb, [255, 255, 255], "white corner");
        let rgb = icc_cmyk_to_rgb_cpu(&[0u8, 0, 0, 255], Some((&table, 2)));
        assert_eq!(rgb, [0, 0, 0], "black corner");
    }
}
