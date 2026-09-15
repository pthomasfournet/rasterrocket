//! AA compositing: Porter-Duff source-over with per-pixel coverage.
//!
//! Single public entry: [`composite_aa_rgb8_opaque`] — RGB pixels with no
//! alpha plane (opaque destination).  Per channel,
//! `c = div255((255 − a_src) · c_dst + a_src · c_src)` with
//! `a_src = div255(a_input · shape)`.
//!
//! # Tiers (x86-64)
//!
//! 1. **AVX-512 BW+VL**, spans ≥ 32 px: 32 pixels (96 destination bytes) per
//!    iteration.  The destination bytes stay interleaved: each 32-byte third
//!    of the block is widened to 32 u16 lanes, its per-lane `a_src` is drawn
//!    from the block's 32-pixel alpha vector with `vpermw` (lane `k` of third
//!    `g` belongs to pixel `(32g + k) / 3`) and its per-lane source colour is
//!    a constant permutation of the RGB triple — no deinterleave or
//!    re-interleave.  Remainder pixels go through the scalar loop.
//! 2. **Scalar** per-pixel loop, also the tier for every other architecture.
//!    LLVM auto-vectorises it under `-C target-cpu=native`.
//!
//! # div255
//!
//! `(v + (v >> 8) + 0x80) >> 8` computes `v / 255` exactly for the operand
//! range here (`v ≤ 255²`), matching `color::convert::div255` in the general
//! path bit for bit, so output does not depend on which tier a span takes.
//! It stays pure u16 lane arithmetic (max intermediate 65 407).

/// Exact `v / 255` (rounded to nearest) for `v` in `[0, 255²]`.
///
/// Same formula as `color::convert::div255`; the intermediate
/// `v + (v >> 8) + 0x80` peaks at 65 407 for `v = 65 025`, so u16 cannot
/// overflow and no clamp is needed (the result is ≤ 255 on this range).
#[inline]
const fn div255_u16(v: u16) -> u16 {
    (v + (v >> 8) + 0x80) >> 8
}

/// Composite a solid RGB source over an opaque destination (no alpha plane).
///
/// The destination is assumed fully opaque (`a_dst = 255`).  The compositing
/// formula simplifies to:
/// ```text
/// c_result = div255((255 - a_src) * c_dst + a_src * c_src)
/// ```
///
/// # Arguments
///
/// - `dst`: packed RGB bytes, length must be `shape.len() * 3`.
/// - `src`: constant source colour applied to every pixel.
/// - `a_input`: source opacity (0 = transparent, 255 = opaque).
/// - `shape`: per-pixel AA coverage, one byte per pixel.
///
/// # Panics (debug only)
///
/// Panics if `dst.len() != shape.len() * 3`.
pub fn composite_aa_rgb8_opaque(dst: &mut [u8], src: [u8; 3], a_input: u8, shape: &[u8]) {
    debug_assert_eq!(
        dst.len(),
        shape.len() * 3,
        "composite_aa_rgb8_opaque: dst length mismatch (got {}, expected {})",
        dst.len(),
        shape.len() * 3,
    );
    dispatch_composite(dst, src, a_input, shape);
}

// ── Scalar tier ───────────────────────────────────────────────────────────────

/// Per-pixel loop; the reference every other tier must match byte for byte.
fn composite_scalar(dst: &mut [u8], src: [u8; 3], a_input: u8, shape: &[u8]) {
    let a_in = u16::from(a_input);
    let [sr, sg, sb] = src.map(u16::from);
    for (&cov, px) in shape.iter().zip(dst.as_chunks_mut::<3>().0) {
        let a_src = div255_u16(a_in * u16::from(cov));
        let inv = 255 - a_src;
        #[expect(clippy::cast_possible_truncation, reason = "div255_u16 result ≤ 255")]
        {
            px[0] = div255_u16(inv * u16::from(px[0]) + a_src * sr) as u8;
            px[1] = div255_u16(inv * u16::from(px[1]) + a_src * sg) as u8;
            px[2] = div255_u16(inv * u16::from(px[2]) + a_src * sb) as u8;
        }
    }
}

// ── AVX-512 tier ──────────────────────────────────────────────────────────────

/// Pixels per AVX-512 block.
#[cfg(target_arch = "x86_64")]
const BLOCK_PX: usize = 32;

/// Lane tables for the three 32-byte thirds of a 32-pixel block: byte `b`
/// of the block belongs to pixel `b / 3` and carries channel `b % 3`.
#[cfg(target_arch = "x86_64")]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "every entry is < 96"
)]
const fn lane_table(third: usize, divide: bool) -> [i16; 32] {
    let mut table = [0i16; 32];
    let mut k = 0;
    while k < 32 {
        let b = third * 32 + k;
        table[k] = if divide {
            (b / 3) as i16
        } else {
            (b % 3) as i16
        };
        k += 1;
    }
    table
}

#[cfg(target_arch = "x86_64")]
const LANE_PIXEL: [[i16; 32]; 3] = [
    lane_table(0, true),
    lane_table(1, true),
    lane_table(2, true),
];

#[cfg(target_arch = "x86_64")]
const LANE_CHANNEL: [[i16; 32]; 3] = [
    lane_table(0, false),
    lane_table(1, false),
    lane_table(2, false),
];

/// AVX-512 tier: whole 32-pixel blocks, scalar loop for the remainder.
///
/// # Safety
///
/// Caller must ensure `avx512bw` and `avx512vl` are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vl")]
unsafe fn composite_avx512(dst: &mut [u8], src: [u8; 3], a_input: u8, shape: &[u8]) {
    use std::arch::x86_64::{
        __m512i, _mm256_loadu_si256, _mm256_storeu_si256, _mm512_add_epi16, _mm512_cvtepi16_epi8,
        _mm512_cvtepu8_epi16, _mm512_loadu_si512, _mm512_mullo_epi16, _mm512_permutexvar_epi16,
        _mm512_set1_epi16, _mm512_set1_epi64, _mm512_srli_epi16, _mm512_sub_epi16,
    };

    // Only whole blocks that both slices cover go through the vector path.
    let n = shape.len().min(dst.len() / 3);
    let blocks = n / BLOCK_PX;

    let a_in = _mm512_set1_epi16(i16::from(a_input));
    let v255 = _mm512_set1_epi16(255);
    let v128 = _mm512_set1_epi16(0x80);
    // Exact v / 255 on u16 lanes: (v + (v >> 8) + 0x80) >> 8.
    let div255 = |v: __m512i| {
        _mm512_srli_epi16::<8>(_mm512_add_epi16(
            _mm512_add_epi16(v, _mm512_srli_epi16::<8>(v)),
            v128,
        ))
    };
    // The RGB triple in u16 lanes 0..=2 of every qword; LANE_CHANNEL selects
    // from those three lanes per byte position.
    let rgb = _mm512_set1_epi64(
        i64::from(src[0]) | (i64::from(src[1]) << 16) | (i64::from(src[2]) << 32),
    );
    // SAFETY: the tables are exactly 64 bytes each.
    let (pixel_of, src_of) = unsafe {
        let load = |t: &[i16; 32]| _mm512_loadu_si512(t.as_ptr().cast());
        (
            [
                load(&LANE_PIXEL[0]),
                load(&LANE_PIXEL[1]),
                load(&LANE_PIXEL[2]),
            ],
            [
                _mm512_permutexvar_epi16(load(&LANE_CHANNEL[0]), rgb),
                _mm512_permutexvar_epi16(load(&LANE_CHANNEL[1]), rgb),
                _mm512_permutexvar_epi16(load(&LANE_CHANNEL[2]), rgb),
            ],
        )
    };

    let sp = shape.as_ptr();
    let dp = dst.as_mut_ptr();
    for block in 0..blocks {
        let px = block * BLOCK_PX;
        // SAFETY: px + 32 ≤ n ≤ shape.len().
        let cov = unsafe { _mm256_loadu_si256(sp.add(px).cast()) };
        let a_src = div255(_mm512_mullo_epi16(_mm512_cvtepu8_epi16(cov), a_in));
        for third in 0..3 {
            let off = px * 3 + third * 32;
            // SAFETY: off + 32 ≤ 3n ≤ dst.len().
            unsafe {
                let d = _mm512_cvtepu8_epi16(_mm256_loadu_si256(dp.add(off).cast()));
                let a = _mm512_permutexvar_epi16(pixel_of[third], a_src);
                let inv = _mm512_sub_epi16(v255, a);
                let v = _mm512_add_epi16(
                    _mm512_mullo_epi16(inv, d),
                    _mm512_mullo_epi16(a, src_of[third]),
                );
                _mm256_storeu_si256(dp.add(off).cast(), _mm512_cvtepi16_epi8(div255(v)));
            }
        }
    }

    let done = blocks * BLOCK_PX;
    composite_scalar(&mut dst[done * 3..], src, a_input, &shape[done..]);
}

// ── Per-arch dispatch ─────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[inline]
fn dispatch_composite(dst: &mut [u8], src: [u8; 3], a_input: u8, shape: &[u8]) {
    if shape.len() >= BLOCK_PX
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vl")
    {
        // SAFETY: both features confirmed present.
        unsafe { composite_avx512(dst, src, a_input, shape) };
    } else {
        composite_scalar(dst, src, a_input, shape);
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn dispatch_composite(dst: &mut [u8], src: [u8; 3], a_input: u8, shape: &[u8]) {
    composite_scalar(dst, src, a_input, shape);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudorandom(n: usize, seed: u32) -> Vec<u8> {
        let mut rng = seed;
        (0..n)
            .map(|_| {
                rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                #[expect(clippy::cast_possible_truncation, reason = "top byte of a u32")]
                {
                    (rng >> 24) as u8
                }
            })
            .collect()
    }

    /// Pixel-by-pixel reference written independently of the tiers.
    fn reference(dst: &mut [u8], src: [u8; 3], a_input: u8, shape: &[u8]) {
        let a_in = u16::from(a_input);
        let [sr, sg, sb] = [u16::from(src[0]), u16::from(src[1]), u16::from(src[2])];
        for (shape_v, px) in shape.iter().zip(dst.chunks_exact_mut(3)) {
            let a_src = div255_u16(a_in * u16::from(*shape_v));
            let inv = 255 - a_src;
            #[expect(clippy::cast_possible_truncation, reason = "div255_u16 result ≤ 255")]
            {
                px[0] = div255_u16(inv * u16::from(px[0]) + a_src * sr) as u8;
                px[1] = div255_u16(inv * u16::from(px[1]) + a_src * sg) as u8;
                px[2] = div255_u16(inv * u16::from(px[2]) + a_src * sb) as u8;
            }
        }
    }

    #[test]
    fn opaque_full_coverage_writes_src() {
        let src = [200u8, 100, 50];
        let shape = [255u8; 4];
        let mut dst = vec![10u8; 12]; // 4 pixels

        composite_aa_rgb8_opaque(&mut dst, src, 255, &shape);

        for i in 0..4 {
            assert_eq!(&dst[i * 3..i * 3 + 3], &[200, 100, 50], "pixel {i}");
        }
    }

    #[test]
    fn opaque_zero_coverage_leaves_dst() {
        let src = [200u8, 100, 50];
        let shape = [0u8; 4];
        let original: Vec<u8> = (0..12).map(|i: u8| i * 10).collect();
        let mut dst = original.clone();

        composite_aa_rgb8_opaque(&mut dst, src, 255, &shape);

        assert_eq!(dst, original);
    }

    #[test]
    fn opaque_half_coverage_blends() {
        let src = [255u8, 255, 255];
        let shape = [128u8];
        let mut dst = vec![0u8; 3]; // black dst

        composite_aa_rgb8_opaque(&mut dst, src, 255, &shape);

        // div255_u16(128 * 255) ≈ 128; blend: div255_u16((255-128)*0 + 128*255) ≈ 128.
        let v = dst[0];
        assert!((125..=131).contains(&v), "expected ~128, got {v}");
    }

    /// Dispatcher-driven parity at every remainder of the 32-pixel block, plus
    /// long spans, across a spread of source colours and opacities.
    #[test]
    fn dispatch_matches_reference_at_every_length() {
        let cases = [
            ([100u8, 150, 200], 200u8),
            ([0, 0, 0], 255),
            ([255, 255, 255], 1),
            ([7, 250, 128], 128),
        ];
        for (src, a_input) in cases {
            for n in (0..=130).chain([255, 256, 257, 1000, 2500]) {
                #[expect(clippy::cast_possible_truncation, reason = "n ≤ 2500")]
                let seed = 0x1234_5678 ^ (n as u32);
                let shape = pseudorandom(n, seed);
                let initial = pseudorandom(n * 3, seed.rotate_left(7));

                let mut want = initial.clone();
                reference(&mut want, src, a_input, &shape);

                let mut got = initial;
                composite_aa_rgb8_opaque(&mut got, src, a_input, &shape);

                assert_eq!(got, want, "src={src:?} a_input={a_input} n={n}");
            }
        }
    }

    /// Direct call into the AVX-512 tier with slices that are not a whole
    /// number of blocks, so its own scalar remainder is exercised regardless
    /// of the dispatcher's width gate.  Skipped where the CPU lacks it.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_tier_matches_reference() {
        if !is_x86_feature_detected!("avx512bw") || !is_x86_feature_detected!("avx512vl") {
            return;
        }
        for n in [1usize, 31, 32, 33, 63, 64, 65, 96, 100, 129] {
            #[expect(clippy::cast_possible_truncation, reason = "n ≤ 129")]
            let seed = 0x9e37_79b9 ^ (n as u32);
            let shape = pseudorandom(n, seed);
            let initial = pseudorandom(n * 3, seed.rotate_left(11));
            let src = [33u8, 66, 99];

            let mut want = initial.clone();
            reference(&mut want, src, 180, &shape);

            let mut got = initial;
            // SAFETY: avx512bw + avx512vl confirmed present above.
            unsafe { composite_avx512(&mut got, src, 180, &shape) };

            assert_eq!(got, want, "n={n}");
        }
    }

    #[test]
    fn opaque_empty_is_noop() {
        let mut dst: Vec<u8> = vec![];
        composite_aa_rgb8_opaque(&mut dst, [1, 2, 3], 255, &[]);
    }
}
