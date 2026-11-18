//! AA compositing: Porter-Duff source-over with per-pixel coverage.
//!
//! Single public entry: [`composite_aa_rgb8_opaque`] — RGB pixels with no
//! alpha plane (opaque destination).  Per channel,
//! `c = div255((255 − a_src) · c_dst + a_src · c_src)` with
//! `a_src = div255(a_input · shape)`.
//!
//! # Tiers (x86-64)
//!
//! 1. **AVX-512 BW+VL**, spans ≥ 2 px: 32 pixels (96 destination bytes) per
//!    iteration.  The destination bytes stay interleaved: each 32-byte third
//!    of the block is widened to 32 u16 lanes, its per-lane `a_src` is drawn
//!    from the block's 32-pixel alpha vector with `vpermw` (lane `k` of third
//!    `g` belongs to pixel `(32g + k) / 3`) and its per-lane source colour is
//!    a constant permutation of the RGB triple — no deinterleave or
//!    re-interleave.  A remainder of 2–31 pixels is one further block under
//!    byte masks (`vmovdqu8` with a lane mask: unselected bytes are neither
//!    read nor written); a single remaining pixel goes through the scalar
//!    loop.
//! 2. **AVX2** (with the `simd-avx2` feature), spans ≥ 32 px: 32 pixels (96
//!    destination bytes) per iteration, the destination again kept
//!    interleaved.  The block's 32 `a_src` values are packed to bytes and
//!    each 32-byte third takes its per-byte alpha by an in-lane byte
//!    shuffle (`vpshufb`, byte `b` of third `g` from pixel `(32g + b) / 3`)
//!    of a fixed arrangement of the alpha vector's 128-bit halves; the
//!    per-lane source colour of each third is a constant vector.  A
//!    remainder of 1–31 pixels is one more block ending at the span's last
//!    pixel, overlapping the previous block.
//! 3. **Scalar** per-pixel loop, also the tier for every other architecture,
//!    for one-pixel spans and, under AVX2 only, for spans below 32 px.
//!    LLVM auto-vectorises it under `-C target-cpu=native`.
//!
//! # div255
//!
//! `(t + (t >> 8)) >> 8` with `t = v + 0x80` computes `v / 255` rounded to
//! nearest for the operand range here (`v ≤ 255²`), matching
//! `color::convert::div255` in the general path bit for bit, so output does
//! not depend on which tier a span takes.  It stays pure u16 lane
//! arithmetic (max intermediate 65 407).

/// Exact `v / 255` (rounded to nearest) for `v` in `[0, 255²]`.
///
/// Same formula as `color::convert::div255`; the intermediate
/// `t + (t >> 8)`, `t = v + 0x80`, peaks at 65 407 for `v = 65 025`, so u16
/// cannot overflow and no clamp is needed (the result is ≤ 255 on this
/// range).
#[inline]
const fn div255_u16(v: u16) -> u16 {
    let t = v + 0x80;
    (t + (t >> 8)) >> 8
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

/// Pixels per block of the AVX-512 and AVX2 tiers: 96 destination bytes,
/// three 32-byte thirds.
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

/// Shortest remainder the AVX-512 tier hands to the masked block; a shorter
/// one (a single pixel) takes the scalar loop, which is cheaper there.
#[cfg(target_arch = "x86_64")]
const MASKED_MIN_PX: usize = 2;

/// AVX-512 tier: whole 32-pixel blocks, then the remainder of 2–31 pixels as
/// one more block under byte masks, so no byte past the span is read or
/// written.  A one-pixel remainder takes the scalar loop.
///
/// # Safety
///
/// Caller must ensure `avx512bw` and `avx512vl` are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vl")]
unsafe fn composite_avx512(dst: &mut [u8], src: [u8; 3], a_input: u8, shape: &[u8]) {
    use std::arch::x86_64::{
        __m256i, __m512i, __mmask32, _mm256_loadu_si256, _mm256_mask_storeu_epi8,
        _mm256_maskz_loadu_epi8, _mm256_storeu_si256, _mm512_add_epi16, _mm512_cvtepi16_epi8,
        _mm512_cvtepu8_epi16, _mm512_loadu_si512, _mm512_mullo_epi16, _mm512_permutexvar_epi16,
        _mm512_set1_epi16, _mm512_set1_epi64, _mm512_srli_epi16, _mm512_sub_epi16,
    };

    /// Mask of the low `min(bits, 32)` lanes.
    const fn low_bits(bits: usize) -> __mmask32 {
        if bits >= 32 { !0 } else { (1 << bits) - 1 }
    }

    // Only pixels that both slices cover are processed.
    let n = shape.len().min(dst.len() / 3);
    let blocks = n / BLOCK_PX;

    let a_in = _mm512_set1_epi16(i16::from(a_input));
    let v255 = _mm512_set1_epi16(255);
    let v128 = _mm512_set1_epi16(0x80);
    // v / 255 rounded to nearest on u16 lanes: (t + (t >> 8)) >> 8, t = v + 0x80.
    let div255 = |v: __m512i| {
        let t = _mm512_add_epi16(v, v128);
        _mm512_srli_epi16::<8>(_mm512_add_epi16(t, _mm512_srli_epi16::<8>(t)))
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
    // Per-pixel a_src of a block's 32 coverage bytes.
    let alpha = |cov: __m256i| div255(_mm512_mullo_epi16(_mm512_cvtepu8_epi16(cov), a_in));
    // Composited bytes of one 32-byte third `d` of a block.
    let blend = |d: __m256i, a_src: __m512i, third: usize| {
        let a = _mm512_permutexvar_epi16(pixel_of[third], a_src);
        let inv = _mm512_sub_epi16(v255, a);
        let v = _mm512_add_epi16(
            _mm512_mullo_epi16(inv, _mm512_cvtepu8_epi16(d)),
            _mm512_mullo_epi16(a, src_of[third]),
        );
        _mm512_cvtepi16_epi8(div255(v))
    };

    let sp = shape.as_ptr();
    let dp = dst.as_mut_ptr();
    for block in 0..blocks {
        let px = block * BLOCK_PX;
        // SAFETY: px + 32 ≤ n ≤ shape.len().
        let a_src = alpha(unsafe { _mm256_loadu_si256(sp.add(px).cast()) });
        for third in 0..3 {
            let off = px * 3 + third * 32;
            // SAFETY: off + 32 ≤ 3n ≤ dst.len().
            unsafe {
                let d = _mm256_loadu_si256(dp.add(off).cast());
                _mm256_storeu_si256(dp.add(off).cast(), blend(d, a_src, third));
            }
        }
    }

    let done = blocks * BLOCK_PX;
    let rest = n - done;
    if rest < MASKED_MIN_PX {
        composite_scalar(&mut dst[done * 3..], src, a_input, &shape[done..]);
        return;
    }
    // SAFETY: the masked load reads only coverage bytes done .. n.
    let a_src = alpha(unsafe { _mm256_maskz_loadu_epi8(low_bits(rest), sp.add(done).cast()) });
    let bytes = rest * 3;
    for third in 0..3 {
        let first = third * 32;
        if first >= bytes {
            break;
        }
        let k = low_bits(bytes - first);
        let off = done * 3 + first;
        // SAFETY: the masked load and store touch only destination bytes
        // off .. 3n, all inside dst.
        unsafe {
            let d = _mm256_maskz_loadu_epi8(k, dp.add(off).cast());
            _mm256_mask_storeu_epi8(dp.add(off).cast(), k, blend(d, a_src, third));
        }
    }
}

// ── AVX2 tier ─────────────────────────────────────────────────────────────────

/// Byte-shuffle tables for the three 32-byte thirds of a 32-pixel block:
/// byte `b` of third `g` belongs to pixel `q = (32g + b) / 3`, which sits at
/// byte `q % 16` of 128-bit half `q / 16` of the alpha vector.  Third 0
/// draws on half 0 only, third 2 on half 1 only, third 1 on half 0 in its
/// low lane and half 1 in its high lane, so each third shuffles a fixed
/// arrangement of the alpha halves within its lanes.
#[cfg(all(target_arch = "x86_64", feature = "simd-avx2"))]
const SHUFFLE_PIXEL: [[u8; 32]; 3] = {
    let mut t = [[0u8; 32]; 3];
    let mut g = 0;
    while g < 3 {
        let mut b = 0;
        while b < 32 {
            #[expect(clippy::cast_possible_truncation, reason = "every entry is < 16")]
            {
                t[g][b] = (((32 * g + b) / 3) % 16) as u8;
            }
            b += 1;
        }
        g += 1;
    }
    t
};

/// Byte position, within a 32-byte third, of u16 lane `j` of the low
/// (`high = false`) or high widening of that third: the in-lane byte
/// unpacks take bytes 0–7 and 16–23 to the low widening, 8–15 and 24–31 to
/// the high one.
#[cfg(all(target_arch = "x86_64", feature = "simd-avx2"))]
const fn widened_byte(j: usize, high: bool) -> usize {
    (j / 8) * 16 + (j % 8) + if high { 8 } else { 0 }
}

/// AVX2 tier: 32 pixels (96 destination bytes) per block.  A span of
/// fewer than 32 pixels takes the scalar loop.  A remainder of 1–31 pixels
/// is covered by one more block ending at the span's last pixel, computed
/// from the destination before any block is stored and stored last: each
/// output pixel depends only on its own input, so the pixels it shares
/// with the preceding block receive the same values twice.
///
/// Bytes are widened to u16 with in-lane unpacks against zero and narrowed
/// back with an in-lane unsigned pack, which restores their order, so no
/// lane crossing is needed except to arrange the alpha halves per third.
/// `div255` here is `mulhi_u16(v + 0x80, 257)`: for `t = v + 0x80 < 2¹⁶`,
/// `⌊257t / 2¹⁶⌋ = ⌊(t + t/256) / 256⌋ = ⌊(t + ⌊t/256⌋) / 256⌋`, the
/// scalar formula.
///
/// # Safety
///
/// Caller must ensure `avx2` is available.
#[cfg(all(target_arch = "x86_64", feature = "simd-avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn composite_avx2(dst: &mut [u8], src: [u8; 3], a_input: u8, shape: &[u8]) {
    use std::arch::x86_64::{
        __m256i, _mm256_add_epi16, _mm256_loadu_si256, _mm256_mulhi_epu16, _mm256_mullo_epi16,
        _mm256_packus_epi16, _mm256_permute2x128_si256, _mm256_set1_epi16, _mm256_setzero_si256,
        _mm256_shuffle_epi8, _mm256_storeu_si256, _mm256_sub_epi16, _mm256_unpackhi_epi8,
        _mm256_unpacklo_epi8,
    };

    // Only pixels that both slices cover are processed.
    let n = shape.len().min(dst.len() / 3);
    if n < BLOCK_PX {
        composite_scalar(dst, src, a_input, shape);
        return;
    }

    let zero = _mm256_setzero_si256();
    let a_in = _mm256_set1_epi16(i16::from(a_input));
    let v255 = _mm256_set1_epi16(255);
    let v128 = _mm256_set1_epi16(0x80);
    let v257 = _mm256_set1_epi16(257);
    let div255 = |v: __m256i| _mm256_mulhi_epu16(_mm256_add_epi16(v, v128), v257);
    // The per-lane source colour of each third's low and high widening.
    let colour = |g: usize, high: bool| -> [i16; 16] {
        std::array::from_fn(|j| i16::from(src[(32 * g + widened_byte(j, high)) % 3]))
    };
    let colours: [[[i16; 16]; 2]; 3] = std::array::from_fn(|g| [colour(g, false), colour(g, true)]);
    // SAFETY: each shuffle and colour table is exactly 32 bytes.
    let (pixel_of, src_of) = unsafe {
        (
            SHUFFLE_PIXEL.map(|t| _mm256_loadu_si256(t.as_ptr().cast())),
            colours.map(|c| c.map(|t| _mm256_loadu_si256(t.as_ptr().cast()))),
        )
    };

    let sp = shape.as_ptr();
    let dp = dst.as_mut_ptr();
    // The composited 96 bytes of the block of pixels `px .. px + 32`.
    let block = |px: usize| -> [__m256i; 3] {
        debug_assert!(px + BLOCK_PX <= n, "block past the span: px={px} n={n}");
        // SAFETY: px + 32 ≤ n ≤ shape.len() and 3(px + 32) ≤ 3n ≤ dst.len().
        unsafe {
            let cov = _mm256_loadu_si256(sp.add(px).cast());
            let lo = div255(_mm256_mullo_epi16(_mm256_unpacklo_epi8(cov, zero), a_in));
            let hi = div255(_mm256_mullo_epi16(_mm256_unpackhi_epi8(cov, zero), a_in));
            let a_src = _mm256_packus_epi16(lo, hi);
            let halves = [
                _mm256_permute2x128_si256::<0x00>(a_src, a_src),
                a_src,
                _mm256_permute2x128_si256::<0x11>(a_src, a_src),
            ];
            let mix = |a: __m256i, d: __m256i, s: __m256i| {
                let inv = _mm256_sub_epi16(v255, a);
                div255(_mm256_add_epi16(
                    _mm256_mullo_epi16(inv, d),
                    _mm256_mullo_epi16(a, s),
                ))
            };
            std::array::from_fn(|g| {
                let a = _mm256_shuffle_epi8(halves[g], pixel_of[g]);
                let d = _mm256_loadu_si256(dp.add(px * 3 + g * 32).cast());
                let lo = mix(
                    _mm256_unpacklo_epi8(a, zero),
                    _mm256_unpacklo_epi8(d, zero),
                    src_of[g][0],
                );
                let hi = mix(
                    _mm256_unpackhi_epi8(a, zero),
                    _mm256_unpackhi_epi8(d, zero),
                    src_of[g][1],
                );
                _mm256_packus_epi16(lo, hi)
            })
        }
    };
    let store = |px: usize, out: [__m256i; 3]| {
        debug_assert!(px + BLOCK_PX <= n, "block past the span: px={px} n={n}");
        for (g, v) in out.into_iter().enumerate() {
            // SAFETY: 3(px + 32) ≤ 3n ≤ dst.len().
            unsafe { _mm256_storeu_si256(dp.add(px * 3 + g * 32).cast(), v) };
        }
    };

    let last = n - BLOCK_PX;
    let tail = (!n.is_multiple_of(BLOCK_PX)).then(|| block(last));
    for b in 0..n / BLOCK_PX {
        let px = b * BLOCK_PX;
        store(px, block(px));
    }
    if let Some(out) = tail {
        store(last, out);
    }
}

// ── Per-arch dispatch ─────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[inline]
fn dispatch_composite(dst: &mut [u8], src: [u8; 3], a_input: u8, shape: &[u8]) {
    if shape.len() >= MASKED_MIN_PX
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vl")
    {
        // SAFETY: both features confirmed present.
        unsafe { composite_avx512(dst, src, a_input, shape) };
        return;
    }
    #[cfg(feature = "simd-avx2")]
    if shape.len() >= BLOCK_PX && is_x86_feature_detected!("avx2") {
        // SAFETY: avx2 confirmed present.
        unsafe { composite_avx2(dst, src, a_input, shape) };
        return;
    }
    composite_scalar(dst, src, a_input, shape);
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
                (rng >> 24) as u8
            })
            .collect()
    }

    /// Pixel-by-pixel reference written independently of the tiers.
    fn reference(dst: &mut [u8], src: [u8; 3], a_input: u8, shape: &[u8]) {
        let a_in = u16::from(a_input);
        let [sr, sg, sb] = [u16::from(src[0]), u16::from(src[1]), u16::from(src[2])];
        for (shape_v, px) in shape.iter().zip(dst.as_chunks_mut::<3>().0) {
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

    /// A tier's entry point, as the dispatcher calls it.
    #[cfg(target_arch = "x86_64")]
    type Tier = unsafe fn(&mut [u8], [u8; 3], u8, &[u8]);

    /// Call `tier` directly for every span length of `lengths`, 16 trials
    /// each from `seed`, against the scalar tier, over pseudorandom
    /// coverage, destination, source colour and opacity.  Both slices sit
    /// between guard bytes: destination guards must come back unchanged, and
    /// full-coverage guards around the coverage slice make any coverage byte
    /// read past the span change the output.
    ///
    /// # Safety
    ///
    /// The CPU must have the features `tier` is compiled for.
    #[cfg(target_arch = "x86_64")]
    unsafe fn assert_tier_matches_scalar(
        lengths: impl IntoIterator<Item = usize>,
        seed: u32,
        tier: Tier,
    ) {
        const GUARD: usize = 64;
        for n in lengths {
            for trial in 0..16u32 {
                let seed = seed ^ (u32::try_from(n).unwrap() << 8) ^ trial;
                let mut shape = vec![0xFFu8; GUARD + n + GUARD];
                shape[GUARD..GUARD + n].copy_from_slice(&pseudorandom(n, seed));
                let shape = &shape[GUARD..GUARD + n];
                let initial = pseudorandom(n * 3, seed.rotate_left(11));
                let params = pseudorandom(4, seed.rotate_left(23));
                let src = [params[0], params[1], params[2]];
                let a_input = params[3];

                let mut want = initial.clone();
                composite_scalar(&mut want, src, a_input, shape);

                let mut buf = vec![0xA5u8; GUARD + n * 3 + GUARD];
                buf[GUARD..GUARD + n * 3].copy_from_slice(&initial);
                // SAFETY: the caller guarantees the tier's features.
                unsafe { tier(&mut buf[GUARD..GUARD + n * 3], src, a_input, shape) };

                assert_eq!(
                    &buf[GUARD..GUARD + n * 3],
                    want.as_slice(),
                    "n={n} trial={trial} src={src:?} a_input={a_input}"
                );
                assert!(
                    buf[..GUARD]
                        .iter()
                        .chain(&buf[GUARD + n * 3..])
                        .all(|&b| b == 0xA5),
                    "n={n} trial={trial}: bytes outside the span were written"
                );
            }
        }
    }

    /// The AVX-512 tier for every span length 0..=64: whole blocks, masked
    /// remainders of every size and the one-pixel scalar remainder.
    /// Skipped where the CPU lacks AVX-512 BW+VL.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_tier_matches_scalar_at_every_length() {
        if !is_x86_feature_detected!("avx512bw") || !is_x86_feature_detected!("avx512vl") {
            return;
        }
        // SAFETY: avx512bw + avx512vl confirmed present above.
        unsafe { assert_tier_matches_scalar(0..=64, 0x9e37_79b9, composite_avx512) };
    }

    /// The AVX2 tier for every span length 0..=100 and a few long ones: the
    /// scalar fallback below one block, whole 32-pixel blocks and every
    /// overlapping remainder.  Skipped where the CPU lacks AVX2.
    #[cfg(all(target_arch = "x86_64", feature = "simd-avx2"))]
    #[test]
    fn avx2_tier_matches_scalar_at_every_length() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        // SAFETY: avx2 confirmed present above.
        unsafe {
            assert_tier_matches_scalar(
                (0..=100).chain([1000, 1001, 1023]),
                0x7f4a_7c15,
                composite_avx2,
            );
        }
    }

    /// Every (coverage, destination) byte pair, in every channel, through
    /// the AVX2 tier against the scalar tier, over a spread of opacities
    /// and source colours.  Skipped where the CPU lacks AVX2.
    #[cfg(all(target_arch = "x86_64", feature = "simd-avx2"))]
    #[test]
    fn avx2_tier_matches_scalar_on_every_coverage_destination_pair() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the high byte of i < 65536"
        )]
        let shape: Vec<u8> = (0..65_536u32).map(|i| (i >> 8) as u8).collect();
        #[expect(clippy::cast_possible_truncation, reason = "the low byte is wanted")]
        let initial: Vec<u8> = (0..65_536u32)
            .flat_map(|i| {
                let d = i as u8;
                [d, d.wrapping_mul(7), !d]
            })
            .collect();
        for a_input in (0..=255u8).step_by(5).chain([1, 254]) {
            for s in (0..=255u8).step_by(15) {
                let src = [s, 255 - s, s.wrapping_mul(3)];
                let mut want = initial.clone();
                composite_scalar(&mut want, src, a_input, &shape);
                let mut got = initial.clone();
                // SAFETY: avx2 confirmed present above.
                unsafe { composite_avx2(&mut got, src, a_input, &shape) };
                assert!(got == want, "a_input={a_input} src={src:?}");
            }
        }
    }

    /// Timing of the scalar, AVX2 and AVX-512 tiers on spans of 4 to 1 920
    /// pixels.  Always passes; run with `--ignored --nocapture` in a release
    /// build to see the times.
    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore = "manual perf smoke test; run with `--ignored --nocapture` for timing"]
    fn composite_tier_timing() {
        const TOTAL_PX: usize = 200_000_000;
        for n in [4usize, 13, 40, 200, 1920] {
            let shape = pseudorandom(n, 1);
            let mut dst = pseudorandom(n * 3, 2);
            let iters = TOTAL_PX / n;
            let mut run = |name: &str, f: &dyn Fn(&mut [u8], &[u8])| {
                f(&mut dst, &shape);
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    f(std::hint::black_box(&mut dst), std::hint::black_box(&shape));
                }
                #[expect(clippy::cast_precision_loss, reason = "timing display only")]
                let ns_px = t0.elapsed().as_secs_f64() * 1e9 / (iters * n) as f64;
                println!("n={n} {name}: {ns_px:.3} ns/px");
            };
            run("scalar", &|d, s| composite_scalar(d, [7, 250, 128], 200, s));
            #[cfg(feature = "simd-avx2")]
            if is_x86_feature_detected!("avx2") {
                // SAFETY: avx2 confirmed present.
                run("avx2", &|d, s| unsafe {
                    composite_avx2(d, [7, 250, 128], 200, s);
                });
            }
            if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vl") {
                // SAFETY: both features confirmed present.
                run("avx512", &|d, s| unsafe {
                    composite_avx512(d, [7, 250, 128], 200, s);
                });
            }
        }
    }

    #[test]
    fn opaque_empty_is_noop() {
        let mut dst: Vec<u8> = vec![];
        composite_aa_rgb8_opaque(&mut dst, [1, 2, 3], 255, &[]);
    }
}
