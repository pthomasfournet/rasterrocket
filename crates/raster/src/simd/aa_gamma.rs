//! Coverage-count → gamma-corrected shape map for AA spans.
//!
//! [`apply_aa_gamma`] replaces every byte of a coverage buffer (values
//! `0..=AA_SIZE²` as produced by `aa_coverage_span`) with `AA_GAMMA[byte]` in
//! place and reports whether any pixel of the span is covered at all.
//!
//! # Tiers (x86-64, most to least preferred)
//!
//! 1. **AVX-512 VBMI** (`avx512bw` + `avx512vbmi`), spans ≥ 64 px: `vpermb`
//!    indexes a 64-entry register copy of the table — 64 px per iteration.
//! 2. **SSSE3**, spans ≥ 16 px: `pshufb` over entries `0..=15`; index 16
//!    aliases entry 0 under `pshufb`'s 4-bit indexing, so lanes equal to 16
//!    are OR-ed with an all-ones compare result to restore `AA_GAMMA[16] =
//!    255` — 16 px per iteration.
//! 3. **Scalar**, branch-free: unconditional table lookup with an
//!    OR-accumulated "any covered" flag.
//!
//! The vector tiers never store through a lane mask.  A span whose length is
//! not a multiple of the vector width ends with one full-width block aligned
//! to the span end; its input is loaded before the loop stores anything, so
//! the overlap re-maps original bytes, never already-mapped ones (the map is
//! not idempotent).
//!
//! On any other architecture the scalar tier is used directly.

use crate::types::AA_GAMMA;

/// `AA_GAMMA` zero-extended to 64 entries for the VBMI byte permute and the
/// SSSE3 byte shuffle (which reads the first 16).
#[cfg(target_arch = "x86_64")]
const AA_GAMMA_64: [u8; 64] = {
    let mut table = [0u8; 64];
    let mut i = 0;
    while i < AA_GAMMA.len() {
        table[i] = AA_GAMMA[i];
        i += 1;
    }
    table
};

// ── Scalar tier ───────────────────────────────────────────────────────────────

/// Scalar tier and the reference the vector tiers are tested against.
///
/// # Panics
///
/// Panics if any byte of `shape` exceeds `AA_SIZE²`.
fn apply_aa_gamma_scalar(shape: &mut [u8]) -> bool {
    let mut any = 0u8;
    for s in shape.iter_mut() {
        let v = AA_GAMMA[usize::from(*s)];
        *s = v;
        any |= v;
    }
    any != 0
}

// ── x86-64 tiers ──────────────────────────────────────────────────────────────

/// SSSE3 tier: 16 px per iteration with an overlapped final block.
///
/// # Safety
///
/// Caller must ensure `ssse3` is available and `shape.len() >= 16`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn apply_aa_gamma_ssse3(shape: &mut [u8]) -> bool {
    use std::arch::x86_64::{
        __m128i, _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8, _mm_or_si128, _mm_set1_epi8,
        _mm_setzero_si128, _mm_shuffle_epi8, _mm_storeu_si128,
    };

    let n = shape.len();
    debug_assert!(
        n >= 16,
        "apply_aa_gamma_ssse3: span of {n} px is narrower than one block"
    );

    // SAFETY: AA_GAMMA_64 is 64 bytes; the load reads its first 16.
    let table = unsafe { _mm_loadu_si128(AA_GAMMA_64.as_ptr().cast()) };
    let sixteen = _mm_set1_epi8(16);
    let map = |v: __m128i| _mm_or_si128(_mm_shuffle_epi8(table, v), _mm_cmpeq_epi8(v, sixteen));

    let p = shape.as_mut_ptr();
    // SAFETY: n ≥ 16, so the final 16 bytes lie inside `shape`.  Loaded
    // before any store so the overlap block maps original bytes.
    let tail = unsafe { _mm_loadu_si128(p.add(n - 16).cast()) };
    let mut acc = _mm_setzero_si128();
    let mut i = 0;
    while i + 16 <= n {
        // SAFETY: i + 16 ≤ n bounds the load and the store.
        unsafe {
            let o = map(_mm_loadu_si128(p.add(i).cast()));
            acc = _mm_or_si128(acc, o);
            _mm_storeu_si128(p.add(i).cast(), o);
        }
        i += 16;
    }
    if i < n {
        let o = map(tail);
        acc = _mm_or_si128(acc, o);
        // SAFETY: same 16-byte range the tail was loaded from.
        unsafe { _mm_storeu_si128(p.add(n - 16).cast(), o) };
    }
    _mm_movemask_epi8(_mm_cmpeq_epi8(acc, _mm_setzero_si128())) != 0xFFFF
}

/// AVX-512 VBMI tier: 64 px per iteration with an overlapped final block.
///
/// # Safety
///
/// Caller must ensure `avx512bw` and `avx512vbmi` are available and
/// `shape.len() >= 64`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vbmi")]
unsafe fn apply_aa_gamma_avx512(shape: &mut [u8]) -> bool {
    use std::arch::x86_64::{
        __mmask64, _mm512_loadu_si512, _mm512_permutexvar_epi8, _mm512_storeu_si512,
        _mm512_test_epi8_mask,
    };

    let n = shape.len();
    debug_assert!(
        n >= 64,
        "apply_aa_gamma_avx512: span of {n} px is narrower than one block"
    );

    // SAFETY: AA_GAMMA_64 is exactly 64 bytes.
    let table = unsafe { _mm512_loadu_si512(AA_GAMMA_64.as_ptr().cast()) };
    let p = shape.as_mut_ptr();
    // SAFETY: n ≥ 64, so the final 64 bytes lie inside `shape`.  Loaded
    // before any store so the overlap block maps original bytes.
    let tail = unsafe { _mm512_loadu_si512(p.add(n - 64).cast()) };
    let mut any: __mmask64 = 0;
    let mut i = 0;
    while i + 64 <= n {
        // SAFETY: i + 64 ≤ n bounds the load and the store.
        unsafe {
            let o = _mm512_permutexvar_epi8(_mm512_loadu_si512(p.add(i).cast()), table);
            any |= _mm512_test_epi8_mask(o, o);
            _mm512_storeu_si512(p.add(i).cast(), o);
        }
        i += 64;
    }
    if i < n {
        let o = _mm512_permutexvar_epi8(tail, table);
        any |= _mm512_test_epi8_mask(o, o);
        // SAFETY: same 64-byte range the tail was loaded from.
        unsafe { _mm512_storeu_si512(p.add(n - 64).cast(), o) };
    }
    any != 0
}

// ── Public dispatch ───────────────────────────────────────────────────────────

/// Map every coverage count in `shape` through `AA_GAMMA` in place.
///
/// Returns `true` if at least one pixel of the span is covered (has a
/// non-zero coverage count), `false` if the whole span is empty.
///
/// # Preconditions
///
/// Every byte of `shape` must be `≤ AA_SIZE²` (16), as produced by
/// `aa_coverage_span`.  The scalar tier panics on a larger value; the vector
/// tiers map it to 0.
pub fn apply_aa_gamma(shape: &mut [u8]) -> bool {
    dispatch_gamma(shape)
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn dispatch_gamma(shape: &mut [u8]) -> bool {
    if shape.len() >= 64
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vbmi")
    {
        // SAFETY: both features confirmed present; span ≥ 64 px.
        unsafe { apply_aa_gamma_avx512(shape) }
    } else if shape.len() >= 16 && is_x86_feature_detected!("ssse3") {
        // SAFETY: ssse3 confirmed present; span ≥ 16 px.
        unsafe { apply_aa_gamma_ssse3(shape) }
    } else {
        apply_aa_gamma_scalar(shape)
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn dispatch_gamma(shape: &mut [u8]) -> bool {
    apply_aa_gamma_scalar(shape)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Pseudorandom coverage counts in `0..=16`.
    fn coverage(n: usize, seed: u32) -> Vec<u8> {
        let mut rng = seed;
        (0..n)
            .map(|_| {
                rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                // (rng >> 24) is 0..=255; % 17 keeps it in the table's domain.
                #[expect(clippy::cast_possible_truncation, reason = "(x >> 24) % 17 < 17")]
                {
                    ((rng >> 24) % 17) as u8
                }
            })
            .collect()
    }

    fn reference(shape: &[u8]) -> (Vec<u8>, bool) {
        let mapped: Vec<u8> = shape.iter().map(|&s| AA_GAMMA[usize::from(s)]).collect();
        let any = shape.iter().any(|&s| s != 0);
        (mapped, any)
    }

    /// Dispatcher-driven parity for every residue of the 16- and 64-wide
    /// blocks, so each tier's overlap tail is exercised on this machine.
    #[test]
    fn dispatch_matches_reference_at_every_length() {
        for n in 0..=200 {
            let shape = coverage(n, 0x1234_5678 ^ n as u32);
            let (want, want_any) = reference(&shape);
            let mut got = shape.clone();
            let got_any = apply_aa_gamma(&mut got);
            assert_eq!(got, want, "mapped bytes differ at n={n}");
            assert_eq!(got_any, want_any, "any-covered flag differs at n={n}");
        }
        for n in [255usize, 256, 257, 1000, 2500] {
            let shape = coverage(n, 0xdead_beef ^ n as u32);
            let (want, want_any) = reference(&shape);
            let mut got = shape;
            assert_eq!(apply_aa_gamma(&mut got), want_any, "n={n}");
            assert_eq!(got, want, "n={n}");
        }
    }

    /// An all-zero span reports "nothing covered" and stays zero; a span
    /// with exactly one covered pixel — in any position, including inside the
    /// overlap tail — reports covered.
    #[test]
    fn any_covered_flag_tracks_single_pixel() {
        for n in [1usize, 15, 16, 17, 63, 64, 65, 130] {
            let mut zero = vec![0u8; n];
            assert!(!apply_aa_gamma(&mut zero), "all-zero span n={n}");
            assert!(zero.iter().all(|&b| b == 0), "all-zero span mutated n={n}");
            for pos in [0, n / 2, n - 1] {
                let mut one = vec![0u8; n];
                one[pos] = 1;
                assert!(apply_aa_gamma(&mut one), "n={n} pos={pos}");
                assert_eq!(one[pos], AA_GAMMA[1], "n={n} pos={pos}");
                assert_eq!(
                    one.iter().filter(|&&b| b != 0).count(),
                    1,
                    "n={n} pos={pos}"
                );
            }
        }
    }

    /// Empty span: no work, nothing covered.
    #[test]
    fn empty_span_is_uncovered() {
        assert!(!apply_aa_gamma(&mut []));
    }

    /// Full coverage (16) is the entry `pshufb` cannot reach on its own; every
    /// tier must map it to 255.
    #[test]
    fn full_coverage_maps_to_255_in_every_tier() {
        for n in [1usize, 16, 40, 64, 200] {
            let mut full = vec![16u8; n];
            assert!(apply_aa_gamma(&mut full));
            assert!(full.iter().all(|&b| b == 255), "n={n}");
        }
    }

    /// Direct calls into each x86 tier, skipped where the CPU lacks it: the
    /// dispatcher only reaches the widest tier the machine supports.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_tiers_match_reference() {
        for n in [
            16usize, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 1000,
        ] {
            let shape = coverage(n, 0x9e37_79b9 ^ n as u32);
            let (want, want_any) = reference(&shape);
            if is_x86_feature_detected!("ssse3") && n >= 16 {
                let mut got = shape.clone();
                // SAFETY: ssse3 confirmed; n ≥ 16.
                let any = unsafe { apply_aa_gamma_ssse3(&mut got) };
                assert_eq!(got, want, "ssse3 n={n}");
                assert_eq!(any, want_any, "ssse3 any n={n}");
            }
            if is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512vbmi")
                && n >= 64
            {
                let mut got = shape.clone();
                // SAFETY: avx512bw + avx512vbmi confirmed; n ≥ 64.
                let any = unsafe { apply_aa_gamma_avx512(&mut got) };
                assert_eq!(got, want, "avx512 n={n}");
                assert_eq!(any, want_any, "avx512 any n={n}");
            }
        }
    }
}
