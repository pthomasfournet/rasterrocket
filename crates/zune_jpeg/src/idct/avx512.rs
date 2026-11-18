//! The 8×8 integer IDCT on AVX-512 (F and BW), two rows to a register.
//!
//! The arithmetic is that of the AVX2 transform (`avx2.rs`): the same
//! 32-bit products, sums and shifts per sample, regrouped so that each
//! 512-bit operation computes two of its 256-bit ones.  Wrapping integer
//! sums do not depend on their order, so the output is the same.
//!
//! A register holds two rows (or, after a transpose, two columns), `[a | b]`
//! with row `a` in lanes 0–7.  A pass takes the rows as `[0 | 4]`,
//! `[2 | 6]`, `[1 | 5]` and `[3 | 7]` and returns its outputs as `[0 | 1]`,
//! `[7 | 6]`, `[3 | 2]` and `[4 | 5]`; each transpose is eight two-source
//! permutations (`vpermt2d`).

#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#![cfg(feature = "x86")]

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::{CONST_BITS, OUTPUT_SHIFT, PASS1_ROUND, PASS1_SHIFT, SCALE_BITS};

/// The rows of a pass's outputs, two to a register.
const PASS_OUT: [[usize; 2]; 4] = [[0, 1], [7, 6], [3, 2], [4, 5]];

/// The permutation indices of a transpose from registers holding the rows
/// `rows` (two each) to registers holding the columns `cols` (two each).
///
/// Stage one gathers, for each pair of outputs (`g`) from each pair of
/// inputs (`h`), the 32 samples those outputs take from those inputs:
/// lane `8d + 4c + q` is column `cols[2g + d][c]` of the `q`-th row the
/// inputs hold.  Stage two takes each output's samples from the two
/// gathered registers in row order.
const fn transpose_index(rows: [[usize; 2]; 4], cols: [[usize; 2]; 4]) -> [[[u32; 16]; 4]; 2] {
    let mut index = [[[0_u32; 16]; 4]; 2];
    let mut g = 0;
    while g < 2 {
        let mut h = 0;
        while h < 2 {
            let mut d = 0;
            while d < 2 {
                let mut c = 0;
                while c < 2 {
                    let col = cols[2 * g + d][c];
                    let mut q = 0;
                    while q < 4 {
                        // Row `q` of the input pair is half `q % 2` of
                        // register `2h + q / 2`, at 16 for the second.
                        let source = 16 * (q / 2) + 8 * (q % 2) + col;
                        index[0][2 * g + h][8 * d + 4 * c + q] = source as u32;
                        q += 1;
                    }
                    c += 1;
                }
                d += 1;
            }
            h += 1;
        }
        g += 1;
    }
    // Input pair `h` holds `row` as its `q`-th row: lane `8 c + row` of
    // output `out` is lane `8 d + 4 c + q` of gathered register `h`.
    let mut h = 0;
    while h < 2 {
        let mut q = 0;
        while q < 4 {
            let row = rows[2 * h + q / 2][q % 2];
            let mut out = 0;
            while out < 4 {
                let d = out % 2;
                let mut c = 0;
                while c < 2 {
                    index[1][out][8 * c + row] = (16 * h + 8 * d + 4 * c + q) as u32;
                    c += 1;
                }
                out += 1;
            }
            q += 1;
        }
        h += 1;
    }
    index
}

/// Pass outputs to the inputs of the second pass: columns `[0 | 4]`,
/// `[2 | 6]`, `[1 | 5]`, `[3 | 7]`.
static TO_COLUMNS: [[[u32; 16]; 4]; 2] =
    transpose_index(PASS_OUT, [[0, 4], [2, 6], [1, 5], [3, 7]]);
/// Second-pass outputs to output rows `[0 | 1]` … `[6 | 7]`.
static TO_ROWS: [[[u32; 16]; 4]; 2] = transpose_index(PASS_OUT, [[0, 1], [2, 3], [4, 5], [6, 7]]);

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn load_index(index: &[u32; 16]) -> __m512i {
    _mm512_loadu_si512(index.as_ptr().cast())
}

/// Transpose the pass outputs `regs` through `index`.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn transpose(regs: [__m512i; 4], index: &[[[u32; 16]; 4]; 2]) -> [__m512i; 4] {
    let [a, b, c, d] = regs;
    let t = [
        _mm512_permutex2var_epi32(a, load_index(&index[0][0]), b),
        _mm512_permutex2var_epi32(c, load_index(&index[0][1]), d),
        _mm512_permutex2var_epi32(a, load_index(&index[0][2]), b),
        _mm512_permutex2var_epi32(c, load_index(&index[0][3]), d),
    ];
    [
        _mm512_permutex2var_epi32(t[0], load_index(&index[1][0]), t[1]),
        _mm512_permutex2var_epi32(t[0], load_index(&index[1][1]), t[1]),
        _mm512_permutex2var_epi32(t[2], load_index(&index[1][2]), t[3]),
        _mm512_permutex2var_epi32(t[2], load_index(&index[1][3]), t[3]),
    ]
}

/// `[lo | hi]` of two 256-bit constants.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn halves(lo: i32, hi: i32) -> __m512i {
    _mm512_mask_blend_epi32(0xFF00, _mm512_set1_epi32(lo), _mm512_set1_epi32(hi))
}

/// One pass over the rows `[0 | 4]`, `[2 | 6]`, `[1 | 5]`, `[3 | 7]`,
/// rounded by `round` and shifted by `SHIFT`; outputs in [`PASS_OUT`]
/// order.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn pass<const SHIFT: u32>(rows: [__m512i; 4], round: i32) -> [__m512i; 4] {
    let [z04, z26, z15, z37] = rows;
    let swap = |x: __m512i| _mm512_shuffle_i64x2::<0x4E>(x, x);
    let add = |x, y| _mm512_add_epi32(x, y);
    let sub = |x, y| _mm512_sub_epi32(x, y);
    let mul = |x, y| _mm512_mullo_epi32(x, y);
    let round = _mm512_set1_epi32(round);

    // Even part: [t3 | t2] and [t0 | t1], then [x0 | x1] and [x3 | x2].
    let p1 = mul(add(z26, swap(z26)), _mm512_set1_epi32(4433));
    let t32 = add(p1, mul(z26, halves(6270, -15137)));
    let s04 = swap(z04);
    let t01 = _mm512_slli_epi32::<{ CONST_BITS as u32 }>(_mm512_mask_sub_epi32(
        add(z04, s04),
        0xFF00,
        s04,
        z04,
    ));
    let x01 = add(add(t01, t32), round);
    let x32 = add(sub(t01, t32), round);

    // Odd part: [t3 | t1] and [t2 | t0] with their sums of products.
    let t31 = mul(z15, halves(12299, 16819));
    let t20 = mul(z37, halves(25172, 2446));
    let p34 = add(
        _mm512_shuffle_i64x2::<0x44>(z37, z15),
        _mm512_shuffle_i64x2::<0xEE>(z37, z15),
    );
    let p5 = mul(add(p34, swap(p34)), _mm512_set1_epi32(9633));
    let p21 = add(p5, mul(add(z37, swap(z15)), halves(-20995, -7373)));
    let p34 = mul(p34, halves(-16069, -3196));
    let t31 = add(add(t31, swap(p21)), _mm512_shuffle_i64x2::<0xEE>(p34, p34));
    let t20 = add(add(t20, p21), _mm512_shuffle_i64x2::<0x44>(p34, p34));

    let t32 = _mm512_shuffle_i64x2::<0x44>(t31, t20);
    let t01 = _mm512_shuffle_i64x2::<0xEE>(t20, t31);
    [
        _mm512_srai_epi32::<SHIFT>(add(x01, t32)),
        _mm512_srai_epi32::<SHIFT>(sub(x01, t32)),
        _mm512_srai_epi32::<SHIFT>(add(x32, t01)),
        _mm512_srai_epi32::<SHIFT>(sub(x32, t01)),
    ]
}

/// The IDCT of `in_vector` into eight rows of `out_vector`, `stride` apart,
/// as [`super::avx2::idct_avx2`] computes it.
///
/// # Safety
/// The CPU must support AVX-512 F and BW.
///
/// # Panics
/// When `out_vector` holds fewer than `7 * stride + 8` samples.
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn idct_avx512(in_vector: &mut [i32; 64], out_vector: &mut [i16], stride: usize) {
    let out = &mut out_vector[..7 * stride + 8];
    let load = |at: usize| _mm512_loadu_si512(in_vector[at..].as_ptr().cast());
    let (r01, r23, r45, r67) = (load(0), load(16), load(32), load(48));

    // All AC coefficients zero: every sample is the DC term scaled,
    // level-shifted and clamped.
    let ac = _mm512_maskz_mov_epi32(0xFFFE, r01);
    let any = _mm512_or_si512(_mm512_or_si512(ac, r23), _mm512_or_si512(r45, r67));
    if _mm512_test_epi32_mask(any, any) == 0 {
        let coeff = ((in_vector[0] + 4 + 1024) >> 3).clamp(0, 255) as i16;
        for row in out.chunks_mut(stride).take(8) {
            row[..8].fill(coeff);
        }
        return;
    }

    let rows = [
        _mm512_shuffle_i64x2::<0x44>(r01, r45),
        _mm512_shuffle_i64x2::<0x44>(r23, r67),
        _mm512_shuffle_i64x2::<0xEE>(r01, r45),
        _mm512_shuffle_i64x2::<0xEE>(r23, r67),
    ];
    let columns = transpose(
        pass::<{ PASS1_SHIFT as u32 }>(rows, PASS1_ROUND),
        &TO_COLUMNS,
    );
    let rows = transpose(
        pass::<{ OUTPUT_SHIFT as u32 }>(columns, SCALE_BITS),
        &TO_ROWS,
    );

    // Narrowed with signed saturation and clamped to 0–255, two rows at a
    // time.
    let (zero, max) = (_mm256_setzero_si256(), _mm256_set1_epi16(255));
    for (pair, row) in rows.iter().zip((0..8).step_by(2)) {
        let samples = _mm256_min_epi16(_mm256_max_epi16(_mm512_cvtsepi32_epi16(*pair), zero), max);
        let at = row * stride;
        _mm_storeu_si128(
            out[at..at + 8].as_mut_ptr().cast(),
            _mm256_castsi256_si128(samples),
        );
        _mm_storeu_si128(
            out[at + stride..at + stride + 8].as_mut_ptr().cast(),
            _mm256_extracti128_si256::<1>(samples),
        );
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    fn supported() -> bool {
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
    }

    /// Pseudo-random coefficient blocks: dense, sparse, DC-only, at
    /// magnitudes up to the 16-bit range of dequantised coefficients and
    /// past it.
    fn blocks() -> impl Iterator<Item = [i32; 64]> {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        (0..200_000).map(move |i| {
            let range: i64 = [16, 256, 2048, 32_768, 1 << 20][i % 5];
            let density = [64, 20, 4, 1][(i / 5) % 4];
            let mut block = [0_i32; 64];
            for (k, c) in block.iter_mut().enumerate() {
                if k == 0 || next() % 64 < density {
                    *c = (next() as i64 % range) as i32;
                }
            }
            block
        })
    }

    /// The AVX-512 IDCT equals the scalar and AVX2 ones, sample for
    /// sample, at strides that leave samples between the rows, which it
    /// leaves untouched.
    #[test]
    fn matches_scalar_and_avx2() {
        if !supported() {
            return;
        }
        let avx2 = is_x86_feature_detected!("avx2");
        for (n, block) in blocks().enumerate() {
            let stride = [8, 11, 24][n % 3];
            let len = 7 * stride + 8;
            let mut got = vec![0x5a5a_i16; len + 3];
            unsafe { idct_avx512(&mut block.clone(), &mut got, stride) };
            let mut scalar = vec![0x5a5a_i16; len + 3];
            super::super::scalar::idct_int(&mut block.clone(), &mut scalar, stride);
            assert_eq!(got, scalar, "block {n}: {block:?}");
            if avx2 {
                let mut vector = vec![0x5a5a_i16; len + 3];
                unsafe { super::super::avx2::idct_avx2(&mut block.clone(), &mut vector, stride) };
                assert_eq!(vector, scalar, "AVX2, block {n}");
            }
        }
    }

    /// The transposes put every sample of the pass outputs where the next
    /// step reads it.
    #[test]
    fn transposes_are_transposes() {
        if !supported() {
            return;
        }
        // Register `i` holds rows PASS_OUT[i]; sample (row, col) = 8 row + col.
        let regs: [__m512i; 4] = core::array::from_fn(|i| {
            let lanes: [i32; 16] =
                core::array::from_fn(|l| (8 * PASS_OUT[i][l / 8] + l % 8) as i32);
            unsafe { _mm512_loadu_si512(lanes.as_ptr().cast()) }
        });
        for (index, cols) in [
            (&TO_COLUMNS, [[0, 4], [2, 6], [1, 5], [3, 7]]),
            (&TO_ROWS, [[0, 1], [2, 3], [4, 5], [6, 7]]),
        ] {
            let out = unsafe { transpose(regs, index) };
            for (reg, pair) in out.iter().zip(cols) {
                let mut lanes = [0_i32; 16];
                unsafe { _mm512_storeu_si512(lanes.as_mut_ptr().cast(), *reg) };
                let want: [i32; 16] = core::array::from_fn(|l| (8 * (l % 8) + pair[l / 8]) as i32);
                assert_eq!(lanes, want, "columns {pair:?}");
            }
        }
    }
}
