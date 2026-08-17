// Phase 5: zigzag inverse + dequant + 8×8 IDCT (LLM fixed-point) + JFIF YCbCr→RGB.
//
// The 1-D IDCT is the integer Loeffler/Ligtenberg/Moschytz (1989) 8-point
// structure with 13-bit fixed-point constants and 64-bit intermediates.
// YCbCr→RGB uses T.871/JFIF full-range equations (round-half-away-from-zero).
//
// Dispatch: one block (8 × 8 × 3 threads) per 8×8 JPEG block.
// threadIdx.x = col (0..7), threadIdx.y = row (0..7), threadIdx.z = component.
// blockIdx.x = block column, blockIdx.y = block row.
//
// Mirror of idct_color.slang — keep the two in sync.

#include <stdint.h>

// ── Zigzag → natural index table ─────────────────────────────────────────────
__constant__ int zigzag_to_natural[64] = {
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
};

// ── LLM fixed-point constants (scaled by 2^13) ───────────────────────────────
// FIX_a_bbbbbbbbb = round(a.bbbbbbbbb * 2^13), a.bbbbbbbbb = cos-derived
// factors of the Loeffler/Ligtenberg/Moschytz 8-point IDCT.
#define CONST_BITS 13
#define PASS1_BITS 2
#define FIX_0_298631336  2446
#define FIX_0_390180644  3196
#define FIX_0_541196100  4433
#define FIX_0_765366865  6270
#define FIX_0_899976223  7373
#define FIX_1_175875602  9633
#define FIX_1_501321110 12299
#define FIX_1_847759065 15137
#define FIX_1_961570560 16069
#define FIX_2_053119869 16819
#define FIX_2_562915447 20995
#define FIX_3_072711026 25172

// Round-half-up right shift of a 64-bit intermediate, narrowed to int32.
// 64-bit: a 13-bit constant times a dequantised coefficient (up to
// 2047 * 255 = 522 985 for DC) reaches ~1.3e10, far past i32::MAX.
__device__ __forceinline__ int descale(long long x, int s) {
    return (int)((x + (1LL << (s - 1))) >> s);
}

// One 1-D 8-point LLM IDCT on v[0..8] in place, descaled by `shift`.
// Row pass: shift = CONST_BITS - PASS1_BITS, leaving outputs scaled up by
// 2^PASS1_BITS. Column pass: shift = CONST_BITS + PASS1_BITS + 3, which
// removes the pass-1 scale and the 8× gain of the two passes, so a
// DC-only block of dequantised value D yields D/8 in every slot.
__device__ void idct_1d(int v[8], int shift) {
    // Even part: indices 0, 4, 2, 6.
    long long z2 = v[2];
    long long z3 = v[6];
    long long z1 = (z2 + z3) * FIX_0_541196100;
    long long tmp2 = z1 - z3 * FIX_1_847759065;
    long long tmp3 = z1 + z2 * FIX_0_765366865;
    long long tmp0 = (long long)(v[0] + v[4]) << CONST_BITS;
    long long tmp1 = (long long)(v[0] - v[4]) << CONST_BITS;
    long long t10 = tmp0 + tmp3; long long t13 = tmp0 - tmp3;
    long long t11 = tmp1 + tmp2; long long t12 = tmp1 - tmp2;

    // Odd part: indices 7, 5, 3, 1 feed the four rotators.
    long long w0 = v[7];
    long long w1 = v[5];
    long long w2 = v[3];
    long long w3 = v[1];
    long long za = w0 + w3;
    long long zb = w1 + w2;
    long long zc = w0 + w2;
    long long zd = w1 + w3;
    long long z5 = (zc + zd) * FIX_1_175875602;
    w0 *= FIX_0_298631336;
    w1 *= FIX_2_053119869;
    w2 *= FIX_3_072711026;
    w3 *= FIX_1_501321110;
    za *= -FIX_0_899976223;
    zb *= -FIX_2_562915447;
    zc = z5 - zc * FIX_1_961570560;
    zd = z5 - zd * FIX_0_390180644;
    w0 += za + zc;
    w1 += zb + zd;
    w2 += zb + zc;
    w3 += za + zd;

    v[0] = descale(t10 + w3, shift);
    v[7] = descale(t10 - w3, shift);
    v[1] = descale(t11 + w2, shift);
    v[6] = descale(t11 - w2, shift);
    v[2] = descale(t12 + w1, shift);
    v[5] = descale(t12 - w1, shift);
    v[3] = descale(t13 + w0, shift);
    v[4] = descale(t13 - w0, shift);
}

__device__ __forceinline__ int clamp_byte(int x) {
    return max(0, min(255, x));
}

// T.871 full-range YCbCr → RGB, 14-bit fixed point.
// round(c * 2^14): 1.402 → 22970, 0.714136 → 11700, 0.344136 → 5638,
// 1.772 → 29032. The +8192 bias makes the >> 14 round half up; the
// result is clamped to [0, 255], so half-up equals half-away-from-zero.
__device__ __forceinline__ void ycbcr_to_rgb(int Y, int Cb, int Cr,
                                              int *r, int *g, int *b) {
    int cb = Cb - 128; int cr = Cr - 128;
    int y0 = (Y << 14) + 8192;
    *r = clamp_byte((y0 + 22970 * cr) >> 14);
    *g = clamp_byte((y0 - 11700 * cr - 5638 * cb) >> 14);
    *b = clamp_byte((y0 + 29032 * cb) >> 14);
}

__device__ __forceinline__ uint32_t pack_rgba8(int r, int g, int b) {
    return (uint32_t)r | ((uint32_t)g << 8) | ((uint32_t)b << 16) | 0xFF000000u;
}

// Per-block groupshared scratch: scratch[component][row][col].
// Each block has 8×8×3 = 192 threads and 3×8×8 = 192 ints.
__shared__ int scratch[3][8][8];

extern "C" __global__ void idct_dequant_colour(
    const int * __restrict__ coefficients,   // zigzag-order DCT coefficients
    const int * __restrict__ qtables,        // quantisation tables, zigzag order (verbatim DQT)
    const int * __restrict__ dc_values,      // absolute DC per block
          uint32_t * __restrict__ pixels_rgba, // RGBA8 output, row-major
    uint32_t width,
    uint32_t height,
    uint32_t num_components,
    uint32_t blocks_wide,
    uint32_t blocks_high,
    uint32_t num_qtables
) {
    const uint32_t col  = threadIdx.x;   // 0..7 within block
    const uint32_t row  = threadIdx.y;   // 0..7 within block
    const uint32_t comp = threadIdx.z;   // component index

    if (comp >= num_components) return;

    const uint32_t bx = blockIdx.x;  // block column
    const uint32_t by = blockIdx.y;  // block row

    if (bx >= blocks_wide || by >= blocks_high) return;

    // Flat block index within this component's grid.
    const uint32_t block_idx = comp * blocks_wide * blocks_high + by * blocks_wide + bx;

    // QT selector: 0 for luma, 1 (clamped) for chroma.
    uint32_t qt_sel = (comp == 0u) ? 0u : 1u;
    if (qt_sel >= num_qtables) qt_sel = num_qtables - 1u;
    const uint32_t qt_base   = qt_sel * 64u;
    const uint32_t coef_base = block_idx * 64u;

    // Step 1: dequantise + inverse-zigzag into shared scratch.
    // Coefficients and qtables are both in zigzag order, so the quantiser
    // for zigzag slot zz_pos is at the same offset — no permutation.
    const uint32_t zz_pos  = row * 8u + col;
    int coef = coefficients[coef_base + zz_pos];
    // DC override: use pre-resolved absolute DC from the host.
    if (zz_pos == 0u) coef = dc_values[block_idx];
    const int qval     = qtables[qt_base + zz_pos];
    const int dequanted = coef * qval;

    const uint32_t nat_pos = (uint32_t)zigzag_to_natural[zz_pos];
    scratch[comp][nat_pos / 8u][nat_pos & 7u] = dequanted;

    __syncthreads();

    // Step 2: row IDCT. Outputs stay scaled up by 2^PASS1_BITS.
    {
        int v[8];
        for (int k = 0; k < 8; k++) v[k] = scratch[comp][row][k];
        idct_1d(v, CONST_BITS - PASS1_BITS);
        for (int k = 0; k < 8; k++) scratch[comp][row][k] = v[k];
    }

    __syncthreads();

    // Step 3: column IDCT + level shift. The pass-2 descale removes the
    // pass-1 scale and the 8x two-pass gain.
    {
        int v[8];
        for (int k = 0; k < 8; k++) v[k] = scratch[comp][k][col];
        idct_1d(v, CONST_BITS + PASS1_BITS + 3);
        for (int k = 0; k < 8; k++)
            scratch[comp][k][col] = clamp_byte(v[k] + 128);
    }

    __syncthreads();

    // Step 4: colour conversion + output (comp 0 only).
    if (comp != 0u) return;

    const uint32_t px = bx * 8u + col;
    const uint32_t py = by * 8u + row;
    if (px >= width || py >= height) return;

    uint32_t rgba;
    if (num_components == 1u) {
        int luma = scratch[0][row][col];
        rgba = pack_rgba8(luma, luma, luma);
    } else {
        int Y = scratch[0][row][col];
        int Cb = scratch[1][row][col];
        int Cr = scratch[2][row][col];
        int r, g, b;
        ycbcr_to_rgb(Y, Cb, Cr, &r, &g, &b);
        rgba = pack_rgba8(r, g, b);
    }
    pixels_rgba[py * width + px] = rgba;
}
