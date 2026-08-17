// Phase 5: zigzag inverse + dequant + 8×8 IDCT (LLM fixed-point) + JFIF YCbCr→RGB.
//
// The 1-D IDCT is the integer Loeffler/Ligtenberg/Moschytz (1989) 8-point
// structure with 13-bit fixed-point constants and 64-bit intermediates.
// YCbCr→RGB uses T.871/JFIF full-range equations (round-half-away-from-zero).
//
// Dispatch: one workgroup (8 threads × 8 threads) per 8×8 JPEG block.
// `gid.z` is the component index (0 = Y/luma, 1 = Cb, 2 = Cr for YCbCr;
// 0 = luma only for grayscale).
// Pixels are written to `pixels_rgba` as packed RGBA8 (one u32 per pixel),
// using 0xFF for the A channel.

[[vk::binding(0, 0)]] StructuredBuffer<int>    coefficients;  // Phase 4 output, zigzag order
[[vk::binding(1, 0)]] StructuredBuffer<int>    qtables;       // 64 × num_qtables, zigzag order (verbatim DQT)
[[vk::binding(2, 0)]] StructuredBuffer<int>    dc_values;     // absolute DC per block, one entry/block
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> pixels_rgba;   // RGBA8 output, row-major

[[vk::push_constant]]
cbuffer PushParams {
    uint width;             // image width in pixels
    uint height;            // image height in pixels
    uint num_components;    // 1 (grayscale) or 3 (YCbCr)
    uint blocks_wide;       // ceil(width / 8)
    uint blocks_high;       // ceil(height / 8)
    uint num_qtables;       // number of quantisation tables (1–4)
};

// Zigzag index → natural (row-major 8×8) order.
// zigzag_to_natural[k] gives the (row * 8 + col) index in natural order
// for zigzag position k.
static const int zigzag_to_natural[64] = {
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
};

// LLM fixed-point constants (scaled by 2^13).
// FIX_a_bbbbbbbbb = round(a.bbbbbbbbb * 2^13), a.bbbbbbbbb = cos-derived
// factors of the Loeffler/Ligtenberg/Moschytz 8-point IDCT.
static const int CONST_BITS = 13;
static const int PASS1_BITS = 2;
static const int FIX_0_298631336 =  2446;
static const int FIX_0_390180644 =  3196;
static const int FIX_0_541196100 =  4433;
static const int FIX_0_765366865 =  6270;
static const int FIX_0_899976223 =  7373;
static const int FIX_1_175875602 =  9633;
static const int FIX_1_501321110 = 12299;
static const int FIX_1_847759065 = 15137;
static const int FIX_1_961570560 = 16069;
static const int FIX_2_053119869 = 16819;
static const int FIX_2_562915447 = 20995;
static const int FIX_3_072711026 = 25172;

// Round-half-up right shift of a 64-bit intermediate, narrowed to int32.
// 64-bit: a 13-bit constant times a dequantised coefficient (up to
// 2047 * 255 = 522 985 for DC) reaches ~1.3e10, far past i32::MAX.
int descale(int64_t x, int s) {
    return int((x + (int64_t(1) << (s - 1))) >> s);
}

// One 1-D 8-point LLM IDCT on v[0..8] in place, descaled by `shift`.
// Row pass: shift = CONST_BITS - PASS1_BITS, leaving outputs scaled up by
// 2^PASS1_BITS. Column pass: shift = CONST_BITS + PASS1_BITS + 3, which
// removes the pass-1 scale and the 8× gain of the two passes, so a
// DC-only block of dequantised value D yields D/8 in every slot.
void idct_1d(inout int v[8], int shift) {
    // Even part: indices 0, 4, 2, 6.
    int64_t z2 = v[2];
    int64_t z3 = v[6];
    int64_t z1 = (z2 + z3) * FIX_0_541196100;
    int64_t tmp2 = z1 - z3 * FIX_1_847759065;
    int64_t tmp3 = z1 + z2 * FIX_0_765366865;
    int64_t tmp0 = int64_t(v[0] + v[4]) << CONST_BITS;
    int64_t tmp1 = int64_t(v[0] - v[4]) << CONST_BITS;
    int64_t t10 = tmp0 + tmp3;
    int64_t t13 = tmp0 - tmp3;
    int64_t t11 = tmp1 + tmp2;
    int64_t t12 = tmp1 - tmp2;

    // Odd part: indices 7, 5, 3, 1 feed the four rotators.
    int64_t w0 = v[7];
    int64_t w1 = v[5];
    int64_t w2 = v[3];
    int64_t w3 = v[1];
    int64_t za = w0 + w3;
    int64_t zb = w1 + w2;
    int64_t zc = w0 + w2;
    int64_t zd = w1 + w3;
    int64_t z5 = (zc + zd) * FIX_1_175875602;
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

// Clamp integer to [0, 255].
int clamp_byte(int x) {
    return clamp(x, 0, 255);
}

// T.871 full-range YCbCr → R, G, B, 14-bit fixed point.
// Coefficients from ITU-T T.871 §7:
//   R = Y + 1.402 * (Cr - 128)
//   G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
//   B = Y + 1.772 * (Cb - 128)
// round(c * 2^14): 1.402 → 22970, 0.714136 → 11700, 0.344136 → 5638,
// 1.772 → 29032. The +8192 bias makes the >> 14 round half up; the
// result is clamped to [0, 255], so half-up equals half-away-from-zero.
int3 ycbcr_to_rgb(int Y, int Cb, int Cr) {
    int cb = Cb - 128;
    int cr = Cr - 128;
    int y0 = (Y << 14) + 8192;
    int r = (y0 + 22970 * cr) >> 14;
    int g = (y0 - 11700 * cr - 5638 * cb) >> 14;
    int b = (y0 + 29032 * cb) >> 14;
    return int3(clamp_byte(r), clamp_byte(g), clamp_byte(b));
}

// Pack R, G, B, A=255 into a little-endian RGBA8 u32.
uint pack_rgba8(int r, int g, int b) {
    return uint(r) | (uint(g) << 8u) | (uint(b) << 16u) | 0xFF000000u;
}

// Groupshared scratch for 3 component planes, each 8×8 pels.
// Layout: scratch[component][row][col].
groupshared int scratch[3][8][8];

// Main entry point: one workgroup = one 8×8 JPEG block.
// tid.x = col (0..7), tid.y = row (0..7), tid.z = component (0..num_components-1).
[shader("compute")][numthreads(8, 8, 3)]
void idct_dequant_colour(
    uint3 tid : SV_DispatchThreadID,
    uint3 lid : SV_GroupThreadID,
    uint3 gid : SV_GroupID
) {
    uint col       = lid.x;    // 0..7 within block
    uint row       = lid.y;    // 0..7 within block
    uint comp      = lid.z;    // component index

    // Only components 0..num_components-1 are active.
    if (comp >= num_components) {
        return;
    }
    // num_qtables == 0 would underflow in `min(qt_sel, num_qtables - 1u)`.
    // The host validates this before dispatch; guard here as defence-in-depth.
    if (num_qtables == 0u) {
        return;
    }

    // Block coordinate in the component plane.
    // For simplicity this kernel assumes 4:4:4 sampling (no subsampling):
    // all components have the same block grid. Subsampling upscaling
    // is deferred to a follow-on kernel or done by duplicating chroma blocks
    // before invoking this kernel.
    uint bx = gid.x;  // block column
    uint by = gid.y;  // block row

    if (bx >= blocks_wide || by >= blocks_high) {
        return;
    }

    // Flat block index within this component's block grid.
    // Widened to uint64_t: for a 3-component 65535×65535 JPEG, block_idx reaches
    // ~201 M which overflows u32; coef_base = block_idx * 64 reaches ~12.9 G.
    // The host rejects images exceeding the 2^26-block limit before dispatch;
    // the 64-bit arithmetic here is defence-in-depth for kernel correctness.
    uint64_t block_idx = uint64_t(comp) * uint64_t(blocks_wide) * uint64_t(blocks_high)
                       + uint64_t(by) * uint64_t(blocks_wide) + uint64_t(bx);

    // --- Step 1: dequantise and inverse zigzag into scratch[comp][row][col] ---
    // The qtable selector is 0 for Y (luma) and 1 for Cb/Cr (chroma) by JFIF convention;
    // clamp to num_qtables - 1 so a grayscale image with one table doesn't OOB.
    uint qt_sel = (comp == 0u) ? 0u : 1u;
    qt_sel = min(qt_sel, num_qtables - 1u);
    uint qt_base = qt_sel * 64u;
    uint64_t coef_base = block_idx * 64u;

    // Load all 64 zigzag coefficients for this thread's block.
    // We need the whole block to run the column IDCT, so each thread
    // loads one coefficient (8 threads × 8 threads = 64 total).
    // Coefficients and qtables are both in zigzag order, so the quantiser
    // for zigzag slot zz_pos is at the same offset — no permutation.
    uint zz_pos = row * 8u + col;
    int  coef   = coefficients[coef_base + zz_pos];
    int  qval   = qtables[qt_base + zz_pos];
    // DC absolute value override: Phase 4 symbols encode DC deltas;
    // `dc_values[block_idx]` is the running absolute DC reconstructed by
    // the host dispatcher. Overwrite slot 0 (zigzag DC position).
    if (zz_pos == 0u) {
        coef = dc_values[block_idx];
    }
    int dequanted = coef * qval;

    // Place into scratch at the natural (row,col) position.
    uint nat_pos = uint(zigzag_to_natural[zz_pos]);
    uint nat_row = nat_pos / 8u;
    uint nat_col = nat_pos & 7u;
    scratch[comp][nat_row][nat_col] = dequanted;

    GroupMemoryBarrierWithGroupSync();

    // --- Step 2: 1-D IDCT on rows (each thread handles one element of one row) ---
    // Thread (row, col) owns row `row` of component `comp`.
    // Outputs stay scaled up by 2^PASS1_BITS.
    {
        int v[8];
        for (uint k = 0u; k < 8u; k++) {
            v[k] = scratch[comp][row][k];
        }
        idct_1d(v, CONST_BITS - PASS1_BITS);
        for (uint k = 0u; k < 8u; k++) {
            scratch[comp][row][k] = v[k];
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // --- Step 3: 1-D IDCT on columns ---
    // Thread (row, col) owns column `col` of component `comp`.
    // The pass-2 descale removes the pass-1 scale and the 8× two-pass gain;
    // then add the JPEG level-shift of 128.
    {
        int v[8];
        for (uint k = 0u; k < 8u; k++) {
            v[k] = scratch[comp][k][col];
        }
        idct_1d(v, CONST_BITS + PASS1_BITS + 3);
        for (uint k = 0u; k < 8u; k++) {
            scratch[comp][k][col] = clamp_byte(v[k] + 128);
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // --- Step 4: colour conversion + output ---
    // Thread (row, col, comp==0) writes the RGBA8 pixel for this (row, col).
    if (comp != 0u) {
        return;
    }

    uint px = bx * 8u + col;
    uint py = by * 8u + row;
    if (px >= width || py >= height) {
        return;
    }

    uint out_idx = py * width + px;
    uint rgba;
    if (num_components == 1u) {
        int luma = scratch[0][row][col];
        rgba = pack_rgba8(luma, luma, luma);
    } else {
        int Y  = scratch[0][row][col];
        int Cb = scratch[1][row][col];
        int Cr = scratch[2][row][col];
        int3 rgb = ycbcr_to_rgb(Y, Cb, Cr);
        rgba = pack_rgba8(rgb.x, rgb.y, rgb.z);
    }
    pixels_rgba[out_idx] = rgba;
}
