// GPU ICC colour transform kernels.
//
// Two entry points:
//
//   icc_cmyk_matrix — profile-free DeviceCMYK conversion: multilinear
//     interpolation over the sixteen solid / overprint corner colours, in the
//     fixed-point arithmetic of the Rust `color::cmyk` module, so the output
//     is byte-identical to the CPU conversion.
//
//   icc_cmyk_clut — the conversion of a tabulated ICC profile through its
//     grid (`color::grid::CmykGrid`): interpolation in the simplex of the
//     pixel's grid cell, five node lookups a pixel through __ldg(), in the
//     32-bit integer arithmetic of `CmykGrid::convert`, so the output is
//     byte-identical to the host's: 8-bit, or with `wide` the 16-bit
//     outputs of `CmykGrid::convert16`.
//
// Input layout:  `n_pixels` × 4 bytes, interleaved CMYK (PDF convention:
//                0 = no ink, 255 = full ink for DeviceCMYK).
// Output layout: matrix kernel, `n_pixels` × 3 bytes, interleaved RGB.
//                Grid kernel, whole 32-bit words: pixel i is the word
//                r | g << 8 | b << 16, or with `wide` the words
//                2i = r | g << 16 and 2i + 1 = b.
//
// Thread/block layout: 256 threads/block, ceil(n_pixels/256) blocks.

// ── Matrix kernel ─────────────────────────────────────────────────────────────

// Interpolation weight of an ink byte on [0, 256]: 0 -> 0, 255 -> 256.
__device__ __forceinline__ int ink_weight(unsigned int x) {
    return (int)(x + (x >> 7));
}

// 256*a + (b - a)*w: one interpolation step, adding 8 fractional bits.
__device__ __forceinline__ int lerp8(int a, int b, int w) {
    return (a << 8) + (b - a) * w;
}

// Drop 8 fractional bits with round-half-up.
__device__ __forceinline__ int round8(int v) {
    return (v + 128) >> 8;
}

// Convert one CMYK pixel to RGB by multilinear interpolation over the corners.
//
// Parameters:
//   cmyk    : interleaved input,  4 bytes per pixel
//   rgb     : interleaved output, 3 bytes per pixel
//   corners : 16 RGB corner colours (48 bytes), index k*8 + c*4 + m*2 + y
//   n       : total pixel count
//
// The corners are reduced along Y, M, C, K.  Intermediates carry 8 fractional
// bits and are rounded after the M and C stages; the K stage rounds to an
// integer.  All intermediates are convex combinations of corner bytes, hence
// non-negative and below 2^31.
extern "C" __global__ void icc_cmyk_matrix(
    const unsigned char* __restrict__ cmyk,
    unsigned char*       __restrict__ rgb,
    const unsigned char* __restrict__ corners,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    unsigned int ci = i * 4;
    unsigned int ri = i * 3;

    int wc = ink_weight(cmyk[ci]);
    int wm = ink_weight(cmyk[ci + 1]);
    int wy = ink_weight(cmyk[ci + 2]);
    int wk = ink_weight(cmyk[ci + 3]);

    for (int ch = 0; ch < 3; ++ch) {
        // Y stage: j = k*4 + c*2 + m pairs corners 2j (y = 0) and 2j + 1 (y = 1).
        int ys[8];
        for (int j = 0; j < 8; ++j) {
            int lo = (int)__ldg(&corners[(2 * j) * 3 + ch]);
            int hi = (int)__ldg(&corners[(2 * j + 1) * 3 + ch]);
            ys[j] = (lo << 8) + (hi - lo) * wy;
        }
        int m0 = round8(lerp8(ys[0], ys[1], wm));
        int m1 = round8(lerp8(ys[2], ys[3], wm));
        int m2 = round8(lerp8(ys[4], ys[5], wm));
        int m3 = round8(lerp8(ys[6], ys[7], wm));
        int c0 = round8(lerp8(m0, m1, wc));
        int c1 = round8(lerp8(m2, m3, wc));
        rgb[ri + ch] = (unsigned char)((lerp8(c0, c1, wk) + (1 << 15)) >> 16);
    }
}

// ── Grid kernel ───────────────────────────────────────────────────────────────

// The grid, 32-bit words (`color::grid::CmykGrid`):
//   [0, 4)              node stride of each input
//   4                   node range: 0 unit, 1 extended
//   GRID_CELLS + 256i+v cell of input i at the sample v: index << 16 | position
//   GRID_NODES + 2j     node j: value 0 | value 1 << 16, then value 2
#define GRID_CELLS 8u
#define GRID_NODES (GRID_CELLS + 4u * 256u)

// The largest position within a cell and the sum of a pixel's weights.
#define GRID_FULL 65535u

// The values of node `j`, clamped to the last of `n_nodes`: a grid whose
// cells reach beyond its nodes reads the last node instead.
__device__ __forceinline__ void grid_node(
    const unsigned int* __restrict__ grid, unsigned int j, unsigned int n_nodes,
    unsigned int* v0, unsigned int* v1, unsigned int* v2
) {
    j = min(j, n_nodes - 1u);
    unsigned int lo = __ldg(&grid[GRID_NODES + 2u * j]);
    unsigned int hi = __ldg(&grid[GRID_NODES + 2u * j + 1u]);
    *v0 = lo & 0xffffu;
    *v1 = lo >> 16;
    *v2 = hi & 0xffffu;
}

// `NodeRange::output` of the node sum `acc`, a 16-bit value.  Extended:
// (2 (acc' - L) + 1) / F with acc' clamped to [L, L + (F^2 - 1) / 2],
// L = (F (F - 1) / 2 + 1) / 2.
__device__ __forceinline__ unsigned int grid_output(unsigned int acc, unsigned int extended) {
    const unsigned int low = (GRID_FULL * (GRID_FULL - 1u) / 2u + 1u) / 2u;
    const unsigned int high = low + (GRID_FULL * GRID_FULL - 1u) / 2u;
    unsigned int v;
    if (extended) {
        v = (2u * (min(max(acc, low), high) - low) + 1u) / GRID_FULL;
    } else {
        v = (acc + GRID_FULL / 2u) / GRID_FULL;
    }
    return v;
}

// The nearest byte of the 16-bit value `v`.
__device__ __forceinline__ unsigned int grid_byte(unsigned int v) {
    return (v * 255u + 32767u) / 65535u;
}

// Output pixel `i`, the 16-bit values `v0`, `v1`, `v2`: with `wide` the
// words 2i = v0 | v1 << 16 and 2i + 1 = v2; otherwise the word of their
// nearest bytes, b0 | b1 << 8 | b2 << 16.
__device__ __forceinline__ void grid_store(
    unsigned int* __restrict__ rgb, unsigned int i,
    unsigned int v0, unsigned int v1, unsigned int v2, unsigned int wide
) {
    if (wide) {
        rgb[2u * i]      = v0 | v1 << 16;
        rgb[2u * i + 1u] = v2;
    } else {
        rgb[i] = grid_byte(v0) | grid_byte(v1) << 8 | grid_byte(v2) << 16;
    }
}

// The larger of `*a` and `*b` into `*a`, the smaller into `*b`.
__device__ __forceinline__ void exchange(unsigned int* a, unsigned int* b) {
    unsigned int hi = max(*a, *b);
    *b = min(*a, *b);
    *a = hi;
}

// cmyk    : interleaved input,  4 bytes per pixel
// rgb     : output words, one per pixel, two with `wide`
// grid    : the grid's words
// n_nodes : the grid's node count
// n       : total pixel count
// wide    : nonzero for 16-bit outputs
//
// Each input's cell gives the base node's offset and a sort key, the
// position above the input's index; the keys, sorted in decreasing order
// by the network (0,1)(2,3)(0,2)(1,3)(1,2), visit the simplex's vertices,
// each weighted by the difference of successive positions.
extern "C" __global__ void icc_cmyk_clut(
    const unsigned char* __restrict__ cmyk,
    unsigned int*        __restrict__ rgb,
    const unsigned int*  __restrict__ grid,
    unsigned int n_nodes,
    unsigned int n,
    unsigned int wide
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    unsigned int key[4];
    unsigned int base = 0u;
    for (unsigned int c = 0u; c < 4u; ++c) {
        unsigned int cell = __ldg(&grid[GRID_CELLS + 256u * c + (unsigned int)__ldg(&cmyk[i * 4u + c])]);
        base += (cell >> 16) * __ldg(&grid[c]);
        key[c] = (cell & 0xffffu) << 3 | c;
    }
    exchange(&key[0], &key[1]);
    exchange(&key[2], &key[3]);
    exchange(&key[0], &key[2]);
    exchange(&key[1], &key[3]);
    exchange(&key[1], &key[2]);

    unsigned int acc0 = 0u, acc1 = 0u, acc2 = 0u;
    unsigned int vertex = base;
    unsigned int previous = GRID_FULL;
    unsigned int v0, v1, v2;
    for (int j = 0; j < 4; ++j) {
        unsigned int f = key[j] >> 3;
        unsigned int weight = previous - f;
        grid_node(grid, vertex, n_nodes, &v0, &v1, &v2);
        acc0 += weight * v0;
        acc1 += weight * v1;
        acc2 += weight * v2;
        previous = f;
        vertex += __ldg(&grid[key[j] & 3u]);
    }
    grid_node(grid, vertex, n_nodes, &v0, &v1, &v2);
    acc0 += previous * v0;
    acc1 += previous * v1;
    acc2 += previous * v2;

    unsigned int extended = __ldg(&grid[4]);
    grid_store(rgb, i,
               grid_output(acc0, extended), grid_output(acc1, extended),
               grid_output(acc2, extended), wide);
}
