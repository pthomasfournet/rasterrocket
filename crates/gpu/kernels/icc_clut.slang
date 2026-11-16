// GPU ICC colour transform kernel — Slang port of icc_clut.cu's grid
// kernel.
//
// Entry point:
//
//   icc_cmyk_clut — the conversion of a tabulated ICC profile through its
//     grid (`color::grid::CmykGrid`): interpolation in the simplex of the
//     pixel's grid cell, five node lookups a pixel, in the 32-bit integer
//     arithmetic of `CmykGrid::convert`, so the output is byte-identical to
//     the host's on every conformant device: 8-bit, or with `wide` the 16-bit
//     outputs of `CmykGrid::convert16`, two bytes each, low first.
//
// The profile-free conversion (the CUDA `icc_cmyk_matrix` entry) always runs
// on the CPU and has no Slang counterpart.
//
// The grid, 32-bit words:
//   [0, 4)              node stride of each input
//   4                   node range: 0 unit, 1 extended
//   GRID_CELLS + 256i+v cell of input i at the sample v: index << 16 | position
//   GRID_NODES + 2j     node j: value 0 | value 1 << 16, then value 2

static const uint GRID_CELLS = 8;
static const uint GRID_NODES = GRID_CELLS + 4 * 256;

// The largest position within a cell and the sum of a pixel's weights.
static const uint GRID_FULL = 65535;

// The values of node `j`, clamped to the last of `n_nodes`: a grid whose
// cells reach beyond its nodes reads the last node instead.
uint3 grid_node(StructuredBuffer<uint> grid, uint j, uint n_nodes)
{
    j = min(j, n_nodes - 1);
    uint lo = grid[GRID_NODES + 2 * j];
    uint hi = grid[GRID_NODES + 2 * j + 1];
    return uint3(lo & 0xffff, lo >> 16, hi & 0xffff);
}

// `NodeRange::output` of the node sums `acc`, 16-bit values.  Extended:
// (2 (acc' - L) + 1) / F with acc' clamped to [L, L + (F^2 - 1) / 2],
// L = (F (F - 1) / 2 + 1) / 2.
uint3 grid_output(uint3 acc, uint extended)
{
    const uint low = (GRID_FULL * (GRID_FULL - 1) / 2 + 1) / 2;
    const uint high = low + (GRID_FULL * GRID_FULL - 1) / 2;
    uint3 v;
    if (extended != 0) {
        v = (2 * (clamp(acc, uint3(low), uint3(high)) - low) + 1) / GRID_FULL;
    } else {
        v = (acc + GRID_FULL / 2) / GRID_FULL;
    }
    return v;
}

// The larger of `a` and `b` into `a`, the smaller into `b`.
void exchange(inout uint a, inout uint b)
{
    uint hi = max(a, b);
    b = min(a, b);
    a = hi;
}

// Each input's cell gives the base node's offset and a sort key, the
// position above the input's index; the keys, sorted in decreasing order
// by the network (0,1)(2,3)(0,2)(1,3)(1,2), visit the simplex's vertices,
// each weighted by the difference of successive positions.
[shader("compute")]
[numthreads(256, 1, 1)]
void icc_cmyk_clut(
    uint3 tid : SV_DispatchThreadID,
    uniform StructuredBuffer<uint8_t> cmyk,
    uniform RWStructuredBuffer<uint8_t> rgb,
    uniform StructuredBuffer<uint> grid,
    uniform uint n_nodes,
    uniform uint n,
    uniform uint wide)
{
    uint i = tid.x;
    if (i >= n) return;

    uint key[4];
    uint base = 0;
    for (uint c = 0; c < 4; ++c) {
        uint cell = grid[GRID_CELLS + 256 * c + uint(cmyk[i * 4 + c])];
        base += (cell >> 16) * grid[c];
        key[c] = (cell & 0xffff) << 3 | c;
    }
    exchange(key[0], key[1]);
    exchange(key[2], key[3]);
    exchange(key[0], key[2]);
    exchange(key[1], key[3]);
    exchange(key[1], key[2]);

    uint3 acc = uint3(0);
    uint vertex = base;
    uint previous = GRID_FULL;
    for (uint j = 0; j < 4; ++j) {
        uint f = key[j] >> 3;
        acc += (previous - f) * grid_node(grid, vertex, n_nodes);
        previous = f;
        vertex += grid[key[j] & 3];
    }
    acc += previous * grid_node(grid, vertex, n_nodes);

    uint3 v = grid_output(acc, grid[4]);
    if (wide != 0) {
        // Channel c at 2 (3i + c), low byte first.
        rgb[i * 6 + 0] = uint8_t(v.x & 0xff);
        rgb[i * 6 + 1] = uint8_t(v.x >> 8);
        rgb[i * 6 + 2] = uint8_t(v.y & 0xff);
        rgb[i * 6 + 3] = uint8_t(v.y >> 8);
        rgb[i * 6 + 4] = uint8_t(v.z & 0xff);
        rgb[i * 6 + 5] = uint8_t(v.z >> 8);
    } else {
        uint3 b = (v * 255 + 32767) / 65535;
        rgb[i * 3 + 0] = uint8_t(b.x);
        rgb[i * 3 + 1] = uint8_t(b.y);
        rgb[i * 3 + 2] = uint8_t(b.z);
    }
}
