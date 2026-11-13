// Page raster from a command list: tile binning and per-tile fine raster
// with in-order compositing.
//
// Four entry points, run in order in one submission:
//   bin_count   one thread per tile: the number of commands whose pixel
//               bounds touch the tile.
//   bin_scan    one workgroup: exclusive prefix sum of the counts into
//               tile_offsets (n_tiles + 1 entries).
//   bin_scatter one thread per tile: the indices of those commands, in
//               command order, at tile_offsets[tile].
//   fine_raster one 16 x 16 workgroup per tile, one thread per pixel: walks
//               the tile's commands in order, keeps the pixel in registers
//               and writes it to the page once.
// fine_raster_rows (alternative to fine_raster) the same output with each
//               scan-table row walked once per tile instead of once per
//               pixel; see its comment.
//
// Command kinds:
//   KIND_SIMPLE  every pixel of the bounds takes the colour (opaque solid
//                source, Normal blend, no shape: the CPU pipe's `simple`
//                variant with an identity transfer).
//   KIND_AA      scanline-AA fill: the pixel's sub-sample count is the
//                aa_fill kernel's walk of the command's intersection table,
//                mapped through AA_GAMMA and composited over an opaque
//                destination as the CPU's `composite_aa_rgb8_opaque`:
//                a = div255(a_input * shape),
//                c = div255((255 - a) * c_dst + a * c_src).
//                A count of 0 leaves the pixel unchanged, as the CPU's
//                composite does for a zero shape byte.

static const int AA_SIZE = 4;
static const uint TILE = 16;
static const uint KIND_SIMPLE = 0;
static const uint KIND_AA = 1;
static const uint SCAN_THREADS = 256;

static const uint AA_GAMMA[17] = {
    0, 4, 11, 21, 32, 45, 59, 74, 90, 108, 126, 145, 166, 187, 209, 231, 255
};

struct Cmd {
    uint kind;
    uint eo;
    uint rgb;       // r | g << 8 | b << 16
    uint a_input;
    int table_y0;   // AA row of the table's row 0
    uint n_rows;
    uint row_base;  // index of the table's first entry in row_starts
    uint pad;
    int x0;         // pixel bounds, inclusive, within the page
    int y0;
    int x1;
    int y1;
};

struct Params {
    uint n_cmds;
    uint width;
    uint height;
    uint tiles_x;
    uint tiles_y;
    uint aa_width;
};

[[vk::push_constant]] ConstantBuffer<Params> pc;

[[vk::binding(0, 0)]] StructuredBuffer<Cmd> cmds;
// Intersections as int triples [x0, x1, count].
[[vk::binding(1, 0)]] StructuredBuffer<int> intersects;
// Absolute offsets into `intersects`; a table of n rows owns n + 1 entries.
[[vk::binding(2, 0)]] StructuredBuffer<uint> row_starts;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> tile_counts;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> tile_offsets;
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> tile_cmds;
// The page, RGB8, row-major, width * 3 bytes per row.
[[vk::binding(6, 0)]] RWStructuredBuffer<uint8_t> page;

bool touches(Cmd c, uint tx, uint ty)
{
    int tx0 = int(tx * TILE);
    int ty0 = int(ty * TILE);
    int tx1 = tx0 + int(TILE) - 1;
    int ty1 = ty0 + int(TILE) - 1;
    return c.x0 <= tx1 && c.x1 >= tx0 && c.y0 <= ty1 && c.y1 >= ty0;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void bin_count(uint3 tid : SV_DispatchThreadID)
{
    uint n_tiles = pc.tiles_x * pc.tiles_y;
    uint t = tid.x;
    if (t >= n_tiles) return;
    uint tx = t % pc.tiles_x;
    uint ty = t / pc.tiles_x;
    uint n = 0;
    for (uint i = 0; i < pc.n_cmds; i++) {
        if (touches(cmds[i], tx, ty)) n++;
    }
    tile_counts[t] = n;
}

groupshared uint scan_sums[SCAN_THREADS];

[shader("compute")]
[numthreads(256, 1, 1)]
void bin_scan(uint3 lid : SV_GroupThreadID)
{
    uint n_tiles = pc.tiles_x * pc.tiles_y;
    uint chunk = (n_tiles + SCAN_THREADS - 1) / SCAN_THREADS;
    uint begin = min(lid.x * chunk, n_tiles);
    uint end = min(begin + chunk, n_tiles);

    uint sum = 0;
    for (uint i = begin; i < end; i++) sum += tile_counts[i];
    scan_sums[lid.x] = sum;
    GroupMemoryBarrierWithGroupSync();

    // Inclusive Hillis-Steele scan of the per-thread sums.
    for (uint step = 1; step < SCAN_THREADS; step <<= 1) {
        uint v = lid.x >= step ? scan_sums[lid.x - step] : 0;
        GroupMemoryBarrierWithGroupSync();
        scan_sums[lid.x] += v;
        GroupMemoryBarrierWithGroupSync();
    }

    uint run = scan_sums[lid.x] - sum;
    for (uint i = begin; i < end; i++) {
        tile_offsets[i] = run;
        run += tile_counts[i];
    }
    if (lid.x == SCAN_THREADS - 1) tile_offsets[n_tiles] = scan_sums[lid.x];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void bin_scatter(uint3 tid : SV_DispatchThreadID)
{
    uint n_tiles = pc.tiles_x * pc.tiles_y;
    uint t = tid.x;
    if (t >= n_tiles) return;
    uint tx = t % pc.tiles_x;
    uint ty = t / pc.tiles_x;
    uint at = tile_offsets[t];
    for (uint i = 0; i < pc.n_cmds; i++) {
        if (touches(cmds[i], tx, ty)) {
            tile_cmds[at] = i;
            at++;
        }
    }
}

// Sub-sample bitmask of one AA row within the four AA columns
// [ax0, ax0 + AA_SIZE); the aa_fill kernel's walk.
uint row_coverage_mask(uint start, uint end, int aa_width, int eo_mask, int ax0)
{
    uint mask = 0;
    int count = 0;
    uint i = start;

    while (i < end) {
        int xx0 = intersects[i * 3];
        if (xx0 >= ax0 + AA_SIZE) break;
        int xx1 = intersects[i * 3 + 1];
        count += intersects[i * 3 + 2];
        i++;
        while (i < end && (intersects[i * 3] <= xx1 || (count & eo_mask) != 0)) {
            xx1 = max(xx1, intersects[i * 3 + 1]);
            count += intersects[i * 3 + 2];
            i++;
        }
        int span_x1 = (i >= end && (count & eo_mask) != 0) ? aa_width : xx1 + 1;

        int bx0 = max(xx0, 0);
        int bx1 = min(span_x1, aa_width);
        if (bx0 < bx1) {
            int lo = max(bx0, ax0);
            int hi = min(bx1, ax0 + AA_SIZE);
            if (lo < hi) {
                mask |= ((1u << uint(hi - lo)) - 1u) << uint(lo - ax0);
            }
        }
    }
    return mask;
}

uint coverage_count(Cmd c, int px, int py)
{
    int eo_mask = (c.eo != 0) ? 1 : ~0;
    int ax0 = px * AA_SIZE;
    int ay0 = py * AA_SIZE;
    uint total = 0;
    for (int yy = 0; yy < AA_SIZE; yy++) {
        int r = ay0 + yy - c.table_y0;
        if (r < 0 || uint(r) >= c.n_rows) continue;
        uint start = row_starts[c.row_base + uint(r)];
        uint end = row_starts[c.row_base + uint(r) + 1];
        total += countbits(row_coverage_mask(start, end, int(pc.aa_width), eo_mask, ax0));
    }
    return total;
}

uint div255(uint x)
{
    uint t = x + 0x80;
    return min((t + (t >> 8)) >> 8, 255u);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void fine_raster(uint3 lid : SV_GroupThreadID, uint3 wid : SV_GroupID)
{
    uint px = wid.x * TILE + lid.x;
    uint py = wid.y * TILE + lid.y;
    if (px >= pc.width || py >= pc.height) return;

    uint tile = wid.y * pc.tiles_x + wid.x;
    uint begin = tile_offsets[tile];
    uint end = tile_offsets[tile + 1];

    int x = int(px);
    int y = int(py);
    uint r = 0;
    uint g = 0;
    uint b = 0;
    for (uint k = begin; k < end; k++) {
        Cmd c = cmds[tile_cmds[k]];
        if (x < c.x0 || x > c.x1 || y < c.y0 || y > c.y1) continue;
        uint sr = c.rgb & 0xFF;
        uint sg = (c.rgb >> 8) & 0xFF;
        uint sb = (c.rgb >> 16) & 0xFF;
        if (c.kind == KIND_SIMPLE) {
            r = sr;
            g = sg;
            b = sb;
            continue;
        }
        uint n = coverage_count(c, x, y);
        if (n == 0) continue;
        uint a = div255(c.a_input * AA_GAMMA[n]);
        uint inv = 255 - a;
        r = div255(inv * r + a * sr);
        g = div255(inv * g + a * sg);
        b = div255(inv * b + a * sb);
    }

    uint o = (py * pc.width + px) * 3;
    page[o] = uint8_t(r);
    page[o + 1] = uint8_t(g);
    page[o + 2] = uint8_t(b);
}

// fine_raster_rows: fine_raster with the scan-table walks shared across the
// tile.  The tile's commands are taken four at a time: thread t walks AA row
// (t % 64) of the tile for command (t / 64) once, over the tile's 64 AA
// columns, and stores the row's 64-bit sub-sample mask in workgroup memory;
// after a barrier every pixel thread composites the four commands in order,
// its count being the popcount of its 4-bit nibble in its four AA rows.
// The walk is row_coverage_mask's over a 64-column window, so every count
// equals fine_raster's.

static const uint BATCH = 4;
static const uint TILE_AA = TILE * 4;

groupshared uint2 row_masks[BATCH][TILE_AA];
groupshared Cmd batch_cmds[BATCH];

// Set bits [lo, hi) of a 64-bit mask held as (low word, high word).
void set_bits(inout uint2 m, int lo, int hi)
{
    int a = clamp(lo, 0, 32);
    int b = clamp(hi, 0, 32);
    if (a < b) m.x |= (b - a == 32 ? 0xFFFFFFFFu : ((1u << uint(b - a)) - 1u)) << uint(a);
    a = clamp(lo - 32, 0, 32);
    b = clamp(hi - 32, 0, 32);
    if (a < b) m.y |= (b - a == 32 ? 0xFFFFFFFFu : ((1u << uint(b - a)) - 1u)) << uint(a);
}

// Sub-sample mask of one AA row over the window [ax0, ax0 + TILE_AA).
uint2 row_window_mask(uint start, uint end, int aa_width, int eo_mask, int ax0)
{
    uint2 mask = uint2(0, 0);
    int count = 0;
    uint i = start;
    int ax1 = ax0 + int(TILE_AA);

    while (i < end) {
        int xx0 = intersects[i * 3];
        if (xx0 >= ax1) break;
        int xx1 = intersects[i * 3 + 1];
        count += intersects[i * 3 + 2];
        i++;
        while (i < end && (intersects[i * 3] <= xx1 || (count & eo_mask) != 0)) {
            xx1 = max(xx1, intersects[i * 3 + 1]);
            count += intersects[i * 3 + 2];
            i++;
        }
        int span_x1 = (i >= end && (count & eo_mask) != 0) ? aa_width : xx1 + 1;

        int bx0 = max(xx0, 0);
        int bx1 = min(span_x1, aa_width);
        if (bx0 < bx1) {
            int lo = max(bx0, ax0);
            int hi = min(bx1, ax1);
            if (lo < hi) set_bits(mask, lo - ax0, hi - ax0);
        }
    }
    return mask;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void fine_raster_rows(uint3 lid : SV_GroupThreadID, uint3 wid : SV_GroupID)
{
    uint px = wid.x * TILE + lid.x;
    uint py = wid.y * TILE + lid.y;
    bool on_page = px < pc.width && py < pc.height;
    uint t = lid.y * TILE + lid.x;
    uint slot = t / TILE_AA;
    uint q = t % TILE_AA;

    uint tile = wid.y * pc.tiles_x + wid.x;
    uint begin = tile_offsets[tile];
    uint end = tile_offsets[tile + 1];
    int ax0 = int(wid.x * TILE_AA);
    int ay = int(wid.y * TILE_AA + q);

    int x = int(px);
    int y = int(py);
    uint r = 0;
    uint g = 0;
    uint b = 0;
    for (uint k0 = begin; k0 < end; k0 += BATCH) {
        uint k = k0 + slot;
        if (k < end) {
            Cmd c = cmds[tile_cmds[k]];
            if (q == 0) batch_cmds[slot] = c;
            uint2 m = uint2(0, 0);
            int row = ay - c.table_y0;
            if (c.kind == KIND_AA && row >= 0 && uint(row) < c.n_rows) {
                int eo_mask = (c.eo != 0) ? 1 : ~0;
                uint start = row_starts[c.row_base + uint(row)];
                uint stop = row_starts[c.row_base + uint(row) + 1];
                m = row_window_mask(start, stop, int(pc.aa_width), eo_mask, ax0);
            }
            row_masks[slot][q] = m;
        }
        GroupMemoryBarrierWithGroupSync();

        uint n_batch = min(BATCH, end - k0);
        for (uint j = 0; j < n_batch; j++) {
            Cmd c = batch_cmds[j];
            if (x < c.x0 || x > c.x1 || y < c.y0 || y > c.y1) continue;
            uint sr = c.rgb & 0xFF;
            uint sg = (c.rgb >> 8) & 0xFF;
            uint sb = (c.rgb >> 16) & 0xFF;
            if (c.kind == KIND_SIMPLE) {
                r = sr;
                g = sg;
                b = sb;
                continue;
            }
            uint shift = (lid.x & 7) * 4;
            uint n = 0;
            for (uint yy = 0; yy < 4; yy++) {
                uint2 m = row_masks[j][lid.y * 4 + yy];
                uint word = lid.x < 8 ? m.x : m.y;
                n += countbits((word >> shift) & 0xFu);
            }
            if (n == 0) continue;
            uint a = div255(c.a_input * AA_GAMMA[n]);
            uint inv = 255 - a;
            r = div255(inv * r + a * sr);
            g = div255(inv * g + a * sg);
            b = div255(inv * b + a * sb);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (on_page) {
        uint o = (py * pc.width + px) * 3;
        page[o] = uint8_t(r);
        page[o + 1] = uint8_t(g);
        page[o + 2] = uint8_t(b);
    }
}
