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
//               coverage-table row searched once per tile row instead of
//               once per pixel; see its comment.
//
// Every command paints its colour at constant alpha a_input with the
// coverage of its table, over the pixels of its inclusive bounds
// (x0, y0) - (x1, y1), which lie on the page.  The table is the CPU
// rasteriser's exact-area coverage: per row, spans [x0, x1, offset] sorted
// by x0, the pixels x0..x1 (inclusive) of a span covered fully when offset
// is -1, else by coverage byte offset + (x - x0).  Pixels in no span are
// uncovered.  A pixel composites over the opaque destination as the CPU's
// composite_aa_rgb8_opaque does:
//   a = div255(a_input * coverage),
//   c = div255((255 - a) * c_dst + a * c_src).
// A coverage of 0 leaves the pixel unchanged, as the CPU's composite does
// for a zero shape byte.  Every pixel starts as the paper colour.

static const uint TILE = 16;
static const uint SCAN_THREADS = 256;

struct Cmd {
    uint kind;      // reserved; 0 = coverage table
    uint reserved;
    uint rgb;       // r | g << 8 | b << 16
    uint a_input;
    int table_y0;   // pixel row of the table's row 0
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
    uint coverage_base;
    uint paper;     // r | g << 8 | b << 16
};

[[vk::push_constant]] ConstantBuffer<Params> pc;

[[vk::binding(0, 0)]] StructuredBuffer<Cmd> cmds;
// Every command's spans as int triples [x0, x1, offset], then from word
// coverage_base on the spans' coverage bytes, four to a word,
// little-endian.
[[vk::binding(1, 0)]] StructuredBuffer<int> table;
// Absolute span offsets; a table of n rows owns n + 1 consecutive entries
// starting at its row_base.
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

uint div255(uint x)
{
    uint t = x + 0x80;
    return min((t + (t >> 8)) >> 8, 255u);
}

// Coverage byte i of the table's coverage words.
uint coverage_byte(uint i)
{
    uint word = uint(table[pc.coverage_base + (i >> 2)]);
    return (word >> ((i & 3u) * 8u)) & 0xFFu;
}

// Index of the first span of [start, end) whose x1 is at least x; end
// when there is none.
uint first_span_from(uint start, uint end, int x)
{
    uint lo = start;
    uint hi = end;
    while (lo < hi) {
        uint mid = (lo + hi) >> 1;
        if (table[mid * 3 + 1] < x) lo = mid + 1; else hi = mid;
    }
    return lo;
}

// Coverage of pixel x in the spans [start, end) of one row.
uint row_coverage(uint start, uint end, int x)
{
    uint i = first_span_from(start, end, x);
    if (i >= end) return 0;
    int x0 = table[i * 3];
    if (x0 > x) return 0;
    int offset = table[i * 3 + 2];
    return offset < 0 ? 255u : coverage_byte(uint(offset + (x - x0)));
}

// c composited over the opaque pixel (r, g, b) at coverage cov.
void composite(Cmd c, uint cov, inout uint r, inout uint g, inout uint b)
{
    uint a = div255(c.a_input * cov);
    uint inv = 255 - a;
    r = div255(inv * r + a * (c.rgb & 0xFF));
    g = div255(inv * g + a * ((c.rgb >> 8) & 0xFF));
    b = div255(inv * b + a * ((c.rgb >> 16) & 0xFF));
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
    uint r = pc.paper & 0xFF;
    uint g = (pc.paper >> 8) & 0xFF;
    uint b = (pc.paper >> 16) & 0xFF;
    for (uint k = begin; k < end; k++) {
        Cmd c = cmds[tile_cmds[k]];
        if (x < c.x0 || x > c.x1 || y < c.y0 || y > c.y1) continue;
        int row = y - c.table_y0;
        if (row < 0 || uint(row) >= c.n_rows) continue;
        uint start = row_starts[c.row_base + uint(row)];
        uint stop = row_starts[c.row_base + uint(row) + 1];
        uint cov = row_coverage(start, stop, x);
        if (cov == 0) continue;
        composite(c, cov, r, g, b);
    }

    uint o = (py * pc.width + px) * 3;
    page[o] = uint8_t(r);
    page[o + 1] = uint8_t(g);
    page[o + 2] = uint8_t(b);
}

// fine_raster_rows: fine_raster with the span search shared across each
// tile row.  The tile's commands are taken BATCH at a time: thread t
// searches tile row (t % TILE) of command (t / TILE) once, for the first
// span reaching the tile's left column, walks the spans that meet the
// tile's TILE columns and stores the row's TILE coverage bytes, four to a
// word, in workgroup memory; after a barrier every pixel thread composites
// the batch's commands in order with its own byte.  Each byte is the one
// row_coverage reads for that pixel, so every pixel equals fine_raster's.

// Commands per batch: the TILE * TILE threads cover TILE rows of each.
static const uint BATCH = TILE;
static const uint ROW_WORDS = TILE / 4;

groupshared uint row_cov[BATCH][TILE][ROW_WORDS];
groupshared Cmd batch_cmds[BATCH];

[shader("compute")]
[numthreads(16, 16, 1)]
void fine_raster_rows(uint3 lid : SV_GroupThreadID, uint3 wid : SV_GroupID)
{
    uint px = wid.x * TILE + lid.x;
    uint py = wid.y * TILE + lid.y;
    bool on_page = px < pc.width && py < pc.height;
    uint t = lid.y * TILE + lid.x;
    uint slot = t / TILE;
    uint q = t % TILE;

    uint tile = wid.y * pc.tiles_x + wid.x;
    uint begin = tile_offsets[tile];
    uint end = tile_offsets[tile + 1];
    int tx0 = int(wid.x * TILE);
    int tx1 = tx0 + int(TILE) - 1;
    int ty = int(wid.y * TILE + q);

    int x = int(px);
    int y = int(py);
    uint r = pc.paper & 0xFF;
    uint g = (pc.paper >> 8) & 0xFF;
    uint b = (pc.paper >> 16) & 0xFF;
    for (uint k0 = begin; k0 < end; k0 += BATCH) {
        uint k = k0 + slot;
        if (k < end) {
            Cmd c = cmds[tile_cmds[k]];
            if (q == 0) batch_cmds[slot] = c;
            uint words[ROW_WORDS];
            for (uint w = 0; w < ROW_WORDS; w++) words[w] = 0;
            int row = ty - c.table_y0;
            if (row >= 0 && uint(row) < c.n_rows) {
                uint start = row_starts[c.row_base + uint(row)];
                uint stop = row_starts[c.row_base + uint(row) + 1];
                for (uint i = first_span_from(start, stop, tx0); i < stop; i++) {
                    int sx0 = table[i * 3];
                    if (sx0 > tx1) break;
                    int sx1 = table[i * 3 + 1];
                    int offset = table[i * 3 + 2];
                    int lo = max(sx0, tx0);
                    int hi = min(sx1, tx1);
                    for (int xx = lo; xx <= hi; xx++) {
                        uint cov = offset < 0 ? 255u : coverage_byte(uint(offset + (xx - sx0)));
                        uint col = uint(xx - tx0);
                        words[col >> 2] |= cov << ((col & 3u) * 8u);
                    }
                }
            }
            for (uint w = 0; w < ROW_WORDS; w++) row_cov[slot][q][w] = words[w];
        }
        GroupMemoryBarrierWithGroupSync();

        uint n_batch = min(BATCH, end - k0);
        for (uint j = 0; j < n_batch; j++) {
            Cmd c = batch_cmds[j];
            if (x < c.x0 || x > c.x1 || y < c.y0 || y > c.y1) continue;
            uint cov = (row_cov[j][lid.y][lid.x >> 2] >> ((lid.x & 3u) * 8u)) & 0xFFu;
            if (cov == 0) continue;
            composite(c, cov, r, g, b);
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
