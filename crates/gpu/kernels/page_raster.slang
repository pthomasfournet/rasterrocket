// Page raster from a command list: one 16 x 16 workgroup per tile walks the
// tile's commands in order, keeps each pixel in a register and writes the
// tile to the page once.
//
// Inputs, prepared on the host:
//   cmds          the page's commands, in paint order.
//   table         every command's coverage spans, as int triples
//                 [x0, x1, offset], then from word coverage_base on the
//                 spans' coverage bytes, four to a word, little-endian.
//                 The pixels x0..x1 (inclusive) of a span are covered
//                 fully when offset is -1, else by byte offset + (x - x0).
//   row_starts    absolute span offsets; a table of n rows owns n + 1
//                 consecutive entries starting at its row_base.
//   tile_offsets  n_tiles + 1 offsets into tile_cmds: tile t's commands are
//                 tile_cmds[tile_offsets[t] .. tile_offsets[t + 1]].
//   tile_cmds     command indices per tile, in command order.
// Output:
//   page          RGB of pc.bits bits per sample, row-major, width * 3
//                 samples per row; a 16-bit sample is two bytes, low
//                 byte first.
//
// A command paints the pixels of its inclusive bounds (x0, y0) - (x1, y1),
// which lie on the page and inside the command's clip rectangle.  A
// pixel's coverage is the CPU rasteriser's, read from the table: the
// span of the pixel's row holding it, found by binary search.  With
// M = 2^bits - 1, the coverage byte becomes the shape cov * M / 255 (exact:
// cov at 8 bits, 257 * cov at 16), and the command composites over the
// opaque pixel as the CPU's AA pipe does at that depth:
//   a = div_max(a_input * shape),
//   c = div_max((M - a) * c_dst + a * c_src),
// a_input, c_src and c_dst being samples of the page's depth.  Every
// operand stays below 2^32: x <= M^2, and div_max adds at most
// 2^(bits - 1) + M to it.  A coverage of 0 leaves the pixel unchanged, as
// the CPU's composite does for a zero shape.

static const uint TILE = 16;

struct Cmd {
    uint kind;      // reserved; 0 = coverage table
    uint reserved;
    uint rg;        // r | g << 16
    uint a_input;
    int table_y0;   // pixel row of the table's row 0
    uint n_rows;
    uint row_base;  // index of the table's first entry in row_starts
    uint b;
    int x0;         // pixel bounds, inclusive
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
    uint paper_rg;  // r | g << 16
    uint paper_b;
    uint bits;      // bits per sample: 8 or 16
};

[[vk::push_constant]] ConstantBuffer<Params> pc;

[[vk::binding(0, 0)]] StructuredBuffer<Cmd> cmds;
[[vk::binding(1, 0)]] StructuredBuffer<int> table;
[[vk::binding(2, 0)]] StructuredBuffer<uint> row_starts;
[[vk::binding(3, 0)]] StructuredBuffer<uint> tile_offsets;
[[vk::binding(4, 0)]] StructuredBuffer<uint> tile_cmds;
[[vk::binding(5, 0)]] RWStructuredBuffer<uint8_t> page;

// x / M rounded to nearest, for x <= M^2, M = 2^bits - 1.
uint div_max(uint x, uint bits)
{
    uint t = x + (1u << (bits - 1));
    return min((t + (t >> bits)) >> bits, (1u << bits) - 1);
}

// Sample v of the page at sample index i, low byte first.
void store(uint i, uint v)
{
    if (pc.bits == 16) {
        page[i * 2] = uint8_t(v & 0xFF);
        page[i * 2 + 1] = uint8_t(v >> 8);
    } else {
        page[i] = uint8_t(v);
    }
}

// Coverage byte i of the table's coverage words.
uint coverage_byte(uint i)
{
    uint word = uint(table[pc.coverage_base + (i >> 2)]);
    return (word >> ((i & 3u) * 8u)) & 0xFFu;
}

// Coverage of pixel x in the spans [start, end) of one row.
uint row_coverage(uint start, uint end, int x)
{
    uint lo = start;
    uint hi = end;
    while (lo < hi) {
        uint mid = (lo + hi) >> 1;
        if (table[mid * 3 + 1] < x) lo = mid + 1; else hi = mid;
    }
    if (lo >= end) return 0;
    int x0 = table[lo * 3];
    if (x0 > x) return 0;
    int offset = table[lo * 3 + 2];
    return offset < 0 ? 255u : coverage_byte(uint(offset + (x - x0)));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void page_raster(uint3 lid : SV_GroupThreadID, uint3 wid : SV_GroupID)
{
    uint px = wid.x * TILE + lid.x;
    uint py = wid.y * TILE + lid.y;
    if (px >= pc.width || py >= pc.height) return;

    uint tile = wid.y * pc.tiles_x + wid.x;
    uint begin = tile_offsets[tile];
    uint end = tile_offsets[tile + 1];
    int x = int(px);
    int y = int(py);
    uint bits = pc.bits;
    uint max_sample = (1u << bits) - 1;
    uint shape_scale = max_sample / 255;
    uint r = pc.paper_rg & 0xFFFF;
    uint g = pc.paper_rg >> 16;
    uint b = pc.paper_b;
    for (uint k = begin; k < end; k++) {
        Cmd c = cmds[tile_cmds[k]];
        if (x < c.x0 || x > c.x1 || y < c.y0 || y > c.y1) continue;
        int row = y - c.table_y0;
        if (row < 0 || uint(row) >= c.n_rows) continue;
        uint start = row_starts[c.row_base + uint(row)];
        uint stop = row_starts[c.row_base + uint(row) + 1];
        uint cov = row_coverage(start, stop, x);
        if (cov == 0) continue;
        uint a = div_max(c.a_input * (cov * shape_scale), bits);
        uint inv = max_sample - a;
        r = div_max(inv * r + a * (c.rg & 0xFFFF), bits);
        g = div_max(inv * g + a * (c.rg >> 16), bits);
        b = div_max(inv * b + a * c.b, bits);
    }

    uint o = (py * pc.width + px) * 3;
    store(o, r);
    store(o + 1, g);
    store(o + 2, b);
}
