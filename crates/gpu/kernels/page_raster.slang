// Page raster from a command list: one 16 x 16 workgroup per tile walks the
// tile's commands in order, keeps each pixel in a register and writes the
// tile to the page once.
//
// Inputs, prepared on the host:
//   cmds          the page's commands, in paint order.
//   intersects    every command's scanline intersection table, as int
//                 triples [x0, x1, count] in AA coordinates.
//   row_starts    absolute offsets into the triples; a table of n rows owns
//                 n + 1 consecutive entries starting at its row_base.
//   tile_offsets  n_tiles + 1 offsets into tile_cmds: tile t's commands are
//                 tile_cmds[tile_offsets[t] .. tile_offsets[t + 1]].
//   tile_cmds     command indices per tile, in command order.
// Output:
//   page          RGB8, row-major, width * 3 bytes per row.
//
// A command paints the pixels of its inclusive bounds (x0, y0) - (x1, y1),
// which lie on the page and inside the command's clip rectangle.  A
// pixel's coverage is its count of covered AA sub-samples: the popcount of
// its 4 x 4 cell in the masks of the table's rows.  The mask of an AA row is
// the CPU scanner's span walk (ScanIterator: a span starts at an
// intersection and absorbs the following ones while they overlap it or the
// winding count is inside; a row ending inside reaches the page's right
// edge), clamped to [0, aa_width).  The count maps through AA_GAMMA to a
// shape byte and composites over the opaque pixel as the CPU's
// composite_aa_rgb8_opaque does:
//   a = div255(a_input * shape),
//   c = div255((255 - a) * c_dst + a * c_src).
// A count of 0 leaves the pixel unchanged, as the CPU's composite does for a
// zero shape byte.
//
// Each pass takes four commands of the tile: thread t walks AA row t % 64
// of the tile for command t / 64 once, over the tile's 64 AA columns, and
// stores the row's 64-bit mask in workgroup memory; after a barrier every
// pixel thread composites the four commands in order, its count being the
// popcount of its 4-bit nibble in its four AA rows.

static const int AA_SIZE = 4;
static const uint TILE = 16;
static const uint TILE_AA = TILE * 4;
static const uint BATCH = 4;

static const uint AA_GAMMA[17] = {
    0, 4, 11, 21, 32, 45, 59, 74, 90, 108, 126, 145, 166, 187, 209, 231, 255
};

struct Cmd {
    uint kind;      // reserved; 0 = coverage table
    uint eo;        // 1 = even-odd, 0 = non-zero winding
    uint rgb;       // r | g << 8 | b << 16
    uint a_input;
    int table_y0;   // AA row of the table's row 0
    uint n_rows;
    uint row_base;  // index of the table's first entry in row_starts
    uint pad;
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
    uint aa_width;
    uint paper;     // r | g << 8 | b << 16
};

[[vk::push_constant]] ConstantBuffer<Params> pc;

[[vk::binding(0, 0)]] StructuredBuffer<Cmd> cmds;
[[vk::binding(1, 0)]] StructuredBuffer<int> intersects;
[[vk::binding(2, 0)]] StructuredBuffer<uint> row_starts;
[[vk::binding(3, 0)]] StructuredBuffer<uint> tile_offsets;
[[vk::binding(4, 0)]] StructuredBuffer<uint> tile_cmds;
[[vk::binding(5, 0)]] RWStructuredBuffer<uint8_t> page;

groupshared uint2 row_masks[BATCH][TILE_AA];
groupshared Cmd batch_cmds[BATCH];

uint div255(uint x)
{
    uint t = x + 0x80;
    return min((t + (t >> 8)) >> 8, 255u);
}

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
void page_raster(uint3 lid : SV_GroupThreadID, uint3 wid : SV_GroupID)
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
    uint r = pc.paper & 0xFF;
    uint g = (pc.paper >> 8) & 0xFF;
    uint b = (pc.paper >> 16) & 0xFF;
    for (uint k0 = begin; k0 < end; k0 += BATCH) {
        uint k = k0 + slot;
        if (k < end) {
            Cmd c = cmds[tile_cmds[k]];
            if (q == 0) batch_cmds[slot] = c;
            uint2 m = uint2(0, 0);
            int row = ay - c.table_y0;
            if (row >= 0 && uint(row) < c.n_rows) {
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
            r = div255(inv * r + a * (c.rgb & 0xFF));
            g = div255(inv * g + a * ((c.rgb >> 8) & 0xFF));
            b = div255(inv * b + a * ((c.rgb >> 16) & 0xFF));
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
