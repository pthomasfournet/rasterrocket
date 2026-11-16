// Page raster from a command list: one 16 x 16 workgroup per tile walks the
// tile's items in order, keeps each pixel in a register and writes the
// tile to the page once.
//
// Compiled twice: as `page_raster` with tables and masks only, and with
// PAGE_PATHS defined as `page_raster_paths`, which also rasterises outlines
// and needs 64-bit integers.
//
// Inputs, prepared on the host:
//   cmds          the page's commands, in paint order.
//   table         every table command's spans, as int triples
//                 [x0, x1, offset]; from word coverage_base on, the
//                 spans' and masks' coverage bytes, four to a word,
//                 little-endian;
//                 from segs_base on, every outline's segments, four ints
//                 [x0, y0, x1, y1] each, fixed point (1/256 pixel); from
//                 lists_base on, the segment indices of every outline
//                 tile; from backdrops_base on, the backdrops, 16 ints
//                 each; from images_base on, the image commands'
//                 records and sampling tables (page_image.slang).  The
//                 pixels x0..x1 (inclusive) of a span are
//                 covered fully when offset is -1, else by byte offset +
//                 (x - x0).
//   row_starts    absolute span offsets; a table of n rows owns n + 1
//                 consecutive entries starting at its row_base.
//   tile_offsets  n_tiles + 1 offsets into tile_items: tile t's items are
//                 tile_items[tile_offsets[t] .. tile_offsets[t + 1]].
//   tile_items    four words per item, in paint order within a tile:
//                 command, list, len, backdrop.  For an outline, the low
//                 14 bits of len count the tile's segments: the one
//                 segment of index list when it is 1, else the segment
//                 indices from word lists_base + list on.  With bit 31 of
//                 len set, backdrop is the cover of every row of the tile;
//                 otherwise it indexes the backdrops, NO_BACKDROP standing
//                 for zero.
// Output:
//   page          RGB of pc.bits bits per sample, row-major, width * 3
//                 samples per row; a 16-bit sample is two bytes, low
//                 byte first.
//
// An image command samples an image per pixel and composites the sampled
// colour (page_image.slang); every other command paints one colour with
// a coverage.
//
// A command paints the pixels of its inclusive bounds (x0, y0) - (x1, y1),
// which lie on the page and inside the command's clip rectangle.
//
// A table command's coverage of a pixel is read from its table: the span
// of the pixel's row holding it, found by binary search.
//
// A mask command's coverage of a pixel is a byte of its mask, which the
// command places with its top-left pixel at (left, top): byte
// (y - top) * width + (x - left) of the mask's bytes, which start at byte
// row_base of the coverage bytes.  The command's bounds lie in the mask's
// rectangle.
//
// An outline command's coverage of a pixel is the CPU's exact-area
// rasteriser's over the tile taken as the window, which equals its
// coverage over the command's window there.  The workgroup takes the
// tile's items in batches of up to BATCH; for each outline item of a
// batch that lists segments it clears one cover and one area cell per
// pixel, adds the tile's backdrop (the cover
// of the segments wholly left of the tile, per row) to the left column,
// deposits the cells of every listed segment row by row as the CPU
// rasteriser does (boundary positions by floor division from the
// segment's endpoints), all items' (segment, row) pairs spread over the
// workgroup's threads; then the covers of every row are summed along it,
// and each pixel, item by item in order, takes the sum at its own cell: v = 512 * cover - area, mapped by the fill rule to an
// area and rounded to a byte, then scaled by the command's weight; an
// outline item listing no segment takes its row's backdrop alone.  The
// deposits are integer additions, so their order does not change a sum;
// the host bounds the coordinates and the segments per tile so that every
// product fits 64 bits and every cell sum 32.
//
// With M = 2^bits - 1, the coverage byte becomes the shape cov * M / 255
// (exact: cov at 8 bits, 257 * cov at 16), and the command composites over
// the opaque pixel as the CPU's AA pipe does at that depth:
//   a = div_max(a_input * shape),
//   c = div_max((M - a) * c_dst + a * c_src),
// a_input, c_src and c_dst being samples of the page's depth.  Every
// operand stays below 2^32: x <= M^2, and div_max adds at most
// 2^(bits - 1) + M to it.  A coverage of 0 leaves the pixel unchanged, as
// the CPU's composite does for a zero shape.

static const uint TILE = 16;
static const uint KIND_PATH = 1;
static const uint KIND_MASK = 2;
static const uint NO_BACKDROP = 0xFFFFFFFFu;
static const uint LIST_LEN = 0x3FFFu;
static const uint UNIFORM_BACKDROP = 0x80000000u;

struct Cmd {
    uint kind;      // 0 = coverage table, 1 = outline, 2 = mask
    uint flags;     // outline: bit 0 even-odd, bits 8..15 weight; mask: left
    uint rg;        // r | g << 16
    uint a_input;
    int table_y0;   // pixel row of the table's row 0; mask: top
    uint n_rows;    // mask: width
    uint row_base;  // index of the table's first entry in row_starts;
                    // mask: offset of its first coverage byte
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
    uint segs_base;
    uint lists_base;
    uint backdrops_base;
    uint images_base;
};

[[vk::push_constant]] ConstantBuffer<Params> pc;

[[vk::binding(0, 0)]] StructuredBuffer<Cmd> cmds;
[[vk::binding(1, 0)]] StructuredBuffer<int> table;
[[vk::binding(2, 0)]] StructuredBuffer<uint> row_starts;
[[vk::binding(3, 0)]] StructuredBuffer<uint> tile_offsets;
[[vk::binding(4, 0)]] StructuredBuffer<uint> tile_items;
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

// The coverage of pixel (x, y) by table command c.
uint table_coverage(Cmd c, int x, int y)
{
    int row = y - c.table_y0;
    if (row < 0 || uint(row) >= c.n_rows) return 0;
    uint start = row_starts[c.row_base + uint(row)];
    uint stop = row_starts[c.row_base + uint(row) + 1];
    return row_coverage(start, stop, x);
}

// The coverage of pixel (x, y), inside its bounds, by mask command c.
uint mask_coverage(Cmd c, int x, int y)
{
    uint row = uint(y - c.table_y0);
    uint col = uint(x - int(c.flags));
    return coverage_byte(c.row_base + row * c.n_rows + col);
}

// The coverage of pixel (x, y), inside its bounds, by table or mask
// command c.
uint flat_coverage(Cmd c, int x, int y)
{
    return c.kind == KIND_MASK ? mask_coverage(c, x, y) : table_coverage(c, x, y);
}

#include "page_image.slang"

// Composite command c over the pixel (r, g, b) with coverage cov > 0.
void composite(Cmd c, uint cov, inout uint r, inout uint g, inout uint b)
{
    uint bits = pc.bits;
    uint max_sample = (1u << bits) - 1;
    uint a = div_max(c.a_input * (cov * (max_sample / 255)), bits);
    uint inv = max_sample - a;
    r = div_max(inv * r + a * (c.rg & 0xFFFF), bits);
    g = div_max(inv * g + a * (c.rg >> 16), bits);
    b = div_max(inv * b + a * c.b, bits);
}

#ifdef PAGE_PATHS

static const int64_t ONE = 256;
static const int64_t FULL_AREA = 2 * 256 * 256;

// Outline items whose cells a workgroup holds at once.
static const uint BATCH = 8;

// The cells of the batch's outline items, TILE * TILE per item; after
// the row scan, g_cover holds each cell's cover summed along its row.
groupshared int g_cover[BATCH * TILE * TILE];
groupshared int g_area[BATCH * TILE * TILE];
// Per batch item: its (segment, row) pairs, 0 unless it deposits cells,
// and its segment list's start.
groupshared uint g_pairs[BATCH];
groupshared uint g_list[BATCH];

// floor(a / b) for 0 < b < 2^32 and |a / b| < 2^40, without a 64-bit
// division: two single-precision estimates of the quotient, each followed
// by its exact remainder, leave it within a few units, which the final
// steps correct.  The first estimate is within |a / b| * 2^-23 + 1 of the
// quotient, so the second divides a remainder below 2^18 * b and is within
// 2 of it.
int64_t floor_div(int64_t a, int64_t b)
{
    float fb = float(b);
    int64_t q = int64_t(float(a) / fb);
    int64_t r = a - q * b;
    q += int64_t(float(r) / fb);
    r = a - q * b;
    while (r < 0) {
        q -= 1;
        r += b;
    }
    while (r >= b) {
        q += 1;
        r -= b;
    }
    return q;
}

// The x of segment (xa, ya) -> (xb, yb), ya < yb, at height y in [ya, yb].
int64_t x_at(int64_t xa, int64_t ya, int64_t xb, int64_t yb, int64_t y)
{
    if (y == ya) return xa;
    if (y == yb) return xb;
    return xa + floor_div((y - ya) * (xb - xa), yb - ya);
}

void deposit(uint base, int64_t ex, int64_t ey, int64_t cover, int64_t area)
{
    uint at = base + uint(ey) * TILE + uint(ex);
    InterlockedAdd(g_cover[at], int(cover));
    InterlockedAdd(g_area[at], int(area));
}

// The CPU rasteriser's hline over a window TILE cells wide: the cells of
// the part of a segment within row ey from left to right, xl <= xr,
// deposited into the cells from index base on.
void hline(uint base, int64_t ey, int64_t xl, int64_t yl, int64_t xr, int64_t yr, int64_t sign)
{
    if (yl == yr) return;
    int64_t w = int64_t(TILE);
    if (xl == xr) {
        int64_t ex = xl >> 8;
        int64_t cover = sign * (yr - yl);
        if (ex < 0) deposit(base, 0, ey, cover, 0);
        else if (ex < w) deposit(base, ex, ey, cover, cover * 2 * (xl - ex * ONE));
        return;
    }
    int64_t first_cell = xl >> 8;
    int64_t end_cell = (xr - 1) >> 8;
    int64_t last_cell = min(end_cell, w - 1);
    if (first_cell == end_cell && first_cell >= 0) {
        if (first_cell < w) {
            int64_t cover = sign * (yr - yl);
            deposit(base, first_cell, ey, cover, cover * (xl + xr - 2 * first_cell * ONE));
        }
        return;
    }
    if (first_cell >= w) return;
    if (xr <= 0) {
        // Wholly left of the tile: its cover folds into the left column.
        deposit(base, 0, ey, sign * (yr - yl), 0);
        return;
    }
    int64_t dx = xr - xl;
    int64_t dy = yr - yl;
    int64_t start = max(first_cell, int64_t(0));
    // y on each cell boundary from start on: yl + floor((n0 + k * step) / dx).
    int64_t n0 = (start * ONE - xl) * dy;
    int64_t step = ONE * dy;
    int64_t quotient = floor_div(n0, dx);
    int64_t q = yl + quotient;
    int64_t r = n0 - quotient * dx;
    int64_t lift = floor_div(step, dx);
    int64_t rem = step - lift * dx;
    if (first_cell < 0) deposit(base, 0, ey, sign * (q - yl), 0);
    for (int64_t cell = start; cell <= last_cell; cell++) {
        int64_t cell_left = cell * ONE;
        int64_t x_in = xl;
        int64_t y_in = yl;
        if (xl < cell_left) {
            x_in = cell_left;
            y_in = q;
        }
        q += lift;
        r += rem;
        if (r >= dx) {
            r -= dx;
            q += 1;
        }
        int64_t x_out = xr;
        int64_t y_out = yr;
        if (xr > cell_left + ONE) {
            x_out = cell_left + ONE;
            y_out = q;
        }
        int64_t cover = sign * (y_out - y_in);
        deposit(base, cell, ey, cover, cover * (x_in + x_out - 2 * cell_left));
    }
}

// Deposit pixel row `row` (0..TILE) of the segment (x0, y0) -> (x1, y1),
// fixed point relative to the tile's top-left corner.
void add_row(uint base, int64_t x0, int64_t y0, int64_t x1, int64_t y1, int64_t row)
{
    if (y0 == y1) return;
    int64_t sign = 1;
    int64_t xa = x0, ya = y0, xb = x1, yb = y1;
    if (y0 > y1) {
        sign = -1;
        xa = x1; ya = y1; xb = x0; yb = y0;
    }
    int64_t row_top = row * ONE;
    if (yb <= row_top || ya >= row_top + ONE) return;
    int64_t x_top = xa;
    int64_t fy_top = ya - row_top;
    if (ya < row_top) {
        x_top = x_at(xa, ya, xb, yb, row_top);
        fy_top = 0;
    }
    int64_t x_bot = xb;
    int64_t fy_bot = yb - row_top;
    if (yb > row_top + ONE) {
        x_bot = x_at(xa, ya, xb, yb, row_top + ONE);
        fy_bot = ONE;
    }
    if (x_top <= x_bot) hline(base, row, x_top, fy_top, x_bot, fy_bot, sign);
    else hline(base, row, x_bot, fy_bot, x_top, fy_top, -sign);
}

// The CPU rasteriser's coverage byte of the accumulated value v.
uint rule_byte(int64_t v, bool even_odd)
{
    int64_t a;
    if (even_odd) {
        // v modulo 2 * FULL_AREA = 2^18, non-negative.
        int64_t m = v & (2 * FULL_AREA - 1);
        a = (m > FULL_AREA) ? 2 * FULL_AREA - m : m;
    } else {
        int64_t m = abs(v);
        a = (m > FULL_AREA) ? FULL_AREA : m;
    }
    // 0 <= a <= FULL_AREA = 2^17: the rounded quotient by 2^17.
    return uint((a * 255 + FULL_AREA / 2) >> 17);
}

// c * weight / 255, rounded to nearest.
uint weighted(uint c, uint weight)
{
    if (weight == 255) return c;
    uint t = c * weight + 0x80u;
    return min((t + (t >> 8)) >> 8, 255u);
}

// The item k of the tile lists.
uint4 load_item(uint k)
{
    return uint4(tile_items[k * 4], tile_items[k * 4 + 1], tile_items[k * 4 + 2],
                 tile_items[k * 4 + 3]);
}

// Whether item of command c deposits cells: an outline listing segments.
bool deposits(Cmd c, uint4 item)
{
    return c.kind == KIND_PATH && (item.z & LIST_LEN) != 0;
}

// The backdrop cover of row `row` of the tile of outline item `item`.
int backdrop_cover(uint4 item, uint row)
{
    if ((item.z & UNIFORM_BACKDROP) != 0) return int(item.w);
    if (item.w == NO_BACKDROP) return 0;
    return table[pc.backdrops_base + item.w * TILE + row];
}

// Clear the cells of the batch k0 .. k0 + n and note each item's pairs;
// thread li clears cell li of every item and notes item li.
void clear_batch(uint k0, uint n, uint li)
{
    for (uint s = 0; s < n; s++) {
        g_cover[s * TILE * TILE + li] = 0;
        g_area[s * TILE * TILE + li] = 0;
    }
    if (li < n) {
        uint4 item = load_item(k0 + li);
        g_pairs[li] = deposits(cmds[item.x], item) ? (item.z & LIST_LEN) * TILE : 0;
        g_list[li] = item.y;
    }
}

// Deposit the cells of the outline items k0 .. k0 + n of tile (tx, ty),
// the item k0 + s into the cells from s * TILE * TILE on: thread li adds
// backdrop row li % TILE of item li / TILE and walks every TILE * TILE-th
// (segment, row) pair of the batch.  Every thread of the workgroup calls
// it together, after clear_batch and a barrier.
void deposit_batch(uint k0, uint n, uint li, uint tx, uint ty)
{
    uint slot = li / TILE;
    if (slot < n && g_pairs[slot] != 0) {
        int cover = backdrop_cover(load_item(k0 + slot), li % TILE);
        if (cover != 0) {
            InterlockedAdd(g_cover[slot * TILE * TILE + (li % TILE) * TILE], cover);
        }
    }
    uint total = 0;
    for (uint s = 0; s < n; s++) total += g_pairs[s];
    int64_t ox = int64_t(tx * TILE) * ONE;
    int64_t oy = int64_t(ty * TILE) * ONE;
    for (uint j = li; j < total; j += TILE * TILE) {
        // The item and pair of pair j.
        uint rest = j;
        uint at_slot = 0;
        while (rest >= g_pairs[at_slot]) {
            rest -= g_pairs[at_slot];
            at_slot++;
        }
        uint seg = g_pairs[at_slot] == TILE
            ? g_list[at_slot]
            : uint(table[pc.lists_base + g_list[at_slot] + rest / TILE]);
        uint at = pc.segs_base + seg * 4;
        add_row(at_slot * TILE * TILE,
                int64_t(table[at]) - ox, int64_t(table[at + 1]) - oy,
                int64_t(table[at + 2]) - ox, int64_t(table[at + 3]) - oy,
                int64_t(rest % TILE));
    }
}

// Sum the covers of every row of the batch along the row, in place:
// thread li scans row li % TILE of item li / TILE.  The host's bounds keep
// every sum within 32 bits.
void scan_batch(uint n, uint li)
{
    uint slot = li / TILE;
    if (slot >= n || g_pairs[slot] == 0) return;
    uint base = slot * TILE * TILE + (li % TILE) * TILE;
    int acc = 0;
    for (uint k = 0; k < TILE; k++) {
        acc += g_cover[base + k];
        g_cover[base + k] = acc;
    }
}

// The coverage of pixel (lx, ly) by the outline command c with its item,
// the batch's slot `slot`, once the batch's cells are deposited.
uint path_coverage(Cmd c, uint4 item, uint slot, uint lx, uint ly)
{
    bool even_odd = (c.flags & 1u) != 0;
    uint weight = (c.flags >> 8) & 0xFFu;
    if ((item.z & LIST_LEN) == 0) {
        // No segment deposits in the tile: every pixel of a row takes the
        // row's backdrop alone.
        int64_t cover = int64_t(backdrop_cover(item, ly));
        return weighted(rule_byte(cover * 2 * ONE, even_odd), weight);
    }
    uint at = slot * TILE * TILE + ly * TILE + lx;
    int64_t v = int64_t(g_cover[at]) * 2 * ONE - int64_t(g_area[at]);
    return weighted(rule_byte(v, even_odd), weight);
}

#endif

void raster(uint3 lid, uint3 wid)
{
    uint px = wid.x * TILE + lid.x;
    uint py = wid.y * TILE + lid.y;
    // Every thread runs every item: outlines synchronise the workgroup.
    bool on_page = px < pc.width && py < pc.height;

    uint tile = wid.y * pc.tiles_x + wid.x;
    uint begin = tile_offsets[tile];
    uint end = tile_offsets[tile + 1];
    int x = int(px);
    int y = int(py);
    uint r = pc.paper_rg & 0xFFFF;
    uint g = pc.paper_rg >> 16;
    uint b = pc.paper_b;
#ifdef PAGE_PATHS
    uint li = lid.y * TILE + lid.x;
    for (uint k0 = begin; k0 < end; k0 += BATCH) {
        uint n = min(BATCH, end - k0);
        // Every thread has read the previous batch's cells.
        GroupMemoryBarrierWithGroupSync();
        clear_batch(k0, n, li);
        GroupMemoryBarrierWithGroupSync();
        deposit_batch(k0, n, li, wid.x, wid.y);
        GroupMemoryBarrierWithGroupSync();
        scan_batch(n, li);
        GroupMemoryBarrierWithGroupSync();
        for (uint s = 0; s < n; s++) {
            uint4 item = load_item(k0 + s);
            Cmd c = cmds[item.x];
            bool inside = on_page && x >= c.x0 && x <= c.x1 && y >= c.y0 && y <= c.y1;
            if (!inside) continue;
            if (c.kind == KIND_IMAGE) {
                image_paint(c, x, y, r, g, b);
                continue;
            }
            uint cov = c.kind == KIND_PATH ? path_coverage(c, item, s, lid.x, lid.y)
                                           : flat_coverage(c, x, y);
            if (cov == 0) continue;
            composite(c, cov, r, g, b);
        }
    }
#else
    for (uint k = begin; k < end; k++) {
        Cmd c = cmds[tile_items[k * 4]];
        bool inside = on_page && x >= c.x0 && x <= c.x1 && y >= c.y0 && y <= c.y1;
        if (!inside) continue;
        if (c.kind == KIND_IMAGE) {
            image_paint(c, x, y, r, g, b);
            continue;
        }
        uint cov = flat_coverage(c, x, y);
        if (cov == 0) continue;
        composite(c, cov, r, g, b);
    }
#endif

    if (!on_page) return;
    uint o = (py * pc.width + px) * 3;
    store(o, r);
    store(o + 1, g);
    store(o + 2, b);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void page_raster(uint3 lid : SV_GroupThreadID, uint3 wid : SV_GroupID)
{
    raster(lid, wid);
}
