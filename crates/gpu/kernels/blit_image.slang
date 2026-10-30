// Image blit kernel — table-driven sampling into an RGBA8 page buffer.
// Slang port of blit_image.cu; semantics must match byte-for-byte.
//
// The source coordinate for output (dx, dy) is `cols[dx] + rows[dy]` in
// Q32 fixed point.  Each table entry is four uint words: x_hi, x_lo, y_hi,
// y_lo — `hi` is the two's-complement integer part, `lo` the fraction.
// Column and row terms are summed with carry from the fraction word, so
// the kernel needs neither floating point nor 64-bit integers.
//
// `filter` bit 0 interpolates between source columns, bit 1 between
// source rows.  An interpolated axis measures the coordinate from sample
// centres by subtracting 2^31 from the fraction with borrow; the taps are
// the floor of that coordinate and its successor, clamped onto the axis,
// and the top 8 bits of its fraction weight the upper tap.  The four taps
// are blended per channel as `((p00·(256−wx) + p10·wx)·(256−wy) +
// (p01·(256−wx) + p11·wx)·wy + 2^15) >> 16`.
//
// Scalars travel via push constants (40 bytes); the tables are storage
// buffers because their size scales with the bounding box.

static const uint WEIGHT_ONE = 256u;
static const uint HALF_SAMPLE = 0x80000000u;

// Taps along one axis of `n` samples for source index `i` (in `[0, n)`)
// and fraction `frac`; `w` is the weight of `hi`.
void axis_taps(int i, uint frac, int n, bool blend, out int lo, out int hi, out uint w)
{
    if (!blend) {
        lo = i;
        hi = i;
        w = 0u;
        return;
    }
    int lower = i - (frac < HALF_SAMPLE ? 1 : 0);
    lo = max(lower, 0);
    hi = min(lower + 1, n - 1);
    w = (lo == hi) ? 0u : ((frac - HALF_SAMPLE) >> 24);
}

// Weighted mean of byte `ch` at the four tap offsets, rounded to nearest.
uint blend_taps(
    StructuredBuffer<uint8_t> src, int ch,
    int o00, int o10, int o01, int o11,
    uint wx, uint wy)
{
    uint r0 = uint(src[o00 + ch]) * (WEIGHT_ONE - wx) + uint(src[o10 + ch]) * wx;
    uint r1 = uint(src[o01 + ch]) * (WEIGHT_ONE - wx) + uint(src[o11 + ch]) * wx;
    return (r0 * (WEIGHT_ONE - wy) + r1 * wy + 0x8000u) >> 16;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void blit_image(
    uint3 gid : SV_DispatchThreadID,
    uniform StructuredBuffer<uint8_t> src,
    uniform RWStructuredBuffer<uint> dst_rgba,   // one RGBA8 pixel per word, R in the low byte
    uniform StructuredBuffer<uint> cols,
    uniform StructuredBuffer<uint> rows,
    uniform int src_w,
    uniform int src_h,
    uniform int src_layout,            // 0 = RGB, 1 = Gray
    uniform int dst_w,
    uniform int dst_h,
    uniform int bx0, uniform int by0,
    uniform int bx1, uniform int by1,
    uniform uint filter)               // bit 0 = columns, bit 1 = rows
{
    // SV_DispatchThreadID is the global lane index (group * groupsize + lane).
    int dx = bx0 + int(gid.x);
    int dy = by0 + int(gid.y);

    // Reject pixels outside the bbox AND outside the page.  Page bounds are
    // load-bearing for memory safety — a bbox that spills off the page
    // would otherwise index dst_rgba out of bounds.
    if (dx < 0 || dy < 0 || dx >= bx1 || dy >= by1 || dx >= dst_w || dy >= dst_h) {
        return;
    }

    uint c = uint(dx - bx0) * 4;
    uint r = uint(dy - by0) * 4;

    uint x_lo = cols[c + 1] + rows[r + 1];
    int ix = asint(cols[c]) + asint(rows[r]) + (x_lo < cols[c + 1] ? 1 : 0);
    uint y_lo = cols[c + 3] + rows[r + 3];
    int iy = asint(cols[c + 2]) + asint(rows[r + 2]) + (y_lo < cols[c + 3] ? 1 : 0);
    if (ix < 0 || ix >= src_w || iy < 0 || iy >= src_h) {
        return;
    }

    int x0, x1, y0, y1;
    uint wx, wy;
    axis_taps(ix, x_lo, src_w, (filter & 1u) != 0u, x0, x1, wx);
    axis_taps(iy, y_lo, src_h, (filter & 2u) != 0u, y0, y1, wy);

    int bpp = src_layout == 0 ? 3 : 1;
    int o00 = (y0 * src_w + x0) * bpp;
    int o10 = (y0 * src_w + x1) * bpp;
    int o01 = (y1 * src_w + x0) * bpp;
    int o11 = (y1 * src_w + x1) * bpp;

    uint dst_px = uint(dy * dst_w + dx);
    if (src_layout == 0) {
        dst_rgba[dst_px] = blend_taps(src, 0, o00, o10, o01, o11, wx, wy)
                         | (blend_taps(src, 1, o00, o10, o01, o11, wx, wy) << 8)
                         | (blend_taps(src, 2, o00, o10, o01, o11, wx, wy) << 16)
                         | 0xFF000000u;
    } else {
        uint g = blend_taps(src, 0, o00, o10, o01, o11, wx, wy);
        dst_rgba[dst_px] = g | (g << 8) | (g << 16) | 0xFF000000u;
    }
}
