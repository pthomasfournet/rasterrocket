// Image blit kernel.
//
// Copies a source image into a destination RGBA8 page buffer through
// caller-supplied sampling tables.  Each output pixel is one thread; the
// source pixel it reads is `cols[dx] + rows[dy]` in Q32 fixed point, so
// the kernel performs no floating-point arithmetic and no per-pixel
// transform of its own.  The caller derives the tables from the image's
// placement (the CPU sampler consumes the same tables, which is what
// makes the two paths select identical source pixels).
//
// Inputs
// ------
//   src         : decoded image bytes, layout-dependent stride
//   src_w       : source width in pixels
//   src_h       : source height in pixels
//   src_layout  : 0 = RGB (3 bytes/pixel), 1 = Gray (1 byte/pixel)
//                 (Mask layout is handled on CPU; not dispatched here.)
//   dst_rgba    : destination RGBA8 buffer (4 bytes/pixel, row-major)
//   dst_w       : destination width in pixels (page width)
//   dst_h       : destination height in pixels (page height)
//   bx0, by0    : top-left of the bounding-box subregion to render
//                 (output coords)
//   bx1, by1    : bottom-right (exclusive)
//   cols        : (bx1 - bx0) entries of 4 u32 words: x_hi, x_lo, y_hi, y_lo
//   rows        : (by1 - by0) entries of the same shape
//
// Table entry words are the Q32 split of one term of the source
// coordinate: `hi` holds the two's-complement integer part, `lo` the
// fraction.  Column and row terms are summed with carry from the fraction
// word; the integer result is the source pixel index.  Samples whose
// index falls outside `[0, src_w) × [0, src_h)` leave the destination
// untouched (the caller allocates the page buffer zero-initialised, so
// untouched pixels composite as fully transparent).

extern "C" __global__ void blit_image(
    const unsigned char* __restrict__ src,
    int src_w,
    int src_h,
    int src_layout,
    unsigned char* __restrict__ dst_rgba,
    int dst_w,
    int dst_h,
    int bx0, int by0,
    int bx1, int by1,
    const unsigned int* __restrict__ cols,
    const unsigned int* __restrict__ rows
) {
    int dx = bx0 + (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int dy = by0 + (int)(blockIdx.y * blockDim.y + threadIdx.y);
    // Reject pixels outside the bbox AND outside the page.  The page
    // bounds (`>= 0` and `>= dst_w/dst_h`) are load-bearing for
    // memory safety: a bbox that spills off the page would otherwise
    // index dst_rgba out of bounds.
    if (dx < 0 || dy < 0 || dx >= bx1 || dy >= by1 || dx >= dst_w || dy >= dst_h) {
        return;
    }

    const unsigned int* c = cols + (size_t)(dx - bx0) * 4;
    const unsigned int* r = rows + (size_t)(dy - by0) * 4;

    unsigned int x_lo = c[1] + r[1];
    int ix = (int)c[0] + (int)r[0] + (x_lo < c[1] ? 1 : 0);
    unsigned int y_lo = c[3] + r[3];
    int iy = (int)c[2] + (int)r[2] + (y_lo < c[3] ? 1 : 0);
    if (ix < 0 || ix >= src_w || iy < 0 || iy >= src_h) {
        return;
    }

    int dst_off = (dy * dst_w + dx) * 4;
    if (src_layout == 0) {
        // RGB: 3 bytes per source pixel
        int src_off = (iy * src_w + ix) * 3;
        dst_rgba[dst_off + 0] = src[src_off + 0];
        dst_rgba[dst_off + 1] = src[src_off + 1];
        dst_rgba[dst_off + 2] = src[src_off + 2];
        dst_rgba[dst_off + 3] = 255;
    } else {
        // Gray: 1 byte per source pixel; broadcast to RGB.
        int src_off = iy * src_w + ix;
        unsigned char g = src[src_off];
        dst_rgba[dst_off + 0] = g;
        dst_rgba[dst_off + 1] = g;
        dst_rgba[dst_off + 2] = g;
        dst_rgba[dst_off + 3] = 255;
    }
}
