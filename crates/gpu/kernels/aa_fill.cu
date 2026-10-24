// Scanline anti-aliased fill: per-pixel sub-sample counts from a scanline
// intersection table.
//
// The table is the CPU rasteriser's `XPathScanner` in AA (4×) coordinates:
// for each AA row, its intersections (x0, x1, count) sorted by x0.  One
// thread per output pixel walks the pixel's four AA sub-rows exactly as the
// CPU's scanline walk does: every intersection covers its own cells
// [x0, x1] whatever its winding count, and a span absorbs the following
// intersections while they touch it or the running count (masked by the
// fill rule) is inside; a row that ends inside runs to aa_width.  Spans are
// clamped to [0, aa_width).  Coverage is the number of the pixel's 16
// sub-samples that lie inside a span (a union — overlapping spans do not
// double count).
//
// The walk stops at the first intersection starting at or beyond the
// pixel's last sub-sample: intersections are sorted by x0, so no later span
// can reach the pixel.
//
// Output: one byte per pixel holding the count 0..=16.  The coverage →
// alpha mapping is applied by the host.
//
// Inputs
//   intersects : int triples [x0, x1, count], grouped by row
//   row_starts : n_rows + 1 offsets; row r is intersects[row_starts[r]
//                .. row_starts[r + 1])
//   n_rows     : rows in the table
//   table_y0   : AA row index of table row 0
//   x_min,y_min: device-pixel origin of the output rectangle
//   width,height: output rectangle extent in pixels
//   aa_width   : bitmap width × 4 — spans are clamped to [0, aa_width)
//   eo         : 1 = even-odd, 0 = non-zero winding
//   coverage   : output, width × height bytes (row-major)

#define AA_SIZE 4
#define BLOCK_W 16
#define BLOCK_H 16

// Sub-sample bitmask of the row's coverage within the four AA columns
// [ax0, ax0 + AA_SIZE).
__device__ unsigned int row_coverage_mask(const int* __restrict__ intersects,
                                          unsigned int start, unsigned int end,
                                          int aa_width, int eo_mask, int ax0)
{
    unsigned int mask = 0;
    int count = 0;
    unsigned int i = start;

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
                mask |= ((1u << (hi - lo)) - 1u) << (lo - ax0);
            }
        }
    }
    return mask;
}

// Grid: (ceil(width / BLOCK_W), ceil(height / BLOCK_H), 1),
// Block: (BLOCK_W, BLOCK_H, 1).
extern "C" __global__ void aa_fill(
    const int* __restrict__ intersects,
    const unsigned int* __restrict__ row_starts,
    unsigned int n_rows,
    int table_y0,
    int x_min,
    int y_min,
    unsigned int width,
    unsigned int height,
    unsigned int aa_width,
    int eo,
    unsigned char* __restrict__ coverage
) {
    unsigned int px = blockIdx.x * BLOCK_W + threadIdx.x;
    unsigned int py = blockIdx.y * BLOCK_H + threadIdx.y;
    if (px >= width || py >= height) return;

    int eo_mask = eo ? 1 : ~0;
    int aa_w = (int)aa_width;
    int ax0 = (x_min + (int)px) * AA_SIZE;
    int ay0 = (y_min + (int)py) * AA_SIZE;

    unsigned int total = 0;
    for (int yy = 0; yy < AA_SIZE; yy++) {
        int r = ay0 + yy - table_y0;
        if (r < 0 || (unsigned int)r >= n_rows) continue;
        unsigned int start = row_starts[r];
        unsigned int end = row_starts[r + 1];
        total += __popc(row_coverage_mask(intersects, start, end, aa_w, eo_mask, ax0));
    }

    coverage[py * width + px] = (unsigned char)total;
}
