// Scanline anti-aliased fill: per-pixel sub-sample counts from a scanline
// intersection table.  Slang port of aa_fill.cu; see that file for the
// walk semantics.
//
// Inputs and outputs match the CUDA kernel byte-for-byte:
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

static const int AA_SIZE = 4;

// Sub-sample bitmask of the row's coverage within the four AA columns
// [ax0, ax0 + AA_SIZE).
uint row_coverage_mask(StructuredBuffer<int> intersects,
                       uint start, uint end,
                       int aa_width, int eo_mask, int ax0)
{
    uint mask = 0;
    int count = 0;
    uint i = start;
    int xx = 0;

    while (i < end || (count & eo_mask) != 0) {
        if ((count & eo_mask) == 0) {
            if (i >= end) break;
            xx = intersects[i * 3];
            if (xx >= ax0 + AA_SIZE) break;
            count += intersects[i * 3 + 2];
            i++;
            continue;
        }

        int span_x1;
        for (;;) {
            if (i >= end) {
                span_x1 = aa_width;
                break;
            }
            count += intersects[i * 3 + 2];
            int span_end = intersects[i * 3 + 1] + 1;
            i++;
            if ((count & eo_mask) == 0) {
                span_x1 = span_end;
                break;
            }
        }

        int bx0 = max(xx, 0);
        int bx1 = min(span_x1, aa_width);
        if (bx0 < bx1) {
            int lo = max(bx0, ax0);
            int hi = min(bx1, ax0 + AA_SIZE);
            if (lo < hi) {
                mask |= ((1u << uint(hi - lo)) - 1u) << uint(lo - ax0);
            }
        }
        if (i >= end) break;
        xx = span_x1;
    }
    return mask;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void aa_fill(
    uint3 lid : SV_GroupThreadID,
    uint3 wid : SV_GroupID,
    uniform StructuredBuffer<int> intersects,
    uniform StructuredBuffer<uint> row_starts,
    uniform uint n_rows,
    uniform int table_y0,
    uniform int x_min,
    uniform int y_min,
    uniform uint width,
    uniform uint height,
    uniform uint aa_width,
    uniform int eo,
    uniform RWStructuredBuffer<uint8_t> coverage)
{
    uint px = wid.x * 16 + lid.x;
    uint py = wid.y * 16 + lid.y;
    if (px >= width || py >= height) return;

    int eo_mask = (eo != 0) ? 1 : ~0;
    int aa_w = int(aa_width);
    int ax0 = (x_min + int(px)) * AA_SIZE;
    int ay0 = (y_min + int(py)) * AA_SIZE;

    uint total = 0;
    for (int yy = 0; yy < AA_SIZE; yy++) {
        int r = ay0 + yy - table_y0;
        if (r < 0 || uint(r) >= n_rows) continue;
        uint start = row_starts[r];
        uint end = row_starts[r + 1];
        total += countbits(row_coverage_mask(intersects, start, end, aa_w, eo_mask, ax0));
    }

    coverage[py * width + px] = uint8_t(total);
}
