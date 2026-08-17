// GPU tile-parallel analytical fill rasterisation.
//
// Algorithm (vello-style signed-area integration):
//
//   Each path segment is "tiled" into one record per tile row it crosses.
//   Records are sorted by (tile_y, tile_x) by the host (CPU or CUB) before
//   passing to this kernel.  `tile_starts[i]` and `tile_counts[i]` tell the
//   fill kernel where to find records for tile i.
//
//   The fill kernel (one block per tile, TILE_W × TILE_H threads) computes
//   per-pixel signed area analytically:
//
//     For each record crossing the pixel row [py, py+1]:
//       - Clip segment to the pixel row.
//       - Accumulate the exact clipped-trapezoid integral of
//         clamp(x(y) - px, 0, 1) over the row (see segment_pixel_area).
//
//   Non-zero winding: coverage = min(|area|, 1) × 255.5, truncated.
//   Even-odd: |area| folded with the period-2 triangle wave peaking at
//   odd integers, × 255.5, truncated.
//
// Tile geometry:
//   TILE_W = 16 pixels wide,  TILE_H = 16 pixels tall.
//   Grid: ceil(out_w / TILE_W) × ceil(out_h / TILE_H).

#define TILE_W 16
#define TILE_H 16

// Tile record emitted by the host for each (segment, tile-row) pair.
// Must match the layout in `gpu/src/lib.rs :: TileRecord`.
struct TileRecord {
    unsigned int key;   // (tile_y << 16) | tile_x — sort key (host fills this)
    float x_enter;     // segment x at top of this tile's y-extent (tile-local x coords)
    float dxdy;        // slope dx/dy (device pixels per device pixel)
    float y0_tile;     // segment start y within tile (0..TILE_H)
    float y1_tile;     // segment end y within tile   (0..TILE_H)
    float sign;        // +1 upward-crossing, -1 downward
    unsigned int _pad; // pad to 32 bytes
    unsigned int _pad2;
};

// Compute the signed winding-area contribution of one segment to pixel
// column `px` for the pixel row [iy0, iy1] (both in tile-local y
// coordinates).
//
// The segment enters at x = x_enter (tile-local) at y = y0_tile, with slope
// dxdy (pixels per pixel).  `iy0` and `iy1` are the already-clipped y extents
// of this segment within the current pixel row.
//
// The contribution is the exact clipped-trapezoid integral
//
//   cover = INTEGRAL over [iy0, iy1] of clamp(x(y) - px, 0, 1) dy
//
// — the segment's winding-count contribution integrated over the pixel's
// interior, so partially-crossed pixels get their true covered area rather
// than a per-y step function sampled at the pixel's left edge (which
// over-weighted diagonal crossings by up to 2×).
//
// Closed form: over [iy0, iy1] the segment traces a linear range [x0, x1].
// Split the y-interval where x(y) crosses px and px + 1:
//
//   - y-span with x >= px + 1 contributes 1 per unit y ("above");
//   - the partial span (x in [px, px+1]) contributes the mean of (x - px)
//     over its linear sweep from `left` to `right`;
//   - y-span with x <= px contributes 0.
//
// Mapping y-fractions through the sorted range [xl, xr] is direction-safe:
// the integrand depends only on the distribution of x values, which is
// uniform along the segment.
//
// Returns sign × cover, where cover is in [0, y_len].
__device__ float segment_pixel_area(float x_enter, float dxdy,
                                    float iy0, float iy1,
                                    float sign, float px)
{
    float y_len = iy1 - iy0;
    if (y_len <= 0.0f) return 0.0f;

    float x0 = x_enter;
    float x1 = x_enter + dxdy * y_len;

    // Left/right x bounds of the segment over [iy0, iy1].
    float xl = fminf(x0, x1);
    float xr = fmaxf(x0, x1);

    // Pixel column spans [px, px + 1).
    float cover;
    if (xr <= px) {
        // Segment entirely to the left of this pixel column: no contribution.
        cover = 0.0f;
    } else if (xl >= px + 1.0f) {
        // Segment entirely to the right: full y_len contributes to winding.
        cover = y_len;
    } else {
        float dx = xr - xl;
        if (dx < 1e-6f) {
            // Near-vertical: constant x ≈ xmid across the row.
            float xmid = 0.5f * (x0 + x1);
            cover = y_len * fminf(fmaxf(xmid - px, 0.0f), 1.0f);
        } else {
            // Clip the x-range [xl, xr] to [px, px+1].
            float left  = fmaxf(xl, px);
            float right = fminf(xr, px + 1.0f);

            // y-fractions (within [0, y_len]) where x(y) = left and x(y) = right.
            // x(y) is linear: x = xl + dx * (y / y_len), so y = (x - xl) / dx * y_len.
            float yf_left  = (left  - xl) / dx * y_len;
            float yf_right = (right - xl) / dx * y_len;

            // y-span where x >= px+1 (fully to the right of this pixel).
            float above = y_len - yf_right;

            cover = (yf_right - yf_left) * ((left + right) * 0.5f - px) + above;
        }
    }

    return sign * cover;
}

// Tile fill kernel.
//
// Grid: (grid_w, grid_h, 1), Block: (TILE_W, TILE_H, 1).
//
// Parameters:
//   records     : tile records sorted by (tile_y << 16 | tile_x)
//   tile_starts : start index of records for each flat tile index
//   tile_counts : number of records for each flat tile index
//   grid_w      : number of tiles in x direction
//   out_w       : output coverage buffer width (pixels)
//   out_h       : output coverage buffer height (pixels)
//   eo          : 1 = even-odd, 0 = non-zero winding
//   coverage    : output, out_w × out_h bytes
extern "C" __global__ void tile_fill(
    const TileRecord* __restrict__ records,
    const unsigned int* __restrict__ tile_starts,
    const unsigned int* __restrict__ tile_counts,
    unsigned int grid_w,
    unsigned int out_w, unsigned int out_h,
    int eo,
    unsigned char* __restrict__ coverage
) {
    unsigned int tile_x = blockIdx.x;
    unsigned int tile_y = blockIdx.y;
    unsigned int px_local = threadIdx.x; // 0..TILE_W-1
    unsigned int py_local = threadIdx.y; // 0..TILE_H-1

    unsigned int px = tile_x * TILE_W + px_local;
    unsigned int py = tile_y * TILE_H + py_local;
    if (px >= out_w || py >= out_h) return;

    unsigned int tile_idx = tile_y * grid_w + tile_x;
    unsigned int rec_start = tile_starts[tile_idx];
    unsigned int rec_count = tile_counts[tile_idx];

    float area = 0.0f;
    float py_f = (float)py_local;

    for (unsigned int r = rec_start; r < rec_start + rec_count; r++) {
        TileRecord rec = records[r];

        // Clip segment to pixel row [py_local, py_local + 1].
        float iy0 = fmaxf(rec.y0_tile, py_f);
        float iy1 = fminf(rec.y1_tile, py_f + 1.0f);
        if (iy0 >= iy1) continue;

        // x of segment at iy0 (x_enter is at y0_tile; advance by dxdy).
        float x_at_iy0 = rec.x_enter + rec.dxdy * (iy0 - rec.y0_tile);
        float px_f = (float)px_local;

        area += segment_pixel_area(x_at_iy0, rec.dxdy, iy0, iy1, rec.sign, px_f);
    }

    int cov;
    if (eo) {
        // Even-odd folds the accumulated signed area with the period-2
        // triangle wave peaking at odd integers: a fully interior pixel
        // of a simple path (|area| = 1) maps to full coverage, and
        // winding-2 overlap regions fold back to zero. min(cov, 255)
        // is load-bearing: float rounding can push a slightly above 1.
        float t = fmodf(fabsf(area), 2.0f);
        float a = (t > 1.0f) ? 2.0f - t : t;
        cov = (int)(a * 255.5f);
    } else {
        cov = (int)(fminf(fabsf(area), 1.0f) * 255.5f);
    }
    coverage[py * out_w + px] = (unsigned char)min(cov, 255);
}
