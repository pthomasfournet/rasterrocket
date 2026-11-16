// Clips and soft-masked objects of the page raster, included by
// page_raster.slang.  Their records are words of the table from word
// pc.mods_base on:
//   clip region  y_min, n_rows, row_base: a coverage table, its rows in
//                row_starts as a table command's;
//   clip         n, then the words of its n regions, newest first;
//   mask plane   offset, deep, x0, y0, x1, y1, outside: the samples of the
//                inclusive rectangle (x0, y0) - (x1, y1) from byte offset
//                of the coverage bytes, row by row, two bytes each, low
//                byte first, when deep; every pixel outside it is outside
//                (x1 < x0 for an empty rectangle);
//   object       plane, x0, y0, x1, y1: the word of its plane's record and
//                its inclusive rectangle.
//
// A clip scales a value v region by region, newest first, as
// v = div255(v * c), c the region's coverage of the pixel.
//
// The items of an object are consecutive in a tile.  A pixel's value before
// the first of them is kept; after the last, inside the object's rectangle,
// with mask value m < M, each channel c becomes
// div_max(c * m + p * (M - m)), p the kept value (below 2^32: at most M^2).

// Modifier word i.
uint mod_word(uint i)
{
    return uint(table[pc.mods_base + i]);
}

// The coverage of pixel (x, y) by the clip region whose record is word w.
uint region_coverage(uint w, int x, int y)
{
    int row = y - int(mod_word(w));
    if (row < 0 || uint(row) >= mod_word(w + 1)) return 0;
    uint at = mod_word(w + 2) + uint(row);
    return row_coverage(row_starts[at], row_starts[at + 1], x);
}

// v scaled at pixel (x, y) by the regions of the clip whose record is word w.
uint clip_scale(uint w, int x, int y, uint v)
{
    uint n = mod_word(w);
    for (uint i = 0; i < n && v != 0; i++) {
        v = div_max(v * region_coverage(mod_word(w + 1 + i), x, y), 8);
    }
    return v;
}

// The value at pixel (x, y) of the mask plane whose record is word w.
uint plane_value(uint w, int x, int y)
{
    int x0 = int(mod_word(w + 2));
    int y0 = int(mod_word(w + 3));
    int x1 = int(mod_word(w + 4));
    int y1 = int(mod_word(w + 5));
    if (x < x0 || x > x1 || y < y0 || y > y1) return mod_word(w + 6);
    uint i = uint(y - y0) * uint(x1 - x0 + 1) + uint(x - x0);
    uint at = mod_word(w);
    if (mod_word(w + 1) != 0) {
        return coverage_byte(at + 2 * i) | (coverage_byte(at + 2 * i + 1) << 8);
    }
    return coverage_byte(at + i);
}

// Blend pixel (x, y) back toward (pr, pg, pb) by the mask of the object
// whose record is word w.
void close_object(uint w, int x, int y, uint pr, uint pg, uint pb, inout uint r, inout uint g,
                  inout uint b)
{
    if (x < int(mod_word(w + 1)) || x > int(mod_word(w + 3)) || y < int(mod_word(w + 2))
        || y > int(mod_word(w + 4))) {
        return;
    }
    uint bits = pc.bits;
    uint max_sample = (1u << bits) - 1;
    uint m = plane_value(mod_word(w), x, y);
    if (m == max_sample) return;
    uint im = max_sample - m;
    r = div_max(r * m + pr * im, bits);
    g = div_max(g * m + pg * im, bits);
    b = div_max(b * m + pb * im, bits);
}

// Before an item of object `object` (1 + its record's word, 0 for none):
// when it is not the open object, close the open one and keep the pixel
// for the new one.
void switch_object(uint object, int x, int y, inout uint open, inout uint pr, inout uint pg,
                   inout uint pb, inout uint r, inout uint g, inout uint b)
{
    if (object == open) return;
    if (open != 0) close_object(open - 1, x, y, pr, pg, pb, r, g, b);
    open = object;
    pr = r;
    pg = g;
    pb = b;
}
