// Image commands of the page raster, included by page_raster.slang.
//
// An image command's row_base is the word of its record in the image
// words (from word pc.images_base of the table on): the image's sampler
// and, after it, its mask's, SAMPLER_WORDS words each:
//   offset   first byte of the samples in the coverage bytes;
//   width, height  in samples;
//   flags    bit 0 16-bit samples (two bytes each, low byte first),
//            bit 1 interpolate columns, bit 2 interpolate rows;
//   x0, y0, x1, y1  the tables' rectangle of page pixels, ends exclusive;
//   cols, rows      the image words of the first column and row entry,
//            four words each: x_hi, x_lo, y_hi, y_lo.
// The command's flags hold the layout (bits 0-1: gray, RGB, stencil), the
// direct composite (bit 2) and the mask (bits 4-5: none, through the
// image's taps, through its own tables); rg and b a stencil's colour.
//
// Pixel (x, y) takes its source coordinate from the column entry of x
// plus the row entry of y, the fractions summed with carry; off the image
// it paints nothing.  An interpolated axis measures the coordinate from
// sample centres (the fraction less 2^31, with borrow): the taps are its
// floor and successor clamped onto the axis, weighted by the top 8 bits
// of its fraction.  Each channel blends as ((p00 (256 - wx) + p10 wx)
// (256 - wy) + (p01 (256 - wx) + p11 wx) wy + 2^15) >> 16, below 2^32 for
// 16-bit samples, at the samples' depth, and an 8-bit value on a 16-bit
// page is widened by 257.  A mask read through its own tables first
// clamps the pixel onto its rectangle, and its taps onto the mask.
//
// The shape is M, the mask's value, or M less a stencil's value, scaled
// by the clip's scaling s of 255 as div_max(shape * s * M / 255); 0 leaves
// the pixel unchanged.  The direct composite writes the colour at shape M
// and (c s + d (M - s)) / M, truncated, otherwise; the pipe composite is
// composite_shape.

static const uint KIND_IMAGE = 3;
static const uint SAMPLER_WORDS = 10;
static const uint SAMPLES_DEEP = 1;
static const uint FILTER_X = 2;
static const uint FILTER_Y = 4;
static const uint LAYOUT_RGB = 1;
static const uint LAYOUT_STENCIL = 2;
static const uint IMAGE_DIRECT = 4;
static const uint MASK_TAPS = 1;
static const uint MASK_OWN = 2;
static const uint HALF_SAMPLE = 0x80000000u;

struct Taps {
    uint x0;
    uint x1;
    uint y0;
    uint y1;
    uint wx;
    uint wy;
};

// Image word i.
uint image_word(uint i)
{
    return uint(table[pc.images_base + i]);
}

// Taps along an axis of n samples at integer part i and fraction frac,
// i clamped onto the axis; w is the weight of hi.
void axis_taps(int i, uint frac, uint n, bool blend, out uint lo, out uint hi, out uint w)
{
    uint last = n - 1;
    w = 0;
    if (i < 0) {
        lo = 0;
        hi = 0;
        return;
    }
    if (uint(i) > last) {
        lo = last;
        hi = last;
        return;
    }
    if (!blend) {
        lo = uint(i);
        hi = uint(i);
        return;
    }
    int lower = i - (frac < HALF_SAMPLE ? 1 : 0);
    lo = uint(max(lower, 0));
    hi = min(uint(lower + 1), last);
    w = (lo == hi) ? 0u : ((frac - HALF_SAMPLE) >> 24);
}

// The taps of pixel (x, y) through the sampler at image word s; false
// when the coordinate lies off the samples and clamped is not set.
bool sampler_taps(uint s, int x, int y, bool clamped, out Taps t)
{
    int rx0 = asint(image_word(s + 4));
    int ry0 = asint(image_word(s + 5));
    if (clamped) {
        x = clamp(x, rx0, asint(image_word(s + 6)) - 1);
        y = clamp(y, ry0, asint(image_word(s + 7)) - 1);
    }
    uint c = image_word(s + 8) + uint(x - rx0) * 4;
    uint r = image_word(s + 9) + uint(y - ry0) * 4;
    uint cx_lo = image_word(c + 1);
    uint x_lo = cx_lo + image_word(r + 1);
    int ix = asint(image_word(c)) + asint(image_word(r)) + (x_lo < cx_lo ? 1 : 0);
    uint cy_lo = image_word(c + 3);
    uint y_lo = cy_lo + image_word(r + 3);
    int iy = asint(image_word(c + 2)) + asint(image_word(r + 2)) + (y_lo < cy_lo ? 1 : 0);
    uint width = image_word(s + 1);
    uint height = image_word(s + 2);
    if (!clamped && (ix < 0 || uint(ix) >= width || iy < 0 || uint(iy) >= height)) {
        return false;
    }
    uint flags = image_word(s + 3);
    axis_taps(ix, x_lo, width, (flags & FILTER_X) != 0, t.x0, t.x1, t.wx);
    axis_taps(iy, y_lo, height, (flags & FILTER_Y) != 0, t.y0, t.y1, t.wy);
    return true;
}

// Sample (x, y), channel ch of channels, of the sampler at image word s.
uint fetch_sample(uint s, uint x, uint y, uint ch, uint channels)
{
    uint i = (y * image_word(s + 1) + x) * channels + ch;
    uint at = image_word(s);
    if ((image_word(s + 3) & SAMPLES_DEEP) != 0) {
        return coverage_byte(at + 2 * i) | (coverage_byte(at + 2 * i + 1) << 8);
    }
    return coverage_byte(at + i);
}

// Channel ch of the sampler at image word s blended over t, at the page's
// depth.
uint blend_sample(uint s, Taps t, uint ch, uint channels)
{
    uint v;
    if (t.wx == 0 && t.wy == 0) {
        v = fetch_sample(s, t.x0, t.y0, ch, channels);
    } else {
        uint r0 = fetch_sample(s, t.x0, t.y0, ch, channels) * (256 - t.wx)
                + fetch_sample(s, t.x1, t.y0, ch, channels) * t.wx;
        uint r1 = fetch_sample(s, t.x0, t.y1, ch, channels) * (256 - t.wx)
                + fetch_sample(s, t.x1, t.y1, ch, channels) * t.wx;
        v = (r0 * (256 - t.wy) + r1 * t.wy + 0x8000u) >> 16;
    }
    bool deep = (image_word(s + 3) & SAMPLES_DEEP) != 0;
    return (pc.bits == 16 && !deep) ? v * 257 : v;
}

// Composite command c over the pixel (r, g, b) with colour (cr, cg, cb)
// and shape s > 0 of the page's depth, as the antialiasing pipe.
void composite_shape(Cmd c, uint s, uint cr, uint cg, uint cb, inout uint r, inout uint g, inout uint b)
{
    uint bits = pc.bits;
    uint max_sample = (1u << bits) - 1;
    uint a = div_max(c.a_input * s, bits);
    uint inv = max_sample - a;
    r = div_max(inv * r + a * cr, bits);
    g = div_max(inv * g + a * cg, bits);
    b = div_max(inv * b + a * cb, bits);
}

// Paint pixel (x, y), inside its bounds, by the image command c.
void image_paint(Cmd c, int x, int y, inout uint r, inout uint g, inout uint b)
{
    uint max_sample = (1u << pc.bits) - 1;
    uint s = c.row_base;
    uint m = s + SAMPLER_WORDS;
    Taps t;
    if (!sampler_taps(s, x, y, false, t)) return;
    uint layout = c.flags & 3u;
    uint mask = (c.flags >> 4) & 3u;
    uint shape = max_sample;
    if (layout == LAYOUT_STENCIL) {
        shape = max_sample - blend_sample(s, t, 0, 1);
    } else if (mask == MASK_TAPS) {
        shape = blend_sample(m, t, 0, 1);
    } else if (mask == MASK_OWN) {
        Taps tm;
        sampler_taps(m, x, y, true, tm);
        shape = blend_sample(m, tm, 0, 1);
    }
    if (shape == 0) return;
    if (c.clip != 0) {
        shape = div_max(shape * (clip_scale(c.clip - 1, x, y, 255u) * (max_sample / 255)), pc.bits);
        if (shape == 0) return;
    }
    uint cr, cg, cb;
    if (layout == LAYOUT_RGB) {
        cr = blend_sample(s, t, 0, 3);
        cg = blend_sample(s, t, 1, 3);
        cb = blend_sample(s, t, 2, 3);
    } else if (layout == LAYOUT_STENCIL) {
        cr = c.rg & 0xFFFF;
        cg = c.rg >> 16;
        cb = c.b;
    } else {
        cr = blend_sample(s, t, 0, 1);
        cg = cr;
        cb = cr;
    }
    if ((c.flags & IMAGE_DIRECT) == 0) {
        composite_shape(c, shape, cr, cg, cb, r, g, b);
    } else if (shape == max_sample) {
        r = cr;
        g = cg;
        b = cb;
    } else {
        uint inv = max_sample - shape;
        r = (cr * shape + r * inv) / max_sample;
        g = (cg * shape + g * inv) / max_sample;
        b = (cb * shape + b * inv) / max_sample;
    }
}
