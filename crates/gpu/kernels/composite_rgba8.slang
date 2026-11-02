// Porter-Duff source-over compositing for RGBA8 pixels.
// Slang port of composite_rgba8.cu — semantics must match byte-for-byte.
//
// Each thread composites one pixel, read and written as one little-endian
// `uint` (R in bits 0-7, A in bits 24-31).  No subgroup ops, no shared memory.
//
// With a_src in [1, 254], a_out = a_src + round(a_dst * (255 - a_src) / 255)
// lies in [1, 255], so the division by a_out is defined and a_out fits a byte.

uint channel(uint pixel, uint c) { return (pixel >> (8 * c)) & 0xFFu; }

[shader("compute")]
[numthreads(256, 1, 1)]
void composite_rgba8(
    uint3 tid : SV_DispatchThreadID,
    uniform StructuredBuffer<uint> src_pixels,
    uniform RWStructuredBuffer<uint> dst_pixels,
    uniform uint n_pixels)
{
    uint i = tid.x;
    if (i >= n_pixels) return;

    uint s = src_pixels[i];
    uint a_src = channel(s, 3);
    if (a_src == 0) return;
    if (a_src == 255) {
        dst_pixels[i] = s;
        return;
    }

    uint d = dst_pixels[i];
    uint a_dst = channel(d, 3);
    uint inv   = 255 - a_src;
    uint a_out = a_src + ((a_dst * inv + 127) / 255);

    uint result = a_out << 24;
    [unroll]
    for (uint c = 0; c < 3; c++) {
        // Floor truncation in the d*a_dst*inv/255 term can push the quotient
        // past 255 (max 261); clamp before packing.
        uint v = (channel(s, c) * a_src + channel(d, c) * a_dst * inv / 255 + a_out / 2) / a_out;
        result |= min(v, 255u) << (8 * c);
    }
    dst_pixels[i] = result;
}
