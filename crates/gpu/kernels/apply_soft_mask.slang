// Multiply each RGBA pixel's alpha by the corresponding soft-mask byte.
// Slang port of apply_soft_mask.cu — semantics must match byte-for-byte.
// Each pixel is one little-endian `uint` (A in bits 24-31).

[shader("compute")]
[numthreads(256, 1, 1)]
void apply_soft_mask(
    uint3 tid : SV_DispatchThreadID,
    uniform RWStructuredBuffer<uint> pixels,
    uniform StructuredBuffer<uint8_t> mask,
    uniform uint n_pixels)
{
    uint i = tid.x;
    if (i >= n_pixels) return;
    uint p = pixels[i];
    uint a = p >> 24;
    uint m = uint(mask[i]);
    pixels[i] = (p & 0x00FFFFFFu) | (((a * m + 127) / 255) << 24);
}
