// Porter-Duff source-over compositing for RGBA8 pixels.
// src_pixels, dst_pixels: RGBA interleaved, 4-byte aligned (one uchar4 per pixel).
// n_pixels: total pixel count
//
// With a_src in [1, 254], a_out = a_src + round(a_dst * (255 - a_src) / 255)
// lies in [1, 255], so the division by a_out is defined and a_out fits a byte.

__device__ __forceinline__ unsigned char blend_channel(
    unsigned int s, unsigned int d,
    unsigned int a_src, unsigned int a_dst, unsigned int inv, unsigned int a_out
) {
    // Floor truncation in the d*a_dst*inv/255 term can push the quotient
    // past 255 (max 261); clamp before the narrowing cast.
    unsigned int v = (s * a_src + d * a_dst * inv / 255 + a_out / 2) / a_out;
    return (unsigned char)min(v, 255u);
}

extern "C" __global__ void composite_rgba8(
    const unsigned char* __restrict__ src_pixels,
    unsigned char* __restrict__ dst_pixels,
    unsigned int n_pixels
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pixels) return;
    const uchar4 s = reinterpret_cast<const uchar4*>(src_pixels)[i];
    if (s.w == 0) return;
    uchar4* dst = reinterpret_cast<uchar4*>(dst_pixels);
    if (s.w == 255) {
        dst[i] = s;
        return;
    }
    const uchar4 d = dst[i];
    unsigned int a_src = s.w;
    unsigned int a_dst = d.w;
    unsigned int inv   = 255 - a_src;
    unsigned int a_out = a_src + ((a_dst * inv + 127) / 255);
    dst[i] = make_uchar4(
        blend_channel(s.x, d.x, a_src, a_dst, inv, a_out),
        blend_channel(s.y, d.y, a_src, a_dst, inv, a_out),
        blend_channel(s.z, d.z, a_src, a_dst, inv, a_out),
        (unsigned char)a_out);
}
