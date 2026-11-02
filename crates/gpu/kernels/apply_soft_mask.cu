// Multiply each RGBA pixel's alpha by the corresponding soft-mask byte.
// pixels: RGBA interleaved, 4-byte aligned (one uchar4 per pixel)
// mask: one byte per pixel [0..255]
// n_pixels: total pixel count
extern "C" __global__ void apply_soft_mask(
    unsigned char* __restrict__ pixels,
    const unsigned char* __restrict__ mask,
    unsigned int n_pixels
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pixels) return;
    uchar4* px = reinterpret_cast<uchar4*>(pixels);
    uchar4 p = px[i];
    p.w = (unsigned char)((p.w * (unsigned int)mask[i] + 127) / 255);
    px[i] = p;
}
