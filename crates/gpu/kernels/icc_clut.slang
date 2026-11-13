// GPU ICC colour transform kernels — Slang port of icc_clut.cu.
//
// Entry point:
//
//   icc_cmyk_clut — full 4D CLUT path for A2B0 profiles.
//     Quadrilinear interpolation over a grid_n^4 CLUT, 16 corner lookups per pixel.
//
// The profile-free conversion (the CUDA `icc_cmyk_matrix` entry) always runs
// on the CPU and has no Slang counterpart.
//
// Each lerp is `fma(w, b - a, a)`, the CUDA kernel's `__fmaf_rn` and the
// CPU fallback's `f32::mul_add`: the product and the sum round once.  The
// lattice coordinates and weights are `precise`, so no contraction changes
// them.  Vulkan does not require `fma` to round once, so the host checks
// the kernel against the CPU fallback on each device before using it
// (`VulkanBackend::icc_clut_is_exact`).
// CLUT access uses StructuredBuffer<uint8_t> (flat buffer), the layout of
// the CUDA path.

// ── CLUT entry point ────────────────────────────────────────────────────────

// Fetch one CLUT node.  Layout (matches Rust baker in
// crates/pdf_interp/src/resources/icc.rs):
//   index = (k_idx * G^3 + c_idx * G^2 + m_idx * G + y_idx) * 3
// K is the outermost (major) axis.
//
// Clamping is mandatory, not soft: FP rounding on normalised inputs at
// extreme channel values can push floor indices to G.
void clut_node(
    StructuredBuffer<uint8_t> clut,
    int ci, int mi, int yi, int ki,
    int G,
    out float r, out float g, out float b)
{
    ci = max(0, min(ci, G - 1));
    mi = max(0, min(mi, G - 1));
    yi = max(0, min(yi, G - 1));
    ki = max(0, min(ki, G - 1));

    int idx = (ki * G * G * G + ci * G * G + mi * G + yi) * 3;
    r = float(uint(clut[idx + 0]));
    g = float(uint(clut[idx + 1]));
    b = float(uint(clut[idx + 2]));
}

// Round-half-up of a value clamped to [0, 255], matching the CPU
// fallback's clamp-then-round (half away from zero equals half up on a
// non-negative range). Computed via the floor difference: adding 0.5f
// first can round up to the next integer for inputs just below a half,
// because the sum's f32 ulp is coarser than the input's. `c - fl` is
// exact on this range (Sterbenz), so the comparison is exact. Written
// out explicitly rather than via round(): halfway direction of
// GLSL.std.450 Round is implementation-defined.
uint8_t round_half_up_byte(float v) {
    float c  = min(max(v, 0.0f), 255.0f);
    float fl = floor(c);
    return uint8_t(int(fl + ((c - fl) >= 0.5f ? 1.0f : 0.0f)));
}

// `a + w·(b − a)` with the product and the sum rounded once.
float lerp_once(float a, float b, float w) {
    precise float d = b - a;
    return fma(w, d, a);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void icc_cmyk_clut(
    uint3 tid : SV_DispatchThreadID,
    uniform StructuredBuffer<uint8_t> cmyk,
    uniform RWStructuredBuffer<uint8_t> rgb,
    uniform StructuredBuffer<uint8_t> clut,
    uniform uint grid_n,
    uniform uint n)
{
    uint i = tid.x;
    if (i >= n) return;

    uint ci_base = i * 4;
    uint ri_base = i * 3;

    float G1 = float(grid_n - 1u);
    precise float fc = float(uint(cmyk[ci_base + 0])) * (G1 / 255.0f);
    precise float fm = float(uint(cmyk[ci_base + 1])) * (G1 / 255.0f);
    precise float fy = float(uint(cmyk[ci_base + 2])) * (G1 / 255.0f);
    precise float fk = float(uint(cmyk[ci_base + 3])) * (G1 / 255.0f);

    int G = int(grid_n);

    int ic0 = int(fc);
    int im0 = int(fm);
    int iy0 = int(fy);
    int ik0 = int(fk);

    int ic1 = min(ic0 + 1, G - 1);
    int im1 = min(im0 + 1, G - 1);
    int iy1 = min(iy0 + 1, G - 1);
    int ik1 = min(ik0 + 1, G - 1);

    precise float wc = fc - float(ic0);
    precise float wm = fm - float(im0);
    precise float wy = fy - float(iy0);
    precise float wk = fk - float(ik0);

    // Quadrilinear: 16 corner lookups, lerp along Y → M → C → K.

    // K = 0 plane
    float rc000, gc000, bc000;  clut_node(clut, ic0, im0, iy0, ik0, G, rc000, gc000, bc000);
    float rc001, gc001, bc001;  clut_node(clut, ic0, im0, iy1, ik0, G, rc001, gc001, bc001);
    float rc010, gc010, bc010;  clut_node(clut, ic0, im1, iy0, ik0, G, rc010, gc010, bc010);
    float rc011, gc011, bc011;  clut_node(clut, ic0, im1, iy1, ik0, G, rc011, gc011, bc011);
    float rc100, gc100, bc100;  clut_node(clut, ic1, im0, iy0, ik0, G, rc100, gc100, bc100);
    float rc101, gc101, bc101;  clut_node(clut, ic1, im0, iy1, ik0, G, rc101, gc101, bc101);
    float rc110, gc110, bc110;  clut_node(clut, ic1, im1, iy0, ik0, G, rc110, gc110, bc110);
    float rc111, gc111, bc111;  clut_node(clut, ic1, im1, iy1, ik0, G, rc111, gc111, bc111);

    float r_c0m0 = lerp_once(rc000, rc001, wy);
    float r_c0m1 = lerp_once(rc010, rc011, wy);
    float r_c1m0 = lerp_once(rc100, rc101, wy);
    float r_c1m1 = lerp_once(rc110, rc111, wy);
    float g_c0m0 = lerp_once(gc000, gc001, wy);
    float g_c0m1 = lerp_once(gc010, gc011, wy);
    float g_c1m0 = lerp_once(gc100, gc101, wy);
    float g_c1m1 = lerp_once(gc110, gc111, wy);
    float b_c0m0 = lerp_once(bc000, bc001, wy);
    float b_c0m1 = lerp_once(bc010, bc011, wy);
    float b_c1m0 = lerp_once(bc100, bc101, wy);
    float b_c1m1 = lerp_once(bc110, bc111, wy);

    float r00 = lerp_once(r_c0m0, r_c0m1, wm);
    float r10 = lerp_once(r_c1m0, r_c1m1, wm);
    float g00 = lerp_once(g_c0m0, g_c0m1, wm);
    float g10 = lerp_once(g_c1m0, g_c1m1, wm);
    float b00 = lerp_once(b_c0m0, b_c0m1, wm);
    float b10 = lerp_once(b_c1m0, b_c1m1, wm);

    float rk0 = lerp_once(r00, r10, wc);
    float gk0 = lerp_once(g00, g10, wc);
    float bk0 = lerp_once(b00, b10, wc);

    // K = 1 plane
    float rc000k1, gc000k1, bc000k1;  clut_node(clut, ic0, im0, iy0, ik1, G, rc000k1, gc000k1, bc000k1);
    float rc001k1, gc001k1, bc001k1;  clut_node(clut, ic0, im0, iy1, ik1, G, rc001k1, gc001k1, bc001k1);
    float rc010k1, gc010k1, bc010k1;  clut_node(clut, ic0, im1, iy0, ik1, G, rc010k1, gc010k1, bc010k1);
    float rc011k1, gc011k1, bc011k1;  clut_node(clut, ic0, im1, iy1, ik1, G, rc011k1, gc011k1, bc011k1);
    float rc100k1, gc100k1, bc100k1;  clut_node(clut, ic1, im0, iy0, ik1, G, rc100k1, gc100k1, bc100k1);
    float rc101k1, gc101k1, bc101k1;  clut_node(clut, ic1, im0, iy1, ik1, G, rc101k1, gc101k1, bc101k1);
    float rc110k1, gc110k1, bc110k1;  clut_node(clut, ic1, im1, iy0, ik1, G, rc110k1, gc110k1, bc110k1);
    float rc111k1, gc111k1, bc111k1;  clut_node(clut, ic1, im1, iy1, ik1, G, rc111k1, gc111k1, bc111k1);

    float r_c0m0k1 = lerp_once(rc000k1, rc001k1, wy);
    float r_c0m1k1 = lerp_once(rc010k1, rc011k1, wy);
    float r_c1m0k1 = lerp_once(rc100k1, rc101k1, wy);
    float r_c1m1k1 = lerp_once(rc110k1, rc111k1, wy);
    float g_c0m0k1 = lerp_once(gc000k1, gc001k1, wy);
    float g_c0m1k1 = lerp_once(gc010k1, gc011k1, wy);
    float g_c1m0k1 = lerp_once(gc100k1, gc101k1, wy);
    float g_c1m1k1 = lerp_once(gc110k1, gc111k1, wy);
    float b_c0m0k1 = lerp_once(bc000k1, bc001k1, wy);
    float b_c0m1k1 = lerp_once(bc010k1, bc011k1, wy);
    float b_c1m0k1 = lerp_once(bc100k1, bc101k1, wy);
    float b_c1m1k1 = lerp_once(bc110k1, bc111k1, wy);

    float r01 = lerp_once(r_c0m0k1, r_c0m1k1, wm);
    float r11 = lerp_once(r_c1m0k1, r_c1m1k1, wm);
    float g01 = lerp_once(g_c0m0k1, g_c0m1k1, wm);
    float g11 = lerp_once(g_c1m0k1, g_c1m1k1, wm);
    float b01 = lerp_once(b_c0m0k1, b_c0m1k1, wm);
    float b11 = lerp_once(b_c1m0k1, b_c1m1k1, wm);

    float rk1 = lerp_once(r01, r11, wc);
    float gk1 = lerp_once(g01, g11, wc);
    float bk1 = lerp_once(b01, b11, wc);

    // Final lerp along K.
    float out_r = lerp_once(rk0, rk1, wk);
    float out_g = lerp_once(gk0, gk1, wk);
    float out_b = lerp_once(bk0, bk1, wk);

    rgb[ri_base + 0] = round_half_up_byte(out_r);
    rgb[ri_base + 1] = round_half_up_byte(out_g);
    rgb[ri_base + 2] = round_half_up_byte(out_b);
}
