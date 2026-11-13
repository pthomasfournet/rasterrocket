//! Parity tests: same kernel via CUDA (`.cu`-via-`nvcc`) and Vulkan
//! (`.slang`-via-`slangc`).
//!
//! Each test composes a small representative input, runs it through
//! the Vulkan kernel and a reference (the CUDA kernel or its CPU twin),
//! and asserts byte-equal output.
//!
//! Gated on the `vulkan` feature.  CUDA-side tests are additionally
//! gated on `gpu-validation` so machines without a CUDA device still
//! run the Vulkan side and validate against a CPU reference.
//!
//! Run with:
//!   `cargo test -p rasterrocket-gpu --features vulkan --test cu_vs_slang_parity`
//!   `cargo test -p rasterrocket-gpu --features 'vulkan,gpu-validation' --test cu_vs_slang_parity`
//!
//! This file has no `aa_fill` or parallel-Huffman cases. The fill kernel's
//! CUDA / Vulkan / host-twin parity tests live in `aa_fill_parity.rs`; the
//! Huffman ones live in `huffman::vulkan_tests`, gated on
//! `gpu-jpeg-huffman` in addition to `vulkan`:
//!   `cargo test -p rasterrocket-gpu --features 'vulkan,gpu-validation,gpu-jpeg-huffman' --lib -- vulkan_tests`

#![cfg(feature = "vulkan")]
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::too_many_arguments,
    clippy::unreadable_literal,
    reason = "test fixture: u32 LCG constants, bounded `i & 0xff` masks, 8-10 arg \
              CPU-reference helpers, and i32/u32 index casts intentionally mirror the \
              kernel-side math byte-for-byte"
)]

use rasterrocket_gpu::backend::GpuBackend;
use rasterrocket_gpu::backend::params::{BLIT_FILTER_X, BLIT_FILTER_Y, BlitParams};
use rasterrocket_gpu::backend::vulkan::VulkanBackend;

// ── icc_cmyk_clut ───────────────────────────────────────────────────

/// Build a synthetic identity CLUT for `grid_n` so the parity test has
/// a deterministic, hand-derivable input.  The kernel quadrilinear-
/// interpolates over the table; with values C and M and Y stored as the
/// node indices' RGB encoding, the result is a smooth ramp we can
/// compare against the CPU path's output byte-for-byte.
fn build_identity_clut(grid_n: u32) -> Vec<u8> {
    let g = grid_n as usize;
    let mut clut = vec![0u8; g * g * g * g * 3];
    // Layout: index = (k * G^3 + c * G^2 + m * G + y) * 3 (matches Rust baker
    // and CUDA kernel).
    for k in 0..g {
        for c in 0..g {
            for m in 0..g {
                for y in 0..g {
                    let idx = (k * g * g * g + c * g * g + m * g + y) * 3;
                    // R encodes C, G encodes M, B encodes Y; K modulates by 1-K/G.
                    let scale = (g - 1 - k) as u32 * 255 / (g - 1).max(1) as u32;
                    let r = (c as u32 * 255 / (g - 1).max(1) as u32) * scale / 255;
                    let gv = (m as u32 * 255 / (g - 1).max(1) as u32) * scale / 255;
                    let b = (y as u32 * 255 / (g - 1).max(1) as u32) * scale / 255;
                    clut[idx] = r as u8;
                    clut[idx + 1] = gv as u8;
                    clut[idx + 2] = b as u8;
                }
            }
        }
    }
    clut
}

fn run_icc_clut_vulkan(cmyk: &[u8], clut: &[u8]) -> Vec<u8> {
    let backend = VulkanBackend::new().expect("VulkanBackend::new");
    rasterrocket_gpu::backend::icc_clut(&backend, cmyk, clut).expect("icc_clut")
}

fn run_icc_clut_cpu(cmyk: &[u8], clut: &[u8], grid_n: u32) -> Vec<u8> {
    rasterrocket_gpu::icc_cmyk_to_rgb_cpu(cmyk, Some((clut, grid_n)))
}

/// `n` pseudo-random bytes from `seed`.
fn pseudo_random(n: usize, seed: u32) -> Vec<u8> {
    let mut rng = seed;
    (0..n)
        .map(|_| {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            (rng >> 24) as u8
        })
        .collect()
}

#[test]
fn icc_clut_corners_only() {
    // Pure C, M, Y, K corners — sample the 16 hypercube corners only.
    let grid_n = 17u32;
    let clut = build_identity_clut(grid_n);
    let mut cmyk = Vec::new();
    for c in [0u8, 255] {
        for m in [0u8, 255] {
            for y in [0u8, 255] {
                for k in [0u8, 255] {
                    cmyk.extend_from_slice(&[c, m, y, k]);
                }
            }
        }
    }
    let cpu = run_icc_clut_cpu(&cmyk, &clut, grid_n);
    let vk = run_icc_clut_vulkan(&cmyk, &clut);
    assert_eq!(cpu, vk, "icc_clut corners");
}

/// Pseudo-random pixels through a pseudo-random table, whose nodes differ
/// arbitrarily from their neighbours: every interpolation step carries a
/// fractional weight and a non-trivial difference, where rounding the
/// product and the sum separately would move some output byte.
///
/// A device that passes the backend's exactness check must match the CPU
/// byte for byte.  One that fails it is never used for the conversion; its
/// outputs must still lie within one level of the CPU's.
#[test]
fn icc_clut_random_inputs() {
    let grid_n = 17u32;
    let clut = pseudo_random((grid_n as usize).pow(4) * 3, 0x0dd_ba11);
    let cmyk = pseudo_random(65_536 * 4, 0x1234_5678);
    let backend = VulkanBackend::new().expect("VulkanBackend::new");
    let cpu = run_icc_clut_cpu(&cmyk, &clut, grid_n);
    let vk = rasterrocket_gpu::backend::icc_clut(&backend, &cmyk, &clut).expect("icc_clut");
    let differing = cpu.iter().zip(&vk).filter(|(a, b)| a != b).count();
    let max_delta = cpu.iter().zip(&vk).map(|(a, b)| a.abs_diff(*b)).max();
    if backend.icc_clut_is_exact() {
        assert_eq!(differing, 0, "icc_clut random: {differing} bytes differ");
    } else {
        eprintln!(
            "device fails the exactness check: {differing} bytes differ (max {max_delta:?}); \
             the renderer converts on the CPU"
        );
        assert!(
            max_delta <= Some(1),
            "icc_clut random: max delta {max_delta:?}"
        );
    }
}

// ── blit_image ──────────────────────────────────────────────────────

/// Q32 split of one sampling-table term: `(hi as u32, lo)` with
/// `hi = floor(v)`.
#[expect(
    clippy::cast_sign_loss,
    reason = "`v - hi` lies in [0, 1), so the scaled fraction is non-negative"
)]
fn split_q32(v: f64) -> (u32, u32) {
    let hi = v.floor();
    (
        (hi as i32).cast_unsigned(),
        ((v - hi) * 4_294_967_296.0) as u32,
    )
}

/// Sampling tables for the separable source map
/// `X = x_col(dx) + x_row(dy)`, `Y = y_col(dx) + y_row(dy)`, evaluated at
/// pixel centres over `bbox` (`[x0, y0, x1, y1]`).  Same layout the
/// kernel reads: four `u32` words per entry, `[x_hi, x_lo, y_hi, y_lo]`.
fn blit_tables(
    bbox: [i32; 4],
    col_terms: impl Fn(f64) -> (f64, f64),
    row_terms: impl Fn(f64) -> (f64, f64),
) -> (Vec<u32>, Vec<u32>) {
    let entry = |(x, y): (f64, f64)| {
        let (xh, xl) = split_q32(x);
        let (yh, yl) = split_q32(y);
        [xh, xl, yh, yl]
    };
    let cols = (bbox[0]..bbox[2])
        .flat_map(|dx| entry(col_terms(f64::from(dx) + 0.5)))
        .collect();
    let rows = (bbox[1]..bbox[3])
        .flat_map(|dy| entry(row_terms(f64::from(dy) + 0.5)))
        .collect();
    (cols, rows)
}

/// Taps `(lo, hi, weight of hi)` along one axis of `n` samples for the
/// in-range integer part `i` and Q32 fraction `frac` of a source
/// coordinate: the sample `i` alone, or, when `blend` is set, the samples
/// whose centres bracket the coordinate, clamped onto the axis, weighted by
/// the top 8 bits of the coordinate's fraction measured from sample centres.
fn blit_axis_taps(i: i32, frac: u32, n: u32, blend: bool) -> (u32, u32, u32) {
    if !blend {
        return (i.cast_unsigned(), i.cast_unsigned(), 0);
    }
    let q = (i64::from(i) << 32) + i64::from(frac) - (1 << 31);
    let lower = q >> 32;
    let index = |v: i64| u32::try_from(v).expect("tap clamped onto the axis");
    (
        index(lower.max(0)),
        index((lower + 1).min(i64::from(n) - 1)),
        index((q >> 24) & 0xFF),
    )
}

/// Replicate the kernel's table walk on the CPU (mirrors the test in
/// `crates/gpu/src/blit.rs`): sum column and row terms with carry, reject
/// out-of-image samples, blend the taps `filter` selects and write the
/// pixel with alpha 255.
fn blit_cpu_reference(
    page: &mut [u8],
    page_w: u32,
    page_h: u32,
    src: &[u8],
    src_w: u32,
    src_h: u32,
    src_layout: u32, // 0 = RGB, 1 = Gray
    bbox: [i32; 4],
    cols: &[u32],
    rows: &[u32],
    filter: u32,
) {
    let bpp = if src_layout == 0 { 3 } else { 1 };
    // Table index is the position within the bbox; page bounds are the
    // kernel's own reject test.
    for (r_idx, dy) in (bbox[1]..bbox[3]).enumerate() {
        let Ok(dy) = u32::try_from(dy) else {
            continue;
        };
        if dy >= page_h {
            continue;
        }
        for (c_idx, dx) in (bbox[0]..bbox[2]).enumerate() {
            let Ok(dx) = u32::try_from(dx) else {
                continue;
            };
            if dx >= page_w {
                continue;
            }
            let c = &cols[c_idx * 4..][..4];
            let r = &rows[r_idx * 4..][..4];
            let (frac_x, carry_x) = c[1].overflowing_add(r[1]);
            let ix = c[0]
                .cast_signed()
                .wrapping_add(r[0].cast_signed())
                .wrapping_add(i32::from(carry_x));
            let (frac_y, carry_y) = c[3].overflowing_add(r[3]);
            let iy = c[2]
                .cast_signed()
                .wrapping_add(r[2].cast_signed())
                .wrapping_add(i32::from(carry_y));
            let (Ok(ux), Ok(uy)) = (u32::try_from(ix), u32::try_from(iy)) else {
                continue;
            };
            if ux >= src_w || uy >= src_h {
                continue;
            }
            let (x0, x1, wx) = blit_axis_taps(ix, frac_x, src_w, filter & BLIT_FILTER_X != 0);
            let (y0, y1, wy) = blit_axis_taps(iy, frac_y, src_h, filter & BLIT_FILTER_Y != 0);
            let at =
                |x: u32, y: u32, ch: usize| u32::from(src[(y * src_w + x) as usize * bpp + ch]);
            let blend = |ch: usize| {
                let row = |y| at(x0, y, ch) * (256 - wx) + at(x1, y, ch) * wx;
                ((row(y0) * (256 - wy) + row(y1) * wy + (1 << 15)) >> 16) as u8
            };
            let rgb = if src_layout == 0 {
                [blend(0), blend(1), blend(2)]
            } else {
                [blend(0); 3]
            };
            let dst_off = (dy * page_w + dx) as usize * 4;
            page[dst_off..dst_off + 3].copy_from_slice(&rgb);
            page[dst_off + 3] = 255;
        }
    }
}

fn run_blit_vulkan(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    src_layout: u32,
    page_w: u32,
    page_h: u32,
    bbox: [i32; 4],
    cols: &[u32],
    rows: &[u32],
    filter: u32,
) -> Vec<u8> {
    let backend = VulkanBackend::new().expect("VulkanBackend::new");
    let dst_bytes = (page_w * page_h * 4) as usize;
    let cols_bytes: &[u8] = bytemuck::cast_slice(cols);
    let rows_bytes: &[u8] = bytemuck::cast_slice(rows);
    let d_src = backend.alloc_device(src.len()).expect("alloc src");
    let d_dst = backend.alloc_device(dst_bytes).expect("alloc dst");
    let d_cols = backend.alloc_device(cols_bytes.len()).expect("alloc cols");
    let d_rows = backend.alloc_device(rows_bytes.len()).expect("alloc rows");
    backend.upload(&d_src, src).expect("upload src");
    let zeros = vec![0u8; dst_bytes];
    backend.upload(&d_dst, &zeros).expect("zero dst");
    backend.upload(&d_cols, cols_bytes).expect("upload cols");
    backend.upload(&d_rows, rows_bytes).expect("upload rows");

    backend.begin_page().expect("begin_page");
    backend
        .record_blit_image(BlitParams {
            src: &d_src,
            dst: &d_dst,
            cols: &d_cols,
            rows: &d_rows,
            src_w,
            src_h,
            src_layout,
            dst_w: page_w,
            dst_h: page_h,
            bbox,
            filter,
        })
        .expect("record_blit_image");
    let fence = backend.submit_page().expect("submit_page");
    backend.wait_page(fence).expect("wait_page");

    let mut out = vec![0u8; dst_bytes];
    backend.download(&d_dst, &mut out).expect("download");
    out
}

/// Run one blit through the CPU reference and Vulkan, nearest-neighbour
/// and then interpolated on both axes, asserting byte equality.
fn check_blit_parity(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    src_layout: u32,
    page: u32,
    col_terms: impl Fn(f64) -> (f64, f64),
    row_terms: impl Fn(f64) -> (f64, f64),
    what: &str,
) {
    let bbox = [0, 0, page as i32, page as i32];
    let (cols, rows) = blit_tables(bbox, col_terms, row_terms);
    for filter in [0, BLIT_FILTER_X | BLIT_FILTER_Y] {
        let mut cpu = vec![0u8; (page * page * 4) as usize];
        blit_cpu_reference(
            &mut cpu, page, page, src, src_w, src_h, src_layout, bbox, &cols, &rows, filter,
        );
        let vk = run_blit_vulkan(
            src, src_w, src_h, src_layout, page, page, bbox, &cols, &rows, filter,
        );
        assert_eq!(cpu, vk, "{what} (filter {filter}) diverges");
        assert!(
            cpu.as_chunks::<4>().0.iter().any(|px| px[3] == 255),
            "{what}: no pixel was written"
        );
    }
}

#[test]
fn blit_identity_rgb() {
    // Identity placement of a 4×4 RGB image into a 4×4 page buffer:
    // page pixel (dx, dy) reads source (dx, dy).
    let src_w = 4u32;
    let src_h = 4u32;
    let pixels: Vec<u8> = (0..src_w * src_h * 3)
        .map(|i| ((i * 7 + 3) % 251) as u8)
        .collect();
    check_blit_parity(
        &pixels,
        src_w,
        src_h,
        0,
        4,
        |px| (px, 0.0),
        |py| (0.0, py),
        "blit identity RGB",
    );
}

#[test]
#[expect(
    clippy::suboptimal_flops,
    reason = "test geometry: the source map is spelled as scale-then-offset for readability"
)]
fn blit_scaled_offset_with_carry() {
    // Fractional scale and offset in both axes with a y-flip: exercises
    // the fraction-word carry, negative integer parts, and rejection of
    // page pixels outside the image.
    let src_w = 4u32;
    let src_h = 4u32;
    let pixels: Vec<u8> = (0..src_w * src_h * 3)
        .map(|i| ((i * 11 + 5) % 253) as u8)
        .collect();
    check_blit_parity(
        &pixels,
        src_w,
        src_h,
        0,
        6,
        |px| ((px - 1.3) * 0.7, 0.0),
        |py| (0.0, 4.0 - (py - 0.6) * 0.9),
        "blit scaled/offset RGB",
    );
}

#[test]
fn blit_gray_layout() {
    let src_w = 4u32;
    let src_h = 4u32;
    let pixels: Vec<u8> = (0..src_w * src_h).map(|i| (i * 17) as u8).collect();
    check_blit_parity(
        &pixels,
        src_w,
        src_h,
        1,
        4,
        |px| (px, 0.0),
        |py| (0.0, py),
        "blit gray-layout",
    );
}
