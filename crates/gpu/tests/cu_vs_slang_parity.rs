//! Parity tests: same kernel via CUDA (`.cu`-via-`nvcc`) and Vulkan
//! (`.slang`-via-`slangc`).
//!
//! Each test composes a small representative input, runs it through
//! the Vulkan kernel and a reference (the CUDA kernel or the host evaluator),
//! and asserts byte-equal output.
//!
//! Gated on the `vulkan` feature.  CUDA-side tests are additionally
//! gated on `gpu-validation` so machines without a CUDA device still
//! run the Vulkan side and validate against a CPU reference.
//!
//! Run with:
//!   `cargo test -p rasterrocket-gpu --features vulkan --test cu_vs_slang_parity`
//!   `cargo test -p rasterrocket-gpu --features 'vulkan,gpu-validation' --test cu_vs_slang_parity`

#![cfg(feature = "vulkan")]
#![expect(
    clippy::unreadable_literal,
    reason = "test fixture: the LCG's u32 constants"
)]

mod common;
#[path = "common/blit.rs"]
mod common_blit;

// ── icc_cmyk_clut ───────────────────────────────────────────────────

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

/// The 16 corners of the CMYK cube and pseudo-random pixels through a
/// pseudo-random grid of uneven spans in either range: the Vulkan kernel
/// interpolates in integers, so it matches the host byte for byte, and
/// passes the backend's exactness probe.
#[test]
fn icc_clut_matches_the_host() {
    use color::grid::NodeRange;
    let backend = crate::common::vulkan_backend().expect("VulkanBackend::new");
    assert!(backend.icc_clut_is_exact(), "the exactness probe fails");
    let mut cmyk: Vec<u8> = (0..16u8)
        .flat_map(|i| [3, 2, 1, 0].map(|b| if i >> b & 1 == 1 { 255 } else { 0 }))
        .collect();
    cmyk.extend(pseudo_random(65_536 * 4, 0x1234_5678));
    for range in [NodeRange::Unit, NodeRange::Extended] {
        let grid = rasterrocket_gpu::backend::random_grid(0x0dd_ba11, [16, 9, 20, 5], range);
        let vk = rasterrocket_gpu::backend::icc_clut(&backend, &cmyk, &grid).expect("icc_clut");
        let differing = grid
            .convert(&cmyk)
            .iter()
            .zip(&vk)
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(differing, 0, "{range:?}: {differing} bytes differ");
    }
}

// ── blit_image ──────────────────────────────────────────────────────

#[test]
fn blit_identity_rgb() {
    common_blit::identity_rgb(&crate::common::vulkan_backend().expect("VulkanBackend::new"));
}

#[test]
fn blit_scaled_offset_with_carry() {
    common_blit::scaled_offset_with_carry(
        &crate::common::vulkan_backend().expect("VulkanBackend::new"),
    );
}

#[test]
fn blit_gray_layout() {
    common_blit::gray_layout(&crate::common::vulkan_backend().expect("VulkanBackend::new"));
}

#[test]
fn blit_reads_an_unpadded_tail_source_exactly() {
    common_blit::unpadded_tail_source(
        &crate::common::vulkan_backend().expect("VulkanBackend::new"),
    );
}
