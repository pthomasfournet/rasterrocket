//! End-to-end render parity for GPU fills: a page rendered with the fill
//! kernel active (`--backend cuda` under `gpu-aa`, or `--backend vulkan`)
//! must be byte-identical to the CPU render of the same page.
//!
//! The device evaluates the CPU rasteriser's scanline intersection table and
//! the counts are composited through the CPU pipeline, so the final page
//! bitmap is the only meaningful check.
//!
//! The device fill path is dormant by default (`GPU_AA_FILL_THRESHOLD` is
//! `usize::MAX`), so these tests require `PDF_RASTER_AA_FILL_THRESHOLD=0`
//! and fail without it rather than compare the CPU render with itself.
//!
//! Run with:
//!   `PDF_RASTER_AA_FILL_THRESHOLD=0 cargo test -p rasterrocket --features gpu-aa,gpu-validation --test gpu_aa_render_parity`
//!   `PDF_RASTER_AA_FILL_THRESHOLD=0 cargo test -p rasterrocket --features vulkan --test gpu_aa_render_parity`

#![cfg(any(
    all(feature = "gpu-aa", feature = "gpu-validation"),
    feature = "vulkan"
))]

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use common::{assert_identical, corpus_fixture, render, session, stream_object, with_page_tree};
use rasterrocket::{BackendPolicy, SessionConfig, open_session};

/// A 400×400 pt page whose fills cover every dispatch decision:
///
/// - a thin rectangle with fractional edges (10 × 160 px at 150 DPI, above
///   the dispatch threshold) — the shape the divergence was first seen on;
/// - a slanted triangle and a Bézier circle (sloped and curved edges);
/// - an even-odd donut;
/// - a fill at constant alpha 0.5 (the general compositing pipe);
/// - a fill under a rectangular clip and one under a diamond path clip
///   (the latter must stay on the CPU);
/// - fills running off the page's left/bottom and right/top edges.
fn fixture_pdf() -> Vec<u8> {
    let content = b"\
0.47843 0.45882 0.44706 rg 100.31 50.27 4.76 76.8 re f\n\
0 0 1 rg 30.3 300.7 m 190.9 330.2 l 60.1 380.4 l h f\n\
1 0 0 rg 291.27 250.61 m 291.27 273.20 272.96 291.51 250.37 291.51 c \
227.78 291.51 209.47 273.20 209.47 250.61 c \
209.47 228.02 227.78 209.71 250.37 209.71 c \
272.96 209.71 291.27 228.02 291.27 250.61 c h f\n\
0 0.5 0 rg 300.5 40.5 80 80 re 320.7 60.3 40 40 re f*\n\
q /GS1 gs 0.6 0.2 0.8 rg 120.4 120.6 90.3 60.7 re f Q\n\
q 20 200 100 50 re W n 0.2 0.2 0.2 rg 0 180 150 100 re f Q\n\
q 200 150 m 260 200 l 200 250 l 140 200 l h W n 0.5 0 0.5 rg 120 140 160 120 re f Q\n\
0.3 0.6 0.9 rg -20.5 -10.3 60 40 re f\n\
0.9 0.5 0.1 rg 370.2 380.7 60 60 re f\n"
        .to_vec();

    let objects = vec![
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 400] \
           /Resources << /ExtGState << /GS1 << /ca 0.5 >> >> >> \
           /Contents 4 0 R >>"
            .to_vec(),
        stream_object("", &content),
    ];
    with_page_tree(&objects)
}

/// Panic unless the device fill path is enabled for this process.
fn require_device_fills() {
    assert_eq!(
        std::env::var("PDF_RASTER_AA_FILL_THRESHOLD").as_deref(),
        Ok("0"),
        "set PDF_RASTER_AA_FILL_THRESHOLD=0: the GPU fill path is dormant by default, \
         and without the override both renders run on the CPU"
    );
}

/// Render the synthetic page under `policy` and against the CPU.
fn assert_synthetic_page_identical(policy: BackendPolicy) {
    require_device_fills();
    let pdf = fixture_pdf();
    let cpu = render(&session(pdf.clone(), BackendPolicy::CpuOnly), 1);
    assert!(
        cpu.2.iter().any(|&b| b != 0xff),
        "fixture rendered blank on the CPU path"
    );
    let gpu = render(&session(pdf, policy), 1);
    assert_identical(&cpu, &gpu, &format!("synthetic fills under {policy:?}"));
}

/// Render `pages` of a corpus fixture under `policy` and against the CPU.
/// Skipped when the fixture is absent.
fn assert_corpus_pages_identical(fixture: &str, pages: &[u32], policy: BackendPolicy) {
    require_device_fills();
    let Some(path) = corpus_fixture(fixture) else {
        return;
    };
    let cpu_session =
        open_session(&path, &SessionConfig::with_policy(BackendPolicy::CpuOnly)).expect("CPU");
    let gpu_session = open_session(&path, &SessionConfig::with_policy(policy))
        .unwrap_or_else(|e| panic!("{policy:?} session: {e}"));
    for &page in pages {
        let cpu = render(&cpu_session, page);
        let gpu = render(&gpu_session, page);
        assert_identical(
            &cpu,
            &gpu,
            &format!("{fixture} page {page} under {policy:?}"),
        );
    }
}

/// The corpus pages the divergence was measured on: corpus-08 page 2 carries
/// the 10 × 160 px rectangle whose partial-coverage edge differed; corpus-02
/// is vector-heavy; corpus-07 pages are full-page scans with vector overlays.
fn assert_corpus_identical(policy: BackendPolicy) {
    assert_corpus_pages_identical("corpus-08-scan-dct-1927.pdf", &[1, 2, 3], policy);
    assert_corpus_pages_identical(
        "corpus-02-native-vector-text.pdf",
        &(1..=16).collect::<Vec<_>>(),
        policy,
    );
    assert_corpus_pages_identical("corpus-07-journal-dct-heavy.pdf", &[1, 2], policy);
}

#[cfg(all(feature = "gpu-aa", feature = "gpu-validation"))]
mod cuda {
    use super::{BackendPolicy, assert_corpus_identical, assert_synthetic_page_identical};

    #[test]
    fn synthetic_fills_render_identically_on_cuda() {
        assert_synthetic_page_identical(BackendPolicy::ForceCuda);
    }

    #[test]
    fn corpus_pages_render_identically_on_cuda() {
        assert_corpus_identical(BackendPolicy::ForceCuda);
    }
}

#[cfg(feature = "vulkan")]
mod vulkan {
    use super::{BackendPolicy, assert_corpus_identical, assert_synthetic_page_identical};

    #[test]
    fn synthetic_fills_render_identically_on_vulkan() {
        assert_synthetic_page_identical(BackendPolicy::ForceVulkan);
    }

    #[test]
    fn corpus_pages_render_identically_on_vulkan() {
        assert_corpus_identical(BackendPolicy::ForceVulkan);
    }
}
