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

use common::{
    assert_identical, corpus_fixture, render, render_diagnosed, session, stream_object,
    with_page_tree,
};
use rasterrocket::{BackendPolicy, SessionConfig, open_session, open_session_from_bytes};

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

/// A 400×400 pt page painted on the overprint canvas: a cyan backdrop, a
/// black fill overprinting it under nonzero overprint mode, a spot-colour
/// triangle and an RGB rule with fractional edges.
fn overprint_pdf() -> Vec<u8> {
    let content = b"\
1 0 0 0 k 50.3 50.3 300 300 re f\n\
/GS2 gs 0 0 0 1 k 100.4 100.6 120.3 60.7 re f\n\
/Spot cs 0.7 scn 30.3 300.7 m 190.9 330.2 l 60.1 380.4 l h f\n\
0.47843 0.45882 0.44706 rg 100.31 50.27 4.76 76.8 re f\n"
        .to_vec();
    let objects = vec![
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 400] \
           /Resources << /ExtGState << /GS2 << /OP true /op true /OPM 1 >> >> \
           /ColorSpace << /Spot [/Separation /Spot /DeviceCMYK \
           << /FunctionType 2 /Domain [0 1] /C0 [0 0 0 0] /C1 [0 0.91 0.76 0] /N 1 >>] >> >> \
           /Contents 4 0 R >>"
            .to_vec(),
        stream_object("", &content),
    ];
    with_page_tree(&objects)
}

/// Render the overprint page with overprint simulation on, under `policy`
/// and against the CPU.  Every fill reaches the device and none fails.
fn assert_overprint_page_identical(policy: BackendPolicy) {
    require_device_fills();
    let open = |policy| {
        let mut config = SessionConfig::with_policy(policy);
        config.overprint = true;
        open_session_from_bytes(overprint_pdf(), &config)
            .unwrap_or_else(|e| panic!("{policy:?} session: {e}"))
    };
    let cpu = render(&open(BackendPolicy::CpuOnly), 1);
    assert!(
        cpu.2.iter().any(|&b| b != 0xff),
        "overprint fixture rendered blank on the CPU path"
    );
    let (gpu, diag) = render_diagnosed(&open(policy), 1);
    assert_eq!(
        (diag.device_fills, diag.device_fill_failures),
        (OVERPRINT_DEVICE_FILLS, 0),
        "device fills painted and failed under {policy:?}"
    );
    assert_identical(&cpu, &gpu, &format!("overprint fills under {policy:?}"));
}

/// Fills of the overprint page: all four reach the device.
const OVERPRINT_DEVICE_FILLS: u32 = 4;

/// Panic unless the device fill path is enabled for this process.
fn require_device_fills() {
    assert_eq!(
        std::env::var("PDF_RASTER_AA_FILL_THRESHOLD").as_deref(),
        Ok("0"),
        "set PDF_RASTER_AA_FILL_THRESHOLD=0: the GPU fill path is dormant by default, \
         and without the override both renders run on the CPU"
    );
}

/// Fills of the synthetic page the device takes: every fill but the one
/// under the diamond path clip, which the device fill declines.
const SYNTHETIC_DEVICE_FILLS: u32 = 8;

/// Render the synthetic page under `policy` and against the CPU.  The
/// device must paint every fill it takes, and fail none: a device fill that
/// fails falls back to the CPU, and the comparison would then hold by
/// construction.
fn assert_synthetic_page_identical(policy: BackendPolicy) {
    require_device_fills();
    let pdf = fixture_pdf();
    let (cpu, cpu_diag) = render_diagnosed(&session(pdf.clone(), BackendPolicy::CpuOnly), 1);
    assert_eq!(
        (cpu_diag.device_fills, cpu_diag.device_fill_failures),
        (0, 0),
        "CPU-only session used a device"
    );
    assert!(
        cpu.2.iter().any(|&b| b != 0xff),
        "fixture rendered blank on the CPU path"
    );
    let (gpu, diag) = render_diagnosed(&session(pdf, policy), 1);
    assert_eq!(
        (diag.device_fills, diag.device_fill_failures),
        (SYNTHETIC_DEVICE_FILLS, 0),
        "device fills painted and failed under {policy:?}"
    );
    assert_identical(&cpu, &gpu, &format!("synthetic fills under {policy:?}"));
}

/// Render `pages` of a corpus fixture under `policy` and against the CPU.
/// No device fill may fail, and some page must be filled on the device.
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
    let mut device_fills = 0;
    for &page in pages {
        let what = format!("{fixture} page {page} under {policy:?}");
        let cpu = render(&cpu_session, page);
        let (gpu, diag) = render_diagnosed(&gpu_session, page);
        assert_eq!(diag.device_fill_failures, 0, "{what}: device fills failed");
        device_fills += diag.device_fills;
        assert_identical(&cpu, &gpu, &what);
    }
    assert!(
        device_fills > 0,
        "{fixture} under {policy:?}: no fill reached the device"
    );
}

/// Corpus pages whose fills reach the device: corpus-08 page 2 carries
/// the 10 × 160 px rectangle whose partial-coverage edge once differed;
/// corpus-05 (46 device fills on pages 1–4), corpus-04 (11) and corpus-11
/// (6 on its one page of CMYK swatches) cover rules, boxes and colour
/// fills.  Pages whose
/// fills are all Type 3 glyphs or under path clips (corpus-02, corpus-07)
/// never reach the device and would compare the CPU with itself.
fn assert_corpus_identical(policy: BackendPolicy) {
    assert_corpus_pages_identical("corpus-08-scan-dct-1927.pdf", &[1, 2, 3], policy);
    assert_corpus_pages_identical("corpus-05-academic-book.pdf", &[1, 2, 3, 4], policy);
    assert_corpus_pages_identical("corpus-04-ebook-mixed.pdf", &[1, 2, 3, 4], policy);
    assert_corpus_pages_identical("corpus-11-cmyk-swatches.pdf", &[1], policy);
}

#[cfg(all(feature = "gpu-aa", feature = "gpu-validation"))]
mod cuda {
    use super::{
        BackendPolicy, assert_corpus_identical, assert_overprint_page_identical,
        assert_synthetic_page_identical,
    };

    #[test]
    fn overprint_fills_render_identically_on_cuda() {
        assert_overprint_page_identical(BackendPolicy::ForceCuda);
    }

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
    use super::{
        BackendPolicy, assert_corpus_identical, assert_overprint_page_identical,
        assert_synthetic_page_identical,
    };

    #[test]
    fn overprint_fills_render_identically_on_vulkan() {
        assert_overprint_page_identical(BackendPolicy::ForceVulkan);
    }

    #[test]
    fn synthetic_fills_render_identically_on_vulkan() {
        assert_synthetic_page_identical(BackendPolicy::ForceVulkan);
    }

    #[test]
    fn corpus_pages_render_identically_on_vulkan() {
        assert_corpus_identical(BackendPolicy::ForceVulkan);
    }
}
