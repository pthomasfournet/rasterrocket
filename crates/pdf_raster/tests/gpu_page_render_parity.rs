//! End-to-end render parity for whole pages rendered on a GPU
//! (`SessionConfig::gpu_pages`): a page the device renders must be
//! byte-identical to the CPU render of the same page, and a page it
//! declines must render on the CPU with the reason reported.
//!
//! Sessions run under the forcing policy `RR_GPU_PAGE_POLICY` names
//! (`vulkan`, the default, or `d3d12`), which opens the device at session
//! open, so every page is offered to it; `PDF_RASTER_VULKAN_DEVICE` or
//! `PDF_RASTER_D3D12_DEVICE` picks the device.  The tests skip when no
//! device opens, unless `PDF_RASTER_REQUIRE_VULKAN=1` or
//! `PDF_RASTER_REQUIRE_D3D12=1` requires the policy's API.
//!
//! Run with:
//!   `cargo test -p rasterrocket --features vulkan --test gpu_page_render_parity`
//!   `RR_GPU_PAGE_POLICY=d3d12 cargo test -p rasterrocket --features d3d12 --test gpu_page_render_parity`

#![cfg(device_pages)]

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;
#[cfg(all(windows, feature = "d3d12"))]
#[path = "common/warp.rs"]
mod warp;

use common::device::{DeviceApi, gpu_page_policy, gpu_pages_or_skip};
use common::{
    Page, SCALE, assert_identical, corpus_fixture, render, render_diagnosed, session,
    stream_object, with_page_tree,
};
use rasterrocket::{
    BackendPolicy, DevicePage, FinishedCanvas, PageDiagnostics, RasterError, RasterSession,
    RenderParams, SampleDepth, SessionConfig, open_session, open_session_from_bytes,
    render_page_canvas_with_diagnostics,
};

/// A config rendering GPU pages under `policy` on the device the
/// environment names for the policy's API, with no device list, so the
/// pages render on the session's device.
fn gpu_config(policy: BackendPolicy) -> SessionConfig {
    let mut config = SessionConfig::with_policy(policy);
    if let Some(api) = DeviceApi::of_policy(policy) {
        api.select_from_env(&mut config);
    }
    config.gpu_pages = true;
    config
}

/// A GPU-page session over `pdf` under [`gpu_page_policy`], or `None`
/// (the test is skipped) when no device opens and the run does not
/// require one.
fn gpu_session(pdf: Vec<u8>) -> Option<RasterSession> {
    let policy = gpu_page_policy();
    let session = gpu_pages_or_skip(policy, open_session_from_bytes(pdf, &gpu_config(policy)));
    #[cfg(all(windows, feature = "d3d12"))]
    if policy == BackendPolicy::ForceD3d12 {
        warp::report_loaded_warp();
    }
    session
}

/// A one-page PDF of `content` on a 400 × 400 pt page whose resources
/// name the graphics state `/T` (alpha) and the fonts `/F1` and `/F2`.
fn page_pdf(content: &[u8]) -> Vec<u8> {
    with_page_tree(&[
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 400] \
           /Resources << /ExtGState << /T << /ca 0.45 /CA 0.7 >> >> \
           /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> \
           /F2 << /Type /Font /Subtype /Type1 /BaseFont /Times-Italic >> >> >> \
           /Contents 4 0 R >>"
            .to_vec(),
        stream_object("", content),
    ])
}

/// Fills and strokes the device renders: rectangles with fractional
/// edges, a triangle, a Bézier circle, an even-odd donut, a fill at
/// constant alpha, fills under rectangular clips, wide strokes with caps,
/// joins and dashes, opaque hairlines, fills off the page's edges, and
/// text filled, stroked, translucent and under a rectangular clip.
const DEVICE_CONTENT: &[u8] = b"\
0.47843 0.45882 0.44706 rg 100.31 50.27 4.76 76.8 re f\n\
0 0 1 rg 30.3 300.7 m 190.9 330.2 l 60.1 380.4 l h f\n\
1 0 0 rg 291.27 250.61 m 291.27 273.20 272.96 291.51 250.37 291.51 c \
227.78 291.51 209.47 273.20 209.47 250.61 c \
209.47 228.02 227.78 209.71 250.37 209.71 c \
272.96 209.71 291.27 228.02 291.27 250.61 c h f\n\
0 0.5 0 rg 300.5 40.5 80 80 re 320.7 60.3 40 40 re f*\n\
q /T gs 0.6 0.2 0.8 rg 120.4 120.6 90.3 60.7 re f 0.1 0.7 0.3 RG 7 w 130 130 m 200 170 l S Q\n\
q 20 200 100 50 re W n 0.2 0.2 0.2 rg 0 180 150 100 re f Q\n\
0.9 0.3 0.1 RG 5.5 w 1 J 2 j [12 4] 3 d 40 40 m 160 90 l 90 160 l S [] 0 d\n\
0 0 0.6 RG 0 w 10 390 m 390 10 l S 0.4 w 15 20 m 380 30 l S\n\
0.3 0.6 0.9 rg -20.5 -10.3 60 40 re f\n\
0.9 0.5 0.1 rg 370.2 380.7 60 60 re f\n\
BT /F1 18 Tf 0 0 0 rg 30.4 350.2 Td (Device text, AVAWfi) Tj ET\n\
q /T gs BT /F2 31 Tf 0.8 0 0.2 rg 0.1 0.1 0.6 RG 0.8 w 2 Tr 60 300 Td (Italic) Tj ET Q\n\
q 200 20 70 30 re W n BT /F1 40 Tf 0 0.4 0 rg 190 25 Td 1 Tr (Clip) Tj 0 Tr (ped) Tj ET Q\n";

/// Page `page` of `session` at [`SCALE`] and 16 bits per sample, its
/// samples as little-endian bytes, with the diagnostics of the render.
fn render16_diagnosed(session: &RasterSession, page: u32) -> (Page, PageDiagnostics) {
    let params = RenderParams::new(SCALE).with_depth(SampleDepth::Sixteen);
    let (canvas, diag) =
        render_page_canvas_with_diagnostics(session, page, &params).expect("render page");
    let FinishedCanvas::Rgb16(bitmap) = canvas else {
        panic!("a 16-bit RGB render, not {canvas:?}");
    };
    let bytes = bitmap.data().iter().flat_map(|v| v.to_le_bytes()).collect();
    ((bitmap.width, bitmap.height, bytes), diag)
}

/// [`render16_diagnosed`] without the diagnostics.
fn render16(session: &RasterSession, page: u32) -> Page {
    render16_diagnosed(session, page).0
}

/// A renderer of a page of a session.
type Render = fn(&RasterSession, u32) -> Page;

/// A renderer of a page of a session: the page and its diagnostics.
type Diagnosed = fn(&RasterSession, u32) -> (Page, PageDiagnostics);

/// The renders of pages at 8 and 16 bits per sample: CPU, diagnosed.
const DEPTHS: [(&str, Render, Diagnosed); 2] = [
    ("8-bit", render, render_diagnosed),
    ("16-bit", render16, render16_diagnosed),
];

#[test]
fn device_page_renders_identically() {
    device_page_renders_identically_with(render, render_diagnosed);
}

#[test]
fn device_page_renders_identically_at_16_bits() {
    device_page_renders_identically_with(render16, render16_diagnosed);
}

/// The page of [`DEVICE_CONTENT`] renders on the device, identically to
/// the CPU, as `cpu` and `gpu` render it.
fn device_page_renders_identically_with(cpu: Render, gpu_render: Diagnosed) {
    let pdf = page_pdf(DEVICE_CONTENT);
    let Some(gpu) = gpu_session(pdf.clone()) else {
        return;
    };
    let cpu = cpu(&session(pdf, BackendPolicy::CpuOnly), 1);
    assert!(cpu.2.iter().any(|&b| b != 0xff), "blank on the CPU");
    let (page, diag) = gpu_render(&gpu, 1);
    assert!(
        matches!(diag.device_page, DevicePage::Rendered { commands } if commands >= 12),
        "the device did not render the page: {:?}",
        diag.device_page
    );
    assert_identical(&cpu, &page, "device page");
}

/// A page clipped by a path, or by text in a clipping mode, renders on
/// the device, identically to the CPU, at 8 and 16 bits.
#[test]
fn clipped_pages_render_on_the_device_identically() {
    for content in [
        &b"0 g BT /F1 120 Tf 7 Tr 10 100 Td (Clip) Tj ET 0 0 1 rg 0 0 400 400 re f"[..],
        b"10 10 m 200 10 l 100 200 l h W n 30 20 m 390 90 l 40 370 l h W* n \
          0 0 1 rg 0 0 400 400 re f /T gs 1 0 0 rg BT /F2 60 Tf 20 60 Td (text) Tj ET",
    ] {
        let pdf = page_pdf(content);
        let Some(gpu) = gpu_session(pdf.clone()) else {
            return;
        };
        let cpu_session = session(pdf, BackendPolicy::CpuOnly);
        for (depth, cpu, gpu_render) in DEPTHS {
            let (page, diag) = gpu_render(&gpu, 1);
            assert!(
                matches!(diag.device_page, DevicePage::Rendered { .. }),
                "{depth}: {:?}",
                diag.device_page
            );
            let want = cpu(&cpu_session, 1);
            assert!(want.2.iter().any(|&b| b != 0xff), "blank on the CPU");
            assert_identical(&want, &page, depth);
        }
    }
}

#[test]
fn declined_pages_render_on_the_cpu_with_the_reason() {
    for (content, reason) in [
        (&b"1 0 0 rg 10 10 100 100 re f /Sh sh"[..], "shading"),
        (
            b"/T gs 0 0 1 RG 0 w 10 10 m 300 300 l S",
            "translucent hairline",
        ),
    ] {
        let pdf = page_pdf(content);
        let Some(gpu) = gpu_session(pdf.clone()) else {
            return;
        };
        let cpu_session = session(pdf, BackendPolicy::CpuOnly);
        for (depth, cpu, gpu_render) in DEPTHS {
            let (page, diag) = gpu_render(&gpu, 1);
            assert_eq!(diag.device_page, DevicePage::Ineligible(reason), "{depth}");
            assert_identical(&cpu(&cpu_session, 1), &page, &format!("{depth} {reason}"));
        }
    }
}

/// Under `Auto` the Vulkan device opens beside the renders: pages render
/// on the CPU until it is open, then on the device, identically.
#[cfg(feature = "vulkan")]
#[test]
fn auto_opens_the_device_beside_the_renders() {
    use std::time::{Duration, Instant};

    let pdf = page_pdf(DEVICE_CONTENT);
    if gpu_page_policy() != BackendPolicy::ForceVulkan {
        eprintln!("skipping: Auto renders GPU pages on Vulkan only");
        return;
    }
    if gpu_session(pdf.clone()).is_none() {
        return;
    }
    // An Auto session opens whether or not a device does.
    let gpu = open_session_from_bytes(pdf.clone(), &gpu_config(BackendPolicy::Auto))
        .expect("an Auto session opens");
    let cpu = render(&session(pdf, BackendPolicy::CpuOnly), 1);
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let (page, diag) = render_diagnosed(&gpu, 1);
        assert_identical(&cpu, &page, "page under Auto");
        match diag.device_page {
            DevicePage::Rendered { .. } => break,
            DevicePage::NotOffered => {}
            other => panic!("unexpected outcome {other:?}"),
        }
        assert!(Instant::now() < deadline, "the device never opened");
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// GPU pages under a backend that renders none are refused, naming the
/// backends that do on the platform: `d3d12` on Windows only.
#[test]
fn gpu_pages_need_a_gpu_page_policy() {
    let mut config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    config.gpu_pages = true;
    match open_session_from_bytes(page_pdf(DEVICE_CONTENT), &config) {
        Err(RasterError::InvalidOptions(msg)) => {
            for named in ["--backend auto", "--backend vulkan"] {
                assert!(msg.contains(named), "{msg}");
            }
            assert_eq!(msg.contains("--backend d3d12"), cfg!(windows), "{msg}");
        }
        Err(other) => panic!("expected InvalidOptions, got {other:?}"),
        Ok(_) => panic!("a CPU-only session accepted GPU pages"),
    }
}

/// Every corpus fixture: the first pages render identically, whatever the
/// device made of them.  The outcomes are printed (`--nocapture`) as the
/// fixtures' GPU coverage.
#[test]
fn corpus_pages_render_identically() {
    corpus_pages_render_identically_with(render, render_diagnosed);
}

/// [`corpus_pages_render_identically`] at 16 bits per sample.
#[test]
fn corpus_pages_render_identically_at_16_bits() {
    corpus_pages_render_identically_with(render16, render16_diagnosed);
}

/// Every corpus fixture's first pages render identically on the CPU and
/// with GPU pages, as `cpu_render` and `gpu_render` render them.
fn corpus_pages_render_identically_with(cpu_render: Render, gpu_render: Diagnosed) {
    const FIXTURES: &[&str] = &[
        "corpus-01-native-text-small.pdf",
        "corpus-02-native-vector-text.pdf",
        "corpus-03-native-text-dense.pdf",
        "corpus-04-ebook-mixed.pdf",
        "corpus-05-academic-book.pdf",
        "corpus-06-modern-layout-dct.pdf",
        "corpus-07-journal-dct-heavy.pdf",
        "corpus-08-scan-dct-1927.pdf",
        "corpus-09-scan-dct-1836.pdf",
        "corpus-10-scan-jbig2-jpx.pdf",
        "corpus-11-cmyk-swatches.pdf",
    ];
    let (mut found, mut on_device) = (0, 0);
    for fixture in FIXTURES {
        let Some(path) = corpus_fixture(fixture) else {
            continue;
        };
        found += 1;
        let cpu = open_session(&path, &SessionConfig::with_policy(BackendPolicy::CpuOnly))
            .expect("CPU session");
        let policy = gpu_page_policy();
        let Some(gpu) = gpu_pages_or_skip(policy, open_session(&path, &gpu_config(policy))) else {
            return;
        };
        for page in 1..=cpu.total_pages().min(4) {
            let what = format!("{fixture} page {page}");
            let (got, diag) = gpu_render(&gpu, page);
            assert_ne!(
                diag.device_page,
                DevicePage::Failed,
                "{what}: device failed"
            );
            assert_ne!(
                diag.device_page,
                DevicePage::NotOffered,
                "{what}: not offered"
            );
            eprintln!("{what}: {:?}", diag.device_page);
            on_device += usize::from(matches!(diag.device_page, DevicePage::Rendered { .. }));
            assert_identical(&cpu_render(&cpu, page), &got, &what);
        }
    }
    // Every first page of the fixtures renders on the device.
    assert!(
        found < FIXTURES.len() || on_device == 41,
        "{on_device} fixture pages rendered on the device"
    );
}
