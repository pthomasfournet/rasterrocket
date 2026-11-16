//! End-to-end check of the ICC CMYK → RGB dispatch of a GPU session: an
//! image in an ICC-based CMYK space of at least `GPU_ICC_CLUT_THRESHOLD`
//! pixels converts on the device through the grid of the profile's
//! tabulated conversion, and every smaller image converts on the CPU; both
//! are a CPU session's conversion byte for byte.
//!
//! Page 1 draws one 1000 × 600 image (600 000 pixels, converted on the
//! device); page 2 draws the same samples as six 1000 × 100 images, each
//! below the threshold.  Both pages are rendered at one pixel per sample in
//! a GPU and a CPU session and must be byte-identical between them.
//!
//! The Vulkan case runs on the device `PDF_RASTER_VULKAN_DEVICE` names, or
//! the automatically ranked one, and is skipped when no Vulkan device
//! opens.
//!
//! Run with:
//!   `cargo test -p rasterrocket --features vulkan,gpu-icc --test gpu_icc_render_parity`
//!   `PDF_RASTER_VULKAN_DEVICE=radv cargo test -p rasterrocket --features vulkan,gpu-icc --test gpu_icc_render_parity`
//!   `cargo test -p rasterrocket --features gpu-icc,gpu-validation --test gpu_icc_render_parity`

#![cfg(all(
    feature = "gpu-icc",
    any(feature = "vulkan", feature = "gpu-validation")
))]

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use std::fmt::Write as _;

use common::{assemble_pdf, assert_identical, session, stream_object};
use rasterrocket::{
    BackendPolicy, PageDiagnostics, RasterSession, RenderParams, SessionConfig,
    open_session_from_bytes, render_page_rgb_with_diagnostics,
};

/// A CMYK output profile whose transform is a four-dimensional lookup.
const CMYK_PROFILE: &[u8] = include_bytes!("../../jpeg2000/assets/CGATS001Compat-v2-micro.icc");

const WIDTH: usize = 1000;
const HEIGHT: usize = 600;

/// `WIDTH × HEIGHT` pseudo-random CMYK samples, row-major from the top.
fn samples() -> Vec<u8> {
    let mut state = 0x1cc0_c1a7_u32;
    (0..WIDTH * HEIGHT * 4)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state.to_be_bytes()[0]
        })
        .collect()
}

/// Strips page 2 draws the samples in, each below the threshold.
const STRIPS: usize = 6;

/// The two-page document described in the module documentation.
fn fixture_pdf() -> Vec<u8> {
    let data = samples();
    let rows = HEIGHT / STRIPS;
    let strip_len = data.len() / STRIPS;
    let image = |height: usize, bytes: &[u8]| {
        stream_object(
            &format!(
                "/Type /XObject /Subtype /Image /Width {WIDTH} /Height {height} \
                 /ColorSpace [/ICCBased 7 0 R] /BitsPerComponent 8"
            ),
            bytes,
        )
    };
    let strips = (0..STRIPS).fold(String::new(), |mut acc, i| {
        let _ = write!(acc, " /S{i} {} 0 R", 9 + i);
        acc
    });
    let page = |contents: u32| {
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {WIDTH} {HEIGHT}] \
             /Resources << /XObject << /Im 8 0 R{strips} >> >> \
             /Contents {contents} 0 R >>"
        )
        .into_bytes()
    };
    // Strip `i` holds sample rows `i * rows ..`, drawn from the top.
    let strip_draws = (0..STRIPS).fold(String::new(), |mut acc, i| {
        let y = HEIGHT - (i + 1) * rows;
        let _ = write!(acc, "q {WIDTH} 0 0 {rows} 0 {y} cm /S{i} Do Q ");
        acc
    });
    let mut objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>".to_vec(),
        page(5),
        page(6),
        stream_object(
            "",
            format!("q {WIDTH} 0 0 {HEIGHT} 0 0 cm /Im Do Q").as_bytes(),
        ),
        stream_object("", strip_draws.as_bytes()),
        stream_object("/N 4", CMYK_PROFILE),
        image(HEIGHT, &data),
    ];
    objects.extend(data.chunks_exact(strip_len).map(|strip| image(rows, strip)));
    assemble_pdf(&objects)
}

/// Page `page` of `session` at one pixel per sample.
fn render(session: &RasterSession, page: u32) -> ((u32, u32, Vec<u8>), PageDiagnostics) {
    let (bmp, diag) = render_page_rgb_with_diagnostics(session, page, &RenderParams::new(1.0))
        .expect("render page");
    ((bmp.width, bmp.height, bmp.data().to_vec()), diag)
}

/// The device converts page 1's image and nothing on page 2, without a
/// failure; both pages are byte-identical to the CPU session's.
fn assert_device_conversion_matches_cpu(gpu: &RasterSession, what: &str) {
    if std::env::var_os("PDF_RASTER_ICC_CLUT_THRESHOLD").is_some() {
        eprintln!(
            "skipping: PDF_RASTER_ICC_CLUT_THRESHOLD overrides the threshold the pages straddle"
        );
        return;
    }
    let cpu_session = session(fixture_pdf(), BackendPolicy::CpuOnly);
    let (device, diag) = render(gpu, 1);
    assert_eq!(
        (diag.device_icc_conversions, diag.device_icc_failures),
        (1, 0),
        "{what}: device ICC conversions and failures on the large image's page \
         (0 conversions without a failure: the device failed the exactness check)"
    );
    assert_eq!((device.0, device.1), (1000, 600), "{what}: page size");
    assert!(
        device.2.iter().any(|&b| b != 0xff),
        "{what}: the image rendered blank"
    );
    let (cpu, _) = render(&cpu_session, 1);
    assert_identical(&cpu, &device, what);
    let (strips, diag) = render(gpu, 2);
    assert_eq!(
        (diag.device_icc_conversions, diag.device_icc_failures),
        (0, 0),
        "{what}: images below the threshold reached the device"
    );
    let (cpu, _) = render(&cpu_session, 2);
    assert_identical(&cpu, &strips, what);
}

#[cfg(feature = "vulkan")]
#[test]
fn icc_cmyk_image_converts_on_vulkan() {
    let mut config = SessionConfig::with_policy(BackendPolicy::ForceVulkan);
    config.vulkan_device = SessionConfig::vulkan_device_from_env();
    let session = match open_session_from_bytes(fixture_pdf(), &config) {
        Ok(session) => session,
        Err(e) => {
            eprintln!("skipping: no Vulkan session ({e})");
            return;
        }
    };
    assert_device_conversion_matches_cpu(&session, "vulkan");
}

#[cfg(feature = "gpu-validation")]
#[test]
fn icc_cmyk_image_converts_on_cuda() {
    let config = SessionConfig::with_policy(BackendPolicy::ForceCuda);
    let session = open_session_from_bytes(fixture_pdf(), &config).expect("CUDA session");
    assert_device_conversion_matches_cpu(&session, "cuda");
}
