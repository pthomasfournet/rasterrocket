//! End-to-end check of the ICC CMYK → RGB dispatch of a GPU session: an
//! image in an ICC-based CMYK space of at least `GPU_ICC_CLUT_THRESHOLD`
//! pixels converts on the device through the grid of the profile's
//! tabulated conversion, and every smaller image converts on the CPU; both
//! are a CPU session's conversion byte for byte.
//!
//! Page 1 draws one 1000 × 600 image (600 000 pixels, converted on the
//! device); page 2 draws the same samples as six 1000 × 100 images, each
//! below the threshold.  Both pages are rendered at one pixel per sample in
//! a GPU and a CPU session and must be byte-identical between them: on an
//! 8-bit canvas, where the device gives 8-bit samples, and on a 16-bit one
//! and in the wide working space under an output profile, where it gives
//! the 16-bit samples the CPU converts the image's samples widened to.
//!
//! The Vulkan case runs on the device `PDF_RASTER_VULKAN_DEVICE` names, or
//! the automatically ranked one; the D3D12 case (Windows) on the adapter
//! `PDF_RASTER_D3D12_DEVICE` names, or the automatically chosen one.  Each
//! is skipped when its session does not open, unless
//! `PDF_RASTER_REQUIRE_VULKAN=1` or `PDF_RASTER_REQUIRE_D3D12=1` requires
//! it.
//!
//! Run with:
//!   `cargo test -p rasterrocket --features vulkan,gpu-icc --test gpu_icc_render_parity`
//!   `PDF_RASTER_VULKAN_DEVICE=radv cargo test -p rasterrocket --features vulkan,gpu-icc --test gpu_icc_render_parity`
//!   `PDF_RASTER_D3D12_DEVICE=warp cargo test -p rasterrocket --features d3d12,gpu-icc --test gpu_icc_render_parity`
//!   `cargo test -p rasterrocket --features gpu-icc,gpu-validation --test gpu_icc_render_parity`

#![cfg(all(
    feature = "gpu-icc",
    any(
        feature = "vulkan",
        all(windows, feature = "d3d12"),
        feature = "gpu-validation"
    )
))]

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use std::fmt::Write as _;

use std::sync::Arc;

#[cfg(any(feature = "vulkan", all(windows, feature = "d3d12")))]
use common::device::DeviceApi;
use common::{assemble_pdf, assert_identical, stream_object};
use rasterrocket::{
    BackendPolicy, FinishedCanvas, OutputProfile, PageDiagnostics, RasterSession, RenderParams,
    SampleDepth, SessionConfig, WorkingSpace, open_session_from_bytes,
    render_page_canvas_with_diagnostics,
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

/// A configuration of the sessions compared: the canvas depth, and the
/// wide working space under an RGB output profile or the default space.
#[derive(Clone, Copy, Debug)]
struct Setup {
    depth: SampleDepth,
    wide: bool,
}

/// The setups every backend is checked in.
const SETUPS: [Setup; 4] = [
    Setup {
        depth: SampleDepth::Eight,
        wide: false,
    },
    Setup {
        depth: SampleDepth::Sixteen,
        wide: false,
    },
    Setup {
        depth: SampleDepth::Eight,
        wide: true,
    },
    Setup {
        depth: SampleDepth::Sixteen,
        wide: true,
    },
];

impl Setup {
    /// `config` set up for this setup.
    fn apply(self, mut config: SessionConfig) -> SessionConfig {
        if self.wide {
            let path = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../tests/fixtures/icc/printer-rgb-v4.icc"
            );
            let profile = OutputProfile::from_bytes(std::fs::read(path).expect("profile"))
                .expect("an output profile");
            config.output_profile = Some(Arc::new(profile));
            config.working_space = Some(WorkingSpace::Wide);
        }
        config
    }
}

/// Page `page` of `session` at one pixel per sample and depth `depth`:
/// its dimensions and its samples' bytes.
fn render(
    session: &RasterSession,
    page: u32,
    depth: SampleDepth,
) -> ((u32, u32, Vec<u8>), PageDiagnostics) {
    let params = RenderParams::new(1.0).with_depth(depth);
    let (canvas, diag) =
        render_page_canvas_with_diagnostics(session, page, &params).expect("render page");
    let page = match canvas {
        FinishedCanvas::Rgb8(b) => (b.width, b.height, b.data().to_vec()),
        FinishedCanvas::Rgb16(b) => (
            b.width,
            b.height,
            b.data().iter().flat_map(|v| v.to_le_bytes()).collect(),
        ),
        other => panic!("an RGB page, not {other:?}"),
    };
    (page, diag)
}

/// The device converts page 1's image and nothing on page 2, without a
/// failure; both pages are byte-identical to the CPU session's, in every
/// one of the [`SETUPS`], `gpu` opening the GPU session of a
/// configuration (`None`: skipped).
fn assert_device_conversion_matches_cpu(
    gpu: impl Fn(&SessionConfig) -> Option<RasterSession>,
    config: &SessionConfig,
    what: &str,
) {
    if std::env::var_os("PDF_RASTER_ICC_CLUT_THRESHOLD").is_some() {
        eprintln!(
            "skipping: PDF_RASTER_ICC_CLUT_THRESHOLD overrides the threshold the pages straddle"
        );
        return;
    }
    for setup in SETUPS {
        let what = format!("{what} {setup:?}");
        let Some(gpu) = gpu(&setup.apply(config.clone())) else {
            return;
        };
        let cpu_config = setup.apply(SessionConfig::with_policy(BackendPolicy::CpuOnly));
        let cpu_session = open_session_from_bytes(fixture_pdf(), &cpu_config).expect("CPU session");
        let (device, diag) = render(&gpu, 1, setup.depth);
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
        let (cpu, _) = render(&cpu_session, 1, setup.depth);
        assert_identical(&cpu, &device, &what);
        let (strips, diag) = render(&gpu, 2, setup.depth);
        assert_eq!(
            (diag.device_icc_conversions, diag.device_icc_failures),
            (0, 0),
            "{what}: images below the threshold reached the device"
        );
        let (cpu, _) = render(&cpu_session, 2, setup.depth);
        assert_identical(&cpu, &strips, &what);
    }
}

#[cfg(feature = "vulkan")]
#[test]
fn icc_cmyk_image_converts_on_vulkan() {
    let mut config = SessionConfig::with_policy(BackendPolicy::ForceVulkan);
    config.vulkan_device = SessionConfig::vulkan_device_from_env();
    let open = |config: &SessionConfig| {
        common::device::or_skip(
            DeviceApi::Vulkan,
            open_session_from_bytes(fixture_pdf(), config),
        )
    };
    assert_device_conversion_matches_cpu(open, &config, "vulkan");
}

#[cfg(all(windows, feature = "d3d12"))]
#[test]
fn icc_cmyk_image_converts_on_d3d12() {
    let mut config = SessionConfig::with_policy(BackendPolicy::ForceD3d12);
    config.d3d12_device = SessionConfig::d3d12_device_from_env();
    let open = |config: &SessionConfig| {
        let session = common::device::or_skip(
            DeviceApi::D3d12,
            open_session_from_bytes(fixture_pdf(), config),
        );
        report_loaded_warp();
        session
    };
    assert_device_conversion_matches_cpu(open, &config, "d3d12");
}

#[cfg(feature = "gpu-validation")]
#[test]
fn icc_cmyk_image_converts_on_cuda() {
    let open = |config: &SessionConfig| {
        Some(open_session_from_bytes(fixture_pdf(), config).expect("CUDA session"))
    };
    let config = SessionConfig::with_policy(BackendPolicy::ForceCuda);
    assert_device_conversion_matches_cpu(open, &config, "cuda");
}

/// Print the `d3d10warp.dll` loaded in the process, if any; with
/// `RR_WARP_VERSION` set, require a loaded one beside the test binary, of
/// that version.
///
/// # Panics
///
/// When `RR_WARP_VERSION` is set and the loaded module is missing, lies
/// elsewhere or has another version.
#[cfg(all(windows, feature = "d3d12"))]
fn report_loaded_warp() {
    let loaded = rasterrocket::dev::loaded_warp();
    match &loaded {
        Some((path, version)) => eprintln!("WARP module: {} version {version}", path.display()),
        None => eprintln!("WARP module: not loaded"),
    }
    if let Ok(expected) = std::env::var("RR_WARP_VERSION") {
        let (path, version) = loaded.expect("RR_WARP_VERSION is set, but no WARP is loaded");
        let exe = std::env::current_exe().expect("test binary path");
        let canonical = |p: &std::path::Path| std::fs::canonicalize(p).ok();
        assert_eq!(
            path.parent().and_then(canonical),
            exe.parent().and_then(canonical),
            "the loaded WARP is not the one beside the test binary: {}",
            path.display()
        );
        assert_eq!(version, expected.trim(), "WARP version");
    }
}
