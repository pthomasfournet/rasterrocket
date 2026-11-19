//! The GPU page scheduler's tests: the generic ones ([`scheduler`]) run on
//! each API compiled in, the others on Vulkan under `Auto` ([`auto`]).

#[cfg(feature = "vulkan")]
mod auto;
mod capture;
mod scheduler;

use gpu::backend::BackendError;
#[cfg(all(windows, feature = "d3d12"))]
use gpu::backend::d3d12::D3d12Backend;
#[cfg(feature = "vulkan")]
use gpu::backend::vulkan::VulkanBackend;

use super::{DevicePages, SessionDevicePages};
use crate::device_init::SessionDevice;
use crate::test_pdf::{assemble, stream};
use crate::{
    BackendPolicy, DevicePage, RasterError, RasterSession, RenderParams, SessionConfig,
    open_session_from_bytes, render_page_rgb_with_diagnostics,
};

/// Fills and strokes the device renders.
const DEVICE: &str = "0 0 1 rg 10.3 10.7 60.2 50.1 re f 1 0 0 RG 3 w 5 5 m 90 70 l S";

/// A translucent hairline, which the recording refuses.
const REFUSED: &str = "/T gs 0 0 1 RG 0 w 10 10 m 90 90 l S";

/// A GPU API whose devices render GPU pages in these tests, under its
/// forcing policy ([`SessionDevice::FORCE`]).
trait Api: SessionDevice {
    /// The variable whose value `1` turns a skip for want of a device into
    /// a failure.
    const REQUIRE: &'static str;

    /// Set the session's device selector of `config` to the one the
    /// environment names.
    fn select_from_env(config: &mut SessionConfig);

    /// The session's GPU pages, which are on this API.
    fn pages(pages: &SessionDevicePages) -> &DevicePages<Self>;

    /// `pages` as a session's GPU pages.
    fn session_pages(pages: DevicePages<Self>) -> SessionDevicePages;

    /// The error a device of the API reports once it is lost.
    fn lost_device() -> BackendError;
}

#[cfg(feature = "vulkan")]
impl Api for VulkanBackend {
    const REQUIRE: &'static str = "PDF_RASTER_REQUIRE_VULKAN";

    fn select_from_env(config: &mut SessionConfig) {
        config.vulkan_device = SessionConfig::vulkan_device_from_env();
    }

    fn pages(pages: &SessionDevicePages) -> &DevicePages<Self> {
        match pages {
            SessionDevicePages::Vulkan(pages) => pages,
            #[cfg(all(windows, feature = "d3d12"))]
            SessionDevicePages::D3d12(_) => panic!("Direct3D 12 GPU pages, not Vulkan"),
        }
    }

    fn session_pages(pages: DevicePages<Self>) -> SessionDevicePages {
        SessionDevicePages::Vulkan(pages)
    }

    fn lost_device() -> BackendError {
        gpu::backend::vulkan::device_lost_error("vkQueueSubmit")
    }
}

#[cfg(all(windows, feature = "d3d12"))]
impl Api for D3d12Backend {
    const REQUIRE: &'static str = "PDF_RASTER_REQUIRE_D3D12";

    fn select_from_env(config: &mut SessionConfig) {
        config.d3d12_device = SessionConfig::d3d12_device_from_env();
    }

    fn pages(pages: &SessionDevicePages) -> &DevicePages<Self> {
        match pages {
            SessionDevicePages::D3d12(pages) => pages,
            #[cfg(feature = "vulkan")]
            SessionDevicePages::Vulkan(_) => panic!("Vulkan GPU pages, not Direct3D 12"),
        }
    }

    fn session_pages(pages: DevicePages<Self>) -> SessionDevicePages {
        SessionDevicePages::D3d12(pages)
    }

    fn lost_device() -> BackendError {
        gpu::backend::d3d12::device_removed_error("Signal")
    }
}

/// The value of `opened`, or `None` (the caller skips) with a line on
/// stderr naming `B`'s API and `what` is missing.
///
/// # Panics
///
/// When `opened` is an error and `B::REQUIRE` is `1`.
fn or_skip<B: Api, T, E: std::fmt::Display>(what: &str, opened: Result<T, E>) -> Option<T> {
    match opened {
        Ok(value) => Some(value),
        Err(e) => {
            assert!(
                !std::env::var(B::REQUIRE).is_ok_and(|v| v.trim() == "1"),
                "{}=1, but {what} on {}: {e}",
                B::REQUIRE,
                B::API
            );
            eprintln!("skipping: {what} on {} ({e})", B::API);
            None
        }
    }
}

/// A PDF of 100 × 100 pt pages, page `i` painting `contents[i]`.
fn pdf(contents: &[&str]) -> Vec<u8> {
    let n = contents.len();
    let kids = (0..n)
        .map(|i| format!("{} 0 R", 3 + 2 * i))
        .collect::<Vec<_>>()
        .join(" ");
    let mut objects = vec![
        "<</Type /Catalog /Pages 2 0 R>>".to_owned(),
        format!("<</Type /Pages /Kids [{kids}] /Count {n}>>"),
    ];
    for (i, content) in contents.iter().enumerate() {
        objects.push(format!(
            "<</Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
             /Resources <</ExtGState <</T <</CA 0.5>> >> >> /Contents {} 0 R>>",
            4 + 2 * i
        ));
        objects.push(stream(content));
    }
    assemble(&objects.iter().map(String::as_str).collect::<Vec<_>>())
}

/// A config rendering GPU pages under `B`'s forcing policy on the device
/// the environment names.
fn gpu_config<B: Api>() -> SessionConfig {
    let mut config = SessionConfig::with_policy(B::FORCE);
    B::select_from_env(&mut config);
    config.gpu_pages = true;
    config
}

/// A session over `pdf` rendering GPU pages on a device of `B` opened
/// with it, or `None` (the test is skipped) when none opens.
fn gpu_session<B: Api>(pdf: Vec<u8>) -> Option<RasterSession> {
    or_skip::<B, _, _>(
        "no device opens",
        open_session_from_bytes(pdf, &gpu_config::<B>()),
    )
}

/// A session over `pdf` rendering GPU pages on every usable GPU of `B`,
/// opened with it, or `None` (the test is skipped) when fewer than two
/// open.
fn two_device_session<B: Api>(pdf: Vec<u8>) -> Option<RasterSession> {
    let mut config = SessionConfig::with_policy(B::FORCE);
    config.gpu_pages = true;
    config.gpu_devices = vec!["all".to_owned()];
    let session = or_skip::<B, _, _>("no device opens", open_session_from_bytes(pdf, &config))?;
    let devices = device_pages::<B>(&session).devices.len();
    if devices < 2 {
        eprintln!("skipping: {devices} usable GPU on {}, not two", B::API);
        return None;
    }
    Some(session)
}

/// A CPU-only session over `pdf` whose GPU pages are `pages`.
fn session_with<B: Api>(pdf: Vec<u8>, pages: DevicePages<B>) -> RasterSession {
    let mut session =
        open_session_from_bytes(pdf, &SessionConfig::with_policy(BackendPolicy::CpuOnly))
            .expect("CPU session");
    session.device_pages = Some(B::session_pages(pages));
    session
}

fn device_pages<B: Api>(session: &RasterSession) -> &DevicePages<B> {
    B::pages(session.device_pages.as_ref().expect("a GPU-page session"))
}

/// Page `page` of `session` at 2× and how the device was used.
fn render(
    session: &RasterSession,
    page: u32,
    params: &RenderParams,
) -> Result<(Vec<u8>, DevicePage), RasterError> {
    render_page_rgb_with_diagnostics(session, page, params)
        .map(|(bitmap, diag)| (bitmap.data().to_vec(), diag.device_page))
}

/// Page `page` of `pdf` rendered on the CPU.
fn cpu(pdf: Vec<u8>, page: u32) -> Vec<u8> {
    let session = open_session_from_bytes(pdf, &SessionConfig::with_policy(BackendPolicy::CpuOnly))
        .expect("CPU session");
    render(&session, page, &RenderParams::new(2.0))
        .expect("CPU render")
        .0
}

/// A session without GPU pages has no counts.
#[test]
fn a_cpu_session_has_no_device_page_counts() {
    let session = open_session_from_bytes(
        pdf(&[DEVICE]),
        &SessionConfig::with_policy(BackendPolicy::CpuOnly),
    )
    .expect("CPU session");
    assert_eq!(session.device_page_counts(), None);
}

/// The tests of [`scheduler`] on the API `$api`, under its module.
macro_rules! scheduler_tests {
    ($module:ident, $api:ty, [$($extra:ident),* $(,)?]) => {
        mod $module {
            scheduler_tests!(@tests $api,
                pages_render_on_the_cpu_once_the_session_wait_for_the_opening_is_over,
                a_cancelled_page_stops_waiting_for_the_opening,
                a_device_failure_renders_on_the_cpu_and_retires_the_device,
                a_kernel_fault_retires_the_device,
                a_page_beyond_the_device_limits_keeps_the_device,
                a_page_beyond_the_memory_left_keeps_the_device,
                a_cancelled_recording_is_not_rendered_again,
                concurrent_pages_all_render_on_the_device,
                a_wide_page_renders_on_the_cpu_and_warns_once,
                gpu_pages_need_vector_antialiasing,
                held_pages_push_claims_to_the_other_device,
                a_failed_device_leaves_the_others_in_service,
                a_device_list_naming_one_device_twice_is_refused,
                $($extra),*
            );
        }
    };
    (@tests $api:ty, $($name:ident),*) => {
        $(
            #[test]
            fn $name() {
                super::scheduler::$name::<$api>();
            }
        )*
    };
}

#[cfg(feature = "vulkan")]
scheduler_tests!(
    vulkan,
    gpu::backend::vulkan::VulkanBackend,
    [
        concurrent_pages_spread_over_two_devices,
        a_lost_device_retires_and_the_page_renders_on_the_cpu,
    ]
);

#[cfg(all(windows, feature = "d3d12"))]
scheduler_tests!(
    d3d12,
    gpu::backend::d3d12::D3d12Backend,
    [
        two_d3d12_adapters_share_pages,
        a_removed_d3d12_device_retires_and_the_page_renders_on_the_cpu,
        an_explicit_warp_renders_gpu_pages,
    ]
);
