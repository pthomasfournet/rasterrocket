//! The Vulkan device of a session is destroyed, with its instance, when the
//! last session holding it drops: no device outlives the sessions, so no
//! driver thread runs while the process exits.
//!
//! [`gpu::backend::vulkan::open_devices`] counts the devices alive in the
//! process; this file is its own test binary and its tests hold [`SERIAL`],
//! so the count is theirs alone.  `PDF_RASTER_VULKAN_DEVICE` picks the
//! device.  The tests skip when no Vulkan device opens.
//!
//! Run with:
//!   `cargo test -p rasterrocket --features vulkan --test vulkan_teardown`

#![cfg(feature = "vulkan")]

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use common::{SCALE, stream_object, with_page_tree};
use gpu::backend::vulkan::open_devices;
use rasterrocket::{
    BackendPolicy, RasterOptions, RasterSession, SessionConfig, open_session_from_bytes,
    raster_pdf, render_channel, render_page_rgb,
};

/// Held by every test: the device count is process-wide.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A one-page PDF of a filled rectangle, a page the GPU page raster renders.
fn page_pdf() -> Vec<u8> {
    with_page_tree(&[
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R >>".to_vec(),
        stream_object("", b"0.2 0.4 0.8 rg 20.5 30.25 120 90 re f\n"),
    ])
}

/// A config of `policy` on the device `PDF_RASTER_VULKAN_DEVICE` names.
fn config(policy: BackendPolicy, gpu_pages: bool) -> SessionConfig {
    let mut config = SessionConfig::with_policy(policy);
    config.vulkan_device = SessionConfig::vulkan_device_from_env();
    config.gpu_pages = gpu_pages;
    config
}

/// A `ForceVulkan` session over [`page_pdf`], or `None` (the test is
/// skipped) when no Vulkan device opens.
fn vulkan_session() -> Option<RasterSession> {
    open_session_from_bytes(page_pdf(), &config(BackendPolicy::ForceVulkan, true))
        .inspect_err(|e| eprintln!("skipping: no Vulkan device: {e}"))
        .ok()
}

fn render(session: &RasterSession) {
    let _ = render_page_rgb(session, 1, SCALE).expect("render page");
}

#[test]
fn a_session_destroys_its_device_when_it_drops() {
    let _serial = serial();
    assert_eq!(open_devices(), 0, "a device is open before any session");
    let Some(session) = vulkan_session() else {
        return;
    };
    assert_eq!(open_devices(), 1);
    render(&session);
    drop(session);
    assert_eq!(open_devices(), 0, "the device outlived its session");
}

#[test]
fn sessions_alive_together_share_one_device() {
    let _serial = serial();
    let Some(first) = vulkan_session() else {
        return;
    };
    let second = vulkan_session().expect("a second session on the open device");
    assert_eq!(open_devices(), 1, "two sessions opened two devices");
    render(&second);
    drop(first);
    assert_eq!(open_devices(), 1, "the device closed under a live session");
    render(&second);
    drop(second);
    assert_eq!(open_devices(), 0);

    let third = vulkan_session().expect("a session after the device closed");
    assert_eq!(open_devices(), 1, "no device reopened");
    render(&third);
    drop(third);
    assert_eq!(open_devices(), 0);
}

/// Under `Auto` the GPU page device is opened on a thread of its own once a
/// page asks for it; the session destroys it as well.
#[test]
fn a_device_opened_for_gpu_pages_is_destroyed_with_its_session() {
    let _serial = serial();
    if vulkan_session().is_none() {
        return;
    }
    assert_eq!(open_devices(), 0);
    let session = open_session_from_bytes(page_pdf(), &config(BackendPolicy::Auto, true))
        .expect("an Auto session");
    assert_eq!(open_devices(), 0, "Auto opened a device at session open");
    let deadline = Instant::now() + Duration::from_secs(60);
    while open_devices() == 0 {
        assert!(
            Instant::now() < deadline,
            "the GPU page device never opened"
        );
        render(&session);
        std::thread::sleep(Duration::from_millis(10));
    }
    render(&session);
    drop(session);
    assert_eq!(open_devices(), 0, "the device outlived its session");
}

/// Set in the child process that [`the_library_entry_points_destroy_the_device`]
/// runs under `PDF_RASTER_BACKEND=vulkan`.
const CHILD: &str = "RASTERROCKET_VULKAN_TEARDOWN_CHILD";

/// `raster_pdf` and `render_channel` take their policy from
/// `PDF_RASTER_BACKEND`; the test runs itself again with it set.
#[test]
fn the_library_entry_points_destroy_the_device() {
    let _serial = serial();
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe().expect("test binary"))
            .args([
                "the_library_entry_points_destroy_the_device",
                "--exact",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("PDF_RASTER_BACKEND", "vulkan")
            .status()
            .expect("run the test under PDF_RASTER_BACKEND=vulkan");
        assert!(status.success(), "the child test failed: {status}");
        return;
    }
    if vulkan_session().is_none() {
        return;
    }
    let scratch = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("vulkan_teardown_{}.pdf", std::process::id()));
    std::fs::write(&scratch, page_pdf()).expect("write the PDF");
    let opts = RasterOptions::new(72.0);

    let mut pages = raster_pdf(&scratch, &opts);
    let (_, first) = pages.next().expect("a page");
    let _ = first.expect("raster_pdf renders the page");
    assert_eq!(open_devices(), 1, "raster_pdf opened no device");
    assert!(pages.next().is_none());
    assert_eq!(
        open_devices(),
        0,
        "raster_pdf kept its device past the last page"
    );
    drop(pages);

    let pages = render_channel(&scratch, &opts, 2);
    let rendered: Vec<_> = pages.iter().collect();
    assert_eq!(rendered.len(), 1);
    assert!(rendered[0].1.is_ok(), "render_channel: {:?}", rendered[0].1);
    assert_eq!(
        open_devices(),
        0,
        "render_channel's device outlived the channel"
    );
    let _ = std::fs::remove_file(&scratch);
}
