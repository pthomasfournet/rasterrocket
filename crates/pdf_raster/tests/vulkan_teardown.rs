//! The Vulkan device of a session is destroyed, with its instance, when the
//! last session holding it drops: no device outlives the sessions, so no
//! driver thread runs while the process exits.
//!
//! [`gpu::backend::vulkan::open_devices`] counts the devices alive in the
//! process; this file is its own test binary and its tests hold
//! [`teardown::serial`], so the count is theirs alone.  `PDF_RASTER_VULKAN_DEVICE` picks the
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
mod teardown;

use std::time::{Duration, Instant};

use gpu::backend::vulkan::open_devices;
use rasterrocket::{BackendPolicy, RasterSession, open_session_from_bytes};
use teardown::{
    assert_library_entry_points_close, config, in_child_under, page_pdf, render, serial,
};

/// A `ForceVulkan` session over [`page_pdf`], or `None` (the test is
/// skipped) when no Vulkan device opens.
fn vulkan_session() -> Option<RasterSession> {
    open_session_from_bytes(page_pdf(), &config(BackendPolicy::ForceVulkan, true))
        .inspect_err(|e| eprintln!("skipping: no Vulkan device: {e}"))
        .ok()
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

/// `raster_pdf` and `render_channel` take their policy from
/// `PDF_RASTER_BACKEND`; the test runs itself again with it set.
#[test]
fn the_library_entry_points_destroy_the_device() {
    let _serial = serial();
    if !in_child_under("the_library_entry_points_destroy_the_device", "vulkan") {
        return;
    }
    if vulkan_session().is_none() {
        return;
    }
    assert_library_entry_points_close(open_devices);
}
