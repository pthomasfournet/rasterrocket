//! Helpers shared by the device-teardown tests.
//!
//! Each such test binary counts the devices of one backend that the
//! process holds open, through a process-wide counter of the gpu crate, so
//! its tests hold [`serial`] and the count is theirs alone.

use std::sync::{Mutex, MutexGuard, PoisonError};

use rasterrocket::{
    BackendPolicy, RasterOptions, RasterSession, SessionConfig, raster_pdf, render_channel,
    render_page_rgb,
};

use crate::common::{SCALE, stream_object, with_page_tree};

/// Held by every test: the device counts are process-wide.
static SERIAL: Mutex<()> = Mutex::new(());

pub(crate) fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A one-page PDF of a filled rectangle, a page the GPU page raster renders.
pub(crate) fn page_pdf() -> Vec<u8> {
    with_page_tree(&[
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R >>".to_vec(),
        stream_object("", b"0.2 0.4 0.8 rg 20.5 30.25 120 90 re f\n"),
    ])
}

/// A config of `policy` on the Vulkan device `PDF_RASTER_VULKAN_DEVICE`
/// names.
pub(crate) fn config(policy: BackendPolicy, gpu_pages: bool) -> SessionConfig {
    let mut config = SessionConfig::with_policy(policy);
    config.vulkan_device = SessionConfig::vulkan_device_from_env();
    config.gpu_pages = gpu_pages;
    config
}

pub(crate) fn render(session: &RasterSession) {
    let _ = render_page_rgb(session, 1, SCALE).expect("render page");
}

/// Set in the child process [`in_child_under`] runs.
const CHILD: &str = "RASTERROCKET_TEARDOWN_CHILD";

/// Whether this process is the child that runs test `name` under
/// `PDF_RASTER_BACKEND=backend`.  In the parent, the child is run and must
/// pass.
pub(crate) fn in_child_under(name: &str, backend: &str) -> bool {
    if std::env::var_os(CHILD).is_some() {
        return true;
    }
    let status = std::process::Command::new(std::env::current_exe().expect("test binary"))
        .args([name, "--exact", "--nocapture"])
        .env(CHILD, "1")
        .env("PDF_RASTER_BACKEND", backend)
        .status()
        .expect("run the test in a child process");
    assert!(status.success(), "the child test failed: {status}");
    false
}

/// `raster_pdf` and `render_channel`, under the policy of
/// `PDF_RASTER_BACKEND`, open one device (counted by `open`) and leave
/// none open once they are done.
pub(crate) fn assert_library_entry_points_close(open: fn() -> usize) {
    let scratch = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("teardown_{}.pdf", std::process::id()));
    std::fs::write(&scratch, page_pdf()).expect("write the PDF");
    let opts = RasterOptions::new(72.0);

    let mut pages = raster_pdf(&scratch, &opts);
    let (_, first) = pages.next().expect("a page");
    let _ = first.expect("raster_pdf renders the page");
    assert_eq!(open(), 1, "raster_pdf opened no device");
    assert!(pages.next().is_none());
    assert_eq!(open(), 0, "raster_pdf kept its device past the last page");
    drop(pages);

    let pages = render_channel(&scratch, &opts, 2);
    let rendered: Vec<_> = pages.iter().collect();
    assert_eq!(rendered.len(), 1);
    assert!(rendered[0].1.is_ok(), "render_channel: {:?}", rendered[0].1);
    assert_eq!(open(), 0, "render_channel's device outlived the channel");
    let _ = std::fs::remove_file(&scratch);
}
