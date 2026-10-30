//! Helpers shared by the fuzz targets: in-memory PDF assembly and a bounded
//! render of a document through the public pipeline.

/// PDF assembly helpers shared with the rendering integration suites.
#[path = "../../crates/pdf_raster/tests/common/mod.rs"]
pub mod common;

use rasterrocket::{BackendPolicy, SessionConfig, open_session_from_bytes, render_page_rgb};

/// Pages rendered per input.
const MAX_PAGES: u32 = 4;

/// Device pixels per point.  A 612 × 792 pt page renders at 61 × 79 px, so
/// an iteration costs the decode and interpretation work, not the fill.
const SCALE: f64 = 0.1;

/// Open `bytes` on the CPU backend and render up to [`MAX_PAGES`] pages at
/// [`SCALE`].  Every error is discarded: the property under test is that
/// opening and rendering return.
pub fn render_document(bytes: Vec<u8>) {
    render_document_at(bytes, SCALE);
}

/// [`render_document`] at `scale` device pixels per point.
pub fn render_document_at(bytes: Vec<u8>, scale: f64) {
    let config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    let Ok(session) = open_session_from_bytes(bytes, &config) else {
        return;
    };
    for page in 1..=session.total_pages().min(MAX_PAGES) {
        drop(render_page_rgb(&session, page, scale));
    }
}

/// Render a one-page document whose 100 × 100 pt page has `resources` as its
/// resource dictionary and `content` as its content stream.  `rest` are the
/// objects from number 5 on; object 4 is the content stream.
pub fn render_page(resources: &str, content: &[u8], rest: Vec<Vec<u8>>) {
    let page = format!(
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
         /Resources {resources} /Contents 4 0 R >>"
    );
    let mut objects = vec![common::stream_object("", content)];
    objects.extend(rest);
    render_document_at(common::one_page(&page, objects), 0.5);
}

/// Splits fuzz input into fixed-size selector bytes and the remaining
/// payload; `None` when the input is shorter than `N`.
pub fn split_selectors<const N: usize>(data: &[u8]) -> Option<([u8; N], &[u8])> {
    let (head, payload) = data.split_first_chunk::<N>()?;
    Some((*head, payload))
}
