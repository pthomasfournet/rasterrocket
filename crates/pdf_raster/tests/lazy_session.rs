//! Verify that opening a session does not eagerly walk the page tree.
//!
//! `open_session` reads `/Pages /Count` directly (memoised by the underlying
//! [`pdf::Document`]) and defers per-page `ObjectId` resolution to first
//! render via the [`pdf::Document::get_page`] descent.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use common::{SQUARE, corpus_fixture, one_page, session, stream_object};
use rasterrocket::{BackendPolicy, RasterError, SessionConfig, open_session};

#[test]
fn session_open_reports_correct_total_pages() {
    let Some(path) = corpus_fixture("corpus-05-academic-book.pdf") else {
        return;
    };
    let session = open_session(&path, &SessionConfig::default()).expect("open");
    assert_eq!(session.total_pages(), 601);
}

#[test]
fn resolve_page_returns_consistent_object_ids() {
    let pdf = one_page(
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Contents 4 0 R >>",
        vec![stream_object("", SQUARE)],
    );
    let session = session(pdf, BackendPolicy::CpuOnly);
    assert_eq!(session.total_pages(), 1);

    // Two calls for the same page return the same object id: the
    // underlying [`pdf::Document::get_page`] is a pure function of
    // `(doc, idx)`.
    let id1 = session.resolve_page(1).expect("page 1 first");
    let id2 = session.resolve_page(1).expect("page 1 second");
    assert_eq!(id1, id2);
    assert_eq!(id1, (3, 0), "the page object");

    // Past the end fails with a 1-based `PageOutOfRange`; the caller does
    // not translate from the descender's 0-based variant.
    match session.resolve_page(2) {
        Err(RasterError::PageOutOfRange { page: 2, total: 1 }) => {}
        other => panic!("expected PageOutOfRange {{ page: 2, total: 1 }}, got {other:?}"),
    }
}
