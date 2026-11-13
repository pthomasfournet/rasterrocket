//! A session opened from shared bytes or from a reader renders exactly as
//! one opened from the file or from an owned `Vec`.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use std::sync::Arc;

use common::{SQUARE, corpus_fixture, one_page, stream_object};
use rasterrocket::{
    BackendPolicy, RasterError, RasterOptions, RasterSession, SessionConfig, open_session,
    open_session_from_bytes, open_session_from_reader, open_session_from_shared, render_page_gray,
};

/// Caller-defined storage, shared by reference count.
struct Storage(Arc<Vec<u8>>);

impl AsRef<[u8]> for Storage {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

/// Gray pixels of the first `pages` pages of `session` at 72 dpi.
fn renders(session: &RasterSession, pages: u32) -> Vec<Vec<u8>> {
    let opts = RasterOptions::new(72.0);
    (1..=pages.min(session.total_pages()))
        .map(|page| {
            render_page_gray(session, page, &opts)
                .expect("page renders")
                .pixels
        })
        .collect()
}

/// Every source kind opens `bytes` (read from `path` when given) to a
/// session rendering the first `pages` pages as the owned `Vec` does.
fn assert_sources_agree(bytes: &[u8], path: Option<&std::path::Path>, pages: u32) {
    let config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    let want = renders(
        &open_session_from_bytes(bytes.to_vec(), &config).expect("owned"),
        pages,
    );
    assert_ne!(want.len(), 0, "renders a page");

    let arc: Arc<[u8]> = Arc::from(bytes);
    let sessions = [
        (
            "Arc<[u8]>",
            open_session_from_shared(arc, &config).expect("Arc<[u8]>"),
        ),
        (
            "&'static [u8]",
            open_session_from_shared(&*Box::leak(bytes.to_vec().into_boxed_slice()), &config)
                .expect("&'static"),
        ),
        (
            "custom storage",
            open_session_from_shared(Storage(Arc::new(bytes.to_vec())), &config)
                .expect("custom storage"),
        ),
        (
            "Cursor",
            open_session_from_reader(std::io::Cursor::new(bytes.to_vec()), &config)
                .expect("reader"),
        ),
    ];
    for (what, session) in &sessions {
        assert!(renders(session, pages) == want, "{what} renders differ");
    }
    if let Some(path) = path {
        let file = std::fs::File::open(path).expect("open file");
        let from_reader = open_session_from_reader(file, &config).expect("file reader");
        assert!(renders(&from_reader, pages) == want, "file reader differs");
        let from_path = open_session(path, &config).expect("path");
        assert!(renders(&from_path, pages) == want, "path differs");
    }
}

#[test]
fn sources_render_identically() {
    let pdf = one_page(
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Contents 4 0 R >>",
        vec![stream_object("", SQUARE)],
    );
    assert_sources_agree(&pdf, None, 1);
}

#[test]
fn sources_render_corpus_documents_identically() {
    for name in [
        "corpus-01-native-text-small.pdf",
        "corpus-06-modern-layout-dct.pdf",
    ] {
        let Some(path) = corpus_fixture(name) else {
            continue;
        };
        let bytes = std::fs::read(&path).expect("read fixture");
        assert_sources_agree(&bytes, Some(&path), 3);
    }
}

#[test]
fn reader_and_shared_errors() {
    struct Failing;
    impl std::io::Read for Failing {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("unplugged"))
        }
    }
    let config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    let err = open_session_from_reader(Failing, &config)
        .err()
        .expect("fails");
    assert!(
        matches!(err, RasterError::Pdf(_)) && err.to_string().contains("unplugged"),
        "{err}"
    );
    assert!(matches!(
        open_session_from_shared(&b"%PDF-1.7 truncated"[..], &config),
        Err(RasterError::Pdf(_))
    ));
}
