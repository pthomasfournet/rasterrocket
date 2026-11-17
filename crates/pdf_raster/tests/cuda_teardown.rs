//! The CUDA context of a session, with its kernel modules and scratch
//! buffers, is freed when the last session holding it drops.
//!
//! [`gpu::open_contexts`] counts the contexts alive in the process; this
//! file is its own test binary and its tests hold [`teardown::serial`], so
//! the count is theirs alone.  The tests skip when no CUDA context opens.
//!
//! Run with:
//!   `cargo test -p rasterrocket --features gpu-icc --test cuda_teardown`

#![cfg(any(feature = "gpu-icc", feature = "cache"))]

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;
mod teardown;

use gpu::open_contexts;
use rasterrocket::{BackendPolicy, RasterSession, open_session_from_bytes};
use teardown::{
    assert_library_entry_points_close, config, in_child_under, page_pdf, render, serial,
};

/// A `ForceCuda` session over [`page_pdf`], or `None` (the test is
/// skipped) when no CUDA context opens.
fn cuda_session() -> Option<RasterSession> {
    open_session_from_bytes(page_pdf(), &config(BackendPolicy::ForceCuda, false))
        .inspect_err(|e| eprintln!("skipping: no CUDA context: {e}"))
        .ok()
}

#[test]
fn a_session_frees_its_context_when_it_drops() {
    let _serial = serial();
    assert_eq!(open_contexts(), 0, "a context is open before any session");
    let Some(session) = cuda_session() else {
        return;
    };
    assert_eq!(open_contexts(), 1);
    render(&session);
    drop(session);
    assert_eq!(open_contexts(), 0, "the context outlived its session");
}

#[test]
fn sessions_alive_together_share_one_context() {
    let _serial = serial();
    let Some(first) = cuda_session() else {
        return;
    };
    let second = cuda_session().expect("a second session on the open context");
    assert_eq!(open_contexts(), 1, "two sessions opened two contexts");
    drop(first);
    assert_eq!(
        open_contexts(),
        1,
        "the context closed under a live session"
    );
    render(&second);
    drop(second);
    assert_eq!(open_contexts(), 0);

    let third = cuda_session().expect("a session after the context closed");
    assert_eq!(open_contexts(), 1, "no context reopened");
    render(&third);
    drop(third);
    assert_eq!(open_contexts(), 0);
}

/// `raster_pdf` and `render_channel` take their policy from
/// `PDF_RASTER_BACKEND`; the test runs itself again with it set.
#[test]
fn the_library_entry_points_free_the_context() {
    let _serial = serial();
    if !in_child_under("the_library_entry_points_free_the_context", "cuda") {
        return;
    }
    if cuda_session().is_none() {
        return;
    }
    assert_library_entry_points_close(open_contexts);
}
