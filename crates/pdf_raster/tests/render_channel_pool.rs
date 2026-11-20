//! `render_channel` never parks a rayon worker on its bounded channel, so a
//! consumer that drains it from inside a pool, or several channels whose
//! pages run parallel work of their own (deskew's angle sweep), always make
//! progress.  Each scenario runs on a thread of its own and is joined with a
//! deadline: a deadlock fails the test instead of hanging the suite.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;

use rasterrocket::{RasterError, RasterOptions, RenderedPage, render_channel};
use rayon::prelude::*;

/// Longest any scenario may take; every one finishes in a few seconds.
const DEADLINE: Duration = Duration::from_secs(120);

/// Run `scenario` on a thread of its own and return its result, failing the
/// test when it has not finished within [`DEADLINE`].  A hung scenario's
/// thread is left behind; the test process ends without it.
fn within_deadline<T: Send + 'static>(
    what: &str,
    scenario: impl FnOnce() -> T + Send + 'static,
) -> T {
    let (tx, rx) = mpsc::channel();
    let thread = std::thread::Builder::new()
        .name(format!("scenario-{what}"))
        .spawn(move || {
            let _ = tx.send(scenario());
        })
        .expect("spawn the scenario thread");
    match rx.recv_timeout(DEADLINE) {
        Ok(result) => result,
        Err(mpsc::RecvTimeoutError::Timeout) => {
            panic!("{what}: no result within {DEADLINE:?} (deadlock)")
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => match thread.join() {
            Err(payload) => std::panic::resume_unwind(payload),
            Ok(()) => panic!("{what}: the scenario ended without a result"),
        },
    }
}

/// A PDF of `pages` pages, 320 × 200 pt each, drawing three slanted bars
/// (a skew the deskew sweep has to search for).
fn bars_pdf(pages: usize) -> Vec<u8> {
    let content: &[u8] = b"0 0 0 rg 10 10 m 300 40 l 300 44 l 10 14 l f \
        10 60 m 300 90 l 300 94 l 10 64 l f 10 110 m 300 140 l 300 144 l 10 114 l f";
    let kids: Vec<String> = (0..pages).map(|i| format!("{} 0 R", 3 + 2 * i)).collect();
    let mut objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        format!(
            "<< /Type /Pages /Kids [{}] /Count {pages} >>",
            kids.join(" ")
        )
        .into_bytes(),
    ];
    for i in 0..pages {
        objects.push(
            format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 320 200] /Contents {} 0 R >>",
                4 + 2 * i
            )
            .into_bytes(),
        );
        objects.push(common::stream_object("", content));
    }
    common::assemble_pdf(&objects)
}

/// `bars_pdf(pages)` written to a scratch file named after `tag`.
fn scratch_pdf(tag: &str, pages: usize) -> PathBuf {
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "render_channel_pool_{tag}_{}.pdf",
        std::process::id()
    ));
    std::fs::write(&path, bars_pdf(pages)).expect("write the scratch PDF");
    path
}

/// The page numbers `rx` delivers, every page of which must render.
fn received(rx: &mpsc::Receiver<(u32, Result<RenderedPage, RasterError>)>) -> Vec<u32> {
    rx.iter()
        .map(|(page, rendered)| {
            let _page = rendered.unwrap_or_else(|e| panic!("page {page}: {e}"));
            page
        })
        .collect()
}

/// The pages of a channel on `path`, drained to its end.
fn drain(path: &Path, opts: &RasterOptions, capacity: usize) -> Vec<u32> {
    received(&render_channel(path, opts, capacity))
}

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("build the rayon pool")
}

/// Every worker of a pool drains a channel of its own (twice as many
/// channels as workers), and every page of every channel arrives in order.
#[test]
fn consumers_on_every_pool_worker_make_progress() {
    let path = scratch_pdf("workers", 3);
    let pages = within_deadline("pool consumers", {
        let path = path.clone();
        move || {
            let pool = pool(2);
            let opts = RasterOptions::new(72.0);
            pool.install(|| {
                (0..2 * pool.current_num_threads())
                    .into_par_iter()
                    .map(|_| drain(&path, &opts, 1))
                    .collect::<Vec<_>>()
            })
        }
    });
    assert_eq!(pages.len(), 4);
    for received in pages {
        assert_eq!(received, [1, 2, 3]);
    }
    let _ = std::fs::remove_file(&path);
}

/// A consumer on the only worker of a one-thread pool, as under
/// `RAYON_NUM_THREADS=1`, receives every page.
#[test]
fn a_consumer_on_a_one_thread_pool_makes_progress() {
    let path = scratch_pdf("single", 3);
    let pages = within_deadline("one-thread pool", {
        let path = path.clone();
        move || {
            let pool = pool(1);
            let opts = RasterOptions::new(72.0);
            pool.install(|| drain(&path, &opts, 1))
        }
    });
    assert_eq!(pages, [1, 2, 3]);
    let _ = std::fs::remove_file(&path);
}

/// Many deskewed channels, opened from a pool worker and drained one after
/// another off the pool: a page's parallel angle sweep never buries the
/// render of the channel being drained under one blocked on its own
/// channel.
#[test]
fn deskewed_channels_drained_in_turn_make_progress() {
    let path = scratch_pdf("deskew", 6);
    let pages = within_deadline("deskewed channels", {
        let path = path.clone();
        move || {
            let pool = pool(2);
            let mut opts = RasterOptions::new(144.0);
            opts.deskew = true;
            let channels: Vec<_> =
                pool.install(|| (0..12).map(|_| render_channel(&path, &opts, 1)).collect());
            channels.iter().map(received).collect::<Vec<_>>()
        }
    });
    assert_eq!(pages.len(), 12);
    for pages in pages {
        assert_eq!(pages, [1, 2, 3, 4, 5, 6]);
    }
    let _ = std::fs::remove_file(&path);
}

/// Every worker of rayon's global pool drains a deskewed channel of its
/// own (twice as many channels as workers): the pages' angle sweeps run
/// on no pool a consumer occupies, so every page arrives.
#[test]
fn deskewed_channels_drained_on_every_global_worker_make_progress() {
    let path = scratch_pdf("global", 3);
    let pages = within_deadline("global-pool consumers", {
        let path = path.clone();
        move || {
            let mut opts = RasterOptions::new(144.0);
            opts.deskew = true;
            (0..2 * rayon::current_num_threads())
                .into_par_iter()
                .map(|_| drain(&path, &opts, 1))
                .collect::<Vec<_>>()
        }
    });
    assert_eq!(pages.len(), 2 * rayon::current_num_threads());
    for pages in pages {
        assert_eq!(pages, [1, 2, 3]);
    }
    let _ = std::fs::remove_file(&path);
}
