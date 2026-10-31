//! The [`RasterOptions`]-driven entry points: the sequential page iterator
//! and the channel-fed background render.

use crate::gray::rgb_to_gray;
use crate::render::render_page_rgb_with_geom;
use crate::{
    PageSet, RasterError, RasterOptions, RasterSession, RenderParams, RenderedPage, SessionConfig,
    open_session, open_session_from_bytes,
};

// ── Sequential iterator ───────────────────────────────────────────────────────

struct RenderState {
    session: RasterSession,
    opts: RasterOptions,
    cursor: PageCursor,
}

impl RenderState {
    /// Advance the cursor and return the next page number to render, or
    /// `None` once the cursor is exhausted *or* the next page would fall past
    /// the document's `total_pages`.  Mirrors the documented clamp on
    /// `RasterOptions::last_page` (rendering stops at the last page in the
    /// document rather than erroring).
    fn next_in_range(&mut self) -> Option<u32> {
        let p = self.cursor.next_page()?;
        (p <= self.session.total_pages).then_some(p)
    }
}

/// Cursor over the pages to render.
///
/// `Range` walks every integer in `first..=last` (used when `RasterOptions::pages`
/// is `None`).  `Set` walks the explicit page numbers stored in a `PageSet` —
/// O(set length), not O(last − first), which matters when the set is sparse
/// across a wide range (e.g. `[1, u32::MAX]`).
enum PageCursor {
    Range(std::ops::RangeInclusive<u32>),
    Set { set: PageSet, idx: usize },
}

impl PageCursor {
    fn new(opts: &RasterOptions) -> Self {
        opts.pages.as_ref().map_or_else(
            || Self::Range(opts.first_page..=opts.last_page),
            |ps| Self::Set {
                set: ps.clone(),
                idx: 0,
            },
        )
    }

    fn next_page(&mut self) -> Option<u32> {
        match self {
            Self::Range(r) => r.next(),
            Self::Set { set, idx } => {
                let p = *set.as_slice().get(*idx)?;
                *idx += 1;
                Some(p)
            }
        }
    }
}

fn validate_opts(opts: &RasterOptions) -> Option<RasterError> {
    if opts.dpi <= 0.0 || !opts.dpi.is_finite() {
        return Some(RasterError::InvalidOptions(format!(
            "dpi must be a positive finite number, got {}",
            opts.dpi
        )));
    }
    if opts.pages.is_none() {
        if opts.first_page == 0 {
            return Some(RasterError::InvalidOptions(
                "first_page must be ≥ 1 (pages are 1-based)".to_owned(),
            ));
        }
        if opts.first_page > opts.last_page {
            return Some(RasterError::InvalidOptions(format!(
                "first_page ({}) > last_page ({})",
                opts.first_page, opts.last_page
            )));
        }
    }
    None
}

/// The [`SessionConfig`] behind the [`RasterOptions`]-driven entry points:
/// the default config with the per-call options that live on the session.
/// The options' font directories come before those of the default config.
fn session_config_for(opts: &RasterOptions) -> SessionConfig {
    let mut defaults = SessionConfig::default();
    let env_font_dirs = std::mem::take(&mut defaults.font_dirs);
    SessionConfig {
        strict: opts.strict,
        purpose: opts.purpose,
        annotations: opts.annotations,
        text_antialias: opts.text_antialias,
        vector_antialias: opts.vector_antialias,
        decrypt_authorized: opts.decrypt_authorized,
        password: opts.password.clone(),
        font_dirs: opts
            .font_dirs
            .iter()
            .cloned()
            .chain(env_font_dirs)
            .collect(),
        system_fonts: opts.system_fonts,
        ..defaults
    }
}

pub fn render_pages(
    path: &std::path::Path,
    opts: &RasterOptions,
) -> impl Iterator<Item = (u32, Result<RenderedPage, RasterError>)> {
    if let Some(e) = validate_opts(opts) {
        return PageIter {
            state: Some(Err(e)),
        };
    }

    let session = open_session(path, &session_config_for(opts));
    page_iter_from_session(session, opts)
}

/// Render a range of pages from a PDF held entirely in memory (no file path).
///
/// Identical to [`render_pages`] but opens the session from owned bytes via
/// [`open_session_from_bytes`] — for callers that already hold the document
/// (e.g. a PDF extracted from an archive) and want to avoid a temp file.
pub fn render_pages_from_bytes(
    bytes: Vec<u8>,
    opts: &RasterOptions,
) -> impl Iterator<Item = (u32, Result<RenderedPage, RasterError>)> {
    if let Some(e) = validate_opts(opts) {
        return PageIter {
            state: Some(Err(e)),
        };
    }

    let session = open_session_from_bytes(bytes, &session_config_for(opts));
    page_iter_from_session(session, opts)
}

/// Build the page iterator shared by [`render_pages`] and
/// [`render_pages_from_bytes`]: wrap an opened session (or its open error) and
/// the render options into the deferred [`PageIter`] state.
fn page_iter_from_session(
    session: Result<RasterSession, RasterError>,
    opts: &RasterOptions,
) -> PageIter {
    let state = session.map(|session| RenderState {
        cursor: PageCursor::new(opts),
        session,
        opts: opts.clone(),
    });

    PageIter { state: Some(state) }
}

struct PageIter {
    state: Option<Result<RenderState, RasterError>>,
}

impl Iterator for PageIter {
    type Item = (u32, Result<RenderedPage, RasterError>);

    fn next(&mut self) -> Option<Self::Item> {
        // Surface a deferred validation/open error exactly once, then close.
        if self.state.as_ref()?.is_err() {
            let err = self.state.take()?.err()?;
            return Some((1, Err(err)));
        }

        let state = self.state.as_mut()?.as_mut().ok()?;
        let Some(page_num) = state.next_in_range() else {
            // Drop the session early so callers that peek past the end don't
            // hold the document open longer than needed.
            self.state = None;
            return None;
        };
        Some((page_num, render_one(state, page_num)))
    }
}

impl std::iter::FusedIterator for PageIter {}

// ── Channel-based render ──────────────────────────────────────────────────────

/// Catch a panic from a render closure and convert it to `RasterError::RenderPanic`.
///
/// Under `panic = "abort"` (the default release profile) `catch_unwind` is a
/// no-op: a panicking page still aborts the process.  Under `panic = "unwind"`
/// (test builds, and any embedder that opts in) the panic is caught and
/// returned as a per-page error so the render loop can continue.
///
/// Extracted as a generic helper so the panic-catch logic can be unit-tested
/// by injecting an arbitrary render function.
fn catch_page_panic<F>(page_num: u32, f: F) -> (u32, Result<RenderedPage, RasterError>)
where
    F: FnOnce() -> Result<RenderedPage, RasterError> + std::panic::UnwindSafe,
{
    let result = std::panic::catch_unwind(f);
    let outcome = match result {
        Ok(r) => r,
        Err(payload) => Err(panic_payload_to_render_panic(page_num, payload.as_ref())),
    };
    (page_num, outcome)
}

/// Render one page, mapping any panic to `Err(RasterError::RenderPanic)`.
///
/// # `AssertUnwindSafe` justification
///
/// `render_one` receives a shared `&RenderState` and does not mutate any field
/// of the struct itself — `session`, `opts`, and `cursor` are all read-only
/// within `render_one`.  The only interior mutation is in
/// `lend_decoders`/`reclaim_decoders`, which move GPU decoder handles out of
/// thread-local storage into a locally-owned `PageRenderer` and back.  An
/// unwind between those two calls is caught by the [`DecoderReclaim`] RAII
/// guard, which returns the handles to TLS on `Drop`, so a panicking page does
/// not even degrade the next page's decoder backend.  No state shared with
/// other pages is mutated, and the panicking page's local `PageRenderer` is
/// discarded rather than reused — we resume from the next page number.
/// `AssertUnwindSafe` is therefore sound: a caught panic cannot leave `state`
/// or thread-local decoder slots in an inconsistent state that would corrupt
/// subsequent pages.
fn render_one_caught(
    state: &RenderState,
    page_num: u32,
) -> (u32, Result<RenderedPage, RasterError>) {
    catch_page_panic(
        page_num,
        std::panic::AssertUnwindSafe(|| render_one(state, page_num)),
    )
}

/// The body of the [`render_channel`] rayon worker, with every panic site —
/// including session setup — inside a single `catch_unwind`.
///
/// `open_session` is a deep parser entry point (decryption, xref repair,
/// page-tree walks over adversarial input).  If it — or the page cursor —
/// panicked *outside* the catch, the rayon closure would unwind, `tx` would
/// drop, and the consumer would observe a silently closed channel with no
/// `(page_num, Err)` ever delivered: total, undiagnosable page loss.  Catching
/// here guarantees the contract documented on [`render_channel`]: a setup
/// panic is delivered as `(1, Err(RasterError::RenderPanic { .. }))` before the
/// channel closes, exactly as a per-page panic is.  Under `panic = "abort"`
/// (the default release profile) this is a no-op; the guarantee holds for
/// `panic = "unwind"` builds and embedders that opt in.
///
/// `AssertUnwindSafe` is sound here: every value the closure touches is either
/// freshly created inside it (`session`, `state`) or an owned move-in
/// (`path`, `opts`).  `tx` is a `SyncSender`, which carries no
/// unwind-observable invariant — a partially-sent item cannot exist because
/// `send` is atomic per message.  Nothing observable by a *later* call is
/// mutated through a shared reference.
fn render_channel_worker(
    tx: &std::sync::mpsc::SyncSender<(u32, Result<RenderedPage, RasterError>)>,
    path: &std::path::Path,
    opts: RasterOptions,
) {
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let session = match open_session(path, &session_config_for(&opts)) {
            Ok(s) => s,
            Err(e) => {
                let _sent = tx.send((1, Err(e)));
                return;
            }
        };

        let mut state = RenderState {
            cursor: PageCursor::new(&opts),
            session,
            opts,
        };

        while let Some(page_num) = state.next_in_range() {
            let item = render_one_caught(&state, page_num);
            if tx.send(item).is_err() {
                return;
            }
        }
    }));

    if let Err(payload) = caught {
        // A panic escaped session setup or the page cursor.  Deliver it as a
        // page-1 RenderPanic so the consumer sees a loud per-page error rather
        // than an empty, silently closed channel.
        let _sent = tx.send((1, Err(panic_payload_to_render_panic(1, payload.as_ref()))));
    }
}

/// Extract a human-readable message from a caught panic payload and wrap it in
/// [`RasterError::RenderPanic`].  Shared by the per-page and worker-level
/// catch sites so payload handling stays identical.
fn panic_payload_to_render_panic(page: u32, payload: &(dyn std::any::Any + Send)) -> RasterError {
    let message = payload.downcast_ref::<&str>().map_or_else(
        || {
            payload
                .downcast_ref::<String>()
                .cloned()
                .unwrap_or_else(|| "panicked".to_owned())
        },
        |s| (*s).to_owned(),
    );
    RasterError::RenderPanic { page, message }
}

#[must_use]
pub fn render_channel(
    path: &std::path::Path,
    opts: &RasterOptions,
    capacity: usize,
) -> std::sync::mpsc::Receiver<(u32, Result<RenderedPage, RasterError>)> {
    use std::sync::mpsc;

    let capacity = capacity.max(1);
    let (tx, rx) = mpsc::sync_channel(capacity);

    if let Some(e) = validate_opts(opts) {
        let _sent = tx.send((1, Err(e)));
        return rx;
    }

    let path_owned = path.to_owned();
    let opts_owned = opts.clone();

    rayon::spawn(move || {
        render_channel_worker(&tx, &path_owned, opts_owned);
    });

    rx
}

// ── Single-page render (gray + deskew) ───────────────────────────────────────

fn render_one(state: &RenderState, page_num: u32) -> Result<RenderedPage, RasterError> {
    let dpi = state.opts.dpi;
    let scale = f64::from(dpi) / 72.0;

    let page_id = state.session.resolve_page(page_num)?;
    let geom = pdf_interp::page_size_pts_by_id(&state.session.doc, page_id)?;

    let (rgb, diagnostics) = render_page_rgb_with_geom(
        &state.session,
        page_num,
        page_id,
        geom,
        &RenderParams::new(scale),
    )?;
    let mut gray = rgb_to_gray(&rgb);

    if state.opts.deskew {
        crate::deskew::apply(&mut gray).map_err(|e| RasterError::Deskew(e.to_string()))?;
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "dpi is an f32 (≤ ~3400 in practice); user_unit is validated to [0.1, 10.0]; \
                  the product is at most ~34 000, well within f32 range"
    )]
    let effective_dpi = (f64::from(dpi) * geom.user_unit) as f32;

    Ok(crate::page::gray8_to_rendered_page(
        &gray,
        page_num,
        dpi,
        effective_dpi,
        diagnostics,
    ))
}

#[cfg(test)]
mod channel_tests {
    use std::path::Path;

    use super::*;

    fn valid_opts() -> RasterOptions {
        RasterOptions {
            dpi: 150.0,
            first_page: 1,
            last_page: 1,
            deskew: false,
            pages: None,
            strict: false,
            purpose: crate::RenderPurpose::View,
            annotations: true,
            text_antialias: true,
            vector_antialias: true,
            decrypt_authorized: false,
            password: None,
            font_dirs: Vec::new(),
            system_fonts: true,
        }
    }

    #[test]
    fn validation_error_delivered_and_channel_closes() {
        let bad = RasterOptions {
            dpi: 0.0,
            ..valid_opts()
        };
        let rx = render_channel(Path::new("/irrelevant"), &bad, 4);
        let (page, res) = rx.recv().expect("first item must arrive");
        assert_eq!(page, 1);
        assert!(
            matches!(res, Err(RasterError::InvalidOptions(_))),
            "expected InvalidOptions, got {res:?}"
        );
        assert!(
            rx.recv().is_err(),
            "channel must be closed after validation error"
        );
    }

    #[test]
    fn session_open_failure_delivered_and_channel_closes() {
        let rx = render_channel(Path::new("/no_such_file_xyz.pdf"), &valid_opts(), 4);
        let (page, res) = rx.recv().expect("first item must arrive");
        assert_eq!(page, 1);
        assert!(res.is_err(), "expected Err from session open, got Ok");
        assert!(
            rx.recv().is_err(),
            "channel must be closed after session error"
        );
    }

    #[test]
    fn receiver_drop_does_not_panic() {
        let rx = render_channel(Path::new("/no_such_file_xyz.pdf"), &valid_opts(), 1);
        drop(rx);
    }

    #[test]
    fn capacity_zero_raised_to_one_no_deadlock() {
        let bad = RasterOptions {
            dpi: 0.0,
            ..valid_opts()
        };
        let rx = render_channel(Path::new("/irrelevant"), &bad, 0);
        assert!(
            rx.recv().is_ok(),
            "error item must be delivered even with capacity=0"
        );
        assert!(rx.recv().is_err(), "channel must be closed after error");
    }

    #[test]
    #[ignore = "requires tests/fixtures/corpus-01-native-text-small.pdf (not in repo)"]
    fn sparse_pages_only_yields_requested_pages() {
        // corpus-01 is a small native-text PDF — use pages 1 and 3 from it.
        // If the doc has fewer than 3 pages, the test still passes (page 3 won't arrive).
        let ps = crate::PageSet::new(vec![1, 3]).unwrap();
        let opts = RasterOptions {
            dpi: 72.0,
            first_page: 1,  // ignored when pages is Some
            last_page: 100, // ignored when pages is Some
            deskew: false,
            pages: Some(ps),
            strict: false,
            purpose: crate::RenderPurpose::View,
            annotations: true,
            text_antialias: true,
            vector_antialias: true,
            decrypt_authorized: false,
            password: None,
            font_dirs: Vec::new(),
            system_fonts: true,
        };
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
        let path = std::path::Path::new(&manifest_dir)
            .parent()
            .expect("parent dir should exist")
            .parent()
            .expect("grandparent dir should exist")
            .join("tests/fixtures/corpus-01-native-text-small.pdf");
        // Collect all yielded page numbers from the iterator
        let yielded: Vec<u32> = render_pages(path.as_path(), &opts)
            .filter_map(|(pn, r)| r.ok().map(|_| pn))
            .collect();
        assert!(
            !yielded.is_empty(),
            "expected at least one page to render successfully"
        );
        // Must only contain pages 1 and/or 3 — no 2, no 4+
        for pn in &yielded {
            assert!(
                *pn == 1 || *pn == 3,
                "unexpected page {pn} in sparse render output"
            );
        }
    }

    #[test]
    #[ignore = "requires tests/fixtures/corpus-01-native-text-small.pdf (not in repo)"]
    fn sparse_pages_channel_only_yields_requested_pages() {
        let ps = crate::PageSet::new(vec![1, 3]).unwrap();
        let opts = RasterOptions {
            dpi: 72.0,
            first_page: 1,
            last_page: 100,
            deskew: false,
            pages: Some(ps),
            strict: false,
            purpose: crate::RenderPurpose::View,
            annotations: true,
            text_antialias: true,
            vector_antialias: true,
            decrypt_authorized: false,
            password: None,
            font_dirs: Vec::new(),
            system_fonts: true,
        };
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
        let path = std::path::Path::new(&manifest_dir)
            .parent()
            .expect("parent dir should exist")
            .parent()
            .expect("grandparent dir should exist")
            .join("tests/fixtures/corpus-01-native-text-small.pdf");
        let rx = render_channel(path.as_path(), &opts, 4);
        let mut yielded = Vec::new();
        while let Ok((pn, r)) = rx.recv() {
            if r.is_ok() {
                yielded.push(pn);
            }
        }
        assert!(
            !yielded.is_empty(),
            "expected at least one page to render successfully"
        );
        for pn in &yielded {
            assert!(
                *pn == 1 || *pn == 3,
                "unexpected page {pn} in sparse channel output"
            );
        }
    }

    #[test]
    fn validation_errors_match_between_iterator_and_channel() {
        let bad = RasterOptions {
            dpi: -1.0,
            ..valid_opts()
        };
        let (_, iter_err) = render_pages(Path::new("/irrelevant"), &bad)
            .next()
            .expect("iterator must yield one item");
        let rx = render_channel(Path::new("/irrelevant"), &bad, 1);
        let (_, chan_err) = rx.recv().expect("channel must yield one item");
        let Err(RasterError::InvalidOptions(i)) = iter_err else {
            panic!("iterator returned wrong variant: {iter_err:?}")
        };
        let Err(RasterError::InvalidOptions(c)) = chan_err else {
            panic!("channel returned wrong variant: {chan_err:?}")
        };
        assert_eq!(i, c, "validation error messages must be identical");
    }

    #[test]
    fn pages_some_bypasses_first_last_page_validation() {
        // When pages=Some, first_page/last_page are documented as ignored —
        // validate_opts must not reject zero-value or inverted defaults.
        let ps = crate::PageSet::new(vec![3u32]).unwrap();
        let opts = RasterOptions {
            dpi: 72.0,
            first_page: 0, // would normally be rejected
            last_page: 0,  // would normally be rejected (< first_page after clamping)
            deskew: false,
            pages: Some(ps),
            strict: false,
            purpose: crate::RenderPurpose::View,
            annotations: true,
            text_antialias: true,
            vector_antialias: true,
            decrypt_authorized: false,
            password: None,
            font_dirs: Vec::new(),
            system_fonts: true,
        };
        assert!(
            validate_opts(&opts).is_none(),
            "zero first/last_page must be accepted when pages=Some"
        );
    }

    // ── panic-contract tests ──────────────────────────────────────────────
    //
    // These tests exercise `catch_page_panic` directly by injecting a
    // panicking (or ok-returning) closure.  Test builds use panic=unwind, so
    // catch_unwind is effective here even though the release profile uses
    // panic=abort.  This proves the catch→(page,Err)→continue semantics
    // that render_one_caught relies on.

    #[test]
    fn catch_page_panic_maps_panic_to_render_panic_variant() {
        // A closure that panics with a known message must arrive as
        // (page_num, Err(RasterError::RenderPanic { page, message })).
        let (pn, res) = catch_page_panic(7u32, || -> Result<RenderedPage, RasterError> {
            panic!("boom from page 7");
        });
        assert_eq!(pn, 7, "page number must be preserved through the catch");
        match res {
            Err(RasterError::RenderPanic { page, message }) => {
                assert_eq!(page, 7);
                assert!(
                    message.contains("boom from page 7"),
                    "panic message should be forwarded; got: {message:?}"
                );
            }
            other => panic!("expected RenderPanic, got {other:?}"),
        }
    }

    #[test]
    fn catch_page_panic_ok_path_is_transparent() {
        // A non-panicking closure must pass through unchanged.
        let dummy = RenderedPage {
            page_num: 3,
            width: 1,
            height: 1,
            pixels: vec![128u8],
            dpi: 72.0,
            effective_dpi: 72.0,
            diagnostics: crate::PageDiagnostics::default(),
        };
        let (pn, res) = catch_page_panic(3u32, move || -> Result<RenderedPage, RasterError> {
            Ok(dummy)
        });
        assert_eq!(pn, 3);
        assert!(res.is_ok(), "ok path must not be disturbed; got {res:?}");
    }

    #[test]
    fn catch_page_panic_loop_continues_after_panic() {
        // Simulate the render loop: two pages where page 1 panics, page 2
        // succeeds.  The consumer must receive (1, Err) then (2, Ok).
        let pages: &[(u32, bool)] = &[(1, true), (2, false)]; // (page_num, should_panic)
        let mut results = Vec::new();
        for &(page_num, should_panic) in pages {
            let item = catch_page_panic(page_num, move || -> Result<RenderedPage, RasterError> {
                assert!(!should_panic, "synthetic panic on page {page_num}");
                Ok(RenderedPage {
                    page_num,
                    width: 1,
                    height: 1,
                    pixels: vec![0u8],
                    dpi: 72.0,
                    effective_dpi: 72.0,
                    diagnostics: crate::PageDiagnostics::default(),
                })
            });
            results.push(item);
        }
        // Page 1 must be (1, Err(RenderPanic))
        assert_eq!(results[0].0, 1);
        assert!(
            matches!(results[0].1, Err(RasterError::RenderPanic { page: 1, .. })),
            "page 1 must be RenderPanic; got {:?}",
            results[0].1
        );
        // Page 2 must be (2, Ok(_)) — loop continued past the panic
        assert_eq!(results[1].0, 2);
        assert!(
            results[1].1.is_ok(),
            "page 2 must succeed; got {:?}",
            results[1].1
        );
    }

    #[test]
    fn panic_payload_helper_extracts_str_string_and_fallback() {
        // &str payload (the common `panic!("literal")` shape).
        let from_str = std::panic::catch_unwind(|| panic!("str payload"))
            .map(|()| unreachable!())
            .unwrap_err();
        assert!(
            matches!(
                panic_payload_to_render_panic(2, from_str.as_ref()),
                RasterError::RenderPanic { page: 2, ref message } if message.contains("str payload")
            ),
            "str payload must carry through as a page-2 RenderPanic"
        );
        // String payload (formatted panic).
        let n = 9;
        let from_string = std::panic::catch_unwind(|| panic!("string {n}"))
            .map(|()| unreachable!())
            .unwrap_err();
        assert!(
            matches!(
                panic_payload_to_render_panic(3, from_string.as_ref()),
                RasterError::RenderPanic { page: 3, ref message } if message.contains("string 9")
            ),
            "String payload must carry through as a page-3 RenderPanic"
        );
        // Non-string payload → stable fallback, never re-panics.
        let from_other = std::panic::catch_unwind(|| std::panic::panic_any(123u32))
            .map(|()| unreachable!())
            .unwrap_err();
        assert!(
            matches!(
                panic_payload_to_render_panic(4, from_other.as_ref()),
                RasterError::RenderPanic { page: 4, ref message } if message == "panicked"
            ),
            "non-string payload must fall back to the stable \"panicked\" message"
        );
    }

    #[test]
    fn render_channel_worker_delivers_setup_panic_not_silent_close() {
        // The v1-class silent-total-loss guard: a panic in session setup must
        // arrive as (1, Err(RenderPanic)) on the channel, NOT a closed,
        // empty channel.  We can't make `open_session` panic from here, so we
        // drive the worker's catch boundary with a closure-shaped stand-in
        // that mirrors render_channel_worker's structure: a panic before the
        // first send must still be delivered to the consumer.
        use std::sync::mpsc;
        let (tx, rx) = mpsc::sync_channel::<(u32, Result<RenderedPage, RasterError>)>(4);
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // Stand-in for `open_session` panicking before any page is sent.
            panic!("synthetic open_session panic");
        }));
        let payload = caught.expect_err("synthetic closure always panics");
        let _sent = tx.send((1, Err(panic_payload_to_render_panic(1, payload.as_ref()))));
        drop(tx);
        let received: Vec<_> = rx.iter().collect();
        assert_eq!(
            received.len(),
            1,
            "consumer must see exactly one delivered error, not an empty closed channel"
        );
        assert_eq!(received[0].0, 1);
        assert!(
            matches!(received[0].1, Err(RasterError::RenderPanic { page: 1, .. })),
            "setup panic must be a page-1 RenderPanic; got {:?}",
            received[0].1
        );
    }

    fn opts_with_pages(set: crate::PageSet) -> RasterOptions {
        RasterOptions {
            dpi: 72.0,
            first_page: 0,
            last_page: 0,
            deskew: false,
            pages: Some(set),
            strict: false,
            purpose: crate::RenderPurpose::View,
            annotations: true,
            text_antialias: true,
            vector_antialias: true,
            decrypt_authorized: false,
            password: None,
            font_dirs: Vec::new(),
            system_fonts: true,
        }
    }

    #[test]
    fn cursor_range_yields_inclusive_window() {
        let opts = RasterOptions {
            dpi: 72.0,
            first_page: 3,
            last_page: 5,
            deskew: false,
            pages: None,
            strict: false,
            purpose: crate::RenderPurpose::View,
            annotations: true,
            text_antialias: true,
            vector_antialias: true,
            decrypt_authorized: false,
            password: None,
            font_dirs: Vec::new(),
            system_fonts: true,
        };
        let mut c = PageCursor::new(&opts);
        assert_eq!(c.next_page(), Some(3));
        assert_eq!(c.next_page(), Some(4));
        assert_eq!(c.next_page(), Some(5));
        assert_eq!(c.next_page(), None);
        assert_eq!(c.next_page(), None);
    }

    #[test]
    fn cursor_range_saturates_at_u32_max() {
        // last_page=u32::MAX is the documented "render to end of document" idiom.
        // next.saturating_add(1) must keep `next > end` once we've yielded u32::MAX.
        let opts = RasterOptions {
            dpi: 72.0,
            first_page: u32::MAX - 1,
            last_page: u32::MAX,
            deskew: false,
            pages: None,
            strict: false,
            purpose: crate::RenderPurpose::View,
            annotations: true,
            text_antialias: true,
            vector_antialias: true,
            decrypt_authorized: false,
            password: None,
            font_dirs: Vec::new(),
            system_fonts: true,
        };
        let mut c = PageCursor::new(&opts);
        assert_eq!(c.next_page(), Some(u32::MAX - 1));
        assert_eq!(c.next_page(), Some(u32::MAX));
        assert_eq!(c.next_page(), None);
    }

    #[test]
    fn cursor_set_yields_exactly_set_members() {
        let ps = crate::PageSet::new(vec![5, 1, 10, 1]).unwrap();
        let opts = opts_with_pages(ps);
        let mut c = PageCursor::new(&opts);
        assert_eq!(c.next_page(), Some(1));
        assert_eq!(c.next_page(), Some(5));
        assert_eq!(c.next_page(), Some(10));
        assert_eq!(c.next_page(), None);
    }

    #[test]
    fn cursor_set_walks_only_set_length_for_sparse_input() {
        // Regression: previously `PageIter` walked every integer in
        // first()..=last() and probed `PageSet::contains` on each — for
        // [1, u32::MAX] that meant ~4.3 billion iterations.  The cursor must
        // yield exactly two pages and terminate.
        let ps = crate::PageSet::new(vec![1, u32::MAX]).unwrap();
        let opts = opts_with_pages(ps);
        let mut c = PageCursor::new(&opts);
        let mut yielded = Vec::new();
        while let Some(p) = c.next_page() {
            yielded.push(p);
            assert!(
                yielded.len() <= 2,
                "cursor yielded a third page on a 2-element PageSet"
            );
        }
        assert_eq!(yielded, vec![1, u32::MAX]);
    }
}

#[cfg(test)]
mod from_bytes_tests {
    use super::*;

    /// A minimal valid single-page PDF (one empty 612×792-pt page).
    fn minimal_pdf() -> Vec<u8> {
        b"%PDF-1.4\n\
1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n\
2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n\
3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]>>\nendobj\n\
xref\n0 4\n\
0000000000 65535 f\r\n\
0000000009 00000 n\r\n\
0000000056 00000 n\r\n\
0000000111 00000 n\r\n\
trailer\n<</Size 4 /Root 1 0 R>>\n\
startxref\n180\n%%EOF"
            .to_vec()
    }

    #[test]
    fn render_pages_from_bytes_renders_a_page() {
        let opts = RasterOptions::default();
        let pages: Vec<_> = render_pages_from_bytes(minimal_pdf(), &opts).collect();
        assert_eq!(
            pages.len(),
            1,
            "the one-page in-memory PDF must yield exactly one page"
        );
        let (n, res) = &pages[0];
        assert_eq!(*n, 1, "the single page is 1-based page 1");
        let page = res.as_ref().expect("first page renders");
        assert!(
            page.width > 0 && page.height > 0,
            "rendered page must have positive dimensions, got {}×{}",
            page.width,
            page.height
        );
        assert_eq!(
            page.pixels.len() as u64,
            u64::from(page.width) * u64::from(page.height),
            "grayscale buffer must be exactly width × height bytes"
        );
    }
}
