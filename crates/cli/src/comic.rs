//! Comic-archive (.cbz/.cb7/.cbt/.cbr) rendering entry for the CLI.
//!
//! Bridges the comic crate's `open_comic` into the same per-page output sink the
//! PDF path uses, so output format, colour mode, naming, and page-range behave
//! identically across input types.

use std::path::Path;

use color::{Gray8, Gray16};
use comic::{ComicError, ComicOptions, RenderedPage, open_comic};
use raster::Bitmap;

use crate::args::Args;
use crate::cleanup::Page;
use crate::render::{self, RenderError};

/// Render a comic archive, writing each decodable page via the shared encoder;
/// an embedded PDF opens with `session`, the PDF path's session settings.
///
/// Returns `(written, failed)`: the count of pages successfully written and the
/// count of pages that decoded-or-rendered-but-could-not-be-written. Per-page
/// errors are reported to stderr as they occur and skipped (one bad page never
/// aborts the rest); the caller maps a non-zero `failed` to a non-zero exit code
/// so a partial failure is observable via `$?`, matching the PDF path.
/// Whole-archive failures surface as the outer `Err`, whose `Display` carries the user-facing
/// message.
pub(crate) fn run(
    args: &Args,
    spill: &crate::ram::SpillPolicy,
    session: pdf_raster::SessionConfig,
) -> Result<(usize, usize), ComicError> {
    // Comics carry no physical resolution, so the user's requested DPI rides
    // along on each RenderedPage purely for downstream OCR feature scaling;
    // it never changes the already-rasterised pixel dimensions. When the x
    // and y DPI differ, x is used (documented behaviour for resolution-less
    // input).
    #[expect(
        clippy::cast_possible_truncation,
        reason = "DPI is a small positive value; f64→f32 is exact in practice"
    )]
    let dpi = args.x_dpi() as f32;
    // The default page window decodes every page; page SELECTION (window,
    // odd/even, single) is done once, below, by the same selector the PDF
    // path uses. Windowing here too would double-apply first/last against a
    // shrunken total.
    let mut opts = ComicOptions::new(dpi);
    opts.session = session;
    opts.depth = args.depth();
    if let Some(mib) = args.max_archive_mib {
        opts.max_archive_bytes = mib << 20;
    }

    let mut pages = open_comic(Path::new(args.input()), &opts)?;

    // A comic with more pages than `i32::MAX` cannot be page-listed or named;
    // fail hard rather than silently mis-number, matching the PDF path.
    let total = i32::try_from(pages.len()).map_err(|_| {
        ComicError::BadArchive(format!("archive has too many pages ({})", pages.len()))
    })?;

    // Decide which page numbers to write via the SAME selector the PDF path uses
    // (`session.total_pages()` there, the full decoded count here), so
    // `--first-page`/`--last-page`/`--odd`/`--even`/`--single` and the clamp
    // warnings behave identically across input types.
    let (selected, warnings) = args
        .build_page_list(total)
        .map_err(ComicError::BadArchive)?;
    for w in &warnings {
        eprintln!("rrocket: warning: {w}");
    }
    // Drop the pages the selector excluded (odd/even/single/window) before
    // any is decoded; `selected` is ascending.
    pages.retain(|page| i32::try_from(page).is_ok_and(|p| selected.binary_search(&p).is_ok()));

    let pool = crate::render_pool(args.num_threads)
        .map_err(|e| ComicError::BadArchive(format!("cannot start the render thread pool: {e}")))?;
    let (mut written, mut failed) = (0usize, 0usize);
    render_in_order(
        &mut pages,
        &pool,
        |job| {
            let page_num = job.page();
            let outcome = job
                .render()
                .map_err(|e| e.to_string())
                .and_then(|page| write_page(args, spill, page, total).map_err(|e| e.to_string()));
            (page_num, outcome)
        },
        |(page_num, outcome)| match outcome {
            Ok(()) => written += 1,
            Err(e) => {
                eprintln!("rrocket: page {page_num}: {e}");
                failed += 1;
            }
        },
    );
    // As on the PDF path: each worker's GPU state (an embedded PDF's) is
    // destroyed while its session is alive.
    let _ = pool.broadcast(|_| pdf_raster::release_gpu_thread_state());
    Ok((written, failed))
}

/// How many pages per pool thread may be taken past the first page whose
/// outcome has not been reported.
const PAGES_AHEAD_PER_THREAD: usize = 2;

/// Run `work` on every remaining page of `pages` on `pool`, and hand each
/// outcome to `report` on the calling thread in reading order.  A page is
/// taken only while it lies fewer than [`PAGES_AHEAD_PER_THREAD`] × threads
/// pages past the first unreported one, which bounds the pages (and their
/// pixel buffers) held at once.  Once a page's `work` unwinds, no further
/// page is taken or reported, and the pool's scope resumes the unwind when
/// the pages in flight have finished.
fn render_in_order<T: Send>(
    pages: &mut comic::ComicPages,
    pool: &rayon::ThreadPool,
    work: impl Fn(comic::PageJob) -> T + Sync,
    mut report: impl FnMut(T),
) {
    /// Sends its slot's outcome on drop: `None` if `work` unwound.
    struct Slot<T> {
        index: usize,
        outcome: Option<T>,
        tx: std::sync::mpsc::Sender<(usize, Option<T>)>,
    }
    impl<T> Drop for Slot<T> {
        fn drop(&mut self) {
            let _ = self.tx.send((self.index, self.outcome.take()));
        }
    }

    let ahead = pool.current_num_threads().max(1) * PAGES_AHEAD_PER_THREAD;
    let work = &work;
    pool.in_place_scope(|scope| {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut ready = std::collections::BTreeMap::new();
        let (mut started, mut reported) = (0usize, 0usize);
        loop {
            while started < reported + ahead {
                let Some(job) = pages.next_job() else { break };
                let tx = tx.clone();
                let index = started;
                scope.spawn(move |_| {
                    let mut slot = Slot {
                        index,
                        outcome: None,
                        tx,
                    };
                    slot.outcome = Some(work(job));
                });
                started += 1;
            }
            if reported == started {
                return;
            }
            // Every spawned slot sends exactly once, and `tx` outlives the
            // loop, so `recv` returns while `reported < started`.
            let Ok((index, Some(outcome))) = rx.recv() else {
                return;
            };
            let _ = ready.insert(index, outcome);
            while let Some(outcome) = ready.remove(&reported) {
                report(outcome);
                reported += 1;
            }
        }
    });
}

/// Write the 8- or 16-bit luma [`RenderedPage`] `page` through the shared page writer
/// ([`render::write_cleaned`]), which applies the camera stages, the tone
/// operations and the output format exactly as the PDF path does. The page's
/// packed pixel buffer becomes the bitmap's buffer.
fn write_page(
    args: &Args,
    spill: &crate::ram::SpillPolicy,
    page: RenderedPage,
    total_pages: i32,
) -> Result<(), RenderError> {
    // Page numbers come from `build_page_list`, bounded by `total_pages`.
    let page_i32 =
        i32::try_from(page.page_num).expect("comic page numbers are bounded by the i32 page count");
    let sink = render::Sink::Files {
        spill,
        format: args.output_format(),
    };
    let (width, height) = (page.width, page.height);
    let luma = match page.pixels {
        pdf_raster::Pixels::U8(pixels) => {
            Page::Gray(Bitmap::<Gray8>::from_packed(width, height, pixels))
        }
        pdf_raster::Pixels::U16(samples) => {
            Page::Gray16(Bitmap::<Gray16>::from_packed(width, height, samples))
        }
    };
    render::write_cleaned(luma, args, sink, page_i32, total_pages)
}
