//! Comic-archive (.cbz/.cb7/.cbt/.cbr) rendering entry for the CLI.
//!
//! Bridges the comic crate's `open_comic` into the same per-page output sink the
//! PDF path uses, so output format, colour mode, naming, and page-range behave
//! identically across input types.

use std::ops::ControlFlow;
use std::path::Path;

use color::{Gray8, Gray16};
use comic::{ComicError, ComicOptions, RenderedPage, open_comic};
use raster::Bitmap;

use crate::args::Args;
use crate::cleanup::Page;
use crate::render::{self, RenderError};

/// A failure of a whole comic run.
#[derive(Debug)]
pub(crate) enum RunError {
    /// The archive could not be opened or read as a comic.
    Archive(ComicError),
    /// The page-selection flags select no page, or contradict the page
    /// count.
    Selection(String),
    /// The render thread pool could not be started.
    Pool(rayon::ThreadPoolBuildError),
}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Archive(e) => e.fmt(f),
            Self::Selection(message) => f.write_str(message),
            Self::Pool(e) => write!(f, "cannot start the render thread pool: {e}"),
        }
    }
}

/// Render a comic archive, writing each decodable page via the shared encoder;
/// an embedded PDF opens with `session`, the PDF path's session settings.
///
/// Returns `(written, failed)`: the count of pages successfully written and the
/// count of pages that decoded-or-rendered-but-could-not-be-written. Per-page
/// errors are reported to stderr as they occur and skipped (one bad page never
/// aborts the rest); the caller maps a non-zero `failed` to a non-zero exit code
/// so a partial failure is observable via `$?`, matching the PDF path.
/// Whole-run failures surface as the outer `Err`, whose `Display` carries the
/// user-facing message.
pub(crate) fn run(
    args: &Args,
    spill: &crate::ram::SpillPolicy,
    session: pdf_raster::SessionConfig,
) -> Result<(usize, usize), RunError> {
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

    let mut pages = open_comic(Path::new(args.input()), &opts).map_err(RunError::Archive)?;

    // A comic with more pages than `i32::MAX` cannot be page-listed or named;
    // fail hard rather than silently mis-number, matching the PDF path.
    let total = i32::try_from(pages.len()).map_err(|_| {
        RunError::Archive(ComicError::BadArchive(format!(
            "archive has too many pages ({} > i32::MAX)",
            pages.len()
        )))
    })?;

    // Decide which page numbers to write via the SAME selector the PDF path uses
    // (`session.total_pages()` there, the full decoded count here), so
    // `--first-page`/`--last-page`/`--odd`/`--even`/`--single` and the clamp
    // warnings behave identically across input types.
    let (selected, warnings) = args.build_page_list(total).map_err(RunError::Selection)?;
    for w in &warnings {
        eprintln!("rrocket: warning: {w}");
    }
    // Drop the pages the selector excluded (odd/even/single/window) before
    // any is decoded; `selected` is ascending.
    pages.retain(|page| i32::try_from(page).is_ok_and(|p| selected.binary_search(&p).is_ok()));

    let pool = crate::pool::RenderPool::new(args.num_threads).map_err(RunError::Pool)?;
    let log = crate::progress::PageLog::new(args.timings, args.progress, selected.len());
    let (mut written, mut failed) = (0usize, 0usize);
    let unreported = crate::pool::run_in_order(
        &pool,
        std::iter::from_fn(|| pages.next_job()),
        |job| {
            let page_num = job.page();
            let outcome = log.page(page_num, || {
                job.render().map_err(|e| e.to_string()).and_then(|page| {
                    write_page(args, spill, page, total).map_err(|e| e.to_string())
                })
            });
            (page_num, outcome)
        },
        |_, (page_num, outcome)| {
            match outcome {
                Ok(()) => written += 1,
                Err(e) => {
                    eprintln!("rrocket: page {page_num}: {e}");
                    failed += 1;
                }
            }
            ControlFlow::Continue(())
        },
    );
    debug_assert!(unreported.is_empty(), "every comic page is reported");
    // `pool` drops here, before `pages`, whose embedded PDF's session holds
    // the device context of the workers' GPU state.
    drop(pool);
    Ok((written, failed))
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

#[cfg(test)]
mod tests {
    use comic::ComicError;

    use super::RunError;

    /// Only archive failures read as archive failures; a page-selection
    /// failure reads as on the PDF path.
    #[test]
    fn only_archive_failures_name_the_archive() {
        let selection = RunError::Selection("no pages selected".to_owned());
        assert_eq!(selection.to_string(), "no pages selected");
        let bad = || ComicError::BadArchive("truncated central directory".to_owned());
        assert_eq!(RunError::Archive(bad()).to_string(), bad().to_string());
    }
}
