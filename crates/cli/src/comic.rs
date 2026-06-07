//! Comic-archive (.cbz/.cb7/.cbt) rendering entry for the CLI.
//!
//! Bridges the comic crate's `open_comic` into the same per-page output sink the
//! PDF path uses, so output format, colour mode, naming, and page-range behave
//! identically across input types.

use std::path::Path;

use color::Rgb8;
use comic::{ComicError, ComicOptions, RenderedPage, open_comic};
use raster::Bitmap;

use crate::args::{Args, OutputFormat};
use crate::render::{self, RenderError};

/// Render a comic archive, writing each decodable page via the shared encoder.
///
/// Returns the number of pages written; per-page errors are reported to stderr
/// and skipped (one bad page never aborts the rest). Whole-archive failures
/// (including a `.cbr` input) surface as the outer `Err`, whose `Display`
/// carries the user-facing message.
pub fn run(args: &Args) -> Result<usize, ComicError> {
    let opts = ComicOptions {
        // Comics carry no physical resolution, so the user's requested DPI rides
        // along on each RenderedPage purely for downstream OCR feature scaling;
        // it never changes the already-rasterised pixel dimensions. When the x
        // and y DPI differ, x is used (documented behaviour for resolution-less
        // input).
        #[expect(
            clippy::cast_possible_truncation,
            reason = "DPI is a small positive value; f64→f32 is exact in practice"
        )]
        dpi: args.x_dpi() as f32,
        // Decode every page; page SELECTION (window, odd/even, single) is done
        // once, below, by the same selector the PDF path uses. Windowing here
        // too would double-apply first/last against a shrunken total.
        first_page: 1,
        last_page: u32::MAX,
        deskew: false,
    };

    let pages = open_comic(Path::new(&args.input), &opts)?;

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

    let mut written = 0usize;
    for (page_num, result) in pages {
        // Skip pages the selector excluded (odd/even/single/window).
        if i32::try_from(page_num).is_ok_and(|p| !selected.contains(&p)) {
            continue;
        }
        match result {
            Ok(page) => match write_page(args, &page, total) {
                Ok(()) => written += 1,
                Err(e) => eprintln!("rrocket: page {page_num}: {e}"),
            },
            Err(e) => eprintln!("rrocket: page {page_num}: {e}"),
        }
    }
    Ok(written)
}

/// Wrap a grayscale [`RenderedPage`] into an [`Rgb8`] bitmap (luma replicated to
/// R=G=B) and write it via the shared [`render::encode_to_path`], so colour-mode
/// and format reuse the PDF path's single writer.
fn write_page(args: &Args, page: &RenderedPage, total_pages: i32) -> Result<(), RenderError> {
    let format = args.output_format();

    // Reject JPEG/TIFF early, exactly as the PDF render path does, before
    // allocating the bitmap. `encode_to_path` also guards, but failing here
    // matches `render_page` and avoids the wasted luma→RGB expansion.
    if matches!(format, OutputFormat::Jpeg | OutputFormat::Tiff) {
        return Err(RenderError::UnsupportedFormatCombination { output: format });
    }

    let mut rgb = Bitmap::<Rgb8>::new(page.width, page.height, 1, false);
    let w = page.width as usize;
    for y in 0..page.height {
        let src = &page.pixels[(y as usize * w)..((y as usize + 1) * w)];
        let row = rgb.row_bytes_mut(y);
        for (x, &luma) in src.iter().enumerate() {
            row[x * 3..x * 3 + 3].copy_from_slice(&[luma, luma, luma]);
        }
    }

    // `page.page_num` is selected from `build_page_list`'s output, which is
    // bounded by `total_pages` (≤ i32::MAX, checked in `run`), so the cast back
    // to i32 cannot overflow.
    let page_i32 = i32::try_from(page.page_num).unwrap_or(total_pages);
    let out_path = crate::naming::output_path_with_prefix(
        &args.output_prefix,
        args,
        page_i32,
        total_pages,
        format,
    );
    render::encode_to_path(&out_path, &rgb, args, format)
}
