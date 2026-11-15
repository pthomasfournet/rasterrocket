//! Per-page rendering: call `rasterrocket` library → apply colour mode → write file.

use std::fs::{self, File};
use std::io::{BufWriter, Seek, Write};
use std::sync::{Mutex, MutexGuard, PoisonError};

use encode::{
    EncodeError, TiffStream, write_jpeg, write_pbm, write_pgm, write_png, write_png_bilevel,
    write_ppm, write_tiff,
};
#[cfg(any(feature = "nvjpeg", feature = "nvjpeg2k", feature = "vaapi"))]
use pdf_raster::{BackendPolicy, ImageFilter};
use pdf_raster::{FinishedCanvas, RasterError, RasterSession, RenderParams, render_page_canvas};

use crate::args::{Args, OutputFormat};
use crate::bands::{self, BandOutput};
use crate::cleanup::{self, Page};
use crate::fax::{self, FaxDocument};
use crate::naming::output_path_with_prefix;
use crate::print::PageSink;
use crate::ram::SpillPolicy;
use crate::{ocr_prep, tone};

// ── Error type ────────────────────────────────────────────────────────────────

/// Error returned by [`render_page`].
#[derive(Debug)]
pub(crate) enum RenderError {
    /// An I/O error writing the output file.
    Io(std::io::Error),
    /// The page could not be rasterised.
    Raster(RasterError),
    /// The encoder rejected the bitmap.
    Encode(EncodeError),
    /// The camera-capture cleanup rejected the page.
    Camera(String),
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Raster(e) => write!(f, "render error: {e}"),
            Self::Encode(e) => write!(f, "encode error: {e}"),
            Self::Camera(e) => write!(f, "camera cleanup error: {e}"),
        }
    }
}

impl std::error::Error for RenderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Raster(e) => Some(e),
            Self::Encode(e) => Some(e),
            Self::Camera(_) => None,
        }
    }
}

impl From<std::io::Error> for RenderError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<RasterError> for RenderError {
    fn from(e: RasterError) -> Self {
        Self::Raster(e)
    }
}
impl From<EncodeError> for RenderError {
    fn from(e: EncodeError) -> Self {
        Self::Encode(e)
    }
}
impl From<camera::CameraError> for RenderError {
    fn from(e: camera::CameraError) -> Self {
        Self::Camera(e.to_string())
    }
}

// ── Per-page render + write ───────────────────────────────────────────────────

/// The document a page belongs to, when the pages are not files of their
/// own.
#[derive(Clone, Copy)]
pub(crate) enum Document<'a> {
    /// One file per page.
    None,
    /// A page of the fax document.
    Fax(&'a FaxDocument),
    /// A page of the print job.
    Print(&'a PageSink),
}

/// Render one page and write it to the appropriate output file.
///
/// `page_num` is 1-based.  Each call prescans the page to determine whether
/// it is text/vector-only, and if so passes `BackendPolicy::CpuOnly` to skip
/// GPU decoder initialisation on this thread.  The prescan runs on the same
/// worker thread as the render, so its cost is hidden inside parallel render
/// time rather than adding a serial phase before the pool starts.
pub(crate) fn render_page(
    session: &RasterSession,
    page_num: u32,
    total_pages: u32,
    args: &Args,
    spill: &SpillPolicy,
    document: Document<'_>,
) -> Result<(), RenderError> {
    // Prescan only when a GPU decoder is compiled in — otherwise it's pure overhead
    // with no benefit (CpuOnly is already the effective policy for every page).
    #[cfg(any(feature = "nvjpeg", feature = "nvjpeg2k", feature = "vaapi"))]
    let effective_policy = match pdf_raster::prescan_session(session, page_num).ok() {
        Some(ref diag) if !diag.has_images => BackendPolicy::CpuOnly,
        Some(ref diag) if matches!(diag.dominant_filter, Some(ImageFilter::Dct)) => {
            session.policy()
        }
        _ => session.policy(),
    };
    #[cfg(not(any(feature = "nvjpeg", feature = "nvjpeg2k", feature = "vaapi")))]
    let effective_policy = session.policy();

    let geom = session.page_geometry(page_num)?;
    let dims = (geom.width_pts, geom.height_pts);
    let fax_layout = match document {
        Document::Fax(_) => Some(fax::layout(dims, args.y_dpi())),
        Document::None | Document::Print(_) => None,
    };
    let (scale_x, scale_y) =
        fax_layout.map_or_else(|| args.scales(dims), |l| (l.scale_x, l.scale_y));
    let mut params = RenderParams::new_xy(scale_x, scale_y)
        .with_policy(effective_policy)
        .with_depth(args.depth());
    if let Some(intent) = args.intent {
        params = params.with_intent(intent.into());
    }
    if args.bpc {
        params = params.with_black_point_compensation();
    }
    if args.transparent {
        params = params.with_transparent_background();
    }
    if args.cmyk() {
        params = params.with_cmyk();
    }
    if let Some(region) = args
        .crop_region(geom.size_px(scale_x, scale_y))
        .map_err(RasterError::InvalidOptions)?
    {
        params = params.with_region(region);
    }
    let size = params
        .region
        .map_or_else(|| geom.size_px(scale_x, scale_y), |r| (r.width, r.height));
    match (BandOutput::of(args), document) {
        (Some(BandOutput::File(format)), Document::None) => {
            return bands::write_page_banded(
                session,
                (page_num, total_pages),
                args,
                (format, spill),
                &params,
                size,
            );
        }
        (Some(BandOutput::Print), Document::Print(sink)) => {
            let dpi = args.page_dpi(dims);
            return bands::print_page_banded(session, page_num, args, sink, &params, (size, dpi));
        }
        _ => {}
    }
    if args.cmyk() {
        return write_cmyk_page(session, (page_num, total_pages), args, spill, &params);
    }
    let page = match render_page_canvas(session, page_num, &params)? {
        FinishedCanvas::Rgb8(rgb) => Page::Rgb(rgb),
        FinishedCanvas::Gray8(gray) => Page::Gray(gray),
        FinishedCanvas::Rgb16(rgb) => Page::Rgb16(rgb),
        FinishedCanvas::Gray16(gray) => Page::Gray16(gray),
        FinishedCanvas::Cmyk8(_) | FinishedCanvas::Cmyk16(_) => {
            return Err(RasterError::InvalidOptions(
                "a CMYK page is written as CMYK TIFF or JPEG; this output takes RGB or gray pages"
                    .to_owned(),
            )
            .into());
        }
    };
    let sink = match (document, fax_layout) {
        (Document::Fax(document), Some(layout)) => Sink::Fax {
            document,
            turn: layout.turn,
        },
        (Document::Print(pages), _) => Sink::Print {
            pages,
            dpi: args.page_dpi(dims),
        },
        _ => Sink::Files {
            spill,
            format: args.output_format(),
        },
    };

    #[expect(
        clippy::cast_possible_wrap,
        reason = "total_pages ≤ i32::MAX; validated in main before calling this function"
    )]
    write_cleaned(page, args, sink, page_num as i32, total_pages as i32)
}

/// Render page `page_num` under `params`, a CMYK page, whole and write it
/// as the page file of the output format, CMYK JPEG: the one CMYK output
/// not written band by band.
fn write_cmyk_page(
    session: &RasterSession,
    (page_num, total_pages): (u32, u32),
    args: &Args,
    spill: &SpillPolicy,
    params: &RenderParams,
) -> Result<(), RenderError> {
    let format = args.output_format();
    let OutputFormat::Jpeg(options) = format else {
        return Err(RasterError::InvalidOptions(
            "a CMYK page is written whole as JPEG only; CMYK TIFF is written band by band"
                .to_owned(),
        )
        .into());
    };
    let FinishedCanvas::Cmyk8(cmyk) = render_page_canvas(session, page_num, params)? else {
        return Err(
            RasterError::InvalidOptions("a CMYK JPEG holds an 8-bit CMYK page".to_owned()).into(),
        );
    };
    #[expect(
        clippy::cast_possible_wrap,
        reason = "page_num and total_pages ≤ i32::MAX; validated in main before calling this function"
    )]
    let (number, total) = (page_num as i32, total_pages as i32);
    let path = output_path_with_prefix(spill.next_prefix(), args, number, total, format, None);
    write_output(&path, |mut out| {
        write_jpeg(
            &cmyk,
            options,
            Some(args.dpi()),
            &mut out,
            args.output_icc.as_ref(),
        )?;
        Ok(out)
    })
}

/// Where a cleaned page goes.
#[derive(Clone, Copy)]
pub(crate) enum Sink<'a> {
    /// One file per page in `format`, under the prefix the spill policy
    /// chooses.
    Files {
        spill: &'a SpillPolicy,
        format: OutputFormat,
    },
    /// A page of the fax document, turned a quarter when `turn`.
    Fax {
        document: &'a FaxDocument,
        turn: bool,
    },
    /// A page of the print job, rendered at `dpi`.
    Print {
        pages: &'a PageSink,
        dpi: (f64, f64),
    },
}

/// Run the camera stages selected on the command line on `page` (see
/// [`cleanup::clean`]), deskew under `--deskew` (see [`cleanup::deskewed`]),
/// the OCR preparation of `--clean-borders` and `--clean-background` (see
/// [`ocr_prep::prepare`]) and the tone operations (see [`tone::prepare`]),
/// then hand each resulting page, one or the `a`/`b` halves of a split
/// spread, to `sink`: a file each (see [`output_path_with_prefix`]), the
/// fax document or the print job.
///
/// Shared by the PDF render path and the comic-archive path.
pub(crate) fn write_cleaned(
    page: Page,
    args: &Args,
    sink: Sink<'_>,
    page_num: i32,
    total_pages: i32,
) -> Result<(), RenderError> {
    let pages = cleanup::clean(page, args.camera_options(), args.luma_output())?;
    for (page, half) in pages {
        let page = if args.deskew {
            cleanup::deskewed(page, args.policy())?
        } else {
            page
        };
        let page = ocr_prep::prepare(page, args)?;
        let page = tone::prepare(page, args)?;
        let (spill, format) = match sink {
            Sink::Files { spill, format } => (spill, format),
            Sink::Fax { document, turn } => {
                document.add(page_num, page, turn)?;
                continue;
            }
            Sink::Print { pages, dpi } => {
                pages.add((page_num, half), &page, dpi)?;
                continue;
            }
        };
        // Pick RAM vs disk per file: when --ram is active, the SpillPolicy
        // hands back the tmpfs prefix until free memory drops below the
        // safety margin, then transparently spills the rest of the run to the
        // user's disk path. When --ram is off, the policy is a passthrough
        // that always returns the original prefix.
        let out_path = output_path_with_prefix(
            spill.next_prefix(),
            args,
            page_num,
            total_pages,
            format,
            half,
        );

        // Pages are written in place, not renamed into place, so that no
        // per-page rename waits on the file system's flush of the data.
        write_output(&out_path, |mut out| {
            if let Some(image) = page.wide_image() {
                return bands::write_whole(out, format, image, args);
            }
            encode_page(&page, format, args, &mut out)?;
            Ok(out)
        })?;
    }
    Ok(())
}

/// `mutex` locked.  A thread that panicked while holding it leaves the
/// data as its last completed update; every guarded update in this crate is
/// a single push, assignment or take, so that state is consistent.
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Pages encoded on the workers that rendered them, handed back in key
/// order.
pub(crate) struct PageCollector<K, P> {
    pages: Mutex<Vec<(K, P)>>,
}

impl<K: Ord + Copy, P> PageCollector<K, P> {
    /// An empty collection.
    pub(crate) const fn new() -> Self {
        Self {
            pages: Mutex::new(Vec::new()),
        }
    }

    /// Hold `page` under `key`.
    pub(crate) fn push(&self, key: K, page: P) {
        lock(&self.pages).push((key, page));
    }

    /// The pages held, in key order, leaving the collection empty.
    pub(crate) fn take_sorted(&self) -> Vec<P> {
        Self::in_key_order(std::mem::take(&mut *lock(&self.pages)))
    }

    /// The pages held under keys up to `last` inclusive, in key order,
    /// leaving the others held.
    pub(crate) fn take_through(&self, last: K) -> Vec<P> {
        let taken = lock(&self.pages)
            .extract_if(.., |(key, _)| *key <= last)
            .collect();
        Self::in_key_order(taken)
    }

    /// How many pages are held.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        lock(&self.pages).len()
    }

    /// The pages of `keyed`, sorted by key.
    fn in_key_order(mut keyed: Vec<(K, P)>) -> Vec<P> {
        keyed.sort_unstable_by_key(|&(key, _)| key);
        keyed.into_iter().map(|(_, page)| page).collect()
    }
}

/// Write-buffer size of an output file: at most
/// [`encode::PACKED_WRITE_BYTES`], so every full packed slice an encoder
/// writes bypasses the buffer.
pub(crate) const OUTPUT_BUFFER_BYTES: usize = encode::PACKED_WRITE_BYTES;

/// Create the page output file `out_path` and fill it through an
/// [`OUTPUT_BUFFER_BYTES`] write buffer with `encode`, which returns the
/// buffer once the file is complete.  A file this call created and failed
/// to complete is removed; a file that could not be created is left as it
/// was.
///
/// # Errors
/// The creation, encoding or final flush failed.
pub(crate) fn write_output(
    out_path: &str,
    encode: impl FnOnce(BufWriter<File>) -> Result<BufWriter<File>, RenderError>,
) -> Result<(), RenderError> {
    let file = File::create(out_path)
        .map_err(|e| std::io::Error::new(e.kind(), format!("cannot create {out_path}: {e}")))?;
    let out = BufWriter::with_capacity(OUTPUT_BUFFER_BYTES, file);
    let written = encode(out).and_then(|mut out| out.flush().map_err(RenderError::from));
    if written.is_err()
        && let Err(e) = fs::remove_file(out_path)
    {
        eprintln!("rrocket: warning: could not remove the incomplete {out_path}: {e}");
    }
    written
}

/// Encode `page`, an 8-bit page, in `format` to `out`: colour, gray or
/// bilevel as the page is, Netpbm choosing PPM, PGM or PBM by it and PNG
/// bit depth 8 or 1, tagged with the output profile by the formats that
/// carry one.  The resolution is recorded by the formats that carry it:
/// PNG, JPEG and TIFF.  A 16-bit page is [`bands::write_whole`]'s.
fn encode_page<W: Write + Seek>(
    page: &Page,
    format: OutputFormat,
    args: &Args,
    out: W,
) -> Result<(), EncodeError> {
    let icc = args.output_icc.as_ref();
    let dpi = Some(args.dpi());
    match (format, page) {
        (OutputFormat::Ppm, Page::Rgb(rgb)) => write_ppm(rgb, out),
        (OutputFormat::Ppm, Page::Gray(luma)) => write_pgm(luma, out),
        (OutputFormat::Ppm, Page::Mono(luma)) => write_pbm(luma, out),
        (OutputFormat::Png, Page::Rgb(rgb)) => write_png(rgb, dpi, out, icc),
        (OutputFormat::Png, Page::Gray(luma)) => write_png(luma, dpi, out, icc),
        (OutputFormat::Png, Page::Mono(luma)) => write_png_bilevel(luma, dpi, out, icc),
        (OutputFormat::Jpeg(options), Page::Rgb(rgb)) => write_jpeg(rgb, options, dpi, out, icc),
        (OutputFormat::Jpeg(options), Page::Gray(luma) | Page::Mono(luma)) => {
            write_jpeg(luma, options, dpi, out, icc)
        }
        (_, Page::Rgb16(_) | Page::Gray16(_)) => Err(EncodeError::UnsupportedMode(
            "a 16-bit page is written by the row-streamed writers",
        )),
        (OutputFormat::Tiff(compression), page) => {
            let image = page.row_image();
            match icc {
                None => write_tiff(image, args.dpi(), compression, out),
                Some(icc) => {
                    let mut tiff = TiffStream::new(
                        out,
                        image.size(),
                        image.layout(),
                        args.dpi(),
                        compression,
                        Some(icc),
                    )?;
                    tiff.push(image)?;
                    tiff.finish().map(drop)
                }
            }
        }
    }
}
