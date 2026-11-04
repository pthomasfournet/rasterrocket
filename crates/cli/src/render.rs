//! Per-page rendering: call `rasterrocket` library → apply colour mode → write file.

use std::fs::{self, File};
use std::io::{BufWriter, Write};

use color::Rgb8;
use encode::{
    EncodeError, TiffImage, write_jpeg, write_pbm, write_pgm, write_png, write_ppm, write_tiff,
};
#[cfg(any(feature = "nvjpeg", feature = "nvjpeg2k", feature = "vaapi"))]
use pdf_raster::{BackendPolicy, ImageFilter};
use pdf_raster::{RasterError, RasterSession, RenderParams, render_page_rgb_with};
use raster::Bitmap;

use crate::args::{Args, OutputFormat};
use crate::cleanup::{self, Page};
use crate::fax::{self, FaxDocument};
use crate::naming::output_path_with_prefix;
use crate::print::PrintJob;
use crate::ram::SpillPolicy;
use crate::tone;

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
    Print(&'a PrintJob),
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
    let mut params = RenderParams::new_xy(scale_x, scale_y).with_policy(effective_policy);
    if let Some(region) = args
        .crop_region(geom.size_px(scale_x, scale_y))
        .map_err(RasterError::InvalidOptions)?
    {
        params = params.with_region(region);
    }
    let rgb: Bitmap<Rgb8> = render_page_rgb_with(session, page_num, &params)?;
    let sink = match (document, fax_layout) {
        (Document::Fax(document), Some(layout)) => Sink::Fax {
            document,
            turn: layout.turn,
        },
        (Document::Print(job), _) => Sink::Print { job },
        _ => Sink::Files {
            spill,
            format: args.output_format(),
        },
    };

    #[expect(
        clippy::cast_possible_wrap,
        reason = "total_pages ≤ i32::MAX; validated in main before calling this function"
    )]
    write_cleaned(
        Page::Rgb(rgb),
        args,
        sink,
        page_num as i32,
        total_pages as i32,
    )
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
    /// A page of the print job.
    Print { job: &'a PrintJob },
}

/// Run the camera stages selected on the command line on `page` (see
/// [`cleanup::clean`]) and the tone operations (see [`tone::prepare`]),
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
        let page = tone::prepare(page, args);
        let (spill, format) = match sink {
            Sink::Files { spill, format } => (spill, format),
            Sink::Fax { document, turn } => {
                document.add(page_num, page, turn)?;
                continue;
            }
            Sink::Print { job } => {
                job.add(page_num, half, &page)?;
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
        write_output(&out_path, |out| {
            encode_page(&page, format, args.dpi(), out).map_err(RenderError::from)
        })?;
    }
    Ok(())
}

/// Write-buffer size of an output file: at most
/// [`encode::PACKED_WRITE_BYTES`], so every full packed slice an encoder
/// writes bypasses the buffer.
pub(crate) const OUTPUT_BUFFER_BYTES: usize = encode::PACKED_WRITE_BYTES;

/// Create the page output file `out_path` and fill it through an
/// [`OUTPUT_BUFFER_BYTES`] write buffer with `encode`.  A file this call
/// created and failed to complete is removed; a file that could not be
/// created is left as it was.
///
/// # Errors
/// The creation, encoding or final flush failed.
pub(crate) fn write_output(
    out_path: &str,
    encode: impl FnOnce(&mut BufWriter<File>) -> Result<(), RenderError>,
) -> Result<(), RenderError> {
    let file = File::create(out_path)
        .map_err(|e| std::io::Error::new(e.kind(), format!("cannot create {out_path}: {e}")))?;
    let mut out = BufWriter::with_capacity(OUTPUT_BUFFER_BYTES, file);
    let written = encode(&mut out).and_then(|()| out.flush().map_err(RenderError::from));
    if written.is_err() {
        drop(out);
        if let Err(e) = fs::remove_file(out_path) {
            eprintln!("rrocket: warning: could not remove the incomplete {out_path}: {e}");
        }
    }
    written
}

/// Encode `page` in `format` to `out`: colour, gray or bilevel as the
/// page is, Netpbm choosing PPM, PGM or PBM by it.  `dpi` is recorded by
/// the formats that carry a resolution.
fn encode_page<W: Write>(
    page: &Page,
    format: OutputFormat,
    dpi: (f64, f64),
    out: W,
) -> Result<(), EncodeError> {
    match (format, page) {
        (OutputFormat::Ppm, Page::Rgb(rgb)) => write_ppm(rgb, out),
        (OutputFormat::Ppm, Page::Gray(luma)) => write_pgm(luma, out),
        (OutputFormat::Ppm, Page::Mono(luma)) => write_pbm(luma, out),
        (OutputFormat::Png, Page::Rgb(rgb)) => write_png(rgb, out),
        (OutputFormat::Png, Page::Gray(luma) | Page::Mono(luma)) => write_png(luma, out),
        (OutputFormat::Jpeg(options), Page::Rgb(rgb)) => write_jpeg(rgb, options, out),
        (OutputFormat::Jpeg(options), Page::Gray(luma) | Page::Mono(luma)) => {
            write_jpeg(luma, options, out)
        }
        (OutputFormat::Tiff(compression), page) => {
            let image = match page {
                Page::Rgb(rgb) => TiffImage::Rgb(rgb),
                Page::Gray(luma) => TiffImage::Gray(luma),
                Page::Mono(luma) => TiffImage::Bilevel(luma),
            };
            write_tiff(image, dpi, compression, out)
        }
    }
}
