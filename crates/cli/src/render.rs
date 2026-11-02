//! Per-page rendering: call `rasterrocket` library → apply colour mode → write file.

use std::fs::{self, File};
use std::io::{BufWriter, Write as _};

use color::{Gray8, Rgb8};
use encode::{
    EncodeError, TiffImage, write_jpeg, write_pbm, write_pgm, write_png, write_ppm, write_tiff,
};
#[cfg(any(feature = "nvjpeg", feature = "nvjpeg2k", feature = "vaapi"))]
use pdf_raster::{BackendPolicy, ImageFilter};
use pdf_raster::{RasterError, RasterSession, RenderParams, render_page_rgb_with};
use raster::Bitmap;

use crate::args::{Args, OutputFormat};
use crate::cleanup::{self, Page};
use crate::naming::output_path_with_prefix;
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
) -> Result<(), RenderError> {
    let format = args.output_format();

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
    let (scale_x, scale_y) = args.scales((geom.width_pts, geom.height_pts));
    let mut params = RenderParams::new_xy(scale_x, scale_y).with_policy(effective_policy);
    if let Some(region) = args
        .crop_region(geom.size_px(scale_x, scale_y))
        .map_err(RasterError::InvalidOptions)?
    {
        params = params.with_region(region);
    }
    let rgb: Bitmap<Rgb8> = render_page_rgb_with(session, page_num, &params)?;

    #[expect(
        clippy::cast_possible_wrap,
        reason = "total_pages ≤ i32::MAX; validated in main before calling this function"
    )]
    write_cleaned(
        rgb,
        args,
        spill,
        page_num as i32,
        total_pages as i32,
        format,
    )
}

/// Run the camera stages selected on the command line on `rgb` (see
/// [`cleanup::clean`]) and the tone operations (see [`tone::prepare`]),
/// then write each resulting page: one file, or the `a`/`b` halves of a
/// split spread (see [`output_path_with_prefix`]).
///
/// Shared by the PDF render path and the comic-archive path.
pub(crate) fn write_cleaned(
    rgb: Bitmap<Rgb8>,
    args: &Args,
    spill: &SpillPolicy,
    page_num: i32,
    total_pages: i32,
    format: OutputFormat,
) -> Result<(), RenderError> {
    let pages = cleanup::clean(rgb, args.camera_options(), args.gray || args.mono)?;
    for (page, half) in pages {
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
        match &tone::prepare(page, args) {
            Page::Rgb(rgb) => encode_to_path(&out_path, rgb, args, format),
            Page::Gray(gray) => encode_gray_to_path(&out_path, gray, args, format),
        }?;
    }
    Ok(())
}

/// Write-buffer size of a page output file: the encoders emit a row at a
/// time, and the file receives them in `write` calls of this size.
const OUTPUT_BUFFER_BYTES: usize = 64 * 1024;

/// Create the page output file `out_path` and fill it through an
/// [`OUTPUT_BUFFER_BYTES`] write buffer with `encode`.  A file this call
/// created and failed to complete is removed; a file that could not be
/// created is left as it was.
///
/// # Errors
/// The creation, encoding or final flush failed.
fn write_output(
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

/// Encode a colour page to `out_path` in `format`.
fn encode_to_path(
    out_path: &str,
    rgb: &Bitmap<Rgb8>,
    args: &Args,
    format: OutputFormat,
) -> Result<(), RenderError> {
    write_output(out_path, |out| {
        match format {
            OutputFormat::Ppm => write_ppm(rgb, out)?,
            OutputFormat::Png => write_png(rgb, out)?,
            OutputFormat::Jpeg(options) => write_jpeg(rgb, options, out)?,
            OutputFormat::Tiff(compression) => {
                write_tiff(TiffImage::Rgb(rgb), args.dpi(), compression, out)?;
            }
        }
        Ok(())
    })
}

/// Encode a luma page to `out_path`: thresholded to two levels under
/// `--mono`, as gray otherwise.
fn encode_gray_to_path(
    out_path: &str,
    gray: &Bitmap<Gray8>,
    args: &Args,
    format: OutputFormat,
) -> Result<(), RenderError> {
    write_output(out_path, |out| {
        if args.mono {
            let mono = tone::gray_to_mono(gray);
            match format {
                OutputFormat::Ppm => write_pbm::<Gray8, _>(&mono, out)?,
                OutputFormat::Png => write_png::<Gray8, _>(&mono, out)?,
                OutputFormat::Jpeg(options) => write_jpeg(&mono, options, out)?,
                OutputFormat::Tiff(compression) => {
                    write_tiff(TiffImage::Bilevel(&mono), args.dpi(), compression, out)?;
                }
            }
        } else {
            match format {
                OutputFormat::Ppm => write_pgm::<Gray8, _>(gray, out)?,
                OutputFormat::Png => write_png::<Gray8, _>(gray, out)?,
                OutputFormat::Jpeg(options) => write_jpeg(gray, options, out)?,
                OutputFormat::Tiff(compression) => {
                    write_tiff(TiffImage::Gray(gray), args.dpi(), compression, out)?;
                }
            }
        }
        Ok(())
    })
}
