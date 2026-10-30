//! Per-page rendering: call `rasterrocket` library → apply colour mode → write file.

#![expect(
    clippy::redundant_pub_crate,
    reason = "bin-only crate with private modules: pub(crate) is the real \
              visibility and cannot escape; widening to pub would misdescribe it"
)]

use std::fs::{self, File};
use std::io::{BufWriter, Write as _};

use color::{Gray8, Rgb8};
use encode::{EncodeError, write_pbm, write_pgm, write_png, write_ppm};
#[cfg(any(feature = "nvjpeg", feature = "nvjpeg2k", feature = "vaapi"))]
use pdf_raster::{BackendPolicy, ImageFilter};
use pdf_raster::{RasterError, RasterSession, RenderParams, render_page_rgb_with, rgb_to_gray};
use raster::Bitmap;

use crate::args::{Args, OutputFormat};
use crate::cleanup::{self, Page};
use crate::naming::output_path_with_prefix;
use crate::ram::SpillPolicy;

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
    /// The requested format + colour-mode combination is not supported.
    UnsupportedFormatCombination {
        /// The output format requested by the user.
        output: OutputFormat,
    },
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Raster(e) => write!(f, "render error: {e}"),
            Self::Encode(e) => write!(f, "encode error: {e}"),
            Self::Camera(e) => write!(f, "camera cleanup error: {e}"),
            Self::UnsupportedFormatCombination { output } => {
                write!(f, "output format {output} is not yet supported")
            }
        }
    }
}

impl std::error::Error for RenderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Raster(e) => Some(e),
            Self::Encode(e) => Some(e),
            Self::Camera(_) | Self::UnsupportedFormatCombination { .. } => None,
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
///
/// JPEG and TIFF output are not yet implemented and return
/// [`RenderError::UnsupportedFormatCombination`].
pub(crate) fn render_page(
    session: &RasterSession,
    page_num: u32,
    total_pages: u32,
    args: &Args,
    spill: &SpillPolicy,
) -> Result<(), RenderError> {
    let format = args.output_format();

    if matches!(format, OutputFormat::Jpeg | OutputFormat::Tiff) {
        return Err(RenderError::UnsupportedFormatCombination { output: format });
    }

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
/// [`cleanup::clean`]), then write each resulting page: one file, or the
/// `a`/`b` halves of a split spread (see [`output_path_with_prefix`]).
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

        // Write directly to the final path. The previous temp-file +
        // atomic-rename dance triggered ext4 auto_da_alloc on every page,
        // which forces dirty data out before the rename can complete —
        // turning batch render into a serial disk-flush wait. For a many-page
        // batch the safety guarantee (no partial file on encode failure)
        // isn't worth that cost; the user re-runs on error anyway. We delete
        // a partial file on encode failure to preserve the spirit.
        let written = match &page {
            Page::Rgb(rgb) => encode_to_path(&out_path, rgb, args, format),
            Page::Gray(gray) => encode_gray_to_path(&out_path, gray, args, format),
        };
        written.inspect_err(|_| {
            let _ = fs::remove_file(&out_path);
        })?;
    }
    Ok(())
}

/// Encode `rgb` to `out_path` honouring `--mono` / `--gray` and the resolved format.
///
/// Shared by the PDF render path and the comic-archive path so colour-mode and
/// format handling stay defined in exactly one place.
pub(crate) fn encode_to_path(
    out_path: &str,
    rgb: &Bitmap<Rgb8>,
    args: &Args,
    format: OutputFormat,
) -> Result<(), RenderError> {
    if args.mono || args.gray {
        return encode_gray_to_path(out_path, &rgb_to_gray(rgb), args, format);
    }
    let mut out = BufWriter::new(File::create(out_path)?);
    match format {
        OutputFormat::Ppm => write_ppm(rgb, &mut out)?,
        OutputFormat::Png => write_png(rgb, &mut out)?,
        OutputFormat::Jpeg | OutputFormat::Tiff => unreachable!("rejected above"),
    }
    out.flush()?;
    Ok(())
}

/// Encode a luma page to `out_path`: thresholded to two levels under
/// `--mono`, as gray otherwise.
fn encode_gray_to_path(
    out_path: &str,
    gray: &Bitmap<Gray8>,
    args: &Args,
    format: OutputFormat,
) -> Result<(), RenderError> {
    let mut out = BufWriter::new(File::create(out_path)?);
    if args.mono {
        let mono = gray_to_mono(gray);
        match format {
            OutputFormat::Ppm => write_pbm::<Gray8, _>(&mono, &mut out)?,
            OutputFormat::Png => write_png::<Gray8, _>(&mono, &mut out)?,
            OutputFormat::Jpeg | OutputFormat::Tiff => unreachable!("rejected above"),
        }
    } else {
        match format {
            OutputFormat::Ppm => write_pgm::<Gray8, _>(gray, &mut out)?,
            OutputFormat::Png => write_png::<Gray8, _>(gray, &mut out)?,
            OutputFormat::Jpeg | OutputFormat::Tiff => unreachable!("rejected above"),
        }
    }
    out.flush()?;
    Ok(())
}

// ── Pixel helpers ─────────────────────────────────────────────────────────────

/// Luma threshold separating dark ink from light paper (inclusive on the light side).
///
/// Values below this → 255 (black ink); values at or above → 0 (white paper).
const MONO_THRESHOLD: u8 = 128;

/// Threshold a grayscale bitmap to 2-level monochrome at the 50% midpoint.
///
/// Values < [`MONO_THRESHOLD`] (dark) → 255 (black ink);
/// values ≥ [`MONO_THRESHOLD`] (light) → 0 (white paper).
fn gray_to_mono(src: &Bitmap<Gray8>) -> Bitmap<Gray8> {
    let mut dst: Bitmap<Gray8> = Bitmap::new(src.width, src.height, 1, false);
    let w = src.width as usize;
    for y in 0..src.height {
        let src_row = &src.row_bytes(y)[..w];
        let dst_row = &mut dst.row_bytes_mut(y)[..w];
        for (dst_px, &luma) in dst_row.iter_mut().zip(src_row.iter()) {
            *dst_px = if luma < MONO_THRESHOLD { 255 } else { 0 };
        }
    }
    dst
}
