//! Image encoding — write a `raster::Bitmap<P>` to PPM, PGM, PBM, PNG, and JPEG.
//!
//! # Supported formats
//!
//! | Function | Format | Accepted pixel types |
//! |----------|--------|--------------------|
//! | [`write_ppm`] | Netpbm P6 binary | colour modes (`Rgb8`, `Bgr8`, `Xbgr8`, `Cmyk8`, `DeviceN8`) |
//! | [`write_pgm`] | Netpbm P5 binary | `Gray8` |
//! | [`write_pbm`] | Netpbm P4 binary | `Gray8` (luma below 128 = black) |
//! | [`write_png`] | PNG | `Gray8` and the colour modes of [`write_ppm`] |
//! | [`write_jpeg`] | JPEG (baseline or progressive) | `Gray8`, `Rgb8`, `Bgr8` |
//! | [`jpeg_gray`] | JPEG (baseline), to a byte vector | `Gray8` |
//! | [`write_tiff`], [`TiffWriter`] | TIFF, one or more pages | `Rgb8`, `Gray8`, bilevel from `Gray8` |
//!
//! All functions consume the output sink (`W: Write`).  Wrap in
//! [`std::io::BufWriter`] at the call site if buffering is needed.
//!
//! # CMYK handling
//!
//! Neither PPM nor PNG natively supports CMYK.  [`write_ppm`] and
//! [`write_png`] convert CMYK/`DeviceN` (process channels only) to RGB with
//! the profile-free process-ink model of `color::cmyk`.  For ICC-accurate colour, convert to
//! `Rgb8` before encoding.

pub(crate) mod jpeg;
pub(crate) mod pbm;
pub(crate) mod pgm;
pub(crate) mod png;
pub(crate) mod ppm;
pub(crate) mod tiff;

pub use jpeg::{JpegOptions, jpeg_gray, write_jpeg};
pub use pbm::write_pbm;
pub use pgm::write_pgm;
pub use png::write_png;
pub use ppm::write_ppm;
pub use tiff::{TiffCompression, TiffImage, TiffWriter, write_tiff};

use std::io;

/// Errors that can occur during encoding.
#[derive(Debug)]
pub enum EncodeError {
    /// An I/O error writing to the output sink.
    Io(io::Error),
    /// The pixel mode is not supported by the chosen encoder.
    ///
    /// The message describes what the caller should do instead.
    UnsupportedMode(&'static str),
    /// An internal error from the `png` encoder (non-I/O).
    PngEncoder(::png::EncodingError),
    /// An internal error from the JPEG encoder.
    Jpeg(::jpeg_encoder::EncodingError),
}

impl std::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::UnsupportedMode(m) => write!(f, "pixel mode not supported: {m}"),
            Self::PngEncoder(e) => write!(f, "PNG encoder error: {e}"),
            Self::Jpeg(e) => write!(f, "JPEG encoder error: {e}"),
        }
    }
}

impl std::error::Error for EncodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::PngEncoder(e) => Some(e),
            Self::Jpeg(e) => Some(e),
            Self::UnsupportedMode(_) => None,
        }
    }
}

impl From<io::Error> for EncodeError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<::png::EncodingError> for EncodeError {
    fn from(e: ::png::EncodingError) -> Self {
        // Unwrap the I/O layer so EncodeError::Io is the canonical I/O path.
        match e {
            ::png::EncodingError::IoError(io_err) => Self::Io(io_err),
            other => Self::PngEncoder(other),
        }
    }
}
