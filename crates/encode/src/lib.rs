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
pub use tiff::{TiffCompression, TiffImage, TiffPage, TiffWriter, write_tiff};

use std::io::{self, Write};

use color::Pixel;
use raster::Bitmap;

/// Slice length in which a packed bitmap is written.
///
/// A bitmap is packed when `stride == width * P::BYTES`. This is also the
/// largest write-buffer capacity at which those slices bypass the buffer: a
/// `BufWriter` whose capacity is at most this passes each slice to its sink
/// without copying it into the buffer.
pub const PACKED_WRITE_BYTES: usize = 64 * 1024;

/// Write the live bytes of every row of `bitmap` (`width * P::BYTES` per
/// row, stride padding excluded) to `out`, top to bottom. A packed bitmap is
/// written straight from its buffer in [`PACKED_WRITE_BYTES`] slices, a
/// padded one row by row.
pub(crate) fn write_live_rows<P: Pixel, W: Write>(
    bitmap: &Bitmap<P>,
    out: &mut W,
) -> io::Result<()> {
    let live = bitmap.width as usize * P::BYTES;
    if bitmap.stride == live {
        for chunk in bitmap.data().chunks(PACKED_WRITE_BYTES) {
            out.write_all(chunk)?;
        }
    } else {
        for y in 0..bitmap.height {
            out.write_all(&bitmap.row_bytes(y)[..live])?;
        }
    }
    Ok(())
}

/// Luma below which a pixel is black in bilevel output, the midpoint of
/// the 8-bit range; lighter pixels are white.
pub const BLACK_BELOW: u8 = 128;

/// Pack the luma of a row into `out`, one bit per pixel, most significant
/// first, 1 for black (below [`BLACK_BELOW`]).  `out` holds
/// `luma.len().div_ceil(8)` bytes and is overwritten.
pub(crate) fn pack_bilevel_row(luma: &[u8], out: &mut [u8]) {
    let (whole, tail) = luma.as_chunks::<8>();
    for (byte, pixels) in out.iter_mut().zip(whole) {
        *byte = pixels
            .iter()
            .fold(0, |bits, &l| (bits << 1) | u8::from(l < BLACK_BELOW));
    }
    if !tail.is_empty() {
        let bits = tail
            .iter()
            .fold(0u8, |bits, &l| (bits << 1) | u8::from(l < BLACK_BELOW));
        out[whole.len()] = bits << (8 - tail.len());
    }
}

/// Errors that can occur during encoding.
#[derive(Debug)]
pub enum EncodeError {
    /// An I/O error writing to the output sink.
    Io(io::Error),
    /// The pixel mode is not supported by the chosen encoder.
    ///
    /// The message describes what the caller should do instead.
    UnsupportedMode(&'static str),
    /// The image or its parameters lie outside what the format holds: an
    /// empty image, a dimension, page count, resolution or file size beyond
    /// the format's limits.
    InvalidInput(&'static str),
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
            Self::InvalidInput(m) => write!(f, "cannot encode: {m}"),
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
            Self::UnsupportedMode(_) | Self::InvalidInput(_) => None,
        }
    }
}

impl From<io::Error> for EncodeError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<::jpeg_encoder::EncodingError> for EncodeError {
    fn from(e: ::jpeg_encoder::EncodingError) -> Self {
        // Unwrap the I/O layer so EncodeError::Io is the canonical I/O path.
        match e {
            ::jpeg_encoder::EncodingError::IoError(io_err) => Self::Io(io_err),
            other => Self::Jpeg(other),
        }
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
