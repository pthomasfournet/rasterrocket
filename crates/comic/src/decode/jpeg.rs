//! Plain JPEG decode for comic pages.
//!
//! Deliberately separate from the PDF `DCTDecode` filter: that path is PDF-aware
//! (CMYK-for-colorspace, dimension cross-checks against the PDF dict). A comic
//! page needs only "JPEG bytes -> RGB", so this is a thin `zune-jpeg` wrapper —
//! shared dependency, not shared logic.

use color::Rgb8;
use raster::Bitmap;
use zune_jpeg::JpegDecoder;
use zune_jpeg::zune_core::bytestream::ZCursor;
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;

use super::{DecodeError, rgb_bitmap_from_tight};

/// Decode baseline/progressive JPEG bytes to an `Rgb8` bitmap.
///
/// # Errors
///
/// [`DecodeError::Codec`] if the stream is not decodable or reports zero dims.
pub fn decode(bytes: &[u8]) -> Result<Bitmap<Rgb8>, DecodeError> {
    let opts = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
    let mut dec = JpegDecoder::new_with_options(ZCursor::new(bytes), opts);
    let pixels = dec
        .decode()
        .map_err(|e| DecodeError::Codec(format!("jpeg: {e}")))?;
    let info = dec
        .info()
        .ok_or_else(|| DecodeError::Codec("jpeg: no image info".to_owned()))?;
    let (w, h) = (u32::from(info.width), u32::from(info.height));
    if w == 0 || h == 0 {
        return Err(DecodeError::Codec(format!("jpeg: zero dimensions {w}x{h}")));
    }
    rgb_bitmap_from_tight(w, h, &pixels)
}
