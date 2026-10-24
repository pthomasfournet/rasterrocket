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

use super::{DecodeError, guard_dimensions, rgb_bitmap_from_tight};

/// Decode baseline/progressive JPEG bytes to an `Rgb8` bitmap.
///
/// The frame header is read first and its dimensions checked by
/// [`guard_dimensions`] before the pixel buffer is allocated.  The decoder's
/// own per-side cap is raised to the shared limit so that check is the single
/// size policy across codecs.
///
/// # Errors
///
/// [`DecodeError::Codec`] if the stream is not decodable, reports zero dims,
/// or declares dimensions beyond the rendered-page limits.
pub(super) fn decode(bytes: &[u8]) -> Result<Bitmap<Rgb8>, DecodeError> {
    let side_cap = pdf_raster::MAX_PX_DIMENSION as usize;
    let opts = DecoderOptions::default()
        .jpeg_set_out_colorspace(ColorSpace::RGB)
        .set_max_width(side_cap)
        .set_max_height(side_cap);
    let mut dec = JpegDecoder::new_with_options(ZCursor::new(bytes), opts);
    dec.decode_headers()
        .map_err(|e| DecodeError::Codec(format!("jpeg: {e}")))?;
    let info = dec
        .info()
        .ok_or_else(|| DecodeError::Codec("jpeg: no image info".to_owned()))?;
    let (w, h) = (u32::from(info.width), u32::from(info.height));
    if w == 0 || h == 0 {
        return Err(DecodeError::Codec(format!("jpeg: zero dimensions {w}x{h}")));
    }
    guard_dimensions("jpeg", w, h)?;
    let pixels = dec
        .decode()
        .map_err(|e| DecodeError::Codec(format!("jpeg: {e}")))?;
    rgb_bitmap_from_tight(w, h, &pixels)
}
