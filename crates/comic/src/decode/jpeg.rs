//! Plain JPEG decode for comic pages.
//!
//! Deliberately separate from the PDF `DCTDecode` filter: that path is PDF-aware
//! (CMYK-for-colorspace, dimension cross-checks against the PDF dict). A comic
//! page needs only "JPEG bytes -> pixels", so this is a thin `zune-jpeg` wrapper —
//! shared dependency, not shared logic.

use zune_jpeg::JpegDecoder;
use zune_jpeg::zune_core::bytestream::ZCursor;
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;

use super::{CodecPixels, DecodeError, gray_from_tight, guard_dimensions, rgb_from_tight};

/// Decode baseline/progressive JPEG bytes: a single-component stream to a
/// `Gray8` bitmap, any other to an `Rgb8` bitmap.
///
/// The frame header is read first and its dimensions checked by
/// [`guard_dimensions`] before the pixel buffer is allocated.  The decoder's
/// own per-side cap is raised to the shared limit so that check is the single
/// size policy across codecs.
///
/// # Errors
///
/// [`DecodeError::TooLarge`] when the frame exceeds the rendered-page
/// limits; [`DecodeError::Codec`] if the stream is not decodable or reports
/// zero dimensions.
pub(super) fn decode(bytes: &[u8]) -> Result<CodecPixels, DecodeError> {
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
    guard_dimensions(w, h)?;
    let gray = dec.input_colorspace() == Some(ColorSpace::Luma);
    if gray {
        dec.set_options(dec.options().jpeg_set_out_colorspace(ColorSpace::Luma));
    }
    let pixels = dec
        .decode()
        .map_err(|e| DecodeError::Codec(format!("jpeg: {e}")))?;
    if gray {
        gray_from_tight(w, h, pixels)
    } else {
        rgb_from_tight(w, h, pixels)
    }
}
