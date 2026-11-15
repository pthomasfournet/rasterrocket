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

use super::{CodecPixels, DecodeError, Target, gray_from_tight, guard_dimensions, rgb_from_tight};

/// Decode baseline/progressive JPEG bytes: a single-component stream to a
/// `Gray8` bitmap, any other to an `Rgb8` bitmap.
///
/// The frame header is read first and its dimensions checked by
/// [`guard_dimensions`] before the pixel buffer is allocated.  The decoder's
/// own per-side cap is raised to the shared limit so that check is the single
/// size policy across codecs.
///
/// A stream whose scan data is incomplete or damaged is drawn as far as it
/// decodes, with a warning.
///
/// # Errors
///
/// [`DecodeError::TooLarge`] when the frame exceeds the rendered-page
/// limits; [`DecodeError::Codec`] if the stream is not decodable or reports
/// zero dimensions.
pub(super) fn decode(bytes: &[u8], target: Target) -> Result<CodecPixels, DecodeError> {
    let (pixels, intact_rows, height) = decode_counting_rows(bytes, target)?;
    if intact_rows < height {
        log::warn!(
            "jpeg: the scan data is incomplete or damaged ({intact_rows} of {height} rows are \
             whole); the page is drawn as far as it decodes"
        );
    }
    Ok(pixels)
}

/// [`decode`], with the number of leading rows decoded from scan data that
/// was present and well formed, and the frame height.
fn decode_counting_rows(
    bytes: &[u8],
    target: Target,
) -> Result<(CodecPixels, u32, u32), DecodeError> {
    let side_cap = pdf_raster::MAX_PX_SIDE as usize;
    // Not strict: the rows a damaged stream still holds are drawn, and the
    // decoder counts them.
    let opts = DecoderOptions::default()
        .jpeg_set_out_colorspace(ColorSpace::RGB)
        .set_max_width(side_cap)
        .set_max_height(side_cap)
        .set_strict_mode(false);
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
    guard_dimensions(w, h, target)?;
    let gray = dec.input_colorspace() == Some(ColorSpace::Luma);
    if gray {
        dec.set_options(dec.options().jpeg_set_out_colorspace(ColorSpace::Luma));
    }
    let pixels = dec
        .decode()
        .map_err(|e| DecodeError::Codec(format!("jpeg: {e}")))?;
    // At most the frame height, which fits `u16`.
    let intact_rows = u32::try_from(dec.intact_rows()).map_or(h, |rows| rows.min(h));
    let pixels = if gray {
        gray_from_tight(w, h, pixels)
    } else {
        rgb_from_tight(w, h, pixels)
    }?;
    Ok((pixels, intact_rows, h))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 4:2:0 colour frame of noise with restart markers, 64 rows.
    const NOISE: &[u8] = include_bytes!("../../../zune_jpeg/tests/images/cut/noise_420_rst.jpg");

    fn rgb(pixels: CodecPixels) -> Vec<u8> {
        match pixels {
            CodecPixels::Rgb(bitmap) => bitmap.data().to_vec(),
            _ => panic!("a colour frame decodes to 8-bit RGB"),
        }
    }

    #[test]
    fn a_complete_stream_is_whole() {
        let (pixels, intact_rows, height) =
            decode_counting_rows(NOISE, crate::decode::EIGHT).expect("decodes");
        assert_eq!(intact_rows, height);
        assert_eq!(
            rgb(pixels),
            rgb(decode(NOISE, crate::decode::EIGHT).expect("decodes"))
        );
    }

    #[test]
    fn a_stream_cut_short_is_drawn_and_counted() {
        let (whole, _, height) =
            decode_counting_rows(NOISE, crate::decode::EIGHT).expect("decodes");
        let whole = rgb(whole);
        let width = whole.len() / 3 / height as usize;

        let cut = &NOISE[..NOISE.len() * 2 / 3];
        let (pixels, intact_rows, cut_height) =
            decode_counting_rows(cut, crate::decode::EIGHT).expect("decodes");
        assert_eq!(cut_height, height);
        assert!(
            0 < intact_rows && intact_rows < height,
            "{intact_rows} of {height} rows"
        );
        // The rows counted whole are the complete decode's, but for the last
        // two, whose chroma is interpolated from the row of MCUs below.
        let same = (intact_rows as usize).saturating_sub(2) * width * 3;
        let pixels = rgb(pixels);
        assert_eq!(pixels.len(), whole.len());
        assert_eq!(pixels[..same], whole[..same]);
    }
}
