//! PNG decode for comic pages, via the `png` crate's reader.

use super::{
    CodecPixels, DecodeError, gray_from_tight, guard_dimensions, keep_leading_samples,
    rgb_from_tight,
};

/// Decode PNG bytes, expanding palette and 16-bit samples and dropping alpha:
/// gray and gray+alpha streams yield a `Gray8` bitmap, every other colour
/// type an `Rgb8` bitmap.
///
/// # Errors
///
/// [`DecodeError::Codec`] on any malformed-stream or unsupported-config error.
pub(super) fn decode(bytes: &[u8]) -> Result<CodecPixels, DecodeError> {
    // `png` 0.18 requires `Read + Seek`; `&[u8]` is only `Read`, so wrap it.
    let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder
        .read_info()
        .map_err(|e| DecodeError::Codec(format!("png: {e}")))?;
    // read_info() parses only the IHDR header; guard the header-claimed
    // dimensions before output_buffer_size() (= width*height*channels) is
    // allocated, so a crafted PNG can't force a giant vec! before it decodes.
    let (w, h) = (reader.info().width, reader.info().height);
    if w == 0 || h == 0 {
        return Err(DecodeError::Codec(format!("png: zero dimensions {w}x{h}")));
    }
    guard_dimensions(w, h)?;
    let size = reader
        .output_buffer_size()
        .ok_or_else(|| DecodeError::Codec("png: output buffer size overflows".to_owned()))?;
    let mut buf = vec![0u8; size];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| DecodeError::Codec(format!("png: {e}")))?;
    buf.truncate(info.buffer_size());
    match info.color_type {
        png::ColorType::Rgb => rgb_from_tight(w, h, buf),
        png::ColorType::Rgba => {
            keep_leading_samples(&mut buf, 4, 3);
            rgb_from_tight(w, h, buf)
        }
        png::ColorType::Grayscale => gray_from_tight(w, h, buf),
        png::ColorType::GrayscaleAlpha => {
            keep_leading_samples(&mut buf, 2, 1);
            gray_from_tight(w, h, buf)
        }
        png::ColorType::Indexed => Err(DecodeError::Codec(
            "png: palette not expanded (unexpected with EXPAND)".to_owned(),
        )),
    }
}
