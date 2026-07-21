//! PNG decode for comic pages, via the `png` crate's reader.

use color::Rgb8;
use raster::Bitmap;

use super::{DecodeError, guard_dimensions, rgb_bitmap_from_tight};

/// Decode PNG bytes to an `Rgb8` bitmap, expanding palette/gray/16-bit and
/// dropping alpha to opaque RGB.
///
/// # Errors
///
/// [`DecodeError::Codec`] on any malformed-stream or unsupported-config error.
pub fn decode(bytes: &[u8]) -> Result<Bitmap<Rgb8>, DecodeError> {
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
    guard_dimensions("png", w, h)?;
    let size = reader
        .output_buffer_size()
        .ok_or_else(|| DecodeError::Codec("png: output buffer size overflows".to_owned()))?;
    let mut buf = vec![0u8; size];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| DecodeError::Codec(format!("png: {e}")))?;
    let frame = &buf[..info.buffer_size()];
    let rgb = match info.color_type {
        png::ColorType::Rgb => frame.to_vec(),
        png::ColorType::Rgba => drop_alpha(frame, 4),
        png::ColorType::Grayscale => expand_gray(frame, 1),
        png::ColorType::GrayscaleAlpha => expand_gray(frame, 2),
        png::ColorType::Indexed => {
            return Err(DecodeError::Codec(
                "png: palette not expanded (unexpected with EXPAND)".to_owned(),
            ));
        }
    };
    rgb_bitmap_from_tight(w, h, &rgb)
}

/// Drop the trailing alpha byte from each `stride`-byte pixel, keeping 3 RGB.
fn drop_alpha(src: &[u8], stride: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len() / stride * 3);
    for px in src.chunks_exact(stride) {
        out.extend_from_slice(&px[..3]);
    }
    out
}

/// Expand a 1- or 2-channel gray(+alpha) buffer to RGB by replicating luma.
fn expand_gray(src: &[u8], stride: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len() / stride * 3);
    for px in src.chunks_exact(stride) {
        out.extend_from_slice(&[px[0], px[0], px[0]]);
    }
    out
}
