//! WebP decode for comic pages, via the pure-Rust `image-webp` crate.

use color::Rgb8;
use raster::Bitmap;

use super::{DecodeError, guard_dimensions, rgb_bitmap_from_tight};

/// Decode WebP bytes (lossy or lossless) to an `Rgb8` bitmap, dropping alpha.
///
/// # Errors
///
/// [`DecodeError::Codec`] on malformed/unsupported streams.
pub fn decode(bytes: &[u8]) -> Result<Bitmap<Rgb8>, DecodeError> {
    let mut dec = image_webp::WebPDecoder::new(std::io::Cursor::new(bytes))
        .map_err(|e| DecodeError::Codec(format!("webp: {e}")))?;
    let (w, h) = dec.dimensions();
    if w == 0 || h == 0 {
        return Err(DecodeError::Codec(format!("webp: zero dimensions {w}x{h}")));
    }
    guard_dimensions("webp", w, h)?;
    // Size the buffer with the decoder's own contract (w*h*channels, where
    // channels follows has_alpha) rather than recomputing the channel rule here.
    let buf_len = dec
        .output_buffer_size()
        .ok_or_else(|| DecodeError::Codec("webp: image too large".to_owned()))?;
    let has_alpha = dec.has_alpha();
    let mut buf = vec![0u8; buf_len];
    dec.read_image(&mut buf)
        .map_err(|e| DecodeError::Codec(format!("webp: {e}")))?;
    let rgb = if has_alpha {
        let mut out = Vec::with_capacity((w as usize) * (h as usize) * 3);
        for px in buf.chunks_exact(4) {
            out.extend_from_slice(&px[..3]);
        }
        out
    } else {
        buf
    };
    rgb_bitmap_from_tight(w, h, &rgb)
}
