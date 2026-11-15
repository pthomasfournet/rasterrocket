//! PNG decode for comic pages, via the `png` crate's reader.

use super::{
    CodecPixels, DecodeError, Target, be_samples, gray_from_tight, gray16_from_tight,
    guard_dimensions, keep_leading_samples, rgb_from_tight, rgb16_from_tight,
};

/// Decode PNG bytes, expanding palette and low-bit samples and dropping
/// alpha: gray and gray+alpha streams yield a gray bitmap, every other
/// colour type an RGB bitmap.  A 16-bit stream keeps its samples on a
/// 16-bit target (`Gray16`, `Rgb16`) and is reduced to 8 bits otherwise.
///
/// # Errors
///
/// [`DecodeError::Codec`] on any malformed-stream or unsupported-config error.
pub(super) fn decode(bytes: &[u8], target: Target) -> Result<CodecPixels, DecodeError> {
    let codec = |e: png::DecodingError| DecodeError::Codec(format!("png: {e}"));
    // `png` 0.18 requires `Read + Seek`; `&[u8]` is only `Read`, so wrap it.
    let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    let deep = target.deep()
        && decoder.read_header_info().map_err(codec)?.bit_depth == png::BitDepth::Sixteen;
    decoder.set_transformations(if deep {
        png::Transformations::EXPAND
    } else {
        png::Transformations::EXPAND | png::Transformations::STRIP_16
    });
    let mut reader = decoder.read_info().map_err(codec)?;
    // read_info() parses only the IHDR header; guard the header-claimed
    // dimensions before output_buffer_size() (= width*height*channels) is
    // allocated, so a crafted PNG can't force a giant vec! before it decodes.
    let (w, h) = (reader.info().width, reader.info().height);
    if w == 0 || h == 0 {
        return Err(DecodeError::Codec(format!("png: zero dimensions {w}x{h}")));
    }
    guard_dimensions(w, h, target)?;
    let size = reader
        .output_buffer_size()
        .ok_or_else(|| DecodeError::Codec("png: output buffer size overflows".to_owned()))?;
    let mut buf = vec![0u8; size];
    let info = reader.next_frame(&mut buf).map_err(codec)?;
    buf.truncate(info.buffer_size());
    let channels = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Indexed => {
            return Err(DecodeError::Codec(
                "png: palette not expanded (unexpected with EXPAND)".to_owned(),
            ));
        }
    };
    if info.bit_depth == png::BitDepth::Sixteen {
        let (samples, rgb) = colour_samples(be_samples(&buf), channels);
        return if rgb {
            rgb16_from_tight(w, h, samples)
        } else {
            gray16_from_tight(w, h, samples)
        };
    }
    let (samples, rgb) = colour_samples(buf, channels);
    if rgb {
        rgb_from_tight(w, h, samples)
    } else {
        gray_from_tight(w, h, samples)
    }
}

/// The colour samples of `samples`, pixels of `channels` samples with any
/// alpha dropped, and whether they are RGB (else gray).
fn colour_samples<T: Copy>(mut samples: Vec<T>, channels: usize) -> (Vec<T>, bool) {
    let keep = if channels >= 3 { 3 } else { 1 };
    if channels != keep {
        keep_leading_samples(&mut samples, channels, keep);
    }
    (samples, keep == 3)
}
