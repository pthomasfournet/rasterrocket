//! JPEG XL decode for comic pages, via the pure-Rust `jxl` decoder of the
//! JPEG XL project.
//!
//! The first frame is the page: an animation yields its first frame.  The
//! decoder's output is 8-bit gray for a gray image and 8-bit RGB otherwise,
//! in its default output colour space; extra channels (alpha among them) are
//! not read.

use jxl::api::{
    Event, JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
    JxlPixelFormat,
};

use super::{CodecPixels, DecodeError, gray_from_tight, guard_dimensions, rgb_from_tight};

/// The most samples (pixels × channels) the decoder may allocate for the
/// image or any of its frames: 1 Gi, the image decode budget of the PDF path.
const MAX_SAMPLES: usize = 1 << 30;

/// Decode a JPEG XL codestream or container to a `Gray8` or `Rgb8` bitmap.
///
/// # Errors
///
/// [`DecodeError::TooLarge`] when the header dimensions exceed the page
/// limits; [`DecodeError::Codec`] on malformed, truncated or unsupported
/// streams.
pub(super) fn decode(bytes: &[u8]) -> Result<CodecPixels, DecodeError> {
    let codec = |e: jxl::error::Error| DecodeError::Codec(format!("jxl: {e}"));
    let truncated = || DecodeError::Codec("jxl: the stream ends before the image".to_owned());
    let mut options = JxlDecoderOptions::default();
    options.sample_limit = Some(MAX_SAMPLES);
    let mut dec = JxlDecoder::new(options);
    let mut input = bytes;

    loop {
        match dec.process(&mut input, None, None).map_err(codec)? {
            Event::BasicInfo => break,
            Event::NeedMoreInput { .. } | Event::Complete => return Err(truncated()),
            Event::FrameHeader | Event::FrameComplete { .. } => {}
        }
    }
    let info = dec.basic_info().ok_or_else(truncated)?;
    let (w, h) = info.size;
    let extra_channels = info.extra_channels.len();
    let side = |v: usize| {
        u32::try_from(v).map_err(|_| DecodeError::Codec(format!("jxl: side of {v} pixels")))
    };
    let (w, h) = (side(w)?, side(h)?);
    if w == 0 || h == 0 {
        return Err(DecodeError::Codec(format!("jxl: zero dimensions {w}x{h}")));
    }
    guard_dimensions(w, h)?;
    let gray = dec
        .current_pixel_format()
        .is_some_and(|format| format.color_type.is_grayscale());
    let (color_type, channels) = if gray {
        (JxlColorType::Grayscale, 1)
    } else {
        (JxlColorType::Rgb, 3)
    };
    dec.set_pixel_format(JxlPixelFormat {
        color_type,
        color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
        extra_channel_format: vec![None; extra_channels],
    })
    .map_err(codec)?;

    let row_bytes = w as usize * channels;
    let mut pixels = vec![0u8; row_bytes * h as usize];
    loop {
        let mut out = [JxlOutputBuffer::new(&mut pixels, h as usize, row_bytes)];
        match dec
            .process(&mut input, Some(&mut out), None)
            .map_err(codec)?
        {
            Event::FrameComplete { .. } => break,
            Event::NeedMoreInput { .. } | Event::Complete => return Err(truncated()),
            Event::BasicInfo | Event::FrameHeader => {}
        }
    }
    if gray {
        gray_from_tight(w, h, pixels)
    } else {
        rgb_from_tight(w, h, pixels)
    }
}

#[cfg(test)]
mod tests {
    use super::decode;
    use crate::decode::CodecPixels;

    /// 16 × 8 lossless gray, sample `(16x + y) mod 256` (see `jxl-fixtures.py`).
    const GRAY: &[u8] = include_bytes!("../../tests/data/gray-16x8.jxl");
    /// 8 × 4 lossless RGB, pixel `(30x, 60y, 255 − 30x)`.
    const RGB: &[u8] = include_bytes!("../../tests/data/rgb-8x4.jxl");

    #[test]
    fn a_lossless_gray_image_decodes_exactly() {
        let Ok(CodecPixels::Gray(bmp)) = decode(GRAY) else {
            panic!("a gray image decodes to gray")
        };
        assert_eq!((bmp.width, bmp.height), (16, 8));
        for y in 0..8 {
            let row: Vec<u8> = (0..16)
                .map(|x| u8::try_from((x * 16 + y) % 256).unwrap())
                .collect();
            assert_eq!(&bmp.row_bytes(y)[..16], row.as_slice(), "row {y}");
        }
    }

    #[test]
    fn a_lossless_rgb_image_decodes_exactly() {
        let Ok(CodecPixels::Rgb(bmp)) = decode(RGB) else {
            panic!("an RGB image decodes to RGB")
        };
        assert_eq!((bmp.width, bmp.height), (8, 4));
        for y in 0..4u32 {
            let row: Vec<u8> = (0..8u32)
                .flat_map(|x| [x * 30, y * 60, 255 - x * 30])
                .map(|v| u8::try_from(v).unwrap())
                .collect();
            assert_eq!(&bmp.row_bytes(y)[..24], row.as_slice(), "row {y}");
        }
    }

    #[test]
    fn a_lossy_image_decodes_to_srgb() {
        // Stored in XYB; the default output is sRGB, so the decoded pixels
        // stay near the encoded ones: the reference decoder's mean error is
        // 1.1 levels, where reading the output as linear light would put the
        // constant blue channel near 56 instead of 128.
        let lossy: &[u8] = include_bytes!("../../tests/data/lossy-32x16.jxl");
        let Ok(CodecPixels::Rgb(bmp)) = decode(lossy) else {
            panic!("a colour image decodes to RGB")
        };
        assert_eq!((bmp.width, bmp.height), (32, 16));
        let (mut total, mut samples) = (0u32, 0u32);
        for y in 0..16u32 {
            for (x, px) in (0..32u32).zip(bmp.row_bytes(y).as_chunks::<3>().0) {
                for (got, want) in px.iter().zip([x * 8, y * 16, 128]) {
                    total += u32::from(*got).abs_diff(want);
                    samples += 1;
                }
            }
        }
        assert!(
            total <= 3 * samples,
            "mean difference {}",
            f64::from(total) / f64::from(samples)
        );
    }

    #[test]
    fn truncated_or_foreign_bytes_are_codec_errors() {
        for cut in [2, GRAY.len() / 2, GRAY.len() - 1] {
            assert!(decode(&GRAY[..cut]).is_err(), "cut at {cut}");
        }
        assert!(decode(b"\xFF\x0Anot a codestream").is_err());
    }
}
