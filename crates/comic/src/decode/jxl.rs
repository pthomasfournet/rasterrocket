//! JPEG XL decode for comic pages, via the pure-Rust `jxl` decoder of the
//! JPEG XL project.
//!
//! The first frame is the page: an animation yields its first frame.  The
//! decoder's output is gray for a gray image and RGB otherwise, in its
//! default output colour space: 16-bit for an image of more than 8 bits
//! per sample on a 16-bit target, 8-bit otherwise.  Extra channels (alpha
//! among them) are not read.

use jxl::api::{
    Endianness, Event, JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
    JxlPixelFormat,
};

use super::{
    CodecPixels, DecodeError, MAX_SAMPLES, Target, be_samples, gray_from_tight, gray16_from_tight,
    guard_dimensions, rgb_from_tight, rgb16_from_tight,
};

/// Decode a JPEG XL codestream or container to a gray or RGB bitmap.
///
/// # Errors
///
/// [`DecodeError::TooLarge`] when the header dimensions exceed the page
/// limits; [`DecodeError::Codec`] on malformed, truncated or unsupported
/// streams.
pub(super) fn decode(bytes: &[u8], target: Target) -> Result<CodecPixels, DecodeError> {
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
    let deep = target.deep() && info.bit_depth.bits_per_sample() > 8;
    let side = |v: usize| {
        u32::try_from(v).map_err(|_| DecodeError::Codec(format!("jxl: side of {v} pixels")))
    };
    let (w, h) = (side(w)?, side(h)?);
    if w == 0 || h == 0 {
        return Err(DecodeError::Codec(format!("jxl: zero dimensions {w}x{h}")));
    }
    guard_dimensions(w, h, target)?;
    let gray = dec
        .current_pixel_format()
        .is_some_and(|format| format.color_type.is_grayscale());
    let (color_type, channels) = if gray {
        (JxlColorType::Grayscale, 1)
    } else {
        (JxlColorType::Rgb, 3)
    };
    let (format, sample_bytes) = if deep {
        let format = JxlDataFormat::U16 {
            endianness: Endianness::BigEndian,
            bit_depth: 16,
        };
        (format, 2)
    } else {
        (JxlDataFormat::U8 { bit_depth: 8 }, 1)
    };
    dec.set_pixel_format(JxlPixelFormat {
        color_type,
        color_data_format: Some(format),
        extra_channel_format: vec![None; extra_channels],
    })
    .map_err(codec)?;

    let row_bytes = w as usize * channels * sample_bytes;
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
    if deep {
        let samples = be_samples(&pixels);
        return if gray {
            gray16_from_tight(w, h, samples)
        } else {
            rgb16_from_tight(w, h, samples)
        };
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
        let Ok(CodecPixels::Gray(bmp)) = decode(GRAY, crate::decode::EIGHT) else {
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
        let Ok(CodecPixels::Rgb(bmp)) = decode(RGB, crate::decode::EIGHT) else {
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
        let Ok(CodecPixels::Rgb(bmp)) = decode(lossy, crate::decode::EIGHT) else {
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
            assert!(
                decode(&GRAY[..cut], crate::decode::EIGHT).is_err(),
                "cut at {cut}"
            );
        }
        assert!(decode(b"\xFF\x0Anot a codestream", crate::decode::EIGHT).is_err());
    }

    #[test]
    fn a_lossless_16_bit_image_keeps_its_samples_on_a_16_bit_target() {
        // 16 × 8 lossless 16-bit gray, sample `(4099x + 8191y) mod 65536`.
        const DEEP: &[u8] = include_bytes!("../../tests/data/gray16-16x8.jxl");
        let Ok(CodecPixels::Gray16(bmp)) = decode(DEEP, crate::decode::SIXTEEN) else {
            panic!("a 16-bit gray image decodes to Gray16")
        };
        assert_eq!((bmp.width, bmp.height), (16, 8));
        for y in 0..8u32 {
            let row: Vec<u16> = (0..16u32)
                .map(|x| u16::try_from((x * 4099 + y * 8191) % 65_536).unwrap())
                .collect();
            assert_eq!(&bmp.row_samples(y)[..16], row.as_slice(), "row {y}");
        }
        // At 8 bits the same image is reduced; an 8-bit image stays 8-bit.
        assert!(matches!(
            decode(DEEP, crate::decode::EIGHT),
            Ok(CodecPixels::Gray(_))
        ));
        assert!(matches!(
            decode(GRAY, crate::decode::SIXTEEN),
            Ok(CodecPixels::Gray(_))
        ));
    }
}
