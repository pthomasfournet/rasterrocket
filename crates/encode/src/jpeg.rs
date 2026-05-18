//! Baseline JPEG encoder for 8-bit grayscale (`Gray8`/`Mono8`) bitmaps.
//!
//! JPEG is absent from the Netpbm/PNG set because cloud OCR consumers
//! (Google Cloud Vision, GPT-5, Mistral) want a compact lossy payload.
//! This is a plain codec — no consumer-specific policy lives here.

use color::{Pixel, PixelMode};
use raster::Bitmap;

use crate::EncodeError;

use jpeg_encoder::{ColorType, Encoder};

/// Encode an 8-bit grayscale (`Gray8`/`Mono8`) bitmap to baseline JPEG bytes.
///
/// `quality` is clamped to `1..=100`. Stride padding is excluded — only the
/// live `width` pixels of each row are encoded.
///
/// # Errors
///
/// - [`EncodeError::UnsupportedMode`] for non-grayscale modes (use
///   [`write_ppm`][crate::write_ppm]/[`write_png`][crate::write_png]), for a
///   zero-width or zero-height bitmap (JPEG has no empty-image encoding), or
///   when a dimension exceeds `u16::MAX` (the JPEG format ceiling,
///   65535 px/side).
/// - [`EncodeError::Jpeg`] on an internal encoder failure.
pub fn jpeg_gray<P: Pixel>(bitmap: &Bitmap<P>, quality: u8) -> Result<Vec<u8>, EncodeError> {
    match P::MODE {
        PixelMode::Mono8 => {}
        PixelMode::Mono1
        | PixelMode::Rgb8
        | PixelMode::Bgr8
        | PixelMode::Xbgr8
        | PixelMode::Cmyk8
        | PixelMode::DeviceN8 => {
            return Err(EncodeError::UnsupportedMode(
                "non-grayscale bitmap: jpeg_gray accepts Gray8/Mono8 only",
            ));
        }
    }

    // Reject empty images here rather than letting the encoder surface a
    // library-internal `ZeroImageDimensions` as `EncodeError::Jpeg`: a
    // zero-sized bitmap is a caller-side input-shape violation, so it belongs
    // with the other `UnsupportedMode` precondition rejections.
    if bitmap.width == 0 || bitmap.height == 0 {
        return Err(EncodeError::UnsupportedMode(
            "zero-width or zero-height bitmap: JPEG cannot encode an empty image",
        ));
    }

    let width = u16::try_from(bitmap.width)
        .map_err(|_| EncodeError::UnsupportedMode("width exceeds JPEG limit (65535 px)"))?;
    let height = u16::try_from(bitmap.height)
        .map_err(|_| EncodeError::UnsupportedMode("height exceeds JPEG limit (65535 px)"))?;

    let q = quality.clamp(1, 100);

    // `width`/`height` are now proven to fit in `u16`, so the row count and
    // row length below fit in `usize` on every supported platform without a
    // lossy `as` cast.
    let w = usize::from(width);

    let mut out = Vec::new();
    let encoder = Encoder::new(&mut out, q);
    let result = if bitmap.stride == w {
        // No row padding: the backing buffer is already the contiguous pixel
        // stream the encoder wants, so feed it directly without a copy.
        encoder.encode(
            &bitmap.data()[..w * usize::from(height)],
            width,
            height,
            ColorType::Luma,
        )
    } else {
        let mut packed = Vec::with_capacity(w * usize::from(height));
        for y in 0..bitmap.height {
            packed.extend_from_slice(&bitmap.row_bytes(y)[..w]);
        }
        encoder.encode(&packed, width, height, ColorType::Luma)
    };
    result.map_err(EncodeError::Jpeg)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use color::{Gray8, Rgb8};
    use raster::Bitmap;

    fn make_gray_bitmap(w: u32, h: u32, fill: u8) -> Bitmap<Gray8> {
        let mut bmp = Bitmap::new(w, h, 1, false);
        for y in 0..h {
            bmp.row_bytes_mut(y).fill(fill);
        }
        bmp
    }

    #[test]
    fn jpeg_gray_produces_valid_jpeg_with_soi_marker() {
        let bmp = make_gray_bitmap(16, 16, 200);
        let bytes = jpeg_gray::<Gray8>(&bmp, 85).unwrap();
        assert_eq!(&bytes[..2], &[0xFF, 0xD8], "missing SOI marker");
        assert_eq!(
            &bytes[bytes.len() - 2..],
            &[0xFF, 0xD9],
            "missing EOI marker"
        );
    }

    #[test]
    fn higher_quality_yields_larger_output() {
        let mut bmp = Bitmap::<Gray8>::new(64, 64, 1, false);
        for y in 0..64 {
            let row = bmp.row_bytes_mut(y);
            for (x, px) in row.iter_mut().enumerate().take(64) {
                *px = (((x * 7 + y as usize * 13) % 256) as u8).wrapping_mul(3);
            }
        }
        let low = jpeg_gray::<Gray8>(&bmp, 20).unwrap();
        let high = jpeg_gray::<Gray8>(&bmp, 95).unwrap();
        assert!(
            high.len() > low.len(),
            "q95 ({}) should exceed q20 ({})",
            high.len(),
            low.len()
        );
    }

    #[test]
    fn rgb8_returns_unsupported_error() {
        let bmp: Bitmap<Rgb8> = Bitmap::new(1, 1, 1, false);
        let result = jpeg_gray::<Rgb8>(&bmp, 85);
        assert!(
            matches!(result, Err(EncodeError::UnsupportedMode(_))),
            "Rgb8 must return UnsupportedMode for jpeg_gray"
        );
    }

    #[test]
    fn zero_dimension_returns_unsupported_error() {
        for (w, h) in [(0, 4), (4, 0), (0, 0)] {
            let bmp: Bitmap<Gray8> = Bitmap::new(w, h, 1, false);
            let result = jpeg_gray::<Gray8>(&bmp, 85);
            assert!(
                matches!(result, Err(EncodeError::UnsupportedMode(_))),
                "{w}x{h} bitmap must return UnsupportedMode, got {result:?}"
            );
        }
    }

    #[test]
    fn dimension_over_u16_returns_error() {
        let bmp: Bitmap<Gray8> = Bitmap::new(70_000, 1, 1, false);
        let result = jpeg_gray::<Gray8>(&bmp, 85);
        assert!(
            matches!(result, Err(EncodeError::UnsupportedMode(_))),
            "dimension > u16::MAX must return UnsupportedMode, got {result:?}"
        );
    }

    #[test]
    fn stride_padding_excluded() {
        // Gray8 is 1 byte/pixel: row_pad=4 → stride=4, row_pad=1 → stride=3.
        // These bitmaps carry identical live pixels but different stride values.
        // If the encoder used the full stride rather than the live width, the
        // padding byte (0x00) would mix into one stream and the outputs would
        // diverge — so byte-identical output is the proof that stride exclusion works.
        let mut padded: Bitmap<Gray8> = Bitmap::new(3, 1, 4, false);
        let mut tight: Bitmap<Gray8> = Bitmap::new(3, 1, 1, false);

        // Confirm the strides genuinely differ before asserting anything else.
        assert_eq!(padded.stride, 4, "padded bitmap should have stride 4");
        assert_eq!(tight.stride, 3, "tight bitmap should have stride 3");

        // Write the same non-zero pixels into the 3 live columns of both bitmaps.
        padded.row_bytes_mut(0)[..3].copy_from_slice(&[10, 20, 30]);
        tight.row_bytes_mut(0)[..3].copy_from_slice(&[10, 20, 30]);

        let bytes_padded = jpeg_gray::<Gray8>(&padded, 85).unwrap();
        let bytes_tight = jpeg_gray::<Gray8>(&tight, 85).unwrap();

        assert_eq!(
            bytes_padded, bytes_tight,
            "stride padding must not affect JPEG output"
        );
        assert_eq!(&bytes_tight[..2], &[0xFF, 0xD8], "missing SOI marker");
    }
}
