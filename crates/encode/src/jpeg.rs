//! Baseline and progressive JPEG encoding of 8-bit gray and RGB bitmaps.

use std::io::Write;

use color::{Pixel, PixelMode};
use raster::Bitmap;

use crate::EncodeError;

use jpeg_encoder::{ColorType, Encoder};

/// How [`write_jpeg`] encodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JpegOptions {
    /// Quality, `1..=100`; values outside are clamped.
    pub quality: u8,
    /// Write a progressive JPEG instead of a baseline one.
    pub progressive: bool,
    /// Compute Huffman tables optimal for the image instead of the
    /// standard tables.
    pub optimize: bool,
}

impl Default for JpegOptions {
    /// Quality 75, baseline, the standard Huffman tables of ITU-T T.81
    /// Annex K.
    fn default() -> Self {
        Self {
            quality: 75,
            progressive: false,
            optimize: false,
        }
    }
}

/// Encode a gray (`Gray8`/`Mono8`), `Rgb8` or `Bgr8` bitmap as JPEG into `out`.
///
/// Stride padding is excluded: only the live `width` pixels of each row
/// are encoded.
///
/// # Errors
///
/// - [`EncodeError::UnsupportedMode`] for other pixel modes.
/// - [`EncodeError::InvalidInput`] for a zero-width or zero-height bitmap
///   (JPEG has no empty-image encoding), or when a dimension exceeds
///   `u16::MAX` (the JPEG format ceiling, 65535 px/side).
/// - [`EncodeError::Io`] when writing or flushing `out` fails.
/// - [`EncodeError::Jpeg`] on another encoder failure.
pub fn write_jpeg<P: Pixel, W: Write>(
    bitmap: &Bitmap<P>,
    options: JpegOptions,
    mut out: W,
) -> Result<(), EncodeError> {
    let (color, bytes_per_pixel) = match P::MODE {
        PixelMode::Mono8 => (ColorType::Luma, 1),
        PixelMode::Rgb8 => (ColorType::Rgb, 3),
        PixelMode::Bgr8 => (ColorType::Bgr, 3),
        PixelMode::Mono1 | PixelMode::Xbgr8 | PixelMode::Cmyk8 | PixelMode::DeviceN8 => {
            return Err(EncodeError::UnsupportedMode(
                "JPEG accepts Gray8, Rgb8 and Bgr8: convert other modes to Rgb8 first",
            ));
        }
    };
    if bitmap.width == 0 || bitmap.height == 0 {
        return Err(EncodeError::InvalidInput(
            "zero-width or zero-height bitmap: JPEG cannot encode an empty image",
        ));
    }
    let width = u16::try_from(bitmap.width)
        .map_err(|_| EncodeError::InvalidInput("width exceeds the JPEG limit of 65535 px"))?;
    let height = u16::try_from(bitmap.height)
        .map_err(|_| EncodeError::InvalidInput("height exceeds the JPEG limit of 65535 px"))?;

    let mut encoder = Encoder::new(&mut out, options.quality.clamp(1, 100));
    encoder.set_progressive(options.progressive);
    encoder.set_optimized_huffman_tables(options.optimize);
    // Both dimensions fit `u16`, so the row length and image size fit `usize`.
    let row = usize::from(width) * bytes_per_pixel;
    let size = row * usize::from(height);
    let result = if bitmap.stride == row {
        // No row padding: the backing buffer is the contiguous pixel stream.
        encoder.encode(&bitmap.data()[..size], width, height, color)
    } else {
        let mut packed = Vec::with_capacity(size);
        for y in 0..bitmap.height {
            packed.extend_from_slice(&bitmap.row_bytes(y)[..row]);
        }
        encoder.encode(&packed, width, height, color)
    };
    result?;
    out.flush()?;
    Ok(())
}

/// Encode an 8-bit grayscale (`Gray8`/`Mono8`) bitmap to baseline JPEG
/// bytes at `quality` (clamped to `1..=100`).
///
/// # Errors
///
/// As [`write_jpeg`]; a colour bitmap is [`EncodeError::UnsupportedMode`].
pub fn jpeg_gray<P: Pixel>(bitmap: &Bitmap<P>, quality: u8) -> Result<Vec<u8>, EncodeError> {
    if P::MODE != PixelMode::Mono8 {
        return Err(EncodeError::UnsupportedMode(
            "non-grayscale bitmap: jpeg_gray accepts Gray8/Mono8 only",
        ));
    }
    let mut out = Vec::new();
    write_jpeg(
        bitmap,
        JpegOptions {
            quality,
            ..JpegOptions::default()
        },
        &mut out,
    )?;
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

    /// A solid `w × h` RGB bitmap of `rgb`.
    fn solid_rgb(w: u32, h: u32, rgb: [u8; 3]) -> Bitmap<Rgb8> {
        let mut bmp = Bitmap::new(w, h, 1, false);
        for y in 0..h {
            for px in bmp.row_bytes_mut(y).as_chunks_mut::<3>().0 {
                *px = rgb;
            }
        }
        bmp
    }

    fn encode(bmp: &Bitmap<Rgb8>, options: JpegOptions) -> Vec<u8> {
        let mut out = Vec::new();
        write_jpeg(bmp, options, &mut out).expect("encodes");
        out
    }

    #[test]
    fn rgb_decodes_back_to_its_colour() {
        let bytes = encode(&solid_rgb(16, 8, [200, 40, 90]), JpegOptions::default());
        let mut decoder =
            zune_jpeg::JpegDecoder::new(zune_jpeg::zune_core::bytestream::ZCursor::new(&bytes));
        let pixels = decoder.decode().expect("decodes");
        assert_eq!(decoder.dimensions(), Some((16, 8)));
        for px in pixels.as_chunks::<3>().0 {
            for (got, want) in px.iter().zip([200u8, 40, 90]) {
                assert!(got.abs_diff(want) <= 3, "{px:?}");
            }
        }
    }

    #[test]
    fn progressive_writes_a_progressive_frame() {
        let bmp = solid_rgb(16, 16, [10, 120, 250]);
        let has_marker = |bytes: &[u8], m: u8| bytes.windows(2).any(|w| w == [0xFF, m]);
        let baseline = encode(&bmp, JpegOptions::default());
        let progressive = encode(
            &bmp,
            JpegOptions {
                progressive: true,
                ..JpegOptions::default()
            },
        );
        assert!(has_marker(&baseline, 0xC0) && !has_marker(&baseline, 0xC2));
        assert!(
            has_marker(&progressive, 0xC2),
            "SOF2 marks a progressive frame"
        );
    }

    #[test]
    fn cmyk_bitmaps_are_refused() {
        let bmp: Bitmap<color::Cmyk8> = Bitmap::new(8, 1, 1, false);
        let result = write_jpeg(&bmp, JpegOptions::default(), Vec::new());
        assert!(matches!(result, Err(EncodeError::UnsupportedMode(_))));
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
                *px = (x * 7 + y as usize * 13).to_le_bytes()[0].wrapping_mul(3);
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
    fn zero_dimension_is_invalid_input() {
        for (w, h) in [(0, 4), (4, 0), (0, 0)] {
            let bmp: Bitmap<Gray8> = Bitmap::new(w, h, 1, false);
            let result = jpeg_gray::<Gray8>(&bmp, 85);
            assert!(
                matches!(result, Err(EncodeError::InvalidInput(_))),
                "{w}x{h} bitmap must return InvalidInput, got {result:?}"
            );
        }
    }

    #[test]
    fn dimension_over_u16_returns_error() {
        let bmp: Bitmap<Gray8> = Bitmap::new(70_000, 1, 1, false);
        let result = jpeg_gray::<Gray8>(&bmp, 85);
        assert!(
            matches!(result, Err(EncodeError::InvalidInput(_))),
            "dimension > u16::MAX must return InvalidInput, got {result:?}"
        );
    }

    /// A sink whose every write fails.
    struct Full;

    impl std::io::Write for Full {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::new(std::io::ErrorKind::StorageFull, "full"))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn a_failing_sink_is_an_io_error() {
        let bmp = make_gray_bitmap(16, 16, 200);
        let result = write_jpeg(&bmp, JpegOptions::default(), Full);
        assert!(
            matches!(&result, Err(EncodeError::Io(e)) if e.kind() == std::io::ErrorKind::StorageFull),
            "{result:?}"
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

    #[test]
    fn multi_row_padded_stride_matches_tight() {
        // The single-row stride test cannot distinguish the stride==w fast
        // path from the row-strip path: with one row there is no inter-row
        // padding to interleave. A MULTI-row padded bitmap does — if the
        // fast path were taken on a padded buffer, padding bytes would mix
        // into the pixel stream between rows and the output would diverge
        // from the tight encoding. Identical output proves both branches
        // of the stride dispatch are correct.
        let w = 5;
        let h = 4;
        let mut padded: Bitmap<Gray8> = Bitmap::new(w, h, 8, false);
        let mut tight: Bitmap<Gray8> = Bitmap::new(w, h, 1, false);
        assert!(
            padded.stride > w as usize,
            "padded stride {} must exceed width {w}",
            padded.stride
        );
        assert_eq!(tight.stride, w as usize, "tight stride must equal width");

        for y in 0..h {
            // Distinct per-pixel content so any row/padding misalignment
            // changes the encoded bytes.
            let row: Vec<u8> = (0..w)
                .map(|x| (y * 31 + x * 7 + 1).to_le_bytes()[0])
                .collect();
            padded.row_bytes_mut(y)[..w as usize].copy_from_slice(&row);
            tight.row_bytes_mut(y)[..w as usize].copy_from_slice(&row);
        }

        let bytes_padded = jpeg_gray::<Gray8>(&padded, 85).unwrap();
        let bytes_tight = jpeg_gray::<Gray8>(&tight, 85).unwrap();
        assert_eq!(
            bytes_padded, bytes_tight,
            "multi-row padded vs tight must encode identically"
        );
    }
}
