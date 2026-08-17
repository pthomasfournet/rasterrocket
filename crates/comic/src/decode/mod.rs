//! Loose-image decode front door.
//!
//! One place that sniffs an image's format by magic bytes, dispatches to the
//! per-codec wrapper, and normalises the result to grayscale — the single
//! decode→bitmap→gray sequence shared by every format. The per-page size guard
//! lives in `rasterrocket` and is applied by the caller so the size error can
//! name the offending archive entry.

mod jpeg;
mod png;
mod tiff;
mod webp;

use color::Rgb8;
use raster::Bitmap;

/// A decoded page in OCR-ready grayscale, plus its pixel dimensions.
pub(crate) struct DecodedImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Tight `width*height` 8-bit grayscale buffer, top-to-bottom.
    pub gray: Vec<u8>,
}

/// Recognised container formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ImageFormat {
    /// JPEG (`FF D8 FF`).
    Jpeg,
    /// PNG (`89 50 4E 47`).
    Png,
    /// WebP (`RIFF....WEBP`).
    WebP,
    /// TIFF (`II*\0` little-endian or `MM\0*` big-endian).
    Tiff,
}

/// Why decoding a single entry failed.
#[derive(Debug)]
pub(crate) enum DecodeError {
    /// No recognised image magic bytes.
    Unsupported,
    /// The codec rejected the byte stream.
    Codec(String),
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported => write!(f, "unrecognised image format"),
            Self::Codec(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for DecodeError {}

/// Detect the image format from leading magic bytes. `None` if unrecognised.
#[must_use]
pub(crate) fn sniff(bytes: &[u8]) -> Option<ImageFormat> {
    if bytes.len() >= 3 && bytes[..3] == [0xFF, 0xD8, 0xFF] {
        return Some(ImageFormat::Jpeg);
    }
    if bytes.len() >= 8 && bytes[..8] == [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A] {
        return Some(ImageFormat::Png);
    }
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return Some(ImageFormat::WebP);
    }
    if bytes.len() >= 4
        && (bytes[..4] == [0x49, 0x49, 0x2A, 0x00] || bytes[..4] == [0x4D, 0x4D, 0x00, 0x2A])
    {
        return Some(ImageFormat::Tiff);
    }
    None
}

/// Decode arbitrary image bytes to grayscale, sniffing the format first.
///
/// # Errors
///
/// [`DecodeError::Unsupported`] if no codec matches the magic bytes;
/// [`DecodeError::Codec`] if the matched codec rejects the stream.
pub(crate) fn decode_image(bytes: &[u8]) -> Result<DecodedImage, DecodeError> {
    let rgb = match sniff(bytes).ok_or(DecodeError::Unsupported)? {
        ImageFormat::Jpeg => jpeg::decode(bytes)?,
        ImageFormat::Png => png::decode(bytes)?,
        ImageFormat::WebP => webp::decode(bytes)?,
        ImageFormat::Tiff => tiff::decode(bytes)?,
    };
    let gray_bmp = pdf_raster::rgb_to_gray(&rgb);
    let w = gray_bmp.width as usize;
    let mut gray = Vec::with_capacity(w * gray_bmp.height as usize);
    for y in 0..gray_bmp.height {
        gray.extend_from_slice(&gray_bmp.row_bytes(y)[..w]);
    }
    Ok(DecodedImage {
        width: gray_bmp.width,
        height: gray_bmp.height,
        gray,
    })
}

/// Reject an image whose header dimensions exceed the rendered-page size limits
/// BEFORE the codec allocates its full pixel buffer (a decode-bomb guard).
///
/// `validate_dimensions` enforces the same per-side and total-area caps the PDF
/// render path uses; calling it here, on the header dims, stops a hostile image
/// from forcing a multi-gigabyte allocation that the post-decode check would
/// only catch after the memory was already taken.
fn guard_dimensions(codec: &str, w: u32, h: u32) -> Result<(), DecodeError> {
    pdf_raster::validate_dimensions(w, h)
        .map_err(|_| DecodeError::Codec(format!("{codec}: image too large ({w}x{h})")))
}

/// Build an `Rgb8` bitmap from a tight interleaved RGB buffer. Shared by the
/// per-codec wrappers so each only has to produce `(w, h, Vec<u8>)`.
fn rgb_bitmap_from_tight(w: u32, h: u32, rgb: &[u8]) -> Result<Bitmap<Rgb8>, DecodeError> {
    let expected = (w as usize)
        .checked_mul(h as usize)
        .and_then(|n| n.checked_mul(3));
    if expected != Some(rgb.len()) {
        return Err(DecodeError::Codec(format!(
            "pixel buffer {} bytes inconsistent with {w}x{h}x3",
            rgb.len()
        )));
    }
    let mut bmp = Bitmap::<Rgb8>::new(w, h, 1, false);
    for y in 0..h {
        let src = &rgb[(y as usize * w as usize * 3)..((y as usize + 1) * w as usize * 3)];
        bmp.row_bytes_mut(y)[..src.len()].copy_from_slice(src);
    }
    Ok(bmp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use color::{Gray8, Rgb8};
    use raster::Bitmap;

    fn solid_png(w: u32, h: u32, rgb: [u8; 3]) -> Vec<u8> {
        let mut bmp = Bitmap::<Rgb8>::new(w, h, 1, false);
        for y in 0..h {
            let row = bmp.row_bytes_mut(y);
            for x in 0..w as usize {
                row[x * 3..x * 3 + 3].copy_from_slice(&rgb);
            }
        }
        let mut out = Vec::new();
        encode::write_png(&bmp, &mut out).unwrap();
        out
    }

    fn solid_jpeg_gray(w: u32, h: u32, v: u8) -> Vec<u8> {
        let mut bmp = Bitmap::<Gray8>::new(w, h, 1, false);
        for y in 0..h {
            for x in 0..w as usize {
                bmp.row_bytes_mut(y)[x] = v;
            }
        }
        encode::jpeg_gray::<Gray8>(&bmp, 90).unwrap()
    }

    /// A solid 8-bit grayscale TIFF, written with the `tiff` crate's own
    /// encoder so the decode path (`DecodingResult::U8` + `ColorType::Gray`)
    /// is exercised end-to-end without a checked-in binary fixture.
    fn solid_tiff_gray(w: u32, h: u32, v: u8) -> Vec<u8> {
        // `::tiff` (the external crate), not the local `tiff` submodule which
        // shadows the crate name inside this module.
        use ::tiff::encoder::{TiffEncoder, colortype};
        let pixels = vec![v; (w * h) as usize];
        let mut out = std::io::Cursor::new(Vec::new());
        TiffEncoder::new(&mut out)
            .unwrap()
            .write_image::<colortype::Gray8>(w, h, &pixels)
            .unwrap();
        out.into_inner()
    }

    #[test]
    fn sniff_detects_by_magic_not_extension() {
        let png = solid_png(2, 2, [255, 0, 0]);
        assert_eq!(sniff(&png), Some(ImageFormat::Png));
        let jpg = solid_jpeg_gray(2, 2, 128);
        assert_eq!(sniff(&jpg), Some(ImageFormat::Jpeg));
        assert_eq!(sniff(b"II*\0and more"), Some(ImageFormat::Tiff));
        assert_eq!(sniff(b"MM\0*and more"), Some(ImageFormat::Tiff));
        let tif = solid_tiff_gray(2, 2, 80);
        assert_eq!(sniff(&tif), Some(ImageFormat::Tiff));
        assert_eq!(sniff(b"not an image"), None);
    }

    #[test]
    fn decode_png_to_expected_pixels() {
        let png = solid_png(2, 2, [10, 200, 30]);
        let img = decode_image(&png).expect("png decodes");
        assert_eq!((img.width, img.height), (2, 2));
        assert_eq!(img.gray.len(), 4);
        assert!(
            img.gray.iter().all(|&p| p == img.gray[0]),
            "solid stays solid"
        );
        assert!(img.gray[0] > 30 && img.gray[0] < 200);
    }

    #[test]
    fn decode_jpeg_to_gray() {
        let jpg = solid_jpeg_gray(4, 4, 137);
        let img = decode_image(&jpg).expect("jpeg decodes");
        assert_eq!((img.width, img.height), (4, 4));
        assert!(img.gray.iter().all(|&p| (i16::from(p) - 137).abs() <= 4));
    }

    #[test]
    fn decode_tiff_gray_to_expected_pixels() {
        // Exercises the TIFF wrapper's DecodingResult::U8 + ColorType::Gray
        // path end-to-end (the most logic-heavy codec match).
        let tif = solid_tiff_gray(3, 2, 90);
        let img = decode_image(&tif).expect("tiff decodes");
        assert_eq!((img.width, img.height), (3, 2));
        assert_eq!(img.gray.len(), 6);
        assert!(
            img.gray.iter().all(|&p| p == 90),
            "solid gray 90 must round-trip exactly (lossless), got {:?}",
            img.gray
        );
    }

    #[test]
    fn truncated_bytes_error_not_panic() {
        let mut png = solid_png(8, 8, [1, 2, 3]);
        png.truncate(png.len() / 2);
        assert!(decode_image(&png).is_err(), "must return Err, not panic");
    }

    #[test]
    fn unknown_format_is_unsupported() {
        assert!(matches!(
            decode_image(b"xxxx not an image xxxx"),
            Err(DecodeError::Unsupported)
        ));
    }

    /// Standard IEEE CRC-32 (the polynomial PNG chunks use), computed without a
    /// dependency so the bomb test can re-checksum a patched IHDR chunk.
    fn crc32(bytes: &[u8]) -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        for &b in bytes {
            crc ^= u32::from(b);
            for _ in 0..8 {
                let mask = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
            }
        }
        !crc
    }

    /// Patch the IHDR width/height of a valid PNG to enormous values and fix the
    /// IHDR CRC, yielding a small file that *claims* a huge canvas — a decode
    /// bomb. PNG layout: 8-byte signature, then the IHDR chunk as
    /// length(4) + "IHDR"(4) + data(13) + crc(4); the CRC covers type+data.
    fn png_bomb(claim_w: u32, claim_h: u32) -> Vec<u8> {
        // IHDR data starts after sig(8) + length(4) + type(4) = offset 16.
        const IHDR_DATA: usize = 16;
        let mut png = solid_png(1, 1, [0, 0, 0]);
        png[IHDR_DATA..IHDR_DATA + 4].copy_from_slice(&claim_w.to_be_bytes());
        png[IHDR_DATA + 4..IHDR_DATA + 8].copy_from_slice(&claim_h.to_be_bytes());
        // Re-CRC over "IHDR" + the 13 data bytes (offsets 12..29).
        let crc = crc32(&png[12..29]);
        png[29..33].copy_from_slice(&crc.to_be_bytes());
        png
    }

    #[test]
    fn guard_dimensions_rejects_huge_accepts_small() {
        assert!(
            guard_dimensions("png", 100_000, 100_000).is_err(),
            "100000x100000 must be rejected"
        );
        assert!(
            guard_dimensions("png", 1024, 768).is_ok(),
            "ordinary page size must pass"
        );
    }

    #[test]
    fn png_decode_bomb_rejected_before_allocation() {
        // A ~70-byte file claiming 100000x100000 (≈30 GB at 3 B/px). The guard
        // must fire on the IHDR dims and return Err before output_buffer_size()
        // is ever allocated, so this completes instantly without an OOM.
        let bomb = png_bomb(100_000, 100_000);
        assert!(
            bomb.len() < 1024,
            "bomb file stays tiny: {} bytes",
            bomb.len()
        );
        match png::decode(&bomb) {
            Err(DecodeError::Codec(m)) => {
                assert!(m.contains("too large"), "message names the cause: {m}");
                assert!(m.contains("100000x100000"), "message names the dims: {m}");
            }
            Err(other) => panic!("expected Codec(too large), got {other:?}"),
            Ok(_) => panic!("decode bomb was not rejected"),
        }
    }
}
