//! Loose-image decode front door.
//!
//! One place that sniffs an image's format by magic bytes, dispatches to the
//! per-codec wrapper, and normalises the result to grayscale — the single
//! decode→bitmap→gray sequence shared by every format.  Every codec checks
//! the header dimensions against the rendered-page limits before it
//! allocates pixels ([`guard_dimensions`]); the caller names the archive entry
//! in the resulting [`DecodeError::TooLarge`].

mod jpeg;
mod png;
mod tiff;
mod webp;

use color::{Gray8, Pixel, Rgb8};
use raster::Bitmap;

/// A codec's output in the colour model the stream carries.
enum CodecPixels {
    /// Single-channel luma.
    Gray(Bitmap<Gray8>),
    /// Interleaved RGB.
    Rgb(Bitmap<Rgb8>),
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
    /// The header dimensions exceed the rendered-page limits.
    TooLarge { width: u32, height: u32 },
    /// The codec rejected the byte stream.
    Codec(String),
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported => write!(f, "unrecognised image format"),
            Self::TooLarge { width, height } => {
                write!(f, "image too large ({width}x{height})")
            }
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

/// Decode arbitrary image bytes to a packed grayscale bitmap, sniffing the
/// format first. Gray streams are returned as decoded; RGB streams are
/// converted in their own allocation by [`pdf_raster::rgb_into_gray`].
///
/// # Errors
///
/// [`DecodeError::Unsupported`] if no codec matches the magic bytes;
/// [`DecodeError::Codec`] if the matched codec rejects the stream.
pub(crate) fn decode_image(bytes: &[u8]) -> Result<Bitmap<Gray8>, DecodeError> {
    let pixels = match sniff(bytes).ok_or(DecodeError::Unsupported)? {
        ImageFormat::Jpeg => jpeg::decode(bytes)?,
        ImageFormat::Png => png::decode(bytes)?,
        ImageFormat::WebP => webp::decode(bytes)?,
        ImageFormat::Tiff => tiff::decode(bytes)?,
    };
    Ok(match pixels {
        CodecPixels::Gray(gray) => gray,
        CodecPixels::Rgb(rgb) => pdf_raster::rgb_into_gray(rgb),
    })
}

/// Reject an image whose header dimensions exceed the rendered-page size
/// limits (the per-side and total-area caps of the PDF render path) before
/// the codec allocates its pixel buffer.
fn guard_dimensions(w: u32, h: u32) -> Result<(), DecodeError> {
    pdf_raster::validate_dimensions(w, h).map_err(|_| DecodeError::TooLarge {
        width: w,
        height: h,
    })
}

/// Unpack gray samples of `bits` (1, 2 or 4) bits, most significant first
/// with each of the `h` rows padded to a byte, into 8-bit luma: sample `v`
/// becomes `v · 255 / (2^bits − 1)`.
fn unpack_gray_bits(packed: &[u8], w: u32, h: u32, bits: u8) -> Result<Vec<u8>, DecodeError> {
    debug_assert!(
        matches!(bits, 1 | 2 | 4),
        "{bits} bits do not pack a byte evenly"
    );
    let per_byte = 8 / usize::from(bits);
    let row_bytes = (w as usize).div_ceil(per_byte);
    let rows = packed.len() / row_bytes.max(1);
    if rows < h as usize {
        return Err(DecodeError::Codec(format!(
            "{} bytes hold {rows} of {h} rows of {w} {bits}-bit samples",
            packed.len()
        )));
    }
    let max = (1u16 << bits) - 1;
    let mask = u8::try_from(max).unwrap_or(u8::MAX);
    let mut out = Vec::with_capacity(w as usize * h as usize);
    for row in packed.chunks_exact(row_bytes).take(h as usize) {
        let samples = row.iter().flat_map(|&byte| {
            (0..per_byte).map(move |i| (byte >> (8 - usize::from(bits) * (i + 1))) & mask)
        });
        out.extend(
            samples
                .take(w as usize)
                .map(|v| u8::try_from(u16::from(v) * 255 / max).unwrap_or(u8::MAX)),
        );
    }
    Ok(out)
}

/// Wrap a tight interleaved RGB buffer as an `Rgb8` bitmap without copying.
fn rgb_from_tight(w: u32, h: u32, rgb: Vec<u8>) -> Result<CodecPixels, DecodeError> {
    packed_from_tight(w, h, rgb).map(CodecPixels::Rgb)
}

/// Wrap a tight 8-bit luma buffer as a `Gray8` bitmap without copying.
fn gray_from_tight(w: u32, h: u32, gray: Vec<u8>) -> Result<CodecPixels, DecodeError> {
    packed_from_tight(w, h, gray).map(CodecPixels::Gray)
}

/// Take over a tight `w·h·P::BYTES` buffer as a packed bitmap, rejecting a
/// buffer whose length disagrees with the dimensions.
fn packed_from_tight<P: Pixel>(w: u32, h: u32, data: Vec<u8>) -> Result<Bitmap<P>, DecodeError> {
    let expected = (w as usize)
        .checked_mul(h as usize)
        .and_then(|n| n.checked_mul(P::BYTES));
    if expected != Some(data.len()) {
        return Err(DecodeError::Codec(format!(
            "pixel buffer {} bytes inconsistent with {w}x{h}x{}",
            data.len(),
            P::BYTES
        )));
    }
    Ok(Bitmap::from_packed(w, h, data))
}

/// Keep the first `keep` of every `channels`-byte pixel in place, dropping
/// the rest (an alpha channel, or a gray image's extra samples).  Byte `c` of
/// pixel `i` moves from `i·channels + c` to `i·keep + c`, never past its
/// source, so a forward pass reads each byte before it is overwritten.
fn keep_leading_samples(buf: &mut Vec<u8>, channels: usize, keep: usize) {
    debug_assert!(
        0 < keep && keep < channels,
        "keep {keep} of {channels} samples"
    );
    let n = buf.len() / channels;
    for i in 0..n {
        for c in 0..keep {
            buf[i * keep + c] = buf[i * channels + c];
        }
    }
    buf.truncate(n * keep);
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

    /// An uncompressed little-endian TIFF of one strip per plane:
    /// `bits` per sample, `samples` per pixel (photometric `BlackIsZero` for
    /// one or two, RGB for three or four), chunky or `planar`, over `data`.
    fn raw_tiff(w: u32, h: u32, bits: u16, samples: u16, planar: bool, data: &[u8]) -> Vec<u8> {
        let planes = if planar { u32::from(samples) } else { 1 };
        let plane_len = u32::try_from(data.len()).unwrap() / planes;
        let photometric = if samples >= 3 { 2 } else { 1 };
        // Header, then the pixel data, then the arrays, then the directory.
        let mut out = b"II*\0\0\0\0\0".to_vec();
        let data_at = u32::try_from(out.len()).unwrap();
        out.extend_from_slice(data);
        let array = |out: &mut Vec<u8>, values: &[u32], short: bool| {
            let at = u32::try_from(out.len()).unwrap();
            for &v in values {
                if short {
                    out.extend_from_slice(&u16::try_from(v).unwrap().to_le_bytes());
                } else {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
            at
        };
        let bits_at = array(&mut out, &vec![u32::from(bits); usize::from(samples)], true);
        let offsets: Vec<u32> = (0..planes).map(|p| data_at + p * plane_len).collect();
        let offsets_at = array(&mut out, &offsets, false);
        let counts_at = array(&mut out, &vec![plane_len; planes as usize], false);
        if out.len() % 2 == 1 {
            out.push(0);
        }
        let ifd = u32::try_from(out.len()).unwrap();
        out[4..8].copy_from_slice(&ifd.to_le_bytes());
        // (tag, type, count, value-or-offset); type 3 SHORT, 4 LONG.
        let inline = |v: u32, n: u32, at: u32| if n == 1 { v } else { at };
        let mut entries: Vec<(u16, u16, u32, u32)> = vec![
            (256, 4, 1, w),
            (257, 4, 1, h),
            (
                258,
                3,
                u32::from(samples),
                if samples <= 2 {
                    u32::from(bits) | (u32::from(bits) << 16) * u32::from(samples - 1)
                } else {
                    bits_at
                },
            ),
            (259, 3, 1, 1),
            (262, 3, 1, photometric),
            (273, 4, planes, inline(data_at, planes, offsets_at)),
            (277, 3, 1, u32::from(samples)),
            (278, 4, 1, h),
            (279, 4, planes, inline(plane_len, planes, counts_at)),
            (284, 3, 1, if planar { 2 } else { 1 }),
        ];
        if samples % 2 == 0 {
            // ExtraSamples: the last sample is unassociated alpha.
            entries.push((338, 3, 1, 2));
        }
        out.extend_from_slice(&u16::try_from(entries.len()).unwrap().to_le_bytes());
        for (tag, kind, count, value) in entries {
            out.extend_from_slice(&tag.to_le_bytes());
            out.extend_from_slice(&kind.to_le_bytes());
            out.extend_from_slice(&count.to_le_bytes());
            out.extend_from_slice(&value.to_le_bytes());
        }
        out.extend_from_slice(&0u32.to_le_bytes());
        out
    }

    #[test]
    fn low_bit_gray_tiffs_unpack_to_luma() {
        // 1 bit, 10 px wide: rows of two bytes, the second padded.
        let tif = raw_tiff(10, 2, 1, 1, false, &[0b1010_0000, 0b1100_0000, 0xFF, 0x00]);
        let img = decode_image(&tif).expect("1-bit tiff decodes");
        assert_eq!(
            img.data(),
            [
                255, 0, 255, 0, 0, 0, 0, 0, 255, 255, //
                255, 255, 255, 255, 255, 255, 255, 255, 0, 0
            ]
        );
        // 4 bits: samples 0, 5 and 15 scale to 0, 85 and 255.
        let tif = raw_tiff(3, 1, 4, 1, false, &[0x05, 0xF0]);
        assert_eq!(decode_image(&tif).unwrap().data(), [0, 85, 255]);
    }

    #[test]
    fn gray_alpha_and_planar_rgb_tiffs_decode() {
        let tif = raw_tiff(2, 1, 8, 2, false, &[10, 255, 200, 0]);
        let Ok(CodecPixels::Gray(bmp)) = tiff::decode(&tif) else {
            panic!("a gray+alpha TIFF must decode to Gray8");
        };
        assert_eq!(bmp.data(), [10, 200]);
        // Planes R, G, B of two pixels.
        let tif = raw_tiff(2, 1, 8, 3, true, &[1, 2, 10, 20, 100, 200]);
        let Ok(CodecPixels::Rgb(bmp)) = tiff::decode(&tif) else {
            panic!("a planar RGB TIFF must decode to Rgb8");
        };
        assert_eq!(bmp.data(), [1, 10, 100, 2, 20, 200]);
    }

    #[test]
    fn a_gray_jpeg_decodes_to_gray() {
        let jpg = solid_jpeg_gray(8, 8, 90);
        assert!(matches!(jpeg::decode(&jpg), Ok(CodecPixels::Gray(_))));
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
        assert_eq!(img.data().len(), 4);
        assert!(
            img.data().iter().all(|&p| p == img.data()[0]),
            "solid stays solid"
        );
        assert!(img.data()[0] > 30 && img.data()[0] < 200);
    }

    #[test]
    fn decode_jpeg_to_gray() {
        let jpg = solid_jpeg_gray(4, 4, 137);
        let img = decode_image(&jpg).expect("jpeg decodes");
        assert_eq!((img.width, img.height), (4, 4));
        assert!(img.data().iter().all(|&p| (i16::from(p) - 137).abs() <= 4));
    }

    #[test]
    fn decode_tiff_gray_to_expected_pixels() {
        // Exercises the TIFF wrapper's DecodingResult::U8 + ColorType::Gray
        // path end-to-end (the most logic-heavy codec match).
        let tif = solid_tiff_gray(3, 2, 90);
        let img = decode_image(&tif).expect("tiff decodes");
        assert_eq!((img.width, img.height), (3, 2));
        assert_eq!(img.data().len(), 6);
        assert!(
            img.data().iter().all(|&p| p == 90),
            "solid gray 90 must round-trip exactly (lossless), got {:?}",
            img.data()
        );
    }

    /// A PNG of colour type `color` over the raw interleaved `samples`.
    fn raw_png(w: u32, h: u32, color: ::png::ColorType, samples: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut enc = ::png::Encoder::new(&mut out, w, h);
        enc.set_color(color);
        enc.set_depth(::png::BitDepth::Eight);
        let mut writer = enc.write_header().unwrap();
        writer.write_image_data(samples).unwrap();
        writer.finish().unwrap();
        out
    }

    #[test]
    fn png_gray_decodes_straight_to_gray() {
        let luma = [0u8, 7, 128, 255, 33, 90];
        let png = raw_png(3, 2, ::png::ColorType::Grayscale, &luma);
        let Ok(CodecPixels::Gray(bmp)) = png::decode(&png) else {
            panic!("a grayscale PNG must decode to a Gray8 bitmap");
        };
        assert_eq!((bmp.width, bmp.height), (3, 2));
        assert_eq!(bmp.data(), luma);
        assert_eq!(decode_image(&png).unwrap().data(), luma);
    }

    #[test]
    fn png_gray_alpha_keeps_luma_drops_alpha() {
        let samples = [10u8, 255, 20, 0, 30, 128, 40, 1];
        let png = raw_png(2, 2, ::png::ColorType::GrayscaleAlpha, &samples);
        let Ok(CodecPixels::Gray(bmp)) = png::decode(&png) else {
            panic!("a gray+alpha PNG must decode to a Gray8 bitmap");
        };
        assert_eq!(bmp.data(), [10, 20, 30, 40]);
    }

    #[test]
    fn leading_samples_compact_in_place() {
        let mut rgba = vec![1, 2, 3, 9, 4, 5, 6, 9, 7, 8, 9, 9];
        keep_leading_samples(&mut rgba, 4, 3);
        assert_eq!(rgba, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let mut ga = vec![1, 9, 2, 9, 3, 9];
        keep_leading_samples(&mut ga, 2, 1);
        assert_eq!(ga, [1, 2, 3]);
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

    /// Patch the SOF0 frame size of a valid baseline JPEG, yielding a small
    /// file whose header claims `claim_w`×`claim_h`.  SOF0 is `FF C0`, then
    /// length(2), precision(1), height(2), width(2).
    fn jpeg_bomb(claim_w: u16, claim_h: u16) -> Vec<u8> {
        let mut jpg = solid_jpeg_gray(8, 8, 0);
        let sof = jpg
            .windows(2)
            .position(|m| m == [0xFF, 0xC0])
            .expect("baseline JPEG has an SOF0 marker");
        jpg[sof + 5..sof + 7].copy_from_slice(&claim_h.to_be_bytes());
        jpg[sof + 7..sof + 9].copy_from_slice(&claim_w.to_be_bytes());
        jpg
    }

    #[test]
    fn jpeg_decode_bomb_rejected_before_allocation() {
        // 30000×30000 = 9e8 px exceeds the area cap; at 3 B/px the decode would
        // allocate ~2.7 GB.  The guard fires on the SOF dims first.
        match jpeg::decode(&jpeg_bomb(30_000, 30_000)) {
            Err(DecodeError::TooLarge { width, height }) => {
                assert_eq!((width, height), (30_000, 30_000));
            }
            Err(other) => panic!("expected TooLarge, got {other:?}"),
            Ok(_) => panic!("decode bomb was not rejected"),
        }
    }

    #[test]
    fn guard_dimensions_rejects_huge_accepts_small() {
        assert!(
            guard_dimensions(100_000, 100_000).is_err(),
            "100000x100000 must be rejected"
        );
        assert!(
            guard_dimensions(1024, 768).is_ok(),
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
            Err(DecodeError::TooLarge { width, height }) => {
                assert_eq!((width, height), (100_000, 100_000));
            }
            Err(other) => panic!("expected TooLarge, got {other:?}"),
            Ok(_) => panic!("decode bomb was not rejected"),
        }
    }
}
