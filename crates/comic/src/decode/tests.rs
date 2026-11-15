//! Tests of the decode front door and the codecs it dispatches to.

use super::*;
use color::{Gray8, Rgb8, Sample};
use raster::Bitmap;

/// The 8-bit page `page` of `bytes` decoded for `target`, of 8 bits.
fn gray8(bytes: &[u8], page: usize, target: Target) -> Result<Bitmap<Gray8>, DecodeError> {
    match decode_image(bytes, page, target)? {
        PageImage::Eight(bmp) => Ok(bmp),
        PageImage::Sixteen(_) => panic!("an 8-bit target yields an 8-bit page"),
    }
}

/// The 16-bit page 0 of `bytes`.
fn gray16(bytes: &[u8]) -> Bitmap<Gray16> {
    match decode_image(bytes, 0, SIXTEEN).expect("decodes") {
        PageImage::Sixteen(bmp) => bmp,
        PageImage::Eight(_) => panic!("a 16-bit target yields a 16-bit page"),
    }
}

fn solid_png(w: u32, h: u32, rgb: [u8; 3]) -> Vec<u8> {
    let mut bmp = Bitmap::<Rgb8>::new(w, h, 1, false);
    for y in 0..h {
        let row = bmp.row_bytes_mut(y);
        for x in 0..w as usize {
            row[x * 3..x * 3 + 3].copy_from_slice(&rgb);
        }
    }
    let mut out = Vec::new();
    encode::write_png(&bmp, &mut out, None).unwrap();
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

/// A TIFF of solid 8-bit gray images, one per `(value, subfile)`: each
/// `w × h` of `value`, its `NewSubfileType` set to `subfile`.
pub(crate) fn multi_tiff_gray(w: u32, h: u32, images: &[(u8, u32)]) -> Vec<u8> {
    use ::tiff::encoder::{TiffEncoder, colortype};
    use ::tiff::tags::Tag;
    let mut out = std::io::Cursor::new(Vec::new());
    let mut enc = TiffEncoder::new(&mut out).unwrap();
    for &(value, subfile) in images {
        let mut image = enc.new_image::<colortype::Gray8>(w, h).unwrap();
        image
            .encoder()
            .write_tag(Tag::NewSubfileType, subfile)
            .unwrap();
        image.write_data(&vec![value; (w * h) as usize]).unwrap();
    }
    out.into_inner()
}

#[test]
fn tiff_pages_skip_thumbnails_and_masks_and_decode_by_directory() {
    let tif = multi_tiff_gray(2, 2, &[(10, 0), (99, 1), (77, 4), (30, 2)]);
    let pages = image_pages(&tif).unwrap();
    assert_eq!(pages.len(), 2);
    assert_eq!(gray8(&tif, pages[0], EIGHT).unwrap().data()[0], 10);
    assert_eq!(gray8(&tif, pages[1], EIGHT).unwrap().data()[0], 30);
    // A file of nothing but thumbnails still shows its first image.
    assert_eq!(image_pages(&multi_tiff_gray(2, 2, &[(5, 1)])).unwrap(), [0]);
    assert_eq!(image_pages(&solid_jpeg_gray(2, 2, 9)).unwrap(), [0]);
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
                u32::from(bits) | ((u32::from(bits) << 16) * u32::from(samples - 1))
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
    if samples.is_multiple_of(2) {
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
    let img = gray8(&tif, 0, EIGHT).expect("1-bit tiff decodes");
    assert_eq!(
        img.data(),
        [
            255, 0, 255, 0, 0, 0, 0, 0, 255, 255, //
            255, 255, 255, 255, 255, 255, 255, 255, 0, 0
        ]
    );
    // 4 bits: samples 0, 5 and 15 scale to 0, 85 and 255.
    let tif = raw_tiff(3, 1, 4, 1, false, &[0x05, 0xF0]);
    assert_eq!(gray8(&tif, 0, EIGHT).unwrap().data(), [0, 85, 255]);
}

#[test]
fn gray_alpha_and_planar_rgb_tiffs_decode() {
    let tif = raw_tiff(2, 1, 8, 2, false, &[10, 255, 200, 0]);
    let Ok(CodecPixels::Gray(bmp)) = tiff::decode(&tif, 0, EIGHT) else {
        panic!("a gray+alpha TIFF must decode to Gray8");
    };
    assert_eq!(bmp.data(), [10, 200]);
    // Planes R, G, B of two pixels.
    let tif = raw_tiff(2, 1, 8, 3, true, &[1, 2, 10, 20, 100, 200]);
    let Ok(CodecPixels::Rgb(bmp)) = tiff::decode(&tif, 0, EIGHT) else {
        panic!("a planar RGB TIFF must decode to Rgb8");
    };
    assert_eq!(bmp.data(), [1, 10, 100, 2, 20, 200]);
}

#[test]
fn rgb_tiffs_with_a_non_alpha_extra_sample_decode() {
    // RGB plus one extra sample of unspecified meaning (ExtraSamples 0):
    // the tiff decoder drops samples past the colour type's own, in
    // chunky and planar images alike.
    let unspecified = |tif: Vec<u8>| {
        let alpha = [
            &338u16.to_le_bytes()[..],
            &3u16.to_le_bytes(),
            &1u32.to_le_bytes(),
            &2u32.to_le_bytes(),
        ]
        .concat();
        let at = tif
            .windows(alpha.len())
            .position(|w| w == alpha)
            .expect("ExtraSamples entry");
        let mut tif = tif;
        tif[at + 8] = 0;
        tif
    };
    let chunky = unspecified(raw_tiff(2, 1, 8, 4, false, &[1, 2, 3, 99, 4, 5, 6, 99]));
    let planar = unspecified(raw_tiff(2, 1, 8, 4, true, &[1, 4, 2, 5, 3, 6, 99, 99]));
    for tif in [chunky, planar] {
        let Ok(CodecPixels::Rgb(bmp)) = tiff::decode(&tif, 0, EIGHT) else {
            panic!("an RGB TIFF with an extra sample must decode to Rgb8");
        };
        assert_eq!(bmp.data(), [1, 2, 3, 4, 5, 6]);
    }
}

#[test]
fn a_gray_jpeg_decodes_to_gray() {
    let jpg = solid_jpeg_gray(8, 8, 90);
    assert!(matches!(
        jpeg::decode(&jpg, EIGHT),
        Ok(CodecPixels::Gray(_))
    ));
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
    let img = gray8(&png, 0, EIGHT).expect("png decodes");
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
    let img = gray8(&jpg, 0, EIGHT).expect("jpeg decodes");
    assert_eq!((img.width, img.height), (4, 4));
    assert!(img.data().iter().all(|&p| (i16::from(p) - 137).abs() <= 4));
}

#[test]
fn decode_tiff_gray_to_expected_pixels() {
    // Exercises the TIFF wrapper's DecodingResult::U8 + ColorType::Gray
    // path end-to-end (the most logic-heavy codec match).
    let tif = solid_tiff_gray(3, 2, 90);
    let img = gray8(&tif, 0, EIGHT).expect("tiff decodes");
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
    let Ok(CodecPixels::Gray(bmp)) = png::decode(&png, EIGHT) else {
        panic!("a grayscale PNG must decode to a Gray8 bitmap");
    };
    assert_eq!((bmp.width, bmp.height), (3, 2));
    assert_eq!(bmp.data(), luma);
    assert_eq!(gray8(&png, 0, EIGHT).unwrap().data(), luma);
}

#[test]
fn png_gray_alpha_keeps_luma_drops_alpha() {
    let samples = [10u8, 255, 20, 0, 30, 128, 40, 1];
    let png = raw_png(2, 2, ::png::ColorType::GrayscaleAlpha, &samples);
    let Ok(CodecPixels::Gray(bmp)) = png::decode(&png, EIGHT) else {
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
    assert!(gray8(&png, 0, EIGHT).is_err(), "must return Err, not panic");
}

#[test]
fn unknown_format_is_unsupported() {
    assert!(matches!(
        gray8(b"xxxx not an image xxxx", 0, EIGHT),
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
    match jpeg::decode(&jpeg_bomb(30_000, 30_000), EIGHT) {
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
        guard_dimensions(100_000, 100_000, EIGHT).is_err(),
        "100000x100000 must be rejected"
    );
    assert!(
        guard_dimensions(1024, 768, EIGHT).is_ok(),
        "ordinary page size must pass"
    );
    let at = |max_bytes, depth| Target::new(max_bytes, depth);
    assert!(guard_dimensions(1000, 1000, at(3_000_000, SampleDepth::Eight)).is_ok());
    assert!(
        guard_dimensions(1000, 1000, at(2_999_999, SampleDepth::Eight)).is_err(),
        "the budget is the caller's, at 3 bytes per pixel"
    );
    assert!(guard_dimensions(1000, 1000, at(6_000_000, SampleDepth::Sixteen)).is_ok());
    assert!(
        guard_dimensions(1000, 1000, at(5_999_999, SampleDepth::Sixteen)).is_err(),
        "6 bytes per pixel at 16 bits"
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
    match png::decode(&bomb, EIGHT) {
        Err(DecodeError::TooLarge { width, height }) => {
            assert_eq!((width, height), (100_000, 100_000));
        }
        Err(other) => panic!("expected TooLarge, got {other:?}"),
        Ok(_) => panic!("decode bomb was not rejected"),
    }
}

/// A PNG of 16-bit samples of colour type `color`.
pub(crate) fn png16(w: u32, h: u32, color: ::png::ColorType, samples: &[u16]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut enc = ::png::Encoder::new(&mut out, w, h);
    enc.set_color(color);
    enc.set_depth(::png::BitDepth::Sixteen);
    let mut writer = enc.write_header().unwrap();
    let bytes: Vec<u8> = samples.iter().flat_map(|v| v.to_be_bytes()).collect();
    writer.write_image_data(&bytes).unwrap();
    writer.finish().unwrap();
    out
}

/// A 16-bit gray or RGB TIFF of `samples`.
pub(crate) fn tiff16(w: u32, h: u32, rgb: bool, samples: &[u16]) -> Vec<u8> {
    use ::tiff::encoder::{TiffEncoder, colortype};
    let mut out = std::io::Cursor::new(Vec::new());
    let mut enc = TiffEncoder::new(&mut out).unwrap();
    if rgb {
        enc.write_image::<colortype::RGB16>(w, h, samples).unwrap();
    } else {
        enc.write_image::<colortype::Gray16>(w, h, samples).unwrap();
    }
    out.into_inner()
}

/// `n` samples of a fixed pseudo-random sequence.
pub(crate) fn noise16(n: usize) -> Vec<u16> {
    let mut seed = 0x2545_f491_u32;
    (0..n)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            u16::try_from(seed >> 16).unwrap()
        })
        .collect()
}

/// The images the depth-8 golden digests cover.
fn golden_inputs() -> Vec<Vec<u8>> {
    let mut inputs: Vec<Vec<u8>> = [
        &include_bytes!("../../tests/data/rgb-8x4.avif")[..],
        include_bytes!("../../tests/data/gray-16x8.avif"),
        include_bytes!("../../tests/data/lossy-64x48.avif"),
        include_bytes!("../../tests/data/deep-32x16.avif"),
        include_bytes!("../../tests/data/grid-120x100.avif"),
        include_bytes!("../../tests/data/turn-8x4.avif"),
        include_bytes!("../../tests/data/gray-16x8.jxl"),
        include_bytes!("../../tests/data/rgb-8x4.jxl"),
        include_bytes!("../../tests/data/lossy-32x16.jxl"),
    ]
    .map(<[u8]>::to_vec)
    .into();
    let n = noise16(5 * 3 * 4);
    inputs.push(png16(5, 3, ::png::ColorType::Rgb, &n[..45]));
    inputs.push(png16(5, 3, ::png::ColorType::Rgba, &n));
    inputs.push(png16(5, 3, ::png::ColorType::Grayscale, &n[..15]));
    inputs.push(tiff16(5, 3, true, &n[..45]));
    inputs.push(tiff16(5, 3, false, &n[..15]));
    inputs.push(solid_png(6, 2, [10, 200, 30]));
    inputs
}

/// The depth-8 pages of fixed inputs, digested before the codecs kept
/// 16 bits.
#[test]
fn eight_bit_pages_match_their_golden_digest() {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for bytes in golden_inputs() {
        let page = gray8(&bytes, 0, EIGHT).unwrap();
        let dims = [page.width.to_le_bytes(), page.height.to_le_bytes()];
        for b in dims.iter().flatten().chain(page.data()) {
            h = (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3);
        }
    }
    assert_eq!(h, GOLDEN, "{h:#x}");
}

const GOLDEN: u64 = 0x7197_9ee4_58ce_c86d;

/// The samples of `bmp`'s rows, packed.
fn samples<P: Pixel>(bmp: &Bitmap<P>) -> Vec<P::Sample> {
    let live = bmp.width as usize * P::NCOMPS;
    (0..bmp.height)
        .flat_map(|y| bmp.row_samples(y)[..live].to_vec())
        .collect()
}

#[test]
fn sixteen_bit_pngs_keep_their_samples() {
    let n = noise16(5 * 3 * 4);
    let gray = png16(5, 3, ::png::ColorType::Grayscale, &n[..15]);
    assert_eq!(samples(&gray16(&gray)), &n[..15]);
    let ga = png16(5, 3, ::png::ColorType::GrayscaleAlpha, &n[..30]);
    let luma: Vec<u16> = n[..30].iter().step_by(2).copied().collect();
    assert_eq!(samples(&gray16(&ga)), luma);
    let rgb = png16(5, 3, ::png::ColorType::Rgb, &n[..45]);
    let want = pdf_raster::rgb16_to_gray16(&Bitmap::<Rgb16>::from_packed(5, 3, n[..45].to_vec()));
    assert_eq!(samples(&gray16(&rgb)), samples(&want));
    let rgba = png16(5, 3, ::png::ColorType::Rgba, &n);
    let colour: Vec<u16> = n
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|px| [px[0], px[1], px[2]])
        .collect();
    let want = pdf_raster::rgb16_to_gray16(&Bitmap::<Rgb16>::from_packed(5, 3, colour));
    assert_eq!(samples(&gray16(&rgba)), samples(&want));
}

#[test]
fn sixteen_bit_tiffs_keep_their_samples() {
    let n = noise16(45);
    assert_eq!(samples(&gray16(&tiff16(5, 3, false, &n[..15]))), &n[..15]);
    let want = pdf_raster::rgb16_to_gray16(&Bitmap::<Rgb16>::from_packed(5, 3, n.clone()));
    assert_eq!(samples(&gray16(&tiff16(5, 3, true, &n))), samples(&want));
    // Planes R, G, B of two 16-bit pixels, little-endian.
    let planes: [u16; 6] = [1000, 2000, 30_000, 40_000, 50_000, 60_000];
    let bytes: Vec<u8> = planes.iter().flat_map(|v| v.to_le_bytes()).collect();
    let Ok(CodecPixels::Rgb16(bmp)) =
        tiff::decode(&raw_tiff(2, 1, 16, 3, true, &bytes), 0, SIXTEEN)
    else {
        panic!("a planar 16-bit RGB TIFF must decode to Rgb16");
    };
    assert_eq!(bmp.data(), [1000, 30_000, 50_000, 2000, 40_000, 60_000]);
    // Gray with an alpha sample, chunky.
    let ga: [u16; 4] = [7, 65_535, 65_000, 0];
    let bytes: Vec<u8> = ga.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(
        samples(&gray16(&raw_tiff(2, 1, 16, 2, false, &bytes))),
        [7, 65_000]
    );
}

#[test]
fn eight_bit_images_widen_exactly_at_16_bits() {
    let tiny = noise16(4)
        .iter()
        .map(|&v| v.to_be_bytes()[0])
        .collect::<Vec<u8>>();
    for bytes in [
        solid_png(4, 3, [10, 200, 30]),
        raw_png(2, 2, ::png::ColorType::Grayscale, &tiny),
        solid_jpeg_gray(8, 8, 90),
        solid_tiff_gray(3, 2, 77),
        raw_tiff(10, 2, 1, 1, false, &[0b1010_0000, 0b1100_0000, 0xFF, 0x00]),
        include_bytes!("../../tests/data/rgb-8x4.avif").to_vec(),
        include_bytes!("../../tests/data/rgb-8x4.jxl").to_vec(),
    ] {
        let eight = gray8(&bytes, 0, EIGHT).expect("decodes");
        let wide: Vec<u16> = samples(&eight).into_iter().map(u16::from_u8).collect();
        let sixteen = gray16(&bytes);
        assert_eq!((sixteen.width, sixteen.height), (eight.width, eight.height));
        assert_eq!(samples(&sixteen), wide);
    }
}
