//! Decoder tests over the fixtures `tests/data/avif-fixtures.py` writes.

use super::decode;
use crate::decode::{CodecPixels, DecodeError, ImageFormat, sniff};

/// 8 × 4 lossless 4:4:4 with the identity matrix, pixel `(30x, 60y, 255 − 30x)`.
const RGB: &[u8] = include_bytes!("../../../tests/data/rgb-8x4.avif");
/// 16 × 8 lossless monochrome, sample `(16x + y) mod 256`; data in `idat`.
const GRAY: &[u8] = include_bytes!("../../../tests/data/gray-16x8.avif");
/// 64 × 48 lossy 4:2:0, limited range, pixel `(4x, 5y, 255 − 2x − 2y)`; two
/// extents.
const LOSSY: &[u8] = include_bytes!("../../../tests/data/lossy-64x48.avif");
/// 32 × 16 lossy 10-bit 4:2:0, BT.709 full range by its `colr` box alone,
/// pixel `(8x, 16y, 128)`.
const DEEP: &[u8] = include_bytes!("../../../tests/data/deep-32x16.avif");
/// A 2 × 2 grid of 64 × 64 lossless tiles cropped to 120 × 100, pixel
/// `(2x, 2y, x + y)`.
const GRID: &[u8] = include_bytes!("../../../tests/data/grid-120x100.avif");
/// `RGB` with `clap` (the middle 6 × 2), `irot` 1 and `imir` 1.
const TURN: &[u8] = include_bytes!("../../../tests/data/turn-8x4.avif");

const ALL: [&[u8]; 6] = [RGB, GRAY, LOSSY, DEEP, GRID, TURN];

/// The decoded RGB pixels of `bytes` as (width, height, tight data).
fn rgb(bytes: &[u8]) -> (u32, u32, Vec<u8>) {
    match decode(bytes) {
        Ok(CodecPixels::Rgb(bmp)) => (bmp.width, bmp.height, bmp.data().to_vec()),
        Ok(CodecPixels::Gray(_)) => panic!("a colour image decodes to RGB"),
        Err(e) => panic!("decode failed: {e}"),
    }
}

/// `width × height` pixels of `pixel(x, y)`, tight.
fn pattern(width: u32, height: u32, pixel: impl Fn(u32, u32) -> [u32; 3]) -> Vec<u8> {
    (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .flat_map(|(x, y)| pixel(x, y))
        .map(|v| u8::try_from(v).unwrap())
        .collect()
}

/// The mean absolute difference of two equally long sample runs.
fn mean_difference(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let total: u32 = a
        .iter()
        .zip(b)
        .map(|(a, b)| u32::from(a.abs_diff(*b)))
        .sum();
    f64::from(total) / f64::from(u32::try_from(a.len()).unwrap())
}

/// `bytes` with the first occurrence of `find` replaced by `replace`.
fn patched(bytes: &[u8], find: &[u8], replace: &[u8]) -> Vec<u8> {
    assert_eq!(find.len(), replace.len());
    let at = bytes
        .windows(find.len())
        .position(|w| w == find)
        .expect("the fixture holds the bytes to patch");
    let mut out = bytes.to_vec();
    out[at..at + find.len()].copy_from_slice(replace);
    out
}

/// `bytes` with `data` written `skip` bytes after the first occurrence of
/// the box type `kind`.
fn with_box_bytes(bytes: &[u8], kind: &[u8], skip: usize, data: &[u8]) -> Vec<u8> {
    let at = bytes
        .windows(4)
        .position(|w| w == kind)
        .expect("the fixture holds the box")
        + 4
        + skip;
    let mut out = bytes.to_vec();
    out[at..at + data.len()].copy_from_slice(data);
    out
}

#[test]
fn sniffing_reads_the_brands() {
    for file in ALL {
        assert_eq!(sniff(file), Some(ImageFormat::Avif));
    }
    let brand = |major: &[u8; 4], compatible: &[u8; 4]| {
        let mut ftyp = b"\0\0\0\x14ftyp".to_vec();
        ftyp.extend_from_slice(major);
        ftyp.extend_from_slice(b"avif");
        ftyp.extend_from_slice(compatible);
        ftyp
    };
    // The minor version is not a brand, whatever its bytes.
    assert_eq!(sniff(&brand(b"heic", b"heix")), None);
    assert_eq!(sniff(&brand(b"heic", b"mif1")), Some(ImageFormat::Avif));
    assert_eq!(sniff(&brand(b"avis", b"msf1")), Some(ImageFormat::Avif));
    assert_eq!(sniff(b"\0\0\0\x10ftypavif"), Some(ImageFormat::Avif));
    assert_eq!(sniff(b"\0\0\0\x10moovavif\0\0\0\0"), None);
}

#[test]
fn a_lossless_identity_image_decodes_exactly() {
    let (w, h, data) = rgb(RGB);
    assert_eq!((w, h), (8, 4));
    assert_eq!(data, pattern(8, 4, |x, y| [x * 30, y * 60, 255 - x * 30]));
}

#[test]
fn a_monochrome_image_in_idat_decodes_exactly_to_gray() {
    let Ok(CodecPixels::Gray(bmp)) = decode(GRAY) else {
        panic!("a monochrome image decodes to gray")
    };
    assert_eq!((bmp.width, bmp.height), (16, 8));
    let want: Vec<u8> = (0..8u32)
        .flat_map(|y| (0..16u32).map(move |x| u8::try_from((x * 16 + y) % 256).unwrap()))
        .collect();
    assert_eq!(bmp.data(), want);
}

#[test]
fn a_lossy_subsampled_image_stays_near_its_source() {
    // Untagged, so BT.601 limited range; chroma interpolated.  Measured
    // mean differences from the source: 0.72 levels here, 0.81 for libavif
    // 1.4.2 (through Pillow 13.0); between the two decodes 0.25, at most 2.
    let (w, h, data) = rgb(LOSSY);
    assert_eq!((w, h), (64, 48));
    let source = pattern(64, 48, |x, y| [x * 4, y * 5, 255 - 2 * x - 2 * y]);
    let mean = mean_difference(&data, &source);
    assert!(mean <= 1.5, "mean difference {mean}");
}

#[test]
fn the_colour_box_overrides_the_sequence_header() {
    // The stream says "unspecified, limited range"; the colr box says
    // BT.709 full range, which is how the picture was encoded.  Measured
    // mean differences from the source: 0.88 levels here, 1.19 for libavif
    // 1.4.2; between the two decodes 0.58, at most 3.
    let source = pattern(32, 16, |x, y| [x * 8, y * 16, 128]);
    let (w, h, data) = rgb(DEEP);
    assert_eq!((w, h), (32, 16));
    let mean = mean_difference(&data, &source);
    assert!(mean <= 2.0, "mean difference {mean}");
    // Without the box the sequence header's limited range stretches the
    // samples.
    let (_, _, untagged) = rgb(&patched(DEEP, b"colr", b"zzzz"));
    let mean = mean_difference(&untagged, &source);
    assert!(mean > 8.0, "mean difference {mean}");
}

#[test]
fn a_grid_composes_its_tiles_and_crops_to_its_size() {
    let (w, h, data) = rgb(GRID);
    assert_eq!((w, h), (120, 100));
    assert_eq!(data, pattern(120, 100, |x, y| [2 * x, 2 * y, x + y]));
}

#[test]
fn clap_irot_and_imir_apply_in_order() {
    // P is the source picture, C its clean aperture: C(x, y) = P(x + 1, y + 1),
    // 6 × 2.  Each case names where output pixel (x, y) comes from in P.
    // The first case is the fixture as written, and is what ffmpeg 7
    // (crop, display matrix, flip) shows for it.
    type Case = (u8, u8, (u32, u32), fn(u32, u32) -> (u32, u32));
    let source = |x: u32, y: u32| [x * 30, y * 60, 255 - x * 30];
    let cases: [Case; 4] = [
        // A quarter turn anticlockwise, then left ↔ right: C(5 − y, 1 − x).
        (1, 1, (2, 6), |x, y| (6 - y, 2 - x)),
        // A quarter turn anticlockwise, then top ↔ bottom: C(y, x).
        (1, 0, (2, 6), |x, y| (y + 1, x + 1)),
        // A half turn, then top ↔ bottom: C(5 − x, y).
        (2, 0, (6, 2), |x, y| (6 - x, y + 1)),
        // A quarter turn clockwise, then left ↔ right: C(y, x) again, since
        // out(x, y) = C(y, 1 − x) mirrored is C(y, x).
        (3, 1, (2, 6), |x, y| (y + 1, x + 1)),
    ];
    for (turns, axis, size, from) in cases {
        let file = with_box_bytes(TURN, b"irot", 0, &[turns]);
        let file = with_box_bytes(&file, b"imir", 0, &[axis]);
        let (w, h, data) = rgb(&file);
        assert_eq!((w, h), size, "irot {turns} imir {axis}");
        let want = pattern(w, h, |x, y| {
            let (px, py) = from(x, y);
            source(px, py)
        });
        assert_eq!(data, want, "irot {turns} imir {axis}");
    }
}

#[test]
fn a_clean_aperture_off_the_pixel_grid_is_not_applied() {
    // Width 5 of 8 centred: the left edge would be at 1.5.
    let file = with_box_bytes(TURN, b"clap", 0, &5u32.to_be_bytes());
    let file = with_box_bytes(&file, b"irot", 0, &[0]);
    let (w, h, _) = rgb(&file);
    assert_eq!((w, h), (8, 4));
}

#[test]
fn unknown_properties_fail_only_when_essential() {
    // `colr` is not essential here; without it the sequence header's own
    // identity matrix applies.
    let (_, _, data) = rgb(&patched(RGB, b"colr", b"zzzz"));
    assert_eq!(data, pattern(8, 4, |x, y| [x * 30, y * 60, 255 - x * 30]));
    let err = decode(&patched(TURN, b"clap", b"zzzz")).err().unwrap();
    assert!(
        err.to_string().contains("essential property 'zzzz'"),
        "{err}"
    );
}

#[test]
fn files_without_a_still_image_are_codec_errors() {
    for (find, replace) in [(b"pitm", b"free"), (b"meta", b"free"), (b"pict", b"vide")] {
        assert!(
            matches!(
                decode(&patched(RGB, find, replace)),
                Err(DecodeError::Codec(_))
            ),
            "{}",
            find.escape_ascii()
        );
    }
    // An item type that is not a coded picture or a grid.
    assert!(decode(&patched(RGB, b"av01", b"hvc1")).is_err());
}

#[test]
fn every_truncation_is_an_error() {
    for file in ALL {
        for cut in 0..file.len() {
            assert!(
                decode(&file[..cut]).is_err(),
                "cut at {cut} of {}",
                file.len()
            );
        }
    }
}

#[test]
fn item_extents_outside_the_file_are_errors() {
    // iloc version 0 of the fixture: after the type, 4 bytes of version
    // and flags, 2 of field sizes, 2 of item count, then item id (2), data
    // reference (2), extent count (2), extent offset (4), extent length (4).
    for (skip, value) in [(14, 0xFFFF_FF00u32), (14, 400), (18, 4000)] {
        let file = with_box_bytes(RGB, b"iloc", skip, &value.to_be_bytes());
        assert!(
            matches!(decode(&file), Err(DecodeError::Codec(_))),
            "{skip} {value}"
        );
    }
}

#[test]
fn absurd_or_wrong_sizes_are_refused() {
    let ispe = |bytes: &[u8], w: u32, h: u32| {
        let file = with_box_bytes(bytes, b"ispe", 4, &w.to_be_bytes());
        with_box_bytes(&file, b"ispe", 8, &h.to_be_bytes())
    };
    assert!(matches!(
        decode(&ispe(RGB, 100_000, 100_000)),
        Err(DecodeError::TooLarge {
            width: 100_000,
            height: 100_000
        })
    ));
    assert!(matches!(
        decode(&ispe(RGB, u32::MAX, 1)),
        Err(DecodeError::TooLarge { .. })
    ));
    // A picture larger than declared stops at the decoder's frame size
    // limit; a smaller one at the comparison after it.
    for (w, h) in [(4, 2), (16, 8), (0, 4)] {
        assert!(
            matches!(decode(&ispe(RGB, w, h)), Err(DecodeError::Codec(_))),
            "{w}x{h}"
        );
    }
    // The grid item's data: version, flags, rows − 1, columns − 1, then
    // the 16-bit output size.  It is the first item of the mdat box.
    let grid = |bytes: &[u8]| with_box_bytes(GRID, b"mdat", 0, bytes);
    assert!(matches!(
        decode(&grid(&[0, 0, 1, 1, 0xFF, 0xFF, 0xFF, 0xFF])),
        Err(DecodeError::TooLarge { .. })
    ));
    assert!(matches!(
        decode(&grid(&[0, 0, 255, 255, 0, 120, 0, 100])),
        Err(DecodeError::Codec(_))
    ));
    assert!(decode(&grid(&[0, 0, 1, 1, 0, 0, 0, 100])).is_err());
    // 32-bit sizes need four more bytes than the item has.
    assert!(decode(&grid(&[0, 1, 1, 1, 0, 120, 0, 100])).is_err());
}

#[test]
fn flipped_bytes_never_panic() {
    for file in ALL {
        let mut damaged = file.to_vec();
        // Every byte of the small files, every third of the grid.
        let step = if file.len() > 1024 { 3 } else { 1 };
        for at in (0..file.len()).step_by(step) {
            damaged[at] = file[at] ^ [0x01, 0x80, 0xFF][at % 3];
            // Either outcome is acceptable; reaching the next iteration
            // is the assertion.
            drop(decode(&damaged));
            damaged[at] = file[at];
        }
    }
}

/// Decode timing, not a check: `AVIF_FILE=page.avif cargo test --release
/// -p rasterrocket-comic --lib -- --ignored avif_decode_time --nocapture`
/// prints the median of `AVIF_RUNS` (default 9) decodes.
#[test]
#[ignore = "a measurement that needs AVIF_FILE"]
fn avif_decode_time() {
    let path = std::env::var("AVIF_FILE").expect("AVIF_FILE names the file to decode");
    let runs = std::env::var("AVIF_RUNS").map_or(9, |n| n.parse().unwrap());
    let file = std::fs::read(&path).unwrap();
    let mut times: Vec<_> = (0..runs)
        .map(|_| {
            let start = std::time::Instant::now();
            let decoded = decode(&file);
            let elapsed = start.elapsed();
            assert!(decoded.is_ok());
            elapsed
        })
        .collect();
    times.sort();
    println!(
        "{path}: median {:?} of {runs}, min {:?}",
        times[times.len() / 2],
        times[0]
    );
}
