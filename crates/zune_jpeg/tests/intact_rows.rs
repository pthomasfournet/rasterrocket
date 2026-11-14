//! `JpegDecoder::intact_rows`: how much of an image was decoded from
//! entropy-coded data that was present.
//!
//! The images under `images/cut/` are noise, so a row decoded from
//! made-up bits never equals the row of the complete decode by chance.

use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;
use zune_jpeg::JpegDecoder;

struct Decoded {
    width: usize,
    height: usize,
    /// Bytes per row.
    stride: usize,
    pixels: Vec<u8>,
    intact_rows: usize,
}

impl Decoded {
    /// The number of leading rows equal to those of `other`.
    fn leading_rows_equal_to(&self, other: &Self) -> usize {
        self.pixels
            .chunks_exact(self.stride)
            .zip(other.pixels.chunks_exact(other.stride))
            .take_while(|(a, b)| a == b)
            .count()
    }
}

fn decode(data: &[u8], strict: bool) -> Result<Decoded, String> {
    let mut headers = JpegDecoder::new(ZCursor::new(data));
    headers.decode_headers().map_err(|e| format!("{e:?}"))?;
    let input = headers.input_colorspace().unwrap();
    let out = match input.num_components() {
        1 => ColorSpace::Luma,
        3 => ColorSpace::RGB,
        _ => input,
    };
    let options = DecoderOptions::default()
        .jpeg_set_out_colorspace(out)
        .set_strict_mode(strict);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(data), options);
    let pixels = decoder.decode().map_err(|e| format!("{e:?}"))?;
    let (width, height) = decoder.dimensions().unwrap();
    Ok(Decoded {
        width,
        height,
        stride: width * out.num_components(),
        pixels,
        intact_rows: decoder.intact_rows(),
    })
}

/// The offset of the first byte of entropy-coded data: past the first
/// SOS segment.
fn entropy_start(data: &[u8]) -> usize {
    let sos = data
        .windows(2)
        .position(|m| m == [0xFF, 0xDA])
        .expect("an SOS marker");
    sos + 2 + usize::from(u16::from_be_bytes([data[sos + 2], data[sos + 3]]))
}

struct Image {
    name: &'static str,
    data: &'static [u8],
    /// Image rows in one MCU row.
    mcu_rows: usize,
    /// Whether the frame is decoded row by row from a single scan, so
    /// that the rows before a cut are exactly the complete decode's.
    single_scan: bool,
    /// Whether the entropy coding tells a scan that stops at a marker
    /// from a complete one, and a complete stream without EOI from a cut
    /// one.  Arithmetic coding does not: its decoder reads zeros past a
    /// marker as a matter of course, and fails at the end of the input.
    marks_its_end: bool,
}

macro_rules! image {
    ($name:literal, $mcu_rows:literal, $single_scan:literal) => {
        image!($name, $mcu_rows, $single_scan, true)
    };
    ($name:literal, $mcu_rows:literal, $single_scan:literal, $marks_its_end:literal) => {
        Image {
            name: $name,
            data: include_bytes!(concat!("images/cut/", $name)),
            mcu_rows: $mcu_rows,
            single_scan: $single_scan,
            marks_its_end: $marks_its_end,
        }
    };
}

const IMAGES: &[Image] = &[
    image!("noise_444.jpg", 8, true),
    image!("noise_420.jpg", 16, true),
    image!("noise_420_rst.jpg", 16, true),
    image!("noise_gray.jpg", 8, true),
    image!("noise_420_ni.jpg", 16, false),
    image!("noise_444_prog.jpg", 8, false),
    image!("noise_420_prog.jpg", 16, false),
    #[cfg(feature = "arith")]
    image!("noise_420_arith.jpg", 16, true, false),
    #[cfg(feature = "arith")]
    image!("noise_420_arith_prog.jpg", 16, false, false),
];

#[test]
fn a_complete_stream_is_intact_with_or_without_eoi() {
    for image in IMAGES {
        for strict in [true, false] {
            let whole = decode(image.data, strict).expect(image.name);
            assert_eq!(whole.intact_rows, whole.height, "{}", image.name);
        }
        if !image.marks_its_end {
            continue;
        }
        let whole = decode(image.data, false).expect(image.name);
        let no_eoi = &image.data[..image.data.len() - 2];
        let got =
            decode(no_eoi, false).unwrap_or_else(|e| panic!("{} without EOI: {e}", image.name));
        assert_eq!(got.intact_rows, got.height, "{} without EOI", image.name);
        assert_eq!(got.pixels, whole.pixels, "{} without EOI", image.name);
    }
}

/// A DNL segment after the first scan (ITU T.81 B.2.5) is read and its
/// line count left unused: the image decodes whole, strict or not.
#[test]
fn a_dnl_segment_after_the_first_scan_is_skipped() {
    for image in IMAGES {
        let data = image.data;
        let after_first_scan = data[entropy_start(data)..]
            .windows(2)
            .position(|m| m == [0xFF, 0xDA])
            .map_or(data.len() - 2, |at| entropy_start(data) + at);
        let dnl = [0xFF, 0xDC, 0x00, 0x04, 0x00, 0x01];
        let with_dnl = [&data[..after_first_scan], &dnl, &data[after_first_scan..]].concat();
        for strict in [true, false] {
            let whole = decode(data, strict).expect(image.name);
            let got = decode(&with_dnl, strict)
                .unwrap_or_else(|e| panic!("{} with DNL, strict {strict}: {e}", image.name));
            assert_eq!(got.intact_rows, got.height, "{} with DNL", image.name);
            assert_eq!(got.pixels, whole.pixels, "{} with DNL", image.name);
        }
    }
}

/// Checks the decode of `image` cut after `cut` bytes, the cut being the
/// end of the input (`eoi` false) or followed by an EOI marker.
fn check_cut(image: &Image, whole: &Decoded, cut: usize, eoi: bool) -> Option<usize> {
    let mut data = image.data[..cut].to_vec();
    if eoi {
        data.extend_from_slice(&[0xFF, 0xD9]);
    }
    let what = format!("{} cut at {cut}, eoi {eoi}", image.name);
    let got = decode(&data, false).ok()?;
    assert_eq!(
        (got.width, got.height),
        (whole.width, whole.height),
        "{what}"
    );
    assert!(got.intact_rows <= got.height, "{what}");
    let equal = got.leading_rows_equal_to(whole);
    if got.intact_rows == got.height {
        assert_eq!(equal, got.height, "{what}: reported intact, differs");
    }
    if image.single_scan {
        // Rows reported intact are the complete decode's, but for the
        // last one or two when chroma is interpolated from the MCU row
        // below.
        let margin = if image.mcu_rows > 8 { 2 } else { 0 };
        assert!(
            equal + margin >= got.intact_rows,
            "{what}: {} rows reported intact, {equal} equal",
            got.intact_rows
        );
        // And every MCU row decoded whole is reported, but for one.
        let whole_mcu_rows = equal / image.mcu_rows * image.mcu_rows;
        assert!(
            got.intact_rows + image.mcu_rows >= whole_mcu_rows,
            "{what}: {} rows reported intact, {equal} equal",
            got.intact_rows
        );
    }
    Some(got.intact_rows)
}

#[test]
fn a_cut_stream_reports_the_rows_before_the_cut() {
    for image in IMAGES {
        let whole = decode(image.data, false).expect(image.name);
        let start = entropy_start(image.data);
        let end = image.data.len() - 2;
        for eoi in [false, true] {
            if eoi && !image.marks_its_end {
                continue;
            }
            let (mut decoded, mut short) = (0, 0);
            for cut in start..end {
                if let Some(rows) = check_cut(image, &whole, cut, eoi) {
                    decoded += 1;
                    short += usize::from(rows < whole.height);
                }
            }
            let what = format!("{}, eoi {eoi}", image.name);
            // Every cut is decoded, and none of them completely.
            assert_eq!(decoded, end - start, "{what}: cuts that failed");
            assert_eq!(short, decoded, "{what}: cuts reported complete");
        }
    }
}

/// A scan whose last byte is a data byte of 0xFF, cut before the stuffed zero
/// that follows it: the input does not say whether that 0xFF is data or the
/// start of a marker, so a row decoded from its bits is not reported whole.
#[test]
fn a_cut_after_a_data_byte_of_ff_is_not_whole() {
    // One 8×8 gray block.  DC table: one 1-bit code, a difference of zero.
    // AC table: one 1-bit code, a coefficient of eight bits after eight
    // zeros.  The block is the DC code and seven such coefficients of +255,
    // the last at position 63: 64 bits, of which the last eight are ones.
    let mut head = vec![0xFF, 0xD8, 0xFF, 0xDB, 0, 67, 0];
    head.extend_from_slice(&[1; 64]);
    head.extend_from_slice(&[0xFF, 0xC0, 0, 11, 8, 0, 8, 0, 8, 1, 1, 0x11, 0]);
    head.extend_from_slice(&[0xFF, 0xC4, 0, 20, 0x00, 1]);
    head.extend_from_slice(&[0; 15]);
    head.push(0x00);
    head.extend_from_slice(&[0xFF, 0xC4, 0, 20, 0x10, 1]);
    head.extend_from_slice(&[0; 15]);
    head.push(0x88);
    head.extend_from_slice(&[0xFF, 0xDA, 0, 8, 1, 1, 0x00, 0, 63, 0]);

    let mut whole = head.clone();
    whole.extend_from_slice(&[0x3F, 0xDF, 0xEF, 0xF7, 0xFB, 0xFD, 0xFE, 0xFF, 0x00]);
    let without_eoi = whole.clone();
    whole.extend_from_slice(&[0xFF, 0xD9]);
    let mut cut = without_eoi.clone();
    assert_eq!(cut.pop(), Some(0x00));

    let reference = decode(&whole, false).expect("the whole stream");
    assert_eq!(reference.intact_rows, 8);
    let got = decode(&without_eoi, false).expect("the stream without EOI");
    assert_eq!(got.intact_rows, 8);
    assert_eq!(got.pixels, reference.pixels);
    let got = decode(&cut, false).expect("the cut stream");
    assert_eq!(got.intact_rows, 0);
}
