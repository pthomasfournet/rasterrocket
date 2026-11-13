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
}

macro_rules! image {
    ($name:literal, $mcu_rows:literal, $single_scan:literal) => {
        Image {
            name: $name,
            data: include_bytes!(concat!("images/cut/", $name)),
            mcu_rows: $mcu_rows,
            single_scan: $single_scan,
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
];

#[test]
fn a_complete_stream_is_intact_with_or_without_eoi() {
    for image in IMAGES {
        for strict in [true, false] {
            let whole = decode(image.data, strict).expect(image.name);
            assert_eq!(whole.intact_rows, whole.height, "{}", image.name);
        }
        let whole = decode(image.data, false).expect(image.name);
        let no_eoi = &image.data[..image.data.len() - 2];
        let got =
            decode(no_eoi, false).unwrap_or_else(|e| panic!("{} without EOI: {e}", image.name));
        assert_eq!(got.intact_rows, got.height, "{} without EOI", image.name);
        assert_eq!(got.pixels, whole.pixels, "{} without EOI", image.name);
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
