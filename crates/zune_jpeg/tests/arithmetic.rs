//! Arithmetic-coded streams against Huffman-coded streams of the same
//! quantised coefficients (lossless transcodes of one file, see
//! `scan_layouts.rs`).
#![cfg(feature = "arith")]

use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;
use zune_jpeg::JpegDecoder;

/// Decodes `data` strictly to its natural colour space: gray, RGB or the
/// four components unchanged.
fn decode(data: &[u8], name: &str) -> (usize, usize, Vec<u8>) {
    let mut headers = JpegDecoder::new(ZCursor::new(data));
    headers
        .decode_headers()
        .unwrap_or_else(|e| panic!("{name}: headers: {e:?}"));
    let input = headers.input_colorspace().unwrap();
    let out = match input.num_components() {
        1 => ColorSpace::Luma,
        3 => ColorSpace::RGB,
        _ => input,
    };
    let options = DecoderOptions::default()
        .jpeg_set_out_colorspace(out)
        .set_strict_mode(true);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(data), options);
    let pixels = decoder
        .decode()
        .unwrap_or_else(|e| panic!("{name}: decode: {e:?}"));
    let (w, h) = decoder.dimensions().unwrap();
    assert_eq!(pixels.len(), w * h * out.num_components(), "{name}");
    (w, h, pixels)
}

macro_rules! image {
    ($name:literal) => {
        (
            $name,
            include_bytes!(concat!("images/scans/", $name)).as_slice(),
        )
    };
}

type Group = &'static [(&'static str, &'static [u8])];

/// The largest difference between two decodes of the same coefficients.
fn max_difference(a: (&str, &[u8]), b: (&str, &[u8])) -> u8 {
    let (a, b) = (decode(a.1, a.0), decode(b.1, b.0));
    assert_eq!((a.0, a.1), (b.0, b.1));
    a.2.iter()
        .zip(&b.2)
        .map(|(x, y)| x.abs_diff(*y))
        .max()
        .unwrap()
}

/// Arithmetic-coded streams of the same coefficients, sequential (with
/// and without restart intervals, in one scan and in a scan per
/// component) and progressive, decode to the image of the Huffman-coded
/// stream.  The entropy decoders report how many coefficients a block
/// has differently, and with it the inverse transform chosen, so a
/// sample may round one level apart.
#[test]
fn arithmetic_coding_matches_huffman_coding() {
    const GROUPS: &[Group] = &[
        &[
            image!("mix_444.jpg"),
            image!("mix_444_arith.jpg"),
            image!("mix_444_arith_rst.jpg"),
            image!("mix_444_arith_ni.jpg"),
            image!("mix_444_arith_prog.jpg"),
        ],
        &[
            image!("mix_420.jpg"),
            image!("mix_420_arith.jpg"),
            image!("mix_420_arith_rst.jpg"),
            image!("mix_420_arith_ni.jpg"),
            image!("mix_420_arith_prog.jpg"),
        ],
        &[
            image!("mix_422.jpg"),
            image!("mix_422_arith.jpg"),
            image!("mix_422_arith_rst.jpg"),
            image!("mix_422_arith_ni.jpg"),
            image!("mix_422_arith_prog.jpg"),
        ],
        &[
            image!("mix_440.jpg"),
            image!("mix_440_arith.jpg"),
            image!("mix_440_arith_rst.jpg"),
            image!("mix_440_arith_ni.jpg"),
            image!("mix_440_arith_prog.jpg"),
        ],
        &[
            image!("gray.jpg"),
            image!("gray_arith.jpg"),
            image!("gray_arith_prog.jpg"),
        ],
        &[
            image!("cmyk.jpg"),
            image!("cmyk_arith.jpg"),
            image!("cmyk_arith_prog_rst.jpg"),
        ],
    ];
    for group in GROUPS {
        let (huffman, arithmetic) = group.split_first().unwrap();
        for &coded in arithmetic {
            let d = max_difference(*huffman, coded);
            assert!(d <= 1, "{}: differs from {} by {d}", coded.0, huffman.0);
        }
    }
}

/// A damaged arithmetic-coded stream is an error or an image, never a
/// panic: every byte of each stream is replaced in turn by three other
/// values, and each stream is cut at every length.
#[test]
fn damaged_arithmetic_streams_decode_or_fail() {
    let images = [
        image!("mix_420_arith.jpg"),
        image!("mix_444_arith_rst.jpg"),
        image!("mix_422_arith_ni.jpg"),
        image!("mix_440_arith_prog.jpg"),
        image!("gray_arith_prog.jpg"),
        image!("cmyk_arith_prog_rst.jpg"),
    ];
    let try_decode = |data: &[u8]| {
        for strict in [false, true] {
            let options = DecoderOptions::default().set_strict_mode(strict);
            let mut decoder = JpegDecoder::new_with_options(ZCursor::new(data), options);
            if let Ok(pixels) = decoder.decode() {
                let (w, h) = decoder.dimensions().unwrap();
                assert!(pixels.len() >= w * h);
                assert!(decoder.intact_rows() <= h);
            }
        }
    };
    for (_, data) in images {
        let mut damaged = data.to_vec();
        for at in 0..data.len() {
            for value in [0x00, 0xFF, data[at] ^ 0x55] {
                damaged[at] = value;
                try_decode(&damaged);
            }
            damaged[at] = data[at];
            try_decode(&data[..at]);
        }
    }
}
