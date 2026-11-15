//! Sample-exact agreement with libjpeg's accurate integer decode.
//!
//! The inverse DCT uses the 13-bit constants and per-pass rounding of the
//! accurate integer transform, and YCbCr → RGB uses 16 fractional bits with
//! halves rounded up, so a frame without chroma subsampling decodes to the
//! same samples.  The references are the payloads of `djpeg -dct int -pnm`
//! (libjpeg-turbo 2.1.5) for `images/scans/gray.jpg` and `mix_444.jpg`;
//! every scan layout of the same coefficients must match them.

use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;
use zune_jpeg::JpegDecoder;

fn assert_matches(name: &str, data: &[u8], out: ColorSpace, reference: &[u8]) {
    let options = DecoderOptions::default()
        .jpeg_set_out_colorspace(out)
        .set_strict_mode(true);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(data), options);
    let pixels = decoder.decode().unwrap_or_else(|e| panic!("{name}: {e:?}"));
    assert_eq!(pixels.len(), reference.len(), "{name}");
    if let Some(i) = pixels.iter().zip(reference).position(|(a, b)| a != b) {
        panic!(
            "{name}: sample {i} is {}, libjpeg has {}",
            pixels[i], reference[i]
        );
    }
}

macro_rules! scans {
    ($($name:literal),*) => {
        [$(($name, include_bytes!(concat!("images/scans/", $name)).as_slice())),*]
    };
}

#[test]
fn gray_matches_libjpeg_islow() {
    let reference = include_bytes!("images/scans/gray.islow.gray");
    for (name, data) in scans!(
        "gray.jpg",
        "gray_prog.jpg",
        "gray_arith.jpg",
        "gray_arith_prog.jpg"
    ) {
        assert_matches(name, data, ColorSpace::Luma, reference);
    }
}

#[test]
fn ycbcr_444_matches_libjpeg_islow() {
    let reference = include_bytes!("images/scans/mix_444.islow.rgb");
    for (name, data) in scans!(
        "mix_444.jpg",
        "mix_444_ni.jpg",
        "mix_444_ni_rst.jpg",
        "mix_444_prog.jpg",
        "mix_444_arith.jpg",
        "mix_444_arith_prog.jpg"
    ) {
        assert_matches(name, data, ColorSpace::RGB, reference);
    }
}
