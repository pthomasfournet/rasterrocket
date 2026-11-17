#![no_main]

//! The JPEG decoder (the zune-jpeg fork): markers, Huffman and arithmetic
//! entropy decoding, baseline, progressive and multi-scan sequential
//! frames, restart intervals, upsampling, colour conversion and the count
//! of intact rows of a damaged stream, with the options of the `DCTDecode`
//! filter.
//!
//! Layout: `[mode]`, then the JPEG stream.  The low two bits of `mode`
//! select the output colour space (luma, RGB, CMYK, YCCK); bit 2 selects
//! strict mode.

use libfuzzer_sys::fuzz_target;
use zune_jpeg::JpegDecoder;
use zune_jpeg::zune_core::bytestream::ZCursor;
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;

const OUT: [ColorSpace; 4] = [
    ColorSpace::Luma,
    ColorSpace::RGB,
    ColorSpace::CMYK,
    ColorSpace::YCCK,
];

/// Largest decoded image, and largest coefficient store, in bytes.
const MAX_BYTES: usize = 64 << 20;

fuzz_target!(|data: &[u8]| {
    let Some(([mode], stream)) = rasterrocket_fuzz::split_selectors::<1>(data) else {
        return;
    };
    let options = DecoderOptions::default()
        .jpeg_set_out_colorspace(OUT[usize::from(mode & 3)])
        .set_max_width(usize::from(u16::MAX))
        .set_max_height(usize::from(u16::MAX))
        .set_strict_mode(mode & 4 != 0);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(stream), options);
    decoder.set_max_coefficient_bytes(MAX_BYTES);
    if decoder.decode_headers().is_err()
        || decoder
            .output_buffer_size()
            .is_none_or(|size| size > MAX_BYTES)
    {
        return;
    }
    if let Ok(pixels) = decoder.decode() {
        std::hint::black_box((pixels.len(), decoder.intact_rows()));
    }
});
