//! Sequential Huffman-coded streams that leave out their Huffman tables
//! (abbreviated streams, Motion JPEG frames): the tables of section K.3 of
//! the JPEG specification apply.

use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;
use zune_jpeg::JpegDecoder;

/// An 8×8 gray baseline frame with a quantisation table and no DHT segment;
/// its one block is a DC difference of zero and an end of block in the
/// standard luminance codes.
fn abbreviated_grayscale() -> Vec<u8> {
    let mut data = vec![0xFF, 0xD8, 0xFF, 0xDB, 0, 67, 0];
    data.extend_from_slice(&[8; 64]);
    data.extend_from_slice(&[
        0xFF, 0xC0, 0, 11, 8, 0, 8, 0, 8, 1, 1, 0x11, 0, 0xFF, 0xDA, 0, 8, 1, 1, 0, 0, 63, 0, 0x2B,
        0xFF, 0xD9,
    ]);
    data
}

#[test]
fn baseline_without_dht_uses_standard_tables() {
    let data = abbreviated_grayscale();
    let options = DecoderOptions::default()
        .set_strict_mode(true)
        .jpeg_set_out_colorspace(ColorSpace::RGB);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(&data), options);
    assert_eq!(decoder.decode().unwrap(), [128; 8 * 8 * 3]);
    assert_eq!(decoder.intact_rows(), 8);
}

#[test]
fn tables_other_than_the_standard_two_are_not_made_up() {
    let mut data = abbreviated_grayscale();
    // The scan header's table selectors: DC table 2, AC table 2.
    let selector = data.len() - 7;
    assert_eq!(data[selector], 0x00);
    data[selector] = 0x22;
    let mut decoder = JpegDecoder::new(ZCursor::new(&data));
    assert!(decoder.decode().is_err());
}

#[test]
fn tables_in_the_stream_are_kept() {
    // The same frame with its own tables: one DC code (difference 0) and one
    // AC code (end of block), each a single 0 bit, so the block is two 0 bits.
    let mut data = abbreviated_grayscale();
    let sos = data.len() - 13;
    assert_eq!(data[sos..sos + 2], [0xFF, 0xDA]);
    let mut dht = vec![0xFF, 0xC4, 0, 20, 0x00, 1];
    dht.extend_from_slice(&[0; 15]);
    dht.push(0);
    dht.extend_from_slice(&[0xFF, 0xC4, 0, 20, 0x10, 1]);
    dht.extend_from_slice(&[0; 15]);
    dht.push(0);
    data.splice(sos..sos, dht);
    let entropy = data.len() - 3;
    assert_eq!(data[entropy], 0x2B);
    // With the standard tables these bits are not a block that ends here.
    data[entropy] = 0x3F;

    let options = DecoderOptions::default()
        .set_strict_mode(true)
        .jpeg_set_out_colorspace(ColorSpace::Luma);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(&data), options);
    assert_eq!(decoder.decode().unwrap(), [128; 8 * 8]);
    assert_eq!(decoder.intact_rows(), 8);
}
