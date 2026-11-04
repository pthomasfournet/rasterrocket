//! Malformed streams, which must be rejected or decoded without panicking.

use crate::{check_malformed, page_information, region_info, segment};

#[test]
fn reused_contexts_of_another_template() {
    check_malformed("malformed_reused_contexts_of_another_template");
}

#[test]
fn mmr_region_with_height_0() {
    check_malformed("malformed_mmr_region_with_height_0");
}

#[test]
fn huffman_value_out_of_range() {
    check_malformed("malformed_huffman_value_out_of_range");
}

#[test]
fn text_placement_out_of_range() {
    check_malformed("malformed_text_placement_out_of_range");
}

/// A generic region whose header declares as large a bitmap as the budget
/// of a 4096 × 4096 page allows, but whose segment carries no coded data,
/// stops decoding once the arithmetic decoder is exhausted, instead of
/// decoding 2^28 pixels from nothing.
#[test]
fn generic_region_without_data_stops_when_the_decoder_is_exhausted() {
    use hayro_jbig2::{DecodeSettings, Decoder, Page};

    struct Discard;

    impl Decoder for Discard {
        fn push_pixel(&mut self, _: bool) {}
        fn push_pixel_chunk(&mut self, _: bool, _: u32) {}
        fn next_line(&mut self) {}
    }

    // GBTEMPLATE 0 with its default AT pixels (7.4.6.3).
    let mut region = region_info(16_384, 16_384, 0, 0, 0);
    region.extend_from_slice(&[0, 3, 0xFF, 0xFD, 0xFF, 2, 0xFE, 0xFE, 0xFE]);
    let mut data = page_information(4096, 4096);
    data.extend(segment(1, 38, &region));

    let page = Page::new_embedded(&data, None, &DecodeSettings::default()).unwrap();
    let start = std::time::Instant::now();
    page.decode(&mut Discard).unwrap();
    // Decoding all rows takes seconds in a debug build; stopping at
    // exhaustion takes milliseconds.
    assert!(
        start.elapsed() < std::time::Duration::from_secs(1),
        "{:?}",
        start.elapsed()
    );
}
