//! Generic regions (6.2): templates, adaptive template pixels, TPGDON and MMR.

use crate::{Collector, check, check_crop, page_information, region_info, segment};

#[test]
fn template0() {
    check("generic_template0", "truchet");
}

#[test]
fn template1() {
    check("generic_template1", "truchet");
}

#[test]
fn template1_custom_at() {
    check("generic_template1_custom_at", "truchet");
}

#[test]
fn template1_custom_at_tpgdon() {
    check("generic_template1_custom_at_tpgdon", "maze");
}

#[test]
fn template2() {
    check("generic_template2", "truchet");
}

#[test]
fn template2_custom_at() {
    check("generic_template2_custom_at", "truchet");
}

#[test]
fn template2_custom_at_tpgdon() {
    check("generic_template2_custom_at_tpgdon", "maze");
}

#[test]
fn template3() {
    check("generic_template3", "truchet");
}

#[test]
fn template3_custom_at() {
    check("generic_template3_custom_at", "truchet");
}

#[test]
fn template3_custom_at_tpgdon() {
    check("generic_template3_custom_at_tpgdon", "maze");
}

#[test]
fn template0_custom_at() {
    check("generic_template0_custom_at", "truchet");
}

#[test]
fn template0_custom_at_tpgdon() {
    check("generic_template0_custom_at_tpgdon", "maze");
}

#[test]
fn template0_ext() {
    check("generic_template0_ext", "truchet");
}

#[test]
fn template0_ext_custom_at() {
    check("generic_template0_ext_custom_at", "truchet");
}

#[test]
fn template0_ext_custom_at_tpgdon() {
    check("generic_template0_ext_custom_at_tpgdon", "maze");
}

#[test]
fn template0_tpgdon() {
    check("generic_template0_tpgdon", "maze");
}

#[test]
fn template1_tpgdon() {
    check("generic_template1_tpgdon", "maze");
}

#[test]
fn template2_tpgdon() {
    check("generic_template2_tpgdon", "maze");
}

#[test]
fn template3_tpgdon() {
    check("generic_template3_tpgdon", "maze");
}

#[test]
fn mmr() {
    check("generic_mmr", "truchet");
}

#[test]
fn unknown_length() {
    check("generic_unknown_length", "truchet");
}

#[test]
fn unknown_length_mmr() {
    check("generic_unknown_length_mmr", "truchet");
}

#[test]
fn unknown_length_custom_at() {
    check("generic_unknown_length_custom_at", "truchet");
}

#[test]
fn trailing_7fff_stripped() {
    check("generic_trailing_7fff_stripped", "honeycomb");
}

#[test]
fn trailing_7fff_stripped_many() {
    check("generic_trailing_7fff_stripped_many", "truchet");
}

#[test]
fn row_ending_in_full_chunks() {
    check_crop(
        "generic_row_ending_in_full_chunks",
        "truchet",
        (0, 30, 32, 23),
    );
}

/// A region location beyond `i32::MAX` lies to the right of the page; it
/// does not wrap around to a negative offset (7.4.1.3).
#[test]
fn region_location_beyond_i32_is_off_page() {
    use hayro_jbig2::{DecodeSettings, Page};

    // An MMR region without data decodes white; XNOR turns the page pixels
    // it covers black.
    let mut region = region_info(16, 1, 0xFFFF_FFF8, 0, 3);
    region.push(0x01);
    let mut data = page_information(8, 1);
    data.extend(segment(1, 38, &region));

    let page = Page::new_embedded(&data, None, &DecodeSettings::default()).unwrap();
    let mut collector = Collector::default();
    page.decode(&mut collector).unwrap();
    assert_eq!(collector.rows, vec![vec![false; 8]]);
}
