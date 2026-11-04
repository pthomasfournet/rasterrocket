//! Streams whose decode would exceed the memory or compositing budget, which
//! must fail with an error instead of exhausting memory or time.

use hayro_jbig2::{DecodeError, DecodeSettings, Globals, OverflowError, Page};

use crate::{Collector, page_information, pattern_dictionary, region_info, segment};

/// An embedded segment (7.2) like [`segment`], referring to segment
/// `referred` (at most 255).
fn referring_segment(number: u32, segment_type: u8, referred: u8, data: &[u8]) -> Vec<u8> {
    let mut segment = number.to_be_bytes().to_vec();
    segment.extend_from_slice(&[segment_type, 0x20, referred, 1]);
    segment.extend_from_slice(&u32::try_from(data.len()).unwrap().to_be_bytes());
    segment.extend_from_slice(data);
    segment
}

/// An immediate halftone region segment (7.4.5) of 1 × 1 pixels at the page
/// origin, referring to pattern dictionary 1, whose `grid_width` ×
/// `grid_height` grid has one cell per pixel and no coded data.
fn halftone_region(grid_width: u32, grid_height: u32) -> Vec<u8> {
    let mut data = region_info(1, 1, 0, 0, 0);
    // HMMR 0, HTEMPLATE 0, HENABLESKIP 0, HCOMBOP OR, HDEFPIXEL 0.
    data.push(0);
    data.extend_from_slice(&grid_width.to_be_bytes());
    data.extend_from_slice(&grid_height.to_be_bytes());
    data.extend_from_slice(&[0; 8]);
    data.extend_from_slice(&256_u16.to_be_bytes());
    data.extend_from_slice(&0_u16.to_be_bytes());
    referring_segment(2, 22, 1, &data)
}

/// A 1 × 1 page holding pattern dictionary 1 and a halftone region using it.
fn halftone_page(pattern_size: u8, grid_width: u32, grid_height: u32) -> Vec<u8> {
    let mut data = page_information(1, 1);
    data.extend(pattern_dictionary(1, pattern_size, pattern_size, 2));
    data.extend(halftone_region(grid_width, grid_height));
    data
}

fn decode(data: &[u8], strict: bool) -> Result<Collector, DecodeError> {
    let page = Page::new_embedded(data, None, &DecodeSettings { strict }).unwrap();
    let mut collector = Collector::default();
    page.decode(&mut collector)?;
    Ok(collector)
}

/// The gray-scale image of a 4096 × 4096 halftone grid holds 2^24 32-bit
/// values, far more than a 1 × 1 page may allocate.
#[test]
fn halftone_grid_beyond_the_memory_budget_is_refused() {
    let data = halftone_page(1, 4096, 4096);

    assert_eq!(
        decode(&data, true).err(),
        Some(DecodeError::Overflow(OverflowError::MemoryBudget))
    );

    // An exhausted budget fails the page in lenient mode too, instead of
    // dropping the region.
    assert_eq!(
        decode(&data, false).err(),
        Some(DecodeError::Overflow(OverflowError::MemoryBudget))
    );
}

/// Rendering a 128 × 128 grid of 255 × 255 patterns writes 2^30 pixels,
/// far more than a 1 × 1 page may composite.
#[test]
fn halftone_rendering_beyond_the_work_budget_is_refused() {
    let data = halftone_page(255, 128, 128);

    for strict in [true, false] {
        assert_eq!(
            decode(&data, strict).err(),
            Some(DecodeError::Overflow(OverflowError::WorkBudget))
        );
    }
}

/// A halftone region well inside the budget still decodes.
#[test]
fn halftone_within_the_budget_decodes() {
    assert!(decode(&halftone_page(1, 64, 64), true).is_ok());
}

/// Global segments have a budget of their own: 65 535 patterns of 1 × 32
/// pixels, with their 32 pre-shifted copies, need 2^32 bits.
#[test]
fn globals_beyond_the_memory_budget_are_refused() {
    let data = pattern_dictionary(1, 1, 32, 65_535);

    assert_eq!(
        Globals::new(&data, &DecodeSettings::default()).err(),
        Some(DecodeError::Overflow(OverflowError::MemoryBudget))
    );
}

/// A symbol dictionary declaring 2^20 new symbols reserves a list entry
/// for each before decoding any, far more than a 1 × 1 page may allocate.
#[test]
fn symbol_dictionary_beyond_the_memory_budget_is_refused() {
    // SDHUFF 1 with the default tables, no exports, 2^20 new symbols.
    let mut dictionary = vec![0, 0x01];
    dictionary.extend_from_slice(&0_u32.to_be_bytes());
    dictionary.extend_from_slice(&(1_u32 << 20).to_be_bytes());
    let mut data = page_information(1, 1);
    data.extend(segment(1, 0, &dictionary));

    for strict in [true, false] {
        assert_eq!(
            decode(&data, strict).err(),
            Some(DecodeError::Overflow(OverflowError::MemoryBudget))
        );
    }
}

/// An immediate generic refinement region (7.4.7) of `size` × `size`
/// pixels with GRTEMPLATE 1, referring to no region and so refining the
/// page, with no coded data.
fn page_refinement(size: u32) -> Vec<u8> {
    let mut region = region_info(size, size, 0, 0, 0);
    region.push(0x01);
    let mut data = page_information(1, 1);
    data.extend(segment(1, 42, &region));
    data
}

/// Refining the page copies the area of the page under the region as its
/// reference, which is charged like the refined bitmap: either of two
/// 6000 × 6000 bitmaps fits the budget of a 1 × 1 page, both do not.
#[test]
fn page_area_of_a_refinement_is_charged() {
    assert!(decode(&page_refinement(4000), false).is_ok());
    assert_eq!(
        decode(&page_refinement(6000), false).err(),
        Some(DecodeError::Overflow(OverflowError::MemoryBudget))
    );
}
