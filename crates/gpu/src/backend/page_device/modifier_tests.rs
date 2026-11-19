//! The device's clips and soft-masked objects against its host model, on
//! both raster kernels.

use color::SampleDepth;

use super::path_tests::assert_device_matches_model;
use crate::backend::page_device::PageDevice;
use crate::backend::vulkan::test_backend;
use crate::page::PageCommands;
use crate::page::test_support::{Rng, object_over_tiles, push_random_modified};

fn random_modified_pages<B: PageDevice>(backend: &B, seed: u64, tables: bool) {
    if !tables && !backend.page_paths() {
        eprintln!("skipping: the device has no 64-bit integers");
        return;
    }
    let mut rng = Rng(seed);
    for depth in [SampleDepth::Eight, SampleDepth::Sixteen] {
        let max = depth.max_sample();
        for (width, height, n) in [(1, 1, 10), (37, 21, 60), (130, 70, 200), (301, 190, 400)] {
            let mut page = PageCommands::new(width, height, depth, [max, max - 9, max / 2]);
            push_random_modified(&mut rng, &mut page, n, tables);
            assert!(tables != page.has_paths() || page.is_empty() || width == 1);
            assert_device_matches_model(
                backend,
                &page,
                &format!("{}-bit {width}×{height}, tables {tables}", depth.bits()),
            );
        }
    }
}

fn clipped_and_masked_tables_match_the_host_model_on<B: PageDevice>(backend: &B) {
    random_modified_pages(backend, 0xC11F_7AB1, true);
}

fn clipped_and_masked_outlines_match_the_host_model_on<B: PageDevice>(backend: &B) {
    random_modified_pages(backend, 0xC11F_0A71, false);
}

/// An opaque tile-covering outline inside a soft-masked object, members
/// absent from some tiles and an object spanning tiles: device = model.
fn an_object_over_tiles_matches_the_host_model_on<B: PageDevice>(backend: &B) {
    if !backend.page_paths() {
        eprintln!("skipping: the device has no 64-bit integers");
        return;
    }
    for depth in [SampleDepth::Eight, SampleDepth::Sixteen] {
        let page = object_over_tiles(depth);
        assert_device_matches_model(backend, &page, &format!("{}-bit", depth.bits()));
    }
}

#[test]
fn vulkan_clipped_and_masked_tables_match_the_host_model() {
    if let Some(backend) = test_backend() {
        clipped_and_masked_tables_match_the_host_model_on(&backend);
    }
}

#[test]
fn vulkan_clipped_and_masked_outlines_match_the_host_model() {
    if let Some(backend) = test_backend() {
        clipped_and_masked_outlines_match_the_host_model_on(&backend);
    }
}

#[test]
fn vulkan_an_object_over_tiles_matches_the_host_model() {
    if let Some(backend) = test_backend() {
        an_object_over_tiles_matches_the_host_model_on(&backend);
    }
}
