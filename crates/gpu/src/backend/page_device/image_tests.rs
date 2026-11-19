//! The device's image commands against its host model, on both raster
//! kernels.

use color::SampleDepth;

use super::path_tests::{assert_device_matches_model, random_page};
use crate::backend::page_device::PageDevice;
use crate::page::PageCommands;
use crate::page::test_support::{Rng, push_random_images};

fn random_images_match_the_host_model_on<B: PageDevice>(backend: &B) {
    let mut rng = Rng(0x1_3A6E_5EED);
    for depth in [SampleDepth::Eight, SampleDepth::Sixteen] {
        let max = depth.max_sample();
        for (width, height, n) in [(1, 1, 4), (37, 21, 30), (130, 70, 120), (403, 260, 300)] {
            let mut page = PageCommands::new(width, height, depth, [max, max - 9, max / 2]);
            push_random_images(&mut rng, &mut page, n);
            assert!(!page.has_paths());
            assert_device_matches_model(
                backend,
                &page,
                &format!("{}-bit {width}×{height}, {n} images", depth.bits()),
            );
        }
    }
}

fn images_among_outlines_match_the_host_model_on<B: PageDevice>(backend: &B) {
    if !backend.page_paths() {
        eprintln!("skipping: the device has no 64-bit integers");
        return;
    }
    let mut rng = Rng(0x0_1A6E_0F11);
    for depth in [SampleDepth::Eight, SampleDepth::Sixteen] {
        for (width, height, n) in [(37, 21, 20), (300, 170, 120)] {
            let mut page = random_page(&mut rng, width, height, n, depth);
            push_random_images(&mut rng, &mut page, n);
            assert!(page.has_paths());
            assert_device_matches_model(
                backend,
                &page,
                &format!("{}-bit {width}×{height}, outlines and images", depth.bits()),
            );
        }
    }
}

on_each_backend! {
    random_images_match_the_host_model => random_images_match_the_host_model_on;
    images_among_outlines_match_the_host_model => images_among_outlines_match_the_host_model_on;
}
