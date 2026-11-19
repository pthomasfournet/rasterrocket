//! The device's masks against its host model, on both raster kernels.

use color::SampleDepth;

use super::path_tests::{assert_device_matches_model, random_page};
use crate::backend::page_device::PageDevice;
use crate::page::test_support::Rng;
use crate::page::{MaskBits, MaskId, PageCommands, PixelBounds};

/// Push `n` placements of a handful of random masks onto `page`: random
/// positions (partly off the page), bounds, colours and alphas.
fn push_random_masks(rng: &mut Rng, page: &mut PageCommands, n: usize) {
    let max = page.depth().max_sample();
    let (width, height) = (i64::from(page.width()), i64::from(page.height()));
    let masks: Vec<MaskId> = (0..5)
        .map(|k| {
            let (w, h) = (
                u32::try_from(rng.range(1, 45)).unwrap(),
                u32::try_from(rng.range(1, 37)).unwrap(),
            );
            let bits = if k == 4 {
                MaskBits::Mono
            } else {
                MaskBits::Coverage
            };
            let data: Vec<u8> = (0..w * h).map(|_| rng.byte()).collect();
            page.add_mask(&data, w, h, bits).expect("add mask")
        })
        .collect();
    let coord = |rng: &mut Rng, extent: i64| i32::try_from(rng.range(-40, extent + 40)).unwrap();
    for _ in 0..n {
        let mask = masks[usize::try_from(rng.range(0, 5)).unwrap()];
        let origin = (coord(rng, width), coord(rng, height));
        let (x0, y0) = (coord(rng, width), coord(rng, height));
        let bounds = PixelBounds {
            x0,
            y0,
            x1: x0 + i32::try_from(rng.range(0, width + 40)).unwrap(),
            y1: y0 + i32::try_from(rng.range(0, height + 40)).unwrap(),
        };
        let alpha =
            [max, max, max / 2 + 1, 1, rng.sample(max)][usize::try_from(rng.range(0, 5)).unwrap()];
        page.push_mask(mask, origin, bounds, [0; 3].map(|_| rng.sample(max)), alpha)
            .expect("push mask");
    }
}

fn random_masks_match_the_host_model_on<B: PageDevice>(backend: &B) {
    let mut rng = Rng(0x3A5C_0DE5);
    for depth in [SampleDepth::Eight, SampleDepth::Sixteen] {
        let max = depth.max_sample();
        for (width, height, n) in [(1, 1, 5), (37, 21, 60), (130, 70, 400), (403, 260, 900)] {
            let mut page = PageCommands::new(width, height, depth, [max, max - 9, max / 2]);
            push_random_masks(&mut rng, &mut page, n);
            assert!(!page.has_paths());
            assert_device_matches_model(
                backend,
                &page,
                &format!("{}-bit {width}×{height}, {n} masks", depth.bits()),
            );
        }
    }
}

fn masks_among_outlines_match_the_host_model_on<B: PageDevice>(backend: &B) {
    if !backend.page_paths() {
        eprintln!("skipping: the device has no 64-bit integers");
        return;
    }
    let mut rng = Rng(0x0A7_1A5);
    for depth in [SampleDepth::Eight, SampleDepth::Sixteen] {
        for (width, height, n) in [(37, 21, 30), (300, 170, 200)] {
            let mut page = random_page(&mut rng, width, height, n, depth);
            push_random_masks(&mut rng, &mut page, n);
            let mut more = random_page(&mut rng, width, height, n / 3, depth);
            push_random_masks(&mut rng, &mut more, n);
            for page in [page, more] {
                assert!(page.has_paths());
                assert_device_matches_model(
                    backend,
                    &page,
                    &format!("{}-bit {width}×{height}, outlines and masks", depth.bits()),
                );
            }
        }
    }
}

on_each_backend! {
    random_masks_match_the_host_model => random_masks_match_the_host_model_on;
    masks_among_outlines_match_the_host_model => masks_among_outlines_match_the_host_model_on;
}
