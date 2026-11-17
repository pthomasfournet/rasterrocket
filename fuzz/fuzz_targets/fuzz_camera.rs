#![no_main]

//! The camera-capture cleanup stages on arbitrary 8-bit gray images: page
//! quad detection, crop, rectification, gutter detection and spread split,
//! dewarp, the replay of the measured geometry at 16 bits, background
//! normalisation, border-junk whitening and binarisation.
//!
//! Layout: `[width_lo, width_hi, stride_pad, stages, threshold]`, then the
//! samples.  The width is the little-endian `u16` plus one, capped at
//! 4096; rows are `stride_pad` samples longer than the width, and the
//! height is as many rows as the samples fill (at most 4096).  Bits 0–3 of
//! `stages` enable crop, rectify, spread split and dewarp, bits 4–7 set the
//! resolution the background tiles are sized for (75 to 1200 dpi); `threshold`
//! below 128 is a fixed threshold, otherwise a Sauvola window of
//! `threshold - 127` pixels.

use camera::{
    CameraOptions, GrayPage, GrayView, Threshold, background_tile_px, binarize, find_border_junk,
    measure_background, process,
};
use libfuzzer_sys::fuzz_target;

/// Largest image side, in pixels.
const MAX_SIDE: usize = 4096;

fuzz_target!(|data: &[u8]| {
    let Some(([lo, hi, pad, stages, threshold], samples)) =
        rasterrocket_fuzz::split_selectors::<5>(data)
    else {
        return;
    };
    let width = (usize::from(u16::from_le_bytes([lo, hi])) + 1).min(MAX_SIDE);
    let stride = width + usize::from(pad);
    // Rows of `stride` samples, the last one `width` long.
    let height = ((samples.len() + stride - width) / stride).min(MAX_SIDE);
    if height == 0 {
        return;
    }
    let img = GrayView {
        data: samples,
        width,
        height,
        stride,
    };

    let mut opts = CameraOptions::default();
    opts.crop = stages & 1 != 0;
    opts.rectify = stages & 2 != 0;
    opts.split_spreads = stages & 4 != 0;
    opts.dewarp = stages & 8 != 0;
    let dpi = 75 * (1 + u32::from(stages >> 4));
    if let Ok(processed) = process(img, &opts) {
        let wide: Vec<u16> = samples.iter().map(|&v| u16::from(v) * 257).collect();
        let wide = GrayView {
            data: &wide,
            width,
            height,
            stride,
        };
        drop(processed.report.replay(wide));
    }

    if let Ok(found) = measure_background(img, background_tile_px(dpi))
        && let Some(background) = found.found()
    {
        drop(background.normalize(img));
    }
    if let Ok(found) = find_border_junk(img)
        && let Some(mask) = found.found()
    {
        let mut page = GrayPage {
            width,
            height,
            data: (0..height)
                .flat_map(|y| &samples[y * stride..y * stride + width])
                .copied()
                .collect(),
        };
        drop(mask.whiten(&mut page));
    }
    let method = if threshold < 128 {
        Threshold::Fixed(threshold * 2)
    } else {
        Threshold::Sauvola {
            window_px: u32::from(threshold - 127),
            k_per_mille: 340,
        }
    };
    drop(binarize(img, method));
});
