//! The conversion to an output profile against an independent CMM: Little
//! CMS's `transicc` converts the same patch chart from the same working
//! space profile, and both results, decoded to Lab through `transicc`, lie
//! within ΔE00 1 of each other on every patch, for every intent with and
//! without black-point compensation, at 8 and 16 bits.
//!
//! Needs `transicc` (Debian/Ubuntu `liblcms2-utils`):
//! `cargo test -p rasterrocket-interp --test output_profile_transicc -- --ignored`.

mod common;

use color::{Rgb8, Rgb16, Sample};
use common::{
    MAX_DELTA_E, delta_e00, fixture, lines, patches, tempfile_dir, transicc, working_space_file,
};
use raster::Bitmap;
use rasterrocket_interp::renderer::{CanvasBuffers, FinishedCanvas};
use rasterrocket_interp::{OutputProfile, RenderingIntent};

#[test]
fn ciede2000_matches_published_pairs() {
    // Sharma, Wu and Dalal, Table 1: pairs 1, 7, 17 and 25.
    for (a, b, want) in [
        ([50.0, 2.6772, -79.7751], [50.0, 0.0, -82.7485], 2.0425),
        ([50.0, 0.0, 0.0], [50.0, -1.0, 2.0], 2.3669),
        ([50.0, 2.5, 0.0], [73.0, 25.0, -18.0], 27.1492),
        (
            [60.2574, -34.0099, 36.2677],
            [60.4626, -34.1751, 39.4387],
            1.2644,
        ),
    ] {
        let got = delta_e00(&a, &b);
        assert!((got - want).abs() < 1e-4, "{a:?} {b:?}: {got} vs {want}");
    }
}

/// Our conversion of `patches` at `bits` to `profile` under `intent`.
fn ours(
    profile: &OutputProfile,
    (intent, bpc): (RenderingIntent, bool),
    patches: &[[f64; 3]],
    bits: u8,
) -> Vec<Vec<f64>> {
    let width = u32::try_from(patches.len()).unwrap();
    let mut buffers = CanvasBuffers::default();
    let canvas = if bits == 8 {
        let mut bmp = Bitmap::<Rgb8>::new(width, 1, 1, false);
        for (px, p) in bmp
            .row_samples_mut(0)
            .as_chunks_mut::<3>()
            .0
            .iter_mut()
            .zip(patches)
        {
            *px = p.map(u8::from_unit);
        }
        FinishedCanvas::Rgb8(bmp)
    } else {
        let mut bmp = Bitmap::<Rgb16>::new(width, 1, 1, false);
        for (px, p) in bmp
            .row_samples_mut(0)
            .as_chunks_mut::<3>()
            .0
            .iter_mut()
            .zip(patches)
        {
            *px = p.map(u16::from_unit);
        }
        FinishedCanvas::Rgb16(bmp)
    };
    let full = if bits == 8 { 255.0 } else { 65535.0 };
    let samples: Vec<f64> = match profile
        .convert(canvas, (intent, bpc), &mut buffers)
        .expect("converts")
    {
        FinishedCanvas::Rgb8(b) => b.row_samples(0).iter().map(|&v| f64::from(v)).collect(),
        FinishedCanvas::Rgb16(b) => b.row_samples(0).iter().map(|&v| f64::from(v)).collect(),
        FinishedCanvas::Gray8(b) => b.row_samples(0).iter().map(|&v| f64::from(v)).collect(),
        FinishedCanvas::Gray16(b) => b.row_samples(0).iter().map(|&v| f64::from(v)).collect(),
        other => panic!("an RGB or gray conversion, not {other:?}"),
    };
    let n = samples.len() / patches.len();
    samples
        .chunks_exact(n)
        .map(|px| px.iter().map(|v| v / full).collect())
        .collect()
}

/// Both conversions of the chart to `name` within [`MAX_DELTA_E`].
fn agrees(name: &str) {
    let dir = tempfile_dir("output-profile");
    let src = working_space_file(&dir);
    let dst = fixture(name);
    let profile = OutputProfile::from_bytes(std::fs::read(&dst).unwrap()).unwrap();
    let (src, dst) = (src.to_str().unwrap(), dst.to_str().unwrap());
    let chart = patches();
    let mut worst = 0.0f64;
    let mut failures = Vec::new();
    for (code, intent) in [
        ("0", RenderingIntent::Perceptual),
        ("1", RenderingIntent::RelativeColorimetric),
        ("2", RenderingIntent::Saturation),
        ("3", RenderingIntent::AbsoluteColorimetric),
    ] {
        for bpc in [false, true] {
            if bpc && intent == RenderingIntent::AbsoluteColorimetric {
                continue;
            }
            for (bits, full) in [(8u8, 255.0), (16, 65535.0)] {
                let mut args = vec![format!("-i{src}"), format!("-o{dst}"), format!("-t{code}")];
                if bpc {
                    args.push("-b".to_owned());
                }
                let args: Vec<&str> = args.iter().map(String::as_str).collect();
                // Both convert the chart's samples of `bits`.
                let chart_lines: Vec<Vec<f64>> = chart
                    .iter()
                    .map(|p| p.iter().map(|v| (v * full).round() / full).collect())
                    .collect();
                // transicc reads and writes floating-point values of 0 to 255:
                // its unrounded result is the reference ours, of integer
                // samples, is measured against.
                let theirs: Vec<Vec<f64>> = transicc(&args, &lines(&chart_lines, 255.0))
                    .iter()
                    .map(|v| v.iter().map(|c| c / 255.0).collect())
                    .collect();
                let mine = ours(&profile, (intent, bpc), &chart, bits);
                let decode = |device: &[Vec<f64>]| {
                    transicc(
                        &[&format!("-i{dst}"), "-o*Lab", "-t1"],
                        &lines(device, 255.0),
                    )
                };
                let (lab_theirs, lab_mine) = (decode(&theirs), decode(&mine));
                let (mut max, mut sum, mut at) = (0.0f64, 0.0, 0);
                for (i, (a, b)) in lab_theirs.iter().zip(&lab_mine).enumerate() {
                    let de = delta_e00(a, b);
                    sum += de;
                    if de > max {
                        (max, at) = (de, i);
                    }
                }
                #[expect(clippy::cast_precision_loss, reason = "a patch count")]
                let mean = sum / lab_mine.len() as f64;
                eprintln!(
                    "{name} {intent} bpc {bpc} {bits}-bit: ΔE00 max {max:.3} (patch {:?}), mean {mean:.3}",
                    chart[at]
                );
                if max > MAX_DELTA_E {
                    failures.push(format!(
                        "{name} {intent} bpc {bpc} {bits}-bit: ΔE00 {max:.3} at patch {:?}: \
                         ours {:?}, transicc {:?}",
                        chart[at], mine[at], theirs[at]
                    ));
                }
                worst = worst.max(max);
            }
        }
    }
    eprintln!("{name}: worst ΔE00 {worst:.3}");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
#[ignore = "needs transicc (liblcms2-utils)"]
fn v4_lut_rgb_printer_agrees_with_transicc() {
    agrees("printer-rgb-v4.icc");
}

#[test]
#[ignore = "needs transicc (liblcms2-utils)"]
fn v2_lut16_rgb_printer_agrees_with_transicc() {
    agrees("printer-rgb-v2.icc");
}

#[test]
#[ignore = "needs transicc (liblcms2-utils)"]
fn gray_printer_agrees_with_transicc() {
    agrees("printer-gray-v4.icc");
}
