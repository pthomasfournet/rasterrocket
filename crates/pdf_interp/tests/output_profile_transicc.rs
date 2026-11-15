//! The conversion to an output profile against an independent CMM: Little
//! CMS's `transicc` converts the same patch chart from the same working
//! space profile, and both results, decoded to Lab through `transicc`, lie
//! within ΔE00 1 of each other on every patch, for every intent with and
//! without black-point compensation, at 8 and 16 bits.
//!
//! Needs `transicc` (Debian/Ubuntu `liblcms2-utils`):
//! `cargo test -p rasterrocket-interp --test output_profile_transicc -- --ignored`.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use color::{Rgb8, Rgb16, Sample};
use moxcms::ColorProfile;
use raster::Bitmap;
use rasterrocket_interp::renderer::{CanvasBuffers, FinishedCanvas};
use rasterrocket_interp::{OutputProfile, RenderingIntent};

/// The largest ΔE00 accepted between the two conversions.
const MAX_DELTA_E: f64 = 1.0;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/icc")
        .join(name)
}

/// The working space as a profile file: sRGB described by its tags.
fn working_space_file(dir: &Path) -> PathBuf {
    let mut srgb = ColorProfile::new_srgb();
    srgb.cicp = None;
    let path = dir.join("working-space.icc");
    std::fs::write(&path, srgb.encode().expect("encodes")).unwrap();
    path
}

/// A 9-level RGB cube, a 64-step neutral ramp and the corners.
fn patches() -> Vec<[f64; 3]> {
    let mut out = Vec::new();
    for r in 0..9 {
        for g in 0..9 {
            for b in 0..9 {
                out.push([r, g, b].map(|v: i32| f64::from(v) / 8.0));
            }
        }
    }
    for i in 0..64 {
        out.push([f64::from(i) / 63.0; 3]);
    }
    out
}

/// `transicc` with `args` on the lines of `input`, its output values.
fn transicc(args: &[&str], input: &str) -> Vec<Vec<f64>> {
    let mut child = Command::new("transicc")
        .arg("-n")
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("transicc runs: install liblcms2-utils");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "transicc {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split_whitespace().map(|v| v.parse().unwrap()).collect())
        .collect()
}

/// Lines of `values`, each scaled by `full`.
fn lines(values: &[Vec<f64>], full: f64) -> String {
    values
        .iter()
        .map(|v| {
            v.iter()
                .map(|c| format!("{:.4}", c * full))
                .collect::<Vec<_>>()
                .join(" ")
                + "\n"
        })
        .collect()
}

/// CIEDE2000 (Sharma, Wu and Dalal 2005), `kL = kC = kH = 1`.
#[expect(
    clippy::suboptimal_flops,
    reason = "the formula written term by term as published, checked against its test pairs"
)]
fn delta_e00(a: &[f64], b: &[f64]) -> f64 {
    let (l1, a1, b1) = (a[0], a[1], a[2]);
    let (l2, a2, b2) = (b[0], b[1], b[2]);
    let c_bar = f64::midpoint(a1.hypot(b1), a2.hypot(b2));
    let g = 0.5 * (1.0 - (c_bar.powi(7) / (c_bar.powi(7) + 25f64.powi(7))).sqrt());
    let (a1p, a2p) = ((1.0 + g) * a1, (1.0 + g) * a2);
    let (c1p, c2p) = (a1p.hypot(b1), a2p.hypot(b2));
    let hue = |b: f64, a: f64| {
        if b == 0.0 && a == 0.0 {
            0.0
        } else {
            b.atan2(a).to_degrees().rem_euclid(360.0)
        }
    };
    let (h1p, h2p) = (hue(b1, a1p), hue(b2, a2p));
    let dl = l2 - l1;
    let dc = c2p - c1p;
    let dh = if c1p * c2p == 0.0 {
        0.0
    } else if (h2p - h1p).abs() <= 180.0 {
        h2p - h1p
    } else if h2p - h1p > 180.0 {
        h2p - h1p - 360.0
    } else {
        h2p - h1p + 360.0
    };
    let dhh = 2.0 * (c1p * c2p).sqrt() * (dh / 2.0).to_radians().sin();
    let lbp = f64::midpoint(l1, l2);
    let cbp = f64::midpoint(c1p, c2p);
    let hbp = if c1p * c2p == 0.0 {
        h1p + h2p
    } else if (h1p - h2p).abs() <= 180.0 {
        f64::midpoint(h1p, h2p)
    } else if h1p + h2p < 360.0 {
        (h1p + h2p + 360.0) / 2.0
    } else {
        (h1p + h2p - 360.0) / 2.0
    };
    let t = 1.0 - 0.17 * (hbp - 30.0).to_radians().cos()
        + 0.24 * (2.0 * hbp).to_radians().cos()
        + 0.32 * (3.0 * hbp + 6.0).to_radians().cos()
        - 0.20 * (4.0 * hbp - 63.0).to_radians().cos();
    let d_theta = 30.0 * (-((hbp - 275.0) / 25.0).powi(2)).exp();
    let rc = 2.0 * (cbp.powi(7) / (cbp.powi(7) + 25f64.powi(7))).sqrt();
    let sl = 1.0 + 0.015 * (lbp - 50.0).powi(2) / (20.0 + (lbp - 50.0).powi(2)).sqrt();
    let sc = 1.0 + 0.045 * cbp;
    let sh = 1.0 + 0.015 * cbp * t;
    let rt = -(2.0 * d_theta).to_radians().sin() * rc;
    ((dl / sl).powi(2) + (dc / sc).powi(2) + (dhh / sh).powi(2) + rt * (dc / sc) * (dhh / sh))
        .sqrt()
}

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
    let dir = tempfile_dir();
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

/// A directory of this test's own.
fn tempfile_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "rr-output-profile-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
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
