//! ICC-based colours rendered under each rendering intent against an
//! independent CMM: a page of patches in an `ICCBased` space, filled under
//! `ri`, rendered at 8 and 16 bits, and the same patches converted by
//! Little CMS's `transicc` from the same profile to its built-in sRGB.
//! Both results are decoded to Lab through `transicc` and compared by
//! ΔE00 on every patch, for every intent and every test profile of
//! `tests/fixtures/icc`: table-based CMYK and RGB printers whose tables
//! differ by intent and whose paper white is tinted, and a gray TRC one.
//!
//! Each intent but absolute colorimetry renders with black-point
//! compensation on (`/UseBlackPtComp /ON`, against `transicc -b`) and
//! off; absolute colorimetry, which is never compensated, once.
//!
//! Asserted, within the bounds of each profile at 8 and 16 bits
//! ([`CMYK_TABLES`], [`CMYK_OFF_NODE`], [`RGB_TABLES`], [`GRAY`]): every
//! case but perceptual and saturation without compensation, which are
//! printed only: Little CMS compensates the black of a perceptual or
//! saturation conversion into a version 4 profile regardless of `-b`, and
//! its built-in sRGB is one, where `/UseBlackPtComp /OFF` leaves it.
//!
//! Needs `transicc` (Debian/Ubuntu `liblcms2-utils`):
//! `cargo test -p rasterrocket --test input_profile_transicc -- --ignored`.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use std::fmt::Write as _;
use std::io::Write as _;
use std::process::{Command, Stdio};

use common::{one_page, stream_object};
use rasterrocket::{
    BackendPolicy, FinishedCanvas, RenderParams, SampleDepth, SessionConfig,
    open_session_from_bytes, render_page_canvas,
};

/// The largest ΔE00 accepted between the two conversions at 8 and at 16
/// bits.
#[derive(Clone, Copy)]
struct Bounds {
    eight: f64,
    sixteen: f64,
}

/// The bounds of the gray profile.
const GRAY: Bounds = Bounds {
    eight: 1.5,
    sixteen: 1.0,
};

/// The bounds of the RGB printers, whose tables are evaluated exactly at
/// the nodes of a grid over the device cube and interpolated tetrahedrally
/// between them.
const RGB_TABLES: Bounds = Bounds {
    eight: 0.75,
    sixteen: 0.1,
};

/// The bounds of the CMYK printers, tabulated as the RGB ones are and
/// interpolated in the 4-simplices of the grid; at 8 bits a level of the
/// output is near 1 ΔE00 in the shadows the CMYK patches reach.
const CMYK_TABLES: Bounds = Bounds {
    eight: 1.0,
    sixteen: 0.1,
};

/// The bounds of the CMYK printers between the nodes of their tables:
/// on [`k_ramp`] and on the lattice of 9 levels offset by half a step,
/// `(i + 1/2) / 9` along each ink.
const CMYK_OFF_NODE: Bounds = Bounds {
    eight: 1.1,
    sixteen: 0.5,
};

/// The path of the test profile `name` of `tests/fixtures/icc`.
fn profile_path(name: &str) -> String {
    format!(
        "{}/../../tests/fixtures/icc/{name}.icc",
        env!("CARGO_MANIFEST_DIR")
    )
}

/// The lattice of `levels` levels along each of `n` components, level
/// `i` at `(i + offset) / (levels − 1 + 2 · offset)`, the first component
/// varying slowest.
fn lattice(n: usize, levels: u32, offset: f64) -> Vec<Vec<f64>> {
    let steps = 2.0f64.mul_add(offset, f64::from(levels - 1));
    let mut out = vec![Vec::new()];
    for _ in 0..n {
        out = out
            .into_iter()
            .flat_map(|p| {
                (0..levels).map(move |i| {
                    let mut q = p.clone();
                    q.push((f64::from(i) + offset) / steps);
                    q
                })
            })
            .collect();
    }
    out
}

/// The patches of an `n`-component space, components in `[0, 1]`: the
/// lattice of 64 levels of gray, 9 of RGB or 5 of CMYK.
fn patches(n: usize) -> Vec<Vec<f64>> {
    let levels = match n {
        1 => 64,
        3 => 9,
        _ => 5,
    };
    lattice(n, levels, 0.0)
}

/// A K ramp of 256 levels at `C = M = Y = 0`, which falls between the
/// nodes of a table along its K axis.
fn k_ramp() -> Vec<Vec<f64>> {
    (0..256u32)
        .map(|k| vec![0.0, 0.0, 0.0, f64::from(k) / 255.0])
        .collect()
}

/// A one-point-high page with one patch a point, each filled with its
/// components in `[/ICCBased profile]` under `intent`, black-point
/// compensation `bpc`.
fn patch_page(
    profile: &[u8],
    n: usize,
    (intent, bpc): (&str, bool),
    patches: &[Vec<f64>],
) -> Vec<u8> {
    let bpc = if bpc { "ON" } else { "OFF" };
    let mut content = format!("/B gs /{intent} ri /P cs\n");
    for (x, p) in patches.iter().enumerate() {
        for v in p {
            let _ = write!(content, "{v} ");
        }
        let _ = writeln!(content, "sc {x} 0 1 1 re f");
    }
    let page = format!(
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {} 1] \
         /Resources << /ColorSpace << /P [/ICCBased 4 0 R] >> \
         /ExtGState << /B << /UseBlackPtComp /{bpc} >> >> >> /Contents 5 0 R >>",
        patches.len()
    );
    one_page(
        &page,
        vec![
            stream_object(&format!("/N {n}"), profile),
            stream_object("", content.as_bytes()),
        ],
    )
}

/// Our sRGB of each patch, at `depth`, in `[0, 255]`.
fn ours(pdf: Vec<u8>, depth: SampleDepth) -> Vec<[f64; 3]> {
    let session = open_session_from_bytes(pdf, &SessionConfig::with_policy(BackendPolicy::CpuOnly))
        .expect("opens");
    let canvas = render_page_canvas(&session, 1, &RenderParams::new(1.0).with_depth(depth))
        .expect("renders");
    let samples: Vec<f64> = match canvas {
        FinishedCanvas::Rgb8(b) => b.row_samples(0).iter().map(|&v| f64::from(v)).collect(),
        FinishedCanvas::Rgb16(b) => b
            .row_samples(0)
            .iter()
            .map(|&v| f64::from(v) * 255.0 / 65535.0)
            .collect(),
        other => panic!("an RGB render, not {other:?}"),
    };
    samples.as_chunks::<3>().0.to_vec()
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
    // The input is written while the output is read: a pipe buffers a
    // bounded amount, and the converter writes each line's answer before
    // it reads the next.
    let mut stdin = child.stdin.take().unwrap();
    let out = std::thread::scope(|s| {
        let writer = s.spawn(move || stdin.write_all(input.as_bytes()).unwrap());
        let out = child.wait_with_output().unwrap();
        writer.join().unwrap();
        out
    });
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

/// The Lab of sRGB values in `[0, 255]`, through `transicc`.
fn lab(rgb: &[[f64; 3]]) -> Vec<Vec<f64>> {
    let mut input = String::new();
    for p in rgb {
        let _ = writeln!(input, "{:.4} {:.4} {:.4}", p[0], p[1], p[2]);
    }
    transicc(&["-i*sRGB", "-o*Lab", "-t1"], &input)
}

/// CIEDE2000 (Sharma, Wu and Dalal 2005), `kL = kC = kH = 1`.
#[expect(
    clippy::suboptimal_flops,
    reason = "the formula written term by term as published"
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

/// Our rendering of the patches of profile `name` (`n` components)
/// against `transicc` under every intent at both depths, asserted within
/// `bounds` but for the perceptual and saturation intents without
/// compensation.
fn agrees(name: &str, n: usize, (label, chart): (&str, &[Vec<f64>]), bounds: Bounds) {
    let path = profile_path(name);
    let profile = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    // transicc reads CMYK in percent and gray or RGB in 0 to 255.
    let full = if n == 4 { 100.0 } else { 255.0 };
    let mut failures = Vec::new();
    for (code, intent, bpc) in [
        ("0", "Perceptual", true),
        ("1", "RelativeColorimetric", true),
        ("2", "Saturation", true),
        ("3", "AbsoluteColorimetric", false),
        ("0", "Perceptual", false),
        ("1", "RelativeColorimetric", false),
        ("2", "Saturation", false),
    ] {
        let claimed = bpc || !matches!(code, "0" | "2");
        let pdf = patch_page(&profile, n, (intent, bpc), chart);
        for (depth, quantum, bound) in [
            (SampleDepth::Eight, 255.0, bounds.eight),
            (SampleDepth::Sixteen, 65535.0, bounds.sixteen),
        ] {
            // Both convert the components quantised to the render's depth.
            let input: String = chart
                .iter()
                .map(|p| {
                    p.iter()
                        .map(|v| format!("{:.6}", (v * quantum).round() / quantum * full))
                        .collect::<Vec<_>>()
                        .join(" ")
                        + "\n"
                })
                .collect();
            let mut args = vec![
                format!("-i{path}"),
                "-o*sRGB".to_owned(),
                format!("-t{code}"),
            ];
            if bpc {
                args.push("-b".to_owned());
            }
            let args: Vec<&str> = args.iter().map(String::as_str).collect();
            let theirs: Vec<[f64; 3]> = transicc(&args, &input)
                .iter()
                .map(|v| [v[0], v[1], v[2]].map(|c| c.clamp(0.0, 255.0)))
                .collect();
            let ours = ours(pdf.clone(), depth);
            assert_eq!(
                ours.len(),
                chart.len(),
                "{name} {intent} bpc {bpc}: one pixel a patch"
            );
            let (lab_ours, lab_theirs) = (lab(&ours), lab(&theirs));
            let (mut worst, mut sum) = (0.0f64, 0.0);
            let mut worst_at = 0;
            for (i, (a, b)) in lab_ours.iter().zip(&lab_theirs).enumerate() {
                let de = delta_e00(a, b);
                sum += de;
                if de > worst {
                    (worst, worst_at) = (de, i);
                }
            }
            #[expect(
                clippy::cast_precision_loss,
                reason = "a patch count is far below 2^52"
            )]
            let mean = sum / chart.len() as f64;
            eprintln!(
                "{name} {label} {intent} bpc {bpc} {depth:?}: mean ΔE00 {mean:.3}, max {worst:.3} at {:?} \
                 (ours {:?}, transicc {:?})",
                chart[worst_at], ours[worst_at], theirs[worst_at]
            );
            if claimed && worst > bound {
                failures.push(format!(
                    "{label} {intent} bpc {bpc} {depth:?}: max ΔE00 {worst:.3}"
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{name}: {failures:?}");
}

#[test]
#[ignore = "needs transicc (liblcms2-utils)"]
fn cmyk_printers_agree_with_little_cms() {
    let (nodes, ramp, half) = (patches(4), k_ramp(), lattice(4, 9, 0.5));
    for name in ["printer-cmyk-v4", "printer-cmyk-v2"] {
        agrees(name, 4, ("lattice", &nodes), CMYK_TABLES);
        agrees(name, 4, ("k-ramp", &ramp), CMYK_OFF_NODE);
        agrees(name, 4, ("half-step", &half), CMYK_OFF_NODE);
    }
}

#[test]
#[ignore = "needs transicc (liblcms2-utils)"]
fn gray_printer_agrees_with_little_cms() {
    agrees("printer-gray-v4", 1, ("lattice", &patches(1)), GRAY);
}

#[test]
#[ignore = "needs transicc (liblcms2-utils)"]
fn rgb_printers_agree_with_little_cms() {
    let chart = patches(3);
    for name in ["printer-rgb-v4", "printer-rgb-v2"] {
        agrees(name, 3, ("lattice", &chart), RGB_TABLES);
    }
}
