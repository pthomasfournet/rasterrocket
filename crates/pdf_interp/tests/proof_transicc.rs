//! The soft proof against an independent CMM: Little CMS's `transicc -p`
//! proofs the same patch chart from the same working space through the
//! same press to the same display, and both results, decoded to Lab
//! through `transicc`, lie within ΔE00 1 of each other on every patch at
//! 16 bits, and within 1.2 at 8 bits, where ink and display values are
//! both rounded to 8 bits.
//!
//! The chart's colours are separated to the press by the press profile's
//! separation table, as a page's colours are painted on an ink canvas,
//! under every first-leg intent with and without black-point
//! compensation; the ink is proofed under both proof intents, at 8 and 16
//! bits, on the working space and on an RGB display profile.  Ink given as
//! such (a page's `DeviceCMYK`) is proofed without the first leg and
//! measured against `transicc` from the press.
//!
//! Needs `transicc` (Debian/Ubuntu `liblcms2-utils`):
//! `cargo test -p rasterrocket-interp --test proof_transicc -- --ignored`.

mod common;

use std::path::Path;
use std::sync::Arc;

use color::{Cmyk8, Cmyk16, Sample};
use common::{
    MAX_DELTA_E, delta_e00, fixture, lines, patches, tempfile_dir, transicc, working_space_file,
};
use raster::Bitmap;
use rasterrocket_interp::renderer::{CanvasBuffers, FinishedCanvas};
use rasterrocket_interp::{OutputProfile, ProofIntent, RenderingIntent, Separation, SoftProof};

/// The largest ΔE00 accepted for the chart at 8 bits: the page's ink and
/// its display values are each rounded to 8 bits, two quantisations
/// measured against `transicc`'s unrounded floating-point output.  Ink
/// proofs and 16-bit charts are held to [`MAX_DELTA_E`].
const MAX_DELTA_E_8_BIT_CHART: f64 = 1.2;

/// The press of every proof.
const PRESS: &str = "printer-cmyk-v4.icc";

/// The worst ΔE00 of `ours` against `theirs`, display values of `display`
/// in `[0, 1]`, both decoded to Lab, over the patches `skip` does not
/// mark; with the index of its patch and the mean.
fn worst(
    display: &str,
    ours: &[Vec<f64>],
    theirs: &[Vec<f64>],
    skip: &[bool],
) -> (f64, usize, f64) {
    let decode = |device: &[Vec<f64>]| {
        transicc(
            &[&format!("-i{display}"), "-o*Lab", "-t1"],
            &lines(device, 255.0),
        )
    };
    let (lab_ours, lab_theirs) = (decode(ours), decode(theirs));
    let (mut max, mut at, mut sum, mut count) = (0.0f64, 0, 0.0, 0u32);
    for (i, (a, b)) in lab_ours.iter().zip(&lab_theirs).enumerate() {
        if skip.get(i).copied().unwrap_or(false) {
            continue;
        }
        count += 1;
        let de = delta_e00(a, b);
        sum += de;
        if de > max {
            (max, at) = (de, i);
        }
    }
    (max, at, sum / f64::from(count))
}

/// `inks`, of `[0, 1]`, painted on an ink canvas of `bits`, as a proofed
/// page is, and proofed by `proof` under `intent`: the display values, of
/// `[0, 1]`.
fn proofed(proof: &SoftProof, intent: ProofIntent, inks: &[[f64; 4]], bits: u8) -> Vec<Vec<f64>> {
    let width = u32::try_from(inks.len()).unwrap();
    let canvas = if bits == 8 {
        let mut bmp = Bitmap::<Cmyk8>::new(width, 1, 1, false);
        for (px, ink) in bmp
            .row_samples_mut(0)
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(inks)
        {
            *px = ink.map(u8::from_unit);
        }
        FinishedCanvas::Cmyk8(bmp)
    } else {
        let mut bmp = Bitmap::<Cmyk16>::new(width, 1, 1, false);
        for (px, ink) in bmp
            .row_samples_mut(0)
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(inks)
        {
            *px = ink.map(u16::from_unit);
        }
        FinishedCanvas::Cmyk16(bmp)
    };
    let samples: Vec<f64> = match proof
        .convert(canvas, intent, &mut CanvasBuffers::default())
        .expect("proofs")
    {
        FinishedCanvas::Rgb8(b) => b.row_samples(0).iter().map(|&v| v.to_unit()).collect(),
        FinishedCanvas::Rgb16(b) => b.row_samples(0).iter().map(|&v| v.to_unit()).collect(),
        other => panic!("an RGB proof, not {other:?}"),
    };
    samples
        .as_chunks::<3>()
        .0
        .iter()
        .map(|px| px.to_vec())
        .collect()
}

/// The ink the press separates each of `chart`'s colours to, as an ink
/// canvas of `bits` holds it.
fn separated(
    press: &OutputProfile,
    (intent, bpc): (RenderingIntent, bool),
    chart: &[[f64; 3]],
    bits: u8,
) -> Vec<[f64; 4]> {
    let Separation::Profile(table) = press.separation(intent, bpc).expect("separates") else {
        panic!("a profile separation")
    };
    chart
        .iter()
        .map(|rgb| {
            if bits == 8 {
                table.ink(rgb.map(u8::from_unit)).map(u8::to_unit)
            } else {
                table.ink(rgb.map(u16::from_unit)).map(u16::to_unit)
            }
        })
        .collect()
}

/// Which patches of `chart` are darker than the press's black: their
/// `L*` below that of the press's darkest separation, the working-space
/// black's, both through `transicc` under relative colorimetry.  Their
/// separation is a clip whose evaluation differs between the two CMMs
/// (`audit/2026-11-15_cmyk-separation-shadow-clip.md`), so the 8-bit
/// comparison, whose rounding leaves no margin for it, omits them.
fn below_press_black(src: &str, press: &str, chart: &[[f64; 3]]) -> Vec<bool> {
    let black_ink = transicc(
        &[&format!("-i{src}"), &format!("-o{press}"), "-t1"],
        "0 0 0\n",
    );
    let black = transicc(
        &[&format!("-i{press}"), "-o*Lab", "-t1"],
        &lines(&black_ink, 1.0),
    )[0][0];
    let chart_lines: Vec<Vec<f64>> = chart.iter().map(|p| p.to_vec()).collect();
    transicc(
        &[&format!("-i{src}"), "-o*Lab", "-t1"],
        &lines(&chart_lines, 255.0),
    )
    .iter()
    .map(|lab| lab[0] < black)
    .collect()
}

/// The display profile file `display` (the working space when `None`).
fn display_file(dir: &Path, display: Option<&str>) -> String {
    display.map_or_else(
        || working_space_file(dir).to_str().unwrap().to_owned(),
        |name| fixture(name).to_str().unwrap().to_owned(),
    )
}

/// The proof of `display`'s output profile, the working space when `None`.
fn soft_proof(display: Option<&str>) -> SoftProof {
    let read = |name: &str| {
        Arc::new(OutputProfile::from_bytes(std::fs::read(fixture(name)).unwrap()).unwrap())
    };
    SoftProof::new(read(PRESS), display.map(read)).expect("a proof")
}

/// The two proof intents and `transicc`'s code of each.
const PROOF_INTENTS: [(&str, ProofIntent); 2] = [
    ("1", ProofIntent::RelativeColorimetric),
    ("3", ProofIntent::AbsoluteColorimetric),
];

/// Our proof of the chart on `display` within [`MAX_DELTA_E`] of
/// `transicc -p` at 16 bits and [`MAX_DELTA_E_8_BIT_CHART`] at 8, the 8-bit
/// chart without the patches darker than the press black
/// ([`below_press_black`]), for every first-leg intent with and without
/// compensation and both proof intents.
fn chart_agrees(display: Option<&str>) {
    let dir = tempfile_dir("proof");
    let src = working_space_file(&dir);
    let src = src.to_str().unwrap();
    let dst = display_file(&dir, display);
    let press_file = fixture(PRESS);
    let press_file = press_file.to_str().unwrap();
    let proof = soft_proof(display);
    let chart = patches();
    let below_black = below_press_black(src, press_file, &chart);
    let mut failures = Vec::new();
    let mut overall = 0.0f64;
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
            for (proof_code, proof_intent) in PROOF_INTENTS {
                for (bits, full) in [(8u8, 255.0), (16, 65535.0)] {
                    let mut args = vec![
                        format!("-i{src}"),
                        format!("-o{dst}"),
                        format!("-p{press_file}"),
                        format!("-t{code}"),
                        format!("-m{proof_code}"),
                    ];
                    if bpc {
                        args.push("-b".to_owned());
                    }
                    let args: Vec<&str> = args.iter().map(String::as_str).collect();
                    let chart_lines: Vec<Vec<f64>> = chart
                        .iter()
                        .map(|p| p.iter().map(|v| (v * full).round() / full).collect())
                        .collect();
                    let theirs: Vec<Vec<f64>> = transicc(&args, &lines(&chart_lines, 255.0))
                        .iter()
                        .map(|v| v.iter().map(|c| c / 255.0).collect())
                        .collect();
                    let inks = separated(proof.press(), (intent, bpc), &chart, bits);
                    let ours = proofed(&proof, proof_intent, &inks, bits);
                    let skip = if bits == 8 {
                        below_black.as_slice()
                    } else {
                        &[]
                    };
                    let (max, at, mean) = worst(&dst, &ours, &theirs, skip);
                    let what = format!(
                        "{} {intent} bpc {bpc} proof {proof_intent} {bits}-bit",
                        display.unwrap_or("working space")
                    );
                    eprintln!(
                        "{what}: ΔE00 max {max:.3} (patch {:?}), mean {mean:.3}",
                        chart[at]
                    );
                    let limit = if bits == 8 {
                        MAX_DELTA_E_8_BIT_CHART
                    } else {
                        MAX_DELTA_E
                    };
                    if max > limit {
                        failures.push(format!(
                            "{what}: ΔE00 {max:.3} at patch {:?}: ours {:?}, transicc {:?}",
                            chart[at], ours[at], theirs[at]
                        ));
                    }
                    overall = overall.max(max);
                }
            }
        }
    }
    eprintln!("worst ΔE00 {overall:.3}");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Ink given as such, a 6-level grid of the ink cube, proofed on
/// `display` within [`MAX_DELTA_E`] of `transicc` from the press.
fn ink_agrees(display: Option<&str>) {
    let dir = tempfile_dir("proof-ink");
    let dst = display_file(&dir, display);
    let press_file = fixture(PRESS);
    let press_file = press_file.to_str().unwrap();
    let proof = soft_proof(display);
    let mut inks = Vec::new();
    for c in 0..6 {
        for m in 0..6 {
            for y in 0..6 {
                for k in 0..6 {
                    inks.push([c, m, y, k].map(|v: i32| f64::from(v) / 5.0));
                }
            }
        }
    }
    let mut failures = Vec::new();
    for (code, intent) in PROOF_INTENTS {
        for (bits, full) in [(8u8, 255.0), (16, 65535.0)] {
            let ink_lines: Vec<Vec<f64>> = inks
                .iter()
                .map(|p| p.iter().map(|v| (v * full).round() / full).collect())
                .collect();
            let theirs: Vec<Vec<f64>> = transicc(
                &[
                    &format!("-i{press_file}"),
                    &format!("-o{dst}"),
                    &format!("-t{code}"),
                ],
                &lines(&ink_lines, 100.0),
            )
            .iter()
            .map(|v| v.iter().map(|c| c / 255.0).collect())
            .collect();
            let ours = proofed(&proof, intent, &inks, bits);
            let (max, at, mean) = worst(&dst, &ours, &theirs, &[]);
            let what = format!(
                "ink on {} proof {intent} {bits}-bit",
                display.unwrap_or("working space")
            );
            eprintln!(
                "{what}: ΔE00 max {max:.3} (ink {:?}), mean {mean:.3}",
                inks[at]
            );
            if max > MAX_DELTA_E {
                failures.push(format!(
                    "{what}: ΔE00 {max:.3} at ink {:?}: ours {:?}, transicc {:?}",
                    inks[at], ours[at], theirs[at]
                ));
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
#[ignore = "needs transicc (liblcms2-utils)"]
fn the_proof_on_the_working_space_agrees_with_transicc() {
    chart_agrees(None);
}

#[test]
#[ignore = "needs transicc (liblcms2-utils)"]
fn the_proof_on_an_rgb_display_profile_agrees_with_transicc() {
    chart_agrees(Some("printer-rgb-v4.icc"));
}

#[test]
#[ignore = "needs transicc (liblcms2-utils)"]
fn ink_proofs_on_the_working_space_agree_with_transicc() {
    ink_agrees(None);
}

#[test]
#[ignore = "needs transicc (liblcms2-utils)"]
fn ink_proofs_on_an_rgb_display_profile_agree_with_transicc() {
    ink_agrees(Some("printer-rgb-v4.icc"));
}
