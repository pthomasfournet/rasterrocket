//! The conversion to an output profile against an independent CMM: Little
//! CMS's `transicc` converts the same patch chart from the same working
//! space profile, and both results, decoded to Lab through `transicc`, lie
//! within ΔE00 1 of each other on every patch, for every intent with and
//! without black-point compensation: from sRGB at 8 and 16 bits, from the
//! wide working space at 16 bits.
//!
//! The conversion of ICC CMYK content to another CMYK output profile is
//! measured the same way: an image of a chart of inks in one test
//! printer's profile, painted on a CMYK page separated through the other,
//! prints within ΔE00 1 of the ink `transicc` converts the chart to, both
//! printed through the output profile and decoded to Lab; under every
//! intent, with and without black-point compensation, in both directions
//! between a printer replacing the whole gray component and one replacing
//! half of it.  A compensated perceptual or saturation conversion to the
//! version 2 printer is held to [`MAX_DELTA_E_V2_COMPENSATED`]: its black,
//! found on the round trip of a ramp, is neutral here and keeps the `a*`
//! and `b*` of the initial black in Little CMS, which moves the darkest
//! colours apart.
//!
//! Needs `transicc` (Debian/Ubuntu `liblcms2-utils`):
//! `cargo test -p rasterrocket-interp --test output_profile_transicc -- --ignored`.

mod common;

use std::sync::Arc;

use color::{Rgb8, Rgb16, Sample, SampleDepth};
use common::{
    MAX_DELTA_E, delta_e00, fixture, lines, patches, tempfile_dir, transicc, working_space_file,
};
use raster::Bitmap;
use rasterrocket_interp::renderer::{CanvasBuffers, FinishedCanvas, PageRenderer, PageView};
use rasterrocket_interp::{CanvasRequest, OutputProfile, RenderingIntent, WorkingSpace};

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

/// Our conversion of `patches` of `space` at `bits` to `profile` under
/// `intent`.
fn ours(
    profile: &OutputProfile,
    space: WorkingSpace,
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
        .convert(canvas, space, (intent, bpc), &mut buffers)
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
    for space in WorkingSpace::ALL {
        agrees_from(name, space);
    }
}

/// Both conversions of the chart of `space` to `name` within
/// [`MAX_DELTA_E`].
fn agrees_from(name: &str, space: WorkingSpace) {
    let dir = tempfile_dir("output-profile");
    let src = working_space_file(&dir, space);
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
                if bits == 8 && space == WorkingSpace::Wide {
                    continue;
                }
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
                let mine = ours(&profile, space, (intent, bpc), &chart, bits);
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
                    "{name} {space:?} {intent} bpc {bpc} {bits}-bit: ΔE00 max {max:.3} (patch {:?}), mean {mean:.3}",
                    chart[at]
                );
                if max > MAX_DELTA_E {
                    failures.push(format!(
                        "{name} {space:?} {intent} bpc {bpc} {bits}-bit: ΔE00 {max:.3} at patch {:?}: \
                         ours {:?}, transicc {:?}",
                        chart[at], mine[at], theirs[at]
                    ));
                }
                worst = worst.max(max);
            }
        }
    }
    eprintln!("{name} {space:?}: worst ΔE00 {worst:.3}");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The chart of sRGB colours, converted to the wide space by `transicc`,
/// then by us from the wide space to `name` at 16 bits, within
/// [`MAX_DELTA_E`] of `transicc`'s conversion of the sRGB colours to
/// `name`, under relative colorimetry: colours inside sRGB print through
/// the wide space as they print from sRGB.
fn srgb_colours_agree_through_the_wide_space(name: &str) {
    let dir = tempfile_dir("output-profile-wide");
    let srgb = working_space_file(&dir, WorkingSpace::Srgb);
    let wide = working_space_file(&dir, WorkingSpace::Wide);
    let dst = fixture(name);
    let profile = OutputProfile::from_bytes(std::fs::read(&dst).unwrap()).unwrap();
    let (srgb, wide, dst) = (
        srgb.to_str().unwrap(),
        wide.to_str().unwrap(),
        dst.to_str().unwrap(),
    );
    let chart: Vec<Vec<f64>> = patches().iter().map(|p| p.to_vec()).collect();
    let in_wide: Vec<[f64; 3]> = transicc(
        &[&format!("-i{srgb}"), &format!("-o{wide}"), "-t1"],
        &lines(&chart, 255.0),
    )
    .iter()
    .map(|v| [v[0], v[1], v[2]].map(|c| (c / 255.0).clamp(0.0, 1.0)))
    .collect();
    let mine = ours(
        &profile,
        WorkingSpace::Wide,
        (RenderingIntent::RelativeColorimetric, false),
        &in_wide,
        16,
    );
    let theirs: Vec<Vec<f64>> = transicc(
        &[&format!("-i{srgb}"), &format!("-o{dst}"), "-t1"],
        &lines(&chart, 255.0),
    )
    .iter()
    .map(|v| v.iter().map(|c| c / 255.0).collect())
    .collect();
    let decode = |device: &[Vec<f64>]| {
        transicc(
            &[&format!("-i{dst}"), "-o*Lab", "-t1"],
            &lines(device, 255.0),
        )
    };
    let (lab_theirs, lab_mine) = (decode(&theirs), decode(&mine));
    let (worst, at) = lab_theirs
        .iter()
        .zip(&lab_mine)
        .map(|(a, b)| delta_e00(a, b))
        .enumerate()
        .fold(
            (0.0f64, 0),
            |(m, at), (i, d)| if d > m { (d, i) } else { (m, at) },
        );
    eprintln!(
        "{name}: sRGB colours through the wide space, worst ΔE00 {worst:.3} at {:?}",
        chart[at]
    );
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        worst <= MAX_DELTA_E,
        "{name}: ΔE00 {worst:.3} at {:?}",
        chart[at]
    );
}

#[test]
#[ignore = "needs transicc (liblcms2-utils)"]
fn srgb_colours_print_through_the_wide_space_as_from_srgb() {
    for name in [
        "printer-rgb-v4.icc",
        "printer-rgb-v2.icc",
        "printer-gray-v4.icc",
    ] {
        srgb_colours_agree_through_the_wide_space(name);
    }
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

/// The separation of the chart and of a dark neutral ramp through the CMYK
/// printer prints, at 8 and 16 bits, within [`MAX_DELTA_E`] of the ink
/// `transicc` separates it to, from either working space under every
/// intent with and without compensation; also below the printer's black,
/// where its relative colorimetric table moves the inks by percents for a
/// few tenths of `a*` and inks of one colour differ widely.  Both inks are
/// decoded to Lab through `transicc`.
#[test]
#[ignore = "needs transicc (liblcms2-utils)"]
fn a_cmyk_separation_prints_as_transicc_s() {
    let dir = tempfile_dir("separation");
    let dst = fixture("printer-cmyk-v4.icc");
    let profile = OutputProfile::from_bytes(std::fs::read(&dst).unwrap()).unwrap();
    let dst = dst.to_str().unwrap();
    let mut chart = patches();
    chart.extend((0..=40).map(|i| [f64::from(i) / 255.0; 3]));
    let rows: Vec<Vec<f64>> = chart.iter().map(|p| p.to_vec()).collect();
    let decode =
        |ink: &[Vec<f64>]| transicc(&[&format!("-i{dst}"), "-o*Lab", "-t1"], &lines(ink, 1.0));
    let mut failures = Vec::new();
    for space in WorkingSpace::ALL {
        let src = working_space_file(&dir, space);
        let src = src.to_str().unwrap();
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
                let table = profile
                    .separation_table(space, intent, bpc)
                    .expect("separates");
                let mut args = vec![format!("-i{src}"), format!("-o{dst}"), format!("-t{code}")];
                if bpc {
                    args.push("-b".to_owned());
                }
                let args: Vec<&str> = args.iter().map(String::as_str).collect();
                let theirs = decode(&transicc(&args, &lines(&rows, 255.0)));
                for bits in [8u8, 16] {
                    let ours: Vec<Vec<f64>> = chart
                        .iter()
                        .map(|rgb| {
                            if bits == 8 {
                                table
                                    .ink(rgb.map(u8::from_unit))
                                    .map(|v| f64::from(v) / 2.55)
                                    .to_vec()
                            } else {
                                table
                                    .ink(rgb.map(u16::from_unit))
                                    .map(|v| f64::from(v) / 655.35)
                                    .to_vec()
                            }
                        })
                        .collect();
                    let ours = decode(&ours);
                    let (worst, at) = ours
                        .iter()
                        .zip(&theirs)
                        .map(|(a, b)| delta_e00(a, b))
                        .enumerate()
                        .fold(
                            (0.0f64, 0),
                            |(m, at), (i, d)| if d > m { (d, i) } else { (m, at) },
                        );
                    eprintln!(
                        "{space:?} {intent} bpc {bpc} {bits}-bit: worst ΔE00 {worst:.3} at {:?}",
                        chart[at]
                    );
                    if worst > MAX_DELTA_E {
                        failures.push(format!(
                            "{space:?} {intent} bpc {bpc} {bits}-bit: ΔE00 {worst:.3} at {:?}",
                            chart[at]
                        ));
                    }
                }
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The largest ΔE00 accepted for a compensated perceptual or saturation
/// conversion to the version 2 printer.
const MAX_DELTA_E_V2_COMPENSATED: f64 = 1.6;

/// A PDF of objects `bodies`, numbered from 1, object 1 the catalog.
fn pdf(bodies: &[Vec<u8>]) -> Vec<u8> {
    let mut out = b"%PDF-1.7\n".to_vec();
    let mut offsets = Vec::new();
    for (i, body) in bodies.iter().enumerate() {
        offsets.push(out.len());
        out.extend_from_slice(format!("{} 0 obj\n", i + 1).as_bytes());
        out.extend_from_slice(body);
        out.extend_from_slice(b"\nendobj\n");
    }
    let xref = out.len();
    out.extend_from_slice(
        format!("xref\n0 {}\n0000000000 65535 f \n", bodies.len() + 1).as_bytes(),
    );
    for off in offsets {
        out.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
            bodies.len() + 1
        )
        .as_bytes(),
    );
    out
}

/// A stream object of dictionary entries `dict` and data `data`.
fn stream(dict: &str, data: &[u8]) -> Vec<u8> {
    let mut s = format!("<< {dict} /Length {} >>\nstream\n", data.len()).into_bytes();
    s.extend_from_slice(data);
    s.extend_from_slice(b"\nendstream");
    s
}

/// A five-level grid over the ink cube and a black ramp.
fn chart() -> Vec<[f64; 4]> {
    let mut out = Vec::new();
    for c in 0..5 {
        for m in 0..5 {
            for y in 0..5 {
                for k in 0..5 {
                    out.push([c, m, y, k].map(|v: i32| f64::from(v) / 4.0));
                }
            }
        }
    }
    out.extend((0..=16).map(|i| [0.0, 0.0, 0.0, f64::from(i) / 16.0]));
    out
}

/// The inks of `chart`, an image in the profile `source`, painted on a
/// 16-bit CMYK page separated through `dest` under `intent` and `bpc`.
fn painted(
    source: &[u8],
    dest: &Arc<OutputProfile>,
    (intent, bpc): (RenderingIntent, bool),
    chart: &[[f64; 4]],
) -> Vec<Vec<f64>> {
    let n = chart.len();
    let samples: Vec<u8> = chart
        .iter()
        .flat_map(|ink| {
            ink.iter()
                .flat_map(|&v| <u16 as color::Sample>::from_unit(v).to_be_bytes())
        })
        .collect();
    let doc = pdf::Document::from_bytes_owned(pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {n} 1] /Contents 4 0 R \
             /Resources << /XObject << /Im 6 0 R >> >> >>"
        )
        .into_bytes(),
        stream("", format!("q {n} 0 0 1 0 0 cm /Im Do Q").as_bytes()),
        stream("/N 4", source),
        stream(
            &format!(
                "/Type /XObject /Subtype /Image /Width {n} /Height 1 /BitsPerComponent 16 \
                 /ColorSpace [/ICCBased 5 0 R]"
            ),
            &samples,
        ),
    ]))
    .expect("the PDF parses");
    let geometry = rasterrocket_interp::page_size_pts(&doc, 1, rasterrocket_interp::PageBox::Crop)
        .expect("a page");
    let view = PageView::whole(&geometry, (1.0, 1.0));
    let separation =
        OutputProfile::separation(dest, WorkingSpace::Srgb, Some(intent), bpc).expect("separates");
    let request = CanvasRequest::Cmyk {
        overprint: false,
        separation,
    };
    let mut renderer = PageRenderer::try_with_page_group_in(
        &view,
        &doc,
        (3, 0),
        None,
        &request,
        SampleDepth::Sixteen,
        CanvasBuffers::default(),
    )
    .expect("the canvas is allocated");
    renderer.execute(&rasterrocket_interp::content::parse(
        format!("q {n} 0 0 1 0 0 cm /Im Do Q").as_bytes(),
    ));
    let FinishedCanvas::Cmyk16(page) = renderer.try_finish_keeping(false).expect("finishes").0
    else {
        panic!("a 16-bit CMYK page")
    };
    page.row_samples(0)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|px| px.iter().map(|&v| f64::from(v) / 655.35).collect())
        .collect()
}

#[test]
#[ignore = "needs transicc (liblcms2-utils)"]
fn icc_cmyk_converts_to_another_press_as_transicc_converts_it() {
    let chart = chart();
    let rows: Vec<Vec<f64>> = chart
        .iter()
        .map(|p| p.iter().map(|v| v * 100.0).collect())
        .collect();
    let mut failures = Vec::new();
    for (from, to) in [
        ("printer-cmyk-v4.icc", "printer-cmyk-v2.icc"),
        ("printer-cmyk-v2.icc", "printer-cmyk-v4.icc"),
    ] {
        let (from_file, to_file) = (fixture(from), fixture(to));
        let source = std::fs::read(&from_file).unwrap();
        let output = Arc::new(OutputProfile::from_bytes(std::fs::read(&to_file).unwrap()).unwrap());
        let (src, dst) = (from_file.to_str().unwrap(), to_file.to_str().unwrap());
        let decode =
            |ink: &[Vec<f64>]| transicc(&[&format!("-i{dst}"), "-o*Lab", "-t1"], &lines(ink, 1.0));
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
                let mut args = vec![format!("-i{src}"), format!("-o{dst}"), format!("-t{code}")];
                if bpc {
                    args.push("-b".to_owned());
                }
                let args: Vec<&str> = args.iter().map(String::as_str).collect();
                let theirs = decode(&transicc(&args, &lines(&rows, 1.0)));
                let ours = decode(&painted(&source, &output, (intent, bpc), &chart));
                let (worst, at) = ours
                    .iter()
                    .zip(&theirs)
                    .map(|(a, b)| delta_e00(a, b))
                    .enumerate()
                    .fold(
                        (0.0f64, 0),
                        |(m, at), (i, d)| if d > m { (d, i) } else { (m, at) },
                    );
                eprintln!(
                    "{from} → {to} {intent} bpc {bpc}: worst ΔE00 {worst:.3} at {:?}",
                    chart[at]
                );
                let limit = if bpc
                    && to.ends_with("v2.icc")
                    && matches!(
                        intent,
                        RenderingIntent::Perceptual | RenderingIntent::Saturation
                    ) {
                    MAX_DELTA_E_V2_COMPENSATED
                } else {
                    MAX_DELTA_E
                };
                if worst > limit {
                    failures.push(format!(
                        "{from} → {to} {intent} bpc {bpc}: ΔE00 {worst:.3} at {:?}",
                        chart[at]
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
