//! Camera-capture cleanup on synthetic captures of real pages.
//!
//! Pages of the corpus fixtures are rendered, then photographed in
//! perspective over a desk, curled toward a spine, or set side by side as
//! a spread (see `synth`).  The tests assert that the cleanup stages
//! recover the known geometry, and that flatbed-style pages pass through
//! byte-identical when every stage is enabled.
//!
//! The fixtures are gitignored; each test skips with a note when absent.
//! Set `CAMERA_EVAL_DIR` to also write the synthetic captures and their
//! corrected versions there as PGM files.

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::suboptimal_flops,
    clippy::many_single_char_names,
    reason = "test image arithmetic on dimensions far below 2^31; synthetic geometry \
              is written as plain formulas for review against the camera model"
)]

#[expect(
    dead_code,
    reason = "shared helpers; this test uses PDF assembly and fixtures"
)]
#[path = "../common/mod.rs"]
mod common;
mod metrics;
mod synth;

use common::{assemble_pdf, corpus_fixture, stream_object};
use rasterrocket::camera::{self, CameraOptions};
use rasterrocket::{RasterOptions, RenderedPage, SpreadHalf, raster_pdf, raster_pdf_from_bytes};
use synth::{Gray, Pose};

/// Resolution of the rendered source pages.
const DPI: f32 = 100.0;

/// Render `pages` of a corpus fixture to grayscale, or `None` when absent.
fn fixture_pages(name: &str, pages: &[u32], opts: &RasterOptions) -> Option<Vec<RenderedPage>> {
    let path = corpus_fixture(name)?;
    let set = rasterrocket::PageSet::new(pages.iter().copied()).expect("page set");
    let opts = RasterOptions {
        dpi: DPI,
        pages: Some(set),
        ..opts.clone()
    };
    Some(
        raster_pdf(&path, &opts)
            .map(|(n, r)| r.unwrap_or_else(|e| panic!("{name} page {n}: {e}")))
            .collect(),
    )
}

fn page_gray(p: &RenderedPage) -> Gray {
    Gray {
        w: p.width as usize,
        h: p.height as usize,
        px: p.pixels.clone(),
    }
}

/// One-page PDF showing `img` as a `DeviceGray` image filling the page, sized
/// so that rendering at [`DPI`] reproduces it pixel for pixel.
fn image_pdf(img: &Gray) -> Vec<u8> {
    let (wpt, hpt) = (
        img.w as f64 * 72.0 / f64::from(DPI),
        img.h as f64 * 72.0 / f64::from(DPI),
    );
    let content = format!("q {wpt:.4} 0 0 {hpt:.4} 0 0 cm /Im0 Do Q");
    assemble_pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {wpt:.4} {hpt:.4}] \
             /Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>"
        )
        .into_bytes(),
        stream_object("", content.as_bytes()),
        stream_object(
            &format!(
                "/Type /XObject /Subtype /Image /Width {} /Height {} \
                 /ColorSpace /DeviceGray /BitsPerComponent 8",
                img.w, img.h
            ),
            &img.px,
        ),
    ])
}

/// Render a one-image PDF of `img` through the public options path.
fn render_capture(img: &Gray, camera: CameraOptions, deskew: bool) -> Vec<RenderedPage> {
    let opts = RasterOptions {
        dpi: DPI,
        deskew,
        camera,
        ..RasterOptions::default()
    };
    raster_pdf_from_bytes(image_pdf(img), &opts)
        .map(|(n, r)| r.unwrap_or_else(|e| panic!("capture page {n}: {e}")))
        .collect()
}

fn dump(name: &str, img: &Gray) {
    if let Some(dir) = std::env::var_os("CAMERA_EVAL_DIR") {
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).expect("create CAMERA_EVAL_DIR");
        img.write_pgm(&dir.join(format!("{name}.pgm")));
    }
}

fn rectify_only() -> CameraOptions {
    let mut c = CameraOptions::default();
    c.rectify = true;
    c
}

fn dewarp_only() -> CameraOptions {
    let mut c = CameraOptions::default();
    c.dewarp = true;
    c
}

const POSES: [Pose; 4] = [
    Pose {
        tilt_x: 0.20,
        tilt_y: 0.0,
        roll: 0.0,
        canvas_scale: 1.25,
        shift: (0.0, 0.0),
    },
    Pose {
        tilt_x: -0.12,
        tilt_y: 0.18,
        roll: 0.05,
        canvas_scale: 1.3,
        shift: (25.0, -15.0),
    },
    Pose {
        tilt_x: 0.30,
        tilt_y: -0.10,
        roll: -0.08,
        canvas_scale: 1.2,
        shift: (-10.0, 20.0),
    },
    Pose {
        tilt_x: 0.05,
        tilt_y: 0.05,
        roll: 0.02,
        canvas_scale: 1.15,
        shift: (0.0, 0.0),
    },
];

#[test]
fn perspective_capture_is_rectified() {
    let Some(pages) = fixture_pages(
        "corpus-05-academic-book.pdf",
        &[20, 21],
        &RasterOptions::default(),
    ) else {
        return;
    };
    for (pi, page) in pages.iter().enumerate() {
        let flat = page_gray(page);
        dump(&format!("persp-{pi}-flat"), &flat);
        for (k, pose) in POSES.iter().enumerate() {
            // The last pose adds a finger over the bottom edge.
            let occluder =
                (k == 3).then_some((flat.w as f64 * 0.6, flat.h as f64 * 1.07, 60.0, 35.0));
            let cap = synth::photograph(&flat, *pose, occluder);
            dump(&format!("persp-{pi}-{k}-capture"), &cap.image);
            let diag = (flat.w as f64).hypot(flat.h as f64);

            let quad = camera::detect_page_quad(&cap.image.to_bitmap())
                .unwrap_or_else(|| panic!("page {pi} pose {k}: no quad found"));
            let err = metrics::corner_error(&quad.corners, &cap.corners);
            eprintln!(
                "page {pi} pose {k}: corner error {err:.2} px ({:.3} % of diagonal)",
                100.0 * err / diag
            );
            assert!(
                err <= 0.004 * diag,
                "page {pi} pose {k}: corner error {err:.2} px"
            );

            let out = render_capture(&cap.image, rectify_only(), false);
            assert_eq!(out.len(), 1);
            let rect = page_gray(&out[0]);
            dump(&format!("persp-{pi}-{k}-rectified"), &rect);
            let aspect = rect.w as f64 / rect.h as f64;
            let want = flat.w as f64 / flat.h as f64;
            let corr = metrics::correlation(&rect, &flat);
            eprintln!(
                "page {pi} pose {k}: aspect {aspect:.4} vs {want:.4}, size {}x{} vs {}x{}, correlation {corr:.3}",
                rect.w, rect.h, flat.w, flat.h
            );
            assert!(
                (aspect / want - 1.0).abs() < 0.02,
                "page {pi} pose {k}: aspect {aspect:.4} vs {want:.4}"
            );
            assert!(corr > 0.9, "page {pi} pose {k}: correlation {corr:.3}");
        }
    }
}

#[test]
fn flatbed_pages_pass_through_unchanged() {
    let cases: [(&str, &[u32]); 4] = [
        ("corpus-01-native-text-small.pdf", &[1, 2, 3]),
        ("corpus-02-native-vector-text.pdf", &[1, 2]),
        ("corpus-05-academic-book.pdf", &[1, 5, 20, 21]),
        ("corpus-04-ebook-mixed.pdf", &[1, 3]),
    ];
    let mut checked = 0;
    for (name, pages) in cases {
        let Some(plain) = fixture_pages(name, pages, &RasterOptions::default()) else {
            continue;
        };
        let all = RasterOptions {
            camera: CameraOptions::all(),
            ..RasterOptions::default()
        };
        let cleaned = fixture_pages(name, pages, &all).expect("fixture present");
        assert_eq!(plain.len(), cleaned.len(), "{name}: page count changed");
        for (a, b) in plain.iter().zip(&cleaned) {
            assert_eq!(b.spread_half, None, "{name} page {}: split", a.page_num);
            assert_eq!(
                (a.width, a.height),
                (b.width, b.height),
                "{name} page {}: resized",
                a.page_num
            );
            assert!(
                a.pixels == b.pixels,
                "{name} page {}: pixels changed",
                a.page_num
            );
            checked += 1;
        }
    }
    eprintln!("{checked} flatbed pages unchanged");
}

#[test]
fn curled_page_is_dewarped() {
    let Some(pages) = fixture_pages(
        "corpus-05-academic-book.pdf",
        &[20, 21, 40],
        &RasterOptions::default(),
    ) else {
        return;
    };
    for (pi, page) in pages.iter().enumerate() {
        let flat = page_gray(page);
        let depth = [0.18, 0.25, 0.32][pi] * flat.w as f64;
        let (curled, target) = synth::curl(&flat, depth);
        dump(&format!("curl-{pi}-flat"), &flat);
        dump(&format!("curl-{pi}-capture"), &curled);
        let out = render_capture(&curled, dewarp_only(), false);
        assert_eq!(out.len(), 1);
        let fixed = page_gray(&out[0]);
        dump(&format!("curl-{pi}-dewarped"), &fixed);
        let (before, before_max) = metrics::line_displacement(&curled, &target);
        let (after, after_max) = metrics::line_displacement(&fixed, &target);
        eprintln!(
            "page {pi}: line displacement rms {before:.2} px → {after:.2} px, max {before_max:.2} px → {after_max:.2} px"
        );
        assert!(
            before > 1.5,
            "page {pi}: synthetic curl too weak ({before:.2} px)"
        );
        assert!(
            after < 0.25 * before,
            "page {pi}: {before:.2} → {after:.2} px"
        );
    }
}

#[test]
fn spread_capture_is_rectified_and_split() {
    let Some(pages) = fixture_pages(
        "corpus-05-academic-book.pdf",
        &[20, 21],
        &RasterOptions::default(),
    ) else {
        return;
    };
    let (left, right) = (page_gray(&pages[0]), page_gray(&pages[1]));
    let spread = synth::spread(&left, &right);
    let pose = Pose {
        tilt_x: 0.15,
        tilt_y: -0.06,
        roll: 0.03,
        canvas_scale: 1.2,
        shift: (0.0, 0.0),
    };
    let cap = synth::photograph(&spread, pose, None);
    dump("spread-capture", &cap.image);
    let out = render_capture(&cap.image, CameraOptions::all(), true);
    assert_eq!(out.len(), 2, "spread must split into two pages");
    assert_eq!(out[0].spread_half, Some(SpreadHalf::Left));
    assert_eq!(out[1].spread_half, Some(SpreadHalf::Right));
    assert!(out.iter().all(|p| p.page_num == 1));
    let total = f64::from(out[0].width + out[1].width);
    let split = f64::from(out[0].width) / total;
    eprintln!(
        "split at {:.4} of the rectified width ({} + {})",
        split, out[0].width, out[1].width
    );
    assert!((split - 0.5).abs() < 0.01, "split at {split:.4}");
    for (half, want) in out.iter().zip([&left, &right]) {
        let got = page_gray(half);
        dump(&format!("spread-{:?}", half.spread_half), &got);
        let corr = metrics::correlation(&got, want);
        eprintln!("{:?}: correlation {corr:.3}", half.spread_half);
        assert!(corr > 0.85, "{:?}: correlation {corr:.3}", half.spread_half);
    }
}

#[test]
fn flat_spread_without_surround_is_split() {
    let Some(pages) = fixture_pages(
        "corpus-05-academic-book.pdf",
        &[20, 21],
        &RasterOptions::default(),
    ) else {
        return;
    };
    let (left, right) = (page_gray(&pages[0]), page_gray(&pages[1]));
    let spread = synth::spread(&left, &right);
    let mut split = CameraOptions::default();
    split.split_spreads = true;
    let out = render_capture(&spread, split, false);
    assert_eq!(out.len(), 2);
    assert!(
        (i64::from(out[0].width) - left.w as i64).abs() <= 3,
        "split at {} vs gutter {}",
        out[0].width,
        left.w
    );

    // The channel delivers both halves, in order, under the source page
    // number; a one-page document then closes the channel.
    let path = std::env::temp_dir().join(format!("camera-spread-{}.pdf", std::process::id()));
    std::fs::write(&path, image_pdf(&spread)).expect("write spread pdf");
    let opts = RasterOptions {
        dpi: DPI,
        camera: split,
        ..RasterOptions::default()
    };
    let items: Vec<_> = rasterrocket::render_channel(&path, &opts, 1)
        .into_iter()
        .collect();
    std::fs::remove_file(&path).expect("remove spread pdf");
    let halves: Vec<_> = items
        .iter()
        .map(|(n, r)| (*n, r.as_ref().expect("page renders").spread_half))
        .collect();
    assert_eq!(
        halves,
        [(1, Some(SpreadHalf::Left)), (1, Some(SpreadHalf::Right))]
    );
}

#[test]
fn curled_spread_is_split_and_each_page_dewarped() {
    let Some(pages) = fixture_pages(
        "corpus-05-academic-book.pdf",
        &[20, 21],
        &RasterOptions::default(),
    ) else {
        return;
    };
    let (left, right) = (page_gray(&pages[0]), page_gray(&pages[1]));
    // Both pages curl toward the gutter between them.
    let depth = 0.18 * left.w as f64;
    let (curled_left, target_left) = synth::curl(&left, depth);
    let (curled_right, target_right) = synth::curl(&right.mirror(), depth);
    let (curled_right, target_right) = (curled_right.mirror(), target_right.mirror());
    let spread = synth::spread(&curled_left, &curled_right);
    dump("curled-spread-capture", &spread);
    let out = render_capture(&spread, CameraOptions::all(), true);
    assert_eq!(out.len(), 2, "curled spread must split into two pages");
    let expected = [
        (&left, &curled_left, &target_left),
        (&right, &curled_right, &target_right),
    ];
    for (half, (flat, curled, target)) in out.iter().zip(expected) {
        let got = page_gray(half);
        dump(&format!("curled-spread-{:?}", half.spread_half), &got);
        assert!(
            (got.w as i64 - flat.w as i64).abs() <= 3,
            "{:?}: width {} vs {}",
            half.spread_half,
            got.w,
            flat.w
        );
        let got = got.resize(flat.w, flat.h);
        let (before, before_max) = metrics::line_displacement(curled, target);
        let (after, after_max) = metrics::line_displacement(&got, target);
        eprintln!(
            "{:?}: line displacement rms {before:.2} px → {after:.2} px, max {before_max:.2} px → {after_max:.2} px",
            half.spread_half
        );
        assert!(
            after < 0.25 * before,
            "{:?}: {before:.2} → {after:.2} px",
            half.spread_half
        );
    }
}
