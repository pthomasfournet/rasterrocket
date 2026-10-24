//! `rasterrocket-camera` on synthetic captures of real pages.
//!
//! Pages of the corpus fixtures are rendered, then photographed in
//! perspective over a desk, curled toward a spine, or set side by side as
//! a spread (see `synth`).  Each capture is embedded in a one-image PDF and
//! rendered back, so the camera stages see what a renderer delivers.  The
//! tests assert that the stages recover the known geometry, and that
//! flatbed-style pages pass through byte-identical when every stage is
//! enabled.
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

use camera::{CameraOptions, GrayView, Processed};
use common::{assemble_pdf, corpus_fixture, stream_object};
use rasterrocket::{RasterOptions, RenderedPage, raster_pdf, raster_pdf_from_bytes};
use synth::{Gray, Pose};

/// Resolution of the rendered source pages.
const DPI: f32 = 100.0;

/// Render `pages` of a corpus fixture to grayscale, or `None` when absent.
fn fixture_pages(name: &str, pages: &[u32]) -> Option<Vec<RenderedPage>> {
    let path = corpus_fixture(name)?;
    let set = rasterrocket::PageSet::new(pages.iter().copied()).expect("page set");
    let opts = RasterOptions {
        dpi: DPI,
        pages: Some(set),
        ..RasterOptions::default()
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

/// Render a one-image PDF of `img`.
fn render_image(img: &Gray) -> Gray {
    let opts = RasterOptions {
        dpi: DPI,
        ..RasterOptions::default()
    };
    let pages: Vec<RenderedPage> = raster_pdf_from_bytes(image_pdf(img), &opts)
        .map(|(n, r)| r.unwrap_or_else(|e| panic!("capture page {n}: {e}")))
        .collect();
    assert_eq!(pages.len(), 1, "one-image PDF renders one page");
    page_gray(&pages[0])
}

fn run_camera(img: &Gray, opts: CameraOptions) -> Processed {
    camera::process(img.view(), &opts).expect("rendered page is a valid image")
}

/// Render a one-image PDF of `img`, run the camera stages on the result,
/// then deskew each resulting page when `deskew` is set.
fn render_capture(img: &Gray, opts: CameraOptions, deskew: bool) -> Vec<Gray> {
    let out = run_camera(&render_image(img), opts);
    out.pages
        .into_iter()
        .map(|p| {
            let page = Gray::from_page(p);
            if deskew {
                let mut bmp = page.to_bitmap();
                rasterrocket::deskew::apply(&mut bmp).expect("deskew");
                Gray::from_bitmap(&bmp)
            } else {
                page
            }
        })
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
    let Some(pages) = fixture_pages("corpus-05-academic-book.pdf", &[20, 21]) else {
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

            let quad = camera::detect_page_quad(cap.image.view())
                .expect("valid image")
                .found()
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
            let rect = &out[0];
            dump(&format!("persp-{pi}-{k}-rectified"), rect);
            let aspect = rect.w as f64 / rect.h as f64;
            let want = flat.w as f64 / flat.h as f64;
            let corr = metrics::correlation(rect, &flat);
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
        let Some(plain) = fixture_pages(name, pages) else {
            continue;
        };
        for page in &plain {
            let view = GrayView::packed(&page.pixels, page.width as usize, page.height as usize);
            let out = camera::process(view, &CameraOptions::all()).expect("valid page");
            let n = page.page_num;
            assert_eq!(out.pages.len(), 1, "{name} page {n}: split");
            let got = &out.pages[0];
            assert_eq!(
                (got.width, got.height),
                (view.width, view.height),
                "{name} page {n}: resized"
            );
            assert!(got.data == page.pixels, "{name} page {n}: pixels changed");
            checked += 1;
        }
    }
    eprintln!("{checked} flatbed pages unchanged");
}

#[test]
fn curled_page_is_dewarped() {
    let Some(pages) = fixture_pages("corpus-05-academic-book.pdf", &[20, 21, 40]) else {
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
        let fixed = &out[0];
        dump(&format!("curl-{pi}-dewarped"), fixed);
        let (before, before_max) = metrics::line_displacement(&curled, &target);
        let (after, after_max) = metrics::line_displacement(fixed, &target);
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
    let Some(pages) = fixture_pages("corpus-05-academic-book.pdf", &[20, 21]) else {
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
    let total = (out[0].w + out[1].w) as f64;
    let split = out[0].w as f64 / total;
    eprintln!(
        "split at {:.4} of the rectified width ({} + {})",
        split, out[0].w, out[1].w
    );
    assert!((split - 0.5).abs() < 0.01, "split at {split:.4}");
    for ((side, got), want) in ["left", "right"].iter().zip(&out).zip([&left, &right]) {
        dump(&format!("spread-{side}"), got);
        let corr = metrics::correlation(got, want);
        eprintln!("{side}: correlation {corr:.3}");
        assert!(corr > 0.85, "{side}: correlation {corr:.3}");
    }
}

#[test]
fn square_spread_on_desk_is_cropped_and_split() {
    let Some(pages) = fixture_pages("corpus-05-academic-book.pdf", &[20, 21]) else {
        return;
    };
    let (left, right) = (page_gray(&pages[0]), page_gray(&pages[1]));
    let spread = synth::spread(&left, &right);
    // A book held square to the camera: no tilt or roll.
    let pose = Pose {
        tilt_x: 0.0,
        tilt_y: 0.0,
        roll: 0.0,
        canvas_scale: 1.2,
        shift: (14.0, -9.0),
    };
    let cap = synth::photograph(&spread, pose, None);
    dump("square-spread-capture", &cap.image);
    let mut crop_split = CameraOptions::default();
    crop_split.crop = true;
    crop_split.split_spreads = true;
    let out = run_camera(&render_image(&cap.image), crop_split);
    let report = &out.report;
    assert!(report.crop.applied(), "crop: {:?}", report.crop);
    assert!(report.split.applied(), "split: {:?}", report.split);
    let rect = report.crop_rect.expect("crop rectangle");
    let edge = cap.corners[0][0] + 0.5;
    eprintln!(
        "crop {}x{} at ({}, {}) vs page {}x{} at ({edge:.1}, {:.1})",
        rect.width,
        rect.height,
        rect.x,
        rect.y,
        spread.w,
        spread.h,
        cap.corners[0][1] + 0.5
    );
    assert!(
        (rect.x as f64 - edge).abs() <= 2.0,
        "crop left edge {}",
        rect.x
    );
    assert!(
        rect.width.abs_diff(spread.w) <= 3,
        "crop width {}",
        rect.width
    );
    let gutter = report.gutter.expect("gutter");
    let source_x = gutter.source_x.expect("cropped cut is a source column");
    let want = edge + left.w as f64;
    eprintln!("gutter at source column {source_x} vs {want:.1}");
    assert!(
        (source_x as f64 - want).abs() <= 3.0,
        "gutter at {source_x}"
    );
    for ((side, got), want) in ["left", "right"]
        .iter()
        .zip(&out.pages)
        .zip([&left, &right])
    {
        let got = Gray::from_page(got.clone());
        dump(&format!("square-spread-{side}"), &got);
        let corr = metrics::correlation(&got, want);
        eprintln!("{side}: {}x{}, correlation {corr:.3}", got.w, got.h);
        assert!(corr > 0.9, "{side}: correlation {corr:.3}");
    }
}

#[test]
fn flat_spread_without_surround_is_split() {
    let Some(pages) = fixture_pages("corpus-05-academic-book.pdf", &[20, 21]) else {
        return;
    };
    let (left, right) = (page_gray(&pages[0]), page_gray(&pages[1]));
    let spread = synth::spread(&left, &right);
    let mut split = CameraOptions::default();
    split.split_spreads = true;
    let out = run_camera(&render_image(&spread), split);
    assert_eq!(out.pages.len(), 2);
    let gutter = out.report.gutter.expect("gutter");
    eprintln!("split at {} vs gutter {}", gutter.x, left.w);
    assert!(
        gutter.x.abs_diff(left.w) <= 3,
        "split at {} vs gutter {}",
        gutter.x,
        left.w
    );
    assert_eq!(gutter.source_x, Some(gutter.x));
    assert_eq!(out.pages[0].width, gutter.x);
}

#[test]
fn curled_spread_is_split_and_each_page_dewarped() {
    let Some(pages) = fixture_pages("corpus-05-academic-book.pdf", &[20, 21]) else {
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
        ("left", &left, &curled_left, &target_left),
        ("right", &right, &curled_right, &target_right),
    ];
    for (got, (side, flat, curled, target)) in out.iter().zip(expected) {
        dump(&format!("curled-spread-{side}"), got);
        assert!(
            got.w.abs_diff(flat.w) <= 3,
            "{side}: width {} vs {}",
            got.w,
            flat.w
        );
        let got = got.resize(flat.w, flat.h);
        let (before, before_max) = metrics::line_displacement(curled, target);
        let (after, after_max) = metrics::line_displacement(&got, target);
        eprintln!(
            "{side}: line displacement rms {before:.2} px → {after:.2} px, max {before_max:.2} px → {after_max:.2} px"
        );
        assert!(after < 0.25 * before, "{side}: {before:.2} → {after:.2} px");
    }
}
