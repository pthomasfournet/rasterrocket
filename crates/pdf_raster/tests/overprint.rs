//! Overprint simulation switch: `SessionConfig::overprint` and
//! `RasterOptions::overprint`.  A black `DeviceCMYK` fill under nonzero
//! overprint mode leaves the cyan beneath it when the switch is on and
//! knocks it out when the switch is off.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use color::cmyk::cmyk_to_rgb;
use common::{render_one, stream_object, with_page_tree};
use rasterrocket::{
    BackendPolicy, RasterOptions, RenderedPage, SessionConfig, open_session_from_bytes,
    render_page_rgb,
};

/// A 10 pt square page: solid cyan, then a black overprinting fill over
/// its left half.
fn page() -> Vec<u8> {
    with_page_tree(&[
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 10 10] \
          /Resources << /ExtGState << /GS << /OP true /OPM 1 >> >> >> \
          /Contents 4 0 R >>"
            .to_vec(),
        stream_object(
            "",
            b"1 0 0 0 k 0 0 10 10 re f /GS gs 0 0 0 1 k 0 0 5 10 re f",
        ),
    ])
}

/// RGB at the left-half and right-half probes of the page rendered with
/// the overprint switch `on`, at one pixel per point.
fn probes(on: bool) -> ([u8; 3], [u8; 3]) {
    let mut config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    config.overprint = on;
    let s = open_session_from_bytes(page(), &config).expect("open session");
    let bmp = render_page_rgb(&s, 1, 1.0).expect("render page");
    let row = bmp.row_bytes(5);
    let rgb = |x: usize| [row[x * 3], row[x * 3 + 1], row[x * 3 + 2]];
    (rgb(2), rgb(7))
}

#[test]
fn overprint_changes_only_when_enabled() {
    let black = cmyk_to_rgb(0, 0, 0, 255);
    let cyan = cmyk_to_rgb(255, 0, 0, 0);
    let cyan_black = cmyk_to_rgb(255, 0, 0, 255);
    assert_ne!(black, cyan_black, "the two composites are distinguishable");

    assert_eq!(probes(false), (black, cyan), "off: black knocks out cyan");
    assert_eq!(probes(true), (cyan_black, cyan), "on: cyan survives");
}

/// Gray value at row `row`, column `x` of `page`.
fn px(page: &RenderedPage, row: u32, x: u32) -> u8 {
    page.pixels[(row * page.width + x) as usize]
}

#[test]
fn overprint_defaults_to_off() {
    assert!(!RasterOptions::default().overprint);
}

#[test]
fn raster_options_overprint_reaches_the_renderer() {
    let default = render_one(page(), &RasterOptions::new(72.0)).expect("default render");
    let mut off = RasterOptions::new(72.0);
    off.overprint = false;
    let off = render_one(page(), &off).expect("off render");
    assert_eq!(off.pixels, default.pixels, "off equals the default");

    let mut on = RasterOptions::new(72.0);
    on.overprint = true;
    let on = render_one(page(), &on).expect("on render");
    assert_eq!((on.width, on.height), (default.width, default.height));
    assert!(
        px(&on, 5, 2) < px(&default, 5, 2),
        "cyan under the black darkens the left half"
    );
    assert_eq!(px(&on, 5, 7), px(&default, 5, 7), "right half unchanged");
}

/// A 10 pt square page filled with an RGB colour, whose resources hold
/// `resources` and whose only annotation's appearance sets overprint.
fn annotated_page(resources: &str) -> Vec<u8> {
    with_page_tree(&[
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 10 10] \
             /Resources << {resources} >> /Contents 4 0 R \
             /Annots [<< /Type /Annot /Subtype /Square /Rect [0 0 1 1] /AP << /N 5 0 R >> >>] >>"
        )
        .into_bytes(),
        stream_object("", b"0.3 0.6 0.9 rg 0 0 10 10 re f"),
        stream_object(
            "/Type /XObject /Subtype /Form /BBox [0 0 1 1] \
             /Resources << /ExtGState << /GS << /OP true >> >> >>",
            b"/GS gs",
        ),
    ])
}

/// The RGB page content of `pdf` rendered with overprint `on` and
/// annotations drawn when `annotations`.
fn render_with(pdf: Vec<u8>, on: bool, annotations: bool) -> Vec<u8> {
    let mut config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    config.overprint = on;
    config.annotations = annotations;
    let s = open_session_from_bytes(pdf, &config).expect("open session");
    render_page_rgb(&s, 1, 1.0)
        .expect("render page")
        .data()
        .to_vec()
}

/// A page whose overprint is set only in an annotation that is not drawn
/// renders as with overprint off.
#[test]
fn overprint_in_a_hidden_annotation_leaves_the_page_alone() {
    let off = render_with(annotated_page(""), false, false);
    // The page content alone renders differently on the ink canvas.
    let unused_op = "/ExtGState << /U << /OP true >> >>";
    assert_ne!(render_with(annotated_page(unused_op), true, false), off);
    assert_eq!(render_with(annotated_page(""), true, false), off);
}
