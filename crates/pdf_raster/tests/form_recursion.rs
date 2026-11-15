//! A form `XObject` that invokes itself, directly or through another form,
//! is not executed again while it is running: the recursive `Do` draws
//! nothing and the rest of the page renders.  A form invoked
//! `MAX_FORM_DEPTH` forms deep is not drawn either, and is reported.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use common::{stream_object, with_page_tree};
use rasterrocket::{BackendPolicy, MAX_FORM_DEPTH, RasterError, RasterOptions, render_page_rgb};

/// A 100 × 100 pt page, rendered at one pixel per point.
const H: u32 = 100;

const RED: [u8; 3] = [255, 0, 0];
const GREEN: [u8; 3] = [0, 255, 0];
const BLUE: [u8; 3] = [0, 0, 255];

/// A page painting a red bar, invoking form `/A` (object 5), and painting
/// a blue bar; `/A` and `/B` (object 6) have the given contents and both
/// see each other in their resources.
fn page(a: &[u8], b: &[u8]) -> Vec<u8> {
    let forms = "/Resources << /XObject << /A 5 0 R /B 6 0 R >> >>";
    let form = |content: &[u8]| {
        stream_object(
            &format!("/Type /XObject /Subtype /Form /BBox [0 0 100 100] {forms}"),
            content,
        )
    };
    with_page_tree(&[
        format!("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 {H}] {forms} /Contents 4 0 R >>")
            .into_bytes(),
        stream_object(
            "",
            b"1 0 0 rg 0 0 20 100 re f /A Do 0 0 1 rg 80 0 20 100 re f",
        ),
        form(a),
        form(b),
    ])
}

/// RGB at the centres of the bars at x 0..20, 40..60 and 80..100.
fn bars(pdf: Vec<u8>) -> Vec<[u8; 3]> {
    let s = common::session(pdf, BackendPolicy::CpuOnly);
    let bmp = render_page_rgb(&s, 1, 1.0).expect("the page renders");
    let data = bmp.data();
    [10_usize, 50, 90]
        .iter()
        .map(|&x| {
            let i = (H as usize / 2) * bmp.stride + x * 3;
            [data[i], data[i + 1], data[i + 2]]
        })
        .collect()
}

#[test]
fn a_form_invoking_itself_renders_once() {
    let px = bars(page(b"0 1 0 rg 40 0 20 100 re f /A Do /A Do", b""));
    assert_eq!(px, [RED, GREEN, BLUE]);
}

#[test]
fn mutually_recursive_forms_render_once_each() {
    let px = bars(page(b"/B Do", b"0 1 0 rg 40 0 20 100 re f /A Do"));
    assert_eq!(px, [RED, GREEN, BLUE]);
}

#[test]
fn a_form_invoked_twice_in_sequence_renders_twice() {
    // Not a cycle: /B is not running when /A invokes it the second time.
    let px = bars(page(b"/B Do 0 0 1 rg /B Do", b"0 1 0 rg 40 0 20 100 re f"));
    assert_eq!(px, [RED, GREEN, BLUE]);
}

/// A page painting a red bar and invoking `/F`, the first of
/// `MAX_FORM_DEPTH + 1` nested forms (objects 5 onwards), then painting a
/// blue bar.  The form invoked `MAX_FORM_DEPTH - 1` forms deep paints the
/// green bar after its `Do`; the one below it paints over the red bar.
fn nested(forms: u32) -> Vec<u8> {
    let mut objects = vec![
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
           /Resources << /XObject << /F 5 0 R >> >> /Contents 4 0 R >>"
            .to_vec(),
        stream_object(
            "",
            b"1 0 0 rg 0 0 20 100 re f /F Do 0 0 1 rg 80 0 20 100 re f",
        ),
    ];
    for k in 0..forms {
        let resources = format!("/Resources << /XObject << /F {} 0 R >> >>", 6 + k);
        let (resources, content): (&str, &[u8]) = match forms - k {
            1 => ("", b"0 0 0 rg 0 0 20 100 re f"),
            2 => (&resources, b"/F Do 0 1 0 rg 40 0 20 100 re f"),
            _ => (&resources, b"/F Do"),
        };
        objects.push(stream_object(
            &format!("/Type /XObject /Subtype /Form /BBox [0 0 100 100] {resources}"),
            content,
        ));
    }
    with_page_tree(&objects)
}

#[test]
fn a_form_too_deep_is_skipped_and_reported() {
    let pdf = nested(MAX_FORM_DEPTH + 1);
    assert_eq!(bars(pdf.clone()), [RED, GREEN, BLUE]);
    let page = common::render_one(pdf, &RasterOptions::new(72.0)).expect("the page renders");
    assert_eq!(page.diagnostics.forms_too_deep, [(5 + MAX_FORM_DEPTH, 0)]);
}

#[test]
fn a_form_too_deep_fails_a_strict_page() {
    let mut opts = RasterOptions::new(72.0);
    opts.strict = true;
    match common::render_one(nested(MAX_FORM_DEPTH + 1), &opts) {
        Err(RasterError::PageBudgetExceeded(msg)) => {
            assert!(
                msg.contains(&format!("more than {MAX_FORM_DEPTH} forms deep")),
                "{msg}"
            );
        }
        other => panic!("expected PageBudgetExceeded, got {other:?}"),
    }
}
