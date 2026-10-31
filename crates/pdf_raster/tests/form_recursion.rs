//! A form `XObject` that invokes itself, directly or through another form,
//! is not executed again while it is running: the recursive `Do` draws
//! nothing and the rest of the page renders.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use common::{stream_object, with_page_tree};
use rasterrocket::{BackendPolicy, render_page_rgb};

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
