//! A form `XObject`'s `BBox` clips its content (ISO 32000-2 §8.10.1),
//! whether or not the form is a transparency group.

#[expect(dead_code, reason = "shared helpers; this test uses only PDF assembly")]
mod common;

use common::{assemble_pdf, stream_object};
use rasterrocket::{BackendPolicy, render_page_rgb};

/// A 100 × 100 pt page, rendered at one pixel per point.
const W: u32 = 100;
const H: u32 = 100;

/// A white page that draws the form `/F` (`BBox` 20 20 60 60, `Matrix`
/// translating by 10 10) whose content fills the whole page red.
/// `group` is spliced into the form dictionary.
fn page(group: &str) -> Vec<u8> {
    assemble_pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W} {H}] \
             /Resources << /XObject << /F 5 0 R >> >> /Contents 4 0 R >>"
        )
        .into_bytes(),
        stream_object("", b"/F Do"),
        stream_object(
            &format!(
                "/Type /XObject /Subtype /Form /BBox [20 20 60 60] \
                 /Matrix [1 0 0 1 10 10] {group}"
            ),
            b"1 0 0 rg -50 -50 200 200 re f",
        ),
    ])
}

/// RGB at user-space `(x, y)` (origin bottom-left).
fn rgb_at(pdf: Vec<u8>, points: &[(u32, u32)]) -> Vec<[u8; 3]> {
    let s = common::session(pdf, BackendPolicy::CpuOnly);
    let bmp = render_page_rgb(&s, 1, 1.0).expect("render page");
    let data = bmp.data();
    points
        .iter()
        .map(|&(x, y)| {
            let i = (H - 1 - y) as usize * bmp.stride + x as usize * 3;
            [data[i], data[i + 1], data[i + 2]]
        })
        .collect()
}

const RED: [u8; 3] = [255, 0, 0];
const WHITE: [u8; 3] = [255, 255, 255];

/// The `BBox` maps to page `[30, 30, 70, 70]`: red inside, white outside.
fn assert_clipped(pdf: Vec<u8>) {
    let got = rgb_at(pdf, &[(31, 31), (68, 68), (28, 50), (50, 72), (5, 5)]);
    assert_eq!(got[..2], [RED, RED], "inside the BBox");
    assert_eq!(got[2..], [WHITE, WHITE, WHITE], "outside the BBox");
}

#[test]
fn a_plain_form_is_clipped_to_its_bbox() {
    assert_clipped(page(""));
}

#[test]
fn a_group_form_is_clipped_to_its_bbox() {
    assert_clipped(page("/Group << /S /Transparency >>"));
}
