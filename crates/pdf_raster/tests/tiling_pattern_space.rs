//! A tiling pattern's matrix maps pattern space to the default coordinate
//! space of the content stream that uses it (ISO 32000-1 §8.7.3.1), so the
//! CTM in effect when the pattern fills a path does not scale its tiles;
//! and its cells repeat on the lattice that matrix spans, the images of
//! `(col · XStep, row · YStep)`, whatever its rotation, skew or step.

#[expect(
    dead_code,
    reason = "shared helpers; this test uses PDF assembly and sessions"
)]
mod common;

use common::{assemble_pdf, session, stream_object};
use rasterrocket::{BackendPolicy, render_page_rgb};

/// Page size in points; rendered at one pixel per point.
const W: u32 = 200;
const H: u32 = 100;

/// One `W` × `H` page running `content`, whose pattern `/P0` is the
/// stream `pattern` (its dictionary entries) painting `cell`.
fn page_with(content: &str, pattern: &str, cell: &[u8]) -> Vec<u8> {
    assemble_pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W} {H}] \
             /Resources << /Pattern << /P0 5 0 R >> >> /Contents 4 0 R >>"
        )
        .into_bytes(),
        stream_object("", content.as_bytes()),
        stream_object(
            &format!(
                "/Type /Pattern /PatternType 1 /PaintType 1 /TilingType 1 {pattern} /Resources << >>"
            ),
            cell,
        ),
    ])
}

/// One `W` × `H` page running `content`, whose pattern `/P0` repeats a
/// 20 × 20 cell with red in its lower-left 10 × 10 quarter.
fn page(content: &str) -> Vec<u8> {
    page_with(
        content,
        "/BBox [0 0 20 20] /XStep 20 /YStep 20",
        b"1 0 0 rg 0 0 10 10 re f",
    )
}

/// RGB at user-space `(x, y)` (origin bottom-left).
fn rgb_at(pdf: Vec<u8>, points: &[(u32, u32)]) -> Vec<[u8; 3]> {
    let s = session(pdf, BackendPolicy::CpuOnly);
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

const WHITE: [u8; 3] = [255, 255, 255];
const RED: [u8; 3] = [255, 0, 0];

/// Whether every channel is below 128.
fn dark(rgb: [u8; 3]) -> bool {
    rgb.iter().all(|&c| c < 128)
}

/// A 20 × 20 cell with a black bar across its middle, `y` 8 to 12.
const BAR_CELL: &[u8] = b"0 g 0 8 20 4 re f";

#[test]
fn a_rotated_pattern_paints_continuous_stripes() {
    // Rotated 45°, the bar's centre lines are y = x + 10√2 + 20√2·k.
    let (c, s) = (
        std::f64::consts::FRAC_1_SQRT_2,
        std::f64::consts::FRAC_1_SQRT_2,
    );
    let pdf = page_with(
        "/Pattern cs /P0 scn 0 0 200 100 re f",
        &format!(
            "/BBox [0 0 20 20] /XStep 20 /YStep 20 /Matrix [{c} {s} {} {c} 0 0]",
            -s
        ),
        BAR_CELL,
    );
    let on: Vec<(u32, u32)> = (10..80).map(|x| (x, x + 14)).collect();
    let off: Vec<(u32, u32)> = (10..70).map(|x| (x, x + 28)).collect();
    let px = rgb_at(pdf, &[on.as_slice(), off.as_slice()].concat());
    for (i, &p) in px[..on.len()].iter().enumerate() {
        assert!(dark(p), "stripe is broken at {:?}: {p:?}", on[i]);
    }
    for (i, &p) in px[on.len()..].iter().enumerate() {
        assert_eq!(p, WHITE, "between stripes at {:?}", off[i]);
    }
}

#[test]
fn a_skewed_pattern_paints_continuous_stripes() {
    // Under [1 0 0.5 1], the bar stays horizontal and its cells shift by
    // 10 per row: the stripes are unbroken across the page.
    let pdf = page_with(
        "/Pattern cs /P0 scn 0 0 200 100 re f",
        "/BBox [0 0 20 20] /XStep 20 /YStep 20 /Matrix [1 0 0.5 1 0 0]",
        BAR_CELL,
    );
    let on: Vec<(u32, u32)> = (0..200).map(|x| (x, 10)).collect();
    let px = rgb_at(pdf, &on);
    for (i, &p) in px.iter().enumerate() {
        assert!(dark(p), "stripe is broken at {:?}: {p:?}", on[i]);
    }
}

#[test]
fn a_fractional_step_does_not_drift() {
    // A 3 × 3 dot every 7.3 pt: the fifth dot is at 36.5 → 37, the tenth
    // at 73; an 8 px period would put them at 40 and 80.
    let pdf = page_with(
        "/Pattern cs /P0 scn 0 0 200 100 re f",
        "/BBox [0 0 7.3 7.3] /XStep 7.3 /YStep 7.3",
        b"0 g 0 0 3 3 re f",
    );
    let px = rgb_at(pdf, &[(1, 1), (38, 1), (41, 1), (74, 1), (78, 1)]);
    assert!(dark(px[0]), "first dot: {:?}", px[0]);
    assert!(dark(px[1]), "fifth dot at 37: {:?}", px[1]);
    assert_eq!(px[2], WHITE, "no dot at 40: {:?}", px[2]);
    assert!(dark(px[3]), "tenth dot at 73: {:?}", px[3]);
    assert_eq!(px[4], WHITE, "between dots: {:?}", px[4]);
}

#[test]
fn a_cell_larger_than_the_fill_paints_in_place() {
    // A 500 pt cell, red for x from 100 to 300, under a 50 × 50 fill.
    let pdf = page_with(
        "0 0 1 rg 0 0 200 100 re f /Pattern cs /P0 scn 150 20 50 50 re f",
        "/BBox [0 0 500 500] /XStep 500 /YStep 500",
        b"1 0 0 rg 100 0 200 100 re f",
    );
    let px = rgb_at(pdf, &[(160, 30), (195, 65), (140, 30), (160, 80)]);
    assert_eq!(px[0], RED, "inside the fill and the red square");
    assert_eq!(px[1], RED, "inside the fill and the red square");
    assert_eq!(px[2], [0, 0, 255], "left of the fill");
    assert_eq!(px[3], [0, 0, 255], "above the fill");
}

#[test]
fn a_cell_bbox_off_the_origin_keeps_its_content() {
    // The cell paints [-5, 5]², repeated every 20: red around (20, 20) and
    // not around (10, 10).
    let pdf = page_with(
        "0 0 1 rg 0 0 200 100 re f /Pattern cs /P0 scn 0 0 200 100 re f",
        "/BBox [-5 -5 15 15] /XStep 20 /YStep 20",
        b"1 0 0 rg -5 -5 10 10 re f",
    );
    let px = rgb_at(pdf, &[(17, 17), (22, 22), (10, 10), (3, 3)]);
    assert_eq!(px[0], RED, "below-left of (20, 20)");
    assert_eq!(px[1], RED, "above-right of (20, 20)");
    assert_eq!(px[2], [0, 0, 255], "between cells");
    assert_eq!(px[3], RED, "the origin cell");
}

#[test]
fn tiling_pattern_cells_ignore_the_ctm_at_the_fill() {
    let px = rgb_at(
        page("0.5 0 0 0.5 0 0 cm /Pattern cs /P0 scn 0 0 400 200 re f"),
        &[(7, 7), (12, 12), (27, 7), (47, 27)],
    );
    assert_eq!(px[0], [255, 0, 0], "inside the first cell's red quarter");
    assert_ne!(px[1], [255, 0, 0], "outside the red quarter");
    assert_eq!(px[2], [255, 0, 0], "the next cell along x");
    assert_eq!(px[3], [255, 0, 0], "a cell up and along");
}
