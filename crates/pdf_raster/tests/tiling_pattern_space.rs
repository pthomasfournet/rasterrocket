//! A tiling pattern's matrix maps pattern space to the default coordinate
//! space of the content stream that uses it (ISO 32000-1 §8.7.3.1), so the
//! CTM in effect when the pattern fills a path does not scale its tiles.

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

/// One `W` × `H` page running `content`, whose pattern `/P0` repeats a
/// 20 × 20 cell with red in its lower-left 10 × 10 quarter.
fn page(content: &str) -> Vec<u8> {
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
            "/Type /Pattern /PatternType 1 /PaintType 1 /TilingType 1 \
             /BBox [0 0 20 20] /XStep 20 /YStep 20 /Resources << >>",
            b"1 0 0 rg 0 0 10 10 re f",
        ),
    ])
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
