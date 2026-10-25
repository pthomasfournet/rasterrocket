//! Painting with patterns (ISO 32000-1 §8.7.3): a tiling pattern paints only
//! what its cell's content paints, leaving the backdrop elsewhere in each
//! cell, for coloured (`PaintType` 1) and uncoloured (`PaintType` 2) patterns
//! alike.

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

/// The backdrop: opaque blue over the whole page.
const BACKDROP: &str = "0 0 1 rg 0 0 200 100 re f ";
const BLUE: [u8; 3] = [0, 0, 255];
const RED: [u8; 3] = [255, 0, 0];
const GREEN: [u8; 3] = [0, 255, 0];

/// One `W` × `H` page running `content`, with pattern `/P0` (object 5,
/// `pattern`, a stream when `pattern_stream` is `Some`) and `extra` as
/// objects 6, 7, …; `resources` adds entries to the page's resources.
fn page(
    content: &str,
    pattern: &str,
    pattern_stream: Option<&[u8]>,
    resources: &str,
    extra: &[Vec<u8>],
) -> Vec<u8> {
    let pattern_obj = pattern_stream.map_or_else(
        || pattern.as_bytes().to_vec(),
        |body| stream_object(pattern, body),
    );
    let mut objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W} {H}] \
             /Resources << /Pattern << /P0 5 0 R >> {resources} >> /Contents 4 0 R >>"
        )
        .into_bytes(),
        stream_object("", content.as_bytes()),
        pattern_obj,
    ];
    objects.extend_from_slice(extra);
    assemble_pdf(&objects)
}

/// A rendered page: its row stride and pixel bytes.
struct Page {
    stride: usize,
    data: Vec<u8>,
}

/// The page rendered at one pixel per point.
fn render(pdf: Vec<u8>) -> Page {
    let s = session(pdf, BackendPolicy::CpuOnly);
    let bmp = render_page_rgb(&s, 1, 1.0).expect("render page");
    assert_eq!((bmp.width, bmp.height), (W, H));
    Page {
        stride: bmp.stride,
        data: bmp.data().to_vec(),
    }
}

/// RGB of the pixel whose lower-left corner is user-space `(x, y)`.
fn rgb(page: &Page, x: u32, y: u32) -> [u8; 3] {
    let i = (H - 1 - y) as usize * page.stride + x as usize * 3;
    [page.data[i], page.data[i + 1], page.data[i + 2]]
}

/// A 20 × 20 tiling cell painting `cell` (a content stream).
fn tiling(paint_type: u8) -> String {
    format!(
        "/Type /Pattern /PatternType 1 /PaintType {paint_type} /TilingType 1 \
         /BBox [0 0 20 20] /XStep 20 /YStep 20 /Resources << >>"
    )
}

#[test]
fn coloured_tiling_cells_leave_the_backdrop_where_unpainted() {
    let bmp = render(page(
        &format!("{BACKDROP}/Pattern cs /P0 scn 0 0 200 100 re f"),
        &tiling(1),
        Some(b"1 0 0 rg 0 0 10 10 re f"),
        "",
        &[],
    ));
    assert_eq!(rgb(&bmp, 5, 5), RED, "the painted quarter");
    assert_eq!(rgb(&bmp, 15, 5), BLUE, "right of the painted quarter");
    assert_eq!(rgb(&bmp, 5, 15), BLUE, "above the painted quarter");
    assert_eq!(rgb(&bmp, 45, 25), RED, "a later cell's painted quarter");
    assert_eq!(rgb(&bmp, 55, 35), BLUE, "a later cell's empty area");
}

#[test]
fn uncoloured_tiling_cells_paint_the_scn_colour_over_the_backdrop() {
    let bmp = render(page(
        &format!("{BACKDROP}/CsP cs 0 1 0 /P0 scn 0 0 200 100 re f"),
        &tiling(2),
        Some(b"0 0 10 10 re f"),
        "/ColorSpace << /CsP [/Pattern /DeviceRGB] >>",
        &[],
    ));
    assert_eq!(rgb(&bmp, 5, 5), GREEN, "the painted quarter takes the tint");
    assert_eq!(rgb(&bmp, 15, 15), BLUE, "the empty area keeps the backdrop");
}

#[test]
fn tiling_cell_edges_blend_with_the_backdrop() {
    // A cell content edge at half a device pixel covers that pixel partly;
    // the extra vertex keeps the outline off the whole-pixel rectangle
    // snapping.
    let bmp = render(page(
        &format!("{BACKDROP}/Pattern cs /P0 scn 0 0 200 100 re f"),
        &tiling(1),
        Some(b"1 0 0 rg 0 0 m 10.5 0 l 10.5 20 l 5 20 l 0 20 l h f"),
        "",
        &[],
    ));
    let [r, g, b] = rgb(&bmp, 10, 5);
    assert!(
        g == 0 && r > 60 && b > 25 && (240..=270).contains(&(u16::from(r) + u16::from(b))),
        "a partly covered pixel mixes red over blue: {:?}",
        [r, g, b]
    );
}
