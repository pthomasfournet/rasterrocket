//! An image `XObject` drawn more than once on a page renders each draw as
//! if it were the only one: a decode kept for reuse must not leak one
//! draw's decode resolution into another.  A JPEG 2000 image decodes at a
//! reduced resolution level for a small draw, so a small draw followed by
//! a large one (and the reverse) exercises reuse across resolutions.

#[expect(
    dead_code,
    reason = "shared helpers; this test uses PDF assembly and sessions"
)]
mod common;

use common::{assemble_pdf, session, stream_object};
use rasterrocket::{BackendPolicy, render_page_rgb};

/// 8 × 8 one-channel 8-bit JP2 with resolution levels below full size.
const GRAY8_JP2: &[u8] = include_bytes!("../../pdf_interp/src/resources/image/testdata/gray8.jp2");

/// Page size in points; it renders at one pixel per point.
const PAGE: (usize, usize) = (16, 8);

/// A 4 × 4 draw at the left edge of the page.
const SMALL: &str = "q 4 0 0 4 0 4 cm /Im Do Q\n";
/// An 8 × 8 draw on the right half of the page.
const LARGE: &str = "q 8 0 0 8 8 0 cm /Im Do Q\n";

/// Render `content` with `/Im` built from `image_dict` and `data` and
/// return the page's RGB rows, top first.
fn render(content: &str, image_dict: &str, data: &[u8]) -> Vec<Vec<u8>> {
    let objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {} {}] \
             /Resources << /XObject << /Im 5 0 R >> >> /Contents 4 0 R >>",
            PAGE.0, PAGE.1
        )
        .into_bytes(),
        stream_object("", content.as_bytes()),
        stream_object(
            &format!("/Type /XObject /Subtype /Image /Width 8 /Height 8 {image_dict}"),
            data,
        ),
    ];
    let s = session(assemble_pdf(&objects), BackendPolicy::CpuOnly);
    let bmp = render_page_rgb(&s, 1, 1.0).expect("render page");
    assert_eq!((bmp.width as usize, bmp.height as usize), PAGE);
    (0..PAGE.1)
        .map(|y| bmp.data()[y * bmp.stride..][..PAGE.0 * 3].to_vec())
        .collect()
}

/// Columns `x0..x1` of every row.
fn columns(rows: &[Vec<u8>], x0: usize, x1: usize) -> Vec<Vec<u8>> {
    rows.iter().map(|r| r[x0 * 3..x1 * 3].to_vec()).collect()
}

/// Draw the image small then large and large then small, and check each
/// draw's half of the page against that draw rendered alone.
fn assert_draws_independent(image_dict: &str, data: &[u8]) {
    let small = columns(&render(SMALL, image_dict, data), 0, 8);
    let large = columns(&render(LARGE, image_dict, data), 8, 16);
    assert_ne!(small, large, "the two draws must differ to be told apart");
    for content in [format!("{SMALL}{LARGE}"), format!("{LARGE}{SMALL}")] {
        let both = render(&content, image_dict, data);
        assert_eq!(columns(&both, 0, 8), small, "small draw in {content:?}");
        assert_eq!(columns(&both, 8, 16), large, "large draw in {content:?}");
    }
}

#[test]
fn jpx_drawn_at_two_sizes_renders_each_draw_alone() {
    assert_draws_independent("/BitsPerComponent 8 /Filter /JPXDecode", GRAY8_JP2);
}

#[test]
fn raw_image_drawn_at_two_sizes_renders_each_draw_alone() {
    let samples: Vec<u8> = (0..64u8).map(|i| i.wrapping_mul(37)).collect();
    assert_draws_independent("/ColorSpace /DeviceGray /BitsPerComponent 8", &samples);
}
