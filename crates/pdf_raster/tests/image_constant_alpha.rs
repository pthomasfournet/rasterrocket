//! An image painted under a non-opaque constant alpha (`/ca` in an
//! `ExtGState`) blends with the backdrop instead of replacing it.

#[expect(dead_code, reason = "shared helpers; this test uses only PDF assembly")]
mod common;

use common::{assemble_pdf, stream_object};
use rasterrocket::{RasterOptions, RenderedPage, raster_pdf_from_bytes};

/// 72 DPI: one pixel per point on the 100 × 100 pt page.
const DPI: f32 = 72.0;

/// One page painted black, then a white 4 × 4 gray image `/Im0` over the
/// top-right quadrant and a white inline image over the bottom-left
/// quadrant, both under the constant alpha `ca`.
fn page_with_white_images(ca: &str) -> Vec<u8> {
    let mut content = b"0 g 0 0 100 100 re f /GS0 gs \
        q 50 0 0 50 50 50 cm /Im0 Do Q \
        q 50 0 0 50 0 0 cm BI /W 2 /H 2 /CS /G /BPC 8 ID "
        .to_vec();
    content.extend_from_slice(&[0xFF; 4]);
    content.extend_from_slice(b" EI Q");
    let objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
             /Resources << /ExtGState << /GS0 << /Type /ExtGState /ca {ca} >> >> \
             /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>"
        )
        .into_bytes(),
        stream_object("", &content),
        stream_object(
            "/Type /XObject /Subtype /Image /Width 4 /Height 4 \
             /ColorSpace /DeviceGray /BitsPerComponent 8",
            &[0xFF; 16],
        ),
    ];
    assemble_pdf(&objects)
}

fn render(pdf: Vec<u8>) -> RenderedPage {
    let opts = RasterOptions::new(DPI);
    let mut pages: Vec<_> = raster_pdf_from_bytes(pdf, &opts).collect();
    assert_eq!(pages.len(), 1, "one page expected");
    pages.remove(0).1.expect("page renders")
}

/// Gray value at PDF user-space `(x, y)` (origin bottom-left).
fn gray_at(page: &RenderedPage, x: u32, y: u32) -> u8 {
    let row = page.height - 1 - y;
    page.pixels[(row * page.width + x) as usize]
}

#[test]
fn image_under_constant_alpha_blends_with_backdrop() {
    let page = render(page_with_white_images("0.25"));
    // White at 25 % over black: 0.25 × 255 ≈ 64.
    for (x, y, what) in [(75, 75, "XObject image"), (25, 25, "inline image")] {
        let g = gray_at(&page, x, y);
        assert!((62..=66).contains(&g), "{what}: gray {g}, expected ≈ 64");
    }
    assert_eq!(gray_at(&page, 75, 25), 0, "unpainted quadrant stays black");
}

#[test]
fn image_under_zero_constant_alpha_paints_nothing() {
    let page = render(page_with_white_images("0"));
    assert_eq!(gray_at(&page, 75, 75), 0);
    assert_eq!(gray_at(&page, 25, 25), 0);
}

#[test]
fn image_under_unit_constant_alpha_is_opaque() {
    let page = render(page_with_white_images("1"));
    assert_eq!(gray_at(&page, 75, 75), 255);
    assert_eq!(gray_at(&page, 25, 25), 255);
}
