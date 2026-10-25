//! An image is painted through the same compositing state as a fill: the
//! nonstroking constant alpha applies to it, and inside a transparency group
//! its shape enters the group's alpha, so the group composites it back.

#[expect(dead_code, reason = "shared helpers; this test uses only PDF assembly")]
mod common;

use common::{assemble_pdf, stream_object};
use rasterrocket::{RasterOptions, RenderedPage, raster_pdf_from_bytes};

/// 72 DPI: one pixel per point on the 100 × 100 pt page.
const DPI: f32 = 72.0;

/// A 2 × 2 black `DeviceGray` image.
fn black_image() -> Vec<u8> {
    stream_object(
        "/Type /XObject /Subtype /Image /Width 2 /Height 2 \
         /ColorSpace /DeviceGray /BitsPerComponent 8",
        &[0, 0, 0, 0],
    )
}

/// One page running `content` with image `/Im0`, form `/Fm0` (whose content
/// is `form_content` and whose dictionary carries `form_extras`) and
/// `ExtGState` `/Half` (`/ca 0.5`) in its resources.
fn page(content: &str, form_content: &str, form_extras: &str) -> Vec<u8> {
    let objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
           /Resources << /XObject << /Im0 5 0 R /Fm0 6 0 R >> \
                         /ExtGState << /Half << /Type /ExtGState /ca 0.5 >> >> >> \
           /Contents 4 0 R >>"
            .to_vec(),
        stream_object("", content.as_bytes()),
        black_image(),
        stream_object(
            &format!(
                "/Type /XObject /Subtype /Form /BBox [0 0 100 100] \
                 /Resources << /XObject << /Im0 5 0 R >> >> {form_extras}"
            ),
            form_content.as_bytes(),
        ),
    ];
    assemble_pdf(&objects)
}

fn render(pdf: Vec<u8>) -> RenderedPage {
    let opts = RasterOptions {
        dpi: DPI,
        ..RasterOptions::default()
    };
    let mut pages: Vec<_> = raster_pdf_from_bytes(pdf, &opts).collect();
    assert_eq!(pages.len(), 1, "one page expected");
    pages.remove(0).1.expect("page renders")
}

/// Gray value at PDF user-space `(x, y)` (origin bottom-left).
fn gray_at(page: &RenderedPage, x: u32, y: u32) -> u8 {
    let row = page.height - 1 - y;
    page.pixels[(row * page.width + x) as usize]
}

const IMAGE_IN_LEFT_HALF: &str = "q 50 0 0 100 0 0 cm /Im0 Do Q";

#[test]
fn image_inside_an_isolated_group_is_composited() {
    let page = render(page(
        "/Fm0 Do",
        IMAGE_IN_LEFT_HALF,
        "/Group << /S /Transparency /I true /CS /DeviceGray >>",
    ));
    assert_eq!(gray_at(&page, 25, 50), 0, "the image reaches the page");
    assert_eq!(gray_at(&page, 75, 50), 255, "the group adds nothing else");
}

#[test]
fn image_inside_a_non_isolated_group_is_composited() {
    let page = render(page(
        "/Fm0 Do",
        IMAGE_IN_LEFT_HALF,
        "/Group << /S /Transparency >>",
    ));
    assert_eq!(gray_at(&page, 25, 50), 0, "the image reaches the page");
    assert_eq!(gray_at(&page, 75, 50), 255, "the backdrop is unchanged");
}

#[test]
fn image_honours_the_nonstroking_constant_alpha() {
    let page = render(page(&format!("q /Half gs {IMAGE_IN_LEFT_HALF} Q"), "", ""));
    let half = gray_at(&page, 25, 50);
    assert!(
        (120..=136).contains(&half),
        "black at ca 0.5 over white: {half}"
    );
}

#[test]
fn opaque_image_outside_any_group_is_unchanged() {
    let page = render(page(IMAGE_IN_LEFT_HALF, "", ""));
    assert_eq!(gray_at(&page, 25, 50), 0);
    assert_eq!(gray_at(&page, 75, 50), 255);
}
