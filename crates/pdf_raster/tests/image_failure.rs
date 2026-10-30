//! A page whose image cannot be decoded renders everything else and reports
//! the skipped image in its diagnostics; under `strict` the page fails.

#[expect(dead_code, reason = "shared helpers; this test uses only PDF assembly")]
mod common;

use common::{assemble_pdf, stream_object};
use rasterrocket::{RasterError, RasterOptions, RenderedPage, raster_pdf_from_bytes};

/// 72 DPI: one pixel per point on the 100 × 100 pt page.
const DPI: f32 = 72.0;

/// One page: a black 20 × 20 pt square at the origin, a line of text, and a
/// `FlateDecode` image `/Im0` whose stream is not a zlib stream, drawn over
/// the top-right quadrant.  `extra_ops` is appended to the content stream.
fn page_with_corrupt_image(extra_ops: &str) -> Vec<u8> {
    let content = format!(
        "0 g 0 0 20 20 re f \
         BT /F1 12 Tf 10 60 Td (Hello) Tj ET \
         q 50 0 0 50 50 50 cm /Im0 Do Q {extra_ops}"
    );
    let objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
           /Resources << /Font << /F1 5 0 R >> /XObject << /Im0 6 0 R >> >> \
           /Contents 4 0 R >>"
            .to_vec(),
        stream_object("", content.as_bytes()),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
        stream_object(
            "/Type /XObject /Subtype /Image /Width 4 /Height 4 \
             /ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode",
            b"this is not a zlib stream",
        ),
    ];
    assemble_pdf(&objects)
}

fn render(pdf: Vec<u8>, strict: bool) -> Result<RenderedPage, RasterError> {
    let mut opts = RasterOptions::new(DPI);
    opts.strict = strict;
    let mut pages: Vec<_> = raster_pdf_from_bytes(pdf, &opts).collect();
    assert_eq!(pages.len(), 1, "one page expected");
    pages.remove(0).1
}

/// Gray value at PDF user-space `(x, y)` (origin bottom-left).
fn gray_at(page: &RenderedPage, x: u32, y: u32) -> u8 {
    let row = page.height - 1 - y;
    page.pixels[(row * page.width + x) as usize]
}

#[test]
fn corrupt_image_is_skipped_and_reported() {
    let page = render(page_with_corrupt_image(""), false).expect("page renders");

    assert_eq!(gray_at(&page, 10, 10), 0, "the square is painted");
    assert_eq!(
        gray_at(&page, 75, 75),
        255,
        "the skipped image leaves the page"
    );
    assert!(page.diagnostics.has_vector_text, "the text is shown");

    let failures = &page.diagnostics.image_failures;
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert_eq!(failures[0].name.as_deref(), Some("Im0"));
    assert!(
        failures[0].reason.starts_with("FlateDecode"),
        "reason names the codec: {}",
        failures[0].reason
    );
}

/// The image's name appears once in the reported failure: the reason holds
/// only what went wrong, and the display prefixes the name.
#[test]
fn failure_names_the_image_once() {
    let pdf = page_with_corrupt_image("q 10 0 0 10 0 0 cm /Im9 Do Q");
    let page = render(pdf, false).expect("page renders");
    let failures = &page.diagnostics.image_failures;
    let missing = failures
        .iter()
        .find(|f| f.name.as_deref() == Some("Im9"))
        .unwrap_or_else(|| panic!("/Im9 reported: {failures:?}"));
    assert_eq!(missing.reason, "not found in the /XObject resources");
    assert_eq!(
        missing.to_string(),
        "image /Im9: not found in the /XObject resources"
    );
}

#[test]
fn content_after_a_skipped_image_still_renders() {
    let page = render(page_with_corrupt_image("0 g 80 0 20 20 re f"), false).expect("page renders");
    assert_eq!(gray_at(&page, 90, 10), 0, "operators after the image run");
}

#[test]
fn corrupt_inline_image_is_reported_without_a_name() {
    let page = render(
        page_with_corrupt_image("BI /W 2 /H 2 /BPC 8 /CS /G /F /Fl ID garbage EI"),
        false,
    )
    .expect("page renders");
    let names: Vec<_> = page
        .diagnostics
        .image_failures
        .iter()
        .map(|f| f.name.as_deref())
        .collect();
    assert_eq!(names, [Some("Im0"), None]);
}

#[test]
fn strict_fails_the_page() {
    match render(page_with_corrupt_image(""), true) {
        Err(RasterError::ImageDecodeFailed(msgs)) => {
            assert_eq!(msgs.len(), 1, "{msgs:?}");
            assert!(
                msgs[0].starts_with("image /Im0: FlateDecode"),
                "{}",
                msgs[0]
            );
        }
        other => panic!("expected ImageDecodeFailed, got {other:?}"),
    }
}

/// A `Do` naming an `XObject` absent from the resources loses whatever the
/// producer meant to draw; it is reported, never skipped silently.
#[test]
fn unresolvable_xobject_name_is_reported() {
    let page = render(page_with_corrupt_image("/Missing Do"), false).expect("page renders");
    let failures = &page.diagnostics.image_failures;
    assert_eq!(failures.len(), 2, "{failures:?}");
    assert_eq!(failures[1].name.as_deref(), Some("Missing"));
    assert!(
        failures[1].reason.contains("not found"),
        "reason says the name did not resolve: {}",
        failures[1].reason
    );
}
