//! A page without its own `/Resources` takes them from its page-tree ancestor
//! (ISO 32000-1 §7.7.3.4).  Every resource category the content stream names
//! must resolve through that inheritance, not only fonts and colour spaces.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use common::{assemble_pdf, gray_at, render_gray, stream_object};

/// One page whose four quadrants are painted through four resource
/// categories, all defined on the `/Pages` node and none on the page:
///
/// - bottom-left: image `XObject` `/Im0` (2 × 2 black `DeviceGray` samples),
/// - bottom-right: form `XObject` `/Fm0` filling its unit area black,
/// - top-left: a black fill under `ExtGState` `/GS0` with `/ca 0.5`,
/// - top-right: axial shading `/Sh0`, constant black.
fn page_with_inherited_resources() -> Vec<u8> {
    let content = "q 50 0 0 50 0 0 cm /Im0 Do Q \
                   q 1 0 0 1 50 0 cm /Fm0 Do Q \
                   q /GS0 gs 0 g 0 50 50 50 re f Q \
                   q 50 50 50 50 re W n /Sh0 sh Q";
    let objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 \
           /Resources << /XObject << /Im0 5 0 R /Fm0 6 0 R >> \
                         /ExtGState << /GS0 << /Type /ExtGState /ca 0.5 >> >> \
                         /Shading << /Sh0 7 0 R >> >> >>"
            .to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Contents 4 0 R >>".to_vec(),
        stream_object("", content.as_bytes()),
        stream_object(
            "/Type /XObject /Subtype /Image /Width 2 /Height 2 \
             /ColorSpace /DeviceGray /BitsPerComponent 8",
            &[0, 0, 0, 0],
        ),
        stream_object(
            "/Type /XObject /Subtype /Form /BBox [0 0 50 50]",
            b"0 g 0 0 50 50 re f",
        ),
        b"<< /ShadingType 2 /ColorSpace /DeviceGray /Coords [50 50 100 100] \
           /Function << /FunctionType 2 /Domain [0 1] /C0 [0] /C1 [0] /N 1 >> \
           /Extend [true true] >>"
            .to_vec(),
    ];
    assemble_pdf(&objects)
}

#[test]
fn every_resource_category_is_inherited_from_the_page_tree() {
    let page = render_gray(page_with_inherited_resources());

    assert_eq!(gray_at(&page, 25, 25), 0, "inherited image XObject");
    assert_eq!(gray_at(&page, 75, 25), 0, "inherited form XObject");
    let half = gray_at(&page, 25, 75);
    assert!(
        (120..=136).contains(&half),
        "inherited ExtGState /ca 0.5 over white: {half}"
    );
    assert_eq!(gray_at(&page, 75, 75), 0, "inherited shading");
    assert!(
        page.diagnostics.has_images,
        "the pre-scan sees the inherited image"
    );
    assert!(
        page.diagnostics.image_failures.is_empty(),
        "{:?}",
        page.diagnostics.image_failures
    );
}
