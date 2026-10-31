//! An image whose declared size exceeds the decode budget is refused before
//! anything is allocated and reported in the page diagnostics; the page
//! still renders.  Sides up to 131 071 pixels are accepted when the budget
//! allows.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use common::{render_gray, stream_object, with_page_tree};

/// One 100 × 100 pt page drawing `/Im0` (dictionary `entries`, stream
/// `data`) over the whole page.
fn page(entries: &str, data: &[u8]) -> Vec<u8> {
    let objects = vec![
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
           /Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>"
            .to_vec(),
        stream_object("", b"q 100 0 0 100 0 0 cm /Im0 Do Q"),
        stream_object(&format!("/Type /XObject /Subtype /Image {entries}"), data),
    ];
    with_page_tree(&objects)
}

#[test]
fn an_image_over_the_decode_budget_is_refused_and_reported() {
    // 65 536 × 65 536 one-bit samples would expand to 4 GiB; the stream
    // holds sixteen bytes.
    let cases = [
        (
            "gray 1-bit",
            "/Width 65536 /Height 65536 /BitsPerComponent 1 /ColorSpace /DeviceGray",
        ),
        (
            "stencil",
            "/Width 65536 /Height 65536 /BitsPerComponent 1 /ImageMask true",
        ),
        (
            "RGB 8-bit",
            "/Width 65536 /Height 65536 /BitsPerComponent 8 /ColorSpace /DeviceRGB",
        ),
    ];
    for (name, entries) in cases {
        let page = render_gray(page(entries, &[0xAA; 16]));
        let [failure] = page.diagnostics.image_failures.as_slice() else {
            panic!(
                "{name}: one failure expected, got {:?}",
                page.diagnostics.image_failures
            );
        };
        assert!(
            !failure.partial && failure.reason.contains("budget"),
            "{name}: reported as {failure:?}"
        );
        assert!(
            page.pixels.iter().all(|&p| p == 255),
            "{name}: the page stays white"
        );
    }
}

#[test]
fn a_wide_image_within_the_budget_decodes() {
    // 131 071 × 2 gray samples: wider than 65 536, well under the budget.
    let row = vec![0x40u8; 131_071];
    let data = [row.as_slice(), row.as_slice()].concat();
    let page = render_gray(page(
        "/Width 131071 /Height 2 /BitsPerComponent 8 /ColorSpace /DeviceGray",
        &data,
    ));
    assert!(
        page.diagnostics.image_failures.is_empty(),
        "{:?}",
        page.diagnostics.image_failures
    );
    let centre = page.pixels[(50 * page.width + 50) as usize];
    assert_eq!(centre, 0x40);
}
