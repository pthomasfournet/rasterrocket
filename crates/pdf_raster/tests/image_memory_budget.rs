//! An image whose declared size exceeds the decode budget is refused before
//! anything is allocated and reported in the page diagnostics; the page
//! still renders.  Sides up to 131 071 pixels are accepted when the budget
//! allows.

#[expect(dead_code, reason = "shared helpers; this test uses only PDF assembly")]
mod common;

use common::{assemble_pdf, stream_object};
use rasterrocket::{RasterOptions, RenderedPage, raster_pdf_from_bytes};

/// One 100 × 100 pt page drawing `/Im0` (dictionary `entries`, stream
/// `data`) over the whole page.
fn page(entries: &str, data: &[u8]) -> Vec<u8> {
    let objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
           /Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>"
            .to_vec(),
        stream_object("", b"q 100 0 0 100 0 0 cm /Im0 Do Q"),
        stream_object(&format!("/Type /XObject /Subtype /Image {entries}"), data),
    ];
    assemble_pdf(&objects)
}

fn render(pdf: Vec<u8>) -> RenderedPage {
    let opts = RasterOptions {
        dpi: 72.0,
        ..RasterOptions::default()
    };
    let mut pages: Vec<_> = raster_pdf_from_bytes(pdf, &opts).collect();
    assert_eq!(pages.len(), 1, "one page expected");
    pages.remove(0).1.expect("page renders")
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
        let page = render(page(entries, &[0xAA; 16]));
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
    let page = render(page(
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
