//! An inline image (`BI … EI`) repeated on a page — the same block in a
//! form drawn several times — is decoded once per page, as an image
//! `XObject` drawn several times is.  A decode is observable through its
//! diagnostics: a truncated inline image reports one partial failure per
//! decode.

#[expect(dead_code, reason = "shared helpers; this test uses only PDF assembly")]
mod common;

use std::fmt::Write as _;

use common::{assemble_pdf, stream_object};
use rasterrocket::{RasterOptions, RenderedPage, raster_pdf_from_bytes};

/// A page drawing form `/Fm` `draws` times; the form holds one 4 × 4 gray
/// inline image whose data is short by half (a partial decode) over the
/// unit square.
fn page(draws: usize) -> Vec<u8> {
    let mut content = String::new();
    for i in 0..draws {
        let _ = write!(content, "q 10 0 0 10 {} 0 cm /Fm Do Q ", i * 10);
    }
    let mut form = b"q 1 0 0 1 0 0 cm BI /W 4 /H 4 /CS /G /BPC 8 ID ".to_vec();
    form.extend_from_slice(&[0x40; 8]);
    form.extend_from_slice(b" EI Q");
    let objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 10] \
           /Resources << /XObject << /Fm 5 0 R >> >> /Contents 4 0 R >>"
            .to_vec(),
        stream_object("", content.as_bytes()),
        stream_object("/Type /XObject /Subtype /Form /BBox [0 0 1 1]", &form),
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
fn a_repeated_inline_image_decodes_once_per_page() {
    let page = render(page(10));
    let failures = &page.diagnostics.image_failures;
    assert_eq!(
        failures.len(),
        1,
        "one decode, one partial report: {failures:?}"
    );
    assert!(failures[0].partial && failures[0].name.is_none());
    // Every draw shows the image: its top half is the decoded 0x40, the
    // bottom half the zero padding.
    for i in 0..10u32 {
        let x = i * 10 + 5;
        let top = page.pixels[(2 * page.width + x) as usize];
        let bottom = page.pixels[(7 * page.width + x) as usize];
        assert_eq!((top, bottom), (0x40, 0x00), "draw {i}");
    }
}
