//! An image in an `ICCBased` colour space is drawn through its embedded
//! matrix/TRC profile, not as the device space of the same channel count.
//!
//! The profiles are gamma 1.8 (the classic Apple RGB tone curve): a mid-tone
//! sample 128 is linear `(128/255)^1.8 ≈ 0.289`, which sRGB-encodes to 146.
//! Treating the samples as sRGB would leave it at 128.  pdfium and poppler
//! both render such images lighter in the same way.

#[expect(dead_code, reason = "shared helpers; this test uses only PDF assembly")]
mod common;

use common::icc::{GAMMA_1_8, gray_profile, rgb_profile};
use common::{assemble_pdf, stream_object};
use rasterrocket::{RasterOptions, RenderedPage, raster_pdf_from_bytes};

/// One 100 × 100 pt page filled by a 2 × 2 image of mid-gray samples in
/// `ICCBased` space `/N n` with `profile`.  `inline` draws it as an inline
/// image instead of an `XObject`.
fn page(n: u8, profile: &[u8], inline: bool) -> Vec<u8> {
    let samples = vec![128u8; 4 * usize::from(n)];
    let (content, resources) = if inline {
        let mut c = b"q 100 0 0 100 0 0 cm BI /W 2 /H 2 /CS /CS0 /BPC 8 ID ".to_vec();
        c.extend_from_slice(&samples);
        c.extend_from_slice(b" EI Q");
        (c, "/ColorSpace << /CS0 [/ICCBased 5 0 R] >>")
    } else {
        (
            b"q 100 0 0 100 0 0 cm /Im0 Do Q".to_vec(),
            "/XObject << /Im0 6 0 R >>",
        )
    };
    let objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
             /Resources << {resources} >> /Contents 4 0 R >>"
        )
        .into_bytes(),
        stream_object("", &content),
        stream_object(&format!("/N {n}"), profile),
        stream_object(
            "/Type /XObject /Subtype /Image /Width 2 /Height 2 \
             /ColorSpace [/ICCBased 5 0 R] /BitsPerComponent 8",
            &samples,
        ),
    ];
    assemble_pdf(&objects)
}

fn render(pdf: Vec<u8>) -> RenderedPage {
    let opts = RasterOptions::new(72.0);
    let mut pages: Vec<_> = raster_pdf_from_bytes(pdf, &opts).collect();
    assert_eq!(pages.len(), 1, "one page expected");
    pages.remove(0).1.expect("page renders")
}

/// The page's centre pixel; the image covers the whole page.
fn centre(page: &RenderedPage) -> u8 {
    page.pixels[(page.height / 2 * page.width + page.width / 2) as usize]
}

#[test]
fn rgb_profile_lightens_an_xobject_image() {
    let v = centre(&render(page(3, &rgb_profile(GAMMA_1_8), false)));
    assert!(v.abs_diff(146) <= 1, "expected ≈146, got {v}");
}

#[test]
fn rgb_profile_lightens_an_inline_image() {
    let v = centre(&render(page(3, &rgb_profile(GAMMA_1_8), true)));
    assert!(v.abs_diff(146) <= 1, "expected ≈146, got {v}");
}

#[test]
fn gray_profile_lightens_an_xobject_image() {
    let v = centre(&render(page(1, &gray_profile(GAMMA_1_8), false)));
    assert!(v.abs_diff(146) <= 1, "expected ≈146, got {v}");
}
