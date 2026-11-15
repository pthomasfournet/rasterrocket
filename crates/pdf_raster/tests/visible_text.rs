//! The glyph counts of `PageDiagnostics`: a page of drawn text, the
//! invisible text layer of a recognised scan, both, and text that cannot
//! mark the page.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use common::{one_page, render_diagnosed, session, stream_object};
use rasterrocket::{BackendPolicy, PageDiagnostics};

/// The diagnostics of a 200 × 100 pt page whose crop box is its left
/// 150 pt, whose `/F` is Helvetica and whose `/Scan` is a gray image,
/// showing `content`.
fn diagnostics(content: &str) -> PageDiagnostics {
    let pdf = one_page(
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /CropBox [0 0 150 100] \
         /Resources << /Font << /F 5 0 R >> /XObject << /Scan 6 0 R >> >> \
         /Contents 4 0 R >>",
        vec![
            stream_object("", content.as_bytes()),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
            stream_object(
                "/Type /XObject /Subtype /Image /Width 2 /Height 2 \
                 /ColorSpace /DeviceGray /BitsPerComponent 8",
                &[40, 200, 200, 40],
            ),
        ],
    );
    render_diagnosed(&session(pdf, BackendPolicy::CpuOnly), 1).1
}

const SCAN: &str = "q 150 0 0 100 0 0 cm /Scan Do Q ";

#[test]
fn drawn_text_is_visible() {
    let d = diagnostics("BT /F 12 Tf 10 50 Td (Native) Tj ET");
    assert!(d.has_visible_text(), "{:?}", d.glyphs);
    assert_eq!((d.glyphs.fill, d.glyphs.invisible), (6, 0));
}

#[test]
fn the_text_layer_of_a_scan_is_not_visible() {
    let d = diagnostics(&format!(
        "{SCAN}BT 3 Tr /F 12 Tf 10 50 Td (Recognised) Tj 0 -14 Td (text) Tj ET"
    ));
    assert!(d.has_images && d.has_vector_text);
    assert!(!d.has_visible_text(), "{:?}", d.glyphs);
    assert_eq!(d.glyphs.invisible, 14);
}

#[test]
fn a_scan_with_drawn_text_is_visible() {
    let d = diagnostics(&format!(
        "{SCAN}BT 3 Tr /F 12 Tf 10 50 Td (Recognised) Tj 0 Tr 0 -14 Td (Stamp) Tj ET"
    ));
    assert!(d.has_visible_text());
    assert_eq!((d.glyphs.fill, d.glyphs.invisible), (5, 10));
}

/// Text outside the crop box, though within the media box, and text in a
/// clip-only mode do not make a page of visible text.
#[test]
fn text_that_cannot_mark_the_page_is_not_visible() {
    let d = diagnostics(
        "BT /F 12 Tf 170 50 Td (Margin) Tj ET \
         BT /F 12 Tf 10 -40 Td (Below) Tj ET \
         BT 7 Tr /F 12 Tf 10 50 Td (Clip) Tj ET",
    );
    assert!(!d.has_visible_text(), "{:?}", d.glyphs);
    assert_eq!((d.glyphs.off_page, d.glyphs.clip_only), (11, 4));
}
