//! A raw image stream shorter than its dictionary declares — truncated or
//! damaged transport data — draws the rows it holds.  The missing samples
//! are zero, the value a short sub-byte stream already reads as, and the
//! image is reported as a partial failure, not as skipped.

#[expect(
    dead_code,
    reason = "shared helpers; this test uses PDF assembly and sessions"
)]
mod common;

use common::{assemble_pdf, stream_object};
use rasterrocket::{RasterOptions, RenderedPage, raster_pdf_from_bytes};

/// Image side in samples; the page is the same size in points and renders
/// at one pixel per point.
const SIDE: u32 = 16;

/// The rows a truncated stream holds: the top half.
const PRESENT_ROWS: usize = (SIDE / 2) as usize;

/// One page drawing a 16 × 16 image with dictionary entries `entries` and
/// stream bytes `data` over the whole page.
fn page(entries: &str, data: &[u8]) -> Vec<u8> {
    let objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {SIDE} {SIDE}] \
             /Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>"
        )
        .into_bytes(),
        stream_object(
            "",
            format!("q {SIDE} 0 0 {SIDE} 0 0 cm /Im0 Do Q").as_bytes(),
        ),
        stream_object(
            &format!("/Type /XObject /Subtype /Image /Width {SIDE} /Height {SIDE} {entries}"),
            data,
        ),
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

/// `ASCIIHexDecode` encoding of `data`.
fn hex(data: &[u8]) -> Vec<u8> {
    let mut out: Vec<u8> = data
        .iter()
        .flat_map(|b| format!("{b:02X}").into_bytes())
        .collect();
    out.push(b'>');
    out
}

/// The top `PRESENT_ROWS` rows of an image whose every pixel is `pixel`.
fn top_rows(pixel: &[u8]) -> Vec<u8> {
    pixel.repeat(SIDE as usize * PRESENT_ROWS)
}

#[test]
fn truncated_raw_images_keep_their_decoded_rows() {
    // (case, dictionary entries, stream bytes, gray of a present row,
    //  gray of a missing row)
    let cases: Vec<(&str, String, Vec<u8>, u8, u8)> = vec![
        (
            "Gray 8",
            "/ColorSpace /DeviceGray /BitsPerComponent 8".into(),
            top_rows(&[128]),
            128,
            0,
        ),
        (
            "RGB 8",
            "/ColorSpace /DeviceRGB /BitsPerComponent 8".into(),
            top_rows(&[128, 128, 128]),
            128,
            0,
        ),
        (
            "Gray 16",
            "/ColorSpace /DeviceGray /BitsPerComponent 16".into(),
            top_rows(&[128, 0]),
            128,
            0,
        ),
        (
            // Missing samples are no ink.
            "CMYK 8",
            "/ColorSpace /DeviceCMYK /BitsPerComponent 8".into(),
            top_rows(&[0, 0, 0, 255]),
            0,
            255,
        ),
        (
            // Missing indices select palette entry 0.
            "Indexed 8",
            "/ColorSpace [/Indexed /DeviceGray 1 <C840>] /BitsPerComponent 8".into(),
            top_rows(&[1]),
            0x40,
            0xC8,
        ),
        (
            "Gray 8 behind ASCIIHex",
            "/ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /ASCIIHexDecode".into(),
            hex(&top_rows(&[128])),
            128,
            0,
        ),
    ];
    for (name, entries, data, present, missing) in cases {
        let page = render(page(&entries, &data));
        let [failure] = page.diagnostics.image_failures.as_slice() else {
            panic!(
                "{name}: one partial failure expected, got {:?}",
                page.diagnostics.image_failures
            );
        };
        assert!(
            failure.partial && failure.name.as_deref() == Some("Im0"),
            "{name}: reported as {failure:?}"
        );
        let gray = |row: u32| page.pixels[(row * page.width + SIDE / 2) as usize];
        assert!(
            gray(2).abs_diff(present) <= 30,
            "{name}: present row gray {} (expected {present})",
            gray(2)
        );
        assert!(
            gray(SIDE - 3).abs_diff(missing) <= 30,
            "{name}: missing row gray {} (expected {missing})",
            gray(SIDE - 3)
        );
    }
}
