//! Colour-key masking (`/Mask [min₁ max₁ …]`, PDF 32000-1 §8.9.6.4)
//! compares the image's samples as stored: palette indices for `Indexed`,
//! colorant values for CMYK, values of the image's own bit depth, before
//! `/Decode` is applied.

#[expect(
    dead_code,
    reason = "shared helpers; this test uses PDF assembly and sessions"
)]
mod common;

use common::{assemble_pdf, session, stream_object};
use rasterrocket::{BackendPolicy, render_page_rgb};

/// Page side in points; it renders at one pixel per point.
const SIDE: u32 = 16;

/// Render a 2 × 1 image with dictionary entries `entries` and samples
/// `data` over a red page, and report whether its left and right pixels
/// were masked (the red page shows through).
fn masked(entries: &str, data: &[u8]) -> (bool, bool) {
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
            format!("1 0 0 rg 0 0 {SIDE} {SIDE} re f q {SIDE} 0 0 {SIDE} 0 0 cm /Im0 Do Q")
                .as_bytes(),
        ),
        stream_object(
            &format!("/Type /XObject /Subtype /Image /Width 2 /Height 1 {entries}"),
            data,
        ),
    ];
    let s = session(assemble_pdf(&objects), BackendPolicy::CpuOnly);
    let bmp = render_page_rgb(&s, 1, 1.0).expect("render page");
    let row = &bmp.data()[8 * bmp.stride..];
    let red = |x: usize| row[x * 3] > 200 && row[x * 3 + 1] < 60 && row[x * 3 + 2] < 60;
    (red(3), red(12))
}

#[test]
fn colour_key_compares_stored_samples() {
    // In every case the left pixel's samples fall within the key ranges
    // and the right pixel's do not.
    let cases: [(&str, &str, &[u8]); 7] = [
        (
            "RGB 8",
            "/ColorSpace /DeviceRGB /BitsPerComponent 8 /Mask [250 255 0 5 0 5]",
            &[255, 0, 0, 0, 0, 255],
        ),
        (
            "Indexed 8",
            "/ColorSpace [/Indexed /DeviceGray 1 <00FF>] /BitsPerComponent 8 /Mask [0 0]",
            &[0, 1],
        ),
        (
            "Indexed 2",
            "/ColorSpace [/Indexed /DeviceGray 3 <00555500>] /BitsPerComponent 2 /Mask [3 3]",
            &[0b1100_0000],
        ),
        (
            "CMYK 8",
            "/ColorSpace /DeviceCMYK /BitsPerComponent 8 /Mask [0 0 0 0 0 0 0 0]",
            &[0, 0, 0, 0, 0, 0, 0, 255],
        ),
        (
            "Gray 4",
            "/ColorSpace /DeviceGray /BitsPerComponent 4 /Mask [15 15]",
            &[0xF0],
        ),
        (
            "Gray 16",
            "/ColorSpace /DeviceGray /BitsPerComponent 16 /Mask [65535 65535]",
            &[0xFF, 0xFF, 0x00, 0x00],
        ),
        (
            "Gray 8 under /Decode [1 0]",
            "/ColorSpace /DeviceGray /BitsPerComponent 8 /Decode [1 0] /Mask [0 0]",
            &[0, 255],
        ),
    ];
    for (name, entries, data) in cases {
        assert_eq!(masked(entries, data), (true, false), "{name}");
    }
}

/// A bilevel codec's samples are one bit each, so its key ranges are
/// compared with the bit values, not with the bytes the decoder expands
/// them to.
#[test]
fn colour_key_on_a_bilevel_codec_compares_one_bit_samples() {
    // Group 4, one row of two pixels: VL1 (a white pixel) then V0 (a black
    // pixel to the end of the row), then EOFB.  Under the default
    // `BlackIs1 false` the white pixel is sample 1.
    let g4 = [0x50, 0x01, 0x00, 0x10];
    assert_eq!(
        masked(
            "/ColorSpace /DeviceGray /BitsPerComponent 1 /Filter /CCITTFaxDecode \
             /DecodeParms << /K -1 /Columns 2 /Rows 1 >> /Mask [1 1]",
            &g4
        ),
        (true, false),
        "CCITT G4, /Mask [1 1] masks the white sample"
    );
    assert_eq!(
        masked(
            "/ColorSpace /DeviceGray /BitsPerComponent 1 /Filter /CCITTFaxDecode \
             /DecodeParms << /K -1 /Columns 2 /Rows 1 >> /Mask [0 0]",
            &g4
        ),
        (false, true),
        "CCITT G4, /Mask [0 0] masks the black sample"
    );
}
