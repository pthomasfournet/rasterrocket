//! An image's `/SMask` decodes through the same `/Filter` chain and
//! `/DecodeParms` forms as the image itself (PDF 32000-1 §7.4, §11.6.5.2):
//! any transport filter, any chain of them ahead of a codec, and
//! `/DecodeParms` given as an array index-aligned to `/Filter`.

#[expect(
    dead_code,
    reason = "shared helpers; this test uses PDF assembly and sessions"
)]
mod common;

use common::{assemble_pdf, session, stream_object};
use rasterrocket::{BackendPolicy, render_page_rgb};

/// Mask side in samples; the page is the same size in points and renders
/// at one pixel per point.
const SIDE: u32 = 16;

/// A CCITT Group 4 (`K -1`) coding of a 16 × 16 image whose left eight
/// columns are white and right eight black: as an 8-bit gray mask it
/// decodes to alpha 255 on the left and 0 on the right.
const G4_RIGHT_BLACK: [u8; 9] = [0x33, 0x17, 0xFF, 0xFF, 0xFF, 0xF0, 0x01, 0x00, 0x10];

/// Raw 8-bit gray rows: alpha 255 on the left eight samples, 0 on the right.
fn raw_left_opaque() -> Vec<u8> {
    let row: Vec<u8> = (0..SIDE).map(|x| if x < 8 { 255 } else { 0 }).collect();
    row.repeat(SIDE as usize)
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

/// `RunLengthDecode` encoding of `data`: one run per row half, each half
/// being one repeated byte.
fn run_length(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for half in data.chunks(8) {
        assert!(half.iter().all(|&b| b == half[0]));
        // Length byte 257 − n repeats the next byte n times.
        out.extend_from_slice(&[249, half[0]]);
    }
    out.push(128);
    out
}

/// Render a black 16 × 16 `DeviceGray` image whose `/SMask` has the
/// filter entries `entries` and encoded bytes `data`, and return the gray
/// of the left and right halves of row 8.
fn masked_halves(entries: &str, data: &[u8]) -> (u8, u8) {
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
            &format!(
                "/Type /XObject /Subtype /Image /Width {SIDE} /Height {SIDE} \
                 /ColorSpace /DeviceGray /BitsPerComponent 8 /SMask 6 0 R"
            ),
            &vec![0; (SIDE * SIDE) as usize],
        ),
        stream_object(
            &format!(
                "/Type /XObject /Subtype /Image /Width {SIDE} /Height {SIDE} \
                 /ColorSpace /DeviceGray {entries}"
            ),
            data,
        ),
    ];
    let s = session(assemble_pdf(&objects), BackendPolicy::CpuOnly);
    let bmp = render_page_rgb(&s, 1, 1.0).expect("render page");
    let row = &bmp.data()[8 * bmp.stride..];
    (row[2 * 3], row[13 * 3])
}

#[test]
fn smask_decodes_through_every_filter_form() {
    let raw = raw_left_opaque();
    let g4_parms = format!("<< /K -1 /Columns {SIDE} /Rows {SIDE} >>");
    let cases: Vec<(&str, String, Vec<u8>)> = vec![
        ("raw", "/BitsPerComponent 8".into(), raw.clone()),
        (
            "ASCIIHex",
            "/BitsPerComponent 8 /Filter /ASCIIHexDecode".into(),
            hex(&raw),
        ),
        (
            "RunLength",
            "/BitsPerComponent 8 /Filter /RunLengthDecode".into(),
            run_length(&raw),
        ),
        (
            "ASCIIHex then RunLength",
            "/BitsPerComponent 8 /Filter [/ASCIIHexDecode /RunLengthDecode]".into(),
            hex(&run_length(&raw)),
        ),
        (
            "CCITT G4, array /DecodeParms",
            format!("/BitsPerComponent 1 /Filter [/CCITTFaxDecode] /DecodeParms [{g4_parms}]"),
            G4_RIGHT_BLACK.to_vec(),
        ),
        (
            "ASCIIHex then CCITT G4",
            format!(
                "/BitsPerComponent 1 /Filter [/ASCIIHexDecode /CCITTFaxDecode] \
                 /DecodeParms [null {g4_parms}]"
            ),
            hex(&G4_RIGHT_BLACK),
        ),
    ];
    for (name, entries, data) in cases {
        let (left, right) = masked_halves(&entries, &data);
        assert!(
            left < 60 && right > 200,
            "{name}: the image must show on the left only; (left, right) gray = ({left}, {right})"
        );
    }
}
