//! Stream filters named by their abbreviations, and `BrotliDecode`, decode
//! like their full-name counterparts.

#[expect(
    dead_code,
    reason = "shared helpers; this test uses PDF assembly and ink coverage"
)]
mod common;

use common::{SQUARE, ink, one_page, stream_object};

/// The bytes spelled by the hexadecimal string `hex`.
fn unhex(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex digit pair"))
        .collect()
}

/// `SQUARE` compressed by zlib.
const SQUARE_ZLIB: &str = "789c33504857303400210b302a4a5548030029140447";

const PAGE: &str = "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Contents 4 0 R >>";

/// The square's coverage when the content stream is `data` under `filter`.
fn content_ink(filter: &str, data: &[u8]) -> f64 {
    let dict = format!("/Filter {filter}");
    ink(one_page(PAGE, vec![stream_object(&dict, data)]))
}

/// A content stream filtered under an abbreviated name (ISO 32000-2
/// Table 92) decodes as under the full name.
#[test]
fn abbreviated_filter_names_decode_content_streams() {
    let ahx = format!("{}>", hex_of(SQUARE));
    let ahx_fl = format!("{}>", SQUARE_ZLIB);
    for (filter, data) in [
        ("/Fl", unhex(SQUARE_ZLIB)),
        ("/AHx", ahx.into_bytes()),
        ("[/AHx /Fl]", ahx_fl.into_bytes()),
        ("/A85", b"0Hb!N0ea_)0H`>/+?2%tEb-A*~>".to_vec()),
        ("/RL", rl_literal(SQUARE)),
    ] {
        let ink = content_ink(filter, &data);
        assert!(
            (ink - 0.32).abs() < 0.01,
            "{filter}: square must render, ink {ink}"
        );
    }
}

/// An image XObject whose `/Filter` is abbreviated decodes as under the
/// full name.
#[test]
fn abbreviated_filter_name_decodes_an_image() {
    let pdf = one_page(
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] \
           /Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>",
        vec![
            stream_object("", b"q 80 0 0 80 10 10 cm /Im0 Do Q"),
            stream_object(
                "/Type /XObject /Subtype /Image /Width 1 /Height 1 \
                 /ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /Fl",
                // One black sample, compressed by zlib.
                &unhex("789c63000000010001"),
            ),
        ],
    );
    let ink = ink(pdf);
    assert!((ink - 0.32).abs() < 0.01, "image must render, ink {ink}");
}

/// A `BrotliDecode` content stream decodes.
#[test]
fn brotli_content_stream_decodes() {
    // `SQUARE` followed by a newline, forty times, compressed by Brotli.
    let data = unhex("1b4703f81d09360ef5a137905383c02894ed189c33391563161506096b970831e63b");
    let ink = content_ink("/BrotliDecode", &data);
    assert!((ink - 0.32).abs() < 0.01, "square must render, ink {ink}");
}

/// `data` as ASCII hexadecimal.
fn hex_of(data: &[u8]) -> String {
    data.iter().map(|b| format!("{b:02x}")).collect()
}

/// `data` as a single `RunLengthDecode` literal run followed by EOD.
fn rl_literal(data: &[u8]) -> Vec<u8> {
    let mut out = vec![u8::try_from(data.len() - 1).expect("at most 128 bytes")];
    out.extend_from_slice(data);
    out.push(128);
    out
}
