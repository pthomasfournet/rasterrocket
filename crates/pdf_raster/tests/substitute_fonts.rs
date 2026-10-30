//! A font without an embedded program draws with a substitute: the
//! installed metric-compatible face when one is found, else the base-14
//! face compiled into the library.  Text in a non-embedded font therefore
//! renders on every host, including one with no fonts installed.

#[expect(
    dead_code,
    reason = "shared helpers; this test uses PDF assembly and sessions"
)]
mod common;

use common::{assemble_pdf, session, stream_object};
use rasterrocket::{BackendPolicy, render_page_rgb};

/// Render a 64 × 32 pt page at one pixel per point showing `text` in the
/// non-embedded font `font` (a font dictionary body) at 24 pt, and return
/// the number of dark pixels.
fn ink(font: &str, text: &str) -> usize {
    let objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 64 32] \
          /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
            .to_vec(),
        stream_object("", format!("BT /F1 24 Tf 4 8 Td {text} Tj ET").as_bytes()),
        font.as_bytes().to_vec(),
    ];
    let s = session(assemble_pdf(&objects), BackendPolicy::CpuOnly);
    let bmp = render_page_rgb(&s, 1, 1.0).expect("render page");
    bmp.data().chunks(3).filter(|px| px[0] < 128).count()
}

/// Each standard-14 family class draws its text.
#[test]
fn standard_fonts_draw_without_an_embedded_program() {
    for base in ["Helvetica", "Times-Bold", "Courier-Oblique"] {
        let font = format!("<< /Type /Font /Subtype /Type1 /BaseFont /{base} >>");
        assert!(ink(&font, "(AB)") > 40, "{base} draws");
    }
}

/// A `WinAnsiEncoding` code above 0x7F selects the glyph its name denotes
/// on the substitute: `quoteright` (0x92) is a small mark, `Eacute` (0xC9)
/// a capital letter.
#[test]
fn winansi_high_codes_resolve_by_name_on_the_substitute() {
    let font = "<< /Type /Font /Subtype /TrueType /BaseFont /Arial \
                /Encoding /WinAnsiEncoding >>";
    let quote = ink(font, "<92>");
    let eacute = ink(font, "<C9>");
    assert!(quote > 0, "quoteright draws");
    assert!(
        eacute > quote * 3,
        "Eacute ({eacute}) outweighs quoteright ({quote})"
    );
}
