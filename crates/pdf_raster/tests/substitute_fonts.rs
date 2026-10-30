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
use rasterrocket::{BackendPolicy, RasterOptions, raster_pdf_from_bytes, render_page_rgb};

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

/// A font whose embedded program is unusable (here the first 100 bytes of
/// a TrueType file) draws with a substitute instead of losing its text.
#[test]
fn corrupt_embedded_program_draws_a_substitute() {
    const AHEM: &[u8] = include_bytes!("../../font_test_data/test_data/ttf/ahem.ttf");
    let objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 64 32] \
          /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
            .to_vec(),
        stream_object("", b"BT /F1 24 Tf 4 8 Td (AB) Tj ET"),
        b"<< /Type /Font /Subtype /TrueType /BaseFont /Ahem /FirstChar 65 /LastChar 66 \
          /Widths [600 600] /Encoding /WinAnsiEncoding /FontDescriptor 6 0 R >>"
            .to_vec(),
        b"<< /Type /FontDescriptor /FontName /Ahem /Flags 32 /FontBBox [0 -200 1000 800] \
          /ItalicAngle 0 /Ascent 800 /Descent -200 /CapHeight 800 /StemV 80 \
          /FontFile2 7 0 R >>"
            .to_vec(),
        stream_object("/Length1 100", &AHEM[..100]),
    ];
    let s = session(assemble_pdf(&objects), BackendPolicy::CpuOnly);
    let bmp = render_page_rgb(&s, 1, 1.0).expect("render page");
    assert!(bmp.data().chunks(3).filter(|px| px[0] < 128).count() > 40);
}

/// A `Tf` naming a font resource the page does not define draws its text
/// in Helvetica and reports the name in the page diagnostics.
#[test]
fn missing_font_resource_draws_helvetica() {
    let objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 64 32] \
          /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
            .to_vec(),
        stream_object("", b"BT /F9 24 Tf 4 8 Td (AB) Tj ET"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
    ];
    let opts = RasterOptions {
        dpi: 72.0,
        ..RasterOptions::default()
    };
    let (_, page) = raster_pdf_from_bytes(assemble_pdf(&objects), &opts)
        .next()
        .expect("one page");
    let page = page.expect("page renders");
    assert!(page.pixels.iter().filter(|&&v| v < 128).count() > 40);
    assert_eq!(page.diagnostics.missing_fonts, ["F9"]);
}

/// `(width, height, dark pixel count)` of the bounding box of the ink
/// `ink` would count, for shape checks.
fn ink_box(font: &str, text: &str) -> (usize, usize, usize) {
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
    let (mut x0, mut y0, mut x1, mut y1, mut dark) = (usize::MAX, usize::MAX, 0, 0, 0);
    for (i, px) in bmp.data().chunks(3).enumerate() {
        if px[0] < 128 {
            let (x, y) = (i % bmp.width as usize, i / bmp.width as usize);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
            dark += 1;
        }
    }
    if dark == 0 {
        return (0, 0, 0);
    }
    (x1 - x0 + 1, y1 - y0 + 1, dark)
}

/// A non-embedded Symbol font draws Greek: code `a` is `alpha`, not the
/// Latin letter the sans-serif substitute would draw.  A named base
/// encoding on the dictionary does not change that.
#[test]
fn symbol_font_draws_its_own_encoding() {
    let helvetica = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>";
    let symbol = "<< /Type /Font /Subtype /Type1 /BaseFont /Symbol >>";
    let symbol_winansi =
        "<< /Type /Font /Subtype /Type1 /BaseFont /Symbol /Encoding /WinAnsiEncoding >>";
    let latin_a = ink_box(helvetica, "(a)");
    let alpha = ink_box(symbol, "(a)");
    assert!(alpha.2 > 0, "alpha draws");
    assert_ne!(alpha, latin_a, "alpha is not the Latin a");
    assert_eq!(ink_box(symbol_winansi, "(a)"), alpha);
    // 0xE5 is `summation` in the Symbol encoding; WinAnsi's `aring` is
    // a small letter, so the capital-height operator outweighs it.
    let summation = ink_box(symbol, "<E5>");
    assert!(
        summation.1 > latin_a.1 * 3 / 2,
        "summation is capital height"
    );
}

/// A non-embedded ZapfDingbats font draws dingbats: code `n` is `a73`, a
/// filled square, whose ink fills its bounding box.
#[test]
fn zapf_dingbats_font_draws_dingbats() {
    let dingbats = "<< /Type /Font /Subtype /Type1 /BaseFont /ZapfDingbats >>";
    let (w, h, dark) = ink_box(dingbats, "(n)");
    assert!(w >= 10 && h >= 10, "square is {w}x{h}");
    assert!(dark * 10 >= w * h * 9, "{dark} dark of {w}x{h} box");
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

/// A composite font without a program, of the Adobe-Japan1 collection
/// and a Shift-JIS encoding, draws its Latin text through CID → Unicode
/// on whatever substitute serves it: `A` is CID 264, which no Latin
/// substitute carries as a glyph index.
#[test]
fn non_embedded_japanese_font_draws_latin_through_cid_to_unicode() {
    let objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 64 32] \
          /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
            .to_vec(),
        stream_object("", b"BT /F1 24 Tf 4 8 Td (AB) Tj ET"),
        b"<< /Type /Font /Subtype /Type0 /BaseFont /MS-Mincho /Encoding /90ms-RKSJ-H \
          /DescendantFonts [6 0 R] >>"
            .to_vec(),
        b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /MS-Mincho \
          /CIDSystemInfo << /Registry (Adobe) /Ordering (Japan1) /Supplement 2 >> \
          /FontDescriptor 7 0 R /DW 1000 /W [231 325 500] >>"
            .to_vec(),
        b"<< /Type /FontDescriptor /FontName /MS-Mincho /Flags 6 /FontBBox [0 -141 1000 859] \
          /ItalicAngle 0 /Ascent 859 /Descent -141 /CapHeight 700 /StemV 80 >>"
            .to_vec(),
    ];
    let s = session(assemble_pdf(&objects), BackendPolicy::CpuOnly);
    let bmp = render_page_rgb(&s, 1, 1.0).expect("render page");
    let dark = bmp.data().chunks(3).filter(|px| px[0] < 128).count();
    assert!(dark > 40, "AB draws ({dark} dark pixels)");
}
