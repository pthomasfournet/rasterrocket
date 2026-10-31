//! A font without an embedded program draws with a substitute: the
//! installed metric-compatible face when one is found, else the base-14
//! face compiled into the library.  Text in a non-embedded font therefore
//! renders on every host, including one with no fonts installed.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use common::{dark_pixels, font_page, ink_span, render_gray, stream_object, text_page};

/// Dark pixels of [`text_page`] showing `text` in `font`.
fn ink(font: &str, text: &str) -> usize {
    dark_pixels(text_page(font, text)).len()
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
    let pdf = font_page(
        (64, 32),
        "<< /Type /Font /Subtype /TrueType /BaseFont /Ahem /FirstChar 65 /LastChar 66 \
         /Widths [600 600] /Encoding /WinAnsiEncoding /FontDescriptor 6 0 R >>",
        "BT /F1 24 Tf 4 8 Td (AB) Tj ET",
        &[
            b"<< /Type /FontDescriptor /FontName /Ahem /Flags 32 /FontBBox [0 -200 1000 800] \
              /ItalicAngle 0 /Ascent 800 /Descent -200 /CapHeight 800 /StemV 80 \
              /FontFile2 7 0 R >>"
                .to_vec(),
            stream_object("/Length1 100", &AHEM[..100]),
        ],
    );
    let dark = dark_pixels(pdf).len();
    assert!(dark > 40, "AB draws ({dark} dark pixels)");
}

/// A `Tf` naming a font resource the page does not define draws its text
/// in Helvetica and reports the name in the page diagnostics.
#[test]
fn missing_font_resource_draws_helvetica() {
    // `/F1` is Times, so the fallback is told apart from the page's own font.
    let times = "<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman >>";
    let page = render_gray(font_page(
        (64, 32),
        times,
        "BT /F9 24 Tf 4 8 Td (AB) Tj ET",
        &[],
    ));
    let helvetica = render_gray(text_page(
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        "(AB)",
    ));
    assert!(helvetica.pixels.iter().any(|&v| v < 128), "Helvetica draws");
    assert!(page.pixels == helvetica.pixels, "/F9 draws as Helvetica");
    assert_eq!(page.diagnostics.missing_fonts, ["F9"]);
}

/// A substitute's glyph is fitted to the advance the document specifies:
/// condensed when the advance is narrower than the glyph, centred in the
/// advance when wider.
#[test]
fn substitute_glyphs_are_fitted_to_the_pdf_widths() {
    let narrow = "<< /Type /Font /Subtype /TrueType /BaseFont /ArialNarrow \
                  /FirstChar 77 /LastChar 77 /Widths [300] /Encoding /WinAnsiEncoding \
                  /FontDescriptor << /Type /FontDescriptor /FontName /ArialNarrow /Flags 32 \
                  /FontBBox [0 -200 1000 900] /ItalicAngle 0 /Ascent 900 /Descent -200 \
                  /CapHeight 700 /StemV 80 >> >>";
    let natural = "<< /Type /Font /Subtype /TrueType /BaseFont /ArialNarrow \
                   /Encoding /WinAnsiEncoding >>";
    let (_, condensed) = ink_span(text_page(narrow, "(M)"));
    let (_, full) = ink_span(text_page(natural, "(M)"));
    assert!(full >= 15, "an unfitted sans M at 24 pt is wide: {full}");
    assert!(condensed <= 9, "M condensed to 300/1000 em: {condensed}");

    let wide = "<< /Type /Font /Subtype /TrueType /BaseFont /ArialNarrow \
                /FirstChar 108 /LastChar 108 /Widths [1500] /Encoding /WinAnsiEncoding \
                /FontDescriptor << /Type /FontDescriptor /FontName /ArialNarrow /Flags 32 \
                /FontBBox [0 -200 1000 900] /ItalicAngle 0 /Ascent 900 /Descent -200 \
                /CapHeight 700 /StemV 80 >> >>";
    let (left_natural, _) = ink_span(text_page(natural, "(l)"));
    let (left_centred, _) = ink_span(text_page(wide, "(l)"));
    assert!(left_natural <= 8, "{left_natural}");
    assert!(
        left_centred >= left_natural + 12,
        "l centred in a 1500-unit advance: {left_centred} vs {left_natural}"
    );
}

/// `(width, height, dark pixel count)` of the bounding box of the ink
/// `ink` would count, for shape checks.
fn ink_box(font: &str, text: &str) -> (usize, usize, usize) {
    let dark = dark_pixels(text_page(font, text));
    let (Some(x0), Some(x1)) = (
        dark.iter().map(|p| p.0).min(),
        dark.iter().map(|p| p.0).max(),
    ) else {
        return (0, 0, 0);
    };
    let (y0, y1) = (
        dark.iter().map(|p| p.1).min(),
        dark.iter().map(|p| p.1).max(),
    );
    let (Some(y0), Some(y1)) = (y0, y1) else {
        unreachable!("x and y extents exist together");
    };
    (x1 - x0 + 1, y1 - y0 + 1, dark.len())
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

/// A non-embedded `ZapfDingbats` font draws dingbats: code `n` is `a73`, a
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
    let pdf = font_page(
        (64, 32),
        "<< /Type /Font /Subtype /Type0 /BaseFont /MS-Mincho /Encoding /90ms-RKSJ-H \
         /DescendantFonts [6 0 R] >>",
        "BT /F1 24 Tf 4 8 Td (AB) Tj ET",
        &[
            b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /MS-Mincho \
              /CIDSystemInfo << /Registry (Adobe) /Ordering (Japan1) /Supplement 2 >> \
              /FontDescriptor 7 0 R /DW 1000 /W [231 325 500] >>"
                .to_vec(),
            b"<< /Type /FontDescriptor /FontName /MS-Mincho /Flags 6 /FontBBox [0 -141 1000 859] \
              /ItalicAngle 0 /Ascent 859 /Descent -141 /CapHeight 700 /StemV 80 >>"
                .to_vec(),
        ],
    );
    let dark = dark_pixels(pdf).len();
    assert!(dark > 40, "AB draws ({dark} dark pixels)");
}
