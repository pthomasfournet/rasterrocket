//! A simple font whose embedded CFF program carries a custom built-in
//! encoding, and whose PDF `Encoding` names no base encoding, resolves its
//! codes through that built-in encoding.
//!
//! The program's glyph names (`C65`, …) are not standard names, so a code
//! resolved through `StandardEncoding` (`A` for code 65) finds no glyph.

#[expect(dead_code, reason = "shared helpers; this test uses only PDF assembly")]
mod common;

use common::{assemble_pdf, stream_object};
use rasterrocket::{RasterOptions, RenderedPage, raster_pdf_from_bytes};

/// 72 DPI: one pixel per point on the 100 × 100 pt page.
const DPI: f32 = 72.0;

/// A CFF INDEX (CFF spec §5) with 2-byte offsets.
fn cff_index(items: &[&[u8]]) -> Vec<u8> {
    let count = u16::try_from(items.len()).expect("small index");
    let mut out = count.to_be_bytes().to_vec();
    if items.is_empty() {
        return out;
    }
    out.push(2);
    let mut offset: u16 = 1;
    out.extend_from_slice(&offset.to_be_bytes());
    for item in items {
        offset += u16::try_from(item.len()).expect("small item");
        out.extend_from_slice(&offset.to_be_bytes());
    }
    for item in items {
        out.extend_from_slice(item);
    }
    out
}

/// A DICT operand in the fixed-size 5-byte integer form, so the Top DICT's
/// length does not depend on the offsets it holds.
fn dict_int(value: usize) -> [u8; 5] {
    let mut out = [29; 5];
    out[1..].copy_from_slice(&i32::try_from(value).expect("small offset").to_be_bytes());
    out
}

/// Length of the Top DICT: three offset operators (`charset`, `Encoding`,
/// `CharStrings`) of one operand each, and `Private` with two operands.
const TOP_DICT_LEN: usize = 6 + 6 + 6 + 11;

/// A bare name-keyed CFF program with glyphs `.notdef` (empty) and `C65`
/// (a filled 500 × 500 unit square at (100, 100)) and a custom format-0
/// encoding that assigns code 65 to `C65`.
fn cff_with_custom_encoding() -> Vec<u8> {
    let header = [1u8, 0, 4, 2];
    let name_index = cff_index(&[b"Synth"]);
    // Custom strings start at SID 391, after the 391 standard strings.
    let string_index = cff_index(&[b"C65"]);
    let global_subrs = cff_index(&[]);
    // Charset format 0: the SID of every glyph after `.notdef`.
    let charset = [0u8, 0x01, 0x87];
    // Encoding format 0: one code, assigned to glyph 1.
    let encoding = [0u8, 1, 65];
    // Type 2 charstrings: `endchar`; and
    // `100 100 rmoveto 500 0 0 500 -500 0 rlineto endchar`.
    let notdef: &[u8] = &[14];
    let square: &[u8] = &[
        239, 239, 21, 248, 136, 139, 139, 248, 136, 252, 136, 139, 5, 14,
    ];
    let charstrings = cff_index(&[notdef, square]);
    // Private DICT: `0 defaultWidthX`.
    let private = [139u8, 20];

    // Top DICT: charset (15), Encoding (16), CharStrings (17), Private (18).
    let top_index_len = 2 + 1 + 2 * 2 + TOP_DICT_LEN;
    let charset_at =
        header.len() + name_index.len() + top_index_len + string_index.len() + global_subrs.len();
    let encoding_at = charset_at + charset.len();
    let charstrings_at = encoding_at + encoding.len();
    let private_at = charstrings_at + charstrings.len();
    let mut top = Vec::with_capacity(TOP_DICT_LEN);
    top.extend_from_slice(&dict_int(charset_at));
    top.push(15);
    top.extend_from_slice(&dict_int(encoding_at));
    top.push(16);
    top.extend_from_slice(&dict_int(charstrings_at));
    top.push(17);
    top.extend_from_slice(&dict_int(private.len()));
    top.extend_from_slice(&dict_int(private_at));
    top.push(18);
    let top_index = cff_index(&[&top]);
    assert_eq!(top_index.len(), top_index_len);

    [
        &header[..],
        &name_index,
        &top_index,
        &string_index,
        &global_subrs,
        &charset,
        &encoding,
        &charstrings,
        &private,
    ]
    .concat()
}

/// One page showing code 65 at 100 pt in the CFF font, with `encoding`
/// spliced into the font dictionary (empty for none).
fn page_showing_code_65(encoding: &str) -> Vec<u8> {
    let content = b"BT /F1 100 Tf 0 0 Td (A) Tj ET";
    let objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
           /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
            .to_vec(),
        stream_object("", content),
        format!(
            "<< /Type /Font /Subtype /Type1 /BaseFont /Synth /FirstChar 65 \
             /LastChar 65 /Widths [700] {encoding} /FontDescriptor 6 0 R >>"
        )
        .into_bytes(),
        b"<< /Type /FontDescriptor /FontName /Synth /Flags 4 \
           /FontBBox [0 0 700 700] /ItalicAngle 0 /Ascent 700 /Descent 0 \
           /CapHeight 700 /StemV 80 /FontFile3 7 0 R >>"
            .to_vec(),
        stream_object("/Subtype /Type1C", &cff_with_custom_encoding()),
    ];
    assemble_pdf(&objects)
}

fn render(pdf: Vec<u8>) -> RenderedPage {
    let opts = RasterOptions::new(DPI);
    let mut pages: Vec<_> = raster_pdf_from_bytes(pdf, &opts).collect();
    assert_eq!(pages.len(), 1, "one page expected");
    pages.remove(0).1.expect("page renders")
}

/// Gray value at PDF user-space `(x, y)` (origin bottom-left).
fn gray_at(page: &RenderedPage, x: u32, y: u32) -> u8 {
    let row = page.height - 1 - y;
    page.pixels[(row * page.width + x) as usize]
}

/// The glyph square spans user space 10–60 pt on both axes.
fn assert_square_painted(page: &RenderedPage, what: &str) {
    assert_eq!(gray_at(page, 35, 35), 0, "{what}: glyph interior is inked");
    assert_eq!(gray_at(page, 80, 80), 255, "{what}: outside stays white");
}

#[test]
fn encoding_without_base_uses_builtin_encoding() {
    let page = render(page_showing_code_65(
        "/Encoding << /Type /Encoding /Differences [] >>",
    ));
    assert_square_painted(&page, "Encoding dict without BaseEncoding");
}

#[test]
fn absent_encoding_uses_builtin_encoding() {
    let page = render(page_showing_code_65(""));
    assert_square_painted(&page, "no Encoding entry");
}

#[test]
fn differences_override_the_builtin_encoding() {
    // A Differences entry naming a glyph the program lacks wins over the
    // built-in encoding: the code shows `.notdef`, which is empty.
    let page = render(page_showing_code_65(
        "/Encoding << /Type /Encoding /Differences [65 /Missing] >>",
    ));
    assert_eq!(gray_at(&page, 35, 35), 255);
}

#[test]
fn explicit_base_encoding_is_honoured() {
    // `/StandardEncoding` names code 65 `A`, which the program lacks.
    let page = render(page_showing_code_65("/Encoding /StandardEncoding"));
    assert_eq!(gray_at(&page, 35, 35), 255);
}
