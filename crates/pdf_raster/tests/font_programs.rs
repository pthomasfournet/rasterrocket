//! Embedded font programs whose glyphs need more than unhinted outlines of
//! a name-keyed or sfnt program: TrueType glyphs assembled by their
//! instructions, and CID-keyed Type 1 (`CIDFontType0` with `FontFile`).
//!
//! The fixtures are described in `crates/font/src/tests/data/generate.py`.
//! At 72 DPI one pixel is one point; text is set at 40 pt from (10, 30) on
//! a 200 × 100 pt page, so a glyph coordinate of `u` font units (1000 per
//! em) lands at pixel column `pen + 0.04 u`.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use common::{render_gray, stream_object, with_page_tree};
use rasterrocket::RenderedPage;

const HINT_ASSEMBLED_BY_NAME: &[u8] =
    include_bytes!("../../font/src/tests/data/hint-assembled-by-name.ttf");
const HINT_ASSEMBLED_BY_TABLES: &[u8] =
    include_bytes!("../../font/src/tests/data/hint-assembled-by-tables.ttf");
const HINT_ASSEMBLED_PLAIN: &[u8] =
    include_bytes!("../../font/src/tests/data/hint-assembled-plain.ttf");
const SYMBOLIC_MAC_ROMAN_NAMED: &[u8] =
    include_bytes!("../../font/src/tests/data/symbolic-mac-roman-named.ttf");
const CID_KEYED_TYPE1: &[u8] = include_bytes!("../../font/src/tests/data/cid-keyed-type1.cid");
const NO_CMAP: &[u8] = include_bytes!("../../font/src/tests/data/no-cmap.ttf");

const PAGE: &[u8] = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] \
    /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>";

/// One page showing `AB` in a simple TrueType font embedding `program`.
fn truetype_page(program: &[u8]) -> Vec<u8> {
    with_page_tree(&[
        PAGE.to_vec(),
        stream_object("", b"BT /F1 40 Tf 10 30 Td (AB) Tj ET"),
        b"<< /Type /Font /Subtype /TrueType /BaseFont /Fixture /FirstChar 65 \
           /LastChar 66 /Widths [1000 1000] /FontDescriptor 6 0 R >>"
            .to_vec(),
        b"<< /Type /FontDescriptor /FontName /Fixture /Flags 32 \
           /FontBBox [0 -200 1000 800] /ItalicAngle 0 /Ascent 800 /Descent -200 \
           /CapHeight 700 /StemV 80 /FontFile2 7 0 R >>"
            .to_vec(),
        stream_object(&format!("/Length1 {}", program.len()), program),
    ])
}

/// One page showing the one-byte `code` in a symbolic simple TrueType
/// font (`Flags` 4, no `Encoding`) embedding `program`.
fn symbolic_truetype_page(program: &[u8], code: u8) -> Vec<u8> {
    with_page_tree(&[
        PAGE.to_vec(),
        stream_object(
            "",
            format!("BT /F1 40 Tf 10 30 Td <{code:02X}> Tj ET").as_bytes(),
        ),
        format!(
            "<< /Type /Font /Subtype /TrueType /BaseFont /Fixture /FirstChar {code} \
             /LastChar {code} /Widths [1000] /FontDescriptor 6 0 R >>"
        )
        .into_bytes(),
        b"<< /Type /FontDescriptor /FontName /Fixture /Flags 4 \
           /FontBBox [0 -200 1000 800] /ItalicAngle 0 /Ascent 800 /Descent -200 \
           /CapHeight 700 /StemV 80 /FontFile2 7 0 R >>"
            .to_vec(),
        stream_object(&format!("/Length1 {}", program.len()), program),
    ])
}

/// One page showing CIDs 1–4 through `Identity-H` in a Type 0 font whose
/// `CIDFontType0` descendant embeds `program` as `FontFile`.
fn cid_type1_page(program: &[u8]) -> Vec<u8> {
    with_page_tree(&[
        PAGE.to_vec(),
        stream_object("", b"BT /F1 40 Tf 10 30 Td <0001000200030004> Tj ET"),
        b"<< /Type /Font /Subtype /Type0 /BaseFont /RRTest-CID-Identity-H \
           /Encoding /Identity-H /DescendantFonts [6 0 R] >>"
            .to_vec(),
        b"<< /Type /Font /Subtype /CIDFontType0 /BaseFont /RRTest-CID \
           /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> \
           /FontDescriptor 7 0 R /DW 1000 >>"
            .to_vec(),
        b"<< /Type /FontDescriptor /FontName /RRTest-CID /Flags 4 \
           /FontBBox [0 -100 1000 900] /ItalicAngle 0 /Ascent 900 /Descent -100 \
           /CapHeight 700 /StemV 80 /FontFile 8 0 R >>"
            .to_vec(),
        stream_object("", program),
    ])
}

/// Inclusive column runs that contain a pixel darker than mid-gray.
fn ink_columns(page: &RenderedPage) -> Vec<(u32, u32)> {
    let (w, h) = (page.width, page.height);
    let inked = |x: u32| (0..h).any(|y| page.pixels[(y * w + x) as usize] < 128);
    let mut runs = Vec::new();
    let mut start = None;
    for x in 0..=w {
        match (x < w && inked(x), start) {
            (true, None) => start = Some(x),
            (false, Some(s)) => {
                runs.push((s, x - 1));
                start = None;
            }
            _ => {}
        }
    }
    runs
}

/// Whether `runs` holds one run spanning about `lo..=hi` (one pixel of
/// slack at each end for coverage at the edges).
fn has_run(runs: &[(u32, u32)], lo: u32, hi: u32) -> bool {
    runs.iter()
        .any(|&(a, b)| a.abs_diff(lo) <= 1 && b.abs_diff(hi) <= 1)
}

#[test]
fn instruction_assembled_truetype_glyphs_render_assembled() {
    // `A` is two squares (x 100..400 and 600..900), `B` one square.
    for program in [HINT_ASSEMBLED_BY_NAME, HINT_ASSEMBLED_BY_TABLES] {
        let runs = ink_columns(&render_gray(truetype_page(program)));
        assert_eq!(runs.len(), 3, "{runs:?}");
        assert!(has_run(&runs, 14, 25), "first square of A: {runs:?}");
        assert!(has_run(&runs, 34, 45), "second square of A: {runs:?}");
        assert!(has_run(&runs, 54, 65), "B: {runs:?}");
    }
}

#[test]
fn ordinary_truetype_glyphs_stay_unhinted() {
    // Not recognised: `A`'s components overlap as stored.
    let runs = ink_columns(&render_gray(truetype_page(HINT_ASSEMBLED_PLAIN)));
    assert_eq!(runs.len(), 2, "{runs:?}");
    assert!(has_run(&runs, 14, 25) && has_run(&runs, 54, 65), "{runs:?}");
}

#[test]
fn symbolic_truetype_code_resolves_through_mac_roman_subtable() {
    // The only cmap subtable, (1,0), maps code 0x21 to `therefore`, whose
    // `post` name also yields a Unicode charmap keyed by U+2234.  Code 0x21
    // selects the square (x 100..400); the contourless `.notdef` leaves
    // the page blank.
    let runs = ink_columns(&render_gray(symbolic_truetype_page(
        SYMBOLIC_MAC_ROMAN_NAMED,
        0x21,
    )));
    assert_eq!(runs.len(), 1, "{runs:?}");
    assert!(has_run(&runs, 14, 25), "therefore: {runs:?}");
}

/// An embedded subset with neither a `cmap` table nor `post` names maps
/// each code to the glyph of that index: code 1 draws GID 1's square.
#[test]
fn truetype_without_cmap_or_names_uses_codes_as_glyph_indices() {
    let runs = ink_columns(&render_gray(symbolic_truetype_page(NO_CMAP, 1)));
    assert!(has_run(&runs, 14, 26), "{runs:?}");
}

#[test]
fn cid_keyed_type1_glyphs_render() {
    // CID 1 square at pen 10, CID 2 triangle at pen 50, CID 3 diamond at
    // pen 90; CID 4 has no glyph.
    let page = render_gray(cid_type1_page(CID_KEYED_TYPE1));
    let runs = ink_columns(&page);
    assert_eq!(runs.len(), 3, "{runs:?}");
    assert!(has_run(&runs, 14, 25), "CID 1: {runs:?}");
    assert!(has_run(&runs, 52, 67), "CID 2: {runs:?}");
    assert!(has_run(&runs, 102, 117), "CID 3: {runs:?}");
    assert!(page.diagnostics.has_vector_text);
}
