//! The text state parameters that shape a glyph run (ISO 32000-2 §9.3):
//! word spacing `Tw` and horizontal scaling `Tz`, measured on Ahem, whose
//! every printing glyph is a full em square.

#[expect(
    dead_code,
    reason = "shared helpers; this test uses PDF assembly and sessions"
)]
mod common;

use common::{assemble_pdf, session, stream_object};
use rasterrocket::{BackendPolicy, render_page_rgb};

const AHEM: &[u8] = include_bytes!("../../font_test_data/test_data/ttf/ahem.ttf");
/// Glyph indices in Ahem.
const SPACE: u32 = 3;
const SQUARE: u32 = 58;

/// `(left, width)` in device pixels of the ink `content` paints on a
/// 160 × 32 pt page at one pixel per point, with `/F1` bound to the font
/// dictionary `font` (which may reference objects `6 0 R` onwards from
/// `extra`).
fn ink_span(font: &str, extra: &[Vec<u8>], content: &str) -> (usize, usize) {
    let mut objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 160 32] \
          /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
            .to_vec(),
        stream_object("", content.as_bytes()),
        font.as_bytes().to_vec(),
    ];
    objects.extend_from_slice(extra);
    let s = session(assemble_pdf(&objects), BackendPolicy::CpuOnly);
    let bmp = render_page_rgb(&s, 1, 1.0).expect("render page");
    let width = bmp.width as usize;
    let columns: Vec<usize> = bmp
        .data()
        .chunks(3)
        .enumerate()
        .filter(|(_, px)| px[0] < 128)
        .map(|(i, _)| i % width)
        .collect();
    let left = columns.iter().copied().min().unwrap_or(0);
    let right = columns.iter().copied().max().map_or(0, |r| r + 1);
    (left, right - left)
}

/// A simple TrueType Ahem: objects 6 (descriptor) and 7 (program).
fn simple_ahem() -> (&'static str, Vec<Vec<u8>>) {
    let font = "<< /Type /Font /Subtype /TrueType /BaseFont /Ahem /FirstChar 32 /LastChar 65 \
                /Widths [1000 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1000] \
                /FontDescriptor 6 0 R /Encoding /WinAnsiEncoding >>";
    let extra = vec![
        b"<< /Type /FontDescriptor /FontName /Ahem /Flags 32 /FontBBox [0 -200 1000 800] \
          /ItalicAngle 0 /Ascent 800 /Descent -200 /CapHeight 800 /StemV 80 \
          /FontFile2 7 0 R >>"
            .to_vec(),
        stream_object("", AHEM),
    ];
    (font, extra)
}

/// A Type 0 Ahem whose `Encoding` is `encoding` (an entry value: a name
/// or `10 0 R` for the stream in `cmap`), with `A` (code/CID 0x41) a
/// square and code/CID 0x20 a space.  Objects 6–9 and, for a stream, 10.
fn composite_ahem(encoding: &str, cmap: Option<&[u8]>) -> (String, Vec<Vec<u8>>) {
    let mut map = vec![0u8; 2 * 0x42];
    map[2 * 0x20..2 * 0x20 + 2].copy_from_slice(&(SPACE as u16).to_be_bytes());
    map[2 * 0x41..2 * 0x41 + 2].copy_from_slice(&(SQUARE as u16).to_be_bytes());
    let font = format!(
        "<< /Type /Font /Subtype /Type0 /BaseFont /Ahem /Encoding {encoding} \
         /DescendantFonts [6 0 R] >>"
    );
    let mut extra = vec![
        b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /Ahem \
          /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> \
          /FontDescriptor 7 0 R /DW 1000 /CIDToGIDMap 9 0 R >>"
            .to_vec(),
        b"<< /Type /FontDescriptor /FontName /Ahem /Flags 4 /FontBBox [0 -200 1000 800] \
          /ItalicAngle 0 /Ascent 800 /Descent -200 /CapHeight 800 /StemV 80 \
          /FontFile2 8 0 R >>"
            .to_vec(),
        stream_object("", AHEM),
        stream_object("", &map),
    ];
    if let Some(body) = cmap {
        extra.push(stream_object("/Type /CMap", body));
    }
    (font, extra)
}

/// `Tw` widens the gap at a one-byte code 32 of a composite font, and
/// leaves a two-byte code 32 alone (PDF §9.3.3).
#[test]
fn word_spacing_applies_to_one_byte_code_32_only() {
    let one_byte = b"1 begincodespacerange\n<00> <FF>\nendcodespacerange\n\
                     1 begincidrange\n<00> <FF> 0\nendcidrange\n";
    let (font, extra) = composite_ahem("10 0 R", Some(one_byte));
    let plain = ink_span(&font, &extra, "BT /F1 16 Tf 8 8 Td <412041> Tj ET");
    let spaced = ink_span(&font, &extra, "BT /F1 16 Tf 8 8 Td 16 Tw <412041> Tj ET");
    assert_eq!(plain, (8, 48));
    assert_eq!(spaced, (8, 64));

    let (font, extra) = composite_ahem("/Identity-H", None);
    let plain = ink_span(&font, &extra, "BT /F1 16 Tf 8 8 Td <004100200041> Tj ET");
    let spaced = ink_span(
        &font,
        &extra,
        "BT /F1 16 Tf 8 8 Td 16 Tw <004100200041> Tj ET",
    );
    assert_eq!(plain, (8, 48));
    assert_eq!(spaced, (8, 48));
}

/// `Tw` applies to the byte 32 of a simple font.
#[test]
fn word_spacing_applies_to_simple_fonts() {
    let (font, extra) = simple_ahem();
    let plain = ink_span(font, &extra, "BT /F1 16 Tf 8 8 Td (A A) Tj ET");
    let spaced = ink_span(font, &extra, "BT /F1 16 Tf 8 8 Td 16 Tw (A A) Tj ET");
    assert_eq!(plain, (8, 48));
    assert_eq!(spaced, (8, 64));
}
