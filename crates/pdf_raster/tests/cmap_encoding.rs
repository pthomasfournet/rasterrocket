//! A Type 0 font's `Encoding` `CMap` splits each string into character
//! codes and maps them to CIDs (ISO 32000-2 §9.7.5, §9.7.6): a predefined
//! `CMap` by name, or an embedded stream with any mix of code lengths,
//! bytes outside UTF-8 and a `usecmap` parent.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use common::{session, stream_object, with_page_tree};
use rasterrocket::{BackendPolicy, render_page_rgb};

/// Ahem: every printing glyph is a full em square; `space` is blank.
const AHEM: &[u8] = include_bytes!("../../font_test_data/test_data/ttf/ahem.ttf");
/// Glyph indices in Ahem.
const SPACE: u32 = 3;
const SQUARE: u32 = 58;

/// The `Encoding` of the font under test.
enum Encoding<'a> {
    /// A predefined `CMap` name.
    Named(&'a str),
    /// An embedded `CMap` stream: extra dictionary entries and body.
    Stream(&'a str, &'a [u8]),
}

/// Render `text` (a hex string) in a `CIDFontType2` Ahem with `encoding`
/// and the `CIDToGIDMap` `cid_to_gid`, at 16 pt on a 128 × 32 pt page at
/// one pixel per point, and count the painted em squares.
fn squares(encoding: &Encoding<'_>, cid_to_gid: &[(u32, u32)], text: &str) -> usize {
    let max_cid = cid_to_gid.iter().map(|&(cid, _)| cid).max().unwrap_or(0);
    let mut map = vec![0u8; 2 * (max_cid as usize + 1)];
    for &(cid, gid) in cid_to_gid {
        let at = 2 * cid as usize;
        map[at..at + 2].copy_from_slice(&u16::try_from(gid).expect("gid").to_be_bytes());
    }
    let encoding_entry = match encoding {
        Encoding::Named(name) => format!("/{name}"),
        Encoding::Stream(..) => "10 0 R".to_owned(),
    };
    let mut objects = vec![
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 128 32] \
          /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
            .to_vec(),
        stream_object(
            "",
            format!("BT /F1 16 Tf 0 3.2 Td <{text}> Tj ET").as_bytes(),
        ),
        format!(
            "<< /Type /Font /Subtype /Type0 /BaseFont /Ahem /Encoding {encoding_entry} \
             /DescendantFonts [6 0 R] >>"
        )
        .into_bytes(),
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
    if let Encoding::Stream(dict, body) = encoding {
        objects.push(stream_object(&format!("/Type /CMap {dict}"), body));
    }
    let s = session(with_page_tree(&objects), BackendPolicy::CpuOnly);
    let bmp = render_page_rgb(&s, 1, 1.0).expect("render page");
    let dark = bmp.data().chunks(3).filter(|px| px[0] < 128).count();
    // Each square is 16 × 16 device pixels.
    (dark + 128) / 256
}

/// `90ms-RKSJ-H` reads `41 93FA` as one-byte `A` (CID 264) and two-byte
/// `日` (CID 3284); under the identity mapping neither CID is reached.
#[test]
fn predefined_cmap_maps_codes_to_cids() {
    let glyphs = [(264, SQUARE), (3284, SQUARE)];
    assert_eq!(
        squares(&Encoding::Named("90ms-RKSJ-H"), &glyphs, "4193FA"),
        2
    );
    // Identity-H reads the same bytes as `4193` and a dangling `FA`.
    let identity = [
        (264, SQUARE),
        (3284, SQUARE),
        (0x4193, SPACE),
        (0xFA, SPACE),
    ];
    assert_eq!(
        squares(&Encoding::Named("Identity-H"), &identity, "4193FA"),
        0
    );
}

/// A stream with one-byte and two-byte codespace ranges splits
/// `41 8140 42` into three codes.
#[test]
fn mixed_width_codespaces_split_the_string() {
    let cmap = Encoding::Stream(
        "",
        b"1 begincodespacerange\n<00> <80>\n<8140> <FFFF>\nendcodespacerange\n\
          1 begincidrange\n<41> <42> 1\nendcidrange\n\
          1 begincidchar\n<8140> 3\nendcidchar\n",
    );
    assert_eq!(
        squares(&cmap, &[(1, SQUARE), (2, SQUARE), (3, SPACE)], "41814042"),
        2
    );
    assert_eq!(
        squares(&cmap, &[(1, SPACE), (2, SPACE), (3, SQUARE)], "41814042"),
        1
    );
}

/// A stream holding bytes outside UTF-8 still parses; under the identity
/// fallback code `0041` would select CID 0x41, mapped blank here.
#[test]
fn non_utf8_stream_is_parsed() {
    let cmap = Encoding::Stream(
        "",
        b"% caf\xE9 \xFF\xFE\n1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n\
          1 begincidrange\n<0041> <0043> 1\nendcidrange\n",
    );
    let glyphs = [(1, SQUARE), (2, SQUARE), (0x41, SPACE), (0x42, SPACE)];
    assert_eq!(squares(&cmap, &glyphs, "00410042"), 2);
}

/// `usecmap` of `Identity-H` in the stream, or `/UseCMap` in its
/// dictionary, supplies the codespace and the identity of every code the
/// stream does not override.
#[test]
fn usecmap_inherits_the_parent_mapping() {
    let glyphs = [(4, SQUARE), (0x41, SPACE), (0x42, SQUARE)];
    let in_stream = Encoding::Stream(
        "",
        b"/Identity-H usecmap\n1 begincidchar\n<0041> 4\nendcidchar\n",
    );
    assert_eq!(squares(&in_stream, &glyphs, "00410042"), 2);
    let in_dict = Encoding::Stream(
        "/UseCMap /Identity-H",
        b"1 begincidchar\n<0041> 4\nendcidchar\n",
    );
    assert_eq!(squares(&in_dict, &glyphs, "00410042"), 2);
}
