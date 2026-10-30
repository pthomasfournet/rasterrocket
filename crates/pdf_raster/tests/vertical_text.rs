//! Vertical writing mode (ISO 32000-2 §9.7.4.3): a Type 0 font whose
//! `Encoding` `CMap` has `WMode` 1 (`Identity-V`, a predefined `-V`
//! `CMap`, or a stream saying so) stacks its glyphs downwards from the
//! pen, each displaced by its position vector, with `TJ` adjustments
//! along y.

#[expect(
    dead_code,
    reason = "shared helpers; this test uses PDF assembly and sessions"
)]
mod common;

use common::{assemble_pdf, session, stream_object};
use rasterrocket::{BackendPolicy, render_page_rgb};

/// Ahem: every printing glyph is a full em square (x 0..1000, y −200..800).
const AHEM: &[u8] = include_bytes!("../../font_test_data/test_data/ttf/ahem.ttf");

/// Three Ahem `X` squares (glyph 58) at 16 pt from (24, 56).
const THREE_SQUARES: &str = "BT /F1 16 Tf 24 56 Td <003A003A003A> Tj ET";

/// The `Encoding` of the font under test.
enum Encoding<'a> {
    /// A predefined `CMap` name.
    Named(&'a str),
    /// An embedded `CMap` stream: extra dictionary entries and body.
    Stream(&'a str, &'a [u8]),
}

/// Bounding box `(x0, y0, x1, y1)` (device pixels, y down, inclusive) and
/// dark pixel count of `content` shown in a `CIDFontType2` Ahem with
/// `encoding` and the extra CIDFont entries `cid_extra`, on a 48 × 64 pt
/// page at one pixel per point.
fn ink(
    encoding: &Encoding<'_>,
    cid_extra: &str,
    content: &str,
) -> ((usize, usize, usize, usize), usize) {
    let encoding_entry = match encoding {
        Encoding::Named(name) => format!("/{name}"),
        Encoding::Stream(..) => "9 0 R".to_owned(),
    };
    let mut objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 48 64] \
          /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
            .to_vec(),
        stream_object("", content.as_bytes()),
        format!(
            "<< /Type /Font /Subtype /Type0 /BaseFont /Ahem /Encoding {encoding_entry} \
             /DescendantFonts [6 0 R] >>"
        )
        .into_bytes(),
        format!(
            "<< /Type /Font /Subtype /CIDFontType2 /BaseFont /Ahem \
             /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> \
             /FontDescriptor 7 0 R /DW 1000 /CIDToGIDMap /Identity {cid_extra} >>"
        )
        .into_bytes(),
        b"<< /Type /FontDescriptor /FontName /Ahem /Flags 4 /FontBBox [0 -200 1000 800] \
          /ItalicAngle 0 /Ascent 800 /Descent -200 /CapHeight 800 /StemV 80 \
          /FontFile2 8 0 R >>"
            .to_vec(),
        stream_object("", AHEM),
    ];
    if let Encoding::Stream(dict, body) = encoding {
        objects.push(stream_object(&format!("/Type /CMap {dict}"), body));
    }
    let s = session(assemble_pdf(&objects), BackendPolicy::CpuOnly);
    let bmp = render_page_rgb(&s, 1, 1.0).expect("render page");
    let w = bmp.width as usize;
    let (mut x0, mut y0, mut x1, mut y1, mut dark) = (usize::MAX, usize::MAX, 0, 0, 0);
    for (i, px) in bmp.data().chunks(3).enumerate() {
        if px[0] < 128 {
            let (x, y) = (i % w, i / w);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
            dark += 1;
        }
    }
    ((x0, y0, x1, y1), dark)
}

/// Whether `encoding` lays [`THREE_SQUARES`] out as a column: x 16..32
/// (the pen x 24 less `vx = w0 / 2 = 8`), at least 46 pt tall.
fn is_column(encoding: &Encoding<'_>) -> bool {
    let ((x0, y0, x1, y1), _) = ink(encoding, "", THREE_SQUARES);
    (x0, x1) == (16, 31) && y1 - y0 >= 46
}

/// Three 16 pt squares shown with `Identity-V` form a column: each
/// glyph's horizontal origin is displaced by −v, with `vx = w0 / 2` and
/// `vy = 880 / 1000` by default, and the pen advances by `−1000 / 1000`
/// em.  The same string with `Identity-H` forms a row.
#[test]
fn identity_v_stacks_glyphs_downwards() {
    let ((x0, y0, x1, y1), dark) = ink(&Encoding::Named("Identity-V"), "", THREE_SQUARES);
    assert!((640..=800).contains(&dark), "{dark} dark pixels");
    assert_eq!((x0, x1), (16, 31), "column spans x {x0}..={x1}");
    assert!(y0 <= 10 && y1 >= 55, "column spans y {y0}..={y1}");
    // Horizontally the row starts at the pen (x 24) and runs off the page.
    let ((hx0, hy0, hx1, hy1), _) = ink(&Encoding::Named("Identity-H"), "", THREE_SQUARES);
    assert!(
        hx0 == 24 && hx1 == 47 && hy1 - hy0 <= 16,
        "row {hx0}..={hx1} × {hy0}..={hy1}"
    );
}

/// `W2` sets a glyph's vertical advance and position vector; `DW2` the
/// defaults.  An advance of −500 overlaps the squares by half.
#[test]
fn w2_and_dw2_set_the_vertical_advance() {
    let identity_v = Encoding::Named("Identity-V");
    let ((_, y0, _, y1), _) = ink(&identity_v, "/W2 [58 [-500 500 800]]", THREE_SQUARES);
    // Three glyphs 8 pt apart cover 16 + 2 × 8 = 32 pt.
    assert!(
        (30..=34).contains(&(y1 - y0 + 1)),
        "column height {}",
        y1 - y0 + 1
    );
    let ((_, y0, _, y1), _) = ink(&identity_v, "/DW2 [800 -500]", THREE_SQUARES);
    assert!(
        (30..=34).contains(&(y1 - y0 + 1)),
        "column height {}",
        y1 - y0 + 1
    );
}

/// A `TJ` adjustment in vertical mode moves the pen along y: `1000`
/// (one em) opens a gap of one square between the second and third glyph.
#[test]
fn tj_adjustments_apply_along_y_in_vertical_mode() {
    let text = "BT /F1 16 Tf 24 60 Td [<003A003A> 1000 <003A>] TJ ET";
    let ((_, y0, _, y1), dark) = ink(&Encoding::Named("Identity-V"), "", text);
    assert!((640..=800).contains(&dark), "{dark} dark pixels");
    // Two squares, a one-em gap, one square: 64 pt from the top, the
    // last square clipped by the page edge.
    assert!(y1 - y0 + 1 >= 56, "column height {}", y1 - y0 + 1);
}

/// A predefined `-V` `CMap` and an embedded stream with `/WMode 1` (in
/// its dictionary or its body) select vertical mode as `Identity-V`
/// does; their `-H` counterparts do not.
#[test]
fn writing_mode_comes_from_the_cmap() {
    assert!(is_column(&Encoding::Named("Identity-V")));
    assert!(!is_column(&Encoding::Named("Identity-H")));
    let body: &[u8] =
        b"1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n1 begincidrange\n<003A> <003A> 58\nendcidrange\n";
    assert!(is_column(&Encoding::Stream("/WMode 1", body)));
    assert!(!is_column(&Encoding::Stream("/WMode 0", body)));
    let vertical_body: &[u8] = b"/WMode 1 def\n1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n1 begincidrange\n<003A> <003A> 58\nendcidrange\n";
    assert!(is_column(&Encoding::Stream("", vertical_body)));
    // 90ms-RKSJ-V inherits its mappings from 90ms-RKSJ-H: the one-byte
    // code 0x41 is CID 264, an Ahem square, and the mode is vertical.
    let ((x0, _, x1, _), dark) = ink(
        &Encoding::Named("90ms-RKSJ-V"),
        "",
        "BT /F1 16 Tf 24 56 Td <4141> Tj ET",
    );
    assert!(
        (450..=550).contains(&dark) && (x0, x1) == (16, 31),
        "{x0}..={x1}, {dark}"
    );
}
