//! The font is a graphics-state parameter (ISO 32000-1 §9.3.1): `Tf`
//! selects a font dictionary through the resources in effect where it runs,
//! and the selection holds, by identity, until the next `Tf`.  A form
//! `XObject` with its own resources may bind a page's font name to another
//! dictionary, and a form that shows text without a `Tf` uses the font in
//! effect at its `Do`.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use common::{session, stream_object, with_page_tree};
use rasterrocket::{BackendPolicy, render_page_rgb};

/// Ahem: every printing glyph is a full em square; `space` is blank.
const AHEM: &[u8] = include_bytes!("../../font_test_data/test_data/ttf/ahem.ttf");

/// A nonsymbolic `TrueType` font drawing code `a` as the glyph `glyph`,
/// with `FontDescriptor` object 8.
fn ahem_font(glyph: &str) -> Vec<u8> {
    format!(
        "<< /Type /Font /Subtype /TrueType /BaseFont /Ahem /FirstChar 97 /LastChar 97 \
         /Widths [1000] /Encoding << /Type /Encoding /BaseEncoding /WinAnsiEncoding \
         /Differences [97 /{glyph}] >> /FontDescriptor 8 0 R >>"
    )
    .into_bytes()
}

/// Render a 32 × 16 pt page at one pixel per point whose content is
/// `page`, with `/F1` bound to an `a`-as-`X` (painted square) font and
/// `/Fm0` a form, drawn at `x = 16`, of content `form` whose own resources
/// bind `form_name` to an `a`-as-`space` (blank) font.  Report whether each
/// half was painted.
fn painted_halves(page: &str, form_name: &str, form: &str) -> (bool, bool) {
    let objects = vec![
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 32 16] \
          /Resources << /Font << /F1 6 0 R >> /XObject << /Fm0 5 0 R >> >> \
          /Contents 4 0 R >>"
            .to_vec(),
        stream_object("", format!("{page} q 1 0 0 1 16 0 cm /Fm0 Do Q").as_bytes()),
        stream_object(
            format!(
                "/Type /XObject /Subtype /Form /BBox [0 0 16 16] \
                 /Resources << /Font << /{form_name} 7 0 R >> >>"
            )
            .as_str(),
            form.as_bytes(),
        ),
        ahem_font("X"),
        ahem_font("space"),
        b"<< /Type /FontDescriptor /FontName /Ahem /Flags 32 /FontBBox [0 -200 1000 800] \
          /ItalicAngle 0 /Ascent 800 /Descent -200 /CapHeight 800 /StemV 80 \
          /FontFile2 9 0 R >>"
            .to_vec(),
        stream_object("", AHEM),
    ];
    let s = session(with_page_tree(&objects), BackendPolicy::CpuOnly);
    let bmp = render_page_rgb(&s, 1, 1.0).expect("render page");
    let row = &bmp.data()[8 * bmp.stride..];
    (row[8 * 3] < 128, row[24 * 3] < 128)
}

/// One resource name bound to two dictionaries selects two fonts, even at
/// the same size.
#[test]
fn form_resources_rebind_a_page_font_name() {
    assert_eq!(
        painted_halves(
            "BT /F1 16 Tf 0 3 Td (a) Tj ET",
            "F1",
            "BT /F1 16 Tf 0 3 Td (a) Tj ET"
        ),
        (true, false)
    );
}

/// A form without a `Tf` shows text in the font selected at its `Do`,
/// though its own resources do not bind that font's name.
#[test]
fn form_without_tf_inherits_the_selected_font() {
    assert_eq!(
        painted_halves("BT /F1 16 Tf 0 3 Td (a) Tj ET", "F2", "BT 0 3 Td (a) Tj ET"),
        (true, true)
    );
}
