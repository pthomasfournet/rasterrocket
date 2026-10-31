//! Type 3 fonts (ISO 32000-2 §9.6.5): glyphs are content streams run with
//! the font's own resources, `d1` descriptions are stencils in the text
//! fill colour, and the text rendering mode's clip component does not
//! apply to them.

#[expect(
    dead_code,
    reason = "shared helpers; this test uses PDF assembly and sessions"
)]
mod common;

use color::Rgb8;
use common::{assemble_pdf, session, stream_object};
use raster::bitmap::Bitmap;
use rasterrocket::{BackendPolicy, render_page_rgb};

/// Render a 64 × 32 pt page at one pixel per point whose `/F1` is the Type 3
/// font dictionary `font`; `extra` holds the objects numbered from 6 that
/// `font` references.
fn render(font: &str, extra: &[Vec<u8>], content: &str) -> Bitmap<Rgb8> {
    let mut objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 64 32] \
          /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
            .to_vec(),
        stream_object("", content.as_bytes()),
        font.as_bytes().to_vec(),
    ];
    objects.extend_from_slice(extra);
    let s = session(assemble_pdf(&objects), BackendPolicy::CpuOnly);
    render_page_rgb(&s, 1, 1.0).expect("render page")
}

/// The RGB value at device pixel `(x, y)`.
fn pixel(page: &Bitmap<Rgb8>, x: usize, y: usize) -> [u8; 3] {
    let at = (y * page.width as usize + x) * 3;
    let px = &page.data()[at..at + 3];
    [px[0], px[1], px[2]]
}

/// Number of dark pixels on the page.
fn ink(page: &Bitmap<Rgb8>) -> usize {
    page.data().chunks(3).filter(|px| px[0] < 128).count()
}

/// A Type 3 font whose `A` runs the `CharProc` `glyph` (object 6) over a
/// 1000-unit em; `resources` is the font's `/Resources` entry.
fn type3(glyph: &[u8], resources: &str) -> (String, Vec<Vec<u8>>) {
    let font = format!(
        "<< /Type /Font /Subtype /Type3 /FontBBox [0 0 1000 1000] \
         /FontMatrix [0.001 0 0 0.001 0 0] /CharProcs << /square 6 0 R >> \
         /Encoding << /Type /Encoding /Differences [65 /square] >> \
         /FirstChar 65 /LastChar 65 /Widths [1000] /Resources {resources} >>"
    );
    (font, vec![stream_object("", glyph)])
}

/// `A` at 16 pt from (8, 8) covers device pixels x 8..24, y 8..24.
const SHOW_A: &str = "BT 0 0 1 rg /F1 16 Tf 8 8 Td (A) Tj ET";

/// A `d1` description paints in the text fill colour whatever colour it
/// sets itself; a `d0` description keeps its own colour.
#[test]
fn d1_glyphs_are_stencils_in_the_fill_colour() {
    let (font, extra) = type3(
        b"1000 0 0 0 1000 1000 d1 1 0 0 rg 0 0 1000 1000 re f",
        "<< >>",
    );
    assert_eq!(pixel(&render(&font, &extra, SHOW_A), 16, 16), [0, 0, 255]);

    let (font, extra) = type3(b"1000 0 d0 1 0 0 rg 0 0 1000 1000 re f", "<< >>");
    assert_eq!(pixel(&render(&font, &extra, SHOW_A), 16, 16), [255, 0, 0]);
}

/// A `d1` description's strokes take the fill colour too.
#[test]
fn d1_strokes_take_the_fill_colour() {
    let (font, extra) = type3(
        b"1000 0 0 0 1000 1000 d1 0 1 0 RG 200 w 0 500 m 1000 500 l S",
        "<< >>",
    );
    assert_eq!(pixel(&render(&font, &extra, SHOW_A), 16, 16), [0, 0, 255]);
}

/// A `CharProc`'s `Do` resolves in the font's own `/Resources`: a form
/// `XObject` and an image mask held there both paint, in the fill colour.
#[test]
fn charprocs_use_the_font_resources() {
    let resources = "<< /XObject << /Fx 7 0 R /Mx 8 0 R >> >>";
    let form = stream_object(
        "/Type /XObject /Subtype /Form /BBox [0 0 1000 1000]",
        b"0 0 1000 1000 re f",
    );
    let mask = stream_object(
        "/Type /XObject /Subtype /Image /Width 2 /Height 2 /ImageMask true \
         /Decode [1 0] /BitsPerComponent 1",
        &[0xC0, 0xC0],
    );
    let (font, mut extra) = type3(b"1000 0 0 0 1000 1000 d1 /Fx Do", resources);
    extra.push(form.clone());
    extra.push(mask.clone());
    assert_eq!(pixel(&render(&font, &extra, SHOW_A), 16, 16), [0, 0, 255]);

    let (font, mut extra) = type3(
        b"1000 0 0 0 1000 1000 d1 q 1000 0 0 1000 0 0 cm /Mx Do Q",
        resources,
    );
    extra.push(form);
    extra.push(mask);
    assert_eq!(pixel(&render(&font, &extra, SHOW_A), 16, 16), [0, 0, 255]);
}

/// Codes outside `Differences` resolve through the base encoding —
/// `/Encoding /WinAnsiEncoding`, or `/BaseEncoding` with a partial
/// `Differences` — and a code without a `CharProc` still advances by its
/// `Widths` entry.
#[test]
fn base_encodings_name_charprocs() {
    let charprocs =
        "/CharProcs << /A 6 0 R /B 6 0 R >> /FirstChar 65 /LastChar 66 /Widths [1000 1000]";
    let glyph = stream_object("", b"1000 0 0 0 1000 1000 d1 0 0 1000 1000 re f");
    let show_ab = "BT /F1 16 Tf 8 8 Td (AB) Tj ET";
    let winansi = format!(
        "<< /Type /Font /Subtype /Type3 /FontBBox [0 0 1000 1000] \
         /FontMatrix [0.001 0 0 0.001 0 0] {charprocs} /Encoding /WinAnsiEncoding >>"
    );
    assert_eq!(
        ink(&render(&winansi, std::slice::from_ref(&glyph), show_ab)),
        2 * 256
    );

    let partial = format!(
        "<< /Type /Font /Subtype /Type3 /FontBBox [0 0 1000 1000] \
         /FontMatrix [0.001 0 0 0.001 0 0] {charprocs} \
         /Encoding << /BaseEncoding /WinAnsiEncoding /Differences [66 /A] >> >>"
    );
    assert_eq!(
        ink(&render(&partial, std::slice::from_ref(&glyph), show_ab)),
        2 * 256
    );

    // No `/Encoding` at all: `StandardEncoding` names the CharProcs.
    let standard = format!(
        "<< /Type /Font /Subtype /Type3 /FontBBox [0 0 1000 1000] \
         /FontMatrix [0.001 0 0 0.001 0 0] {charprocs} >>"
    );
    assert_eq!(
        ink(&render(&standard, std::slice::from_ref(&glyph), show_ab)),
        2 * 256
    );

    // `C` has a width but no CharProc: `A C A` paints two squares with a
    // gap, not two adjacent squares.
    let gap = "<< /Type /Font /Subtype /Type3 /FontBBox [0 0 1000 1000] \
         /FontMatrix [0.001 0 0 0.001 0 0] /CharProcs << /A 6 0 R >> \
         /FirstChar 65 /LastChar 67 /Widths [1000 1000 1000] /Encoding /WinAnsiEncoding >>"
        .to_string();
    let page = render(&gap, &[glyph], "BT /F1 16 Tf 8 8 Td (ACA) Tj ET");
    assert_eq!(ink(&page), 2 * 256);
    assert_eq!(pixel(&page, 16, 16), [0, 0, 0]);
    assert_eq!(pixel(&page, 32, 16), [255, 255, 255]);
    assert_eq!(pixel(&page, 48, 16), [0, 0, 0]);
}

/// The clip component of a rendering mode does not apply to Type 3 text:
/// mode 7 paints the glyph and leaves the clip alone; mode 3 is invisible.
#[test]
fn clip_modes_paint_type3_glyphs() {
    let (font, extra) = type3(b"1000 0 0 0 1000 1000 d1 0 0 1000 1000 re f", "<< >>");
    let painted = render(&font, &extra, "BT 7 Tr /F1 16 Tf 8 8 Td (A) Tj ET");
    assert_eq!(pixel(&painted, 16, 16), [0, 0, 0]);
    let after = render(
        &font,
        &extra,
        "BT 7 Tr /F1 16 Tf 8 8 Td (A) Tj ET 0 0 64 32 re f",
    );
    assert_eq!(ink(&after), 64 * 32, "the clip is unchanged");
    let invisible = render(&font, &extra, "BT 3 Tr /F1 16 Tf 8 8 Td (A) Tj ET");
    assert_eq!(ink(&invisible), 0);
}

/// A `d1` glyph whose content leaves its declared bounding box draws
/// whole: the box `[0 0 500 500]` holds a quarter of the 1000-unit square
/// it fills, and all of the square's 16 × 16 pixels are inked.
#[test]
fn d1_content_outside_its_box_draws_whole() {
    let (font, extra) = type3(b"1000 0 0 0 500 500 d1 0 0 1000 1000 re f", "<< >>");
    assert_eq!(ink(&render(&font, &extra, SHOW_A)), 16 * 16);
}

/// Repeated `d1` glyphs at the same pixel phase draw the same pixels,
/// each a 16 × 16 square in the fill colour.
#[test]
fn repeated_d1_glyphs_draw_alike() {
    let (font, extra) = type3(
        b"1000 0 0 0 1000 1000 d1 100 100 m 900 500 l 100 900 l f",
        "<< >>",
    );
    let page = render(&font, &extra, "BT 0 0 1 rg /F1 16 Tf 8 8 Td (AA) Tj ET");
    for y in 8..24 {
        for x in 8..24 {
            assert_eq!(pixel(&page, x, y), pixel(&page, x + 16, y), "({x}, {y})");
        }
    }
    let inked = ink(&page);
    assert!(
        inked > 2 * 60 && inked.is_multiple_of(2),
        "{inked} dark pixels"
    );
}

/// An image carrying its own colours inside a `d1` description paints
/// nothing, as pdftoppm and pdfium draw it: the description is a stencil
/// and may hold only image masks.  The same image in a `d0` description
/// paints in its own colour.
#[test]
fn d1_glyphs_skip_colour_images() {
    let image: &[u8] = b"1000 0 0 1000 0 0 cm BI /W 1 /H 1 /CS /RGB /BPC 8 ID \xff\x00\x00 EI";
    let d1 = [b"1000 0 0 0 1000 1000 d1 ".as_slice(), image].concat();
    let (font, extra) = type3(&d1, "<< >>");
    assert_eq!(ink(&render(&font, &extra, SHOW_A)), 0);

    let d0 = [b"1000 0 d0 ".as_slice(), image].concat();
    let (font, extra) = type3(&d0, "<< >>");
    assert_eq!(pixel(&render(&font, &extra, SHOW_A), 16, 16), [255, 0, 0]);
}
