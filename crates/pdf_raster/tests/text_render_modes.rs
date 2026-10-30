//! Text rendering modes (PDF §9.3.6): filled, stroked, filled-then-stroked,
//! invisible and clipping glyphs.
//!
//! The font is the plain TrueType fixture described in
//! `crates/font/src/tests/data/generate.py`; its `B` is a closed square
//! covering x 100..400, y 0..300 font units (1000 per em).  At 72 DPI one
//! pixel is one point; text is set at 100 pt from (50, 50) on a
//! 200 × 200 pt page, so the square spans device columns 60..90 and rows
//! 120..150, centred on (75, 135).

#[expect(dead_code, reason = "shared helpers; this test uses only PDF assembly")]
mod common;

use common::{assemble_pdf, stream_object};
use rasterrocket::{BackendPolicy, render_page_rgb};

const PLAIN_TRUETYPE: &[u8] = include_bytes!("../../font/src/tests/data/hint-assembled-plain.ttf");

/// Centre row of the square.
const MID_Y: u32 = 135;

/// One page whose content stream is `content`, with the fixture as `/F1`.
fn page(content: &[u8]) -> Vec<u8> {
    page_with(content, "", Vec::new())
}

/// One page whose content stream is `content`, with the fixture as `/F1`,
/// `resources` added to the resource dictionary and `extra` appended as
/// objects 8 onwards.
fn page_with(content: &[u8], resources: &str, extra: Vec<Vec<u8>>) -> Vec<u8> {
    let mut objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] \
             /Resources << /Font << /F1 5 0 R >> {resources} >> /Contents 4 0 R >>"
        )
        .into_bytes(),
        stream_object("", content),
        b"<< /Type /Font /Subtype /TrueType /BaseFont /Fixture /FirstChar 65 \
           /LastChar 66 /Widths [1000 1000] /FontDescriptor 6 0 R >>"
            .to_vec(),
        b"<< /Type /FontDescriptor /FontName /Fixture /Flags 32 \
           /FontBBox [0 -200 1000 800] /ItalicAngle 0 /Ascent 800 /Descent -200 \
           /CapHeight 700 /StemV 80 /FontFile2 7 0 R >>"
            .to_vec(),
        stream_object(
            &format!("/Length1 {}", PLAIN_TRUETYPE.len()),
            PLAIN_TRUETYPE,
        ),
    ];
    objects.extend(extra);
    assemble_pdf(&objects)
}

/// A rendered RGB page.
struct Page {
    width: u32,
    data: Vec<u8>,
}

impl Page {
    fn rgb(&self, x: u32, y: u32) -> [u8; 3] {
        let i = ((y * self.width + x) * 3) as usize;
        [self.data[i], self.data[i + 1], self.data[i + 2]]
    }

    /// Total ink: the sum over all bytes of `255 - value`.
    fn ink(&self) -> u64 {
        self.data.iter().map(|&v| u64::from(255 - v)).sum()
    }
}

/// Render the page whose content stream is `content` at 72 DPI on the CPU.
fn render(content: &str) -> Page {
    render_pdf(page(content.as_bytes()))
}

/// Render page 1 of `pdf` at 72 DPI on the CPU.
fn render_pdf(pdf: Vec<u8>) -> Page {
    let session = common::session(pdf, BackendPolicy::CpuOnly);
    let bmp = render_page_rgb(&session, 1, 1.0).expect("page renders");
    let width = bmp.width;
    let data = (0..bmp.height)
        .flat_map(|y| bmp.row(y).iter().flat_map(|p| [p.r, p.g, p.b]))
        .collect();
    Page { width, data }
}

const WHITE: [u8; 3] = [255, 255, 255];

fn is_dark(px: [u8; 3]) -> bool {
    px.iter().all(|&c| c < 96)
}

#[test]
fn stroke_mode_paints_the_outline_only() {
    let page = render("BT /F1 100 Tf 2 w 1 Tr 50 50 Td (B) Tj ET");
    assert_eq!(page.rgb(75, MID_Y), WHITE, "interior is not filled");
    assert!(is_dark(page.rgb(60, MID_Y)), "left edge stroked");
    assert!(is_dark(page.rgb(89, MID_Y)), "right edge stroked");
    assert!(is_dark(page.rgb(75, 120)), "top edge stroked");
    assert_eq!(page.rgb(55, MID_Y), WHITE, "nothing beyond the stroke");
}

#[test]
fn fill_stroke_mode_is_heavier_than_fill() {
    let fill = render("BT /F1 100 Tf 6 w 0 Tr 50 50 Td (B) Tj ET");
    let fill_stroke = render("BT /F1 100 Tf 6 w 2 Tr 50 50 Td (B) Tj ET");
    assert!(is_dark(fill.rgb(75, MID_Y)) && is_dark(fill_stroke.rgb(75, MID_Y)));
    // A 6 px stroke centred on the edge at column 60 reaches column 57.
    assert_eq!(fill.rgb(58, MID_Y), WHITE);
    assert!(is_dark(fill_stroke.rgb(58, MID_Y)));
    assert!(fill_stroke.ink() > fill.ink());
}

#[test]
fn stroke_uses_the_stroking_colour() {
    let page = render("BT /F1 100 Tf 0 0 1 rg 1 0 0 RG 6 w 2 Tr 50 50 Td (B) Tj ET");
    assert_eq!(
        page.rgb(75, MID_Y),
        [0, 0, 255],
        "fill is the non-stroking colour"
    );
    assert_eq!(
        page.rgb(60, MID_Y),
        [255, 0, 0],
        "stroke is the stroking colour"
    );
    assert_eq!(
        page.rgb(58, MID_Y),
        [255, 0, 0],
        "stroke lies outside the fill"
    );
}

/// Largest per-byte difference between two renders of one page size.
fn max_delta(a: &Page, b: &Page) -> u8 {
    a.data
        .iter()
        .zip(&b.data)
        .map(|(x, y)| x.abs_diff(*y))
        .max()
        .unwrap_or(0)
}

#[test]
fn stroke_width_is_measured_in_user_space() {
    // The same device-space glyph set three ways: through the font size,
    // through a text matrix scaling text space by 2, and through a CTM
    // scaling user space by 2.  The text matrix leaves a 2-unit line width
    // at 2 device pixels; the CTM makes a 1-unit width 2 pixels.
    let plain = render("BT /F1 100 Tf 2 w 1 Tr 50 50 Td (B) Tj ET");
    let text_scaled = render("BT /F1 50 Tf 2 w 1 Tr 2 0 0 2 50 50 Tm (B) Tj ET");
    let user_scaled = render("2 0 0 2 0 0 cm BT /F1 50 Tf 1 w 1 Tr 25 25 Td (B) Tj ET");
    assert!(max_delta(&plain, &text_scaled) <= 2);
    assert!(max_delta(&plain, &user_scaled) <= 2);
    assert_eq!(text_scaled.rgb(57, MID_Y), WHITE, "width not scaled by Tm");
}

#[test]
fn invisible_mode_paints_nothing() {
    let page = render("BT /F1 100 Tf 3 Tr 50 50 Td (B) Tj ET");
    assert_eq!(page.ink(), 0);
}

#[test]
fn clip_mode_restricts_later_painting_to_the_glyph() {
    let page = render("BT /F1 100 Tf 7 Tr 50 50 Td (B) Tj ET 0 0 1 rg 0 0 200 200 re f");
    assert_eq!(
        page.rgb(75, MID_Y),
        [0, 0, 255],
        "inside the glyph is painted"
    );
    assert_eq!(page.rgb(55, MID_Y), WHITE, "outside the glyph is clipped");
    assert_eq!(page.rgb(10, 10), WHITE, "outside the glyph is clipped");
}

#[test]
fn stroke_clip_mode_strokes_and_clips() {
    let page =
        render("BT /F1 100 Tf 1 0 0 RG 2 w 5 Tr 50 50 Td (B) Tj ET 0 0 1 rg 0 0 200 200 re f");
    assert_eq!(page.rgb(75, MID_Y), [0, 0, 255], "clip admits the interior");
    assert_eq!(page.rgb(55, MID_Y), WHITE, "clip excludes the exterior");
    // The stroke's outer half lies outside the clip and stays red.
    assert_eq!(
        page.rgb(59, MID_Y),
        [255, 0, 0],
        "stroke painted before the clip"
    );
}

#[test]
fn glyphs_filled_as_paths_are_one_object_under_a_soft_mask() {
    // At 2100 pt the em exceeds the glyph-bitmap limit, so the glyphs fill
    // as paths.  The character spacing sets the second square 200 units after the
    // first: the first alone covers columns 0..30, both cover 30..200.  A
    // luminosity mask of grey 0.5 makes the text half-opaque; overlapping
    // glyphs of one text object are one shape, so the overlap is no
    // darker than a single glyph (PDF 11.6.4.4).
    let mask_group = stream_object(
        "/Type /XObject /Subtype /Form /BBox [0 0 200 200] \
         /Group << /S /Transparency /CS /DeviceGray >>",
        b"0.5 g 0 0 200 200 re f",
    );
    let page = render_pdf(page_with(
        b"/GS0 gs 1 0 0 rg BT /F1 2100 Tf -1680 Tc -600 -200 Td (BB) Tj ET",
        "/ExtGState << /GS0 << /SMask << /S /Luminosity /G 8 0 R >> >> >>",
        vec![mask_group],
    ));
    let single = page.rgb(10, 100);
    let overlap = page.rgb(100, 100);
    assert_eq!(single[0], 255);
    assert!(
        (120..=136).contains(&single[1]),
        "one glyph is half-opaque: {single:?}"
    );
    assert_eq!(overlap, single, "the overlap is painted once");
}
