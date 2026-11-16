//! An image whose device extent along an axis is below one pixel draws
//! over the one pixel containing that axis's centre, holding the mean of
//! its samples along the axis, even when it covers no pixel centre.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use common::{one_page, render_gray, stream_object};

/// The page side in points, rendered at one pixel per point.
const SIDE: u32 = 20;

/// The pixels of a white 20 × 20 pt page running `content` that are not
/// white, as `(column, row, gray)` with the origin at the top left.
fn marks(content: &[u8]) -> Vec<(u32, u32, u8)> {
    let pdf = one_page(
        &format!("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {SIDE} {SIDE}] /Contents 4 0 R >>"),
        vec![stream_object("", content)],
    );
    let page = render_gray(pdf);
    let gray = page.pixels.as_u8().expect("8-bit gray");
    (0..SIDE)
        .flat_map(|row| (0..SIDE).map(move |col| (col, row)))
        .filter_map(|(col, row)| {
            let g = gray[(row * SIDE + col) as usize];
            (g != 255).then_some((col, row, g))
        })
        .collect()
}

/// `content` with the inline 1 × 1 stencil mask that paints its sample
/// drawn through the matrix `cm`.
fn dot_mask(cm: &str) -> Vec<u8> {
    let mut content = format!("0 g q {cm} cm BI /W 1 /H 1 /BPC 1 /IM true ID ").into_bytes();
    content.extend_from_slice(b"\x00 EI Q");
    content
}

/// Every pixel of columns `cols` and rows `rows`, at `gray`.
fn block(cols: std::ops::Range<u32>, rows: std::ops::Range<u32>, gray: u8) -> Vec<(u32, u32, u8)> {
    rows.flat_map(|row| cols.clone().map(move |col| (col, row, gray)))
        .collect()
}

/// A mask 0.6 px wide over x 10.6 .. 11.2, between the pixel centres 10.5
/// and 11.5, paints column 10, which holds its centre 10.9.
#[test]
fn a_mask_between_two_pixel_centres_paints_the_pixel_holding_its_centre() {
    assert_eq!(
        marks(&dot_mask("0.6 0 0 10 10.6 5")),
        block(10..11, 5..15, 0)
    );
}

/// Drawn right to left over the same span, it paints the same column.
#[test]
fn a_flipped_thin_mask_paints_the_same_pixel() {
    assert_eq!(
        marks(&dot_mask("-0.6 0 0 10 11.2 5")),
        block(10..11, 5..15, 0)
    );
}

/// A mask 0.6 px high over y 10.6 .. 11.2 paints the row holding y 10.9,
/// device row 9.
#[test]
fn a_mask_between_two_pixel_rows_paints_the_row_holding_its_centre() {
    assert_eq!(
        marks(&dot_mask("10 0 0 0.6 5 10.6")),
        block(5..15, 9..10, 0)
    );
    // Turned a quarter, its columns run up the page over the same span.
    assert_eq!(
        marks(&dot_mask("0 0.6 -10 0 15 10.6")),
        block(5..15, 9..10, 0)
    );
}

/// A six-sample stencil drawn 0.6 px wide paints its one pixel with the
/// mean of its row: `011110` paints two samples in six, gray 170.
#[test]
fn a_thin_stencil_paints_the_mean_of_its_samples() {
    let mut content = b"0 g q 0.6 0 0 10 10.6 5 cm BI /W 6 /H 1 /BPC 1 /IM true ID ".to_vec();
    content.extend_from_slice(b"\x78 EI Q");
    assert_eq!(marks(&content), block(10..11, 5..15, 170));
}

/// An image with its own colours widens as a stencil does.
#[test]
fn a_thin_gray_image_paints_the_pixel_holding_its_centre() {
    let mut content = b"q 0.6 0 0 10 10.6 5 cm BI /W 1 /H 1 /BPC 8 /CS /G ID ".to_vec();
    content.extend_from_slice(b"\x40 EI Q");
    assert_eq!(marks(&content), block(10..11, 5..15, 0x40));
}

/// A stencil turned 45° and 0.42 px thick marks a pixel in every row it
/// crosses, instead of breaking up where it passes between pixel centres.
#[test]
fn a_thin_rotated_mask_marks_every_row_it_crosses() {
    let marked = marks(&dot_mask("10 10 -0.3 0.3 4 4"));
    // The band runs over y 4 .. 14.3, device rows 5 .. 16; its ends may
    // fall short of a centre.
    for row in 7..15 {
        assert!(
            marked.iter().any(|&(_, r, g)| r == row && g < 128),
            "row {row} is unmarked: {marked:?}"
        );
    }
}
