//! The OCR-preparation stages of `rasterrocket-camera` leave born-digital
//! pages alone: measured on a rendered page, each declines.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use camera::{Detected, GrayView, background_tile_px, measure_background};
use common::{render_one, text_page};
use rasterrocket::RasterOptions;

const DPI: u16 = 150;

/// A line of Helvetica text rendered to 8-bit gray at [`DPI`]:
/// `(width, height, samples)`.
fn rendered_text() -> (usize, usize, Vec<u8>) {
    let pdf = text_page(
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        "(Abg)",
    );
    let page = render_one(pdf, &RasterOptions::new(f32::from(DPI))).expect("page renders");
    let px = page.pixels.as_u8().expect("8-bit gray").to_vec();
    assert!(px.iter().any(|&v| v < 128), "the text draws");
    (page.width as usize, page.height as usize, px)
}

#[test]
fn born_digital_page_has_no_background_to_normalise() {
    let (w, h, px) = rendered_text();
    let img = GrayView::packed(&px, w, h);
    // At the resolution's tile size and at the smallest one, so the page
    // spans enough tiles for the white-background criterion to decide.
    for tile in [background_tile_px(u32::from(DPI)), 8] {
        let got = measure_background(img, tile).expect("valid image");
        assert!(matches!(got, Detected::Declined(_)), "tile {tile}: {got:?}");
    }
    let Detected::Declined(reason) = measure_background(img, 8).expect("valid image") else {
        unreachable!("declined above");
    };
    assert!(reason.contains("already white"), "{reason}");
}
