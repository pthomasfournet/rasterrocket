//! `/UserUnit` (ISO 32000-2 §14.11.2) sets the size of one user-space unit
//! in multiples of 1/72 inch.  A render at a given scale therefore sizes
//! the bitmap by `UserUnit` and maps the content onto all of it.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use common::{one_page, session, stream_object};
use rasterrocket::{
    BackendPolicy, RasterOptions, RenderParams, RenderRegion, render_page_gray, render_page_rgb,
    render_page_rgb_with,
};

const BLACK: [u8; 3] = [0, 0, 0];
const WHITE: [u8; 3] = [255, 255, 255];

/// A 200 × 100 pt page with `/UserUnit 2`: black over the whole box, with
/// a white 20 × 10 unit square in the box's upper-right corner.
fn page() -> Vec<u8> {
    one_page(
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /UserUnit 2 /Contents 4 0 R >>",
        vec![stream_object(
            "",
            b"0 g 0 0 200 100 re f 1 g 180 90 20 10 re f",
        )],
    )
}

fn px(bmp: &raster::Bitmap<color::Rgb8>, x: u32, y: u32) -> [u8; 3] {
    let row = bmp.row_bytes(y);
    let i = x as usize * 3;
    [row[i], row[i + 1], row[i + 2]]
}

#[test]
fn user_unit_scales_the_content_with_the_bitmap() {
    let s = session(page(), BackendPolicy::CpuOnly);
    let geom = s.page_geometry(1).expect("page geometry");
    assert!((geom.user_unit - 2.0).abs() < f64::EPSILON);
    assert_eq!(geom.size_px(1.0, 1.0), (400, 200));

    let bmp = render_page_rgb(&s, 1, 1.0).expect("page renders");
    assert_eq!((bmp.width, bmp.height), (400, 200));
    // The black rectangle reaches every corner of the bitmap but the white
    // square's, which covers device x 360..400, y 0..20.
    assert_eq!(px(&bmp, 0, 199), BLACK);
    assert_eq!(px(&bmp, 399, 199), BLACK);
    assert_eq!(px(&bmp, 0, 0), BLACK);
    assert_eq!(px(&bmp, 359, 10), BLACK);
    assert_eq!(px(&bmp, 361, 10), WHITE);
    assert_eq!(px(&bmp, 399, 0), WHITE);
    assert_eq!(px(&bmp, 380, 21), BLACK);
}

#[test]
fn user_unit_applies_to_each_axis_scale_and_to_a_region() {
    let s = session(page(), BackendPolicy::CpuOnly);

    let wide = render_page_rgb_with(&s, 1, &RenderParams::new_xy(2.0, 0.5)).expect("anisotropic");
    assert_eq!((wide.width, wide.height), (800, 100));
    // The white square now covers device x 720..800, y 0..10.
    assert_eq!(px(&wide, 719, 5), BLACK);
    assert_eq!(px(&wide, 721, 5), WHITE);
    assert_eq!(px(&wide, 799, 11), BLACK);

    let whole = render_page_rgb(&s, 1, 1.0).expect("page renders");
    let region = RenderRegion::new(340, 0, 60, 40);
    let tile = render_page_rgb_with(&s, 1, &RenderParams::new(1.0).with_region(region))
        .expect("region renders");
    assert_eq!((tile.width, tile.height), (60, 40));
    for y in 0..40 {
        for x in 0..60 {
            assert_eq!(px(&tile, x, y), px(&whole, 340 + x, y), "pixel ({x}, {y})");
        }
    }
}

/// The page is 200 × 2 / 72 = 5.5̅ inches wide; rendered at 72 dpi it is 400
/// pixels, 72 per physical inch, which is the resolution the page reports.
#[test]
fn reported_dpi_is_pixels_per_physical_inch() {
    let s = session(page(), BackendPolicy::CpuOnly);
    let mut opts = RasterOptions::default();
    opts.dpi = 72.0;
    let page = render_page_gray(&s, 1, &opts).expect("page renders");
    assert_eq!(page.width, 400);
    let inches = 200.0 * 2.0 / 72.0;
    assert!((f64::from(page.width) / inches - f64::from(page.dpi)).abs() < 1e-9);
}
