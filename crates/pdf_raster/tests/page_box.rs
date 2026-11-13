//! `PageBox` selects the boundary box a page is measured and rendered on,
//! through a session's `SessionConfig::page_box` and through
//! `RasterOptions::page_box` for the one-call entry points.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use common::{one_page, render_one};
use rasterrocket::{BackendPolicy, PageBox, RasterOptions, SessionConfig, open_session_from_bytes};

/// A 200 × 100 point `MediaBox` with a 120 × 60 `CropBox` at (40, 20) and a
/// 100 × 50 `TrimBox` at (50, 25).
fn boxed_page() -> Vec<u8> {
    one_page(
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /CropBox [40 20 160 80] \
         /TrimBox [50 25 150 75] >>",
        Vec::new(),
    )
}

#[test]
fn a_session_measures_pages_on_its_box() {
    for (page_box, size, origin) in [
        (PageBox::Media, (200.0, 100.0), (0.0, 0.0)),
        (PageBox::Crop, (120.0, 60.0), (40.0, 20.0)),
        (PageBox::Trim, (100.0, 50.0), (50.0, 25.0)),
        // No `BleedBox`: the `CropBox`.
        (PageBox::Bleed, (120.0, 60.0), (40.0, 20.0)),
    ] {
        let mut config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
        config.page_box = page_box;
        let session = open_session_from_bytes(boxed_page(), &config).expect("opens");
        let geometry = session.page_geometry(1).expect("geometry");
        assert_eq!(
            (geometry.width_pts, geometry.height_pts),
            size,
            "{page_box:?}"
        );
        assert_eq!(
            (geometry.origin_x, geometry.origin_y),
            origin,
            "{page_box:?}"
        );
    }
}

#[test]
fn raster_options_select_the_box_of_the_one_call_entry_points() {
    let mut opts = RasterOptions::new(72.0);
    assert_eq!(opts.page_box, PageBox::Crop, "the default");
    let crop = render_one(boxed_page(), &opts).expect("renders");
    opts.page_box = PageBox::Media;
    let media = render_one(boxed_page(), &opts).expect("renders");
    assert_eq!((crop.width, crop.height), (120, 60));
    assert_eq!((media.width, media.height), (200, 100));
}
