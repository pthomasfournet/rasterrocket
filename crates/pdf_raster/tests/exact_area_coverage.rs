//! Antialiased paths are covered by their exact area: a stroke centred on a
//! row boundary splits evenly between the two rows, and one centred on a
//! row covers that row alone, a stroke thinner than a pixel being widened
//! to one.
//!
//! The page is a 40 × 40 pt page with, from the top: a `1 w` line at
//! y = 30 (on the boundary of device rows 9 and 10), a `1 w` line at
//! y = 19.5 (centred on row 20), a `0.3 w` line at y = 9.5 (centred on row
//! 30) and a 1 pt high rectangle filled at y = 4.5 (which fill adjustment
//! snaps to row 35).

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use common::{one_page, session, stream_object};
use rasterrocket::{BackendPolicy, render_page_rgb};

/// The page's content stream.
const LINES: &[u8] = b"0 G 1 w 0 30 m 40 30 l S 0 19.5 m 40 19.5 l S \
    0.3 w 0 9.5 m 40 9.5 l S 0 g 0 4.5 40 1 re f";

/// The grey level (red channel) of rows 0..40 of column 20 of the page
/// rendered at one pixel per point.
fn column() -> Vec<u8> {
    let pdf = one_page(
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 40 40] /Contents 4 0 R >>",
        vec![stream_object("", LINES)],
    );
    let bmp = render_page_rgb(&session(pdf, BackendPolicy::CpuOnly), 1, 1.0).expect("render");
    assert_eq!((bmp.width, bmp.height), (40, 40));
    (0..40)
        .map(|row| bmp.data()[row * bmp.stride + 20 * 3])
        .collect()
}

#[test]
fn a_stroke_on_a_row_boundary_splits_evenly() {
    let col = column();
    let (above, below) = (col[9], col[10]);
    assert!(above.abs_diff(below) <= 1, "{above} vs {below}: {col:?}");
    assert!(above.abs_diff(127) <= 1, "{col:?}");
    assert_eq!((col[8], col[11]), (255, 255), "{col:?}");
}

#[test]
fn a_stroke_centred_on_a_row_covers_that_row_alone() {
    let col = column();
    assert_eq!(col[20], 0, "{col:?}");
    assert_eq!((col[19], col[21]), (255, 255), "{col:?}");
}

#[test]
fn a_thin_stroke_widened_to_a_pixel_covers_its_row_alone() {
    let col = column();
    assert_eq!(col[30], 0, "{col:?}");
    assert_eq!((col[29], col[31]), (255, 255), "{col:?}");
}

#[test]
fn an_adjusted_fill_covers_whole_rows() {
    let col = column();
    assert_eq!(col[35], 0, "{col:?}");
    assert_eq!((col[34], col[36]), (255, 255), "{col:?}");
}
