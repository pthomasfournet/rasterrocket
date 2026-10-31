//! A line width of 0 strokes the thinnest line the device can draw, one
//! pixel wide (ISO 32000-2 §8.4.3.2), including the sides of a rectangle of
//! zero width or height.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use common::{one_page, session, stream_object};
use rasterrocket::{BackendPolicy, render_page_rgb};

/// A 100 × 100 pt page, rendered at one pixel per point.
const SIZE: u32 = 100;

/// The pixels of a white page running `content` that are not white, as
/// `(column, row)` in device space (origin top-left).
fn marked(content: &str) -> Vec<(u32, u32)> {
    let pdf = one_page(
        &format!("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {SIZE} {SIZE}] /Contents 4 0 R >>"),
        vec![stream_object("", content.as_bytes())],
    );
    let bmp = render_page_rgb(&session(pdf, BackendPolicy::CpuOnly), 1, 1.0).expect("render");
    let data = bmp.data();
    let mut out = Vec::new();
    for row in 0..SIZE {
        for col in 0..SIZE {
            let i = row as usize * bmp.stride + col as usize * 3;
            if data[i..i + 3] != [255, 255, 255] {
                assert_eq!(&data[i..i + 3], &[0, 0, 0], "hairline pixel ({col}, {row})");
                out.push((col, row));
            }
        }
    }
    out
}

/// A stroked rectangle of zero width is a one-pixel-wide vertical hairline
/// over its height, the same pixels as the stroked line along its side.
#[test]
fn zero_width_rectangle_strokes_a_one_pixel_hairline() {
    let rect = marked("0 w 20 20 0 60 re S");
    assert_eq!(rect, marked("0 w 20 20 m 20 80 l S"));
    assert!(rect.iter().all(|&(col, _)| col == 20), "{rect:?}");
    assert_eq!(rect.len(), 61, "rows 20 through 80");
}

/// A stroked rectangle of zero height is a one-pixel-high horizontal
/// hairline.
#[test]
fn zero_height_rectangle_strokes_a_one_pixel_hairline() {
    let rect = marked("0 w 20 50 60 0 re S");
    assert_eq!(rect, marked("0 w 20 50 m 80 50 l S"));
    assert!(rect.iter().all(|&(_, row)| row == SIZE - 50), "{rect:?}");
    assert_eq!(rect.len(), 61, "columns 20 through 80");
}
