//! An inline image (`BI … ID … EI`) converts its samples through its colour
//! space as an image `XObject` does: a `Separation` or `DeviceN` space named
//! from the resources through its tint transform, an `Indexed` space through
//! its palette (inline palette string, abbreviated names, a named base).

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use common::{assert_rgb_near, session, stream_object, with_page_tree};
use rasterrocket::{BackendPolicy, render_page_rgb};

/// Page size in points; rendered at one pixel per point.
const W: u32 = 200;
const H: u32 = 100;

/// A page painting a 2 × 1 inline image with parameters `params` and
/// samples `data` over the whole page.  Resources: `/CS1` is
/// `[/Separation /Red /DeviceRGB]` (tint t ↦ (1, 1 − t, 1 − t)) and `/CS2`
/// is `[/DeviceN [/Cyan /Magenta /Yellow] /DeviceCMYK]` (c m y ↦ c m y 0).
fn rgb_of_inline_image(params: &str, data: &[u8]) -> [[u8; 3]; 2] {
    let mut content = format!("q {W} 0 0 {H} 0 0 cm BI /W 2 /H 1 {params} ID ").into_bytes();
    content.extend_from_slice(data);
    content.extend_from_slice(b"\nEI Q");
    let objects = vec![
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W} {H}] \
             /Resources << /ColorSpace << /CS1 5 0 R /CS2 7 0 R >> >> /Contents 4 0 R >>"
        )
        .into_bytes(),
        stream_object("", &content),
        b"[/Separation /Red /DeviceRGB 6 0 R]".to_vec(),
        b"<< /FunctionType 2 /Domain [0 1] /C0 [1 1 1] /C1 [1 0 0] /N 1 >>".to_vec(),
        b"[/DeviceN [/Cyan /Magenta /Yellow] /DeviceCMYK 8 0 R]".to_vec(),
        stream_object(
            "/FunctionType 4 /Domain [0 1 0 1 0 1] /Range [0 1 0 1 0 1 0 1]",
            b"{ 0 }",
        ),
    ];
    let s = session(with_page_tree(&objects), BackendPolicy::CpuOnly);
    let bmp = render_page_rgb(&s, 1, 1.0).expect("render page");
    let row = &bmp.data()[50 * bmp.stride..];
    let at = |x: usize| [row[x * 3], row[x * 3 + 1], row[x * 3 + 2]];
    [at(50), at(150)]
}

#[test]
fn inline_separation_image_uses_its_tint_transform() {
    let [full, half] = rgb_of_inline_image("/CS /CS1 /BPC 8", &[0xFF, 0x80]);
    assert_rgb_near(full, [255, 0, 0], 1, "tint 1");
    assert_rgb_near(half, [255, 127, 127], 1, "tint 0.5");
}

#[test]
fn inline_multi_colorant_devicen_image_uses_its_tint_transform() {
    // (0, 1, 1) and (1, 0, 1) through DeviceCMYK: red and green.
    let [red, green] = rgb_of_inline_image("/CS /CS2 /BPC 8", &[0, 255, 255, 255, 0, 255]);
    assert!(red[0] > 180 && red[1] < 60 && red[2] < 60, "red: {red:?}");
    assert!(green[1] > 120 && green[0] < 60, "green: {green:?}");
}

#[test]
fn inline_indexed_image_reads_its_inline_palette() {
    for params in [
        "/ColorSpace [/Indexed /DeviceRGB 1 <FF00000000FF>] /BitsPerComponent 8",
        "/CS [/I /RGB 1 <FF00000000FF>] /BPC 8",
    ] {
        let [first, second] = rgb_of_inline_image(params, &[0, 1]);
        assert_eq!(first, [255, 0, 0], "{params}");
        assert_eq!(second, [0, 0, 255], "{params}");
    }
}

#[test]
fn inline_indexed_image_over_a_named_separation() {
    let [full, half] = rgb_of_inline_image("/CS [/I /CS1 1 <FF80>] /BPC 8", &[0, 1]);
    assert_rgb_near(full, [255, 0, 0], 1, "palette tint 1");
    assert_rgb_near(half, [255, 127, 127], 1, "palette tint 0.5");
}
