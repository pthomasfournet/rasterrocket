//! A shading's colour at every point is the colour-space conversion of its
//! function's output there (ISO 32000-1 §8.7.4.5), whatever the colour
//! space (`DeviceN` with a multi-input tint transform included) and however
//! many stops a stitching function has.

#[expect(
    dead_code,
    reason = "shared helpers; this test uses PDF assembly and sessions"
)]
mod common;

use common::{assemble_pdf, session, stream_object};
use rasterrocket::{BackendPolicy, render_page_rgb};

/// Page size in points; rendered at one pixel per point.
const W: u32 = 200;
const H: u32 = 100;

/// `[/DeviceN [/Cyan /Magenta /Yellow] /DeviceCMYK <tint>]` whose Type 4
/// tint transform (object 5) maps `c m y` to `c m y 0`.
const CMY_DEVICEN: &str = "[/DeviceN [/Cyan /Magenta /Yellow] /DeviceCMYK 5 0 R]";

fn cmy_tint() -> Vec<u8> {
    stream_object(
        "/FunctionType 4 /Domain [0 1 0 1 0 1] /Range [0 1 0 1 0 1 0 1]",
        b"{ 0 }",
    )
}

/// Red (0, 1, 1) to green (1, 0, 0.59) in `CMY_DEVICEN`, wrapped in a
/// one-function stitching function.
const RED_TO_GREEN_CMY: &str = "<< /FunctionType 3 /Domain [0 1] /Bounds [] /Encode [0 1] \
     /Functions [<< /FunctionType 2 /Domain [0 1] /C0 [0 1 1] /C1 [1 0 0.59] /N 1 >>] >>";

/// `DeviceRGB` red → green → blue with the middle stop at 0.5.
const RED_GREEN_BLUE: &str = "<< /FunctionType 3 /Domain [0 1] /Bounds [0.5] /Encode [0 1 0 1] \
     /Functions [<< /FunctionType 2 /Domain [0 1] /C0 [1 0 0] /C1 [0 1 0] /N 1 >> \
                 << /FunctionType 2 /Domain [0 1] /C0 [0 1 0] /C1 [0 0 1] /N 1 >>] >>";

/// One `W` × `H` page running `content`, with shading `/Sh0` (`shading`),
/// colour space `/CS0` (`CMY_DEVICEN`) and the tint transform as object 5.
fn page(content: &str, shading: &str) -> Vec<u8> {
    let objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W} {H}] \
             /Resources << /Shading << /Sh0 6 0 R >> /ColorSpace << /CS0 {CMY_DEVICEN} >> >> \
             /Contents 4 0 R >>"
        )
        .into_bytes(),
        stream_object("", content.as_bytes()),
        cmy_tint(),
        shading.as_bytes().to_vec(),
    ];
    assemble_pdf(&objects)
}

/// RGB at user-space `(x, y)` (origin bottom-left).
fn rgb_at(pdf: Vec<u8>, points: &[(u32, u32)]) -> Vec<[u8; 3]> {
    let s = session(pdf, BackendPolicy::CpuOnly);
    let bmp = render_page_rgb(&s, 1, 1.0).expect("render page");
    assert_eq!((bmp.width, bmp.height), (W, H));
    let data = bmp.data();
    let stride = bmp.stride;
    points
        .iter()
        .map(|&(x, y)| {
            let i = (H - 1 - y) as usize * stride + x as usize * 3;
            [data[i], data[i + 1], data[i + 2]]
        })
        .collect()
}

fn assert_close(got: [u8; 3], want: [u8; 3], tol: u8, what: &str) {
    assert!(
        got.iter().zip(want).all(|(&g, w)| g.abs_diff(w) <= tol),
        "{what}: got {got:?}, want {want:?} ± {tol}"
    );
}

#[test]
fn axial_devicen_midpoint_matches_a_flat_fill_of_the_function_value() {
    // Top half: the shading along x.  Bottom half: the function's value at
    // t = 0.5, (0.5, 0.5, 0.795), as a flat fill in the same space.
    let content = "q 0 50 200 50 re W n /Sh0 sh Q \
                   /CS0 cs 0.5 0.5 0.795 scn 80 0 40 50 re f";
    let shading = format!(
        "<< /ShadingType 2 /ColorSpace {CMY_DEVICEN} /Coords [0 0 200 0] \
         /Extend [true true] /Function {RED_TO_GREEN_CMY} >>"
    );
    let px = rgb_at(
        page(content, &shading),
        &[(100, 75), (100, 25), (0, 75), (199, 75)],
    );
    assert_close(px[0], px[1], 3, "shading at t = 0.5 vs flat fill");
    assert!(px[2][0] > 180 && px[2][1] < 80, "start is red: {:?}", px[2]);
    assert!(px[3][1] > px[3][0], "end is green: {:?}", px[3]);
}

#[test]
fn axial_multi_stop_function_passes_through_every_stop() {
    let shading = format!(
        "<< /ShadingType 2 /ColorSpace /DeviceRGB /Coords [0 0 200 0] \
         /Extend [true true] /Function {RED_GREEN_BLUE} >>"
    );
    let px = rgb_at(page("/Sh0 sh", &shading), &[(100, 50), (0, 50), (199, 50)]);
    assert_close(px[0], [0, 255, 0], 4, "middle stop");
    assert_close(px[1], [255, 0, 0], 4, "first stop");
    assert_close(px[2], [0, 0, 255], 4, "last stop");
}

#[test]
fn radial_multi_stop_function_passes_through_every_stop() {
    // Concentric circles about (100, 50), radius 0 → 50.
    let shading = format!(
        "<< /ShadingType 3 /ColorSpace /DeviceRGB /Coords [100 50 0 100 50 50] \
         /Extend [true true] /Function {RED_GREEN_BLUE} >>"
    );
    let px = rgb_at(
        page("/Sh0 sh", &shading),
        &[(125, 50), (100, 50), (160, 50)],
    );
    assert_close(px[0], [0, 255, 0], 12, "middle stop at radius 25");
    assert_close(px[1], [255, 0, 0], 12, "first stop at the centre");
    assert_close(px[2], [0, 0, 255], 4, "last stop beyond radius 50");
}
