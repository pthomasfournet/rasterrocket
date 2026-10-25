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
    page_with(content, shading.as_bytes(), &[])
}

/// [`page`] with shading object `shading` (a dictionary or stream body) and
/// `extra` appended as objects 7, 8, ….
fn page_with(content: &str, shading: &[u8], extra: &[Vec<u8>]) -> Vec<u8> {
    let mut objects = vec![
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
        shading.to_vec(),
    ];
    objects.extend_from_slice(extra);
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

/// Type 4 function of `(x, y)` running `program`.
fn xy_function(outputs: usize, program: &str) -> Vec<u8> {
    let range = "0 1 ".repeat(outputs);
    stream_object(
        &format!("/FunctionType 4 /Domain [0 1 0 1] /Range [{range}]"),
        program.as_bytes(),
    )
}

/// Points of the Domain square `[0, 1]²` mapped by `Matrix [100 0 0 50 50 25]`
/// onto the page, at shading-space `(0.1, 0.1)`, `(0.9, 0.1)`, `(0.1, 0.9)`
/// and `(0.9, 0.9)` (pixel centres sit half a point further in).
const TYPE1_PROBES: [(u32, u32); 4] = [(60, 30), (140, 30), (60, 70), (140, 70)];
const TYPE1_EXPECTED: [[u8; 3]; 4] = [[27, 28, 0], [231, 28, 0], [27, 232, 0], [231, 232, 0]];

#[test]
fn function_based_shading_reads_both_coordinates_through_its_matrix() {
    // Function(x, y) = (x, y, 0).
    let shading = "<< /ShadingType 1 /ColorSpace /DeviceRGB /Domain [0 1 0 1] \
                   /Matrix [100 0 0 50 50 25] /Function 7 0 R >>";
    let px = rgb_at(
        page_with(
            "q 50 25 100 50 re W n /Sh0 sh Q",
            shading.as_bytes(),
            &[xy_function(3, "{ 0 }")],
        ),
        &TYPE1_PROBES,
    );
    for (got, want) in px.into_iter().zip(TYPE1_EXPECTED) {
        assert_close(got, want, 4, "Function(x, y) = (x, y, 0)");
    }
}

#[test]
fn function_based_shading_takes_an_array_of_two_input_functions() {
    // Function = [x, y, 0], one single-output function per channel.
    let shading = "<< /ShadingType 1 /ColorSpace /DeviceRGB /Domain [0 1 0 1] \
                   /Matrix [100 0 0 50 50 25] /Function [7 0 R 8 0 R 9 0 R] >>";
    let px = rgb_at(
        page_with(
            "q 50 25 100 50 re W n /Sh0 sh Q",
            shading.as_bytes(),
            &[
                xy_function(1, "{ pop }"),
                xy_function(1, "{ exch pop }"),
                xy_function(1, "{ pop pop 0 }"),
            ],
        ),
        &TYPE1_PROBES,
    );
    for (got, want) in px.into_iter().zip(TYPE1_EXPECTED) {
        assert_close(got, want, 4, "Function = [x, y, 0]");
    }
}

/// Probes along y = 50 where a mesh coloured with `t = x / 200` takes
/// `RED_GREEN_BLUE` at t = 0.2, 0.5 and 0.8 (pixel centres add 0.0025).
const MESH_PROBES: [(u32, u32); 3] = [(40, 50), (100, 50), (160, 50)];
const MESH_EXPECTED: [[u8; 3]; 3] = [[152, 103, 0], [0, 254, 1], [0, 101, 154]];

#[test]
fn lattice_mesh_with_function_interpolates_t_through_every_stop() {
    // A 2 × 2 lattice over the page, t = 0 on the left and 1 on the right
    // (8-bit coordinates and t).
    let shading = stream_object(
        &format!(
            "/ShadingType 5 /ColorSpace /DeviceRGB /BitsPerCoordinate 8 \
             /BitsPerComponent 8 /VerticesPerRow 2 /Decode [0 200 0 100 0 1] \
             /Function {RED_GREEN_BLUE}"
        ),
        &[0, 0, 0, 255, 0, 255, 0, 255, 0, 255, 255, 255],
    );
    let px = rgb_at(page_with("/Sh0 sh", &shading, &[]), &MESH_PROBES);
    for (got, want) in px.into_iter().zip(MESH_EXPECTED) {
        assert_close(got, want, 6, "type 5 mesh, t = x / 200");
    }
}

/// Big-endian 16-bit coordinate of `v` on `[0, extent]`.
fn coord16(v: f64, extent: f64) -> [u8; 2] {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "v ∈ [0, extent], so the product lies in [0, 65535]"
    )]
    let raw = (v / extent * 65535.0).round() as u16;
    raw.to_be_bytes()
}

#[test]
fn coons_mesh_with_function_interpolates_t_through_every_stop() {
    // One flat patch over the page; boundary points clockwise from (0, 0)
    // along y = 0, then up x = 200, back along y = 100 and down x = 0.
    let third = |k: f64| k * 200.0 / 3.0;
    let boundary = [
        (0.0, 0.0),
        (third(1.0), 0.0),
        (third(2.0), 0.0),
        (200.0, 0.0),
        (200.0, 100.0 / 3.0),
        (200.0, 200.0 / 3.0),
        (200.0, 100.0),
        (third(2.0), 100.0),
        (third(1.0), 100.0),
        (0.0, 100.0),
        (0.0, 200.0 / 3.0),
        (0.0, 100.0 / 3.0),
    ];
    let mut data = vec![0u8]; // flag 0
    for (x, y) in boundary {
        data.extend(coord16(x, 200.0));
        data.extend(coord16(y, 100.0));
    }
    // Corner t at (0, 0), (200, 0), (200, 100), (0, 100).
    data.extend([0, 255, 255, 0]);
    let shading = stream_object(
        &format!(
            "/ShadingType 6 /ColorSpace /DeviceRGB /BitsPerCoordinate 16 \
             /BitsPerComponent 8 /BitsPerFlag 8 /Decode [0 200 0 100 0 1] \
             /Function {RED_GREEN_BLUE}"
        ),
        &data,
    );
    let px = rgb_at(page_with("/Sh0 sh", &shading, &[]), &MESH_PROBES);
    for (got, want) in px.into_iter().zip(MESH_EXPECTED) {
        assert_close(got, want, 6, "type 6 mesh, t = x / 200");
    }
}

#[test]
fn mesh_in_a_five_colorant_space_converts_every_vertex() {
    // /CS5 = [/DeviceN [/C /M /Y /K /Spot] /DeviceCMYK { pop }]: the spot is
    // dropped.  Every vertex of the upper half is (0, 1, 1, 0, 0.5); the flat
    // fill of the lower strip is the same colour.
    let mut data = Vec::new();
    for (x, y) in [(0u8, 128u8), (255, 128), (0, 255), (255, 255)] {
        data.extend([x, y, 0, 255, 255, 0, 128]);
    }
    let objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W} {H}] \
             /Resources << /Shading << /Sh0 6 0 R >> /ColorSpace << /CS5 7 0 R >> >> \
             /Contents 4 0 R >>"
        )
        .into_bytes(),
        stream_object("", b"/Sh0 sh /CS5 cs 0 1 1 0 0.5 scn 0 0 200 40 re f"),
        stream_object(
            "/FunctionType 4 /Domain [0 1 0 1 0 1 0 1 0 1] /Range [0 1 0 1 0 1 0 1]",
            b"{ pop }",
        ),
        stream_object(
            "/ShadingType 5 /ColorSpace 7 0 R /BitsPerCoordinate 8 \
             /BitsPerComponent 8 /VerticesPerRow 2 \
             /Decode [0 200 0 100 0 1 0 1 0 1 0 1 0 1]",
            &data,
        ),
        b"[/DeviceN [/Cyan /Magenta /Yellow /Black /Spot] /DeviceCMYK 5 0 R]".to_vec(),
    ];
    let px = rgb_at(assemble_pdf(&objects), &[(100, 80), (100, 20)]);
    assert_close(px[0], px[1], 2, "mesh vertex colour vs flat fill");
    assert!(px[0][0] > 180 && px[0][1] < 80, "red: {:?}", px[0]);
}
