//! A shading paints only the points within its extent (ISO 32000-1
//! §8.7.4.5): an axial or radial shading leaves the points beyond its
//! parameter range unpainted unless `Extend` carries it there, a radial
//! shading leaves the points on none of its circles unpainted, and a
//! function-based shading leaves the points outside its `Domain` unpainted.
//! The backdrop survives at every unpainted point.  `BBox` clips the
//! shading (§8.7.4.3).

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

/// The backdrop every test paints first: opaque blue over the whole page.
const BACKDROP: &str = "0 0 1 rg 0 0 200 100 re f ";
const BLUE: [u8; 3] = [0, 0, 255];

/// `DeviceRGB` red at t = 0 to green at t = 1.
const RED_TO_GREEN: &str = "<< /FunctionType 2 /Domain [0 1] /C0 [1 0 0] /C1 [0 1 0] /N 1 >>";

/// One `W` × `H` page running `content`, with shading `/Sh0` (object 5),
/// pattern `/P0` (object 6, `pattern`) and `extra` as objects 7, 8, ….
fn page(content: &str, shading: &[u8], pattern: &str, extra: &[Vec<u8>]) -> Vec<u8> {
    let mut objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W} {H}] \
             /Resources << /Shading << /Sh0 5 0 R >> /Pattern << /P0 6 0 R >> >> \
             /Contents 4 0 R >>"
        )
        .into_bytes(),
        stream_object("", content.as_bytes()),
        shading.to_vec(),
        pattern.as_bytes().to_vec(),
    ];
    objects.extend_from_slice(extra);
    assemble_pdf(&objects)
}

/// A shading pattern over shading object 5.
const PATTERN: &str = "<< /PatternType 2 /Shading 5 0 R >>";

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

/// Whether `rgb` is a red-to-green ramp colour (no blue).
fn is_ramp(rgb: [u8; 3]) -> bool {
    rgb[2] < 8 && u16::from(rgb[0]) + u16::from(rgb[1]) > 200
}

/// Axial red → green from x = 50 to x = 150 along y = 50, with `extra`
/// dictionary entries.
fn axial(extra: &str) -> Vec<u8> {
    format!(
        "<< /ShadingType 2 /ColorSpace /DeviceRGB /Coords [50 50 150 50] \
         /Function {RED_TO_GREEN} {extra} >>"
    )
    .into_bytes()
}

#[test]
fn axial_without_extend_leaves_the_backdrop_beyond_its_axis() {
    let px = rgb_at(
        page(&format!("{BACKDROP}/Sh0 sh"), &axial(""), PATTERN, &[]),
        &[(10, 50), (49, 20), (100, 50), (151, 80), (190, 50)],
    );
    assert_eq!(px[0], BLUE, "before the start point");
    assert_eq!(px[1], BLUE, "just before the start point");
    assert!(is_ramp(px[2]), "on the axis: {:?}", px[2]);
    assert_eq!(px[3], BLUE, "just beyond the end point");
    assert_eq!(px[4], BLUE, "beyond the end point");
}

#[test]
fn axial_with_extend_paints_beyond_its_axis() {
    let px = rgb_at(
        page(
            &format!("{BACKDROP}/Sh0 sh"),
            &axial("/Extend [true true]"),
            PATTERN,
            &[],
        ),
        &[(10, 50), (190, 50)],
    );
    assert!(px[0][0] > 250 && px[0][1] < 5, "start colour: {:?}", px[0]);
    assert!(px[1][1] > 250 && px[1][0] < 5, "end colour: {:?}", px[1]);
}

#[test]
fn axial_extended_at_one_end_paints_only_that_end() {
    let px = rgb_at(
        page(
            &format!("{BACKDROP}/Sh0 sh"),
            &axial("/Extend [false true]"),
            PATTERN,
            &[],
        ),
        &[(10, 50), (190, 50)],
    );
    assert_eq!(px[0], BLUE, "start not extended");
    assert!(px[1][1] > 250, "end extended: {:?}", px[1]);
}

#[test]
fn radial_with_non_concentric_circles_paints_only_their_sweep() {
    // Circle (50, 50) r 10 sweeping to (130, 50) r 30: neither encloses the
    // other, so the shading is the band the circles sweep.
    let shading = format!(
        "<< /ShadingType 3 /ColorSpace /DeviceRGB /Coords [50 50 10 130 50 30] \
         /Function {RED_TO_GREEN} >>"
    );
    let px = rgb_at(
        page(
            &format!("{BACKDROP}/Sh0 sh"),
            shading.as_bytes(),
            PATTERN,
            &[],
        ),
        &[
            (45, 50),
            (100, 50),
            (155, 50),
            (20, 50),
            (190, 50),
            (100, 95),
            (10, 5),
        ],
    );
    for (i, what) in ["start circle", "mid band", "end circle"]
        .iter()
        .enumerate()
    {
        assert!(is_ramp(px[i]), "{what}: {:?}", px[i]);
    }
    for (i, what) in [
        "behind the start",
        "beyond the end",
        "beside the band",
        "far corner",
    ]
    .iter()
    .enumerate()
    {
        assert_eq!(px[i + 3], BLUE, "{what}");
    }
}

#[test]
fn function_based_shading_leaves_the_backdrop_outside_its_domain() {
    // Domain [0 1]² mapped onto (50, 25)–(150, 75), painted with no clip.
    let shading = "<< /ShadingType 1 /ColorSpace /DeviceRGB /Domain [0 1 0 1] \
                   /Matrix [100 0 0 50 50 25] /Function 7 0 R >>";
    let function = stream_object(
        "/FunctionType 4 /Domain [0 1 0 1] /Range [0 1 0 1 0 1]",
        b"{ pop pop 1 1 0 }",
    );
    let px = rgb_at(
        page(
            &format!("{BACKDROP}/Sh0 sh"),
            shading.as_bytes(),
            PATTERN,
            &[function],
        ),
        &[
            (100, 50),
            (60, 30),
            (20, 50),
            (100, 90),
            (180, 10),
            (100, 10),
        ],
    );
    assert_eq!(px[0], [255, 255, 0], "inside the Domain");
    assert_eq!(px[1], [255, 255, 0], "inside the Domain");
    for (i, got) in px.iter().enumerate().skip(2) {
        assert_eq!(*got, BLUE, "probe {i} outside the Domain");
    }
}

#[test]
fn rotated_function_based_shading_leaves_its_domain_corners_unpainted() {
    // A unit Domain square centred on (100, 50), rotated 45° and 40 pt on
    // a side: its device bounding square holds four unpainted corners.
    let shading = "<< /ShadingType 1 /ColorSpace /DeviceRGB /Domain [-0.5 0.5 -0.5 0.5] \
                   /Matrix [28.2843 28.2843 -28.2843 28.2843 100 50] /Function 7 0 R >>";
    let function = stream_object(
        "/FunctionType 4 /Domain [-0.5 0.5 -0.5 0.5] /Range [0 1 0 1 0 1]",
        b"{ pop pop 1 1 0 }",
    );
    let px = rgb_at(
        page(
            &format!("{BACKDROP}/Sh0 sh"),
            shading.as_bytes(),
            PATTERN,
            &[function],
        ),
        &[(100, 50), (100, 75), (125, 50), (78, 72), (122, 28)],
    );
    assert_eq!(px[0], [255, 255, 0], "centre");
    assert_eq!(px[1], [255, 255, 0], "near the top vertex");
    assert_eq!(px[2], [255, 255, 0], "near the right vertex");
    assert_eq!(px[3], BLUE, "top-left corner of the bounding square");
    assert_eq!(px[4], BLUE, "bottom-right corner of the bounding square");
}

#[test]
fn bbox_clips_the_shading() {
    let px = rgb_at(
        page(
            &format!("{BACKDROP}/Sh0 sh"),
            &axial("/Extend [true true] /BBox [0 0 100 100]"),
            PATTERN,
            &[],
        ),
        &[(50, 50), (150, 50)],
    );
    assert!(is_ramp(px[0]), "inside the BBox: {:?}", px[0]);
    assert_eq!(px[1], BLUE, "outside the BBox");
}
