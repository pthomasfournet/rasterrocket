//! A shading paints only the points within its extent (ISO 32000-1
//! §8.7.4.5): an axial or radial shading leaves the points beyond its
//! parameter range unpainted unless `Extend` carries it there, a radial
//! shading leaves the points on none of its circles unpainted, and a
//! function-based shading leaves the points outside its `Domain` unpainted.
//! The backdrop survives at every unpainted point.  `BBox` clips the shading;
//! `Background` fills the area of a shading pattern before the shading, and
//! has no effect through `sh` (§8.7.4.3).

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use common::{rgb_at, stream_object, with_page_tree};

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
    with_page_tree(&objects)
}

/// A shading pattern over shading object 5.
const PATTERN: &str = "<< /PatternType 2 /Shading 5 0 R >>";

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
fn background_applies_to_a_shading_pattern_but_not_to_sh() {
    let shading = axial("/Background [1 1 0]");
    let probes = [(10, 50), (100, 50), (190, 50)];

    let sh = rgb_at(
        page(&format!("{BACKDROP}/Sh0 sh"), &shading, PATTERN, &[]),
        &probes,
    );
    assert_eq!(sh[0], BLUE, "sh ignores Background");
    assert!(is_ramp(sh[1]), "sh paints the shading: {:?}", sh[1]);
    assert_eq!(sh[2], BLUE, "sh ignores Background");

    let scn = rgb_at(
        page(
            &format!("{BACKDROP}/Pattern cs /P0 scn 0 0 200 100 re f"),
            &shading,
            PATTERN,
            &[],
        ),
        &probes,
    );
    assert_eq!(scn[0], [255, 255, 0], "the pattern fills with Background");
    assert!(
        is_ramp(scn[1]),
        "the pattern paints the shading: {:?}",
        scn[1]
    );
    assert_eq!(scn[2], [255, 255, 0], "the pattern fills with Background");
}

#[test]
fn shading_pattern_without_background_leaves_the_backdrop() {
    let px = rgb_at(
        page(
            &format!("{BACKDROP}/Pattern cs /P0 scn 0 0 200 100 re f"),
            &axial(""),
            PATTERN,
            &[],
        ),
        &[(10, 50), (100, 50), (190, 50)],
    );
    assert_eq!(px[0], BLUE);
    assert!(is_ramp(px[1]), "{:?}", px[1]);
    assert_eq!(px[2], BLUE);
}

#[test]
fn shading_pattern_fill_is_limited_to_the_path() {
    let px = rgb_at(
        page(
            &format!("{BACKDROP}/Pattern cs /P0 scn 80 20 40 60 re f"),
            &axial("/Extend [true true] /Background [1 1 0]"),
            PATTERN,
            &[],
        ),
        &[(100, 50), (60, 50), (100, 90)],
    );
    assert!(is_ramp(px[0]), "inside the path: {:?}", px[0]);
    assert_eq!(px[1], BLUE, "outside the path");
    assert_eq!(px[2], BLUE, "outside the path");
}

#[test]
fn shading_pattern_space_is_the_default_space_not_the_ctm() {
    // The CTM scale must not move the gradient: the pattern's space is the
    // page's default coordinate space (§8.7.3.1).
    let px = rgb_at(
        page(
            &format!("{BACKDROP}0.5 0 0 0.5 0 0 cm /Pattern cs /P0 scn 0 0 400 200 re f"),
            &axial(""),
            PATTERN,
            &[],
        ),
        &[(40, 50), (100, 50), (160, 50)],
    );
    assert_eq!(px[0], BLUE, "x = 40 is before the start point");
    assert!(is_ramp(px[1]), "{:?}", px[1]);
    assert_eq!(px[2], BLUE, "x = 160 is beyond the end point");
}

#[test]
fn shading_pattern_matrix_maps_pattern_space_to_default_space() {
    // Matrix doubles pattern space: the axis 25 → 75 lands on 50 → 150.
    let shading = "<< /ShadingType 2 /ColorSpace /DeviceRGB /Coords [25 25 75 25] \
                   /Function 7 0 R >>";
    let px = rgb_at(
        page(
            &format!("{BACKDROP}/Pattern cs /P0 scn 0 0 200 100 re f"),
            shading.as_bytes(),
            "<< /PatternType 2 /Shading 5 0 R /Matrix [2 0 0 2 0 0] >>",
            &[RED_TO_GREEN.as_bytes().to_vec()],
        ),
        &[(40, 50), (100, 50), (160, 50)],
    );
    assert_eq!(px[0], BLUE);
    assert!(is_ramp(px[1]), "{:?}", px[1]);
    assert_eq!(px[2], BLUE);
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

#[test]
fn mesh_shading_pattern_fill_is_limited_to_the_path() {
    // A 2 × 2 lattice over the page, red on the left and green on the right.
    let shading = stream_object(
        &format!(
            "/ShadingType 5 /ColorSpace /DeviceRGB /BitsPerCoordinate 8 \
             /BitsPerComponent 8 /VerticesPerRow 2 /Decode [0 200 0 100 0 1] \
             /Function {RED_TO_GREEN}"
        ),
        &[0, 0, 0, 255, 0, 255, 0, 255, 0, 255, 255, 255],
    );
    let px = rgb_at(
        page(
            &format!("{BACKDROP}/Pattern cs /P0 scn 50 25 100 50 re f"),
            &shading,
            PATTERN,
            &[],
        ),
        &[(100, 50), (20, 50), (100, 90)],
    );
    assert!(is_ramp(px[0]), "inside the path: {:?}", px[0]);
    assert_eq!(px[1], BLUE, "outside the path");
    assert_eq!(px[2], BLUE, "outside the path");
}
