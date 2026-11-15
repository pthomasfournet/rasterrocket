//! A page of smooth tone ramps whose levels show the depth of a render:
//! an axial shading, a luminosity soft-mask vignette and a Gouraud mesh,
//! each black to white over the page's 2048 points.

use super::{stream_object, with_page_tree};

/// Width of the page in points: 2048 device pixels at one pixel a point.
pub const WIDTH: u32 = 2048;

/// Height of the page in points.
pub const HEIGHT: u32 = 600;

/// The bands of the page, top to bottom, as `(name, device row at their
/// middle)` at one pixel a point.
pub const BANDS: [(&str, u32); 3] = [("axial", 100), ("soft mask", 300), ("Gouraud mesh", 500)];

/// The Type 4 mesh: two triangles over the bottom band, black at `x = 0`
/// and white at `x = 2048`, of 16-bit coordinates and colours.
fn mesh() -> Vec<u8> {
    let mut data = Vec::new();
    let black = [0u16; 3];
    let white = [u16::MAX; 3];
    let (x1, y1) = (u16::MAX, u16::MAX);
    for (x, y, colour) in [
        (0, 0, black),
        (x1, 0, white),
        (0, y1, black),
        (x1, 0, white),
        (x1, y1, white),
        (0, y1, black),
    ] {
        data.push(0);
        for v in [x, y].into_iter().chain(colour) {
            data.extend_from_slice(&v.to_be_bytes());
        }
    }
    stream_object(
        "/ShadingType 4 /ColorSpace /DeviceRGB /BitsPerCoordinate 16 /BitsPerComponent 16 \
         /BitsPerFlag 8 /Decode [0 2048 0 200 0 1 0 1 0 1]",
        &data,
    )
}

/// The page: an axial `DeviceRGB` shading black to white across the top
/// band; black painted over the middle band through a luminosity soft mask
/// whose group is a radial `DeviceGray` vignette, white at its centre; and
/// the Gouraud mesh of [`mesh`] over the bottom band.
#[must_use]
pub fn gradient_page() -> Vec<u8> {
    let content = b"q 0 400 2048 200 re W n /Ax sh Q \
                    q /GS0 gs 0 g 0 200 2048 200 re f Q \
                    q 0 0 2048 200 re W n /Mesh sh Q";
    let vignette = b"/V sh";
    with_page_tree(&[
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {WIDTH} {HEIGHT}] \
             /Resources << /Shading << /Ax 5 0 R /Mesh 6 0 R >> \
             /ExtGState << /GS0 << /SMask << /S /Luminosity /G 7 0 R >> >> >> >> \
             /Contents 4 0 R >>"
        )
        .into_bytes(),
        stream_object("", content),
        b"<< /ShadingType 2 /ColorSpace /DeviceRGB /Coords [0 0 2048 0] \
           /Function << /FunctionType 2 /Domain [0 1] /C0 [0 0 0] /C1 [1 1 1] /N 1 >> >>"
            .to_vec(),
        mesh(),
        stream_object(
            "/Type /XObject /Subtype /Form /BBox [0 0 2048 600] \
             /Group << /S /Transparency /CS /DeviceGray >> \
             /Resources << /Shading << /V 8 0 R >> >>",
            vignette,
        ),
        b"<< /ShadingType 3 /ColorSpace /DeviceGray /Coords [1024 300 0 1024 300 1100] \
           /Function << /FunctionType 2 /Domain [0 1] /C0 [1] /C1 [0] /N 1 >> \
           /Extend [true true] >>"
            .to_vec(),
    ])
}
