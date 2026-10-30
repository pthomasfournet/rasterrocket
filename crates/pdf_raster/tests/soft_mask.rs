//! Soft masks installed through an `ExtGState` `SMask` entry scale the
//! opacity of everything painted while they are in effect.

#[expect(dead_code, reason = "shared helpers; this test uses only PDF assembly")]
mod common;

use common::{assemble_pdf, stream_object};
use rasterrocket::{RasterOptions, RenderedPage, raster_pdf_from_bytes};

/// 72 DPI: one pixel per point on the 100 × 100 pt page.
const DPI: f32 = 72.0;

/// A one-page document: `content` with `ExtGState` entries `gstates` (the
/// inner text of the `/ExtGState` dictionary), plus `extra` objects
/// numbered from 5.
fn page(gstates: &str, xobjects: &str, content: &[u8], extra: Vec<Vec<u8>>) -> Vec<u8> {
    let mut objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
             /Resources << /ExtGState << {gstates} >> /XObject << {xobjects} >> >> \
             /Contents 4 0 R >>"
        )
        .into_bytes(),
        stream_object("", content),
    ];
    objects.extend(extra);
    assemble_pdf(&objects)
}

/// A transparency-group form over the whole page running `content`.
fn group_form(resources: &str, content: &[u8]) -> Vec<u8> {
    stream_object(
        &format!(
            "/Type /XObject /Subtype /Form /BBox [0 0 100 100] \
             /Group << /S /Transparency /CS /DeviceRGB >> /Resources << {resources} >>"
        ),
        content,
    )
}

fn render(pdf: Vec<u8>) -> RenderedPage {
    let opts = RasterOptions::new(DPI);
    let mut pages: Vec<_> = raster_pdf_from_bytes(pdf, &opts).collect();
    assert_eq!(pages.len(), 1, "one page expected");
    pages.remove(0).1.expect("page renders")
}

/// Gray value at PDF user-space `(x, y)` (origin bottom-left).
fn gray_at(page: &RenderedPage, x: u32, y: u32) -> u8 {
    let row = page.height - 1 - y;
    page.pixels[(row * page.width + x) as usize]
}

/// Luminosity mask whose group paints the left half white over the default
/// black backdrop: a black fill shows on the left only.
#[test]
fn luminosity_mask_admits_paint_where_the_group_is_light() {
    let pdf = page(
        "/GS0 << /SMask << /S /Luminosity /G 5 0 R >> >>",
        "",
        b"/GS0 gs 0 g 0 0 100 100 re f",
        vec![group_form("", b"1 g 0 0 50 100 re f")],
    );
    let page = render(pdf);
    assert_eq!(gray_at(&page, 25, 50), 0, "mask 1: painted");
    assert_eq!(gray_at(&page, 75, 50), 255, "mask 0: backdrop shows");
}

/// The structure of a faded page background: an image inside a group,
/// under a constant alpha, behind a vertical luminosity gradient.  The
/// image is fully masked at the top and shows at the constant alpha at the
/// bottom.
#[test]
fn gradient_mask_fades_an_image_drawn_in_a_group() {
    let shading = "/Shading << /Sh0 << /ShadingType 2 /ColorSpace /DeviceGray \
        /Coords [0 0 0 100] /Extend [true true] \
        /Function << /FunctionType 2 /Domain [0 1] /C0 [1] /C1 [0] /N 1 >> >> >>";
    let pdf = page(
        "/GS0 << /ca 0.5 >>",
        "/Fm0 6 0 R",
        b"/GS0 gs /Fm0 Do",
        vec![
            group_form(shading, b"/Sh0 sh"),
            group_form(
                "/ExtGState << /GS1 << /SMask << /S /Luminosity /G 5 0 R >> >> >> \
                 /XObject << /Im0 7 0 R >>",
                b"/GS1 gs q 100 0 0 100 0 0 cm /Im0 Do Q",
            ),
            stream_object(
                "/Type /XObject /Subtype /Image /Width 2 /Height 2 \
                 /ColorSpace /DeviceGray /BitsPerComponent 8",
                &[0; 4],
            ),
        ],
    );
    let page = render(pdf);
    let bottom = gray_at(&page, 50, 1);
    let middle = gray_at(&page, 50, 50);
    let top = gray_at(&page, 50, 98);
    // Black at alpha 0.5 × mask over white.
    assert!((126..=134).contains(&bottom), "bottom {bottom}, mask ≈ 1");
    assert!((185..=197).contains(&middle), "middle {middle}, mask ≈ 0.5");
    assert!(top >= 252, "top {top}, mask ≈ 0");
}

/// Alpha mask: the group's opacity, not its colour, is the mask.
#[test]
fn alpha_mask_takes_the_group_opacity() {
    let pdf = page(
        "/GS0 << /SMask << /S /Alpha /G 5 0 R >> >>",
        "",
        b"/GS0 gs 0 g 0 0 100 100 re f",
        vec![group_form(
            "/ExtGState << /H << /ca 0.5 >> >>",
            b"/H gs 0 g 0 0 50 100 re f",
        )],
    );
    let page = render(pdf);
    let left = gray_at(&page, 25, 50);
    assert!((125..=130).contains(&left), "left {left}, mask 0.5");
    assert_eq!(
        gray_at(&page, 75, 50),
        255,
        "mask 0 where the group is empty"
    );
}

/// The backdrop colour fills the luminosity mask wherever the group leaves
/// it uncovered.
#[test]
fn luminosity_backdrop_colour_applies_outside_the_group() {
    let pdf = page(
        "/GS0 << /SMask << /S /Luminosity /G 5 0 R /BC [1] >> >>",
        "",
        b"/GS0 gs 0 g 0 0 100 100 re f",
        vec![group_form("", b"0 g 0 0 50 100 re f")],
    );
    let page = render(pdf);
    assert_eq!(gray_at(&page, 25, 50), 255, "black group: mask 0");
    assert_eq!(gray_at(&page, 75, 50), 0, "white backdrop: mask 1");
}

/// The mask applies to each object painted while it is in effect (ISO
/// 32000-2 §11.6.4.3): two black squares painted over each other at mask
/// 0.5 darken twice, inside a form without a group as on the page.
#[test]
fn a_mask_applies_to_each_object_of_a_form_without_a_group() {
    let two_squares = b"0 g 20 20 60 60 re f 0 g 20 20 60 60 re f";
    let pdf = page(
        "/GS0 << /SMask << /S /Luminosity /G 5 0 R >> >>",
        "/Fm0 6 0 R",
        b"/GS0 gs /Fm0 Do",
        vec![
            group_form("", b"0.5 g 0 0 100 100 re f"),
            stream_object(
                "/Type /XObject /Subtype /Form /BBox [0 0 100 100]",
                two_squares,
            ),
        ],
    );
    let page = render(pdf);
    let twice = gray_at(&page, 50, 50);
    assert!((60..=68).contains(&twice), "two masked objects: {twice}");
    assert_eq!(gray_at(&page, 10, 10), 255, "outside the squares");
}

/// A form with a group is one object: its content composites unmasked
/// into the group, and the mask applies to the group once.
#[test]
fn a_mask_applies_once_to_a_form_with_a_group() {
    let pdf = page(
        "/GS0 << /SMask << /S /Luminosity /G 5 0 R >> >>",
        "/Fm0 6 0 R",
        b"/GS0 gs /Fm0 Do",
        vec![
            group_form("", b"0.5 g 0 0 100 100 re f"),
            group_form("", b"0 g 20 20 60 60 re f 0 g 20 20 60 60 re f"),
        ],
    );
    let page = render(pdf);
    let once = gray_at(&page, 50, 50);
    assert!((125..=130).contains(&once), "one masked group: {once}");
}

/// A masked object outside the clip paints nothing, and a mask set while
/// the clip is empty does not disturb later painting.
#[test]
fn a_masked_object_outside_the_clip_paints_nothing() {
    let pdf = page(
        "/GS0 << /SMask << /S /Luminosity /G 5 0 R >> >>",
        "",
        b"q 0 0 50 100 re W n /GS0 gs 0 g 60 0 40 100 re f Q 0 g 0 0 20 20 re f",
        vec![group_form("", b"1 g 0 0 100 100 re f")],
    );
    let page = render(pdf);
    assert_eq!(gray_at(&page, 80, 50), 255, "clipped away");
    assert_eq!(gray_at(&page, 10, 10), 0, "painted after Q");
}

/// `/SMask /None` and `Q` both lift the mask.
#[test]
fn soft_mask_is_lifted_by_none_and_by_restore() {
    let pdf = page(
        "/GS0 << /SMask << /S /Luminosity /G 5 0 R >> >> /GS1 << /SMask /None >>",
        "",
        b"q /GS0 gs 0 g 0 0 100 50 re f Q 0 g 0 50 50 50 re f \
          /GS0 gs /GS1 gs 50 50 50 50 re f",
        vec![group_form("", b"")],
    );
    let page = render(pdf);
    assert_eq!(
        gray_at(&page, 50, 25),
        255,
        "empty mask group: nothing painted"
    );
    assert_eq!(gray_at(&page, 25, 75), 0, "after Q: unmasked");
    assert_eq!(gray_at(&page, 75, 75), 0, "after /SMask /None: unmasked");
}

/// An image with its own `/SMask` is drawn through that mask instead of the
/// graphics state's: under a mask of 0 everywhere, the image with an opaque
/// `/SMask` paints and the image without one does not.
#[test]
fn an_image_smask_replaces_the_graphics_state_mask() {
    let image = |smask: &str| {
        stream_object(
            &format!(
                "/Type /XObject /Subtype /Image /Width 1 /Height 1 \
                 /ColorSpace /DeviceGray /BitsPerComponent 8 {smask}"
            ),
            &[0],
        )
    };
    let pdf = page(
        "/GS0 << /SMask << /S /Luminosity /G 5 0 R >> >>",
        "/Own 6 0 R /Plain 8 0 R",
        b"/GS0 gs q 50 0 0 100 0 0 cm /Own Do Q q 50 0 0 100 50 0 cm /Plain Do Q",
        vec![
            group_form("", b""),
            image("/SMask 7 0 R"),
            stream_object(
                "/Type /XObject /Subtype /Image /Width 1 /Height 1 \
                 /ColorSpace /DeviceGray /BitsPerComponent 8",
                &[255],
            ),
            image(""),
        ],
    );
    let page = render(pdf);
    assert_eq!(gray_at(&page, 25, 50), 0, "own /SMask: image paints");
    assert_eq!(gray_at(&page, 75, 50), 255, "graphics-state mask 0: hidden");
}
