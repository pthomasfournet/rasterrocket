//! An image is painted through the same compositing state as a fill: the
//! nonstroking constant alpha and the blend mode apply to it, and inside a
//! transparency group its shape enters the group's alpha, so the group
//! composites it back.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use common::{gray_at, render_gray, session, stream_object, with_page_tree};
use rasterrocket::{BackendPolicy, render_page_rgb};

/// A 2 × 2 black `DeviceGray` image.
fn black_image() -> Vec<u8> {
    stream_object(
        "/Type /XObject /Subtype /Image /Width 2 /Height 2 \
         /ColorSpace /DeviceGray /BitsPerComponent 8",
        &[0, 0, 0, 0],
    )
}

/// One page running `content` with image `/Im0`, form `/Fm0` (whose content
/// is `form_content` and whose dictionary carries `form_extras`) and
/// `ExtGState` `/Half` (`/ca 0.5`) in its resources.
fn page(content: &str, form_content: &str, form_extras: &str) -> Vec<u8> {
    let objects = vec![
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
           /Resources << /XObject << /Im0 5 0 R /Fm0 6 0 R >> \
                         /ExtGState << /Half << /Type /ExtGState /ca 0.5 >> >> >> \
           /Contents 4 0 R >>"
            .to_vec(),
        stream_object("", content.as_bytes()),
        black_image(),
        stream_object(
            &format!(
                "/Type /XObject /Subtype /Form /BBox [0 0 100 100] \
                 /Resources << /XObject << /Im0 5 0 R >> >> {form_extras}"
            ),
            form_content.as_bytes(),
        ),
    ];
    with_page_tree(&objects)
}

const IMAGE_IN_LEFT_HALF: &str = "q 50 0 0 100 0 0 cm /Im0 Do Q";

#[test]
fn image_inside_an_isolated_group_is_composited() {
    let page = render_gray(page(
        "/Fm0 Do",
        IMAGE_IN_LEFT_HALF,
        "/Group << /S /Transparency /I true /CS /DeviceGray >>",
    ));
    assert_eq!(gray_at(&page, 25, 50), 0, "the image reaches the page");
    assert_eq!(gray_at(&page, 75, 50), 255, "the group adds nothing else");
}

#[test]
fn image_inside_a_non_isolated_group_is_composited() {
    let page = render_gray(page(
        "/Fm0 Do",
        IMAGE_IN_LEFT_HALF,
        "/Group << /S /Transparency >>",
    ));
    assert_eq!(gray_at(&page, 25, 50), 0, "the image reaches the page");
    assert_eq!(gray_at(&page, 75, 50), 255, "the backdrop is unchanged");
}

#[test]
fn image_honours_the_nonstroking_constant_alpha() {
    let page = render_gray(page(&format!("q /Half gs {IMAGE_IN_LEFT_HALF} Q"), "", ""));
    let half = gray_at(&page, 25, 50);
    assert!(
        (120..=136).contains(&half),
        "black at ca 0.5 over white: {half}"
    );
}

#[test]
fn opaque_image_outside_any_group_is_unchanged() {
    let page = render_gray(page(IMAGE_IN_LEFT_HALF, "", ""));
    assert_eq!(gray_at(&page, 25, 50), 0);
    assert_eq!(gray_at(&page, 75, 50), 255);
}

/// RGB at x = 25 and x = 75 on row 50 of a red 100 × 100 page over which a
/// gray-128 image `XObject` (left half) and inline image (right half) are
/// painted under blend mode `bm`.
fn images_over_red(bm: &str) -> [[u8; 3]; 2] {
    let content = b"1 0 0 rg 0 0 100 100 re f q /BM0 gs \
                    q 50 0 0 100 0 0 cm /Im0 Do Q \
                    q 50 0 0 100 50 0 cm BI /W 1 /H 1 /CS /G /BPC 8 ID \x80\nEI Q Q";
    let objects = vec![
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
             /Resources << /XObject << /Im0 5 0 R >> \
                           /ExtGState << /BM0 << /BM /{bm} >> >> >> \
             /Contents 4 0 R >>"
        )
        .into_bytes(),
        stream_object("", content),
        stream_object(
            "/Type /XObject /Subtype /Image /Width 1 /Height 1 \
             /ColorSpace /DeviceGray /BitsPerComponent 8",
            &[0x80],
        ),
    ];
    let s = session(with_page_tree(&objects), BackendPolicy::CpuOnly);
    let bmp = render_page_rgb(&s, 1, 1.0).expect("render page");
    let row = &bmp.data()[50 * bmp.stride..];
    let at = |x: usize| [row[x * 3], row[x * 3 + 1], row[x * 3 + 2]];
    [at(25), at(75)]
}

#[test]
fn images_honour_the_blend_mode() {
    for (bm, want) in [
        ("Normal", [128, 128, 128]),
        ("Multiply", [128, 0, 0]),
        ("Screen", [255, 128, 128]),
        ("Difference", [127, 128, 128]),
    ] {
        for (got, what) in images_over_red(bm).into_iter().zip(["XObject", "inline"]) {
            assert!(
                got.iter().zip(want).all(|(&g, w)| g.abs_diff(w) <= 1),
                "{what} image under /BM /{bm}: got {got:?}, want {want:?}"
            );
        }
    }
}
