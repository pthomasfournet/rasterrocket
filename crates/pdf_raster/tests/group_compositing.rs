//! Transparency groups composite as the group's own colour and alpha (ISO
//! 32000-2 §11.4.7): pixels a group never paints are left alone whatever
//! blend mode it is composited with, and a blend mode applied inside a
//! non-isolated group blends with the backdrop under the group.

#[expect(dead_code, reason = "shared helpers; this test uses only PDF assembly")]
mod common;

use common::{assemble_pdf, stream_object};
use rasterrocket::{BackendPolicy, render_page_rgb};

/// A 100 × 100 pt page, rendered at one pixel per point.
const W: u32 = 100;
const H: u32 = 100;

/// A page painted mid-grey, then running `content` with ExtGState entries
/// `gstates` and the form `/F` (object 5): a `/Group` form over the page
/// painting `form_content`, with the group entries `group`.
fn page(gstates: &str, content: &str, group: &str, form_content: &str) -> Vec<u8> {
    assemble_pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W} {H}] \
             /Resources << /ExtGState << {gstates} >> /XObject << /F 5 0 R >> >> \
             /Contents 4 0 R >>"
        )
        .into_bytes(),
        stream_object("", format!("0.5 g 0 0 {W} {H} re f {content}").as_bytes()),
        stream_object(
            &format!(
                "/Type /XObject /Subtype /Form /BBox [0 0 {W} {H}] \
                 /Group << /S /Transparency /CS /DeviceRGB {group} >> \
                 /Resources << /ExtGState << {gstates} >> >>"
            ),
            form_content.as_bytes(),
        ),
    ])
}

/// RGB at user-space `(x, y)` (origin bottom-left).
fn rgb_at(pdf: Vec<u8>, points: &[(u32, u32)]) -> Vec<[u8; 3]> {
    let s = common::session(pdf, BackendPolicy::CpuOnly);
    let bmp = render_page_rgb(&s, 1, 1.0).expect("render page");
    let data = bmp.data();
    points
        .iter()
        .map(|&(x, y)| {
            let i = (H - 1 - y) as usize * bmp.stride + x as usize * 3;
            [data[i], data[i + 1], data[i + 2]]
        })
        .collect()
}

const GREY: [u8; 3] = [128, 128, 128];
/// Blue multiplied with mid-grey.
const BLUE_ON_GREY: [u8; 3] = [0, 0, 128];

/// A blue square at 40..60 inside the form.
const SQUARE: &str = "0 0 1 rg 40 40 20 20 re f";

#[test]
fn a_non_isolated_group_under_multiply_leaves_unpainted_pixels_alone() {
    let px = rgb_at(
        page(
            "/M << /BM /Multiply >>",
            "q /M gs /F Do Q",
            "/I false",
            SQUARE,
        ),
        &[(50, 50), (10, 10), (90, 50), (50, 90)],
    );
    assert_eq!(px[0], BLUE_ON_GREY, "the square, multiplied onto grey");
    assert_eq!(px[1], GREY, "unpainted pixel");
    assert_eq!(px[2], GREY, "unpainted pixel");
    assert_eq!(px[3], GREY, "unpainted pixel");
}

#[test]
fn an_isolated_group_under_multiply_leaves_unpainted_pixels_alone() {
    let px = rgb_at(
        page(
            "/M << /BM /Multiply >>",
            "q /M gs /F Do Q",
            "/I true",
            SQUARE,
        ),
        &[(50, 50), (10, 10)],
    );
    assert_eq!(px[0], BLUE_ON_GREY, "the square, multiplied onto grey");
    assert_eq!(px[1], GREY, "unpainted pixel");
}

#[test]
fn a_non_isolated_group_under_half_alpha_blends_only_its_content() {
    let px = rgb_at(
        page("/A << /ca 0.5 >>", "q /A gs /F Do Q", "/I false", SQUARE),
        &[(50, 50), (10, 10)],
    );
    assert!(
        px[0]
            .iter()
            .zip(&[64, 64, 191])
            .all(|(a, b)| a.abs_diff(*b) <= 1),
        "blue at half alpha over grey: {:?}",
        px[0]
    );
    assert_eq!(px[1], GREY, "unpainted pixel");
}

/// Inside a non-isolated group, Multiply sees the grey backdrop; inside an
/// isolated group it sees nothing, so the square stays blue.
#[test]
fn a_blend_mode_inside_a_group_sees_the_backdrop_only_when_non_isolated() {
    let inner = "q /M gs 0 0 1 rg 40 40 20 20 re f Q";
    let px = rgb_at(
        page("/M << /BM /Multiply >>", "/F Do", "/I false", inner),
        &[(50, 50), (10, 10)],
    );
    assert_eq!(
        px[0], BLUE_ON_GREY,
        "non-isolated: multiplied with the backdrop"
    );
    assert_eq!(px[1], GREY, "unpainted pixel");

    let px = rgb_at(
        page("/M << /BM /Multiply >>", "/F Do", "/I true", inner),
        &[(50, 50), (10, 10)],
    );
    assert_eq!(px[0], [0, 0, 255], "isolated: nothing to multiply with");
    assert_eq!(px[1], GREY, "unpainted pixel");
}

/// A blend mode inside a non-isolated group blends with the group's own
/// earlier content over the backdrop: a white square painted first inside
/// the group turns blue multiplied with white into blue.
#[test]
fn a_blend_mode_inside_a_non_isolated_group_sees_earlier_group_content() {
    let inner = "1 g 40 40 20 20 re f q /M gs 0 0 1 rg 40 40 20 20 re f Q";
    let px = rgb_at(
        page("/M << /BM /Multiply >>", "/F Do", "/I false", inner),
        &[(50, 50), (10, 10)],
    );
    assert_eq!(px[0], [0, 0, 255], "blue multiplied with the group's white");
    assert_eq!(px[1], GREY, "unpainted pixel");
}

/// Two group forms in sequence under half alpha, `/F` painting the square
/// and `/G` painting nothing, with `/G` also sent an unbalanced `Q`: the
/// second group composites nothing of the first, and the stray `Q` neither
/// lifts the group's clip nor the alpha.
#[test]
fn a_later_group_composites_only_its_own_content() {
    let pdf = assemble_pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W} {H}] \
             /Resources << /ExtGState << /A << /ca 0.5 >> >> \
             /XObject << /F 5 0 R /G 6 0 R >> >> /Contents 4 0 R >>"
        )
        .into_bytes(),
        stream_object(
            "",
            format!("0.5 g 0 0 {W} {H} re f q /A gs /F Do Q q /A gs /G Do Q").as_bytes(),
        ),
        stream_object(
            &format!(
                "/Type /XObject /Subtype /Form /BBox [0 0 {W} {H}] \
                 /Group << /S /Transparency /CS /DeviceRGB >>"
            ),
            SQUARE.as_bytes(),
        ),
        stream_object(
            "/Type /XObject /Subtype /Form /BBox [40 40 50 50] \
             /Group << /S /Transparency /CS /DeviceRGB >>",
            b"Q Q Q 1 0 0 rg 0 0 100 100 re f",
        ),
    ]);
    let px = rgb_at(pdf, &[(50, 50), (10, 10), (45, 45)]);
    assert!(
        px[0]
            .iter()
            .zip(&[64, 64, 191])
            .all(|(a, b)| a.abs_diff(*b) <= 1),
        "the square once at half alpha: {:?}",
        px[0]
    );
    assert_eq!(px[1], GREY, "outside both groups");
    assert!(
        px[2]
            .iter()
            .zip(&[160, 32, 96])
            .all(|(a, b)| a.abs_diff(*b) <= 1),
        "red at half alpha over the half-alpha square, inside the second group's box: {:?}",
        px[2]
    );
}
