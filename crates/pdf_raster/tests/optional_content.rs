//! Optional content (ISO 32000-2 §8.11): content in a group that is OFF in
//! the default configuration is not drawn, and every marked-content
//! sequence (§14.6), optional or not, is closed by exactly one `EMC`.

#[expect(dead_code, reason = "shared helpers; this test uses only PDF assembly")]
mod common;

use common::{assemble_pdf, stream_object};
use rasterrocket::{BackendPolicy, render_page_rgb};

/// A 100 × 100 pt page, rendered at one pixel per point.
const W: u32 = 100;
const H: u32 = 100;

const WHITE: [u8; 3] = [255, 255, 255];
const RED: [u8; 3] = [255, 0, 0];
const GREEN: [u8; 3] = [0, 255, 0];
const BLUE: [u8; 3] = [0, 0, 255];

/// A page with `content`, catalog `/OCProperties` `ocprops`, page
/// resources `resources`, and `extra` objects numbered from 5 on.
fn page(ocprops: &str, resources: &str, content: &str, extra: Vec<Vec<u8>>) -> Vec<u8> {
    let mut objects = vec![
        format!("<< /Type /Catalog /Pages 2 0 R /OCProperties {ocprops} >>").into_bytes(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W} {H}] \
             /Resources << {resources} >> /Contents 4 0 R >>"
        )
        .into_bytes(),
        stream_object("", content.as_bytes()),
    ];
    objects.extend(extra);
    assemble_pdf(&objects)
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

/// Groups 5 (OFF in `/D`) and 6 (ON).
const TWO_GROUPS: &str = "<< /OCGs [5 0 R 6 0 R] /D << /OFF [5 0 R] >> >>";
const OFF_GROUP: &[u8] = b"<< /Type /OCG /Name (off) >>";
const ON_GROUP: &[u8] = b"<< /Type /OCG /Name (on) >>";

/// Objects 5 and 6: the OFF and ON groups.
fn groups() -> Vec<Vec<u8>> {
    vec![OFF_GROUP.to_vec(), ON_GROUP.to_vec()]
}

/// Objects 5 and 6, then object 7: a form with `dict` entries and `content`.
fn groups_and_form(dict: &str, content: &[u8]) -> Vec<Vec<u8>> {
    let mut objects = groups();
    objects.push(stream_object(
        &format!("/Type /XObject /Subtype /Form /BBox [0 0 100 100] {dict}"),
        content,
    ));
    objects
}

/// The centres of three vertical bars at x 0..20, 40..60 and 80..100.
const BARS: [(u32, u32); 3] = [(10, 50), (50, 50), (90, 50)];

#[test]
fn a_nested_marked_content_sequence_does_not_end_the_hidden_group() {
    let px = rgb_at(
        page(
            TWO_GROUPS,
            "/Properties << /pOff 5 0 R >>",
            "/OC /pOff BDC 1 0 0 rg 0 0 20 100 re f \
             /Span << /MCID 3 >> BDC EMC \
             0 1 0 rg 40 0 20 100 re f EMC \
             0 0 1 rg 80 0 20 100 re f",
            groups(),
        ),
        &BARS,
    );
    assert_eq!(px, [WHITE, WHITE, BLUE]);
}

#[test]
fn a_bmc_sequence_inside_a_hidden_group_does_not_end_it() {
    let px = rgb_at(
        page(
            TWO_GROUPS,
            "/Properties << /pOff 5 0 R >>",
            "/OC /pOff BDC /Artifact BMC 1 0 0 rg 0 0 20 100 re f EMC \
             0 1 0 rg 40 0 20 100 re f EMC \
             0 0 1 rg 80 0 20 100 re f",
            groups(),
        ),
        &BARS,
    );
    assert_eq!(px, [WHITE, WHITE, BLUE]);
}

#[test]
fn an_unterminated_hidden_group_in_a_form_ends_with_the_form() {
    let px = rgb_at(
        page(
            TWO_GROUPS,
            "/XObject << /F 7 0 R >>",
            "1 0 0 rg 0 0 20 100 re f /F Do 0 0 1 rg 80 0 20 100 re f",
            groups_and_form(
                "/Resources << /Properties << /pOff 5 0 R >> >>",
                b"/OC /pOff BDC 0 1 0 rg 40 0 20 100 re f",
            ),
        ),
        &BARS,
    );
    assert_eq!(px, [RED, WHITE, BLUE]);
}

#[test]
fn a_form_cannot_close_a_sequence_of_the_stream_that_invokes_it() {
    let px = rgb_at(
        page(
            TWO_GROUPS,
            "/Properties << /pOn 6 0 R >> /XObject << /F 7 0 R >>",
            "/OC /pOn BDC /F Do 1 0 0 rg 0 0 20 100 re f EMC \
             0 0 1 rg 80 0 20 100 re f",
            groups_and_form(
                "/Resources << /Properties << /pOff 5 0 R >> >>",
                b"EMC /OC /pOff BDC",
            ),
        ),
        &BARS,
    );
    assert_eq!(px, [RED, WHITE, BLUE]);
}

#[test]
fn an_unterminated_hidden_group_in_the_page_does_not_hide_annotations() {
    let mut pdf_objects = groups();
    // Object 7: an annotation whose appearance (object 8) fills its rect.
    pdf_objects.push(
        b"<< /Type /Annot /Subtype /Square /Rect [80 0 100 100] /AP << /N 8 0 R >> >>".to_vec(),
    );
    pdf_objects.push(stream_object(
        "/Type /XObject /Subtype /Form /BBox [0 0 20 100]",
        b"0 0 1 rg 0 0 20 100 re f",
    ));
    let mut objects = vec![
        format!("<< /Type /Catalog /Pages 2 0 R /OCProperties {TWO_GROUPS} >>").into_bytes(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W} {H}] \
             /Resources << /Properties << /pOff 5 0 R >> >> /Contents 4 0 R /Annots [7 0 R] >>"
        )
        .into_bytes(),
        stream_object(
            "",
            b"1 0 0 rg 0 0 20 100 re f /OC /pOff BDC 0 1 0 rg 40 0 20 100 re f",
        ),
    ];
    objects.extend(pdf_objects);
    let px = rgb_at(assemble_pdf(&objects), &BARS);
    assert_eq!(px, [RED, WHITE, BLUE]);
}

#[test]
fn content_after_the_group_closes_is_drawn() {
    let px = rgb_at(
        page(
            TWO_GROUPS,
            "/Properties << /pOff 5 0 R /pOn 6 0 R >>",
            "/OC /pOff BDC 1 0 0 rg 0 0 20 100 re f EMC \
             /OC /pOn BDC 0 1 0 rg 40 0 20 100 re f EMC \
             0 0 1 rg 80 0 20 100 re f",
            groups(),
        ),
        &BARS,
    );
    assert_eq!(px, [WHITE, GREEN, BLUE]);
}
