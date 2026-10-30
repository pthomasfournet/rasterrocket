//! The page group (ISO 32000-2 §11.4.7): the page's content is a
//! transparency group composited onto the white medium once complete, so a
//! blend mode applied on the page sees only what the page has painted, not
//! the paper, and a page whose `/Group` has `/K true` is a knockout group.

#[expect(dead_code, reason = "shared helpers; this test uses only PDF assembly")]
mod common;

use common::{one_page, stream_object};
use rasterrocket::{BackendPolicy, render_page_rgb};

/// A 100 × 100 pt page, rendered at one pixel per point.
const W: u32 = 100;
const H: u32 = 100;

/// A page with entries `page_entries`, `ExtGState` resources `gstates` and
/// content `content`.
fn page(page_entries: &str, gstates: &str, content: &str) -> Vec<u8> {
    one_page(
        &format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W} {H}] {page_entries} \
             /Resources << /ExtGState << {gstates} >> >> /Contents 4 0 R >>"
        ),
        vec![stream_object("", content.as_bytes())],
    )
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

const WHITE: [u8; 3] = [255, 255, 255];
const GREY: [u8; 3] = [128, 128, 128];

/// A grey bar over the lower half of the page, then a blue square screened
/// over it and over the blank upper half.
const SCREENED_SQUARE: &str = "0.5 g 0 0 100 50 re f q /S gs 0 0 1 rg 25 25 50 50 re f Q";

/// Screen leaves blue over the blank paper, whatever the page's `/Group`
/// says, and screens it with the grey where the page painted grey.
#[test]
fn a_blend_mode_on_the_page_does_not_see_the_paper() {
    for group in [
        "",
        "/Group << /S /Transparency /CS /DeviceRGB /I true >>",
        "/Group << /S /Transparency /CS /DeviceRGB /I false >>",
    ] {
        let px = rgb_at(
            page(group, "/S << /BM /Screen >>", SCREENED_SQUARE),
            &[(50, 60), (50, 35), (10, 90), (10, 10)],
        );
        assert_eq!(px[0], [0, 0, 255], "{group}: blue over blank paper");
        assert_eq!(px[1], [128, 128, 255], "{group}: blue screened with grey");
        assert_eq!(px[2], WHITE, "{group}: unpainted paper");
        assert_eq!(px[3], GREY, "{group}: the grey bar");
    }
}

/// Without a blend mode in reach the page composites onto white directly:
/// half-alpha black over paper is mid-grey either way.
#[test]
fn constant_alpha_on_the_page_composites_onto_white() {
    let px = rgb_at(
        page("", "/A << /ca 0.5 >>", "/A gs 0 g 25 25 50 50 re f"),
        &[(50, 50), (10, 10)],
    );
    assert!(
        px[0].iter().all(|&v| v.abs_diff(128) <= 1),
        "onto white: {:?}",
        px[0]
    );
    assert_eq!(px[1], WHITE);
    let px = rgb_at(
        page(
            "",
            "/A << /ca 0.5 >> /M << /BM /Multiply >>",
            "/A gs 0 g 25 25 50 50 re f",
        ),
        &[(50, 50), (10, 10)],
    );
    assert!(
        px[0].iter().all(|&v| v.abs_diff(128) <= 1),
        "as a group: {:?}",
        px[0]
    );
    assert_eq!(px[1], WHITE);
}

/// A knockout page group: two half-alpha black squares overlapping at
/// 40..60 leave one layer where they overlap.
#[test]
fn a_knockout_page_group_knocks_out() {
    let content = "/A gs 0 g 20 20 40 40 re f 40 40 40 40 re f";
    let stacked = rgb_at(page("", "/A << /ca 0.5 >>", content), &[(50, 50)]);
    assert!(
        stacked[0].iter().all(|&v| v.abs_diff(64) <= 1),
        "two layers over white: {:?}",
        stacked[0]
    );
    let knocked = rgb_at(
        page(
            "/Group << /S /Transparency /K true >>",
            "/A << /ca 0.5 >>",
            content,
        ),
        &[(50, 50), (30, 30), (10, 10)],
    );
    for (what, px) in [
        ("the second square alone", knocked[0]),
        ("the first square", knocked[1]),
    ] {
        assert!(px.iter().all(|&v| v.abs_diff(128) <= 1), "{what}: {px:?}");
    }
    assert_eq!(knocked[2], WHITE, "unpainted paper");
}
