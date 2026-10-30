//! Annotation appearances (ISO 32000-2 §12.5): the annotation flags decide
//! whether an annotation is drawn for the render purpose (§12.5.3), the
//! appearance stream is mapped onto `/Rect` by Algorithm 8.1 (§12.5.5), a
//! state-keyed appearance without `/AS` falls back to the field value and
//! then to `Off`, `/Annots` may hold annotation dictionaries directly, and
//! a markup annotation without a normal appearance is drawn from its
//! geometry and colours (§12.5.6).

#[expect(dead_code, reason = "shared helpers; this test uses only PDF assembly")]
mod common;

use common::{assemble_pdf, stream_object};
use rasterrocket::{
    BackendPolicy, RenderPurpose, SessionConfig, open_session_from_bytes, render_page_rgb,
};

/// A 100 × 100 pt page, rendered at one pixel per point.
const W: u32 = 100;
const H: u32 = 100;

const WHITE: [u8; 3] = [255, 255, 255];
const BLUE: [u8; 3] = [0, 0, 255];
const GREY: [u8; 3] = [128, 128, 128];

/// A blank page whose `/Annots` is `annots`, with `extra` objects numbered
/// from 4 on.
fn page(annots: &str, extra: Vec<Vec<u8>>) -> Vec<u8> {
    page_with(annots, "", extra)
}

/// [`page`] with additional page dictionary entries `page_entries`.
fn page_with(annots: &str, page_entries: &str, extra: Vec<Vec<u8>>) -> Vec<u8> {
    let mut objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W} {H}] \
             /Annots {annots} {page_entries} >>"
        )
        .into_bytes(),
    ];
    objects.extend(extra);
    assemble_pdf(&objects)
}

/// A form `XObject` with `dict` entries and `content`.
fn form(dict: &str, content: &[u8]) -> Vec<u8> {
    stream_object(&format!("/Type /XObject /Subtype /Form {dict}"), content)
}

/// A 20 × 20 appearance filled blue.
fn blue_square() -> Vec<u8> {
    form("/BBox [0 0 20 20]", b"0 0 1 rg 0 0 20 20 re f")
}

/// RGB at user-space `(x, y)` (origin bottom-left), rendered for `purpose`.
fn rgb_at(pdf: Vec<u8>, purpose: RenderPurpose, points: &[(u32, u32)]) -> Vec<[u8; 3]> {
    let mut config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    config.purpose = purpose;
    let s = open_session_from_bytes(pdf, &config).expect("open session");
    let bmp = render_page_rgb(&s, 1, 1.0).expect("render page");
    let (w, h) = (bmp.width, bmp.height);
    let data = bmp.data();
    points
        .iter()
        .map(|&(x, y)| {
            assert!(x < w && y < h);
            let i = (h - 1 - y) as usize * bmp.stride + x as usize * 3;
            [data[i], data[i + 1], data[i + 2]]
        })
        .collect()
}

/// Whether a `/Square` annotation with `/F flags` over (10..30, 10..30) is
/// drawn for `purpose`.
fn drawn_with_flags(flags: u32, purpose: RenderPurpose) -> bool {
    let pdf = page(
        "[4 0 R]",
        vec![
            format!("<< /Type /Annot /Subtype /Square /Rect [10 10 30 30] /F {flags} /AP << /N 5 0 R >> >>")
                .into_bytes(),
            blue_square(),
        ],
    );
    let px = rgb_at(pdf, purpose, &[(20, 20)])[0];
    assert!(px == BLUE || px == WHITE, "{px:?}");
    px == BLUE
}

const HIDDEN: u32 = 2;
const PRINT: u32 = 4;
const NO_VIEW: u32 = 32;

#[test]
fn the_flags_decide_whether_an_annotation_is_viewed() {
    assert!(drawn_with_flags(0, RenderPurpose::View));
    assert!(drawn_with_flags(PRINT, RenderPurpose::View));
    assert!(!drawn_with_flags(HIDDEN, RenderPurpose::View));
    assert!(!drawn_with_flags(HIDDEN | PRINT, RenderPurpose::View));
    assert!(!drawn_with_flags(NO_VIEW, RenderPurpose::View));
    assert!(!drawn_with_flags(NO_VIEW | PRINT, RenderPurpose::View));
}

#[test]
fn annotations_off_draws_the_page_content_alone() {
    let pdf = page(
        "[4 0 R]",
        vec![
            b"<< /Type /Annot /Subtype /Square /Rect [10 10 30 30] /AP << /N 5 0 R >> >>".to_vec(),
            blue_square(),
        ],
    );
    for (annotations, want) in [(true, BLUE), (false, WHITE)] {
        let mut config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
        config.annotations = annotations;
        let s = open_session_from_bytes(pdf.clone(), &config).expect("open session");
        let bmp = render_page_rgb(&s, 1, 1.0).expect("render page");
        let i = (H - 1 - 20) as usize * bmp.stride + 20 * 3;
        assert_eq!(bmp.data()[i..i + 3], want, "annotations = {annotations}");
    }
}

#[test]
fn the_flags_decide_whether_an_annotation_is_printed() {
    assert!(!drawn_with_flags(0, RenderPurpose::Print));
    assert!(drawn_with_flags(PRINT, RenderPurpose::Print));
    assert!(drawn_with_flags(NO_VIEW | PRINT, RenderPurpose::Print));
    assert!(!drawn_with_flags(HIDDEN | PRINT, RenderPurpose::Print));
}

#[test]
fn the_appearance_matrix_transforms_the_bbox_before_it_is_fitted() {
    // A 50 × 20 appearance rotated a quarter turn by its /Matrix is 20 × 50,
    // which fills the 20 × 50 rect at (40, 25) without scaling.
    let pdf = page(
        "[4 0 R]",
        vec![
            b"<< /Type /Annot /Subtype /Stamp /Rect [40 25 60 75] /AP << /N 5 0 R >> >>".to_vec(),
            form(
                "/BBox [0 0 50 20] /Matrix [0 1 -1 0 0 0]",
                b"0 0 1 rg 0 0 50 20 re f",
            ),
        ],
    );
    let px = rgb_at(
        pdf,
        RenderPurpose::View,
        &[
            (50, 50),
            (42, 27),
            (58, 73),
            (35, 50),
            (65, 50),
            (50, 20),
            (50, 80),
        ],
    );
    assert_eq!(px, [BLUE, BLUE, BLUE, WHITE, WHITE, WHITE, WHITE]);
}

/// A state-keyed appearance `/N << /On 5 0 R /Off 6 0 R >>` (blue and grey)
/// on an annotation with the entries `entries`, plus `extra` objects from 7.
fn state_keyed(entries: &str, extra: Vec<Vec<u8>>) -> [u8; 3] {
    let mut objects = vec![
        format!(
            "<< /Type /Annot /Subtype /Widget /Rect [10 10 30 30] \
             /AP << /N << /On 5 0 R /Off 6 0 R >> >> {entries} >>"
        )
        .into_bytes(),
        blue_square(),
        form("/BBox [0 0 20 20]", b"0.5 g 0 0 20 20 re f"),
    ];
    objects.extend(extra);
    rgb_at(page("[4 0 R]", objects), RenderPurpose::View, &[(20, 20)])[0]
}

#[test]
fn a_state_keyed_appearance_uses_its_appearance_state() {
    assert_eq!(state_keyed("/AS /On", vec![]), BLUE);
    assert_eq!(state_keyed("/AS /Off /V /On", vec![]), GREY);
}

#[test]
fn a_state_keyed_appearance_without_a_state_uses_the_field_value() {
    assert_eq!(state_keyed("/V /On", vec![]), BLUE);
    assert_eq!(
        state_keyed("/Parent 7 0 R", vec![b"<< /FT /Btn /V /On >>".to_vec()]),
        BLUE
    );
}

#[test]
fn a_state_keyed_appearance_without_a_state_or_value_uses_off() {
    assert_eq!(state_keyed("", vec![]), GREY);
}

#[test]
fn a_direct_annotation_dictionary_is_drawn() {
    let pdf = page(
        "[<< /Type /Annot /Subtype /Square /Rect [10 10 30 30] /AP << /N 4 0 R >> >>]",
        vec![blue_square()],
    );
    assert_eq!(rgb_at(pdf, RenderPurpose::View, &[(20, 20)]), [BLUE]);
}

#[test]
fn a_no_rotate_annotation_stays_upright_on_a_rotated_page() {
    // On a page rotated a quarter turn clockwise, user point (x, y) shows
    // at device point (y, x) counted from the top-left.  A NoRotate
    // annotation keeps the device position of its rect's upper-left corner
    // and is drawn upright from there; without NoRotate it turns with the
    // page.
    let annot = |flags: u32| {
        page_with(
            "[4 0 R]",
            "/Rotate 90",
            vec![
                format!(
                    "<< /Type /Annot /Subtype /Stamp /Rect [10 0 50 20] /F {flags} \
                     /AP << /N 5 0 R >> >>"
                )
                .into_bytes(),
                form("/BBox [0 0 40 20]", b"0 0 1 rg 0 0 40 20 re f"),
            ],
        )
    };
    // `rgb_at` counts y from the bottom.
    let dev = |dx: u32, dy: u32| (dx, H - 1 - dy);
    let probes = [dev(10, 45), dev(45, 20)];
    // Turned with the page, the rect spans device x 0..20, y 10..50.
    let rotated = rgb_at(annot(0), RenderPurpose::View, &probes);
    assert_eq!(rotated, [BLUE, WHITE]);
    // The upper-left corner (10, 20) is at device (20, 10); upright, the
    // 40 × 20 appearance spans device x 20..60, y 10..30.
    let upright = rgb_at(annot(16), RenderPurpose::View, &probes);
    assert_eq!(upright, [WHITE, BLUE]);
}

/// RGB at `points` of a page whose only annotation is the direct
/// dictionary with entries `annot` and no appearance stream.
fn synthesised(annot: &str, points: &[(u32, u32)]) -> Vec<[u8; 3]> {
    let pdf = page(&format!("[<< /Type /Annot {annot} >>]"), vec![]);
    rgb_at(pdf, RenderPurpose::View, points)
}

const RED: [u8; 3] = [255, 0, 0];
const YELLOW: [u8; 3] = [255, 255, 0];
const BLACK: [u8; 3] = [0, 0, 0];

#[test]
fn a_square_without_an_appearance_is_drawn_from_its_colours() {
    let px = synthesised(
        "/Subtype /Square /Rect [10 10 60 60] /C [1 0 0] /IC [0 0 1] /BS << /W 4 >>",
        &[(35, 35), (11, 35), (58, 35), (5, 35)],
    );
    assert_eq!(px, [BLUE, RED, RED, WHITE]);
}

#[test]
fn a_circle_without_an_appearance_is_an_inscribed_ellipse() {
    let px = synthesised(
        "/Subtype /Circle /Rect [10 10 90 90] /C [0 0 0] /IC [1 1 0]",
        &[(50, 50), (12, 12), (50, 10)],
    );
    // The centre is filled, the corner outside the ellipse is not, and the
    // ellipse's border touches the middle of the bottom edge.
    assert_eq!(px, [YELLOW, WHITE, BLACK]);
}

#[test]
fn an_ink_annotation_without_an_appearance_strokes_its_paths() {
    let px = synthesised(
        "/Subtype /Ink /Rect [10 10 90 90] /C [0 0 1] /InkList [[20 50 80 50]] /BS << /W 4 >>",
        &[(50, 50), (50, 51), (50, 60)],
    );
    assert_eq!(px, [BLUE, BLUE, WHITE]);
}

#[test]
fn a_highlight_without_an_appearance_multiplies_its_colour_over_the_page() {
    let px = synthesised(
        "/Subtype /Highlight /Rect [10 40 90 60] /QuadPoints [10 60 90 60 10 40 90 40]",
        &[(50, 50), (50, 70)],
    );
    assert_eq!(px, [YELLOW, WHITE]);
}

#[test]
fn an_underline_and_a_strike_out_are_drawn_at_the_bottom_and_middle() {
    let quad = "/Rect [10 40 90 60] /QuadPoints [10 60 90 60 10 40 90 40]";
    // A 1 pt line on a pixel boundary covers two pixel rows half each.
    let inked = |px: Vec<[u8; 3]>| px.iter().map(|p| p[0] < 200).collect::<Vec<_>>();
    let probes = [(50, 41), (50, 50)];
    let under = synthesised(&format!("/Subtype /Underline {quad}"), &probes);
    assert_eq!(inked(under), [true, false]);
    let strike = synthesised(&format!("/Subtype /StrikeOut {quad}"), &probes);
    assert_eq!(inked(strike), [false, true]);
}

#[test]
fn a_text_annotation_without_an_appearance_draws_a_note_icon() {
    let px = synthesised("/Subtype /Text /Rect [10 70 30 90]", &[(12, 86), (50, 50)]);
    assert_eq!(px, [YELLOW, WHITE]);
}

#[test]
fn a_constant_opacity_applies_to_a_synthesised_appearance() {
    let px = synthesised(
        "/Subtype /Square /Rect [10 10 60 60] /IC [0 0 0] /C [] /CA 0.5",
        &[(35, 35)],
    );
    assert!((126..=129).contains(&px[0][0]), "{:?}", px[0]);
}

#[test]
fn an_annotation_with_an_appearance_is_not_synthesised() {
    // The appearance fills blue; a synthesised square would have a red border.
    let pdf = page(
        "[4 0 R]",
        vec![
            b"<< /Type /Annot /Subtype /Square /Rect [10 10 30 30] /C [1 0 0] /AP << /N 5 0 R >> >>"
                .to_vec(),
            blue_square(),
        ],
    );
    assert_eq!(rgb_at(pdf, RenderPurpose::View, &[(10, 20)]), [BLUE]);
}
