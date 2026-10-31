//! Annotation appearances (ISO 32000-2 §12.5): the annotation flags decide
//! whether an annotation is drawn for the render purpose (§12.5.3), the
//! appearance stream is mapped onto `/Rect` by Algorithm 8.1 (§12.5.5), a
//! state-keyed appearance without `/AS` falls back to the field value and
//! then to `Off`, `/Annots` may hold annotation dictionaries directly, and
//! a markup annotation without a normal appearance is drawn from its
//! geometry and colours (§12.5.6).  Appearances are drawn with the initial
//! graphics state, whatever the page content leaves behind (§12.5.5).

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use common::{rgb_at_for, stream_object, with_page_tree};
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
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W} {H}] \
             /Annots {annots} {page_entries} >>"
        )
        .into_bytes(),
    ];
    objects.extend(extra);
    with_page_tree(&objects)
}

/// A form `XObject` with `dict` entries and `content`.
fn form(dict: &str, content: &[u8]) -> Vec<u8> {
    stream_object(&format!("/Type /XObject /Subtype /Form {dict}"), content)
}

/// A 20 × 20 appearance filled blue.
fn blue_square() -> Vec<u8> {
    form("/BBox [0 0 20 20]", b"0 0 1 rg 0 0 20 20 re f")
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
    let px = rgb_at_for(pdf, purpose, &[(20, 20)])[0];
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
    let px = rgb_at_for(
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
    rgb_at_for(page("[4 0 R]", objects), RenderPurpose::View, &[(20, 20)])[0]
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

/// Without `/AS`, a field value that names no appearance state selects
/// `Off`, as does no value at all.
#[test]
fn a_state_keyed_appearance_without_a_state_or_value_uses_off() {
    assert_eq!(state_keyed("", vec![]), GREY);
    assert_eq!(state_keyed("/V /Maybe", vec![]), GREY);
    assert_eq!(
        state_keyed("/Parent 7 0 R", vec![b"<< /FT /Btn /V /Maybe >>".to_vec()]),
        GREY
    );
}

#[test]
fn a_direct_annotation_dictionary_is_drawn() {
    let pdf = page(
        "[<< /Type /Annot /Subtype /Square /Rect [10 10 30 30] /AP << /N 4 0 R >> >>]",
        vec![blue_square()],
    );
    assert_eq!(rgb_at_for(pdf, RenderPurpose::View, &[(20, 20)]), [BLUE]);
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
    let rotated = rgb_at_for(annot(0), RenderPurpose::View, &probes);
    assert_eq!(rotated, [BLUE, WHITE]);
    // The upper-left corner (10, 20) is at device (20, 10); upright, the
    // 40 × 20 appearance spans device x 20..60, y 10..30.
    let upright = rgb_at_for(annot(16), RenderPurpose::View, &probes);
    assert_eq!(upright, [WHITE, BLUE]);
}

/// RGB at `points` of a page whose only annotation is the direct
/// dictionary with entries `annot` and no appearance stream.
fn synthesised(annot: &str, points: &[(u32, u32)]) -> Vec<[u8; 3]> {
    let pdf = page(&format!("[<< /Type /Annot {annot} >>]"), vec![]);
    rgb_at_for(pdf, RenderPurpose::View, points)
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
    // A mid-grey band across the middle of the highlighted strip: multiply
    // keeps the band visible under the yellow, where a normal blend would
    // paint it over.
    let pdf = page_with(
        "[<< /Type /Annot /Subtype /Highlight /Rect [10 40 90 60] \
         /QuadPoints [10 60 90 60 10 40 90 40] >>]",
        "/Contents 4 0 R",
        vec![stream_object("", b"0.5 g 10 45 80 10 re f")],
    );
    let px = rgb_at_for(pdf, RenderPurpose::View, &[(50, 50), (50, 58), (50, 70)]);
    assert_eq!(px, [[128, 128, 0], YELLOW, WHITE]);
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
fn line_endings_are_drawn_at_both_ends_of_a_line() {
    // A 2 pt line from x 20 to 80 at y 50: a closed arrow filled blue at
    // its start and a 12 pt square at its end.  The arrow's wings reach
    // 12 pt back along the line and 6.9 pt either side of it.
    let px = synthesised(
        "/Subtype /Line /Rect [10 30 90 70] /L [20 50 80 50] /BS << /W 2 >> \
         /C [1 0 0] /IC [0 0 1] /LE [/ClosedArrow /Square]",
        &[(29, 53), (29, 60), (80, 53), (50, 55)],
    );
    assert_eq!(px, [BLUE, WHITE, BLUE, WHITE]);
    let none = synthesised(
        "/Subtype /Line /Rect [10 30 90 70] /L [20 50 80 50] /BS << /W 2 >> \
         /C [1 0 0] /IC [0 0 1]",
        &[(29, 53), (80, 53)],
    );
    assert_eq!(none, [WHITE, WHITE]);
}

#[test]
fn a_poly_line_carries_open_arrows_along_its_end_segments() {
    // The last segment runs up from (50 20) to (50 80); its open arrow's
    // wings run back down to y 74, 3.5 pt either side.
    let px = synthesised(
        "/Subtype /PolyLine /Rect [10 10 90 90] /Vertices [20 20 50 20 50 80] \
         /C [0 0 1] /LE [/None /OpenArrow]",
        &[(47, 75), (52, 75), (50, 85)],
    );
    let inked = |p: [u8; 3]| p[0] < 200;
    assert!(inked(px[0]) && inked(px[1]), "{px:?}");
    assert_eq!(px[2], WHITE);
}

/// The number of points of the 1 pt grid in the user-space box
/// `[x0 y0 x1 y1]` of a page whose only annotation is `annot` (direct, no
/// appearance stream) that `pick` selects.
fn count_synthesised(annot: &str, b: [u32; 4], pick: impl Fn([u8; 3]) -> bool) -> usize {
    let points: Vec<(u32, u32)> = (b[1]..b[3])
        .flat_map(|y| (b[0]..b[2]).map(move |x| (x, y)))
        .collect();
    synthesised(annot, &points)
        .into_iter()
        .filter(|&p| pick(p))
        .count()
}

#[test]
fn a_free_text_annotation_without_an_appearance_shows_its_text() {
    let annot = "/Subtype /FreeText /Rect [10 10 90 90] /C [1 1 0] \
                 /DA (/Helv 12 Tf 0 0 1 rg)";
    let bluish = |p: [u8; 3]| p[2] > 150 && p[0] < 150;
    // Text runs from the top of the box, inside the 1 pt border and the
    // 2 pt margin; the bottom of the box stays the /C fill.
    let text = [13, 60, 87, 87];
    let with = count_synthesised(
        &format!("{annot} /Contents (Hello Hello Hello)"),
        text,
        bluish,
    );
    let without = count_synthesised(annot, text, bluish);
    assert!(with > 40, "{with} text pixels");
    assert_eq!(without, 0);
    let px = synthesised(
        &format!("{annot} /Contents (Hello)"),
        &[(50, 15), (10, 50), (5, 50)],
    );
    assert_eq!(px, [YELLOW, BLUE, WHITE]);
}

#[test]
fn free_text_falls_back_to_the_plain_text_of_its_rich_text() {
    let annot = "/Subtype /FreeText /Rect [10 10 90 90] /DA (/Helv 12 Tf 0 g) \
                 /RC (<body><p>Rich</p><p>text</p></body>)";
    let dark = count_synthesised(annot, [13, 13, 87, 87], |p| p[0] < 100);
    assert!(dark > 30, "{dark} text pixels");
}

#[test]
fn a_free_text_callout_line_runs_to_its_first_point() {
    let px = synthesised(
        "/Subtype /FreeText /Rect [10 10 90 90] /RD [40 40 0 0] /CL [15 15 30 30 50 30] \
         /DA (/Helv 12 Tf 1 0 0 rg) /BS << /W 2 >> /Contents (x)",
        &[(22, 22), (40, 30), (20, 40)],
    );
    // The callout is stroked in the text colour; the area beside it, outside
    // the box moved in by /RD, is not painted.
    assert_eq!(px, [RED, RED, WHITE]);
}

/// A page with a note (object 4) whose pop-up (object 5) has the entries
/// `popup` and the window `[20 20 80 80]`; the note has the flags
/// `note_flags`.  Object 6 is a blue appearance.
fn note_with_popup(popup: &str, note_flags: u32) -> Vec<u8> {
    page(
        "[4 0 R 5 0 R]",
        vec![
            format!(
                "<< /Type /Annot /Subtype /Text /Rect [0 80 20 100] /F {note_flags} \
                 /T (Reviewer) /Contents (Looks good) /Popup 5 0 R >>"
            )
            .into_bytes(),
            format!("<< /Type /Annot /Subtype /Popup /Parent 4 0 R /Rect [20 20 80 80] {popup} >>")
                .into_bytes(),
            form("/BBox [0 0 60 60]", b"0 0 1 rg 0 0 60 60 re f"),
        ],
    )
}

#[test]
fn a_closed_popup_is_not_drawn_even_with_an_appearance() {
    let probe = &[(50, 30)];
    let closed = rgb_at_for(
        note_with_popup("/AP << /N 6 0 R >>", 0),
        RenderPurpose::View,
        probe,
    );
    assert_eq!(closed, [WHITE]);
    let open = rgb_at_for(
        note_with_popup("/Open true /AP << /N 6 0 R >>", 0),
        RenderPurpose::View,
        probe,
    );
    assert_eq!(open, [BLUE]);
}

#[test]
fn an_open_popup_without_an_appearance_shows_its_parents_text() {
    let pdf = || note_with_popup("/Open true", 0);
    let px = rgb_at_for(pdf(), RenderPurpose::View, &[(50, 25), (20, 50), (10, 50)]);
    assert_eq!(px, [YELLOW, BLACK, WHITE]);
    // The title and contents are set from the top of the window.
    let points: Vec<(u32, u32)> = (50..77)
        .flat_map(|y| (24..77).map(move |x| (x, y)))
        .collect();
    let dark = rgb_at_for(pdf(), RenderPurpose::View, &points)
        .iter()
        .filter(|p| p[0] < 100)
        .count();
    assert!(dark > 60, "{dark} text pixels");
    // A pop-up of a hidden note is hidden with it.
    let hidden = rgb_at_for(
        note_with_popup("/Open true", HIDDEN),
        RenderPurpose::View,
        &[(50, 25)],
    );
    assert_eq!(hidden, [WHITE]);
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
    assert_eq!(rgb_at_for(pdf, RenderPurpose::View, &[(10, 20)]), [BLUE]);
}

/// Page content that leaves an unbalanced `q`, a doubled CTM and a clip of
/// the lower-left 10 × 10 units does not move or clip the annotation drawn
/// after it.
#[test]
fn an_annotation_is_drawn_with_the_initial_graphics_state() {
    let pdf = page_with(
        "[4 0 R]",
        "/Contents 6 0 R",
        vec![
            b"<< /Type /Annot /Subtype /Square /Rect [40 40 60 60] /F 4 /AP << /N 5 0 R >> >>"
                .to_vec(),
            blue_square(),
            stream_object("", b"q 2 0 0 2 0 0 cm 0 0 10 10 re W n 0 g 0 0 10 10 re f"),
        ],
    );
    let px = rgb_at_for(pdf, RenderPurpose::View, &[(50, 50), (10, 10), (30, 30)]);
    assert_eq!(px[0], BLUE, "the appearance at its rect");
    assert_eq!(px[1], [0, 0, 0], "the content's square, scaled by its CTM");
    assert_eq!(px[2], WHITE, "nothing between them");
}
