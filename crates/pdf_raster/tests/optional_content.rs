//! Optional content (ISO 32000-2 §8.11): content governed by a group that is
//! OFF in the default configuration, or by a membership dictionary that
//! evaluates to hidden, is not drawn, whether it is a `BDC /OC` sequence, an
//! `XObject` or an annotation; and every marked-content sequence (§14.6),
//! optional or not, is closed by exactly one `EMC`.

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

/// RGB at user-space `(x, y)` (origin bottom-left), rendered for viewing.
fn rgb_at(pdf: Vec<u8>, points: &[(u32, u32)]) -> Vec<[u8; 3]> {
    rgb_at_for(pdf, RenderPurpose::View, points)
}

/// RGB at user-space `(x, y)` (origin bottom-left), rendered for `purpose`.
fn rgb_at_for(pdf: Vec<u8>, purpose: RenderPurpose, points: &[(u32, u32)]) -> Vec<[u8; 3]> {
    let mut config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    config.purpose = purpose;
    let s = open_session_from_bytes(pdf, &config).expect("open session");
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

/// A red bar, the middle bar governed by `BDC /OC /M` with `/M` bound to
/// `property` in the page's `Properties`, and a blue bar; the catalog's
/// `/OCProperties` is `ocprops` and `extra` objects are numbered from 5 on.
fn governed_bar(ocprops: &str, property: &str, extra: Vec<Vec<u8>>) -> Vec<u8> {
    page(
        ocprops,
        &format!("/Properties << /M {property} >>"),
        "1 0 0 rg 0 0 20 100 re f \
         /OC /M BDC 0 1 0 rg 40 0 20 100 re f EMC \
         0 0 1 rg 80 0 20 100 re f",
        extra,
    )
}

#[test]
fn a_form_xobject_with_a_hidden_oc_is_not_drawn() {
    let px = rgb_at(
        page(
            TWO_GROUPS,
            "/XObject << /F 7 0 R >>",
            "1 0 0 rg 0 0 20 100 re f /F Do 0 0 1 rg 80 0 20 100 re f",
            groups_and_form("/OC 5 0 R", b"0 1 0 rg 40 0 20 100 re f"),
        ),
        &BARS,
    );
    assert_eq!(px, [RED, WHITE, BLUE]);
}

#[test]
fn a_form_xobject_with_a_visible_oc_is_drawn() {
    let px = rgb_at(
        page(
            TWO_GROUPS,
            "/XObject << /F 7 0 R >>",
            "/F Do",
            groups_and_form("/OC 6 0 R", b"0 1 0 rg 40 0 20 100 re f"),
        ),
        &BARS,
    );
    assert_eq!(px, [WHITE, GREEN, WHITE]);
}

#[test]
fn an_image_xobject_with_a_hidden_oc_is_not_drawn() {
    let mut extra = groups();
    extra.push(stream_object(
        "/Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceRGB \
         /BitsPerComponent 8 /OC 5 0 R",
        &[0, 255, 0],
    ));
    let px = rgb_at(
        page(
            TWO_GROUPS,
            "/XObject << /I 7 0 R >>",
            "1 0 0 rg 0 0 20 100 re f q 20 0 0 100 40 0 cm /I Do Q \
             0 0 1 rg 80 0 20 100 re f",
            extra,
        ),
        &BARS,
    );
    assert_eq!(px, [RED, WHITE, BLUE]);
}

#[test]
fn an_annotation_with_a_hidden_oc_is_not_drawn() {
    let mut objects = vec![
        format!("<< /Type /Catalog /Pages 2 0 R /OCProperties {TWO_GROUPS} >>").into_bytes(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W} {H}] \
             /Contents 4 0 R /Annots [7 0 R 8 0 R] >>"
        )
        .into_bytes(),
        stream_object("", b"1 0 0 rg 0 0 20 100 re f"),
    ];
    objects.extend(groups());
    // Objects 7 and 8: annotations over the middle and right bars whose
    // appearance (object 9) fills the rect; the first is in the OFF group.
    objects.push(
        b"<< /Type /Annot /Subtype /Square /Rect [40 0 60 100] /OC 5 0 R /AP << /N 9 0 R >> >>"
            .to_vec(),
    );
    objects.push(
        b"<< /Type /Annot /Subtype /Square /Rect [80 0 100 100] /OC 6 0 R /AP << /N 9 0 R >> >>"
            .to_vec(),
    );
    objects.push(stream_object(
        "/Type /XObject /Subtype /Form /BBox [0 0 20 100]",
        b"0 0 1 rg 0 0 20 100 re f",
    ));
    let px = rgb_at(assemble_pdf(&objects), &BARS);
    assert_eq!(px, [RED, WHITE, BLUE]);
}

#[test]
fn a_membership_dictionary_hides_when_its_group_is_off() {
    // /P defaults to AnyOn.
    let px = rgb_at(
        governed_bar(TWO_GROUPS, "<< /Type /OCMD /OCGs 5 0 R >>", groups()),
        &BARS,
    );
    assert_eq!(px, [RED, WHITE, BLUE]);
}

#[test]
fn a_membership_dictionary_applies_its_policy() {
    let cases = [
        ("/AnyOn", GREEN),
        ("/AllOn", WHITE),
        ("/AnyOff", GREEN),
        ("/AllOff", WHITE),
    ];
    for (policy, middle) in cases {
        let ocmd = format!("<< /Type /OCMD /OCGs [5 0 R 6 0 R] /P {policy} >>");
        let px = rgb_at(governed_bar(TWO_GROUPS, &ocmd, groups()), &BARS);
        assert_eq!(px, [RED, middle, BLUE], "/P {policy}");
    }
}

#[test]
fn an_indirect_membership_dictionary_is_evaluated() {
    let mut extra = groups();
    extra.push(b"<< /Type /OCMD /OCGs [5 0 R] /P /AllOn >>".to_vec());
    let px = rgb_at(governed_bar(TWO_GROUPS, "7 0 R", extra), &BARS);
    assert_eq!(px, [RED, WHITE, BLUE]);
}

#[test]
fn a_visibility_expression_takes_precedence_over_the_policy() {
    let cases = [
        ("[/Not 6 0 R]", WHITE),
        ("[/Not 5 0 R]", GREEN),
        ("[/And 5 0 R 6 0 R]", WHITE),
        ("[/Or 5 0 R 6 0 R]", GREEN),
        ("[/And [/Not 5 0 R] 6 0 R]", GREEN),
    ];
    for (ve, middle) in cases {
        // /OCGs and /P alone would show the content.
        let ocmd = format!("<< /Type /OCMD /OCGs [6 0 R] /VE {ve} >>");
        let px = rgb_at(governed_bar(TWO_GROUPS, &ocmd, groups()), &BARS);
        assert_eq!(px, [RED, middle, BLUE], "/VE {ve}");
    }
}

#[test]
fn base_state_off_hides_every_group_not_listed_in_on() {
    let ocprops = "<< /OCGs [5 0 R 6 0 R] /D << /BaseState /OFF /ON [6 0 R] >> >>";
    assert_eq!(
        rgb_at(governed_bar(ocprops, "5 0 R", groups()), &BARS),
        [RED, WHITE, BLUE]
    );
    assert_eq!(
        rgb_at(governed_bar(ocprops, "6 0 R", groups()), &BARS),
        [RED, GREEN, BLUE]
    );
}

#[test]
fn a_group_listed_in_both_on_and_off_is_off() {
    let ocprops = "<< /OCGs [5 0 R] /D << /ON [5 0 R] /OFF [5 0 R] >> >>";
    let px = rgb_at(governed_bar(ocprops, "5 0 R", groups()), &BARS);
    assert_eq!(px, [RED, WHITE, BLUE]);
}

#[test]
fn a_group_whose_intent_the_configuration_excludes_is_ignored() {
    // Group 5 is OFF but has intent Design; the configuration's intent
    // defaults to View, so the group does not affect visibility.
    let extra = vec![
        b"<< /Type /OCG /Name (design) /Intent /Design >>".to_vec(),
        ON_GROUP.to_vec(),
    ];
    let px = rgb_at(governed_bar(TWO_GROUPS, "5 0 R", extra.clone()), &BARS);
    assert_eq!(px, [RED, GREEN, BLUE]);
    // A configuration with intent All considers it.
    let all = "<< /OCGs [5 0 R 6 0 R] /D << /OFF [5 0 R] /Intent /All >> >>";
    let px = rgb_at(governed_bar(all, "5 0 R", extra), &BARS);
    assert_eq!(px, [RED, WHITE, BLUE]);
}

/// Group 5 is ON in `/D`, but its usage dictionary turns it off for viewing
/// and on for printing, and `/AS` applies both usages.
const USAGE_GROUPS: &str = "<< /OCGs [5 0 R] /D << /ON [5 0 R] /AS [ \
     << /Event /View /OCGs [5 0 R] /Category [/View] >> \
     << /Event /Print /OCGs [5 0 R] /Category [/Print] >> ] >> >>";

/// Object 5: the group of [`USAGE_GROUPS`].
fn usage_group() -> Vec<Vec<u8>> {
    vec![
        b"<< /Type /OCG /Name (screen-off) \
           /Usage << /View << /ViewState /OFF >> /Print << /PrintState /ON >> >> >>"
            .to_vec(),
    ]
}

#[test]
fn the_view_usage_applies_when_rendering_for_viewing() {
    let px = rgb_at_for(
        governed_bar(USAGE_GROUPS, "5 0 R", usage_group()),
        RenderPurpose::View,
        &BARS,
    );
    assert_eq!(px, [RED, WHITE, BLUE]);
}

#[test]
fn the_print_usage_applies_when_rendering_for_printing() {
    let px = rgb_at_for(
        governed_bar(USAGE_GROUPS, "5 0 R", usage_group()),
        RenderPurpose::Print,
        &BARS,
    );
    assert_eq!(px, [RED, GREEN, BLUE]);
}

#[test]
fn a_usage_dictionary_does_not_apply_without_an_as_entry() {
    let ocprops = "<< /OCGs [5 0 R] /D << /ON [5 0 R] >> >>";
    let px = rgb_at(governed_bar(ocprops, "5 0 R", usage_group()), &BARS);
    assert_eq!(px, [RED, GREEN, BLUE]);
}
