//! Spot colours in additive transparency groups on pages of ink: the
//! group's RGB beside the page's spot channels, so a spot keeps its
//! channel and its overprint through the group.

use super::*;

/// Page `(4, 0)`, 10 points square, whose form `/G` (5) is a transparency
/// group of `group` entries drawing `content`; `/Spot` is a `Separation`
/// colour space whose solid tint is `0 0.5 1 0`, `/Half` a fill alpha of
/// one half and `/Op` nonzero overprint, in the page's and the form's
/// resources.
fn spot_doc(group: &str, content: &str) -> Document {
    let res = "/ExtGState << /Half << /ca 0.5 >> /Op << /op true /OPM 1 >> >> \
               /ColorSpace << /Spot [/Separation /Spot /DeviceCMYK << /FunctionType 2 \
               /Domain [0 1] /C0 [0 0 0 0] /C1 [0 0.5 1 0] /N 1 >>] >>";
    crate::test_helpers::make_doc_with_objects(&[
        (
            4,
            format!("<< /Type /Page /Resources << {res} /XObject << /G 5 0 R >> >> >>")
                .into_bytes(),
        ),
        (
            5,
            stream(
                &format!(
                    "/Type /XObject /Subtype /Form /BBox [0 0 10 10] \
                     /Group << /S /Transparency {group} >> /Resources << {res} >>"
                ),
                content.as_bytes(),
            ),
        ),
    ])
}

/// `content` in an isolated `DeviceRGB` group and on the page itself, on
/// `request`'s canvas at 8 bits.
fn grouped_and_direct(content: &str, request: &CanvasRequest) -> (Vec<u16>, Vec<u16>) {
    let group = spot_doc("/CS /DeviceRGB /I true", content);
    let page = spot_doc("", "");
    (
        samples(&render_page(&group, request, SampleDepth::Eight, "/G Do").0),
        samples(&render_page(&page, request, SampleDepth::Eight, content).0),
    )
}

/// Cyan in RGB, whose process ink the device conversion and the
/// process-ink model's inverse take without black, then the spot.
const CYAN_THEN: &str = "0 1 1 rg 0 0 10 10 re f";

/// A CMYK page simulating overprint, separated by the device conversion.
const CMYK_OVERPRINT: CanvasRequest = CanvasRequest::Cmyk {
    overprint: true,
    separation: Separation::Device,
};

/// A spot at half alpha over cyan in an RGB group of a CMYK page: the
/// spot's channel holds its half tint and the cyan mixes with white, as
/// on the page itself.
#[test]
fn a_spot_at_half_alpha_keeps_its_channel_through_an_rgb_group() {
    let content = format!("{CYAN_THEN} /Half gs /Spot cs 1 scn 0 0 10 10 re f");
    let (group, page) = grouped_and_direct(&content, &CMYK);
    assert_close(&group, &page, "half alpha");
    assert_eq!(
        &page[..4],
        &[127, 64, 128, 0],
        "half cyan beside half the spot"
    );
}

/// A spot under overprint in an RGB group leaves the cyan under it, on a
/// CMYK page and in the overprint preview.
#[test]
fn a_spot_overprints_in_an_rgb_group() {
    let content = format!("{CYAN_THEN} /Op gs /Spot cs 1 scn 0 0 10 10 re f");
    let preview = CanvasRequest::ProcessRgb { overprint: true };
    for request in [&CMYK_OVERPRINT, &preview] {
        let (group, page) = grouped_and_direct(&content, request);
        assert_close(&group, &page, &format!("{request:?}"));
    }
    let (group, _) = grouped_and_direct(&content, &CMYK_OVERPRINT);
    assert_eq!(&group[..4], &[255, 128, 255, 0], "cyan and the spot");
}

/// A spot in a gray group nested in an RGB group of a CMYK page is
/// reduced with its spot ink folded in, not as white.
#[test]
fn a_gray_group_in_an_rgb_group_takes_the_gray_of_a_spot() {
    let doc = crate::test_helpers::make_doc_with_objects(&[
        (
            4,
            b"<< /Type /Page /Resources << /XObject << /G 5 0 R >> >> >>".to_vec(),
        ),
        (
            5,
            stream(
                "/Type /XObject /Subtype /Form /BBox [0 0 10 10] \
                 /Group << /S /Transparency /CS /DeviceRGB /I true >> \
                 /Resources << /XObject << /H 6 0 R >> >>",
                b"/H Do",
            ),
        ),
        (
            6,
            stream(
                "/Type /XObject /Subtype /Form /BBox [0 0 10 10] \
                 /Group << /S /Transparency /CS /DeviceGray /I true >> \
                 /Resources << /ColorSpace << /Spot [/Separation /Spot /DeviceCMYK \
                 << /FunctionType 2 /Domain [0 1] /C0 [0 0 0 0] /C1 [0 0 0 1] /N 1 >>] >> >>",
                b"/Spot cs 1 scn 0 0 10 10 re f",
            ),
        ),
    ]);
    let got = samples(&render_page(&doc, &CMYK, SampleDepth::Eight, "/G Do").0);
    assert!(
        got[3] > 200,
        "the black spot reduces to a dark gray: {got:?}"
    );
}
