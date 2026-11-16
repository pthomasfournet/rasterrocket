//! Transparency groups of an ICC CMYK space: painted in the profile's ink
//! and converted back through the profile.

use super::*;
use crate::resources::colorspace::icc::IccConversion;
use crate::{ColourIntent, ColourTarget, OutputProfile, RenderingIntent, WorkingSpace};

/// The synthetic CMYK printer of `tests/fixtures/icc` whose conversions
/// differ visibly from the process-ink model.
fn printer() -> Vec<u8> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/icc/printer-cmyk-v2.icc"
    );
    std::fs::read(path).expect("the fixture is present")
}

/// Page `(4, 0)`, 10 points square, whose form `/G` (5) is a transparency
/// group of `group` entries drawing `content`; `/P` is the ICC CMYK space
/// of the printer (6), `/Half` a fill alpha of one half and `/Spot` a
/// `Separation` colour space of a `DeviceCMYK` alternate.  The page's
/// `/Group` is `page_group`.
fn profile_doc(page_group: &str, group: &str, content: &str) -> Document {
    let res = "/ExtGState << /Half << /ca 0.5 >> >> /ColorSpace << /P [/ICCBased 6 0 R] \
               /Spot [/Separation /Spot /DeviceCMYK << /FunctionType 2 /Domain [0 1] \
               /C0 [0 0 0 0] /C1 [0 0.5 1 0] /N 1 >>] >>";
    crate::test_helpers::make_doc_with_objects(&[
        (
            4,
            format!(
                "<< /Type /Page {page_group} /Resources << {res} /XObject << /G 5 0 R >> >> >>"
            )
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
        (6, stream("/N 4", &printer())),
    ])
}

/// The group attributes of the printer's space.
const PRINTER: &str = "/CS [/ICCBased 6 0 R] /I true";

/// The first pixel of `content` drawn in a group of `group` entries on an
/// RGB page at `depth`.
fn in_group(group: &str, content: &str, depth: SampleDepth) -> Vec<u16> {
    let doc = profile_doc("", group, content);
    samples(&render_page(&doc, &CanvasRequest::Rgb, depth, "/G Do").0)[..3].to_vec()
}

/// The first pixel of `content` drawn on the page itself, of page group
/// `page_group`, on the canvas `request` asks for at `depth`.
fn on_page(
    page_group: &str,
    request: &CanvasRequest,
    content: &str,
    depth: SampleDepth,
) -> Vec<u16> {
    let doc = profile_doc(page_group, "", "");
    samples(&render_page(&doc, request, depth, content).0)[..3].to_vec()
}

/// The 8-bit sRGB of the printer's ink `cmyk`, as a colour of its space
/// converts on an RGB page.
fn printed(cmyk: [f64; 4]) -> [u8; 3] {
    let doc = profile_doc("", "", "");
    let target = ColourTarget::new(ColourIntent::INITIAL, WorkingSpace::Srgb);
    IccConversion::compile(&printer(), 4, target)
        .and_then(|c| c.convert::<u8>(&doc, &cmyk))
        .expect("the printer converts")
}

/// The printer's ink of the sRGB colour `rgb`, in `[0, 1]`.
fn separated(rgb: [u8; 3]) -> [f64; 4] {
    let profile = OutputProfile::from_bytes(printer()).expect("a CMYK output profile");
    let table = profile
        .separation_table(
            WorkingSpace::Srgb,
            RenderingIntent::RelativeColorimetric,
            true,
        )
        .expect("the printer separates");
    table.ink(rgb).map(|v: u8| f64::from(v) / 255.0)
}

/// Whether the samples `got` lie within `tolerance` of `want`.
fn near(got: &[u16], want: &[u8], tolerance: u16) -> bool {
    got.iter()
        .zip(want)
        .all(|(&g, &w)| g.abs_diff(u16::from(w)) <= tolerance)
}

/// Red under blue at half alpha in the printer's space is the printer's
/// colour of the mixture of their inks, unlike the process-ink model's.
#[test]
fn red_under_blue_mixes_the_printers_inks() {
    let (red, blue) = (separated([255, 0, 0]), separated([0, 0, 255]));
    let mixed: [f64; 4] = std::array::from_fn(|i| f64::midpoint(red[i], blue[i]));
    let want = printed(mixed);
    let got = in_group(PRINTER, HALF, SampleDepth::Eight);
    assert!(near(&got, &want, 2), "{got:?} for {want:?}");
    let model = in_group("/CS /DeviceCMYK /I true", HALF, SampleDepth::Eight);
    assert!(
        got.iter().zip(&model).any(|(a, b)| a.abs_diff(*b) > 8),
        "{got:?} against the model's {model:?}"
    );
    let wide = in_group(PRINTER, HALF, SampleDepth::Sixteen);
    let wide: Vec<u16> = wide.iter().map(|v| (v + 128) / 257).collect();
    assert!(near(&wide, &want, 2), "16 bits: {wide:?} for {want:?}");
}

/// A colour of the group's own space paints its components and shows the
/// colour it shows outside the group.
#[test]
fn a_colour_of_the_groups_space_keeps_its_value() {
    let content = "/P cs 0.2 0.5 0.1 0.1 sc 0 0 10 10 re f";
    for depth in [SampleDepth::Eight, SampleDepth::Sixteen] {
        let got = in_group(PRINTER, content, depth);
        let want = on_page("", &CanvasRequest::Rgb, content, depth);
        assert_close(&got, &want, &format!("{depth:?}"));
    }
}

/// A `DeviceCMYK` colour is the process-ink model's colour in the group as
/// on the page, within the round trip through the printer.
#[test]
fn a_device_cmyk_colour_keeps_its_colour() {
    let content = "0.2 0.5 0.1 0.1 k 0 0 10 10 re f";
    let got = in_group(PRINTER, content, SampleDepth::Eight);
    let want = on_page("", &CanvasRequest::Rgb, content, SampleDepth::Eight);
    let off = got.iter().zip(&want).map(|(a, b)| a.abs_diff(*b)).max();
    assert!(off <= Some(3), "{got:?} for {want:?}");
}

/// A spot colour in the group slots in the group's own ink map, its tints
/// the printer's ink of its alternate's colour: it shows the colour its
/// alternate colour shows in the group, within the quantisation of the
/// spot's table, not the colour of the alternate's components read as the
/// printer's ink.
#[test]
fn a_spot_colour_in_the_group_shows_its_alternates_colour() {
    let spot = in_group(
        PRINTER,
        "/Spot cs 0.7 scn 0 0 10 10 re f",
        SampleDepth::Eight,
    );
    let alternate = "0 0.35 0.7 0 k 0 0 10 10 re f";
    let want = in_group(PRINTER, alternate, SampleDepth::Eight);
    let off = spot.iter().zip(&want).map(|(a, b)| a.abs_diff(*b)).max();
    assert!(off <= Some(4), "{spot:?} for the alternate's {want:?}");
    let read_as_ink = printed([0.0, 0.35, 0.7, 0.0]);
    assert!(
        !near(&spot, &read_as_ink, 8),
        "{spot:?} against {read_as_ink:?}"
    );
}

/// A page whose group is the printer's space composites as a form group of
/// that space does, and finishes through the printer.
#[test]
fn a_profile_page_group_composites_in_the_printers_ink() {
    let request = CanvasRequest::ProfileRgb {
        profile: (6, 0),
        space: WorkingSpace::Srgb,
    };
    for depth in [SampleDepth::Eight, SampleDepth::Sixteen] {
        let page = on_page(
            "/Group << /S /Transparency /CS [/ICCBased 6 0 R] >>",
            &request,
            HALF,
            depth,
        );
        assert_close(
            &page,
            &in_group(PRINTER, HALF, depth),
            &format!("{depth:?}"),
        );
    }
}

/// A page of a profile that does not separate paints in the process-ink
/// model's ink and reports the approximation.
#[test]
fn a_profile_page_group_that_cannot_separate_is_approximated() {
    let doc = profile_doc("", "", "");
    let request = CanvasRequest::ProfileRgb {
        profile: (5, 0),
        space: WorkingSpace::Srgb,
    };
    let (canvas, diag) = render_page(&doc, &request, SampleDepth::Eight, HALF);
    assert!(diag.approximated_group_spaces.cmyk_profile);
    let model = samples(&render_page(&doc, &INK, SampleDepth::Eight, HALF).0);
    assert_eq!(samples(&canvas), model);
}
