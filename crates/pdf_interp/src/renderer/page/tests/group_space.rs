//! Transparency groups composited in their own colour space: a CMYK group
//! over an RGB page composites in ink, an sRGB group over a wide page in
//! sRGB, an RGB group over an ink page in RGB, a gray group in gray; a
//! group whose space is its canvas's paints as a group without one.

use color::SampleDepth;
use pdf::Document;

use super::*;
use crate::renderer::FinishedCanvas;
use crate::separation::{CanvasRequest, Separation};

/// A stream object of `dict` entries holding `data`.
fn stream(dict: &str, data: &[u8]) -> Vec<u8> {
    let mut s = format!("<< {dict} /Length {} >>\nstream\n", data.len()).into_bytes();
    s.extend_from_slice(data);
    s.extend_from_slice(b"\nendstream");
    s
}

/// Page `(4, 0)`, 10 points square, whose form `/G` (5) is a transparency
/// group of `group` entries drawing `content`; `/Half` is a fill alpha of
/// one half, `/Mul` the `Multiply` and `/Diff` the `Difference` blend mode
/// and `/Spot` a `Separation`
/// colour space, in the page's and the form's resources.  Object 6 is an
/// RGB ICC profile that is not sRGB, object 7 a CMYK one (its data unused).
fn doc(group: &str, content: &str) -> Document {
    let res = "/ExtGState << /Half << /ca 0.5 >> /Mul << /BM /Multiply >> \
               /Diff << /BM /Difference >> >> /ColorSpace << /Spot [/Separation /Spot /DeviceCMYK \
               << /FunctionType 2 /Domain [0 1] /C0 [0 0 0 0] /C1 [0 0.5 1 0] /N 1 >>] >>";
    let adobe = moxcms::ColorProfile::new_adobe_rgb()
        .encode()
        .expect("encodes");
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
        (6, stream("/N 3", &adobe)),
        (7, stream("/N 4", b"")),
    ])
}

/// `content` rendered on page `(4, 0)` of `doc`, 10 pixels square, on the
/// canvas `request` asks for at `depth`, and its diagnostics.
fn render_page(
    doc: &Document,
    request: &CanvasRequest,
    depth: SampleDepth,
    content: &str,
) -> (FinishedCanvas, PageDiagnostics) {
    let view = PageRenderer::unit_view(10, 10);
    let mut renderer = PageRenderer::try_with_page_group_in(
        &view,
        doc,
        (4, 0),
        None,
        request,
        depth,
        super::super::CanvasBuffers::default(),
    )
    .expect("the canvas is allocated");
    renderer.execute(&crate::content::parse(content.as_bytes()));
    let (canvas, diag, _) = renderer.try_finish_keeping(false).expect("finishes");
    (canvas, diag)
}

/// The samples of a finished page, widened to 16 bits.
fn samples(canvas: &FinishedCanvas) -> Vec<u16> {
    match canvas {
        FinishedCanvas::Rgb8(b) => b.data().iter().map(|&v| u16::from(v)).collect(),
        FinishedCanvas::Rgb16(b) => b.data().to_vec(),
        FinishedCanvas::Cmyk8(b) => b.data().iter().map(|&v| u16::from(v)).collect(),
        FinishedCanvas::Cmyk16(b) => b.data().to_vec(),
        other => panic!("an RGB or CMYK page, not {other:?}"),
    }
}

/// `content` rendered on `request`'s canvas at `depth` in a group of
/// `group` entries drawn by the page, the page's samples.
fn grouped(group: &str, content: &str, request: &CanvasRequest, depth: SampleDepth) -> Vec<u16> {
    samples(&render_page(&doc(group, content), request, depth, "/G Do").0)
}

/// `content` rendered directly on `request`'s canvas at `depth`.
fn direct(content: &str, request: &CanvasRequest, depth: SampleDepth) -> Vec<u16> {
    samples(&render_page(&doc("", ""), request, depth, content).0)
}

/// Assert that `got` and `want` differ by at most one level per sample:
/// a group composites its objects on a transparent canvas, the page on
/// opaque paper, and the two round their mixtures alike to a level.
fn assert_close(got: &[u16], want: &[u16], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    let off = got.iter().zip(want).map(|(a, b)| a.abs_diff(*b)).max();
    assert!(off <= Some(1), "{what}: {got:?} vs {want:?}");
}

/// The ink canvas of the process-ink model, finished as sRGB, without
/// overprint.
const INK: CanvasRequest = CanvasRequest::ProcessRgb { overprint: false };
/// A CMYK page separated by the device conversion.
const CMYK: CanvasRequest = CanvasRequest::Cmyk {
    overprint: false,
    separation: Separation::Device,
};

/// Red under blue at half alpha.
const HALF: &str = "1 0 0 rg 0 0 10 10 re f /Half gs 0 0 1 rg 0 0 10 10 re f";
/// Red under blue in the `Multiply` blend mode.
const MULTIPLY: &str = "1 0 0 rg 0 0 10 10 re f /Mul gs 0 0 1 rg 0 0 10 10 re f";

#[test]
fn a_cmyk_group_on_an_rgb_page_composites_in_ink() {
    for depth in [SampleDepth::Eight, SampleDepth::Sixteen] {
        for content in [HALF, MULTIPLY] {
            let cmyk = grouped(
                "/CS /DeviceCMYK /I true",
                content,
                &CanvasRequest::Rgb,
                depth,
            );
            let ink = direct(content, &INK, depth);
            assert_close(&cmyk, &ink, &format!("{depth:?} {content}"));
            let rgb = grouped("/I true", content, &CanvasRequest::Rgb, depth);
            assert_ne!(cmyk, rgb, "{depth:?} {content}: blending in ink differs");
        }
    }
}

#[test]
fn a_non_isolated_cmyk_group_blends_with_its_backdrop_in_ink() {
    let content = "/Mul gs 0 0 1 rg 0 0 10 10 re f";
    let doc = doc("/CS /DeviceCMYK", content);
    let page = "1 0 0 rg 0 0 10 10 re f /G Do";
    let got = samples(&render_page(&doc, &CanvasRequest::Rgb, SampleDepth::Eight, page).0);
    let want = direct(
        "1 0 0 rg 0 0 10 10 re f /Mul gs 0 0 1 rg 0 0 10 10 re f",
        &INK,
        SampleDepth::Eight,
    );
    assert_close(&got, &want, "multiply over the backdrop");
}

/// The backdrop converted into a group's space reaches its blend modes
/// alone: where the group paints nothing the parent keeps its value.
#[test]
fn a_converted_group_leaves_unpainted_pixels_as_they_were() {
    for depth in [SampleDepth::Eight, SampleDepth::Sixteen] {
        let doc = doc("/CS /DeviceCMYK", "/Mul gs 0 0 1 rg 0 0 5 10 re f");
        let page = "0.8 0.3 0.1 rg 0 0 10 10 re f /G Do";
        let got = samples(&render_page(&doc, &CanvasRequest::Rgb, depth, page).0);
        let want = direct("0.8 0.3 0.1 rg 0 0 10 10 re f", &CanvasRequest::Rgb, depth);
        assert_eq!(&got[3 * 7..3 * 8], &want[3 * 7..3 * 8], "{depth:?}");
        assert_ne!(
            &got[0..3],
            &want[0..3],
            "{depth:?}: the painted half blends"
        );
    }
}

#[test]
fn a_spot_colour_keeps_its_channel_in_a_cmyk_group() {
    let content = "/Spot cs 0.7 scn 0 0 10 10 re f /Half gs 0 0 1 rg 0 0 10 10 re f";
    let cmyk = grouped(
        "/CS /DeviceCMYK /I true",
        content,
        &CanvasRequest::Rgb,
        SampleDepth::Eight,
    );
    assert_close(&cmyk, &direct(content, &INK, SampleDepth::Eight), "spot");
}

#[test]
fn a_group_in_its_canvas_space_paints_as_one_without_a_space() {
    let cases: [(&str, &CanvasRequest); 4] = [
        ("/CS /DeviceRGB", &CanvasRequest::Rgb),
        ("/CS [/ICCBased 6 0 R]", &CanvasRequest::Rgb),
        ("/CS [/ICCBased 6 0 R]", &CanvasRequest::WideRgb),
        ("/CS /DeviceCMYK", &CMYK),
    ];
    for (space, request) in cases {
        let depth = if *request == CanvasRequest::WideRgb {
            SampleDepth::Sixteen
        } else {
            SampleDepth::Eight
        };
        for content in [HALF, MULTIPLY] {
            assert_eq!(
                grouped(&format!("{space} /I true"), content, request, depth),
                grouped("/I true", content, request, depth),
                "{space} on {request:?}: {content}"
            );
        }
    }
}

#[test]
fn an_srgb_group_on_a_wide_page_composites_in_srgb() {
    let srgb = grouped(
        "/CS /DeviceRGB /I true",
        HALF,
        &CanvasRequest::Rgb,
        SampleDepth::Sixteen,
    );
    let wide = grouped(
        "/CS /DeviceRGB /I true",
        HALF,
        &CanvasRequest::WideRgb,
        SampleDepth::Sixteen,
    );
    for (s, w) in srgb.as_chunks::<3>().0.iter().zip(wide.as_chunks::<3>().0) {
        assert_eq!(*w, crate::working_space::from_srgb16(*s));
    }
    let in_wide = grouped(
        "/I true",
        HALF,
        &CanvasRequest::WideRgb,
        SampleDepth::Sixteen,
    );
    assert_ne!(
        wide, in_wide,
        "blending in sRGB differs from blending in the wide space"
    );
}

#[test]
fn an_rgb_group_on_a_cmyk_page_separates_its_result() {
    let k = "0 0 0 1 k 0 0 10 10 re f";
    let got = grouped("/CS /DeviceRGB /I true", k, &CMYK, SampleDepth::Eight);
    let black = color::cmyk::cmyk_to_rgb(0, 0, 0, 255);
    let want = color::cmyk::rgb_to_device_cmyk(black);
    assert_eq!(&got[..4], &want.map(u16::from));
    assert_eq!(&direct(k, &CMYK, SampleDepth::Eight)[..4], &[0, 0, 0, 255]);
}

#[test]
fn a_gray_group_composites_the_gray_of_its_colours() {
    let red = "1 0 0 rg 0 0 10 10 re f";
    for space in [
        "/DeviceGray",
        "[/CalGray << /WhitePoint [0.9505 1 1.089] >>]",
    ] {
        let rgb = grouped(
            &format!("/CS {space} /I true"),
            red,
            &CanvasRequest::Rgb,
            SampleDepth::Eight,
        );
        assert_eq!(&rgb[..3], &[76, 76, 76], "{space}");
    }
    let cmyk = grouped("/CS /DeviceGray /I true", red, &CMYK, SampleDepth::Eight);
    assert_eq!(&cmyk[..4], &[0, 0, 0, 255 - 76], "black ink alone");
}

/// Blue over red in a blend mode, in a gray group: the blend function
/// acts on their grays, 76 and 28 by the `DeviceRGB` → `DeviceGray`
/// weights.  Blending their components would give black for `Multiply`
/// and magenta, gray 104, for `Difference`.
const GRAY_BLENDS: [(&str, u16); 2] = [("/Mul", 8), ("/Diff", 48)];

/// Red under blue in the blend mode of the `ExtGState` `mode`.
fn blended(mode: &str) -> String {
    format!("1 0 0 rg 0 0 10 10 re f {mode} gs 0 0 1 rg 0 0 10 10 re f")
}

#[test]
fn a_gray_group_blends_the_gray_of_its_colours() {
    for (mode, want) in GRAY_BLENDS {
        let content = blended(mode);
        for depth in [SampleDepth::Eight, SampleDepth::Sixteen] {
            let rgb = grouped(
                "/CS /DeviceGray /I true",
                &content,
                &CanvasRequest::Rgb,
                depth,
            );
            let want = if depth == SampleDepth::Eight {
                want
            } else {
                want * 257
            };
            assert!(
                rgb[..3].iter().all(|v| v.abs_diff(want) <= 257 / 2),
                "{mode} {depth:?}: {rgb:?}"
            );
        }
        let cmyk = grouped(
            "/CS /DeviceGray /I true",
            &content,
            &CMYK,
            SampleDepth::Eight,
        );
        assert_eq!(
            &cmyk[..4],
            &[0, 0, 0, 255 - want],
            "{mode}: black ink alone"
        );
    }
}

/// A non-isolated gray group blends with the gray of its backdrop.
#[test]
fn a_non_isolated_gray_group_blends_with_the_gray_of_its_backdrop() {
    for (mode, want) in GRAY_BLENDS {
        let doc = doc(
            "/CS /DeviceGray",
            &format!("{mode} gs 0 0 1 rg 0 0 10 10 re f"),
        );
        let page = "1 0 0 rg 0 0 10 10 re f /G Do";
        let got = samples(&render_page(&doc, &CanvasRequest::Rgb, SampleDepth::Eight, page).0);
        assert_eq!(&got[..3], &[want; 3], "{mode}");
    }
}

/// Page `(4, 0)` of page group `page_group` drawing form 5, a group of
/// `group` entries that draws form 6, a group without a space drawing
/// `content`.
fn nested_doc(page_group: &str, group: &str, content: &str) -> Document {
    let res = "/ExtGState << /Mul << /BM /Multiply >> /Diff << /BM /Difference >> >>";
    let form = |group: &str, xobjects: &str, content: &str| {
        stream(
            &format!(
                "/Type /XObject /Subtype /Form /BBox [0 0 10 10] \
                 /Group << /S /Transparency {group} >> /Resources << {res} {xobjects} >>"
            ),
            content.as_bytes(),
        )
    };
    crate::test_helpers::make_doc_with_objects(&[
        (
            4,
            format!(
                "<< /Type /Page {page_group} /Resources << {res} /XObject << /G 5 0 R >> >> >>"
            )
            .into_bytes(),
        ),
        (5, form(group, "/XObject << /H 6 0 R >>", "/H Do")),
        (6, form("/I true", "", content)),
    ])
}

/// A group without a space inherits a gray group's, and blends in gray.
#[test]
fn a_group_without_a_space_inherits_gray_blending() {
    for (mode, want) in GRAY_BLENDS {
        let doc = nested_doc("", "/CS /DeviceGray /I true", &blended(mode));
        let got = samples(&render_page(&doc, &CanvasRequest::Rgb, SampleDepth::Eight, "/G Do").0);
        assert_eq!(&got[..3], &[want; 3], "{mode}");
    }
}

/// A page whose group is gray blends in gray, as a gray form group does;
/// a page painted in ink keeps blending in ink.
#[test]
fn a_gray_page_group_blends_in_gray() {
    for (mode, want) in GRAY_BLENDS {
        let doc = nested_doc("/Group << /S /Transparency /CS /DeviceGray >>", "", "");
        let paint = |request: &CanvasRequest| {
            let view = PageRenderer::unit_view(10, 10);
            let mut renderer = PageRenderer::try_with_page_group_in(
                &view,
                &doc,
                (4, 0),
                crate::page_group(&doc, (4, 0)),
                request,
                SampleDepth::Eight,
                super::super::CanvasBuffers::default(),
            )
            .expect("the canvas is allocated");
            renderer.execute(&crate::content::parse(blended(mode).as_bytes()));
            samples(&renderer.try_finish_keeping(false).expect("finishes").0)
        };
        assert_eq!(&paint(&CanvasRequest::Rgb)[..3], &[want; 3], "{mode}");
        assert_eq!(
            paint(&CMYK),
            direct(&blended(mode), &CMYK, SampleDepth::Eight),
            "{mode} on an ink page"
        );
    }
}

#[test]
fn an_approximated_group_space_is_reported() {
    let doc = doc("/CS [/ICCBased 7 0 R] /I true", HALF);
    let (_, diag) = render_page(&doc, &CanvasRequest::Rgb, SampleDepth::Eight, "/G Do");
    assert!(diag.approximated_group_spaces.cmyk_profile);
    assert!(!diag.approximated_group_spaces.rgb_profile);
    let doc = doc_rgb_profiled();
    let (_, diag) = render_page(&doc, &CanvasRequest::Rgb, SampleDepth::Eight, "/G Do");
    assert!(diag.approximated_group_spaces.rgb_profile);
    let doc = crate::test_helpers::make_doc_with_objects(&[(4, b"<< /Type /Page >>".to_vec())]);
    let (_, diag) = render_page(&doc, &CanvasRequest::Rgb, SampleDepth::Eight, HALF);
    assert!(!diag.approximated_group_spaces.any());
}

/// [`doc`] of a group in the non-sRGB RGB profile, drawing [`HALF`].
fn doc_rgb_profiled() -> Document {
    doc("/CS [/ICCBased 6 0 R] /I true", HALF)
}

/// A page drawing form 5, a transparency group of `group` entries whose
/// resources make the non-sRGB profile (6) the default RGB space.
fn default_rgb_doc(group: &str) -> Document {
    let adobe = moxcms::ColorProfile::new_adobe_rgb()
        .encode()
        .expect("encodes");
    crate::test_helpers::make_doc_with_objects(&[
        (
            4,
            b"<< /Type /Page /Resources << /XObject << /G 5 0 R >> >> >>".to_vec(),
        ),
        (
            5,
            stream(
                &format!(
                    "/Type /XObject /Subtype /Form /BBox [0 0 10 10] \
                     /Group << /S /Transparency {group} /I true >> \
                     /Resources << /ExtGState << /Half << /ca 0.5 >> >> \
                     /ColorSpace << /DefaultRGB [/ICCBased 6 0 R] >> >>"
                ),
                HALF.as_bytes(),
            ),
        ),
        (6, stream("/N 3", &adobe)),
    ])
}

/// A `DefaultRGB` of the non-sRGB profile makes a `DeviceRGB` group's
/// space that profile, which on a wide page is approximated by the page's
/// own space: the group paints as one without a space.
#[test]
fn a_default_rgb_space_replaces_a_device_group_space() {
    let render = |group| {
        render_page(
            &default_rgb_doc(group),
            &CanvasRequest::WideRgb,
            SampleDepth::Sixteen,
            "/G Do",
        )
    };
    let (canvas, diag) = render("/CS /DeviceRGB");
    assert!(diag.approximated_group_spaces.rgb_profile);
    assert_eq!(samples(&canvas), samples(&render("").0));
}

/// Page `(4, 0)` whose `/M` is a luminosity soft mask of backdrop entries
/// `bc` whose group (5), of `/CS space`, draws `content` over the page;
/// `/C` names the group's space in the group's resources.
fn mask_doc(space: &str, bc: &str, content: &str) -> Document {
    crate::test_helpers::make_doc_with_objects(&[
        (
            4,
            format!(
                "<< /Type /Page /Resources << /ExtGState << \
                 /M << /SMask << /S /Luminosity /G 5 0 R {bc} >> >> >> >> >>"
            )
            .into_bytes(),
        ),
        (
            5,
            stream(
                &format!(
                    "/Type /XObject /Subtype /Form /BBox [0 0 10 10] \
                     /Group << /S /Transparency /CS {space} >> \
                     /Resources << /ColorSpace << /C {space} >> >>"
                ),
                content.as_bytes(),
            ),
        ),
    ])
}

/// Black painted through the luminosity mask of `mask_doc(space, bc,
/// content)`: the first pixel of the 8-bit sRGB page, `255 − mask`.
fn masked(space: &str, bc: &str, content: &str) -> u16 {
    let doc = mask_doc(space, bc, content);
    let page = "/M gs 0 g 0 0 10 10 re f";
    samples(&render_page(&doc, &CanvasRequest::Rgb, SampleDepth::Eight, page).0)[0]
}

/// A CMYK mask group takes the conventional gray of its ink: black ink
/// alone is a mask of zero.
#[test]
fn a_cmyk_luminosity_mask_takes_the_device_gray_of_its_ink() {
    assert_eq!(masked("/DeviceCMYK", "", "0 0 0 1 k 0 0 10 10 re f"), 255);
    assert_eq!(masked("/DeviceCMYK", "", "0 0 0 0 k 0 0 10 10 re f"), 0);
    // 1 − (0.59 + 0.11) of solid magenta and yellow.
    assert_eq!(
        masked("/DeviceCMYK", "", "0 1 1 0 k 0 0 10 10 re f"),
        255 - 76
    );
}

/// A colour that is not ink takes its own gray in a CMYK mask, as in an
/// RGB one: the device separation's ink has the colour's gray.
#[test]
fn a_cmyk_luminosity_mask_takes_the_gray_of_an_rgb_colour() {
    for colour in ["0.5 g", "0.2 0.6 0.8 rg", "1 0 0 rg"] {
        let content = format!("{colour} 0 0 10 10 re f");
        let cmyk = masked("/DeviceCMYK", "", &content);
        let rgb = masked("/DeviceRGB", "", &content);
        assert!(cmyk.abs_diff(rgb) <= 1, "{colour}: {cmyk} vs {rgb}");
    }
}

/// `BC` of a CMYK mask group is ink: no ink is a full mask, and the
/// default is black ink, a mask of zero.
#[test]
fn a_cmyk_mask_backdrop_is_ink() {
    assert_eq!(masked("/DeviceCMYK", "/BC [0 0 0 0]", ""), 0);
    assert_eq!(masked("/DeviceCMYK", "", ""), 255);
    assert_eq!(masked("/DeviceCMYK", "/BC [0 0 0 0.5]", ""), 128);
}

/// A mask group of a `CalRGB` or `CalGray` space takes the luminance `Y`
/// of its colour (ISO 32000-1 §11.5.3): a gray of 0.5 under a gamma of
/// 2.2 is a mask of 0.5^2.2 = 0.218, where the luma of its encoded value
/// is about 0.5.  A device space takes the luma of its components.
#[test]
fn a_cal_luminosity_mask_takes_the_luminance_of_its_colour() {
    let srgb = "/Matrix [0.4124 0.2126 0.0193 0.3576 0.7152 0.1192 0.1805 0.0722 0.9505]";
    for (space, colour) in [
        (
            "[/CalGray << /WhitePoint [0.9505 1 1.089] /Gamma 2.2 >>]",
            "0.5",
        ),
        (
            &*format!("[/CalRGB << /WhitePoint [0.9505 1 1.089] /Gamma [2.2 2.2 2.2] {srgb} >>]"),
            "0.5 0.5 0.5",
        ),
    ] {
        let got = masked(space, "", &format!("/C cs {colour} sc 0 0 10 10 re f"));
        assert!(got.abs_diff(200) <= 1, "{space}: {got}");
    }
    let device = masked("/DeviceRGB", "", "/C cs 0.5 0.5 0.5 sc 0 0 10 10 re f");
    assert!(device.abs_diff(128) <= 1, "{device}");
}
