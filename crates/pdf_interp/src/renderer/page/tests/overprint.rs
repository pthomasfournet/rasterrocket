//! Overprint: which channels a painting operation leaves to the backdrop.

use pdf::Document;
use raster::pipe::ChannelMask;

use super::*;

/// Page `(4, 0)` whose `/ExtGState` resources are the entries `states` and
/// whose `/ColorSpace` resources are the entries `spaces`.
fn overprint_doc(states: &str, spaces: &str) -> Document {
    crate::test_helpers::make_doc_with_objects(&[(
        4,
        format!(
            "<< /Type /Page /Resources << /ExtGState << {states} >> \
             /ColorSpace << {spaces} >> >> >>"
        ),
    )])
}

/// `/Spot`, a spot colorant whose solid tint prints as 91 % magenta and
/// 76 % yellow.
fn spot_space() -> String {
    format!(
        "/Spot [/Separation /Spot /DeviceCMYK {}]",
        tint_to_cmyk("0 0.91 0.76 0")
    )
}

/// The RGB of pixel `(x, y)` of the 10-pixel-square page `rgb`.
fn rgb_at(rgb: &[u8], x: usize, y: usize) -> [u8; 3] {
    let i = (y * 10 + x) * 3;
    [rgb[i], rgb[i + 1], rgb[i + 2]]
}

/// The ink of pixel `(x, y)` of the 10-pixel-square ink page `ink`.
fn ink_at(ink: &[u8], x: usize, y: usize) -> [u8; 8] {
    let i = (y * 10 + x) * 8;
    ink[i..i + 8].try_into().unwrap()
}

/// A solid cyan page over which the left half is painted again.
const CYAN_PAGE: &str = "1 0 0 0 k 0 0 10 10 re f";

/// A black `DeviceCMYK` fill under nonzero overprint mode leaves the
/// cyan it does not paint: the fill has no cyan component.
#[test]
fn k_text_overprints_cyan() {
    let doc = overprint_doc("/GS_op << /OP true /OPM 1 >>", "");
    let content = format!("{CYAN_PAGE} /GS_op gs 0 0 0 1 k 0 0 5 10 re f");
    let rgb = render_canvas(&doc, 10, content.as_bytes(), true);
    assert_eq!(rgb_at(&rgb, 2, 5), color::cmyk::cmyk_to_rgb(255, 0, 0, 255));
    assert_eq!(rgb_at(&rgb, 7, 5), color::cmyk::cmyk_to_rgb(255, 0, 0, 0));
}

/// Under overprint mode 0 every process component is painted, a zero
/// one included: the black fill knocks the cyan out.
#[test]
fn opm0_knocks_out_zero_components() {
    let doc = overprint_doc("/GS_op << /OP true /OPM 0 >>", "");
    let content = format!("{CYAN_PAGE} /GS_op gs 0 0 0 1 k 0 0 5 10 re f");
    let rgb = render_canvas(&doc, 10, content.as_bytes(), true);
    assert_eq!(rgb_at(&rgb, 2, 5), color::cmyk::cmyk_to_rgb(0, 0, 0, 255));
    assert_eq!(rgb_at(&rgb, 7, 5), color::cmyk::cmyk_to_rgb(255, 0, 0, 0));
}

/// With overprint off the mode is irrelevant: the fill knocks out.
#[test]
fn op_false_knocks_out() {
    let doc = overprint_doc("/GS_op << /OP false /OPM 1 >>", "");
    let content = format!("{CYAN_PAGE} /GS_op gs 0 0 0 1 k 0 0 5 10 re f");
    let rgb = render_canvas(&doc, 10, content.as_bytes(), true);
    assert_eq!(rgb_at(&rgb, 2, 5), color::cmyk::cmyk_to_rgb(0, 0, 0, 255));
}

/// A spot colour paints its own channel only: the cyan under it stays,
/// and the page shows cyan with the spot's process equivalent on top.
#[test]
fn spot_overprints_process() {
    let doc = overprint_doc("/GS << /op true >>", &spot_space());
    let content = format!("{CYAN_PAGE} /GS gs /Spot cs 1 scn 0 0 5 10 re f");
    let ink = render_ink(&doc, content.as_bytes());
    assert_eq!(ink_at(&ink, 2, 5), [255, 0, 0, 0, 255, 0, 0, 0]);
    assert_eq!(ink_at(&ink, 7, 5), [255, 0, 0, 0, 0, 0, 0, 0]);
    // The spot's process equivalent at full tint is 0 0.91 0.76 0.
    let rgb = render_canvas(&doc, 10, content.as_bytes(), true);
    let want = color::cmyk::cmyk_to_rgb(255, 232, 194, 0);
    let got = rgb_at(&rgb, 2, 5);
    assert!(
        got.iter().zip(&want).all(|(g, w)| g.abs_diff(*w) <= 1),
        "got {got:?}, want {want:?}"
    );
    assert_eq!(rgb_at(&rgb, 7, 5), color::cmyk::cmyk_to_rgb(255, 0, 0, 0));
    // Without overprint the spot knocks the cyan out.
    let doc = overprint_doc("/GS << /op false >>", &spot_space());
    let ink = render_ink(&doc, content.as_bytes());
    assert_eq!(ink_at(&ink, 2, 5), [0, 0, 0, 0, 255, 0, 0, 0]);
}

/// An RGB colour paints every channel: with overprint on it still
/// knocks the cyan out.
#[test]
fn rgb_fill_with_op_knocks_out() {
    let doc = overprint_doc("/GS << /OP true /OPM 1 >>", "");
    let content = format!("{CYAN_PAGE} /GS gs 1 0 0 rg 0 0 5 10 re f");
    let (fill, ink) = ink_fill(&doc, content.as_bytes());
    assert_eq!(fill.paints(), ChannelMask::ALL);
    let [c, m, y, k] = color::cmyk::rgb_to_ink([255, 0, 0]);
    assert_eq!(c, 0, "red prints without cyan");
    assert_eq!(ink_at(&ink, 2, 5), [c, m, y, k, 0, 0, 0, 0]);
    assert_eq!(ink_at(&ink, 7, 5), [255, 0, 0, 0, 0, 0, 0, 0]);
}

/// `OP` governs strokes and `op` fills: the black stroke leaves the
/// cyan, the black fill knocks it out.
#[test]
fn stroke_uses_op_stroke() {
    let doc = overprint_doc("/GS << /OP true /op false /OPM 1 >>", "");
    let content = format!("{CYAN_PAGE} /GS gs 0 0 0 1 K 0 0 0 1 k 4 w 0 5 m 4 5 l S 6 0 4 10 re f");
    let ink = render_ink(&doc, content.as_bytes());
    assert_eq!(ink_at(&ink, 2, 5), [255, 0, 0, 255, 0, 0, 0, 0], "stroke");
    assert_eq!(ink_at(&ink, 8, 5), [0, 0, 0, 255, 0, 0, 0, 0], "fill");
    assert_eq!(ink_at(&ink, 5, 0), [255, 0, 0, 0, 0, 0, 0, 0], "untouched");
}

/// The overprint option leaves a page with no overprint on the RGB
/// canvas, rendered byte for byte as without the option.
#[test]
fn overprint_option_without_op_is_identical() {
    let doc = overprint_doc("/GS << /ca 1 >>", &spot_space());
    let content = b"0.2 0.5 0.1 0.3 k 0 0 10 10 re f /GS gs 1 0 0 rg 2 2 4 4 re f \
                    /Spot cs 0.7 scn 6 6 3 3 re f";
    let render = |overprint: bool| {
        let mut renderer = PageRenderer::new(10, 10, &doc, (4, 0));
        renderer.set_overprint(overprint);
        renderer.execute(&crate::content::parse(content));
        // An RGB canvas, whatever the option.
        assert!(matches!(renderer.canvas, PageCanvas::Rgb(_)));
        renderer.finish().0.data().to_vec()
    };
    let (on, off) = (render(true), render(false));
    assert!(on.iter().any(|&v| v != u8::MAX), "the content paints");
    assert_eq!(on, off);
}

/// Page `(4, 0)` of a solid cyan fill followed by the form `/F`, a
/// transparency group isolated as `isolated` says, with its own
/// `/GS_op` of `OP true OPM 1`, whose content is `content`.
fn group_doc(isolated: bool, content: &str) -> Document {
    crate::test_helpers::make_doc_with_objects(&[
        (
            4,
            "<< /Type /Page /Resources << /XObject << /F 5 0 R >> >> >>".to_owned(),
        ),
        (
            5,
            stream_object(
                &format!(
                    "/Type /XObject /Subtype /Form /BBox [0 0 10 10] \
                     /Group << /S /Transparency /I {isolated} >> \
                     /Resources << /ExtGState << /GS_op << /OP true /OPM 1 >> >> >>"
                ),
                content,
            ),
        ),
    ])
}

/// Inside a non-isolated group an overprinting fill leaves the cyan of
/// the page under the group; inside an isolated group it paints over a
/// transparent backdrop, so the group knocks the cyan out when it is
/// composited.
#[test]
fn overprint_in_a_non_isolated_group_keeps_the_backdrop() {
    let content = format!("{CYAN_PAGE} /F Do");
    let glyph = "/GS_op gs 0 0 0 1 k 0 0 5 10 re f";
    let rgb = render_canvas(&group_doc(false, glyph), 10, content.as_bytes(), true);
    assert_eq!(rgb_at(&rgb, 2, 5), color::cmyk::cmyk_to_rgb(255, 0, 0, 255));
    assert_eq!(rgb_at(&rgb, 7, 5), color::cmyk::cmyk_to_rgb(255, 0, 0, 0));
    let rgb = render_canvas(&group_doc(true, glyph), 10, content.as_bytes(), true);
    assert_eq!(rgb_at(&rgb, 2, 5), color::cmyk::cmyk_to_rgb(0, 0, 0, 255));
    assert_eq!(rgb_at(&rgb, 7, 5), color::cmyk::cmyk_to_rgb(255, 0, 0, 0));
}

/// A stencil mask paints the fill colour, under the fill's overprint
/// and mode: a black stencil under mode 1 leaves the cyan, under mode 0
/// knocks it out.
#[test]
fn stencil_mask_overprints_with_opm() {
    let stencil = "/GS_op gs 0 0 0 1 k q 5 0 0 10 0 0 cm BI /W 1 /H 1 /IM true ID \x00 EI Q";
    for (opm, left) in [(1, [255, 0, 0, 255]), (0, [0, 0, 0, 255])] {
        let doc = overprint_doc(&format!("/GS_op << /OP true /OPM {opm} >>"), "");
        let ink = render_ink(&doc, format!("{CYAN_PAGE} {stencil}").as_bytes());
        let [c, m, y, k] = left;
        assert_eq!(ink_at(&ink, 2, 5), [c, m, y, k, 0, 0, 0, 0], "OPM {opm}");
        assert_eq!(ink_at(&ink, 7, 5), [255, 0, 0, 0, 0, 0, 0, 0], "OPM {opm}");
    }
}

/// Glyphs overprint as fills do: black text under mode 1 leaves the cyan
/// everywhere it is painted, and knocks it out without overprint.
#[test]
fn text_overprints_like_a_fill() {
    let text = "0 0 0 1 k BT /F1 14 Tf 0 1 Td (H) Tj ET";
    let render = |states: &str| {
        let doc = crate::test_helpers::make_doc_with_objects(&[
            (
                4,
                format!(
                    "<< /Type /Page /Resources << /ExtGState << {states} >> \
                     /Font << /F1 5 0 R >> >> >>"
                ),
            ),
            (
                5,
                "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_owned(),
            ),
        ]);
        render_ink(&doc, format!("{CYAN_PAGE} /GS gs {text}").as_bytes())
    };
    let ink = render("/GS << /OP true /OPM 1 >>");
    let pixels = ink.as_chunks::<8>().0;
    assert!(pixels.iter().any(|px| px[3] > 0), "the glyph paints");
    assert!(
        pixels.iter().all(|px| px[0] == 255),
        "the cyan is kept everywhere"
    );
    let ink = render("/GS << /OP false >>");
    let pixels = ink.as_chunks::<8>().0;
    assert!(pixels.iter().any(|px| px[3] > 0 && px[0] < 255));
}

/// A cached Type 3 glyph — a `d1` shape — paints the fill colour under
/// the fill's overprint.
#[test]
fn type3_glyph_overprints() {
    let doc = crate::test_helpers::make_doc_with_objects(&[
        (
            4,
            "<< /Type /Page /Resources << /Font << /T 5 0 R >> \
             /ExtGState << /GS_op << /OP true /OPM 1 >> >> >> >>"
                .to_owned(),
        ),
        (
            5,
            "<< /Type /Font /Subtype /Type3 /FontBBox [0 0 1 1] \
             /FontMatrix [1 0 0 1 0 0] /CharProcs << /a 6 0 R >> \
             /Encoding << /Type /Encoding /Differences [97 /a] >> \
             /FirstChar 97 /LastChar 97 /Widths [1] >>"
                .to_owned(),
        ),
        (6, stream_object("", "1 0 0 0 1 1 d1 0 0 1 1 re f")),
    ]);
    let content = format!("{CYAN_PAGE} /GS_op gs 0 0 0 1 k BT /T 5 Tf 0 0 Td (a) Tj ET");
    let ink = render_ink(&doc, content.as_bytes());
    // The glyph covers the bottom-left 5 × 5 pixels.
    assert_eq!(ink_at(&ink, 2, 7), [255, 0, 0, 255, 0, 0, 0, 0]);
    assert_eq!(ink_at(&ink, 7, 2), [255, 0, 0, 0, 0, 0, 0, 0]);
}

/// A shading paints the channels of its colour space, with no overprint
/// mode: a `DeviceCMYK` shading leaves a spot under it, an RGB one
/// knocks it out — through `sh` and through a shading pattern alike.
#[test]
fn shadings_overprint_by_colour_space() {
    const K: &str = "<< /FunctionType 2 /Domain [0 1] /C0 [0 0 0 1] /C1 [0 0 0 1] /N 1 >>";
    const BLACK: &str = "<< /FunctionType 2 /Domain [0 1] /C0 [0 0 0] /C1 [0 0 0] /N 1 >>";
    let doc = crate::test_helpers::make_doc_with_objects(&[(
        4,
        format!(
            "<< /Type /Page /Resources << /ExtGState << /GS << /op true >> >> \
             /ColorSpace << {} >> \
             /Shading << /K << /ShadingType 2 /ColorSpace /DeviceCMYK /Coords [0 0 10 0] \
             /Function {K} /Extend [true true] >> \
             /Rgb << /ShadingType 2 /ColorSpace /DeviceRGB /Coords [0 0 10 0] \
             /Function {BLACK} /Extend [true true] >> >> \
             /Pattern << /PK << /PatternType 2 /Shading << /ShadingType 2 \
             /ColorSpace /DeviceCMYK /Coords [0 0 10 0] /Function {K} \
             /Extend [true true] >> >> >> >> >>",
            spot_space()
        ),
    )]);
    let spot_page = "/Spot cs 1 scn 0 0 10 10 re f /GS gs";
    for (paint, spot) in [
        ("q 0 0 5 10 re W n /K sh Q", 255),
        ("q 0 0 5 10 re W n /Rgb sh Q", 0),
        ("/Pattern cs /PK scn 0 0 5 10 re f", 255),
    ] {
        let ink = render_ink(&doc, format!("{spot_page} {paint}").as_bytes());
        let left = ink_at(&ink, 2, 5);
        assert!(left[3] > 0, "{paint}: the shading paints black: {left:?}");
        assert_eq!(left[4], spot, "{paint}: the spot channel");
        assert_eq!(ink_at(&ink, 7, 5), [0, 0, 0, 0, 255, 0, 0, 0], "{paint}");
    }
}

/// Inside a knockout group an overprinting fill keeps the channels it
/// does not paint of the group's initial backdrop, the cyan of the page.
#[test]
fn overprint_in_a_knockout_group_keeps_the_backdrop() {
    let doc = crate::test_helpers::make_doc_with_objects(&[
        (
            4,
            "<< /Type /Page /Resources << /XObject << /F 5 0 R >> >> >>".to_owned(),
        ),
        (
            5,
            stream_object(
                "/Type /XObject /Subtype /Form /BBox [0 0 10 10] \
                 /Group << /S /Transparency /K true >> \
                 /Resources << /ExtGState << /GS_op << /OP true /OPM 1 >> >> >>",
                "/GS_op gs 0 0 0 1 k 0 0 5 10 re f",
            ),
        ),
    ]);
    let content = format!("{CYAN_PAGE} /F Do");
    let ink = render_ink(&doc, content.as_bytes());
    assert_eq!(ink_at(&ink, 2, 5), [255, 0, 0, 255, 0, 0, 0, 0]);
    assert_eq!(ink_at(&ink, 7, 5), [255, 0, 0, 0, 0, 0, 0, 0]);
}

/// A colour of four zero process components under nonzero overprint
/// mode paints no channel: the page stays as the backdrop.
#[test]
fn all_zero_cmyk_under_opm1_paints_nothing() {
    let doc = overprint_doc("/GS_op << /OP true /op true /OPM 1 >>", "");
    let content = format!("{CYAN_PAGE} /GS_op gs 0 0 0 0 k 0 0 5 10 re f");
    let ink = render_ink(&doc, content.as_bytes());
    assert_eq!(ink_at(&ink, 2, 5), [255, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(ink_at(&ink, 7, 5), [255, 0, 0, 0, 0, 0, 0, 0]);
}

/// Page `(4, 0)` of a solid cyan fill followed by the form `/F`, a
/// non-isolated group whose content paints black under overprint through
/// the luminosity mask `/M`; the mask group's content is 50 % grey painted
/// under the blend mode `blend`.
fn masked_group_doc(blend: &str) -> Document {
    crate::test_helpers::make_doc_with_objects(&[
        (
            4,
            "<< /Type /Page /Resources << /XObject << /F 5 0 R >> >> >>".to_owned(),
        ),
        (
            5,
            stream_object(
                "/Type /XObject /Subtype /Form /BBox [0 0 10 10] \
                 /Group << /S /Transparency /I false >> \
                 /Resources << /ExtGState << /GS_op << /OP true /OPM 1 >> \
                 /M << /SMask << /S /Luminosity /G 6 0 R /BC [0] >> >> >> >>",
                "/GS_op gs /M gs 0 0 0 1 k 0 0 10 10 re f",
            ),
        ),
        (
            6,
            stream_object(
                &format!(
                    "/Type /XObject /Subtype /Form /BBox [0 0 10 10] \
                     /Group << /S /Transparency /CS /DeviceGray >> \
                     /Resources << /ExtGState << /B << /BM /{blend} >> >> >>"
                ),
                "/B gs 0.5 g 0 0 10 10 re f",
            ),
        ),
    ])
}

/// A soft mask set inside a non-isolated group is rendered against its own
/// backdrop `BC`, not the group's: a blend mode in the mask content has
/// nothing to blend with, so the mask is the 50 % grey it paints and the
/// masked black prints at half strength over the cyan it overprints.
#[test]
fn soft_mask_in_a_group_ignores_the_group_backdrop() {
    let render = |blend: &str| {
        let doc = masked_group_doc(blend);
        let mut renderer = PageRenderer::new(10, 10, &doc, (4, 0));
        renderer.set_overprint(true);
        assert!(matches!(renderer.canvas, PageCanvas::Ink(_)));
        renderer.execute(&crate::content::parse(
            format!("{CYAN_PAGE} /F Do").as_bytes(),
        ));
        let PageCanvas::Ink(ink) = &renderer.canvas else {
            panic!("an ink canvas");
        };
        ink.data().to_vec()
    };
    let (multiply, normal) = (render("Multiply"), render("Normal"));
    for (x, y) in [(0, 0), (1, 0), (2, 0), (5, 5), (9, 9)] {
        let [c, m, yl, k, s @ ..] = ink_at(&multiply, x, y);
        assert_eq!((c, m, yl, s), (255, 0, 0, [0; 4]), "({x}, {y}): {k}");
        assert!((126..=129).contains(&k), "({x}, {y}): black at {k}");
    }
    assert_eq!(multiply, normal);
}
