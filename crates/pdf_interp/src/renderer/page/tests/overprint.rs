//! Overprint: which channels a painting operation leaves to the backdrop.

use pdf::Document;
use raster::pipe::ChannelMask;

use super::*;
use crate::renderer::page::CanvasKind;

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

/// An RGB or grey colour paints every channel and is not eligible for
/// nonzero overprint mode: under overprint an RGB fill over a spot knocks
/// the spot out where it paints and leaves it beside.
#[test]
fn rgb_colour_knocks_out_spots() {
    let doc = overprint_doc("/GS << /op true /OPM 1 >>", &spot_space());
    let content = "/GS gs /Spot cs 1 scn 0 0 10 10 re f 0.2 0.4 0.6 rg 0 0 5 10 re f";
    let (fill, ink) = ink_fill(&doc, content.as_bytes());
    assert_eq!(fill.paints(), ChannelMask::ALL);
    assert!(!fill.opm_eligible());
    assert_eq!(ink_at(&ink, 2, 5)[4], 0, "the spot under the RGB fill");
    assert_eq!(ink_at(&ink, 7, 5)[4], 255, "the spot beside it");
    let (fill, _) = ink_fill(&doc, b"0.2 g");
    assert_eq!(fill.paints(), ChannelMask::ALL);
}

/// An indexed colour paints as its palette entry in the base space: a
/// black entry of a `DeviceCMYK` palette overprints the cyan under
/// nonzero overprint mode, and an entry of a spot palette paints the
/// spot's channel alone.
#[test]
fn indexed_colour_paints_as_its_base_space() {
    let spaces = format!(
        "/Ix [/Indexed /DeviceCMYK 0 <000000FF>] /Is [/Indexed {} 0 <FF>] {}",
        spot_space().trim_start_matches("/Spot "),
        spot_space()
    );
    let doc = overprint_doc("/GS_op << /OP true /OPM 1 >>", &spaces);
    let content = format!("{CYAN_PAGE} /GS_op gs /Ix cs 0 sc 0 0 5 10 re f");
    let ink = render_ink(&doc, content.as_bytes());
    assert_eq!(ink_at(&ink, 2, 5), [255, 0, 0, 255, 0, 0, 0, 0]);
    assert_eq!(ink_at(&ink, 7, 5), [255, 0, 0, 0, 0, 0, 0, 0]);
    let content = format!("{CYAN_PAGE} /GS_op gs /Is cs 0 sc 0 0 5 10 re f");
    let ink = render_ink(&doc, content.as_bytes());
    assert_eq!(ink_at(&ink, 2, 5), [255, 0, 0, 0, 255, 0, 0, 0]);
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

/// The translucent stroke of a fill-and-stroke overprints under `OP`, not
/// `op`: a half-opaque black stroke keeps the cyan under `OP true` and
/// knocks half of it out under `op true` alone.
#[test]
fn translucent_fill_and_stroke_uses_op_stroke() {
    for (states, cyan) in [
        ("/GS << /OP true /op false /OPM 1 /CA 0.5 >>", 255),
        ("/GS << /OP false /op true /OPM 1 /CA 0.5 >>", 127),
    ] {
        let doc = overprint_doc(states, "");
        let content = format!("{CYAN_PAGE} /GS gs 0 0 0 1 K 0 0 0 1 k 4 w 0 5 m 4 5 l B");
        let ink = render_ink(&doc, content.as_bytes());
        let [c, m, y, k, ..] = ink_at(&ink, 2, 5);
        assert_eq!((c, m, y), (cyan, 0, 0), "{states}");
        assert!((127..=128).contains(&k), "{states}: black at {k}");
    }
}

/// A page whose graphics states do not set overprint is not put on the
/// ink canvas, whatever else it paints.
#[test]
fn a_page_without_op_does_not_use_overprint() {
    let doc = overprint_doc("/GS << /ca 1 /OPM 1 >>", &spot_space());
    assert!(!crate::page_uses_overprint(&doc, (4, 0), true));
    let doc = overprint_doc("/GS << /op true >>", &spot_space());
    assert!(crate::page_uses_overprint(&doc, (4, 0), true));
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
        assert!(crate::page_uses_overprint(&doc, (4, 0), true));
        render_ink(&doc, format!("{CYAN_PAGE} /F Do").as_bytes())
    };
    let (multiply, normal) = (render("Multiply"), render("Normal"));
    for (x, y) in [(0, 0), (1, 0), (2, 0), (5, 5), (9, 9)] {
        let [c, m, yl, k, s @ ..] = ink_at(&multiply, x, y);
        assert_eq!((c, m, yl, s), (255, 0, 0, [0; 4]), "({x}, {y}): {k}");
        assert!((126..=129).contains(&k), "({x}, {y}): black at {k}");
    }
    assert_eq!(multiply, normal);
}

/// A page group finished without the composite onto paper keeps its alpha
/// plane on either canvas: opaque where painted, zero elsewhere, the ink
/// canvas's colour converted to sRGB.
#[test]
fn a_transparent_finish_keeps_the_page_alpha() {
    let doc = overprint_doc("", "");
    let group = crate::resources::TransparencyGroupParams {
        isolated: true,
        knockout: false,
    };
    for ink in [false, true] {
        let mut renderer = if ink {
            PageRenderer::new_ink(10, 10, &doc, (4, 0))
        } else {
            PageRenderer::new(10, 10, &doc, (4, 0))
        };
        renderer.set_page_group(Some(group));
        renderer.execute(&crate::content::parse(b"1 0 0 0 k 0 0 5 10 re f"));
        let (bitmap, _) = renderer.try_finish_transparent().unwrap();
        let alpha = bitmap.alpha_plane().expect("an alpha plane");
        assert_eq!(alpha[2], u8::MAX, "ink {ink}: painted");
        assert_eq!(alpha[7], 0, "ink {ink}: unpainted");
        assert_eq!(
            rgb_at(bitmap.data(), 2, 0),
            color::cmyk::cmyk_to_rgb(255, 0, 0, 0),
            "ink {ink}: cyan"
        );
        assert_eq!(rgb_at(bitmap.data(), 7, 0), [0; 3], "ink {ink}: no colour");
    }
}

/// On an ink page the soft-mask groups are RGB and the transparency groups
/// ink; alternating between them reuses one pooled canvas of each kind.
#[test]
fn group_pool_keeps_one_canvas_per_kind() {
    let doc = crate::test_helpers::make_doc_with_objects(&[
        (
            4,
            "<< /Type /Page /Resources << /XObject << /F 5 0 R >> \
             /ExtGState << /M << /SMask << /S /Luminosity /G 6 0 R >> >> \
             /N << /SMask /None >> >> >> >>"
                .to_owned(),
        ),
        (
            5,
            stream_object(
                "/Type /XObject /Subtype /Form /BBox [0 0 10 10] \
                 /Group << /S /Transparency >>",
                "0 0 0 1 k 0 0 5 10 re f",
            ),
        ),
        (
            6,
            stream_object(
                "/Type /XObject /Subtype /Form /BBox [0 0 10 10] \
                 /Group << /S /Transparency /CS /DeviceGray >>",
                "0.5 g 0 0 10 10 re f",
            ),
        ),
    ]);
    let mut renderer = ink_renderer(&doc);
    renderer.execute(&crate::content::parse(
        b"/M gs /N gs /F Do /M gs /N gs /F Do /M gs /N gs /F Do",
    ));
    let kinds: Vec<_> = renderer.group_pool.iter().map(PageCanvas::kind).collect();
    assert_eq!(kinds.len(), 2, "{kinds:?}");
    assert!(kinds.contains(&CanvasKind::Rgb) && kinds.contains(&CanvasKind::Ink));
}

/// The antialiased edge of a grey or RGB black fill on an ink canvas
/// converts to a neutral grey: the ink of black, composited with paper at
/// partial coverage, prints with no colour cast.
#[test]
fn black_fill_edges_stay_neutral_on_the_ink_canvas() {
    let doc = overprint_doc("/GS << /op true >>", "");
    for colour in ["0 g", "0 0 0 rg", "0.05 g"] {
        let content = format!("{colour} 0 0 m 10 0 l 0 10 l f");
        let rgb = render_canvas(&doc, 10, content.as_bytes(), true);
        let edges: Vec<[u8; 3]> = rgb
            .as_chunks::<3>()
            .0
            .iter()
            .copied()
            .filter(|px| px.iter().all(|&v| (40..250).contains(&v)))
            .collect();
        assert!(!edges.is_empty(), "{content}: the diagonal has edge pixels");
        for [r, g, b] in edges {
            let chroma = r.max(g).max(b) - r.min(g).min(b);
            assert!(chroma <= 5, "{content}: edge pixel {:?}", [r, g, b]);
        }
    }
}

/// `/GS_t`: overprint under mode 1 with a transfer function that halves
/// every ink — `f(x) = (1 + x) / 2` on the additive value.
const HALVING: &str = "/GS_t << /OP true /op true /OPM 1 \
     /TR << /FunctionType 2 /Domain [0 1] /C0 [0.5] /C1 [1] /N 1 >> >>";

/// Under a transfer function a process colour still overprints: the black
/// fill keeps the cyan under it and prints its black mapped through the
/// function, on the complement of the ink.
#[test]
fn transfer_keeps_process_overprint() {
    let doc = overprint_doc(HALVING, "");
    let content = format!("{CYAN_PAGE} /GS_t gs 0 0 0 1 k 0 0 5 10 re f");
    let ink = render_ink(&doc, content.as_bytes());
    assert_eq!(ink_at(&ink, 2, 5), [255, 0, 0, 127, 0, 0, 0, 0]);
    assert_eq!(ink_at(&ink, 7, 5), [255, 0, 0, 0, 0, 0, 0, 0]);
}

/// Under a transfer function a spot colour keeps its slot and paints it
/// alone; its tint maps through the gray function.
#[test]
fn transfer_keeps_spot_separation() {
    let doc = overprint_doc(HALVING, &spot_space());
    let content = format!("{CYAN_PAGE} /GS_t gs /Spot cs 1 scn 0 0 5 10 re f");
    let ink = render_ink(&doc, content.as_bytes());
    assert_eq!(ink_at(&ink, 2, 5), [255, 0, 0, 0, 127, 0, 0, 0]);
    assert_eq!(ink_at(&ink, 7, 5), [255, 0, 0, 0, 0, 0, 0, 0]);
}

/// The ink of a grey colour or a grey image sample under a transfer
/// function is its converted ink mapped through the ink tables.
#[test]
fn transfer_maps_derived_ink_after_conversion() {
    let doc = overprint_doc(HALVING, "");
    let want = color::cmyk::rgb_to_ink([0; 3]).map(|v| v / 2);
    for content in [
        "/GS_t gs 0 g 0 0 10 10 re f",
        "/GS_t gs q 10 0 0 10 0 0 cm BI /W 1 /H 1 /CS /G /BPC 8 ID \x00 EI Q",
    ] {
        let ink = render_ink(&doc, content.as_bytes());
        let px = ink_at(&ink, 5, 5);
        assert!(
            px[..4].iter().zip(want).all(|(&a, b)| a.abs_diff(b) <= 1),
            "{content}: {px:?}, want {want:?}"
        );
    }
}

/// `/GS_i`: overprint with the inverting transfer function `f(x) = 1 − x`,
/// for which `f(1) ≠ 1`: an absent ink maps to full ink; `/GS_k`: the same
/// function without overprint.
const INVERTING: &str = "/GS_i << /OP true /op true \
     /TR << /FunctionType 2 /Domain [0 1] /C0 [1] /C1 [0] /N 1 >> >> \
     /GS_k << /OP false /op false \
     /TR << /FunctionType 2 /Domain [0 1] /C0 [1] /C1 [0] /N 1 >> >>";

/// A transfer function maps only the channels that carry a component of
/// the colour.  A grey, an RGB colour and a grey image sample, whose ink
/// holds no spot, leave every spot channel empty, the free ones included,
/// so a spot that takes a slot later shows only where it was painted.  A
/// spot colour that knocks out leaves the process channels empty.
#[test]
fn transfer_maps_only_the_colours_own_channels() {
    let doc = overprint_doc(INVERTING, &spot_space());
    for content in [
        "1 g 0 0 10 10 re f",
        "0.2 0.4 0.6 rg 0 0 10 10 re f",
        "10 0 0 10 0 0 cm BI /W 1 /H 1 /CS /G /BPC 8 ID \x40 EI",
    ] {
        let content = format!("q /GS_i gs {content} Q /Spot cs 1 scn 8 0 2 10 re f");
        let ink = render_ink(&doc, content.as_bytes());
        let px = ink_at(&ink, 2, 5);
        assert!(px[..4].iter().any(|&v| v != 0), "{content}: {px:?}");
        assert_eq!(px[4..], [0; 4], "{content}: {px:?}");
        assert_eq!(ink_at(&ink, 9, 5)[4], 255, "{content}");
    }
    let ink = render_ink(&doc, b"/GS_k gs /Spot cs 1 scn 0 0 10 10 re f");
    assert_eq!(ink_at(&ink, 5, 5), [0; 8]);
}

/// A shading in a `Separation` space paints its spot channel alone, so
/// under overprint the cyan under it stays: through `sh` for an axial, a
/// function-based and a mesh shading, and through a shading pattern with a
/// `Background`.  Without overprint it knocks the cyan out.
#[test]
fn spot_shadings_paint_their_colorant_alone() {
    let spot = format!(
        "[/Separation /Spot /DeviceCMYK {}]",
        tint_to_cmyk("0 0.91 0.76 0")
    );
    let one = "<< /FunctionType 2 /Domain [0 1] /C0 [1] /C1 [1] /N 1 >>";
    // Two triangles covering the page, every vertex at tint 1.
    let mesh: String = [[0, 0], [10, 0], [0, 10], [10, 10]]
        .iter()
        .enumerate()
        .flat_map(|(i, &[x, y])| [u8::from(i > 2), x, y, 127])
        .map(char::from)
        .collect();
    let constant = "{ pop pop 1 }";
    let doc = |op: bool| {
        crate::test_helpers::make_doc_with_objects(&[
            (
                4,
                format!(
                    "<< /Type /Page /Resources << /ExtGState << /GS << /op {op} >> >> \
                     /Shading << /Axial << /ShadingType 2 /ColorSpace {spot} \
                     /Coords [0 0 10 0] /Function {one} /Extend [true true] >> \
                     /Fn << /ShadingType 1 /ColorSpace {spot} /Domain [0 10 0 10] \
                     /Function 6 0 R >> /Mesh 5 0 R >> \
                     /Pattern << /P << /PatternType 2 /Shading << /ShadingType 2 \
                     /ColorSpace {spot} /Background [1] /BBox [0 0 1 1] \
                     /Coords [0 0 10 0] /Function {one} >> >> >> >> >>"
                ),
            ),
            (
                5,
                format!(
                    "<< /ShadingType 4 /ColorSpace {spot} /BitsPerCoordinate 8 \
                     /BitsPerComponent 8 /BitsPerFlag 8 /Decode [0 255 0 255 0 2.0079] \
                     /Length {} >>\nstream\n{mesh}\nendstream",
                    mesh.len()
                ),
            ),
            (
                6,
                format!(
                    "<< /FunctionType 4 /Domain [0 10 0 10] /Range [0 1] /Length {} >>\n\
                     stream\n{constant}\nendstream",
                    constant.len()
                ),
            ),
        ])
    };
    let cyan = "1 0 0 0 k 0 0 10 10 re f /GS gs";
    for paint in [
        "q 0 0 5 10 re W n /Axial sh Q",
        "q 0 0 5 10 re W n /Fn sh Q",
        "q 0 0 5 10 re W n /Mesh sh Q",
        "/Pattern cs /P scn 0 0 5 10 re f",
    ] {
        let content = format!("{cyan} {paint}");
        let ink = render_ink(&doc(true), content.as_bytes());
        assert_eq!(ink_at(&ink, 2, 5), [255, 0, 0, 0, 255, 0, 0, 0], "{paint}");
        assert_eq!(ink_at(&ink, 7, 5), [255, 0, 0, 0, 0, 0, 0, 0], "{paint}");
        let ink = render_ink(&doc(false), content.as_bytes());
        assert_eq!(ink_at(&ink, 2, 5), [0, 0, 0, 0, 255, 0, 0, 0], "{paint}");
    }
}

/// An image in a `Separation` space paints its spot channel alone, so
/// under overprint the cyan under it stays: an image `XObject` magnified
/// and reduced, and an inline image.  Without overprint it knocks the
/// cyan out.
#[test]
fn spot_images_paint_their_colorant_alone() {
    // 127 under `/Decode [0 2.0079]` is the full tint.
    let image = |id: u32, size: usize| {
        let data = "\x7f".repeat(size * size);
        (
            id,
            format!(
                "<< /Type /XObject /Subtype /Image /Width {size} /Height {size} \
                 /ColorSpace [/Separation /Spot /DeviceCMYK {}] /BitsPerComponent 8 \
                 /Decode [0 2.0079] /Length {} >>\nstream\n{data}\nendstream",
                tint_to_cmyk("0 0.91 0.76 0"),
                data.len()
            ),
        )
    };
    let doc = |op: bool| {
        crate::test_helpers::make_doc_with_objects(&[
            (
                4,
                format!(
                    "<< /Type /Page /Resources << /ExtGState << /GS << /op {op} >> >> \
                     /ColorSpace << {} >> /XObject << /Big 5 0 R /Small 6 0 R >> >> >>",
                    spot_space()
                ),
            ),
            image(5, 40),
            image(6, 2),
        ])
    };
    let cyan = "1 0 0 0 k 0 0 10 10 re f /GS gs q 5 0 0 10 0 0 cm";
    for paint in [
        "/Big Do",
        "/Small Do",
        "BI /W 1 /H 1 /CS /Spot /BPC 8 /D [0 2.0079] ID \x7f EI",
    ] {
        let content = format!("{cyan} {paint} Q");
        let ink = render_ink(&doc(true), content.as_bytes());
        assert_eq!(ink_at(&ink, 2, 5), [255, 0, 0, 0, 255, 0, 0, 0], "{paint}");
        assert_eq!(ink_at(&ink, 7, 5), [255, 0, 0, 0, 0, 0, 0, 0], "{paint}");
        let ink = render_ink(&doc(false), content.as_bytes());
        assert_eq!(ink_at(&ink, 2, 5), [0, 0, 0, 0, 255, 0, 0, 0], "{paint}");
    }
}

/// An image in the space of a fifth spot, which finds no slot, prints its
/// alternate colour in process ink as a fill of that space does: under
/// overprint both leave the spot slots beneath them.
#[test]
fn an_unslotted_spot_image_paints_the_process_channels() {
    let tint = tint_to_cmyk("0 0.91 0.76 0");
    let spaces = ["A", "B", "C", "D", "E"]
        .map(|name| format!("/{name} [/Separation /{name} /DeviceCMYK {tint}]"))
        .join(" ");
    let doc = overprint_doc("/GS << /OP true /op true >>", &spaces);
    let slots = "/GS gs /A cs 1 scn 0 0 10 10 re f /B cs 1 scn 0 0 10 10 re f \
                 /C cs 1 scn 0 0 10 10 re f /D cs 1 scn 0 0 10 10 re f";
    for paint in [
        "q 5 0 0 10 0 0 cm BI /W 1 /H 1 /CS /E /BPC 8 ID \x7f EI Q",
        "/E cs 0.5 scn 0 0 5 10 re f",
    ] {
        let content = format!("{slots} {paint}");
        let ink = render_ink(&doc, content.as_bytes());
        let px = ink_at(&ink, 2, 5);
        assert_eq!(px[4..], [255; 4], "{paint}: {px:?}");
        assert!(px[1] > 0 && px[2] > 0, "{paint}: {px:?}");
    }
}

/// A coloured tiling pattern's cell renders in ink with the page's spot
/// slots: a spot colour in the cell paints its slot, and a cell whose
/// content overprints leaves the cyan under the pattern.  Overprint set
/// by the paint using the pattern does not reach the cell's content.
#[test]
fn tiling_cells_keep_spots_and_their_own_overprint() {
    let doc = |cell_op: bool, cells: f64| {
        let cell = format!("/GS gs /Spot cs 1 scn 0 0 {cells} {cells} re f");
        crate::test_helpers::make_doc_with_objects(&[
            (
                4,
                format!(
                    "<< /Type /Page /Resources << /ExtGState << /OP << /op true >> >> \
                     /ColorSpace << {} >> /Pattern << /P 5 0 R >> >> >>",
                    spot_space()
                ),
            ),
            (
                5,
                format!(
                    "<< /PatternType 1 /PaintType 1 /TilingType 1 /BBox [0 0 {cells} {cells}] \
                     /XStep {cells} /YStep {cells} /Resources << /ColorSpace << {} >> \
                     /ExtGState << /GS << /op {cell_op} >> >> >> /Length {} >>\n\
                     stream\n{cell}\nendstream",
                    spot_space(),
                    cell.len()
                ),
            ),
        ])
    };
    // A cell larger than the fill, tiled, and below one pixel in area.
    for cells in [10.0, 2.0, 0.5] {
        let paint = format!("{CYAN_PAGE} /Pattern cs /P scn 0 0 5 10 re f");
        let ink = render_ink(&doc(false, cells), paint.as_bytes());
        assert_eq!(ink_at(&ink, 2, 5), [0, 0, 0, 0, 255, 0, 0, 0], "{cells}");
        assert_eq!(ink_at(&ink, 7, 5), [255, 0, 0, 0, 0, 0, 0, 0], "{cells}");
        let ink = render_ink(&doc(true, cells), paint.as_bytes());
        assert_eq!(ink_at(&ink, 2, 5), [255, 0, 0, 0, 255, 0, 0, 0], "{cells}");
        let paint = format!("{CYAN_PAGE} /OP gs /Pattern cs /P scn 0 0 5 10 re f");
        let ink = render_ink(&doc(false, cells), paint.as_bytes());
        assert_eq!(ink_at(&ink, 2, 5), [0, 0, 0, 0, 255, 0, 0, 0], "{cells}");
    }
}

/// An `ICCBased` CMYK colour prints on an ink canvas as the process ink
/// reproducing the profile's colour, with the colour's zero components:
/// a black from the profile stays black ink alone and keeps the cyan
/// under it under nonzero overprint mode, and a two-ink colour prints
/// with those two inks.
#[test]
fn icc_cmyk_prints_the_profiles_colour_with_its_own_inks() {
    const PROFILE: &[u8] =
        include_bytes!("../../../../../jpeg2000/assets/CGATS001Compat-v2-micro.icc");
    let mut profile = format!("<< /N 4 /Length {} >>\nstream\n", PROFILE.len()).into_bytes();
    profile.extend_from_slice(PROFILE);
    profile.extend_from_slice(b"\nendstream");
    let page = b"<< /Type /Page /Resources << /ExtGState << /GS << /op true /OPM 1 >> >> \
                 /ColorSpace << /ICC [/ICCBased 5 0 R] >> >> >>"
        .to_vec();
    let doc = crate::test_helpers::make_doc_with_objects(&[(4, page), (5, profile)]);
    let space = crate::resources::ColorSpace::IccBased {
        n: 4,
        stream_id: Some((5, 0)),
    };
    let prints = |ink: [u8; 8]| color::cmyk::cmyk_to_rgb(ink[0], ink[1], ink[2], ink[3]);
    let near = |a: [u8; 3], b: [u8; 3], tolerance: u8| {
        a.iter().zip(b).all(|(&x, y)| x.abs_diff(y) <= tolerance)
    };

    let black = space.convert_to_rgb(&doc, crate::ColourTarget::SRGB, &[0.0, 0.0, 0.0, 1.0]);
    let content = format!("{CYAN_PAGE} /GS gs /ICC cs 0 0 0 1 sc 0 0 5 10 re f");
    let ink = render_ink(&doc, content.as_bytes());
    let k = ink_at(&ink, 2, 5);
    assert_eq!(
        (k[0], k[1], k[2], &k[4..]),
        (255, 0, 0, &[0; 4][..]),
        "{k:?}"
    );
    let alone = prints([0, 0, 0, k[3], 0, 0, 0, 0]);
    let mean = black.iter().map(|&v| u32::from(v)).sum::<u32>() / 3;
    assert!(
        u32::from(alone[0]).abs_diff(mean) <= 1,
        "{alone:?} for {black:?}"
    );

    let blue = space.convert_to_rgb(&doc, crate::ColourTarget::SRGB, &[0.6, 0.3, 0.0, 0.0]);
    let ink = render_ink(&doc, b"/ICC cs 0.6 0.3 0 0 sc 0 0 10 10 re f");
    let cm = ink_at(&ink, 5, 5);
    assert_eq!((cm[2], cm[3]), (0, 0), "{cm:?}");
    assert!(
        near(prints(cm), blue, 3),
        "{cm:?} prints {:?}, not {blue:?}",
        prints(cm)
    );
}

/// The page's ink map is the page's when it finishes: a renderer still
/// sharing it is a bug that debug builds report rather than finish the
/// page without its spot tables.
#[test]
#[should_panic(expected = "outlived the page")]
fn finishing_while_the_ink_map_is_shared_fails() {
    let doc = overprint_doc("", &spot_space());
    let mut renderer = ink_renderer(&doc);
    renderer.execute(&crate::content::parse(b"/Spot cs 1 scn 0 0 10 10 re f"));
    let _shared = std::rc::Rc::clone(&renderer.ink_map);
    let _ = renderer.finish();
}
