//! Pages finished as CMYK: the process ink of an ink canvas, its spots
//! folded in, with overprint simulated only when requested.

use color::cmyk::rgb_to_device_cmyk;
use color::{Sample, SampleDepth};
use pdf::Document;

use super::*;
use crate::renderer::FinishedCanvas;
use crate::separation::{CanvasRequest, Separation};

/// Page `(4, 0)` whose `/ExtGState` resources are the entries `states` and
/// whose `/ColorSpace` resources are the entries `spaces`.
fn page_doc(states: &str, spaces: &str) -> Document {
    crate::test_helpers::make_doc_with_objects(&[(
        4,
        format!(
            "<< /Type /Page /Resources << /ExtGState << {states} >> \
             /ColorSpace << {spaces} >> >> >>"
        ),
    )])
}

/// The request of a CMYK page separated by the device conversion,
/// simulating overprint when `overprint`.
const fn cmyk(overprint: bool) -> CanvasRequest {
    CanvasRequest::Cmyk {
        overprint,
        separation: Separation::Device,
    }
}

/// `content` rendered on page `(4, 0)` of `doc`, `size` pixels square, as
/// a CMYK page of samples of `depth` under `request`: four samples a
/// pixel, widened to 16 bits.
fn render(
    doc: &Document,
    (size, depth): (u32, SampleDepth),
    request: &CanvasRequest,
    content: &[u8],
) -> Vec<u16> {
    let view = PageRenderer::unit_view(size, size);
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
    renderer.execute(&crate::content::parse(content));
    match renderer.try_finish_keeping(false).expect("finishes").0 {
        FinishedCanvas::Cmyk8(b) => b.data().iter().map(|&v| u16::from(v)).collect(),
        FinishedCanvas::Cmyk16(b) => b.data().to_vec(),
        other => panic!("a CMYK page, not {other:?}"),
    }
}

/// The CMYK of pixel `(x, y)` of the 10-pixel-square page `page`.
fn at(page: &[u16], x: usize, y: usize) -> [u16; 4] {
    let i = (y * 10 + x) * 4;
    page[i..i + 4].try_into().unwrap()
}

/// The samples of `v` ([0, 1] each) at `depth`, widened to 16 bits.
fn of_unit(v: [f64; 4], depth: SampleDepth) -> [u16; 4] {
    v.map(|v| match depth {
        SampleDepth::Eight => u16::from(u8::from_unit(v)),
        SampleDepth::Sixteen => u16::from_unit(v),
    })
}

const DEPTHS: [SampleDepth; 2] = [SampleDepth::Eight, SampleDepth::Sixteen];

/// The `DeviceCMYK` colours of the swatch fixture.
const SWATCHES: [[f64; 4]; 6] = [
    [0.87, 0.01, 0.0, 0.0],
    [1.0, 0.46, 0.05, 0.18],
    [0.0, 0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 1.0, 0.0],
    [0.5, 0.5, 0.5, 0.0],
];

#[test]
fn device_cmyk_fills_keep_their_values() {
    let doc = page_doc("", "");
    for depth in DEPTHS {
        for swatch in SWATCHES {
            let [c, m, y, k] = swatch;
            let content = format!("{c} {m} {y} {k} k 0 0 10 10 re f");
            let page = render(&doc, (10, depth), &cmyk(false), content.as_bytes());
            assert_eq!(
                at(&page, 5, 5),
                of_unit(swatch, depth),
                "{swatch:?} at {depth:?}"
            );
        }
        let blank = render(&doc, (10, depth), &cmyk(false), b"");
        assert!(blank.iter().all(|&v| v == 0), "paper is no ink");
    }
}

#[test]
fn a_fill_in_a_transparency_group_keeps_its_values() {
    let doc = crate::test_helpers::make_doc_with_objects(&[
        (
            4,
            "<< /Type /Page /Resources << /XObject << /F 5 0 R >> >> >>".to_owned(),
        ),
        (
            5,
            super::stream_object(
                "/Type /XObject /Subtype /Form /BBox [0 0 10 10] \
                 /Group << /S /Transparency /I true >>",
                "1 0.46 0.05 0.18 k 0 0 10 10 re f",
            ),
        ),
    ]);
    for depth in DEPTHS {
        let page = render(&doc, (10, depth), &cmyk(false), b"/F Do");
        assert_eq!(at(&page, 5, 5), of_unit(SWATCHES[1], depth), "{depth:?}");
    }
}

#[test]
fn overprint_is_simulated_only_when_requested() {
    let doc = page_doc("/GS_op << /OP true /op true /OPM 1 >>", "");
    let content = b"1 0 0 0 k 0 0 10 10 re f /GS_op gs 0 0 0 1 k 0 0 5 10 re f";
    let knocked = render(&doc, (10, SampleDepth::Eight), &cmyk(false), content);
    assert_eq!(at(&knocked, 2, 5), [0, 0, 0, 255]);
    assert_eq!(at(&knocked, 7, 5), [255, 0, 0, 0]);
    let kept = render(&doc, (10, SampleDepth::Eight), &cmyk(true), content);
    assert_eq!(at(&kept, 2, 5), [255, 0, 0, 255]);
    assert_eq!(at(&kept, 7, 5), [255, 0, 0, 0]);
}

#[test]
fn gray_and_rgb_separate_by_the_device_conversion() {
    let doc = page_doc("", "");
    for (content, rgb) in [
        (&b"0 g"[..], [0u8; 3]),
        (b"0.5 g", [128; 3]),
        (b"1 0 0 rg", [255, 0, 0]),
        (b"0.4 0.8 0.1 rg", [102, 204, 26]),
    ] {
        let fill = [content, b" 0 0 10 10 re f"].concat();
        let page = render(&doc, (10, SampleDepth::Eight), &cmyk(false), &fill);
        let want = rgb_to_device_cmyk(rgb).map(u16::from);
        assert_eq!(
            at(&page, 5, 5),
            want,
            "{}",
            String::from_utf8_lossy(content)
        );
    }
    let page = render(
        &doc,
        (10, SampleDepth::Sixteen),
        &cmyk(false),
        b"0 g 0 0 10 10 re f",
    );
    assert_eq!(
        at(&page, 5, 5),
        [0, 0, 0, 65_535],
        "black text is black ink alone"
    );
}

/// The black-generation and undercolour-removal functions of the graphics
/// state apply to the device conversion of an RGB colour, from the paint
/// after the `gs` that sets them, also for a colour set before it; a gray
/// stays black ink alone; `/Default` restores the device's functions.
#[test]
fn black_generation_and_undercolour_removal_apply_to_rgb() {
    let zero = "<< /FunctionType 2 /Domain [0 1] /C0 [0] /C1 [0] /N 1 >>";
    let doc = page_doc(
        &format!("/None << /BG {zero} /UCR {zero} >> /Back << /BG2 /Default /UCR2 /Default >>"),
        "/CG [/CalGray << /WhitePoint [0.9505 1 1.089] >>]",
    );
    for depth in DEPTHS {
        let page = render(
            &doc,
            (10, depth),
            &cmyk(false),
            b"0.2 0.4 0.6 rg 0 0 2 10 re f /None gs 2 0 2 10 re f \
              0.2 g 4 0 2 10 re f /CG cs 0.2 sc 6 0 2 10 re f \
              /Back gs 0.2 0.4 0.6 rg 8 0 2 10 re f",
        );
        let device = of_unit([0.4, 0.2, 0.0, 0.4], depth);
        assert_eq!(at(&page, 1, 5), device, "{depth:?}: before the gs");
        assert_eq!(
            at(&page, 3, 5),
            of_unit([0.8, 0.6, 0.4, 0.0], depth),
            "{depth:?}: no black generated, no undercolour removed"
        );
        assert_eq!(
            at(&page, 5, 5),
            of_unit([0.0, 0.0, 0.0, 0.8], depth),
            "{depth:?}: DeviceGray"
        );
        let cal = at(&page, 7, 5);
        // Black generated as the device does; the conversion through XYZ
        // leaves the colour neutral within a few 16-bit levels.
        assert!(
            cal[..3].iter().all(|&v| v <= 8) && cal[3] > of_unit([0.0, 0.0, 0.0, 0.5], depth)[3],
            "{depth:?}: CalGray {cal:?}"
        );
        assert_eq!(at(&page, 9, 5), device, "{depth:?}: /Default");
    }
}

#[test]
fn a_spot_folds_into_its_process_equivalent() {
    let space = format!(
        "/Spot [/Separation /Spot /DeviceCMYK {}]",
        super::tint_to_cmyk("0 0.91 0.76 0")
    );
    let doc = page_doc("/GS << /op true >>", &space);
    for depth in DEPTHS {
        let page = render(
            &doc,
            (10, depth),
            &cmyk(false),
            b"/Spot cs 1 scn 0 0 10 10 re f",
        );
        assert_eq!(
            at(&page, 5, 5),
            of_unit([0.0, 0.91, 0.76, 0.0], depth),
            "{depth:?}"
        );
    }
    // Overprinting cyan, the spot keeps the cyan under it.
    let content = b"1 0 0 0 k 0 0 10 10 re f /GS gs /Spot cs 1 scn 0 0 10 10 re f";
    let page = render(&doc, (10, SampleDepth::Eight), &cmyk(true), content);
    assert_eq!(at(&page, 5, 5), [255, 232, 194, 0]);
}

/// An `ICCBased` CMYK colour paints its components on a CMYK page: they
/// are a CMYK device's values already.
#[test]
fn icc_cmyk_colours_keep_their_components() {
    const PROFILE: &[u8] =
        include_bytes!("../../../../../jpeg2000/assets/CGATS001Compat-v2-micro.icc");
    let mut profile = format!("<< /N 4 /Length {} >>\nstream\n", PROFILE.len()).into_bytes();
    profile.extend_from_slice(PROFILE);
    profile.extend_from_slice(b"\nendstream");
    let page =
        b"<< /Type /Page /Resources << /ColorSpace << /ICC [/ICCBased 5 0 R] >> >> >>".to_vec();
    let doc = crate::test_helpers::make_doc_with_objects(&[(4, page), (5, profile)]);
    for depth in DEPTHS {
        let page = render(
            &doc,
            (10, depth),
            &cmyk(false),
            b"/ICC cs 0.6 0.3 0 0.1 sc 0 0 10 10 re f",
        );
        assert_eq!(
            at(&page, 5, 5),
            of_unit([0.6, 0.3, 0.0, 0.1], depth),
            "{depth:?}"
        );
    }
}

/// The overprint preview of the same page is unchanged by the CMYK
/// finish: it renders through the same canvas to sRGB.
#[test]
fn the_overprint_preview_still_finishes_as_srgb() {
    let doc = page_doc("", "");
    let mut renderer = PageRenderer::new_ink(10, 10, &doc, (4, 0));
    renderer.execute(&crate::content::parse(b"0.2 0.5 0.1 0.3 k 0 0 10 10 re f"));
    let (finished, _, _) = renderer.try_finish_keeping(false).expect("finishes");
    assert!(matches!(finished, FinishedCanvas::Rgb8(_)), "{finished:?}");
}

/// Page `(4, 0)` drawing the image `Im`, object 5, of dictionary entries
/// `dict` and samples `samples`.
fn image_doc(dict: &str, samples: &[u8]) -> Document {
    let mut image = format!(
        "<< /Type /XObject /Subtype /Image /Width 1 /Height 1 {dict} /Length {} >>\nstream\n",
        samples.len()
    )
    .into_bytes();
    image.extend_from_slice(samples);
    image.extend_from_slice(b"\nendstream");
    crate::test_helpers::make_doc_with_objects(&[
        (
            4,
            b"<< /Type /Page /Resources << /XObject << /Im 5 0 R >> >> >>".to_vec(),
        ),
        (5, image),
    ])
}

#[test]
fn device_cmyk_images_paint_their_samples() {
    let doc = image_doc(
        "/ColorSpace /DeviceCMYK /BitsPerComponent 8",
        &[10, 100, 50, 0],
    );
    for depth in DEPTHS {
        let page = render(&doc, (10, depth), &cmyk(false), b"10 0 0 10 0 0 cm /Im Do");
        let want = [10u8, 100, 50, 0].map(|v| match depth {
            SampleDepth::Eight => u16::from(v),
            SampleDepth::Sixteen => u16::from_u8(v),
        });
        assert_eq!(at(&page, 5, 5), want, "{depth:?}");
    }
    let wide = image_doc(
        "/ColorSpace /DeviceCMYK /BitsPerComponent 16",
        &[0x12, 0x34, 0x56, 0x78, 0, 0, 0x7f, 0x01],
    );
    let page = render(
        &wide,
        (10, SampleDepth::Sixteen),
        &cmyk(false),
        b"10 0 0 10 0 0 cm /Im Do",
    );
    assert_eq!(at(&page, 5, 5), [0x1234, 0x5678, 0, 0x7f01]);
    let inverted = image_doc(
        "/ColorSpace /DeviceCMYK /BitsPerComponent 8 /Decode [1 0 1 0 1 0 1 0]",
        &[255, 155, 205, 255],
    );
    let page = render(
        &inverted,
        (10, SampleDepth::Eight),
        &cmyk(false),
        b"10 0 0 10 0 0 cm /Im Do",
    );
    assert_eq!(at(&page, 5, 5), [0, 100, 50, 0], "/Decode applies");
}

/// On the overprint preview a `DeviceCMYK` image paints its samples as
/// ink too, so that it overprints with its own channels.
#[test]
fn device_cmyk_images_paint_their_samples_on_the_preview() {
    let doc = image_doc(
        "/ColorSpace /DeviceCMYK /BitsPerComponent 8",
        &[10, 100, 50, 0],
    );
    let mut renderer = PageRenderer::new_ink(10, 10, &doc, (4, 0));
    renderer.execute(&crate::content::parse(b"10 0 0 10 0 0 cm /Im Do"));
    let PageCanvas::Ink(ink) = &renderer.canvas else {
        panic!("an ink canvas");
    };
    let i = (5 * 10 + 5) * 8;
    assert_eq!(ink.data()[i..i + 8], [10, 100, 50, 0, 0, 0, 0, 0]);
}

/// A shading in `DeviceCMYK` paints its colours as ink: no black enters
/// a cyan-to-black ramp's magenta and yellow, and a constant colour keeps
/// its value exactly.
#[test]
fn device_cmyk_shadings_paint_their_colours() {
    let shading = |c0: &str, c1: &str| {
        crate::test_helpers::make_doc_with_objects(&[(
            4,
            format!(
                "<< /Type /Page /Resources << /Shading << /Sh << /ShadingType 2 \
                 /ColorSpace /DeviceCMYK /Coords [0 0 10 0] /Extend [true true] \
                 /Function << /FunctionType 2 /Domain [0 1] /C0 [{c0}] /C1 [{c1}] /N 1 >> \
                 >> >> >> >>"
            ),
        )])
    };
    for depth in DEPTHS {
        let flat = shading("0.87 0.01 0 0", "0.87 0.01 0 0");
        let page = render(&flat, (10, depth), &cmyk(false), b"/Sh sh");
        assert_eq!(at(&page, 5, 5), of_unit(SWATCHES[0], depth), "{depth:?}");
        let ramp = shading("1 0 0 0", "0 0 0 1");
        let page = render(&ramp, (10, depth), &cmyk(false), b"/Sh sh");
        for x in 0..10 {
            let [c, m, y, k] = at(&page, x, 5);
            assert_eq!((m, y), (0, 0), "{depth:?} x {x}: {c} {m} {y} {k}");
            assert!(
                c > 0 && (k > 0) == (x > 0),
                "{depth:?} x {x}: {c} {m} {y} {k}"
            );
        }
    }
}

/// An `ICCBased` CMYK image paints its samples on a CMYK page and its
/// sRGB conversion on the preview.
#[test]
fn icc_cmyk_images_paint_their_samples_on_a_cmyk_page() {
    const PROFILE: &[u8] =
        include_bytes!("../../../../../jpeg2000/assets/CGATS001Compat-v2-micro.icc");
    let mut profile = format!("<< /N 4 /Length {} >>\nstream\n", PROFILE.len()).into_bytes();
    profile.extend_from_slice(PROFILE);
    profile.extend_from_slice(b"\nendstream");
    let mut image = b"<< /Type /XObject /Subtype /Image /Width 1 /Height 1 \
        /ColorSpace [/ICCBased 6 0 R] /BitsPerComponent 8 /Length 4 >>\nstream\n"
        .to_vec();
    image.extend_from_slice(&[10, 100, 50, 0]);
    image.extend_from_slice(b"\nendstream");
    let doc = crate::test_helpers::make_doc_with_objects(&[
        (
            4,
            b"<< /Type /Page /Resources << /XObject << /Im 5 0 R >> >> >>".to_vec(),
        ),
        (5, image),
        (6, profile),
    ]);
    let page = render(
        &doc,
        (10, SampleDepth::Eight),
        &cmyk(false),
        b"10 0 0 10 0 0 cm /Im Do",
    );
    assert_eq!(at(&page, 5, 5), [10, 100, 50, 0]);
}
