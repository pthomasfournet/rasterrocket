//! Pages composited in the wide working space: exact white, fills and
//! images of one colour alike, blending in the working space, and soft
//! masks independent of it.

use pdf::Document;

use super::*;
use crate::WorkingSpace;
use crate::renderer::FinishedCanvas;
use crate::separation::CanvasRequest;

/// `stream` object bytes of `dict` entries and `data`.
fn stream(dict: &str, data: &[u8]) -> Vec<u8> {
    let mut s = format!("<< {dict} /Length {} >>\nstream\n", data.len()).into_bytes();
    s.extend_from_slice(data);
    s.extend_from_slice(b"\nendstream");
    s
}

/// The 1 × 1 image `XObject` of `space` holding the samples `px`.
fn image(space: &str, px: &[u8]) -> Vec<u8> {
    stream(
        &format!(
            "/Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace {space} \
             /BitsPerComponent 8"
        ),
        px,
    )
}

/// `content` rendered on page `(4, 0)` of `doc`, 10 pixels square, at 16
/// bits in `space`: three samples a pixel.
fn render(doc: &Document, space: WorkingSpace, content: &str) -> Vec<u16> {
    let request = match space {
        WorkingSpace::Srgb => CanvasRequest::Rgb,
        WorkingSpace::Wide => CanvasRequest::WideRgb,
    };
    let view = PageRenderer::unit_view(10, 10);
    let mut renderer = PageRenderer::try_with_page_group_in(
        &view,
        doc,
        (4, 0),
        None,
        &request,
        color::SampleDepth::Sixteen,
        super::super::CanvasBuffers::default(),
    )
    .expect("the canvas is allocated");
    renderer.execute(&crate::content::parse(content.as_bytes()));
    match renderer.try_finish_keeping(false).expect("finishes").0 {
        FinishedCanvas::Rgb16(b) => b.data().to_vec(),
        other => panic!("a 16-bit RGB page, not {other:?}"),
    }
}

/// The samples of pixel `(x, 0)` of a 10-pixel-square page.
fn at(page: &[u16], x: usize) -> [u16; 3] {
    page[3 * x..3 * x + 3].try_into().unwrap()
}

/// An RGB matrix/TRC profile: the Adobe RGB (1998) primaries adapted to
/// D50 and a pure power law of 563/256.
fn adobe_profile() -> Vec<u8> {
    moxcms::ColorProfile::new_adobe_rgb()
        .encode()
        .expect("encodes")
}

/// The sRGB profile, described by its tags.
fn srgb_profile() -> Vec<u8> {
    let mut srgb = moxcms::ColorProfile::new_srgb();
    srgb.cicp = None;
    srgb.encode().expect("encodes")
}

/// Page `(4, 0)` whose resources hold: `/A` the Adobe RGB profile (5),
/// `/S` the sRGB profile (6) and `/L` a D65 Lab space as colour spaces;
/// `/Half` a constant alpha of one half; images `/Rgb` (7), `/Adobe`
/// (8), `/Cmyk` (9) of the samples `51 153 204 (0)`, `/White` (10) the
/// sRGB-tagged white; `/Fm` (11) a form filling its box in the
/// `DeviceRGB` colour `0.2 0.6 0.8`; `/Lum` a luminosity soft mask whose
/// group (12) draws `/Rgb` and `/Fm` over the left and right halves.
fn doc() -> Document {
    let adobe = adobe_profile();
    let srgb = srgb_profile();
    let mask_group = b"q 5 0 0 10 0 0 cm /Rgb Do Q q 1 0 0 1 5 0 cm /Fm Do Q";
    crate::test_helpers::make_doc_with_objects(&[
        (
            4,
            b"<< /Type /Page /Resources << \
              /ColorSpace << /A [/ICCBased 5 0 R] /S [/ICCBased 6 0 R] \
              /L [/Lab << /WhitePoint [0.9505 1 1.089] >>] >> \
              /ExtGState << /Half << /ca 0.5 >> \
              /Lum << /SMask << /S /Luminosity /G 12 0 R >> >> >> \
              /XObject << /Rgb 7 0 R /Adobe 8 0 R /Cmyk 9 0 R /White 10 0 R /Fm 11 0 R >> \
              >> >>"
                .to_vec(),
        ),
        (5, stream("/N 3", &adobe)),
        (6, stream("/N 3", &srgb)),
        (7, image("/DeviceRGB", &[51, 153, 204])),
        (8, image("[/ICCBased 5 0 R]", &[51, 153, 204])),
        (9, image("/DeviceCMYK", &[51, 153, 204, 0])),
        (10, image("[/ICCBased 6 0 R]", &[255, 255, 255])),
        (
            11,
            stream(
                "/Type /XObject /Subtype /Form /BBox [0 0 5 10]",
                b"0.2 0.6 0.8 rg 0 0 5 10 re f",
            ),
        ),
        (
            12,
            stream(
                "/Type /XObject /Subtype /Form /BBox [0 0 10 10] \
                 /Group << /S /Transparency /CS /DeviceRGB >> \
                 /Resources << /XObject << /Rgb 7 0 R /Fm 11 0 R >> >>",
                mask_group,
            ),
        ),
    ])
}

/// White is the working white in the wide space, whatever space it is
/// given in: `DeviceRGB`, `DeviceGray`, an sRGB-tagged image, `DeviceCMYK`
/// without ink and Lab `100 0 0`.
#[test]
fn white_is_exact_in_the_wide_space() {
    let d = doc();
    let content = "0 g 0 0 10 10 re f \
                   1 1 1 rg 0 0 2 10 re f \
                   1 g 2 0 2 10 re f \
                   q 2 0 0 10 4 0 cm /White Do Q \
                   0 0 0 0 k 6 0 2 10 re f \
                   /L cs 100 0 0 sc 8 0 2 10 re f";
    let page = render(&d, WorkingSpace::Wide, content);
    for x in 0..10 {
        assert_eq!(at(&page, x), [65_535; 3], "pixel {x}");
    }
}

/// A colour converts to the same 16-bit canvas value as a fill and as an
/// 8-bit image of the same components, in either working space:
/// `DeviceRGB`, a matrix/TRC ICC space and `DeviceCMYK`.
#[test]
fn a_fill_and_an_image_of_one_colour_agree() {
    let d = doc();
    for space in WorkingSpace::ALL {
        for (fill, im) in [
            ("0.2 0.6 0.8 rg", "/Rgb"),
            ("/A cs 0.2 0.6 0.8 sc", "/Adobe"),
            ("0.2 0.6 0.8 0 k", "/Cmyk"),
        ] {
            let content = format!("{fill} 0 0 5 10 re f q 5 0 0 10 5 0 cm {im} Do Q");
            let page = render(&d, space, &content);
            assert_eq!(at(&page, 2), at(&page, 7), "{space:?} {fill}");
        }
    }
}

/// The linear value of the sRGB-encoded `v`.
fn decode(v: f64) -> f64 {
    if v <= 0.040_45 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// The sRGB encoding of the linear `l`.
#[expect(
    clippy::suboptimal_flops,
    reason = "the IEC 61966-2-1 formula as published"
)]
fn encode(l: f64) -> f64 {
    let l = l.clamp(0.0, 1.0);
    if l <= 0.003_130_8 {
        12.92 * l
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    }
}

/// The wide value, in `[0, 1]`, of the sRGB colour `rgb`: through the D50
/// XYZ of the sRGB primaries (IEC 61966-2-1, Bradford-adapted) and the
/// inverse of the ROMM RGB matrix (ISO 22028-2), solved here by Cramer's
/// rule, each row of the composite scaled to sum to 1.
#[expect(
    clippy::suboptimal_flops,
    reason = "Cramer's rule written as the cofactor expansion"
)]
fn wide_of(rgb: [f64; 3]) -> [f64; 3] {
    const SRGB: [[f64; 3]; 3] = [
        [0.436_065_7, 0.385_147_1, 0.143_066_4],
        [0.222_488_4, 0.716_873_2, 0.060_607_9],
        [0.013_916_0, 0.097_076_4, 0.714_096_1],
    ];
    const ROMM: [[f64; 3]; 3] = [
        [0.7977, 0.1352, 0.0313],
        [0.2880, 0.7119, 0.0001],
        [0.0, 0.0, 0.8249],
    ];
    let det = |m: [[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    // Column c of the composite solves ROMM · x = SRGB column c.
    let mut composite = [[0.0; 3]; 3];
    for c in 0..3 {
        for r in 0..3 {
            let mut m = ROMM;
            for (row, srgb) in m.iter_mut().zip(SRGB) {
                row[r] = srgb[c];
            }
            composite[r][c] = det(m) / det(ROMM);
        }
    }
    let lin = rgb.map(decode);
    composite.map(|row| {
        let sum: f64 = row.iter().sum();
        encode(row.iter().zip(lin).map(|(m, l)| m * l).sum::<f64>() / sum)
    })
}

/// Two saturated colours composited at constant alpha one half mix in the
/// working space: the result is the mean of their wide values, weighted by
/// the alpha of the graphics state, held as the byte 128.
#[test]
fn blending_happens_in_the_working_space() {
    let d = doc();
    let content = "1 0 0 rg 0 0 10 10 re f /Half gs 0 0 1 rg 0 0 10 10 re f";
    let page = render(&d, WorkingSpace::Wide, content);
    let (red, blue) = (wide_of([1.0, 0.0, 0.0]), wide_of([0.0, 0.0, 1.0]));
    let got = at(&page, 5);
    for c in 0..3 {
        let alpha: f64 = 128.0 / 255.0;
        let want = alpha.mul_add(blue[c] - red[c], red[c]) * 65_535.0;
        assert!(
            (f64::from(got[c]) - want).abs() <= 3.0,
            "channel {c}: {got:?} vs {want}"
        );
    }
    // The sRGB mix differs: the spaces' primaries differ.
    let srgb = render(&d, WorkingSpace::Srgb, content);
    assert_ne!(at(&srgb, 5), got);
}

/// A luminosity mask whose group draws coloured content gives the same
/// mask in either space, though the page has drawn the same image and
/// form, kept by the page's memos, in the wide space first.
#[test]
fn a_soft_mask_does_not_depend_on_the_working_space() {
    let d = doc();
    let content = "q 5 0 0 10 0 0 cm /Rgb Do Q q 1 0 0 1 5 0 cm /Fm Do Q \
                   q 5 0 0 10 0 0 cm /Rgb Do Q q 1 0 0 1 5 0 cm /Fm Do Q \
                   0 g 0 0 10 10 re f /Lum gs 1 g 0 0 10 10 re f";
    let wide = render(&d, WorkingSpace::Wide, content);
    let srgb = render(&d, WorkingSpace::Srgb, content);
    assert_eq!(wide, srgb);
    let mask = at(&wide, 2);
    assert!(
        mask[0] > 0 && mask[0] < 65_535 && mask[0] == mask[1],
        "{mask:?}"
    );
}
