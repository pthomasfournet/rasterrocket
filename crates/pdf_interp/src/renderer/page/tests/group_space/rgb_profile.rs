//! Transparency groups of an RGB space that is not sRGB: `CalRGB` and an
//! ICC profile composite in the components of their own space, a colour
//! painted in them converted in from the working space and their result
//! converted out.

use moxcms::{ColorProfile, Layout, TransformOptions};

use super::*;

/// The Adobe RGB group, isolated.
const ADOBE: &str = "/CS [/ICCBased 6 0 R] /I true";

/// A `CalRGB` space of the Adobe RGB primaries and white, gamma 2.2,
/// isolated.
const CAL_ADOBE: &str = "/CS [/CalRGB << /WhitePoint [0.9505 1 1.089] /Gamma [2.2 2.2 2.2] \
     /Matrix [0.5767 0.2974 0.0270 0.1856 0.6273 0.0707 0.1882 0.0753 0.9911] >>] /I true";

/// An orange in `Multiply` over a blue, both of `DeviceRGB`.
const ORANGE_ON_BLUE: &str = "0.2 0.6 0.9 rg 0 0 10 10 re f /Mul gs 0.9 0.5 0.1 rg 0 0 10 10 re f";

/// The sRGB value of the sRGB colours `backdrop` and `source` multiplied
/// in Adobe RGB, by the converter's floating-point transforms.
fn multiplied_in_adobe(backdrop: [f64; 3], source: [f64; 3]) -> [u16; 3] {
    let options = TransformOptions {
        rendering_intent: moxcms::RenderingIntent::RelativeColorimetric,
        prefer_fixed_point: false,
        ..TransformOptions::default()
    };
    let srgb = crate::WorkingSpace::Srgb.profile();
    let adobe = ColorProfile::new_adobe_rgb();
    let xform = |from: &ColorProfile, to: &ColorProfile, rgb: [f64; 3]| {
        let t = from
            .create_transform_f64(Layout::Rgb, to, Layout::Rgb, options)
            .expect("a transform");
        let mut out = [0.0; 3];
        t.transform(&rgb, &mut out).expect("converts");
        out
    };
    let (b, s) = (xform(&srgb, &adobe, backdrop), xform(&srgb, &adobe, source));
    let product = std::array::from_fn(|i| b[i] * s[i]);
    xform(&adobe, &srgb, product).map(|v| {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "an 8-bit level"
        )]
        let level = (v.clamp(0.0, 1.0) * 255.0).round() as u16;
        level
    })
}

/// The largest difference between the first pixel of `got` and `want`.
fn off(got: &[u16], want: &[u16]) -> u16 {
    got.iter()
        .zip(want)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0)
}

#[test]
fn an_icc_group_blends_in_its_profile() {
    let want = multiplied_in_adobe([0.2, 0.6, 0.9], [0.9, 0.5, 0.1]);
    let (canvas, diag) = render_page(
        &doc(ADOBE, ORANGE_ON_BLUE),
        &CanvasRequest::Rgb,
        SampleDepth::Eight,
        "/G Do",
    );
    assert!(!diag.approximated_group_spaces.any());
    let got = samples(&canvas);
    assert!(off(&got[..3], &want) <= 2, "{:?} vs {want:?}", &got[..3]);
    let in_srgb = grouped(
        "/I true",
        ORANGE_ON_BLUE,
        &CanvasRequest::Rgb,
        SampleDepth::Eight,
    );
    assert!(
        off(&got[..3], &in_srgb[..3]) > 20,
        "blending in sRGB differs"
    );
    let wide = grouped(
        ADOBE,
        ORANGE_ON_BLUE,
        &CanvasRequest::Rgb,
        SampleDepth::Sixteen,
    );
    let narrowed: Vec<u16> = wide[..3].iter().map(|&v| (v + 128) / 257).collect();
    assert!(
        off(&narrowed, &want) <= 2,
        "16 bits: {narrowed:?} vs {want:?}"
    );
}

#[test]
fn a_non_isolated_group_blends_with_its_backdrop_in_its_profile() {
    let want = multiplied_in_adobe([0.2, 0.6, 0.9], [0.9, 0.5, 0.1]);
    let (canvas, _) = render_page(
        &doc(
            "/CS [/ICCBased 6 0 R]",
            "/Mul gs 0.9 0.5 0.1 rg 0 0 10 10 re f",
        ),
        &CanvasRequest::Rgb,
        SampleDepth::Eight,
        "0.2 0.6 0.9 rg 0 0 10 10 re f /G Do",
    );
    let got = samples(&canvas);
    assert!(off(&got[..3], &want) <= 2, "{:?} vs {want:?}", &got[..3]);
}

#[test]
fn a_cal_rgb_group_blends_as_its_icc_twin() {
    let cal = grouped(
        CAL_ADOBE,
        ORANGE_ON_BLUE,
        &CanvasRequest::Rgb,
        SampleDepth::Eight,
    );
    let icc = grouped(
        ADOBE,
        ORANGE_ON_BLUE,
        &CanvasRequest::Rgb,
        SampleDepth::Eight,
    );
    assert!(off(&cal, &icc) <= 2, "{:?} vs {:?}", &cal[..3], &icc[..3]);
}

#[test]
fn a_group_on_an_ink_page_blends_as_on_an_rgb_page() {
    let rgb = grouped(
        ADOBE,
        ORANGE_ON_BLUE,
        &CanvasRequest::Rgb,
        SampleDepth::Eight,
    );
    let ink = grouped(ADOBE, ORANGE_ON_BLUE, &INK, SampleDepth::Eight);
    // The ink page reproduces a colour through the inverse of the
    // process-ink model, within four counts.
    assert!(off(&ink, &rgb) <= 4, "{:?} vs {:?}", &ink[..3], &rgb[..3]);
}

#[test]
fn a_colour_of_the_groups_space_keeps_its_value() {
    let content = "/Adobe cs 0.3 0.7 0.2 sc 0 0 10 10 re f";
    for depth in [SampleDepth::Eight, SampleDepth::Sixteen] {
        let got = grouped(ADOBE, content, &CanvasRequest::Rgb, depth);
        let want = direct(content, &CanvasRequest::Rgb, depth);
        let tolerance = if depth == SampleDepth::Eight { 1 } else { 64 };
        assert!(
            off(&got, &want) <= tolerance,
            "{depth:?}: {:?} vs {:?}",
            &got[..3],
            &want[..3]
        );
    }
}
