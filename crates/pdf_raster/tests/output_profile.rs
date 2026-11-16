//! Pages converted to an output profile: in bands as whole, at both
//! depths, with and without alpha; the intent the page converts under;
//! the refusals of what the session cannot convert.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use std::fmt::Write as _;
use std::num::NonZeroU32;
use std::sync::Arc;

use common::gradients::gradient_page;
use common::{one_page, stream_object};
use rasterrocket::{
    BackendPolicy, BandBitmap, FinishedCanvas, OutputProfile, PreserveBlack, RasterError,
    RasterSession, RenderParams, RenderingIntent, SampleDepth, SessionConfig,
    open_session_from_bytes, render_page_bands, render_page_canvas,
    render_page_canvas_with_diagnostics, render_page_rgb_with,
};

fn profile(name: &str) -> Arc<OutputProfile> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/icc")
        .join(name);
    Arc::new(OutputProfile::from_bytes(std::fs::read(path).unwrap()).unwrap())
}

fn session(pdf: Vec<u8>, output: Option<Arc<OutputProfile>>) -> RasterSession {
    let mut config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    config.output_profile = output;
    open_session_from_bytes(pdf, &config).expect("open session")
}

/// The colour and the alpha samples of `canvas`, each as their values.
fn planes(canvas: BandBitmap<'_>) -> (Vec<u32>, Vec<u32>) {
    fn of<P: color::Pixel>(b: &raster::Bitmap<P>) -> (Vec<u32>, Vec<u32>) {
        let live = b.width as usize * P::NCOMPS;
        let wide = |s: &P::Sample| u32::try_from(color::Sample::signed(*s)).unwrap();
        let colour = (0..b.height)
            .flat_map(|y| {
                b.row_samples(y)[..live]
                    .iter()
                    .map(wide)
                    .collect::<Vec<_>>()
            })
            .collect();
        let alpha = b
            .alpha_plane()
            .map_or_else(Vec::new, |a| a.iter().map(wide).collect());
        (colour, alpha)
    }
    match canvas {
        BandBitmap::Rgb8(b) => of(b),
        BandBitmap::Rgb16(b) => of(b),
        BandBitmap::Gray8(b) => of(b),
        BandBitmap::Gray16(b) => of(b),
        BandBitmap::Cmyk8(b) => of(b),
        BandBitmap::Cmyk16(b) => of(b),
    }
}

fn whole_planes(canvas: &FinishedCanvas) -> (Vec<u32>, Vec<u32>) {
    planes(BandBitmap::of(canvas))
}

/// The bands of 7 rows of page 1, their colour and alpha samples stacked.
fn banded_planes(s: &RasterSession, params: &RenderParams) -> (Vec<u32>, Vec<u32>) {
    let params = params.clone().with_band_rows(NonZeroU32::new(7).unwrap());
    let (mut colour, mut alpha) = (Vec::new(), Vec::new());
    render_page_bands(s, 1, &params, |band| {
        let (c, a) = planes(band.bitmap);
        colour.extend(c);
        alpha.extend(a);
        Ok::<(), std::convert::Infallible>(())
    })
    .map(drop)
    .expect("bands render");
    (colour, alpha)
}

#[test]
fn bands_converted_to_an_output_profile_are_the_converted_page() {
    for (name, gray) in [("printer-rgb-v4.icc", false), ("printer-gray-v4.icc", true)] {
        let s = session(gradient_page(), Some(profile(name)));
        for depth in [SampleDepth::Eight, SampleDepth::Sixteen] {
            for transparent in [false, true] {
                if gray && transparent {
                    continue;
                }
                let mut params = RenderParams::new(0.25)
                    .with_depth(depth)
                    .with_intent(RenderingIntent::Perceptual);
                if transparent {
                    params = params.with_transparent_background();
                }
                let whole = render_page_canvas(&s, 1, &params).expect("renders");
                assert_eq!(
                    matches!(whole, FinishedCanvas::Gray8(_) | FinishedCanvas::Gray16(_)),
                    gray,
                    "{name}: the profile's colour space"
                );
                assert!(
                    whole_planes(&whole) == banded_planes(&s, &params),
                    "{name} {depth:?} transparent {transparent}"
                );
            }
        }
    }
}

#[test]
fn a_profile_changes_the_pixels_and_none_leaves_them() {
    let plain = session(gradient_page(), None);
    let managed = session(gradient_page(), Some(profile("printer-rgb-v4.icc")));
    let params = RenderParams::new(0.25);
    let a = whole_planes(&render_page_canvas(&plain, 1, &params).unwrap()).0;
    let b = whole_planes(&render_page_canvas(&managed, 1, &params).unwrap()).0;
    assert_eq!(a.len(), b.len());
    assert_ne!(a, b, "the printer profile converts the page");
}

/// A page of two fills, under `intents` (`ri` operands) in turn.
fn intent_page(intents: &[&str]) -> Vec<u8> {
    let mut content = String::new();
    for (i, ri) in intents.iter().enumerate() {
        let x = i * 10;
        write!(content, "/{ri} ri 0.2 0.5 0.8 rg {x} 0 10 10 re f ").expect("a String takes it");
    }
    one_page(
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 20 10] /Contents 4 0 R >>",
        vec![stream_object("", content.as_bytes())],
    )
}

fn output_intent(pdf: Vec<u8>, params: &RenderParams) -> Option<RenderingIntent> {
    let s = session(pdf, Some(profile("printer-rgb-v4.icc")));
    let (_, diag) = render_page_canvas_with_diagnostics(&s, 1, params).unwrap();
    diag.output_intent
}

#[test]
fn a_page_converts_under_its_one_intent_else_relative_colorimetry() {
    use RenderingIntent::{Perceptual, RelativeColorimetric, Saturation};
    let params = RenderParams::new(1.0);
    assert_eq!(
        output_intent(intent_page(&["Perceptual", "Perceptual"]), &params),
        Some(Perceptual)
    );
    assert_eq!(
        output_intent(intent_page(&["Perceptual", "Saturation"]), &params),
        Some(RelativeColorimetric),
        "mixed intents"
    );
    assert_eq!(
        output_intent(
            intent_page(&["Perceptual", "Saturation"]),
            &params.clone().with_intent(Saturation)
        ),
        Some(Saturation)
    );
    let s = session(intent_page(&["Perceptual"]), None);
    let (_, diag) = render_page_canvas_with_diagnostics(&s, 1, &params).unwrap();
    assert_eq!(diag.output_intent, None, "no profile, no conversion");
}

#[test]
fn what_the_session_cannot_convert_is_refused() {
    let invalid =
        |r: Result<FinishedCanvas, RasterError>| matches!(r, Err(RasterError::InvalidOptions(_)));
    let plain = session(gradient_page(), None);
    let params = RenderParams::new(0.25);
    assert!(invalid(render_page_canvas(
        &plain,
        1,
        &params.clone().with_intent(RenderingIntent::Perceptual)
    )));
    assert!(invalid(render_page_canvas(
        &plain,
        1,
        &params.clone().with_black_point_compensation()
    )));
    let managed = session(gradient_page(), Some(profile("printer-rgb-v4.icc")));
    assert!(invalid(render_page_canvas(
        &managed,
        1,
        &params
            .clone()
            .with_intent(RenderingIntent::AbsoluteColorimetric)
            .with_black_point_compensation()
    )));
    let gray = session(gradient_page(), Some(profile("printer-gray-v4.icc")));
    assert!(matches!(
        render_page_rgb_with(&gray, 1, &params),
        Err(RasterError::InvalidOptions(_))
    ));
    // Black preservation needs CMYK ink converted to a CMYK profile's.
    let k_only = params.with_preserve_black(PreserveBlack::KOnly);
    assert!(invalid(render_page_canvas(&managed, 1, &k_only)));
    let press = session(gradient_page(), Some(profile("printer-cmyk-v4.icc")));
    assert!(invalid(render_page_canvas(&press, 1, &k_only)));
    let cmyk = k_only.with_cmyk();
    assert!(render_page_canvas(&press, 1, &cmyk).is_ok());
    assert!(invalid(render_page_canvas(
        &press,
        1,
        &cmyk.with_intent(RenderingIntent::AbsoluteColorimetric)
    )));
}
