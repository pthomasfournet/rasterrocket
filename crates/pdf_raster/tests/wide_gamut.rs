//! The wide working space: colours outside sRGB kept up to the output
//! profile, opaque sRGB content through it as through sRGB, one session's
//! decodes kept apart by working space, and the refusals of the wide space
//! without an output profile or at 8 bits.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use std::fmt::Write as _;
use std::sync::Arc;

use common::icc::adobe_rgb_profile;
use common::{assemble_pdf, one_page, stream_object};
use rasterrocket::{
    BackendPolicy, FinishedCanvas, OutputProfile, RasterError, RenderParams, SampleDepth,
    SessionConfig, WorkingSpace, open_session_from_bytes, render_page_canvas,
};

/// The 16-bit RGB samples of page `page` of `pdf` rendered at 1 pixel a
/// point, 16 bits, converted to `output` from `space` (`None`: the
/// session's default).
fn render(
    pdf: &[u8],
    page: u32,
    output: &Arc<OutputProfile>,
    space: Option<WorkingSpace>,
    overprint: bool,
) -> Vec<u16> {
    let mut config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    config.output_profile = Some(Arc::clone(output));
    config.working_space = space;
    config.overprint = overprint;
    let session = open_session_from_bytes(pdf.to_vec(), &config).expect("opens");
    samples(&session, page)
}

/// The 16-bit RGB samples of page `page` of `session` at 16 bits.
fn samples(session: &rasterrocket::RasterSession, page: u32) -> Vec<u16> {
    let params = RenderParams::new(1.0).with_depth(SampleDepth::Sixteen);
    match render_page_canvas(session, page, &params).expect("renders") {
        FinishedCanvas::Rgb16(b) => b.data().to_vec(),
        other => panic!("a 16-bit RGB page, not {other:?}"),
    }
}

/// The saturated Adobe RGB colours outside sRGB the tests paint.
const SATURATED: [[f64; 3]; 4] = [
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 1.0],
    [0.2, 0.8, 0.0],
    [0.0, 0.6, 0.4],
];

/// A page one point high holding each colour of [`SATURATED`] twice, as a
/// fill in `[/ICCBased Adobe]` and as one 8-bit pixel of an image in it.
fn adobe_page() -> Vec<u8> {
    let n = SATURATED.len();
    let mut content = String::from("/A cs\n");
    let mut pixels = Vec::new();
    for (x, c) in SATURATED.iter().enumerate() {
        let _ = writeln!(content, "{} {} {} sc {x} 0 1 1 re f", c[0], c[1], c[2]);
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a component in [0, 1] scaled to a byte"
        )]
        pixels.extend(c.map(|v| (v * 255.0).round() as u8));
    }
    let _ = write!(content, "q {n} 0 0 1 {n} 0 cm /Im Do Q");
    let page = format!(
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {} 1] \
         /Resources << /ColorSpace << /A [/ICCBased 4 0 R] >> /XObject << /Im 6 0 R >> >> \
         /Contents 5 0 R >>",
        2 * n
    );
    one_page(
        &page,
        vec![
            stream_object("/N 3", &adobe_rgb_profile()),
            stream_object("", content.as_bytes()),
            stream_object(
                &format!(
                    "/Type /XObject /Subtype /Image /Width {n} /Height 1 \
                     /ColorSpace [/ICCBased 4 0 R] /BitsPerComponent 8"
                ),
                &pixels,
            ),
        ],
    )
}

/// The linear light of the Adobe RGB (1998) device value `v` in `[0, 1]`.
fn adobe_linear(v: f64) -> f64 {
    v.powf(563.0 / 256.0)
}

/// The largest difference in linear light, relative to white, between each
/// patch of `page` and its colour of [`SATURATED`] (repeated: fills, then
/// image pixels), both Adobe RGB device values.  Linear light measures the
/// colour: the pure power law of Adobe RGB magnifies a difference near 0
/// without bound.
fn worst_error(page: &[u16]) -> f64 {
    page.as_chunks::<3>()
        .0
        .iter()
        .zip(SATURATED.iter().chain(&SATURATED))
        .flat_map(|(got, want)| {
            got.iter()
                .zip(want)
                .map(|(&g, &w)| (adobe_linear(f64::from(g) / 65_535.0) - adobe_linear(w)).abs())
        })
        .fold(0.0, f64::max)
}

/// Adobe RGB colours outside sRGB, painted and converted to an Adobe RGB
/// output profile at 16 bits, come back as they were through the wide
/// working space; through sRGB they are clipped.
#[test]
fn colours_outside_srgb_reach_a_wide_output_profile() {
    let pdf = adobe_page();
    let output = Arc::new(OutputProfile::from_bytes(adobe_rgb_profile()).unwrap());
    let wide = worst_error(&render(&pdf, 1, &output, None, false));
    let srgb = worst_error(&render(&pdf, 1, &output, Some(WorkingSpace::Srgb), false));
    eprintln!("worst error in linear light: wide {wide:.6}, sRGB {srgb:.6}");
    assert!(wide <= ROUND_TRIP_BOUND, "wide: {wide}");
    assert!(srgb > 0.05, "sRGB: {srgb}");
}

/// The largest difference in linear light, relative to white, accepted
/// between a colour and its round trip through the 16-bit wide working
/// space to an output profile: 26 levels of a 16-bit sRGB sample at white
/// (the round trip's measured worst), about ΔE00 0.1.  Measured: 3.1e-4
/// between the two working spaces, 4.4e-4 for the Adobe RGB round trip.
const ROUND_TRIP_BOUND: f64 = 1e-3;

/// A page one point high of a 9-level cube of `DeviceRGB` patches.
fn cube_page() -> Vec<u8> {
    let mut content = String::new();
    let mut x = 0;
    for r in 0..9 {
        for g in 0..9 {
            for b in 0..9 {
                let v = |i: i32| f64::from(i) / 8.0;
                let _ = writeln!(content, "{} {} {} rg {x} 0 1 1 re f", v(r), v(g), v(b));
                x += 1;
            }
        }
    }
    let page = format!("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {x} 1] /Contents 4 0 R >>");
    one_page(&page, vec![stream_object("", content.as_bytes())])
}

/// The RGB test printer profile.
fn printer() -> Arc<OutputProfile> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/icc/printer-rgb-v4.icc"
    );
    Arc::new(OutputProfile::from_bytes(std::fs::read(path).unwrap()).unwrap())
}

/// Opaque colours inside sRGB convert through the wide working space as
/// through sRGB, within the quantisation of its 16-bit samples.
#[test]
fn opaque_srgb_colours_print_alike_in_either_space() {
    let pdf = cube_page();
    let output = Arc::new(OutputProfile::from_bytes(adobe_rgb_profile()).unwrap());
    let wide = render(&pdf, 1, &output, None, false);
    let srgb = render(&pdf, 1, &output, Some(WorkingSpace::Srgb), false);
    let linear = |v: &u16| adobe_linear(f64::from(*v) / 65_535.0);
    let worst = wide
        .iter()
        .zip(&srgb)
        .map(|(a, b)| (linear(a) - linear(b)).abs())
        .fold(0.0, f64::max);
    eprintln!("worst difference in linear light: {worst:.6}");
    assert!(worst <= ROUND_TRIP_BOUND, "{worst}");
}

/// A document of two pages drawing one image; the first sets overprint.
fn shared_image_pdf() -> Vec<u8> {
    let page = |extra: &str| {
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 2 1] /Resources << \
             /XObject << /Im 5 0 R >> /ExtGState << /Op << /OP true /op true >> >> >> \
             /Contents {extra} >>"
        )
        .into_bytes()
    };
    assemble_pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>".to_vec(),
        page("6 0 R"),
        page("7 0 R"),
        stream_object(
            "/Type /XObject /Subtype /Image /Width 2 /Height 1 /ColorSpace /DeviceRGB \
             /BitsPerComponent 8",
            &[0, 200, 60, 230, 40, 10],
        ),
        stream_object("", b"/Op gs q 2 0 0 1 0 0 cm /Im Do Q"),
        stream_object("", b"q 2 0 0 1 0 0 cm /Im Do Q"),
    ])
}

/// A page painted as an overprint preview, in sRGB, and a page of the wide
/// space drawing the same image in one session each get their own decode:
/// the wide page renders as it does alone.
#[test]
fn an_image_shared_by_an_srgb_and_a_wide_page_is_decoded_for_each() {
    let pdf = shared_image_pdf();
    let output = printer();
    let mut config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    config.output_profile = Some(Arc::clone(&output));
    config.overprint = true;
    let session = open_session_from_bytes(pdf.clone(), &config).expect("opens");
    let _preview = samples(&session, 1);
    let shared = samples(&session, 2);
    let alone = render(&pdf, 2, &output, None, true);
    assert_eq!(shared, alone);
}

#[test]
fn the_wide_space_needs_an_output_profile() {
    let mut config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    config.working_space = Some(WorkingSpace::Wide);
    let Err(err) = open_session_from_bytes(cube_page(), &config) else {
        panic!("the session opens");
    };
    assert!(matches!(err, RasterError::InvalidOptions(_)), "{err}");
}

#[test]
fn the_wide_space_needs_16_bits() {
    let mut config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    config.output_profile = Some(printer());
    config.working_space = Some(WorkingSpace::Wide);
    let session = open_session_from_bytes(cube_page(), &config).expect("opens");
    let err = render_page_canvas(&session, 1, &RenderParams::new(1.0)).unwrap_err();
    assert!(matches!(err, RasterError::InvalidOptions(_)), "{err}");
}
