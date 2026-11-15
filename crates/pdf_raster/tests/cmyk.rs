//! CMYK pages: `RenderParams::with_cmyk` and `RasterOptions::color`.  A
//! page's `DeviceCMYK` colours keep their values, its bands stack into the
//! whole page at either depth, overprint follows the session's switch, and
//! the byte budget counts the ink canvas.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use std::num::NonZeroU32;

use color::Sample as _;

use common::{corpus_fixture, render_one, stream_object, with_page_tree};
use rasterrocket::{
    BackendPolicy, BandBitmap, ColorModel, FinishedCanvas, PixelFormat, Pixels, RasterError,
    RasterOptions, RasterSession, RenderParams, SampleDepth, SessionConfig, open_session,
    open_session_from_bytes, render_page_bands, render_page_canvas, render_page_rgb_with,
};

/// A 10 pt square page: solid cyan, then a black overprinting fill over
/// its left half, then an RGB red square and a gray one on the right.
fn page() -> Vec<u8> {
    with_page_tree(&[
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 10 10] \
          /Resources << /ExtGState << /GS << /OP true /op true /OPM 1 >> >> >> \
          /Contents 4 0 R >>"
            .to_vec(),
        stream_object(
            "",
            b"1 0 0 0 k 0 0 10 10 re f /GS gs 0 0 0 1 k 0 0 5 10 re f \
              1 0 0 rg 6 0 2 2 re f 0.5 g 8 8 2 2 re f",
        ),
    ])
}

fn session(pdf: Vec<u8>, overprint: bool) -> RasterSession {
    let mut config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    config.overprint = overprint;
    open_session_from_bytes(pdf, &config).expect("open session")
}

/// The CMYK samples, at their own values in `u16`, rows packed, and the
/// width of a CMYK canvas or band.
fn samples(bitmap: BandBitmap<'_>) -> (u32, Vec<u16>) {
    fn packed<P: color::Pixel>(b: &raster::Bitmap<P>) -> Vec<u16>
    where
        P::Sample: Into<u16>,
    {
        (0..b.height)
            .flat_map(|y| b.row_samples(y)[..b.width as usize * 4].to_vec())
            .map(Into::into)
            .collect()
    }
    match bitmap {
        BandBitmap::Cmyk8(b) => (b.width, packed(b)),
        BandBitmap::Cmyk16(b) => (b.width, packed(b)),
        _ => panic!("a CMYK render yields CMYK bitmaps"),
    }
}

/// The CMYK of pixel `(x, y)` of the `width`-pixel-wide samples `page`.
fn at((width, page): &(u32, Vec<u16>), x: u32, y: u32) -> [u16; 4] {
    let i = ((y * width + x) * 4) as usize;
    page[i..i + 4].try_into().unwrap()
}

fn whole(s: &RasterSession, params: &RenderParams) -> (u32, Vec<u16>) {
    let canvas = render_page_canvas(s, 1, params).expect("render");
    samples(BandBitmap::of(&canvas))
}

#[test]
fn cmyk_pages_keep_device_cmyk_and_follow_the_overprint_switch() {
    let params = RenderParams::new(1.0).with_cmyk();
    let off = whole(&session(page(), false), &params);
    assert_eq!(at(&off, 2, 5), [0, 0, 0, 255], "off: black knocks out cyan");
    assert_eq!(at(&off, 7, 5), [255, 0, 0, 0]);
    assert_eq!(
        at(&off, 7, 9),
        [0, 255, 255, 0],
        "red is magenta and yellow"
    );
    assert_eq!(at(&off, 9, 1), [0, 0, 0, 127], "gray is black ink alone");
    let on = whole(&session(page(), true), &params);
    assert_eq!(at(&on, 2, 5), [255, 0, 0, 255], "on: cyan survives");
    let wide = whole(
        &session(page(), false),
        &params.with_depth(SampleDepth::Sixteen),
    );
    assert_eq!(at(&wide, 7, 5), [65_535, 0, 0, 0]);
    assert_eq!(at(&wide, 2, 5), [0, 0, 0, 65_535]);
}

#[test]
fn cmyk_bands_stack_into_the_whole_page() {
    let Some(path) = corpus_fixture("corpus-11-cmyk-swatches.pdf") else {
        return;
    };
    let config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    let s = open_session(&path, &config).expect("open");
    for depth in [SampleDepth::Eight, SampleDepth::Sixteen] {
        let params = RenderParams::new(1.0).with_cmyk().with_depth(depth);
        let want = whole(&s, &params);
        assert!(want.1.iter().any(|&v| v != 0), "the swatches paint");
        for rows in [1, 7, 1000] {
            let params = params
                .clone()
                .with_band_rows(NonZeroU32::new(rows).expect("rows > 0"));
            let (mut width, mut got) = (0, Vec::new());
            let _diagnostics = render_page_bands(&s, 1, &params, |band| {
                let (w, band) = samples(band.bitmap);
                width = w;
                got.extend(band);
                Ok::<(), std::convert::Infallible>(())
            })
            .expect("bands");
            assert_eq!((width, got), want, "{depth:?}, bands of {rows} rows");
        }
    }
}

/// The swatches of the fixture, in `DeviceCMYK`, one 60 pt square each
/// along the bottom of the page, read at their centres.
#[test]
fn the_swatch_fixture_keeps_its_values() {
    let Some(path) = corpus_fixture("corpus-11-cmyk-swatches.pdf") else {
        return;
    };
    let s = open_session(&path, &SessionConfig::with_policy(BackendPolicy::CpuOnly)).expect("open");
    let page = whole(&s, &RenderParams::new(1.0).with_cmyk());
    let height = u32::try_from(page.1.len() / 4).unwrap() / page.0;
    let swatches: [[f64; 4]; 6] = [
        [0.87, 0.01, 0.0, 0.0],
        [1.0, 0.46, 0.05, 0.18],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.5, 0.5, 0.5, 0.0],
    ];
    for (i, swatch) in swatches.into_iter().enumerate() {
        let x = u32::try_from(i).unwrap() * 60 + 30;
        let want = swatch.map(|v| u16::from(u8::from_unit(v)));
        assert_eq!(at(&page, x, height - 30), want, "swatch {i}");
    }
}

#[test]
fn raster_options_yield_cmyk_pages() {
    for (depth, format) in [
        (SampleDepth::Eight, PixelFormat::Cmyk8),
        (SampleDepth::Sixteen, PixelFormat::Cmyk16),
    ] {
        let mut opts = RasterOptions::new(72.0);
        opts.color = ColorModel::Cmyk;
        opts.depth = depth;
        let page = render_one(page(), &opts).expect("render");
        assert_eq!(page.format, format);
        assert_eq!(page.format.samples_per_pixel(), 4);
        assert_eq!(page.pixels.len(), 10 * 10 * 4);
        let first = match &page.pixels {
            Pixels::U8(v) => v[..4].iter().map(|&v| u16::from(v)).collect::<Vec<_>>(),
            Pixels::U16(v) => v[..4].to_vec(),
        };
        let solid = u16::MAX >> if depth == SampleDepth::Eight { 8 } else { 0 };
        assert_eq!(
            first,
            [0, 0, 0, solid],
            "{depth:?}: the top-left pixel is black ink"
        );
    }
}

#[test]
fn cmyk_pages_are_refused_where_they_have_no_meaning() {
    let invalid = |r: Result<rasterrocket::RenderedPage, RasterError>, what: &str| {
        assert!(
            matches!(r, Err(RasterError::InvalidOptions(_))),
            "{what}: {r:?}"
        );
    };
    let mut opts = RasterOptions::new(72.0);
    opts.color = ColorModel::Cmyk;
    opts.deskew = true;
    invalid(render_one(page(), &opts), "deskew");
    let mut opts = RasterOptions::new(72.0);
    opts.color = ColorModel::Cmyk;
    opts.transparent_background = true;
    invalid(render_one(page(), &opts), "a transparent backdrop");

    let s = session(page(), false);
    let params = RenderParams::new(1.0).with_cmyk();
    assert!(matches!(
        render_page_rgb_with(&s, 1, &params),
        Err(RasterError::InvalidOptions(_))
    ));
    assert!(matches!(
        render_page_canvas(&s, 1, &params.clone().with_transparent_background()),
        Err(RasterError::InvalidOptions(_))
    ));
    assert!(
        matches!(
            render_page_canvas(&s, 1, &params),
            Ok(FinishedCanvas::Cmyk8(_))
        ),
        "the page itself renders"
    );
}

/// The byte budget counts the eight samples of the ink canvas a CMYK page
/// is painted on: a budget that holds the RGB page refuses the CMYK one.
#[test]
fn the_byte_budget_counts_the_ink_canvas() {
    let mut config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    config.max_page_bytes = 10 * 10 * 3;
    let s = open_session_from_bytes(page(), &config).expect("open");
    assert!(render_page_canvas(&s, 1, &RenderParams::new(1.0)).is_ok());
    assert!(matches!(
        render_page_canvas(&s, 1, &RenderParams::new(1.0).with_cmyk()),
        Err(RasterError::PageTooLarge { bytes: 800, .. })
    ));
    // A page simulating overprint is painted on ink too.
    config.overprint = true;
    let s = open_session_from_bytes(page(), &config).expect("open");
    assert!(matches!(
        render_page_canvas(&s, 1, &RenderParams::new(1.0)),
        Err(RasterError::PageTooLarge { bytes: 800, .. })
    ));
}

/// A session of [`page`] converting to the test profile `name`.
fn profiled(name: &str) -> RasterSession {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/icc")
        .join(name);
    let mut config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    config.output_profile = Some(std::sync::Arc::new(
        rasterrocket::OutputProfile::from_bytes(std::fs::read(path).expect("read profile"))
            .expect("an output profile"),
    ));
    open_session_from_bytes(page(), &config).expect("open session")
}

/// Under a CMYK output profile the page's `DeviceCMYK` colours keep
/// their values and its other colours separate through the profile: the
/// gray to nearly black ink alone, as the profile replaces gray components.
#[test]
fn a_cmyk_output_profile_separates_what_is_not_cmyk() {
    let session = profiled("printer-cmyk-v4.icc");
    let params = RenderParams::new(1.0).with_cmyk();
    let canvas = render_page_canvas(&session, 1, &params).expect("render");
    let page = samples(BandBitmap::of(&canvas));
    assert_eq!(at(&page, 2, 5), [0, 0, 0, 255]);
    assert_eq!(at(&page, 7, 5), [255, 0, 0, 0]);
    let [c, m, y, k] = at(&page, 9, 1);
    assert!(k > 100 && c.max(m).max(y) < 12, "gray: {:?}", [c, m, y, k]);
    let (_, diag) =
        rasterrocket::render_page_canvas_with_diagnostics(&session, 1, &params).expect("render");
    assert_eq!(
        diag.output_intent,
        Some(rasterrocket::RenderingIntent::RelativeColorimetric)
    );
    assert!(
        matches!(
            render_page_canvas(&session, 1, &RenderParams::new(1.0)),
            Err(RasterError::InvalidOptions(_))
        ),
        "an sRGB page under a CMYK profile"
    );
    let rgb = profiled("printer-rgb-v4.icc");
    assert!(
        matches!(
            render_page_canvas(&rgb, 1, &params),
            Err(RasterError::InvalidOptions(_))
        ),
        "a CMYK page under an RGB profile"
    );
}
