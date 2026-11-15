//! Banded rendering: the bands of a page, stacked, are the whole-page
//! render byte for byte, whatever the band height and wherever its edges
//! cross the content.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use std::num::NonZeroU32;

use common::seams::{GEOMETRIES, seam_page};
use common::{SCALE, corpus_fixture, one_page, stream_object};
use rasterrocket::{
    BackendPolicy, BandError, RasterError, RasterSession, RenderParams, RenderRegion,
    SessionConfig, open_session, open_session_from_bytes, render_page_bands,
    render_page_rgb_with_diagnostics,
};

/// The pixels of a render, alpha plane after colour: `(width, height,
/// bytes)`.
type Pixels = (u32, u32, Vec<u8>);

fn pixels_of(bitmap: &raster::Bitmap<color::Rgb8>, out: &mut Vec<u8>, alpha: &mut Vec<u8>) {
    for row in 0..bitmap.height {
        out.extend_from_slice(&bitmap.row_bytes(row)[..bitmap.width as usize * 3]);
        if let Some(a) = bitmap.alpha_row(row) {
            alpha.extend_from_slice(a);
        }
    }
}

/// The whole-page render of page 1 under `params`.
fn whole(
    session: &RasterSession,
    params: &RenderParams,
) -> (Pixels, rasterrocket::PageDiagnostics) {
    let (bmp, diag) =
        render_page_rgb_with_diagnostics(session, 1, params).expect("whole-page render");
    let (mut out, mut alpha) = (Vec::new(), Vec::new());
    pixels_of(&bmp, &mut out, &mut alpha);
    out.extend(alpha);
    ((bmp.width, bmp.height, out), diag)
}

/// The bands of page 1 under `params` in bands of `rows`, stacked, with
/// the diagnostics; asserts each band's place and width.
fn banded(
    session: &RasterSession,
    params: &RenderParams,
    rows: u32,
) -> (Pixels, rasterrocket::PageDiagnostics) {
    let params = params
        .clone()
        .with_band_rows(NonZeroU32::new(rows).expect("rows > 0"));
    let (mut out, mut alpha, mut next_y, mut width) = (Vec::new(), Vec::new(), 0, 0);
    let diag = render_page_bands(session, 1, &params, |band| {
        assert_eq!(band.y, next_y, "bands arrive in order");
        let rasterrocket::BandBitmap::Rgb8(bitmap) = band.bitmap else {
            panic!("an 8-bit render yields 8-bit bands");
        };
        assert!(bitmap.height <= rows);
        width = bitmap.width;
        next_y += bitmap.height;
        pixels_of(bitmap, &mut out, &mut alpha);
        Ok::<(), std::convert::Infallible>(())
    })
    .unwrap_or_else(|e| panic!("bands of {rows} rows: {e}"));
    out.extend(alpha);
    ((width, next_y, out), diag)
}

fn session_of(pdf: Vec<u8>, config: &SessionConfig) -> RasterSession {
    open_session_from_bytes(pdf, config).expect("open")
}

/// Assert that bands of each height in `rows` stack into the whole-page
/// render of `params`, with the same reported failures.
fn assert_bands_are_the_page(session: &RasterSession, params: &RenderParams, what: &str) {
    let (full, full_diag) = whole(session, params);
    let h = full.1;
    for rows in [1, 7, 64, h - 1, h, h + 5] {
        let (stacked, diag) = banded(session, params, rows.max(1));
        common::assert_identical(&full, &stacked, &format!("{what}, bands of {rows} rows"));
        assert_eq!(diag.image_failures, full_diag.image_failures, "{what}");
        assert_eq!(diag.content_failures, full_diag.content_failures, "{what}");
        assert_eq!(diag.missing_fonts, full_diag.missing_fonts, "{what}");
        assert_eq!(diag.has_images, full_diag.has_images, "{what}");
        assert_eq!(diag.has_vector_text, full_diag.has_vector_text, "{what}");
    }
}

#[test]
fn bands_stack_into_the_whole_page() {
    for extra in GEOMETRIES {
        let s = session_of(
            seam_page(extra),
            &SessionConfig::with_policy(BackendPolicy::CpuOnly),
        );
        assert_bands_are_the_page(&s, &RenderParams::new(SCALE), extra);
    }
}

#[test]
fn bands_over_a_transparent_backdrop_keep_the_alpha_plane() {
    let s = session_of(
        seam_page(""),
        &SessionConfig::with_policy(BackendPolicy::CpuOnly),
    );
    let params = RenderParams::new(SCALE).with_transparent_background();
    assert_bands_are_the_page(&s, &params, "transparent");
}

#[test]
fn bands_at_anisotropic_scales_and_of_a_region() {
    let s = session_of(
        seam_page(""),
        &SessionConfig::with_policy(BackendPolicy::CpuOnly),
    );
    assert_bands_are_the_page(
        &s,
        &RenderParams::new_xy(SCALE * 1.3, SCALE * 0.7),
        "anisotropic",
    );
    let region = RenderRegion::new(31, 17, 250, 150);
    assert_bands_are_the_page(&s, &RenderParams::new(SCALE).with_region(region), "region");
}

#[test]
fn bands_without_antialiasing() {
    let mut config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    config.vector_antialias = false;
    config.text_antialias = false;
    let s = session_of(seam_page(""), &config);
    assert_bands_are_the_page(&s, &RenderParams::new(SCALE), "aliased");
}

/// Overprint on an ink canvas: a spot colour slotted in the first band
/// keeps its slot in the next.
#[test]
fn bands_of_an_overprinted_page() {
    let content = b"0 0 1 0 k 10 10 100 60 re f /OPG gs /Sp cs 1 scn 50 30 120 60 re f \
        /Sp2 cs 0.7 scn 20 5 60 90 re f";
    let pdf = one_page(
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] \
         /Resources << /ExtGState << /OPG << /OP true /op true /OPM 1 >> >> \
         /ColorSpace << /Sp [/Separation /Spot /DeviceCMYK \
             << /FunctionType 2 /Domain [0 1] /C0 [0 0 0 0] /C1 [0 1 0 0] /N 1 >>] \
         /Sp2 [/Separation /Other /DeviceCMYK \
             << /FunctionType 2 /Domain [0 1] /C0 [0 0 0 0] /C1 [0.5 0 1 0] /N 1 >>] >> >> \
         /Contents 4 0 R >>",
        vec![stream_object("", content)],
    );
    let mut config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    config.overprint = true;
    let s = session_of(pdf, &config);
    assert_bands_are_the_page(&s, &RenderParams::new(SCALE), "overprint");
}

#[test]
fn bands_of_a_page_with_annotations() {
    let pdf = one_page(
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Annots [5 0 R] \
         /Contents 4 0 R >>",
        vec![
            stream_object("", b"0.9 0.9 0.5 rg 0 0 200 100 re f"),
            b"<< /Type /Annot /Subtype /Square /Rect [20 20 120 80] /AP << /N 6 0 R >> >>".to_vec(),
            stream_object(
                "/Type /XObject /Subtype /Form /BBox [0 0 100 60]",
                b"0 0 1 RG 3 w 2 2 96 56 re S",
            ),
        ],
    );
    let s = session_of(pdf, &SessionConfig::with_policy(BackendPolicy::CpuOnly));
    assert_bands_are_the_page(&s, &RenderParams::new(SCALE), "annotations");
}

/// The corpus fixtures, by file name.
const CORPUS: [&str; 10] = [
    "corpus-01-native-text-small.pdf",
    "corpus-02-native-vector-text.pdf",
    "corpus-03-native-text-dense.pdf",
    "corpus-04-ebook-mixed.pdf",
    "corpus-05-academic-book.pdf",
    "corpus-06-modern-layout-dct.pdf",
    "corpus-07-journal-dct-heavy.pdf",
    "corpus-08-scan-dct-1927.pdf",
    "corpus-09-scan-dct-1836.pdf",
    "corpus-10-scan-jbig2-jpx.pdf",
];

#[test]
fn corpus_pages_band_exactly() {
    let params = RenderParams::new(50.0 / 72.0);
    for name in CORPUS {
        let Some(path) = corpus_fixture(name) else {
            continue;
        };
        let s = open_session(&path, &SessionConfig::with_policy(BackendPolicy::CpuOnly))
            .expect("open corpus fixture");
        let (full, _) = whole(&s, &params);
        let (stacked, _) = banded(&s, &params, 64);
        common::assert_identical(&full, &stacked, name);
    }
}

#[test]
fn a_sink_error_stops_the_bands() {
    let s = session_of(
        seam_page(""),
        &SessionConfig::with_policy(BackendPolicy::CpuOnly),
    );
    let params = RenderParams::new(SCALE).with_band_rows(NonZeroU32::new(50).unwrap());
    let mut calls = 0;
    let err = render_page_bands(&s, 1, &params, |_| {
        calls += 1;
        if calls == 2 { Err("disk full") } else { Ok(()) }
    })
    .expect_err("the sink failed");
    assert!(matches!(err, BandError::Sink("disk full")), "{err:?}");
    assert_eq!(calls, 2, "no band after the failing one");
}

/// The page budget bounds the page the bands make up, not a band.
#[test]
fn the_budget_bounds_the_banded_page() {
    let probe = session_of(
        seam_page(""),
        &SessionConfig::with_policy(BackendPolicy::CpuOnly),
    );
    let (w, h) = probe
        .page_geometry(1)
        .expect("geometry")
        .size_px(SCALE, SCALE);
    let mut config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    config.max_page_bytes = u64::from(w) * u64::from(h) * 3 - 1;
    let s = session_of(seam_page(""), &config);
    let params = RenderParams::new(SCALE).with_band_rows(NonZeroU32::new(8).unwrap());
    let err = render_page_bands(&s, 1, &params, |_| Ok::<(), std::convert::Infallible>(()))
        .expect_err("over the budget");
    assert!(
        matches!(err, BandError::Raster(RasterError::PageTooLarge { .. })),
        "{err:?}"
    );
}
