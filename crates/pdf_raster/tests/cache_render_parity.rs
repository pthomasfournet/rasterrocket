//! End-to-end render parity: a page rendered through the device-resident
//! image cache (`--backend cuda` with the `cache` feature) must be
//! byte-identical to the CPU render of the same page.
//!
//! The cached path decodes JPEGs once, keeps the pixels in VRAM, samples
//! them with the blit kernel, and composites the result onto the page in
//! content-stream order.  Everything downstream of the decode differs from
//! the CPU path, so the only meaningful check is the final page bitmap.
//!
//! Run with:
//!   `cargo test -p rasterrocket --features cache,gpu-validation --test cache_render_parity`

#![cfg(all(feature = "cache", feature = "gpu-validation"))]

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use common::{assert_identical, corpus_fixture, render, session, stream_object, with_page_tree};
use jpeg_encoder::{ColorType, Encoder};
use rasterrocket::{BackendPolicy, SessionConfig, open_session};

// ── fixture ───────────────────────────────────────────────────────────────────

/// Baseline JPEG of a smooth gradient with a sharp checker overlay, so a
/// one-source-pixel sampling shift produces both small and full-range
/// deltas.
fn jpeg(width: u16, height: u16, gray: bool) -> Vec<u8> {
    let channels = if gray { 1 } else { 3 };
    let mut pixels = Vec::with_capacity(usize::from(width) * usize::from(height) * channels);
    for y in 0..height {
        for x in 0..width {
            let checker = ((x / 7) + (y / 5)) % 2 == 0;
            let base = u8::try_from((u32::from(x) * 255 / u32::from(width)) & 0xff).expect("u8");
            let ramp = u8::try_from((u32::from(y) * 255 / u32::from(height)) & 0xff).expect("u8");
            if gray {
                pixels.push(if checker { base } else { 255 - ramp });
            } else {
                pixels.extend_from_slice(&[base, ramp, if checker { 230 } else { 30 }]);
            }
        }
    }
    let mut out = Vec::new();
    let encoder = Encoder::new(&mut out, 90);
    encoder
        .encode(
            &pixels,
            width,
            height,
            if gray {
                ColorType::Luma
            } else {
                ColorType::Rgb
            },
        )
        .expect("jpeg encode");
    out
}

/// A 300×240 pt page exercising every placement the sampler distinguishes:
///
/// - `/ImA` (RGB JPEG, 397×261) drawn axis-aligned at a fractional offset
///   and downscaled, then again x-flipped (a same-object cache hit);
/// - `/ImB` (gray JPEG, 50×80) drawn rotated and sheared;
/// - `/ImC` (the `/ImA` bytes under a second object with an `/SMask`),
///   which must stay on the CPU path on both backends;
/// - a filled rectangle and text painted *after* the images, overlapping
///   `/ImA`, so an out-of-order composite would show.
fn fixture_pdf() -> Vec<u8> {
    let rgb = jpeg(397, 261, false);
    let gray = jpeg(50, 80, true);
    // Soft mask for /ImC: opaque on the left half, a vertical ramp on the right.
    let smask: Vec<u8> = (0..261u32)
        .flat_map(|y| {
            (0..397u32).map(move |x| {
                if x < 198 {
                    255
                } else {
                    u8::try_from((y * 255 / 260) & 0xff).expect("u8")
                }
            })
        })
        .collect();
    let content = b"\
q 143.37 0 0 90.61 20.123 130.77 cm /ImA Do Q\n\
q 60.5 15.25 -10.75 75.5 200.33 140.17 cm /ImB Do Q\n\
q -120 0 0 60 290 20 cm /ImA Do Q\n\
q 100 0 0 63 30 30 cm /ImC Do Q\n\
0 0 1 rg 60 150 80 50 re f\n\
0.9 0.1 0.1 RG 4 w 10 120 m 290 230 l S\n\
BT /F1 18 Tf 30 200 Td (Parity) Tj ET\n"
        .to_vec();

    let objects = vec![
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 240] \
           /Resources << /XObject << /ImA 5 0 R /ImB 6 0 R /ImC 7 0 R >> \
                         /Font << /F1 9 0 R >> >> \
           /Contents 4 0 R >>"
            .to_vec(),
        stream_object("", &content),
        stream_object(
            "/Type /XObject /Subtype /Image /Width 397 /Height 261 \
             /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode",
            &rgb,
        ),
        stream_object(
            "/Type /XObject /Subtype /Image /Width 50 /Height 80 \
             /ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /DCTDecode",
            &gray,
        ),
        stream_object(
            "/Type /XObject /Subtype /Image /Width 397 /Height 261 \
             /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode \
             /SMask 8 0 R",
            &rgb,
        ),
        stream_object(
            "/Type /XObject /Subtype /Image /Width 397 /Height 261 \
             /ColorSpace /DeviceGray /BitsPerComponent 8",
            &smask,
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
    ];
    with_page_tree(&objects)
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[test]
fn synthetic_page_renders_identically_through_the_cache() {
    let pdf = fixture_pdf();
    let cpu = render(&session(pdf.clone(), BackendPolicy::CpuOnly), 1);
    assert!(
        cpu.2.iter().any(|&b| b != 0xff),
        "fixture rendered blank on the CPU path"
    );

    let session = session(pdf, BackendPolicy::ForceCuda);
    let cache = session
        .image_cache()
        .expect("ForceCuda session must carry an image cache");
    assert!(cache.is_empty(), "cache must start empty");

    // Cold pass: every eligible JPEG misses, decodes, and is inserted.
    let cold = render(&session, 1);
    assert_identical(&cpu, &cold, "cold cache");
    // /ImA and /ImB are cached; /ImC carries an /SMask and must not be.
    assert_eq!(
        cache.len(),
        2,
        "exactly the two unmasked JPEGs should be cached"
    );

    // Warm pass: both images hit by (doc, object) alias.
    let warm = render(&session, 1);
    assert_identical(&cpu, &warm, "warm cache");
    assert_eq!(cache.len(), 2, "a warm pass must not insert again");
}

/// Magnified draws, which the sampler interpolates: a gray JPEG drawn
/// rotated (both axes) and axis-aligned at about 3× across and 1:1 down
/// (columns only), and an RGB JPEG with `/Interpolate true` drawn at about
/// 10× (both axes).
#[test]
fn interpolated_draws_render_identically_through_the_cache() {
    for draw in [
        "q 60.5 15.25 -10.75 75.5 200.33 140.17 cm /ImB Do Q",
        "q 70 0 0 38.4 150.4 5 cm /ImB Do Q",
        "q 190.3 0 0 101.7 10.6 30.2 cm /ImD Do Q",
    ] {
        assert_cached_draw_identical(draw);
    }
}

/// Render `draw` (which may use `/ImB`, a 50×80 gray JPEG, and `/ImD`, a
/// 40×21 RGB JPEG with `/Interpolate true`) on both backends and assert the
/// pages are byte-identical and the image came from the cache.
fn assert_cached_draw_identical(draw: &str) {
    let content = draw.as_bytes();
    let objects = vec![
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 240] \
           /Resources << /XObject << /ImB 5 0 R /ImD 6 0 R >> >> /Contents 4 0 R >>"
            .to_vec(),
        stream_object("", content),
        stream_object(
            "/Type /XObject /Subtype /Image /Width 50 /Height 80 \
             /ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /DCTDecode",
            &jpeg(50, 80, true),
        ),
        stream_object(
            "/Type /XObject /Subtype /Image /Width 40 /Height 21 \
             /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode \
             /Interpolate true",
            &jpeg(40, 21, false),
        ),
    ];
    let pdf = with_page_tree(&objects);
    let cpu = render(&session(pdf.clone(), BackendPolicy::CpuOnly), 1);
    let session = session(pdf, BackendPolicy::ForceCuda);
    let gpu = render(&session, 1);
    assert_identical(&cpu, &gpu, draw);
    assert_eq!(
        session.image_cache().expect("image cache").len(),
        1,
        "{draw}: the image should be drawn from the cache"
    );
}

/// Render `pages` of a corpus fixture on both backends and assert each is
/// byte-identical, checking that the cache grows by exactly
/// `cached_per_page[i]` entries after page `pages[i]`.  Skipped when the
/// fixture is absent.
fn assert_corpus_pages_identical(fixture: &str, pages: &[u32], cached_per_page: &[usize]) {
    let Some(path) = corpus_fixture(fixture) else {
        return;
    };
    let cpu_session =
        open_session(&path, &SessionConfig::with_policy(BackendPolicy::CpuOnly)).expect("CPU");
    let cuda_session = open_session(&path, &SessionConfig::with_policy(BackendPolicy::ForceCuda))
        .expect("CUDA session (needs a CUDA device)");
    let cache = cuda_session.image_cache().expect("image cache");
    let mut expected_len = 0;
    for (&page, &cached) in pages.iter().zip(cached_per_page) {
        let cpu = render(&cpu_session, page);
        let gpu = render(&cuda_session, page);
        assert_identical(&cpu, &gpu, &format!("{fixture} page {page}"));
        expected_len += cached;
        assert_eq!(
            cache.len(),
            expected_len,
            "{fixture} page {page}: cache entry count"
        );
    }
}

/// The corpus pages the divergence was first measured on: full-page
/// `DeviceRGB` JPEG scans at a fractional offset, one cached entry each.
#[test]
fn corpus_scan_pages_render_identically_through_the_cache() {
    assert_corpus_pages_identical("corpus-07-journal-dct-heavy.pdf", &[1, 2], &[1, 1]);
}

/// Pages whose images must bypass the cache next to one that must not.
/// Pages 1–2 draw Flate-encoded `CalRGB` images carrying an `/SMask` over
/// vector fills and text — nothing is cacheable, so the cache stays empty
/// and the page must still match the CPU render exactly.  Page 3 is the
/// first DCT scan and inserts one entry.
#[test]
fn corpus_masked_flate_pages_render_identically_and_bypass_the_cache() {
    assert_corpus_pages_identical("corpus-08-scan-dct-1927.pdf", &[1, 2, 3], &[0, 0, 1]);
}
