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

use std::fmt::Write as _;
use std::path::PathBuf;

use jpeg_encoder::{ColorType, Encoder};
use rasterrocket::{
    BackendPolicy, RasterSession, SessionConfig, open_session, open_session_from_bytes,
    render_page_rgb,
};

/// 150 DPI: the resolution the divergence was originally measured at.
const SCALE: f64 = 150.0 / 72.0;

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

/// Assemble a one-page PDF from `objects` (1-based, in order) with an
/// exact xref table.
fn assemble_pdf(objects: &[Vec<u8>]) -> Vec<u8> {
    let mut pdf = b"%PDF-1.5\n%\xE2\xE3\xCF\xD3\n".to_vec();
    let mut offsets = Vec::with_capacity(objects.len());
    for (i, body) in objects.iter().enumerate() {
        offsets.push(pdf.len());
        pdf.extend_from_slice(format!("{} 0 obj\n", i + 1).as_bytes());
        pdf.extend_from_slice(body);
        pdf.extend_from_slice(b"\nendobj\n");
    }
    let xref_at = pdf.len();
    let mut xref = format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1);
    for off in offsets {
        let _ = writeln!(xref, "{off:010} 00000 n ");
    }
    let _ = write!(
        xref,
        "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n",
        objects.len() + 1
    );
    pdf.extend_from_slice(xref.as_bytes());
    pdf
}

fn stream_object(dict: &str, data: &[u8]) -> Vec<u8> {
    let mut obj = format!("<< {dict} /Length {} >>\nstream\n", data.len()).into_bytes();
    obj.extend_from_slice(data);
    obj.extend_from_slice(b"\nendstream");
    obj
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
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
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
    assemble_pdf(&objects)
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn render(session: &RasterSession, page: u32) -> (u32, u32, Vec<u8>) {
    let bmp = render_page_rgb(session, page, SCALE).expect("render page");
    (bmp.width, bmp.height, bmp.data().to_vec())
}

/// Assert two renders are byte-identical, reporting the divergence
/// statistics the way the original finding was measured.
fn assert_identical(cpu: &(u32, u32, Vec<u8>), gpu: &(u32, u32, Vec<u8>), what: &str) {
    assert_eq!((cpu.0, cpu.1), (gpu.0, gpu.1), "{what}: dimensions differ");
    assert_eq!(cpu.2.len(), gpu.2.len(), "{what}: byte length differs");
    let differing = cpu.2.iter().zip(&gpu.2).filter(|(a, b)| a != b).count();
    let max_delta = cpu
        .2
        .iter()
        .zip(&gpu.2)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0);
    assert_eq!(
        differing,
        0,
        "{what}: {differing} of {} bytes differ between CPU and cached-CUDA renders (max delta {max_delta})",
        cpu.2.len()
    );
}

fn cpu_session(bytes: Vec<u8>) -> RasterSession {
    open_session_from_bytes(bytes, &SessionConfig::with_policy(BackendPolicy::CpuOnly))
        .expect("CPU session")
}

fn cuda_session(bytes: Vec<u8>) -> RasterSession {
    open_session_from_bytes(bytes, &SessionConfig::with_policy(BackendPolicy::ForceCuda))
        .expect("CUDA session (needs a CUDA device)")
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[test]
fn synthetic_page_renders_identically_through_the_cache() {
    let pdf = fixture_pdf();
    let cpu = render(&cpu_session(pdf.clone()), 1);
    assert!(
        cpu.2.iter().any(|&b| b != 0xff),
        "fixture rendered blank on the CPU path"
    );

    let session = cuda_session(pdf);
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

/// Render `pages` of a corpus fixture on both backends and assert each is
/// byte-identical, checking that the cache grows by exactly
/// `cached_per_page[i]` entries after page `pages[i]`.  Skipped when the
/// fixture is absent.
fn assert_corpus_pages_identical(fixture: &str, pages: &[u32], cached_per_page: &[usize]) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(fixture);
    if !path.exists() {
        eprintln!("skipping: corpus fixture absent ({})", path.display());
        return;
    }
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
