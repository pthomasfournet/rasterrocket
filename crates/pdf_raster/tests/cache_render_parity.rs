//! End-to-end render parity: a page rendered through the device-resident
//! image cache (`--backend cuda` or `--backend vulkan` with the `cache`
//! feature) must be byte-identical to the CPU render of the same page.
//!
//! The cached path decodes JPEGs once, keeps the pixels in VRAM, samples
//! them with the blit kernel, and composites the result onto the page in
//! content-stream order.  Everything downstream of the decode differs from
//! the CPU path, so the only meaningful check is the final page bitmap.
//!
//! Every test runs on CUDA (under `gpu-validation`) and on Vulkan (under
//! `vulkan`).  The Vulkan case runs on the device `PDF_RASTER_VULKAN_DEVICE`
//! names, or the automatically ranked one, and is skipped when no Vulkan
//! device opens.
//!
//! Run with:
//!   `cargo test -p rasterrocket --features cache,gpu-validation --test cache_render_parity`
//!   `cargo test -p rasterrocket --features cache,vulkan --test cache_render_parity`
//!   `PDF_RASTER_VULKAN_DEVICE=radv cargo test -p rasterrocket --features cache,vulkan --test cache_render_parity`

#![cfg(all(feature = "cache", any(feature = "gpu-validation", feature = "vulkan")))]

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use common::{
    assert_identical, corpus_fixture, render, render_diagnosed, session, stream_object,
    with_page_tree,
};
use jpeg_encoder::{ColorType, Encoder};
use rasterrocket::{
    BackendPolicy, FinishedCanvas, RasterError, RasterSession, RenderParams, SampleDepth,
    SessionConfig, open_session, open_session_from_bytes, render_page_canvas_with_diagnostics,
};

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
    fixture_pdf_with("", "")
}

/// [`fixture_pdf`] with `resources` added to the page's resource dictionary
/// and `prefix` run before its content.
fn fixture_pdf_with(resources: &str, prefix: &str) -> Vec<u8> {
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
    let mut content = prefix.as_bytes().to_vec();
    content.extend_from_slice(
        b"\
q 200.5 0 0 131.2 40.25 50.5 cm /ImA Do Q\n\
q 143.37 0 0 90.61 20.123 130.77 cm /ImA Do Q\n\
q 60.5 15.25 -10.75 75.5 200.33 140.17 cm /ImB Do Q\n\
q -120 0 0 60 290 20 cm /ImA Do Q\n\
q 100 0 0 63 30 30 cm /ImC Do Q\n\
0 0 1 rg 60 150 80 50 re f\n\
0.9 0.1 0.1 RG 4 w 10 120 m 290 230 l S\n\
BT /F1 18 Tf 30 200 Td (Parity) Tj ET\n",
    );

    let objects = vec![
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 240] \
             /Resources << /XObject << /ImA 5 0 R /ImB 6 0 R /ImC 7 0 R >> \
             /Font << /F1 9 0 R >> {resources} >> /Contents 4 0 R >>"
        )
        .into_bytes(),
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

// ── backends ──────────────────────────────────────────────────────────────────

/// The GPU backend a cached session renders on.
#[derive(Clone, Copy, Debug)]
enum Backend {
    #[cfg(feature = "gpu-validation")]
    Cuda,
    #[cfg(feature = "vulkan")]
    Vulkan,
}

impl Backend {
    /// The session configuration forcing this backend.
    fn config(self) -> SessionConfig {
        match self {
            #[cfg(feature = "gpu-validation")]
            Self::Cuda => SessionConfig::with_policy(BackendPolicy::ForceCuda),
            #[cfg(feature = "vulkan")]
            Self::Vulkan => {
                let mut config = SessionConfig::with_policy(BackendPolicy::ForceVulkan);
                config.vulkan_device = SessionConfig::vulkan_device_from_env();
                config
            }
        }
    }

    /// The name of the backend in `ImageCache::backend_name`.
    const fn name(self) -> &'static str {
        match self {
            #[cfg(feature = "gpu-validation")]
            Self::Cuda => "cuda",
            #[cfg(feature = "vulkan")]
            Self::Vulkan => "vulkan",
        }
    }

    /// `opened`, or `None` (the test is skipped) for a Vulkan session that
    /// failed to open; a CUDA session must open.  An opened session must
    /// carry an image cache of this backend.
    #[cfg_attr(
        not(feature = "vulkan"),
        expect(
            clippy::unnecessary_wraps,
            reason = "only a Vulkan session is skipped; without `vulkan` every backend must open"
        )
    )]
    fn session(self, opened: Result<RasterSession, RasterError>) -> Option<RasterSession> {
        match opened {
            Ok(session) => {
                let cache = session
                    .image_cache()
                    .unwrap_or_else(|| panic!("{self:?} session must carry an image cache"));
                assert_eq!(cache.backend_name(), self.name(), "cache backend");
                Some(session)
            }
            #[cfg(feature = "vulkan")]
            Err(e) if matches!(self, Self::Vulkan) => {
                eprintln!("skipping: no Vulkan session ({e})");
                None
            }
            Err(e) => panic!("{self:?} session: {e}"),
        }
    }

    /// A session of this backend on the PDF `bytes`.
    fn session_from_bytes(self, bytes: Vec<u8>) -> Option<RasterSession> {
        self.session(open_session_from_bytes(bytes, &self.config()))
    }
}

/// One `#[test]` per backend compiled in, named `$cuda` and `$vulkan`, each
/// calling `$check` with its backend.
macro_rules! on_each_backend {
    ($check:ident, $cuda:ident, $vulkan:ident) => {
        #[cfg(feature = "gpu-validation")]
        #[test]
        fn $cuda() {
            $check(Backend::Cuda);
        }

        #[cfg(feature = "vulkan")]
        #[test]
        fn $vulkan() {
            $check(Backend::Vulkan);
        }
    };
}

// ── tests ─────────────────────────────────────────────────────────────────────

fn synthetic_page_renders_identically(backend: Backend) {
    let pdf = fixture_pdf();
    let cpu = render(&session(pdf.clone(), BackendPolicy::CpuOnly), 1);
    assert!(
        cpu.2.iter().any(|&b| b != 0xff),
        "fixture rendered blank on the CPU path"
    );

    let Some(session) = backend.session_from_bytes(pdf) else {
        return;
    };
    let cache = session.image_cache().expect("image cache");
    assert!(cache.is_empty(), "cache must start empty");

    // Cold pass: every eligible JPEG misses, decodes, and is inserted.
    let (cold, diag) = render_diagnosed(&session, 1);
    assert_identical(&cpu, &cold, "cold cache");
    assert!(
        diag.device_image_blits > 0,
        "the cached images are drawn by the device blit kernel"
    );
    // /ImA is first drawn at its resolution, so it is cached and its later
    // reduced draws read it back; /ImB is cached; /ImC carries an /SMask
    // and must not be.
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

on_each_backend!(
    synthetic_page_renders_identically,
    synthetic_page_renders_identically_through_the_cuda_cache,
    synthetic_page_renders_identically_through_the_vulkan_cache
);

/// A page painted on the overprint canvas draws its cached images through
/// the device blit kernel and converts the samples to ink,
/// byte-identically to the CPU.
fn overprint_page_renders_identically(backend: Backend) {
    let pdf = fixture_pdf_with("/ExtGState << /GSo << /OP true /OPM 1 >> >>", "/GSo gs\n");
    let mut config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    config.overprint = true;
    let cpu = render(
        &open_session_from_bytes(pdf.clone(), &config).expect("CPU session"),
        1,
    );
    assert!(
        cpu.2.iter().any(|&b| b != 0xff),
        "fixture rendered blank on the CPU path"
    );
    let mut config = backend.config();
    config.overprint = true;
    let Some(session) = backend.session(open_session_from_bytes(pdf, &config)) else {
        return;
    };
    let (gpu, diag) = render_diagnosed(&session, 1);
    assert_identical(&cpu, &gpu, "overprint page");
    assert!(
        diag.device_image_blits > 0 && diag.image_failures.is_empty(),
        "the cached images are drawn by the device blit kernel \
         ({} blits, failures {:?})",
        diag.device_image_blits,
        diag.image_failures
    );
    assert_eq!(
        session.image_cache().expect("image cache").len(),
        2,
        "the two unmasked JPEGs are drawn from the cache"
    );
}

on_each_backend!(
    overprint_page_renders_identically,
    overprint_page_renders_identically_through_the_cuda_cache,
    overprint_page_renders_identically_through_the_vulkan_cache
);

/// A 16-bit page keeps its JPEGs host-resident: the blit kernel samples at
/// 8 bits, so the images neither enter the cache nor reach the device
/// blit, and the page is byte-identical to the CPU's at 16 bits.
fn sixteen_bit_page_bypasses_the_cache(backend: Backend) {
    let params = RenderParams::new(common::SCALE).with_depth(SampleDepth::Sixteen);
    let render16 = |session: &RasterSession| {
        let (canvas, diag) =
            render_page_canvas_with_diagnostics(session, 1, &params).expect("16-bit render");
        let FinishedCanvas::Rgb16(bitmap) = canvas else {
            panic!("an RGB page, not {canvas:?}");
        };
        (bitmap.data().to_vec(), diag)
    };
    let pdf = fixture_pdf();
    let (cpu, _) = render16(&session(pdf.clone(), BackendPolicy::CpuOnly));
    let Some(session) = backend.session_from_bytes(pdf) else {
        return;
    };
    let (gpu, diag) = render16(&session);
    assert!(cpu == gpu, "the 16-bit pages differ");
    assert_eq!(
        diag.device_image_blits, 0,
        "a 16-bit page took the device blit"
    );
    assert!(diag.image_failures.is_empty(), "{:?}", diag.image_failures);
    assert!(
        session.image_cache().expect("image cache").is_empty(),
        "a 16-bit page filled the image cache"
    );
}

on_each_backend!(
    sixteen_bit_page_bypasses_the_cache,
    sixteen_bit_page_bypasses_the_cuda_cache,
    sixteen_bit_page_bypasses_the_vulkan_cache
);

/// Magnified draws, which the sampler interpolates: a gray JPEG drawn
/// rotated (both axes) and axis-aligned at about 3× across and 1:1 down
/// (columns only), and an RGB JPEG with `/Interpolate true` drawn at about
/// 10× (both axes).
fn interpolated_draws_render_identically(backend: Backend) {
    for draw in [
        "q 60.5 15.25 -10.75 75.5 200.33 140.17 cm /ImB Do Q",
        "q 70 0 0 38.4 150.4 5 cm /ImB Do Q",
        "q 190.3 0 0 101.7 10.6 30.2 cm /ImD Do Q",
    ] {
        assert_cached_draw_identical(backend, draw);
    }
}

on_each_backend!(
    interpolated_draws_render_identically,
    interpolated_draws_render_identically_through_the_cuda_cache,
    interpolated_draws_render_identically_through_the_vulkan_cache
);

/// Render `draw` (which may use `/ImB`, a 50×80 gray JPEG, and `/ImD`, a
/// 40×21 RGB JPEG with `/Interpolate true`) on the CPU and on `backend` and
/// assert the pages are byte-identical and the image came from the cache.
fn assert_cached_draw_identical(backend: Backend, draw: &str) {
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
    let Some(session) = backend.session_from_bytes(pdf) else {
        return;
    };
    let gpu = render(&session, 1);
    assert_identical(&cpu, &gpu, draw);
    assert_eq!(
        session.image_cache().expect("image cache").len(),
        1,
        "{draw}: the image should be drawn from the cache"
    );
}

/// Render `pages` of a corpus fixture at `dpi` on the CPU and on `backend`
/// and assert each is byte-identical, checking that the cache grows by
/// exactly `cached_per_page[i]` entries after page `pages[i]`.  Skipped
/// when the fixture is absent.
fn assert_corpus_pages_identical(
    backend: Backend,
    fixture: &str,
    dpi: f64,
    pages: &[u32],
    cached_per_page: &[usize],
) {
    let render = |session: &RasterSession, page: u32| {
        let bmp = rasterrocket::render_page_rgb(session, page, dpi / 72.0).expect("render page");
        (bmp.width, bmp.height, bmp.data().to_vec())
    };
    let Some(path) = corpus_fixture(fixture) else {
        return;
    };
    let cpu_session =
        open_session(&path, &SessionConfig::with_policy(BackendPolicy::CpuOnly)).expect("CPU");
    let Some(gpu_session) = backend.session(open_session(&path, &backend.config())) else {
        return;
    };
    let cache = gpu_session.image_cache().expect("image cache");
    let mut expected_len = 0;
    for (&page, &cached) in pages.iter().zip(cached_per_page) {
        let cpu = render(&cpu_session, page);
        let gpu = render(&gpu_session, page);
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
/// `DeviceRGB` JPEG scans (337 and 257 ppi) at a fractional offset, one
/// cached entry each at 360 dpi.  At 150 dpi they are drawn below their
/// resolution and bypass the cache.
fn corpus_scan_pages_render_identically(backend: Backend) {
    let fixture = "corpus-07-journal-dct-heavy.pdf";
    assert_corpus_pages_identical(backend, fixture, 360.0, &[1, 2], &[1, 1]);
    assert_corpus_pages_identical(backend, fixture, 150.0, &[1, 2], &[0, 0]);
}

on_each_backend!(
    corpus_scan_pages_render_identically,
    corpus_scan_pages_render_identically_through_the_cuda_cache,
    corpus_scan_pages_render_identically_through_the_vulkan_cache
);

/// Pages whose images must bypass the cache next to one that must not.
/// Pages 1–2 draw Flate-encoded `CalRGB` images carrying an `/SMask` over
/// vector fills and text — nothing is cacheable, so the cache stays empty
/// and the page must still match the CPU render exactly.  Page 3 is the
/// first DCT scan and inserts one entry.
fn corpus_masked_flate_pages_bypass_the_cache(backend: Backend) {
    assert_corpus_pages_identical(
        backend,
        "corpus-08-scan-dct-1927.pdf",
        150.0,
        &[1, 2, 3],
        &[0, 0, 1],
    );
}

on_each_backend!(
    corpus_masked_flate_pages_bypass_the_cache,
    corpus_masked_flate_pages_render_identically_and_bypass_the_cuda_cache,
    corpus_masked_flate_pages_render_identically_and_bypass_the_vulkan_cache
);

/// An `Auto` session holds no image cache, as it holds no GPU backend, and
/// renders as `CpuOnly` does.
#[test]
fn auto_session_holds_no_image_cache() {
    let auto = session(fixture_pdf(), BackendPolicy::Auto);
    assert!(auto.image_cache().is_none(), "Auto built an image cache");
    let cpu = render(&session(fixture_pdf(), BackendPolicy::CpuOnly), 1);
    assert_identical(&cpu, &render(&auto, 1), "Auto against CpuOnly");
}
