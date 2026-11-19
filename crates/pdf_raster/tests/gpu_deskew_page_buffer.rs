//! Deskew under `BackendPolicy::ForceCuda` in a `gpu-deskew` build: the
//! NPP rotation of a skewed page written through a reused [`PageBuffer`].
//!
//! The GPU rotation is not the CPU's byte for byte: its border is the CPU
//! scalar kernel's, every other sample is NPP's bilinear, within 2 levels
//! of the CPU's at 8 bits and 8 levels of 65 535 at 16 bits (the
//! `deskew::rotate` module).  These tests assert that bound against a
//! `CpuOnly` session, and assert byte identity where the code promises it:
//! between two renders through one `PageBuffer`, and between a
//! `PageBuffer` render and the owned `render_page_gray` of the same
//! session.  A `PageBuffer` holds 8-bit pixels only, so the 16-bit page is
//! the owned render's.
//!
//! The tests skip when the CUDA rotation fails (no device), unless
//! `PDF_RASTER_REQUIRE_CUDA=1` requires it.
//!
//! Run with:
//!   `CUDA_ARCH=sm_75 cargo test -p rasterrocket --features gpu-deskew --test gpu_deskew_page_buffer`

#![cfg(feature = "gpu-deskew")]

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use std::fmt::Write as _;

use common::{assemble_pdf, session, stream_object};
use rasterrocket::{
    BackendPolicy, PageBuffer, Pixels, RasterError, RasterOptions, RasterSession, RenderedPage,
    SampleDepth, deskew::MIN_CORRECT_DEG, render_page_gray,
};

/// The largest difference, in levels of 255, between a sample of the GPU
/// rotation and the CPU rotation's.
const BOUND_8: i64 = 2;

/// The largest difference, in levels of 65 535, between a sample of the
/// GPU rotation and the CPU rotation's.
const BOUND_16: i64 = 8;

const DPI: f32 = 150.0;

/// The byte the padding of the output rows holds, which no render writes.
const PAD: u8 = 0xA5;

/// A 400 × 400 pt page of thirty bars rotated by 2°, which deskew
/// corrects, with a dark frame whose edges the rotation carries past the
/// page border.
fn skewed_page() -> Vec<u8> {
    let mut content = String::from("q 0.99939 0.03490 -0.03490 0.99939 0 0 cm 0 g\n");
    for i in 0..30 {
        writeln!(content, "40 {} 320 5 re f", 30 + i * 11).expect("write to a String");
    }
    content.push_str("Q 0.3 g 0 0 400 6 re f 0 394 400 6 re f 0 0 6 400 re f 394 0 6 400 re f");
    assemble_pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 400] /Contents 4 0 R >>".to_vec(),
        stream_object("", content.as_bytes()),
    ])
}

/// Whether `PDF_RASTER_REQUIRE_CUDA` is `1`: a test that would skip for
/// want of a CUDA device fails instead.
fn cuda_required() -> bool {
    std::env::var("PDF_RASTER_REQUIRE_CUDA").is_ok_and(|v| v == "1")
}

/// A `ForceCuda` session over [`skewed_page`] and its deskewed owned
/// render at `depth`, or `None` (the test is skipped) when the session or
/// the GPU rotation reports no CUDA backend.
fn cuda_render(depth: SampleDepth) -> Option<(RasterSession, RenderedPage)> {
    let skip = |e: &RasterError| {
        assert!(
            !cuda_required(),
            "PDF_RASTER_REQUIRE_CUDA=1, but the CUDA rotation is unavailable: {e}"
        );
        eprintln!("skipping: no CUDA rotation: {e}");
    };
    let config = rasterrocket::SessionConfig::with_policy(BackendPolicy::ForceCuda);
    let session = match rasterrocket::open_session_from_bytes(skewed_page(), &config) {
        Ok(session) => session,
        Err(e @ RasterError::BackendUnavailable(_)) => {
            skip(&e);
            return None;
        }
        Err(e) => panic!("ForceCuda session: {e}"),
    };
    match render_page_gray(&session, 1, &deskew_opts(depth)) {
        Ok(page) => Some((session, page)),
        Err(e @ RasterError::BackendUnavailable(_)) => {
            skip(&e);
            None
        }
        Err(e) => panic!("ForceCuda deskew: {e}"),
    }
}

fn deskew_opts(depth: SampleDepth) -> RasterOptions {
    let mut opts = RasterOptions::new(DPI);
    opts.deskew = true;
    opts.depth = depth;
    opts
}

/// Rows `0..height` of a `width`-byte-wide image at `stride` in `buf`.
fn rows(buf: &[u8], width: u32, height: u32, stride: usize) -> Vec<u8> {
    (0..height as usize)
        .flat_map(|y| buf[y * stride..][..width as usize].iter().copied())
        .collect()
}

/// The largest absolute difference between `a` and `b`, sample by sample.
fn max_difference<T: Copy + Into<i64>>(a: &[T], b: &[T]) -> i64 {
    assert_eq!(a.len(), b.len(), "sample counts differ");
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x.into() - y.into()).abs())
        .max()
        .unwrap_or(0)
}

/// The skew `page` measured, asserted large enough to be corrected, so the
/// page was rotated.
fn corrected_angle(page: &RenderedPage) -> f32 {
    let skew = page.skew.expect("the skewed page has a confident angle");
    assert!(skew.angle_deg.abs() >= MIN_CORRECT_DEG, "{skew:?}");
    skew.angle_deg
}

/// An 8-bit gray page deskewed on the GPU into one `PageBuffer` twice, at
/// a padded stride, writes the same bytes both times, those of the owned
/// render of the same session, within [`BOUND_8`] of the CPU deskew and
/// with the CPU's measured skew; the padding bytes are left as they are.
#[test]
fn gpu_deskew_into_a_reused_page_buffer() {
    let Some((cuda, owned)) = cuda_render(SampleDepth::Eight) else {
        return;
    };
    let Pixels::U8(owned_px) = &owned.pixels else {
        panic!("an 8-bit page");
    };
    let angle = corrected_angle(&owned);
    let (width, height) = (owned.width, owned.height);
    assert_eq!(cuda.page_size_px(1, DPI).unwrap(), (width, height));

    let stride = width as usize + 7;
    let opts = deskew_opts(SampleDepth::Eight);
    let mut buffer = PageBuffer::new();
    let mut renders = Vec::new();
    for pass in 0..2 {
        let mut out = vec![PAD; stride * height as usize];
        let info = buffer
            .render_gray_into(&cuda, 1, &opts, &mut out, stride)
            .unwrap_or_else(|e| panic!("pass {pass}: ForceCuda gray into: {e}"));
        assert_eq!((info.width, info.height), (width, height));
        let skew = info.skew.expect("a measured skew");
        assert_eq!(skew.angle_deg.to_bits(), angle.to_bits(), "pass {pass}");
        for row in out.chunks(stride) {
            assert!(
                row[width as usize..].iter().all(|&b| b == PAD),
                "pass {pass}: a padding byte was written"
            );
        }
        renders.push(rows(&out, width, height, stride));
    }
    assert!(
        renders[0] == renders[1],
        "the reused buffer's second render differs"
    );
    assert!(
        &renders[0] == owned_px,
        "the PageBuffer render differs from render_page_gray"
    );

    let cpu = session(skewed_page(), BackendPolicy::CpuOnly);
    let mut out = vec![0; stride * height as usize];
    let info = PageBuffer::new()
        .render_gray_into(&cpu, 1, &opts, &mut out, stride)
        .expect("CpuOnly gray into");
    assert_eq!(
        info.skew.map(|s| s.angle_deg.to_bits()),
        Some(angle.to_bits())
    );
    let cpu_px = rows(&out, width, height, stride);
    let diff = max_difference(&renders[0], &cpu_px);
    eprintln!("8-bit page {width}×{height} at {angle}°: max |GPU − CPU| = {diff}");
    assert!(diff <= BOUND_8, "GPU deskew {diff} levels from the CPU's");
}

/// A 16-bit gray page deskewed on the GPU lies within [`BOUND_16`] levels
/// of the CPU deskew, at the CPU's measured skew; a `PageBuffer` refuses
/// the depth before drawing.
#[test]
fn gpu_deskew_at_16_bits() {
    let Some((cuda, owned)) = cuda_render(SampleDepth::Sixteen) else {
        return;
    };
    let Pixels::U16(gpu_px) = &owned.pixels else {
        panic!("a 16-bit page");
    };
    let angle = corrected_angle(&owned);

    let cpu = session(skewed_page(), BackendPolicy::CpuOnly);
    let want = render_page_gray(&cpu, 1, &deskew_opts(SampleDepth::Sixteen)).expect("CpuOnly");
    let Pixels::U16(cpu_px) = &want.pixels else {
        panic!("a 16-bit page");
    };
    assert_eq!(
        want.skew.map(|s| s.angle_deg.to_bits()),
        Some(angle.to_bits())
    );
    let diff = max_difference(gpu_px, cpu_px);
    eprintln!(
        "16-bit page {}×{} at {angle}°: max |GPU − CPU| = {diff}",
        owned.width, owned.height
    );
    assert!(diff <= BOUND_16, "GPU deskew {diff} levels from the CPU's");

    let mut out = vec![0; owned.width as usize * owned.height as usize * 2];
    assert!(matches!(
        PageBuffer::new().render_gray_into(
            &cuda,
            1,
            &deskew_opts(SampleDepth::Sixteen),
            &mut out,
            owned.width as usize * 2
        ),
        Err(RasterError::InvalidOptions(_))
    ));
}
