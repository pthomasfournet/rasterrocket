//! Rendering into caller-owned buffers through a reused [`PageBuffer`]
//! writes the bytes of the owned-buffer entry points, and rejects output
//! buffers that cannot hold the page before drawing anything.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use std::fmt::Write as _;

use common::{assemble_pdf, corpus_fixture, session, stream_object};
use rasterrocket::{
    BackendPolicy, PageBuffer, PixelFormat, RasterError, RasterOptions, RasterSession,
    SessionConfig, open_session, open_session_from_bytes, render_page_gray, render_page_rgb,
    render_page_rgba,
};

/// A three-page document whose pages differ in size: a wide strip, a tall
/// page with a transparency group and a square page of lines rotated by
/// 2°, which deskew corrects.
fn three_pages() -> Vec<u8> {
    let mut skewed = String::from("q 0.99939 0.03490 -0.03490 0.99939 0 0 cm 0 g\n");
    for i in 0..30 {
        writeln!(skewed, "40 {} 320 5 re f", 30 + i * 11).expect("write to a String");
    }
    skewed.push('Q');
    assemble_pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R 4 0 R 5 0 R] /Count 3 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Contents 6 0 R >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 90 300] /Contents 7 0 R \
           /Group << /S /Transparency /CS /DeviceRGB >> >>"
            .to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 400] /Contents 8 0 R >>".to_vec(),
        stream_object("", b"1 0 0 rg 10 10 80 80 re f 0 0 1 rg 120 20 60 60 re f"),
        stream_object("", b"0 0.5 0 rg 10 10 50 250 re f"),
        stream_object("", skewed.as_bytes()),
    ])
}

/// Row `y` of a `row_bytes`-wide image at `stride` in `buf`.
fn rows(buf: &[u8], height: u32, row_bytes: usize, stride: usize) -> Vec<u8> {
    (0..height as usize)
        .flat_map(|y| buf[y * stride..][..row_bytes].iter().copied())
        .collect()
}

/// Every page of `session`, in an order that shrinks then grows the
/// canvas, rendered through one `PageBuffer` in every format at a tight and
/// a padded stride, compared with the owned-buffer entry points.
fn assert_matches_owned(session: &RasterSession, pages: &[u32], dpi: f32) {
    let mut buffer = PageBuffer::new();
    let mut out = Vec::new();
    for &page in pages {
        let (width, height) = session.page_size_px(page, dpi).expect("page size");
        for pad in [0, 13] {
            for deskew in [false, true] {
                let mut opts = RasterOptions::new(dpi);
                opts.deskew = deskew;
                let want = render_page_gray(session, page, &opts).expect("owned gray");
                assert_eq!((want.width, want.height), (width, height), "page {page}");
                let stride = width as usize + pad;
                out.resize(stride * height as usize, 0);
                let info = buffer
                    .render_gray_into(session, page, &opts, &mut out, stride)
                    .expect("gray into");
                assert_eq!((info.width, info.height), (width, height));
                assert_eq!(info.format, PixelFormat::Gray8);
                assert_eq!(info.page_num, page);
                assert!(
                    rows(&out, height, width as usize, stride) == want.pixels,
                    "page {page}: gray differs (deskew {deskew}, stride {stride})"
                );
            }

            let opts = RasterOptions::new(dpi);
            let want = render_page_rgb(session, page, f64::from(dpi) / 72.0).expect("owned rgb");
            let row = width as usize * 3;
            let stride = row + pad;
            out.resize(stride * height as usize, 0);
            let info = buffer
                .render_rgb_into(session, page, &opts, &mut out, stride)
                .expect("rgb into");
            assert_eq!(info.format, PixelFormat::Rgb8);
            let want_rows: Vec<u8> = (0..height)
                .flat_map(|y| want.row_bytes(y)[..row].iter().copied())
                .collect();
            assert!(
                rows(&out, height, row, stride) == want_rows,
                "page {page}: rgb differs (stride {stride})"
            );

            let want = render_page_rgba(session, page, &opts).expect("owned rgba");
            let row = width as usize * 4;
            let stride = row + pad;
            out.resize(stride * height as usize, 0);
            let info = buffer
                .render_rgba_into(session, page, &opts, &mut out, stride)
                .expect("rgba into");
            assert_eq!(info.format, PixelFormat::Rgba8);
            assert!(
                rows(&out, height, row, stride) == want.pixels,
                "page {page}: rgba differs (stride {stride})"
            );
        }
    }
}

#[test]
fn reused_buffer_matches_owned_renders() {
    let session = session(three_pages(), BackendPolicy::CpuOnly);
    assert_matches_owned(&session, &[3, 1, 2, 3, 1], 150.0);
}

#[test]
fn reused_buffer_matches_owned_renders_on_corpus_pages() {
    let config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    for (name, pages) in [
        ("corpus-01-native-text-small.pdf", &[1, 2][..]),
        ("corpus-04-ebook-mixed.pdf", &[1, 5]),
        ("corpus-08-scan-dct-1927.pdf", &[1, 2]),
    ] {
        let Some(path) = corpus_fixture(name) else {
            continue;
        };
        let session = open_session(&path, &config).expect("open corpus fixture");
        assert_matches_owned(&session, pages, 72.0);
    }
}

#[test]
fn deskew_rotates_the_skewed_page() {
    let session = session(three_pages(), BackendPolicy::CpuOnly);
    let mut opts = RasterOptions::new(150.0);
    let plain = render_page_gray(&session, 3, &opts).expect("plain");
    opts.deskew = true;
    let deskewed = render_page_gray(&session, 3, &opts).expect("deskewed");
    assert_ne!(
        plain.pixels, deskewed.pixels,
        "the fixture exercises rotation"
    );
}

#[test]
fn page_size_px_is_the_rendered_size() {
    let session = session(three_pages(), BackendPolicy::CpuOnly);
    assert_eq!(session.page_size_px(1, 72.0).unwrap(), (200, 100));
    assert_eq!(session.page_size_px(2, 144.0).unwrap(), (180, 600));
    // 400 pt at 100 dpi is 555.55… px, rounded to nearest.
    assert_eq!(session.page_size_px(3, 100.0).unwrap(), (556, 556));
    assert!(matches!(
        session.page_size_px(1, 0.0),
        Err(RasterError::InvalidOptions(_))
    ));
    assert!(matches!(
        session.page_size_px(4, 72.0),
        Err(RasterError::PageOutOfRange { page: 4, total: 3 })
    ));
}

#[test]
fn output_buffer_errors_leave_the_buffer_untouched() {
    let session = session(three_pages(), BackendPolicy::CpuOnly);
    let opts = RasterOptions::new(72.0);
    let mut buffer = PageBuffer::new();
    // Page 1 at 72 dpi is 200 × 100.
    let mut out = vec![0xAA; 200 * 100 * 4];

    let err = buffer
        .render_gray_into(&session, 1, &opts, &mut out, 199)
        .unwrap_err();
    assert!(matches!(
        err,
        RasterError::OutputStrideTooSmall {
            stride: 199,
            row_bytes: 200
        }
    ));
    let err = buffer
        .render_gray_into(&session, 1, &opts, &mut out[..200 * 100 - 1], 200)
        .unwrap_err();
    assert!(matches!(
        err,
        RasterError::OutputBufferTooSmall {
            len: 19_999,
            needed: 20_000
        }
    ));
    let err = buffer
        .render_rgb_into(&session, 1, &opts, &mut out, 200 * 3 - 1)
        .unwrap_err();
    assert!(matches!(
        err,
        RasterError::OutputStrideTooSmall { row_bytes: 600, .. }
    ));
    let err = buffer
        .render_rgba_into(&session, 1, &opts, &mut out[..800 * 99], 800)
        .unwrap_err();
    assert!(matches!(
        err,
        RasterError::OutputBufferTooSmall { needed: 80_000, .. }
    ));
    assert!(out.iter().all(|&b| b == 0xAA), "no error writes the buffer");

    // The last row needs no padding.
    let stride = 256;
    let mut exact = vec![0; 99 * stride + 200];
    let info = buffer
        .render_gray_into(&session, 1, &opts, &mut exact, stride)
        .expect("exactly large enough");
    assert_eq!((info.width, info.height), (200, 100));

    let mut deskew = RasterOptions::new(72.0);
    deskew.deskew = true;
    assert!(matches!(
        buffer.render_rgb_into(&session, 1, &deskew, &mut out, 600),
        Err(RasterError::InvalidOptions(_))
    ));
}

/// A page simulated on an ink canvas converts into the reused storage and
/// matches the owned render; the plain page beside it shares the buffer.
#[test]
fn reused_buffer_matches_owned_renders_on_an_ink_canvas() {
    let pdf = assemble_pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 120 80] \
           /Resources << /ExtGState << /GS << /OP true /OPM 1 >> >> >> \
           /Contents 5 0 R >>"
            .to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 6 0 R >>".to_vec(),
        stream_object(
            "",
            b"1 0 0 0 k 0 0 120 80 re f /GS gs 0 0 0 1 k 0 0 60 80 re f",
        ),
        stream_object("", b"0 1 0 rg 20 20 100 100 re f"),
    ]);
    let mut config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    config.overprint = true;
    let session = open_session_from_bytes(pdf, &config).expect("open");
    assert_matches_owned(&session, &[2, 1, 2, 1], 72.0);
}
