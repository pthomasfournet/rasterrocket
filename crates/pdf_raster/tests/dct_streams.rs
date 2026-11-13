//! `DCTDecode` images whose JPEG streams are coded, laid out or damaged in
//! ways the usual encoder output is not: arithmetic entropy coding,
//! components spread over several scans, a missing EOI marker, data that
//! stops short.  The page renders the
//! image whichever way its coefficients are coded, and reports an image
//! as partial exactly when its stream lacks data.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use common::{render_gray, stream_object, with_page_tree};
use rasterrocket::RenderedPage;

/// 43 × 37 colour image, 4:2:0, every component in one scan; and lossless
/// transcodes of it with one scan per component, and with `Y` in one scan
/// and `Cb Cr` in another.
const ONE_SCAN: &[u8] = include_bytes!("../../zune_jpeg/tests/images/scans/mix_420.jpg");
const SCAN_PER_COMPONENT: &[u8] =
    include_bytes!("../../zune_jpeg/tests/images/scans/mix_420_ni.jpg");
const LUMA_THEN_CHROMA: &[u8] =
    include_bytes!("../../zune_jpeg/tests/images/scans/mix_420_y_cbcr.jpg");

/// The same coefficients arithmetic-coded: sequential (SOF9) and
/// progressive (SOF10).
const ARITHMETIC: &[u8] = include_bytes!("../../zune_jpeg/tests/images/scans/mix_420_arith.jpg");
const ARITHMETIC_PROGRESSIVE: &[u8] =
    include_bytes!("../../zune_jpeg/tests/images/scans/mix_420_arith_prog.jpg");

/// 52 × 44 noise images: baseline with a restart marker after every MCU
/// row, and progressive.
const BASELINE_NOISE: &[u8] = include_bytes!("../../zune_jpeg/tests/images/cut/noise_420_rst.jpg");
const PROGRESSIVE_NOISE: &[u8] =
    include_bytes!("../../zune_jpeg/tests/images/cut/noise_420_prog.jpg");

/// A page of `w × h` points drawing the `w × h` JPEG `data` over all of
/// it, rendered at one pixel per point.
fn render(data: &[u8], w: u32, h: u32) -> RenderedPage {
    let objects = vec![
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w} {h}] \
             /Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>"
        )
        .into_bytes(),
        stream_object("", format!("q {w} 0 0 {h} 0 0 cm /Im0 Do Q").as_bytes()),
        stream_object(
            &format!(
                "/Type /XObject /Subtype /Image /Width {w} /Height {h} \
                 /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode"
            ),
            data,
        ),
    ];
    render_gray(with_page_tree(&objects))
}

/// The offsets of the markers of `data` after its first scan header that
/// are not restart markers: the headers of later scans and the EOI.
fn markers_after_first_scan(data: &[u8]) -> Vec<usize> {
    let first = data
        .windows(2)
        .position(|m| m == [0xFF, 0xDA])
        .expect("an SOS marker");
    (first + 2..data.len() - 1)
        .filter(|&at| data[at] == 0xFF && !matches!(data[at + 1], 0x00 | 0xD0..=0xD7 | 0xFF))
        .collect()
}

#[test]
fn every_scan_layout_of_an_image_renders_alike() {
    let reference = render(ONE_SCAN, 43, 37);
    assert_eq!(reference.diagnostics.image_failures, []);
    assert!(
        reference.pixels.iter().any(|&p| p < 200),
        "the image is drawn"
    );
    for (name, data) in [
        ("a scan per component", SCAN_PER_COMPONENT),
        ("luma then chroma", LUMA_THEN_CHROMA),
    ] {
        let page = render(data, 43, 37);
        assert_eq!(page.diagnostics.image_failures, [], "{name}");
        assert_eq!(page.pixels, reference.pixels, "{name}");
    }
}

/// The arithmetic decoder reports a block's coefficient count its own
/// way, and with it the inverse transform chosen, so a sample may round
/// one level apart from the Huffman-coded stream's.
#[test]
fn an_arithmetic_coded_image_renders_as_its_huffman_coded_twin() {
    let reference = render(ONE_SCAN, 43, 37);
    for (name, data) in [
        ("sequential", ARITHMETIC),
        ("progressive", ARITHMETIC_PROGRESSIVE),
    ] {
        assert!(
            data.windows(2)
                .any(|m| m == [0xFF, 0xC9] || m == [0xFF, 0xCA]),
            "{name}: an arithmetic-coded frame"
        );
        let page = render(data, 43, 37);
        assert_eq!(page.diagnostics.image_failures, [], "{name}");
        let worst = page
            .pixels
            .iter()
            .zip(&reference.pixels)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .expect("a rendered page");
        assert!(worst <= 1, "{name}: differs from the reference by {worst}");
    }
}

#[test]
fn a_stream_without_eoi_is_whole() {
    for (name, data) in [
        ("baseline", BASELINE_NOISE),
        ("progressive", PROGRESSIVE_NOISE),
        ("a scan per component", SCAN_PER_COMPONENT),
    ] {
        let (w, h) = if data == SCAN_PER_COMPONENT {
            (43, 37)
        } else {
            (52, 44)
        };
        let reference = render(data, w, h);
        let page = render(&data[..data.len() - 2], w, h);
        assert_eq!(page.diagnostics.image_failures, [], "{name}");
        assert_eq!(page.pixels, reference.pixels, "{name}");
    }
}

#[test]
fn a_stream_that_stops_short_is_drawn_and_reported_partial() {
    let cuts = |data: &[u8]| -> Vec<usize> {
        // In the middle of the entropy-coded data, and at every marker
        // that starts a later scan (none in the baseline stream).
        let markers = markers_after_first_scan(data);
        // The last marker is the EOI: a cut there is the stream without it.
        let (_eoi, later_scans) = markers.split_last().expect("an EOI marker");
        let mut cuts = later_scans.to_vec();
        cuts.push(data.len() * 6 / 10);
        cuts
    };
    for (name, data) in [
        ("baseline", BASELINE_NOISE),
        ("progressive", PROGRESSIVE_NOISE),
    ] {
        for at in cuts(data) {
            for eoi in [false, true] {
                let mut cut = data[..at].to_vec();
                if eoi {
                    cut.extend_from_slice(&[0xFF, 0xD9]);
                }
                let what = format!("{name} cut at {at}, EOI {eoi}");
                let page = render(&cut, 52, 44);
                let [failure] = page.diagnostics.image_failures.as_slice() else {
                    panic!(
                        "{what}: one failure expected, got {:?}",
                        page.diagnostics.image_failures
                    );
                };
                assert!(
                    failure.partial && failure.reason.contains("incomplete"),
                    "{what}: reported as {failure:?}"
                );
                assert!(
                    page.pixels.iter().any(|&p| p < 200),
                    "{what}: what decoded is drawn"
                );
            }
        }
    }
}
