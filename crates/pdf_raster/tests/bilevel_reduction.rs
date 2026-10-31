//! A bilevel image — a stencil mask, an image under an explicit stencil
//! `/Mask`, or a 1-bit image — drawn below its resolution covers each
//! device pixel in proportion to its painted area instead of point-sampling
//! one source sample per pixel.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use common::{render_one, stream_object, with_page_tree};
use rasterrocket::{RasterOptions, RenderedPage};

/// Source samples per side; drawn over the 100 pt page.
const SIDE: usize = 400;

/// 1-bit rows in which every fourth column (x ≡ 1 mod 4) is 0 and the rest
/// are 1: `10111011` per byte.  Whichever sample value paints, every 4 × 4
/// block holds the same coverage, a quarter or three quarters.
fn every_fourth_column_zero() -> Vec<u8> {
    vec![0xBB; SIDE / 8 * SIDE]
}

/// Image dictionary for the `SIDE × SIDE` bilevel stream.
fn bilevel_dict(extra: &str) -> String {
    format!(
        "/Type /XObject /Subtype /Image /Width {SIDE} /Height {SIDE} /BitsPerComponent 1 {extra}"
    )
}

/// One page painting `/Im0` over the whole 100 × 100 pt page with fill
/// colour black; `objects` are numbered from 5, `/Im0` being object 5.
fn page(objects: Vec<Vec<u8>>) -> Vec<u8> {
    let mut all = vec![
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
           /Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>"
            .to_vec(),
        stream_object("", b"0 g q 100 0 0 100 0 0 cm /Im0 Do Q"),
    ];
    all.extend(objects);
    with_page_tree(&all)
}

fn stencil() -> Vec<u8> {
    page(vec![stream_object(
        &bilevel_dict("/ImageMask true"),
        &every_fourth_column_zero(),
    )])
}

fn image_under_explicit_mask() -> Vec<u8> {
    page(vec![
        stream_object(
            "/Type /XObject /Subtype /Image /Width 2 /Height 2 \
             /ColorSpace /DeviceGray /BitsPerComponent 8 /Mask 6 0 R",
            &[0; 4],
        ),
        stream_object(
            &bilevel_dict("/ImageMask true"),
            &every_fourth_column_zero(),
        ),
    ])
}

fn one_bit_image() -> Vec<u8> {
    page(vec![stream_object(
        &bilevel_dict("/ColorSpace /DeviceGray"),
        &every_fourth_column_zero(),
    )])
}

fn render(pdf: Vec<u8>, dpi: f32) -> RenderedPage {
    render_one(pdf, &RasterOptions::new(dpi)).expect("page renders")
}

/// At 288 dpi the source maps 1:1 onto device pixels: no reduction, so
/// every pixel is solid black or white, the columns alternating as in the
/// source.  At 72 dpi each device pixel spans a 4 × 4 source block and must
/// carry that block's mean; point sampling instead gives every pixel the
/// value of one source column, solid black or white.
fn assert_reduction_conserves_coverage(pdf: &[u8], what: &str) {
    let full = render(pdf.to_vec(), 288.0);
    assert_eq!((full.width, full.height), (400, 400));
    let row = &full.pixels[..400];
    assert!(
        row.iter().all(|&g| g == 0 || g == 255) && row[0] != row[1],
        "{what}: full-resolution row is not the crisp source pattern: {:?}",
        &row[..8],
    );
    for (i, &g) in full.pixels.iter().enumerate() {
        assert_eq!(
            g,
            row[i % 400],
            "{what}: full-resolution pixel ({}, {})",
            i % 400,
            i / 400
        );
    }

    let block_mean = row[..4].iter().map(|&g| u32::from(g)).sum::<u32>() / 4;
    let reduced = render(pdf.to_vec(), 72.0);
    assert_eq!((reduced.width, reduced.height), (100, 100));
    for (i, &g) in reduced.pixels.iter().enumerate() {
        assert!(
            u32::from(g).abs_diff(block_mean) <= 2,
            "{what}: reduced pixel ({}, {}) = {g}, expected the block mean {block_mean}",
            i % 100,
            i / 100,
        );
    }
}

#[test]
fn stencil_mask_coverage_averages_when_reduced() {
    assert_reduction_conserves_coverage(&stencil(), "stencil");
}

#[test]
fn explicit_stencil_mask_coverage_averages_when_reduced() {
    assert_reduction_conserves_coverage(&image_under_explicit_mask(), "explicit /Mask");
}

#[test]
fn one_bit_image_averages_when_reduced() {
    assert_reduction_conserves_coverage(&one_bit_image(), "1-bit image");
}
