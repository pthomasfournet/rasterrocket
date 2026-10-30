//! A raw `DeviceRGB` image drawn below its resolution is area-averaged
//! onto the device pixels, whatever the stream carries past its declared
//! length: a 4× reduction of alternating black and white columns is a flat
//! mid gray, not a point-sampled checker.

#[expect(dead_code, reason = "shared helpers; this test uses only PDF assembly")]
mod common;

use common::{assemble_pdf, stream_object};
use rasterrocket::{RasterOptions, RenderedPage, raster_pdf_from_bytes};

/// Source samples per side; drawn over the 100 pt page at one pixel per
/// point, a 4× reduction.
const SIDE: usize = 400;

/// RGB rows alternating black and white columns, plus `trailing` bytes
/// past the declared length.
fn columns(trailing: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(SIDE * SIDE * 3 + trailing);
    for _ in 0..SIDE {
        for x in 0..SIDE {
            let v = if x % 2 == 0 { 0x00 } else { 0xFF };
            data.extend_from_slice(&[v, v, v]);
        }
    }
    data.resize(SIDE * SIDE * 3 + trailing, 0x55);
    data
}

fn render(data: &[u8]) -> RenderedPage {
    let objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
           /Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>"
            .to_vec(),
        stream_object("", b"q 100 0 0 100 0 0 cm /Im0 Do Q"),
        stream_object(
            &format!(
                "/Type /XObject /Subtype /Image /Width {SIDE} /Height {SIDE} \
                 /BitsPerComponent 8 /ColorSpace /DeviceRGB"
            ),
            data,
        ),
    ];
    let opts = RasterOptions::new(72.0);
    let mut pages: Vec<_> = raster_pdf_from_bytes(assemble_pdf(&objects), &opts).collect();
    assert_eq!(pages.len(), 1, "one page expected");
    pages.remove(0).1.expect("page renders")
}

#[test]
fn reduced_rgb_image_is_area_averaged_with_or_without_trailing_bytes() {
    for trailing in [0, 12] {
        let page = render(&columns(trailing));
        let interior: Vec<u8> = (10..90)
            .flat_map(|y| (10..90).map(move |x| (x, y)))
            .map(|(x, y)| page.pixels[(y * page.width + x) as usize])
            .collect();
        assert!(
            interior.iter().all(|&v| (120..=136).contains(&v)),
            "{trailing} trailing bytes: a 4× reduction of black and white columns is mid gray, \
             got min {} max {}",
            interior.iter().min().unwrap(),
            interior.iter().max().unwrap()
        );
    }
}
