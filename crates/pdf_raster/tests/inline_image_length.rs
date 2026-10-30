//! Inline image data ends where its parameters say it does (PDF 32000-1
//! §8.9.7; `/Length` from PDF 2.0): an unfiltered image's data is
//! `⌈W·BPC·n/8⌉·H` bytes, a filtered one's is `/L`, so image bytes that
//! happen to read ` EI ` neither cut the image short nor run as operators.

#[expect(
    dead_code,
    reason = "shared helpers; this test uses PDF assembly and sessions"
)]
mod common;

use common::{assemble_pdf, session, stream_object};
use rasterrocket::{BackendPolicy, render_page_rgb};

/// Page side in points; it renders at one pixel per point.
const SIDE: u32 = 16;

/// Paint a 4 × 2 inline image with parameters `params` and data `data`
/// over the page, then a red square in the bottom-left corner, and return
/// the pixels at the centre of the image's top-left sample, of its bottom
/// row, and of the red square.
fn render(params: &str, data: &[u8]) -> [[u8; 3]; 3] {
    let mut content = format!("q {SIDE} 0 0 {SIDE} 0 0 cm BI /W 4 /H 2 {params} ID ").into_bytes();
    content.extend_from_slice(data);
    content.extend_from_slice(b"\nEI Q 1 0 0 rg 0 0 2 2 re f");
    let objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {SIDE} {SIDE}] /Contents 4 0 R >>")
            .into_bytes(),
        stream_object("", &content),
    ];
    let s = session(assemble_pdf(&objects), BackendPolicy::CpuOnly);
    let bmp = render_page_rgb(&s, 1, 1.0).expect("render page");
    let at = |x: usize, y: usize| {
        let i = y * bmp.stride + x * 3;
        let p = &bmp.data()[i..i + 3];
        [p[0], p[1], p[2]]
    };
    [at(2, 4), at(8, 12), at(1, 15)]
}

fn assert_near(name: &str, got: [u8; 3], want: [u8; 3]) {
    let close = got.iter().zip(want).all(|(&g, w)| g.abs_diff(w) <= 2);
    assert!(close, "{name}: got {got:?}, want {want:?}");
}

/// The bytes ` EI ` as image samples.
const EI: [u8; 4] = *b" EI ";

#[test]
fn unfiltered_data_length_comes_from_the_dimensions() {
    let mut gray = EI.to_vec();
    gray.extend_from_slice(&[0x80; 4]);
    let [first, bottom, red] = render("/CS /G /BPC 8", &gray);
    assert_near("gray, first sample", first, [0x20; 3]);
    assert_near("gray, bottom row", bottom, [0x80; 3]);
    assert_near("gray, operators after EI", red, [255, 0, 0]);

    // RGB: three bytes per sample, so ` EI ` straddles two samples.
    let mut rgb = vec![0x20, 0x20, 0x20];
    rgb.extend_from_slice(&EI);
    rgb.extend_from_slice(&[0x20; 5]);
    rgb.extend_from_slice(&[0x80; 12]);
    let [first, bottom, red] = render("/CS /RGB /BPC 8", &rgb);
    assert_near("RGB, first sample", first, [0x20; 3]);
    assert_near("RGB, bottom row", bottom, [0x80; 3]);
    assert_near("RGB, operators after EI", red, [255, 0, 0]);
}

#[test]
fn filtered_data_length_comes_from_l() {
    // RunLength: a 4-byte literal run ` EI `, 4 repeats of 0x80, EOD.
    let mut rl = vec![3];
    rl.extend_from_slice(&EI);
    rl.extend_from_slice(&[253, 0x80, 128]);
    let params = format!("/CS /G /BPC 8 /F /RL /L {}", rl.len());
    let [first, bottom, red] = render(&params, &rl);
    assert_near("RunLength, first sample", first, [0x20; 3]);
    assert_near("RunLength, bottom row", bottom, [0x80; 3]);
    assert_near("RunLength, operators after EI", red, [255, 0, 0]);
}
