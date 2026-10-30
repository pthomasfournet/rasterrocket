//! An image whose `/SMask` carries `/Matte` holds colours pre-blended
//! with the matte colour, `c' = m + α·(c − m)` (PDF 32000-1 §11.6.5.3);
//! the original colour is recovered before compositing, so a
//! half-transparent pixel shows its true colour, not the matte's.

#[expect(
    dead_code,
    reason = "shared helpers; this test uses PDF assembly and sessions"
)]
mod common;

use common::{assemble_pdf, session, stream_object};
use rasterrocket::{BackendPolicy, render_page_rgb};

/// Page side in points; it renders at one pixel per point.
const SIDE: u32 = 16;

/// Render a 1 × 1 image (colour space `cs`, samples `base`) at alpha 128
/// with `/Matte matte` over a white page, and return the centre pixel.
fn render(cs: &str, base: &[u8], smask_extra: &str) -> [u8; 3] {
    let objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {SIDE} {SIDE}] \
             /Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>"
        )
        .into_bytes(),
        stream_object(
            "",
            format!("q {SIDE} 0 0 {SIDE} 0 0 cm /Im0 Do Q").as_bytes(),
        ),
        stream_object(
            &format!(
                "/Type /XObject /Subtype /Image /Width 1 /Height 1 \
                 /ColorSpace {cs} /BitsPerComponent 8 /SMask 6 0 R"
            ),
            base,
        ),
        stream_object(
            &format!(
                "/Type /XObject /Subtype /Image /Width 1 /Height 1 \
                 /ColorSpace /DeviceGray /BitsPerComponent 8 {smask_extra}"
            ),
            &[128],
        ),
    ];
    let s = session(assemble_pdf(&objects), BackendPolicy::CpuOnly);
    let bmp = render_page_rgb(&s, 1, 1.0).expect("render page");
    let at = 8 * bmp.stride + 8 * 3;
    let px = &bmp.data()[at..at + 3];
    [px[0], px[1], px[2]]
}

fn assert_near(name: &str, got: [u8; 3], want: [u8; 3]) {
    let close = got.iter().zip(want).all(|(&g, w)| g.abs_diff(w) <= 6);
    assert!(close, "{name}: got {got:?}, want about {want:?}");
}

#[test]
fn matte_is_removed_before_compositing() {
    // Blue pre-blended with a red matte at alpha ½ is (128, 0, 128);
    // un-matted it is blue again, which over white at alpha ½ is
    // (128, 127, 255).  Drawn as stored it would be (191, 127, 191).
    assert_near(
        "RGB, red matte",
        render("/DeviceRGB", &[128, 0, 128], "/Matte [1 0 0]"),
        [128, 127, 255],
    );
    // White pre-blended with a black matte at alpha ½ is 128; un-matted
    // it is white, invisible over the white page.
    assert_near(
        "Gray, black matte",
        render("/DeviceGray", &[128], "/Matte [0]"),
        [255, 255, 255],
    );
    // CMYK: no ink pre-blended with a full-black (K = 1) matte at alpha ½
    // is K = 128; un-matted it is no ink.
    assert_near(
        "CMYK, black matte",
        render("/DeviceCMYK", &[0, 0, 0, 128], "/Matte [0 0 0 1]"),
        [255, 255, 255],
    );
}

#[test]
fn without_matte_the_samples_composite_as_stored() {
    assert_near(
        "RGB, no matte",
        render("/DeviceRGB", &[128, 0, 128], ""),
        [191, 127, 191],
    );
}
