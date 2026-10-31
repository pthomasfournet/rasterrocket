//! An image whose `/SMask` carries `/Matte` holds colours pre-blended
//! with the matte colour, `c' = m + α·(c − m)` (PDF 32000-1 §11.6.5.3);
//! the original colour is recovered before compositing, so a
//! half-transparent pixel shows its true colour, not the matte's.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use common::{assert_rgb_near, rgb_at, stream_object, with_page_tree};

/// Page side in points; it renders at one pixel per point.
const SIDE: u32 = 16;

/// Render a 1 × 1 image (colour space `cs`, samples `base`) at alpha 128
/// with `/Matte matte` over a white page, and return the centre pixel.
fn render(cs: &str, base: &[u8], smask_extra: &str) -> [u8; 3] {
    let objects = vec![
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
    rgb_at(with_page_tree(&objects), &[(SIDE / 2, SIDE / 2)])[0]
}

/// [`assert_rgb_near`] within ±6 per channel.
fn assert_near(name: &str, got: [u8; 3], want: [u8; 3]) {
    assert_rgb_near(got, want, 6, name);
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
