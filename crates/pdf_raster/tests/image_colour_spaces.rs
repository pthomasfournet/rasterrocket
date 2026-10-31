//! An image converts its samples through its colour space exactly as a fill
//! in that space does: `CalRGB` and `CalGray` parameters, the `/Alternate`
//! of an unusable ICC profile, the `/Default*` entry of the resources for a
//! device space, and a `Separation` or `DeviceN` space of `/None` colorants
//! (which paints nothing).  Each page fills its left half through the
//! colour space and draws a one-pixel image of the same sample over its
//! right half; the two halves must agree.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use common::{assert_rgb_near, session, stream_object, with_page_tree};
use rasterrocket::{BackendPolicy, render_page_rgb};

/// Page size in points; rendered at one pixel per point.
const W: u32 = 200;
const H: u32 = 100;

const CALRGB: &str = "[/CalRGB << /WhitePoint [0.9505 1.0 1.089] /Gamma [1.8 1.8 1.8] >>]";
const CALGRAY: &str = "[/CalGray << /WhitePoint [0.9505 1.0 1.089] /Gamma 1.8 >>]";
const LAB: &str = "[/Lab << /WhitePoint [0.9505 1.0 1.089] >>]";

/// Render `content` on a page whose `/ColorSpace` resources are
/// `color_spaces`, with `/Im1` the image object 5 (`image`) and any
/// further objects from 6 on.  Returns the pixels at the centres of the
/// left and right halves.
fn halves(
    content: &[u8],
    color_spaces: &str,
    image: Vec<u8>,
    rest: Vec<Vec<u8>>,
) -> ([u8; 3], [u8; 3]) {
    let mut objects = vec![
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W} {H}] \
             /Resources << /ColorSpace << {color_spaces} >> /XObject << /Im1 5 0 R >> >> \
             /Contents 4 0 R >>"
        )
        .into_bytes(),
        stream_object("", content),
        image,
    ];
    objects.extend(rest);
    let s = session(with_page_tree(&objects), BackendPolicy::CpuOnly);
    let bmp = render_page_rgb(&s, 1, 1.0).expect("render page");
    let row = &bmp.data()[50 * bmp.stride..];
    let at = |x: usize| [row[x * 3], row[x * 3 + 1], row[x * 3 + 2]];
    (at(50), at(150))
}

/// A one-pixel image of `samples` in colour space `cs`.
fn image(cs: &str, samples: &[u8]) -> Vec<u8> {
    stream_object(
        &format!(
            "/Type /XObject /Subtype /Image /Width 1 /Height 1 /BitsPerComponent 8 \
             /ColorSpace {cs}"
        ),
        samples,
    )
}

/// The left half filled through `/CS1` with `fill_operands`, the right half
/// covered by `/Im1`.
fn fill_and_image(fill_operands: &str) -> Vec<u8> {
    format!("/CS1 cs {fill_operands} sc 0 0 100 {H} re f q 100 0 0 {H} 100 0 cm /Im1 Do Q")
        .into_bytes()
}

/// The left half painted by `fill`, the right half by a one-pixel inline
/// image of `samples` with parameters `params`.
fn fill_and_inline(fill: &str, params: &str, samples: &[u8]) -> Vec<u8> {
    let mut content =
        format!("{fill} 0 0 100 {H} re f q 100 0 0 {H} 100 0 cm BI /W 1 /H 1 {params} ID ")
            .into_bytes();
    content.extend_from_slice(samples);
    content.extend_from_slice(b" EI Q");
    content
}

#[test]
fn cal_rgb_image_matches_the_fill() {
    let (fill, img) = halves(
        &fill_and_image("0.5 0.5 0.5"),
        &format!("/CS1 {CALRGB}"),
        image(CALRGB, &[128, 128, 128]),
        vec![],
    );
    assert_ne!(fill, [128, 128, 128], "gamma 1.8 lightens the mid-tone");
    assert_rgb_near(img, fill, 1, "CalRGB image");
}

#[test]
fn cal_gray_image_matches_the_fill() {
    let (fill, img) = halves(
        &fill_and_image("0.5"),
        &format!("/CS1 {CALGRAY}"),
        image(CALGRAY, &[128]),
        vec![],
    );
    assert_ne!(fill, [128, 128, 128], "gamma 1.8 lightens the mid-tone");
    assert_rgb_near(img, fill, 1, "CalGray image");
}

#[test]
fn unusable_icc_profile_image_converts_through_its_alternate() {
    // An unparseable profile with a Lab alternate: the sample (128, 128,
    // 128) is L* 50.2, a* = b* ≈ 0 under Lab's default ranges, a mid gray.
    let icc = stream_object(&format!("/N 3 /Alternate {LAB}"), b"not an icc profile");
    let (fill, img) = halves(
        &fill_and_image("50.2 0 0"),
        "/CS1 [/ICCBased 6 0 R]",
        image("[/ICCBased 6 0 R]", &[128, 128, 128]),
        vec![icc],
    );
    assert!(
        fill.iter().all(|&c| c.abs_diff(119) <= 2),
        "Lab 50 is mid gray, got {fill:?}"
    );
    assert_rgb_near(img, fill, 1, "ICC image through its Lab alternate");

    // A CalGray alternate for a one-component profile.
    let icc = stream_object(&format!("/N 1 /Alternate {CALGRAY}"), b"garbage");
    let (fill, img) = halves(
        &fill_and_image("0.5"),
        "/CS1 [/ICCBased 6 0 R]",
        image("[/ICCBased 6 0 R]", &[128]),
        vec![icc],
    );
    assert_ne!(fill, [128, 128, 128]);
    assert_rgb_near(img, fill, 1, "ICC image through its CalGray alternate");
}

#[test]
fn device_space_image_takes_the_default_entry() {
    // `/DefaultGray` is a CalGray space: both the `g` fill and the DeviceGray
    // image take it.
    let (fill, img) = halves(
        format!("0.5 g 0 0 100 {H} re f q 100 0 0 {H} 100 0 cm /Im1 Do Q").as_bytes(),
        &format!("/DefaultGray {CALGRAY}"),
        image("/DeviceGray", &[128]),
        vec![],
    );
    assert_ne!(
        fill,
        [128, 128, 128],
        "the default space applies to the fill"
    );
    assert_rgb_near(img, fill, 1, "DeviceGray image under /DefaultGray");

    // `/DefaultRGB` likewise, for an XObject and an inline image.
    let (fill, img) = halves(
        format!("0.5 0.5 0.5 rg 0 0 100 {H} re f q 100 0 0 {H} 100 0 cm /Im1 Do Q").as_bytes(),
        &format!("/DefaultRGB {CALRGB}"),
        image("/DeviceRGB", &[128, 128, 128]),
        vec![],
    );
    assert_ne!(fill, [128, 128, 128]);
    assert_rgb_near(img, fill, 1, "DeviceRGB image under /DefaultRGB");
    let (fill, inline) = halves(
        &fill_and_inline("0.5 0.5 0.5 rg", "/CS /RGB /BPC 8", &[128, 128, 128]),
        &format!("/DefaultRGB {CALRGB}"),
        image("/DeviceRGB", &[128, 128, 128]),
        vec![],
    );
    assert_rgb_near(inline, fill, 1, "inline /RGB image under /DefaultRGB");
}

#[test]
fn none_colorant_images_paint_nothing() {
    const TINT: &str = "<< /FunctionType 2 /Domain [0 1] /C0 [1] /C1 [0] /N 1 >>";
    let sep_none = format!("[/Separation /None /DeviceGray {TINT}]");
    let dn_none = "[/DeviceN [/None /None] /DeviceGray \
                   << /FunctionType 2 /Domain [0 1 0 1] /C0 [1] /C1 [0] /N 1 >>]";
    let sep_spot = format!("[/Separation /Spot /DeviceGray {TINT}]");
    // A red page; the image covers the right half.
    let red_page = format!("1 0 0 rg 0 0 {W} {H} re f q 100 0 0 {H} 100 0 cm /Im1 Do Q");
    let cases: [(&str, &[u8]); 2] = [(&sep_none, &[255]), (dn_none, &[255, 255])];
    for (cs, samples) in cases {
        let (left, right) = halves(red_page.as_bytes(), "", image(cs, samples), vec![]);
        assert_eq!(left, [255, 0, 0]);
        assert_eq!(
            right,
            [255, 0, 0],
            "{cs}: an image of /None colorants marks nothing"
        );
    }
    // The same tint on a real colorant paints.
    let (_, right) = halves(red_page.as_bytes(), "", image(&sep_spot, &[255]), vec![]);
    assert_eq!(
        right,
        [0, 0, 0],
        "a /Spot colorant at full tint paints black"
    );
    // An inline image in a /None space marks nothing either.
    let (_, right) = halves(
        &fill_and_inline(
            &format!("1 0 0 rg 0 0 {W} {H} re f 1 0 0 rg"),
            "/CS /CS1 /BPC 8",
            &[255],
        ),
        &format!("/CS1 {sep_none}"),
        image("/DeviceGray", &[0]),
        vec![],
    );
    assert_eq!(right, [255, 0, 0], "inline /None image marks nothing");
}

#[test]
fn indexed_image_converts_its_palette_through_the_base_space() {
    let (fill, img) = halves(
        &fill_and_image("0.5 0.5 0.5"),
        &format!("/CS1 {CALRGB}"),
        image(&format!("[/Indexed {CALRGB} 0 <808080>]"), &[0]),
        vec![],
    );
    assert_ne!(fill, [128, 128, 128]);
    assert_rgb_near(img, fill, 1, "Indexed over CalRGB");
}

/// A Lab space with `/Range [-50 50 -50 50]`.
const LAB_RANGE: &str = "[/Lab << /WhitePoint [0.9505 1.0 1.089] /Range [-50 50 -50 50] >>]";

#[test]
fn lab_image_takes_its_range_as_the_default_decode() {
    // Sample (128, 255, 128): L* 50.2, a* at the top of the range (+50),
    // b* at the middle (0).  Without the range a* would be +100.
    let (fill, img) = halves(
        &fill_and_image("50.2 50 0"),
        &format!("/CS1 {LAB_RANGE}"),
        image(LAB_RANGE, &[128, 255, 128]),
        vec![],
    );
    assert!(
        fill[0] > 150 && fill[0] < 230,
        "a* 50 is a muted magenta, got {fill:?}"
    );
    assert_rgb_near(img, fill, 2, "Lab image under /Range");
    // The same sample under the default range is a saturated a* +100.
    let (fill, img) = halves(
        &fill_and_image("50.2 100 0"),
        &format!("/CS1 {LAB}"),
        image(LAB, &[128, 255, 128]),
        vec![],
    );
    assert!(fill[0] > 240, "a* 100 saturates red, got {fill:?}");
    assert_rgb_near(img, fill, 2, "Lab image under the default range");
}

#[test]
fn lab_image_decode_array_maps_its_samples() {
    let (fill, img) = halves(
        &fill_and_image("50.2 20 0"),
        &format!("/CS1 {LAB}"),
        stream_object(
            &format!(
                "/Type /XObject /Subtype /Image /Width 1 /Height 1 /BitsPerComponent 8 \
                 /ColorSpace {LAB} /Decode [0 100 -20 20 -20 20]"
            ),
            &[128, 255, 128],
        ),
        vec![],
    );
    assert_rgb_near(img, fill, 2, "Lab image under /Decode");
}

#[test]
fn indexed_over_lab_scales_the_palette_to_the_range() {
    let (fill, img) = halves(
        &fill_and_image("50.2 50 0"),
        &format!("/CS1 {LAB_RANGE}"),
        image(&format!("[/Indexed {LAB_RANGE} 0 <80ff80>]"), &[0]),
        vec![],
    );
    assert_rgb_near(img, fill, 2, "Indexed over Lab with /Range");
}

#[test]
fn lab_fill_components_are_clipped_to_the_range() {
    // ISO 32000-2 §8.6.5.4: a*, b* outside /Range are adjusted to the
    // nearest valid value, so a* 100 in a [-50 50] space paints as a* 50.
    let (clipped, _) = halves(
        &fill_and_image("50.2 100 0"),
        &format!("/CS1 {LAB_RANGE}"),
        image("/DeviceGray", &[0]),
        vec![],
    );
    let (at_limit, _) = halves(
        &fill_and_image("50.2 50 0"),
        &format!("/CS1 {LAB_RANGE}"),
        image("/DeviceGray", &[0]),
        vec![],
    );
    assert_eq!(clipped, at_limit);
}

#[test]
fn lab_white_point_is_adapted_so_neutrals_stay_neutral() {
    // A D50 Lab space: L* 50 with a* = b* = 0 is a neutral gray after
    // adaptation, not the warm cast of D50 white seen from D65.
    const LAB_D50: &str = "[/Lab << /WhitePoint [0.9642 1.0 0.8249] >>]";
    let (fill, img) = halves(
        &fill_and_image("50.2 0 0"),
        &format!("/CS1 {LAB_D50}"),
        image(LAB_D50, &[128, 128, 128]),
        vec![],
    );
    assert!(
        fill[0].abs_diff(fill[2]) <= 2 && fill[0].abs_diff(119) <= 3,
        "neutral mid gray expected, got {fill:?}"
    );
    assert_rgb_near(img, fill, 2, "D50 Lab image");
}
