//! Magnified images are interpolated (PDF 32000-1 §8.9.5.3): a draw that
//! upsamples an image blends neighbouring samples bilinearly when the image
//! asks for it with `/Interpolate true`, and on its own below fourfold
//! magnification, on both axes when the draw is rotated; at larger
//! magnification an image without the flag keeps its samples as uniform
//! blocks.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use common::{session, stream_object, with_page_tree};
use rasterrocket::{BackendPolicy, render_page_rgb};

/// Page size in points; the page renders at one pixel per point.
const PAGE: (u32, u32) = (100, 20);

/// Render `content` on the page, with image `XObject` `/Im` built from
/// `image_dict` and `samples`, and return the red channel of every pixel,
/// row-major.
fn render(content: &str, image_dict: &str, samples: &[u8]) -> Vec<u8> {
    let (w, h) = PAGE;
    let objects = vec![
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w} {h}] \
             /Resources << /XObject << /Im 5 0 R >> >> /Contents 4 0 R >>"
        )
        .into_bytes(),
        stream_object("", content.as_bytes()),
        stream_object(
            &format!("/Type /XObject /Subtype /Image /BitsPerComponent 8 {image_dict}"),
            samples,
        ),
    ];
    let s = session(with_page_tree(&objects), BackendPolicy::CpuOnly);
    let bmp = render_page_rgb(&s, 1, 1.0).expect("render page");
    (0..h as usize)
        .flat_map(|y| (0..w as usize).map(move |x| (x, y)))
        .map(|(x, y)| bmp.data()[y * bmp.stride + x * 3])
        .collect()
}

/// As [`render`], with the image's `/SMask` built from `mask_dict` and
/// `mask_samples` (object 6).
fn render_masked(
    content: &str,
    image_dict: &str,
    samples: &[u8],
    mask_dict: &str,
    mask_samples: &[u8],
) -> Vec<u8> {
    let (w, h) = PAGE;
    let objects = vec![
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w} {h}] \
             /Resources << /XObject << /Im 5 0 R >> >> /Contents 4 0 R >>"
        )
        .into_bytes(),
        stream_object("", content.as_bytes()),
        stream_object(
            &format!(
                "/Type /XObject /Subtype /Image /BitsPerComponent 8 {image_dict} /SMask 6 0 R"
            ),
            samples,
        ),
        stream_object(
            &format!(
                "/Type /XObject /Subtype /Image /BitsPerComponent 8 /ColorSpace /DeviceGray \
                 {mask_dict}"
            ),
            mask_samples,
        ),
    ];
    let s = session(with_page_tree(&objects), BackendPolicy::CpuOnly);
    let bmp = render_page_rgb(&s, 1, 1.0).expect("render page");
    (0..h as usize)
        .flat_map(|y| (0..w as usize).map(move |x| (x, y)))
        .map(|(x, y)| bmp.data()[y * bmp.stride + x * 3])
        .collect()
}

/// Row `y` of a page rendered by [`render`].
fn row(page: &[u8], y: usize) -> &[u8] {
    &page[y * PAGE.0 as usize..][..PAGE.0 as usize]
}

/// Number of distinct values in `values`.
fn distinct(values: &[u8]) -> usize {
    let mut seen = [false; 256];
    for &v in values {
        seen[usize::from(v)] = true;
    }
    seen.iter().filter(|&&s| s).count()
}

/// A black and a white sample drawn 50 px wide each.
const PAIR: [u8; 2] = [0x00, 0xFF];
const PAIR_DICT: &str = "/Width 2 /Height 1 /ColorSpace /DeviceGray";
const PAIR_DRAW: &str = "q 100 0 0 20 0 0 cm /Im Do Q";

#[test]
fn interpolate_flag_blends_a_large_magnification() {
    let page = render(PAIR_DRAW, &format!("{PAIR_DICT} /Interpolate true"), &PAIR);
    let r = row(&page, 10);
    // Sample centres sit at device x 25 and 75; between them the value
    // rises linearly, and outside them it holds the edge sample.
    assert_eq!(r[10], 0x00);
    assert_eq!(r[90], 0xFF);
    assert!(
        (120..=136).contains(&r[50]),
        "midpoint {} is not about half-way",
        r[50]
    );
    assert!(r[25..75].windows(2).all(|p| p[0] <= p[1]), "ramp {r:?}");
    assert!(distinct(r) > 40, "ramp has {} levels", distinct(r));
}

#[test]
fn large_magnification_without_the_flag_keeps_whole_samples() {
    let page = render(PAIR_DRAW, PAIR_DICT, &PAIR);
    let r = row(&page, 10);
    assert!(r[..50].iter().all(|&v| v == 0x00), "{r:?}");
    assert!(r[50..].iter().all(|&v| v == 0xFF), "{r:?}");
}

#[test]
fn small_magnification_blends_without_the_flag() {
    // Forty alternating columns over 100 px: 2.5× across, 2× down.
    let samples: Vec<u8> = (0..40 * 10)
        .map(|i| if i % 2 == 0 { 0x00 } else { 0xFF })
        .collect();
    let page = render(
        PAIR_DRAW,
        "/Width 40 /Height 10 /ColorSpace /DeviceGray",
        &samples,
    );
    let r = row(&page, 10);
    assert!(
        r.iter().any(|&v| (0x40..=0xC0).contains(&v)),
        "no blended pixel in {r:?}"
    );
}

#[test]
fn rotated_draw_blends() {
    // The pair at 50× magnification, a quarter turn plus 10°, flagged:
    // the blend is not limited to axis-aligned draws.
    let draw = "q 0.98481 0.17365 -0.17365 0.98481 50 -5 cm 30 0 0 20 0 0 cm /Im Do Q";
    let flagged = render(draw, &format!("{PAIR_DICT} /Interpolate true"), &PAIR);
    assert!(
        flagged.iter().any(|&v| (0x40..=0xC0).contains(&v)),
        "no blended pixel in a rotated flagged draw"
    );
}

/// Below fourfold magnification a rotated draw blends both axes, even one
/// that an axis-aligned draw at the same scale would not.
#[test]
fn rotated_draw_blends_an_unmagnified_axis() {
    // A hundred alternating columns drawn 100 px wide: one sample per pixel
    // across (not blended when axis-aligned), 1.6× down.
    let samples: Vec<u8> = (0..100 * 10)
        .map(|i| if i % 2 == 0 { 0x00 } else { 0xFF })
        .collect();
    let dict = "/Width 100 /Height 10 /ColorSpace /DeviceGray";
    let blended = |page: &[u8]| {
        // Device row 9 (user y 10), columns 20..80: inside the image for
        // both draws, away from its antialiased edges.
        row(page, 9)[20..80]
            .iter()
            .filter(|&&v| (0x40..=0xC0).contains(&v))
            .count()
    };
    let aligned = render("q 100 0 0 16 0 2 cm /Im Do Q", dict, &samples);
    assert_eq!(blended(&aligned), 0, "axis-aligned: {:?}", row(&aligned, 9));
    // The same draw turned by 3°.
    let rotated = render(
        "q 0.99863 0.05234 -0.05234 0.99863 0 2 cm 100 0 0 16 0 0 cm /Im Do Q",
        dict,
        &samples,
    );
    assert!(blended(&rotated) > 10, "rotated: {:?}", row(&rotated, 9));
}

/// A soft mask's own `/Interpolate` blends its alpha when it is magnified,
/// whatever the image's flag; without it the alpha keeps whole samples.
#[test]
fn soft_mask_interpolate_flag_blends_its_own_grid() {
    // A black 2 × 1 image under a 2 × 1 mask that is transparent on the
    // left and opaque on the right, drawn at 50× magnification.
    let black = [0x00, 0x00];
    let flagged = render_masked(
        PAIR_DRAW,
        PAIR_DICT,
        &black,
        "/Width 2 /Height 1 /Interpolate true",
        &PAIR,
    );
    let r = row(&flagged, 10);
    assert_eq!((r[10], r[90]), (0xFF, 0x00), "ends hold the edge alpha");
    assert!(
        (120..=136).contains(&r[50]),
        "midpoint {} is not about half-way",
        r[50]
    );
    assert!(distinct(r) > 40, "ramp has {} levels", distinct(r));

    let plain = render_masked(PAIR_DRAW, PAIR_DICT, &black, "/Width 2 /Height 1", &PAIR);
    let r = row(&plain, 10);
    assert!(r[..50].iter().all(|&v| v == 0xFF), "{r:?}");
    assert!(r[50..].iter().all(|&v| v == 0x00), "{r:?}");
}

#[test]
fn inline_image_interpolate_abbreviation_blends() {
    let mut content = b"q 100 0 0 20 0 0 cm BI /W 2 /H 1 /CS /G /BPC 8 /I true ID ".to_vec();
    content.extend_from_slice(&PAIR);
    content.extend_from_slice(b"\nEI Q");
    let (w, h) = PAGE;
    let objects = vec![
        format!("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w} {h}] /Contents 4 0 R >>")
            .into_bytes(),
        stream_object("", &content),
    ];
    let s = session(with_page_tree(&objects), BackendPolicy::CpuOnly);
    let bmp = render_page_rgb(&s, 1, 1.0).expect("render page");
    let mid = bmp.data()[10 * bmp.stride + 50 * 3];
    assert!((120..=136).contains(&mid), "midpoint {mid}");
}
