//! Images recorded for the device against the CPU render.

use std::fmt::Write as _;

use pdf::Document;

use super::super::super::xobject::IMAGE_DECODES;
use super::super::DeviceRecording;
use color::SampleDepth;

use super::{DEPTHS, Rng, VulkanBackend, backend, cpu, device, record, samples};

/// `n` pseudo-random bytes from `seed`, as hexadecimal.
fn hex(seed: u64, n: usize) -> String {
    let mut rng = Rng(seed);
    let mut s = String::with_capacity(n * 2 + 1);
    for _ in 0..n {
        let _ = write!(s, "{:02x}", rng.below(256));
    }
    s.push('>');
    s
}

/// An image `XObject` of `dict` (`/Width`, `/Height`, colour space, bits)
/// whose `n` bytes of samples are pseudo-random from `seed`.
fn image(dict: &str, seed: u64, n: usize) -> String {
    let data = hex(seed, n);
    format!(
        "<< /Type /XObject /Subtype /Image {dict} /Filter /ASCIIHexDecode /Length {} >>\n\
         stream\n{data}\nendstream",
        data.len()
    )
}

/// The images [`random_images`] draws, by name.
const IMAGES: [&str; 13] = [
    "/G", "/C", "/S", "/D", "/M", "/N", "/W", "/Q", "/X", "/K", "/B", "/I", "/U",
];

/// A page whose resources name the graphics states of the parent tests
/// plus `/A` (constant alpha 254 of 255) and `/R` (an inverting transfer
/// function), and the images of [`IMAGES`]: gray, RGB, two stencils (the
/// second decoded inverted), RGB under a soft mask of its own size and of
/// another size, 16-bit gray, RGB under a 16-bit mask, an interpolated
/// RGB image, a colour-keyed one, a large gray one, an indexed one, and a
/// 1 × 1 one.
#[expect(
    clippy::too_many_lines,
    reason = "one entry per object of the document"
)]
fn image_doc() -> Document {
    let objects: Vec<(u32, String)> = vec![
        (
            4,
            "<< /Type /Page /MediaBox [0 0 200 200] /Resources << /ExtGState << \
             /T << /ca 0.37 /CA 0.61 >> /O << /ca 1 /CA 1 >> /M << /BM /Multiply >> \
             /A << /ca 0.996 >> /R << /TR 21 0 R >> >> \
             /XObject << /G 5 0 R /C 6 0 R /S 7 0 R /D 8 0 R /M 10 0 R /N 12 0 R \
             /W 13 0 R /Q 15 0 R /X 16 0 R /K 17 0 R /B 18 0 R /I 19 0 R /U 20 0 R >> \
             >> >>"
                .into(),
        ),
        (
            5,
            image(
                "/Width 7 /Height 5 /ColorSpace /DeviceGray /BitsPerComponent 8",
                5,
                35,
            ),
        ),
        (
            6,
            image(
                "/Width 6 /Height 4 /ColorSpace /DeviceRGB /BitsPerComponent 8",
                6,
                72,
            ),
        ),
        (7, image("/Width 9 /Height 7 /ImageMask true", 7, 14)),
        (
            8,
            image("/Width 5 /Height 3 /ImageMask true /Decode [1 0]", 8, 3),
        ),
        (
            9,
            image(
                "/Width 5 /Height 5 /ColorSpace /DeviceGray /BitsPerComponent 8",
                9,
                25,
            ),
        ),
        (
            10,
            image(
                "/Width 5 /Height 5 /ColorSpace /DeviceRGB /BitsPerComponent 8 /SMask 9 0 R",
                10,
                75,
            ),
        ),
        (
            11,
            image(
                "/Width 9 /Height 6 /ColorSpace /DeviceGray /BitsPerComponent 8 /Interpolate true",
                11,
                54,
            ),
        ),
        (
            12,
            image(
                "/Width 4 /Height 3 /ColorSpace /DeviceRGB /BitsPerComponent 8 /SMask 11 0 R",
                12,
                36,
            ),
        ),
        (
            13,
            image(
                "/Width 3 /Height 3 /ColorSpace /DeviceGray /BitsPerComponent 16",
                13,
                18,
            ),
        ),
        (
            14,
            image(
                "/Width 4 /Height 4 /ColorSpace /DeviceGray /BitsPerComponent 16",
                14,
                32,
            ),
        ),
        (
            15,
            image(
                "/Width 4 /Height 4 /ColorSpace /DeviceRGB /BitsPerComponent 8 /SMask 14 0 R",
                15,
                48,
            ),
        ),
        (
            16,
            image(
                "/Width 3 /Height 2 /ColorSpace /DeviceRGB /BitsPerComponent 8 /Interpolate true",
                16,
                18,
            ),
        ),
        (
            17,
            image(
                "/Width 4 /Height 4 /ColorSpace /DeviceRGB /BitsPerComponent 8 \
                 /Mask [0 160 0 160 0 160]",
                17,
                48,
            ),
        ),
        (
            18,
            image(
                "/Width 60 /Height 45 /ColorSpace /DeviceGray /BitsPerComponent 8",
                18,
                2700,
            ),
        ),
        (
            19,
            image(
                "/Width 4 /Height 4 /ColorSpace [/Indexed /DeviceRGB 3 <ff000000ff000000ff808080>] \
                 /BitsPerComponent 2",
                19,
                4,
            ),
        ),
        (
            20,
            image(
                "/Width 1 /Height 1 /ColorSpace /DeviceRGB /BitsPerComponent 8",
                20,
                3,
            ),
        ),
        (
            21,
            "<< /FunctionType 2 /Domain [0 1] /C0 [1] /C1 [0] /N 1 >>".into(),
        ),
    ];
    crate::test_helpers::make_doc_with_objects(&objects)
}

/// A random image matrix for a page `w × h`: axis-aligned (flipped or not,
/// from below a pixel to the whole page), a quarter turn, or rotated and
/// skewed.
#[expect(
    clippy::many_single_char_names,
    reason = "coordinates and matrix entries"
)]
fn image_matrix(rng: &mut Rng, w: u32, h: u32) -> String {
    let (x, y) = (rng.coord(w), rng.coord(h));
    let size = |rng: &mut Rng, span: u32| match rng.below(4) {
        0 => rng.unit().mul_add(2.0, 0.2),
        1 => f64::from(span) * rng.unit().mul_add(1.2, 0.3),
        _ => rng.unit().mul_add(40.0, 2.0),
    };
    let (sx, sy) = (size(rng, w), size(rng, h));
    let sign = |rng: &mut Rng| if rng.below(4) == 0 { -1.0 } else { 1.0 };
    match rng.below(5) {
        0 => format!("0 {} {} 0 {x} {y} cm", sy * sign(rng), -sx * sign(rng)),
        1 => {
            let t = rng.unit() * std::f64::consts::TAU;
            let (s, c) = t.sin_cos();
            format!("{} {} {} {} {x} {y} cm", sx * c, sx * s, -sy * s, sy * c)
        }
        2 => format!(
            "{sx} {} {} {sy} {x} {y} cm",
            sx * 0.2 * rng.unit(),
            sy * 0.3 * rng.unit()
        ),
        _ => format!("{} 0 0 {} {x} {y} cm", sx * sign(rng), sy * sign(rng)),
    }
}

/// A random content stream of image draws interleaved with fills: every
/// image of [`IMAGES`] and inline images, through the matrices of
/// [`image_matrix`], opaque, at alpha 254 and translucent, under rectangle
/// clips, stencils in random fill colours.
#[expect(
    clippy::many_single_char_names,
    reason = "coordinates and the stream under construction"
)]
fn random_images(rng: &mut Rng, width: u32, height: u32) -> Vec<u8> {
    let (w, h) = (width, height);
    let mut s: Vec<Vec<u8>> = vec![b"1 0 0 1 0.21 -0.37 cm".to_vec()];
    for _ in 0..24 {
        s.push(b"q".to_vec());
        if rng.below(4) == 0 {
            let (x, y, dx, dy) = (rng.coord(w), rng.coord(h), rng.coord(w), rng.coord(h));
            s.push(format!("{x} {y} {} {} re W n", dx / 2.0, dy / 2.0).into_bytes());
        }
        s.push(["/O gs", "/O gs", "/A gs", "/T gs"][rng.below(4) as usize].into());
        s.push(format!("{} {} {} rg", rng.unit(), rng.unit(), rng.unit()).into_bytes());
        if rng.below(4) == 0 {
            let (x, y) = (rng.coord(w), rng.coord(h));
            s.push(
                format!("{x} {y} {} {} re f", rng.coord(w) / 3.0, rng.coord(h) / 3.0).into_bytes(),
            );
        }
        s.push(image_matrix(rng, w, h).into_bytes());
        let pick = rng.below(u32::try_from(IMAGES.len()).expect("few") + 2) as usize;
        if let Some(name) = IMAGES.get(pick) {
            s.push(format!("{name} Do").into_bytes());
        } else {
            let (iw, ih) = (1 + rng.below(5), 1 + rng.below(4));
            let (cs, n) = if pick == IMAGES.len() {
                ("/RGB", 3)
            } else {
                ("/G", 1)
            };
            // Bytes that are neither white space nor delimiters.
            let mut inline = format!("BI /W {iw} /H {ih} /CS {cs} /BPC 8 ID ").into_bytes();
            inline.extend(
                (0..iw * ih * n).map(|_| 0x50 + u8::try_from(rng.below(0x60)).expect("a byte")),
            );
            inline.extend(b" EI");
            s.push(inline);
        }
        s.push(b"Q".to_vec());
    }
    s.join(&b' ')
}

/// Random image pages recorded and rendered on `backend` (or the host
/// model) match the CPU render at 8 and 16 bits.
fn recorded_image_pages_match_the_cpu_render(backend: Option<&VulkanBackend>, paths: bool) {
    let doc = image_doc();
    let mut rng = Rng(0x1A6E);
    for (width, height) in [(1, 1), (64, 48), (150, 97), (41, 203)] {
        for round in 0..8 {
            let content = random_images(&mut rng, width, height);
            match record(&doc, width, height, &content) {
                DeviceRecording::Refused(reason) => panic!(
                    "{width}×{height}, round {round} refused: {reason}:\n{}",
                    String::from_utf8_lossy(&content)
                ),
                DeviceRecording::Recorded(page) => {
                    assert!(
                        width < 10 || page.len() > 8,
                        "{} commands recorded",
                        page.len()
                    );
                }
            }
            let mut r = super::renderer(&doc, width, height, SampleDepth::Eight);
            r.execute(&crate::content::parse(&content));
            assert!(
                r.diag.has_images && r.diag.image_failures.is_empty(),
                "images fail: {:?}",
                r.diag.image_failures
            );
            for depth in DEPTHS {
                let want = cpu(&doc, width, height, depth, &content);
                assert_eq!(
                    device(backend, &doc, (width, height), depth, &content, paths),
                    want,
                    "{} bits, {width}×{height}, round {round}:\n{}",
                    depth.bits(),
                    String::from_utf8_lossy(&content)
                );
            }
        }
    }
}

#[test]
fn device_image_pages_match_the_cpu_render() {
    let Some(backend) = backend() else {
        return;
    };
    recorded_image_pages_match_the_cpu_render(Some(&backend), backend.page_paths());
}

#[test]
fn device_image_pages_of_tables_only_match_the_cpu_render() {
    let Some(backend) = backend() else {
        return;
    };
    recorded_image_pages_match_the_cpu_render(Some(&backend), false);
}

#[test]
fn host_model_of_image_pages_matches_the_cpu_render() {
    recorded_image_pages_match_the_cpu_render(None, true);
}

/// The composite at the boundary of the direct path: constant alpha 255
/// (direct) and 254 (the pipe), for an image, a masked image and a
/// stencil, over a painted backdrop, at 8 and 16 bits, on the device and
/// the host model.
#[test]
fn both_image_composites_match_the_cpu_render_at_the_alpha_boundary() {
    let doc = image_doc();
    let backend = backend();
    for gs in ["/O", "/A", "/T"] {
        for name in ["/C", "/M", "/N", "/Q", "/S", "/X"] {
            let content = format!(
                "0.2 0.7 0.4 rg 0 0 40 30 re f {gs} gs 0.9 0.1 0.5 rg \
                 q 33 0 0 23 3.3 2.9 cm {name} Do Q q 0 7 -9 0 30 1 cm {name} Do Q"
            );
            for depth in DEPTHS {
                let want = cpu(&doc, 40, 30, depth, content.as_bytes());
                for b in [backend.as_ref(), None] {
                    assert_eq!(
                        device(b, &doc, (40, 30), depth, content.as_bytes(), true),
                        want,
                        "{content} at {} bits, device {}",
                        depth.bits(),
                        b.is_some()
                    );
                }
            }
        }
    }
}

#[test]
fn image_draws_the_device_does_not_take_refuse_the_recording() {
    let doc = image_doc();
    let refused = |content: &[u8]| match record(&doc, 50, 50, content) {
        DeviceRecording::Refused(reason) => reason,
        DeviceRecording::Recorded(_) => "",
    };
    assert_eq!(refused(b"q 40 0 0 40 0 0 cm /C Do Q"), "");
    assert_eq!(refused(b"BI /W 1 /H 1 /CS /G /BPC 8 ID P EI"), "");
    assert_eq!(refused(b"/M gs 40 0 0 40 0 0 cm /C Do"), "blend mode");
    assert_eq!(refused(b"/M gs 40 0 0 40 0 0 cm /S Do"), "blend mode");
    assert_eq!(
        refused(b"0 0 m 40 0 l 20 40 l h W n 40 0 0 40 0 0 cm /C Do"),
        "path clip"
    );
    assert_eq!(
        refused(b"/R gs 40 0 0 40 0 0 cm /G Do"),
        "image transfer function"
    );
    assert_eq!(
        refused(b"/Pattern cs /P scn 40 0 0 40 0 0 cm /S Do"),
        "pattern"
    );
}

/// A page refused after it drew an image renders on the CPU from the
/// image's first decode, and the same as a CPU render.
#[test]
fn a_page_refused_after_an_image_decodes_it_once() {
    const CONTENT: &[u8] = b"q 40 0 0 40 1 1 cm /C Do Q q 50 0 0 50 0 0 cm /B Do Q \
                             0 0 m 40 0 l 20 40 l h W n 0 0 50 50 re f \
                             q 40 0 0 40 1 1 cm /C Do Q";
    let doc = image_doc();
    for depth in DEPTHS {
        let decodes = || IMAGE_DECODES.with(std::cell::Cell::get);
        let before = decodes();
        let mut first = super::unpainted(&doc, 50, 50, depth);
        first.record_for_device(true).expect("an RGB page records");
        first.execute(&crate::content::parse(CONTENT));
        assert!(matches!(
            first.take_device_recording(),
            Some(DeviceRecording::Refused("path clip"))
        ));
        assert_eq!(decodes() - before, 2, "the recording decodes each image");
        let fallback = first.take_device_fallback();
        let mut second = super::renderer(&doc, 50, 50, depth);
        second.resume_after_device(fallback);
        second.execute(&crate::content::parse(CONTENT));
        assert_eq!(decodes() - before, 2, "the CPU pass decodes nothing again");
        assert_eq!(samples(&second), cpu(&doc, 50, 50, depth, CONTENT));
    }
}

/// A page with images is not ruled out before it is interpreted.
#[test]
fn image_operators_are_recorded() {
    let ops = crate::content::parse(b"/C Do BI /W 1 /H 1 /CS /G /BPC 8 ID P EI");
    assert_eq!(
        super::device_page_ineligible(false, false, true, &ops),
        None
    );
}
