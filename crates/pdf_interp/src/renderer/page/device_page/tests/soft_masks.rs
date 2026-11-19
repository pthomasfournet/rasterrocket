//! Soft masks recorded for the device against the CPU render.

use gpu::backend::page_device::PageDevice;
use pdf::Document;

use super::super::DeviceRecording;
use super::forms::stream;
use super::text::random_text;
use super::{DEPTHS, Rng, VulkanBackend, backend, cpu, device, random_content, record};

/// The soft masks of [`mask_doc`], by graphics-state name.
const MASKS: [&str; 8] = ["/LG", "/LR", "/LK", "/LC", "/LT", "/LN", "/AL", "/AT"];

/// A page whose graphics states set soft masks: luminosity groups of a
/// gray, an RGB, a CMYK and a `CalRGB` space over a backdrop `/BC` in
/// that space (`/LG`, `/LR`, `/LK`, `/LC`), one with a transfer function
/// (`/LT`), one without `/BC` or a group space (`/LN`), and alpha groups
/// (`/AL`, and `/AT` with a transfer function); besides `/T`, `/O` and
/// the fonts of the other tests.  The groups paint gray and coloured
/// fills, translucent ones, a stroke and text.
fn mask_doc() -> Document {
    let group = |cs: &str| format!("/Type /XObject /Subtype /Form /BBox [0 0 200 200] {cs}");
    let lum = "0.25 g 0 0 200 200 re f 0.9 0.2 0.4 rg 20 15 90 70 re f \
               /T gs 0 0 1 rg 60 40 120 120 re f /O gs 1 g 4 w 10 190 m 190 10 l S \
               0 g BT /F1 28 Tf 30 100 Td (Mask) Tj ET";
    let cmyk = "0 0 0 0.3 k 0 0 200 200 re f 0.1 0.7 0 0 k 30 30 100 80 re f \
                /T gs 0 0 0 1 k 70 60 100 100 re f";
    let alpha = "/T gs 0 g 10 10 120 110 re f /O gs 0.5 g 80 70 100 90 re f \
                 BT /F2 30 Tf 20 150 Td (Alpha) Tj ET";
    let res = "/Resources << /ExtGState << /T << /ca 0.45 /CA 0.45 >> /O << /ca 1 >> >> \
               /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> \
               /F2 << /Type /Font /Subtype /Type1 /BaseFont /Times-BoldItalic >> >> >>";
    let cal = "[/CalRGB << /WhitePoint [0.9505 1 1.089] /Gamma [2.2 2.2 2.2] >>]";
    let objects: Vec<(u32, String)> = vec![
        (
            4,
            "<< /Type /Page /MediaBox [0 0 200 200] /Resources << /ExtGState << \
             /T << /ca 0.37 /CA 0.61 >> /O << /ca 1 /CA 1 >> \
             /LG << /SMask << /S /Luminosity /G 5 0 R /BC [0.1] >> >> \
             /LR << /SMask << /S /Luminosity /G 6 0 R /BC [0 0.5 1] >> >> \
             /LK << /SMask << /S /Luminosity /G 7 0 R /BC [0 0 0 0.5] >> >> \
             /LC << /SMask << /S /Luminosity /G 8 0 R /BC [0.3 0.3 0.3] >> >> \
             /LT << /SMask << /S /Luminosity /G 5 0 R /TR 10 0 R >> >> \
             /LN << /SMask << /S /Luminosity /G 9 0 R >> >> \
             /AL << /SMask << /S /Alpha /G 11 0 R >> >> \
             /AT << /SMask << /S /Alpha /G 11 0 R /TR 10 0 R >> >> \
             /NM << /SMask /None >> >> \
             /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> \
             /F2 << /Type /Font /Subtype /Type1 /BaseFont /Times-BoldItalic >> \
             /F3 << /Type /Font /Subtype /Type1 /BaseFont /Courier >> >> >> >>"
                .into(),
        ),
        (
            5,
            stream(
                &format!(
                    "{} {res}",
                    group("/Group << /S /Transparency /CS /DeviceGray >>")
                ),
                lum,
            ),
        ),
        (
            6,
            stream(
                &format!(
                    "{} {res}",
                    group("/Group << /S /Transparency /CS /DeviceRGB >>")
                ),
                lum,
            ),
        ),
        (
            7,
            stream(
                &format!(
                    "{} {res}",
                    group("/Group << /S /Transparency /CS /DeviceCMYK >>")
                ),
                cmyk,
            ),
        ),
        (
            8,
            stream(
                &format!(
                    "{} {res}",
                    group(&format!("/Group << /S /Transparency /CS {cal} >>"))
                ),
                lum,
            ),
        ),
        (9, stream(&format!("{} {res}", group("")), lum)),
        (
            10,
            "<< /FunctionType 2 /Domain [0 1] /C0 [1] /C1 [0.1] /N 1.7 >>".into(),
        ),
        (
            11,
            stream(
                &format!("{} {res}", group("/Group << /S /Transparency /I true >>")),
                alpha,
            ),
        ),
    ];
    crate::test_helpers::make_doc_with_objects(&objects)
}

/// A random page painting fills, strokes and text under random soft
/// masks, some of them inside path clips, the mask in effect changed or
/// removed between groups, and one opaque fill of the whole page under a
/// mask.
fn random_masked(rng: &mut Rng, width: u32, height: u32) -> Vec<u8> {
    let mut s = Vec::new();
    for k in 0..6 {
        let mask = MASKS[usize::try_from(rng.below(8)).expect("small")];
        s.extend(format!("q {mask} gs ").as_bytes());
        if rng.below(3) == 0 {
            s.extend(
                format!(
                    "{} {} m {} {} l {} {} l h W n ",
                    rng.coord(width),
                    rng.coord(height),
                    rng.coord(width),
                    rng.coord(height),
                    rng.coord(width),
                    rng.coord(height)
                )
                .as_bytes(),
            );
        }
        if k == 2 {
            s.extend(b"0.8 0.1 0.1 rg 0 0 200 200 re f ");
        }
        let body = if rng.below(2) == 0 {
            random_content(rng, width, height)
        } else {
            random_text(rng, width, height)
        };
        s.extend(&body);
        if rng.below(3) == 0 {
            s.extend(b" /NM gs 0 0 1 rg 5 5 30 30 re f");
        }
        s.extend(b" Q ");
    }
    s
}

/// Random masked pages recorded and rendered on `backend` (or the host
/// model) match the CPU render at 8 and 16 bits.
fn masked_pages_match_the_cpu_render(backend: Option<&VulkanBackend>, paths: bool) {
    let doc = mask_doc();
    let mut rng = Rng(0x5_3A5C);
    for (width, height) in [(1, 1), (97, 83), (200, 200)] {
        for round in 0..4 {
            let content = random_masked(&mut rng, width, height);
            let text = String::from_utf8_lossy(&content);
            if let DeviceRecording::Refused(reason) = record(&doc, width, height, &content) {
                panic!("{width}×{height}, round {round} refused: {reason}\n{text}");
            }
            for depth in DEPTHS {
                let want = cpu(&doc, width, height, depth, &content);
                assert!(
                    device(backend, &doc, (width, height), depth, &content, paths) == want,
                    "{width}×{height}, round {round}, {} bits, outlines {paths}:\n{text}",
                    depth.bits()
                );
            }
        }
    }
}

#[test]
fn device_masked_pages_match_the_cpu_render() {
    let Some(backend) = backend() else {
        return;
    };
    masked_pages_match_the_cpu_render(Some(&backend), backend.page_paths());
}

#[test]
fn device_masked_pages_of_tables_only_match_the_cpu_render() {
    let Some(backend) = backend() else {
        return;
    };
    masked_pages_match_the_cpu_render(Some(&backend), false);
}

#[test]
fn host_model_of_masked_pages_matches_the_cpu_render() {
    masked_pages_match_the_cpu_render(None, true);
}

/// Every mask over overlapping translucent glyphs, a fill and a stroke,
/// each one object blended back once; a paint whose region lies outside
/// the clip, which the CPU leaves unmasked, records unmasked.
#[test]
fn each_mask_over_one_object_matches_the_cpu_render() {
    let backend = backend();
    let doc = mask_doc();
    for mask in MASKS {
        let content = format!(
            "q {mask} gs /T gs 0.1 0.5 0.9 rg BT /F1 60 Tf -8 Tc 10 60 Td (WWAV) Tj ET \
             0.9 0.6 0 rg 20 20 150 40 re f 0 0.6 0 RG 9 w 1 J 0 0 m 200 200 l S Q \
             q 0 0 1 1 re W n {mask} gs 0 g 100 100 50 50 re f Q"
        );
        assert!(matches!(
            record(&doc, 200, 200, content.as_bytes()),
            DeviceRecording::Recorded(_)
        ));
        for depth in DEPTHS {
            let want = cpu(&doc, 200, 200, depth, content.as_bytes());
            for paths in [true, false] {
                assert_eq!(
                    device(
                        backend.as_ref(),
                        &doc,
                        (200, 200),
                        depth,
                        content.as_bytes(),
                        paths
                    ),
                    want,
                    "{mask}, {} bits, outlines {paths}",
                    depth.bits()
                );
            }
        }
    }
}

/// One plane is held per mask set by `gs`, whatever the number of
/// objects painted under it.
#[test]
fn a_mask_is_held_once_for_its_objects() {
    let doc = mask_doc();
    let bytes = |content: &[u8]| match record(&doc, 200, 200, content) {
        DeviceRecording::Recorded(page) => page.bytes(),
        DeviceRecording::Refused(reason) => panic!("refused: {reason}"),
    };
    let one = bytes(b"/LG gs 0 g 0 0 10 10 re f");
    let three = bytes(b"/LG gs 0 g 0 0 10 10 re f 0 0 10 10 re f 0 0 10 10 re f");
    assert!(
        three - one < 1_000,
        "three objects take {} bytes more",
        three - one
    );
}
