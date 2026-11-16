//! Form `XObjects` recorded for the device against the CPU render.

use pdf::Document;

use super::super::DeviceRecording;
use super::{DEPTHS, Rng, VulkanBackend, backend, cpu, device, record};

/// A content stream object.
fn stream(dict: &str, content: &str) -> String {
    format!(
        "<< {dict} /Length {} >>\nstream\n{content}\nendstream",
        content.len()
    )
}

/// The forms [`random_forms`] draws, by name.
const FORMS: [&str; 5] = ["/F", "/S", "/N", "/P", "/E"];

/// A page drawing forms: `/F` (fills, a stroke, text and an image under
/// its own matrix and box), `/S` (scaled and offset, its own resources),
/// `/N` (drawing `/F` twice inside it), `/P` (its box larger than its
/// content), `/E` (empty), `/G` (a transparency group), `/R` (invoking
/// itself), and soft masks `/SM` and `/SN` whose groups fill their boxes
/// (the second's form lacks its group attributes); the page's
/// one annotation `/A` draws `/F` as its appearance.
fn form_doc() -> Document {
    let objects: Vec<(u32, String)> = vec![
        (
            4,
            "<< /Type /Page /MediaBox [0 0 120 90] /Annots [12 0 R] /Resources << \
             /ExtGState << /T << /ca 0.4 /CA 0.6 >> /O << /ca 1 /CA 1 >> \
             /SM << /SMask << /S /Luminosity /G 11 0 R >> >> \
             /SN << /SMask << /S /Luminosity /G 15 0 R >> >> >> \
             /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> \
             /XObject << /F 5 0 R /S 6 0 R /N 7 0 R /P 8 0 R /E 9 0 R /G 10 0 R \
             /R 13 0 R /Im 14 0 R >> >> >>"
                .into(),
        ),
        (
            5,
            stream(
                "/Type /XObject /Subtype /Form /BBox [0 0 40 30] /Matrix [1 0 0 1 2 3]",
                "0.8 0.2 0.1 rg 0 0 30 20 re f 0 0 1 RG 2 w 5 5 m 35 25 l S \
                 BT /F1 9 Tf 2 22 Td (Form) Tj ET q 12 0 0 9 20 2 cm /Im Do Q",
            ),
        ),
        (
            6,
            stream(
                "/Type /XObject /Subtype /Form /BBox [0 0 10 10] /Matrix [2.5 0 0 1.5 7 9] \
                 /Resources << /XObject << /Im 14 0 R >> >>",
                "0.1 0.9 0.3 rg 1 1 8 8 re f q 6 0 0 6 2 2 cm /Im Do Q",
            ),
        ),
        (
            7,
            stream(
                "/Type /XObject /Subtype /Form /BBox [0 0 100 80] \
                 /Resources << /XObject << /F 5 0 R >> >>",
                "/F Do q 1 0 0 1 30 25 cm /F Do Q",
            ),
        ),
        (
            8,
            stream(
                "/Type /XObject /Subtype /Form /BBox [-50 -50 200 200]",
                "0.5 0.5 0.9 rg 0 0 1000 1000 re f",
            ),
        ),
        (
            9,
            stream("/Type /XObject /Subtype /Form /BBox [0 0 10 10]", ""),
        ),
        (
            10,
            stream(
                "/Type /XObject /Subtype /Form /BBox [0 0 20 20] \
                 /Group << /S /Transparency >>",
                "0 0 20 20 re f",
            ),
        ),
        (
            11,
            stream(
                "/Type /XObject /Subtype /Form /BBox [0 0 120 90] \
                 /Group << /S /Transparency /CS /DeviceGray >>",
                "0.6 g 0 0 120 90 re f",
            ),
        ),
        (
            12,
            "<< /Type /Annot /Subtype /Square /Rect [70 50 110 80] \
             /AP << /N 5 0 R >> >>"
                .into(),
        ),
        (
            13,
            stream(
                "/Type /XObject /Subtype /Form /BBox [0 0 20 20] \
                 /Resources << /XObject << /R 13 0 R >> >>",
                "0 0 1 rg 0 0 5 5 re f /R Do",
            ),
        ),
        (
            14,
            stream(
                "/Type /XObject /Subtype /Image /Width 3 /Height 2 /ColorSpace /DeviceRGB \
                 /BitsPerComponent 8 /Filter /ASCIIHexDecode",
                "ff000000ff000000ff808080ffff0000ffff>",
            ),
        ),
        (
            15,
            stream(
                "/Type /XObject /Subtype /Form /BBox [0 0 120 90]",
                "0.3 g 0 0 120 90 re f",
            ),
        ),
    ];
    crate::test_helpers::make_doc_with_objects(&objects)
}

/// A random content stream drawing the forms of [`FORMS`] under random
/// axis-aligned matrices, opaque and translucent, under rectangle clips,
/// with fills between.
fn random_forms(rng: &mut Rng) -> Vec<u8> {
    let mut s = vec![String::from("1 0 0 1 0.3 -0.2 cm")];
    for _ in 0..12 {
        s.push("q".into());
        if rng.below(4) == 0 {
            let (x, y) = (rng.coord(120), rng.coord(90));
            s.push(format!("{x} {y} 60 50 re W n"));
        }
        s.push(if rng.below(3) == 0 { "/T gs" } else { "/O gs" }.into());
        if rng.below(3) == 0 {
            s.push(format!(
                "{} g {} {} 20 15 re f",
                rng.unit(),
                rng.coord(120),
                rng.coord(90)
            ));
        }
        let (sx, sy) = (rng.unit().mul_add(1.5, 0.3), rng.unit().mul_add(1.5, 0.3));
        let flip = if rng.below(4) == 0 { -1.0 } else { 1.0 };
        s.push(format!(
            "{} 0 0 {sy} {} {} cm",
            sx * flip,
            rng.coord(120),
            rng.coord(90)
        ));
        s.push(format!(
            "{} Do",
            FORMS[usize::try_from(rng.below(5)).expect("small")]
        ));
        s.push("Q".into());
    }
    s.join(" ").into_bytes()
}

/// Random form pages recorded and rendered on `backend` (or the host
/// model) match the CPU render at 8 and 16 bits.
fn recorded_form_pages_match_the_cpu_render(backend: Option<&VulkanBackend>, paths: bool) {
    let doc = form_doc();
    let mut rng = Rng(0xF0_4A5);
    for round in 0..8 {
        let content = random_forms(&mut rng);
        if let DeviceRecording::Refused(reason) = record(&doc, 120, 90, &content) {
            panic!(
                "round {round} refused: {reason}:\n{}",
                String::from_utf8_lossy(&content)
            );
        }
        for depth in DEPTHS {
            assert_eq!(
                device(backend, &doc, (120, 90), depth, &content, paths),
                cpu(&doc, 120, 90, depth, &content),
                "{} bits, round {round}:\n{}",
                depth.bits(),
                String::from_utf8_lossy(&content)
            );
        }
    }
}

#[test]
fn device_form_pages_match_the_cpu_render() {
    let Some(backend) = backend() else {
        return;
    };
    recorded_form_pages_match_the_cpu_render(Some(&backend), backend.page_paths());
}

#[test]
fn host_model_of_form_pages_matches_the_cpu_render() {
    recorded_form_pages_match_the_cpu_render(None, true);
}

#[test]
fn forms_the_device_does_not_take_refuse_the_recording() {
    let doc = form_doc();
    let refused = |content: &[u8]| match record(&doc, 120, 90, content) {
        DeviceRecording::Refused(reason) => reason,
        DeviceRecording::Recorded(_) => "",
    };
    assert_eq!(refused(b"/F Do /N Do /E Do /R Do"), "");
    assert_eq!(refused(b"/G Do"), "transparency group");
    // A box under a rotated matrix clips by a path.
    assert_eq!(refused(b"0.8 0.6 -0.6 0.8 30 10 cm /F Do"), "path clip");
    assert_eq!(refused(b"/SM gs 0 0 10 10 re f"), "soft mask");
}

/// A soft mask's group paints its own canvas while the page is recorded:
/// none of its paints lands in the page's commands, and a page that sets
/// a mask and paints nothing under it records only its own paints.
#[test]
fn a_soft_mask_group_records_nothing_on_the_page() {
    let doc = form_doc();
    for mask in ["/SN", "/SM"] {
        let content = format!("q {mask} gs Q 0 0 1 rg 0 0 5 5 re f");
        match record(&doc, 120, 90, content.as_bytes()) {
            DeviceRecording::Recorded(page) => {
                assert_eq!(page.len(), 1, "{mask}: the mask group's fill was recorded");
            }
            DeviceRecording::Refused(reason) => panic!("{mask} refused: {reason}"),
        }
    }
    let content = b"q /SM gs Q 0 0 1 rg 0 0 5 5 re f /F Do";
    for depth in DEPTHS {
        assert_eq!(
            device(None, &doc, (120, 90), depth, content, true),
            cpu(&doc, 120, 90, depth, content),
            "{} bits",
            depth.bits()
        );
    }
}

/// An annotation's appearance stream, a form, records with the page.
#[test]
fn annotation_appearances_record_with_the_page() {
    let doc = form_doc();
    let backend = backend();
    for depth in DEPTHS {
        let paint = |mut r: super::PageRenderer<'_>, recorded: bool| {
            if recorded {
                r.record_for_device(true).expect("an RGB page records");
            }
            r.execute(&crate::content::parse(b"0.9 g 0 0 50 40 re f"));
            r.render_annotations((4, 0));
            if recorded {
                let DeviceRecording::Recorded(page) = r.take_device_recording().expect("recorded")
                else {
                    panic!("the annotation was refused");
                };
                assert!(page.len() > 4, "the appearance was not recorded");
                r.paint_device_page(backend.as_ref()?, &page)
                    .expect("device page");
            }
            Some(super::samples(&r))
        };
        let want = paint(super::renderer(&doc, 120, 90, depth), false);
        let got = paint(super::unpainted(&doc, 120, 90, depth), true);
        if got.is_some() {
            assert_eq!(got, want, "{} bits", depth.bits());
        }
    }
}
