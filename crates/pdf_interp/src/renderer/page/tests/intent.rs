//! The rendering intents recorded for a page's marking operators.

use pdf::Document;

use super::*;
use crate::{IntentSet, RenderingIntent};

/// Page `(4, 0)` whose `/ExtGState` resources are the entries `states`.
fn doc(states: &str) -> Document {
    crate::test_helpers::make_doc_with_objects(&[(
        4,
        format!("<< /Type /Page /Resources << /ExtGState << {states} >> >> >>"),
    )])
}

/// The intents `content` marks under on a page of `doc`.
fn intents(doc: &Document, content: &str) -> IntentSet {
    let mut renderer = PageRenderer::new(10, 10, doc, (4, 0));
    renderer.execute(&crate::content::parse(content.as_bytes()));
    renderer.finish().1.content_intents
}

fn set(members: &[RenderingIntent]) -> IntentSet {
    let mut set = IntentSet::EMPTY;
    for &m in members {
        set.insert(m);
    }
    set
}

#[test]
fn marks_record_the_graphics_state_intent() {
    use RenderingIntent::{Perceptual, RelativeColorimetric, Saturation};
    let d = doc("/Sat << /RI /Saturation >>");
    assert_eq!(intents(&d, ""), IntentSet::EMPTY);
    assert_eq!(intents(&d, "/Perceptual ri"), IntentSet::EMPTY, "no mark");
    assert_eq!(intents(&d, "0 0 5 5 re f"), set(&[RelativeColorimetric]));
    assert_eq!(
        intents(&d, "/Perceptual ri 0 0 5 5 re S"),
        set(&[Perceptual])
    );
    assert_eq!(intents(&d, "/Sat gs 0 0 5 5 re B"), set(&[Saturation]));
    assert_eq!(
        intents(&d, "q /Perceptual ri Q 0 0 5 5 re f"),
        set(&[RelativeColorimetric]),
        "Q restores the intent"
    );
    assert_eq!(
        intents(&d, "/Vivid ri 0 0 5 5 re f"),
        set(&[RelativeColorimetric]),
        "an unknown name is relative colorimetric"
    );
    assert_eq!(
        intents(&d, "0 0 5 5 re f /Saturation ri 0 0 5 5 re f"),
        set(&[RelativeColorimetric, Saturation])
    );
}

#[test]
fn an_image_marks_under_its_own_intent() {
    use RenderingIntent::{AbsoluteColorimetric, Perceptual};
    let d = doc("");
    let image = |intent: &str| {
        format!("/Perceptual ri q 5 0 0 5 0 0 cm BI /W 1 /H 1 /CS /G /BPC 8 {intent} ID x EI Q")
    };
    assert_eq!(intents(&d, &image("")), set(&[Perceptual]));
    assert_eq!(
        intents(&d, &image("/Intent /AbsoluteColorimetric")),
        set(&[AbsoluteColorimetric])
    );
}

/// Page `(4, 0)` whose `/ColorSpace` resource `/P` is `[/ICCBased 5 0 R]`,
/// stream 5 the synthetic CMYK printer of `tests/fixtures/icc`, whose
/// tables differ by intent; its `/ExtGState` resources `/Perc` set the
/// perceptual intent and `/NoBpc` black-point compensation off.  The
/// image `/Im` (6) is one pixel of the ink `0 1 1 0` in that space, `/ImP`
/// (7) the same with its own `/Intent /Perceptual`; the shading `/Sh` (8)
/// is that ink throughout.
fn icc_doc() -> Document {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/icc/printer-cmyk-v4.icc"
    );
    let profile = std::fs::read(path).expect("the test profile is present");
    let stream = |dict: &str, data: &[u8]| {
        let mut s = format!("<< {dict} /Length {} >>\nstream\n", data.len()).into_bytes();
        s.extend_from_slice(data);
        s.extend_from_slice(b"\nendstream");
        s
    };
    let image = "/Type /XObject /Subtype /Image /Width 1 /Height 1 \
                 /ColorSpace [/ICCBased 5 0 R] /BitsPerComponent 8";
    let ink = [0u8, 255, 255, 0];
    crate::test_helpers::make_doc_with_objects(&[
        (
            4,
            b"<< /Type /Page /Resources << /ColorSpace << /P [/ICCBased 5 0 R] >> \
              /ExtGState << /Perc << /RI /Perceptual >> /NoBpc << /UseBlackPtComp /OFF >> >> \
              /XObject << /Im 6 0 R /ImP 7 0 R >> /Shading << /Sh 8 0 R >> >> >>"
                .to_vec(),
        ),
        (5, stream("/N 4", &profile)),
        (6, stream(image, &ink)),
        (7, stream(&format!("{image} /Intent /Perceptual"), &ink)),
        (
            8,
            b"<< /ShadingType 2 /ColorSpace [/ICCBased 5 0 R] /Coords [0 0 10 0] \
              /Function << /FunctionType 2 /Domain [0 1] /C0 [0 1 1 0] /C1 [0 1 1 0] /N 1 >> \
              /Extend [true true] >>"
                .to_vec(),
        ),
    ])
}

/// The colour of pixel `(x, 0)` of `content` rendered on a 10 × 10 page
/// of `doc`.
fn pixel(doc: &Document, content: &str, x: usize) -> [u8; 3] {
    let mut renderer = PageRenderer::new(10, 10, doc, (4, 0));
    renderer.execute(&crate::content::parse(content.as_bytes()));
    let bitmap = renderer.finish().0;
    let data = bitmap.data();
    [data[3 * x], data[3 * x + 1], data[3 * x + 2]]
}

/// The colour of the first pixel of `content` rendered on a page of `doc`.
fn first_pixel(doc: &Document, content: &str) -> [u8; 3] {
    pixel(doc, content, 0)
}

/// A colour converts under the intent in effect when it is painted, set
/// before or after the colour.
#[test]
fn a_colour_paints_under_the_intent_in_effect_at_the_paint() {
    let d = icc_doc();
    let fill = "/P cs 0 1 1 0 sc 0 0 10 10 re f";
    let relative = first_pixel(&d, fill);
    let perceptual = first_pixel(&d, &format!("/Perceptual ri {fill}"));
    assert_ne!(relative, perceptual, "the tables differ");
    assert_eq!(
        first_pixel(&d, "/P cs 0 1 1 0 sc /Perceptual ri 0 0 10 10 re f"),
        perceptual,
        "ri after the colour"
    );
    assert_eq!(
        first_pixel(&d, "/P cs 0 1 1 0 sc /Perc gs 0 0 10 10 re f"),
        perceptual,
        "gs /RI after the colour"
    );
    assert_eq!(
        first_pixel(&d, "/P CS 0 1 1 0 SC /Perceptual ri 2 w 0 9.5 m 10 9.5 l S"),
        first_pixel(&d, "/Perceptual ri /P CS 0 1 1 0 SC 2 w 0 9.5 m 10 9.5 l S"),
        "the stroke colour alike"
    );
    assert_eq!(
        first_pixel(&d, "/P CS 0 1 1 0 SC 2 w 0 9.5 m 10 9.5 l S"),
        relative,
        "the stroke of the fill's colour"
    );
    assert_eq!(
        first_pixel(
            &d,
            "/Perceptual ri /P cs 0 1 1 0 sc /RelativeColorimetric ri 0 0 10 10 re f"
        ),
        relative,
        "back to relative"
    );
}

/// An image converts under its own `/Intent`, else the graphics state's,
/// and one image drawn under two intents on a page is converted under
/// each.
#[test]
fn an_image_converts_under_its_intent() {
    let d = icc_doc();
    let relative = first_pixel(&d, "/P cs 0 1 1 0 sc 0 0 10 10 re f");
    let perceptual = first_pixel(&d, "/Perceptual ri /P cs 0 1 1 0 sc 0 0 10 10 re f");
    let draw = "q 10 0 0 10 0 0 cm /Im Do Q";
    assert_eq!(first_pixel(&d, draw), relative);
    assert_eq!(
        first_pixel(&d, &format!("/Perceptual ri {draw}")),
        perceptual
    );
    assert_eq!(first_pixel(&d, "q 10 0 0 10 0 0 cm /ImP Do Q"), perceptual);
    let both = "q 5 0 0 10 0 0 cm /Im Do Q /Perceptual ri q 5 0 0 10 5 0 cm /Im Do Q";
    assert_eq!(
        [pixel(&d, both, 0), pixel(&d, both, 9)],
        [relative, perceptual]
    );
    let inline = "q 5 0 0 10 0 0 cm BI /W 1 /H 1 /CS /P /BPC 8 /F /AHx ID 00FFFF00> EI Q \
                  /Perceptual ri q 5 0 0 10 5 0 cm BI /W 1 /H 1 /CS /P /BPC 8 /F /AHx ID 00FFFF00> EI Q";
    assert_eq!(
        [pixel(&d, inline, 0), pixel(&d, inline, 9)],
        [relative, perceptual],
        "inline images alike"
    );
}

/// A shading converts under the intent in effect where it paints, and
/// one shading painted under two intents on a page is resolved for each.
#[test]
fn a_shading_converts_under_the_intent_in_effect() {
    let d = icc_doc();
    let relative = first_pixel(&d, "/P cs 0 1 1 0 sc 0 0 10 10 re f");
    let perceptual = first_pixel(&d, "/Perceptual ri /P cs 0 1 1 0 sc 0 0 10 10 re f");
    let both = "q 0 0 5 10 re W n /Sh sh Q /Perc gs q 5 0 5 10 re W n /Sh sh Q";
    let got = [pixel(&d, both, 0), pixel(&d, both, 9)];
    for (got, want) in got.iter().zip([relative, perceptual]) {
        assert!(
            got.iter().zip(want).all(|(&g, w)| g.abs_diff(w) <= 1),
            "{got:?} against {want:?}"
        );
    }
}

/// Black-point compensation is on by default and `/UseBlackPtComp /OFF`
/// turns it off, converting the colour in effect again: the printer's
/// black, `L* = 8` (sRGB 24), paints black with it.
#[test]
fn use_black_pt_comp_switches_compensation() {
    let d = icc_doc();
    let black = "/P cs 0 0 0 1 sc";
    let on = first_pixel(&d, &format!("{black} 0 0 10 10 re f"));
    assert!(on.iter().all(|&c| c <= 1), "{on:?}");
    for content in [
        format!("{black} /NoBpc gs 0 0 10 10 re f"),
        format!("/NoBpc gs {black} 0 0 10 10 re f"),
    ] {
        let off = first_pixel(&d, &content);
        assert!(
            off.iter().all(|&c| c.abs_diff(24) <= 1),
            "{content}: {off:?}"
        );
    }
}
