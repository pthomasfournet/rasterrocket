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
/// tables differ by intent, and whose `/ExtGState` resource `/Perc` sets
/// the perceptual intent.
fn icc_doc() -> Document {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/icc/printer-cmyk-v4.icc"
    );
    let profile = std::fs::read(path).expect("the test profile is present");
    let mut stream = format!("<< /N 4 /Length {} >>\nstream\n", profile.len()).into_bytes();
    stream.extend_from_slice(&profile);
    stream.extend_from_slice(b"\nendstream");
    crate::test_helpers::make_doc_with_objects(&[
        (
            4,
            b"<< /Type /Page /Resources << /ColorSpace << /P [/ICCBased 5 0 R] >> \
              /ExtGState << /Perc << /RI /Perceptual >> >> >> >>"
                .to_vec(),
        ),
        (5, stream),
    ])
}

/// The colour of the first pixel of `content` rendered on a page of `doc`.
fn first_pixel(doc: &Document, content: &str) -> [u8; 3] {
    let mut renderer = PageRenderer::new(10, 10, doc, (4, 0));
    renderer.execute(&crate::content::parse(content.as_bytes()));
    let bitmap = renderer.finish().0;
    let data = bitmap.data();
    [data[0], data[1], data[2]]
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
