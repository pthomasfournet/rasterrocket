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
