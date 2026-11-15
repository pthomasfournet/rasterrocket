use super::*;
use crate::test_helpers::{make_doc, make_doc_with_catalog_and_objects};

/// The bytes of the synthetic CMYK printer of `tests/fixtures/icc`.
fn cmyk_bytes() -> Vec<u8> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/icc/printer-cmyk-v4.icc"
    );
    std::fs::read(path).expect("the fixture is present")
}

/// A profile stream object of `bytes`.
fn stream(bytes: &[u8], n: u8) -> Vec<u8> {
    let mut body = format!("<</N {n} /Length {}>>\nstream\n", bytes.len()).into_bytes();
    body.extend_from_slice(bytes);
    body.extend_from_slice(b"\nendstream");
    body
}

#[test]
fn a_document_without_output_intents_has_none() {
    let doc = make_doc("");
    assert_eq!(
        document_output_intent(&doc).unwrap_err(),
        OutputIntentError::Missing
    );
}

/// A `GTS_PDFX` intent is chosen over one listed before it, and its
/// profile is the press.
#[test]
fn the_pdfx_intent_is_chosen_first() {
    let cmyk = cmyk_bytes();
    let srgb = moxcms::ColorProfile::new_srgb().encode().unwrap();
    let doc = make_doc_with_catalog_and_objects(
        " /OutputIntents [4 0 R 5 0 R]",
        &[
            (
                4,
                b"<</Type /OutputIntent /S /GTS_PDFA1 /DestOutputProfile 6 0 R>>".to_vec(),
            ),
            (
                5,
                b"<</Type /OutputIntent /S /GTS_PDFX /OutputConditionIdentifier (FOGRA39) \
                  /DestOutputProfile 7 0 R>>"
                    .to_vec(),
            ),
            (6, stream(&srgb, 3)),
            (7, stream(&cmyk, 4)),
        ],
    );
    let intent = document_output_intent(&doc).expect("an output intent");
    assert_eq!(intent.subtype, "GTS_PDFX");
    assert_eq!(intent.condition.as_deref(), Some("FOGRA39"));
    assert_eq!(intent.profile, (7, 0));
    let press = intent.press_profile(&doc).expect("a CMYK press");
    assert_eq!(press.space(), OutputSpace::Cmyk);
}

/// Without a `GTS_PDFX` intent another subtype's serves, and an RGB
/// profile is refused as no press, naming the subtype and its space.
#[test]
fn another_subtype_serves_and_an_rgb_profile_is_no_press() {
    let srgb = moxcms::ColorProfile::new_srgb().encode().unwrap();
    let doc = make_doc_with_catalog_and_objects(
        " /OutputIntents [<</S /GTS_PDFA1 /DestOutputProfile 4 0 R>>]",
        &[(4, stream(&srgb, 3))],
    );
    let intent = document_output_intent(&doc).unwrap();
    assert_eq!(intent.subtype, "GTS_PDFA1");
    let err = intent.press_profile(&doc).unwrap_err();
    assert_eq!(
        err,
        OutputIntentError::NotCmyk {
            subtype: "GTS_PDFA1".to_owned(),
            space: "RGB".to_owned()
        }
    );
    let message = err.to_string();
    assert!(message.contains("/GTS_PDFA1") && message.contains("profile of RGB values"));
    assert!(message.contains("nothing to proof against; use --output-profile"));
}

/// A `GTS_PDFX` intent that embeds no profile yields to one that does;
/// with none embedding one the error names the condition, or the
/// external reference.
#[test]
fn intents_without_an_embedded_profile() {
    let cmyk = cmyk_bytes();
    let doc = make_doc_with_catalog_and_objects(
        " /OutputIntents [<</S /GTS_PDFX /OutputConditionIdentifier (FOGRA39)>> \
         <</S /ISO_PDFE1 /DestOutputProfile 4 0 R>>]",
        &[(4, stream(&cmyk, 4))],
    );
    assert_eq!(document_output_intent(&doc).unwrap().subtype, "ISO_PDFE1");

    let doc = make_doc(" /OutputIntents [<</S /GTS_PDFX /OutputConditionIdentifier (FOGRA39)>>]");
    let err = document_output_intent(&doc).unwrap_err();
    assert_eq!(
        err,
        OutputIntentError::NoProfile {
            subtype: "GTS_PDFX".to_owned(),
            condition: Some("FOGRA39".to_owned()),
            external: false
        }
    );
    assert!(
        err.to_string()
            .contains("(FOGRA39) embeds no /DestOutputProfile")
    );

    let doc = make_doc(" /OutputIntents [<</S /GTS_PDFX /DestOutputProfileRef <<>> >>]");
    assert!(matches!(
        document_output_intent(&doc).unwrap_err(),
        OutputIntentError::NoProfile { external: true, .. }
    ));
}

/// A profile stream that is not a profile is unreadable.
#[test]
fn a_profile_that_does_not_parse_is_unreadable() {
    let doc = make_doc_with_catalog_and_objects(
        " /OutputIntents [<</S /GTS_PDFX /DestOutputProfile 4 0 R>>]",
        &[(4, stream(b"not a profile", 4))],
    );
    let intent = document_output_intent(&doc).unwrap();
    assert!(matches!(
        intent.press_profile(&doc).unwrap_err(),
        OutputIntentError::Unreadable { .. }
    ));
}

/// A page's own array is read; a page without one has none of its own.
#[test]
fn a_page_s_own_intents_are_read() {
    let cmyk = cmyk_bytes();
    let doc = make_doc_with_catalog_and_objects(
        "",
        &[
            (
                4,
                b"<</Type /Page /OutputIntents [<</S /GTS_PDFX /DestOutputProfile 6 0 R>>]>>"
                    .to_vec(),
            ),
            (5, b"<</Type /Page>>".to_vec()),
            (6, stream(&cmyk, 4)),
        ],
    );
    let own = page_output_intent(&doc, (4, 0))
        .expect("the page has intents")
        .expect("one embeds a profile");
    assert_eq!(own.profile, (6, 0));
    assert!(page_output_intent(&doc, (5, 0)).is_none());
}
