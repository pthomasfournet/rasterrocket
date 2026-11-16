//! Tests of the ICC profile-stream conversions.

use super::*;

/// The conversion under `intent` into sRGB.
const fn srgb(intent: ColourIntent) -> ColourTarget {
    ColourTarget::new(intent, WorkingSpace::Srgb)
}
use crate::resources::colorspace::ColorSpace;
use crate::test_helpers::make_doc_with_stream;

fn srgb_doc() -> Document {
    let bytes = ColorProfile::new_srgb()
        .encode()
        .expect("moxcms encodes its own sRGB profile");
    make_doc_with_stream(&bytes, " /N 3")
}

/// An sRGB `/DefaultRGB` converts exactly as `DeviceRGB` does, so the
/// device space stays selected.
#[test]
fn srgb_default_space_keeps_the_device_space() {
    let doc = srgb_doc();
    let conv =
        IccConversion::load(&doc, (2, 0), 3, ColourTarget::SRGB).expect("sRGB profile compiles");
    assert!(conv.is_device_equivalent());
    let mut dict = pdf::Dictionary::new();
    dict.set(
        "DefaultRGB",
        pdf::Object::Array(vec![
            pdf::Object::Name(b"ICCBased".to_vec()),
            pdf::Object::Reference((2, 0)),
        ]),
    );
    assert_eq!(
        super::super::default_space(&doc, &dict, &ColorSpace::DeviceRgb),
        None
    );
}

#[test]
fn profile_stream_is_compiled_once_per_document() {
    let doc = srgb_doc();
    let first =
        IccConversion::load(&doc, (2, 0), 3, ColourTarget::SRGB).expect("sRGB profile compiles");
    let second =
        IccConversion::load(&doc, (2, 0), 3, ColourTarget::SRGB).expect("sRGB profile compiles");
    assert!(Arc::ptr_eq(&first, &second));
}

#[test]
fn matrix_trc_profile_compiles_to_the_shaper() {
    let doc = srgb_doc();
    let conv =
        IccConversion::load(&doc, (2, 0), 3, ColourTarget::SRGB).expect("sRGB profile compiles");
    assert!(matches!(*conv, IccConversion::Shaper(_)));
    assert_eq!(
        conv.convert::<u8>(&doc, &[1.0, 0.0, 0.0]),
        Some([255, 0, 0])
    );
    assert_eq!(conv.convert::<u8>(&doc, &[1.0, 0.0]), None);
}

#[test]
fn unusable_profiles_do_not_compile() {
    let bytes = ColorProfile::new_srgb().encode().expect("encodes");
    let intent = ColourTarget::SRGB;
    assert!(IccConversion::compile(&bytes, 1, intent).is_none());
    assert!(IccConversion::compile(b"not a profile", 3, intent).is_none());
}

/// The bytes of the test profile `name` of `tests/fixtures/icc`, a
/// synthetic printer whose `AToB0`, `AToB1` and `AToB2` tables differ
/// and whose paper white is tinted.
fn fixture(name: &str) -> Vec<u8> {
    let path = format!(
        "{}/../../tests/fixtures/icc/{name}.icc",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// `[/ICCBased <printer-cmyk-v4>]` and the conversion of every intent.
fn cmyk_printer() -> (Document, [Arc<IccConversion>; 4]) {
    let doc = make_doc_with_stream(&fixture("printer-cmyk-v4"), " /N 4");
    let conv = RenderingIntent::ALL.map(|intent| {
        IccConversion::load(&doc, (2, 0), 4, srgb(ColourIntent::new(intent, false)))
            .expect("compiles")
    });
    (doc, conv)
}

#[test]
fn each_intent_converts_through_its_own_table() {
    use RenderingIntent::{AbsoluteColorimetric, Perceptual, RelativeColorimetric, Saturation};
    let (doc, conv) = cmyk_printer();
    let at = |intent: RenderingIntent| &conv[intent.index()];
    for (a, b) in [
        (RelativeColorimetric, Perceptual),
        (RelativeColorimetric, Saturation),
        (Perceptual, Saturation),
        (RelativeColorimetric, AbsoluteColorimetric),
    ] {
        assert!(!Arc::ptr_eq(at(a), at(b)), "{a} and {b} share a conversion");
    }
    // A saturated ink: the perceptual table maps it back out of the
    // compressed gamut, so the two differ.
    let ink = [0.0, 1.0, 1.0, 0.0];
    let relative = at(RelativeColorimetric).convert::<u8>(&doc, &ink).unwrap();
    let perceptual = at(Perceptual).convert::<u8>(&doc, &ink).unwrap();
    assert_ne!(relative, perceptual);
    // Repeated loads return the compiled conversion.
    let again = IccConversion::load(
        &doc,
        (2, 0),
        4,
        srgb(ColourIntent::new(RelativeColorimetric, false)),
    )
    .unwrap();
    assert!(Arc::ptr_eq(&again, at(RelativeColorimetric)));
}

#[test]
fn absolute_colorimetry_reproduces_the_paper_white() {
    use RenderingIntent::{AbsoluteColorimetric, RelativeColorimetric};
    let (doc, conv) = cmyk_printer();
    let paper = [0.0; 4];
    let relative = conv[RelativeColorimetric.index()].convert::<u8>(&doc, &paper);
    assert!(
        relative.unwrap().iter().all(|&c| c >= 254),
        "relative paper {relative:?}"
    );
    // The paper white (0.93, 0.965, 0.76) relative to D50: darker, and
    // yellow, its blue the lowest channel; an independent converter
    // gives (252.9, 250.9, 245.2).
    let [r, g, b] = conv[AbsoluteColorimetric.index()]
        .convert::<u8>(&doc, &paper)
        .unwrap();
    assert!(
        r < 254 && g < 254 && b + 5 < r.min(g),
        "absolute paper {:?}",
        [r, g, b]
    );
    let wide = conv[AbsoluteColorimetric.index()]
        .convert::<u16>(&doc, &paper)
        .unwrap();
    assert_eq!(wide.map(|v| u8::from_unit(v.to_unit())), [r, g, b]);
}

/// Black-point compensation maps the printer's black, `L* = 8` under
/// relative colorimetry, to sRGB black, as an independent converter
/// does (`-t1` gives 23.5, `-t1 -b` 0), and keeps white.
#[test]
fn compensation_maps_the_source_black_to_srgb_black() {
    let doc = make_doc_with_stream(&fixture("printer-cmyk-v4"), " /N 4");
    let at = |bpc, ink: &[f64]| {
        let intent = ColourIntent::new(RenderingIntent::RelativeColorimetric, bpc);
        let conv = IccConversion::load(&doc, (2, 0), 4, srgb(intent)).expect("compiles");
        conv.convert::<u8>(&doc, ink).expect("converts")
    };
    let black = [0.0, 0.0, 0.0, 1.0];
    assert!(at(false, &black).iter().all(|&c| c.abs_diff(24) <= 1));
    assert!(
        at(true, &black).iter().all(|&c| c <= 1),
        "{:?}",
        at(true, &black)
    );
    assert!(at(true, &[0.0; 4]).iter().all(|&c| c >= 254));
    // Compensation selects another conversion; absolute never compensates.
    let alike = |a, b| IccConversion::converts_alike(&doc, (2, 0), 4, a, b);
    let relative = |bpc| ColourIntent::new(RenderingIntent::RelativeColorimetric, bpc);
    let absolute = |bpc| ColourIntent::new(RenderingIntent::AbsoluteColorimetric, bpc);
    assert!(!alike(relative(true), relative(false)));
    assert!(alike(absolute(true), absolute(false)));
}

/// A matrix profile whose black is the PCS origin compiles one
/// conversion with compensation on or off.
#[test]
fn compensation_of_a_black_origin_is_the_plain_conversion() {
    let doc = srgb_doc();
    let relative = |bpc| ColourIntent::new(RenderingIntent::RelativeColorimetric, bpc);
    assert!(IccConversion::converts_alike(
        &doc,
        (2, 0),
        3,
        relative(true),
        relative(false)
    ));
}

/// A gray TRC profile under absolute colorimetry reproduces its tinted
/// paper white, which an independent converter gives as
/// (252.9, 250.9, 245.1); under relative colorimetry the paper is white.
#[test]
fn absolute_gray_reproduces_the_paper_white() {
    let doc = make_doc_with_stream(&fixture("printer-gray-v4"), " /N 1");
    let at = |intent| {
        let conv = IccConversion::load(&doc, (2, 0), 1, srgb(ColourIntent::new(intent, false)))
            .expect("compiles");
        conv.convert::<u8>(&doc, &[1.0]).expect("converts")
    };
    assert_eq!(at(RenderingIntent::RelativeColorimetric), [255; 3]);
    let [r, g, b] = at(RenderingIntent::AbsoluteColorimetric);
    assert!(
        r.abs_diff(253) <= 1 && g.abs_diff(251) <= 1 && b.abs_diff(245) <= 1,
        "absolute paper {:?}",
        [r, g, b]
    );
}

/// A matrix/TRC profile has one model: every intent but an
/// ICC-absolute one off a D50 media white shares its conversion.
#[test]
fn a_matrix_profile_shares_one_conversion() {
    let doc = srgb_doc();
    let conv = RenderingIntent::ALL.map(|intent| {
        IccConversion::load(&doc, (2, 0), 3, srgb(ColourIntent::new(intent, true)))
            .expect("compiles")
    });
    assert!(conv.iter().all(|c| Arc::ptr_eq(c, &conv[0])));
    assert!(matches!(*conv[0], IccConversion::Shaper(_)));
}

/// A profile with both a matrix/TRC model and an `AToB0` table converts
/// through the table.
#[test]
fn a_device_to_pcs_table_takes_precedence_over_the_matrix() {
    let printer = ColorProfile::new_from_slice(&fixture("printer-rgb-v4")).expect("parses");
    let mut both = ColorProfile::new_srgb();
    both.lut_a_to_b_perceptual = printer.lut_a_to_b_perceptual;
    let bytes = both.encode().expect("encodes");
    let doc = make_doc_with_stream(&bytes, " /N 3");
    let conv = IccConversion::load(&doc, (2, 0), 3, ColourTarget::SRGB).unwrap();
    assert!(matches!(*conv, IccConversion::Transform { .. }));
}

/// Every form of a CMYK printer's tables, to either working space,
/// compiles to the tabulated grid, whose 16-bit executor is given with it.
#[test]
fn cmyk_tables_compile_to_the_grid() {
    let doc = make_doc_with_stream(&fixture("printer-cmyk-v4"), " /N 4");
    for space in WorkingSpace::ALL {
        for intent in RenderingIntent::ALL {
            for bpc in [false, true] {
                let target = ColourTarget::new(ColourIntent::new(intent, bpc), space);
                let conv = IccConversion::load(&doc, (2, 0), 4, target).expect("compiles");
                let IccConversion::Transform {
                    n: 4,
                    recipe: None,
                    xform16,
                    ..
                } = &*conv
                else {
                    panic!("{intent} bpc {bpc} {space:?}: not the grid");
                };
                assert!(xform16.get().is_some_and(Option::is_some));
            }
        }
    }
}

/// `[/ICCBased <unparseable profile, /N n, /Alternate alt>]`.
fn garbage_doc(n: u8, alternate: &str) -> Document {
    make_doc_with_stream(b"not a profile", &format!(" /N {n} /Alternate {alternate}"))
}

fn icc(n: u8) -> ColorSpace {
    ColorSpace::IccBased {
        n,
        stream_id: Some((2, 0)),
    }
}

#[test]
fn unusable_profile_converts_through_its_alternate() {
    let doc = garbage_doc(3, "[/Lab << /WhitePoint [0.9505 1 1.089] >>]");
    let gray = icc(3).convert_to_rgb(&doc, crate::ColourTarget::SRGB, &[50.0, 0.0, 0.0]);
    assert!(gray.iter().all(|&c| c.abs_diff(119) <= 1), "got {gray:?}");
}

#[test]
fn alternate_with_another_component_count_is_ignored() {
    let doc = garbage_doc(3, "/DeviceGray");
    assert_eq!(
        icc(3).convert_to_rgb(&doc, crate::ColourTarget::SRGB, &[1.0, 0.0, 0.0]),
        [255, 0, 0]
    );
}

#[test]
fn derived_alternates_are_ignored() {
    let doc = garbage_doc(3, "[/ICCBased 2 0 R]");
    assert_eq!(
        icc(3).convert_to_rgb(&doc, crate::ColourTarget::SRGB, &[1.0, 0.0, 0.0]),
        [255, 0, 0]
    );
    // A Separation alternate whose own alternate is this profile.
    let doc = garbage_doc(
        1,
        "[/Separation /Spot [/ICCBased 2 0 R] << /FunctionType 2 /Domain [0 1] /N 1 >>]",
    );
    assert_eq!(
        icc(1).convert_to_rgb(&doc, crate::ColourTarget::SRGB, &[1.0]),
        [255; 3]
    );
}

#[test]
fn cal_alternate_converts_through_its_parameters() {
    // CalGray with the default Gamma 1: 0.5 is linear light, sRGB 188.
    let doc = garbage_doc(1, "[/CalGray << /WhitePoint [0.9505 1 1.089] >>]");
    let gray = icc(1).convert_to_rgb(&doc, crate::ColourTarget::SRGB, &[0.5]);
    assert!(gray.iter().all(|&c| c.abs_diff(188) <= 1), "got {gray:?}");
}
