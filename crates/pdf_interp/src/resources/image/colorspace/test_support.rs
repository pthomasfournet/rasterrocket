//! Images, colour spaces and documents the image-colour tests share.

use pdf::{Dictionary, Document, Object};

use super::budget::TintBudget;
use super::decoded::Decoded;
use super::{ImageColour, Layout, Samples};
use crate::ColourTarget;
use crate::resources::colorspace::{ColorSpace, decoded_sample};
use crate::resources::image::ImageDescriptor;
use crate::test_helpers::{make_doc_with_objects, make_doc_with_stream};

pub(super) fn name_obj(n: &[u8]) -> Object {
    Object::Name(n.to_vec())
}

pub(super) fn decode_arr(values: &[i64]) -> Object {
    Object::Array(values.iter().map(|&v| Object::Integer(v)).collect())
}

/// Convert `data` of `layout` as an image whose dictionary holds
/// `/ColorSpace cs` and `extras`.
pub(super) fn convert(
    doc: &Document,
    cs: Object,
    extras: &[(&str, Object)],
    layout: Layout,
    data: &[u8],
) -> ImageDescriptor {
    convert_in(doc, cs, extras, layout, data, None).expect("converts")
}

/// [`convert`], drawing on `budget`; `None` when conversion fails.
pub(super) fn convert_in(
    doc: &Document,
    cs: Object,
    extras: &[(&str, Object)],
    layout: Layout,
    data: &[u8],
    budget: Option<&TintBudget>,
) -> Option<ImageDescriptor> {
    let width = pixels(data.len(), layout);
    let samples = Samples::cpu(width, 1, layout, data.to_vec());
    convert_samples(doc, cs, extras, samples, budget)
}

/// Convert the 16-bit `data` of `layout` as an image whose dictionary
/// holds `/ColorSpace cs` and `extras`.
pub(super) fn convert16_of(
    doc: &Document,
    cs: Object,
    extras: &[(&str, Object)],
    layout: Layout,
    data: &[u16],
) -> ImageDescriptor {
    let width = pixels(data.len(), layout);
    let samples = Samples::cpu16(width, 1, layout, data.to_vec());
    convert_samples(doc, cs, extras, samples, None).expect("converts")
}

/// Convert `samples` to sRGB for an 8-bit canvas, as an image whose
/// dictionary holds `/ColorSpace cs` and `extras`.
fn convert_samples(
    doc: &Document,
    cs: Object,
    extras: &[(&str, Object)],
    samples: Samples,
    budget: Option<&TintBudget>,
) -> Option<ImageDescriptor> {
    let mut dict = Dictionary::new();
    dict.set("ColorSpace", cs);
    for (k, v) in extras {
        dict.set(*k, v.clone());
    }
    ImageColour::resolve(doc, &dict, None).convert(
        doc,
        (ColourTarget::SRGB, color::SampleDepth::Eight),
        samples,
        budget,
        #[cfg(feature = "gpu-icc")]
        &crate::resources::image::IccHandles::none(),
    )
}

/// The pixel count of `len` samples of `layout` in one row.
fn pixels(len: usize, layout: Layout) -> u32 {
    let n = match layout {
        Layout::Mask => 1,
        Layout::Color(n) => usize::from(n),
    };
    u32::try_from(len / n).unwrap()
}

/// `/Decode [0 1 …]` for up to nine components.
const UNIT_DECODE: [(f64, f64); 9] = [(0.0, 1.0); 9];

/// The conversion of `n`-component samples in `space` under the unit
/// `/Decode`, with no page budget.
pub(super) fn unit_decoded<'a>(doc: &'a Document, space: &'a ColorSpace, n: usize) -> Decoded<'a> {
    Decoded::new(
        doc,
        ColourTarget::SRGB,
        space,
        &UNIT_DECODE[..n],
        255.0,
        None,
        None,
    )
}

pub(super) fn cal(family: &str, gamma: &str) -> Object {
    pdf::parse_object(
        format!("[/{family} << /WhitePoint [0.9505 1.0 1.089] /Gamma {gamma} >>]").as_bytes(),
        &mut 0,
    )
    .unwrap()
}

/// Two-colorant `[/Black /Spot]` `DeviceN` space whose Type 4 tint
/// transform adds Black to K and maps the spot to the teal CMYK
/// `(1, 0.08, 0.26, 0.38)` per unit tint.
pub(super) fn devicen_teal() -> (Document, Object) {
    const PROGRAM: &str = "{ exch 1 index 0.08 mul 2 index 0.26 mul 3 2 roll \
                           3 index 0.38 mul add dup 1 gt { pop 1 } if }";
    let body = format!(
        "<< /FunctionType 4 /Domain [0 1 0 1] /Range [0 1 0 1 0 1 0 1] /Length {} >>\n\
         stream\n{PROGRAM}\nendstream",
        PROGRAM.len()
    );
    let doc = make_doc_with_objects(&[(4, &body)]);
    let cs = Object::Array(vec![
        name_obj(b"DeviceN"),
        Object::Array(vec![name_obj(b"Black"), name_obj(b"Spot")]),
        name_obj(b"DeviceCMYK"),
        Object::Reference((4, 0)),
    ]);
    (doc, cs)
}

pub(super) fn teal_rgb() -> [u8; 3] {
    color::cmyk::cmyk_f64_to_rgb(1.0, 0.08, 0.26, 0.38)
}

/// A `DeviceN` space of `n` colorants converting to `DeviceCMYK`
/// through the Type 4 program `program`.
pub(super) fn devicen_cmyk(n: usize, program: &str) -> (Document, Object) {
    let domain = "0 1 ".repeat(n);
    let body = format!(
        "<< /FunctionType 4 /Domain [{domain}] /Range [0 1 0 1 0 1 0 1] /Length {} >>\n\
         stream\n{program}\nendstream",
        program.len()
    );
    let doc = make_doc_with_objects(&[(4, &body)]);
    let names = (0..n)
        .map(|i| name_obj(format!("Spot{i}").as_bytes()))
        .collect();
    let cs = Object::Array(vec![
        name_obj(b"DeviceN"),
        Object::Array(names),
        name_obj(b"DeviceCMYK"),
        Object::Reference((4, 0)),
    ]);
    (doc, cs)
}

/// `count` tuples of `n` samples from a xorshift generator.
pub(super) fn noise(n: usize, count: usize) -> Vec<u8> {
    let mut state = 0x9e37_79b9_u32;
    (0..n * count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state.to_le_bytes()[1]
        })
        .collect()
}

/// The exact colour of the sample tuple `px` in `space`.
pub(super) fn exact(doc: &Document, space: &ColorSpace, px: &[u8]) -> [u8; 3] {
    let comps: Vec<f64> = px
        .iter()
        .map(|&s| decoded_sample(s, (0.0, 1.0), 255.0))
        .collect();
    space.convert_to_rgb(doc, ColourTarget::SRGB, &comps)
}

/// A program of `n` inputs and four outputs, smooth in each, padded
/// with `pad` no-op pairs to raise its cost.
pub(super) fn smooth_program(n: usize, pad: usize) -> String {
    let fold = " 4 1 roll 0.7 mul 4 1 roll 0.8 mul 4 1 roll 0.9 mul 4 1 roll";
    let mut program = format!("{{{}", " 1 pop".repeat(pad));
    for _ in 4..n {
        program.push_str(" 0.5 mul add");
    }
    program.push_str(fold);
    program.push_str(" }");
    program
}

/// A document whose object 2 is an ICC profile stream with `/N n`,
/// gamma 1.8 (`u8Fixed8` 461) on sRGB primaries.
pub(super) fn gamma_1_8_doc(n: u8) -> Document {
    let gamma = moxcms::ToneReprCurve::Lut(vec![461]);
    let profile = if n == 1 {
        let mut p = moxcms::ColorProfile::new_gray_with_gamma(1.0);
        p.gray_trc = Some(gamma);
        p
    } else {
        let mut p = moxcms::ColorProfile::new_srgb();
        p.cicp = None;
        p.red_trc = Some(gamma.clone());
        p.green_trc = Some(gamma.clone());
        p.blue_trc = Some(gamma);
        p
    };
    make_doc_with_stream(&profile.encode().unwrap(), &format!(" /N {n}"))
}

pub(super) fn icc_cs() -> Object {
    Object::Array(vec![name_obj(b"ICCBased"), Object::Reference((2, 0))])
}
