//! Spaces, colorants and documents the colour-space tests share.

use std::sync::Arc;

use pdf::{Document, Object};

use super::{ColorSpace, Colorant, resolve};
use crate::test_helpers::{empty_doc, make_doc_with_objects};
use crate::{ColourIntent, ColourTarget, WorkingSpace};

/// The intent of the conversions under test, the initial one.
pub(super) const RI: ColourTarget = ColourTarget::new(
    ColourIntent::new(crate::RenderingIntent::RelativeColorimetric, true),
    WorkingSpace::Srgb,
);

pub(super) fn name_obj(s: &[u8]) -> Object {
    Object::Name(s.to_vec())
}

/// A spot colorant.
pub(super) fn spot() -> Colorant {
    Colorant::parse(b"Spot")
}

/// `n` spot colorants.
pub(super) fn spots(n: usize) -> Arc<[Colorant]> {
    vec![spot(); n].into()
}

/// The colour space of the PDF object source `src`.
pub(super) fn parse_space(doc: &Document, src: &str) -> ColorSpace {
    let obj = pdf::parse_object(src.as_bytes(), &mut 0).expect("colour-space object parses");
    resolve(doc, &obj)
}

/// `[/Lab << /WhitePoint [0.9505 1 1.089] >>]`.
pub(super) fn lab_d65() -> ColorSpace {
    parse_space(&empty_doc(), "[/Lab << /WhitePoint [0.9505 1 1.089] >>]")
}

/// Type 4 tint transform of a two-colorant `[/Black /Spot]` space into
/// CMYK: Black adds to K; the spot is a teal `(1, 0.08, 0.26, 0.38)` per
/// unit tint.
const DEVICEN_TEAL_FN: &str = "{ exch 1 index 0.08 mul 2 index 0.26 mul 3 2 roll \
                               3 index 0.38 mul add dup 1 gt { pop 1 } if }";

/// A document whose object 4 is [`DEVICEN_TEAL_FN`].
pub(super) fn devicen_teal_doc() -> Document {
    let body = format!(
        "<< /FunctionType 4 /Domain [0 1 0 1] /Range [0 1 0 1 0 1 0 1] /Length {} >>\n\
         stream\n{DEVICEN_TEAL_FN}\nendstream",
        DEVICEN_TEAL_FN.len()
    );
    make_doc_with_objects(&[(4, &body)])
}
