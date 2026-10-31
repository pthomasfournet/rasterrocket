//! The fill and stroke of one `B` / `b` operator form one object whose
//! stroke knocks out the fill (ISO 32000-2 §11.7.4.4): a semi-transparent
//! stroke composites with the backdrop under the path, not over the fill.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use common::{rgb_at, stream_object, with_page_tree};

/// A 100 × 100 pt page, rendered at one pixel per point.
const W: u32 = 100;
const H: u32 = 100;

/// A white page running `content` with the `ExtGState` `/H` (alpha 0.5).
fn page(content: &str) -> Vec<u8> {
    with_page_tree(&[
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W} {H}] \
             /Resources << /ExtGState << /H << /CA 0.5 /ca 0.5 >> >> >> \
             /Contents 4 0 R >>"
        )
        .into_bytes(),
        stream_object("", content.as_bytes()),
    ])
}

/// Half-alpha red over white.
const PINK: [u8; 3] = [255, 127, 127];
/// Half-alpha blue over white.
const LIGHT_BLUE: [u8; 3] = [127, 127, 255];

/// A 10 pt red stroke at half alpha around a blue-filled square: the
/// stroke's inner half, which overlaps the fill, shows the same colour as
/// its outer half, and the fill shows inside the stroke.
#[test]
fn a_translucent_stroke_knocks_out_the_fill() {
    let pdf = page("/H gs 10 w 0 0 1 rg 1 0 0 RG 20 20 60 60 re B");
    let [outer, inner, fill, page] = rgb_at(pdf, &[(17, 50), (22, 50), (50, 50), (5, 5)])[..]
    else {
        unreachable!("four points sampled");
    };
    assert_eq!(outer, PINK, "outer half of the stroke");
    assert_eq!(inner, PINK, "inner half of the stroke knocks out the fill");
    assert_eq!(fill, LIGHT_BLUE, "fill inside the stroke");
    assert_eq!(page, [255, 255, 255], "page outside the path");
}
