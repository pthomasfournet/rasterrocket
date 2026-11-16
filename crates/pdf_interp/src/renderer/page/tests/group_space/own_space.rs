//! Images, shadings and patterns given in the space of a `CalRGB` or ICC
//! RGB transparency group paint their components on the group's canvas,
//! as fills of the space do, so a colour of the space outside the working
//! space composites unclipped.

use super::*;

/// The Adobe RGB group, isolated.
const ADOBE: &str = "/CS [/ICCBased 6 0 R] /I true";

/// An orange of the group's space, outside sRGB, filling the page.
const FILL: &str = "/Adobe cs 1 0.2 0 sc 0 0 10 10 re f";

/// The same orange painted by an image, an `Indexed` image, a 16-bit image
/// in another stream of the profile, a shading and a tiling pattern.
const PAINTS: [&str; 6] = [
    "q 10 0 0 10 0 0 cm /Orange Do Q",
    "q 10 0 0 10 0 0 cm /Index Do Q",
    "q 10 0 0 10 0 0 cm /Orange16 Do Q",
    "/Adobe2 cs 1 0.2 0 sc 0 0 10 10 re f",
    "/Sh sh",
    "/Pattern cs /P scn 0 0 10 10 re f",
];

/// `paint` in `Multiply` over a green of the group's space.
fn multiplied(paint: &str) -> String {
    format!("/Adobe cs 0.5 0.8 0.5 sc 0 0 10 10 re f /Mul gs {paint}")
}

/// The largest difference between two pages' samples.
fn off(got: &[u16], want: &[u16]) -> u16 {
    got.iter()
        .zip(want)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0)
}

#[test]
fn a_paint_of_the_groups_space_blends_its_components() {
    for request in [CanvasRequest::Rgb, INK] {
        for depth in [SampleDepth::Eight, SampleDepth::Sixteen] {
            let want = grouped(ADOBE, &multiplied(FILL), &request, depth);
            // An orange of sRGB, which blends as the orange clipped by a
            // conversion through the working space does.
            let clipped = grouped(
                ADOBE,
                &multiplied("1 0.2 0 rg 0 0 10 10 re f"),
                &request,
                depth,
            );
            let level = if depth == SampleDepth::Eight { 1 } else { 257 };
            assert!(
                off(&clipped, &want) > 15 * level,
                "{request:?} {depth:?}: the sRGB orange blends apart"
            );
            for paint in PAINTS {
                let got = grouped(ADOBE, &multiplied(paint), &request, depth);
                assert!(
                    off(&got, &want) <= level,
                    "{request:?} {depth:?} {paint}: {:?} vs {:?}",
                    &got[..3],
                    &want[..3]
                );
            }
        }
    }
}

#[test]
fn a_paint_drawn_on_the_page_first_blends_its_components_in_the_group() {
    // The page draws the shading and the pattern through a form, under
    // the matrix the group draws them under.
    let page_paint = |paint: &'static str| match paint {
        "/Sh sh" => "/Shaded Do",
        "/Pattern cs /P scn 0 0 10 10 re f" => "/Tiled Do",
        other => other,
    };
    for request in [CanvasRequest::Rgb, INK] {
        for paint in PAINTS {
            let group = doc(ADOBE, &multiplied(paint));
            let alone = render_page(&group, &request, SampleDepth::Eight, "/G Do").0;
            let after = render_page(
                &group,
                &request,
                SampleDepth::Eight,
                &format!("{} /G Do", page_paint(paint)),
            )
            .0;
            assert_eq!(
                samples(&after),
                samples(&alone),
                "{request:?} {paint}: the page's paint is not the group's"
            );
        }
    }
}
