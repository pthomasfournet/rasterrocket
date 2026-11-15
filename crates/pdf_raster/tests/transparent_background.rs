//! Rendering over a transparent backdrop: `RasterOptions::transparent_background`
//! yields RGBA pages whose alpha is the page's coverage × opacity, with
//! colour not premultiplied by it, and `RenderParams::transparent_background`
//! the matching RGB bitmap with an alpha plane.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use common::{one_page, render_one, stream_object};
use rasterrocket::{
    BackendPolicy, PixelFormat, RasterError, RasterOptions, RenderParams, RenderedPage,
    SessionConfig, open_session_from_bytes, render_page_rgb_with,
};

/// A 100 × 100 pt page, rendered at one pixel per point.
const W: u32 = 100;
const H: u32 = 100;

/// A page with `ExtGState` resources `gstates`, the form `/F` (object 5)
/// painting `form_content` as a transparency group with entries `group`,
/// and content `content`.
fn page(gstates: &str, content: &str, group: &str, form_content: &str) -> Vec<u8> {
    one_page(
        &format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W} {H}] \
             /Resources << /ExtGState << {gstates} >> /XObject << /F 5 0 R >> >> \
             /Contents 4 0 R >>"
        ),
        vec![
            stream_object("", content.as_bytes()),
            stream_object(
                &format!(
                    "/Type /XObject /Subtype /Form /BBox [0 0 {W} {H}] \
                     /Group << /S /Transparency /CS /DeviceRGB {group} >> \
                     /Resources << /ExtGState << {gstates} >> >>"
                ),
                form_content.as_bytes(),
            ),
        ],
    )
}

/// A page painting `content`, with `ExtGState` resources `gstates`.
fn simple(gstates: &str, content: &str) -> Vec<u8> {
    page(gstates, content, "", "")
}

/// Page 1 of `pdf` at 72 dpi over a transparent backdrop.
fn rgba(pdf: Vec<u8>) -> RenderedPage {
    let mut opts = RasterOptions::new(72.0);
    opts.transparent_background = true;
    let page = render_one(pdf, &opts).expect("page renders");
    assert_eq!(page.format, PixelFormat::Rgba8);
    assert_eq!(page.pixels.len(), (W * H * 4) as usize);
    page
}

/// RGBA of `page` at `(x, y)`, counted from the bottom-left corner.
fn at(page: &RenderedPage, x: u32, y: u32) -> [u8; 4] {
    let i = (((page.height - 1 - y) * page.width + x) * 4) as usize;
    page.pixels.as_u8().unwrap()[i..i + 4]
        .try_into()
        .expect("four bytes")
}

#[test]
fn unpainted_pixels_are_transparent_and_painted_ones_opaque() {
    let page = rgba(simple("", "1 0 0 rg 20 20 40 40 re f"));
    assert_eq!(at(&page, 40, 40), [255, 0, 0, 255], "inside the square");
    assert_eq!(at(&page, 5, 5), [0, 0, 0, 0], "blank page");
    assert_eq!(at(&page, 80, 80), [0, 0, 0, 0], "blank page");
}

/// A triangle whose hypotenuse crosses pixels diagonally: the pixels it
/// partly covers take partial alpha and keep the full fill colour, alpha
/// rising across the edge from the outside in.
#[test]
fn partial_coverage_at_the_edges_is_alpha_not_premultiplied_colour() {
    let page = rgba(simple("", "0 0 1 rg 20 20 m 80 20 l 20 80 l h f"));
    assert_eq!(at(&page, 30, 30), [0, 0, 255, 255], "interior");
    assert_eq!(at(&page, 70, 70), [0, 0, 0, 0], "outside");
    let edge: Vec<[u8; 4]> = page
        .pixels
        .as_u8()
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .copied()
        .filter(|px| px[3] != 0 && px[3] != 255)
        .collect();
    assert!(edge.len() >= 60, "{} partly covered pixels", edge.len());
    for px in &edge {
        assert_eq!(px[..3], [0, 0, 255], "edge colour is straight: {px:?}");
    }
    // Along the row y = 50 the hypotenuse x + y = 100 crosses x = 50.
    let row: Vec<u8> = (46..54).map(|x| at(&page, x, 50)[3]).collect();
    assert!(
        row.windows(2).all(|w| w[0] >= w[1]),
        "alpha falls outward: {row:?}"
    );
    assert!(
        row.iter().any(|&a| a != 0 && a != 255),
        "an edge pixel: {row:?}"
    );
}

/// Constant alpha leaves the colour as painted and halves the alpha.
#[test]
fn constant_alpha_is_carried_in_the_alpha_channel() {
    let page = rgba(simple(
        "/A << /ca 0.5 >>",
        "/A gs 0 1 0 rg 20 20 40 40 re f",
    ));
    let [r, g, b, a] = at(&page, 40, 40);
    assert_eq!([r, g, b], [0, 255, 0]);
    assert!(a.abs_diff(128) <= 1, "alpha {a}");
}

/// Two overlapping opaque squares in a group drawn at half opacity: the
/// group composites as one object, so the overlap has the same alpha as
/// either square alone, and the later square's colour.
#[test]
fn a_transparency_group_composites_as_one_object() {
    let form = "1 0 0 rg 20 20 40 40 re f 0 0 1 rg 40 40 40 40 re f";
    let page = rgba(page("/A << /ca 0.5 >>", "q /A gs /F Do Q", "/I true", form));
    let red = at(&page, 30, 30);
    let overlap = at(&page, 50, 50);
    let blue = at(&page, 70, 70);
    for (px, rgb, what) in [
        (red, [255, 0, 0], "red only"),
        (overlap, [0, 0, 255], "overlap"),
        (blue, [0, 0, 255], "blue only"),
    ] {
        assert_eq!(px[..3], rgb, "{what}");
        assert!(px[3].abs_diff(128) <= 1, "{what}: alpha {}", px[3]);
    }
    assert_eq!(at(&page, 90, 10), [0, 0, 0, 0], "outside the group");
}

/// A blend mode on the page sees only what the page painted: multiply over
/// the transparent backdrop paints the source colour, and over the grey
/// bar multiplies with it.
#[test]
fn a_blend_mode_blends_with_the_page_alone() {
    let page = rgba(simple(
        "/M << /BM /Multiply >>",
        "0.5 g 0 0 100 50 re f q /M gs 0 0 1 rg 25 25 50 50 re f Q",
    ));
    assert_eq!(at(&page, 50, 60), [0, 0, 255, 255], "over nothing");
    assert_eq!(at(&page, 50, 35), [0, 0, 128, 255], "over the grey bar");
    assert_eq!(at(&page, 10, 90), [0, 0, 0, 0], "unpainted");
}

/// The RGB bitmap of a transparent render has an alpha plane; the default
/// render has none and is unchanged by the option's existence.
#[test]
fn render_params_select_the_alpha_plane() {
    let pdf = simple("", "1 0 0 rg 20 20 40 40 re f");
    let session = open_session_from_bytes(pdf, &SessionConfig::with_policy(BackendPolicy::CpuOnly))
        .expect("open session");
    let opaque = render_page_rgb_with(&session, 1, &RenderParams::new(1.0)).expect("renders");
    assert!(!opaque.has_alpha());
    let clear = render_page_rgb_with(
        &session,
        1,
        &RenderParams::new(1.0).with_transparent_background(),
    )
    .expect("renders");
    let alpha = clear.alpha_plane().expect("an alpha plane");
    assert_eq!(alpha.len(), (W * H) as usize);
    assert_eq!(alpha[0], 0, "top-left corner is blank");
    assert_eq!(
        clear.row_bytes(H - 1 - 40)[40 * 3..41 * 3],
        opaque.row_bytes(H - 1 - 40)[40 * 3..41 * 3],
        "painted colour"
    );
}

#[test]
fn deskew_is_refused_with_a_transparent_background() {
    let mut opts = RasterOptions::new(72.0);
    opts.transparent_background = true;
    opts.deskew = true;
    assert!(matches!(
        render_one(simple("", ""), &opts),
        Err(RasterError::InvalidOptions(_))
    ));
}
