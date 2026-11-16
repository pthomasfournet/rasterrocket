//! A page composites in its transparency group's colour space (ISO 32000-2
//! §11.4.7): a page whose group declares a CMYK space is painted in ink
//! and finished through the process-ink model, a page whose group declares
//! gray finishes gray, and a page whose group declares sRGB composites in
//! sRGB where the wide working space would be chosen.  A page declaring
//! no space, or its canvas's own, renders as before.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use common::{rgb_at, stream_object, with_page_tree};
use rasterrocket::{
    BackendPolicy, FinishedCanvas, OutputProfile, RenderParams, SampleDepth, SessionConfig,
    WorkingSpace, open_session_from_bytes, render_page_canvas,
};

/// A 10 × 10 pt page running `content`, its `/Group` entries `group`
/// (none when empty), `/Half` a fill alpha of one half and `/F` a form
/// (object 5) whose transparency group of entries `form_group` draws
/// `form`.
fn page(group: &str, content: &str, form_group: &str, form: &str) -> Vec<u8> {
    let group = if group.is_empty() {
        String::new()
    } else {
        format!("/Group << /S /Transparency {group} >>")
    };
    let res = "/ExtGState << /Half << /ca 0.5 >> >>";
    with_page_tree(&[
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 10 10] {group} \
             /Resources << {res} /XObject << /F 5 0 R >> >> /Contents 4 0 R >>"
        )
        .into_bytes(),
        stream_object("", content.as_bytes()),
        stream_object(
            &format!(
                "/Type /XObject /Subtype /Form /BBox [0 0 10 10] \
                 /Group << /S /Transparency {form_group} >> /Resources << {res} >>"
            ),
            form.as_bytes(),
        ),
    ])
}

/// Red under blue at half alpha.
const HALF: &str = "1 0 0 rg 0 0 10 10 re f /Half gs 0 0 1 rg 0 0 10 10 re f";

/// The centre pixel of `pdf`'s first page.
fn centre(pdf: Vec<u8>) -> [u8; 3] {
    rgb_at(pdf, &[(5, 5)])[0]
}

#[test]
fn a_cmyk_page_group_composites_in_ink() {
    let cmyk = centre(page("/CS /DeviceCMYK", HALF, "", ""));
    let none = centre(page("", HALF, "", ""));
    assert_ne!(cmyk, none, "blending in ink differs from blending in sRGB");
    // The same content in a CMYK form group over an sRGB page composites
    // alike, within the rounding of a transparent group canvas.
    let nested = centre(page("", "/F Do", "/CS /DeviceCMYK /I true", HALF));
    assert!(
        cmyk.iter().zip(&nested).all(|(a, b)| a.abs_diff(*b) <= 1),
        "{cmyk:?} vs {nested:?}"
    );
}

/// An opaque CMYK fill prints its own ink either way: its colour moves
/// only by the rounding of its 8-bit ink, within one level.
#[test]
fn an_opaque_cmyk_fill_keeps_its_colour_under_a_cmyk_page_group() {
    let fill = "0.1 0.7 0.2 0.05 k 0 0 10 10 re f";
    let cmyk = centre(page("/CS /DeviceCMYK", fill, "", ""));
    let none = centre(page("", fill, "", ""));
    assert!(
        cmyk.iter().zip(&none).all(|(a, b)| a.abs_diff(*b) <= 1),
        "{cmyk:?} vs {none:?}"
    );
}

#[test]
fn a_page_in_its_canvas_space_renders_as_one_without_a_space() {
    for content in [HALF, "/F Do"] {
        assert_eq!(
            centre(page("/CS /DeviceRGB", content, "/I true", HALF)),
            centre(page("", content, "/I true", HALF)),
            "{content}"
        );
    }
}

#[test]
fn a_gray_page_group_finishes_gray() {
    let red = "1 0 0 rg 0 0 10 10 re f";
    assert_eq!(centre(page("/CS /DeviceGray", red, "", "")), [76, 76, 76]);
}

/// A page whose group declares sRGB composites in sRGB at 16 bits under an
/// output profile, as a session asking for sRGB does; a page declaring
/// no space composites in the wide space.
#[test]
fn a_page_declaring_srgb_composites_in_srgb_under_a_profile() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/icc/printer-rgb-v4.icc"
    );
    let profile =
        std::sync::Arc::new(OutputProfile::from_bytes(std::fs::read(path).unwrap()).unwrap());
    let render = |pdf: Vec<u8>, space: Option<WorkingSpace>| {
        let mut config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
        config.output_profile = Some(std::sync::Arc::clone(&profile));
        config.working_space = space;
        let s = open_session_from_bytes(pdf, &config).unwrap();
        let params = RenderParams::new(1.0).with_depth(SampleDepth::Sixteen);
        match render_page_canvas(&s, 1, &params).unwrap() {
            FinishedCanvas::Rgb16(b) => b.data().to_vec(),
            other => panic!("an RGB page, not {other:?}"),
        }
    };
    let declared = page("/CS /DeviceRGB", HALF, "", "");
    let none = page("", HALF, "", "");
    assert_eq!(
        render(declared.clone(), None),
        render(declared.clone(), Some(WorkingSpace::Srgb))
    );
    assert_eq!(
        render(none.clone(), None),
        render(none, Some(WorkingSpace::Wide))
    );
    assert_ne!(
        render(declared.clone(), None),
        render(declared, Some(WorkingSpace::Wide))
    );
}

/// A page whose group composites in CMYK keeps its alpha over a
/// transparent background: a painted pixel is opaque, an unpainted one
/// transparent.
#[test]
fn a_cmyk_page_group_keeps_its_alpha_over_a_transparent_background() {
    let pdf = page("/CS /DeviceCMYK", "1 0 0 rg 0 0 5 10 re f", "", "");
    let s =
        open_session_from_bytes(pdf, &SessionConfig::with_policy(BackendPolicy::CpuOnly)).unwrap();
    let params = RenderParams::new(1.0).with_transparent_background();
    let FinishedCanvas::Rgb8(b) = render_page_canvas(&s, 1, &params).unwrap() else {
        panic!("an 8-bit RGB page");
    };
    let alpha = b.alpha_row(5).expect("an alpha plane");
    assert_eq!((alpha[2], alpha[8]), (255, 0));
}
