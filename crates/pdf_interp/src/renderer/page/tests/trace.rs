//! The paint trace: its lines, and its independence from the window a
//! renderer draws.

use pdf::Document;

use super::*;
use crate::renderer::page::PageView;

/// Page `(4, 0)`, 64 × 64 pt: `/F1` is Helvetica, `/Fm` a transparency
/// group showing text, `/Sh` an axial shading, `/Masked` a luminosity
/// soft mask whose group is `/Fm`'s.
fn doc() -> Document {
    let form = "BT /F1 12 Tf 2 2 Td (MM) Tj ET";
    crate::test_helpers::make_doc_with_objects(&[
        (
            4,
            "<< /Type /Page /Resources << /Font << /F1 5 0 R >> \
             /XObject << /Fm 6 0 R >> /Shading << /Sh 7 0 R >> \
             /ExtGState << /Masked << /SMask << /S /Luminosity /G 6 0 R >> >> \
             /Half << /ca 0.5 >> >> >> >>"
                .to_owned(),
        ),
        (
            5,
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_owned(),
        ),
        (
            6,
            format!(
                "<< /Type /XObject /Subtype /Form /BBox [0 0 64 64] \
                 /Resources << /Font << /F1 5 0 R >> >> \
                 /Group << /S /Transparency /CS /DeviceGray >> /Length {} >>\n\
                 stream\n{form}\nendstream",
                form.len()
            ),
        ),
        (
            7,
            "<< /ShadingType 2 /ColorSpace /DeviceRGB /Coords [0 0 64 0] \
             /Function << /FunctionType 2 /Domain [0 1] /C0 [1 0 0] /C1 [0 0 1] /N 1 >> >>"
                .to_owned(),
        ),
    ])
}

/// Content painting with every traced operation.
const CONTENT: &[u8] = b"q 0 0 1 rg 10 10 20 5 re f Q \
    q 1 0 0 RG 2 w [3 1] 0 d 5 5 m 50 40 30 10 60 60 c S Q \
    q 8 8 40 40 re W n /Half gs 0.2 g 0 0 64 64 re f /Sh sh Q \
    q /Masked gs 0 g 4 30 50 20 re f Q \
    /Fm Do \
    BT /F1 10 Tf 1 0 0 1 3 50 Tm 1 Tr (Hi) Tj ET";

/// The trace of `CONTENT` drawn in the windows `windows` of the whole
/// 64 × 64 page, in order, each by its own renderer sharing the page's
/// state, the first recording the trace.
fn trace_in(windows: &[(u32, u32, u32, u32)]) -> String {
    let doc = doc();
    let ops = crate::content::parse(CONTENT);
    let page = PageRenderer::unit_view(64, 64);
    let mut state: Option<super::super::PageState> = None;
    for (i, &(x, y, w, h)) in windows.iter().enumerate() {
        let view: PageView = page.window(x, y, w, h);
        let mut renderer = PageRenderer::with_view(&view, &doc, (4, 0));
        if i == 0 {
            renderer.set_paint_trace();
        }
        if let Some(state) = state.take() {
            renderer.adopt_page_state(state);
        }
        renderer.execute(&ops);
        state = Some(renderer.try_finish_band(false).expect("finishes").1);
    }
    state
        .and_then(|mut s| s.take_paint_trace())
        .expect("the first window records the trace")
        .into_string()
}

#[test]
fn a_fill_prints_its_path_rule_and_paint() {
    let doc = doc();
    let mut renderer = PageRenderer::new(64, 64, &doc, (4, 0));
    renderer.set_paint_trace();
    renderer.execute(&crate::content::parse(b"0 0 1 rg 10 10 20 5 re f"));
    let trace = renderer.take_paint_trace().expect("recorded").into_string();
    let mut lines = trace.lines();
    assert_eq!(lines.next(), Some("rasterrocket-paint-trace 1"));
    assert_eq!(lines.next(), Some("page px=64x64 scale=1"));
    let fill = lines.next().expect("the fill's line");
    assert!(
        fill.starts_with("fill nz n=5 sub=1 cub=0 bbox=10,49,30,54 h="),
        "{fill}"
    );
    assert!(
        fill.contains(
            " pts=m10,54 l30,54 l30,49 l10,49 l10,54 z paint=DeviceRGB(0,0,1) ca=255 clip="
        ),
        "{fill}"
    );
    assert!(fill.ends_with(" clip=0000000000000000"), "{fill}");
    assert_eq!(lines.next(), None, "{trace}");
}

#[test]
fn every_traced_operation_is_recorded() {
    let trace = trace_in(&[(0, 0, 64, 64)]);
    for kind in [
        "fill nz ",
        "stroke ",
        "clip ",
        "sh /Sh(7 0 R)",
        "smask ",
        "form (6 0 R)",
        "form end",
        "text /F1(5 0 R)",
        "glyph c=48 sel=",
    ] {
        assert!(
            trace.lines().any(|l| l.trim_start().starts_with(kind)),
            "no {kind:?} line in\n{trace}"
        );
    }
    assert!(trace.contains(" dash=[3,1] phase=0"), "{trace}");
    assert!(trace.contains(" ca=128 "), "{trace}");
}

#[test]
fn the_trace_is_the_same_in_every_window() {
    let whole = trace_in(&[(0, 0, 64, 64)]);
    assert_eq!(trace_in(&[(0, 0, 64, 20), (0, 20, 64, 44)]), whole);
    assert_eq!(trace_in(&[(0, 0, 64, 1), (0, 1, 64, 63)]), whole);
    assert_eq!(trace_in(&[(30, 40, 10, 10)]), whole);
}

#[test]
fn a_later_window_records_nothing() {
    let first = trace_in(&[(0, 0, 64, 32)]);
    assert_eq!(trace_in(&[(0, 0, 64, 32), (0, 32, 64, 32)]), first);
}

#[test]
fn a_type3_glyph_description_is_not_recorded() {
    let charproc = "10 0 0 0 10 10 d1 0 0 10 10 re f";
    let doc = crate::test_helpers::make_doc_with_objects(&[
        (
            4,
            "<< /Type /Page /Resources << /Font << /T3 5 0 R >> >> >>".to_owned(),
        ),
        (
            5,
            "<< /Type /Font /Subtype /Type3 /FontBBox [0 0 10 10] \
             /FontMatrix [0.1 0 0 0.1 0 0] /CharProcs << /a 6 0 R >> \
             /Encoding << /Differences [97 /a] >> /FirstChar 97 /LastChar 97 \
             /Widths [10] >>"
                .to_owned(),
        ),
        (
            6,
            format!(
                "<< /Length {} >>\nstream\n{charproc}\nendstream",
                charproc.len()
            ),
        ),
    ]);
    let mut renderer = PageRenderer::new(64, 64, &doc, (4, 0));
    renderer.set_paint_trace();
    renderer.execute(&crate::content::parse(b"BT /T3 20 Tf 5 5 Td (aa) Tj ET"));
    let trace = renderer.take_paint_trace().expect("recorded").into_string();
    let lines: Vec<&str> = trace.lines().skip(2).collect();
    assert_eq!(lines.len(), 3, "{trace}");
    assert!(
        lines[0].starts_with("text /T3(5 0 R) size=20 Tr=0"),
        "{trace}"
    );
    assert_eq!(lines[1], "  glyph c=61 sel=97 pen=5,59");
    assert_eq!(lines[2], "  glyph c=61 sel=97 pen=25,59");
}
