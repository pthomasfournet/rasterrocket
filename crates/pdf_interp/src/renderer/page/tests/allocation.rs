//! Bitmaps a page render allocates beyond its canvas: a refusal stops the
//! render with [`RenderAbort::AllocationFailed`]; the knockout coverage
//! canvas covers the stroke's rectangle alone.

use pdf::Document;
use raster::path::Path;

use super::*;
use crate::resources::{ColorSpace, TransparencyGroupParams};

/// Page `(4, 0)` naming the transparency-group form `/F`, the `ExtGState`
/// `/M`, whose soft mask is the luminosity of the form `(6, 0)`, and the
/// `ExtGState` `/H` of stroke alpha one half.
fn group_and_mask_doc() -> Document {
    crate::test_helpers::make_doc_with_objects(&[
        (
            4,
            "<< /Type /Page /Resources << /XObject << /F 5 0 R >> \
             /ExtGState << /M << /SMask << /S /Luminosity /G 6 0 R >> >> \
             /H << /CA 0.5 >> >> >> >>"
                .to_owned(),
        ),
        (
            5,
            stream_object(
                "/Type /XObject /Subtype /Form /BBox [0 0 10 10] \
                 /Group << /S /Transparency >>",
                "0 g 0 0 10 10 re f",
            ),
        ),
        (
            6,
            stream_object(
                "/Type /XObject /Subtype /Form /BBox [0 0 10 10] \
                 /Group << /S /Transparency /CS /DeviceGray >>",
                "1 g 0 0 10 10 re f",
            ),
        ),
    ])
}

/// Run `content` on a 10-pixel-square page of [`group_and_mask_doc`] whose
/// group and mask canvases are refused; the abort and the page.
fn render_refusing_canvases(content: &[u8]) -> (Option<RenderAbort>, Vec<u8>) {
    let doc = group_and_mask_doc();
    let mut renderer = PageRenderer::new(10, 10, &doc, (4, 0));
    renderer.max_canvas_px = 0;
    renderer.execute(&crate::content::parse(content));
    let abort = renderer.budget_status().cloned();
    (abort, renderer.finish().0.data().to_vec())
}

const REFUSED: Option<RenderAbort> = Some(RenderAbort::AllocationFailed {
    width: 10,
    height: 10,
});

/// Run `content` on a 10-pixel-square page of [`group_and_mask_doc`],
/// knocking out within its page group when `knockout`, with every snapshot
/// of the canvas refused; the abort and the page.
fn render_refusing_snapshots(content: &[u8], knockout: bool) -> (Option<RenderAbort>, Vec<u8>) {
    let doc = group_and_mask_doc();
    let mut renderer = PageRenderer::new(10, 10, &doc, (4, 0));
    if knockout {
        let group = TransparencyGroupParams {
            isolated: true,
            knockout: true,
        };
        renderer.try_set_page_group(Some(group)).unwrap();
    }
    renderer.max_snapshot_px = 0;
    renderer.execute(&crate::content::parse(content));
    let abort = renderer.budget_status().cloned();
    (abort, renderer.finish().0.data().to_vec())
}

/// A snapshot refused under a soft mask, in a knockout group or for a
/// knocked-out fill and stroke stops the render with nothing painted.
#[test]
fn a_refused_snapshot_stops_the_render() {
    for (content, knockout) in [
        (&b"/M gs 0 g 0 0 10 10 re f 0 g 0 0 10 10 re f"[..], false),
        (b"0 g 0 0 10 10 re f", true),
        (b"/H gs 0 g 0 G 0 0 10 10 re B 0 g 0 0 10 10 re f", false),
    ] {
        let what = String::from_utf8_lossy(content);
        let (abort, page) = render_refusing_snapshots(content, knockout);
        assert_eq!(abort, REFUSED, "{what}");
        assert!(page.iter().all(|&b| b == 255), "{what}: nothing is painted");
    }
}

/// A transparency group whose bitmap is refused stops the render: neither
/// the group's content nor anything after it is painted.
#[test]
fn a_refused_group_bitmap_stops_the_render() {
    let (abort, page) = render_refusing_canvases(b"/F Do 0 g 0 0 10 10 re f");
    assert_eq!(abort, REFUSED);
    assert!(page.iter().all(|&b| b == 255), "nothing is painted");
}

/// A soft mask whose canvas is refused stops the render.
#[test]
fn a_refused_soft_mask_canvas_stops_the_render() {
    let (abort, page) = render_refusing_canvases(b"/M gs 0 g 0 0 10 10 re f");
    assert_eq!(abort, REFUSED);
    assert!(page.iter().all(|&b| b == 255), "nothing is painted");
}

/// The first refusal is the one reported.
#[test]
fn an_earlier_abort_is_kept() {
    let doc = group_and_mask_doc();
    let mut renderer = PageRenderer::new(10, 10, &doc, (4, 0));
    renderer.abort_allocation(raster::BitmapAllocError {
        width: 3,
        height: 4,
    });
    renderer.abort_allocation(raster::BitmapAllocError {
        width: 5,
        height: 6,
    });
    assert_eq!(
        renderer.budget_status(),
        Some(&RenderAbort::AllocationFailed {
            width: 3,
            height: 4
        })
    );
}

/// A page group whose canvas the allocator refuses is an error that leaves
/// the renderer as it was.
#[test]
fn a_refused_page_group_canvas_is_an_error() {
    let doc = group_and_mask_doc();
    let mut renderer = PageRenderer::new(4, 4, &doc, (4, 0));
    let group = TransparencyGroupParams {
        isolated: true,
        knockout: true,
    };
    renderer.try_set_page_group(Some(group)).unwrap();
    assert!(renderer.canvas.has_alpha());
    renderer.try_set_page_group(None).unwrap();
    assert!(!renderer.canvas.has_alpha());

    let (width, height) = (renderer.width, renderer.height);
    (renderer.width, renderer.height) = (u32::MAX, 1 << 20);
    let refused = renderer.try_set_page_group(Some(group)).unwrap_err();
    assert_eq!((refused.width, refused.height), (u32::MAX, 1 << 20));
    assert!(renderer.page_group.is_none());
    assert!(!renderer.group.knockout);
    assert_eq!(renderer.canvas.dims(), (width, height));
    assert!(!renderer.canvas.has_alpha());
}

/// The coverage of the stroke of `path` over `rect`, drawn white on a black
/// page-sized canvas: the reference the rectangle-sized canvas reproduces.
fn page_sized_coverage(
    r: &mut PageRenderer<'_>,
    path: &Path,
    line_width: f64,
    rect: PixelRect,
) -> Vec<u8> {
    r.gstate.save();
    {
        let gs = r.gstate.current_mut();
        gs.soft_mask = None;
        gs.stroke_alpha = u8::MAX;
        gs.blend_mode = BlendMode::Normal;
        gs.stroke_pattern = None;
        gs.stroke_color_space = ColorSpace::DeviceGray;
        gs.stroke_color = RasterColor::from_bytes([u8::MAX; 3]);
        gs.transfer = None;
    }
    let black = PageCanvas::Rgb(Bitmap::<Rgb8>::new(r.width, r.height, 1, false));
    let page = std::mem::replace(&mut r.canvas, black);
    r.stroke_within(path, line_width, Some(rect));
    let canvas = std::mem::replace(&mut r.canvas, page).into_rgb(&[None; 4]);
    r.gstate.restore();
    let (cx0, cx1) = (rect.x0 as usize * 3, rect.x1 as usize * 3);
    (rect.y0..rect.y1)
        .flat_map(|y| canvas.row_bytes(y)[cx0..cx1].iter().step_by(3).copied())
        .collect()
}

/// The knockout coverage of a `B` stroke, drawn on a canvas of the
/// stroke's rectangle with the device matrix and clip moved to its origin,
/// is the coverage a page-sized canvas gives over that rectangle: under
/// rotated, scaled and fractionally translated matrices, path and
/// rectangle clips, dashes, caps and joins, hairlines, with and without
/// antialiasing.
#[test]
fn knockout_coverage_is_the_page_sized_coverage() {
    const SETUPS: [&str; 6] = [
        "",
        "7.5 3.25 50 41.5 re W n",
        "4 4 m 60 9 l 41 61 l 3 30 l h W n 0.8 0.3 -0.3 0.8 12.37 7.61 cm",
        "q 10 10 m 54 12 l 30 58 l h W* n 1.7 0 0 0.6 3.1 9.9 cm",
        "2 4 m 62 3 l 60 62 l 1 50 l h W n 0 1.1 -1.4 0 61.3 2.2 cm",
        "0.31 0 0 0.31 0.49 0.77 cm",
    ];
    const STROKES: [&str; 4] = [
        "3 w 2 J 1 j 2 3 m 40 5 l 30 50 25 10 8 40 c",
        "0 w 5 5 m 58 31 l 12 57 l",
        "1.5 w 0 J 0 j 10 M [4 2] 1 d 6 6 m 50 8 l 20 44 l h",
        "6 w 1 J 2 j 30 30 m 31 30.5 l 50 50 m 20 52 l",
    ];
    let doc = crate::test_helpers::empty_doc();
    for antialias in [true, false] {
        for setup in SETUPS {
            for stroke in STROKES {
                let mut r = PageRenderer::new(64, 64, &doc, (1, 0));
                r.set_antialias(antialias, antialias);
                r.execute(&crate::content::parse(
                    format!("{setup} {stroke}").as_bytes(),
                ));
                let path = r.path.take().expect("the content builds a path").build();
                let line_width = r.min_stroke_width();
                let Some(rect) = r
                    .stroke_bounds(&path, line_width)
                    .and_then(|b| r.within_clip(b))
                else {
                    continue;
                };
                let want = page_sized_coverage(&mut r, &path, line_width, rect);
                let got = r.stroke_coverage(&path, line_width, rect).unwrap();
                assert!(
                    !antialias || want.iter().any(|&c| c > 0),
                    "{setup} {stroke}: the stroke covers its rectangle"
                );
                assert_eq!(got, want, "{setup} {stroke}, antialias {antialias}");
            }
        }
    }
}

/// A child renderer whose canvas is refused — a tiling pattern's cell, a
/// cached Type 3 glyph — stops the render with nothing painted.
#[test]
fn a_refused_child_canvas_stops_the_render() {
    let doc = crate::test_helpers::make_doc_with_objects(&[
        (
            4,
            "<< /Type /Page /Resources << /Pattern << /P 5 0 R >> \
             /Font << /T 6 0 R >> >> >>"
                .to_owned(),
        ),
        (
            5,
            stream_object(
                "/Type /Pattern /PatternType 1 /PaintType 1 /TilingType 1 \
                 /BBox [0 0 4 4] /XStep 4 /YStep 4 /Resources << >>",
                "0 g 0 0 2 2 re f",
            ),
        ),
        (
            6,
            "<< /Type /Font /Subtype /Type3 /FontBBox [0 0 1 1] \
             /FontMatrix [1 0 0 1 0 0] /CharProcs << /a 7 0 R >> \
             /Encoding << /Type /Encoding /Differences [97 /a] >> \
             /FirstChar 97 /LastChar 97 /Widths [1] >>"
                .to_owned(),
        ),
        (7, stream_object("", "1 0 0 0 1 1 d1 0 0 1 1 re f")),
    ]);
    for content in [
        &b"/Pattern cs /P scn 0 0 10 10 re f 0 g 0 0 10 10 re f"[..],
        b"BT /T 8 Tf 1 1 Td (a) Tj ET 0 g 0 0 10 10 re f",
    ] {
        let what = String::from_utf8_lossy(content);
        let mut renderer = PageRenderer::new(10, 10, &doc, (4, 0));
        renderer.max_canvas_px = 0;
        renderer.execute(&crate::content::parse(content));
        assert!(
            matches!(
                renderer.budget_status(),
                Some(RenderAbort::AllocationFailed { .. })
            ),
            "{what}: {:?}",
            renderer.budget_status()
        );
        let page = renderer.finish().0.data().to_vec();
        assert!(page.iter().all(|&b| b == 255), "{what}: nothing is painted");
    }
}
