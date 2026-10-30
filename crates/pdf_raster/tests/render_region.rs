//! Region rendering, page geometry and cancellation through the session API.
//!
//! A region render draws the device pixels the whole-page render places in
//! that window, so tiles that cover a page stitch into the whole-page bitmap
//! byte for byte.

#[expect(dead_code, reason = "shared helpers; this test uses a subset")]
mod common;

use common::{SCALE, corpus_fixture, one_page, session, stream_object};
use rasterrocket::{
    BackendPolicy, CancelToken, RasterError, RasterSession, RenderParams, RenderRegion,
    SessionConfig, open_session, render_page_rgb, render_page_rgb_with,
};

/// Content exercising antialiased edges, strokes, a clip, a shading and
/// text: everything whose placement depends on the device transform.
const CONTENT: &[u8] = b"\
q 0.2 0.4 0.8 rg 13.3 7.7 m 187.1 52.9 l 61.4 93.2 l h f Q \
q 0.8 0 0 RG 2.7 w 1 J 5 95 m 195 5 l S Q \
q 30 20 140 60 re W n 1 0 0 rg 0 0 200 100 re f Q \
q 1 0 0 1 120 30 cm /Sh sh Q \
BT /F 17.3 Tf 21.1 41.7 Td (Tiles) Tj ET";

/// A page with a rotation and an off-origin crop box, content [`CONTENT`].
fn page_pdf(extra: &str) -> Vec<u8> {
    one_page(
        &format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] {extra} \
             /Resources << /Font << /F 5 0 R >> /Shading << /Sh 6 0 R >> >> \
             /Contents 4 0 R >>"
        ),
        vec![
            stream_object("", CONTENT),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
            b"<< /ShadingType 2 /ColorSpace /DeviceRGB /Coords [0 0 60 0] \
               /Function << /FunctionType 2 /Domain [0 1] /C0 [1 1 0] /C1 [0 0.5 1] /N 1 >> \
               /Extend [true true] >>"
                .to_vec(),
        ],
    )
}

/// Render `session`'s page 1 in a `cols × rows` grid of regions and stitch
/// them; returns the stitched RGB bytes and the page size.
fn stitched(session: &RasterSession, scale: f64, cols: u32, rows: u32) -> (u32, u32, Vec<u8>) {
    let (w, h) = session
        .page_geometry(1)
        .expect("page geometry")
        .size_px(scale, scale);
    let mut out = vec![0u8; w as usize * h as usize * 3];
    let cuts = |n: u32, len: u32| (0..=n).map(move |i| len * i / n).collect::<Vec<_>>();
    let (xs, ys) = (cuts(cols, w), cuts(rows, h));
    for pair_y in ys.windows(2) {
        for pair_x in xs.windows(2) {
            let region = RenderRegion::new(
                pair_x[0],
                pair_y[0],
                pair_x[1] - pair_x[0],
                pair_y[1] - pair_y[0],
            );
            let tile =
                render_page_rgb_with(session, 1, &RenderParams::new(scale).with_region(region))
                    .unwrap_or_else(|e| panic!("region {region:?}: {e}"));
            assert_eq!((tile.width, tile.height), (region.width, region.height));
            for row in 0..region.height {
                let src = &tile.row_bytes(row)[..region.width as usize * 3];
                let at = ((region.y + row) as usize * w as usize + region.x as usize) * 3;
                out[at..at + src.len()].copy_from_slice(src);
            }
        }
    }
    (w, h, out)
}

/// The whole-page render as tight RGB bytes.
fn whole(session: &RasterSession, scale: f64) -> (u32, u32, Vec<u8>) {
    let bmp = render_page_rgb(session, 1, scale).expect("whole page");
    let mut out = Vec::with_capacity(bmp.width as usize * bmp.height as usize * 3);
    for row in 0..bmp.height {
        out.extend_from_slice(&bmp.row_bytes(row)[..bmp.width as usize * 3]);
    }
    (bmp.width, bmp.height, out)
}

#[test]
fn tiles_compose_into_the_whole_page() {
    for extra in [
        "",
        "/Rotate 90",
        "/Rotate 180",
        "/Rotate 270",
        "/CropBox [17 9 181 97]",
        "/CropBox [17 9 181 97] /Rotate 90",
    ] {
        let s = session(page_pdf(extra), BackendPolicy::CpuOnly);
        let full = whole(&s, SCALE);
        let tiles = stitched(&s, SCALE, 3, 2);
        common::assert_identical(
            &full,
            &tiles,
            &format!("3×2 tiles of a page with `{extra}`"),
        );
    }
}

#[test]
fn corpus_pages_tile_exactly() {
    let Some(path) = corpus_fixture("corpus-04-ebook-mixed.pdf") else {
        return;
    };
    let s = open_session(&path, &SessionConfig::with_policy(BackendPolicy::CpuOnly))
        .expect("open corpus fixture");
    let scale = 50.0 / 72.0;
    let full = whole(&s, scale);
    let tiles = stitched(&s, scale, 4, 3);
    common::assert_identical(&full, &tiles, "corpus-04 page 1 in 4×3 tiles");
}

#[test]
fn anisotropic_scales_size_and_tile_the_page() {
    for extra in ["", "/Rotate 90"] {
        let s = session(page_pdf(extra), BackendPolicy::CpuOnly);
        let geom = s.page_geometry(1).expect("page geometry");
        let params = RenderParams::new_xy(3.0, 1.25);
        let bmp = render_page_rgb_with(&s, 1, &params).expect("anisotropic render");
        assert_eq!((bmp.width, bmp.height), geom.size_px(3.0, 1.25));
        let mut full = Vec::new();
        for row in 0..bmp.height {
            full.extend_from_slice(&bmp.row_bytes(row)[..bmp.width as usize * 3]);
        }
        let (w, h) = (bmp.width, bmp.height);
        let mut tiles = vec![0u8; full.len()];
        for (x, y, tw, th) in [(0, 0, w / 2, h), (w / 2, 0, w - w / 2, h)] {
            let tile = render_page_rgb_with(
                &s,
                1,
                &params.clone().with_region(RenderRegion::new(x, y, tw, th)),
            )
            .expect("anisotropic region");
            for row in 0..th {
                let src = &tile.row_bytes(row)[..tw as usize * 3];
                let at = ((y + row) as usize * w as usize + x as usize) * 3;
                tiles[at..at + src.len()].copy_from_slice(src);
            }
        }
        common::assert_identical(&(w, h, full), &(w, h, tiles), extra);
    }
}

#[test]
fn page_geometry_reports_boxes_and_pixel_size() {
    let s = session(
        page_pdf("/CropBox [17 9 181 97] /Rotate 90"),
        BackendPolicy::CpuOnly,
    );
    let geom = s.page_geometry(1).expect("page geometry");
    assert_eq!(geom.rotate_cw, 90);
    assert_eq!(geom.media_box, Some([0.0, 0.0, 200.0, 100.0]));
    assert_eq!(geom.crop_box, Some([17.0, 9.0, 181.0, 97.0]));
    assert_eq!((geom.origin_x, geom.origin_y), (17.0, 9.0));
    // Rotated 90°: the 164 × 88 pt crop box is 88 pt wide on the device.
    assert_eq!((geom.width_pts, geom.height_pts), (88.0, 164.0));
    let bmp = render_page_rgb(&s, 1, 2.0).expect("render");
    assert_eq!((bmp.width, bmp.height), geom.size_px(2.0, 2.0));
    assert!(matches!(
        s.page_geometry(2),
        Err(RasterError::PageOutOfRange { page: 2, total: 1 })
    ));
}

#[test]
fn a_region_past_the_page_or_empty_is_rejected() {
    let s = session(page_pdf(""), BackendPolicy::CpuOnly);
    let (w, h) = s.page_geometry(1).expect("geometry").size_px(1.0, 1.0);
    for region in [
        RenderRegion::new(w - 10, 0, 11, 10),
        RenderRegion::new(0, h, 1, 1),
        RenderRegion::new(0, 0, 0, 5),
        RenderRegion::new(u32::MAX, 0, 2, 2),
    ] {
        let got = render_page_rgb_with(&s, 1, &RenderParams::new(1.0).with_region(region));
        assert!(
            matches!(got, Err(RasterError::InvalidOptions(_))),
            "{region:?}: {:?}",
            got.map(|bmp| (bmp.width, bmp.height))
        );
    }
}

#[test]
fn the_size_limit_applies_to_the_region_not_the_page() {
    // At this scale the page is 40 000 px wide, past the per-side limit.
    let scale = 200.0;
    let s = session(page_pdf(""), BackendPolicy::CpuOnly);
    assert!(matches!(
        render_page_rgb(&s, 1, scale),
        Err(RasterError::PageTooLarge { .. })
    ));
    let region = RenderRegion::new(20_000, 10_000, 64, 32);
    let tile = render_page_rgb_with(&s, 1, &RenderParams::new(scale).with_region(region))
        .expect("a small region of an oversized page renders");
    assert_eq!((tile.width, tile.height), (64, 32));
}

#[test]
fn a_cancelled_token_stops_the_render() {
    let s = session(page_pdf(""), BackendPolicy::CpuOnly);
    let token = CancelToken::new();
    let params = RenderParams::new(SCALE).with_cancel(token.clone());
    assert!(
        render_page_rgb_with(&s, 1, &params).is_ok(),
        "an uncancelled token renders"
    );
    token.cancel();
    assert!(matches!(
        render_page_rgb_with(&s, 1, &params),
        Err(RasterError::Cancelled)
    ));
}
