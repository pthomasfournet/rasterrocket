//! `SessionConfig::rotation` (and `RasterOptions::rotation`) turns each
//! page in the page transform: every render entry point produces the turned
//! page's size, and a page whose edges fall on whole device pixels renders
//! to exactly the unturned render's pixels, moved by a quarter-turn bitmap
//! rotation.
//!
//! Byte identity holds only for content whose rasterisation is invariant
//! under the reflections and transpositions of the device axes a quarter
//! turn applies.  These are not: an antialiased edge at a fractional sample
//! position (a span covers the sample cells from the floor of its left end
//! to the floor of its right end, which a reflection does not preserve); a
//! zero-width line (one pixel per scanline crossing, so it depends on which
//! device axis runs along the scanlines); text (a glyph origin keeps a
//! quarter-pixel fraction in device x and none in device y); and an
//! interpolated image (its per-axis source coordinates and blend weights are
//! truncated, which is not symmetric when the image axes trade device
//! axes).  [`EXACT`] therefore draws only fills, clips and a nearest-neighbour
//! image on whole pixels.  A page whose size in pixels is not whole is also
//! excluded: the fraction of a pixel by which the bitmap exceeds the page
//! lies past its right and bottom edges, whichever way it is turned.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use common::{corpus_fixture, one_page, render_one, stream_object};
use rasterrocket::{
    BackendPolicy, InvalidRotation, PageBuffer, PixelFormat, RasterOptions, RasterSession,
    RenderParams, RenderRegion, Rotation, SessionConfig, open_session, open_session_from_bytes,
    render_page_gray, render_page_rgb, render_page_rgb_with, render_page_rgba,
};

/// Fills, a clip and a nearest-neighbour image whose edges lie on whole
/// pixels at one and two pixels per point, on a 120 × 80 pt page.
const EXACT: &str = "0 g 10 10 30 20 re f 1 0 0 rg 70 5 40 60 re f 0 0 1 rg 2 50 6 28 re f \
    q 30 20 40 30 re W n 0 1 0 rg 0 0 120 80 re f Q \
    q 40 0 0 30 75 40 cm BI /W 4 /H 3 /CS /RGB /BPC 8 /F /AHx ID \
    ff0000 00ff00 0000ff ffff00 00ffff ff00ff 808080 202020 c0c0c0 ff8000 0080ff 80ff00> EI Q";

/// [`EXACT`] with antialiased diagonal edges, a stroke, a hairline, text and
/// an interpolated image: everything whose placement depends on the device
/// transform.
const RICH: &str = "0.2 0.4 0.8 rg 13.3 7.7 m 117.1 52.9 l 61.4 73.2 l h f \
    0.8 0 0 RG 2.7 w 1 J 5 75 m 115 5 l S 0 G 0 w 5 5 m 115 75 l S \
    q 31.3 6.9 -9.2 27.6 40.7 22.1 cm BI /W 4 /H 3 /CS /RGB /BPC 8 /I true /F /AHx ID \
    ff0000 00ff00 0000ff ffff00 00ffff ff00ff 808080 202020 c0c0c0 ff8000 0080ff 80ff00> EI Q \
    BT /F 17.3 Tf 21.1 41.7 Td (Turns) Tj ET";

const ROTATIONS: [Rotation; 4] = [
    Rotation::None,
    Rotation::Cw90,
    Rotation::Cw180,
    Rotation::Cw270,
];

/// A 120 × 80 pt page with `/Rotate rotate` drawing `content`.
fn page(rotate: u16, content: &str) -> Vec<u8> {
    one_page(
        &format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 120 80] /Rotate {rotate} \
             /Resources << /Font << /F 5 0 R >> >> /Contents 4 0 R >>"
        ),
        vec![
            stream_object("", content.as_bytes()),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
        ],
    )
}

fn config(rotation: Rotation) -> SessionConfig {
    let mut config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    config.rotation = rotation;
    config
}

fn open(pdf: Vec<u8>, rotation: Rotation) -> RasterSession {
    open_session_from_bytes(pdf, &config(rotation)).expect("opens")
}

/// An image: width, height and tightly packed pixels.
type Image = (u32, u32, Vec<u8>);

/// `img`, of `bpp` bytes per pixel, turned clockwise by `rotation`.
fn turn(img: &Image, bpp: usize, rotation: Rotation) -> Image {
    let (w, h) = (img.0 as usize, img.1 as usize);
    let (dw, dh) = if rotation.is_quarter() {
        (h, w)
    } else {
        (w, h)
    };
    let mut out = Vec::with_capacity(img.2.len());
    for dy in 0..dh {
        for dx in 0..dw {
            // The source pixel that lands at (dx, dy).
            let (sx, sy) = match rotation {
                Rotation::None => (dx, dy),
                Rotation::Cw90 => (dy, h - 1 - dx),
                Rotation::Cw180 => (w - 1 - dx, h - 1 - dy),
                Rotation::Cw270 => (w - 1 - dy, dx),
            };
            out.extend_from_slice(&img.2[(sy * w + sx) * bpp..][..bpp]);
        }
    }
    let side = |n: usize| u32::try_from(n).expect("a test page side fits u32");
    (side(dw), side(dh), out)
}

/// Assert `got` is byte-identical to `want`, naming the first difference.
fn assert_same(got: &Image, want: &Image, what: &str) {
    assert_eq!((got.0, got.1), (want.0, want.1), "{what}: size");
    if let Some(i) = got.2.iter().zip(&want.2).position(|(a, b)| a != b) {
        let differing = got.2.iter().zip(&want.2).filter(|(a, b)| a != b).count();
        panic!(
            "{what}: {differing} bytes differ, the first at byte {i} ({} vs {})",
            got.2[i], want.2[i]
        );
    }
}

fn rgb(session: &RasterSession, scale: f64) -> Image {
    let bmp = render_page_rgb(session, 1, scale).expect("renders");
    (bmp.width, bmp.height, bmp.data().to_vec())
}

fn gray(session: &RasterSession, dpi: f32) -> Image {
    let page = render_page_gray(session, 1, &RasterOptions::new(dpi)).expect("renders");
    assert_eq!(page.format, PixelFormat::Gray8);
    (page.width, page.height, page.pixels)
}

fn rgba(session: &RasterSession, dpi: f32) -> Image {
    let page = render_page_rgba(session, 1, &RasterOptions::new(dpi)).expect("renders");
    assert_eq!(page.format, PixelFormat::Rgba8);
    (page.width, page.height, page.pixels)
}

#[test]
fn turned_renders_are_the_unturned_render_turned() {
    for dpi in [72.0_f32, 144.0] {
        let scale = f64::from(dpi) / 72.0;
        for rotate in [0, 90] {
            let unturned = open(page(rotate, EXACT), Rotation::None);
            let base_rgb = rgb(&unturned, scale);
            let base_gray = gray(&unturned, dpi);
            let base_rgba = rgba(&unturned, dpi);
            for rotation in ROTATIONS {
                let what = format!("{dpi} dpi, /Rotate {rotate}, {rotation:?}");
                let turned = open(page(rotate, EXACT), rotation);
                assert_same(
                    &rgb(&turned, scale),
                    &turn(&base_rgb, 3, rotation),
                    &format!("{what}, RGB"),
                );
                assert_same(
                    &gray(&turned, dpi),
                    &turn(&base_gray, 1, rotation),
                    &format!("{what}, gray"),
                );
                assert_same(
                    &rgba(&turned, dpi),
                    &turn(&base_rgba, 4, rotation),
                    &format!("{what}, transparent RGBA"),
                );
            }
        }
    }
}

#[test]
fn geometry_and_page_size_px_are_the_turned_page() {
    for rotate in [0_u16, 90] {
        for rotation in ROTATIONS {
            let session = open(page(rotate, RICH), rotation);
            let geometry = session.page_geometry(1).expect("geometry");
            let total = (rotate + rotation.degrees()) % 360;
            assert_eq!(geometry.rotate_cw, total, "/Rotate {rotate}, {rotation:?}");
            let upright = total % 180 == 0;
            let want = if upright {
                (120.0, 80.0)
            } else {
                (80.0, 120.0)
            };
            assert_eq!((geometry.width_pts, geometry.height_pts), want);
            // 150 dpi: 250 × 166.67 px, rounded as the renderer rounds it.
            let size = session.page_size_px(1, 150.0).expect("size");
            let want = if upright { (250, 167) } else { (167, 250) };
            assert_eq!(size, want, "/Rotate {rotate}, {rotation:?}");
            let rendered = gray(&session, 150.0);
            assert_eq!((rendered.0, rendered.1), size);
            let rendered = rgb(&session, 150.0 / 72.0);
            assert_eq!((rendered.0, rendered.1), size);
        }
    }
}

/// The window `r` of `img`, of `bpp` bytes per pixel.
fn window(img: &Image, bpp: usize, r: RenderRegion) -> Image {
    let row = img.0 as usize * bpp;
    let mut out = Vec::new();
    for y in r.y..r.y + r.height {
        let start = y as usize * row + r.x as usize * bpp;
        out.extend_from_slice(&img.2[start..start + r.width as usize * bpp]);
    }
    (r.width, r.height, out)
}

/// A region is placed on the turned page: it is the window of the turned
/// whole-page render.
fn assert_regions_on_turned_page(session: &RasterSession, scale: f64, what: &str) {
    let whole = rgb(session, scale);
    let (w, h) = (whole.0, whole.1);
    for r in [
        RenderRegion::new(0, 0, w / 2, h / 3),
        RenderRegion::new(w / 3, h / 2, w - w / 3, h - h / 2),
        RenderRegion::new(w - 7, 3, 7, h - 3),
    ] {
        let mut params = RenderParams::new(scale);
        params.region = Some(r);
        let bmp = render_page_rgb_with(session, 1, &params).expect("region renders");
        assert_same(
            &(bmp.width, bmp.height, bmp.data().to_vec()),
            &window(&whole, 3, r),
            &format!("{what}, region {r:?}"),
        );
    }
    let mut params = RenderParams::new(scale);
    params.region = Some(RenderRegion::new(0, 0, w + 1, 1));
    assert!(
        render_page_rgb_with(session, 1, &params).is_err(),
        "{what}: a region past the turned page's right edge is rejected"
    );
}

#[test]
fn regions_are_placed_on_the_turned_page() {
    for rotate in [0, 90] {
        for rotation in ROTATIONS {
            let session = open(page(rotate, RICH), rotation);
            assert_regions_on_turned_page(
                &session,
                150.0 / 72.0,
                &format!("/Rotate {rotate}, {rotation:?}"),
            );
        }
    }
}

#[test]
fn page_buffer_renders_the_turned_page() {
    let mut buffer = PageBuffer::new();
    for rotation in ROTATIONS {
        let session = open(page(90, RICH), rotation);
        let opts = RasterOptions::new(150.0);
        let (w, h) = session.page_size_px(1, 150.0).expect("size");
        let mut out = vec![0; w as usize * h as usize * 4];

        let info = buffer
            .render_gray_into(&session, 1, &opts, &mut out, w as usize)
            .expect("gray renders");
        assert_eq!((info.width, info.height), (w, h), "{rotation:?}");
        let got = (w, h, out[..w as usize * h as usize].to_vec());
        assert_same(&got, &gray(&session, 150.0), &format!("{rotation:?}, gray"));

        let info = buffer
            .render_rgb_into(&session, 1, &opts, &mut out, w as usize * 3)
            .expect("RGB renders");
        assert_eq!((info.width, info.height), (w, h), "{rotation:?}");
        let got = (w, h, out[..w as usize * h as usize * 3].to_vec());
        assert_same(
            &got,
            &rgb(&session, 150.0 / 72.0),
            &format!("{rotation:?}, RGB"),
        );

        let mut transparent = opts.clone();
        transparent.transparent_background = true;
        let info = buffer
            .render_rgba_into(&session, 1, &transparent, &mut out, w as usize * 4)
            .expect("RGBA renders");
        assert_eq!((info.width, info.height), (w, h), "{rotation:?}");
        let got = (w, h, out.clone());
        assert_same(&got, &rgba(&session, 150.0), &format!("{rotation:?}, RGBA"));
    }
}

#[test]
fn deskew_straightens_the_turned_page_and_keeps_its_size() {
    for rotation in ROTATIONS {
        let session = open(page(0, RICH), rotation);
        let mut opts = RasterOptions::new(150.0);
        opts.deskew = true;
        let page = render_page_gray(&session, 1, &opts).expect("renders with deskew");
        assert_eq!(
            Ok((page.width, page.height)),
            session.page_size_px(1, 150.0).map_err(|e| e.to_string()),
            "{rotation:?}"
        );
    }
}

#[test]
fn raster_options_turn_the_pages_of_the_one_call_entry_points() {
    let mut opts = RasterOptions::new(72.0);
    assert_eq!(opts.rotation, Rotation::None, "the default");
    let base = render_one(page(90, EXACT), &opts).expect("renders");
    let base = (base.width, base.height, base.pixels);
    for rotation in ROTATIONS {
        opts.rotation = rotation;
        let got = render_one(page(90, EXACT), &opts).expect("renders");
        assert_same(
            &(got.width, got.height, got.pixels),
            &turn(&base, 1, rotation),
            &format!("{rotation:?}"),
        );
    }
    opts.rotation = Rotation::Cw270;
    opts.transparent_background = true;
    let got = render_one(page(0, EXACT), &opts).expect("renders");
    assert_eq!(
        (got.width, got.height, got.format),
        (80, 120, PixelFormat::Rgba8)
    );
}

#[test]
fn rotations_are_whole_quarter_turns() {
    assert_eq!(Rotation::from_degrees(-90), Ok(Rotation::Cw270));
    assert_eq!(Rotation::from_degrees(810), Ok(Rotation::Cw90));
    let err = Rotation::from_degrees(45).expect_err("45° is rejected");
    assert_eq!(err, InvalidRotation { degrees: 45 });
    assert_eq!(
        err.to_string(),
        "rotation 45° is not a multiple of 90 degrees"
    );
}

/// The corpus fixtures turned every way: sizes match `page_size_px` and
/// exchange on quarter turns, and regions land on the turned page.
#[test]
fn corpus_pages_turn() {
    for name in [
        "corpus-01-native-text-small.pdf",
        "corpus-02-native-vector-text.pdf",
        "corpus-03-native-text-dense.pdf",
        "corpus-04-ebook-mixed.pdf",
        "corpus-05-academic-book.pdf",
        "corpus-06-modern-layout-dct.pdf",
        "corpus-07-journal-dct-heavy.pdf",
        "corpus-08-scan-dct-1927.pdf",
        "corpus-09-scan-dct-1836.pdf",
        "corpus-10-scan-jbig2-jpx.pdf",
    ] {
        let Some(path) = corpus_fixture(name) else {
            continue;
        };
        let mut upright = None;
        for rotation in ROTATIONS {
            let session = open_session(&path, &config(rotation)).expect("opens");
            let size = session.page_size_px(1, 50.0).expect("size");
            let page = gray(&session, 50.0);
            assert_eq!((page.0, page.1), size, "{name} {rotation:?}");
            let (w, h) = *upright.get_or_insert(size);
            let want = if rotation.is_quarter() {
                (h, w)
            } else {
                (w, h)
            };
            assert_eq!(size, want, "{name} {rotation:?}");
            if rotation == Rotation::Cw90 {
                assert_regions_on_turned_page(&session, 50.0 / 72.0, name);
            }
        }
    }
}
