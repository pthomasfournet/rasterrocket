//! Rendering at 16 bits per sample: smooth tones keep more levels than an
//! 8-bit render has, and a 16-bit render narrowed to 8 bits stays within
//! the 8-bit render's rounding of it.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use std::collections::BTreeSet;

use color::{Rgb8, Rgb16, Sample};
use common::gradients::{BANDS, WIDTH, gradient_page};
use raster::Bitmap;
use rasterrocket::{
    BackendPolicy, FinishedCanvas, PixelFormat, Pixels, RasterOptions, RasterSession, RenderParams,
    SampleDepth, SessionConfig, open_session_from_bytes, render_page_canvas, render_page_gray,
    render_page_rgba,
};

fn session(pdf: Vec<u8>) -> RasterSession {
    open_session_from_bytes(pdf, &SessionConfig::with_policy(BackendPolicy::CpuOnly))
        .expect("open session")
}

/// Page 1 of `session` at one pixel a point and samples of `depth`.
fn render(session: &RasterSession, depth: SampleDepth) -> FinishedCanvas {
    render_page_canvas(session, 1, &RenderParams::new(1.0).with_depth(depth)).expect("renders")
}

/// The number of distinct values of channel `c` among the RGB `samples`.
fn levels<S: Sample>(samples: &[S], c: usize) -> usize {
    samples
        .as_chunks::<3>()
        .0
        .iter()
        .map(|px| px[c])
        .collect::<BTreeSet<_>>()
        .len()
}

fn rgb16(canvas: FinishedCanvas) -> Bitmap<Rgb16> {
    match canvas {
        FinishedCanvas::Rgb16(b) => b,
        FinishedCanvas::Rgb8(_) => panic!("a 16-bit render"),
    }
}

fn rgb8(canvas: FinishedCanvas) -> Bitmap<Rgb8> {
    match canvas {
        FinishedCanvas::Rgb8(b) => b,
        FinishedCanvas::Rgb16(_) => panic!("an 8-bit render"),
    }
}

/// Each ramp of the gradient page has more than 256 levels in every
/// channel at 16 bits (the axial one a level for nearly every pixel of its
/// 2048), and at most 256 at 8 bits.
#[test]
fn a_16_bit_render_keeps_the_levels_of_smooth_tones() {
    let session = session(gradient_page());
    let wide = rgb16(render(&session, SampleDepth::Sixteen));
    let narrow = rgb8(render(&session, SampleDepth::Eight));
    assert_eq!((wide.width, narrow.width), (WIDTH, WIDTH));
    for (name, y) in BANDS {
        for c in 0..3 {
            let l16 = levels(wide.row_samples(y), c);
            let l8 = levels(narrow.row_samples(y), c);
            eprintln!("{name}, channel {c}: {l16} levels at 16 bits, {l8} at 8");
            assert!(l16 > 256, "{name}, channel {c}: {l16} levels at 16 bits");
            assert!(l8 <= 256, "{name}, channel {c}: {l8} levels at 8 bits");
        }
    }
    let axial = levels(wide.row_samples(BANDS[0].1), 0);
    assert!(
        axial >= 2000,
        "the axial ramp has {axial} levels over 2048 pixels"
    );
}

/// The largest difference, and the count of samples that differ, between
/// `narrow` and `wide` narrowed to 8 bits over the rows `rows`, per
/// channel.
fn narrowed_difference(
    wide: &Bitmap<Rgb16>,
    narrow: &Bitmap<Rgb8>,
    rows: std::ops::Range<u32>,
) -> [(u8, usize); 3] {
    let mut out = [(0u8, 0usize); 3];
    for y in rows {
        let w = wide.width as usize * 3;
        for (i, (&a, &b)) in wide.row_samples(y)[..w]
            .iter()
            .zip(&narrow.row_samples(y)[..w])
            .enumerate()
        {
            let d = a.to_u8().abs_diff(b);
            let (max, count) = &mut out[i % 3];
            *max = (*max).max(d);
            *count += usize::from(d != 0);
        }
    }
    out
}

/// The 16-bit render of the gradient page narrowed to 8 bits is the 8-bit
/// render to within one level in the axial and soft-mask bands.  In the
/// mesh band the 8-bit Gouraud interpolation itself runs up to 1.5 levels
/// below the exact colour beside the triangles' shared edge, where the
/// 16-bit one stays within 0.05 of a level: the bound there is two.
#[test]
fn a_16_bit_render_narrows_to_the_8_bit_render() {
    let session = session(gradient_page());
    let wide = rgb16(render(&session, SampleDepth::Sixteen));
    let narrow = rgb8(render(&session, SampleDepth::Eight));
    for (name, y) in BANDS {
        let table = narrowed_difference(&wide, &narrow, y - 100..y + 100);
        eprintln!("{name}: {table:?}");
        let bound = if name == "Gouraud mesh" { 2 } else { 1 };
        assert!(
            table.iter().all(|&(max, _)| max <= bound),
            "{name}: {table:?}"
        );
    }
}

/// A page of the paints whose 16-bit compositing differs from the 8-bit
/// code: a tiling pattern, a non-isolated group under a blend mode, a
/// knockout group, a transfer function, CIE-based and CMYK colours, a
/// stencil mask and an image at constant alpha.
fn paints_page() -> Vec<u8> {
    let content = b"0.9 0.8 0.5 rg 0 0 400 200 re f \
        q /Pattern cs /P0 scn 0 100 100 100 re f Q \
        q /GS0 gs /G0 Do Q \
        q /GS1 gs 0.2 0.4 0.6 rg 200 100 100 100 re f Q \
        q /Cal cs 0.3 0.6 0.2 sc 300 100 50 100 re f /Lab cs 60 20 -30 sc 350 100 50 100 re f Q \
        q 0.6 0.2 0.1 0.05 k 0 0 100 100 re f Q \
        q 0 0 1 rg 100 0 100 100 cm /Mask Do Q \
        q /GS2 gs 200 0 100 100 cm /Img Do Q \
        q /K0 Do Q";
    common::with_page_tree(&[
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 200] \
           /Resources << /Pattern << /P0 5 0 R >> /XObject << /G0 6 0 R /Mask 7 0 R \
           /Img 8 0 R /K0 9 0 R >> /ExtGState << /GS0 << /BM /Multiply /ca 0.7 >> \
           /GS1 << /TR << /FunctionType 2 /Domain [0 1] /C0 [0.1] /C1 [0.9] /N 1.7 >> >> \
           /GS2 << /ca 0.6 >> >> \
           /ColorSpace << /Cal [/CalRGB << /WhitePoint [0.9505 1 1.089] /Gamma [1.8 1.8 1.8] >>] \
           /Lab [/Lab << /WhitePoint [0.9505 1 1.089] >>] >> >> /Contents 4 0 R >>"
            .to_vec(),
        common::stream_object("", content),
        common::stream_object(
            "/Type /Pattern /PatternType 1 /PaintType 1 /TilingType 1 /BBox [0 0 20 20] \
             /XStep 20 /YStep 20 /Resources << >>",
            b"0.8 0.1 0.3 rg 0 0 10 10 re f 0.2 0.3 0.9 rg 10 10 10 10 re f",
        ),
        common::stream_object(
            "/Type /XObject /Subtype /Form /BBox [0 0 400 200] \
             /Group << /S /Transparency /I false >> \
             /Resources << /ExtGState << /S << /BM /Screen /ca 0.5 >> >> >>",
            b"0.3 0.7 0.2 rg 100 100 100 100 re f /S gs 0.9 0.2 0.6 rg 150 120 100 60 re f",
        ),
        common::stream_object(
            "/Type /XObject /Subtype /Image /Width 4 /Height 4 /ImageMask true \
             /BitsPerComponent 1 /Decode [1 0]",
            &[0xa0, 0x50, 0xa0, 0x50],
        ),
        common::stream_object(
            "/Type /XObject /Subtype /Image /Width 2 /Height 2 /ColorSpace /DeviceRGB \
             /BitsPerComponent 8 /Interpolate true",
            &[255, 0, 0, 0, 255, 0, 0, 0, 255, 250, 250, 20],
        ),
        common::stream_object(
            "/Type /XObject /Subtype /Form /BBox [300 0 400 100] \
             /Group << /S /Transparency /K true >> \
             /Resources << /ExtGState << /H << /ca 0.5 >> >> >>",
            b"/H gs 1 0 0 rg 300 0 70 70 re f 0 0 1 rg 330 30 70 70 re f",
        ),
    ])
}

/// The paints of [`paints_page`] at 16 bits, narrowed to 8, are the
/// 8-bit render's to within two levels: the 8-bit pipe's truncating
/// divides and the blend modes' own rounding at 8 bits.
#[test]
fn every_paint_narrows_to_its_8_bit_render() {
    let session = session(paints_page());
    let params = RenderParams::new(2.0);
    let wide = rgb16(
        render_page_canvas(
            &session,
            1,
            &params.clone().with_depth(SampleDepth::Sixteen),
        )
        .expect("16-bit render"),
    );
    let narrow = rgb8(render_page_canvas(&session, 1, &params).expect("8-bit render"));
    let table = narrowed_difference(&wide, &narrow, 0..wide.height);
    eprintln!("paints page: {table:?}");
    assert!(table.iter().all(|&(max, _)| max <= 2), "{table:?}");
}

/// A one-page PDF drawing the image `image` (a stream object body) over the
/// whole of its `w × h` point page.
fn image_page(w: u32, h: u32, image: Vec<u8>) -> Vec<u8> {
    common::with_page_tree(&[
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w} {h}] \
             /Resources << /XObject << /Im 5 0 R >> >> /Contents 4 0 R >>"
        )
        .into_bytes(),
        common::stream_object("", format!("q {w} 0 0 {h} 0 0 cm /Im Do Q").as_bytes()),
        image,
    ])
}

/// A 16-bit `DeviceRGB` Flate image of 1024 × 2 pixels, red rising from 0
/// to 65535 across it, green falling, blue at a third of red.
fn flate_ramp() -> Vec<u8> {
    use std::io::Write as _;
    let mut raw = Vec::new();
    for _ in 0..2 {
        for x in 0..1024u32 {
            let r = u16::try_from(x * 65_535 / 1023).unwrap();
            for v in [r, u16::MAX - r, r / 3] {
                raw.extend_from_slice(&v.to_be_bytes());
            }
        }
    }
    let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    z.write_all(&raw).unwrap();
    common::stream_object(
        "/Type /XObject /Subtype /Image /Width 1024 /Height 2 /ColorSpace /DeviceRGB \
         /BitsPerComponent 16 /Filter /FlateDecode",
        &z.finish().unwrap(),
    )
}

/// The 16-bit gray JPEG 2000 ramp of `tests/data/ramp16.jp2`, 1024 × 4,
/// lossless, as an image `XObject`.
fn jpx_ramp() -> Vec<u8> {
    common::stream_object(
        "/Type /XObject /Subtype /Image /Width 1024 /Height 4 /ColorSpace /DeviceGray \
         /BitsPerComponent 16 /Filter /JPXDecode",
        include_bytes!("data/ramp16.jp2"),
    )
}

/// A black 8-bit image of 1024 × 2 pixels under a 16-bit Flate soft mask
/// rising from transparent to opaque across it.
fn masked_page() -> Vec<u8> {
    use std::io::Write as _;
    let mut alpha = Vec::new();
    for _ in 0..2 {
        for x in 0..1024u32 {
            alpha.extend_from_slice(&u16::try_from(x * 65_535 / 1023).unwrap().to_be_bytes());
        }
    }
    let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    z.write_all(&alpha).unwrap();
    let mut pdf_objects = vec![
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 1024 2] \
           /Resources << /XObject << /Im 5 0 R >> >> /Contents 4 0 R >>"
            .to_vec(),
        common::stream_object("", b"q 1024 0 0 2 0 0 cm /Im Do Q"),
        common::stream_object(
            "/Type /XObject /Subtype /Image /Width 1024 /Height 2 /ColorSpace /DeviceRGB \
             /BitsPerComponent 8 /SMask 6 0 R",
            &vec![0u8; 1024 * 2 * 3],
        ),
    ];
    pdf_objects.push(common::stream_object(
        "/Type /XObject /Subtype /Image /Width 1024 /Height 2 /ColorSpace /DeviceGray \
         /BitsPerComponent 16 /Filter /FlateDecode",
        &z.finish().unwrap(),
    ));
    common::with_page_tree(&pdf_objects)
}

/// A 16-bit soft mask keeps its levels at 16 bits: the black image it
/// masks fades over the white page through more than 256 levels.
#[test]
fn a_16_bit_soft_mask_keeps_its_levels() {
    let session = session(masked_page());
    let wide = rgb16(render(&session, SampleDepth::Sixteen));
    let narrow = rgb8(render(&session, SampleDepth::Eight));
    let (l16, l8) = (
        levels(wide.row_samples(1), 0),
        levels(narrow.row_samples(1), 0),
    );
    eprintln!("soft-masked image: {l16} levels at 16 bits, {l8} at 8");
    assert!(l16 > 256 && l8 <= 256, "{l16} levels at 16 bits, {l8} at 8");
    let table = narrowed_difference(&wide, &narrow, 0..wide.height);
    assert!(table.iter().all(|&(max, _)| max <= 1), "{table:?}");
}

/// Images of 16 bits a sample keep their levels at 16 bits — a Flate RGB
/// ramp and a JPEG 2000 gray ramp, drawn one pixel a sample, show more
/// than 256 levels in every channel — and at 8 bits draw their high bytes
/// as before, at most 256.
#[test]
fn sixteen_bit_images_keep_their_levels() {
    for (name, pdf) in [
        ("Flate", image_page(1024, 2, flate_ramp())),
        ("JPX", image_page(1024, 4, jpx_ramp())),
    ] {
        let session = session(pdf);
        let wide = rgb16(render(&session, SampleDepth::Sixteen));
        let narrow = rgb8(render(&session, SampleDepth::Eight));
        for c in 0..3 {
            let l16 = levels(wide.row_samples(1), c);
            let l8 = levels(narrow.row_samples(1), c);
            eprintln!("{name} image, channel {c}: {l16} levels at 16 bits, {l8} at 8");
            assert!(l16 > 256, "{name}, channel {c}: {l16} levels at 16 bits");
            assert!(l8 <= 256, "{name}, channel {c}: {l8} levels at 8 bits");
        }
        let table = narrowed_difference(&wide, &narrow, 0..wide.height);
        eprintln!("{name} image narrowed: {table:?}");
        assert!(table.iter().all(|&(max, _)| max <= 1), "{name}: {table:?}");
    }
}

/// The batch pages of a 16-bit render: 16-bit gray, and 16-bit RGBA over
/// a transparent backdrop.
#[test]
fn batch_pages_carry_16_bit_samples() {
    let session = session(gradient_page());
    let mut opts = RasterOptions::new(72.0);
    opts.depth = SampleDepth::Sixteen;
    let gray = render_page_gray(&session, 1, &opts).expect("gray page");
    assert_eq!(gray.format, PixelFormat::Gray16);
    let Pixels::U16(samples) = &gray.pixels else {
        panic!("16-bit samples");
    };
    assert_eq!(samples.len(), (gray.width * gray.height) as usize);
    let row = &samples[(BANDS[0].1 * gray.width) as usize..][..gray.width as usize];
    assert!(row.iter().collect::<BTreeSet<_>>().len() > 256);
    let rgba = render_page_rgba(&session, 1, &opts).expect("RGBA page");
    assert_eq!(rgba.format, PixelFormat::Rgba16);
    assert_eq!(rgba.pixels.len(), (rgba.width * rgba.height * 4) as usize);
    assert_eq!(rgba.pixels.byte_len(), rgba.pixels.len() * 2);
}

/// A 16-bit render is refused where it cannot hold: by the 8-bit entry
/// points, under overprint simulation, and with deskew.
#[test]
fn a_16_bit_render_is_refused_where_it_cannot_hold() {
    let params = RenderParams::new(1.0).with_depth(SampleDepth::Sixteen);
    let plain = session(gradient_page());
    let Err(err) = rasterrocket::render_page_rgb_with(&plain, 1, &params) else {
        panic!("an 8-bit entry point renders 16-bit samples");
    };
    assert!(err.to_string().contains("render_page_canvas"), "{err}");
    let mut config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    config.overprint = true;
    let overprint = open_session_from_bytes(gradient_page(), &config).expect("open session");
    let err = render_page_canvas(&overprint, 1, &params).unwrap_err();
    assert!(err.to_string().contains("overprint"), "{err}");
    let mut opts = RasterOptions::new(72.0);
    opts.depth = SampleDepth::Sixteen;
    opts.deskew = true;
    let err = render_page_gray(&plain, 1, &opts).unwrap_err();
    assert!(err.to_string().contains("deskew"), "{err}");
}

/// The 16-vs-8 table of the corpus fixtures and the gradient page: each
/// page rendered at 150 dpi at both depths, the 16-bit render narrowed, the
/// largest difference and the differing samples per channel printed.  Run
/// with `--ignored --nocapture`; fixtures absent from the checkout are
/// skipped.
#[test]
#[ignore = "a report over the corpus fixtures, which are not committed"]
fn report_16_bit_against_8_bit() {
    let mut pages: Vec<(String, Vec<u8>)> = vec![("gradient page".to_owned(), gradient_page())];
    for n in 1..=11 {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
        let Some(entry) = std::fs::read_dir(&dir).ok().and_then(|mut d| {
            d.find_map(|e| {
                let e = e.ok()?;
                let name = e.file_name().into_string().ok()?;
                name.starts_with(&format!("corpus-{n:02}-"))
                    .then_some(e.path())
            })
        }) else {
            continue;
        };
        let name = entry.file_name().unwrap().to_string_lossy().into_owned();
        pages.push((name, std::fs::read(&entry).expect("read fixture")));
    }
    for (name, pdf) in pages {
        let session = session(pdf);
        for page in 1..=session.total_pages().min(3) {
            let at = |depth| {
                render_page_canvas(
                    &session,
                    page,
                    &RenderParams::new(150.0 / 72.0).with_depth(depth),
                )
            };
            let (Ok(wide), Ok(narrow)) = (at(SampleDepth::Sixteen), at(SampleDepth::Eight)) else {
                eprintln!("{name} p{page}: does not render");
                continue;
            };
            let wide = rgb16(wide);
            let rows = 0..wide.height;
            let table = narrowed_difference(&wide, &rgb8(narrow), rows);
            eprintln!("{name} p{page}: {table:?}");
        }
    }
}
