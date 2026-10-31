//! Helpers shared by the end-to-end render parity tests: synthetic PDF
//! assembly, session construction per backend policy, and byte-exact page
//! comparison.

pub(crate) mod icc;

use std::fmt::Write as _;
use std::path::PathBuf;

use rasterrocket::{
    BackendPolicy, RasterError, RasterOptions, RasterSession, RenderPurpose, RenderedPage,
    SessionConfig, open_session_from_bytes, raster_pdf_from_bytes, render_page_rgb,
};

/// 150 DPI: the resolution the parity findings were measured at.
pub(crate) const SCALE: f64 = 150.0 / 72.0;

/// Assemble a one-page PDF from `objects` (1-based, in order) with an
/// exact xref table.
pub(crate) fn assemble_pdf(objects: &[Vec<u8>]) -> Vec<u8> {
    let mut pdf = b"%PDF-1.5\n%\xE2\xE3\xCF\xD3\n".to_vec();
    let mut offsets = Vec::with_capacity(objects.len());
    for (i, body) in objects.iter().enumerate() {
        offsets.push(pdf.len());
        pdf.extend_from_slice(format!("{} 0 obj\n", i + 1).as_bytes());
        pdf.extend_from_slice(body);
        pdf.extend_from_slice(b"\nendobj\n");
    }
    let xref_at = pdf.len();
    let mut xref = format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1);
    for off in offsets {
        let _ = writeln!(xref, "{off:010} 00000 n ");
    }
    let _ = write!(
        xref,
        "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n",
        objects.len() + 1
    );
    pdf.extend_from_slice(xref.as_bytes());
    pdf
}

/// A stream object body with `dict` entries and an exact `/Length`.
pub(crate) fn stream_object(dict: &str, data: &[u8]) -> Vec<u8> {
    let mut obj = format!("<< {dict} /Length {} >>\nstream\n", data.len()).into_bytes();
    obj.extend_from_slice(data);
    obj.extend_from_slice(b"\nendstream");
    obj
}

/// Set to make an absent corpus fixture fail its test instead of skipping it.
pub(crate) const REQUIRE_FIXTURES_VAR: &str = "PDF_RASTER_REQUIRE_FIXTURES";

/// Path of a corpus fixture under `tests/fixtures`, or `None` (with a
/// note on stderr) when it is absent from this checkout.  The corpus is
/// gitignored, so a test that needs it passes vacuously without it.
///
/// # Panics
///
/// When the fixture is absent and [`REQUIRE_FIXTURES_VAR`] is set.
pub(crate) fn corpus_fixture(name: &str) -> Option<PathBuf> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name);
    if path.exists() {
        return Some(path);
    }
    assert!(
        std::env::var_os(REQUIRE_FIXTURES_VAR).is_none(),
        "corpus fixture absent ({}) and {REQUIRE_FIXTURES_VAR} is set",
        path.display()
    );
    eprintln!(
        "skipping: corpus fixture absent ({}); set {REQUIRE_FIXTURES_VAR} to fail instead",
        path.display()
    );
    None
}

/// `(width, height, rgb bytes)` of one rendered page.
pub(crate) fn render(session: &RasterSession, page: u32) -> (u32, u32, Vec<u8>) {
    let bmp = render_page_rgb(session, page, SCALE).expect("render page");
    (bmp.width, bmp.height, bmp.data().to_vec())
}

/// Assert two renders are byte-identical, reporting the divergence
/// statistics the way the original findings were measured.
pub(crate) fn assert_identical(cpu: &(u32, u32, Vec<u8>), gpu: &(u32, u32, Vec<u8>), what: &str) {
    assert_eq!((cpu.0, cpu.1), (gpu.0, gpu.1), "{what}: dimensions differ");
    assert_eq!(cpu.2.len(), gpu.2.len(), "{what}: byte length differs");
    let differing = cpu.2.iter().zip(&gpu.2).filter(|(a, b)| a != b).count();
    let max_delta = cpu
        .2
        .iter()
        .zip(&gpu.2)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0);
    assert_eq!(
        differing,
        0,
        "{what}: {differing} of {} bytes differ between the CPU and GPU renders (max delta {max_delta})",
        cpu.2.len()
    );
}

/// Open `bytes` under `policy`.
pub(crate) fn session(bytes: Vec<u8>, policy: BackendPolicy) -> RasterSession {
    open_session_from_bytes(bytes, &SessionConfig::with_policy(policy))
        .unwrap_or_else(|e| panic!("{policy:?} session: {e}"))
}

/// Catalog 1 and a page tree 2 whose one kid is object 3, then `objects`
/// from object 3 on (the first is the page).
pub(crate) fn with_page_tree(objects: &[Vec<u8>]) -> Vec<u8> {
    let mut all = Vec::with_capacity(objects.len() + 2);
    all.push(b"<< /Type /Catalog /Pages 2 0 R >>".to_vec());
    all.push(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec());
    all.extend_from_slice(objects);
    assemble_pdf(&all)
}

/// [`with_page_tree`] with `page` as object 3 and `rest` from object 4 on.
pub(crate) fn one_page(page: &str, rest: Vec<Vec<u8>>) -> Vec<u8> {
    let mut objects = vec![page.as_bytes().to_vec()];
    objects.extend(rest);
    with_page_tree(&objects)
}

/// Fraction of page 1's pixels that are not white, at 72 dpi on the CPU.
pub(crate) fn ink(pdf: Vec<u8>) -> f64 {
    let s = session(pdf, BackendPolicy::CpuOnly);
    let bmp = render_page_rgb(&s, 1, 1.0).expect("page renders");
    let (px, _) = bmp.data().as_chunks::<3>();
    let total = px.len();
    let dark = px.iter().filter(|p| p.iter().any(|&c| c < 128)).count();
    #[expect(
        clippy::cast_precision_loss,
        reason = "pixel counts of a test page are far below 2^52"
    )]
    let frac = dark as f64 / total as f64;
    frac
}

/// A 200 × 100 pt page whose content fills the 80 × 80 square at (10, 10):
/// 32 % of the page.
pub(crate) const SQUARE: &[u8] = b"0 g 10 10 80 80 re f";

/// A one-page PDF whose `width` × `height` pt page runs `content`
/// (object 4) with the font dictionary `font` (object 5) as `/F1`; `extra`
/// are the objects from 6 on that `font` references.
pub(crate) fn font_page(
    (width, height): (u32, u32),
    font: &str,
    content: &str,
    extra: &[Vec<u8>],
) -> Vec<u8> {
    let mut rest = vec![
        stream_object("", content.as_bytes()),
        font.as_bytes().to_vec(),
    ];
    rest.extend_from_slice(extra);
    one_page(
        &format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] \
             /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        rest,
    )
}

/// [`font_page`] on a 64 × 32 pt page showing the string operand `text`
/// at 24 pt from (4, 8).
pub(crate) fn text_page(font: &str, text: &str) -> Vec<u8> {
    font_page(
        (64, 32),
        font,
        &format!("BT /F1 24 Tf 4 8 Td {text} Tj ET"),
        &[],
    )
}

/// The device position `(x, y)`, origin top-left, of every dark pixel (red
/// below 128) of page 1 of `pdf`, rendered at 72 dpi under `config`.
pub(crate) fn dark_pixels_with(pdf: Vec<u8>, config: &SessionConfig) -> Vec<(usize, usize)> {
    let s = open_session_from_bytes(pdf, config).expect("open session");
    let bmp = render_page_rgb(&s, 1, 1.0).expect("render page");
    let mut dark = Vec::new();
    for y in 0..bmp.height {
        let (px, _) = bmp.row_bytes(y)[..bmp.width as usize * 3].as_chunks::<3>();
        let row = px.iter().enumerate().filter(|(_, p)| p[0] < 128);
        dark.extend(row.map(|(x, _)| (x, y as usize)));
    }
    dark
}

/// [`dark_pixels_with`] under the default CPU configuration.
pub(crate) fn dark_pixels(pdf: Vec<u8>) -> Vec<(usize, usize)> {
    dark_pixels_with(pdf, &SessionConfig::with_policy(BackendPolicy::CpuOnly))
}

/// `(left, width)` in device pixels of the dark pixels of page 1 of `pdf`
/// (see [`dark_pixels`]); `(0, 0)` when there are none.
pub(crate) fn ink_span(pdf: Vec<u8>) -> (usize, usize) {
    let dark = dark_pixels(pdf);
    let left = dark.iter().map(|&(x, _)| x).min().unwrap_or(0);
    let right = dark.iter().map(|&(x, _)| x).max().map_or(0, |r| r + 1);
    (left, right - left)
}

/// The single page of `pdf` rendered through the batch API under `opts`.
///
/// # Panics
///
/// When the document does not yield exactly one page.
pub(crate) fn render_one(pdf: Vec<u8>, opts: &RasterOptions) -> Result<RenderedPage, RasterError> {
    let mut pages: Vec<_> = raster_pdf_from_bytes(pdf, opts).collect();
    assert_eq!(pages.len(), 1, "one page expected");
    pages.remove(0).1
}

/// [`render_one`] at 72 dpi (one pixel per point) with the default options.
///
/// # Panics
///
/// When the page fails to render.
pub(crate) fn render_gray(pdf: Vec<u8>) -> RenderedPage {
    render_one(pdf, &RasterOptions::new(72.0)).expect("page renders")
}

/// Gray value of `page` at pixel `(x, y)`, counted from the bottom-left
/// corner; at 72 dpi that is the user-space point `(x, y)`.
pub(crate) fn gray_at(page: &RenderedPage, x: u32, y: u32) -> u8 {
    assert!(
        x < page.width && y < page.height,
        "({x}, {y}) is off the page"
    );
    let row = page.height - 1 - y;
    page.pixels[(row * page.width + x) as usize]
}

/// RGB of page 1 of `pdf`, rendered at 72 dpi on the CPU for `purpose`, at
/// each of `points`, counted from the bottom-left corner (user space).
pub(crate) fn rgb_at_for(
    pdf: Vec<u8>,
    purpose: RenderPurpose,
    points: &[(u32, u32)],
) -> Vec<[u8; 3]> {
    let mut config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    config.purpose = purpose;
    let s = open_session_from_bytes(pdf, &config).expect("open session");
    let bmp = render_page_rgb(&s, 1, 1.0).expect("render page");
    points
        .iter()
        .map(|&(x, y)| {
            assert!(
                x < bmp.width && y < bmp.height,
                "({x}, {y}) is off the page"
            );
            let row = bmp.row_bytes(bmp.height - 1 - y);
            let i = x as usize * 3;
            [row[i], row[i + 1], row[i + 2]]
        })
        .collect()
}

/// [`rgb_at_for`] rendered for viewing.
pub(crate) fn rgb_at(pdf: Vec<u8>, points: &[(u32, u32)]) -> Vec<[u8; 3]> {
    rgb_at_for(pdf, RenderPurpose::View, points)
}

/// `data` as an `ASCIIHexDecode` stream body, with its `>` terminator.
pub(crate) fn ascii_hex(data: &[u8]) -> Vec<u8> {
    let mut out = String::with_capacity(data.len() * 2 + 1);
    for b in data {
        let _ = write!(out, "{b:02X}");
    }
    out.push('>');
    out.into_bytes()
}

/// Assert that every channel of `got` is within `tol` of `want`.
pub(crate) fn assert_rgb_near(got: [u8; 3], want: [u8; 3], tol: u8, what: &str) {
    assert!(
        got.iter().zip(want).all(|(&g, w)| g.abs_diff(w) <= tol),
        "{what}: got {got:?}, want {want:?} ± {tol}"
    );
}
