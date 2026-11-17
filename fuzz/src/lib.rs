//! Helpers shared by the fuzz targets: in-memory PDF and archive assembly,
//! and bounded renders of a document or comic archive through the public
//! pipeline.

/// PDF assembly helpers shared with the rendering integration suites.
#[path = "../../crates/pdf_raster/tests/common/mod.rs"]
pub mod common;

use rasterrocket::{
    BackendPolicy, Rotation, SessionConfig, open_session_from_bytes, render_page_rgb,
};

/// Pages rendered per input.
const MAX_PAGES: u32 = 4;

/// Device pixels per point.  A 612 × 792 pt page renders at 61 × 79 px, so
/// an iteration costs the decode and interpretation work, not the fill.
const SCALE: f64 = 0.1;

/// Open `bytes` on the CPU backend and render up to [`MAX_PAGES`] pages at
/// [`SCALE`].  Every error is discarded: the property under test is that
/// opening and rendering return.
pub fn render_document(bytes: Vec<u8>) {
    render_document_at(bytes, SCALE);
}

/// [`render_document`] at `scale` device pixels per point.
pub fn render_document_at(bytes: Vec<u8>, scale: f64) {
    render_document_with(bytes, scale, &cpu_config());
}

/// [`render_document_at`] in a session opened with `config`.
pub fn render_document_with(bytes: Vec<u8>, scale: f64, config: &SessionConfig) {
    let Ok(session) = open_session_from_bytes(bytes, config) else {
        return;
    };
    for page in 1..=session.total_pages().min(MAX_PAGES) {
        drop(render_page_rgb(&session, page, scale));
    }
}

/// The session configuration of the targets: the CPU backend, every other
/// setting at its default.
#[must_use]
pub fn cpu_config() -> SessionConfig {
    SessionConfig::with_policy(BackendPolicy::CpuOnly)
}

/// Render a one-page document whose 100 × 100 pt page has `resources` as its
/// resource dictionary and `content` as its content stream.  `rest` are the
/// objects from number 5 on; object 4 is the content stream.
pub fn render_page(resources: &str, content: &[u8], rest: Vec<Vec<u8>>) {
    render_page_with(resources, content, rest, &cpu_config());
}

/// [`render_page`] in a session opened with `config`.
pub fn render_page_with(
    resources: &str,
    content: &[u8],
    rest: Vec<Vec<u8>>,
    config: &SessionConfig,
) {
    let page = format!(
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
         /Resources {resources} /Contents 4 0 R >>"
    );
    let mut objects = vec![common::stream_object("", content)];
    objects.extend(rest);
    render_document_with(common::one_page(&page, objects), 0.5, config);
}

/// Splits fuzz input into fixed-size selector bytes and the remaining
/// payload; `None` when the input is shorter than `N`.
pub fn split_selectors<const N: usize>(data: &[u8]) -> Option<([u8; N], &[u8])> {
    let (head, payload) = data.split_first_chunk::<N>()?;
    Some((*head, payload))
}

/// Largest image side, in pixels, the decoder targets request.
const MAX_SIDE: u16 = 4096;

/// `(width, height)` from two little-endian `u16`s, each plus one and capped
/// at [`MAX_SIDE`]: never zero, and a bitmap of at most 16 Mpx.
pub fn image_dimensions(bytes: [u8; 4]) -> (u32, u32) {
    let side = |lo, hi| u32::from(u16::from_le_bytes([lo, hi]).saturating_add(1).min(MAX_SIDE));
    (side(bytes[0], bytes[1]), side(bytes[2], bytes[3]))
}

/// Pages of a comic archive rendered per input.
const MAX_COMIC_PAGES: usize = 4;

/// Decompressed-archive and page budget of the comic targets, in bytes.
const COMIC_BUDGET: u64 = 64 << 20;

/// Open the comic archive `bytes`, its container chosen by the extension of
/// `file_name`, and render up to [`MAX_COMIC_PAGES`] pages.  Bit 0 of
/// `flags` renders at 16 bits, bit 1 deskews, bits 2–3 turn the pages by
/// that many quarter turns.  Every error is discarded.
pub fn render_comic(file_name: &str, bytes: Vec<u8>, flags: u8) {
    let mut opts = comic::ComicOptions::new(72.0);
    opts.max_archive_bytes = COMIC_BUDGET;
    opts.depth = if flags & 1 == 0 {
        comic::SampleDepth::Eight
    } else {
        comic::SampleDepth::Sixteen
    };
    opts.deskew = flags & 2 != 0;
    opts.session = cpu_config();
    opts.session.max_page_bytes = COMIC_BUDGET;
    opts.session.rotation = [
        Rotation::None,
        Rotation::Cw90,
        Rotation::Cw180,
        Rotation::Cw270,
    ][usize::from(flags >> 2) & 3];
    let Ok(pages) = comic::open_comic_bytes(file_name, bytes, &opts) else {
        return;
    };
    for (_, page) in pages.take(MAX_COMIC_PAGES) {
        drop(page);
    }
}

/// A POSIX tar archive holding the one regular file `name` (at most 100
/// bytes) with the contents `data`.
#[must_use]
pub fn tar_of(name: &str, data: &[u8]) -> Vec<u8> {
    let mut header = [0u8; 512];
    header[..name.len()].copy_from_slice(name.as_bytes());
    header[100..108].copy_from_slice(b"0000644\0");
    header[108..116].copy_from_slice(b"0000000\0");
    header[116..124].copy_from_slice(b"0000000\0");
    header[124..136].copy_from_slice(format!("{:011o}\0", data.len()).as_bytes());
    header[136..148].copy_from_slice(b"00000000000\0");
    header[156] = b'0';
    header[257..265].copy_from_slice(b"ustar\x0000");
    // The checksum is the byte sum of the header with its own field blank.
    header[148..156].fill(b' ');
    let sum: u32 = header.iter().map(|&b| u32::from(b)).sum();
    header[148..156].copy_from_slice(format!("{sum:06o}\0 ").as_bytes());
    let mut out = header.to_vec();
    out.extend_from_slice(data);
    out.resize(out.len().next_multiple_of(512) + 1024, 0);
    out
}
