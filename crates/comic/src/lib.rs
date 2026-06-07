//! Comic-archive image input for the rasterrocket OCR pipeline.
//!
//! Turns an archive of loose raster images (`.cbz` ZIP, `.cb7` 7-Zip,
//! `.cbt` TAR) into a stream of grayscale [`pdf_raster::RenderedPage`]s — the
//! same OCR-ready artifact `rasterrocket` produces for PDF pages — so comic and
//! scan archives flow into Tesseract / Google Cloud Vision unchanged.
//!
//! `.cbr` (RAR) is intentionally unsupported: RAR is proprietary with no
//! permissively-licensed Rust decoder. Such inputs return an error carrying
//! conversion guidance (`ComicError::RarUnsupported`, defined alongside the
//! public API).

mod archive;
mod decode;
mod order;
#[cfg(test)]
mod test_helpers;

use std::path::Path;

pub use pdf_raster::RenderedPage;

/// A comic archive rendered to pages: each is its 1-based page number paired
/// with the page or the per-page error that replaced it. Whole-archive failures
/// are the outer `Err`; one corrupt page is an inner `Err` and never aborts the
/// run.
type ComicPages = Vec<(u32, Result<RenderedPage, ComicError>)>;

/// Options controlling comic-archive rendering. Mirrors the relevant subset of
/// `rasterrocket::RasterOptions` so the CLI can pass one config to either path.
#[derive(Debug, Clone)]
pub struct ComicOptions {
    /// DPI reported on the produced [`RenderedPage`]. Comic images carry no
    /// physical resolution, so this is a declared value passed through to
    /// `RenderedPage::dpi` / `effective_dpi` for downstream OCR (Tesseract uses
    /// it for feature scaling). Must be > 0.
    pub dpi: f32,
    /// First page (1-based, inclusive), counted over decodable images in
    /// reading order. Must be ≥ 1.
    pub first_page: u32,
    /// Last page (1-based, inclusive). Must be ≥ `first_page`. Clamped to the
    /// image count when it exceeds the number of pages.
    pub last_page: u32,
    /// Run deskew on each decoded page before producing the `RenderedPage`.
    pub deskew: bool,
}

impl Default for ComicOptions {
    fn default() -> Self {
        Self {
            dpi: 300.0,
            first_page: 1,
            last_page: u32::MAX,
            deskew: false,
        }
    }
}

/// Errors from opening or reading a comic archive.
#[derive(Debug)]
pub enum ComicError {
    /// The archive file could not be opened or read.
    Open(std::io::Error),
    /// The container exists but could not be parsed as its claimed format.
    BadArchive(String),
    /// `.cbr` (RAR) input — intentionally unsupported.
    RarUnsupported,
    /// The archive opened but contained zero decodable images.
    NoImages,
    /// A single entry failed to decode. Yielded per-page; never aborts the run.
    Decode {
        /// The archive entry name that failed.
        entry: String,
        /// Human-readable reason.
        reason: String,
    },
    /// A decoded image exceeded the per-page pixel-size safety limit.
    TooLarge {
        /// The archive entry name.
        entry: String,
        /// Decoded width in pixels.
        width: u32,
        /// Decoded height in pixels.
        height: u32,
    },
    /// Deskew failed on a page.
    Deskew(String),
    /// No images and two or more PDFs — the archive is ambiguous.
    AmbiguousArchive(String),
    /// An embedded PDF failed to open or render.
    Pdf(String),
}

impl std::fmt::Display for ComicError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Open(e) => write!(f, "cannot open comic archive: {e}"),
            Self::BadArchive(m) => write!(f, "malformed comic archive: {m}"),
            Self::RarUnsupported => write!(
                f,
                ".cbr (RAR) archives are not supported: RAR is a proprietary \
                 format with no permissively-licensed Rust decoder. Convert it \
                 to .cbz first (unzip the RAR and re-zip the images, or use a \
                 comic-archive converter). .cbz, .cb7, and .cbt work directly."
            ),
            Self::NoImages => write!(f, "comic archive contains no decodable images"),
            Self::Decode { entry, reason } => {
                write!(f, "failed to decode {entry}: {reason}")
            }
            Self::TooLarge {
                entry,
                width,
                height,
            } => write!(
                f,
                "image {entry} is {width}×{height}px, exceeding the per-page size limit"
            ),
            Self::Deskew(m) => write!(f, "deskew failed: {m}"),
            Self::AmbiguousArchive(m) => write!(f, "ambiguous comic archive: {m}"),
            Self::Pdf(m) => write!(f, "embedded PDF render failed: {m}"),
        }
    }
}

impl std::error::Error for ComicError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Open(e) => Some(e),
            _ => None,
        }
    }
}

/// Open a comic archive and render its pages as grayscale [`RenderedPage`]s.
///
/// Yields `(page_num, Result<RenderedPage, ComicError>)` in reading order. A
/// per-page decode failure is yielded as `Err` and iteration continues — one
/// corrupt image never aborts the archive (matching `rasterrocket::raster_pdf`).
///
/// Content routing, decided after the entry list is built:
/// - **Page images present** → render the images; any PDF entries are ignored
///   with a warning. Images always win.
/// - **No images, exactly one PDF** → render that PDF's pages (its own 1-based
///   page numbering).
/// - **No images, two or more PDFs** → [`ComicError::AmbiguousArchive`], naming
///   the PDFs rather than silently guessing.
/// - **No images, no PDF** → [`ComicError::NoImages`].
///
/// # Errors
///
/// Returns `Err` *before* iteration for whole-archive failures: a `.cbr` input
/// ([`ComicError::RarUnsupported`]), an unreadable/malformed container, an
/// ambiguous PDF-only archive ([`ComicError::AmbiguousArchive`]), or an archive
/// with no decodable content ([`ComicError::NoImages`]).
pub fn open_comic(path: &Path, opts: &ComicOptions) -> Result<ComicPages, ComicError> {
    if opts.dpi <= 0.0 {
        return Err(ComicError::BadArchive("dpi must be > 0".to_owned()));
    }
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| ComicError::BadArchive("input path has no file name".to_owned()))?;
    let bytes = std::fs::read(path).map_err(ComicError::Open)?;

    let mut archive = archive::open_archive_from_ext(file_name, bytes)?;

    // Partition entries by content kind. Images win; a lone PDF is the fallback.
    let all_names = archive.entry_names();
    let mut image_names: Vec<String> = all_names
        .iter()
        .filter(|n| order::is_image_file(n))
        .cloned()
        .collect();
    let pdf_names: Vec<String> = all_names
        .iter()
        .filter(|n| order::is_pdf_file(n))
        .cloned()
        .collect();

    if image_names.is_empty() {
        // No page images: fall back to an embedded PDF if exactly one exists.
        return match pdf_names.len() {
            0 => Err(ComicError::NoImages),
            1 => render_embedded_pdf(archive.as_mut(), &pdf_names[0], opts),
            _ => Err(ComicError::AmbiguousArchive(format!(
                "no page images and {} PDFs ({}); extract and render the one you want",
                pdf_names.len(),
                pdf_names.join(", ")
            ))),
        };
    }
    if !pdf_names.is_empty() {
        log::warn!(
            "comic: archive has page images and {} PDF(s) ({}); rendering the images, ignoring the PDF(s)",
            pdf_names.len(),
            pdf_names.join(", ")
        );
    }

    // Natural-sort the images into reading order.
    image_names.sort_by(|a, b| order::natural_cmp(a, b));

    // First pass: read each entry and keep only those whose *content* sniffs as
    // a recognised image (the extension may have lied). Unsupported content is
    // skipped with a format-named warning — graceful degradation — and never
    // consumes a page number, so the page window applies to decodable pages
    // only. An entry that genuinely fails to read is kept so it can surface as a
    // per-page Err in the second pass, preserving its reading-order slot.
    let mut decodable: Vec<(String, Result<Vec<u8>, ComicError>)> = Vec::new();
    for name in image_names {
        match archive.read_entry(&name) {
            Ok(raw) if decode::sniff(&raw).is_some() => decodable.push((name, Ok(raw))),
            Ok(_) => log::warn!(
                "comic: skipping {name}: unrecognised image format; convert it to PNG or JPG"
            ),
            Err(e) => decodable.push((name, Err(e))),
        }
    }
    if decodable.is_empty() {
        return Err(ComicError::NoImages);
    }

    // Second pass: assign 1-based page numbers over kept entries, apply the
    // window, and render each. Per-page errors are yielded (not panicked) and
    // iteration continues.
    let first = opts.first_page.max(1);
    let last = opts.last_page;
    let mut out: ComicPages = Vec::new();
    let mut page_num: u32 = 0;
    for (name, raw) in decodable {
        page_num = page_num.saturating_add(1);
        if page_num < first || page_num > last {
            continue;
        }
        let page = raw.and_then(|bytes| render_one(&name, &bytes, page_num, opts));
        out.push((page_num, page));
    }
    if out.is_empty() {
        // The window excluded every kept page.
        return Err(ComicError::NoImages);
    }
    Ok(out)
}

/// Decode, guard, deskew, and assemble one page from already-read bytes. Errors
/// are returned (not panicked) so the caller yields them per-page and keeps
/// going. `raw` is guaranteed to sniff as a recognised image by the caller.
fn render_one(
    name: &str,
    raw: &[u8],
    page_num: u32,
    opts: &ComicOptions,
) -> Result<RenderedPage, ComicError> {
    let decoded = decode::decode_image(raw).map_err(|e| ComicError::Decode {
        entry: name.to_owned(),
        reason: e.to_string(),
    })?;

    pdf_raster::validate_dimensions(decoded.width, decoded.height).map_err(|_| {
        ComicError::TooLarge {
            entry: name.to_owned(),
            width: decoded.width,
            height: decoded.height,
        }
    })?;

    // Rebuild a Gray8 bitmap so deskew + the shared constructor apply uniformly.
    let mut bmp = raster::Bitmap::<color::Gray8>::new(decoded.width, decoded.height, 1, false);
    let w = decoded.width as usize;
    for y in 0..decoded.height {
        let src = &decoded.gray[(y as usize * w)..((y as usize + 1) * w)];
        bmp.row_bytes_mut(y)[..w].copy_from_slice(src);
    }

    if opts.deskew {
        pdf_raster::deskew::apply(&mut bmp).map_err(|e| ComicError::Deskew(e.to_string()))?;
    }

    Ok(pdf_raster::gray8_to_rendered_page(
        &bmp,
        page_num,
        opts.dpi,
        opts.dpi, // comics carry no UserUnit; effective_dpi == dpi
        pdf_raster::PageDiagnostics::default(),
    ))
}

/// Extract a single PDF entry's bytes and render it via the in-memory PDF
/// pipeline, mapping render errors into [`ComicError::Pdf`]. Page numbers are the
/// PDF's own 1-based page sequence — they are not renumbered.
fn render_embedded_pdf(
    archive: &mut dyn archive::Archive,
    name: &str,
    opts: &ComicOptions,
) -> Result<ComicPages, ComicError> {
    let bytes = archive.read_entry(name)?;
    let ropts = pdf_raster::RasterOptions {
        dpi: opts.dpi,
        first_page: opts.first_page,
        last_page: opts.last_page,
        deskew: opts.deskew,
        pages: None,
    };
    let out: ComicPages = pdf_raster::raster_pdf_from_bytes(bytes, &ropts)
        .map(|(n, r)| (n, r.map_err(|e| ComicError::Pdf(e.to_string()))))
        .collect();
    if out.is_empty() {
        // A valid PDF with zero pages yields nothing — surface that as a clear
        // error rather than a silent empty success. A PDF whose pages all *error*
        // still produces per-page `Err` items here (out is non-empty), so those
        // failures surface individually and are not masked by this check.
        return Err(ComicError::Pdf(format!("embedded PDF {name} has no pages")));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rar_message_is_actionable() {
        let msg = ComicError::RarUnsupported.to_string();
        assert!(msg.contains(".cbr"), "names the format");
        assert!(msg.to_lowercase().contains(".cbz"), "suggests the fix");
        assert!(msg.to_lowercase().contains("rar"), "explains why");
    }

    #[test]
    fn options_default_matches_raster_defaults() {
        let o = ComicOptions::default();
        assert!((o.dpi - 300.0).abs() < f32::EPSILON);
        assert_eq!(o.first_page, 1);
        assert_eq!(o.last_page, u32::MAX);
        assert!(!o.deskew);
    }

    use crate::test_helpers::make_cbz;

    /// A solid grayscale JPEG page, `w`×`h`, every pixel `v`.
    fn jpeg_page(w: u32, h: u32, v: u8) -> Vec<u8> {
        use color::Gray8;
        use raster::Bitmap;
        let mut b = Bitmap::<Gray8>::new(w, h, 1, false);
        for y in 0..h {
            for x in 0..w as usize {
                b.row_bytes_mut(y)[x] = v;
            }
        }
        encode::jpeg_gray::<Gray8>(&b, 90).unwrap()
    }

    /// A minimal valid single-page PDF (one empty 612×792-pt page) — a
    /// known-good fixture for the embedded-PDF routing tests.
    fn minimal_pdf() -> Vec<u8> {
        b"%PDF-1.4\n\
1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n\
2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n\
3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]>>\nendobj\n\
xref\n0 4\n\
0000000000 65535 f\r\n\
0000000009 00000 n\r\n\
0000000056 00000 n\r\n\
0000000111 00000 n\r\n\
trailer\n<</Size 4 /Root 1 0 R>>\n\
startxref\n180\n%%EOF"
            .to_vec()
    }

    fn write_temp_cbz(name: &str, bytes: &[u8]) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(name);
        std::fs::write(&path, bytes).unwrap();
        (dir, path)
    }

    #[test]
    fn open_comic_yields_ordered_pages() {
        // Two solid pages + one non-image entry; names out of natural order.
        let cbz = make_cbz(&[
            ("page10.jpg", &jpeg_page(4, 4, 200)),
            ("page2.jpg", &jpeg_page(4, 4, 100)),
            ("ComicInfo.xml", b"<x/>"),
        ]);
        let (_dir, path) = write_temp_cbz("book.cbz", &cbz);

        let pages = open_comic(&path, &ComicOptions::default()).unwrap();
        assert_eq!(pages.len(), 2, "two images, xml filtered out");
        // page2 (value ~100) must come before page10 (value ~200): natural sort.
        let (n1, r1) = &pages[0];
        let (n2, r2) = &pages[1];
        assert_eq!((*n1, *n2), (1, 2));
        let p1 = r1.as_ref().unwrap();
        let p2 = r2.as_ref().unwrap();
        assert!(p1.pixels[0] < p2.pixels[0], "page2 darker than page10");
        assert_eq!((p1.width, p1.height), (4, 4));
    }

    #[test]
    fn empty_archive_is_no_images() {
        let cbz = make_cbz(&[]);
        let (_dir, path) = write_temp_cbz("empty.cbz", &cbz);
        assert!(matches!(
            open_comic(&path, &ComicOptions::default()),
            Err(ComicError::NoImages)
        ));
    }

    #[test]
    fn cbr_returns_rar_unsupported() {
        let (_dir, path) = write_temp_cbz("x.cbr", b"Rar!\x1a\x07\x00");
        assert!(matches!(
            open_comic(&path, &ComicOptions::default()),
            Err(ComicError::RarUnsupported)
        ));
    }

    #[test]
    fn lying_extension_is_skipped_not_errored() {
        // An entry named .jpg whose bytes are not an image must be skipped
        // (warned), not surfaced as a page — and must not consume a page number.
        let cbz = make_cbz(&[
            ("001.jpg", b"NOT AN IMAGE AT ALL"),
            ("002.jpg", &jpeg_page(4, 4, 120)),
        ]);
        let (_dir, path) = write_temp_cbz("mixed.cbz", &cbz);

        let pages = open_comic(&path, &ComicOptions::default()).unwrap();
        assert_eq!(
            pages.len(),
            1,
            "the junk entry is skipped, the real one kept"
        );
        assert_eq!(pages[0].0, 1, "kept page is numbered 1, not 2");
        assert!(pages[0].1.is_ok());
    }

    #[test]
    fn cbz_with_only_a_pdf_renders_it() {
        let cbz = make_cbz(&[("book.pdf", &minimal_pdf())]);
        let (_dir, path) = write_temp_cbz("book.cbz", &cbz);
        let pages = open_comic(&path, &ComicOptions::default()).unwrap();
        assert!(!pages.is_empty(), "the embedded PDF must yield pages");
        assert_eq!(pages[0].0, 1, "PDF's own 1-based page numbering");
        assert!(pages[0].1.is_ok());
    }

    #[test]
    fn cbz_with_two_pdfs_is_ambiguous() {
        let cbz = make_cbz(&[("vol1.pdf", &minimal_pdf()), ("vol2.pdf", &minimal_pdf())]);
        let (_dir, path) = write_temp_cbz("two.cbz", &cbz);
        assert!(matches!(
            open_comic(&path, &ComicOptions::default()),
            Err(ComicError::AmbiguousArchive(_))
        ));
    }

    #[test]
    fn images_win_over_pdf() {
        // Both an image page and a PDF present: render the image, ignore the PDF.
        let cbz = make_cbz(&[
            ("page1.jpg", &jpeg_page(4, 4, 90)),
            ("book.pdf", &minimal_pdf()),
        ]);
        let (_dir, path) = write_temp_cbz("both.cbz", &cbz);
        let pages = open_comic(&path, &ComicOptions::default()).unwrap();
        assert_eq!(pages.len(), 1, "only the image page is rendered");
        let page = pages[0].1.as_ref().unwrap();
        assert_eq!(
            (page.width, page.height),
            (4, 4),
            "the rendered page is the 4×4 image, not a 612×792-pt PDF page"
        );
    }
}
