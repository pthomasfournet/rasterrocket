//! Comic-archive image input for the rasterrocket OCR pipeline.
//!
//! Turns an archive of loose raster images (`.cbz` ZIP, `.cb7` 7-Zip,
//! `.cbt` TAR, `.cbr` RAR) into a stream of grayscale [`pdf_raster::RenderedPage`]s, 8- or
//! 16-bit ([`ComicOptions::depth`]) — the
//! same OCR-ready artifact `rasterrocket` produces for PDF pages — so comic and
//! scan archives flow into Tesseract / Google Cloud Vision unchanged.

mod archive;
mod decode;
mod order;
#[cfg(test)]
mod test_helpers;
mod turn;

use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;

pub use archive::DEFAULT_MAX_ARCHIVE_BYTES;
pub use pdf_raster::{RenderedPage, SampleDepth};
use raster::Bitmap;

/// The pages of an opened comic archive, rendered one at a time as the
/// iterator advances.
///
/// Each item is a 1-based page number paired with the page or the per-page
/// error that replaced it; one corrupt page never ends the iteration.  The
/// page-image entries are decompressed when the archive is opened and each
/// is held until its last page is rendered, so memory falls as the pages are
/// yielded; their sum is bounded by [`ComicOptions::max_archive_bytes`].  An
/// embedded PDF is held whole and its pages rendered from it.  [`len`](ExactSizeIterator::len)
/// is the number of pages still to be yielded, and [`retain`](Self::retain)
/// drops pages before any of them is rendered.
pub struct ComicPages {
    opts: ComicOptions,
    source: Source,
}

/// One page of a page-image entry, rendered on demand.
struct ImagePage {
    /// 1-based page number; 0 until the pages are numbered.
    page: u32,
    /// The entry's name, with the page within it for a multi-page image.
    name: String,
    /// The entry's bytes, shared by its pages, or the error reading them.
    bytes: Result<Arc<Vec<u8>>, ComicError>,
    /// The page within the entry, as [`decode::image_pages`] lists it.
    index: usize,
}

/// What [`ComicPages`] renders its pages from.
enum Source {
    /// Page images in reading order.
    Images(VecDeque<ImagePage>),
    /// The archive's single embedded PDF and the page numbers to render.
    Pdf {
        session: Box<pdf_raster::RasterSession>,
        pages: VecDeque<u32>,
    },
}

impl ComicPages {
    /// Keep only the pages whose number satisfies `keep`; the others are
    /// discarded unrendered.
    pub fn retain(&mut self, mut keep: impl FnMut(u32) -> bool) {
        match &mut self.source {
            Source::Images(entries) => entries.retain(|image| keep(image.page)),
            Source::Pdf { pages, .. } => pages.retain(|page| keep(*page)),
        }
    }
}

impl Iterator for ComicPages {
    type Item = (u32, Result<RenderedPage, ComicError>);

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.source {
            Source::Images(entries) => {
                let ImagePage {
                    page,
                    name,
                    bytes,
                    index,
                } = entries.pop_front()?;
                let rendered =
                    bytes.and_then(|bytes| render_one(&name, &bytes, index, page, &self.opts));
                Some((page, rendered))
            }
            Source::Pdf { session, pages } => {
                let page = pages.pop_front()?;
                let mut ropts = pdf_raster::RasterOptions::new(self.opts.dpi);
                ropts.deskew = self.opts.deskew;
                ropts.depth = self.opts.depth;
                let rendered = pdf_raster::render_page_gray(session, page, &ropts)
                    .map_err(|e| ComicError::Pdf(e.to_string()));
                Some((page, rendered))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = match &self.source {
            Source::Images(entries) => entries.len(),
            Source::Pdf { pages, .. } => pages.len(),
        };
        (n, Some(n))
    }
}

impl ExactSizeIterator for ComicPages {}

impl std::iter::FusedIterator for ComicPages {}

impl std::fmt::Debug for ComicPages {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComicPages")
            .field("remaining", &self.len())
            .finish_non_exhaustive()
    }
}

/// Options controlling comic-archive rendering. Mirrors the relevant subset of
/// `rasterrocket::RasterOptions` so the CLI can pass one config to either path.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ComicOptions {
    /// DPI reported on the produced [`RenderedPage`]. Comic images carry no
    /// physical resolution, so this is a declared value passed through to
    /// `RenderedPage::dpi` for downstream OCR (Tesseract uses
    /// it for feature scaling); an embedded PDF is rendered at it. Must be
    /// positive and finite.
    pub dpi: f32,
    /// First page (1-based, inclusive), counted over decodable images in
    /// reading order. Must be ≥ 1.
    pub first_page: u32,
    /// Last page (1-based, inclusive). Must be ≥ `first_page`. Clamped to the
    /// image count when it exceeds the number of pages.
    pub last_page: u32,
    /// Run deskew on each decoded page before producing the `RenderedPage`,
    /// which carries the skew measured in `RenderedPage::skew`.
    pub deskew: bool,
    /// The depth of the pages' samples.  Default [`SampleDepth::Eight`]:
    /// [`PixelFormat::Gray8`](pdf_raster::PixelFormat::Gray8) pages.
    /// [`SampleDepth::Sixteen`] yields
    /// [`PixelFormat::Gray16`](pdf_raster::PixelFormat::Gray16) pages: an
    /// image of more than 8 bits per sample (PNG, TIFF, JPEG XL, AVIF)
    /// keeps its levels, an 8-bit image is widened exactly (`v · 257`), and
    /// an embedded PDF renders at 16 bits.
    pub depth: SampleDepth,
    /// The session an embedded PDF is opened with: its overprint,
    /// antialiasing, annotation, font, decryption and backend settings.
    /// Its [`rotation`](pdf_raster::SessionConfig::rotation) also turns
    /// every page image, before deskew.
    /// Default [`SessionConfig::default`](pdf_raster::SessionConfig::default).
    pub session: pdf_raster::SessionConfig,
    /// The most bytes the archive's entries may decompress to in total.  An
    /// archive that inflates past it is refused with
    /// [`ComicError::ArchiveTooLarge`] before the first page is rendered.
    /// Must be positive.  Default [`DEFAULT_MAX_ARCHIVE_BYTES`].
    pub max_archive_bytes: u64,
}

impl ComicOptions {
    /// Options for rendering every page at `dpi`, every other field at its
    /// [`Default`] value.
    ///
    /// The struct is `#[non_exhaustive]`: start from this or from
    /// [`ComicOptions::default`] and assign the fields to change.
    #[must_use]
    pub fn new(dpi: f32) -> Self {
        Self {
            dpi,
            ..Self::default()
        }
    }
}

impl Default for ComicOptions {
    fn default() -> Self {
        Self {
            dpi: 300.0,
            first_page: 1,
            last_page: u32::MAX,
            deskew: false,
            depth: SampleDepth::Eight,
            session: pdf_raster::SessionConfig::default(),
            max_archive_bytes: DEFAULT_MAX_ARCHIVE_BYTES,
        }
    }
}

/// Errors from opening or reading a comic archive.
#[derive(Debug)]
#[non_exhaustive]
pub enum ComicError {
    /// [`ComicOptions`] fields violate their documented constraints.
    InvalidOptions(String),
    /// The archive file could not be opened or read.
    Open(std::io::Error),
    /// The container exists but could not be parsed as its claimed format.
    BadArchive(String),
    /// The archive's entries decompress to more than
    /// [`ComicOptions::max_archive_bytes`] in total.
    ArchiveTooLarge {
        /// The decompressed-size budget the archive exceeded, in bytes.
        limit: u64,
    },
    /// The archive opened but contained zero decodable images.
    NoImages,
    /// A single entry failed to decode. Yielded per-page; never aborts the run.
    Decode {
        /// The archive entry name that failed.
        entry: String,
        /// Human-readable reason.
        reason: String,
    },
    /// An image's header dimensions exceed the page byte budget
    /// ([`SessionConfig::max_page_bytes`](pdf_raster::SessionConfig::max_page_bytes)
    /// at 3 samples per pixel of [`ComicOptions::depth`]: 3 bytes, or 6 at
    /// 16 bits) or the side ceiling.
    TooLarge {
        /// The archive entry name.
        entry: String,
        /// Decoded width in pixels.
        width: u32,
        /// Decoded height in pixels.
        height: u32,
    },
    /// Deskew was refused for a page larger than
    /// [`DESKEW_MAX_SIDE`](pdf_raster::deskew::DESKEW_MAX_SIDE).
    Deskew(String),
    /// No images and two or more PDFs — the archive is ambiguous.
    AmbiguousArchive(String),
    /// An embedded PDF failed to open or render.
    Pdf(String),
}

impl std::fmt::Display for ComicError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidOptions(m) => write!(f, "invalid comic options: {m}"),
            Self::Open(e) => write!(f, "cannot open comic archive: {e}"),
            Self::BadArchive(m) => write!(f, "malformed comic archive: {m}"),
            Self::ArchiveTooLarge { limit } => write!(
                f,
                "comic archive decompresses to more than its {limit}-byte budget \
                 (possible decompression bomb); raise ComicOptions::max_archive_bytes \
                 (rrocket --max-archive-mib) if the archive is genuine"
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
                "image {entry} is {width}×{height}px, over the page budget \
                 (SessionConfig::max_page_bytes, rrocket --max-page-mib) or the \
                 per-side ceiling of {} px",
                pdf_raster::MAX_PX_SIDE
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

/// Open a comic archive for rendering its pages as grayscale [`RenderedPage`]s.
///
/// The returned [`ComicPages`] yields `(page_num, Result<RenderedPage,
/// ComicError>)` in reading order, decoding each page when it is reached. A
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
/// Returns `Err` *before* iteration for whole-archive failures: options that
/// violate their constraints ([`ComicError::InvalidOptions`]), an
/// unreadable/malformed container, an archive that decompresses past
/// [`ComicOptions::max_archive_bytes`] ([`ComicError::ArchiveTooLarge`]), an
/// ambiguous PDF-only archive ([`ComicError::AmbiguousArchive`]), an embedded
/// PDF that cannot be opened ([`ComicError::Pdf`]), or an archive with no
/// decodable content in the page window ([`ComicError::NoImages`]).
pub fn open_comic(path: &Path, opts: &ComicOptions) -> Result<ComicPages, ComicError> {
    validate_options(opts)?;
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| ComicError::BadArchive("input path has no file name".to_owned()))?;
    let bytes = std::fs::read(path).map_err(ComicError::Open)?;

    let budget = archive::ArchiveBudget::new(opts.max_archive_bytes);
    let mut archive = archive::open_archive_from_ext(file_name, bytes, budget)?;

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
    // per-page Err in the second pass, preserving its reading-order slot. A
    // multi-page image (a TIFF) yields one page per image it holds.
    let image = |name, bytes, index| ImagePage {
        page: 0,
        name,
        bytes,
        index,
    };
    let mut decodable: Vec<ImagePage> = Vec::new();
    for name in image_names {
        match archive.take_entry(&name) {
            Ok(raw) if decode::sniff(&raw).is_some() => {
                let indices = decode::image_pages(&raw).unwrap_or_else(|_| vec![0]);
                let bytes = Arc::new(raw);
                let several = indices.len() > 1;
                for (n, index) in (1..).zip(indices) {
                    let label = if several {
                        format!("{name} (page {n})")
                    } else {
                        name.clone()
                    };
                    decodable.push(image(label, Ok(Arc::clone(&bytes)), index));
                }
            }
            Ok(_) => log::warn!(
                "comic: skipping {name}: unrecognised image format; convert it to PNG or JPG"
            ),
            // The budget bounds the whole archive, so exceeding it is not
            // the failure of one page.
            Err(e @ ComicError::ArchiveTooLarge { .. }) => return Err(e),
            Err(e) => decodable.push(image(name, Err(e), 0)),
        }
    }
    if decodable.is_empty() {
        return Err(ComicError::NoImages);
    }

    // Second pass: assign 1-based page numbers over kept entries and apply
    // the window. Decoding is deferred to `ComicPages::next`.
    let entries: VecDeque<_> = (1..=u32::MAX)
        .zip(decodable)
        .filter(|(page, _)| (opts.first_page..=opts.last_page).contains(page))
        .map(|(page, image)| ImagePage { page, ..image })
        .collect();
    if entries.is_empty() {
        // The window excluded every kept page.
        return Err(ComicError::NoImages);
    }
    Ok(ComicPages {
        opts: opts.clone(),
        source: Source::Images(entries),
    })
}

/// Reject options outside their documented constraints.
fn validate_options(opts: &ComicOptions) -> Result<(), ComicError> {
    if !(opts.dpi.is_finite() && opts.dpi > 0.0) {
        return Err(ComicError::InvalidOptions(format!(
            "dpi must be a positive finite number, got {}",
            opts.dpi
        )));
    }
    if opts.first_page == 0 {
        return Err(ComicError::InvalidOptions(
            "first_page must be ≥ 1 (pages are 1-based)".to_owned(),
        ));
    }
    if opts.max_archive_bytes == 0 {
        return Err(ComicError::InvalidOptions(
            "max_archive_bytes must be positive (it bounds the archive's decompressed size)"
                .to_owned(),
        ));
    }
    if opts.first_page > opts.last_page {
        return Err(ComicError::InvalidOptions(format!(
            "first_page ({}) > last_page ({})",
            opts.first_page, opts.last_page
        )));
    }
    Ok(())
}

/// Decode, guard, deskew, and assemble one page from already-read bytes. Errors
/// are returned (not panicked) so the caller yields them per-page and keeps
/// going. `raw` is guaranteed to sniff as a recognised image by the caller.
fn render_one(
    name: &str,
    raw: &[u8],
    index: usize,
    page_num: u32,
    opts: &ComicOptions,
) -> Result<RenderedPage, ComicError> {
    let target = decode::Target::new(opts.session.max_page_bytes, opts.depth);
    let page = decode::decode_image(raw, index, target).map_err(|e| match e {
        decode::DecodeError::TooLarge { width, height } => ComicError::TooLarge {
            entry: name.to_owned(),
            width,
            height,
        },
        other => ComicError::Decode {
            entry: name.to_owned(),
            reason: other.to_string(),
        },
    })?;
    let diagnostics = pdf_raster::PageDiagnostics::default();
    let (mut page, skew) = match page {
        decode::PageImage::Eight(bmp) => {
            let (bmp, skew) = prepared(bmp, opts)?;
            let page = pdf_raster::gray8_into_rendered_page(bmp, page_num, opts.dpi, diagnostics);
            (page, skew)
        }
        decode::PageImage::Sixteen(bmp) => {
            let (bmp, skew) = prepared(bmp, opts)?;
            let page = pdf_raster::gray16_into_rendered_page(bmp, page_num, opts.dpi, diagnostics);
            (page, skew)
        }
    };
    page.skew = skew;
    Ok(page)
}

/// `bmp` turned by the session's rotation, then deskewed when asked; the
/// skew deskew measured.
fn prepared<P: pdf_raster::deskew::DeskewPixel>(
    bmp: Bitmap<P>,
    opts: &ComicOptions,
) -> Result<(Bitmap<P>, Option<pdf_raster::deskew::Skew>), ComicError> {
    let mut bmp = turn::turn(bmp, opts.session.rotation);
    let skew = if opts.deskew {
        pdf_raster::deskew::apply(&mut bmp, opts.session.policy)
            .map_err(|e| ComicError::Deskew(e.to_string()))?
    } else {
        None
    };
    Ok((bmp, skew))
}

/// Open a single PDF entry as a session whose pages [`ComicPages`] renders
/// on demand, mapping errors into [`ComicError::Pdf`]. Page numbers are the
/// PDF's own 1-based page sequence — they are not renumbered.
fn render_embedded_pdf(
    archive: &mut dyn archive::Archive,
    name: &str,
    opts: &ComicOptions,
) -> Result<ComicPages, ComicError> {
    let bytes = archive.take_entry(name)?;
    let session = pdf_raster::open_session_from_bytes(bytes, &opts.session)
        .map_err(|e| ComicError::Pdf(format!("embedded PDF {name}: {e}")))?;
    let total = session.total_pages();
    if total == 0 {
        return Err(ComicError::Pdf(format!("embedded PDF {name} has no pages")));
    }
    let pages: VecDeque<u32> = (opts.first_page..=opts.last_page.min(total)).collect();
    if pages.is_empty() {
        return Err(ComicError::Pdf(format!(
            "embedded PDF {name} has {total} pages, none in {}..={}",
            opts.first_page, opts.last_page
        )));
    }
    Ok(ComicPages {
        opts: opts.clone(),
        source: Source::Pdf {
            session: Box::new(session),
            pages,
        },
    })
}

#[cfg(test)]
mod tests;
