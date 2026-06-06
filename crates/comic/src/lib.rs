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

use std::path::Path;

pub use pdf_raster::RenderedPage;

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

/// Open a comic archive and iterate its pages as grayscale [`RenderedPage`]s.
///
/// Yields `(page_num, Result<RenderedPage, ComicError>)` in reading order. A
/// per-page decode failure is yielded as `Err` and iteration continues — one
/// corrupt image never aborts the archive (matching `rasterrocket::raster_pdf`).
///
/// # Errors
///
/// Returns `Err` *before* iteration for whole-archive failures: a `.cbr` input
/// ([`ComicError::RarUnsupported`]), an unreadable/malformed container, or an
/// archive with no decodable images ([`ComicError::NoImages`]).
#[expect(
    clippy::type_complexity,
    reason = "the nested Result tuple is the deliberate public API; aliasing it would obscure the shape for callers"
)]
#[expect(
    clippy::missing_const_for_fn,
    reason = "stub only; the real implementation performs I/O and cannot be const"
)]
pub fn open_comic(
    path: &Path,
    opts: &ComicOptions,
) -> Result<Vec<(u32, Result<RenderedPage, ComicError>)>, ComicError> {
    // Stub: the eager Vec keeps the public signature testable until archive
    // reading and decode are wired in.
    let _ = (path, opts);
    Err(ComicError::NoImages)
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
}
