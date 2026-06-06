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

/// Placeholder so the freshly-scaffolded crate compiles before the public API
/// lands.
#[doc(hidden)]
pub const fn __scaffold() {}
