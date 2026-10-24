//! PDF rasterisation library — zero subprocess, zero Leptonica.
//!
//! Converts PDF pages to 8-bit grayscale pixel buffers ready for Tesseract OCR
//! (or any other consumer) without writing files or spawning processes.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use std::path::Path;
//! use rasterrocket::{RasterOptions, raster_pdf};
//!
//! let opts = RasterOptions { dpi: 300.0, first_page: 1, last_page: 5, deskew: true, pages: None, strict: false };
//! for (page_num, result) in raster_pdf(Path::new("scan.pdf"), &opts) {
//!     match result {
//!         Ok(page) => {
//!             // page.pixels: Vec<u8>, 8-bit grayscale, width × height, top-to-bottom
//!             // pass to tesseract::ocr_from_frame(&page.pixels, page.width, page.height, 1, page.width, "eng")
//!         }
//!         Err(e) => eprintln!("page {page_num}: {e}"),
//!     }
//! }
//! ```
//!
//! To render a PDF already held in memory (e.g. extracted from an archive)
//! without writing a temp file, use [`raster_pdf_from_bytes`] — same iterator
//! contract as [`raster_pdf`], taking owned bytes instead of a path.
//!
//! For Google Cloud Vision pipelines, [`encode_for_gcv`] turns a
//! [`RenderedPage`] into a size-fitted [`GcvImage`] (JPEG bytes + fitting
//! metadata) guaranteed to stay within GCV's request limits; call
//! [`GcvImage::to_base64`] for the inline `content` string — no external
//! image crate, no network guesswork.
//!
//! # OCR integration
//!
//! `rasterrocket` is a rasteriser, not an OCR framework.  For Tesseract, ocrs,
//! Google Cloud Vision, GPT-5, and Mistral integration patterns — including the
//! `TessApi`-per-thread reuse pattern and zero-copy `set_image_from_mem` — see
//! the project wiki:
//!
//! - [OCR Integration](https://github.com/pthomasfournet/rasterrocket/wiki/OCR-Integration)
//!   (Tesseract via `leptess`, ocrs)
//! - [LLM Vision OCR Integration](https://github.com/pthomasfournet/rasterrocket/wiki/LLM-Vision-OCR-Integration)
//!   (Google Cloud Vision, GPT-5, Mistral)
//!
//! # Streaming: render and process pages in parallel
//!
//! For large documents, [`render_channel`] renders on a background rayon thread
//! and sends pages as they complete, so downstream work (e.g. OCR) can start
//! immediately rather than waiting for the full render:
//!
//! ```rust,no_run
//! use std::path::Path;
//! use rasterrocket::{RasterOptions, render_channel};
//!
//! // `rayon` is a transitive dependency of `rasterrocket`; add it explicitly
//! // to your Cargo.toml only if you call rayon APIs directly: rayon = "1"
//! //
//! // The consumer `recv` loop MUST run on the calling thread (not inside a
//! // `rayon::scope` or a worker closure): `render_channel`'s producer also
//! // runs on rayon's global pool, so a recv loop inside a scope task could
//! // deadlock if the pool fills with scope tasks all blocked on `recv`.
//! //
//! // `rayon::spawn` *inside* the recv-arm is safe — those tasks return
//! // without blocking on recv themselves.  For non-CPU consumer work
//! // (Tesseract OCR, disk I/O), prefer `std::thread::spawn` to keep the
//! // rayon pool free for the renderer.
//! let opts = RasterOptions {
//!     dpi: 300.0,
//!     first_page: 1,
//!     last_page: u32::MAX,
//!     deskew: true,
//!     pages: None,
//!     strict: false,
//! };
//! let rx = render_channel(Path::new("scan.pdf"), &opts, 4); // 4-page buffer
//! while let Ok((page_num, result)) = rx.recv() {
//!     match result {
//!         Ok(page) => rayon::spawn(move || {
//!             // process page — overlaps with the next render
//!             let _ = (page_num, page.pixels);
//!         }),
//!         Err(e) => eprintln!("page {page_num}: {e}"),
//!     }
//! }
//! // Note: rayon::spawn tasks may still be running here. Use a rayon scope
//! // or channel to join them if you need to wait for all work to complete.
//! ```
//!
//! # Sparse page selection
//!
//! To render only specific pages (e.g. a subset identified by a prior scan),
//! use [`PageSet`]:
//!
//! ```rust,no_run
//! use std::path::Path;
//! use rasterrocket::{PageSet, RasterOptions, raster_pdf};
//!
//! let pages = PageSet::new(vec![1, 5, 23, 100]).unwrap();
//! let opts = RasterOptions {
//!     dpi: 300.0,
//!     first_page: 1,        // ignored when pages is Some
//!     last_page: u32::MAX,  // ignored when pages is Some
//!     deskew: true,
//!     pages: Some(pages),
//!     strict: false,
//! };
//! for (page_num, result) in raster_pdf(Path::new("scan.pdf"), &opts) {
//!     // Only pages 1, 5, 23, and 100 are rendered — intermediates are skipped.
//!     match result {
//!         Ok(page) => { /* process page.pixels */ let _ = (page_num, page); }
//!         Err(e) => eprintln!("page {page_num}: {e}"),
//!     }
//! }
//! ```

mod config;
#[cfg(feature = "vaapi")]
pub(crate) mod decode_queue;
pub mod deskew;
pub mod gcv;
pub(crate) mod gpu_init;
mod gray;
pub mod page;
mod render;
#[cfg(feature = "vulkan")]
mod vk_init;

use std::path::Path;

pub use config::{BackendPolicy, DEFAULT_VAAPI_DEVICE, SessionConfig};
pub use gcv::{GcvBudget, GcvError, GcvImage, encode_for_gcv};
pub use gray::rgb_to_gray;
pub use page::{MAX_PX_AREA, MAX_PX_DIMENSION, gray8_to_rendered_page, validate_dimensions};
pub use pdf_interp::renderer::{ImageFailure, PageDiagnostics};
pub use pdf_interp::resources::ImageFilter;
pub use render::{
    RasterError, RasterSession, open_session, open_session_from_bytes, prescan_session,
    render_page_rgb, render_page_rgb_hinted,
};

/// Session-level API for explicit control over PDF opening and per-page rendering.
///
/// Use these when [`raster_pdf`] or [`render_channel`] don't give you enough
/// control — e.g. when you need to share one open document across multiple
/// threads, customise the [`SessionConfig`] (GPU backend, VA-API device, cache
/// prefetch), or call [`prescan_session`] before deciding how to render each page.
///
/// All items in this module are also re-exported at the crate root for backward
/// compatibility.
pub mod session {
    pub use super::{
        open_session, open_session_from_bytes, prescan_session, render_page_rgb,
        render_page_rgb_hinted, rgb_to_gray,
    };
}

/// Eagerly release GPU decoders on every rayon worker thread.
///
/// Call this via `pool.broadcast` **before** dropping `pool` or allowing
/// process exit to proceed:
///
/// ```rust,no_run
/// # use rasterrocket::release_gpu_decoders;
/// # let pool: rayon::ThreadPool = todo!();
/// let _ = pool.broadcast(|_| release_gpu_decoders());
/// drop(pool);
/// ```
///
/// This drops each thread's `NvJpegDecoder` / `NvJpeg2kDecoder` / Vulkan
/// Huffman decoder while the CUDA driver is still fully alive.  Without
/// this call, all workers call their destructors concurrently at process
/// exit into a driver that has already started its own `atexit` shutdown,
/// causing undefined behaviour (typically a segfault or hang).
///
/// After this call the TLS slots hold `Uninitialised`, so their own
/// destructors at process exit are no-ops.
///
/// # When to call
///
/// - Multi-page pipelines that use an explicit `rayon::ThreadPool` — call
///   once on the pool after rendering completes, before dropping the pool.
/// - Single-page or short runs using the global rayon pool — calling is
///   optional but harmless; `raster_pdf` and `render_channel` do not call
///   it automatically (they don't own the pool).
///
/// # No-op builds
///
/// When none of the GPU decoder features (`nvjpeg`, `nvjpeg2k`,
/// `gpu-jpeg-huffman + vulkan`) are compiled in, this function is a
/// no-op and can be omitted entirely.  It is always safe to call.
///
/// # Panics
///
/// Never panics.
#[cfg_attr(
    not(any(
        feature = "nvjpeg",
        feature = "nvjpeg2k",
        all(feature = "gpu-jpeg-huffman", feature = "vulkan"),
    )),
    expect(
        clippy::missing_const_for_fn,
        reason = "body collapses to empty when no GPU-decoder feature is on"
    )
)]
pub fn release_gpu_decoders() {
    #[cfg(feature = "nvjpeg")]
    gpu_init::release_nvjpeg_this_thread();
    #[cfg(feature = "nvjpeg2k")]
    gpu_init::release_nvjpeg2k_this_thread();
    #[cfg(all(feature = "gpu-jpeg-huffman", feature = "vulkan"))]
    gpu_init::release_jpeg_vk_this_thread();
}

/// True when the PDF at `path` is encrypted (uses the PDF Standard
/// Security Handler — an `/Encrypt` entry in the trailer).
///
/// A cheap structural probe: only the xref table and trailer are parsed,
/// no objects are decoded.  Returns `false` (rather than erroring) when
/// the file cannot be opened or parsed — callers that need a hard parse
/// error get it from [`open_session`] proper; this probe exists solely so
/// the CLI can decide whether to show its decryption liability prompt
/// without paying a full session open.
#[must_use]
pub fn is_encrypted(path: &Path) -> bool {
    pdf::Document::open(path).is_ok_and(|d| d.is_encrypted())
}

/// Message shown when an encrypted document's decryption is not authorised.
///
/// Used when the CLI liability gate is declined, or on a non-interactive
/// run with no operator bypass.  Never the misleading "document has no
/// pages".
#[must_use]
pub fn decrypt_gate_declined_message() -> String {
    pdf::msg_gate_declined()
}

// ── PageSet ───────────────────────────────────────────────────────────────────

/// A validated, sorted, deduplicated set of 1-based page numbers.
///
/// Constructed via [`PageSet::new`].  Clone is O(1) — the underlying storage is
/// reference-counted.  Use as the [`RasterOptions::pages`] field to render a
/// sparse subset of pages without visiting intermediate ones.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageSet(std::sync::Arc<[u32]>);

impl PageSet {
    /// Construct a `PageSet` from any collection of 1-based page numbers.
    ///
    /// Accepts `Vec<u32>`, arrays, slices (via `.to_vec()`), and any
    /// `IntoIterator<Item = u32>`.  The input is sorted and deduplicated
    /// before storage.
    ///
    /// # Errors
    ///
    /// Returns [`RasterError::InvalidOptions`] if the resulting set is empty
    /// (all-zeros input counts) or any value is 0.
    pub fn new(pages: impl IntoIterator<Item = u32>) -> Result<Self, RasterError> {
        let mut v: Vec<u32> = pages.into_iter().collect();
        v.sort_unstable();
        v.dedup();
        if v.is_empty() {
            return Err(RasterError::InvalidOptions(
                "PageSet must contain at least one page".to_owned(),
            ));
        }
        if v[0] == 0 {
            return Err(RasterError::InvalidOptions(
                "PageSet contains page 0 — pages are 1-based".to_owned(),
            ));
        }
        Ok(Self(v.into_boxed_slice().into()))
    }

    /// Returns `true` if `page` is in this set.  O(log n).
    #[must_use]
    pub fn contains(&self, page: u32) -> bool {
        self.0.binary_search(&page).is_ok()
    }

    /// The smallest page number in the set.
    ///
    /// # Panics
    ///
    /// Never panics on a correctly constructed `PageSet` — non-emptiness is a
    /// structural invariant enforced by [`PageSet::new`].
    #[must_use]
    pub fn first(&self) -> u32 {
        *self.0.first().expect("PageSet is non-empty by invariant")
    }

    /// The largest page number in the set.
    ///
    /// # Panics
    ///
    /// Never panics on a correctly constructed `PageSet` — non-emptiness is a
    /// structural invariant enforced by [`PageSet::new`].
    #[must_use]
    pub fn last(&self) -> u32 {
        *self.0.last().expect("PageSet is non-empty by invariant")
    }

    /// Number of pages in the set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Iterate the pages in ascending order.
    pub fn iter(&self) -> std::iter::Copied<std::slice::Iter<'_, u32>> {
        self.0.iter().copied()
    }

    /// View the pages as a sorted, deduplicated slice.  O(1).
    #[must_use]
    pub fn as_slice(&self) -> &[u32] {
        &self.0
    }

    /// Always returns `false`.  A `PageSet` is guaranteed non-empty by
    /// construction (see [`PageSet::new`]); this method exists to satisfy the
    /// `clippy::len_without_is_empty` lint.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }
}

impl<'a> IntoIterator for &'a PageSet {
    type Item = u32;
    type IntoIter = std::iter::Copied<std::slice::Iter<'a, u32>>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// ── Render options & output ───────────────────────────────────────────────────

/// Options controlling how pages are rendered.
#[derive(Debug, Clone)]
pub struct RasterOptions {
    /// Must be > 0.  Pass this same value to Tesseract's `set_source_resolution`
    /// — lying about DPI degrades OCR accuracy because Tesseract uses it for
    /// internal scaling.  Recommended: 300 DPI for scanned documents.
    pub dpi: f32,

    /// First page (1-based, inclusive).  Must be ≥ 1.  Ignored when
    /// [`pages`](Self::pages) is `Some` — the render window is then derived
    /// from the `PageSet` bounds.
    pub first_page: u32,

    /// Last page (1-based, inclusive).  Must be ≥ `first_page`.  If it exceeds
    /// the document's page count, rendering stops at the last page in the
    /// document rather than returning an error.  Ignored when
    /// [`pages`](Self::pages) is `Some`.
    pub last_page: u32,

    /// Uses an intensity-weighted projection-profile sweep (no binarisation
    /// threshold) with GPU bilinear rotation via CUDA NPP (`gpu-deskew` feature)
    /// or CPU bilinear fallback.  Corrects skew up to ±7° with sub-0.05°
    /// accuracy.  Disable for native-text PDFs that are never physically skewed.
    pub deskew: bool,

    /// Sparse page selection.  When `Some`, only the pages in the [`PageSet`]
    /// are rendered and yielded; intermediate pages are skipped without
    /// rendering.  When `None`, all pages in `first_page..=last_page` are
    /// rendered.
    pub pages: Option<PageSet>,

    /// Whether an image that cannot be decoded or drawn fails its page; see
    /// [`SessionConfig::strict`].  Default `false`: the image is skipped and
    /// reported in [`PageDiagnostics::image_failures`].
    pub strict: bool,
}

impl Default for RasterOptions {
    fn default() -> Self {
        Self {
            dpi: 300.0,
            first_page: 1,
            last_page: u32::MAX,
            deskew: false,
            pages: None,
            strict: false,
        }
    }
}

/// A single rendered page, returned as 8-bit grayscale pixels.
#[derive(Debug)]
pub struct RenderedPage {
    /// Page number (1-based).
    pub page_num: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Raw pixel bytes: 8-bit grayscale, tightly packed (`stride == width`),
    /// top-to-bottom, left-to-right.  Length is `width * height`.
    pub pixels: Vec<u8>,
    /// The DPI at which this page was rendered (`opts.dpi`).
    ///
    /// This is the raw render resolution, ignoring any `UserUnit` scaling in the
    /// document.  Use [`effective_dpi`](Self::effective_dpi) — not this field — when
    /// reporting resolution to downstream consumers such as Tesseract.
    pub dpi: f32,
    /// Physical resolution of the rendered bitmap, accounting for `UserUnit` scaling.
    ///
    /// `effective_dpi = opts.dpi × UserUnit`.  For the vast majority of documents
    /// `UserUnit` is 1.0 and this equals `dpi`.  Always pass this value to
    /// `tesseract::set_source_resolution`; lying about DPI degrades OCR accuracy
    /// because Tesseract uses it for internal feature scaling.
    ///
    /// Omitting the resolution call causes Tesseract to fall back to 70 DPI, which
    /// severely degrades recognition accuracy.
    pub effective_dpi: f32,
    /// Lightweight metadata collected at zero extra cost during rendering.
    ///
    /// Use this to route pages to different OCR configurations — e.g. skip deskew
    /// on `has_vector_text = true` pages, or set Tesseract PSM based on `is_scan`.
    pub diagnostics: PageDiagnostics,
}

// ── Top-level render entry points ─────────────────────────────────────────────

/// Render a range of pages from a PDF file.
///
/// Returns an iterator yielding `(page_num, Result<RenderedPage, RasterError>)`
/// for each page in `opts.first_page..=opts.last_page`.  A per-page error does
/// not abort remaining pages — the caller decides whether to skip or propagate.
///
/// GPU resources are initialised lazily on first use and reused across pages.
/// Backend selection follows [`SessionConfig::default`] — Vulkan when present,
/// CUDA next, CPU last; the `PDF_RASTER_BACKEND` env var overrides the
/// default.  Use [`open_session`] + [`render_page_rgb`] directly when you
/// need explicit [`SessionConfig`] control.
///
/// # Errors
///
/// - [`RasterError::InvalidOptions`] — `opts` violates documented constraints.
/// - [`RasterError::Pdf`] — document cannot be opened or parsed.
/// - [`RasterError::PageOutOfRange`] — requested page exceeds the document.
/// - [`RasterError::PageDegenerate`] / [`RasterError::PageTooLarge`] — malformed geometry.
/// - [`RasterError::Deskew`] — deskew rotation failed.
/// - [`RasterError::BackendUnavailable`] — `PDF_RASTER_BACKEND` requested a
///   forced backend (e.g. `cuda`, `vulkan`) and that backend's runtime
///   is unavailable.  Cannot occur if the env var is unset or `auto`.
///
/// Per-page render errors are yielded as `(page_num, Err(...))` and the
/// iterator advances to the next page.  If the renderer panics on a page,
/// the panic propagates out of `Iterator::next` — this iterator does not
/// catch panics.  Use [`render_channel`] for panic-isolated delivery under
/// `panic = "unwind"` builds.
pub fn raster_pdf(
    path: &Path,
    opts: &RasterOptions,
) -> impl Iterator<Item = (u32, Result<RenderedPage, RasterError>)> {
    render::render_pages(path, opts)
}

/// Render a range of pages from an in-memory PDF (no file path).
///
/// Identical to [`raster_pdf`] but takes owned PDF bytes — for callers that
/// already hold the document (e.g. a PDF extracted from an archive) and want to
/// avoid writing a temp file.  Backend selection follows
/// [`SessionConfig::default`].
///
/// # Errors
///
/// Same as [`raster_pdf`]; [`RasterError::Pdf`] if `bytes` is not a parseable
/// PDF.  An encrypted in-memory PDF surfaces the encrypted error — the
/// in-memory path does not go through the transparent-decrypt gate.
pub fn raster_pdf_from_bytes(
    bytes: Vec<u8>,
    opts: &RasterOptions,
) -> impl Iterator<Item = (u32, Result<RenderedPage, RasterError>)> {
    render::render_pages_from_bytes(bytes, opts)
}

/// Render a range of pages concurrently using a bounded sync channel.
///
/// Spawns a background Rayon task that renders pages in ascending order and
/// sends each `(page_num, Result<RenderedPage, RasterError>)` to the returned
/// [`Receiver`](std::sync::mpsc::Receiver) as it completes.
///
/// `capacity` is the maximum number of rendered pages buffered before the
/// producer blocks.  Use `2`–`8` for typical OCR pipelines (one page rendering
/// while the previous is being OCR-processed).  `capacity = 0` is silently
/// raised to `1`.
///
/// # Errors delivered through the channel
///
/// - Invalid options → `(1, Err(RasterError::InvalidOptions(...)))`, channel closes.
/// - File open failure → `(1, Err(RasterError::Pdf(...)))`, channel closes.
/// - Forced backend unavailable (e.g. `PDF_RASTER_BACKEND=cuda` with no
///   CUDA driver) → `(1, Err(RasterError::BackendUnavailable(...)))`,
///   channel closes.
/// - Per-page render errors → `(page_num, Err(...))`, rendering of
///   subsequent pages continues.
/// - Per-page panics → delivered as `(page_num, Err(RasterError::RenderPanic
///   { .. }))` and rendering continues, **but only under `panic = "unwind"`
///   builds** (test profiles, and embedders that override the panic strategy).
///   Under the crate's default `panic = "abort"` release profile a panicking
///   page aborts the entire process before any recovery is possible.
///   Panic-freedom on the supported corpus is pursued by hardening the
///   renderer to return errors rather than panicking.
#[must_use]
pub fn render_channel(
    path: &Path,
    opts: &RasterOptions,
    capacity: usize,
) -> std::sync::mpsc::Receiver<(u32, Result<RenderedPage, RasterError>)> {
    render::render_channel(path, opts, capacity)
}

#[cfg(test)]
mod page_set_tests {
    use super::*;

    #[test]
    fn empty_input_is_rejected() {
        assert!(matches!(
            PageSet::new(vec![]),
            Err(RasterError::InvalidOptions(_))
        ));
    }

    #[test]
    fn zero_page_is_rejected() {
        assert!(matches!(
            PageSet::new(vec![0, 1, 2]),
            Err(RasterError::InvalidOptions(_))
        ));
    }

    #[test]
    fn valid_input_is_accepted() {
        let ps = PageSet::new(vec![3, 1, 2]).unwrap();
        assert_eq!(ps.first(), 1);
        assert_eq!(ps.last(), 3);
        assert_eq!(ps.len(), 3);
        assert!(!ps.is_empty());
    }

    #[test]
    fn duplicates_are_deduplicated() {
        let ps = PageSet::new(vec![2, 1, 2, 3, 1]).unwrap();
        assert_eq!(ps.len(), 3);
    }

    #[test]
    fn contains_works() {
        let ps = PageSet::new(vec![1, 5, 10]).unwrap();
        assert!(ps.contains(1));
        assert!(ps.contains(5));
        assert!(ps.contains(10));
        assert!(!ps.contains(2));
        assert!(!ps.contains(11));
    }

    #[test]
    fn clone_is_cheap() {
        let ps = PageSet::new(vec![1, 2, 3]).unwrap();
        let ps2 = ps.clone();
        // Both point to the same allocation — Arc pointer equality
        assert!(std::ptr::eq(ps.0.as_ptr(), ps2.0.as_ptr()));
    }

    #[test]
    fn sole_zero_is_rejected() {
        assert!(matches!(
            PageSet::new(vec![0u32]),
            Err(RasterError::InvalidOptions(_))
        ));
    }

    #[test]
    fn single_max_page_is_accepted() {
        let ps = PageSet::new(vec![u32::MAX]).unwrap();
        assert_eq!(ps.first(), u32::MAX);
        assert_eq!(ps.last(), u32::MAX);
        assert_eq!(ps.len(), 1);
        assert!(!ps.is_empty());
        assert!(ps.contains(u32::MAX));
        assert!(!ps.contains(u32::MAX - 1));
    }

    #[test]
    fn accepts_iterator_input() {
        // IntoIterator covers slices, arrays, ranges, etc.
        let ps = PageSet::new([3u32, 1, 2]).unwrap();
        assert_eq!(ps.len(), 3);
        assert_eq!(ps.first(), 1);
    }

    #[test]
    fn equality_is_value_based() {
        let a = PageSet::new(vec![1, 2, 3]).unwrap();
        let b = PageSet::new([3u32, 1, 2]).unwrap(); // same content, different origin
        assert_eq!(a, b);
    }

    #[test]
    fn raster_options_with_pages_none_is_valid() {
        let opts = RasterOptions {
            dpi: 150.0,
            first_page: 1,
            last_page: 5,
            deskew: false,
            pages: None,
            strict: false,
        };
        assert!(opts.pages.is_none());
        let default_opts = RasterOptions::default();
        assert_eq!(default_opts.dpi, 300.0);
        assert_eq!(default_opts.first_page, 1);
        assert_eq!(default_opts.last_page, u32::MAX);
        assert!(!default_opts.deskew);
        assert!(default_opts.pages.is_none());
    }

    #[test]
    fn raster_options_with_pages_some_is_valid() {
        let ps = PageSet::new(vec![1, 3, 5]).unwrap();
        let opts = RasterOptions {
            dpi: 150.0,
            first_page: 1,
            last_page: 5,
            deskew: false,
            pages: Some(ps),
            strict: false,
        };
        assert!(opts.pages.is_some());
    }
}
