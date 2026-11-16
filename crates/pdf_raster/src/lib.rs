//! PDF rasterisation library — zero subprocess, zero Leptonica.
//!
//! Converts PDF pages to 8-bit grayscale pixel buffers ready for Tesseract OCR
//! (or any other consumer) without writing files or spawning processes.
//! [`RasterOptions`] also selects RGBA pages over a transparent backdrop,
//! CMYK pages ([`ColorModel::Cmyk`]), 16 bits per sample
//! ([`SampleDepth::Sixteen`]) and the conversion to an ICC output profile
//! ([`OutputProfile`]); [`render_page_bands`] renders a page larger than
//! memory as bands of rows.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use std::path::Path;
//! use rasterrocket::{RasterOptions, raster_pdf};
//!
//! let mut opts = RasterOptions::new(300.0);
//! opts.last_page = 5;
//! opts.deskew = true;
//! for (page_num, result) in raster_pdf(Path::new("scan.pdf"), &opts) {
//!     match result {
//!         Ok(page) => {
//!             // page.pixels.as_u8(): 8-bit grayscale, width × height, top-to-bottom
//!             // pass to tesseract::ocr_from_frame(pixels, page.width, page.height, 1, page.width, "eng")
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
//! let mut opts = RasterOptions::new(300.0);
//! opts.deskew = true;
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
//! let mut opts = RasterOptions::new(300.0);
//! opts.deskew = true;
//! opts.pages = Some(pages); // first_page / last_page are then ignored
//! for (page_num, result) in raster_pdf(Path::new("scan.pdf"), &opts) {
//!     // Only pages 1, 5, 23, and 100 are rendered — intermediates are skipped.
//!     match result {
//!         Ok(page) => { /* process page.pixels */ let _ = (page_num, page); }
//!         Err(e) => eprintln!("page {page_num}: {e}"),
//!     }
//! }
//! ```

mod bands;
mod batch;
mod config;
#[cfg(feature = "vaapi")]
pub(crate) mod decode_queue;
mod decoders;
pub mod deskew;
#[cfg(feature = "vulkan")]
mod device_pages;
mod error;
pub(crate) mod gcv;
pub(crate) mod gpu_init;
mod gray;
mod open;
mod output;
pub(crate) mod page;
mod page_buffer;
mod proof;
mod render;
mod request;
#[cfg(test)]
mod test_pdf;
#[cfg(feature = "vulkan")]
mod vk_init;

use std::path::Path;

/// Development tools of the library's own tests and benchmarks; not part
/// of its stable interface, and changed without notice.
#[doc(hidden)]
pub mod dev {
    use crate::{RasterError, RasterSession, RenderParams};

    /// The paint trace of page `page_num` (1-based) rendered as `params`
    /// describes.
    ///
    /// The trace holds one line of text per painting operation of the
    /// page, in the page's device space at the render's scale, and is the
    /// same whatever band height or region `params` gives.
    ///
    /// # Errors
    ///
    /// As [`crate::render_page_bands`].
    pub fn paint_trace(
        session: &RasterSession,
        page_num: u32,
        params: &RenderParams,
    ) -> Result<String, RasterError> {
        crate::bands::paint_trace(session, page_num, params)
    }
}

pub use bands::{Band, BandBitmap, BandError, render_page_bands};
pub use color::SampleDepth;
pub use config::{BackendPolicy, DEFAULT_VAAPI_DEVICE, ParseBackendPolicyError, SessionConfig};
pub use error::RasterError;
pub use gcv::{GcvBudget, GcvError, GcvImage, encode_for_gcv};
pub use gray::{rgb_into_gray, rgb_to_gray, rgb16_into_gray16, rgb16_to_gray16};
pub use open::{
    RasterSession, open_session, open_session_from_bytes, open_session_from_reader,
    open_session_from_shared, prescan_session,
};
pub use page::{
    CanvasLayout, DEFAULT_BAND_BYTES, DEFAULT_MAX_PAGE_BYTES, MAX_PX_SIDE,
    gray8_into_rendered_page, gray8_to_rendered_page, gray16_into_rendered_page,
    validate_page_bytes,
};
pub use page_buffer::{PageBuffer, PageInfo};
pub use pdf::Password;
pub use pdf_interp::renderer::FinishedCanvas;
pub use pdf_interp::renderer::{
    ContentFailure, DOCUMENT_IMAGE_BUDGET_BYTES, DevicePage, GlyphCounts, ImageFailure,
    MAX_FORM_DEPTH, PageDiagnostics,
};
pub use pdf_interp::resources::ImageFilter;
pub use pdf_interp::{
    FormType, InvalidRotation, MediaKind, PageBox, PageGeometry, RenderPurpose, Rotation,
    ScriptLocation, UnsupportedFeature,
};
pub use pdf_interp::{
    IntentSet, OutputIntentError, OutputProfile, OutputProfileError, OutputSpace, ProofIntent,
    RenderingIntent, WorkingSpace,
};
pub use render::{
    render_page_canvas, render_page_canvas_with_diagnostics, render_page_cmyk, render_page_gray,
    render_page_rgb, render_page_rgb_with, render_page_rgb_with_diagnostics, render_page_rgba,
};
pub use request::{CancelToken, ColorModel, RenderParams, RenderRegion};

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
/// This drops each thread's `NvJpegDecoder` / `NvJpeg2kDecoder` while the CUDA driver is still fully alive.  Without
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
/// When neither GPU decoder feature (`nvjpeg`, `nvjpeg2k`) is compiled in, this function is a
/// no-op and can be omitted entirely.  It is always safe to call.
///
/// # Panics
///
/// Never panics.
#[cfg_attr(
    not(any(feature = "nvjpeg", feature = "nvjpeg2k")),
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
}

/// True when the PDF at `path` is encrypted (uses the PDF Standard
/// Security Handler — an `/Encrypt` entry in the trailer).
///
/// A cheap structural probe: only the xref table and trailer are parsed,
/// no objects are decoded and no password is needed.  Returns `false`
/// (rather than erroring) when the file cannot be opened or parsed —
/// callers that need a hard parse error get it from [`open_session`]
/// proper; this probe exists solely so the CLI can decide whether to show
/// its decryption liability prompt without paying a full session open.
#[must_use]
pub fn is_encrypted(path: &Path) -> bool {
    pdf::is_encrypted(path).unwrap_or(false)
}

/// [`is_encrypted`] for a document held in memory, such as one read from
/// standard input; `false` when `bytes` do not parse.
#[must_use]
pub fn is_encrypted_bytes(bytes: &[u8]) -> bool {
    pdf::is_encrypted_bytes(bytes).unwrap_or(false)
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
#[expect(
    clippy::struct_excessive_bools,
    reason = "each bool is an independent caller-facing switch"
)]
#[non_exhaustive]
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

    /// Straighten gray pages: measure the skew of each by an
    /// intensity-weighted projection-profile sweep (no binarisation
    /// threshold) and rotate it bilinearly, on the GPU via CUDA NPP under
    /// [`BackendPolicy::ForceCuda`] in a build with the `gpu-deskew`
    /// feature, on the CPU otherwise.  Measures skew within ±6.75° to about
    /// `2 / W` radians on a page `W` pixels wide (0.045° for A4 at
    /// 300 dpi); a page without a confident angle, such as a photograph or
    /// a blank page, is left as it is.  The measurement is reported in
    /// [`RenderedPage::skew`].  See [`deskew`] for the method and limits.
    pub deskew: bool,

    /// Sparse page selection.  When `Some`, only the pages in the [`PageSet`]
    /// are rendered and yielded; intermediate pages are skipped without
    /// rendering.  When `None`, all pages in `first_page..=last_page` are
    /// rendered.
    pub pages: Option<PageSet>,

    /// Whether an image or a content stream that cannot be decoded, or
    /// decodes only in part, fails its page; see [`SessionConfig::strict`].
    /// Default `false`: it is skipped (or drawn in part) and reported in [`PageDiagnostics::image_failures`] or
    /// [`PageDiagnostics::content_failures`].
    pub strict: bool,

    /// What the pages are rendered for; see [`SessionConfig::purpose`].
    /// Default [`RenderPurpose::View`].
    pub purpose: RenderPurpose,

    /// Whether annotations are drawn; see [`SessionConfig::annotations`].
    /// Default `true`.
    pub annotations: bool,

    /// Whether glyphs are antialiased; see [`SessionConfig::text_antialias`].
    /// Default `true`.
    pub text_antialias: bool,

    /// Whether paths are antialiased; see [`SessionConfig::vector_antialias`].
    /// Default `true`.
    pub vector_antialias: bool,

    /// Whether overprint is simulated; see [`SessionConfig::overprint`].
    /// Default `false`.
    pub overprint: bool,

    /// Whether an encrypted document may be decrypted; see
    /// [`SessionConfig::decrypt_authorized`].  Default `false`: an encrypted
    /// document is an error.
    pub decrypt_authorized: bool,

    /// Password for an encrypted document; see [`SessionConfig::password`].
    /// Default `None`.  No effect unless
    /// [`decrypt_authorized`](Self::decrypt_authorized) is `true`.
    pub password: Option<Password>,

    /// Whether [`raster_pdf`] and [`render_channel`] copy the file into
    /// memory instead of memory-mapping it; see
    /// [`SessionConfig::read_into_memory`].  Default `false`.  `true` keeps
    /// a truncation of the file by another process from raising `SIGBUS`
    /// during the render.
    pub read_into_memory: bool,

    /// Extra directories of installed fonts, searched before those
    /// `RROCKET_FONT_DIR` names and the standard ones; see
    /// [`SessionConfig::font_dirs`].  Default empty.
    pub font_dirs: Vec<std::path::PathBuf>,

    /// Whether the standard font directories are searched; see
    /// [`SessionConfig::system_fonts`].  Default `true`.
    pub system_fonts: bool,

    /// Host bytes of decoded images kept for reuse by later pages; see
    /// [`SessionConfig::image_budget`].  Default
    /// [`DOCUMENT_IMAGE_BUDGET_BYTES`]; 0 keeps nothing.
    pub image_budget: usize,

    /// The most bytes one page bitmap may hold; see
    /// [`SessionConfig::max_page_bytes`].  Default [`DEFAULT_MAX_PAGE_BYTES`].
    pub max_page_bytes: u64,

    /// The page boundary box each page is measured and rendered on; see
    /// [`SessionConfig::page_box`].  Default [`PageBox::Crop`].
    pub page_box: PageBox,

    /// A rotation added to each page's `/Rotate`; see
    /// [`SessionConfig::rotation`].  Default [`Rotation::None`].
    pub rotation: Rotation,

    /// Whether pages are rendered over a transparent backdrop instead of
    /// white paper.  Default `false`: pages are 8-bit gray.  `true` yields
    /// [`PixelFormat::Rgba8`] pages whose alpha is the page's coverage ×
    /// opacity, colour not premultiplied by it; a pixel nothing paints is
    /// `[0, 0, 0, 0]`.  Incompatible with [`deskew`](Self::deskew).
    pub transparent_background: bool,

    /// The depth of the pages' samples (see [`RenderParams::depth`]).
    /// Default [`SampleDepth::Eight`]; [`SampleDepth::Sixteen`] yields
    /// [`PixelFormat::Gray16`] pages, or [`PixelFormat::Rgba16`] over a
    /// transparent backdrop, of [`Pixels::U16`].  Incompatible with
    /// [`deskew`](Self::deskew).
    pub depth: SampleDepth,

    /// The ICC profile the pages are converted to; see
    /// [`SessionConfig::output_profile`].  Default `None`.  The gray pages
    /// take a gray profile, the RGBA pages of
    /// [`transparent_background`](Self::transparent_background) an RGB one,
    /// the CMYK pages of [`color`](Self::color) a CMYK one.
    pub output_profile: Option<std::sync::Arc<OutputProfile>>,

    /// The rendering intent of the conversion to
    /// [`output_profile`](Self::output_profile); see
    /// [`RenderParams::intent`].  Default `None`.
    pub intent: Option<RenderingIntent>,

    /// Whether the conversion to [`output_profile`](Self::output_profile)
    /// compensates the black point; see
    /// [`RenderParams::black_point_compensation`].  Default `false`.
    pub black_point_compensation: bool,

    /// Soft-proof every page against its output intent; see
    /// [`SessionConfig::proof`].  Default `None`.  Incompatible with
    /// [`transparent_background`](Self::transparent_background) and CMYK
    /// [`color`](Self::color).
    pub proof: Option<ProofIntent>,

    /// The colour model of the pages (see [`RenderParams::color`]).
    /// Default [`ColorModel::Rgb`]: gray pages, or RGBA over a transparent
    /// backdrop; [`ColorModel::Cmyk`] yields [`PixelFormat::Cmyk8`] pages,
    /// or [`PixelFormat::Cmyk16`] at [`depth`](Self::depth) 16.
    /// Incompatible with [`deskew`](Self::deskew) and
    /// [`transparent_background`](Self::transparent_background).
    pub color: ColorModel,

    /// Cancellation flag polled while each page renders; default `None`.
    ///
    /// Once it is raised, the page in progress stops before its next
    /// content-stream operator and is yielded as [`RasterError::Cancelled`],
    /// and the iteration (or channel) then ends: no later page is rendered.
    pub cancel: Option<CancelToken>,
}

impl RasterOptions {
    /// Options for rendering every page at `dpi`, every other field at its
    /// [`Default`] value.
    ///
    /// The struct is `#[non_exhaustive]`: start from this or from
    /// [`RasterOptions::default`] and assign the fields to change.
    #[must_use]
    pub fn new(dpi: f32) -> Self {
        Self {
            dpi,
            ..Self::default()
        }
    }
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
            purpose: RenderPurpose::View,
            annotations: true,
            text_antialias: true,
            vector_antialias: true,
            overprint: false,
            decrypt_authorized: false,
            password: None,
            read_into_memory: false,
            font_dirs: Vec::new(),
            system_fonts: true,
            image_budget: DOCUMENT_IMAGE_BUDGET_BYTES,
            max_page_bytes: DEFAULT_MAX_PAGE_BYTES,
            page_box: PageBox::Crop,
            rotation: Rotation::None,
            transparent_background: false,
            depth: SampleDepth::Eight,
            output_profile: None,
            intent: None,
            black_point_compensation: false,
            proof: None,
            color: ColorModel::Rgb,
            cancel: None,
        }
    }
}

/// The layout of the pixels of a [`RenderedPage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PixelFormat {
    /// One byte of gray per pixel, 0 black.
    Gray8,
    /// Four bytes per pixel, `[R, G, B, A]` in sRGB; alpha 0 is
    /// transparent and the colour is not premultiplied by it.
    Rgba8,
    /// Three bytes per pixel, `[R, G, B]` in sRGB.  Written by
    /// [`PageBuffer::render_rgb_into`]; a [`RenderedPage`] is never in it.
    Rgb8,
    /// One 16-bit sample of gray per pixel, 0 black.
    Gray16,
    /// Three 16-bit samples per pixel, `[R, G, B]` in sRGB.
    Rgb16,
    /// Four 16-bit samples per pixel, `[R, G, B, A]` in sRGB, alpha not
    /// premultiplied.
    Rgba16,
    /// Four bytes per pixel, `[C, M, Y, K]`, 0 no ink.
    Cmyk8,
    /// Four 16-bit samples per pixel, `[C, M, Y, K]`, 0 no ink.
    Cmyk16,
}

impl PixelFormat {
    /// Bytes per pixel.
    #[must_use]
    pub const fn bytes_per_pixel(self) -> usize {
        self.samples_per_pixel() * self.depth().bytes() as usize
    }

    /// Samples per pixel.
    #[must_use]
    pub const fn samples_per_pixel(self) -> usize {
        match self {
            Self::Gray8 | Self::Gray16 => 1,
            Self::Rgb8 | Self::Rgb16 => 3,
            Self::Rgba8 | Self::Rgba16 | Self::Cmyk8 | Self::Cmyk16 => 4,
        }
    }

    /// The depth of the samples.
    #[must_use]
    pub const fn depth(self) -> SampleDepth {
        match self {
            Self::Gray8 | Self::Rgb8 | Self::Rgba8 | Self::Cmyk8 => SampleDepth::Eight,
            Self::Gray16 | Self::Rgb16 | Self::Rgba16 | Self::Cmyk16 => SampleDepth::Sixteen,
        }
    }
}

/// The samples of a [`RenderedPage`], of its format's depth.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pixels {
    /// 8-bit samples.
    U8(Vec<u8>),
    /// 16-bit samples, native-endian in memory.
    U16(Vec<u16>),
}

impl Pixels {
    /// The 8-bit samples; `None` for 16-bit ones.
    #[must_use]
    pub fn as_u8(&self) -> Option<&[u8]> {
        match self {
            Self::U8(v) => Some(v),
            Self::U16(_) => None,
        }
    }

    /// The 16-bit samples; `None` for 8-bit ones.
    #[must_use]
    pub fn as_u16(&self) -> Option<&[u16]> {
        match self {
            Self::U16(v) => Some(v),
            Self::U8(_) => None,
        }
    }

    /// The number of samples.
    #[must_use]
    pub const fn len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::U16(v) => v.len(),
        }
    }

    /// Whether there are no samples.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The bytes the samples occupy.
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::U16(v) => v.len() * 2,
        }
    }
}

/// A single rendered page: 8-bit grayscale pixels, RGBA pixels when
/// rendered over a transparent backdrop, or CMYK pixels under
/// [`ColorModel::Cmyk`].
#[derive(Debug)]
#[non_exhaustive]
pub struct RenderedPage {
    /// Page number (1-based).
    pub page_num: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// The samples in [`format`](Self::format), tightly packed
    /// (`width * format.samples_per_pixel()` a row), top-to-bottom,
    /// left-to-right: [`Pixels::U8`] for an 8-bit format, [`Pixels::U16`]
    /// for a 16-bit one.  Holds `width * height * format.samples_per_pixel()`
    /// samples: `width * height` for the default gray pages.
    pub pixels: Pixels,
    /// The layout of [`pixels`](Self::pixels): [`PixelFormat::Gray8`], or
    /// [`PixelFormat::Rgba8`] under [`RasterOptions::transparent_background`],
    /// or [`PixelFormat::Cmyk8`] under [`ColorModel::Cmyk`];
    /// [`PixelFormat::Gray16`], [`PixelFormat::Rgba16`] and
    /// [`PixelFormat::Cmyk16`] at [`RasterOptions::depth`] 16.
    pub format: PixelFormat,
    /// Physical resolution of the bitmap in pixels per inch (`opts.dpi`).
    ///
    /// The page's size includes its `/UserUnit` (ISO 32000-2 §14.11.2), so a
    /// render at `opts.dpi` has `opts.dpi` pixels per physical inch whatever
    /// the unit.  Pass this value to `tesseract::set_source_resolution`:
    /// Tesseract scales its features by it, and without the call it assumes
    /// 70 DPI, which severely degrades recognition.
    pub dpi: f32,
    /// Lightweight metadata collected at zero extra cost during rendering.
    ///
    /// Use this to route pages to different OCR configurations — e.g. skip deskew
    /// on `has_vector_text = true` pages, or send only the pages without
    /// [`PageDiagnostics::has_visible_text`] to OCR.
    pub diagnostics: PageDiagnostics,
    /// The skew [`RasterOptions::deskew`] measured and, when at least
    /// [`deskew::MIN_CORRECT_DEG`], corrected.  `None` when deskew is off or
    /// the page has no confident angle, which leaves the page as rendered.
    pub skew: Option<deskew::Skew>,
}

// ── Top-level render entry points ─────────────────────────────────────────────

/// Render a range of pages from a PDF file.
///
/// Returns an iterator yielding `(page_num, Result<RenderedPage, RasterError>)`
/// for each page in `opts.first_page..=opts.last_page`.  A per-page error does
/// not abort remaining pages — the caller decides whether to skip or propagate.
///
/// GPU resources are initialised lazily on first use and reused across pages.
/// Backend selection follows [`SessionConfig::default`]: the policy
/// `PDF_RASTER_BACKEND` names, else [`BackendPolicy::Auto`], which renders on
/// the CPU unless a GPU kernel is enabled by its threshold.  Use
/// [`open_session`] + [`render_page_rgb`] directly when you need explicit
/// [`SessionConfig`] control.
///
/// # Errors
///
/// - [`RasterError::InvalidOptions`] — `opts` violates documented constraints.
/// - [`RasterError::Pdf`] — document cannot be opened or parsed.
/// - [`RasterError::PageOutOfRange`] — requested page exceeds the document.
/// - [`RasterError::PageDegenerate`] / [`RasterError::PageTooLarge`] /
///   [`RasterError::PageExceedsCeiling`] — malformed geometry, or a page
///   beyond the byte budget or the side ceiling at the requested scale.
/// - [`RasterError::PageAllocationFailed`] — the allocator refused a bitmap
///   the page's render needs, within the limits: its page bitmap, or a
///   transparency group, soft mask, knockout coverage, tiling-pattern or
///   Type 3 glyph bitmap drawn on it, or a snapshot of the canvas.  Under
///   Linux's default overcommit a large allocation is seldom refused; the
///   process can instead be stopped by the out-of-memory killer when the
///   memory is touched.
/// - [`RasterError::Deskew`] — deskew was asked of a page larger than
///   [`deskew::DESKEW_MAX_SIDE`]; a failed GPU rotation under a forced CUDA
///   backend is [`RasterError::BackendUnavailable`].
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
    batch::render_pages(path, opts)
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
/// PDF.  An encrypted document is decrypted in process under the same
/// options as [`raster_pdf`].
pub fn raster_pdf_from_bytes(
    bytes: Vec<u8>,
    opts: &RasterOptions,
) -> impl Iterator<Item = (u32, Result<RenderedPage, RasterError>)> {
    batch::render_pages_from_bytes(bytes, opts)
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
    batch::render_channel(path, opts, capacity)
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
            last_page: 5,
            ..RasterOptions::default()
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
            last_page: 5,
            pages: Some(ps),
            ..RasterOptions::default()
        };
        assert!(opts.pages.is_some());
    }
}
