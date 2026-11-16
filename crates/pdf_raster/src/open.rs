//! Opening a document as a [`RasterSession`]: backend initialisation, the
//! device image cache and the per-document state shared by every page.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

#[cfg(feature = "vulkan")]
use crate::vk_init::init_vk_backend;
use crate::{BackendPolicy, FormType, RasterError, SessionConfig, UnsupportedFeature};

/// An opened PDF document ready for per-page rendering.
///
/// Constructed via [`open_session`].  Provides both a sequential iterator
/// ([`raster_pdf`](crate::raster_pdf)) and a direct per-page call
/// ([`render_page_rgb`](crate::render_page_rgb)) for parallel consumers such as the CLI.
///
/// `Sync` because the document is read-only after construction, the GPU context
/// is `Arc`-wrapped, and the VA-API decode queue is `Arc`-wrapped (its inner
/// `mpsc::Sender` is `Send + Sync`).
#[expect(
    clippy::struct_excessive_bools,
    reason = "each bool is an independent session switch copied from SessionConfig"
)]
pub struct RasterSession {
    pub(crate) doc: Arc<pdf::Document>,
    /// Fonts of `doc` resolved once and shared by every page rendered.
    pub(crate) fonts: Arc<pdf_interp::renderer::DocumentFonts>,
    /// Image `XObjects` decoded once for every page of `doc` that draws
    /// them, up to [`crate::SessionConfig::image_budget`].
    pub(crate) images: Arc<pdf_interp::renderer::DocumentImages>,
    pub(crate) total_pages: u32,
    pub(crate) policy: BackendPolicy,
    /// The byte budget of a page or region bitmap
    /// ([`crate::SessionConfig::max_page_bytes`]).
    pub(crate) max_page_bytes: u64,
    /// The bytes of a band ([`crate::SessionConfig::band_bytes`]).
    pub(crate) band_bytes: u64,
    /// Fail a page on an image failure instead of skipping the image
    /// ([`crate::SessionConfig::strict`]).
    pub(crate) strict: bool,
    /// The document's optional-content configuration for
    /// [`crate::SessionConfig::purpose`], shared by every page rendered.
    pub(crate) optional_content: Arc<pdf_interp::OptionalContent>,
    /// Draw annotations over the page content
    /// ([`crate::SessionConfig::annotations`]).
    pub(crate) annotations: bool,
    /// Antialias glyphs ([`crate::SessionConfig::text_antialias`]).
    pub(crate) text_antialias: bool,
    /// Antialias paths ([`crate::SessionConfig::vector_antialias`]).
    pub(crate) vector_antialias: bool,
    /// Simulate overprint ([`crate::SessionConfig::overprint`]).
    pub(crate) overprint: bool,
    /// The page boundary box every page is measured and rendered on
    /// ([`crate::SessionConfig::page_box`]).
    pub(crate) page_box: pdf_interp::PageBox,
    /// The rotation added to every page's `/Rotate`
    /// ([`crate::SessionConfig::rotation`]).
    pub(crate) rotation: pdf_interp::Rotation,
    /// The document's interactive form type, reported in every page's
    /// [`PageDiagnostics::form_type`](crate::PageDiagnostics::form_type).
    pub(crate) form_type: pdf_interp::FormType,
    /// Set once a rendered page has been found to carry JavaScript, so the
    /// warning is emitted once per document.
    pub(crate) javascript_warned: AtomicBool,
    /// Set once a rendered page has been found to carry a 3D or multimedia
    /// annotation, so the warning is emitted once per document.
    pub(crate) media_warned: AtomicBool,
    /// The output profile ([`crate::SessionConfig::output_profile`]).
    pub(crate) output_profile: Option<Arc<pdf_interp::OutputProfile>>,
    /// The soft proof of the pages ([`crate::SessionConfig::proof`]).
    pub(crate) proof: Option<crate::proof::SessionProof>,
    /// Whether a page of mixed rendering intents has been reported.
    pub(crate) mixed_intents_warned: AtomicBool,
    /// The working space asked for ([`crate::SessionConfig::working_space`])
    /// and the departures from it reported so far.
    pub(crate) working_space: crate::output::SessionSpace,
    /// The document-level features the renderer ignores.
    pub(crate) unsupported_features: Vec<pdf_interp::UnsupportedFeature>,
    #[cfg(any(feature = "gpu-icc", feature = "cache"))]
    pub(crate) gpu_ctx: Option<Arc<gpu::GpuCtx>>,
    /// Vulkan compute backend.  `Some` when the `vulkan` feature is
    /// enabled, the policy is `ForceVulkan`, and the backend was
    /// constructed.  Mutually exclusive with `gpu_ctx`: `init_gpu_ctx`
    /// returns `None` under `ForceVulkan`.  Shared across all pages.
    #[cfg(feature = "vulkan")]
    pub(crate) vk_backend: Option<Arc<gpu::backend::vulkan::VulkanBackend>>,
    /// The GPU page queue ([`crate::SessionConfig::gpu_pages`]); `None`
    /// renders every page on the CPU.
    #[cfg(feature = "vulkan")]
    pub(crate) device_pages: Option<crate::device_pages::DevicePages>,
    /// Single-threaded VA-API JPEG decode queue.  One worker thread owns the
    /// `VapiJpegDecoder`; all Rayon page-render threads share handles to it.
    /// `None` when the `vaapi` feature is disabled, policy is `CpuOnly` /
    /// `ForceCuda` / `ForceVulkan`, or VA-API initialisation failed (soft
    /// failure on `Auto`).
    #[cfg(feature = "vaapi")]
    pub(crate) vaapi_queue: Option<Arc<gpu::DecodeQueue<gpu::vaapi::VapiJpegDecoder>>>,
    /// Device-resident image cache, on the session's GPU backend.  `Some`
    /// when the `cache` feature is enabled and the session resolved to a
    /// Vulkan backend or a CUDA context (see [`new_image_cache`]); `None`
    /// otherwise (CPU-only fallback).  Shared across all pages so
    /// content-hash dedup spans the whole render.
    #[cfg(feature = "cache")]
    pub(crate) image_cache: Option<gpu::cache::ImageCache>,
    /// Stable document identifier used as the cache's secondary alias
    /// key: the BLAKE3 hash of the document's bytes.
    #[cfg(feature = "cache")]
    pub(crate) doc_id: gpu::cache::DocId,
    /// Image-cache prefetcher handle — kept alive for the session's
    /// lifetime so the discovery + worker threads can drain.
    /// `None` when the cache is disabled or the user opted out via
    /// [`crate::SessionConfig::prefetch`].  Drop semantics: cancels
    /// in-flight prefetch and joins workers; safe to drop mid-render.
    #[cfg(feature = "cache")]
    #[expect(
        dead_code,
        reason = "held for Drop side-effect (cancels + joins prefetcher); \
                  callers don't read it but session lifetime owns it"
    )]
    pub(crate) prefetch: Option<pdf_interp::cache::PrefetchHandle>,
}

impl RasterSession {
    /// Total number of pages in the document.
    #[must_use]
    pub const fn total_pages(&self) -> u32 {
        self.total_pages
    }

    /// The geometry of page `page_num` (1-based): its size in points after
    /// `/Rotate`, [`SessionConfig::rotation`] and `UserUnit`, the rotation
    /// (the page's `/Rotate` plus the session's), `UserUnit`, the origin of
    /// the box selected by [`SessionConfig::page_box`], and the `MediaBox`
    /// and `CropBox`.  No content is parsed.
    ///
    /// [`PageGeometry::size_px`](crate::PageGeometry::size_px) gives the
    /// pixel size a render at given scales produces.
    ///
    /// # Errors
    /// [`RasterError::PageOutOfRange`] when `page_num` is `0` or exceeds the
    /// page count; [`RasterError::Pdf`] when the page tree cannot be walked;
    /// [`RasterError::InvalidPageGeometry`] when `UserUnit` is out of range.
    pub fn page_geometry(&self, page_num: u32) -> Result<crate::PageGeometry, RasterError> {
        let page_id = self.resolve_page(page_num)?;
        self.geometry_by_id(page_id)
    }

    /// The geometry of the page object `page_id` as this session renders
    /// it: measured on [`SessionConfig::page_box`] and turned by
    /// [`SessionConfig::rotation`].
    pub(crate) fn geometry_by_id(
        &self,
        page_id: pdf::ObjectId,
    ) -> Result<crate::PageGeometry, RasterError> {
        Ok(
            pdf_interp::page_size_pts_by_id(&self.doc, page_id, self.page_box)?
                .rotated(self.rotation),
        )
    }

    /// The pixel size `(width, height)` of page `page_num` (1-based)
    /// rendered whole at `dpi`: the size of the page every render entry
    /// point produces at that resolution, rounded as the renderer rounds
    /// it.  Deskew keeps the size.  No content is parsed.
    ///
    /// # Errors
    /// The errors of [`Self::page_geometry`];
    /// [`RasterError::InvalidOptions`] when `dpi` is not a positive finite
    /// number; [`RasterError::PageDegenerate`],
    /// [`RasterError::PageTooLarge`] or [`RasterError::PageExceedsCeiling`]
    /// when a render at `dpi` would fail with them.
    pub fn page_size_px(&self, page_num: u32, dpi: f32) -> Result<(u32, u32), RasterError> {
        let scale = f64::from(dpi) / 72.0;
        crate::render::check_scale(scale)?;
        let geom = self.page_geometry(page_num)?;
        let view = crate::render::page_view(
            &geom,
            (scale, scale),
            None,
            crate::CanvasLayout::new(color::SampleDepth::Eight, false).bytes_per_px(),
            self.max_page_bytes,
        )?;
        Ok((view.width, view.height))
    }

    /// Borrow the underlying [`pdf::Document`] for read-only operations.
    ///
    /// Prefer [`prescan_session`] over calling this directly — it avoids
    /// exposing `pdf::Document` in your call graph.
    #[doc(hidden)]
    #[must_use]
    pub fn doc(&self) -> &pdf::Document {
        &self.doc
    }

    /// The document's interactive form type.
    #[must_use]
    pub const fn form_type(&self) -> FormType {
        self.form_type
    }

    /// The document-level features the renderer ignores: JavaScript
    /// attached to the catalog, the name tree or the interactive form, an
    /// XFA form and a portable collection, in that order.  The features of
    /// one page are in
    /// [`PageDiagnostics::unsupported_features`](crate::PageDiagnostics::unsupported_features).
    /// Empty when the document uses none.
    #[must_use]
    pub fn unsupported_features(&self) -> &[UnsupportedFeature] {
        &self.unsupported_features
    }

    /// The profile every page is converted to
    /// ([`SessionConfig::output_profile`]).
    #[must_use]
    pub const fn output_profile(&self) -> Option<&Arc<pdf_interp::OutputProfile>> {
        self.output_profile.as_ref()
    }

    /// The backend policy this session was opened with.
    #[must_use]
    pub const fn policy(&self) -> BackendPolicy {
        self.policy
    }

    /// The device-resident image cache this session renders through, on
    /// its Vulkan backend or CUDA context, or `None` when neither was
    /// initialised (CPU-only session).
    ///
    /// Exposed for diagnostics and tests that need to observe which images
    /// took the cached path; the renderer wires the cache in itself.
    #[cfg(feature = "cache")]
    #[must_use]
    pub const fn image_cache(&self) -> Option<&gpu::cache::ImageCache> {
        self.image_cache.as_ref()
    }

    /// Resolve a 1-based page number to its [`pdf::ObjectId`].
    ///
    /// Each call performs one logarithmic page-tree descent.  Per-render
    /// callers should resolve once at the entry point and pass the id into
    /// `pdf_interp::page_size_pts_by_id` and `pdf_interp::parse_page_by_id`
    /// so the single descent serves all three uses; an earlier shape with a
    /// session-side `RwLock<HashMap>` cache turned out to be hot-write
    /// cold-read on every contest event (each page rendered once) and was
    /// dropped.
    ///
    /// Prefer [`prescan_session`] and [`render_page_rgb`](crate::render_page_rgb) over calling this
    /// directly — they avoid exposing `pdf::ObjectId` in your call graph.
    ///
    /// # Errors
    /// [`RasterError::PageOutOfRange`] when `page_num` is `0` or exceeds
    /// `self.total_pages`; [`RasterError::Pdf`] when the underlying page-tree
    /// descent fails (malformed `/Pages` node).
    #[doc(hidden)]
    pub fn resolve_page(&self, page_num: u32) -> Result<pdf::ObjectId, RasterError> {
        // The upper-bound check is structurally duplicated by
        // `Document::get_page`, but doing it here keeps the user-facing
        // error 1-based (the descent returns 0-based `PageOutOfRange`).
        // The `page_num == 0` check is load-bearing: guards `page_num - 1`
        // from u32 underflow.
        if page_num == 0 || page_num > self.total_pages {
            return Err(RasterError::PageOutOfRange {
                page: page_num,
                total: self.total_pages,
            });
        }
        self.doc.get_page(page_num - 1).map_err(RasterError::from)
    }
}

/// Classify page `page_num` (1-based) without rendering any pixels.
///
/// Wraps [`pdf_interp::prescan_page`] using the document already loaded in
/// `session` — callers do not need to access `session.doc()` directly.
///
/// Returns a [`crate::PageDiagnostics`] with `has_images`, `has_vector_text`,
/// `dominant_filter`, and a conservative `source_ppi_hint`.
///
/// # Errors
///
/// Returns a [`RasterError`] if `page_num` is outside the document or the
/// page geometry is invalid.
pub fn prescan_session(
    session: &RasterSession,
    page_num: u32,
) -> Result<crate::PageDiagnostics, RasterError> {
    pdf_interp::prescan_page(&session.doc, page_num, session.page_box).map_err(RasterError::from)
}

// Compile-time assertions: RasterSession must be Sync (shared across rayon threads) and
// Send (moved into the rayon::spawn closure in render_channel).
const _: fn() = || {
    const fn assert_sync<T: Sync>() {}
    const fn assert_send<T: Send>() {}
    assert_sync::<RasterSession>();
    assert_send::<RasterSession>();
};

/// Open a PDF and create a [`RasterSession`] for rendering.
///
/// Reads `/Pages /Count` directly (O(1) on well-formed PDFs) and defers
/// per-page id resolution until the first render of each page — opening
/// a 100 000-page document and rendering one page no longer pays for a
/// full page-tree walk.  GPU context (AA/ICC) is initialised here; JPEG
/// decoders are initialised lazily per rayon worker thread on first page
/// render.
///
/// An encrypted document opens only with `config.decrypt_authorized`; its
/// objects are then decrypted in process, with `config.password` when the
/// empty user password does not unlock it.
///
/// A file of at least 256 KiB is memory-mapped unless
/// `config.read_into_memory` is set: truncating a mapped file while the
/// session is alive makes the next read past the new end of file raise
/// `SIGBUS`, which terminates the process (see
/// [`SessionConfig::read_into_memory`]).
///
/// # Errors
///
/// - [`RasterError::Pdf`] if the file cannot be opened or parsed, or is
///   encrypted and decryption is not authorised or the password is wrong;
///   with `config.read_into_memory`, also when the file's bytes cannot be
///   allocated.
/// - [`RasterError::BackendUnavailable`] if `config.policy` is
///   `ForceCuda`, `ForceVaapi` or `ForceVulkan` and the required backend
///   was not compiled in or fails to initialise, or if `config.gpu_pages`
///   is set and the thread that opens its device cannot start.
/// - [`RasterError::InvalidOptions`] if `config.gpu_pages` cannot be
///   honoured (see [`SessionConfig::check_gpu_pages`]), or if
///   `config.vulkan_device` matches no usable Vulkan device or several.
pub fn open_session(
    path: &std::path::Path,
    config: &SessionConfig,
) -> Result<RasterSession, RasterError> {
    let doc = pdf_interp::open_with(path, &config.open_options()).map_err(RasterError::from)?;
    open_session_with_doc(doc, config)
}

/// Open a PDF held entirely in memory (e.g. extracted from an archive) and
/// create a [`RasterSession`] for rendering.
///
/// Same behaviour, backend selection and decryption as [`open_session`];
/// only the document source differs — no file path is touched.
///
/// # Errors
///
/// - [`RasterError::Pdf`] if `bytes` is not a parseable PDF, or is an
///   encrypted document whose decryption is not authorised or whose
///   password is wrong.
/// - [`RasterError::BackendUnavailable`] if `config.policy` is
///   `ForceCuda`, `ForceVaapi` or `ForceVulkan` and the required backend
///   was not compiled in or fails to initialise, or if `config.gpu_pages`
///   is set and the thread that opens its device cannot start.
/// - [`RasterError::InvalidOptions`] if `config.gpu_pages` cannot be
///   honoured (see [`SessionConfig::check_gpu_pages`]), or if
///   `config.vulkan_device` matches no usable Vulkan device or several.
pub fn open_session_from_bytes(
    bytes: Vec<u8>,
    config: &SessionConfig,
) -> Result<RasterSession, RasterError> {
    let doc =
        pdf_interp::open_bytes_with(bytes, &config.open_options()).map_err(RasterError::from)?;
    open_session_with_doc(doc, config)
}

/// [`open_session_from_bytes`] for bytes the caller keeps shared.
///
/// `bytes` is any `AsRef<[u8]>` storage: an `Arc<[u8]>`, a `&'static [u8]`,
/// a reference-counted byte buffer type, or a mapping the caller made.  The
/// bytes are read in place, never copied, and held by the session until it
/// is dropped.
///
/// `bytes.as_ref()` must return the same bytes on every call: the parser
/// reads the cross-reference table once at open and indexes the bytes by
/// its offsets afterwards.
///
/// # Errors
///
/// As [`open_session_from_bytes`].
pub fn open_session_from_shared(
    bytes: impl AsRef<[u8]> + Send + Sync + 'static,
    config: &SessionConfig,
) -> Result<RasterSession, RasterError> {
    let doc =
        pdf_interp::open_shared_with(bytes, &config.open_options()).map_err(RasterError::from)?;
    open_session_with_doc(doc, config)
}

/// [`open_session_from_bytes`] for a document read from `reader`, such as
/// a network stream or a member of an archive.
///
/// The parser needs random access to the whole document as one contiguous
/// slice, so `reader` is read to its end into memory first; nothing is
/// parsed before then, and a `Seek` source gains nothing over a plain
/// [`Read`](std::io::Read) one.  A file on disk opens without the copy
/// through [`open_session`].
///
/// The read is unbounded: the session holds the whole document, however
/// long the source runs.  Bound an untrusted reader with
/// [`Read::take`](std::io::Read::take); a source longer than the bound is
/// then opened as a truncated document.
///
/// # Errors
///
/// [`RasterError::Pdf`] wrapping an I/O error when `reader` fails;
/// otherwise as [`open_session_from_bytes`].
pub fn open_session_from_reader(
    mut reader: impl std::io::Read,
    config: &SessionConfig,
) -> Result<RasterSession, RasterError> {
    let mut bytes = Vec::new();
    let _ = reader
        .read_to_end(&mut bytes)
        .map_err(|e| RasterError::from(pdf::PdfError::from(e)))?;
    open_session_from_bytes(bytes, config)
}

/// Why `policy` cannot run in this build: a `Force*` policy whose backend
/// was not compiled in (`ForceVulkan` without the `vulkan` feature,
/// `ForceCuda` without any CUDA feature).  Without this gate the backend's
/// cfg-gated initialisation disappears and the forced render silently runs
/// on the CPU, the fallback the `Force*` variants exist to prevent.
const fn uncompiled_backend(policy: BackendPolicy) -> Option<&'static str> {
    let vulkan = cfg!(feature = "vulkan");
    let cuda = cfg!(any(feature = "gpu-icc", feature = "cache"));
    match policy {
        BackendPolicy::ForceVulkan if !vulkan => Some(
            "ForceVulkan requires the `vulkan` Cargo feature; \
             rebuild with `--features vulkan` or pick another --backend.",
        ),
        BackendPolicy::ForceCuda if !cuda => Some(
            "ForceCuda requires the `gpu-icc` or `cache` Cargo feature; \
             rebuild with the desired feature set or pick another --backend.",
        ),
        _ => None,
    }
}

/// Shared session assembly from an already-opened document — the backend init,
/// image cache, doc id, prefetcher, and `RasterSession` construction common to
/// both the path and in-memory entry points.
fn open_session_with_doc(
    doc: pdf::Document,
    config: &SessionConfig,
) -> Result<RasterSession, RasterError> {
    let doc = Arc::new(doc);
    let total_pages = doc.page_count_fast();

    if let Some(reason) = uncompiled_backend(config.policy) {
        return Err(RasterError::BackendUnavailable(reason.to_owned()));
    }
    config.check()?;
    // The Vulkan backend initialises up front so that failures surface
    // here rather than mid-render.
    #[cfg(feature = "vulkan")]
    let vk_backend = init_vk_backend(config.policy, config.vulkan_device.as_deref())?;
    #[cfg(feature = "vulkan")]
    let device_pages = crate::device_pages::DevicePages::for_session(config, vk_backend.as_ref())?;

    #[cfg(any(feature = "gpu-icc", feature = "cache"))]
    let gpu_ctx = init_gpu_ctx(config.policy)?;

    #[cfg(feature = "vaapi")]
    let vaapi_queue = crate::decode_queue::build_vaapi_queue(&config.vaapi_device, config.policy)
        .map_err(RasterError::BackendUnavailable)?
        .map(Arc::new);

    #[cfg(feature = "cache")]
    let image_cache = new_image_cache(
        gpu_ctx.as_deref(),
        #[cfg(feature = "vulkan")]
        vk_backend.as_ref(),
    );

    // DocId: BLAKE3 of the PDF bytes.  Stable per content; editing
    // the PDF naturally invalidates the disk tier because the hash
    // changes.  Costs one full BLAKE3 hash at session open
    // (~250 MB/s; ~40ms for a 10MB PDF) — paid once per session, not
    // per page.  Borrows the bytes already mmapped by `pdf_interp::open`
    // so the hash adds zero IO.
    #[cfg(feature = "cache")]
    let doc_id = {
        let hash = gpu::cache::ContentHash::of(doc.bytes());
        gpu::cache::DocId(hash.0)
    };

    // Spawn the prefetcher last (after the cache + doc_id are
    // resolved).  Skipped when the user didn't opt in or when no
    // cache exists to prefetch into.
    #[cfg(feature = "cache")]
    let prefetch = if config.prefetch
        && let Some(cache) = image_cache.as_ref()
    {
        let mut prefetch_config = pdf_interp::cache::PrefetchConfig::default();
        prefetch_config.scale = config.prefetch_scale;
        prefetch_config.rotation = config.rotation;
        pdf_interp::cache::spawn_prefetch(Arc::clone(&doc), cache.clone(), doc_id, prefetch_config)
            .inspect_err(|e| {
                log::warn!("image prefetch disabled: cannot start its threads ({e})");
            })
            .ok()
    } else {
        None
    };

    let proof = config.proof_session(&doc)?;
    let optional_content = Arc::new(pdf_interp::OptionalContent::new(&doc, config.purpose));
    let form_type = pdf_interp::form_type(&doc);
    let unsupported_features = pdf_interp::document_unsupported_features(&doc);
    Ok(RasterSession {
        doc,
        fonts: Arc::new(pdf_interp::renderer::DocumentFonts::new(
            &config.font_dirs,
            config.system_fonts,
        )),
        images: Arc::new(pdf_interp::renderer::DocumentImages::new(
            config.image_budget,
        )),
        total_pages,
        policy: config.policy,
        max_page_bytes: config.max_page_bytes,
        band_bytes: config.band_bytes,
        strict: config.strict,
        optional_content,
        annotations: config.annotations,
        text_antialias: config.text_antialias,
        vector_antialias: config.vector_antialias,
        overprint: config.overprint,
        page_box: config.page_box,
        rotation: config.rotation,
        form_type,
        javascript_warned: AtomicBool::new(false),
        media_warned: AtomicBool::new(false),
        output_profile: config.output_profile.clone(),
        proof,
        mixed_intents_warned: AtomicBool::new(false),
        working_space: crate::output::SessionSpace::new(config.working_space),
        unsupported_features,
        #[cfg(any(feature = "gpu-icc", feature = "cache"))]
        gpu_ctx,
        #[cfg(feature = "vulkan")]
        vk_backend,
        #[cfg(feature = "vulkan")]
        device_pages,
        #[cfg(feature = "vaapi")]
        vaapi_queue,
        #[cfg(feature = "cache")]
        image_cache,
        #[cfg(feature = "cache")]
        doc_id,
        #[cfg(feature = "cache")]
        prefetch,
    })
}

/// Build a session's device image cache on the backend the session
/// resolved to: the Vulkan backend when it holds one, otherwise the CUDA
/// context, otherwise none.  The two are not initialised together (see
/// `open_session_with_doc`), so the cache follows the backend every other
/// kernel of the session runs on.
///
/// The disk tier is attached when `PDF_RASTER_CACHE_DIR` names its root
/// ([`gpu::cache::DiskTier::try_new`]); otherwise the cache is in-process
/// only.  The host tier's budget is the default.
#[cfg(feature = "cache")]
fn new_image_cache(
    ctx: Option<&gpu::GpuCtx>,
    #[cfg(feature = "vulkan")] vk_backend: Option<&Arc<gpu::backend::vulkan::VulkanBackend>>,
) -> Option<gpu::cache::ImageCache> {
    #[cfg(feature = "vulkan")]
    if let Some(backend) = vk_backend {
        return Some(gpu::cache::ImageCache::Vulkan(Arc::new(with_disk_tier(
            new_vulkan_image_cache(backend),
        ))));
    }
    let ctx = ctx?;
    Some(gpu::cache::ImageCache::Cuda(Arc::new(with_disk_tier(
        new_cuda_image_cache(ctx),
    ))))
}

/// `cache` with the disk tier attached when its root resolves.
#[cfg(feature = "cache")]
fn with_disk_tier<B: gpu::cache::CacheBackend>(
    cache: gpu::cache::DeviceImageCache<B>,
) -> gpu::cache::DeviceImageCache<B> {
    match gpu::cache::DiskTier::try_new() {
        Some(disk) => cache.with_disk(disk),
        None => cache,
    }
}

/// A device image cache on `ctx`, under the default VRAM budget.
///
/// The cache owns a stream of its own, so uploads issued by any page
/// worker queue independently of that worker's stream; each blit waits
/// on the write event of the upload it reads.
#[cfg(feature = "cache")]
fn new_cuda_image_cache(ctx: &gpu::GpuCtx) -> gpu::cache::DeviceImageCache {
    let stream = ctx.new_stream().unwrap_or_else(|e| {
        log::warn!(
            "rasterrocket: image-cache stream creation failed ({e}); sharing this thread's stream"
        );
        ctx.stream()
    });
    gpu::cache::DeviceImageCache::new(
        stream,
        gpu::cache::VramBudget::DEFAULT,
        gpu::cache::HostBudget::DEFAULT,
    )
}

/// A device image cache on `backend`, under the VRAM budget the device
/// reports ([`gpu::cache::VramBudget::from_backend`]), or the default one
/// when the query fails.
#[cfg(all(feature = "cache", feature = "vulkan"))]
fn new_vulkan_image_cache(
    backend: &Arc<gpu::backend::vulkan::VulkanBackend>,
) -> gpu::cache::DeviceImageCache<gpu::backend::vulkan::VulkanBackend> {
    let vram = gpu::cache::VramBudget::from_backend(&**backend).unwrap_or_else(|e| {
        log::warn!("rasterrocket: Vulkan VRAM budget query failed ({e}); using the default cap");
        gpu::cache::VramBudget::DEFAULT
    });
    gpu::cache::DeviceImageCache::new(Arc::clone(backend), vram, gpu::cache::HostBudget::DEFAULT)
}

/// Initialise the CUDA GPU context for ICC colour transforms and image blits.
///
/// Returns `None` on `Auto`, `CpuOnly` and `ForceVulkan`.  Errors loudly on
/// `ForceCuda` if init fails; logs a warning and returns `None` on
/// `ForceVaapi` if init fails.
///
/// The CUDA context, stream, and 7 PTX modules cost ~240 ms warm /
/// ~1100 ms cold to build, and are process-wide state — there is no
/// per-session work hidden inside.  We therefore cache the init result
/// in a process-wide `OnceLock` so workloads that open many short-lived
/// sessions (e.g. one page per archive across 100 archives) pay the
/// cost once instead of once per `open_session` call.
///
/// Failures are also cached: on `ForceVaapi` we retain the
/// fallback `None` so we don't re-attempt CUDA init every session and
/// log the same warning hundreds of times.  `ForceCuda` still surfaces
/// the cached error message verbatim because the caller asked us to
/// fail loud rather than fall back.
#[cfg(any(feature = "gpu-icc", feature = "cache"))]
static GPU_CTX: std::sync::OnceLock<Result<Arc<gpu::GpuCtx>, String>> = std::sync::OnceLock::new();

#[cfg(any(feature = "gpu-icc", feature = "cache"))]
fn init_gpu_ctx(policy: BackendPolicy) -> Result<Option<Arc<gpu::GpuCtx>>, RasterError> {
    if matches!(
        policy,
        BackendPolicy::Auto | BackendPolicy::CpuOnly | BackendPolicy::ForceVulkan
    ) {
        return Ok(None);
    }

    let cached = GPU_CTX.get_or_init(|| match gpu::GpuCtx::init() {
        Ok(ctx) => Ok(Arc::new(ctx)),
        Err(e) => Err(e.to_string()),
    });

    match cached {
        Ok(ctx) => Ok(Some(Arc::clone(ctx))),
        Err(e) => {
            if matches!(policy, BackendPolicy::ForceCuda) {
                Err(RasterError::BackendUnavailable(format!(
                    "CUDA GPU context required but unavailable: {e}. \
                     Verify with `nvidia-smi` that the driver is loaded."
                )))
            } else {
                log::warn!(
                    "rasterrocket: GPU initialisation failed ({e}); \
                     falling back to CPU. Run `nvidia-smi` to verify the driver is loaded."
                );
                Ok(None)
            }
        }
    }
}
