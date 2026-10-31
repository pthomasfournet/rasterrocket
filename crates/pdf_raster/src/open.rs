//! Opening a document as a [`RasterSession`]: backend initialisation, the
//! device image cache and the per-document state shared by every page.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

#[cfg(feature = "vulkan")]
use crate::vk_init::init_vk_backend;
use crate::{BackendPolicy, FormType, RasterError, SessionConfig};

/// An opened PDF document ready for per-page rendering.
///
/// Constructed via [`open_session`].  Provides both a sequential iterator
/// ([`raster_pdf`](crate::raster_pdf)) and a direct per-page call
/// ([`render_page_rgb`]) for parallel consumers such as the CLI.
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
    pub(crate) total_pages: u32,
    pub(crate) policy: BackendPolicy,
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
    /// The document's interactive form type, reported in every page's
    /// [`PageDiagnostics::form_type`](crate::PageDiagnostics::form_type).
    pub(crate) form_type: pdf_interp::FormType,
    /// Set once a rendered page has been found to carry JavaScript, so the
    /// warning is emitted once per document.
    pub(crate) javascript_warned: AtomicBool,
    #[cfg(any(feature = "gpu-aa", feature = "gpu-icc", feature = "cache"))]
    pub(crate) gpu_ctx: Option<Arc<gpu::GpuCtx>>,
    /// Vulkan compute backend.  `Some` when the `vulkan` feature is
    /// enabled, the policy resolves to `Auto` or `ForceVulkan`, and
    /// `VulkanBackend::new` succeeded.  Mutually exclusive with
    /// `gpu_ctx` in practice — under `Auto` the caller skips CUDA
    /// init when this is `Some`; under `ForceVulkan` `init_gpu_ctx`
    /// returns `None` regardless.  Shared across all pages.
    #[cfg(feature = "vulkan")]
    pub(crate) vk_backend: Option<Arc<gpu::backend::vulkan::VulkanBackend>>,
    /// [`crate::SessionConfig::vulkan_device`], for the per-thread Vulkan
    /// JPEG decoders, which open their own logical device.
    #[cfg(all(feature = "vulkan", feature = "gpu-jpeg-huffman"))]
    pub(crate) vulkan_device: Option<String>,
    /// Single-threaded VA-API JPEG decode queue.  One worker thread owns the
    /// `VapiJpegDecoder`; all Rayon page-render threads share handles to it.
    /// `None` when the `vaapi` feature is disabled, policy is `CpuOnly` /
    /// `ForceCuda` / `ForceVulkan`, or VA-API initialisation failed (soft
    /// failure on `Auto`).
    #[cfg(feature = "vaapi")]
    pub(crate) vaapi_queue: Option<Arc<gpu::DecodeQueue<gpu::vaapi::VapiJpegDecoder>>>,
    /// Device-resident image cache.  `Some` when the `cache`
    /// feature is enabled and a CUDA device is available; `None`
    /// otherwise (CPU-only fallback).  Shared across all pages so
    /// content-hash dedup spans the whole render.
    #[cfg(feature = "cache")]
    pub(crate) image_cache: Option<Arc<gpu::cache::DeviceImageCache>>,
    /// Stable document identifier used as the cache's secondary alias
    /// key.  Today derived from the file path or a per-session UUID;
    /// any 32-byte content-addressable identifier works.
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
    /// `/Rotate` and `UserUnit`, rotation, `UserUnit`, the origin of the
    /// selected box, and the `MediaBox` and `CropBox`.  No content is parsed.
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
        pdf_interp::page_size_pts_by_id(&self.doc, page_id).map_err(RasterError::from)
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

    /// The backend policy this session was opened with.
    #[must_use]
    pub const fn policy(&self) -> BackendPolicy {
        self.policy
    }

    /// The device-resident image cache this session renders through, or
    /// `None` when no CUDA context was initialised (CPU-only session).
    ///
    /// Exposed for diagnostics and tests that need to observe which images
    /// took the cached path; the renderer wires the cache in itself.
    #[cfg(feature = "cache")]
    #[must_use]
    pub const fn image_cache(&self) -> Option<&Arc<gpu::cache::DeviceImageCache>> {
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
    /// Prefer [`prescan_session`] and [`render_page_rgb`] over calling this
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
    pdf_interp::prescan_page(&session.doc, page_num).map_err(RasterError::from)
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
/// # Errors
///
/// - [`RasterError::Pdf`] if the file cannot be opened or parsed, or is
///   encrypted and decryption is not authorised or the password is wrong.
/// - [`RasterError::BackendUnavailable`] if `config.policy` is `ForceCuda` or
///   `ForceVaapi` and the required GPU context fails to initialise.
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
/// - [`RasterError::BackendUnavailable`] if `config.policy` is `ForceCuda` or
///   `ForceVaapi` and the required GPU context fails to initialise.
pub fn open_session_from_bytes(
    bytes: Vec<u8>,
    config: &SessionConfig,
) -> Result<RasterSession, RasterError> {
    let doc =
        pdf_interp::open_bytes_with(bytes, &config.open_options()).map_err(RasterError::from)?;
    open_session_with_doc(doc, config)
}

/// Why `policy` cannot run in this build: a `Force*` policy whose backend
/// was not compiled in (`ForceVulkan` without the `vulkan` feature,
/// `ForceCuda` without any CUDA feature).  Without this gate the backend's
/// cfg-gated initialisation disappears and the forced render silently runs
/// on the CPU, the fallback the `Force*` variants exist to prevent.
const fn uncompiled_backend(policy: BackendPolicy) -> Option<&'static str> {
    let vulkan = cfg!(feature = "vulkan");
    let cuda = cfg!(any(
        feature = "gpu-aa",
        feature = "gpu-icc",
        feature = "cache"
    ));
    match policy {
        BackendPolicy::ForceVulkan if !vulkan => Some(
            "ForceVulkan requires the `vulkan` Cargo feature; \
             rebuild with `--features vulkan` or pick another --backend.",
        ),
        BackendPolicy::ForceCuda if !cuda => Some(
            "ForceCuda requires at least one of the `gpu-aa`, `gpu-icc`, or \
             `cache` Cargo features; rebuild with the desired feature set or \
             pick another --backend.",
        ),
        _ => None,
    }
}

/// The CUDA context of a session under `policy`.  Under `Auto` a Vulkan
/// backend wins (`vk_won_auto`) and CUDA is not initialised, which saves
/// its ~240 ms start-up for a path the session will not use.
#[cfg(any(feature = "gpu-aa", feature = "gpu-icc", feature = "cache"))]
fn session_gpu_ctx(
    policy: BackendPolicy,
    vk_won_auto: bool,
) -> Result<Option<Arc<gpu::GpuCtx>>, RasterError> {
    let ctx = if vk_won_auto {
        None
    } else {
        init_gpu_ctx(policy)?
    };
    if matches!(policy, BackendPolicy::Auto) {
        log_auto_resolution_once(vk_won_auto, ctx.is_some());
    }
    Ok(ctx)
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
    // The Vulkan backend initialises up front so that failures surface
    // here rather than mid-render.
    #[cfg(feature = "vulkan")]
    let vk_backend = init_vk_backend(config.policy, config.vulkan_device.as_deref())?;
    #[cfg(feature = "vulkan")]
    let vk_won_auto = vk_backend.is_some() && matches!(config.policy, BackendPolicy::Auto);
    #[cfg(not(feature = "vulkan"))]
    let vk_won_auto = false;

    #[cfg(any(feature = "gpu-aa", feature = "gpu-icc", feature = "cache"))]
    let gpu_ctx = session_gpu_ctx(config.policy, vk_won_auto)?;
    #[cfg(not(any(feature = "gpu-aa", feature = "gpu-icc", feature = "cache")))]
    if matches!(config.policy, BackendPolicy::Auto) {
        log_auto_resolution_once(vk_won_auto, false);
    }

    #[cfg(feature = "vaapi")]
    let vaapi_queue = crate::decode_queue::build_vaapi_queue(&config.vaapi_device, config.policy)
        .map_err(RasterError::BackendUnavailable)?
        .map(Arc::new);

    #[cfg(feature = "cache")]
    let image_cache = gpu_ctx.as_deref().map(new_image_cache);

    // DocId: BLAKE3 of the PDF bytes.  Stable per content; editing
    // the PDF naturally invalidates the disk tier because the hash
    // changes.  Costs one full BLAKE3 hash at session open
    // (~250 MB/s; ~40ms for a 10MB PDF) — paid once per session, not
    // per page.  Borrows the bytes already mmapped by `pdf_interp::open`
    // so the hash adds zero IO.
    #[cfg(feature = "cache")]
    let doc_id = {
        let hash = gpu::cache::DeviceImageCache::hash_bytes(doc.bytes());
        gpu::cache::DocId(hash.0)
    };

    // Spawn the prefetcher last (after the cache + doc_id are
    // resolved).  Skipped when the user didn't opt in or when no
    // cache exists to prefetch into.
    #[cfg(feature = "cache")]
    let prefetch = if config.prefetch
        && let Some(cache) = image_cache.as_ref()
    {
        pdf_interp::cache::spawn_prefetch(
            Arc::clone(&doc),
            Arc::clone(cache),
            doc_id,
            pdf_interp::cache::PrefetchConfig::default(),
        )
        .inspect_err(|e| {
            log::warn!("image prefetch disabled: cannot start its threads ({e})");
        })
        .ok()
    } else {
        None
    };

    let optional_content = Arc::new(pdf_interp::OptionalContent::new(&doc, config.purpose));
    let form_type = pdf_interp::form_type(&doc);
    Ok(RasterSession {
        doc,
        fonts: Arc::new(pdf_interp::renderer::DocumentFonts::new(
            &config.font_dirs,
            config.system_fonts,
        )),
        total_pages,
        policy: config.policy,
        strict: config.strict,
        optional_content,
        annotations: config.annotations,
        text_antialias: config.text_antialias,
        vector_antialias: config.vector_antialias,
        form_type,
        javascript_warned: AtomicBool::new(false),
        #[cfg(any(feature = "gpu-aa", feature = "gpu-icc", feature = "cache"))]
        gpu_ctx,
        #[cfg(feature = "vulkan")]
        vk_backend,
        #[cfg(all(feature = "vulkan", feature = "gpu-jpeg-huffman"))]
        vulkan_device: config.vulkan_device.clone(),
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

/// Build a session's device image cache on `ctx`.
///
/// The cache owns a stream of its own, so uploads issued by any page
/// worker queue independently of that worker's stream; each blit waits
/// on the write event of the upload it reads.  Budgets are the defaults.
/// The disk tier is attached when its root resolves (`HOME`,
/// `XDG_CACHE_HOME` or `PDF_RASTER_CACHE_DIR` set); otherwise the cache
/// is in-process only.
#[cfg(feature = "cache")]
fn new_image_cache(ctx: &gpu::GpuCtx) -> Arc<gpu::cache::DeviceImageCache> {
    let stream = ctx.new_stream().unwrap_or_else(|e| {
        log::warn!(
            "rasterrocket: image-cache stream creation failed ({e}); sharing this thread's stream"
        );
        ctx.stream()
    });
    let mut cache = gpu::cache::DeviceImageCache::new(
        stream,
        gpu::cache::VramBudget::DEFAULT,
        gpu::cache::HostBudget::DEFAULT,
    );
    if let Some(disk) = gpu::cache::DiskTier::try_new() {
        cache = cache.with_disk(disk);
    }
    Arc::new(cache)
}

/// Makes [`log_auto_resolution_once`] fire once per process;
/// `open_session` may run once per page (comic archives).
static AUTO_RESOLUTION_LOGGED: std::sync::Once = std::sync::Once::new();

/// Log, once per process at `debug` level, which backend
/// [`BackendPolicy::Auto`] resolved to.
fn log_auto_resolution_once(vulkan_won: bool, cuda_available: bool) {
    AUTO_RESOLUTION_LOGGED.call_once(|| {
        if vulkan_won {
            log::debug!("rasterrocket: Auto backend resolution: Vulkan won, CUDA init skipped");
        } else if cuda_available {
            log::debug!("rasterrocket: Auto backend resolution: CUDA");
        } else {
            log::debug!("rasterrocket: Auto backend resolution: CPU (no GPU backend available)");
        }
    });
}

/// Initialise the CUDA GPU context for AA fill and ICC colour transforms.
///
/// Returns `None` on `CpuOnly` and `ForceVulkan`.  Errors loudly on
/// `ForceCuda` if init fails; logs a warning and returns `None` on
/// `Auto` / `ForceVaapi` if init fails.
///
/// The caller is expected to short-circuit *before* calling this when
/// Vulkan already won under `Auto` (see `open_session`) — keeping that
/// dispatch decision at the call site lets this function stay
/// policy-pure.
///
/// The CUDA context, stream, and 7 PTX modules cost ~240 ms warm /
/// ~1100 ms cold to build, and are process-wide state — there is no
/// per-session work hidden inside.  We therefore cache the init result
/// in a process-wide `OnceLock` so workloads that open many short-lived
/// sessions (e.g. one page per archive across 100 archives) pay the
/// cost once instead of once per `open_session` call.
///
/// Failures are also cached: on `Auto`/`ForceVaapi` we retain the
/// fallback `None` so we don't re-attempt CUDA init every session and
/// log the same warning hundreds of times.  `ForceCuda` still surfaces
/// the cached error message verbatim because the caller asked us to
/// fail loud rather than fall back.
#[cfg(any(feature = "gpu-aa", feature = "gpu-icc", feature = "cache"))]
static GPU_CTX: std::sync::OnceLock<Result<Arc<gpu::GpuCtx>, String>> = std::sync::OnceLock::new();

#[cfg(any(feature = "gpu-aa", feature = "gpu-icc", feature = "cache"))]
fn init_gpu_ctx(policy: BackendPolicy) -> Result<Option<Arc<gpu::GpuCtx>>, RasterError> {
    if matches!(policy, BackendPolicy::CpuOnly | BackendPolicy::ForceVulkan) {
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
