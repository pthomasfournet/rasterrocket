//! Per-page operator dispatcher.
//!
//! [`PageRenderer`] holds a target [`Bitmap`] and a `GStateStack`, iterates
//! over a decoded operator sequence, and calls into the `raster` crate for
//! each painting operator.
//!
//! Sub-modules (all private):
//! - `operators` — the operator dispatch table and marked-content tracking.
//! - `fill_stroke` — the fill-and-stroke operators as one object.
//! - `text`, `text_ops`, `type3` — text showing, its text-space arithmetic
//!   and Type 3 glyph descriptions.
//! - `patterns`, `pattern_memo` — tiling and shading patterns and their
//!   per-page memo.
//! - `shading_paint` — the `sh` operator and shading-pattern fills and strokes.
//! - `soft_mask` — `ExtGState` soft masks and the painting of one object under them.
//! - `knockout` — the painting of one object in a knockout group.
//! - `image_memo`, `image_reduce`, `image_sampler` — image decode memo,
//!   area-averaged reduction and device-pixel sampling.
//! - `pixel_rect` — rectangles of device pixels.
//! - `annotations` — annotation appearances, stored and synthesised.
//! - `gpu_ops`, `vk_ops` — CUDA and Vulkan antialiased fill dispatch.
//!
//! # What is implemented
//!
//! - Graphics state: `q Q cm w J j M d ri i gs`
//! - Path construction: `m l c v y h re`
//! - Path painting: `S s f F f* B B* b b* n`
//! - Clip paths: `W W*` (winding and even-odd; intersected into the current `Clip`)
//! - Colour: `g G rg RG k K sc scn SC SCN cs CS`
//! - Text objects + state: `BT ET Tf Tc Tw Tz TL Ts Tr Td TD Tm T*`
//! - Text showing: `Tj TJ ' "` (glyphs from the font crate)
//! - Font encoding `Differences` array — resolved via Adobe Glyph List
//!
//! - Type 0 / `CIDFont` composite fonts — `Encoding` `CMap` (embedded
//!   stream `CMaps`, `Identity-H`/`V` and the predefined CJK `CMaps`),
//!   `CIDToGIDMap`, `DW`/`W` advance widths, vertical writing mode, and
//!   multi-byte character code iteration
//!
//! - Text render modes 4–7 — fill/stroke painting combined with text-as-clip
//!   (`glyph_path` outlines accumulated into the clip region per PDF §9.3.6)
//! - `PatternType` 1 tiling patterns and `PatternType` 2 shading patterns via `scn`/`SCN`
//! - Type 3 paint-procedure fonts — `CharProc` content streams, `d0`/`d1` metrics
//! - `ExtGState` alpha constants, blend modes, soft masks and transfer
//!   functions
//!
//! - Optional content (layers) — `BDC /OC` sequences and `XObjects` and
//!   annotations carrying `/OC` are skipped when hidden under the default
//!   configuration (see `crate::optional_content`)
//! - Annotation rendering — `AP/N` normal appearance streams drawn after page
//!   content, and synthesised appearances where `/AP` is absent
//! - Transparency groups — Form `XObjects` with `Group /S /Transparency`, isolated
//!   and knockout, and the page group, rendered into an intermediate bitmap via
//!   `raster::transparency` and composited back (PDF §11.6.6)

mod annotations;
mod devices;
mod diagnostics;
mod fill_stroke;
mod gpu_ops;
mod image_blit;
mod image_memo;
mod image_reduce;
mod image_sampler;
mod knockout;
mod operators;
mod paint;
mod pattern_memo;
mod patterns;
mod pixel_rect;
mod shading_paint;
mod soft_mask;
mod text;
mod text_ops;
mod type3;
mod view;
#[cfg(feature = "vulkan")]
mod vk_ops;
mod xobject;

#[cfg(feature = "cache")]
use devices::CacheState;
pub use diagnostics::{ImageFailure, PageDiagnostics, RenderAbort};
use image_blit::composite_onto_white;
use paint::components_to_color;
pub use view::PageView;
use view::initial_gstate;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use color::Rgb8;
use pdf::{Document, ObjectId};
use raster::{Bitmap, path::PathBuilder, transparency::Backdrop};

use super::font_cache::{DocumentFonts, FontCache};
use super::gstate::{Ctm, GStateStack};
use crate::PageGeometry;
use crate::content::Operator;
use crate::resources::{IMAGE_FILTER_COUNT, PageResources};
#[cfg(any(feature = "gpu-aa", feature = "gpu-icc", feature = "cache"))]
use gpu::GpuCtx;
#[cfg(feature = "vaapi")]
use gpu::JpegQueueHandle;
#[cfg(feature = "gpu-jpeg-huffman")]
use gpu::backend::cuda::CudaBackend;
#[cfg(feature = "vulkan")]
use gpu::backend::vulkan::VulkanBackend;
#[cfg(feature = "gpu-jpeg-huffman")]
use gpu::jpeg_decoder::JpegGpuDecoder;
#[cfg(feature = "nvjpeg")]
use gpu::nvjpeg::NvJpegDecoder;
#[cfg(feature = "nvjpeg2k")]
use gpu::nvjpeg2k::NvJpeg2kDecoder;

/// Identity CTM for passing to raster functions — coordinate transform is
/// already baked into the path points by `to_device`.
const DEVICE_MATRIX: [f64; 6] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

/// Maximum nesting depth for Form `XObjects`.
///
/// A form never re-enters itself (see `PageRenderer::active_forms`), so this
/// bounds the nesting of distinct forms.
const MAX_FORM_DEPTH: u32 = 32;

/// Maximum side, in device pixels, of the bitmap on which a sub-pixel
/// pattern cell is sampled for its mean.
const MAX_TILE_PX: f64 = 4096.0;

/// Maximum nesting depth for tiling patterns.
///
/// `render_pattern_tile` spawns a fresh `PageRenderer` for each tile, so the
/// `form_depth` Form-XObject guard doesn't apply.  Without an independent
/// cap, a self-referential or cyclic pattern chain (`scn /A f` inside
/// pattern A's own content stream) would recurse until stack overflow.
/// Real-world patterns nest 0–1 deep; 4 is a conservative bound that still
/// allows hatch-of-hatch decorative uses without supporting attacker-crafted
/// cycles.
const MAX_PATTERN_DEPTH: u32 = 4;

/// Maximum operators executed per page render before aborting.
///
/// Blunt safety net — no legitimate page has remotely this many operators.
/// A pathological page (infinite-loop content stream, runaway form recursion)
/// reaches this limit quickly; a normal page never comes close.  Not a
/// performance-tuning knob.
const MAX_PAGE_OPS: u64 = 50_000_000;

/// Wall-clock deadline per page render in seconds.
///
/// Blunt safety net — far longer than any legitimate page needs at any DPI.
/// A genuine infinite loop hits `MAX_PAGE_OPS` first (much faster), so in
/// practice this is a last-resort catch for a single extremely-expensive op
/// that doesn't pass through the op-count check.  Not a performance-tuning
/// knob.
const PAGE_RENDER_DEADLINE_SECS: u64 = 60;

/// How many operators to execute between wall-clock deadline checks.
///
/// Amortises `Instant::now()` cost: one syscall per 4 096 ops is negligible
/// overhead.  The overrun after a deadline is bounded to ≤ 4 096 extra ops.
const DEADLINE_CHECK_INTERVAL: u64 = 4_096;

/// Upper bound on distinct unsupported-operator keywords tracked per page for
/// warn-once deduplication.
///
/// Blunt safety net: a real PDF has at most a handful of distinct unsupported
/// operators; this is far beyond that.  Caps the attacker-controlled auxiliary
/// allocation an adversarial content stream (up to `MAX_PAGE_OPS` distinct,
/// arbitrarily long junk keywords) can force, which would otherwise grow to
/// gigabytes before the operator budget aborts the page.  Not a tuning knob.
pub(super) const MAX_WARNED_UNKNOWN_OPS: usize = 256;

/// Renders a decoded operator sequence onto a `Bitmap<Rgb8>`.
pub struct PageRenderer<'doc> {
    /// Target pixel buffer.
    pub(super) bitmap: Bitmap<Rgb8>,
    /// Width of the drawn window (the bitmap) in device pixels.
    width: u32,
    /// Height of the drawn window (the bitmap) in device pixels.
    height: u32,
    /// Width of the whole page in device pixels; equals `width` unless the
    /// renderer draws a window of the page.
    page_width: u32,
    /// Graphics state save/restore stack.
    pub(super) gstate: GStateStack,
    /// Current path under construction, if any.
    path: Option<PathBuilder>,
    /// Pending clip rule set by `W`/`W*` (PDF §8.5.4).
    ///
    /// `W`/`W*` do NOT consume the path — they set this flag so the next
    /// painting or `n` operator intersects the current path into the clip
    /// region before (or instead of) painting.  `false` = non-zero winding,
    /// `true` = even-odd.  Cleared by every path-terminating operator.
    pending_clip: Option<bool>,
    /// Union of the device-space glyph outlines that text shown in a
    /// clipping render mode has added since the last `ET`; `None` when no
    /// such text has been shown.  `ET` intersects the clip with it
    /// (PDF §9.3.6).
    text_clip: Option<raster::path::Path>,
    /// Font face cache for this page.
    font_cache: FontCache,
    /// Rasterise glyphs with 8-bit coverage; `false` draws them bilevel.
    text_antialias: bool,
    /// Fill, stroke and clip paths with supersampled coverage; `false`
    /// scan-converts them at pixel centres.
    vector_antialias: bool,
    /// Resource accessor for the current page or innermost form.
    pub(super) resources: PageResources<'doc>,
    /// The page `/Rotate`, clockwise degrees, normalised to 0, 90, 180 or 270.
    rotate_cw: u16,
    /// Current Form `XObject` nesting depth (0 = top-level page).
    form_depth: u32,
    /// The form `XObject` streams being executed, outermost first.  A form
    /// already on this stack is not entered again.
    active_forms: Vec<ObjectId>,
    /// Current tiling-pattern nesting depth (0 = top-level page).  Spawned
    /// tile renderers carry this incremented; capped at `MAX_PATTERN_DEPTH`
    /// to break self-referential pattern cycles (see `resolve_fill_pattern`).
    pub(super) pattern_depth: u32,
    /// The page's initial CTM: user space to the device window.
    initial_ctm: Ctm,
    /// The matrix from the default coordinate space of the content stream
    /// being executed to the device (PDF §8.7.3.1): the page's initial CTM,
    /// a form's CTM once its `Matrix` is applied, or a pattern tile's CTM.
    /// A shading pattern's `Matrix` is concatenated onto it.
    pub(super) pattern_ctm: Ctm,
    /// The transparency group being painted into.
    group: GroupContext,
    /// The page group the page is painted as (see [`crate::page_group`]):
    /// the bitmap is then transparent and is composited onto white by
    /// [`finish`](Self::finish).  `None` paints onto opaque white directly.
    page_group: Option<crate::resources::TransparencyGroupParams>,
    /// Transparent page-sized bitmaps for transparency groups, each
    /// returned cleared when its group is composited; one per nesting level
    /// is allocated over the page's life.
    group_pool: Vec<Bitmap<Rgb8>>,
    /// Accumulated page diagnostics, populated during rendering.
    diag: PageDiagnostics,
    /// Image `XObjects` already decoded on this page, shared with the
    /// renderers of its pattern cells.
    image_memo: image_memo::SharedImageMemo,
    /// Shadings and tiling cells already resolved on this page, shared with
    /// the renderers of its pattern cells.
    pattern_memo: pattern_memo::SharedPatternMemo,
    /// Maximum operators this page may execute before the render is aborted.
    op_budget: u64,
    /// Absolute wall-clock deadline for this page render.
    deadline: Option<std::time::Instant>,
    /// Number of operators executed so far this page.
    ops_executed: u64,
    /// Set to `Some(reason)` when the render stops early; `None` means ok.
    /// Checked after `execute`; surfaces as `Err` in the per-page result.
    budget_exceeded: Option<RenderAbort>,
    /// Caller-owned cancellation flag, polled before every operator; `true`
    /// stops the render with [`RenderAbort::Cancelled`].
    cancel: Option<Arc<AtomicBool>>,
    /// Distinct unknown operator keywords already warned about this page,
    /// so a content stream that repeats an unsupported op doesn't flood the
    /// log.  Bounded at [`MAX_WARNED_UNKNOWN_OPS`] entries.
    warned_unknown_ops: std::collections::HashSet<Vec<u8>>,
    /// Set once the distinct-keyword cap is reached, so the "no longer
    /// individually reporting" summary warning is emitted exactly once.
    warned_unknown_ops_capped: bool,
    /// Per-filter blit counts used to compute `diag.dominant_filter`.
    filter_counts: [u32; IMAGE_FILTER_COUNT],
    /// Open marked-content sequences of the content stream being executed.
    /// Each content stream starts with none open (see
    /// [`execute_content_stream`](Self::execute_content_stream)).
    marked_content: operators::MarkedContent,
    /// The document's optional-content configuration, resolved for the
    /// [`RenderPurpose::View`](crate::RenderPurpose::View) purpose on first
    /// use unless [`set_optional_content`](Self::set_optional_content)
    /// supplied one.
    optional_content: std::cell::OnceCell<Arc<crate::OptionalContent>>,
    /// GPU parallel-Huffman JPEG decoder via the CUDA backend.
    /// `None` means the path is inactive.
    #[cfg(feature = "gpu-jpeg-huffman")]
    jpeg_gpu: Option<JpegGpuDecoder<CudaBackend>>,
    /// GPU parallel-Huffman JPEG decoder via the Vulkan backend.
    /// Mutually exclusive with `jpeg_gpu` — only one is `Some` per render call.
    #[cfg(all(feature = "gpu-jpeg-huffman", feature = "vulkan"))]
    jpeg_vk: Option<JpegGpuDecoder<VulkanBackend>>,
    /// GPU-accelerated JPEG decoder, present when the `nvjpeg` feature is enabled
    /// and a CUDA device is available.  `None` means CPU-only JPEG decode.
    #[cfg(feature = "nvjpeg")]
    nvjpeg: Option<NvJpegDecoder>,
    /// VA-API JPEG decode queue handle (AMD/Intel iGPU), present when the
    /// `vaapi` feature is enabled and the DRM render node is accessible.
    ///
    /// A [`JpegQueueHandle`] is a cheaply-cloneable sender to a single OS
    /// thread that owns the `VapiJpegDecoder`.  All Rayon workers share the
    /// same worker thread via cloned handles, eliminating Mesa driver contention.
    #[cfg(feature = "vaapi")]
    vaapi_jpeg_queue: Option<JpegQueueHandle>,
    /// GPU-accelerated JPEG 2000 decoder, present when the `nvjpeg2k` feature is
    /// enabled and a CUDA device is available.  `None` means CPU-only JPX decode.
    #[cfg(feature = "nvjpeg2k")]
    nvjpeg2k: Option<NvJpeg2kDecoder>,
    /// Shared GPU context for AA fill dispatch and ICC CMYK→RGB colour conversion.
    /// Present when `gpu-aa` or `gpu-icc` features are enabled.
    #[cfg(any(feature = "gpu-aa", feature = "gpu-icc", feature = "cache"))]
    gpu_ctx: Option<Arc<GpuCtx>>,
    /// Vulkan compute backend, attached when the session was opened with
    /// [`BackendPolicy::ForceVulkan`].  When set, the AA fill dispatch
    /// and the ICC CMYK→RGB path prefer the Vulkan
    /// trait surface (`alloc → upload → record_* → submit → wait → download`)
    /// over the CUDA `GpuCtx` path.  Mutually exclusive with `gpu_ctx` in
    /// practice — the device image cache is CUDA-only, so opening a session under
    /// `ForceVulkan` skips CUDA init entirely (`rasterrocket::render::open_session`).
    /// The image-blit path (`DevicePageBuffer`) stays CUDA-only and is gated
    /// behind `gpu_ctx`, so under Vulkan blits silently land on the CPU
    /// fallback inside the renderer.
    #[cfg(feature = "vulkan")]
    vk_backend: Option<Arc<VulkanBackend>>,
    /// Device-image-cache state.  `Some` only when the `cache` feature is
    /// on AND a cache was wired in via [`Self::set_image_cache`];
    /// `None` keeps the renderer fully CPU-resident.
    #[cfg(feature = "cache")]
    cache_state: Option<CacheState>,
}

/// The transparency group being painted into: the page group, or a form's.
#[derive(Clone, Default)]
struct GroupContext {
    /// The backdrop under a non-isolated group, consulted by blend modes
    /// applied inside it; `None` for an isolated group and for the page.
    backdrop: Option<Arc<Backdrop>>,
    /// Whether the group is a knockout group, in which each object
    /// composites with the group's initial backdrop (see the `knockout`
    /// module).
    knockout: bool,
}

impl<'doc> PageRenderer<'doc> {
    /// Create a renderer for a blank white page of `width × height` pixels,
    /// where 1 user-space unit = 1 device pixel (72 dpi).
    ///
    /// `doc` and `page_id` are used to resolve font and resource dictionaries.
    #[must_use]
    pub fn new(width: u32, height: u32, doc: &'doc Document, page_id: ObjectId) -> Self {
        let geom = PageGeometry {
            width_pts: f64::from(width),
            height_pts: f64::from(height),
            rotate_cw: 0,
            user_unit: 1.0,
            origin_x: 0.0,
            origin_y: 0.0,
            media_box: None,
            crop_box: None,
        };
        Self::with_view(&PageView::whole(&geom, (1.0, 1.0)), doc, page_id)
    }

    /// Create a renderer for `view`: a white bitmap of the view's window, with
    /// the initial CTM that maps the page's user space onto that window.
    ///
    /// # Panics
    ///
    /// Panics if `view.scale_x` or `view.scale_y` is not a positive finite
    /// number.
    #[must_use]
    pub fn with_view(view: &PageView, doc: &'doc Document, page_id: ObjectId) -> Self {
        let (sx, sy) = (view.scale_x, view.scale_y);
        assert!(
            sx.is_finite() && sx > 0.0 && sy.is_finite() && sy > 0.0,
            "PageRenderer::with_view: scales must be positive finite numbers, got {sx} × {sy}"
        );
        debug_assert!(
            u64::from(view.x) + u64::from(view.width) <= u64::from(view.page_width)
                && u64::from(view.y) + u64::from(view.height) <= u64::from(view.page_height),
            "PageRenderer::with_view: window lies outside the page: {view:?}"
        );
        let (width, height, rotate_cw) = (view.width, view.height, view.rotate_cw);

        let mut bitmap = Bitmap::<Rgb8>::new(width, height, 1, false);
        bitmap.data_mut().fill(255u8); // white background

        let initial_ctm = view.initial_ctm();
        let gstate = initial_gstate(width, height, initial_ctm, true);

        let font_cache = FontCache::new(Arc::default(), true);
        let resources = PageResources::new(doc, page_id);

        Self {
            bitmap,
            width,
            height,
            page_width: view.page_width,
            gstate,
            path: None,
            pending_clip: None,
            text_clip: None,
            font_cache,
            text_antialias: true,
            vector_antialias: true,
            resources,
            form_depth: 0,
            active_forms: Vec::new(),
            rotate_cw,
            pattern_depth: 0,
            initial_ctm,
            pattern_ctm: initial_ctm,
            group: GroupContext::default(),
            page_group: None,
            group_pool: Vec::new(),
            diag: PageDiagnostics::default(),
            image_memo: image_memo::ImageMemo::shared(),
            pattern_memo: pattern_memo::PatternMemo::shared(),
            op_budget: MAX_PAGE_OPS,
            deadline: Some(
                std::time::Instant::now()
                    + std::time::Duration::from_secs(PAGE_RENDER_DEADLINE_SECS),
            ),
            ops_executed: 0,
            budget_exceeded: None,
            cancel: None,
            warned_unknown_ops: std::collections::HashSet::new(),
            warned_unknown_ops_capped: false,
            filter_counts: [0u32; IMAGE_FILTER_COUNT],
            marked_content: operators::MarkedContent::default(),
            optional_content: std::cell::OnceCell::new(),
            #[cfg(feature = "gpu-jpeg-huffman")]
            jpeg_gpu: None,
            #[cfg(all(feature = "gpu-jpeg-huffman", feature = "vulkan"))]
            jpeg_vk: None,
            #[cfg(feature = "nvjpeg")]
            nvjpeg: None,
            #[cfg(feature = "vaapi")]
            vaapi_jpeg_queue: None,
            #[cfg(feature = "nvjpeg2k")]
            nvjpeg2k: None,
            #[cfg(any(feature = "gpu-aa", feature = "gpu-icc", feature = "cache"))]
            gpu_ctx: None,
            #[cfg(feature = "vulkan")]
            vk_backend: None,
            #[cfg(feature = "cache")]
            cache_state: None,
        }
    }

    /// Resolve indirect fonts through `fonts`, a store shared by every page
    /// of this renderer's document, instead of a store private to the page.
    /// Call before execute.
    pub fn set_document_fonts(&mut self, fonts: Arc<DocumentFonts>) {
        self.font_cache = FontCache::new(fonts, self.text_antialias);
    }

    /// Select antialiasing: `text` for glyph bitmaps, `vector` for path
    /// fills, strokes and clips.  Both default to `true`.  Call before
    /// execute: the page clip is reset to the whole bitmap in the matching
    /// coverage mode.
    pub fn set_antialias(&mut self, text: bool, vector: bool) {
        self.text_antialias = text;
        self.vector_antialias = vector;
        self.font_cache = FontCache::new(Arc::clone(self.font_cache.document_fonts()), text);
        self.gstate = initial_gstate(self.width, self.height, self.initial_ctm, vector);
    }

    /// Draw optional content as `optional_content` configures it, and
    /// annotations for its purpose.  Call before execute.
    pub fn set_optional_content(&mut self, optional_content: Arc<crate::OptionalContent>) {
        self.optional_content = std::cell::OnceCell::from(optional_content);
    }

    /// Paint the page as the transparency group `group` (ISO 32000-2
    /// §11.4.7): onto a transparent bitmap, composited onto white by
    /// [`finish`](Self::finish), so that a blend mode sees only what the
    /// page has painted and a knockout page group knocks out.  `None`, the
    /// default, paints onto opaque white, which gives the same pixels when
    /// nothing on the page blends (see [`crate::page_group()`]).  Call before
    /// execute.
    pub fn set_page_group(&mut self, group: Option<crate::resources::TransparencyGroupParams>) {
        self.page_group = group;
        self.group.knockout = group.is_some_and(|g| g.knockout);
        self.bitmap = Bitmap::new(self.width, self.height, 1, group.is_some());
        if group.is_none() {
            self.bitmap.data_mut().fill(u8::MAX);
        }
    }

    /// The document's optional-content configuration in effect.
    pub(super) fn optional_content(&self) -> &Arc<crate::OptionalContent> {
        self.optional_content.get_or_init(|| {
            Arc::new(crate::OptionalContent::new(
                self.resources.doc(),
                crate::RenderPurpose::View,
            ))
        })
    }

    /// Whether content governed by the optional content group or
    /// membership dictionary `oc` is visible.
    pub(super) fn oc_visible(&self, oc: &pdf::Object) -> bool {
        self.optional_content().is_visible(self.resources.doc(), oc)
    }

    /// Why the render stopped early, if it did.
    ///
    /// `Some(reason)` means the page was aborted before all of its operators
    /// ran: it exceeded the operator count cap, the wall-clock deadline or
    /// the form nesting limit, or it was cancelled.  The rendered bitmap is
    /// partial and the per-page `Result` must be an `Err`.
    ///
    /// Check this after [`execute`](Self::execute) and before
    /// [`finish`](Self::finish).
    #[must_use]
    pub const fn budget_status(&self) -> Option<&RenderAbort> {
        self.budget_exceeded.as_ref()
    }

    /// Stop the render with [`RenderAbort::Cancelled`] once `cancel` reads
    /// `true`.  The flag is polled before every operator, including those of
    /// forms, Type 3 glyphs and pattern cells.  Call before execute.
    pub fn set_cancel(&mut self, cancel: Arc<AtomicBool>) {
        self.cancel = Some(cancel);
    }

    /// Seed this (child) renderer's work watchdog from a parent renderer.
    ///
    /// Tiling-pattern tiles and any other sub-render that spawns a fresh
    /// [`PageRenderer`] would otherwise each get a brand-new 50 M-op budget and
    /// a brand-new wall-clock deadline — a denial-of-service escape hatch: the per-page
    /// watchdog is blind to all work done inside a child renderer, so a
    /// pathological pattern content stream (nested up to `MAX_PATTERN_DEPTH`)
    /// could burn arbitrarily many ops / seconds while the parent's
    /// `execute()` loop is blocked synchronously inside the child render.
    ///
    /// The page budget MUST be *shared*, not reset, across every level of
    /// sub-rendering.  This adopts the parent's *remaining* op allowance and
    /// its exact wall-clock `Instant` deadline (the same monotonic instant, so
    /// the deadline does not slide forward per child).  An already-tripped
    /// parent budget is inherited so the child does no work at all.  Call
    /// [`fold_child_into`](Self::fold_child_into) on the parent afterwards to
    /// account for the work the child actually performed.
    pub(super) fn adopt_parent_budget(&mut self, parent: &Self) {
        // Remaining allowance = parent budget − parent ops already spent.
        // saturating_sub: if the parent is already at/over budget the child
        // gets zero allowance and trips on its first op.
        self.op_budget = parent.op_budget.saturating_sub(parent.ops_executed);
        self.ops_executed = 0;
        self.deadline = parent.deadline;
        self.cancel.clone_from(&parent.cancel);
        self.budget_exceeded.clone_from(&parent.budget_exceeded);
    }

    /// Draw image `XObjects`, shadings and tiling cells from `parent`'s
    /// memos, and keep this renderer's in them, so a pattern cell does not
    /// decode or rasterise what the page already has.
    pub(super) fn share_memos(&mut self, parent: &Self) {
        self.image_memo = std::rc::Rc::clone(&parent.image_memo);
        self.pattern_memo = std::rc::Rc::clone(&parent.pattern_memo);
    }

    /// Fold a finished child renderer (a tiling-pattern tile) into its
    /// parent: its watchdog consumption, so the shared per-page budget keeps
    /// accounting for work done inside sub-renders, and its skipped images,
    /// so they are reported on the page.
    ///
    /// Without this the parent's `ops_executed` would not reflect the child's
    /// work and a sequence of expensive pattern fills would never trip the
    /// page budget.  A budget breach inside the child is propagated verbatim
    /// so the outer `execute()` bails loudly on its next iteration.
    ///
    /// A cell is rasterised once per phase and tint, so an image failure or
    /// a missing font it reports is folded once, however many rasters the
    /// page renders.  Whether images or text were drawn carries over; the
    /// cell's filter counts and PPI hint do not, being measured in the
    /// cell's pixels rather than the page's.
    pub(super) fn fold_child_into(&self, parent: &mut Self) {
        parent.ops_executed = parent.ops_executed.saturating_add(self.ops_executed);
        if parent.budget_exceeded.is_none() && self.budget_exceeded.is_some() {
            parent.budget_exceeded.clone_from(&self.budget_exceeded);
        }
        let diag = &mut parent.diag;
        for failure in &self.diag.image_failures {
            if !diag.image_failures.contains(failure) {
                diag.image_failures.push(failure.clone());
            }
        }
        for font in &self.diag.missing_fonts {
            if !diag.missing_fonts.contains(font) {
                diag.missing_fonts.push(font.clone());
            }
        }
        diag.has_images |= self.diag.has_images;
        diag.has_vector_text |= self.diag.has_vector_text;
    }

    /// Override the operator-count budget.  Test-only — used to trigger the
    /// watchdog with a small synthetic vector without running 50 M operators.
    #[cfg(test)]
    pub(crate) const fn set_op_budget_for_test(&mut self, budget: u64) {
        self.op_budget = budget;
    }

    /// Override the wall-clock deadline.  Pass `None` to disable the
    /// deadline entirely (test-only — lets tests isolate the op-count path).
    #[cfg(test)]
    pub(crate) const fn set_deadline_for_test(&mut self, dl: Option<std::time::Instant>) {
        self.deadline = dl;
    }

    /// Return the number of operators executed so far.  Test-only accessor
    /// so watchdog unit tests can verify the counter without accessing a
    /// private field from a child module.
    #[cfg(test)]
    pub(crate) const fn ops_executed_for_test(&self) -> u64 {
        self.ops_executed
    }

    /// `(tracked distinct unknown-op keywords, cap-reached flag)`.  Test-only
    /// accessor so the dedup unit test can assert the set stays bounded under
    /// adversarial input without touching private fields cross-module.
    #[cfg(test)]
    pub(crate) fn warned_unknown_ops_state_for_test(&self) -> (usize, bool) {
        (
            self.warned_unknown_ops.len(),
            self.warned_unknown_ops_capped,
        )
    }

    /// Consume the renderer and return the finished bitmap and page diagnostics.
    #[must_use]
    pub fn finish(mut self) -> (Bitmap<Rgb8>, PageDiagnostics) {
        self.diag.dominant_filter = crate::resources::image::dominant_filter(&self.filter_counts);

        if self.page_group.is_some() {
            composite_onto_white(&mut self.bitmap);
        }
        (self.bitmap, self.diag)
    }

    /// Execute a slice of decoded operators in order.
    ///
    /// Stops early and sets [`budget_exceeded`](Self::budget_status) if the
    /// page exceeds the operator-count cap (`MAX_PAGE_OPS`) or the wall-clock
    /// deadline (`PAGE_RENDER_DEADLINE_SECS`).  The deadline is checked once
    /// every `DEADLINE_CHECK_INTERVAL` ops to amortise `Instant::now()` cost.
    pub fn execute(&mut self, ops: &[Operator]) {
        for op in ops {
            // Bail early if a previous call (e.g. a recursive form XObject)
            // already tripped the budget.
            if self.budget_exceeded.is_some() {
                return;
            }
            if self
                .cancel
                .as_ref()
                .is_some_and(|cancel| cancel.load(Ordering::Relaxed))
            {
                self.budget_exceeded = Some(RenderAbort::Cancelled {
                    ops: self.ops_executed,
                });
                return;
            }
            self.execute_one(op);
            // saturating_add: the op cap fires at MAX_PAGE_OPS long before u64
            // could wrap, but a test or future caller may set a huge budget
            // with the deadline disabled — never silently wrap to a low count
            // (which would mask, not catch, an unbounded stream).
            self.ops_executed = self.ops_executed.saturating_add(1);
            if self.ops_executed > self.op_budget {
                self.budget_exceeded = Some(RenderAbort::OperatorBudget {
                    ops: self.ops_executed,
                });
                return;
            }
            // Check the wall-clock deadline periodically to keep the syscall cost low.
            if self.ops_executed.is_multiple_of(DEADLINE_CHECK_INTERVAL)
                && self
                    .deadline
                    .is_some_and(|dl| std::time::Instant::now() >= dl)
            {
                self.budget_exceeded = Some(RenderAbort::Deadline {
                    ops: self.ops_executed,
                });
                return;
            }
        }
    }

    /// Execute `ops` as a content stream nested in the current one: a form
    /// `XObject`, a Type 3 glyph description or a tiling pattern cell.
    ///
    /// The nested stream's `q` and `Q` balance within it (PDF §8.4.2): its
    /// `Q`s cannot pop the states saved before it, and the states its
    /// unbalanced `q`s leave are discarded when it ends.  Marked-content
    /// sequences cannot span content streams (PDF §14.6), so the nested
    /// stream starts with no open sequence and the enclosing stream's
    /// sequences are restored afterwards, whatever the nested stream left
    /// open or closed.  A nested stream is executed only while every
    /// enclosing sequence is visible, so the visibility it inherits is
    /// unchanged.
    fn execute_content_stream(&mut self, ops: &[Operator]) {
        let scope = self.gstate.enter_stream();
        let enclosing = std::mem::take(&mut self.marked_content);
        self.execute(ops);
        self.marked_content = enclosing;
        self.gstate.leave_stream(scope);
    }

    /// Render all annotations on the page (PDF §12.5).
    ///
    /// Each annotation shown for the render purpose (its `/F` flags) and
    /// visible under its `/OC` that has an `AP/N` (normal appearance) stream
    /// is rendered after the page content stream.  Only the `/N` entry is
    /// used; rollover (`/R`) and down (`/D`) appearances are ignored (this is
    /// a rasterizer, not an interactive viewer).
    ///
    /// Annotations with no appearance stream are silently skipped — many
    /// annotation types (e.g. `Link`) have no visible appearance by default.
    ///
    /// Call this after [`execute`](Self::execute) and before [`finish`](Self::finish).
    pub fn render_annotations(&mut self, page_id: ObjectId) {
        // PDF §12.5.5: an appearance is drawn in default user space with the
        // initial graphics state, whatever the content stream left behind: an
        // unbalanced `q`, a clip, a CTM, an unterminated text object's
        // clipping outlines or an open marked-content sequence.
        self.gstate = initial_gstate(
            self.width,
            self.height,
            self.initial_ctm,
            self.vector_antialias,
        );
        self.path = None;
        self.pending_clip = None;
        self.text_clip = None;
        self.marked_content = operators::MarkedContent::default();
        annotations::render_annotations(self, page_id);
    }
}

#[cfg(test)]
mod tests;
