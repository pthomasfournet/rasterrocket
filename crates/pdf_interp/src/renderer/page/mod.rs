//! Per-page operator dispatcher.
//!
//! [`PageRenderer`] holds a target canvas (an RGB or ink [`Bitmap`]) and a
//! `GStateStack`, iterates over a decoded operator sequence, and calls into the
//! `raster` crate for each painting operator.
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
//! - `form_memo` — parsed form `XObjects`, kept per page.
//! - `pixel_rect` — rectangles of device pixels.
//! - `canvas` — the page bitmap, of RGB or ink pixels, and the dispatch of
//!   paint calls to either.
//! - `annotations` — annotation appearances, stored and synthesised.
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
mod band;
mod canvas;
mod colour_set;
pub(crate) mod depth;
mod device_blit;
#[cfg(feature = "vulkan")]
mod device_page;
mod devices;
mod diagnostics;
mod fill_stroke;
mod finished;
mod form_memo;
mod group_space;
mod image_blit;
mod image_compositor;
mod image_hairline;
mod image_memo;
mod image_reduce;
mod image_sampler;
mod ink_map;
mod knockout;
mod operators;
mod paint;
mod pattern_memo;
mod patterns;
mod pixel_rect;
mod shading_paint;
mod soft_mask;
mod space_convert;
mod text;
mod text_census;
mod text_context;
mod text_glyphs;
mod text_ops;
mod text_shaping;
mod text_stroke;
mod trace;
mod type3;
mod view;
mod xobject;

pub use band::PageState;
use canvas::{CanvasKind, PageCanvas};
#[cfg(feature = "vulkan")]
pub use device_page::{DeviceFallback, DeviceRecording, device_page_ineligible};
#[cfg(feature = "cache")]
use devices::CacheState;
pub(crate) use diagnostics::source_ppi;
pub use diagnostics::{
    DevicePage, GlyphCounts, GroupSpaceApproximations, ImageFailure, PageDiagnostics, RenderAbort,
};
pub use finished::FinishedCanvas;
#[cfg(feature = "cache")]
pub(crate) use image_reduce::draw_reduces;
use paint::components_to_color;
pub use trace::PaintTrace;
pub use view::PageView;
use view::initial_gstate;
#[cfg(feature = "cache")]
pub(crate) use xobject::image_draw_size;

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use color::{Rgb8, SampleDepth};
use pdf::{Document, ObjectId};
use raster::pipe::ChannelMask;
use raster::{Bitmap, path::PathBuilder};

use super::color::InkChannels;
use super::font_cache::{DocumentFonts, FontCache};
use super::gstate::{Ctm, GStateStack};
use crate::PageGeometry;
use crate::content::Operator;
use crate::resources::{IMAGE_FILTER_COUNT, PageResources};
use crate::separation::{CanvasRequest, InkSetup};
#[cfg(any(feature = "gpu-icc", feature = "cache"))]
use gpu::GpuCtx;
#[cfg(feature = "vaapi")]
use gpu::JpegQueueHandle;
#[cfg(feature = "vulkan")]
use gpu::backend::vulkan::VulkanBackend;
#[cfg(feature = "nvjpeg")]
use gpu::nvjpeg::NvJpegDecoder;
#[cfg(feature = "nvjpeg2k")]
use gpu::nvjpeg2k::NvJpeg2kDecoder;

/// How a renderer obtains its GPU JPEG 2000 decoder and which decodes it
/// offers it.
#[cfg(feature = "nvjpeg2k")]
#[derive(Clone, Copy)]
enum Nvjpeg2kSupply {
    /// The attached decoder, if any, is offered every decode.
    Attached,
    /// The decoder is offered only the draws
    /// `PageRenderer::nvjpeg2k_offered` selects, and is obtained from the
    /// source, while it is `Some`, by the first decode of theirs that uses
    /// it and finds none attached.
    OnDemand(Option<crate::resources::image::Jpeg2kSource>),
}

#[cfg(feature = "nvjpeg2k")]
impl Nvjpeg2kSupply {
    /// The source a decode offered the decoder obtains it from: none when
    /// it is attached.
    const fn source(&mut self) -> Option<&mut Option<crate::resources::image::Jpeg2kSource>> {
        match self {
            Self::Attached => None,
            Self::OnDemand(source) => Some(source),
        }
    }

    /// The dispatch threshold of the decoder, in pixels.
    fn jpx_threshold(&self) -> u32 {
        use crate::resources::image::{JpxSupply, jpx_threshold};
        jpx_threshold(match self {
            Self::Attached => JpxSupply::Attached,
            Self::OnDemand(_) => JpxSupply::OnDemand,
        })
    }
}

/// Identity CTM for passing to raster functions — coordinate transform is
/// already baked into the path points by `to_device`.
const DEVICE_MATRIX: [f64; 6] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

/// Maximum nesting depth for Form `XObjects`.
///
/// A form never re-enters itself (see `PageRenderer::active_forms`), so this
/// bounds the nesting of distinct forms, and with it the recursion of the
/// form interpreter.  A form invoked this many forms deep is not drawn and
/// the page continues (see [`PageDiagnostics::forms_too_deep`]).  The
/// page-group and overprint scans follow resources to the same depth.
pub const MAX_FORM_DEPTH: u32 = 40;

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

/// The page-sized storage of a render, carried from one page to the next:
/// the buffers of the RGB bitmap and of the ink canvas.
///
/// [`PageRenderer::try_with_page_group_in`] builds the page's canvas in
/// them, and [`PageRenderer::try_finish_keeping`] returns those the
/// finished bitmap does not hold: the ink canvas, and the alpha plane of a
/// page group composited onto paper.  Reclaiming the bitmap's own storage
/// into [`rgb`](Self::rgb) ([`raster::BitmapBuffers::reclaim`]) completes
/// the set for the next page.
#[derive(Debug, Default)]
pub struct CanvasBuffers {
    /// The RGB bitmap's colour buffer and alpha plane.
    pub rgb: raster::BitmapBuffers,
    /// The 16-bit RGB bitmap's colour buffer and alpha plane.
    pub rgb16: raster::BitmapBuffers<u16>,
    /// The ink canvas's colour buffer and alpha plane.
    pub ink: raster::BitmapBuffers,
    /// The 16-bit ink canvas's colour buffer and alpha plane.
    pub ink16: raster::BitmapBuffers<u16>,
    /// The gray bitmap's colour buffer and alpha plane, of a canvas
    /// converted to a gray output profile.
    pub gray: raster::BitmapBuffers,
    /// The 16-bit gray bitmap's colour buffer and alpha plane.
    pub gray16: raster::BitmapBuffers<u16>,
    /// The CMYK bitmap's colour buffer and alpha plane, of an ink canvas
    /// finished as CMYK.
    pub cmyk: raster::BitmapBuffers,
    /// The 16-bit CMYK bitmap's colour buffer and alpha plane.
    pub cmyk16: raster::BitmapBuffers<u16>,
}

/// The ink map of a page, shared by the renderers of its pattern cells
/// and Type 3 glyphs.
type SharedInkMap = Rc<RefCell<ink_map::InkMap>>;

/// The ink map of a finished page.  The renderers sharing it are children,
/// finished before the page; were one to outlive the page, the map is
/// taken from it rather than lost, and debug builds fail.
fn page_ink_map(ink_map: SharedInkMap) -> ink_map::InkMap {
    debug_assert_eq!(
        Rc::strong_count(&ink_map),
        1,
        "a child renderer outlived the page that shares its ink map"
    );
    Rc::try_unwrap(ink_map).map_or_else(|shared| shared.take(), RefCell::into_inner)
}

/// Renders a decoded operator sequence onto a page canvas of RGB or ink
/// pixels, finished as a `Bitmap<Rgb8>`.
pub struct PageRenderer<'doc> {
    /// Target pixel buffer.
    canvas: PageCanvas,
    /// The working space colours convert into: the page's, sRGB while a
    /// soft-mask group is painted.
    space: crate::WorkingSpace,
    /// The working space an ink canvas finished as RGB converts into.
    finish_space: crate::WorkingSpace,
    /// The storage not in `canvas`: for an ink canvas, the RGB buffers the
    /// finished bitmap is built in; for an RGB canvas, the ink buffers,
    /// carried through to the next page.
    buffers: CanvasBuffers,
    /// Width of the drawn window (the bitmap) in device pixels.
    width: u32,
    /// Height of the drawn window (the bitmap) in device pixels.
    height: u32,
    /// Width of the page box in points, before rotation: the basis of the
    /// source-resolution hint.
    unrotated_width_pts: f64,
    /// Width of the whole page in device pixels; equals `width` unless the
    /// renderer draws a window of the page.
    page_width: u32,
    /// Height of the whole page in device pixels; equals `height` unless the
    /// renderer draws a window of the page.
    page_height: u32,
    /// Offset of the drawn window's top-left corner from the page's, in
    /// device pixels.  Every CTM in the graphics state maps to the whole
    /// page's device space; a device coordinate is computed there and
    /// translated by this integer offset last, so that a window holds the
    /// values the whole-page render computes for the same pixels.
    window_origin: (u32, u32),
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
    /// The end of the string shown last, while only operators that keep
    /// shaping context have followed it (see [`text_context`]).
    text_tail: Option<text_context::TextTail>,
    /// Font face cache for this page.
    font_cache: FontCache,
    /// Rasterise glyphs with 8-bit coverage; `false` draws them bilevel.
    text_antialias: bool,
    /// Fill, stroke and clip paths with exact-area coverage; `false`
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
    /// The page's initial CTM: user space to the whole page's device space.
    initial_ctm: Ctm,
    /// The matrix from the default coordinate space of the content stream
    /// being executed to the device (PDF §8.7.3.1): the page's initial CTM,
    /// a form's CTM once its `Matrix` is applied, or a pattern tile's CTM.
    /// A shading pattern's `Matrix` is concatenated onto it.
    pub(super) pattern_ctm: Ctm,
    /// The transparency group being painted into.
    group: GroupContext,
    /// The page group the page is painted as (see [`mod@crate::page_group`]):
    /// the bitmap is then transparent and is composited onto white by
    /// [`finish`](Self::finish).  `None` paints onto opaque white directly.
    page_group: Option<crate::resources::TransparencyGroupParams>,
    /// Transparent page-sized canvases for transparency groups, each
    /// returned cleared when its group is composited; one per nesting level
    /// and canvas kind is allocated over the page's life.
    group_pool: Vec<PageCanvas>,
    /// The conversions of the `CalRGB` and ICC group spaces the page's
    /// groups composited in, by space and working space; `None` for a
    /// space that does not convert.
    group_rgb: Vec<group_space::GroupRgbEntry>,
    /// The spot colorants of the page and their channels on an ink canvas.
    ink_map: SharedInkMap,
    /// The settings of an ink canvas: whether overprint is simulated, the
    /// separator of colours not given as ink, and whether the page
    /// finishes as CMYK.  Shared by the renderers of the page's content
    /// (see [`Self::share_memos`]).
    pub(super) ink: InkSetup,
    /// The channels the renderer's paints wrote — the union over each
    /// paint of the channels its overprint leaves it, every channel for a
    /// paint without overprint — and those their colours carry.
    painted: Cell<InkChannels>,
    /// Accumulated page diagnostics, populated during rendering.
    diag: PageDiagnostics,
    /// Whether shown glyphs are added to [`PageDiagnostics::glyphs`]:
    /// not in the renderer of a pattern cell or Type 3 glyph, nor in a
    /// window after a page's first (each window executes the whole
    /// content, which the first has counted), nor while a soft mask's
    /// group or a Type 3 glyph's description executes.
    census: text_census::Census,
    /// The page's paint trace while it is recorded (see the `trace`
    /// module): only in the renderer of a page's first window, never in
    /// the renderer of a pattern cell or Type 3 glyph.
    trace: Option<Box<trace::PaintTrace>>,
    /// Image `XObjects` already decoded on this page, shared with the
    /// renderers of its pattern cells, and the document's shared decodes.
    image_memo: image_memo::SharedImageMemo,
    /// The tint-transform work left to the images of the page, shared with
    /// the renderers of its pattern cells and Type 3 glyphs.
    tint_budget: std::rc::Rc<crate::resources::image::colorspace::TintBudget>,
    /// Shadings and tiling cells already resolved on this page, shared with
    /// the renderers of its pattern cells.
    pattern_memo: pattern_memo::SharedPatternMemo,
    /// Form `XObjects` already parsed on this page, shared with the
    /// renderers of its pattern cells.
    form_memo: form_memo::SharedFormMemo,
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
    /// The most pixels a canvas [`Self::try_canvas`] allocates may hold;
    /// a larger one is refused.
    #[cfg(test)]
    max_canvas_px: u64,
    /// The most pixels a [`Self::take_region`] snapshot may hold; a larger
    /// one is refused.
    #[cfg(test)]
    max_snapshot_px: u64,
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
    /// Which decodes are offered [`Self::nvjpeg2k`], and how it is obtained.
    #[cfg(feature = "nvjpeg2k")]
    nvjpeg2k_supply: Nvjpeg2kSupply,
    /// Shared CUDA context for ICC CMYK→RGB colour conversion and image
    /// blits.  Present when the `gpu-icc` or `cache` feature is enabled.
    #[cfg(any(feature = "gpu-icc", feature = "cache"))]
    gpu_ctx: Option<Arc<GpuCtx>>,
    /// Vulkan compute backend, attached when the session was opened with
    /// [`BackendPolicy::ForceVulkan`]; the ICC CMYK→RGB CLUT then runs
    /// on it, and the image cache, when attached, keeps its
    /// slabs and runs its blits on the same device.  Such a session
    /// initialises no CUDA context: `gpu_ctx` is `None` and no nvJPEG or
    /// nvJPEG2000 decoder is built.
    #[cfg(feature = "vulkan")]
    vk_backend: Option<Arc<VulkanBackend>>,
    /// Device-image-cache state.  `Some` only when the `cache` feature is
    /// on AND a cache was wired in via [`Self::set_image_cache`];
    /// `None` keeps the renderer fully CPU-resident.
    #[cfg(feature = "cache")]
    cache_state: Option<CacheState>,
    /// The page's paint commands while it is recorded for the device page
    /// raster instead of painted; see [`Self::record_for_device`].
    #[cfg(feature = "vulkan")]
    device_page: Option<Box<device_page::Recording>>,
}

/// The 8-bit bitmap of `finished`, the canvas of a renderer of 8-bit
/// samples.
///
/// # Panics
///
/// On a 16-bit bitmap: a renderer's depth is fixed when it is built.
fn rgb8(finished: FinishedCanvas) -> Bitmap<Rgb8> {
    match finished {
        FinishedCanvas::Rgb8(rgb) => rgb,
        other => {
            panic!("an 8-bit RGB finish, not {other:?}: a renderer's canvas is RGB of its depth")
        }
    }
}

/// The work watchdog of a page render, as one renderer hands it to the
/// next ([`PageRenderer::remaining_budget`]).
pub(super) struct WorkBudget {
    /// Operators the render may still execute.
    ops: u64,
    /// Absolute wall-clock deadline.
    deadline: Option<std::time::Instant>,
    /// Caller-owned cancellation flag.
    cancel: Option<Arc<AtomicBool>>,
    /// Why the render stopped, if it did.
    exceeded: Option<RenderAbort>,
}

/// The transparency group being painted into: the page group, or a form's.
#[derive(Clone, Default)]
struct GroupContext {
    /// The backdrop under a non-isolated group, consulted by blend modes
    /// applied inside it; `None` for an isolated group and for the page.
    backdrop: Option<Arc<depth::CanvasBackdrop>>,
    /// Whether the group is a knockout group, in which each object
    /// composites with the group's initial backdrop (see the `knockout`
    /// module).
    knockout: bool,
    /// The gray blending space of a group composited in gray, the space
    /// blend modes inside it blend in; `None` for a group blending in the
    /// components of its canvas.
    gray: Option<Rc<group_space::GrayReduction>>,
}

impl<'doc> PageRenderer<'doc> {
    /// Create a renderer for a blank white page of `width × height` pixels,
    /// where 1 user-space unit = 1 device pixel (72 dpi).
    ///
    /// `doc` and `page_id` are used to resolve font and resource dictionaries.
    #[must_use]
    pub fn new(width: u32, height: u32, doc: &'doc Document, page_id: ObjectId) -> Self {
        Self::with_view(&Self::unit_view(width, height), doc, page_id)
    }

    /// [`Self::new`] on an ink canvas, whatever the page's graphics states.
    #[cfg(test)]
    pub(crate) fn new_ink(width: u32, height: u32, doc: &'doc Document, page_id: ObjectId) -> Self {
        let view = Self::unit_view(width, height);
        Self::with_page_group(
            &view,
            doc,
            page_id,
            None,
            &CanvasRequest::ProcessRgb { overprint: true },
        )
    }

    /// The whole of a `width` × `height`-point page at one pixel per point.
    fn unit_view(width: u32, height: u32) -> PageView {
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
        PageView::whole(&geom, (1.0, 1.0))
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
        Self::with_page_group(view, doc, page_id, None, &CanvasRequest::Rgb)
    }

    /// [`Self::with_view`] for a page painted as the transparency group
    /// `group` (see [`Self::set_page_group`]), on the canvas `canvas`
    /// requests: sRGB, or process and spot inks finished as sRGB with
    /// overprint (ISO 32000-1 §8.6.7) simulated, or finished as CMYK.  The
    /// canvas is allocated once: transparent for a group and opaque white
    /// otherwise.  The overprint preview renders a page without overprint
    /// as overprint off does, at the cost of the ink model, so it is meant
    /// for pages for which [`crate::page_uses_overprint`] holds.
    ///
    /// # Panics
    ///
    /// As [`Self::with_view`], and when the canvas cannot be allocated.
    #[must_use]
    pub fn with_page_group(
        view: &PageView,
        doc: &'doc Document,
        page_id: ObjectId,
        group: Option<crate::resources::TransparencyGroupParams>,
        canvas: &CanvasRequest,
    ) -> Self {
        Self::try_with_page_group(view, doc, page_id, group, canvas)
            .expect("PageRenderer::with_page_group: canvas allocation failed")
    }

    /// [`Self::with_page_group`] that reports a canvas the allocator
    /// refuses as an error.
    ///
    /// # Errors
    ///
    /// [`raster::BitmapAllocError`] when the canvas cannot be allocated.
    ///
    /// # Panics
    ///
    /// As [`Self::with_view`].
    pub fn try_with_page_group(
        view: &PageView,
        doc: &'doc Document,
        page_id: ObjectId,
        group: Option<crate::resources::TransparencyGroupParams>,
        canvas: &CanvasRequest,
    ) -> Result<Self, raster::BitmapAllocError> {
        Self::try_with_page_group_in(
            view,
            doc,
            page_id,
            group,
            canvas,
            SampleDepth::Eight,
            CanvasBuffers::default(),
        )
    }

    /// [`Self::try_with_page_group`] of samples of `depth`, built in
    /// `buffers` ([`raster::Bitmap::try_new_filled_in`]): an RGB canvas is
    /// allocated in `buffers.rgb`, or `buffers.rgb16` at 16 bits; an ink
    /// canvas in `buffers.ink` or `buffers.ink16`, converting on finish
    /// into the RGB buffers of its depth, or the CMYK ones (`cmyk`,
    /// `cmyk16`) for a page finished as CMYK.
    /// [`Self::try_finish_keeping`] returns them for the next page.
    ///
    /// # Errors
    ///
    /// [`raster::BitmapAllocError`] when the canvas cannot be allocated.
    ///
    /// # Panics
    ///
    /// As [`Self::with_view`].
    pub fn try_with_page_group_in(
        view: &PageView,
        doc: &'doc Document,
        page_id: ObjectId,
        group: Option<crate::resources::TransparencyGroupParams>,
        canvas: &CanvasRequest,
        depth: SampleDepth,
        buffers: CanvasBuffers,
    ) -> Result<Self, raster::BitmapAllocError> {
        let alpha = group.is_some();
        let mut spare = buffers;
        let kind = if canvas.is_ink() {
            CanvasKind::ink(depth)
        } else {
            CanvasKind::rgb(depth)
        };
        let finish_space = canvas.space();
        debug_assert!(
            finish_space == crate::WorkingSpace::Srgb || depth == SampleDepth::Sixteen,
            "the wide working space is painted at 16 bits"
        );
        let (ink, working, approximated) = canvas.ink_setup_in(doc);
        let canvas = PageCanvas::try_new_in(kind, view.width, view.height, alpha, &mut spare)?;
        let mut renderer = Self::with_canvas(view, doc, page_id, canvas, group, ink, working);
        renderer.buffers = spare;
        renderer.finish_space = finish_space;
        if approximated {
            renderer.diag.approximated_group_spaces.cmyk_profile = true;
        }
        Ok(renderer)
    }

    /// A renderer for `view` painting onto `canvas`, a bitmap of the view's
    /// window, as the page group `group`, an ink canvas set up as `ink`,
    /// its colours converting into `space`.
    ///
    /// # Panics
    ///
    /// As [`Self::with_view`].
    fn with_canvas(
        view: &PageView,
        doc: &'doc Document,
        page_id: ObjectId,
        canvas: PageCanvas,
        group: Option<crate::resources::TransparencyGroupParams>,
        ink: InkSetup,
        space: crate::WorkingSpace,
    ) -> Self {
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
        debug_assert_eq!(
            canvas.dims(),
            (width, height),
            "the canvas covers the window"
        );

        let initial_ctm = view.page_ctm();
        let gstate = initial_gstate(width, height, initial_ctm, true);

        let font_cache = FontCache::new(Arc::default(), true);
        let resources = PageResources::new(doc, page_id);
        let ink_map = ink_map::InkMap::new(canvas.kind().depth(), ink.separator.clone());

        let mut renderer = Self {
            canvas,
            space,
            finish_space: space,
            buffers: CanvasBuffers::default(),
            width,
            height,
            unrotated_width_pts: view.unrotated_width_pts(),
            page_width: view.page_width,
            page_height: view.page_height,
            window_origin: (view.x, view.y),
            gstate,
            path: None,
            pending_clip: None,
            text_clip: None,
            text_tail: None,
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
            page_group: group,
            group_pool: Vec::new(),
            group_rgb: Vec::new(),
            ink_map: Rc::new(RefCell::new(ink_map)),
            ink,
            painted: Cell::new(InkChannels::NONE),
            diag: PageDiagnostics::default(),
            census: text_census::Census::Counted,
            trace: None,
            image_memo: image_memo::ImageMemo::shared(page_id, Arc::default()),
            tint_budget: std::rc::Rc::new(crate::resources::image::colorspace::TintBudget::page()),
            pattern_memo: pattern_memo::PatternMemo::shared(),
            form_memo: form_memo::FormMemo::shared(),
            op_budget: MAX_PAGE_OPS,
            deadline: Some(
                std::time::Instant::now()
                    + std::time::Duration::from_secs(PAGE_RENDER_DEADLINE_SECS),
            ),
            ops_executed: 0,
            budget_exceeded: None,
            cancel: None,
            #[cfg(test)]
            max_canvas_px: u64::MAX,
            #[cfg(test)]
            max_snapshot_px: u64::MAX,
            warned_unknown_ops: std::collections::HashSet::new(),
            warned_unknown_ops_capped: false,
            filter_counts: [0u32; IMAGE_FILTER_COUNT],
            marked_content: operators::MarkedContent::default(),
            optional_content: std::cell::OnceCell::new(),
            #[cfg(feature = "nvjpeg")]
            nvjpeg: None,
            #[cfg(feature = "vaapi")]
            vaapi_jpeg_queue: None,
            #[cfg(feature = "nvjpeg2k")]
            nvjpeg2k: None,
            #[cfg(feature = "nvjpeg2k")]
            nvjpeg2k_supply: Nvjpeg2kSupply::Attached,
            #[cfg(any(feature = "gpu-icc", feature = "cache"))]
            gpu_ctx: None,
            #[cfg(feature = "vulkan")]
            vk_backend: None,
            #[cfg(feature = "cache")]
            cache_state: None,
            #[cfg(feature = "vulkan")]
            device_page: None,
        };
        renderer.group = renderer.page_group_context(group);
        renderer
    }

    /// The group context of the page painted as the page group `group`: a
    /// knockout group when it says so, and blending in gray when its space
    /// is gray and its canvas holds RGB (ISO 32000-2 §11.4.7).  A page
    /// painted in ink composites in ink whatever its group declares.
    fn page_group_context(
        &self,
        group: Option<crate::resources::TransparencyGroupParams>,
    ) -> GroupContext {
        let gray = group.and_then(|g| g.space) == Some(crate::resources::GroupSpace::Gray)
            && !self.canvas.kind().is_ink();
        GroupContext {
            knockout: group.is_some_and(|g| g.knockout),
            gray: gray.then(|| self.gray_reduction()),
            ..GroupContext::default()
        }
    }

    /// Resolve indirect fonts through `fonts`, a store shared by every page
    /// of this renderer's document, instead of a store private to the page.
    /// Call before execute.
    ///
    /// A store serves the first document whose renderer it is given to: a
    /// store already serving another document is ignored, with a warning,
    /// and the page resolves its fonts itself.
    pub fn set_document_fonts(&mut self, fonts: Arc<DocumentFonts>) {
        if !fonts.bind(self.resources.doc()) {
            log::warn!(
                "rasterrocket-interp: the document font store serves another document — \
                 resolving this page's fonts without it"
            );
            return;
        }
        self.font_cache = FontCache::new(fonts, self.text_antialias);
    }

    /// Decode image `XObjects` drawn on several pages once through
    /// `images`, a store shared by every page of this renderer's document,
    /// instead of once per page.  Call before execute.
    ///
    /// A store serves the first document whose renderer it is given to: a
    /// store already serving another document is ignored, with a warning,
    /// and the page decodes its images itself.
    pub fn set_document_images(&mut self, images: Arc<super::DocumentImages>) {
        if !images.bind(self.resources.doc()) {
            log::warn!(
                "rasterrocket-interp: the document image store serves another document — \
                 decoding this page's images without it"
            );
            return;
        }
        let page = self.image_memo.borrow().page();
        self.image_memo = image_memo::ImageMemo::shared(page, images);
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
    ///
    /// # Panics
    ///
    /// When the replacement canvas cannot be allocated.
    pub fn set_page_group(&mut self, group: Option<crate::resources::TransparencyGroupParams>) {
        self.try_set_page_group(group)
            .expect("PageRenderer::set_page_group: canvas allocation failed");
    }

    /// [`Self::set_page_group`] that reports a canvas the allocator refuses
    /// as an error.  On error the renderer is unchanged.
    ///
    /// # Errors
    ///
    /// [`raster::BitmapAllocError`] when the replacement canvas cannot be
    /// allocated.
    pub fn try_set_page_group(
        &mut self,
        group: Option<crate::resources::TransparencyGroupParams>,
    ) -> Result<(), raster::BitmapAllocError> {
        debug_assert_eq!(
            self.ops_executed, 0,
            "set_page_group must precede execute: the page canvas is replaced"
        );
        if group.is_some() != self.page_group.is_some() {
            let kind = self.canvas.kind();
            self.canvas = PageCanvas::try_new(kind, self.width, self.height, group.is_some())?;
        }
        self.page_group = group;
        self.group = self.page_group_context(group);
        Ok(())
    }

    /// A transparent canvas of `kind`, `width × height`, or `None` with the
    /// render stopped by [`RenderAbort::AllocationFailed`] when the
    /// allocator refuses it.
    fn try_canvas(&mut self, kind: CanvasKind, width: u32, height: u32) -> Option<PageCanvas> {
        #[cfg(test)]
        if self.refuses_test_canvas(width, height) {
            return None;
        }
        PageCanvas::try_new(kind, width, height, true)
            .map_err(|e| self.abort_allocation(e))
            .ok()
    }

    /// Whether the test limit on canvas-sized bitmaps (`max_canvas_px`)
    /// refuses a `width × height` one, stopping the render as a refused
    /// allocation does.
    #[cfg(test)]
    pub(super) fn refuses_test_canvas(&mut self, width: u32, height: u32) -> bool {
        let refused = u64::from(width) * u64::from(height) > self.max_canvas_px;
        if refused {
            self.abort_allocation(raster::BitmapAllocError { width, height });
        }
        refused
    }

    /// Stop the render with [`RenderAbort::AllocationFailed`] for the
    /// bitmap `e` describes, unless it has already stopped.
    fn abort_allocation(&mut self, e: raster::BitmapAllocError) {
        log::warn!("rasterrocket-interp: {e} — aborting page render");
        if self.budget_exceeded.is_none() {
            self.budget_exceeded = Some(RenderAbort::AllocationFailed {
                width: e.width,
                height: e.height,
            });
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
    /// the form nesting limit, a bitmap it needs could not be allocated, or
    /// it was cancelled.  The rendered bitmap is
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
        self.adopt_budget(parent.remaining_budget());
    }

    /// The work this renderer may still do: its operator budget less the
    /// operators it executed (zero once spent, so a renderer adopting it
    /// trips on its first operator), its deadline, its cancellation flag
    /// and why it stopped, if it did.
    pub(super) fn remaining_budget(&self) -> WorkBudget {
        WorkBudget {
            ops: self.op_budget.saturating_sub(self.ops_executed),
            deadline: self.deadline,
            cancel: self.cancel.clone(),
            exceeded: self.budget_exceeded.clone(),
        }
    }

    /// Continue the page's work watchdog from `budget`, a
    /// [`Self::remaining_budget`] of an earlier renderer of the page.
    pub(super) fn adopt_budget(&mut self, budget: WorkBudget) {
        self.op_budget = budget.ops;
        self.ops_executed = 0;
        self.deadline = budget.deadline;
        self.cancel = budget.cancel;
        self.budget_exceeded = budget.exceeded;
    }

    /// Draw image and form `XObjects`, shadings and tiling cells from
    /// `parent`'s memos, and keep this renderer's in them, so a pattern cell
    /// does not decode, parse or rasterise what the page already has.  The
    /// cell's images also spend the page's tint-transform budget.
    pub(super) fn share_memos(&mut self, parent: &Self) {
        self.image_memo = std::rc::Rc::clone(&parent.image_memo);
        self.tint_budget = std::rc::Rc::clone(&parent.tint_budget);
        self.pattern_memo = std::rc::Rc::clone(&parent.pattern_memo);
        self.form_memo = std::rc::Rc::clone(&parent.form_memo);
        self.ink_map = Rc::clone(&parent.ink_map);
    }

    /// The channels the renderer's paints wrote (see the `painted` field).
    pub(super) const fn painted_channels(&self) -> InkChannels {
        self.painted.get()
    }

    /// Record a paint writing the channels `overprint`, every channel when
    /// `None`, of a colour carried in the channels `carries`.
    pub(super) fn record_painted(&self, overprint: Option<ChannelMask>, carries: ChannelMask) {
        let paint = InkChannels {
            paints: overprint.unwrap_or(ChannelMask::ALL),
            carries,
        };
        self.painted.set(self.painted.get().union(paint));
    }

    /// Consume a child renderer and return its canvas as painted, in its
    /// own pixel type, sharing its spot slots with the parent.
    fn finish_canvas(self) -> PageCanvas {
        self.into_parts(true).0
    }

    /// A renderer of `width × height` pixels, resolving resources in
    /// `context`, that paints content drawn for this one (a tiling cell, a
    /// Type 3 glyph) onto a transparent canvas of `kind` whose alpha plane is
    /// the content's coverage.  It shares this renderer's document fonts,
    /// antialiasing, operation budget, memos and optional content.  `None`,
    /// with the render stopped, when the allocator refuses the canvas.
    fn try_child(
        &mut self,
        kind: CanvasKind,
        width: u32,
        height: u32,
        context: ObjectId,
    ) -> Option<Self> {
        let canvas = self.try_canvas(kind, width, height)?;
        let view = Self::unit_view(width, height);
        let mut child = Self::with_canvas(
            &view,
            self.resources.doc(),
            context,
            canvas,
            None,
            self.ink.clone(),
            self.space,
        );
        child.set_document_fonts(Arc::clone(self.font_cache.document_fonts()));
        child.set_antialias(self.text_antialias, self.vector_antialias);
        child.adopt_parent_budget(self);
        child.share_memos(self);
        child.set_optional_content(Arc::clone(self.optional_content()));
        child.census = text_census::Census::Uncounted;
        Some(child)
    }

    /// Fold a finished child renderer (a tiling-pattern tile or a Type 3
    /// glyph) into its parent: its watchdog consumption, so the shared
    /// per-page budget keeps accounting for work done inside sub-renders,
    /// its skipped images and forms, so they are reported on the page, and its device
    /// fill counts (zero while children are built without a device).
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
        diag.approximated_group_spaces = diag
            .approximated_group_spaces
            .union(self.diag.approximated_group_spaces);
        for failure in &self.diag.image_failures {
            if !diag.image_failures.contains(failure) {
                diag.image_failures.push(failure.clone());
            }
        }
        for &id in &self.diag.forms_too_deep {
            if !diag.forms_too_deep.contains(&id) {
                diag.forms_too_deep.push(id);
            }
        }
        for font in &self.diag.missing_fonts {
            if !diag.missing_fonts.contains(font) {
                diag.missing_fonts.push(font.clone());
            }
        }
        diag.has_images |= self.diag.has_images;
        diag.has_vector_text |= self.diag.has_vector_text;
        diag.device_image_blits = diag
            .device_image_blits
            .saturating_add(self.diag.device_image_blits);
        diag.device_icc_conversions = diag
            .device_icc_conversions
            .saturating_add(self.diag.device_icc_conversions);
        diag.device_icc_failures = diag
            .device_icc_failures
            .saturating_add(self.diag.device_icc_failures);
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

    /// Consume the renderer of 8-bit samples and return the finished
    /// bitmap, in sRGB, and page diagnostics.  An ink canvas converts with
    /// the process equivalents of the page's spot colorants, one requested
    /// as CMYK too; its CMYK finish is [`Self::try_finish_keeping`]'s.
    ///
    /// # Panics
    ///
    /// On a renderer of 16-bit samples ([`Self::try_with_page_group_in`]
    /// at [`SampleDepth::Sixteen`]), which finishes through
    /// [`Self::try_finish_keeping`].
    #[must_use]
    pub fn finish(self) -> (Bitmap<Rgb8>, PageDiagnostics) {
        let finish = self.ink_finish();
        let (canvas, _, ink_map, diag) = self.into_parts(true);
        let ink_map = page_ink_map(ink_map);
        let rgb = rgb8(canvas.into_finished(&ink_map.luts(), &finish));
        (rgb, Self::finish_diagnostics(diag, ink_map))
    }

    /// [`Self::finish`] that reports an RGB bitmap the allocator refuses for
    /// an ink canvas as an error instead of aborting.
    ///
    /// # Errors
    ///
    /// The allocator refused the page's RGB bitmap.
    ///
    /// # Panics
    ///
    /// As [`Self::finish`].
    pub fn try_finish(self) -> Result<(Bitmap<Rgb8>, PageDiagnostics), raster::BitmapAllocError> {
        self.try_finish_keeping(false)
            .map(|(rgb, diag, _)| (rgb8(rgb), diag))
    }

    /// [`Self::try_finish`] without the composite onto paper: the bitmap of
    /// a page painted as a page group keeps its alpha plane, the page's
    /// shape × opacity, under colour that is not premultiplied by it
    /// (colour and alpha are 0 where nothing was painted).  A page painted
    /// without a page group has no alpha plane and is opaque.
    ///
    /// # Errors
    ///
    /// The allocator refused the page's RGB bitmap.
    ///
    /// # Panics
    ///
    /// As [`Self::finish`].
    pub fn try_finish_transparent(
        self,
    ) -> Result<(Bitmap<Rgb8>, PageDiagnostics), raster::BitmapAllocError> {
        self.try_finish_keeping(true)
            .map(|(rgb, diag, _)| (rgb8(rgb), diag))
    }

    /// [`Self::try_finish_transparent`] when `transparent`, and
    /// [`Self::try_finish`] otherwise, at the renderer's depth, also
    /// returning the storage the bitmap does not hold (see
    /// [`CanvasBuffers`]).
    ///
    /// # Errors
    ///
    /// The allocator refused the page's RGB bitmap.
    pub fn try_finish_keeping(
        self,
        transparent: bool,
    ) -> Result<(FinishedCanvas, PageDiagnostics, CanvasBuffers), raster::BitmapAllocError> {
        let finish = self.ink_finish();
        let (canvas, mut buffers, ink_map, diag) = self.into_parts(!transparent);
        let ink_map = page_ink_map(ink_map);
        let mut finished = canvas.try_into_finished_in(&ink_map.luts(), &finish, &mut buffers)?;
        if transparent {
            finished.zero_transparent_colour();
        }
        Ok((finished, Self::finish_diagnostics(diag, ink_map), buffers))
    }

    /// The finished canvas, with the storage not in it, the ink map, shared
    /// with the parent in a child renderer, and the diagnostics; a page
    /// group is composited onto paper when `onto_paper`, its alpha plane
    /// joining the storage.
    fn into_parts(
        mut self,
        onto_paper: bool,
    ) -> (PageCanvas, CanvasBuffers, SharedInkMap, PageDiagnostics) {
        self.diag.dominant_filter = crate::resources::image::dominant_filter(&self.filter_counts);
        if onto_paper && self.page_group.is_some() {
            self.canvas.composite_onto_white(&mut self.buffers);
        }
        let Self {
            canvas,
            buffers,
            ink_map,
            diag,
            ..
        } = self;
        (canvas, buffers, ink_map, diag)
    }

    /// `diag` with the spot colorants `ink_map` could not slot.
    fn finish_diagnostics(mut diag: PageDiagnostics, ink_map: ink_map::InkMap) -> PageDiagnostics {
        diag.unslotted_spots = ink_map.into_unslotted();
        diag
    }

    /// Execute a slice of decoded operators in order.
    ///
    /// Stops early and sets [`budget_exceeded`](Self::budget_status) if the
    /// page exceeds the operator-count cap (`MAX_PAGE_OPS`) or the wall-clock
    /// deadline (`PAGE_RENDER_DEADLINE_SECS`).  The deadline is checked once
    /// every `DEADLINE_CHECK_INTERVAL` ops to amortise `Instant::now()` cost.
    pub fn execute(&mut self, ops: &[Operator]) {
        for (i, op) in ops.iter().enumerate() {
            // Bail early if a previous call (e.g. a recursive form XObject)
            // already tripped the budget.
            if self.budget_exceeded.is_some() {
                return;
            }
            #[cfg(feature = "vulkan")]
            if self.device_refused() {
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
            self.execute_one(op, &ops[i + 1..]);
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
    /// unchanged.  Shaping context ends with the stream: the strings shown
    /// in it are no neighbours of those shown after it.
    fn execute_content_stream(&mut self, ops: &[Operator]) {
        let scope = self.gstate.enter_stream();
        let enclosing = std::mem::take(&mut self.marked_content);
        self.execute(ops);
        self.text_tail = None;
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
        self.text_tail = None;
        self.marked_content = operators::MarkedContent::default();
        #[cfg(feature = "vulkan")]
        if self.device_refused() {
            return;
        }
        self.trace_line("annotations");
        annotations::render_annotations(self, page_id);
    }
}

#[cfg(test)]
mod tests;
