//! Per-page operator dispatcher.
//!
//! [`PageRenderer`] holds a target [`Bitmap`] and a [`GStateStack`], iterates
//! over a decoded operator sequence, and calls into the `raster` crate for
//! each painting operator.
//!
//! Sub-modules (all private; named for orientation only):
//! - `operators` — `execute_one` operator dispatch table.
//! - `text` — `show_text` and `show_text_type3` rendering.
//! - `patterns` — `resolve_pattern`, `resolve_fill_pattern` and `render_pattern_tile`.
//! - `shading_paint` — the `sh` operator and shading-pattern fills and strokes.
//! - `text_ops` — `GlyphRecord` and `text_to_device` helpers.
//! - `annotations` — annotation appearance stream rendering.
//! - `gpu_ops` — GPU-accelerated fill dispatch.
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
//! - Type 0 / `CIDFont` composite fonts — `Encoding` `CMap` (including
//!   embedded stream `CMaps` and `Identity-H`/`V`), `CIDToGIDMap`, `DW`/`W`
//!   advance widths, and multi-byte character code iteration
//!
//! - Text render modes 4–7 — fill/stroke painting combined with text-as-clip
//!   (`glyph_path` outlines accumulated into the clip region per PDF §9.3.6)
//! - `PatternType` 1 tiling patterns and `PatternType` 2 shading patterns via `scn`/`SCN`
//! - Type 3 paint-procedure fonts — `CharProc` content streams, `d0`/`d1` metrics
//!
//! - Optional content (layers) — `BDC /OC` sequences and `XObjects` and
//!   annotations carrying `/OC` are skipped when hidden under the default
//!   configuration (see [`crate::optional_content`])
//! - Annotation rendering — `AP/N` normal appearance streams blitted after page content
//! - Transparency groups — Form `XObjects` with `Group /S /Transparency` rendered into an
//!   intermediate bitmap via `raster::transparency` and composited back (PDF §11.6.6)
//!
//! # Not yet implemented
//!
//! - `ExtGState` blend mode (`BM`) — only `Normal` is currently mapped
//! - Type 0 named `CMaps` other than `Identity-H`/`V` (e.g. `/GB-EUC-H`)

mod annotation_synthesis;
mod annotations;
mod fill_stroke;
mod gpu_ops;
mod image_memo;
mod image_reduce;
mod image_sampler;
mod operators;
mod pattern_memo;
mod patterns;
mod pixel_rect;
mod shading_paint;
mod soft_mask;
mod text;
mod text_ops;
#[cfg(feature = "vulkan")]
mod vk_ops;
mod widget_appearance;

use patterns::PatternPaint;
use pdf::{ContentFailure, Document, ObjectId};

use color::Rgb8;
use color::convert::div255;
use raster::{
    Bitmap, ClipResult, eo_fill, fill,
    path::PathBuilder,
    pipe::{Pattern, PipeSrc, PipeState},
    state::TransferSet,
    stroke::{StrokeParams, stroke},
    transparency::{Backdrop, GroupParams, begin_group, paint_group},
    types::{BlendMode, LineCap, LineJoin},
    xpath::XPath,
};

use super::color::RasterColor;
use super::font_cache::{DocumentFonts, FontCache};
use super::gstate::{Ctm, GStateStack, ctm_multiply, ctm_transform, invert_affine};
use crate::content::Operator;
use crate::resources::SoftMask;
use crate::resources::{
    ColorSpace, IMAGE_FILTER_COUNT, ImageColorSpace, ImageFilter, PageResources,
};
#[cfg(any(feature = "gpu-aa", feature = "gpu-icc", feature = "cache"))]
use gpu::GpuCtx;
#[cfg(feature = "vaapi")]
use gpu::JpegQueueHandle;
#[cfg(feature = "gpu-jpeg-huffman")]
use gpu::backend::cuda::CudaBackend;
#[cfg(feature = "vulkan")]
use gpu::backend::vulkan::VulkanBackend;
#[cfg(feature = "cache")]
use gpu::cache::{DeviceImageCache, DevicePageBuffer, DocId};
#[cfg(feature = "gpu-jpeg-huffman")]
use gpu::jpeg_decoder::JpegGpuDecoder;
#[cfg(feature = "nvjpeg")]
use gpu::nvjpeg::NvJpegDecoder;
#[cfg(feature = "nvjpeg2k")]
use gpu::nvjpeg2k::NvJpeg2kDecoder;
use image_sampler::{Filter, GridClip, MaskSampler, SampleGrid};
use pixel_rect::PixelRect;
use std::sync::Arc;

const _: () = assert!(
    IMAGE_FILTER_COUNT == 6,
    "ImageFilter variant count changed — update filter_counts array size and FILTERS in finish()"
);

/// Lightweight page metadata collected at zero extra cost during rendering.
///
/// Returned alongside the finished bitmap from [`PageRenderer::finish`].
/// Callers can use this to route pages to different OCR configurations without
/// a separate post-hoc analysis pass.
#[derive(Debug, Clone, Default)]
pub struct PageDiagnostics {
    /// `true` when at least one image `XObject` or inline image was blitted.
    pub has_images: bool,
    /// `true` when at least one text-showing operator (`Tj`, `TJ`, `'`, `"`)
    /// was executed with a non-empty glyph sequence.  `false` on scan-only pages.
    pub has_vector_text: bool,
    /// The image filter seen most often on this page.
    ///
    /// `None` when no images were decoded (e.g. a pure-vector page).  When
    /// multiple filters are equally frequent the last one in [`ImageFilter`]
    /// discriminant order wins (Rust's `max_by_key` is last-wins on ties) —
    /// stable across identical pages.
    pub dominant_filter: Option<ImageFilter>,
    /// Estimated source pixels-per-inch of the dominant image, if any.
    ///
    /// Computed as `(image_width_px / page_width_pts) × 72`.  Useful as a hint
    /// for DPI auto-selection: rendering at the source PPI avoids upsampling a
    /// 67-PPI scan at 300 DPI (4.5× upscale with no information gain).
    ///
    /// `None` when no images were blitted or page geometry is unavailable.
    pub source_ppi_hint: Option<f32>,
    /// Images that were present but could not be drawn.  Each one was skipped
    /// and the page rendered without it, so the region it covers shows
    /// whatever was painted beneath.  Empty on a complete render.
    pub image_failures: Vec<ImageFailure>,
    /// Page content streams that could not be decoded.  Each one was
    /// skipped and the page rendered from the others, so whatever it would
    /// have drawn is missing.  Empty on a complete render.
    pub content_failures: Vec<ContentFailure>,
    /// Font resource names (without the leading `/`) that `Tf` selected but
    /// the `/Font` resources in effect do not define.  Their text is drawn
    /// in Helvetica.  Empty when every font resolved.
    pub missing_fonts: Vec<String>,
    /// The document's interactive form type.  [`FormType::XfaFull`] marks
    /// a dynamic XFA form, whose rendered pages are the placeholder content
    /// stored in the file rather than the form.
    ///
    /// [`FormType::XfaFull`]: crate::FormType::XfaFull
    pub form_type: crate::FormType,
}

/// An image the renderer skipped because it could not be decoded or drawn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageFailure {
    /// `XObject` resource name without the leading `/` (e.g. `Im0`); `None`
    /// for an inline image.
    pub name: Option<String>,
    /// Codec or blit error, prefixed by the failing filter where known
    /// (e.g. `FlateDecode: invalid distance`).
    pub reason: String,
}

impl std::fmt::Display for ImageFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.name {
            Some(name) => write!(f, "image /{name}: {}", self.reason),
            None => write!(f, "inline image: {}", self.reason),
        }
    }
}

impl PageDiagnostics {
    /// Suggest a render DPI based on the source image resolution embedded in the PDF.
    ///
    /// Returns the source PPI rounded up to the nearest standard DPI step
    /// (72 / 96 / 150 / 200 / 300 / 400 / 600), clamped to `[min_dpi, max_dpi]`.
    ///
    /// Returns `None` when [`source_ppi_hint`](Self::source_ppi_hint) is absent
    /// (no images on the page), in which case the caller should use their default DPI.
    ///
    /// ## Rationale
    ///
    /// Many scanned PDFs store images at 62–67 PPI.  Rendering at 300 DPI upsamples
    /// 4.5× with no additional information.  Passing the suggested DPI to
    /// the renderer (`rasterrocket::RasterOptions::dpi`) renders at the minimum resolution that avoids
    /// upsampling, bounded by the caller's quality floor (`min_dpi`) and ceiling
    /// (`max_dpi`).
    ///
    /// Typical call: `diag.suggested_dpi(150.0, 300.0)` — never render below
    /// 150 DPI (Tesseract minimum for acceptable accuracy), never above 300 DPI.
    #[must_use]
    pub fn suggested_dpi(&self, min_dpi: f32, max_dpi: f32) -> Option<f32> {
        // Standard DPI steps in ascending order.  Any PPI above the last step
        // falls back to the last step (600), which is then clamped by max_dpi.
        const STEPS: [f32; 7] = [72.0, 96.0, 150.0, 200.0, 300.0, 400.0, 600.0];
        const STEPS_MAX: f32 = STEPS[STEPS.len() - 1]; // 600.0 — infallible const index
        let ppi = self.source_ppi_hint?;
        let stepped = STEPS
            .iter()
            .copied()
            .find(|&s| s >= ppi)
            .unwrap_or(STEPS_MAX);
        Some(stepped.clamp(min_dpi, max_dpi))
    }
}

/// Identity CTM for passing to raster functions — coordinate transform is
/// already baked into the path points by `to_device`.
const DEVICE_MATRIX: [f64; 6] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

/// `path` with every point mapped through `m`.  An affine map carries Bézier
/// control points to the control points of the mapped curve, so flags and
/// stroke-adjust hints carry over unchanged.
fn transform_path(path: &raster::path::Path, m: &[f64; 6]) -> raster::path::Path {
    raster::path::Path {
        pts: path
            .pts
            .iter()
            .map(|p| raster::xpath::transform(m, p.x, p.y))
            .collect(),
        ..path.clone()
    }
}

/// Device-pixel extent `(width, height)` of an image drawn under `ctm`: the
/// lengths of the images of its unit x and y vectors, rounded up.  Correct
/// under rotation and skew.  `None` for a degenerate or non-finite matrix.
fn image_draw_size(ctm: &[f64; 6]) -> Option<(u32, u32)> {
    let extent = |dx: f64, dy: f64| {
        let len = dx.hypot(dy).ceil();
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "len is finite, at least 1 and at most u32::MAX, checked first"
        )]
        (len.is_finite() && (1.0..=f64::from(u32::MAX)).contains(&len)).then_some(len as u32)
    };
    Some((extent(ctm[0], ctm[1])?, extent(ctm[2], ctm[3])?))
}

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
    /// Page width in device pixels.
    width: u32,
    /// Page height in device pixels.
    height: u32,
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
    /// Font face cache for this page.
    font_cache: FontCache,
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
    /// The matrix from the default coordinate space of the content stream
    /// being executed to the device (PDF §8.7.3.1): the page's initial CTM,
    /// a form's CTM once its `Matrix` is applied, or a pattern tile's CTM.
    /// A shading pattern's `Matrix` is concatenated onto it.
    pub(super) pattern_ctm: Ctm,
    /// The backdrop of the non-isolated transparency group being painted
    /// into, consulted by blend modes applied inside it; `None` outside one.
    group_backdrop: Option<Arc<Backdrop>>,
    /// Transparent page-sized bitmaps for transparency groups, each
    /// returned cleared when its group is composited; one per nesting level
    /// is allocated over the page's life.
    group_pool: Vec<Bitmap<Rgb8>>,
    /// Accumulated page diagnostics, populated during rendering.
    diag: PageDiagnostics,
    /// Resource name of the image `XObject` being drawn, set for the duration
    /// of its blit so a blit failure is attributed to it; `None` while an
    /// inline image is drawn.
    current_image: Option<String>,
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
    /// Set to `Some(reason)` when the per-page budget is exceeded; `None` means ok.
    /// Checked after `execute`; surfaces as `Err` in the per-page result.
    budget_exceeded: Option<String>,
    /// Distinct unknown operator keywords already warned about this page,
    /// so a content stream that repeats an unsupported op doesn't flood the
    /// log.  Bounded at [`MAX_WARNED_UNKNOWN_OPS`] entries.
    warned_unknown_ops: std::collections::HashSet<Vec<u8>>,
    /// Set once the distinct-keyword cap is reached, so the "no longer
    /// individually reporting" summary warning is emitted exactly once.
    warned_unknown_ops_capped: bool,
    /// Per-filter blit counts used to compute `diag.dominant_filter`.
    filter_counts: [u32; IMAGE_FILTER_COUNT],
    /// Open marked-content sequences of the content stream being executed,
    /// innermost last.
    ///
    /// `BMC` and `BDC` push one entry each and `EMC` pops one: `false` for an
    /// optional-content sequence whose group is hidden, `true` otherwise.
    /// Content is skipped whenever any entry is `false`.  Each content
    /// stream starts with an empty stack (see
    /// [`execute_content_stream`](Self::execute_content_stream)).
    ocg_stack: Vec<bool>,
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
    /// Per-page cache of baked ICC CMYK→RGB CLUT tables.  Keyed by a hash of the
    /// raw ICC profile bytes so that repeated images sharing the same profile (the
    /// common case in press PDFs) only pay the bake cost once per page.
    #[cfg(feature = "gpu-icc")]
    icc_clut_cache: crate::resources::image::IccClutCache,
    /// Device-image-cache state.  `Some` only when the `cache` feature is
    /// on AND a cache was wired in via [`Self::set_image_cache`];
    /// `None` keeps the renderer fully CPU-resident.
    #[cfg(feature = "cache")]
    cache_state: Option<CacheState>,
}

/// Renderer-side device-image-cache state bundled together so the "cache
/// wired in" condition is a single `Option<CacheState>` rather than three
/// implicit invariants on separate fields.
///
/// Without this bundling, `doc_id: DocId` had to default to
/// `DocId([0u8; 32])` — a valid hash that would collide with anyone
/// using the all-zeros doc id, a sharp foot-gun.
#[cfg(feature = "cache")]
struct CacheState {
    /// Device-resident image cache.  Shared across pages so
    /// content-hash dedup spans the whole render session.
    cache: Arc<DeviceImageCache>,
    /// Stable identifier for the source PDF — combined with the
    /// per-image PDF object id, forms the cache's `(DocId, ObjId)`
    /// alias key for fast same-document lookups.
    doc_id: DocId,
    /// Lazy-allocated device-resident page buffer.  `None` until the
    /// first GPU image blit allocates it.  Between blits every pixel is
    /// zero (alpha 0): each blit composites its rows onto `bitmap` and
    /// clears them before returning.
    page_buffer: Option<DevicePageBuffer>,
}

/// Build the initial page CTM that maps PDF user space to device pixels.
///
/// `scale = dpi / 72`.  `rotate_cw` is the page `/Rotate`, which callers must
/// pass already normalised to one of {0, 90, 180, 270} (the only values ISO
/// 32000-2 §14.11.6 defines; `page_size_pts_by_id` snaps `/Rotate` to this set
/// upstream).  Any other value is a caller contract violation: it trips a
/// `debug_assert` and, in release, renders unrotated rather than silently
/// emitting a wrong-but-plausible rotation.  `origin_x`/`origin_y` are the
/// selected page box's lower-left corner in PDF user space (ISO 32000-2
/// §14.11.2); zero for the common box-at-origin case.
///
/// The `to_device` helper applies an additional Y-flip (`height_px - dy`), so
/// these matrices only carry scale + rotation + the box-origin pre-translation;
/// the flip is not folded in here.  `w`/`h` are the output bitmap dimensions in
/// points (`width_px / scale`, `height_px / scale`); for 90°/270° the caller has
/// already swapped the pixel dimensions, so here `w` is the post-rotation width.
///
/// PDF user-space content coordinates are absolute: a content point at PDF
/// `(X, Y)` with `X ∈ [llx, llx+boxW]`, `Y ∈ [lly, lly+boxH]`.  We want the box
/// lower-left `(llx, lly)` to land at device origin, so each branch is the
/// box-at-origin matrix with the substitution `X → X-llx`, `Y → Y-lly` — i.e. an
/// innermost pre-translation `T(-llx, -lly)` composed under scale+rotate.  Per
/// branch (`s = scale`):
///   Rotate 0   → `[ s,  0,  0,  s,     -llx·s,     -lly·s]`
///   Rotate 90  → `[ 0, -s,  s,  0,     -lly·s,  (h+llx)·s]`
///   Rotate 180 → `[-s,  0,  0, -s,  (w+llx)·s,  (h+lly)·s]`
///   Rotate 270 → `[ 0,  s, -s,  0,  (w+lly)·s,     -llx·s]`
/// When `llx = lly = 0` every branch reduces to the box-at-origin value, so
/// origin-at-zero PDFs (the vast majority) are unchanged in scale.
///
/// The `initial_ctm_rotate{0,90,180,270}_maps_box_corners_to_device_pixels`
/// tests pin each box corner to its required device pixel for every rotation,
/// so a sign error in any branch fails a test rather than shipping a mirrored
/// or upside-down render.
///
/// Degenerate inputs cannot reach here as silent garbage: `scale` is asserted
/// finite-positive by the sole caller, and `origin_x`/`origin_y` originate from
/// `page_size_pts_by_id`, which rejects non-finite / zero-area boxes upstream
/// and logs malformed-box recovery.  This function therefore performs only the
/// arithmetic; it does not re-validate.
fn build_initial_ctm(
    width: u32,
    height: u32,
    scale: f64,
    rotate_cw: u16,
    origin_x: f64,
    origin_y: f64,
) -> Ctm {
    let s = scale;
    let llx = origin_x;
    let lly = origin_y;
    let w = f64::from(width) / s;
    let h = f64::from(height) / s;
    match rotate_cw % 360 {
        // Rotate 0: standard scale + y-flip handled by to_device.
        0 => [s, 0.0, 0.0, s, -llx * s, -lly * s],
        // Rotate 90 CW: swap axes. Bitmap width = original H, height = original W.
        90 => [0.0, -s, s, 0.0, -lly * s, (h + llx) * s],
        // Rotate 180: flip both axes.
        180 => [-s, 0.0, 0.0, -s, (w + llx) * s, (h + lly) * s],
        // Rotate 270 CW (= 90 CCW): swap axes, opposite orientation.
        270 => [0.0, s, -s, 0.0, (w + lly) * s, -llx * s],
        // Non-spec value: a caller contract violation (callers normalise to
        // {0,90,180,270} upstream).  Fail loudly in debug; in release fall back
        // to the unrotated matrix so a stray value renders upright rather than
        // silently rotated-and-wrong.
        other => {
            debug_assert!(
                false,
                "build_initial_ctm: rotate_cw must be normalised to {{0,90,180,270}}, got {other}"
            );
            [s, 0.0, 0.0, s, -llx * s, -lly * s]
        }
    }
}

impl<'doc> PageRenderer<'doc> {
    /// Create a renderer for a blank white page of `width × height` pixels,
    /// where 1 user-space unit = 1 device pixel (72 dpi).
    ///
    /// `doc` and `page_id` are used to resolve font and resource dictionaries.
    #[must_use]
    pub fn new(width: u32, height: u32, doc: &'doc Document, page_id: ObjectId) -> Self {
        Self::new_scaled(width, height, 1.0, 0, 0.0, 0.0, doc, page_id)
    }

    /// Create a renderer with an initial uniform scale in the CTM.
    ///
    /// `scale = dpi / 72.0` maps PDF points to device pixels.  `rotate_cw` is
    /// the page `/Rotate` value (one of 0, 90, 180, 270); the bitmap dimensions
    /// must already be swapped for 90°/270° before calling this method.
    /// `origin_x`/`origin_y` are the lower-left corner of the selected page box
    /// in PDF user space (from `PageGeometry::origin_x`/`origin_y`); zero for
    /// the common case where the box starts at the origin.
    ///
    /// # Panics
    ///
    /// Panics if `scale` is not a positive finite number.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "renderer configuration has no natural grouping; a builder would add more boilerplate than it removes"
    )]
    pub fn new_scaled(
        width: u32,
        height: u32,
        scale: f64,
        rotate_cw: u16,
        origin_x: f64,
        origin_y: f64,
        doc: &'doc Document,
        page_id: ObjectId,
    ) -> Self {
        assert!(
            scale.is_finite() && scale > 0.0,
            "PageRenderer::new_scaled: scale must be a positive finite number, got {scale}"
        );

        let mut bitmap = Bitmap::<Rgb8>::new(width, height, 1, false);
        bitmap.data_mut().fill(255u8); // white background

        let mut gstate = GStateStack::new(width, height);
        let initial_ctm = build_initial_ctm(width, height, scale, rotate_cw, origin_x, origin_y);
        gstate.current_mut().ctm = initial_ctm;

        let font_cache = FontCache::new(Arc::default());
        let resources = PageResources::new(doc, page_id);

        Self {
            bitmap,
            width,
            height,
            gstate,
            path: None,
            pending_clip: None,
            font_cache,
            resources,
            form_depth: 0,
            active_forms: Vec::new(),
            rotate_cw,
            pattern_depth: 0,
            pattern_ctm: initial_ctm,
            group_backdrop: None,
            group_pool: Vec::new(),
            diag: PageDiagnostics::default(),
            current_image: None,
            image_memo: image_memo::ImageMemo::shared(),
            pattern_memo: pattern_memo::PatternMemo::shared(),
            op_budget: MAX_PAGE_OPS,
            deadline: Some(
                std::time::Instant::now()
                    + std::time::Duration::from_secs(PAGE_RENDER_DEADLINE_SECS),
            ),
            ops_executed: 0,
            budget_exceeded: None,
            warned_unknown_ops: std::collections::HashSet::new(),
            warned_unknown_ops_capped: false,
            filter_counts: [0u32; IMAGE_FILTER_COUNT],
            ocg_stack: Vec::new(),
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
            #[cfg(feature = "gpu-icc")]
            icc_clut_cache: crate::resources::image::IccClutCache::new(),
            #[cfg(feature = "cache")]
            cache_state: None,
        }
    }

    /// Attach a GPU parallel-Huffman JPEG decoder to this renderer.
    #[cfg(feature = "gpu-jpeg-huffman")]
    pub fn set_jpeg_gpu(&mut self, dec: Option<JpegGpuDecoder<CudaBackend>>) {
        self.jpeg_gpu = dec;
    }

    /// Detach and return the GPU parallel-Huffman JPEG decoder for reuse.
    #[cfg(feature = "gpu-jpeg-huffman")]
    pub const fn take_jpeg_gpu(&mut self) -> Option<JpegGpuDecoder<CudaBackend>> {
        self.jpeg_gpu.take()
    }

    /// Attach a Vulkan parallel-Huffman JPEG decoder to this renderer.
    ///
    /// Mutually exclusive with [`set_jpeg_gpu`] — call one or the other per
    /// render, never both.
    #[cfg(all(feature = "gpu-jpeg-huffman", feature = "vulkan"))]
    pub fn set_jpeg_vk(&mut self, dec: Option<JpegGpuDecoder<VulkanBackend>>) {
        self.jpeg_vk = dec;
    }

    /// Detach and return the Vulkan parallel-Huffman JPEG decoder for reuse.
    #[cfg(all(feature = "gpu-jpeg-huffman", feature = "vulkan"))]
    pub const fn take_jpeg_vk(&mut self) -> Option<JpegGpuDecoder<VulkanBackend>> {
        self.jpeg_vk.take()
    }

    /// [`crate::resources::image::GPU_JPEG_THRESHOLD_PX`] are decoded on the
    /// GPU via nvJPEG rather than the CPU JPEG decoder.
    ///
    /// Calling this with `None` detaches any existing decoder (reverts to CPU).
    #[cfg(feature = "nvjpeg")]
    pub fn set_nvjpeg(&mut self, dec: Option<NvJpegDecoder>) {
        self.nvjpeg = dec;
    }

    /// Attach a VA-API JPEG decode queue handle to this renderer.
    ///
    /// The handle is a cheaply-cloneable sender to a dedicated OS worker thread
    /// that owns the `VapiJpegDecoder`.  `DCTDecode` baseline JPEG streams with
    /// pixel area ≥ [`crate::resources::image::GPU_JPEG_THRESHOLD_PX`] are
    /// submitted to the queue instead of being decoded on the calling thread,
    /// eliminating Mesa driver contention across Rayon workers.
    ///
    /// The handle is dropped when the renderer drops — no explicit reclaim step
    /// is needed because the queue worker lives in `RasterSession`, which always
    /// outlives the renderer.
    #[cfg(feature = "vaapi")]
    pub fn set_vaapi_queue(&mut self, handle: JpegQueueHandle) {
        self.vaapi_jpeg_queue = Some(handle);
    }

    /// Attach a GPU JPEG 2000 decoder to this renderer.
    ///
    /// When set, `JPXDecode` image streams with pixel area ≥
    /// [`crate::resources::image::GPU_JPEG2K_THRESHOLD_PX`] are decoded on the
    /// GPU via nvJPEG2000 rather than the CPU JPEG 2000 decoder.
    ///
    /// Call with `None` to revert to CPU-only JPEG 2000 decode.
    #[cfg(feature = "nvjpeg2k")]
    pub fn set_nvjpeg2k(&mut self, dec: Option<NvJpeg2kDecoder>) {
        self.nvjpeg2k = dec;
    }

    /// Detach and return the nvJPEG decoder so the caller can reuse it.
    ///
    /// Returns `None` if no decoder was attached (e.g. GPU init failed).
    /// Used by the CLI to return the decoder to its thread-local slot
    /// after each page render so it survives across pages.
    #[cfg(feature = "nvjpeg")]
    pub const fn take_nvjpeg(&mut self) -> Option<NvJpegDecoder> {
        self.nvjpeg.take()
    }

    /// Detach and return the nvJPEG2000 decoder so the caller can reuse it.
    ///
    /// Returns `None` if no decoder was attached (e.g. GPU init failed).
    /// Used by the CLI to return the decoder to its thread-local slot
    /// after each page render so it survives across pages.
    #[cfg(feature = "nvjpeg2k")]
    pub const fn take_nvjpeg2k(&mut self) -> Option<NvJpeg2kDecoder> {
        self.nvjpeg2k.take()
    }

    /// Attach a GPU context for antialiased fill dispatch.
    ///
    /// When set, solid-colour fills under a rectangular clip whose pixel
    /// bounding-box area reaches the dispatch threshold (dormant by default;
    /// see [`gpu::GPU_AA_FILL_THRESHOLD`]) have their
    /// coverage evaluated on the GPU.  The device walks the same 4× scanline
    /// intersection table the CPU fill builds and the counts are composited
    /// through the CPU pipeline, so the rendered bytes are identical to the
    /// CPU-only render.
    ///
    /// Call with `None` to revert to CPU-only AA.
    #[cfg(any(feature = "gpu-aa", feature = "gpu-icc", feature = "cache"))]
    pub fn set_gpu_ctx(&mut self, ctx: Option<Arc<GpuCtx>>) {
        self.gpu_ctx = ctx;
    }

    /// Attach a Vulkan compute backend for kernel dispatch.
    ///
    /// When set, GPU-eligible AA fill and ICC CMYK→RGB calls
    /// are routed through the [`gpu::backend::GpuBackend`] trait surface
    /// (`alloc → upload → record_* → submit → wait → download`) on the
    /// Vulkan recorder rather than the CUDA `GpuCtx`.  The CUDA path is
    /// not removed; it remains live for sessions opened with
    /// `BackendPolicy::Auto` / `BackendPolicy::ForceCuda` (defined in `rasterrocket`).
    ///
    /// The device-resident image cache (`DeviceImageCache`,
    /// `DevicePageBuffer`) is CUDA-only — under Vulkan the renderer
    /// runs uncached (CPU-resident image decode + blit).
    #[cfg(feature = "vulkan")]
    pub fn set_vk_backend(&mut self, backend: Option<Arc<VulkanBackend>>) {
        self.vk_backend = backend;
    }

    /// Attach a device-resident image cache.
    ///
    /// When set, JPEG image decode goes through the cache: a content-
    /// hash hit returns an `Arc<CachedDeviceImage>` with no decode
    /// work; a miss decodes on CPU, uploads to VRAM, caches, and
    /// returns the same handle.  The renderer dispatches `Gpu`-variant
    /// images to a CUDA blit kernel that writes into a per-page
    /// `DevicePageBuffer` and composites the result onto `bitmap` in
    /// content-stream order, so the page renders byte-identically to
    /// the CPU-only path.
    ///
    /// `doc_id` should be a stable identifier for the source PDF
    /// (typically `BLAKE3(pdf_bytes)` or similar).  It's combined
    /// with the per-image PDF object number to form the cache's
    /// secondary alias key for fast same-document lookups.
    ///
    /// Call with `None` to revert to CPU-only image decode.
    ///
    /// **Caller contract:** call before any rendering operators
    /// execute on this renderer; today no caller wires or unwires a
    /// cache mid-page.
    #[cfg(feature = "cache")]
    pub fn set_image_cache(&mut self, cache: Option<Arc<DeviceImageCache>>, doc_id: DocId) {
        self.cache_state = cache.map(|cache| CacheState {
            cache,
            doc_id,
            page_buffer: None,
        });
    }

    /// Resolve indirect fonts through `fonts`, a store shared by every page
    /// of this renderer's document, instead of a store private to the page.
    /// Call before execute.
    pub fn set_document_fonts(&mut self, fonts: Arc<DocumentFonts>) {
        self.font_cache = FontCache::new(fonts);
    }

    /// Draw optional content as `optional_content` configures it, and
    /// annotations for its purpose.  Call before execute.
    pub fn set_optional_content(&mut self, optional_content: Arc<crate::OptionalContent>) {
        self.optional_content = std::cell::OnceCell::from(optional_content);
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

    /// Images skipped so far on this page; also returned in
    /// [`PageDiagnostics::image_failures`] by [`finish`](Self::finish).
    #[must_use]
    pub fn image_failures(&self) -> &[ImageFailure] {
        &self.diag.image_failures
    }

    /// Content streams skipped on this page; also returned in
    /// [`PageDiagnostics::content_failures`] by [`finish`](Self::finish).
    #[must_use]
    pub fn content_failures(&self) -> &[ContentFailure] {
        &self.diag.content_failures
    }

    /// Record the page content streams that were skipped before execution.
    pub fn record_content_failures(&mut self, failures: Vec<ContentFailure>) {
        self.diag.content_failures.extend(failures);
    }

    /// Record an image that could not be drawn.  The image is skipped and the
    /// page render continues.
    pub(super) fn record_image_failure(&mut self, name: Option<String>, reason: String) {
        let failure = ImageFailure { name, reason };
        log::debug!("rasterrocket-interp: skipped {failure}");
        self.diag.image_failures.push(failure);
    }

    /// Record a font resource `name` that `Tf` selected but the resources in
    /// effect do not define, once per page.
    pub(super) fn record_missing_font(&mut self, name: &[u8]) {
        let name = String::from_utf8_lossy(name);
        if self.diag.missing_fonts.iter().any(|n| *n == name) {
            return;
        }
        log::warn!("rasterrocket-interp: font resource /{name} is not defined — drawing Helvetica");
        self.diag.missing_fonts.push(name.into_owned());
    }

    /// The per-page work-budget breach reason, if any.
    ///
    /// `Some(reason)` means the page was aborted early because it exceeded
    /// either the operator count cap or the wall-clock deadline.  The rendered
    /// bitmap is partial and the per-page `Result` must be an `Err`.
    ///
    /// Check this after [`execute`](Self::execute) and before
    /// [`finish`](Self::finish).
    #[must_use]
    pub fn budget_status(&self) -> Option<&str> {
        self.budget_exceeded.as_deref()
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
    pub(super) fn fold_child_into(&self, parent: &mut Self) {
        parent.ops_executed = parent.ops_executed.saturating_add(self.ops_executed);
        if parent.budget_exceeded.is_none() && self.budget_exceeded.is_some() {
            parent.budget_exceeded.clone_from(&self.budget_exceeded);
        }
        parent
            .diag
            .image_failures
            .extend(self.diag.image_failures.iter().cloned());
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
        // Resolve dominant_filter from per-filter blit counts.
        // Order matches the ImageFilter discriminants (Dct=0, Jpx=1, CcittFax=2, Jbig2=3, Flate=4, Raw=5).
        const FILTERS: [ImageFilter; 6] = [
            ImageFilter::Dct,
            ImageFilter::Jpx,
            ImageFilter::CcittFax,
            ImageFilter::Jbig2,
            ImageFilter::Flate,
            ImageFilter::Raw,
        ];
        self.diag.dominant_filter = self
            .filter_counts
            .iter()
            .zip(FILTERS.iter())
            .filter(|(count, _)| **count > 0)
            .max_by_key(|(count, _)| *count)
            .map(|(_, filter)| *filter);

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
            self.execute_one(op);
            // saturating_add: the op cap fires at MAX_PAGE_OPS long before u64
            // could wrap, but a test or future caller may set a huge budget
            // with the deadline disabled — never silently wrap to a low count
            // (which would mask, not catch, an unbounded stream).
            self.ops_executed = self.ops_executed.saturating_add(1);
            if self.ops_executed > self.op_budget {
                self.budget_exceeded = Some(format!(
                    "page render exceeded operator budget ({} ops); aborting",
                    self.ops_executed,
                ));
                return;
            }
            // Check the wall-clock deadline periodically to keep the syscall cost low.
            if self.ops_executed.is_multiple_of(DEADLINE_CHECK_INTERVAL)
                && self
                    .deadline
                    .is_some_and(|dl| std::time::Instant::now() >= dl)
            {
                self.budget_exceeded = Some(format!(
                    "page render exceeded {PAGE_RENDER_DEADLINE_SECS}s deadline \
                     after {} ops; aborting",
                    self.ops_executed,
                ));
                return;
            }
        }
    }

    /// Execute `ops` as a content stream nested in the current one: a form
    /// `XObject` or a Type 3 glyph description.
    ///
    /// Marked-content sequences cannot span content streams (PDF §14.6), so
    /// the nested stream starts with no open sequence and the enclosing
    /// stream's sequences are restored afterwards, whatever the nested
    /// stream left open or closed.  A nested stream is executed only while
    /// every enclosing sequence is visible, so the visibility it inherits is
    /// unchanged.
    fn execute_content_stream(&mut self, ops: &[Operator]) {
        let enclosing = std::mem::take(&mut self.ocg_stack);
        self.execute(ops);
        self.ocg_stack = enclosing;
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
        annotations::render_annotations(self, page_id);
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn set_fill(&mut self, c: RasterColor) {
        let gs = self.gstate.current_mut();
        gs.fill_color = c;
        gs.fill_pattern = None;
    }

    fn set_stroke(&mut self, c: RasterColor) {
        let gs = self.gstate.current_mut();
        gs.stroke_color = c;
        gs.stroke_pattern = None;
    }

    /// `g`, `rg`, `k`: select the device space `device`, or the default
    /// space that replaces it (PDF §8.6.5.6), for fill operations and set
    /// the fill colour to `comps` in that space (PDF §8.6.8).
    fn set_fill_device(&mut self, device: ColorSpace, comps: &[f64]) {
        let space = self.resources.device_space(device);
        let c = self.color_for_components(&space, comps);
        self.set_fill(c);
        self.gstate.current_mut().fill_color_space = space;
    }

    /// `G`, `RG`, `K`: the stroking counterpart of [`Self::set_fill_device`].
    fn set_stroke_device(&mut self, device: ColorSpace, comps: &[f64]) {
        let space = self.resources.device_space(device);
        let c = self.color_for_components(&space, comps);
        self.set_stroke(c);
        self.gstate.current_mut().stroke_color_space = space;
    }

    /// Resolve `sc`/`scn` (or `SC`/`SCN`) component operands against an
    /// already-selected colour space.
    ///
    /// For plain device spaces the raw components ARE the colour, so the
    /// fast [`components_to_color`] path is kept (and stays bit-identical
    /// for RGB/CMYK/Gray).  Parameterised spaces — `Separation`, `DeviceN`,
    /// `ICCBased`, `Indexed`, `Lab` — must run the components through the
    /// space's tint/profile transform (PDF §8.6.6): treating a Separation
    /// tint of `1.0` as a grey of `1.0` paints white and silently drops all
    /// content on spot-colour pages.
    fn color_for_components(&self, cs: &ColorSpace, comps: &[f64]) -> RasterColor {
        match cs {
            ColorSpace::DeviceGray | ColorSpace::DeviceRgb | ColorSpace::DeviceCmyk => {
                components_to_color(comps)
            }
            // Pattern operands are handled by the pattern dispatch, not here;
            // fall back to the legacy heuristic for any stray component.
            ColorSpace::Pattern { .. } => components_to_color(comps),
            _ => RasterColor::from_bytes(cs.convert_to_rgb(self.resources.doc(), comps)),
        }
    }

    fn path_builder(&mut self) -> &mut PathBuilder {
        self.path.get_or_insert_with(PathBuilder::new)
    }

    /// Transform a user-space point to device space using the current CTM.
    /// Applies a y-flip because PDF uses bottom-left origin but bitmaps use top-left.
    fn to_device(&self, x: f64, y: f64) -> (f64, f64) {
        let ctm = &self.gstate.current().ctm;
        let (dx, dy) = ctm_transform(ctm, x, y);
        (dx, f64::from(self.height) - dy)
    }

    /// The user-space → device-space matrix: the CTM followed by the y-flip
    /// that [`Self::to_device`] applies.
    fn device_matrix(&self) -> [f64; 6] {
        let m = self.gstate.current().ctm;
        [
            m[0],
            -m[1],
            m[2],
            -m[3],
            m[4],
            f64::from(self.height) - m[5],
        ]
    }

    /// Apply a pending `W`/`W*` clip using the current path, then discard the path.
    ///
    /// Called by `EndPath` (`n`) when no painting follows.
    fn apply_pending_clip(&mut self) {
        if let Some(even_odd) = self.pending_clip.take() {
            if let Some(builder) = self.path.take() {
                self.clip_path_into_gstate(&builder.build(), even_odd);
            } else {
                log::debug!("rasterrocket-interp: W/W* with no current path — clip unchanged");
            }
        }
    }

    /// Apply a pending `W`/`W*` clip using an already-built path.
    ///
    /// Called by paint helpers (`do_fill`, `do_stroke`, `do_fill_then_stroke`)
    /// when a `W`/`W*` preceded the painting operator.  The same path is used
    /// for both clipping and painting (PDF §8.5.4).
    fn apply_pending_clip_with(&mut self, path: &raster::path::Path) {
        if let Some(even_odd) = self.pending_clip.take() {
            self.clip_path_into_gstate(path, even_odd);
        }
    }

    /// Intersect `path` into the current graphics-state clip region.
    fn clip_path_into_gstate(&mut self, path: &raster::path::Path, even_odd: bool) {
        let flatness = self.gstate.current().flatness.max(0.1);
        let xpath = XPath::new(path, &DEVICE_MATRIX, flatness, true);
        self.gstate
            .current_mut()
            .clip
            .clip_to_path(&xpath, even_odd);
    }

    /// Build a [`PipeState`] with the given opacity and blend mode;
    /// `backdrop` is the enclosing non-isolated group's, for a blend mode
    /// that consults it (see [`Self::blend_backdrop`]).
    fn make_pipe(a_input: u8, blend_mode: BlendMode, backdrop: Option<&Backdrop>) -> PipeState<'_> {
        PipeState {
            blend_mode,
            a_input,
            overprint_mask: 0xFFFF_FFFF,
            overprint_additive: false,
            transfer: TransferSet::identity_rgb(),
            backdrop: backdrop.map(Backdrop::as_pipe_backdrop),
        }
    }

    /// The backdrop a paint under `blend_mode` blends with: the enclosing
    /// non-isolated group's, unless the mode is `Normal`, whose result does
    /// not depend on it.
    fn blend_backdrop(&self, blend_mode: BlendMode) -> Option<Arc<Backdrop>> {
        (blend_mode != BlendMode::Normal)
            .then(|| self.group_backdrop.clone())
            .flatten()
    }

    fn do_fill(&mut self, even_odd: bool) {
        let Some(builder) = self.path.take() else {
            self.pending_clip = None;
            return;
        };
        let path = builder.build();
        self.apply_pending_clip_with(&path);
        self.fill_path(&path, even_odd);
    }

    fn do_stroke(&mut self) {
        let Some(builder) = self.path.take() else {
            self.pending_clip = None;
            return;
        };
        let path = builder.build();
        self.apply_pending_clip_with(&path);
        self.stroke_path(&path);
    }

    fn do_fill_then_stroke(&mut self, even_odd: bool) {
        let Some(builder) = self.path.take() else {
            self.pending_clip = None;
            return;
        };
        let path = builder.build();
        self.apply_pending_clip_with(&path);
        self.fill_and_stroke(&path, even_odd);
    }

    /// Fill `path` using the current fill colour or pattern.  Shared by `do_fill`
    /// and `do_fill_then_stroke`.
    fn fill_path(&mut self, path: &raster::path::Path, even_odd: bool) {
        // Stroke adjustment is always on: a lone axis-aligned quadrilateral
        // fills whole device pixels.
        if self.gstate.current().fill_color_space.paints_nothing() {
            return;
        }
        let hinted = path.with_fill_adjust_hints();
        let path = hinted.as_ref().unwrap_or(path);
        let region = self.paint_region(path.pts.iter().map(|p| (p.x, p.y)), 1.0);
        self.paint_masked(region, |r| r.fill_within(path, even_odd, region));
    }

    /// Fill `path`, whose pixels lie within `region`.
    fn fill_within(
        &mut self,
        path: &raster::path::Path,
        even_odd: bool,
        region: Option<PixelRect>,
    ) {
        let gs = self.gstate.current();
        let clip = gs.clip.clone_shared();
        let flatness = gs.flatness.max(0.1);
        let backdrop = self.blend_backdrop(gs.blend_mode);
        let pipe = Self::make_pipe(gs.fill_alpha, gs.blend_mode, backdrop.as_deref());

        // Resolve the fill source — pattern or solid colour.
        let pat_name = gs.fill_pattern.clone();
        let solid_color = gs.painted_fill_color().to_vec();
        let _ = gs; // end the immutable gstate borrow before the &mut self pattern resolve

        // A pattern colour that does not resolve, or whose object lies
        // outside the clip, paints nothing: the fill colour is not a
        // fallback for it.
        let tiled = match pat_name
            .as_deref()
            .map(|name| region.and_then(|region| self.resolve_pattern(name, region)))
        {
            Some(Some(PatternPaint::Shading(pattern))) => {
                self.paint_shading_pattern(&pattern, false, region, |r, pipe, shading| {
                    r.fill_with_shading(path, even_odd, clip, pipe, flatness, shading);
                });
                return;
            }
            Some(Some(PatternPaint::Tiled(tiled))) => Some(tiled),
            Some(None) => return,
            None => None,
        };
        let src = tiled.as_ref().map_or(PipeSrc::Solid(&solid_color), |p| {
            PipeSrc::Pattern(p as &dyn Pattern)
        });

        // GPU fill dispatch — solid-colour fills only (patterns not yet
        // GPU-accelerated).  Cloning the `Arc` releases the immutable borrow
        // before the mutable `self` borrow inside the helpers.
        //
        // The device evaluates the CPU scanner's intersection table with
        // the CPU's span walk and composites through the CPU pipeline, so
        // either branch yields the bytes the CPU fill below would.  The
        // Vulkan branch fires before the CUDA branch when both are present,
        // but in practice `rasterrocket::render::open_session` wires only
        // one of them based on `BackendPolicy`, so the second branch only
        // runs under `Auto` / `ForceCuda`.
        #[cfg(feature = "vulkan")]
        if tiled.is_none()
            && let Some(vk) = self.vk_backend.clone()
            && self.try_vk_aa_fill(path, even_odd, flatness, &pipe, &src, &vk)
        {
            return;
        }
        #[cfg(feature = "gpu-aa")]
        if tiled.is_none()
            && let Some(ctx) = self.gpu_ctx.clone()
            && self.try_gpu_aa_fill(path, even_odd, flatness, &pipe, &src, &ctx)
        {
            return;
        }

        if even_odd {
            eo_fill(
                &mut self.bitmap,
                &clip,
                path,
                &pipe,
                &src,
                &DEVICE_MATRIX,
                flatness,
                true,
            );
        } else {
            fill(
                &mut self.bitmap,
                &clip,
                path,
                &pipe,
                &src,
                &DEVICE_MATRIX,
                flatness,
                true,
            );
        }
        drop(tiled);
    }

    /// Attempt to rasterise `path` with the GPU scanline-AA kernel.
    ///
    /// Returns `true` if the GPU path was taken (caller skips the CPU fill).
    /// Returns `false` if the fill is not device-eligible (see
    /// `gpu_ops::prepare_aa_fill`) or the GPU call fails (warning logged;
    /// CPU fill used as fallback).
    #[cfg(feature = "gpu-aa")]
    fn try_gpu_aa_fill(
        &mut self,
        path: &raster::path::Path,
        even_odd: bool,
        flatness: f64,
        pipe: &raster::pipe::PipeState<'_>,
        src: &raster::pipe::PipeSrc<'_>,
        ctx: &gpu::GpuCtx,
    ) -> bool {
        gpu_ops::try_gpu_aa_fill(self, path, even_odd, flatness, pipe, src, ctx)
    }

    /// Vulkan twin of [`Self::try_gpu_aa_fill`].  See [`vk_ops::try_vk_aa_fill`].
    #[cfg(feature = "vulkan")]
    fn try_vk_aa_fill(
        &mut self,
        path: &raster::path::Path,
        even_odd: bool,
        flatness: f64,
        pipe: &raster::pipe::PipeState<'_>,
        src: &raster::pipe::PipeSrc<'_>,
        backend: &Arc<VulkanBackend>,
    ) -> bool {
        vk_ops::try_vk_aa_fill(self, backend, path, even_odd, flatness, pipe, src)
    }

    /// Stroke `path` with the current line width, widened to at least one
    /// device pixel.
    fn stroke_path(&mut self, path: &raster::path::Path) {
        let line_width = self.min_stroke_width();
        self.stroke_path_at(path, line_width);
    }

    /// Stroke a text outline (render modes 1, 2, 5, 6) with the current line
    /// width as given: the outline's own fill carries the glyph shape, and a
    /// widened outline stroke thickens every stem.
    pub(super) fn stroke_text_outline(&mut self, path: &raster::path::Path) {
        let line_width = self.gstate.current().line_width;
        self.stroke_path_at(path, line_width);
    }

    fn stroke_path_at(&mut self, path: &raster::path::Path, line_width: f64) {
        let region = self.stroke_bounds(path, line_width);
        self.paint_masked(region, |r| r.stroke_within(path, line_width, region));
    }

    /// The current line width, raised so the stroke is at least one device
    /// pixel thick, the device thickness being the user width times the
    /// CTM's area scale `√|det|`.  Width 0 keeps its one-pixel hairline;
    /// under a singular CTM the width is already in device pixels.
    fn min_stroke_width(&self) -> f64 {
        let line_width = self.gstate.current().line_width;
        if line_width <= 0.0 {
            return line_width;
        }
        let device = self.device_matrix();
        let scale = invert_affine(&device).map_or(1.0, |_| {
            device[0]
                .mul_add(device[3], -(device[1] * device[2]))
                .abs()
                .sqrt()
        });
        line_width.max(1.0 / scale)
    }

    /// The pixels a stroke of `path` can touch: the path's bounds grown by
    /// the stroke's reach — half the line width times the miter limit (or
    /// √2, for a square cap), in device pixels — within the clip.  Under a
    /// singular CTM the stroker works in device space with the line width
    /// unscaled.
    fn stroke_bounds(&self, path: &raster::path::Path, line_width: f64) -> Option<PixelRect> {
        let gs = self.gstate.current();
        let device = self.device_matrix();
        let scale = invert_affine(&device).map_or(1.0, |_| {
            device[0].hypot(device[1]).max(device[2].hypot(device[3]))
        });
        let reach = line_width / 2.0 * scale * gs.miter_limit.max(std::f64::consts::SQRT_2);
        self.paint_region(path.pts.iter().map(|p| (p.x, p.y)), reach + 1.0)
    }

    /// Stroke `path` `line_width` wide, whose stroke lies within `region`.
    fn stroke_within(
        &mut self,
        path: &raster::path::Path,
        line_width: f64,
        region: Option<PixelRect>,
    ) {
        // Clone all graphics state data before any mutable borrow.
        let gs = self.gstate.current();
        if gs.stroke_color_space.paints_nothing() {
            return;
        }
        let clip = gs.clip.clone_shared();
        let backdrop = self.blend_backdrop(gs.blend_mode);
        let pipe = Self::make_pipe(gs.stroke_alpha, gs.blend_mode, backdrop.as_deref());
        let line_cap = gs.line_cap;
        let line_join = gs.line_join;
        let miter_limit = gs.miter_limit;
        let flatness = gs.flatness.max(0.1);
        let line_dash = gs.dash.0.clone();
        let line_dash_phase = gs.dash.1;
        let pat_name = gs.stroke_pattern.clone();
        let solid_color = gs.painted_stroke_color().to_vec();
        // gs immutable borrow ends here.

        // PDF §8.4.3.2 / §8.4.3.6: the line width and dash lengths are
        // user-space distances under the CTM at stroke time.  The path is held
        // in device space, so it is mapped back through the inverse device
        // matrix and the stroker transforms the outline it builds; flatness
        // stays a device-space tolerance.  A singular CTM collapses the path
        // to a line or point, so it is stroked in device space unchanged.
        let device = self.device_matrix();
        let device_flatness = flatness;
        let (path, matrix, flatness) = invert_affine(&device).map_or_else(
            || (path.clone(), DEVICE_MATRIX, flatness),
            |inv| {
                let scale = device[0]
                    .mul_add(device[3], -(device[1] * device[2]))
                    .abs()
                    .sqrt();
                (transform_path(path, &inv), device, flatness / scale)
            },
        );

        let params = StrokeParams {
            line_width,
            line_cap,
            line_join,
            miter_limit,
            flatness,
            line_dash: &line_dash,
            line_dash_phase,
            stroke_adjust: false,
            vector_antialias: true,
        };

        // Resolve the stroke source — pattern or solid colour.  A pattern
        // colour that does not resolve paints nothing.
        let tiled = match pat_name
            .as_deref()
            .map(|name| region.and_then(|region| self.resolve_pattern(name, region)))
        {
            Some(Some(PatternPaint::Shading(pattern))) => {
                self.paint_shading_pattern(&pattern, true, region, |r, pipe, shading| {
                    r.stroke_with_shading(
                        &path,
                        &matrix,
                        &params,
                        clip,
                        pipe,
                        device_flatness,
                        shading,
                    );
                });
                return;
            }
            Some(Some(PatternPaint::Tiled(tiled))) => Some(tiled),
            Some(None) => return,
            None => None,
        };
        let src = tiled.as_ref().map_or(PipeSrc::Solid(&solid_color), |p| {
            PipeSrc::Pattern(p as &dyn Pattern)
        });

        stroke(
            &mut self.bitmap,
            &clip,
            &path,
            &pipe,
            &src,
            &matrix,
            &params,
        );
        drop(tiled);
    }

    // ── XObject rendering ─────────────────────────────────────────────────────

    /// Execute a `Do` operator: look up and render the named `XObject`.
    ///
    /// An `XObject` whose `/OC` is hidden is neither decoded nor drawn.
    /// Image `XObjects` are blitted directly.  Form `XObjects` are executed
    /// recursively via [`do_form_xobject`].
    fn do_xobject(&mut self, name: &[u8]) {
        if let Some(oc) = self.resources.xobject_optional_content(name)
            && !self.oc_visible(&oc)
        {
            return;
        }
        // Try Form first (cheap dict lookup, no pixel decoding).
        if let Some(form) = self.resources.form_xobject(name) {
            self.do_form_xobject(&form);
            return;
        }
        // Try Image, decoding it only when this page has not already.
        let draw_size = image_draw_size(&self.gstate.current().ctm);
        let id = self.resources.xobject_id(name);
        let memoized = id.and_then(|id| self.image_memo.borrow().get(id, draw_size));
        if let Some(decoded) = memoized {
            self.draw_image_xobject(name, &decoded);
            return;
        }
        let request = crate::resources::ImageRequest { name, draw_size };
        let mut ctx = crate::resources::image::ImageDecodeCtx {
            doc: self.resources.doc(),
            color_spaces: None,
            #[cfg(feature = "nvjpeg")]
            nvjpeg: self.nvjpeg.as_mut(),
            #[cfg(feature = "vaapi")]
            vaapi: self.vaapi_jpeg_queue.as_ref(),
            #[cfg(feature = "nvjpeg2k")]
            nvjpeg2k: self.nvjpeg2k.as_mut(),
            #[cfg(feature = "gpu-icc")]
            icc: crate::resources::image::IccHandles {
                gpu_ctx: self.gpu_ctx.as_deref(),
                clut_cache: Some(&mut self.icc_clut_cache),
            },
            #[cfg(feature = "cache")]
            image_cache: self.cache_state.as_ref().map(|cs| &cs.cache),
            #[cfg(feature = "cache")]
            doc_id: self.cache_state.as_ref().map(|cs| cs.doc_id),
        };
        // Route to whichever GPU JPEG decoder is active (only one is Some per render).
        // Each branch is a separately monomorphised call; no decoder is wasted.
        #[cfg(all(feature = "gpu-jpeg-huffman", feature = "vulkan"))]
        let resolution = if self.jpeg_vk.is_some() {
            self.resources
                .image(request, &mut ctx, self.jpeg_vk.as_mut())
        } else {
            self.resources
                .image(request, &mut ctx, self.jpeg_gpu.as_mut())
        };
        #[cfg(not(all(feature = "gpu-jpeg-huffman", feature = "vulkan")))]
        let resolution = self.resources.image(
            request,
            &mut ctx,
            #[cfg(feature = "gpu-jpeg-huffman")]
            self.jpeg_gpu.as_mut(),
        );
        self.handle_image_resolution(name, resolution, id.map(|id| (id, draw_size)));
    }

    /// Dispatch on an [`ImageResolution`] returned from [`PageResources::image`].
    ///
    /// - `Ok` → blit the image, keeping it in the page's image memo under
    ///   `memo_key` (the stream object and the draw size it was decoded
    ///   for) when given.
    /// - `DecodeFailed` → record the failure in the page diagnostics and skip
    ///   the image; the pixel region keeps what was painted beneath it.
    /// - `Absent` → a non-image `XObject` subtype with no raster appearance;
    ///   log a debug message and skip.
    fn handle_image_resolution(
        &mut self,
        name: &[u8],
        resolution: crate::resources::image::ImageResolution,
        memo_key: Option<(ObjectId, Option<(u32, u32)>)>,
    ) {
        use crate::resources::image::ImageResolution;
        match resolution {
            ImageResolution::Ok(decoded) => match memo_key {
                Some((id, draw_size)) => {
                    let decoded = self.image_memo.borrow_mut().insert(id, draw_size, decoded);
                    self.draw_image_xobject(name, &decoded);
                }
                None => self.draw_image_xobject(name, &decoded),
            },
            ImageResolution::DecodeFailed(msg) => {
                self.record_image_failure(Some(String::from_utf8_lossy(name).into_owned()), msg);
            }
            ImageResolution::Absent => {
                log::debug!(
                    "rasterrocket-interp: Do /{} skipped (XObject subtype has no raster appearance)",
                    String::from_utf8_lossy(name)
                );
            }
        }
    }

    /// Blit the decoded image `XObject` `name`, attributing a blit failure
    /// to it.
    fn draw_image_xobject(&mut self, name: &[u8], decoded: &crate::resources::DecodedImage) {
        self.current_image = Some(String::from_utf8_lossy(name).into_owned());
        self.blit_image(decoded);
        self.current_image = None;
    }

    /// Execute a Form `XObject`'s content stream in the current graphics context.
    ///
    /// PDF §8.10.1: the form's `Matrix` is concatenated onto the current CTM,
    /// graphics state is saved/restored around execution, and the form's own
    /// `Resources` dict (if present) is used to resolve fonts and images inside
    /// the form.
    ///
    /// When the form carries a `Group /S /Transparency` entry, content is rendered
    /// into an intermediate group bitmap and composited back (PDF §11.6.6).
    ///
    /// A form invoked while it is already executing, directly or through
    /// other forms, draws nothing.
    fn do_form_xobject(&mut self, form: &crate::resources::FormXObject) {
        if self.form_depth >= MAX_FORM_DEPTH {
            self.budget_exceeded = Some(format!(
                "Form XObject nesting depth {MAX_FORM_DEPTH} exceeded; aborting page render"
            ));
            log::warn!(
                "rasterrocket-interp: Form XObject nesting depth {MAX_FORM_DEPTH} exceeded — aborting"
            );
            return;
        }
        // Propagate an already-tripped budget: a recursive form call after
        // the outer execute() set budget_exceeded must not execute further.
        if self.budget_exceeded.is_some() {
            return;
        }
        if self.active_forms.contains(&form.resources_id) {
            log::warn!(
                "rasterrocket-interp: Form XObject ({}, {}) invokes itself — recursive Do skipped",
                form.resources_id.0,
                form.resources_id.1
            );
            return;
        }
        self.active_forms.push(form.resources_id);

        // Capture the parent's compositing state before saving, so the group is
        // painted using the opacity/blend that was active when the form was invoked
        // (PDF §11.6.6: the group is composited using the surrounding context).
        let parent_fill_alpha = self.gstate.current().fill_alpha;
        let parent_blend_mode = self.gstate.current().blend_mode;

        // Save graphics state (equivalent to `q`).
        self.gstate.save();
        self.form_depth += 1;

        // Concatenate the form's Matrix onto the current CTM.
        // PDF §8.10.1: the form's /Matrix maps form space to the parent user
        // space in effect when `Do` was invoked; it pre-multiplies, exactly as
        // `cm` does (§8.3.4) — new_CTM = form.matrix × old_CTM, never the
        // reverse.  Post-multiplying scales the matrix's translation by the
        // wrong factor and pushes nested-form content off the device page.
        let old_ctm = self.gstate.current().ctm;
        let form_ctm = ctm_multiply(&form.matrix, &old_ctm);
        self.gstate.current_mut().ctm = form_ctm;
        let parent_pattern_ctm = std::mem::replace(&mut self.pattern_ctm, form_ctm);
        // PDF §8.10.1: the form's BBox, in form space, clips its content.
        if let Some(bbox) = form.bbox {
            self.clip_to_user_rect(&bbox);
        }

        // Switch to the form's resource context, keeping the parent for restore.
        let child_resources = self.resources.for_form(form);
        let parent_resources = std::mem::replace(&mut self.resources, child_resources);

        // Transparency group: render into an intermediate bitmap, then
        // composite it back onto the page.
        let mut group = form
            .transparency
            .and_then(|tg| self.group_params(form.bbox, tg))
            .map(|params| {
                let bitmap = self.group_bitmap();
                begin_group(
                    &self.bitmap,
                    self.group_backdrop.as_deref(),
                    &self.gstate.current().clip,
                    params,
                    bitmap,
                )
            });

        // Swap in the group bitmap when present.  PDF §11.6.6: the alpha
        // constants and blend mode start at their initial values inside a
        // transparency group; the parent's values apply once, when the group
        // is composited.
        // A group is one object under the soft mask in effect: the mask
        // applies when the group is composited, not to its content.
        let group_mask = group
            .as_ref()
            .and_then(|_| self.gstate.current_mut().soft_mask.take());

        // Blend modes inside the group see its own backdrop: the parent's
        // pixels for a non-isolated group, nothing for an isolated one.
        let parent_backdrop = group.as_mut().map(|g| {
            std::mem::swap(&mut self.bitmap, &mut g.bitmap);
            let (x0, y0, x1, y1) = g.region();
            let gs = self.gstate.current_mut();
            gs.fill_alpha = 255;
            gs.stroke_alpha = 255;
            gs.blend_mode = BlendMode::Normal;
            // The content is clipped to the region, so the group bitmap is
            // painted nowhere else and clearing the region restores it.
            gs.clip
                .clip_to_rect(f64::from(x0), f64::from(y0), f64::from(x1), f64::from(y1));
            std::mem::replace(&mut self.group_backdrop, g.backdrop.take().map(Arc::new))
        });

        // Parse and execute the form's content stream.  Its `Q`s cannot
        // pop the state saved above.
        let ops = crate::content::parse(&form.content);
        let floor = self.gstate.raise_floor();
        self.execute_content_stream(&ops);
        self.gstate.lower_floor(floor);

        // Swap the group bitmap back and composite using the parent's opacity/blend
        // (captured before gstate.save() — the form's own state is irrelevant here).
        if let (Some(mut g), Some(parent_backdrop)) = (group, parent_backdrop) {
            std::mem::swap(&mut self.bitmap, &mut g.bitmap);
            self.group_backdrop = parent_backdrop;
            let backdrop = self.blend_backdrop(parent_blend_mode);
            let pipe = Self::make_pipe(parent_fill_alpha, parent_blend_mode, backdrop.as_deref());
            let (x0, y0, x1, y1) = g.region();
            let region = PixelRect::new(x0, y0, x1, y1);
            // Residual Clip is unused — parent gstate is restored immediately below.
            let (_, bitmap) = match group_mask {
                Some(mask) => self.paint_through_soft_mask(&mask, region, |r| {
                    paint_group(&mut r.bitmap, g, &pipe)
                }),
                None => paint_group(&mut self.bitmap, g, &pipe),
            };
            self.group_pool.push(bitmap);
        }

        // Restore resources and graphics state.
        self.resources = parent_resources;
        self.pattern_ctm = parent_pattern_ctm;
        self.form_depth -= 1;
        let _ = self.active_forms.pop();
        self.gstate.restore();
    }

    /// A transparent bitmap of the current bitmap's size for a transparency
    /// group: one from the pool, or a fresh one.
    fn group_bitmap(&mut self) -> Bitmap<Rgb8> {
        let (width, height) = (self.bitmap.width, self.bitmap.height);
        self.group_pool
            .pop()
            .filter(|b| b.width == width && b.height == height)
            .unwrap_or_else(|| Bitmap::new(width, height, 1, true))
    }

    /// The parameters of a transparency group `tg` whose form has `bbox`:
    /// the box's extent under the current CTM, in device pixels, or the
    /// whole page when the form has no box.  `None` when the extent is
    /// non-finite; the form then paints directly onto the page.
    fn group_params(
        &self,
        bbox: Option<[f64; 4]>,
        tg: crate::resources::TransparencyGroupParams,
    ) -> Option<GroupParams> {
        let Some(bbox) = bbox else {
            return Some(GroupParams {
                x_min: 0,
                y_min: 0,
                x_max: i32::try_from(self.width).unwrap_or(i32::MAX),
                y_max: i32::try_from(self.height).unwrap_or(i32::MAX),
                isolated: tg.isolated,
                soft_mask_type: raster::transparency::SoftMaskType::None,
            });
        };
        let ctm = self.gstate.current().ctm;
        let page_h = f64::from(self.height);
        let [bx0, by0, bx1, by1] = bbox;
        let corners = [
            ctm_transform(&ctm, bx0, by0),
            ctm_transform(&ctm, bx1, by0),
            ctm_transform(&ctm, bx0, by1),
            ctm_transform(&ctm, bx1, by1),
        ];
        if corners
            .iter()
            .any(|(x, y)| !x.is_finite() || !y.is_finite())
        {
            log::warn!(
                "rasterrocket-interp: transparency group BBox is non-finite — rendering without group"
            );
            return None;
        }
        let left = corners
            .iter()
            .map(|(x, _)| *x)
            .fold(f64::INFINITY, f64::min)
            .floor();
        let top = corners
            .iter()
            .map(|(_, y)| page_h - y)
            .fold(f64::INFINITY, f64::min)
            .floor();
        let right = corners
            .iter()
            .map(|(x, _)| *x)
            .fold(f64::NEG_INFINITY, f64::max)
            .ceil();
        let bottom = corners
            .iter()
            .map(|(_, y)| page_h - y)
            .fold(f64::NEG_INFINITY, f64::max)
            .ceil();

        #[expect(
            clippy::cast_possible_truncation,
            reason = "f64→i32 cast saturates in Rust ≥ 1.45 (no UB); \
                      begin_group clamps GroupParams to bitmap bounds"
        )]
        let params = GroupParams {
            x_min: left as i32,
            y_min: top as i32,
            x_max: right as i32,
            y_max: bottom as i32,
            isolated: tg.isolated,
            soft_mask_type: raster::transparency::SoftMaskType::None,
        };
        Some(params)
    }

    /// Blit a decoded image `XObject` onto the bitmap using the current CTM.
    ///
    /// The CTM maps the unit square (image space) onto the page; every
    /// device pixel inside the image's bounding box samples the image at its
    /// centre, through a [`SampleGrid`] shared by the CPU sampler below and
    /// the GPU blit kernel: the nearest sample, or a bilinear blend along
    /// the axes [`Filter::for_draw`] selects for a magnified or rotated
    /// draw.  A host-resident image or stencil drawn below its resolution
    /// is first area-averaged onto the device pixels (`image_reduce`).  A
    /// mask is resampled the same way and sampled through a [`MaskSampler`]
    /// at its own resolution.
    ///
    /// `decoded.source_width` is the stream's own pixel width, which the
    /// image's width falls below when it was decoded at reduced resolution;
    /// it feeds the `source_ppi_hint` diagnostic.
    fn blit_image(&mut self, decoded: &crate::resources::DecodedImage) {
        let img = &decoded.image;
        let source_width = decoded.source_width;
        // Degenerate image — nothing to blit.
        if img.width == 0 || img.height == 0 {
            return;
        }
        // A stencil mask paints the fill colour, which may paint nothing.
        if img.color_space == ImageColorSpace::Mask
            && self.gstate.current().fill_color_space.paints_nothing()
        {
            return;
        }
        // PDF §8.9.6: stencil masks (/ImageMask true → ImageColorSpace::Mask)
        // do not carry an /SMask.  The Mask arm below intentionally skips
        // the smask gate that the Rgb / Gray arms apply; this assert pins
        // the spec invariant so a future decoder change that sets `smask`
        // on a Mask descriptor surfaces in debug builds.
        debug_assert!(
            img.smask.is_none() || !matches!(img.color_space, ImageColorSpace::Mask),
            "PDF §8.9.6: image masks must not carry SMask",
        );

        // Track diagnostics: image presence and per-filter counts.
        self.diag.has_images = true;
        let filter_idx = img.filter as usize;
        self.filter_counts[filter_idx] = self.filter_counts[filter_idx].saturating_add(1);

        // Update source_ppi_hint: take the widest image seen (best resolution proxy).
        // ppi = (image_px_width / page_pts_width) × 72.
        let page_pts_w = f64::from(self.width) / {
            // For unrotated pages CTM[0] carries the scale; for 90°/270° rotations
            // CTM[0] is 0 and CTM[1] carries it.  Take the larger of the two to
            // handle all axis-aligned cases.  Falls back to 1.0 for a degenerate CTM
            // (page_pts_w would be width in pixels — wrong, but finite).
            let ctm = self.gstate.current().ctm;
            let ctm_scale = ctm[0].abs().max(ctm[1].abs());
            if ctm_scale > 0.0 { ctm_scale } else { 1.0 }
        };
        #[expect(
            clippy::cast_possible_truncation,
            reason = "src_ppi is finite and positive, checked below; f64→f32 is safe for PPI values in practice (well under f32::MAX)"
        )]
        let src_ppi = ((f64::from(source_width) / page_pts_w) * 72.0) as f32;
        if src_ppi.is_finite() && src_ppi > 0.0 {
            self.diag.source_ppi_hint = Some(match self.diag.source_ppi_hint {
                Some(prev) if prev >= src_ppi => prev,
                _ => src_ppi,
            });
        }

        // An image or mask drawn below its resolution is area-averaged
        // first, so that point sampling neither drops thin features nor
        // turns them into whole pixels.  The image and its mask resample
        // separately, each on its own grid, and each comes with the matrix
        // that draws the resampled samples.
        let page_ctm = self.gstate.current().ctm;
        let reduced_mask = img
            .smask
            .as_ref()
            .and_then(|m| image_reduce::reduce_mask_for_draw(m, &page_ctm));
        let (mask_src, mask_ctm) = match &reduced_mask {
            Some((m, c)) => (Some(m), *c),
            None => (img.smask.as_ref(), page_ctm),
        };
        let reduced = image_reduce::reduce_for_draw(img, &page_ctm);
        let (img, ctm) = match &reduced {
            Some((i, c)) => (i, *c),
            None => (img, page_ctm),
        };
        let filter = Filter::for_draw(&ctm, img.width, img.height, decoded.interpolate);
        let Some(grid) =
            SampleGrid::new(&ctm, self.width, self.height, img.width, img.height, filter)
        else {
            log::debug!(
                "rasterrocket-interp: blit_image: image maps to no page pixels (CTM {ctm:?}) — skipping"
            );
            return;
        };

        // The grid covers the image's bounding box on the page; the clip
        // region decides which of those pixels the image may paint.
        let clip = self.gstate.current().clip.clone_shared();
        let mut grid_clip = GridClip::new(&clip, &grid);
        if grid_clip.bbox() == ClipResult::AllOutside {
            return;
        }

        if matches!(img.color_space, ImageColorSpace::Mask)
            && self.gstate.current().fill_pattern.is_some()
            && self.fill_stencil_with_pattern(img, &grid)
        {
            return;
        }

        // The image is one object under the soft mask in effect, unless its
        // own `/SMask` replaces that mask.
        let mut paint = |r: &mut Self| {
            r.composite_image(
                img,
                &ctm,
                mask_src.map(|m| (m, mask_ctm)),
                &grid,
                &mut grid_clip,
                decoded.interpolate,
            );
        };
        if decoded.own_soft_mask {
            paint(self);
        } else {
            let region = PixelRect::new(grid.x0, grid.y0, grid.x1, grid.y1);
            self.paint_masked(region, paint);
        }
    }

    /// Paint the fill pattern through the stencil mask `img` sampled
    /// through `grid` (PDF §8.9.6.2: the stencil's set samples are painted
    /// with the current fill colour, here a pattern).  The pattern fills the
    /// grid's rectangle, scaled at each pixel by the stencil coverage, so a
    /// tiling or shading pattern of any kind — with its `Background`, `BBox`
    /// and `ExtGState` — paints exactly as it does through a path fill.
    ///
    /// Returns `false`, painting nothing, when the stencil is not host
    /// resident.
    fn fill_stencil_with_pattern(
        &mut self,
        img: &crate::resources::ImageDescriptor,
        grid: &SampleGrid,
    ) -> bool {
        let Some(bytes) = img.data.as_cpu() else {
            return false;
        };
        let Some(rect) = PixelRect::new(grid.x0, grid.y0, grid.x1, grid.y1) else {
            return true;
        };
        let img_w = img.width as usize;
        let sample = |x: u32, y: u32| {
            bytes
                .get(y as usize * img_w + x as usize)
                .copied()
                .unwrap_or(u8::MAX)
        };
        let coverage: Vec<u8> = (rect.y0..rect.y1)
            .flat_map(|dy| (rect.x0..rect.x1).map(move |dx| (dx, dy)))
            .map(|(dx, dy)| {
                grid.sample(dx, dy)
                    .map_or(0, |taps| u8::MAX - taps.blend(sample))
            })
            .collect();
        let (x0, y0) = (f64::from(rect.x0), f64::from(rect.y0));
        let (x1, y1) = (f64::from(rect.x1), f64::from(rect.y1));
        let area = shading_paint::polygon(&[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]);
        self.paint_through_coverage(rect, &coverage, |r| r.fill_path(&area, false));
        true
    }

    /// Composite `img`, sampled through `grid` under `ctm`, onto the bitmap
    /// within `grid_clip`; `mask` is its soft mask with the matrix that
    /// draws it, sampled bilinearly when `interpolate` is set.
    #[expect(
        clippy::too_many_lines,
        reason = "GPU dispatch, decoder-contract checks, the mask sampler and the per-pixel loop share the grid and CTM state"
    )]
    fn composite_image(
        &mut self,
        img: &crate::resources::ImageDescriptor,
        ctm: &Ctm,
        mask: Option<(&SoftMask, Ctm)>,
        grid: &SampleGrid,
        grid_clip: &mut GridClip<'_>,
        interpolate: bool,
    ) {
        // GPU image-blit path: device-resident pixels sampled by the CUDA
        // kernel through the same grid, composited onto `bitmap` before
        // this call returns so later operators paint over the image in
        // content-stream order.
        //
        // The kernel composites opaquely and takes no mask: the decoder
        // keeps every masked image host-resident, so masked images of any
        // resolution take the CPU path below.
        #[cfg(feature = "cache")]
        if let crate::resources::image::ImageData::Gpu(cached) = &img.data {
            debug_assert!(img.smask.is_none(), "device-resident images carry no mask");
            if self.try_gpu_blit_image(cached, grid, grid_clip) {
                return;
            }
            // Promotion or kernel dispatch failed and there are no host
            // bytes to fall back to.  The image decoded successfully but
            // cannot be rendered: this is the same silent-blank class the
            // `ImageResolution` contract eradicates upstream, so record it
            // as an image failure rather than dropping the image with only a
            // log line.
            self.record_image_failure(
                self.current_image.clone(),
                "GPU blit failed, no host fallback".to_owned(),
            );
            return;
        }

        // CPU path: needs host-resident pixels.
        let Some(img_bytes) = img.data.as_cpu() else {
            // Decoded pixels exist but are device-resident with no GPU
            // dispatch available to blit them — unrenderable, not absent.
            self.record_image_failure(
                self.current_image.clone(),
                "pixels device-resident but GPU dispatch unavailable".to_owned(),
            );
            return;
        };
        // Decoder contract: bytes.len() == width × height × bpp.  Promote
        // the contract to a release-mode check so a truncated buffer (an
        // adversarial PDF reporting larger dims than the decoder produced)
        // skips the image cleanly instead of indexing past the buffer in
        // the inner loop.
        let bpp = img.color_space.bytes_per_pixel();
        match check_image_bytes_len(img.width, img.height, bpp, img_bytes.len()) {
            Ok(_) => {}
            Err(ImageLenError::DimensionOverflow) => {
                // A decoder-contract violation on an image the codec
                // reported as decoded — surface, never silently blank.
                let msg = format!(
                    "dimensions overflow usize ({}×{}×{bpp} bpp)",
                    img.width, img.height,
                );
                self.record_image_failure(self.current_image.clone(), msg);
                return;
            }
            Err(ImageLenError::ShortBuffer { expected }) => {
                let msg = format!(
                    "decoder produced short buffer ({} bytes, expected {expected})",
                    img_bytes.len(),
                );
                self.record_image_failure(self.current_image.clone(), msg);
                return;
            }
        }

        let fill_color = self.gstate.current().painted_fill_color();
        let mask = match mask {
            Some((m, mask_ctm)) => {
                let Some(sampler) = MaskSampler::new(
                    m,
                    &mask_ctm,
                    ctm,
                    self.width,
                    self.height,
                    img.width,
                    img.height,
                    interpolate,
                ) else {
                    return;
                };
                Some(sampler)
            }
            None => None,
        };
        let img_w = img.width as usize;
        let backdrop = self.blend_backdrop(self.gstate.current().blend_mode);
        // Channel `ch` of sample `(x, y)`.  Taps lie on the image, so the
        // index is inside the buffer by the length check above; the `get`
        // keeps the loop panic-free regardless.
        let channel = |x: u32, y: u32, ch: usize| {
            img_bytes
                .get((y as usize * img_w + x as usize) * bpp + ch)
                .copied()
                .unwrap_or(u8::MAX)
        };
        let gs_transfer = self.gstate.current().transfer.clone();
        let transfer = |px: [u8; 3]| gs_transfer.as_ref().map_or(px, |t| t.apply(px));
        let compositor = ImageCompositor::new(
            &self.bitmap,
            self.gstate.current().fill_alpha,
            self.gstate.current().blend_mode,
            backdrop.as_deref(),
        );

        for dy in grid.y0..grid.y1 {
            let Some(row_clip) = grid_clip.row(dy) else {
                continue;
            };
            let (row, mut alpha_row) = self.bitmap.row_and_alpha_mut(dy);
            for dx in grid.x0..grid.x1 {
                let coverage = row_clip.coverage(dx);
                if coverage == 0 {
                    continue;
                }
                let Some(taps) = grid.sample(dx, dy) else {
                    continue;
                };
                // Stencil masks paint the fill colour with coverage
                // `255 − sample` and carry no soft mask (asserted at entry).
                let alpha = match img.color_space {
                    ImageColorSpace::Mask => div255(
                        u32::from(u8::MAX - taps.blend(|x, y| channel(x, y, 0)))
                            * u32::from(coverage),
                    ),
                    ImageColorSpace::Rgb | ImageColorSpace::Gray => div255(
                        u32::from(mask.as_ref().map_or(u8::MAX, |m| m.alpha(dx, dy, &taps)))
                            * u32::from(coverage),
                    ),
                };
                if alpha == 0 {
                    continue;
                }
                // Image samples map through the transfer function; the
                // stencil colour is already the painted fill colour.
                let src = match img.color_space {
                    ImageColorSpace::Rgb => {
                        transfer([0, 1, 2].map(|ch| taps.blend(|x, y| channel(x, y, ch))))
                    }
                    ImageColorSpace::Gray => transfer([taps.blend(|x, y| channel(x, y, 0)); 3]),
                    ImageColorSpace::Mask => fill_color,
                };
                compositor.paint(row, alpha_row.as_deref_mut(), dx, dy, src, alpha);
            }
        }
    }

    /// GPU image blit: sample `cached` through `grid` on the device, then
    /// download the touched rows of the page buffer and composite them onto
    /// `bitmap`.  Returns `true` once the pixels are on the host bitmap;
    /// `false` if any prerequisite is missing (no `GpuCtx`, page-buffer
    /// alloc fails, kernel dispatch or download errors), in which case the
    /// bitmap is untouched.
    ///
    /// Compositing before returning keeps content-stream order: anything
    /// painted after the image lands on top of it, and an image inside a
    /// transparency group lands in the group's bitmap.
    #[cfg(feature = "cache")]
    fn try_gpu_blit_image(
        &mut self,
        cached: &Arc<gpu::cache::CachedDeviceImage>,
        grid: &SampleGrid,
        grid_clip: &mut GridClip<'_>,
    ) -> bool {
        use gpu::blit::BlitBbox;

        let backdrop = self.blend_backdrop(self.gstate.current().blend_mode);
        let Some(gpu_ctx) = self.gpu_ctx.as_deref() else {
            log::debug!("blit_image: GPU image blit skipped — no GpuCtx attached");
            return false;
        };
        let Some(cache_state) = self.cache_state.as_mut() else {
            // Shouldn't happen: ImageData::Gpu only originates from a
            // wired-in cache.  Defensive check; cheap.
            return false;
        };
        // Stream ordering: the blit kernel and the page buffer's row
        // download and clear all run on this thread's stream, so they
        // execute in order.  The kernel writes the page buffer through a
        // shared device pointer, which records no cross-stream write event;
        // the buffer is therefore bound to the stream of the thread that
        // renders the page.  `cached` was uploaded on the cache's stream;
        // the launch waits on its write event.

        // Grid edges are clamped to the page, so they fit i32 whenever the
        // page does; a page wider than i32::MAX pixels cannot exist.
        let (Ok(x0), Ok(y0), Ok(x1), Ok(y1)) = (
            i32::try_from(grid.x0),
            i32::try_from(grid.y0),
            i32::try_from(grid.x1),
            i32::try_from(grid.y1),
        ) else {
            return false;
        };
        let bbox = BlitBbox { x0, y0, x1, y1 };

        // Lazy-allocate the per-page device buffer on first GPU
        // image; subsequent images on the same page reuse it.
        if cache_state.page_buffer.is_none() {
            match DevicePageBuffer::new(gpu_ctx.stream(), self.width, self.height) {
                Ok(buf) => cache_state.page_buffer = Some(buf),
                Err(e) => {
                    log::warn!("blit_image: DevicePageBuffer alloc failed: {e}");
                    return false;
                }
            }
        }
        let Some(buf) = cache_state.page_buffer.as_mut() else {
            return false;
        };

        if let Err(e) = gpu_ctx.blit_image_to_buffer(
            cached,
            buf,
            bbox,
            grid.cols(),
            grid.rows(),
            grid.filter().bits(),
        ) {
            log::warn!("blit_image: GPU kernel dispatch failed: {e}");
            return false;
        }
        let compositor = ImageCompositor::new(
            &self.bitmap,
            self.gstate.current().fill_alpha,
            self.gstate.current().blend_mode,
            backdrop.as_deref(),
        );
        let transfer = self.gstate.current().transfer.clone();
        let bitmap = &mut self.bitmap;
        let downloaded = buf.download_rows_with(grid.y0, grid.y1, |y, rows| {
            composite_rgba_rows(
                bitmap,
                rows,
                y,
                grid,
                grid_clip,
                compositor,
                transfer.as_deref(),
            );
        });
        if let Err(e) = downloaded {
            log::warn!("blit_image: device page buffer download failed: {e}");
            // The buffer now holds this image's pixels with alpha set;
            // drop it so a later blit starts from a zeroed buffer.  Bands
            // composited before the failure stay on the bitmap; the image
            // is still reported as not drawn.
            cache_state.page_buffer = None;
            return false;
        }
        // Clear the touched rows so the next blit's alpha test sees only
        // its own writes.  On failure drop the buffer for the same reason
        // as above; the composite already succeeded.
        if let Err(e) = buf.zero_rows(grid.y0, grid.y1) {
            log::warn!("blit_image: device page buffer clear failed: {e}");
            cache_state.page_buffer = None;
        }
        true
    }
}

/// Source-over whole rows of a downloaded RGBA8 page buffer, the first of
/// which is page row `y0`, onto `bitmap`, restricted to the grid's columns
/// and to the pixels `grid_clip` covers.  Alpha is `255` where the blit
/// kernel wrote a sample and `0` elsewhere, so the composite copies each
/// written sample, blended by the clip coverage on rows the clip edge
/// crosses.  Samples map through `transfer` when one is in effect.
#[cfg(feature = "cache")]
fn composite_rgba_rows(
    bitmap: &mut Bitmap<Rgb8>,
    rgba: &[u8],
    y0: u32,
    grid: &SampleGrid,
    grid_clip: &mut GridClip<'_>,
    compositor: ImageCompositor<'_>,
    transfer: Option<&crate::resources::transfer::Transfer>,
) {
    use gpu::cache::RGBA_BPP;

    let width = bitmap.width as usize;
    let src_stride = width * RGBA_BPP;
    for (src_row, dy) in rgba.chunks_exact(src_stride.max(1)).zip(y0..) {
        let Some(row_clip) = grid_clip.row(dy) else {
            continue;
        };
        let (row, mut alpha_row) = bitmap.row_and_alpha_mut(dy);
        for dx in grid.x0..grid.x1 {
            let s = &src_row[dx as usize * RGBA_BPP..dx as usize * RGBA_BPP + RGBA_BPP];
            let coverage = row_clip.coverage(dx);
            if s[3] == 0 || coverage == 0 {
                continue;
            }
            let rgb = [s[0], s[1], s[2]];
            compositor.paint(
                row,
                alpha_row.as_deref_mut(),
                dx,
                dy,
                transfer.map_or(rgb, |t| t.apply(rgb)),
                coverage,
            );
        }
    }
}

/// Composites image samples onto a page or group bitmap.
///
/// An image paints with the nonstroking constant alpha and the blend mode in
/// effect (ISO 32000-1 §11.6.4.4, §11.3.5), and inside a transparency group
/// its shape accumulates into the group's alpha plane like any other object.
/// When none of these apply (no alpha plane, `ca` = 1, `Normal`), a sample is
/// blended onto the destination by its coverage alone; otherwise every sample
/// runs through the raster pipe that fills use.
#[derive(Clone, Copy)]
struct ImageCompositor<'a> {
    /// `None` selects the direct coverage blend.
    pipe: Option<PipeState<'a>>,
}

impl<'a> ImageCompositor<'a> {
    fn new(
        bitmap: &Bitmap<Rgb8>,
        fill_alpha: u8,
        blend_mode: BlendMode,
        backdrop: Option<&'a Backdrop>,
    ) -> Self {
        let direct =
            bitmap.alpha_plane().is_none() && fill_alpha == 255 && blend_mode == BlendMode::Normal;
        Self {
            pipe: (!direct).then(|| PageRenderer::make_pipe(fill_alpha, blend_mode, backdrop)),
        }
    }

    /// Paint `src` at column `dx` of row `dy` with coverage `coverage`.
    /// `row` is the row's pixel bytes and `alpha_row` its alpha plane row,
    /// if the bitmap has one.  A column past the row's end paints nothing.
    fn paint(
        &self,
        row: &mut [u8],
        alpha_row: Option<&mut [u8]>,
        dx: u32,
        dy: u32,
        src: [u8; 3],
        coverage: u8,
    ) {
        let x = dx as usize;
        let Some(dst) = row.get_mut(x * 3..x * 3 + 3) else {
            return;
        };
        let Some(pipe) = &self.pipe else {
            blend_rgb(dst, src, coverage);
            return;
        };
        let pixel_pipe = PipeState {
            a_input: div255(u32::from(coverage) * u32::from(pipe.a_input)),
            ..*pipe
        };
        if pixel_pipe.a_input == 0 {
            return;
        }
        let dst_alpha = alpha_row.and_then(|a| a.get_mut(x..=x));
        #[expect(
            clippy::cast_possible_wrap,
            reason = "dx and dy are below the bitmap's u32 dimensions, which \
                      device bitmaps keep under 2^31"
        )]
        raster::pipe::render_span::<Rgb8>(
            &pixel_pipe,
            &PipeSrc::Solid(&src),
            dst,
            dst_alpha,
            None,
            dx as i32,
            dx as i32,
            dy as i32,
        );
    }
}

/// Paint `src` onto the RGB triple `dst` with coverage `alpha`; an opaque
/// sample copies, anything else blends per channel.
#[inline]
fn blend_rgb(dst: &mut [u8], src: [u8; 3], alpha: u8) {
    if alpha == 255 {
        dst.copy_from_slice(&src);
    } else {
        let a = u16::from(alpha);
        // Each output channel blends against its own existing value —
        // the destination is RGB and channels may differ.
        dst[0] = blend_u8(src[0], dst[0], a);
        dst[1] = blend_u8(src[1], dst[1], a);
        dst[2] = blend_u8(src[2], dst[2], a);
    }
}

// ── Component → colour helpers ────────────────────────────────────────────────

/// Map a raw component slice to a [`RasterColor`] by channel count.
pub(super) fn components_to_color(comps: &[f64]) -> RasterColor {
    match comps {
        [g] => RasterColor::gray(*g),
        [r, g, b] => RasterColor::rgb(*r, *g, *b),
        [c, m, y, k] => RasterColor::cmyk(*c, *m, *y, *k),
        _ => RasterColor::default(),
    }
}

/// Map a PDF line cap integer (0–2) to [`LineCap`]; defaults to `Butt`.
const fn int_to_cap(v: i32) -> LineCap {
    match v {
        1 => LineCap::Round,
        2 => LineCap::Projecting,
        _ => LineCap::Butt,
    }
}

/// Map a PDF line join integer (0–2) to [`LineJoin`]; defaults to `Miter`.
const fn int_to_join(v: i32) -> LineJoin {
    match v {
        1 => LineJoin::Round,
        2 => LineJoin::Bevel,
        _ => LineJoin::Miter,
    }
}

/// Porter-Duff source-over blend for a single 8-bit channel.
///
/// Formula: `(src * alpha + dst * (255 - alpha)) / 255`.
///
/// Maximum intermediate value: `255 * 255 = 65025 < u16::MAX`, so no overflow
/// is possible regardless of the alpha split.
#[expect(
    clippy::cast_possible_truncation,
    reason = "result is (x*a + y*ia)/255 where a+ia=255; max value is 255*255/255=255, always fits u8"
)]
fn blend_u8(src: u8, dst: u8, alpha: u16) -> u8 {
    let ia = 255_u16 - alpha;
    ((u16::from(src) * alpha + u16::from(dst) * ia) / 255) as u8
}

/// Reason `blit_image`'s decoder-contract check rejected an image.
#[derive(Debug, PartialEq, Eq)]
enum ImageLenError {
    /// `width × height × bpp` overflows `usize`.
    DimensionOverflow,
    /// Buffer is shorter than the declared dimensions require.
    ShortBuffer { expected: usize },
}

/// Validates the decoder contract `actual_len >= width × height × bpp` and
/// returns the expected length on success.  Pulled out of `blit_image` so the
/// two guard branches (overflow, short buffer) are unit-testable without
/// constructing a `PageRenderer`.
fn check_image_bytes_len(
    width: u32,
    height: u32,
    bpp: usize,
    actual_len: usize,
) -> Result<usize, ImageLenError> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(bpp))
        .ok_or(ImageLenError::DimensionOverflow)?;
    if actual_len < expected {
        return Err(ImageLenError::ShortBuffer { expected });
    }
    Ok(expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_draw_size_measures_the_transformed_unit_square() {
        assert_eq!(
            super::image_draw_size(&[200.0, 0.0, 0.0, 100.0, 5.0, 7.0]),
            Some((200, 100))
        );
        // Rotated 90°: the image's width lies along device y.
        assert_eq!(
            super::image_draw_size(&[0.0, 200.0, -100.0, 0.0, 0.0, 0.0]),
            Some((200, 100))
        );
        // A fractional extent rounds up.
        assert_eq!(
            super::image_draw_size(&[30.0, 40.0, 0.0, 10.4, 0.0, 0.0]),
            Some((50, 11))
        );
        // Flipped (negative scale) measures the same.
        assert_eq!(
            super::image_draw_size(&[-200.0, 0.0, 0.0, -100.0, 0.0, 0.0]),
            Some((200, 100))
        );
    }

    #[test]
    fn image_draw_size_rejects_degenerate_matrices() {
        assert_eq!(
            super::image_draw_size(&[0.0, 0.0, 0.0, 100.0, 0.0, 0.0]),
            None
        );
        assert_eq!(
            super::image_draw_size(&[f64::NAN, 0.0, 0.0, 1.0, 0.0, 0.0]),
            None
        );
        assert_eq!(
            super::image_draw_size(&[1e300, 0.0, 0.0, 1.0, 0.0, 0.0]),
            None
        );
    }

    fn diag_with_ppi(ppi: f32) -> PageDiagnostics {
        PageDiagnostics {
            has_images: true,
            source_ppi_hint: Some(ppi),
            ..PageDiagnostics::default()
        }
    }

    #[test]
    fn suggested_dpi_rounds_up_to_nearest_step() {
        // 67 PPI → next step is 72.
        assert_eq!(diag_with_ppi(67.0).suggested_dpi(72.0, 600.0), Some(72.0));
        // 73 PPI → next step is 96.
        assert_eq!(diag_with_ppi(73.0).suggested_dpi(72.0, 600.0), Some(96.0));
        // 150 PPI exactly → stays at 150.
        assert_eq!(diag_with_ppi(150.0).suggested_dpi(72.0, 600.0), Some(150.0));
        // 280 PPI → next step is 300.
        assert_eq!(diag_with_ppi(280.0).suggested_dpi(72.0, 600.0), Some(300.0));
    }

    #[test]
    fn suggested_dpi_respects_min_clamp() {
        // 67 PPI → step is 72, but min=150 clamps it up.
        assert_eq!(diag_with_ppi(67.0).suggested_dpi(150.0, 300.0), Some(150.0));
    }

    #[test]
    fn suggested_dpi_respects_max_clamp() {
        // 700 PPI → above all steps; last step is 600, then clamped to max=300.
        assert_eq!(diag_with_ppi(700.0).suggested_dpi(72.0, 300.0), Some(300.0));
    }

    #[test]
    fn suggested_dpi_none_when_no_images() {
        let diag = PageDiagnostics::default(); // source_ppi_hint is None
        assert_eq!(diag.suggested_dpi(150.0, 300.0), None);
    }

    // ── build_initial_ctm: per-/Rotate-branch + page-box origin ────────────
    //
    // Each expected matrix below is hand-computed from ISO 32000-2 §8.3.4 by
    // composing the box-at-origin scale+rotate with an innermost pre-translation
    // T(-llx, -lly).  The locked render-regression baseline only exercises
    // (0,0)-origin pages, so it cannot catch an origin-sign error; these tests
    // pin the sign for every rotation against an independent derivation.

    fn assert_ctm_eq(got: Ctm, want: Ctm) {
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() < 1e-9,
                "ctm[{i}]: got {g}, want {w} (full got {got:?}, want {want:?})"
            );
        }
    }

    #[test]
    fn initial_ctm_zero_origin_is_pixel_neutral_for_every_rotation() {
        // With llx = lly = 0 each branch must equal the pre-origin-fix matrix,
        // proving origin-at-zero PDFs (the overwhelming majority) are unchanged.
        let (wpx, hpx, s) = (600u32, 800u32, 2.0);
        let w = f64::from(wpx) / s;
        let h = f64::from(hpx) / s;
        assert_ctm_eq(
            build_initial_ctm(wpx, hpx, s, 0, 0.0, 0.0),
            [s, 0.0, 0.0, s, 0.0, 0.0],
        );
        assert_ctm_eq(
            build_initial_ctm(wpx, hpx, s, 90, 0.0, 0.0),
            [0.0, -s, s, 0.0, 0.0, h * s],
        );
        assert_ctm_eq(
            build_initial_ctm(wpx, hpx, s, 180, 0.0, 0.0),
            [-s, 0.0, 0.0, -s, w * s, h * s],
        );
        assert_ctm_eq(
            build_initial_ctm(wpx, hpx, s, 270, 0.0, 0.0),
            [0.0, s, -s, 0.0, w * s, 0.0],
        );
    }

    #[test]
    fn initial_ctm_nonzero_origin_pretranslates_per_rotation() {
        // CropBox like the lecouteux repro: non-zero lower-left origin.
        let (wpx, hpx, s) = (794u32, 1286u32, 2.0);
        let (llx, lly) = (462.0_f64, 23.0_f64);
        let w = f64::from(wpx) / s;
        let h = f64::from(hpx) / s;

        assert_ctm_eq(
            build_initial_ctm(wpx, hpx, s, 0, llx, lly),
            [s, 0.0, 0.0, s, -llx * s, -lly * s],
        );
        assert_ctm_eq(
            build_initial_ctm(wpx, hpx, s, 90, llx, lly),
            [0.0, -s, s, 0.0, -lly * s, (h + llx) * s],
        );
        assert_ctm_eq(
            build_initial_ctm(wpx, hpx, s, 180, llx, lly),
            [-s, 0.0, 0.0, -s, (w + llx) * s, (h + lly) * s],
        );
        assert_ctm_eq(
            build_initial_ctm(wpx, hpx, s, 270, llx, lly),
            [0.0, s, -s, 0.0, (w + lly) * s, -llx * s],
        );
    }

    /// Page geometry input for `assert_box_corners_map_to`.
    ///
    /// `wpx`/`hpx` are the output bitmap pixel dimensions, already swapped for
    /// 90°/270° exactly as the renderer's caller swaps them before `new_scaled`.
    struct CornerCase {
        wpx: u32,
        hpx: u32,
        s: f64,
        rotate: u16,
        llx: f64,
        lly: f64,
    }

    /// Expected device-pixel landing of each selected-box corner (origin
    /// top-left): lower-left, lower-right, upper-left, upper-right in PDF user
    /// space.  Named fields make a transposed corner a compile-readable error
    /// rather than a silent geometry mismatch.
    struct ExpectedCorners {
        ll: (f64, f64),
        lr: (f64, f64),
        ul: (f64, f64),
        ur: (f64, f64),
    }

    /// Map the four selected-box corners through `build_initial_ctm` + the
    /// `to_device` Y-flip and assert each lands at its expected device pixel.
    ///
    /// Expected positions are an independent cross-check: they match both
    /// `pdftoppm` and `mutool` (verified on identical fixtures), neither of
    /// which shares code with this renderer.
    fn assert_box_corners_map_to(case: &CornerCase, want: &ExpectedCorners) {
        let CornerCase {
            wpx,
            hpx,
            s,
            rotate,
            llx,
            lly,
        } = *case;
        let ctm = build_initial_ctm(wpx, hpx, s, rotate, llx, lly);
        // For 0°/180° the box is wpx×hpx; for 90°/270° the caller swapped the
        // bitmap dims, so the box in points is hpx×wpx.  Resolve box pts from
        // the unrotated orientation.
        let (bw, bh) = if rotate % 180 == 0 {
            (f64::from(wpx) / s, f64::from(hpx) / s)
        } else {
            (f64::from(hpx) / s, f64::from(wpx) / s)
        };
        let page_h = f64::from(hpx);
        let to_dev = |x: f64, y: f64| {
            let (dx, dy) = ctm_transform(&ctm, x, y);
            (dx, page_h - dy)
        };
        let corners = [
            ("LL", (llx, lly), want.ll),
            ("LR", (llx + bw, lly), want.lr),
            ("UL", (llx, lly + bh), want.ul),
            ("UR", (llx + bw, lly + bh), want.ur),
        ];
        for (name, (x, y), (wx, wy)) in corners {
            let (px, py) = to_dev(x, y);
            assert!(
                (px - wx).abs() < 1e-9 && (py - wy).abs() < 1e-9,
                "rotate {rotate} {name}: got ({px}, {py}), want ({wx}, {wy})"
            );
        }
    }

    #[test]
    fn initial_ctm_rotate0_maps_box_corners_to_device_pixels() {
        // Unrotated, non-(0,0)-origin box: lower-left → device bottom-left,
        // upper-right → device top-right.  The sign check the locked
        // (0,0)-origin baseline structurally cannot perform.
        assert_box_corners_map_to(
            &CornerCase {
                wpx: 400,
                hpx: 600,
                s: 2.0,
                rotate: 0,
                llx: 50.0,
                lly: 30.0,
            },
            &ExpectedCorners {
                ll: (0.0, 600.0),   // → bottom-left
                lr: (400.0, 600.0), // → bottom-right
                ul: (0.0, 0.0),     // → top-left
                ur: (400.0, 0.0),   // → top-right
            },
        );
    }

    #[test]
    fn initial_ctm_rotate90_maps_box_corners_to_device_pixels() {
        // Caller swaps pixel dims for 90°: bitmap is 600×400.
        assert_box_corners_map_to(
            &CornerCase {
                wpx: 600,
                hpx: 400,
                s: 2.0,
                rotate: 90,
                llx: 50.0,
                lly: 30.0,
            },
            &ExpectedCorners {
                ll: (0.0, 0.0),     // → top-left
                lr: (0.0, 400.0),   // → bottom-left
                ul: (600.0, 0.0),   // → top-right
                ur: (600.0, 400.0), // → bottom-right
            },
        );
    }

    #[test]
    fn initial_ctm_rotate180_maps_box_corners_to_device_pixels() {
        assert_box_corners_map_to(
            &CornerCase {
                wpx: 400,
                hpx: 600,
                s: 2.0,
                rotate: 180,
                llx: 50.0,
                lly: 30.0,
            },
            &ExpectedCorners {
                ll: (400.0, 0.0),   // → top-right
                lr: (0.0, 0.0),     // → top-left
                ul: (400.0, 600.0), // → bottom-right
                ur: (0.0, 600.0),   // → bottom-left
            },
        );
    }

    #[test]
    fn initial_ctm_rotate270_maps_box_corners_to_device_pixels() {
        // Caller swaps pixel dims for 270°: bitmap is 600×400.
        assert_box_corners_map_to(
            &CornerCase {
                wpx: 600,
                hpx: 400,
                s: 2.0,
                rotate: 270,
                llx: 50.0,
                lly: 30.0,
            },
            &ExpectedCorners {
                ll: (600.0, 400.0), // → bottom-right
                lr: (600.0, 0.0),   // → top-right
                ul: (0.0, 400.0),   // → bottom-left
                ur: (0.0, 0.0),     // → top-left
            },
        );
    }

    #[test]
    #[should_panic(expected = "rotate_cw must be normalised")]
    fn initial_ctm_non_spec_rotation_trips_debug_assert() {
        // Callers must normalise /Rotate to {0,90,180,270}; a non-spec value is a
        // contract violation and must fail loudly in debug rather than silently
        // selecting a rotation arm and rendering a wrong-but-plausible page.
        let _ = build_initial_ctm(100, 100, 1.0, 45, 0.0, 0.0);
    }

    #[test]
    fn initial_ctm_non_spec_rotation_release_falls_back_to_unrotated() {
        // Release builds (debug_assertions off) must not panic on a stray value:
        // the fallback is the unrotated matrix, so the page renders upright.
        // Only meaningful without debug assertions; debug builds panic (above).
        if cfg!(debug_assertions) {
            return;
        }
        assert_ctm_eq(
            build_initial_ctm(100, 100, 1.0, 45, 0.0, 0.0),
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        );
    }

    // ── check_image_bytes_len ──────────────────────────────────────────────

    #[test]
    fn check_image_bytes_len_accepts_exact() {
        // 2 × 2 × Rgb (bpp=3) → expected 12 bytes.
        assert_eq!(check_image_bytes_len(2, 2, 3, 12), Ok(12));
    }

    #[test]
    fn check_image_bytes_len_accepts_over_long_buffer() {
        // Over-long buffers are fine — only short buffers are rejected.
        assert_eq!(check_image_bytes_len(2, 2, 3, 15), Ok(12));
    }

    #[test]
    fn check_image_bytes_len_rejects_short_buffer() {
        assert_eq!(
            check_image_bytes_len(2, 2, 3, 6),
            Err(ImageLenError::ShortBuffer { expected: 12 }),
        );
    }

    #[test]
    fn check_image_bytes_len_rejects_dimension_overflow() {
        // u32::MAX × u32::MAX × 3 overflows usize on 64-bit targets.
        assert_eq!(
            check_image_bytes_len(u32::MAX, u32::MAX, 3, 0),
            Err(ImageLenError::DimensionOverflow),
        );
    }

    #[test]
    fn check_image_bytes_len_one_by_one_gray() {
        // 1 × 1 × Gray (bpp=1) → expected 1 byte.
        assert_eq!(check_image_bytes_len(1, 1, 1, 1), Ok(1));
    }

    // ── blend_u8 ──────────────────────────────────────────────────────────────

    #[test]
    fn blend_u8_opaque_returns_src() {
        assert_eq!(blend_u8(200, 50, 255), 200);
        assert_eq!(blend_u8(0, 255, 255), 0);
        assert_eq!(blend_u8(255, 0, 255), 255);
    }

    #[test]
    fn blend_u8_transparent_returns_dst() {
        // alpha=0 is short-circuited by the caller (continue), but the function
        // itself must still be well-defined (returns dst).
        assert_eq!(blend_u8(200, 50, 0), 50);
    }

    #[test]
    fn blend_u8_50pct_midpoints_correctly() {
        // src=255, dst=0, alpha=128 → (255*128 + 0*127)/255 = 128.
        assert_eq!(blend_u8(255, 0, 128), 128);
        // src=0, dst=255, alpha=128 → (0*128 + 255*127)/255 = 127.
        assert_eq!(blend_u8(0, 255, 128), 127);
    }

    #[test]
    fn blend_u8_no_overflow_at_max_values() {
        // Worst-case intermediate: 255*254 + 255*1 = 65025, fits u16.
        assert_eq!(blend_u8(255, 255, 254), 255);
        assert_eq!(blend_u8(255, 255, 1), 255);
    }

    // ── per-page work watchdog ────────────────────────────────────────────────

    /// Verify the operator-count cap fires: a tiny `op_budget` is tripped,
    /// fewer than all ops run, and `budget_status()` reports the breach.
    #[test]
    fn watchdog_op_budget_fires() {
        use crate::content::Operator;
        use crate::test_helpers::empty_doc;

        let doc = empty_doc();
        // Construct a minimal renderer; (1,1) = 1-pixel white page at 72 dpi.
        // empty_doc has no pages, so we use object id (1,0) (the catalog) as
        // the page_id — no real page resources are needed because EndPath is a
        // pure graphics-state no-op that touches no resources.
        let mut renderer = PageRenderer::new(1, 1, &doc, (1, 0));

        // Override op_budget to a small value so the test terminates instantly.
        renderer.set_op_budget_for_test(10);
        renderer.set_deadline_for_test(None); // disable wall-clock cap; only op count fires

        // Build a vector of 50 no-op EndPath operators — well over the budget.
        let ops: Vec<Operator> = std::iter::repeat_n(Operator::EndPath, 50).collect();
        renderer.execute(&ops);

        assert!(
            renderer.budget_status().is_some(),
            "budget_status() must be Some after exceeding op_budget"
        );
        assert!(
            renderer.ops_executed_for_test() <= 11, // budget=10 → fires at op 11
            "no more than budget+1 ops should have run, got {}",
            renderer.ops_executed_for_test()
        );
    }

    /// Verify the wall-clock deadline fires when set to an already-elapsed instant.
    #[test]
    fn watchdog_deadline_fires() {
        use crate::content::Operator;
        use crate::test_helpers::empty_doc;

        let doc = empty_doc();
        let mut renderer = PageRenderer::new(1, 1, &doc, (1, 0));

        // Set an already-elapsed deadline and a very large op budget so the
        // deadline is the only thing that can fire.
        renderer.set_op_budget_for_test(u64::MAX);
        renderer.set_deadline_for_test(Some(
            std::time::Instant::now()
                .checked_sub(std::time::Duration::from_secs(1))
                .unwrap(),
        ));

        // Build exactly DEADLINE_CHECK_INTERVAL no-op ops so the periodic check
        // is guaranteed to run at least once.
        let ops: Vec<Operator> = std::iter::repeat_n(
            Operator::EndPath,
            usize::try_from(DEADLINE_CHECK_INTERVAL).unwrap(),
        )
        .collect();
        renderer.execute(&ops);

        assert!(
            renderer.budget_status().is_some(),
            "budget_status() must be Some after deadline is exceeded"
        );
    }

    /// The page work-budget MUST be shared with — not reset for — a child
    /// renderer (tiling-pattern tile).  A fresh `PageRenderer` would otherwise
    /// get its own full budget and a fresh deadline, letting a pathological
    /// pattern content stream escape the per-page watchdog entirely.  This
    /// pins both halves of the contract: `adopt_parent_budget` carries the
    /// parent's *remaining* allowance + exact deadline (and inherits an
    /// already-tripped breach), and `fold_child_into` rolls the child's
    /// consumption and any breach back into the parent.
    /// An image skipped inside a child renderer (a tiling-pattern tile) is
    /// reported on the page that owns the tile.
    #[test]
    fn child_image_failures_fold_into_parent() {
        use crate::test_helpers::empty_doc;

        let doc = empty_doc();
        let mut parent = PageRenderer::new(1, 1, &doc, (1, 0));
        parent.record_image_failure(Some("Im0".to_owned()), "parent".to_owned());
        let mut child = PageRenderer::new(1, 1, &doc, (1, 0));
        child.record_image_failure(None, "child".to_owned());

        child.fold_child_into(&mut parent);

        let reasons: Vec<_> = parent
            .finish()
            .1
            .image_failures
            .into_iter()
            .map(|f| (f.name, f.reason))
            .collect();
        assert_eq!(
            reasons,
            [
                (Some("Im0".to_owned()), "parent".to_owned()),
                (None, "child".to_owned())
            ]
        );
    }

    #[test]
    fn watchdog_budget_shared_with_child_renderer() {
        use crate::content::Operator;
        use crate::test_helpers::empty_doc;

        let doc = empty_doc();

        // Parent with a 100-op budget that has already spent 90 ops.
        let mut parent = PageRenderer::new(1, 1, &doc, (1, 0));
        parent.set_op_budget_for_test(100);
        parent.set_deadline_for_test(None);
        let ops90: Vec<Operator> = std::iter::repeat_n(Operator::EndPath, 90).collect();
        parent.execute(&ops90);
        assert!(parent.budget_status().is_none(), "90 < 100 must not trip");
        assert_eq!(parent.ops_executed_for_test(), 90);

        // Child adopts the *remaining* 10-op allowance, not a fresh 100.
        let mut child = PageRenderer::new(1, 1, &doc, (1, 0));
        child.adopt_parent_budget(&parent);
        assert_eq!(
            child.op_budget, 10,
            "child must inherit remaining allowance"
        );
        assert_eq!(child.ops_executed_for_test(), 0);
        assert!(
            child.deadline.is_none(),
            "child must inherit exact deadline"
        );

        // Child runs 50 ops — well over its 10-op share — and trips.
        let ops50: Vec<Operator> = std::iter::repeat_n(Operator::EndPath, 50).collect();
        child.execute(&ops50);
        assert!(
            child.budget_status().is_some(),
            "child must trip on the shared remaining budget"
        );

        // Folding back: parent's counter advances by the child's work and the
        // breach propagates so the parent's execute() bails loudly next.
        child.fold_child_into(&mut parent);
        assert!(
            parent.ops_executed_for_test() >= 90 + 11,
            "parent must account the child's work, got {}",
            parent.ops_executed_for_test()
        );
        assert!(
            parent.budget_status().is_some(),
            "a child breach must propagate to the parent"
        );

        // An already-tripped parent must hand the child a zero allowance so it
        // does no work at all.
        let mut child2 = PageRenderer::new(1, 1, &doc, (1, 0));
        child2.adopt_parent_budget(&parent);
        assert!(
            child2.budget_status().is_some(),
            "child must inherit an already-tripped parent breach"
        );
        let one: Vec<Operator> = vec![Operator::EndPath];
        child2.execute(&one);
        assert_eq!(
            child2.ops_executed_for_test(),
            0,
            "a child of a tripped parent must execute zero ops"
        );
    }

    // ── unsupported-operator warn-once dedup ──────────────────────────────────

    /// A hostile content stream with far more distinct junk operators than the
    /// cap must not grow `warned_unknown_ops` without bound: the set is capped
    /// at `MAX_WARNED_UNKNOWN_OPS` and the cap-reached flag is set (so the
    /// summary warning fires exactly once).  Exact repeats of a tracked keyword
    /// must not inflate the set.
    #[test]
    fn unknown_op_dedup_set_is_bounded_under_adversarial_input() {
        use crate::content::Operator;
        use crate::test_helpers::empty_doc;

        let doc = empty_doc();
        let mut renderer = PageRenderer::new(1, 1, &doc, (1, 0));
        renderer.set_op_budget_for_test(u64::MAX);
        renderer.set_deadline_for_test(None);

        // Ten times the cap of *distinct* junk keywords, each repeated twice to
        // also exercise the dedup-hit (no-allocation) path.
        let distinct = MAX_WARNED_UNKNOWN_OPS * 10;
        let mut ops: Vec<Operator> = Vec::with_capacity(distinct * 2);
        for i in 0..distinct {
            let kw = format!("junk{i}").into_bytes();
            ops.push(Operator::Unknown(kw.clone()));
            ops.push(Operator::Unknown(kw));
        }
        renderer.execute(&ops);

        let (tracked, capped) = renderer.warned_unknown_ops_state_for_test();
        assert_eq!(
            tracked, MAX_WARNED_UNKNOWN_OPS,
            "warned set must be hard-capped, got {tracked}"
        );
        assert!(
            capped,
            "cap-reached flag must be set so the summary warning fires"
        );
    }

    /// The first occurrence of every distinct unsupported operator (below the
    /// cap) is tracked — proving the first-occurrence WARN still surfaces and
    /// the dedup never swallows the signal that content may be incomplete.
    #[test]
    fn unknown_op_first_occurrence_is_tracked_per_keyword() {
        use crate::content::Operator;
        use crate::test_helpers::empty_doc;

        let doc = empty_doc();
        let mut renderer = PageRenderer::new(1, 1, &doc, (1, 0));

        // Three distinct keywords, the first repeated three times.
        renderer.execute(&[
            Operator::Unknown(b"foo".to_vec()),
            Operator::Unknown(b"foo".to_vec()),
            Operator::Unknown(b"foo".to_vec()),
            Operator::Unknown(b"bar".to_vec()),
            Operator::Unknown(b"baz".to_vec()),
        ]);

        let (tracked, capped) = renderer.warned_unknown_ops_state_for_test();
        assert_eq!(tracked, 3, "one tracked entry per distinct keyword");
        assert!(!capped, "cap must not trip below MAX_WARNED_UNKNOWN_OPS");
    }

    /// Each page gets a fresh `PageRenderer`, so the dedup set is inherently
    /// per-page.  Verify a freshly constructed renderer starts with an empty
    /// set (no cross-page leakage of suppression state).
    #[test]
    fn unknown_op_dedup_state_is_per_renderer() {
        use crate::test_helpers::empty_doc;

        let doc = empty_doc();
        let renderer = PageRenderer::new(1, 1, &doc, (1, 0));
        assert_eq!(
            renderer.warned_unknown_ops_state_for_test(),
            (0, false),
            "a new per-page renderer must start with an empty, uncapped dedup set"
        );
    }

    // ── Image clipping ────────────────────────────────────────────────────────

    /// Run `content` on a `size × size` white page at 72 dpi and return the
    /// red channel of every pixel, row-major.  The page has no resources, so
    /// the content must paint through inline images only.
    fn render_gray(size: u32, content: &[u8]) -> Vec<u8> {
        render_gray_on(&crate::test_helpers::empty_doc(), (1, 0), size, content)
    }

    /// [`render_gray`] on page `page_id` of `doc`, whose resources the
    /// content may name.
    fn render_gray_on(doc: &Document, page_id: ObjectId, size: u32, content: &[u8]) -> Vec<u8> {
        let mut renderer = PageRenderer::new(size, size, doc, page_id);
        renderer.execute(&crate::content::parse(content));
        let stride = renderer.bitmap.stride;
        let data = renderer.bitmap.data();
        (0..size)
            .flat_map(|y| (0..size).map(move |x| (x, y)))
            .map(|(x, y)| data[y as usize * stride + x as usize * 3])
            .collect()
    }

    /// A 2×2 all-black `DeviceGray` inline image stretched over the whole
    /// 8 pt page.  Without a clip it would blacken every pixel.
    const BLACK_IMAGE_OVER_PAGE: &[u8] =
        b"q 8 0 0 8 0 0 cm BI /W 2 /H 2 /CS /G /BPC 8 ID \x00\x00\x00\x00 EI Q";

    #[test]
    fn image_blit_honours_rectangular_clip() {
        let content = [
            b"q 0 0 4 8 re W n ".as_slice(),
            BLACK_IMAGE_OVER_PAGE,
            b" Q",
        ]
        .concat();
        let px = render_gray(8, &content);
        for y in 0..8 {
            for x in 0..8 {
                let want = if x < 4 { 0 } else { 255 };
                assert_eq!(px[y * 8 + x], want, "pixel ({x}, {y})");
            }
        }
    }

    /// An L-shaped path clip over a black image: the image shows through the
    /// L only.  Column 4 of the top half lies on the clip's vertical edge and
    /// carries the clip's antialiased coverage (4 of 16 sub-samples, 223
    /// after the gamma table), exactly like a fill through the same clip.
    #[test]
    fn image_blit_honours_path_clip() {
        let content = [
            b"q 0 0 m 8 0 l 8 4 l 4 4 l 4 8 l 0 8 l h W n ".as_slice(),
            BLACK_IMAGE_OVER_PAGE,
            b" Q",
        ]
        .concat();
        let px = render_gray(8, &content);
        for y in 0..8 {
            for x in 0..8 {
                let want = match (y < 4, x) {
                    (true, 4) => 223,
                    (true, 5..) => 255,
                    _ => 0,
                };
                assert_eq!(px[y * 8 + x], want, "pixel ({x}, {y})");
            }
        }
    }

    /// The line width is a user-space distance: under a 4× CTM a 1-unit
    /// horizontal stroke covers 4 device rows, centred on the path.
    #[test]
    fn stroke_width_scales_with_ctm() {
        let px = render_gray(16, b"4 0 0 4 0 0 cm 0 G 1 w 0 2 m 4 2 l S");
        let dark: Vec<usize> = (0..16).filter(|&y| px[y * 16 + 8] < 128).collect();
        assert_eq!(dark, [6, 7, 8, 9], "stroke rows at x = 8");
    }

    // ── Colorant /None ────────────────────────────────────────────────────────

    /// Page `(4, 0)` naming `/N` = `[/Separation /None …]`, `/A` =
    /// `[/Separation /All …]`, `/NN` = `[/DeviceN [/None /None] …]` and
    /// `/NC` = `[/DeviceN [/None /Cyan] …]`, all over `DeviceGray`, where
    /// a tint of 1 in the first colorant is black.
    fn none_colorant_doc() -> Document {
        const PS: &str = "{ pop 1 exch sub }";
        crate::test_helpers::make_doc_with_objects(&[
            (
                4,
                "<< /Type /Page /Resources << /ColorSpace << \
                 /N [/Separation /None /DeviceGray 5 0 R] \
                 /A [/Separation /All /DeviceGray 5 0 R] \
                 /NN [/DeviceN [/None /None] /DeviceGray 6 0 R] \
                 /NC [/DeviceN [/None /Cyan] /DeviceGray 6 0 R] >> >> >>"
                    .to_owned(),
            ),
            (
                5,
                "<< /FunctionType 2 /Domain [0 1] /C0 [1] /C1 [0] /N 1 >>".to_owned(),
            ),
            (
                6,
                format!(
                    "<< /FunctionType 4 /Domain [0 1 0 1] /Range [0 1] /Length {} >>\n\
                     stream\n{PS}\nendstream",
                    PS.len()
                ),
            ),
        ])
    }

    fn render_none(content: &[u8]) -> Vec<u8> {
        render_gray_on(&none_colorant_doc(), (4, 0), 8, content)
    }

    #[test]
    fn separation_none_paints_nothing() {
        let fill = render_none(b"/N cs 1 scn 0 0 8 8 re f");
        assert!(fill.iter().all(|&v| v == 255), "fill painted: {fill:?}");
        let stroke = render_none(b"/N CS 1 SCN 4 w 0 4 m 8 4 l S");
        assert!(
            stroke.iter().all(|&v| v == 255),
            "stroke painted: {stroke:?}"
        );
        let stencil =
            render_none(b"/N cs 1 scn q 8 0 0 8 0 0 cm BI /W 1 /H 1 /IM true /BPC 1 ID \x00 EI Q");
        assert!(
            stencil.iter().all(|&v| v == 255),
            "stencil painted: {stencil:?}"
        );
    }

    #[test]
    fn separation_all_paints_its_tint() {
        let px = render_none(b"/A cs 1 scn 0 0 8 8 re f");
        assert!(px.iter().all(|&v| v == 0), "got {px:?}");
    }

    #[test]
    fn device_n_paints_nothing_only_when_every_colorant_is_none() {
        let none = render_none(b"/NN cs 1 0 scn 0 0 8 8 re f");
        assert!(none.iter().all(|&v| v == 255), "got {none:?}");
        let cyan = render_none(b"/NC cs 1 0 scn 0 0 8 8 re f");
        assert!(cyan.iter().all(|&v| v == 0), "got {cyan:?}");
    }

    /// `g` and `G` select `DeviceGray` (PDF §8.6.8), leaving the `/None` space.
    #[test]
    fn device_colour_operators_leave_a_none_space() {
        let fill = render_none(b"/N cs 1 scn 0 g 0 0 8 8 re f");
        assert!(fill.iter().all(|&v| v == 0), "got {fill:?}");
        let stroke = render_none(b"/N CS 1 SCN 0 G 8 w 0 4 m 8 4 l S");
        assert!(stroke.iter().all(|&v| v == 0), "got {stroke:?}");
    }

    // ── Default colour spaces ─────────────────────────────────────────────────

    /// Page `(4, 0)` whose `/ColorSpace` resources hold `/DefaultGray` =
    /// `default_gray`.
    fn default_gray_doc(default_gray: &str) -> Document {
        crate::test_helpers::make_doc_with_objects(&[(
            4,
            format!(
                "<< /Type /Page /Resources << /ColorSpace << /DefaultGray {default_gray} >> >> >>"
            ),
        )])
    }

    /// A linear (`/Gamma 1`) D65 `CalGray` space: a value of 0.5 is half
    /// luminance, 188 after sRGB encoding, where `DeviceGray` 0.5 is 128.
    const LINEAR_CAL_GRAY: &str = "[/CalGray << /WhitePoint [0.9505 1 1.089] /Gamma 1 >>]";

    /// `DefaultGray` replaces `DeviceGray` whether it is selected by `cs`,
    /// `CS`, `g` or `G` (PDF §8.6.5.6).
    #[test]
    fn default_gray_replaces_every_selection_of_device_gray() {
        let doc = default_gray_doc(LINEAR_CAL_GRAY);
        for content in [
            b"0.5 g 0 0 8 8 re f".as_slice(),
            b"/DeviceGray cs 0.5 sc 0 0 8 8 re f",
            b"0.5 G 8 w 0 4 m 8 4 l S",
            b"/DeviceGray CS 0.5 SC 8 w 0 4 m 8 4 l S",
        ] {
            let px = render_gray_on(&doc, (4, 0), 8, content);
            assert!(
                px.iter().all(|v| (186..=189).contains(v)),
                "{}: {px:?}",
                String::from_utf8_lossy(content)
            );
        }
    }

    /// A default space of the wrong component count leaves the device space
    /// in effect.
    #[test]
    fn default_space_of_another_component_count_is_ignored() {
        let doc = default_gray_doc("[/CalRGB << /WhitePoint [0.9505 1 1.089] >>]");
        let px = render_gray_on(&doc, (4, 0), 8, b"0.5 g 0 0 8 8 re f");
        assert!(px.iter().all(|&v| v == 128), "got {px:?}");
    }

    // ── Transfer functions ────────────────────────────────────────────────────

    /// Page `(4, 0)` with `ExtGState` `/Inv` = `<< /TR 5 0 R >>` and `/Id` =
    /// `<< /TR /Identity >>`, where object 5 inverts its input.
    fn transfer_doc() -> Document {
        const PS: &str = "{ 1 exch sub }";
        crate::test_helpers::make_doc_with_objects(&[
            (
                4,
                "<< /Type /Page /Resources << /ExtGState << \
                 /Inv << /TR 5 0 R >> /Id << /TR /Identity >> >> >> >>"
                    .to_owned(),
            ),
            (
                5,
                format!(
                    "<< /FunctionType 4 /Domain [0 1] /Range [0 1] /Length {} >>\n\
                     stream\n{PS}\nendstream",
                    PS.len()
                ),
            ),
        ])
    }

    fn render_transfer(content: &[u8]) -> Vec<u8> {
        render_gray_on(&transfer_doc(), (4, 0), 8, content)
    }

    /// Fills, strokes and image samples map through the transfer function
    /// (PDF §10.5): 0.2 gray (51) paints 204.
    #[test]
    fn transfer_function_maps_painted_colours() {
        for content in [
            b"/Inv gs 0.2 g 0 0 8 8 re f".as_slice(),
            b"/Inv gs 0.2 G 8 w 0 4 m 8 4 l S",
            b"/Inv gs q 8 0 0 8 0 0 cm BI /W 1 /H 1 /CS /G /BPC 8 ID \x33 EI Q",
        ] {
            let px = render_transfer(content);
            assert!(
                px.iter().all(|&v| v == 204),
                "{}: {px:?}",
                String::from_utf8_lossy(content)
            );
        }
    }

    /// The transfer function is graphics state: `Q` and `/TR /Identity`
    /// both end it.
    #[test]
    fn transfer_function_is_restored_and_reset() {
        let restored = render_transfer(b"q /Inv gs Q 0.2 g 0 0 8 8 re f");
        assert!(restored.iter().all(|&v| v == 51), "got {restored:?}");
        let reset = render_transfer(b"/Inv gs /Id gs 0.2 g 0 0 8 8 re f");
        assert!(reset.iter().all(|&v| v == 51), "got {reset:?}");
    }

    /// The transfer function maps the object's colour, not the composited
    /// result: black under an inverting transfer paints white everywhere,
    /// including the partially covered edge pixels of a fill.
    #[test]
    fn transfer_function_leaves_the_backdrop_alone() {
        let px = render_transfer(b"/Inv gs 0 g 0.5 0.5 6.3 6.3 re f");
        assert!(px.iter().all(|&v| v == 255), "got {px:?}");
    }

    /// A stroke thinner than a device pixel is widened to one pixel: a
    /// 0.1-unit rule centred in a pixel row paints that row solid.
    #[test]
    fn thin_stroke_widens_to_one_device_pixel() {
        let px = render_gray(16, b"0 G 0.1 w 0 4.5 m 16 4.5 l S");
        let column: Vec<u8> = (0..16).map(|y| px[y * 16 + 8]).collect();
        assert_eq!(column[11], 0, "rule row solid, column {column:?}");
        assert!(
            column
                .iter()
                .enumerate()
                .all(|(y, &g)| (10..=12).contains(&y) || g == 255),
            "no paint beyond the rule's neighbour rows, column {column:?}"
        );
    }

    /// An image whose bounding box misses the clip entirely paints nothing.
    #[test]
    fn image_blit_outside_clip_paints_nothing() {
        let content = [
            b"q 0 0 1 1 re W n q 4 0 0 4 4 4 cm ".as_slice(),
            b"BI /W 2 /H 2 /CS /G /BPC 8 ID \x00\x00\x00\x00 EI Q Q",
        ]
        .concat();
        let px = render_gray(8, &content);
        assert!(px.iter().all(|&g| g == 255), "no pixel may be painted");
    }
}
