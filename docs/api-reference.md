# API Reference

## `rasterrocket` crate — public API

The `rasterrocket` crate is the integration entry point. It wraps `rasterrocket-interp` and `rasterrocket-render` behind a simple, stable API.

**Extensibility.** These public types are `#[non_exhaustive]`, so fields and variants can be added in a minor release: `RasterError`, `BackendPolicy`, `RenderPurpose`, `PageBox`, `FormType`, `UnsupportedFeature`, `ScriptLocation`, `MediaKind`, `DevicePage`, `ImageFilter`, `PixelFormat`, `GcvError`, `deskew::DeskewError` (match them with a `_` arm); `RenderedPage`, `PageInfo`, `PageDiagnostics`, `ImageFailure`, `ContentFailure`, `PageGeometry`, `GroupSpaceApproximations`, `GcvImage` (outputs: read their fields, do not construct them); `GlyphCounts` and `DevicePageCounts` (outputs); `ColorModel`, `OutputProfileError` and `OutputIntentError` (match them with a `_` arm); `RasterOptions`, `SessionConfig`, `RenderParams`, `GcvBudget` (inputs: start from `RasterOptions::new(dpi)`, `SessionConfig::with_policy`, `RenderParams::new`, or `Default`, then assign fields). `RenderRegion` is a plain rectangle, `Rotation` a closed set of quarter turns and `InvalidRotation` carries the rejected degrees; all three stay exhaustive, as do `Pixels`, `SampleDepth`, `FinishedCanvas`, `BandBitmap`, `Band`, `BandError`, `CanvasLayout`, `RenderingIntent`, `ProofIntent`, `IntentSet`, `OutputSpace`, `PreserveBlack` and `WorkingSpace`. `BackendPolicy::name()` is the inverse of its `FromStr`.

---

### `raster_pdf`

```rust
pub fn raster_pdf(
    path: &Path,
    opts: &RasterOptions,
) -> impl Iterator<Item = (u32, Result<RenderedPage, RasterError>)>
```

Renders a range of pages from a PDF file. Returns an iterator that yields `(page_num, result)` for each page in `opts.first_page..=opts.last_page`.

**Behaviour:**

- Pages are rendered in ascending order.
- A per-page error does not abort remaining pages. The caller decides whether to skip or propagate.
- If `opts.last_page` exceeds the document's page count, rendering stops at the last page — no error is returned for the overshoot.
- GPU resources are initialised lazily on first use and reused across pages.

**Errors (yielded as iterator items):**

- `RasterError::InvalidOptions` — `opts` violates constraints (e.g. `dpi ≤ 0`, `first_page > last_page`), or, in a `vulkan` build, `PDF_RASTER_VULKAN_DEVICE` matches no single usable device (in a Windows `d3d12` build under `PDF_RASTER_BACKEND=d3d12`, `PDF_RASTER_D3D12_DEVICE` no single usable adapter). Yielded as `(1, Err(...))`, iterator ends immediately.
- `RasterError::Pdf` — document cannot be opened or parsed. Yielded as `(1, Err(...))`, iterator ends immediately.
- `RasterError::BackendUnavailable` — `PDF_RASTER_BACKEND` names a backend (`cuda`, `vaapi`, `vulkan`, `d3d12`) that cannot be initialised. Yielded as `(1, Err(...))`, iterator ends immediately.
- `RasterError::PageOutOfRange` — page number exceeds document length.
- `RasterError::PageDegenerate` — page has zero pixel dimensions.
- `RasterError::PageTooLarge` / `RasterError::PageExceedsCeiling` — the bitmap exceeds the byte budget (`SessionConfig::max_page_bytes`, default 1.8 GB), or a side exceeds `MAX_PX_SIDE` (2²⁴).
- `RasterError::PageAllocationFailed` — the page is within those limits, but the allocator refused its bitmap, or a bitmap drawn on it: a transparency group, soft mask, knockout coverage, tiling-pattern cell or screen, or Type 3 glyph bitmap, or a snapshot of the canvas.
- `RasterError::InvalidPageGeometry` — `UserUnit` outside `[0.1, 10.0]`.
- `RasterError::ImageDecodeFailed` / `RasterError::ContentDecodeFailed` / `RasterError::ShadingFailed` — only when `opts.strict` is set; see [`RasterOptions`](#rasteroptions).
- `RasterError::PageBudgetExceeded` — the page exceeded the per-page operator budget or wall-clock deadline.
- `RasterError::Deskew` — deskew was asked of a page with a side over `deskew::DESKEW_MAX_SIDE`. A GPU rotation that fails under `BackendPolicy::ForceCuda` is `RasterError::BackendUnavailable`.

A panic while rendering a page propagates out of `Iterator::next`; this iterator does not catch panics.

---

### `raster_pdf_from_bytes`

```rust
pub fn raster_pdf_from_bytes(
    bytes: Vec<u8>,
    opts: &RasterOptions,
) -> impl Iterator<Item = (u32, Result<RenderedPage, RasterError>)>
```

Identical to [`raster_pdf`](#raster_pdf) but renders a PDF held entirely in
memory — no file path, no temp file. For callers that already hold the document
bytes (e.g. a PDF extracted from an archive). Backend selection, page ordering,
and the per-page error contract are the same as `raster_pdf`.

An encrypted document is handled exactly as by `raster_pdf`: decrypted in
process when `opts.decrypt_authorized` is set, with `opts.password` when the
empty user password does not open it.

---

### `render_channel`

```rust
pub fn render_channel(
    path: &Path,
    opts: &RasterOptions,
    capacity: usize,
) -> std::sync::mpsc::Receiver<(u32, Result<RenderedPage, RasterError>)>
```

Renders pages concurrently in a background Rayon task, streaming results through a bounded sync channel.

**`capacity`** — maximum number of rendered pages buffered before the producer blocks (natural backpressure). `capacity = 0` is silently raised to `1`. Use `2`–`8` for typical OCR pipelines.

**Error contract:**

- Options validation failure → `(1, Err(RasterError::InvalidOptions(...)))`, channel closes.
- File open failure → `(1, Err(RasterError::Pdf(...)))`, channel closes.
- Forced backend unavailable (e.g. `PDF_RASTER_BACKEND=cuda` with no CUDA driver) → `(1, Err(RasterError::BackendUnavailable(...)))`, channel closes.
- Per-page failures → `(page_num, Err(...))`, remaining pages continue.
- Per-page panic → `(page_num, Err(RasterError::RenderPanic { .. }))` and rendering continues, **only** in builds with `panic = "unwind"` (test profiles, or an embedder that overrides the workspace's `panic = "abort"` release profile). Under `panic = "abort"` a panicking page aborts the process.

If the `Receiver` is dropped before the producer finishes, the producer exits cleanly on its next `send` — no panic.

---

### Session API

Lower-level functions at the crate root for explicit control over PDF opening and per-page rendering: `open_session`, `open_session_from_bytes`, `open_session_from_shared`, `open_session_from_reader`, `prescan_session`, `render_page_gray`, `render_page_rgb`, `render_page_rgb_with`, `render_page_rgb_with_diagnostics`, `render_page_rgba`, `render_page_gray_alpha`, `render_page_cmyk`, `render_page_canvas`, `render_page_canvas_with_diagnostics`, `render_page_bands`, `rgb_to_gray`, `rgb_into_gray`, `rgb16_to_gray16` and `rgb16_into_gray16`; and [`PageBuffer`](#pagebuffer) for rendering into buffers the caller owns.

```rust
pub fn open_session(path: &Path, config: &SessionConfig) -> Result<RasterSession, RasterError>
pub fn open_session_from_bytes(bytes: Vec<u8>, config: &SessionConfig) -> Result<RasterSession, RasterError>
pub fn open_session_from_shared(
    bytes: impl AsRef<[u8]> + Send + Sync + 'static,
    config: &SessionConfig,
) -> Result<RasterSession, RasterError>
pub fn open_session_from_reader(reader: impl std::io::Read, config: &SessionConfig) -> Result<RasterSession, RasterError>
```

`open_session_from_bytes` is the in-memory counterpart to `open_session` (the session-level analogue of `raster_pdf_from_bytes`): same behaviour, taking owned PDF bytes instead of a path.

The parser reads the document as one contiguous slice, indexed by the cross-reference offsets: a file is memory-mapped (from 256 KiB; a shorter one is read, and so is any file when `config.read_into_memory` is set), owned bytes are held as they are, and `open_session_from_shared` holds any `AsRef<[u8]>` storage the caller shares (`Arc<[u8]>`, `&'static [u8]`, a reference-counted byte buffer type such as `bytes::Bytes`, a mapping the caller made) and reads it in place, never copying it. `bytes.as_ref()` must return the same bytes on every call. `open_session_from_reader` reads a `Read` source (a socket, an archive member) to its end into memory and opens that; a `Seek` source gains nothing, since nothing is parsed before the whole document is in memory, and a file on disk is better opened by path. The read is unbounded, so bound an untrusted reader with `Read::take(max_bytes)`: the session holds the whole document, and a source longer than the bound is opened as a truncated document. A reader's I/O error is `RasterError::Pdf`. Every source renders the same document to identical pixels.

A mapped file shares the operating system's page cache, so the session's bytes are only as stable as the file: if another process truncates the file while the session is alive, the next read of a mapped page past the new end of file raises `SIGBUS`, which terminates the whole process instead of failing the render, and a write to the file changes bytes the cross-reference offsets already index. A long-running process rendering files it does not control sets `SessionConfig::read_into_memory` (or `RasterOptions::read_into_memory`): the whole file is read at open into a buffer the session owns, at the cost of the file's size in memory, and later changes to the file do not reach it. A file whose buffer cannot be allocated is `RasterError::Pdf` wrapping an I/O error of kind `OutOfMemory` that names the size. `rrocket` maps: it renders one file and exits.

An encrypted document (Standard security handler, revisions 2–6: RC4 40–128-bit, AES-128, AES-256, crypt filters) is refused with `RasterError::Pdf` unless `config.decrypt_authorized` is `true`; with it, the document is decrypted in process — no subprocess, no temporary file. The empty user password is tried first (a permissions-only document needs none), then `config.password` as the user and as the owner password; a wrong password, or none where one is needed, is a `RasterError::Pdf` saying which. A document using another security handler (public-key encryption, `/Adobe.PubSec`) is refused by name.

Opens the PDF and reads the page count from the page-tree root (`/Pages /Count`); no page object is read until that page is rendered. Also initialises the shared CUDA context (for the `gpu-icc` or `cache` features; never under `Auto`) according to `config.policy`, and, with the `cache` feature, the device image cache on the backend the session resolved to (the Vulkan or Direct3D 12 backend when it holds one, otherwise the CUDA context). Errors with `RasterError::Pdf` if the file is unreadable or corrupt. A JavaScript-bearing PDF is **not** an error — it opens and renders normally; a loud `WARN` is emitted per detected catalog JS entry point (see `open` below), the first rendered page carrying page or annotation JavaScript emits one more, and no `/JS` is ever decoded or executed. Errors with `RasterError::BackendUnavailable` if `config.policy` is `ForceCuda`, `ForceVaapi`, `ForceVulkan` or `ForceD3d12` and the required backend cannot be initialised (for `ForceVulkan`, also when the crate was built without the `vulkan` feature; for `ForceD3d12`, without the `d3d12` feature or on a platform other than Windows; for `ForceCuda`, when it was built without `gpu-icc`, `cache`, `gpu-deskew` and `nvjpeg2k`; a `gpu-deskew`-only build opens no CUDA context with the session, and a GPU rotation that then fails is the page's `BackendUnavailable`). Errors with `RasterError::InvalidOptions` if `config.gpu_pages` is set without a backend that renders pages (a policy other than `Auto` or `ForceVulkan` in a `vulkan` build, or `ForceD3d12` in a Windows `d3d12` build) or with `vector_antialias` off; see [`SessionConfig`](#sessionconfig).

```rust
pub fn render_page_rgb(
    session: &RasterSession,
    page_num: u32,
    scale: f64,
) -> Result<Bitmap<Rgb8>, RasterError>
```

Renders one whole page to an RGB bitmap. `Bitmap` is `rasterrocket_render::Bitmap` and `Rgb8` / `Gray8` are `rasterrocket_color` types; `rasterrocket` does not re-export them, so code that names them depends on those crates too. `scale` is the pixel-per-point multiplier, `dpi / 72.0`. Equivalent to `render_page_rgb_with(session, page_num, &RenderParams::new(scale))`; for different horizontal and vertical resolutions use `RenderParams::new_xy`.

Safe to call from multiple Rayon threads concurrently. GPU image decoders are managed per-thread via `thread_local!`.

```rust
pub fn render_page_gray(
    session: &RasterSession,
    page_num: u32,
    opts: &RasterOptions,
) -> Result<RenderedPage, RasterError>
```

Renders one page to the grayscale `RenderedPage` that `raster_pdf` yields for it (`Gray8`, or `Gray16` at `opts.depth` 16). These `opts` fields apply: `opts.dpi`, `opts.depth`, `opts.deskew`, `opts.intent`, `opts.black_point_compensation`, `opts.preserve_black` and `opts.cancel`; the page-selection fields are ignored, and the fields mirrored on `SessionConfig` were fixed when the session was opened. `opts.transparent_background` is not consulted: `render_page_rgba` is its counterpart.

```rust
pub fn render_page_rgba(
    session: &RasterSession,
    page_num: u32,
    opts: &RasterOptions,
) -> Result<RenderedPage, RasterError>
```

Renders one page over a transparent backdrop to the `PixelFormat::Rgba8` (`Rgba16` at depth 16) `RenderedPage` that `raster_pdf` yields when `opts.transparent_background` is set: `opts.dpi`, `opts.depth`, `opts.intent`, `opts.black_point_compensation`, `opts.preserve_black` and `opts.cancel` apply; `opts.deskew` is `RasterError::InvalidOptions`.

`render_page_gray_alpha(session, page_num, opts)` renders the `PixelFormat::GrayAlpha8` (`GrayAlpha16` at depth 16) page of `opts.gray_alpha`: `[Y, A]`, the gray `rgb_to_gray` gives of the RGBA page's straight colour (the device gray of a gray output profile), the alpha the RGBA page's. Where the alpha is full the gray is `render_page_gray`'s. The `opts` fields that apply are `render_page_rgba`'s. An RGB output profile and `opts.deskew` are `RasterError::InvalidOptions`.

```rust
pub fn render_page_cmyk(
    session: &RasterSession,
    page_num: u32,
    opts: &RasterOptions,
) -> Result<RenderedPage, RasterError>
```

Renders one page to the `PixelFormat::Cmyk8` (or, at `opts.depth` 16, `Cmyk16`) `RenderedPage` that `raster_pdf` yields when `opts.color` is `ColorModel::Cmyk` (see **CMYK pages** below): `opts.dpi`, `opts.depth`, `opts.intent`, `opts.black_point_compensation`, `opts.preserve_black` and `opts.cancel` apply; `opts.deskew`, `opts.transparent_background` and a session output profile that is not CMYK are `RasterError::InvalidOptions`.

```rust
pub fn render_page_rgb_with(
    session: &RasterSession,
    page_num: u32,
    params: &RenderParams,
) -> Result<Bitmap<Rgb8>, RasterError>

#[non_exhaustive]
pub struct RenderParams {
    pub scale_x: f64,                    // x_dpi / 72
    pub scale_y: f64,                    // y_dpi / 72
    pub region: Option<RenderRegion>,    // None = whole page
    pub policy: Option<BackendPolicy>,   // None = the session's
    pub cancel: Option<CancelToken>,     // None = render to completion
    pub transparent_background: bool,    // false = composited onto white
    pub band_rows: Option<NonZeroU32>,   // None = as SessionConfig::band_bytes hold
    pub depth: SampleDepth,              // Eight (default) or Sixteen
    pub intent: Option<RenderingIntent>, // None = the page's one intent, else relative
    pub black_point_compensation: bool,  // false
    pub preserve_black: Option<PreserveBlack>, // None = colorimetric CMYK → CMYK
    pub color: ColorModel,               // Rgb (default) or Cmyk
}
#[non_exhaustive]
pub enum ColorModel { Rgb, Cmyk }
impl RenderParams {
    pub const fn new(scale: f64) -> Self                     // same scale on both axes
    pub const fn new_xy(scale_x: f64, scale_y: f64) -> Self  // non-square pixels
    pub const fn with_region(self, region: RenderRegion) -> Self
    pub const fn with_policy(self, policy: BackendPolicy) -> Self
    pub fn with_cancel(self, cancel: CancelToken) -> Self
    pub const fn with_transparent_background(self) -> Self
    pub const fn with_depth(self, depth: SampleDepth) -> Self
    pub const fn with_intent(self, intent: RenderingIntent) -> Self
    pub const fn with_black_point_compensation(self) -> Self
    pub const fn with_preserve_black(self, mode: PreserveBlack) -> Self
    pub const fn with_cmyk(self) -> Self
    pub const fn with_band_rows(self, rows: NonZeroU32) -> Self
    pub const fn canvas_layout(&self) -> CanvasLayout  // bytes a canvas pixel holds
}

pub struct RenderRegion { pub x: u32, pub y: u32, pub width: u32, pub height: u32 }
impl RenderRegion {
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Self
}

#[derive(Clone, Default)]
pub struct CancelToken { /* Arc<AtomicBool> */ }
impl CancelToken {
    pub fn new() -> Self
    pub fn cancel(&self)
    pub fn is_cancelled(&self) -> bool
}
```

Renders page `page_num` as `params` describes:

- **`scale_x` / `scale_y`** are device pixels per point along the bitmap's axes, applied after `/Rotate`. Unequal scales give non-square pixels (300 × 150 dpi is `new_xy(300.0 / 72.0, 150.0 / 72.0)`); the page is `PageGeometry::size_px(scale_x, scale_y)` pixels.
- **`region`** draws only a window of the page, in device pixels counted from the page's top-left corner at those scales. The bitmap is the region's size, and it holds exactly the pixels the whole-page render places there, so regions that tile a page stitch into the whole-page bitmap byte for byte. The byte budget (`SessionConfig::max_page_bytes`) applies to the region, so a page too large to render whole (a map or poster at high DPI) can be rendered in tiles; see also `render_page_bands`. An empty region, or one extending past the page, is `RasterError::InvalidOptions`.
- **`policy`** overrides the session's backend policy for this page's image decoders; pass `BackendPolicy::CpuOnly` for a page `prescan_session` reports without images to skip GPU decoder setup.
- **`cancel`**: after `CancelToken::cancel()` on any clone, the render stops before its next content-stream operator (form, Type 3 glyph and pattern-cell content included) and returns `RasterError::Cancelled`; a render started after cancellation returns it without drawing. An operator already running (one image decode, one fill) completes first.

```rust
pub fn render_page_rgb_with_diagnostics(
    session: &RasterSession,
    page_num: u32,
    params: &RenderParams,
) -> Result<(Bitmap<Rgb8>, PageDiagnostics), RasterError>
```

`render_page_rgb_with`, also returning the [`PageDiagnostics`](#pagediagnostics) the render collected; same errors.

```rust
pub fn render_page_bands<E>(
    session: &RasterSession,
    page_num: u32,
    params: &RenderParams,
    sink: impl FnMut(Band<'_>) -> Result<(), E>,
) -> Result<PageDiagnostics, BandError<E>>

pub struct Band<'a> { pub y: u32, pub bitmap: BandBitmap<'a> }
pub enum BandBitmap<'a> {
    Rgb8(&'a Bitmap<Rgb8>), Rgb16(&'a Bitmap<Rgb16>),
    Gray8(&'a Bitmap<Gray8>), Gray16(&'a Bitmap<Gray16>),    // a gray output profile
    Cmyk8(&'a Bitmap<Cmyk8>), Cmyk16(&'a Bitmap<Cmyk16>),    // ColorModel::Cmyk
}
impl BandBitmap<'_> {
    pub const fn of(canvas: &FinishedCanvas) -> BandBitmap<'_>  // a whole canvas as one band
    pub const fn dims(self) -> (u32, u32)
    pub const fn height(self) -> u32
}
pub enum BandError<E> { Raster(RasterError), Sink(E) }

pub fn render_page_canvas(session: &RasterSession, page_num: u32, params: &RenderParams)
    -> Result<FinishedCanvas, RasterError>
pub fn render_page_canvas_with_diagnostics(session: &RasterSession, page_num: u32, params: &RenderParams)
    -> Result<(FinishedCanvas, PageDiagnostics), RasterError>
pub enum FinishedCanvas {                                     // re-exported from rasterrocket-interp
    Rgb8(Bitmap<Rgb8>), Rgb16(Bitmap<Rgb16>), Gray8(Bitmap<Gray8>),
    Gray16(Bitmap<Gray16>), Cmyk8(Bitmap<Cmyk8>), Cmyk16(Bitmap<Cmyk16>),
}
impl FinishedCanvas {
    pub const fn dims(&self) -> (u32, u32)
    pub const fn depth(&self) -> SampleDepth
    pub const fn is_cmyk(&self) -> bool
}

pub struct OutputProfile { /* the ICC bytes, their space and compiled conversions */ }
impl OutputProfile {
    pub fn from_bytes(bytes: impl Into<Arc<[u8]>>) -> Result<Self, OutputProfileError>
    pub const fn bytes(&self) -> &Arc<[u8]>
    pub const fn space(&self) -> OutputSpace                 // Rgb, Gray or Cmyk
    pub fn prepare(&self, space: WorkingSpace, intent: RenderingIntent, bpc: bool) -> Result<(), OutputProfileError>
}
#[non_exhaustive]
pub enum OutputProfileError { Unreadable(String), Class(&'static str), ColorSpace(String), Pcs,
    Transform { .. }, Conversion(String), Mismatch(OutputSpace) }
pub enum RenderingIntent { RelativeColorimetric, AbsoluteColorimetric, Perceptual, Saturation }
pub enum ProofIntent { RelativeColorimetric, AbsoluteColorimetric }  // rendering_intent, simulates_paper
#[non_exhaustive]
pub enum OutputIntentError { Missing, NoProfile { subtype, condition, external },
    Unreadable { subtype, reason }, NotCmyk { subtype, space } }
pub struct IntentSet { /* a set of RenderingIntent */ }      // contains, is_empty, single, iter, union
```

Renders the page (or `params.region`) as horizontal bands of whole rows from the top, handing each to `sink` in order, so a page larger than memory is drawn a band at a time. Stacked, the bands are the bitmap `render_page_rgb_with` returns for the same `params`, byte for byte. A band holds `params.band_rows` rows (`RenderParams::with_band_rows`), else as many as `SessionConfig::band_bytes` hold. The content is parsed once and executed per band; the bands share the page's decoded images, forms, shadings and tiling cells, its tint budget, its spot-ink slots and its diagnostics, and each has its own operator budget and deadline. A page of one band renders as `render_page_rgb_with` does (on the GPU page device when the session has one); several bands render on the CPU. The byte budget bounds the page the bands make up. Errors are `render_page_rgb_with`'s as `BandError::Raster` (under `strict`, an image failure fails the band where it is met), or the sink's as `BandError::Sink`, after which no band is rendered.

**16 bits per sample.** `RenderParams::with_depth(SampleDepth::Sixteen)` renders the page at 16 bits per channel: colour conversions, shadings, soft masks, transparency groups, blend modes, overprint simulation and images of more than 8 bits keep their levels. `render_page_canvas(session, page_num, &params) -> Result<FinishedCanvas, RasterError>` (and `render_page_canvas_with_diagnostics`) returns the page at the depth asked for, `FinishedCanvas::Rgb8(Bitmap<Rgb8>)` or `FinishedCanvas::Rgb16(Bitmap<Rgb16>)`; `render_page_rgb*` serve 8-bit renders and refuse a 16-bit one with `RasterError::InvalidOptions` naming `render_page_canvas`. `render_page_bands` hands each band as `Band { y, bitmap: BandBitmap::Rgb8(..) | BandBitmap::Rgb16(..) }`. A 16-bit canvas pixel holds 6 bytes (8 over a transparent backdrop) of the page budget (`CanvasLayout::bytes_per_px`). The 8-bit path is unchanged, byte for byte. `PageBuffer` is 8-bit only (`InvalidOptions` at depth 16); deskew runs at either depth (see the `deskew` module).

**CMYK pages.** `RenderParams::with_cmyk()` (`color: ColorModel::Cmyk`) renders the page in the process inks of a CMYK device on a canvas of four process and four spot inks: `render_page_canvas` returns `FinishedCanvas::Cmyk8(Bitmap<Cmyk8>)` or `Cmyk16(Bitmap<Cmyk16>)`, `render_page_bands` `BandBitmap::Cmyk8` / `Cmyk16` bands (`BandBitmap::of(&canvas)` views a whole canvas as a band), samples `[C, M, Y, K]`, 0 no ink. `DeviceCMYK` and `ICCBased` CMYK colours, images and shadings and the process colorants of `Separation` / `DeviceN` colours keep their values (`ICCBased` CMYK whose profile's tags differ from the session's CMYK output profile's converts to its ink: the source's `AToB` and the output profile's `BToA` tables evaluated in floating point, with the absolute-colorimetric white scale or the ISO 18619 black-point map between them, tabulated per source profile, intent and compensation on a 17⁴ grid and interpolated in 4-simplices; within ΔE00 1 of Little CMS); an `Indexed` image over a space of ink paints its palette's ink; the first four spot colorants of the page are painted as spot inks and folded into their process equivalents at the end; other colours separate through the session's CMYK output profile, else by the ISO 32000-2 §10.4.2 device conversion under the graphics state's black generation and undercolour removal (`BG`, `UCR`; gray is black ink alone). Overprint is simulated when `SessionConfig::overprint` is set and knocked out otherwise. A CMYK canvas pixel holds 8 bytes (16 at 16 bits) of the page budget; `CanvasLayout { depth, alpha, ink }` counts the ink canvas of a CMYK page and of a page the session paints on the overprint preview. Refused (`RasterError::InvalidOptions`): `with_transparent_background`, an RGB or gray output profile, and `render_page_rgb*`, `render_page_gray` and `render_page_rgba`, which return RGB or gray. `RasterOptions::color` selects the same for `raster_pdf` and `render_channel`, yielding `PixelFormat::Cmyk8` / `Cmyk16` pages.

**Soft proofs.** `SessionConfig::proof: Option<ProofIntent>` (mirrored by `RasterOptions::proof`) shows every page as it prints on the press its output intent names (ISO 32000-2 §14.11.5): the page's own `/OutputIntents`, else the catalog's; of the array, the first intent of subtype `GTS_PDFX` that embeds a `/DestOutputProfile`, else the first of any subtype that does. The profile must be CMYK. `open_session` reads the document's intent and returns `RasterError::OutputIntent` when it has none (and no page has its own: `OutputIntentError::Missing`), when no intent embeds a profile or one refers to an external one (`NoProfile`, `external`), for a profile that does not parse (`Unreadable`), and for one that is not CMYK (`NotCmyk`, naming the subtype and the space: a PDF/A sRGB intent describes a display, so there is nothing to proof against), and `RasterError::InvalidOptions` for a CMYK `output_profile`. A page whose own intent fails fails with the same error. Each page is painted as a CMYK page is (see **CMYK pages**), its colours that are not ink separated through the press profile under `RenderParams::intent` (else each object's own intent) and `black_point_compensation`, overprint simulated under `SessionConfig::overprint`, and its ink is then converted to sRGB, or to an RGB or gray `output_profile` (the display), under the `ProofIntent` without compensation: `RelativeColorimetric` maps the paper to white, `AbsoluteColorimetric` reproduces the paper's white and the press's black as measured, which needs sRGB or a matrix/TRC display profile (a table-based one is `RasterError::OutputProfile` on the first page). The pages are those of the display: `FinishedCanvas::Rgb8` / `Rgb16` (gray for a gray display), `PixelFormat::Gray8` / `Rgb*` through `raster_pdf`. The ink-to-display conversion is tabulated on a 33⁴ grid (7.1 MB) per press and proof intent, on the first page that needs it, and interpolated in the 4-simplex of each pixel; it matches Little CMS's soft proof within ΔE00 1 at 16 bits and 1.2 at 8 bits. A proofed page's canvas is the ink canvas, counted by the page budget. Refused per page (`RasterError::InvalidOptions`): `with_cmyk` and `with_transparent_background`. `PageDiagnostics::output_intent` is `RenderParams::intent`, else the page's one content intent, else relative colorimetric. In `rasterrocket-interp`, `document_output_intent` / `page_output_intent` choose the `OutputIntent`, `OutputIntent::press_profile` reads its press and `SoftProof::new(press, display)` / `convert(canvas, intent, buffers)` is the conversion.

**Output profiles.** `SessionConfig::output_profile` holds an `OutputProfile` (`OutputProfile::from_bytes(bytes) -> Result<OutputProfile, OutputProfileError>`): an RGB, gray or CMYK ICC profile of the output, display or colour-space class with an XYZ or Lab PCS, table-based (`AToB`/`BToA`) or matrix/TRC. `from_bytes` refuses anything else: `ColorSpace` for another data colour space, `Class` for a device-link, abstract, named-colour or input profile, `Pcs`, and `Unreadable` for bytes that do not parse. Each page, or each band of `render_page_bands`, is converted to it once composited: RGB pages to the profile's RGB device values, in place; gray profiles yield `FinishedCanvas::Gray8` / `Gray16` and `BandBitmap::Gray8` / `Gray16`. The alpha plane is unchanged, colour 0 where alpha is 0. The conversion runs on 16-bit samples at both depths and is compiled once per intent and compensation, on first use (`OutputProfile::prepare(space, intent, bpc)` compiles it ahead for the `WorkingSpace` `space`). Its intent is `RenderParams::intent` when given, else the one intent the page's marking operators used (`ri`, `ExtGState` `/RI`, an image's `/Intent`; `PageDiagnostics::content_intents`), else relative colorimetric; a page of mixed intents converts relative colorimetric and the session's first such page logs a warning. `PageDiagnostics::output_intent` records the intent used. `RenderParams::with_black_point_compensation` asks for ISO 18619 black-point compensation; a perceptual or saturation conversion to a v4 profile always compensates. Absolute colorimetry scales to the profile's media white. The conversion matches Little CMS within ΔE00 1. An intent or compensation without a profile, and compensation under absolute colorimetry, are `RasterError::InvalidOptions`; a conversion the profile cannot make fails the page with `RasterError::OutputProfile`. A gray profile makes `render_page_rgb*` and `render_page_rgba` `InvalidOptions` (use `render_page_canvas`, `render_page_bands`, `render_page_gray` or `render_page_gray_alpha`), an RGB one `render_page_gray` and `render_page_gray_alpha`; `PageBuffer` refuses a session with a profile. `OutputProfile::bytes()` is what the encoders' `IccProfile::new` takes to tag files (`rasterrocket-encode`: `write_png`, `PngWriter`, `TiffStream`, `TiffPageEncoder`, `write_jpeg`, `PdfWriter`). A CMYK profile (`OutputSpace::Cmyk`) converts no finished page: it separates CMYK pages as they are painted, through `OutputProfile::separation(&profile, space, (intent, bpc), preserve) -> Result<Separation, OutputProfileError>` (`preserve` the `Option<PreserveBlack>` of `RenderParams::preserve_black`) (`Separation::Profile(Arc<PressSeparation>)`): each object separates under `RenderParams::intent` when given, else under its own intent (`ri`, `ExtGState` `/RI`, an image's `/Intent`), through `OutputProfile::separation_table(space, intent, bpc)`, a 65³ table of the profile's 16-bit conversion from the working space built once per intent and compensation on first use (`PressSeparation::table(intent)`); the table of `RenderParams::intent`, else of relative colorimetry, is built before the page is painted, and an intent whose table cannot be built later separates through it, with a warning. `DeviceCMYK` and other process colours pass through unconverted. An sRGB page under a CMYK profile, and a CMYK page under an RGB or gray one, are `RasterError::InvalidOptions` (`OutputProfileError::Mismatch` from `convert` and `separation`). In `rasterrocket-encode`, `write_jpeg` takes `Cmyk8` bitmaps (Adobe YCCK, `APP14` transform 2), and `RowImage::Cmyk` / `Cmyk16` with `RowLayout::Cmyk` / `Cmyk16` write `Separated` TIFF (`InkSet` 1); `IccProfile` accepts CMYK profiles.

**Working space.** `SessionConfig::working_space: Option<WorkingSpace>` is the RGB space pages are composited in before the conversion to the output profile. `None` (the default) is automatic: `WorkingSpace::Wide` for a session with an `output_profile`, `WorkingSpace::Srgb` otherwise. `Wide` has the ROMM RGB primaries, the D50 white and the sRGB tone curve, so colours outside sRGB (wide-gamut images, saturated `Lab` and CMYK colours) reach the output conversion unclipped; `DeviceRGB`, `DeviceCMYK` and the other device colours keep their sRGB meaning, converted into it through one 16-bit table. Blending, and transparency groups that declare no colour space of their own, composite in the working space; transfer functions map its components. A wide page is painted at 16 bits at either `RenderParams::depth`, an 8-bit page rounded to 8 bits after the output conversion; pages painted on the overprint preview (ink canvas) are composited in sRGB. `Some(Srgb)` composites every page in sRGB at the render's depth. `Some(Wide)` without an output profile is `RasterError::InvalidOptions` from `open_session`, and `SessionConfig::check_working_space()` applies the same check before a document is opened. `raster_pdf`, `raster_pdf_from_bytes` and `render_channel` use the automatic choice. A wide page draws no image from the device image cache, which holds 8-bit samples, and `gpu_pages` renders it on the CPU.

```rust
pub fn rgb_to_gray(src: &Bitmap<Rgb8>) -> Bitmap<Gray8>
```

`DeviceRGB` → `DeviceGray` conversion with the weights of ISO 32000-1 §10.3.3, `Y = 0.30·R + 0.59·G + 0.11·B`, in 15-bit fixed point with round-half-up (`(9830·R + 19333·G + 3605·B + 16384) >> 15`), so `R = G = B` maps to itself.

```rust
pub fn rgb_into_gray(src: Bitmap<Rgb8>) -> Bitmap<Gray8>
```

The same conversion, byte for byte, in the RGB bitmap's own allocation: the gray rows are written packed (`stride == width`) over the front of the colour buffer, which keeps its capacity; an alpha plane is kept, as `rgb_to_gray` copies it (every gray conversion keeps its source's alpha, the gray not premultiplied by it). No second buffer is allocated, so a gray render peaks at the memory of the RGB render. `raster_pdf`, `render_channel`, `render_page_gray`, the comic decoder and the CLI's `--gray` / `--mono` use it. `rgb16_into_gray16(src: Bitmap<Rgb16>) -> Bitmap<Gray16>` is the same at 16 bits, sample for sample `rgb16_to_gray16`. `rasterrocket_render::Bitmap::converted::<Q>()` copies a bitmap to the depth of `Q`'s samples: widened exactly from 8 bits (`v · 257`), narrowed to the nearest 8-bit level. `rasterrocket_render::Bitmap::into_data` (the colour buffer, `stride × height` bytes) and `Bitmap::from_packed(width, height, data)` (a packed bitmap over a buffer of exactly `width × height × bytes-per-pixel`; panics otherwise) move a buffer out of and into a bitmap without a copy; `Bitmap::with_alpha_plane(alpha)` puts an alpha plane of `width × height` samples (or none) in place of a bitmap's own, and `Bitmap::heap_bytes()` counts the bytes of its colour, alpha and painted-channels buffers. `rasterrocket_render::interleave_alpha::<S, N>(colour, alpha, out)` interleaves pixels of `N` samples with their alpha (opaque for `None`) into pixels of `N + 1`, and `push_with_alpha::<S, N>` appends them to a `Vec`; they replace `interleave_rgba`, which is `interleave_alpha::<u8, 3>`.

```rust
pub fn prescan_session(
    session: &RasterSession,
    page_num: u32,
) -> Result<PageDiagnostics, RasterError>
```

Classifies page `page_num` without rendering any pixels. Returns `PageDiagnostics` with image/text presence, dominant filter, and a PPI hint. Use this before `render_page_rgb` to choose the backend policy (e.g. skip GPU init for vector-only pages).

Errors with `RasterError::PageOutOfRange` or `RasterError::InvalidPageGeometry` if the page is invalid.

---

### `PageBuffer`

```rust
#[derive(Debug, Default)]
pub struct PageBuffer { /* the page canvas, kept between renders */ }

impl PageBuffer {
    pub fn new() -> Self
    pub fn render_gray_into(&mut self, session: &RasterSession, page_num: u32,
        opts: &RasterOptions, out: &mut [u8], stride: usize) -> Result<PageInfo, RasterError>
    pub fn render_rgb_into(&mut self, session: &RasterSession, page_num: u32,
        opts: &RasterOptions, out: &mut [u8], stride: usize) -> Result<PageInfo, RasterError>
    pub fn render_rgba_into(&mut self, session: &RasterSession, page_num: u32,
        opts: &RasterOptions, out: &mut [u8], stride: usize) -> Result<PageInfo, RasterError>
    pub fn render_gray_alpha_into(&mut self, session: &RasterSession, page_num: u32,
        opts: &RasterOptions, out: &mut [u8], stride: usize) -> Result<PageInfo, RasterError>
}

#[derive(Debug)]
#[non_exhaustive]
pub struct PageInfo {
    pub page_num: u32,
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,     // Gray8, Rgb8, Rgba8 or GrayAlpha8
    pub dpi: f32,
    pub diagnostics: PageDiagnostics,
    pub skew: Option<deskew::Skew>,  // as RenderedPage::skew
}
```

Renders whole pages at `opts.dpi` into memory the caller owns. Every page is drawn on a canvas the size of the page: RGB, or ink when the session simulates overprint on a page that sets it, with an alpha plane when the page is painted as a page group. A `PageBuffer` keeps the allocations of both canvases and of the alpha plane between renders and writes the finished page into `out`, row `y` at `out[y * stride..]`, in the method's format. Once they have grown to the largest page rendered through them, a render allocates no page-sized buffer: neither canvas, alpha plane nor output. Allocations of the drawing itself (decoded images, glyphs, transparency groups, the skew profile of `deskew`) are unchanged, and so are the two page-sized copies a GPU rotation makes under the `gpu-deskew` feature.

| Method | Pixels written | `opts` fields used |
|---|---|---|
| `render_gray_into` | those of `render_page_gray`, one byte per pixel; with `deskew` the canvas is converted to gray in place and rotated straight into `out` | `dpi`, `deskew`, `intent`, `black_point_compensation`, `preserve_black`, `cancel` |
| `render_rgb_into` | those of `render_page_rgb` at `dpi / 72`, `[R, G, B]` | `dpi`, `intent`, `black_point_compensation`, `preserve_black`, `cancel`; `deskew` is `InvalidOptions` |
| `render_rgba_into` | those of `render_page_rgba`, `[R, G, B, A]` straight alpha over a transparent backdrop | as `render_rgb_into` |
| `render_gray_alpha_into` | those of `render_page_gray_alpha`, `[Y, A]`, the gray not premultiplied, over a transparent backdrop | as `render_rgb_into` |

`stride` is in bytes and at least `width * format.bytes_per_pixel()`; `out` holds at least `(height - 1) * stride + width * bytes_per_pixel` bytes (the last row needs no padding); bytes between rows are left alone. A shorter stride is `RasterError::OutputStrideTooSmall`, a shorter buffer `RasterError::OutputBufferTooSmall`, both returned before anything is drawn and with `out` untouched. `RasterSession::page_size_px(page_num, dpi)` gives `(width, height)` beforehand. The methods take `&mut self`, so a thread rendering in parallel keeps its own `PageBuffer`; a render that fails while the page is drawn (cancelled, over its budget or deadline, refused a bitmap the drawing needs, or a strict session's failure) keeps the canvas for the next render, which paints it anew; one whose page canvas or finished bitmap cannot be allocated releases it.

```rust
let mut buffer = PageBuffer::new();
let mut out = Vec::new();
for page in 1..=session.total_pages() {
    let (w, h) = session.page_size_px(page, opts.dpi)?;
    out.resize(w as usize * h as usize, 0);   // grows to the largest page, then stays
    let info = buffer.render_gray_into(&session, page, &opts, &mut out, w as usize)?;
    // out[..w * h] holds the page
}
```

Measured with a counting global allocator (`examples/page_buffer_allocs.rs`), 20 pages at 150 dpi, gray, warm session caches, CPU, against `render_page_gray`:

| Fixture | `render_page_gray` bytes / page | page-sized allocations / page | ms / page | `PageBuffer` bytes / page | page-sized allocations / page | ms / page |
|---|---|---|---|---|---|---|
| corpus-01 (native text) | 8.89 MB | 2.00 | 1.79 | 0.47 MB | 0.00 | 1.14 |
| corpus-04 (ebook, mixed) | 14.16 MB | 2.20 | 4.58 | 5.47 MB | 0.20 | 4.43 |
| corpus-06 (DCT layout) | 39.52 MB | 6.20 | 15.3 | 30.82 MB | 4.20 | 15.2 |
| corpus-08 (DCT scan) | 41.40 MB | 2.00 | 43.5 | 12.48 MB | 0.00 | 43.9 |

The remaining page-sized allocations are decoded images; the allocation count falls by three per page, out of the thousands the drawing makes.

---

### `RasterSession`

```rust
pub struct RasterSession { /* opaque */ }

impl RasterSession {
    pub const fn total_pages(&self) -> u32
    pub const fn policy(&self) -> BackendPolicy
    pub const fn form_type(&self) -> FormType
    pub fn unsupported_features(&self) -> &[UnsupportedFeature]
    pub const fn output_profile(&self) -> Option<&Arc<OutputProfile>>
    pub fn page_geometry(&self, page_num: u32) -> Result<PageGeometry, RasterError>
    pub fn page_size_px(&self, page_num: u32, dpi: f32) -> Result<(u32, u32), RasterError>
    #[cfg(feature = "cache")]
    pub const fn image_cache(&self) -> Option<&gpu::cache::ImageCache>  // diagnostics; CUDA, Vulkan or Direct3D 12
    #[cfg(any(feature = "vulkan", all(windows, feature = "d3d12")))]
    pub fn device_page_counts(&self) -> Option<DevicePageCounts>     // where gpu_pages sent each page; None without gpu_pages
    // doc() and resolve_page() are #[doc(hidden)] — use prescan_session instead
}
```

An opened, read-only document. `Sync + Send` — safe to share across Rayon threads.

`unsupported_features` lists the document-level features the renderer ignores, found when the session opens: JavaScript attached to the catalog (`OpenAction`, `DocumentActions`), the `/Names /JavaScript` tree (`NameTree`), `/AcroForm /AA` (`FormActions`) or the `/AA` of a field in the `/AcroForm /Fields` tree (`FormFields`), an XFA form, and a portable collection, in that order; see [`UnsupportedFeature`](#unsupportedfeature). A page's own are in `PageDiagnostics::unsupported_features`.

`page_geometry` reads a page's size, rotation, `UserUnit` and boxes without parsing its content; see [`PageGeometry`](#pagegeometry). `geometry.size_px(scale_x, scale_y)` is the bitmap size a render at those scales produces, so an embedder can size buffers or plan tiles before rendering. `page_size_px(page_num, dpi)` is that size for a whole-page render at `dpi` (scale `dpi / 72` on both axes, rounded to nearest as the renderer rounds it; deskew keeps it), and it fails with the `InvalidOptions`, `PageDegenerate`, `PageTooLarge` or `PageExceedsCeiling` that render would.

---

### `RasterOptions`

```rust
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RasterOptions {
    pub dpi: f32,
    pub first_page: u32,
    pub last_page: u32,
    pub deskew: bool,
    pub pages: Option<PageSet>,
    pub strict: bool,
    pub purpose: RenderPurpose,
    pub annotations: bool,
    pub text_antialias: bool,
    pub vector_antialias: bool,
    pub overprint: bool,
    pub decrypt_authorized: bool,
    pub password: Option<Password>,
    pub read_into_memory: bool,
    pub font_dirs: Vec<PathBuf>,
    pub system_fonts: bool,
    pub image_budget: usize,
    pub max_page_bytes: u64,
    pub page_box: PageBox,
    pub rotation: Rotation,
    pub transparent_background: bool,
    pub gray_alpha: bool,
    pub depth: SampleDepth,
    pub output_profile: Option<Arc<OutputProfile>>,
    pub intent: Option<RenderingIntent>,
    pub black_point_compensation: bool,
    pub preserve_black: Option<PreserveBlack>,
    pub proof: Option<ProofIntent>,
    pub color: ColorModel,
    pub cancel: Option<CancelToken>,
}
```

| Field | Constraints | Notes |
|---|---|---|
| `dpi` | `> 0`, finite | Render resolution in pixels per physical inch. |
| `first_page` | `≥ 1` | 1-based inclusive. Ignored when `pages` is `Some`. |
| `last_page` | `≥ first_page` | 1-based inclusive. Clamped to document length silently. Ignored when `pages` is `Some`. |
| `cancel` | — | When `Some` and cancelled, the page in progress stops at its next operator and is yielded as `RasterError::Cancelled`; the iterator or channel then ends. |
| `deskew` | — | Applies intensity-weighted projection-profile deskew to gray pages: skew within ±6.75°, measured to about 2/W radians on a page W px wide (0.05° for A4 at 300 dpi). A page without a confident angle (best-to-worst sweep score ratio below 3, e.g. a photograph or a blank page) is left as it is. The measurement is in `RenderedPage::skew`. Disable for native-text PDFs. |
| `pages` | — | When `Some`, only the pages in the `PageSet` are rendered; `first_page`/`last_page` are ignored. |
| `strict` | — | When `true`, a page with an image that cannot be decoded or drawn, or is drawn only in part, returns `RasterError::ImageDecodeFailed`, and a page with a content stream that cannot be decoded, or decodes only in part, returns `RasterError::ContentDecodeFailed`, and a page with a shading whose `Function` yields no colour returns `RasterError::ShadingFailed`. When `false`, the image, stream or shading is skipped (or drawn from the data that decoded) and listed in `PageDiagnostics::image_failures`, `PageDiagnostics::content_failures` or `PageDiagnostics::shading_failures`. |
| `purpose` | — | `RenderPurpose::View` (on-screen viewing) or `RenderPurpose::Print`. See [`RenderPurpose`](#renderpurpose). |
| `annotations` | — | When `false`, annotations (form-field widgets included) are not drawn and pages show their content alone. |
| `text_antialias` | — | When `false`, glyphs are rasterised bilevel (no partial coverage). Text drawn as paths follows `vector_antialias`. |
| `vector_antialias` | — | When `false`, path fills, strokes and clips are scan-converted at pixel centres with hard edges, always on the CPU. |
| `overprint` | — | When `true`, overprint (`OP`, `op`, `OPM`) is simulated on pages that set it; see [`SessionConfig`](#sessionconfig). |
| `decrypt_authorized` | — | Whether an encrypted document may be decrypted; see [`SessionConfig`](#sessionconfig). `false` makes an encrypted document an error. |
| `password` | — | Password for an encrypted document, tried as user then owner password after the empty user password. No effect unless `decrypt_authorized` is `true`. |
| `read_into_memory` | — | When `true`, `raster_pdf` and `render_channel` read the file into memory instead of memory-mapping it, so a truncation of the file by another process cannot raise `SIGBUS`; see [`SessionConfig`](#sessionconfig). |
| `font_dirs` | — | Extra directories of installed fonts, searched before those `RROCKET_FONT_DIR` names and the standard ones; see [`SessionConfig`](#sessionconfig). |
| `system_fonts` | — | When `false`, the standard font directories are not searched; see [`SessionConfig`](#sessionconfig). |
| `image_budget` | — | Host bytes of decoded images kept for reuse by later pages; 0 keeps none. See [`SessionConfig`](#sessionconfig). |
| `max_page_bytes` | — | The most bytes one page canvas may hold (`CanvasLayout::bytes_per_px` per pixel); see [`SessionConfig`](#sessionconfig). |
| `page_box` | — | The page boundary box each page is measured and rendered on; see [`PageBox`](#pagebox). |
| `rotation` | — | A clockwise quarter-turn rotation added to each page's `/Rotate`; see [`Rotation`](#rotation). |
| `depth` | — | `SampleDepth::Eight` (default) or `SampleDepth::Sixteen`, which renders at 16 bits per channel and yields `PixelFormat::Gray16` pages, or `Rgba16` with `transparent_background`, of `Pixels::U16`. |
| `output_profile` | not RGB for gray pages, not gray for RGBA pages | The ICC profile the pages are converted to; see [`SessionConfig`](#sessionconfig). The gray pages take a gray profile, the RGBA pages of `transparent_background` an RGB one, its gray and alpha pages (`gray_alpha`) a gray one; another pairing is `RasterError::InvalidOptions` before any page renders. |
| `intent` | needs `output_profile` or `proof` | The rendering intent of the conversion (under `proof`, of the separation to the press); see `RenderParams::intent`. |
| `black_point_compensation` | needs `output_profile` or `proof`, not with `AbsoluteColorimetric` | Black-point compensation in the conversion (under `proof`, in the separation to the press). |
| `preserve_black` | needs `ColorModel::Cmyk` with a CMYK `output_profile`, or `proof`; not with `AbsoluteColorimetric` | Black preservation where `ICCBased` CMYK of another profile converts to the output profile's (or the press's) ink: `PreserveBlack::KOnly` keeps a colour of black ink alone (C = M = Y = 0) black ink alone, its K through the K → K curve matching the two profiles' blacks by L\*; `PreserveBlack::KPlane` also gives every other colour the curve's K and solves its C, M and Y for its colour, scaled to the destination's total ink limit (a colour that K cannot print within ΔE76 1 converts colorimetrically). `None` converts colorimetrically. See `RenderParams::preserve_black`. |
| `proof` | not with `transparent_background` or `ColorModel::Cmyk` | Soft-proof every page against its output intent; see **Soft proofs** under [Session API](#session-api). |
| `color` | not `Cmyk` with `deskew` or `transparent_background` | `ColorModel::Rgb` (default): gray pages, or RGBA over a transparent backdrop; `ColorModel::Cmyk`: `PixelFormat::Cmyk8` pages (`Cmyk16` at depth 16) of the process inks (see **CMYK pages** under the session API). |
| `transparent_background` | not with `deskew` | When `true`, pages are rendered over a transparent backdrop instead of white paper and yielded as `PixelFormat::Rgba8`: `[R, G, B, A]`, alpha the page's coverage × opacity, colour not premultiplied, `[0, 0, 0, 0]` where nothing paints. The page is painted as an isolated transparency group (knockout when its `/Group` says so). With `deskew` the options are `RasterError::InvalidOptions`. |
| `gray_alpha` | only with `transparent_background` | When `true`, the pages of `transparent_background` are `PixelFormat::GrayAlpha8` (`GrayAlpha16` at depth 16), `[Y, A]`: see `render_page_gray_alpha`. Without `transparent_background` the options are `RasterError::InvalidOptions`. |

`RasterOptions` implements `Default` and is `#[non_exhaustive]`: build it with `RasterOptions::new(dpi)` (every other field at its default) or `RasterOptions::default()`, then assign fields; struct literals and `..Default::default()` updates do not compile outside the crate.

```rust
let mut opts = RasterOptions::new(150.0);
opts.deskew = true;
```

Defaults: `dpi = 300.0`, `first_page = 1`, `last_page = u32::MAX`, `deskew = false`, `pages = None`, `strict = false`, `purpose = RenderPurpose::View`, `annotations = true`, `text_antialias = true`, `vector_antialias = true`, `overprint = false`, `decrypt_authorized = false`, `password = None`, `read_into_memory = false`, `font_dirs` empty, `system_fonts = true`, `image_budget = DOCUMENT_IMAGE_BUDGET_BYTES` (256 MiB), `max_page_bytes = DEFAULT_MAX_PAGE_BYTES` (1.8 GB), `page_box = PageBox::Crop`, `rotation = Rotation::None`, `transparent_background = false`, `gray_alpha = false`, `depth = SampleDepth::Eight`, `output_profile = None`, `intent = None`, `black_point_compensation = false`, `preserve_black = None`, `proof = None`, `color = ColorModel::Rgb`, `cancel = None`.

---

### `RenderPurpose`

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RenderPurpose { View, Print }
```

What a page is rendered for. It decides two things:

- **Optional content (layers, ISO 32000-2 §8.11).** The document's default configuration (`/OCProperties /D`) gives each group its state: `/BaseState`, then `/ON`, then `/OFF`, so a group listed in both is off. Then each `/AS` usage application dictionary whose `/Event` is `View` (or `Print` under `RenderPurpose::Print`) applies the `ViewState`, `PrintState` or `ExportState` of the groups it lists, for each category it names. A group whose `/Intent` does not match the configuration's `/Intent` is ignored. Both intents default to `View`, and `All` matches every intent. Content is hidden when its group is off or its membership dictionary (`/OCMD`, through `/VE` or through `/OCGs` with `/P`) evaluates to hidden. This applies to `BDC /OC` sequences, to image and form XObjects carrying `/OC`, and to annotations carrying `/OC`.
- **Annotation flags (ISO 32000-2 §12.5.3).** Under `View`, annotations flagged `NoView` are not drawn. Under `Print`, only annotations flagged `Print` are drawn. Annotations flagged `Hidden` are never drawn.

---

### `PageSet`

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageSet(/* opaque Arc<[u32]> */);

impl PageSet {
    pub fn new(pages: impl IntoIterator<Item = u32>) -> Result<Self, RasterError>
    pub fn contains(&self, page: u32) -> bool  // O(log n)
    pub fn first(&self) -> u32
    pub fn last(&self) -> u32
    pub fn len(&self) -> usize
    pub fn is_empty(&self) -> bool  // always false by invariant
    pub fn iter(&self) -> impl Iterator<Item = u32>  // ascending
    pub fn as_slice(&self) -> &[u32]
}
impl<'a> IntoIterator for &'a PageSet { type Item = u32; /* … */ }
```

A validated, sorted, deduplicated set of 1-based page numbers. The internal storage is reference-counted (`Arc<[u32]>`); `clone()` is O(1). Used in `RasterOptions::pages` to render a sparse subset of pages without visiting intermediate ones.

**`PageSet::new`** accepts any `IntoIterator<Item = u32>` (Vec, array, slice, range). The input is sorted and deduplicated before storage. Returns `RasterError::InvalidOptions` if the resulting set is empty or contains a zero page number.

```rust
let pages = PageSet::new(vec![1, 5, 23, 100])?;
let mut opts = RasterOptions::new(300.0);
opts.deskew = true;
opts.pages = Some(pages); // first_page / last_page are then ignored
```

---

### `RenderedPage`

```rust
#[derive(Debug)]
#[non_exhaustive]
pub struct RenderedPage {
    pub page_num: u32,
    pub width: u32,
    pub height: u32,
    pub pixels: Pixels,
    pub format: PixelFormat,
    pub dpi: f32,
    pub diagnostics: PageDiagnostics,
    pub skew: Option<deskew::Skew>,  // measured by deskew; None when off or not confident
}

pub enum Pixels { U8(Vec<u8>), U16(Vec<u16>) }   // U16: native-endian samples
impl Pixels {
    pub fn as_u8(&self) -> Option<&[u8]>
    pub fn as_u16(&self) -> Option<&[u16]>
    pub const fn len(&self) -> usize        // samples
    pub const fn is_empty(&self) -> bool
    pub const fn byte_len(&self) -> usize   // bytes the samples occupy
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PixelFormat {
    Gray8, Rgba8, Rgb8, Gray16, GrayAlpha8, GrayAlpha16, Rgb16, Rgba16, Cmyk8, Cmyk16,
}
impl PixelFormat {
    pub const fn bytes_per_pixel(self) -> usize    // 1, 4, 3, 2, 2, 4, 6, 8, 4 or 8
    pub const fn samples_per_pixel(self) -> usize  // 1, 2, 3 or 4
    pub const fn depth(self) -> SampleDepth
}

pub enum SampleDepth { Eight, Sixteen }            // re-exported from rasterrocket-color
pub struct CanvasLayout { pub depth: SampleDepth, pub alpha: bool, pub ink: bool }
impl CanvasLayout {
    pub const fn new(depth: SampleDepth, alpha: bool) -> Self  // an RGB canvas
    pub const fn with_ink(self, ink: bool) -> Self
    pub const fn bytes_per_px(self) -> u32         // 3 or 8 samples, + alpha, × depth bytes
}
```

| Field | Notes |
|---|---|
| `page_num` | 1-based page number. |
| `width`, `height` | Output bitmap dimensions in pixels. |
| `pixels` | In `format`, tightly packed, top-to-bottom: `Pixels::U8` for the 8-bit formats, `Pixels::U16` (native-endian `u16`) for the 16-bit ones. It holds exactly `width * height * format.samples_per_pixel()` samples: `width * height` for the default 8-bit grayscale pages. |
| `format` | `PixelFormat::Gray8` (one byte per pixel, 0 black), or `PixelFormat::Rgba8` (`[R, G, B, A]`, straight alpha) under `RasterOptions::transparent_background`; `Gray16` / `Rgba16` under `RasterOptions::depth = SampleDepth::Sixteen`; `Cmyk8` / `Cmyk16` (`[C, M, Y, K]`, 0 no ink) under `RasterOptions::color = ColorModel::Cmyk`. `PixelFormat::Rgb8` is only written by `PageBuffer::render_rgb_into`. |
| `dpi` | Pixels per physical inch (`opts.dpi`). The page size includes its `/UserUnit`, so this holds for any unit. Pass it to `tesseract::set_source_resolution`. |
| `diagnostics` | Lightweight rendering metadata — see `PageDiagnostics`. |

### `PageDiagnostics`

```rust
pub use rasterrocket_interp::renderer::PageDiagnostics;
```

Collected at zero extra cost during rendering.

| Field | Type | Notes |
|---|---|---|
| `has_images` | `bool` | At least one image XObject or inline image was rendered. |
| `has_vector_text` | `bool` | At least one text-showing operator (`Tj`, `TJ`, `'`, `"`) was executed. `false` on scan-only pages. |
| `dominant_filter` | `Option<ImageFilter>` | Most common image decode filter on this page (`None` for pure-vector pages). |
| `source_ppi_hint` | `Option<f32>` | Estimated native pixels-per-inch of the dominant image. Computed as `(image_width_px / page_width_pts) × 72` for the widest image, its width in source samples; `prescan_session` divides by the page box's width before `/Rotate`. `None` when no images were blitted. |
| `image_failures` | `Vec<ImageFailure>` | Images that were present but could not be decoded or drawn (skipped, so the region shows whatever was painted beneath), and images drawn only partly because their stream lacked samples (`partial`). Empty on a complete render. Includes images inside tiling-pattern tiles. A JPEG is partial exactly when its entropy-coded data stops short of its last row or is damaged; it is drawn as far as it decodes. An image whose decoded samples would exceed 1 GiB (width × height × components) is refused before anything is allocated and listed here, and so is a JPEG whose coefficient planes, held whole by a progressive frame or one coded in several scans, do not fit in what its samples leave of that budget. |
| `content_failures` | `Vec<ContentFailure>` | Elements of the page's `/Contents` that could not be decoded (or are not streams), each skipped while the page is drawn from the others, and elements that decoded only in part (`partial`), executed up to where their data was lost. Either way whatever the lost data held is missing. Empty on a complete render. A page none of whose content streams decode fails with `RasterError::Pdf` instead. |
| `forms_too_deep` | `Vec<ObjectId>` | Form XObjects not drawn because they lie more than `MAX_FORM_DEPTH` (40) forms deep, each once, by object identifier; the rest of the page rendered without them. Under `strict` such a form fails the page with `RasterError::PageBudgetExceeded`. |
| `shading_failures` | `Vec<ShadingFailure>` | Shadings not drawn because their `Function` yields no colour (a Type 4 program that does not parse, or a function that fails at every sample of a Type 1 shading or at any sample of the others), each once, by resource name (`name`, without the leading `/`) and whether it is a shading pattern's (`pattern`); the rest of the page rendered without them. Under `strict` such a shading fails the page with `RasterError::ShadingFailed`. |
| `missing_fonts` | `Vec<String>` | Font resource names (without the leading `/`) that `Tf` selected but the `/Font` resources in effect do not define, once each. Their text is drawn in Helvetica, so it is present but in the wrong face. Not a `strict` failure. |
| `unslotted_spots` | `Vec<String>` | Spot colorants that found no spot ink on a page rendered with overprint simulation (the four spot inks being held by the page's first four spots). Each printed its alternate colour in process ink, so it overprints as that process colour rather than as a separate ink. Empty on every other page. |
| `unsupported_features` | `Vec<UnsupportedFeature>` | Features on this page the renderer ignores: JavaScript in the page's `/AA`, an annotation's `/A` or `/AA` or the `/AA` of a field on its `/Parent` chain, and 3D and multimedia annotations (`/3D`, `/RichMedia`, `/Movie`, `/Sound`, `/Screen`); see [`UnsupportedFeature`](#unsupportedfeature). Empty when there are none. The document-level ones are on the session. |
| `approximated_group_spaces` | `GroupSpaceApproximations` | The transparency-group colour spaces the page composited in an approximation: `cmyk_profile`, a CMYK ICC group space over an RGB canvas composited through the process-ink model as `DeviceCMYK` is; `rgb_profile`, a `CalRGB` or non-sRGB ICC RGB group space whose conversions do not build, composited in the canvas's RGB working space. `any()` is either. |
| `form_type` | `FormType` | The document's interactive form type (the same on every page). `XfaFull` means a dynamic XFA form: XFA is not processed, so the page is the placeholder content stored in the file, often a notice asking for an XFA-capable viewer, and its text is not the form's. |
| `device_image_blits` | `u32` | Images drawn through the image cache's blit kernel from device-resident samples. 0 when the session holds no image cache, and for images the CPU draws: masked images, draws below the image's resolution, and draws beyond the blit tables. |
| `device_icc_conversions` | `u32` | Images in an ICC-based CMYK space converted to RGB on a GPU backend. 0 when the session holds no backend, and for images below the device's size threshold, which convert on the CPU with the same bytes. |
| `device_icc_failures` | `u32` | ICC CMYK conversions a GPU backend took and failed, each then performed on the CPU with the same bytes. Non-zero means the device path is broken on this host; the first failure of the process is also logged at `warn`. |
| `device_page` | `DevicePage` | Whether the page was rendered whole on a GPU (`SessionConfig::gpu_pages`), and why not when it was ruled out or its recording was refused. |
| `content_intents` | `IntentSet` | The rendering intents in force at the page's marking operators: the graphics state's (`ri`, `ExtGState` `/RI`, relative colorimetric where none is set) for paths, text and shadings, an image's own `/Intent` where it has one. Empty on a page that marks nothing. |
| `output_intent` | `Option<RenderingIntent>` | The intent the page was converted to `SessionConfig::output_profile` under; `None` without a profile. |
| `glyphs` | `GlyphCounts` | The glyphs the page's text-showing operators showed, by text rendering mode and by why a glyph cannot mark the page; see [`GlyphCounts`](#glyphcounts). All zero from `prescan_session`, which executes nothing. |

`PageDiagnostics::has_omissions() -> bool` (a `const fn`) is `true` when the page was drawn with something left out or substituted: a skipped image (`image_failures`) or content stream (`content_failures`), a form past the depth limit (`forms_too_deep`), a shading whose function yields no colour (`shading_failures`), a font that resolved to nothing (`missing_fonts`) or an unsupported feature (`unsupported_features`). An entry of those lists that was drawn in part (`partial`) counts; the glyph counts and the device fields do not. Read the fields for the cause.

#### `GlyphCounts`

```rust
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct GlyphCounts {
    pub fill: u32,          // 0 Tr
    pub stroke: u32,        // 1 Tr
    pub fill_stroke: u32,   // 2 Tr
    pub invisible: u32,     // 3 Tr: the text layer a recogniser lays over a scan
    pub clip_painted: u32,  // 4, 5, 6 Tr (and a Type 3 glyph in 7 Tr, whose description paints)
    pub clip_only: u32,     // 7 Tr
    pub zero_size: u32,     // singular text matrix, font size 0, Tz 0; an em below one device pixel (not Type 3)
    pub unrendered: u32,    // no glyph for the code, an unloadable font program, a Type 3 code without a CharProc,
                            // or shown before any font was selected
    pub off_page: u32,      // the em square widened by one em on each side misses the page box
    pub transparent: u32,   // every paint of the mode has opacity 0 or a colour space of colorants None
}
impl GlyphCounts { pub const fn visible(&self) -> u32 }   // fill + stroke + fill_stroke + clip_painted
impl PageDiagnostics { pub const fn has_visible_text(&self) -> bool }  // glyphs.visible() > 0
```

One glyph per character code shown, each counted once. A glyph in a painting mode (0, 1, 2, 4, 5, 6) is counted under its mode only when it can mark the page, else under the first of `zero_size`, `unrendered`, `off_page`, `transparent` that applies; a glyph in mode 3 or 7 is counted under its mode. The counts are the page's: the same whether it is rendered whole, in bands (`render_page_bands`) or as a `RenderRegion`, at any `band_rows`. Only the `zero_size` case of an em below one pixel depends on the render's resolution.

Not considered: the clip (a glyph clipped away still counts as visible), soft masks, blend modes, a colour equal to the backdrop's (white text on white), and empty outlines (a space counts as drawn). Glyphs inside a tiling pattern's cell, a soft mask's group and a Type 3 glyph's own description are not counted; a Type 3 glyph counts as one glyph of its string, and text in a substitute or fallback face counts as any other.

`has_visible_text()` is the test for skipping OCR: `false` on a scanned page whose only text is an invisible `3 Tr` layer, or that has none; `true` on a page that draws native text, whether or not it also has images or an invisible layer:

```rust
let d = &page.diagnostics;
if d.has_visible_text() {
    // native text: extract it, skip OCR
} else if d.glyphs.invisible > 0 {
    // a recognised scan: an earlier OCR's text layer is present
} else if d.has_images {
    // a scan without a text layer: OCR it
}
```

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum DevicePage {
    #[default]
    NotOffered,                  // not offered: no GPU pages, the device did not open (or not within `GPU_PAGE_OPEN_WAIT`) or retired;
                                 // also a recording stopped by the work budget or cancellation
    Rendered { commands: u32 },  // rendered on the device from this many paint commands
    Ineligible(&'static str),    // paints with what the GPU raster does not render (named); rendered on the CPU
    DeviceLimit,                 // the page exceeds the device's limits or the memory it has left; rendered on the CPU, device kept
    Failed,                      // the device failed on the page and is retired; rendered on the CPU
}
```

Every `DevicePage` outcome yields the same bytes. The reasons `Ineligible` names: before interpretation, `"overprint simulation"` (an ink canvas), `"page group"`, `"aliased vector paint"`, `"the wide working space"`, `"shading"` (`sh`) and `"Type 3 glyph"` (a CharProc's `d0`/`d1`, which only a Type 3 glyph's own description runs); while recording, `"pattern"`, `"blend mode"` (other than `Normal`), `"transparency group"` (a form), `"translucent hairline"`, `"translucent fill and stroke"`, `"image transfer function"`, `"image sampling range"`, `"device-resident image"` (an image-cache image whose host copy cannot be read), `"aliased path clip"`, `"soft mask depth"`, `"soft mask range"`, `"outline range"`, `"command list size"` (more than `PAGE_COMMANDS_MAX_BYTES`, 128 MiB, of device inputs) and `"command list allocation"`; from `render_page_bands`, `"larger than one band"` (a page drawn in several bands of `SessionConfig::band_bytes`, on the CPU; the first such page of a session warns).

`RasterSession::device_page_counts()` (`vulkan` feature, or `d3d12` on Windows) totals the session's pages so far as a `DevicePageCounts` (`#[non_exhaustive]`, an output): `rendered` (on the devices), `by_device` (`Vec<(String, u32)>`, the pages each device rendered, in the order of `SessionConfig::gpu_devices`, one entry for a session on one device; a device is named by its driver's name, the adapter description on Direct3D 12, or while not open by its selector), `ineligible` (`Vec<(&'static str, u32)>`, by the reasons above, in order of first occurrence), the `NotOffered` pages split into `opening` (the device had not opened when the session's wait, `GPU_PAGE_OPEN_WAIT`, ended), `no_device` (it did not open), `retired` (after a device error) and `stopped` (a recording stopped by the work budget, deadline or cancellation), `device_limit` (the `DeviceLimit` pages: `BackendError::DeviceLimit` or `OutOfMemory`, the device kept in service) and `failed` (the `Failed` pages, each retiring its device). `Display` names them `device limit` and `device failed`. `cpu()` sums the pages rendered on the CPU; `Display` gives the line the session logs at `info` when it drops, e.g. `598 on the device, 3 on the CPU (shading: 3)`. `None` for a session without `gpu_pages`.

```rust
#[non_exhaustive]
pub enum FormType {
    NoForm,         // no /AcroForm in the catalog
    AcroForm,       // /AcroForm without /XFA
    XfaFull,        // /AcroForm with /XFA and catalog /NeedsRendering true (dynamic XFA)
    XfaForeground,  // /AcroForm with /XFA otherwise (static XFA; pages render normally)
}
```

Opening a dynamic XFA document (`XfaFull`) also logs one `warn`.

#### `UnsupportedFeature`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum UnsupportedFeature {
    JavaScript(ScriptLocation),          // a JavaScript action; never executed
    MediaAnnotation {                    // interactive content not rendered
        kind: MediaKind,
        index: usize,                    // position in the page's /Annots
        has_appearance: bool,            // has /AP /N, drawn in its place when annotations render
    },
    Xfa { dynamic: bool },               // /AcroForm /XFA; dynamic = /NeedsRendering true
    Portfolio,                           // catalog /Collection; embedded documents not rendered
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ScriptLocation {
    OpenAction,                          // catalog /OpenAction            (document)
    DocumentActions,                     // catalog /AA                    (document)
    NameTree,                            // catalog /Names /JavaScript     (document)
    FormActions,                         // /AcroForm /AA                  (document)
    FormFields,                          // /AA of a field in /AcroForm /Fields, via /Kids (document)
    PageActions,                         // page /AA                       (page)
    AnnotationAction { index: usize },   // annotation /A                  (page)
    AnnotationActions { index: usize },  // annotation /AA, e.g. a field's calculate script (page)
    FieldActions { index: usize },       // /AA of a field on the annotation's /Parent chain (page)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MediaKind { ThreeD, RichMedia, Movie, Sound, Screen }  // subtype() -> "3D", "RichMedia", …
```

Features a document uses that the renderer ignores, as typed data. The page renders without them; what is missing is what a script would have drawn, hidden or filled in, the 3D model or media clip (its poster appearance, when it has one, is drawn), the XFA layout of a dynamic form, or the documents a portfolio carries. `RasterSession::unsupported_features()` returns the document-level ones (`JavaScript` at a document location, `Xfa`, `Portfolio`); `PageDiagnostics::unsupported_features` lists a page's (`JavaScript` at a page location, `MediaAnnotation`), from a render and from `prescan_session` alike, page `/AA` first and then per annotation in `/Annots` order its `MediaAnnotation`, its `/A` script, its `/AA` scripts and the `/AA` scripts of the fields on its `/Parent` chain. Detection is structural: no script, model or media stream is decoded; an action counts as JavaScript only when it runs a script (`/S /JavaScript`, or `/S /Rendition` with `/JS`, also along its `/Next` chain), never on bare `/A` or `/AA` presence, and the `/Names /JavaScript` tree counts only when it holds an entry; the field hierarchy is followed at most 64 levels (`/Parent` up from an annotation, `/Kids` down from `/AcroForm /Fields`), each object once, so a cycle ends; at most 8 192 annotations per page are inspected, a shared action is inspected once per scan, and a scan stops with a warning after 2^20 steps, reporting what it found. All three types implement `Display` (`JavaScript (annotation 3 /AA)`, `/3D annotation 0 (normal appearance only)`, `dynamic XFA form`, …), and `UnsupportedFeature::is_javascript()` tests for a script. The warn logs stay: opening a document warns about catalog JavaScript, a dynamic XFA form and a portfolio, and rendering warns about the first page with JavaScript and the first with a 3D or multimedia annotation, once per session each. The CLI prints nothing beyond these logs.

```rust
#[non_exhaustive]
pub struct ImageFailure {
    pub name: Option<String>,  // XObject resource name without '/', e.g. "Im0"; None for an inline image
    pub reason: String,        // codec or blit error, prefixed by the filter where known, e.g. "FlateDecode: …";
                               // for a partial image, what the stream lacked, e.g. "raw data short: 1200 of 4800 bytes; …"
    pub partial: bool,         // the image was drawn from the samples it had, the rest padded; false when skipped
}
```

`ImageFailure` implements `Display` as `image /Im0: <reason>` or `inline image: <reason>`, with ` (partial)` after the name for a partial image. Without `strict`, each failure is also logged at `warn` level as `page N: skipped …` (or `page N: image /Im0 (partial): …`). Under `strict`, a partial image fails the page like a skipped one: its content is incomplete.

```rust
#[non_exhaustive]
pub struct ContentFailure {
    pub object: u32,     // object number of the /Contents element
    pub reason: String,  // decode error, e.g. "unsupported filter: FooDecode";
                         // for a partial element, the filter that stopped early and why
    pub partial: bool,   // the bytes decoded before the loss were kept and executed
}
```

`ContentFailure` implements `Display` as `content stream obj 5: <reason>`, with ` (partial)` after the object number for a partial element, and is logged the same way. Under `strict`, a partial content stream fails the page like an undecodable one.

#### `PageDiagnostics::suggested_dpi`

```rust
pub fn suggested_dpi(&self, min_dpi: f32, max_dpi: f32) -> Option<f32>
```

Suggests a re-render DPI based on `source_ppi_hint`. Returns `None` for vector/text-only pages. Snaps to the nearest standard step (72, 96, 150, 200, 300, 400, 600) and clamps to `[min_dpi, max_dpi]`.

```rust
if let Some(native_dpi) = page.diagnostics.suggested_dpi(150.0, 600.0) {
    if (native_dpi - opts.dpi).abs() > 10.0 {
        // Re-render at native resolution
    }
}
```

Use `diagnostics` to route pages to different OCR configurations:

```rust
// Page looks like a scan (has images, no vector text)
if page.diagnostics.has_images && !page.diagnostics.has_vector_text {
    // Sauvola handles uneven scan backgrounds better than Otsu
    tesseract.set_variable("thresholding_method", "2");
}

// Avoid deskew overhead on native-text pages
if !page.diagnostics.has_images && page.diagnostics.has_vector_text {
    // deskew: false for this page
}
```

---

### `BackendPolicy`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum BackendPolicy {
    Auto,        // the CPU: opens no GPU backend, image cache or nvJPEG decoder at start-up; gpu_pages opens its Vulkan devices on demand, and with nvjpeg2k and a set PDF_RASTER_JPX_THRESHOLD a thread builds its nvJPEG2000 decoder on demand (default)
    CpuOnly,     // Skip all GPU init entirely
    ForceCuda,   // Require CUDA; error if unavailable
    #[cfg(feature = "vaapi")]
    ForceVaapi,  // Require VA-API JPEG; error if unavailable — only when `vaapi` feature is enabled
    ForceVulkan, // Require the Vulkan compute backend; error if unavailable
                 // (or if the binary was built without `--features vulkan`)
    ForceD3d12,  // Require the Direct3D 12 compute backend (Windows); error if no adapter opens
                 // (or if the binary was built without `--features d3d12` or for another platform)
}
```

Controls which compute backend is used. The `Force*` variants convert silent GPU fallbacks into hard `RasterError::BackendUnavailable` errors so you know immediately whether the expected hardware path is actually active. `ForceVaapi` is only present when the `vaapi` Cargo feature is enabled.

`ForceVulkan` runs the session's GPU work on the Vulkan compute backend (cross-vendor: NVIDIA, AMD, Intel (tested on a UHD 630 under Windows only), Apple via `MoltenVK`; see [Tested hardware](../README.md#tested-hardware)).  In a build with the `cache` feature the device-resident image cache keeps its images in Vulkan device memory and draws them with the Vulkan blit kernel, byte-identical to the CPU render; with `gpu-icc`, ICC CMYK→RGB runs on the Vulkan device too. Sessions open at the same time with the same `vulkan_device` share one Vulkan device; the last of them to drop destroys it, and a later session builds it again.

`ForceD3d12` (added after v2.4.1; additive, since the enum is `#[non_exhaustive]`) opens the Direct3D 12 adapter `SessionConfig::d3d12_device` names, or the first hardware adapter with dedicated video memory, else the first hardware adapter. With `gpu-icc` the ICC CMYK→RGB lookup table runs on it, byte-identical to the CPU, after a one-time exactness probe (a device that fails it converts on the CPU); with `cache` the device image cache keeps its slabs on the adapter and its blit kernel draws them, byte-identical to the CPU, under the same rules as on CUDA and Vulkan (a JPEG drawn below its resolution is not cached, nor anything for a 16-bit canvas, which never takes the device blit); with `gpu_pages` qualifying pages render whole on the adapter, or on the adapters `gpu_devices` lists (selectors as for `d3d12_device`; `all` is every hardware adapter and never WARP, which only `warp` or its index names), byte-identical to the CPU at 8 and 16 bits, and `device_page_counts` names each adapter by its DXGI description. One session's GPU pages are all on one API: Vulkan under `Auto` and `ForceVulkan`, Direct3D 12 under `ForceD3d12`. It initialises no CUDA context or Vulkan device. Tested on an RTX 5070, an AMD Radeon iGPU and the pinned WARP 1.0.22 on Windows 11.

`BackendPolicy` parses from `auto`, `cpu`, `cuda`, `vulkan`, `d3d12` and, with the `vaapi` feature, `vaapi` (`FromStr`; case-insensitive, whitespace trimmed, empty is `Auto`), and `name()` returns that spelling; `BackendPolicy::ALL` lists the policies of the build. A name it does not accept is a `ParseBackendPolicyError`: `value()` returns the rejected input, and its `Display` names it and lists the accepted names (`unknown backend "cdua"; expected one of auto, cpu, cuda, vulkan, d3d12`, with `vaapi` after `cuda` in a `vaapi` build). `BackendPolicy::from_env()` reads `PDF_RASTER_BACKEND`, and `from_env_var(name)` any other variable; an unrecognised value logs a warning and resolves to `Auto`.

---

### `SessionConfig`

```rust
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub policy: BackendPolicy,
    pub vaapi_device: String,  // default: DEFAULT_VAAPI_DEVICE ("auto", probe render nodes)
    pub vulkan_device: Option<String>,  // default: None (automatic)
    pub d3d12_device: Option<String>,   // default: None (automatic); read under ForceD3d12 only
    // `prefetch: bool` and `prefetch_scale: Option<(f64, f64)>` exist only when `cache` feature is enabled
    pub decrypt_authorized: bool,  // default: false
    pub password: Option<Password>, // default: None
    pub read_into_memory: bool,    // default: false (a file of 256 KiB or more is mapped)
    pub strict: bool,              // default: false
    pub purpose: RenderPurpose,    // default: RenderPurpose::View
    pub annotations: bool,         // default: true
    pub text_antialias: bool,      // default: true
    pub vector_antialias: bool,    // default: true
    pub overprint: bool,           // default: false
    pub font_dirs: Vec<PathBuf>,   // default: RROCKET_FONT_DIR (Default), empty (with_policy)
    pub system_fonts: bool,        // default: true
    pub image_budget: usize,       // default: DOCUMENT_IMAGE_BUDGET_BYTES (256 MiB)
    pub max_page_bytes: u64,       // default: DEFAULT_MAX_PAGE_BYTES (1.8 GB)
    pub band_bytes: u64,           // default: DEFAULT_BAND_BYTES (64 MiB)
    pub page_box: PageBox,         // default: PageBox::Crop
    pub rotation: Rotation,        // default: Rotation::None
    pub gpu_pages: bool,           // default: false
    pub gpu_devices: Vec<String>,  // default: empty (the session's device; vulkan_device under Auto)
    pub output_profile: Option<Arc<OutputProfile>>, // default: None (sRGB pages)
    pub proof: Option<ProofIntent>,                 // default: None (no soft proof)
    pub working_space: Option<WorkingSpace>,        // default: None (Wide under an output profile, else Srgb)
}

impl Default for SessionConfig { /* policy from PDF_RASTER_BACKEND, Vulkan device from PDF_RASTER_VULKAN_DEVICE, D3D12 adapter from PDF_RASTER_D3D12_DEVICE, font directories from RROCKET_FONT_DIR */ }

impl SessionConfig {
    pub fn with_policy(policy: BackendPolicy) -> Self;      // reads no env vars
    pub fn vulkan_device_from_env() -> Option<String>;     // PDF_RASTER_VULKAN_DEVICE
    pub fn d3d12_device_from_env() -> Option<String>;      // PDF_RASTER_D3D12_DEVICE
    pub fn font_dirs_from_env() -> Vec<PathBuf>;           // RROCKET_FONT_DIR
    pub fn check_working_space(&self) -> Result<(), RasterError>; // Wide needs an output profile
    pub fn check_gpu_pages(&self) -> Result<(), RasterError>;     // gpu_pages needs Auto/ForceVulkan (vulkan) or ForceD3d12 (d3d12, Windows), vector AA
}
```

Passed to `open_session`. `SessionConfig` is `#[non_exhaustive]` — construct it via `SessionConfig::default()` or `SessionConfig::with_policy(policy)`, not a struct literal. This is required because the `prefetch` field only exists when the `cache` Cargo feature is enabled. `prefetch_scale` (default `None`) gives the prefetcher the device pixels per point `(x, y)` at which every page is rendered, when that is one scale for all pages: an image all of whose draws on its page are smaller than the image is then not prefetched, since the render resamples such a draw on the host and bypasses the cache; with `None` every JPEG image is prefetched. `vaapi_device` defaults to `"auto"` (`DEFAULT_VAAPI_DEVICE`): the first `/dev/dri/renderD*` node, in ascending number, whose driver supports VA-API baseline JPEG decode. Set it to a render-node path to use that node instead. Under `ForceVaapi` a failure is `RasterError::BackendUnavailable` naming the render nodes present; under `Auto` JPEG falls back to the CPU. `CpuOnly`, `ForceCuda`, `ForceVulkan` and `ForceD3d12` never open a render node. Set `vulkan_device` to choose the Vulkan device: an index in Vulkan enumeration order (from 0) or a case-insensitive device-name substring; a number that is not a valid index is matched as a name. A value that matches no device, more than one, or a device without Vulkan 1.3 and a compute queue makes `open_session` return `RasterError::InvalidOptions` listing the devices, under `Auto` as well as `ForceVulkan`; `Auto` checks the value against the enumerated devices without opening one. `None` picks automatically. `strict` has the same meaning as `RasterOptions::strict`, for renders through a session (`render_page_rgb` and friends); `raster_pdf`, `raster_pdf_from_bytes` and `render_channel` take it from `RasterOptions`. `purpose` works the same way and mirrors `RasterOptions::purpose`; see [`RenderPurpose`](#renderpurpose). `annotations = false` renders page content without annotations and mirrors `RasterOptions::annotations`. `text_antialias` and `vector_antialias` (both default `true`) switch antialiasing of glyph bitmaps and of path fills, strokes and clips, and mirror the `RasterOptions` fields of the same names. `overprint` (default `false`, mirrored by `RasterOptions::overprint`) simulates overprint (ISO 32000-1 §8.6.7): a page on which some `ExtGState` reachable from its resources, or from the appearance streams of annotations that are drawn, sets `OP` or `op` is painted on a canvas of four process and four spot inks, where an overprinting object leaves the inks it does not paint (under `OPM 1`, also the zero components of a process CMYK colour), and is converted to RGB when finished. The first four distinct spot colorants of a page take the spot inks; a colour with spots that do not fit prints its alternate colour in process ink and is listed in `PageDiagnostics::unslotted_spots`. Other pages render byte-identically to `overprint = false`. The RGB-to-ink table is built once per process on first use (about 0.75 s; concurrent renders wait for it). Sampled images and shadings in `Separation` / `DeviceN` spaces and tiling-pattern cells paint the inks of their own colorants; `ICCBased` CMYK colours print as the process ink that reproduces their profile's colour; a transfer function (`TR`, `TR2`) is applied in ink space, its red, green, blue and gray functions to cyan, magenta, yellow, and to black and the spot inks, and a transferred colour keeps its inks and overprint fields. A black or near-black gray or RGB colour prints as a rich black balanced so that its antialiased edges stay neutral. The device image blit, when enabled, draws on the ink canvas with the bytes of the CPU. Not simulated: a transparency group as a whole paints every ink. `decrypt_authorized` permits decrypting an encrypted document (default `false`: such a document is an error, whatever `password` holds); `password` is tried as the user and then the owner password after the empty user password. Both mirror the `RasterOptions` fields of the same names. `Password` (re-exported from the `pdf` crate) converts from `&str` and `String` (`Some("secret".into())`) and prints as `Password(<redacted>)` under `Debug`, so a logged `SessionConfig` does not disclose it. `font_dirs` lists extra directories of installed fonts, searched in order before the standard font directories: a font the document does not embed draws with an installed substitute (the metric-compatible Liberation family, or a face of the font's script) and, for a character the substitute lacks, with an installed face that covers it, and both are looked up there first. `SessionConfig::default()` reads them from `RROCKET_FONT_DIR` (`PATH` syntax), `with_policy` sets none; sessions naming the same directories share one index of the installed fonts, built from one walk of the directories and their subdirectories (a directory reached through several paths or a symbolic link is read once; the walk stops after 10 000 directories with a warning) and rebuilt when a directory changes. `RasterOptions::font_dirs` is searched before the `RROCKET_FONT_DIR` directories by `raster_pdf`, `raster_pdf_from_bytes` and `render_channel`. The standard font directories are those of the build target (the Windows and macOS ones only on those targets). `system_fonts = false` (mirrored by `RasterOptions::system_fonts`) stops the search of the standard font directories and of the fixed locations of the Liberation, Arial, Times New Roman and Courier New faces: non-embedded fonts then draw from `font_dirs` and the faces compiled into the library only, the base-14 substitutes and, with the default `bundled-fonts` feature, the bundled Noto fallback faces. Lookup order for a non-embedded font: an installed substitute of its class and script, then for a character it lacks an installed face that covers it, then a bundled face, then nothing. `image_budget` (mirrored by `RasterOptions::image_budget`) bounds the host bytes of decoded images the session keeps so that an image `XObject` drawn on several pages (a letterhead, a background scan) is decoded once: an image is kept from the second page that draws it, while it fits, and what is kept stays resident until the session is dropped. The default is `DOCUMENT_IMAGE_BUDGET_BYTES` (256 MiB, re-exported at the crate root); 0 keeps nothing, so every page decodes the images it draws, for small-memory hosts and embedders that hold many sessions. Images a page draws several times are still decoded once per page, within a per-page budget. `max_page_bytes` (mirrored by `RasterOptions::max_page_bytes`) bounds the bitmap of one render, see [`MAX_PX_SIDE`](#max_px_side--default_max_page_bytes--default_band_bytes); `band_bytes` sizes the bands of `render_page_bands` (no `RasterOptions` counterpart: its entry points return pages whole). `page_box` selects the page boundary box every page of the session is measured (`page_geometry`, `prescan_session`) and rendered on; see [`PageBox`](#pagebox). `raster_pdf`, `raster_pdf_from_bytes` and `render_channel` take it from `RasterOptions::page_box`. `rotation` turns every page of the session clockwise by a whole number of quarter turns on top of its own `/Rotate`; see [`Rotation`](#rotation). `raster_pdf`, `raster_pdf_from_bytes` and `render_channel` take it from `RasterOptions::rotation`. `gpu_pages` (default `false`; no `RasterOptions` counterpart) renders pages whole on a GPU beside the CPU workers: on Vulkan under `Auto` and `ForceVulkan`, on Direct3D 12 under `ForceD3d12`. Under `ForceVulkan` and `ForceD3d12` the session's device is used; under `Auto` the device of `vulkan_device` is opened on a thread of its own when the first page that may render on it is rendered, so a job none of whose pages can opens none. Every page that may render on the device does: its thread waits for the device to open (the session's one wait, `GPU_PAGE_OPEN_WAIT`, 30 seconds counted from the first page that waits and polled every 100 ms, so a page cancelled meanwhile returns the cancellation; once it has passed, pages render on the CPU at once while no device is open, with a warning, and a device that opens later is still taken), records the page and waits for the device, which renders one page at a time while up to two others stage their inputs or read their pixels back. Threads rendering pages at once record them in parallel, so on cheap pages the device sets the pace. A page is interpreted into paint commands, and the device computes each pixel with the CPU's integer arithmetic, so the page is byte-identical to the CPU's at 8 and 16 bits: fills and strokes of solid colours as the outlines the CPU's exact-area rasteriser walks (as its coverage tables on a device without `shaderInt64`), opaque hairlines as their pixel runs, glyphs as the glyph cache's coverage masks (cached Type 3 glyphs included), images, inline images and stencil masks through the CPU's sampling tables (the CPU decodes, colour-converts and downsamples them), form XObjects and annotation appearances as their content, path clips (the clipping text modes 4–7 included) as the coverage tables of their regions, and the objects painted under a soft mask blended back by the mask, which the CPU renders. A page renders on the CPU, with the same bytes, when it is on the ink canvas (overprint simulation), is painted as a page group, is composited in the wide working space, or paints with a transparency-group form, a blend mode other than `Normal`, a pattern or shading, a Type 3 glyph that runs its CharProc, or more than 128 MiB of device inputs (`DevicePage::Ineligible` names the reason). The canvas, page group, working space, shadings and CharProcs rule a page out before it is interpreted; any other reason stops the recording at the first such paint, and the CPU re-render draws the images it decoded, shares the recording's deadline and has its own operator budget. A page that `render_page_bands` draws in several bands renders on the CPU (`DevicePage::Ineligible("larger than one band")`, warned once a session); a page of one band renders as `render_page_rgb_with` does. A page the device fails on renders on the CPU. A page beyond the device's limits (`BackendError::DeviceLimit`) or beyond the memory it has left (`BackendError::OutOfMemory`, after the idle buffer sets are freed and the allocation retried once) is logged at `warn` and leaves the device in service; any other device error retires the device for the session, with one warning naming the device and the error. Among those is `BackendError::KernelFault`: the outline raster checks every division it makes and reports one it could not complete exactly through a status buffer, which is read before any pixel is copied out, so the faulting page's device pixels are discarded. On the vector and image documents measured, a GPU page usually costs less than a CPU page and a run with few threads is faster; with many threads the CPU is faster (by 1.1–4× on the fixtures at 24 threads), most on cheap pages such as plain text. Tested on an RTX 5070, an RTX 2080 SUPER, an AMD Raphael iGPU (RADV), an Intel UHD 630 (under Windows), an A18 Pro (MoltenVK) and llvmpipe. `PageDiagnostics::device_page` reports where each page rendered. `gpu_devices` (default empty) lists the devices the GPU pages render on, all of the policy's API: Vulkan devices (selectors in the `vulkan_device` syntax) under `Auto` and `ForceVulkan`, Direct3D 12 adapters (selectors in the `d3d12_device` syntax) under `ForceD3d12`, or the one entry `"all"` for every usable GPU of the API (never a CPU implementation of Vulkan or WARP); empty renders them on the device of `vulkan_device` under `Auto`, and on the session's device under `ForceVulkan` and `ForceD3d12`. Each page takes the open device where it expects the shortest wait (the pages the device holds and itself, times the device's recent mean time per page raster, `PageDevice::page_mean()`: an exponentially weighted mean (newest run 1/8) of the page rasters that succeeded, the device's first excluded; a device that has rendered none yet counts the fastest known mean), so a faster device takes more pages; a device error retires that device alone, and later pages go to the others. `open_session` resolves the list against the enumerated devices once per list per process (without opening a device under `Auto`; under `ForceVulkan` and `ForceD3d12` it opens every one, failing as for its own device) and returns `RasterError::InvalidOptions` listing the devices when an entry matches no device or several, two name the same device, or `all` finds no GPU; `check_gpu_pages()` refuses a non-empty list without `gpu_pages`, `all` beside other entries, an empty entry, and under `Auto` a list beside `vulkan_device`, which it replaces there. Under `Auto` the devices open one after another (device creation is serialised in the process) and a page waits for the first. Tested with two devices of different vendors (an RTX 5070 and an AMD Raphael iGPU); more devices and datacenter GPUs are untested. `open_session` refuses `gpu_pages` with `RasterError::InvalidOptions` unless the policy is `Auto` or `ForceVulkan` in a build with the `vulkan` feature or `ForceD3d12` in a Windows build with the `d3d12` feature, or with `vector_antialias` off, and `SessionConfig::check_gpu_pages()` applies the same check before any document is opened (the message names the fields and their `rrocket` flags); a device that does not open is logged at `error` and leaves every page on the CPU. `output_profile` (default `None`, mirrored by `RasterOptions::output_profile`) converts every page to an ICC profile; see **Output profiles** under [Session API](#session-api).

### `PageBox`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum PageBox { Media, #[default] Crop, Bleed, Trim, Art }
impl PageBox {
    pub const fn key(self) -> &'static str  // "MediaBox", "CropBox", …
}
```

A page boundary box (ISO 32000-2 §14.11.2), re-exported from `rasterrocket-interp`. `Media` is the `MediaBox`, own or inherited; when it is missing the `CropBox` stands in. `Crop` (the default) is the `CropBox`, own or inherited, intersected with the `MediaBox`; the `MediaBox` when the `CropBox` is absent or the intersection is empty. `Bleed`, `Trim` and `Art` read the page dictionary's own `BleedBox`, `TrimBox` or `ArtBox` (these keys are not inherited from `/Pages` nodes), intersected with the `MediaBox`; a page without a usable box of that kind, or whose box lies outside the `MediaBox`, takes the `Crop` region. `/Rotate` and `UserUnit` apply to every box, and the selected box's lower-left corner is the page origin (`PageGeometry::origin_x` / `origin_y`).

### `Rotation`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Rotation { #[default] None, Cw90, Cw180, Cw270 }
impl Rotation {
    pub const fn from_degrees(degrees: i64) -> Result<Rotation, InvalidRotation>;
    pub const fn degrees(self) -> u16;      // 0, 90, 180 or 270
    pub const fn is_quarter(self) -> bool;  // Cw90 or Cw270
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidRotation { pub degrees: i64 }  // impl Error + Display
```

An extra clockwise rotation (`SessionConfig::rotation`, `RasterOptions::rotation`), re-exported from `rasterrocket-interp` with its error. `from_degrees` takes any multiple of 90, negative for counter-clockwise, and reduces it modulo 360 (`-90` and `630` are `Cw270`); any other value is `InvalidRotation`. The rotation is added to the page's `/Rotate` before the page geometry is computed, so the page is drawn turned rather than turned as a bitmap afterwards: `PageGeometry` (`rotate_cw` is the sum, width and height exchanged on a quarter turn), `RasterSession::page_size_px`, every render entry point, a `PageBuffer`, `RenderRegion`s (placed on the turned page), deskew (run on the turned page) and the gray, RGB and transparent RGBA outputs all see the turned page. Annotations flagged `NoRotate` stay upright on the turned page. `prescan_session`'s PPI hint ignores the rotation. A page whose pixel size is whole at the render's resolution renders its fills, clips and nearest-neighbour images to the unturned render turned as a bitmap, byte for byte; antialiased edges at fractional sample positions, zero-width lines, text and interpolated images can differ at the edges, because their scan conversion depends on the direction of the device axes, and on a page whose pixel size is fractional the fraction of a pixel by which the bitmap exceeds the page stays past its right and bottom edges. `ComicOptions::session.rotation` also turns comic-archive page images (an exact pixel rotation, before deskew).

---

### `deskew` module

The `deskew` module exposes a limited public API:

- `deskew::detect<P: DeskewPixel>(&Bitmap<P>) -> Result<Option<Skew>, DeskewError>` — measures the skew of a grayscale bitmap: an intensity-weighted projection-profile sweep of 57 angles (0.25° apart, ±`MAX_SEARCH_DEG` = 7°) on a 4× reduction, refined on a 2× reduction. `None` when the page has no confident angle: a confidence (best over worst sweep score) below `MIN_CONFIDENCE` (3.0), a flat profile (blank or uniform page), or a best angle within one sweep step of either end (beyond ±6.75°). The angle resolves about 2/W radians on a page W px wide; pages narrower than about 1200 px measure less reliably.
- `deskew::Skew { angle_deg: f32, confidence: f32 }` — the tilt of the page's lines, counter-clockwise positive, and the confidence (at least `MIN_CONFIDENCE`).
- `deskew::apply<P: DeskewPixel>(&mut Bitmap<P>, policy: BackendPolicy) -> Result<Option<Skew>, DeskewError>` — `detect`, then rotates the bitmap in place to correct a skew of at least `MIN_CORRECT_DEG` (0.05°); returns the skew measured, and leaves the bitmap as it is when that is `None` or smaller. `DeskewPixel` is a sealed trait of `Gray8` and `Gray16`; a 16-bit page is measured on its nearest 8-bit levels and rotated at 16 bits. A page rotates on the GPU (NPP) under `BackendPolicy::ForceCuda` in a `gpu-deskew` build, at either depth, and on the CPU otherwise. The GPU rotation's samples whose source position lies within 2 pixels of the page's edges are recomputed with the CPU's kernel, so its white background meets the page as the CPU's does; elsewhere NPP's bilinear differs from the CPU's by at most 2 levels at 8 bits and 8 levels of 65 535 at 16 bits (on the RTX 5070, NPP 13.1: at most 1 level at 8 bits and 7 levels of 65 535 at 16 bits on a 2133 × 3520 scan, at most 2 on random images up to 257 pixels a side). `DeskewError::TooLarge` for a side over `DESKEW_MAX_SIDE`; `DeskewError::GpuRotation` when the forced GPU rotation fails.
- `deskew::DeskewError` (`#[non_exhaustive]`) — error type of `detect` and `apply`: `TooLarge`, a side over `DESKEW_MAX_SIDE`; `GpuRotation`, the GPU rotation `ForceCuda` requires failed (`apply` only).

The `deskew::rotate` submodule is `pub(crate)` and not part of the public API.

---

### `release_gpu_thread_state`

```rust
pub fn release_gpu_thread_state()
```

Drops the GPU state of the calling thread: its nvJPEG and nvJPEG2000 decoders, CUDA streams, page-locked staging buffers and NPP rotator. Call via `pool.broadcast` before dropping the pool:

```rust
let _ = pool.broadcast(|_| rasterrocket::release_gpu_thread_state());
drop(pool);
```

A dropped pool's threads are not joined, so without it their thread-local destructors can call into the CUDA driver while the process runs its exit handlers. A thread's streams also hold their CUDA context, which is released only once no thread holds a stream on it. No-op without a CUDA feature (`nvjpeg`, `nvjpeg2k`, `gpu-icc`, `cache`, `gpu-deskew`). Never panics.

---

### `DeviceHold`

```rust
pub struct DeviceHold { /* … */ }
impl DeviceHold {
    pub fn open(config: &SessionConfig) -> Result<Self, RasterError>;
    pub fn is_empty(&self) -> bool;
}
```

A session holds its GPU devices only while it lives: sessions alive at once share one Vulkan device per `vulkan_device` selector, one Direct3D 12 device per `d3d12_device` selector and one CUDA context, the last of them to drop destroys it, and the next session builds it again (measured 74–89 ms for the Vulkan device on an RTX 5070, 50–65 ms on RADV, 52–62 ms for the CUDA context). `DeviceHold::open` opens what a session of `config` opens up front: the Vulkan device under `ForceVulkan`, the Direct3D 12 adapter of `d3d12_device` under `ForceD3d12`, and the CUDA context under `ForceCuda` and `ForceVaapi`. With `gpu_pages` it also opens the GPU-page devices on the policy's API, which a session under `Auto` opens on its first page; a session under `ForceVulkan` or `ForceD3d12` that lists its own device in `gpu_devices` shares the hold's backend for that entry rather than opening the device a second time; every session opened while the hold lives, under a configuration naming the same devices (including `raster_pdf` and `render_channel` under the same `PDF_RASTER_BACKEND`), shares them, and they are destroyed once the hold and those sessions have dropped. It fails as `open_session` fails for the configuration before reading a document; a GPU-page device that does not open under `Auto` is logged at `error` and not held. Under any other configuration it holds nothing (`is_empty`).

---

### `RasterError`

```rust
#[non_exhaustive]
pub enum RasterError {
    InvalidOptions(String),
    Pdf(rasterrocket_interp::InterpError),
    PageOutOfRange { page: u32, total: u32 },
    PageDegenerate { width: u32, height: u32 },
    PageTooLarge { width: u32, height: u32, bytes: u64, limit: u64 },
    PageExceedsCeiling { width: u32, height: u32 },
    PageAllocationFailed { width: u32, height: u32 },
    Deskew(String),
    OutputStrideTooSmall { stride: usize, row_bytes: usize },  // PageBuffer output
    OutputBufferTooSmall { len: usize, needed: usize },       // PageBuffer output
    InvalidPageGeometry(String),
    BackendUnavailable(String),  // forced backend failed to init
    ImageDecodeFailed(Vec<String>),
    ContentDecodeFailed(Vec<String>),
    ShadingFailed(Vec<String>),  // strict: a shading's Function yields no colour
    PageBudgetExceeded(String),
    Cancelled,                   // stopped through a CancelToken
    OutputProfile(OutputProfileError), // the output profile admits no conversion for the page
    OutputIntent(OutputIntentError),   // a soft proof finds no CMYK press for the document or page
    RenderPanic { page: u32, message: String },
}
```

`PageTooLarge` is a bitmap over the session's byte budget
(`SessionConfig::max_page_bytes`); `PageExceedsCeiling` a page side over
`MAX_PX_SIDE`, which no budget raises.
`PageAllocationFailed` is a page within both limits for which the allocator
refused a bitmap: the page's own, or a transparency group, soft mask,
knockout coverage, tiling-pattern cell or screen, or Type 3 glyph bitmap drawn
on it, or a snapshot of the canvas (the process does not abort); render it at
a lower resolution or in regions. `RenderPanic`
carries a per-page panic caught by `render_channel`, which happens only under
`panic = "unwind"`; with the `panic = "abort"` release profile a panic aborts
the process.

`ImageDecodeFailed`, `ContentDecodeFailed` and `ShadingFailed` are only returned under `strict`; they carry one message per failed image, content stream or shading. `Deskew` is returned when deskew is asked of a page larger than `deskew::DESKEW_MAX_SIDE` (32 768 px a side). `deskew` on a render whose output is not gray (`render_page_rgba`, `render_page_gray_alpha`, `render_page_cmyk`, `PageBuffer::render_rgb_into`, `render_rgba_into` and `render_gray_alpha_into`, `transparent_background` or `ColorModel::Cmyk` in `raster_pdf` and `render_channel`) is `InvalidOptions`, with the one message `deskew applies to gray pages only; render the page as gray (render_page_gray, render_gray_into) or turn deskew off`.

Implements `std::error::Error` with a `source()` chain: `RasterError::Pdf(e)`, `OutputProfile(e)` and `OutputIntent(e)` have `e` as their source. `BackendUnavailable` is only returned when the policy is `ForceCuda`, `ForceVaapi`, `ForceVulkan` or `ForceD3d12`.

---

### `MAX_PX_SIDE` / `DEFAULT_MAX_PAGE_BYTES` / `DEFAULT_BAND_BYTES`

```rust
pub const MAX_PX_SIDE: u32 = 1 << 24;
pub const DEFAULT_MAX_PAGE_BYTES: u64 = 1_800_000_000;
pub const DEFAULT_BAND_BYTES: u64 = 64 << 20;
pub fn validate_page_bytes(width: u32, height: u32, bytes_per_px: u32, max_bytes: u64) -> Result<(), RasterError>;
```

A render's canvas holds at most `SessionConfig::max_page_bytes` (default
`DEFAULT_MAX_PAGE_BYTES`, 1.8 GB: a 600 MP page of 8-bit RGB): width × height
× `CanvasLayout::bytes_per_px`, computed in `u64`: 3 bytes a pixel for 8-bit
RGB, 4 over a transparent backdrop, 8 on the ink canvas of a CMYK page or of
a page on the overprint preview (9 with alpha), and twice as many at 16
bits. Over it the render
is `PageTooLarge`. No page is more than `MAX_PX_SIDE` pixels on a side
(`PageExceedsCeiling`), whatever the budget: every pixel coordinate is then
exact in `f32` and coordinate sums stay inside `i32`.

Both are checked before the bitmap is allocated, and they bound what one call
draws: with a `RenderParams::region` the budget applies to the region, so a
page larger than it can be rendered as tiles, and `render_page_bands` bounds
the page by it while holding a band in memory. `validate_page_bytes` is the
check, shared with the comic decoders.

Images inside a page have their own budget, fixed in `rasterrocket-interp`
(not configurable): at most 131 071 px per declared side and 1 GiB of decoded
samples (width × height × components); a JPEG that holds its coefficient
planes whole (a progressive frame, or one coded in several scans) must fit
them, two bytes per sample, in what its samples leave. An image over it is refused before
decompression and reported in `PageDiagnostics::image_failures`, or fails the
page under `strict`. The other resource bounds of the interpreter are listed
under [Size limits](cli-reference.md#size-limits) in the CLI reference.

---

### Re-exports

```rust
pub use rasterrocket_color::SampleDepth;
pub use rasterrocket_interp::renderer::FinishedCanvas;
pub use rasterrocket_interp::renderer::{
    ContentFailure, DOCUMENT_IMAGE_BUDGET_BYTES, DevicePage, GlyphCounts, GroupSpaceApproximations,
    ImageFailure, MAX_FORM_DEPTH, PageDiagnostics,
};
pub use rasterrocket_interp::resources::ImageFilter;
pub use rasterrocket_interp::{
    FormType, InvalidRotation, MediaKind, PageBox, PageGeometry, RenderPurpose, Rotation,
    ScriptLocation, UnsupportedFeature,
};
pub use rasterrocket_interp::{
    IntentSet, OutputIntentError, OutputProfile, OutputProfileError, OutputSpace, PreserveBlack,
    ProofIntent, RenderingIntent, WorkingSpace,
};
pub use rasterrocket_parser::Password;
// crate-own constants: DEFAULT_VAAPI_DEVICE, GPU_PAGE_OPEN_WAIT, MAX_PX_SIDE,
// DEFAULT_MAX_PAGE_BYTES, DEFAULT_BAND_BYTES
```

`ImageFilter` identifies which decode filter was used for an embedded image (`Dct`, `Jpx`, `CcittFax`, `Jbig2`, `Flate`, `Raw`). Available through `PageDiagnostics` for routing decisions.

The crate root also exports `is_encrypted(path) -> bool` (a structural probe of the trailer for `/Encrypt`; `false` when the file cannot be read) and `is_encrypted_bytes(&[u8]) -> bool` (the same probe of a document in memory) and `decrypt_gate_declined_message() -> String` (the error text for an encrypted document whose decryption was not authorised), which the CLI uses for its liability prompt, and `gray8_to_rendered_page` / `gray8_into_rendered_page` / `gray16_into_rendered_page` / `validate_page_bytes`, the helpers every `RenderedPage` producer shares. `gray8_into_rendered_page(bmp, page_num, dpi, diagnostics)` takes the bitmap by value and hands a packed buffer (as `rgb_into_gray` returns) to the page without a copy; a padded one is copied row by row as by `gray8_to_rendered_page`. `gray16_into_rendered_page` does the same for a `Bitmap<Gray16>`, yielding a `PixelFormat::Gray16` page.

---

### Google Cloud Vision input (`encode_for_gcv`)

```rust
pub fn encode_for_gcv(
    page: &RenderedPage,
    budget: &GcvBudget,
) -> Result<GcvImage, GcvError>;

pub struct GcvBudget {
    pub max_base64_bytes: usize, // default 10 * 1024 * 1024
    pub min_quality: u8,         // default 60
    pub start_quality: u8,       // default 90
}

pub struct GcvImage {
    pub jpeg: Vec<u8>, // fitted baseline grayscale JPEG bytes
    pub quality: u8,
    pub width: u32,
    pub height: u32,
}

impl GcvImage {
    pub fn to_base64(&self) -> String;
}

pub enum GcvError {
    Unfittable { smallest_base64: usize, budget: usize },
    Encode(rasterrocket_encode::EncodeError),
    NotGray(PixelFormat),   // the page is not PixelFormat::Gray8
}
```

Encodes a `RenderedPage` to a grayscale JPEG guaranteed to fit Google Cloud Vision's *binding* request limit — 10 MB of base64 inside the `images:annotate` JSON request, **not** the 20 MB raw-file limit — decided deterministically with no network call.

Strategy: try `start_quality` at native resolution; binary-search quality down toward `min_quality`; only if the quality floor still overflows, box-downscale (aspect-preserving, never below GCV's ~1024 px OCR short-side floor, never over the 75 MP server-side-resize cap) and retry. If no candidate fits without breaking those floors, returns `GcvError::Unfittable` — never an over-budget or oversized payload. `GcvImage::to_base64()` reproduces exactly the length measured during fitting (the proxy is byte-exact against the real encoding).

`GcvBudget::default()` encodes the GCV limits directly. Leave `RasterOptions::deskew` off (the default) for this path — GCV deskews internally. The raw `jpeg` field is the universal artifact (disk, GCS `files:asyncBatchAnnotate`); `to_base64()` is the inline-`content` form. No HTTP/auth/batching is performed — rasterrocket renders pixels, it is not a GCV API client. See the [LLM Vision OCR Integration](https://github.com/pthomasfournet/rasterrocket/wiki/LLM-Vision-OCR-Integration) wiki for end-to-end examples.

The `rasterrocket-encode` crate also exposes the underlying codec directly:

```rust
pub fn jpeg_gray<P: Pixel>(bitmap: &Bitmap<P>, quality: u8)
    -> Result<Vec<u8>, EncodeError>;
```

Baseline 8-bit grayscale JPEG (`Gray8`/`Mono8` only; `quality` clamped 1–100). A plain codec with no consumer-specific policy.

The page writers record the resolution: `write_png(bitmap, dpi: Option<(f64, f64)>, out, icc)`, `write_png_bilevel` (bit depth 1, black below 128) and `PngWriter::new(out, size, color, dpi, icc)` write a `pHYs` chunk in whole pixels per metre; `write_jpeg(bitmap, options, dpi, out, icc)` writes the JFIF density in whole dots per inch (1 to 65535). `None` writes no `pHYs` chunk and the 1:1 JFIF aspect ratio; a resolution the format cannot hold is `EncodeError::InvalidInput`. `PngColor::Bilevel` streams a 1-bit PNG from gray bands.

---

## `rasterrocket-comic` crate — comic-archive input

Turns a comic/scan archive (`.cbz` ZIP, `.cb7` 7-Zip, `.cbt` TAR, `.cbr` RAR) of page
images into the same grayscale `RenderedPage` stream the PDF path produces. Use
it when your input is an image archive rather than a PDF.

### `open_comic`

```rust
pub fn open_comic(path: &Path, opts: &ComicOptions) -> Result<ComicPages, ComicError>

pub struct ComicPages { /* … */ }
impl Iterator for ComicPages { type Item = (u32, Result<RenderedPage, ComicError>); }
impl ExactSizeIterator for ComicPages {}
impl ComicPages {
    pub fn retain(&mut self, keep: impl FnMut(u32) -> bool);
    pub fn next_job(&mut self) -> Option<PageJob>;
}
pub fn open_comic_bytes(file_name: &str, bytes: Vec<u8>, opts: &ComicOptions) -> Result<ComicPages, ComicError>
pub struct PageJob { /* … */ }   // Send; owns what its page is rendered from
impl PageJob { pub fn render(self) -> Result<RenderedPage, ComicError>; }
```

`open_comic_bytes` opens an archive held in memory, choosing its container by the extension of `file_name` (`.cbz`, `.cb7`, `.cbt` or `.cbr`, in any case). `next_job` takes the next page unrendered, as a `PageJob` that can be rendered on any thread; the iterator is `next_job` followed by `PageJob::render`.

Opens an archive. `ComicPages` yields `(page_num, result)` in reading order and
decodes each page only when it is reached. The page-image entries are
decompressed when the archive is opened and each is held until its last page
is yielded, so memory is bounded by `max_archive_bytes` and falls as pages are
yielded; a per-page decode failure is an inner `Err` and does not abort the
rest, matching `raster_pdf`. `len()` is the number of pages still to come, and
`retain` drops pages before any is decoded.

**Content routing** (decided after the entry list is built):

- **Page images present** → render the images (numeric-aware natural sort by
  filename, magic-byte format detection over JPEG/PNG/WebP/TIFF/JPEG XL/AVIF). A multi-page
  TIFF entry yields one page per image, its reduced-resolution copies and
  transparency masks skipped. A JPEG page whose scan data stops short or is
  damaged is drawn as far as it decodes, the rest filled in, and logged at
  `warn`. Any PDF entry is ignored with a warning — images always win.
- **No images, exactly one PDF** → render that PDF's pages via
  `open_session_from_bytes` and `render_page_gray` (the PDF's own page
  numbering).
- **No images, two or more PDFs** → `ComicError::AmbiguousArchive`, naming the
  PDFs rather than guessing.
- **No images, no PDF** → `ComicError::NoImages`.

**Whole-archive errors (outer `Err`):** `InvalidOptions` (a non-finite or
non-positive `dpi`, `first_page` 0 or after `last_page`, `max_archive_bytes`
0), `Open`/`BadArchive` (unreadable or malformed container; a damaged or
encrypted member of a solid RAR), `ArchiveTooLarge { limit }` (the entries
decompress to more than `max_archive_bytes` in total), `AmbiguousArchive`, `Pdf` (an embedded PDF that cannot be opened or has no
page in the window), `NoImages`. **Per-page errors (inner `Err`):** `Decode`, `TooLarge`, `Deskew` (a page
with a side over `deskew::DESKEW_MAX_SIDE`, or a failed GPU rotation under `ForceCuda`), `Pdf`, and `BadArchive` for an entry that cannot be read on its own (a damaged
ZIP entry, a damaged or encrypted member of a non-solid RAR).

**`ComicOptions`** mirrors the relevant subset of `RasterOptions`: `dpi: f32`,
`first_page: u32`, `last_page: u32`, `deskew: bool`, `depth: SampleDepth`
(default `Eight`, `Gray8` pages; `Sixteen` yields `Gray16` pages: a PNG,
TIFF, JPEG XL or AVIF image of more than 8 bits a sample keeps its levels,
an 8-bit image is widened exactly, `v · 257`, an embedded PDF renders at 16
bits, and the page budget counts 6 bytes a pixel), `max_archive_bytes: u64`
(default `DEFAULT_MAX_ARCHIVE_BYTES`, 4 GiB: the most bytes all entries
together may decompress to), and `session:
SessionConfig` (default `SessionConfig::default()`), the session an embedded
PDF is opened with: its overprint, antialiasing, annotation, purpose, font,
decryption and backend settings; its `rotation` also turns every page image, before deskew. The DPI rides along on each
`RenderedPage` for downstream OCR; comic images carry no intrinsic resolution.
Both `ComicOptions` and `ComicError` are `#[non_exhaustive]`: build the options
with `ComicOptions::new(dpi)` or `ComicOptions::default()` and assign the fields
to change, and give a `match` on `ComicError` a wildcard arm.

**Hardening:** entries are read under a 512 MiB per-entry decompressed-size cap
and the archive-wide `max_archive_bytes` (decompression-bomb defence), and image dimensions are validated before a codec
allocates its pixel buffer (decode-bomb defence). `.cb7` and `.cbr` archives
are decoded whole when opened, with decoder allocations (LZMA and RAR
dictionaries, the PPMd model) bounded by 1 GiB and the RAR reader's whole
workspace by 2 GiB.

---

## `rasterrocket-camera` crate — camera-capture page cleanup

Cleanup of camera-captured pages, measured on 8-bit grayscale images and
replayed at 8 or 16 bits: page crop, perspective rectification, double-page
spread splitting and (experimental) dewarp. Independent of PDF: it depends on `std` alone, runs on the calling
thread with no global state, and returns on any input without panicking.
Deskew is not part of it (see the `deskew` module above). Three
OCR-preparation operations for scanned pages sit beside the pipeline:
background normalisation, binarisation and scan border cleanup (see
[OCR preparation](#ocr-preparation)).

Each stage is a detector followed by a resampler. A detector that is not
confident declines and the page passes through unchanged, with the reason in
the `Report`; a flatbed scan or born-digital page comes back as given. Only
malformed input is an error.

### Types

```rust
pub trait Sample: Copy + Default + Eq + Debug + 'static { const MAX: Self; /* sealed: u8, u16 */ }
pub struct GrayView<'a, S: Sample = u8> { pub data: &'a [S], pub width: usize, pub height: usize, pub stride: usize }
pub struct GrayPage<S: Sample = u8> { pub width: usize, pub height: usize, pub data: Vec<S> }   // packed, stride == width
pub struct Rect { pub x: usize, pub y: usize, pub width: usize, pub height: usize }
pub struct Quad { pub corners: [[f64; 2]; 4] }   // TL, TR, BR, BL, at the page edge, source pixels

#[non_exhaustive]
#[derive(Default)]                                // every stage off
pub struct CameraOptions { pub crop: bool, pub rectify: bool, pub split_spreads: bool, pub dewarp: bool }

impl<'a, S: Sample> GrayView<'a, S> {
    pub const fn packed(data: &'a [S], width: usize, height: usize) -> Self;
    pub const fn validate(&self) -> Result<(), CameraError>;   // lengths and strides in samples
}
impl<S: Sample> GrayPage<S> { pub fn view(&self) -> GrayView<'_, S>; }
impl Quad { pub fn bounding_rect(&self, width: usize, height: usize) -> Option<Rect>; }
impl CameraOptions { pub const fn all() -> Self; pub const fn any(&self) -> bool; }
```

`CameraOptions` is `#[non_exhaustive]`: start from `CameraOptions::default()`
or `CameraOptions::all()` and set fields.

### `process`

```rust
pub fn process(img: GrayView<'_>, opts: &CameraOptions) -> Result<Processed, CameraError>;

pub struct Processed { pub pages: Vec<GrayPage>, pub report: Report }
```

Runs crop or rectify, then split, then dewarp on each resulting page.
`pages` holds one page, or the left then the right page of a split spread.
`rectify` supersedes `crop` when both are on and the warp applies; when the
quad is found but cannot be rectified, `crop` applies instead.

```rust
pub struct Report {
    pub crop: Outcome,
    pub rectify: Outcome,
    pub split: Outcome,
    pub dewarp: Vec<Outcome>,          // one per output page
    pub quad: Option<Quad>,            // when crop or rectify ran and found one
    pub crop_rect: Option<Rect>,       // when the crop applied
    pub gutter: Option<Gutter>,        // when the split applied
    pub homography: Option<[f64; 9]>,  // row-major, rectified pixel → source pixel
    pub timings: Timings,              // quad, crop, rectify, split, dewarp, output, total
    /* private: the replayable transform */
}
impl Report {
    pub fn replay<S: Sample>(&self, img: GrayView<'_, S>) -> Result<Vec<GrayPage<S>>, CameraError>;
}

#[non_exhaustive]
pub enum Outcome { Off, Applied, Skipped(String) }   // Outcome::applied()

pub struct Gutter {
    pub x: usize,                  // first column of the right page, in the split stage's input
    pub source_x: Option<usize>,   // the same column in the source image (no rectification)
    pub source: [[f64; 2]; 2],     // top and bottom ends of the cut, source pixels
}
```

`Report::replay` applies the computed geometry to another image of the same
size — e.g. each channel of the colour image whose luma was processed, or the
16-bit image whose 8-bit levels were processed (`GrayView<'_, u16>`), which it
resamples at 16 bits — and reproduces `pages` when given the processed image
itself. `process` and the stage functions take 8-bit views.

### Stage functions

```rust
pub fn detect_page_quad(img: GrayView<'_>) -> Result<Detected<Quad>, CameraError>;
pub fn crop(img: GrayView<'_>, rect: Rect) -> Result<GrayPage, CameraError>;
pub fn rectify(img: GrayView<'_>, quad: &Quad) -> Result<Detected<GrayPage>, CameraError>;
pub fn find_gutter(img: GrayView<'_>) -> Result<Detected<usize>, CameraError>;
pub fn split_at(img: GrayView<'_>, x: usize) -> Result<[GrayPage; 2], CameraError>;
pub fn dewarp(img: GrayView<'_>) -> Result<Detected<GrayPage>, CameraError>;

pub enum Detected<T> { Found(T), Declined(String) }   // Detected::found() -> Option<T>
```

| Stage | Detects | Declines when |
|---|---|---|
| page quad (crop, rectify) | A light page on a darker surround: Otsu mask, maximum-area hull quadrilateral, edges refined to sub-pixel at the paper/surround step. `crop` cuts to the pixels whose centres the quad spans; `rectify` maps the quad onto an upright rectangle at the aspect ratio recovered from its vanishing geometry. | The page covers > 96 % of the frame (flatbed); surround on fewer than 3 sides; paper/surround contrast < 35 levels; quad not convex, angles outside 50°–130°, or IoU with the mask < 0.92. |
| split | Width/height in 1.15–2.6 and a gutter in the central 30 %: a paper-level shadow valley over ink-free columns, or an ink-free run ≥ 6 % of the width bounded by text. | No such gutter, or either half carries < 15 % of the ink. A two-column landscape page (column gap < 6 %) is not split. |
| dewarp | Text lines tracked by their baselines and fitted to `u = h + a(t) + b(t)·h`, a page bent about a vertical axis seen from above; columns are resampled vertically. | Fewer than 5 tracked lines; lines already straight; the model does not explain the curvature; the correction would bend lines by < max(1.5 px, 0.1 line pitch). |

**Experimental — `dewarp`:** corrects vertical displacement only; glyphs
foreshortened near a spine stay narrow. Validated on synthetic curls of real
pages, not yet on real captures.

### OCR preparation

```rust
pub struct Background { /* per-pixel gain; private */ }
pub const fn background_tile_px(dpi: u32) -> usize;   // dpi / 8, at least 8
pub fn measure_background(img: GrayView<'_>, tile_px: usize) -> Result<Detected<Background>, CameraError>;
impl Background {
    pub fn normalize<S: Sample>(&self, img: GrayView<'_, S>) -> Result<GrayPage<S>, CameraError>;
}

pub enum Threshold { Fixed(u8), Sauvola { window_px: u32, k_per_mille: u16 } }
impl Threshold { pub const fn sauvola_at(dpi: u32) -> Self; }   // window 0.33 × dpi, k = 0.34
pub fn binarize(img: GrayView<'_>, method: Threshold) -> Result<GrayPage, CameraError>;   // 0 or 255

pub struct BorderMask { /* column runs per row, source pixels; private */ }
pub fn find_border_junk(img: GrayView<'_>) -> Result<Detected<BorderMask>, CameraError>;
impl BorderMask {
    pub fn whiten<S: Sample>(&self, img: &mut GrayPage<S>) -> Result<(), CameraError>;
}
```

Each operation is measured on an 8-bit luma view. `Background::normalize` and
`BorderMask::whiten` apply the measurement to any image of the same size, at
its own depth: the luma itself, each channel of the colour image it was taken
from, or a 16-bit version (`SizeMismatch` otherwise, as for `Report::replay`).
The arithmetic is integer throughout, so output is identical on every
platform. Sizes are given in pixels; `background_tile_px` and
`Threshold::sauvola_at` derive them from a resolution.

| Operation | Method | Declines when |
|---|---|---|
| `measure_background` | Per tile of `tile_px`², the mean of the samples above a floor (the larger of luma 80 and 7/8 of the tile's mean above 80, which keeps anti-aliased glyph edges out); a tile with under a third of its samples above the floor is a hole, filled from its neighbours. The map is smoothed by a box two tiles wide on each side, centred, so a linear illumination is kept exactly, interpolated bilinearly at every pixel and inverted into a gain. `normalize` writes `min(MAX, v · 255 · 16 / (15 · bg))`: a sample at or above 15/16 of its local background becomes white, and ink keeps its tone relative to the paper. | The page spans fewer than 4 × 4 tiles; more than half the tiles are holes; the background is already white (5th percentile of the map ≥ 250), as on a born-digital or already normalised page. |
| `binarize` | `Fixed(t)`: samples below `t` are black. `Sauvola`: black when `v < m · (1 + k · (s / 128 − 1))`, `m` and `s` the mean and standard deviation over the window centred on the pixel (side `window_px`, rounded up to odd, at most 2047), truncated at the image edges to the samples it covers. Window sums come from `u64` integral images; the comparison is exact in integers. Samples below 32 are always black, so a solid area stays solid. Raising `k` never adds black. | Never; a zero window is `ZeroSize`. |
| `find_border_junk` | On a plane downsampled to at most 1600 px a side, every dark (luma < 128) 4-connected region that touches the image edge and lies wholly within a band 6 % of the shorter side deep along the edges it touches. The mask is dilated by one analysis pixel and kept as runs at source resolution; `whiten` sets them to white, keeping the page size. | More than half the plane is dark (a reversed-out or full-bleed page); no dark region touches the edge; every one reaches past the band (a bar or rule running into the page). |

**Limits.** Background normalisation reads a light photograph or tint as
background and loses its gradual shading; a paper tint survives in colour,
since one gain scales every channel. Apply it to scans, not to pages with a
text layer (`PageDiagnostics::has_visible_text`). Sauvola assumes dark ink on
light paper: invert a reversed-out page first. It thresholds a halftoned
photograph dot by dot. Border cleanup leaves punch holes, which do not reach
the edge, and soft gray gutter shadows, which are not dark.

### `CameraError`

Implements `std::error::Error` and `Display`; each message names the bad
value and the requirement. Variants: `ZeroDimension`, `TooLarge` (a side above
`MAX_DIMENSION` = 2³⁰), `StrideTooSmall`, `BufferTooShort` (fewer than
`stride · (height − 1) + width` bytes), `Overflow`, `RectOutOfBounds` (`crop`),
`SplitOutOfRange` (`split_at`), `SizeMismatch` (`Report::replay`,
`Background::normalize`, `BorderMask::whiten`), `ZeroSize` (a zero
`tile_px` or `window_px`).

### Timing

`cargo run --release -p rasterrocket-camera --example timing` processes a
synthetic 4656 × 3496 spread on a desk, single-threaded. On a Ryzen 9 9900X3D:
crop + split 11 ms; all stages 345 ms (rectify 200 ms, dewarp of both pages
137 ms).

---

## `rasterrocket-interp` crate — lower-level API

Direct use of `rasterrocket-interp` is not required for most consumers. Use it when building a custom render loop or accessing document metadata without rendering.

### `open`

```rust
pub fn open(path: impl AsRef<Path>) -> Result<pdf::Document, InterpError>
pub fn open_with(path: impl AsRef<Path>, options: &pdf::OpenOptions) -> Result<pdf::Document, InterpError>
pub fn open_bytes(bytes: Vec<u8>) -> Result<pdf::Document, InterpError>
pub fn open_bytes_with(bytes: Vec<u8>, options: &pdf::OpenOptions) -> Result<pdf::Document, InterpError>
pub fn open_shared_with(
    bytes: impl AsRef<[u8]> + Send + Sync + 'static,
    options: &pdf::OpenOptions,
) -> Result<pdf::Document, InterpError>
```

Opens and validates a PDF, from a path or from bytes. `open` and `open_bytes` refuse an encrypted document; the `_with` forms take a `pdf::OpenOptions` (`OpenOptions::new(decrypt_authorized: bool, password: Option<Password>)`, or `default()` to refuse encrypted input; its `read_into_memory` field, default `false`, reads the file into memory instead of mapping it, a file whose buffer cannot be allocated failing with a `PdfError::Io` of kind `OutOfMemory`), with the same meaning as the `SessionConfig` fields of those names: an authorised encrypted document is decrypted in process as its objects are read (the `pdf` crate's Standard security handler), and `Document::is_encrypted()` then reports `true`. `open_shared_with` reads bytes the caller keeps shared in place, without a copy (`pdf::Document::from_shared` / `from_shared_with`; `bytes.as_ref()` must return the same bytes on every call). `pdf::is_encrypted(path) -> Result<bool, PdfError>` probes the trailer for `/Encrypt` without opening the document, and `pdf::is_encrypted_bytes(&[u8])` does so for a document in memory.

 If a catalog JavaScript entry point is detected (checked locations: `/OpenAction`, catalog `/AA`, `/Names/JavaScript`, `/AcroForm/AA`, the `/AA` of each field in the `/AcroForm/Fields` tree, each flagged only when genuinely a JavaScript action: `/S /JavaScript`, or `/S /Rendition` with a `/JS` script, also along an action's `/Next` chain), a loud `WARN` is emitted per location and the document still opens — rasterrocket has no JavaScript engine and never executes `/JS`, so a script's structural presence cannot change the static rendered page. No JS is parsed or evaluated. A dynamic XFA form and a portable collection are warned about the same way. No page object is read at open; page-level `/AA`, annotation `/A`/`/AA` and the `/AA` of the fields on an annotation's `/Parent` chain are checked per page by `page_has_javascript(doc, page_id) -> bool`. The same checks return typed data: `document_unsupported_features(doc) -> Vec<UnsupportedFeature>` and `page_unsupported_features(doc, page_id) -> Vec<UnsupportedFeature>`; see [`UnsupportedFeature`](#unsupportedfeature).

The returned `pdf::Document` comes from the in-tree parser crate (`rasterrocket-parser`, imported as `pdf`). It is a lazy reader over one contiguous slice (a mapping of a file of 256 KiB or more, owned bytes, or the caller's shared bytes): opening the document parses only the xref table and trailer; individual objects resolve on demand via byte-offset seek with a per-object `Arc` cache shared safely across worker threads. Each object stream (`ObjStm`) is decompressed once and its directory reused. `Document::linearization_hints()` returns `Option<LinearizationHints>`: the `/Linearized` dictionary of a Fast Web View file, found as the file's first object, or `None`, also for a file updated after it was linearized (its `/L` no longer the file's length). The page count is always the page tree's.

### `page_count`

```rust
pub fn page_count(doc: &Document) -> u32
```

Total pages. Saturates at `u32::MAX` for pathological documents (> 4 billion pages).

### `page_size_pts`

```rust
pub fn page_size_pts(doc: &Document, page_num: u32, page_box: PageBox) -> Result<PageGeometry, InterpError>
pub fn page_size_pts_by_id(doc: &Document, page_id: ObjectId, page_box: PageBox) -> Result<PageGeometry, InterpError>
```

Returns geometry for page `page_num` (1-based), or the page object `page_id`, measured on `page_box` (see [`PageBox`](#pagebox)). `width_pts` and `height_pts` are already adjusted for rotation and `UserUnit` scaling — use them directly as output bitmap dimensions. `prescan_page(doc, page_num, page_box)` takes the box for the page width its PPI estimate divides by.

Falls back to US Letter (612 × 792 pt) when neither `CropBox` nor `MediaBox` can be read.

### `parse_page_content`

```rust
pub fn parse_page_content<'a>(doc: &Document, page_id: ObjectId, bytes: &'a [u8]) -> Vec<Operator<'a>>
```

Parses `bytes`, the page's content from `Document::page_content(page_id)`, into its operator sequence. An inline image that names its colour space from the page's resources has its data length computed through them. The operators borrow `bytes` wherever an operand needs no decoding (`Operator::into_owned` copies them out), so the caller keeps the content alive while it uses them. Typically called internally by the renderer; exposed for tooling (e.g. the `dump_ops` example).

### `PageGeometry`

```rust
#[non_exhaustive]
pub struct PageGeometry {
    pub width_pts: f64,   // output width in PDF points (rotation + UserUnit applied)
    pub height_pts: f64,  // output height in PDF points (rotation + UserUnit applied)
    pub rotate_cw: u16,   // 0, 90, 180, or 270
    pub user_unit: f64,   // UserUnit scale factor, validated to [0.1, 10.0]
    pub origin_x: f64,    // lower-left corner of the selected box, user space
    pub origin_y: f64,
    pub media_box: Option<[f64; 4]>,  // [llx, lly, urx, ury], own or inherited
    pub crop_box: Option<[f64; 4]>,   // as authored, before intersecting with media_box
}
impl PageGeometry {
    pub fn size_px(&self, scale_x: f64, scale_y: f64) -> (u32, u32)
    pub const fn rotated(self, extra: Rotation) -> Self
}
```

Dimensions are swapped for 90°/270° rotations so that `width_pts` always corresponds to the horizontal extent of the rendered bitmap. Re-exported as `rasterrocket::PageGeometry` and returned by `RasterSession::page_geometry`, measured on the session's `SessionConfig::page_box`. `media_box` and `crop_box` describe the page whichever box is selected.

`size_px(scale_x, scale_y)` is the rendered pixel size, `(width_pts × scale_x).round()` by `(height_pts × scale_y).round()`, saturating at `u32::MAX` (a NaN or non-positive product gives 0). `rotated(extra)` is the geometry with a [`Rotation`](#rotation) added to the page's: `rotate_cw` the sum modulo 360, width and height exchanged on a quarter turn, origin, `UserUnit` and boxes unchanged.

### `InterpError`

```rust
#[non_exhaustive]
pub enum InterpError {
    Pdf(pdf::PdfError),
    PageOutOfRange { page: u32, total: u32 },
    InvalidPageGeometry(String),
}
```

Implements `std::error::Error`; `InterpError::Pdf(e)` has `e` as its `source()`. A `match` on it needs a wildcard arm. A missing resource is not an error: the operator that names it draws nothing (a missing font draws in Helvetica and is listed in `PageDiagnostics::missing_fonts`), and the per-page operator budget is reported by the `rasterrocket` crate as `RasterError::PageBudgetExceeded`.

### Other public types

`PageView`, `PrefetchConfig` and the resource and content types (`ExtGStateParams`, `ShadingPattern`, `SoftMaskParams`, `FormXObject`, `TransparencyGroupParams`, `ResourceKey`, `Operator`, `TextArrayElement`) are `#[non_exhaustive]`: they cannot be built with a struct literal or matched exhaustively outside the crate. A `PageView` is built with `PageView::whole(&geometry, (scale_x, scale_y))`, the whole page with an initial CTM of `scale × UserUnit` device pixels per unit, and narrowed with `.window(x, y, width, height)` to a device-pixel window of it; `PageRenderer::with_view(&view, doc, page_id)` renders it on an RGB canvas. `PageRenderer::with_page_group(&view, doc, page_id, group, &canvas)` also takes the page group (`page_group(doc, page_id)`, ISO 32000-1 §11.4.7) and a `CanvasRequest`: `Rgb`; `ProcessRgb { overprint }`, a canvas of four process and four spot inks, its colours that are not ink separated by the inverse of the process-ink model, converted to RGB at the finish and simulating overprint when `overprint` (an overprint preview, or a page whose transparency group composites in CMYK); or `Cmyk { overprint, separation }`, the same canvas, simulating overprint only when `overprint`, finished as CMYK by `try_finish_keeping` (`FinishedCanvas::Cmyk8` / `Cmyk16`), its colours that are not ink separated by `Separation::Device` (ISO 32000-2 §10.4.2.3) or `Separation::Profile` (a CMYK output profile's `separation`). The canvas is allocated once and its kind cannot change afterwards. `page_uses_overprint(doc, page_id, annotations) -> bool` reports whether an `ExtGState` reachable from the page's resources (through forms, patterns and Type 3 fonts, to the renderer's form depth) or, when `annotations`, from its annotations' appearance streams sets `OP` or `op`; the `rasterrocket` crate asks for `ProcessRgb { overprint: true }` when `overprint && page_uses_overprint(..)`, for `ProcessRgb { overprint: false }` when `page_group_space(doc, page_id)` (the page group's `/CS`, ISO 32000-2 §11.4.7) is a CMYK space, and for `Cmyk` for every page of a CMYK render. A form's transparency group composites in the space of its `/CS` (`TransparencyGroupParams::space`, a `GroupSpace`); `PageDiagnostics::approximated_group_spaces` reports a CMYK ICC or non-sRGB RGB group space composited in an approximation. `PageRenderer::set_document_images(Arc<DocumentImages>)` shares image decodes between the renderers of one document's pages; `DocumentImages::new(budget)` bounds what the store keeps (0 keeps nothing; `DocumentImages::default()` keeps up to `DOCUMENT_IMAGE_BUDGET_BYTES`), and `budget()` reads it back. A store is keyed by object number, so it serves the first document whose renderer it is given to (identified by `Document::instance_id()`, distinct for every `Document` constructed in the process): given to a renderer of another document it is ignored with a warning, and that page decodes its own images. `PageRenderer::set_document_fonts(Arc<DocumentFonts>)`, which shares resolved fonts the same way, binds its store to one document likewise: given to a renderer of another document it is ignored with a warning, and that page resolves its own fonts. A bitmap the render needs beyond the page canvas (a transparency group, soft mask, knockout coverage canvas, tiling-pattern cell or screen, Type 3 glyph canvas, or a snapshot of the canvas) that the allocator refuses stops the render with `RenderAbort::AllocationFailed { width, height }` from `budget_status()`, which the `rasterrocket` crate reports as `RasterError::PageAllocationFailed`; `try_with_page_group`, `try_set_page_group` and `try_finish` report a refused page canvas as `raster::BitmapAllocError`, where `with_page_group`, `set_page_group` and `finish` panic. `ExtGStateParams` carries `overprint_stroke` (`OP`), `overprint_fill` (`op`, defaulting to `OP`) and `overprint_mode: Option<OverprintMode>` (`OPM`: `OverprintMode::Zero` or `NonZero`; `None` when absent or not 0 or 1). `cache::spawn_prefetch` (`cache` feature) returns `io::Result<PrefetchHandle>`: a prefetch thread that cannot be started is an error, not a panic. `PrefetchConfig::scale` and `PrefetchConfig::rotation` (defaults `None` and `Rotation::None`) carry the same scale and the session rotation.

---

## Hardware compatibility

### CPU

**Supported:** x86-64 (AMD and Intel) and `aarch64` (ARM / Apple Silicon). The hardware each claim was tested on is listed in the README's [Tested hardware](../README.md#tested-hardware); anything else is untested.

**x86-64:**
- AVX2 SIMD blend and fill paths are runtime-detected with a scalar fallback.
- AVX-512 (`avx512f/bw/vl/dq/vnni/vpopcntdq` and related sub-extensions) is activated by building with `-C target-cpu=native` on a compatible CPU. Developed and benchmarked on an AMD Ryzen 9900X3D.
- All Intel consumer CPUs (Alder/Raptor/Arrow Lake) have AVX2; AVX-512 is disabled on them — Xeon only. Of Intel CPUs only an i7-8700K (Coffee Lake, AVX2) is tested.

**aarch64:**
- NEON is used unconditionally (mandatory on all ARMv8-A). No runtime detection needed.
- Tested on an Apple A18 Pro (macOS) and in CI on GitHub's arm64 Linux and Apple-silicon macOS runners (CPU only); untested on M-series hardware and on other aarch64 Linux machines. No Apple Metal native backend yet (Vulkan via `MoltenVK` covers Apple).

### GPU

**NVIDIA (CUDA 13.x driver; the driver-API binding is `cuda-13030`):** tested on an RTX 5070 (`sm_120`), and on an RTX 2080 SUPER (`sm_75`) from an nvcc 12.4 build under Linux on driver 570 and from a CUDA 13.5 build under Windows (see the README's [Tested hardware](../README.md#tested-hardware)); other NVIDIA GPUs are untested.

| Feature flag | Minimum requirement | Notes |
|---|---|---|
| `nvjpeg` | CUDA-capable NVIDIA GPU | `libnvjpeg.so` ships with the CUDA 13 toolkit |
| `nvjpeg2k` | CUDA-capable NVIDIA GPU | `libnvjpeg2k.so` is a separate download; the build looks under `/usr/lib/x86_64-linux-gnu/libnvjpeg2k/13`, headers under `/usr/include/libnvjpeg2k/13` unless `NVJPEG2K_INCLUDE_DIR` names another directory |
| `gpu-icc` | CUDA-capable NVIDIA GPU | CUDA runtime only |
| `gpu-deskew` | CUDA-capable NVIDIA GPU | Requires CUDA NPP: `libnppig.so` + `libnppc.so` |
| `cache` | CUDA-capable NVIDIA GPU, a Vulkan 1.3+ device with `vulkan`, or a Direct3D 12 adapter with `d3d12` (Windows) | Device-resident 3-tier image cache, on the session's backend |

`cudarc` is pinned to the `cuda-13030` driver-API binding; a CUDA 13.x driver is required.

**Vulkan compute (cross-vendor — NVIDIA, AMD, Intel (tested on a UHD 630 under Windows only), Apple via `MoltenVK`):**

| Feature flag | Supported hardware | Libraries required |
|---|---|---|
| `vulkan` | Any Vulkan 1.3+ device | Vulkan ICD (e.g. `mesa-vulkan-drivers`, NVIDIA driver); `slangc` from the LunarG Vulkan SDK at *build* time. |

Vulkan covers whole-page rendering (`gpu_pages`: fills, strokes, text, images, forms, annotation appearances, path clips and soft masks, byte-identical to the CPU at 8 and 16 bits), tested on an RTX 5070, an RTX 2080 SUPER, an AMD Raphael iGPU (RADV), an Intel UHD 630 (under Windows), an A18 Pro's GPU (MoltenVK) and llvmpipe. Intel GPUs under Linux (ANV), M-series Macs and datacenter GPUs are untested.  With `cache`, the device image cache runs on the Vulkan device under `--backend vulkan`; `nvjpeg` stays CUDA-only.

**VA-API (Linux iGPU/dGPU — AMD VCN; Intel Quick Sync and Intel Arc untested):**

| Feature flag | Supported hardware | Libraries required |
|---|---|---|
| `vaapi` | AMD VCN (tested on Raphael, VCN 4.0); Intel UHD 630+ and Intel Arc (untested) | `libva.so.2`, `libva-drm.so.2` |

VA-API provides hardware JPEG baseline decode; CMYK and progressive JPEGs fall through to the CPU. When both `nvjpeg` and `vaapi` are active, nvJPEG is tried first. Both share the `GPU_JPEG_THRESHOLD_PX` gate, which is `u32::MAX`, so in a default build neither decodes anything and every JPEG is decoded on the CPU (see [GPU dispatch thresholds](#gpu-dispatch-thresholds)).

Under `BackendPolicy::Auto`, GPU initialisation failures at runtime log a warning and fall back to the CPU — no error is returned, rendering continues normally. The `Force*` policies turn them into `RasterError::BackendUnavailable`.

### Platform support

See the platform table in [Getting Started](getting-started.md#platform-support)
and the guides for [Linux](platforms/linux.md), [macOS](platforms/macos.md) and
[BSD and Windows](platforms/other.md).

## Feature flags

### `rasterrocket` features

| Feature | Requires | Effect |
|---|---|---|
| `nvjpeg` | CUDA 13, `libnvjpeg.so` | GPU JPEG decode (DCTDecode). Wired but disabled: `GPU_JPEG_THRESHOLD_PX` is `u32::MAX`, so JPEGs decode on the CPU (zune-jpeg). |
| `nvjpeg2k` | CUDA 13, `libnvjpeg2k.so` | GPU JPEG 2000 decode (JPXDecode). Dormant under `Auto` (`GPU_JPEG2K_THRESHOLD_PX` = `u32::MAX`); under `ForceCuda` it dispatches from 2 097 152 px (`GPU_JPEG2K_ATTACHED_THRESHOLD_PX`). `PDF_RASTER_JPX_THRESHOLD` sets the pixel area for both. Below that area, or for a codestream the device does not take, the CPU decoder (pure-Rust `hayro-jpeg2000`) decodes. The device takes one or three unsigned components of 8 bits, or of 9 to 16 on a 16-bit canvas, on the image grid, under the reversible 5-3 wavelet only (under the irreversible 9-7 one nvJPEG2000's samples differ from the CPU's by a level), with no palette, channel definition or colour conversion, no code-block style but predictable termination, and every tile complete; a draw at half the image's size or less on both axes, which the CPU decodes at a reduced resolution level, decodes on the CPU as well. A decline is a warning, once per kind. |
| `gpu-icc` | CUDA 13, `vulkan`, or `d3d12` (Windows) | An image in an ICC-based CMYK space converts through the grid the CPU tabulates for the profile (`color::grid::CmykGrid`, its cells aligned with the profile's CLUT) on the session's GPU backend (Vulkan or Direct3D 12 when attached, else CUDA) from 128 000 px (`PDF_RASTER_ICC_CLUT_THRESHOLD` overrides it); the device output is byte-identical to the CPU evaluation of the table. Smaller images, those a Vulkan or Direct3D 12 device whose kernel fails a one-time exactness check declines, and those a device conversion fails on convert through the profile transform, as without a backend; a GPU session therefore differs from a CPU session only in the images counted in `PageDiagnostics::device_icc_conversions`. |
| `gpu-deskew` | CUDA 13, CUDA NPP | GPU bilinear rotation (nppiRotate) under `ForceCuda`, its border recomputed with the CPU kernel, within 2 levels of the CPU elsewhere at 8 bits and 8 levels of 65 535 at 16 (see `deskew::apply`). Without the feature, the CPU rotation. |
| `cache` | CUDA 13, `vulkan`, or `d3d12` (Windows) | Device-resident image cache (3-tier VRAM/host/disk). Cross-document content-hash dedup. Runs on the session's backend: CUDA, Vulkan under `--backend vulkan` with the `vulkan` feature, or Direct3D 12 under `ForceD3d12` with the `d3d12` feature; an `Auto` session has none. A JPEG drawn below its resolution (every scanned page rendered under its scan resolution) is neither looked up nor inserted: it is area-averaged on the host from host-resident samples. Disk-tier persistence is opt-in via `PDF_RASTER_CACHE_DIR`. |
| `vaapi` | `libva.so.2`, `libva-drm.so.2` | VA-API JPEG baseline decode on Linux iGPU/dGPU (tested on an AMD iGPU; untested on Intel); adds `BackendPolicy::ForceVaapi` and `--backend vaapi`. Shares the disabled `GPU_JPEG_THRESHOLD_PX` gate with `nvjpeg`, so JPEGs still decode on the CPU. |
| `vulkan` | Vulkan 1.3+ ICD; LunarG `slangc` at build time. | Vulkan compute backend. Whole-page rendering (`gpu_pages`; with `gpu-icc`, the ICC CMYK lookup) is written for any Vulkan 1.3+ device (NVIDIA, AMD, Intel (tested on a UHD 630 under Windows only), Apple via `MoltenVK`); see [Tested hardware](../README.md#tested-hardware). With `cache`, the device image cache and its blit kernel run on the Vulkan device. No nvJPEG under this backend. |
| `d3d12` | Windows; a Direct3D 12 adapter with shader model 6.0 or later, or WARP; slangc 2026.20 and DXC 1.9.2609 at build time | Direct3D 12 compute backend: adds `ForceD3d12` sessions (`SessionConfig::d3d12_device`, `PDF_RASTER_D3D12_DEVICE`), which with `gpu-icc` run the ICC CMYK lookup on the adapter, with `cache` the device image cache and its blit kernel, and with `gpu_pages` whole pages on one or several adapters (`gpu_devices`). Elsewhere it compiles the DXIL and `ForceD3d12` is refused with `BackendUnavailable`. The CLI forwards it (`--features d3d12`, `--backend d3d12`). |
| `gpu-validation` | CUDA device at test time | Enables GPU vs CPU parity tests (`cargo test -p rasterrocket-gpu --features gpu-validation`). |
| `bundled-fonts` | — (**on by default**) | Noto fallback faces for Greek, Cyrillic, Hebrew, Arabic, Devanagari, Thai, Bengali, Tamil, Armenian, Georgian, Ethiopic and symbols compiled into the library (~0.75 MB, SIL OFL 1.1; provenance in `crates/pdf_interp/fonts/README.md`). A character of a non-embedded font that neither its substitute nor any installed face covers is drawn from them; an installed face is always preferred. Disable with `default-features = false` for a slim build. |

Under `Auto`, GPU initialisation failures log a warning and fall back to CPU — they do not return errors.  `cudarc` is pinned to the `cuda-13030` driver-API binding, so a CUDA 13.x driver is required.

### GPU dispatch thresholds

| Path | Threshold | Constant |
|---|---|---|
| nvJPEG (DCTDecode) | **disabled** (`u32::MAX`) | `GPU_JPEG_THRESHOLD_PX` |
| nvJPEG2000 (JPXDecode), `Auto` | **dormant** (`u32::MAX`; `PDF_RASTER_JPX_THRESHOLD` sets an area) | `GPU_JPEG2K_THRESHOLD_PX` |
| nvJPEG2000 (JPXDecode), `ForceCuda` | ≥ 2 097 152 px (area; `PDF_RASTER_JPX_THRESHOLD` overrides) | `GPU_JPEG2K_ATTACHED_THRESHOLD_PX` |
| GPU ICC CLUT | ≥ 128 000 px (area) | `GPU_ICC_CLUT_THRESHOLD` |

The nvJPEG threshold is `u32::MAX` **by design**, so JPEG decode runs on the
CPU: nvJPEG's `GPU_HYBRID` backend lost to 24-thread CPU decode on consumer
hardware. Under `Auto` nvJPEG2000 is dormant too: a thread's decoder build
costs more process time than the device saves on a single page, and over
runs of pages the device wins on some documents and ties or loses on others.
Set `PDF_RASTER_JPX_THRESHOLD` to a pixel area to enable it. `ForceCuda`,
which asks for the device, keeps the 2 097 152 px area threshold. A session for which a set `PDF_RASTER_JPX_THRESHOLD` cannot act (a build without `nvjpeg2k`, or the `cpu`, `vulkan`, `d3d12` or `vaapi` backend) logs a warning saying so, once per process.

