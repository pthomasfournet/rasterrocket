# API Reference

## `rasterrocket` crate — public API

The `rasterrocket` crate is the integration entry point. It wraps `rasterrocket-interp` and `rasterrocket-render` behind a simple, stable API.

**Extensibility.** These public types are `#[non_exhaustive]`, so fields and variants can be added in a minor release: `RasterError`, `BackendPolicy`, `RenderPurpose`, `FormType`, `ImageFilter`, `GcvError` (match them with a `_` arm); `RenderedPage`, `PageDiagnostics`, `ImageFailure`, `ContentFailure`, `PageGeometry`, `GcvImage` (outputs: read their fields, do not construct them); `RasterOptions`, `SessionConfig`, `RenderParams`, `GcvBudget` (inputs: start from `RasterOptions::new(dpi)`, `SessionConfig::with_policy`, `RenderParams::new`, or `Default`, then assign fields). `RenderRegion` is a plain rectangle and stays exhaustive. `BackendPolicy::name()` is the inverse of its `FromStr`.

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

- `RasterError::InvalidOptions` — `opts` violates constraints (e.g. `dpi ≤ 0`, `first_page > last_page`), or, in a `vulkan` build, `PDF_RASTER_VULKAN_DEVICE` matches no single usable device. Yielded as `(1, Err(...))`, iterator ends immediately.
- `RasterError::Pdf` — document cannot be opened or parsed. Yielded as `(1, Err(...))`, iterator ends immediately.
- `RasterError::BackendUnavailable` — `PDF_RASTER_BACKEND` names a backend (`cuda`, `vaapi`, `vulkan`) that cannot be initialised. Yielded as `(1, Err(...))`, iterator ends immediately.
- `RasterError::PageOutOfRange` — page number exceeds document length.
- `RasterError::PageDegenerate` — page has zero pixel dimensions.
- `RasterError::PageTooLarge` / `RasterError::PageAreaTooLarge` — pixel dimensions exceed `MAX_PX_DIMENSION` (32 768) or the area exceeds `MAX_PX_AREA`.
- `RasterError::InvalidPageGeometry` — `UserUnit` outside `[0.1, 10.0]`.
- `RasterError::ImageDecodeFailed` / `RasterError::ContentDecodeFailed` — only when `opts.strict` is set; see [`RasterOptions`](#rasteroptions).
- `RasterError::PageBudgetExceeded` — the page exceeded the per-page operator budget or wall-clock deadline.
- `RasterError::Deskew` — deskew rotation failed (rare; graceful fallback applied when possible).

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

### `rasterrocket::session` module

Lower-level API for explicit control over PDF opening and per-page rendering. All items are also re-exported at the crate root for backward compatibility.

```rust
pub mod session {
    pub use super::{
        open_session, open_session_from_bytes, prescan_session, render_page_gray,
        render_page_rgb, render_page_rgb_with, rgb_to_gray,
    };
}
```

```rust
pub fn open_session(path: &Path, config: &SessionConfig) -> Result<RasterSession, RasterError>
pub fn open_session_from_bytes(bytes: Vec<u8>, config: &SessionConfig) -> Result<RasterSession, RasterError>
```

`open_session_from_bytes` is the in-memory counterpart to `open_session` (the session-level analogue of `raster_pdf_from_bytes`): same behaviour, taking owned PDF bytes instead of a path.

An encrypted document (Standard security handler, revisions 2–6: RC4 40–128-bit, AES-128, AES-256, crypt filters) is refused with `RasterError::Pdf` unless `config.decrypt_authorized` is `true`; with it, the document is decrypted in process — no subprocess, no temporary file. The empty user password is tried first (a permissions-only document needs none), then `config.password` as the user and as the owner password; a wrong password, or none where one is needed, is a `RasterError::Pdf` saying which. A document using another security handler (public-key encryption, `/Adobe.PubSec`) is refused by name.

Opens the PDF and reads the page count from the page-tree root (`/Pages /Count`); no page object is read until that page is rendered. Also initialises the shared CUDA context (for the `gpu-aa`, `gpu-icc` or `cache` features; not when Vulkan is chosen under `Auto`) according to `config.policy`. Errors with `RasterError::Pdf` if the file is unreadable or corrupt. A JavaScript-bearing PDF is **not** an error — it opens and renders normally; a loud `WARN` is emitted per detected catalog JS entry point (see `open` below), the first rendered page carrying page or annotation JavaScript emits one more, and no `/JS` is ever decoded or executed. Errors with `RasterError::BackendUnavailable` if `config.policy` is `ForceCuda`, `ForceVaapi` or `ForceVulkan` and the required backend cannot be initialised (for `ForceVulkan`, also when the crate was built without the `vulkan` feature; for `ForceCuda`, when it was built without any of `gpu-aa`, `gpu-icc` or `cache`).

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

Renders one page to the grayscale `RenderedPage` that `raster_pdf` yields for it: `opts.dpi` and `opts.deskew` apply; the page-selection fields are ignored, and the fields mirrored on `SessionConfig` were fixed when the session was opened.

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
}
impl RenderParams {
    pub const fn new(scale: f64) -> Self                     // same scale on both axes
    pub const fn new_xy(scale_x: f64, scale_y: f64) -> Self  // non-square pixels
    pub const fn with_region(self, region: RenderRegion) -> Self
    pub const fn with_policy(self, policy: BackendPolicy) -> Self
    pub fn with_cancel(self, cancel: CancelToken) -> Self
}

pub struct RenderRegion { pub x: u32, pub y: u32, pub width: u32, pub height: u32 }

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
- **`region`** draws only a window of the page, in device pixels counted from the page's top-left corner at those scales. The bitmap is the region's size, and it holds exactly the pixels the whole-page render places there, so regions that tile a page stitch into the whole-page bitmap byte for byte. `MAX_PX_DIMENSION` and `MAX_PX_AREA` apply to the region, so a page too large to render whole (a map or poster at high DPI) can be rendered in tiles. An empty region, or one extending past the page, is `RasterError::InvalidOptions`.
- **`policy`** overrides the session's backend policy for this page's image decoders; pass `BackendPolicy::CpuOnly` for a page `prescan_session` reports without images to skip GPU decoder setup.
- **`cancel`**: after `CancelToken::cancel()` on any clone, the render stops before its next content-stream operator (form, Type 3 glyph and pattern-cell content included) and returns `RasterError::Cancelled`; a render started after cancellation returns it without drawing. An operator already running (one image decode, one fill) completes first.

```rust
pub fn rgb_to_gray(src: &Bitmap<Rgb8>) -> Bitmap<Gray8>
```

`DeviceRGB` → `DeviceGray` conversion with the weights of ISO 32000-1 §10.3.3, `Y = 0.30·R + 0.59·G + 0.11·B`, in 15-bit fixed point with round-half-up (`(9830·R + 19333·G + 3605·B + 16384) >> 15`), so `R = G = B` maps to itself. `raster_pdf`, `render_channel` and the CLI's `--gray` / `--mono` all use it.

```rust
pub fn prescan_session(
    session: &RasterSession,
    page_num: u32,
) -> Result<PageDiagnostics, RasterError>
```

Classifies page `page_num` without rendering any pixels. Returns `PageDiagnostics` with image/text presence, dominant filter, and a PPI hint. Use this before `render_page_rgb` to choose the backend policy (e.g. skip GPU init for vector-only pages).

Errors with `RasterError::PageOutOfRange` or `RasterError::InvalidPageGeometry` if the page is invalid.

---

### `RasterSession`

```rust
pub struct RasterSession { /* opaque */ }

impl RasterSession {
    pub const fn total_pages(&self) -> u32
    pub const fn policy(&self) -> BackendPolicy
    pub const fn form_type(&self) -> FormType
    pub fn page_geometry(&self, page_num: u32) -> Result<PageGeometry, RasterError>
    #[cfg(feature = "cache")]
    pub const fn image_cache(&self) -> Option<&Arc<DeviceImageCache>>  // diagnostics
    // doc() and resolve_page() are #[doc(hidden)] — use prescan_session instead
}
```

An opened, read-only document. `Sync + Send` — safe to share across Rayon threads.

`page_geometry` reads a page's size, rotation, `UserUnit` and boxes without parsing its content; see [`PageGeometry`](#pagegeometry). `geometry.size_px(scale_x, scale_y)` is the bitmap size a render at those scales produces, so an embedder can size buffers or plan tiles before rendering.

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
    pub decrypt_authorized: bool,
    pub password: Option<Password>,
    pub font_dirs: Vec<PathBuf>,
    pub system_fonts: bool,
}
```

| Field | Constraints | Notes |
|---|---|---|
| `dpi` | `> 0`, finite | Render resolution. Pass `effective_dpi` (not `dpi`) to Tesseract. |
| `first_page` | `≥ 1` | 1-based inclusive. Ignored when `pages` is `Some`. |
| `last_page` | `≥ first_page` | 1-based inclusive. Clamped to document length silently. Ignored when `pages` is `Some`. |
| `deskew` | — | Applies intensity-weighted projection-profile deskew (±7°, sub-0.05° accuracy). Disable for native-text PDFs. |
| `pages` | — | When `Some`, only the pages in the `PageSet` are rendered; `first_page`/`last_page` are ignored. |
| `strict` | — | When `true`, a page with an image that cannot be decoded or drawn, or is drawn only in part, returns `RasterError::ImageDecodeFailed`, and a page with a content stream that cannot be decoded, or decodes only in part, returns `RasterError::ContentDecodeFailed`. When `false`, the image or stream is skipped (or drawn from the data that decoded) and listed in `PageDiagnostics::image_failures` or `PageDiagnostics::content_failures`. |
| `purpose` | — | `RenderPurpose::View` (on-screen viewing) or `RenderPurpose::Print`. See [`RenderPurpose`](#renderpurpose). |
| `annotations` | — | When `false`, annotations (form-field widgets included) are not drawn and pages show their content alone. |
| `text_antialias` | — | When `false`, glyphs are rasterised bilevel (no partial coverage). Text drawn as paths follows `vector_antialias`. |
| `vector_antialias` | — | When `false`, path fills, strokes and clips are scan-converted at pixel centres with hard edges, always on the CPU. |
| `decrypt_authorized` | — | Whether an encrypted document may be decrypted; see [`SessionConfig`](#sessionconfig). `false` makes an encrypted document an error. |
| `password` | — | Password for an encrypted document, tried as user then owner password after the empty user password. No effect unless `decrypt_authorized` is `true`. |
| `font_dirs` | — | Extra directories of installed fonts, searched before those `RROCKET_FONT_DIR` names and the standard ones; see [`SessionConfig`](#sessionconfig). |
| `system_fonts` | — | When `false`, the standard font directories are not searched; see [`SessionConfig`](#sessionconfig). |

`RasterOptions` implements `Default` and is `#[non_exhaustive]`: build it with `RasterOptions::new(dpi)` (every other field at its default) or `RasterOptions::default()`, then assign fields; struct literals and `..Default::default()` updates do not compile outside the crate.

```rust
let mut opts = RasterOptions::new(150.0);
opts.deskew = true;
```

Defaults: `dpi = 300.0`, `first_page = 1`, `last_page = u32::MAX`, `deskew = false`, `pages = None`, `strict = false`, `purpose = RenderPurpose::View`, `annotations = true`, `text_antialias = true`, `vector_antialias = true`, `decrypt_authorized = false`, `password = None`, `font_dirs` empty, `system_fonts = true`.

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
}
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
    pub pixels: Vec<u8>,
    pub dpi: f32,
    pub effective_dpi: f32,
    pub diagnostics: PageDiagnostics,
}
```

| Field | Notes |
|---|---|
| `page_num` | 1-based page number. |
| `width`, `height` | Output bitmap dimensions in pixels. |
| `pixels` | 8-bit grayscale, tightly packed (`stride == width`), top-to-bottom. Length is exactly `width * height`. |
| `dpi` | Raw render DPI (`opts.dpi`). |
| `effective_dpi` | `opts.dpi × UserUnit`. Pass this to `tesseract::set_source_resolution`. For the vast majority of documents `UserUnit = 1.0` and this equals `dpi`. |
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
| `source_ppi_hint` | `Option<f32>` | Estimated native pixels-per-inch of the dominant image. Computed as `(image_width_px / page_width_pts) × 72`. `None` when no images were blitted. |
| `image_failures` | `Vec<ImageFailure>` | Images that were present but could not be decoded or drawn (skipped, so the region shows whatever was painted beneath), and images drawn only partly because their stream lacked samples (`partial`). Empty on a complete render. Includes images inside tiling-pattern tiles. An image whose decoded samples would exceed 1 GiB (width × height × components) is refused before anything is allocated and listed here. |
| `content_failures` | `Vec<ContentFailure>` | Elements of the page's `/Contents` that could not be decoded (or are not streams), each skipped while the page is drawn from the others, and elements that decoded only in part (`partial`), executed up to where their data was lost. Either way whatever the lost data held is missing. Empty on a complete render. A page none of whose content streams decode fails with `RasterError::Pdf` instead. |
| `missing_fonts` | `Vec<String>` | Font resource names (without the leading `/`) that `Tf` selected but the `/Font` resources in effect do not define, once each. Their text is drawn in Helvetica, so it is present but in the wrong face. Not a `strict` failure. |
| `form_type` | `FormType` | The document's interactive form type (the same on every page). `XfaFull` means a dynamic XFA form: XFA is not processed, so the page is the placeholder content stored in the file, often a notice asking for an XFA-capable viewer, and its text is not the form's. |

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
    Auto,        // Vulkan if compiled in and present, else CUDA, else CPU; silent fallback (default)
    CpuOnly,     // Skip all GPU init entirely
    ForceCuda,   // Require CUDA; error if unavailable
    #[cfg(feature = "vaapi")]
    ForceVaapi,  // Require VA-API JPEG; error if unavailable — only when `vaapi` feature is enabled
    ForceVulkan, // Require the Vulkan compute backend; error if unavailable
                 // (or if the binary was built without `--features vulkan`)
}
```

Controls which compute backend is used. The `Force*` variants convert silent GPU fallbacks into hard `RasterError::BackendUnavailable` errors so you know immediately whether the expected hardware path is actually active. `ForceVaapi` is only present when the `vaapi` Cargo feature is enabled.

`ForceVulkan` runs the AA-fill kernel (and, in a build with `gpu-jpeg-huffman`, the parallel-Huffman JPEG decode kernels) on the Vulkan compute backend (cross-vendor: NVIDIA, AMD, Intel, Apple via `MoltenVK`).  The device-resident image cache is CUDA-only, so under `ForceVulkan` the renderer runs uncached; ICC CMYK→RGB stays on the CPU.  Both GPU paths are dormant by default (see [GPU dispatch thresholds](#gpu-dispatch-thresholds)): `PDF_RASTER_AA_FILL_THRESHOLD=0` enables the AA fill and `PDF_RASTER_HUFFMAN_THRESHOLD=0` the Huffman decode.

`BackendPolicy` parses from `auto`, `cpu`, `cuda`, `vaapi` (in a build with the `vaapi` feature) and `vulkan` (`FromStr`; case-insensitive, whitespace trimmed, empty is `Auto`), and `name()` returns that spelling. `BackendPolicy::from_env()` reads `PDF_RASTER_BACKEND`, and `from_env_var(name)` any other variable; an unrecognised value logs a warning and resolves to `Auto`.

---

### `SessionConfig`

```rust
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub policy: BackendPolicy,
    pub vaapi_device: String,  // default: DEFAULT_VAAPI_DEVICE ("auto", probe render nodes)
    pub vulkan_device: Option<String>,  // default: None (automatic)
    // `prefetch: bool` exists only when `cache` feature is enabled
    pub decrypt_authorized: bool,  // default: false
    pub password: Option<Password>, // default: None
    pub strict: bool,              // default: false
    pub purpose: RenderPurpose,    // default: RenderPurpose::View
    pub annotations: bool,         // default: true
    pub text_antialias: bool,      // default: true
    pub vector_antialias: bool,    // default: true
    pub font_dirs: Vec<PathBuf>,   // default: RROCKET_FONT_DIR (Default), empty (with_policy)
    pub system_fonts: bool,        // default: true
}

impl Default for SessionConfig { /* policy from PDF_RASTER_BACKEND, Vulkan device from PDF_RASTER_VULKAN_DEVICE, font directories from RROCKET_FONT_DIR */ }

impl SessionConfig {
    pub fn with_policy(policy: BackendPolicy) -> Self;      // reads no env vars
    pub fn vulkan_device_from_env() -> Option<String>;     // PDF_RASTER_VULKAN_DEVICE
    pub fn font_dirs_from_env() -> Vec<PathBuf>;           // RROCKET_FONT_DIR
}
```

Passed to `open_session`. `SessionConfig` is `#[non_exhaustive]` — construct it via `SessionConfig::default()` or `SessionConfig::with_policy(policy)`, not a struct literal. This is required because the `prefetch` field only exists when the `cache` Cargo feature is enabled. `vaapi_device` defaults to `"auto"` (`DEFAULT_VAAPI_DEVICE`): the first `/dev/dri/renderD*` node, in ascending number, whose driver supports VA-API baseline JPEG decode. Set it to a render-node path to use that node instead. Under `ForceVaapi` a failure is `RasterError::BackendUnavailable` naming the render nodes present; under `Auto` JPEG falls back to the CPU. `CpuOnly`, `ForceCuda` and `ForceVulkan` never open a render node. Set `vulkan_device` to choose the Vulkan device: an index in Vulkan enumeration order (from 0) or a case-insensitive device-name substring; a number that is not a valid index is matched as a name. A value that matches no device, more than one, or a device without Vulkan 1.3 and a compute queue makes `open_session` return `RasterError::InvalidOptions` listing the devices, under `Auto` as well as `ForceVulkan`. `None` picks automatically. `strict` has the same meaning as `RasterOptions::strict`, for renders through a session (`render_page_rgb` and friends); `raster_pdf`, `raster_pdf_from_bytes` and `render_channel` take it from `RasterOptions`. `purpose` works the same way and mirrors `RasterOptions::purpose`; see [`RenderPurpose`](#renderpurpose). `annotations = false` renders page content without annotations and mirrors `RasterOptions::annotations`. `text_antialias` and `vector_antialias` (both default `true`) switch antialiasing of glyph bitmaps and of path fills, strokes and clips, and mirror the `RasterOptions` fields of the same names. `decrypt_authorized` permits decrypting an encrypted document (default `false`: such a document is an error, whatever `password` holds); `password` is tried as the user and then the owner password after the empty user password. Both mirror the `RasterOptions` fields of the same names. `Password` (re-exported from the `pdf` crate) converts from `&str` and `String` (`Some("secret".into())`) and prints as `Password(<redacted>)` under `Debug`, so a logged `SessionConfig` does not disclose it. `font_dirs` lists extra directories of installed fonts, searched in order before the standard font directories: a font the document does not embed draws with an installed substitute (the metric-compatible Liberation family, or a face of the font's script) and, for a character the substitute lacks, with an installed face that covers it, and both are looked up there first. `SessionConfig::default()` reads them from `RROCKET_FONT_DIR` (`PATH` syntax), `with_policy` sets none; sessions naming the same directories share one index of the installed fonts. `RasterOptions::font_dirs` is searched before the `RROCKET_FONT_DIR` directories by `raster_pdf`, `raster_pdf_from_bytes` and `render_channel`. `system_fonts = false` (mirrored by `RasterOptions::system_fonts`) stops the search of the standard font directories and of the fixed locations of the Liberation, Arial, Times New Roman and Courier New faces: non-embedded fonts then draw from `font_dirs` and the faces compiled into the library only, the base-14 substitutes and, with the default `bundled-fonts` feature, the bundled Noto fallback faces. Lookup order for a non-embedded font: an installed substitute of its class and script, then for a character it lacks an installed face that covers it, then a bundled face, then nothing.

---

### `deskew` module

The `deskew` module exposes a limited public API:

- `deskew::apply(&mut Bitmap<Gray8>) -> Result<(), DeskewError>` — detects the skew of a grayscale bitmap and rotates it in place to correct it; angles too small to matter are left alone, and a failed detection counts as no skew.
- `deskew::DeskewError` — error type returned by `apply` (a failed rotation).

The `deskew::detect` and `deskew::rotate` submodules are `pub(crate)` and are not part of the public API.

---

### `release_gpu_decoders`

```rust
pub fn release_gpu_decoders()
```

Eagerly drops GPU decoder state (nvJPEG, nvJPEG2000, Vulkan Huffman) on the calling rayon worker thread. Call via `pool.broadcast` before dropping the pool:

```rust
let _ = pool.broadcast(|_| rasterrocket::release_gpu_decoders());
drop(pool);
```

No-op when none of the GPU decoder features (`nvjpeg`, `nvjpeg2k`, `gpu-jpeg-huffman + vulkan`) are compiled in. Never panics.

---

### `RasterError`

```rust
#[non_exhaustive]
pub enum RasterError {
    InvalidOptions(String),
    Pdf(rasterrocket_interp::InterpError),
    PageOutOfRange { page: u32, total: u32 },
    PageDegenerate { width: u32, height: u32 },
    PageTooLarge { width: u32, height: u32 },
    PageAreaTooLarge { width: u32, height: u32, area: u64 },
    Deskew(String),
    InvalidPageGeometry(String),
    BackendUnavailable(String),  // forced backend failed to init
    ImageDecodeFailed(Vec<String>),
    ContentDecodeFailed(Vec<String>),
    PageBudgetExceeded(String),
    Cancelled,                   // stopped through a CancelToken
    RenderPanic { page: u32, message: String },
}
```

`PageAreaTooLarge` is distinct from `PageTooLarge`: each side can be within
`MAX_PX_DIMENSION` while the product still exceeds `MAX_PX_AREA`. `RenderPanic`
carries a per-page panic caught by `render_channel`, which happens only under
`panic = "unwind"`; with the `panic = "abort"` release profile a panic aborts
the process.

`ImageDecodeFailed` and `ContentDecodeFailed` are only returned under `strict`; they carry one message per failed image or content stream.

Implements `std::error::Error` with a `source()` chain. `RasterError::Pdf(e)` has `e` as its source for chained error reporting. `BackendUnavailable` is only returned when the policy is `ForceCuda`, `ForceVaapi` or `ForceVulkan`.

---

### `MAX_PX_DIMENSION` / `MAX_PX_AREA`

```rust
pub const MAX_PX_DIMENSION: u32 = 32_768;
pub const MAX_PX_AREA: u64 = 600_000_000;
```

`MAX_PX_DIMENSION` is the maximum accepted pixel dimension (width or height);
`PageTooLarge` is returned if either exceeds it. At 150 DPI this corresponds to
~366 inches (~9.3 metres).

`MAX_PX_AREA` caps total raster area and is computed in `u64` (closing a latent
overflow). A page whose sides are each legal but whose product exceeds this
returns `PageAreaTooLarge`.

Both are checked before the bitmap is allocated, and they bound what one call
draws: with a `RenderParams::region` they apply to the region, so a page larger
than either can be rendered as tiles.

Images inside a page have their own budget, in `rasterrocket-interp`
(`resources::image::MAX_IMAGE_SIDE` = 131 071 px per declared side,
`MAX_DECODED_BYTES` = 1 GiB of decoded samples, width × height × components):
an image over it is refused before decompression and reported in
`PageDiagnostics::image_failures`, or fails the page under `strict`.

---

### Re-exports

```rust
pub use rasterrocket_interp::renderer::{ContentFailure, ImageFailure, PageDiagnostics};
pub use rasterrocket_interp::resources::ImageFilter;
pub use rasterrocket_interp::{FormType, PageGeometry, RenderPurpose};
pub use rasterrocket_parser::Password;
```

`ImageFilter` identifies which decode filter was used for an embedded image (`Dct`, `Jpx`, `CcittFax`, `Jbig2`, `Flate`, `Raw`). Available through `PageDiagnostics` for routing decisions.

The crate root also exports `is_encrypted(path) -> bool` (a structural probe of the trailer for `/Encrypt`; `false` when the file cannot be read) and `decrypt_gate_declined_message() -> String` (the error text for an encrypted document whose decryption was not authorised), which the CLI uses for its liability prompt, and `gray8_to_rendered_page` / `validate_dimensions`, the helpers every `RenderedPage` producer shares.

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
}
```

Encodes a `RenderedPage` to a grayscale JPEG guaranteed to fit Google Cloud Vision's *binding* request limit — 10 MB of base64 inside the `images:annotate` JSON request, **not** the 20 MB raw-file limit — decided deterministically with no network call.

Strategy: try `start_quality` at native resolution; binary-search quality down toward `min_quality`; only if the quality floor still overflows, box-downscale (aspect-preserving, never below GCV's ~1024 px OCR short-side floor, never over the 75 MP server-side-resize cap) and retry. If no candidate fits without breaking those floors, returns `GcvError::Unfittable` — never an over-budget or oversized payload. `GcvImage::to_base64()` reproduces exactly the length measured during fitting (the proxy is byte-exact against the real encoding).

`GcvBudget::default()` encodes the GCV limits directly. Leave `RasterOptions::deskew` off (the default) for this path — GCV deskews internally. The raw `jpeg` field is the universal artifact (disk, GCS `files:asyncBatchAnnotate`); `to_base64()` is the inline-`content` form. No HTTP/auth/batching is performed — rasterrocket renders pixels, it is not a GCV API client. See the [LLM Vision OCR Integration](../../../wiki/LLM-Vision-OCR-Integration) wiki for end-to-end examples.

The `rasterrocket-encode` crate also exposes the underlying codec directly:

```rust
pub fn jpeg_gray<P: Pixel>(bitmap: &Bitmap<P>, quality: u8)
    -> Result<Vec<u8>, EncodeError>;
```

Baseline 8-bit grayscale JPEG (`Gray8`/`Mono8` only; `quality` clamped 1–100). A plain codec with no consumer-specific policy.

---

## `rasterrocket-comic` crate — comic-archive input

Turns a comic/scan archive (`.cbz` ZIP, `.cb7` 7-Zip, `.cbt` TAR) of page
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
}
```

Opens an archive. `ComicPages` yields `(page_num, result)` in reading order and
decodes each page only when it is reached, so memory does not grow with the
page count; a per-page decode failure is an inner `Err` and does not abort the
rest, matching `raster_pdf`. `len()` is the number of pages still to come, and
`retain` drops pages before any is decoded.

**Content routing** (decided after the entry list is built):

- **Page images present** → render the images (numeric-aware natural sort by
  filename, magic-byte format detection over JPEG/PNG/WebP/TIFF). Any PDF entry
  is ignored with a warning — images always win.
- **No images, exactly one PDF** → render that PDF's pages via
  `open_session_from_bytes` and `render_page_gray` (the PDF's own page
  numbering).
- **No images, two or more PDFs** → `ComicError::AmbiguousArchive`, naming the
  PDFs rather than guessing.
- **No images, no PDF** → `ComicError::NoImages`.

**Whole-archive errors (outer `Err`):** `InvalidOptions` (a non-finite or
non-positive `dpi`, `first_page` 0 or after `last_page`), `RarUnsupported` (a
`.cbr` input), `Open`/`BadArchive` (unreadable or malformed container),
`AmbiguousArchive`, `Pdf` (an embedded PDF that cannot be opened or has no
page in the window), `NoImages`. **Per-page errors (inner `Err`):** `Decode`, `TooLarge`, `Deskew`,
`Pdf`.

**`ComicOptions`** mirrors the relevant subset of `RasterOptions`: `dpi: f32`,
`first_page: u32`, `last_page: u32`, `deskew: bool`. The DPI rides along on each
`RenderedPage` for downstream OCR; comic images carry no intrinsic resolution.
Both `ComicOptions` and `ComicError` are `#[non_exhaustive]`: build the options
with `ComicOptions::new(dpi)` or `ComicOptions::default()` and assign the fields
to change, and give a `match` on `ComicError` a wildcard arm.

**Hardening:** entries are read under a 512 MiB per-entry decompressed-size cap
(decompression-bomb defence), and image dimensions are validated before a codec
allocates its pixel buffer (decode-bomb defence). `.cbr` (RAR) is refused — no
permissively-licensed Rust RAR decoder is linked.

---

## `rasterrocket-camera` crate — camera-capture page cleanup

Cleanup of camera-captured pages on 8-bit grayscale images: page crop,
perspective rectification, double-page spread splitting and (experimental)
dewarp. Independent of PDF: it depends on `std` alone, runs on the calling
thread with no global state, and returns on any input without panicking.
Deskew is not part of it (see the `deskew` module above).

Each stage is a detector followed by a resampler. A detector that is not
confident declines and the page passes through unchanged, with the reason in
the `Report`; a flatbed scan or born-digital page comes back as given. Only
malformed input is an error.

### Types

```rust
pub struct GrayView<'a> { pub data: &'a [u8], pub width: usize, pub height: usize, pub stride: usize }
pub struct GrayPage { pub width: usize, pub height: usize, pub data: Vec<u8> }   // packed, stride == width
pub struct Rect { pub x: usize, pub y: usize, pub width: usize, pub height: usize }
pub struct Quad { pub corners: [[f64; 2]; 4] }   // TL, TR, BR, BL, at the page edge, source pixels

#[non_exhaustive]
#[derive(Default)]                                // every stage off
pub struct CameraOptions { pub crop: bool, pub rectify: bool, pub split_spreads: bool, pub dewarp: bool }

impl GrayView<'_> {
    pub const fn packed(data: &[u8], width: usize, height: usize) -> GrayView<'_>;
    pub const fn validate(&self) -> Result<(), CameraError>;
}
impl GrayPage { pub fn view(&self) -> GrayView<'_>; }
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
    pub fn replay(&self, img: GrayView<'_>) -> Result<Vec<GrayPage>, CameraError>;
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
size — e.g. each channel of the colour image whose luma was processed — and
reproduces `pages` when given the processed image itself.

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

### `CameraError`

Implements `std::error::Error` and `Display`; each message names the bad
value and the requirement. Variants: `ZeroDimension`, `TooLarge` (a side above
`MAX_DIMENSION` = 2³⁰), `StrideTooSmall`, `BufferTooShort` (fewer than
`stride · (height − 1) + width` bytes), `Overflow`, `RectOutOfBounds` (`crop`),
`SplitOutOfRange` (`split_at`), `SizeMismatch` (`Report::replay`).

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
```

Opens and validates a PDF, from a path or from bytes. `open` and `open_bytes` refuse an encrypted document; the `_with` forms take `pdf::OpenOptions { decrypt_authorized: bool, password: Option<Password> }`, with the same meaning as the `SessionConfig` fields of those names: an authorised encrypted document is decrypted in process as its objects are read (the `pdf` crate's Standard security handler), and `Document::is_encrypted()` then reports `true`. `pdf::is_encrypted(path) -> Result<bool, PdfError>` probes the trailer for `/Encrypt` without opening the document.

 If a catalog JavaScript entry point is detected (checked locations: `/OpenAction`, catalog `/AA`, `/Names/JavaScript`, `/AcroForm/AA`, each flagged only when genuinely a `/S /JavaScript` action), a loud `WARN` is emitted per location and the document still opens — rasterrocket has no JavaScript engine and never executes `/JS`, so a script's structural presence cannot change the static rendered page. No JS is parsed or evaluated. No page object is read at open; page-level `/AA` and annotation `/A`/`/AA` are checked per page by `page_has_javascript(doc, page_id) -> bool`, which the render path calls before each page and warns on once per session.

The returned `pdf::Document` comes from the in-tree `pdf` crate (replaced lopdf 0.40 in v0.6.0). It is a lazy, mmap-backed reader: opening the file parses only the xref table and trailer; individual objects resolve on demand via byte-offset seek with a per-object `Arc` cache shared safely across worker threads. Object stream (`ObjStm`) decompression is cached once and reused. The API mirrors the lopdf method names previously used here.

### `page_count`

```rust
pub fn page_count(doc: &Document) -> u32
```

Total pages. Saturates at `u32::MAX` for pathological documents (> 4 billion pages).

### `page_size_pts`

```rust
pub fn page_size_pts(doc: &Document, page_num: u32) -> Result<PageGeometry, InterpError>
```

Returns geometry for page `page_num` (1-based). `width_pts` and `height_pts` are already adjusted for rotation and `UserUnit` scaling — use them directly as output bitmap dimensions.

Falls back to US Letter (612 × 792 pt) when neither `CropBox` nor `MediaBox` can be read.

### `parse_page`

```rust
pub fn parse_page(doc: &Document, page_num: u32) -> Result<Vec<Operator>, InterpError>
```

Parses the content stream for page `page_num` and returns the decoded operator sequence. Typically called internally by the renderer; exposed for tooling (e.g. `dump_ops` example).

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
}
```

Dimensions are swapped for 90°/270° rotations so that `width_pts` always corresponds to the horizontal extent of the rendered bitmap. Re-exported as `rasterrocket::PageGeometry` and returned by `RasterSession::page_geometry`.

`size_px(scale_x, scale_y)` is the rendered pixel size, `(width_pts × scale_x).round()` by `(height_pts × scale_y).round()`, saturating at `u32::MAX`.

### `InterpError`

```rust
pub enum InterpError {
    Pdf(pdf::PdfError),
    PageOutOfRange { page: u32, total: u32 },
    MissingResource(String),
    InvalidPageGeometry(String),
    PageBudget(String),
}
```

Implements `std::error::Error`. `InterpError::Pdf(e)` chains to `pdf::PdfError`.

---

## Hardware compatibility

### CPU

**Supported:** x86-64 (AMD and Intel) and `aarch64` (ARM / Apple Silicon).

**x86-64:**
- AVX2 SIMD blend and fill paths are runtime-detected with a scalar fallback.
- AVX-512 (`avx512f/bw/vl/dq/vnni/vpopcntdq` and related sub-extensions) is activated by building with `-C target-cpu=native` on a compatible CPU. Developed and benchmarked on an AMD Ryzen 9900X3D.
- All Intel consumer CPUs (Alder/Raptor/Arrow Lake) have AVX2; AVX-512 is disabled on them — Xeon only.

**aarch64:**
- NEON is used unconditionally (mandatory on all ARMv8-A). No runtime detection needed.
- SVE2 (`svcnt_u8_z` popcount tier) is available behind the `nightly-sve2` feature of `rasterrocket-render` on nightly Rust (`rasterrocket` and `rasterrocket-cli` do not forward it). Gives up to 4× NEON throughput on wide-SVE2 server chips (Graviton4 full width).
- `cargo check --target aarch64-unknown-linux-gnu` is clean; no Apple Metal native backend yet (Vulkan via `MoltenVK` covers Apple).

### GPU

**NVIDIA (CUDA 13.0–13.3):**

| Feature flag | Minimum requirement | Notes |
|---|---|---|
| `nvjpeg` | CUDA-capable NVIDIA GPU | `libnvjpeg.so` ships with the CUDA 13 toolkit |
| `nvjpeg2k` | CUDA-capable NVIDIA GPU | `libnvjpeg2k.so` is a separate download; build script probes `/13` then `/12` |
| `gpu-aa` | CUDA-capable NVIDIA GPU | CUDA runtime only |
| `gpu-icc` | CUDA-capable NVIDIA GPU | CUDA runtime only |
| `gpu-deskew` | CUDA-capable NVIDIA GPU | Requires CUDA NPP: `libnppig.so` + `libnppc.so` |
| `cache` | CUDA-capable NVIDIA GPU | Device-resident 3-tier image cache (CUDA-only) |

`cudarc` is pinned to the `cuda-13030` driver-API binding; a CUDA 13.x driver is required.

**Vulkan compute (cross-vendor — NVIDIA, AMD, Intel, Apple via `MoltenVK`):**

| Feature flag | Supported hardware | Libraries required |
|---|---|---|
| `vulkan` | Any Vulkan 1.3+ device | Vulkan ICD (e.g. `mesa-vulkan-drivers`, NVIDIA driver); `slangc` from the LunarG Vulkan SDK at *build* time. Implies `gpu-aa`. |

Vulkan covers the AA-fill kernel, and the parallel-Huffman JPEG decode kernels when `gpu-jpeg-huffman` is also enabled (`vulkan` does not imply it).  `cache` and `nvjpeg` stay CUDA-only; under `--backend vulkan` the renderer runs uncached.

**VA-API (Linux iGPU/dGPU — AMD VCN, Intel Quick Sync, Intel Arc):**

| Feature flag | Supported hardware | Libraries required |
|---|---|---|
| `vaapi` | AMD VCN (Raphael+), Intel UHD 630+, Intel Arc | `libva.so.2`, `libva-drm.so.2` |

VA-API provides hardware JPEG baseline decode; CMYK and progressive JPEGs fall through to the CPU. When both `nvjpeg` and `vaapi` are active, nvJPEG is tried first. Both share the `GPU_JPEG_THRESHOLD_PX` gate, which is `u32::MAX`, so in a default build neither decodes anything and every JPEG is decoded on the CPU (see [GPU dispatch thresholds](#gpu-dispatch-thresholds)).

Under `BackendPolicy::Auto`, GPU initialisation failures at runtime log a warning and fall back to the CPU — no error is returned, rendering continues normally. The `Force*` policies turn them into `RasterError::BackendUnavailable`.

### Platform support

| Platform | CPU SIMD | GPU acceleration | Status |
|---|---|---|---|
| x86-64 AMD (Ryzen) | AVX-512 + AVX2 | NVIDIA CUDA + AMD VA-API + Vulkan | **Supported** |
| x86-64 Intel (consumer) | AVX2 | NVIDIA CUDA + Intel VA-API + Vulkan | **Supported** |
| x86-64 Intel (Xeon) | AVX-512 + AVX2 | NVIDIA CUDA + Intel VA-API + Vulkan | **Supported** |
| aarch64 Linux (Graviton, RPi) | NEON + SVE2 † | Vulkan (Mesa) | CPU full, Vulkan untested on aarch64 |
| Apple Silicon (M1–M4) | NEON | Vulkan via `MoltenVK` (untested) | CPU full, Vulkan untested |
| AMD/Radeon ROCm | — | — | Not implemented (Vulkan covers Radeon) |
| Apple Metal (native) | — | — | Not implemented (Vulkan via `MoltenVK` is the path) |

† SVE2 requires the `rasterrocket-render` feature `nightly-sve2` and nightly Rust.

## Feature flags

### `rasterrocket` features

| Feature | Requires | Effect |
|---|---|---|
| `nvjpeg` | CUDA 13, `libnvjpeg.so` | GPU JPEG decode (DCTDecode). Wired but disabled: `GPU_JPEG_THRESHOLD_PX` is `u32::MAX`, so JPEGs decode on the CPU (zune-jpeg). |
| `nvjpeg2k` | CUDA 13, `libnvjpeg2k.so` | GPU JPEG 2000 decode (JPXDecode). Falls back to the CPU decoder (pure-Rust `hayro-jpeg2000`) below 262 144 px in area (`GPU_JPEG2K_THRESHOLD_PX`) or for sub-sampled chroma. |
| `gpu-aa` | CUDA 13 | GPU evaluation of the 4× scanline-AA coverage for solid fills; output is byte-identical to the CPU fill. Dormant by default: the CPU fill is faster at every size measured, so dispatch is off unless `PDF_RASTER_AA_FILL_THRESHOLD` sets a pixel-area threshold. Path clips and pattern fills always stay on the CPU. |
| `gpu-icc` | CUDA 13 | GPU ICC CMYK→RGB via 4D CLUT. Falls back to CPU AVX-512 matrix formula below 500 000 px. |
| `gpu-deskew` | CUDA 13, CUDA NPP | GPU bilinear rotation (nppiRotate). Falls back to CPU bilinear when disabled. |
| `cache` | CUDA 13 | Device-resident image cache (3-tier VRAM/host/disk). Cross-document content-hash dedup. CUDA-only; no Vulkan support today. Disk-tier persistence is opt-in via `PDF_RASTER_CACHE_DIR`. |
| `vaapi` | `libva.so.2`, `libva-drm.so.2` | VA-API JPEG baseline decode on Linux iGPU/dGPU; adds `BackendPolicy::ForceVaapi` and `--backend vaapi`. Shares the disabled `GPU_JPEG_THRESHOLD_PX` gate with `nvjpeg`, so JPEGs still decode on the CPU. |
| `vulkan` | Vulkan 1.3+ ICD; LunarG `slangc` at build time. Implies `gpu-aa`. | Vulkan compute backend. The AA-fill kernel (and, with `gpu-jpeg-huffman`, the parallel-Huffman JPEG decode kernels) run on any Vulkan 1.3+ device (NVIDIA, AMD, Intel, Apple via `MoltenVK`). No nvJPEG / `cache` support under this backend. |
| `gpu-jpeg-huffman` | CUDA 13 or `vulkan` | Custom parallel-Huffman GPU JPEG decoder (baseline 4:4:4, one or three components). Dormant: `GPU_JPEG_HUFFMAN_THRESHOLD_PX` is `u32::MAX`; `PDF_RASTER_HUFFMAN_THRESHOLD=<area>` overrides it at runtime. |
| `gpu-validation` | CUDA device at test time | Enables GPU vs CPU parity tests (`cargo test -p rasterrocket-gpu --features gpu-validation`). |
| `bundled-fonts` | — (**on by default**) | Noto fallback faces for Greek, Cyrillic, Hebrew, Arabic, Devanagari, Thai and symbols compiled into the library (~0.33 MB, SIL OFL 1.1; provenance in `crates/pdf_interp/fonts/README.md`). A character of a non-embedded font that neither its substitute nor any installed face covers is drawn from them; an installed face is always preferred. Disable with `default-features = false` for a slim build. |

Under `Auto`, GPU initialisation failures log a warning and fall back to CPU — they do not return errors.  `cudarc` is pinned to the `cuda-13030` driver-API binding, so a CUDA 13.x driver is required.

### GPU dispatch thresholds

| Path | Threshold | Constant |
|---|---|---|
| nvJPEG (DCTDecode) | **disabled** (`u32::MAX`) | `GPU_JPEG_THRESHOLD_PX` |
| GPU parallel Huffman (DCTDecode) | **dormant** (`u32::MAX`) | `GPU_JPEG_HUFFMAN_THRESHOLD_PX` |
| nvJPEG2000 (JPXDecode) | ≥ 262 144 px (area) | `GPU_JPEG2K_THRESHOLD_PX` |
| GPU AA fill | **dormant** (`usize::MAX`) | `GPU_AA_FILL_THRESHOLD` |
| GPU ICC CLUT | ≥ 500 000 px (area) | `GPU_ICC_CLUT_THRESHOLD` |

Both JPEG thresholds are `u32::MAX` **by design**, so JPEG decode runs on the
CPU by default: nvJPEG's `GPU_HYBRID` backend lost to 24-thread CPU decode on
consumer hardware, and the custom parallel-Huffman path is wired end-to-end but
off pending threshold tuning. Set `PDF_RASTER_HUFFMAN_THRESHOLD=0` to enable the
Huffman path for benchmarking. nvJPEG2000 is genuinely active above its area
threshold.

Fill dispatch order: GPU AA fill → CPU scanline AA.  The GPU path evaluates
the CPU rasteriser's own 4× scanline intersection table and composites through
the CPU pipeline, so a GPU render is byte-identical to the CPU render; it is
taken only for solid-colour fills under a rectangular clip.  `GPU_AA_FILL_THRESHOLD`
is `usize::MAX` for the same reason as the JPEG thresholds: the kernel replaces
only the span walk and sub-sample count, so the upload and readback cost more
than the CPU work they save, and CPU renders are faster at every fill size
measured.  Set `PDF_RASTER_AA_FILL_THRESHOLD=0` to force eligible fills onto the
device, e.g. for the parity tests.
