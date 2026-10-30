# Getting Started

rasterrocket converts PDF pages to pixel buffers in pure Rust. No subprocesses, no Leptonica, no Poppler in the render path.

## What it does

Given a PDF file, rasterrocket renders each page to an 8-bit grayscale buffer in memory. The primary use case is feeding pages directly into Tesseract OCR without writing any intermediate files.

## Hardware requirements

### CPU

**Supported:** x86-64 (AMD and Intel) and `aarch64` (ARM / Apple Silicon).

**x86-64:**
- AVX2 SIMD paths are runtime-detected with a scalar fallback.
- AVX-512 (`avx512f/bw/vl/dq/vnni/vpopcntdq` and extensions) is active when built with `-C target-cpu=native` on a compatible CPU. Developed and benchmarked on an AMD Ryzen 9900X3D.
- All Intel consumer CPUs (Alder/Raptor/Arrow Lake) have AVX2; AVX-512 is disabled on them — Xeon only.

**aarch64:**
- NEON is used unconditionally (mandatory on all ARMv8-A).
- SVE2 (`svcnt_u8_z` popcount tier) is available behind the `nightly-sve2` Cargo feature on nightly Rust. Gives up to 4× NEON throughput on wide-SVE2 server chips (Graviton4 full width).
- `cargo check --target aarch64-unknown-linux-gnu` is clean; no Apple Metal GPU backend yet.

### GPU (optional)

Three GPU backends are available on Linux:

**NVIDIA (CUDA 13.0–13.3):**

| Feature | Minimum GPU | Library required |
|---|---|---|
| `nvjpeg` | Any CUDA-capable NVIDIA GPU | `libnvjpeg.so` (ships with the CUDA 13 toolkit) |
| `nvjpeg2k` | Any CUDA-capable NVIDIA GPU | `libnvjpeg2k.so` (separate download; build script probes `/13` then `/12`) |
| `gpu-aa` | Any CUDA-capable NVIDIA GPU | CUDA runtime |
| `gpu-icc` | Any CUDA-capable NVIDIA GPU | CUDA runtime |
| `gpu-deskew` | Any CUDA-capable NVIDIA GPU | CUDA NPP (`libnppig.so`, `libnppc.so`) |
| `cache` | Any CUDA-capable NVIDIA GPU | CUDA runtime; opt-in disk-tier persistence via `PDF_RASTER_CACHE_DIR` |

`cudarc` is pinned to the `cuda-13030` driver-API binding, so a CUDA 13.x driver is required.

**Vulkan compute (cross-vendor — NVIDIA, AMD, Intel, Apple via `MoltenVK`):**

| Feature | Supported hardware | Libraries required |
|---|---|---|
| `vulkan` | Any Vulkan 1.3+ device | Vulkan ICD (e.g. `mesa-vulkan-drivers`, `nvidia-driver`); `slangc` from the LunarG Vulkan SDK at *build* time |

Vulkan covers the AA-fill kernel, and the parallel-Huffman JPEG decode kernels when the `gpu-jpeg-huffman` feature is also enabled (`vulkan` does not imply it).  No `nvjpeg` / `cache` support today; under `--backend vulkan` the renderer runs uncached.  Both kernels are dormant by default: the AA fill (`GPU_AA_FILL_THRESHOLD = usize::MAX`) is byte-identical to the CPU fill but slower end to end, and the Huffman decode (`GPU_JPEG_HUFFMAN_THRESHOLD_PX = u32::MAX`) loses to the multi-threaded CPU decoder; `PDF_RASTER_AA_FILL_THRESHOLD=0` and `PDF_RASTER_HUFFMAN_THRESHOLD=0` enable them.  Runs on the RTX 5070 and on AMD RADV; Intel ANV and Apple (MoltenVK) are untested.

**VA-API (Linux iGPU/dGPU — AMD VCN, Intel Quick Sync, Intel Arc):**

| Feature | Supported hardware | Libraries required |
|---|---|---|
| `vaapi` | AMD VCN (Raphael+), Intel UHD 630+, Intel Arc | `libva.so.2`, `libva-drm.so.2` |

VA-API provides hardware JPEG baseline decode; CMYK and progressive JPEGs fall through to the CPU. It shares the nvJPEG size gate (`GPU_JPEG_THRESHOLD_PX`), which is `u32::MAX`, so in a default build JPEGs are decoded on the CPU even with `vaapi` enabled.

All GPU features fall back to CPU automatically if initialisation fails — a message is logged but no error is returned.  Use `--backend cuda` / `--backend vulkan` / `--backend vaapi` to convert silent fallbacks into hard errors.

## Platform support

| Platform | CPU SIMD | GPU acceleration | Status |
|---|---|---|---|
| x86-64 AMD (Ryzen) | AVX-512 + AVX2 | NVIDIA CUDA + AMD VA-API + Vulkan | **Supported** |
| x86-64 Intel (consumer) | AVX2 | NVIDIA CUDA + Intel VA-API + Vulkan | **Supported** |
| x86-64 Intel (Xeon) | AVX-512 + AVX2 | NVIDIA CUDA + Intel VA-API + Vulkan | **Supported** |
| aarch64 Linux (Graviton, RPi) | NEON + SVE2† | Vulkan (Mesa) | CPU full, Vulkan untested on aarch64 |
| Apple Silicon (M1–M4) | NEON | Vulkan via `MoltenVK` (untested) | CPU full, Vulkan untested |
| AMD/Radeon ROCm | — | — | Not implemented (Vulkan covers Radeon) |
| Apple Metal (native) | — | — | Not implemented (Vulkan via `MoltenVK` is the path) |

† SVE2 requires `nightly-sve2` Cargo feature and nightly Rust.

## Installation

Add `rasterrocket` to your `Cargo.toml` as a git dependency:

```toml
[dependencies]
rasterrocket = { git = "https://github.com/pthomasfournet/rasterrocket" }
```

> **Not distributed via crates.io.** `rasterrocket` and `rasterrocket-cli` were
> published there once at 1.0.1 and then abandoned; those versions are frozen,
> unmaintained, and missing everything added since. Use the git dependency
> above, not a version pin.

For GPU acceleration — NVIDIA (CUDA 13) + Vulkan (cross-vendor) + VA-API (Linux iGPU/dGPU):

```toml
[dependencies]
# CUDA GPU features (NVIDIA only; full feature set including the image cache):
rasterrocket = { git = "https://github.com/pthomasfournet/rasterrocket", features = ["nvjpeg", "nvjpeg2k", "gpu-aa", "gpu-icc", "gpu-deskew", "cache"] }

# Vulkan compute (cross-vendor — NVIDIA, AMD, Intel, Apple via MoltenVK):
rasterrocket = { git = "https://github.com/pthomasfournet/rasterrocket", features = ["vulkan"] }

# VA-API (AMD/Intel iGPU on Linux — libva required):
rasterrocket = { git = "https://github.com/pthomasfournet/rasterrocket", features = ["vaapi"] }
```

For reproducible builds pin the dependency to a release tag (`tag = "v1.5.0"`) or a commit (`rev = "..."`). Do not use `cargo add rasterrocket` without `--git`: it would resolve to the abandoned crates.io release.

## Quickstart

`RasterOptions` and the other option types are `#[non_exhaustive]`: start from
`RasterOptions::new(dpi)` or `RasterOptions::default()` and assign the fields
you change.

```rust
use std::path::Path;
use rasterrocket::{RasterOptions, raster_pdf};

let opts = RasterOptions::new(300.0);

for (page_num, result) in raster_pdf(Path::new("document.pdf"), &opts) {
    match result {
        Ok(page) => {
            // page.pixels: Vec<u8>, 8-bit grayscale
            // length = page.width * page.height
            // layout: top-to-bottom, left-to-right, no padding (stride == width)
            println!("page {page_num}: {}×{} px", page.width, page.height);
        }
        Err(e) => eprintln!("page {page_num}: {e}"),
    }
}
```

## Tesseract integration

The buffer maps directly onto Tesseract's raw-image input. With the
[`leptess`](https://crates.io/crates/leptess) crate (see the wiki's OCR
Integration page for the full pattern, including reusing one engine per thread):

```rust
use std::path::Path;
use rasterrocket::{RasterOptions, raster_pdf};

let mut opts = RasterOptions::new(300.0);
opts.deskew = true;

for (page_num, result) in raster_pdf(Path::new("scan.pdf"), &opts) {
    let page = match result {
        Ok(p) => p,
        Err(e) => { eprintln!("page {page_num}: {e}"); continue; }
    };
    // Hand Tesseract:
    //   data            = &page.pixels
    //   width, height   = page.width, page.height
    //   bytes per pixel = 1 (8-bit grayscale)
    //   bytes per line  = page.width (stride == width, no padding)
    //   resolution      = page.effective_dpi
    //
    // Always pass effective_dpi, not dpi: it includes the page's UserUnit,
    // and a wrong resolution degrades recognition.
}
```

**Do not binarise the pixels before passing to Tesseract.** The LSTM engine reads grayscale values directly for feature extraction; binarising discards information it would have used. For uneven scan backgrounds, configure Tesseract's Sauvola thresholding instead:

```rust
// has_images && !has_vector_text is a reliable heuristic for scanned pages
if page.diagnostics.has_images && !page.diagnostics.has_vector_text {
    // Sauvola handles uneven backgrounds better than Otsu
    tesseract_handle.set_variable("thresholding_method", "2");
}
```

## DPI guidelines

| Document type | Recommended DPI | Notes |
|---|---|---|
| Scanned documents | 300 | Standard for OCR |
| High-quality scans | 400–600 | When source scan is high-res |
| Native/vector PDFs | 150–200 | Text is resolution-independent |
| Thumbnails / preview | 72–96 | Not for OCR |

Use `page.diagnostics.suggested_dpi(min, max)` to re-render at the document's native image resolution:

```rust
// `page` is a RenderedPage rendered with `opts`.
if let Some(native_dpi) = page.diagnostics.suggested_dpi(150.0, 600.0) {
    if (native_dpi - opts.dpi).abs() > 10.0 {
        // Re-render at native resolution to avoid up/downsampling artefacts
    }
}
```

## Concurrent rendering

For multi-document pipelines, use `render_channel` to render pages in the background while OCR processes the previous one:

```rust
use std::path::Path;
use rasterrocket::{RasterOptions, render_channel};

let mut opts = RasterOptions::new(300.0);
opts.last_page = 100;
opts.deskew = true;

// capacity=4: up to 4 rendered pages buffered before the producer blocks
let rx = render_channel(Path::new("scan.pdf"), &opts, 4);

for (page_num, result) in rx {
    match result {
        Ok(page) => { /* OCR page.pixels */ }
        Err(e)   => eprintln!("page {page_num}: {e}"),
    }
}
```

## Error handling

Per-page errors do not abort the remaining pages. Handle them per-iteration:

```rust
use std::path::Path;
use rasterrocket::{RasterError, RasterOptions, RenderedPage, raster_pdf};

fn render_all(path: &Path, opts: &RasterOptions, mut process: impl FnMut(RenderedPage)) -> Result<(), RasterError> {
    for (page_num, result) in raster_pdf(path, opts) {
        match result {
            Ok(page) => process(page),
            Err(RasterError::PageTooLarge { width, height }) => {
                eprintln!("page {page_num}: {width}×{height} exceeds limit — skipping");
            }
            Err(e) => return Err(e),
        }
    }
    Ok(())
}
```

`RasterError` is `#[non_exhaustive]`, so a `match` on it always needs a catch-all arm.

Document-open errors (bad path, corrupt PDF, an encrypted document without authorisation) are yielded as `(1, Err(...))` and the iterator ends immediately after.

A page is rendered at most 32 768 pixels on a side and 600 MP in area
(`MAX_PX_DIMENSION`, `MAX_PX_AREA`); a larger one is `PageTooLarge` or
`PageAreaTooLarge`, and can be rendered in tiles through the session API below.
Images have their own decode budget. See
[Size limits](cli-reference.md#size-limits).

An image that cannot be decoded or drawn does not fail its page. The image is
skipped, the rest of the page renders, a warning names the page and image, and
the failure is listed in `page.diagnostics.image_failures` (image name and
codec error). Check that list to flag a page for review or re-OCR:

```rust
if !page.diagnostics.image_failures.is_empty() {
    eprintln!("page {}: {} image(s) missing", page.page_num, page.diagnostics.image_failures.len());
}
```

A `/Contents` stream that cannot be decoded is treated the same way: the page
is drawn from its other content streams and the skipped one is listed in
`page.diagnostics.content_failures`. A stream or image whose data ends early
(a truncated download, a corrupt filter) is drawn from what decoded and listed
with `partial: true`.

Set `RasterOptions::strict` to make such a page an `Err` instead, the
behaviour before v1.5.0.

## Page geometry, regions, non-square pixels and cancellation

The session API opens a document once and renders pages on demand, from any
number of threads. It returns RGB bitmaps; `rgb_to_gray` converts them.

```rust
use std::path::Path;
use rasterrocket::{
    CancelToken, RasterError, RenderParams, RenderRegion, SessionConfig, open_session,
    render_page_rgb_with,
};

fn top_left_tile(path: &Path) -> Result<(), RasterError> {
    let session = open_session(path, &SessionConfig::default())?;
    let geometry = session.page_geometry(1)?; // size, rotation, UserUnit, boxes; no content parsed
    let (sx, sy) = (600.0 / 72.0, 300.0 / 72.0); // 600 × 300 dpi: non-square pixels
    let (width, height) = geometry.size_px(sx, sy);

    let cancel = CancelToken::new(); // clone it to another thread and call cancel() there
    let params = RenderParams::new_xy(sx, sy)
        .with_region(RenderRegion::new(0, 0, width.min(4096), height.min(4096)))
        .with_cancel(cancel.clone());
    let tile = render_page_rgb_with(&session, 1, &params)?;
    println!("{}×{} tile of a {width}×{height} page", tile.width, tile.height);
    Ok(())
}
```

A region holds exactly the pixels the whole-page render puts there, so tiles
stitch into the whole page. A cancelled render returns `RasterError::Cancelled`
before its next content-stream operator.

## Annotations, form fields and layers

Annotations are drawn after the page content, in `/Annots` order, from their normal appearance stream (`/AP /N`). Set `RasterOptions::annotations = false` (CLI: `--hide-annotations`) to render the page content alone. The annotation flags apply for the render purpose (`RasterOptions::purpose`, `--purpose`). Hidden annotations are never drawn. NoView annotations are omitted when viewing. When printing, only annotations flagged Print are drawn. An annotation, an XObject or a marked-content sequence whose optional content group (layer) is off in the document's default configuration is not drawn.

An annotation without a normal appearance is drawn from its dictionary. Square, Circle, Line, PolyLine, Polygon, Ink, Highlight, Underline, StrikeOut, Squiggly and Text (note icon) annotations use their geometry and colours. Form field widgets of text, choice and check/radio fields show the field value, styled by `/DA`, `/MK` and the form's `/DR`. When the form sets `/NeedAppearances`, text and choice fields are always redrawn this way, because their stored appearance may not show the value. A filled form whose values live only in `/V` therefore renders with its values, which OCR then reads.

## Security

rasterrocket is a rasterizer with no JavaScript engine: it never parses, decodes, or executes `/JS`. JavaScript-bearing PDFs (common in Acrobat/InDesign interactive exports and gov/tax/bank forms) still open and render — refusing them would be a false-negative total loss on valid documents. When a JavaScript entry point is detected, a loud `WARN` is emitted on stderr (by default, no `RUST_LOG` opt-in needed) naming each location, so the operator knows the script was ignored. Detection is purely structural (dict-key presence and `/S` subtype names); no JavaScript string is ever read.

Flagged only when genuinely JavaScript (never on bare key presence — that would cry wolf):

- `/Catalog/OpenAction` whose action has `/S /JavaScript`
- `/Catalog/AA` with a `/S /JavaScript` sub-action (a transition/GoTo `/AA` is not flagged)
- `/Catalog/Names/JavaScript` (the JS-specific name subtree)
- `/Catalog/AcroForm/AA` with a `/S /JavaScript` sub-action (a `/S /SubmitForm` AcroForm is not flagged)
- page-level `/AA` and per-annotation/widget `/A`/`/AA` with a `/S /JavaScript` action (the calculate/format/validate JS of a fillable form lives here — the most common real-world placement)

The four catalog locations are checked when the document opens, with one warning per location. Opening reads no page objects, so page and annotation actions are checked as each page is rendered: the first rendered page that carries JavaScript produces one warning for the document, naming that page. A page that is never rendered is never inspected, and at most 8192 annotations are inspected per page.

### XFA forms

XFA is not processed either. A dynamic XFA form (`/AcroForm /XFA` with catalog `/NeedsRendering true`) renders the placeholder page stored in the file, usually a notice asking for an XFA-capable viewer. Opening one logs a `WARN`, and `PageDiagnostics::form_type` (and `RasterSession::form_type()`) report `FormType::XfaFull`, so OCR pipelines can tell that page from real content. Static XFA forms (`FormType::XfaForeground`) carry ordinary pages and render normally.

### Encrypted PDFs

A PDF encrypted with the Standard security handler — RC4 (40 to 128 bits),
AES-128 or AES-256, revisions 2 to 6 — is decrypted in process, behind an
opt-in gate. The library refuses by default: an encrypted document makes
`open_session` (and `open_session_from_bytes`) return a `RasterError::Pdf`
that says so. Set `SessionConfig::decrypt_authorized` (or
`RasterOptions::decrypt_authorized`) to allow it. The CLI asks for a
private-copy / liability confirmation, or takes `--decrypt-owned` (or
`RROCKET_DECRYPT_OWNED=1`) for unattended runs. Use it only for documents you
own or are entitled to access.

A document whose user password is empty (permissions-only protection) opens
with no password. One that needs a password takes it from
`SessionConfig::password` / `RasterOptions::password` (CLI: `--password`,
alias `--upw` / `--opw`); it is tried as the user password and then as the
owner password, and a wrong one is an error naming the problem. A password
is an input to decryption, not an authorisation: without the gate it does
nothing. Public-key encryption (`/Adobe.PubSec`) is not supported. Nothing
is written to disk and no subprocess runs, for encrypted or unencrypted
input.

## Building the CLI

```bash
# CPU-only (no CUDA dependency)
cargo build --release -p rasterrocket-cli

# With all GPU features
# Default CUDA_ARCH is sm_80 (Ampere); override to match your card.
CUDA_ARCH=sm_120 cargo build --release -p rasterrocket-cli \
  --features "nvjpeg,nvjpeg2k,gpu-aa,gpu-icc,gpu-deskew,cache"

# Vulkan backend (needs slangc from the LunarG SDK at build time)
cargo build --release -p rasterrocket-cli --features vulkan
```

The CLI crate forwards each of these feature names to the library.

`CUDA_ARCH` must match your GPU's Compute Capability (e.g. `sm_75` for RTX 20-series, `sm_86` for RTX 30-series, `sm_89` for RTX 40-series, `sm_120` for RTX 50-series). Look yours up at [developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus). See the [README's GPU matrix](../README.md#picking-cuda_arch-for-your-gpu) for the common-case table and the [feature-flag list](../README.md#feature-flags).

The release binary lands at `target/release/rrocket`. There is no `cargo install`
path — the crates.io packages are abandoned at 1.0.1.

### A note on output destination

The CLI defaults to writing rendered pages into a fresh `/dev/shm/rrocket-<pid>-<nanos>/` tmpfs directory whenever `OUTPUT_PREFIX` looks like a bare stem (e.g. `out`, `p`), and **removes that directory when it exits**; the directory is printed on stdout so a consumer running alongside can read the pages. To keep the pages, give a path-like prefix (anything containing `/` or starting with `.`, e.g. `./out`), which writes where you asked, or pass `--no-ram`. `--ram-path <DIR>` overrides the tmpfs location. A built-in spill policy falls through to disk when MemAvailable drops below 1 GiB.

See [cli-reference.md](cli-reference.md) for CLI usage.
