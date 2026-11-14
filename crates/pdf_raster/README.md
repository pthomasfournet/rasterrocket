# rasterrocket

The library entry point of the rasterrocket workspace: PDF pages to 8-bit
grayscale pixel buffers (or RGB bitmaps through the session API), in process
and in pure Rust, ready for Tesseract or any other OCR engine.

```rust
use std::path::Path;
use rasterrocket::{RasterOptions, raster_pdf};

let mut opts = RasterOptions::new(300.0); // every other option at its default
opts.deskew = true;
for (page_num, result) in raster_pdf(Path::new("scan.pdf"), &opts) {
    match result {
        Ok(page) => { /* page.pixels: width × height bytes; page.dpi */ }
        Err(e) => eprintln!("page {page_num}: {e}"),
    }
}
```

- `raster_pdf`, `raster_pdf_from_bytes`, `render_channel`: page iterators
  and a bounded background channel.
- `open_session` + `render_page_rgb_with` / `render_page_gray`: one open
  document, per-page renders from any thread, with `RenderParams` for
  per-axis scale, a `RenderRegion` window and a `CancelToken`.
- `PageBuffer::render_gray_into` / `render_rgb_into` / `render_rgba_into`:
  pages written into a buffer the caller owns and reuses, at its stride,
  with the page canvas kept between renders; `RasterSession::page_size_px`
  sizes the buffer first.
- `open_session_from_shared` (`Arc<[u8]>`, `&'static [u8]` or any other
  `AsRef<[u8]>` storage, read in place) and `open_session_from_reader`
  (any `Read`, read to its end) beside `open_session` and
  `open_session_from_bytes`; `read_into_memory` makes `open_session` copy
  the file instead of mapping it.
- `page_box` (`PageBox`, default the `CropBox`) and `rotation` (`Rotation`,
  quarter turns added to each page's `/Rotate`) set the page geometry.
- `RasterSession::unsupported_features` and
  `PageDiagnostics::unsupported_features` list the features the renderer
  ignores: JavaScript, XFA, portable collections, 3D and multimedia
  annotations.
- `SessionConfig::gpu_pages` (`vulkan` feature) renders pages made of solid
  fills and strokes whole on a Vulkan GPU beside the CPU workers, with the
  same bytes; `PageDiagnostics::device_page` reports where each page
  rendered.
- `transparent_background` (off by default) renders over a transparent
  backdrop: RGBA pages (`PixelFormat::Rgba8`, straight alpha) from the batch
  entry points and `render_page_rgba`, and an RGB bitmap with an alpha plane
  from `render_page_rgb_with` with `RenderParams::with_transparent_background`.
- `encode_for_gcv`: a size-fitted grayscale JPEG for Google Cloud Vision.
- Encrypted documents are decrypted in process when `decrypt_authorized` is set.
- `overprint` (off by default) simulates overprint and spot inks on pages
  that set it; see the [API reference](../../docs/api-reference.md#sessionconfig).
- `rgb_to_gray` / `rgb_into_gray` (in place, no second buffer) and
  `gray8_into_rendered_page` for callers that build their own gray pages.

Public types are `#[non_exhaustive]`: start from `RasterOptions::new(dpi)`,
`SessionConfig::default()` or `RenderParams::new(scale)` and assign or chain.
The GPU features, and the default `bundled-fonts` feature (Noto fallback
faces for non-CJK scripts, ~0.75 MB; `default-features = false` omits them),
are listed in the [API reference](../../docs/api-reference.md#feature-flags).

Distributed from GitHub only; depend on it with
`rasterrocket = { git = "https://github.com/pthomasfournet/rasterrocket" }`.
The crates.io release (1.0.1) is abandoned and receives no updates. Full
documentation: [Getting Started](../../docs/getting-started.md) and the
[API reference](../../docs/api-reference.md).
