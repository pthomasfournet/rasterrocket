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
- `encode_for_gcv`: a size-fitted grayscale JPEG for Google Cloud Vision.
- Encrypted documents are decrypted in process when `decrypt_authorized` is set.

Public types are `#[non_exhaustive]`: start from `RasterOptions::new(dpi)`,
`SessionConfig::default()` or `RenderParams::new(scale)` and assign or chain.
The GPU features, and the default `bundled-fonts` feature (Noto fallback
faces for non-CJK scripts, ~0.33 MB; `default-features = false` omits them),
are listed in the [API reference](../../docs/api-reference.md#feature-flags).

Distributed from GitHub only; depend on it with
`rasterrocket = { git = "https://github.com/pthomasfournet/rasterrocket" }`.
The crates.io release (1.0.1) is abandoned and receives no updates. Full
documentation: [Getting Started](../../docs/getting-started.md) and the
[API reference](../../docs/api-reference.md).
