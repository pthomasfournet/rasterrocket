# rasterrocket-comic

Comic and scan archives as rasterrocket input: `open_comic` reads a `.cbz`
(ZIP), `.cb7` (7-Zip) or `.cbt` (TAR) archive, decodes its JPEG, PNG, WebP or
TIFF pages in natural (numeric-aware) filename order, and returns the same
grayscale `RenderedPage`s the PDF path produces. An archive holding a single
PDF instead of images renders that PDF. `.cbr` (RAR) is refused with a
convert-to-`.cbz` message.

```rust
use std::path::Path;
use rasterrocket_comic::{ComicError, ComicOptions, open_comic};

fn ocr_archive(path: &Path) -> Result<(), ComicError> {
    let mut opts = ComicOptions::default();
    opts.dpi = 300.0; // recorded on each page for OCR; images keep their pixel size
    for (page_num, result) in open_comic(path, &opts)? {
        match result {
            Ok(page) => { /* page.pixels, page.width, page.height */ }
            Err(e) => eprintln!("page {page_num}: {e}"),
        }
    }
    Ok(())
}
```

All pages are decoded before `open_comic` returns. Entries are read under a
512 MiB decompressed-size cap and image dimensions are checked before
decoding. `rrocket` uses this crate for archive input. See the
[API reference](../../docs/api-reference.md#rasterrocket-comic-crate--comic-archive-input).
