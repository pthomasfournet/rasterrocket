# rasterrocket-comic

Comic and scan archives as rasterrocket input: `open_comic` reads a `.cbz`
(ZIP), `.cb7` (7-Zip), `.cbt` (TAR) or `.cbr` (RAR) archive, decodes its JPEG, PNG, WebP, JPEG XL, AVIF or
TIFF pages (every page of a multi-page TIFF) in natural (numeric-aware)
filename order, and returns the same
grayscale `RenderedPage`s the PDF path produces. An archive holding a single
PDF instead of images renders that PDF.

```rust
use std::path::Path;
use rasterrocket_comic::{ComicError, ComicOptions, open_comic};

fn ocr_archive(path: &Path) -> Result<(), ComicError> {
    let mut opts = ComicOptions::default();
    opts.dpi = 300.0; // recorded on each page for OCR; images keep their pixel size
    for (page_num, result) in open_comic(path, &opts)? {
        match result {
            Ok(page) => { /* page.pixels.as_u8(), page.width, page.height */ }
            Err(e) => eprintln!("page {page_num}: {e}"),
        }
    }
    Ok(())
}
```

`open_comic` returns an iterator that decodes each page when it is reached;
its `len()` is the number of pages still to come and `retain` drops pages
before any is decoded. `next_job` takes the next page as a `PageJob`, which
owns what its page is rendered from and renders on any thread, so pages can
decode in parallel (`rrocket` renders them on its thread pool). Entries are read under a
512 MiB decompressed-size cap per entry and a total cap for the archive
(`ComicOptions::max_archive_bytes`, default 4 GiB; an archive over it fails
with `ComicError::ArchiveTooLarge`), and image dimensions are checked before
decoding. An embedded PDF opens with `ComicOptions::session`, a
`SessionConfig` (overprint, antialiasing, annotations, fonts, decryption,
backend), whose `rotation` also turns every page image; `rrocket` fills it from the same flags as for PDF input and uses
this crate for archive input. See the
[API reference](../../docs/api-reference.md#rasterrocket-comic-crate--comic-archive-input).

## AVIF pages and the `av1-asm` feature

AVIF pages are read by this crate's own HEIF container parser and decoded by
the workspace's local fork of `rav1d` (`crates/rav1d`): the primary item, a
coded picture or a grid of tiles, with its clean aperture, rotation and
mirroring applied; 8, 10 and 12 bits; alpha planes are not read and ICC
profiles are not applied.

The default feature `av1-asm` builds the decoder's hand-written assembly
kernels. An x86 build then runs the `nasm` executable at build time (it must
be on `PATH`); an arm64 build assembles them with the C compiler. Without the
feature (`default-features = false`) the decoder is Rust alone, needs no
assembler, and is several times slower.
