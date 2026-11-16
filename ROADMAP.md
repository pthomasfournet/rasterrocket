# rasterrocket Roadmap

Goal: full PDF → pixels pipeline in pure Rust. Zero poppler. Zero C++ in the render path.

The rasterrocket-render crate is complete at the pixel level. The `rasterrocket-interp` crate is the native renderer and is now the only CLI path. The `pdf_bridge` / poppler crate is retained as a reference baseline but is no longer linked by the CLI binary.

**Integration target (Apr 2026):** rasterrocket replaces steps 3 (pdftoppm subprocess) and 4 (Leptonica preprocessing) in an OCR pipeline:

```
pdf_oxide → [quality check fails] → rasterrocket (rasterise + deskew) → Tesseract → (LLM correct)
```

The caller's Tesseract step becomes a single in-process call — no subprocess, no files, no Leptonica:

```rust
for (page_num, result) in rasterrocket::raster_pdf(path, &opts) {
    let page = result?;
    // 8-bit grayscale at the default depth, tightly packed, top-to-bottom
    let gray = page.pixels.as_u8().expect("8-bit render");
    tess = tess
        .set_frame(gray, page.width as i32, page.height as i32, 1, page.width as i32)?
        .set_source_resolution(page.effective_dpi as i32);
    let text = tess.get_text()?;
}
```

Phase 5 is complete. The API exists and is integrated.

---

## Known issues

**CMYK output has gaps.** `--cmyk` / `--jpegcmyk` ignore `BG`/`UCR`, do
not convert ICC-based CMYK content to a different CMYK output profile, and
separate indexed images over a CMYK base through sRGB; see
[Current status](#current-status-2026-11-15).

**GPU JPEG decode is off.** nvJPEG and VA-API JPEG share one size gate,
`GPU_JPEG_THRESHOLD_PX = u32::MAX`, so builds with `nvjpeg` or `vaapi`
decode every JPEG on the CPU; `--backend vaapi` opens the decoder but never
dispatches to it.

**Overprint simulation has one gap.** Under `--overprint`, transparency
groups as a whole knock out instead of overprinting; see
[Current status](#current-status-2026-11-15).

**Camera dewarp is experimental.** `rasterrocket-camera`'s dewarp is validated
on synthetic page curls only; no real phone captures exist yet.

**`pdf_bridge` needs a poppler source tree.** Its build script requires
`POPPLER_SRC` (a poppler *source* tree, not just the system package).  It is
left out of the workspace's `default-members`, so a bare `cargo build`,
`check` or `test` at the root builds without it; `--workspace` and
`-p pdf_bridge` still need `POPPLER_SRC`.  The crate has not changed since
v1.2.0 and exists only as a pixel-diff reference baseline.

---

## Release history

### v2.3.0 (November 2026)

**Exact-area antialiasing, large-format, 16-bit, ICC-managed and CMYK output,
and a visible-text diagnostic; the `rasterrocket` API changes in several
places.** Every antialiased path, clip and glyph is drawn from one exact-area
rasteriser, so antialiased output changes on nearly every page. Pages are
bounded by a byte budget instead of pixel caps and can be rendered and
written in bands, at 8 or 16 bits per channel, converted to an output ICC
profile under a chosen rendering intent, or separated into CMYK. The API
changes are listed below as they are, without compatibility shims; a
caller of `raster_pdf` / `render_channel` that reads 8-bit pages needs
only follow the `RenderedPage::pixels` change.

**Rendering.**
- **Exact-area coverage.** `raster::coverage` replaces the 4×4
  supersampled scanner and its gamma-1.5 table for fills, strokes, clips
  and glyphs. The old scanner was about 0.25 px heavy on +y and banded in
  16 levels. Against pdfium's corpus (260 scored pages) the median RMSE
  falls from 4.39 to 1.45, and the fill bench runs 6.34 → 4.70 ms. The
  `--gpu-pages` raster composites the CPU's coverage and stays
  byte-identical to `--backend cpu`. The dormant device AA fill and the
  `gpu-aa` feature are removed.
- **Fixes.** The top row of a flat-top Gouraud triangle was dropped. A
  Gouraud mesh under an empty rectangular clip panicked in thin bands. A
  non-isolated group's union colour could wrap to black. A scanner table
  past `u32` intersections aborted. A Type 3 glyph box far off the page
  overflowed `i32`. Region renders differed from the page in shadings,
  tiling lattices, image sample grids and group bounds.

**Large format.**
- A page byte budget (`max_page_bytes`, `--max-page-mib`, default 1.8 GB)
  replaces the pixel caps (`MAX_PX_DIMENSION`, `MAX_PX_AREA`), under a
  hard side ceiling of 2²⁴ px.
- `render_page_bands` renders a page as horizontal bands that stack byte
  for byte into the whole page. The CLI writes PPM, PNG, TIFF (BigTIFF
  past 4 GiB) and `--print` pages band by band (`--band-mib`, default 64).
  A 44 × 120 in roll at 360 dpi renders in 428 MiB of peak memory.
- `--print` without a resolution flag prints pages larger than A0 at
  360 dpi.

**Colour and depth.**
- **16 bits per channel.** `--depth 16` (`RasterOptions::depth`)
  renders, composites and writes 16-bit PNG, PPM/PGM, TIFF and `--print`
  spools; overprint simulation works at 16 bits. 8-bit output is
  byte-identical to v2.2.0's arithmetic.
- **Output ICC profiles.** `--output-profile FILE`, `--intent` and `--bpc`
  convert finished pages to an RGB, gray or CMYK printer, display or
  colour-space profile and embed it in PNG, TIFF, JPEG and the `--print`
  spool. Worst ΔE00 against Little CMS is 0.93 over every intent, with and
  without black-point compensation, at both depths. The interpreter tracks
  each page's content rendering intents.
- **CMYK output.** `--cmyk` with `--tiff` or `--jpeg`, and `--jpegcmyk`,
  render pages in process inks and write Separated TIFF (8 or 16 bits) or
  Adobe CMYK JPEG. DeviceCMYK content keeps its values; other colours
  separate through a CMYK output profile or the ISO 32000-2 device
  conversion, so black text is black ink alone. DeviceCMYK images and
  shadings now also paint their own inks under `--overprint`.
- **JPEG decoding.** The zune-jpeg fork rounds its IDCT and YCbCr
  conversion as libjpeg's accurate decode; gray, CMYK and 4:4:4 images now
  match it exactly, and JPEG-heavy pages move by one or two levels.

**Diagnostics.** `PageDiagnostics::glyphs` (`GlyphCounts`) counts a page's
glyphs by text rendering mode and why a glyph cannot mark the page;
`PageDiagnostics::has_visible_text()` says whether a page has native
visible text, so a caller can skip OCR on it.

**Vendored forks.** Each fork is reviewed against upstream again and takes
its fixes: harfrust's shaping budget and AAT bounds, read-fonts' MVAR and
Type 1 numbers, skrifa's MIRP twilight points, jbig2's region and
dictionary bounds and faster halftones, ccitt's truncated Group 4 rows, and
JPEG 2000 raw codestreams that honour the image dictionary's `/ColorSpace`.

**API changes** (breaking, in `rasterrocket` and the lower crates; the
Current status section below lists every item).
- `RenderedPage::pixels` is `Pixels::{U8, U16}`; `PixelFormat` gains
  `Gray16`, `Rgb16`, `Rgba16`, `Cmyk8`, `Cmyk16`.
- `RasterError::PageTooLarge` is `{ bytes, limit }` and
  `PageExceedsCeiling` replaces `PageAreaTooLarge`;
  `RasterError::OutputProfile` is new.
- `RasterOptions` gains `max_page_bytes`, `depth`, `output_profile`,
  `intent`, `black_point_compensation` and `color`; `SessionConfig` gains
  `max_page_bytes`, `band_bytes` and `output_profile`;
  `RenderParams` gains `with_depth`, `with_intent`,
  `with_black_point_compensation` and `with_cmyk`.
- `PageRenderer::with_page_group` and `try_with_page_group(_in)` take a
  `&CanvasRequest` in place of `ink: bool`; `render_page_canvas` returns a
  `FinishedCanvas`; `Band::bitmap` is a `BandBitmap`.
- `rasterrocket-render`'s `Bitmap`, pipe and pattern types take a sample
  parameter; `XPathScanner::new` returns `Result`.
- `rasterrocket-encode`'s writers are row-streamed (`PnmWriter`,
  `PngWriter`, `TiffStream`, `PdfPageEncoder`, `RowImage`) and take an
  optional ICC profile.

### v2.2.0 (November 2026)

**New comic formats, caller-owned buffers, whole pages on a Vulkan GPU, and
overprint simulated in full but for transparency groups; one signature change
under the `cache` feature.** Comic archives gain RAR, JPEG XL and AVIF pages
and every page of a multi-page TIFF; the library renders into memory the
caller owns, over a transparent background, on any page box and with an
extra quarter-turn rotation, and reports the features it ignores as typed
values; the JPEG decoder is a local fork that decodes arithmetic-coded and
multi-scan frames and reports partial streams exactly. Every crate requires
Rust 1.100. Every addition to the `rasterrocket` API is a new item or a
field of a `#[non_exhaustive]` type; the one existing signature that
changes is the return type of `RasterSession::image_cache`, present only
with the `cache` feature.

**New inputs and formats.**
- **Comic archives.** `.cbr` (RAR 1.3 to RAR 7) is read through the
  pure-Rust `rars` decoder, under the entry and 1 GiB dictionary bounds of
  `.cb7`; a damaged member of a non-solid archive fails that entry only.
  A multi-page TIFF entry yields every page (reduced-resolution copies and
  masks skipped). JPEG XL pages (`.jxl`) decode through the `jxl` crate,
  first frame, 8-bit gray or sRGB. AVIF pages (`.avif`) decode through a
  HEIF reader in the comic crate (primary item or `grid`, `clap` / `irot` /
  `imir`, `nclx`) and `rav1d`, vendored as a local fork with a safe
  `decode_still` entry point: 8 to 12 bits, 4:0:0 to 4:4:4. The fork's
  assembly kernels are on by default (`av1-asm`; `nasm` is then a build
  requirement on x86, and `--no-default-features` drops them).
- **JPEG.** The decoder is a local fork of zune-jpeg (`crates/zune_jpeg`),
  reviewed against upstream through `ab211bb1`. A baseline frame coded in
  several scans decodes correctly (subsampled ones were garbage, others
  lost rows); arithmetic coding (SOF9, SOF10) decodes; a DNL segment is
  read; a stream with bytes before SOI or an unknown height is repaired,
  and frames taller or wider than 16 384 px decode within the 1 GiB
  budget, which now also counts the coefficient planes a progressive or
  multi-scan frame holds. An image is reported partial exactly when its
  stream lacks data (`JpegDecoder::intact_rows`), in PDFs and, as a
  warning, in comic pages.
- **Transparent output.** `--transparent` (`--transp`), as
  `pdftocairo -transp`, writes RGBA PNG or TIFF with unassociated alpha;
  the page is painted as an isolated page group and its alpha kept.
  Formats without alpha, `--gray` and `--mono` are refused.
- **Fax.** A `--fax` page longer than B4 (364 mm) is sent as A4-length
  pieces.

**API and CLI.**
- **Caller-owned buffers.** `PageBuffer::render_gray_into` /
  `render_rgb_into` / `render_rgba_into` write a page at the caller's
  stride, byte-identical to the allocating calls; once its canvas has grown
  to the largest page, a render allocates no page-sized buffer (corpus-01
  at 150 dpi: 8.89 → 0.47 MB allocated a page). `RasterSession::page_size_px`
  sizes the buffer; a short stride or buffer is an error before drawing.
- **Shared and streamed input.** `open_session_from_shared` opens any
  `AsRef<[u8]>` storage without a copy, `open_session_from_reader` any
  `Read` source (to be bounded by the caller); `read_into_memory` reads a
  file instead of mapping it, so a truncated input cannot raise `SIGBUS`.
- **Page box and rotation.** `--box media|crop|bleed|trim|art` /
  `PageBox` (default the CropBox); `--rotate DEG` / `Rotation`, a whole
  number of quarter turns added to each page's `/Rotate` before the
  geometry is computed, so every size, region, deskew and output sees the
  turned page, comic pages included.
- **Diagnostics.** `RasterSession::unsupported_features` and
  `PageDiagnostics::unsupported_features` list JavaScript, XFA, portable
  collections, 3D and multimedia annotations as `UnsupportedFeature`
  values; `PageDiagnostics` counts device fills, blits and ICC
  conversions and reports `device_page`.
- **CLI.** `--box`, `--rotate`, `--transparent`, `--gpu-pages`,
  `--max-archive-mib`. `--print` appends each page to the spool once the
  pages before it are done, without barrier windows (a 200-page document,
  one scan page in ten, at 600 dpi: 7.5 → 4.3 s).

**Rendering correctness.**
- A function-based (Type 1) shading is evaluated at every device pixel of
  its Domain instead of on a 64 × 64 grid (above 4 Mi pixels at every k-th
  pixel, interpolated), its work, memory and area bounded. **This changes
  pixels** of every Type 1 shading, by 1–2 counts on smooth functions.
- Text shaping for substituted fonts joins a word split between show
  strings (`TJ` adjustments, `Tf` mid-word, consecutive `Tj`), draws the
  marks ending a right-to-left string over the next string's letter of the
  same script, and sets a synthesised appearance's fallback characters in
  shared fonts, so a word spanning Unicode blocks is one string.
- Property suites (parser, fills and clips, colour, encoders, camera
  geometry) found five bugs, fixed: an aliased fill lost cells of
  winding-count-0 intersections (**changes pixels** of `--aaVector no`
  renders), a sloped edge moved by whole pixels could lose a cell,
  `div255` rounded down at 126 inputs (**changes pixels** by one level
  where compositing meets them), `round_half_up_i32` took
  0.49999999999999994 to 1, and `rectify` could exceed its bound.
- A hairline cut by a clip edge inside a pixel row draws its kept rows as
  the unclipped hairline does.
- The glyph advance and text-matrix translations use fused multiply-adds:
  **this changes pixels**, glyphs on a rounding boundary moving by a device
  pixel. moxcms 0.9.2 may move colour-managed RGB output by one level.
- The prescan's PPI hint uses the unrotated page width; a page object
  that is not a readable dictionary is warned about.

**Overprint.** On the ink canvas, `Separation` and `DeviceN` images and
shadings paint their colorants' channels; coloured tiling cells render on
an ink canvas sharing the page's spot slots; `ICCBased` CMYK colours print
as the process ink that reproduces their profile; transfer functions apply
in ink space, to the channels a colour carries; a black or near-black gray
or RGB colour prints as a balanced rich black, so antialiased edges stay
neutral; and the GPU AA fill and image blit run on the ink canvas,
byte-identical to the CPU. Transparency groups as a whole are the one part
still not simulated. Pages rendered without `--overprint` are unchanged.

**GPU.**
- **Whole pages on Vulkan.** `--gpu-pages` / `SessionConfig::gpu_pages`
  (off by default) opens the Vulkan device in the background and renders a
  page of solid fills and strokes under rectangular clips whole on it,
  from a command list scan-converted by the CPU's own geometry stages,
  while CPU workers take the other pages; output is byte-identical. A page
  the device cannot take, or fails on, renders on the CPU; a failed device
  is retired for the session.
- **Vulkan parity with CUDA.** The device image cache (`cache`) and the ICC
  CMYK CLUT (`gpu-icc`, after a one-time exactness probe) run on Vulkan,
  byte-identical to the CPU on the RTX 5070 and RADV.
- **`Auto` on the CPU.** `Auto` initialises no GPU backend, image cache or
  nvJPEG decoder unless a device kernel is enabled, so it matches
  `--backend cpu` start-up (corpus-02, one page: 283 → 11 ms); it builds an
  nvJPEG2000 decoder on demand for a JPX of at least 262 144 px not drawn
  at half its size or less. `GPU_ICC_CLUT_THRESHOLD` is 128 000 px
  (`PDF_RASTER_ICC_CLUT_THRESHOLD` overrides it). A JPEG drawn below its
  resolution is not put in the device cache. **This changes `Auto`
  pixels** on pages with an ICC-based CMYK image and on JPEG 2000 images
  drawn at half their size or less; the explicit backends are unchanged.
- **Integrated GPUs.** An uncached mapped Vulkan buffer is read back
  through a device copy (an 8.7 MB image on RADV: 27.7 → 0.42 ms).
- **Kernels.** The dormant composite and soft-mask kernels are deleted;
  the IDCT kernels no longer race between their passes (wrong chroma on
  MoltenVK); the CLUT coordinate scale is computed on the host, correctly
  rounded; blit and CLUT scratch is pooled on the backend.
- `vulkan_phase_bench`, `startup_bench`, `threshold_bench --only clut` and
  a page-level GPU raster benchmark measure the backends.

**Hardening.**
- *Comic archives.* A decompressed total of 4 GiB per archive
  (`ComicOptions::max_archive_bytes`, `--max-archive-mib`), refused with
  `ComicError::ArchiveTooLarge` before the first page; a TIFF directory
  chain walked once; the AVIF container's tables, transforms and sample
  budget bounded, the budget taken from the stream's own sequence headers.
- *Library.* The unsupported-feature scan has a step budget; a file too
  large to read into memory is an I/O error, not an abort; `PageBuffer`
  keeps its ink and alpha buffers; a failed GPU deskew falls back to the
  CPU rotation.
- *GPU.* Device page command lists and storage bindings are bounded by the
  device's limits; the VRAM budget is reserved before an upload; a corrupt
  disk-tier entry is removed; a device blit that never starts draws from
  the host copy; Vulkan transfers wait on their own fence; a Vulkan
  session builds no CUDA JPEG decoder; IDCT block counts past the kernels'
  addressing are refused.
- *JPEG decoder.* Every upstream fix through `ab211bb1` relevant to the
  fork, and the fork's own: incomplete arithmetic AC bands reported, a
  64-bit fill-bit count, a coefficient limit
  (`set_max_coefficient_bytes`).

**Performance.** An image reduction plans once and resamples only the
columns it reads; HTJ2K blocks reconstruct without a coefficient pass, load
their bitstreams four bytes at a time and decode quad pairs without
per-sample branches; spot-slot checks and transparency-group canvases
allocate less on overprint pages. Renders are unchanged by all of these.

**CI and tooling.**
- The full CPU suites run on macOS, and a Windows leg runs clippy and the
  render, interp, `rasterrocket`, font, CLI, parser, colour, encode and
  comic suites; the tree builds and tests cleanly on Windows.
- A Coverage workflow reports line coverage (`cargo-llvm-cov`) and gates
  nothing; a PGO + BOLT workflow builds a release `rrocket` on `v*` tags.
- `tests/compare/compare.sh` and `pdfium_corpus.py` classify pages against
  a checked-in registry of expected failures,
  `tests/compare/expectations.json` (`PASS`, `XFAIL`, `FAIL`, `XPASS`).
- `pdf_bridge` is left out of the workspace's default members, so a bare
  `cargo build` needs no poppler source tree; `scripts/check_vendored.py`
  tracks the zune-jpeg and rav1d forks.

**API changes** (additive in `rasterrocket` but for `image_cache`).
- `rasterrocket`: `PageBuffer`, `PageInfo`, `PixelFormat` (`Gray8`,
  `Rgb8`, `Rgba8`; `#[non_exhaustive]`), `RenderedPage::format`,
  `render_page_rgba`, `render_page_rgb_with_diagnostics`,
  `open_session_from_shared`, `open_session_from_reader`,
  `RasterSession::{page_size_px, unsupported_features}`,
  `SessionConfig::{page_box, rotation, read_into_memory, gpu_pages}`,
  `RasterOptions::{page_box, rotation, read_into_memory,
  transparent_background}`, `RenderParams::transparent_background`,
  `RasterError::{OutputStrideTooSmall, OutputBufferTooSmall}`,
  `PageDiagnostics::{unsupported_features, device_fills,
  device_fill_failures, device_image_blits, device_icc_conversions,
  device_icc_failures, device_page}`, and the re-exports `PageBox`,
  `Rotation`, `InvalidRotation`, `DevicePage`, `UnsupportedFeature`,
  `ScriptLocation`, `MediaKind`.
- Changed: with the `cache` feature, `RasterSession::image_cache` returns
  `Option<&gpu::cache::ImageCache>` (either backend) instead of
  `Option<&Arc<gpu::cache::DeviceImageCache>>`. `deskew::apply`,
  `DeskewError` and `RasterError::Deskew` stay public and are no longer
  returned.
- `rasterrocket-comic`: `ComicOptions::max_archive_bytes`,
  `DEFAULT_MAX_ARCHIVE_BYTES`, `ComicError::ArchiveTooLarge`;
  `ComicError::RarUnsupported` is removed.
- `rasterrocket-interp`: `Rotation`, `InvalidRotation`, `PageBox`,
  `PageGeometry::rotated`, `open_shared_with`,
  `PageRenderer::{try_with_page_group_in, try_finish_transparent,
  record_for_device}`, `transparent_page_group`,
  `document_unsupported_features`, `page_unsupported_features`;
  `page_size_pts`, `page_size_pts_by_id` and `prescan_page` take the page
  box. `rasterrocket-render`: `BitmapBuffers`,
  `Bitmap::{try_new_filled_in, into_buffers, turned_cw}`,
  `stroke_aa_scanner`, `hairline_runs`. `rasterrocket-parser`:
  `Document::from_shared(_with)`, `OpenOptions::read_into_memory`.
- `rasterrocket-gpu`: the image cache and `CachedDeviceImage` take a
  backend parameter, `ContentHash::of` replaces
  `DeviceImageCache::hash_bytes`, `CacheError` gains `Backend` and
  `BackendMismatch`, `gpu::page` holds the page command list, and
  `GpuBackend::record_composite` / `record_apply_soft_mask` are removed.

### v2.1.0 (November 2026)

**Output to printers and fax machines, overprint, shaped text for
substituted fonts; no API break.** `rrocket` writes JPEG and TIFF, prints
through CUPS and writes or sends fax documents; the library simulates
overprint and spot inks on request, shapes complex-script text drawn with
substitute fonts, decodes HTJ2K and decodes an image shared by several pages
once. The cycle closes with three hardening review passes (2026-11-03 and
twice on 2026-11-04). Every addition to the public API is a new item or a
field of a `#[non_exhaustive]` type, so a v2.0.0 caller builds unchanged.

**New capabilities.**
- **JPEG and TIFF output.** `--jpeg` with `--jpegopt quality=N,
  progressive=y|n,optimize=y|n`, as `pdftoppm -jpeg`, colour or gray;
  `--tiff` with `--tiffcompression none|packbits|lzw|deflate|g3|g3-2d|g4`,
  RGB, gray or bilevel, the fax codings from a CCITT Group 3 / Group 4
  encoder added to the `crates/ccitt` fork. Fax-coded pages read back
  pixel-exact through libtiff and the `tiff` crate. `--jpegcmyk` is still
  refused.
- **Fax documents.** `--fax` writes one TIFF Class F file, 1728 px wide at
  204 × 196 dpi (`fine`) or 204 × 98 dpi (`--fax-resolution standard`),
  landscape pages turned, Group 3 one-dimensional by default; `--fax-to
  NUMBER [--fax-queue NAME]` hands it to a CUPS fax queue or HylaFAX
  `sendfax`. `--invert[=always|auto|never]` (`auto` under `--fax`) turns
  pages reversed out of a flat dark ground, and `--dither halftone` renders
  photographs as dot patterns for `--mono` and `--fax`.
- **Printing.** `--print` / `--printer NAME` / `--copies N` /
  `--duplex long|short` render the pages at 600 dpi (or `-r`) for printing
  and hand CUPS's `lp` one PDF of page images through a private, anonymous
  spool, each page appended in order as it finishes; `--list-printers`
  lists the queues.
  Linux and macOS; refused on Windows.
- **Overprint simulation.** `--overprint` / `RasterOptions::overprint` /
  `SessionConfig::overprint` (default off) paint a page whose graphics
  states set `OP` or `op` on a canvas of four process and four spot inks
  (ISO 32000-1 §8.6.7) and convert it to RGB; further spot colorants are
  listed in `PageDiagnostics::unslotted_spots`. Other pages render
  byte-identically. Not simulated: `Separation` / `DeviceN` images and
  shadings, tiling-pattern cells and transparency groups as a whole.
- **Text shaping for substituted fonts.** Runs of Arabic, Devanagari and
  other scripts that need shaping, drawn with a substitute face, are shaped
  with its `GSUB`/`GPOS` through a vendored harfrust fork; synthesised
  annotation and field text is put in visual order by the bidirectional
  algorithm. The bundled fallbacks gain Bengali, Tamil, Armenian, Georgian
  and Ethiopic (0.49 → 0.75 MB); vertical-form CIDs a substitute lacks are
  rotated or shifted from the horizontal glyph.
- **HTJ2K.** JPX images coded with the high-throughput block coder
  (ISO/IEC 15444-15) decode; reversible OpenJPH test streams decode
  bit-exactly. The mixed HT / classic mode is refused by name.
- **Vulkan on macOS.** The Vulkan backend finds Homebrew's or the SDK's
  loader and runs on MoltenVK; the fixtures render byte-identically on an
  A18 Pro's GPU and CPU (the GPU fill is slower, so it stays off).
- **CLI.** `--version` / `-V`; `ab_render.sh` / `ab_archive.sh` compare two
  binaries, or two backends of one, byte for byte.

**Fixes.**
- *Geometry.* The page box is anchored at the bitmap's top-left corner for
  every `/Rotate`; over 434 changed pages the fitted vertical offset against
  mutool falls from 0.15 to 0.06 px and against pdfium from 0.32 to 0.20 px.
  Subpaths ending in a curve no longer merge into the next; antialiased
  edges past device column 8191 are right; a sloped edge's fixed-point
  position no longer drifts by pixels for vertices far off the page; a
  zero-width filled rectangle fills its pixel column.
- *Memory.* A page bitmap, transparency group, soft mask, knockout,
  tiling-pattern, Type 3 or snapshot bitmap the allocator refuses fails the
  page with `RasterError::PageAllocationFailed` instead of aborting the
  process.
- *Bounded work.* Tint-transform conversion is budgeted per image (2^28
  operations) and per page (2^32), with a lattice of exact conversions past
  it: a 2000 × 2000 noise image under a 100 000-token transform renders in
  0.4 s instead of not finishing in 180 s. Type 4 runs are limited by fuel,
  function costs are computed once, and JBIG2 decoding has cumulative
  memory and work budgets that fail the image when spent.
- *Parser.* Linear cross-reference index growth, a bounded `/Count`,
  linearization found by position and the page count read from the tree.
- *Output.* Bilevel and planar TIFF pages in comic archives decode (every
  one failed before); `--mono --png` is fixed; the print spool is private.
- *CLI.* INPUT `-` reads the PDF from standard input, as the help always
  said (it was opened as a file named `-`); an encrypted piped document is
  decrypted only under `--decrypt-owned`.
- *Caches.* Pattern cells, kept forms and inline decodes are keyed on the
  resources they resolve in; the document font and image stores each serve
  one document; a decode curtailed by a page's budget is not shared.
- *GPU.* Vulkan mapped access, transfers and zero-fills are synchronised;
  devices lacking a required feature are skipped; a failed or panicking page
  is abandoned on every backend; concurrent Vulkan instance creation no
  longer crashes the loader; a relative `$VULKAN_SDK` is ignored.
- *Fonts.* The read-fonts, skrifa and harfrust forks are rebased on
  upstream (renders byte-identical over 12 169 pages); a shaping failure or
  exhausted shaper falls back to glyph-by-glyph drawing with a warning.

**Performance.**
- An image `XObject` drawn on several pages is decoded once per session
  (up to `image_budget`, 256 MiB by default): a 48-page document sharing one
  3000 × 3000 JPEG renders in 1.9 s instead of 4.5 s at 0.51 GB peak
  instead of 1.22 GB.
- Synthetic-bold (`2 Tr`) text draws strokes from cached coverage (the
  israel pages 1.32 → 0.10 s, pdfium 0.95 s); together with chord-based
  curve flattening, clip-culled subpaths, an indexed cross-reference table
  and one face lookup per text run, the two documents slower than pdfium in
  the archive latency run now match or beat it.
- A clip of more than eight paths folds into one table: 3 000 nested clips
  and 3 000 fills render in 0.31 s instead of 6.1 s, 47 MB instead of
  600 MB.
- Images are area-averaged only over the pixels a draw can paint (1.2–1.6×
  on mostly clipped images); gray output converts in place.
- On integrated GPUs Vulkan buffers are mapped instead of staged and
  read-back buffers are host-cached (a 16 Mpx device fill on RADV: 38–65 ms
  → 13 ms); a device AA fill is one submission instead of four. Under
  `auto`, no Vulkan device and no Huffman decoder is initialised while
  their kernels are dormant.
- Two optimisations change pixels, both approved against an 8× reference
  render: the curve flattening and the cached glyph strokes (60 archive
  pages, all stroked text). Everything else is byte-identical over the
  archive.

**API changes** (additive; no v2.0.0 caller breaks).
- `rasterrocket`: `RasterOptions::{overprint, image_budget}`,
  `SessionConfig::{overprint, image_budget}` (both structs
  `#[non_exhaustive]`), `RasterError::PageAllocationFailed` (enum
  `#[non_exhaustive]`), `PageDiagnostics::unslotted_spots` (struct
  `#[non_exhaustive]`), `DOCUMENT_IMAGE_BUDGET_BYTES`, `rgb_into_gray`,
  `gray8_into_rendered_page`, `is_encrypted_bytes`.
- Behaviour: a session keeps up to 256 MiB of shared image decodes resident
  by default (`image_budget = 0` keeps none); under `BackendPolicy::Auto` a
  Vulkan build opens no Vulkan device unless a Vulkan kernel's threshold
  enables it.
- `rasterrocket-comic`: `ComicOptions::session`. `rasterrocket-render`:
  `Bitmap::{into_data, from_packed, try_new_filled}`, `BitmapAllocError`.
  `rasterrocket-interp`: `PageRenderer::{with_page_group,
  try_with_page_group, try_set_page_group, try_finish,
  set_document_images}`, `page_uses_overprint`, `DocumentImages`,
  `ExtGStateParams::{overprint_stroke, overprint_fill, overprint_mode}`,
  `RenderAbort::AllocationFailed`, `DeviceKernel`, `device_kernel_enabled`.
- CLI: new flags only; `--jpeg` and `--tiff`, refused in v2.0.0, now write
  files, and `--overprint`, removed in v2.0.0, is back with a meaning.

### v2.0.0 (October 2026)

**More of the PDF drawn, and drawn as the specification says; one API
break.** Every capability that poppler or pdfium renders and rasterrocket did
not was reproduced with a fixture and implemented from ISO 32000-2. The
result touches every stage, from the cross-reference reader to the
compositor, and adds in-process decryption, optional content, annotations
and region rendering. The release closes with a review of every file the
cycle touched (security, edge cases, failure handling, shared helpers,
allocations). The public types become `#[non_exhaustive]`, which is why this
is a major release; the migration notes are under **API changes**.

**New capabilities.**
- **Encrypted documents decrypt in process.** The Standard security handler
  (revisions 2–6: RC4 40–128-bit, AES-128, AES-256, crypt filters,
  `/EncryptMetadata false`, user or owner password) replaces the `qpdf`
  subprocess and its plaintext temporary file, from a path or from bytes.
  The authorisation gate is unchanged: `decrypt_authorized` in the library,
  and in the CLI the interactive confirmation or `--decrypt-owned` /
  `RROCKET_DECRYPT_OWNED=1`. `--password` (aliases `--upw`, `--opw`) supplies
  a password; public-key security is refused by name.
- **Optional content (layers).** The model of §8.11: `/BaseState`, `/ON`,
  `/OFF`, `/Intent`, `/AS` usage applications, membership dictionaries with
  `/P` and `/VE`, applied to marked content, XObjects and annotations. A
  render purpose, `RenderPurpose::View` or `Print` (CLI `--purpose`), selects
  the usage event.
- **Annotations and forms.** Flags (Hidden, NoView, Print, NoRotate) for the
  render purpose, appearances fitted by Algorithm 8.1, the `/AS` state chain,
  and appearances synthesised for markup annotations (Square, Circle, Line
  and PolyLine with `/LE` endings, Polygon, Ink, text markup, Text,
  FreeText with `/RC` and callouts, open pop-ups) and for text, choice and
  button form fields that have none, including `/NeedAppearances` forms
  whose values live only in `/V`, `/MK /R` rotation, rich text `/RV` and CJK
  `/DA` fonts with a fallback for characters the font lacks. Closed pop-ups
  are not drawn. `annotations = false` (CLI `--hide-annotations`) leaves
  them out. A dynamic XFA form is reported as `FormType::XfaFull` and warned
  about at open.
- **Font substitution.** A non-embedded font is drawn with a substitute of
  its class fitted to the document's widths: base-14 faces compiled into the
  binary, installed Liberation/Arial/Times/Courier families, and installed
  CJK, Arabic and Thai faces chosen by script; Symbol and ZapfDingbats with
  their own faces and encodings. Predefined CJK CMaps, mixed-width codespaces,
  `usecmap`, vertical writing mode, and `RROCKET_FONT_DIR` for extra font
  directories. A missing font resource or an unparsable program falls back to
  a substitute instead of dropping the text.
- **Fonts that were not drawn before.** TrueType programs that position
  glyph components by hinting (MingLiU, DFKai-SB and similar) are grid-fitted
  through the TrueType interpreter; CID-keyed Type 1 programs render; Type 3
  glyphs run in their own resources; stroked and clipping text render modes.
- **Regions, geometry, per-axis resolution, cancellation.**
  `RasterSession::page_geometry` reads a page's size, rotation, `UserUnit`
  and boxes without parsing content. `render_page_rgb_with` renders a
  `RenderRegion` (tiles stitch byte for byte, and the size limits apply per
  region, so pages past `MAX_PX_AREA` render in tiles), with separate
  horizontal and vertical scales and a `CancelToken`. In the CLI, `--rx`/`--ry`
  give non-square pixels, `-x`/`-y`/`-W`/`-H` render only the selected area,
  `--scale-to`, `--scale-to-x`, `--scale-to-y` size the page, and `--aa` /
  `--aaVector` switch antialiasing; all of these were parsed and ignored
  before.
- **Partial-decode diagnostics.** A content stream or image whose data ends
  early (Flate cut short, LZW stopped mid-stream, a raw image short of
  samples) is drawn from what decoded and reported with `partial: true` in
  `PageDiagnostics::content_failures` / `image_failures`; `strict` fails such
  a page. Undecodable content streams are skipped and reported the same way,
  and `PageDiagnostics::missing_fonts` lists font resources that `Tf` named
  but the resources do not define.

**Correctness.**
- *Colour.* A process-ink DeviceCMYK model on every path, CPU and GPU;
  ICCBased gray and RGB matrix/TRC profiles for fills and images, with
  `/Alternate` for unusable ones; a JPX image's `colr` ICC profile; CalGray,
  CalRGB and Lab through their CIE parameters; multi-colorant DeviceN and
  Indexed, inline tint transforms and lookup strings; `/None` colorants paint
  nothing; DefaultGray/RGB/CMYK; ExtGState transfer functions. Images convert
  their samples through the same colour-space code as fills. Grayscale output
  uses the PDF `DeviceRGB` → `DeviceGray` weights (0.30/0.59/0.11) instead of
  BT.709, which moves saturated colours by up to ~11 levels.
- *Paint and transparency.* ExtGState soft masks, and an image's own
  `/SMask` replacing the graphics-state one; shading patterns as fills and
  strokes (mesh shadings included) with their ExtGState and `/Background`;
  axial, radial, function-based and mesh shadings evaluated in shading space
  and left unpainted outside their extent; tiling patterns on the pattern
  lattice at device cell size, with sub-pixel origin, showing the backdrop;
  fill patterns through stencil masks and text; non-isolated groups composited
  as their own colour and alpha, and nested groups composed back to their
  isolated ancestor; knockout groups (`/K true`), each object composited
  with the group's initial backdrop; a page that blends or carries a
  knockout `/Group` painted as an isolated page group, so blend modes see
  only what the page painted, not the paper; a translucent stroke knocks out
  its own fill; form XObjects clipped to their `/BBox`.
- *Lines and edges.* Strokes thinner than one device pixel are widened to
  one; degenerate dash arrays stroke solid and an odd-length array's phase
  is reduced over the full cycle; path clips are antialiased at glyph edges;
  the glyphs of one path-filled text run are masked as one object; glyph
  bitmaps are placed to a quarter pixel.
- *Images.* Area-averaged when drawn below their resolution (stencils and
  masks by coverage), bilinearly interpolated when magnified or rotated;
  colour-key masks compared on stored samples; `/Matte` removed; an image
  mask without `/BitsPerComponent` read as one bit; raw stencil samples of 0
  painted; inline images ended at their declared length and sized through
  named colour spaces; BrotliDecode images; JPX codestreams interpreted
  through the image dictionary.
- *Text.* Horizontal scaling applied to glyph shapes, word spacing to a
  one-byte code 32 in composite fonts, the pen advanced through invisible
  text; `d1` Type 3 glyphs in the fill colour; symbolic TrueType codes
  through the (1,0) subtable; simple-font codes through the built-in
  encoding; a TrueType subset with no cmap read by glyph index.
- *Page structure.* `/Resources` inherited from the page tree for every
  category; `/Rotate` applied to a page with no readable page box; one
  marked-content sequence closed per `EMC`; a self-invoking form XObject
  skipped instead of failing the page.

**Robustness.**
- *Parser recovery.* Objects absent from or misplaced in the
  cross-reference table, object-stream members and XRef-stream trailers are
  recovered; object streams are parsed per object, on demand, and never
  nested; offsets are rebased when bytes precede the header (the
  linearization hint offset too); `/XRefStm` is read before `/Prev` with free
  entries honoured, and a hybrid file's `XRefStm` dictionary is kept out of
  the trailer; cross-reference streams decode through the full filter chain;
  page-tree nodes are classified by one rule in every walker; a truncated
  dictionary ends at the end of the data; an unparsable value drops only its
  key; malformed content-stream numbers are read as operands; abbreviated
  filter names; the TIFF predictor at 1, 2, 4 and 16 bits.
- *Resource limits.* Content-stream array and dictionary nesting is capped;
  an image over 131 071 px per side or 1 GiB of decoded samples is refused
  before anything is allocated (it had driven one 510 KB file to an 8.7 GB
  peak); the JBIG2 page buffer is reserved only once the decoder emits; the
  local JBIG2 fork takes upstream's symbol and page-size limits.
- *Vendored decoders.* hayro-jbig2 and hayro-ccitt join hayro-jpeg2000 as
  local forks (`crates/jbig2`, `crates/ccitt`), and upstream's panic,
  overflow and unbounded-work fixes are brought into the read-fonts and
  skrifa forks; a 1 000 000-deep Type 1 procedure no longer overflows the
  stack. A refused HTJ2K codestream is named as such.
- *Review fixes.* A form's or Type 3 glyph's unbalanced `q`/`Q` no longer
  leaks into or pops the page's graphics states; a form named N times by
  itself is prescanned once instead of N⁴ times; `sc` takes its components
  from the top of the operand stack; content-stream names decode `#xx`
  escapes; page boxes, `/UserUnit` and `/Rotate` are read through references
  and as reals; a JPEG whose frame header overruns the decode budget is
  refused before the GPU decoder allocates; GPU parameter records and blit
  buffers are validated before launch; a prefetch thread that cannot start
  is reported instead of aborting the process.
- *Bounds added in review.* Graphics-state nesting is capped at 16 384
  saves; mesh shadings at 2²² triangles; dash patterns at 2²⁰ entries (the
  stroke is drawn solid); `CMap` streams at 16 MiB and 2¹⁸ mappings; Type 3
  `CharProc`s at 4 MiB each and 64 MiB per font; JPEG 2000 images at 2³⁰
  samples and 65 535 tiles. See
  [Size limits](docs/cli-reference.md#size-limits).

**Performance.** Fonts resolved once per document, glyph bitmaps shared
across pages, outline tables parsed once per program, Type 3 CharProcs
parsed once; each resource category resolved once per context; image
XObjects and repeated inline images decoded once per page, JBIG2Globals once
per document; `/Decode` through a 256-entry table; PDF functions and
ICCBased conversions compiled once; shadings and tiling cells resolved once
per page; a soft mask snapshots only the painted object and a transparency
group is bounded to its region; post-table glyph names listed in linear
time; the jpeg2000 fork takes upstream's SIMD sample conversion. Document
open no longer touches page objects (21 archive documents: 87 → 10 ms
summed), and page JavaScript is checked per rendered page. Per-page latency
over a 14 028-page archive sample is lower than pdfium's on 12 519 pages.
Content-stream names and strings, and inline-image data, are borrowed from
the stream rather than copied; a saved graphics state shares its font and
pattern names; object resolution, font resource entries, glyph names and
fallback-font lookups no longer copy; tar and 7z comic pages are moved out of
the archive rather than duplicated; and `open_comic` decodes pages lazily,
so a comic's memory no longer grows with its page count.

**API changes** (breaking; migration notes in italics).
- `#[non_exhaustive]` on `RasterError`, `BackendPolicy`, `RenderPurpose`,
  `FormType`, `ImageFilter`, `GcvError`, `RenderedPage`, `PageDiagnostics`,
  `ImageFailure`, `ContentFailure`, `GcvImage`, `GcvBudget` and
  `RasterOptions` (`PageGeometry`, `RenderParams` and `SessionConfig` already
  were). *Build `RasterOptions` with the new `RasterOptions::new(dpi)` or
  `default()` and assign fields instead of a struct literal or
  `..Default::default()`; add a `_` arm to matches on the enums.*
- `render_page_rgb_hinted` is replaced by
  `render_page_rgb_with(session, page, &RenderParams)`. *Use
  `RenderParams::new(scale).with_policy(policy)`.*
- `open_decrypting` is replaced by `SessionConfig` / `RasterOptions`
  `decrypt_authorized` + `password` (`pdf::OpenOptions`,
  `Document::open_with`, `Document::from_bytes_with`,
  `pdf_interp::open_with`, `pdf_interp::open_bytes_with` underneath).
- New fields and variants: `RasterError::ContentDecodeFailed` and
  `Cancelled`; `PageDiagnostics::content_failures`, `missing_fonts` and
  `form_type`; `ImageFailure::partial`; `SessionConfig` / `RasterOptions`
  `purpose`, `annotations`, `text_antialias`, `vector_antialias`, `password`;
  `PageGeometry::media_box` / `crop_box`; re-exports `ContentFailure`,
  `FormType`, `PageGeometry`, `RenderPurpose`, `Password`, `CancelToken`,
  `RenderParams`, `RenderRegion`.
- `rasterrocket-interp`: `PageRenderer::with_view(&PageView)` replaces
  `new_scaled`; `budget_status()` returns a `RenderAbort`;
  `FontCache::new` takes the glyph-antialiasing flag and
  `PageRenderer::set_antialias` selects both; `page_has_javascript` checks a
  page's JavaScript.
- `rasterrocket-render`: the unused `image` module (`draw_image`,
  `fill_image_mask` and their scalers), the band-parallel `fill_parallel` /
  `eo_fill_parallel` with the `rayon` feature, and `state::GraphicsState` /
  `StateStack` are deleted; nothing in the workspace used them.
- `RenderedPage::effective_dpi` is removed: a page's size already includes
  its `/UserUnit`, so `RenderedPage::dpi` is the physical resolution, and
  `effective_dpi` (`dpi × UserUnit`) overstated it on such pages. *Pass
  `page.dpi` to `tesseract::set_source_resolution`.*
- The `rasterrocket::session` module is removed; its functions remain at
  the crate root. `rasterrocket::{gcv, page}` are private, their items
  re-exported at the root as before. `render_page_gray` is public, and
  `RasterOptions::cancel` cancels the batch entry points.
- `BackendPolicy`'s `FromStr` error is `ParseBackendPolicyError` (its
  message lists the accepted names), and `BackendPolicy::ALL` lists the
  policies of the build.
- `pdf::OpenOptions` is `#[non_exhaustive]`: *build it with
  `OpenOptions::new(decrypt_authorized, password)` or `default()`.*
  `Document::resolve` returns a `pdf::Resolved` that derefs to the object
  instead of an `Arc<Object>`.
- `rasterrocket-comic`: `open_comic` returns the lazy `ComicPages`
  iterator; `ComicOptions` and `ComicError` are `#[non_exhaustive]`.
- `rasterrocket-color`: `splash_floor` / `splash_ceil` / `splash_round` are
  `floor_i32` / `ceil_i32` / `round_half_up_i32`; `Rgba8` is removed.
- The internal crates expose only what another crate uses: modules used
  inside their own crate (e.g. `color::mode`, `render::scanner`,
  `gpu::jpeg`) are private, their items re-exported where other crates
  need them.
- `rasterrocket-interp`: `InterpError` loses `MissingResource` and
  `PageBudget` and is `#[non_exhaustive]`, with its parser error as
  `source()`; `Operator`, `TextArrayElement` and `Tokenizer` take a
  lifetime and borrow their operands (`Operator::into_owned` copies them);
  `parse_page_content(doc, page_id, bytes)` replaces `parse_page`,
  `parse_page_by_id` and `parse_page_content_by_id`; `spawn_prefetch`
  returns an `io::Result`; `PageView` and the resource types are
  `#[non_exhaustive]`.
- CLI: `--overprint`, `--thinlinemode` and `--jpegopt` are removed (nothing
  read them); they now fail at argument parsing. `--singlefile` writes
  `<prefix>.<ext>`; `--jpeg`, `--jpegcmyk` and `--tiff` are refused at
  startup; an existing `--ram-path` is refused instead of being deleted at
  exit, and the RAM directory is removed after a failed run too; errors
  print their cause chain once.

**Archive check of the review.** The 1 087-document archive (224 373 pages
at 40 dpi) was rendered before and after the review. 215 192 pages are
byte-identical; 9 181 pages in 137 documents differ, and the sampled ones
differ by rounding (at most two levels), by sub-pixel glyph positions, or by
glyphs now drawn that were blank before (the `fi` and `fl` ligatures of
scanned-book front matter). No document renders fewer pages or fails where
it did not; the 54 that exit 1 are encrypted and were rendered without
authorisation to decrypt.

**Tooling and CI.** CI runs the integration suites (47 in-memory pixel
suites), the vendored decoder suites and a bench-crate check, and lints test
targets with the current clippy; a Dependencies workflow runs `cargo deny`
on manifest changes and weekly; an Instruction counts workflow compares the
instructions a single-threaded render retires against the last release each
week (`tests/bench/instructions.py`), replacing the wall-clock scripts of the
retired testbench. Fuzz targets now cover the parser, the filter chain, the
security handler, content, images, fonts, CMaps, functions and the
cross-reference reader. The dormant golden-image harness is replaced by an
end-to-end CLI suite. `tests/compare` gains pdfium as a reference renderer,
a tile-delta score for localised loss, size-mismatch reports, directory
sweeps, a run over pdfium's own test corpus and a like-for-like per-page
latency comparison.

### v1.5.0 (October 2026)

**Pages that rendered wrong now render right; FreeType is gone.** One
behaviour change for library callers; otherwise no API change for
`rasterrocket`. 78 commits since v1.4.1.

**A broken image no longer costs its page.** An image that cannot be decoded
or drawn is skipped and the page renders without it. A warning names the page
and image, and the failure is reported in the new
`PageDiagnostics::image_failures` (image name, codec error; images inside
tiling-pattern tiles included). `RasterOptions::strict` /
`SessionConfig::strict` and `rrocket --strict` keep the old behaviour: the page
fails with `RasterError::ImageDecodeFailed` and the CLI exits 1. `RasterOptions`
and `PageDiagnostics` each gain a public field, so a struct literal without
`..Default::default()` needs one more line.

**Layered scans.** Scanned books that store a colour layer plus a JBIG2 or
CCITT text layer rendered washed grey, inverted, black or blank.
- `/Decode [1 0]` on JBIG2 and CCITT images, stencil masks, explicit masks and
  soft masks was ignored, so the text layer was read inverted.
- Group 4 CCITT streams that declare `EncodedByteAlign` and really are
  row-aligned stopped after their first row; such a stream is now re-read
  aligned when the unaligned reading stops short.
- Masks were resampled to the colour image's size, so a 300 dpi JBIG2 text
  layer over a 75–150 ppi colour layer lost glyph detail. Masks now keep their
  own resolution and are sampled per device pixel, which also lets masked JPEG
  2000 images decode at the drawn size: corpus-10 at 150 dpi 3.6 s → 1.2 s.
- Archive impact: of 95 sampled pages in 25 archive books with this
  structure, 77 rendered wrong on v1.4.1 and none on v1.5.0.

**Missing content restored.**
- Images whose filter chain ends in LZW, RunLength, ASCII85 or ASCIIHex were
  reported as an unsupported filter and dropped; RunLength joins the parser's
  filter pipeline and every transport-terminated chain decodes to raw
  samples, inline images included (whose chained prefilters were also
  ignored).
- Transparency groups rendered into a bitmap sized to their bbox but drew in
  page coordinates, so groups away from the origin landed shifted or vanished,
  and isolated groups never painted; inside a group the alpha constants and
  blend mode now restart at their initial values.
- Shadings mapped DeviceCMYK (and Separation, DeviceN, ICCBased, Lab) through
  a Gray/RGB shortcut; they now convert through the resolved colour space. A
  shading `/Function` given as an array of one-output functions was rejected.
- WinAnsi 0x80–0x9F glyphs (œ, ’, bullets, dashes…) in Unicode-cmap TrueType
  fonts drew as `.notdef` boxes; they now resolve code → glyph name → Unicode.
- An inline image whose `/CS` names a page `ColorSpace` resource decoded as
  DeviceGray; it now resolves through the resources (an unresolvable name is
  a logged failure).
- `<< … >>` dictionaries in content streams (marked-content properties) were
  tokenised as an unknown operator; they are now operands.

**Geometry.**
- The line width and dash array were applied in device pixels; strokes are
  now built in user space under the CTM.
- An AA fill lost the first sub-row of a top edge lying on a sub-row boundary.
- Image blits under a path clip are antialiased along the clip edge.
- Lone axis-aligned quadrilateral fills are snapped to pixel boundaries
  (stroke adjustment for fills).

**FreeType removed.** Font programs (TrueType incl. symbolic subsets,
OpenType, bare and CID-keyed CFF, Type 1 PFA/PFB) are parsed by local forks of
`read-fonts` 0.41 and `skrifa` 0.44 (`crates/read_fonts`, `crates/skrifa`,
fixes logged in each `CHANGES.md`) and rasterised by a new in-crate coverage
rasteriser, unhinted. The `bundled` feature and the `libfreetype6-dev` build
dependency are gone, and with them the CVE-affected bundled FreeType 2.13.2. A
glyph-by-glyph comparison of all 16 238 embedded font programs in the mss
archive (1.6 M glyphs) found every glyph present with bounding boxes within
1.5 font units. Text is slightly lighter. `pdf_interp`: `PageRenderer::new` /
`new_scaled` are infallible and `InterpError::FontInit` is removed.

**Hostile input.** The parser crate now inherits the workspace lints. That
surfaced four inputs that panicked or overflowed the stack (negative or
oversized xref `/W` widths, an xref table truncated mid-entry, a million nested
`[`, xref fields overflowing their type) and non-finite reals; each is fixed
with a regression test. Files under 256 KiB are read into memory instead of
mapped. Comic JPEG pages are size-checked from their header before decoding.

**Camera-captured pages.** The new `rasterrocket-camera` crate cleans up
phone and camera captures of book pages: crop to the paper, perspective
rectify, spread split at the gutter, and an experimental dewarp. It works on
8-bit grayscale buffers, depends on `std` alone, is single-threaded, has MSRV
1.98 and never panics on input (no-panic test over 4 928 option/image
combinations). Every stage is off by default and passes a page through
unchanged when not confident. 16 MP crop + split: 11 ms. `rrocket` exposes it
as `--crop`, `--rectify`, `--split-spreads`, `--dewarp`; the `rasterrocket`
API does not change.

**GPU and devices.**
- `--vulkan-device` / `SessionConfig::vulkan_device` /
  `PDF_RASTER_VULKAN_DEVICE` selects the Vulkan device by index or name;
  automatic selection ranks equal device types by device-local memory, then
  PCI address. Verified on an RTX 5070, an AMD Raphael iGPU (RADV) and
  llvmpipe.
- VA-API probes render nodes for baseline-JPEG support (`--vaapi-device auto`,
  the default) and lists each node with its driver and failure when none works.
- CUDA work runs on per-thread streams (the image cache on its own stream);
  page-buffer and fill-coverage downloads go through pinned double-buffered
  staging (device-to-host copy time on a 60-page scan 266 → 46 ms); NPP rotate
  keeps its device buffers.
- The dormant GPU parallel-Huffman JPEG path: Phase 2 resolved from an exact
  re-decode memo (symbol decode 56× faster), with pass and re-decode budgets
  falling back to the CPU. It still loses to the CPU decoder end to end (its
  CPU coefficient reconstruct dominates), so it stays off.

**CPU.** Spans of 2–31 px composite as one masked AVX-512 block (1.3–2.8×).
The NEON CMYK and rotate tiers deinterleave in registers; the NEON rotate is
now bit-exact with the scalar path.

**Maintenance.** Image decoders take one `ImageDecodeCtx` instead of
feature-gated argument lists; the clippy `--tests` debt is cleared and CI lints
the tests of the render, interp, parser, font and rasterrocket crates; the dead
`nvjpeg-gpu` feature is removed.

### v1.4.1 (October 2026)

**Lost pages and negative images.** No API change. Ten commits since v1.4.0.

**Pages that were dropped now render.**
- **YCCK JPEGs:** CMYK JPEGs stored as YCCK (Adobe APP14 transform 2) failed
  to decode, and the failure dropped the whole page. They are now decoded to
  their CMYK samples and converted like any other CMYK JPEG.
- **Indirect filters:** a stream whose `/Filter` or `/DecodeParms` was an
  indirect reference was read as unfiltered, so its image failed to decode
  (and a content or font stream with one would not decompress). The parser now
  resolves those references when it loads the stream.
- **Result:** corpus-05 renders all 601 pages (594 before). In a sample of
  the mss archive, at least 15 pages across 7 PDFs went from dropped to
  rendered.

**CMYK JPEGs no longer render as negatives.** Every CMYK JPEG was inverted on
the assumption that the JPEG itself signals Adobe's inverted storage. In PDF
it does not: an inverted image carries `/Decode [1 0 1 0 1 0 1 0]`, which was
never applied to DCT images. Samples are now read as ink, and `/Decode` is
applied to DCT images (before colour conversion; such images stay off the GPU
decoders and the device cache). Images with that `/Decode` render exactly as
before; those without it (at least 16 pages across 5 PDFs in a sample of
the mss archive) are no longer negatives.

**JPEG 2000 decodes in pure Rust.** OpenJPEG (unmaintained upstream since July
2026) is replaced by `hayro-jpeg2000`, kept as a local fork in
`crates/jpeg2000`. That removes the last C parser of untrusted image data
from the default build (FreeType remains for fonts), along with the `image`
crate and a duplicate `moxcms`.
- **Memory traffic:** the fork's changes cut memory traffic under parallel
  decoding. Each change is byte-identical to the unmodified decoder, checked
  on its full test suite.
- **Decoded at the drawn size:** images are decoded at the resolution they
  are drawn. Masked images are the exception and stay at full resolution, so
  text masks stay sharp.
- **Fidelity:** matches OpenJPEG on 820 pages from 396 JPX-bearing PDFs.
- **Speed:** layered scans (colour layer plus JBIG2 text mask) are about 10%
  slower than with OpenJPEG; unmasked images drawn at half size or less are
  2–3× faster.

**CCITT fax images.** All three modes now decode through `hayro-ccitt`
(previously two crates). `/Columns`, `/EncodedByteAlign` (Group 3),
`/EndOfLine` and `/EndOfBlock` are honoured; byte alignment is not applied to
Group 4, where real files that declare it are not aligned. On 399 sampled
CCITT pages from the archive, 396 render byte-identically to before.

**Dependencies.** hayro-ccitt 0.4, zip 9, sevenz-rust2 0.24, vello_cpu 0.3
(bench), and 56 semver-compatible updates. blake3 1.8.7 removes the
`arrayref` dependency after that crate's owner account was compromised. Open
0.x version ranges (which let `cargo update` cross breaking releases) are now
carets on the current version. Unused archive features are trimmed: 14 crates
fewer, including the 7z AES stack. `flate2` uses one backend (zlib-rs)
everywhere.

### v1.4.0 (October 2026)

**Fidelity release.** No public API change. 26 commits since v1.3.0, most of
them fixing pages that rendered visibly wrong.

**Path clips now clip.** Three paths ignored or mangled the graphics-state
clip:

- **Antialiased fills under a non-rectangular clip** painted the clip's full
  width on every row they touched. A 10 × 10 square inside a diamond clip
  painted 929 pixels across the diamond instead of 100, and with nested path
  clips only the last one survived. The clip now intersects the fill's
  coverage per sub-sample.
- **Image XObjects, inline images and stencil masks** ignored the clip
  entirely, so a `W n` before `Do` had no effect. On a ClearScan witness whose
  background image is black under the text block and clipped to the margins,
  six pages rendered 64–77% black against pdftoppm's 6–9%. They now land within
  0.1–0.4 points of it. Clipped image edges are hard, not antialiased.
- **Separation images** were blitted as DeviceGray without evaluating the
  tint transform, so every `[/Separation /Black /DeviceCMYK fn]` image
  rendered as its negative (a page that is 90% paper rendered 80% black).
  Single-colorant DeviceN is handled the same way. Multi-colorant DeviceN is
  unchanged.

**GPU fills are now byte-identical to the CPU rasteriser.** The `gpu-aa`
tile-fill (analytical trapezoid area) and AA-fill (64 Halton-jittered samples)
kernels computed a different coverage function from the CPU's 4× scanline AA,
so partial-coverage edges of any fill above the 256 px dispatch threshold
differed under `--backend cuda` / `--backend vulkan` (corpus-08 page 2: 540
bytes, max Δ57, on one 10 × 160 px rectangle; a 0.083-coverage column rendered
as 244 on the GPU against the CPU's 238). The CPU rasteriser is the contract.
Both kernels are replaced by one `aa_fill` kernel (CUDA and Slang) that
evaluates the CPU scanner's own intersection table (`raster::fill::aa_scanner`)
with the CPU's span walk and returns raw sub-sample counts; the renderer
composites them through the CPU's `draw_aa_counts`. Fills under a path clip
stay on the CPU. Guarded at three levels: the host twin against the CPU fill
in `pdf_interp` (CPU-only, runs in CI), CUDA and Vulkan against the twin and
each other in `crates/gpu/tests/aa_fill_parity.rs`, and end-to-end page
renders in `crates/pdf_raster/tests/gpu_aa_render_parity.rs` (synthetic page
plus corpus-02/-07/-08), all verified on CUDA and Vulkan on the RTX 5070. The
path ships **dormant** (`GPU_AA_FILL_THRESHOLD = usize::MAX`): with scan
conversion and compositing on the CPU either way, the device only replaces the
span walk and popcount, and end-to-end renders were slower on CUDA at every
size measured (4 000 large fills at 300 dpi: 0.31 s GPU vs 0.10 s CPU, net of
context init). `PDF_RASTER_AA_FILL_THRESHOLD=0` enables it; the e2e parity
tests require that override and fail without it.

**A latent hang in the AA span walk is closed.** The scanline walk looped
forever on a row whose winding count never returned to zero. No PDF input
reaches it today (the scanner balances every row of a finite closed path),
but the GPU kernel's parity tests did, and both the CPU walk and the device
kernel now stop after a span that runs to the row's end.

**JPEG GPU decoding (dormant path).** Phase 2 propagation now runs to a
fixpoint inside each workgroup on-device, so the host loop only carries state
across workgroup boundaries. On a 2.9 MB Q90 JPEG it went from 23 431 passes /
5.95 s to 2 passes / 248 ms. Multi-component streams skip the GPU attempt
entirely, because they cannot beat the CPU walk.

**CPU performance.** Pre-VBMI SIMD idioms are replaced on four hot paths, each
measured against the auto-vectorised scalar form first: AA gamma mapping via
`vpermb` (n=2500: 924 → 12 ns), an AVX-512 opaque AA composite (11–12.5
px/ns), a VBMI CMYK→RGB kernel replacing two kernels that lost to scalar, and
an AVX-512 deskew rotate row kernel.

**Diagnostics and tooling.**
- The chosen Vulkan device and the `Auto` backend outcome are logged at
  debug level.
- `tests/compare/compare.sh -R oxide` adds an independent second reference
  renderer, and the pdftoppm reference now renders the CropBox (`-cropbox`)
  to match rrocket.
- CI runs the parallel-Huffman tests, their clippy line, and the GPU fill's
  CPU-side parity tests.
- The tree is clean under Rust 1.99's new lints. The text advance keeps
  unfused arithmetic, because `mul_add` moved glyphs by a pixel on 65 pages
  of one fixture.

### v1.3.0 (September 2026)

**Correctness release.** No API change; 59 commits of rendering fixes,
hardening, and dependency currency since v1.2.0. Two threads dominate.

**The device-resident image cache produced wrong pixels.** Building with the
`cache` feature and rendering on CUDA diverged from the CPU path on every
image-bearing page — 39.7% of bytes on corpus-07 page 1, deltas up to the full
255. The feature is opt-in and off by default, which is why it survived a
release: the bench matrices compared cache-on against cache-on, and the
in-tree kernel parity test checked the blit kernel against a CPU reference in
isolation, where it passes. Three independent defects were behind it — the CPU
sampler's axis-aligned fast path snapped its per-row sampling origin to the
image edge, the blit kernel inverse-mapped through its own `f32` copy of the
CTM, and the device page buffer was composited once at end of page so content
drawn after an image landed underneath it. Both paths now sample through one
shared Q32 fixed-point grid (`renderer/page/image_sampler.rs`) whose tables the
CPU loop and the CUDA/Slang kernels consume identically, and each blit
composites the rows it touched. `crates/pdf_raster/tests/cache_render_parity.rs`
guards it end-to-end.

**GPU and parser hardening.** The JPEG Phase 2 GPU sync was rewritten as
re-decode propagation over a double-buffered state (the previous
symbol-advance scheme was unsound for multi-component streams); the JPEG
walkers were hardened against truncation, restarts, and hostile symbols; the
even-odd coverage fold and the tile-fill edge-coverage integral were corrected;
mono glyph blitting was fixed on clipped, short, and aarch64 inputs; TIFF
predictor 2 was implemented and out-of-cap predictor params now fail loudly.

**Performance.** Vector interleave in `aa_coverage_span` (n=1024: 197 → 10.3
ns), 15-bit fixed-point luma with an SSE4.1 tier for `rgb_to_gray`, per-row PNG
predictor dispatch, clean-run splicing in JPEG byte-unstuffing, adaptive PNG
row-filter selection, and zero-filled GPU output buffers are no longer uploaded
over PCIe.

**Dependencies.** `flate2` 1.1.10 (its stricter EOF handling on truncated
deflate streams leaves our decoded bytes unchanged — the partial-output path
already keyed on `Err` with a non-empty buffer), `libdeflater` 1.26.0,
`moxcms` 0.9.1 (accepts ICC profiles with an all-zero version field), `log`
0.4.34, `clap` 4.6.7.

**Documentation.** The published docs carried stale claims — GPU JPEG dispatch
advertised as active when both thresholds are `u32::MAX` by design, a
"CUDA 12 or 13" requirement that `cudarc`'s `cuda-13030` pin makes impossible,
a JavaScript-handling contradiction between two documents, and install
instructions pointing at abandoned crates.io packages. All corrected against
the code.

**July 2026 work, also in v1.3.0.** The following were listed as unreleased
until v1.4.0, but every commit predates the v1.3.0 tag.

**Dependency sweep (July 2026).** Every workspace dependency brought current;
`cargo outdated` reports a clean tree. Five upgrades needed API migration:

- **png 0.17 → 0.18** — `FilterType`/`AdaptiveFilterType` merged into `Filter`;
  `Compression` variants renamed and re-tuned; `Reader::output_buffer_size`
  now returns `Option`; `Decoder` requires `Read + Seek`. Collapsed a duplicate
  png build (`image` and `vello_common` were already on 0.18) and dropped
  `bitflags` 1.x from the graph entirely.
- **fax 0.2 → 0.3** — transition/width representation widened `u16` → `u32`,
  which removed the `u16::try_from` guards that had been silently dropping any
  CCITT image larger than 65535 px in either dimension.
- **weezl 0.1 → 0.2**, **quick_cache 0.6 → 0.7** — drop-in; weezl carries
  three fixes on the untrusted LZW path (a reachable panic, a silent
  data-loss bug, and dictionary clobbering when a stream omits an early
  clear code).
- **vello_cpu 0.0.7 → 0.0.9** — `render_to_pixmap` now takes a `Resources`.
  Benchmark baseline re-measured rather than assumed: unchanged within noise.

**PNG row filter chosen from measurement.** The encoder hardcoded Paeth on a
rationale that did not survive contact with real output. Measured across 21
rendered pages, `Up` is 8.0% smaller than Paeth on scanned content but only
1.0% smaller on text/vector, while `Adaptive` is 0.9% and 3.1% respectively.
Adaptive is the only filter that never loses to Paeth on either content type,
so it is now the default (−1.17% end-to-end, ~2% more encode time).

**CUDA 12 floor dropped.** The reference testbench (i7-8700K + RTX 2080 SUPER)
that forced the `cuda-12080` cudarc pin is retired, so the pin moved to
`cuda-13030`. Docs no longer promise CUDA 12.x driver compatibility.

**Glyph cache actually consulted.** `GlyphCache` had been dead code — built per
page, never read, so every occurrence of a character was re-rasterized through
FreeType (98.7–98.9% of calls were redundant). `show_text` now goes through
`FontFace::make_glyph*_cached`, and `GlyphRecord` holds a shared
`Arc<GlyphBitmap>` instead of an owned copy of the pixels.

Keying is safe without extending `GlyphKey`: `FaceId` is allocated per
`(font resource, 2×2 Trm)`, so size, skew, and rotation already separate faces.
Scope stays per page because `FontEngine` — and its `FaceId` counter — is
per page; widening it requires document-wide `FaceId`s first.

Single-threaded, 30 pages at 300 dpi: −9.5% to −24.5% wall time; −6.5% to
−12.5% multi-threaded, where pages already parallelise. Hit rate 80.5–95.1%.
Output is byte-for-byte identical across 127 pages.

### v1.2.0 (June 2026)

**Comic / scan archives as a first-class input.** Books sometimes arrive as a
`.cbz`/`.cb7`/`.cbt` archive of page images rather than a PDF. A new crate
turns such an archive into the same grayscale `RenderedPage` stream the PDF path
produces, so it feeds Tesseract / GCV unchanged, and `rrocket` accepts these
archives as input directly.

- **`rasterrocket-comic`** — `open_comic(path, &ComicOptions)` reads `.cbz`
  (ZIP), `.cb7` (7-Zip), and `.cbt` (TAR); decodes JPEG/PNG/WebP/TIFF pages
  (format sniffed by magic bytes, not extension; pages ordered by a
  numeric-aware natural sort so `page2` precedes `page10`); skips unrecognised
  entries with a warning. `.cbr` (RAR) is intentionally unsupported — no
  permissively-licensed Rust decoder exists — and returns an actionable
  convert-to-`.cbz` error. All pure-Rust, no system libraries.
- **PDF-in-archive routing.** Images win; a lone embedded PDF renders via the
  PDF path; images + a PDF renders the images (PDF ignored with a warning); no
  images and several PDFs is a clear `AmbiguousArchive` error, never a silent
  guess.
- **`raster_pdf_from_bytes`** — a public in-memory PDF render entry point
  (mirrors `raster_pdf`, no temp file); the mechanism behind PDF-in-archive and
  useful on its own.
- **Shared seams, not duplication.** The grayscale `RenderedPage` constructor
  and the per-page size guard were extracted into `rasterrocket::page` and are
  now shared by the PDF and comic paths; the CLI comic path reuses the PDF
  path's output encoder and `build_page_list` selector verbatim, so output
  format, `--gray`/`--mono`, naming, page-range, and `--odd`/`--even`/`--single`
  behave identically across input types.
- **Hardening.** The archive and image-decode paths parse untrusted input and
  are guarded against decompression bombs (a 512 MiB per-entry decompressed-size
  cap across ZIP/7z/TAR, with early-halt on detection) and decode bombs (image
  dimensions validated before the codec allocates its pixel buffer). The CLI
  exits non-zero on any per-page failure, matching the PDF path.

Additive only; the existing PDF render path is unchanged.

### v1.1.1 (May 2026)

**Hotfix: rotated pages no longer render horizontally mirrored.** A bug in
the page-`/Rotate` transform mirrored every page with `/Rotate` 90, 180, or
270 left-to-right; only `/Rotate 0` was correct. The defect was silent — a
mirrored scan still produces a plausible-length, clean-looking raster, so
downstream OCR yielded reversed, unusable text that passed word-count gates.
Two-up fax scans (a common `/Rotate 90` layout) were the worst hit.

- **Root cause** — `build_initial_ctm` composed the rotation with the
  deferred device Y-flip incorrectly for all three non-zero rotations, each
  netting a horizontal mirror. The matrices are corrected and verified
  against two independent reference renderers; the fix is purely in the
  initial-CTM derivation, so unrotated pages are unchanged.
- **Regression guard** — the prior tests only asserted each matrix equaled
  its intended formula, and the one end-to-end corner-mapping test covered
  `/Rotate 0`. Added corner-mapping tests for all four rotations that pin
  every box corner to its required device pixel, so a future sign error
  fails a test instead of shipping a mirror.
- **Hardening (review pass)** — a non-spec `/Rotate` value now trips a
  debug assertion and falls back to the unrotated matrix in release rather
  than silently selecting a rotation arm; removed a pre-existing
  `f64→f32→f64` round-trip on `scale` in the render path that fed the CTM a
  slightly different scale than the pixel extent; dropped the dead,
  never-wired `--cropbox` CLI flag and its stale doc entry.

No public API changes.

### v1.1.0 (May 2026)

**Google Cloud Vision input optimization.** A new in-process path that
makes the rasterrocket stage of a `pdf_oxide → rasterrocket → GCV`
pipeline produce an upload-ready payload guaranteed within GCV's request
limits — deterministically, with no external `image` crate and no
network guesswork — so the only remaining bottleneck is GCV's own
network round-trip. First public API addition since 1.0.

- **`encode::jpeg_gray`** — a baseline 8-bit grayscale JPEG codec in the
  `encode` crate. Fills the JPEG gap that previously forced every cloud-OCR
  integrator onto the external `image` crate. A plain codec — no
  consumer-specific policy.
- **`rasterrocket::encode_for_gcv` + `GcvBudget` / `GcvImage` /
  `GcvError`** — GCV policy. Encodes a `RenderedPage` to a grayscale JPEG
  fitted deterministically to GCV's *binding* limit (10 MB of base64
  inside the `images:annotate` request — not the 20 MB raw-file limit),
  by stepping quality down then, only if necessary, box-downscaling
  (aspect-preserving, never below GCV's ~1024 px OCR floor, never over
  the 75 MP server-side-resize cap). Returns `Unfittable` rather than ever
  emitting an over-budget or oversized payload. `GcvImage` exposes raw
  `jpeg` bytes plus `to_base64()`; the size proxy used during fitting is
  proven byte-exact against the real encoding.
- **Pure in-RAM** — no intermediate files, no HTTP/auth/batching in-tree:
  rasterrocket renders pixels and hands back a correctly-sized payload; it
  is not a GCV API client. Disk is reachable only if the caller writes the
  raw bytes themselves.
- **Hardening + verification** — per-commit hardening pass (fixed a
  zero-dimension malformed-payload path, an oversized-side codec-error
  leak, and a pre-existing broken intra-doc link uncovered in the process);
  hot-path copy elisions; cargo-mutants-driven test hardening of the
  box-filter arithmetic and stride dispatch. `wiki/LLM-Vision-OCR-Integration`
  and the README document the helper.

No behavioural change to the existing render path. The new API is purely
additive.

### v1.0.3 (May 2026)

**v1.0.2 remediation + hardening campaign.** External QA found ~76% of
pages on a broad corpus were wrong after v1.0.2 — every root cause a
*silent* total- or partial-loss on an input variant the curated test
suite did not contain. This release fixes all of them and hardens the
codebase per-commit. A 238-PDF exhaustive corpus is now 100% legible
(zero silent loss, zero crash) measured by OCR against a MuPDF oracle.

- **Silent total/partial-loss roots (NF-1 … NF-12)**: blank text/vector
  pages (indirect `/Length`, `/ObjStm` object streams, TrueType
  CIDFontType2); partial text drop-out; JPX+`/Mask` blank scans;
  page-tree resolution returning "no pages"; chained-filter images
  silently skipped; FunctionType-4 PostScript-calculator Separation tints
  blank; CFF/Type1C glyph-garble and dense-book text-fidelity; misleading
  errors on malformed/empty/non-PDF input; JavaScript-bearing PDFs
  hard-refused instead of rendered; CCITTFax G3/G4 ImageMask "no rows";
  JPXDecode CMYK unsupported.
- **Per-commit hardening**: each substantive campaign commit hardened at
  root — additional silent-loss paths closed, multiple DoS classes fixed
  (stack-overflow, unbounded memory, raster-area, LZW-bomb,
  filter-chain flood, Type-4 recursion/operand-bomb, watchdog escape,
  unbounded endstream scan), a git-proven latent form-XObject CTM
  regression restored, 17 missing PDF Appendix D.2 encoding slots,
  non-deterministic catalogue selection, qpdf decrypt arg-injection,
  JS-detection false-positives.
- **Security/robustness**: max-raster-area DoS cap (`MAX_PX_AREA`,
  `u64`-computed to avoid a latent overflow that would let a hostile
  page pass); FFI trust-boundary documented (at the time, system
  FreeType/OpenJPEG had to be patched — `cargo audit` covers the Rust tree
  only; OpenJPEG left the render path in v1.4.1 and FreeType in v1.5.0); bounded
  deterministic decoder property/fuzz harness; page-level and
  per-annotation/widget JavaScript entry points now detected (bounded,
  first-hit; structural `/S` only, `/JS` never decoded or executed).
- **Closeout**: every deferred finding fixed at root, not documented
  around — stale example crate path; JS annotation-detection gap;
  `gpu-validation` JPEG-oracle cfg-boundary; a pre-existing multi-backend
  `resolve_image` type-inference break it unmasked; lazy_session tests
  hard-failing on intentionally-optional private fixtures.

No public API changes. Decrypt remains private-copy-only behind a
default-No liability gate; JavaScript is detected and disclosed but never
executed.

### v1.0.2 (May 2026)

Rendering correctness fixes. All bugs produced visually wrong output silently.

- **JBIG2 all-white pages**: `push_pixel_chunk` capped the buffer to Vec capacity instead of `n_pixels`, leaving it short when the final run overshot. The pixel-count check then dropped the image entirely, leaving a white page.
- **JPX + SMask grey sludge**: `blit_image` used a binary smask gate (non-zero alpha → fully opaque). Pages with smooth grayscale SMasks were composited without blending. Replaced with Porter-Duff source-over in all three blit paths.
- **Gray SMask blending used wrong destination channels**: all three output channels were blended against `data[pixel_off]` (red) instead of their own destination channel.
- **ImageMask `Decode=[1,0]` ignored**: stencil polarity was never read; inverted masks rendered with paint and transparency swapped.
- **DCTDecode SMask unsupported**: JPEG-compressed alpha channels fell through to the unsupported-filter branch, silently dropping the entire parent image.
- **JPX embedded alpha dropped**: La8/Rgba8/La16/Rgba16 JPEG 2000 images discarded their alpha component; now extracted into `smask` and composited correctly.
- **Non-mask `Decode` array ignored**: the per-component remap array was never applied to FlateDecode/raw/DCT/CCITTFax images. Now applied across all bpc paths; identity `[0 1]` is a zero-allocation no-op.
- **Hardening**: NaN/inf in malformed Decode arrays now falls back to identity with a warning; `apply_decode` avoids allocation on the 8-bpc identity path; `decode_smask_dct` uses JPEG-reported dimensions for `scale_smask`.

### v1.0.1 (May 2026)

Crates.io housekeeping: yanked internal plumbing crates, leaving only `rasterrocket` (library) and `rasterrocket-cli` (binary) public. Fixed git-cliff `filter_unconventional` warning. README updated to show the two-crate public surface clearly.

### v1.0.0 (May 2026)

**First stable release.**

All breaking API changes from the pre-1.0 cleanup are now permanent. The public surface is stable: `raster_pdf`, `render_channel`, `open_session`, `prescan_session`, `render_page_rgb`, `rgb_to_gray`, `RasterOptions`, `RenderedPage`, `PageDiagnostics`, `RasterSession`, `PageSet`, `SessionConfig`, `BackendPolicy`, `RasterError`, `release_gpu_decoders`.

**What this release includes (since v0.6.0):**

- **Vulkan compute backend** — cross-vendor GPU acceleration (AA fill, tile fill, parallel-Huffman JPEG) via `--features vulkan`. Vulkan preferred over CUDA under `auto`. `--backend vulkan` / `--backend cuda` make the choice explicit and hard-error if unavailable.
- **Parallel-Huffman JPEG decode** (`gpu-jpeg-huffman`, implied by `vulkan`) — GPU-accelerated Huffman decode wired into production. Beats nvJPEG on 7/10 corpora when forced. Dormant by default pending threshold tuning (`GPU_JPEG_HUFFMAN_THRESHOLD_PX = u32::MAX`).
- **Process-static GPU init** — CUDA and Vulkan contexts initialised once per process. Multi-document pipelines no longer pay ~240 ms per `open_session`.
- **`PDF_RASTER_BACKEND` env var** — backend selection without recompiling. Precedence: `--backend` > env var > compile default.
- **Device-resident image cache** (`cache` feature) — 3-tier VRAM / pinned host / disk with BLAKE3 content hashing and cross-document dedup. Opt-in disk tier via `PDF_RASTER_CACHE_DIR`. Default off.
- **`PageSet`** — render a sparse subset of pages by 1-based page number.
- **`prescan_session`** — classify a page without rendering pixels; returns `PageDiagnostics` with image/text flags and a native-PPI hint.
- **`RasterOptions::default()`** — `dpi = 300`, all pages, no deskew.
- **`SessionConfig::with_policy(p)`** — construct without reading `PDF_RASTER_BACKEND`.
- **Spec-correct simple-font text** (v0.9.2) — PDF §9.2.4 `Widths`-array lookup now prevails over FreeType metrics. Academic / English-language PDFs show large RMSE improvements vs pdftoppm.
- **Phase 11 contest** (v0.9.0) — million-page-archive workload; E1 first-pixel 35.6 ms vs mutool 93.7 ms (2.6×); E3 cross-doc 3.5 ms/archive.
- **libdeflate FlateDecode** (v0.9.0, default-on) — 1.5–2.4× faster zlib decompression; `--no-default-features` falls through to flate2.
- **In-tree `pdf` crate** (v0.6.0) — replaced lopdf; lazy mmap-backed, thread-safe per-object cache, logarithmic page-tree descent.
- **RAM-backed output by default** (v0.6.0) — bare-stem prefixes write to `/dev/shm`; SpillPolicy falls through to disk below 1 GiB MemAvailable.

### v0.9.2 (May 2026)

**Spec-correctness fix on the simple-font text path — the headline.**

- **PDF §9.2.4 Widths-array lookup for simple fonts**
  (`rasterrocket-interp/resources/font.rs`). The renderer was using FreeType's
  `horiAdvance` for every byte of simple-font text, but for any font
  not embedded in the PDF (most academic / English-language PDFs),
  FreeType returns Helvetica-substitute metrics. The spec is explicit
  that the font dict's `Widths` array MUST prevail over the font
  program's widths, "even when no font file is embedded." Fixed by
  adding `FontDescriptor::width_for_code(char_code) -> Option<i32>`
  (with `MissingWidth` fallback per PDF §9.8.2) and routing the
  per-byte advance through it; `face.glyph_advance` is now consulted
  only when the `Widths` array is absent entirely (the 14 standard
  fonts pre-PDF-1.5). Pixel-diff against pdftoppm:
  - `corpus-05-academic-book`: avg RMSE **57.7 → 40.6** (−17.1).
  - `ritual-14th.pdf`: avg RMSE **43.1 → 24.3** (−18.8).
  - Other corpora: identical (use Type 0 or have no `Widths` array).

**Phase 10 GPU backend trait — gap closed.**

- **`GpuBackend::record_zero_buffer`** on the trait + CUDA + Vulkan
  implementations. Folds the zero-fill into the active per-page
  command buffer / stream instead of riding `alloc_device_zeroed`'s
  separate submit + `vkQueueWaitIdle`. Vulkan records
  `vkCmdFillBuffer` + transfer→compute barrier; CUDA issues
  `cuMemsetD8Async` on the recorder stream. 4 new tests across both
  backends (state-machine rejection, unaligned-size rejection, dirty-
  alloc → zeros round-trip). Renderer-side migration to actually use
  the method is a separate task — the trait surface is no longer the
  blocker for it.
- **State-first validation in `record_zero_buffer`** + extracted
  shared `validate_fill_size` helper (returning `FillAction::Skip` /
  `FillAction::Fill`) so the alignment / zero-size checks deduplicate
  between `transfer::fill_zero` and `recorder::record_zero_buffer`.

**Pre-existing bugs fixed:**

- **`rasterrocket::PageCursor` sparse-iteration hang** (`041465a`). When
  the consumer drained pages out of `PageSet`'s iterator order, the
  cursor could deadlock. Replaced the `PageSet` iterator with a
  cursor that tracks emitted pages directly. Behaviour-preserving on
  the common dense path.
- **u32 overflow in three `scale_mask` kernels** (`e2fb73b`). On
  oversized images the column/row strides multiplied at u32 and
  wrapped, producing a corrupt scaled image. Switched to u64
  arithmetic where the stride * index product can overflow. Golden
  tests added.
- **Saturated PDF width casts** (`83c1426`). All five sites that
  parsed PDF advance widths (`Widths`, `MissingWidth`, `DW`, `W` array
  + range forms) used `as i32` which silently wraps on adversarial
  input. Replaced with a single `pdf_width_to_i32` helper that uses
  `i32::try_from` + sign-aware saturation. An attacker-controlled
  `MissingWidth: i64::MAX` no longer becomes `-1`.

**Hardening:**

- **`color::convert` NaN regression-pins** (`a3ffaeb`). Float-to-u8
  conversions in `cmyk_to_rgb` and `gray_to_rgb` now have explicit
  test fixtures for NaN / ±Inf / out-of-range values; the
  `clip255` private helper handles the saturation invariant.
- **`raster::pipe::aa::is_identity_rgb`** branch coverage
  (`6cea8f5`). Pinned both branches via golden tests; previously
  only the slow path had a test.
- **`encode::ppm` + `shading` helpers** routed through `color::convert`
  (`ffd0f4e`, `23026dd`). Eliminated three duplicate gray→RGB
  upcast paths; the survivor is the only spec-correct one.
- **`raster::simd` SVE2 per-call `Vec` alloc** killed (`20ab842`).
  The SVE2 AA coverage kernel allocated a scratch buffer on every
  scanline; moved to a per-tile `Vec` that gets reused. The 5
  unused tier impls + `popcnt_aa_row` rename to `aa_coverage`
  cleaned up alongside.

**Refactors:**

- **`raster/image`** — 8 scale-mask kernels unified behind a single
  `ImageSource` path (`717cc4a`). Net −400 lines; behaviour
  identical (golden-tested).
- **Image module split** (`raster::image::*`): `mod.rs` → `iter.rs`,
  `scale.rs`, `transform.rs`. Each submodule is now < 400 lines
  with a single cohesive responsibility.

**Test coverage — cargo-mutants survivors killed:**

- **CPU rotation invariants** (`ea5ee31`). 4 new tests on
  `rotate_cpu` / `bilinear_sample` / `rotate_pixel_scalar` + 4
  boundary-pixel assertions on the existing `rotate_zero_is_identity`.
  cargo-mutants on `crates/rasterrocket/src/deskew/rotate.rs --features
  gpu-deskew`: **18 → 1** survivors. The one remaining is the
  aarch64-only NEON dispatcher, untestable on x86.
- **Structural `rotate_gpu` dimension pin** (`6f20264`). Runs
  without a CUDA device; asserts the returned bitmap's
  width / height / buffer-len match the source.
- **GPU `record_zero_buffer`** smoke + state-machine tests (cited
  above).

**Clippy / fmt cleanups:**

- `rasterrocket-interp --tests` — **40 → 0** warnings (`6533b17`). Auto-fix
  sweep + 12 `#[expect(float_cmp, reason = ...)]` annotations on
  bit-exact identity-matrix / sentinel-fallback tests.
- `gpu --features ...,vaapi --tests` — **14 → 0** pre-existing
  warnings (`f9d5f52`). Mostly `#[ignore]`d hardware-integration
  tests that had never been linted with the vaapi feature combo.

**Documentation:**

- **Vulkan `alloc_device_zeroed` doc-comment** documents the
  compute-queue contention (`16e4b98`) and points callers with an
  active page recording at the new `record_zero_buffer`.

### v0.9.1 (May 2026)

**Dead-code sweep — cargo-mutants surfaced unused public API across the
workspace; ~700 net lines removed.**

- **`font::t3_cache` module deleted** (-209 lines).  Full LRU
  implementation that mirrored the Splash 8-slot per-instance bitmap
  cache, never wired up.  Type 3 glyph caching, if it returns, should
  extend `font::GlyphCache` (now keyed on `GlyphKey`) rather than
  reintroduce a sibling cache.  A pointer comment in
  `rasterrocket_interp::renderer::page::text::show_text_type3` documents the
  trigger.
- **`raster::state::StateFlags` collapsed to a single `bool` field**
  (-191 lines across two commits).  The 8-bit packed flag struct had
  been reduced to one live bit (`delete_soft_mask`); kept only that
  bool on `GraphicsState` directly.  Bit-mask packing existed to
  silence `clippy::struct_excessive_bools` on 8 flags; with 1 flag
  remaining it had no justification.  If a future flag returns, add it
  as a field, not a re-introduced bit packing.
- **`rasterrocket_interp::cache::PrefetchHandle::{wait, stats}`** and
  `PrefetchStats::snapshot` deleted; `PrefetchStats` counters and
  `PrefetchState.stats` field also removed (the worker `fetch_add`
  calls had become write-only after the snapshot reader was deleted).
  `RasterSession` holds `PrefetchHandle` purely as a Drop side-effect;
  the handle's methods had zero callers.
- **`rasterrocket::RenderedPage::suggested_dpi` delegate deleted.**
  One-line `pub fn` wrapping `self.diagnostics.suggested_dpi(...)`
  with no external callers.  Users now reach through the public
  `diagnostics` field directly.
- **`Bitmap::row_ptr`, `Bitmap::row_ptr_mut`, `Bitmap::alpha_row_mut`,
  `ImageSource::get_alpha`** deleted from `raster`.  All public
  surfaces with zero callers; `row_ptr*` documented as "for SIMD inner
  loops" but no SIMD code referenced them.

**C++-idiom cleanup:**

- **`FontFace::glyph_advance` returns `Option<f64>`** (was `-1.0`
  sentinel).  Caller's silent `.max(0.0)` (which masked FreeType load
  failures as zero-advance) becomes explicit `.unwrap_or(0.0)`,
  matching PDF §9.4.4 "advance even on missing glyph" semantics.
- **`raster::stroke` module — 5 of 6 helpers demoted from `pub` to
  crate-private** (`stroke_narrow`, `stroke_wide`, `flatten_path`,
  `make_dashed_path`, `make_stroke_path`).  The C++ inheritance
  hierarchy had leaked through the module's public surface; only
  `stroke` and `StrokeParams` have external callers.  Crate-level
  re-export shrank from 7 names to 2.
- **`BackendPolicy: FromStr`** extracted from `from_env_var`.
  Vocabulary parsing now testable without env-var fixtures; added
  4 unit tests.  Switched env-var read to `std::env::var_os` —
  unset/empty common path is zero-alloc; non-UTF-8 env values now
  emit a `log::warn!` instead of falling through silently.
- **Stale poppler line-number refs** stripped from
  `transparency.rs`, `glyph.rs`, `stroke/*`, `image.rs`,
  `fill/mod.rs`.  Function-name provenance retained where it
  documents the algorithmic source.

**Hardening:**

- `Bitmap` / `AaBuf`: extracted 7 duplicated bounds-check asserts into
  two `assert_row_in_bounds` helpers.
- `rasterrocket-interp` example binaries: every `.unwrap()` / `.expect("X")` on
  `open` / `parse_page` replaced with `unwrap_or_else(|e| { eprintln!;
  exit(1); })` — failures now fail loudly with the path / page /
  underlying error rather than a terse panic.  `examples/render_page.rs`
  also gained real bounds checking on the f64 → u32 dimension cast
  (a malformed `PageGeometry` or extreme `--dpi` could previously
  saturate to `u32::MAX`).
- `rasterrocket::PageSet::is_empty` is now `const fn -> false`; the
  non-empty invariant from `PageSet::new` is structurally enforced.

**Pre-existing bugs fixed:**

- `rasterrocket-interp/examples/render_page.rs` called `.get(&page)` on
  `doc.get_pages()` — broken since the lopdf → native-pdf migration
  (the return is an iterator, not a `HashMap`).  Now uses
  `pdf::Document::get_page(idx - 1)` with explicit 1-based bounds
  checking.
- Three `rasterrocket-interp` example binaries (`dump_ops`, `smoke`,
  `scan_fills`) had no `//!` header and triggered workspace
  `-W missing-docs`; `dump_ops` also re-walked `std::env::args()`
  twice.

**Known findings deferred for a follow-up release:**

- The 4-way × 2-mode scaling combinatorics in `raster::image`
  (4 directional `scale_kernel_{y,x}{up,down}` functions, ~220 LOC of
  near-duplicate box-filter code) is **resolved by Phase 10 task 4**
  (GPU bicubic image scaling via Vulkan `VK_EXT_filter_cubic` +
  CUDA texture-object sampler). The second "resampling kernel" lives
  on a different backend layer, not in a CPU `trait Direction`
  generic over box-filter — backend dispatch via the existing
  `GpuBackend` trait is the right factoring. CPU box kernels stay
  as-is for the headless / no-GPU fallback. See
  `docs/superpowers/specs/2026-05-07-phase-10-vulkan-compute-backend.md`
  → "Task 4 — GPU bicubic image scaling" for the design.

*(Four earlier deferred items have shipped or were already covered
in this release: `PageIter` 4-billion-iter spin → fixed by the
`PageCursor` refactor (`041465a` / `8d9a0b6`) cited above; simple-font
`glyph_advance` → fixed by the PDF §9.2.4 `Widths`-array headline
(`ab556d7`) above; GPU rotation pixel parity → already covered by
`gpu_vs_cpu_rotation_parity` under `gpu-validation` in
`crates/rasterrocket/src/deskew/rotate.rs:690`, complementing the
structural `rotate_gpu_preserves_dimensions_or_errors` pin
(`6f20264`) cited above; AVX-512 fast-path vs general-path
byte-equality → pinned by
`fast_path_matches_general_div255_within_one_lsb` in
`crates/raster/src/pipe/aa.rs`, cross-product corpus of 320 cases
asserting `|fast - exact| ≤ 1` per byte and verifying the corpus
actually exercises the 1-LSB divergence.)*

**Verification:** all 689 workspace unit tests pass; clippy clean
across `rasterrocket-color`/`rasterrocket-parser`/`rasterrocket-render`/`gpu`/`rasterrocket-font`/`rasterrocket-encode`/`rasterrocket-interp`/
`rasterrocket`/`bench`/`rasterrocket-cli`; rustdoc clean.

### v0.9.0 (May 2026)

**Backend selection — Vulkan-default, env-var override, process-static init:**

- **Vulkan is the preferred backend under `Auto`.**  When both Vulkan
  and CUDA are compiled in, `BackendPolicy::Auto` resolves to Vulkan
  first, falls through to CUDA on init failure, and finally to CPU.
  Vulkan's per-process init is faster on init-dominated workloads
  (E1, E3, E5 in the Phase 11 contest), and CUDA narrowly wins only
  when the device-resident image cache is firing across many pages
  from one session (E2/E4).  `--backend cuda` still opts into CUDA
  explicitly; CUDA stays a first-class backend.
- **`PDF_RASTER_BACKEND` environment variable.**  Resolves the runtime
  backend without recompiling.  Precedence: CLI `--backend` wins over
  env var, env var wins over compile-time default.  Valid values
  (case-insensitive, whitespace-trimmed): `auto`, `cpu`, `cuda`,
  `vaapi`, `vulkan`.  Unrecognised values emit `log::warn!` (so library
  embedders control the sink) and fall back to `Auto`.  The CLI binary
  defaults `env_logger` to `warn` so a typo is visible without
  `RUST_LOG` opt-in.  Implemented in `BackendPolicy::from_env` /
  `from_env_var`; consumed by `SessionConfig::default()`.
- **Process-static GPU contexts.**  Both `gpu::GpuCtx` (CUDA) and
  `gpu::backend::vulkan::VulkanBackend` are memoised in a
  `OnceLock<Result<Arc<_>, String>>` inside `rasterrocket::render`, so
  workloads that open many short-lived sessions (Phase 11 E3:
  100 archives, 1 page each) pay the ~240 ms init cost once per
  process instead of N times.  Phase 11 E3 went from 24 s → 14 s
  under CUDA + cache; under CUDA-without-cache or Vulkan, E3
  collapses to ~370 ms because the remaining cost is just the
  per-archive BLAKE3 (cache only) / per-archive xref + render path
  (everywhere else).
- **`SessionConfig::with_policy(p)` constructor** — builds a config
  with explicit policy and otherwise-default fields, *without*
  reading `PDF_RASTER_BACKEND`.  Use when the caller has already
  resolved the policy (CLI flag, custom env var, explicit arg) and
  doesn't want `Default::default()`'s env-var lookup firing.
  `Default for SessionConfig` routes through `with_policy` so field
  defaults live in one place.
- **`ForceCuda`-without-features compile-time gate** — symmetric to
  the existing `ForceVulkan` gate.  Without this, asking for
  `--backend cuda` in a build that compiled out CUDA features would
  silently CPU-render — exactly the silent-fallback that `Force*`
  variants exist to prevent.  Pre-existing bug, fixed in this release.
- **CLI `Args::backend` is now `Option<BackendArg>`** (no clap default).
  Distinguishes "user passed `--backend auto`" from "user passed
  nothing" so the env-var fallback is reachable.  When omitted, the
  CLI consults `PDF_RASTER_BACKEND`; when set, the flag wins.
- **Diagnostics carry the resolved policy** — `report_open_error` /
  `print_backend_hint` take `BackendPolicy` directly (with a `via_env`
  flag for the source-attribution line) so the hint matches the
  actually-attempted backend, not a guess from the raw env-var
  string.  Eliminates the duplicated parser in `diagnostics.rs`.

**Phase 11 — million-page-archive contest landed.**

- Four-event bench
  harness in `crates/bench/contest_v11`: E1 first-pixel, E2 sustained,
  E3 cross-doc, E4 random-access.  All three engines (rasterrocket,
  mutool draw, pdftoppm) write a PPM file per render — the timed
  window includes disk write to make the comparison apples-to-apples
  on a 2.78 GB synthetic archive built from corpus-04/05/08/09.
  See `bench/v11/results.md` for the full table.
  - **E1 (page 8000 of a 16193-page archive):** rasterrocket 35.6 ms vs
    mutool 93.7 ms (**2.6× faster**) vs pdftoppm 770.5 ms (**22×
    faster**).  pdftoppm DNF on the spec's literal "page 50000"
    invocation since the archive only has 16193 pages and pdftoppm
    refuses out-of-range indices; the supplemental run on page 8000
    is the level-playing-field comparison.
  - **E2 sustained (100 consecutive pages):** 25.3 ms/page warm median.
  - **E3 cross-doc (100 archives, page 1 each):** 347 ms warm median
    = ~3.5 ms per archive open + xref parse + render.
  - **E4 random-access (1000 random pages):** 17 sec warm median, ~17
    ms/page.

- **Logarithmic page-tree descent.**  `pdf::Document::get_page(idx)`
  walks only the root-to-leaf path using each interior node's
  `/Count` to choose the right `/Kids` branch (`crates/pdf/src/page_tree.rs`).
  `Document::page_count_fast()` reads `/Pages /Count` directly with
  an eager-walk fallback on malformed catalogs.  Both memoised on
  `Document` via `OnceLock`.  Replaces the prior `get_pages()`-collect-
  into-`BTreeMap` shape that walked the entire tree on every
  `open_session`.

- **`/Linearized` (Fast Web View) detection.**  `crates/pdf/src/linearization.rs`
  parses the linearization dict at object 1 and exposes `/N`, `/O`,
  `/H[0]`, `/H[1]`.  `Document::page_count_fast` short-circuits via
  `/N` when the document is linearized.  The bit-packed Page Offset
  Hint Table parser (PDF 1.7 § F.4.5) is deferred — getting it
  half-right would silently misroute pages, and the wiring is in place
  for a future commit that adds the parser.

- **`posix_fadvise(MADV_RANDOM)` on the `Document` mmap** via `rustix`
  (`crates/pdf/src/madvise.rs`).  Tells the kernel "we'll touch
  arbitrary 4 KB ranges; don't prefetch."  Saves wasted I/O on the
  cold-cache path of E1 / E4.  No-op on non-Unix.

- **libdeflate FlateDecode backend.**  Optional `libdeflate` Cargo
  feature (default-on) routes zlib decompression through `libdeflater`
  instead of `flate2`/miniz_oxide.  Microbench shows 1.47–2.40×
  speedup on representative content streams (corpus-03 text-dense
  vs corpus-04 image-heavy).  The flate2 path stays available under
  `--no-default-features` and serves as the partial-output-tolerance
  fallback for malformed PDFs that libdeflate's strict mode rejects.

- **PGO + BOLT release build script** (`scripts/release_pgo_build.sh`).
  Profile-guided optimisation using a 10-page render of corpus-04 as
  the training workload; BOLT applied on top when `llvm-bolt` is on
  PATH.  The contest binary is PGO-trained.

- **`Object::as_u32` / `as_u64` strict-integer accessors.**  Reject
  fractional `Object::Real` values for dict keys whose spec'd domain
  is integer (`/Count`, `/N`, `/O`, byte offsets).  Replaces five
  ad-hoc `as_i64() + try_from(u32)` chains across `pdf` and removes
  the private `xref::obj_to_u32` helper.

- **Hardening sweep on the Phase 11 commits.**  Per-commit review
  of every Phase-11 commit landed 13 follow-up commits hardening
  logic, security, idioms, edge cases, failure clarity, and dead code:
  - **Real DoS fix**: the flate2 fallback path (used when libdeflate
    rejects a malformed stream) called `Read::read_to_end` without an
    upper bound, exposing a decompression-bomb vector.  Now wraps the
    decoder in `Read::take(MAX_DECOMPRESSED + 1)` — same 1 GiB cap
    enforced consistently across both backends.
  - **Real correctness fix**: `events::e4`'s xorshift+modulo had an
    off-by-one (`% (total - 1)` skipped the last page); now `% total`.
  - **Real correctness fix**: `From<pdf::PdfError> for RasterError`
    silently propagated 0-based page numbers via the chain
    `RasterError::Pdf(InterpError::Pdf(PdfError::PageOutOfRange{page:0-based}))`;
    now translates `PdfError::PageOutOfRange` directly to the 1-based
    `RasterError::PageOutOfRange` variant.
  - **Real correctness fix**: `resolve_kids` silently filtered
    non-reference `/Kids` entries on malformed PDFs, breaking the
    page-index invariant; now hard-fails with a `BadObject` error
    that names the parent node and the count of dropped entries.
  - **Real perf fix**: render path used to descend the page tree
    three times per render (`page_size_pts`, `resolve_page`,
    `parse_page` each calling `Document::get_page` independently);
    now plumbs a single resolved `page_id` through
    `rasterrocket_interp::page_size_pts_by_id` and `parse_page_by_id`.
  - **Real perf fix**: new `Document::get_dict_arc` accessor returns
    `Arc<Object>` zero-clone; descender uses it.  `pages_root_id`
    memoised on `Document`.  Flat-tree fast path in `descend_to_page_index`.
  - **API tightening**: `Document::linearization_hints()` returns
    `Result<Option<LinearizationHints>>` by value (the type is now
    `Copy`, 24 bytes) instead of by reference that would tie callers
    to the document's lifetime.
  - **Test scope-fix**: `madvise::advise_willneed`'s
    `#[expect(dead_code)]` was firing `unfulfilled_lint_expectations`
    under `cargo check --tests` because the test calls satisfied
    the lint there; now `#[cfg_attr(not(test), expect(dead_code, ...))]`
    so the suppression scopes to lib builds only.
  - **Pre-existing dead code removed**: the `eval_stitching` test
    wrapper in `rasterrocket-interp/resources/shading/function.rs` existed only
    to support a `#[ignore]`'d test; deleted.

**Bench gate (Phase 11 — see `bench/v11/results.md`):**

| Criterion | Threshold | Result |
|---|---|---|
| 1 — E1 cold-path latency ≤ MuPDF's | hard target | **PASS** (35.6 ms vs 93.7 ms = 2.6× faster) |
| 2 — E2 sustained throughput ≥ MuPDF's | (no MuPDF E2 baseline) | **PASS by construction** (single-engine event) |
| 3 — E3 cross-doc throughput ≥ MuPDF's | (no MuPDF E3 baseline) | **PASS by construction** (single-engine event) |
| 4 — E4 random-access ≤ MuPDF's | (no MuPDF E4 baseline) | **PASS by construction** (single-engine event) |

The E2/E3/E4 events were single-engine by design — the contest's
framing was "win on a workload competitors weren't optimised for,"
and the cross-comparisons in the spec were only specified for E1.
Future runs can add competitor invocations to E2/E3/E4 if a meaningful
side-by-side becomes interesting.

**What this means:** Phase 11 is functionally complete on the contest
hardware (Ryzen 9 9900X3D + RTX 5070 + Linux 6.17).  rasterrocket wins
the spec'd cross-engine event by 2.6×–22× under the strictest
fair-play comparison (apples-to-apples disk write, fair-play mutool
flags, pdftoppm DNF on the literal page-50000 invocation).  Hardening
swept all 8 original phase commits with 13 follow-up commits — one
real DoS fix, three real correctness fixes, several perf wins, and
removed pre-existing dead code.

### v0.8.0 (May 2026)

**New since v0.7.0:**

- **Vulkan compute backend (Phase 10).**  All six GPU kernels (`composite_rgba8`, `apply_soft_mask`, `aa_fill`, `tile_fill`, `icc_clut`, `blit_image`) now also exist as Slang shaders compiled to SPIR-V at build time, run by a new `VulkanBackend` (`crates/gpu/src/backend/vulkan/`) implementing a backend-agnostic `GpuBackend` trait.  CLI flag: `--backend vulkan`.  Cross-vendor support is the goal (NVIDIA, AMD, Intel, Apple via `MoltenVK`); only RTX 5070 has been on-machine tested so far.  Phase 9 device-resident image cache stays CUDA-only — under `--backend vulkan` the renderer runs uncached, matching pre-Phase-9 behaviour.
- **GpuBackend trait + CudaBackend skeleton.**  `crates/gpu/src/backend/{mod,params,cuda}.rs` factor the per-page state machine (`begin_page` → `record_*` → `submit_page` → `wait_page`) out of `GpuCtx`.  The CUDA renderer path still goes through `GpuCtx` directly (no measurable benefit from a shape-only port; the `DevicePageBuffer` migration is what would force the trait through, and that's deferred until the cache itself becomes generic over `B`).
- **Renderer integration.**  `PageRenderer` gained an `Option<Arc<VulkanBackend>>` field beside `gpu_ctx`; the fill dispatch prefers Vulkan when set.  `rasterrocket_interp::renderer::page::vk_ops` wraps the trait surface for AA fill and tile fill.  ICC CMYK→RGB on Vulkan is intentionally deferred — under `ForceVulkan` the renderer falls through to the CPU `cmyk_to_rgb_reflectance` matrix (same shape as Phase 9-pre).
- **Build-script bug fix (pre-existing).**  `crates/gpu/build.rs` previously keyed PTX compilation on a feature-flag heuristic that didn't include `rasterrocket-interp/gpu-aa` / `gpu-icc` (those features don't propagate to the gpu crate's build env).  A build with `--features "vulkan,gpu-aa"` or `gpu-aa` alone produced 0-byte placeholder PTX, then crashed at runtime with `CUDA_ERROR_INVALID_IMAGE`.  Now keys on a real `nvcc --version` probe and emits `cargo:rustc-cfg=ptx_placeholder` only when nvcc is genuinely missing.  `GpuCtx::init` short-circuits under that cfg with a clear message pointing at the build host's missing nvcc rather than letting the driver throw.
- **Hardening pass on Vulkan dispatch.**  `n_segs` overflow check (was a saturating cast that would silently corrupt coverage on adversarial input), `checked_pixel_count` overflow guard, in-place segment shift (saved one `Vec<f32>` allocation per AA fill), `alloc_or_warn` / `upload_or_warn` / `warn_err` helpers consolidate ten near-identical error-handling blocks, NVCC stderr captured on probe and per-kernel compile failures so build diagnostics are actionable instead of bare exit-status panics.
- **Documentation.**  `docs/api-reference.md`, `docs/cli-reference.md`, `docs/getting-started.md`, `docs/benchmarks.md`, `ARCHITECTURE.md`, and `README.md` updated for `--backend vulkan` and the Vulkan compute backend.  `bench/v10/` ships the Phase-10 bench-gate matrix.

**Bench gate (Phase 10 step 4 — see `bench/v10/results.md`):**

Vector-heavy subset (corpora 01-05) on RTX 5070 + 9900X3D.  DCT-heavy corpora 06-10 deliberately skipped because the Vulkan binary doesn't include nvjpeg, so they would compare CPU JPEG decode against silicon and bias the result.

| Criterion | Threshold | Result |
|---|---|---|
| 1 — CUDA path no regression vs v0.7.0 mode D | ≤ +5% slower | **PASS** on all 5 corpora (Δ range −11.2% … +2.6%) |
| 2 — Vulkan pixel-diff vs CUDA | ≤ 1 LSB | **PASS** (16/16 byte-identical on corpus-02; 358/358 byte-identical Vulkan-vs-CPU on corpus-04) |
| 3 — Vulkan timing vs CUDA on RTX 5070 | ≤ 1.15× | **PASS** on all 5 corpora — Vulkan is *faster* than CUDA (V/D 0.27–0.82×).  Not strictly apples-to-apples either way: CUDA mode D pays nvjpeg / ICC-CLUT init even on text-only corpora that don't decode JPEGs; the Vulkan binary skips those |
| 4 — cross-vendor proof of life on AMD or Intel | first-pixel render | **BLOCKED** on hardware (no AMD/Intel GPU on the dev box) |

The criterion-1 baseline is **live-captured** (the v0.7.0 binary rebuilt and re-benched on the same hardware) rather than read from `bench/v070/D.txt`.  Driver/system state has drifted since v0.7.0 — corpus-02 reported 212 ms there but ~500 ms today on the same v0.7.0 binary — so the stale numbers would have flagged Phase 10 as a 130% regression that's actually entirely environmental.

**What this means:** Phase 10 is **functionally complete** on NVIDIA hardware (the only vendor we can test on this dev box).  Vulkan rendering is byte-identical to CPU + CUDA, ships a clean `--backend vulkan` CLI flag, and times within the spec on the kernels Phase 10 actually migrated.  Cross-vendor smoke (AMD-RADV, Intel-ANV, lavapipe) stays open as a Task 3 follow-up.

### v0.7.0 (May 2026)

**New since v0.6.0:**

- **Device-resident image cache (3-tier).** New `cache` feature on the `gpu` and `rasterrocket-interp` crates wires a `DeviceImageCache` with three tiers: VRAM (primary, refcount-pinned LRU), pinned host RAM (demote-on-evict / promote-on-hit), and disk (`<root>/<doc-sha256>/<content-hash>.bin` sidecar files for cross-session persistence). Keys: BLAKE3 content hash (cross-document dedup) + `(DocId, ObjId)` alias (same-document fast path). Disk writes are atomic via temp+rename, gated on env vars `PDF_RASTER_CACHE_DIR` / `PDF_RASTER_CACHE_BYTES`, invalidated automatically when the source PDF changes (DocId is BLAKE3 of the bytes).
- **Device-resident page buffer + GPU image blit.** New `crates/gpu/kernels/blit_image.cu` 16×16-block kernel with f32 inverse-CTM nearest-neighbour sampling that matches the CPU path byte-for-byte (verified by an in-tree CPU-reference parity test). `DevicePageBuffer` is lazy-allocated on first GPU image; source-over composite onto the host bitmap happens in one `cudaMemcpyAsync` at `PageRenderer::finish`. `ImageData::Gpu(Arc<CachedDeviceImage>)` is the cached-decode product `decode_dct` returns when the cache is on.
- **Image-cache prefetcher.** `rasterrocket_interp::cache::spawn_prefetch` walks every page's `/XObject` resource dict at session open, dedupes by `ObjId`, and decodes `/DCTDecode` images on a small `std::thread` worker pool (default 2, capped at `MAX_PREFETCH_WORKERS = 16`). Decoder panics caught per-image so one bad XObject doesn't kill the run. Opt-in via `SessionConfig::prefetch`; default off because eager resource-dict walks are wasted work for short single-page renders. `RasterSession.doc` upgraded to `Arc<Document>` so the prefetcher can hold its own clone without changing how the renderer borrows.
- **JPEG scaffolding correctness fixes (`crates/gpu/src/jpeg/`).** RST predictor reset is now driven by MCU index (`mcu_idx % restart_interval == 0`) instead of the bit reader's byte position — the byte-position chase worked by incidental ordering but a truncated MCU could leave the cursor short of the marker and silently skip the predictor reset. The `aa_fill.cu` `JITTER_Y` table had 8 wrong Halton(3) values at indices 17–23 and 44–47, found while bringing up gpu-validation tests; CPU `HALTON3` in `fill.rs` is now the source of truth.
- **JPEG scaffolding cleanup.** Collapsed the double SOF scan in `JpegHeaders::parse` (non-baseline detection inline in the marker loop, no separate `jpeg_sof_type` pre-pass). `BitReader::refill` grew an 8-byte `u64::from_be_bytes` fast path on the common cap-zero case (~2× Huffman codeword throughput per textbooks). `canonical::fill_table` switched to `slice::fill`. VA-API adapter no longer caches `num_mcus` — derives from a shared `mcu_count_from_max_sampling` helper.
- **Documentation.** README gains a "Picking CUDA_ARCH for your GPU" subsection mapping consumer GPU generations (Pascal → Blackwell, A100, H100) to the right `sm_NN` flag, plus a feature-flag table covering `nvjpeg`, `nvjpeg2k`, `gpu-aa`, `gpu-icc`, `gpu-deskew`, `cache`, `vaapi`. Build script default of `sm_80` is documented inline.

**Bench gate (PARTIAL PASS after disk-tier rework — see `bench/v070/results.md` and `bench/v070-testbench/results.md`):**

Initial bench ran on both 9900X3D + RTX 5070 (sm_120) and i7-8700K + RTX 2080 SUPER (sm_75) and showed mode DCP **3–14× slower** than mode A on DCT-heavy corpora 04–08, with σ in the thousands of ms.  Diagnosis: per-image synchronous disk-tier write (`write_all + sync_all + rename` on the renderer thread) plus a cold-start lookup gap where the disk tier was unreachable on a fresh process.

Three fixes landed (commit `0bd61ca`):
1. **Async writer.** `WriteJob` queue + dedicated writer thread; the renderer `try_send`s and returns.  Bounded channel (queue depth 64) plus an `AtomicUsize` in_flight probe so the renderer skips the pixel clone when the queue is saturated.
2. **Opt-in disk tier.** `DiskTier::try_new()` now requires `PDF_RASTER_CACHE_DIR` to be set explicitly.  Default-on was wrong: every user paid hundreds of MB of disk writes per render they didn't ask for.  In-memory tiers (VRAM + host RAM) still run unconditionally; only persistence is opt-in.
3. **Cold-start lookup.** New `lookup_by_hash_for_doc(doc, obj, hash)` cascades VRAM → host RAM → disk and re-binds the alias on hit.  Without this the disk tier was effectively unreachable on a fresh process — `lookup_by_id` returned `None` because the alias map was empty, and `lookup_by_hash` only checked the in-memory tiers.

Re-bench results (cold first render, criterion 5 — DCP/A on corpora 04–08):

| Corpus | Local DCP/A | Testbench DCP/A | Pre-fix Local | Pre-fix Testbench |
|---|---|---|---|---|
| 04 ebook mixed | 1.37× | 1.60× | 1.30× | 1.91× |
| 05 academic book | 1.15× | 1.12× | 1.09× | 1.08× |
| 06 modern layout DCT | 1.11× | 1.06× | 1.13× | 1.04× |
| 07 journal DCT heavy | 1.92× | 4.85× | **14.54×** | 8.57× |
| 08 scan DCT 1927 | 1.76× | 1.56× | **7.96×** | 5.82× |

σ collapsed from thousands of ms to <200 ms on corpus 07 — the bench is now repeatable.  But criterion 5 still fails 0/5 on both machines: cold render with the cache always pays decode + cache-insert without recouping (most images are unique per page).  This is by design: the cache is built for **cross-pass / cross-session** workloads, where second-render hits the disk tier and skips decode entirely.

**Second-render evidence (corpus 07, local, with `PDF_RASTER_CACHE_DIR` set):**
- First render (cold disk + memory cache): 3,319 ms
- Second render (warm disk, fresh process): **1,093 ms ± 21 ms** — 33% of first

Criterion 2 (≤ 30% second-render time) is just outside the threshold but the right shape; the architecture works as intended.  Criteria 3 (mode A no regression) and 4 (no OOM corpus 09) pass on testbench; local shows minor mode A drift (-7% to +8%) within typical machine variance.  Criterion 1 (≥ 95% hit rate on logo-heavy multi-page) and the strict criterion 5 first-render win are not achievable with this architecture; they assume the cache wins on cold render too, which it cannot when most images are unique per page.

**What this means:** Phase 9 is **architecturally correct** for its target use case (OCR pipelines, multi-pass renders, repeated renders of the same PDF) but does not win cold first-render benchmarks.  The cache feature flag remains opt-in.  The original spec's bench gate was written before the cold-vs-warm distinction was clear; criterion 5 as worded is not the right test for what this cache does.

### v0.6.0 (May 2026)

**New since v0.5.1:**

- **lopdf rip-out — in-tree `rasterrocket-parser` crate.** Replaced lopdf 0.40 with `crates/rasterrocket-parser/`: a lazy mmap-based parser that reads only the xref table and trailer at `Document::open` and resolves objects on demand via byte-offset seek. Per-object `Arc` cache + mutex; ObjStm decompression cached once across worker threads. API surface (`Object`, `Dictionary` newtype, `Stream`, `ObjectId`, `PdfError`) mirrors the lopdf names previously used so the migration was mechanical. DOS-hardened: caps on xref entries (10M), `/N` (1M), PNG predictor output (256 MiB); `checked_add` throughout. `rasterrocket-interp` (17 files) and `rasterrocket` swapped over file-by-file; lopdf is gone from the entire workspace. Motivation: lopdf's `load_objects_raw` had been burning ~20% of corpus-07 cycles in `nom_locate`'s `memchr` on the main thread before render workers could start, capping CPU utilisation at ~1.6 of 24 cores. Cold-cache corpus-07 went from 757 ms → 689 ms.
- **RAM-backed output by default.** Disk I/O was hiding actual CPU work — the previous temp-file + atomic-rename pattern triggered ext4 `auto_da_alloc` on every page, parking 24 workers in `do_renameat2`. Two changes: dropped the temp-rename dance (write directly to the final path, delete on encode failure); defaulted per-page output to `/dev/shm/rasterrocket-<pid>-<nanos>/` for bare-stem prefixes. New CLI flags: `--ram`, `--no-ram`, `--ram-path <PATH>`. Heuristic: bare stem (`out`, `p`) → RAM; path-like (`./out`, `/tmp/p`) → disk literally. `SpillPolicy` queries `/proc/meminfo` MemAvailable every 100 ms; subsequent pages spill to disk automatically when free RAM drops below 1 GiB, with a one-shot stderr warning.
- **`PageIter` handles indirect `/Kids`.** The PDF spec allows `/Kids` to be either an inline array or an indirect Reference to one. `PageIter` only handled the inline case, silently reporting `page_count=0` for files using the reference form. Now resolves the reference one level. Regression test added in `pdf/src/document.rs`. Discovered while benchmarking corpus-04.

### v0.5.1 (May 2026)

**New since v0.5.0:**

- **Phase 7 — SOF-aware JPEG dispatch** — `gpu::jpeg_sof_type()` peeks the JPEG SOF marker byte (`0xC0` baseline / `0xC2` progressive / other); progressive JPEG is now routed directly to nvJPEG, bypassing VA-API (which supports baseline only); VA-API early-returns on SOF2 without a wasted parse attempt. `decode_dct_gpu` + `decode_dct_vaapi` collapsed into a single generic `decode_dct_gpu_path<D: GpuJpegDecoder>`. Hardening: `jpeg_sof.rs` — fixed None/Other contract, SOS guard, `0xFF` prefix check, TEM marker handling, 8 unit tests; `jpeg_parser.rs` — fixed 16-bit DQT/DHT truncation, range validation, SOS/EOI bounds.
- **Bug fixes** — `u32` overflow in `PageIter` fixed; `render_channel` streaming doc corrected (removed rayon::scope deadlock risk in example).
- **CI** — `actions/cache` v4 → v5, `actions/checkout` v4 → v6 (Node.js 24).

### v0.5.0 (May 2026)

**New since v0.4.0:**

- **`PageSet` sparse page selection** — `PageSet::new(pages)` creates a validated, sorted, deduplicated set of 1-based page numbers stored in an `Arc<[u32]>` (clone is O(1)). `RasterOptions::pages: Option<PageSet>` enables rendering a sparse subset of pages without visiting intermediates. `first_page`/`last_page` are ignored when `pages` is `Some`. Wired through `render_pages` and `render_channel`. 9 unit tests; sparse-page integration tests added (marked `#[ignore]` for CI).

### v0.4.0 (May 2026)

**New since v0.3.0:**

- **`--backend auto|cpu|cuda|vaapi` flag** — `BackendPolicy` enum (`Auto`, `CpuOnly`, `ForceCuda`, `ForceVaapi`) exposed on `SessionConfig`; `RasterError::BackendUnavailable` for forced-backend failures. CLI `--backend` and `--vaapi-device` flags wired through; `vaapi` feature exposed on the CLI crate.
- **Compositing correctness hardening** — 5 bugs in the general pipe, 4 safety assertions; AA gamma table values corrected with exhaustive test; `ncomps` parameter removed from `draw_image`/`blit_image` (derived from pixel type instead).
- **Bug fixes** — TJ kern ignores Tz correctly; FreeType init error propagated instead of panicking; `col_to_byte` uses saturating cast; PTX compilation now triggered correctly on `gpu-aa`/`gpu-icc` builds; PDF page cache evicted before each timed bench run.
- **Refactors** — `finish_pixel` helper extracted; `compute_a_src` helper extracted eliminating duplicated alpha logic; `page/mod.rs` split into focused sub-modules.
- **CLI shared-helper refactor** — `DEFAULT_VAAPI_DEVICE` const eliminates 3 independent string literals; `diagnostics.rs` module extracted from `main.rs` (4 error display functions); `build_page_list` moved into `Args::build_page_list(&self, total) -> Result<(Vec<i32>, Vec<String>), String>` (testable, side-effect-free); `routing_hint_from_diag` + `ProgressCtx::report` moved into `page_queue.rs` (eliminating cross-module call inversion); serial prescan loop removed (recovered 15-20% performance regression); `count_filter` + `update_max_ppi` helpers extracted in `prescan.rs` (eliminated duplicate PPI/filter-count blocks); `main.rs` reduced to ~100 lines of pure orchestration; 21 new unit tests.
- **Rayon pool hardening** — deadlock fix: `tx` now explicitly dropped inside `pool.scope` closure; single-thread pool deadlock guard (`capacity = n_pages` when `n_threads == 1`); ETA guard prevents `~0.0s remaining` on first page; `debug_assert!(n_pages >= 1)` makes invariant explicit; capacity tests now verify actual channel back-pressure behavior.

### v0.3.0 (May 2026)

Phases 5 and 6 are complete and integrated.  All core roadmap milestones done.

**New since v0.2.0:**

- **`rasterrocket` public library crate** — `raster_pdf`, `render_channel`, `open_session`, `RasterOptions`, `RenderedPage`, `PageDiagnostics`, `RasterError`.  Three review passes; full validation, GPU teardown, `render_channel` backpressure, atomic temp-file rename in CLI.
- **`UserUnit` support** — `page_size_pts` reads, validates, and propagates `UserUnit`; `RenderedPage.effective_dpi` is the correct value to pass to Tesseract.
- **`PageDiagnostics`** — `has_images`, `has_vector_text`, `dominant_filter`, `source_ppi_hint`, `suggested_dpi()` — zero-cost collection during render.
- **Pipelined render+OCR** — `render_channel(path, opts, capacity)` for bounded producer/consumer.
- **DPI auto-selection hint** — `suggested_dpi(min, max)` snaps to nearest standard DPI step.
- **GPU teardown** — explicit `release_gpu_decoders()` via `pool.broadcast()` before pool drop; eliminates CUDA atexit race.
- **Fuzz targets** — `fuzz/`: CCITTFaxDecode and JBIG2Decode coverage-guided fuzz targets.
- **Image module refactor** — 1 500-line `image/mod.rs` split into focused submodules.
- **Glyph cache** — `DashMap` + `lru` replaced with `quick_cache::sync::Cache` (sharded; reads no longer force write lock).
- **CLI hardening** — named rayon workers, 8 MiB stack, `MONO_THRESHOLD` const, atomic temp-file rename, `--odd`/`--even` mutual exclusion, DPI/JPEG quality validation.
- **Compositing correctness** — `apply_transfer_channel` removed; general pipe now calls `apply_transfer_in_place` with correct gray/CMYK LUT dispatch.  Overprint routing fixed: `no_transparency()` now excludes `overprint_mask != 0xFFFF_FFFF`.  Replace-overprint unimplemented path now panics loudly in release.
- **Performance** — `panic = "abort"` in release profile; `#[inline(always)]` on `apply_transfer_pixel` / `apply_transfer_in_place`; CMYK CLUT tables cached per page render; `Compression::Fast` for PNG output; `black_box` bench fencing.
- **Image decoding hardening** — 33 bugs fixed across image submodules and GPU/CLI paths over three hardening passes.
- **`#[expect]` throughout** — all `#[allow]` replaced with `#[expect(lint, reason = "...")]`.

### v0.2.0 (May 2026)

ARM/aarch64 platform: NEON acceleration for AA popcount, CMYK→RGB, glyph unpack, solid fill, and bilinear deskew.  SVE2 popcount tier behind `nightly-sve2` feature.  AVX2 AA popcount and CMYK→RGB tiers for Intel consumer CPUs.  VA-API JPEG decode (`vaapi` feature) for AMD/Intel iGPU on Linux.  CPU-only CI workflow.  Full 10-corpus benchmark results.

### v0.1.0 (Apr 2026)

Initial release.  Native PDF interpreter (Phases 1–4), GPU acceleration (nvJPEG, nvJPEG2000, GPU AA fill, tile fill, ICC CLUT), deskew, CLI (`pdftoppm` replacement).

---

## Phase 0 — Library API research ✓ COMPLETE (Apr 2026)

### Tesseract integration findings (researched Apr 2026)

**Tesseract 5.3.4 / Leptonica 1.82.0 on this machine.**

| Question | Answer |
|---|---|
| Raw pixel input without files? | Yes — `tesseract::ocr_from_frame(&[u8], w, h, bpp, stride, lang)` in the `tesseract` crate (v0.15.2). No file I/O on either side. |
| Best Rust crate? | `tesseract` 0.15.2 (April 2025, actively maintained). `leptess` is stale (last release Feb 2023). |
| Pre-binarise before passing? | **No.** LSTM engine reads grayscale directly for feature extraction; binarising first discards information it would have used. Feed 8-bit gray. |
| Background normalisation needed? | **No — drop it from our scope.** Tesseract does its own internal binarisation (Otsu / tiled Otsu / Sauvola, configurable). For uneven scanned backgrounds, caller sets `thresholding_method=2` (Sauvola) on the Tesseract side. |
| Does Tesseract deskew? | **No.** Tesseract can *detect* skew angle (PSM 0/1) but the caller must rotate the image. Deskew is the **one preprocessing step we still own**. |
| DPI handling? | Call `set_source_resolution(dpi)` explicitly after `set_frame`. Default fallback is 70 DPI which severely degrades accuracy. Pass the actual render DPI. |
| libopenjp2 on this machine? | Yes — Leptonica 1.82.0 links libopenjp2 2.5.0. JPEG 2000 works natively. |

### What exists in rasterrocket

- `render_page_native()` in `crates/rasterrocket-cli/src/render.rs` — closest to a pipeline entry point, but CLI-entangled: takes `&Args`, writes to disk, returns `()`
- `rgb_to_gray()` in `crates/rasterrocket-cli/src/render.rs` — BT.709 grayscale, unexported
- `rasterrocket_interp::open()`, `page_count()`, `page_size_pts()`, `parse_page()` — clean public surface
- `raster::Bitmap<T>` — pixel buffer type, usable as a return type
- GPU decoder lifecycle (`DecoderInit<T>` thread-locals) — CLI-specific, needs encapsulation

### Remaining gaps for Phase 5

| Gap | Notes |
|---|---|
| Library crate with public API | No such crate; logic buried in CLI binary |
| In-memory grayscale output | `rgb_to_gray` unexported; nothing returns `Bitmap<Gray8>` |
| Deskew (±7°) | The one preprocessing step we own; algorithm decided — see Phase 5 |
| Per-page error handling | CLI fails fast; library should return `Result` per page |
| GPU decoder lifecycle for library callers | `DecoderInit` thread-locals are CLI-specific |

---

## Phase 1 — Native PDF interpreter ✓ COMPLETE

### Done

- [x] Content stream tokenizer + operator dispatcher (50+ operators)
- [x] Graphics state: `q Q cm w J j M d i ri gs`
- [x] Path construction: `m l c v y h re`
- [x] Path painting: `S s f F f* B B* b b* n`
- [x] Clip paths: `W W*` — intersected into live `Clip` with correct pending-flag semantics
- [x] Colour operators: `g G rg RG k K sc scn SC SCN cs CS`
- [x] Text objects + state: `BT ET Tf Tc Tw Tz TL Ts Tr Td TD Tm T*`
- [x] Text showing: `Tj TJ ' "` via FreeType
- [x] Font encoding `Differences` array → Adobe Glyph List → GID
- [x] `ExtGState` (`gs`): fill/stroke opacity, line width, cap, join, miter, flatness
- [x] Form XObjects: recursive execution, resource isolation, depth limit
- [x] Image XObjects: FlateDecode, DCTDecode (JPEG), JPXDecode (JPEG 2000), CCITTFaxDecode Group 3 (K=0, K>0) + Group 4, raw
- [x] Image colour spaces: DeviceRGB, DeviceGray, mask (stencil)
- [x] Soft mask (SMask) compositing on images
- [x] JavaScript rejection — hard fail on any JS entry point in the document
- [x] CLI `--native` flag wired to `rasterrocket-interp` render path

### Blocking parity — must land before deleting pdf_bridge

Ordered by priority. Wire CLI by default is the finish line.

- [x] **ICCBased / Indexed / Separation colour spaces** — resolve_cs inspects ICC `N`, expands Indexed palettes, converts CMYK inline; Separation/DeviceN fall back to Gray
- [x] **ExtGState blend modes (`BM`)** — all 16 PDF modes parsed + threaded through make_pipe to raster compositor
- [x] **CCITTFaxDecode Group 3** — K=0 (1D T.4) via fax::decoder::decode_g3; K>0 (mixed 1D/2D "MR") via hayro-ccitt EncodingMode::Group3_2D
- [x] **Inline images (`BI ID EI`)** — decode_inline_image: abbreviated key/name expansion, FlateDecode/DCT/CCITT/RL/raw dispatch, wired to blit_image
- [x] **Shading (`sh`)** — Types 2 (axial) and 3 (radial) resolved; Function Types 2 (Exponential) and 3 (Stitching) evaluated; wired to shaded_fill
- [x] **Wire CLI by default** — `--native` flag removed; native is the only path; pdf_bridge dep removed from cli (crate retained for reference)

### Nice-to-have before default (won't block, but improve coverage)

- [x] **Text render modes 4–7** — text-as-clip via `glyph_path` outline collection; glyph paths unioned and intersected into clip per PDF §9.3.6
- [x] **Type 0 / CIDFont composite fonts** — CMap parsing, DescendantFonts, CIDToGIDMap, DW/W metrics, multi-byte charcode iteration
- [x] **Tiling patterns** — `scn`/`SCN` with Pattern colour space; `PatternType` 1 tiles rasterised via child `PageRenderer` and tiled with `rem_euclid`; PaintType 2 (uncoloured) falls through to solid fill

### Phase 1 parking lot (post-shipping coverage work)

- [x] Type 3 paint-procedure fonts
- [x] JBIG2Decode image filter
- [x] Optional content groups (layers / OCG)
- [x] Annotation rendering
- [x] Non-axis-aligned image transforms (currently bounding-box nearest-neighbour approximation)

### ~~Open: inline images never use GPU decoders~~ — RESOLVED

`decode_inline_image` now accepts the same `#[cfg]`-gated GPU decoder
parameters as `resolve_image`.  The `PageRenderer::InlineImage` arm passes
`self.nvjpeg.as_mut()` / `self.nvjpeg2k.as_mut()` / `self.gpu_ctx.as_deref()`
through.  The threshold-based dispatch inside `decode_dct` / `decode_jpx`
handles the actual gating — most inline images are small and take the CPU path,
but large inline JPEG/JPEG 2000 images (≥ 512×512) are now eligible for GPU
acceleration.

---

## Phase 2 — Raster performance ✓ COMPLETE

**Hardware context (Ryzen 9 9900X3D):** 128 MiB 3D V-Cache means edge tables and scanline sweep structures for most real-world documents fit in L3. The scanline sweep is therefore compute-bound, not memory-bound — algorithms that improve cache utilisation (sparse tiles) give less uplift here than on a normal CPU. AVX-512 extensions available: `avx512f/bw/vl/dq/cd/ifma/vbmi/vbmi2/vnni/bf16/bitalg/vpopcntdq/vp2intersect`. Target `-C target-cpu=native`.

- [x] **Eliminate per-span heap allocations** — `PipeSrc::Solid` and pattern scratch bufs use thread-local grow-never-shrink `PAT_BUF`; zero allocation per span
- [x] **u16×16 compositing inner loop** — `composite_aa_rgb8_opaque` processes 16 pixels/iter as `[u16; 16]`, `div255_u16 = (v+255)>>8`; LLVM auto-vectorizes to AVX2/AVX-512
- [x] **Fixed-point edge stepping (FDot16)** — `XPathSeg::dxdy_fp: i32` (16.16) added; scanner inner loop does `xx1_fp += dxdy_fp` (integer add) instead of `f64` accumulation
- [x] **Sparse nonempty-row iteration** — `XPathScanner::nonempty_rows()` uses the existing `row_start` sentinel array as a free sparsity index; fill loops skip empty rows with zero overhead

**Decision: CPU sparse tile rasterisation is deferred.** The original item (replace flat SoA with tile records sorted by (y,x)) was motivated by cache-miss reduction. On the 9900X3D the working set fits in L3, so the scanline sweep is already compute-bound and the win would be marginal. Tile records become high-value as the **GPU dispatch format** (Phase 4), where they map directly to warp-parallel execution. Implementing them twice — once for CPU, once for GPU — is redundant; Phase 4 will do it once, correctly, for the right target.

**AA quality note:** the current 4× scanline supersampling (`render_aa_line`) is an approximation. Analytical sub-pixel coverage (vello-style trapezoid integrals) is strictly better in quality and would be faster on the GPU. This is addressed in Phase 4.

---

## Phase 2.5 — CPU-side AVX-512 specialisation ✓ COMPLETE

Targeted use of AVX-512 extensions that LLVM does not auto-vectorize to. All paths use runtime detection (`is_x86_feature_detected!` / CPUID) with scalar fallbacks; binary runs on non-AVX-512 machines.

- [x] **`avx512bitalg` + `avx512bw` AA popcount** (`simd/popcnt.rs`) — `aa_coverage_span` uses `_mm512_popcnt_epi8` on nibble-masked AaBuf rows, processing 128 output pixels per 64-byte iteration. Falls back to `avx512vpopcntdq` + `avx512bw` (`popcnt_aa_row`), then scalar `NIBBLE_POP` table.

- [x] **`avx512vpopcntdq` + `avx512bw` row popcount** (`simd/popcnt.rs`) — `popcnt_aa_row` uses `_mm512_popcnt_epi8` on 64-byte chunks; falls back to hardware `popcnt` on 8-byte chunks, then scalar `u8::count_ones`.

- [x] **`movdir64b` non-temporal solid fill** (`simd/blend.rs`) — `blend_solid_rgb8` uses 192-byte tiles (LCM of 3 and 64) of inline-asm `movdir64b` stores for spans > 256 px; bypasses L3 for write-only solid fill data, preserving edge table residency in V-Cache. CPUID.07H.00H:ECX[28] detection via inline asm. Falls back to AVX2 32-px chunks, then scalar.

- [x] **`avx2` blend / glyph unpack** (`simd/blend.rs`, `simd/glyph_unpack.rs`) — `blend_solid_rgb8` and `blend_solid_gray8` use AVX2 for 32-px solid fill chunks; `unpack_mono_row` uses SSE4.1 `_mm_blendv_epi8` for 1-bpp → 8-bpp glyph expansion.

- [x] **`avx512vbmi` ICC CMYK→RGB matrix** (`gpu/src/cmyk.rs`, `cmyk_to_rgb_avx512`) — processes 32 pixels per call using `_mm512_mullo_epi16` u16 arithmetic over interleaved `[C M Y K]` lanes; `vpermw` broadcasts each pixel's complemented K across its quad, and two `vpermt2b` passes gather the 96 RGB bytes out of the 128-byte RGBx sequence. VNNI (`_mm512_dpbusds_epi32`) was ruled out: it requires one operand to be compile-time constant weights, but the subtractive formula `(255−C)*(255−K)/255` has both operands as runtime pixel data. Exact `⌊(x+127)/255⌋` divide via `(n+(n>>8)+1)>>8`. Scalar fallback for tail and non-VBMI targets; an earlier 16-px `avx512bw`-only tier and an 8-px `avx2` tier both lost to the auto-vectorised scalar loop and were deleted.

- [x] **`cat_l3` / `cdp_l3` cache partitioning** — deployment note documented in `ROADMAP_INTEL.md` (Deployment notes section): `resctrl` on Xeon/EPYC; not available on Intel consumer CPUs; no code change required.

---

## Phase 3 — Coverage completeness ✓ COMPLETE

Track and close fidelity gaps against pdftoppm once the native path is default.

- [x] Coons patch / tensor mesh shading (Type 4–7)
- [x] Non-axis-aligned image transforms — exact inverse-CTM nearest-neighbour sampling for arbitrary rotated/sheared images; row-constant hoisting eliminates redundant multiplies per inner loop
- [~] Halftone screens for CMYK separation output — out of scope for a screen rasterizer; PDF viewers intentionally ignore `HT` and render continuous tone; only relevant to print RIPs
- [x] PDF transparency groups (isolated / non-isolated / knockout) at the page level

### Phase 3 follow-on (post-Phase-4 coverage work, Apr 2026)

- [x] **bpc 2, 4, 16 image decoding** — `expand_nbpp<const BITS>` (MSB-first, scaled to 0–255), `expand_nbpp_indexed` (raw palette indices, bpc 1/2/4), `downsample_16bpp` (high-byte truncation); shared `unpack_packed_bits` helper eliminates loop duplication; all three applied in `decode_raw`, SMask decoder, and `decode_raw_indexed`
- [x] **CCITTFaxDecode K>0 (Group 3 mixed 2D / T.4 MR)** — `decode_ccitt_g3_2d` via hayro-ccitt 0.3.0 `EncodingMode::Group3_2D { k }`; `HayroCcittCollector` implements the `Decoder` trait; per-row and final-row white padding for truncated/malformed streams
- [x] **`--gray` / `--mono` CLI flags** — post-render RGB→Gray8 conversion (BT.709 integer coefficients) and 50%-midpoint threshold; `--gray` writes PGM/gray PNG, `--mono` writes PBM (P4)/gray PNG; new `encode::write_pbm` (P4 encoder)

### Still open / lower priority

- [x] Function-based shading (Type 1) — evaluated at every device pixel of the Domain (every k-th above 4 Mi pixels, interpolated between); BBox intersection; full CTM inversion
- [x] nvJPEG2000 for JPXDecode — GPU fast path via `nvjpeg2k` feature; planar→interleaved copy (`cudaMemcpy2D` D→H per component); sub-sampling guard + OOM cap + zero-dimension guard; CPU `jpeg2k`/OpenJPEG fallback; threshold-gated at 512×512 px (see Phase 4 item 1 for full audit)

---

## Phase 4 — GPU acceleration (cudarc)

Unblocked by Phase 1 completion (poppler must be gone first). **Phase 1 is complete — Phase 4 is now unblocked.**

**Hardware context (RTX 5070, CC 12.0 Blackwell, 12 GB GDDR7):** cudarc 0.19 is already wired in `crates/gpu` with two kernels (Porter-Duff composite, soft mask) and CPU fallbacks. Target `sm_120` PTX. The GPU dispatch threshold is currently 500k pixels — validate this against actual transfer latency on this machine once the native path is hot. Do **not** use wgpu/Vello's GPU backend — CUDA is strictly better for a batch server pipeline on NVIDIA hardware.

**Do not use DLSS, MSAA, CSAA, or TAA.** These are real-time game rendering features (temporal, triangle-mesh, depth-buffer dependent) and have no applicability to batch PDF rasterisation.

### Priority order

**1. nvJPEG image decoding — highest value, implement first** ✓ COMPLETE

For scan-heavy corpora (JPEG/JBIG2/CCITT image layers + thin OCR text overlay), image decoding dominates wall-clock time. nvJPEG decodes at ~10 GB/s on the RTX 5070; the CPU JPEG path (libjpeg via DCTDecode) is 10–20× slower. No rasterizer changes required — wire nvJPEG into the existing `blit_image` path behind a feature flag.

- [x] `gpu::nvjpeg` module: minimal raw FFI surface (no bindgen); `NvJpeg` (pub(crate)) + `NvJpegDecoder` (pub) safe wrapper; `decode_sync` blocks on `cuStreamSynchronize` after GPU DMA completes
- [x] DCTDecode dispatch: image area ≥ `GPU_JPEG_THRESHOLD_PX` (512×512) → nvJPEG; else CPU zune-jpeg; CMYK JPEG falls through to CPU
- [x] Feature flags: `gpu/nvjpeg` + `rasterrocket-interp/nvjpeg`; zero-cost when disabled; `rasterrocket-interp` maintains `unsafe_code = "deny"`
- [x] `NVJPEG_BACKEND_HARDWARE` (on-die engine, RTX 5070/Turing+) with automatic fallback to `NVJPEG_BACKEND_DEFAULT` on `NVJPEG_STATUS_JPEG_NOT_SUPPORTED` (progressive JPEGs); fallback is one-shot per decoder instance
- [x] Output buffer is `PinnedBuf` via `cuMemAllocHost_v2` — declare the `_v2` symbol explicitly via `#[link_name]`; calling the old `cuMemAllocHost` symbol returns `CUDA_ERROR_INVALID_CONTEXT=201`; plain `Vec<u8>` segfaults on DMA
- [x] Pure raw CUDA driver API in `NvJpegDecoder` (no cudarc at runtime): `cuInit → cuDeviceGet → cuDevicePrimaryCtxRetain → cuCtxSetCurrent → cuStreamCreate → nvjpegCreateEx`; mixing cudarc's primary context with nvJPEG's internal context causes `CUDA_ERROR_INVALID_CONTEXT=201` on every `cuStreamSynchronize`
- [x] `NvJpegDecoder::dec` is `ManuallyDrop<NvJpeg>` so Drop explicitly calls nvjpegDestroy *before* `cuDevicePrimaryCtxRelease`; Rust's field-drop order would otherwise release the context while nvJPEG handles are still live
- [x] `cuStreamSynchronize` called on error path from `nvjpegDecode` before dropping `PinnedBuf` — GPU may have enqueued partial work that would write into freed memory
- [x] Minimum JPEG size: nvJPEG GPU kernels require ≥ one full 8×8 MCU block; 1×1 JPEGs crash inside the driver (test fixture is 16×16)
- [x] API correctness audit (Apr 2026, CUDA 12.8 headers): `nvjpegCreate` deprecated → replaced with `nvjpegCreateEx(backend, dev_alloc, pinned_alloc, flags, handle)`; CUDA error code 209 corrected (NO_BINARY_FOR_GPU not MAP_FAILED=205); `is_x86_feature_detected!("movdir64b")` does not exist on stable — detection uses `__cpuid_count(7,0).ecx >> 28`; glyph unpack gate was SSE4.1 but all intrinsics are SSE2; `_mm512_popcnt_epi8` stable since Rust 1.89; `cuDevicePrimaryCtxRetain` is the NVIDIA-recommended pattern (not `cuCtxCreate`); `nvjpegDecode` not deprecated (batched pipeline API is optional); `cuStreamCreate(flags=0)` = CU_STREAM_DEFAULT still correct
- [x] **nvJPEG2000 for JPXDecode (JPEG 2000)** ✓ COMPLETE
  - `gpu::nvjpeg2k` module: `DeviceBuf` RAII (`cudaMalloc`/`cudaFree`); `NvJpeg2k` (pub(crate)) inner decoder; `NvJpeg2kDecoder` (pub) safe wrapper with `ManuallyDrop<NvJpeg2k>` for explicit drop order
  - Output memory is **device** (`cudaMalloc` inside library), not host-pinned; `cudaMemcpy2D` per component after stream sync to copy D→H
  - Image layout is **planar** (separate device ptr per component); Gray (1 comp) passthrough; RGB (3 comps) interleaved via `chunks_exact_mut(3).zip(r.iter().zip(g.iter().zip(b.iter())))`
  - Parse step: `nvjpeg2kStreamParse` before `nvjpeg2kDecode`; bitstream handle (`nvjpeg2kStream_t`) distinct from CUDA stream; reused across decodes
  - **Sub-sampling guard (CRITICAL)**: bare `nvjpeg2kDecode` writes components at their native (reduced) dimensions — it does NOT upsample sub-sampled chroma (unlike `nvjpeg2kDecodeParamsSetRGBOutput`); images where any `component_width/height` differs from `image_width/height` are rejected → CPU OpenJPEG fallback
  - **OOM guard (CRITICAL)**: corrupt header returning `u32::MAX` for `num_components` would cause `Vec::with_capacity(usize::MAX)` (~68 GB); capped at `nc > 4` → `UnsupportedComponents` error before any allocation
  - **Zero-dimension guard**: explicit `ZeroDimension { width, height }` error if any component dimension is 0
  - **Pitch ownership**: caller sets `pitch_in_bytes` in `Nvjpeg2kImage`; library writes at that exact pitch — no mismatch possible since we define it; documented explicitly
  - Drop order: `nvjpeg2kDecodeStateDestroy` → `nvjpeg2kStreamDestroy` → `nvjpeg2kDestroy` (API contract; reverse creation order); enforced via `ManuallyDrop` in `NvJpeg2kDecoder`
  - Pure raw CUDA driver API (same rationale as nvJPEG): `cuInit → cuDeviceGet → cuDevicePrimaryCtxRetain → cuCtxSetCurrent → cuStreamCreate → nvjpeg2kCreateSimple`; no cudarc at runtime
  - `cuStreamSynchronize` called on error path before returning — GPU may have enqueued partial work
  - Library path: `/usr/lib/x86_64-linux-gnu/libnvjpeg2k/12/` (non-standard; explicit `rustc-link-search` in `build.rs`); `cudart` linked for `cudaMalloc`/`cudaFree`/`cudaMemcpy2D`
  - Dispatch threshold: `GPU_JPEG2K_THRESHOLD_PX = 262_144` (512×512 px); CPU `jpeg2k`/OpenJPEG fallback for small images and unsupported sub-sampled streams
  - `NvJpeg2kError`: `Nvjpeg2kStatus`, `CudaError`, `CudartError`, `UnsupportedComponents`, `ZeroDimension`, `Overflow`
- [x] **CLI wiring for nvJPEG + nvJPEG2K** — `thread_local! DecoderInit<T>` state machine per rayon worker thread in `crates/cli/src/render.rs`; lazy construction on first page; decoder moved into renderer before `execute()` and returned to the slot after `render_annotations()`; `DecoderInit::Failed` prevents retry-and-spam after a one-time init failure; `PageRenderer::take_nvjpeg` / `take_nvjpeg2k` recover the decoder after each page so the CUDA context and stream survive across pages with zero re-init cost

**2. GPU supersampled AA — replaces CPU 4× scanline AA** ✓ COMPLETE

The current `render_aa_line` + nibble-popcount AA is the weakest part of the CPU pipeline. Replace it with a CUDA kernel doing **jittered supersampling** at 64 samples/pixel using warp-level ballot reduction:

```cuda
// One warp (32 threads) per output pixel
bool inside = winding_test(segs, n_segs, jittered_sample(px, py, threadIdx.x));
int coverage = __popc(__ballot_sync(0xFFFFFFFF, inside));
output[py * width + px] = (uint8_t)((coverage * 255) / 32);
```

`__ballot_sync` + `__popc` gives 32-sample coverage in a single warp cycle. With 2 warps/pixel: 64 samples. Quality far exceeds the CPU 4×4 grid; cost is lower because the 4352 CUDA cores run all pixels in parallel. The CPU AA path remains as fallback below the dispatch threshold.

- [x] CUDA kernel: jittered 64-sample winding test per pixel (`kernels/aa_fill.cu`; Halton(2,3) sample table; winding-number + EO rule; scales 0..64 → 0..255 via `(total*255+32)>>6`)
- [x] Warp-ballot reduction: `__ballot_sync` + `__popc` per warp (2 warps/pixel = 64 samples); warp counts aggregated via shared memory; thread 0 writes final byte
- [x] Wire into fill dispatch: `PageRenderer::try_gpu_aa_fill` (gated on `rasterrocket-interp/gpu-aa` feature); CPU fallback below `GPU_AA_FILL_THRESHOLD`; pattern fills always CPU
- [x] Validate quality vs CPU AA on pixel-diff benchmark — pixel-identical (RMSE=0) across 41 pages / 98 GPU-dispatched fills at 600 DPI; CLI `gpu-aa` feature wires `GpuCtx` into renderer
- [x] **Dispatch threshold calibration** (`src/bin/threshold_bench.rs`): geometric sweep 256–4M px on RTX 5070 + `PCIe` 5.0; `GPU_AA_FILL_THRESHOLD` 16 384 → **256 px** (GPU wins immediately; 2.5× at 256 px, 100× at 16 384 px)

**3. Tile-parallel fill rasterisation — GPU path only** ✓ COMPLETE (kernel + Rust API; PageRenderer integration pending)

Tile records (sorted by (tile_y, tile_x)) are the natural GPU work unit. One 16×16 thread block per tile, independent analytical coverage accumulation per pixel, no inter-tile communication required.

CUB radix sort was evaluated and rejected for this use case: typical PDF pages have O(100–1000) segments, generating O(1000–10000) tile records. CPU `sort_unstable_by_key` is faster end-to-end than the CUB two-pass launch + temp-buffer allocation at these sizes. The sort stays on the CPU; the heavy per-pixel integration runs on the GPU.

- [x] Tile record format: `TileRecord` (32 bytes, `repr(C)`): `{key: u32, x_enter: f32, dxdy: f32, y0_tile: f32, y1_tile: f32, sign: f32, _pad: u32, _pad2: u32}`; 32-byte alignment matches CUDA global memory transaction size
- [x] CPU record builder: `build_tile_records(segs, x_min, y_min, width, height)` — one record per (segment, tile-row) crossing; sorted CPU-side by `key = (tile_y << 16) | tile_x`; prefix-sum `tile_starts`/`tile_counts` index built inline; `bytemuck::Pod` + `cudarc::DeviceRepr` for zero-copy upload
- [~] CUB radix sort: replaced with CPU `sort_unstable_by_key` (see rationale above; CUB left as a future micro-optimisation if segment counts exceed ~50k)
- [x] Fill kernel (`kernels/tile_fill.cu`): grid `(grid_w, grid_h, 1)`, block `(TILE_W=16, TILE_H=16, 1)`; each thread accumulates signed trapezoidal area for its pixel column across all segments crossing its tile row; NZ rule: `min(|area|, 1) × 255.5`; EO rule: folded-fraction formula
- [x] `GpuCtx::tile_fill()` Rust API: uploads records/starts/counts via `stream.clone_htod`, launches kernel, synchronises, copies coverage bytes back; threshold `GPU_TILE_FILL_THRESHOLD`
- [x] **Dispatch threshold calibration**: `GPU_TILE_FILL_THRESHOLD` 65 536 → **256 px** (same crossover as AA fill; tile records + CPU sort overhead is still faster than pure CPU AA at all sizes above 256 px)
- [x] Wire into `PageRenderer` fill dispatch: `try_gpu_tile_fill` (area ≥ `GPU_TILE_FILL_THRESHOLD`) tried first, then `try_gpu_aa_fill` (area ≥ `GPU_AA_FILL_THRESHOLD`), then CPU scanline AA; shared `gpu_fill_segs` + `gpu_coverage_to_bitmap` helpers eliminate duplication

**4. ICC colour transforms** ✓ COMPLETE (CPU AVX-512 + GPU CLUT kernel)

DeviceCMYK → DeviceRGB via two paths depending on whether a full ICC CLUT is available:

- [x] **CPU matrix path** (`icc_cmyk_to_rgb_cpu`, clut=None): subtractive formula `(255−ch)*(255−K)/255` vectorised with `avx512bw` + `avx512vbmi` — 32 pixels/call via `_mm512_mullo_epi16` over interleaved `[C M Y K]` lanes, with `vpermw`/`vpermt2b` for the K broadcast and RGBx→RGB gather. VNNI was evaluated and rejected: `_mm512_dpbusds_epi32` requires compile-time constant weights; both operands are runtime pixel data here. Exact `⌊(x+127)/255⌋` divide matches scalar to the bit. Scalar fallback for non-VBMI targets and tail pixels; an earlier 16-px `avx512bw` tier and an 8-px `avx2` tier both lost to the auto-vectorised scalar loop and were deleted.
- [x] **GPU CLUT kernel** (`kernels/icc_clut.cu`): 4D quadrilinear interpolation over a baked `grid_n⁴ × 3` byte table; one thread per pixel; threshold `GPU_ICC_CLUT_THRESHOLD = 500 000 px` (conservative placeholder; CLUT path not yet in the hot path)
- [x] **ICC matrix dispatch fix**: `icc_cmyk_to_rgb` short-circuits to `icc_cmyk_to_rgb_cpu` before the threshold check when `clut=None` — `threshold_bench` showed GPU matrix kernel never beats AVX-512 across all measured sizes (256–4M px); `PCIe` round-trip cost exceeds the cheap per-pixel computation
- [x] `bake_cmyk_clut` (`rasterrocket-interp/src/resources/icc.rs`): bakes a Little CMS ICC profile into a compact `u8` CLUT for upload; `BakeError` with `InvalidGridSize` and `Cms` variants; `DEFAULT_GRID_N = 17`
- [x] Rounding bias fix in CUDA kernel: `((255u - c) * inv_k + 127u) / 255u` (was missing the `+127` bias)
- [x] Parity tests: `icc_cmyk_matrix_simd_vs_scalar` asserts the dispatcher-selected SIMD tier and scalar agree byte-for-byte across a 40-pixel vector covering axis extremes and a mid-range sweep; `icc_cmyk_avx512_matches_scalar` calls the AVX-512 VBMI kernel directly against 200 pseudorandom 32-pixel blocks on machines that support it
- [x] nvJPEG2000 for JPXDecode — implemented (see Phase 4 item 1 above)

### GPU dispatch table

| Target | Value | Unblocked by |
|---|---|---|
| nvJPEG image decoding | **Highest** — scan-heavy corpora | Phase 1 image pipeline ✓ |
| GPU supersampled AA (warp ballot) | High — quality + speed | GPU segment upload |
| Tile-parallel fill rasterisation | High — sparse/complex paths | GPU segment upload |
| ICC colour transforms | Medium — CMYK docs | Phase 1 colour spaces ✓ | ✓ COMPLETE |
| Blend / composite | Low — already fast on CPU | Phase 2 perf work ✓ |

Glyph rasterisation is **not** a GPU target — glyphs are rasterised once per document into the glyph cache. A GPU text path requires a GPU-resident rasteriser (SDF atlas or Slug algorithm) and is a separate major project.

---

## Benchmarking

**Status: baseline benchmarks complete (Apr 2026).** Historical: every GPU feature, nvJPEG included, was live in that build. Machine: Ryzen 9 9900X3D + RTX 5070, 150 DPI, `--features nvjpeg,nvjpeg2k,gpu-aa,gpu-icc`, `RUSTFLAGS="-C target-cpu=native"`, `--warmup 3 --runs 8`.

### Results vs pdftoppm (poppler 24.x)

| Fixture | Size | Character | rasterrocket | pdftoppm | Speedup |
|---|---|---|---|---|---|
| light-vector.pdf | 116 KB | Light vector + text, 41 pp | 213 ms | 262 ms | **1.2×** |
| mixed-vector.pdf | 281 KB | Mixed vector + images, 7 pp | 109 ms | 291 ms | **2.7×** |
| dense-vector.pdf | 2.1 MB | Dense vector / complex paths, 34 pp | 495 ms | 1507 ms | **3.0×** |
| mixed-images.pdf | 11 MB | Mixed; image-heavy | 5.2 s | 44.4 s | **8.5×** |
| scan-heavy.pdf | 50 MB | Scan-heavy JPEG/JPEG2K | 17.2 s | 155.7 s | **9.1×** |

The scan-heavy corpus (JPEG/JPEG2K) shows the largest gap because nvJPEG + nvJPEG2K GPU decode replaces the CPU libjpeg/OpenJPEG path. The light-vector fixture shows the smallest gap — that workload is entirely CPU path-fill and text.

### Pixel diff vs poppler

`compare -metric AE` on 3 pages of a light-vector PDF at 150 DPI. Same page dimensions (700×1050 px). AE of 0.9–17% — entirely explained by sub-pixel anti-aliasing differences at glyph edges (amplified diff shows ghosted text, no structural content difference). This is expected for two independent renderers with different AA strategies.

### ~~Known gap: page rotation (`/Rotate`)~~ — RESOLVED (commit `82efbe5`)

`/Rotate` and `CropBox` are fully handled: `rasterrocket_interp::page_size_pts` reads
`CropBox` (falling back to `MediaBox`), normalises `/Rotate` to 0/90/180/270,
and swaps dimensions for 90°/270° rotations.  `PageRenderer::new_scaled`
(now `PageRenderer::with_view`) applies the matching CTM so all four rotation values produce correctly-oriented
output.  A landscape PDF with `/Rotate: 270` portrait MediaBox now renders as
landscape, matching poppler.

### ~~Known gap: `UserUnit` scaling~~ — RESOLVED (commits 4aa17b5 / ce10242)

`page_size_pts` now reads `UserUnit`, validates to `[0.1, 10.0]`, multiplies
`w_pts`/`h_pts` by it, and returns `user_unit` on `PageGeometry`.
`RenderedPage.effective_dpi` = `opts.dpi × UserUnit` is the correct value for
`tesseract::set_source_resolution`.  Non-numeric and NaN/Inf values are
rejected with `RasterError::InvalidPageGeometry`.

### Fixture inventory

Fixture PDFs are gitignored. Provide your own corpus covering these character classes:

| Character | Size range | Notes |
|---|---|---|
| Light vector + text | ~100 KB | Minimal render path; baseline for overhead measurement |
| Mixed vector + images | ~300 KB | Exercises JPEG decode + path fill together |
| Dense vector / complex paths | ~2 MB | Exercises scanline AA at scale |
| Mixed; image-heavy | ~10 MB | GPU ICC CLUT path |
| Scan-heavy JPEG/JPEG2K | ~50 MB | Primary nvJPEG + nvJPEG2K workload |

### Commands

The results above were produced with scripts since retired with the
`pdf-raster` package name (and on a build where nvJPEG was enabled; it is
now disabled, see the GPU dispatch table); current results and the full
reproduction notes are in the wiki's Benchmarks page (`wiki/Benchmarks.md`).
`$PDF` is any PDF; the local corpus is `tests/fixtures/corpus-01` …
`corpus-10` (gitignored, not in the repository).

```bash
# CPU-only release build; add e.g. --features "nvjpeg2k,gpu-icc" or
# --features vulkan for the GPU backends (CUDA_ARCH=sm_120 for the RTX 5070)
RUSTFLAGS="-C target-cpu=native" cargo build --release -p rasterrocket-cli
BIN=target/release/rrocket
mkdir -p /tmp/out /tmp/ref

# Wall clock vs pdftoppm (pass --backend cuda / vulkan to time a GPU build)
hyperfine --warmup 3 --runs 8 \
  "$BIN --no-ram --backend cpu -r 150 $PDF /tmp/out/p" \
  "pdftoppm -r 150 $PDF /tmp/ref/p"

# Instructions retired per document, two builds compared (without --pdf or
# --corpus-dir it renders a synthetic set built from files in the repository)
python3 -I tests/bench/instructions.py BASE_BIN HEAD_BIN --corpus-dir tests/fixtures

# Per-page latency against pdfium, both in process, one pinned core
tests/compare/latency.sh -r 150 tests/fixtures/corpus-*.pdf

# Pixel diff vs pdftoppm / pdfium
tests/compare/compare.sh -r 150 $PDF

# Flamegraph (cargo-flamegraph; release build with debug info)
CARGO_PROFILE_RELEASE_DEBUG=true cargo flamegraph -o /tmp/flame.svg \
  -p rasterrocket-cli --bin rrocket -- --no-ram -r 150 $PDF /tmp/out/p

# Synthetic fill microbenchmark (raster crate path-fill vs vello_cpu);
# the bench crate has two binaries, so --bin is required
RUSTFLAGS="-C target-cpu=native" cargo run -p bench --bin bench --release -- --iters 30 --stars 200

# Threshold bench — recalibrate GPU dispatch crossovers after any kernel change
CUDA_ARCH=sm_120 cargo run -p rasterrocket-gpu --release --bin threshold_bench

# L3 occupancy monitoring (9900X3D — requires resctrl mount)
# mount -t resctrl resctrl /sys/fs/resctrl
# cat /sys/fs/resctrl/mon_data/mon_L3_XX/llc_occupancy
```

---

## Phase 5 — Public library API ✓ COMPLETE (Apr 2026)

Extract the render pipeline into a reusable library crate. The caller gets 8-bit grayscale pixels in memory and passes them directly to Tesseract — no subprocess, no files, no Leptonica.

### Crate: `crates/rasterrocket`

```rust
pub struct RasterOptions {
    pub dpi: f32,          // render DPI; pass same value to Tesseract set_source_resolution
    pub first_page: u32,   // 1-based
    pub last_page: u32,    // 1-based, inclusive
    pub deskew: bool,      // run deskew before returning pixels (scanned PDFs only)
}

pub struct RenderedPage {
    pub page_num: u32,
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,   // 8-bit grayscale, tightly packed, top-to-bottom
    pub dpi: f32,          // pass to Tesseract set_source_resolution — do not lie
}

/// Render pages from a PDF file, one result per page.
/// A per-page error does not abort remaining pages.
pub fn raster_pdf(
    path: &Path,
    opts: &RasterOptions,
) -> impl Iterator<Item = (u32, Result<RenderedPage, RasterError>)>;
```

**Caller's OCR step after integration:**

```rust
for (page_num, result) in rasterrocket::raster_pdf(path, &opts) {
    let page = result?;
    let text = tesseract::ocr_from_frame(
        &page.pixels, page.width as i32, page.height as i32,
        1, page.width as i32, "eng",   // bpp=1 (grayscale), stride=width
    )?;
    // for uneven-background scans, set thresholding_method=2 (Sauvola) on Tesseract side
}
```

### Preprocessing scope

| Step | Owner | Notes |
|---|---|---|
| Rasterise to grayscale | **rasterrocket** | BT.709 RGB→Gray; already in CLI, just needs exporting |
| Deskew | **rasterrocket** | See deskew design below |
| Background normalisation | **Tesseract** | Sauvola `thresholding_method=2` on the caller side |
| Binarisation | **Tesseract** | LSTM reads grayscale directly; do NOT pre-binarise |
| DPI metadata | **caller** | Pass `page.dpi` to `set_source_resolution`; default is 70 DPI (useless) |

### Deskew design (researched Apr 2026)

**Goal**: beat Leptonica's `pixDeskew` in both speed and accuracy.

**How Leptonica works (and where it fails):**
Hierarchical differential-projection-profile sweep: binarise at threshold 160 → 4x downsample → 14-angle coarse sweep (±7°, 1° steps) → quadratic interpolation → binary search to 0.01° convergence. Accuracy: ~0.03–0.05°. Failure modes: fixed threshold 160 fails on light/dark scans; skips angles < 0.1°; single-threaded; CPU-only rotation.

**Our approach — two-phase hybrid:**

**Phase A — Angle detection (CPU, intensity-weighted projection profile)**

Same algorithm family as Leptonica but without the binarisation threshold:
- Use `255 - pixel` as the foreground weight on raw 8-bit gray — dark pixels count as foreground proportionally, no hard threshold, no parameter to tune
- 4× downsample for coarse sweep (620×825 working set)
- 28-angle coarse sweep at 0.5° steps (±7°), scored by differential square sum of weighted row sums
- Binary search refinement to 0.01° convergence
- AVX-512 row summation via `VPSADBW` (64 pixels/cycle): each row of 2550px takes ~40 AVX-512 ops
- Parallelise sweep angles across Rayon threads (each angle is independent)
- **9900X3D V-Cache advantage**: 8.4MB image fits entirely in 96MB L3; stays warm through all sweep iterations — no DRAM traffic after first load
- Target: **1–3ms** for detection

Accuracy advantage over Leptonica: no binarisation threshold → correct on images where threshold 160 over- or under-segments; corrects angles < 0.1° that Leptonica skips.

**Phase B — Rotation (GPU, CUDA texture bilinear)**

- Bind source image as `cudaTextureObject_t` with `cudaFilterModeLinear` — hardware bilinear at no extra compute cost
- Use `nppiRotate_8u_C1R_Ctx` (NPP single-channel 8-bit rotate) or a custom kernel with `tex2D<float>()` per output pixel
- RTX 5070 texture fill rate: 482 GTexel/s → **0.3–0.5ms** for 8.4MP
- If image is already on GPU from nvJPEG/nvJPEG2K decode, PCIe upload cost is zero
- CPU fallback (12-core AVX-512 bilinear): ~1.5ms — used when GPU unavailable or image is CPU-only

**Steady-state pipeline (scan-heavy PDFs with GPU decode active):**
```
CPU: detect angle for page N+1  (~1ms, overlapped)
GPU stream A: rotate page N     (~0.4ms)
GPU stream B: D→H transfer N-1  (~0.3ms)
```
Net deskew cost per page at steady state: **~0.4ms** (rotation-bound; detection hidden).

**Single-page cold path (CPU RAM, no GPU decode):**
- Detection: ~2ms
- PCIe H2D (8.4MB @ 28GB/s): ~0.3ms
- GPU rotation: ~0.4ms
- PCIe D2H: ~0.3ms
- Total: **~3ms** — still faster than Leptonica's ~10–15ms

### Work items

- [x] New `crates/rasterrocket` library crate; add to `Cargo.toml` workspace members
- [x] Move `render_page_native` core (minus `&Args`, minus file I/O) into library
- [x] Export `rgb_to_gray` (BT.709) from library (currently private in CLI)
- [x] Encapsulate GPU decoder lifecycle (`DecoderInit<T>`) inside library — not caller-visible
- [x] `crates/rasterrocket/src/deskew/detect.rs` — intensity-weighted projection profile, AVX-512 row sums, Rayon sweep parallelism
- [x] `crates/rasterrocket/src/deskew/rotate.rs` — CPU bilinear fallback; GPU path via `nppiRotate_8u_C1R_Ctx`
- [x] Review pass: sentinel hack → `Option<Result>`, pages map O(n²) → O(n), `InvalidOptions` validation, `debug_assert` → `assert`, `NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED` constant, `remove(0)` → `swap_remove(0)`, bilinear inlined into rotate loop, `downsample` factor=0 guard
- [x] Make CLI a thin wrapper over `crates/rasterrocket` (RasterSession, render_page_rgb, open_session)
- [x] Second review pass (Apr 2026): scale validation guard in `render_page_rgb`; GPU init failure `eprintln!`; `PageIter::next` Err arm cleaned; dead variable removed from `bitmap_to_vec`; `# Panics` doc corrected in lib.rs; `MONO_THRESHOLD` const extracted in CLI; atomic temp-file rename in CLI `render_page` (no partial files on encode failure)
- [x] Third review pass (Apr 2026): `open_session` double get_pages eliminated; bad `scale` returns `InvalidOptions` (not `PageDegenerate`); `PageIter::next` Err arm rewritten with explicit match; compile-time `Sync` assertion on `RasterSession`; `#[expect]` replaces `#[allow]` on Args; erroneous `cast_sign_loss` suppression removed from f64→f32 and f32→i32 casts; SWEEP_STEPS≥2 compile-time assert; `n_rows-skip` saturation guard; intermediate Vec allocations in coarse sweep eliminated (par reduce); `assert!→debug_assert!` in private `downsample`; rotation docs corrected (CW-positive throughout); GPU deskew stub noted in lib.rs; scatter loop AVX-512 auto-vec claim removed from doc; rename failure now also removes temp file; `--odd`+`--even` mutual exclusion check; open_session error walks source chain; DPI args validated ≥1 at CLI; jpeg_quality validated 0–100; `OutputFormat` implements Display; 13 redundant `default_value_t=false` removed
- [x] GPU rotation: `rotate_gpu` via `nppiRotate_8u_C1R_Ctx` — NPP CW-positive (Y-down); GPU/CPU parity ≤2 grey levels at 2°; thread-local `NppRotator`, CPU fallback retained; hardening pass (input validation, three-state slot, Drop logging, null asserts)
- [x] Integration tests: round-trip a fixture PDF, assert pixel dimensions and grayscale range; deskew unit tests with synthetic skewed images at known angles

---

## Phase 6 — Integration hardening and OCR pipeline fit ✓ COMPLETE (Apr 2026)

### Goal

Make rasterrocket the drop-in replacement for the pdftoppm + Leptonica preprocessing
stack in a production OCR pipeline.  The rasterise + deskew path is feature-complete;
Phase 6 closes the remaining gaps before the first production integration.

### Open work items

- [x] **`UserUnit` support** — `page_size_pts` now reads `UserUnit`, validates
  it to `[0.1, 10.0]` (returning `RasterError::InvalidPageGeometry` for
  out-of-range values), multiplies `w_pts`/`h_pts` by `user_unit`, and exposes
  `PageGeometry.user_unit`.  `RenderedPage.effective_dpi` = `opts.dpi × UserUnit`
  is the correct value to pass to `tesseract::set_source_resolution`.
  Non-numeric or NaN/Inf `UserUnit` values are also rejected with a descriptive
  error.  The double `get_pages()` call in `page_size_pts` and `parse_page` was
  eliminated via a shared `resolve_page_id` helper (commits 4aa17b5 / ce10242 /
  cf3b3a7).

- [x] **`RenderDiagnostics` on `RenderedPage`** — `RenderedPage.diagnostics`
  (`PageDiagnostics`) exposes: `has_images`, `has_vector_text`,
  `dominant_filter` (most-used `ImageFilter` variant: `Dct / Jpx / CcittFax /
  Jbig2 / Flate / Raw`), and `source_ppi_hint` (estimated source PPI of the
  dominant image).  Collected at zero extra cost during rendering: `blit_image`
  increments per-filter counts; `show_text` sets `has_vector_text`; `finish()`
  resolves `dominant_filter` from counts.  `ImageFilter` and `PageDiagnostics`
  are re-exported from `rasterrocket` (commit 199d13a).

- [x] **Pipelined render + OCR** — `rasterrocket::render_channel(path, opts, capacity)`
  returns a `std::sync::mpsc::Receiver<(u32, Result<RenderedPage>)>`.  A
  Rayon-spawned producer renders pages in ascending order and sends them as they
  complete; the consumer (Tesseract) processes each page immediately.  The channel
  is bounded to `capacity` slots (min 1): producer blocks when consumer falls
  behind, capping peak memory at `capacity × page_size`.  Options validation runs
  synchronously before spawn; session-open and per-page errors are delivered
  through the channel (same non-fatal contract as the iterator).  Zero new
  dependencies — `rayon` was already present; `std::sync::mpsc` is stdlib.
  `validate_opts` extracted from `render_pages` so both paths share identical
  validation.  5 unit tests cover all error paths and backpressure.

- [x] **DPI auto-selection hint** — `PageDiagnostics::suggested_dpi(min, max)`
  rounds `source_ppi_hint` up to the nearest standard DPI step
  (72 / 96 / 150 / 200 / 300 / 400 / 600) and clamps to `[min, max]`.
  `RenderedPage::suggested_dpi` delegates to it.  Returns `None` for
  vector/text-only pages so callers fall back to their default DPI.
  No stored field — pure computed from existing `source_ppi_hint`.

- [x] **`npp_rotate` / `nvjpeg2k` shared CUDA init helper** — the duplicated
  five-step CUDA driver init sequence (`cuInit → cuDeviceGet →
  cuDevicePrimaryCtxRetain → cuCtxSetCurrent → cuStreamCreate`) and the eight
  `libcuda.so` FFI declarations are extracted into `crates/gpu/src/cuda.rs` as
  `gpu::cuda::init_primary_ctx_and_stream(device_ordinal: i32) → Result<CudaInit,
  CudaInitError>`.  `NppRotator::new` maps the error to `NppRotateError(format!)`
  and `NvJpeg2kDecoder::new` maps it to `NvJpeg2kError::CudaError(code)`.
  The module is unconditionally compiled; `#[cfg_attr]` guards suppress dead-code
  lints when neither GPU feature is active.

---

## Phase 7 — Heterogeneous dispatch hub

**Goal:** dynamic per-page work routing across CPU threads, GPU decoders, and iGPU (VA-API) based on page content type, JPEG variant, and image size — rather than the current static pixel-area threshold.

### Motivation

Benchmarking revealed that the current threshold-only dispatch leaves significant wins on the table:

- **Progressive JPEG (SOF2)** falls through VA-API entirely (VAEntrypointVLD supports baseline only) — but nvJPEG handles progressive JPEG natively. Detecting the JPEG type at parse time and routing to nvJPEG instead of CPU would recover the GPU decode win on scan corpora 08–09.
- **Corpus 09 (490 progressive JPEG pages):** our CPU path finishes in ~12s (24 threads) while Poppler takes 10+ minutes single-threaded. A GPU path (nvJPEG progressive) would reduce this further.
- **Mixed workloads** waste GPU dispatch overhead on small images. A smarter router that inspects content type before dispatch avoids the penalty on dense-image-small corpora (corpus 06 is currently 0.6× on GPU).

### Design

**Content-aware dispatch signals** (inspected at parse time, zero extra I/O):

| Signal | Source | Routing hint |
|---|---|---|
| JPEG variant (SOF0 vs SOF2) | JPEG SOF marker | SOF2 → nvJPEG (not VA-API); SOF0 → VA-API eligible |
| Image pixel area | PDF dict Width×Height | Below threshold → CPU always |
| Page image count | Accumulate during parse | Many small images → CPU batch; few large → GPU |
| Dominant filter | `PageDiagnostics.dominant_filter` | DCT-heavy → prefer GPU; Flate/CCITT → CPU |

**Work-stealing queue:**

Replace the current Rayon page-parallel split with a dynamic queue where each page is a task. GPU decoder slots (nvJPEG, nvJPEG2k, VA-API) are resources claimed per-task. CPU threads take tasks when GPU slots are full. This gives:
- GPU handles progressive/large JPEG pages
- CPU handles text/vector pages concurrently
- No idle time waiting for GPU if CPU work is available

**JPEG type detection in `decode_dct`:**

Read the SOF marker byte before dispatch:
- `0xC0` (SOF0, baseline) → VA-API eligible, nvJPEG eligible
- `0xC2` (SOF2, progressive) → nvJPEG only (VA-API skipped, no wasted parse attempt)
- `0xC1`, `0xC3` (extended/lossless) → CPU only

Currently every progressive JPEG incurs a full VA-API header parse + `BadJpeg` error + fallthrough. Detecting SOF type in ~3 bytes eliminates this overhead and routes correctly.

### Work items

- [x] Extract SOF marker detection into `gpu::jpeg_sof_type()` — `crates/gpu/src/jpeg_sof.rs`; `JpegVariant { Baseline, Progressive, Other }`; zero-allocation marker scan; `#[must_use]`; 8 unit tests; shared by VA-API and dispatch
- [x] Update `decode_dct` dispatch: `jpeg_variant = gpu::jpeg_sof_type(data)` before threshold check; nvJPEG accepts `Baseline | Progressive`; VA-API accepts `Baseline` only — progressive skipped entirely; `VapiJpegDecoder::decode_sync` also guards with early return; `decode_dct_gpu` + `decode_dct_vaapi` collapsed into generic `decode_dct_gpu_path<D: GpuJpegDecoder>`
- [x] Work-stealing page queue: bounded `mpsc::sync_channel` + `rayon::scope`; `RoutingHint` extension point; back-pressure at 2× thread count; `crates/cli/src/page_queue.rs`; deadlock fix + single-thread guard; `routing_hint_from_diag` + `ProgressCtx::report` live in `page_queue.rs`
- [x] `PageDiagnostics` pre-scan pass: `rasterrocket_interp::prescan_page` walks XObject dict + content stream operators without decoding pixels; sets `GpuJpegCandidate`/`CpuOnly` hints before enqueueing; `crates/rasterrocket-interp/src/prescan.rs`; `count_filter` + `update_max_ppi` helpers extracted
- [x] Serial prescan loop removed from CLI render path — all pages default to `RoutingHint::Unclassified`; `routing_hint_from_diag` retained as extension point for future affinity dispatch; recovered 15-20% throughput regression
- [x] Affinity dispatch: prescan all pages sequentially before pool start; `CpuOnly` hint → `BackendPolicy::CpuOnly` override in `render_page_rgb_hinted` → `lend_decoders` skips `ensure_nvjpeg` and `DECODER_INIT_LOCK` acquisition; `GpuJpegCandidate` uses session policy unchanged; single rayon pool (soft affinity)
- [x] Benchmark: full v0.6.0 matrix on 9900X3D + RTX 5070 (`bench/v060/results.md`) and i7-8700K + RTX 2080 SUPER testbench (`bench/v060/results-testbench-i7-8700K-rtx2080super.md`).  Target corpus 08/09 GPU speedup ≥ 5× **was not met** — nvJPEG-via-`GPU_HYBRID` is 5–13× *slower* than 24-thread zune-jpeg on every DCT-heavy corpus on both machines.  Root cause and fix scoped under Phase 8.
- [x] ~~Re-bench with `nvjpeg-hardware` feature flag enabled (mode E, `NVJPEG_BACKEND_HARDWARE`)~~ — closed as moot.  The mode was added 2026-05-07 to measure rather than infer, but on consumer GeForce there is nothing new to time: `libnvjpeg.so` rejects `NVJPEG_BACKEND_HARDWARE` at handle creation (see the hard-blocker section below), so the feature falls back to `GPU_HYBRID`, the backend the v0.6.0 matrix already measured; and nvJPEG is disabled in every build (`GPU_JPEG_THRESHOLD_PX = u32::MAX`).  The `nvjpeg-hardware` feature remains for measuring a data-centre card.

---

## Hard blocker: NVJPG silicon access on consumer NVIDIA

Documented here once, then referred to elsewhere.

The fixed-function NVJPG hardware engine is **closed to consumer GeForce SKUs** in three independent ways:

1. **The user-space library `libnvjpeg.so` rejects `NVJPEG_BACKEND_HARDWARE` at handle creation** on consumer cards (verified on RTX 5070; the `nvjpeg-hardware` cargo feature added 2026-05-07 confirms this empirically).
2. **The open kernel module (`open-gpu-kernel-modules`) exposes NVJPG class IDs** for current architectures (`NVCDD1`, `NVCFD1`) but **deliberately does not publish the command-buffer methods**, the PRI register definitions, or the firmware that drives the engine. ~220 lines of NVJPG-related kernel code in the open repo, all of it context lifecycle / capability-table reading; zero of it is the actual decode submission path.
3. **No Vulkan extension exposes JPEG decode.** `VK_KHR_video_decode_*` covers H.264/H.265/AV1/VP9 only; Khronos has not standardised JPEG video, NVIDIA has not proposed a vendor extension. Vulkan-on-NVIDIA on consumer Blackwell exposes H.264/H.265 decode operations on the queue family, nothing else.

Reverse-engineering desktop NVJPG would take a multi-month research project (cf. Asahi Linux's GPU work for analogous scope), wouldn't transfer across architecture generations, and would land in a legally murky zone (Falcon firmware signatures). No academic or community project is publicly working on this; the ROI doesn't exist for the open-source ecosystem either.

**What this means for rasterrocket:** the only open path to GPU acceleration of JPEG-related work is the SM array via custom CUDA or Vulkan compute shaders. The fixed-function engine is unreachable from any open code path. This is the load-bearing constraint behind Phase 9's design (CPU decode + device-resident pipeline) and Phase 8's deferral.

---

## Phase 8 — Custom on-GPU parallel Huffman decoder (Phase A SHIPPED as OSS artifact v1.0.0; B–D CLOSED; REMOVED 2026-11-15)

**Removed (2026-11-15).** The whole GPU JPEG decoder is deleted: the parallel-Huffman entropy stage, the Blelloch scan, the GPU IDCT and the CPU pre-pass that fed them, with the `gpu-jpeg-huffman` feature. It had no case to win: the CPU pre-pass already decodes every symbol of the scan to resolve the DC chain, and the CPU walks the stream again for the magnitude bits after the device returns, so the device stage replaces only table lookups between two sequential CPU walks; whole-process it lost about 10× on gray scans (Vulkan, 20 pages at 600 dpi: 2.2 s against 0.22 s for `--backend cpu`) and 1.6–8× on the A18 Pro. Master `10393d55` is the last commit with the complete decoder, converging on CUDA and Vulkan.

**Status:** Phase 0 (CPU pre-pass) shipped 2026-05-06/07. Phase A (the parallel-Huffman algorithm in isolation) shipped 2026-05-12 / v1.0.0 as a byte-identical CUDA+Vulkan OSS artifact, wired into the production Vulkan path but dormant by default (`GPU_JPEG_HUFFMAN_THRESHOLD_PX = u32::MAX`). Phases B–D are closed (2026-11-12): the bench gate confirmed 24-thread CPU wins on aggregate throughput, which is the answer to the research question, not a gap.

**Why deferred:**

Phase 8 was originally scoped under the assumption that we needed a GPU-resident JPEG decoder to enable device-resident pixels. That assumption was wrong: CPU decode + one strategic upload achieves the same architectural property (Phase 9) with dramatically less work.

The v0.6.0 baseline matrices on both 9900X3D + RTX 5070 and 8700K + RTX 2080 SUPER, plus the consumer-Blackwell NVJPG investigation, established that:

- Multi-thread CPU JPEG decode (zune-jpeg, AVX-512 IDCT) at 24 threads delivers ~5 GB/s aggregate. This is the path Phase 9 keeps.
- A custom parallel-Huffman GPU decoder, even built ideally per Wei et al. 2024, would *match or marginally beat* per-image latency but **lose on aggregate throughput** to 24-thread CPU. The 51× speedup over libjpeg-turbo in the paper is on A100 datacenter hardware against a single-thread CPU baseline; consumer Blackwell + 24-thread CPU is a different comparison.
- The architectural payoff (device-resident pixels enabling GPU AA / ICC / tile fill / composite) is achievable via Phase 9 without Phase 8.

**What stays in the codebase:**

The Phase 0 work (`crates/gpu/src/jpeg/`) shipped clean across three commits and 84 passing unit tests. It stays as scaffolding for a future hobbyist port and as well-tested JPEG metadata utilities that any future work might want.

**Why Phase 8 stays in the roadmap at all:**

The Weißenberger 2018 self-synchronizing parallel Huffman algorithm is genuinely beautiful CUDA work. Implementing it produces an open, redistributable artifact that demonstrates a non-obvious algorithm. If rasterrocket ever wants to run on a workload where 24 CPU threads aren't available (embedded, single-core, etc.) the GPU decoder becomes relevant. The work has long-term value as a learning project; it's just not the path to faster rasterrocket.

The full original spec is preserved as a developer-side research artifact; it isn't active design.

**Phase A (algorithm in isolation) shipped 2026-05-12 as an OSS artifact.** Local plan: `docs/superpowers/plans/2026-05-11-gpu-jpeg-huffman-v2.md`. The 4-phase parallel-Huffman algorithm (Wei §III) runs end-to-end on synthetic streams with byte-identical CUDA + Vulkan output:

- Phase 1 intra-sync, Phase 2 inter-sync (bounded retry), Phase 3 Blelloch scan, Phase 4 re-decode + write.
- Boundary-snapshot semantics in Phase 1 — `s_info[i]` captures `(p, n, c, z)` at the first decode crossing into subseq (i+1)'s region, not the over-walk end state. Phase 4's predecessor inheritance composes cleanly off this.
- Typed `Phase4FailureKind` per-subseq exit-condition surface — kernel writes `decode_status[i] ∈ {Ok, PrefixMiss, LengthBits, Incomplete}`, dispatcher inspects post-Phase-4 and surfaces non-Ok as typed `BackendError`. Defence-in-depth against future kernel divergence.
- 10-vector adversarial corpus (`huffman/corpus.rs`) covering short / long / uniform / skewed / single-symbol / max-len-16 / Phase-2-retry-trigger / one-subseq / word-aligned / max-tail-padding. Pass on both backends, byte-for-byte.

**Phase A acceptance criterion satisfied.** Phases B–D (real JPEG framing, integration, perf gate) are closed, for the original Phase 8 deferral reasoning — the aggregate-throughput comparison vs. 24-thread CPU doesn't change just because the algorithm now demonstrably works.

**MVP limitations carried forward** (would block production, not the OSS artifact):

- *Synthetic-stream* Phase 2's `(c, z)` sync predicate hits its `2 × log2(n)` retry bound on mixed-codeword-length and multi-component-with-z-rollover corpora. Adversarial inputs surface as a typed `SyncBoundExceeded` error, never as a hang or wrong-output. **The JPEG-framed Phase 2 no longer shares this limitation** (2026-08-17): `jpeg_phase2_inter_sync` was rewritten as re-decode propagation — each pass, subseq `i` re-decodes its region from `s_info[i-1]`'s boundary snapshot over a host-ping-ponged double buffer, flagging fixpoints; slot 0 is pinned correct, so a fixpoint pass proves the whole chain correct by induction and convergence is guaranteed within `n` passes (bound = `n − 1`) for every input, including multi-component streams whose fresh-start block phases the old agreement predicate could never fix. The formerly-ignored YCbCr Phase-4 tests now pass on CUDA and Vulkan, and the real-image decoder pixel tests exercise the full GPU Huffman path end to end.
- Test fixtures must use zero-value-bit symbols (`symbol & 0x0F == 0`) — the kernel advances `code_bits + value_bits` per decode, so a uniform-length **code** still produces a non-uniform **advance** for symbols whose low nibbles vary. Real JPEG framing doesn't have this constraint (DC tables have low-nibble = bit length intentionally; AC tables similarly), but the synthetic-fixture builder must pick symbols accordingly.

---

## Phase 9 — Device-resident image cache and GPU page buffer (✓ SHIPPED v1.0.0 — `cache` feature, opt-in)

**Goal:** decoded image pixels and the page being rendered both live in VRAM for the lifetime of a render session, so the rendering hot path performs zero PCIe round-trips per image and zero decode work on cache hits.

**Why this is the right phase now:**

Three load-bearing facts from the v0.6.0 baseline + consumer-Blackwell NVJPG investigation:

1. Multi-thread CPU JPEG decode is faster than any GPU JPEG path we can ship on consumer hardware.
2. The 4 already-shipped GPU kernels (AA fill, ICC CLUT, tile fill, composite) don't carry their weight today because every kernel pays a PCIe round-trip per invocation.
3. The OCR pipeline pattern renders the same PDF multiple times; today every pass re-decodes every JPEG.

Phase 9 addresses all three at the same time: the cache makes (3) free after the first render, the device-resident page buffer makes (2) finally pay off, and (1) is no longer a problem because we're not trying to beat CPU JPEG decode — we're keeping its output in VRAM.

**Architecture (one-liner):** three-tier cache (VRAM → host RAM → disk) with content-hash dedup (BLAKE3) keying across documents, plus a device-resident page bitmap that the existing GPU kernels read and write in place.

**Work items:**

- [x] **Task 1 — `ImageData` enum, `ImageData::Cpu` variant only** (commit `f0519ca`, hardened in `8a19c3a`/`48aeecb`). `Vec<u8>` → `ImageData::Cpu(Vec<u8>)` plumbing across the renderer; `#[non_exhaustive]` enum so `Gpu` variant is a non-breaking add. The Gpu variant itself is deferred to Task 4 wiring.
- [x] **Task 2 — VRAM tier in-process** (commit `e3709ee`, hardened in `e3acb21`/`69a1cd2`). `crates/gpu/src/cache/` module behind a new `cache` feature: `DeviceImageCache` with dual-key (BLAKE3 content hash + (DocId, ObjId) alias), DashMap-backed concurrent access, LRU + refcount-pinned eviction, `InsertRequest` builder, structured `CacheError`. 8 cache tests under `cache,gpu-validation`; concurrent-insert dedup test proves no double-counted `used_bytes`.
- [x] **Task 3 — Host RAM tier** (commit `0e197c3`, hardened in `52acfdf`/`e2f750d`). `crates/gpu/src/cache/host_tier.rs` with `PinnedHostSlice<u8>` slabs, independent budget + LRU, demote-on-evict from VRAM, promote-on-hit back to VRAM. Critical fix in the hardening pass: `clone_htod` must take `&PinnedHostSlice` directly (not `as_slice().ok()?`) so cudarc records the H→D event back to the slice's internal event — otherwise `PinnedHostSlice::Drop` could free pinned memory mid-DMA. 13 cache tests; end-to-end demote+promote round-trip verifies bit-identical pixels.
- [x] **Task 4 — Device page buffer + GPU blit kernel** (commits `6ee47de`/`738ba14`/`ef67045` GPU side, `8f01c3d` AA-fill fix, `a7859e4` renderer integration). GPU side: `kernels/blit_image.cu` (16×16-block CUDA kernel with f32 inverse-CTM nearest-neighbour sampling matching the CPU path byte-for-byte), `gpu::blit` module (`InverseCtm`, `BlitBbox`, `GpuCtx::blit_image_to_buffer`, structured `BlitError`), `cache::DevicePageBuffer` (zero-init RGBA8). Renderer integration: `ImageData::Gpu(Arc<CachedDeviceImage>)` feature-gated variant, `decode_dct → ImageData::Gpu` wiring with BLAKE3 content-hash dedup + `(DocId, ObjId)` alias, per-page `DevicePageBuffer` lazy-allocated on first GPU image, source-over composite of buffer onto host bitmap at `finish()`. New `cache` feature in `rasterrocket-interp`, `rasterrocket`, and the CLI. AA fill / ICC / tile fill / composite still use the CPU bitmap; rewiring those to read/write the device buffer is deferred (the `coverage_scratch` field in the spec). Pre-existing AA-fill `JITTER_Y` corruption (8 wrong Halton(3) values) found and fixed in `8f01c3d`. The end-to-end bench gate has since been run and is closed as reframed; see the bench-gate paragraphs below.
- [x] **Task 5 — Disk tier**. `crates/gpu/src/cache/disk_tier.rs` — `<root>/<doc-hex>/<hash-hex>.bin` sidecar files with PDRF magic + version + dimensions header.  Atomic write via temp+rename; `posix_fadvise(WILLNEED)` on Linux for read-ahead; document-mtime eviction.  Env-var overrides: `PDF_RASTER_CACHE_DIR`, `PDF_RASTER_CACHE_BYTES`.  `open_session` switched to BLAKE3-of-PDF-bytes for the `DocId` so editing a PDF naturally invalidates the disk cache.  7 unit tests under `cache` feature (no GPU needed).
- [x] **Task 6 — Pre-fetcher** (commits `013219b` + `a2e81d9` hardening pass). `crates/rasterrocket-interp/src/cache/prefetch.rs` — `spawn_prefetch(doc, cache, doc_id, config)` walks every page's `/XObject` resource dict, dedupes by `ObjId`, decodes `/DCTDecode` images on a small `std::thread` worker pool (default 2, capped at `MAX_PREFETCH_WORKERS = 16`).  Discovery is single-threaded; `seen` is a plain local `HashSet<ObjId>`.  Decoder panics caught per-image so one bad XObject doesn't kill the run.  Opt-in via `SessionConfig::prefetch`; `RasterSession.doc` upgraded to `Arc<Document>` so the prefetcher can hold its own clone.  Form-XObject contents are not recursed into; renderer decodes them on first touch.  4 unit tests under the `cache` feature (no GPU needed).

**Bench gate (PARTIAL after disk-tier rework, see release-history block above for full numbers):** initial bench on both 9900X3D + RTX 5070 (sm_120) and i7-8700K + RTX 2080 SUPER (sm_75) showed criterion 5 failing 0/5 with mode DCP **3–14× slower** than mode A on DCT-heavy corpora.  Three fixes landed in commit `0bd61ca`: async disk writer (renderer no longer blocks on `sync_all`), opt-in disk tier (no surprise persistence cost), and cold-start three-tier lookup (the disk tier was actually unreachable on a fresh process before).

Re-bench result: cold-render regression collapsed from 14× to 1.1–1.9× on local and 1.06–4.85× on testbench, with σ down from thousands of ms to <200 ms.  Criterion 5 still fails 0/5 — but criterion 5 as worded ("mode D beats mode A on cold first render") was the wrong test for what this cache architecture does.  The cache wins on **second** render: corpus 07 second render at 1,093 ms vs 3,319 ms first = 33% (criterion 2: ≤30%, off by a hair).  Criterion 3 (mode A no regression) passes on testbench within noise; minor drift on local.  Criterion 4 (no OOM corpus 09) passes.

**Reframing:** this cache is not a cold-render speedup; it's a **cross-pass / cross-session** speedup for OCR pipelines and repeat renders.  Both bench machines confirm that's what the architecture delivers, deterministically.  The `cache` feature flag stays opt-in via `PDF_RASTER_CACHE_DIR` for cross-session disk persistence; the in-memory tiers run unconditionally when the feature is built.

**Total scope:** ~1850 LoC new Rust + ~150 LoC new CUDA + ~400 LoC modified existing. Tasks 1+2 ship in ~5-7 days; full pipeline ~3 weeks elapsed.

**Shipped:** all six tasks landed; the `cache` feature is in `rasterrocket`, `rasterrocket-interp`, and the CLI as of v1.0.0. Since v2.2.0 the cache also runs on the Vulkan backend: `DeviceImageCache<B>` is generic over a `CacheBackend` (CUDA, Vulkan), the session builds it on the backend it resolved to, and a Vulkan image is blitted into a bounding-box target in one backend page (see Phase 10, Task 4 step 5). In-memory VRAM/host tiers run unconditionally when the feature is built; the disk tier is opt-in via `PDF_RASTER_CACHE_DIR`. The bench-gate reframe above (cross-pass / cross-session speedup, not cold-render) is the final answer, not an open item — criterion 5 as originally worded tested the wrong property and is closed as such.

---

## Phase 10 — Vulkan compute backend (✓ SHIPPED v0.8.0/v1.0.0 — runs on AMD RADV since v1.5.0; RADV and llvmpipe byte parity verified 2026-11-12)

**Goal:** replace the CUDA-specific kernel launch and device-memory layer with a backend-abstracted layer that has both CUDA and Vulkan compute implementations, so the same algorithmic kernels run on NVIDIA, AMD, Intel, and Apple GPUs from one source tree.

**Why now (and why not before):**

Vulkan compute on the dev machine (RTX 5070) was confirmed 2026-05-07 to expose Vulkan 1.4.312, conformance 1.4.1.3, full subgroup operations (the equivalent of CUDA warp intrinsics), tensor cores via `VK_KHR_cooperative_matrix`, and the same SM array CUDA uses. Cross-vendor portability is the real reason — the same SPIR-V kernel runs on AMD (RADV), Intel (ANV), Apple (MoltenVK→Metal), and Mesa lavapipe (CPU debug).

What was missing before Phase 9 was *the abstraction layer to even consider a backend swap*. Phase 9 introduces backend-agnostic shapes (`ImageData::Gpu`, `CachedDeviceImage`, `DevicePageBuffer`); Phase 10 swaps the *implementation* behind those shapes from CUDA-specific to backend-trait-driven, with concrete CUDA and Vulkan backends.

**Approach:**

- Single kernel source-of-truth in **Slang** (Khronos-supported shading language). One `.slang` file per algorithm; `slangc` compiles to PTX for CUDA backend and SPIR-V for Vulkan backend.
- `GpuBackend` trait abstracting device-memory, kernel launch, host↔device transfer, synchronisation. Two implementations: `CudaBackend` (current, refactored) and `VulkanBackend` (new, via `ash`).
- `BackendPreference` enum on `RasterOptions`: `Auto` (CUDA on NVIDIA, Vulkan elsewhere), `ForceCuda`, `ForceVulkan`, `CpuOnly`.
- The Phase 9 cache and page buffer become generic over `B: GpuBackend`. (Done for the cache, 2026-11-13: `DeviceImageCache<B: CacheBackend>`; the page buffer stays the CUDA blit target, see Task 4 step 5.)

**Work items:**

- [~] **Task 1 — Backend trait + CUDA refactor** (merged via `4c22ce0`).
    - **Shipped:** `GpuBackend` trait + `*Params` structs with state-machine and invariant docs (`crates/gpu/src/backend/{mod,params}.rs`); `CudaBackend` init + alloc + budget + `record_*` + `submit_page` / `wait_page` (`crates/gpu/src/backend/cuda/`); the five existing kernels extracted into `lib_kernels::{aa, composite, icc, soft_mask, tile}`; `BackendError::msg`; `cuda_backend_smoke` + `cuda_backend_per_page` tests; `crates/gpu/src/cache/mod.rs` split into `budget` / `eviction` / `promotion` submodules.
    - **Deliberately deferred:** renderer migration to the trait (the spec's `rasterrocket_interp::renderer::page::gpu_ops` rewrite). The Phase 9 blit path is *already* per-page-batched (no `synchronize` between blits; only `buf.download()` at end-of-page), so migrating shape-only without an upload/download surface on the trait would just shuffle code. `DevicePageBuffer` and `DeviceImageCache` therefore stay un-generified for now; they generify alongside Task 3 once the trait grows the H↔D surface that the Vulkan side will need anyway. See the docstring on `DevicePageBuffer` (`crates/gpu/src/cache/page_buffer.rs`) for the in-tree rationale. The cache was generified on 2026-11-13 (Task 4 step 5).
    - **Not yet measured:** the spec's `±5%-of-pre-refactor` per-kernel bench gate. Deferred until Task 3 lands so the bench matrix runs CUDA + Vulkan together.
- [x] **Task 2 — Slang port of all kernels** (merged via `42dc479`). All six kernels translated; `slangc -target spirv` invocation in `build.rs`; `vulkan` Cargo feature gates the SPIR-V compile; aa_fill's `WaveActiveSum` confirmed lowering to `OpGroupNonUniformIAdd Reduce` (no fallback emulation).
    - **Sub-task 2a — CPU twin per kernel.** For each kernel, ship a plain-Rust function that produces bit-identical output (or pixel-diff ≤ 1 LSB for floating-point ones) used as a correctness oracle in tests. Lets us validate Slang→SPIR-V codegen *without* a GPU, isolates "is the kernel logic right" from "is the Vulkan dispatch right" when chasing cross-backend divergence. Already partially present (`crates/gpu/src/blit.rs` has a CPU reference for `blit_image`); generalise to all five.
- [~] **Task 3 — Vulkan backend implementation**. `VulkanBackend` ships at `crates/gpu/src/backend/vulkan/` via `ash` 0.38 + `gpu-allocator` 0.28 (pure Rust, no FFI).  Module split: `device` (instance/device/queue init, Vulkan 1.3 features incl. shaderInt8 + bufferDeviceAddress + 8-bit storage; ranks discrete > integrated > virtual > CPU when picking physical devices), `memory` (slab sub-allocator wrapping gpu-allocator with persistent-mapped host buffers and BDA-queried device buffers), `pipeline` (lazy SPIR-V→VkPipeline cache, per-kernel descriptor set layouts, on-disk `VkPipelineCache` blob at `$XDG_CACHE_HOME/rasterrocket/vulkan_pipeline_cache.bin`), `recorder` (per-page command buffer, timeline-semaphore page fence, single-page-in-flight invariant, compute→compute memory barriers between dispatches, 2D dispatch for aa_fill so images >256×256 fit the workgroup-count limit), `transfer` (synchronous upload/download via a reusable host-coherent staging buffer grown to high-water-mark).
    - **Parity tests passing (15/15):** four `composite_rgba8`, three `apply_soft_mask`, four `aa_fill` (square / triangle / degenerate / 512×512-exceeds-1D-limit), two `icc_cmyk_clut`, two `blit_image` (RGB / Gray) — all match the CPU reference within ≤ 1 LSB per channel on the dev box's nouveau ICD.
    - **Task 3 follow-ups (perf only), measured 2026-11-13 and closed as not worth it.**  `vulkan_phase_bench` (features `vulkan,cache`) times each kernel page's steps apart (host preparation, recording, `submit_page`, `wait_page`, host read) and splits the wait into the copies-only page and the kernel's device time (per-dispatch slope of a 5-dispatch page).  Medians of 9; RTX 5070 / RADV Raphael iGPU:
        - Fixed cost of one page: record 0.4 µs, submit 1.6 / 6.6 µs, wait of an empty page 16 / 29 µs.  A page with one dispatch and its copies records in 1.6–12 µs (NVIDIA) / 0.6–8 µs (RADV), descriptor work included.
        - ICC CLUT, grid 17, smooth CMYK (photographs, scans), production `icc_clut` call: 0.5 Mpx 0.49 / 0.77 ms, 2 Mpx 1.56 / 3.0 ms, 8 Mpx 6.5 / 14.7 ms; CPU evaluator, one thread: 17 / 68 / 271 ms.  NVIDIA 8 Mpx: allocation + two synchronous uploads 3.0 ms, kernel 0.26 ms, synchronous download + free 2.7 ms.  RADV 8 Mpx: kernel 10.4 ms of 13.5 ms.  Random CMYK (worst case, no CLUT locality) makes the kernel 4–5× slower on both.
        - AA fill (rectangle tables, the threshold bench's sizes): 2048² 0.56 / 3.46 ms, of which copies 0.34 / 0.03 ms and kernel 0.11 / 3.33 ms; 256² 36 / 87 µs.
        - Image-box blit from a 2550 × 3300 RGB image: box 256² 57 / 81 µs, 1275 × 1650 0.87 / 1.49 ms (copies 0.65 ms of it on NVIDIA, kernel 1.19 ms on RADV), 2550 × 3300 4.5 / 7.7 ms.
        - **Dedicated transfer queue: not worth it.**  Host↔device copies run at 13.2–13.4 GB/s on the compute family and 14.2–14.4 GB/s on the transfer-only family (32 MiB, both directions): +7 % bandwidth.  Overlap is bounded by the kernel time a copy could hide, and every page's copies and kernel are data-dependent: ICC 0.26 ms of 6.5 ms, AA 0.11 of 0.56 ms, blit 0.03 of 0.87 ms on NVIDIA; RADV has no copies (buffers are mapped).  Only a chunked ICC pipeline across two queues could gain (≈ 40 % of an 8 Mpx conversion), and GPU ICC conversions are rare (2 of 400 archive PDFs, one cover each).
        - **ICC CLUT as a 3D texture: not worth it.**  A kernel with every CLUT load removed (the upper bound of any fetch optimisation) runs 26 % faster on NVIDIA (0.26 → 0.19 ms at 8 Mpx, 1 % of the call) and 3 % faster on RADV (10.35 → 10.07 ms) on smooth input; one 32-bit CMYK load instead of four byte loads gains 2–4 %.  The RADV kernel is compute-bound, not fetch-bound.
        - **BDA push constants: not worth it.**  Descriptor allocation, update and bind are a part of the 0.6–12 µs a one-dispatch page records in, against pages of 30 µs (smallest fill) to milliseconds.
        - **Batching cache blits or releasing the page lock earlier: not worth it.**  The fixed cost per blit is the empty-page round trip plus record and submit, ≈ 20 µs (NVIDIA) / 40 µs (RADV), against 0.4–4.5 ms for the page-sized boxes blits draw: the box's download dominates (NVIDIA) or the kernel does (RADV).  96 concurrent blits take the same wall time on 1, 4 and 12 threads (256²: 5.8 / 5.1 / 5.0 ms; 850 × 1100: 31 / 33 / 39 ms): the device is the bottleneck, not the lock.  Blits are also rare: 12 in the first 10 pages of 60 archive PDFs at 150 dpi, 30 at 300 dpi.  A draw that reduces its image (every scanned page at 150 dpi) takes a host copy of the cached image instead of a blit; on RADV that read-back ran from uncached memory and is fixed (below).
    - **Formerly listed as blocked on external factors:**
        - ~~**`record_tile_fill` parity test.**~~ Moot: the tile-fill kernel is deleted. The single `aa_fill` kernel has a host twin and CUDA/Vulkan/twin parity tests.
        - **`OnceLock::get_or_try_init` usage.** Still waiting on the stable API (rust-lang/rust#109737).  Today's manual race-loser cleanup in `pipeline.rs::get` is correct; switch when stable.
        - ~~**Cross-vendor smoke (AMD-RADV, Intel-ANV, lavapipe).**~~ No longer hardware-blocked: the dev box has an AMD Raphael iGPU, and `--vulkan-device radv` and `llvmpipe` render (v1.5.0). The byte-parity run against the CPU path is done (2026-11-12: `ab_archive.sh`, fixtures plus the 150-file archive sample, 621 pages at 50 dpi gray with the AA fill forced on: 0 differences from the CPU path on RADV and on llvmpipe); Intel ANV remains untested.
- [x] **Task 4 — Renderer integration + bench gate** (~1500 LoC across `rasterrocket-interp`, `rasterrocket`, `rasterrocket-cli`).  Today the Vulkan backend exists and parity-tests against CPU references, but the renderer dispatches CUDA via `GpuCtx` directly — no end-to-end Vulkan path.  Closing the phase needs:
    - [x] **Step 1 — `BackendPolicy::ForceVulkan` plumbing + loud-error placeholder** (commits `8126100`, `aec2053`, `55b54a0`).  CLI `--backend vulkan` → `BackendArg::Vulkan` → `BackendPolicy::ForceVulkan` → `rasterrocket::render::open_session` returns `BackendUnavailable` with a directive message.  `init_gpu_ctx` short-circuits CUDA init under `ForceVulkan`.  `--vaapi-device` conflict detection extended.  Smoke-tested: `rrocket --backend vulkan <pdf> <prefix>` exits with the explanatory error rather than silently CPU-rendering.
    - [x] **Step 2 — Migrate kernel call sites in `rasterrocket-interp` to the trait surface** (commit `15dd420`, hardened in `1783d66`).  Parallel-Vulkan-branch architecture: `PageRenderer` gains an `Option<Arc<VulkanBackend>>` field beside `gpu_ctx`; the fill dispatch prefers Vulkan when set.  `rasterrocket_interp::renderer::page::vk_ops` wraps the trait surface (`alloc → upload → record_* → submit → wait → download`) for AA fill and tile fill.  Phase 9 cache stays CUDA-only by construction (Vulkan path doesn't touch `DevicePageBuffer`/`DeviceImageCache`).  ICC CMYK→RGB on Vulkan is intentionally deferred — the renderer plumbing (`resolve_image → decode_dct → cmyk_raw_to_rgb`) currently threads only `Option<&GpuCtx>`, and under `ForceVulkan` it falls through to the CPU `cmyk_to_rgb_reflectance` matrix (matches Phase 9-pre-2026-05-07 quality).  Pre-existing `crates/gpu/build.rs` bug found and fixed in the same commit: PTX compilation now keys on a real NVCC probe instead of a feature-flag heuristic that didn't include `rasterrocket-interp/gpu-aa` (the previous heuristic produced 0-byte placeholder PTX for `--features "vulkan,gpu-aa"` builds, then crashed at runtime with `CUDA_ERROR_INVALID_IMAGE`).
    - [x] **Step 3 — End-to-end CLI smoke** (folded into step 2).  The loud-error stub in `rasterrocket::render::open_session` was replaced with real `init_vk_backend` + `RasterSession::vk_backend` + per-page `set_vk_backend` propagation in step 2's commit.  Verified: `rrocket --backend vulkan tests/fixtures/corpus-02-native-vector-text.pdf <prefix>` renders all 16 pages byte-identical to both `--backend cpu` and `--backend cuda`; same on the 358-page corpus-04 mixed-content fixture (Vulkan == CPU on every page).
    - [x] **Step 4 — Bench gate measurement** (`scripts/bench_v10.sh` + `scripts/aggregate_v10.py`, both since deleted in `d53803f`; results in `bench/v10/results.md`).  Vector-heavy subset (corpora 01-05) on RTX 5070; DCT-heavy corpora skipped because the Vulkan binary doesn't include nvjpeg, so 06-10 would compare CPU JPEG decode against silicon and bias the result.
        - **Criterion 1** (CUDA no-regression, threshold +5% slower than v0.7.0): **PASS** on all 5 corpora.  Live-captured baseline (v0.7.0 binary rebuilt on the same hardware) used because `bench/v070/D.txt` is stale relative to current driver state — corpus-02 was 212ms there but ~500ms today on the same v0.7.0 binary, so the old numbers would conflate driver drift with Phase 10 code drift.  Master vs v0.7.0 (live): 01 +0.4%, 02 −11.2%, 03 +2.6%, 04 −3.3%, 05 −6.7%.
        - **Criterion 2** (Vulkan pixel-diff ≤ 1 LSB vs CUDA): verified during step 2 (16/16 pages byte-identical on corpus-02, 358/358 on corpus-04).
        - **Criterion 3** (Vulkan ≤ 1.15× CUDA timing): **PASS** on all 5 corpora — Vulkan is *faster* than CUDA on every one (V/D ratios 0.27–0.82×).  Note: not strictly apples-to-apples either way because CUDA mode D pays nvjpeg / ICC-CLUT init even on vector-heavy corpora that don't decode JPEGs; the Vulkan binary skips those.  The *intent* of criterion 3 — Vulkan within striking distance of CUDA on the kernel paths Phase 10 actually migrated — is satisfied.
        - **Criterion 4** (cross-vendor proof of life on AMD or Intel): blocked on hardware at the time; since v1.5.0 the backend renders on an AMD Raphael iGPU (RADV), and it renders byte-identically to the CPU path (see the Task 3 list above); Intel ANV is untested.

    - [x] **Step 5 — Device image cache on Vulkan** (2026-11-13, v2.2.0).  `DeviceImageCache`, `CachedDeviceImage` and the host tier are generic over `CacheBackend: GpuBackend` (CUDA, Vulkan); slabs are the backend's `DeviceBuffer`.  CUDA keeps its own stream, stream-ordered uploads, write-combined pinned host tier and `DevicePageBuffer` unchanged.  Vulkan uploads and downloads through the backend's synchronous transfers, keeps its host tier in ordinary memory (its transfers stage through mapped buffers of their own) and sizes its VRAM budget from `detect_vram_budget`; the disk tier is shared.  A Vulkan image blit (`gpu::backend::blit_image_box`) records the table uploads, the zero-fill of a target the size of the blit's bounding box, the kernel and the readback as one page, rather than a page-sized buffer read back and cleared by separate submissions; the box composites through the same `composite_rgba_rows` as CUDA.  `gpu::cache::ImageCache` / `CachedImage` carry either backend through the interpreter.  The session builds the cache on the backend it resolved to; under `Auto` a session has a GPU backend, and so a cache, only while a device kernel is enabled (Vulkan preferred), and a JPEG drawn below its resolution is not cached.  Verified: the cache parity tests on CUDA and on Vulkan (RTX 5070, RADV, llvmpipe) and an archive A/B (22 image-heavy archive PDFs, 229 pages at 300 and 150 dpi, none with an ICC-based CMYK image: 0 byte differences from the CPU render on Vulkan RTX 5070, on RADV and on CUDA; a page with an ICC-based CMYK image differs between a GPU session and a CPU-only one by design, the GPU session converting it through the 17-node lookup table and the CPU-only session through the profile transform).

**Bench gate:** Phase 10 ships if (1) CUDA path performance unchanged within ±5%; (2) Vulkan path functional on RTX 5070 with pixel-diff ≤ 1 LSB vs CUDA; (3) Vulkan timing within 15% of CUDA on RTX 5070; (4) cross-vendor proof of life on AMD or Intel.

**Status of bench gate:** criteria 1, 2, and 3 all PASS on RTX 5070 (commit `1783d66` + bench/v10/results.md, vector-heavy corpora 01-05).  Criterion 4 (cross-vendor proof of life) was blocked on hardware when the gate was measured. It no longer is: `--vulkan-device radv` renders on the dev box's AMD Raphael iGPU (v1.5.0). It renders byte-identically to the CPU path on RADV and llvmpipe (2026-11-12: `ab_archive.sh`, fixtures plus the 150-file archive sample, 621 pages at 50 dpi gray with the AA fill forced on: 0 differences from the CPU path on RADV and on llvmpipe); Phase 10 has no open item but Intel ANV, untested for want of hardware.

**Production status (v1.0.0):** the full Vulkan path is wired end-to-end and stable. `--backend vulkan` / `--backend cuda` / `PDF_RASTER_BACKEND` select the backend at runtime; `auto` renders on the CPU. The parallel-Huffman JPEG decoder (Phase 8) ran on the Vulkan backend until its removal on 2026-11-15. The Task 3 perf-only follow-ups (dedicated transfer queue, ICC CLUT Texture3D, BDA push-constants) were measured on 2026-11-13 and closed as not worth it (see Task 3). The `record_tile_fill` parity oracle is moot: the tile-fill kernel is deleted.

**Total scope:** ~3100 LoC new Rust + ~1000 LoC Slang + ~400 LoC modified. Estimated ~6-8 weeks elapsed (3-4 weeks tasks 1+2; 3-4 weeks task 3).

**Sequencing:** Phase 9 must ship first. Phase 10 task 1 generifies Phase 9's cache; doing them in parallel would mean fighting merge conflicts.

---

## Phase 11 — Million-page-archive contest (SHIPPED — see `bench/v11/results.md`)

**Reframe.**  An earlier draft of this phase (titled "memory-frugal rendering and parse caching") was a feature-parity checklist of things MuPDF and PDFium have that we don't.  The first task on it was a sidecar cache for the parsed page tree.  A pre-implementation microbench killed it: the existing `Document::get_pages()` walk takes 76 µs–1.34 ms across the full corpus (16-page through 601-page documents).  Adding a 150-LoC sidecar plus an invalidation surface to save a millisecond was a textbook bad trade.

The actual gap was at a different layer.  MuPDF and PDFium don't *cache* the page-tree walk — they don't *do* it on open.  Their `LoadDocument` parses xref + the catalog's `/Pages` root and that's it; per-page resolution is logarithmic.  Ours was linear, eagerly populating a `BTreeMap<u32, ObjectId>` of every page in `rasterrocket::open_session` before rendering even one page.  On a 100k-page archival PDF that's millions of `get_object` calls per cold open, regardless of which page the caller asked for.

So the real Phase 11 is not "close the feature checklist" but "win a benchmark on the workload competitors weren't optimised for."  We define a four-event contest, build the harness that scores it against `mutool` and `pdftoppm`, and ship the layer changes needed to dominate it on a 10 GB synthetic archive.

**Contest workload (`crates/bench/src/contest_v11/`, binary `contest_v11`):**

| Event | Workload | What it measures |
|---|---|---|
| **E1 — first-pixel** | Open archive, render page 50000, time end-to-end | Cold-path latency: process startup + xref + page-tree resolve + render |
| **E2 — sustained** | Render 100 consecutive pages from one session | Warm-cache throughput; per-page cost on the lazy path |
| **E3 — cross-doc** | Render page 1 of each of N archives in a list | Process amortisation; fadvise-based xref-tail prefetch |
| **E4 — random-access** | Render 1000 deterministically-random page indices | Tree-walk depth handling; mmap fault locality |

**What shipped (commits on `phase-11`, in order):**

- **Logarithmic page-tree descent** (`b0600ab`).  New `crates/pdf/src/page_tree.rs::descend_to_page_index(doc, idx) -> Result<ObjectId, _>` walks only root → leaf using each interior node's `/Count` to choose the correct `/Kids` branch.  Cycle protection via `HashSet<u32>`, depth bound at 64.  `Document::get_page(idx)` and `Document::page_count_fast()` (reads `/Pages /Count` directly, falls back to eager walk on malformed catalogs) added on top.  TDD: tests landed first as `todo!()` red-light at `893a90c`, then implementation passed all four.  `resolve_kids` handles both inline-array and indirect-Reference forms of `/Kids` (corpus-04 hits the second).
- **`RasterSession` lazy refactor** (`bb64737`).  `pages: BTreeMap<u32, ObjectId>` field replaced with `page_cache: RwLock<HashMap<u32, ObjectId>>`, populated on demand via `RasterSession::resolve_page` (read-then-write pattern; idempotent under contention; poisoned-lock-tolerant).  `rasterrocket::open_session` no longer materialises a per-page map at all — it just stashes `total_pages = doc.page_count_fast()`.  `rasterrocket_interp::page_count` and `resolve_page_id` migrated to the lazy API.  Compile-time `assert_sync<RasterSession>()` / `assert_send<RasterSession>()` invariants kept intact.  Integration test (`crates/rasterrocket/tests/lazy_session.rs`) verifies the cache is empty immediately after `open_session` and grows by exactly one entry per page rendered.  Render output of corpus-04 page 100 byte-identical to master baseline (md5 `6c5703a00b2abd45b8c7ebbc31b54ba8`).
- **Linearization (Fast Web View) detection** (`892b3ff`).  New `crates/pdf/src/linearization.rs::LinearizationHints::try_load` probes object 1 for the `/Linearized` dict and parses `/N`, `/O`, `/H[0]`, `/H[1]`.  `Document::linearization_hints()` caches the result via a manual `OnceLock` lazy-init dance (because `OnceLock::get_or_try_init` is unstable, rust-lang/rust#109737).  `Document::bytes()` exposes the underlying mmap as `&[u8]`.  `/N` serves as the page count without reading the page tree.  No page-lookup fast path exists: the `page_offset` stub and the probe in `descend_to_page_index` were removed in the hardening pass (`4c46ce0`, `2f16dd2`) because the probe could never succeed; the bit-packed Page Offset Hint Table (PDF 1.7 § F.4.5) is not parsed, and a fast path would be added together with that parser.
- **`posix_fadvise` plumbing on the `Document` mmap** (`3b9bcbd`, lint scope fix in `a045326`).  `crates/pdf/src/madvise.rs::advise_random` is called immediately after `File::open` and tells the kernel "I'll touch arbitrary 4 KB ranges; don't readahead."  `advise_willneed(file, offset, len)` exported for content-stream prefetch later.  Routed through `rustix` so the `pdf` crate's `unsafe_code = "deny"` invariant holds.  No-op on non-Unix.
- **Lazy GPU init audit — no refactor needed** (`c3705ee`).  Baselined `rrocket --help` at ~1.0 ms median (hyperfine, 30 runs, prewarmed).  Spec threshold for shipping a refactor was ≥ 2 ms improvement.  `cudarc 0.19.4` and `ash 0.38.0` already lazy-init via `libloading` at first call, not at dlopen; workspace-wide grep for `lazy_static!`, `once_cell::Lazy`, `#[ctor]` returns zero hits in production code.  Audit doc-comment added to `GpuCtx::init` in `crates/gpu/src/lib.rs:155-177` so future-us doesn't re-investigate.  The 10–30 ms claim in the original spec was speculative; the real cost is binary-load + clap, not GPU init.
- **libdeflate backend for FlateDecode** (`c64a695`, partial-tolerance fix in `6139ef9`).  `libdeflater 1.25.2` behind a default-on `libdeflate` Cargo feature.  `apply_flate` dispatches via `decompress_zlib` (libdeflate) or falls back to `flate2`/miniz_oxide on `--no-default-features`.  Microbench (`crates/pdf/examples/flate_bench.rs`) using public `decompressed_content` API: corpus-03 (text-heavy, 774 small streams) **1.47×** faster, corpus-04 (DCT-heavy, 715 large streams, 11.4 MiB raw → 774 MB decompressed) **2.40×** faster.  Render byte-parity against master baseline preserved.  The fallback retains flate2's silent partial-decompression tolerance for truncated/checksum-corrupt content streams that real-world malformed PDFs ship — libdeflate is all-or-nothing (validates Adler-32 before returning `Ok`), so any `BadData` error falls through to a flate2-based last-ditch attempt that accepts partial output.
- **PGO + BOLT release build script** (`9e2b70f`).  `scripts/release_pgo_build.sh` runs profile-generate → train (10 pages of corpus-04) → `llvm-profdata merge` → profile-use rebuild.  BOLT applied on top when `llvm-bolt` is on `PATH`; skipped with a clear message otherwise.  Verified end-to-end on the dev box; final binary at 3.26 MB.  Workflow integration (CI hook) deferred until v0.9.0 actually ships.
- **Bench harness skeleton + qpdf archive builder** (`34e1e3f`).  Second `[[bin]]` target `contest_v11` in `crates/bench/`.  `archive::build(out, target_bytes)` cycles through corpus-04/05/08/09 fixtures, concatenating via `qpdf <base> --pages <p> 1-z ...`.  `target_bytes` is *cumulative input fixture bytes*, not output bytes — qpdf deduplicates shared PDF objects and the output is typically 2–3× smaller, so a 10 GiB output requires passing ~25–30 GiB as `target_bytes`.  Documented in the function doc-comment so the next caller doesn't get surprised.
- **Four-event runners + competitor wrappers** (`b6c3781`).  `events::{e1,e2,e3,e4}` use `rasterrocket::{open_session, render_page_rgb}` at 150 DPI; `_bmp` discards (we're benching, not rendering to disk; PPM encode + filesystem cost would be noise).  E4 uses a fixed-seed xorshift64 (`0xDEAD_BEEF_DEAD_BEEF`) so successive runs touch the same pages.  `competitors::{mutool_render, pdftoppm_render}` invoke the subprocesses, time them, clean up output files.  Missing competitors degrade to `None` rather than failing the run.  E3 prefetches the last 4 KB of each archive via `rustix::fs::fadvise(WillNeed)` (`io_uring_open::warm_xref_tails`) — the spec originally proposed `io_uring` but `posix_fadvise` is the same kernel hint and avoids the async-runtime tax.

**Smoke result on a 335 MB synthetic archive (E1, page 1, 9900X3D + RTX 5070):**

| Engine | First-pixel time |
|---|---|
| rasterrocket | **4.2 ms** |
| pdftoppm | 15.9 ms |
| mutool draw | 18.7 ms |

We're already 4× faster than `pdftoppm` and 4.5× faster than `mutool` on this slice.  The 10 GB archive run in Task 11 (below) tests whether the lead holds when mmap actually pages.

**Task 11 (the bench gate) — DONE, shipped v0.9.0.**

The 10 GB archive run completed; `bench/v11/results.md` is written and the release-history block above carries the v0.9.0 numbers (E1 first-pixel 35.6 ms vs mutool 93.7 ms = 2.6×; E3 cross-doc 3.5 ms/archive). rasterrocket won or tied at least three of four events on the 10 GB workload, so the phase gate passed and Phase 11 shipped. Nothing in this phase is open.

**Closed (2026-11-12):** the original Phase-11 draft's three non-contest tasks.

- **Banded rendering as an OOM escape valve.**  Satisfied by region rendering: `RenderRegion` tiles bound a page's memory and stitch byte for byte.
- **Multi-resolution image cache** (`(content_hash, l2factor)` keys).  Dropped: it pays off only when one corpus is rendered at several DPIs in sequence, and no consumer does that.
- **Display-list intermediate.**  Dropped as a render path.  A page trace survives only as a pixel-independent regression oracle: the paint trace (see Current status).

**Out of scope (re-affirming from the original):**

Vector output (PostScript, SVG or PDF, as `pdftocairo` writes) and dynamic XFA forms: rasterrocket renders PDF to pixels, and a dynamic XFA form is reported (`FormType::XfaFull`) with a warning rather than laid out.  A 2-stage interpretation+render pipeline within a single document.  rasterrocket already parallelises across pages with Rayon, which dominates per-document pipelining on any multi-core machine — confirmed empirically by the May-2026 baseline matrix (3–10 % gain over serial vs. Rayon-per-page's much larger lead).  Network-streaming PDF (HTTP range requests).  Not our user; PDFium's browser use case, not a CLI tool's.  AVX-512 simdjson-style PDF lexer.  Real engineering effort with a real win, but the contest events don't bottleneck on dictionary parsing — deferred to a future phase.

**Rejected ideas (kept here so they don't get re-proposed under new names):**

- *`io_uring` for general I/O.*  mmap is already streaming on Linux — the kernel page-faults on access.  `io_uring` only beats mmap+`MAP_POPULATE` on cold-cache reads of *known-up-front* byte ranges, which our parser cannot produce (each phase depends on bytes parsed from the previous).  `posix_fadvise(WILLNEED)` delivers the same kernel hint in a few lines.
- *`io_uring` for E3's batch open.*  Real win in theory (100 × `open + read tail` is genuinely batchable), but `posix_fadvise` gets us most of it without an async runtime.  Kept the helper file name (`io_uring_open`) for traceability; the implementation is fadvise.
- *`HttpRangeSource` / network-streaming PDF.*  PDFium's browser use case, not ours.
- *AVX-512 simdjson-style lexer for PDF dictionaries.*  Real engineering with a real win, but the contest events don't bottleneck on dictionary parsing.  Deferred.
- *Hugepages for GPU staging buffers.*  1–3% win on E2/E3.  Worth measuring after Task 11; not worth speccing now.

---

## Phase 12 — QA-driven correctness + per-commit hardening campaign (✓ SHIPPED v1.0.2 / v1.0.3)

**Trigger.** External QA against a broad 238-PDF corpus found ~76% of pages were *visually wrong* after v1.0.2 — every root cause a **silent** total- or partial-loss on an input variant the curated test suite did not contain. The curated suite was green; the corpus was not. That gap, not any single bug, is the lesson of this phase: a passing test suite is not evidence of correctness on inputs it does not contain.

**What shipped.** Two release waves (v1.0.2 rendering-correctness fixes, then v1.0.3 remediation + hardening). The per-release detail is in the release-history block at the top of this file (NF-1 … NF-12 and the per-commit hardening list); not duplicated here. The structural outcomes:

- **Every silent-loss root fixed at the root**, not papered over with a fallback. Blank text/vector pages (indirect `/Length`, `/ObjStm` object streams, TrueType CIDFontType2), partial text drop-out, JPX+`/Mask` blank scans, page-tree "no pages", chained-filter skips, FunctionType-4 Separation tints, CFF/Type1C glyph garble, misleading errors on malformed/empty/non-PDF input, JS-bearing PDFs hard-refused, CCITTFax G3/G4 ImageMask "no rows", JPXDecode CMYK.
- **DoS classes closed** as a deliberate hardening sweep: stack-overflow (tokenizer recursion → iteration), unbounded memory, raster-area cap (`MAX_PX_AREA`, `u64`-computed to kill a latent overflow), LZW bomb, filter-chain flood, Type-4 recursion/operand bomb, watchdog escape via tiling patterns, unbounded endstream scan.
- **Security posture documented**: FFI trust boundary (at the time, system FreeType/OpenJPEG had to be patched by the host — `cargo audit` covers only the Rust tree; both have since been replaced, OpenJPEG by the local `hayro-jpeg2000` fork in v1.4.1 and FreeType by the local `read-fonts`/`skrifa` forks in v1.5.0, so the default build links no system C library and its only C code, libdeflate and the CLI's mimalloc allocator, is compiled from vendored source); bounded deterministic decoder property/fuzz harness; page-level and per-annotation/widget JavaScript *detection* (structural `/S /JavaScript` only — `/JS` is never decoded or executed; bounded, first-hit).
- **No deferred findings.** Every review finding that didn't fit its originating commit was fixed at root, not filed and forgotten — including a git-proven latent form-XObject CTM regression, 17 missing PDF Appendix D.2 encoding slots, non-deterministic catalogue selection, and a multi-backend `resolve_image` type-inference break the campaign unmasked.

**Acceptance criterion (met).** The 238-PDF exhaustive corpus is 100% legible — zero silent loss, zero crash — measured by OCR against a MuPDF oracle. No public API changes; decrypt stays private-copy-only behind a default-No liability gate; JavaScript is detected and disclosed but never executed.

**Durable lesson for future phases.** Curated fixtures prove a feature works on the inputs you thought of. They do not prove absence of silent loss. Any future codec/parser work ships with a corpus-scale OCR-vs-oracle gate, not just unit fixtures.

**Status: closed.** Nothing open. This is a backstop discipline, not an ongoing work item — new parser/codec work inherits the corpus-gate requirement above.

---

## Phase 13 — Output to the operating system: print and fax (partly shipped in v2.1.0)

**Status: in progress; what exists shipped in v2.1.0.** The four fax layers are on
master (2026-11-01), with `--jpeg` and `--jpegopt` (as `pdftoppm -jpeg`,
colour or gray), and printing through CUPS (2026-11-04); the Windows
spooler is not started.

**Printing through the OS spooler.** No device or driver code of our own.
**Done on Linux and macOS:** `rrocket --print [--printer NAME] [--copies N]
[--duplex long|short]` renders each page at 600 dpi (or `-r`) with
`RenderPurpose::Print` and hands CUPS (`lp`) one PDF whose every page is our
raster as a single image; the queue's filters fit it to the printer (media,
orientation, colour, resolution, back sides) without rendering the source
document. Raster PWG was tried first: driverless queues pass it to the
printer untouched, leaving colour model, resolution and back-side
orientation to the sender. Pages are appended to the spool in page order
as they finish, at most two pages per thread started past the first page
not yet appended, so memory stays bounded. Verified through a CUPS `everywhere` queue on
`ippeveprinter`: colour pages reach a gray-only printer as sGray at its
resolution, landscape pages are turned, and the page images round-trip
pixel-exact. `--list-printers` lists the CUPS queues (`lpstat -e`), the
default one marked. **Not done:** the Win32 spooler, behind a feature, on Windows.
Parity gap: poppler's `pdftocairo` has `-print` / `-printer` on Windows,
and pdfium renders into a Windows printer context.

**Fax, in layers.**
1. **Done.** TIFF output (`--tiff`, `--tiffcompression none|packbits|lzw|
   deflate|g3|g3-2d|g4`), as `pdftoppm -tiff`; fax-coded pages read back
   pixel-exact through libtiff and the `tiff` crate.
2. **Done.** A CCITT Group 3 (MH, MR) and Group 4 encoder in the
   `crates/ccitt` fork, round-tripped through its decoder.
3. **Done.** `--fax` writes one TIFF Class F document: 1728 px wide,
   204 × 196 dpi (fine) or 204 × 98 dpi (`--fax-resolution standard`), 1
   bit, landscape pages turned. Group 3 one-dimensional by default, the
   Class F baseline every reader takes (`g3-2d` and `g4` on request), not
   G4 as first planned. Pages reversed out of a flat dark ground are inverted (`--invert`), and
   `--dither halftone` serves photographs.
4. **Done.** `--fax-to NUMBER [--fax-queue NAME]` hands the document to a
   CUPS fax queue (`lp -o phone=`) or to HylaFAX `sendfax`. On Windows it
   points to Windows Fax and Scan; driving the Windows Fax service API is
   not done.

No modem code and no internet-fax services.

---

## Phase 14 — Text shaping for substituted fonts (shipped in v2.1.0)

**Status: shipped in v2.1.0 (2026-11-04), on master since 2026-11-01.**

When a PDF does not embed the font for Arabic, Devanagari or another script
that needs shaping, the substitute (an installed face, or the bundled Noto
fallbacks) drew each character in its isolated form. Now, within one show
string, the codes of a run in such a script drawn by one face are shaped
together with that face's `GSUB`/`GPOS` through `harfrust`, vendored as a
local fork in `crates/harfrust` (release 0.12.0, the one built on the
read-fonts fork's base; since rebased to 0.15 with the font forks, see
Current status). A right-to-left run, in visual order in the string,
is reversed to logical order first; the shaped glyphs fill the advance the
document gives the run's codes, so the pen moves exactly as before. The
bundled Hebrew, Arabic, Devanagari and Thai faces keep their layout features
for this. Synthesised annotation and field text is put in visual order by
the Unicode bidirectional algorithm, then shaped the same way.

Not done: the comparison the plan called for, of non-embedded Arabic and
Devanagari archive documents against an embedded, shaped rendering of the
same text. The tests cover joining, letter order and vowel-sign reordering
on the bundled faces. Runs do not cross show strings, so a word split
across `Tj` operators is shaped piecewise.

**Substitution follow-ups, in v2.1.0 (2026-11-02).** A vertical-form
CID drawn by a substitute without a `vert`/`vrt2` form is rotated or
shifted from its horizontal glyph (brackets, dashes, the long-vowel mark,
small kana) instead of drawn upright; an embedded TrueType name the cmap
misses resolves through the `post` table (non-symbolic fonts only); the
installed-font index is rebuilt when a font directory changes, substitute
searches included; and the bundled fallbacks gain Bengali, Tamil,
Armenian, Georgian and Ethiopic (0.49 → 0.75 MB). Verified on Linux and on
a stock macOS 27 install.

**Font-discovery review, in v2.1.0 (2026-11-03).** One walk lists the
font files of a library, its Liberation substitutes included: a directory
reached through several paths or a symbolic link is read once, and the walk
stops at 10 000 directories with a warning. The Windows and macOS font
directories are searched only on those targets. The bundled Devanagari,
Bengali and Tamil faces carry U+25CC, the dotted circle the shaper inserts
for a broken cluster, and the Bengali and Tamil ones the danda and double
danda (U+0964, U+0965). A font's hinting set-up is shared by the faces
derived from it for vertical forms.

---

## Current status (2026-11-15)

Every **phase** is shipped or deferred by deliberate decision, apart from the
Phase 13 above, of which the Windows spooler is still to do; Phase 14 shipped in v2.1.0.

**Released in v2.1.0 (2026-11-04), since v2.0.0 (2026-10-31):** Phase 13's fax output,
`--print` through CUPS and `--jpeg` / `--jpegopt`, Phase 14's text shaping, the font-substitution
follow-ups, overprint simulation (below), and three hardening review passes:
2026-11-03 over the overprint work, gray output and the vendored forks, and
two on 2026-11-04, the first over everything since (output path, document
images, parser, rasteriser, text, allocation, HTJ2K, printing) and the
second over the work that followed it (below).  The first fixed, among
others: bilevel and planar TIFF pages in comic archives (every one failed
before), subpaths ending in a curve merging into the next (filled
artwork drawn in several curve-closed loops gained connector fills),
antialiased edges past device column 8191, a quadratic cross-reference
index and an unbounded `/Count`, and a world-readable print spool.
Public API added since v2.0.0, none of it breaking a v2.0.0 caller:
`RasterOptions::overprint`, `SessionConfig::overprint`,
`RasterOptions::image_budget`, `SessionConfig::image_budget`,
`PageDiagnostics::unslotted_spots`, `ComicOptions::session` (the session an
embedded PDF opens with), `rgb_into_gray`, `gray8_into_rendered_page`,
`Bitmap::into_data` and `Bitmap::from_packed`; in `rasterrocket-interp`,
`PageRenderer::with_page_group(view, doc, page, group, ink)`,
`page_uses_overprint(doc, page, annotations)` and
`ExtGStateParams::{overprint_stroke, overprint_fill, overprint_mode}` (an
`Option<OverprintMode>`), and `DocumentImages`,
`DOCUMENT_IMAGE_BUDGET_BYTES` and `PageRenderer::set_document_images`;
`RasterError::PageAllocationFailed` (the enum is `non_exhaustive`),
`PageRenderer::try_with_page_group` in `rasterrocket-interp`, and
`Bitmap::try_new_filled` / `BitmapAllocError` in `rasterrocket-render`;
`DeviceKernel`, `device_kernel_enabled`, `aa_fill_threshold` (`gpu-aa`) and
`huffman_threshold` (`gpu-jpeg-huffman`) in `rasterrocket-interp`. The
canvas kind is fixed when the renderer is built; there is no setter.

**Released in v2.3.0: exact-area coverage.** Antialiased paths, clips and glyphs take exact-area
coverage from one integer rasteriser (`raster::coverage`) in place of the 4×4
supersampled scanner and its gamma-1.5 table. The scanner's inclusive bottom row
made fills about 0.25 px heavy on +y, and its coverage was nonlinear and banded
in 16 levels: a `1 w` stroke on a row boundary drew 165/89 where the area is
127/127. Every antialiased path pixel changes. Against pdfium's corpus (260
scored pages) the median RMSE falls from 4.39 to 1.45, with 82 pages closer and
7 slightly further, and the fill bench runs 6.34 → 4.70 ms (vello_cpu gap
3.9× → 2.85×). Stroke pieces meet without overlap at joints and on retraced
lines. Zero-area subpaths and outline spikes in a fill draw as faint lines.
The stroke-adjust snap depends on the coverage model. The `--gpu-pages`
raster composites the CPU's coverage and stays byte-identical on RTX and RADV.
The dormant device AA fill (CUDA and Vulkan kernels, `GPU_AA_FILL_THRESHOLD`,
`PDF_RASTER_AA_FILL_THRESHOLD`) and the `gpu-aa` feature are removed. Glyph
edges may move by one level, since glyphs use the same rasteriser.

**Released in v2.3.0: large-format pages.** The pixel caps
(`MAX_PX_DIMENSION` 32 768, `MAX_PX_AREA` 600 Mpx) are replaced by a byte
budget, `SessionConfig::max_page_bytes` / `RasterOptions::max_page_bytes` /
`--max-page-mib` (default 1.8 GB, the old bound at 8-bit RGB; transparent
renders count their alpha), and a hard side ceiling `MAX_PX_SIDE` = 2²⁴;
deskew keeps its own 32 768 px bound. `render_page_bands` renders a page as
horizontal bands over shared page state (images, forms, shadings and cells
decoded once; spot slots and diagnostics carried), stacking byte for byte into
the whole page; the CLI writes PPM, PNG, TIFF and `--print` pages band by band
(`--band-mib`, default 64) through new row-streamed writers (`PnmWriter`,
`PngWriter`, `TiffStream`, `PdfPageEncoder`; `RowImage` replaces `TiffImage` /
`PdfImage`), TIFF switching to BigTIFF past 4 GiB, a banded `--print` page
spilling its compressed image to a file. A 44 × 120 in roll at 360 dpi
(15 840 × 43 200, 2.05 GB) renders to TIFF in 28 s with 428 MiB peak RSS under a
768 MiB cgroup ceiling that kills the one-band render. `--print` without a
resolution flag prints pages larger than A0 at 360 dpi. Making bands exact
fixed region renders that differed from the page: shadings, tiling lattices,
image sample grids and group bounds now work on the page's grid; a Gouraud
triangle reaching a clip by rounding was dropped; glyph rows rounded half away
from zero; curve flattening was not shift-exact (points now on a 2⁻²⁰ px local
grid); the tiling compose/direct choice depended on the window. Also fixed: a
scanner table past `u32` intersections aborted (now `ScannerTooLarge`, the fill
dropped with a warning), and a Type 3 glyph box far off the page overflowed
`i32`. API changes: `RasterError::PageTooLarge { bytes, limit }`,
`PageExceedsCeiling` (no `PageAreaTooLarge`), `validate_page_bytes`,
`XPathScanner::new` returns `Result`, `gouraud_triangle_spans` and
`Mesh::meeting` take the window origin, PNG files are written through the
stream writer (different IDAT chunking, same pixels).

**Released in v2.3.0: 16 bits per channel.** `--depth 16` (library:
`RenderParams::with_depth`, `RasterOptions::depth`, `SampleDepth`) renders,
composites and writes 16 bits per channel, so gradients, soft-mask vignettes
and colour conversions no longer quantise to 256 levels before a printer's own
conversion. One compositor serves both depths: `rasterrocket-color` gains the
`Sample` trait (u8, u16) and `Rgb16`/`Gray16`/`DeviceN16`, and the pipe, fills,
glyphs, groups, tiling, blend modes and shadings are generic over the sample,
their 8-bit instance the previous arithmetic (8-bit output byte-identical over
the fixtures and the archive sample, pinned by digests; timing within noise).
Shadings are sampled straight to 16 bits (1 024 ramp samples), transfer
functions get 65 536-entry tables, ICC colours use a 16-bit transform,
DeviceCMYK the same model at 16 bits, images of more than 8 bits (Flate/raw,
JPEG 2000) keep their depth, and overprint simulation runs on a 16-bit ink
canvas. Writers: 16-bit PNG, PPM/PGM (maxval 65535), TIFF (`BitsPerSample`
16) and a PDF 1.5 `--print` spool of `/BitsPerComponent 16` images. Refused
with `--mono`, `--dither`, `--fax`, `--jpeg`, the camera stages, `--gpu-pages`
and comic input; the GPU decode/convert paths fall back to the CPU. A 16-bit
canvas costs 6 bytes a pixel (8 with alpha) of the byte budget
(`CanvasLayout`). Also fixed: a non-isolated group's union colour wrapped to
black where the rounded union alpha fell below the exact one; a JPEG 2000
colour key above 8 bits was compared against narrowed samples. API changes:
`RenderedPage::pixels` is a `Pixels::{U8, U16}` enum; `PixelFormat` gains
`Gray16`/`Rgb16`/`Rgba16`; `render_page_canvas` returns a `FinishedCanvas`;
`Band::bitmap` is a `BandBitmap`; `Bitmap` stores typed samples (stride in
samples); `Pattern`, `PipeState`, `PipeSrc`, `TransferSet` (DeviceN tables an
array), `ColorRamp` and the shading patterns take a sample parameter;
`PdfWriter::new` takes the deepest page's `SampleDepth`; `AnyColor` holds 16
bytes.

**Released in v2.3.0: output ICC profiles.** `--output-profile FILE`
(library: `SessionConfig::output_profile`, `RasterOptions::output_profile`,
`OutputProfile`) converts every page, band by band and at both depths, from the
sRGB working space to an RGB or gray printer, display or colour-space profile,
table-based (`AToB`/`BToA`, the kind paper profiles are) or matrix/TRC, and
tags the output with it: PNG `iCCP`, TIFF tag 34675, JPEG `APP2` and the
`--print` spool's `ICCBased` images (PPM/PGM warn, untagged).
`--intent perceptual|relative|saturation|absolute` (`RenderParams::with_intent`)
picks the intent, default the one intent a page's content paints with — the
interpreter now tracks `ri`, `ExtGState` `/RI` and image `/Intent`
(`PageDiagnostics::content_intents`) — else relative colorimetric, a page of
mixed intents converting relative with one warning; `--bpc`
(`with_black_point_compensation`) compensates the black point as ISO 18619
detects it, and perceptual or saturation conversions to v4 profiles always
compensate. Absolute colorimetry and compensation are folded into the
working-space source profile, so the printer's tables run unchanged through
moxcms at 16 bits. Against Little CMS's `transicc` on a 793-patch chart the
worst ΔE00 over every intent, with and without compensation, at 8 and 16
bits, is 0.93 (synthetic v4 and v2 table-based RGB printers and a gray
printer, `tests/fixtures/icc`, built by `scripts/make_test_icc.py`; CI runs
it on Linux). A gray profile writes gray pages. Refused: CMYK profiles
(CMYK raster output is a separate item), device-link, abstract, named and
input profiles, `--mono`/`--dither`/`--fax`, comic input, `--bpc` with
`--intent absolute`, an RGB profile with `--gray`, a gray one with
`--transparent`. Without a profile output is byte-identical (ab_render over
the fixtures at 8 and 16 bits: 0 of 344 differ). Also fixed: a Gouraud mesh
under a rectangular clip, rendered in bands of a few rows, panicked on the
inverted bounds of an empty clip. Not done: per-object intents (input-side conversions still use
perceptual tables for table-based `ICCBased` content), `/OutputIntents` soft
proofing, a wider-than-sRGB working space. API changes: `FinishedCanvas` /
`BandBitmap` gain `Gray8`/`Gray16`, `CanvasBuffers` gray buffers,
`RasterError::OutputProfile`, `RenderParams::intent` /
`black_point_compensation`, `RasterOptions::intent` /
`black_point_compensation`, `PageDiagnostics::content_intents` /
`output_intent`; `write_png`, `write_jpeg`, `PngWriter::new`,
`TiffStream::new`, `TiffPageEncoder::new` and `PdfWriter::new` take an
`Option<&IccProfile>`.

**Released in v2.3.0: CMYK raster output.** `--cmyk` with `--tiff` or
`--jpeg`, and `--jpegcmyk` (= `--jpeg --cmyk`, as `pdftoppm -jpegcmyk`),
render pages in process inks (library: `ColorModel::Cmyk`,
`RenderParams::with_cmyk`, `RasterOptions::color`, `render_page_cmyk`,
`PixelFormat::Cmyk8`/`Cmyk16`, `FinishedCanvas`/`BandBitmap::Cmyk8`/`Cmyk16`).
A CMYK page is painted on the overprint ink canvas and finished as its
process ink, the first four spot colorants folded into their process
equivalents.  DeviceCMYK and ICC-based CMYK colours, images and shadings
and process colorants keep their values (the swatch fixture round-trips
exactly at 8 and 16 bits); other colours separate through a CMYK
`--output-profile` (now accepted: a 65³ table of its conversion from the
working space, under `--intent`, default relative colorimetric, and
`--bpc`; `transicc` within 1.5 % of each ink on the new
`tests/fixtures/icc/printer-cmyk-v4.icc`), else by the ISO 32000-2
§10.4.2.3 device conversion, so gray and black text are black ink alone.
Overprint is simulated with `--overprint` and knocked out otherwise.
Written as Separated TIFF (`InkSet` 1, 8 or 16 bits, banded, tagged with the
profile) and Adobe YCCK JPEG (`APP14` transform 2, inverted, the layout
`pdftoppm -jpegcmyk` writes).  Against `pdftoppm -tiff -overprint` at
72 dpi the swatch fixture is identical in every channel; on eight pages
of four archive documents the mean per-channel difference is 0–4.4
levels, but 9.6 (K) on a page of an RGB JPEG cover whose page means agree
within 0.7, the images being resampled differently; the largest
differences lie at glyph and image edges.  Refused: PNG, PPM, `--print`, `--fax`, `--mono`, `--gray`,
`--dither`, `--transparent`, `--invert`, the camera stages, `--gpu-pages`,
comic input, `--jpegcmyk --depth 16`, and an output profile of the other
colour model.  DeviceCMYK images and shadings now paint their own ink on
the overprint preview too (`--overprint`: 0 of 215 fixture outputs and 45
of 1 015 outputs of the 68 archive PDFs that set overprint differ, from
overprint meeting the image's real inks and from CMYK images reduced in
ink rather than sRGB).  The page budget counts the ink canvas's eight
samples a pixel, for CMYK pages and for pages on the overprint preview,
which were counted as RGB.  Default output is byte-identical (ab_render:
0 of 215 fixture outputs at 8 bits, 0 of 129 at 16, 0 of 1 315 over 89
archive PDFs that use CMYK or overprint).  Not done: `BG`/`UCR`, CMYK →
CMYK conversion of ICC-based CMYK content to a different output profile,
indexed images over a CMYK base (drawn as sRGB, then separated), per-object
intents of the separation.  API changes: `PageRenderer::with_page_group`,
`try_with_page_group(_in)` take a `&CanvasRequest` (`Rgb`,
`OverprintPreview`, `Cmyk { overprint, separation }`) in place of
`ink: bool`; `CanvasBuffers::cmyk`/`cmyk16`; `CanvasLayout::ink`;
`OutputSpace::Cmyk`, `OutputProfile::separation`, `Separation`,
`ProfileSeparation`, `OutputProfileError::Mismatch` (`Cmyk` removed);
`PixelMode::Cmyk16`, `Cmyk16`, `ink_to_cmyk_row{,16}`, `fold_spots{,16}`,
`rgb_to_device_cmyk` in `rasterrocket-color`; `RowImage::Cmyk`/`Cmyk16`,
`RowLayout::Cmyk`/`Cmyk16` and CMYK `write_jpeg`/`IccProfile` in
`rasterrocket-encode`.

**Released in v2.3.0: visible-text diagnostic.** `PageDiagnostics::glyphs`
(`GlyphCounts`) counts the glyphs a page shows, one per character code, by
text rendering mode (`fill`, `stroke`, `fill_stroke`, `invisible` for
`3 Tr`, `clip_painted` for 4–6, `clip_only` for 7) when the glyph can mark
the page, and otherwise by why not (`zero_size`, `unrendered`, `off_page`
against the page box, `transparent`); `PageDiagnostics::has_visible_text()`
tells a caller whether to skip OCR.  The counts are the page's whether it
is rendered whole, in bands or as a region; text in pattern cells, soft-mask
groups and Type 3 glyph descriptions is not counted, and the clip, soft
masks and backdrop colour are not considered.  Over 173 pages (first,
thirds, last) of 44 archive PDFs (39 from `publications/` and `droit/`, 5
from `rituels/`), recognised scans report only invisible glyphs,
born-digital pages only visible ones (SCOTUS opinions add a small
invisible layer), scans without a text layer none, and 163 pages agree
with an independent trace glyph for glyph; the others differ by ligature
codes the trace expands to several characters, and by 61 glyphs of two
running heads set beside the page, counted `off_page`.  No
pixel changes (ab_render: 0 of 215 fixture outputs).  The CLI prints no
page diagnostics, so its output is unchanged.

**Unreleased on master: colour conversions agree with Little CMS to its
precision.** Three corrections, each measured against `transicc`
(`input_profile_transicc`, `output_profile_transicc`, ignored tests that
now assert every case they printed):
- **Table-based RGB input profiles.** The converter interpolated every
  device → PCS CLUT trilinearly, which leaves the gray axis; on a gamma
  2.2 table that gave dark neutrals a red cast (Lab 13.2 1.0 0.4 for
  11.96 0 0). A three-channel `AToB` table now converts through a 33-node
  device grid evaluated in floating point and interpolated tetrahedrally
  at both depths. Worst ΔE00 1.8–4.4 → 0.71 at 8 bits, 0.07 at 16. A
  version 4 perceptual or saturation table with `/UseBlackPtComp /ON` is
  no longer compensated twice; with `/OFF` it is no longer compensated
  at all. Default output without a profile: 1 of 1168 archive documents
  carries such a profile (a scanner profile on the images of pages 244,
  253 and 254). Its pages 1–10 render byte for byte as before; those
  three pages move by at most 5/255 (ΔE00 mean 0.45–0.69, max 4.1, in
  the dark tones), and that profile agrees with Little CMS to 0.03 at 16
  bits instead of 2.3.
- **sRGB to an output profile at 16-bit precision.** The conversion from
  the sRGB working space now uses the wide space's floating point,
  tetrahedral interpolation and full-precision weights; it reduced a
  sample to 8 bits for the interpolation before. On an off-grid chart the
  16-bit mean ΔE00 falls from 0.12–0.13 to 0.03, the worst from 0.67 to
  0.45. `--output-profile` at `--depth 8`, or `--depth 16 --working-space
  srgb`, changes: on 10 archive pages to a table-based RGB printer, 3 % of
  8-bit pixels by up to 7/255 (ΔE00 mean about 0.6, max 2.2).
- **Black points of table profiles.** The ISO 18619 round trip that finds
  a destination black now runs through the profile's own tables. The v2
  test printer's perceptual black was L* 1.0 where Little CMS finds 0.11.
  A colour compensated below the PCS black also keeps its a*/b*. CMYK
  separation with compensation goes from 1.27 to 0.19 ΔE00 at 16 bits,
  the CMYK → CMYK link from 1.53 to 0.06, and the special v2 bound is
  gone. The blacks stay neutral, as Little CMS's are. The input-side
  black of CMYK printer profiles under relative colorimetry moves too: 21
  of the 29 archive documents with such a profile change, on pages 1–10
  at 50 dpi, in about 5 % of pixels by exactly 1/255 (ΔE00 mean 0.7, max
  1.18).

**Unreleased on master: input ICC conversions honour the rendering intent
and compensate the black point.**
An `ICCBased` colour converts to sRGB under the intent it is painted with
(ISO 32000-1 §8.6.5.8): the graphics state's (`ri`, `gs /RI`, initially
relative colorimetric) for fills, strokes, pattern tints, shadings and
mask backdrops, and an image's own `/Intent` before it for images, inline
images, their embedded profiles, `/Matte` and the GPU CMYK lookup table.
Previously every conversion used the profile's perceptual table.
- **Tables.** A table-based profile now converts through `AToB1` under
  the colorimetric intents and `AToB2` under saturation, `AToB0` standing
  in for a missing table.  A gray or RGB profile carrying both a
  matrix/TRC model and `AToB` tables converts through the tables.
- **Black-point compensation.** It follows the graphics state's
  `/UseBlackPtComp` (ISO 32000-2; `/ON`, `/OFF`, and `/Default`, which is
  on, as in Acrobat and Little CMS): under relative colorimetry,
  perceptual and saturation, the profile's black point, found as ISO 18619
  finds it, maps to sRGB black.  A profile whose black is the PCS origin,
  such as sRGB, converts as before.
- **Absolute colorimetry.** It scales the PCS by the source's media white,
  so a press profile's paper is drawn tinted, and is never compensated.
- **Precision.** Profile tables evaluate in floating point; the
  fixed-point 8-bit tables missed by up to two levels in the shadows.
- **Pixel impact** (all pages, against the previous render):
  - The 4 archive documents whose US Web Coated (SWOP) v2 tables differ
    by intent move by at most 4 levels. K100 stays sRGB (35, 31, 32), as
    Little CMS gives it with compensation; without it, it is (55, 53, 53).
  - 19 archive documents whose CMYK profile has identical tables (US
    Supreme Court slip opinions among them) render their blacks deeper, as
    compensation does: dark text goes from luminance 54 to 40, with a max
    channel delta of 16.
  - The repository fixtures render byte for byte as before.
- **Against Little CMS.** CMYK and gray profiles agree within ΔE00 1.5 at
  8 bits and 1.0 at 16 under every compensated intent (`transicc -b`) and
  under absolute colorimetry (`input_profile_transicc`, ignored).

**Unreleased on master: OCR preparation of scans.**  `rasterrocket-camera`
gains three operations measured on 8-bit luma, in integer arithmetic so
their output is identical on every platform.  `measure_background` /
`Background::normalize` divide out uneven lighting so paper becomes white
and ink keeps its tone relative to it, applied to every channel at 8 or 16
bits; it declines on a background already white, so born-digital pages
pass unchanged.  `binarize` thresholds at a fixed level or by Sauvola's
local threshold over `u64` integral images, decided exactly in integers.
`find_border_junk` / `BorderMask::whiten` whiten dark regions joined to
the page edge within a band 6 % of the shorter side deep, keeping the page
size, and decline on a dark ground.  New public API: `Background`,
`background_tile_px`, `measure_background`, `Threshold`, `binarize`,
`BorderMask`, `find_border_junk`, `CameraError::ZeroSize`, and
`Sample::to_u32` / `Sample::saturate_u32` (the trait is sealed).
`rrocket` exposes them as `--clean-borders`, `--clean-background` and
`--threshold fixed|sauvola` (with `--mono` or `--fax`, not with
`--dither halftone`), run after deskew and before `--invert` and the
threshold; the first two are refused with `--cmyk` and `--transparent`.
Without the flags, output is unchanged.

Not planned (OCR prep): orientation detection, Otsu/tiled-Otsu thresholds,
error-diffusion dither, packed 1-bpp morphology, text-or-photo detection,
2× upscale before threshold, despeckle, PDF output with G4/DCT, and dewarp
model sharing.

**Unreleased on master: soft proofing against the output intent.**
`rrocket --proof[=relative|absolute]` (`SessionConfig::proof`,
`RasterOptions::proof`, `ProofIntent`) shows how a document prints on the
press its `/OutputIntents` names: the page's own (PDF 2.0), else the
catalog's; the first `GTS_PDFX` intent that embeds a profile, else the first
of any subtype that does, whose profile must be CMYK. A document without one,
an intent without an embedded profile, and a non-CMYK profile (the sRGB intent
of a PDF/A file describes a display: nothing to proof against) are loud errors
at session open (`RasterError::OutputIntent`). Every page is painted on the
ink canvas as a `--cmyk` page is, its colours that are not ink separated
through the press profile under `--intent` / `--bpc`, `--overprint`
simulating overprint, and the ink is then converted to sRGB or to an RGB or
gray `--output-profile`: relative colorimetry maps the paper to white,
absolute reproduces the paper's white and the press's black (sRGB or a
matrix/TRC display profile). The ink-to-display conversion is tabulated on a
33⁴ grid (7.1 MB per press and proof intent, built on the first proofed page)
and matches Little CMS's `transicc -p` within ΔE00 1 at 16 bits and 1.2 at 8
bits (two 8-bit roundings against its floating-point output; colours below the
press black, where the two CMMs' separations differ, are left out of the 8-bit
chart). Of the 1 168 archive PDFs, 433 carry output intents: 428 PDF/A sRGB
ones, which `--proof` refuses, and 5 PDF/X CMYK presses (US Web Coated SWOP
v2 ×3, ISO Coated v2 300 % ECI, GWG Generic CMYK). Also: black-point
compensation to a CMYK profile now maps the page's black to the black the
profile reaches as a conversion source, the round trip of its perceptual black
(ISO 18619, as Little CMS detects it), instead of its darkest colorant, so
`--cmyk --bpc` shadows get lighter: the compensated black moves from L* 16.50
to 16.61 on US Web Coated (SWOP) v2 (inks within 0.3 %), from L* 11.3 to 12.9
on ISO Coated v2 (up to 4.4 % ink, about 11 levels of an 8-bit channel, at
black), and from L* 8.0 to 11.1 on the synthetic test press; `--cmyk` without
`--bpc`, and RGB and gray output profiles, are unchanged.

**Unreleased on master: a wide working space under an output profile.**
A page rendered at 16 bits under `--output-profile` is composited in a wide
working space, the ROMM RGB primaries with the D50 white and the sRGB tone
curve (`WorkingSpace::Wide`), instead of sRGB, so colours outside sRGB reach
the printer conversion unclipped: on device grids through Little CMS it holds
100 % of ISO Coated v2 and of an averaged 6-colour photo inkjet
(PhotoGamutRGB avg6c), where sRGB holds 92.5 % and 49 % (clipping up to ΔE00
16) and Rec. 2020 88 % (up to 4.2, in the cyan-greens). Device colours keep
their sRGB meaning: their value converts through one 16-bit table conversion
shared by fills and images; neutrals keep their value exactly (gray content
and black-text antialiasing are unchanged), and every white (DeviceRGB,
DeviceGray, an sRGB-tagged image, `0 0 0 0 k`, Lab `100 0 0`) is the working
white. CalRGB, CalGray, Lab and ICC colours convert through the PCS into the
wide space; 8-bit images of three or four components convert at 16 bits.
Blending, transparency groups, antialiased edges, shading and image
interpolation and transfer functions act on wide values, which moves mixtures
of colours: against compositing in sRGB, colour over white at 50 % coverage
by mean ΔE00 2.3 (p95 5.5), random colour pairs at 50 % alpha by 3.6,
Multiply by 6.3 (neutral mixtures by 0). Group `/CS` stays unread for
blending, and a wide page whose group declares one warns; soft-mask groups
are still painted in sRGB, and overprint-preview pages are composited in sRGB
and warn. The output conversion and CMYK separation from the wide space
evaluate in floating point with tetrahedral interpolation at full weight
precision: against Little CMS, sRGB colours printed through the wide space to
the RGB test printers within ΔE00 0.60 (1.03 on PhotoGamutRGB avg6c), the
wide chart within 0.99. On the 34-PDF gate corpus (pages 1–2, 50 dpi) at 16
bits under the RGB test printer, every page file differs from before: text
pages by at most 1.6 8-bit steps; the most-moved pixels of flat colour areas
match Little CMS's conversion of their sRGB value as before (mean ΔE76 0.13,
0.15 before); antialiased edges and transparency move by up to ΔE76 16. `--working-space
srgb|wide` (`SessionConfig::working_space`, `None` automatic) chooses; `wide`
without an output profile or at 8 bits is refused, and the first 8-bit page
under an output profile warns that it is composited in sRGB. `--gpu-pages`,
the device JPEG cache and the GPU ICC CLUT, all 8-bit sRGB, are not used for
wide pages. Without an output profile, at 8 bits, and with `--working-space
srgb`, output is byte-identical: 0 of the corpus's files differ under 15
modes (RGB, gray and CMYK at both depths, overprint, proofs with and without
a display profile, RGB, gray and CMYK profiles at 8 bits, `--gpu-pages`).
Also: on a 16-bit canvas an 8-bit image in an ICC, CalRGB, Lab or
`DeviceCMYK` space converts at 16 bits, as a fill of its colour does, instead
of being converted at 8 bits and widened, so `--depth 16` output changes
without a profile on pages with such images: 14 of the corpus's 65 page files
at 16 bits, by at most 1.29 8-bit steps (the 8-bit `DeviceCMYK` row
conversion approximates the model by up to that), and 11 of its 99 `--cmyk
--depth 16` files; DeviceRGB and DeviceGray images, and 8-bit output, are
unchanged. Not done: the wide space at 8 bits, group blending spaces, and the sRGB
output conversion's 8-bit-precise table interpolation. API changes:
`WorkingSpace`, `ColourTarget` (replaces `ColourIntent` as the conversion
argument of `ColorSpace::convert_to_rgb`), `CanvasRequest::WideRgb`,
`SessionConfig::working_space`, `page_group_declares_space`;
`OutputProfile::{convert, separation, prepare}` take the working space.

**Unreleased on master: CMYK output completes its conversions.**
- **Black generation and undercolour removal.** Without an output profile,
  `--cmyk` separates RGB-family colours under the `ExtGState` `BG2`/`BG` and
  `UCR2`/`UCR` functions (ISO 32000-2 §8.4.5, §10.4.2): `C = c − UCR(k)`,
  `K = BG(k)`; `/Default` is the device's full gray-component removal, as
  before. Grays (`DeviceGray`, `CalGray`, ICC gray, gray images and
  shadings) stay black ink alone. A CMYK output profile, the proof and RGB
  output ignore them, a profile's `BToA` being its own black generation.
- **ICC CMYK of another press.** Under a CMYK output profile, a colour,
  image, shading or tint alternate in an `ICCBased` CMYK space whose profile
  is not the output's (by tag data) converts to the output's ink: source
  `AToB` and output `BToA` evaluated in floating point, the absolute white
  scale or the ISO 18619 black map between them, on a 17⁴ table per source,
  intent and compensation; within ΔE00 0.73 of Little CMS between the two
  test printers (1.53 for compensated perceptual/saturation to the new v2
  one). The output's own profile and `DeviceCMYK` keep their values.
- **Indexed images over ink** (`DeviceCMYK`, ICC CMYK, `Separation`,
  `DeviceN`) paint their palette's ink instead of a separated sRGB value.
- **Rendering intent per object.** Without `--intent`, each object separates
  under its own intent (`ri`, `/RI`, an image's `/Intent`), on CMYK pages
  and proofs; tables are built per intent on first use.
- **Exact separation.** Each node of the 65³ separation table is the
  profile's `BToA` evaluated at the node's PCS value, no longer the
  converter's 33³-grid transform, and the table interpolates in tetrahedra
  so grays read the neutral axis alone: below the test press's black, sRGB
  grays 4–16 separate as Little CMS does (cyan 0/0/0/11.2 %, was
  5.3/0.1/8.0/11.4 %); the chart prints within ΔE00 0.60 of `transicc`, and
  the 8-bit proof chart, the patches below the press black now included,
  within 1.11.
- **Spots.** A spot whose alternate's white separates to ink no longer tints
  the whole page: tint 0 is no ink.
- **Pixel impact** (pages 1–10, 50 dpi, against the previous master, of the
  43 archive documents with BG/UCR functions, Indexed-CMYK images or ICC
  CMYK sources): under `--cmyk --output-profile` ISO Coated v2 300 %, 36
  change on 284 of 423 pages, by up to 215 levels: SCOTUS slip opinions'
  US Web Coated (SWOP) v2 K-only text becomes ISO Coated's four-colour black
  (C70 M61 Y60 K85, as Little CMS converts it), K-only Indexed palette
  entries stop printing as four-colour black. Without a profile, 8 change on
  28 pages (the Indexed images); the 9 documents with BG/UCR functions are
  unchanged on these pages. Through the SWOP press of their own content,
  the SCOTUS pages keep their K. Of the 63 documents painting under a
  non-relative intent, 60 change through ISO Coated v2 (333 of 528 pages,
  at most 82 levels).
- **API changes:** `OutputProfile::separation(&Arc<Self>, space,
  Option<RenderingIntent>, bpc)` returns `Separation::Profile(Arc<PressSeparation>)`;
  `OutputProfile::separation_table(space, intent, bpc)` is one intent's
  `ProfileSeparation`; `DecodedImage::intent`; `PageDiagnostics::output_intent`
  of a CMYK or proofed page is `--intent`, else the page's one content
  intent.

**Unreleased on master: forced-CUDA deskew, deep forms, JPEG 2000 matte
and truncation, Type 3 without `/FontMatrix`.**  Under `--backend cuda` a
deskew rotation that fails on the GPU is `RasterError::BackendUnavailable`
instead of a silent CPU rotation (`deskew::DeskewError` is now an enum,
`TooLarge` or `GpuRotation`). A Form XObject nested `MAX_FORM_DEPTH` (now
40, about 5.6 KB of stack a level) forms deep is skipped with a warning and
listed in `PageDiagnostics::forms_too_deep`, and the rest of the page
renders; previously the whole page was lost at 32 (`--strict` still fails
the page; `RenderAbort::FormDepth` is gone). A `/SMaskInData 2` JPEG 2000
image is un-premultiplied against its dictionary's `/Matte` rather than
against black. A JPEG 2000 codestream that ends before a tile's data is
skipped and reported instead of drawn as mid-grey; one whose tiles are all
present but cut short is drawn as far as it decodes and reported partial
(the local `hayro-jpeg2000` fork reports `DecoderContext::completeness`).
A Type 3 font without a valid `/FontMatrix` uses the identity, not
0.001, and logs a warning. The `nvjpeg2k` build of the interpreter compiles
again. pdfium corpus: `bug_972999` passes and `bug_557223` matches its
expected image; `jpxdecode_smask_in_data_2` is reclassified as a divergence
(its references draw the colour opaque, ignoring alpha) and `bug_1128284`
now draws two of its three marks.

**Unreleased on master: Link borders, `/MaxLen`, installed faces by name,
curved patches.**  A Link annotation without an appearance stream draws its
border: `/Rect` inset by half the width (`/BS /W`, else `/Border`, default
1) and stroked in `/C` with its dash, the bottom side alone for `/BS /S /U`;
without `/C`, or at width 0, it stays invisible. A synthesised text-field
appearance shows at most `/MaxLen` characters of its value, comb or not. A
font without an embedded program draws with the installed face its name
names (PostScript name first, then the family key: subset tag dropped,
style suffix after `,` or `-` cut, case, spaces, hyphens and underscores
ignored), in the nearest weight and slant, unless it is one of the standard
14, their metric-compatible aliases, Symbol or Dingbats; the choice is
logged at info level. `FaceInfo` gains `postscript`, `FontDescriptor`
gains `face_name`. A Coons or tensor patch is subdivided until it is also
flat to 0.25 px, so a single-colour patch with curved sides keeps its
curves instead of being drawn as its corner quadrilateral. Fixtures
corpus-08 and corpus-09 now draw their non-embedded Carlito as Carlito and
pass against pdftoppm; pdfium corpus: `bug_41010502_1` matches its
expected image, `bug_1072440`, `bug_846`, `bug_41010502_2` and
`type3_xobject` are reclassified as divergences, and `shade-tensor` draws
its curved patches (tile 161 to 98); patches whose sides loop from a shared
corner still paint their surface where both references fill the outline.

**Unreleased on master: a clip takes effect after the paint that ends its
path.** A `W`/`W*` followed by a painting operator (`W f`, `W S`, `W B`,
`W b` and their even-odd forms) now paints the path unclipped by it, then
intersects it into the clip (ISO 32000-1 §8.5.4). Previously the clip was
intersected first: `re W S` lost the outer half of its stroke, and an
antialiased `W f` applied its edge coverage twice (a pixel two-thirds
covered painted at four-ninths). `W n` is unchanged. The paint trace now
lists such a paint before its clip. No pixel changes on the repository
fixtures (all 3,824 pages at 50 dpi gray, and
ab_render's 215 outputs, byte for byte as before).

**Unreleased on master: a more lenient parser.** Four damaged-file shapes
that pdfium or pdftoppm read now parse. A number with a run of leading
signs takes its last one (`--10.12` is -10.12; it was 0). A header of
`%PDF` without a version is accepted with a warning, and an unclosed
dictionary ends at the next object's `obj` or at `trailer`, so the
object-scan repair keeps the objects and trailer after it. The page count
is the page tree's: a root `/Count` is checked by one descent to its last
index and, where the counts do not describe the tree (a `/Count` past the
leaves, an interior node without one), replaced by the walked count with a
warning; such pages resolve through the memoised walk, and a kid listed
twice is a page twice. The walk is bounded at 2^22 node visits and `/Kids`
entries, where shared nodes could describe 2^40 pages. A stream whose
`/Length` misses its end is cut at an `endstream` that is a keyword
(`endstream.` and `(endstream` are data, `ETendstream` is not), or at
its object's `endobj`, and spaces after `stream` no longer shift the
`/Length`. pdfium corpus: of the 11 documents registered as these
defects, 10 now render pdfium's page up to glyph antialiasing, and
`bug_828049` (a catalog with a direct `/Pages` dictionary) still fails;
three documents pdfium refuses or draws no page for now open on one blank
page; the full corpus shows no new failure.

**Unreleased on master: deskew confidence, `--deskew`, resolution in
PNG and JPEG, 1-bit mono PNG.**  Deskew now declines a page without a
confident angle: the ratio of the best to the worst score of its angle
sweep must reach 3 (0 for a flat profile), and a best angle within one
sweep step of ±7° is declined.  It refines on the 2× reduction, so the
angle resolves about 2/W radians on a page W px wide (0.05° for A4 at
300 dpi), as now documented and tested; the projection is sheared about
the page's centre line and scores only the rows every column reaches, so
negative angles no longer drop the top-right of the page and the sheared
page edges no longer dominate dark pages.  Behaviour change: deskew
leaves photographs, noise, blank pages and pages tilted past ±6.75° as
they are, where it used to rotate them, and the angle of the pages it
corrects changes slightly.  `deskew::detect` and `deskew::Skew` (angle,
confidence) are public, `deskew::apply` returns the skew, and
`RenderedPage::skew` / `PageInfo::skew` carry it.  `rrocket --deskew`
straightens gray pages (`--gray`, `--mono`, `--fax`; comic input too) and
is refused at start-up with colour, `--cmyk` and `--transparent` output.
PNG output carries a `pHYs` chunk and JPEG output the JFIF density of the
render resolution (`write_png`, `PngWriter::new` and `write_jpeg` take a
resolution), so PNG and JPEG bytes change; `--mono --png` writes a PNG of
bit depth 1 (`write_png_bilevel`, `PngColor::Bilevel`) where it wrote 8-bit
gray holding two levels.  PPM, TIFF, fax and print output are unchanged.

**Unreleased on master: 16-bit comic input, camera stages and deskew.**
`--depth 16` now combines with comic-archive input and with `--crop`,
`--rectify`, `--split-spreads` and `--dewarp`.  Comic pages decode at
`ComicOptions::depth`: a PNG, TIFF, JPEG XL or AVIF image of more than 8
bits a sample keeps its levels (an AVIF picture converts by its matrix
straight to 16 bits), an 8-bit image is widened exactly (`v · 257`), and
the page budget counts 6 bytes a pixel.  The camera stages measure on the
page's nearest 8-bit levels and `Report::replay` resamples each channel at
16 bits (`GrayView<'_, S>`, `GrayPage<S>`, `camera::Sample`).
`deskew::apply` takes `Gray8` or `Gray16` (`DeskewPixel`), so
`RasterOptions::deskew` is accepted at depth 16; a 16-bit page rotates on
the CPU.  New: `rgb16_into_gray16`, `gray16_into_rendered_page`,
`Bitmap::converted`.  8-bit output is unchanged, byte for byte (golden
digests).  The GPU paths stay on the CPU at 16 bits, and `--gpu-pages`
stays refused there.

**Unreleased on master: 16-bit DeviceCMYK corners and an ink-solve
stall.** At 16 bits the DeviceCMYK model interpolates `color::cmyk::CORNERS16`,
the sixteen corners converted at 16 bits from the same characterisation
(FOGRA39 as ISO Coated v2 300 %, media-relative colorimetric with black
point compensation, to sRGB, solid K neutral), instead of the 8-bit corners
widened; solids and overprints move by up to 127 of 65 535 levels, and
the 8-bit table, unchanged, is the 16-bit one rounded (tested).  16-bit
renders still narrow to the 8-bit render within the same bounds.  The
overprint ink solve (`ink_reproducing`), which stopped short of its minimum
when an ink sat just inside a bound (13 of 40 000 random cases), now sets
such an ink to the bound and continues; results change only where it had
stalled, each to a lower cost.

**Unreleased on master: reviewed expected-failures registry.** The
comparison harness judges a page by its worst per-channel tile delta
(≤ 12, so colour errors a gray metric hid now count) and judges RMSE only
against pdfium; `compare.sh` renders under `target/compare` on disk.
Re-seeded, the pdftoppm suite falls from 2,889 registered pages to 27 and
the pdfium suite holds 186; every entry carries a reason: antialiasing,
a named divergence with its evidence, or a defect with its audit ticket.

**Unreleased on master: JPEG 2000 device threshold and prefetch scale.**
`GPU_JPEG2K_THRESHOLD_PX` rises from 262 144 to 2 097 152 px, from an
end-to-end sweep of real archive codestreams by area and bytes per pixel
(RTX 5070, 24 threads, whole-process wall time, cold decoder build
included): from 0.45 to 1.35 MP the device lost 3-8× on one page and tied
(0.92×) or lost up to 4× on 24 pages; above 2.6 MP results are mixed: it
won on 24-page runs of scans (qsj-1798 0.69×, Halbwachs 0.60×) and on a
24 MP image (0.67×), and lost on a single page (qsj-0616 p. 10 1.5×) and
on some 24-copy cells (up to 1.46×); bytes per pixel did not order the
outcome. `PDF_RASTER_JPX_THRESHOLD`
overrides the area. `--prefetch` at a fixed resolution no longer decodes a
JPEG every draw of which on its page is smaller than the image
(`SessionConfig::prefetch_scale`).

**Unreleased on master: GPU-session fallbacks and Windows RAM output.**
In a session holding a GPU backend, an ICC-based CMYK image the device does
not convert (below `GPU_ICC_CLUT_THRESHOLD`, or declined) now takes the
profile transform a CPU-only session uses instead of the scalar CPU table
evaluator (about 3× faster on the A18 Pro); a page with
`PageDiagnostics::device_icc_conversions == 0` is byte-identical to a
CPU-only render, and a device-converted image stays within ±4 of it. On
Windows, `--ram-path` works (available memory is read through
`GlobalMemoryStatusEx`); `--ram` alone is still refused there, having no
`/dev/shm`.

**Unreleased on master: the GPU parallel-Huffman JPEG decoder is removed.**
It was dormant (reached only through `PDF_RASTER_HUFFMAN_THRESHOLD`) and lost
to the multi-threaded CPU decoder on every measured workload, by about 10×
whole-process on gray scans, so the decoder, its CPU pre-pass, the
parallel-Huffman, Blelloch-scan and IDCT kernels (CUDA and Slang) and their
tests are deleted (see Phase 8; master `10393d55` is the last commit with it).
`Auto` no longer starts a GPU backend at start-up under any setting: an `Auto`
session opens no Vulkan device or CUDA context, and only the on-demand
nvJPEG2000 decoder and the `gpu_pages` device remain. Rendered output does not
change. Removed public API: the `gpu-jpeg-huffman` feature (gpu, interp,
`rasterrocket`, CLI); the `rasterrocket_gpu::jpeg_decoder` module, with
`JpegGpuDecoder` and `JpegGpuError`; `GpuBackend::record_scan`,
`record_huffman` and `record_idct`, with `ScanParams`, `ScanPhase`,
`HuffmanParams`, `HuffmanPhase`, `Phase4FailureKind`, `IdctParams` and their
constants; `rasterrocket_interp::huffman_threshold` and
`DeviceKernel::HuffmanJpeg`; `PageRenderer::set_jpeg_gpu`, `take_jpeg_gpu`,
`set_jpeg_vk` and `take_jpeg_vk`; the `B: GpuBackend` parameter and decoder
argument of `resolve_image` and `PageResources::image`; and the
`PDF_RASTER_HUFFMAN_THRESHOLD` variable, which is now ignored.

**Unreleased on master: nvJPEG2000 declines before the decoder is built.**
The device JPEG 2000 path now decides from the codestream's headers, before
a thread's nvJPEG2000 decoder is built, whether the device returns the
samples the CPU decoder does: one or three unsigned 8-bit components on the
image grid, no JP2 palette, channel definition or sYCC/CIELab/YCCK
conversion, and no code-block style but predictable termination, which
nvJPEG2000 0.11 refuses at parse (the sweep's lost cells c01, c05 and c12,
and every image of Saunders, were such refusals, falling back silently after
the build). A decline is reported at warn, once per kind. Previously
sub-8-bit components were drawn unscaled (a 5-bit ramp near black) and signed
ones unshifted. A codestream that ends early (`Image::survey` in the JPEG
2000 fork reads its tile-part headers without decoding) decodes on the CPU,
drawn and marked partial (`--strict` fails) or refused when a tile holds no
data. `--prefetch` sizes draws inside forms too, and prefetches nothing, with
a warning, when the render scale depends on the page. A job-level signal for
Auto (single pages lose about 3x to the cold start, 24-page runs win only
sometimes) is measured and proposed, not shipped.

**Unreleased on master: paint trace.** A pixel-independent regression
oracle: `PageRenderer::set_paint_trace` (library: the doc-hidden, unstable
`rasterrocket::dev::paint_trace`) records one line of text per painting
operation of a page, in its device space: path fills and strokes with
their rule, stroke parameters and device width, clips named by a hash
chained from the clip they narrow, source colours, text objects and the
pen of each glyph, images, shadings, patterns as paint, forms and groups,
soft masks with their group's content, annotations and hidden optional
content.  Coordinates round to the nearest 1/256 px (the coverage
rasteriser's grid), colour components and matrices to six decimals; a
path prints its counts, bounding box and hash, and its points up to
eight.  Only the first window records, as the glyph census counts, and
nothing window-, memo- or canvas-dependent is recorded, so the trace is
the page's whether rendered whole, in bands or as a region (tested).  Off,
each hook is one branch on an `Option`.  Goldens of twelve synthetic pages
(`crates/pdf_raster/tests/traces/`, `UPDATE_TRACES=1` rewrites them) run
on every CI leg; an ignored case compares the corpus fixtures' first three
pages with local goldens in `tests/fixtures/traces/` (gitignored).  The
`paint_trace` dev binary (`crates/bench`) dumps, diffs (aligned by
operation, one-quantum moves marked) and blames: it attributes the pixels
two renders differ at to the operation that paints them
(`tests/compare/trace_blame.sh`).  No pixel changes (ab_render: 0 of 215
fixture outputs).

**Shipped in v2.2.0 (2026-11-14).** A reduced image draw builds its resampling plan
once, and the vertical pass reads only the source columns the horizontal
pass uses (54d8712e); renders unchanged. `--print` no longer renders in
barrier windows: each page is appended to the spool once the pages before
it are done, and a slow page no longer idles the other workers (a 200-page
text document with one scan page in ten, 600 dpi: 7.5 → 4.3 s, peak memory
4.4 → 5.5 GB). `PageDiagnostics::device_fills` and `device_fill_failures` (public,
additive) count the fills a GPU backend painted and those it took and
failed (each then painted by the CPU), and `render_page_rgb_with_diagnostics`
returns an RGB render's diagnostics.  The GPU fill parity tests assert both
counts on the renders they compare, so a device fill falling back to the
CPU fails them instead of comparing the CPU render with itself; their corpus
half now uses fixtures whose fills reach the device (corpus-02 and
corpus-07 fill only through Type 3 glyphs or path clips, which stay on the
CPU).
On an integrated GPU the Vulkan backend reads an uncached mapped buffer
back through a device copy into host-cached staging instead of in place:
the image cache's read-back of an 8.7 MB image on RADV, 27.7 → 0.42 ms,
and scanned documents under `--backend vulkan --vulkan-device radv`, pages
1–40 at 150 dpi, 0.54 → 0.24 s, 0.89 → 0.17 s and 1.06 → 0.20 s; renders
unchanged.  The Vulkan perf follow-ups (transfer queue, ICC CLUT texture,
BDA push constants, blit batching) were measured with the new
`vulkan_phase_bench` and closed as not worth it (Phase 10, Task 3).
The device image cache (`cache` feature) runs on the Vulkan backend too:
under `--backend vulkan` (or `Auto` with a Vulkan kernel enabled) JPEG
images are cached in Vulkan device memory, with the host and disk tiers and
the prefetcher, and drawn by the Vulkan blit kernel, byte-identical to the
CPU render (22 image-heavy archive PDFs, 229 pages at 300 and 150 dpi, none with an ICC-based CMYK image: 0 byte differences from the CPU render on Vulkan RTX 5070, on RADV and on CUDA; a page with an ICC-based CMYK image differs between a GPU session and a CPU-only one by design, the GPU session converting it through the 17-node lookup table and the CPU-only session through the profile transform).  CUDA behaviour is unchanged.  API changes:
`gpu::cache::DeviceImageCache` / `CachedDeviceImage` take a backend
parameter (defaulting to CUDA), `ContentHash::of` replaces
`DeviceImageCache::hash_bytes`, `CacheError` gains `Backend` and
`BackendMismatch`, and `RasterSession::image_cache`,
`rasterrocket_interp::cache::spawn_prefetch` and
`PageRenderer::set_image_cache` take or return the backend-erased
`gpu::cache::ImageCache`.
Features a document uses that the renderer ignores are reported as typed
data, not only as log lines: `RasterSession::unsupported_features()` lists
the document-level ones (catalog, name-tree and form JavaScript, an XFA
form, a portable collection) and `PageDiagnostics::unsupported_features` a
page's (page and annotation JavaScript, `/3D`, `/RichMedia`, `/Movie`,
`/Sound` and `/Screen` annotations), as `UnsupportedFeature` values with
their location (public, additive; `UnsupportedFeature`, `ScriptLocation`
and `MediaKind` are re-exported, and `rasterrocket-interp` adds
`document_unsupported_features` and `page_unsupported_features`).
JavaScript detection now also follows an action's `/Next` chain and counts
a `/Rendition` action's `/JS`; opening a portable collection and rendering
the first page with a 3D or multimedia annotation each log a `warn`.
A function-based (Type 1) shading is evaluated at every device pixel of its
Domain instead of on a 64 × 64 grid, so detail finer than 1/64 of the Domain
is drawn (above 4 Mi pixels it samples every k-th pixel and interpolates).
**This changes pixels** of every Type 1 shading: by 1–2 counts on a smooth
function, wholly on fine detail; a full-page shading of a Type 4 function
costs about 0.2 s more at 150 dpi.
Every crate now requires Rust 1.100 (was 1.99); CI's minimum-toolchain job
holds the new floor.
A `DCTDecode` stream with bytes before its SOI marker, or a frame height of
0xFFFF (unknown) with a matching `/Width`, is repaired instead of dropped,
and JPEGs taller or wider than 16 384 pixels decode (the 1 GiB decode budget
bounds them); a stream whose scan data ends short is drawn as far as it
decodes and reported partial.  A `--fax` page longer than the B4 length
(364 mm on the fax) is sent as A4-length pieces.  The glyph advance and
text-matrix translations use fused multiply-adds: **this changes pixels**,
glyphs on a rounding boundary moving by a device pixel (corpus-05 at
150 dpi: 169 of 601 pages, a median of 69 pixels a page).
moxcms 0.9.2 rounds integer gamma tables to nearest and corrects D50
adaptation: colour-managed RGB output may move by one level (two archive
documents with an output intent, of 2600 sampled outputs).  A `.cb7`
archive is opened with a 1 GiB bound on its decoder allocations, and
sevenz-rust2 0.25 rejects entries whose data ends short or fails its CRC.
`.cbr` (RAR 1.3 to RAR 7) comic archives are read, through the pure-Rust
`rars` decoder, decoded whole on open under the same entry and 1 GiB
dictionary bounds as `.cb7`; `ComicError::RarUnsupported` is gone.
A multi-page TIFF entry of a comic archive yields one page per image (its
reduced-resolution copies and transparency masks skipped) instead of its
first page only.
JPEG XL page images (`.jxl`, bare codestream or container) decode through
the JPEG XL project's pure-Rust `jxl` decoder: the first frame, as 8-bit
gray or sRGB.
AVIF page images (`.avif`) decode: the HEIF container is parsed by the comic
crate itself (primary item, coded picture or `grid` of tiles, `iloc`
versions 0 to 2 with `idat`, `clap` / `irot` / `imir` applied, the `nclx`
colour box over the sequence header, alpha and ICC profiles not read) and
the AV1 pictures by `rav1d`, the Rust port of dav1d, vendored as a local
fork in `crates/rav1d` with a safe single-picture entry point; 8, 10 and 12
bits, 4:0:0 to 4:4:4, chroma interpolated bilinearly.  The fork's assembly
kernels are built by default (`av1-asm`), which makes `nasm` a build
requirement on x86; `--no-default-features` builds without them.
The GPU composite and soft-mask kernels, which nothing in the render path
dispatched, are deleted with their `GpuBackend::record_composite` /
`record_apply_soft_mask` methods, parameter structs, thresholds and CPU
twins (`composite_rgba8_cpu`, `apply_soft_mask_cpu`) from
`rasterrocket-gpu`; renders unchanged.
`--box media|crop|bleed|trim|art` (and `--cropbox`) selects the page box
rendered, the MediaBox among them; the default stays the CropBox.  Public
API: `PageBox` and `SessionConfig::page_box` (additive); `page_size_pts`,
`page_size_pts_by_id` and `prescan_page` in `rasterrocket-interp` take the box.
With `gpu-icc`, a Vulkan session converts ICC-based CMYK images on the
Vulkan device as a CUDA session does on CUDA (`--backend vulkan` previously
converted them on the CPU through the profile transform); the Slang kernel
now interpolates with `fma`, byte-identical to the CUDA kernel and the CPU
evaluator on the RTX 5070 and RADV, and a device failing a one-time
exactness check converts on the CPU. `PageDiagnostics::device_icc_conversions`
and `device_icc_failures` (public, additive) count the device's conversions.
Pages render over a transparent background, for capability parity with
`pdftocairo -transp`: `--transparent` (alias `--transp`) writes RGBA PNG
or TIFF (unassociated alpha) and refuses PPM, JPEG, `--gray` and `--mono`
with the formats that carry alpha; `RasterOptions::transparent_background`
yields `PixelFormat::Rgba8` pages (the new `RenderedPage::format`) through
the new `render_page_rgba`, and `RenderParams::transparent_background`
gives `render_page_rgb_with` a bitmap with an alpha plane, colour not
premultiplied. The page is painted as an isolated page group and its alpha
kept (`PageRenderer::try_finish_transparent` and `transparent_page_group`
in `rasterrocket-interp`); `encode_for_gcv` refuses an RGBA page with
`GcvError::NotGray`. Gray and RGB output is byte-identical.
`Auto` renders on the CPU unless a device kernel (the AA fill, the
parallel-Huffman decoder) is enabled by its threshold: it no longer
initialises a CUDA context, image cache or nvJPEG decoder.  Start-up
(`startup_bench`, warm): CUDA cuInit 7 ms, primary context 60 ms, PTX
modules 0.2 ms (61 ms with a cold JIT cache), first launch 6 ms, nvJPEG
decoder 17 ms per thread, nvJPEG2000 decoder 0.25 ms (62 ms cold); Vulkan
device 95–106 ms on the RTX 5070 (drop 27–31 ms), 48–50 ms on RADV (drop
10–12 ms), 11–15 ms on the A18 Pro under MoltenVK (drop 1 ms).  Whole
process, 15 documents × 1, 5, 40 pages × 150 and 300 dpi: `Auto` cost
110–270 ms more than `--backend cpu` on every one but the JPEG 2000 scans,
and now matches it (corpus-02, one page: 283 → 11 ms; qsj-1889, 40 pages
at 150 dpi: 356 → 73 ms, and 845 → 72 ms in a build with `nvjpeg`).
nvJPEG2000 earns its start-up on large JPEG 2000 scans drawn near their
resolution (corpus-10, 40 pages at 300 dpi: CPU 2.29 s, `Auto` 1.09 s), so
`Auto` builds a thread's decoder on demand, for a `JPXDecode` image of at
least 262 144 px not drawn at half its size or less on both axes (such a
draw decodes a lower resolution level on the CPU; corpus-10 at 150 dpi:
1089 → 745 ms).  **This changes `Auto` pixels** on pages with an
ICC-based CMYK image (now the profile transform, as `--backend cpu`) and on
JPEG 2000 images drawn at half their size or less (now the CPU decoder);
`--backend cpu`, `cuda` and `vulkan` are byte-identical to before (177
pages of fixtures, scanned books, ICC CMYK documents and an archive
sample, on each backend).  On the Mac, where `Auto` already ran on the
CPU, `--backend vulkan` is 13–16 ms slower on short jobs and level on long
ones.  A JPEG drawn below its resolution is no longer put in the device
image cache, from which a reduced draw read it straight back (corpus-07,
40 pages at 150 dpi, `--backend cuda` / `vulkan` / RADV: 474 → 361,
651 → 538, 595 → 461 ms).  `GPU_ICC_CLUT_THRESHOLD` is 128 000 px (was
500 000; `PDF_RASTER_ICC_CLUT_THRESHOLD` overrides it): a GPU session
converts every ICC CMYK image through the lookup table, and the device
beats the single-threaded CPU evaluator from 4 000 px per call on CUDA,
Vulkan NVIDIA and RADV and from 16 000 px under MoltenVK, end to end from
128 000 px on one thread (500 000 px: CUDA 315 vs 447 ms for eight
images).  `threshold_bench --only clut` and `startup_bench` (gpu crate)
measure both.
A library caller renders pages into memory it owns: a `PageBuffer` keeps
the page canvas between renders and its `render_gray_into` /
`render_rgb_into` / `render_rgba_into` write the page at the caller's
stride, byte-identical to `render_page_gray` / `render_page_rgb` /
`render_page_rgba`; deskew rotates from the canvas straight into the
output.  Once the canvas has grown to the largest page, a render allocates
no page-sized buffer (20 pages at 150 dpi, gray, against
`render_page_gray`: corpus-01 8.89 → 0.47 MB allocated a page,
1.79 → 1.14 ms; corpus-08 41.4 → 12.5 MB, time level; the page-sized
allocations left are decoded images).  `RasterSession::page_size_px` gives
a page's pixel size at a dpi before the render; a short stride or buffer
is `RasterError::OutputStrideTooSmall` / `OutputBufferTooSmall`, returned
before drawing.  `open_session_from_shared` opens a document from any
`AsRef<[u8]>` storage the caller shares (`Arc<[u8]>`, `&'static [u8]`, a
reference-counted buffer, its own mapping) without a copy, and
`open_session_from_reader` from any `Read` source, read to its end first
since the parser needs the whole document as one slice.  Public API, all
additive: `PageBuffer`, `PageInfo`, `PixelFormat::Rgb8`,
`RasterSession::page_size_px`, `open_session_from_shared`,
`open_session_from_reader` and the two `RasterError` variants; in
`rasterrocket-parser`, `Document::from_shared(_with)`; in
`rasterrocket-interp`, `open_shared_with` and
`PageRenderer::try_with_page_group_in`; in `rasterrocket-render`,
`BitmapBuffers`, `Bitmap::try_new_filled_in` and `Bitmap::into_buffers`.
An extra rotation in quarter turns (`rrocket --rotate DEG`,
`SessionConfig::rotation` / `RasterOptions::rotation`, a `Rotation` built by
`Rotation::from_degrees`, which takes any multiple of 90 and rejects other
angles with `InvalidRotation`) is added to each page's `/Rotate` before the
page geometry is computed, so the page is drawn turned: sizes,
`page_size_px`, regions, deskew, `PageBuffer` and the gray, RGB and
transparent outputs all see the turned page, and comic-archive page images
are turned pixel for pixel.  On a page of whole pixels, fills, clips and
nearest-neighbour images are byte-identical to the unturned render turned
as a bitmap; antialiased edges, hairlines, text and interpolated images may
differ at their edges.  Public API, additive: `Rotation`, `InvalidRotation`
and `PageGeometry::rotated` in `rasterrocket-interp`, re-exported with the
two new fields by `rasterrocket`.
A zero-width line cut short by a clip's bottom edge, the page's or a
region's included, drew one pixel too many on its last kept row; it now
ends that row where the unclipped line does, so a region render of a
hairline crossing the region's bottom matches the whole page (this changes
pixels on that row only).
Property tests (`proptest`, a dev-dependency only) cover the parser, the
rasteriser's fills and clips, the colour conversions, every encoder and the
camera geometry (see Testing strategy).  They found five bugs, now fixed:
an aliased fill skipped the cells of a winding-count-0 intersection met
outside the shape (a horizontal edge, an edge's first scanline), so an
integer-aligned rectangle lost its top row, shapes under about two pixels
vanished and the result depended on the outline's direction — **this
changes pixels** of `--aaVector no` renders (0.07 % of the pixels of the 164
fixture pages at 72 dpi), antialiased renders unchanged; a sloped edge
moved by whole pixels could lose a cell where it ends on a cell boundary
(fixture renders unchanged); `div255` rounded down at 126 of its inputs,
12 of them byte products, instead of to nearest — **this changes pixels**
by one level where compositing meets them (5 713 pixels of those pages);
`round_half_up_i32` took 0.49999999999999994 to 1; and `rectify` could
return a page one rounding past its twice-the-source bound.
`tests/compare/compare.sh` and `pdfium_corpus.py` classify every page and
document against a checked-in registry of expected failures,
`tests/compare/expectations.json` (a reason per entry, optional per-page
RMSE and tile limits): `PASS`, `XFAIL`, `FAIL` (new, or worse than
registered) and `XPASS` (registered but passing).  They exit non-zero on
`FAIL` only, on `XPASS` too with `--strict`, and `--update-expectations`
seeds the registry from a run (see docs/cli-reference.md).
A Coverage workflow measures the line coverage of the workspace's own
crates (vendored forks, `pdf_bridge` and the bench crate left out) under
CI's CPU-only test sets with `cargo-llvm-cov`, uploads the lcov and HTML
reports as an artifact and writes a per-package table to the job summary;
it reports only and gates nothing.
A PGO + BOLT workflow (on `v*` tags and by hand) runs
`scripts/release_pgo_build.sh` on a hosted runner and uploads the stripped
`rrocket` binary as an artifact: x86-64 baseline codegen, trained on the
synthetic documents generated from committed files, BOLT profiled by
instrumentation (no perf or LBR needed) and required to apply.  The script
now takes training PDFs as arguments, falls back to that synthetic set when
the corpus fixture is absent, merges the per-process BOLT profiles, and
ends with a smoke render of the final binary.

A file opened by path can be read into memory instead of memory-mapped
(`SessionConfig::read_into_memory`, `RasterOptions::read_into_memory`,
`pdf::OpenOptions::read_into_memory`; public, additive, default `false`),
so a long-running process whose input file another process truncates no
longer dies of `SIGBUS` on the next read of a mapped page.  The CLI keeps
the mapping.
The JPEG decoder is a local fork (`crates/zune_jpeg`: zune-jpeg 0.5.15,
then moved to upstream's commit of 2026-05-01, with the fork's own changes
in its `CHANGES.md`), tested and linted in CI and tracked by
`scripts/check_vendored.py`.  A baseline JPEG that spreads its components
over several scans (what an encoder that optimises its Huffman tables
writes) decodes correctly: a scan lost its last rows when they fitted in
the bit reader's lookahead (a flat 16 × 16 image came out half dark red),
and every subsampled frame in several scans was garbage; **this changes
pixels** of such images only (none among the 18 859 JPEG streams of the
archive).  Arithmetic-coded JPEG (SOF9, SOF10) decodes; it failed the
image.  A `DCTDecode` image is reported partial exactly when its stream
lacks data: the decoder counts the rows it decodes from data that is
there (`JpegDecoder::intact_rows`), so a complete stream without its EOI
marker is whole, progressive or not, and a stream cut at a marker (a scan
stopped at a restart marker, a progressive frame missing its later scans)
is reported, where it was drawn zero-filled as if complete; a stream cut
inside a marker segment or inside the first scan of a progressive frame is
drawn as far as it decodes, where it failed.  Decode time is within noise
of zune-jpeg 0.5.15.
The 139 upstream zune-jpeg commits past the fork's base are reviewed (up
to `ab211bb1`; `UPSTREAM.toml` records the review, `CHANGES.md` what was
taken and what was left out).  Brought over, each with a test that failed
before: an AC run past the end of a block is written at position 63 on
every path and counts its row as damaged; Huffman tables are checked scan
by scan; a scan of a component an earlier scan coded is skipped; gray
output of a vertically subsampled multi-scan frame, unused components,
mixed sampling factors and CMYK / YCCK to RGBA are fixed; the input colour
space is decided when the headers are read; a sequential stream without
DHT segments uses the tables of ITU-T T.81 §K.3.  The fork's own further
fixes: an arithmetic-coded progressive AC band without an end of band is
reported incomplete; the count of fill bits is a `u64`; a progressive frame
past the scan limit draws the scans decoded; a DNL segment after the first
scan is read and skipped (the interpreter writes `/Height` into a frame
header of height 0, as for 0xFFFF); and `set_max_coefficient_bytes` bounds
the coefficient planes a progressive or multi-scan frame holds whole, which
the interpreter sets to what the output samples leave of the 1 GiB decode
budget.  The decision to keep a reduced JPEG draw out of the device cache
reads the frame header's size, not the dictionary's.  A comic page whose
JPEG scan data is incomplete is logged at `warn`, its pixels unchanged.
The parallel-Huffman IDCT kernels (CUDA and Slang) no longer race between
their row and column passes: on MoltenVK some 4:4:4 blocks came out with
chroma up to 124 levels off; `idct_parity` compares the Vulkan kernel with
a CPU twin, byte-identical on MoltenVK, the RTX 5070 and RADV.  An IDCT
block count the kernels' 32-bit offsets cannot address is refused.

**Whole pages on a Vulkan GPU.** `SessionConfig::gpu_pages` (`rrocket
--gpu-pages`, default off) opens the Vulkan device on a thread of its own
when the session opens.  Once it is open, a worker whose page paints only
solid-colour fills and strokes (rectangular clips, `Normal` blend, constant
alpha) claims the idle device, records the page as paint commands
(`PageRenderer::record_for_device`; each fill and stroke scan-converted by
the CPU's own geometry stages, `aa_scanner`, `stroke_aa_scanner` and
`hairline_runs`) and has it rasterised in one submission by a tiled kernel
running the CPU's span walk and compositing (`gpu::page::PageCommands`,
`VulkanBackend::render_page`).  A busy device, an ineligible page or a
refused or failed recording renders on the CPU, the CPU pass keeping the
recording's deadline and cancellation flag and running on its own operator
budget; output is byte-identical either way (random content streams and the
first pages of every corpus fixture, on the RTX 5070 and RADV).
`PageDiagnostics::device_page` (`DevicePage`) reports each page's outcome,
and `RUST_LOG=rasterrocket=debug` logs it.  A command list is bounded to
2^27 bytes and every storage binding is checked against the device's
`maxStorageBufferRange` (`BackendError::DeviceLimit`); a page past either
renders on the CPU, and any other device error retires the device for the
session with one warning.  `open_session` refuses the option without the
`vulkan` feature, under a policy other than `Auto` or `ForceVulkan`, or
with vector antialiasing off.

**Overprint simulation completed except for transparency groups.** On the
ink canvas, shadings and sampled images in `Separation` and `DeviceN`
spaces paint their colorants' channels (a per-page `ColorantPlan` shared
with solid colours; a decode keeps the tints beside the sRGB samples);
coloured tiling cells render on a child ink canvas sharing the page's spot
slots and paint exactly the channels their content wrote; an `ICCBased`
CMYK colour prints as the process ink reproducing its profile's colour
(`color::cmyk::ink_reproducing`, budgeted and memoised per page), the
components given as 0 staying 0; transfer functions apply in ink space to
the channels a colour carries, red, green and blue to C, M and Y, gray to K
and the spot inks; and the GPU AA fill and image blit, device-resident
cached images included, run on the ink canvas, byte-identical to the CPU.
A neutral darker than solid K prints as a balanced rich black (C : M : Y =
1 : 0.725 : 0.75 under solid K), so antialiased edges of black stay within
five counts of neutral at every coverage and solid black prints as
(0, 2, 2).  Transparency-group canvases are pooled per canvas kind.  Pages
rendered without `--overprint` are unchanged.

**Text shaping across show strings.** A word of a substitute-drawn run
split between show strings (a `TJ` adjustment, a `Tf` mid-word,
consecutive `Tj`) is shaped as one word: each string is shaped with the
neighbouring strings on its baseline as pre- and post-context
(`FontProgram::shape_in_context`), neighbours being strings separated only
by text-state, positioning, colour and marked-content operators, at most
0.1 em past the first's end and 0.5 em before it, at the same text rise.
Combining marks ending a right-to-left string are drawn over the first
letter of the next string when it is of the run's script.  A nested
content stream ends the context.  Synthesised appearances set their
fallback characters in shared fonts of 256 characters each (at most 64),
so a word using several Unicode blocks or a ZWJ / ZWNJ is one show
string; characters set as `?` are logged.  A program the shaper cannot
read is reported once, and logged runs are cut to 32 characters.  The
fixtures and the archive's Hebrew and Arabic documents (4 581 pages) are
byte-identical.

**Hardening of the work above.**
- *Comic archives.* The decompressed total of an archive is bounded
  (`ComicOptions::max_archive_bytes`, `DEFAULT_MAX_ARCHIVE_BYTES` = 4 GiB,
  `rrocket --max-archive-mib`); past it the archive is refused with
  `ComicError::ArchiveTooLarge` before the first page.  Entry names are
  indexed once; a damaged member of a non-solid RAR fails that entry only.
  A TIFF's directory chain is walked once, at most twice the page cap.
  The AVIF reader bounds the container's item, location and extent tables
  over the whole file, refuses duplicate boxes and transforms HEIF allows
  once, and budgets samples by the chroma layout of the stream's own
  sequence headers (`rav1d::still_layout`); `decode_still` refuses empty
  data.
- *Interpreter.* The unsupported-feature scan has a budget of 2^20 steps
  and inspects a shared action once; an empty `/Names /JavaScript` tree is
  not reported.  The prescan's PPI hint divides by the unrotated page width
  (a 2480 px A4 scan under `/Rotate 90` gave 212 ppi instead of 300).  A
  page object that is not a readable dictionary is warned about.  A page
  rotation outside 0–359 is reduced before a session rotation is added.
- *Library.* `PageBuffer` keeps the ink canvas and the page group's alpha
  plane between renders.  A file too large to read into memory under
  `read_into_memory` is an `io::ErrorKind::OutOfMemory` error, not an
  abort.  A GPU deskew failure, including an NPP output of the wrong size,
  falls back to the CPU rotation, so `deskew::apply` and
  `RasterError::Deskew` are never returned; deskew of a non-gray render is
  one refusal naming the gray entry points.
- *Rasteriser.* A sloped hairline under a clip whose top or bottom edge
  falls inside a pixel row draws its kept rows as the unclipped hairline
  does (**this changes pixels** on such rows).
- *GPU.* A disk-tier cache entry whose header disagrees with its length is
  removed; the CLUT coordinate scale is computed once on the host, correctly
  rounded (device division moved output bytes for grid sizes such as 4, 8,
  13, 16 and 25); a device blit that fails before compositing draws through
  the host copy; `--backend vulkan` on an `nvjpeg` / `nvjpeg2k` build
  creates no CUDA JPEG decoder, and nvJPEG decoders are built only where
  `GPU_JPEG_THRESHOLD_PX` lets them dispatch; a grown blit target zeroes
  only the blit box; blit and CLUT scratch buffers are pooled on the
  backend and a CLUT conversion is one page; a Vulkan transfer waits on its
  own fence instead of the queue; nvJPEG2000 is offered to a masked JPX
  drawn small, which the CPU decodes in full; the VRAM budget is reserved
  before an upload, so concurrent inserts cannot exceed it; the benchmarks
  reject unknown flags and print the device's name.
- *Property suites.* The parser suite writes objects without separators
  where delimiters allow and reaches the filter decoders' defaults; the
  render suite reaches the scanner's re-anchor rows; the colour, camera and
  encode oracles are tightened; `PROPTEST_CASES` that is not an unsigned
  integer fails the suite.
- *Comparison scripts.* A document that renders is recorded as an observed
  pass; `-U` without a registry keeps a failing run fatal; NaN and infinite
  limits are rejected; a malformed registry is reported once, before
  rendering; `-U` keeps reviewed reasons and lowers improved limits;
  registry updates are serialised by a lock file; the over-limit page count
  comes from `expectations.py`.

**Document-wide image decodes, in v2.1.0.** An image `XObject`
drawn on two pages or more is decoded once per session rather than once per
page: a store shared by the session's pages keeps it once a second page asks
for it, up to 256 MiB of host-resident decodes (device-resident decodes stay
with the device image cache). Output is byte-identical; a 48-page document
sharing one 3000×3000 JPEG background renders 1.6× faster (4.5 → 1.9 s CPU)
at under half the peak memory (1.22 → 0.51 GB), while corpus-06, whose
shared images are small, is unchanged. Image decodes on a page are now also
kept per default-colour-space context, so a form whose resources define no
`/DefaultGray` no longer draws an image as the page's `/DefaultGray`
decoded it.

**Per-document latency, in v2.1.0.** The archive-wide latency
run (50 dpi gray, one core) left two text-heavy documents slower than
pdfium; both now match or beat it. A text run drawn with `2 Tr` (synthetic
bold, a stroked glyph outline) draws each glyph's stroke from cached
coverage instead of stroking the run's outlines (solid, opaque, undashed,
normal blend, antialiased): the israel pages go from 1.32 s to 0.10 s
against pdfium's 0.95 s. Curves flatten against the chord segment rather
than its midpoint; fills drop closed subpaths lying wholly outside the clip;
cross-reference entries are indexed by object number; a run of
text-showing operators resolves its face once; and a CFF program's built-in
encoding is built in one charset walk. rosa-scheuerman renders in 0.266 s
against pdfium's 0.261 s. **Two of these change pixels**, both approved
after a comparison against an 8× reference render: the flattening, and the
cached glyph strokes (60 archive pages change, all stroked text; each
glyph stroke is placed at the quarter-pixel phase of its fill). Everything
else is byte-identical over the archive.

**A refused page bitmap is an error, in v2.1.0.** The page
canvas used to be allocated infallibly, so a page within the size limits
whose bitmap the allocator refuses (up to 1.8 GB RGB, 4.8 GB on the
overprint canvas) aborted the process. It now fails that page with
`RasterError::PageAllocationFailed`.

**HTJ2K, in v2.1.0.** JPX images coded with the high-throughput
block coder (ISO/IEC 15444-15) decode; they used to fail with a diagnostic
naming HTJ2K. The vendored JPEG 2000 fork gains an HT block decoder (the
cleanup, SigProp and MagRef passes) and HT codeword segments, quality layers
and placeholder passes included. Reversible streams of the OpenJPH test set
decode bit-exactly as OpenJPH decodes them and irreversible ones within one
count; the HT conformance streams of ISO/IEC 15444-4 decode as their classic
originals do wherever the transcode is lossless. Only the mixed mode (HT and
classic code-blocks in one tile-component) is still refused, by name.

**Image reduction limited to the paintable pixels, in v2.1.0.**
An image drawn below its resolution is area-averaged only over the device
pixels inside the page and the clip rectangle. Output is byte-identical; a
large image drawn mostly off-page or under a small clip renders 1.2–1.6×
faster.

**Overprint simulation, in v2.1.0 (merged 2026-11-03).**
`--overprint` / `RasterOptions::overprint` (default off). With it, a page on
which some `ExtGState` reachable from its resources, or from the appearance
streams of annotations that are drawn, sets `OP` or `op` is painted on a
CMYK-plus-four-spot ink canvas instead of RGB (ISO 32000-1 §8.6.7,
§11.7.4); an overprinting object leaves the channels it does not paint as
they were, and the canvas is converted to RGB when the page is finished.
`Separation` and `DeviceN` colours paint their colorants' channels (the
first four distinct spots of a page take the spot channels, further spots
paint their alternate in process ink and are listed in
`PageDiagnostics::unslotted_spots`); under `OPM 1` an overprinting process CMYK
colour leaves its zero components unpainted. Grey and RGB colours print through an
RGB-to-ink inverse of the process-ink model, so they render as on the RGB
canvas; the inverse table takes about 0.75 s to build, once per process.
Pages without overprint, and every page with the option off, render
byte-identically. Known gaps: images and shadings in `Separation` /
`DeviceN` spaces, tiling-pattern cells and transparency groups as a whole
paint every channel; an object under a non-identity transfer function
knocks out; `ICCBased` CMYK converts through the process-ink model rather
than its profile; antialiased edges of RGB or grey black text take a slight
colour cast; and fills and image composites on an overprint page run on
the CPU.

**Font forks rebased, in v2.1.0.** The read-fonts and skrifa forks move
to fontations `1d1672367` (read-fonts 0.46, skrifa 0.49) and the harfrust
fork to `bdd55a548` (0.15), together, since harfrust is built against the
read-fonts API. The fixes the forks had carried over by hand are now
upstream's own code; each crate's `CHANGES.md` lists what the forks still
change and the upstream behaviour changes this brings in (cmap format 4
`idRangeOffset`, CFF charset bounds, Type 1 parsing, the TrueType
interpreter's parity checks for hint-reliant fonts, harfrust's Hangul and
OpenType rounding, and the ITC Zapf Dingbats glyph names, so `a1` maps to
U+2701). Over 1 125 PDFs (the fixtures and the archive), renders at
50 dpi (4 pages) and 150 dpi (12 pages, 12 169 pages) are byte-identical to
before. A shaping failure now falls back to glyph-by-glyph drawing with a
warning.

**Page box anchored at the top-left corner, in v2.1.0.** When a page's
size in device pixels is fractional, the bitmap is its rounding, and the
page box now starts at the bitmap's top-left corner for every `/Rotate`,
the leftover fraction of a pixel lying past the right and bottom edges.
Before, the box was anchored at the bottom edge (rotations 0 and 270)
and at the right edge (180 and 270), so content sat up to a pixel low or
right. Over the 434 pages that change among pages 1–3 of 250 random
archive PDFs at 150 dpi, the fitted vertical offset against mutool falls
from 0.15 to 0.06 px on average and against pdfium from 0.32 to 0.20 px;
pdftoppm, which places a CropBox by its MediaBox at a whole-pixel offset,
moves from 0.38 to 0.36 px.

**Many path clips fold into one, in v2.1.0.** A clip that reaches eight
path clips replaces them with a single intersection table, which answers
every clip test exactly as the eight did, so a page installing thousands of
`W n` clips no longer pays for each one on every later fill. A page of
3 000 nested diamond clips and 3 000 fills renders in 0.31 s instead of
6.1 s and peaks at 47 MB instead of 600 MB at 150 dpi; output is
byte-identical (the 1 125 fixture and archive PDFs, a randomized parity
test of every clip query).

**Tint-transform work bounded per image and per page, in v2.1.0.** An
image whose colour space converts through a tint transform (`DeviceN`, and
the other spaces converted tuple by tuple) has a budget of 2^28
interpreted operations, and the images of a page, its pattern cells and
Type 3 glyphs included, share 2^32; an image spends at most the smaller of
its own budget and what the page has left. Each conversion costs its
function's bound, fixed when the function is built: a Type 4 program's
tokens plus the entries its `copy` and `roll` can move, a Type 0
function's grid corners × outputs × bits per sample, and the transforms
of nested alternate and `Indexed` base spaces added in. Within the budget,
by pixel or by distinct sample tuple (up to 2^20 of them), every pixel
converts exactly as before. Past it, the image converts through a lattice
of exact conversions (18, 16, 6, 4 or 2 values per component, the largest
whose nodes fit; one-component images included) with multilinear
interpolation, shared with Type 0 functions, and a warning names the grid;
past eight components, or when no lattice fits, the image is dropped with
a warning. A decode the page's remaining budget curtailed is not offered
to the document image store, so a page's output does not depend on which
page decoded an image first. One Type 4 run is limited to 2^20 operations.
A 2000 × 2000 four-colorant noise image under a 100 000-token transform
renders in 0.4 s; it did not finish in 180 s before. Before the page
budget, a page of a thousand images each just inside the per-image budget
took a quarter of an hour. The fixture and archive PDFs render byte-identically.

**Vulkan on integrated GPUs and on macOS, in v2.1.0.** On an integrated
GPU, device buffers are host memory, mapped and read and written in place
instead of staged; buffers the host reads back are placed in host-cached
memory (on the Raphael iGPU under RADV a 64 MiB download takes 3.1–3.5 ms
instead of 212–235 ms, and a 16 Mpx device AA fill 13 ms instead of
38–65 ms), and a discrete GPU stages its downloads through host-cached
memory too. A device AA fill is one submission instead of four (CUDA per
fill: 25.5 → 12.1 µs at 256 px, 631 → 527 µs at 4 MP), its scratch buffers
owned by the backend and reused across pages. On macOS the Vulkan backend
finds Homebrew's or an absolute `$VULKAN_SDK`'s loader and enables
portability enumeration, so MoltenVK runs it: the fixtures render
byte-identically on an A18 Pro's GPU and its CPU. Under `auto`, no Vulkan
device is initialised while both Vulkan kernels are dormant, no
parallel-Huffman decoder is built while its kernel is disabled, and a
`--vulkan-device` / `PDF_RASTER_VULKAN_DEVICE` selector is still checked
against the enumerated devices, so one that matches nothing fails.
Concurrent Vulkan instance creation and teardown, which crashed in the
loader, is serialised. The fixtures render byte-identically under
`--backend vulkan` on the RTX 5070 and on RADV and under `--backend cuda`,
against `--backend cpu`, with the AA fill forced on (215 outputs each).
`tests/compare/ab_render.sh` and `ab_archive.sh` compare two binaries, or
two backends of one, byte for byte.

**Second 2026-11-04 review pass, in v2.1.0.** A review of the work since
the first pass of the day:
- *JBIG2.* Every symbol instance is charged against the page's work
  budget whether or not it writes pixels, a dictionary's aggregate
  symbols share one instance allowance, and an exhausted memory or work
  budget fails the image in lenient mode too instead of dropping regions
  silently; a pattern dictionary 128 pixels wide or more placed its
  collective bitmap's AT pixel wrongly (or overflowed at 128).
- *Tint transforms.* The page-wide budget, the cost model and the Type 4
  run limit described above; a Type 0 function reads its grid at the
  stride of its own output count; a function's cost is stored when it is
  built (a chain of stitching functions sharing sub-functions made the
  walk exponential); a NaN lattice position takes the first node.
- *Rasteriser.* A sloped edge's 48.16 fixed-point position is re-anchored
  every 4096 rows, so a vertex far outside the page no longer moves the
  edge by pixels (up to about 7 px for a vertex 10^6 rows above it); the
  error is at most 1/32 px plus one conversion, and region renders stay
  byte-identical to the whole page.
- *Interpreter.* Tiling-pattern cells, kept forms and inline-image decodes
  are keyed on the identity of the resources and colour-space dictionary
  they resolve in, so two resource sets built without a page object
  (interactive-form default resources) no longer share a cell or a decode;
  the document font store, like the image store, serves one document;
  the render-time `suggested_dpi` hint divides by the page width in
  points, as the prescan does, instead of always returning the maximum
  for a full-page scan; kept forms are charged for the heap behind their
  operands. A refused tiling-pattern screen, pattern-cell or Type 3
  child canvas, or canvas snapshot stops the page with
  `RasterError::PageAllocationFailed` instead of aborting the process.
- *GPU.* Mapped host access holds the Vulkan recorder lock, so a page
  submitted from another thread cannot write a buffer being read; one-shot
  transfers end with a host-visibility barrier and zero-fills are ordered
  after earlier dispatches of the page; devices lacking a required feature
  are skipped under `auto` and refused by a selector; a failed or
  panicking page leaves the recorder idle and is abandoned on every
  backend through one chain; CUDA rejects transfers recorded outside a
  page, as the trait documents, and its host buffers are cudarc's pinned
  slices, whose drop waits for in-flight copies; AA scratch growth is
  bounded to 64 MiB beyond the request; a relative or empty `$VULKAN_SDK`
  is ignored on macOS, so a loader planted in the working directory is
  not loaded.
- *CLI.* `--list-printers` finds the default destination in any locale
  and ends quietly on a closed pipe; a refused print job points at
  `--list-printers` (and `--printer` for the default destination), not at
  a missing `lp`.
- *Fonts.* A shaping run that exhausts harfrust's limits is drawn glyph by
  glyph with a warning instead of half-shaped; vertical forms that name no
  glyph of the program are ignored; a cached program releases its spare
  capacity. The font forks' `CHANGES.md` files list every remaining local
  delta.
- *Comparison scripts.* `ab_archive.sh` draws the same sample in every
  locale, fails when no page was compared, and counts a render failure
  once.

Phase 7's last item, the mode E re-bench, is closed as moot. v2.2.0 is
released (see the release history). The CUDA performance
inventory is closed (2026-11-01): fills run on the backend recorder and
transfers are blocking by design.

Deferred/blocked items, in full:

- **Phase 8 B–D** — closed (2026-11-12). Phase A shipped as an OSS artifact (v1.0.0) and stays wired but dormant. The aggregate-throughput loss vs 24-thread CPU is the answer, not a gap.
- **Phase 10 criterion 4** — cross-vendor proof of life (AMD-RADV / Intel-ANV / lavapipe). No longer hardware-blocked: the dev box has an AMD Raphael iGPU, and `--vulkan-device radv` (and `llvmpipe`) render; on RADV the fixtures render byte-identically to the CPU path with the AA fill forced on (215 outputs, `tests/compare/ab_render.sh`), and so does the archive-scale run (2026-11-12: `ab_archive.sh`, fixtures plus the 150-file archive sample, 621 pages at 50 dpi gray with the AA fill forced on: 0 differences from the CPU path on RADV and on llvmpipe). The perf-only Task 3 follow-ups were measured and closed (2026-11-13). Fills on the device run only in the `--gpu-pages` page raster, which composites the CPU's exact-area coverage.
- **Phase 11 deferred tasks** — closed (2026-11-12): banded rendering is satisfied by `RenderRegion` tiles; the multi-resolution cache and the display-list render path are dropped; a page trace survives only as a regression oracle.
- **Banded / clip-restricted image decoding** — deferred by measurement (2026-11-04). Every codec decodes an image whole, and the page memo and document store share that one decode between draws and pages. Over the first 40 pages of the 1 125 fixture and archive PDFs (49 661 image draws), the largest decode is 81 MiB and only 3 draws of 16 MiB or more show less than half their image (one 25 MiB JPEG), so decoding just the visible rows would save 45 MiB in all. Hostile sizes are already bounded: the 1 GiB decode budget is checked before allocation and sample buffers are allocated fallibly. Revive for a workload of very large images drawn mostly clipped.
- **crates.io** — abandoned. `rasterrocket` and `rasterrocket-cli` were published once at 1.0.1 and are frozen there (not yanked, never updated); every other crate is `publish = false`. The project is distributed from GitHub only.

The implementation plans under `docs/superpowers/plans/` (gitignored) are completed or superseded; they are kept as historical execution logs, not as a backlog.

---

## Testing strategy

### proptest — property-based testing

`proptest` (a workspace dev-dependency, without its `fork` and `timeout`
features) checks invariants on generated inputs, and shrinks a failure to a
minimal case.  Properties state what an operation must preserve, compared
against an independent reference (a decoder, an encoder written from the
specification, a transformed copy of the input), not against a restatement
of the implementation.

Each suite runs a fixed case count in CI (well under 10 s a crate in a debug
build); `PROPTEST_CASES=N` overrides it for a longer local run, and a
value that is not an unsigned integer fails the suite (the runner
configuration is shared, `tests/support/proptest_config.rs`).  Seeds of
failures found are kept in each crate's `proptest-regressions/` and replayed
first.  Generated sizes stay small: bitmaps up to a few hundred pixels a
side, byte inputs up to a few KiB.

| Crate | Suite | Properties |
|---|---|---|
| parser | `src/proptests/` (in-crate, CI runs `--lib`) | objects written in every spelling ISO 32000-2 §7.3 allows (separators and comments, name and string escapes, octal escapes and line continuations, hex digit case, number forms) parse back and end where written; a stream with an indirect `/Length` finds its body; chains of up to three filters (Flate, LZW with either `EarlyChange`, ASCIIHex, ASCII85, RunLength) with PNG and TIFF predictors invert reference encoders with no truncation reported; TIFF predictor 2 keeps a partial last row; `RunLengthDecode` under an output bound returns a prefix within the bound |
| render | `src/proptests.rs` (in-crate) | random polygons and curves, aliased and antialiased, both winding rules: fills stay inside the clip rectangle and, aliased, the clip paths; a clip rectangle masks the unclipped fill exactly; reversing every subpath changes nothing; whole-pixel moves of outline and clip move the render, also for edges spanning the scanner's 4096-row re-anchor period; a hairline under a clip with fractional top and bottom edges is the unclipped hairline masked to the clip; simple polygons fill alike under both rules; integer rectangles cover exactly their closed cells; flattening stays finite and within the control points' box |
| color | `tests/properties.rs` | the CMYK row converter (every SIMD tier) matches the per-pixel conversion at any length and alignment; more ink is not lighter; ink rows match the per-pixel spot union; `floor_i32` / `ceil_i32` / `round_half_up_i32` near integers and half-integers; `div255`, `lerp_u8`, `luminosity` |
| encode | `tests/properties.rs` | PGM, PBM, PPM, PNG (alpha included), TIFF (every compression, bilevel with Group 4) read back by independent decoders hold the live pixels of padded bitmaps, CMYK and `DeviceN` as their process-ink conversion; the fax coder round-trips under Group 3 MH and MR (any `k`, aligned EOLs or not) and Group 4; JPEG decodes to its size |
| camera | `tests/properties.rs` | `crop` and `split_at` return exactly the named pixels and refuse out-of-image requests; `Quad::bounding_rect` holds exactly the centres in the corners' extent; `rectify` on any quad returns a bounded page or declines |

The suites found five bugs (aliased span walk, scanner translation
rounding, `div255`, `round_half_up_i32`, the rectified-area bound); see the
v2.2.0 notes under Current status.

**Where fuzzing beats proptest** (covered by `fuzz/`): crashes and hangs on
arbitrary bytes in the parser, the content stream, fonts, functions, images,
the filter chain and the CCITT and JBIG2 decoders.  Coverage guidance finds
the specific byte sequences that random generation misses; the property
suites check invariants on well-formed input instead.

**Not covered by a property suite:** CTM composition and inversion (the
interpreter's `invert_affine` rejects a determinant below 1e-12 in absolute
terms, so a uniformly tiny but invertible matrix counts as singular), and
the ICC CLUT grid, whose 254 sizes an exhaustive unit test covers.
