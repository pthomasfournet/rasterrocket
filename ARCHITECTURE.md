# rasterrocket — Architecture

End-to-end description of the codebase: crate topology, data flow, key
abstractions, and platform strategy.

---

## 1. Crate topology

```
rasterrocket-color
  │   pixel types, colour math, convert helpers
  │
  ├── rasterrocket-render
  │     software rasterizer — paths, fills, compositing, SIMD
  │     │
  │     ├── rasterrocket-font
  │     │     font programs → Path → coverage bitmaps; glyph cache
  │     │     │
  │     │     ├── skrifa, read-fonts  (crates/skrifa, crates/read_fonts: local forks)
  │     │     │     TrueType / OpenType / CFF / Type 1 parsing and outlines
  │     │     │
  │     │     └── harfrust  (crates/harfrust: local fork)
  │     │           GSUB / GPOS shaping of substitute-drawn text
  │     │
  │     ├── rasterrocket-encode
  │     │     PPM / PGM / PBM / PNG / JPEG output; TIFF (BigTIFF past 4 GiB),
  │     │     CCITT G3/G4 coded for fax; PDF of page images (the --print
  │     │     document); row-streamed writers; ICC profile tagging
  │     │
  │     └── rasterrocket-interp
  │           PDF content stream interpreter + operator dispatcher
  │           │
  │           ├── gpu  (optional, feature-gated)
  │           │     CUDA and Vulkan kernels, whole-page Vulkan raster,
  │           │     device image cache, nvJPEG/nvJPEG2k and VA-API decoders,
  │           │     NPP rotate
  │           │
  │           ├── zune-jpeg  (crates/zune_jpeg: local fork, pure Rust)
  │           │     JPEG decoder for DCTDecode
  │           │
  │           ├── hayro-jpeg2000  (crates/jpeg2000: local fork, pure Rust)
  │           │     JPEG 2000 decoder for JPXDecode
  │           │
  │           ├── hayro-jbig2, hayro-ccitt  (crates/jbig2, crates/ccitt: local forks)
  │           │     JBIG2Decode and CCITTFaxDecode
  │           │
  │           └── rasterrocket
  │                 public OCR library — raster_pdf(), render_channel()
  │                 │
  │                 └── rasterrocket-cli
  │                       `rrocket` binary (pdftoppm-style flags); hands fax
  │                       and print documents to lp / sendfax
  │
  ├── rasterrocket-comic  (comic-archive input: .cbz/.cb7/.cbt/.cbr → RenderedPage;
  │     │                  depends on rasterrocket + render + color; used by the CLI)
  │     │
  │     └── rav1d  (crates/rav1d: local fork)
  │           AV1 decoder for AVIF page images
  │
  ├── rasterrocket-parser  (in-tree lazy mmap PDF parser; used by rasterrocket-interp)
  │
  ├── rasterrocket-camera  (camera-capture page cleanup on 8-bit gray: crop, rectify,
  │                         spread split, dewarp; std only, no workspace deps; used by
  │                         the CLI)
  │
  ├── pdf_bridge  (unused in render path; poppler reference baseline only)
  │
  └── bench  (vello_cpu throughput comparison; not linked by rasterrocket-cli)
```

`rasterrocket-color` and `rasterrocket-render` are the foundation. Nothing above them has knowledge of PDF.
`rasterrocket-parser` reads the PDF file structure (cross-reference, objects,
streams, filters); `rasterrocket-interp` interprets content streams on top of
it. Everything below the interpreter is PDF-agnostic.

---

## 2. Data flow — one page, start to finish

```
PDF bytes on disk
    │
    ▼ rasterrocket_parser::Document  (in-tree lazy mmap parser)
rasterrocket_interp::open()
    │
    ▼ Vec<Operator>
rasterrocket_interp::parse_page_content()    content tokenizer → operator decode (operands borrow the content)
    │
    ▼ PageCanvas  (blank, sized by PageGeometry::size_px or a band's window):
    │             Bitmap<Rgb8> / Bitmap<Rgb16>, or an ink canvas
    │             Bitmap<DeviceN8> / Bitmap<DeviceN16> of CMYK + four spot
    │             inks when overprint is simulated and page_uses_overprint
    │             holds, or the page is rendered as CMYK
rasterrocket_interp::PageRenderer::try_with_page_group_in(&PageView::whole(..), .., &CanvasRequest, CanvasBuffers)
    │                                 builds the canvas in the buffers of the
    │                                 previous page (PageBuffer); initial CTM
    ▼
PageRenderer::execute(ops)           dispatch loop:
    ├── path ops  ──────────────────▶ raster::fill / stroke / eo_fill
    ├── text ops  ──────────────────▶ font::FontFace → GlyphBitmap
    │                                 raster::fill_glyph
    ├── image ops ──────────────────▶ resources::resolve_image
    │   DCT      ───────────────────▶ zune-jpeg fork (nvJPEG / VA-API behind
    │                                 their features)
    │   JPX      ───────────────────▶ hayro-jpeg2000, at the image's drawn
    │                                 resolution (or gpu::NvJpeg2kDecoder)
    │   CCITT/JBIG2 ────────────────▶ hayro-ccitt / hayro-jbig2
    │   Flate    ───────────────────▶ libdeflate (default) / flate2 fallback
    │   ICC CMYK ───────────────────▶ profile transform: its tabulated grid on
    │                                 the CPU or (gpu::backend::icc_clut) on
    │                                 a session's CUDA or Vulkan backend
    │                                 PageRenderer::blit_image
    ├── shading ops ────────────────▶ raster::shaded_fill / gouraud_triangle_fill
    ├── tiling ops  ────────────────▶ raster::TiledPattern
    └── transparency groups ────────▶ raster begin_group / paint_group
    │
    │   (with SessionConfig::gpu_pages, a qualifying page is recorded as
    │    gpu::page::PageCommands instead and rendered by
    │    VulkanBackend::render_page into the same canvas; see §3.6)
    │
    ▼ PageRenderer::try_finish_keeping(transparent)
    │                                 page group onto white, unless rendered over
    │                                 a transparent background; the overprint
    │                                 preview's ink canvas is converted to sRGB
    │                                 (color::cmyk::ink_to_rgb_row, spots through
    │                                 their process equivalents), a CMYK page's
    │                                 finished as its process ink
    ▼ FinishedCanvas  Rgb8 / Rgb16 (+ alpha plane when transparent), Cmyk8 / Cmyk16
OutputProfile::convert (optional)    to an RGB profile in place, to a gray one into
    │                                 Gray8 / Gray16
rasterrocket::rgb_into_gray()        Y = 0.30 R + 0.59 G + 0.11 B (ISO 32000-1 §10.3.3),
    │                                 written over the RGB buffer in place
    │                                 (rgb16_to_gray16 at 16 bits)
    │
    ▼ Pixels::U8 / Pixels::U16  (width × height gray samples; a transparent
    │           render skips the gray step and yields RGBA, a CMYK one CMYK)
RenderedPage { page_num, pixels, format, width, height, dpi, diagnostics }
    │
    ▼
deskew (optional)                    projection-profile sweep ±7°
    │   gpu::npp_rotate  (or cpu bilinear fallback)
    │
    ▼
caller / Tesseract OCR
```

No file I/O after the initial `open()`, and no subprocess: an encrypted
document is decrypted in process as its objects are parsed.

---

## 3. Crates in detail

### 3.1 `rasterrocket-color`

Single source of truth for pixel types and colour math. No SIMD, no I/O.

**Types** (re-exported at the crate root)
- `Pixel` trait — `Copy + Pod + Zeroable + Send + Sync + 'static`, with its
  sample type and channel count
- `Sample` trait — `u8` and `u16`: the compositing arithmetic written once
  over `S::MAX` and a wide product type (`u32`, `u64`), conversions between
  depths; `SampleDepth` (`Eight`, `Sixteen`)
- Concrete: `Rgb8`, `Gray8`, `Cmyk8`, `DeviceN8`, and at 16 bits `Rgb16`,
  `Gray16`, `Cmyk16`, `DeviceN16`; `AnyColor`
- `PixelMode` enum — `Mono1 | Mono8 | Rgb8 | Bgr8 | Xbgr8 | Cmyk8 | DeviceN8 |
  Mono16 | Rgb16 | DeviceN16 | Cmyk16`
- `TransferLut` — `[u8; 256]` newtype for transfer functions

**Primitives** (`convert`)
- `div255(u32) → u8` — rounded division by 255, used throughout compositing
- `lerp_u8`, `luminosity`, `ycbcr_to_rgb`, `gray_to_u8`, `rgb_to_bytes`
- `floor_i32`, `ceil_i32`, `round_half_up_i32` — saturating `f64 → i32`

**DeviceCMYK** (`cmyk`)
- `cmyk_to_rgb`, `cmyk_f64_to_rgb`, `cmyk_to_rgb_row`, `cmyk_to_rgb_vec` —
  the process-ink model: multilinear interpolation over the sRGB colours of
  the sixteen solid and overprint corners, in fixed point, so the scalar
  code, the SIMD tiers (`cmyk/simd.rs`) and the GPU kernel give identical
  bytes
- `rgb_to_ink` (`cmyk/inverse.rs`) — the inverse of that model with
  grey-component replacement, tabulated on a 65³ grid built once per
  process on first use (about 0.75 s); neutrals come from a 256-entry grey
  table. `ink_to_rgb_row` converts a row of `[C, M, Y, K, S0..S3]` ink
  pixels to sRGB, folding each spot in through its `SpotLut`; `rgb_to_ink16`,
  `SpotLut16` and `ink_to_rgb_row16` are the same at 16 bits, and the
  ink-to-CMYK rows fold the spots into process ink for CMYK output
- `ink_reproducing` (`cmyk/matching.rs`) — the CMYK inks printing a given
  sRGB colour, solved from a given set of inks by projected Gauss-Newton on
  the same model: an ink given as 0 stays 0, and of the solutions the one
  nearest the given amounts is returned. `cmyk/model.rs` holds the corner
  weights and ink rounding the two solvers share

### 3.2 `rasterrocket-render`

~17 700 lines. The software rendering engine. PDF-agnostic — takes paths,
bitmaps, and compositing parameters; knows nothing about operators or resources.

**`Bitmap<P: Pixel>`**
The primary output container. Width, height, stride, typed samples of `P`'s
sample type (8 or 16 bits), optional separate alpha plane. Generic over `P` — the compiler generates one code path
per pixel format with no runtime mode dispatch in hot loops.
`Bitmap::into_buffers` detaches the colour buffer and alpha plane as a
`BitmapBuffers`, and `Bitmap::try_new_filled_in` builds a later bitmap in
them, so a sequence of pages reuses one set of allocations.
`Bitmap::turned_cw` turns a bitmap by quarter turns, moving pixels without
resampling.

**Path pipeline**
```
PathBuilder  →  Path  →  flatten_path()  →  XPath (edge table)
                          make_stroke_path()      │
                                     ┌────────────┴────────────┐
                                     ▼                         ▼
                      coverage::AreaRaster          XPathScanner  →  ScanIterator
                      (antialiased: exact-area      (aliased: pixel-centre
                       coverage rows,                (x0, x1, y) spans)
                       CoverageTable)
```
Bézier flattening by De Casteljau subdivision: a curve is a segment once both
control points lie within the flatness of its chord segment. Stroke expansion
produces a fill path from
the outline — strokes and fills share a single rasterizer. The geometry
stages are public on their own: `fill::aa_fill_coverage` (the coverage
tables an antialiased fill composites), `aa_stroke_coverage` (that of a
stroke's outline) and `hairline_runs` (the pixel runs of a zero-width
stroke), so a caller can scan-convert exactly as `fill` and `stroke` do and
composite elsewhere.

**Coverage rasteriser** (`coverage/`)
One integer rasteriser serves antialiased fills, clips, strokes and glyphs
(`rasterrocket-font` calls it too). `AreaRaster` scan-converts the flattened
outline in fixed point of 1/256 pixel: each segment, clipped to the raster
window, deposits per cell its signed cover and area, and a running sum along
each row gives every pixel the exact fraction of its area inside the outline
under the non-zero or even-odd rule, as a byte `round(255 · area)`. Moving an
outline and its window by whole pixels moves the coverage exactly, and a
pixel's coverage does not depend on the window, so tiles and bands of a page
match the whole-page render. A `CoverageTable` holds a region's coverage as
per-row spans, each fully covered or carrying one byte per pixel. With
vector antialiasing off, paths are scan-converted at pixel centres by
`XPathScanner`.

Three compositing pipeline variants, selected at paint time by `PipeState::no_transparency()` / `use_aa_path()`, each generic over the sample type, so one formula serves 8- and 16-bit canvases (coverage stays 8-bit and is widened exactly at the boundary):

| Variant | When |
|---|---|
| `simple` | `a_input=255`, Normal blend, no soft mask, no group |
| `aa` | Shape byte present, Normal blend, no soft mask, no group correction |
| `general` | Everything else — soft mask, blend modes, transparency groups |

**Transparency groups**
`begin_group` / `paint_group` composite a group back into its parent with the
group's blend mode and opacity. The group bitmap is page-addressed but painted
only within the group's region (its `/BBox` ∩ clip ∩ page); the renderer keeps
a pool of transparent bitmaps, one per nesting level, and `paint_group`
composites and clears that region only, so per-group cost scales with the
region. Every group accumulates its own colour and alpha from transparent; a
non-isolated group also keeps a copy of the backdrop under it, which blend
modes inside the group consult (ISO 32000-2 §11.4.7). In a knockout group
(`/K true`, §11.4.6.2) the interpreter paints each object onto its region
cleared to transparent and merges the result with the group as it was,
interpolating by the object's shape (`renderer/page/knockout.rs`), so every
object composites with the group's initial backdrop.

**Overprint** (ISO 32000-1 §8.6.7, §11.7.4)
A paint may carry a channel mask (`PipeState::overprint`, a
`pipe::ChannelMask`) naming the channels it writes. `render_span` composites
the span as usual, then restores every channel outside the mask
(`restore_unpainted`) under the `CompatibleOverprint` rule, source value 0:
the channel keeps its premultiplied value, exactly over an opaque pixel;
inside a non-isolated group the stored value is solved so that the group
over its backdrop keeps the pre-paint colour. A mask of every channel, or none, takes the unmasked
path. The interpreter derives the mask per paint on an ink canvas: process
CMYK writes C, M, Y and K (under `OPM 1` without its zero components), a
`Separation` / `DeviceN` colour its colorants' channels, anything else every
channel; with overprint off, or on an RGB canvas, there is no mask.

**Page group** (ISO 32000-2 §11.4.7)
The page's content is itself a transparency group, composited onto the
white medium once complete: a blend mode on the page sees only what the page
has painted, never the paper. `pdf_interp::page_group` decides per page
whether that matters — a knockout `/Group`, or a blend mode other than
`Normal` reachable from the page's resources (through forms, patterns, Type 3
fonts) or its annotations' appearance streams — and only then is the page
painted onto a transparent bitmap that `finish` composites onto white; every
other page paints onto opaque white directly, which gives the same pixels.

**Soft masks**
An `ExtGState` `/SMask` renders its group into a page-sized mask plane
(luminosity or alpha, through the mask's transfer function). An object painted
under it is painted normally; the renderer then blends the result with a
snapshot taken before painting, weighted by the mask. The snapshot covers
only the device rectangle the object can touch (∩ clip), not the page.

**Pattern types**
- `TiledPattern` — implements `Pattern`; composites the rasterised cell at
  every point of the pattern lattice (`origin + col·col_step + row·row_step`,
  the image of `(XStep, YStep)` under the pattern matrix), so skewed and
  rotated patterns repeat along their own axes. A lattice finer than a pixel
  is served as its mean colour and coverage. Generic over the pixel type,
  so a cell rasterised in ink tiles an ink canvas.
- Shading patterns — axial, radial, function-based, Gouraud triangle mesh

**SIMD module** (`simd/`)
Per-arch dispatch helpers; each arch gets its own function, not a cfg/return chain:
```
blend_solid_rgb8 / blend_solid_gray8  (movdir64b → AVX2 → NEON → scalar)
composite_aa_rgb8_opaque              (AVX-512 → scalar)
unpack_mono_row                       (SSE2 → NEON → scalar)
```
The 8-bit kernels are reached through the sample's byte view; 16-bit
canvases take the generic scalar code.

### 3.3 `rasterrocket-font`

Font programs, glyph rasterisation and the glyph cache.  No C font library:
programs are parsed by the local forks `crates/read_fonts` and `crates/skrifa`
(vendored like `crates/jpeg2000`; see their `CHANGES.md`).

- `FontProgram` — detects the format from the leading bytes (sfnt TrueType /
  OpenType / collection, bare CFF incl. CID-keyed, Type 1 PFA/PFB), parses it
  once, and answers charmap, glyph-name and CID → GID lookups, advances and
  unhinted outlines
- `FontEngine` — one per page renderer; parses programs, sharing one
  `FontProgram` between faces built from identical bytes or the same file, and
  draws face IDs from a `FaceIds` source that may be shared across engines
  (the interpreter shares one per document)
- `FontFace` — a program at one em → device matrix; `make_glyph*` → `GlyphBitmap`,
  `glyph_path*` → `raster::Path`; faces above 2048 px/em prefer path fill
- `PathPen` (`outline`) — outline commands → `raster::Path` with the matrix applied
- `coverage::rasterize` — exact-area, non-zero-winding coverage rasteriser
  (8-bit AA or 1-bit mask), bounded per glyph
- `GlyphCache` — `quick_cache::sync::Cache<GlyphKey, Arc<GlyphBitmap>>`, bounded
  by bitmap bytes; valid for every face whose ID comes from one `FaceIds`
  source, so the interpreter keeps one per document, shared by all its pages
- `GlyphKey` — `{ face_id, glyph_id, size_px, x_frac, x_scale, aa }`
- `FontProgram::vertical_selector` — maps a glyph to its vertical form through
  the face's `vert` / `vrt2` GSUB features, for vertical writing with a
  substitute face
- `sfnt_info` — family, weight, slant, serif class and `OS/2` code-page ranges
  of each face in an sfnt file, read from the `OS/2` and `name` tables alone,
  and a face's Unicode `Coverage`, read from its `cmap` table alone; the
  interpreter indexes installed fonts with it without loading them

### 3.4 `rasterrocket-encode`

Thin I/O layer. Takes `Bitmap<P>`, or an image as bands of whole rows
(`RowImage`, stored as a `RowLayout`), and writes file format bytes to a
`Write` sink.

- `write_ppm` — Netpbm P6 from `Rgb8`, `Bgr8`, `Xbgr8`, `Cmyk8` or `DeviceN8`;
  CMYK and DeviceN process channels convert through the process-ink model
- `write_pgm` — Netpbm P5 grayscale; a packed bitmap (`stride == width`) is
  written in 64 KiB slices, a padded one row by row
- `write_pbm` — Netpbm P4 1-bit
- `write_png` — `png` crate; `Gray8` as grayscale, the colour modes of
  `write_ppm` as RGB (RGBA when the bitmap has an alpha plane)
- `write_jpeg` — JPEG of a `Gray8`, `Rgb8`, `Bgr8` or `Cmyk8` bitmap (CMYK
  as Adobe YCCK, `APP14` transform 2) under `JpegOptions` (quality,
  progressive, optimised tables); `jpeg_gray` is the baseline grayscale form
  used by `encode_for_gcv`
- `write_tiff` / `TiffWriter` — baseline TIFF of RGB, RGBA (unassociated
  alpha), gray, CMYK (`Separated`, `InkSet` 1) or bilevel pages, 8 or 16
  bits per sample, uncompressed, PackBits, LZW, Deflate or, for bilevel
  pages, CCITT G3 / G4; a file past the 4 GiB of classic TIFF is written as
  BigTIFF
- Row-streamed writers — `PnmWriter` (PPM, PGM, PBM, 16-bit of maxval
  65535), `PngWriter` (bit depth 8 or 16), `TiffStream` / `TiffPageEncoder`
  (on a seekable sink, strips first and the directory last) and
  `PdfPageEncoder` take an image band by band, so a page larger than memory
  is written as it renders; the bytes do not depend on the band heights.
  `TiffPage::encode` and `PdfPage::encode` run through the same encoders
- `PdfWriter` — the `--print` document: a PDF of page images, 8 or 16 bits
  per component
- `IccProfile` — an RGB, gray or CMYK ICC profile, its header checked;
  written as a PNG `iCCP` chunk, TIFF tag 34675, JPEG `APP2` segments, or
  an `ICCBased` colour space on every page image of a `PdfWriter`. A writer
  refuses a profile of another colour space than its image's

No pixel processing beyond colour-mode conversion at the boundary.

### 3.4a `rasterrocket-parser` (`crates/pdf`)

The file-structure layer: a lazy, memory-mapped parser. `Document::open`
reads only the cross-reference; objects are parsed on first access and
cached as `Arc<Object>`.

- **Byte source** (`document.rs`). A file of at least 256 KiB is
  memory-mapped, a smaller one read into memory;
  `OpenOptions::read_into_memory` reads a file of any size, for a file that
  may change while the document is open. `Document::from_bytes_owned` takes
  an owned buffer and `Document::from_shared` any
  `AsRef<[u8]> + Send + Sync` value (an `Arc<[u8]>`, a mapping the caller
  holds) without a copy; the value must return the same bytes on every
  call, since offsets are read from it once.
- **Cross-reference** (`xref.rs`). The chain of sections is walked newest
  first in a loop over visited offsets. Each section contributes its own
  entries, then those of its `/XRefStm` stream, before `/Prev` is followed
  (ISO 32000-1 §7.5.8.4). The first entry seen for an object number wins,
  free entries included, so an older section cannot resurrect a freed
  object.
- **Recovery** (`repair.rs`). When the chain cannot be read, or yields no
  `/Root`, one pass over the file records every `N G obj` header. It steps
  over each parsed body, so headers inside strings or stream data are not
  taken. It also expands `/Type /ObjStm` containers and merges `trailer` and
  `/Type /XRef` dictionaries. When an xref entry leads to the wrong object,
  to a free one, or is absent from the table, the same header scan locates
  the object by number, either as a direct header or as a member of a
  scanned object stream. If the scan has no header for that number, the
  object is missing; a different object is never returned in its place.
- **Header base.** Bytes before `%PDF-` (a mail header, a wrapper) offset
  every position in the file. `read_xref` returns that base with the table
  and `Document::body()` exposes the file from the header on, so offsets
  are applied as written.
- **Object streams** (`objstm.rs`). Each container is decoded and its
  directory read once per document. Objects are parsed on demand and looked
  up by number; the xref index is a hint checked against the directory.
  A container is decoded only when it is an object stream (`/Type /ObjStm`,
  or untyped with integer `/N` and `/First`), and it is always read from a
  direct entry, so object streams never nest.
- **Streams and filters** (`stream.rs`). The decode filter chain, with
  bounded output. Abbreviated filter names (`/Fl`, `/A85`, …) are accepted
  on every stream, `BrotliDecode` is decoded, and the TIFF predictor is
  undone at 1, 2, 4, 8 and 16 bits per component. The same chain decodes
  cross-reference streams, and its filter set (`is_decoded_filter`) is the
  image path's set of transport filters. A page's `/Contents`
  elements decode independently: an element that fails is skipped with a
  warning, and the page fails only when none decodes. A filter that keeps
  the bytes decoded before corrupt or missing data (Flate, LZW, ASCII85,
  RunLength, Brotli) reports the truncation, and the element is listed as
  a partial `ContentFailure`.
- **Objects** (`object.rs`). Array and dictionary nesting is capped at 256
  levels. A dictionary value that does not parse drops only its key; the
  rest of the dictionary is kept.
- **Page tree** (`page_tree.rs`). Logarithmic descent by `/Count`. When a
  node's `/Count` equals its number of kids the kid is indexed directly,
  then probed: an interior node continues the descent. The descent and the
  eager walk (`get_pages`) classify nodes by one rule (`is_interior`):
  `/Type /Page` is a page, `/Type /Pages` is interior, and any other node
  is interior exactly when it has `/Kids`.
- **Derived values** (`derived.rs`). `Document::derived(id, make)` memoises
  a value that a higher layer computes from an object, keyed by object and
  type, for the lifetime of the document. The interpreter uses it to parse
  each `JBIG2Globals` stream once per document rather than once per image.
- **Encryption** (`crypt/`). The Standard security handler (ISO 32000-2
  §7.6): revisions 2–4 derive the file key from the password with MD5 and
  check it against `/U` and `/O` (RC4); revisions 5 and 6 hash the password
  with SHA-256 (and, for 6, the AES/SHA-2 iteration of Algorithm 2.B) and
  unwrap the key from `/UE` or `/OE` with AES-256.  The empty user password
  is tried first, then the caller's password as user and as owner password.
  Every string and stream of an indirect object is decrypted as the object
  is parsed, under the per-object key of Algorithm 1 (RC4, AESV2) or the
  file key (AESV3), by the cipher its crypt filter names (`/StmF`, `/StrF`,
  a stream's own `/Crypt` filter, `/Identity`); cross-reference streams,
  objects inside object streams, the `/Encrypt` dictionary and (under
  `/EncryptMetadata false`) metadata streams are left as written.  Whether
  decryption is permitted at all is the caller's decision
  (`OpenOptions::decrypt_authorized`); a password is only an input to it.
  Public-key encryption (`/Adobe.PubSec`) is not supported and is refused by
  name.

A reference that resolves to nothing is the null object (§7.3.10). For
example, a missing `/Contents` element is skipped and the rest of the page
is drawn.

### 3.5 `rasterrocket-interp`

PDF parsing and operator dispatch. Uses the in-tree `rasterrocket-parser` crate (lazy mmap parser).

**Sub-modules**

```
content/
  tokenizer.rs       — byte-level PDF token scanner; nesting capped at 256
  operands.rs        — operand stack + type coercion
  operator.rs        — 150+ typed operator variants
  mod.rs             — parse(bytes) → Vec<Operator>

intent.rs            — RenderingIntent and IntentSet: the intents a page's
                       marking operators painted under
output_profile.rs    — OutputProfile: a finished page converted to an RGB or
                       gray ICC output profile, compiled once per intent and
                       black-point compensation (working_space.rs: the sRGB
                       source with the intent's adjustments); a CMYK
                       profile's separation table
separation.rs        — CanvasRequest (sRGB, overprint preview, CMYK) and the
                       Separator that turns non-ink colours into ink: the
                       process-ink inverse, a CMYK profile's separation, or
                       the ISO 32000-2 §10.4.2.3 device conversion
optional_content.rs  — OptionalContent: the /OCProperties /D configuration
                       resolved once per session for a RenderPurpose; OCG and
                       OCMD (/P, /VE) visibility for BDC /OC, XObject /OC and
                       annotation /OC
page_group.rs        — the page group (§11.4.7): whether a page is painted as
                       an isolated group over white (a knockout page group, or
                       a blend mode reachable from its resources or annotations);
                       page_uses_overprint: whether an ExtGState reachable the
                       same way sets OP or op. Both walk forms to the
                       renderer's form depth
prescan.rs           — classify page content without rendering pixels
rotation.rs          — Rotation: quarter turns added to a page's /Rotate
                       (PageGeometry::rotated)
unsupported.rs       — UnsupportedFeature: ignored features found by a
                       structural scan under a step budget
form_type.rs         — FormType of the catalog's form (AcroForm, XFA)
cache/               — device image-cache helpers + background prefetcher
                       (`cache` feature)

renderer/
  mod.rs             — re-exports PageRenderer, PageView, PageDiagnostics
  gstate.rs          — CTM, graphics state, GStateStack; q depth capped at
                       16 384, q/Q balanced within each nested stream
  color.rs           — colour values → device components (DeviceCMYK via
                       the process-ink model in `color::cmyk`)
  text.rs            — text state (Tc, Tw, Tz, TL, Ts, Tr)
  font_cache/        — font loading; per-page FontCache, per-document
                       DocumentFonts store and GlyphCache; code_to_gid.rs:
                       the code → glyph table of a loaded face; vertical.rs:
                       vertical forms drawn from horizontal glyphs
  document_images.rs — decoded image XObjects shared by the pages of a
                       document, kept from a second page's request on
  document_binding.rs — the one document a per-document store serves
  encodings.rs       — base encodings of Annex D
  fallback.rs        — per-character fallback of a substitute to installed faces
  substitutes.rs     — substitute programs for non-embedded fonts
  system_fonts.rs    — installed CJK / Arabic / Thai faces; per-character fallback faces
  bundled_fonts.rs   — Noto fallback faces compiled in (`bundled-fonts`)
  page/
    mod.rs           — PageRenderer: construction, execute(), finish()
    canvas.rs        — PageCanvas: an RGB or an ink (DeviceN8 / DeviceN16)
                       bitmap at 8 or 16 bits; with_canvas! dispatch;
                       InkPattern converts sRGB sources (shadings, tiling
                       cells, images, meshes) to ink
    depth.rs         — CanvasSample: paint sources and buffers at the
                       canvas's depth; 8-bit sources widened exactly
    band.rs          — PageState: what the renderers of one page's bands
                       share (decoded images, forms, shadings, tiling cells,
                       tint budget, spot slots, diagnostics)
    ink_map.rs       — InkMap: spot colorants of a page assigned to the four
                       spot channels, each with its SpotLut; the
                       ColorantPlan of each Separation / DeviceN space
    image_compositor.rs — image samples onto a page or group bitmap
    view.rs          — PageView: device-space placement and initial CTM
                       (rotation, UserUnit, window)
    operators.rs     — execute_one: one operator → renderer call
    paint.rs         — path painting: colour, clip, fill and stroke
    fill_stroke.rs   — B / B* / b / b* as one object (stroke knocks out fill)
    xobject.rs       — Do: image and form XObjects, form groups
    form_memo.rs     — parsed form XObjects of one page, kept from their
                       second use on
    image_blit.rs    — a decoded image or stencil mask onto the page through
                       the clip, on the CPU or the device blit kernel
    device_blit.rs   — a device-resident image through the blit kernel of the
                       image cache's backend, or read back to the host
    device_page.rs   — record_for_device: a page's paints (fills, strokes,
                       glyphs, images, forms, path clips, soft-masked
                       objects) recorded as PageCommands for the GPU page
                       raster
    diagnostics.rs   — PageDiagnostics, DevicePage, ImageFailure, RenderAbort
    text_census.rs   — GlyphCounts: shown glyphs by text rendering mode, or
                       by why a glyph cannot mark the page
    devices.rs       — GPU decoders, kernels and image cache of a renderer
    text.rs          — show_text: a string's glyph layout, by code or by
                       shaped run
    text_glyphs.rs   — a shown string's glyphs filled, stroked and clipped
                       as its rendering mode requires
    text_context.rs  — the neighbouring strings on a baseline that are a
                       string's shaping context
    text_stroke.rs   — per-glyph stroke coverage of stroked text (Tr 1, 2, 5, 6)
    type3.rs         — show_text_type3; coverage cache of d1 glyphs
    text_ops.rs      — text-space arithmetic shared by show_text and TJ
    text_shaping.rs  — runs of substitute-drawn text shaped with the face's
                       GSUB/GPOS
    annotations/
      mod.rs         — annotation appearance streams: /F flags, /OC,
                       appearance state, Algorithm 8.1 fit, NoRotate;
                       synthesised appearances drawn as forms
      markup.rs      — appearances synthesised for markup annotations
                       without /AP /N, from geometry and colours
      widget.rs      — appearances synthesised for form field widgets
                       (text, choice, check/radio, push button) from /V,
                       /DA, /MK (with its /R rotation) and /AcroForm /DR,
                       when /AP /N is absent or, for text and choice
                       fields, /NeedAppearances is set
      free_text.rs   — appearances synthesised for FreeText without /AP /N:
                       /C fill, border in the /DA colour, /Contents (else
                       the plain text of /RC) wrapped in the /RD box,
                       aligned by /Q; /CL callout with its /LE ending
      popup.rs       — pop-ups: drawn only when /Open; without /AP /N, a
                       window holding the parent's /T and /Contents
      line_ending.rs — /LE line endings (arrows, square, circle, diamond,
                       butt, slash) in a frame along the line
      text.rs        — /DA parsing, the /DR-derived resources, text strings
                       and rich text reduced to plain text
      typeset.rs     — Unicode text set as runs of codes in font resources;
                       line wrapping and layout; each line in visual
                       order (UAX #9), mirrored at right-to-left levels
      encoder.rs     — Unicode → codes of a font (inverse of its encoding,
                       ToUnicode or collection); CJK fallback Type 0 fonts
                       for characters the /DA font cannot encode
      content.rs     — object access and content-stream fragments shared
                       by the synthesised appearances
    patterns.rs      — tiling patterns on the pattern lattice
    pattern_memo.rs  — resolved shadings and tiling cells of one page
    shading_paint.rs — `sh` and shading patterns
    soft_mask.rs     — ExtGState SMask: luminosity / alpha masks; one
                       object's painting under the mask in effect
    knockout.rs      — one object's painting in a knockout group
    pixel_rect.rs    — device-pixel rectangles (group and snapshot bounds)
    image_sampler.rs — fixed-point sampling grid; nearest / bilinear
    image_reduce.rs  — area-averaging of minified images
    image_memo.rs    — decoded images of one page: XObjects by object,
                       inline images by context and bytes

resources/
  mod.rs             — PageResources; lazy resolve_*
  dict_ext.rs        — typed accessors for `rasterrocket_parser::Dictionary`
  font/              — font dictionaries and descriptors; cid.rs: Type 0
                       fonts, CIDToGIDMap, W / W2 metrics; type3.rs: Type 3
                       fonts, each CharProc parsed once
  font_substitute.rs — typeface class of a non-embedded font
  cmap.rs            — CMap / ToUnicode parsing
  cmap_predefined.rs — predefined CMaps compiled into the binary
  colorspace.rs      — colour-space taxonomy for `cs` / `CS`; Separation /
                       DeviceN tint transforms, Indexed lookups, `/None`
                       colorants, DefaultGray / DefaultRGB / DefaultCMYK
                       (selected per resource context, memoised in
                       `PageResources`)
  colorspace/
    cal.rs           — CalGray / CalRGB through WhitePoint, Gamma, Matrix
    icc.rs           — ICCBased compiled once per document: matrix/TRC
                       shaper or moxcms transform, else `/Alternate`
    lab.rs           — Lab through Range and WhitePoint, Bradford-adapted
                       to D65
  icc_shaper.rs      — matrix/TRC ICC input profiles evaluated to sRGB
  lattice.rs         — multilinear interpolation corners over a regular grid
  tiling.rs          — Type 1 tiling pattern dictionaries
  transfer.rs        — ExtGState /TR and /TR2 tabulated to 256-entry tables,
                       applied to painted colours and image samples
  shading/           — shading types 1–7
    gouraud.rs       — Type 4 / 5 triangle meshes
    patch.rs         — Type 6 / 7 patch meshes
    stream.rs        — packed mesh vertex data
    mesh.rs          — vertex colours; at most 2^22 triangles per mesh
    triangles.rs     — mesh triangles indexed by device row
    function.rs      — PDF functions compiled once per document, evaluated
                       by arithmetic (Types 0, 2, 3, 4)
    function/        — sampled.rs (Type 0), postscript.rs (Type 4)
  image/
    mod.rs           — ImageDescriptor; XObject resolution
    dispatch.rs      — filter chain → samples → colour conversion (images
                       and masks)
    raw.rs           — raw / Flate samples expanded to one byte each
    bitpack.rs       — 1/2/4/16-bpc expansion
    codecs/          — dct/, jpx.rs, ccitt.rs, jbig2.rs; JPX decodes at
                       `ImageRequest::draw_size` unless masked; CMYK/YCCK
                       JPEG samples are ink; dct/stream.rs reads the frame
                       header that routes and budgets a JPEG and repairs
                       two producer faults before any decoder reads it
    icc_device.rs    — ICC CMYK → RGB through the profile's lookup table on
                       the session's backend (see §3.6)
    colorspace/      — ImageColour: the samples' colour space and their
                       conversion to gray / RGB
    smask.rs         — SMask decode and /Matte removal
    inline.rs        — BI … ID … EI; data length from the dictionary, a
                       named colour space resolved through the resources
```

**Image colour** (`resources/image/colorspace/`)

Every codec yields samples in the image's own colour space, before
`/Decode`: stencil bytes, or `n` bytes per pixel.  `ImageColour`, resolved
once per image from the dictionary and the `/ColorSpace` resources in
effect, converts them through the same `ColorSpace` conversions the fill
operators use, so an image is drawn exactly as a fill of the same colour
would be: `CalRGB` and `CalGray` parameters, Lab, ICC profiles (matrix/TRC
as a shaper, any other through `moxcms`, an unusable one through its
`/Alternate`), `Separation` / `DeviceN` tint transforms, `Indexed`
palettes, and the `/Default*` space of a device space.  A one-component
image converts through a 256-entry table with `/Decode` folded in, kept one
byte per pixel when every entry is neutral; three- and four-component
images remap `/Decode` at 8 bits and then convert (a shaper, a profile
transform, the device CMYK model, or a memoised per-pixel conversion for Lab
and `DeviceN`).  An image whose colorants are all `/None` is not drawn.
A JPEG 2000 image whose dictionary names no colour space converts through
the ICC profile of its `colr` box, when it has one.  Colour-key masks are
compared with the samples before any of this.

**Image decode budget** (`resources/image/limits.rs`)

An image's sides may each reach `MAX_IMAGE_SIDE` (131 071), and its
decoded samples `width × height × components` may reach
`MAX_DECODED_BYTES` (1 GiB); a larger image is refused before anything is
decoded or allocated and reported as a decode failure, so a hostile
dictionary cannot drive the process to an allocation abort.  Within the
budget every sample buffer the decoders and the colour conversion create
is reserved fallibly: an allocation the machine cannot satisfy fails the
image, not the process.

**Ink canvas** (`renderer/page/canvas.rs`, `ink_map.rs`)

A page simulated under overprint is painted in `[C, M, Y, K, S0..S3]`
ink and converted to sRGB once finished. The first four distinct spot
colorants met on the page take the four spot slots; the `InkMap` is shared
by the renderers of the page's tiling cells and Type 3 glyphs, so a spot
takes one slot throughout the page. Each `Separation` / `DeviceN` space
resolves once per page into a `ColorantPlan`: the channels each component
paints, or, when its spots do not all hold a slot, the process ink of its
alternate colour (listed in `PageDiagnostics::unslotted_spots`). Solid
colours, shadings (their ramps and mesh vertices carry ink samples) and
sampled images (their tints resampled beside the sRGB samples) paint
through the plan, so they write only their colorants' channels and leave
the others to the overprint mask. A coloured tiling cell is rasterised on
an ink child canvas, and the tile paints the union of the channels its
paints wrote. An ICC-based CMYK solid colour paints the process ink that
reproduces the profile's sRGB colour (`color::cmyk::ink_reproducing`,
memoised per page), keeping its zero components zero. Transfer functions
act on ink as on the additive complement: red, green and blue on C, M and
Y, gray on K and the spots; ink derived from sRGB is converted first and
then mapped. The device blit's samples go through the image compositor's ink
conversion, so it produces the CPU's ink bytes.

The same canvas serves CMYK output (`CanvasRequest::Cmyk`): overprint is
simulated only when the session asks for it, and the finished page is its
process ink with the slotted spots folded in (`FinishedCanvas::Cmyk8` /
`Cmyk16`). `DeviceCMYK`, `ICCBased` CMYK and the process colorants of
`Separation` / `DeviceN` colours keep their values; every other colour is
separated as it is painted, through the CMYK output profile's table
(`OutputProfile::separation`) or the ISO 32000-2 §10.4.2.3 device conversion.

**16 bits per sample** (`renderer/page/depth.rs`). A page canvas, RGB or ink,
holds 8- or 16-bit samples, chosen per render. Shadings evaluate straight to
16 bits, mesh shadings subdivide to a tolerance of the canvas's depth,
images of more than 8 bits decode at 16 bits, and soft masks, groups and
blend modes composite at the canvas's depth.

**Bands** (`renderer/page/band.rs`). A page rendered in bands is drawn by
one renderer per band window over the whole content stream; `PageState`
carries what one renderer decoded and resolved to the next, so each image is
decoded once per page, spot slots are assigned as in the whole-page render,
and the page reports one set of diagnostics.

**Fonts** (`renderer/font_cache/` and its neighbours)

- **Embedded programs.** A font with an embedded program draws it. Simple
  fonts map code → glyph through `/Encoding` and `/Differences` over a base
  encoding (`renderer/encodings.rs`, the tables of Annex D), then the program's
  own cmap or glyph names.
- **Substitutes for non-embedded fonts.** `resources/font_substitute.rs`
  classifies a non-embedded font as sans-serif, serif or monospaced, regular,
  bold, italic or bold italic. The class is read from `/BaseFont` first and
  from the descriptor's `/Flags`, `/FontWeight` and `/ItalicAngle` where the
  name is silent. `renderer/substitutes.rs` then picks an installed
  metric-compatible face (Liberation, Arial, Times New Roman, Courier New)
  when one is found. Otherwise it uses the base-14 face compiled into the
  binary (`crates/pdf_interp/fonts/foxit/`, BSD-3). Symbol and ZapfDingbats
  always draw with their compiled-in faces and built-in encodings.
- **Installed fonts.** A `FontLibrary` (`renderer/system_fonts.rs`) walks
  the session's font directories (`SessionConfig::font_dirs`, from
  `RROCKET_FONT_DIR` by default), then, unless `SessionConfig::system_fonts`
  is off, the standard font directories of the build target (the Windows
  and macOS ones only on those targets), on first use, and indexes every
  sfnt face by family and `OS/2` code-page range. One walk lists the font
  files of a library: a directory reached through several paths or a
  symbolic link is read once (keyed by device and inode on Unix), and the
  walk stops at 10 000 directories with a warning. The installed
  metric-compatible substitutes are looked up in the same listing. Libraries are shared
  process-wide per directory list and switch; each holds its parsed
  programs, fallback rankings and installed substitutes.
- **Scripts the bundled faces do not cover.** For CJK, Arabic and Thai the
  library selects a face by script, serif class, weight and slant. A non-embedded CID
  font whose ordering is Japan1, GB1, CNS1 or Korea1 maps CID → Unicode →
  glyph in that face.
- **Per-character fallback.** A code whose Unicode value is known (its
  glyph name in a simple font; its CID in a known collection, or its
  `ToUnicode` entry, in a composite font) and which the substitute cannot
  draw is drawn from the first installed face, ranked by serif class,
  per-script preference, weight and slant, whose `cmap` maps it. A face's
  `cmap` coverage is read when a lookup first reaches it. Resolution is
  lazy (`renderer/fallback.rs`): the codes a text operator shows are
  looked up before its glyph loop, each character once per font, and the
  fallback face is loaded at the font's matrix. Every script an installed
  font covers draws, whatever the font's classified script.
- **Bundled fallback faces.** With the `bundled-fonts` feature (on by
  default in `rasterrocket` and the CLI), Noto subsets for Greek and
  Cyrillic, Hebrew, Arabic, Devanagari, Thai, Bengali, Tamil, Armenian,
  Georgian and Ethiopic (sans-serif and serif) and three symbol faces are
  compiled in (`crates/pdf_interp/fonts/noto/`, zlib-compressed, ~0.75 MB;
  `renderer/bundled_fonts.rs`). They are the last resort of the
  per-character fallback, after every installed face. The faces of scripts
  that need shaping keep their layout features; the Devanagari, Bengali
  and Tamil faces also carry U+25CC, the dotted circle the shaper inserts
  for a broken cluster, and the Bengali and Tamil ones the danda and
  double danda.
- **Shaping.** In a string shown with a substitute, the codes of a run in a
  script that needs shaping (right-to-left: Hebrew, Arabic, Syriac and
  neighbours; left-to-right: the Indic scripts, Thai, Lao, Tibetan, Myanmar,
  Khmer, Mongolian) drawn by one face are shaped together with that face's
  `GSUB`/`GPOS` (`renderer/page/text_shaping.rs`, through the harfrust fork
  in `crates/harfrust`). A right-to-left run, in visual order in the string,
  is reversed to logical order first. The shaped glyphs fill the advance
  the document gives the run's codes, condensed when wider and centred when
  narrower, so the pen moves exactly as without shaping. A run never
  crosses a show string, but the characters of the strings shown next to
  it on its baseline (`renderer/page/text_context.rs`: the next string read
  ahead, the previous one kept; both in the same content stream, at most
  0.1 em apart or 0.5 em overlapping, separated only by operators that change the text state, the colour or
  marked content) are its shaping context, so a word split between strings
  joins across the split.
- **Composite fonts.** `resources/cmap.rs` parses `CMap` streams: codespace
  ranges of mixed width, `usecmap`, and `ToUnicode`. The predefined `CMaps`
  of ISO 32000-2 Table 116 are compressed into the binary
  (`crates/pdf_interp/cmaps/*.z`) and parsed on first use
  (`resources/cmap_predefined.rs`).
- **Vertical writing.** With `WMode 1` the glyph origin moves by the
  descendant's `/W2` / `/DW2` metrics and the pen advances down by `w1y`.
  Substitute faces take their vertical glyph forms from GSUB, as do the
  vertical-form CIDs of a collection (those only its `UCS2-V` `CMap`
  reaches), in either writing mode. A collection CID keeps every code
  point that maps to it and draws the first the substitute maps.
- **Stroked text.** In the stroking render modes, under a solid, opaque,
  undashed stroke in the normal blend mode with antialiasing on, each
  glyph's stroke is rendered once into coverage at its quarter-pixel phase
  (`renderer/page/text_stroke.rs`), kept in the document glyph cache under
  a face identifier derived from the face and the stroke parameters, and
  composited in the stroke colour. Any other stroke strokes the union of
  the run's device-space outlines as a path.
- **Type 3 glyphs.** A `CharProc` runs per glyph under the glyph-to-device
  matrix. A `d1` glyph of an indirect font, shown with an opaque colour in
  the normal blend mode, is a stencil: its coverage is rendered once per
  matrix and 1/16-pixel phase by running the `CharProc` in a child renderer
  onto a transparent bitmap sized from the declared bounding box, kept in
  the document glyph cache and composited in the fill colour. Content that
  reaches the bitmap's margin runs directly instead.
- **Sharing.** Descriptors, parsed programs, code → glyph tables and glyph
  bitmaps of indirect font dictionaries live in a `DocumentFonts` store
  shared by every page of a document. Installed substitute and fallback
  programs are shared by every document of a font library, compiled-in
  substitutes process-wide.

**PageResources** — holds an immutable reference to the `rasterrocket_parser::Document`. All resource
resolution is stateless and on-demand; no per-page pre-computation. This
makes unit testing straightforward: inject a `Document` and page `ObjectId`.

**GPU integration points** (feature-gated; `renderer/page/device_blit.rs`,
`device_page.rs`, `resources/image/icc_device.rs`, the thresholds in
`resources/image/thresholds.rs`)
- DCTDecode: `gpu::NvJpegDecoder::decode_sync()` if pixels ≥
  `GPU_JPEG_THRESHOLD_PX` (dormant: `u32::MAX`)
- JPXDecode: `gpu::NvJpeg2kDecoder::decode_sync()` if pixels ≥
  `GPU_JPEG2K_ATTACHED_THRESHOLD_PX` (2 097 152) for the decoder `ForceCuda`
  attaches, or `GPU_JPEG2K_THRESHOLD_PX` (dormant: `u32::MAX`, so no
  decoder is built) under `Auto`; `PDF_RASTER_JPX_THRESHOLD` overrides both;
  under `Auto` only when the draw
  does not reduce the image to half its size or less on both axes, and the
  thread's decoder is then built on demand
- ICC CMYK→RGB: with a GPU backend in the session, an image of at least
  `GPU_ICC_CLUT_THRESHOLD` pixels (128 000; `PDF_RASTER_ICC_CLUT_THRESHOLD`
  overrides) whose conversion is tabulated converts through the table's
  grid (`color::grid::CmykGrid`) by `gpu::backend::icc_clut` on the
  device, which interpolates it in the integer arithmetic of the CPU
  conversion; the Vulkan backend when the session holds one, otherwise
  CUDA. Smaller images, and those the device declines or fails, convert on
  the CPU, so a GPU session's output is a CPU session's byte for byte. A
  Vulkan device that fails a one-time exactness probe
  (`VulkanBackend::icc_clut_is_exact`) is not used.
- Image blit: an image held in the device image cache is sampled by the
  blit kernel of the cache's backend and composited on the host; a draw
  that reduces the image, or a blit that does not start, draws through the
  image's host copy.
- Whole page: `PageRenderer::record_for_device` (see §3.6, *Page raster*).

GPU decoders, CUDA streams, page-locked staging buffers and the NPP rotator live in
thread-local storage on rayon worker threads (one set per thread). They are torn down
eagerly via a `rasterrocket::release_gpu_thread_state()` broadcast before the rayon
pool drops — this avoids the CUDA driver teardown race at process exit. A thread's
stream or staging set on a context whose `GpuCtx` has dropped is also dropped when the
thread next creates one (`gpu::live`).

The CUDA context (`GpuCtx`) and the Vulkan backends are held weakly by process-wide
registries (`registry::shared`): sessions alive at once share them, the last to drop
destroys them, and a `DeviceHold` keeps them open across sessions.

**Error handling**
- Malformed operators — silently discarded (lenient parsing; real PDFs have junk)
- Missing resources — the operator naming one draws nothing; a font name the
  resources do not define draws in Helvetica and is listed in
  `PageDiagnostics::missing_fonts`
- JavaScript — **not** an error. There is no JavaScript engine and `/JS` is never
  decoded, so a script's structural presence cannot change the static render.
  Detection is structural (`/S /JavaScript` only) and emits a loud `WARN` per
  entry point; the document opens and renders normally. The catalog entry
  points are checked at open; page `/AA` and annotation `/A`/`/AA` are checked
  per rendered page, warned once per session.  `unsupported.rs` reports
  every ignored feature as a typed `UnsupportedFeature`: per document
  (`document_unsupported_features`: catalog JavaScript, XFA, portable
  collection), returned by `RasterSession::unsupported_features`, and per
  page (`page_unsupported_features`: page and annotation JavaScript, 3D and
  multimedia annotations), in `PageDiagnostics::unsupported_features` from
  a render and from a prescan. Detection reads only dictionary keys and
  `/S` / `/Subtype` names, never a script, model or media stream, and each
  scan is bounded by a step budget. `InterpError` (`#[non_exhaustive]`) has no
  `JavaScript` variant — its variants are `Pdf`, `PageOutOfRange` and
  `InvalidPageGeometry`.
- XFA — not processed. `form_type.rs` classifies the catalog's form
  (`/AcroForm`, `/XFA`, `/NeedsRendering`) as `FormType`; a dynamic XFA form
  warns once at open, and the type is carried on the session and in every
  page's `PageDiagnostics::form_type`.

### 3.6 `gpu`

GPU compute kernels (CUDA + Vulkan), CUDA decoders, and VA-API decoders.  Not
linked unless at least one `gpu-*` / `vaapi` / `vulkan` / `cache` feature is
active.

**Backend abstraction.**  `crates/gpu/src/backend/` factors the per-page state
machine out of `GpuCtx`:

```text
GpuBackend trait          (backend/mod.rs)
├── CudaBackend           (backend/cuda/)        — wraps GpuCtx + per-page recorder
└── VulkanBackend         (backend/vulkan/)      — ash 0.38 + gpu-allocator + slangc-compiled SPIR-V
```

The trait surface is `begin_page → record_* → submit_page → wait_page`
(`run_page` wraps the four and abandons a page whose step fails), with two
kernel `record_*` methods (blit, ICC CLUT), the copy and zero-fill records, `alloc_device` / `alloc_host_pinned`
/ `upload` / `download`, and `scratch()`.  GATs (`type DeviceBuffer`) make it
usable only as a generic parameter, not `dyn` — callers monomorphise per
backend; the trait is sealed to the two backends of the crate.  The
backend-generic operations (`backend::blit_image`, `backend::icc_clut`) each
run as one page: uploads, kernel and download.
Their device buffers come from the backend's `DeviceScratch`, one
`ScratchPool` per operation, which lends a scratch to each operation in
flight and keeps it, at the largest size it served, for later pages.  The
renderer holds `Option<Arc<GpuCtx>>` for CUDA and `Option<Arc<VulkanBackend>>`
for Vulkan as parallel fields and the image cache as the backend-erased
`gpu::cache::ImageCache`.

**`GpuCtx`** (CUDA) — one per process, holds the CUDA context and all compiled
PTX kernels as `cudarc` modules.  Shared via `Arc<GpuCtx>`.  `cudarc` is pinned
to the `cuda-13030` driver-API binding, so a CUDA 13.x driver is required.

**`VulkanBackend`** — one per device selector (`SessionConfig::vulkan_device`),
shared by every session naming it; loads the Vulkan instance, picks a device
(an index or a name substring when selected, otherwise ranked discrete >
integrated > virtual > CPU), creates a single compute queue, and lazy-loads
SPIR-V → `VkPipeline` per kernel.  Persistent `VkPipelineCache` blob at
`$XDG_CACHE_HOME/rasterrocket/vulkan_pipeline_cache.bin` across runs.  Shared
via `Arc<VulkanBackend>`.  The renderer dispatches the ICC grid conversion
and the cached-image blit through it, and `VulkanBackend::render_page`
renders whole pages (*Page raster* below).

**Backend start-up under `Auto`.**  A session under `BackendPolicy::Auto`
initialises no GPU backend, image cache or nvJPEG decoder: it renders on the
CPU, except JPEG 2000 images for which, when `PDF_RASTER_JPX_THRESHOLD` sets a
pixel area, a thread's nvJPEG2000 decoder is built on demand, and pages `gpu_pages` sends to a Vulkan device it opens when the first page that may render on it is reached.
`ForceCuda` / `ForceVulkan` initialise their backend.

**Kernel inventory.**  Two kernel sources exist in **both** `.cu` (CUDA, compiled to
PTX by `nvcc`) and `.slang` (Slang, compiled to SPIR-V by `slangc`) at build
time, gated on the `vulkan` feature; `page_raster` exists in Slang only, compiled twice.

| Kernel | CUDA file | Slang file | Operation | Threshold |
|---|---|---|---|---|
| `icc_cmyk_clut` | icc_clut.cu | icc_clut.slang | ICC CMYK→RGB through a tabulated conversion's grid, in its simplices in integers (byte-identical to `CmykGrid::convert`) | 128 000 px |
| `icc_cmyk_matrix` | icc_clut.cu | — | profile-free CMYK→RGB by the process-ink model; not dispatched by the renderer | — |
| `blit_image` | blit_image.cu | blit_image.slang | Cached-image sampling onto an RGBA target (`cache` feature) | always |
| `page_raster` | — | page_raster.slang (+ page_image.slang, page_modifiers.slang) | Whole page from a `PageCommands` list, one workgroup per 16-px tile: coverage tables, glyph masks, images, path clips and soft-masked objects | `gpu_pages`, device without `shaderInt64` |
| `page_raster_paths` | — | page_raster.slang with `PAGE_PATHS` | As `page_raster`, and computes the exact-area coverage of outlines with the CPU rasteriser's integer walk; needs 64-bit integers | `gpu_pages`, device with `shaderInt64` |

Four kernel-level parity tests in `crates/gpu/tests/cu_vs_slang_parity.rs` (ICC
grid and blit) run the Vulkan kernel against the CUDA kernel or the host
evaluator and require byte-equal output.

**Page raster** (`gpu::page`, `backend/vulkan/page.rs`, Vulkan only).  A
`PageCommands` list holds a page as paint commands over an opaque RGB page
of 8 or 16 bits per sample, each limited to an inclusive pixel rectangle and
painted in push order: one colour at a constant alpha whose per-pixel
coverage is a coverage table the CPU's exact-area rasteriser computed
(`CoverageRows`), a closed outline whose coverage the device computes with
that rasteriser's integer walk (`AreaPath`), or a coverage mask the page
holds once and any number of commands place (`MaskId`, the glyph cache's
bitmaps); or an image sampled per pixel through the CPU image sampler's
tables (`ImageDraw`).  A command may be clipped by the coverage tables of
the CPU's path-clip regions (`ClipId`), and the commands of an object
painted under a soft mask are blended back toward the pixels before them by
the mask's plane (`PlaneId`).  Tables, outlines, masks and samples are
copied into shared flat arrays, and the device inputs, binning included,
are bounded by `PAGE_COMMANDS_MAX_BYTES` (128 MiB): a push past it fails
and leaves the list unchanged.  The host bins the commands into 16-pixel
tiles (`PageCommands::bin`); an outline's tile lists the segments that may
deposit a cell in it and its backdrop, and an outline tile covered fully
and opaquely hides the items below it.  `VulkanBackend::render_page` stages
the inputs, runs one workgroup per tile that walks its items in order and
composites each pixel as the CPU's `aa` pipe (or, for an image at full
opacity, the CPU's direct image composite) does, and reads the page back,
in one submission and one wait.  Outlines need `shaderInt64`
(`page_raster_paths`); on a device without it an outline is recorded as
its coverage table and the page runs `page_raster`, and so is an outline
past the device's integer range (`PATH_COORD_LIMIT`, `PATH_SEGMENTS`,
`PATH_TILE_SEGMENTS`).  `gpu::page::reference`
is the host model of the kernel, independent of the backend.

The interpreter records a page (`PageRenderer::record_for_device`,
`renderer/page/device_page.rs` and its `glyphs`, `images`, `clips` and
`soft_masks` modules) instead of painting it.  A fill or stroke of a solid
colour is appended as the outlines the CPU's rasteriser would rasterise
(`raster::fill::aa_fill_outlines`, `raster::aa_stroke_outline`), an opaque
hairline as the runs it paints (`raster::hairline_runs`); a glyph drawn from
a cached coverage bitmap (cached glyph strokes and cached Type 3 glyphs
included) as a mask command; an image, inline image or stencil mask, which
the CPU decodes, colour-converts and downsamples as usual, as an image
command; a form `XObject` or annotation appearance as the paints of its
content; a path clip as its regions' coverage tables, the clipping text
modes 4–7 becoming ordinary path clips at `ET`; a soft mask as the plane
the CPU renders when `gs` sets it, holding only its non-uniform part.  A
paint outside that model — a pattern or shading, a blend mode other than
`Normal`, a transparency-group form, a Type 3 glyph whose CharProc runs, a
translucent hairline or translucent fill-and-stroke, a transfer function on
an image — refuses the recording with a reason and stops the content
stream, and so does a page past the command budget ("command list size").
An ink canvas (overprint simulation; a CMYK page is on one), a page group,
vector antialiasing off, or a content-stream operator that paints by a
shading or runs a Type 3 CharProc rule the page out before interpretation
(`renderer::device_page_ineligible`), and the page queue keeps a page of
the wide working space on the CPU.  The device composites every command
with the CPU's arithmetic, so a recorded page is byte-identical to the CPU
page at 8 and 16 bits.  The queue that offers pages to the device is
`rasterrocket`'s (§3.7).

**Build-script model.**  `crates/gpu/build.rs` probes `nvcc --version` directly:
when nvcc works, real PTX is compiled regardless of features.  When nvcc fails
(no CUDA toolkit on a CI runner), 0-byte placeholder PTX files are written and
`cargo:rustc-cfg=ptx_placeholder` is emitted; `GpuCtx::init` short-circuits
under that cfg with a clear error pointing at the build host.  Slang→SPIR-V
compile is gated on `CARGO_FEATURE_VULKAN`.

**Optional decoders** (feature-gated)
- `nvjpeg` — `NvJpegDecoder`, TLS one-per-thread, primary `GPU_HYBRID`, fallback `DEFAULT`
- `nvjpeg2k` — `NvJpeg2kDecoder`, same TLS pattern; C++ exception shim in `shim/nvjpeg2k_shim.cpp`
- `gpu-deskew` — `npp_rotate::rotate_gray8()` via `nppiRotate_8u_C1R_Ctx`
- `vaapi` — `VapiJpegDecoder`; VA-API JPEG baseline decode on Linux iGPU/dGPU (AMD VCN, tested on Raphael VCN 4.0; Intel Quick Sync and Intel Arc untested); links `libva.so.2` + `libva-drm.so.2`. Dispatch priority: nvJPEG → VA-API → zune-jpeg (CPU). CMYK and progressive JPEG fall through to CPU.

**Device image cache** (`cache` feature, CUDA or Vulkan)
- `DeviceImageCache<B: CacheBackend>` — three tiers: VRAM (refcount-pinned LRU), host RAM (demote-on-evict / promote-on-hit; `cuMemAllocHost` slabs on CUDA, ordinary memory on Vulkan), disk (`<root>/<doc-blake3>/<content-hash>.bin` sidecar files, shared by both backends; opt-in via `PDF_RASTER_CACHE_DIR`).  `CacheBackend` extends `GpuBackend` with the cache's transfers outside a page; a slab is the backend's `DeviceBuffer`.  `ImageCache` and `CachedImage` erase the backend: a session builds its cache on the Vulkan backend when it holds one, otherwise on the CUDA context.
- CUDA blit: `DevicePageBuffer` — zero-init RGBA8 per page; lazy-allocated on first GPU image. Each blit samples the cached image through the same Q32 column/row tables the CPU sampler uses (`renderer/page/image_sampler.rs`), then the touched rows are downloaded, source-over composited onto the host bitmap in content-stream order, and re-zeroed.  Vulkan blit: `backend::blit_image_box` uploads the tables, zero-fills a target the size of the draw's bounding box, runs the kernel and reads the box back as one page, and the box is composited through the same `composite_rgba_rows`.  Either way cache-on and cache-off renders are byte-identical. A draw that reduces the image (`image_reduce::draw_reduces`) first copies the cached pixels back to the host (`DeviceImageCache::download`) so they are area-averaged exactly as host-resident pixels are; the kernel samples only images drawn at or near full resolution, and a JPEG drawn below its resolution is never put in the cache.
- BLAKE3 content hashing (`ContentHash::of`) keys cross-document dedup; `(DocId, ObjId)` alias keys same-document fast paths.

**CPU fallbacks** — every GPU function has a pure-Rust CPU counterpart. The
dispatch logic is in the same function; the threshold is the only branch.

**CMYK CPU path** — the process-ink model of `color::cmyk`, in three tiers on
x86-64: AVX-512 (`avx512f+avx512bw+avx512vbmi`), AVX2 and scalar, selected by
runtime detection. aarch64 uses NEON (`vld4q_u8` / `vst3q_u8`), always on.
The GPU's profile-free CMYK kernel evaluates the same fixed-point arithmetic.

### 3.7 `rasterrocket`

Public library crate. The stable API surface.

```rust
RasterOptions::new(dpi)   // #[non_exhaustive]; fields: dpi, first_page, last_page, deskew,
                          // pages, strict, purpose, annotations, text_antialias,
                          // vector_antialias, overprint, decrypt_authorized, password,
                          // read_into_memory, font_dirs, system_fonts, image_budget,
                          // max_page_bytes, page_box, rotation, transparent_background,
                          // depth, output_profile, intent, black_point_compensation,
                          // color, cancel

raster_pdf(path, opts) → impl Iterator<Item = (u32, Result<RenderedPage, RasterError>)>
raster_pdf_from_bytes(bytes, opts) → impl Iterator<Item = (u32, Result<RenderedPage, RasterError>)>
render_channel(path, opts, capacity) → Receiver<(u32, Result<RenderedPage, RasterError>)>
open_session(path, config) → Result<RasterSession, RasterError>
open_session_from_bytes(bytes, config) → Result<RasterSession, RasterError>
open_session_from_shared(impl AsRef<[u8]> + Send + Sync, config) → Result<RasterSession, RasterError>
open_session_from_reader(impl Read, config) → Result<RasterSession, RasterError>
render_page_gray(session, page_num, &RasterOptions) → Result<RenderedPage, RasterError>
render_page_rgba(session, page_num, &RasterOptions) → Result<RenderedPage, RasterError>   // transparent
render_page_cmyk(session, page_num, &RasterOptions) → Result<RenderedPage, RasterError>   // CMYK
render_page_rgb(session, page_num, scale) → Result<Bitmap<Rgb8>, RasterError>
render_page_rgb_with(session, page_num, &RenderParams) → Result<Bitmap<Rgb8>, RasterError>
render_page_rgb_with_diagnostics(session, page_num, &RenderParams)
    → Result<(Bitmap<Rgb8>, PageDiagnostics), RasterError>
    // RenderParams { scale_x, scale_y, region: Option<RenderRegion>, policy,
    //                cancel: Option<CancelToken>, transparent_background,
    //                band_rows, depth, intent, black_point_compensation, color }
render_page_canvas(session, page_num, &RenderParams) → Result<FinishedCanvas, RasterError>
    // at the requested depth and colour model: Rgb8 / Rgb16 / Gray8 / Gray16 / Cmyk8 / Cmyk16
render_page_bands(session, page_num, &RenderParams, sink: FnMut(Band) → Result<(), E>)
    → Result<PageDiagnostics, BandError<E>>   // a page as bands of rows
PageBuffer::new().render_{gray,rgb,rgba}_into(session, page_num, &opts, out, stride)
    → Result<PageInfo, RasterError>   // into a caller-owned buffer
session.page_geometry(page_num) → Result<PageGeometry, RasterError>   // size, rotation, boxes; size_px(sx, sy)
session.unsupported_features() → &[UnsupportedFeature]
prescan_session(session, page_num) → Result<PageDiagnostics, RasterError>
encode_for_gcv(page, budget) → Result<GcvImage, GcvError>   // Cloud Vision input
release_gpu_thread_state()   // call via pool.broadcast() before pool drops
DeviceHold::open(&config)    // keep the config's GPU devices open across sessions
```

The `*_from_bytes` variants render an in-memory PDF with no temp file, with
the same decryption behaviour as the path variants; `open_session_from_shared`
parses bytes the caller shares without copying them
(`Document::from_shared`), and `open_session_from_reader` reads its source to
the end into memory first. Every function and type
is exported at the crate root; the modules that implement them are private,
except `deskew`.

`RenderedPage` carries: `page_num`, `pixels: Pixels` (`U8` or `U16`, tightly
packed, in `format`: gray, RGBA under `transparent_background`, CMYK under
`ColorModel::Cmyk`, at 8 or 16 bits), `width`, `height`, `format`, `dpi`
(pixels per physical inch), and `PageDiagnostics`.

**Page budget and bands.** A render's canvas holds at most
`SessionConfig::max_page_bytes` (default `DEFAULT_MAX_PAGE_BYTES`, 1.8 GB),
counted by `CanvasLayout::bytes_per_px`: 3 bytes a pixel for 8-bit RGB, 4
with alpha, 8 on an ink canvas (a CMYK page, or a page on the overprint
preview), 9 with alpha, twice as many at 16 bits; no side exceeds
`MAX_PX_SIDE` (2²⁴). `render_page_bands` renders a page as horizontal bands
of `RenderParams::band_rows` rows, else as many as `SessionConfig::band_bytes`
hold (default 64 MiB), each band a window of the page drawn by its own
renderer (§3.5, *Bands*); stacked, the bands are the whole-page bitmap byte
for byte.

**Output profile.** `SessionConfig::output_profile` converts each finished
page, or band, to an RGB or gray ICC profile under the page's rendering
intent (`RenderParams::intent`, else the one intent its content used, else
relative colorimetric), with optional black-point compensation; a CMYK
profile separates CMYK pages as they are painted instead.

`PageDiagnostics::suggested_dpi(min, max)` — hint for re-rendering at native image
resolution. Returns `None` for vector/text-only pages.

**Page geometry.** `SessionConfig::page_box` (`PageBox`: `Media`, `Crop`
— the default — `Bleed`, `Trim`, `Art`) selects the boundary box a page is
measured and rendered in; every box other than the `MediaBox` is
intersected with it, and a missing `BleedBox` / `TrimBox` / `ArtBox` falls
back to the effective `CropBox`. `SessionConfig::rotation` (`Rotation`, a
quarter-turn count) is added to each page's `/Rotate` before the page is
drawn, so content is rasterised in the turned orientation and a quarter
turn exchanges the page's width and height in every reported size.

**Transparent background.** Under `transparent_background` the page is
painted as an isolated page group (`transparent_page_group`, knockout as the
page's `/Group` says) and finished without the composite onto paper: the
bitmap keeps the group's alpha plane, colour not premultiplied, colour and
alpha 0 where nothing was painted. Deskew is refused with it.

**Caller-owned buffers.** A `PageBuffer` keeps the page canvases between
renders (`CanvasBuffers`: the RGB bitmap's and the ink canvas's
`raster::BitmapBuffers`) and writes each finished page into the caller's
buffer at a given stride, in the page's pixel format. Once its buffers have
grown to the largest page, a render allocates no page-sized buffer: neither
canvas, alpha plane nor output. The bytes are those of the owned-buffer
entry points.

**GPU page queue** (`device_pages.rs`). With `SessionConfig::gpu_pages`
(requires the `vulkan` feature, policy `Auto` or `ForceVulkan`, and vector
antialiasing; otherwise `open_session` returns `InvalidOptions`), pages
render on the session's Vulkan device under `ForceVulkan`; under `Auto` the
device is opened on a thread of its own when the first page that is not
ruled out (`device_page_ineligible`) asks for it, and every page renders on
the CPU until it is open, so a job none of whose pages qualifies opens
nothing and a one-page job never uses it. Up to two pages (`IN_FLIGHT`)
hold the device at once: a worker whose page is not ruled out claims it
while fewer hold it, records the page and renders it there, the device
rasterising one page at a time, so one page is recorded while the other
rasterises; a worker that finds the device fully claimed, or a page ruled
out, renders on the CPU, as does a page of the wide working space. The
page canvas is faulted in before any device buffer is allocated, and not
cleared: the device writes every sample. A page whose recording is
refused, or which the device fails on, is executed again on the CPU in the
first pass's bitmap buffers, drawing the images the first pass decoded,
sharing the page's deadline, with its own operator budget. A device error
retires the device for the session; a page beyond the device's limits
renders on the CPU and leaves it in service. `render_page_bands` renders
on the CPU, so a page the CLI renders in bands (larger than `--band-mib`)
does not reach the device. Pages are returned by index, so output order
does not depend on where a page rendered; `PageDiagnostics::device_page`
(`DevicePage`) records the outcome. The device is the process-wide Vulkan
backend of its selector (`vk_init.rs`), held weakly: sessions open at once
share it, and the last of them to drop destroys it.

`render_channel` spawns a Rayon background task and feeds pages into a bounded
`sync_channel`. Typical use: `capacity=4`, Tesseract consumes from the other end
while the next page renders.

### 3.8 `hayro-jpeg2000` (`crates/jpeg2000`)

A local fork of the pure-Rust JPEG 2000 decoder, vendored with its licence
files and upstream test suite (`cargo test -p hayro-jpeg2000 --features image`).
It is not published, not released by `cargo release` (`release = false`), and
changes are never sent upstream; `crates/jpeg2000/CHANGES.md` lists every
local change. The local changes cut memory traffic under parallel decoding
(strip-mined vertical IDWT, fused MCT and level shift, no redundant zeroing or
copying) and add `Image::decode_u8`, which fuses the inverse MCT into the
8-bit output. Each keeps output byte-identical to the vendored baseline.
The fork adds an HTJ2K (ISO/IEC 15444-15) block decoder, which upstream
lacks: an HT code-block yields the sign, magnitude and bitplane the Annex D
decoder produces, so reconstruction and dequantisation are shared; a
tile-component mixing HT and Annex D code-blocks is refused when its COD or
COC marker is read. A
codestream whose header asks for more than 2^30 decoded samples over all
components, or more than 65 535 tiles, is refused before its buffers are
allocated (`ValidationError::ImageTooLarge`).

### 3.8a `read-fonts` and `skrifa` (`crates/read_fonts`, `crates/skrifa`)

Local forks of the font-parsing crates, vendored like `crates/jpeg2000`: not
published, not released by `cargo release`, never sent upstream, base recorded
in each `UPSTREAM.toml`, every local change listed in each `CHANGES.md`.
`crates/font_test_data` is a copy of upstream's test fonts, with the small
build changes its `Cargo.toml` header lists, so both upstream test suites run (`cargo test -p read-fonts -p skrifa`). The local
changes make embedded PDF font programs load where upstream rejected them
(Type 1 programs with an unknown `/Encoding`, a `%!PS-Adobe-3.0
Resource-Font` header or single-token procedures such as `{3}`; sfnt
programs without `hhea`/`hmtx`).

### 3.8b `hayro-jbig2` and `hayro-ccitt` (`crates/jbig2`, `crates/ccitt`)

Local forks of the JBIG2 and CCITT fax decoders. They are vendored in the
same way as `crates/jpeg2000`: not published, never sent upstream, the base
recorded in `UPSTREAM.toml`, every local change listed in `CHANGES.md`. The
JBIG2 fork depends on the local CCITT fork.

Both bases are ahead of their last crates.io releases by upstream fixes
(overflows, truncated-data recovery, a strict mode). The JBIG2 API parses
embedded global segments once into `Globals`, then opens each image with
`Page::new_embedded`. The interpreter builds `Globals` with the `std`
feature, which makes them `Send + Sync` so they can be shared across pages.

### 3.8c `zune-jpeg` and `rav1d` (`crates/zune_jpeg`, `crates/rav1d`)

Local forks vendored in the same way (base in `UPSTREAM.toml`, every change
in `CHANGES.md`). `zune-jpeg` decodes `DCTDecode` images, the JPEG pages of
comic archives and, through a `[patch.crates-io]` entry, the JPEG strips the
`tiff` crate reads. The fork decodes sequential frames coded in several
scans and arithmetic-coded frames (SOF9, SOF10; the `arith` feature, on by
default), reads and skips a DNL segment, and adds
`JpegDecoder::intact_rows` (the rows decoded from intact scan data; the
interpreter reports an image with fewer as partial) and
`JpegDecoder::set_max_coefficient_bytes` (a bound on the coefficients a
frame holds whole). `rav1d` decodes the AV1 pictures of AVIF comic pages
through the fork's safe single-picture entry point `decode_still`
(`still_layout` reads a stream's chroma layout without decoding); its
`asm` feature builds the assembly kernels.

### 3.9 `rasterrocket-cli`

Binary entry point. Thin glue over `rasterrocket` + `rasterrocket-encode`.

```
clap args → SessionConfig → open_session → rayon ThreadPool
  → pool.install(|| per page: prescan_session,
                    render_page_bands (streamed outputs) or render_page_canvas,
                    tone (gray, mono, invert) per band,
                    encode::PnmWriter / PngWriter / TiffStream / write_jpeg)
  → pool.broadcast(release_gpu_thread_state) // teardown before pool drops
  → main returns ExitCode                   // RAM-output guard and pool drop
```

All rendering is inside `pool.install()`. The broadcast teardown is the last thing
that happens before the pool is dropped, ensuring GPU decoders are released while
the CUDA driver is fully live. Every exit path returns from `main` rather than
calling `process::exit`, so the RAM output directory is removed after a failed
page too.

`--box`, `--rotate` and `--gpu-pages` set `SessionConfig::page_box`,
`rotation` and `gpu_pages`; `--transparent` sets
`RenderParams::transparent_background`; it requires PNG or TIFF output,
which carry the alpha plane, and PDF input. `--depth`, `--cmyk` (and
`--jpegcmyk`), `--intent` and `--bpc` set `RenderParams::depth`, `color`,
`intent` and `black_point_compensation`, `--output-profile` sets
`SessionConfig::output_profile` and tags the files with it, and
`--max-page-mib` / `--band-mib` set `max_page_bytes` and `band_bytes`
(`bands.rs`: PPM, PNG, TIFF and the `--print` spool are written band by
band; JPEG, `--fax`, `--invert=auto` and the camera stages take the page as
one band). Under `--print` without a resolution flag each page takes 600
dpi, or 360 dpi when its 600 dpi canvas would pass the bytes of a 600 Mpx
8-bit RGB page.

`--crop`, `--rectify`, `--split-spreads` and `--dewarp` run `rasterrocket-camera`
on each rendered page before encoding (PDF and comic input alike). Detection
runs on the page luma; gray and mono output is written from the processed luma,
and RGB output replays the computed geometry (`Report::replay`) on each colour
channel. Split halves are written as `out-07a` / `out-07b`.

### 3.10 `rasterrocket-camera`

Cleanup of camera-captured pages on 8-bit grayscale images, independent of PDF
and of every other workspace crate (std only). Input is a borrowed, strided
`GrayView`; output is owned, packed `GrayPage`s.

```
process(GrayView, &CameraOptions) → Result<Processed { pages, report }, CameraError>

crop or rectify   page quad: Otsu mask → max-area hull quad → sub-pixel edge refit
                  crop:    axis-aligned bounding box of the quad (a view, no copy)
                  rectify: homography onto an upright rectangle at the physical
                           aspect ratio; supersedes crop
split             gutter shadow or facing inner margins near the centre (a view)
dewarp, per page  text-line tracking → surface fit → vertical resample (experimental)
```

A detector that is not confident declines and the page passes through; the
`Report` records each stage's outcome and reason, the quad, crop rectangle,
gutter (framed and source coordinates), homography and per-stage timings.
`CameraError` is reserved for malformed input. Everything runs on the calling
thread with no global state, and a randomized sweep (`tests/no_panic.rs`)
checks that no input panics any entry point. Deskew is not part of the crate;
it stays in `rasterrocket`.

### 3.11 `rasterrocket-comic`

Comic and scan archives as a stream of `RenderedPage`s, the artefact the PDF
path produces. `open_comic(path, &ComicOptions)` returns `ComicPages`, an
iterator of `(page_num, Result<RenderedPage, _>)` in natural-sort order of
the image entries (`order.rs`); a corrupt page yields its error and the
iteration continues.

```
archive/   zip.rs (.cbz), sevenz.rs (.cb7), tar.rs (.cbt), rar.rs (.cbr: RAR 1.3–7,
           decode only)
decode/    jpeg.rs (zune-jpeg fork), png.rs, webp.rs, tiff.rs (every page of a
           multi-page file, one per image file directory), jxl.rs (first frame),
           avif/ (container.rs: the HEIF item structure read here; AV1 pictures
           by the rav1d fork; color.rs: planes to 8-bit gray or RGB)
turn.rs    the session's Rotation applied to each page image
```

Every page-image entry is decompressed when the archive opens and held until
its pages are rendered. One `ArchiveBudget` is charged by every entry
whatever the container: an entry is capped at 512 MiB and the sum over the
archive at `ComicOptions::max_archive_bytes` (default 4 GiB, `rrocket
--max-archive-mib`); an archive past it is refused with
`ComicError::ArchiveTooLarge` before the first page renders. The AVIF
container parser checks every offset against its buffer, caps every table
across the file and descends a fixed set of boxes without recursing on
input. An embedded PDF entry is rendered through `rasterrocket` with
`ComicOptions::session`.

---

## 4. Feature flag graph

```
rasterrocket-cli    → rasterrocket       → rasterrocket-interp → rasterrocket-gpu
──────────────────────────────────────────────────────────────────────────────────
nvjpeg              → nvjpeg             → nvjpeg              → nvjpeg
nvjpeg2k            → nvjpeg2k           → nvjpeg2k            → nvjpeg2k
vaapi               → vaapi              → vaapi               → vaapi
gpu-icc             → gpu-icc            → gpu-icc             → (crate only)
gpu-deskew          → gpu-deskew         →                     → gpu-deskew
cache               → cache              → cache               → cache
vulkan              → vulkan             → vulkan              → vulkan
gpu-validation      → gpu-validation     →                     → gpu-validation
bundled-fonts *     → bundled-fonts *    → bundled-fonts       →

* on by default

rasterrocket-render features (orthogonal, no cross-crate effect):
  simd-avx2        default on; enables blend/fill AVX2 paths

AVX-512 tiers are selected at runtime via `is_x86_feature_detected!`, so they
need no feature flag.
```

`dep:gpu` in `rasterrocket` and `rasterrocket-interp` is optional — the `gpu`
crate is only compiled when at least one GPU feature is active. CPU-only builds
require no CUDA toolkit.

---

## 5. SIMD and platform strategy

### Dispatch pattern

All public SIMD functions delegate to a private per-arch `dispatch_*` helper.
Each arch gets its own clean function; no cfg/return fallthrough chains.

```rust
pub fn unpack_mono_row(packed: &[u8], width: usize, out: &mut [u8]) {
    dispatch_unpack(packed, width, out);
}

#[cfg(target_arch = "x86_64")]
fn dispatch_unpack(packed: &[u8], width: usize, out: &mut [u8]) {
    /* sse2 → scalar */
}

#[cfg(target_arch = "aarch64")]
fn dispatch_unpack(packed: &[u8], width: usize, out: &mut [u8]) {
    /* neon, always present */
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn dispatch_unpack(packed: &[u8], width: usize, out: &mut [u8]) {
    /* scalar */
}
```

### Coverage by operation and platform

| Operation | File | Ryzen (AVX-512) | Intel consumer (AVX2) | ARM NEON |
|---|---|---|---|---|
| Solid fill (RGB) | `raster/src/simd/blend.rs` | `movdir64b` → AVX2 | AVX2 | `vst3q_u8` |
| Solid fill (Gray) | `raster/src/simd/blend.rs` | `movdir64b` → AVX2 | AVX2 | `vst1q_u8` |
| AA composite | `raster/src/simd/composite.rs` | AVX-512 | scalar | scalar |
| Glyph unpack | `raster/src/simd/glyph_unpack.rs` | SSE2 | SSE2 | `vtstq_u8` |
| CMYK→RGB | `color/src/cmyk/simd.rs` | AVX-512 VBMI | AVX2 | `vld4q_u8` + `vst3q_u8` |

Exact-area coverage (`raster/src/coverage/`) is integer scalar code on every
platform.

### Key platform facts

Tested on a Ryzen 9 9900X3D (AVX-512, and the AVX2 and scalar tiers) and an
Apple A18 Pro (NEON), plus CI's GitHub-hosted runners; the "Intel consumer"
column has not run on an Intel CPU since the i7-8700K testbench was retired
(2026-07). See the
README's [Tested hardware](README.md#tested-hardware).

**x86-64 (Intel consumer)**
- AVX2 is the ceiling on all Alder/Raptor/Arrow Lake. AVX-512 is architecturally
  disabled on Arrow Lake; disabled by microcode on Raptor. No current Intel consumer
  CPU has usable AVX-512 — Xeon only.
- `movdir64b` is present on i9-14900K and Core Ultra 9 285K; absent on i7-8700K.

**ARM / aarch64**
- NEON is mandatory on all ARMv8-A. No runtime detection needed — `#[cfg(target_arch = "aarch64")]` is sufficient.
- `vcntq_u8` is hardware-native on all targets (M1–M4, Cortex-A72, Cortex-A76).
  Never use `__builtin_popcountll` on ARM — it compiles to a software sequence.
- `STNP` is an allocation hint only, not a cache-bypass guarantee. For zero-fill use
  `DC ZVA` (zeros full cacheline without read). For non-zero solid fill: `vst1q_u8`.
- No kernel uses SVE2; the NEON tiers serve every aarch64 target.

### iGPU / integrated compute

**Intel iGPU (UHD/Iris/Arc laptop) and AMD VCN**
VA-API JPEG baseline decode is implemented (`vaapi` feature), tested on an AMD
Raphael iGPU (VCN 4.0, Mesa radeonsi); untested on Intel. Hardware handles
YUV surface allocation and IDCT; the result is extracted to an `Rgb8` bitmap.
Transfer overhead makes other compute (ICC, fill) not worth it on non-UMA x86.
CMYK and progressive JPEGs fall through to CPU; the `nvjpeg` feature takes
priority when both are active.

**Apple Silicon**
No native Metal backend exists; the Vulkan backend runs through `MoltenVK`,
tested on an A18 Pro (MoltenVK 1.4.2, byte-identical to the CPU). The CPU
paths use NEON. M-series chips (M1–M4) are untested, CPU and GPU alike,
except for CI's virtualised Apple-silicon runner (CPU suites only).

**Raspberry Pi 4/5**
VideoCore GPU is not accessible as a general-purpose compute API via a stable
userspace interface. NEON is the correct answer on Pi; untested on Pi
hardware.

---

## 6. Concurrency model

```
main thread
  │
  ├── rayon ThreadPool (N workers, default = logical cores)
  │     │
  │     ├── worker 0: render page N   ─── TLS NvJpegDecoder (lazy init)
  │     ├── worker 1: render page N+1 ─── TLS NvJpegDecoder
  │     └── ...
  │
  ├── GPU page device opener (gpu_pages under Auto only): started by the
  │     first page that may use the device, opens it, then exits (joined
  │     when the session drops); up to two workers claim the open device,
  │     one recording while the other's page rasterises on it
  │
  │   pool.install() blocks until all pages done
  │
  ├── pool.broadcast(release_gpu_thread_state) ← explicit teardown while driver live
  │
  └── pool dropped
```

**GPU decoder lifecycle**
Each rayon worker thread lazily initialises a `NvJpegDecoder` on first use and
stores it in thread-local storage (under `Auto`, only when the nvJPEG
threshold enables it; a thread's `NvJpeg2kDecoder` is built when its first
qualifying JPEG 2000 image is drawn, and only while `PDF_RASTER_JPX_THRESHOLD`
enables it under `Auto`); its CUDA streams, staging buffers and NPP rotator are
thread-local as well. A dropped pool's threads are not joined, so their TLS
destructors can run while the process runs its exit handlers, racing the CUDA
driver's own atexit shutdown. The fix: call `release_gpu_thread_state()` via
`pool.broadcast()` after `pool.install()` returns. This drops each object while the
driver is still fully live; the TLS destructors find nothing to destroy.

**Font engine**
Each page renderer owns its `FontEngine`. What depends only on an indirect font
dictionary (descriptor, parsed program, code → glyph table, face IDs) lives in
the session's `DocumentFonts` store, a `Mutex`-guarded map of entries whose
fields are `OnceLock`s, so it is resolved once per document; one `GlyphCache` per document
holds the glyph bitmaps of every page through `quick_cache::sync::Cache`
(sharded, no write lock on read hits).

---

## 7. Error model

**Per-document errors** — abort the whole document:
- `InterpError::Pdf` — PDF parse failure (`rasterrocket-parser`)
- (JavaScript is **not** a per-document error — it is detected, warned about, and
  rendered; see §3.5)

**Per-page errors** — skip the page, continue rendering:
- `InterpError::PageOutOfRange`
- `InterpError::InvalidPageGeometry`
- `RasterError::PageDegenerate` / `RasterError::PageTooLarge` /
  `RasterError::PageExceedsCeiling`
- `RasterError::PageAllocationFailed` — the page bitmap could not be allocated
- `RasterError::OutputBufferTooSmall` / `RasterError::OutputStrideTooSmall` —
  a `PageBuffer` render's output buffer cannot hold the page; it is left
  untouched
- `RasterError::PageBudgetExceeded` — operator budget or wall-clock deadline
- `RasterError::Cancelled` — stopped through a `CancelToken`
- `RasterError::ImageDecodeFailed` / `RasterError::ContentDecodeFailed` —
  only under `strict`

**Reported degradation** — the page renders, the loss is listed in
`PageDiagnostics`:
- an image that cannot be decoded or drawn, or is drawn only in part —
  `image_failures`
- a `/Contents` element that cannot be decoded, or decodes only in part —
  `content_failures`
- a font name the resources do not define — `missing_fonts`
- a spot colorant that found no spot slot on the ink canvas and printed as
  its process equivalent — `unslotted_spots`
- a feature the renderer ignores — `unsupported_features`

**Silent degradation** — never exposed to the caller:
- Unrecognised PDF operators — discarded
- GPU unavailable / below threshold — CPU path used
- GPU page refused or failed — the page is rendered again on the CPU with
  the same bytes (`PageDiagnostics::device_page` records where it rendered)
- Resource bounds (q depth, mesh triangles, dash count, CMap size) — the
  excess is cut off; see the CLI reference's Size limits

**Invariant violations** — panic (not user-triggerable):
- CUDA returns success but gives null handle — driver bug only
- `OUT_DIR` missing during build — cargo invariant

---

## 8. Key design decisions

**No Leptonica; no subprocess.**
Everything runs in-process, decryption included. The Poppler wrapper (`pdf_bridge`) is kept as a
reference baseline for pixel-diff regression tests; it is never linked in the
production render path.

**Generic over pixel type.**
`Bitmap<P>` and all fill functions are generic over `P: Pixel`. The compiler
monomorphises one code path per pixel format. This eliminates runtime mode
dispatch in hot loops at the cost of slightly larger binaries.

**Lazy resource resolution.**
`PageResources` holds an immutable `&Document`. Fonts, images, colour spaces,
and shadings are resolved on first access. No per-page pre-computation pass.

**Compositing pipeline is selected, not branched.**
`PipeState::no_transparency()` / `use_aa_path()` select one of three monomorphised pipeline variants
before the fill loop starts. The inner loop has no mode branches.

**GPU is additive, not required.**
Every GPU function has a CPU counterpart. Feature flags gate the GPU dep entirely.
A `cargo build` without any `gpu-*` feature produces a fully functional binary that
requires no CUDA toolkit and links no NVIDIA libraries.

**One GPU context per process.**
`GpuCtx` is `Arc<_>`, initialised once, shared by all worker threads. PTX kernels
are loaded at init time. The driver serialises concurrent CUDA calls internally.
A `VulkanBackend` is built once per device selector and shared the same way;
its single recording state serialises whole pages (`GpuBackend::run_page`).

**SIMD dispatch is per-arch, not per-feature.**
ARM NEON is unconditionally available on aarch64 — no runtime probe. x86 SIMD
uses `is_x86_feature_detected!` at runtime. Each arch gets a private `dispatch_*`
function; there is no single god-function with cfg-gated returns.
