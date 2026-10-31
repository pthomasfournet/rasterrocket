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
  │     │     └── skrifa, read-fonts  (crates/skrifa, crates/read_fonts: local forks)
  │     │           TrueType / OpenType / CFF / Type 1 parsing and outlines
  │     │
  │     ├── rasterrocket-encode
  │     │     PPM / PGM / PBM / PNG output, baseline grayscale JPEG
  │     │
  │     └── rasterrocket-interp
  │           PDF content stream interpreter + operator dispatcher
  │           │
  │           ├── gpu  (optional, feature-gated)
  │           │     CUDA kernels, nvJPEG/nvJPEG2k decoders, NPP rotate
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
  │                       `rrocket` binary (pdftoppm-style flags)
  │
  ├── rasterrocket-comic  (comic-archive input: .cbz/.cb7/.cbt → RenderedPage;
  │                        depends on rasterrocket + render + color; used by the CLI)
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
rasterrocket_interp::parse_page()    content tokenizer → operator decode
    │
    ▼ Bitmap<Rgb8>  (blank, sized by PageGeometry::size_px)
rasterrocket_interp::PageRenderer::with_view(&PageView::whole(..))
    │                                 allocates the target bitmap; initial CTM
    ▼
PageRenderer::execute(ops)           dispatch loop:
    ├── path ops  ──────────────────▶ raster::fill / stroke / eo_fill
    ├── text ops  ──────────────────▶ font::FontFace → GlyphBitmap
    │                                 raster::fill_glyph
    ├── image ops ──────────────────▶ resources::resolve_image
    │   DCT      ───────────────────▶ gpu::NvJpegDecoder  (or zune-jpeg)
    │   JPX      ───────────────────▶ gpu::NvJpeg2kDecoder (or hayro-jpeg2000,
    │                                 at the image's drawn resolution)
    │   CCITT/JBIG2 ────────────────▶ hayro-ccitt / hayro-jbig2
    │   Flate    ───────────────────▶ libdeflate (default) / flate2 fallback
    │   CMYK     ───────────────────▶ gpu::icc_cmyk_to_rgb (or AVX-512/scalar)
    │                                 PageRenderer::blit_image
    ├── shading ops ────────────────▶ raster::shaded_fill / gouraud_triangle_fill
    ├── tiling ops  ────────────────▶ raster::TiledPattern
    └── transparency groups ────────▶ raster begin_group / paint_group
    │
    ▼ Bitmap<Rgb8>
rasterrocket::rgb_to_gray()          Y = 0.30 R + 0.59 G + 0.11 B (ISO 32000-1 §10.3.3)
    │
    ▼ Vec<u8>  (width × height, 8-bit grayscale)
RenderedPage { pixels, width, height, dpi, effective_dpi, diagnostics }
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
- `Pixel` trait — `Copy + Pod + Zeroable + Send + Sync + 'static`
- Concrete: `Rgb8`, `Gray8`, `Cmyk8`, `DeviceN8`, `AnyColor`
- `PixelMode` enum — `Mono1 | Mono8 | Rgb8 | Bgr8 | Xbgr8 | Cmyk8 | DeviceN8`
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

### 3.2 `rasterrocket-render`

~17 700 lines. The software rendering engine. PDF-agnostic — takes paths,
bitmaps, and compositing parameters; knows nothing about operators or resources.

**`Bitmap<P: Pixel>`**
The primary output container. Width, height, stride, `Vec<u8>` data, optional
separate alpha plane. Generic over `P` — the compiler generates one code path
per pixel format with no runtime mode dispatch in hot loops.

**Path pipeline**
```
PathBuilder  →  Path  →  flatten_path()  →  XPath (edge table)
                          make_stroke_path()      │
                                                  ▼
                                           XPathScanner  →  ScanIterator
                                                            (x0, x1, y) spans
```
Bézier flattening via De Casteljau. Stroke expansion produces a fill path from
the outline — strokes and fills share a single rasterizer.

**Fill rasterizer**
Active edge table scan, 4× supersampled AA (`AA_SIZE = 4`). Coverage per output
pixel = popcount of 4 nibbles across 4 AA sub-rows. Hot path:
`aa_coverage_span()` → SIMD popcount tier (see §5).

Three compositing pipeline variants, selected at paint time by `PipeState::no_transparency()` / `use_aa_path()`:

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
  is served as its mean colour and coverage.
- Shading patterns — axial, radial, function-based, Gouraud triangle mesh

**SIMD module** (`simd/`)
Per-arch dispatch helpers; each arch gets its own function, not a cfg/return chain:
```
blend_solid_rgb8 / blend_solid_gray8  (movdir64b → AVX2 → NEON → scalar)
composite_aa_rgb8_opaque              (AVX-512 → scalar)
aa_coverage_span                      (AVX-512 BITALG → AVX2 → SVE2 → NEON → scalar)
apply_aa_gamma                        (AVX-512 VBMI → SSSE3 → scalar)
unpack_mono_row                       (SSE2 → NEON → scalar)
```
SVE2 tier requires `nightly-sve2` Cargo feature (nightly Rust + `stdarch_aarch64_sve`).

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
- `GlyphKey` — `{ face_id, glyph_id, size_px, x_frac, aa }`
- `FontProgram::vertical_selector` — maps a glyph to its vertical form through
  the face's `vert` / `vrt2` GSUB features, for vertical writing with a
  substitute face
- `sfnt_info` — family, weight, slant, serif class and `OS/2` code-page ranges
  of each face in an sfnt file, read from the `OS/2` and `name` tables alone,
  and a face's Unicode `Coverage`, read from its `cmap` table alone; the
  interpreter indexes installed fonts with it without loading them

### 3.4 `rasterrocket-encode`

Thin I/O layer. Takes `Bitmap<P>`, writes file format bytes to a `Write` sink.

- `write_ppm` — Netpbm P6 from `Rgb8`, `Bgr8`, `Xbgr8`, `Cmyk8` or `DeviceN8`;
  CMYK and DeviceN process channels convert through the process-ink model
- `write_pgm` — Netpbm P5 grayscale
- `write_pbm` — Netpbm P4 1-bit
- `write_png` — `png` crate; `Gray8` as grayscale, the colour modes of
  `write_ppm` as RGB (RGBA when the bitmap has an alpha plane)
- `jpeg_gray` — baseline 8-bit grayscale JPEG of a `Gray8` bitmap

No pixel processing beyond colour-mode conversion at the boundary.

### 3.4a `rasterrocket-parser` (`crates/pdf`)

The file-structure layer: a lazy, memory-mapped parser. `Document::open`
reads only the cross-reference; objects are parsed on first access and
cached as `Arc<Object>`.

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

optional_content.rs  — OptionalContent: the /OCProperties /D configuration
                       resolved once per session for a RenderPurpose; OCG and
                       OCMD (/P, /VE) visibility for BDC /OC, XObject /OC and
                       annotation /OC
page_group.rs        — the page group (§11.4.7): whether a page is painted as
                       an isolated group over white (a knockout page group, or
                       a blend mode reachable from its resources or annotations)
prescan.rs           — classify page content without rendering pixels
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
                       the code → glyph table of a loaded face
  encodings.rs       — base encodings of Annex D
  fallback.rs        — per-character fallback of a substitute to installed faces
  substitutes.rs     — substitute programs for non-embedded fonts
  system_fonts.rs    — installed CJK / Arabic / Thai faces; per-character fallback faces
  bundled_fonts.rs   — Noto fallback faces compiled in (`bundled-fonts`)
  page/
    mod.rs           — PageRenderer: construction, execute(), finish()
    view.rs          — PageView: device-space placement and initial CTM
                       (rotation, UserUnit, window)
    operators.rs     — execute_one: one operator → renderer call
    paint.rs         — path painting: colour, clip, fill and stroke
    fill_stroke.rs   — B / B* / b / b* as one object (stroke knocks out fill)
    xobject.rs       — Do: image and form XObjects, form groups
    image_blit.rs    — a decoded image or stencil mask onto the page through
                       the clip, on the CPU or the device blit kernel
    diagnostics.rs   — PageDiagnostics, ImageFailure, RenderAbort
    devices.rs       — GPU decoders, kernels and image cache of a renderer
    text.rs          — show_text
    type3.rs         — show_text_type3; coverage cache of d1 glyphs
    text_ops.rs      — glyph records, advances, text-space → device
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
                       line wrapping and layout
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
    gpu_ops.rs       — CUDA dispatch thresholds + optional gpu:: calls
    vk_ops.rs        — Vulkan dispatch (AA fill)

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
    icc/clut.rs      — CMYK → sRGB lookup table baked from a moxcms
                       transform for the GPU path, kept with the profile's
                       per-document conversion
    lab.rs           — Lab through Range and WhitePoint, Bradford-adapted
                       to D65
  icc_shaper.rs      — matrix/TRC ICC input profiles evaluated to sRGB
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
    codecs/          — dct.rs, jpx.rs, ccitt.rs, jbig2.rs; JPX decodes at
                       `ImageRequest::draw_size` unless masked; CMYK/YCCK
                       JPEG samples are ink
    colorspace.rs    — ImageColour: the samples' colour space and their
                       conversion to gray / RGB
    smask.rs         — SMask decode and /Matte removal
    inline.rs        — BI … ID … EI; data length from the dictionary, a
                       named colour space resolved through the resources
```

**Image colour** (`resources/image/colorspace.rs`)

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

**Image decode budget** (`resources/image/mod.rs`)

An image's sides may each reach `MAX_IMAGE_SIDE` (131 071), and its
decoded samples `width × height × components` may reach
`MAX_DECODED_BYTES` (1 GiB); a larger image is refused before anything is
decoded or allocated and reported as a decode failure, so a hostile
dictionary cannot drive the process to an allocation abort.  Within the
budget every sample buffer the decoders and the colour conversion create
is reserved fallibly: an allocation the machine cannot satisfy fails the
image, not the process.

**Fonts** (`renderer/font_cache.rs` and its neighbours)

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
  is off, the standard font directories, on first use, and indexes every
  sfnt face by family and `OS/2` code-page range. Libraries are shared
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
  Cyrillic, Hebrew, Arabic, Devanagari and Thai (sans-serif and serif) and
  three symbol faces are compiled in (`crates/pdf_interp/fonts/noto/`,
  zlib-compressed, ~0.33 MB; `renderer/bundled_fonts.rs`). They are the
  last resort of the per-character fallback, after every installed face.
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

**GPU integration points** (feature-gated, in `renderer/page/gpu_ops.rs` for
CUDA and `renderer/page/vk_ops.rs` for Vulkan)
- DCTDecode: call `gpu::NvJpegDecoder::decode()` if pixels ≥ `GPU_JPEG_THRESHOLD_PX`
- JPXDecode: call `gpu::NvJpeg2kDecoder::decode()` similarly
- CMYK→RGB: call `gpu::icc_cmyk_to_rgb()` if pixels ≥ `GPU_ICC_CLUT_THRESHOLD`
- AA fill: `gpu::GpuCtx::aa_fill()` if the bbox area ≥ `GPU_AA_FILL_THRESHOLD`
  (dormant: `usize::MAX`; override with `PDF_RASTER_AA_FILL_THRESHOLD`)
  (solid-colour fills under a rectangular clip only).  The device evaluates
  the CPU scanner's intersection table with the CPU's span walk and the
  counts go through the CPU compositing pipeline, so the output is
  byte-identical to the CPU fill.

GPU decoders live in thread-local storage on rayon worker threads (one per thread).
They are torn down eagerly via `rasterrocket::release_gpu_decoders()` broadcast before
the rayon pool drops — this avoids the CUDA driver teardown race at process exit.

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
  per rendered page (`pdf_interp::page_has_javascript`), warned once per
  session. `InterpError` (`#[non_exhaustive]`) has no
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

The trait surface is `begin_page → record_* → submit_page → wait_page`, with
six `record_*` methods (one per kernel) plus `alloc_device` / `alloc_host_pinned`
/ `upload_async`.  GATs (`type DeviceBuffer`) make it usable only as a generic
parameter, not `dyn` — callers monomorphise per backend.  The renderer
holds `Option<Arc<GpuCtx>>` for CUDA and `Option<Arc<VulkanBackend>>` for
Vulkan as parallel fields rather than going through the trait at every call
site, so the device image cache, whose buffers are `CudaSlice<u8>`, stays
CUDA-typed.

**`GpuCtx`** (CUDA) — one per process, holds the CUDA context and all compiled
PTX kernels as `cudarc` modules.  Shared via `Arc<GpuCtx>`.  `cudarc` is pinned
to the `cuda-13030` driver-API binding, so a CUDA 13.x driver is required.

**`VulkanBackend`** — one per process; loads the Vulkan instance, picks a
discrete device (ranks discrete > integrated > virtual > CPU), creates a single
compute queue, and lazy-loads SPIR-V → `VkPipeline` per kernel.  Persistent
`VkPipelineCache` blob at `$XDG_CACHE_HOME/rasterrocket/vulkan_pipeline_cache.bin`
across runs.  Shared via `Arc<VulkanBackend>`.  The renderer dispatches AA fill
through this; ICC CMYK→RGB and the `cache` feature stay CUDA-only,
so `--backend vulkan` runs uncached and the CMYK matrix path falls to CPU AVX-512.

**Kernel inventory.**  Eight kernels exist in **both** `.cu` (CUDA, compiled to
PTX by `nvcc`) and `.slang` (Slang, compiled to SPIR-V by `slangc`) at build
time, gated on the `vulkan` feature.  The table below lists the five render-path
kernels; the remaining three — `blelloch_scan`, `idct_color`, and
`parallel_huffman` — belong to the GPU JPEG pipeline and are dormant by default
(`GPU_JPEG_HUFFMAN_THRESHOLD_PX = u32::MAX`).

| Kernel | CUDA file | Slang file | Operation | Threshold |
|---|---|---|---|---|
| `composite_rgba8` | composite_rgba8.cu | composite_rgba8.slang | Porter-Duff source-over | no production caller † |
| `apply_soft_mask` | apply_soft_mask.cu | apply_soft_mask.slang | Per-pixel alpha multiply | no production caller † |
| `aa_fill` | aa_fill.cu | aa_fill.slang | 4× scanline-AA sub-sample counts from the CPU scanner's intersection table (byte-identical to the CPU fill) | dormant (`usize::MAX`) |
| `icc_clut` | icc_clut.cu | icc_clut.slang | CMYK→RGB via 4D quadrilinear CLUT | 500K px |
| `icc_clut` matrix | icc_clut.cu | icc_clut.slang | CMYK→RGB via matrix (always CPU) | — |
| `blit_image` | blit_image.cu | blit_image.slang | Cached-image source-over composite (`cache` feature) | always |

† The GPU `composite_rgba8` / `apply_soft_mask` kernels are built and reachable
through the backend `record_*` surface, but the renderer composites on the CPU
(`composite_rgba8_cpu` in `crates/gpu/src/composite.rs`). They carry no dispatch
threshold because nothing in the render path dispatches them.

16 kernel-level parity tests in `crates/gpu/tests/cu_vs_slang_parity.rs` confirm
SPIR-V vs CUDA outputs within ≤ 1 LSB per channel on the dev box.

**Build-script model.**  `crates/gpu/build.rs` probes `nvcc --version` directly:
when nvcc works, real PTX is compiled regardless of features.  When nvcc fails
(no CUDA toolkit on a CI runner), 0-byte placeholder PTX files are written and
`cargo:rustc-cfg=ptx_placeholder` is emitted; `GpuCtx::init` short-circuits
under that cfg with a clear error pointing at the build host.  Slang→SPIR-V
compile is gated on `CARGO_FEATURE_VULKAN`.

**Optional decoders** (feature-gated)
- `nvjpeg` — `NvJpegDecoder`, TLS one-per-thread, primary `GPU_HYBRID`, fallback `DEFAULT`
- `nvjpeg2k` — `NvJpeg2kDecoder`, same TLS pattern; C++ exception shim in `shim/nvjpeg2k_shim.cpp`
- `gpu-deskew` — `npp_rotate()` via `nppiRotate_8u_C1R_Ctx`
- `vaapi` — `VapiJpegDecoder`; VA-API JPEG baseline decode on Linux iGPU/dGPU (AMD VCN, Intel Quick Sync, Intel Arc); links `libva.so.2` + `libva-drm.so.2`. Dispatch priority: nvJPEG → VA-API → zune-jpeg (CPU). CMYK and progressive JPEG fall through to CPU.

**Device image cache** (`cache` feature, CUDA-only)
- `DeviceImageCache` — three tiers: VRAM (refcount-pinned LRU), pinned host RAM (`cuMemAllocHost` slabs, demote-on-evict / promote-on-hit), disk (`<root>/<doc-blake3>/<content-hash>.bin` sidecar files; opt-in via `PDF_RASTER_CACHE_DIR`).
- `DevicePageBuffer` — zero-init RGBA8 per page; lazy-allocated on first GPU image. Each blit samples the cached image through the same Q32 column/row tables the CPU sampler uses (`renderer/page/image_sampler.rs`), then the touched rows are downloaded, source-over composited onto the host bitmap in content-stream order, and re-zeroed — so cache-on and cache-off renders are byte-identical. A draw that reduces the image (`image_reduce::draw_reduces`) first copies the cached pixels back to the host (`DeviceImageCache::download`) so they are area-averaged exactly as host-resident pixels are; the kernel samples only images drawn at or near full resolution.
- BLAKE3 content hashing keys cross-document dedup; `(DocId, ObjId)` alias keys same-document fast paths.

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
                          // vector_antialias, decrypt_authorized, password,
                          // font_dirs, system_fonts, cancel

raster_pdf(path, opts) → impl Iterator<Item = (u32, Result<RenderedPage, RasterError>)>
raster_pdf_from_bytes(bytes, opts) → impl Iterator<Item = (u32, Result<RenderedPage, RasterError>)>
render_channel(path, opts, capacity) → Receiver<(u32, Result<RenderedPage, RasterError>)>
open_session(path, config) → Result<RasterSession, RasterError>
open_session_from_bytes(bytes, config) → Result<RasterSession, RasterError>
render_page_gray(session, page_num, &RasterOptions) → Result<RenderedPage, RasterError>
render_page_rgb(session, page_num, scale) → Result<Bitmap<Rgb8>, RasterError>
render_page_rgb_with(session, page_num, &RenderParams) → Result<Bitmap<Rgb8>, RasterError>
    // RenderParams { scale_x, scale_y, region: Option<RenderRegion>, policy, cancel: Option<CancelToken> }
session.page_geometry(page_num) → Result<PageGeometry, RasterError>   // size, rotation, boxes; size_px(sx, sy)
prescan_session(session, page_num) → Result<PageDiagnostics, RasterError>
encode_for_gcv(page, budget) → Result<GcvImage, GcvError>   // Cloud Vision input
release_gpu_decoders()   // call via pool.broadcast() before pool drops
```

The `*_from_bytes` variants render an in-memory PDF with no temp file, with
the same decryption behaviour as the path variants. Every function and type
is exported at the crate root; the modules that implement them are private,
except `deskew`.

`RenderedPage` carries: `pixels: Vec<u8>` (8-bit gray, width×height), `width`,
`height`, `dpi`, `effective_dpi` (= dpi × UserUnit), and `PageDiagnostics`.

`PageDiagnostics::suggested_dpi(min, max)` — hint for re-rendering at native image
resolution. Returns `None` for vector/text-only pages.

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
HTJ2K codestreams (ISO/IEC 15444-15, signalled by the CAP marker) are refused
with a diagnostic that names them: the high-throughput block coder exists
neither upstream nor here, so such an image is skipped and reported. A
codestream whose header asks for more than 2^30 decoded samples over all
components, or more than 65 535 tiles, is refused before its buffers are
allocated (`ValidationError::ImageTooLarge`).

### 3.8a `read-fonts` and `skrifa` (`crates/read_fonts`, `crates/skrifa`)

Local forks of the font-parsing crates, vendored like `crates/jpeg2000`: not
published, not released by `cargo release`, never sent upstream, base recorded
in each `UPSTREAM.toml`, every local change listed in each `CHANGES.md`.
`crates/font_test_data` is an unmodified copy of upstream's test fonts so both
upstream test suites run (`cargo test -p read-fonts -p skrifa`). The local
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

### 3.9 `rasterrocket-cli`

Binary entry point. Thin glue over `rasterrocket` + `rasterrocket-encode`.

```
clap args → SessionConfig → open_session → rayon ThreadPool
  → pool.install(|| per page: prescan_session, render_page_rgb_with,
                    encode::write_ppm / write_pgm / write_pbm / write_png)
  → pool.broadcast(release_gpu_decoders)    // teardown before pool drops
  → main returns ExitCode                   // RAM-output guard and pool drop
```

All rendering is inside `pool.install()`. The broadcast teardown is the last thing
that happens before the pool is dropped, ensuring GPU decoders are released while
the CUDA driver is fully live. Every exit path returns from `main` rather than
calling `process::exit`, so the RAM output directory is removed after a failed
page too.

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

---

## 4. Feature flag graph

```
rasterrocket-cli    → rasterrocket       → rasterrocket-interp → rasterrocket-gpu
──────────────────────────────────────────────────────────────────────────────────
nvjpeg              → nvjpeg             → nvjpeg              → nvjpeg
nvjpeg2k            → nvjpeg2k           → nvjpeg2k            → nvjpeg2k
vaapi               → vaapi              → vaapi               → vaapi
gpu-aa              → gpu-aa             → gpu-aa              → (crate only)
gpu-icc             → gpu-icc            → gpu-icc             → (crate only)
gpu-deskew          → gpu-deskew         →                     → gpu-deskew
cache               → cache              → cache               → cache
vulkan              → vulkan             → vulkan (+ gpu-aa)   → vulkan
gpu-jpeg-huffman    → gpu-jpeg-huffman   → gpu-jpeg-huffman    → gpu-jpeg-huffman
gpu-validation      → gpu-validation     →                     → gpu-validation
bundled-fonts *     → bundled-fonts *    → bundled-fonts       →

* on by default

rasterrocket-render features (orthogonal, no cross-crate effect):
  simd-avx2        default on; enables blend/fill AVX2 paths
  nightly-sve2     aarch64 only; SVE2 coverage tier (requires nightly Rust)

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
pub(crate) fn aa_coverage_span(rows: [&[u8]; 4], x0: usize, shape: &mut [u8]) {
    if shape.is_empty() { return; }
    dispatch_coverage(rows, x0, shape);
}

#[cfg(target_arch = "x86_64")]
fn dispatch_coverage(rows: [&[u8]; 4], x0: usize, shape: &mut [u8]) {
    /* avx512bitalg (≥ 128 px) → avx2 (≥ 64 px) → scalar */
}

#[cfg(target_arch = "aarch64")]
fn dispatch_coverage(rows: [&[u8]; 4], x0: usize, shape: &mut [u8]) {
    // SVE2 tier above NEON when the `nightly-sve2` feature is active
    #[cfg(feature = "nightly-sve2")]
    if is_aarch64_feature_detected!("sve2") {
        unsafe { aa_coverage_span_sve2(rows, x0, shape) };
        return;
    }
    unsafe { aa_coverage_span_neon(rows, x0, shape) };
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn dispatch_coverage(rows: [&[u8]; 4], x0: usize, shape: &mut [u8]) {
    aa_coverage_span_scalar(rows, x0, shape);
}
```

### Coverage by operation and platform

| Operation | File | Ryzen (AVX-512) | Intel consumer (AVX2) | ARM NEON | ARM SVE2 |
|---|---|---|---|---|---|
| AA coverage span | `raster/src/simd/aa_coverage.rs` | BITALG `_mm512_popcnt_epi8` | AVX2 VPSHUFB | `vcntq_u8` + `vst2q_u8` | `svcnt_u8_z` † |
| AA gamma map | `raster/src/simd/aa_gamma.rs` | AVX-512 VBMI | SSSE3 | scalar | scalar |
| Solid fill (RGB) | `raster/src/simd/blend.rs` | `movdir64b` → AVX2 | AVX2 | `vst3q_u8` | (NEON sufficient) |
| Solid fill (Gray) | `raster/src/simd/blend.rs` | `movdir64b` → AVX2 | AVX2 | `vst1q_u8` | (NEON sufficient) |
| AA composite | `raster/src/simd/composite.rs` | AVX-512 | scalar | scalar | scalar |
| Glyph unpack | `raster/src/simd/glyph_unpack.rs` | SSE2 | SSE2 | `vtstq_u8` | (NEON sufficient) |
| CMYK→RGB | `color/src/cmyk/simd.rs` | AVX-512 VBMI | AVX2 | `vld4q_u8` + `vst3q_u8` | (NEON sufficient) |

† SVE2 tier requires `nightly-sve2` Cargo feature.

### Key platform facts

**x86-64 (Intel consumer)**
- AVX2 is the ceiling on all Alder/Raptor/Arrow Lake. AVX-512 is architecturally
  disabled on Arrow Lake; disabled by microcode on Raptor. No current Intel consumer
  CPU has usable AVX-512 — Xeon only.
- `movdir64b` is present on i9-14900K and Core Ultra 9 285K; absent on i7-8700K.
- AA coverage has an AVX2 tier (VPSHUFB nibble lookup,
  `aa_coverage_span_avx2`), so every Intel consumer CPU has a vectorised
  path.

**ARM / aarch64**
- NEON is mandatory on all ARMv8-A. No runtime detection needed — `#[cfg(target_arch = "aarch64")]` is sufficient.
- `vcntq_u8` is hardware-native on all targets (M1–M4, Cortex-A72, Cortex-A76).
  Never use `__builtin_popcountll` on ARM — it compiles to a software sequence.
- `STNP` is an allocation hint only, not a cache-bypass guarantee. For zero-fill use
  `DC ZVA` (zeros full cacheline without read). For non-zero solid fill: `vst1q_u8`.
- Apple M4 has SVE2 (128-bit fixed width). An SVE2 coverage tier is implemented
  behind the `nightly-sve2` feature (`svcnt_u8_z`). At 128-bit
  width it matches NEON throughput; on wide-SVE2 server chips (Graviton4 full
  width) it gives up to 4× NEON throughput.

### iGPU / integrated compute

**Intel iGPU (UHD/Iris/Arc laptop) and AMD VCN**
VA-API JPEG baseline decode is implemented (`vaapi` feature). Hardware handles
YUV surface allocation and IDCT; the result is extracted to an `Rgb8` bitmap.
Transfer overhead makes other compute (ICC, fill) not worth it on non-UMA x86.
CMYK and progressive JPEGs fall through to CPU; the `nvjpeg` feature takes
priority when both are active.

**Apple Silicon (M1–M4)**
No native Metal backend exists; the Vulkan backend runs through `MoltenVK`
(untested). The CPU paths use NEON.

**Raspberry Pi 4/5**
VideoCore GPU is not accessible as a general-purpose compute API via a stable
userspace interface. NEON is the correct answer on Pi.

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
  │   pool.install() blocks until all pages done
  │
  ├── pool.broadcast(release_gpu_decoders)   ← explicit teardown while driver live
  │
  └── pool dropped
```

**GPU decoder lifecycle**
Each rayon worker thread lazily initialises a `NvJpegDecoder` on first use and
stores it in thread-local storage. At process exit all TLS destructors fire
concurrently, which races with the CUDA driver's own atexit shutdown. The fix:
call `release_gpu_decoders()` via `pool.broadcast()` after `pool.install()` returns.
This drops each decoder while the driver is still fully live; the TLS destructors
see `Uninitialised` and are no-ops.

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
- `InterpError::Pdf` — PDF parse failure (in-tree `pdf` crate)
- (JavaScript is **not** a per-document error — it is detected, warned about, and
  rendered; see §3.5)

**Per-page errors** — skip the page, continue rendering:
- `InterpError::PageOutOfRange`
- `InterpError::InvalidPageGeometry`
- `RasterError::PageDegenerate` / `RasterError::PageTooLarge` /
  `RasterError::PageAreaTooLarge`
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

**Silent degradation** — never exposed to the caller:
- Unrecognised PDF operators — discarded
- GPU unavailable / below threshold — CPU path used
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

**SIMD dispatch is per-arch, not per-feature.**
ARM NEON is unconditionally available on aarch64 — no runtime probe. x86 SIMD
uses `is_x86_feature_detected!` at runtime. Each arch gets a private `dispatch_*`
function; there is no single god-function with cfg-gated returns.
