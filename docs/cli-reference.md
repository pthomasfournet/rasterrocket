# CLI Reference

```
rrocket [OPTIONS] <INPUT> <OUTPUT_PREFIX>
```

Renders pages to image files. Drop-in replacement for `pdftoppm` in scripts.

**Arguments:**

- `<INPUT>` — path to the input file. A PDF, or a comic/scan archive whose
  extension is `.cbz` (ZIP), `.cb7` (7-Zip), or `.cbt` (TAR). The format is
  chosen by extension: comic archives take the comic path, everything else is
  treated as a PDF. `.cbr` (RAR) is unsupported and exits with a
  convert-to-`.cbz` message. Pass `-` to read a PDF from stdin (stdin is always
  treated as a PDF).
- `<OUTPUT_PREFIX>` — output filename prefix. The page number and extension are appended automatically (e.g. prefix `out` → `out-1.ppm`, `out-2.ppm`, …).

**Comic archives:** pages are the archive's image entries, ordered by a
numeric-aware natural sort, decoded from JPEG/PNG/WebP/TIFF. All output flags
below (page range, format, `--gray`/`--mono`, `--odd`/`--even`, `--single`)
behave identically to the PDF path. An archive that contains a single PDF
instead of loose images renders that PDF. `--progress`/`--timings` are PDF-only
and have no effect on comic input.

---

## Page range

| Flag | Short | Default | Description |
|---|---|---|---|
| `--first-page N` | `-f N` | `1` | First page to render (1-based). |
| `--last-page N` | `-l N` | last page | Last page to render (1-based). |
| `--odd` | `-o` | off | Render only odd-numbered pages. |
| `--even` | `-e` | off | Render only even-numbered pages. |
| `--singlefile` | | off | Stop after the first matching page. |

`--odd` and `--even` are mutually exclusive.

If `--first-page` < 1, it is clamped to 1 with a warning. If `--last-page` exceeds the document length, it is clamped to the last page with a warning.

---

## Resolution and scaling

| Flag | Default | Description |
|---|---|---|
| `-r DPI` / `--resolution DPI` | `150` | Render resolution for both axes. |
| `--rx DPI` | same as `-r` | Horizontal resolution only (overrides `-r` for X). |
| `--ry DPI` | same as `-r` | Vertical resolution only (overrides `-r` for Y). |
| `--scale-to N` | — | Scale the longest edge to N pixels (preserves aspect ratio). |
| `--scale-to-x N` | — | Scale width to N pixels. |
| `--scale-to-y N` | — | Scale height to N pixels. |

DPI values must be positive finite numbers; non-positive values are rejected at startup.

---

## Crop

| Flag | Short | Description |
|---|---|---|
| `--crop-x N` | `-x N` | Crop X offset in pixels. |
| `--crop-y N` | `-y N` | Crop Y offset in pixels. |
| `--crop-width N` | `-W N` | Crop width in pixels. |
| `--crop-height N` | `-H N` | Crop height in pixels. |

The page boundary is selected automatically: the CropBox clamped to the
MediaBox, per ISO 32000-2 §14.11.2. There is no flag to switch between them.

---

## Output format

| Flag | Extension | Description |
|---|---|---|
| _(default)_ | `.ppm` | RGB PPM (Netpbm P6). |
| `--gray` | `.pgm` | Grayscale PGM (Netpbm P5, BT.709). Can combine with `--png`. |
| `--mono` | `.pbm` | 1-bit mono PBM (Netpbm P4, 50% threshold). Can combine with `--png`. |
| `--png` | `.png` | PNG. Combine with `--gray` / `--mono` for grayscale/mono PNG. |
| `--jpeg` | `.jpg` | JPEG. |
| `--jpegcmyk` | `.jpg` | JPEG in CMYK colour space. |
| `--tiff` | `.tif` | TIFF. |
| `--jpegopt N` | — | JPEG quality 0–100 (default 75). |

**Extension matrix:**

| Format flag | `--mono` | `--gray` | Neither |
|---|---|---|---|
| _(default / PPM)_ | `.pbm` | `.pgm` | `.ppm` |
| `--png` | `.png` | `.png` | `.png` |
| `--jpeg` | `.jpg` | `.jpg` | `.jpg` |
| `--tiff` | `.tif` | `.tif` | `.tif` |

---

## Rendering

| Flag | Default | Description |
|---|---|---|
| `--aa yes\|no` | `yes` | Enable raster anti-aliasing. |
| `--aaVector yes\|no` | `yes` | Enable vector anti-aliasing. |
| `--hide-annotations` | off | Skip rendering PDF annotations. |
| `--overprint` | off | Enable overprint preview. |
| `--thinlinemode MODE` | `default` | Thin-line rendering: `default`, `solid`, or `shape`. |

Fonts a document does not embed are drawn with substitutes: the installed
Liberation (or Arial / Times New Roman / Courier New) family when found,
else faces compiled into the binary; CJK, Arabic and Thai text uses an
installed face of that script. `RROCKET_FONT_DIR` names extra directories
to search first, in `PATH` syntax (`RROCKET_FONT_DIR=/opt/fonts:$HOME/fonts`).

---

## Camera-captured pages

These flags run [`rasterrocket-camera`](api-reference.md#rasterrocket-camera-crate--camera-capture-page-cleanup)
on each rendered page. A stage whose detector is not confident leaves the page
unchanged, so flatbed scans and born-digital pages are written exactly as
without the flags.

| Flag | Default | Description |
|---|---|---|
| `--crop` | off | Find a page photographed on a darker surround and crop to its bounding box. Unrelated to `-x`/`-y`/`-W`/`-H`, which crop a fixed pixel region. |
| `--rectify` | off | Find that page, correct the perspective and crop to the page. Supersedes `--crop` when it applies. |
| `--split-spreads` | off | Split a double-page spread at its gutter. The halves of page N are written with an `a` (left) and `b` (right) suffix: `out-07a.png`, `out-07b.png`. Pages not detected as spreads keep their usual name. |
| `--dewarp` | off | Straighten text lines curved by a page bending toward the spine. Experimental: vertical displacement only. Pages whose lines are already straight, or with fewer than five text lines, are written unchanged. |

The stages run crop or rectify → split → dewarp, on PDF and comic-archive input
alike. Detection runs on the page's luma. `--gray` and `--mono` output is
written from the processed luma; colour output applies the same geometry to
each colour channel, so colour is kept.

```bash
# Phone captures of an open book, held square to the camera: crop and split
rrocket --crop --split-spreads --gray --png -r 300 book.pdf out

# Captures at an angle: rectify, split and straighten each page
rrocket --rectify --split-spreads --dewarp --gray --png -r 300 book.pdf out
```

---

## Passwords

| Flag | Description |
|---|---|
| `--opw PASSWORD` | Owner password for encrypted PDFs. |
| `--upw PASSWORD` | User password for encrypted PDFs. |
| `--decrypt-owned` | Authorise decrypting an encrypted (Standard Security Handler) document without the interactive confirmation. Decryption runs `qpdf`, which must be installed. Use it only for documents you own or are entitled to access. `RROCKET_DECRYPT_OWNED=1` does the same. |

Without `--decrypt-owned` or `RROCKET_DECRYPT_OWNED=1`, an encrypted document
prompts for a private-copy / liability confirmation (default: no). A
non-interactive run with neither set exits with an error instead of removing
the protection silently.

---

## Parallelism and output

| Flag | Default | Description |
|---|---|---|
| `--threads N` | `0` (auto) | Number of render threads. `0` = one thread per logical CPU. |
| `--sep CHAR` | `-` | Separator between prefix and page number (`out-1.ppm`, `out_1.ppm`, etc.). |
| `--forcenum N` | — | Zero-pad page numbers to at least N digits (`001`, `002`, …). |
| `-P` / `--progress` | off | Print per-page progress (page done, elapsed, ETA) to stderr. |
| `--timings` | off | Print per-page wall time and thread index to stderr for profiling. |
| `--strict` | off | Fail a page when one of its images or content streams cannot be decoded or drawn. Without it the image or stream is skipped, the page is written, and a warning naming the page and object goes to stderr. |

---

## Output destination (RAM-backed by default)

| Flag | Default | Description |
|---|---|---|
| `--ram` | (heuristic) | Force RAM-backed output even when `OUTPUT_PREFIX` looks like a path. Mutually exclusive with `--no-ram`. |
| `--no-ram` | (heuristic) | Force on-disk output. Disables the default tmpfs redirect for bare stems and writes pages exactly at `OUTPUT_PREFIX`. Mutually exclusive with `--ram`. |
| `--ram-path PATH` | `/dev/shm/rasterrocket-<pid>-<nanos>/` | Override the tmpfs directory used by RAM-backed output. Implies `--ram`. |

By default `rrocket` redirects rendered pages to a freshly-created tmpfs directory under `/dev/shm` whenever the output prefix is a bare stem, because writing to disk is 10–20× slower than RAM and dominates wall time on JPEG-heavy workloads.

**Bare stem vs path heuristic:** prefixes containing `/` or starting with `.` (e.g. `./out`, `/tmp/p`, `out/dir/p`) are treated as path-like and write to disk literally. Bare stems (e.g. `out`, `p`, `cover`) are redirected to tmpfs. `--ram` overrides the heuristic to redirect anyway; `--no-ram` overrides it to write to disk anyway.

A built-in spill policy polls `/proc/meminfo` every 100 ms and falls subsequent pages through to disk when MemAvailable drops below 1 GiB, with a one-shot warning printed to stderr.

---

## Backend selection

| Flag | Default | Description |
|---|---|---|
| `--backend auto\|cpu\|cuda\|vaapi\|vulkan` | `auto` | Compute backend for image decoding and GPU fills. |
| `--vaapi-device PATH\|auto` | `auto` | VA-API DRM render node; `auto` probes for one (only used with `auto` or `vaapi`). |
| `--vulkan-device INDEX\|NAME` | automatic | Vulkan device, by index or name substring (only used with `auto` or `vulkan`). |

**`auto`** — GPU when available, silent CPU fallback (same as pre-v0.4.0).

**`cpu`** — CPU only. All GPU init is skipped entirely. Use this for benchmarking, debugging, or on machines without a GPU.

**`cuda`** — Require CUDA. Exits with an error if nvJPEG or the GPU context cannot be initialised. Use this to confirm the GPU path is actually active rather than silently falling back.

**`vulkan`** — Require the Vulkan compute backend.  Exits with an error if Vulkan initialisation fails or the binary was built without `--features vulkan`.  AA-fill and parallel-Huffman JPEG decode kernels dispatch through `VulkanBackend` (both dormant by default; see `PDF_RASTER_AA_FILL_THRESHOLD` / `PDF_RASTER_HUFFMAN_THRESHOLD`); the device-resident image cache is CUDA-only so the renderer runs uncached under this mode; ICC CMYK→RGB stays on the CPU AVX-512 fallback.  Cross-vendor (NVIDIA, AMD, Intel, Apple via `MoltenVK`).

**`vaapi`** — Require VA-API JPEG decoding. Exits with an error if no usable render node is found or the given one cannot be opened. Use this to confirm iGPU/dGPU decoding is active.

**`--vaapi-device`** defaults to `auto`: rrocket walks `/dev/dri/renderD*` in ascending number and uses the first node whose driver initialises and offers VA-API baseline JPEG decode, so on an iGPU + dGPU host the node numbering does not matter. If none qualifies, `--backend vaapi` fails with a list of every node and why it was rejected, and `--backend auto` decodes JPEG on the CPU after one warning. If a path is given and cannot be used, the error lists the render nodes that exist:

```text
rrocket: backend unavailable: VA-API JPEG decoder unavailable on /dev/dri/renderD129: VA-API error -1: vaInitialize: unknown libva error
render nodes present:
  /dev/dri/renderD128 (pci 0000:72:00.0, amdgpu)
  /dev/dri/renderD129 (pci 0000:01:00.0, nvidia)
```

`RUST_LOG=debug` logs the chosen node once per process, with its PCI address, kernel driver and VA-API vendor string. `--vaapi-device` has no effect with `--backend cpu`, `--backend cuda`, or `--backend vulkan` — rrocket will reject the combination with a clear error rather than silently ignoring it, and those backends never open a render node.

**`--vulkan-device`** picks the Vulkan device on a host with more than one. The value is either an index in Vulkan enumeration order, counted from 0 (the `GPU0`, `GPU1`, … numbering of `vulkaninfo --summary`), or a case-insensitive substring of the device name (`5070`, `radv`, `llvmpipe`). A number that is not a valid index is matched as a name, so a model number works. An exact name match beats substring matches. A value that matches no device, or more than one, is an error that lists every device:

```text
rrocket: invalid raster options: Vulkan device selector "nosuch": no device name contains it
available Vulkan devices:
  0: NVIDIA GeForce RTX 5070 (DISCRETE_GPU, 12227 MiB device-local, pci 0000:01:00.0)
  1: AMD Ryzen 9 9900X3D 12-Core Processor (RADV RAPHAEL_MENDOCINO) (INTEGRATED_GPU, 10498 MiB device-local, pci 0000:72:00.0)
  2: llvmpipe (LLVM 21.1.8, 256 bits) (CPU, 30471 MiB device-local)
```

That error is fatal under `--backend auto` as well, because a selector that matches nothing is a typo, not missing hardware. When `--vulkan-device` is omitted, the `PDF_RASTER_VULKAN_DEVICE` environment variable is read (same syntax). Without either, rrocket prefers a discrete GPU, then integrated, virtual, and CPU devices. Among devices of the same type it takes the one with more device-local memory, then the lower PCI address, then the lower device UUID, so the choice does not depend on the order the driver lists devices in. No vendor is preferred: on a host with an NVIDIA and a non-NVIDIA discrete GPU, pass `--vulkan-device` (or use `--backend cuda`) if you need the CUDA-only features such as the image cache. `RUST_LOG=debug` logs the chosen device and how it was chosen. `--vulkan-device` is rejected with `--backend cpu`, `cuda` or `vaapi`.

```bash
# Confirm CUDA is active (fails loudly if no NVIDIA GPU)
rrocket --backend cuda -r 150 document.pdf out

# Confirm Vulkan is active (fails loudly if no Vulkan 1.3+ device)
rrocket --backend vulkan -r 150 document.pdf out

# Force CPU-only (useful for benchmarking without GPU)
rrocket --backend cpu -r 150 document.pdf out

# Run the Vulkan backend on the integrated GPU instead of the discrete one
rrocket --backend vulkan --vulkan-device radv document.pdf out

# Use VA-API on a specific render node instead of the probed one
rrocket --backend vaapi --vaapi-device /dev/dri/renderD129 document.pdf out
```

---

## Exit codes

| Code | Meaning |
|---|---|
| `0` | All pages rendered. A page with a skipped image or content stream counts as rendered unless `--strict` is set; the warning is on stderr. |
| `1` | One or more pages failed (errors printed to stderr), or invalid arguments. |

---

## Examples

```bash
# Render all pages at 150 DPI (default), output out-1.ppm, out-2.ppm, …
rrocket document.pdf out

# Render pages 3–7 at 300 DPI, grayscale PNG
rrocket -f 3 -l 7 -r 300 --gray --png document.pdf out

# Render all pages at 300 DPI, 4 threads, with progress
rrocket -r 300 --threads 4 -P document.pdf out

# Render only odd pages
rrocket --odd document.pdf out

# Render first page only (--singlefile stops after first match)
rrocket --singlefile document.pdf cover

# Render at 150 DPI, zero-pad page numbers to 3 digits: out-001.ppm
rrocket -r 150 --forcenum 3 document.pdf out

# Render with underscore separator: out_1.ppm
rrocket --sep _ document.pdf out

# Compare output against pdftoppm (pixel-diff test)
tests/compare/compare.sh -r 150 -f 1 -l 5 document.pdf
```

---

## Pixel-diff comparison

`tests/compare/compare.sh` compares rasterrocket output against pdftoppm page-by-page using ImageMagick RMSE.

**Requirements:** `pdftoppm`, `rrocket` (release build in `$PATH` or `target/release/`), ImageMagick (`compare`), `bc`.

```bash
tests/compare/compare.sh [OPTIONS] <PDF>

  -r DPI          Render resolution (default: 150)
  -f PAGE         First page (default: 1)
  -l PAGE         Last page (default: all)
  -t THRESHOLD    Max RMSE per page, 0–255 scale (default: 2.0)
  -v              Verbose: print all pages, not just failures
  -o DIR          Write diff images to DIR
  -n              Dry run: print commands without executing
```

Exit code 0 if all pages are within threshold; 1 if any fail.

```bash
# Example output
── Summary ──────────────────────────────────────────────────────────────
  PDF:       document.pdf
  DPI:       150
  Pages:     5 total, 5 passed, 0 failed
  Avg RMSE:  0.8432 / 255
  Max RMSE:  1.2100 / 255
  Threshold: 2.0 / 255
─────────────────────────────────────────────────────────────────────────
PASSED
```
