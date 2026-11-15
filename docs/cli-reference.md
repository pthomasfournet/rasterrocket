# CLI Reference

```
rrocket [OPTIONS] <INPUT> <OUTPUT_PREFIX>
rrocket --print [OPTIONS] <INPUT>
rrocket --list-printers
```

Renders pages to image files. Flag names and output naming follow `pdftoppm`
wherever rrocket has the same feature, so most `pdftoppm` command lines run
unchanged; the differences are listed under each section (for one, the
CropBox, not the MediaBox, is rendered by default).

`-h` / `--help` lists the flags and `-V` / `--version` prints the version.

**Arguments:**

- `<INPUT>` — path to the input file. A PDF, or a comic/scan archive whose
  extension is `.cbz` (ZIP), `.cb7` (7-Zip), `.cbt` (TAR) or `.cbr` (RAR, from
  RAR 1.3 to RAR 7). The format is chosen by extension: comic archives take the
  comic path, everything else is treated as a PDF. `-` reads a PDF from standard input
  (`rrocket - out < doc.pdf`); an encrypted one is decrypted only under
  `--decrypt-owned`, since no terminal is left to prompt on. A path that
  begins with `-` must follow a literal `--` (or be written `./-name.pdf`) so
  it is not read as an option.
- `<OUTPUT_PREFIX>` — output filename prefix. The page number and extension are appended automatically, the number zero-padded to the width of the document's page count (prefix `out` → `out-1.ppm` … `out-9.ppm` for a 9-page document, `out-01.ppm` … `out-12.ppm` for a 12-page one). A bare prefix such as `out` writes to a temporary directory in RAM, not the current directory; see [Output destination](#output-destination-ram-backed-by-default).

**Comic archives:** pages are the archive's image entries, ordered by a
numeric-aware natural sort, decoded from JPEG/PNG/WebP/TIFF/JPEG XL/AVIF (a multi-page TIFF
gives one page per image). All output flags
below (page range, format, `--gray`/`--mono`, `--odd`/`--even`, `--singlefile`,
the camera-capture flags) behave identically to the PDF path. Pages are decoded
to grayscale, so colour output carries gray pixels; `-r` only sets the
resolution recorded for the page, never its pixel size. An archive that
contains a single PDF instead of loose images renders that PDF, with the
rendering flags (`--aa`, `--aaVector`, `--overprint`, `--hide-annotations`,
`--no-system-fonts`), `--box`, `--rotate`, `--strict`, `--purpose` and the backend flags; an
invalid backend combination is refused for comic input as for a PDF. The
crop, scale-to and password flags, `--threads`, `--progress` and
`--timings` apply to PDF input only and have no effect on comic input. An archive's entries are decompressed into memory when it opens;
`--max-archive-mib MIB` (default 4096, at least 1) bounds their total, and an archive over it is refused
whole as a possible decompression bomb, exit status 1. A page image is held to `--max-page-mib`
as decoded, at 3 bytes a pixel.

---

## Page range

| Flag | Short | Default | Description |
|---|---|---|---|
| `--first-page N` | `-f N` | `1` | First page to render (1-based). |
| `--last-page N` | `-l N` | last page | Last page to render (1-based). |
| `--odd` | `-o` | off | Render only odd-numbered pages. |
| `--even` | `-e` | off | Render only even-numbered pages. |
| `--singlefile` | | off | Write only the first selected page, as `<prefix>.<ext>` with no page number (`--split-spreads` halves: `<prefix>-a`, `<prefix>-b`). |

`--odd` and `--even` are mutually exclusive.

If `--first-page` < 1, it is clamped to 1 with a warning. If `--last-page` exceeds the document length, it is clamped to the last page with a warning.

---

## Resolution and scaling

| Flag | Default | Description |
|---|---|---|
| `-r DPI` / `--resolution DPI` | `150` | Render resolution for both axes. |
| `--rx DPI` | same as `-r` | Horizontal resolution only (overrides `-r` for X). |
| `--ry DPI` | same as `-r` | Vertical resolution only (overrides `-r` for Y). Unequal `--rx`/`--ry` give non-square pixels: the page is `width × rx / 72` by `height × ry / 72` pixels. |
| `--scale-to PIXELS` | — | Fit each page in a PIXELS × PIXELS box: the longer edge is PIXELS pixels, aspect ratio kept. Overrides the resolution flags and `--scale-to-x`/`--scale-to-y`. |
| `--scale-to-x PIXELS` | — | Scale each page to PIXELS pixels wide; without `--scale-to-y` the height keeps the aspect ratio. Overrides `-r`/`--rx`. |
| `--scale-to-y PIXELS` | — | Scale each page to PIXELS pixels high; without `--scale-to-x` the width keeps the aspect ratio. Overrides `-r`/`--ry`. Both together give exactly the two sizes (non-square pixels). |

DPI values must be positive finite numbers and scale-to sizes positive integers; other values are rejected at startup.

---

## Crop

| Flag | Short | Description |
|---|---|---|
| `--crop-x PIXELS` | `-x PIXELS` | Left edge of the written area, pixels from the page's left edge (default 0). |
| `--crop-y PIXELS` | `-y PIXELS` | Top edge of the written area, pixels from the page's top edge (default 0). |
| `--crop-width PIXELS` | `-W PIXELS` | Width of the written area (default: to the right edge; clipped to it). |
| `--crop-height PIXELS` | `-H PIXELS` | Height of the written area (default: to the bottom edge; clipped to it). |

Coordinates are pixels of the page as rendered at the chosen resolution or
scale. Only the selected area is rendered, so a crop of a page too large to
render whole at that resolution still succeeds. An offset outside a page
fails that page. These flags apply to PDF input; comic-archive pages are
written whole.

---

## Page box

| Flag | Default | Description |
|---|---|---|
| `--box BOX` | `crop` | The page boundary box rendered: `media`, `crop`, `bleed`, `trim` or `art` (ISO 32000-2 §14.11.2). |
| `--cropbox` | | `--box crop`, under the `pdftoppm` name. Refused together with `--box`. |

`crop` renders the CropBox intersected with the MediaBox (the MediaBox when
the page has no CropBox); `media` renders the whole MediaBox, which is what
`pdftoppm` renders without `-cropbox`. `bleed`, `trim` and `art` render the
page's own BleedBox, TrimBox or ArtBox intersected with the MediaBox; these
three are not inherited from the page tree, and a page without the box named
renders its CropBox. The pixel coordinates of the crop flags, the
`--scale-to` sizes, `--print` and `--fax` all measure the selected box. For a
comic archive holding a PDF, `--box` applies to that PDF.

---

## Rotation

| Flag | Default | Description |
|---|---|---|
| `--rotate DEG` | `0` | Turn every page DEG degrees clockwise on top of its own `/Rotate`: a multiple of 90, negative for counter-clockwise, reduced modulo 360 (`-90` is `270`). Other values are rejected at startup. |

The page is drawn turned, not turned as a bitmap afterwards, so a quarter
turn exchanges its width and height, and the crop flags, the `--scale-to`
sizes, the camera-capture flags, `--print` and `--fax` all work on the
turned page. Comic-archive page images are turned pixel for pixel, and a
comic archive holding a PDF turns that PDF's pages.

---

## Size limits

**Page raster.** One page (or the area `-x`/`-y`/`-W`/`-H` select) is at
most `--max-page-mib` of canvas: 3 bytes a pixel for 8-bit RGB, 4 under
`--transparent`, 8 on the ink canvas of a `--cmyk` page or of a page
`--overprint` simulates, and twice as many under `--depth 16`:
by default 1 716 MiB (1.8 GB, a 600 MP page; A0 at 600 dpi fits). No page is
more than 16 777 216 pixels on a side, whatever the budget. Both are checked
from the page geometry before any pixel buffer is allocated. A page over
either is refused: rrocket prints `rrocket: page N: render error: a W×H page
is B bytes, over the page budget of L bytes; …` (or `page pixel dimensions
W×H exceed the per-side ceiling of 16777216 px; …`) followed by a hint naming
`--max-page-mib`, `-r` and the tile flags, the other pages still render, and
the run exits with status 1. Raise `--max-page-mib` for a genuinely large page
(a poster, a roll banner).

**Bands.** A page larger than `--band-mib` (default 64 MiB) renders in
horizontal bands of whole rows, each written to the output as it completes,
so memory holds a band per render thread rather than the page: a 44 × 120 in
roll at 360 dpi (15 840 × 43 200 px, 2 GB of RGB) prints or writes in well
under 1 GB. PPM, PNG, TIFF and `--print` output stream this way, and the file
is byte for byte the one a single band writes. JPEG output, `--fax`,
`--invert=auto` and the camera stages need the whole page and render it in
one band, under the page budget. A TIFF whose data would pass 4 GiB is written
as BigTIFF. Every band executes the page's content, so a page of many bands
takes longer to interpret than one of few.

**Images.** An image may declare at most 131 071 pixels on a side and decode
to at most 1 GiB of samples (width × height × components). An image over
either is refused before it is decompressed: it is skipped, the page renders
without it, and a warning naming the page and image goes to stderr; the
library reports it in `PageDiagnostics::image_failures`. With `--strict` the
page fails instead. A JPEG 2000 codestream is further limited to 2³⁰ decoded
samples over all components and 65 535 tiles.

**Other bounds.** The interpreter bounds the work and memory one page can
demand. None of these bounds fails the page; the rest of the page renders:

| Bound | Limit | Beyond it |
|---|---|---|
| Graphics-state saves (`q`) | 16 384 nested | deeper `q`/`Q` pairs share one state (one warning) |
| Array and dictionary nesting in a content stream | 256 levels | a deeper container is read as empty and its contents as elements of the enclosing one |
| Triangles decoded from one mesh shading | 2²² | the rest of the mesh is not drawn (one warning) |
| Dashes in one stroke | 2²⁰ | the stroke is drawn solid |
| `CMap` stream | 16 MiB | the stream is ignored and codes map by identity (warning) |
| Mappings of one `CMap` | 2¹⁸ | the rest are ignored (warning) |
| Type 3 `CharProc` | 4 MiB each, 64 MiB per font | the glyph is skipped (warning) |
| Tint-transform work of one image (`Separation`, `DeviceN` and the other spaces converted colour by colour) | 2²⁸ interpreted operations, and 2³² for the images of a page together | the image converts through a lattice of exact conversions with multilinear interpolation (warning); when no lattice fits what is left, the image is not drawn (warning) |
| One run of a Type 4 (PostScript calculator) function | 2²⁰ operations, `copy` and `roll` counting each entry they move | the evaluation fails (one warning per function) |
| JBIG2 page decode | memory besides the page bitmap: 16 × the page area + 2²⁶ bits, at most 2³³ bits (1 GiB); compositing work: 64 page areas + 2²⁸ pixels, at most 2³⁶ | the image fails (warning), as an image over the image limits does |

---

## Output format

| Flag | Extension | Description |
|---|---|---|
| _(default)_ | `.ppm` | RGB PPM (Netpbm P6). |
| `--gray` | `.pgm` | Grayscale PGM (Netpbm P5), converted with the PDF `DeviceRGB` → `DeviceGray` weights (0.30 R + 0.59 G + 0.11 B), or by a gray `--output-profile` (see [Colour management](#colour-management)). The page is rendered in RGB and converted in place, so peak memory is that of the RGB render. Can combine with `--png`, `--jpeg` and `--tiff`. |
| `--mono` | `.pbm` | 1-bit mono PBM (Netpbm P4): luma below 128 black, or halftoned with `--dither halftone`. Can combine with `--png`, `--jpeg` and `--tiff`. |
| `--png` | `.png` | PNG. Combine with `--gray` / `--mono` for grayscale/mono PNG, or with `--transparent` for RGBA. |
| `--jpeg` | `.jpg` | JPEG, colour or, with `--gray` / `--mono`, gray. |
| `--jpegopt OPTIONS` | | JPEG options, as `pdftoppm -jpegopt`: `quality=N` (0–100, default 75), `progressive=y\|n`, `optimize=y\|n` (optimal Huffman tables), comma-separated, e.g. `--jpegopt quality=90,progressive=y`. Needs `--jpeg` or `--jpegcmyk`. |
| `--jpegcmyk` | `.jpg` | CMYK JPEG, as `pdftoppm -jpegcmyk`: `--jpeg --cmyk`. |
| `--cmyk` | | Render CMYK pages (see [CMYK output](#cmyk-output)) and write them as CMYK TIFF (with `--tiff`) or CMYK JPEG (with `--jpeg`). |
| `--tiff` | `.tif` | Baseline TIFF, one file per page: RGB (RGBA with `--transparent`), gray with `--gray`, or 1 bit per pixel with `--mono` (`WhiteIsZero`, the fax interpretation). The resolution tags carry `-r` / `--rx` / `--ry`. |
| `--depth 8\|16` | | Bits per sample of the rendered pages, default `8`. `--depth 16` renders at 16 bits per channel throughout: colour conversions, shadings (evaluated straight to 16 bits), soft masks, transparency groups, blend modes, overprint simulation and images of more than 8 bits keep their levels, so smooth gradients do not band at 256 levels. Written as 16-bit PPM/PGM (maxval 65535, big-endian), PNG (bit depth 16), TIFF (`BitsPerSample` 16) or, with `--print`, a PDF 1.5 spool of `/BitsPerComponent 16` images. Combines with `--gray`, `--transparent`, `--invert`, `--overprint` and the band and size flags; a 16-bit page costs twice the memory (see [Size limits](#size-limits)). Combines with `--cmyk --tiff` for 16-bit CMYK TIFF. Refused at startup with `--mono`, `--dither`, `--fax`, `--jpeg`/`--jpegopt`/`--jpegcmyk`, `--crop`, `--rectify`, `--split-spreads`, `--dewarp`, `--gpu-pages` and comic-archive input ("--depth 16 is refused with --mono: it writes 1-bit pages; use --gray for 16-bit gray"), exit status 1. GPU decode and conversion paths are bypassed for the CPU's, which they match. |
| `--transparent` (`--transp`) | | Render over a transparent background instead of white paper, as `pdftocairo -transp`: RGBA PNG, or a TIFF with a fourth, unassociated-alpha sample (`ExtraSamples` 2). Alpha is the page's coverage × opacity and the colour is not premultiplied by it; a pixel nothing paints is `0, 0, 0, 0`. The page is painted as an isolated transparency group (a knockout one when its `/Group` says so), so a blend mode on the page sees only what the page painted. Needs `--png` or `--tiff`: PPM and JPEG carry no alpha and are refused at startup ("--transparent needs an output with an alpha channel: PPM has none; add --png or --tiff"), exit status 1, nothing rendered. Also refused with `--gray`, `--mono`, `--invert`, `--crop`, `--rectify`, `--dewarp`, `--split-spreads`, `--fax`, `--print` and comic-archive input. Works with every `--backend`. |
| `--tiffcompression METHOD` | | TIFF compression: `none` (default, as `pdftoppm`), `packbits`, `lzw`, `deflate`, or, with `--mono`, the CCITT fax codings `g3` (one-dimensional, MH), `g3-2d` (two-dimensional, MR; period 4 at 150 lines per inch or more, else 2) and `g4` (MMR, the most compact for bilevel pages). Fax-coded pages are one strip with byte-aligned end-of-line codes. `jpeg` is refused: use `--jpeg` for JPEG files. Needs `--tiff`. |
| `--fax` | `.tif` | A fax document: one TIFF Class F file (RFC 2306), `OUTPUT_PREFIX.tif`, holding every selected page. Each page is 1728 pixels across (the ITU-T T.4 scan line) at 204 pixels per inch, laid by its shorter side — a landscape page is turned a quarter counterclockwise, its top edge to the left — at `--fax-resolution` lines per inch, bilevel, one strip, coded `g3` (one-dimensional, the Class F baseline every fax reader takes; the default), `g3-2d` or `g4` by `--tiffcompression`. Pages reversed out of a flat dark ground are inverted (`--invert=auto`) unless `--invert` says otherwise; `--dither halftone` applies. If any page fails to render, no document is written and the pages not yet started are skipped. A page that runs longer than 364 mm on the fax (the T.30 B4 length) is sent as consecutive fax pages of at most 297 mm (the A4 length every receiver accepts); A4, Letter, Legal, A3 and Tabloid stay whole, since only the ratio of a page's sides sets its fax length. A very long, narrow page (a receipt roll) can still exceed the page budget at fax resolution and then fails the document. The document is always written to disk, even for a bare output prefix: `--ram` and `--ram-path` are refused. Fixes the format, geometry and resolution of its pages: refuses `--png`, `--jpeg`, `--gray`, `-r`/`--rx`/`--ry`, the `--scale-to` flags, `-x`/`-y`/`-W`/`-H`, and the camera stages `--crop`, `--rectify`, `--dewarp` and `--split-spreads`. At most 65 535 fax pages, counting the pieces of cut pages. PDF input only. |
| `--fax-resolution fine\|standard` | `fine` | Fax lines per inch: 196 (`fine`) or 98 (`standard`). Needs `--fax`. |
| `--fax-to NUMBER` | | Send the fax document once written, through the operating system's fax service: the CUPS fax queue `--fax-queue` names (`lp -d NAME -o phone=DIAL`), else HylaFAX (`sendfax -n -d DIAL`). Implies `--fax`. The number is dial characters — digits, `+`, `*`, `#`, the pause `,` — and the separators space, `-`, `.`, `(`, `)`, beginning with a digit, `+`, `*` or `#`; the dial string handed to the service drops the separators (`"+1 (555) 010-9999"` dials `+15550109999`). rrocket waits for the service to accept the job, the service's output going to standard error; a refusal or a missing service exits with status 1, naming where the document is. No modem code: dialling is the service's. Refused on Windows, where neither service exists: write the document with `--fax` and send it with Windows Fax and Scan. |
| `--fax-queue NAME` | | The CUPS fax queue `--fax-to` submits to: a name without white space or control characters, not beginning with `-`. Needs `--fax-to`. |
| `--dither METHOD` | | How `--mono` and `--fax` render intermediate grays: `none` (default), a 50% threshold that keeps text, line art and scanned text sharp, or `halftone`, dot patterns that keep photographs and shading legible but also dot text that is gray rather than black (a photographed or scanned page): a 4 × 4 dispersed ordered dither below 300 dpi, a stochastic clustered dot at 300 dpi or more. Needs `--mono` or `--fax`. |
| `--invert[=WHEN]` | | Invert the page's tones, as a negative, before encoding: `always` (also `--invert` alone), `never` (default; `auto` under `--fax`), or `auto`, only pages reversed out of a flat dark ground — over three fifths of the pixels within one band of 32 luma levels below mid-gray, as light text on a black or dark-coloured ground. A dark photograph or a page of saturated colours spreads over many bands and is left as it is. Applies to every format; with `--gray` / `--mono` the test and the inversion use the luma. The value takes `=`: `--invert auto` reads `auto` as the input file. |

`--gray` and `--mono` are mutually exclusive, and at most one of `--png`,
`--jpeg` and `--tiff` may be given; a conflicting pair exits with status 1.

**Extension matrix:**

| Format flag | `--mono` | `--gray` | Neither |
|---|---|---|---|
| _(default / PPM)_ | `.pbm` | `.pgm` | `.ppm` |
| `--png` | `.png` | `.png` | `.png` |
| `--jpeg` | `.jpg` | `.jpg` | `.jpg` |
| `--tiff` | `.tif` | `.tif` | `.tif` |

A gray `--output-profile` writes gray pages and names them as `--gray` does.

---

## CMYK output

`--cmyk` (with `--tiff` or `--jpeg`) and `--jpegcmyk` render each page in
the process inks of a CMYK printer, C, M, Y and K, 0 no ink:

- `DeviceCMYK` colours, images and shadings, `ICCBased` CMYK colours and
  images, and the process colorants of `Separation` and `DeviceN` colours
  keep their values: a `0 0 0 1 k` text is black ink alone, a 100 % cyan
  swatch cyan alone.
- Spot colorants fold into their process equivalents (their alternate
  colour's process ink).  The first four spot colorants of a page are kept
  as inks while it is painted, so an overprinting spot keeps the inks
  beneath it under `--overprint`; further ones print their alternate colour.
- Every other colour (`DeviceGray`, `DeviceRGB`, `Lab`, `CalRGB`, `ICCBased`
  RGB and gray, images and shadings in them) separates through the CMYK
  `--output-profile` when one is given, under `--intent` (default relative
  colorimetric) and `--bpc`; without one, by the conversion ISO 32000-2
  §10.4.2.3 defines for a device without a profile, as `pdftoppm -jpegcmyk`
  does: `c = 1 − r`, `m = 1 − g`, `y = 1 − b`, the common part moved to
  black, so gray is black ink alone (`0.5 g` is 50 % K) and black text is
  K only.  The `ExtGState` black-generation and undercolour-removal
  functions (`BG`, `UCR`) are not applied.
- Overprint (`OP`, `op`, `OPM`) is simulated with `--overprint`, as on RGB
  output; without it every object knocks out every ink beneath it.

A CMYK TIFF is `Separated` (Photometric 5) with `InkSet` 1, four samples of
8 or 16 bits (`--depth 16`), written band by band like other TIFFs, under any
`--tiffcompression` but the fax codings.  A CMYK JPEG is 8-bit, written as
Adobe JPEGs are: an `APP14` marker of colour transform 2 (YCCK), its samples
in the inverted sense that marker implies, so that readers of Adobe CMYK
JPEGs (Photoshop, Pillow, libjpeg-based viewers) recover the inks; this is
the layout `pdftoppm -jpegcmyk` writes.  `--jpegopt` applies.  With a CMYK
`--output-profile` both are tagged with it (TIFF tag 34675, JPEG `APP2`).

A CMYK page is painted on a canvas of eight inks a pixel, against the page
budget (see [Size limits](#size-limits)): 8 bytes a pixel, 16 at 16 bits.

Refused at start-up, exit status 1, nothing rendered: PNG and PPM, which hold
no CMYK ("--cmyk needs an output that holds CMYK: PPM has none; add --tiff or
--jpeg"); `--print`, `--fax`, `--mono`, `--gray`, `--dither`,
`--transparent`, `--invert`, `--crop`, `--rectify`, `--split-spreads`,
`--dewarp`, `--gpu-pages` and comic-archive input ("--cmyk is refused with
--gray: it writes gray pages"); `--jpegcmyk` with `--depth 16`; an RGB or gray
`--output-profile` with CMYK output, and a CMYK one without it.

Differences from `pdftoppm`: `pdftoppm -tiff -overprint` writes CMYK TIFF;
rrocket's `--tiff --overprint` stays RGB, and CMYK TIFF is `--tiff --cmyk`.
Spot inks fold into process ink as the union of coverages
(`a + b − a·b`), where `pdftoppm` adds and clips.

```bash
# CMYK TIFF of a press PDF at 300 dpi, overprint simulated
rrocket --cmyk --tiff --overprint -r 300 press.pdf out

# CMYK JPEG, separated through the printer's profile
rrocket --jpegcmyk --jpegopt quality=95 --output-profile coated.icc press.pdf out
```

---

## Colour management

Pages are composited in sRGB.  With an output profile each page is then
converted, once composited and band by band, to the profile's device values,
and the output is tagged with the profile so that a colour-managed RIP or
viewer reads the values for what they are.

| Flag | Default | Description |
|---|---|---|
| `--output-profile FILE` | | Convert every page to the ICC profile `FILE` and tag the output with it: a PNG `iCCP` chunk, TIFF tag 34675 (`InterColorProfile`), JPEG `APP2` `ICC_PROFILE` segments, and with `--print` an `ICCBased` colour space on every image of the spool (PDF 1.5, 1.6 or 1.7 as the profile's ICC version needs). PPM and PGM cannot carry a profile: their pixels are still converted, and a warning says so at start-up. The profile is an RGB, gray or CMYK profile of the output (`prtr`), display (`mntr`) or colour-space (`spac`) class, with an XYZ or Lab connection space; table-based (`AToB`/`BToA`) printer profiles, the kind paper profiles are, and matrix/TRC profiles alike. A gray profile writes gray pages (PGM, gray PNG, JPEG or TIFF, gray spool pages), the gray coming from the profile rather than the luma weights of `--gray`. A CMYK profile takes `--cmyk` or `--jpegcmyk` and separates the colours that are not CMYK already (see [CMYK output](#cmyk-output)). Works at both `--depth`s (the conversion runs on 16-bit samples either way) and with `--transparent` (an RGB profile; the alpha is unchanged). `--invert` complements the converted device values. Refused at start-up, exit status 1, nothing rendered: an unreadable file, a file that is not an ICC profile, a CMYK profile without `--cmyk` / `--jpegcmyk` and an RGB or gray one with them, a device-link, abstract, named-colour or input profile, an RGB profile with `--gray`, a gray one with `--transparent`, and `--mono`, `--dither`, `--fax` and comic-archive input. |
| `--intent INTENT` | the content's, else `relative` | The rendering intent of the conversion (of the separation, for a CMYK profile, which is relative colorimetric by default whatever the content's intents): `perceptual`, `relative` (relative colorimetric), `saturation` or `absolute` (absolute colorimetric: the source white is reproduced on the paper, its tint simulated). Without it, a page converts under the one intent its content paints with (`ri`, an `ExtGState` `/RI`, an image's `/Intent`; relative colorimetric where it sets none), and a page that mixes intents converts relative colorimetric, with one warning naming the intents seen. With it, the conversion is compiled at start-up, so a profile that cannot convert under it fails before anything renders. Needs `--output-profile`. |
| `--bpc` | off | Black-point compensation (ISO 18619): the page's black maps to the profile's darkest neutral, so shadow detail is kept on a paper whose black is lighter than the page's. A perceptual or saturation conversion to an ICC v4 profile always compensates, as the v4 perceptual connection space requires. Needs `--output-profile`; refused with `--intent absolute`. |

The conversion matches Little CMS (`transicc`) within ΔE00 1 on every patch
of a test chart, for every intent with and without `--bpc`, at 8 and 16 bits.
A page's colours outside sRGB are clipped to it before the conversion, and a
page converts under a single intent: objects of different intents on one page
share it.

```bash
# A fine-art print file: 16-bit TIFF for the paper profile, perceptual
rrocket --depth 16 --tiff --output-profile paper.icc --intent perceptual -r 360 art.pdf out

# Print through the profile, with black-point compensation
rrocket --print --output-profile paper.icc --bpc art.pdf
```

---

## Printing

| Flag | Default | Description |
|---|---|---|
| `--print` | | Print the selected pages through the operating system's print service instead of writing files; no `OUTPUT_PREFIX` is given. The pages are rendered at `-r` / `--rx` / `--ry` (default 600 dpi, or 360 dpi for a page whose 600 dpi canvas would hold more than the 1.8 GB of a 600 MP 8-bit RGB page: at 8 bits a page of more than about 1 667 sq in, larger than A0 or 36 × 44 in; under `--depth 16`, half that area), in colour, or gray with `--gray` / `--mono`, for printing (`--purpose print` unless `--purpose` is given), and written as one PDF holding each page as a single image at that resolution, which CUPS's `lp` submits. The queue's filters fit the images to the printer — media and orientation, colour, resolution and two-sided sheets — without rendering the pages again. Pages are begun in page order and appended to the spool in page order as they finish, a page once it and every page before it are done; no page is begun more than two per render thread past the first page not yet appended, so a long document is not held in memory. The spool is an anonymous temporary file, readable by no other user and gone however rrocket ends, in `$TMPDIR` when set, else `/var/tmp`; `lp` reads it as its standard input. If any page fails to render, nothing is submitted and the pages not yet started are skipped. rrocket waits for `lp` to accept the job, `lp`'s output going to standard error; a refusal exits with status 1. The job is named after the input file. Refuses `--png`, `--jpeg`, `--tiff`, `--singlefile`, `--fax`, `--ram` / `--ram-path` / `--no-ram`, `--sep`, `--forcenum` and the `--scale-to` flags. PDF input only. Refused on Windows, which has no `lp`. |
| `--printer NAME` | default destination | The CUPS destination (`lp -d NAME`), a queue or `queue/instance`; `--list-printers` lists them. Implies `--print`. When `lp` refuses the job, the message points at `--list-printers`. |
| `--list-printers` | | List the CUPS destinations `--printer` can name (`lpstat -e`), one per line, the default destination (`lpstat -d`) followed by a tab and `default`, then exit. Takes no input, output prefix or other flag. With no queues, writes nothing to standard output and says so on standard error. Refused on Windows. |
| `--copies N` | the queue's setting | Copies (1–9999), `lp -n N`. Needs `--print`. |
| `--duplex long\|short` | the printer's setting | Two-sided printing, the sheet turning on its long or short edge (`-o sides=two-sided-long-edge` / `two-sided-short-edge`). Needs `--print`. |

```bash
# The queues --printer can name
rrocket --list-printers

# Print pages 1–4 to the default printer at 600 dpi
rrocket --print -l 4 document.pdf

# Two copies, two-sided, gray, on the queue "Office"
rrocket --printer Office --copies 2 --duplex long --gray document.pdf
```

---

## Rendering

| Flag | Default | Description |
|---|---|---|
| `--aa yes\|no` | `yes` | Antialias text. `no` draws glyphs with hard, bilevel edges. |
| `--aaVector yes\|no` | `yes` | Antialias path fills, strokes and clips. `no` draws them with hard, bilevel edges. |
| `--overprint` | off | Simulate overprint (`OP`, `op`, `OPM`; ISO 32000-1 §8.6.7) for pages that use it. Without it every object knocks out what lies beneath it. A page on which some graphics state reachable from its content or from a drawn annotation sets `OP` or `op` is painted in four process and four spot inks and converted to RGB at the end of the page; an overprinting object leaves the inks it does not paint. A transfer function (`TR`, `TR2`) maps the inks: its red, green, blue and gray functions apply to cyan, magenta, yellow, and to black and the spot inks. The first four spot colorants of a page take the spot inks; further ones print their alternate colour in process ink and are named in a warning. Other pages render exactly as without the flag. With `--cmyk` the same simulation runs on the CMYK page, which is then written as its inks. The first overprint page builds a colour table once per process (about 0.75 s, 4.4 MB); pages rendering in parallel wait for it, so the first overprint page of a run is that much slower. Images, shadings and tiling-pattern cells paint the inks of their own colorants, `DeviceCMYK` ones their process inks; `ICCBased` CMYK colours print as the process ink that reproduces their profile's colour. Not simulated: a transparency group as a whole paints every ink. A black or near-black gray or RGB colour prints as a rich black balanced so that its antialiased edges stay neutral, within two counts of black when solid. |
| `--hide-annotations` | off | Draw no annotations (form fields, comments, stamps): the page content alone. |
| `--no-system-fonts` | off | Do not search the system font directories for fonts the document does not embed: draw them from `RROCKET_FONT_DIR` and the faces compiled into the binary only. |

Fonts a document does not embed are drawn with substitutes: the installed
Liberation (or Arial / Times New Roman / Courier New) family when found,
else faces compiled into the binary; CJK, Arabic and Thai text uses an
installed face of that script, and any character the substitute lacks is
drawn from an installed face that covers it. `RROCKET_FONT_DIR` names extra directories
to search first, in `PATH` syntax (`RROCKET_FONT_DIR=/opt/fonts:$HOME/fonts`), for both
the substitutes and the per-character faces; library callers set the same list through
`SessionConfig::font_dirs`. A character no installed face covers is drawn from the Noto
fallback faces compiled into the binary (Greek, Cyrillic, Hebrew, Arabic, Devanagari, Thai,
Bengali, Tamil, Armenian, Georgian, Ethiopic, symbols; the default `bundled-fonts` feature), so these scripts draw on a server with no
fonts installed; a build with `--no-default-features` omits them.

The font directories are read once per process and directory list, including
their subdirectories; a directory reached through more than one path or a
symbolic link is read once, and the walk stops after 10 000 directories with
a warning. The system directories searched are those of the operating
system rrocket was built for: the Windows and macOS font directories are
searched only on Windows and macOS.

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
| `--password PASSWORD` | Password for an encrypted PDF, tried as the user password and then as the owner password. `--upw` and `--opw` are aliases. Not needed when the user password is empty. |
| `--decrypt-owned` | Authorise decrypting an encrypted (Standard Security Handler) document without the interactive confirmation. Use it only for documents you own or are entitled to access. `RROCKET_DECRYPT_OWNED=1` does the same. |

Decryption runs in process (RC4, AES-128 and AES-256; revisions 2 to 6);
nothing is written to disk. Without `--decrypt-owned` or
`RROCKET_DECRYPT_OWNED=1`, an encrypted document prompts for a private-copy /
liability confirmation (default: no). A non-interactive run with neither set
exits with an error instead of removing the protection silently. A password
is an input to decryption, not an authorisation: `--password` alone does not
pass the gate. A wrong password, or none where one is needed, is an error
that says so.

---

## Parallelism and output

| Flag | Default | Description |
|---|---|---|
| `--threads N` | `0` (auto) | Number of render threads. `0` = one thread per logical CPU. |
| `--max-page-mib MIB` | `1716` (1.8 GB) | The most a page's bitmap may hold, in MiB of canvas (3 bytes a pixel for 8-bit RGB, 4 under `--transparent`, 8 on the ink canvas of `--cmyk` and `--overprint` pages, twice as many under `--depth 16`), 1 to 2³¹. A larger page is refused before anything is allocated; see [Size limits](#size-limits). Comic pages are held to it as decoded. |
| `--band-mib MIB` | `64` | The memory of a band of a page while it renders, 1 to 65 536. A page larger than one band renders in bands written as each completes; see [Size limits](#size-limits). |
| `--sep CHAR` | `-` | Separator between prefix and page number (`out-1.ppm`, `out_1.ppm`, etc.). |
| `--forcenum DIGITS` | — | Zero-pad page numbers to at least DIGITS digits (`001`, `002`, …). |
| `-P` / `--progress` | off | Print per-page progress (page done, elapsed, ETA) to stderr. |
| `--timings` | off | Print per-page wall time and thread index to stderr for profiling. |
| `--strict` | off | Fail a page when one of its images or content streams cannot be decoded or drawn, or decodes only in part. Without it the image or stream is skipped (or drawn from the data that decoded), the page is written, and a warning naming the page and object goes to stderr. |
| `--purpose view\|print` | `view` (`print` under `--print`) | What the pages are rendered for. `view`: optional-content groups (layers) take their View usage state, and annotations flagged NoView are omitted. `print`: layers take their Print usage state, and only annotations flagged Print are drawn. Annotations flagged Hidden are never drawn. |

---

## Output destination (RAM-backed by default)

| Flag | Default | Description |
|---|---|---|
| `--ram` | (heuristic) | Force RAM-backed output even when `OUTPUT_PREFIX` looks like a path. Refused on systems without `/dev/shm`; use `--ram-path` there. Mutually exclusive with `--no-ram`. |
| `--no-ram` | (heuristic) | Force on-disk output. Disables the default tmpfs redirect for bare stems and writes pages exactly at `OUTPUT_PREFIX`. Mutually exclusive with `--ram`. |
| `--ram-path PATH` | `/dev/shm/rrocket-<pid>-<nanos>/` | Override the tmpfs directory used by RAM-backed output. Implies `--ram`. The path must not exist: rrocket creates the directory and removes it, with everything in it, on exit like the default one; an existing path is refused. |

By default `rrocket` redirects rendered pages to a freshly-created tmpfs directory under `/dev/shm` whenever the output prefix is a bare stem, because writing to disk is 10–20× slower than RAM and dominates wall time on JPEG-heavy workloads. The directory is printed on stderr at startup and on stdout (so `dir=$(rrocket doc.pdf out)` captures it), and the prefix's file name is the stem inside it (`<dir>/out-1.ppm`, …). On systems without `/dev/shm` (macOS, Windows, the BSDs) there is no default redirect: a bare stem writes to the working directory, `--ram` is refused, and RAM-backed output needs `--ram-path` pointing into a RAM disk you have mounted. On macOS, `diskutil erasevolume APFS rrocket $(diskutil image attach --noMount ram://4194304)` mounts a 2 GiB one at `/Volumes/rrocket` (`hdiutil attach -nomount ram://4194304` on releases without `diskutil image`). On Apple-silicon Macs a RAM disk is no faster than the internal SSD for page output. RAM output is refused where the system's available memory cannot be read (the BSDs), since pages could not be moved to disk before memory runs out. The RAM directory is readable by its owner only.

**The RAM directory is removed when rrocket exits**, whether every page rendered or not. RAM output is for a consumer that reads pages while rrocket is still running; to keep the pages, give a path-like prefix (`./out`, `/data/out`) or pass `--no-ram`. A process killed by a signal, or aborted by a panic, leaves the directory behind; `/dev/shm` and a RAM disk are cleared at reboot, a `--ram-path` elsewhere is not.

**Bare stem vs path heuristic:** prefixes containing `/` or starting with `.` (e.g. `./out`, `/tmp/p`, `out/dir/p`) are treated as path-like and write to disk literally. Bare stems (e.g. `out`, `p`, `cover`) are redirected to tmpfs. `--ram` overrides the heuristic to redirect anyway; `--no-ram` overrides it to write to disk anyway.

A built-in spill policy reads the kernel's available-memory estimate before each page is written (at most once per 100 ms). From the first page that finds it below 1 GiB, that page and every later one are written to disk at `OUTPUT_PREFIX`, with a one-shot warning on stderr. The estimate is `MemAvailable` from `/proc/meminfo` on Linux, `kern.memorystatus_level` × `hw.memsize` on macOS and `ullAvailPhys` of `GlobalMemoryStatusEx` on Windows.

---

## Backend selection

| Flag | Default | Description |
|---|---|---|
| `--backend auto\|cpu\|cuda\|vaapi\|vulkan` | `auto` | Compute backend for image decoding and GPU fills. |
| `--vaapi-device PATH\|auto` | `auto` | VA-API DRM render node; `auto` probes for one (only used with `auto` or `vaapi`). |
| `--vulkan-device INDEX\|NAME` | automatic | Vulkan device, by index or name substring (only used with `auto` or `vulkan`). |
| `--gpu-pages` | off | Render qualifying pages whole on a Vulkan GPU beside the CPU workers. Needs a build with the `vulkan` feature, `--backend auto` or `vulkan`, and `--aaVector yes`; see below. |
| `--prefetch` | off | Prefetch image XObjects into the device-resident image cache when the document is opened. Present only in a binary built with the `cache` feature; worth it for long or repeated renders of one document. At a fixed resolution (no `--scale-to*`, fax or per-page print resolution) an image every draw of which on its page is smaller than the image is not prefetched: such a draw is resampled on the CPU and never reads the cache. |

When `--backend` is omitted, the `PDF_RASTER_BACKEND` environment variable is
read (same values; `vaapi` only in a build with the `vaapi` feature, and an
unrecognised value warns and means `auto`); if neither is set the backend is `auto`.

**`auto`** — the CPU, unless the parallel-Huffman JPEG decoder is enabled by its threshold; then Vulkan if compiled in and a device is present, else CUDA, else CPU, falling back silently at every step. The decoder is dormant by default, so `auto` initialises no GPU backend (no Vulkan device, CUDA context, image cache or nvJPEG decoder) unless `PDF_RASTER_HUFFMAN_THRESHOLD` is set to a pixel area below `u32::MAX` in a `gpu-jpeg-huffman` build (a value that does not parse warns and leaves the decoder dormant). A GPU session's start-up (a CUDA context and its kernels, about 70 ms, or a Vulkan device, 50–100 ms, plus their teardown) costs more than the image cache and the ICC lookup table save on every document measured, scanned, vector and mixed, 1 to 40 pages at 150 and 300 dpi. In an `nvjpeg2k` build, `auto` decodes a `JPXDecode` image of at least 262 144 px on the GPU unless the page draws it at half its size or less on both axes (the CPU then decodes a lower resolution level, which is faster); a rendering thread builds its nvJPEG2000 decoder when it first meets such an image, and never otherwise. `--backend cuda` decodes every such image on the GPU. `--backend cuda` and `--backend vulkan` always initialise their backend. VA-API (`vaapi` builds) is unaffected.

**`cpu`** — CPU only. All GPU init is skipped entirely. Use this for benchmarking, debugging, or on machines without a GPU.

**`cuda`** — Require CUDA. Exits with an error if the CUDA context cannot be initialised, or if the binary was built without the `gpu-icc` and `cache` features (the CUDA paths that need a context). Use this to confirm the GPU path is actually active rather than silently falling back.

**`vulkan`** — Require the Vulkan compute backend.  Exits with an error if Vulkan initialisation fails or the binary was built without `--features vulkan`.  The parallel-Huffman JPEG decode kernels, in a build that also has `gpu-jpeg-huffman`, dispatch through `VulkanBackend` (dormant by default; see `PDF_RASTER_HUFFMAN_THRESHOLD`); in a build with `cache`, the device-resident image cache keeps its images on the Vulkan device and draws them with the Vulkan blit kernel (a JPEG drawn below its resolution is not cached, under `--backend cuda` too); with `gpu-icc`, ICC CMYK→RGB runs on the Vulkan device.  Cross-vendor (NVIDIA, AMD, Intel, Apple via `MoltenVK`).

**`vaapi`** — Require VA-API JPEG decoding (only in a binary built with the `vaapi` feature). Exits with an error if no usable render node is found or the given one cannot be opened. The decoder is opened, but VA-API dispatch shares the nvJPEG size gate (`GPU_JPEG_THRESHOLD_PX`), which is `u32::MAX`, so JPEGs are still decoded on the CPU.

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

That error is fatal under `--backend auto` as well, because a selector that matches nothing is a typo, not missing hardware. When `--vulkan-device` is omitted, the `PDF_RASTER_VULKAN_DEVICE` environment variable is read (same syntax). Without either, rrocket prefers a discrete GPU, then integrated, virtual, and CPU devices. Among devices of the same type it takes the one with more device-local memory, then the lower PCI address, then the lower device UUID, so the choice does not depend on the order the driver lists devices in. No vendor is preferred: on a host with an NVIDIA and a non-NVIDIA discrete GPU with a Vulkan kernel enabled, pass `--vulkan-device` to choose the device, or `--backend cuda` to run on CUDA. With every GPU kernel dormant, `auto` leaves Vulkan idle and renders on the CPU. `RUST_LOG=debug` logs the chosen device and how it was chosen. `--vulkan-device` is rejected with `--backend cpu`, `cuda` or `vaapi`.

**`--gpu-pages`** opens a Vulkan device beside the CPU workers (the backend's own device under `--backend vulkan`, else the one `--vulkan-device` or `PDF_RASTER_VULKAN_DEVICE` names, else the automatically ranked one). The device opens in the background, and until it is open every page renders on the CPU. Once it is open, a page made only of solid fills and strokes (rectangle clips, Normal blend, constant alpha) renders on it while no other page holds it; every other page, and a qualifying page that finds it busy, renders on the CPU, with identical output. The flag is refused when the arguments are checked, before a password prompt and for comic input as well, with exit status 1: in a build without `vulkan` ("GPU page rendering (`gpu_pages`, --gpu-pages) needs a build with the `vulkan` feature; …"), with `--backend cpu`, `cuda` or `vaapi`, or the same value in `PDF_RASTER_BACKEND` ("… runs on Vulkan, which backend `cpu` (`policy`, --backend cpu or PDF_RASTER_BACKEND=cpu) does not use; use --backend auto or --backend vulkan, or drop --gpu-pages"), and with `--aaVector no` ("… paints antialiased vector paths only, and vector antialiasing is off (`vector_antialias`, --aaVector no); use --aaVector yes or drop --gpu-pages"). A device that does not open is logged at `error` and every page renders on the CPU. A page beyond the device's limits renders on the CPU with a warning and leaves the device in service; any other device error retires the device for the session with one warning, and that page and every later one render on the CPU. `RUST_LOG=rasterrocket=debug` logs each page the device renders, with its command count, and why each page it declines renders on the CPU.

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
| `1` | One or more pages failed (errors printed to stderr), the document could not be opened (or its decryption was not authorised), or the arguments are invalid. Clap's own argument errors exit `2`. |

Errors go to stderr as `rrocket: <message>` (`rrocket: page N: <message>` for a
page), sorted by page number, each followed by one `  caused by: <cause>` line
per level of its source chain that adds information the message does not
already contain.

---

## Examples

```bash
# Render all pages at 150 DPI (default) to ./out-1.ppm, ./out-2.ppm, …
rrocket document.pdf ./out

# Render pages 3–7 at 300 DPI, grayscale PNG
rrocket -f 3 -l 7 -r 300 --gray --png document.pdf out

# Render all pages at 300 DPI, 4 threads, with progress
rrocket -r 300 --threads 4 -P document.pdf out

# Render only odd pages
rrocket --odd document.pdf out

# Render the first page only, to cover.ppm
rrocket --singlefile document.pdf ./cover

# Render at 150 DPI, zero-pad page numbers to 3 digits: out-001.ppm
rrocket -r 150 --forcenum 3 document.pdf out

# Render with underscore separator: out_1.ppm
rrocket --sep _ document.pdf out

# Render only the top-left 1000 × 800 pixels of each page at 300 DPI
rrocket -r 300 -x 0 -y 0 -W 1000 -H 800 --png document.pdf tile

# Fit each page in a 2000-pixel box
rrocket --scale-to 2000 --png document.pdf out

# Encrypted document you own, with its password, no prompt
rrocket --decrypt-owned --password secret document.pdf out

# Print rendering: layers in their Print state, print-flagged annotations only
rrocket --purpose print document.pdf out

# Print-production file: simulate overprint and spot inks, PNG on disk
rrocket --overprint -r 300 --png document.pdf ./proof

# Compare output against pdftoppm (pixel-diff test)
tests/compare/compare.sh -r 150 -f 1 -l 5 document.pdf
```

---

## Pixel-diff comparison

`tests/compare/compare.sh` compares rrocket output page by page against a reference renderer (pdftoppm, run with `-cropbox`, by default; pdfium through a `pypdfium2` virtualenv with `-R pdfium`). A page fails when its RMSE exceeds `-t`, when the largest difference in mean luminance over 32 × 32 px tiles exceeds `-k` (this catches missing content that barely moves the page-wide RMSE), or when the two renders differ in size by more than 1 px in either dimension; renders within 1 px of each other are cropped to their common size before scoring. A document fails as a whole when rrocket exits non-zero.

**Requirements:** the reference renderer, `rrocket` built in release mode at `target/release/rrocket` (`RROCKET_BIN` names another binary), ImageMagick (`compare`, `identify`, `convert`), `bc`, `python3`. The pdfium virtualenv is `tests/compare/.venv` unless `PDFIUM_VENV` names another; `PDF_OXIDE_BIN` locates the `oxide` reference.

```bash
tests/compare/compare.sh [OPTIONS] <PDF>
tests/compare/compare.sh [OPTIONS] <DIR>   # every *.pdf under DIR, one summary row each

  -r DPI          Render resolution, a positive integer (default: 150)
  -f PAGE         First page (default: 1)
  -l PAGE         Last page (default: all)
  -t THRESHOLD    Max RMSE per page, 0–255 scale (default: 2.0)
  -k TILE_DELTA   Max tile delta per page, 0–255 scale (default: 12)
  -R RENDERER     Reference renderer: pdftoppm (default), pdfium or oxide
  -v              Verbose: print per-page stats even when passing
  -o DIR          Write diff images to DIR
  -n              Dry run: print commands without executing
  -s              Print one tab-separated summary row instead of the report
  -E FILE         Expectations registry (default tests/compare/expectations.json; '' for none)
  -U              Rewrite the registry from this run (--update-expectations)
  -S              Fail on surprise passes too (--strict)
  -h              Print the usage and exit
```

Exit code 0 when no page fails unexpectedly (see below); 1 on unexpected failures or errors. With `-U` and a registry the exit code is 0 unless an error stopped the run; `-U` with `-E ''` records nothing, and a failing run stays fatal. A dry run (`-n`) always exits 0. Before rendering, the registry is loaded once (`python3 -I tests/compare/expectations.py check`); a malformed one stops the run with a single error rather than failing every document.

The `-s` row, and each row of directory mode, holds the tab-separated columns `status pages failed size avg_rmse max_rmse tile stderr pdf`: `status` is the worst verdict (`ERROR` in directory mode when the run stopped before scoring), `failed` counts the pages over the limits, expected or not, and the average RMSE is over the pages scored, size mismatches and missing pages excluded. Directory mode prints a tally of the rows by status on standard error.

```bash
# Example output
── Summary ──────────────────────────────────────────────────────────────
  PDF:        document.pdf
  Reference:  pdftoppm
  DPI:        150
  Pages:      5 total, 5 within the limits, 0 over (0 size mismatches)
  Avg RMSE:   0.8432 / 255
  Max RMSE:   1.2100 / 255  (page-3)
  Max tile:   5.1200 / 255  (page-3)
  Thresholds: RMSE 2.0 / 255, tile 12.0 / 255
  rrocket stderr: 0 lines
  Verdicts:   0 unexpected failures, 0 expected failures, 0 surprise passes
  Registry:   tests/compare/expectations.json
─────────────────────────────────────────────────────────────────────────
PASSED
```

### Expected results

`compare.sh` and `tests/compare/pdfium_corpus.py` (rrocket on pdfium's own test corpus, scored against its expected images) share an expectations registry, `tests/compare/expectations.json`, so that known, accepted differences do not hide a new one. The registry holds one list of entries per suite: the harness with the settings that decide what passes (reference renderer, resolution, RMSE and tile limits). An entry names a document, or one of its pages, the failure expected (`differs`, `size`, `missing`, `unreadable` for a page; `error`, `strict`, `panic`, … for a document), a mandatory `reason`, and for `differs` optional `rmse` / `tile` limits that replace the suite's for that page; a limit is a finite, non-negative number, and a registry with any other value is malformed:

```json
{"file": "tests/fixtures/corpus-01-native-text-small.pdf", "page": 3,
 "outcome": "differs", "rmse": 15.32, "reason": "unclassified (seeded 2026-11-13)"}
```

Each document and page is classified; a document that renders is an observed pass, so a document entry for a failure that no longer occurs is a surprise pass:

| Verdict | Meaning |
|---|---|
| `PASS` | within the limits, not registered |
| `XFAIL` | a registered failure, within the registered limits |
| `FAIL` | unexpected: not registered, a different failure, or worse than registered |
| `XPASS` | surprise pass: registered but now passing; remove the entry |

Both harnesses exit 1 on any `FAIL`; a surprise pass is reported, and is fatal only with `--strict` (`-S`). The summary row of `compare.sh -s` carries the worst verdict in its status column, and `pdfium_corpus.py` adds a `verdict` column to its tables.

`--update-expectations` (`-U`) rewrites the registry from the run instead of failing: each new `FAIL`, or one with an outcome other than the registered one, becomes an entry with the reason `unclassified (seeded YYYY-MM-DD)` and limits 2 % above the observed values over the suite's limits, rounded up to 0.01; a `FAIL` worse than registered, of the registered outcome, gets those limits but keeps its reason, followed by `(worsened YYYY-MM-DD)`; an `XFAIL` `differs` entry's limits are lowered to 2 % above the observed values where that is lower, so an improvement is locked in, and a registered limit at or above the suite's own is dropped once its metric is within the suite's limit; each `XPASS` entry is removed; and entries for documents or pages the run did not cover are kept. Concurrent `-U` runs on one registry are serialised by a lock on `.expectations.json.lock` beside it, so parallel runs over different PDFs lose no entry. Review the diff, and replace each seeded reason with the actual cause once it is known.

```bash
# Seed or refresh the entries for the corpus fixtures
for f in tests/fixtures/corpus-*.pdf; do tests/compare/compare.sh -s -U "$f"; done

# The same for pdfium's corpus
python3 -I tests/compare/pdfium_corpus.py --pdfium ~/pdfium-src --out /tmp/pc --update-expectations
```

A PDF inside the repository is registered under its repository path, any other under its file name. The classification logic has a unit test, `python3 -I tests/compare/test_expectations.py`, which CI runs.

---

## Build and backend A/B comparison

Two scripts compare one `rrocket` binary against another byte for byte. Use them to check that a change leaves output untouched, or that a GPU backend matches the CPU. Both need bash 4.4 or later; on macOS that means Homebrew's `bash`. Each render runs under `timeout`, or `gtimeout` on macOS, when one is installed.

```bash
# Every output mode (RGB PNG, JPEG, gray PNG, mono PNG, gray PGM), 50 dpi, first 3 pages
tests/compare/ab_render.sh BEFORE AFTER OUTDIR [--] PDF...

# Corpus scan: 50 dpi gray, first 4 pages; LIST holds one PDF path per line
tests/compare/ab_archive.sh BEFORE AFTER OUTPREFIX [LIST]

# A GPU backend against the CPU, with one binary
AFTER_FLAGS="--backend vulkan" tests/compare/ab_render.sh \
  target/release/rrocket target/release/rrocket /tmp/ab -- tests/fixtures/*.pdf
```

`BEFORE_FLAGS` and `AFTER_FLAGS` append arguments to one side only, and `AB_TIMEOUT` sets the per-render limit in seconds. `ab_render.sh` prints a `DIFF` line for each differing file, an `EXIT` line when the two exit statuses differ, a `COUNT` line when the two sides exited alike but wrote different numbers of pages, and a `FAILED` line when both failed alike. It renders into a temporary directory inside `OUTDIR` and removes it on exit, and it fails when no output was compared at all.

Without a `LIST`, `ab_archive.sh` scans the repository fixtures plus a fixed sample of 150 PDFs from the directory `$AB_ARCHIVE` names. The sample is the same on every run, and the archive is skipped when `AB_ARCHIVE` is unset or names no directory. It writes `OUTPREFIX.list`, `OUTPREFIX.diffs` and `OUTPREFIX.errors`. The diffs file has a `PAGE`, `EXIT` or `COUNT` line for each difference: differing pixels on a page both sides wrote, a different exit status, or differing page counts under the same exit status. The errors file keeps the output of every render that exited non-zero. A file both builds fail on alike prints a `FAILED` line with the first line of the error.

Both scripts exit with 0 when nothing differs and something was compared, 1 when something differs or nothing was compared, and 2 on bad usage. The archive sample needs `shuf` (`gshuf` from Homebrew `coreutils` on macOS).
