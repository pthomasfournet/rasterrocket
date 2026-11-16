//! Command-line argument definitions.

use std::num::NonZeroU32;

use camera::CameraOptions;
use clap::Parser;
use pdf_raster::{BackendPolicy, PageBox, RenderPurpose, RenderRegion, Rotation, SessionConfig};

/// Parse `--jpegopt`: comma-separated `quality=N` (0–100),
/// `progressive=y|n` and `optimize=y|n`, as `pdftoppm -jpegopt` takes them.
fn parse_jpeg_options(s: &str) -> Result<encode::JpegOptions, String> {
    let mut options = encode::JpegOptions::default();
    for item in s.split(',').filter(|item| !item.is_empty()) {
        let (key, value) = item
            .split_once('=')
            .ok_or_else(|| format!("{item:?} is not KEY=VALUE"))?;
        let yes_no = || match value {
            "y" => Ok(true),
            "n" => Ok(false),
            _ => Err(format!("{key} takes y or n, not {value:?}")),
        };
        match key {
            "quality" => {
                options.quality = value
                    .parse()
                    .ok()
                    .filter(|q| *q <= 100)
                    .ok_or_else(|| format!("quality takes 0 to 100, not {value:?}"))?;
            }
            "progressive" => options.progressive = yes_no()?,
            "optimize" => options.optimize = yes_no()?,
            _ => {
                return Err(format!(
                    "unknown JPEG option {key:?}; the options are quality, progressive and optimize"
                ));
            }
        }
    }
    Ok(options)
}

/// Parse `--rotate`: an integer number of degrees, a multiple of 90.
fn parse_rotation(s: &str) -> Result<Rotation, String> {
    let degrees: i64 = s
        .parse()
        .map_err(|_| format!("'{s}' is not a whole number of degrees"))?;
    Rotation::from_degrees(degrees).map_err(|e| e.to_string())
}

/// Parser that rejects non-positive or non-finite DPI values at the CLI boundary.
fn parse_positive_dpi(s: &str) -> Result<f64, String> {
    let v: f64 = s
        .parse()
        .map_err(|_| format!("'{s}' is not a valid number"))?;
    if v.is_finite() && v > 0.0 {
        Ok(v)
    } else {
        Err(format!("DPI must be a positive finite number, got {s}"))
    }
}

/// Renders PDF pages to images.
#[derive(Parser, Debug)]
#[command(name = "rrocket", version, about, long_about = None)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "each bool maps to a distinct CLI flag"
)]
pub(crate) struct Args {
    /// Input file: a PDF, or a `.cbz`, `.cb7`, `.cbt` or `.cbr` comic archive; "-"
    /// reads a PDF from standard input. A path that begins with '-' must be
    /// passed after a literal `--` (or as `./-name.pdf`) so it is not parsed
    /// as an option.
    #[arg(required_unless_present = "list_printers")]
    input: Option<String>,

    /// Output filename prefix (page number and extension are appended).
    /// Not given with --print, which writes no files.
    #[arg(required_unless_present_any = ["print", "printer", "list_printers"])]
    pub output_prefix: Option<String>,

    // ── Page range ──────────────────────────────────────────────────────────
    /// First page to render (1-based, default: 1).
    #[arg(
        short = 'f',
        long = "first-page",
        default_value_t = 1,
        value_name = "N"
    )]
    pub first_page: i32,

    /// Last page to render (1-based, default: last page).
    #[arg(short = 'l', long = "last-page", value_name = "N")]
    pub last_page: Option<i32>,

    /// Render only odd pages.
    #[arg(short = 'o', long = "odd")]
    pub odd_only: bool,

    /// Render only even pages.
    #[arg(short = 'e', long = "even")]
    pub even_only: bool,

    /// Write only the first selected page, named `OUTPUT_PREFIX` plus the
    /// extension, without a page number.
    #[arg(long = "singlefile")]
    pub single_file: bool,

    // ── Resolution / scaling ─────────────────────────────────────────────────
    /// Render resolution in DPI (both axes, default 150).
    #[arg(
        short = 'r',
        long = "resolution",
        value_name = "DPI",
        value_parser = parse_positive_dpi
    )]
    pub resolution: Option<f64>,

    /// Horizontal resolution in DPI (overrides -r).
    #[arg(long = "rx", value_name = "DPI", value_parser = parse_positive_dpi)]
    pub resolution_x: Option<f64>,

    /// Vertical resolution in DPI (overrides -r).
    #[arg(long = "ry", value_name = "DPI", value_parser = parse_positive_dpi)]
    pub resolution_y: Option<f64>,

    /// Scale each page to fit a PIXELS × PIXELS box: the longer edge is
    /// PIXELS long and the aspect ratio is kept.  Overrides -r, --rx, --ry,
    /// --scale-to-x and --scale-to-y.
    #[arg(long = "scale-to", value_name = "PIXELS", value_parser = clap::value_parser!(u32).range(1..))]
    pub scale_to: Option<u32>,

    /// Scale each page to PIXELS wide.  Without --scale-to-y the height
    /// keeps the aspect ratio.  Overrides -r and --rx.
    #[arg(long = "scale-to-x", value_name = "PIXELS", value_parser = clap::value_parser!(u32).range(1..))]
    pub scale_to_x: Option<u32>,

    /// Scale each page to PIXELS high.  Without --scale-to-x the width
    /// keeps the aspect ratio.  Overrides -r and --ry.
    #[arg(long = "scale-to-y", value_name = "PIXELS", value_parser = clap::value_parser!(u32).range(1..))]
    pub scale_to_y: Option<u32>,

    // ── Crop ─────────────────────────────────────────────────────────────────
    /// Left edge of the rendered area, in pixels from the page's left edge
    /// (default 0).
    #[arg(short = 'x', long, value_name = "PIXELS")]
    pub crop_x: Option<u32>,

    /// Top edge of the rendered area, in pixels from the page's top edge
    /// (default 0).
    #[arg(short = 'y', long, value_name = "PIXELS")]
    pub crop_y: Option<u32>,

    /// Width of the rendered area in pixels (default: to the page's right
    /// edge; clipped to it).
    #[arg(short = 'W', long = "crop-width", value_name = "PIXELS", value_parser = clap::value_parser!(u32).range(1..))]
    pub crop_w: Option<u32>,

    /// Height of the rendered area in pixels (default: to the page's bottom
    /// edge; clipped to it).
    #[arg(short = 'H', long = "crop-height", value_name = "PIXELS", value_parser = clap::value_parser!(u32).range(1..))]
    pub crop_h: Option<u32>,

    /// The page boundary box rendered: `media`, `crop` (default), `bleed`,
    /// `trim` or `art`.  Every box but the `MediaBox` is intersected with the
    /// `MediaBox`; a page without the bleed, trim or art box named renders
    /// its `CropBox`.
    #[arg(
        long = "box",
        value_name = "BOX",
        value_enum,
        default_value = "crop",
        conflicts_with = "cropbox"
    )]
    pub page_box: BoxArg,

    /// Render the `CropBox`: `--box crop`, the default.
    #[arg(long = "cropbox")]
    pub cropbox: bool,

    /// Turn every page DEG degrees clockwise in addition to its own
    /// rotation: a multiple of 90, negative for counter-clockwise (default
    /// 0).  The page is drawn turned, so -x, -y, -W, -H and the --scale-to
    /// flags apply to the turned page.
    #[arg(
        long = "rotate",
        value_name = "DEG",
        allow_negative_numbers = true,
        value_parser = parse_rotation
    )]
    pub rotate: Option<Rotation>,

    // ── Output format ────────────────────────────────────────────────────────
    /// Output PNG (default: PPM).
    #[arg(long)]
    pub png: bool,

    /// Output JPEG.
    #[arg(long)]
    pub jpeg: bool,

    /// JPEG options, as `pdftoppm -jpegopt`: `quality=N` (0–100, default
    /// 75), `progressive=y|n`, `optimize=y|n`, comma-separated.
    #[arg(long = "jpegopt", value_name = "OPTIONS", value_parser = parse_jpeg_options)]
    pub jpegopt: Option<encode::JpegOptions>,

    /// Output CMYK JPEG: `--jpeg --cmyk`.
    #[arg(long)]
    pub jpegcmyk: bool,

    /// Render CMYK pages and write them as CMYK TIFF (with --tiff) or
    /// CMYK JPEG (with --jpeg, Adobe YCCK).  `DeviceCMYK`, ICC-based CMYK
    /// and process colorants keep their values; spot colorants fold into
    /// their process equivalents; other colours separate through a CMYK
    /// --output-profile, else as ISO 32000 §10.4.2 converts them, under
    /// the content's black generation and undercolour removal (gray is
    /// black ink alone).  Overprint is simulated with --overprint.  Not
    /// with PNG, PPM, --print, --fax, --mono, --gray, --transparent,
    /// --invert, the camera cleanup flags, --deskew, --clean-borders,
    /// --clean-background, --gpu-pages or comic input.
    #[arg(long)]
    pub cmyk: bool,

    /// Output TIFF.
    #[arg(long)]
    pub tiff: bool,

    /// TIFF compression: `none` (default), `packbits`, `lzw`, `deflate`,
    /// or, with --mono, the fax codings `g3` (one-dimensional), `g3-2d`
    /// and `g4`.
    #[arg(long = "tiffcompression", value_name = "METHOD", value_enum)]
    pub tiffcompression: Option<TiffMethod>,

    /// Render over a transparent background instead of white paper: PNG
    /// or TIFF with an alpha channel, the page's coverage × opacity, colour
    /// not premultiplied.  Needs --png or --tiff; not with --gray, --mono,
    /// --invert, the camera cleanup flags, --fax or --print.
    #[arg(long = "transparent", visible_alias = "transp")]
    pub transparent: bool,

    /// Output gray: PGM, or a gray PNG, JPEG or TIFF.
    #[arg(long = "gray")]
    pub gray: bool,

    /// Bits per sample of the rendered pages: `8` (default) or `16`.
    /// 16-bit pages are rendered at 16 bits per sample (shadings, soft
    /// masks and colour conversions keep their levels) and written as
    /// 16-bit PPM and PGM (maxval 65535), PNG and TIFF, or into the
    /// --print spool; not with --mono, --dither, --fax, --jpeg, the camera
    /// cleanup flags or --gpu-pages, nor for comic archives.
    #[arg(long = "depth", value_name = "BITS", value_enum, default_value = "8")]
    pub depth: DepthArg,

    /// Output 1-bit mono: PBM, or a two-level PNG, JPEG or TIFF.
    #[arg(long = "mono")]
    pub mono: bool,

    /// Convert every page to the ICC profile FILE once composited and tag
    /// the output with it (PNG iCCP, TIFF tag 34675, JPEG APP2, the
    /// --print spool's page images; PPM and PGM cannot carry it).  An
    /// RGB, gray or CMYK output, display or colour-space profile; a gray
    /// one writes gray pages, a CMYK one takes --cmyk or --jpegcmyk and
    /// separates the colours that are not CMYK already.  Not with --mono,
    /// --dither, --fax or comic input.
    #[arg(long = "output-profile", value_name = "FILE")]
    pub output_profile: Option<std::path::PathBuf>,

    /// The rendering intent of the conversion to --output-profile:
    /// `perceptual`, `relative` (relative colorimetric), `saturation` or
    /// `absolute` (absolute colorimetric).  Default: the one intent a
    /// page's content uses (`ri`, `/RI`, an image's `/Intent`), else
    /// relative colorimetric; for a CMYK profile, relative colorimetric.
    /// Under --proof, the intent of the separation to the press, default
    /// relative colorimetric.
    #[arg(long = "intent", value_name = "INTENT", value_enum)]
    pub intent: Option<IntentArg>,

    /// Black-point compensation in the conversion to --output-profile
    /// (under --proof, in the separation to the press): the page's black
    /// maps to the profile's darkest neutral.  Not with --intent absolute.
    #[arg(long = "bpc")]
    pub bpc: bool,

    /// Soft-proof the pages: show on screen how they print on the press
    /// the document's PDF/X output intent names.  Each page is painted as
    /// that press's process ink (`DeviceCMYK` and process colorants as
    /// given, other colours separated through the intent's profile under
    /// --intent and --bpc, overprint simulated with --overprint), then
    /// converted to sRGB, or to an RGB or gray --output-profile.
    /// `relative` (the bare flag) maps the paper to white; `absolute`
    /// simulates the paper's white and the press's black as measured.  The
    /// document (or the page) must carry `/OutputIntents` with an embedded
    /// CMYK profile.  Not with --cmyk, --jpegcmyk, --transparent, a CMYK
    /// --output-profile, --mono, --dither, --fax or comic input.
    #[arg(
        long = "proof",
        value_name = "INTENT",
        value_enum,
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "relative"
    )]
    pub proof: Option<ProofArg>,

    /// The working space the pages are composited in before the conversion
    /// to --output-profile: `wide` (the ROMM RGB primaries with the sRGB
    /// tone curve), which keeps colours outside sRGB up to the conversion,
    /// or `srgb`.  Default: `wide` at --depth 16 under --output-profile,
    /// `srgb` otherwise; an 8-bit page under --output-profile is
    /// composited in sRGB and warns once that its colours outside sRGB
    /// clip, unless `srgb` is given.  Blending and transparency groups
    /// composite in the working space, and transfer functions map its
    /// components; overprint-preview pages (--overprint) are composited in
    /// sRGB.  `wide` needs --output-profile and --depth 16.
    #[arg(long = "working-space", value_name = "SPACE", value_enum)]
    pub working_space: Option<WorkingSpaceArg>,

    /// The ICC profile the output is tagged with: --output-profile's,
    /// set from the session's profile once it is read.
    #[arg(skip)]
    pub output_icc: Option<encode::IccProfile>,

    /// Write a fax document: one TIFF Class F file, `OUTPUT_PREFIX.tif`,
    /// every page 1728 pixels wide (landscape pages turned), bilevel, at
    /// the --fax-resolution, coded as --tiffcompression `g3` (default),
    /// `g3-2d` or `g4`.  Pages reversed out of a dark ground are inverted unless --invert
    /// says otherwise.
    #[arg(long)]
    pub fax: bool,

    /// Fax resolution: `fine` (204 × 196 dpi, default) or `standard`
    /// (204 × 98 dpi).
    #[arg(long = "fax-resolution", value_name = "RESOLUTION", value_enum)]
    pub fax_resolution: Option<FaxResolution>,

    /// Send the fax document to NUMBER once written: through the CUPS fax
    /// queue --fax-queue names, else through `HylaFAX`'s `sendfax`.  Implies
    /// --fax.
    #[arg(long = "fax-to", value_name = "NUMBER", value_parser = crate::fax::parse_number)]
    pub fax_to: Option<String>,

    /// The CUPS fax queue --fax-to submits to (`lp -d NAME -o phone=NUMBER`).
    #[arg(long = "fax-queue", value_name = "NAME", requires = "fax_to", value_parser = crate::deliver::parse_queue)]
    pub fax_queue: Option<String>,

    /// How --mono and --fax render intermediate grays: `none` (default), a 50%
    /// threshold that keeps text and line art sharp, or `halftone`, dot
    /// patterns that keep photographs and shading legible: a dispersed
    /// ordered dither below 300 dpi, a stochastic clustered dot at 300 dpi
    /// or more.
    #[arg(
        long = "dither",
        value_name = "METHOD",
        value_enum,
        default_value = "none"
    )]
    pub dither: DitherArg,

    /// Invert the page's tones, as a negative: `always`, `never`, or
    /// `auto`, only pages reversed out of a flat dark ground (over three
    /// fifths of the pixels within one dark tone band), such as light text
    /// on black; dark photographs are left as they are.
    /// `--invert` alone is `always`.  Default: `never`.
    #[arg(
        long = "invert",
        value_name = "WHEN",
        value_enum,
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "always"
    )]
    pub invert: Option<InvertWhen>,

    /// Print the pages through the system's print service: CUPS's `lp`
    /// takes them as one PDF of page images at the resolution flags
    /// (default 600 dpi, 360 dpi for a page larger than A0), colour, or
    /// gray with --gray and --mono, and fits
    /// them to the printer.  Pages render for printing unless --purpose
    /// says otherwise.  No `OUTPUT_PREFIX` is given.
    #[arg(long)]
    pub print: bool,

    /// The printer --print submits to, a CUPS queue (`lp -d NAME`), else
    /// the default destination.  Implies --print.
    #[arg(long = "printer", value_name = "NAME", value_parser = crate::deliver::parse_queue)]
    pub printer: Option<String>,

    /// List the printers --printer can name, the CUPS queues, one per line,
    /// the default destination followed by a tab and `default`; then exit.
    /// Takes no input or other option.
    #[arg(long = "list-printers", exclusive = true)]
    pub list_printers: bool,

    /// Copies --print asks for.
    #[arg(long = "copies", value_name = "N", value_parser = clap::value_parser!(u32).range(1..=9999))]
    pub copies: Option<u32>,

    /// Two-sided printing with --print, the sheet turning on its `long` or
    /// `short` edge.  Default: the printer's own setting.
    #[arg(long = "duplex", value_name = "EDGE", value_enum)]
    pub duplex: Option<crate::print::Duplex>,

    // ── Rendering options ────────────────────────────────────────────────────
    /// Antialias text (default yes).  `no` draws glyphs with hard,
    /// bilevel edges.
    #[arg(long = "aa", value_name = "yes|no", default_value = "yes")]
    pub antialias: AaFlag,

    /// Antialias path fills, strokes and clips (default yes).  `no` draws
    /// them with hard, bilevel edges.
    #[arg(long = "aaVector", value_name = "yes|no", default_value = "yes")]
    pub vector_antialias: AaFlag,

    /// Simulate overprint (`OP`, `op`, `OPM`) for pages that use it
    /// (default off), on RGB and gray output and on --cmyk pages.  Without
    /// it every object knocks out what lies beneath it.
    #[arg(long)]
    pub overprint: bool,

    /// Do not draw annotations (form fields, comments, stamps and other
    /// PDF annotations); render the page content alone.
    #[arg(long = "hide-annotations")]
    pub hide_annotations: bool,

    /// Do not search the system font directories for fonts the document
    /// does not embed: draw them from `RROCKET_FONT_DIR` and the faces
    /// compiled into the binary only.
    #[arg(long = "no-system-fonts")]
    pub no_system_fonts: bool,

    // ── Encrypted-document decryption ────────────────────────────────────────
    /// Password for an encrypted PDF, tried as the user password and then
    /// as the owner password (`--upw` and `--opw` are aliases).
    ///
    /// Not needed for a document whose user password is empty.  A password
    /// is an input to decryption, not an authorisation: decrypting still
    /// requires the liability confirmation, `--decrypt-owned` or
    /// `RROCKET_DECRYPT_OWNED=1`.
    #[arg(long = "password", visible_aliases = ["upw", "opw"], value_name = "PASSWORD")]
    pub password: Option<String>,

    /// Authorise decrypting an encrypted (PDF Standard Security Handler)
    /// document without the interactive liability confirmation.
    ///
    /// Intended SOLELY for documents you own or are otherwise legally
    /// entitled to access.  Setting this (or the `RROCKET_DECRYPT_OWNED=1`
    /// environment variable) affirms you have the lawful right to decrypt
    /// the document.  Without it, an encrypted document triggers an
    /// interactive private-copy / liability prompt (default: no); a
    /// non-interactive run with neither set aborts with a clear error
    /// rather than silently stripping protection.
    #[arg(long = "decrypt-owned")]
    pub decrypt_owned: bool,

    // ── Parallelism ──────────────────────────────────────────────────────────
    /// Number of threads (0 = auto-detect).
    #[arg(long = "threads", value_name = "N", default_value_t = 0)]
    pub num_threads: usize,

    // ── Output naming ────────────────────────────────────────────────────────
    /// Separator character between prefix and page number (default "-").
    #[arg(long = "sep", value_name = "CHAR", default_value = "-")]
    pub separator: char,

    /// Zero-pad page numbers to at least N digits.
    #[arg(long = "forcenum", value_name = "DIGITS")]
    pub force_num_digits: Option<usize>,

    // ── Progress ─────────────────────────────────────────────────────────────
    /// Print progress to stderr: pages done, elapsed time, and ETA.
    #[arg(short = 'P', long = "progress")]
    pub progress: bool,

    /// Print per-page wall time and thread index to stderr for profiling.
    #[arg(long = "timings")]
    pub timings: bool,

    // ── RAM-backed output ─────────────────────────────────────────────────────
    /// Force RAM-backed output even when `OUTPUT_PREFIX` looks like a path.
    ///
    /// By default the program already redirects bare-stem prefixes (e.g.
    /// `rrocket doc.pdf out`) to a freshly-created tmpfs directory under
    /// `/dev/shm`, because writing to disk is 10–20× slower than RAM and
    /// dominates wall time. Path-like prefixes (anything with `/` or
    /// starting with `.`) opt out of that redirect and write to disk, as do
    /// all prefixes on systems without `/dev/shm` (macOS, Windows, the BSDs).
    ///
    /// Use `--ram` to override the heuristic: redirect to tmpfs even when
    /// the user-supplied prefix would otherwise be treated as a real path.
    /// Without `/dev/shm`, `--ram` is refused; pass `--ram-path` instead.
    /// Mutually exclusive with `--no-ram`.
    #[arg(long = "ram", conflicts_with = "no_ram")]
    pub ram: bool,

    /// Force on-disk output. Disables the default tmpfs redirect for bare
    /// stems and writes pages exactly at `OUTPUT_PREFIX`.
    /// Mutually exclusive with `--ram`.
    #[arg(long = "no-ram", conflicts_with = "ram")]
    pub no_ram: bool,

    /// Override the tmpfs directory used by RAM-backed output. Implies the
    /// RAM redirect. Default is a fresh `/dev/shm/rrocket-<pid>-<nanos>/`.
    /// On systems without `/dev/shm` (macOS, Windows), point it into a RAM disk.
    /// The path must not exist: the directory is created for the run and
    /// removed, with its contents, on exit.
    #[arg(long = "ram-path", value_name = "PATH", conflicts_with = "no_ram")]
    pub ram_path: Option<String>,

    /// Comic input: the most an archive's entries may decompress to in
    /// total, in MiB (default 4096).  An archive over it is refused whole as
    /// a possible decompression bomb; raise it for a genuine archive that
    /// large.  Decompressed entries are held in memory while the pages
    /// render.
    #[arg(long = "max-archive-mib", value_name = "MIB", value_parser = clap::value_parser!(u64).range(1..=(u64::MAX >> 20)))]
    pub max_archive_mib: Option<u64>,

    /// The most a page's bitmap may hold, in MiB of canvas: 3 bytes a pixel
    /// for 8-bit RGB, 6 for 16-bit (default 1716, i.e. 1.8 GB, a 600 Mpx
    /// page of 8-bit RGB).  A larger page is refused before anything is
    /// allocated; raise it for a genuinely large page (a poster or a roll
    /// banner), or lower the resolution.  Comic pages are held to it as
    /// decoded.  At most 2^31 MiB (2 PiB), the 16-bit RGBA bitmap of the
    /// largest page the renderer accepts at any budget.
    #[arg(long = "max-page-mib", value_name = "MIB", value_parser = clap::value_parser!(u64).range(1..=MAX_PAGE_MIB))]
    pub max_page_mib: Option<u64>,

    /// Set by [`Self::apply_print_preset`] when the print resolution is
    /// chosen per page.
    #[arg(skip)]
    pub print_resolution_per_page: bool,

    /// The most memory a band of a page holds while it renders, in MiB
    /// (default 64; 1 to 65536).  A page larger than one band renders in
    /// horizontal bands written to the output as each completes, so memory
    /// holds one band per worker thread, not the page; PPM, PNG, TIFF and
    /// --print output stream.  JPEG output, --fax, --invert auto, --deskew,
    /// --clean-borders, --clean-background, --threshold sauvola and the
    /// camera stages need the whole page and render it in one band.
    #[arg(long = "band-mib", value_name = "MIB", value_parser = clap::value_parser!(u64).range(1..=MAX_BAND_MIB))]
    pub band_mib: Option<u64>,

    // ── Backend selection ─────────────────────────────────────────────────────
    /// Compute backend for image decoding and GPU kernels.
    ///
    /// `auto`   — CPU, unless the GPU Huffman JPEG decoder is enabled by its
    ///            threshold override (`PDF_RASTER_HUFFMAN_THRESHOLD`); then
    ///            Vulkan if compiled in
    ///            and present, else CUDA, else CPU (silent fallback at every
    ///            step).  Independently, a binary built with `nvjpeg2k`
    ///            decodes a JPEG 2000 image of at least 2 097 152 pixels on
    ///            the GPU when it is drawn at more than half its size, and
    ///            --gpu-pages opens a Vulkan device.  This is the default.
    /// `cpu`    — CPU only; all GPU init is skipped.
    /// `cuda`   — Require CUDA (nvJPEG/ICC/image cache); exit with error if unavailable.
    /// `vaapi`  — Require VA-API JPEG; exit with error if the DRM device cannot be opened.
    /// `vulkan` — Require the Vulkan compute backend.
    ///
    /// When `--backend` is omitted, the `PDF_RASTER_BACKEND` environment
    /// variable is consulted (same valid values).  If neither is set,
    /// the resolved policy is `auto`.
    #[arg(long = "backend", value_name = "BACKEND", verbatim_doc_comment)]
    pub backend: Option<BackendArg>,

    /// VA-API DRM render node, or `auto` (the default).
    ///
    /// `auto` probes /dev/dri/renderD* in ascending number and uses the
    /// first node whose driver supports VA-API baseline JPEG decode.
    /// Only used with `--backend auto` or `--backend vaapi` in a binary
    /// built with the `vaapi` feature.
    #[arg(long = "vaapi-device", value_name = "PATH|auto", verbatim_doc_comment)]
    pub vaapi_device: Option<String>,

    /// Vulkan device: an index in enumeration order (from 0) or a
    /// case-insensitive substring of the device name.
    ///
    /// An all-digit value that is not a valid index is matched as a name.
    /// A value matching no device, or several, is an error that lists the
    /// available devices.  When omitted, `PDF_RASTER_VULKAN_DEVICE` is
    /// consulted; if neither is set the device is picked automatically.
    /// Only used with `--backend auto` or `--backend vulkan`.
    #[arg(
        long = "vulkan-device",
        value_name = "INDEX|NAME",
        verbatim_doc_comment
    )]
    pub vulkan_device: Option<String>,

    /// Render pages whole on a Vulkan GPU beside the CPU workers.
    ///
    /// The device opens in the background; once open, a page made only of
    /// solid fills and strokes (rectangle clips, Normal blend, constant
    /// alpha) renders on it while it is idle, every other page on the CPU,
    /// with identical output.  Needs a build with the `vulkan` feature,
    /// --backend auto or vulkan, and --aaVector yes.
    #[arg(long = "gpu-pages", verbatim_doc_comment)]
    pub gpu_pages: bool,

    /// Fail a page when one of its images or content streams cannot be
    /// decoded or drawn.
    ///
    /// By default the image or stream is skipped, the page is written
    /// without it, a warning naming the page and object goes to stderr, and
    /// the exit status is unaffected.  With `--strict` the page is not written and the run
    /// exits 1.
    #[arg(long = "strict")]
    pub strict: bool,

    /// What the pages are rendered for.
    ///
    /// `view`  — on-screen viewing (the default without --print): layers
    ///           take their View usage state and annotations marked not for
    ///           viewing are omitted.
    /// `print` — printing (the default with --print): layers take their
    ///           Print usage state and only annotations marked for printing
    ///           are drawn.
    #[arg(long = "purpose", value_name = "PURPOSE", verbatim_doc_comment)]
    pub purpose: Option<PurposeArg>,

    // ── Camera-capture cleanup ───────────────────────────────────────────────
    /// Detect a page photographed on a darker surround and crop to its
    /// bounding box.
    ///
    /// Pages without a clear boundary against a darker surround (flatbed
    /// scans, born-digital pages) are written unchanged.
    #[arg(long = "crop")]
    pub crop: bool,

    /// Detect a page photographed on a darker surround, correct the
    /// perspective and crop to the page.  Supersedes `--crop`.
    ///
    /// Pages without a clear boundary against a darker surround (flatbed
    /// scans, born-digital pages) are written unchanged.
    #[arg(long = "rectify")]
    pub rectify: bool,

    /// Straighten text lines curved by a page bending toward the spine.
    /// Experimental.
    ///
    /// Pages whose text lines are already straight, or that carry fewer
    /// than five lines of text, are written unchanged.
    #[arg(long = "dewarp")]
    pub dewarp: bool,

    /// Straighten pages scanned at a slant: measure the tilt of each page's
    /// lines, up to ±6.75°, and rotate the page to correct it.  Gray output
    /// only (`--gray`, `--mono` or `--fax`).
    ///
    /// Pages without a confident angle (photographs, blank pages, pages
    /// without lines of text) are written unchanged.
    #[arg(long = "deskew")]
    pub deskew: bool,

    /// Split a double-page spread at its gutter into two output files.
    ///
    /// The halves of page N are written as N with an `a` (left) and `b`
    /// (right) suffix, e.g. `out-07a.png` and `out-07b.png`; pages not
    /// detected as spreads keep their usual name.
    #[arg(long = "split-spreads")]
    pub split_spreads: bool,

    // ── OCR preparation ─────────────────────────────────────────────────────
    /// Whiten the dark strips a scanner leaves along the page edges: dark
    /// regions joined to the edge and lying within 6% of the shorter side
    /// of it.  The page keeps its size.
    ///
    /// Pages without such regions (born-digital pages, pages on a dark
    /// ground) are written unchanged.
    #[arg(long = "clean-borders")]
    pub clean_borders: bool,

    /// Make the paper white under uneven lighting; ink keeps its tone
    /// relative to the paper around it.
    ///
    /// Pages whose background is already white (born-digital pages) are
    /// written unchanged.  A light photograph loses its gradual shading
    /// and a paper tint survives in colour: meant for scans.
    #[arg(long = "clean-background")]
    pub clean_background: bool,

    /// How --mono and --fax choose black: `fixed` (default), luma below
    /// 128, or `sauvola`, a threshold from the mean and spread of the
    /// 0.33-inch square around each pixel, which follows paper whose
    /// brightness varies across the page.  Not with --dither halftone.
    #[arg(
        long = "threshold",
        value_name = "METHOD",
        value_enum,
        default_value = "fixed"
    )]
    pub threshold: ThresholdArg,

    /// Prefetch image `XObject`s into the device-resident image cache at
    /// session open.  Has no effect on a session without an image cache,
    /// such as one under `--backend cpu`, or `--backend auto` while every
    /// GPU kernel is dormant.  Off by default — the prefetch walk is wasted work for
    /// short single-page renders, but a clear win on multi-page or
    /// multi-pass workloads (OCR pipelines).
    #[cfg(feature = "cache")]
    #[arg(long = "prefetch")]
    pub prefetch: bool,
}

/// `--purpose` argument value parsed by clap.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum PurposeArg {
    /// On-screen viewing.
    View,
    /// Printing.
    Print,
}

impl From<PurposeArg> for RenderPurpose {
    fn from(p: PurposeArg) -> Self {
        match p {
            PurposeArg::View => Self::View,
            PurposeArg::Print => Self::Print,
        }
    }
}

/// `--box` argument value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum BoxArg {
    /// The `MediaBox`.
    Media,
    /// The `CropBox`.
    Crop,
    /// The `BleedBox`.
    Bleed,
    /// The `TrimBox`.
    Trim,
    /// The `ArtBox`.
    Art,
}

impl From<BoxArg> for PageBox {
    fn from(b: BoxArg) -> Self {
        match b {
            BoxArg::Media => Self::Media,
            BoxArg::Crop => Self::Crop,
            BoxArg::Bleed => Self::Bleed,
            BoxArg::Trim => Self::Trim,
            BoxArg::Art => Self::Art,
        }
    }
}

/// `--fax-resolution` argument value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum FaxResolution {
    /// 196 lines per inch.
    Fine,
    /// 98 lines per inch.
    Standard,
}

impl FaxResolution {
    /// Lines per inch.
    const fn y_dpi(self) -> f64 {
        match self {
            Self::Fine => 196.0,
            Self::Standard => 98.0,
        }
    }
}

/// `--depth` argument value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum DepthArg {
    /// 8 bits per sample.
    #[value(name = "8")]
    Eight,
    /// 16 bits per sample.
    #[value(name = "16")]
    Sixteen,
}

impl From<DepthArg> for pdf_raster::SampleDepth {
    fn from(depth: DepthArg) -> Self {
        match depth {
            DepthArg::Eight => Self::Eight,
            DepthArg::Sixteen => Self::Sixteen,
        }
    }
}

/// `--intent` argument value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum IntentArg {
    /// Perceptual.
    Perceptual,
    /// Relative colorimetric.
    Relative,
    /// Saturation.
    Saturation,
    /// Absolute colorimetric.
    Absolute,
}

impl From<IntentArg> for pdf_raster::RenderingIntent {
    fn from(intent: IntentArg) -> Self {
        match intent {
            IntentArg::Perceptual => Self::Perceptual,
            IntentArg::Relative => Self::RelativeColorimetric,
            IntentArg::Saturation => Self::Saturation,
            IntentArg::Absolute => Self::AbsoluteColorimetric,
        }
    }
}

/// `--working-space` argument value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum WorkingSpaceArg {
    /// sRGB.
    Srgb,
    /// The ROMM RGB primaries with the sRGB tone curve.
    Wide,
}

impl From<WorkingSpaceArg> for pdf_raster::WorkingSpace {
    fn from(space: WorkingSpaceArg) -> Self {
        match space {
            WorkingSpaceArg::Srgb => Self::Srgb,
            WorkingSpaceArg::Wide => Self::Wide,
        }
    }
}

/// `--proof` argument value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum ProofArg {
    /// Relative colorimetric: the paper maps to white.
    Relative,
    /// Absolute colorimetric: the paper's white and the press's black as
    /// measured.
    Absolute,
}

impl From<ProofArg> for pdf_raster::ProofIntent {
    fn from(intent: ProofArg) -> Self {
        match intent {
            ProofArg::Relative => Self::RelativeColorimetric,
            ProofArg::Absolute => Self::AbsoluteColorimetric,
        }
    }
}

/// `--dither` argument value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum DitherArg {
    /// Threshold at mid-gray.
    None,
    /// Halftone screen.
    Halftone,
}

/// `--threshold` argument value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum ThresholdArg {
    /// Luma below 128 is black.
    Fixed,
    /// Sauvola's local threshold.
    Sauvola,
}

/// `--invert` argument value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum InvertWhen {
    /// Every page.
    Always,
    /// Pages reversed out of a flat dark ground.
    Auto,
    /// No page.
    Never,
}

/// Yes/no flag for anti-aliasing options.
#[derive(Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum AaFlag {
    /// Anti-aliasing enabled.
    Yes,
    /// Anti-aliasing disabled.
    No,
}

impl Args {
    /// The camera-capture cleanup stages selected by `--crop`,
    /// `--rectify`, `--dewarp` and `--split-spreads`.
    pub(crate) fn camera_options(&self) -> CameraOptions {
        let mut camera = CameraOptions::default();
        camera.crop = self.crop;
        camera.rectify = self.rectify;
        camera.dewarp = self.dewarp;
        camera.split_spreads = self.split_spreads;
        camera
    }

    /// The halftone screen of `--dither halftone` at the output
    /// resolution, or `None` for thresholding.
    pub(crate) fn screen(&self) -> Option<raster::ScreenParams> {
        (self.dither == DitherArg::Halftone).then(|| {
            if self.x_dpi().min(self.y_dpi()) < 300.0 {
                raster::ScreenParams {
                    kind: raster::ScreenType::Dispersed,
                    size: 4,
                    dot_radius: 2,
                }
            } else {
                raster::ScreenParams {
                    kind: raster::ScreenType::StochasticClustered,
                    size: 64,
                    dot_radius: 2,
                }
            }
        })
    }

    /// The bytes of a band of a page ([`Self::band_mib`]).
    pub(crate) fn band_bytes(&self) -> u64 {
        self.band_mib
            .map_or(pdf_raster::DEFAULT_BAND_BYTES, |mib| mib << 20)
    }

    /// Which pages `--invert` turns.
    pub(crate) fn invert_when(&self) -> InvertWhen {
        self.invert.unwrap_or(InvertWhen::Never)
    }

    /// The depth of the rendered pages' samples.
    pub(crate) fn depth(&self) -> pdf_raster::SampleDepth {
        self.depth.into()
    }

    /// Effective horizontal DPI.
    pub(crate) fn x_dpi(&self) -> f64 {
        self.resolution_x.or(self.resolution).unwrap_or(150.0)
    }

    /// Effective vertical DPI.
    pub(crate) fn y_dpi(&self) -> f64 {
        self.resolution_y.or(self.resolution).unwrap_or(150.0)
    }

    /// The resolution `(x, y)` recorded in output files that carry one:
    /// the resolution flags, also under the `--scale-to` flags.
    pub(crate) fn dpi(&self) -> (f64, f64) {
        (self.x_dpi(), self.y_dpi())
    }

    /// Device pixels per point `(x, y)` for a page of `width_pts × height_pts`
    /// (after `/Rotate`): from `--scale-to` when given, else from
    /// `--scale-to-x` / `--scale-to-y`, an axis given neither taking the
    /// other's scale when only one is set, and otherwise from the resolution
    /// flags.  A fax page takes its scales from [`crate::fax::layout`].
    pub(crate) fn scales(&self, (width_pts, height_pts): (f64, f64)) -> (f64, f64) {
        if let Some(n) = self.scale_to {
            let s = f64::from(n) / width_pts.max(height_pts);
            return (s, s);
        }
        let x = self.scale_to_x.map(|n| f64::from(n) / width_pts);
        let y = self.scale_to_y.map(|n| f64::from(n) / height_pts);
        match (x, y) {
            (Some(x), Some(y)) => (x, y),
            (Some(x), None) => (x, x),
            (None, Some(y)) => (y, y),
            (None, None) => {
                let (x, y) = self.page_dpi((width_pts, height_pts));
                (x / 72.0, y / 72.0)
            }
        }
    }

    /// The scales [`Self::scales`] gives every page, when they do not
    /// depend on the page: `None` under the `--scale-to` flags, a fax
    /// layout, or a per-page print resolution.
    #[cfg(feature = "cache")]
    pub(crate) fn uniform_scales(&self) -> Option<(f64, f64)> {
        let per_page = self.faxing()
            || self.print_resolution_per_page
            || self.scale_to.is_some()
            || self.scale_to_x.is_some()
            || self.scale_to_y.is_some();
        (!per_page).then(|| (self.x_dpi() / 72.0, self.y_dpi() / 72.0))
    }

    /// The area `-x`/`-y`/`-W`/`-H` select on a page of `page_w × page_h`
    /// pixels, or `None` when none of them is given.  A width or height
    /// reaching past the page is clipped to it.
    ///
    /// Returns `Err(message)` when the offset lies outside the page.
    pub(crate) fn crop_region(
        &self,
        (page_w, page_h): (u32, u32),
    ) -> Result<Option<RenderRegion>, String> {
        if self.crop_x.is_none()
            && self.crop_y.is_none()
            && self.crop_w.is_none()
            && self.crop_h.is_none()
        {
            return Ok(None);
        }
        let (x, y) = (self.crop_x.unwrap_or(0), self.crop_y.unwrap_or(0));
        if x >= page_w || y >= page_h {
            return Err(format!(
                "crop offset ({x}, {y}) lies outside the {page_w}×{page_h} page"
            ));
        }
        let width = self.crop_w.map_or(page_w - x, |w| w.min(page_w - x));
        let height = self.crop_h.map_or(page_h - y, |h| h.min(page_h - y));
        Ok(Some(RenderRegion::new(x, y, width, height)))
    }

    /// Validate the output-format flags: refuse the formats that are not
    /// implemented and the mutually exclusive combinations.
    ///
    /// Returns `Err` with a human-readable message naming the alternative.
    /// Call this once at startup, before any page is rendered.
    pub(crate) fn validate_format_flags(&self) -> Result<(), String> {
        self.validate_depth_flags()?;
        self.validate_cmyk_flags()?;
        self.validate_transparent_flags()?;
        self.validate_output_profile_flags()?;
        self.validate_fax_flags()?;
        self.validate_print_flags()?;
        self.validate_deskew_flags()?;
        self.validate_threshold_flags()?;
        if self.mono && self.gray {
            return Err("--mono and --gray are mutually exclusive".to_owned());
        }
        if [self.png, self.jpeg_output(), self.tiff]
            .iter()
            .filter(|&&f| f)
            .count()
            > 1
        {
            return Err("--png, --jpeg (--jpegcmyk) and --tiff are mutually exclusive".to_owned());
        }
        if self.dither != DitherArg::None && !(self.mono || self.faxing()) {
            return Err("--dither applies to --mono and --fax output only".to_owned());
        }
        if self.jpegopt.is_some() && !self.jpeg_output() {
            return Err("--jpegopt applies to --jpeg and --jpegcmyk output only".to_owned());
        }
        match self.tiffcompression {
            Some(_) if !(self.tiff || self.faxing()) => {
                Err("--tiffcompression applies to --tiff output only".to_owned())
            }
            Some(TiffMethod::Jpeg) => Err(
                "JPEG-compressed TIFF is not written; use --jpeg for JPEG files, \
                 or --tiffcompression deflate for compact lossless TIFF"
                    .to_owned(),
            ),
            Some(method @ (TiffMethod::G3 | TiffMethod::G3TwoD | TiffMethod::G4))
                if !(self.mono || self.faxing()) =>
            {
                Err(format!(
                    "--tiffcompression {} codes bilevel pages; add --mono",
                    method.name()
                ))
            }
            _ => Ok(()),
        }
    }

    /// Whether pages are rendered as CMYK: `--cmyk`, or `--jpegcmyk`,
    /// which implies it.
    pub(crate) const fn cmyk(&self) -> bool {
        self.cmyk || self.jpegcmyk
    }

    /// Whether pages are written as JPEG: `--jpeg`, or `--jpegcmyk`, which
    /// implies it.
    pub(crate) const fn jpeg_output(&self) -> bool {
        self.jpeg || self.jpegcmyk
    }

    /// Refuse CMYK pages with the outputs that hold no CMYK and the flags
    /// that work on RGB, gray or bilevel pages.
    fn validate_cmyk_flags(&self) -> Result<(), String> {
        if !self.cmyk() {
            return Ok(());
        }
        let flag = if self.cmyk { "--cmyk" } else { "--jpegcmyk" };
        let colour = "it works on RGB or gray pages; drop it for CMYK pages";
        let camera = "the camera cleanup works on gray pages; drop it for CMYK pages";
        let refused = [
            (
                self.png,
                "--png",
                "PNG holds no CMYK; write CMYK pages with --tiff or --jpeg",
            ),
            (
                self.printing(),
                "--print",
                "the print spool takes RGB or gray pages; drop --cmyk to print",
            ),
            (
                self.faxing(),
                "--fax",
                "a fax holds bilevel pages; drop --cmyk",
            ),
            (self.mono, "--mono", "it writes bilevel pages"),
            (self.gray, "--gray", "it writes gray pages"),
            (
                self.dither != DitherArg::None,
                "--dither",
                "it writes bilevel pages",
            ),
            (
                self.transparent,
                "--transparent",
                "a CMYK page is opaque, over white paper",
            ),
            (self.invert.is_some(), "--invert", colour),
            (self.crop, "--crop", camera),
            (self.rectify, "--rectify", camera),
            (self.split_spreads, "--split-spreads", camera),
            (self.dewarp, "--dewarp", camera),
            (self.deskew, "--deskew", "it straightens gray pages"),
            (self.clean_borders, "--clean-borders", colour),
            (self.clean_background, "--clean-background", colour),
            (
                self.gpu_pages,
                "--gpu-pages",
                "the GPU page raster composites RGB pages; drop --gpu-pages for CMYK pages",
            ),
            (
                self.is_comic_input(),
                "comic-archive input",
                "its pages are decoded RGB or gray images",
            ),
        ];
        if let Some((_, what, why)) = refused.iter().find(|(set, _, _)| *set) {
            return Err(format!("{flag} is refused with {what}: {why}"));
        }
        if !(self.tiff || self.jpeg_output()) {
            return Err(format!(
                "{flag} needs an output that holds CMYK: PPM has none; add --tiff or --jpeg"
            ));
        }
        Ok(())
    }

    /// Refuse `--depth 16` with the flags and input whose pages are 8-bit
    /// or bilevel.
    fn validate_depth_flags(&self) -> Result<(), String> {
        if self.depth == DepthArg::Eight {
            return Ok(());
        }
        let jpeg = "baseline JPEG is 8-bit; use --png or --tiff for 16-bit pages";
        let refused = [
            (
                self.mono,
                "--mono",
                "it writes 1-bit pages; use --gray for 16-bit gray",
            ),
            (
                self.dither != DitherArg::None,
                "--dither",
                "it renders 1-bit pages; drop it for 16-bit pages",
            ),
            (
                self.faxing(),
                "--fax",
                "a fax holds bilevel G3/G4 pages; drop --depth 16",
            ),
            (self.jpeg, "--jpeg", jpeg),
            (self.jpegopt.is_some(), "--jpegopt", jpeg),
            (self.jpegcmyk, "--jpegcmyk", jpeg),
            (
                self.gpu_pages,
                "--gpu-pages",
                "the GPU page raster composites 8-bit pages; drop --gpu-pages for 16-bit pages",
            ),
        ];
        match refused.iter().find(|(set, _, _)| *set) {
            Some((_, what, why)) => Err(format!("--depth 16 is refused with {what}: {why}")),
            None => Ok(()),
        }
    }

    /// Refuse `--transparent` with an output that carries no alpha channel
    /// and with the flags that work on opaque pages.
    fn validate_transparent_flags(&self) -> Result<(), String> {
        if !self.transparent {
            return Ok(());
        }
        let opaque = [
            (self.gray, "--gray"),
            (self.mono, "--mono"),
            (self.invert.is_some(), "--invert"),
            (self.crop, "--crop"),
            (self.rectify, "--rectify"),
            (self.dewarp, "--dewarp"),
            (self.split_spreads, "--split-spreads"),
            (self.deskew, "--deskew"),
            (self.clean_borders, "--clean-borders"),
            (self.clean_background, "--clean-background"),
            (self.faxing(), "--fax"),
            (self.printing(), "--print"),
        ];
        if let Some((_, flag)) = opaque.iter().find(|(set, _)| *set) {
            return Err(format!(
                "--transparent writes colour pages with an alpha channel; drop {flag}"
            ));
        }
        if self.is_comic_input() {
            return Err(
                "--transparent renders PDF input; a comic archive's pages are opaque images"
                    .to_owned(),
            );
        }
        if self.png || self.tiff {
            return Ok(());
        }
        let format = if self.jpeg_output() { "JPEG" } else { "PPM" };
        Err(format!(
            "--transparent needs an output with an alpha channel: {format} has none; \
             add --png or --tiff"
        ))
    }

    /// Refuse `--intent` and `--bpc` without `--output-profile` or
    /// `--proof`, `--bpc` under absolute colorimetry, and `--output-profile`
    /// and `--proof` with the bilevel outputs and comic input, `--proof`
    /// with CMYK and transparent output.
    fn validate_output_profile_flags(&self) -> Result<(), String> {
        self.validate_proof_flags()?;
        if self.working_space == Some(WorkingSpaceArg::Wide) {
            if self.output_profile.is_none() {
                return Err(
                    "--working-space wide carries its gamut to the conversion to the output \
                     profile: add --output-profile"
                        .to_owned(),
                );
            }
            if self.depth() == pdf_raster::SampleDepth::Eight {
                return Err(
                    "--working-space wide is composited at 16 bits, at which its levels are as \
                     fine as sRGB's at 8: add --depth 16"
                        .to_owned(),
                );
            }
        }
        if self.output_profile.is_none() && self.proof.is_none() {
            return match (self.intent.is_some(), self.bpc) {
                (true, _) => {
                    Err("--intent applies to --output-profile and --proof only".to_owned())
                }
                (_, true) => Err("--bpc applies to --output-profile and --proof only".to_owned()),
                _ => Ok(()),
            };
        }
        if self.bpc && self.intent == Some(IntentArg::Absolute) {
            return Err(
                "--bpc does not apply to --intent absolute, which reproduces measured colour: \
                 drop --bpc or choose another intent"
                    .to_owned(),
            );
        }
        let flag = if self.output_profile.is_some() {
            "--output-profile"
        } else {
            "--proof"
        };
        let bilevel = format!("bilevel output carries no colour to convert; drop {flag}");
        let bilevel = bilevel.as_str();
        let refused = [
            (self.mono, "--mono", bilevel),
            (self.dither != DitherArg::None, "--dither", bilevel),
            (self.faxing(), "--fax", bilevel),
            (
                self.is_comic_input(),
                "comic-archive input",
                "its pages are decoded images, not rendered PDF pages",
            ),
        ];
        match refused.iter().find(|(set, _, _)| *set) {
            Some((_, what, why)) => Err(format!("{flag} is refused with {what}: {why}")),
            None => Ok(()),
        }
    }

    /// Refuse `--proof` with CMYK and transparent output.
    fn validate_proof_flags(&self) -> Result<(), String> {
        if self.proof.is_none() {
            return Ok(());
        }
        if self.cmyk() {
            let flag = if self.cmyk { "--cmyk" } else { "--jpegcmyk" };
            return Err(format!(
                "--proof shows the press on a display and writes RGB or gray pages, and {flag} \
                 writes CMYK: drop one (for the press's own ink, use {flag} with the output \
                 intent's profile as --output-profile)"
            ));
        }
        if self.transparent {
            return Err(
                "--proof paints the pages on the press's opaque paper: drop --transparent"
                    .to_owned(),
            );
        }
        Ok(())
    }

    /// The session's output profile, read from `--output-profile` and
    /// checked against the output: an RGB profile refuses `--gray`, a
    /// gray one `--transparent`; with `--intent`, its conversion is
    /// compiled.  `None` without the flag.
    fn output_profile(&self) -> Result<Option<std::sync::Arc<pdf_raster::OutputProfile>>, String> {
        let Some(path) = &self.output_profile else {
            return Ok(None);
        };
        let shown = path.display();
        let bytes = std::fs::read(path)
            .map_err(|e| format!("cannot read the output profile {shown}: {e}"))?;
        let profile = pdf_raster::OutputProfile::from_bytes(bytes)
            .map_err(|e| format!("--output-profile {shown}: {e}"))?;
        match profile.space() {
            pdf_raster::OutputSpace::Cmyk if self.proof.is_some() => {
                return Err(format!(
                    "--output-profile {shown} is a CMYK profile and --proof shows the press on a \
                     display: give an RGB or gray display profile, or drop it for sRGB"
                ));
            }
            pdf_raster::OutputSpace::Cmyk if !self.cmyk() => {
                return Err(format!(
                    "--output-profile {shown} is a CMYK profile, which separates CMYK pages: \
                     add --cmyk with --tiff or --jpeg, or --jpegcmyk"
                ));
            }
            pdf_raster::OutputSpace::Rgb | pdf_raster::OutputSpace::Gray if self.cmyk() => {
                return Err(format!(
                    "--output-profile {shown} is an RGB or gray profile and --cmyk writes CMYK \
                     pages: give a CMYK output profile, or drop --cmyk"
                ));
            }
            pdf_raster::OutputSpace::Rgb if self.gray => {
                return Err(format!(
                    "--output-profile {shown} is an RGB profile and --gray writes gray pages: \
                     give a gray output profile, or drop --gray"
                ));
            }
            pdf_raster::OutputSpace::Gray if self.transparent => {
                return Err(format!(
                    "--output-profile {shown} is a gray profile and --transparent writes colour \
                     pages with alpha: give an RGB output profile, or drop --transparent"
                ));
            }
            _ => {}
        }
        if let Some(intent) = self.intent
            && self.proof.is_none()
        {
            profile
                .prepare(self.resolved_working_space(), intent.into(), self.bpc)
                .map_err(|e| format!("--output-profile {shown}: {e}"))?;
        }
        Ok(Some(std::sync::Arc::new(profile)))
    }

    /// The working space the pages are composited in, as the session
    /// resolves it for a page that is not an overprint preview:
    /// `--working-space`, else the wide space at --depth 16 under
    /// --output-profile, else sRGB.
    fn resolved_working_space(&self) -> pdf_raster::WorkingSpace {
        match self.working_space {
            Some(space) => space.into(),
            None if self.output_profile.is_some()
                && self.depth() == pdf_raster::SampleDepth::Sixteen =>
            {
                pdf_raster::WorkingSpace::Wide
            }
            None => pdf_raster::WorkingSpace::Srgb,
        }
    }

    /// The warning of an output that cannot carry the output profile:
    /// PPM and PGM files, whose pixels are still converted.
    pub(crate) fn untagged_output_warning(&self) -> Option<String> {
        let netpbm =
            !(self.png || self.jpeg_output() || self.tiff || self.printing() || self.faxing());
        (self.output_profile.is_some() && netpbm).then(|| {
            "PPM and PGM files cannot carry an ICC profile: the pages hold --output-profile's \
             device values untagged; add --png or --tiff to embed the profile"
                .to_owned()
        })
    }

    /// The input path, `-` for standard input; empty only under
    /// `--list-printers`, which clap admits without an input and which
    /// renders nothing.
    pub(crate) fn input(&self) -> &str {
        self.input.as_deref().unwrap_or("")
    }

    /// Whether the input is a comic archive, by its extension.
    pub(crate) fn is_comic_input(&self) -> bool {
        std::path::Path::new(self.input())
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| {
                ["cbz", "cb7", "cbt", "cbr"]
                    .iter()
                    .any(|comic| e.eq_ignore_ascii_case(comic))
            })
    }

    /// Whether pages are written in luma: `--gray`, `--mono`, or a gray
    /// output profile.
    pub(crate) fn luma_output(&self) -> bool {
        self.gray
            || self.mono
            || self
                .output_icc
                .as_ref()
                .is_some_and(|icc| icc.channels() == 1)
    }

    /// Refuse `--deskew` with colour output: deskew straightens gray pages.
    fn validate_deskew_flags(&self) -> Result<(), String> {
        if self.deskew && !(self.luma_output() || self.faxing()) {
            return Err(
                "--deskew applies to gray pages only: add --gray, --mono or --fax, \
                        or drop --deskew"
                    .to_owned(),
            );
        }
        Ok(())
    }

    /// Refuse `--threshold sauvola` without bilevel output and with
    /// `--dither halftone`, which chooses black by its screen.
    fn validate_threshold_flags(&self) -> Result<(), String> {
        if self.threshold == ThresholdArg::Fixed {
            return Ok(());
        }
        if !(self.mono || self.faxing()) {
            return Err(
                "--threshold sauvola applies to bilevel output only: add --mono or --fax, \
                 or drop --threshold"
                    .to_owned(),
            );
        }
        if self.dither == DitherArg::Halftone {
            return Err(
                "--threshold sauvola and --dither halftone both choose which pixels are black; \
                 drop one"
                    .to_owned(),
            );
        }
        Ok(())
    }

    /// The resolution, in dots per inch, that sizes the OCR-preparation
    /// windows of a `width × height` page: the smaller of the output
    /// resolutions, or for comic input, whose pages carry none, the long
    /// side over 11 inches.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the value is clamped to [1, u32::MAX] before the cast"
    )]
    pub(crate) fn ocr_dpi(&self, width: u32, height: u32) -> u32 {
        if self.is_comic_input() {
            return (width.max(height) / 11).max(1);
        }
        self.x_dpi()
            .min(self.y_dpi())
            .round()
            .clamp(1.0, f64::from(u32::MAX)) as u32
    }

    /// Whether a fax document is written: `--fax`, or `--fax-to`, which
    /// implies it.
    pub(crate) const fn faxing(&self) -> bool {
        self.fax || self.fax_to.is_some()
    }

    /// The output prefix; empty under `--print`, which writes no files.
    pub(crate) fn prefix(&self) -> &str {
        self.output_prefix.as_deref().unwrap_or("")
    }

    /// Whether the pages are printed: `--print`, or `--printer`, which
    /// implies it.
    pub(crate) const fn printing(&self) -> bool {
        self.print || self.printer.is_some()
    }

    /// Refuse the flags `--print` excludes, and the print-only flags
    /// without it.
    fn validate_print_flags(&self) -> Result<(), String> {
        if !self.printing() {
            return match (self.copies.is_some(), self.duplex.is_some()) {
                (true, _) => Err("--copies applies to --print only".to_owned()),
                (_, true) => Err("--duplex applies to --print only".to_owned()),
                _ => Ok(()),
            };
        }
        if cfg!(windows) {
            return Err(
                "--print submits through CUPS's lp, which Windows does not have; render the \
                 pages to files and print them from an application"
                    .to_owned(),
            );
        }
        if let Some(prefix) = &self.output_prefix {
            return Err(format!(
                "--print writes no files; drop the output prefix {prefix:?}"
            ));
        }
        if self.is_comic_input() {
            return Err(
                "--print renders PDF input; a comic archive's pages have no page size".to_owned(),
            );
        }
        if self.faxing() {
            return Err("--print and --fax are mutually exclusive".to_owned());
        }
        let excluded = [
            (self.png, "--png"),
            (self.jpeg_output(), "--jpeg"),
            (self.tiff, "--tiff"),
            (self.single_file, "--singlefile"),
            (self.ram || self.ram_path.is_some(), "--ram / --ram-path"),
            (self.no_ram, "--no-ram"),
            (self.separator != '-', "--sep"),
            (self.force_num_digits.is_some(), "--forcenum"),
            (self.scale_to.is_some(), "--scale-to"),
            (self.scale_to_x.is_some(), "--scale-to-x"),
            (self.scale_to_y.is_some(), "--scale-to-y"),
        ];
        if let Some((_, flag)) = excluded.iter().find(|(set, _)| *set) {
            return Err(format!(
                "--print sends page images measured by their resolution; drop {flag}"
            ));
        }
        Ok(())
    }

    /// Expand `--print`: implied by `--printer`, at the resolution
    /// [`crate::print::default_dpi`] picks for each page unless a
    /// resolution flag is given ([`crate::print::PRINT_DPI`] as the
    /// resolution of the run, for what does not depend on the page).  Call
    /// after [`Self::validate_format_flags`].
    pub(crate) const fn apply_print_preset(&mut self) {
        if !self.printing() {
            return;
        }
        self.print = true;
        if self.resolution.is_none() && self.resolution_x.is_none() && self.resolution_y.is_none() {
            self.resolution = Some(crate::print::PRINT_DPI);
            self.print_resolution_per_page = true;
        }
    }

    /// The resolution `(x, y)` of a page of `width_pts × height_pts`: the
    /// print default for its size under `--print` without a resolution
    /// flag, else the run's ([`Self::dpi`]).
    pub(crate) fn page_dpi(&self, page_pts: (f64, f64)) -> (f64, f64) {
        if self.print_resolution_per_page {
            let layout = pdf_raster::CanvasLayout::new(self.depth(), false);
            let dpi = crate::print::default_dpi(page_pts, layout.bytes_per_px());
            (dpi, dpi)
        } else {
            self.dpi()
        }
    }

    /// Refuse the flags `--fax` fixes or excludes, and the fax-only flags
    /// without it.
    fn validate_fax_flags(&self) -> Result<(), String> {
        if !self.faxing() {
            return if self.fax_resolution.is_some() {
                Err("--fax-resolution applies to --fax output only".to_owned())
            } else {
                Ok(())
            };
        }
        if self.is_comic_input() {
            return Err(
                "--fax renders PDF input; a comic archive's pages have no page size".to_owned(),
            );
        }
        if self.ram || self.ram_path.is_some() {
            return Err(
                "--fax writes its document after the last page; RAM output serves page files \
                 read during the run, so drop --ram / --ram-path"
                    .to_owned(),
            );
        }
        if cfg!(windows) && self.fax_to.is_some() {
            return Err(
                "--fax-to submits through lp or sendfax, which Windows has neither of; write the \
                 document with --fax and send it with Windows Fax and Scan"
                    .to_owned(),
            );
        }
        let fixed = [
            (self.png, "--png"),
            (self.jpeg_output(), "--jpeg"),
            (self.gray, "--gray"),
            (self.resolution.is_some(), "-r"),
            (self.resolution_x.is_some(), "--rx"),
            (self.resolution_y.is_some(), "--ry"),
            (self.scale_to.is_some(), "--scale-to"),
            (self.scale_to_x.is_some(), "--scale-to-x"),
            (self.scale_to_y.is_some(), "--scale-to-y"),
            (self.crop_x.is_some(), "-x"),
            (self.crop_y.is_some(), "-y"),
            (self.crop_w.is_some(), "-W"),
            (self.crop_h.is_some(), "-H"),
            (self.crop, "--crop"),
            (self.rectify, "--rectify"),
            (self.dewarp, "--dewarp"),
            (self.split_spreads, "--split-spreads"),
        ];
        if let Some((_, flag)) = fixed.iter().find(|(set, _)| *set) {
            return Err(format!(
                "--fax fixes the format, geometry and resolution of its pages; drop {flag} \
                 (--fax-resolution chooses fine or standard)"
            ));
        }
        match self.tiffcompression {
            None | Some(TiffMethod::G3 | TiffMethod::G3TwoD | TiffMethod::G4) => Ok(()),
            Some(method) => Err(format!(
                "--fax codes its pages g3, g3-2d or g4, not {}",
                method.name()
            )),
        }
    }

    /// Expand `--fax` into the flags it implies: bilevel TIFF, Group 3
    /// one-dimensional coding unless another fax coding is chosen, the fax
    /// resolution, and `--invert=auto` unless `--invert` is given.  Call
    /// after [`Self::validate_format_flags`].
    pub(crate) fn apply_fax_preset(&mut self) {
        if !self.faxing() {
            return;
        }
        self.fax = true;
        self.tiff = true;
        self.mono = true;
        self.tiffcompression = Some(self.tiffcompression.unwrap_or(TiffMethod::G3));
        self.invert = Some(self.invert.unwrap_or(InvertWhen::Auto));
        self.resolution_x = Some(crate::fax::FAX_X_DPI);
        self.resolution_y = Some(self.fax_resolution.unwrap_or(FaxResolution::Fine).y_dpi());
    }

    /// The TIFF compression `--tiffcompression` selects at the vertical
    /// resolution.
    pub(crate) fn tiff_compression(&self) -> encode::TiffCompression {
        self.tiffcompression
            .unwrap_or(TiffMethod::None)
            .compression(self.y_dpi())
    }

    /// Resolved output format.
    pub(crate) fn output_format(&self) -> OutputFormat {
        if self.png {
            OutputFormat::Png
        } else if self.jpeg_output() {
            OutputFormat::Jpeg(self.jpegopt.unwrap_or_default())
        } else if self.tiff {
            OutputFormat::Tiff(self.tiff_compression())
        } else {
            OutputFormat::Ppm
        }
    }
}

/// Largest `--band-mib`: 64 GiB.
const MAX_BAND_MIB: u64 = 1 << 16;

/// Largest `--max-page-mib`: the 16-bit RGBA bitmap of a page whose sides
/// are both [`pdf_raster::MAX_PX_SIDE`].
const MAX_PAGE_MIB: u64 = 1 << 31;

/// The ITU-T T.4 period of two-dimensional Group 3 coding at fine
/// resolution (at least 150 lines per inch) and at standard resolution.
const K_FINE: NonZeroU32 = NonZeroU32::new(4).expect("4 is not zero");
const K_STANDARD: NonZeroU32 = NonZeroU32::new(2).expect("2 is not zero");

/// `--tiffcompression` argument value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum TiffMethod {
    /// Uncompressed.
    None,
    /// `PackBits` run-length coding.
    #[value(name = "packbits")]
    PackBits,
    /// LZW.
    Lzw,
    /// Deflate (zlib).
    Deflate,
    /// CCITT Group 3, one-dimensional (MH); needs --mono.
    #[value(name = "g3")]
    G3,
    /// CCITT Group 3, two-dimensional (MR); needs --mono.
    #[value(name = "g3-2d")]
    G3TwoD,
    /// CCITT Group 4 (MMR); needs --mono.
    #[value(name = "g4")]
    G4,
    /// JPEG-in-TIFF: accepted for a clear refusal, not written.
    #[value(hide = true)]
    Jpeg,
}

impl TiffMethod {
    /// The name the method is given on the command line.
    fn name(self) -> String {
        clap::ValueEnum::to_possible_value(&self)
            .map_or_else(String::new, |v| v.get_name().to_owned())
    }

    /// The writer's compression at `y_dpi` vertical resolution.  Group 3
    /// two-dimensional coding takes the period [`K_FINE`] or
    /// [`K_STANDARD`].
    fn compression(self, y_dpi: f64) -> encode::TiffCompression {
        match self {
            Self::None | Self::Jpeg => encode::TiffCompression::None,
            Self::PackBits => encode::TiffCompression::PackBits,
            Self::Lzw => encode::TiffCompression::Lzw,
            Self::Deflate => encode::TiffCompression::Deflate,
            Self::G3 => encode::TiffCompression::Group3 { k: None },
            Self::G3TwoD => encode::TiffCompression::Group3 {
                k: Some(if y_dpi >= 150.0 { K_FINE } else { K_STANDARD }),
            },
            Self::G4 => encode::TiffCompression::Group4,
        }
    }
}

/// Output image format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OutputFormat {
    /// Netpbm: PPM, PGM under `--gray`, PBM under `--mono` (default).
    Ppm,
    /// PNG.
    Png,
    /// JPEG, gray under `--gray` and `--mono`.
    Jpeg(encode::JpegOptions),
    /// TIFF, gray under `--gray` and bilevel under `--mono`.
    Tiff(encode::TiffCompression),
}

impl OutputFormat {
    /// File extension for this format, taking `--gray` / `--mono` into account.
    ///
    /// - PPM + mono → `.pbm` (P4 binary Netpbm, 1-bit)
    /// - PPM + gray → `.pgm` (P5 grayscale Netpbm, 8-bit)
    /// - PNG + gray/mono → `.png` (grayscale PNG)
    /// - JPEG → `.jpg`, gray or colour
    /// - TIFF → `.tif`, colour, gray or bilevel
    /// - All other combinations use the format's natural extension.
    pub(crate) const fn extension_with_mode(self, gray: bool, mono: bool) -> &'static str {
        match (self, mono, gray) {
            (Self::Ppm, true, _) => "pbm",
            (Self::Ppm, false, true) => "pgm",
            (Self::Ppm, false, false) => "ppm",
            (Self::Png, _, _) => "png",
            (Self::Jpeg(_), _, _) => "jpg",
            (Self::Tiff(_), _, _) => "tif",
        }
    }
}

// ── Backend selection ─────────────────────────────────────────────────────────

/// `--backend` argument value parsed by clap.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum BackendArg {
    /// Auto-select: Vulkan if compiled in, present and one of its kernels is
    /// enabled, else CUDA, else CPU.
    /// Silent fallback at every step.  This is the default; when omitted,
    /// `PDF_RASTER_BACKEND` is also consulted.
    Auto,
    /// Skip all GPU init; use CPU paths only.
    Cpu,
    /// Require CUDA — error loudly if CUDA / nvJPEG is unavailable.
    Cuda,
    /// Require VA-API JPEG — error loudly if the DRM device cannot be opened.
    #[cfg(feature = "vaapi")]
    Vaapi,
    /// Require the Vulkan compute backend — error loudly if Vulkan
    /// initialisation fails.  In a build with the `cache` feature the
    /// device-resident image cache runs on the Vulkan device.
    Vulkan,
}

impl Args {
    /// The backend policy: `--backend` when given, else the one
    /// `PDF_RASTER_BACKEND` names, else [`BackendPolicy::Auto`].
    pub(crate) fn policy(&self) -> BackendPolicy {
        match self.backend {
            Some(BackendArg::Auto) => BackendPolicy::Auto,
            Some(BackendArg::Cpu) => BackendPolicy::CpuOnly,
            Some(BackendArg::Cuda) => BackendPolicy::ForceCuda,
            #[cfg(feature = "vaapi")]
            Some(BackendArg::Vaapi) => BackendPolicy::ForceVaapi,
            Some(BackendArg::Vulkan) => BackendPolicy::ForceVulkan,
            None => BackendPolicy::from_env(),
        }
    }

    /// Build a [`SessionConfig`] from the `--backend` / `--vaapi-device` /
    /// `--vulkan-device` flags.
    ///
    /// Rejects `--vaapi-device` when combined with `--backend cpu`,
    /// `--backend cuda` or `--backend vulkan`, and `--vulkan-device` with
    /// any backend but `auto` and `vulkan` — those modes never touch the
    /// device in question, so allowing the combination would silently
    /// ignore a user expectation.  `PDF_RASTER_VULKAN_DEVICE` is consulted
    /// only when `--vulkan-device` is absent and is never rejected.
    /// Rejects `--gpu-pages` with a backend other than `auto` and `vulkan`,
    /// with `--aaVector no`, and in a binary without the `vulkan` feature
    /// ([`SessionConfig::check_gpu_pages`]), so the refusal precedes any
    /// decryption prompt and covers comic input, which opens no session.
    pub(crate) fn session_config(&self) -> Result<SessionConfig, String> {
        // Precedence: explicit `--backend <X>` wins; otherwise consult
        // `PDF_RASTER_BACKEND`; otherwise default to Auto.  Distinguishing
        // "user passed --backend auto" from "user passed nothing" requires
        // `Option<BackendArg>` rather than a clap `default_value` — with
        // a default value the two cases are indistinguishable and the env
        // var would never get a chance to override.
        let policy = self.policy();

        if self.vaapi_device.is_some()
            && matches!(
                policy,
                BackendPolicy::CpuOnly | BackendPolicy::ForceCuda | BackendPolicy::ForceVulkan
            )
        {
            return Err(format!(
                "--vaapi-device has no effect with --backend {}.\n\
                 VA-API is only used with --backend auto or --backend vaapi.",
                policy.name()
            ));
        }

        if self.vulkan_device.is_some()
            && !matches!(policy, BackendPolicy::Auto | BackendPolicy::ForceVulkan)
        {
            return Err(format!(
                "--vulkan-device has no effect with --backend {}.\n\
                 Vulkan is only used with --backend auto or --backend vulkan.",
                policy.name()
            ));
        }

        let mut config = SessionConfig::with_policy(policy);
        if let Some(node) = &self.vaapi_device {
            config.vaapi_device.clone_from(node);
        }
        config.vulkan_device = self
            .vulkan_device
            .clone()
            .or_else(SessionConfig::vulkan_device_from_env);
        config.gpu_pages = self.gpu_pages;
        config.font_dirs = SessionConfig::font_dirs_from_env();
        config.system_fonts = !self.no_system_fonts;
        config.strict = self.strict;
        config.purpose = self.purpose.map_or_else(
            || {
                if self.printing() {
                    RenderPurpose::Print
                } else {
                    RenderPurpose::View
                }
            },
            Into::into,
        );
        config.annotations = !self.hide_annotations;
        config.text_antialias = self.antialias == AaFlag::Yes;
        config.vector_antialias = self.vector_antialias == AaFlag::Yes;
        config.overprint = self.overprint;
        config.page_box = self.page_box.into();
        config.rotation = self.rotate.unwrap_or_default();
        if let Some(mib) = self.max_page_mib {
            config.max_page_bytes = mib << 20;
        }
        config.band_bytes = self.band_bytes();
        config.output_profile = self.output_profile()?;
        config.proof = self.proof.map(Into::into);
        config.working_space = self.working_space.map(Into::into);
        #[cfg(feature = "cache")]
        {
            config.prefetch = self.prefetch;
            config.prefetch_scale = self.uniform_scales();
        }
        config.check_gpu_pages().map_err(|e| e.to_string())?;
        Ok(config)
    }

    /// Build the filtered, clamped list of 1-based page numbers to render.
    ///
    /// Returns `Err(message)` if no pages fall within the requested range so the
    /// caller can handle display and exit uniformly.  Clamping warnings are
    /// returned alongside the page list so the caller decides how to display them.
    ///
    /// Precondition: `total >= 1`, validated by the caller before this is invoked.
    pub(crate) fn build_page_list(&self, total: i32) -> Result<(Vec<i32>, Vec<String>), String> {
        let requested_first = self.first_page;
        let requested_last = self.last_page.unwrap_or(total);

        let first = requested_first.max(1);
        let last = requested_last.min(total);

        let mut warnings = Vec::new();
        if requested_first < 1 {
            warnings.push(format!("first page {requested_first} < 1; clamped to 1"));
        }
        if requested_last > total {
            warnings.push(format!(
                "last page {requested_last} exceeds document length ({total}); clamped to {total}"
            ));
        }

        if first > last {
            return Err(format!(
                "first page ({first}) is after last page ({last}); nothing to render"
            ));
        }

        let pages: Vec<i32> = (first..=last)
            .filter(|&p| {
                // odd_only and even_only are mutually exclusive; caller enforces this.
                if self.odd_only {
                    p % 2 == 1
                } else if self.even_only {
                    p % 2 == 0
                } else {
                    true
                }
            })
            .take(if self.single_file { 1 } else { usize::MAX })
            .collect();

        if pages.is_empty() {
            return Err("no pages selected by the current filter combination".to_owned());
        }

        Ok((pages, warnings))
    }
}

/// `flags` parsed between the program name and the operands `in.pdf out`.
#[cfg(test)]
pub(crate) fn args_with(flags: &[&str]) -> Args {
    let mut argv = vec!["rrocket"];
    argv.extend_from_slice(flags);
    argv.extend(["in.pdf", "out"]);
    Args::parse_from(argv)
}

#[cfg(test)]
mod page_list_tests {
    use super::*;

    fn base_args() -> Args {
        args_with(&[])
    }

    #[test]
    fn all_pages_when_no_filter() {
        let args = base_args();
        let (pages, warnings) = args.build_page_list(3).unwrap();
        assert_eq!(pages, vec![1, 2, 3]);
        assert_eq!(warnings, Vec::<String>::new());
    }

    #[test]
    fn clamps_first_page_below_one() {
        let mut args = base_args();
        args.first_page = -5;
        let (pages, warnings) = args.build_page_list(3).unwrap();
        assert_eq!(pages[0], 1);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("clamped to 1"));
    }

    #[test]
    fn clamps_last_page_beyond_total() {
        let mut args = base_args();
        args.last_page = Some(999);
        let (pages, warnings) = args.build_page_list(3).unwrap();
        assert_eq!(*pages.last().unwrap(), 3);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("clamped to 3"));
    }

    #[test]
    fn odd_only_filter() {
        let mut args = base_args();
        args.odd_only = true;
        let (pages, _) = args.build_page_list(5).unwrap();
        assert_eq!(pages, vec![1, 3, 5]);
    }

    #[test]
    fn even_only_filter() {
        let mut args = base_args();
        args.even_only = true;
        let (pages, _) = args.build_page_list(5).unwrap();
        assert_eq!(pages, vec![2, 4]);
    }

    #[test]
    fn err_when_first_after_last() {
        let mut args = base_args();
        args.first_page = 5;
        args.last_page = Some(3);
        let result = args.build_page_list(10);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("nothing to render"));
    }

    #[test]
    fn err_when_no_pages_match_filter() {
        let mut args = base_args();
        args.first_page = 2;
        args.last_page = Some(2);
        args.odd_only = true;
        let result = args.build_page_list(5);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("no pages selected"));
    }

    #[test]
    fn single_file_takes_first_page_only() {
        let mut args = base_args();
        args.single_file = true;
        let (pages, _) = args.build_page_list(10).unwrap();
        assert_eq!(pages, vec![1]);
    }

    #[test]
    fn ocr_windows_follow_the_resolution_or_the_comic_page_size() {
        assert_eq!(args_with(&["-r", "300"]).ocr_dpi(2550, 3300), 300);
        assert_eq!(
            args_with(&["--rx", "200", "--ry", "100"]).ocr_dpi(10, 10),
            100
        );
        assert_eq!(args_with(&[]).ocr_dpi(10, 10), 150);
        let comic = Args::parse_from(["rrocket", "book.cbz", "out"]);
        assert_eq!(comic.ocr_dpi(2200, 3300), 300);
        assert_eq!(comic.ocr_dpi(5, 3), 1);
    }

    #[test]
    fn gpu_pages_conflicts_are_refused_naming_the_flags() {
        let refusal = |flags: &[&str]| {
            args_with(flags)
                .session_config()
                .expect_err("a conflicting --gpu-pages was accepted")
        };
        let conflicts: [(&[&str], &str); 3] = [
            (&["--backend", "cpu"], "--backend cpu"),
            (&["--backend", "cuda"], "--backend cuda"),
            (
                &["--backend", "vulkan", "--aaVector", "no"],
                "--aaVector no",
            ),
        ];
        for (flags, named) in conflicts {
            let msg = refusal(&[&["--gpu-pages"], flags].concat());
            let named = if cfg!(feature = "vulkan") {
                named
            } else {
                "`vulkan` feature"
            };
            assert!(msg.contains("--gpu-pages") && msg.contains(named), "{msg}");
        }
        assert_eq!(
            args_with(&["--gpu-pages", "--backend", "vulkan"])
                .session_config()
                .is_ok_and(|config| config.gpu_pages),
            cfg!(feature = "vulkan")
        );
        assert!(
            args_with(&["--backend", "cpu", "--aaVector", "no"])
                .session_config()
                .is_ok()
        );
    }

    #[test]
    fn vulkan_device_flag_reaches_session_config() {
        let config = args_with(&["--vulkan-device", "5070"])
            .session_config()
            .unwrap();
        assert_eq!(config.vulkan_device.as_deref(), Some("5070"));
        let config = args_with(&["--backend", "vulkan", "--vulkan-device", "0"])
            .session_config()
            .unwrap();
        assert_eq!(config.vulkan_device.as_deref(), Some("0"));
    }

    #[test]
    fn vaapi_device_defaults_to_probe_and_is_rejected_when_unused() {
        assert_eq!(
            base_args().session_config().unwrap().vaapi_device,
            pdf_raster::DEFAULT_VAAPI_DEVICE
        );
        // Passing the default spelling explicitly is still a user choice
        // and is rejected where VA-API is never opened.
        let err = args_with(&["--backend", "cpu", "--vaapi-device", "auto"])
            .session_config()
            .unwrap_err();
        assert!(
            err.contains("--vaapi-device has no effect with --backend cpu"),
            "{err}"
        );
    }

    #[test]
    fn vulkan_device_rejected_with_non_vulkan_backends() {
        for backend in ["cpu", "cuda"] {
            let err = args_with(&["--backend", backend, "--vulkan-device", "0"])
                .session_config()
                .unwrap_err();
            assert!(
                err.contains(&format!(
                    "--vulkan-device has no effect with --backend {backend}"
                )),
                "{err}"
            );
        }
    }

    #[test]
    fn max_archive_mib_parses_and_rejects_zero() {
        assert_eq!(args_with(&[]).max_archive_mib, None);
        assert_eq!(
            args_with(&["--max-archive-mib", "8192"]).max_archive_mib,
            Some(8192)
        );
        assert!(
            Args::try_parse_from(["rrocket", "--max-archive-mib", "0", "in.cbz", "out"]).is_err()
        );
    }

    #[test]
    fn max_page_mib_sets_the_page_budget_and_rejects_zero_and_absurd_values() {
        let config = |argv: &[&str]| args_with(argv).session_config().unwrap();
        assert_eq!(
            config(&[]).max_page_bytes,
            pdf_raster::DEFAULT_MAX_PAGE_BYTES
        );
        assert_eq!(
            config(&["--max-page-mib", "2100"]).max_page_bytes,
            2100 << 20
        );
        let refused = |mib: &str| {
            Args::try_parse_from(["rrocket", "--max-page-mib", mib, "in.pdf", "out"]).is_err()
        };
        let side = u64::from(pdf_raster::MAX_PX_SIDE);
        assert_eq!(MAX_PAGE_MIB, (side * side * 8) >> 20);
        assert!(refused("0"));
        assert!(refused(&(MAX_PAGE_MIB + 1).to_string()));
        assert!(!refused(&MAX_PAGE_MIB.to_string()));
    }

    #[test]
    fn band_mib_sizes_the_bands_and_rejects_zero_and_absurd_values() {
        let config = |argv: &[&str]| args_with(argv).session_config().unwrap();
        assert_eq!(config(&[]).band_bytes, pdf_raster::DEFAULT_BAND_BYTES);
        assert_eq!(config(&["--band-mib", "8"]).band_bytes, 8 << 20);
        let refused = |mib: &str| {
            Args::try_parse_from(["rrocket", "--band-mib", mib, "in.pdf", "out"]).is_err()
        };
        assert!(refused("0"));
        assert!(refused("65537"));
        assert!(!refused("65536"));
    }

    #[test]
    fn strict_flag_reaches_session_config() {
        let lenient = base_args().session_config().unwrap();
        assert!(!lenient.strict);
        let strict = args_with(&["--strict"]).session_config().unwrap();
        assert!(strict.strict);
    }

    #[cfg(feature = "cache")]
    #[test]
    fn the_prefetch_scale_is_the_resolution_unless_it_depends_on_the_page() {
        let scale = |argv: &[&str]| args_with(argv).session_config().unwrap().prefetch_scale;
        assert_eq!(scale(&["--rx", "144", "--ry", "36"]), Some((2.0, 0.5)));
        assert_eq!(scale(&["--scale-to", "400"]), None);
        assert_eq!(scale(&["--scale-to-y", "300"]), None);
    }

    #[test]
    fn scale_to_flags_set_the_page_scales() {
        let page = (200.0, 100.0);
        let scales = |argv: &[&str]| args_with(argv).scales(page);
        assert_eq!(scales(&["-r", "144"]), (2.0, 2.0));
        assert_eq!(scales(&["--rx", "144", "--ry", "36"]), (2.0, 0.5));
        assert_eq!(scales(&["--scale-to", "400", "-r", "300"]), (2.0, 2.0));
        assert_eq!(scales(&["--scale-to-x", "100"]), (0.5, 0.5));
        assert_eq!(scales(&["--scale-to-y", "300"]), (3.0, 3.0));
        assert_eq!(
            scales(&["--scale-to-x", "100", "--scale-to-y", "300"]),
            (0.5, 3.0)
        );
        assert_eq!(
            scales(&["--scale-to", "50", "--scale-to-x", "100"]),
            (0.25, 0.25)
        );
    }

    #[test]
    fn crop_flags_select_a_clipped_region() {
        let region = |argv: &[&str]| args_with(argv).crop_region((100, 50));
        assert_eq!(region(&[]), Ok(None));
        assert_eq!(
            region(&["-x", "10", "-y", "5", "-W", "20", "-H", "30"]),
            Ok(Some(RenderRegion::new(10, 5, 20, 30)))
        );
        assert_eq!(
            region(&["-x", "90"]),
            Ok(Some(RenderRegion::new(90, 0, 10, 50)))
        );
        assert_eq!(
            region(&["-W", "500", "-H", "7"]),
            Ok(Some(RenderRegion::new(0, 0, 100, 7)))
        );
        assert!(region(&["-y", "50"]).is_err());
    }

    #[test]
    fn hide_annotations_flag_reaches_session_config() {
        assert!(base_args().session_config().unwrap().annotations);
        let hidden = args_with(&["--hide-annotations"]).session_config().unwrap();
        assert!(!hidden.annotations);
    }

    #[test]
    fn box_flag_reaches_session_config() {
        assert_eq!(
            base_args().session_config().unwrap().page_box,
            PageBox::Crop
        );
        for (name, page_box) in [
            ("media", PageBox::Media),
            ("crop", PageBox::Crop),
            ("bleed", PageBox::Bleed),
            ("trim", PageBox::Trim),
            ("art", PageBox::Art),
        ] {
            let config = args_with(&["--box", name]).session_config().unwrap();
            assert_eq!(config.page_box, page_box, "--box {name}");
        }
        let cropbox = args_with(&["--cropbox"]).session_config().unwrap();
        assert_eq!(cropbox.page_box, PageBox::Crop);
        assert!(
            Args::try_parse_from(["rrocket", "--cropbox", "--box", "media", "in.pdf", "out"])
                .is_err()
        );
    }

    #[test]
    fn rotate_flag_reaches_session_config() {
        assert_eq!(
            base_args().session_config().unwrap().rotation,
            Rotation::None
        );
        for (degrees, rotation) in [
            ("0", Rotation::None),
            ("90", Rotation::Cw90),
            ("180", Rotation::Cw180),
            ("270", Rotation::Cw270),
            ("-90", Rotation::Cw270),
            ("450", Rotation::Cw90),
            ("-720", Rotation::None),
        ] {
            let config = args_with(&["--rotate", degrees]).session_config().unwrap();
            assert_eq!(config.rotation, rotation, "--rotate {degrees}");
        }
        for bad in ["45", "-1", "90.0", "right"] {
            assert!(
                Args::try_parse_from(["rrocket", "--rotate", bad, "in.pdf", "out"]).is_err(),
                "--rotate {bad}"
            );
        }
    }

    #[test]
    fn overprint_flag_reaches_session_config() {
        assert!(!base_args().session_config().unwrap().overprint);
        let config = args_with(&["--overprint"]).session_config().unwrap();
        assert!(config.overprint);
    }

    #[test]
    fn antialias_flags_reach_session_config() {
        let config = base_args().session_config().unwrap();
        assert!(config.text_antialias && config.vector_antialias);
        let config = args_with(&["--aa", "no"]).session_config().unwrap();
        assert!(!config.text_antialias && config.vector_antialias);
        let config = args_with(&["--aaVector", "no"]).session_config().unwrap();
        assert!(config.text_antialias && !config.vector_antialias);
    }

    #[test]
    fn depth_sixteen_takes_the_16_bit_outputs_and_refuses_8_bit_ones() {
        assert_eq!(base_args().depth(), pdf_raster::SampleDepth::Eight);
        for flags in [
            &["--depth", "16"][..],
            &["--depth", "16", "--png"],
            &["--depth", "16", "--tiff", "--gray"],
            &["--depth", "16", "--png", "--transparent"],
            &["--depth", "16", "--invert=auto"],
            &["--depth", "8", "--mono"],
        ] {
            let args = args_with(flags);
            assert!(args.validate_format_flags().is_ok(), "{flags:?}");
        }
        assert_eq!(
            args_with(&["--depth", "16"]).depth(),
            pdf_raster::SampleDepth::Sixteen
        );
        if cfg!(not(windows)) {
            let print = Args::parse_from(["rrocket", "--print", "--depth", "16", "in.pdf"]);
            assert!(print.validate_format_flags().is_ok());
        }
        assert!(
            args_with(&["--depth", "16", "--overprint"])
                .validate_format_flags()
                .is_ok()
        );
        assert!(Args::try_parse_from(["rrocket", "--depth", "12", "in.pdf", "out"]).is_err());
        for (flags, needle) in [
            (&["--mono"][..], "use --gray"),
            (&["--dither", "halftone"], "--dither"),
            (&["--fax"], "--fax"),
            (&["--fax-to", "5551234"], "--fax"),
            (&["--jpeg"], "use --png or --tiff"),
            (&["--jpeg", "--jpegopt", "quality=90"], "--jpeg"),
            (&["--jpegcmyk"], "--jpegcmyk"),
            (&["--gpu-pages"], "drop --gpu-pages"),
        ] {
            let mut argv = vec!["--depth", "16"];
            argv.extend_from_slice(flags);
            let err = args_with(&argv).validate_format_flags().unwrap_err();
            assert!(
                err.starts_with("--depth 16 is refused") && err.contains(needle),
                "{flags:?}: {err}"
            );
        }
        for flags in [
            &["--crop"][..],
            &["--rectify", "--png"],
            &["--split-spreads", "--tiff"],
            &["--dewarp", "--gray"],
        ] {
            let mut argv = vec!["--depth", "16"];
            argv.extend_from_slice(flags);
            assert!(
                args_with(&argv).validate_format_flags().is_ok(),
                "{flags:?}"
            );
        }
        let comic = Args::parse_from(["rrocket", "--depth", "16", "--crop", "book.cbz", "out"]);
        assert!(comic.validate_format_flags().is_ok());
    }

    #[cfg(not(windows))]
    #[test]
    fn print_resolution_keeps_its_byte_budget_at_16_bits() {
        // 36 × 44 in at 600 dpi: 570 Mpx, under the 8-bit threshold, over
        // the 16-bit one.
        let page = (36.0 * 72.0, 44.0 * 72.0);
        let mut eight = Args::parse_from(["rrocket", "--print", "in.pdf"]);
        let mut sixteen = Args::parse_from(["rrocket", "--print", "--depth", "16", "in.pdf"]);
        let (eight_ok, sixteen_ok) = (
            eight.validate_format_flags(),
            sixteen.validate_format_flags(),
        );
        assert!(eight_ok.is_ok() && sixteen_ok.is_ok());
        eight.apply_print_preset();
        sixteen.apply_print_preset();
        assert_eq!(eight.page_dpi(page).0, crate::print::PRINT_DPI);
        assert_eq!(sixteen.page_dpi(page).0, crate::print::LARGE_FORMAT_DPI);
    }

    #[test]
    fn cmyk_takes_tiff_or_jpeg_and_jpegcmyk_is_jpeg_cmyk() {
        for argv in [
            &["--cmyk", "--tiff"][..],
            &["--cmyk", "--jpeg"],
            &["--jpegcmyk"],
        ] {
            assert!(args_with(argv).validate_format_flags().is_ok(), "{argv:?}");
        }
        let jpeg = args_with(&["--jpegcmyk", "--jpegopt", "quality=90"]);
        assert!(jpeg.validate_format_flags().is_ok());
        assert!(jpeg.cmyk() && jpeg.jpeg_output());
        assert!(matches!(jpeg.output_format(), OutputFormat::Jpeg(o) if o.quality == 90));
        assert!(!base_args().cmyk());
        for (argv, needle) in [
            (&["--cmyk"][..], "PPM has none; add --tiff or --jpeg"),
            (&["--cmyk", "--png"], "--cmyk is refused with --png"),
            (
                &["--jpegcmyk", "--gray"],
                "--jpegcmyk is refused with --gray",
            ),
            (&["--cmyk", "--tiff", "--mono"], "--mono"),
            (&["--cmyk", "--tiff", "--transparent"], "--transparent"),
            (&["--cmyk", "--tiff", "--invert"], "--invert"),
            (&["--cmyk", "--tiff", "--crop"], "--crop"),
            (&["--cmyk", "--tiff", "--split-spreads"], "--split-spreads"),
            (&["--cmyk", "--print"], "--print"),
            (&["--cmyk", "--fax"], "--fax"),
            (&["--jpegcmyk", "--gpu-pages"], "--gpu-pages"),
            (&["--jpegcmyk", "--tiff"], "mutually exclusive"),
            (&["--jpegcmyk", "--depth", "16"], "baseline JPEG is 8-bit"),
        ] {
            let err = args_with(argv).validate_format_flags().unwrap_err();
            assert!(err.contains(needle), "{argv:?}: {err}");
        }
        let comic = Args::parse_from(["rrocket", "--jpegcmyk", "book.cbz", "out"]);
        let err = comic.validate_format_flags().unwrap_err();
        assert!(err.contains("comic-archive input"), "{err}");
        assert!(
            args_with(&["--cmyk", "--tiff", "--depth", "16"])
                .validate_format_flags()
                .is_ok()
        );
    }

    #[test]
    fn fax_expands_to_bilevel_fax_coded_tiff_and_refuses_what_it_fixes() {
        let mut fax = args_with(&["--fax", "--fax-resolution", "standard"]);
        assert!(fax.validate_format_flags().is_ok());
        fax.apply_fax_preset();
        assert!(fax.mono && fax.tiff);
        assert_eq!(fax.invert_when(), InvertWhen::Auto);
        assert_eq!(fax.dpi(), (204.0, 98.0));
        assert_eq!(
            fax.output_format(),
            OutputFormat::Tiff(encode::TiffCompression::Group3 { k: None })
        );
        let mut g4 = args_with(&["--fax", "--tiffcompression", "g4", "--invert=never"]);
        assert!(g4.validate_format_flags().is_ok());
        g4.apply_fax_preset();
        assert_eq!(g4.invert_when(), InvertWhen::Never);
        assert_eq!(g4.dpi(), (204.0, 196.0));
        assert_eq!(
            g4.output_format(),
            OutputFormat::Tiff(encode::TiffCompression::Group4)
        );
        for (flags, needle) in [
            (&["--fax", "--png"][..], "drop --png"),
            (&["--fax", "-r", "300"], "drop -r"),
            (&["--fax", "--scale-to", "800"], "drop --scale-to"),
            (&["--fax", "--gray"], "drop --gray"),
            (&["--fax", "--tiffcompression", "lzw"], "not lzw"),
            (&["--fax-resolution", "fine"], "--fax output only"),
            (&["--fax", "--split-spreads"], "drop --split-spreads"),
            (&["--fax", "--crop"], "drop --crop"),
            (&["--fax", "--ram"], "drop --ram"),
        ] {
            let err = args_with(flags).validate_format_flags().unwrap_err();
            assert!(err.contains(needle), "{flags:?}: {err}");
        }
    }

    #[test]
    fn fax_refuses_comic_input() {
        let comic = Args::parse_from(["rrocket", "--fax", "book.CBZ", "out"]);
        assert!(comic.is_comic_input());
        let err = comic.validate_format_flags().unwrap_err();
        assert!(err.contains("PDF input"), "{err}");
    }

    #[test]
    fn fax_to_implies_fax_and_a_queue_needs_a_number() {
        let mut to = args_with(&["--fax-to", "+1 555 0100", "--fax-queue", "Fax"]);
        to.apply_fax_preset();
        assert!(to.fax && to.mono && to.tiff);
        assert!(Args::try_parse_from(["rrocket", "--fax-queue", "Fax", "in.pdf", "out"]).is_err());
        assert!(Args::try_parse_from(["rrocket", "--fax-to", "-x", "in.pdf", "out"]).is_err());
    }

    #[cfg(not(windows))]
    #[test]
    fn fax_to_is_validated_as_a_fax_document() {
        let to = args_with(&["--fax-to", "+1 555 0100", "--fax-queue", "Fax"]);
        assert!(to.validate_format_flags().is_ok());
        let err = args_with(&["--fax-to", "5550100", "--png"])
            .validate_format_flags()
            .unwrap_err();
        assert!(err.contains("drop --png"), "{err}");
    }

    /// Windows has neither `lp` nor `sendfax`: `--fax-to`, `--print` and
    /// `--printer` are refused before any other check, naming the
    /// alternative.
    #[cfg(windows)]
    #[test]
    fn submission_flags_are_refused_on_windows() {
        let err = args_with(&["--fax-to", "5550100", "--png"])
            .validate_format_flags()
            .unwrap_err();
        assert!(err.contains("Windows Fax and Scan"), "{err}");
        for flags in [&["--print"][..], &["--printer", "Office"]] {
            let err = print_args(flags).unwrap_err();
            assert!(err.contains("Windows does not have"), "{flags:?}: {err}");
        }
        assert!(crate::print::list_printers().is_err());
    }

    #[test]
    fn tiff_compression_resolves_and_its_misuse_is_refused() {
        let tiff = args_with(&["--tiff"]);
        assert!(tiff.validate_format_flags().is_ok());
        assert_eq!(
            tiff.output_format(),
            OutputFormat::Tiff(encode::TiffCompression::None)
        );
        assert_eq!(tiff.output_format().extension_with_mode(false, true), "tif");
        for (dpi, k) in [("196", 4), ("98", 2)] {
            let fax = args_with(&["--tiff", "--mono", "--tiffcompression", "g3-2d", "-r", dpi]);
            assert!(fax.validate_format_flags().is_ok());
            assert_eq!(
                fax.output_format(),
                OutputFormat::Tiff(encode::TiffCompression::Group3 {
                    k: NonZeroU32::new(k)
                })
            );
        }
        for (flags, needle) in [
            (&["--tiffcompression", "lzw"][..], "--tiff output only"),
            (&["--tiff", "--tiffcompression", "jpeg"], "--jpeg"),
            (&["--tiff", "--tiffcompression", "g4"], "g4 codes bilevel"),
            (&["--tiff", "--png"], "mutually exclusive"),
        ] {
            let err = args_with(flags).validate_format_flags().unwrap_err();
            assert!(err.contains(needle), "{flags:?}: {err}");
        }
    }

    #[test]
    fn transparent_needs_an_alpha_output_and_colour_pages() {
        for flags in [
            &["--transparent", "--png"][..],
            &["--transp", "--tiff", "--tiffcompression", "deflate"],
        ] {
            let args = args_with(flags);
            assert!(args.transparent);
            assert!(args.validate_format_flags().is_ok(), "{flags:?}");
        }
        for (flags, needle) in [
            (&["--transparent"][..], "PPM has none; add --png or --tiff"),
            (&["--transparent", "--jpeg"], "JPEG has none"),
            (&["--transparent", "--gray"], "drop --gray"),
            (&["--transparent", "--png", "--mono"], "drop --mono"),
            (&["--transparent", "--png", "--invert"], "drop --invert"),
            (&["--transparent", "--png", "--crop"], "drop --crop"),
            (&["--transparent", "--fax"], "drop --fax"),
        ] {
            let err = args_with(flags).validate_format_flags().unwrap_err();
            assert!(err.contains(needle), "{flags:?}: {err}");
        }
        let comic = Args::parse_from(["rrocket", "--transparent", "--png", "a.cbz", "out"]);
        let err = comic.validate_format_flags().unwrap_err();
        assert!(err.contains("PDF input"), "{err}");
    }

    #[test]
    fn jpeg_options_parse_as_pdftoppm_takes_them() {
        let options = parse_jpeg_options("quality=90,progressive=y,optimize=y").unwrap();
        assert_eq!(
            options,
            encode::JpegOptions {
                quality: 90,
                progressive: true,
                optimize: true
            }
        );
        assert_eq!(
            parse_jpeg_options("").unwrap(),
            encode::JpegOptions::default()
        );
        for bad in [
            "quality=101",
            "quality=x",
            "progressive=yes",
            "speed=1",
            "quality",
        ] {
            assert!(parse_jpeg_options(bad).is_err(), "{bad}");
        }
        let jpeg = args_with(&["--jpeg", "--jpegopt", "quality=40"]);
        assert!(jpeg.validate_format_flags().is_ok());
        assert!(matches!(jpeg.output_format(), OutputFormat::Jpeg(o) if o.quality == 40));
        for flags in [&["--jpegopt", "quality=40"][..], &["--jpeg", "--png"]] {
            assert!(
                args_with(flags).validate_format_flags().is_err(),
                "{flags:?}"
            );
        }
    }

    fn print_args(flags: &[&str]) -> Result<Args, String> {
        let mut command_line = vec!["rrocket"];
        command_line.extend_from_slice(flags);
        command_line.push("in.pdf");
        let mut parsed = Args::try_parse_from(command_line).map_err(|e| e.to_string())?;
        parsed.validate_format_flags()?;
        parsed.apply_print_preset();
        Ok(parsed)
    }

    #[cfg(not(windows))]
    #[test]
    fn print_needs_no_prefix_and_defaults_to_600_dpi_for_printing() {
        let args = print_args(&["--print"]).unwrap();
        assert_eq!(args.dpi(), (600.0, 600.0));
        assert_eq!(args.session_config().unwrap().purpose, RenderPurpose::Print);
        // The resolution follows the page: 360 dpi for a 44 × 120 in roll.
        let (letter, roll) = ((612.0, 792.0), (3168.0, 8640.0));
        assert_eq!(args.page_dpi(letter), (600.0, 600.0));
        assert_eq!(args.page_dpi(roll), (360.0, 360.0));
        assert_eq!(args.scales(roll), (5.0, 5.0));
        let args = print_args(&["--printer", "Office", "--rx", "300", "--ry", "600"]).unwrap();
        assert!(args.print);
        assert_eq!(args.dpi(), (300.0, 600.0));
        assert_eq!(
            args.page_dpi(roll),
            (300.0, 600.0),
            "a resolution flag wins"
        );
        let args = print_args(&["--print", "--purpose", "view", "-r", "1200"]).unwrap();
        assert_eq!(args.dpi(), (1200.0, 1200.0));
        assert_eq!(args.session_config().unwrap().purpose, RenderPurpose::View);
        // Without --print the prefix is required.
        assert!(Args::try_parse_from(["rrocket", "in.pdf"]).is_err());
    }

    #[cfg(not(windows))]
    #[test]
    fn print_refuses_file_output_and_print_flags_need_it() {
        for (flags, message) in [
            (&["--print", "--png"][..], "drop --png"),
            (&["--print", "--scale-to", "100"], "drop --scale-to"),
            (&["--print", "--ram"], "drop --ram"),
            (&["--print", "--fax"], "mutually exclusive"),
        ] {
            let error = print_args(flags).err().unwrap();
            assert!(error.contains(message), "{flags:?}: {error}");
        }
        let with_prefix = Args::parse_from(["rrocket", "--print", "in.pdf", "out"]);
        assert!(
            with_prefix
                .validate_format_flags()
                .unwrap_err()
                .contains("output prefix")
        );
        for flags in [&["--copies", "2"][..], &["--duplex", "long"]] {
            assert!(
                args_with(flags).validate_format_flags().is_err(),
                "{flags:?}"
            );
        }
        assert!(Args::try_parse_from(["rrocket", "--print", "--copies", "0", "in.pdf"]).is_err());
        assert!(Args::try_parse_from(["rrocket", "--printer", "-x", "in.pdf"]).is_err());
    }

    #[test]
    fn purpose_flag_reaches_session_config() {
        let view = base_args().session_config().unwrap();
        assert_eq!(view.purpose, RenderPurpose::View);
        let print = args_with(&["--purpose", "print"]).session_config().unwrap();
        assert_eq!(print.purpose, RenderPurpose::Print);
    }
}
