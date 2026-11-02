//! Command-line argument definitions.

use camera::CameraOptions;
use clap::Parser;
use pdf_raster::{BackendPolicy, RenderPurpose, RenderRegion, SessionConfig};

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
    /// Input PDF file ("-" for stdin). A path that begins with '-' must be
    /// passed after a literal `--` (or as `./-name.pdf`) so it is not parsed
    /// as an option.
    pub input: String,

    /// Output filename prefix (page number and extension are appended).
    pub output_prefix: String,

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

    /// CMYK JPEG output: not implemented; refused at startup.
    #[arg(long)]
    pub jpegcmyk: bool,

    /// Output TIFF.
    #[arg(long)]
    pub tiff: bool,

    /// TIFF compression: `none` (default), `packbits`, `lzw`, `deflate`,
    /// or, with --mono, the fax codings `g3` (one-dimensional), `g3-2d`
    /// and `g4`.
    #[arg(long = "tiffcompression", value_name = "METHOD", value_enum)]
    pub tiffcompression: Option<TiffMethod>,

    /// Output grey-scale PPM/PNG.
    #[arg(long = "gray")]
    pub gray: bool,

    /// Output 1-bit mono: PBM, or a two-level PNG, JPEG or TIFF.
    #[arg(long = "mono")]
    pub mono: bool,

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
    #[arg(long = "fax-queue", value_name = "NAME", requires = "fax_to")]
    pub fax_queue: Option<String>,

    /// How --mono renders intermediate grays: `none` (default), a 50%
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

    // ── Rendering options ────────────────────────────────────────────────────
    /// Antialias text (default yes).  `no` draws glyphs with hard,
    /// bilevel edges.
    #[arg(long = "aa", value_name = "yes|no", default_value = "yes")]
    pub antialias: AaFlag,

    /// Antialias path fills, strokes and clips (default yes).  `no` draws
    /// them with hard, bilevel edges.
    #[arg(long = "aaVector", value_name = "yes|no", default_value = "yes")]
    pub vector_antialias: AaFlag,

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
    /// all prefixes on systems without `/dev/shm` (macOS, the BSDs).
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
    /// On systems without `/dev/shm` (macOS), point it into a RAM disk.
    /// The path must not exist: the directory is created for the run and
    /// removed, with its contents, on exit.
    #[arg(long = "ram-path", value_name = "PATH", conflicts_with = "no_ram")]
    pub ram_path: Option<String>,

    // ── Backend selection ─────────────────────────────────────────────────────
    /// Compute backend for image decoding and GPU fills.
    ///
    /// `auto`   — Vulkan if compiled in and present, else CUDA, else CPU
    ///            (silent fallback at every step).  This is the default.
    /// `cpu`    — CPU only; all GPU init is skipped.
    /// `cuda`   — Require CUDA (nvJPEG/AA fill/ICC); exit with error if unavailable.
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
    /// `view`  — on-screen viewing (the default): layers take their View
    ///           usage state and annotations marked not for viewing are omitted.
    /// `print` — printing: layers take their Print usage state and only
    ///           annotations marked for printing are drawn.
    #[arg(
        long = "purpose",
        value_name = "PURPOSE",
        default_value = "view",
        verbatim_doc_comment
    )]
    pub purpose: PurposeArg,

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

    /// Split a double-page spread at its gutter into two output files.
    ///
    /// The halves of page N are written as N with an `a` (left) and `b`
    /// (right) suffix, e.g. `out-07a.png` and `out-07b.png`; pages not
    /// detected as spreads keep their usual name.
    #[arg(long = "split-spreads")]
    pub split_spreads: bool,

    /// Prefetch image `XObject`s into the device-resident image cache at
    /// session open.  Only meaningful when the binary was built with the
    /// `cache` feature; ignored otherwise.  Off by default — the prefetch
    /// walk is wasted work for short single-page renders, but a clear win
    /// on multi-page or multi-pass workloads (OCR pipelines).
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

/// `--dither` argument value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum DitherArg {
    /// Threshold at mid-gray.
    None,
    /// Halftone screen.
    Halftone,
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

    /// Which pages `--invert` turns.
    pub(crate) fn invert_when(&self) -> InvertWhen {
        self.invert.unwrap_or(InvertWhen::Never)
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
    /// (after `/Rotate`): under `--fax` the fax layout (see
    /// [`crate::fax::layout`]), else from `--scale-to` when given, else from
    /// `--scale-to-x` / `--scale-to-y`, an axis given neither taking the
    /// other's scale when only one is set, and otherwise from the resolution
    /// flags.
    pub(crate) fn scales(&self, (width_pts, height_pts): (f64, f64)) -> (f64, f64) {
        if self.fax {
            let layout = crate::fax::layout((width_pts, height_pts), self.y_dpi());
            return (layout.scale_x, layout.scale_y);
        }
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
            (None, None) => (self.x_dpi() / 72.0, self.y_dpi() / 72.0),
        }
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
        if self.jpegcmyk {
            return Err(
                "--jpegcmyk output is not implemented; use --jpeg, --png or the default PPM output"
                    .to_owned(),
            );
        }
        self.validate_fax_flags()?;
        if self.mono && self.gray {
            return Err("--mono and --gray are mutually exclusive".to_owned());
        }
        if [self.png, self.jpeg, self.tiff]
            .iter()
            .filter(|&&f| f)
            .count()
            > 1
        {
            return Err("--png, --jpeg and --tiff are mutually exclusive".to_owned());
        }
        if self.dither != DitherArg::None && !(self.mono || self.faxing()) {
            return Err("--dither applies to --mono and --fax output only".to_owned());
        }
        if self.jpegopt.is_some() && !self.jpeg {
            return Err("--jpegopt applies to --jpeg output only".to_owned());
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

    /// Whether a fax document is written: `--fax`, or `--fax-to`, which
    /// implies it.
    pub(crate) const fn faxing(&self) -> bool {
        self.fax || self.fax_to.is_some()
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
        let fixed = [
            (self.png, "--png"),
            (self.jpeg, "--jpeg"),
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
        ];
        if let Some((_, flag)) = fixed.iter().find(|(set, _)| *set) {
            return Err(format!(
                "--fax fixes the format, size and resolution of its pages; drop {flag} \
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
        } else if self.jpeg {
            OutputFormat::Jpeg(self.jpegopt.unwrap_or_default())
        } else if self.tiff {
            OutputFormat::Tiff(self.tiff_compression())
        } else {
            OutputFormat::Ppm
        }
    }
}

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
    /// two-dimensional coding takes the ITU-T T.4 period: 4 at fine
    /// resolution (at least 150 lines per inch), 2 at standard.
    fn compression(self, y_dpi: f64) -> encode::TiffCompression {
        match self {
            Self::None | Self::Jpeg => encode::TiffCompression::None,
            Self::PackBits => encode::TiffCompression::PackBits,
            Self::Lzw => encode::TiffCompression::Lzw,
            Self::Deflate => encode::TiffCompression::Deflate,
            Self::G3 => encode::TiffCompression::Group3 { k: None },
            Self::G3TwoD => encode::TiffCompression::Group3 {
                k: Some(if y_dpi >= 150.0 { 4 } else { 2 }),
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
    /// Auto-select: Vulkan if compiled in and present, else CUDA, else CPU.
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
    /// initialisation fails.  The device-resident image cache is
    /// CUDA-only; under this mode the session runs without it.
    Vulkan,
}

impl Args {
    /// Build a [`SessionConfig`] from the `--backend` / `--vaapi-device` /
    /// `--vulkan-device` flags.
    ///
    /// Rejects `--vaapi-device` when combined with `--backend cpu`,
    /// `--backend cuda` or `--backend vulkan`, and `--vulkan-device` with
    /// any backend but `auto` and `vulkan` — those modes never touch the
    /// device in question, so allowing the combination would silently
    /// ignore a user expectation.  `PDF_RASTER_VULKAN_DEVICE` is consulted
    /// only when `--vulkan-device` is absent and is never rejected.
    pub(crate) fn session_config(&self) -> Result<SessionConfig, String> {
        // Precedence: explicit `--backend <X>` wins; otherwise consult
        // `PDF_RASTER_BACKEND`; otherwise default to Auto.  Distinguishing
        // "user passed --backend auto" from "user passed nothing" requires
        // `Option<BackendArg>` rather than a clap `default_value` — with
        // a default value the two cases are indistinguishable and the env
        // var would never get a chance to override.
        let policy = match self.backend {
            Some(BackendArg::Auto) => BackendPolicy::Auto,
            Some(BackendArg::Cpu) => BackendPolicy::CpuOnly,
            Some(BackendArg::Cuda) => BackendPolicy::ForceCuda,
            #[cfg(feature = "vaapi")]
            Some(BackendArg::Vaapi) => BackendPolicy::ForceVaapi,
            Some(BackendArg::Vulkan) => BackendPolicy::ForceVulkan,
            None => BackendPolicy::from_env(),
        };

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
        config.font_dirs = SessionConfig::font_dirs_from_env();
        config.system_fonts = !self.no_system_fonts;
        config.strict = self.strict;
        config.purpose = self.purpose.into();
        config.annotations = !self.hide_annotations;
        config.text_antialias = self.antialias == AaFlag::Yes;
        config.vector_antialias = self.vector_antialias == AaFlag::Yes;
        #[cfg(feature = "cache")]
        {
            config.prefetch = self.prefetch;
        }
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

#[cfg(test)]
mod page_list_tests {
    use super::*;

    fn base_args() -> Args {
        Args::parse_from(["rrocket", "in.pdf", "out"])
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
    fn vulkan_device_flag_reaches_session_config() {
        let config = Args::parse_from(["rrocket", "--vulkan-device", "5070", "in.pdf", "out"])
            .session_config()
            .unwrap();
        assert_eq!(config.vulkan_device.as_deref(), Some("5070"));
        let config = Args::parse_from([
            "rrocket",
            "--backend",
            "vulkan",
            "--vulkan-device",
            "0",
            "in.pdf",
            "out",
        ])
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
        let err = Args::parse_from([
            "rrocket",
            "--backend",
            "cpu",
            "--vaapi-device",
            "auto",
            "in.pdf",
            "out",
        ])
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
            let err = Args::parse_from([
                "rrocket",
                "--backend",
                backend,
                "--vulkan-device",
                "0",
                "in.pdf",
                "out",
            ])
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
    fn strict_flag_reaches_session_config() {
        let lenient = base_args().session_config().unwrap();
        assert!(!lenient.strict);
        let strict = Args::parse_from(["rrocket", "--strict", "in.pdf", "out"])
            .session_config()
            .unwrap();
        assert!(strict.strict);
    }

    #[test]
    fn scale_to_flags_set_the_page_scales() {
        let page = (200.0, 100.0);
        let scales = |argv: &[&str]| {
            let mut full = vec!["rrocket"];
            full.extend_from_slice(argv);
            full.extend_from_slice(&["in.pdf", "out"]);
            Args::parse_from(full).scales(page)
        };
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
        let region = |argv: &[&str]| {
            let mut full = vec!["rrocket"];
            full.extend_from_slice(argv);
            full.extend_from_slice(&["in.pdf", "out"]);
            Args::parse_from(full).crop_region((100, 50))
        };
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
        let hidden = Args::parse_from(["rrocket", "--hide-annotations", "in.pdf", "out"])
            .session_config()
            .unwrap();
        assert!(!hidden.annotations);
    }

    #[test]
    fn antialias_flags_reach_session_config() {
        let config = base_args().session_config().unwrap();
        assert!(config.text_antialias && config.vector_antialias);
        let config = Args::parse_from(["rrocket", "--aa", "no", "in.pdf", "out"])
            .session_config()
            .unwrap();
        assert!(!config.text_antialias && config.vector_antialias);
        let config = Args::parse_from(["rrocket", "--aaVector", "no", "in.pdf", "out"])
            .session_config()
            .unwrap();
        assert!(config.text_antialias && !config.vector_antialias);
    }

    #[test]
    fn unimplemented_formats_are_refused_up_front() {
        let err = Args::parse_from(["rrocket", "--jpegcmyk", "in.pdf", "out"])
            .validate_format_flags()
            .unwrap_err();
        assert!(
            err.starts_with("--jpegcmyk") && err.contains("--jpeg,"),
            "{err}"
        );
        assert!(base_args().validate_format_flags().is_ok());
    }

    /// `flags` parsed between the program name and the operands.
    fn args_with(flags: &[&str]) -> Args {
        let mut argv = vec!["rrocket"];
        argv.extend_from_slice(flags);
        argv.extend(["in.pdf", "out"]);
        Args::parse_from(argv)
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
        ] {
            let err = args_with(flags).validate_format_flags().unwrap_err();
            assert!(err.contains(needle), "{flags:?}: {err}");
        }
    }

    #[test]
    fn fax_to_implies_fax_and_a_queue_needs_a_number() {
        let mut to = args_with(&["--fax-to", "+1 555 0100", "--fax-queue", "Fax"]);
        assert!(to.validate_format_flags().is_ok());
        to.apply_fax_preset();
        assert!(to.fax && to.mono && to.tiff);
        assert!(Args::try_parse_from(["rrocket", "--fax-queue", "Fax", "in.pdf", "out"]).is_err());
        assert!(Args::try_parse_from(["rrocket", "--fax-to", "-x", "in.pdf", "out"]).is_err());
        let err = args_with(&["--fax-to", "5550100", "--png"])
            .validate_format_flags()
            .unwrap_err();
        assert!(err.contains("drop --png"), "{err}");
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
                OutputFormat::Tiff(encode::TiffCompression::Group3 { k: Some(k) })
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
        let jpeg = Args::parse_from([
            "rrocket",
            "--jpeg",
            "--jpegopt",
            "quality=40",
            "in.pdf",
            "out",
        ]);
        assert!(jpeg.validate_format_flags().is_ok());
        assert!(matches!(jpeg.output_format(), OutputFormat::Jpeg(o) if o.quality == 40));
        for flags in [&["--jpegopt", "quality=40"][..], &["--jpeg", "--png"]] {
            assert!(
                args_with(flags).validate_format_flags().is_err(),
                "{flags:?}"
            );
        }
    }

    #[test]
    fn purpose_flag_reaches_session_config() {
        let view = base_args().session_config().unwrap();
        assert_eq!(view.purpose, RenderPurpose::View);
        let print = Args::parse_from(["rrocket", "--purpose", "print", "in.pdf", "out"])
            .session_config()
            .unwrap();
        assert_eq!(print.purpose, RenderPurpose::Print);
    }
}
