//! Backend selection policy and session configuration.

#[cfg(doc)]
use crate::{PageDiagnostics, RasterSession, open_session, open_session_from_bytes};

use std::path::PathBuf;
use std::sync::Arc;

use crate::{PageBox, Password, RasterError, RenderPurpose, Rotation};

// ── Backend policy ────────────────────────────────────────────────────────────

/// Controls which compute backend is used for image decoding and GPU kernels.
///
/// The default is [`Auto`](BackendPolicy::Auto), which renders on the CPU
/// unless a device kernel is enabled by its threshold, and then prefers
/// Vulkan when compiled in, falls through to CUDA, and finally to the CPU
/// paths.  The `Force*` variants turn silent fallbacks into hard
/// errors so you can tell immediately whether the expected hardware path is
/// actually being taken.
///
/// At runtime, [`SessionConfig::default()`] resolves the policy via
/// [`BackendPolicy::from_env`], so users can switch backends per process by
/// setting `PDF_RASTER_BACKEND={auto,cpu,cuda,vaapi,vulkan}` without
/// rebuilding.  [`SessionConfig::with_policy`] bypasses the env var.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum BackendPolicy {
    /// Auto: the CPU, unless the parallel-Huffman JPEG decoder is enabled
    /// by its threshold, which it is not by default; then prefer Vulkan when present, fall through to
    /// CUDA, then CPU.  Silent fallback at every step.  A CPU session
    /// initialises no GPU backend, image cache or nvJPEG decoder; with the
    /// `nvjpeg2k` feature a rendering thread builds its nvJPEG2000 decoder
    /// when it first draws a `JPXDecode` image the decoder takes at more
    /// than half its size on some axis; smaller draws decode on the CPU.
    #[default]
    Auto,
    /// CPU only.  All GPU init is skipped; no CUDA or VA-API calls are made.
    CpuOnly,
    /// Require CUDA (nvJPEG2000, ICC CLUT, image blits, and nvJPEG while its
    /// threshold enables it).
    /// Returns [`RasterError::BackendUnavailable`] if CUDA initialisation fails
    /// rather than falling back to CPU.
    ForceCuda,
    /// Require VA-API JPEG decoding.
    /// Returns [`RasterError::BackendUnavailable`] if the VA-API device cannot
    /// be opened rather than falling back to CPU.
    ///
    /// Only available when the `vaapi` Cargo feature is enabled.
    #[cfg(feature = "vaapi")]
    ForceVaapi,
    /// Require the Vulkan compute backend.  Returns
    /// [`RasterError::BackendUnavailable`] if Vulkan initialisation fails
    /// (or the binary was built without `--features vulkan`).
    ///
    /// The ICC CMYK→RGB CLUT and image blit kernels dispatch
    /// through `VulkanBackend`, and in a build with the `cache` feature the
    /// device-resident image cache keeps its slabs on the Vulkan device.
    ForceVulkan,
}

impl BackendPolicy {
    /// Resolve a policy from the `PDF_RASTER_BACKEND` environment variable.
    ///
    /// Convenience wrapper for [`BackendPolicy::from_env_var`] with the
    /// canonical name.  Accepted values (case-insensitive): `auto`, `cpu`,
    /// `cuda`, `vulkan`, and `vaapi` in a build with the `vaapi` feature.
    /// Unset, empty, or unrecognised values fall back to
    /// [`BackendPolicy::Auto`] — unrecognised values also emit a
    /// `log::warn!` so a typo doesn't silently mis-route.
    ///
    /// Precedence: an explicit `--backend` flag (CLI) wins over
    /// `PDF_RASTER_BACKEND`, which in turn wins over the compile-time
    /// default.  This lets a binary ship Vulkan-default while users
    /// override per-process without recompiling.
    #[must_use]
    pub fn from_env() -> Self {
        Self::from_env_var("PDF_RASTER_BACKEND")
    }

    /// Resolve a policy from a named environment variable.
    ///
    /// Used by tools (e.g. the contest harness) that want their own
    /// env-var namespace separate from `PDF_RASTER_BACKEND` so a
    /// developer's personal default doesn't leak into bench runs.
    /// Same value vocabulary and warning semantics as
    /// [`BackendPolicy::from_env`].
    ///
    /// Whitespace around the value is trimmed so `PDF_RASTER_BACKEND=" cuda "`
    /// (e.g. quoted in a shell config that didn't strip) still resolves.
    /// Matching is case-insensitive.  Unrecognised values emit a
    /// `log::warn!` (not stderr) so library embedders control the sink.
    #[must_use]
    pub fn from_env_var(name: &str) -> Self {
        // Unset / empty / unreadable env var → Auto with no allocation.
        let Some(raw) = std::env::var_os(name) else {
            return Self::Auto;
        };
        let Some(s) = raw.to_str() else {
            log::warn!("{name} contains non-UTF-8 bytes; using Auto");
            return Self::Auto;
        };
        s.parse().unwrap_or_else(|e: ParseBackendPolicyError| {
            log::warn!("{name}: {e}; using auto");
            Self::Auto
        })
    }

    /// Every policy this build accepts, in [`name`](Self::name) order.
    pub const ALL: &[Self] = &[
        Self::Auto,
        Self::CpuOnly,
        Self::ForceCuda,
        #[cfg(feature = "vaapi")]
        Self::ForceVaapi,
        Self::ForceVulkan,
    ];
}

/// A backend name that [`BackendPolicy`]'s `FromStr` does not accept.
///
/// Its `Display` names the rejected value and the names this build accepts
/// (`vaapi` only with the `vaapi` feature).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseBackendPolicyError {
    value: String,
}

impl ParseBackendPolicyError {
    /// The rejected input, as given.
    #[must_use]
    pub fn value(&self) -> &str {
        &self.value
    }
}

impl std::fmt::Display for ParseBackendPolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unknown backend {:?}; expected one of", self.value)?;
        for (i, policy) in BackendPolicy::ALL.iter().enumerate() {
            let sep = if i == 0 { " " } else { ", " };
            write!(f, "{sep}{}", policy.name())?;
        }
        Ok(())
    }
}

impl std::error::Error for ParseBackendPolicyError {}

impl BackendPolicy {
    /// The name [`FromStr`](std::str::FromStr) and `PDF_RASTER_BACKEND`
    /// accept for this policy: `auto`, `cpu`, `cuda`, `vaapi` or `vulkan`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::CpuOnly => "cpu",
            Self::ForceCuda => "cuda",
            #[cfg(feature = "vaapi")]
            Self::ForceVaapi => "vaapi",
            Self::ForceVulkan => "vulkan",
        }
    }
}

impl std::str::FromStr for BackendPolicy {
    type Err = ParseBackendPolicyError;

    /// Parse a backend name.  Whitespace is trimmed and matching is
    /// case-insensitive.  Empty input maps to `Auto`.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let name = s.trim();
        if name.is_empty() {
            return Ok(Self::Auto);
        }
        Self::ALL
            .iter()
            .copied()
            .find(|policy| policy.name().eq_ignore_ascii_case(name))
            .ok_or_else(|| ParseBackendPolicyError {
                value: s.to_owned(),
            })
    }
}

// ── Session configuration ─────────────────────────────────────────────────────

/// Default [`SessionConfig::vaapi_device`]: `auto`, which probes
/// `/dev/dri/renderD*` in ascending node number and uses the first node
/// whose driver supports VA-API baseline JPEG decode.
pub const DEFAULT_VAAPI_DEVICE: &str = "auto";

/// Configuration for opening a [`RasterSession`].
///
/// Passed to [`open_session`].  Use [`Default::default()`] for the standard
/// behaviour: backend resolved from `PDF_RASTER_BACKEND` env var (or `Auto`
/// if unset), Vulkan device from `PDF_RASTER_VULKAN_DEVICE` (or automatic),
/// probed DRM render node, image-cache prefetch off.
///
/// This struct is `#[non_exhaustive]`: construct it via [`SessionConfig::default()`]
/// or [`SessionConfig::with_policy()`] rather than a struct literal.  This allows
/// new fields to be added without a breaking change — in particular, the `prefetch`
/// field only exists when the `cache` Cargo feature is enabled.
#[non_exhaustive]
#[derive(Debug, Clone)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "each bool is an independent caller-facing switch"
)]
pub struct SessionConfig {
    /// Backend selection policy.
    pub policy: BackendPolicy,
    /// VA-API DRM render node path, or [`DEFAULT_VAAPI_DEVICE`] (`auto`, the
    /// default) to probe for one.
    ///
    /// When no node qualifies, or the given node cannot be opened, `Auto`
    /// decodes JPEG on the CPU and `ForceVaapi`
    /// fails [`open_session`] with [`RasterError::BackendUnavailable`]
    /// naming the render nodes present.  Only relevant when the `vaapi`
    /// feature is enabled and `policy` is [`Auto`](BackendPolicy::Auto) or
    /// `ForceVaapi`.
    pub vaapi_device: String,
    /// Vulkan physical device to use, or `None` to pick automatically.
    ///
    /// Either an index in device enumeration order (counted from 0) or a
    /// case-insensitive substring of the device name; an all-digit value
    /// that is not a valid index is matched as a name.  A value that does
    /// not resolve to exactly one usable device makes [`open_session`]
    /// return [`RasterError::InvalidOptions`] listing the enumerated
    /// devices, under `Auto` as well as `ForceVulkan`; with every Vulkan kernel
    /// dormant, `Auto` resolves it against the enumerated devices without
    /// opening one.
    ///
    /// [`SessionConfig::with_policy`] sets `None`; [`SessionConfig::default`]
    /// reads `PDF_RASTER_VULKAN_DEVICE` (see
    /// [`SessionConfig::vulkan_device_from_env`]).  Only relevant when the
    /// `vulkan` feature is enabled and `policy` is
    /// [`Auto`](BackendPolicy::Auto) or [`ForceVulkan`](BackendPolicy::ForceVulkan).
    pub vulkan_device: Option<String>,
    /// Whether to spawn the image-cache prefetcher at session open.
    /// Default `false` — opt-in because the prefetcher reads every
    /// page's resource dict eagerly, which is wasted work for short
    /// single-page renders.  Long renders, multi-pass renders (OCR
    /// pipelines), and re-renders of the same PDF benefit.
    ///
    /// No effect when the `cache` feature is disabled or when GPU
    /// initialisation fails (no cache → nowhere to prefetch into).
    #[cfg(feature = "cache")]
    pub prefetch: bool,
    /// Device pixels per point `(x, y)` at which every page of the session
    /// is rendered, when that is one scale for all pages (`dpi / 72` at a
    /// fixed resolution); `None` (the default) when unknown or page
    /// dependent.  With a scale the prefetcher skips an image every draw of
    /// which on its page is smaller than the image, which the render
    /// resamples on the host without the image cache; without one it
    /// prefetches every JPEG image.  No effect unless
    /// [`prefetch`](Self::prefetch) is set.
    #[cfg(feature = "cache")]
    pub prefetch_scale: Option<(f64, f64)>,
    /// Whether the caller has authorised decrypting an encrypted (PDF
    /// Standard Security Handler) document.
    ///
    /// Default `false`.  When the document is encrypted and this is
    /// `false`, [`open_session`] and [`open_session_from_bytes`] return a
    /// clear [`RasterError::Pdf`] explaining the document is encrypted
    /// (never the misleading "no pages"), whatever [`password`](Self::password)
    /// holds.  The CLI sets this only after an interactive private-copy /
    /// liability confirmation or an explicit `--decrypt-owned` /
    /// `RROCKET_DECRYPT_OWNED=1` operator bypass.  No effect on
    /// unencrypted documents.
    pub decrypt_authorized: bool,
    /// Password for an encrypted document, tried as the user password and
    /// then as the owner password after the empty user password (which
    /// alone opens a permissions-only document).
    ///
    /// Default `None`.  An input to decryption, not an authorisation: it has
    /// no effect unless [`decrypt_authorized`](Self::decrypt_authorized) is
    /// `true`, and none on an unencrypted document.  `Password` converts
    /// from `&str` and `String`, and its `Debug` output is redacted.
    pub password: Option<Password>,
    /// Whether [`open_session`] copies the file into memory instead of
    /// memory-mapping it.
    ///
    /// Default `false`: a file of at least 256 KiB is mapped, and its bytes
    /// are read from the page cache as pages render.  The mapping is only as
    /// stable as the file: if another process truncates the file while the
    /// session is alive, the first access to a mapped page past the new end
    /// of file raises `SIGBUS`, which terminates the whole process rather
    /// than failing the render.  `true` reads the whole file at open into a
    /// buffer the session owns, so later changes to the file cannot reach
    /// it, at the cost of the file's size in memory.  For a long-running
    /// process rendering files it does not control.  No effect on the
    /// sessions opened from bytes, shared storage or a reader, which hold
    /// their own bytes.
    pub read_into_memory: bool,
    /// Whether an image or a content stream that cannot be decoded, or
    /// decodes only in part, fails its page.
    ///
    /// Default `false`: the image or stream is skipped (or drawn from the
    /// data that decoded), the page renders, a warning naming the page and
    /// the object is logged, and
    /// the failure is reported in [`PageDiagnostics::image_failures`] or
    /// [`PageDiagnostics::content_failures`].  When `true` the page returns
    /// [`RasterError::ImageDecodeFailed`] or
    /// [`RasterError::ContentDecodeFailed`] instead.  A Form `XObject`
    /// nested too deep to draw is likewise skipped and reported in
    /// [`PageDiagnostics::forms_too_deep`], or, when `true`, fails the page
    /// with [`RasterError::PageBudgetExceeded`].
    pub strict: bool,
    /// What the pages are rendered for: on-screen viewing (the default) or
    /// printing.
    ///
    /// Selects the optional-content usage event applied to the document's
    /// layers and the annotation flags honoured (see [`RenderPurpose`]).
    pub purpose: RenderPurpose,
    /// Whether annotations (PDF §12.5, form-field widgets included) are drawn
    /// over the page content.
    ///
    /// Default `true`: every annotation shown for [`purpose`](Self::purpose)
    /// is drawn.  `false` renders the page content alone.
    pub annotations: bool,
    /// Whether glyphs are drawn with antialiased (8-bit coverage) edges.
    ///
    /// Default `true`.  `false` rasterises each glyph bilevel: a pixel is
    /// inked or not.  Text drawn as paths (Type 3 glyphs, stroked or
    /// clipping render modes) follows [`vector_antialias`](Self::vector_antialias).
    pub text_antialias: bool,
    /// Whether path fills, strokes and clips are antialiased.
    ///
    /// Default `true`: coverage is sampled on a 4 × 4 grid per pixel.
    /// `false` scan-converts at pixel centres, giving hard bilevel edges;
    /// such fills always run on the CPU.
    pub vector_antialias: bool,
    /// Whether overprint (`OP`, `op`, `OPM`; PDF 2.0 §8.6.7) is simulated.
    ///
    /// Default `false`: every object knocks out what lies beneath it.  `true`
    /// paints a page whose graphics states set overprint on a canvas of
    /// process and spot inks, so that an overprinting object leaves the inks
    /// it does not paint, and converts the result to RGB; pages without
    /// overprint render unchanged.
    pub overprint: bool,
    /// Extra directories of installed fonts, searched in order before the
    /// standard font directories.
    ///
    /// A font the document does not embed is drawn with an installed
    /// substitute (the metric-compatible Liberation family, a face of the
    /// font's script) and, for a character its substitute lacks, with an
    /// installed face that covers it; these directories are searched first
    /// for both.  [`SessionConfig::with_policy`] sets none;
    /// [`SessionConfig::default`] reads `RROCKET_FONT_DIR` (see
    /// [`SessionConfig::font_dirs_from_env`]).  Sessions naming the same
    /// directories share one index of the installed fonts.
    pub font_dirs: Vec<PathBuf>,
    /// Whether the standard font directories (and the fixed locations of
    /// the metric-compatible Liberation, Arial, Times New Roman and Courier
    /// New faces) are searched after [`font_dirs`](Self::font_dirs).
    ///
    /// Default `true`.  `false` draws non-embedded fonts from the
    /// directories in `font_dirs` and the faces compiled into the library
    /// only: the base-14 substitutes and, with the `bundled-fonts` feature
    /// (on by default), the bundled Greek, Cyrillic, Hebrew, Arabic,
    /// Devanagari, Thai and symbol fallback faces.
    pub system_fonts: bool,
    /// Host bytes of decoded images the session keeps for reuse by later
    /// pages: an image `XObject` drawn on several pages is decoded once
    /// while its decode fits, and the decodes kept stay resident for the
    /// session's life.
    ///
    /// Default [`DOCUMENT_IMAGE_BUDGET_BYTES`](crate::DOCUMENT_IMAGE_BUDGET_BYTES)
    /// (256 MiB).  0 keeps nothing: every page decodes the images it draws.
    pub image_budget: usize,
    /// The most bytes the bitmap of one render may hold: the page, or the
    /// region drawn, at its width × height × bytes per pixel (3 for 8-bit
    /// RGB, 4 over a transparent backdrop).
    ///
    /// Default [`DEFAULT_MAX_PAGE_BYTES`](crate::DEFAULT_MAX_PAGE_BYTES)
    /// (1.8 GB).  A larger bitmap is refused with
    /// [`RasterError::PageTooLarge`] before it is allocated; the sides of a
    /// page are held to [`MAX_PX_SIDE`](crate::MAX_PX_SIDE) whatever the
    /// budget.  Under Linux's default overcommit an allocation past the
    /// machine's memory is not refused but stopped by the out-of-memory
    /// killer when it is filled, so the budget is the bound a hostile page
    /// size meets.
    pub max_page_bytes: u64,
    /// The bytes each band of [`render_page_bands`](crate::render_page_bands)
    /// holds at most: its rows are this over the bytes of one row, and at
    /// least one.
    ///
    /// Default [`DEFAULT_BAND_BYTES`](crate::DEFAULT_BAND_BYTES) (64 MiB).
    /// The other entry points render pages whole and do not read it.
    pub band_bytes: u64,
    /// The page boundary box (ISO 32000-2 §14.11.2) every page of the
    /// session is measured and rendered on.
    ///
    /// Default [`PageBox::Crop`]: the `CropBox` intersected with the
    /// `MediaBox`.  [`PageBox::Media`] renders the whole `MediaBox`;
    /// [`PageBox::Bleed`], [`PageBox::Trim`] and [`PageBox::Art`] render the
    /// page's own box of that kind intersected with the `MediaBox`, or the
    /// `CropBox` region when the page has none.
    pub page_box: PageBox,
    /// A rotation added to every page's `/Rotate` before it is drawn.
    ///
    /// The page is rasterised in the turned orientation: a quarter turn
    /// exchanges the page's width and height in every size the session
    /// reports and renders, and a [`RenderRegion`](crate::RenderRegion) is
    /// placed on the turned page.  Default [`Rotation::None`].
    pub rotation: Rotation,
    /// Whether pages are rendered whole on a Vulkan GPU beside the CPU
    /// workers.
    ///
    /// Default `false`.  `true` opens the device of
    /// [`vulkan_device`](Self::vulkan_device) on a thread of its own when the
    /// session opens; once it is open, a page whose every paint the GPU
    /// raster renders (fills and strokes of solid colours under rectangular
    /// clips, `Normal` blend, constant alpha) is rendered on it while it is
    /// idle, and every other page on the CPU, with the same bytes.
    /// [`PageDiagnostics::device_page`](crate::PageDiagnostics::device_page)
    /// reports where each page rendered and why.  A device that does not
    /// open is logged at `error` and leaves every page on the CPU.
    ///
    /// Requires a build with the `vulkan` feature,
    /// [`policy`](Self::policy) [`Auto`](BackendPolicy::Auto) or
    /// [`ForceVulkan`](BackendPolicy::ForceVulkan), and
    /// [`vector_antialias`](Self::vector_antialias); otherwise
    /// [`open_session`] returns [`RasterError::InvalidOptions`] (see
    /// [`check_gpu_pages`](Self::check_gpu_pages)).
    pub gpu_pages: bool,
    /// The ICC profile every page is converted to once composited, its
    /// device values the pages' pixels.
    ///
    /// Default `None`: pages are sRGB.  An RGB profile yields RGB pages of
    /// its device values, a gray one gray pages
    /// ([`FinishedCanvas::Gray8`](crate::FinishedCanvas::Gray8),
    /// [`Gray16`](crate::FinishedCanvas::Gray16)).  Each page converts under
    /// [`RenderParams::intent`](crate::RenderParams::intent), else the one
    /// intent its marking operators used, else relative colorimetric (see
    /// [`PageDiagnostics::output_intent`](crate::PageDiagnostics::output_intent)).
    pub output_profile: Option<Arc<pdf_interp::OutputProfile>>,
    /// Soft-proof every page against its output intent: paint it as the
    /// process ink of the press the document's (or page's)
    /// `/OutputIntents` names, then convert that ink to the display, the
    /// working space or [`output_profile`](Self::output_profile), under
    /// the [`ProofIntent`](pdf_interp::ProofIntent):
    /// [`AbsoluteColorimetric`](pdf_interp::ProofIntent::AbsoluteColorimetric)
    /// simulates the paper white.
    ///
    /// Default `None`.  The output intent is the first of subtype
    /// `GTS_PDFX` that embeds a profile, else the first of any subtype
    /// that does, and its profile must be CMYK; [`open_session`] returns
    /// [`RasterError::OutputIntent`] otherwise.  Colours not given as ink
    /// separate to the press under [`RenderParams::intent`](crate::RenderParams::intent)
    /// (else relative colorimetric) and
    /// [`RenderParams::black_point_compensation`](crate::RenderParams::black_point_compensation);
    /// overprint is simulated when [`overprint`](Self::overprint) is set.
    /// Incompatible with a CMYK [`output_profile`](Self::output_profile),
    /// and per page with CMYK output and a transparent backdrop.
    ///
    /// [`open_session`]: crate::open_session
    pub proof: Option<pdf_interp::ProofIntent>,
    /// The working space pages are composited in before the conversion to
    /// the [`output_profile`](Self::output_profile).
    ///
    /// Default `None`, automatic: the wide space
    /// ([`WorkingSpace::Wide`](pdf_interp::WorkingSpace::Wide)) for a page
    /// rendered at 16 bits under an output profile, sRGB otherwise.  The
    /// wide space has the primaries of ROMM RGB and the sRGB tone curve;
    /// colours outside sRGB keep their gamut up to the output conversion,
    /// and blending and transparency groups composite in it.  A page
    /// painted as an overprint preview is composited in sRGB.
    /// `Some(Srgb)` composites every page in sRGB; `Some(Wide)` is refused
    /// by [`open_session`] without an output profile and per page at
    /// 8 bits.
    ///
    /// [`open_session`]: crate::open_session
    pub working_space: Option<pdf_interp::WorkingSpace>,
}

impl SessionConfig {
    /// Build a config with the given backend policy and otherwise-default
    /// fields, **without** consulting `PDF_RASTER_BACKEND`.
    ///
    /// Use this when you have already resolved the policy from a more
    /// specific source (a CLI flag, a different env var, an explicit
    /// argument) and don't want `Default::default()`'s env-var lookup
    /// firing — which would emit a spurious warning if the env var were
    /// set to a malformed value, even though that value is about to be
    /// overwritten anyway.
    #[must_use]
    pub fn with_policy(policy: BackendPolicy) -> Self {
        Self {
            policy,
            vaapi_device: DEFAULT_VAAPI_DEVICE.to_owned(),
            vulkan_device: None,
            #[cfg(feature = "cache")]
            prefetch: false,
            #[cfg(feature = "cache")]
            prefetch_scale: None,
            decrypt_authorized: false,
            password: None,
            read_into_memory: false,
            strict: false,
            purpose: RenderPurpose::View,
            annotations: true,
            text_antialias: true,
            vector_antialias: true,
            overprint: false,
            font_dirs: Vec::new(),
            system_fonts: true,
            image_budget: crate::DOCUMENT_IMAGE_BUDGET_BYTES,
            max_page_bytes: crate::DEFAULT_MAX_PAGE_BYTES,
            band_bytes: crate::DEFAULT_BAND_BYTES,
            page_box: PageBox::Crop,
            rotation: Rotation::None,
            gpu_pages: false,
            output_profile: None,
            proof: None,
            working_space: None,
        }
    }

    /// The checks [`open_session`](crate::open_session) applies:
    /// [`check_gpu_pages`](Self::check_gpu_pages) and
    /// [`check_working_space`](Self::check_working_space).
    pub(crate) fn check(&self) -> Result<(), RasterError> {
        self.check_gpu_pages()?;
        self.check_working_space()
    }

    /// Refuse the wide working space without an output profile: only the
    /// conversion to one carries the wide gamut.
    ///
    /// # Errors
    ///
    /// [`RasterError::InvalidOptions`] for
    /// [`working_space`](Self::working_space) `Some(Wide)` without an
    /// [`output_profile`](Self::output_profile).
    pub fn check_working_space(&self) -> Result<(), RasterError> {
        if self.working_space == Some(pdf_interp::WorkingSpace::Wide)
            && self.output_profile.is_none()
        {
            return Err(RasterError::InvalidOptions(
                "the wide working space carries its gamut to the output profile's conversion:                  give the session an output profile (SessionConfig::output_profile, rrocket                  --output-profile)"
                    .to_owned(),
            ));
        }
        Ok(())
    }

    /// The soft proof of `doc` this config asks for, its document-level
    /// press read and tabulated; `None` without [`proof`](Self::proof).
    ///
    /// # Errors
    ///
    /// [`RasterError::InvalidOptions`] for a CMYK
    /// [`output_profile`](Self::output_profile); the errors of the
    /// document's output intent ([`RasterError::OutputIntent`]) and of
    /// its proof ([`RasterError::OutputProfile`]).
    pub(crate) fn proof_session(
        &self,
        doc: &pdf::Document,
    ) -> Result<Option<crate::proof::SessionProof>, RasterError> {
        let Some(intent) = self.proof else {
            return Ok(None);
        };
        if self
            .output_profile
            .as_ref()
            .is_some_and(|p| p.space() == pdf_interp::OutputSpace::Cmyk)
        {
            return Err(RasterError::InvalidOptions(
                "a soft proof shows the press on a display: its output profile is RGB or gray, \
                 not CMYK (for the press's own ink, render CMYK pages with the press profile \
                 as the output profile instead)"
                    .to_owned(),
            ));
        }
        crate::proof::SessionProof::open(doc, intent, self.output_profile.clone()).map(Some)
    }

    /// Check that this config can render GPU pages when it asks for them
    /// ([`gpu_pages`](Self::gpu_pages)): a build with the `vulkan` feature,
    /// [`policy`](Self::policy) [`Auto`](BackendPolicy::Auto) or
    /// [`ForceVulkan`](BackendPolicy::ForceVulkan), and
    /// [`vector_antialias`](Self::vector_antialias) on, since the device
    /// raster paints antialiased coverage only.  [`open_session`] applies
    /// the same check; calling it first refuses the combination before any
    /// document is opened.
    ///
    /// # Errors
    ///
    /// [`RasterError::InvalidOptions`] naming the conflicting fields and
    /// their `rrocket` flags (`--gpu-pages`, `--backend`, `--aaVector`).
    pub fn check_gpu_pages(&self) -> Result<(), RasterError> {
        if !self.gpu_pages {
            return Ok(());
        }
        let refuse = |why: String| {
            Err(RasterError::InvalidOptions(format!(
                "GPU page rendering (`gpu_pages`, --gpu-pages) {why}"
            )))
        };
        if !cfg!(feature = "vulkan") {
            return refuse(
                "needs a build with the `vulkan` feature; rebuild with `--features vulkan` \
                 or drop --gpu-pages"
                    .to_owned(),
            );
        }
        if !matches!(
            self.policy,
            BackendPolicy::Auto | BackendPolicy::ForceVulkan
        ) {
            return refuse(format!(
                "runs on Vulkan, which backend `{0}` (`policy`, --backend {0} or \
                 PDF_RASTER_BACKEND={0}) does not use; use --backend auto or --backend vulkan, \
                 or drop --gpu-pages",
                self.policy.name()
            ));
        }
        if !self.vector_antialias {
            return refuse(
                "paints antialiased vector paths only, and vector antialiasing is off \
                 (`vector_antialias`, --aaVector no); use --aaVector yes or drop --gpu-pages"
                    .to_owned(),
            );
        }
        Ok(())
    }

    /// The parser's view of this config's decryption and file-access
    /// fields.
    pub(crate) fn open_options(&self) -> pdf::OpenOptions {
        let mut options = pdf::OpenOptions::new(self.decrypt_authorized, self.password.clone());
        options.read_into_memory = self.read_into_memory;
        options
    }

    /// The Vulkan device selector named by the `PDF_RASTER_VULKAN_DEVICE`
    /// environment variable.
    ///
    /// `None` when the variable is unset, empty after trimming, or not
    /// UTF-8 (the last with a `log::warn!`).  Surrounding whitespace is
    /// trimmed.
    #[must_use]
    pub fn vulkan_device_from_env() -> Option<String> {
        let raw = std::env::var_os(VULKAN_DEVICE_ENV)?;
        let Some(s) = raw.to_str() else {
            log::warn!("{VULKAN_DEVICE_ENV} contains non-UTF-8 bytes; ignoring it");
            return None;
        };
        let s = s.trim();
        (!s.is_empty()).then(|| s.to_owned())
    }

    /// The font directories named by the `RROCKET_FONT_DIR` environment
    /// variable, in the platform's `PATH` syntax
    /// (`/opt/fonts:/home/me/fonts` on Unix), in order; empty when unset.
    /// Empty entries are skipped.
    #[must_use]
    pub fn font_dirs_from_env() -> Vec<PathBuf> {
        std::env::var_os(FONT_DIR_ENV).map_or_default(|value| {
            std::env::split_paths(&value)
                .filter(|dir| !dir.as_os_str().is_empty())
                .collect()
        })
    }
}

/// Environment variable read by [`SessionConfig::vulkan_device_from_env`].
const VULKAN_DEVICE_ENV: &str = "PDF_RASTER_VULKAN_DEVICE";

/// Environment variable read by [`SessionConfig::font_dirs_from_env`].
const FONT_DIR_ENV: &str = "RROCKET_FONT_DIR";

impl Default for SessionConfig {
    /// Policy from `PDF_RASTER_BACKEND` ([`BackendPolicy::from_env`]),
    /// Vulkan device from `PDF_RASTER_VULKAN_DEVICE`
    /// ([`SessionConfig::vulkan_device_from_env`]), font directories from
    /// `RROCKET_FONT_DIR` ([`SessionConfig::font_dirs_from_env`]), every
    /// other field as in [`SessionConfig::with_policy`].
    fn default() -> Self {
        let mut config = Self::with_policy(BackendPolicy::from_env());
        config.vulkan_device = Self::vulkan_device_from_env();
        config.font_dirs = Self::font_dirs_from_env();
        config
    }
}

#[cfg(test)]
mod tests {
    use super::BackendPolicy;

    #[test]
    fn backend_policy_parses_known_names() {
        use std::str::FromStr;
        assert_eq!(BackendPolicy::from_str("auto"), Ok(BackendPolicy::Auto));
        assert_eq!(BackendPolicy::from_str("cpu"), Ok(BackendPolicy::CpuOnly));
        assert_eq!(
            BackendPolicy::from_str("cuda"),
            Ok(BackendPolicy::ForceCuda)
        );
        #[cfg(feature = "vaapi")]
        assert_eq!(
            BackendPolicy::from_str("vaapi"),
            Ok(BackendPolicy::ForceVaapi)
        );
        assert_eq!(
            BackendPolicy::from_str("vulkan"),
            Ok(BackendPolicy::ForceVulkan)
        );
    }

    #[test]
    fn backend_policy_name_round_trips() {
        use std::str::FromStr;
        for &policy in BackendPolicy::ALL {
            assert_eq!(BackendPolicy::from_str(policy.name()), Ok(policy));
        }
    }

    #[test]
    fn backend_policy_empty_is_auto() {
        use std::str::FromStr;
        assert_eq!(BackendPolicy::from_str(""), Ok(BackendPolicy::Auto));
    }

    #[test]
    fn backend_policy_trims_and_ignores_case() {
        use std::str::FromStr;
        assert_eq!(
            BackendPolicy::from_str("  CUDA  "),
            Ok(BackendPolicy::ForceCuda)
        );
    }

    #[test]
    fn backend_policy_rejects_unknown_naming_the_accepted_names() {
        use std::str::FromStr;
        let err = BackendPolicy::from_str("metal").unwrap_err();
        assert_eq!(err.value(), "metal");
        let msg = err.to_string();
        assert!(
            msg.contains("\"metal\"") && msg.contains("auto, cpu, cuda"),
            "{msg}"
        );
        assert_eq!(msg.contains("vaapi"), cfg!(feature = "vaapi"), "{msg}");
    }
}
