//! Backend selection policy and session configuration.

#[cfg(doc)]
use crate::{PageDiagnostics, RasterError, RasterSession, open_session, open_session_from_bytes};

use std::path::PathBuf;

use crate::{Password, RenderPurpose};

// ── Backend policy ────────────────────────────────────────────────────────────

/// Controls which compute backend is used for image decoding and GPU fills.
///
/// The default is [`Auto`](BackendPolicy::Auto), which prefers Vulkan when
/// compiled in, falls through to CUDA when Vulkan is unavailable, and finally
/// to the CPU paths.  The `Force*` variants turn silent fallbacks into hard
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
    /// Auto: prefer Vulkan when present, fall through to CUDA, then CPU.
    /// Silent fallback at every step.
    #[default]
    Auto,
    /// CPU only.  All GPU init is skipped; no CUDA or VA-API calls are made.
    CpuOnly,
    /// Require CUDA (nvJPEG, nvJPEG2000, GPU AA fill, ICC CLUT).
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
    /// AA fill and tile fill kernels dispatch through `VulkanBackend`;
    /// the device-resident image cache (`DeviceImageCache`,
    /// `DevicePageBuffer`) is CUDA-only, so under `ForceVulkan` JPEG
    /// images decode and composite on the CPU, and ICC CMYK→RGB runs on
    /// the CPU as well.
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
    /// devices, under `Auto` as well as `ForceVulkan`.
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
    /// Whether an image or a content stream that cannot be decoded, or
    /// decodes only in part, fails its page.
    ///
    /// Default `false`: the image or stream is skipped (or drawn from the
    /// data that decoded), the page renders, a warning naming the page and
    /// the object is logged, and
    /// the failure is reported in [`PageDiagnostics::image_failures`] or
    /// [`PageDiagnostics::content_failures`].  When `true` the page returns
    /// [`RasterError::ImageDecodeFailed`] or
    /// [`RasterError::ContentDecodeFailed`] instead.
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
            decrypt_authorized: false,
            password: None,
            strict: false,
            purpose: RenderPurpose::View,
            annotations: true,
            text_antialias: true,
            vector_antialias: true,
            font_dirs: Vec::new(),
            system_fonts: true,
        }
    }

    /// The parser's view of this config's decryption fields.
    pub(crate) fn open_options(&self) -> pdf::OpenOptions {
        pdf::OpenOptions::new(self.decrypt_authorized, self.password.clone())
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
        std::env::var_os(FONT_DIR_ENV)
            .map(|value| {
                std::env::split_paths(&value)
                    .filter(|dir| !dir.as_os_str().is_empty())
                    .collect()
            })
            .unwrap_or_default()
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
