//! [`BackendPolicy`]: which compute backend a session uses, and its names.

#[cfg(doc)]
use crate::{RasterError, SessionConfig};

/// Controls which compute backend is used for image decoding and GPU kernels.
///
/// The default is [`Auto`](BackendPolicy::Auto), which renders on the CPU
/// and opens no Vulkan device or CUDA context at start-up.  The `Force*`
/// variants select a device backend and turn silent fallbacks into hard
/// errors so you can tell immediately whether the expected hardware path is
/// actually being taken.
///
/// At runtime, [`SessionConfig::default()`] resolves the policy via
/// [`BackendPolicy::from_env`], so users can switch backends per process by
/// setting `PDF_RASTER_BACKEND={auto,cpu,cuda,vaapi,vulkan,d3d12}` without
/// rebuilding.  [`SessionConfig::with_policy`] bypasses the env var.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum BackendPolicy {
    /// Auto: the CPU.  The session initialises no Vulkan device, CUDA
    /// context, image cache or nvJPEG decoder at start-up.  With the
    /// `nvjpeg2k` feature and a finite `PDF_RASTER_JPX_THRESHOLD`, a
    /// rendering thread builds its nvJPEG2000 decoder when it first draws a
    /// `JPXDecode` image the decoder takes at more than half its size on
    /// some axis; smaller draws decode on the CPU.  Its default
    /// threshold disables the decoder.
    #[default]
    Auto,
    /// CPU only.  All GPU init is skipped; no CUDA or VA-API calls are made.
    CpuOnly,
    /// Require CUDA (nvJPEG2000 for images of at least 2²¹ pixels, ICC CLUT,
    /// image blits, and nvJPEG while its threshold enables it).
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
    /// Require the Direct3D 12 compute backend (Windows).  Returns
    /// [`RasterError::BackendUnavailable`] if no D3D12 adapter opens, or the
    /// binary was built without `--features d3d12` or for another platform.
    ///
    /// The session's device is the adapter
    /// [`SessionConfig::d3d12_device`] names, or the automatically chosen
    /// hardware adapter.  On the adapter run the ICC CMYK→RGB CLUT kernel
    /// (with the `gpu-icc` feature, once the device passes the same
    /// exactness check as a Vulkan device; images it does not convert run
    /// on the CPU), the device-resident image cache and the image blit
    /// kernel that draws from it (with the `cache` feature), and GPU pages
    /// (with [`SessionConfig::gpu_pages`]).  Every other stage runs on the
    /// CPU.
    ForceD3d12,
}

impl BackendPolicy {
    /// Resolve a policy from the `PDF_RASTER_BACKEND` environment variable.
    ///
    /// Convenience wrapper for [`BackendPolicy::from_env_var`] with the
    /// canonical name.  Accepted values (case-insensitive): `auto`, `cpu`,
    /// `cuda`, `vulkan`, `d3d12`, and `vaapi` in a build with the `vaapi`
    /// feature.
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
        Self::ForceD3d12,
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
    /// accept for this policy: `auto`, `cpu`, `cuda`, `vaapi`, `vulkan` or
    /// `d3d12`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::CpuOnly => "cpu",
            Self::ForceCuda => "cuda",
            #[cfg(feature = "vaapi")]
            Self::ForceVaapi => "vaapi",
            Self::ForceVulkan => "vulkan",
            Self::ForceD3d12 => "d3d12",
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
