//! Skipping a test for want of a GPU, unless the run requires one.

/// A GPU API a test opens a session or device on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceApi {
    /// Vulkan; required by `PDF_RASTER_REQUIRE_VULKAN=1`.
    Vulkan,
    /// Direct3D 12; required by `PDF_RASTER_REQUIRE_D3D12=1`.
    D3d12,
}

impl DeviceApi {
    /// The variable whose value `1` turns a skip into a failure.
    pub const fn require_var(self) -> &'static str {
        match self {
            Self::Vulkan => "PDF_RASTER_REQUIRE_VULKAN",
            Self::D3d12 => "PDF_RASTER_REQUIRE_D3D12",
        }
    }

    /// The API's name in messages.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Vulkan => "Vulkan",
            Self::D3d12 => "D3D12",
        }
    }

    /// Whether the run requires the API: [`Self::require_var`] is `1`.
    pub fn required(self) -> bool {
        std::env::var(self.require_var()).is_ok_and(|v| v.trim() == "1")
    }
}

/// The value of `opened`, or `None` (the caller skips) with a line on
/// stderr naming `api` and the error.
///
/// # Panics
///
/// When `opened` is an error and the run requires `api`
/// ([`DeviceApi::required`]).
pub fn or_skip<T, E: std::fmt::Display>(api: DeviceApi, opened: Result<T, E>) -> Option<T> {
    match opened {
        Ok(value) => Some(value),
        Err(e) => {
            assert!(
                !api.required(),
                "{}=1, but no {} device opens: {e}",
                api.require_var(),
                api.name()
            );
            eprintln!("skipping: no {} device ({e})", api.name());
            None
        }
    }
}

/// The variable naming the policy of the GPU-page tests:
/// `RR_GPU_PAGE_POLICY`.
pub const GPU_PAGE_POLICY_VAR: &str = "RR_GPU_PAGE_POLICY";

/// The forcing policy the GPU-page tests run under: the policy
/// [`GPU_PAGE_POLICY_VAR`] names (`vulkan` or `d3d12`), `ForceVulkan`
/// when it is unset or empty.
///
/// # Panics
///
/// When the variable names a policy that renders no GPU pages under
/// force, or no policy.
pub fn gpu_page_policy() -> rasterrocket::BackendPolicy {
    use rasterrocket::BackendPolicy;
    let value = std::env::var(GPU_PAGE_POLICY_VAR).unwrap_or_default();
    if value.trim().is_empty() {
        return BackendPolicy::ForceVulkan;
    }
    match value.parse() {
        Ok(policy @ (BackendPolicy::ForceVulkan | BackendPolicy::ForceD3d12)) => policy,
        _ => panic!("{GPU_PAGE_POLICY_VAR}={value:?}: expected `vulkan` or `d3d12`"),
    }
}

impl DeviceApi {
    /// The API the GPU pages of a session under `policy` render on.
    pub const fn of_policy(policy: rasterrocket::BackendPolicy) -> Option<Self> {
        use rasterrocket::BackendPolicy;
        match policy {
            BackendPolicy::Auto | BackendPolicy::ForceVulkan => Some(Self::Vulkan),
            BackendPolicy::ForceD3d12 => Some(Self::D3d12),
            _ => None,
        }
    }

    /// Set `config`'s device selector of the API to the one its
    /// environment variable (`PDF_RASTER_VULKAN_DEVICE`,
    /// `PDF_RASTER_D3D12_DEVICE`) names.
    pub fn select_from_env(self, config: &mut rasterrocket::SessionConfig) {
        use rasterrocket::SessionConfig;
        match self {
            Self::Vulkan => config.vulkan_device = SessionConfig::vulkan_device_from_env(),
            Self::D3d12 => config.d3d12_device = SessionConfig::d3d12_device_from_env(),
        }
    }
}

/// [`or_skip`] for a GPU-page session under `policy`: a failure is a skip
/// unless the run requires the API the policy renders on
/// (`PDF_RASTER_REQUIRE_VULKAN` for Vulkan, `PDF_RASTER_REQUIRE_D3D12`
/// for Direct3D 12).
///
/// # Panics
///
/// When `opened` is an error and the run requires the policy's API, or
/// the policy renders no GPU pages.
pub fn gpu_pages_or_skip<T, E: std::fmt::Display>(
    policy: rasterrocket::BackendPolicy,
    opened: Result<T, E>,
) -> Option<T> {
    let api =
        DeviceApi::of_policy(policy).unwrap_or_else(|| panic!("{policy:?} renders no GPU pages"));
    or_skip(api, opened)
}
