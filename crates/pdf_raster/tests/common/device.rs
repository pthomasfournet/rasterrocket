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
