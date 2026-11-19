//! Holding the GPU devices of a session configuration open across
//! sessions.

#[cfg(any(
    feature = "gpu-icc",
    feature = "cache",
    feature = "vulkan",
    all(windows, feature = "d3d12")
))]
use std::sync::Arc;

use crate::{RasterError, SessionConfig};

/// The GPU devices a [`SessionConfig`] opens, held open while the hold
/// lives.
///
/// A session holds its devices only while it lives: the last session on a
/// device destroys it, and the next session builds it again.  Every session
/// opened while a hold lives, under a configuration naming the same
/// devices, shares the held ones instead; the devices are destroyed once
/// the hold and every such session have dropped.
///
/// A hold of a configuration opens what a session of it opens: the Vulkan
/// device of [`SessionConfig::vulkan_device`] under
/// [`ForceVulkan`](crate::BackendPolicy::ForceVulkan), the Direct3D 12
/// adapter of [`SessionConfig::d3d12_device`] under
/// [`ForceD3d12`](crate::BackendPolicy::ForceD3d12), the CUDA context
/// under [`ForceCuda`](crate::BackendPolicy::ForceCuda) and
/// `ForceVaapi`, and with [`SessionConfig::gpu_pages`] the devices of the
/// GPU pages ([`SessionConfig::gpu_devices`]) on the API of the policy,
/// which a session under [`Auto`](crate::BackendPolicy::Auto) opens when a
/// page first asks for one.  It holds nothing under
/// every other configuration.
#[must_use = "the held devices close when the hold drops"]
pub struct DeviceHold {
    #[cfg(any(feature = "gpu-icc", feature = "cache"))]
    cuda: Option<Arc<gpu::GpuCtx>>,
    #[cfg(feature = "vulkan")]
    vulkan: Vec<Arc<gpu::backend::vulkan::VulkanBackend>>,
    #[cfg(all(windows, feature = "d3d12"))]
    d3d12: Vec<Arc<gpu::backend::d3d12::D3d12Backend>>,
}

impl DeviceHold {
    /// Open and hold the devices a session of `config` opens.
    ///
    /// # Errors
    ///
    /// The errors [`open_session`](crate::open_session) returns for
    /// `config` before it reads the document:
    /// [`RasterError::InvalidOptions`] for an invalid configuration or an
    /// unresolvable Vulkan or Direct3D 12 selector, and
    /// [`RasterError::BackendUnavailable`] for a forced backend that is not
    /// compiled in or does not open.  A GPU-page device that does not open
    /// under `Auto` is logged at `error`, as a session logs it, and is not
    /// held.  An unresolvable [`SessionConfig::gpu_devices`] entry is
    /// [`RasterError::InvalidOptions`].
    pub fn open(config: &SessionConfig) -> Result<Self, RasterError> {
        crate::open::check_backend_compiled(config.policy)?;
        config.check()?;
        #[cfg(feature = "vulkan")]
        let vulkan = held_devices(config)?;
        #[cfg(all(windows, feature = "d3d12"))]
        let d3d12 = held_devices(config)?;
        Ok(Self {
            #[cfg(any(feature = "gpu-icc", feature = "cache"))]
            cuda: crate::open::init_gpu_ctx(config.policy)?,
            #[cfg(feature = "vulkan")]
            vulkan,
            #[cfg(all(windows, feature = "d3d12"))]
            d3d12,
        })
    }

    /// Whether the hold holds no device.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.held().is_empty()
    }

    /// The backends of the devices held.
    #[cfg_attr(
        not(any(
            feature = "gpu-icc",
            feature = "cache",
            feature = "vulkan",
            all(windows, feature = "d3d12")
        )),
        expect(
            clippy::unused_self,
            reason = "no device is held without a GPU feature"
        )
    )]
    fn held(&self) -> Vec<&'static str> {
        let held: &[Option<&'static str>] = &[
            #[cfg(any(feature = "gpu-icc", feature = "cache"))]
            self.cuda.is_some().then_some("CUDA"),
            #[cfg(feature = "vulkan")]
            (!self.vulkan.is_empty()).then_some("Vulkan"),
            #[cfg(all(windows, feature = "d3d12"))]
            (!self.d3d12.is_empty()).then_some("Direct3D 12"),
        ];
        held.iter().flatten().copied().collect()
    }
}

/// The devices of `B` a session of `config` opens: its own device and
/// its GPU page devices, each once.
///
/// # Errors
///
/// As [`crate::device_init::init_backend`] and
/// [`crate::device_pages::open_all`].
#[cfg(any(feature = "vulkan", all(windows, feature = "d3d12")))]
fn held_devices<B: crate::device_init::SessionDevice>(
    config: &SessionConfig,
) -> Result<Vec<Arc<B>>, RasterError> {
    let session =
        crate::device_init::init_backend::<B>(config.policy, B::session_selector(config))?;
    let mut held = crate::device_pages::open_all(config, session.as_ref())?;
    let session = session.filter(|s| !held.iter().any(|h| Arc::ptr_eq(h, s)));
    held.extend(session);
    Ok(held)
}

impl std::fmt::Debug for DeviceHold {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceHold")
            .field("held", &self.held())
            .finish()
    }
}

#[cfg(all(test, feature = "vulkan"))]
mod tests {
    use gpu::backend::page_device::PageDevice;

    use super::DeviceHold;
    use crate::{BackendPolicy, SessionConfig};

    /// Under `ForceVulkan` with every GPU listed, the hold holds one
    /// backend per device: the session's own device is not opened a second
    /// time under its index.
    #[test]
    fn a_hold_opens_each_gpu_page_device_once() {
        let mut config = SessionConfig::with_policy(BackendPolicy::ForceVulkan);
        config.gpu_pages = true;
        config.gpu_devices = vec!["all".to_owned()];
        let hold = match DeviceHold::open(&config) {
            Ok(hold) => hold,
            Err(e) => {
                eprintln!("skipping: no Vulkan device: {e}");
                return;
            }
        };
        let mut indices: Vec<usize> = hold.vulkan.iter().map(|b| b.device_index()).collect();
        let count = indices.len();
        indices.sort_unstable();
        indices.dedup();
        assert_eq!(indices.len(), count, "a device is held twice");
    }
}

#[cfg(all(test, windows, feature = "d3d12"))]
mod d3d12_tests {
    use super::DeviceHold;
    use crate::{BackendPolicy, RasterError, SessionConfig};

    /// Under `ForceD3d12` the hold holds the adapter of `d3d12_device`;
    /// one naming no adapter is refused as an invalid option.
    #[test]
    fn a_hold_holds_the_d3d12_adapter() {
        let mut config = SessionConfig::with_policy(BackendPolicy::ForceD3d12);
        config.d3d12_device = SessionConfig::d3d12_device_from_env();
        match DeviceHold::open(&config) {
            Ok(hold) => assert_eq!(hold.held(), ["Direct3D 12"]),
            Err(e) => {
                assert!(
                    !std::env::var("PDF_RASTER_REQUIRE_D3D12").is_ok_and(|v| v == "1"),
                    "PDF_RASTER_REQUIRE_D3D12=1, but no D3D12 adapter opens: {e}"
                );
                eprintln!("skipping: no D3D12 adapter: {e}");
            }
        }
        config.d3d12_device = Some("no-such-gpu-7f3c".to_owned());
        assert!(matches!(
            DeviceHold::open(&config),
            Err(RasterError::InvalidOptions(_))
        ));
    }
}
