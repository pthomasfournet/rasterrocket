//! Holding the GPU devices of a session configuration open across
//! sessions.

#[cfg(any(feature = "gpu-icc", feature = "cache", feature = "vulkan"))]
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
/// [`ForceVulkan`](crate::BackendPolicy::ForceVulkan), the CUDA context
/// under [`ForceCuda`](crate::BackendPolicy::ForceCuda) and
/// `ForceVaapi`, and with [`SessionConfig::gpu_pages`] the devices of the
/// GPU pages ([`SessionConfig::gpu_devices`]), which a session under
/// [`Auto`](crate::BackendPolicy::Auto) opens when a page first asks for
/// one.  It holds nothing under
/// every other configuration.
#[must_use = "the held devices close when the hold drops"]
pub struct DeviceHold {
    #[cfg(any(feature = "gpu-icc", feature = "cache"))]
    cuda: Option<Arc<gpu::GpuCtx>>,
    #[cfg(feature = "vulkan")]
    vulkan: Vec<Arc<gpu::backend::vulkan::VulkanBackend>>,
}

impl DeviceHold {
    /// Open and hold the devices a session of `config` opens.
    ///
    /// # Errors
    ///
    /// The errors [`open_session`](crate::open_session) returns for
    /// `config` before it reads the document:
    /// [`RasterError::InvalidOptions`] for an invalid configuration or an
    /// unresolvable Vulkan selector, and
    /// [`RasterError::BackendUnavailable`] for a forced backend that is not
    /// compiled in or does not open.  A GPU-page device that does not open
    /// under `Auto` is logged at `error`, as a session logs it, and is not
    /// held.  An unresolvable [`SessionConfig::gpu_devices`] entry is
    /// [`RasterError::InvalidOptions`].
    pub fn open(config: &SessionConfig) -> Result<Self, RasterError> {
        crate::open::check_backend_compiled(config.policy)?;
        config.check()?;
        #[cfg(feature = "vulkan")]
        let vulkan = {
            let session =
                crate::vk_init::init_vk_backend(config.policy, config.vulkan_device.as_deref())?;
            let mut held = crate::device_pages::open_all(config, session.as_ref())?;
            let session = session.filter(|s| !held.iter().any(|h| Arc::ptr_eq(h, s)));
            held.extend(session);
            held
        };
        Ok(Self {
            #[cfg(any(feature = "gpu-icc", feature = "cache"))]
            cuda: crate::open::init_gpu_ctx(config.policy)?,
            #[cfg(feature = "vulkan")]
            vulkan,
        })
    }

    /// Whether the hold holds no device.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.held().is_empty()
    }

    /// The backends of the devices held.
    #[cfg_attr(
        not(any(feature = "gpu-icc", feature = "cache", feature = "vulkan")),
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
        ];
        held.iter().flatten().copied().collect()
    }
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
