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
/// `ForceVaapi`, and, under
/// [`Auto`](crate::BackendPolicy::Auto) with
/// [`SessionConfig::gpu_pages`], the device of the GPU pages, which a
/// session opens when a page first asks for it.  It holds nothing under
/// every other configuration.
#[must_use = "the held devices close when the hold drops"]
pub struct DeviceHold {
    #[cfg(any(feature = "gpu-icc", feature = "cache"))]
    cuda: Option<Arc<gpu::GpuCtx>>,
    #[cfg(feature = "vulkan")]
    vulkan: Option<Arc<gpu::backend::vulkan::VulkanBackend>>,
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
    /// held.
    pub fn open(config: &SessionConfig) -> Result<Self, RasterError> {
        crate::open::check_backend_compiled(config.policy)?;
        config.check()?;
        #[cfg(feature = "vulkan")]
        let vulkan = match crate::vk_init::init_vk_backend(
            config.policy,
            config.vulkan_device.as_deref(),
        )? {
            Some(backend) => Some(backend),
            None if config.gpu_pages && matches!(config.policy, crate::BackendPolicy::Auto) => {
                crate::device_pages::open_device(config.vulkan_device.as_deref())
            }
            None => None,
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
            self.vulkan.is_some().then_some("Vulkan"),
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
