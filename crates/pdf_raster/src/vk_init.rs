//! Process-wide Vulkan compute backends, one per device selector.
//!
//! A backend owns a logical device and its compiled pipelines, so it is
//! built once per distinct [`SessionConfig::vulkan_device`] value and shared
//! by every session that names the same selector.  Construction results,
//! failures included, are retained: repeated sessions neither rebuild the
//! device nor retry an init that already failed.
//!
//! [`SessionConfig::vulkan_device`]: crate::SessionConfig::vulkan_device

use std::sync::{Arc, Mutex, PoisonError};

use gpu::backend::BackendError;
use gpu::backend::vulkan::{DeviceSelector, VulkanBackend};

use crate::{BackendPolicy, RasterError};

/// Why a backend could not be constructed.
#[derive(Debug, Clone)]
enum InitError {
    /// The configured selector did not resolve to one usable device.
    /// Fatal under every policy that initialises Vulkan.
    Selection(String),
    /// Vulkan is absent or failed to initialise.  Fatal only under
    /// [`BackendPolicy::ForceVulkan`].
    Unavailable(String),
}

impl From<BackendError> for InitError {
    fn from(e: BackendError) -> Self {
        match e {
            BackendError::DeviceSelection(msg) => Self::Selection(msg),
            other => Self::Unavailable(other.to_string()),
        }
    }
}

type Entry = (Option<String>, Result<Arc<VulkanBackend>, InitError>);

/// Constructed backends keyed by selector; `None` keys the automatically
/// ranked device.  A linear scan suffices: a process names a handful of
/// selectors at most.
static BACKENDS: Mutex<Vec<Entry>> = Mutex::new(Vec::new());

/// Return the Vulkan backend for `selector` under `policy`.
///
/// Policies other than `Auto` and `ForceVulkan` return `Ok(None)` without
/// touching Vulkan.  An unavailable Vulkan is `Ok(None)` under `Auto` (the
/// caller falls through to CUDA, then CPU) and an error under
/// `ForceVulkan`.  A selector that does not resolve is an error under both,
/// since it is a configuration mistake rather than missing hardware.
///
/// # Errors
///
/// [`RasterError::InvalidOptions`] for an unresolvable selector;
/// [`RasterError::BackendUnavailable`] for an unavailable Vulkan under
/// `ForceVulkan`.
pub(crate) fn init_vk_backend(
    policy: BackendPolicy,
    selector: Option<&str>,
) -> Result<Option<Arc<VulkanBackend>>, RasterError> {
    if !matches!(policy, BackendPolicy::Auto | BackendPolicy::ForceVulkan) {
        return Ok(None);
    }
    match cached_backend(selector) {
        Ok(b) => Ok(Some(b)),
        Err(InitError::Selection(msg)) => Err(RasterError::InvalidOptions(msg)),
        Err(InitError::Unavailable(e)) => {
            if matches!(policy, BackendPolicy::ForceVulkan) {
                Err(RasterError::BackendUnavailable(format!(
                    "Vulkan backend required but unavailable: {e}. \
                     Verify with `vulkaninfo` that a Vulkan 1.3+ device is present."
                )))
            } else {
                log::debug!("rasterrocket: Vulkan unavailable under Auto ({e}); trying CUDA next");
                Ok(None)
            }
        }
    }
}

/// Parse `selector` into the form [`VulkanBackend::with_selector`] takes.
///
/// # Errors
///
/// [`BackendError::DeviceSelection`] for an empty selector.
pub(crate) fn parse_selector(
    selector: Option<&str>,
) -> Result<Option<DeviceSelector>, BackendError> {
    selector.map(str::parse).transpose()
}

/// Look up or construct the backend for `selector`.  The lock is held
/// across construction so concurrent first sessions build one device.
fn cached_backend(selector: Option<&str>) -> Result<Arc<VulkanBackend>, InitError> {
    let mut cache = BACKENDS.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((_, result)) = cache.iter().find(|(key, _)| key.as_deref() == selector) {
        return result.clone();
    }
    let result = construct(selector);
    cache.push((selector.map(str::to_owned), result.clone()));
    result
}

fn construct(selector: Option<&str>) -> Result<Arc<VulkanBackend>, InitError> {
    let parsed = parse_selector(selector)?;
    Ok(Arc::new(VulkanBackend::with_selector(parsed.as_ref())?))
}
