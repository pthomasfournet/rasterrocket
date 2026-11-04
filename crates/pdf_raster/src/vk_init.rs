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
use pdf_interp::{DeviceKernel, device_kernel_enabled};

use crate::{BackendPolicy, RasterError};

/// Whether [`BackendPolicy::Auto`] initialises the Vulkan backend: only
/// when a kernel that runs on it in production is enabled, the AA fill or
/// the parallel-Huffman JPEG decoder ([`device_kernel_enabled`]).
///
/// A Vulkan backend that wins `Auto` also keeps the CUDA context from
/// initialising, so with every Vulkan kernel disabled `Auto` falls through
/// to CUDA (whose CLUT kernel is live), then the CPU; conversely, enabling a
/// Vulkan kernel turns the CUDA paths (CLUT, device image cache) off under
/// `Auto`.
fn auto_wants_vulkan() -> bool {
    device_kernel_enabled(DeviceKernel::AaFill) || device_kernel_enabled(DeviceKernel::HuffmanJpeg)
}

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
/// touching Vulkan, and so does `Auto` while every Vulkan kernel is dormant
/// (see [`auto_wants_vulkan`]).  An unavailable Vulkan is `Ok(None)` under `Auto` (the
/// caller falls through to CUDA, then CPU) and an error under
/// `ForceVulkan`.  A selector that does not resolve is an error whenever
/// Vulkan is initialised, under either policy, since it is a configuration
/// mistake rather than missing hardware; under `Auto` with every Vulkan
/// kernel dormant it is still resolved against the enumerated devices, but
/// no device is opened.
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
    if matches!(policy, BackendPolicy::Auto) && !auto_wants_vulkan() {
        log::debug!("rasterrocket: every Vulkan kernel is dormant; Auto leaves Vulkan idle");
        if let Some(selector) = selector
            && let Err(InitError::Selection(msg)) = checked_selector(selector)
        {
            return Err(RasterError::InvalidOptions(msg));
        }
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

/// Selectors already checked by [`checked_selector`], with the outcome.
static CHECKED: Mutex<Vec<(String, Result<(), InitError>)>> = Mutex::new(Vec::new());

/// Check `selector` against the enumerated devices without opening one
/// ([`VulkanBackend::check_selector`]), once per selector per process.  An
/// unavailable Vulkan is not a selection error: `Auto` falls through then
/// as it does when it opens Vulkan.
fn checked_selector(selector: &str) -> Result<(), InitError> {
    let mut cache = CHECKED.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((_, result)) = cache.iter().find(|(key, _)| key == selector) {
        return result.clone();
    }
    let result = parse_selector(Some(selector))
        .and_then(|parsed| {
            VulkanBackend::check_selector(&parsed.expect("Some selector parses to Some"))
        })
        .map_err(InitError::from);
    cache.push((selector.to_owned(), result.clone()));
    result
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

#[cfg(test)]
mod tests {
    use super::init_vk_backend;
    use crate::{BackendPolicy, RasterError};

    /// A selector naming no device is a configuration error under `Auto`
    /// even while every Vulkan kernel is dormant (the default thresholds of
    /// this test process) and no device is opened.
    #[test]
    fn auto_rejects_an_unresolvable_selector_while_vulkan_is_idle() {
        const SELECTOR: &str = "no-such-gpu-7f3c";
        match init_vk_backend(BackendPolicy::Auto, Some(SELECTOR)) {
            Err(RasterError::InvalidOptions(msg)) => {
                assert!(
                    msg.contains(SELECTOR),
                    "message does not name the selector: {msg}"
                );
            }
            Ok(None) if gpu::backend::vulkan::VulkanBackend::new().is_err() => {
                eprintln!("skipping: no Vulkan device to resolve against");
            }
            Ok(backend) => panic!(
                "expected InvalidOptions, got Ok (backend opened: {})",
                backend.is_some()
            ),
            Err(other) => panic!("expected InvalidOptions, got {other:?}"),
        }
    }
}
