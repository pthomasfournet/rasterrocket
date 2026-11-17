//! Process-wide Vulkan compute backends, one per device selector.
//!
//! A backend owns a logical device and its compiled pipelines, so it is
//! built once per distinct [`SessionConfig::vulkan_device`] value and shared
//! by every session alive at once that names the same selector.  The
//! registry holds a backend weakly: it is destroyed, its device and instance
//! with it, when the last session holding it drops, and a later session
//! builds it anew.  Failures are retained: repeated sessions do not retry an
//! init that already failed.
//!
//! [`SessionConfig::vulkan_device`]: crate::SessionConfig::vulkan_device

use std::sync::{Arc, Mutex, PoisonError, Weak};

use crate::{BackendPolicy, RasterError};
use gpu::backend::BackendError;
use gpu::backend::vulkan::{DeviceSelector, VulkanBackend};

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

type Entry = (Option<String>, Result<Weak<VulkanBackend>, InitError>);

/// Constructed backends keyed by selector, held weakly, and the selectors
/// whose construction failed; `None` keys the automatically ranked device.
/// A linear scan suffices: a process names a handful of selectors at most.
static BACKENDS: Mutex<Vec<Entry>> = Mutex::new(Vec::new());

/// Return the Vulkan backend for `selector` under `policy`.
///
/// Only `ForceVulkan` opens a device; every other policy returns `Ok(None)`
/// without opening one.  A selector that does not resolve is an error under
/// `ForceVulkan` and `Auto`, since it is a configuration mistake rather than
/// missing hardware; under `Auto` it is resolved against the enumerated
/// devices only.  An unavailable Vulkan is an error under `ForceVulkan`.
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
    match policy {
        BackendPolicy::ForceVulkan => {}
        BackendPolicy::Auto => {
            if let Some(selector) = selector
                && let Err(InitError::Selection(msg)) = checked_selector(selector)
            {
                return Err(RasterError::InvalidOptions(msg));
            }
            return Ok(None);
        }
        _ => return Ok(None),
    }
    match cached_backend(selector) {
        Ok(b) => Ok(Some(b)),
        Err(InitError::Selection(msg)) => Err(RasterError::InvalidOptions(msg)),
        Err(InitError::Unavailable(e)) => Err(RasterError::BackendUnavailable(format!(
            "Vulkan backend required but unavailable: {e}. \
             Verify with `vulkaninfo` that a Vulkan 1.3+ device is present."
        ))),
    }
}

/// Parse `selector` into the form [`VulkanBackend::with_selector`] takes.
///
/// # Errors
///
/// [`BackendError::DeviceSelection`] for an empty selector.
fn parse_selector(selector: Option<&str>) -> Result<Option<DeviceSelector>, BackendError> {
    selector.map(str::parse).transpose()
}

/// Selectors already checked by [`checked_selector`], with the outcome.
static CHECKED: Mutex<Vec<(String, Result<(), InitError>)>> = Mutex::new(Vec::new());

/// Check `selector` against the enumerated devices without opening one
/// ([`VulkanBackend::check_selector`]), once per selector per process.  An
/// unavailable Vulkan is not a selection error: `Auto` falls through then
/// as it does when it opens Vulkan.
fn checked_selector(selector: &str) -> Result<(), InitError> {
    memo(&CHECKED, selector.to_owned(), || {
        selector
            .parse::<DeviceSelector>()
            .and_then(|parsed| VulkanBackend::check_selector(&parsed))
            .map_err(InitError::from)
    })
}

/// The backend for `selector`, opened unless a live session holds it and
/// shared by every session naming the same selector, under any policy.
///
/// # Errors
///
/// Why the device did not open: a selector that does not resolve, or an
/// unavailable Vulkan.
pub(crate) fn open_backend(selector: Option<&str>) -> Result<Arc<VulkanBackend>, String> {
    cached_backend(selector).map_err(|e| match e {
        InitError::Selection(msg) | InitError::Unavailable(msg) => msg,
    })
}

/// The live backend for `selector`, or the failure of its construction,
/// or else a backend constructed now.  The lock is held across
/// construction so concurrent first sessions build one device.
fn cached_backend(selector: Option<&str>) -> Result<Arc<VulkanBackend>, InitError> {
    let mut cache = BACKENDS.lock().unwrap_or_else(PoisonError::into_inner);
    let key = selector.map(str::to_owned);
    if let Some(at) = cache.iter().position(|(k, _)| *k == key) {
        match &cache[at].1 {
            Ok(backend) => {
                if let Some(backend) = backend.upgrade() {
                    return Ok(backend);
                }
                drop(cache.swap_remove(at));
            }
            Err(e) => return Err(e.clone()),
        }
    }
    let backend = construct(selector);
    cache.push((
        key,
        backend.as_ref().map(Arc::downgrade).map_err(Clone::clone),
    ));
    backend
}

/// The value `cache` holds for `key`, computed by `make` and retained on
/// first use.  The lock is held across `make`, so concurrent first callers
/// compute it once; a poisoned lock is taken over, since every entry is
/// pushed whole.
fn memo<K: PartialEq, T: Clone>(cache: &Mutex<Vec<(K, T)>>, key: K, make: impl FnOnce() -> T) -> T {
    let mut cache = cache.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((_, value)) = cache.iter().find(|(k, _)| *k == key) {
        return value.clone();
    }
    let value = make();
    cache.push((key, value.clone()));
    value
}

fn construct(selector: Option<&str>) -> Result<Arc<VulkanBackend>, InitError> {
    let parsed = parse_selector(selector)?;
    Ok(Arc::new(VulkanBackend::with_selector(parsed.as_ref())?))
}

#[cfg(test)]
mod tests {
    use super::init_vk_backend;
    use crate::{BackendPolicy, RasterError};

    /// A selector naming no device is a configuration error under `Auto`,
    /// which opens no device.
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
