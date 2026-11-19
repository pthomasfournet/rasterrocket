//! Process-wide compute backends, one per API and device selector.
//!
//! A backend owns a logical device and its compiled pipelines, so it is
//! built once per distinct selector ([`SessionConfig::vulkan_device`] for
//! Vulkan) and shared by every session alive at once that names the same
//! selector.  The registry of each API ([`SessionDevice::registry`]) holds a
//! backend weakly: it is destroyed, its device and instance with it, when
//! the last session holding it drops, and a later session builds it anew.
//! Failures are retained: repeated sessions do not retry an init that
//! already failed.
//!
//! [`SessionConfig::vulkan_device`]: crate::SessionConfig::vulkan_device

use std::sync::{Arc, Mutex, PoisonError};

use crate::registry::{Registry, shared};
use crate::{BackendPolicy, RasterError};
use gpu::backend::BackendError;
use gpu::backend::page_device::PageDevice;
use gpu::backend::select::DeviceSelector;
use gpu::backend::vulkan::VulkanBackend;

/// Why a backend could not be constructed.
#[derive(Debug, Clone)]
pub(crate) enum InitError {
    /// The configured selector did not resolve to one usable device.
    /// Fatal under every policy that initialises the backend.
    Selection(String),
    /// The API is absent or failed to initialise.  Fatal only under the
    /// backend's forcing policy ([`SessionDevice::FORCE`]).
    Unavailable(String),
}

impl From<BackendError> for InitError {
    fn from(e: BackendError) -> Self {
        match e {
            BackendError::DeviceSelection { message, .. } => Self::Selection(message),
            other => Self::Unavailable(other.to_string()),
        }
    }
}

/// The device indices a device list resolves to, or why it does not.
pub(crate) type Resolution = Result<Vec<usize>, InitError>;

/// A page device a session opens: its process-wide state and the words
/// its errors use.
pub(crate) trait SessionDevice: PageDevice + Send + Sync + 'static {
    /// Constructed backends keyed by selector, held weakly, and the
    /// selectors whose construction failed; `None` keys the automatically
    /// ranked device.
    fn registry() -> &'static Registry<Option<String>, Self, InitError>;

    /// Device lists already resolved by [`resolve_page_devices`], with the
    /// outcome.
    fn resolved() -> &'static Mutex<Vec<(Vec<String>, Resolution)>>;

    /// The API's name, as errors and logs give it.
    const API: &'static str;

    /// How to check that a device of the API is present, as the error of
    /// an unavailable API under [`Self::FORCE`] advises.
    const PROBE_HINT: &'static str;

    /// The policy that requires the backend and opens its device with the
    /// session.
    const FORCE: BackendPolicy;
}

impl SessionDevice for VulkanBackend {
    fn registry() -> &'static Registry<Option<String>, Self, InitError> {
        static BACKENDS: Registry<Option<String>, VulkanBackend, InitError> =
            Mutex::new(Vec::new());
        &BACKENDS
    }

    fn resolved() -> &'static Mutex<Vec<(Vec<String>, Resolution)>> {
        static RESOLVED: Mutex<Vec<(Vec<String>, Resolution)>> = Mutex::new(Vec::new());
        &RESOLVED
    }

    const API: &'static str = "Vulkan";
    const PROBE_HINT: &'static str =
        "Verify with `vulkaninfo` that a Vulkan 1.3+ device is present.";
    const FORCE: BackendPolicy = BackendPolicy::ForceVulkan;
}

/// The error of an unavailable API under [`SessionDevice::FORCE`].
fn unavailable<B: SessionDevice>(e: &str) -> RasterError {
    RasterError::BackendUnavailable(format!(
        "{} backend required but unavailable: {e}. {}",
        B::API,
        B::PROBE_HINT
    ))
}

/// Return the backend `B` for `selector` under `policy`.
///
/// Only [`SessionDevice::FORCE`] opens a device; every other policy returns
/// `Ok(None)` without opening one.  A selector that does not resolve is an
/// error under `FORCE` and `Auto`, since it is a configuration mistake
/// rather than missing hardware; under `Auto` it is resolved against the
/// enumerated devices only.  An unavailable API is an error under `FORCE`.
///
/// # Errors
///
/// [`RasterError::InvalidOptions`] for an unresolvable selector;
/// [`RasterError::BackendUnavailable`] for an unavailable API under
/// `FORCE`.
pub(crate) fn init_backend<B: SessionDevice>(
    policy: BackendPolicy,
    selector: Option<&str>,
) -> Result<Option<Arc<B>>, RasterError> {
    if policy == BackendPolicy::Auto {
        if let Some(selector) = selector
            && let Err(InitError::Selection(msg)) = checked_selector::<B>(selector)
        {
            return Err(RasterError::InvalidOptions(msg));
        }
        return Ok(None);
    }
    if policy != B::FORCE {
        return Ok(None);
    }
    match cached_backend::<B>(selector) {
        Ok(b) => Ok(Some(b)),
        Err(InitError::Selection(msg)) => Err(RasterError::InvalidOptions(msg)),
        Err(InitError::Unavailable(e)) => Err(unavailable::<B>(&e)),
    }
}

/// Parse `selector` into the form [`PageDevice::open`] takes.
///
/// # Errors
///
/// [`BackendError::DeviceSelection`] for an empty selector.
fn parse_selector(selector: Option<&str>) -> Result<Option<DeviceSelector>, BackendError> {
    selector.map(str::parse).transpose()
}

/// Selectors already checked by [`checked_selector`], keyed by API
/// ([`SessionDevice::API`]) and selector, with the outcome.
type Checked = Vec<((&'static str, String), Result<(), InitError>)>;
static CHECKED: Mutex<Checked> = Mutex::new(Vec::new());

/// Check `selector` against the enumerated devices of `B` without opening
/// one ([`PageDevice::check_selector`]), once per API and selector per
/// process.  An unavailable API is not a selection error: `Auto` falls
/// through then as it does when it opens the backend.
fn checked_selector<B: SessionDevice>(selector: &str) -> Result<(), InitError> {
    memo(&CHECKED, (B::API, selector.to_owned()), || {
        selector
            .parse::<DeviceSelector>()
            .and_then(|parsed| B::check_selector(&parsed))
            .map_err(InitError::from)
    })
}

/// The selectors of the GPU page devices `list` names
/// ([`crate::SessionConfig::gpu_devices`], not empty): each device's
/// enumeration index, which names it for the rest of the process, resolved
/// once per list per process without opening a device.  `all` is every
/// usable GPU of `B`.  Under `Auto` an unavailable API is not an error: the
/// entries are returned as given (`all` as the automatically ranked
/// device), and their opening fails, logged, when a page asks for them.
///
/// # Errors
///
/// [`RasterError::InvalidOptions`] listing the enumerated devices when an
/// entry does not resolve, two name the same device, or `all` finds no
/// GPU; [`RasterError::BackendUnavailable`] for an unavailable API under
/// [`SessionDevice::FORCE`].
pub(crate) fn resolve_page_devices<B: SessionDevice>(
    policy: BackendPolicy,
    list: &[String],
) -> Result<Vec<Option<String>>, RasterError> {
    let all = list.len() == 1 && crate::config::is_all(&list[0]);
    let outcome = memo(B::resolved(), list.to_vec(), || {
        if all {
            B::usable_gpus()
        } else {
            list.iter()
                .map(|s| s.parse::<DeviceSelector>())
                .collect::<Result<Vec<_>, _>>()
                .and_then(|selectors| B::resolve_devices(&selectors))
        }
        .map_err(InitError::from)
    });
    match outcome {
        Ok(indices) => Ok(indices.into_iter().map(|i| Some(i.to_string())).collect()),
        Err(InitError::Selection(msg)) => Err(RasterError::InvalidOptions(msg)),
        Err(InitError::Unavailable(e)) if policy == B::FORCE => Err(unavailable::<B>(&e)),
        Err(InitError::Unavailable(_)) if all => Ok(vec![None]),
        Err(InitError::Unavailable(_)) => Ok(list.iter().map(|s| Some(s.clone())).collect()),
    }
}

/// The backend `B` for `selector`, opened unless a live session holds it
/// and shared by every session naming the same selector, under any policy.
///
/// # Errors
///
/// Why the device did not open: a selector that does not resolve, or an
/// unavailable API.
pub(crate) fn open_backend<B: SessionDevice>(selector: Option<&str>) -> Result<Arc<B>, String> {
    cached_backend::<B>(selector).map_err(|e| match e {
        InitError::Selection(msg) | InitError::Unavailable(msg) => msg,
    })
}

/// The live backend for `selector`, or the failure of its construction,
/// or else a backend constructed now.  The lock is held across
/// construction so concurrent first sessions build one device.
fn cached_backend<B: SessionDevice>(selector: Option<&str>) -> Result<Arc<B>, InitError> {
    shared(B::registry(), selector.map(str::to_owned), || {
        construct::<B>(selector)
    })
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

fn construct<B: SessionDevice>(selector: Option<&str>) -> Result<Arc<B>, InitError> {
    let parsed = parse_selector(selector)?;
    Ok(Arc::new(B::open(parsed.as_ref())?))
}

#[cfg(test)]
mod tests {
    use super::init_backend;
    use crate::{BackendPolicy, RasterError};
    use gpu::backend::vulkan::VulkanBackend;

    /// A selector naming no device is a configuration error under `Auto`,
    /// which opens no device.
    #[test]
    fn auto_rejects_an_unresolvable_selector_while_vulkan_is_idle() {
        const SELECTOR: &str = "no-such-gpu-7f3c";
        match init_backend::<VulkanBackend>(BackendPolicy::Auto, Some(SELECTOR)) {
            Err(RasterError::InvalidOptions(msg)) => {
                assert!(
                    msg.contains(SELECTOR),
                    "message does not name the selector: {msg}"
                );
            }
            Ok(None) if VulkanBackend::new().is_err() => {
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
