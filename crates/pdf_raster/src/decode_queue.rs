// `pub(crate)` items inside a `pub(crate)` module trigger clippy::redundant_pub_crate
// when nursery lints are enabled, but the explicitness aids readability in a module
// that is only conditionally compiled and accessed from a single call site.
#![expect(
    clippy::redundant_pub_crate,
    reason = "explicitness aids readability for conditionally-compiled single-call-site module"
)]
//! Construction helpers that build [`gpu::DecodeQueue`] instances for each
//! supported hardware JPEG decoder backend.
//!
//! These functions follow the same `BackendPolicy` semantics as the TLS-based
//! `ensure_*` helpers in `gpu_init`: soft failures are logged and silently
//! ignored on `Auto`; hard failures on `Force*` policies are returned as `Err`.
//!
//! Construction is serialised through [`crate::gpu_init::DECODER_INIT_LOCK`]
//! because `nvjpegCreateEx` and `vaInitialize` are not safe to call concurrently.

use std::sync::{Mutex, Once, OnceLock, PoisonError};

use crate::{BackendPolicy, DEFAULT_VAAPI_DEVICE, gpu_init::DECODER_INIT_LOCK};

/// Result of probing the host's render nodes, computed once per process.
/// `Ok` holds the chosen node path; `Err` the message naming every node
/// and why it was rejected.
static PROBED_NODE: OnceLock<Result<String, String>> = OnceLock::new();

/// Makes the CPU-fallback warning under `Auto` fire once per process;
/// later sessions log the same condition at `debug`.
static AUTO_FALLBACK_WARNED: Once = Once::new();

/// Render-node paths already logged by [`log_chosen_node_once`].
static LOGGED_NODES: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Log `path` as the VA-API render node at `debug` level, once per
/// process per distinct path.
fn log_chosen_node_once(path: &str, detail: &str) {
    let first = {
        let mut logged = LOGGED_NODES.lock().unwrap_or_else(PoisonError::into_inner);
        let first = !logged.iter().any(|p| p == path);
        if first {
            logged.push(path.to_owned());
        }
        first
    };
    if first {
        log::debug!("vaapi: chosen render node \"{path}\" {detail}");
    }
}

/// The node [`gpu::vaapi::find_jpeg_render_node`] chooses, probed on first
/// call and cached.  The caller holds [`DECODER_INIT_LOCK`]: probing runs
/// `vaInitialize` on every node it tries.
fn probed_node() -> Result<String, String> {
    PROBED_NODE
        .get_or_init(|| {
            gpu::vaapi::find_jpeg_render_node().map(|found| {
                log_chosen_node_once(
                    &found.node.path,
                    &format!(
                        "{} vendor=\"{}\" (probed)",
                        identity(&found.node),
                        found.vendor
                    ),
                );
                found.node.path
            })
        })
        .clone()
}

/// `pci=<address> driver=<name>` for `node`, `unknown` where sysfs did not
/// resolve a field.
fn identity(node: &gpu::vaapi::RenderNode) -> String {
    format!(
        "pci={} driver={}",
        node.pci.as_deref().unwrap_or("unknown"),
        node.driver.as_deref().unwrap_or("unknown")
    )
}

/// Build a VA-API JPEG decode queue.
///
/// `drm_node` is a render-node path, or [`DEFAULT_VAAPI_DEVICE`] (`auto`)
/// to use the first render node, in ascending number, whose driver
/// supports baseline JPEG decode.  Opens a `VapiJpegDecoder` there, wraps
/// it in a [`gpu::DecodeQueue`], and returns it ready for use.
///
/// Returns `Ok(None)` without touching VA-API under `CpuOnly`, `ForceCuda`
/// and `ForceVulkan`, and (after logging) when no usable node exists and
/// `policy` is `Auto`.  Returns `Err(msg)` when `policy` is `ForceVaapi`
/// and initialisation fails, so the caller can surface it as
/// [`crate::RasterError::BackendUnavailable`]; the message lists the
/// render nodes present.
#[cfg(feature = "vaapi")]
pub(crate) fn build_vaapi_queue(
    drm_node: &str,
    policy: BackendPolicy,
) -> Result<Option<gpu::DecodeQueue<gpu::vaapi::VapiJpegDecoder>>, String> {
    if matches!(
        policy,
        BackendPolicy::CpuOnly | BackendPolicy::ForceCuda | BackendPolicy::ForceVulkan
    ) {
        return Ok(None);
    }

    let guard = DECODER_INIT_LOCK
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    let result = if drm_node == DEFAULT_VAAPI_DEVICE {
        probed_node().and_then(|path| {
            gpu::vaapi::VapiJpegDecoder::new(&path)
                .map_err(|e| format!("VA-API JPEG decoder unavailable on probed node {path}: {e}"))
        })
    } else {
        gpu::vaapi::VapiJpegDecoder::new(drm_node)
            .inspect(|_| log_chosen_node_once(drm_node, "(configured)"))
            .map_err(|e| {
                format!(
                    "VA-API JPEG decoder unavailable on {drm_node}: {e}\nrender nodes present:\n{}",
                    gpu::vaapi::describe_render_nodes(&gpu::vaapi::render_nodes())
                )
            })
    };
    drop(guard);

    match result {
        Ok(dec) => Ok(Some(gpu::DecodeQueue::new(dec, "vaapi-jpeg-worker"))),
        Err(msg) => {
            if matches!(policy, BackendPolicy::ForceVaapi) {
                Err(msg)
            } else {
                let mut warned = false;
                AUTO_FALLBACK_WARNED.call_once(|| {
                    warned = true;
                    log::warn!(
                        "rasterrocket: VA-API JPEG unavailable; JPEG images will be decoded on CPU: {msg}"
                    );
                });
                if !warned {
                    log::debug!("rasterrocket: VA-API JPEG unavailable: {msg}");
                }
                Ok(None)
            }
        }
    }
}
