//! Lending the calling thread's GPU image decoders to a page renderer and
//! returning them afterwards.

#[cfg(any(feature = "nvjpeg", feature = "nvjpeg2k"))]
use crate::gpu_init;
use crate::{BackendPolicy, RasterError, RasterSession};
#[cfg(feature = "nvjpeg2k")]
use pdf_interp::DeviceKernel;

/// Lend per-thread GPU JPEG decoders to the renderer for one page.
///
/// `effective_policy` is normally `session.policy` but callers may pass
/// [`BackendPolicy::CpuOnly`] to skip GPU decoder init for this page regardless
/// of the session-level policy — used by affinity dispatch for `CpuOnly` pages.
///
/// On `CpuOnly` this is a no-op.  The nvJPEG decoder is built only where
/// `wants_nvjpeg_decoder` says so, and the nvJPEG2000 decoder as
/// `nvjpeg2k_supply` says: on `Auto` only when the page draws an image it
/// takes, never under `ForceVulkan` or `ForceD3d12` nor while the policy's threshold
/// disables it (`nvjpeg2k_supply`).  On `ForceCuda`/`ForceVaapi` init
/// failure is returned as `RasterError::BackendUnavailable` rather than
/// silently falling back.
#[cfg_attr(
    not(any(feature = "nvjpeg", feature = "nvjpeg2k")),
    expect(
        clippy::unnecessary_wraps,
        reason = "Result<()> only carries an error from the GPU-decoder init paths"
    )
)]
#[cfg_attr(
    not(any(feature = "nvjpeg", feature = "nvjpeg2k", feature = "vaapi")),
    expect(
        clippy::missing_const_for_fn,
        reason = "body collapses to a single match when no GPU-decoder feature is on"
    )
)]
pub(crate) fn lend_decoders(
    session: &RasterSession,
    renderer: &mut pdf_interp::renderer::PageRenderer,
    effective_policy: BackendPolicy,
) -> Result<(), RasterError> {
    if matches!(effective_policy, BackendPolicy::CpuOnly) {
        return Ok(());
    }
    #[cfg(not(feature = "vaapi"))]
    let _ = session;
    #[cfg(not(any(feature = "nvjpeg", feature = "nvjpeg2k", feature = "vaapi")))]
    let _ = renderer;

    #[cfg(feature = "nvjpeg")]
    if wants_nvjpeg_decoder(
        effective_policy,
        pdf_interp::device_kernel_enabled(pdf_interp::DeviceKernel::NvJpeg),
    ) {
        gpu_init::ensure_nvjpeg(effective_policy).map_err(RasterError::BackendUnavailable)?;
        gpu_init::NVJPEG_DEC.with(|cell| {
            if let gpu_init::DecoderInit::Ready(slot) = &mut *cell.borrow_mut() {
                renderer.set_nvjpeg(slot.take());
            }
        });
    }

    #[cfg(feature = "nvjpeg2k")]
    match nvjpeg2k_supply(effective_policy, pdf_interp::device_kernel_enabled) {
        Nvjpeg2kSupply::OnDemand => {
            // A decoder this thread already built is lent; otherwise the
            // renderer builds one when it first meets an image it takes.
            gpu_init::NVJPEG2K_DEC.with(|cell| {
                if let gpu_init::DecoderInit::Ready(slot) = &mut *cell.borrow_mut() {
                    renderer.set_nvjpeg2k(slot.take());
                }
            });
            renderer.set_nvjpeg2k_source(Some(gpu_init::take_nvjpeg2k_for_auto));
        }
        Nvjpeg2kSupply::Eager => {
            gpu_init::ensure_nvjpeg2k(effective_policy).map_err(RasterError::BackendUnavailable)?;
            gpu_init::NVJPEG2K_DEC.with(|cell| {
                if let gpu_init::DecoderInit::Ready(slot) = &mut *cell.borrow_mut() {
                    renderer.set_nvjpeg2k(slot.take());
                }
            });
        }
        Nvjpeg2kSupply::None => {}
    }

    #[cfg(feature = "vaapi")]
    if let Some(queue) = &session.vaapi_queue {
        renderer.set_vaapi_queue(queue.handle());
    }

    Ok(())
}

/// Whether `policy` builds this thread's nvJPEG decoder, `kernel_enabled`
/// being [`pdf_interp::device_kernel_enabled`] for it.  Only `Auto` and
/// `ForceCuda` decode on CUDA, and only while the decoder can dispatch: a
/// disabled threshold sends no JPEG to it.
#[cfg(feature = "nvjpeg")]
const fn wants_nvjpeg_decoder(policy: BackendPolicy, kernel_enabled: bool) -> bool {
    match policy {
        BackendPolicy::Auto | BackendPolicy::ForceCuda => kernel_enabled,
        BackendPolicy::CpuOnly | BackendPolicy::ForceVulkan | BackendPolicy::ForceD3d12 => false,
        #[cfg(feature = "vaapi")]
        BackendPolicy::ForceVaapi => false,
    }
}

/// How a rendering thread obtains its nvJPEG2000 decoder.
#[cfg(feature = "nvjpeg2k")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Nvjpeg2kSupply {
    /// Built when the renderer first draws a `JPXDecode` image it takes.
    OnDemand,
    /// Built before the page renders; a failure fails the policy.
    Eager,
    /// Never built.
    None,
}

/// The nvJPEG2000 supply of `policy`, `enabled` being
/// [`pdf_interp::device_kernel_enabled`]: on demand under `Auto` while
/// [`DeviceKernel::NvJpeg2kOnDemand`] is enabled, eager under `ForceCuda`
/// while [`DeviceKernel::NvJpeg2kAttached`] is, and otherwise none, as under
/// the policies that initialise no CUDA context.
#[cfg(feature = "nvjpeg2k")]
fn nvjpeg2k_supply(
    policy: BackendPolicy,
    enabled: impl Fn(DeviceKernel) -> bool,
) -> Nvjpeg2kSupply {
    match policy {
        BackendPolicy::Auto if enabled(DeviceKernel::NvJpeg2kOnDemand) => Nvjpeg2kSupply::OnDemand,
        BackendPolicy::ForceCuda if enabled(DeviceKernel::NvJpeg2kAttached) => {
            Nvjpeg2kSupply::Eager
        }
        BackendPolicy::Auto | BackendPolicy::ForceCuda => Nvjpeg2kSupply::None,
        BackendPolicy::CpuOnly | BackendPolicy::ForceVulkan | BackendPolicy::ForceD3d12 => {
            Nvjpeg2kSupply::None
        }
        #[cfg(feature = "vaapi")]
        BackendPolicy::ForceVaapi => Nvjpeg2kSupply::None,
    }
}

/// Why a JPEG 2000 dispatch threshold in the environment has no effect on
/// a session of `policy`, or `None` when the session can use it.
///
/// The threshold steers the nvJPEG2000 decoder, which exists only with the
/// `nvjpeg2k` feature and runs only under the policies that may start a
/// CUDA context (`Auto`, `ForceCuda`).
const fn inert_jpx_threshold(policy: BackendPolicy) -> Option<&'static str> {
    if !cfg!(feature = "nvjpeg2k") {
        return Some("this build has no `nvjpeg2k` feature");
    }
    match policy {
        BackendPolicy::Auto | BackendPolicy::ForceCuda => None,
        BackendPolicy::CpuOnly => Some("backend `cpu` starts no CUDA context"),
        BackendPolicy::ForceVulkan => Some("backend `vulkan` starts no CUDA context"),
        BackendPolicy::ForceD3d12 => Some("backend `d3d12` starts no CUDA context"),
        #[cfg(feature = "vaapi")]
        BackendPolicy::ForceVaapi => Some("backend `vaapi` starts no CUDA context"),
    }
}

/// Warn, once per process, when `PDF_RASTER_JPX_THRESHOLD` is set but a
/// session of `policy` cannot use it ([`inert_jpx_threshold`]).
pub(crate) fn warn_inert_jpx_threshold(policy: BackendPolicy) {
    static WARNED: std::sync::Once = std::sync::Once::new();
    if std::env::var_os("PDF_RASTER_JPX_THRESHOLD").is_none() {
        return;
    }
    if let Some(reason) = inert_jpx_threshold(policy) {
        WARNED.call_once(|| {
            log::warn!(
                "rasterrocket: PDF_RASTER_JPX_THRESHOLD is set but has no effect: {reason}; \
                 JPEG 2000 images decode on the CPU"
            );
        });
    }
}

/// Return GPU JPEG decoders from the renderer back into TLS slots for reuse.
///
/// The VA-API path is omitted here: `JpegQueueHandle` is cheaply cloneable and
/// is simply dropped with the renderer — no reclaim step is needed.  The
/// `Arc<DecodeQueue>` in `RasterSession` keeps the worker alive across pages.
#[cfg_attr(
    not(any(feature = "nvjpeg", feature = "nvjpeg2k")),
    expect(
        clippy::missing_const_for_fn,
        reason = "non-const only in GPU-decoder builds"
    )
)]
fn reclaim_decoders(renderer: &mut pdf_interp::renderer::PageRenderer) {
    #[cfg(not(any(feature = "nvjpeg", feature = "nvjpeg2k")))]
    let _ = renderer;
    #[cfg(feature = "nvjpeg")]
    gpu_init::NVJPEG_DEC.with(|cell| {
        if let gpu_init::DecoderInit::Ready(slot) = &mut *cell.borrow_mut() {
            *slot = renderer.take_nvjpeg();
        }
    });
    #[cfg(feature = "nvjpeg2k")]
    gpu_init::NVJPEG2K_DEC.with(|cell| {
        if let gpu_init::DecoderInit::Ready(slot) = &mut *cell.borrow_mut() {
            *slot = renderer.take_nvjpeg2k();
        }
    });
}

// ── RAII guard: return decoder handles to TLS even on unwind ─────────────────

/// RAII guard that calls [`reclaim_decoders`] when dropped.
///
/// `lend_decoders` moves per-thread GPU decoder handles out of TLS into the
/// `PageRenderer`. The guard is armed *before* `lend_decoders` is called so it
/// covers three exit paths, each running `reclaim_decoders` exactly once:
///
/// - **`?`-early-return inside `lend_decoders`**: a later `ensure_*` failing
///   would otherwise drop the renderer with earlier handles already moved in —
///   a partial-move leak. The armed guard reclaims them on the way out.
/// - **panic-unwind in `execute`/`render_annotations`** (`panic=unwind` test
///   builds or unwind-configured embedders): the stack unwind triggers `Drop`,
///   returning the handles to TLS rather than dropping them, so subsequent
///   pages on the same Rayon worker still find their decoder slots populated.
/// - **normal success**: `drop(guard)` is called explicitly — the `&mut`
///   borrow is released and `renderer` is accessible directly for the
///   `budget_status()`/`image_failures()`/`finish()` tail.
///
/// Reclaim is sound after a *partial* lend: each decoder's ensure+take+set is
/// atomic per feature, so earlier features are fully lent and reclaim restores
/// them, while a feature whose `ensure_*` failed left its TLS cell non-`Ready`
/// and reclaim's `if let Ready` guard skips it (no clobber).
pub(crate) struct DecoderReclaim<'r, 'doc>(pub &'r mut pdf_interp::renderer::PageRenderer<'doc>);

impl Drop for DecoderReclaim<'_, '_> {
    fn drop(&mut self) {
        reclaim_decoders(self.0);
    }
}

#[cfg(all(test, any(feature = "nvjpeg", feature = "nvjpeg2k")))]
mod cuda_decoder_tests {
    use crate::BackendPolicy;

    /// No nvJPEG decoder is built while its threshold disables it, nor
    /// under a policy that initialises no CUDA context.
    #[cfg(feature = "nvjpeg")]
    #[test]
    fn nvjpeg_decoders_are_built_only_where_they_can_dispatch() {
        use super::wants_nvjpeg_decoder;
        for policy in [BackendPolicy::Auto, BackendPolicy::ForceCuda] {
            assert!(!wants_nvjpeg_decoder(policy, false), "{policy:?}");
            assert!(wants_nvjpeg_decoder(policy, true), "{policy:?}");
        }
        for policy in [
            BackendPolicy::CpuOnly,
            BackendPolicy::ForceVulkan,
            BackendPolicy::ForceD3d12,
        ] {
            assert!(!wants_nvjpeg_decoder(policy, true), "{policy:?}");
        }
        #[cfg(feature = "vaapi")]
        assert!(!wants_nvjpeg_decoder(BackendPolicy::ForceVaapi, true));
    }

    /// A Vulkan session starts no CUDA nvJPEG2000 decoder, and no session
    /// starts one while its kernel is disabled.
    #[cfg(feature = "nvjpeg2k")]
    #[test]
    fn nvjpeg2k_is_supplied_only_to_cuda_capable_policies() {
        use super::{Nvjpeg2kSupply, nvjpeg2k_supply};
        use pdf_interp::DeviceKernel;
        let only = |on: DeviceKernel| move |kernel: DeviceKernel| kernel == on;
        let auto = BackendPolicy::Auto;
        let cuda = BackendPolicy::ForceCuda;
        assert_eq!(nvjpeg2k_supply(auto, |_| true), Nvjpeg2kSupply::OnDemand);
        assert_eq!(nvjpeg2k_supply(cuda, |_| true), Nvjpeg2kSupply::Eager);
        let on_demand = only(DeviceKernel::NvJpeg2kOnDemand);
        let attached = only(DeviceKernel::NvJpeg2kAttached);
        assert_eq!(nvjpeg2k_supply(auto, on_demand), Nvjpeg2kSupply::OnDemand);
        assert_eq!(nvjpeg2k_supply(auto, attached), Nvjpeg2kSupply::None);
        assert_eq!(nvjpeg2k_supply(cuda, attached), Nvjpeg2kSupply::Eager);
        assert_eq!(nvjpeg2k_supply(cuda, on_demand), Nvjpeg2kSupply::None);
        for policy in [
            BackendPolicy::ForceVulkan,
            BackendPolicy::ForceD3d12,
            BackendPolicy::CpuOnly,
        ] {
            assert_eq!(nvjpeg2k_supply(policy, |_| true), Nvjpeg2kSupply::None);
        }
        for policy in [auto, cuda] {
            assert_eq!(nvjpeg2k_supply(policy, |_| false), Nvjpeg2kSupply::None);
        }
    }

    /// By default `ForceCuda` builds its nvJPEG2000 decoder before the
    /// page renders, while `Auto` builds none.
    #[cfg(feature = "nvjpeg2k")]
    #[test]
    fn force_cuda_is_not_dormant_by_default() {
        use super::{Nvjpeg2kSupply, nvjpeg2k_supply};
        if std::env::var_os("PDF_RASTER_JPX_THRESHOLD").is_some() {
            return;
        }
        let enabled = pdf_interp::device_kernel_enabled;
        assert_eq!(
            nvjpeg2k_supply(BackendPolicy::ForceCuda, enabled),
            Nvjpeg2kSupply::Eager
        );
        assert_eq!(
            nvjpeg2k_supply(BackendPolicy::Auto, enabled),
            Nvjpeg2kSupply::None
        );
    }
}

#[cfg(test)]
mod inert_jpx_tests {
    use super::inert_jpx_threshold;
    use crate::BackendPolicy;

    /// The threshold acts only with the decoder compiled in and under a
    /// policy that may start CUDA.
    #[test]
    fn a_jpx_threshold_is_inert_without_the_decoder_or_a_cuda_policy() {
        let decoder = cfg!(feature = "nvjpeg2k");
        assert_eq!(inert_jpx_threshold(BackendPolicy::Auto).is_none(), decoder);
        assert_eq!(
            inert_jpx_threshold(BackendPolicy::ForceCuda).is_none(),
            decoder
        );
        assert!(inert_jpx_threshold(BackendPolicy::CpuOnly).is_some());
        assert!(inert_jpx_threshold(BackendPolicy::ForceVulkan).is_some());
        assert!(inert_jpx_threshold(BackendPolicy::ForceD3d12).is_some());
    }
}
