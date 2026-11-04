//! Lending the calling thread's GPU image decoders to a page renderer and
//! returning them afterwards.

#[cfg(any(feature = "nvjpeg", feature = "nvjpeg2k", feature = "gpu-jpeg-huffman"))]
use crate::gpu_init;
use crate::{BackendPolicy, RasterError, RasterSession};

/// Lend per-thread GPU JPEG decoders to the renderer for one page.
///
/// `effective_policy` is normally `session.policy` but callers may pass
/// [`BackendPolicy::CpuOnly`] to skip GPU decoder init for this page regardless
/// of the session-level policy — used by affinity dispatch for `CpuOnly` pages.
///
/// On `CpuOnly` this is a no-op.  On `ForceCuda`/`ForceVaapi` init failure is
/// returned as `RasterError::BackendUnavailable` rather than silently falling back.
#[cfg_attr(
    not(any(feature = "nvjpeg", feature = "nvjpeg2k", feature = "gpu-jpeg-huffman")),
    expect(
        clippy::unnecessary_wraps,
        reason = "Result<()> only carries an error from the GPU-decoder init paths"
    )
)]
#[cfg_attr(
    not(any(
        feature = "nvjpeg",
        feature = "nvjpeg2k",
        feature = "vaapi",
        feature = "gpu-jpeg-huffman"
    )),
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
    #[cfg(not(any(
        feature = "nvjpeg",
        feature = "nvjpeg2k",
        feature = "vaapi",
        feature = "gpu-jpeg-huffman"
    )))]
    let _ = renderer;

    #[cfg(feature = "gpu-jpeg-huffman")]
    lend_huffman_decoder(session, renderer, effective_policy)?;

    #[cfg(feature = "nvjpeg")]
    {
        let dispatch_nvjpeg = {
            #[cfg(feature = "vaapi")]
            {
                !matches!(effective_policy, BackendPolicy::ForceVaapi)
            }
            #[cfg(not(feature = "vaapi"))]
            {
                true
            }
        };
        if dispatch_nvjpeg {
            gpu_init::ensure_nvjpeg(effective_policy).map_err(RasterError::BackendUnavailable)?;
            gpu_init::NVJPEG_DEC.with(|cell| {
                if let gpu_init::DecoderInit::Ready(slot) = &mut *cell.borrow_mut() {
                    renderer.set_nvjpeg(slot.take());
                }
            });
        }
    }

    #[cfg(feature = "nvjpeg2k")]
    {
        let dispatch_nvjpeg2k = {
            #[cfg(feature = "vaapi")]
            {
                !matches!(effective_policy, BackendPolicy::ForceVaapi)
            }
            #[cfg(not(feature = "vaapi"))]
            {
                true
            }
        };
        if dispatch_nvjpeg2k {
            gpu_init::ensure_nvjpeg2k(effective_policy).map_err(RasterError::BackendUnavailable)?;
            gpu_init::NVJPEG2K_DEC.with(|cell| {
                if let gpu_init::DecoderInit::Ready(slot) = &mut *cell.borrow_mut() {
                    renderer.set_nvjpeg2k(slot.take());
                }
            });
        }
    }

    #[cfg(feature = "vaapi")]
    if let Some(queue) = &session.vaapi_queue {
        renderer.set_vaapi_queue(queue.handle());
    }

    Ok(())
}

/// Whether `policy` builds this thread's parallel-Huffman JPEG decoder,
/// `kernel_enabled` being [`pdf_interp::device_kernel_enabled`] for it.
/// `Auto` builds it only while the kernel can dispatch; `ForceCuda` and
/// `ForceVulkan` build it regardless, so a missing device fails the forced
/// policy; `CpuOnly` and `ForceVaapi` never build it.
#[cfg(feature = "gpu-jpeg-huffman")]
const fn wants_huffman_decoder(policy: BackendPolicy, kernel_enabled: bool) -> bool {
    match policy {
        BackendPolicy::Auto => kernel_enabled,
        BackendPolicy::ForceCuda | BackendPolicy::ForceVulkan => true,
        BackendPolicy::CpuOnly => false,
        #[cfg(feature = "vaapi")]
        BackendPolicy::ForceVaapi => false,
    }
}

/// Lend this thread's parallel-Huffman JPEG decoder to `renderer`: the
/// Vulkan one when Vulkan is the active backend, the CUDA one otherwise.
/// Nothing is built or lent when [`wants_huffman_decoder`] says no.
#[cfg(feature = "gpu-jpeg-huffman")]
fn lend_huffman_decoder(
    session: &RasterSession,
    renderer: &mut pdf_interp::renderer::PageRenderer,
    effective_policy: BackendPolicy,
) -> Result<(), RasterError> {
    let kernel_enabled = pdf_interp::device_kernel_enabled(pdf_interp::DeviceKernel::HuffmanJpeg);
    if !wants_huffman_decoder(effective_policy, kernel_enabled) {
        return Ok(());
    }
    // Vulkan wins Auto when a Vulkan backend is available.
    #[cfg(feature = "vulkan")]
    if matches!(effective_policy, BackendPolicy::ForceVulkan)
        || (matches!(effective_policy, BackendPolicy::Auto) && session.vk_backend.is_some())
    {
        gpu_init::ensure_jpeg_vk_huffman(effective_policy, session.vulkan_device.as_deref())
            .map_err(RasterError::BackendUnavailable)?;
        gpu_init::JPEG_VK_DEC.with(|cell| {
            if let gpu_init::JpegGpuInit::Ready(slot) = &mut *cell.borrow_mut() {
                renderer.set_jpeg_vk(slot.take());
            }
        });
        return Ok(());
    }
    #[cfg(not(feature = "vulkan"))]
    let _ = session;
    gpu_init::ensure_jpeg_gpu_huffman(effective_policy).map_err(RasterError::BackendUnavailable)?;
    gpu_init::JPEG_CUDA_DEC.with(|cell| {
        if let gpu_init::JpegGpuInit::Ready(slot) = &mut *cell.borrow_mut() {
            renderer.set_jpeg_gpu(slot.take());
        }
    });
    Ok(())
}

/// Return GPU JPEG decoders from the renderer back into TLS slots for reuse.
///
/// The VA-API path is omitted here: `JpegQueueHandle` is cheaply cloneable and
/// is simply dropped with the renderer — no reclaim step is needed.  The
/// `Arc<DecodeQueue>` in `RasterSession` keeps the worker alive across pages.
#[cfg_attr(
    not(any(feature = "nvjpeg", feature = "nvjpeg2k", feature = "gpu-jpeg-huffman")),
    expect(
        clippy::missing_const_for_fn,
        reason = "non-const only in GPU-decoder builds"
    )
)]
fn reclaim_decoders(renderer: &mut pdf_interp::renderer::PageRenderer) {
    #[cfg(not(any(feature = "nvjpeg", feature = "nvjpeg2k", feature = "gpu-jpeg-huffman")))]
    let _ = renderer;
    #[cfg(feature = "gpu-jpeg-huffman")]
    gpu_init::JPEG_CUDA_DEC.with(|cell| {
        if let gpu_init::JpegGpuInit::Ready(slot) = &mut *cell.borrow_mut() {
            *slot = renderer.take_jpeg_gpu();
        }
    });
    #[cfg(all(feature = "gpu-jpeg-huffman", feature = "vulkan"))]
    gpu_init::JPEG_VK_DEC.with(|cell| {
        if let gpu_init::JpegGpuInit::Ready(slot) = &mut *cell.borrow_mut() {
            *slot = renderer.take_jpeg_vk();
        }
    });
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

#[cfg(all(test, feature = "gpu-jpeg-huffman"))]
mod tests {
    use super::wants_huffman_decoder;
    use crate::BackendPolicy;

    #[test]
    fn auto_builds_no_huffman_decoder_while_the_kernel_is_disabled() {
        assert!(!wants_huffman_decoder(BackendPolicy::Auto, false));
        assert!(wants_huffman_decoder(BackendPolicy::Auto, true));
        for forced in [BackendPolicy::ForceCuda, BackendPolicy::ForceVulkan] {
            assert!(wants_huffman_decoder(forced, false), "{forced:?}");
        }
        assert!(!wants_huffman_decoder(BackendPolicy::CpuOnly, true));
        #[cfg(feature = "vaapi")]
        assert!(!wants_huffman_decoder(BackendPolicy::ForceVaapi, true));
    }
}
