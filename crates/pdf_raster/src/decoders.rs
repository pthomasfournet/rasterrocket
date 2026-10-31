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
pub fn lend_decoders(
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

/// Lend this thread's parallel-Huffman JPEG decoder to `renderer`: the
/// Vulkan one when Vulkan is the active backend, the CUDA one otherwise.
/// Nothing is lent under `ForceVaapi`.
#[cfg(feature = "gpu-jpeg-huffman")]
fn lend_huffman_decoder(
    session: &RasterSession,
    renderer: &mut pdf_interp::renderer::PageRenderer,
    effective_policy: BackendPolicy,
) -> Result<(), RasterError> {
    let dispatch_huffman = {
        #[cfg(feature = "vaapi")]
        {
            !matches!(effective_policy, BackendPolicy::ForceVaapi)
        }
        #[cfg(not(feature = "vaapi"))]
        {
            true
        }
    };
    if dispatch_huffman {
        // Vulkan wins Auto when a Vulkan backend is available.
        #[cfg(feature = "vulkan")]
        let vulkan_active = matches!(effective_policy, BackendPolicy::ForceVulkan)
            || (matches!(effective_policy, BackendPolicy::Auto) && session.vk_backend.is_some());
        #[cfg(not(feature = "vulkan"))]
        let vulkan_active = false;

        if vulkan_active {
            #[cfg(feature = "vulkan")]
            {
                gpu_init::ensure_jpeg_vk_huffman(
                    effective_policy,
                    session.vulkan_device.as_deref(),
                )
                .map_err(RasterError::BackendUnavailable)?;
                gpu_init::JPEG_VK_DEC.with(|cell| {
                    if let gpu_init::JpegGpuInit::Ready(slot) = &mut *cell.borrow_mut() {
                        renderer.set_jpeg_vk(slot.take());
                    }
                });
            }
        } else {
            gpu_init::ensure_jpeg_gpu_huffman(effective_policy)
                .map_err(RasterError::BackendUnavailable)?;
            gpu_init::JPEG_CUDA_DEC.with(|cell| {
                if let gpu_init::JpegGpuInit::Ready(slot) = &mut *cell.borrow_mut() {
                    renderer.set_jpeg_gpu(slot.take());
                }
            });
        }
    }
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
pub struct DecoderReclaim<'r, 'doc>(pub &'r mut pdf_interp::renderer::PageRenderer<'doc>);

impl Drop for DecoderReclaim<'_, '_> {
    fn drop(&mut self) {
        reclaim_decoders(self.0);
    }
}
