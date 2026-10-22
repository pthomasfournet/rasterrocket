//! Vulkan-backed kernel dispatch helpers for [`PageRenderer`].
//!
//! Parallel branch to [`super::gpu_ops`] (which targets the CUDA `GpuCtx`).
//! Each helper follows the trait surface idiom:
//! `alloc → upload → begin_page → record_* → submit → wait → download`.
//!
//! Failure modes are reported as `bool`: returning `false` lets the caller
//! fall through to the CPU path, log-on-error keeps the renderer's
//! "GPU is opportunistic; CPU is the source of truth" contract.
//!
//! These helpers are gated on `feature = "vulkan"`; when the feature is off
//! the entire module compiles to nothing.

#![cfg(feature = "vulkan")]

use std::sync::Arc;

use gpu::backend::GpuBackend;
use gpu::backend::params::AaFillParams;
use gpu::backend::vulkan::VulkanBackend;

use super::PageRenderer;
use super::gpu_ops::{aa_fill_threshold, gpu_coverage_to_bitmap, prepare_aa_fill};

type DeviceBuffer = <VulkanBackend as GpuBackend>::DeviceBuffer;

/// Allocate `size` bytes on `backend`; on failure log a contextualised
/// warning at `warn` and return `None`.
///
/// `op_name` names the high-level operation (e.g. `"vulkan aa_fill"`)
/// and `what` names the buffer (`"intersects"`, `"coverage"`, …).  Together
/// they produce a single, scannable log line at every call site.
fn alloc_or_warn(
    backend: &VulkanBackend,
    size: usize,
    op_name: &'static str,
    what: &'static str,
) -> Option<DeviceBuffer> {
    backend
        .alloc_device(size)
        .map_err(|e| log::warn!("{op_name}: alloc_device({what}) failed: {e}"))
        .ok()
}

/// Upload `bytes` into `dst`; on failure log a contextualised warning
/// and return `false`.  See [`alloc_or_warn`] for the naming contract.
fn upload_or_warn(
    backend: &VulkanBackend,
    dst: &DeviceBuffer,
    bytes: &[u8],
    op_name: &'static str,
    what: &'static str,
) -> bool {
    backend
        .upload_sync(dst, bytes)
        .map_err(|e| log::warn!("{op_name}: upload_sync({what}) failed: {e}"))
        .is_ok()
}

/// Map any `gpu::backend::Result<T>` to `Option<T>`, logging the
/// failure with the call's contextual labels.
fn warn_err<T>(r: gpu::backend::Result<T>, op_name: &'static str, step: &'static str) -> Option<T> {
    match r {
        Ok(v) => Some(v),
        Err(e) => {
            log::warn!("{op_name}: vulkan {step} failed: {e}");
            None
        }
    }
}

/// Run a closure with a freshly-recorded page on `backend`, downloading the
/// supplied `output` device buffer back to host on success.
///
/// The closure receives `&backend` and is expected to call any `record_*`
/// method exactly once.  After it returns successfully, this helper calls
/// `submit_page` / `wait_page` and downloads `output` into a freshly
/// allocated `Vec<u8>` of length `out_len`.
///
/// All error variants are logged at `warn` level (with the supplied
/// `op_name` prefix) and surfaced as `None` so callers can fall back to CPU.
fn run_one_kernel(
    backend: &VulkanBackend,
    output: &DeviceBuffer,
    out_len: usize,
    op_name: &'static str,
    record: impl FnOnce(&VulkanBackend) -> gpu::backend::Result<()>,
) -> Option<Vec<u8>> {
    warn_err(backend.begin_page(), op_name, "begin_page")?;
    warn_err(record(backend), op_name, "record_*")?;
    let fence = warn_err(backend.submit_page(), op_name, "submit_page")?;
    warn_err(backend.wait_page(fence), op_name, "wait_page")?;

    let mut out = vec![0u8; out_len];
    warn_err(
        backend.download_sync(output, &mut out),
        op_name,
        "download_sync",
    )?;
    Some(out)
}

/// Attempt to rasterise `path` with the Vulkan scanline-AA fill kernel.
///
/// Mirrors [`super::gpu_ops::try_gpu_aa_fill`] but routes through the
/// Vulkan trait surface.
///
/// Returns `true` if the GPU path was taken (caller skips the CPU fill).
/// Returns `false` if [`prepare_aa_fill`] declines the fill, the table
/// has no intersections or more rows than `u32` holds, or any Vulkan
/// call fails (warning logged; CPU fill used as fallback).
pub(super) fn try_vk_aa_fill(
    renderer: &mut PageRenderer<'_>,
    backend: &Arc<VulkanBackend>,
    path: &raster::path::Path,
    even_odd: bool,
    flatness: f64,
    pipe: &raster::pipe::PipeState<'_>,
    src: &raster::pipe::PipeSrc<'_>,
) -> bool {
    const OP: &str = "vulkan aa_fill";

    let Some(fill) = prepare_aa_fill(
        &renderer.gstate.current().clip,
        renderer.bitmap.width,
        renderer.bitmap.height,
        path,
        flatness,
        even_odd,
        aa_fill_threshold(),
    ) else {
        return false;
    };
    // The backend rejects zero-size buffers; a table without intersections
    // paints nothing on either path, so the CPU fill is the exact fallback.
    if fill.intersects.is_empty() {
        return false;
    }
    let Ok(n_rows) = u32::try_from(fill.table().n_rows()) else {
        log::warn!("{OP}: table row count exceeds u32::MAX; falling back to CPU");
        return false;
    };
    let Some(n_pixels) = fill.rect.pixel_count() else {
        return false;
    };

    let intersect_bytes: &[u8] = bytemuck::cast_slice(&fill.intersects);
    let row_start_bytes: &[u8] = bytemuck::cast_slice(&fill.row_starts);

    let Some(d_intersects) = alloc_or_warn(backend, intersect_bytes.len(), OP, "intersects") else {
        return false;
    };
    let Some(d_row_starts) = alloc_or_warn(backend, row_start_bytes.len(), OP, "row_starts") else {
        return false;
    };
    let Some(d_cov) = alloc_or_warn(backend, n_pixels, OP, "coverage") else {
        return false;
    };
    if !upload_or_warn(backend, &d_intersects, intersect_bytes, OP, "intersects")
        || !upload_or_warn(backend, &d_row_starts, row_start_bytes, OP, "row_starts")
    {
        return false;
    }

    let Some(mut counts) = run_one_kernel(backend, &d_cov, n_pixels, OP, |be| {
        be.record_aa_fill(AaFillParams {
            intersects: &d_intersects,
            row_starts: &d_row_starts,
            coverage: &d_cov,
            n_rows,
            table_y0: fill.table_y0,
            rect: fill.rect,
            aa_width: fill.aa_width,
            fill_rule: u8::from(even_odd),
        })
    }) else {
        return false;
    };

    gpu_coverage_to_bitmap(renderer, &mut counts, fill.rect, pipe, src);
    true
}

// ICC CMYK→RGB on Vulkan is intentionally not wired here.  The CUDA dispatcher
// for the matrix path always short-circuits to CPU AVX-512 (PCIe round-trip
// dominates), and the CLUT path's renderer plumbing (`resolve_image` →
// `decode_dct` → `cmyk_raw_to_rgb`) currently threads only `Option<&GpuCtx>`.
// Adding a parallel `Option<&VulkanBackend>` parameter through that chain is
// scoped to a follow-up; under `BackendPolicy::ForceVulkan` `gpu_ctx` is
// `None`, so the CMYK path falls back to the CPU `cmyk_to_rgb_reflectance`
// matrix — a quality regression for CMYK images with embedded ICC profiles
// vs. CUDA, not a correctness bug.  Phase 9-pre-2026-05-07 had the same
// behaviour and it was acceptable then.
