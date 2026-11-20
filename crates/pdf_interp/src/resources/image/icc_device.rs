//! Device dispatch of the ICC CMYK → RGB conversion.
//!
//! An image of 8-bit samples in an ICC-based CMYK space of at least
//! [`gpu::GPU_ICC_CLUT_THRESHOLD`] pixels converts on the session's GPU
//! backend through the grid of its tabulated conversion
//! ([`IccConversion::cmyk_grid`]), which the device kernels interpolate
//! in the integer arithmetic of the CPU conversion, to 8-bit samples or,
//! where the CPU converts the samples widened at 16 bits, to 16-bit ones.  Every other image, and
//! one the device declines or fails, converts on the CPU, as in a session
//! without a backend.  A GPU session's output is therefore a CPU
//! session's, byte for byte.
//!
//! The Vulkan or Direct3D 12 backend is used when the session holds one (a
//! session holds at most one of them), otherwise the CUDA context; each
//! runs [`gpu::backend::icc_clut`].  A Vulkan or Direct3D 12 device whose
//! kernel fails the backend's exactness check
//! (`VulkanBackend::icc_clut_is_exact`, `D3d12Backend::icc_clut_is_exact`)
//! is not used, nor is a Vulkan device whose earlier work could not be
//! confirmed complete (`VulkanBackend::is_lost`).

#![cfg(feature = "gpu-icc")]

use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use color::grid::CmykGrid;
use gpu::GpuCtx;
use gpu::backend::GpuBackend;
#[cfg(all(windows, feature = "d3d12"))]
use gpu::backend::d3d12::D3d12Backend;
#[cfg(feature = "vulkan")]
use gpu::backend::vulkan::VulkanBackend;

use crate::resources::colorspace::icc::IccConversion;

/// The GPU backends an ICC CMYK conversion may run on, and the counters
/// its outcome is recorded in.
pub struct IccHandles<'a> {
    /// CUDA context; used when no Vulkan or Direct3D 12 backend is
    /// attached.
    pub gpu_ctx: Option<&'a Arc<GpuCtx>>,
    /// Vulkan backend; used in preference to the CUDA context.
    #[cfg(feature = "vulkan")]
    pub vk_backend: Option<&'a VulkanBackend>,
    /// Direct3D 12 backend; used in preference to the CUDA context.
    #[cfg(all(windows, feature = "d3d12"))]
    pub d3d12_backend: Option<&'a D3d12Backend>,
    /// Counters of the device conversions; `None` counts nothing.
    pub counts: Option<&'a DeviceIccCounts>,
}

impl IccHandles<'_> {
    /// No backend: conversion runs on the CPU through the profile
    /// transform.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            gpu_ctx: None,
            #[cfg(feature = "vulkan")]
            vk_backend: None,
            #[cfg(all(windows, feature = "d3d12"))]
            d3d12_backend: None,
            counts: None,
        }
    }

    /// Whether a backend is attached.
    const fn has_device(&self) -> bool {
        #[cfg(feature = "vulkan")]
        if self.vk_backend.is_some() {
            return true;
        }
        #[cfg(all(windows, feature = "d3d12"))]
        if self.d3d12_backend.is_some() {
            return true;
        }
        self.gpu_ctx.is_some()
    }

    fn count(&self, field: impl Fn(&DeviceIccCounts) -> &Cell<u32>) {
        if let Some(counts) = self.counts {
            let cell = field(counts);
            cell.set(cell.get().saturating_add(1));
        }
    }
}

/// Device conversions of ICC CMYK images: those the device converted and
/// those it took and failed.
#[derive(Debug, Default)]
pub struct DeviceIccCounts {
    converted: Cell<u32>,
    failed: Cell<u32>,
}

impl DeviceIccCounts {
    /// Conversions the device performed.
    #[must_use]
    pub const fn converted(&self) -> u32 {
        self.converted.get()
    }

    /// Conversions the device took and failed, each then performed
    /// through the profile transform.
    #[must_use]
    pub const fn failed(&self) -> u32 {
        self.failed.get()
    }
}

/// The outcome of offering a conversion to a device.
enum DeviceIcc<T> {
    /// The device converted the pixels.
    Converted(Vec<T>),
    /// The device does not take the conversion.
    Declined,
    /// The device failed.
    Failed,
}

/// An output sample of the device conversion: `u8`, the outputs of
/// [`CmykGrid::convert`], or `u16`, those of [`CmykGrid::convert16`].
pub(super) trait ClutSample: Sized {
    /// The conversion of `cmyk` through `grid` on `backend`.
    fn convert<B: GpuBackend>(
        backend: &B,
        cmyk: &[u8],
        grid: &CmykGrid,
    ) -> gpu::backend::Result<Vec<Self>>;
}

impl ClutSample for u8 {
    fn convert<B: GpuBackend>(
        backend: &B,
        cmyk: &[u8],
        grid: &CmykGrid,
    ) -> gpu::backend::Result<Vec<Self>> {
        gpu::backend::icc_clut(backend, cmyk, grid)
    }
}

impl ClutSample for u16 {
    fn convert<B: GpuBackend>(
        backend: &B,
        cmyk: &[u8],
        grid: &CmykGrid,
    ) -> gpu::backend::Result<Vec<Self>> {
        gpu::backend::icc_clut16(backend, cmyk, grid)
    }
}

/// The RGB conversion of the 8-bit CMYK pixels `cmyk` on the device,
/// through the grid of `conv`, to samples of `T`: those of the conversion's
/// 8-bit transform for `u8`, of its 16-bit transform of the samples
/// widened (`257 · v`) for `u16`.  `None` when the caller converts through
/// the profile transform instead: no backend is attached, the image is
/// below [`gpu::GPU_ICC_CLUT_THRESHOLD`] pixels, the conversion is not a
/// tabulated one, or the device declines or fails.  A device failure is
/// counted and logged (see [`report_failure`]).
pub(super) fn cmyk_clut_to_rgb<T: ClutSample>(
    icc: &IccHandles<'_>,
    cmyk: &[u8],
    conv: &IccConversion,
) -> Option<Vec<T>> {
    if !icc.has_device() || cmyk.len() / 4 < icc_clut_threshold() {
        return None;
    }
    let grid = conv.cmyk_grid()?;
    let grid = &*grid;
    let outcome = 'device: {
        #[cfg(feature = "vulkan")]
        if let Some(vk) = icc.vk_backend {
            if vk.is_lost() || !vk.icc_clut_is_exact() {
                break 'device DeviceIcc::Declined;
            }
            break 'device run(vk, cmyk, grid);
        }
        #[cfg(all(windows, feature = "d3d12"))]
        if let Some(d3d12) = icc.d3d12_backend {
            if !d3d12.icc_clut_is_exact() {
                break 'device DeviceIcc::Declined;
            }
            break 'device run(d3d12, cmyk, grid);
        }
        let Some(ctx) = icc.gpu_ctx else {
            break 'device DeviceIcc::Declined;
        };
        run(
            &gpu::backend::cuda::CudaBackend::from_ctx(Arc::clone(ctx)),
            cmyk,
            grid,
        )
    };
    match outcome {
        DeviceIcc::Converted(rgb) => {
            icc.count(|c| &c.converted);
            Some(rgb)
        }
        DeviceIcc::Failed => {
            icc.count(|c| &c.failed);
            None
        }
        DeviceIcc::Declined => None,
    }
}

/// Effective device ICC CLUT threshold: [`gpu::GPU_ICC_CLUT_THRESHOLD`]
/// unless `PDF_RASTER_ICC_CLUT_THRESHOLD` holds a decimal pixel count (`0` =
/// every conversion on the device, `usize::MAX` = none).  Read once, on
/// first use.
fn icc_clut_threshold() -> usize {
    use std::sync::OnceLock;
    static THRESHOLD: OnceLock<usize> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        super::thresholds::threshold_from_env(
            "PDF_RASTER_ICC_CLUT_THRESHOLD",
            gpu::GPU_ICC_CLUT_THRESHOLD,
        )
    })
}

/// Convert on `backend`; a failure is logged.
fn run<B: GpuBackend, T: ClutSample>(backend: &B, cmyk: &[u8], grid: &CmykGrid) -> DeviceIcc<T> {
    match T::convert(backend, cmyk, grid) {
        Ok(rgb) => DeviceIcc::Converted(rgb),
        Err(e) => {
            report_failure(B::NAME, &e);
            DeviceIcc::Failed
        }
    }
}

/// Log a failed device conversion: once per process at `warn`, whichever
/// backend failed, and at `debug` after that.  Every failure is also
/// counted in the page's
/// [`PageDiagnostics::device_icc_failures`](crate::renderer::PageDiagnostics::device_icc_failures).
fn report_failure(backend: &'static str, e: &dyn std::fmt::Display) {
    static WARNED: AtomicBool = AtomicBool::new(false);
    if WARNED.swap(true, Ordering::Relaxed) {
        log::debug!("{backend} ICC CMYK→RGB failed: {e}; converted on the CPU");
    } else {
        log::warn!(
            "{backend} ICC CMYK→RGB failed: {e}; converted on the CPU \
             (later failures are logged at debug)"
        );
    }
}
