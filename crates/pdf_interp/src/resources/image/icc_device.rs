//! Device dispatch of the ICC CMYK → RGB conversion.
//!
//! An image in an ICC-based CMYK space converts through the profile's
//! `DEFAULT_GRID_N`⁴ lookup table, baked once per profile, whenever the
//! session holds a GPU backend: on the device at or above
//! [`gpu::GPU_ICC_CLUT_THRESHOLD`] pixels, and below it on the CPU by
//! [`gpu::icc_cmyk_to_rgb_cpu`], the evaluator both device kernels
//! reproduce byte for byte.  The bytes of a conversion therefore depend on
//! neither the image's size nor the backend, nor on whether the device
//! call succeeded.  A session without a backend converts through the
//! profile transform itself.
//!
//! The Vulkan backend is used when the session holds one, otherwise the
//! CUDA context; both run [`gpu::backend::icc_clut`].  A Vulkan device
//! whose kernel fails the backend's exactness check
//! (`VulkanBackend::icc_clut_is_exact`) is not used.

#![cfg(feature = "gpu-icc")]

use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use gpu::GpuCtx;
use gpu::backend::GpuBackend;
#[cfg(feature = "vulkan")]
use gpu::backend::vulkan::VulkanBackend;

use crate::resources::colorspace::icc::IccConversion;
use crate::resources::colorspace::icc::clut::DEFAULT_GRID_N;

/// The GPU backends an ICC CMYK conversion may run on, and the counters
/// its outcome is recorded in.
pub struct IccHandles<'a> {
    /// CUDA context; used when no Vulkan backend is attached.
    pub gpu_ctx: Option<&'a Arc<GpuCtx>>,
    /// Vulkan backend; used in preference to the CUDA context.
    #[cfg(feature = "vulkan")]
    pub vk_backend: Option<&'a VulkanBackend>,
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
            counts: None,
        }
    }

    /// Whether a backend is attached.
    const fn has_device(&self) -> bool {
        #[cfg(feature = "vulkan")]
        if self.vk_backend.is_some() {
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

    /// Conversions the device took and failed, each then performed on the
    /// CPU.
    #[must_use]
    pub const fn failed(&self) -> u32 {
        self.failed.get()
    }
}

/// The outcome of offering a conversion to a device.
enum DeviceIcc {
    /// The device converted the pixels.
    Converted(Vec<u8>),
    /// The device does not take the conversion.
    Declined,
    /// The device failed.
    Failed,
}

/// The RGB conversion of the CMYK pixels `cmyk` through the lookup table
/// of `conv`, or `None` when no backend is attached or the table cannot be
/// baked, the caller then converting through the profile transform.
///
/// Images below [`gpu::GPU_ICC_CLUT_THRESHOLD`] pixels, and those the
/// device declines or fails, convert on the CPU by
/// [`gpu::icc_cmyk_to_rgb_cpu`], which the device reproduces byte for
/// byte.  A device failure is counted and logged (see
/// [`report_failure`]).
pub(super) fn cmyk_clut_to_rgb(
    icc: &IccHandles<'_>,
    cmyk: &[u8],
    conv: &IccConversion,
) -> Option<Vec<u8>> {
    if !icc.has_device() {
        return None;
    }
    let clut = conv.cmyk_clut()?;
    let clut = &*clut;
    let cpu = || gpu::icc_cmyk_to_rgb_cpu(cmyk, Some((clut, DEFAULT_GRID_N)));
    if cmyk.len() / 4 < icc_clut_threshold() {
        return Some(cpu());
    }
    let outcome = 'device: {
        #[cfg(feature = "vulkan")]
        if let Some(vk) = icc.vk_backend {
            if !vk.icc_clut_is_exact() {
                break 'device DeviceIcc::Declined;
            }
            break 'device run(vk, cmyk, clut);
        }
        let Some(ctx) = icc.gpu_ctx else {
            break 'device DeviceIcc::Declined;
        };
        run(
            &gpu::backend::cuda::CudaBackend::from_ctx(Arc::clone(ctx)),
            cmyk,
            clut,
        )
    };
    match outcome {
        DeviceIcc::Converted(rgb) => {
            icc.count(|c| &c.converted);
            Some(rgb)
        }
        DeviceIcc::Failed => {
            icc.count(|c| &c.failed);
            Some(cpu())
        }
        DeviceIcc::Declined => Some(cpu()),
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
        let Ok(value) = std::env::var("PDF_RASTER_ICC_CLUT_THRESHOLD") else {
            return gpu::GPU_ICC_CLUT_THRESHOLD;
        };
        value.trim().parse().unwrap_or_else(|_| {
            log::warn!(
                "PDF_RASTER_ICC_CLUT_THRESHOLD={value:?} is not a pixel count \
                 (a non-negative integer); using the default"
            );
            gpu::GPU_ICC_CLUT_THRESHOLD
        })
    })
}

/// Convert on `backend`; a failure is logged.
fn run<B: GpuBackend>(backend: &B, cmyk: &[u8], clut: &[u8]) -> DeviceIcc {
    match gpu::backend::icc_clut(backend, cmyk, clut) {
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
