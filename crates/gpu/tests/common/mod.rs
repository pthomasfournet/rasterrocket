//! Helpers shared by the integration tests.

#![cfg(feature = "vulkan")]

use rasterrocket_gpu::backend::Result;
use rasterrocket_gpu::backend::page_device::PageDevice;
use rasterrocket_gpu::backend::select::DeviceSelector;
use rasterrocket_gpu::backend::vulkan::VulkanBackend;

/// A backend on the device `PDF_RASTER_VULKAN_DEVICE` names, or on the
/// automatically ranked one when it is unset or empty.
///
/// # Panics
/// When the variable holds a selector that does not parse.
pub(crate) fn vulkan_backend() -> Result<VulkanBackend> {
    let selector = std::env::var("PDF_RASTER_VULKAN_DEVICE")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.parse::<DeviceSelector>()
                .expect("PDF_RASTER_VULKAN_DEVICE holds a device selector")
        });
    VulkanBackend::open(selector.as_ref())
}
