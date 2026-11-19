//! The Direct3D 12 backend of the integration tests.

use rasterrocket_gpu::backend::d3d12::D3d12Backend;
use rasterrocket_gpu::backend::page_device::PageDevice as _;
use rasterrocket_gpu::backend::select::DeviceSelector;

/// The backend on the adapter `PDF_RASTER_D3D12_DEVICE` names (`warp` or
/// a device selector), or on the automatically chosen one when it is
/// unset or empty; `None` after reporting the skip when none opens.
///
/// # Panics
/// When the variable holds a selector that does not parse, or when no
/// adapter opens and `PDF_RASTER_REQUIRE_D3D12` is `1`.
pub(crate) fn test_backend() -> Option<D3d12Backend> {
    let selector = std::env::var("PDF_RASTER_D3D12_DEVICE")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.parse::<DeviceSelector>()
                .expect("PDF_RASTER_D3D12_DEVICE holds a device selector")
        });
    match D3d12Backend::open(selector.as_ref()) {
        Ok(backend) => Some(backend),
        Err(e) => {
            assert!(
                !std::env::var("PDF_RASTER_REQUIRE_D3D12").is_ok_and(|v| v == "1"),
                "PDF_RASTER_REQUIRE_D3D12=1, but no D3D12 backend opens: {e}"
            );
            eprintln!("skipping: no D3D12 backend: {e}");
            None
        }
    }
}
