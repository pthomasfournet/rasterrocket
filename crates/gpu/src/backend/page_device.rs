//! The whole-page raster of a backend: a [`PageCommands`] list rendered
//! on one device, one submission and one wait per page.

use std::time::Duration;

use crate::backend::select::DeviceSelector;
use crate::backend::{BackendError, GpuBackend, Result};
use crate::page::{PageCommands, PageSample};

/// The duration of each stage of one page rendered by
/// [`PageDevice::render_page_timed`].
///
/// The host stages run in order and together span the call; the device
/// stages are measured by timestamps the page's command buffer writes
/// between them, so they lie within `submit_wait`.  A transfer the device
/// does not need (its buffers are host-mapped) takes no device time.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PageTimes {
    /// Host: binning the commands into tiles and packing the coverage
    /// tables.
    pub prepare: Duration,
    /// Host: writing the inputs into the staging buffers, or into the
    /// device buffers in place.
    pub write: Duration,
    /// Host: beginning and recording the page's command buffer.
    pub record: Duration,
    /// Host: submitting the page and waiting for it to complete.
    pub submit_wait: Duration,
    /// Host: copying the pixels into the caller's samples.
    pub read: Duration,
    /// Device: copying the inputs from the staging buffers, and zeroing
    /// the status words of a page with outlines.
    pub device_upload: Duration,
    /// Device: the raster dispatch.
    pub device_raster: Duration,
    /// Device: copying the page into the read-back buffer.
    pub device_download: Duration,
}

/// A backend that renders whole pages from a [`PageCommands`] list.
///
/// Closed to the backends of this crate, as [`GpuBackend`] is.
pub trait PageDevice: GpuBackend + Sized {
    /// Open the device `selector` names, or the automatically ranked one
    /// without a selector, with its kernels.
    ///
    /// # Errors
    /// [`BackendError::DeviceSelection`] when `selector` matches no device,
    /// several devices, or a device that is not eligible; an error when no
    /// suitable device is present, a required feature is missing, a kernel
    /// fails to load, or a driver call fails.
    fn open(selector: Option<&DeviceSelector>) -> Result<Self>;

    /// Check that `selector` names exactly one device that can host the
    /// backend, without opening it.
    ///
    /// # Errors
    /// [`BackendError::DeviceSelection`] when it does not resolve; the
    /// driver's error when the API is unavailable.
    fn check_selector(selector: &DeviceSelector) -> Result<()>;

    /// The enumeration indices of the devices `selectors` name, in their
    /// order, without opening any: each resolves as in [`Self::open`], and
    /// the index, as a selector, names the same device for the rest of the
    /// process.
    ///
    /// # Errors
    /// [`BackendError::DeviceSelection`] when `selectors` is empty, one
    /// does not resolve, or two name the same device; the driver's error
    /// when the API is unavailable.
    fn resolve_devices(selectors: &[DeviceSelector]) -> Result<Vec<usize>>;

    /// The enumeration indices of every device that can host the backend
    /// and is a GPU, not a CPU implementation, without opening any.
    ///
    /// # Errors
    /// [`BackendError::DeviceSelection`] when there is none; the driver's
    /// error when the API is unavailable.
    fn usable_gpus() -> Result<Vec<usize>>;

    /// The device's position in enumeration order: as a selector, it
    /// names this device.
    #[must_use]
    fn device_index(&self) -> usize;

    /// The device's name.
    #[must_use]
    fn device_name(&self) -> String;

    /// The mean time a page raster ([`Self::render_page`]) holds the
    /// device, from recording its commands to the end of the wait for
    /// them: an exponentially weighted mean over the page rasters that
    /// succeeded, the newest weighing 1/8, the backend's first excluded.
    /// `None` until a page raster after the first has succeeded.
    #[must_use]
    fn page_mean(&self) -> Option<Duration>;

    /// Whether the device rasterises outlines ([`crate::page::AreaPath`]):
    /// it supports 64-bit integers in kernels.  A page with outlines
    /// renders only on such a device.
    #[must_use]
    fn page_paths(&self) -> bool;

    /// Render `page` on the device and write its pixels into `out`, RGB
    /// samples of the page's depth: row `y` at
    /// `out[y × stride..][..width × 3]`, `stride` in samples.  Samples
    /// between rows are left unchanged.
    ///
    /// Concurrent calls stage their inputs and read their pixels in
    /// parallel, each with buffers of its own, and run on the device one
    /// at a time.
    ///
    /// # Errors
    ///
    /// [`BackendError::DeviceLimit`] when an input or the page exceeds a
    /// buffer limit of the device, before anything is allocated;
    /// [`BackendError::OutOfMemory`] when the page's buffers do not fit in
    /// the memory left; [`BackendError::KernelFault`] when the raster of
    /// the page's outlines reports a division it could not complete
    /// exactly, its pixels then being discarded; otherwise an error when
    /// the page has no pixel, its depth is not that of `S`, `out` or
    /// `stride` is too small for it, it has outlines the device does not
    /// rasterise, an allocation fails otherwise, or the device reports an
    /// error.  `out` is unchanged then.
    fn render_page<S: PageSample>(
        &self,
        page: &PageCommands,
        out: &mut [S],
        stride: usize,
    ) -> Result<()>;

    /// As [`Self::render_page`], and return the duration of each of the
    /// page's stages.
    ///
    /// # Errors
    ///
    /// As [`Self::render_page`]; also when the device writes no
    /// timestamps, before anything is staged.
    fn render_page_timed<S: PageSample>(
        &self,
        page: &PageCommands,
        out: &mut [S],
        stride: usize,
    ) -> Result<PageTimes>;

    /// Whether `error` retires the device: every error but `DeviceLimit`
    /// and `OutOfMemory`, which concern the page alone.
    #[must_use]
    fn retires_device(error: &BackendError) -> bool {
        !matches!(
            error,
            BackendError::DeviceLimit { .. } | BackendError::OutOfMemory { .. }
        )
    }
}

#[cfg(all(test, feature = "vulkan"))]
mod image_tests;
#[cfg(all(test, feature = "vulkan"))]
mod mask_tests;
#[cfg(all(test, feature = "vulkan"))]
mod modifier_tests;
#[cfg(all(test, feature = "vulkan"))]
pub(crate) mod path_tests;
#[cfg(all(test, feature = "vulkan"))]
mod tests;
