//! The GPU page queue: pages rendered whole on a Vulkan device beside the
//! CPU workers ([`SessionConfig::gpu_pages`]).
//!
//! The device is opened on a thread of its own when the session opens, so
//! the workers start rendering at once; until it is open every page renders
//! on the CPU.  The device takes one page at a time: a worker whose page
//! qualifies claims it while it is idle, records the page's paint commands
//! and renders them on it, and a worker that finds it busy, or a page that
//! does not qualify, renders on the CPU.  Pages are returned by index, so
//! the output order does not depend on where a page rendered.
//!
//! [`SessionConfig::gpu_pages`]: crate::SessionConfig::gpu_pages

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::thread::JoinHandle;

use gpu::backend::vulkan::VulkanBackend;
use pdf_interp::renderer::{DevicePage, DeviceRecording, PageRenderer};

use crate::RasterError;
use crate::render::PageJob;

/// The device of a session's GPU pages, and whether a page holds it.
pub(crate) struct DevicePages {
    /// The device once its opening has finished: `None` when it failed.
    device: Arc<OnceLock<Option<Arc<VulkanBackend>>>>,
    /// The thread opening the device; joined when the session drops, so
    /// the process never exits while a driver is initialising.
    opener: Mutex<Option<JoinHandle<()>>>,
    /// Set while a page holds the device.
    busy: AtomicBool,
    /// Set once a device failure has been logged.
    failure_logged: AtomicBool,
}

/// The device, held by one page until dropped.
pub(crate) struct Claim<'a> {
    pages: &'a DevicePages,
    /// The device.
    pub(crate) backend: &'a VulkanBackend,
}

impl Drop for Claim<'_> {
    fn drop(&mut self) {
        self.pages.busy.store(false, Ordering::Release);
    }
}

impl DevicePages {
    /// The GPU pages of a session opened with `config` that holds
    /// `backend`: `None` unless the config asks for them, on `backend`
    /// when the session opened one, and otherwise on a device opened
    /// beside the workers.
    ///
    /// # Errors
    ///
    /// As [`Self::open`].
    pub(crate) fn for_session(
        config: &crate::SessionConfig,
        backend: Option<&Arc<VulkanBackend>>,
    ) -> Result<Option<Self>, RasterError> {
        if !config.gpu_pages {
            return Ok(None);
        }
        Ok(Some(match backend {
            Some(backend) => Self::on(Arc::clone(backend)),
            None => Self::open(config.vulkan_device.clone())?,
        }))
    }

    /// GPU pages on `backend`, a device the session has already opened.
    pub(crate) fn on(backend: Arc<VulkanBackend>) -> Self {
        Self {
            device: Arc::new(OnceLock::from(Some(backend))),
            opener: Mutex::new(None),
            busy: AtomicBool::new(false),
            failure_logged: AtomicBool::new(false),
        }
    }

    /// GPU pages on the device `selector` names (the automatically ranked
    /// one when `None`), opened on a new thread.  A device that does not
    /// open is logged at `error` and leaves every page on the CPU.
    ///
    /// # Errors
    ///
    /// [`RasterError::BackendUnavailable`] when the thread cannot start.
    pub(crate) fn open(selector: Option<String>) -> Result<Self, RasterError> {
        let device = Arc::new(OnceLock::new());
        let slot = Arc::clone(&device);
        let opener = std::thread::Builder::new()
            .name("rrocket-gpu-pages".to_owned())
            .spawn(move || {
                let started = std::time::Instant::now();
                let backend = crate::vk_init::open_backend(selector.as_deref())
                    .inspect(|_| {
                        log::debug!(
                            "rasterrocket: GPU page device open after {:.0} ms",
                            started.elapsed().as_secs_f64() * 1e3
                        );
                    })
                    .inspect_err(|e| {
                        log::error!(
                            "rasterrocket: GPU page rendering was requested, but the Vulkan \
                             device did not open ({e}); every page renders on the CPU"
                        );
                    })
                    .ok();
                // The slot is set by this thread alone.
                drop(slot.set(backend));
            })
            .map_err(|e| {
                RasterError::BackendUnavailable(format!(
                    "cannot start the thread that opens the GPU page device: {e}"
                ))
            })?;
        Ok(Self {
            device,
            opener: Mutex::new(Some(opener)),
            busy: AtomicBool::new(false),
            failure_logged: AtomicBool::new(false),
        })
    }

    /// The device, if it is open and no page holds it.
    pub(crate) fn try_claim(&self) -> Option<Claim<'_>> {
        let backend = self.device.get()?.as_deref()?;
        self.busy
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
            .then_some(Claim {
                pages: self,
                backend,
            })
    }

    /// Log a device failure on page `page_num` at `warn`, the first time
    /// only.
    pub(crate) fn log_failure(&self, page_num: u32, error: &dyn std::fmt::Display) {
        if !self.failure_logged.swap(true, Ordering::Relaxed) {
            log::warn!(
                "page {page_num}: the GPU page raster failed ({error}); the page and any \
                 later failing page render on the CPU"
            );
        }
    }
}

impl Drop for DevicePages {
    fn drop(&mut self) {
        let opener = self
            .opener
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(opener) = opener
            && opener.join().is_err()
        {
            log::warn!("rasterrocket: the thread opening the GPU page device panicked");
        }
    }
}

/// Why `job`'s page cannot be recorded, known before interpretation.
fn ruled_out(job: &PageJob<'_>) -> Option<&'static str> {
    if job.ink {
        return Some("overprint simulation");
    }
    if job.group.is_some() {
        return Some("page group");
    }
    if !job.session.vector_antialias {
        return Some("aliased vector paint");
    }
    pdf_interp::renderer::device_page_operators(job.ops).err()
}

/// Render `job`'s page, on the session's GPU page device when it qualifies
/// and the device is free, and on the CPU otherwise; return the renderer
/// that executed it and how the device was used.
///
/// A page whose recording is refused, or which the device fails on, is
/// executed again on the CPU with the bitmap buffers of the first pass.
///
/// # Errors
///
/// As [`PageJob::run`].
pub(crate) fn render<'a>(
    job: &PageJob<'a>,
    buffers: raster::BitmapBuffers,
) -> Result<(PageRenderer<'a>, DevicePage), RasterError> {
    let Some(pages) = job.session.device_pages.as_ref() else {
        return Ok((job.run(buffers)?, DevicePage::NotOffered));
    };
    if let Some(reason) = ruled_out(job) {
        log::debug!("page {}: rendered on the CPU: {reason}", job.page_num);
        return Ok((job.run(buffers)?, DevicePage::Ineligible(reason)));
    }
    let Some(claim) = pages.try_claim() else {
        return Ok((job.run(buffers)?, DevicePage::NotOffered));
    };
    let mut renderer = job.renderer(buffers)?;
    if let Err(reason) = renderer.record_for_device() {
        drop(claim);
        log::debug!("page {}: rendered on the CPU: {reason}", job.page_num);
        job.execute(&mut renderer)?;
        return Ok((renderer, DevicePage::Ineligible(reason)));
    }
    job.execute(&mut renderer)?;
    let outcome = match renderer.take_device_recording() {
        Some(DeviceRecording::Recorded(page)) if renderer.budget_status().is_none() => {
            match renderer.paint_device_page(claim.backend, &page) {
                Ok(()) => {
                    let commands = u32::try_from(page.len()).unwrap_or(u32::MAX);
                    log::debug!(
                        "page {}: rendered on the GPU from {commands} commands",
                        job.page_num
                    );
                    return Ok((renderer, DevicePage::Rendered { commands }));
                }
                Err(e) => {
                    pages.log_failure(job.page_num, &e);
                    DevicePage::Failed
                }
            }
        }
        Some(DeviceRecording::Refused(reason)) => {
            log::debug!("page {}: rendered on the CPU: {reason}", job.page_num);
            DevicePage::Ineligible(reason)
        }
        // A render stopped by its budget is reported by the CPU pass.
        _ => DevicePage::NotOffered,
    };
    drop(claim);
    let buffers = renderer.try_finish().map_or_else(
        |_| raster::BitmapBuffers::default(),
        |(bitmap, _)| bitmap.into_buffers(),
    );
    Ok((job.run(buffers)?, outcome))
}
