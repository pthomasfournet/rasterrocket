//! The GPU page queue: pages rendered whole on Vulkan devices beside the
//! CPU workers ([`SessionConfig::gpu_pages`]).
//!
//! A session renders its GPU pages on one device, or on several
//! ([`SessionConfig::gpu_devices`]).  Unless the session holds its devices
//! already, each is opened on a thread of its own when the first page that
//! qualifies asks for one, so a job none of whose pages qualifies opens
//! none.  Every page that qualifies renders on a device: its worker waits
//! until a device is open (at most [`OPEN_WAIT`]), takes the open device in
//! service that the fewest pages hold, records the page's paint commands
//! and renders them on it.  The workers record in parallel; each device
//! renders one page at a time while others stage their inputs or read
//! their pixels back.  A page that does not qualify renders on the CPU.
//! Pages are returned by index, so the output order does not depend on
//! where a page rendered.
//!
//! A device error retires that device for the session: later pages render
//! on the session's other devices, or on the CPU once none is in service.
//! A page too large for a device's limits renders on the CPU and leaves the
//! device in service.
//!
//! Where every page offered went is counted for the session
//! ([`DevicePageCounts`]) and logged at `info` when the session drops.
//!
//! [`SessionConfig::gpu_pages`]: crate::SessionConfig::gpu_pages
//! [`SessionConfig::gpu_devices`]: crate::SessionConfig::gpu_devices

mod counts;
mod devices;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

#[cfg(test)]
use gpu::backend::BackendError;
use gpu::backend::vulkan::VulkanBackend;
use pdf_interp::renderer::{CanvasBuffers, DevicePage, DeviceRecording, PageRenderer};

pub use counts::DevicePageCounts;
use counts::{Miss, Outcome};
pub(crate) use devices::open_device;
use devices::{Claim, Device, Openings, State};

use crate::render::PageJob;
use crate::{BackendPolicy, RasterError};

/// The longest a page waits for a device to open before it renders on the
/// CPU.
const OPEN_WAIT: Duration = Duration::from_secs(30);

/// The devices of a session's GPU pages.
pub(crate) struct DevicePages {
    /// The devices, in the order of [`crate::SessionConfig::gpu_devices`].
    devices: Vec<Device>,
    /// The wake-up of pages waiting for a device to open.
    openings: Arc<Openings>,
    /// The longest a page waits for a device to open: [`OPEN_WAIT`].
    open_wait: Duration,
    /// Set once a page has stopped waiting for a device to open.
    open_timed_out: AtomicBool,
    /// Set once a page of the wide working space has been rendered on the
    /// CPU in its place.
    wide_declined: AtomicBool,
    /// Where the pages offered went, but for [`DevicePageCounts::by_device`].
    counts: Mutex<DevicePageCounts>,
    /// Test hooks.
    #[cfg(test)]
    faults: TestFaults,
}

/// See [`DevicePages::faults`].
#[cfg(test)]
#[derive(Default)]
pub(crate) struct TestFaults {
    /// The error the next device page returns instead of rendering.
    fail_next: Mutex<Option<BackendError>>,
    /// Cancelled as the next recording starts.
    cancel_recording: Mutex<Option<crate::CancelToken>>,
    /// Pages executed again on the CPU after their recording.
    cpu_reruns: std::sync::atomic::AtomicU32,
}

#[cfg(test)]
impl TestFaults {
    /// The value `slot` holds, leaving it empty.
    fn take<T>(slot: &Mutex<Option<T>>) -> Option<T> {
        slot.lock().unwrap_or_else(PoisonError::into_inner).take()
    }
}

/// The selectors of the GPU page devices of `config`, one per device:
/// [`crate::SessionConfig::gpu_devices`] resolved to device indices, or
/// [`crate::SessionConfig::vulkan_device`] when it is empty.
///
/// # Errors
///
/// As [`crate::vk_init::resolve_page_devices`].
fn selectors(config: &crate::SessionConfig) -> Result<Vec<Option<String>>, RasterError> {
    if config.gpu_devices.is_empty() {
        Ok(vec![config.vulkan_device.clone()])
    } else {
        crate::vk_init::resolve_page_devices(config.policy, &config.gpu_devices)
    }
}

/// The GPU page devices of `config` opened now: under
/// [`BackendPolicy::ForceVulkan`] every one, or the error of the first that
/// does not open; under `Auto` with [`crate::SessionConfig::gpu_pages`]
/// those that open, each that does not being logged at `error`; otherwise
/// none.
///
/// # Errors
///
/// As [`selectors`]; under `ForceVulkan`, as
/// [`crate::vk_init::init_vk_backend`].
pub(crate) fn open_all(
    config: &crate::SessionConfig,
) -> Result<Vec<Arc<VulkanBackend>>, RasterError> {
    if !config.gpu_pages {
        return Ok(Vec::new());
    }
    let selectors = selectors(config)?;
    match config.policy {
        BackendPolicy::ForceVulkan => selectors
            .iter()
            .filter_map(|s| {
                crate::vk_init::init_vk_backend(config.policy, s.as_deref()).transpose()
            })
            .collect(),
        BackendPolicy::Auto => Ok(selectors
            .iter()
            .filter_map(|s| open_device(s.as_deref()))
            .collect()),
        _ => Ok(Vec::new()),
    }
}

impl DevicePages {
    /// The GPU pages of a session opened with `config` that holds
    /// `backend`: `None` unless the config asks for them.  Under
    /// [`BackendPolicy::ForceVulkan`] the devices are open: `backend` when
    /// [`crate::SessionConfig::gpu_devices`] is empty, else each listed
    /// device, `backend` for its own device and the others opened now;
    /// otherwise each is opened when a page first asks for one.
    ///
    /// # Errors
    ///
    /// As [`selectors`]; under `ForceVulkan`, as
    /// [`crate::vk_init::init_vk_backend`].
    pub(crate) fn for_session(
        config: &crate::SessionConfig,
        backend: Option<&Arc<VulkanBackend>>,
    ) -> Result<Option<Self>, RasterError> {
        if !config.gpu_pages {
            return Ok(None);
        }
        let selectors = selectors(config)?;
        let devices = match (backend, config.gpu_devices.is_empty()) {
            (Some(backend), true) => vec![Device::open(
                config.vulkan_device.clone(),
                Arc::clone(backend),
            )],
            (Some(session), false) => selectors
                .into_iter()
                .map(|s| {
                    if s == Some(session.device_index().to_string()) {
                        return Ok(Device::open(s, Arc::clone(session)));
                    }
                    crate::vk_init::init_vk_backend(BackendPolicy::ForceVulkan, s.as_deref())?
                        .map(|b| Device::open(s, b))
                        .ok_or_else(|| {
                            RasterError::BackendUnavailable(
                                "the Vulkan backend opened no GPU page device".to_owned(),
                            )
                        })
                })
                .collect::<Result<_, _>>()?,
            (None, _) => selectors.into_iter().map(Device::deferred).collect(),
        };
        Ok(Some(Self::new(devices)))
    }

    fn new(devices: Vec<Device>) -> Self {
        Self {
            devices,
            openings: Arc::default(),
            open_wait: OPEN_WAIT,
            open_timed_out: AtomicBool::new(false),
            wide_declined: AtomicBool::new(false),
            counts: Mutex::new(DevicePageCounts::default()),
            #[cfg(test)]
            faults: TestFaults::default(),
        }
    }

    /// The open device in service that the fewest pages hold, the first
    /// listed among equals; else whether one may still open (`Ok(None)`)
    /// or why none will.
    fn pick(&self) -> Result<Option<Claim<'_>>, Miss> {
        let mut best: Option<(&Device, &VulkanBackend, usize)> = None;
        let (mut opening, mut retired) = (false, false);
        for device in &self.devices {
            match device.state() {
                State::Open(backend, held) => {
                    if best.is_none_or(|(_, _, least)| held < least) {
                        best = Some((device, backend, held));
                    }
                }
                State::Opening => opening = true,
                State::Retired => retired = true,
                State::NoDevice => {}
            }
        }
        match best {
            Some((device, backend, _)) => Ok(Some(device.claim(backend))),
            None if opening => Ok(None),
            None if retired => Err(Miss::Retired),
            None => Err(Miss::NoDevice),
        }
    }

    /// A device for a page, or why there is none.  While no device is open
    /// but one may still open, every device not yet opening starts opening
    /// and the call waits for one, at most [`OPEN_WAIT`]; the first call
    /// that stops waiting is logged at `warn`.
    pub(crate) fn claim(&self) -> Result<Claim<'_>, Miss> {
        if let Some(claim) = self.pick()? {
            return Ok(claim);
        }
        for device in &self.devices {
            device.start_opening(&self.openings);
        }
        let deadline = Instant::now() + self.open_wait;
        let mut guard = self.openings.lock();
        loop {
            if let Some(claim) = self.pick()? {
                return Ok(claim);
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                drop(guard);
                if !self.open_timed_out.swap(true, Ordering::Relaxed) {
                    log::warn!(
                        "rasterrocket: no GPU page device has opened after {:.0} s; pages \
                         render on the CPU until one does",
                        self.open_wait.as_secs_f64()
                    );
                }
                return Err(Miss::Opening);
            }
            guard = self.openings.wait(guard, left);
        }
    }

    /// Count one page under `outcome`.
    fn count(&self, outcome: Outcome) {
        self.counts
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .add(outcome);
    }

    /// Where the pages offered so far went.
    pub(crate) fn counts(&self) -> DevicePageCounts {
        let mut counts = self
            .counts
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        counts.by_device = self
            .devices
            .iter()
            .map(|d| (d.name(), d.rendered.load(Ordering::Relaxed)))
            .collect();
        counts
    }

    /// The warning of page `page_num` of the wide working space rendered
    /// on the CPU, the first such page of the session.
    fn wide_declined(&self, page_num: u32) -> Option<String> {
        (!self.wide_declined.swap(true, Ordering::Relaxed)).then(|| {
            format!(
                "page {page_num} is composited in the wide working space, which the GPU page \
                 raster does not paint: it and every later page of that space render on the \
                 CPU; the sRGB working space (SessionConfig::working_space, rrocket \
                 --working-space srgb) keeps them on the GPU, their colours outside sRGB \
                 clipped"
            )
        })
    }
}

impl Drop for DevicePages {
    fn drop(&mut self) {
        let counts = self.counts();
        if counts.rendered > 0 || counts.cpu() > 0 || counts.stopped > 0 {
            log::info!("rasterrocket: GPU pages: {counts}");
        }
        for device in &mut self.devices {
            device.join_opener();
        }
    }
}

/// The reason a page of the wide working space is rendered on the CPU.
const WIDE: &str = "the wide working space";

/// Why `job`'s page cannot be recorded, known before interpretation.
fn ruled_out(job: &PageJob<'_>) -> Option<&'static str> {
    if job.space() != pdf_interp::WorkingSpace::Srgb {
        return Some(WIDE);
    }
    pdf_interp::renderer::device_page_ineligible(
        job.canvas.is_ink(),
        job.group.is_some(),
        job.session.vector_antialias,
        job.ops,
    )
}

/// The paint commands of `job`'s page as the GPU page queue records them,
/// or why the page is not recorded; nothing is rendered.
///
/// # Errors
///
/// As [`PageJob::run`]; [`crate::render::check_aborted`]'s error when the
/// recording stopped on the page's budget, deadline or cancellation.
pub(crate) fn record(
    job: &PageJob<'_>,
) -> Result<Result<gpu::page::PageCommands, &'static str>, RasterError> {
    if let Some(reason) = ruled_out(job) {
        return Ok(Err(reason));
    }
    let mut renderer = job.unpainted_renderer(CanvasBuffers::default())?;
    if let Err(reason) = renderer.record_for_device(true) {
        return Ok(Err(reason));
    }
    job.execute(&mut renderer)?;
    let recording = renderer.take_device_recording();
    crate::render::check_aborted(&renderer)?;
    match recording {
        Some(DeviceRecording::Recorded(page)) => Ok(Ok(*page)),
        Some(DeviceRecording::Refused(reason)) => Ok(Err(reason)),
        None => unreachable!("a renderer recording for the device yields its recording"),
    }
}

/// Render `job`'s page, on one of the session's GPU page devices when it
/// qualifies and a device is open and in service, and on the CPU
/// otherwise; return the renderer that executed it and how the device was
/// used.
///
/// A page whose recording is refused, or which the device fails on, is
/// executed again on the CPU with the bitmap buffers of the first pass;
/// the two passes share the page's deadline, and the CPU pass has its own
/// operator budget, so the page's outcome is the CPU's.  A recording
/// stopped by the operator budget, the deadline or cancellation is
/// returned as it stopped, for the caller to report: the CPU pass would
/// stop the same way.
///
/// # Errors
///
/// As [`PageJob::run`].
pub(crate) fn render<'a>(
    job: &PageJob<'a>,
    buffers: &mut CanvasBuffers,
) -> Result<(PageRenderer<'a>, DevicePage), RasterError> {
    let Some(pages) = job.session.device_pages.as_ref() else {
        return Ok((job.run(buffers)?, DevicePage::NotOffered));
    };
    if let Some(reason) = ruled_out(job) {
        log::debug!("page {}: rendered on the CPU: {reason}", job.page_num);
        pages.count(Outcome::Ineligible(reason));
        if let Some(warning) = (reason == WIDE)
            .then(|| pages.wide_declined(job.page_num))
            .flatten()
        {
            log::warn!("{warning}");
        }
        return Ok((job.run(buffers)?, DevicePage::Ineligible(reason)));
    }
    let claim = match pages.claim() {
        Ok(claim) => claim,
        Err(miss) => {
            log::debug!(
                "page {}: rendered on the CPU: {}",
                job.page_num,
                miss.describe()
            );
            pages.count(Outcome::Missed(miss));
            return Ok((job.run(buffers)?, DevicePage::NotOffered));
        }
    };
    // The device paints every sample of the canvas; a page it does not
    // paint is painted paper white first.
    let mut renderer = job.unpainted_renderer(std::mem::take(buffers))?;
    if let Err(reason) = renderer.record_for_device(claim.backend.page_paths()) {
        drop(claim);
        log::debug!("page {}: rendered on the CPU: {reason}", job.page_num);
        pages.count(Outcome::Ineligible(reason));
        renderer.paint_paper();
        let renderer = job.executed(renderer, buffers)?;
        return Ok((renderer, DevicePage::Ineligible(reason)));
    }
    #[cfg(test)]
    if let Some(token) = TestFaults::take(&pages.faults.cancel_recording) {
        token.cancel();
    }
    let mut renderer = job.executed(renderer, buffers)?;
    let recording = renderer.take_device_recording();
    if renderer.budget_status().is_some() {
        pages.count(Outcome::Stopped);
        renderer.paint_paper();
        return Ok((renderer, DevicePage::NotOffered));
    }
    let outcome = match recording {
        Some(DeviceRecording::Recorded(page)) => {
            #[cfg(test)]
            let painted = TestFaults::take(&pages.faults.fail_next)
                .map_or_else(|| renderer.paint_device_page(claim.backend, &page), Err);
            #[cfg(not(test))]
            let painted = renderer.paint_device_page(claim.backend, &page);
            match painted {
                Ok(()) => {
                    let commands = u32::try_from(page.len()).unwrap_or(u32::MAX);
                    log::debug!(
                        "page {}: rendered on the GPU from {commands} commands",
                        job.page_num
                    );
                    pages.count(Outcome::Rendered);
                    let _ = claim.device.rendered.fetch_add(1, Ordering::Relaxed);
                    return Ok((renderer, DevicePage::Rendered { commands }));
                }
                Err(e) => {
                    claim.device.fail(job.page_num, &e);
                    pages.count(Outcome::Failed);
                    DevicePage::Failed
                }
            }
        }
        Some(DeviceRecording::Refused(reason)) => {
            log::debug!("page {}: rendered on the CPU: {reason}", job.page_num);
            pages.count(Outcome::Ineligible(reason));
            DevicePage::Ineligible(reason)
        }
        None => unreachable!("a renderer recording for the device yields its recording"),
    };
    drop(claim);
    #[cfg(test)]
    let _ = pages.faults.cpu_reruns.fetch_add(1, Ordering::Relaxed);
    let fallback = renderer.take_device_fallback();
    let mut cpu = job.renderer(renderer.into_buffers())?;
    cpu.resume_after_device(fallback);
    Ok((job.executed(cpu, buffers)?, outcome))
}

#[cfg(test)]
mod tests;
