//! The GPU page queue: pages rendered whole on Vulkan devices beside the
//! CPU workers ([`SessionConfig::gpu_pages`]).
//!
//! A session renders its GPU pages on one device, or on several
//! ([`SessionConfig::gpu_devices`]).  Unless the session holds its devices
//! already, each is opened on a thread of its own when the first page that
//! qualifies asks for one, so a job none of whose pages qualifies opens
//! none.  Every page that qualifies renders on a device: its worker waits
//! until a device is open (until [`OPEN_WAIT`] after the first page that
//! waited), takes the open device in
//! service where it expects the shortest wait (the pages holding the
//! device and itself, times the device's mean time per page raster:
//! [`choose`]),
//! records the page's paint commands and renders them on it.  The workers record in parallel; each device
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
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

#[cfg(test)]
use gpu::backend::BackendError;
use gpu::backend::vulkan::VulkanBackend;
use pdf_interp::renderer::{CanvasBuffers, DevicePage, DeviceRecording, PageRenderer};

pub use counts::DevicePageCounts;
use counts::{Miss, Outcome};
pub(crate) use devices::open_device;
use devices::{Claim, Device, Load, Openings, State};

use crate::render::PageJob;
use crate::{BackendPolicy, CancelToken, RasterError};

/// The longest the pages of a session wait for a device to open, from the
/// first page that waits; pages render on the CPU afterwards while none is
/// open.
const OPEN_WAIT: Duration = crate::GPU_PAGE_OPEN_WAIT;

/// The interval at which a page waiting for a device to open checks its
/// cancellation.
const CANCEL_POLL: Duration = Duration::from_millis(100);

/// The devices of a session's GPU pages.
pub(crate) struct DevicePages {
    /// The devices, in the order of [`crate::SessionConfig::gpu_devices`].
    devices: Vec<Device>,
    /// The wake-up of pages waiting for a device to open.
    openings: Arc<Openings>,
    /// The longest the pages wait for a device to open: [`OPEN_WAIT`].
    open_wait: Duration,
    /// The instant after which no page waits for a device to open: set by
    /// the first page that waits, `open_wait` later.
    open_deadline: OnceLock<Instant>,
    /// Set once a page has found `open_deadline` passed.
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

/// An open GPU page device and the selector it is opened with.
type OpenDevice = (Option<String>, Arc<VulkanBackend>);

/// The GPU page devices of `config` under [`BackendPolicy::ForceVulkan`],
/// with the selector each is opened with: `session`, the session's own
/// device, for the entry that names it (or for
/// [`crate::SessionConfig::vulkan_device`] when
/// [`crate::SessionConfig::gpu_devices`] is empty), each other one opened
/// now.
///
/// # Errors
///
/// As [`selectors`] and [`crate::vk_init::init_vk_backend`].
fn forced_backends(
    config: &crate::SessionConfig,
    session: &Arc<VulkanBackend>,
) -> Result<Vec<OpenDevice>, RasterError> {
    if config.gpu_devices.is_empty() {
        return Ok(vec![(config.vulkan_device.clone(), Arc::clone(session))]);
    }
    let own = Some(session.device_index().to_string());
    selectors(config)?
        .into_iter()
        .map(|s| {
            if s == own {
                return Ok((s, Arc::clone(session)));
            }
            crate::vk_init::init_vk_backend(BackendPolicy::ForceVulkan, s.as_deref())?
                .map(|b| (s, b))
                .ok_or_else(|| {
                    RasterError::BackendUnavailable(
                        "the Vulkan backend opened no GPU page device".to_owned(),
                    )
                })
        })
        .collect()
}

/// The GPU page devices of `config` opened now, beside `session`, the
/// session's own device: under [`BackendPolicy::ForceVulkan`] every one
/// ([`forced_backends`]), or the error of the first that does not open;
/// under `Auto` with [`crate::SessionConfig::gpu_pages`] those that open,
/// each that does not being logged at `error`; otherwise none.
///
/// # Errors
///
/// As [`selectors`]; under `ForceVulkan`, as [`forced_backends`].
pub(crate) fn open_all(
    config: &crate::SessionConfig,
    session: Option<&Arc<VulkanBackend>>,
) -> Result<Vec<Arc<VulkanBackend>>, RasterError> {
    if !config.gpu_pages {
        return Ok(Vec::new());
    }
    match (config.policy, session) {
        (BackendPolicy::ForceVulkan, Some(session)) => Ok(forced_backends(config, session)?
            .into_iter()
            .map(|(_, backend)| backend)
            .collect()),
        (BackendPolicy::Auto, _) => Ok(selectors(config)?
            .iter()
            .filter_map(|s| open_device(s.as_deref()))
            .collect()),
        _ => Ok(Vec::new()),
    }
}

impl DevicePages {
    /// The GPU pages of a session opened with `config` that holds
    /// `backend`: `None` unless the config asks for them.  Under
    /// [`BackendPolicy::ForceVulkan`] the devices are open
    /// ([`forced_backends`]); otherwise each is opened when a page first
    /// asks for one.
    ///
    /// # Errors
    ///
    /// As [`selectors`]; under `ForceVulkan`, as [`forced_backends`].
    pub(crate) fn for_session(
        config: &crate::SessionConfig,
        backend: Option<&Arc<VulkanBackend>>,
    ) -> Result<Option<Self>, RasterError> {
        if !config.gpu_pages {
            return Ok(None);
        }
        let devices = match backend {
            Some(session) => forced_backends(config, session)?
                .into_iter()
                .map(|(s, b)| Device::open(s, b))
                .collect(),
            None => selectors(config)?
                .into_iter()
                .map(Device::deferred)
                .collect(),
        };
        Ok(Some(Self::new(devices)))
    }

    fn new(devices: Vec<Device>) -> Self {
        Self {
            devices,
            openings: Arc::default(),
            open_wait: OPEN_WAIT,
            open_deadline: OnceLock::new(),
            open_timed_out: AtomicBool::new(false),
            wide_declined: AtomicBool::new(false),
            counts: Mutex::new(DevicePageCounts::default()),
            #[cfg(test)]
            faults: TestFaults::default(),
        }
    }

    /// The open device in service where one more page expects the
    /// shortest wait ([`choose`]), the first listed among equals; else
    /// whether one may still open (`Ok(None)`) or why none will.
    fn pick(&self) -> Result<Option<Claim<'_>>, Miss> {
        let mut open: Vec<(&Device, &VulkanBackend, Load)> = Vec::new();
        let (mut opening, mut retired) = (false, false);
        for device in &self.devices {
            match device.state() {
                State::Open(backend, load) => open.push((device, backend, load)),
                State::Opening => opening = true,
                State::Retired => retired = true,
                State::NoDevice => {}
            }
        }
        let loads: Vec<Load> = open.iter().map(|&(_, _, load)| load).collect();
        match choose(&loads).map(|i| open[i]) {
            Some((device, backend, _)) => Ok(Some(device.claim(backend))),
            None if opening => Ok(None),
            None if retired => Err(Miss::Retired),
            None => Err(Miss::NoDevice),
        }
    }

    /// A device for a page, or why there is none.
    ///
    /// While no device is open but one may still open, every device not
    /// yet opening starts opening and the call waits for one, until
    /// [`OPEN_WAIT`] after the first call that waited: the deadline is the
    /// session's, so once it has passed a call returns [`Miss::Opening`]
    /// at once while no device is open, and a device that opens later is
    /// still taken.  The first call that finds the deadline passed is
    /// logged at `warn`.  The wait polls `cancel` every [`CANCEL_POLL`] and
    /// returns [`Miss::Cancelled`] once it is raised.
    pub(crate) fn claim(&self, cancel: Option<&CancelToken>) -> Result<Claim<'_>, Miss> {
        if let Some(claim) = self.pick()? {
            return Ok(claim);
        }
        for device in &self.devices {
            device.start_opening(&self.openings);
        }
        let deadline = *self
            .open_deadline
            .get_or_init(|| Instant::now() + self.open_wait);
        let mut guard = self.openings.lock();
        loop {
            if let Some(claim) = self.pick()? {
                return Ok(claim);
            }
            if cancel.is_some_and(CancelToken::is_cancelled) {
                return Err(Miss::Cancelled);
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
            let slice = if cancel.is_some() {
                left.min(CANCEL_POLL)
            } else {
                left
            };
            guard = self.openings.wait(guard, slice);
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

/// The index of the load where one more page expects the shortest wait,
/// the first among equals: the held pages and that page times the device's
/// mean page raster, the shortest mean known for a device without one yet
/// (1 ns when none has one), so a new device takes pages in turn with the
/// fastest until its own page rasters give it a mean.
fn choose(loads: &[Load]) -> Option<usize> {
    let fallback = loads.iter().filter_map(|l| l.mean_nanos).min().unwrap_or(1);
    loads
        .iter()
        .enumerate()
        .min_by_key(|&(i, l)| (l.wait(fallback), i))
        .map(|(i, _)| i)
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
    let claim = match pages.claim(job.params.cancel.as_ref()) {
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
