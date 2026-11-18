//! The GPU page queue: pages rendered whole on a Vulkan device beside the
//! CPU workers ([`SessionConfig::gpu_pages`]).
//!
//! Unless the session holds a device of its own, the device is opened on a
//! thread of its own when the first page that qualifies asks for it, so a
//! job none of whose pages qualifies never opens it.  Every page that
//! qualifies renders on the device: its worker waits for the device to
//! open (at most [`OPEN_WAIT`]), records the page's paint commands and
//! renders them on it.  The workers record in parallel; the device renders
//! one page at a time while others stage their inputs or read their pixels
//! back.  A page that does not qualify renders on the CPU.  Pages are
//! returned by index, so the output order does not depend on where a page
//! rendered.
//!
//! A device error retires the device for the session: every later page
//! renders on the CPU.  A page too large for the device's limits renders
//! on the CPU and leaves the device in service.
//!
//! Where every page offered went is counted for the session
//! ([`DevicePageCounts`]) and logged at `info` when the session drops.
//!
//! [`SessionConfig::gpu_pages`]: crate::SessionConfig::gpu_pages

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use gpu::backend::BackendError;
use gpu::backend::vulkan::VulkanBackend;
use pdf_interp::renderer::{CanvasBuffers, DevicePage, DeviceRecording, PageRenderer};

use crate::RasterError;
use crate::render::PageJob;

/// The longest a page waits for the device to open before it renders on
/// the CPU.
const OPEN_WAIT: Duration = Duration::from_secs(30);

/// The device of a session's GPU pages.
pub(crate) struct DevicePages {
    /// The device once its opening has finished.
    device: Arc<Opened>,
    /// The opening of the device.
    opener: Mutex<Opener>,
    /// Set once the device has failed; it is offered no page afterwards.
    retired: AtomicBool,
    /// The longest a page waits for the device to open: [`OPEN_WAIT`].
    open_wait: Duration,
    /// Set once a page has stopped waiting for the device to open.
    open_timed_out: AtomicBool,
    /// Set once a page of the wide working space has been rendered on the
    /// CPU in its place.
    wide_declined: AtomicBool,
    /// Where the pages offered went.
    counts: Mutex<DevicePageCounts>,
    /// Test hooks.
    #[cfg(test)]
    faults: TestFaults,
}

/// A device opened on another thread, once its opening has finished:
/// `None` when it failed.
#[derive(Default)]
struct Opened {
    slot: OnceLock<Option<Arc<VulkanBackend>>>,
    /// Held while the slot is set; `settled` is notified once it is.
    lock: Mutex<()>,
    settled: Condvar,
}

impl Opened {
    fn new(slot: OnceLock<Option<Arc<VulkanBackend>>>) -> Self {
        Self {
            slot,
            ..Self::default()
        }
    }

    /// The outcome of the opening, once it has finished.
    fn get(&self) -> Option<&Option<Arc<VulkanBackend>>> {
        self.slot.get()
    }

    /// Record the outcome of the opening and wake the pages waiting for
    /// it.  Only the first outcome is kept.
    fn settle(&self, backend: Option<Arc<VulkanBackend>>) {
        let guard = self.lock.lock().unwrap_or_else(PoisonError::into_inner);
        drop(self.slot.set(backend));
        drop(guard);
        self.settled.notify_all();
    }

    /// The outcome of the opening, waiting at most `wait` for it.
    fn wait(&self, wait: Duration) -> Option<&Option<Arc<VulkanBackend>>> {
        let deadline = Instant::now() + wait;
        let mut guard = self.lock.lock().unwrap_or_else(PoisonError::into_inner);
        loop {
            if let Some(outcome) = self.slot.get() {
                return Some(outcome);
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return None;
            }
            guard = self
                .settled
                .wait_timeout(guard, left)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }
}

/// The opening of a session's GPU page device.
enum Opener {
    /// Not started: the selector of the device to open, the automatically
    /// ranked one when `None`.
    Deferred(Option<String>),
    /// The thread opening the device; joined when the session drops, so
    /// the process never exits while a driver is initialising.
    Running(JoinHandle<()>),
    /// Nothing to open or to join: the device came with the session, or
    /// its opening could not start.
    Settled,
}

/// Where the pages of a GPU-page session went
/// ([`RasterSession::device_page_counts`](crate::RasterSession::device_page_counts)).
///
/// Every page the session rendered, or stopped rendering, is counted once,
/// under the first of these that applies to it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct DevicePageCounts {
    /// Pages the device rendered.
    pub rendered: u32,
    /// Pages painting with something the GPU page raster does not render,
    /// by reason, in the order each reason first occurred; rendered on the
    /// CPU.
    pub ineligible: Vec<(&'static str, u32)>,
    /// Pages rendered on the CPU because the device had not opened after
    /// they waited 30 seconds for it.
    pub opening: u32,
    /// Pages rendered on the CPU because the device did not open.
    pub no_device: u32,
    /// Pages rendered on the CPU after a device error retired the device.
    pub retired: u32,
    /// Pages the device failed on, or which exceed its limits; rendered on
    /// the CPU.
    pub failed: u32,
    /// Pages whose recording stopped on their work budget, deadline or
    /// cancellation; neither the device nor the CPU completed them.
    pub stopped: u32,
}

impl DevicePageCounts {
    /// Pages rendered on the CPU: every page counted but those the device
    /// rendered and those stopped.
    #[must_use]
    pub fn cpu(&self) -> u32 {
        self.ineligible
            .iter()
            .fold(0u32, |n, &(_, k)| n.saturating_add(k))
            .saturating_add(self.opening)
            .saturating_add(self.no_device)
            .saturating_add(self.retired)
            .saturating_add(self.failed)
    }

    /// Count one page under `outcome`.
    fn add(&mut self, outcome: Outcome) {
        let slot = match outcome {
            Outcome::Rendered => &mut self.rendered,
            Outcome::Ineligible(reason) => {
                let i = self
                    .ineligible
                    .iter()
                    .position(|&(r, _)| r == reason)
                    .unwrap_or_else(|| {
                        self.ineligible.push((reason, 0));
                        self.ineligible.len() - 1
                    });
                &mut self.ineligible[i].1
            }
            Outcome::Missed(Miss::Opening) => &mut self.opening,
            Outcome::Missed(Miss::NoDevice) => &mut self.no_device,
            Outcome::Missed(Miss::Retired) => &mut self.retired,
            Outcome::Failed => &mut self.failed,
            Outcome::Stopped => &mut self.stopped,
        };
        *slot = slot.saturating_add(1);
    }
}

impl std::fmt::Display for DevicePageCounts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} on the device, {} on the CPU",
            self.rendered,
            self.cpu()
        )?;
        let misses = [
            ("device opening", self.opening),
            ("no device", self.no_device),
            ("device retired", self.retired),
            ("device failed", self.failed),
        ];
        let mut parts = self
            .ineligible
            .iter()
            .copied()
            .chain(misses)
            .filter(|&(_, n)| n > 0)
            .peekable();
        if parts.peek().is_some() {
            f.write_str(" (")?;
            for (i, (what, n)) in parts.enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{what}: {n}")?;
            }
            f.write_str(")")?;
        }
        if self.stopped > 0 {
            write!(f, "; {} stopped", self.stopped)?;
        }
        Ok(())
    }
}

/// Why [`DevicePages::claim`] did not give the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Miss {
    /// The device did not open within [`OPEN_WAIT`].
    Opening,
    /// The device did not open.
    NoDevice,
    /// A device error retired the device.
    Retired,
}

impl Miss {
    /// The miss as the page's debug log names it.
    const fn describe(self) -> &'static str {
        match self {
            Self::Opening => "the device is still being opened",
            Self::NoDevice => "the device did not open",
            Self::Retired => "the device is retired",
        }
    }
}

/// Where one page went, as [`DevicePageCounts`] counts it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    Rendered,
    Ineligible(&'static str),
    Missed(Miss),
    Failed,
    Stopped,
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

/// The device, given to one page.
pub(crate) struct Claim<'a> {
    #[cfg(test)]
    pages: &'a DevicePages,
    /// The device.
    pub(crate) backend: &'a VulkanBackend,
}

/// The GPU-page device of `selector`, or `None`, logged at `error`, when
/// it does not open.
pub(crate) fn open_device(selector: Option<&str>) -> Option<Arc<VulkanBackend>> {
    let started = std::time::Instant::now();
    crate::vk_init::open_backend(selector)
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
        .ok()
}

impl DevicePages {
    /// The GPU pages of a session opened with `config` that holds
    /// `backend`: `None` unless the config asks for them, on `backend`
    /// when the session opened one, and otherwise on a device opened when
    /// a page first asks for it.
    pub(crate) fn for_session(
        config: &crate::SessionConfig,
        backend: Option<&Arc<VulkanBackend>>,
    ) -> Option<Self> {
        config.gpu_pages.then(|| {
            backend.map_or_else(
                || {
                    Self::new(
                        OnceLock::new(),
                        Opener::Deferred(config.vulkan_device.clone()),
                    )
                },
                |backend| Self::new(OnceLock::from(Some(Arc::clone(backend))), Opener::Settled),
            )
        })
    }

    fn new(device: OnceLock<Option<Arc<VulkanBackend>>>, opener: Opener) -> Self {
        Self {
            device: Arc::new(Opened::new(device)),
            opener: Mutex::new(opener),
            retired: AtomicBool::new(false),
            open_wait: OPEN_WAIT,
            open_timed_out: AtomicBool::new(false),
            wide_declined: AtomicBool::new(false),
            counts: Mutex::new(DevicePageCounts::default()),
            #[cfg(test)]
            faults: TestFaults::default(),
        }
    }

    /// Start opening the device on a new thread unless it has been
    /// started.  A device that does not open, or a thread that does not
    /// start, is logged at `error` and leaves every page on the CPU.
    fn start_opening(&self) {
        let mut opener = self.opener.lock().unwrap_or_else(PoisonError::into_inner);
        let Opener::Deferred(selector) = &mut *opener else {
            return;
        };
        let selector = selector.take();
        let slot = Arc::clone(&self.device);
        let spawned = std::thread::Builder::new()
            .name("rrocket-gpu-pages".to_owned())
            .spawn(move || {
                slot.settle(open_device(selector.as_deref()));
            });
        *opener = match spawned {
            Ok(thread) => Opener::Running(thread),
            Err(e) => {
                log::error!(
                    "rasterrocket: GPU page rendering was requested, but the thread that opens \
                     the Vulkan device did not start ({e}); every page renders on the CPU"
                );
                self.device.settle(None);
                Opener::Settled
            }
        };
    }

    /// The device, if it opens and is in service, or why not.  A call
    /// before the device is open starts opening it and waits for it, at
    /// most [`OPEN_WAIT`]; the first call that stops waiting is logged at
    /// `warn`.
    pub(crate) fn claim(&self) -> Result<Claim<'_>, Miss> {
        if self.retired.load(Ordering::Acquire) {
            return Err(Miss::Retired);
        }
        let device = if let Some(device) = self.device.get() {
            device
        } else {
            self.start_opening();
            self.device.wait(self.open_wait).ok_or_else(|| {
                if !self.open_timed_out.swap(true, Ordering::Relaxed) {
                    log::warn!(
                        "rasterrocket: the GPU page device has not opened after {:.0} s; pages \
                         render on the CPU until it does",
                        self.open_wait.as_secs_f64()
                    );
                }
                Miss::Opening
            })?
        };
        let backend = device.as_deref().ok_or(Miss::NoDevice)?;
        Ok(Claim {
            #[cfg(test)]
            pages: self,
            backend,
        })
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
        self.counts
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Account for the device failing on page `page_num` with `error`: a
    /// page beyond the device's limits is logged at `warn`; any other error
    /// retires the device, logged at `warn` once.
    fn fail(&self, page_num: u32, error: &BackendError) {
        if matches!(error, BackendError::DeviceLimit { .. }) {
            log::warn!(
                "page {page_num}: too large for the GPU page device ({error}); the page \
                 renders on the CPU"
            );
        } else if !self.retired.swap(true, Ordering::AcqRel) {
            log::warn!(
                "page {page_num}: the GPU page raster failed ({error}); the device is retired, \
                 and this and every later page render on the CPU"
            );
        }
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
        let counts = self
            .counts
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner);
        if *counts != DevicePageCounts::default() {
            log::info!("rasterrocket: GPU pages: {counts}");
        }
        let opener = std::mem::replace(
            self.opener
                .get_mut()
                .unwrap_or_else(PoisonError::into_inner),
            Opener::Settled,
        );
        if let Opener::Running(opener) = opener
            && opener.join().is_err()
        {
            log::warn!("rasterrocket: the thread opening the GPU page device panicked");
        }
    }
}

/// Render `page`, the recording of `renderer`'s page, on the claimed device
/// into the renderer's canvas.
fn device_render(
    claim: &Claim<'_>,
    renderer: &mut PageRenderer<'_>,
    page: &gpu::page::PageCommands,
) -> Result<(), BackendError> {
    #[cfg(test)]
    if let Some(e) = TestFaults::take(&claim.pages.faults.fail_next) {
        return Err(e);
    }
    renderer.paint_device_page(claim.backend, page)
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

/// Render `job`'s page, on the session's GPU page device when it qualifies
/// and the device is open and in service, and on the CPU otherwise; return
/// the renderer that executed it and how the device was used.
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
            match device_render(&claim, &mut renderer, &page) {
                Ok(()) => {
                    let commands = u32::try_from(page.len()).unwrap_or(u32::MAX);
                    log::debug!(
                        "page {}: rendered on the GPU from {commands} commands",
                        job.page_num
                    );
                    pages.count(Outcome::Rendered);
                    return Ok((renderer, DevicePage::Rendered { commands }));
                }
                Err(e) => {
                    pages.fail(job.page_num, &e);
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
    #[cfg(test)]
    let _ = pages.faults.cpu_reruns.fetch_add(1, Ordering::Relaxed);
    let fallback = renderer.take_device_fallback();
    let mut cpu = job.renderer(renderer.into_buffers())?;
    cpu.resume_after_device(fallback);
    Ok((job.executed(cpu, buffers)?, outcome))
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;
    use std::sync::atomic::Ordering;
    use std::time::{Duration, Instant};

    use gpu::backend::BackendError;

    use super::{DevicePageCounts, DevicePages, Miss, Opened, Opener, Outcome};
    use crate::test_pdf::{assemble, stream};
    use crate::{
        BackendPolicy, CancelToken, DevicePage, RasterError, RasterSession, RenderParams,
        SessionConfig, open_session_from_bytes, render_page_rgb_with_diagnostics,
    };

    /// Fills and strokes the device renders.
    const DEVICE: &str = "0 0 1 rg 10.3 10.7 60.2 50.1 re f 1 0 0 RG 3 w 5 5 m 90 70 l S";

    /// A translucent hairline, which the recording refuses.
    const REFUSED: &str = "/T gs 0 0 1 RG 0 w 10 10 m 90 90 l S";

    /// A PDF of 100 × 100 pt pages, page `i` painting `contents[i]`.
    fn pdf(contents: &[&str]) -> Vec<u8> {
        let n = contents.len();
        let kids = (0..n)
            .map(|i| format!("{} 0 R", 3 + 2 * i))
            .collect::<Vec<_>>()
            .join(" ");
        let mut objects = vec![
            "<</Type /Catalog /Pages 2 0 R>>".to_owned(),
            format!("<</Type /Pages /Kids [{kids}] /Count {n}>>"),
        ];
        for (i, content) in contents.iter().enumerate() {
            objects.push(format!(
                "<</Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
                 /Resources <</ExtGState <</T <</CA 0.5>> >> >> /Contents {} 0 R>>",
                4 + 2 * i
            ));
            objects.push(stream(content));
        }
        assemble(&objects.iter().map(String::as_str).collect::<Vec<_>>())
    }

    /// A session over `pdf` rendering GPU pages on a device opened with it,
    /// or `None` (the test is skipped) when no Vulkan device opens.
    fn gpu_session(pdf: Vec<u8>) -> Option<RasterSession> {
        let mut config = SessionConfig::with_policy(BackendPolicy::ForceVulkan);
        config.vulkan_device = SessionConfig::vulkan_device_from_env();
        config.gpu_pages = true;
        open_session_from_bytes(pdf, &config)
            .inspect_err(|e| eprintln!("skipping: no Vulkan device: {e}"))
            .ok()
    }

    fn device_pages(session: &RasterSession) -> &DevicePages {
        session.device_pages.as_ref().expect("a GPU-page session")
    }

    /// Page `page` of `session` at 2× and how the device was used.
    fn render(
        session: &RasterSession,
        page: u32,
        params: &RenderParams,
    ) -> Result<(Vec<u8>, DevicePage), RasterError> {
        render_page_rgb_with_diagnostics(session, page, params)
            .map(|(bitmap, diag)| (bitmap.data().to_vec(), diag.device_page))
    }

    /// Page `page` of `pdf` rendered on the CPU.
    fn cpu(pdf: Vec<u8>, page: u32) -> Vec<u8> {
        let session =
            open_session_from_bytes(pdf, &SessionConfig::with_policy(BackendPolicy::CpuOnly))
                .expect("CPU session");
        render(&session, page, &RenderParams::new(2.0))
            .expect("CPU render")
            .0
    }

    /// A wait for an opening returns its outcome once it is settled, by
    /// another thread or before the wait, and nothing once the wait is
    /// over.
    #[test]
    fn a_wait_for_the_opening_ends_with_its_outcome_or_its_time() {
        let opened = Opened::new(OnceLock::new());
        let started = Instant::now();
        assert!(opened.wait(Duration::from_millis(30)).is_none());
        assert!(started.elapsed() >= Duration::from_millis(30));
        std::thread::scope(|scope| {
            let waiter = scope.spawn(|| opened.wait(Duration::from_secs(60)).cloned());
            std::thread::sleep(Duration::from_millis(20));
            opened.settle(None);
            assert_eq!(
                waiter.join().expect("waiter").map(|d| d.is_some()),
                Some(false)
            );
        });
        assert!(matches!(opened.wait(Duration::ZERO), Some(None)));
    }

    /// A page that waited its time for a device still opening renders on
    /// the CPU, counted under `opening`, and warns once.
    #[test]
    fn a_page_renders_on_the_cpu_once_its_wait_for_the_opening_is_over() {
        let doc = pdf(&[DEVICE, DEVICE]);
        let session = auto_session(doc.clone(), None);
        let mut pages = DevicePages::new(OnceLock::new(), Opener::Settled);
        pages.open_wait = Duration::from_millis(20);
        let mut session = session;
        session.device_pages = Some(pages);
        for page in 1..=2 {
            let (bitmap, outcome) =
                render(&session, page, &RenderParams::new(2.0)).expect("render");
            assert_eq!(outcome, DevicePage::NotOffered);
            assert_eq!(bitmap, cpu(doc.clone(), page));
        }
        let pages = device_pages(&session);
        assert!(pages.open_timed_out.load(Ordering::Relaxed));
        assert_eq!(pages.counts().opening, 2);
    }

    /// A device error renders the page on the CPU, identically, and
    /// retires the device: later pages are not offered to it.
    #[test]
    fn a_device_failure_renders_on_the_cpu_and_retires_the_device() {
        let doc = pdf(&[DEVICE, DEVICE]);
        let Some(session) = gpu_session(doc.clone()) else {
            return;
        };
        let pages = device_pages(&session);
        let params = RenderParams::new(2.0);
        *pages.faults.fail_next.lock().expect("lock") = Some(BackendError::msg("device lost"));
        let (failed, outcome) = render(&session, 1, &params).expect("render");
        assert_eq!(outcome, DevicePage::Failed);
        assert_eq!(failed, cpu(doc.clone(), 1));
        assert_eq!(pages.faults.cpu_reruns.load(Ordering::Relaxed), 1);
        let (later, outcome) = render(&session, 2, &params).expect("render");
        assert_eq!(outcome, DevicePage::NotOffered);
        assert_eq!(later, cpu(doc, 2));
        let counts = session.device_page_counts().expect("a GPU-page session");
        assert_eq!((counts.failed, counts.retired, counts.cpu()), (1, 1, 2));
        assert_eq!(counts.rendered, 0);
    }

    /// A page beyond the device's limits renders on the CPU and leaves the
    /// device in service.
    #[test]
    fn a_page_beyond_the_device_limits_keeps_the_device() {
        let doc = pdf(&[DEVICE, DEVICE]);
        let Some(session) = gpu_session(doc.clone()) else {
            return;
        };
        let pages = device_pages(&session);
        let params = RenderParams::new(2.0);
        *pages.faults.fail_next.lock().expect("lock") = Some(BackendError::DeviceLimit {
            what: "render_page pixels",
            need: 1 << 40,
            limit: "maxStorageBufferRange",
            max: 1 << 27,
        });
        let (page, outcome) = render(&session, 1, &params).expect("render");
        assert_eq!(outcome, DevicePage::Failed);
        assert_eq!(page, cpu(doc.clone(), 1));
        let (page, outcome) = render(&session, 2, &params).expect("render");
        assert!(
            matches!(outcome, DevicePage::Rendered { .. }),
            "{outcome:?}"
        );
        assert_eq!(page, cpu(doc, 2));
    }

    /// A refused recording renders on the CPU once; a cancelled recording
    /// is reported cancelled without rendering the page again.
    #[test]
    fn a_cancelled_recording_is_not_rendered_again() {
        let doc = pdf(&[REFUSED, DEVICE]);
        let Some(session) = gpu_session(doc.clone()) else {
            return;
        };
        let pages = device_pages(&session);
        let (page, outcome) = render(&session, 1, &RenderParams::new(2.0)).expect("render");
        assert_eq!(outcome, DevicePage::Ineligible("translucent hairline"));
        assert_eq!(page, cpu(doc, 1));
        assert_eq!(pages.faults.cpu_reruns.load(Ordering::Relaxed), 1);

        let token = CancelToken::new();
        *pages.faults.cancel_recording.lock().expect("lock") = Some(token.clone());
        let params = RenderParams::new(2.0).with_cancel(token);
        assert!(matches!(
            render(&session, 2, &params),
            Err(RasterError::Cancelled)
        ));
        assert_eq!(pages.faults.cpu_reruns.load(Ordering::Relaxed), 1);
        let counts = pages.counts();
        assert_eq!(counts.ineligible, [("translucent hairline", 1)]);
        assert_eq!((counts.stopped, counts.cpu(), counts.rendered), (1, 1, 0));
    }

    /// Pages rendered at once by several threads all render on the device,
    /// with the CPU's bytes.
    #[test]
    fn concurrent_pages_all_render_on_the_device() {
        const PAGES: u32 = 12;
        let doc = pdf(&[DEVICE; PAGES as usize]);
        let Some(session) = gpu_session(doc.clone()) else {
            return;
        };
        let want = cpu(doc, 1);
        let outcomes: Vec<DevicePage> = std::thread::scope(|scope| {
            let workers: Vec<_> = (1..=PAGES)
                .map(|page| {
                    let session = &session;
                    scope.spawn(move || render(session, page, &RenderParams::new(2.0)))
                })
                .collect();
            workers
                .into_iter()
                .map(|w| {
                    let (page, outcome) = w.join().expect("worker").expect("render");
                    assert_eq!(page, want, "{outcome:?}");
                    outcome
                })
                .collect()
        });
        assert!(
            outcomes
                .iter()
                .all(|o| matches!(o, DevicePage::Rendered { .. })),
            "{outcomes:?}"
        );
        let counts = device_pages(&session).counts();
        assert_eq!((counts.rendered, counts.cpu()), (PAGES, 0), "{counts}");
    }

    /// A page of the wide working space renders on the CPU, as a CPU
    /// session renders it, and the first such page warns.
    #[test]
    fn a_wide_page_renders_on_the_cpu_and_warns_once() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/icc/printer-rgb-v4.icc"
        );
        let profile = std::sync::Arc::new(
            pdf_interp::OutputProfile::from_bytes(std::fs::read(path).expect("read"))
                .expect("profile"),
        );
        let doc = pdf(&[DEVICE, DEVICE]);
        let mut config = SessionConfig::with_policy(BackendPolicy::ForceVulkan);
        config.vulkan_device = SessionConfig::vulkan_device_from_env();
        config.gpu_pages = true;
        config.output_profile = Some(std::sync::Arc::clone(&profile));
        let Ok(session) = open_session_from_bytes(doc.clone(), &config)
            .inspect_err(|e| eprintln!("skipping: no Vulkan device: {e}"))
        else {
            return;
        };
        let mut cpu_config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
        cpu_config.output_profile = Some(profile);
        let cpu_session = open_session_from_bytes(doc, &cpu_config).expect("CPU session");
        let params = RenderParams::new(2.0);
        let (page, outcome) = render(&session, 1, &params).expect("render");
        assert_eq!(outcome, DevicePage::Ineligible("the wide working space"));
        assert_eq!(
            page,
            render(&cpu_session, 1, &params).expect("CPU render").0
        );
        assert_eq!(
            device_pages(&session).wide_declined(2),
            None,
            "warned on page 1"
        );
    }

    /// An `Auto` GPU-page session over `pdf` opening the device `selector`
    /// names.
    fn auto_session(pdf: Vec<u8>, selector: Option<String>) -> RasterSession {
        let mut config = SessionConfig::with_policy(BackendPolicy::Auto);
        config.vulkan_device = selector;
        config.gpu_pages = true;
        open_session_from_bytes(pdf, &config).expect("an Auto session opens")
    }

    /// Under `Auto`, rendering pages none of which the device can take
    /// opens no device, and the session's drop joins nothing.
    #[test]
    fn auto_opens_no_device_for_pages_it_cannot_take() {
        const SHADING: &str = "0 g 0 0 10 10 re f /Sh sh";
        let doc = pdf(&[SHADING, SHADING]);
        let session = auto_session(doc.clone(), SessionConfig::vulkan_device_from_env());
        for page in 1..=2 {
            let (bitmap, outcome) =
                render(&session, page, &RenderParams::new(2.0)).expect("render");
            assert_eq!(outcome, DevicePage::Ineligible("shading"));
            assert_eq!(bitmap, cpu(doc.clone(), page));
        }
        let pages = device_pages(&session);
        assert!(pages.device.get().is_none(), "a device was opened");
        assert!(
            matches!(*pages.opener.lock().expect("lock"), Opener::Deferred(_)),
            "the opening started"
        );
        assert_eq!(pages.counts().ineligible, [("shading", 2)]);
    }

    /// Under `Auto`, the first page the device could take starts opening
    /// it, waits for it and renders on it, as does a later page (on the
    /// CPU when no device opens), identically.
    #[test]
    fn auto_opens_the_device_for_the_first_page_it_could_take() {
        let doc = pdf(&[DEVICE, DEVICE]);
        let session = auto_session(doc.clone(), SessionConfig::vulkan_device_from_env());
        let pages = device_pages(&session);
        assert!(matches!(
            *pages.opener.lock().expect("lock"),
            Opener::Deferred(_)
        ));
        let params = RenderParams::new(2.0);
        let (first, first_outcome) = render(&session, 1, &params).expect("render");
        assert!(
            matches!(*pages.opener.lock().expect("lock"), Opener::Running(_)),
            "the first page did not start opening the device"
        );
        let device_open = matches!(pages.device.get(), Some(Some(_)));
        let (second, second_outcome) = render(&session, 2, &params).expect("render");
        let counts = pages.counts();
        for outcome in [first_outcome, second_outcome] {
            if device_open {
                assert!(
                    matches!(outcome, DevicePage::Rendered { .. }),
                    "{outcome:?}"
                );
            } else {
                assert_eq!(outcome, DevicePage::NotOffered);
            }
        }
        if device_open {
            assert_eq!(counts.rendered, 2, "{counts}");
        } else {
            eprintln!("no Vulkan device opened");
            assert_eq!(counts.no_device, 2, "{counts}");
        }
        assert_eq!(first, cpu(doc.clone(), 1));
        assert_eq!(second, cpu(doc, 2));
    }

    /// Each outcome is counted under its own field, ineligible pages by
    /// reason in order of first occurrence, and the summary names every
    /// non-zero count.
    #[test]
    fn counts_tally_and_summarise_every_outcome() {
        let mut counts = DevicePageCounts::default();
        assert_eq!(counts.to_string(), "0 on the device, 0 on the CPU");
        for outcome in [
            Outcome::Rendered,
            Outcome::Rendered,
            Outcome::Ineligible("shading"),
            Outcome::Ineligible("translucent hairline"),
            Outcome::Ineligible("shading"),
            Outcome::Missed(Miss::Opening),
            Outcome::Missed(Miss::NoDevice),
            Outcome::Missed(Miss::Retired),
            Outcome::Failed,
            Outcome::Stopped,
        ] {
            counts.add(outcome);
        }
        assert_eq!(
            counts.ineligible,
            [("shading", 2), ("translucent hairline", 1)]
        );
        assert_eq!((counts.rendered, counts.cpu(), counts.stopped), (2, 7, 1));
        assert_eq!(
            counts.to_string(),
            "2 on the device, 7 on the CPU (shading: 2, translucent hairline: 1, \
             device opening: 1, no device: 1, device retired: 1, \
             device failed: 1); 1 stopped"
        );
    }

    /// A session without GPU pages has no counts.
    #[test]
    fn a_cpu_session_has_no_device_page_counts() {
        let session = open_session_from_bytes(
            pdf(&[DEVICE]),
            &SessionConfig::with_policy(BackendPolicy::CpuOnly),
        )
        .expect("CPU session");
        assert_eq!(session.device_page_counts(), None);
    }

    /// GPU pages are refused at open without vector antialiasing, which
    /// the device raster needs.
    #[test]
    fn gpu_pages_need_vector_antialiasing() {
        let mut config = SessionConfig::with_policy(BackendPolicy::ForceVulkan);
        config.gpu_pages = true;
        config.vector_antialias = false;
        match open_session_from_bytes(pdf(&[DEVICE]), &config) {
            Err(RasterError::InvalidOptions(msg)) => assert!(msg.contains("antialias"), "{msg}"),
            Err(other) => panic!("expected InvalidOptions, got {other:?}"),
            Ok(_) => panic!("a session without vector antialiasing accepted GPU pages"),
        }
    }
}
