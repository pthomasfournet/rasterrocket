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
//! A device error retires the device for the session: every later page
//! renders on the CPU.  A page too large for the device's limits renders
//! on the CPU and leaves the device in service.
//!
//! [`SessionConfig::gpu_pages`]: crate::SessionConfig::gpu_pages

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::thread::JoinHandle;

use gpu::backend::BackendError;
use gpu::backend::vulkan::VulkanBackend;
use pdf_interp::renderer::{CanvasBuffers, DevicePage, DeviceRecording, PageRenderer};

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
    /// Set once the device has failed; it is offered no page afterwards.
    retired: AtomicBool,
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
            retired: AtomicBool::new(false),
            #[cfg(test)]
            faults: TestFaults::default(),
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
            retired: AtomicBool::new(false),
            #[cfg(test)]
            faults: TestFaults::default(),
        })
    }

    /// The device, if it is open, in service and no page holds it.
    pub(crate) fn try_claim(&self) -> Option<Claim<'_>> {
        if self.retired.load(Ordering::Acquire) {
            return None;
        }
        let backend = self.device.get()?.as_deref()?;
        self.busy
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
            .then_some(Claim {
                pages: self,
                backend,
            })
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

/// Why `job`'s page cannot be recorded, known before interpretation.
fn ruled_out(job: &PageJob<'_>) -> Option<&'static str> {
    pdf_interp::renderer::device_page_ineligible(
        job.ink,
        job.group.is_some(),
        job.session.vector_antialias,
        job.ops,
    )
}

/// Render `job`'s page, on the session's GPU page device when it qualifies
/// and the device is free, and on the CPU otherwise; return the renderer
/// that executed it and how the device was used.
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
    buffers: CanvasBuffers,
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
    #[cfg(test)]
    if let Some(token) = TestFaults::take(&pages.faults.cancel_recording) {
        token.cancel();
    }
    job.execute(&mut renderer)?;
    let recording = renderer.take_device_recording();
    if renderer.budget_status().is_some() {
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
                    return Ok((renderer, DevicePage::Rendered { commands }));
                }
                Err(e) => {
                    pages.fail(job.page_num, &e);
                    DevicePage::Failed
                }
            }
        }
        Some(DeviceRecording::Refused(reason)) => {
            log::debug!("page {}: rendered on the CPU: {reason}", job.page_num);
            DevicePage::Ineligible(reason)
        }
        None => unreachable!("a renderer recording for the device yields its recording"),
    };
    drop(claim);
    #[cfg(test)]
    let _ = pages.faults.cpu_reruns.fetch_add(1, Ordering::Relaxed);
    let fallback = renderer.take_device_fallback();
    let buffers = renderer.try_finish_keeping(false).map_or_else(
        |_| CanvasBuffers::default(),
        |(bitmap, _, mut spare)| {
            spare.rgb.reclaim(bitmap);
            spare
        },
    );
    let mut cpu = job.renderer(buffers)?;
    cpu.resume_after_device(fallback);
    job.execute(&mut cpu)?;
    Ok((cpu, outcome))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use gpu::backend::BackendError;

    use super::DevicePages;
    use crate::test_pdf::{assemble, stream};
    use crate::{
        BackendPolicy, CancelToken, DevicePage, RasterError, RasterSession, RenderParams,
        SessionConfig, open_session_from_bytes, render_page_rgb_with_diagnostics,
    };

    /// Fills and strokes the device renders.
    const DEVICE: &str = "0 0 1 rg 10.3 10.7 60.2 50.1 re f 1 0 0 RG 3 w 5 5 m 90 70 l S";

    /// A paint under a path clip, which the recording refuses.
    const CLIPPED: &str = "10 10 m 90 10 l 50 90 l h W n 0 0 1 rg 0 0 100 100 re f";

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
                "<</Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Contents {} 0 R>>",
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
        let doc = pdf(&[CLIPPED, DEVICE]);
        let Some(session) = gpu_session(doc.clone()) else {
            return;
        };
        let pages = device_pages(&session);
        let (page, outcome) = render(&session, 1, &RenderParams::new(2.0)).expect("render");
        assert_eq!(outcome, DevicePage::Ineligible("path clip"));
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
    }

    /// Pages rendered at once by several threads each take the device or
    /// the CPU, with the CPU's bytes either way.
    #[test]
    fn concurrent_pages_share_the_device() {
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
                .all(|o| matches!(o, DevicePage::Rendered { .. } | DevicePage::NotOffered)),
            "{outcomes:?}"
        );
        assert!(
            outcomes
                .iter()
                .any(|o| matches!(o, DevicePage::Rendered { .. })),
            "{outcomes:?}"
        );
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
