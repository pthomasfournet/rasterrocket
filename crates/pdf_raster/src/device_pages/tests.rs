use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use gpu::backend::BackendError;

use super::DevicePages;
use super::devices::Device;
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
    let session = open_session_from_bytes(pdf, &SessionConfig::with_policy(BackendPolicy::CpuOnly))
        .expect("CPU session");
    render(&session, page, &RenderParams::new(2.0))
        .expect("CPU render")
        .0
}

/// A page that waited its time for a device still opening renders on
/// the CPU, counted under `opening`, and warns once.
#[test]
fn a_page_renders_on_the_cpu_once_its_wait_for_the_opening_is_over() {
    let doc = pdf(&[DEVICE, DEVICE]);
    let mut session = auto_session(doc.clone(), None);
    let mut pages = DevicePages::new(vec![Device::never_opening()]);
    pages.open_wait = Duration::from_millis(30);
    session.device_pages = Some(pages);
    for page in 1..=2 {
        let started = Instant::now();
        let (bitmap, outcome) = render(&session, page, &RenderParams::new(2.0)).expect("render");
        assert!(started.elapsed() >= Duration::from_millis(30));
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
        pdf_interp::OutputProfile::from_bytes(std::fs::read(path).expect("read")).expect("profile"),
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
        let (bitmap, outcome) = render(&session, page, &RenderParams::new(2.0)).expect("render");
        assert_eq!(outcome, DevicePage::Ineligible("shading"));
        assert_eq!(bitmap, cpu(doc.clone(), page));
    }
    let pages = device_pages(&session);
    assert!(!pages.devices[0].settled(), "a device was opened");
    assert!(pages.devices[0].deferred_still(), "the opening started");
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
    assert!(pages.devices[0].deferred_still());
    let params = RenderParams::new(2.0);
    let (first, first_outcome) = render(&session, 1, &params).expect("render");
    assert!(
        pages.devices[0].opening_started(),
        "the first page did not start opening the device"
    );
    let device_open = pages.devices[0].is_open();
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

/// A session over `pdf` rendering GPU pages on every usable GPU, opened
/// with it, or `None` (the test is skipped) when fewer than two open.
fn two_device_session(pdf: Vec<u8>) -> Option<RasterSession> {
    let mut config = SessionConfig::with_policy(BackendPolicy::ForceVulkan);
    config.gpu_pages = true;
    config.gpu_devices = vec!["all".to_owned()];
    let session = open_session_from_bytes(pdf, &config)
        .inspect_err(|e| eprintln!("skipping: no Vulkan device: {e}"))
        .ok()?;
    if device_pages(&session).devices.len() < 2 {
        eprintln!("skipping: fewer than two usable GPUs");
        return None;
    }
    Some(session)
}

/// A page takes the open device the fewest pages hold, the first listed
/// among equals.
#[test]
fn a_page_takes_the_device_the_fewest_pages_hold() {
    let Some(session) = two_device_session(pdf(&[DEVICE])) else {
        return;
    };
    let pages = device_pages(&session);
    let first = pages.claim().expect("a device");
    let second = pages.claim().expect("a device");
    let third = pages.claim().expect("a device");
    assert!(std::ptr::eq(first.device, &raw const pages.devices[0]));
    assert!(std::ptr::eq(second.device, &raw const pages.devices[1]));
    assert!(std::ptr::eq(third.device, &raw const pages.devices[0]));
    assert_eq!(
        (pages.devices[0].in_flight(), pages.devices[1].in_flight()),
        (2, 1)
    );
    drop(first);
    let fourth = pages.claim().expect("a device");
    assert!(std::ptr::eq(fourth.device, &raw const pages.devices[0]));
    drop((second, third, fourth));
    assert_eq!(
        (pages.devices[0].in_flight(), pages.devices[1].in_flight()),
        (0, 0)
    );
}

/// Pages rendered at once on two devices all render on a device, with the
/// CPU's bytes, and the counts name the pages of each device.
#[test]
fn concurrent_pages_spread_over_two_devices() {
    const PAGES: u32 = 24;
    let doc = pdf(&[DEVICE; PAGES as usize]);
    let Some(session) = two_device_session(doc.clone()) else {
        return;
    };
    let want = cpu(doc, 1);
    std::thread::scope(|scope| {
        let workers: Vec<_> = (1..=PAGES)
            .map(|page| {
                let session = &session;
                scope.spawn(move || render(session, page, &RenderParams::new(2.0)))
            })
            .collect();
        for w in workers {
            let (page, outcome) = w.join().expect("worker").expect("render");
            assert!(
                matches!(outcome, DevicePage::Rendered { .. }),
                "{outcome:?}"
            );
            assert_eq!(page, want);
        }
    });
    let counts = device_pages(&session).counts();
    assert_eq!((counts.rendered, counts.cpu()), (PAGES, 0), "{counts}");
    assert_eq!(counts.by_device.len(), device_pages(&session).devices.len());
    let sum: u32 = counts.by_device.iter().map(|&(_, n)| n).sum();
    assert_eq!(sum, PAGES, "{counts}");
}

/// A device error retires that device only: later pages render on the
/// other device, identically.
#[test]
fn a_failed_device_leaves_the_others_in_service() {
    let doc = pdf(&[DEVICE, DEVICE, DEVICE]);
    let Some(session) = two_device_session(doc.clone()) else {
        return;
    };
    let pages = device_pages(&session);
    let params = RenderParams::new(2.0);
    *pages.faults.fail_next.lock().expect("lock") = Some(BackendError::msg("device lost"));
    let (page, outcome) = render(&session, 1, &params).expect("render");
    assert_eq!(outcome, DevicePage::Failed);
    assert_eq!(page, cpu(doc.clone(), 1));
    for n in 2..=3 {
        let (page, outcome) = render(&session, n, &params).expect("render");
        assert!(
            matches!(outcome, DevicePage::Rendered { .. }),
            "{outcome:?}"
        );
        assert_eq!(page, cpu(doc.clone(), n));
    }
    let counts = pages.counts();
    assert_eq!((counts.failed, counts.rendered), (1, 2), "{counts}");
    assert_eq!(counts.by_device[0].1, 0, "{counts}");
    assert_eq!(counts.by_device[1].1, 2, "{counts}");
}

/// A device list naming one device twice is refused at open.
#[test]
fn a_device_list_naming_one_device_twice_is_refused() {
    if gpu::backend::vulkan::VulkanBackend::usable_gpus().is_err() {
        eprintln!("skipping: no usable Vulkan GPU");
        return;
    }
    for policy in [BackendPolicy::Auto, BackendPolicy::ForceVulkan] {
        let mut config = SessionConfig::with_policy(policy);
        config.gpu_pages = true;
        config.gpu_devices = vec!["0".to_owned(), "0".to_owned()];
        match open_session_from_bytes(pdf(&[DEVICE]), &config) {
            Err(RasterError::InvalidOptions(msg)) => {
                assert!(msg.contains("both name device 0"), "{msg}");
            }
            Err(other) => panic!("expected InvalidOptions, got {other:?}"),
            Ok(_) => panic!("a device list naming one device twice was accepted"),
        }
    }
}

/// A device list is refused without GPU pages, with `all` beside other
/// entries, with an empty entry, and beside a Vulkan device under `Auto`.
#[test]
fn malformed_device_lists_are_refused_before_any_device_is_asked() {
    let refusal = |edit: &dyn Fn(&mut SessionConfig)| {
        let mut config = SessionConfig::with_policy(BackendPolicy::Auto);
        config.gpu_pages = true;
        config.gpu_devices = vec!["0".to_owned()];
        edit(&mut config);
        match config.check_gpu_pages() {
            Err(RasterError::InvalidOptions(msg)) => msg,
            other => panic!("expected InvalidOptions, got {other:?}"),
        }
    };
    assert!(refusal(&|c| c.gpu_pages = false).contains("--gpu-pages"));
    assert!(refusal(&|c| c.gpu_devices.push("ALL".to_owned())).contains("`all` beside"));
    assert!(refusal(&|c| c.gpu_devices.push(" ".to_owned())).contains("empty entry"));
    assert!(refusal(&|c| c.vulkan_device = Some("0".to_owned())).contains("--vulkan-device"));
    let mut config = SessionConfig::with_policy(BackendPolicy::ForceVulkan);
    config.gpu_pages = true;
    config.gpu_devices = vec!["0".to_owned()];
    config.vulkan_device = Some("0".to_owned());
    assert!(
        config.check_gpu_pages().is_ok(),
        "ForceVulkan uses its device too"
    );
}
