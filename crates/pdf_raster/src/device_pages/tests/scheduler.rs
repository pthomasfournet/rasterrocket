//! The scheduler's tests, generic over the API whose devices render the
//! GPU pages; [`super`] runs each on every API compiled in.

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use gpu::backend::BackendError;

use super::super::devices::{Device, Load};
use super::super::{DevicePages, choose};
use super::capture::warnings_of;
use super::{
    Api, DEVICE, REFUSED, cpu, device_pages, gpu_config, gpu_session, pdf, render, session_with,
    two_device_session,
};
use crate::{BackendPolicy, CancelToken, DevicePage, RasterError, RenderParams, SessionConfig};

/// The first page waits for a device still opening until the session's
/// deadline and renders on the CPU, counted under `opening`, with one
/// warning; a later page finds the deadline passed and does not wait.
pub(super) fn pages_render_on_the_cpu_once_the_session_wait_for_the_opening_is_over<B: Api>() {
    let doc = pdf(&[DEVICE, DEVICE]);
    let mut pages = DevicePages::<B>::new(vec![Device::never_opening()]);
    pages.open_wait = Duration::from_millis(300);
    let session = session_with(doc.clone(), pages);
    for page in 1..=2 {
        let started = Instant::now();
        let (bitmap, outcome) = render(&session, page, &RenderParams::new(2.0)).expect("render");
        let waited = started.elapsed();
        if page == 1 {
            assert!(
                waited >= Duration::from_millis(300),
                "page 1 waited {waited:?}"
            );
        } else {
            assert!(
                waited < Duration::from_millis(300),
                "page 2 waited {waited:?}"
            );
        }
        assert_eq!(outcome, DevicePage::NotOffered);
        assert_eq!(bitmap, cpu(doc.clone(), page));
    }
    let pages = device_pages::<B>(&session);
    assert!(pages.open_timed_out.load(Ordering::Relaxed));
    assert_eq!(pages.counts().opening, 2);
}

/// A page cancelled while it waits for a device to open stops waiting,
/// returns the cancellation and is counted as stopped.
pub(super) fn a_cancelled_page_stops_waiting_for_the_opening<B: Api>() {
    let session = session_with(
        pdf(&[DEVICE]),
        DevicePages::<B>::new(vec![Device::never_opening()]),
    );
    let token = CancelToken::new();
    let cancel = token.clone();
    let started = Instant::now();
    let canceller = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(50));
        cancel.cancel();
    });
    let result = render(&session, 1, &RenderParams::new(2.0).with_cancel(token));
    canceller.join().expect("the cancelling thread");
    assert!(
        matches!(result, Err(RasterError::Cancelled)),
        "expected Cancelled, got {:?}",
        result.map(|(_, outcome)| outcome)
    );
    assert!(started.elapsed() < Duration::from_secs(5));
    let counts = device_pages::<B>(&session).counts();
    assert_eq!((counts.stopped, counts.opening), (1, 0));
}

/// A device failure of `error` renders the page on the CPU, identically,
/// and retires the device: later pages are not offered to it.
fn assert_failure_retires_the_device<B: Api>(error: BackendError) {
    let doc = pdf(&[DEVICE, DEVICE]);
    let Some(session) = gpu_session::<B>(doc.clone()) else {
        return;
    };
    let pages = device_pages::<B>(&session);
    let params = RenderParams::new(2.0);
    *pages.faults.fail_next.lock().expect("lock") = Some(error);
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

/// A device error renders the page on the CPU, identically, and
/// retires the device.
pub(super) fn a_device_failure_renders_on_the_cpu_and_retires_the_device<B: Api>() {
    assert_failure_retires_the_device::<B>(BackendError::msg("device lost"));
}

/// A raster that reports a fault shows the device's arithmetic
/// unreliable: the page renders on the CPU and the device is retired.
pub(super) fn a_kernel_fault_retires_the_device<B: Api>() {
    assert_failure_retires_the_device::<B>(BackendError::KernelFault {
        kernel: "page_raster_paths",
        detail: "floor_div(a = -1, b = 3) did not converge within 16 correction steps; \
                 the device's float division is unreliable"
            .to_owned(),
    });
}

/// A page failed with `error` renders on the CPU and leaves the device in
/// service: the next page renders on it.
fn assert_failure_keeps_the_device<B: Api>(error: BackendError) {
    let doc = pdf(&[DEVICE, DEVICE]);
    let Some(session) = gpu_session::<B>(doc.clone()) else {
        return;
    };
    let pages = device_pages::<B>(&session);
    let params = RenderParams::new(2.0);
    *pages.faults.fail_next.lock().expect("lock") = Some(error);
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

/// A page beyond the device's limits renders on the CPU and leaves the
/// device in service.
pub(super) fn a_page_beyond_the_device_limits_keeps_the_device<B: Api>() {
    assert_failure_keeps_the_device::<B>(BackendError::DeviceLimit {
        what: "render_page pixels",
        need: 1 << 40,
        limit: "maxStorageBufferRange",
        max: 1 << 27,
        unit: "bytes",
    });
}

/// A page beyond the memory the device has left renders on the CPU and
/// leaves the device in service.
pub(super) fn a_page_beyond_the_memory_left_keeps_the_device<B: Api>() {
    assert_failure_keeps_the_device::<B>(BackendError::OutOfMemory {
        what: "render_page pixels",
        bytes: 1 << 30,
    });
}

/// A refused recording renders on the CPU once; a cancelled recording
/// is reported cancelled without rendering the page again.
pub(super) fn a_cancelled_recording_is_not_rendered_again<B: Api>() {
    let doc = pdf(&[REFUSED, DEVICE]);
    let Some(session) = gpu_session::<B>(doc.clone()) else {
        return;
    };
    let pages = device_pages::<B>(&session);
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

/// Pages `1..=pages` of `session` rendered at once, each by a thread of
/// its own: every one renders on a device, with `want`'s bytes.
fn render_concurrently(session: &crate::RasterSession, pages: u32, want: &[u8]) {
    std::thread::scope(|scope| {
        let workers: Vec<_> = (1..=pages)
            .map(|page| scope.spawn(move || render(session, page, &RenderParams::new(2.0))))
            .collect();
        for w in workers {
            let (page, outcome) = w.join().expect("worker").expect("render");
            assert!(
                matches!(outcome, DevicePage::Rendered { .. }),
                "{outcome:?}"
            );
            assert_eq!(page, want, "{outcome:?}");
        }
    });
}

/// Pages rendered at once by several threads all render on the device,
/// with the CPU's bytes.
pub(super) fn concurrent_pages_all_render_on_the_device<B: Api>() {
    const PAGES: u32 = 12;
    let doc = pdf(&[DEVICE; PAGES as usize]);
    let Some(session) = gpu_session::<B>(doc.clone()) else {
        return;
    };
    render_concurrently(&session, PAGES, &cpu(doc, 1));
    let counts = device_pages::<B>(&session).counts();
    assert_eq!((counts.rendered, counts.cpu()), (PAGES, 0), "{counts}");
}

/// A page of the wide working space renders on the CPU, as a CPU
/// session renders it, and the first such page warns.
pub(super) fn a_wide_page_renders_on_the_cpu_and_warns_once<B: Api>() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/icc/printer-rgb-v4.icc"
    );
    let profile = std::sync::Arc::new(
        pdf_interp::OutputProfile::from_bytes(std::fs::read(path).expect("read")).expect("profile"),
    );
    let doc = pdf(&[DEVICE, DEVICE]);
    let mut config = gpu_config::<B>();
    config.output_profile = Some(std::sync::Arc::clone(&profile));
    let Some(session) = super::or_skip::<B, _, _>(
        "no device opens",
        crate::open_session_from_bytes(doc.clone(), &config),
    ) else {
        return;
    };
    let mut cpu_config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    cpu_config.output_profile = Some(profile);
    let cpu_session = crate::open_session_from_bytes(doc, &cpu_config).expect("CPU session");
    let params = RenderParams::new(2.0);
    let (page, outcome) = render(&session, 1, &params).expect("render");
    assert_eq!(outcome, DevicePage::Ineligible("the wide working space"));
    assert_eq!(
        page,
        render(&cpu_session, 1, &params).expect("CPU render").0
    );
    assert_eq!(
        device_pages::<B>(&session).wide_declined(2),
        None,
        "warned on page 1"
    );
}

/// A page drawn in several bands renders on the CPU, as a CPU session
/// draws it, without asking for a device; it is counted and reported as
/// ineligible, and the first such page warns.
pub(super) fn a_banded_page_renders_on_the_cpu_and_is_counted<B: Api>() {
    let doc = pdf(&[DEVICE, DEVICE]);
    let mut session = session_with(
        doc.clone(),
        DevicePages::<B>::new(vec![Device::never_opening()]),
    );
    // 200 px rows of 3 bytes: four bands of 50 rows.
    session.band_bytes = 200 * 3 * 50;
    let mut cpu_session =
        crate::open_session_from_bytes(doc, &SessionConfig::with_policy(BackendPolicy::CpuOnly))
            .expect("CPU session");
    cpu_session.band_bytes = session.band_bytes;
    let params = RenderParams::new(2.0);
    let banded = |session: &crate::RasterSession, page| {
        let mut rows = Vec::new();
        let diag = crate::render_page_bands(session, page, &params, |band| {
            let crate::BandBitmap::Rgb8(bitmap) = band.bitmap else {
                panic!("an RGB band");
            };
            rows.extend_from_slice(bitmap.data());
            Ok::<(), std::convert::Infallible>(())
        })
        .expect("banded render");
        (rows, diag.device_page)
    };
    let warnings = warnings_of(|| {
        for page in 1..=2 {
            let (rows, outcome) = banded(&session, page);
            assert_eq!(outcome, DevicePage::Ineligible("larger than one band"));
            assert_eq!(rows, banded(&cpu_session, page).0, "page {page}");
        }
    });
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(
        warnings[0].starts_with("page 1 is larger than one band of 1 MiB"),
        "{}",
        warnings[0]
    );
    let counts = device_pages::<B>(&session).counts();
    assert_eq!(counts.ineligible, [("larger than one band", 2)]);
    assert_eq!((counts.rendered, counts.cpu(), counts.opening), (0, 2, 0));
    assert_eq!(
        counts.to_string(),
        "0 on the device, 2 on the CPU (larger than one band: 2)"
    );
}

/// GPU pages are refused at open without vector antialiasing, which
/// the device raster needs.
pub(super) fn gpu_pages_need_vector_antialiasing<B: Api>() {
    let mut config = SessionConfig::with_policy(B::FORCE);
    config.gpu_pages = true;
    config.vector_antialias = false;
    match crate::open_session_from_bytes(pdf(&[DEVICE]), &config) {
        Err(RasterError::InvalidOptions(msg)) => assert!(msg.contains("antialias"), "{msg}"),
        Err(other) => panic!("expected InvalidOptions, got {other:?}"),
        Ok(_) => panic!("a session without vector antialiasing accepted GPU pages"),
    }
}

/// A page goes where its expected wait is shortest: held pages and itself
/// times the mean run, a device without runs taking the shortest known
/// mean, the first device among equals.
#[test]
fn a_page_goes_where_it_expects_the_shortest_wait() {
    let load = |held, mean: Option<u128>| Load {
        held,
        mean_nanos: mean,
    };
    assert_eq!(choose(std::iter::empty()), None);
    assert_eq!(choose([load(0, None), load(0, None)].into_iter()), Some(0));
    assert_eq!(choose([load(1, None), load(0, None)].into_iter()), Some(1));
    // Three pages on a 2 ms device wait less than one on a 7 ms device;
    // four do not.
    let (fast, slow) = (Some(2_000_000), Some(7_000_000));
    assert_eq!(choose([load(2, fast), load(0, slow)].into_iter()), Some(0));
    assert_eq!(choose([load(3, fast), load(0, slow)].into_iter()), Some(1));
    // A device without a mean counts the fastest known mean.
    assert_eq!(choose([load(2, fast), load(2, None)].into_iter()), Some(0));
    assert_eq!(choose([load(2, fast), load(1, None)].into_iter()), Some(1));
    assert_eq!(
        choose([load(0, slow), load(1, None), load(0, fast)].into_iter()),
        Some(2)
    );
    assert_eq!(load(usize::MAX, Some(u128::MAX)).wait(1), u128::MAX);
}

/// Each claim takes the device where one more page expects the shortest
/// wait, so the pages held on a device push later claims to the other one;
/// released claims free their device.
pub(super) fn held_pages_push_claims_to_the_other_device<B: Api>() {
    let Some(session) = two_device_session::<B>(pdf(&[DEVICE])) else {
        return;
    };
    let pages = device_pages::<B>(&session);
    let mut claims = Vec::new();
    for _ in 0..64 {
        claims.push(pages.claim(None).expect("a device"));
        if pages.devices.iter().all(|d| d.in_flight() > 0) {
            break;
        }
    }
    assert!(
        pages.devices.iter().all(|d| d.in_flight() > 0),
        "64 held pages stayed on one device"
    );
    let held: usize = pages.devices.iter().map(Device::in_flight).sum();
    assert_eq!(held, claims.len());
    drop(claims);
    assert!(pages.devices.iter().all(|d| d.in_flight() == 0));
}

/// Pages rendered at once on every usable GPU (`all`) all render on a
/// device, with the CPU's bytes, and the counts name each device with the
/// pages it rendered: every device renders some, together all of them.
/// Enough pages are rendered at once that each device holds one before
/// any has a mean page raster.
fn pages_spread_over_two_devices<B: Api>() {
    const PAGES: u32 = 48;
    let doc = pdf(&[DEVICE; PAGES as usize]);
    let Some(session) = two_device_session::<B>(doc.clone()) else {
        return;
    };
    render_concurrently(&session, PAGES, &cpu(doc, 1));
    let counts = device_pages::<B>(&session).counts();
    eprintln!("{counts}");
    assert_eq!((counts.rendered, counts.cpu()), (PAGES, 0), "{counts}");
    assert_eq!(
        counts.by_device.len(),
        device_pages::<B>(&session).devices.len()
    );
    assert!(
        counts.by_device.iter().all(|(_, n)| *n > 0),
        "a device rendered no page: {counts}"
    );
    let sum: u32 = counts.by_device.iter().map(|&(_, n)| n).sum();
    assert_eq!(sum, PAGES, "{counts}");
    assert!(
        counts
            .by_device
            .iter()
            .all(|(name, _)| !name.contains("Basic Render") && !name.contains("llvmpipe")),
        "`all` listed a software device: {counts}"
    );
}

/// [`pages_spread_over_two_devices`].
#[cfg(feature = "vulkan")]
pub(super) fn concurrent_pages_spread_over_two_devices<B: Api>() {
    pages_spread_over_two_devices::<B>();
}

/// [`pages_spread_over_two_devices`] on Direct3D 12 adapters.
#[cfg(all(windows, feature = "d3d12"))]
pub(super) fn two_d3d12_adapters_share_pages<B: Api>() {
    pages_spread_over_two_devices::<B>();
}

/// A device error retires that device only: later pages render on the
/// other device, identically.
pub(super) fn a_failed_device_leaves_the_others_in_service<B: Api>() {
    let doc = pdf(&[DEVICE, DEVICE, DEVICE]);
    let Some(session) = two_device_session::<B>(doc.clone()) else {
        return;
    };
    let pages = device_pages::<B>(&session);
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
    let failed = pages
        .devices
        .iter()
        .position(Device::is_retired)
        .expect("a retired device");
    assert_eq!(counts.by_device[failed].1, 0, "{counts}");
    assert_eq!(counts.by_device[1 - failed].1, 2, "{counts}");
}

/// Page 1 of `session`, failed by the error of a device the API reports
/// lost ([`Api::lost_device`]): it renders on the CPU, identically to page
/// 1 of `doc`, and one device is retired with one warning naming it and
/// the likely cause of the loss (a GPU timeout exactly on Windows).
/// Returns the index of the retired device.
fn assert_lost_device_retired<B: Api>(session: &crate::RasterSession, doc: &[u8]) -> usize {
    let pages = device_pages::<B>(session);
    *pages.faults.fail_next.lock().expect("lock") = Some(B::lost_device());
    let mut first = None;
    let warnings =
        warnings_of(|| first = Some(render(session, 1, &RenderParams::new(2.0)).expect("render")));
    let (page, outcome) = first.expect("page 1 rendered");
    assert_eq!(outcome, DevicePage::Failed);
    assert_eq!(page, cpu(doc.to_vec(), 1));
    assert_eq!(pages.counts().failed, 1);
    let retired: Vec<usize> = (0..pages.devices.len())
        .filter(|&i| pages.devices[i].is_retired())
        .collect();
    let [failed] = retired[..] else {
        panic!("retired devices {retired:?}, not one");
    };
    let name = pages.devices[failed].name();
    let [warning] = &warnings[..] else {
        panic!("warnings {warnings:?}, not one");
    };
    assert!(warning.contains(&name), "{warning}");
    assert!(
        warning.contains(gpu::backend::DEVICE_LOST_CAUSE),
        "{warning}"
    );
    assert_eq!(warning.contains("TDR"), cfg!(windows), "{warning}");
    failed
}

/// A session's only device, reported lost, is retired with its cause
/// ([`assert_lost_device_retired`]); the next page is not offered to a
/// device and renders on the CPU, identically.
pub(super) fn a_lost_only_device_retires_and_the_pages_render_on_the_cpu<B: Api>() {
    let doc = pdf(&[DEVICE, DEVICE]);
    let Some(session) = gpu_session::<B>(doc.clone()) else {
        return;
    };
    assert_eq!(assert_lost_device_retired::<B>(&session, &doc), 0);
    let (page, outcome) = render(&session, 2, &RenderParams::new(2.0)).expect("render");
    assert_eq!(outcome, DevicePage::NotOffered);
    assert_eq!(page, cpu(doc, 2));
    let counts = device_pages::<B>(&session).counts();
    assert_eq!(
        (counts.failed, counts.retired, counts.rendered, counts.cpu()),
        (1, 1, 0, 2),
        "{counts}"
    );
}

/// Of two devices, the one reported lost is retired with its cause
/// ([`assert_lost_device_retired`]); the next page renders on the other
/// device, identically.
fn a_lost_device_is_retired_with_its_cause<B: Api>() {
    let doc = pdf(&[DEVICE, DEVICE]);
    let Some(session) = two_device_session::<B>(doc.clone()) else {
        return;
    };
    let failed = assert_lost_device_retired::<B>(&session, &doc);
    let (page, outcome) = render(&session, 2, &RenderParams::new(2.0)).expect("render");
    assert!(
        matches!(outcome, DevicePage::Rendered { .. }),
        "{outcome:?}"
    );
    assert_eq!(page, cpu(doc, 2));
    let counts = device_pages::<B>(&session).counts();
    assert_eq!(counts.by_device[failed].1, 0, "{counts}");
    assert_eq!(counts.by_device[1 - failed].1, 1, "{counts}");
}

/// [`a_lost_device_is_retired_with_its_cause`].
#[cfg(feature = "vulkan")]
pub(super) fn a_lost_device_retires_and_the_page_renders_on_the_cpu<B: Api>() {
    a_lost_device_is_retired_with_its_cause::<B>();
}

/// [`a_lost_device_is_retired_with_its_cause`] with the error of a
/// signal on a removed adapter (`DXGI_ERROR_DEVICE_REMOVED`).
#[cfg(all(windows, feature = "d3d12"))]
pub(super) fn a_removed_d3d12_device_retires_and_the_page_renders_on_the_cpu<B: Api>() {
    a_lost_device_is_retired_with_its_cause::<B>();
}

/// A device list naming one device twice is refused at open, under each
/// policy that renders GPU pages on the API.
pub(super) fn a_device_list_naming_one_device_twice_is_refused<B: Api>() {
    if let Err(e) = B::usable_gpus() {
        eprintln!("skipping: no usable GPU on {}: {e}", B::API);
        return;
    }
    let policies = if B::UNDER_AUTO {
        vec![BackendPolicy::Auto, B::FORCE]
    } else {
        vec![B::FORCE]
    };
    for policy in policies {
        let mut config = SessionConfig::with_policy(policy);
        config.gpu_pages = true;
        config.gpu_devices = vec!["0".to_owned(), "0".to_owned()];
        match crate::open_session_from_bytes(pdf(&[DEVICE]), &config) {
            Err(RasterError::InvalidOptions(msg)) => {
                assert!(msg.contains("both name device 0"), "{msg}");
            }
            Err(other) => panic!("expected InvalidOptions, got {other:?}"),
            Ok(_) => panic!("a device list naming one device twice was accepted"),
        }
    }
}

/// The WARP adapter, named explicitly (`warp`, or its listed index),
/// renders GPU pages, identically to the CPU, under its own name.
#[cfg(all(windows, feature = "d3d12"))]
pub(super) fn an_explicit_warp_renders_gpu_pages<B: Api>() {
    let doc = pdf(&[DEVICE]);
    let mut config = SessionConfig::with_policy(B::FORCE);
    config.gpu_pages = true;
    config.gpu_devices = vec!["warp".to_owned()];
    let Some(session) = super::or_skip::<B, _, _>(
        "the WARP adapter does not open",
        crate::open_session_from_bytes(doc.clone(), &config),
    ) else {
        return;
    };
    let (page, outcome) = render(&session, 1, &RenderParams::new(2.0)).expect("render");
    assert!(
        matches!(outcome, DevicePage::Rendered { .. }),
        "{outcome:?}"
    );
    assert_eq!(page, cpu(doc.clone(), 1));
    let counts = device_pages::<B>(&session).counts();
    let [(name, 1)] = &counts.by_device[..] else {
        panic!("{counts}");
    };
    assert!(name.contains("Basic Render"), "{counts}");

    let warp = B::resolve_devices(&["warp".parse().expect("a selector")]).expect("WARP listed");
    config.gpu_devices = vec![warp[0].to_string()];
    let by_index = crate::open_session_from_bytes(doc, &config).expect("WARP by its index");
    let counts = device_pages::<B>(&by_index).counts();
    assert_eq!(counts.by_device.len(), 1, "{counts}");
    assert!(counts.by_device[0].0.contains("Basic Render"), "{counts}");
}
