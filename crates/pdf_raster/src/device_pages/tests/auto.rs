//! GPU pages under `Auto`, which renders them on Vulkan devices opened
//! when a page first asks for one.

use gpu::backend::vulkan::VulkanBackend;

use super::{DEVICE, cpu, device_pages, pdf, render};
use crate::{BackendPolicy, DevicePage, RasterError, RasterSession, RenderParams, SessionConfig};

/// An `Auto` GPU-page session over `pdf` opening the device `selector`
/// names.
fn auto_session(pdf: Vec<u8>, selector: Option<String>) -> RasterSession {
    let mut config = SessionConfig::with_policy(BackendPolicy::Auto);
    config.vulkan_device = selector;
    config.gpu_pages = true;
    crate::open_session_from_bytes(pdf, &config).expect("an Auto session opens")
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
    let pages = device_pages::<VulkanBackend>(&session);
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
    let pages = device_pages::<VulkanBackend>(&session);
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
