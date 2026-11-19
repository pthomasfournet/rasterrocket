use super::BackendPolicy;

#[test]
fn backend_policy_parses_known_names() {
    use std::str::FromStr;
    assert_eq!(BackendPolicy::from_str("auto"), Ok(BackendPolicy::Auto));
    assert_eq!(BackendPolicy::from_str("cpu"), Ok(BackendPolicy::CpuOnly));
    assert_eq!(
        BackendPolicy::from_str("cuda"),
        Ok(BackendPolicy::ForceCuda)
    );
    #[cfg(feature = "vaapi")]
    assert_eq!(
        BackendPolicy::from_str("vaapi"),
        Ok(BackendPolicy::ForceVaapi)
    );
    assert_eq!(
        BackendPolicy::from_str("vulkan"),
        Ok(BackendPolicy::ForceVulkan)
    );
    assert_eq!(
        BackendPolicy::from_str("d3d12"),
        Ok(BackendPolicy::ForceD3d12)
    );
    assert_eq!(
        BackendPolicy::from_str("D3D12"),
        Ok(BackendPolicy::ForceD3d12)
    );
    assert_eq!(BackendPolicy::ForceD3d12.name(), "d3d12");
}

#[test]
fn backend_policy_name_round_trips() {
    use std::str::FromStr;
    for &policy in BackendPolicy::ALL {
        assert_eq!(BackendPolicy::from_str(policy.name()), Ok(policy));
    }
}

#[test]
fn backend_policy_empty_is_auto() {
    use std::str::FromStr;
    assert_eq!(BackendPolicy::from_str(""), Ok(BackendPolicy::Auto));
}

#[test]
fn backend_policy_trims_and_ignores_case() {
    use std::str::FromStr;
    assert_eq!(
        BackendPolicy::from_str("  CUDA  "),
        Ok(BackendPolicy::ForceCuda)
    );
}

#[test]
fn backend_policy_rejects_unknown_naming_the_accepted_names() {
    use std::str::FromStr;
    let err = BackendPolicy::from_str("metal").unwrap_err();
    assert_eq!(err.value(), "metal");
    let msg = err.to_string();
    assert!(
        msg.contains("\"metal\"") && msg.contains("auto, cpu, cuda"),
        "{msg}"
    );
    assert_eq!(msg.contains("vaapi"), cfg!(feature = "vaapi"), "{msg}");
}

/// GPU pages are accepted under `ForceD3d12` in a Windows build with
/// the `d3d12` feature, with a device list beside the session's own
/// adapter; elsewhere they are refused naming the feature.
#[test]
fn gpu_pages_accept_force_d3d12() {
    use super::SessionConfig;
    use crate::RasterError;
    let mut config = SessionConfig::with_policy(BackendPolicy::ForceD3d12);
    config.gpu_pages = true;
    config.gpu_devices = vec!["all".to_owned()];
    config.d3d12_device = Some("0".to_owned());
    let d3d12 = cfg!(all(windows, feature = "d3d12"));
    match config.check_gpu_pages() {
        Ok(()) => assert!(d3d12, "accepted without the D3D12 backend"),
        Err(RasterError::InvalidOptions(msg)) => {
            assert!(!d3d12, "{msg}");
            let named = if cfg!(windows) {
                "`d3d12` feature"
            } else {
                "Windows only"
            };
            assert!(msg.contains(named), "{msg}");
        }
        Err(other) => panic!("expected InvalidOptions, got {other:?}"),
    }
}

/// GPU pages under a backend that renders none are refused naming the
/// backends that do on the platform: `auto`, `vulkan`, and `d3d12` on
/// Windows only.
#[test]
fn gpu_pages_are_refused_under_other_backends_naming_those_that_render() {
    use super::SessionConfig;
    use crate::RasterError;
    for policy in [BackendPolicy::CpuOnly, BackendPolicy::ForceCuda] {
        let mut config = SessionConfig::with_policy(policy);
        config.gpu_pages = true;
        let Err(RasterError::InvalidOptions(msg)) = config.check_gpu_pages() else {
            panic!("{policy:?}: GPU pages accepted");
        };
        if cfg!(device_pages) {
            for named in [
                format!("--backend {}", policy.name()),
                "--backend auto".to_owned(),
                "--backend vulkan".to_owned(),
            ] {
                assert!(msg.contains(&named), "{policy:?}: {msg}");
            }
            assert_eq!(
                msg.contains("d3d12"),
                cfg!(windows),
                "{policy:?}: Direct3D 12 named off Windows or omitted on it: {msg}"
            );
        } else {
            assert!(msg.contains("`vulkan`"), "{policy:?}: {msg}");
            assert_eq!(msg.contains("`d3d12`"), cfg!(windows), "{policy:?}: {msg}");
        }
    }
}

/// The variable a child process of this test runs under.
const CHILD: &str = "RR_CONFIG_ENV_CHILD";

/// Run the test `name` of this binary alone in a child process, under
/// `env` and without the device variables the parent has.
fn run_child(name: &str, env: &[(&str, &str)]) {
    let exe = std::env::current_exe().expect("the test binary's path");
    let status = std::process::Command::new(exe)
        .args(["--exact", name, "--test-threads=1", "--quiet"])
        .env(CHILD, "1")
        .env_remove(super::session::VULKAN_DEVICE_ENV)
        .env_remove(super::session::D3D12_DEVICE_ENV)
        .envs(env.iter().copied())
        .status()
        .expect("the child test runs");
    assert!(status.success(), "the child test {name} failed: {status}");
}

/// `PDF_RASTER_D3D12_DEVICE`, trimmed, is the D3D12 selector, and the
/// Vulkan one stays unset; `SessionConfig::default` takes it.
#[test]
fn d3d12_device_from_env_reads_its_variable() {
    use super::SessionConfig;
    if std::env::var_os(CHILD).is_some() {
        assert_eq!(
            SessionConfig::d3d12_device_from_env().as_deref(),
            Some("warp")
        );
        assert_eq!(SessionConfig::vulkan_device_from_env(), None);
        let config = SessionConfig::default();
        assert_eq!(config.d3d12_device.as_deref(), Some("warp"));
        assert_eq!(config.vulkan_device, None);
        return;
    }
    run_child(
        "config::tests::d3d12_device_from_env_reads_its_variable",
        &[("PDF_RASTER_D3D12_DEVICE", "  warp ")],
    );
}

/// `PDF_RASTER_VULKAN_DEVICE`, trimmed, is the Vulkan selector, and the
/// D3D12 one stays unset.
#[test]
fn vulkan_device_from_env_reads_its_variable() {
    use super::SessionConfig;
    if std::env::var_os(CHILD).is_some() {
        assert_eq!(
            SessionConfig::vulkan_device_from_env().as_deref(),
            Some("2080")
        );
        assert_eq!(SessionConfig::d3d12_device_from_env(), None);
        return;
    }
    run_child(
        "config::tests::vulkan_device_from_env_reads_its_variable",
        &[("PDF_RASTER_VULKAN_DEVICE", " 2080")],
    );
}

/// A blank device variable selects nothing.
#[test]
fn a_blank_device_variable_is_no_selector() {
    use super::SessionConfig;
    if std::env::var_os(CHILD).is_some() {
        assert_eq!(SessionConfig::d3d12_device_from_env(), None);
        assert_eq!(SessionConfig::vulkan_device_from_env(), None);
        return;
    }
    run_child(
        "config::tests::a_blank_device_variable_is_no_selector",
        &[
            ("PDF_RASTER_D3D12_DEVICE", "  "),
            ("PDF_RASTER_VULKAN_DEVICE", ""),
        ],
    );
}
