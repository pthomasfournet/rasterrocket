//! The WARP module a Direct3D 12 test loaded.  Included only by the test
//! binaries of a Windows build with the `d3d12` feature, whose library
//! exports [`rasterrocket::dev::loaded_warp`].

/// Print the `d3d10warp.dll` loaded in the process, if any; with
/// `RR_WARP_VERSION` set, require a loaded one beside the test binary, of
/// that version.
///
/// # Panics
///
/// When `RR_WARP_VERSION` is set and the loaded module is missing, lies
/// elsewhere or has another version.
pub(super) fn report_loaded_warp() {
    let loaded = rasterrocket::dev::loaded_warp();
    match &loaded {
        Some((path, version)) => eprintln!("WARP module: {} version {version}", path.display()),
        None => eprintln!("WARP module: not loaded"),
    }
    if let Ok(expected) = std::env::var("RR_WARP_VERSION") {
        let (path, version) = loaded.expect("RR_WARP_VERSION is set, but no WARP is loaded");
        let exe = std::env::current_exe().expect("test binary path");
        let canonical = |p: &std::path::Path| std::fs::canonicalize(p).ok();
        assert_eq!(
            path.parent().and_then(canonical),
            exe.parent().and_then(canonical),
            "the loaded WARP is not the one beside the test binary: {}",
            path.display()
        );
        assert_eq!(version, expected.trim(), "WARP version");
    }
}
