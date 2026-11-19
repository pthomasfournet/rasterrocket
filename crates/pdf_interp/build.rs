//! Build script.
//!
//! Emits the `device_pages` cfg: set when the crate is built with a GPU
//! backend that renders whole pages on the device, i.e. when
//! `any(feature = "vulkan", all(windows, feature = "d3d12"))` holds.  The
//! `d3d12` feature compiles to nothing off Windows, so it counts only on a
//! Windows target.  The cfg applies to every target of the package (library,
//! tests, examples) and is declared to `check-cfg`.

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rustc-check-cfg=cfg(device_pages)");
    let feature = |name: &str| std::env::var_os(format!("CARGO_FEATURE_{name}")).is_some();
    let windows = std::env::var_os("CARGO_CFG_WINDOWS").is_some();
    if feature("VULKAN") || (windows && feature("D3D12")) {
        println!("cargo::rustc-cfg=device_pages");
    }
}
