//! Build script: with the `vulkan` feature, compiles
//! `kernels/page_raster.slang` to SPIR-V in `OUT_DIR` with `slangc`
//! (`$SLANGC` overrides the binary), with the profile the gpu crate's
//! kernels use.  Every `[shader("compute")]` entry of the file is kept.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=kernels/");
    println!("cargo:rerun-if-env-changed=SLANGC");
    if env::var("CARGO_FEATURE_VULKAN").is_err() {
        return;
    }
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let slangc = env::var("SLANGC").unwrap_or_else(|_| "slangc".to_owned());
    let src = PathBuf::from("kernels/page_raster.slang");
    let spv = out_dir.join("page_raster.spv");
    let output = Command::new(&slangc)
        .args(["-target", "spirv", "-profile", "spirv_1_5", "-o"])
        .arg(&spv)
        .arg(&src)
        .output()
        .unwrap_or_else(|e| panic!("failed to run slangc ({slangc}): {e}"));
    assert!(
        output.status.success(),
        "slangc failed for {} ({}):\n{}",
        src.display(),
        output.status,
        String::from_utf8_lossy(&output.stderr).trim(),
    );
}
