// The kernels `build.rs` compiles, by the names their artifacts take in
// `OUT_DIR`; shared by `build.rs` and the library's tests through
// `include!`.

/// Kernels with a CUDA source as well as a Slang one: compiled to PTX by
/// nvcc, and by slangc to SPIR-V (`vulkan`) and DXIL (`d3d12`).
pub(crate) const CUDA_KERNELS: &[&str] = &["icc_clut", "blit_image"];

/// Kernels with a Slang source only, compiled to SPIR-V (`vulkan`) and
/// DXIL (`d3d12`).  The library embeds `page_raster` and
/// `page_raster_paths` in every build, the `floor_div_probe` kernels and
/// `page_raster_paths_poisoned` in its tests only.  A build script does not
/// see `cfg(test)`, so the test-only kernels are compiled in every build
/// and are never embedded outside the tests.
pub(crate) const SLANG_ONLY_KERNELS: &[&str] = &[
    "page_raster",
    "page_raster_paths",
    "floor_div_probe",
    "floor_div_probe_poisoned",
    "page_raster_paths_poisoned",
];
