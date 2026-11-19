//! `page_gpu_bench`: the page-level GPU raster benchmark; see `run.rs` for
//! its modes and flags.  Built with `--features vulkan`, `--features d3d12`
//! (Windows), or both.

#[cfg(any(feature = "vulkan", all(windows, feature = "d3d12")))]
mod run;
#[cfg(any(feature = "vulkan", all(windows, feature = "d3d12")))]
mod stars;

#[cfg(any(feature = "vulkan", all(windows, feature = "d3d12")))]
fn main() {
    run::main();
}

#[cfg(not(any(feature = "vulkan", all(windows, feature = "d3d12"))))]
fn main() {
    eprintln!("page_gpu_bench needs `--features vulkan`, or `--features d3d12` on Windows");
    std::process::exit(2);
}
