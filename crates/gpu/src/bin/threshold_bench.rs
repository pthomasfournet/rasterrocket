//! GPU dispatch threshold calibration.
// All timing values are display-only; f64 precision loss on u64 nanosecond
// counts (ratios, formatted durations) is intentional and inconsequential.
#![expect(
    clippy::cast_precision_loss,
    reason = "display-only timing ratios and ns→µs/ms conversions"
)]
//!
//! Measures the breakeven pixel count where GPU dispatch becomes faster than
//! the CPU fallback for each kernel, accounting for actual `PCIe` 5.0 H2D/D2H
//! latency on this machine.
//!
//! Run:
//! ```text
//! CUDA_ARCH=sm_120 cargo build --release -p rasterrocket-gpu --bin threshold_bench
//! target/release/threshold_bench [--iters N] [--warmup N] [--only icc|clut]
//! ```
//!
//! With the `vulkan` feature, `--vulkan-device NAME` (repeatable) adds the
//! ICC CLUT sweep on each named Vulkan device.
//!
//! Output: a table per kernel and a recommended threshold per constant.

#[path = "bench_common/mod.rs"]
mod bench_common;

use std::fmt::Display;
use std::sync::Arc;
use std::time::Instant;

use bench_common::smooth_cmyk;
use rasterrocket_gpu::GpuCtx;
use rasterrocket_gpu::backend::cuda::CudaBackend;
use rasterrocket_gpu::backend::{GpuBackend, icc_clut, random_grid};

/// Unwrap a GPU `Result`, printing the error with context and exiting with
/// status 1.  Used inside timing closures where `?` is not available.
fn gpu_unwrap<T, E: Display>(res: Result<T, E>, ctx: &str) -> T {
    res.unwrap_or_else(|e| {
        eprintln!("threshold_bench: GPU error in {ctx}: {e}");
        std::process::exit(1);
    })
}

// ── Config ─────────────────────────────────────────────────────────────────

struct Config {
    /// Number of timed iterations per (kernel, size) cell.
    iters: u32,
    /// Number of warmup iterations (timed but discarded).
    warmup: u32,
    /// Run only the named benchmark (`icc` or `clut`).
    only: Option<String>,
    /// Vulkan devices the ICC CLUT sweep also runs on.
    vulkan_devices: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            iters: 20,
            warmup: 5,
            only: None,
            vulkan_devices: Vec::new(),
        }
    }
}

/// Command-line usage.
const USAGE: &str = "threshold_bench [--iters N] [--warmup N] [--only icc|clut] \
                     [--vulkan-device NAME]...";

fn parse_config() -> Config {
    let mut cfg = Config::default();
    let known = ["--iters", "--warmup", "--only", "--vulkan-device"];
    for (flag, value) in bench_common::flag_pairs(USAGE, &known) {
        match flag.as_str() {
            "--iters" => cfg.iters = bench_common::parse_value(USAGE, &flag, &value),
            "--warmup" => cfg.warmup = bench_common::parse_value(USAGE, &flag, &value),
            "--only" => {
                cfg.only = Some(bench_common::only_value(USAGE, value, &["icc", "clut"]));
            }
            _ if cfg!(feature = "vulkan") => cfg.vulkan_devices.push(value),
            _ => bench_common::usage_error(
                USAGE,
                "--vulkan-device needs a build with the `vulkan` feature",
            ),
        }
    }
    cfg
}

// ── Geometric pixel-count sweep ─────────────────────────────────────────────

/// Sizes to sweep: powers-of-2 from 2^8 to 2^22 (256 to 4M pixels).
fn sweep_sizes() -> Vec<usize> {
    (8u32..=22).map(|e| 1usize << e).collect()
}

// ── Timing helpers ──────────────────────────────────────────────────────────

fn median_ns(mut samples: Vec<u64>) -> u64 {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

/// Run `f` `warmup` times then `iters` times; return median duration in ns.
fn time_ns<F: FnMut()>(mut f: F, warmup: u32, iters: u32) -> u64 {
    for _ in 0..warmup {
        f();
    }
    // as_nanos() → u128; truncation to u64 is safe: u64::MAX ns ≈ 584 years.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "u64::MAX ns ≈ 584 years; no benchmark runs that long"
    )]
    let samples: Vec<u64> = (0..iters)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_nanos() as u64
        })
        .collect();
    median_ns(samples)
}

// ── ICC CMYK→RGB benchmark ──────────────────────────────────────────────────

fn bench_icc_cmyk(ctx: &GpuCtx, cfg: &Config) {
    // NOTE: this bench measures the raw GPU matrix kernel (clut=None) against
    // the CPU SIMD path.  The dispatch in icc_cmyk_to_rgb() always short-circuits to CPU
    // for clut=None regardless of pixel count, so the crossover below is informational
    // only.  To calibrate GPU_ICC_CLUT_THRESHOLD, re-run with a real CLUT workload.
    println!("\n══ icc_cmyk: GPU matrix kernel vs CPU SIMD (matrix path, informational) ══");
    println!(
        "{:>10}  {:>12}  {:>12}  {:>8}  winner",
        "pixels", "cpu_ns", "gpu_ns", "ratio"
    );
    println!("{}", "─".repeat(60));

    let mut crossover: Option<usize> = None;

    for &n in &sweep_sizes() {
        // CMYK input: alternating test values covering full range
        // i % 256 is in 0..=255 — truncation to u8 is exact by construction.
        #[expect(clippy::cast_possible_truncation, reason = "i % 256 is in 0..=255")]
        let cmyk: Vec<u8> = (0..n * 4).map(|i| (i % 256) as u8).collect();

        let cpu_ns = time_ns(
            || {
                let _ = color::cmyk::cmyk_to_rgb_vec(&cmyk);
            },
            cfg.warmup,
            cfg.iters,
        );

        // Use the unconditional GPU path to measure actual dispatch cost at
        // all sizes, not just those above the current threshold.
        let gpu_ns = time_ns(
            || {
                let _ = gpu_unwrap(ctx.icc_cmyk_to_rgb_gpu(&cmyk, None), "icc_cmyk_to_rgb_gpu");
            },
            cfg.warmup,
            cfg.iters,
        );

        let ratio = cpu_ns as f64 / gpu_ns as f64;
        let winner = if gpu_ns < cpu_ns { "GPU ✓" } else { "CPU" };

        println!(
            "{:>10}  {:>12}  {:>12}  {:>7.2}x  {}",
            n,
            fmt_ns(cpu_ns),
            fmt_ns(gpu_ns),
            ratio,
            winner
        );

        if crossover.is_none() && gpu_ns < cpu_ns {
            crossover = Some(n);
        }
    }

    println!();
    match crossover {
        Some(px) => println!(
            "  → Matrix kernel crossover at {px} px — but dispatch always uses CPU for \
             clut=None; GPU_ICC_CLUT_THRESHOLD={} applies to CLUT path only",
            rasterrocket_gpu::GPU_ICC_CLUT_THRESHOLD,
        ),
        None => println!(
            "  → GPU matrix kernel never faster — CPU AVX-512 dominates at all sizes. \
             Dispatch short-circuits to CPU for clut=None (expected)."
        ),
    }
}

// ── ICC CMYK→RGB CLUT benchmark ─────────────────────────────────────────────

/// The production device conversion ([`icc_clut`]: staged uploads, kernel
/// and download in one page) on `backend` against the host evaluator of the
/// same grid ([`color::grid::CmykGrid::convert`], one pixel at a time; the
/// renderer's CPU conversion of the grid runs in blocks and is faster), on
/// a grid of 17⁴ nodes, across image sizes.
fn bench_icc_clut<B: GpuBackend>(label: &str, backend: &B, cfg: &Config) {
    println!(
        "\n══ icc_clut on {label}: device call vs host evaluator (17⁴ nodes, smooth input) ══"
    );
    println!(
        "{:>10}  {:>12}  {:>12}  {:>8}  winner",
        "pixels", "cpu", "device", "ratio"
    );
    println!("{}", "─".repeat(60));
    let grid = random_grid(1, [16; 4], color::grid::NodeRange::Extended);
    let mut crossover: Option<usize> = None;
    for n in [
        1_000, 4_000, 16_000, 32_000, 64_000, 128_000, 256_000, 512_000, 1_000_000, 2_000_000,
        4_000_000, 8_000_000,
    ] {
        let cmyk = smooth_cmyk(n);
        let cpu_ns = time_ns(
            || {
                let _ = std::hint::black_box(grid.convert(&cmyk));
            },
            cfg.warmup.min(2),
            cfg.iters.min(7),
        );
        let dev_ns = time_ns(
            || {
                let _ =
                    std::hint::black_box(gpu_unwrap(icc_clut(backend, &cmyk, &grid), "icc_clut"));
            },
            cfg.warmup,
            cfg.iters,
        );
        let winner = if dev_ns < cpu_ns { "device ✓" } else { "CPU" };
        println!(
            "{:>10}  {:>12}  {:>12}  {:>7.2}x  {}",
            n,
            fmt_ns(cpu_ns),
            fmt_ns(dev_ns),
            cpu_ns as f64 / dev_ns as f64,
            winner
        );
        if crossover.is_none() && dev_ns < cpu_ns {
            crossover = Some(n);
        }
    }
    match crossover {
        Some(px) => println!("  → device faster from {px} px"),
        None => println!("  → device never faster"),
    }
}

/// [`bench_icc_clut`] on each Vulkan device named in `cfg`, timing the
/// backend's construction and first exactness check.
#[cfg(feature = "vulkan")]
fn bench_icc_clut_vulkan(cfg: &Config) {
    use rasterrocket_gpu::backend::page_device::PageDevice;
    use rasterrocket_gpu::backend::select::DeviceSelector;
    use rasterrocket_gpu::backend::vulkan::VulkanBackend;
    for name in &cfg.vulkan_devices {
        let selector: DeviceSelector = gpu_unwrap(name.parse(), "device selector");
        let t = Instant::now();
        let backend = gpu_unwrap(VulkanBackend::open(Some(&selector)), "Vulkan init");
        let init = t.elapsed();
        let t = Instant::now();
        let exact = backend.icc_clut_is_exact();
        println!(
            "\nVulkan {name} ({}): backend init {init:.1?}, exactness check {:.1?} (exact: {exact}), integrated: {}",
            backend.device_name(),
            t.elapsed(),
            backend.is_integrated()
        );
        bench_icc_clut(&format!("Vulkan {name}"), &backend, cfg);
    }
}

// ── Formatting ──────────────────────────────────────────────────────────────

fn fmt_ns(ns: u64) -> String {
    if ns >= 1_000_000 {
        format!("{:.2} ms", ns as f64 / 1_000_000.0)
    } else if ns >= 1_000 {
        format!("{:.1} µs", ns as f64 / 1_000.0)
    } else {
        format!("{ns} ns")
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let cfg = parse_config();

    println!("GPU dispatch threshold calibration");
    println!("iters={}, warmup={}", cfg.iters, cfg.warmup);

    let ctx = match GpuCtx::init() {
        Ok(ctx) => {
            match cudarc::driver::CudaContext::new(0).and_then(|c| c.name()) {
                Ok(name) => println!("CUDA device 0: {name}"),
                Err(e) => println!("CUDA device 0: name unavailable ({e})"),
            }
            Some(Arc::new(ctx))
        }
        Err(e) => {
            eprintln!("threshold_bench: CUDA init failed ({e}); CUDA benchmarks skipped");
            None
        }
    };
    let run = |name: &str| cfg.only.as_deref().is_none_or(|only| only == name);
    if let Some(ctx) = &ctx {
        if run("icc") {
            bench_icc_cmyk(ctx, &cfg);
        }
        if run("clut") {
            bench_icc_clut("CUDA", &CudaBackend::from_ctx(Arc::clone(ctx)), &cfg);
        }
    }
    #[cfg(feature = "vulkan")]
    if run("clut") {
        bench_icc_clut_vulkan(&cfg);
    }

    println!("\nDone.");
}
