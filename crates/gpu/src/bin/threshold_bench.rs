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
//! target/release/threshold_bench [--iters N] [--warmup N]
//! ```
//!
//! Output: a table per kernel and a recommended threshold per constant.

use std::time::Instant;

use rasterrocket_gpu::{
    AA_SIZE, AaFillRect, AaScanTable, GpuCtx, ScanIntersect, aa_fill_cpu, icc_cmyk_to_rgb_cpu,
};

/// Unwrap a GPU `Result`, printing the error with context and exiting with
/// status 1.  Used inside timing closures where `?` is not available.
fn gpu_unwrap<T>(res: Result<T, Box<dyn std::error::Error>>, ctx: &str) -> T {
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
}

impl Default for Config {
    fn default() -> Self {
        Self {
            iters: 20,
            warmup: 5,
        }
    }
}

fn parse_config() -> Config {
    let mut cfg = Config::default();
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--iters" => {
                if let Some(v) = args.next() {
                    cfg.iters = v.parse().unwrap_or(cfg.iters);
                }
            }
            "--warmup" => {
                if let Some(v) = args.next() {
                    cfg.warmup = v.parse().unwrap_or(cfg.warmup);
                }
            }
            _ => {}
        }
    }
    cfg
}

// ── Geometric pixel-count sweep ─────────────────────────────────────────────

/// Sizes to sweep: powers-of-2 from 2^8 to 2^22 (256 to 4M pixels).
fn sweep_sizes() -> Vec<usize> {
    (8u32..=22).map(|e| 1usize << e).collect()
}

/// Width/height for a near-square bounding box of `n` pixels.
fn dims(n: usize) -> (u32, u32) {
    // n ≤ 2^22 = 4M; sqrt ≤ 2048; fits u32 without truncation.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "n ≤ 4_194_304 (22-bit); exact in f64 (52-bit mantissa); sqrt ≤ 2049 — fits u32"
    )]
    let w = (n as f64).sqrt().ceil() as u32;
    #[expect(clippy::cast_possible_truncation, reason = "n ≤ 4_194_304 — fits u32")]
    let h = (n as u32).div_ceil(w);
    (w, h)
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

// ── Intersection-table fixture: a w×h rectangle at the origin ──────────────

/// Scanline intersection table of a w×h pixel rectangle at (0, 0): every
/// AA row carries the left edge at AA column 0 and the right edge at
/// `w × AA_SIZE - 1`.
fn rect_table(w: u32, h: u32) -> (Vec<ScanIntersect>, Vec<u32>) {
    let n_rows = h as usize * AA_SIZE as usize;
    let right = i32::try_from(w).expect("width fits i32") * AA_SIZE - 1;
    let mut intersects = Vec::with_capacity(n_rows * 2);
    let mut row_starts = Vec::with_capacity(n_rows + 1);
    for r in 0..n_rows {
        row_starts.push(u32::try_from(r * 2).expect("table fits u32"));
        intersects.push(ScanIntersect {
            x0: 0,
            x1: 0,
            count: -1,
        });
        intersects.push(ScanIntersect {
            x0: right,
            x1: right,
            count: 1,
        });
    }
    row_starts.push(u32::try_from(n_rows * 2).expect("table fits u32"));
    (intersects, row_starts)
}

// ── AA fill benchmark ───────────────────────────────────────────────────────

fn bench_aa_fill(ctx: &GpuCtx, cfg: &Config) {
    println!("\n══ aa_fill: GPU scanline walk vs host twin (informational) ══");
    println!(
        "{:>10}  {:>12}  {:>12}  {:>8}  winner",
        "pixels", "cpu_ns", "gpu_ns", "ratio"
    );
    println!("{}", "─".repeat(60));

    for &n in &sweep_sizes() {
        let (w, h) = dims(n);
        let (intersects, row_starts) = rect_table(w, h);
        let table = AaScanTable {
            intersects: &intersects,
            row_starts: &row_starts,
            y_min: 0,
        };
        let rect = AaFillRect {
            x: 0,
            y: 0,
            width: w,
            height: h,
        };
        let aa_width = w * AA_SIZE.unsigned_abs();

        let cpu_ns = time_ns(
            || {
                let _ = aa_fill_cpu(&table, rect, aa_width, false);
            },
            cfg.warmup,
            cfg.iters,
        );

        let gpu_ns = time_ns(
            || {
                let _ = gpu_unwrap(ctx.aa_fill(&table, rect, aa_width, false), "aa_fill");
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
    }

    println!();
    println!(
        "  → The host twin is the per-pixel parity oracle, not the production CPU fill \
         (bit-run spans + SIMD popcount), so this table cannot calibrate \
         GPU_AA_FILL_THRESHOLD.  Calibrate end to end: render with \
         PDF_RASTER_AA_FILL_THRESHOLD=<px> under --backend cuda against --backend cpu."
    );
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
                let _ = icc_cmyk_to_rgb_cpu(&cmyk, None);
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
    println!("Machine: Ryzen 9 9900X3D + RTX 5070 (CC 12.0, PCIe 5.0)");

    let ctx = GpuCtx::init().unwrap_or_else(|e| {
        eprintln!("threshold_bench: GPU init failed: {e}");
        eprintln!("Run `nvidia-smi` to verify the driver is loaded.");
        std::process::exit(1);
    });

    bench_aa_fill(&ctx, &cfg);
    bench_icc_cmyk(&ctx, &cfg);

    println!("\nDone.");
}
