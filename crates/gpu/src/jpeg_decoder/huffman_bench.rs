//! Wall-clock benchmark of the parallel-Huffman symbol decode against
//! the sequential CPU coefficient walk, over a directory of JPEGs.
//!
//! Ignored by default; run in release mode on a CUDA host:
//!
//! ```text
//! PDF_RASTER_HUFFMAN_BENCH_DIR=/path/to/jpegs \
//!   cargo test --release -p rasterrocket-gpu \
//!   --features cache,gpu-validation,gpu-jpeg-huffman --lib \
//!   -- huffman_bench --ignored --nocapture
//! ```
//!
//! Optional variables: `PDF_RASTER_HUFFMAN_BENCH_SSB` (comma-separated
//! subsequence sizes; `pick` selects `pick_subsequence_size`; default
//! `pick,128,256,512,1024,2048,4096`), `PDF_RASTER_HUFFMAN_BENCH_ITERS`
//! (timed repetitions per configuration, default 7; the median is
//! reported), and `PDF_RASTER_HUFFMAN_BENCH_MAX_PASSES` /
//! `PDF_RASTER_HUFFMAN_BENCH_MAX_REDECODES` (Phase 2 budget, default
//! `Phase2Budget::PRODUCTION`).
//!
//! Every GPU configuration is checked coefficient-for-coefficient
//! against the CPU walk before it is timed, so a reported time is
//! always the time of a correct decode. `sym_ms` is the GPU symbol
//! decode alone; `gpu_ms` adds the CPU magnitude-bit pass that turns
//! symbols into coefficients — the full replacement for `cpu_ms`. A
//! configuration that exhausts the Phase 2 budget is reported as
//! `fallback`: `sym_ms` is the time spent before the GPU decode gave
//! up and `gpu_ms` that time plus `cpu_ms`, the cost the decoder's CPU
//! fallback pays.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::backend::cuda::CudaBackend;
use crate::jpeg_decoder::cpu_prepass::{JpegPreparedInput, prepare_jpeg};
use crate::jpeg_decoder::decoder::{extract_coefficients_pub, symbols_to_coefficients};
use crate::jpeg_decoder::huffman::{Phase2Budget, decode_jpeg_symbols_within};
use crate::jpeg_decoder::pick_subsequence_size;

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_owned())
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn time_median<T>(iters: usize, mut f: impl FnMut() -> T) -> Duration {
    let samples = (0..iters)
        .map(|_| {
            let t = Instant::now();
            let value = std::hint::black_box(f());
            let elapsed = t.elapsed();
            drop(value);
            elapsed
        })
        .collect();
    median(samples)
}

fn jpeg_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .is_some_and(|x| x.eq_ignore_ascii_case("jpg") || x.eq_ignore_ascii_case("jpeg"))
        })
        .collect();
    files.sort();
    files
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// Benchmark settings read from the environment.
struct Settings {
    iters: usize,
    budget: Phase2Budget,
    sizes: String,
}

impl Settings {
    fn from_env() -> Self {
        let int = |name: &str, default: u32| -> u32 {
            env_or(name, &default.to_string())
                .parse()
                .unwrap_or_else(|_| panic!("{name} must be an integer"))
        };
        Self {
            iters: int("PDF_RASTER_HUFFMAN_BENCH_ITERS", 7) as usize,
            budget: Phase2Budget {
                passes: int(
                    "PDF_RASTER_HUFFMAN_BENCH_MAX_PASSES",
                    Phase2Budget::PRODUCTION.passes,
                ),
                redecodes: int(
                    "PDF_RASTER_HUFFMAN_BENCH_MAX_REDECODES",
                    Phase2Budget::PRODUCTION.redecodes,
                ),
            },
            sizes: env_or(
                "PDF_RASTER_HUFFMAN_BENCH_SSB",
                "pick,128,256,512,1024,2048,4096",
            ),
        }
    }
}

/// One prepared input with its CPU-walk reference and timing.
struct Case {
    name: String,
    prep: JpegPreparedInput,
    coef: Vec<i32>,
    dc: Vec<i32>,
    cpu: Duration,
}

impl Case {
    /// Prints one result row; `passes` is the pass count or a tag.
    fn row(&self, ssb: &str, passes: &dyn std::fmt::Display, sym: Duration, gpu: Duration) {
        let n = self.prep.bitstream.length_bits.div_ceil(
            ssb.trim_start_matches('*')
                .parse()
                .expect("label carries the size"),
        );
        println!(
            "{:<28} {:>2} {:>8.1} {ssb:>6} {n:>6} {passes:>6} {:>9.2} {:>9.2} {:>9.2} {:>6.2}",
            self.name,
            self.prep.components.len(),
            f64::from(self.prep.bitstream.length_bits) / 8192.0,
            ms(sym),
            ms(gpu),
            ms(self.cpu),
            ms(gpu) / ms(self.cpu),
        );
    }

    /// Checks then times one subsequence size.
    fn bench_size(&self, backend: &CudaBackend, settings: &Settings, ssb: u32, label: &str) {
        let decode = || decode_jpeg_symbols_within(backend, &self.prep, ssb, settings.budget);
        let first = match decode() {
            Ok(d) => d,
            Err(e) if e.to_string().contains("did not converge") => {
                let fail = time_median(settings.iters, || decode().expect_err("budget exhausted"));
                self.row(label, &"fallbk", fail, fail + self.cpu);
                return;
            }
            Err(e) => panic!("{} ssb {ssb}: GPU decode failed: {e}", self.name),
        };
        let (coef, dc) = symbols_to_coefficients(&self.prep, &first.symbols).expect("reconstruct");
        assert!(
            coef == self.coef && dc == self.dc,
            "{} ssb {ssb}: GPU coefficients differ from the CPU walk",
            self.name
        );
        let sym = time_median(settings.iters, || decode().expect("gpu decode"));
        let gpu = time_median(settings.iters, || {
            let d = decode().expect("gpu decode");
            symbols_to_coefficients(&self.prep, &d.symbols).expect("reconstruct")
        });
        self.row(label, &first.phase2_passes, sym, gpu);
    }
}

/// Prepares `path` and times its CPU walk; `None` for inputs the GPU
/// decoder does not accept.
fn load_case(path: &Path, iters: usize) -> Option<Case> {
    let name = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    let bytes = std::fs::read(path).expect("read jpeg");
    let prep = match prepare_jpeg(&bytes) {
        Ok(p) if p.rst_marker_count == 0 => p,
        Ok(_) => {
            println!("{name:<28} skipped: restart markers");
            return None;
        }
        Err(e) => {
            println!("{name:<28} skipped: {e}");
            return None;
        }
    };
    let (coef, dc, _, _) = extract_coefficients_pub(&prep).expect("cpu walk");
    let cpu = time_median(iters, || extract_coefficients_pub(&prep).expect("cpu walk"));
    Some(Case {
        name,
        prep,
        coef,
        dc,
        cpu,
    })
}

#[test]
#[ignore = "benchmark: needs PDF_RASTER_HUFFMAN_BENCH_DIR, a CUDA device, and --release"]
fn huffman_bench() {
    let Ok(dir) = std::env::var("PDF_RASTER_HUFFMAN_BENCH_DIR") else {
        eprintln!("PDF_RASTER_HUFFMAN_BENCH_DIR unset; nothing to benchmark");
        return;
    };
    let Ok(backend) = CudaBackend::new() else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    let settings = Settings::from_env();
    println!(
        "{:<28} {:>2} {:>8} {:>6} {:>6} {:>6} {:>9} {:>9} {:>9} {:>6}",
        "file", "nc", "KiB", "ssb", "n", "passes", "sym_ms", "gpu_ms", "cpu_ms", "ratio"
    );
    for path in jpeg_files(Path::new(&dir)) {
        let Some(case) = load_case(&path, settings.iters) else {
            continue;
        };
        for size in settings.sizes.split(',').map(str::trim) {
            if size == "pick" {
                let ssb = pick_subsequence_size(&case.prep);
                case.bench_size(&backend, &settings, ssb, &format!("*{ssb}"));
            } else {
                let ssb = size.parse().expect("subsequence size must be an integer");
                case.bench_size(&backend, &settings, ssb, size);
            }
        }
    }
}
