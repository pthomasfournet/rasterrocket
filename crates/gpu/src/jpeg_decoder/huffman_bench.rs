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
//! reported) and `PDF_RASTER_HUFFMAN_BENCH_MAX_PASSES` (Phase 2
//! dispatch budget, default the production `JPEG_PHASE2_MAX_PASSES`).
//!
//! Every GPU configuration is checked coefficient-for-coefficient
//! against the CPU walk before it is timed, so a reported time is
//! always the time of a correct decode. `sym_ms` is the GPU symbol
//! decode alone; `gpu_ms` adds the CPU magnitude-bit pass that turns
//! symbols into coefficients — the full replacement for `cpu_ms`. A
//! configuration that exhausts the pass budget is reported as
//! `fallback`: `sym_ms` is the time spent before the GPU decode gave
//! up and `gpu_ms` that time plus `cpu_ms`, the cost the decoder's CPU
//! fallback pays.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::backend::cuda::CudaBackend;
use crate::jpeg_decoder::cpu_prepass::prepare_jpeg;
use crate::jpeg_decoder::decoder::{extract_coefficients_pub, symbols_to_coefficients};
use crate::jpeg_decoder::huffman::{JPEG_PHASE2_MAX_PASSES, decode_jpeg_symbols_within};
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
    let iters: usize = env_or("PDF_RASTER_HUFFMAN_BENCH_ITERS", "7")
        .parse()
        .expect("PDF_RASTER_HUFFMAN_BENCH_ITERS must be an integer");
    let max_passes: u32 = env_or(
        "PDF_RASTER_HUFFMAN_BENCH_MAX_PASSES",
        &JPEG_PHASE2_MAX_PASSES.to_string(),
    )
    .parse()
    .expect("PDF_RASTER_HUFFMAN_BENCH_MAX_PASSES must be an integer");
    let sizes = env_or(
        "PDF_RASTER_HUFFMAN_BENCH_SSB",
        "pick,128,256,512,1024,2048,4096",
    );

    println!(
        "{:<28} {:>2} {:>8} {:>6} {:>6} {:>6} {:>9} {:>9} {:>9} {:>6}",
        "file", "nc", "KiB", "ssb", "n", "passes", "sym_ms", "gpu_ms", "cpu_ms", "ratio"
    );
    for path in jpeg_files(Path::new(&dir)) {
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        let bytes = std::fs::read(&path).expect("read jpeg");
        let prep = match prepare_jpeg(&bytes) {
            Ok(p) if p.rst_marker_count == 0 => p,
            Ok(_) => {
                println!("{name:<28} skipped: restart markers");
                continue;
            }
            Err(e) => {
                println!("{name:<28} skipped: {e}");
                continue;
            }
        };
        let (cpu_coef, cpu_dc, _, _) = extract_coefficients_pub(&prep).expect("cpu walk");
        let cpu = time_median(iters, || extract_coefficients_pub(&prep).expect("cpu walk"));
        let kib = f64::from(prep.bitstream.length_bits) / 8192.0;

        for size in sizes.split(',').map(str::trim) {
            let ssb = if size == "pick" {
                pick_subsequence_size(&prep)
            } else {
                size.parse().expect("subsequence size must be an integer")
            };
            let label = if size == "pick" {
                format!("*{ssb}")
            } else {
                ssb.to_string()
            };
            let n = prep.bitstream.length_bits.div_ceil(ssb);
            let decode = || decode_jpeg_symbols_within(&backend, &prep, ssb, max_passes);
            let first = match decode() {
                Ok(d) => d,
                Err(e) if e.to_string().contains("did not converge") => {
                    let fail = time_median(iters, || decode().expect_err("budget exhausted"));
                    println!(
                        "{name:<28} {:>2} {kib:>8.1} {label:>6} {n:>6} {:>6} {:>9.2} {:>9.2} {:>9.2} {:>6.2}",
                        prep.components.len(),
                        "fallbk",
                        ms(fail),
                        ms(fail + cpu),
                        ms(cpu),
                        ms(fail + cpu) / ms(cpu),
                    );
                    continue;
                }
                Err(e) => panic!("{name} ssb {ssb}: GPU decode failed: {e}"),
            };
            let (coef, dc) = symbols_to_coefficients(&prep, &first.symbols).expect("reconstruct");
            assert!(
                coef == cpu_coef && dc == cpu_dc,
                "{name} ssb {ssb}: GPU coefficients differ from the CPU walk"
            );
            let sym = time_median(iters, || decode().expect("gpu decode"));
            let gpu = time_median(iters, || {
                let d = decode().expect("gpu decode");
                symbols_to_coefficients(&prep, &d.symbols).expect("reconstruct")
            });
            println!(
                "{name:<28} {:>2} {kib:>8.1} {label:>6} {n:>6} {:>6} {:>9.2} {:>9.2} {:>9.2} {:>6.2}",
                prep.components.len(),
                first.phase2_passes,
                ms(sym),
                ms(gpu),
                ms(cpu),
                ms(gpu) / ms(cpu),
            );
        }
    }
}
