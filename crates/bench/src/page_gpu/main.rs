//! Page-level GPU raster benchmark: a whole page rendered from a command
//! list in one Vulkan submission, against the CPU page render.
//!
//! The document is `--pages` Letter pages of `--stars` random solid stars
//! (see `stars.rs`).  The GPU page is: the CPU scan conversion of every fill
//! into a command list (`cmdlist.rs`), the host writes of that list into
//! mapped memory, then one submission that bins the commands into 16 × 16
//! tiles, rasterises and composites every tile in command order, and makes
//! the page readable by the host.  The CPU page is the library's render of
//! the same document (`render_page_rgb`), and the raster crate's fill of the
//! same paths without the PDF interpreter.
//!
//! Modes:
//! - `report` (default): device start-up, byte parity of every page, the
//!   per-page steps (medians over `--iters` runs after `--warmup`), with the
//!   device stages from timestamp queries, and the whole document with the
//!   CPU workers alone, the GPU alone, and the GPU beside the CPU workers.
//! - `cpu`, `gpu`, `both`: render the document once and exit, for timing
//!   the whole process from outside (`both` = `--threads` CPU workers plus
//!   the GPU thread).
//! - `--raster`: the fine-raster kernel, `rows` (default, each scan-table
//!   row walked once per tile) or `pixel` (walked once per pixel).
//! - `--write-pdf PATH`: write the document and exit.
//!
//! Run:
//! ```text
//! cargo build --release -p bench --features vulkan --bin page_gpu_bench
//! target/release/page_gpu_bench [--device NAME|INDEX] [--pages 5] [--stars 300] [--dpi 150]
//!     [--iters 9] [--warmup 2] [--threads N] [--mode report|cpu|gpu|both] [--raster rows|pixel]
//!     [--write-pdf PATH]
//! ```
//! The device defaults to `PDF_RASTER_VULKAN_DEVICE`, else the first
//! discrete GPU, else the first integrated one.

mod cmdlist;
mod stars;
mod vk;

use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use pdf_raster::{BackendPolicy, RasterSession, SessionConfig, open_session_from_bytes};

use cmdlist::{CmdList, packed_rows, render_cpu};
use stars::{Star, generate, page_px, write_pdf};
use vk::{Gpu, PageTimes};

struct Config {
    pages: usize,
    stars: usize,
    dpi: f64,
    seed: u64,
    iters: usize,
    warmup: usize,
    threads: usize,
    device: Option<String>,
    mode: String,
    raster_rows: bool,
    write_pdf: Option<String>,
}

fn parse_config() -> Config {
    let mut cfg = Config {
        pages: 5,
        stars: 300,
        dpi: 150.0,
        seed: 0x5EED_0005_7A25,
        iters: 9,
        warmup: 2,
        threads: std::thread::available_parallelism().map_or(1, std::num::NonZero::get),
        device: std::env::var("PDF_RASTER_VULKAN_DEVICE").ok(),
        mode: "report".to_owned(),
        raster_rows: true,
        write_pdf: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next();
        match (flag.as_str(), value) {
            ("--pages", Some(v)) => cfg.pages = v.parse().expect("--pages N"),
            ("--stars", Some(v)) => cfg.stars = v.parse().expect("--stars N"),
            ("--dpi", Some(v)) => cfg.dpi = v.parse().expect("--dpi N"),
            ("--seed", Some(v)) => cfg.seed = v.parse().expect("--seed N"),
            ("--iters", Some(v)) => cfg.iters = v.parse().expect("--iters N"),
            ("--warmup", Some(v)) => cfg.warmup = v.parse().expect("--warmup N"),
            ("--threads", Some(v)) => cfg.threads = v.parse().expect("--threads N"),
            ("--device", Some(v)) => cfg.device = Some(v),
            ("--mode", Some(v)) => cfg.mode = v,
            ("--write-pdf", Some(v)) => cfg.write_pdf = Some(v),
            ("--raster", Some(v)) => {
                cfg.raster_rows = match v.as_str() {
                    "rows" => true,
                    "pixel" => false,
                    _ => panic!("--raster rows|pixel"),
                };
            }
            (f, _) => panic!("unknown or incomplete flag {f}"),
        }
    }
    assert!(
        cfg.iters > 0 && cfg.pages > 0,
        "--iters and --pages must be >= 1"
    );
    cfg
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "runs take well under 584 years"
)]
fn ns(t: Instant) -> u64 {
    t.elapsed().as_nanos() as u64
}

#[expect(clippy::cast_precision_loss, reason = "display only")]
fn ms(ns: u64) -> String {
    format!("{:.2}", ns as f64 / 1e6)
}

/// Median, minimum and maximum.
fn stats(mut v: Vec<u64>) -> (u64, u64, u64) {
    v.sort_unstable();
    (v[v.len() / 2], v[0], v[v.len() - 1])
}

fn fmt_stats(v: Vec<u64>) -> String {
    let (med, lo, hi) = stats(v);
    format!("{:>9} ms  [{} – {}]", ms(med), ms(lo), ms(hi))
}

struct Doc {
    stars: Vec<Vec<Star>>,
    pdf: Vec<u8>,
    scale: f64,
    width: u32,
    height: u32,
}

fn open_cpu(doc: &Doc) -> RasterSession {
    open_session_from_bytes(
        doc.pdf.clone(),
        &SessionConfig::with_policy(BackendPolicy::CpuOnly),
    )
    .expect("open the star document")
}

fn render_lib(session: &RasterSession, page: usize, scale: f64) -> raster::Bitmap<color::Rgb8> {
    let n = u32::try_from(page + 1).expect("few pages");
    pdf_raster::render_page_rgb(session, n, scale).expect("CPU page render")
}

/// Render every page once: `cpu_workers` threads through the library and,
/// with `gpu`, the calling thread through the GPU, all taking pages from
/// one queue, the GPU first.  Returns the wall time and the number of pages the GPU took.
fn run_doc(
    doc: &Doc,
    session: Option<&RasterSession>,
    gpu: Option<&mut Gpu>,
    cpu_workers: usize,
) -> (u64, usize) {
    let next = AtomicUsize::new(0);
    let n = doc.stars.len();
    let start = Instant::now();
    let mut gpu_pages = 0;
    // The GPU, idle at the start, takes the first page before any worker.
    let first = if gpu.is_some() {
        next.fetch_add(1, Ordering::Relaxed)
    } else {
        n
    };
    std::thread::scope(|s| {
        for _ in 0..cpu_workers {
            let session = session.expect("CPU workers need a session");
            let next = &next;
            s.spawn(move || {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    black_box(render_lib(session, i, doc.scale));
                }
            });
        }
        if let Some(gpu) = gpu {
            let mut cl = CmdList::default();
            let mut out = vec![0u8; doc.width as usize * doc.height as usize * 3];
            let mut i = first;
            while i < n {
                cl.build(&doc.stars[i], doc.scale, doc.width, doc.height);
                gpu.render(&cl, &mut out).expect("GPU page");
                black_box(&out);
                gpu_pages += 1;
                i = next.fetch_add(1, Ordering::Relaxed);
            }
        }
    });
    (ns(start), gpu_pages)
}

fn compare(name: &str, page: usize, got: &[u8], want: &[u8], width: u32) -> bool {
    assert_eq!(got.len(), want.len(), "{name}: page sizes differ");
    let diffs: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
    if diffs.is_empty() {
        return true;
    }
    let max = diffs
        .iter()
        .map(|&i| got[i].abs_diff(want[i]))
        .max()
        .unwrap_or(0);
    let px = diffs[0] / 3;
    println!(
        "  {name} page {}: {} bytes differ (max |Δ| {max}); first at pixel ({}, {})",
        page + 1,
        diffs.len(),
        px % width as usize,
        px / width as usize
    );
    false
}

fn main() {
    let process_start = Instant::now();
    let cfg = parse_config();
    let scale = cfg.dpi / 72.0;
    let (width, height) = page_px(scale);
    let stars = generate(cfg.pages, cfg.stars, cfg.seed);
    let pdf = write_pdf(&stars);
    if let Some(path) = &cfg.write_pdf {
        std::fs::write(path, &pdf).expect("write the PDF");
        println!("wrote {path} ({} pages, {} bytes)", cfg.pages, pdf.len());
        return;
    }
    let doc = Doc {
        stars,
        pdf,
        scale,
        width,
        height,
    };
    match cfg.mode.as_str() {
        "report" => report(&cfg, &doc),
        "cpu" | "gpu" | "both" => {
            let session = (cfg.mode != "gpu").then(|| open_cpu(&doc));
            let mut gpu = (cfg.mode != "cpu").then(|| {
                let mut g = Gpu::new(cfg.device.as_deref()).expect("Vulkan device");
                g.raster_rows = cfg.raster_rows;
                g
            });
            let workers = if cfg.mode == "gpu" { 0 } else { cfg.threads };
            let (wall, gpu_pages) = run_doc(&doc, session.as_ref(), gpu.as_mut(), workers);
            println!(
                "{} pages, mode {}, {workers} CPU workers: render {} ms, GPU took {gpu_pages}; \
                 process so far {} ms",
                cfg.pages,
                cfg.mode,
                ms(wall),
                ms(ns(process_start))
            );
        }
        m => panic!("unknown --mode {m}"),
    }
}

#[expect(
    clippy::too_many_lines,
    clippy::many_single_char_names,
    reason = "one linear report; t, w, h, a, b, s are a timer, page extents and samples"
)]
fn report(cfg: &Config, doc: &Doc) {
    let t = Instant::now();
    let mut gpu = Gpu::new(cfg.device.as_deref()).expect("Vulkan device");
    let init = ns(t);
    gpu.raster_rows = cfg.raster_rows;
    println!(
        "device: {} ({}), page {}x{} at {} dpi, {} pages x {} stars, {} threads",
        gpu.name,
        if gpu.integrated {
            "integrated"
        } else {
            "discrete"
        },
        doc.width,
        doc.height,
        cfg.dpi,
        cfg.pages,
        cfg.stars,
        cfg.threads
    );
    println!(
        "page read by the host {}",
        if gpu.page_in_place {
            "in place"
        } else {
            "after a device copy"
        }
    );
    let [load, instance, device, rest] = gpu.startup;
    println!(
        "Vulkan start-up {} ms: loader {} + instance {} + device {} + pipelines and objects {}",
        ms(init),
        ms(load),
        ms(instance),
        ms(device),
        ms(rest)
    );

    let t = Instant::now();
    let session = open_cpu(doc);
    println!("library session open: {} ms", ms(ns(t)));

    // Parity, and the first GPU page (buffer creation included).
    let (w, h) = (doc.width, doc.height);
    let mut cl = CmdList::default();
    let mut out = vec![0u8; w as usize * h as usize * 3];
    let mut identical = true;
    let mut first_gpu_page = None;
    let mut raster_kernel = [Vec::new(), Vec::new()];
    for (i, page) in doc.stars.iter().enumerate() {
        let lib = packed_rows(&render_lib(&session, i, doc.scale));
        let direct = packed_rows(&render_cpu(page, doc.scale, w, h));
        let t = Instant::now();
        cl.build(page, doc.scale, w, h);
        gpu.render(&cl, &mut out).expect("GPU page");
        first_gpu_page.get_or_insert(ns(t));
        identical &= compare("raster fill vs library", i, &direct, &lib, w);
        for rows in [false, true] {
            gpu.raster_rows = rows;
            out.fill(0);
            let s = gpu.render(&cl, &mut out).expect("GPU page");
            raster_kernel[usize::from(rows)].push(s.gpu_raster);
            let name = if rows {
                "GPU (rows) vs library"
            } else {
                "GPU (pixel) vs library"
            };
            identical &= compare(name, i, &out, &lib, w);
        }
        gpu.raster_rows = cfg.raster_rows;
    }
    println!(
        "first GPU page (buffer creation included): {} ms",
        ms(first_gpu_page.unwrap_or(0))
    );
    println!(
        "byte parity: {}",
        if identical {
            "every page identical (GPU = raster fill = library render)"
        } else {
            "DIFFERENT, see above"
        }
    );
    let [pixel_k, rows_k] = raster_kernel;
    println!(
        "device fine raster per page: pixel kernel {} ms, rows kernel {} ms (median); timed below: {}",
        ms(stats(pixel_k).0),
        ms(stats(rows_k).0),
        if cfg.raster_rows { "rows" } else { "pixel" }
    );
    println!(
        "command list: {} commands, {:.1} MB, {} tile references (last page)",
        cl.cmds.len(),
        f64::from(u32::try_from(cl.bytes()).unwrap_or(u32::MAX)) / 1e6,
        cl.tile_refs
    );

    // Per-page steps, one thread.
    let mut lib_t = Vec::new();
    let mut direct_t = Vec::new();
    let mut prep_t = Vec::new();
    let mut gpu_total = Vec::new();
    let mut steps: Vec<PageTimes> = Vec::new();
    for iter in 0..cfg.warmup + cfg.iters {
        let keep = iter >= cfg.warmup;
        for (i, page) in doc.stars.iter().enumerate() {
            let t = Instant::now();
            black_box(render_lib(&session, i, doc.scale));
            let a = ns(t);
            let t = Instant::now();
            black_box(render_cpu(page, doc.scale, w, h));
            let b = ns(t);
            let t = Instant::now();
            cl.build(page, doc.scale, w, h);
            let prep = ns(t);
            let s = gpu.render(&cl, &mut out).expect("GPU page");
            let total = ns(t);
            if keep {
                lib_t.push(a);
                direct_t.push(b);
                prep_t.push(prep);
                gpu_total.push(total);
                steps.push(s);
            }
        }
    }
    let pick = |f: fn(&PageTimes) -> u64| fmt_stats(steps.iter().map(f).collect());
    println!(
        "\nper page, one thread (median [min – max] of {} samples)",
        lib_t.len()
    );
    println!("  CPU library render_page_rgb      {}", fmt_stats(lib_t));
    println!("  CPU raster fill, no interpreter  {}", fmt_stats(direct_t));
    println!(
        "  GPU page, total                  {}",
        fmt_stats(gpu_total)
    );
    println!("    CPU scan conversion + list     {}", fmt_stats(prep_t));
    println!("    host writes of the list        {}", pick(|s| s.upload));
    println!("    record + submit                {}", pick(|s| s.submit));
    println!("    wait (submit to fence)         {}", pick(|s| s.wait));
    println!("    host copy of the page          {}", pick(|s| s.read));
    println!("    device: binning                {}", pick(|s| s.gpu_bin));
    println!(
        "    device: fine raster            {}",
        pick(|s| s.gpu_raster)
    );
    println!(
        "    device: page copy              {}",
        pick(|s| s.gpu_copy)
    );

    // Whole document.
    let threads = cfg.threads.max(1);
    let mut runs: Vec<(String, Vec<u64>, usize)> = vec![
        ("CPU, 1 thread".to_owned(), Vec::new(), 0),
        (format!("CPU, {threads} threads"), Vec::new(), 0),
        ("GPU alone (1 driving thread)".to_owned(), Vec::new(), 0),
        (format!("GPU + {threads} CPU workers"), Vec::new(), 0),
        (
            format!("GPU + {} CPU workers", threads.saturating_sub(1)),
            Vec::new(),
            0,
        ),
    ];
    for iter in 0..cfg.warmup + cfg.iters {
        let results = [
            run_doc(doc, Some(&session), None, 1),
            run_doc(doc, Some(&session), None, threads),
            run_doc(doc, None, Some(&mut gpu), 0),
            run_doc(doc, Some(&session), Some(&mut gpu), threads),
            run_doc(
                doc,
                Some(&session),
                Some(&mut gpu),
                threads.saturating_sub(1),
            ),
        ];
        if iter >= cfg.warmup {
            for (run, (wall, gp)) in runs.iter_mut().zip(results) {
                run.1.push(wall);
                run.2 += gp;
            }
        }
    }
    println!(
        "\nwhole document, {} pages, in process (median [min – max] of {} runs)",
        cfg.pages, cfg.iters
    );
    for (name, walls, gp) in runs {
        #[expect(clippy::cast_precision_loss, reason = "display only")]
        let share = gp as f64 / cfg.iters as f64;
        println!(
            "  {name:<30} {}  GPU pages/run {share:.1}",
            fmt_stats(walls)
        );
    }
}
