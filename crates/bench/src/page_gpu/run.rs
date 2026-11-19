//! Page-level GPU raster benchmark: a whole page rendered from its paint
//! commands in one submission on a Vulkan or Direct3D 12 device, against the
//! CPU page render.
//!
//! The document is `--pages` Letter pages of `--stars` random solid stars
//! (see `stars.rs`).  The GPU page is the library's GPU page path in two
//! steps: the recording of the page's paint commands by the interpreter
//! (`pdf_raster::dev::device_page_commands`, the CPU's exact-area coverage
//! of every fill), then `PageDevice::render_page_timed`, which reports
//! the duration of each of its host and device stages.  The CPU page is the
//! library's render of the same page (`render_page_rgb`).
//!
//! Modes:
//! - `report` (default): device start-up, byte parity of every page, the
//!   per-page stages (medians over `--iters` runs after `--warmup`), and the
//!   whole document with the CPU workers alone, the GPU alone, and the GPU
//!   beside the CPU workers.
//! - `page`: `report` without the whole-document runs: start-up, byte
//!   parity, the per-page stages and the read-back bandwidth.
//! - `cpu`, `gpu`, `both`: render the document once and exit, for timing
//!   the whole process from outside (`both` = `--threads` CPU workers plus
//!   the GPU thread).
//! - `--write-pdf PATH`: write the document and exit.
//! - `--pdf PATH`: the first `--pages` pages of the PDF at `PATH` instead
//!   of the star document; every page must be recordable.
//! - `--first N`: the pages start at page `N` (1-based) of the document,
//!   so `--first 7 --pages 1` times page 7 alone.
//!
//! The report prints, per page, the device raster time, the device and host
//! copies of the page and the bytes copied, with the bandwidth of each copy
//! (the page bytes over its median time).
//!
//! Run:
//! ```text
//! cargo build --release -p bench --features vulkan --bin page_gpu_bench
//! cargo build --release -p bench --features d3d12 --bin page_gpu_bench   # Windows
//! target/release/page_gpu_bench [--api vulkan|d3d12] [--device NAME|INDEX] [--pages 5]
//!     [--first 1] [--stars 300] [--dpi 150] [--iters 9] [--warmup 2] [--threads N]
//!     [--mode report|page|cpu|gpu|both] [--write-pdf PATH] [--pdf PATH]
//! ```
//! `--api` picks the backend (default `vulkan`), at run time, so one binary
//! built with both features serves both; a name whose feature is not built
//! (`d3d12` off Windows) is an error.  The device defaults to
//! `PDF_RASTER_VULKAN_DEVICE` or `PDF_RASTER_D3D12_DEVICE` by API, else the
//! device the backend ranks first.

use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use gpu::backend::page_device::{PageDevice, PageTimes};
use gpu::backend::select::DeviceSelector;
use gpu::page::PageCommands;
use pdf_raster::{BackendPolicy, RasterSession, RenderParams, SessionConfig};

use crate::stars::{generate, write_pdf};

struct Config {
    api: String,
    pages: usize,
    first: usize,
    stars: usize,
    dpi: f64,
    seed: u64,
    iters: usize,
    warmup: usize,
    threads: usize,
    device: Option<String>,
    mode: String,
    write_pdf: Option<String>,
    pdf: Option<String>,
}

fn parse_config() -> Config {
    let mut cfg = Config {
        api: "vulkan".to_owned(),
        pages: 5,
        first: 1,
        stars: 300,
        dpi: 150.0,
        seed: 0x5EED_0005_7A25,
        iters: 9,
        warmup: 2,
        threads: std::thread::available_parallelism().map_or(1, std::num::NonZero::get),
        device: None,
        mode: "report".to_owned(),
        write_pdf: None,
        pdf: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next();
        match (flag.as_str(), value) {
            ("--api", Some(v)) => cfg.api = v,
            ("--first", Some(v)) => cfg.first = v.parse().expect("--first N"),
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
            ("--pdf", Some(v)) => cfg.pdf = Some(v),
            (f, _) => panic!("unknown or incomplete flag {f}"),
        }
    }
    assert!(
        cfg.iters > 0 && cfg.pages > 0 && cfg.first > 0,
        "--iters, --pages and --first must be >= 1"
    );
    cfg
}

impl Config {
    /// The 0-based indices of the pages to render.
    const fn range(&self) -> std::ops::Range<usize> {
        self.first - 1..self.first - 1 + self.pages
    }
}

/// A page-raster backend the benchmark can drive.
trait Api: PageDevice + Sync {
    /// The environment variable naming the default device.
    const DEVICE_ENV: &'static str;
}

#[cfg(feature = "vulkan")]
impl Api for gpu::backend::vulkan::VulkanBackend {
    const DEVICE_ENV: &'static str = "PDF_RASTER_VULKAN_DEVICE";
}

#[cfg(all(windows, feature = "d3d12"))]
impl Api for gpu::backend::d3d12::D3d12Backend {
    const DEVICE_ENV: &'static str = "PDF_RASTER_D3D12_DEVICE";
}

/// Run the benchmark on the backend `--api` names.
pub fn main() {
    let cfg = parse_config();
    match cfg.api.as_str() {
        #[cfg(feature = "vulkan")]
        "vulkan" => run::<gpu::backend::vulkan::VulkanBackend>(&cfg),
        #[cfg(all(windows, feature = "d3d12"))]
        "d3d12" => run::<gpu::backend::d3d12::D3d12Backend>(&cfg),
        a => panic!("--api {a}: not vulkan or d3d12, or not built into this binary"),
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "runs take well under 584 years"
)]
fn ns(d: Duration) -> u64 {
    d.as_nanos() as u64
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

fn pick_ns(steps: &[PageTimes], f: fn(&PageTimes) -> Duration) -> u64 {
    stats(steps.iter().map(|s| ns(f(s))).collect()).0
}

#[expect(clippy::cast_precision_loss, reason = "display only")]
fn gb_per_s(bytes: usize, ns: u64) -> f64 {
    if ns == 0 {
        0.0
    } else {
        bytes as f64 / ns as f64
    }
}

fn fmt_stats(v: Vec<u64>) -> String {
    let (med, lo, hi) = stats(v);
    format!("{:>9} ms  [{} – {}]", ms(med), ms(lo), ms(hi))
}

fn open_device<B: Api>(selector: Option<&str>) -> B {
    let env = std::env::var(B::DEVICE_ENV).ok();
    let selector = selector
        .or(env.as_deref())
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.parse::<DeviceSelector>().expect("--device selector"));
    B::open(selector.as_ref()).expect("GPU device")
}

fn open_cpu(pdf: &[u8]) -> RasterSession {
    pdf_raster::open_session_from_bytes(
        pdf.to_vec(),
        &SessionConfig::with_policy(BackendPolicy::CpuOnly),
    )
    .expect("open the star document")
}

fn page_num(page: usize) -> u32 {
    u32::try_from(page + 1).expect("few pages")
}

fn render_lib(session: &RasterSession, page: usize, scale: f64) -> raster::Bitmap<color::Rgb8> {
    pdf_raster::render_page_rgb(session, page_num(page), scale).expect("CPU page render")
}

/// The paint commands the library's GPU page path records for `page`.
fn record(session: &RasterSession, page: usize, scale: f64) -> PageCommands {
    pdf_raster::dev::device_page_commands(session, page_num(page), &RenderParams::new(scale))
        .expect("record the page")
        .unwrap_or_else(|reason| panic!("page {} is not recordable: {reason}", page + 1))
}

/// `page` rendered on `gpu` into `out`, packed rows.
fn render_gpu<B: Api>(gpu: &B, page: &PageCommands, out: &mut Vec<u8>) -> PageTimes {
    let row = page.width() as usize * 3;
    out.resize(row * page.height() as usize, 0);
    gpu.render_page_timed(page, out, row).expect("GPU page")
}

/// Render every page once: `cpu_workers` threads through the library and,
/// with `gpu`, the calling thread through the GPU, all taking pages from
/// one queue, the GPU first.  Returns the wall time and the number of pages
/// the GPU took.
fn run_doc<B: Api>(
    session: &RasterSession,
    pages: std::ops::Range<usize>,
    scale: f64,
    gpu: Option<&B>,
    cpu_workers: usize,
) -> (u64, usize) {
    let (n, next) = (pages.end, AtomicUsize::new(pages.start));
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
            let next = &next;
            s.spawn(move || {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    black_box(render_lib(session, i, scale));
                }
            });
        }
        if let Some(gpu) = gpu {
            let mut out = Vec::new();
            let mut i = first;
            while i < n {
                let page = record(session, i, scale);
                black_box(render_gpu(gpu, &page, &mut out));
                gpu_pages += 1;
                i = next.fetch_add(1, Ordering::Relaxed);
            }
        }
    });
    (ns(start.elapsed()), gpu_pages)
}

/// The bitmap's pixels without row padding.
fn packed_rows(bmp: &raster::Bitmap<color::Rgb8>) -> Vec<u8> {
    let row = bmp.width as usize * 3;
    bmp.data()
        .chunks(bmp.stride)
        .take(bmp.height as usize)
        .flat_map(|r| &r[..row])
        .copied()
        .collect()
}

fn compare(page: usize, got: &[u8], want: &[u8], width: u32) -> bool {
    assert_eq!(got.len(), want.len(), "page sizes differ");
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
        "  page {}: {} bytes differ (max |Δ| {max}); first at pixel ({}, {})",
        page + 1,
        diffs.len(),
        px % width as usize,
        px / width as usize
    );
    false
}

fn run<B: Api>(cfg: &Config) {
    let process_start = Instant::now();
    assert!(
        cfg.pdf.is_some() || cfg.first == 1,
        "--first needs --pdf: the star document is generated from page 1"
    );
    let scale = cfg.dpi / 72.0;
    let pdf = cfg.pdf.as_ref().map_or_else(
        || write_pdf(&generate(cfg.pages, cfg.stars, cfg.seed)),
        |path| std::fs::read(path).expect("read --pdf"),
    );
    if let Some(path) = &cfg.write_pdf {
        std::fs::write(path, &pdf).expect("write the PDF");
        println!("wrote {path} ({} pages, {} bytes)", cfg.pages, pdf.len());
        return;
    }
    match cfg.mode.as_str() {
        "report" | "page" => report::<B>(cfg, &pdf, scale),
        "cpu" | "gpu" | "both" => {
            let session = open_cpu(&pdf);
            let gpu = (cfg.mode != "cpu").then(|| open_device::<B>(cfg.device.as_deref()));
            let workers = if cfg.mode == "gpu" { 0 } else { cfg.threads };
            let (wall, gpu_pages) = run_doc(&session, cfg.range(), scale, gpu.as_ref(), workers);
            println!(
                "{} pages, mode {}, {workers} CPU workers: render {} ms, GPU took {gpu_pages}; \
                 process so far {} ms",
                cfg.pages,
                cfg.mode,
                ms(wall),
                ms(ns(process_start.elapsed()))
            );
        }
        m => panic!("unknown --mode {m}"),
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "one linear report: start-up, parity, per-page stages, whole document"
)]
fn report<B: Api>(cfg: &Config, pdf: &[u8], scale: f64) {
    let t = Instant::now();
    let gpu = open_device::<B>(cfg.device.as_deref());
    let init = ns(t.elapsed());
    let t = Instant::now();
    let session = open_cpu(pdf);
    let session_open = ns(t.elapsed());

    // Parity, and the first GPU page (pipeline and buffer creation
    // included).
    let mut out = Vec::new();
    let mut identical = true;
    let mut first_gpu_page = None;
    let mut last = None;
    for i in cfg.range() {
        let lib = render_lib(&session, i, scale);
        let t = Instant::now();
        let page = record(&session, i, scale);
        render_gpu(&gpu, &page, &mut out);
        first_gpu_page.get_or_insert(ns(t.elapsed()));
        identical &= compare(i, &out, &packed_rows(&lib), lib.width);
        last = Some(page);
    }
    let last = last.expect("at least one page");
    println!(
        "device: {} ({} backend), page {}x{} at {} dpi, {} pages x {} stars, {} threads",
        gpu.device_name(),
        cfg.api,
        last.width(),
        last.height(),
        cfg.dpi,
        cfg.pages,
        cfg.stars,
        cfg.threads
    );
    println!(
        "{} start-up {} ms; library session open {} ms",
        cfg.api,
        ms(init),
        ms(session_open)
    );
    println!(
        "first GPU page (pipeline and buffer creation included): {} ms",
        ms(first_gpu_page.unwrap_or(0))
    );
    println!(
        "byte parity: {}",
        if identical {
            "every page identical (GPU page = library render)"
        } else {
            "DIFFERENT, see above"
        }
    );
    println!(
        "command list: {} commands, {:.1} MB of device inputs (last page)",
        last.len(),
        f64::from(u32::try_from(last.bytes()).unwrap_or(u32::MAX)) / 1e6,
    );

    // Per-page stages, one thread.
    let mut lib_t = Vec::new();
    let mut record_t = Vec::new();
    let mut gpu_total = Vec::new();
    let mut steps: Vec<PageTimes> = Vec::new();
    for iter in 0..cfg.warmup + cfg.iters {
        let keep = iter >= cfg.warmup;
        for i in cfg.range() {
            let t = Instant::now();
            black_box(render_lib(&session, i, scale));
            let lib = ns(t.elapsed());
            let t = Instant::now();
            let page = record(&session, i, scale);
            let recorded = ns(t.elapsed());
            let s = render_gpu(&gpu, &page, &mut out);
            let total = ns(t.elapsed());
            if keep {
                lib_t.push(lib);
                record_t.push(recorded);
                gpu_total.push(total);
                steps.push(s);
            }
        }
    }
    let pick = |f: fn(&PageTimes) -> Duration| fmt_stats(steps.iter().map(|s| ns(f(s))).collect());
    println!(
        "\nper page, one thread (median [min – max] of {} samples)",
        lib_t.len()
    );
    println!("  CPU library render_page_rgb      {}", fmt_stats(lib_t));
    println!(
        "  GPU page, total                  {}",
        fmt_stats(gpu_total)
    );
    println!(
        "    recording (interpreter + coverage) {}",
        fmt_stats(record_t)
    );
    println!("    host: binning + table packing  {}", pick(|s| s.prepare));
    println!("    host: input writes             {}", pick(|s| s.write));
    println!("    host: begin + record           {}", pick(|s| s.record));
    println!(
        "    host: submit + wait            {}",
        pick(|s| s.submit_wait)
    );
    println!("    host: copy of the page         {}", pick(|s| s.read));
    println!(
        "    device: input copies           {}",
        pick(|s| s.device_upload)
    );
    println!(
        "    device: raster                 {}",
        pick(|s| s.device_raster)
    );
    println!(
        "    device: page copy              {}",
        pick(|s| s.device_download)
    );
    let page_bytes = out.len();
    println!(
        "\nread-back of {page_bytes} page bytes ({}x{} RGB8)",
        last.width(),
        last.height()
    );
    for (name, times) in [
        ("device page copy", pick_ns(&steps, |s| s.device_download)),
        ("host copy        ", pick_ns(&steps, |s| s.read)),
    ] {
        println!(
            "  {name} median {} ms  {:.2} GB/s",
            ms(times),
            gb_per_s(page_bytes, times)
        );
    }

    if cfg.mode == "page" {
        return;
    }

    // Whole document.
    let threads = cfg.threads.max(1);
    let n = cfg.range();
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
            run_doc::<B>(&session, n.clone(), scale, None, 1),
            run_doc::<B>(&session, n.clone(), scale, None, threads),
            run_doc(&session, n.clone(), scale, Some(&gpu), 0),
            run_doc(&session, n.clone(), scale, Some(&gpu), threads),
            run_doc(
                &session,
                n.clone(),
                scale,
                Some(&gpu),
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
