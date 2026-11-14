//! Per-phase timing of the Vulkan backend's kernel pages.
//!
//! Each kernel is run as its production sequence with the steps timed
//! apart: host preparation (writing the inputs where the device reads
//! them), recording (`begin_page` and the `record_*` calls), `submit_page`,
//! `wait_page`, and the host read of the result.  Two derived figures
//! split the wait: the copies-only page (the page's transfers with no
//! dispatch) and the kernel's device time, the per-dispatch slope of a
//! page that records the dispatch [`REPEATS`] times against one that
//! records it once.  Every figure is the median of `--iters` runs after
//! `--warmup` discarded ones.
//!
//! Run:
//! ```text
//! cargo build --release -p rasterrocket-gpu --features vulkan,cache --bin vulkan_phase_bench
//! target/release/vulkan_phase_bench [--device NAME] [--iters N] [--warmup N] [--only icc|blit|lock|readback]
//! ```
#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    reason = "display-only timing conversions, elapsed nanoseconds fit u64, sizes are \
              positive and below 2^32; c/m/y/k/w/h name channels and extents"
)]

#[path = "bench_common/mod.rs"]
mod bench_common;

use std::fmt::Write as _;
use std::time::Instant;

use bench_common::smooth_cmyk;
use rasterrocket_gpu::backend::params::{BLIT_FILTER_X, BlitParams, IccClutParams};
use rasterrocket_gpu::backend::vulkan::{DeviceSelector, VulkanBackend};
use rasterrocket_gpu::backend::{BlitScratch, BlitSource, GpuBackend, blit_image_box, icc_clut};
use rasterrocket_gpu::icc_cmyk_to_rgb_cpu;

/// Dispatches of the long page the kernel's device time is derived from.
const REPEATS: u32 = 5;

struct Config {
    iters: usize,
    warmup: usize,
    device: Option<String>,
    only: Option<String>,
}

/// Command-line usage.
const USAGE: &str = "vulkan_phase_bench [--device NAME] [--iters N] [--warmup N] \
                     [--only icc|blit|lock|readback]";

fn parse_config() -> Config {
    let mut cfg = Config {
        iters: 9,
        warmup: 3,
        device: None,
        only: None,
    };
    let known = ["--iters", "--warmup", "--device", "--only"];
    for (flag, value) in bench_common::flag_pairs(USAGE, &known) {
        match flag.as_str() {
            "--iters" => cfg.iters = bench_common::parse_value(USAGE, &flag, &value),
            "--warmup" => cfg.warmup = bench_common::parse_value(USAGE, &flag, &value),
            "--device" => cfg.device = Some(value),
            _ => {
                cfg.only = Some(bench_common::only_value(
                    USAGE,
                    value,
                    &["icc", "blit", "lock", "readback"],
                ));
            }
        }
    }
    cfg
}

fn ok<T>(r: rasterrocket_gpu::backend::Result<T>, ctx: &str) -> T {
    r.unwrap_or_else(|e| panic!("{ctx}: {e}"))
}

fn median(mut v: Vec<u64>) -> u64 {
    v.sort_unstable();
    v[v.len() / 2]
}

fn ns(t: Instant) -> u64 {
    t.elapsed().as_nanos() as u64
}

fn us(ns: u64) -> String {
    format!("{:.1}", ns as f64 / 1e3)
}

/// The timed steps of one page run.
#[derive(Default, Clone, Copy)]
struct Steps {
    prep: u64,
    record: u64,
    submit: u64,
    wait: u64,
    read: u64,
}

impl Steps {
    const fn total(self) -> u64 {
        self.prep + self.record + self.submit + self.wait + self.read
    }
}

/// Medians of each step over `cfg.iters` runs of `run`.
fn median_steps(cfg: &Config, mut run: impl FnMut() -> Steps) -> Steps {
    for _ in 0..cfg.warmup {
        let _ = run();
    }
    let runs: Vec<Steps> = (0..cfg.iters).map(|_| run()).collect();
    let pick = |f: fn(&Steps) -> u64| median(runs.iter().map(f).collect());
    Steps {
        prep: pick(|s| s.prep),
        record: pick(|s| s.record),
        submit: pick(|s| s.submit),
        wait: pick(|s| s.wait),
        read: pick(|s| s.read),
    }
}

/// One page: `record` between `begin_page` and `submit_page`, then the
/// wait.  The host steps around it are timed by the caller.
fn page(backend: &VulkanBackend, record: impl FnOnce(&VulkanBackend)) -> Steps {
    let t = Instant::now();
    ok(backend.begin_page(), "begin_page");
    record(backend);
    let record_ns = ns(t);
    let t = Instant::now();
    let fence = ok(backend.submit_page(), "submit_page");
    let submit_ns = ns(t);
    let t = Instant::now();
    ok(backend.wait_page(fence), "wait_page");
    Steps {
        record: record_ns,
        submit: submit_ns,
        wait: ns(t),
        ..Steps::default()
    }
}

/// A device buffer and, on a device the host cannot reach in place, the
/// host buffer its transfers go through.
struct Slot {
    host: Option<<VulkanBackend as GpuBackend>::HostBuffer>,
    device: <VulkanBackend as GpuBackend>::DeviceBuffer,
}

impl Slot {
    fn new(backend: &VulkanBackend, len: usize, to_device: bool) -> Self {
        let device = ok(
            if to_device {
                backend.alloc_device(len)
            } else {
                backend.alloc_device_readback(len)
            },
            "alloc device",
        );
        let host = (!backend.device_memory_is_host_mapped()).then(|| {
            ok(
                if to_device {
                    backend.alloc_host_pinned(len)
                } else {
                    backend.alloc_host_readback(len)
                },
                "alloc host",
            )
        });
        Self { host, device }
    }

    fn write(&mut self, backend: &VulkanBackend, bytes: &[u8]) {
        let target = match self.host.as_mut() {
            Some(h) => ok(backend.host_bytes_mut(h), "host bytes"),
            // SAFETY: the bench waits for every page before touching a slot.
            None => unsafe { backend.device_bytes_mut(&mut self.device) }.expect("mapped"),
        };
        target[..bytes.len()].copy_from_slice(bytes);
    }

    fn read(&self, backend: &VulkanBackend, out: &mut [u8]) {
        let src = self.host.as_ref().map_or_else(
            // SAFETY: as in `write`.
            || unsafe { backend.device_bytes(&self.device) }.expect("mapped"),
            |h| ok(backend.host_bytes(h), "host bytes"),
        );
        out.copy_from_slice(&src[..out.len()]);
    }

    fn record_upload(&self, backend: &VulkanBackend, len: usize) {
        if let Some(h) = &self.host {
            // SAFETY: the page is waited on before the slot is reused.
            ok(
                unsafe { backend.record_copy_to_device(h, &self.device, len) },
                "copy up",
            );
        }
    }

    fn record_download(&mut self, backend: &VulkanBackend, len: usize) {
        if let Some(h) = self.host.as_mut() {
            // SAFETY: as in `record_upload`.
            ok(
                unsafe { backend.record_copy_to_host(&self.device, h, len) },
                "copy down",
            );
        }
    }

    fn free(self, backend: &VulkanBackend) {
        if let Some(h) = self.host {
            backend.free_host_pinned(h);
        }
        backend.free_device(self.device);
    }
}

fn header(title: &str) {
    println!("\n== {title} ==  (µs, median)");
    println!(
        "{:>14} {:>9} {:>9} {:>9} {:>9} {:>9} {:>10} {:>10} {:>10} {:>10}",
        "size", "prep", "record", "submit", "wait", "read", "total", "xfer_pg", "exec", "cpu"
    );
}

fn row(label: &str, s: Steps, xfer_wait: u64, exec: u64, cpu: Option<u64>) {
    println!(
        "{:>14} {:>9} {:>9} {:>9} {:>9} {:>9} {:>10} {:>10} {:>10} {:>10}",
        label,
        us(s.prep),
        us(s.record),
        us(s.submit),
        us(s.wait),
        us(s.read),
        us(s.total()),
        us(xfer_wait),
        us(exec),
        cpu.map_or_else(|| "-".to_string(), us)
    );
}

/// Wait of an empty page: the submission round trip with no work.
fn bench_empty(backend: &VulkanBackend, cfg: &Config) {
    let s = median_steps(cfg, || page(backend, |_| ()));
    println!(
        "\nempty page: record {} µs, submit {} µs, wait {} µs",
        us(s.record),
        us(s.submit),
        us(s.wait)
    );
}

/// A pseudo-random byte stream.
fn noise(len: usize, mut state: u32) -> Vec<u8> {
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state.to_be_bytes()[0]
        })
        .collect()
}

fn bench_icc(backend: &VulkanBackend, cfg: &Config) {
    // The renderer bakes every CLUT on a 17-node lattice.
    let grid_n = 17_u32;
    let table = noise((grid_n as usize).pow(4) * 3, 7);
    for smooth in [true, false] {
        header(&format!(
            "icc_clut grid {grid_n}, {} pixels: one-shot buffers split (prep = alloc + 2 sync uploads, read = sync download + free)",
            if smooth { "smooth" } else { "random" }
        ));
        for mpx in [0.5_f64, 2.0, 8.0] {
            let n = (mpx * 1e6) as usize;
            let cmyk = if smooth {
                smooth_cmyk(n)
            } else {
                noise(n * 4, 11)
            };
            let cpu = {
                let mut v = Vec::new();
                for _ in 0..3 {
                    let t = Instant::now();
                    let _ =
                        std::hint::black_box(icc_cmyk_to_rgb_cpu(&cmyk, Some((&table, grid_n))));
                    v.push(ns(t));
                }
                median(v)
            };
            // The production entry point, timed whole.
            let whole = median_steps(cfg, || {
                let t = Instant::now();
                let _ = ok(icc_clut(backend, &cmyk, &table), "icc_clut");
                Steps {
                    read: ns(t),
                    ..Steps::default()
                }
            })
            .read;
            // Its steps, apart.
            let n_px = n as u32;
            let mut rgb = vec![0; n * 3];
            let mut split = |repeats: u32| {
                let t = Instant::now();
                let c = ok(backend.alloc_device(cmyk.len()), "alloc");
                let l = ok(backend.alloc_device(table.len()), "alloc");
                let r = ok(backend.alloc_device_readback(n * 3), "alloc");
                ok(backend.upload(&c, &cmyk), "upload");
                ok(backend.upload(&l, &table), "upload");
                let prep = ns(t);
                let s = page(backend, |b| {
                    for _ in 0..repeats {
                        ok(
                            b.record_icc_clut(IccClutParams {
                                cmyk: &c,
                                rgb: &r,
                                clut: &l,
                                n_pixels: n_px,
                            }),
                            "record",
                        );
                    }
                });
                let t = Instant::now();
                ok(backend.download(&r, &mut rgb), "download");
                backend.free_device(c);
                backend.free_device(l);
                backend.free_device(r);
                Steps {
                    prep,
                    read: ns(t),
                    ..s
                }
            };
            let one = median_steps(cfg, || split(1));
            let many = median_steps(cfg, || split(REPEATS));
            let exec = many.wait.saturating_sub(one.wait) / u64::from(REPEATS - 1);
            row(&format!("{mpx} Mpx"), one, 0, exec, Some(cpu));
            println!("{:>14} whole icc_clut call {} µs", "", us(whole));
        }
    }
}

/// Sampling tables of a box of `box_n` pixels along one axis mapping onto
/// `src_n` source samples, entry layout `[x_hi, x_lo, y_hi, y_lo]`.
fn table(box_n: u32, src_n: u32, x: bool) -> Vec<u32> {
    (0..box_n)
        .flat_map(|i| {
            let q = ((u64::from(i) << 32) + (1 << 31)) * u64::from(src_n) / u64::from(box_n);
            let (hi, lo) = ((q >> 32) as u32, q as u32);
            if x { [hi, lo, 0, 0] } else { [0, 0, hi, lo] }
        })
        .collect()
}

fn bench_blit(backend: &VulkanBackend, cfg: &Config) {
    header("blit_image_box split (prep = table writes, read = box copy-out)");
    let (sw, sh) = (2550_u32, 3300_u32);
    let image = noise((sw * sh * 3) as usize, 3);
    let src_buf = ok(backend.alloc_device(image.len()), "alloc src");
    ok(backend.upload(&src_buf, &image), "upload src");
    for (bw, bh) in [(256_u32, 256_u32), (850, 1100), (1275, 1650), (2550, 3300)] {
        let cols = table(bw, sw, true);
        let rows = table(bh, sh, false);
        let cb: &[u8] = bytemuck::cast_slice(&cols);
        let rb: &[u8] = bytemuck::cast_slice(&rows);
        let target_len = (bw * bh * 4) as usize;
        let mut d_c = Slot::new(backend, cb.len(), true);
        let mut d_r = Slot::new(backend, rb.len(), true);
        let mut tgt = Slot::new(backend, target_len, false);
        let mut out = vec![0; target_len];
        let bbox = [
            0,
            0,
            i32::try_from(bw).expect("fits"),
            i32::try_from(bh).expect("fits"),
        ];
        let mut run = |kernels: u32, copies: bool| {
            let t = Instant::now();
            d_c.write(backend, cb);
            d_r.write(backend, rb);
            let prep = ns(t);
            let s = page(backend, |b| {
                if copies {
                    d_c.record_upload(b, cb.len());
                    d_r.record_upload(b, rb.len());
                    ok(b.record_zero_buffer(&tgt.device, target_len), "zero");
                }
                for _ in 0..kernels {
                    ok(
                        b.record_blit_image(BlitParams {
                            src: &src_buf,
                            dst: &tgt.device,
                            cols: &d_c.device,
                            rows: &d_r.device,
                            src_w: sw,
                            src_h: sh,
                            src_layout: 0,
                            dst_w: bw,
                            dst_h: bh,
                            bbox,
                            filter: BLIT_FILTER_X,
                        }),
                        "record",
                    );
                }
                if copies {
                    tgt.record_download(b, target_len);
                }
            });
            let t = Instant::now();
            tgt.read(backend, &mut out);
            Steps {
                prep,
                read: ns(t),
                ..s
            }
        };
        let one = median_steps(cfg, || run(1, true));
        let xfer = median_steps(cfg, || run(0, true));
        let k1 = median_steps(cfg, || run(1, false));
        let kn = median_steps(cfg, || run(REPEATS, false));
        let exec = kn.wait.saturating_sub(k1.wait) / u64::from(REPEATS - 1);
        row(&format!("{bw}x{bh}"), one, xfer.wait, exec, None);
        d_c.free(backend);
        d_r.free(backend);
        tgt.free(backend);
    }
    backend.free_device(src_buf);
}

/// Wall time of `total` blits of one box through [`blit_image_box`], run by
/// `threads` threads at once, each with its own scratch: what the page lock
/// serialises.
fn bench_lock(backend: &VulkanBackend, cfg: &Config) {
    println!("\n== concurrent blit_image_box: wall ms for 96 blits (median) ==");
    let (sw, sh) = (1700_u32, 2200_u32);
    let image = noise((sw * sh * 3) as usize, 5);
    let src_buf = ok(backend.alloc_device(image.len()), "alloc src");
    ok(backend.upload(&src_buf, &image), "upload src");
    let src = BlitSource::<VulkanBackend> {
        pixels: &src_buf,
        width: sw,
        height: sh,
        layout: 0,
    };
    let total = 96_usize;
    for (bw, bh) in [(256_u32, 256_u32), (850, 1100)] {
        let cols = table(bw, sw, true);
        let rows = table(bh, sh, false);
        let bbox = [
            0,
            0,
            i32::try_from(bw).expect("fits"),
            i32::try_from(bh).expect("fits"),
        ];
        let mut line = format!("{:>14}", format!("{bw}x{bh}"));
        for threads in [1_usize, 4, 12] {
            let wall = median_steps(cfg, || {
                let t = Instant::now();
                std::thread::scope(|s| {
                    for _ in 0..threads {
                        let _ = s.spawn(|| {
                            let mut scratch = BlitScratch::default();
                            for _ in 0..total / threads {
                                let px = ok(
                                    blit_image_box(
                                        backend,
                                        &mut scratch,
                                        &src,
                                        bbox,
                                        &cols,
                                        &rows,
                                        BLIT_FILTER_X,
                                    ),
                                    "blit",
                                );
                                let _ = std::hint::black_box(px.len());
                            }
                        });
                    }
                });
                Steps {
                    read: ns(t),
                    ..Steps::default()
                }
            })
            .read;
            write!(line, "  {threads:>2}T {:>8.2}", wall as f64 / 1e6).expect("write to a String");
        }
        println!("{line}");
    }
    backend.free_device(src_buf);
}

/// Host read-back of a device buffer written by [`GpuBackend::upload`]:
/// the image cache's read-back of a cached image, against the same read
/// from a read-back buffer.
fn bench_readback(backend: &VulkanBackend, cfg: &Config) {
    println!("\n== device buffer read-back (µs, median) ==");
    for len in [1_usize << 20, 8_700_000, 25_000_000] {
        let data = noise(len, 9);
        let mut out = vec![0; len];
        let written = ok(backend.alloc_device(len), "alloc");
        let read_buf = ok(backend.alloc_device_readback(len), "alloc");
        ok(backend.upload(&written, &data), "upload");
        ok(backend.upload(&read_buf, &data), "upload");
        let time = |f: &mut dyn FnMut()| {
            median_steps(cfg, || {
                let t = Instant::now();
                f();
                Steps {
                    read: ns(t),
                    ..Steps::default()
                }
            })
            .read
        };
        let upload = time(&mut || ok(backend.upload(&written, &data), "upload"));
        let plain = time(&mut || ok(backend.download(&written, &mut out), "download"));
        let readback = time(&mut || ok(backend.download(&read_buf, &mut out), "download"));
        assert_eq!(out, data);
        println!(
            "{:>10} B  upload {:>9}  download(write buf) {:>9}  download(readback buf) {:>9}",
            len,
            us(upload),
            us(plain),
            us(readback),
        );
        backend.free_device(written);
        backend.free_device(read_buf);
    }
}

fn main() {
    let cfg = parse_config();
    let selector = cfg
        .device
        .as_deref()
        .map(|s| s.parse::<DeviceSelector>().expect("device selector"));
    let backend = ok(VulkanBackend::with_selector(selector.as_ref()), "backend");
    println!(
        "device {} integrated={} iters={} warmup={}",
        backend.device_name(),
        backend.is_integrated(),
        cfg.iters,
        cfg.warmup
    );
    let want = |k: &str| cfg.only.as_deref().is_none_or(|o| o == k);
    bench_empty(&backend, &cfg);
    if want("icc") {
        bench_icc(&backend, &cfg);
    }
    if want("blit") {
        bench_blit(&backend, &cfg);
    }
    if want("lock") {
        bench_lock(&backend, &cfg);
    }
    if want("readback") {
        bench_readback(&backend, &cfg);
    }
}
