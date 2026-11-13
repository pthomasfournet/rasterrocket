//! Heap allocations and time per page of gray renders through the
//! owned-buffer entry point and through a reused `PageBuffer`.
//!
//! Usage: `page_buffer_allocs <dpi> <pdf> [pages]`.  Every page up to
//! `pages` (default all) is rendered once to warm the session's font and
//! image caches, then each mode renders them again under a counting
//! allocator.  Prints, per mode: allocations per page, bytes allocated per
//! page, allocations of at least a page's gray size per page, and mean
//! time per page.

use std::alloc::{GlobalAlloc, Layout, System};
use std::process::ExitCode;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use rasterrocket::{
    BackendPolicy, PageBuffer, RasterOptions, SessionConfig, open_session, render_page_gray,
};

/// Counts the allocations made through it; a reallocation counts as one.
struct Counting;

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
static LARGE: AtomicU64 = AtomicU64::new(0);
/// Size from which an allocation counts as large.
static LARGE_FROM: AtomicUsize = AtomicUsize::new(usize::MAX);

impl Counting {
    fn record(size: usize) {
        let _ = ALLOCS.fetch_add(1, Ordering::Relaxed);
        let _ = BYTES.fetch_add(size as u64, Ordering::Relaxed);
        if size >= LARGE_FROM.load(Ordering::Relaxed) {
            let _ = LARGE.fetch_add(1, Ordering::Relaxed);
        }
    }
}

// SAFETY: every method forwards to `System` with the caller's arguments.
#[expect(unsafe_code, reason = "a global allocator is an unsafe trait")]
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        Self::record(layout.size());
        // SAFETY: forwarded under the caller's contract.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        Self::record(layout.size());
        // SAFETY: forwarded under the caller's contract.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarded under the caller's contract.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        Self::record(new_size);
        // SAFETY: forwarded under the caller's contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// `(allocations, bytes, large allocations)` so far.
fn counters() -> (u64, u64, u64) {
    (
        ALLOCS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
        LARGE.load(Ordering::Relaxed),
    )
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(dpi), Some(pdf)) = (
        args.first()
            .and_then(|s| s.parse::<f32>().ok())
            .filter(|d| d.is_finite() && *d > 0.0),
        args.get(1),
    ) else {
        eprintln!("usage: page_buffer_allocs <dpi> <pdf> [pages]");
        return ExitCode::from(2);
    };
    let config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    let session = match open_session(std::path::Path::new(pdf), &config) {
        Ok(session) => session,
        Err(e) => {
            eprintln!("{pdf}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let last = args
        .get(2)
        .and_then(|s| s.parse::<u32>().ok())
        .map_or_else(|| session.total_pages(), |n| n.min(session.total_pages()));
    let pages: Vec<u32> = (1..=last).collect();
    let opts = RasterOptions::new(dpi);

    // The smallest page's gray size: an allocation of at least this many
    // bytes is a page-sized buffer.
    let mut sizes = Vec::with_capacity(pages.len());
    let mut widths = Vec::with_capacity(pages.len());
    for &page in &pages {
        match session.page_size_px(page, dpi) {
            Ok((w, h)) => {
                sizes.push(w as usize * h as usize);
                widths.push(w as usize);
            }
            Err(e) => {
                eprintln!("page {page}: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    let largest = sizes.iter().copied().max().unwrap_or(0);
    LARGE_FROM.store(
        sizes.iter().copied().min().unwrap_or(usize::MAX),
        Ordering::Relaxed,
    );

    let mut buffer = PageBuffer::new();
    let mut out = vec![0u8; largest];
    for &page in &pages {
        if let Err(e) = render_page_gray(&session, page, &opts) {
            eprintln!("page {page}: {e}");
            return ExitCode::FAILURE;
        }
    }

    let n = f64::from(last);
    #[expect(
        clippy::cast_precision_loss,
        reason = "counter deltas stay far below 2^52"
    )]
    let report = |mode: &str, before: (u64, u64, u64), secs: f64| {
        let after = counters();
        println!(
            "{mode}\tallocs/page {:.1}\tbytes/page {:.0}\tpage-sized/page {:.2}\tms/page {:.3}",
            (after.0 - before.0) as f64 / n,
            (after.1 - before.1) as f64 / n,
            (after.2 - before.2) as f64 / n,
            secs * 1000.0 / n,
        );
    };

    for _ in 0..3 {
        let before = counters();
        let t = Instant::now();
        for &page in &pages {
            let page = render_page_gray(&session, page, &opts).expect("owned render");
            let _ = std::hint::black_box(&page.pixels);
        }
        report("owned", before, t.elapsed().as_secs_f64());

        let before = counters();
        let t = Instant::now();
        for (&page, &stride) in pages.iter().zip(&widths) {
            let info = buffer
                .render_gray_into(&session, page, &opts, &mut out, stride)
                .expect("buffer render");
            let _ = std::hint::black_box((&out, info));
        }
        report("page_buffer", before, t.elapsed().as_secs_f64());
    }
    ExitCode::SUCCESS
}
