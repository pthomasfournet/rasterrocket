//! Per-page render latency, in process: one session per document, pages
//! rendered sequentially on the calling thread, RGB then gray.
//! Usage: `page_latency <dpi> <pdf>...`; prints `pdf\tpage\topen_us\tpage_us`.

use std::time::Instant;

use rasterrocket::{BackendPolicy, SessionConfig, open_session, render_page_rgb, rgb_to_gray};

fn main() {
    let mut args = std::env::args().skip(1);
    let dpi: f64 = args.next().and_then(|s| s.parse().ok()).expect("dpi");
    let config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    for pdf in args {
        let t = Instant::now();
        let Ok(session) = open_session(std::path::Path::new(&pdf), &config) else {
            continue;
        };
        let open_us = t.elapsed().as_micros();
        for page in 1..=session.total_pages() {
            let t = Instant::now();
            let ok = render_page_rgb(&session, page, dpi / 72.0).map(|bmp| rgb_to_gray(&bmp));
            let us = t.elapsed().as_micros();
            if ok.is_ok() {
                println!("{pdf}\t{page}\t{open_us}\t{us}");
            }
        }
    }
}
