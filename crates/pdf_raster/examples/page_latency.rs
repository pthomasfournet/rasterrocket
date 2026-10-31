//! Per-page render latency, in process: one session per document, pages
//! rendered sequentially on the calling thread, RGB then gray.
//! Usage: `page_latency <dpi> <pdf>...`; prints `pdf\tpage\topen_us\tpage_us`.
//! Documents that cannot be opened and pages that fail to render are
//! reported on stderr and left out of the table.

use std::process::ExitCode;
use std::time::Instant;

use rasterrocket::{BackendPolicy, SessionConfig, open_session, render_page_rgb, rgb_to_gray};

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(dpi) = args
        .next()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|d| d.is_finite() && *d > 0.0)
    else {
        eprintln!("usage: page_latency <dpi> <pdf>...  (dpi a positive number)");
        return ExitCode::from(2);
    };
    let config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    for pdf in args {
        let t = Instant::now();
        let session = match open_session(std::path::Path::new(&pdf), &config) {
            Ok(session) => session,
            Err(e) => {
                eprintln!("{pdf}: cannot open: {e}");
                continue;
            }
        };
        let open_us = t.elapsed().as_micros();
        for page in 1..=session.total_pages() {
            let t = Instant::now();
            let rendered = render_page_rgb(&session, page, dpi / 72.0).map(|bmp| rgb_to_gray(&bmp));
            let us = t.elapsed().as_micros();
            match rendered {
                Ok(_) => println!("{pdf}\t{page}\t{open_us}\t{us}"),
                Err(e) => eprintln!("{pdf}: page {page}: {e}"),
            }
        }
    }
    ExitCode::SUCCESS
}
