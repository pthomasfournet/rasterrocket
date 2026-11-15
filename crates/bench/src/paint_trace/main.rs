//! Paint-trace tools.
//!
//! Subcommands:
//!   - `dump PDF [-p PAGE] [-r DPI]`: print the paint trace of a page
//!     (default page 1 at 72 dpi), as `rasterrocket::dev::paint_trace`
//!     records it.
//!   - `diff A.trace B.trace`: the operations of two traces that differ,
//!     aligned by operation, with the attributes that moved.
//!   - `blame TRACE BEFORE.pnm AFTER.pnm`: the operations of `TRACE` that
//!     cover the pixels at which two renders of its page differ, most
//!     pixels first.  The renders are binary PPM or PGM files of 8-bit
//!     samples at the trace's resolution.
//!
//! Exit status: 0 when nothing differs (`diff`, `blame`) or the trace was
//! printed (`dump`), 1 on a difference, 2 on bad usage or unreadable input.

mod blame;
mod diff;
mod pnm;
mod trace;

use std::path::Path;
use std::process::ExitCode;

use pdf_raster::{BackendPolicy, RenderParams, SessionConfig};

const USAGE: &str = "usage:\n  paint_trace dump PDF [-p PAGE] [-r DPI]\n  \
                     paint_trace diff A.trace B.trace\n  \
                     paint_trace blame TRACE BEFORE.pnm AFTER.pnm";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("dump") => dump(&args[1..]),
        Some("diff") => match &args[1..] {
            [a, b] => read(a)
                .and_then(|a| Ok((a, read(b)?)))
                .map(|(a, b)| diff::run(&a, &b)),
            _ => Err(USAGE.to_owned()),
        },
        Some("blame") => match &args[1..] {
            [t, before, after] => blame_files(t, before, after),
            _ => Err(USAGE.to_owned()),
        },
        _ => Err(USAGE.to_owned()),
    };
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(message) => {
            eprintln!("paint_trace: {message}");
            ExitCode::from(2)
        }
    }
}

/// The text of the file `path`.
fn read(path: &str) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))
}

/// `dump`: print the trace of a page.
fn dump(args: &[String]) -> Result<bool, String> {
    let mut pdf = None;
    let (mut page, mut dpi) = (1_u32, 72.0_f64);
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut value = |name: &str| {
            it.next()
                .ok_or_else(|| format!("{name} needs a value\n{USAGE}"))
                .cloned()
        };
        match arg.as_str() {
            "-p" => {
                let v = value("-p")?;
                page = v
                    .parse()
                    .map_err(|_| format!("-p {v}: not a page number"))?;
            }
            "-r" => {
                let v = value("-r")?;
                dpi = v
                    .parse()
                    .ok()
                    .filter(|d: &f64| d.is_finite() && *d > 0.0)
                    .ok_or_else(|| format!("-r {v}: not a positive resolution"))?;
            }
            _ if pdf.is_none() => pdf = Some(arg.clone()),
            _ => return Err(format!("unexpected argument {arg}\n{USAGE}")),
        }
    }
    let pdf = pdf.ok_or_else(|| USAGE.to_owned())?;
    let session = pdf_raster::open_session(
        Path::new(&pdf),
        &SessionConfig::with_policy(BackendPolicy::CpuOnly),
    )
    .map_err(|e| format!("cannot open {pdf}: {e}"))?;
    let trace = pdf_raster::dev::paint_trace(&session, page, &RenderParams::new(dpi / 72.0))
        .map_err(|e| format!("{pdf} page {page}: {e}"))?;
    print!("{trace}");
    Ok(true)
}

/// `blame`: attribute the pixels two renders differ at to the operations
/// of a trace.
fn blame_files(trace: &str, before: &str, after: &str) -> Result<bool, String> {
    let trace = read(trace)?;
    let before = pnm::read(Path::new(before))?;
    let after = pnm::read(Path::new(after))?;
    blame::run(&trace::parse(&trace), &before, &after)
}
