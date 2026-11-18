//! Cross-OS benchmark harness.
//!
//! Subcommands:
//!   - `library --group G --row R --dpi D --threads N --reps K [--deskew]
//!     [--page-times FILE] PDF[:FIRST-LAST]...` — in-memory library renders;
//!     one tab-separated sample line per rep and document on stdout.
//!
//! The caller sets `RAYON_NUM_THREADS`; the harness never does.

mod library;

use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;

use library::{DocRange, Sample, deskew_rep_diag, library_rep_diag};

/// Parsed `library` command line.
struct LibraryArgs {
    group: String,
    row: String,
    dpi: f64,
    threads: usize,
    reps: u32,
    deskew: bool,
    page_times: Option<PathBuf>,
    docs: Vec<DocRange>,
}

fn value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next().ok_or_else(|| format!("{flag}: missing value"))
}

fn number<T: std::str::FromStr>(text: &str, flag: &str) -> Result<T, String> {
    text.parse()
        .map_err(|_| format!("{flag}: cannot parse {text:?}"))
}

fn parse_library(mut args: impl Iterator<Item = String>) -> Result<LibraryArgs, String> {
    let (mut group, mut row, mut dpi, mut threads, mut reps) = (None, None, None, None, None);
    let mut deskew = false;
    let mut page_times = None;
    let mut docs = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--group" => group = Some(value(&mut args, &arg)?),
            "--row" => row = Some(value(&mut args, &arg)?),
            "--dpi" => dpi = Some(number::<f64>(&value(&mut args, &arg)?, &arg)?),
            "--threads" => threads = Some(number::<usize>(&value(&mut args, &arg)?, &arg)?),
            "--reps" => reps = Some(number::<u32>(&value(&mut args, &arg)?, &arg)?),
            "--deskew" => deskew = true,
            "--page-times" => page_times = Some(PathBuf::from(value(&mut args, &arg)?)),
            flag if flag.starts_with("--") => return Err(format!("unknown flag {flag}")),
            _ => docs.push(DocRange::parse(&arg)?),
        }
    }
    let need = |v: Option<String>, flag: &str| v.ok_or_else(|| format!("{flag} is required"));
    let dpi = dpi.ok_or("--dpi is required")?;
    let threads = threads.ok_or("--threads is required")?;
    let reps = reps.ok_or("--reps is required")?;
    if !(dpi.is_finite() && dpi > 0.0) {
        return Err(format!("--dpi must be positive, got {dpi}"));
    }
    if threads == 0 || reps == 0 {
        return Err("--threads and --reps must be at least 1".to_owned());
    }
    if deskew && page_times.is_some() {
        return Err("--deskew cannot be combined with --page-times".to_owned());
    }
    if deskew && threads != 1 {
        return Err(format!(
            "--deskew runs one worker; --threads {threads} conflicts"
        ));
    }
    if docs.is_empty() {
        return Err("no PDF given".to_owned());
    }
    Ok(LibraryArgs {
        group: need(group, "--group")?,
        row: need(row, "--row")?,
        dpi,
        threads,
        reps,
        deskew,
        page_times,
        docs,
    })
}

fn run_library(args: &LibraryArgs) -> Result<(), String> {
    let mut sink = match &args.page_times {
        Some(path) => {
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            Some((path, file))
        }
        None => None,
    };
    for doc in &args.docs {
        let name = doc.file_name();
        for rep in 1..=args.reps {
            let mut times = Vec::new();
            let (r, threads) = if args.deskew {
                // Detection runs on the global pool; the row times one worker.
                (deskew_rep_diag(doc, args.dpi)?, 1)
            } else {
                let r = library_rep_diag(
                    doc,
                    args.dpi,
                    args.threads,
                    sink.is_some().then_some(&mut times),
                )?;
                (r, args.threads)
            };
            let sample = Sample {
                group: args.group.clone(),
                row: args.row.clone(),
                doc: name.clone(),
                pages: r.pages,
                threads,
                rep,
                wall_us: r.wall_us,
            };
            println!("{}", sample.tsv_line());
            if r.degraded > 0 {
                eprintln!(
                    "os_bench: {name} rep {rep}: {} of {} page(s) reported diagnostics",
                    r.degraded, r.pages
                );
            }
            if let Some((path, file)) = sink.as_mut() {
                times.sort_unstable();
                for (page, us) in &times {
                    writeln!(file, "{}\t{name}\t{page}\t{us}", args.row)
                        .map_err(|e| format!("{}: {e}", path.display()))?;
                }
            }
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let result = match args.next().as_deref() {
        Some("library") => parse_library(args).and_then(|a| run_library(&a)),
        _ => Err(
            "usage: os_bench library --group G --row R --dpi D --threads N --reps K \
             [--deskew] [--page-times FILE] PDF[:FIRST-LAST]..."
                .to_owned(),
        ),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("os_bench: {e}");
            ExitCode::from(1)
        }
    }
}
