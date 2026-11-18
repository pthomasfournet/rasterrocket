//! Cross-OS benchmark harness.
//!
//! Subcommands:
//!   - `library --group G --row R --dpi D --threads N --reps K [--deskew]
//!     [--page-times FILE] PDF[:FIRST-LAST]...` — in-memory library renders;
//!     one tab-separated sample line per rep and document on stdout.
//!
//!   - `process --group G --row R --rrocket PATH --dpi D --threads N --reps K
//!     (--disk DIR | --ram ROOT | --none) [--invocations M] [--expect-pages]
//!     [PDF[:FIRST-LAST]...] [-- EXTRA...]` — child-process rows.
//!
//! The caller sets `RAYON_NUM_THREADS`; the harness never does.

mod library;
mod process;

use std::ffi::OsString;
use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;

use library::{DocRange, Sample, deskew_rep, library_rep};
use pdf_raster::{BackendPolicy, SessionConfig, open_session};
use process::{Sink, process_rep, ram_path, rrocket_argv};

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
                (deskew_rep(doc, args.dpi)?, 1)
            } else {
                let r = library_rep(
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

/// Parsed `process` command line.
#[derive(Debug)]
struct ProcessArgs {
    group: String,
    row: String,
    rrocket: PathBuf,
    dpi: u32,
    threads: usize,
    reps: u32,
    /// `None` for `--none`.
    sink: Option<Sink>,
    invocations: u32,
    expect_pages: bool,
    docs: Vec<DocRange>,
    extra: Vec<OsString>,
}

fn parse_process(mut args: impl Iterator<Item = String>) -> Result<ProcessArgs, String> {
    let (mut group, mut row, mut rrocket, mut dpi, mut threads, mut reps) =
        (None, None, None, None, None, None);
    let mut sink: Option<Option<Sink>> = None;
    let mut spill = None;
    let mut invocations = None;
    let mut expect_pages = false;
    let mut docs = Vec::new();
    let mut extra = Vec::new();
    let mut set_sink = |s: Option<Sink>, flag: &str| {
        if sink.replace(s).is_some() {
            Err(format!("{flag}: only one of --disk, --ram, --none"))
        } else {
            Ok(())
        }
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--" => {
                extra.extend(args.by_ref().map(OsString::from));
            }
            "--group" => group = Some(value(&mut args, &arg)?),
            "--row" => row = Some(value(&mut args, &arg)?),
            "--rrocket" => rrocket = Some(PathBuf::from(value(&mut args, &arg)?)),
            "--dpi" => dpi = Some(number::<u32>(&value(&mut args, &arg)?, &arg)?),
            "--threads" => threads = Some(number::<usize>(&value(&mut args, &arg)?, &arg)?),
            "--reps" => reps = Some(number::<u32>(&value(&mut args, &arg)?, &arg)?),
            "--invocations" => invocations = Some(number::<u32>(&value(&mut args, &arg)?, &arg)?),
            "--expect-pages" => expect_pages = true,
            "--disk" => {
                let dir = PathBuf::from(value(&mut args, &arg)?);
                set_sink(Some(Sink::Disk { dir }), &arg)?;
            }
            "--ram" => {
                // The spill directory is attached once all flags are read.
                let root = PathBuf::from(value(&mut args, &arg)?);
                set_sink(
                    Some(Sink::Ram {
                        root,
                        spill: PathBuf::new(),
                    }),
                    &arg,
                )?;
            }
            "--spill" => spill = Some(PathBuf::from(value(&mut args, &arg)?)),
            "--none" => set_sink(None, &arg)?,
            flag if flag.starts_with("--") => return Err(format!("unknown flag {flag}")),
            _ => docs.push(DocRange::parse(&arg)?),
        }
    }
    let mut sink = sink.ok_or("one of --disk, --ram, --none is required")?;
    match (&mut sink, spill) {
        (Some(Sink::Ram { spill: slot, .. }), Some(dir)) => *slot = dir,
        (Some(Sink::Ram { .. }), None) => return Err("--ram requires --spill DIR".to_owned()),
        (_, Some(_)) => return Err("--spill is only valid with --ram".to_owned()),
        _ => {}
    }
    let dpi = dpi.ok_or("--dpi is required")?;
    let threads = threads.ok_or("--threads is required")?;
    let reps = reps.ok_or("--reps is required")?;
    let invocations = invocations.unwrap_or(1);
    if dpi == 0 || threads == 0 || reps == 0 || invocations == 0 {
        return Err("--dpi, --threads, --reps and --invocations must be at least 1".to_owned());
    }
    if sink.is_none() && !docs.is_empty() {
        return Err("--none runs rrocket with EXTRA alone; no PDF may be given".to_owned());
    }
    if sink.is_some() && docs.is_empty() {
        return Err("no PDF given".to_owned());
    }
    if sink.is_none() && expect_pages {
        return Err("--expect-pages needs a PDF".to_owned());
    }
    Ok(ProcessArgs {
        group: group.ok_or("--group is required")?,
        row: row.ok_or("--row is required")?,
        rrocket: rrocket.ok_or("--rrocket is required")?,
        dpi,
        threads,
        reps,
        sink,
        invocations,
        expect_pages,
        docs,
        extra,
    })
}

/// Run `one(n)` for `n` in `0..invocations`, each timed alone, and return the
/// mean span in microseconds.
fn mean_invocation_us(
    invocations: u32,
    mut one: impl FnMut(u32) -> Result<u64, String>,
) -> Result<u64, String> {
    let mut sum = 0u64;
    for n in 0..invocations {
        sum += one(n)?;
    }
    Ok(sum / u64::from(invocations))
}

/// The `pages` field: the invocation count when several are averaged, else
/// the range's page count.
const fn sample_pages(invocations: u32, range_pages: u32) -> u32 {
    if invocations > 1 {
        invocations
    } else {
        range_pages
    }
}

fn run_process(args: &ProcessArgs) -> Result<(), String> {
    let emit = |doc: &str, pages: u32, rep: u32, wall_us: u64| {
        let sample = Sample {
            group: args.group.clone(),
            row: args.row.clone(),
            doc: doc.to_owned(),
            pages,
            threads: args.threads,
            rep,
            wall_us,
        };
        println!("{}", sample.tsv_line());
    };
    let Some(sink) = &args.sink else {
        // No document: one child run per invocation, nothing to count.
        for rep in 1..=args.reps {
            let wall = mean_invocation_us(args.invocations, |_| {
                process_rep(&args.rrocket, &args.extra, None, None)
            })?;
            emit("-", args.invocations, rep, wall);
        }
        return Ok(());
    };
    if let Sink::Ram { root, .. } = sink
        && !root.is_dir()
    {
        return Err(format!(
            "{}: RAM root does not exist (it must be mounted before the run)",
            root.display()
        ));
    }
    for doc in &args.docs {
        let name = doc.file_name();
        // The page count is read before any clock starts.
        let total = open_session(
            &doc.path,
            &SessionConfig::with_policy(BackendPolicy::CpuOnly),
        )
        .map_err(|e| format!("{name}: {e}"))?
        .total_pages();
        let (first, last) = doc.clamp(total)?;
        let range_pages = last - first + 1;
        let expect = match sink {
            Sink::Disk { .. } => args.expect_pages.then_some(range_pages),
            Sink::Ram { .. } => Some(0),
        };
        for rep in 1..=args.reps {
            let wall = mean_invocation_us(args.invocations, |n| {
                let (command, out_dir) =
                    rrocket_argv(doc, args.dpi, args.threads, sink, rep, n, &args.extra);
                if let Sink::Ram { root, .. } = sink {
                    let fresh = ram_path(root, rep, n);
                    if fresh.exists() {
                        return Err(format!("{}: RAM path already exists", fresh.display()));
                    }
                }
                process_rep(&args.rrocket, &command, Some(&out_dir), expect)
                    .map_err(|e| format!("{name} rep {rep}: {e}"))
            })?;
            emit(
                &name,
                sample_pages(args.invocations, range_pages),
                rep,
                wall,
            );
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let result = match args.next().as_deref() {
        Some("library") => parse_library(args).and_then(|a| run_library(&a)),
        Some("process") => parse_process(args).and_then(|a| run_process(&a)),
        _ => Err(
            "usage: os_bench library --group G --row R --dpi D --threads N --reps K \
             [--deskew] [--page-times FILE] PDF[:FIRST-LAST]...\n       \
             os_bench process --group G --row R --rrocket PATH --dpi D --threads N --reps K \
             (--disk DIR | --ram ROOT | --none) [--invocations M] [--expect-pages] \
             [PDF[:FIRST-LAST]...] [-- EXTRA...]"
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

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    fn parse(args: &[&str]) -> Result<ProcessArgs, String> {
        parse_process(args.iter().map(|s| (*s).to_owned()))
    }

    const BASE: [&str; 12] = [
        "--group",
        "g",
        "--row",
        "r",
        "--rrocket",
        "rr",
        "--dpi",
        "150",
        "--threads",
        "2",
        "--reps",
        "1",
    ];

    fn with(extra: &[&str]) -> Result<ProcessArgs, String> {
        let all: Vec<&str> = BASE.iter().chain(extra).copied().collect();
        parse(&all)
    }

    #[test]
    fn double_dash_splits_pdfs_from_extra() {
        let a = with(&[
            "--disk",
            "d",
            "a.pdf:2-3",
            "--",
            "--backend",
            "cpu",
            "b.pdf",
        ])
        .unwrap();
        assert_eq!(a.docs.len(), 1);
        assert_eq!((a.docs[0].first, a.docs[0].last), (2, 3));
        assert_eq!(a.extra, ["--backend", "cpu", "b.pdf"].map(OsString::from));
    }

    #[test]
    fn sinks_are_exclusive_and_required() {
        assert!(with(&["--disk", "d", "--none", "a.pdf"]).is_err());
        assert!(with(&["--disk", "d", "--ram", "r", "--spill", "s", "a.pdf"]).is_err());
        assert!(with(&["a.pdf"]).is_err());
    }

    #[test]
    fn none_takes_no_pdf_and_no_expect_pages() {
        assert!(with(&["--none", "a.pdf"]).is_err());
        assert!(with(&["--none", "--expect-pages"]).is_err());
        assert!(with(&["--none", "--", "--version"]).is_ok());
    }

    #[test]
    fn ram_requires_spill_and_spill_requires_ram() {
        assert!(
            with(&["--ram", "r", "a.pdf"])
                .unwrap_err()
                .contains("--spill")
        );
        assert!(with(&["--disk", "d", "--spill", "s", "a.pdf"]).is_err());
        let a = with(&["--ram", "r", "--spill", "s", "a.pdf"]).unwrap();
        assert!(matches!(a.sink, Some(Sink::Ram { spill, .. }) if spill == Path::new("s")));
    }

    #[test]
    fn invocations_are_timed_alone_and_averaged() {
        let mut seen = Vec::new();
        let spans = [10u64, 20, 60];
        let mean = mean_invocation_us(3, |n| {
            seen.push(n);
            Ok(spans[n as usize])
        })
        .unwrap();
        assert_eq!(seen, [0, 1, 2]);
        assert_eq!(mean, 30);
        assert!(
            mean_invocation_us(3, |n| if n == 1 { Err("boom".into()) } else { Ok(1) }).is_err()
        );
        assert_eq!(sample_pages(3, 16), 3);
        assert_eq!(sample_pages(1, 16), 16);
    }
}
