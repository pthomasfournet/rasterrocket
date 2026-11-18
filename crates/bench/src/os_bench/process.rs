//! Child-process rows: time `rrocket` invocations end to end.
//!
//! Every invocation starts on an empty output directory, emptied and checked
//! before the clock starts.  A failed child, or one that leaves the wrong
//! number of files, aborts the row with an error; no number is produced.

use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use crate::library::DocRange;

/// Where a child's pages go.
#[derive(Debug, Clone)]
pub(crate) enum Sink {
    /// Pages are written under `dir` with the in-memory output disabled.
    Disk { dir: PathBuf },
    /// Pages are held in memory under `root/r<rep>-<n>`, which the child
    /// removes on exit.  `spill` is an empty directory whose `p` prefix the
    /// child falls back to when memory runs short; it must stay empty.
    Ram { root: PathBuf, spill: PathBuf },
}

/// The in-memory output path of invocation `n` of repetition `rep`.
pub(crate) fn ram_path(root: &Path, rep: u32, n: u32) -> PathBuf {
    root.join(format!("r{rep}-{n}"))
}

/// Marker file naming a directory as the harness's own; only such a directory
/// (or an empty one) is ever emptied.
const MARKER: &str = ".os_bench";

/// Build the child's arguments and return them with the directory the pages
/// land in (for `Ram`, the spill directory, which must hold no files after
/// the run).  A range ending at `u32::MAX` omits `-l`: the child then renders
/// to the last page.
pub(crate) fn rrocket_argv(
    doc: &DocRange,
    dpi: u32,
    threads: usize,
    sink: &Sink,
    rep: u32,
    n: u32,
    extra: &[OsString],
) -> (Vec<OsString>, PathBuf) {
    let mut argv: Vec<OsString> = vec!["-r".into(), dpi.to_string().into()];
    argv.push("-f".into());
    argv.push(doc.first.to_string().into());
    if doc.last != u32::MAX {
        argv.push("-l".into());
        argv.push(doc.last.to_string().into());
    }
    argv.push("--threads".into());
    argv.push(threads.to_string().into());
    let (dir, prefix) = match sink {
        Sink::Disk { dir } => {
            argv.push("--no-ram".into());
            (dir.clone(), dir.join("p"))
        }
        Sink::Ram { root, spill } => {
            argv.push("--ram-path".into());
            argv.push(ram_path(root, rep, n).into_os_string());
            (spill.clone(), spill.join("p"))
        }
    };
    argv.extend(extra.iter().cloned());
    argv.push(doc.path.clone().into_os_string());
    argv.push(prefix.into_os_string());
    (argv, dir)
}

fn count_files(dir: &Path) -> Result<usize, String> {
    let entries = fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut n = 0;
    for entry in entries {
        let entry = entry.map_err(|e| format!("{}: {e}", dir.display()))?;
        if entry.file_name() != MARKER && !entry.file_type().is_ok_and(|t| t.is_dir()) {
            n += 1;
        }
    }
    Ok(n)
}

/// Empty `dir` (creating it if absent) and check it is empty.  The directory
/// is marked as the harness's own; a non-empty directory without the marker,
/// `.`, the working directory and any ancestor of it are refused.
fn empty_dir(dir: &Path) -> Result<(), String> {
    let err = |e: std::io::Error| format!("{}: {e}", dir.display());
    if dir.as_os_str().is_empty() || dir == Path::new(".") {
        return Err(format!(
            "{}: refusing to empty the working directory",
            dir.display()
        ));
    }
    fs::create_dir_all(dir).map_err(err)?;
    let real = fs::canonicalize(dir).map_err(err)?;
    let cwd = std::env::current_dir()
        .and_then(fs::canonicalize)
        .map_err(|e| format!("working directory: {e}"))?;
    if cwd.starts_with(&real) {
        return Err(format!(
            "{}: refusing to empty the working directory or one of its ancestors",
            dir.display()
        ));
    }
    let marker = dir.join(MARKER);
    if !marker.exists() {
        if fs::read_dir(dir).map_err(err)?.next().is_some() {
            return Err(format!(
                "{}: refusing to empty a non-empty directory without the {MARKER} marker",
                dir.display()
            ));
        }
        fs::write(&marker, b"").map_err(|e| format!("{}: {e}", marker.display()))?;
    }
    for entry in fs::read_dir(dir).map_err(err)? {
        let path = entry.map_err(err)?.path();
        if path == marker {
            continue;
        }
        if path.is_dir() {
            fs::remove_dir_all(&path).map_err(err)?;
        } else {
            fs::remove_file(&path).map_err(err)?;
        }
    }
    match fs::read_dir(dir).map_err(err)?.count() {
        1 => Ok(()),
        left => Err(format!(
            "{}: {} entries remain after emptying",
            dir.display(),
            left.saturating_sub(1)
        )),
    }
}

/// Run the child once and return its wall microseconds.  `out_dir` is emptied
/// and checked empty before the clock starts.  A non-zero exit is an error
/// carrying the child's stderr; with `expect_files`, the file count in
/// `out_dir` afterwards must match.
pub(crate) fn process_rep(
    rrocket: &Path,
    argv: &[OsString],
    out_dir: Option<&Path>,
    expect_files: Option<u32>,
) -> Result<u64, String> {
    if let Some(dir) = out_dir {
        empty_dir(dir)?;
    }
    let start = Instant::now();
    let output = Command::new(rrocket)
        .args(argv)
        .output()
        .map_err(|e| format!("{}: {e}", rrocket.display()))?;
    let wall = u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX);
    if !output.status.success() {
        return Err(format!(
            "{} {argv:?} failed ({}): {}",
            rrocket.display(),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim_end()
        ));
    }
    if let (Some(dir), Some(want)) = (out_dir, expect_files) {
        let found = count_files(dir)?;
        if found != want as usize {
            return Err(format!(
                "{}: {found} of {want} expected output files",
                dir.display()
            ));
        }
    }
    Ok(wall)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/os-bench")
            .join(format!("test-{}-{tag}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn doc() -> DocRange {
        DocRange {
            path: PathBuf::from("x.pdf"),
            first: 1,
            last: 4,
        }
    }

    fn cargo() -> PathBuf {
        PathBuf::from(env!("CARGO"))
    }

    #[test]
    fn disk_argv_forces_no_ram_and_a_path_prefix() {
        let dir = PathBuf::from("out");
        let (argv, d) = rrocket_argv(&doc(), 150, 12, &Sink::Disk { dir: dir.clone() }, 1, 0, &[]);
        assert!(argv.iter().any(|a| a == "--no-ram"));
        assert_eq!(d, dir);
        let last = PathBuf::from(argv.last().unwrap());
        assert_eq!(last.parent().unwrap(), dir);
        let s = last.to_string_lossy().into_owned();
        assert!(s.ends_with("/p") || s.ends_with("\\p"), "{s}");
        assert!(argv.iter().any(|a| a == "-l"));
    }

    #[test]
    fn bare_range_omits_the_last_page_flag() {
        let mut d = doc();
        d.last = u32::MAX;
        let sink = Sink::Disk {
            dir: PathBuf::from("o"),
        };
        let (argv, _) = rrocket_argv(&d, 150, 1, &sink, 1, 0, &[]);
        assert!(!argv.iter().any(|a| a == "-l"));
    }

    #[test]
    fn ram_row_paths_are_fresh_and_absent() {
        let root = scratch("ram");
        let spill = root.join("spill");
        let sink = Sink::Ram {
            root: root.clone(),
            spill: spill.clone(),
        };
        let mut seen = Vec::new();
        for rep in 1..=3 {
            for n in 0..=2 {
                let (argv, dir) = rrocket_argv(&doc(), 150, 4, &sink, rep, n, &[]);
                assert_eq!(dir, spill);
                let at = argv.iter().position(|a| a == "--ram-path").unwrap();
                let p = PathBuf::from(&argv[at + 1]);
                assert!(p.starts_with(&root));
                assert!(!p.exists());
                assert!(!seen.contains(&p));
                seen.push(p);
                assert!(!argv.iter().any(|a| a == "--no-ram"));
                let prefix = PathBuf::from(argv.last().unwrap());
                assert!(prefix.starts_with(&spill) && prefix != spill);
            }
        }
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn process_rep_rejects_a_failing_child() {
        let err = process_rep(&cargo(), &["--no-such-flag".into()], None, None).unwrap_err();
        assert!(err.contains("--no-such-flag"), "{err}");
    }

    #[test]
    fn process_row_counts_output_files() {
        let dir = scratch("count");
        let err = process_rep(&cargo(), &["--version".into()], Some(&dir), Some(3)).unwrap_err();
        assert!(err.contains("0 of 3"), "{err}");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn empty_dir_refuses_an_unmarked_non_empty_dir_and_the_cwd() {
        let dir = scratch("marker");
        let other = dir.join("foreign");
        fs::create_dir(&other).unwrap();
        fs::write(other.join("keep"), b"x").unwrap();
        let err = empty_dir(&other).unwrap_err();
        assert!(err.contains("foreign") && err.contains(MARKER), "{err}");
        assert!(other.join("keep").exists());
        assert!(empty_dir(Path::new(".")).is_err());
        assert!(empty_dir(&std::env::current_dir().unwrap()).is_err());
        let own = dir.join("own");
        empty_dir(&own).unwrap();
        assert!(own.join(MARKER).exists());
        fs::write(own.join("x"), b"x").unwrap();
        empty_dir(&own).unwrap();
        assert!(!own.join("x").exists());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn process_row_starts_each_rep_on_an_empty_dir() {
        let dir = scratch("empty");
        empty_dir(&dir).unwrap();
        fs::write(dir.join("stale"), b"x").unwrap();
        fs::create_dir(dir.join("sub")).unwrap();
        process_rep(&cargo(), &["--version".into()], Some(&dir), Some(0)).unwrap();
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 1); // the marker only
        fs::remove_dir_all(&dir).unwrap();
    }
}
