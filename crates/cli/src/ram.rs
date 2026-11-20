//! RAM-backed render output (default) with dynamic spill-to-disk when free
//! memory drops below a safety margin.
//!
//! # Default behaviour
//!
//! By default this module redirects per-page writes to a fresh tmpfs dir
//! (`/dev/shm/rrocket-<pid>-<nanos>/`) because writing to disk costs
//! 10–20× more wall time than writing to RAM and dominates render latency.
//! The user's `OUTPUT_PREFIX` becomes the file stem inside that dir, so
//! downstream consumers that read `out-NNN.ppm` keep working unchanged.
//!
//! The default applies only where `/dev/shm` exists; elsewhere (macOS, Windows,
//! the BSDs) a bare stem is written to the working directory like any prefix,
//! and RAM output needs `--ram-path` on a RAM disk the user has mounted.
//!
//! The redirect is suppressed when `OUTPUT_PREFIX` looks like a path
//! (contains `/`, starts with `.`) — those are taken literally as the user
//! asking for a real disk location, and under `--fax`, whose one document
//! is written after the last page. See [`should_use_ram`] for the rule.
//!
//! Flags:
//! - `--no-ram` forces disk regardless of prefix shape
//! - `--ram` forces RAM regardless of prefix shape
//! - `--ram-path <PATH>` overrides the default tmpfs location and implies RAM
//!
//! # What this module returns
//!
//! 1. A [`RamDirGuard`] that removes the tmpfs dir on drop (no-op when on disk).
//!    An interrupt stops the run through `main`'s return (see
//!    [`crate::interrupt`]), so it drops the guard too; a process killed
//!    outright or aborted by a panic leaves the directory.
//! 2. A [`SpillPolicy`] queried per-page: returns the RAM prefix while free
//!    memory is comfortable and the RAM file system has room for the pages
//!    in flight, and the disk prefix from the first page that finds either
//!    tight, or that finds the RAM file system full when it is written.
//!
//! The chosen directory is printed on stderr and on stdout at startup, so
//! callers can capture it via `out=$(rrocket doc.pdf p)`.
//!
//! # Why dynamic instead of pre-flight?
//!
//! A static "this PDF will produce N MB of output, do you have room?" check
//! relies on (a) reading every page's `MediaBox` at startup (slow) and (b)
//! guessing PNG/JPEG compression ratios (impossible). The dynamic approach
//! lets the renderer start producing pages immediately and only switches to
//! disk if the kernel reports tight memory or a nearly full RAM file
//! system — at which point we've already
//! benefited from RAM-speed writes for the early pages.

use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::args::Args;
use crate::render::{RenderError, write_output};

/// Minimum free RAM (bytes) we insist remains after the next write. Below
/// this, the writer spills to disk. 1 GiB is enough headroom for one more
/// in-flight page bitmap (worst-case ~50 MB at 600 DPI A3) plus normal
/// per-thread scratch and OS workings.
const RAM_SAFETY_MARGIN: u64 = 1024 * 1024 * 1024;

/// Free space (bytes) the RAM file system keeps above the pages in flight.
/// A RAM file system has a size of its own, half the memory by default for
/// `/dev/shm`, reached long before the available memory runs out.
const RAM_SPACE_MARGIN: u64 = 64 * 1024 * 1024;

/// How long a memory-availability reading is cached before being re-probed.
/// A probe costs ~10µs but would otherwise run N×M times per render
/// (N pages × M threads); 100ms is fast enough to react to a sudden memory
/// crunch and slow enough that the probe cost is negligible.
const PROBE_TTL: Duration = Duration::from_millis(100);

/// RAII handle to a tmpfs directory created for `--ram` mode.
///
/// `Drop` removes the directory and everything inside it. If removal fails
/// (e.g. another process moved the directory) the error is logged and
/// swallowed — there is no actionable response at process exit.
pub(crate) struct RamDirGuard {
    /// `Some(path)` when this guard owns a directory; `None` when `--ram`
    /// was off and the guard is a no-op.
    dir: Option<PathBuf>,
}

impl Drop for RamDirGuard {
    fn drop(&mut self) {
        let Some(dir) = self.dir.take() else { return };
        if let Err(e) = fs::remove_dir_all(&dir) {
            // Stderr only; the user's stdout may already be committed to a
            // pipeline and a stray log line would corrupt downstream parsing.
            note!(
                "rrocket: warning: could not remove ram dir {}: {e}",
                dir.display()
            );
        }
    }
}

/// Per-page write-target policy: RAM (the RAM directory) or disk (the
/// user's original `OUTPUT_PREFIX` location).
///
/// One instance is shared by all worker threads.
pub(crate) struct SpillPolicy {
    targets: Targets,
    /// The cached readings, taken under the lock by the first caller after
    /// they expire.
    probe: Mutex<Probe>,
    /// Set by the first page sent to disk; every later page goes there too.
    spilled: AtomicBool,
    /// The size in bytes of the largest page file written so far.
    largest_page: AtomicU64,
}

/// Where a [`SpillPolicy`] writes pages.
enum Targets {
    /// Every page to the user's prefix: RAM output is off.
    Disk(String),
    /// Pages to `ram`, a prefix inside the RAM directory `dir`, while they
    /// fit there, then to `disk`.
    Ram {
        dir: PathBuf,
        ram: String,
        disk: String,
    },
}

/// Where the next page goes, and why it no longer goes to RAM.
#[derive(Debug, PartialEq, Eq)]
enum Verdict {
    Ram,
    Spill(&'static str),
}

/// Where the next page goes: to RAM while the available memory, when
/// readable, stays at least [`RAM_SAFETY_MARGIN`], and the free space of
/// the RAM file system, when readable, holds `in_flight` pages of
/// `page_bytes` above [`RAM_SPACE_MARGIN`].  Unreadable memory spills,
/// since pages could not be moved before memory runs out; unreadable free
/// space does not, the write of a page that finds the file system full
/// spilling instead (see [`SpillPolicy::write_page`]).
fn verdict(
    available: Option<u64>,
    free_space: Option<u64>,
    page_bytes: u64,
    in_flight: u64,
) -> Verdict {
    match available {
        None => return Verdict::Spill("free memory can no longer be read"),
        Some(bytes) if bytes < RAM_SAFETY_MARGIN => {
            return Verdict::Spill("free memory dropped below safety margin");
        }
        Some(_) => {}
    }
    let needed = page_bytes
        .saturating_mul(in_flight.max(1))
        .saturating_add(RAM_SPACE_MARGIN);
    if free_space.is_some_and(|free| free < needed) {
        return Verdict::Spill("the RAM file system is nearly full");
    }
    Verdict::Ram
}

impl SpillPolicy {
    /// A policy writing pages to `targets`.
    const fn new(targets: Targets) -> Self {
        Self {
            targets,
            probe: Mutex::new(Probe::new()),
            spilled: AtomicBool::new(false),
            largest_page: AtomicU64::new(0),
        }
    }

    /// The prefix for the next page write: the user's prefix when RAM
    /// output is off; otherwise the RAM prefix while pages fit in RAM (see
    /// [`verdict`]), and the disk prefix from then on.
    #[must_use]
    pub(crate) fn next_prefix(&self) -> &str {
        self.choose().0
    }

    /// [`Self::next_prefix`], and whether it is the RAM prefix.
    fn choose(&self) -> (&str, bool) {
        let (dir, ram, disk) = match &self.targets {
            Targets::Disk(prefix) => return (prefix, false),
            Targets::Ram { dir, ram, disk } => (dir, ram, disk),
        };
        if self.spilled.load(Ordering::Relaxed) {
            return (disk, false);
        }
        let (available, free_space) = {
            let mut probe = crate::render::lock(&self.probe);
            probe.refresh_if_stale(dir);
            (probe.available, probe.free_space)
        };
        let in_flight = u64::try_from(rayon::current_num_threads()).unwrap_or(u64::MAX);
        match verdict(
            available,
            free_space,
            self.largest_page.load(Ordering::Relaxed),
            in_flight,
        ) {
            Verdict::Ram => (ram, true),
            Verdict::Spill(reason) => (self.spill(reason), false),
        }
    }

    /// Send every later page to disk, telling the user the first time why;
    /// the disk prefix.
    fn spill(&self, reason: &str) -> &str {
        let disk = match &self.targets {
            Targets::Disk(prefix) | Targets::Ram { disk: prefix, .. } => prefix,
        };
        if !self.spilled.swap(true, Ordering::Relaxed) {
            note!("rrocket: {reason}; writing subsequent pages to disk with prefix {disk}");
        }
        disk
    }

    /// Write a page file, named by `path_of` from the prefix
    /// [`Self::next_prefix`] chooses, with `encode` (see
    /// [`write_output`]).  A page that finds the RAM file system full
    /// spills: it is written again, and every later page, to disk.
    ///
    /// # Errors
    ///
    /// The write failed, on disk or for a reason other than a full RAM
    /// file system.
    pub(crate) fn write_page(
        &self,
        path_of: impl Fn(&str) -> String,
        mut encode: impl FnMut(BufWriter<File>) -> Result<BufWriter<File>, RenderError>,
    ) -> Result<(), RenderError> {
        let (prefix, in_ram) = self.choose();
        match write_output(&path_of(prefix), &mut encode) {
            Ok(bytes) => {
                if in_ram {
                    let _ = self.largest_page.fetch_max(bytes, Ordering::Relaxed);
                }
                Ok(())
            }
            Err(e) if in_ram && storage_full(&e) => {
                let disk = self.spill("the RAM file system is full");
                write_output(&path_of(disk), encode).map(drop)
            }
            Err(e) => Err(e),
        }
    }
}

/// Whether `e` is, or is caused by, a full file system.
fn storage_full(e: &RenderError) -> bool {
    let mut next: Option<&(dyn std::error::Error + 'static)> = Some(e);
    while let Some(error) = next {
        if error
            .downcast_ref::<std::io::Error>()
            .is_some_and(|io| io.kind() == std::io::ErrorKind::StorageFull)
        {
            return true;
        }
        next = error.source();
    }
    false
}

/// Cached [`available_memory`] and [`free_space`] readings, refreshed
/// after [`PROBE_TTL`].
struct Probe {
    /// When the readings were taken; `None` before the first.
    last_read: Option<Instant>,
    /// The available memory; `None` when the platform offers no reading.
    available: Option<u64>,
    /// The free space of the RAM file system; `None` when it cannot be
    /// read.
    free_space: Option<u64>,
}

impl Probe {
    const fn new() -> Self {
        Self {
            last_read: None,
            available: None,
            free_space: None,
        }
    }

    /// Take the readings again, of the RAM directory `dir`, when they are
    /// older than [`PROBE_TTL`].
    fn refresh_if_stale(&mut self, dir: &Path) {
        if self.last_read.is_some_and(|t| t.elapsed() < PROBE_TTL) {
            return;
        }
        self.available = available_memory();
        self.free_space = free_space(dir);
        self.last_read = Some(Instant::now());
    }
}

/// The space, in bytes, an unprivileged process can still write on the
/// file system holding `dir`; `None` when it cannot be read.
#[cfg(unix)]
#[expect(
    unsafe_code,
    reason = "statvfs is the only interface to a file system's free space"
)]
fn free_space(dir: &Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt as _;
    let path = std::ffi::CString::new(dir.as_os_str().as_bytes()).ok()?;
    // SAFETY: an all-zero `statvfs` is a valid value of the C structure.
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: `path` is NUL-terminated and `stat` is writable; the call
    // writes only within it.
    if unsafe { libc::statvfs(path.as_ptr(), &raw mut stat) } != 0 {
        return None;
    }
    #[expect(
        clippy::useless_conversion,
        reason = "the field types differ between platforms"
    )]
    let (blocks, size) = (u64::from(stat.f_bavail), u64::from(stat.f_frsize));
    Some(blocks.saturating_mul(size))
}

/// No free-space reading on this platform: a full RAM disk is found by the
/// write that fails.
#[cfg(not(unix))]
const fn free_space(_: &Path) -> Option<u64> {
    None
}

/// The kernel's estimate, in bytes, of the memory a new workload can
/// allocate without swapping; `None` where the platform offers no such
/// estimate or it cannot be read.
#[cfg(target_os = "linux")]
fn available_memory() -> Option<u64> {
    read_mem_available()
}

/// The kernel's estimate, in bytes, of the memory a new workload can
/// allocate without swapping: `kern.memorystatus_level`, the percentage of
/// memory that is free or reclaimable (file cache, compressible pages), of
/// `hw.memsize`.  The free-page count alone is no such estimate: the file
/// cache and the compressor keep it near zero on a healthy system.
#[cfg(target_os = "macos")]
fn available_memory() -> Option<u64> {
    let level = u32::from_ne_bytes(sysctl_bytes(c"kern.memorystatus_level")?);
    let total = u64::from_ne_bytes(sysctl_bytes(c"hw.memsize")?);
    (level <= 100).then(|| total / 100 * u64::from(level))
}

/// The memory, in bytes, a new workload can allocate without paging:
/// `ullAvailPhys` of `GlobalMemoryStatusEx`, the free, zeroed and standby
/// (file cache) pages together.
#[cfg(windows)]
#[expect(
    unsafe_code,
    reason = "GlobalMemoryStatusEx is the only interface to the system's memory statistics"
)]
fn available_memory() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut status = MEMORYSTATUSEX {
        dwLength: u32::try_from(size_of::<MEMORYSTATUSEX>()).ok()?,
        ..MEMORYSTATUSEX::default()
    };
    // SAFETY: `status` is a writable `MEMORYSTATUSEX` whose `dwLength`
    // holds its size, as the call requires; the call writes only within it.
    let ok = unsafe { GlobalMemoryStatusEx(&raw mut status) };
    (ok != 0).then_some(status.ullAvailPhys)
}

/// No available-memory estimate on this platform.
#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
const fn available_memory() -> Option<u64> {
    None
}

/// The value of sysctl `name`, when it is exactly `N` bytes long.
#[cfg(target_os = "macos")]
#[expect(
    unsafe_code,
    reason = "sysctlbyname is the only interface to the kernel's memory statistics"
)]
fn sysctl_bytes<const N: usize>(name: &std::ffi::CStr) -> Option<[u8; N]> {
    let mut buf = [0_u8; N];
    let mut len = N;
    // SAFETY: `name` is NUL-terminated; `buf` is writable for `len` bytes,
    // and the kernel writes at most `len` bytes, storing the count written
    // in `len`; a null `newp` with `newlen` 0 sets nothing.
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            buf.as_mut_ptr().cast(),
            &raw mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && len == N).then_some(buf)
}

/// Read `MemAvailable` from `/proc/meminfo` (in bytes). Returns `None` if
/// the file is missing or malformed (a container without /proc, …).
#[cfg(target_os = "linux")]
fn read_mem_available() -> Option<u64> {
    let text = fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            // Format: "MemAvailable:   12345678 kB"
            let kb = rest.split_whitespace().next()?.parse::<u64>().ok()?;
            return Some(kb.saturating_mul(1024));
        }
    }
    None
}

/// Decide whether the renderer should redirect output to a tmpfs directory.
///
/// Precedence (highest first):
/// 1. `--no-ram` → always disk
/// 2. `--ram` or `--ram-path` → always RAM
/// 3. Heuristic on `output_prefix`, applied only when [`RAM_ROOT`] exists
///    (it does not on macOS, Windows or the BSDs, nor in some containers):
///    - bare stem (e.g. `out`, `p`) → RAM (default)
///    - path-like (`./out`, `dir/p`, `/abs/path`) → disk
///
/// Path-like means the prefix contains a path separator or starts with `.`
/// (covers `./out`, `../out`, and absolute paths). The intent: a user who
/// types `rrocket doc.pdf out` gets the fast path; a user who types
/// `rrocket doc.pdf ./out` or `rrocket doc.pdf /data/x` gets the
/// path they asked for.
fn should_use_ram(args: &Args) -> bool {
    // A fax document is one file written after the last page, which no
    // reader consumes during the run, and a print job writes no page
    // files: neither uses RAM (`--ram` with either is refused at
    // validation).
    !args.faxing()
        && !args.printing()
        && use_ram_for(
            args.no_ram,
            args.ram || args.ram_path.is_some(),
            args.prefix(),
            Path::new(RAM_ROOT).is_dir(),
        )
}

/// Pure-function form of [`should_use_ram`] for unit testing without an
/// `Args` value. `force_ram` is `--ram` or `--ram-path`; `ram_root_present`
/// gates the heuristic on the existence of [`RAM_ROOT`].
fn use_ram_for(no_ram: bool, force_ram: bool, prefix: &str, ram_root_present: bool) -> bool {
    if no_ram {
        return false;
    }
    if force_ram {
        return true;
    }
    if !ram_root_present {
        return false;
    }
    // Bare stem: non-empty, doesn't start with '.', no path separators.
    !prefix.is_empty()
        && !prefix.starts_with('.')
        && !prefix.contains(std::path::MAIN_SEPARATOR)
        && !prefix.contains('/')
}

/// Apply the RAM/disk policy: when [`should_use_ram`] returns true, create a
/// RAM directory and rewrite `args.output_prefix` to live inside it.
/// Always returns a [`SpillPolicy`]; when on disk it passes the prefix
/// through.  Where available memory cannot be read, the heuristic writes
/// to disk and an explicit `--ram` or `--ram-path` is refused, since pages
/// could not be spilled before memory ran out.
///
/// # Errors
/// The directory cannot be created (see [`create_ram_dir`]), or RAM output
/// was requested where available memory cannot be read.
pub(crate) fn redirect_to_ram(args: &mut Args) -> std::io::Result<(RamDirGuard, SpillPolicy)> {
    let original_prefix = args.prefix().to_owned();
    let passthrough = |prefix| {
        (
            RamDirGuard { dir: None },
            SpillPolicy::new(Targets::Disk(prefix)),
        )
    };

    if !should_use_ram(args) {
        return Ok(passthrough(original_prefix));
    }
    if available_memory().is_none() {
        if args.ram || args.ram_path.is_some() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "RAM output needs the system's available memory, which cannot be read \
                 here; write to disk with --no-ram or a path prefix",
            ));
        }
        return Ok(passthrough(original_prefix));
    }

    let dir = create_ram_dir(args.ram_path.as_deref().map(Path::new), Path::new(RAM_ROOT))?;
    let guard = RamDirGuard {
        dir: Some(dir.clone()),
    };

    // The stem of the user's prefix names the files inside the directory;
    // a prefix with no file name ("/", "") falls back to "out".
    let stem = Path::new(&original_prefix)
        .file_name()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("out");
    let ram_prefix = dir.join(stem).to_string_lossy().into_owned();

    note!(
        "rrocket: writing to RAM at {} (auto-removed on exit; use --no-ram to write to disk)",
        dir.display()
    );
    announce(&mut std::io::stdout().lock(), &dir).map_err(|e| {
        std::io::Error::new(
            e.kind(),
            format!("cannot write the RAM directory to standard output: {e}"),
        )
    })?;

    args.output_prefix = Some(ram_prefix.clone());
    Ok((
        guard,
        SpillPolicy::new(Targets::Ram {
            dir,
            ram: ram_prefix,
            disk: original_prefix,
        }),
    ))
}

/// Write `dir`, the RAM directory, as a line to `out`, standard output.  A
/// reader that closed the pipe did not want the name, so a broken pipe is
/// not an error.
///
/// # Errors
///
/// Any other write error.
fn announce(out: &mut impl std::io::Write, dir: &Path) -> std::io::Result<()> {
    match writeln!(out, "{}", dir.display()).and_then(|()| out.flush()) {
        Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
        result => result,
    }
}

/// Create the RAM directory: `explicit` (`--ram-path`) when given,
/// otherwise a fresh `rrocket-<pid>-<nanos>` under `root`.  The directory
/// is created, never adopted, since it is removed with its contents on
/// exit; and it is readable by its owner alone, since the pages of the
/// document are written into it.
///
/// # Errors
/// [`std::io::ErrorKind::AlreadyExists`] when `explicit` exists;
/// [`std::io::ErrorKind::NotFound`] naming the remedy when `root` does not;
/// any other error of the creation, naming the directory.
fn create_ram_dir(explicit: Option<&Path>, root: &Path) -> std::io::Result<PathBuf> {
    use std::io::{Error, ErrorKind};

    let dir = explicit.map_or_else(|| default_ram_dir(root), Path::to_path_buf);
    #[cfg(unix)]
    let builder = {
        use std::os::unix::fs::DirBuilderExt as _;
        let mut builder = fs::DirBuilder::new();
        let _ = builder.mode(0o700);
        builder
    };
    #[cfg(not(unix))]
    let builder = fs::DirBuilder::new();
    builder
        .create(&dir)
        .map_err(|e| match (explicit, e.kind()) {
            (Some(path), ErrorKind::AlreadyExists) => Error::new(
                e.kind(),
                format!(
                    "--ram-path {} already exists; rrocket removes the directory and \
                 everything in it on exit, so pass a path that does not exist yet",
                    path.display()
                ),
            ),
            (None, ErrorKind::NotFound) => Error::new(
                e.kind(),
                format!(
                    "--ram writes under {}, which this system does not have; \
                 pass --ram-path DIR on a RAM disk instead",
                    root.display()
                ),
            ),
            _ => Error::new(
                e.kind(),
                format!("cannot create RAM directory {}: {e}", dir.display()),
            ),
        })?;
    Ok(dir)
}

/// tmpfs mount under which the default RAM directories are created.
const RAM_ROOT: &str = "/dev/shm";

/// A default `<root>/rrocket-<pid>-<nanos>` path.  The process id and the
/// time in nanoseconds distinguish concurrent runs; a collision fails the
/// creation instead of sharing a directory.
fn default_ram_dir(root: &Path) -> PathBuf {
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    root.join(format!("rrocket-{pid}-{nanos}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ram_path_refuses_an_existing_directory_and_keeps_its_contents() {
        let parent = tempfile::tempdir().unwrap();
        let keep = parent.path().join("keep.txt");
        fs::write(&keep, b"user data").unwrap();
        let err = create_ram_dir(Some(parent.path()), Path::new(RAM_ROOT)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::AlreadyExists);
        assert!(err.to_string().contains("does not exist yet"), "{err}");
        assert!(keep.exists());
        let fresh = parent.path().join("fresh");
        assert_eq!(
            create_ram_dir(Some(&fresh), Path::new(RAM_ROOT)).unwrap(),
            fresh
        );
        assert!(fresh.is_dir());
    }

    #[test]
    fn default_dir_is_created_under_its_root_for_its_owner_alone() {
        let root = tempfile::tempdir().unwrap();
        let dir = create_ram_dir(None, root.path()).unwrap();
        assert!(dir.starts_with(root.path()), "got {}", dir.display());
        assert!(dir.is_dir());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mode = fs::metadata(&dir).unwrap().permissions().mode();
            assert_eq!(mode & 0o077, 0, "mode {mode:o}");
        }
    }

    #[test]
    fn the_directory_is_announced_and_a_closed_pipe_is_not_an_error() {
        struct Closed;
        impl std::io::Write for Closed {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::ErrorKind::BrokenPipe.into())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut out = Vec::new();
        announce(&mut out, Path::new("/ram/dir")).unwrap();
        assert_eq!(out, b"/ram/dir\n");
        announce(&mut Closed, Path::new("/ram/dir")).unwrap();
    }

    #[test]
    fn default_dir_without_its_root_names_the_remedy() {
        let root = tempfile::tempdir().unwrap();
        let missing = root.path().join("absent");
        let err = create_ram_dir(None, &missing).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        assert!(err.to_string().contains("--ram-path"), "{err}");
        assert!(!missing.exists(), "the root is not created");
    }

    #[cfg(any(target_os = "linux", target_os = "macos", windows))]
    #[test]
    fn available_memory_is_plausible() {
        let bytes = available_memory().expect("this platform has a memory estimate");
        assert!(bytes > 0, "available memory read as zero");
        assert!(
            bytes < 1024_u64.pow(5),
            "implausibly large available memory"
        );
    }

    /// A RAM policy whose cached readings are `available` and
    /// `free_space`.
    fn policy_with(available: Option<u64>, free_space: Option<u64>) -> SpillPolicy {
        let policy = SpillPolicy::new(Targets::Ram {
            dir: "/ram".into(),
            ram: "/ram/out".into(),
            disk: "/disk/out".into(),
        });
        *policy.probe.lock().unwrap() = Probe {
            last_read: Some(Instant::now()),
            available,
            free_space,
        };
        policy
    }

    #[test]
    fn unreadable_memory_spills_to_disk() {
        let policy = policy_with(None, None);
        assert_eq!(policy.next_prefix(), "/disk/out");
        assert!(policy.spilled.load(Ordering::Relaxed));
    }

    #[test]
    fn comfortable_memory_keeps_pages_in_ram() {
        assert_eq!(
            policy_with(Some(RAM_SAFETY_MARGIN), None).next_prefix(),
            "/ram/out"
        );
    }

    #[test]
    fn a_spill_is_kept_when_memory_recovers() {
        let policy = policy_with(Some(RAM_SAFETY_MARGIN - 1), None);
        assert_eq!(policy.next_prefix(), "/disk/out");
        policy.probe.lock().unwrap().available = Some(RAM_SAFETY_MARGIN);
        assert_eq!(policy.next_prefix(), "/disk/out");
    }

    /// The RAM file system fills long before memory runs out: the free
    /// space must hold the pages in flight, the size of the largest page
    /// so far, above the margin.
    #[test]
    fn a_nearly_full_ram_file_system_spills_with_memory_to_spare() {
        const MIB: u64 = 1024 * 1024;
        let plenty = Some(20 * 1024 * MIB);
        let page = 104 * MIB;
        let room = page * 12 + RAM_SPACE_MARGIN;
        assert_eq!(verdict(plenty, Some(room), page, 12), Verdict::Ram);
        assert_eq!(
            verdict(plenty, Some(room - 1), page, 12),
            Verdict::Spill("the RAM file system is nearly full")
        );
        // Before any page is written only the margin is asked for.
        assert_eq!(verdict(plenty, Some(RAM_SPACE_MARGIN), 0, 12), Verdict::Ram);
        assert!(matches!(
            verdict(plenty, Some(RAM_SPACE_MARGIN - 1), 0, 12),
            Verdict::Spill(_)
        ));
        // Unreadable free space is no reason to spill; the memory still is.
        assert_eq!(verdict(plenty, None, page, 12), Verdict::Ram);
        assert!(matches!(
            verdict(Some(RAM_SAFETY_MARGIN - 1), Some(u64::MAX), 0, 1),
            Verdict::Spill(_)
        ));
        assert_eq!(
            verdict(plenty, Some(u64::MAX - 1), u64::MAX, u64::MAX),
            Verdict::Spill("the RAM file system is nearly full")
        );
    }

    #[test]
    fn the_largest_page_written_sets_the_room_asked_for() {
        let dir = tempfile::tempdir().unwrap();
        let ram = dir.path().join("ram");
        fs::create_dir(&ram).unwrap();
        let disk = dir.path().join("disk");
        fs::create_dir(&disk).unwrap();
        let policy = SpillPolicy::new(Targets::Ram {
            dir: ram.clone(),
            ram: ram.join("p").to_string_lossy().into_owned(),
            disk: disk.join("p").to_string_lossy().into_owned(),
        });
        *policy.probe.lock().unwrap() = Probe {
            last_read: Some(Instant::now()),
            available: Some(u64::MAX),
            free_space: Some(RAM_SPACE_MARGIN + 1000),
        };
        let page = |bytes: usize| {
            move |mut out: BufWriter<File>| {
                use std::io::Write as _;
                out.write_all(&vec![0; bytes])?;
                Ok(out)
            }
        };
        let path_of = |n| move |prefix: &str| format!("{prefix}-{n}");
        let one_thread = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        one_thread.install(|| {
            policy.write_page(path_of(1), page(100)).unwrap();
            assert!(ram.join("p-1").exists());
            assert_eq!(policy.largest_page.load(Ordering::Relaxed), 100);
            // One page in flight: 100 bytes fit above the margin, 1001 not.
            policy.write_page(path_of(2), page(1001)).unwrap();
            assert!(ram.join("p-2").exists());
            policy.write_page(path_of(3), page(1)).unwrap();
            assert!(disk.join("p-3").exists(), "the largest page no longer fits");
        });
    }

    #[test]
    fn a_full_file_system_is_found_through_the_error_chain() {
        let full = || std::io::Error::from(std::io::ErrorKind::StorageFull);
        assert!(storage_full(&RenderError::Io(full())));
        assert!(storage_full(&RenderError::Encode(encode::EncodeError::Io(
            full()
        ))));
        assert!(!storage_full(&RenderError::Io(std::io::Error::other("no"))));
        assert!(!storage_full(&RenderError::Camera("no".into())));
    }

    /// A RAM write that finds the file system full is written again on
    /// disk, and so is every later page; any other failure is the page's.
    #[test]
    fn a_full_ram_file_system_spills_the_page_and_the_rest() {
        let dir = tempfile::tempdir().unwrap();
        let ram = dir.path().join("ram");
        fs::create_dir(&ram).unwrap();
        let disk = dir.path().join("disk");
        fs::create_dir(&disk).unwrap();
        let policy = SpillPolicy::new(Targets::Ram {
            dir: ram.clone(),
            ram: ram.join("p").to_string_lossy().into_owned(),
            disk: disk.join("p").to_string_lossy().into_owned(),
        });
        *policy.probe.lock().unwrap() = Probe {
            last_read: Some(Instant::now()),
            available: Some(u64::MAX),
            free_space: None,
        };
        let path_of = |n| move |prefix: &str| format!("{prefix}-{n}");
        let other = policy.write_page(path_of(1), |_| {
            Err(RenderError::Io(std::io::Error::other("encoder failed")))
        });
        assert!(other.is_err(), "a failure other than a full file system");
        assert!(!policy.spilled.load(Ordering::Relaxed));

        let mut attempts = 0;
        policy
            .write_page(path_of(2), |mut out| {
                use std::io::Write as _;
                attempts += 1;
                if attempts == 1 {
                    return Err(RenderError::Io(std::io::ErrorKind::StorageFull.into()));
                }
                out.write_all(b"page")?;
                Ok(out)
            })
            .unwrap();
        assert_eq!(attempts, 2);
        assert_eq!(fs::read(disk.join("p-2")).unwrap(), b"page");
        assert!(
            fs::read_dir(&ram).unwrap().next().is_none(),
            "RAM is left empty"
        );
        policy.write_page(path_of(3), Ok).unwrap();
        assert!(disk.join("p-3").exists());
    }

    #[test]
    fn probe_caches_within_ttl() {
        let mut probe = Probe::new();
        assert!(probe.last_read.is_none());
        probe.refresh_if_stale(Path::new("/"));
        let first = probe.last_read;
        assert!(first.is_some(), "the first check reads");
        // Immediately re-checking should NOT update last_read.
        probe.refresh_if_stale(Path::new("/"));
        assert_eq!(probe.last_read, first);
    }

    #[cfg(unix)]
    #[test]
    fn free_space_is_read_where_the_directory_exists() {
        assert!(free_space(Path::new("/")).is_some());
        assert_eq!(free_space(Path::new("/no/such/rrocket/dir")), None);
    }

    #[test]
    fn passthrough_policy_returns_original_prefix() {
        let p = SpillPolicy::new(Targets::Disk("/foo/bar".into()));
        assert_eq!(p.next_prefix(), "/foo/bar");
    }

    #[test]
    fn heuristic_bare_stem_uses_ram() {
        assert!(use_ram_for(false, false, "out", true));
        assert!(use_ram_for(false, false, "p", true));
        assert!(use_ram_for(false, false, "page", true));
    }

    #[test]
    fn heuristic_path_like_prefix_uses_disk() {
        assert!(!use_ram_for(false, false, "./out", true));
        assert!(!use_ram_for(false, false, "../out", true));
        assert!(!use_ram_for(false, false, "/tmp/out", true));
        assert!(!use_ram_for(false, false, "dir/p", true));
        assert!(!use_ram_for(false, false, ".hidden", true));
    }

    #[test]
    fn heuristic_empty_prefix_uses_disk() {
        // Defensive: an empty prefix could match the bare-stem rule by
        // accident, but every-page write would land at "-NNN.ppm" in the
        // cwd which is bizarre. Better to fall through to disk and let
        // the user see the obvious-broken behaviour.
        assert!(!use_ram_for(false, false, "", true));
    }

    #[test]
    fn explicit_ram_flag_overrides_path_heuristic() {
        // User typed --ram with a path-like prefix: respect the flag.
        assert!(use_ram_for(false, true, "./out", true));
        assert!(use_ram_for(false, true, "/abs/p", true));
    }

    #[test]
    fn explicit_no_ram_flag_overrides_bare_stem() {
        // User typed --no-ram with a bare stem: respect the flag.
        assert!(!use_ram_for(true, false, "out", true));
    }

    #[test]
    fn bare_stem_uses_disk_without_ram_root() {
        // No tmpfs mount (macOS, Windows, the BSDs): the heuristic writes where asked,
        // while an explicit --ram or --ram-path keeps its meaning.
        assert!(!use_ram_for(false, false, "out", false));
        assert!(use_ram_for(false, true, "out", false));
    }
}
