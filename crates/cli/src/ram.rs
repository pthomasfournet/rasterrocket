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
//! The default applies only where `/dev/shm` exists; elsewhere (macOS, the
//! BSDs) a bare stem is written to the working directory like any prefix,
//! and RAM output needs `--ram-path` on a RAM disk the user has mounted.
//!
//! The redirect is suppressed when `OUTPUT_PREFIX` looks like a path
//! (contains `/`, starts with `.`) — those are taken literally as the user
//! asking for a real disk location. See [`should_use_ram`] for the rule.
//!
//! Flags:
//! - `--no-ram` forces disk regardless of prefix shape
//! - `--ram` forces RAM regardless of prefix shape
//! - `--ram-path <PATH>` overrides the default tmpfs location and implies RAM
//!
//! # What this module returns
//!
//! 1. A [`RamDirGuard`] that removes the tmpfs dir on drop (no-op when on disk).
//! 2. A [`SpillPolicy`] queried per-page: returns the RAM prefix while free
//!    memory is comfortable, and the disk prefix from the first page that
//!    finds it tight.
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
//! disk if the kernel reports tight memory — at which point we've already
//! benefited from RAM-speed writes for the early pages.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::args::Args;

/// Minimum free RAM (bytes) we insist remains after the next write. Below
/// this, the writer spills to disk. 1 GiB is enough headroom for one more
/// in-flight page bitmap (worst-case ~50 MB at 600 DPI A3) plus normal
/// per-thread scratch and OS workings.
const RAM_SAFETY_MARGIN: u64 = 1024 * 1024 * 1024;

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
            eprintln!(
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
    /// The cached available-memory reading, taken under the lock by the
    /// first caller after it expires.
    probe: Mutex<MemoryProbe>,
    /// Set by the first page sent to disk; every later page goes there too.
    spilled: AtomicBool,
}

/// Where a [`SpillPolicy`] writes pages.
enum Targets {
    /// Every page to the user's prefix: RAM output is off.
    Disk(String),
    /// Pages to `ram` while memory allows, then to `disk`.
    Ram { ram: String, disk: String },
}

impl SpillPolicy {
    /// A policy writing pages to `targets`.
    const fn new(targets: Targets) -> Self {
        Self {
            targets,
            probe: Mutex::new(MemoryProbe::new()),
            spilled: AtomicBool::new(false),
        }
    }

    /// The prefix for the next page write: the user's prefix when RAM
    /// output is off; otherwise the RAM prefix until available memory is
    /// below [`RAM_SAFETY_MARGIN`] or unreadable, and the disk prefix from
    /// then on.
    #[must_use]
    pub(crate) fn next_prefix(&self) -> &str {
        let (ram, disk) = match &self.targets {
            Targets::Disk(prefix) => return prefix,
            Targets::Ram { ram, disk } => (ram, disk),
        };
        if self.spilled.load(Ordering::Relaxed) {
            return disk;
        }
        let available = self.available_bytes();
        if available.is_some_and(|bytes| bytes >= RAM_SAFETY_MARGIN) {
            return ram;
        }
        if !self.spilled.swap(true, Ordering::Relaxed) {
            let reason = if available.is_some() {
                "free memory dropped below safety margin"
            } else {
                "free memory can no longer be read"
            };
            eprintln!("rrocket: {reason}; writing subsequent pages to disk with prefix {disk}");
        }
        disk
    }

    /// Available memory in bytes, re-probed at most once per [`PROBE_TTL`];
    /// `None` when the platform offers no reading.
    fn available_bytes(&self) -> Option<u64> {
        let mut probe = self
            .probe
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        probe.refresh_if_stale();
        probe.last_available_bytes
    }
}

/// Cached [`available_memory`] reading. Refreshed on TTL.
struct MemoryProbe {
    /// When `last_available_bytes` was read; `None` before the first read.
    last_read: Option<Instant>,
    /// The last reading; `None` when the platform offers none.
    last_available_bytes: Option<u64>,
}

impl MemoryProbe {
    const fn new() -> Self {
        Self {
            last_read: None,
            last_available_bytes: None,
        }
    }

    fn refresh_if_stale(&mut self) {
        if self.last_read.is_some_and(|t| t.elapsed() < PROBE_TTL) {
            return;
        }
        self.last_available_bytes = available_memory();
        self.last_read = Some(Instant::now());
    }
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

/// No available-memory estimate on this platform.
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
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
///    (it does not on macOS or the BSDs, nor in some containers):
///    - bare stem (e.g. `out`, `p`) → RAM (default)
///    - path-like (`./out`, `dir/p`, `/abs/path`) → disk
///
/// Path-like means the prefix contains a path separator or starts with `.`
/// (covers `./out`, `../out`, and absolute paths). The intent: a user who
/// types `rrocket doc.pdf out` gets the fast path; a user who types
/// `rrocket doc.pdf ./out` or `rrocket doc.pdf /data/x` gets the
/// path they asked for.
fn should_use_ram(args: &Args) -> bool {
    use_ram_for(
        args.no_ram,
        args.ram || args.ram_path.is_some(),
        &args.output_prefix,
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
    let original_prefix = args.output_prefix.clone();
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

    // The stem of the user's prefix names the files inside the directory;
    // a prefix with no file name ("/", "") falls back to "out".
    let stem = Path::new(&original_prefix)
        .file_name()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("out");
    let ram_prefix = dir.join(stem).to_string_lossy().into_owned();

    eprintln!(
        "rrocket: writing to RAM at {} (auto-removed on exit; use --no-ram to write to disk)",
        dir.display()
    );
    println!("{}", dir.display());

    args.output_prefix.clone_from(&ram_prefix);
    Ok((
        RamDirGuard { dir: Some(dir) },
        SpillPolicy::new(Targets::Ram {
            ram: ram_prefix,
            disk: original_prefix,
        }),
    ))
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
    let mut builder = fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt as _;
        let _ = builder.mode(0o700);
    }
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
    fn default_dir_without_its_root_names_the_remedy() {
        let root = tempfile::tempdir().unwrap();
        let missing = root.path().join("absent");
        let err = create_ram_dir(None, &missing).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        assert!(err.to_string().contains("--ram-path"), "{err}");
        assert!(!missing.exists(), "the root is not created");
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn available_memory_is_plausible() {
        let bytes = available_memory().expect("this platform has a memory estimate");
        assert!(bytes > 0, "available memory read as zero");
        assert!(
            bytes < 1024_u64.pow(5),
            "implausibly large available memory"
        );
    }

    /// A RAM policy whose cached memory reading is `reading`.
    fn policy_with(reading: Option<u64>) -> SpillPolicy {
        let policy = SpillPolicy::new(Targets::Ram {
            ram: "/ram/out".into(),
            disk: "/disk/out".into(),
        });
        *policy.probe.lock().unwrap() = MemoryProbe {
            last_read: Some(Instant::now()),
            last_available_bytes: reading,
        };
        policy
    }

    #[test]
    fn unreadable_memory_spills_to_disk() {
        let policy = policy_with(None);
        assert_eq!(policy.next_prefix(), "/disk/out");
        assert!(policy.spilled.load(Ordering::Relaxed));
    }

    #[test]
    fn comfortable_memory_keeps_pages_in_ram() {
        assert_eq!(
            policy_with(Some(RAM_SAFETY_MARGIN)).next_prefix(),
            "/ram/out"
        );
    }

    #[test]
    fn a_spill_is_kept_when_memory_recovers() {
        let policy = policy_with(Some(RAM_SAFETY_MARGIN - 1));
        assert_eq!(policy.next_prefix(), "/disk/out");
        policy.probe.lock().unwrap().last_available_bytes = Some(RAM_SAFETY_MARGIN);
        assert_eq!(policy.next_prefix(), "/disk/out");
    }

    #[test]
    fn probe_caches_within_ttl() {
        let mut probe = MemoryProbe::new();
        assert!(probe.last_read.is_none());
        probe.refresh_if_stale();
        let first = probe.last_read;
        assert!(first.is_some(), "the first check reads");
        // Immediately re-checking should NOT update last_read.
        probe.refresh_if_stale();
        assert_eq!(probe.last_read, first);
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
        // No tmpfs mount (macOS, the BSDs): the heuristic writes where asked,
        // while an explicit --ram or --ram-path keeps its meaning.
        assert!(!use_ram_for(false, false, "out", false));
        assert!(use_ram_for(false, true, "out", false));
    }
}
