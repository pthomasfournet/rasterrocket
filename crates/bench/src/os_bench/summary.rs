//! Result summaries: per-row statistics, the environment whitelist and the
//! cross-OS table.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use crate::library::Sample;

/// Environment key/value pairs of one run.
pub(crate) type EnvMap = BTreeMap<String, String>;

/// The only keys an `env.tsv` may hold.
pub(crate) const ENV_KEYS: &[&str] = &[
    "os",
    "os_build",
    "kernel",
    "commit",
    "rustc",
    "profile",
    "cpu_model",
    "logical_cpus",
    "ram_gib",
    "power",
    "governor",
    "hags",
    "defender_rt",
    "defender_excl",
    "gpu_driver",
    "cuda",
    "fs",
    "ram_root_fs",
    "date",
];

/// Keys that must be present, and must agree between the two sides.
const REQUIRED_KEYS: &[&str] = &["os", "commit", "rustc", "profile"];
const MATCHED_KEYS: &[&str] = &["commit", "rustc", "profile"];

/// A row is noisy when (max − min) / median exceeds this.
const NOISE_LIMIT: f64 = 0.05;

/// Microsecond timings stay far below 2^53, where `f64` is exact.
#[expect(
    clippy::cast_precision_loss,
    reason = "timings in microseconds are far below 2^53"
)]
const fn to_f64(v: u64) -> f64 {
    v as f64
}

/// Drop a leading byte-order mark and split into non-empty lines (`lines`
/// already removes the CR of a CRLF ending).
fn clean_lines(text: &str) -> impl Iterator<Item = &str> {
    text.strip_prefix('\u{feff}')
        .unwrap_or(text)
        .lines()
        .filter(|l| !l.is_empty())
}

/// Parse `key\tvalue` lines.  A key outside [`ENV_KEYS`], a repeated key and a
/// missing required key are errors.
pub(crate) fn read_env(text: &str) -> Result<EnvMap, String> {
    let mut map = EnvMap::new();
    for line in clean_lines(text) {
        let (key, value) = line
            .split_once('\t')
            .ok_or_else(|| format!("env line {line:?} has no tab"))?;
        if !ENV_KEYS.contains(&key) {
            return Err(format!("env key {key:?} is not in the whitelist"));
        }
        if map.insert(key.to_owned(), value.to_owned()).is_some() {
            return Err(format!("env key {key:?} appears twice"));
        }
    }
    for key in REQUIRED_KEYS {
        if !map.contains_key(*key) {
            return Err(format!("env key {key:?} is missing"));
        }
    }
    Ok(map)
}

/// Parse `results.tsv` lines (`Sample::tsv_line` format).
pub(crate) fn read_results(text: &str) -> Result<Vec<Sample>, String> {
    clean_lines(text)
        .enumerate()
        .map(|(i, line)| {
            let f: Vec<&str> = line.split('\t').collect();
            let [group, row, doc, pages, threads, rep, wall] = f[..] else {
                return Err(format!(
                    "results line {}: expected 7 fields, got {}",
                    i + 1,
                    f.len()
                ));
            };
            let num = |s: &str, name: &str| {
                s.parse::<u64>()
                    .map_err(|_| format!("results line {}: {name} {s:?} is not a number", i + 1))
            };
            let small = |v: u64, name: &str| {
                u32::try_from(v)
                    .map_err(|_| format!("results line {}: {name} {v} out of range", i + 1))
            };
            Ok(Sample {
                group: group.to_owned(),
                row: row.to_owned(),
                doc: doc.to_owned(),
                pages: small(num(pages, "pages")?, "pages")?,
                threads: usize::try_from(num(threads, "threads")?)
                    .map_err(|_| format!("results line {}: threads out of range", i + 1))?,
                rep: small(num(rep, "rep")?, "rep")?,
                wall_us: num(wall, "wall_us")?,
            })
        })
        .collect()
}

/// Statistics of one (row, document, threads) group.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RowStat {
    pub row: String,
    pub doc: String,
    pub threads: usize,
    /// Page count, equal across the group's samples.
    pub pages: u32,
    pub reps: usize,
    pub median_us: u64,
    /// `(max − min) / median`.
    pub spread: f64,
    pub noisy: bool,
}

/// Group samples by (row, doc, threads), in that order.
pub(crate) fn row_stats(samples: &[Sample]) -> Vec<RowStat> {
    let mut groups: BTreeMap<(&str, &str, usize), Vec<&Sample>> = BTreeMap::new();
    for s in samples {
        groups
            .entry((&s.row, &s.doc, s.threads))
            .or_default()
            .push(s);
    }
    groups
        .into_iter()
        .map(|((row, doc, threads), group)| {
            let mut walls: Vec<u64> = group.iter().map(|s| s.wall_us).collect();
            walls.sort_unstable();
            let n = walls.len();
            let median_us = if n % 2 == 1 {
                walls[n / 2]
            } else {
                // Midpoint without overflow.
                let (lo, hi) = (walls[n / 2 - 1], walls[n / 2]);
                lo + (hi - lo) / 2
            };
            let range = to_f64(walls[n - 1] - walls[0]);
            let spread = if median_us == 0 {
                0.0
            } else {
                range / to_f64(median_us)
            };
            RowStat {
                row: row.to_owned(),
                doc: doc.to_owned(),
                threads,
                pages: group[0].pages,
                reps: n,
                median_us,
                spread,
                noisy: spread > NOISE_LIMIT,
            }
        })
        .collect()
}

/// Per-page cost of reading and writing files: process minus library, per
/// page; signed because the process row can be faster.
pub(crate) fn file_cost_us_per_page(process: &RowStat, library: &RowStat, pages: u32) -> i64 {
    let diff = i128::from(process.median_us) - i128::from(library.median_us);
    i64::try_from(diff / i128::from(pages.max(1))).unwrap_or(if diff < 0 {
        i64::MIN
    } else {
        i64::MAX
    })
}

/// `row\tdoc` lines of the noisy rows, each pair once.
pub(crate) fn noisy_lines(stats: &[RowStat]) -> String {
    let pairs: BTreeSet<(&str, &str)> = stats
        .iter()
        .filter(|s| s.noisy)
        .map(|s| (s.row.as_str(), s.doc.as_str()))
        .collect();
    pairs.iter().fold(String::new(), |mut out, (r, d)| {
        let _ = writeln!(out, "{r}\t{d}");
        out
    })
}

type Key<'a> = (&'a str, &'a str, usize);

fn index(stats: &[RowStat]) -> BTreeMap<Key<'_>, &RowStat> {
    stats
        .iter()
        .map(|s| ((s.row.as_str(), s.doc.as_str(), s.threads), s))
        .collect()
}

fn ms(us: u64) -> String {
    format!("{:.2} ms", to_f64(us) / 1000.0)
}

/// `windows / ubuntu`, or `—` when the Ubuntu value is zero.
fn ratio(ubuntu: u64, windows: u64) -> String {
    if ubuntu == 0 {
        "—".to_owned()
    } else {
        format!("{:.2}", to_f64(windows) / to_f64(ubuntu))
    }
}

/// The side's `S-version` row, if it has one.
fn startup(stats: &[RowStat]) -> Option<&RowStat> {
    stats.iter().find(|s| s.row == "S-version")
}

/// (document, threads) to µs per page.
type FileCosts<'a> = BTreeMap<(&'a str, usize), i64>;

/// Per-page file costs per (document, threads).  A `D12` row pairs with the
/// `L12` row of the same document and thread count.  Process start-up does
/// not depend on the thread count, so the side's single `S-version` row
/// (whatever its threads) is subtracted from `D12` before the per-page
/// division; a side with `D12`/`L12` pairs but not exactly one `S-version`
/// row, or a start-up median above the `D12` median, is an error.
fn file_costs(stats: &[RowStat]) -> Result<FileCosts<'_>, String> {
    let mut out = BTreeMap::new();
    for d in stats.iter().filter(|s| s.row == "D12") {
        let Some(l) = stats
            .iter()
            .find(|s| s.row == "L12" && s.doc == d.doc && s.threads == d.threads)
        else {
            continue;
        };
        if d.pages != l.pages {
            return Err(format!(
                "{}@{}: D12 has {} page(s), L12 has {}",
                d.doc, d.threads, d.pages, l.pages
            ));
        }
        let startups: Vec<&RowStat> = stats.iter().filter(|s| s.row == "S-version").collect();
        let [st] = startups[..] else {
            return Err(format!(
                "{}@{}: expected exactly one S-version row, found {}",
                d.doc,
                d.threads,
                startups.len()
            ));
        };
        let mut adjusted = d.clone();
        adjusted.median_us = d.median_us.checked_sub(st.median_us).ok_or_else(|| {
            format!(
                "{}@{}: start-up median {} us exceeds D12 median {} us",
                d.doc, d.threads, st.median_us, d.median_us
            )
        })?;
        out.insert(
            (d.doc.as_str(), d.threads),
            file_cost_us_per_page(&adjusted, l, d.pages),
        );
    }
    Ok(out)
}

fn same_pages(a: &[RowStat], b: &[RowStat]) -> Result<(), String> {
    let (ia, ib) = (index(a), index(b));
    for (k, sa) in &ia {
        if let Some(sb) = ib.get(k)
            && sa.pages != sb.pages
        {
            return Err(format!(
                "row {} doc {}: {} page(s) on one side, {} on the other",
                k.0, k.1, sa.pages, sb.pages
            ));
        }
    }
    Ok(())
}

/// The env of both sides, key by key.
fn render_env(out: &mut String, a: &EnvMap, b: &EnvMap) {
    out.push_str("\n| env | Ubuntu | Windows |\n|---|---|---|\n");
    let env_keys: BTreeSet<&String> = a.keys().chain(b.keys()).collect();
    for k in env_keys {
        let show = |m: &EnvMap| m.get(k).map_or("—", String::as_str).to_owned();
        let _ = writeln!(out, "| {k} | {} | {} |", show(a), show(b));
    }
}

/// Markdown comparison: column A is Ubuntu, column B is Windows.  Refuses
/// when `commit`, `rustc` or `profile` differ.
pub(crate) fn render_table(
    a: (&EnvMap, &[RowStat]),
    b: (&EnvMap, &[RowStat]),
) -> Result<String, String> {
    for key in MATCHED_KEYS {
        let (va, vb) = (a.0.get(*key), b.0.get(*key));
        if va != vb {
            return Err(format!(
                "{key} differs: {:?} against {:?}",
                va.map_or("", String::as_str),
                vb.map_or("", String::as_str)
            ));
        }
    }
    same_pages(a.1, b.1)?;
    let (ia, ib) = (index(a.1), index(b.1));
    let keys: BTreeSet<Key<'_>> = ia.keys().chain(ib.keys()).copied().collect();

    let mut out = String::new();
    out.push_str("| row | doc | Ubuntu median | Windows median | Windows/Ubuntu | flags |\n");
    out.push_str("|---|---|---:|---:|---:|---|\n");
    for k in &keys {
        let (sa, sb) = (ia.get(k), ib.get(k));
        let cell = |s: Option<&&RowStat>| s.map_or("—".to_owned(), |s| ms(s.median_us));
        let r = match (sa, sb) {
            (Some(x), Some(y)) => ratio(x.median_us, y.median_us),
            _ => "—".to_owned(),
        };
        let mut flags = Vec::new();
        if sa.is_some_and(|s| s.noisy) {
            flags.push("noisy (Ubuntu)");
        }
        if sb.is_some_and(|s| s.noisy) {
            flags.push("noisy (Windows)");
        }
        let _ = writeln!(
            out,
            "| {}@{} | {} | {} | {} | {} | {} |",
            k.0,
            k.2,
            k.1,
            cell(sa),
            cell(sb),
            r,
            flags.join(", ")
        );
    }

    out.push_str("\n**Start-up** (`S-version`): ");
    match (startup(a.1), startup(b.1)) {
        (Some(x), Some(y)) => {
            let _ = writeln!(
                out,
                "Ubuntu {}, Windows {}, ratio {}",
                ms(x.median_us),
                ms(y.median_us),
                ratio(x.median_us, y.median_us)
            );
        }
        _ => out.push_str("—\n"),
    }

    let (ca, cb) = (file_costs(a.1)?, file_costs(b.1)?);
    let docs: BTreeSet<(&str, usize)> = ca.keys().chain(cb.keys()).copied().collect();
    out.push_str("\n**Per-page file cost** (`D12` against `L12`");
    out.push_str(", start-up subtracted");
    out.push_str("):\n");
    if docs.is_empty() {
        out.push_str("—\n");
    }
    for key in &docs {
        let show = |m: &FileCosts<'_>| m.get(key).map_or("—".to_owned(), |v| format!("{v} µs"));
        let _ = writeln!(
            out,
            "- {}@{}: Ubuntu {}, Windows {}",
            key.0,
            key.1,
            show(&ca),
            show(&cb)
        );
    }

    out.push_str("\n**Render ratio** (`L12`, Windows/Ubuntu):\n");
    let mut any = false;
    for k in keys.iter().filter(|k| k.0 == "L12") {
        if let (Some(x), Some(y)) = (ia.get(k), ib.get(k)) {
            any = true;
            let _ = writeln!(
                out,
                "- {}@{}: {}",
                k.1,
                k.2,
                ratio(x.median_us, y.median_us)
            );
        }
    }
    if !any {
        out.push_str("—\n");
    }

    render_env(&mut out, a.0, b.0);
    Ok(out)
}

/// A re-run replaces the first run: every (group, row, doc, threads) key in
/// `rerun` drops that key's samples from `first`.
pub(crate) fn merge_rerun(first: Vec<Sample>, rerun: Vec<Sample>) -> Vec<Sample> {
    let key = |s: &Sample| (s.group.clone(), s.row.clone(), s.doc.clone(), s.threads);
    let replaced: BTreeSet<_> = rerun.iter().map(key).collect();
    first
        .into_iter()
        .filter(|s| !replaced.contains(&key(s)))
        .chain(rerun)
        .collect()
}

/// Read `env.tsv`, `results.tsv` and, when present, `rerun.tsv` from a result
/// directory.
pub(crate) fn read_dir(dir: &std::path::Path) -> Result<(EnvMap, Vec<RowStat>), String> {
    let read = |name: &str| {
        let path = dir.join(name);
        std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))
    };
    let parse = |name: &str, text: &str| {
        read_results(text).map_err(|e| format!("{}: {e}", dir.join(name).display()))
    };
    let env_path = dir.join("env.tsv");
    let env = read_env(&read("env.tsv")?).map_err(|e| format!("{}: {e}", env_path.display()))?;
    let mut samples = parse("results.tsv", &read("results.tsv")?)?;
    if dir.join("rerun.tsv").exists() {
        samples = merge_rerun(samples, parse("rerun.tsv", &read("rerun.tsv")?)?);
    }
    let stats = row_stats(&samples);
    let keys: BTreeSet<(&str, &str)> = samples
        .iter()
        .filter(|s| s.row == "S-version")
        .map(|s| (s.group.as_str(), s.row.as_str()))
        .collect();
    let rows = stats.iter().filter(|s| s.row == "S-version").count();
    if keys.len() != 1 || rows != 1 {
        return Err(format!(
            "{}: expected exactly one S-version key, found {} (group, row) and {rows} (row, doc, threads)",
            dir.display(),
            keys.len()
        ));
    }
    Ok((env, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(row: &str, doc: &str, pages: u32, wall_us: u64) -> Sample {
        Sample {
            group: "g".into(),
            row: row.into(),
            doc: doc.into(),
            pages,
            threads: 4,
            rep: 1,
            wall_us,
        }
    }

    fn stat(row: &str, doc: &str, pages: u32, median_us: u64) -> RowStat {
        RowStat {
            row: row.into(),
            doc: doc.into(),
            threads: 4,
            pages,
            reps: 3,
            median_us,
            spread: 0.0,
            noisy: false,
        }
    }

    const ENV: &str = "os\tlinux\ncommit\taaa\nrustc\t1.100\nprofile\trelease\n";

    fn env(commit: &str) -> EnvMap {
        read_env(&ENV.replace("aaa", commit)).unwrap()
    }

    #[test]
    fn spread_over_five_percent_is_noisy() {
        let go = |w: [u64; 3]| {
            let s: Vec<Sample> = w.iter().map(|&u| sample("r", "d", 1, u)).collect();
            row_stats(&s).remove(0)
        };
        let quiet = go([100, 103, 104]);
        assert_eq!(quiet.median_us, 103);
        assert!(!quiet.noisy);
        let loud = go([100, 104, 106]);
        assert!(loud.noisy);
        assert!((loud.spread - 6.0 / 104.0).abs() < 1e-9);
    }

    #[test]
    fn summarize_refuses_mismatched_commits() {
        let err = render_table((&env("aaa"), &[]), (&env("bbb"), &[])).unwrap_err();
        assert!(err.contains("commit") && err.contains("aaa") && err.contains("bbb"));
    }

    #[test]
    fn summarize_rejects_unlisted_env_keys() {
        let err = read_env(&format!("{ENV}hostname\tbox\n")).unwrap_err();
        assert!(err.contains("hostname"));
        assert!(read_env("os\tlinux\n").unwrap_err().contains("commit"));
    }

    #[test]
    fn read_env_tolerates_a_mark_and_crlf() {
        let plain = read_env(ENV).unwrap();
        let windows = format!("\u{feff}{}", ENV.replace('\n', "\r\n"));
        assert_eq!(read_env(&windows).unwrap(), plain);
    }

    #[test]
    fn file_cost_is_process_minus_library_per_page() {
        let p = stat("D12", "d", 16, 1_600_000);
        let l = stat("L12", "d", 16, 1_000_000);
        assert_eq!(file_cost_us_per_page(&p, &l, 16), 37_500);
        assert_eq!(file_cost_us_per_page(&l, &p, 16), -37_500);
    }

    #[test]
    fn table_lists_every_row_of_either_side() {
        let a = [stat("L12", "d", 2, 2000)];
        let b = [stat("L12", "d", 2, 3000), stat("D12-excl", "d", 2, 4000)];
        let t = render_table((&env("aaa"), &a), (&env("aaa"), &b)).unwrap();
        let line = t.lines().find(|l| l.contains("D12-excl")).unwrap();
        assert!(line.contains("| — |") && line.contains("4.00 ms"));
        assert!(t.lines().any(|l| l.contains("L12@4") && l.contains("1.50")));
    }

    fn quiet(row: &str, walls: &[u64]) -> Vec<RowStat> {
        let s: Vec<Sample> = walls.iter().map(|&u| sample(row, "d", 1, u)).collect();
        row_stats(&s)
    }

    #[test]
    fn exactly_five_percent_is_not_noisy_and_even_median_is_midpoint() {
        // Spread 5/100 is not above the limit; only "exceeds 5 %" flags.
        assert!(!quiet("r", &[100, 100, 105])[0].noisy);
        assert!(quiet("r", &[100, 100, 106])[0].noisy);
        assert_eq!(quiet("r", &[10, 20, 30, 41])[0].median_us, 25);
    }

    #[test]
    fn rustc_and_profile_mismatches_are_refused() {
        for key in ["rustc", "profile"] {
            let (a, mut b) = (env("aaa"), env("aaa"));
            b.insert(key.to_owned(), "other".to_owned());
            let err = render_table((&a, &[]), (&b, &[])).unwrap_err();
            assert!(err.contains(key) && err.contains("other"), "{err}");
        }
    }

    #[test]
    fn malformed_env_lines_are_errors() {
        let err = read_env(&format!("{ENV}os\tagain\n")).unwrap_err();
        assert!(err.contains("twice") && err.contains("os"));
        let err = read_env(&format!("{ENV}date\n")).unwrap_err();
        assert!(err.contains("no tab"));
    }

    #[test]
    fn page_count_mismatch_is_an_error() {
        let a = [stat("L12", "d", 2, 1)];
        let b = [stat("L12", "d", 3, 1)];
        let err = render_table((&env("aaa"), &a), (&env("aaa"), &b)).unwrap_err();
        assert!(err.contains("2 page") && err.contains('3'));
    }

    #[test]
    fn noisy_lines_list_each_pair_once() {
        let mut v = vec![stat("r", "d", 1, 1), stat("q", "e", 1, 1)];
        v[0].noisy = true;
        let mut w = v[0].clone();
        w.threads = 8;
        v.push(w);
        assert_eq!(noisy_lines(&v), "r\td\n");
    }

    #[test]
    fn rerun_replaces_the_key_and_leaves_others() {
        let first = vec![
            sample("r", "d", 1, 100),
            sample("r", "d", 1, 900),
            sample("q", "d", 1, 5),
        ];
        let rerun = vec![sample("r", "d", 1, 200)];
        let merged = merge_rerun(first, rerun);
        let stats = row_stats(&merged);
        let r = stats.iter().find(|s| s.row == "r").unwrap();
        assert_eq!((r.reps, r.median_us), (1, 200));
        assert_eq!(stats.iter().find(|s| s.row == "q").unwrap().median_us, 5);
    }

    #[test]
    fn startup_checks_and_threads_matching() {
        let side = [
            stat("S-version", "-", 1, 2_000),
            stat("D12", "d", 2, 1_000),
            stat("L12", "d", 2, 500),
        ];
        let err = render_table((&env("aaa"), &side), (&env("aaa"), &side)).unwrap_err();
        assert!(err.contains("2000") && err.contains("1000"));
        // Start-up is matched whatever its threads (here 1 against 4).
        let mut ok = side.clone();
        ok[0].median_us = 100;
        let t = render_table((&env("aaa"), &ok), (&env("aaa"), &ok)).unwrap();
        assert!(t.contains("- d@4: Ubuntu 200 µs"), "{t}");
        // An L12 at other threads is not a partner: no cost line.
        let mut other = ok.clone();
        other[2].threads = 8;
        let t = render_table((&env("aaa"), &other), (&env("aaa"), &other)).unwrap();
        assert!(!t.contains("- d@4"));
        // No S-version, or two, beside a D12/L12 pair is an error.
        let none = &ok[1..];
        let err = render_table((&env("aaa"), none), (&env("aaa"), &ok)).unwrap_err();
        assert!(err.contains("exactly one S-version"));
        let mut two = ok.to_vec();
        two.push(stat("S-version", "-", 1, 5));
        two[3].threads = 9;
        assert!(render_table((&env("aaa"), &two), (&env("aaa"), &ok)).is_err());
    }

    #[test]
    fn file_cost_line_subtracts_startup() {
        let side = [
            stat("S-version", "-", 1, 100_000),
            stat("D12", "d", 10, 1_100_000),
            stat("L12", "d", 10, 900_000),
        ];
        let t = render_table((&env("aaa"), &side), (&env("aaa"), &side)).unwrap();
        assert!(t.contains("- d@4: Ubuntu 10000 µs, Windows 10000 µs"));
    }

    #[test]
    fn results_round_trip_and_bad_lines_are_named() {
        let s = sample("r", "d", 3, 9);
        let back = read_results(&format!("\u{feff}{}\r\n", s.tsv_line())).unwrap();
        assert_eq!(back[0].wall_us, 9);
        assert!(read_results("a\tb\n").unwrap_err().contains("line 1"));
    }

    /// A scratch result directory under the target tree, removed on drop.
    struct Scratch(std::path::PathBuf);

    impl Scratch {
        fn new(name: &str) -> Self {
            let dir = std::path::PathBuf::from(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../target/os-bench-test"
            ))
            .join(name);
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("env.tsv"), ENV).unwrap();
            Self(dir)
        }

        fn results(&self, lines: &[&str]) -> &std::path::Path {
            std::fs::write(self.0.join("results.tsv"), lines.join("\n")).unwrap();
            &self.0
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn read_dir_needs_exactly_one_startup_key() {
        let rows = |extra: &[&'static str]| {
            let mut v = vec!["L\tL12\td\t2\t4\t1\t500"];
            v.extend_from_slice(extra);
            v
        };
        let zero = Scratch::new("read_dir_zero");
        let err = read_dir(zero.results(&rows(&[]))).unwrap_err();
        assert!(err.contains("read_dir_zero") && err.contains("exactly one S-version"));

        let one = Scratch::new("read_dir_one");
        let (_, stats) = read_dir(one.results(&rows(&[
            "S\tS-version\t-\t1\t1\t1\t90",
            "S\tS-version\t-\t1\t1\t2\t100",
        ])))
        .unwrap();
        assert_eq!(stats.iter().filter(|s| s.row == "S-version").count(), 1);

        let two = Scratch::new("read_dir_two");
        let err = read_dir(two.results(&rows(&[
            "S\tS-version\t-\t1\t1\t1\t90",
            "T\tS-version\t-\t1\t1\t1\t90",
        ])))
        .unwrap_err();
        assert!(err.contains("read_dir_two") && err.contains("2 (group, row)"));

        let threads = Scratch::new("read_dir_two_threads");
        let err = read_dir(threads.results(&rows(&[
            "S\tS-version\t-\t1\t1\t1\t90",
            "S\tS-version\t-\t1\t2\t1\t90",
        ])))
        .unwrap_err();
        assert!(err.contains("2 (row, doc, threads)"));
    }
}
