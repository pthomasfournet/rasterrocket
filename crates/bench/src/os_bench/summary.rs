//! Result summaries: per-row statistics, the environment whitelist and the
//! cross-OS table.

#![expect(
    clippy::cast_precision_loss,
    reason = "microsecond timings stay far below 2^52, where f64 is exact"
)]

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

/// Drop a leading byte-order mark and split into lines without CR.
fn clean_lines(text: &str) -> impl Iterator<Item = &str> {
    text.strip_prefix('\u{feff}')
        .unwrap_or(text)
        .lines()
        .map(|l| l.trim_end_matches('\r'))
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
            let range = (walls[n - 1] - walls[0]) as f64;
            let spread = if median_us == 0 {
                0.0
            } else {
                range / median_us as f64
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
    format!("{:.2} ms", us as f64 / 1000.0)
}

fn ratio(a: u64, b: u64) -> String {
    if a == 0 {
        "—".to_owned()
    } else {
        format!("{:.2}", b as f64 / a as f64)
    }
}

/// The side's `S-version` median, if it has one.
fn startup(stats: &[RowStat]) -> Option<&RowStat> {
    stats.iter().find(|s| s.row == "S-version")
}

/// Per-page file costs per document, start-up subtracted when known.
fn file_costs(stats: &[RowStat]) -> Result<BTreeMap<&str, i64>, String> {
    let start = startup(stats).map_or(0, |s| s.median_us);
    let mut out = BTreeMap::new();
    for d in stats.iter().filter(|s| s.row == "D12") {
        let Some(l) = stats.iter().find(|s| s.row == "L12" && s.doc == d.doc) else {
            continue;
        };
        if d.pages != l.pages {
            return Err(format!(
                "{}: D12 has {} page(s), L12 has {}",
                d.doc, d.pages, l.pages
            ));
        }
        let mut adjusted = d.clone();
        adjusted.median_us = d.median_us.saturating_sub(start);
        out.insert(d.doc.as_str(), file_cost_us_per_page(&adjusted, l, d.pages));
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
    let docs: BTreeSet<&str> = ca.keys().chain(cb.keys()).copied().collect();
    out.push_str("\n**Per-page file cost** (`D12` against `L12`");
    if startup(a.1).is_none() || startup(b.1).is_none() {
        out.push_str("; start-up not subtracted on a side without `S-version`");
    }
    out.push_str("):\n");
    if docs.is_empty() {
        out.push_str("—\n");
    }
    for d in &docs {
        let show =
            |m: &BTreeMap<&str, i64>| m.get(d).map_or("—".to_owned(), |v| format!("{v} µs"));
        let _ = writeln!(out, "- {d}: Ubuntu {}, Windows {}", show(&ca), show(&cb));
    }

    out.push_str("\n**Render ratio** (`L12`, Windows/Ubuntu):\n");
    let mut any = false;
    for k in keys.iter().filter(|k| k.0 == "L12") {
        if let (Some(x), Some(y)) = (ia.get(k), ib.get(k)) {
            any = true;
            let _ = writeln!(out, "- {}: {}", k.1, ratio(x.median_us, y.median_us));
        }
    }
    if !any {
        out.push_str("—\n");
    }

    out.push_str("\n| env | Ubuntu | Windows |\n|---|---|---|\n");
    let env_keys: BTreeSet<&String> = a.0.keys().chain(b.0.keys()).collect();
    for k in env_keys {
        let show = |m: &EnvMap| m.get(k).map_or("—", String::as_str).to_owned();
        let _ = writeln!(out, "| {k} | {} | {} |", show(a.0), show(b.0));
    }
    Ok(out)
}

/// Read `env.tsv` and `results.tsv` from a result directory.
pub(crate) fn read_dir(dir: &std::path::Path) -> Result<(EnvMap, Vec<RowStat>), String> {
    let read = |name: &str| {
        let path = dir.join(name);
        std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))
    };
    let env_path = dir.join("env.tsv");
    let env = read_env(&read("env.tsv")?).map_err(|e| format!("{}: {e}", env_path.display()))?;
    let res_path = dir.join("results.tsv");
    let samples =
        read_results(&read("results.tsv")?).map_err(|e| format!("{}: {e}", res_path.display()))?;
    Ok((env, row_stats(&samples)))
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

    #[test]
    fn file_cost_line_subtracts_startup() {
        let side = [
            stat("S-version", "-", 1, 100_000),
            stat("D12", "d", 10, 1_100_000),
            stat("L12", "d", 10, 900_000),
        ];
        let t = render_table((&env("aaa"), &side), (&env("aaa"), &side)).unwrap();
        assert!(t.contains("- d: Ubuntu 10000 µs, Windows 10000 µs"));
    }

    #[test]
    fn results_round_trip_and_bad_lines_are_named() {
        let s = sample("r", "d", 3, 9);
        let back = read_results(&format!("\u{feff}{}\r\n", s.tsv_line())).unwrap();
        assert_eq!(back[0].wall_us, 9);
        assert!(read_results("a\tb\n").unwrap_err().contains("line 1"));
    }
}
