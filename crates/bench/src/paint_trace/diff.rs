//! The differences between two paint traces, by operation.

use std::collections::BTreeSet;

/// The most cells of the alignment table: past it, the differing middle
/// of the traces is compared line by line.
const MAX_CELLS: usize = 1 << 25;

/// The device quantum of a trace's coordinates.
const QUANTUM: f64 = 1.0 / 256.0;

/// What a line aligns by: its indentation, its first word, and a
/// resource that word names.
fn key(line: &str) -> String {
    let depth = line.len() - line.trim_start().len();
    let mut words = line.trim_start().split(' ');
    let kind = words.next().unwrap_or("");
    let resource = words
        .next()
        .filter(|w| w.starts_with('/') || w.starts_with('('))
        .unwrap_or("");
    format!("{depth} {kind} {resource}")
}

/// A piece of a line: text, or a number.
#[derive(PartialEq)]
enum Piece<'a> {
    Text(&'a str),
    Number(f64),
    Hash,
}

/// The pieces of `line`, split at its delimiters: a 16-digit hexadecimal
/// name is a hash, a leading number of a piece a number.
fn pieces(line: &str) -> Vec<Piece<'_>> {
    let mut out = Vec::new();
    for word in line.split(|c: char| " ,=<>[]()|".contains(c)) {
        if word.len() == 16 && word.bytes().all(|b| b.is_ascii_hexdigit()) {
            out.push(Piece::Hash);
            continue;
        }
        let start = word
            .find(|c: char| c.is_ascii_digit() || c == '-')
            .unwrap_or(word.len());
        let (head, tail) = word.split_at(start);
        match tail.parse::<f64>() {
            Ok(v) if !tail.is_empty() => {
                out.push(Piece::Text(head));
                out.push(Piece::Number(v));
            }
            _ => out.push(Piece::Text(word)),
        }
    }
    out
}

/// Whether `a` and `b` differ only in hashes and in numbers that moved by
/// at most one quantum: a difference of the last bits of the arithmetic,
/// not of the operation.
fn quantum_only(a: &str, b: &str) -> bool {
    let (a, b) = (pieces(a), pieces(b));
    a.len() == b.len()
        && a.iter().zip(&b).all(|pair| match pair {
            (Piece::Number(x), Piece::Number(y)) => (x - y).abs() <= QUANTUM * 1.000_001,
            (x, y) => x == y,
        })
}

/// The attributes (`key=value` words, and `pts`) whose values differ
/// between `a` and `b`.
fn changed(a: &str, b: &str) -> Vec<String> {
    let attrs = |line: &str| -> Vec<(String, String)> {
        line.split(' ')
            .filter_map(|w| w.split_once('='))
            .map(|(k, v)| (k.to_owned(), v.to_owned()))
            .collect()
    };
    let (a, b) = (attrs(a), attrs(b));
    let keys: BTreeSet<&str> = a.iter().chain(&b).map(|(k, _)| k.as_str()).collect();
    keys.into_iter()
        .filter(|k| {
            let of = |l: &[(String, String)]| {
                l.iter()
                    .filter(|(key, _)| key == k)
                    .map(|(_, v)| v.clone())
                    .collect::<Vec<_>>()
            };
            of(&a) != of(&b)
        })
        .map(str::to_owned)
        .collect()
}

/// One step of an alignment.
enum Step {
    Same(usize, usize),
    Delete(usize),
    Insert(usize),
}

/// The alignment of `old` and `new` by [`key`], as the longest common
/// subsequence of keys.
fn align(old: &[&str], new: &[&str]) -> Vec<Step> {
    let (n, m) = (old.len(), new.len());
    if n.saturating_mul(m) > MAX_CELLS {
        eprintln!(
            "paint_trace: {n} × {m} differing lines exceed the alignment table; \
             comparing them line by line"
        );
        let mut steps: Vec<Step> = (0..n.min(m)).map(|i| Step::Same(i, i)).collect();
        steps.extend((m..n).map(Step::Delete));
        steps.extend((n..m).map(Step::Insert));
        return steps;
    }
    let (ka, kb): (Vec<String>, Vec<String>) = (
        old.iter().map(|l| key(l)).collect(),
        new.iter().map(|l| key(l)).collect(),
    );
    // `table[i][j]`: the longest common subsequence of `ka[i..]`, `kb[j..]`.
    let mut table = vec![vec![0_u32; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            table[i][j] = if ka[i] == kb[j] {
                table[i + 1][j + 1] + 1
            } else {
                table[i + 1][j].max(table[i][j + 1])
            };
        }
    }
    let (mut i, mut j) = (0, 0);
    let mut steps = Vec::new();
    while i < n || j < m {
        if i < n && j < m && ka[i] == kb[j] {
            steps.push(Step::Same(i, j));
            (i, j) = (i + 1, j + 1);
        } else if j < m && (i == n || table[i][j + 1] >= table[i + 1][j]) {
            steps.push(Step::Insert(j));
            j += 1;
        } else {
            steps.push(Step::Delete(i));
            i += 1;
        }
    }
    steps
}

/// Print the operations in which the traces `a` and `b` differ;
/// `true` when they are identical.
pub fn run(a: &str, b: &str) -> bool {
    let (la, lb): (Vec<&str>, Vec<&str>) = (a.lines().collect(), b.lines().collect());
    let prefix = la.iter().zip(&lb).take_while(|(x, y)| x == y).count();
    let suffix = la[prefix..]
        .iter()
        .rev()
        .zip(lb[prefix..].iter().rev())
        .take_while(|(x, y)| x == y)
        .count();
    let (ma, mb) = (
        &la[prefix..la.len() - suffix],
        &lb[prefix..lb.len() - suffix],
    );
    if ma.is_empty() && mb.is_empty() {
        println!("the traces are identical");
        return true;
    }
    let (mut changes, mut quanta) = (0_usize, 0_usize);
    for step in align(ma, mb) {
        match step {
            Step::Same(i, j) if ma[i] == mb[j] => {}
            Step::Same(i, j) => {
                changes += 1;
                let quantum = quantum_only(ma[i], mb[j]);
                quanta += usize::from(quantum);
                println!(
                    "~ A:{} B:{} {}{}",
                    prefix + i + 1,
                    prefix + j + 1,
                    changed(ma[i], mb[j]).join(","),
                    if quantum { " (quantum only)" } else { "" }
                );
                println!("  - {}", ma[i].trim_start());
                println!("  + {}", mb[j].trim_start());
            }
            Step::Delete(i) => {
                changes += 1;
                println!("- A:{} {}", prefix + i + 1, ma[i].trim_start());
            }
            Step::Insert(j) => {
                changes += 1;
                println!("+ B:{} {}", prefix + j + 1, mb[j].trim_start());
            }
        }
    }
    println!("{changes} operations differ, {quanta} of them by one quantum only");
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_one_quantum_move_is_quantum_only() {
        let a = "fill nz bbox=10,20,30,40 h=0123456789abcdef pts=m10,20 l30,40";
        let b = "fill nz bbox=10.00390625,20,30,40 h=fedcba9876543210 pts=m10.00390625,20 l30,40";
        assert!(quantum_only(a, b));
        assert_eq!(changed(a, b), ["bbox", "h", "pts"]);
        let c = "fill nz bbox=11,20,30,40 h=fedcba9876543210 pts=m11,20 l30,40";
        assert!(!quantum_only(a, c));
        assert!(!quantum_only(
            a,
            "fill eo bbox=10,20,30,40 h=0123456789abcdef pts=m10,20 l30,40"
        ));
    }

    #[test]
    fn an_inserted_operation_keeps_the_rest_aligned() {
        let a = ["fill a", "stroke b", "sh /S"];
        let b = ["fill a", "clip x", "stroke b", "sh /S"];
        let steps = align(&a, &b);
        assert!(matches!(
            steps.as_slice(),
            [
                Step::Same(0, 0),
                Step::Insert(1),
                Step::Same(1, 2),
                Step::Same(2, 3)
            ]
        ));
    }
}
