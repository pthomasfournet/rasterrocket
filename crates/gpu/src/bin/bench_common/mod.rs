//! Fixtures and command-line handling shared by the GPU benchmarks.

use std::str::FromStr;

/// Print `message` and `usage` to standard error and exit with status 2.
pub(crate) fn usage_error(usage: &str, message: &str) -> ! {
    eprintln!("{message}\nusage: {usage}");
    std::process::exit(2);
}

/// The `--flag VALUE` pairs of the command line.  An argument that is not a
/// flag named in `known`, or a flag without a value, is a usage error.
pub(crate) fn flag_pairs(usage: &str, known: &[&str]) -> Vec<(String, String)> {
    let mut args = std::env::args().skip(1);
    let mut pairs = Vec::new();
    while let Some(flag) = args.next() {
        if !known.contains(&flag.as_str()) {
            usage_error(usage, &format!("unknown argument {flag}"));
        }
        let Some(value) = args.next() else {
            usage_error(usage, &format!("{flag} needs a value"));
        };
        pairs.push((flag, value));
    }
    pairs
}

/// `value` of `flag` parsed as a `T`; an unparsable value is a usage error.
pub(crate) fn parse_value<T: FromStr>(usage: &str, flag: &str, value: &str) -> T {
    value
        .parse()
        .unwrap_or_else(|_| usage_error(usage, &format!("{flag}: invalid value {value}")))
}

/// `value` when it names one of `benches`; any other name is a usage error.
pub(crate) fn only_value(usage: &str, value: String, benches: &[&str]) -> String {
    if !benches.contains(&value.as_str()) {
        usage_error(
            usage,
            &format!("--only {value}: expected one of {}", benches.join(", ")),
        );
    }
    value
}

/// `n` CMYK pixels of a 2000-pixel-wide image whose channels vary smoothly
/// along x, y and the diagonals: neighbouring pixels share CLUT cells, as
/// in photographs and scans.
pub(crate) fn smooth_cmyk(n: usize) -> Vec<u8> {
    let width = 2000;
    let height = n.div_ceil(width);
    (0..n)
        .flat_map(|i| {
            let (col, row) = (i % width, i / width);
            let cyan = col * 255 / width;
            let magenta = row * 255 / height;
            #[expect(clippy::cast_possible_truncation, reason = "each value is at most 255")]
            [
                cyan as u8,
                magenta as u8,
                cyan.midpoint(magenta) as u8,
                ((col + 2 * row) % 256) as u8,
            ]
        })
        .collect()
}
