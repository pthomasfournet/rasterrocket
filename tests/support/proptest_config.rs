//! The runner configuration of the workspace's property suites, included
//! into each through `#[path]`.

use proptest::test_runner::Config;

/// The runner configuration: `cases` cases, or the count `PROPTEST_CASES`
/// sets.
///
/// # Panics
///
/// Panics if `PROPTEST_CASES` is set to anything but an unsigned integer.
#[must_use]
pub(super) fn config(cases: u32) -> Config {
    let cases = match std::env::var("PROPTEST_CASES") {
        Ok(v) => v
            .parse()
            .unwrap_or_else(|_| panic!("PROPTEST_CASES must be an unsigned integer, got {v:?}")),
        Err(std::env::VarError::NotPresent) => cases,
        Err(std::env::VarError::NotUnicode(v)) => panic!(
            "PROPTEST_CASES must be an unsigned integer, got \"{}\"",
            v.display()
        ),
    };
    Config {
        cases,
        ..Config::default()
    }
}
