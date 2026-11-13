//! Property tests: the object parser against a reference serialiser, and
//! the stream filters against reference encoders.
//!
//! Each property runs the case count given to [`config`] unless
//! `PROPTEST_CASES` sets another.

mod filters;
mod objects;

use proptest::test_runner::Config;

/// The runner configuration: `cases` cases, or `PROPTEST_CASES` when set.
fn config(cases: u32) -> Config {
    let cases = std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(cases);
    Config {
        cases,
        ..Config::default()
    }
}
