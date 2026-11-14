//! Property tests: the object parser against a reference serialiser, and
//! the stream filters against reference encoders.
//!
//! Each property runs the case count given to [`config`] unless
//! `PROPTEST_CASES` sets another.

mod filters;
mod objects;

#[path = "../../../../tests/support/proptest_config.rs"]
mod proptest_config;
use proptest_config::config;
