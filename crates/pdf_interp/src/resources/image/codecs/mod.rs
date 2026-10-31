//! Compressed-image codec decoders: CCITT, DCT (JPEG), JPX (JPEG 2000),
//! JBIG2, one module each.
//!
//! The decoders are called from the filter dispatch (`super::dispatch`).
//! Under `cfg(fuzzing)`, `decode_ccitt` and `decode_jbig2` are additionally
//! reachable through the `fuzz_entry` wrappers in `image/mod.rs`, which
//! bridge the visibility gap without widening it in normal builds.

mod ccitt;
mod dct;
mod jbig2;
mod jpx;

pub(super) use ccitt::decode_ccitt;
#[cfg(feature = "cache")]
pub(super) use dct::DctCacheCtx;
pub(super) use dct::decode_dct;
pub(super) use jbig2::decode_jbig2;
pub(super) use jpx::{JpxOptions, decode_jpx};
