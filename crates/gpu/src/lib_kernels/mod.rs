//! Per-kernel implementations split out of `lib.rs`.
//!
//! Each module owns a single GPU kernel's host-side launch logic. The CUDA
//! backend's `record_*` methods call into these modules; tests call the
//! same `pub fn` entry points directly.

pub(crate) mod aa;
pub(crate) mod composite;
#[cfg(feature = "gpu-jpeg-huffman")]
pub(crate) mod huffman;
pub(crate) mod icc;
#[cfg(feature = "gpu-jpeg-huffman")]
pub(crate) mod idct;
#[cfg(feature = "gpu-jpeg-huffman")]
pub(crate) mod scan;
pub(crate) mod soft_mask;
