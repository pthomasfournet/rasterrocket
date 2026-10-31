//! Parallel-Huffman JPEG decoder for the GPU compute path.
//!
//! Algorithm: Weißenberger & Schmidt, "A massively parallel Huffman
//! decoder" (TPDS 2021), §III–IV.
//!
//! This module is the orchestrator; GPU kernels (when they land)
//! live under `crates/gpu/kernels/jpeg/`. Today the surface is the
//! CPU-side primitives every backend will share: bitstream packing,
//! 2-tier codetable construction, a scalar reference decoder, and a
//! synthetic-stream oracle used by cross-backend bit-identity tests.

mod bitstream;
#[cfg(test)]
mod codetable;
mod cpu_prepass;
#[cfg(test)]
mod cpu_reference;
#[cfg(feature = "gpu-jpeg-huffman")]
pub(crate) mod decoder;
#[cfg(feature = "gpu-jpeg-huffman")]
pub(crate) mod device_image;
mod dispatch_util;
mod error;
#[cfg(feature = "gpu-jpeg-huffman")]
pub(crate) mod huffman;
#[cfg(all(test, feature = "gpu-jpeg-huffman", feature = "gpu-validation"))]
mod huffman_bench;
mod jpeg_framing;
pub(crate) mod phase1_oracle;
#[cfg(test)]
mod phase2_oracle;
mod scan;
#[cfg(feature = "gpu-jpeg-huffman")]
pub(crate) mod subsequence_size;

pub(crate) use bitstream::{PackedBitstream, pack_be_words};
#[cfg(test)]
pub(crate) use cpu_prepass::prepare_jpeg;
pub(crate) use cpu_prepass::{JpegPreparedInput, build_mcu_schedule};
#[cfg(feature = "gpu-jpeg-huffman")]
pub use decoder::JpegGpuDecoder;
#[cfg(feature = "gpu-jpeg-huffman")]
pub use device_image::DeviceImage;
pub use error::JpegGpuError;
#[cfg(test)]
pub(crate) use jpeg_framing::decode_scan_symbols;
#[cfg(feature = "gpu-jpeg-huffman")]
pub(crate) use subsequence_size::pick_subsequence_size;

#[cfg(test)]
pub(crate) mod tests {
    pub(crate) mod fixtures;
    pub(crate) mod synthetic;
}
