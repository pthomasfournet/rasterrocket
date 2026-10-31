//! JPEG support — shared parsing primitives consumed by VA-API and the
//! on-GPU decoder.
//!
//! | Module | Responsibility |
//! |---|---|
//! | [`headers`] | Walk a JPEG marker stream; extract DQT, DHT, SOF, SOS, DRI. |
//! | [`canonical`] | Build a canonical Huffman lookup table from the DHT-form `(num_codes, values)`. |
//! | [`unstuff`] | Strip `0xFF 0x00 → 0xFF` byte-stuffing; record RST marker positions. |
//! | [`dc_chain`] | Resolve per-block absolute DC values across the entropy stream. |
//! | [`prepass`] | Top-level orchestrator: bytes → [`CpuPrepassOutput`]. |

#[cfg(feature = "gpu-jpeg-huffman")]
pub(crate) mod bitreader;
#[cfg(feature = "gpu-jpeg-huffman")]
pub(crate) mod canonical;
#[cfg(feature = "gpu-jpeg-huffman")]
pub(crate) mod dc_chain;
pub(crate) mod headers;
#[cfg(feature = "gpu-jpeg-huffman")]
pub(crate) mod prepass;
#[cfg(feature = "gpu-jpeg-huffman")]
pub(crate) mod unstuff;

#[cfg(test)]
pub(crate) mod test_fixtures;

/// Number of 8×8 blocks one component contributes per MCU
/// (ISO/IEC 10918-1 § A.2.3): a non-interleaved single-component scan
/// always emits 1, regardless of the component's sampling factors; an
/// interleaved scan emits `h × v`.
///
/// # Panics
///
/// Panics if `h × v` overflows `u8` — the header parser caps both
/// factors at 4, so a trip here means a walker bypassed it.
#[cfg(feature = "gpu-jpeg-huffman")]
#[must_use]
pub(crate) const fn component_blocks_per_mcu(
    scan_components: usize,
    h_sampling: u8,
    v_sampling: u8,
) -> u8 {
    if scan_components == 1 {
        1
    } else {
        match h_sampling.checked_mul(v_sampling) {
            Some(bpm) => bpm,
            None => panic!("header parser caps sampling factors at 4"),
        }
    }
}

#[cfg(feature = "gpu-jpeg-huffman")]
pub(crate) use canonical::CanonicalCodebook;
#[cfg(all(test, feature = "gpu-jpeg-huffman"))]
pub(crate) use canonical::{validate_canonical_table, visit_canonical_codes};
#[cfg(feature = "gpu-jpeg-huffman")]
pub(crate) use dc_chain::DcValues;
#[cfg(all(test, feature = "gpu-jpeg-huffman"))]
pub(crate) use headers::JpegHuffmanTable;
#[cfg(feature = "gpu-jpeg-huffman")]
pub(crate) use headers::{DhtClass, JpegHeaderError, JpegQuantTable};
#[cfg(feature = "gpu-jpeg-huffman")]
pub(crate) use prepass::{CpuPrepassError, CpuPrepassOutput, run_cpu_prepass};
