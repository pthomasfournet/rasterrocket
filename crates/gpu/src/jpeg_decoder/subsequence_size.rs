//! Subsequence size of the parallel-Huffman JPEG decoder.

use crate::jpeg_decoder::JpegPreparedInput;

/// Subsequence bit-width [`pick_subsequence_size`] returns.
pub const JPEG_SUBSEQUENCE_BITS: u32 = 512;

/// Choose the subsequence bit-width for the parallel-Huffman JPEG
/// dispatcher.
///
/// The size trades three costs of `decode_jpeg_symbols`' cost model
/// against each other: Phase 1 and Phase 4 walk two and one regions
/// per thread, so their latency grows with the size; a Phase 2 miss
/// (a slot whose true left input matches neither memo entry) costs one
/// sequential region re-decode, also growing with the size; and the
/// miss rate falls as the size grows, because a fresh-start Phase 1
/// walker has more bits in which to synchronise with the true decode
/// before its region ends. None of the three depends on the
/// quantisation tables — a stream's quality changes its density, not
/// whether its walkers synchronise — so the choice is a constant, and
/// `prep` does not enter it.
#[must_use]
pub const fn pick_subsequence_size(_prep: &JpegPreparedInput) -> u32 {
    JPEG_SUBSEQUENCE_BITS
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg_decoder::cpu_prepass::prepare_jpeg;

    #[test]
    fn size_is_independent_of_quality() {
        for bytes in [
            &include_bytes!("../../../../tests/fixtures/jpeg/q95_scan.jpg")[..],
            &include_bytes!("../../../../tests/fixtures/jpeg/q20.jpg")[..],
        ] {
            let prep = prepare_jpeg(bytes).unwrap();
            assert_eq!(pick_subsequence_size(&prep), JPEG_SUBSEQUENCE_BITS);
        }
    }
}
