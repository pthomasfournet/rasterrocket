//! Subsequence size of the parallel-Huffman JPEG decoder.

use crate::jpeg_decoder::JpegPreparedInput;

/// Subsequence bit-width [`pick_subsequence_size`] returns.
pub(crate) const JPEG_SUBSEQUENCE_BITS: u32 = 1024;

/// Choose the subsequence bit-width for the parallel-Huffman JPEG
/// dispatcher.
///
/// The size trades three costs of `decode_jpeg_symbols`' cost model
/// against each other: Phase 1 and Phase 4 walk two and one regions
/// per thread, so their latency grows with the size; a Phase 2 miss
/// (a slot whose true left input matches neither memo entry) costs one
/// sequential region re-decode, also growing with the size; and the
/// re-decode budget counts re-decodes, not bits. The fraction of
/// fresh-start walkers that miss is set by the content (walkers that
/// stop at a decode error, and parses that stay valid but wrong across
/// uniform areas) and changes little with the size, but a wrong parse
/// that persists over a stretch of the stream costs one re-decode per
/// region of that stretch, so a larger size spends fewer of a
/// workgroup's re-decodes on it. None of the three depends on the
/// quantisation tables — a stream's quality changes its density, not
/// whether its walkers synchronise — so the choice is a constant, and
/// `prep` does not enter it.
#[must_use]
pub(crate) const fn pick_subsequence_size(_prep: &JpegPreparedInput) -> u32 {
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
