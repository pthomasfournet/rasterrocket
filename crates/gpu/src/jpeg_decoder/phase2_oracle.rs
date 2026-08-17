//! CPU oracles for the Phase 2 inter-sequence-sync kernels.
//!
//! ## Synthetic streams (Weißenberger & Schmidt 2021 §III-C, Alg. 4)
//!
//! `phase2_run_to_sync` walks the per-subsequence `s_info` array
//! produced by Phase 1, checks each `(i, i+1)` pair for `(c, z)`
//! agreement, advances any unsynced subseq by one symbol per
//! iteration, and repeats until every pair agrees (`Converged`) or
//! the retry bound (`2 * ceil(log2(n))` per the paper) is exhausted
//! (`SyncBoundExceeded`).
//!
//! Subseq `i` is "synced with its right neighbour" when:
//!
//! - `i + 1 == num_subsequences` (last subseq is trivially synced); OR
//! - `s_info[i].p >= (i + 1) * subsequence_bits` (subseq `i`'s walk
//!   reached at least the start of subseq `i+1`); AND
//! - `s_info[i].c == s_info[i+1].c`; AND
//! - `s_info[i].z == s_info[i+1].z`.
//!
//! The predicate relies on the `(c, z)` rollover happening identically
//! across subsequences when the codebook + bitstream are well-formed —
//! true for the synthetic state machine, where `(c, z)` advance on a
//! fixed schedule independent of absolute stream position.
//!
//! ## JPEG-framed streams: propagation, not agreement
//!
//! The agreement predicate cannot sync JPEG-framed multi-component
//! streams: no in-stream marker carries the component phase, so a
//! fresh-start walker beginning mid-MCU holds a permanently wrong
//! `block_in_mcu`, and two wrong walkers can agree with each other.
//! `jpeg_phase2_run_to_sync` instead propagates truth rightward:
//! each pass, subseq `i` re-decodes its region from `s_info[i - 1]`'s
//! boundary snapshot and rewrites `s_info[i]`; a pass with no change
//! is a fixpoint of the whole chain and therefore correct (slot 0 is
//! correct by construction). See `jpeg_phase2_retry_bound` for the
//! bound derivation.

#![cfg(test)]

use crate::jpeg::CanonicalCodebook;
use crate::jpeg_decoder::PackedBitstream;
use crate::jpeg_decoder::phase1_oracle::{
    JpegStep, StepOutcome, SubsequenceState, try_decode_one_jpeg_symbol, try_decode_one_symbol,
};

/// Outcome of the Phase 2 retry loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Phase2Outcome {
    /// Every subseq is synced with its right neighbour. `iterations`
    /// is the 0-based index of the pass that found every pair
    /// aligned — so a pre-synced Phase 1 output returns `0`, and a
    /// stream that needed one advance pass returns `1`.
    Converged { iterations: u32 },
    /// Retry bound exhausted without convergence. Returned after
    /// `bound + 1` passes; the `s_info` state is left at whatever the
    /// last pass produced.
    SyncBoundExceeded { bound: u32 },
}

/// Maximum number of sync-and-advance passes before giving up.
///
/// `2 * ceil(log2(n))` per the plan. `next_power_of_two(0) = 1` and
/// `next_power_of_two(1) = 1`, both with `trailing_zeros() = 0`, so
/// the math returns `0` cleanly for `n ≤ 1` without a special case.
#[must_use]
pub(super) fn retry_bound(num_subsequences: usize) -> u32 {
    let pow2_exp = num_subsequences.next_power_of_two().trailing_zeros();
    2u32.saturating_mul(pow2_exp)
}

/// One sync-and-advance pass.
///
/// For each `i` in `0..s_info.len() - 1`:
/// - If `(c, z)` agree between `s_info[i]` and `s_info[i+1]` AND
///   `s_info[i].p >= (i+1) * subsequence_bits`, the pair is synced.
/// - Otherwise, advance `s_info[i]` by one symbol via
///   `try_decode_one_symbol`. If the advance step misses (no codeword
///   / past `length_bits`), `s_info[i]` stays unchanged for this pass.
///
/// Returns the number of unsynced subsequences after this pass.
fn phase2_sync_pass(
    s_info: &mut [SubsequenceState],
    bitstream: &PackedBitstream,
    tables: &[CanonicalCodebook],
    subsequence_bits: u32,
) -> u32 {
    let n = s_info.len();
    if n <= 1 {
        return 0;
    }
    let mut unsynced = 0u32;
    for i in 0..n - 1 {
        let me = s_info[i];
        let nxt = s_info[i + 1];
        let nxt_start_p = u32::try_from(i + 1)
            .expect("subseq idx fits u32 by HuffmanParams::validate")
            .saturating_mul(subsequence_bits);
        let in_range = me.p >= nxt_start_p;
        let aligned = me.c == nxt.c && me.z == nxt.z;
        if in_range && aligned {
            continue;
        }
        // Not synced — advance me by one symbol. A miss leaves the
        // state unchanged; this pass is reported as still unsynced.
        //
        // `count_to = nxt_start_p`: a symbol *starting* before the
        // next subseq's start_bit belongs to *my* region (count it
        // in `me.n`); a symbol starting past that boundary belongs
        // to the next subseq (do not count).
        if let StepOutcome::Advanced(next) =
            try_decode_one_symbol(me, bitstream, tables, nxt_start_p)
        {
            s_info[i] = next;
        }
        unsynced += 1;
    }
    unsynced
}

/// Drive the Phase 2 sync loop to convergence.
///
/// Returns `Phase2Outcome::Converged { iterations }` once every pair
/// agrees, or `SyncBoundExceeded { bound }` after `retry_bound`
/// passes without convergence.
///
/// # Panics
/// Panics if `tables` is empty (caller bug — at least one codetable
/// is required for any decode work).
pub(super) fn phase2_run_to_sync(
    s_info: &mut [SubsequenceState],
    bitstream: &PackedBitstream,
    tables: &[CanonicalCodebook],
    subsequence_bits: u32,
) -> Phase2Outcome {
    assert!(
        !tables.is_empty(),
        "phase2_run_to_sync requires at least one codetable"
    );
    let bound = retry_bound(s_info.len());
    for iter in 0..=bound {
        let unsynced = phase2_sync_pass(s_info, bitstream, tables, subsequence_bits);
        if unsynced == 0 {
            return Phase2Outcome::Converged { iterations: iter };
        }
    }
    Phase2Outcome::SyncBoundExceeded { bound }
}

/// Pass bound for the JPEG-framed propagation Phase 2 loop.
///
/// Derivation: slot 0 is pinned (its Phase 1 walker starts at bit 0
/// in the true initial state, so its snapshot is correct by
/// construction), and each pass recomputes slot `i` from the previous
/// pass's slot `i - 1` — so after `k` passes, slots `0..=k` hold
/// their final values. All `n` slots are final after `n - 1` passes;
/// the next pass observes every recompute as a fixpoint and reports
/// convergence. The dispatch loop's `0..=bound` shape runs
/// `bound + 1` passes, so `bound = n - 1` guarantees convergence
/// detection for every input — including streams with decode errors,
/// because the re-decode is a deterministic function of the
/// predecessor state and therefore still stabilises.
#[must_use]
pub(super) fn jpeg_phase2_retry_bound(num_subsequences: usize) -> u32 {
    u32::try_from(num_subsequences.saturating_sub(1)).unwrap_or(u32::MAX)
}

/// Re-decode one subsequence's region from its predecessor's boundary
/// snapshot, returning the recomputed boundary snapshot.
///
/// `inherited` is `s_info[i - 1]`: the `(p, block_in_mcu, z_in_block)`
/// handoff state into region `i`; `n` restarts at 0 so it counts only
/// this region's symbols. The walk stops at the first symbol whose
/// advance reaches `region_end` (`min(length_bits,
/// (i + 1) * subsequence_bits)`) — the post-advance state at that
/// point is the snapshot. If `inherited.p >= region_end` already
/// (one symbol straddled the whole region), the inherited state *is*
/// the snapshot and the region owns zero symbols.
///
/// On a decode error the walk stops and returns the error-point
/// state. The re-decode is deterministic, so an erroring region still
/// reaches a fixpoint; the error itself surfaces later as a typed
/// Phase 4 failure.
fn jpeg_phase2_redecode(
    inherited: SubsequenceState,
    region_end: u32,
    bitstream: &PackedBitstream,
    dc_codebooks: &[&CanonicalCodebook],
    ac_codebooks: &[&CanonicalCodebook],
    mcu_schedule: &[u32],
    blocks_per_mcu: u32,
) -> SubsequenceState {
    let mut state = SubsequenceState {
        p: inherited.p,
        n: 0,
        c: inherited.c,
        z: inherited.z,
    };
    // Each symbol consumes ≥ 1 bit, so (region_end - p) + 1 iterations
    // bound the loop even for degenerate codebooks.
    let max_iters = region_end.saturating_sub(state.p) + 1;
    for _ in 0..max_iters {
        if state.p >= region_end {
            break;
        }
        match try_decode_one_jpeg_symbol(
            state,
            bitstream,
            dc_codebooks,
            ac_codebooks,
            mcu_schedule,
            blocks_per_mcu,
            region_end,
        ) {
            JpegStep::Advanced(next) => state = next,
            JpegStep::Failed(fail_state, _stop) => {
                state = fail_state;
                break;
            }
        }
    }
    state
}

/// Drive the JPEG-framed Phase 2 propagation loop to convergence.
///
/// Mirrors the `jpeg_phase2_inter_sync` kernels: each pass is a
/// Jacobi step — every slot `i ≥ 1` re-decodes its region from the
/// *previous* pass's `s_info[i - 1]` and rewrites `s_info[i]`; slot 0
/// is copied through unchanged. A pass in which no slot changes is a
/// fixpoint of the whole chain: slot 0 is correct by construction and
/// each following slot is the deterministic re-decode of its
/// predecessor, so the converged `s_info` carries the true absolute
/// block phase for every subsequence — the property the
/// single-symbol-advance predicate this replaces could not establish
/// for multi-component streams.
///
/// Returns `Converged { iterations }` (0-based index of the fixpoint
/// pass) or `SyncBoundExceeded` after `jpeg_phase2_retry_bound + 1`
/// passes — unreachable for deterministic inputs, kept as
/// defence-in-depth.
pub(super) fn jpeg_phase2_run_to_sync(
    s_info: &mut [SubsequenceState],
    bitstream: &PackedBitstream,
    dc_codebooks: &[&CanonicalCodebook],
    ac_codebooks: &[&CanonicalCodebook],
    mcu_schedule: &[u32],
    blocks_per_mcu: u32,
    subsequence_bits: u32,
) -> Phase2Outcome {
    let n = s_info.len();
    if n <= 1 {
        return Phase2Outcome::Converged { iterations: 0 };
    }
    let bound = jpeg_phase2_retry_bound(n);
    let length_bits = bitstream.length_bits;
    for iter in 0..=bound {
        let prev = s_info.to_vec();
        let mut all_fixed = true;
        for i in 1..n {
            let region_end = length_bits.min(
                u32::try_from(i + 1)
                    .expect("subseq idx fits u32 by HuffmanParams::validate")
                    .saturating_mul(subsequence_bits),
            );
            let out = jpeg_phase2_redecode(
                prev[i - 1],
                region_end,
                bitstream,
                dc_codebooks,
                ac_codebooks,
                mcu_schedule,
                blocks_per_mcu,
            );
            if out != prev[i] {
                all_fixed = false;
            }
            s_info[i] = out;
        }
        if all_fixed {
            return Phase2Outcome::Converged { iterations: iter };
        }
    }
    Phase2Outcome::SyncBoundExceeded { bound }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg_decoder::phase1_oracle::phase1_walk_snapshot;
    use crate::jpeg_decoder::tests::fixtures::{book4_codebook, book4_stream};

    /// Build a Phase 1 output by running `phase1_walk` per subsequence.
    /// Mirrors what the kernel's Phase 1 dispatch would produce.
    fn build_phase1_s_info(
        bitstream: &PackedBitstream,
        tables: &[CanonicalCodebook],
        subsequence_bits: u32,
    ) -> Vec<SubsequenceState> {
        let n = bitstream.length_bits.div_ceil(subsequence_bits);
        (0..n)
            .map(|i| {
                let start_bit = i * subsequence_bits;
                let hard_limit = bitstream.length_bits.min(start_bit + 2 * subsequence_bits);
                let count_to = bitstream.length_bits.min(start_bit + subsequence_bits);
                let (state, _stop) =
                    phase1_walk_snapshot(bitstream, tables, start_bit, hard_limit, count_to);
                state
            })
            .collect()
    }

    #[test]
    fn retry_bound_zero_for_zero_or_one_subseqs() {
        assert_eq!(retry_bound(0), 0);
        assert_eq!(retry_bound(1), 0);
    }

    #[test]
    fn retry_bound_matches_2_log2_ceil() {
        // 2 → ceil(log2(2)) = 1 → bound = 2
        assert_eq!(retry_bound(2), 2);
        // 3 → next_pow2 = 4 → exp 2 → bound = 4
        assert_eq!(retry_bound(3), 4);
        // 4 → next_pow2 = 4 → exp 2 → bound = 4
        assert_eq!(retry_bound(4), 4);
        // 1024 → exp 10 → bound = 20
        assert_eq!(retry_bound(1024), 20);
    }

    #[test]
    fn empty_or_singleton_converges_in_zero_iterations() {
        let stream = book4_stream(&[0x00; 100]);
        let book = [book4_codebook()];
        // 0 subseqs.
        let mut empty: Vec<SubsequenceState> = vec![];
        let r = phase2_run_to_sync(&mut empty, &stream, &book, 128);
        assert_eq!(r, Phase2Outcome::Converged { iterations: 0 });

        // 1 subseq (anything well-formed).
        let mut one = build_phase1_s_info(&stream, &book, 256);
        assert_eq!(one.len(), 1);
        let r = phase2_run_to_sync(&mut one, &stream, &book, 256);
        assert_eq!(r, Phase2Outcome::Converged { iterations: 0 });
    }

    #[test]
    fn well_formed_uniform_stream_converges_in_zero_iterations() {
        // 1024 length-2 codewords = 2048 bits. 16 subseqs of 128 bits
        // each, all the same width (length_bits is an exact multiple
        // of subsequence_bits). Every subseq decodes 128 symbols, so
        // end-state (c, z) matches across all pairs. Phase 2 sees a
        // pre-synced s_info and converges on the first sweep.
        //
        // Stream sizing matters: a stream where the final subseq is
        // shorter than the others can land its (c, z) anywhere
        // depending on how far phase1_walk gets, and the bound
        // `2 * log2(n)` is not enough advance steps to realign a
        // worst-case trailing subseq (needs O(subsequence_bits) in
        // the MVP). That's a different test (would need a larger
        // bound to exercise).
        let stream = book4_stream(&[0x00; 1024]);
        let book = [book4_codebook()];
        assert_eq!(stream.length_bits % 128, 0, "test setup invariant");
        let mut s_info = build_phase1_s_info(&stream, &book, 128);
        let r = phase2_run_to_sync(&mut s_info, &stream, &book, 128);
        assert_eq!(r, Phase2Outcome::Converged { iterations: 0 });
    }

    #[test]
    fn unsynced_input_advances_until_aligned() {
        // Build a pre-synced s_info on a stream sized to a clean
        // multiple of subsequence_bits, then deliberately perturb one
        // subseq's z so it doesn't match its neighbour. The retry
        // loop should advance it by one symbol until z realigns.
        //
        // For book4's length-2 codewords, advancing by one symbol
        // bumps z by 1 (mod 64). Perturbing s_info[0].z by -1
        // re-aligns after exactly one advance.
        let stream = book4_stream(&[0x00; 1024]);
        let book = [book4_codebook()];
        let mut s_info = build_phase1_s_info(&stream, &book, 128);
        s_info[0].z = (s_info[0].z + 63) % 64;
        let r = phase2_run_to_sync(&mut s_info, &stream, &book, 128);
        match r {
            Phase2Outcome::Converged { iterations } => {
                // First pass: subseq 0 unsynced → advance once.
                // Second pass: subseq 0 now aligned → all synced.
                assert!(
                    iterations <= 2,
                    "expected fast convergence, got {iterations} iterations"
                );
            }
            Phase2Outcome::SyncBoundExceeded { bound } => {
                panic!("unexpected SyncBoundExceeded with bound {bound}");
            }
        }
    }

    // ── JPEG-framed propagation tests ─────────────────────────────────

    use crate::jpeg_decoder::phase1_oracle::phase1_jpeg_walk_snapshot;
    use crate::jpeg_decoder::{JpegPreparedInput, build_mcu_schedule, prepare_jpeg};

    static COLOUR_32X32_444: &[u8] =
        include_bytes!("../../../../tests/fixtures/jpeg/colour_32x32_444.jpg");

    /// Fresh-start Phase 1 `s_info`, one `phase1_jpeg_walk_snapshot`
    /// per subsequence — what the JPEG Phase 1 kernel dispatch produces.
    fn build_jpeg_phase1_s_info(
        prep: &JpegPreparedInput,
        mcu_schedule: &[u32],
        blocks_per_mcu: u32,
        subsequence_bits: u32,
    ) -> Vec<SubsequenceState> {
        let dc_refs = prep.dc_codebooks_for_dispatch();
        let ac_refs = prep.ac_codebooks_for_dispatch();
        let length_bits = prep.bitstream.length_bits;
        let n = length_bits.div_ceil(subsequence_bits);
        (0..n)
            .map(|i| {
                let start_bit = i * subsequence_bits;
                let hard_limit = length_bits.min(start_bit + 2 * subsequence_bits);
                let count_to = length_bits.min(start_bit + subsequence_bits);
                let (state, _stop) = phase1_jpeg_walk_snapshot(
                    &prep.bitstream,
                    &dc_refs,
                    &ac_refs,
                    mcu_schedule,
                    blocks_per_mcu,
                    start_bit,
                    hard_limit,
                    count_to,
                );
                state
            })
            .collect()
    }

    /// Ground truth: one continuous walk from bit 0. Region `k`'s
    /// snapshot is the post-advance state of the first symbol crossing
    /// `(k + 1) * subsequence_bits`; its `n` is the number of symbols
    /// whose first bit lands in `[k * ssb, (k + 1) * ssb)`. This is a
    /// single sequential decode — structurally independent of the
    /// Jacobi propagation it checks.
    ///
    /// The entropy segment is bit-padded to a byte boundary, so the
    /// walk normally ends on a decode error inside the padding rather
    /// than exactly at `length_bits`. Regions whose boundary the walk
    /// never crossed take the terminal error-point state — the same
    /// semantics the fresh-start walker and the re-decode use.
    fn sequential_truth(
        prep: &JpegPreparedInput,
        mcu_schedule: &[u32],
        blocks_per_mcu: u32,
        subsequence_bits: u32,
    ) -> Vec<SubsequenceState> {
        let dc_refs = prep.dc_codebooks_for_dispatch();
        let ac_refs = prep.ac_codebooks_for_dispatch();
        let length_bits = prep.bitstream.length_bits;
        let n_regions = length_bits.div_ceil(subsequence_bits);
        let mut snaps: Vec<Option<SubsequenceState>> = vec![None; n_regions as usize];
        let mut counts = vec![0u32; n_regions as usize];

        let mut state = SubsequenceState {
            p: 0,
            n: 0,
            c: 0,
            z: 0,
        };
        while state.p < length_bits {
            let p_before = state.p;
            let next = match try_decode_one_jpeg_symbol(
                state,
                &prep.bitstream,
                &dc_refs,
                &ac_refs,
                mcu_schedule,
                blocks_per_mcu,
                length_bits,
            ) {
                JpegStep::Advanced(next) => next,
                JpegStep::Failed(fail_state, _stop) => {
                    // AcOverflow consumes the codeword before failing;
                    // the count it left behind belongs to this bucket.
                    if fail_state.n != state.n {
                        counts[(p_before / subsequence_bits) as usize] += 1;
                    }
                    state = fail_state;
                    break;
                }
            };
            counts[(p_before / subsequence_bits) as usize] += 1;
            // A symbol crossing one or more region boundaries supplies
            // the same post-advance snapshot to each crossed region —
            // regions a symbol straddles entirely own zero symbols.
            let first = p_before / subsequence_bits;
            let last = (next.p - 1) / subsequence_bits;
            for k in first..=last.min(n_regions - 1) {
                let crossed = next.p >= length_bits.min((k + 1) * subsequence_bits);
                if crossed && snaps[k as usize].is_none() {
                    snaps[k as usize] = Some(next);
                }
            }
            state = next;
        }

        snaps
            .into_iter()
            .zip(counts)
            .map(|(snap, n)| {
                let mut s = snap.unwrap_or(state);
                s.n = n;
                s
            })
            .collect()
    }

    #[test]
    fn jpeg_retry_bound_is_num_subsequences_minus_one() {
        assert_eq!(jpeg_phase2_retry_bound(0), 0);
        assert_eq!(jpeg_phase2_retry_bound(1), 0);
        assert_eq!(jpeg_phase2_retry_bound(2), 1);
        assert_eq!(jpeg_phase2_retry_bound(17), 16);
    }

    #[test]
    fn jpeg_empty_or_singleton_converges_in_zero_iterations() {
        use crate::jpeg::test_fixtures::GRAY_16X16_JPEG;
        let prep = prepare_jpeg(GRAY_16X16_JPEG).unwrap();
        let (sched, bpm) = build_mcu_schedule(&prep);
        let dc_refs = prep.dc_codebooks_for_dispatch();
        let ac_refs = prep.ac_codebooks_for_dispatch();

        let mut empty: Vec<SubsequenceState> = vec![];
        let r = jpeg_phase2_run_to_sync(
            &mut empty,
            &prep.bitstream,
            &dc_refs,
            &ac_refs,
            &sched,
            bpm,
            128,
        );
        assert_eq!(r, Phase2Outcome::Converged { iterations: 0 });

        let big = prep.bitstream.length_bits.next_power_of_two();
        let mut one = build_jpeg_phase1_s_info(&prep, &sched, bpm, big);
        assert_eq!(one.len(), 1);
        let r = jpeg_phase2_run_to_sync(
            &mut one,
            &prep.bitstream,
            &dc_refs,
            &ac_refs,
            &sched,
            bpm,
            big,
        );
        assert_eq!(r, Phase2Outcome::Converged { iterations: 0 });
    }

    #[test]
    fn jpeg_propagation_converges_to_truth_on_grayscale() {
        use crate::jpeg::test_fixtures::GRAY_16X16_JPEG;
        let prep = prepare_jpeg(GRAY_16X16_JPEG).unwrap();
        let (sched, bpm) = build_mcu_schedule(&prep);
        let dc_refs = prep.dc_codebooks_for_dispatch();
        let ac_refs = prep.ac_codebooks_for_dispatch();
        // The fixture's entropy stream is short; size subsequences off
        // its length so the test always spans several of them.
        let ssb = (prep.bitstream.length_bits / 4).max(1);

        let mut s_info = build_jpeg_phase1_s_info(&prep, &sched, bpm, ssb);
        assert!(s_info.len() > 1, "fixture must span multiple subsequences");
        let r = jpeg_phase2_run_to_sync(
            &mut s_info,
            &prep.bitstream,
            &dc_refs,
            &ac_refs,
            &sched,
            bpm,
            ssb,
        );
        assert!(
            matches!(r, Phase2Outcome::Converged { .. }),
            "expected convergence, got {r:?}"
        );
        assert_eq!(s_info, sequential_truth(&prep, &sched, bpm, ssb));
    }

    #[test]
    fn jpeg_propagation_converges_to_truth_on_ycbcr_444() {
        // The multi-component case the single-symbol-advance predicate
        // could never sync: fresh-start Phase 1 walkers hold wrong
        // block phases for 3-component streams (no in-stream marker
        // carries component phase), so truth must propagate rightward
        // from subsequence 0.
        for ssb in [16u32, 32] {
            let prep = prepare_jpeg(COLOUR_32X32_444).unwrap();
            assert_eq!(prep.components.len(), 3, "fixture must be 3-component");
            let (sched, bpm) = build_mcu_schedule(&prep);
            let dc_refs = prep.dc_codebooks_for_dispatch();
            let ac_refs = prep.ac_codebooks_for_dispatch();

            let truth = sequential_truth(&prep, &sched, bpm, ssb);
            let mut s_info = build_jpeg_phase1_s_info(&prep, &sched, bpm, ssb);
            assert_ne!(
                s_info, truth,
                "precondition: fresh-start Phase 1 output must be wrong \
                 (otherwise this test is vacuous)"
            );

            let r = jpeg_phase2_run_to_sync(
                &mut s_info,
                &prep.bitstream,
                &dc_refs,
                &ac_refs,
                &sched,
                bpm,
                ssb,
            );
            assert!(
                matches!(r, Phase2Outcome::Converged { .. }),
                "ssb={ssb}: expected convergence, got {r:?}"
            );
            assert_eq!(s_info, truth, "ssb={ssb}: converged s_info must be true");
        }
    }

    #[test]
    fn adversarial_input_returns_sync_bound_exceeded() {
        // Construct an s_info where (c, z) of every pair disagrees
        // permanently. We can't easily produce this via real decoding
        // (the codebook ensures alignment), so synthesise: every subseq
        // gets a unique z value. Advancing won't fix it within bound.
        let stream = book4_stream(&[0x00; 1000]);
        let book = [book4_codebook()];
        let mut s_info = build_phase1_s_info(&stream, &book, 128);
        // Force each subseq to a distinct z so no neighbour pair
        // aligns. After one advance, each subseq's z moves by 1, so
        // the relative phase stays constant. Bound for 16 subseqs is
        // 2 * ceil(log2(16)) = 8 passes.
        for (i, s) in s_info.iter_mut().enumerate() {
            s.z = u32::try_from(i).expect("test fixture: 16 fits u32") & 63;
        }
        let r = phase2_run_to_sync(&mut s_info, &stream, &book, 128);
        assert!(
            matches!(r, Phase2Outcome::SyncBoundExceeded { .. }),
            "expected SyncBoundExceeded, got {r:?}"
        );
    }
}
