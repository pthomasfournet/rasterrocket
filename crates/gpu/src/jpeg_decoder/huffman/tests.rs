//! CUDA parity tests of the parallel-Huffman dispatchers against the
//! CPU oracles.

use super::*;
use crate::backend::cuda::CudaBackend;
use crate::jpeg::headers::{DhtClass, JpegHuffmanTable};
use crate::jpeg_decoder::build_mcu_schedule;
use crate::jpeg_decoder::phase1_oracle::{phase1_jpeg_walk_snapshot, phase1_walk_snapshot};
use crate::jpeg_decoder::prepare_jpeg;
use crate::jpeg_decoder::tests::fixtures::{book4_codebook, book4_stream};

static COLOUR_32X32_444: &[u8] =
    include_bytes!("../../../../../tests/fixtures/jpeg/colour_32x32_444.jpg");
static COLOUR_128X128_444: &[u8] =
    include_bytes!("../../../../../tests/fixtures/jpeg/colour_128x128_444.jpg");
static COLOUR_128X128_420: &[u8] =
    include_bytes!("../../../../../tests/fixtures/jpeg/colour_128x128_420.jpg");

fn try_cuda() -> Option<CudaBackend> {
    CudaBackend::new().ok()
}

/// Run the kernel + run the CPU oracle for every subsequence;
/// assert byte-identical state per subsequence.
fn assert_phase1_parity(stream: &PackedBitstream, tables: &[CanonicalCodebook], subseq_bits: u32) {
    let Some(b) = try_cuda() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let gpu = dispatch_phase1_intra_sync(&b, stream, tables, subseq_bits).expect("dispatch ok");
    let num_subseq = stream.length_bits.div_ceil(subseq_bits);
    assert_eq!(
        u32::try_from(gpu.len()).expect("subseq count fits u32"),
        num_subseq
    );
    for seq_idx in 0..num_subseq {
        let start_bit = seq_idx * subseq_bits;
        let hard_limit = stream.length_bits.min(start_bit + 2 * subseq_bits);
        let count_to = stream.length_bits.min(start_bit + subseq_bits);
        let (cpu, _stop) = phase1_walk_snapshot(stream, tables, start_bit, hard_limit, count_to);
        assert_eq!(
            gpu[seq_idx as usize], cpu,
            "subsequence {seq_idx} GPU vs CPU state mismatch"
        );
    }
}

#[test]
fn phase1_single_subsequence_matches_cpu() {
    // 100 length-2 codewords = 200 bits in one 256-bit subsequence.
    let symbols = vec![0x00; 100];
    let stream = book4_stream(&symbols);
    let book = [book4_codebook()];
    assert_phase1_parity(&stream, &book, 256);
}

#[test]
fn phase1_two_subsequences_match_cpu() {
    let symbols = vec![0x00; 128];
    let stream = book4_stream(&symbols);
    let book = [book4_codebook()];
    assert_phase1_parity(&stream, &book, 128);
}

#[test]
fn phase1_many_subsequences_match_cpu() {
    let symbols = vec![0x00; 1000];
    let stream = book4_stream(&symbols);
    let book = [book4_codebook()];
    assert_phase1_parity(&stream, &book, 128);
}

#[test]
fn phase1_mixed_symbols_match_cpu() {
    // Mixed codeword lengths exercise both length-2 and length-3 codes.
    let symbols: Vec<u8> = (0..1000u32).map(|i| (i % 4) as u8).collect();
    let stream = book4_stream(&symbols);
    let book = [book4_codebook()];
    assert_phase1_parity(&stream, &book, 128);
}

#[test]
fn phase1_multi_component_match_cpu() {
    let symbols = vec![0x00; 200];
    let stream = book4_stream(&symbols);
    // Three components — z rolls over every 64 symbols,
    // c walks 0 → 1 → 2 → 0.
    let book = [book4_codebook(), book4_codebook(), book4_codebook()];
    assert_phase1_parity(&stream, &book, 64);
}

#[test]
fn build_gpu_codebook_packs_entries_correctly() {
    // 1-entry table: symbol 42 at length 1 (code "0"). Half the
    // 65536-entry LUT is filled (prefixes starting with 0).
    let table = JpegHuffmanTable {
        class: DhtClass::Dc,
        table_id: 0,
        num_codes: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        values: vec![42],
    };
    let book = CanonicalCodebook::build(&table).unwrap();
    let flat = build_gpu_codebook(&[book]);
    assert_eq!(flat.len(), HUFFMAN_CODEBOOK_ENTRIES);
    // (1 << 8) | 42 = 298. Prefixes < 0x8000 → length-1 "0" match.
    assert_eq!(flat[0], 298);
    assert_eq!(flat[0x7FFF], 298);
    // Prefixes ≥ 0x8000 → no codeword, num_bits=0.
    assert_eq!(flat[0x8000], 0);
}

/// Phase 2 dispatcher: well-formed stream (`length_bits` is an
/// exact multiple of `subsequence_bits`) converges on the first
/// pass. Phase 1's output is already pre-synced.
#[test]
fn phase2_well_formed_stream_converges_immediately_cuda() {
    let Some(b) = try_cuda() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    // 1024 length-2 codewords = 2048 bits, 16 subseqs of 128 bits.
    let stream = book4_stream(&[0x00; 1024]);
    let book = [book4_codebook()];
    let (s_info, outcome) = dispatch_phase1_then_phase2(&b, &stream, &book, 128).expect("dispatch");
    // 16 subseqs.
    assert_eq!(s_info.len(), 16);
    // Pre-synced stream — first pass sees all flags = 1.
    assert_eq!(outcome, Phase2Outcome::Converged { iterations: 0 });
}

/// Phase 2 dispatcher matches the CPU oracle: run both on the
/// same input, the final `s_info` should be byte-identical.
#[test]
fn phase2_matches_cpu_oracle_well_formed_cuda() {
    let Some(b) = try_cuda() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let stream = book4_stream(&[0x00; 1024]);
    let book = [book4_codebook()];

    // GPU: Phase 1 + Phase 2.
    let (gpu_s_info, _outcome) =
        dispatch_phase1_then_phase2(&b, &stream, &book, 128).expect("dispatch");

    // CPU: Phase 1 + Phase 2 in the same shape.
    let mut cpu_s_info: Vec<SubsequenceState> = (0u32..16)
        .map(|i| {
            let start_bit = i * 128;
            let hard_limit = stream.length_bits.min(start_bit + 2 * 128);
            let count_to = stream.length_bits.min(start_bit + 128);
            let (state, _stop) =
                phase1_walk_snapshot(&stream, &book, start_bit, hard_limit, count_to);
            state
        })
        .collect();
    let cpu_outcome = crate::jpeg_decoder::phase2_oracle::phase2_run_to_sync(
        &mut cpu_s_info,
        &stream,
        &book,
        128,
    );
    assert!(matches!(
        cpu_outcome,
        crate::jpeg_decoder::phase2_oracle::Phase2Outcome::Converged { .. }
    ));

    assert_eq!(gpu_s_info, cpu_s_info);
}

/// Phase 3 dispatcher: exclusive-scan over `s_info[i].n` produces
/// write offsets that match the CPU exclusive-scan reference.
/// The pre-synced stream means every subseq's `n` equals its
/// subsequence-bit / codeword-length ratio, but the test asserts
/// the scan property, not the specific count values.
#[test]
fn phase3_offsets_match_exclusive_scan_cuda() {
    let Some(b) = try_cuda() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let stream = book4_stream(&[0x00; 1024]);
    let book = [book4_codebook()];
    let (s_info, _outcome) =
        dispatch_phase1_then_phase2(&b, &stream, &book, 128).expect("phase1+2");
    let offsets = dispatch_phase3_offsets(&b, &s_info).expect("phase3");

    assert_eq!(offsets.len(), s_info.len());
    assert_eq!(offsets[0], 0, "exclusive scan starts at 0");
    for i in 1..offsets.len() {
        assert_eq!(
            offsets[i],
            offsets[i - 1] + s_info[i - 1].n,
            "offset[{i}] = offset[{}] + n[{}]",
            i - 1,
            i - 1
        );
    }
}

/// Phase 3 with a single subsequence: scan returns `[0]`.
#[test]
fn phase3_single_subsequence_cuda() {
    let Some(b) = try_cuda() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    // 100 symbols of length 2 = 200 bits in one 256-bit subseq.
    let stream = book4_stream(&[0x00; 100]);
    let book = [book4_codebook()];
    let (s_info, _outcome) =
        dispatch_phase1_then_phase2(&b, &stream, &book, 256).expect("phase1+2");
    assert_eq!(s_info.len(), 1);
    let offsets = dispatch_phase3_offsets(&b, &s_info).expect("phase3");
    assert_eq!(offsets, vec![0]);
}

/// Phase 3 with empty input is a no-op — `Vec::new()` round-trip.
#[test]
fn phase3_empty_input_cuda() {
    let Some(b) = try_cuda() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let offsets = dispatch_phase3_offsets(&b, &[]).expect("phase3 empty");
    assert_eq!(offsets.len(), 0);
}

/// Phase 3 over Phase 2's `s_info` matches a host-side scan of
/// the same `.n` values. This is the contract Phase 3 must
/// guarantee — converting the *meaning* of `.n` to "symbols in
/// this subseq's own region" is Phase 4's job (re-decode pass).
#[test]
fn phase3_matches_cpu_exclusive_scan_mixed_cuda() {
    let Some(b) = try_cuda() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let symbols: Vec<u8> = (0..1000u32).map(|i| (i % 4) as u8).collect();
    let stream = book4_stream(&symbols);
    let book = [book4_codebook()];
    let (s_info, _outcome) =
        dispatch_phase1_then_phase2(&b, &stream, &book, 128).expect("phase1+2");

    let counts: Vec<u32> = s_info.iter().map(|s| s.n).collect();
    let cpu_off = crate::jpeg_decoder::scan::test_helpers::cpu_exclusive_scan(&counts);
    let gpu_off = dispatch_phase3_offsets(&b, &s_info).expect("phase3");
    assert_eq!(gpu_off, cpu_off);
}

/// End-to-end: encode synthetic symbols → run Phases 1..4 on GPU
/// → verify the decoded symbol stream matches the input.
///
/// Uniform symbol stream (all 0x00 → code "00"): every subseq's
/// own region is a clean integer number of length-2 codewords,
/// no straddling. This is the simplest possible end-to-end test.
#[test]
fn end_to_end_uniform_stream_cuda() {
    let Some(b) = try_cuda() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let symbols: Vec<u8> = vec![0x00; 1024];
    let stream = book4_stream(&symbols);
    let book = [book4_codebook()];
    let decoded = dispatch_phase1_through_phase4(&b, &stream, &book, 128).expect("e2e");

    assert_eq!(decoded.len(), symbols.len(), "symbol count mismatch");
    let got: Vec<u8> = decoded
        .iter()
        .map(|&v| u8::try_from(v & 0xFF).expect("symbol fits u8"))
        .collect();
    assert_eq!(got, symbols, "decoded stream diverges from input");
}

/// End-to-end on **mixed** codeword lengths (lengths 2 + 3), 1000
/// symbols at `subsequence_bits = 128`.  The `(c, z)` sync
/// predicate can need `O(subseq_bits)` advance steps per subseq on
/// this corpus, which exceeds the `2 * log2(n)` retry bound: Phase 2
/// never converges and the dispatcher surfaces `SyncBoundExceeded`.
#[test]
fn end_to_end_mixed_codeword_lengths_cuda_sync_bound_exceeded() {
    let Some(b) = try_cuda() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let symbols: Vec<u8> = (0..1000u32).map(|i| (i % 4) as u8).collect();
    let stream = book4_stream(&symbols);
    let book = [book4_codebook()];
    let e = dispatch_phase1_through_phase4(&b, &stream, &book, 128)
        .expect_err("expected SyncBoundExceeded on this mixed-length corpus");
    let msg = format!("{e}");
    assert!(
        msg.contains("Phase 2 did not converge"),
        "unexpected error: {msg}"
    );
}

// ── JPEG Phase 1 parity tests ─────────────────────────────────────────

/// Every Phase 1 memo entry JPEG Phase 1 writes on CUDA is the
/// exact re-decode of its region from its key, per the CPU oracle,
/// and every last-exact entry starts out matching nothing. The
/// fixtures and sizes leave many walkers unsynchronised (wrong
/// block phase), so wrong-but-exact keys are covered too.
#[test]
fn jpeg_phase1_memo_entries_are_exact_redecodes_cuda() {
    use crate::jpeg_decoder::phase2_oracle::{jpeg_phase2_redecode, jpeg_region_end};
    let Some(b) = try_cuda() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let fixtures: [(&str, &[u8]); 3] = [
        (
            "q95_scan",
            include_bytes!("../../../../../tests/fixtures/jpeg/q95_scan.jpg"),
        ),
        ("444", COLOUR_128X128_444),
        ("420", COLOUR_128X128_420),
    ];
    for (name, bytes) in fixtures {
        let prep = prepare_jpeg(bytes).expect("fixture must prepare");
        let (sched, bpm) = build_mcu_schedule(&prep);
        let dc_refs = prep.dc_codebooks_for_dispatch();
        let ac_refs = prep.ac_codebooks_for_dispatch();
        let length_bits = prep.bitstream.length_bits;
        let (mut keyed, mut slots) = (0usize, 0usize);
        for subseq_bits in [16u32, 32, 64] {
            let memo = dispatch_jpeg_phase1_memo(&b, &prep, subseq_bits).expect("dispatch");
            slots += memo.len();
            for (i, [key, value, last_key, _]) in memo.iter().enumerate() {
                assert_eq!(last_key.p, u32::MAX, "{name} ssb={subseq_bits} slot {i}");
                if i == 0 || key.p == u32::MAX {
                    continue;
                }
                keyed += 1;
                let expected = jpeg_phase2_redecode(
                    *key,
                    jpeg_region_end(i, subseq_bits, length_bits),
                    &prep.bitstream,
                    &dc_refs,
                    &ac_refs,
                    &sched,
                    bpm,
                );
                assert_eq!(
                    *value, expected,
                    "{name} ssb={subseq_bits} slot {i}: memo value is not the re-decode of its key"
                );
            }
        }
        assert!(keyed * 4 > slots, "{name}: {keyed} of {slots} slots keyed");
    }
}

/// Run `dispatch_jpeg_phase1_intra_sync` on CUDA and verify the
/// per-subsequence state matches the CPU oracle for every subseq.
#[test]
fn jpeg_phase1_cuda_matches_cpu_oracle_grayscale() {
    use crate::jpeg::test_fixtures::GRAY_16X16_JPEG;
    let Some(b) = try_cuda() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let prep = prepare_jpeg(GRAY_16X16_JPEG).expect("baseline grayscale must prepare");
    let (mcu_sched, blocks_per_mcu) = build_mcu_schedule(&prep);

    // Use a small subsequence so we get multiple subseqs on the 16×16 fixture.
    let subsequence_bits = 32u32;
    let gpu_states = dispatch_jpeg_phase1_intra_sync(&b, &prep, subsequence_bits)
        .expect("JPEG Phase 1 dispatch");

    let dc_refs = prep.dc_codebooks_for_dispatch();
    let ac_refs = prep.ac_codebooks_for_dispatch();
    let num_subseq = prep.bitstream.length_bits.div_ceil(subsequence_bits);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "gpu_states.len() is a subsequence count on a 16×16 test fixture, far below u32::MAX"
    )]
    let gpu_state_count = gpu_states.len() as u32;
    assert_eq!(
        gpu_state_count, num_subseq,
        "GPU state count must equal num_subsequences"
    );

    for seq_idx in 0..num_subseq {
        let start_bit = seq_idx * subsequence_bits;
        let hard_limit = prep
            .bitstream
            .length_bits
            .min(start_bit + 2 * subsequence_bits);
        let count_to = prep.bitstream.length_bits.min(start_bit + subsequence_bits);

        let (cpu, _stop) = phase1_jpeg_walk_snapshot(
            &prep.bitstream,
            &dc_refs,
            &ac_refs,
            &mcu_sched,
            blocks_per_mcu,
            start_bit,
            hard_limit,
            count_to,
        );
        assert_eq!(
            gpu_states[seq_idx as usize], cpu,
            "subsequence {seq_idx}: GPU vs CPU JPEG Phase 1 state mismatch"
        );
    }
}

/// Pin the stall detector's boundary: errors within one maximal
/// symbol of the stream end are the normal byte-padding tail;
/// anything shorter is a mid-stream stall.
#[test]
fn stall_detector_allows_only_the_padding_tail() {
    // 27 = max codeword (16) + max DC magnitude (11).
    assert!(!jpeg_stream_stalled(1000, 1000), "clean end");
    assert!(
        !jpeg_stream_stalled(973, 1000),
        "error exactly at the allowance"
    );
    assert!(jpeg_stream_stalled(972, 1000), "one bit past the allowance");
    assert!(jpeg_stream_stalled(0, 1000), "stall at stream start");
    assert!(!jpeg_stream_stalled(0, 0), "empty stream never stalls");
    // saturating_add keeps the check meaningful near u32::MAX.
    assert!(!jpeg_stream_stalled(u32::MAX - 5, u32::MAX));
}

// ── JPEG Phase 2 tests ────────────────────────────────────────────────

/// JPEG Phase 1+2 on CUDA converges on the grayscale fixture (single
/// subseq — trivially synced — and small multi-subseq with a stream
/// whose `length_bits` is a clean multiple of `subsequence_bits`).
#[test]
fn jpeg_phase2_cuda_converges_on_grayscale() {
    use crate::jpeg::test_fixtures::GRAY_16X16_JPEG;
    let Some(b) = try_cuda() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let prep = prepare_jpeg(GRAY_16X16_JPEG).expect("baseline grayscale must prepare");
    // Use a small subsequence_bits to get multiple subseqs; the
    // grayscale fixture is small enough that all should converge
    // on the first Phase 2 pass.
    let (s_info, outcome) = dispatch_jpeg_phase1_then_phase2(&b, &prep, 32).expect("dispatch ok");
    let num_subseq = prep.bitstream.length_bits.div_ceil(32);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "s_info.len() is a subsequence count on a 16×16 test fixture, far below u32::MAX"
    )]
    let s_info_count = s_info.len() as u32;
    assert_eq!(s_info_count, num_subseq);
    assert!(
        matches!(outcome, Phase2Outcome::Converged { .. }),
        "expected convergence on grayscale, got {outcome:?}"
    );
}

/// After JPEG Phase 1+2 on CUDA, every subseq's snapshot position
/// must have advanced past its own start bit (the basic sync invariant).
#[test]
fn jpeg_phase2_cuda_snapshots_past_start_bit() {
    use crate::jpeg::test_fixtures::GRAY_16X16_JPEG;
    let Some(b) = try_cuda() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let prep = prepare_jpeg(GRAY_16X16_JPEG).expect("baseline grayscale must prepare");
    let subseq_bits = 32u32;
    let (s_info, _) =
        dispatch_jpeg_phase1_then_phase2(&b, &prep, subseq_bits).expect("dispatch ok");
    for (i, s) in s_info.iter().enumerate() {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "i is a subsequence index on a 16×16 test fixture, far below u32::MAX"
        )]
        let start_p = i as u32 * subseq_bits;
        assert!(
            s.p >= start_p,
            "subseq {i}: snapshot p={} is before start_bit={start_p}",
            s.p
        );
    }
}

/// CPU mirror of `dispatch_jpeg_phase1_then_phase2`: per-subseq
/// fresh-start Phase 1 snapshots, then the propagation oracle.
fn cpu_jpeg_phase1_then_phase2(
    prep: &crate::jpeg_decoder::JpegPreparedInput,
    subseq_bits: u32,
) -> (Vec<SubsequenceState>, Phase2Outcome) {
    use crate::jpeg_decoder::phase2_oracle::jpeg_phase2_run_to_sync;
    let (mcu_sched, blocks_per_mcu) = build_mcu_schedule(prep);
    let dc_refs = prep.dc_codebooks_for_dispatch();
    let ac_refs = prep.ac_codebooks_for_dispatch();
    let length_bits = prep.bitstream.length_bits;
    let mut s_info: Vec<SubsequenceState> = (0..length_bits.div_ceil(subseq_bits))
        .map(|i| {
            let start_bit = i * subseq_bits;
            let hard_limit = length_bits.min(start_bit + 2 * subseq_bits);
            let count_to = length_bits.min(start_bit + subseq_bits);
            let (state, _stop) = phase1_jpeg_walk_snapshot(
                &prep.bitstream,
                &dc_refs,
                &ac_refs,
                &mcu_sched,
                blocks_per_mcu,
                start_bit,
                hard_limit,
                count_to,
            );
            state
        })
        .collect();
    let outcome = jpeg_phase2_run_to_sync(
        &mut s_info,
        &prep.bitstream,
        &dc_refs,
        &ac_refs,
        &mcu_sched,
        blocks_per_mcu,
        subseq_bits,
    );
    (s_info, outcome)
}

/// JPEG Phase 1+2 on CUDA reproduces the CPU propagation oracle
/// exactly on the multi-component fixture — final states and
/// outcome both. This is the case the old agreement predicate
/// could never sync (fresh-start walkers hold wrong block phases).
#[test]
fn jpeg_phase2_cuda_matches_propagation_oracle_on_ycbcr() {
    let Some(b) = try_cuda() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let prep = prepare_jpeg(COLOUR_32X32_444).expect("colour 4:4:4 JPEG must prepare");
    assert_eq!(prep.components.len(), 3, "fixture must be 3-component");
    let subseq_bits = 32u32;

    let (gpu_s_info, gpu_outcome) =
        dispatch_jpeg_phase1_then_phase2(&b, &prep, subseq_bits).expect("dispatch ok");
    let (cpu_s_info, cpu_outcome) = cpu_jpeg_phase1_then_phase2(&prep, subseq_bits);

    assert_eq!(gpu_outcome, cpu_outcome, "GPU vs CPU Phase 2 outcome");
    assert_eq!(gpu_s_info, cpu_s_info, "GPU vs CPU converged s_info");
}

/// Same parity on inputs spanning several workgroups, so the
/// kernel's on-device local loop, its cross-workgroup hand-off
/// through the host pass, and the pass count all have to agree
/// with the oracle's model of them. Covers both 4:4:4 (three
/// blocks per MCU) and 4:2:0 (six).
#[test]
fn jpeg_phase2_cuda_matches_propagation_oracle_across_workgroups() {
    let Some(b) = try_cuda() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    for (name, bytes) in [("444", COLOUR_128X128_444), ("420", COLOUR_128X128_420)] {
        for subseq_bits in [16u32, 64] {
            let prep = prepare_jpeg(bytes).expect("colour JPEG must prepare");
            let (gpu_s_info, gpu_outcome) =
                dispatch_jpeg_phase1_then_phase2(&b, &prep, subseq_bits).expect("dispatch");
            assert!(
                gpu_s_info.len() > HUFFMAN_THREADS as usize,
                "{name} ssb={subseq_bits}: fixture must span more than one workgroup"
            );
            assert!(
                matches!(gpu_outcome, Phase2Outcome::Converged { .. }),
                "{name} ssb={subseq_bits}: expected convergence, got {gpu_outcome:?}"
            );
            let (cpu_s_info, cpu_outcome) = cpu_jpeg_phase1_then_phase2(&prep, subseq_bits);
            assert_eq!(
                gpu_outcome, cpu_outcome,
                "{name} ssb={subseq_bits}: GPU vs CPU Phase 2 outcome"
            );
            assert_eq!(
                gpu_s_info, cpu_s_info,
                "{name} ssb={subseq_bits}: GPU vs CPU converged s_info"
            );
        }
    }
}

/// The Phase 2 pass budget is exact: a budget one short of the
/// oracle's converged pass count reports non-convergence, the
/// oracle's count converges, and the capped production decoder
/// errors (the CPU fallback trigger) rather than returning symbols
/// when its budget runs out.
#[test]
fn jpeg_phase2_pass_budget_is_enforced_cuda() {
    let Some(b) = try_cuda() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let prep = prepare_jpeg(COLOUR_128X128_420).expect("colour JPEG must prepare");
    let subseq_bits = 16;
    let Phase2Outcome::Converged { iterations } = cpu_jpeg_phase1_then_phase2(&prep, subseq_bits).1
    else {
        panic!("oracle must converge");
    };
    let needed = iterations + 1;
    assert!(
        needed >= 2,
        "precondition: fixture must need several passes"
    );

    let passes = |passes: u32| Phase2Budget {
        passes,
        redecodes: u32::MAX,
    };
    let run = |budget: Phase2Budget| {
        let mut setup = JpegDispatchSetup::new(&b, &prep, subseq_bits).expect("setup");
        run_jpeg_phase1_and_phase2(&b, &mut setup, budget).expect("dispatch")
    };
    assert_eq!(run(passes(needed - 1)), Phase2Run::PassBudgetExhausted);
    assert_eq!(run(passes(needed)), Phase2Run::Converged(iterations));

    let err = decode_jpeg_symbols_within(&b, &prep, subseq_bits, passes(needed - 1))
        .expect_err("an exhausted pass budget must refuse the stream");
    assert!(err.to_string().contains("did not converge"), "{err}");
    let decoded =
        decode_jpeg_symbols_within(&b, &prep, subseq_bits, passes(needed)).expect("decode");
    assert_eq!(decoded.phase2_passes, needed);
}

/// A workgroup that needs more re-decodes than the budget allows
/// stops the propagation (the fixture's fresh-start walkers hold
/// wrong block phases, so some re-decode is needed), and the capped
/// production decoder refuses the stream; an unlimited budget on
/// the same input converges to the oracle's chain.
#[test]
fn jpeg_phase2_redecode_budget_is_enforced_cuda() {
    let Some(b) = try_cuda() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let prep = prepare_jpeg(COLOUR_128X128_420).expect("colour JPEG must prepare");
    let subseq_bits = 16;
    let no_redecodes = Phase2Budget {
        passes: u32::MAX,
        redecodes: 0,
    };
    let mut setup = JpegDispatchSetup::new(&b, &prep, subseq_bits).expect("setup");
    assert_eq!(
        run_jpeg_phase1_and_phase2(&b, &mut setup, no_redecodes).expect("dispatch"),
        Phase2Run::RedecodeBudgetExhausted
    );
    let err = decode_jpeg_symbols_within(&b, &prep, subseq_bits, no_redecodes)
        .expect_err("an exhausted re-decode budget must refuse the stream");
    assert!(err.to_string().contains("re-decodes"), "{err}");

    let (gpu_s_info, _) =
        dispatch_jpeg_phase1_then_phase2(&b, &prep, subseq_bits).expect("dispatch");
    assert_eq!(
        gpu_s_info,
        cpu_jpeg_phase1_then_phase2(&prep, subseq_bits).0
    );
}

// ── JPEG Phase 4 CUDA tests ───────────────────────────────────────────

/// JPEG Phases 1–4 on CUDA decode `GRAY_16X16_JPEG` and the resulting
/// symbol stream matches the CPU oracle (`decode_scan_symbols`).
#[test]
fn jpeg_phase4_cuda_matches_oracle_on_grayscale() {
    use crate::jpeg::test_fixtures::GRAY_16X16_JPEG;
    use crate::jpeg_decoder::decode_scan_symbols;
    let Some(b) = try_cuda() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let prep = prepare_jpeg(GRAY_16X16_JPEG).expect("baseline grayscale must prepare");
    let gpu_syms =
        dispatch_jpeg_phase1_through_phase4(&b, &prep, 32).expect("CUDA phase4 dispatch");
    let cpu_syms = decode_scan_symbols(&prep).expect("CPU oracle");
    assert_eq!(
        gpu_syms.len(),
        cpu_syms.len(),
        "CUDA vs CPU symbol count mismatch"
    );
    for (i, (&g, &c)) in gpu_syms.iter().zip(cpu_syms.iter()).enumerate() {
        assert_eq!(
            g, c,
            "CUDA vs CPU symbol[{i}] mismatch: GPU={g:#04x} CPU={c:#04x}"
        );
    }
}

/// JPEG Phases 1–4 returns the same symbol stream regardless of
/// subsequence granularity (16 vs 32 vs 64 bits).
#[test]
fn jpeg_phase4_cuda_stable_across_subseq_sizes() {
    use crate::jpeg::test_fixtures::GRAY_16X16_JPEG;
    let Some(b) = try_cuda() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let prep = prepare_jpeg(GRAY_16X16_JPEG).expect("baseline grayscale must prepare");
    let syms16 = dispatch_jpeg_phase1_through_phase4(&b, &prep, 16).expect("subseq=16 dispatch");
    let syms32 = dispatch_jpeg_phase1_through_phase4(&b, &prep, 32).expect("subseq=32 dispatch");
    let syms64 = dispatch_jpeg_phase1_through_phase4(&b, &prep, 64).expect("subseq=64 dispatch");
    assert_eq!(syms16, syms32, "subseq 16 vs 32 symbol mismatch");
    assert_eq!(syms32, syms64, "subseq 32 vs 64 symbol mismatch");
}

/// JPEG Phases 1–4 on CUDA decode a 4:4:4 YCbCr image and the resulting
/// symbol stream matches the CPU oracle.  This exercises
/// `blocks_per_mcu=3` (one block per component per MCU), the MCU
/// schedule rotation for `z_in_block=0` → `block_in_mcu` advance, and
/// multi-component AC framing.
#[test]
fn jpeg_phase4_cuda_matches_oracle_on_ycbcr_444() {
    use crate::jpeg_decoder::decode_scan_symbols;
    let Some(b) = try_cuda() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let prep = prepare_jpeg(COLOUR_32X32_444).expect("colour 4:4:4 JPEG must prepare");
    assert_eq!(
        prep.components.len(),
        3,
        "fixture must be 3-component YCbCr"
    );
    let gpu_syms =
        dispatch_jpeg_phase1_through_phase4(&b, &prep, 32).expect("CUDA phase4 dispatch");
    let cpu_syms = decode_scan_symbols(&prep).expect("CPU oracle");
    assert_eq!(
        gpu_syms.len(),
        cpu_syms.len(),
        "CUDA vs CPU symbol count mismatch on YCbCr fixture"
    );
    for (i, (&g, &c)) in gpu_syms.iter().zip(cpu_syms.iter()).enumerate() {
        assert_eq!(
            g, c,
            "CUDA vs CPU symbol[{i}] mismatch on YCbCr: GPU={g:#04x} CPU={c:#04x}"
        );
    }
}

/// JPEG Phases 1–4 on CUDA decode a 4:2:0 YCbCr image spanning
/// several workgroups and match the CPU oracle: six blocks per MCU
/// (four luma, two chroma) and a multi-pass Phase 2.
#[test]
fn jpeg_phase4_cuda_matches_oracle_on_ycbcr_420_across_workgroups() {
    use crate::jpeg_decoder::decode_scan_symbols;
    let Some(b) = try_cuda() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let prep = prepare_jpeg(COLOUR_128X128_420).expect("colour 4:2:0 JPEG must prepare");
    let (_, blocks_per_mcu) = build_mcu_schedule(&prep);
    assert_eq!(blocks_per_mcu, 6, "fixture must be 4:2:0");
    let subseq_bits = 32u32;
    assert!(
        prep.bitstream.length_bits.div_ceil(subseq_bits) > HUFFMAN_THREADS,
        "fixture must span more than one workgroup"
    );
    let gpu_syms =
        dispatch_jpeg_phase1_through_phase4(&b, &prep, subseq_bits).expect("CUDA phase4 dispatch");
    let cpu_syms = decode_scan_symbols(&prep).expect("CPU oracle");
    assert_eq!(gpu_syms, cpu_syms, "CUDA vs CPU symbol stream on 4:2:0");
}

/// JPEG Phases 1–4 returns a stable symbol stream for the 4:4:4 YCbCr
/// fixture across subsequence sizes 16, 32, and 64 bits.  Validates that
/// the MCU schedule rotation (`block_in_mcu` advance on
/// `z_in_block=64`) is consistent regardless of where subsequence
/// boundaries fall.
#[test]
fn jpeg_phase4_cuda_ycbcr_stable_across_subseq_sizes() {
    let Some(b) = try_cuda() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let prep = prepare_jpeg(COLOUR_32X32_444).expect("colour 4:4:4 JPEG must prepare");
    let syms16 = dispatch_jpeg_phase1_through_phase4(&b, &prep, 16).expect("subseq=16 dispatch");
    let syms32 = dispatch_jpeg_phase1_through_phase4(&b, &prep, 32).expect("subseq=32 dispatch");
    let syms64 = dispatch_jpeg_phase1_through_phase4(&b, &prep, 64).expect("subseq=64 dispatch");
    assert_eq!(syms16, syms32, "YCbCr subseq 16 vs 32 symbol mismatch");
    assert_eq!(syms32, syms64, "YCbCr subseq 32 vs 64 symbol mismatch");
}
