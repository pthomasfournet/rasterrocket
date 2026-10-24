//! Vulkan parity tests of the parallel-Huffman dispatchers, against
//! the CPU oracles and against the CUDA backend.

use super::*;
use crate::backend::cuda::CudaBackend;
use crate::backend::vulkan::VulkanBackend;
use crate::jpeg_decoder::phase1_oracle::phase1_walk_snapshot;
use crate::jpeg_decoder::prepare_jpeg;
use crate::jpeg_decoder::tests::fixtures::{book4_codebook, book4_stream};

fn try_vulkan() -> Option<VulkanBackend> {
    VulkanBackend::new().ok()
}

fn try_cuda() -> Option<CudaBackend> {
    CudaBackend::new().ok()
}

/// Run the kernel on both backends + the CPU oracle for every
/// subsequence; assert byte-identical state across all three.
fn cross_backend_parity(stream: &PackedBitstream, tables: &[CanonicalCodebook], subseq_bits: u32) {
    let (Some(cuda), Some(vk)) = (try_cuda(), try_vulkan()) else {
        eprintln!("skipping: need both CUDA and Vulkan");
        return;
    };
    let gpu_cuda =
        dispatch_phase1_intra_sync(&cuda, stream, tables, subseq_bits).expect("cuda dispatch");
    let gpu_vk =
        dispatch_phase1_intra_sync(&vk, stream, tables, subseq_bits).expect("vulkan dispatch");
    let num_subseq = stream.length_bits.div_ceil(subseq_bits);
    for seq_idx in 0..num_subseq {
        let start_bit = seq_idx * subseq_bits;
        let hard_limit = stream.length_bits.min(start_bit + 2 * subseq_bits);
        let count_to = stream.length_bits.min(start_bit + subseq_bits);
        let (cpu, _stop) = phase1_walk_snapshot(stream, tables, start_bit, hard_limit, count_to);
        let i = seq_idx as usize;
        assert_eq!(gpu_cuda[i], cpu, "subseq {seq_idx}: CUDA vs CPU");
        assert_eq!(gpu_vk[i], cpu, "subseq {seq_idx}: Vulkan vs CPU");
        assert_eq!(gpu_cuda[i], gpu_vk[i], "subseq {seq_idx}: CUDA vs Vulkan");
    }
}

/// Vulkan-only parity check (skipped if Vulkan is unavailable but
/// CUDA is — keeps the test useful on Vulkan-only hosts).
fn vulkan_vs_cpu(stream: &PackedBitstream, tables: &[CanonicalCodebook], subseq_bits: u32) {
    let Some(vk) = try_vulkan() else {
        eprintln!("skipping: no Vulkan device");
        return;
    };
    let gpu = dispatch_phase1_intra_sync(&vk, stream, tables, subseq_bits).expect("vulkan");
    let num_subseq = stream.length_bits.div_ceil(subseq_bits);
    for seq_idx in 0..num_subseq {
        let start_bit = seq_idx * subseq_bits;
        let hard_limit = stream.length_bits.min(start_bit + 2 * subseq_bits);
        let count_to = stream.length_bits.min(start_bit + subseq_bits);
        let (cpu, _stop) = phase1_walk_snapshot(stream, tables, start_bit, hard_limit, count_to);
        assert_eq!(gpu[seq_idx as usize], cpu, "subseq {seq_idx} Vulkan vs CPU");
    }
}

#[test]
fn phase1_single_subsequence_vulkan_vs_cpu() {
    let stream = book4_stream(&[0x00; 100]);
    let book = [book4_codebook()];
    vulkan_vs_cpu(&stream, &book, 256);
}

#[test]
fn phase1_many_subsequences_vulkan_vs_cpu() {
    let stream = book4_stream(&[0x00; 1000]);
    let book = [book4_codebook()];
    vulkan_vs_cpu(&stream, &book, 128);
}

#[test]
fn phase1_mixed_symbols_vulkan_vs_cpu() {
    let symbols: Vec<u8> = (0..1000u32).map(|i| (i % 4) as u8).collect();
    let stream = book4_stream(&symbols);
    let book = [book4_codebook()];
    vulkan_vs_cpu(&stream, &book, 128);
}

#[test]
fn phase1_multi_component_vulkan_vs_cpu() {
    let stream = book4_stream(&[0x00; 200]);
    let book = [book4_codebook(), book4_codebook(), book4_codebook()];
    vulkan_vs_cpu(&stream, &book, 64);
}

#[test]
fn cuda_vs_vulkan_single_subsequence() {
    let stream = book4_stream(&[0x00; 100]);
    let book = [book4_codebook()];
    cross_backend_parity(&stream, &book, 256);
}

#[test]
fn cuda_vs_vulkan_many_subsequences() {
    let stream = book4_stream(&[0x00; 1000]);
    let book = [book4_codebook()];
    cross_backend_parity(&stream, &book, 128);
}

#[test]
fn cuda_vs_vulkan_mixed_symbols() {
    let symbols: Vec<u8> = (0..1000u32).map(|i| (i % 4) as u8).collect();
    let stream = book4_stream(&symbols);
    let book = [book4_codebook()];
    cross_backend_parity(&stream, &book, 128);
}

#[test]
fn cuda_vs_vulkan_multi_component() {
    let stream = book4_stream(&[0x00; 200]);
    let book = [book4_codebook(), book4_codebook(), book4_codebook()];
    cross_backend_parity(&stream, &book, 64);
}

/// Phase 2 dispatcher: well-formed stream converges on both
/// backends and produces byte-identical `s_info`.
#[test]
fn phase2_well_formed_stream_cuda_vs_vulkan() {
    let (Some(cuda), Some(vk)) = (try_cuda(), try_vulkan()) else {
        eprintln!("skipping: need both CUDA and Vulkan");
        return;
    };
    let stream = book4_stream(&[0x00; 1024]);
    let book = [book4_codebook()];
    let (cuda_s, cuda_o) = dispatch_phase1_then_phase2(&cuda, &stream, &book, 128).expect("cuda");
    let (vk_s, vk_o) = dispatch_phase1_then_phase2(&vk, &stream, &book, 128).expect("vulkan");
    assert_eq!(cuda_o, vk_o, "outcome mismatch");
    assert_eq!(cuda_s, vk_s, "s_info mismatch");
    assert_eq!(cuda_o, Phase2Outcome::Converged { iterations: 0 });
}

/// Phase 3 dispatcher: same input through both backends yields
/// byte-identical offset vectors, and both match the CPU
/// exclusive-scan reference.
#[test]
fn phase3_offsets_cuda_vs_vulkan() {
    let (Some(cuda), Some(vk)) = (try_cuda(), try_vulkan()) else {
        eprintln!("skipping: need both CUDA and Vulkan");
        return;
    };
    // Mixed symbols to get a non-trivial `n` distribution.
    let symbols: Vec<u8> = (0..1000u32).map(|i| (i % 4) as u8).collect();
    let stream = book4_stream(&symbols);
    let book = [book4_codebook()];
    let (cuda_s, _) =
        dispatch_phase1_then_phase2(&cuda, &stream, &book, 128).expect("cuda phase1+2");
    let (vk_s, _) = dispatch_phase1_then_phase2(&vk, &stream, &book, 128).expect("vulkan phase1+2");
    // Pin upstream agreement before pinning Phase 3 — otherwise a
    // Phase 1/2 divergence on the mixed corpus would surface as a
    // confusing "Vulkan diverges from CPU oracle" (the CPU oracle
    // is built from CUDA's counts).
    assert_eq!(
        cuda_s, vk_s,
        "Phase 1+2 must agree before Phase 3 can be compared"
    );
    let cuda_off = dispatch_phase3_offsets(&cuda, &cuda_s).expect("cuda phase3");
    let vk_off = dispatch_phase3_offsets(&vk, &vk_s).expect("vulkan phase3");

    let counts: Vec<u32> = cuda_s.iter().map(|s| s.n).collect();
    let cpu_off = crate::jpeg_decoder::scan::test_helpers::cpu_exclusive_scan(&counts);

    assert_eq!(cuda_off, cpu_off, "CUDA diverges from CPU oracle");
    assert_eq!(vk_off, cpu_off, "Vulkan diverges from CPU oracle");
    assert_eq!(cuda_off, vk_off, "CUDA and Vulkan disagree");
}

/// End-to-end Phases 1..4 on both backends; the decoded symbol
/// streams must be byte-identical and equal to the original
/// input. Uniform-length corpus stays within Phase 2's retry
/// bound on both backends.
#[test]
fn end_to_end_uniform_cuda_vs_vulkan() {
    let (Some(cuda), Some(vk)) = (try_cuda(), try_vulkan()) else {
        eprintln!("skipping: need both CUDA and Vulkan");
        return;
    };
    let symbols: Vec<u8> = vec![0x00; 1024];
    let stream = book4_stream(&symbols);
    let book = [book4_codebook()];
    let cuda_decoded =
        dispatch_phase1_through_phase4(&cuda, &stream, &book, 128).expect("cuda e2e");
    let vk_decoded = dispatch_phase1_through_phase4(&vk, &stream, &book, 128).expect("vulkan e2e");

    assert_eq!(cuda_decoded, vk_decoded, "CUDA and Vulkan diverge");
    let got: Vec<u8> = cuda_decoded
        .iter()
        .map(|&v| u8::try_from(v & 0xFF).expect("symbol fits u8"))
        .collect();
    assert_eq!(got, symbols, "decoded stream diverges from input");
}

// ── JPEG Phase 1 cross-backend parity tests ───────────────────────────

use crate::jpeg_decoder::build_mcu_schedule;
use crate::jpeg_decoder::phase1_oracle::phase1_jpeg_walk_snapshot;

/// JPEG Phase 1 CUDA vs Vulkan: both backends must produce byte-
/// identical per-subseq state on the grayscale fixture.
#[test]
fn jpeg_phase1_cuda_vs_vulkan_grayscale() {
    use crate::jpeg::test_fixtures::GRAY_16X16_JPEG;
    let (Some(cuda), Some(vk)) = (try_cuda(), try_vulkan()) else {
        eprintln!("skipping: need both CUDA and Vulkan");
        return;
    };
    let prep = prepare_jpeg(GRAY_16X16_JPEG).expect("baseline grayscale must prepare");
    let subsequence_bits = 32u32;
    let cuda_states =
        dispatch_jpeg_phase1_intra_sync(&cuda, &prep, subsequence_bits).expect("CUDA JPEG Phase 1");
    let vk_states =
        dispatch_jpeg_phase1_intra_sync(&vk, &prep, subsequence_bits).expect("Vulkan JPEG Phase 1");

    assert_eq!(
        cuda_states.len(),
        vk_states.len(),
        "state count mismatch between CUDA and Vulkan"
    );
    for (i, (cs, vs)) in cuda_states.iter().zip(vk_states.iter()).enumerate() {
        assert_eq!(
            cs, vs,
            "subsequence {i}: CUDA vs Vulkan JPEG Phase 1 mismatch"
        );
    }
}

/// JPEG Phase 1 Vulkan vs CPU oracle: each subseq's GPU state must
/// match the host-side `phase1_jpeg_walk_snapshot` result.
#[test]
fn jpeg_phase1_vulkan_matches_cpu_oracle_grayscale() {
    use crate::jpeg::test_fixtures::GRAY_16X16_JPEG;
    let Some(vk) = try_vulkan() else {
        eprintln!("skipping: no Vulkan device");
        return;
    };
    let prep = prepare_jpeg(GRAY_16X16_JPEG).expect("baseline grayscale must prepare");
    let (mcu_sched, blocks_per_mcu) = build_mcu_schedule(&prep);
    let subsequence_bits = 32u32;
    let gpu_states =
        dispatch_jpeg_phase1_intra_sync(&vk, &prep, subsequence_bits).expect("Vulkan JPEG Phase 1");

    let dc_refs = prep.dc_codebooks_for_dispatch();
    let ac_refs = prep.ac_codebooks_for_dispatch();
    let num_subseq = prep.bitstream.length_bits.div_ceil(subsequence_bits);

    for seq_idx in 0..num_subseq {
        let start_bit = seq_idx * subsequence_bits;
        let hard_limit = prep
            .bitstream
            .length_bits
            .min(start_bit + 2 * subsequence_bits);
        let count_to = prep.bitstream.length_bits.min(start_bit + subsequence_bits);
        let (cpu, _) = phase1_jpeg_walk_snapshot(
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
            "subsequence {seq_idx}: Vulkan vs CPU JPEG Phase 1 mismatch"
        );
    }
}

// ── JPEG Phase 2 cross-backend tests ──────────────────────────────────

/// JPEG Phase 1+2 on Vulkan converges on the grayscale fixture.
#[test]
fn jpeg_phase2_vulkan_converges_on_grayscale() {
    use crate::jpeg::test_fixtures::GRAY_16X16_JPEG;
    let Some(vk) = try_vulkan() else {
        eprintln!("skipping: no Vulkan device");
        return;
    };
    let prep = prepare_jpeg(GRAY_16X16_JPEG).expect("baseline grayscale must prepare");
    let (s_info, outcome) = dispatch_jpeg_phase1_then_phase2(&vk, &prep, 32).expect("dispatch ok");
    let num_subseq = prep.bitstream.length_bits.div_ceil(32);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "s_info.len() is a subsequence count on a 16×16 test fixture, far below u32::MAX"
    )]
    let s_info_count = s_info.len() as u32;
    assert_eq!(s_info_count, num_subseq);
    assert!(
        matches!(outcome, Phase2Outcome::Converged { .. }),
        "expected Vulkan convergence on grayscale, got {outcome:?}"
    );
}

/// JPEG Phase 1+2 CUDA and Vulkan produce byte-identical `s_info`
/// after convergence on the grayscale fixture.
#[test]
fn jpeg_phase2_cuda_vs_vulkan_grayscale() {
    use crate::jpeg::test_fixtures::GRAY_16X16_JPEG;
    let (Some(cuda), Some(vk)) = (try_cuda(), try_vulkan()) else {
        eprintln!("skipping: need both CUDA and Vulkan");
        return;
    };
    let prep = prepare_jpeg(GRAY_16X16_JPEG).expect("baseline grayscale must prepare");
    let (cuda_s, cuda_o) =
        dispatch_jpeg_phase1_then_phase2(&cuda, &prep, 32).expect("cuda dispatch");
    let (vk_s, vk_o) = dispatch_jpeg_phase1_then_phase2(&vk, &prep, 32).expect("vulkan dispatch");
    // Pin outcome agreement first so a backend divergence in
    // sync detection surfaces as a "outcome mismatch" before the
    // potentially-confusing byte-diff.
    assert_eq!(
        cuda_o, vk_o,
        "Phase 2 outcome mismatch: CUDA={cuda_o:?} Vulkan={vk_o:?}"
    );
    assert_eq!(
        cuda_s, vk_s,
        "Phase 2 s_info mismatch: CUDA and Vulkan disagree"
    );
}

/// JPEG Phase 1+2 CUDA and Vulkan produce byte-identical `s_info`
/// after convergence on the 3-component fixture — the case where
/// Phase 2 must genuinely propagate corrected block phases, so a
/// Slang/CUDA divergence in the re-decode walk cannot hide behind
/// an already-synced Phase 1 output.
#[test]
fn jpeg_phase2_cuda_vs_vulkan_ycbcr() {
    static COLOUR_32X32_444: &[u8] =
        include_bytes!("../../../../../tests/fixtures/jpeg/colour_32x32_444.jpg");
    cuda_vs_vulkan_jpeg_phase2("32x32 4:4:4", COLOUR_32X32_444, 32);
}

/// Same on inputs spanning several workgroups, so the two
/// backends' on-device local loops and host pass counts must
/// agree — 4:4:4 and 4:2:0 framing both.
#[test]
fn jpeg_phase2_cuda_vs_vulkan_across_workgroups() {
    static COLOUR_128X128_444: &[u8] =
        include_bytes!("../../../../../tests/fixtures/jpeg/colour_128x128_444.jpg");
    static COLOUR_128X128_420: &[u8] =
        include_bytes!("../../../../../tests/fixtures/jpeg/colour_128x128_420.jpg");
    cuda_vs_vulkan_jpeg_phase2("128x128 4:4:4", COLOUR_128X128_444, 16);
    cuda_vs_vulkan_jpeg_phase2("128x128 4:2:0", COLOUR_128X128_420, 16);
}

/// JPEG Phase 1+2 on both backends: outcomes equal, both
/// converged, `s_info` byte-identical.
fn cuda_vs_vulkan_jpeg_phase2(name: &str, bytes: &[u8], subseq_bits: u32) {
    let (Some(cuda), Some(vk)) = (try_cuda(), try_vulkan()) else {
        eprintln!("skipping: need both CUDA and Vulkan");
        return;
    };
    let prep = prepare_jpeg(bytes).expect("colour JPEG must prepare");
    let (cuda_s, cuda_o) =
        dispatch_jpeg_phase1_then_phase2(&cuda, &prep, subseq_bits).expect("cuda dispatch");
    let (vk_s, vk_o) =
        dispatch_jpeg_phase1_then_phase2(&vk, &prep, subseq_bits).expect("vulkan dispatch");
    assert_eq!(
        cuda_o, vk_o,
        "{name}: Phase 2 outcome mismatch: CUDA={cuda_o:?} Vulkan={vk_o:?}"
    );
    assert!(
        matches!(cuda_o, Phase2Outcome::Converged { .. }),
        "{name}: expected convergence, got {cuda_o:?}"
    );
    assert_eq!(
        cuda_s, vk_s,
        "{name}: Phase 2 s_info mismatch: CUDA and Vulkan disagree"
    );
}

// ── JPEG Phase 4 cross-backend tests ──────────────────────────────────

/// JPEG Phases 1–4 on Vulkan decodes `GRAY_16X16_JPEG` and the symbol
/// stream matches the CPU oracle.
#[test]
fn jpeg_phase4_vulkan_matches_oracle_on_grayscale() {
    use crate::jpeg::test_fixtures::GRAY_16X16_JPEG;
    use crate::jpeg_decoder::decode_scan_symbols;
    let Some(vk) = try_vulkan() else {
        eprintln!("skipping: no Vulkan device");
        return;
    };
    let prep = prepare_jpeg(GRAY_16X16_JPEG).expect("baseline grayscale must prepare");
    let gpu_syms =
        dispatch_jpeg_phase1_through_phase4(&vk, &prep, 32).expect("Vulkan phase4 dispatch");
    let cpu_syms = decode_scan_symbols(&prep).expect("CPU oracle");
    assert_eq!(
        gpu_syms.len(),
        cpu_syms.len(),
        "Vulkan vs CPU symbol count mismatch"
    );
    for (i, (&g, &c)) in gpu_syms.iter().zip(cpu_syms.iter()).enumerate() {
        assert_eq!(
            g, c,
            "Vulkan vs CPU symbol[{i}] mismatch: GPU={g:#04x} CPU={c:#04x}"
        );
    }
}

/// JPEG Phases 1–4 CUDA and Vulkan produce byte-identical symbol
/// streams on `GRAY_16X16_JPEG`.
#[test]
fn jpeg_phase4_cuda_vs_vulkan_grayscale() {
    use crate::jpeg::test_fixtures::GRAY_16X16_JPEG;
    let (Some(cuda), Some(vk)) = (try_cuda(), try_vulkan()) else {
        eprintln!("skipping: need both CUDA and Vulkan");
        return;
    };
    let prep = prepare_jpeg(GRAY_16X16_JPEG).expect("baseline grayscale must prepare");
    let cuda_syms = dispatch_jpeg_phase1_through_phase4(&cuda, &prep, 32).expect("cuda phase4");
    let vk_syms = dispatch_jpeg_phase1_through_phase4(&vk, &prep, 32).expect("vulkan phase4");
    assert_eq!(
        cuda_syms, vk_syms,
        "JPEG Phase 4 symbol stream: CUDA vs Vulkan mismatch"
    );
}
