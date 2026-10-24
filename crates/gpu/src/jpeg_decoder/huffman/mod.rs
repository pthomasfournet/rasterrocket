//! Host-side dispatch for the parallel-Huffman JPEG decoder kernels.
//!
//! `flatten_codebooks` packs `CanonicalCodebook`s into the device-side
//! `u32[num_components * 65536]` layout the kernels expect.
//! [`dispatch_jpeg_phase1_through_phase4`] is the production entry
//! point; the synthetic-stream dispatchers in `synthetic.rs` serve the
//! oracle-parity tests only.

// Per-phase parity tests live in this file's `mod tests` /
// `mod vulkan_tests`; the 10-vector adversarial-corpus acceptance
// tests live in the sibling `corpus.rs`.
#[cfg(all(test, feature = "gpu-validation"))]
mod corpus;
#[cfg(all(test, feature = "gpu-validation"))]
mod synthetic;
#[cfg(all(test, feature = "gpu-validation"))]
use synthetic::{
    dispatch_phase1_intra_sync, dispatch_phase1_then_phase2, dispatch_phase1_through_phase4,
    dispatch_phase3_offsets,
};

use crate::backend::params::{
    HUFFMAN_CODEBOOK_ENTRIES, HUFFMAN_THREADS, HuffmanParams, HuffmanPhase,
};
use crate::backend::{BackendError, GpuBackend, Result};
use crate::jpeg::CanonicalCodebook;
#[cfg(all(test, feature = "gpu-validation"))]
use crate::jpeg_decoder::PackedBitstream;
use crate::jpeg_decoder::dispatch_util::DeviceBufferGuard;
use crate::jpeg_decoder::phase1_oracle::SubsequenceState;
#[cfg(test)]
use crate::jpeg_decoder::phase2_oracle::Phase2Outcome;

/// Pass bound for the JPEG-framed Phase 2 propagation loop: the
/// number of kernel workgroups, `ceil(n / HUFFMAN_THREADS)`.
///
/// Slot 0 is pinned correct (its fresh start is the true stream
/// start). Each pass, every workgroup propagates to a local fixpoint
/// from its left input (the previous pass's last slot of the
/// workgroup before it), so it leaves holding the exact chain from
/// that input. Workgroup `g` is therefore final after pass `g`, all
/// `G` workgroups after pass `G - 1`, and pass `G` observes every
/// slot as a fixpoint. The `0..=bound` dispatch loop runs at most
/// `G + 1` passes and always detects convergence — the re-decode is
/// deterministic even for erroring streams, so the non-convergence
/// error path is defence-in-depth only.
///
/// Single definition shared with the CPU oracle
/// (`phase2_oracle::jpeg_phase2_run_to_sync`) so the bound embedded
/// in compared `Phase2Outcome`s cannot drift between the two.
pub(in crate::jpeg_decoder) const fn jpeg_phase2_retry_bound(num_subsequences: u32) -> u32 {
    num_subsequences.div_ceil(HUFFMAN_THREADS)
}

/// Flatten an iterator of `&CanonicalCodebook` into the GPU layout:
/// `u32[num_components * 65536]`, each entry packing `(num_bits << 8) | symbol`.
///
/// `num_bits == 0` (no codeword matches the prefix) is preserved verbatim —
/// the kernel uses that as its miss sentinel.
fn flatten_codebooks<'a>(
    codebooks: impl ExactSizeIterator<Item = &'a CanonicalCodebook>,
) -> Vec<u32> {
    let mut out = Vec::with_capacity(codebooks.len() * HUFFMAN_CODEBOOK_ENTRIES);
    for book in codebooks {
        for entry in book.table() {
            out.push((u32::from(entry.num_bits) << 8) | u32::from(entry.symbol));
        }
    }
    out
}

#[cfg(all(test, feature = "gpu-validation"))]
#[must_use]
fn build_gpu_codebook(codebooks: &[CanonicalCodebook]) -> Vec<u32> {
    flatten_codebooks(codebooks.iter())
}

#[must_use]
fn build_gpu_codebook_refs(codebooks: &[&CanonicalCodebook]) -> Vec<u32> {
    flatten_codebooks(codebooks.iter().copied())
}

/// Device buffers and derived dimensions shared by every JPEG-framed
/// one-shot dispatch: entropy bitstream, flat AC/DC codebooks, MCU
/// schedule, the ping-ponged pair of per-subsequence state buffers,
/// and the Phase 2 fixpoint flags.
///
/// Owns the alloc/upload preamble shared by every JPEG-framed
/// dispatcher; the phase flow stays explicit at each call site.
/// `s_info` always names the most recently written state buffer —
/// [`run_jpeg_phase1_and_phase2`] swaps the pair after each pass.
struct JpegDispatchSetup<'a, B: GpuBackend> {
    bitstream: DeviceBufferGuard<'a, B>,
    codebook: DeviceBufferGuard<'a, B>,
    dc_codebook: DeviceBufferGuard<'a, B>,
    mcu_schedule: DeviceBufferGuard<'a, B>,
    s_info: DeviceBufferGuard<'a, B>,
    s_info_scratch: DeviceBufferGuard<'a, B>,
    sync_flags: DeviceBufferGuard<'a, B>,
    length_bits: u32,
    subsequence_bits: u32,
    num_subsequences: u32,
    num_components: u32,
    blocks_per_mcu: u32,
}

impl<'a, B: GpuBackend> JpegDispatchSetup<'a, B> {
    /// Allocate and upload everything the JPEG-framed phases share.
    ///
    /// # Errors
    /// Returns `BackendError` when `subsequence_bits` is zero, the
    /// component count does not fit `u32`, or any allocation/upload
    /// fails.
    fn new(
        backend: &'a B,
        prep: &crate::jpeg_decoder::JpegPreparedInput,
        subsequence_bits: u32,
    ) -> Result<Self> {
        if subsequence_bits == 0 {
            return Err(BackendError::msg(
                "jpeg huffman dispatch: subsequence_bits must be > 0",
            ));
        }
        let (mcu_sched_host, blocks_per_mcu) = crate::jpeg_decoder::build_mcu_schedule(prep);

        let ac_flat = build_gpu_codebook_refs(&prep.ac_codebooks_for_dispatch());
        let dc_flat = build_gpu_codebook_refs(&prep.dc_codebooks_for_dispatch());

        let num_components = u32::try_from(prep.components.len())
            .map_err(|_| BackendError::msg("components.len() does not fit in u32"))?;
        let length_bits = prep.bitstream.length_bits;
        let num_subsequences = length_bits.div_ceil(subsequence_bits);

        // peek16 reads `word_idx + 1` even at the stream tail, so the
        // bitstream buffer needs one zero word of headroom.
        let bitstream_bytes = ((length_bits.div_ceil(32) as usize) + 1) * 4;
        let s_info_bytes = (num_subsequences as usize) * std::mem::size_of::<SubsequenceState>();
        let flags_bytes = (num_subsequences as usize) * 4;

        let bitstream = DeviceBufferGuard::alloc_zeroed(backend, bitstream_bytes)?;
        let codebook = DeviceBufferGuard::alloc(backend, ac_flat.len() * 4)?;
        let dc_codebook = DeviceBufferGuard::alloc(backend, dc_flat.len() * 4)?;
        let mcu_schedule = DeviceBufferGuard::alloc(backend, mcu_sched_host.len() * 4)?;
        let s_info = DeviceBufferGuard::alloc(backend, s_info_bytes)?;
        let s_info_scratch = DeviceBufferGuard::alloc(backend, s_info_bytes)?;
        // Zeroed so a kernel guard that returns without writing (e.g.
        // the defensive blocks_per_mcu check) reads back as
        // all-unsynced, never as spurious convergence.
        let sync_flags = DeviceBufferGuard::alloc_zeroed(backend, flags_bytes)?;

        let _up1 = backend.upload_async(
            bitstream.as_ref(),
            bytemuck::cast_slice(&prep.bitstream.words),
        )?;
        let _up2 = backend.upload_async(codebook.as_ref(), bytemuck::cast_slice(&ac_flat))?;
        let _up3 = backend.upload_async(dc_codebook.as_ref(), bytemuck::cast_slice(&dc_flat))?;
        let _up4 =
            backend.upload_async(mcu_schedule.as_ref(), bytemuck::cast_slice(&mcu_sched_host))?;

        Ok(Self {
            bitstream,
            codebook,
            dc_codebook,
            mcu_schedule,
            s_info,
            s_info_scratch,
            sync_flags,
            length_bits,
            subsequence_bits,
            num_subsequences,
            num_components,
            blocks_per_mcu,
        })
    }

    /// `HuffmanParams` skeleton with the shared buffers and dims; the
    /// per-phase helpers below fill in the phase-specific fields.
    const fn base_params(&self, phase: HuffmanPhase) -> HuffmanParams<'_, B> {
        HuffmanParams {
            bitstream: self.bitstream.as_ref(),
            codebook: self.codebook.as_ref(),
            s_info: self.s_info.as_ref(),
            sync_flags: None,
            s_info_prev: None,
            offsets: None,
            symbols_out: None,
            decode_status: None,
            dc_codebook: Some(self.dc_codebook.as_ref()),
            mcu_schedule: Some(self.mcu_schedule.as_ref()),
            length_bits: self.length_bits,
            subsequence_bits: self.subsequence_bits,
            num_components: self.num_components,
            total_symbols: 0,
            blocks_per_mcu: self.blocks_per_mcu,
            phase,
        }
    }

    /// Record and run JPEG Phase 1 into `s_info`.
    #[cfg(all(test, feature = "gpu-validation"))]
    fn run_phase1(&self, backend: &B) -> Result<()> {
        backend.begin_page()?;
        backend.record_huffman(self.base_params(HuffmanPhase::JpegPhase1IntraSync))?;
        let fence = backend.submit_page()?;
        backend.wait_page(fence)
    }

    /// Download the current `s_info` states.
    fn download_s_info(&self, backend: &B) -> Result<Vec<SubsequenceState>> {
        let mut out = vec![
            <SubsequenceState as bytemuck::Zeroable>::zeroed();
            self.num_subsequences as usize
        ];
        let dst = bytemuck::cast_slice_mut::<SubsequenceState, u8>(&mut out);
        let handle = backend.download_async(self.s_info.as_ref(), dst)?;
        backend.wait_download(handle)?;
        Ok(out)
    }

    /// Hand every buffer back through `free_device` (the happy-path
    /// counterpart of the guards' Drop-frees).
    fn free_all(self, backend: &B) {
        backend.free_device(self.bitstream.take());
        backend.free_device(self.codebook.take());
        backend.free_device(self.dc_codebook.take());
        backend.free_device(self.mcu_schedule.take());
        backend.free_device(self.s_info.take());
        backend.free_device(self.s_info_scratch.take());
        backend.free_device(self.sync_flags.take());
    }
}

/// Run JPEG Phase 1, then drive the Phase 2 propagation to
/// convergence: each pass is one `JpegPhase2InterSync` dispatch reading
/// the previous pass's states (`setup.s_info`) and writing the
/// recomputed ones (`setup.s_info_scratch`), followed by a buffer swap
/// — so
/// `setup.s_info` always holds the last-written states, and on
/// convergence the two buffers are identical (a fixpoint pass changes
/// nothing). Propagation within a workgroup happens on-device inside
/// the dispatch; the host loop only carries truth across workgroup
/// boundaries.
///
/// Phase 1 is recorded in the first pass's page: the recorder orders
/// consecutive dispatches of one page, so the only host wait per pass
/// is the one the convergence check needs.
///
/// Returns `Some(iterations)` (0-based index of the converged pass) or
/// `None` when the [`jpeg_phase2_retry_bound`] pass budget runs out —
/// unreachable for a correct kernel (the re-decode is deterministic,
/// so even erroring streams stabilise); kept as defence-in-depth.
///
/// One loop shared by the oracle-parity dispatcher and the production
/// Phase 1–4 dispatcher, so the code the parity tests exercise is the
/// code production runs.
fn run_jpeg_phase1_and_phase2<B: GpuBackend>(
    backend: &B,
    setup: &mut JpegDispatchSetup<'_, B>,
) -> Result<Option<u32>> {
    let bound = jpeg_phase2_retry_bound(setup.num_subsequences);
    let mut flags_host = vec![0u32; setup.num_subsequences as usize];
    for iter in 0..=bound {
        backend.begin_page()?;
        if iter == 0 {
            backend.record_huffman(setup.base_params(HuffmanPhase::JpegPhase1IntraSync))?;
        }
        let mut params = setup.base_params(HuffmanPhase::JpegPhase2InterSync);
        params.s_info = setup.s_info_scratch.as_ref();
        params.s_info_prev = Some(setup.s_info.as_ref());
        params.sync_flags = Some(setup.sync_flags.as_ref());
        backend.record_huffman(params)?;
        let fence = backend.submit_page()?;
        backend.wait_page(fence)?;

        let dst = bytemuck::cast_slice_mut::<u32, u8>(&mut flags_host);
        let handle = backend.download_async(setup.sync_flags.as_ref(), dst)?;
        backend.wait_download(handle)?;

        std::mem::swap(&mut setup.s_info, &mut setup.s_info_scratch);
        if flags_host.iter().all(|&f| f == 1) {
            return Ok(Some(iter));
        }
    }
    Ok(None)
}

/// True when the converged chain's terminal snapshot stopped short of
/// the stream tail: a decode error inside any region stalls the
/// propagation at the error bit — every downstream snapshot collapses
/// to it, and Phase 4, replaying exactly the walks that produced the
/// snapshots, exits cleanly at `end_p` without re-encountering the
/// error. The stall is only observable here.
///
/// The tail allowance is one maximal symbol (16-bit codeword + 11
/// magnitude bits): byte-padding legitimately ends every stream in a
/// decode error within that distance of `length_bits`. For wire input
/// the CPU pre-pass has already decoded the full stream, so a
/// mid-stream stall indicates a kernel bug or GPU fault, not a
/// malformed file.
const fn jpeg_stream_stalled(terminal_p: u32, length_bits: u32) -> bool {
    const MAX_SYMBOL_BITS: u32 = 16 + 11;
    terminal_p.saturating_add(MAX_SYMBOL_BITS) < length_bits
}

/// One-shot JPEG Phase 1 dispatch using real JPEG framing.
///
/// Builds the MCU schedule + flat codebook buffers from `prep`, uploads
/// them alongside the entropy bitstream, runs `JpegPhase1IntraSync`,
/// downloads the per-subsequence state, and frees all device buffers.
///
/// # Errors
/// Returns `BackendError` for any alloc / upload / dispatch / download
/// failure.
#[cfg(all(test, feature = "gpu-validation"))]
fn dispatch_jpeg_phase1_intra_sync<B: GpuBackend>(
    backend: &B,
    prep: &crate::jpeg_decoder::JpegPreparedInput,
    subsequence_bits: u32,
) -> Result<Vec<SubsequenceState>> {
    if prep.bitstream.length_bits == 0 {
        return Ok(Vec::new());
    }
    let setup = JpegDispatchSetup::new(backend, prep, subsequence_bits)?;
    setup.run_phase1(backend)?;
    let out = setup.download_s_info(backend)?;
    setup.free_all(backend);
    Ok(out)
}

/// One-shot JPEG Phase 1 + Phase 2 dispatch using real JPEG framing.
///
/// Runs JPEG Phase 1, then the shared Phase 2 propagation loop.
/// Returns the final `s_info` and the `Phase2Outcome`.
///
/// Test/oracle path only — production callers should drive the trait
/// directly to keep buffers alive across pages.
///
/// # Errors
/// Returns `BackendError` if any alloc/upload/dispatch/download
/// fails. Bound-exceeded is reported via the outcome, not as an error.
#[cfg(all(test, feature = "gpu-validation"))]
fn dispatch_jpeg_phase1_then_phase2<B: GpuBackend>(
    backend: &B,
    prep: &crate::jpeg_decoder::JpegPreparedInput,
    subsequence_bits: u32,
) -> Result<(Vec<SubsequenceState>, Phase2Outcome)> {
    if prep.bitstream.length_bits == 0 {
        return Ok((Vec::new(), Phase2Outcome::Converged { iterations: 0 }));
    }
    let mut setup = JpegDispatchSetup::new(backend, prep, subsequence_bits)?;
    let outcome = match run_jpeg_phase1_and_phase2(backend, &mut setup)? {
        Some(iterations) => Phase2Outcome::Converged { iterations },
        None => Phase2Outcome::SyncBoundExceeded {
            bound: jpeg_phase2_retry_bound(setup.num_subsequences),
        },
    };
    let s_info_out = setup.download_s_info(backend)?;
    setup.free_all(backend);
    Ok((s_info_out, outcome))
}

/// Result of one parallel-Huffman decode of a JPEG entropy stream.
#[derive(Debug)]
#[expect(
    clippy::redundant_pub_crate,
    reason = "pub(crate) is intentional: parent module is pub(crate); inner items need explicit visibility for documentation and grepping"
)]
pub(crate) struct JpegSymbolDecode {
    /// Decoded Huffman symbol stream, one `u32` per symbol.
    pub symbols: Vec<u32>,
    /// Phase 2 dispatches (host round trips) the propagation took.
    pub phase2_passes: u32,
}

/// Test wrapper over [`decode_jpeg_symbols`] returning the symbols only.
///
/// # Errors
/// As [`decode_jpeg_symbols`].
#[cfg(all(test, feature = "gpu-validation"))]
fn dispatch_jpeg_phase1_through_phase4<B: GpuBackend>(
    backend: &B,
    prep: &crate::jpeg_decoder::JpegPreparedInput,
    subsequence_bits: u32,
) -> Result<Vec<u32>> {
    decode_jpeg_symbols(backend, prep, subsequence_bits).map(|d| d.symbols)
}

/// Run all four JPEG-framed phases and return the decoded symbol
/// stream.
///
/// Uses the JPEG-framed kernels: `JpegPhase1IntraSync`, the shared
/// `JpegPhase2InterSync` propagation loop, CPU-side exclusive scan for
/// Phase 3, and `JpegPhase4Redecode`.
///
/// # Cost model
///
/// With `n = ceil(length_bits / subsequence_bits)` subsequences and
/// `W = HUFFMAN_THREADS` lanes per workgroup, Phases 1 and 4
/// are one dispatch each with per-thread work of one or two regions.
/// Phase 2 is `1 + ceil(r / W)` synchronous host round trips, where
/// `r` is the longest run of subsequences whose fresh-start Phase 1
/// chain never re-synchronises with the true decode; each round trip
/// costs up to `min(r, W)` sequential region re-decodes on-device.
/// Grayscale streams have `r ≈ 0` (the block phase is trivially
/// right and the zig-zag index self-corrects at the next EOB).
/// Multi-component streams have `r` bounded only by the content:
/// short optimised codes over flat regions — document scans — never
/// re-synchronise a wrong block phase, so `r ≈ n` and the phase
/// costs `n` sequential region re-decodes spread over `ceil(n / W)`
/// round trips, which is slower than the CPU walk for any stream
/// size. The dispatcher enforces no cap; a caller routing by cost
/// must decide from `prep.components.len()` and `length_bits`.
///
/// # Errors
/// Returns `BackendError` if any allocation, upload, kernel dispatch,
/// or download fails; if Phase 2 does not converge within the retry
/// bound; if the converged chain stalled mid-stream on a decode error
/// (see [`jpeg_stream_stalled`]); or if Phase 4 reports a per-subseq
/// decode failure.
#[expect(
    clippy::redundant_pub_crate,
    reason = "pub(crate) is intentional: parent module is pub(crate); inner items need explicit visibility for documentation and grepping"
)]
pub(crate) fn decode_jpeg_symbols<B: GpuBackend>(
    backend: &B,
    prep: &crate::jpeg_decoder::JpegPreparedInput,
    subsequence_bits: u32,
) -> Result<JpegSymbolDecode> {
    if prep.bitstream.length_bits == 0 {
        return Ok(JpegSymbolDecode {
            symbols: Vec::new(),
            phase2_passes: 0,
        });
    }
    let mut setup = JpegDispatchSetup::new(backend, prep, subsequence_bits)?;
    let Some(converged_pass) = run_jpeg_phase1_and_phase2(backend, &mut setup)? else {
        return Err(BackendError::msg(
            "dispatch_jpeg_phase1_through_phase4: Phase 2 did not converge within retry bound",
        ));
    };
    let phase2_passes = converged_pass + 1;

    // Download s_info for CPU-side Phase 3 (exclusive scan) and the
    // stall check.
    let s_info_host = setup.download_s_info(backend)?;

    if let Some(last) = s_info_host.last()
        && jpeg_stream_stalled(last.p, setup.length_bits)
    {
        return Err(BackendError::msg(format!(
            "dispatch_jpeg_phase1_through_phase4: decode stalled at bit {} of {} — \
             mid-stream decode error propagated through the chain",
            last.p, setup.length_bits,
        )));
    }

    // Phase 3: CPU exclusive prefix scan over per-subsequence symbol counts.
    let counts: Vec<u32> = s_info_host.iter().map(|s| s.n).collect();
    let (offsets_host, total_symbols) = crate::jpeg_decoder::scan::exclusive_scan_checked(&counts)
        .ok_or_else(|| {
            BackendError::msg(
                "dispatch_jpeg_phase1_through_phase4: total symbol count overflows u32",
            )
        })?;

    if total_symbols == 0 {
        setup.free_all(backend);
        return Ok(JpegSymbolDecode {
            symbols: Vec::new(),
            phase2_passes,
        });
    }

    let flags_bytes = (setup.num_subsequences as usize) * 4;
    let offsets_buf = DeviceBufferGuard::alloc(backend, flags_bytes)?;
    let _up5 = backend.upload_async(offsets_buf.as_ref(), bytemuck::cast_slice(&offsets_host))?;

    let symbols_buf = DeviceBufferGuard::alloc_zeroed(backend, (total_symbols as usize) * 4)?;
    // Zero-init = Phase4FailureKind::Ok, so a subsequence the kernel
    // does not reach reads back clean.
    let decode_status_buf = DeviceBufferGuard::alloc_zeroed(backend, flags_bytes)?;

    // Phase 4.
    backend.begin_page()?;
    let mut params = setup.base_params(HuffmanPhase::JpegPhase4Redecode);
    params.offsets = Some(offsets_buf.as_ref());
    params.symbols_out = Some(symbols_buf.as_ref());
    params.decode_status = Some(decode_status_buf.as_ref());
    params.total_symbols = total_symbols;
    backend.record_huffman(params)?;
    let fence = backend.submit_page()?;
    backend.wait_page(fence)?;

    check_phase4_status(
        backend,
        &decode_status_buf,
        setup.num_subsequences,
        "dispatch_jpeg_phase1_through_phase4",
    )?;
    let symbols_host = download_u32s(backend, &symbols_buf, total_symbols as usize)?;

    setup.free_all(backend);
    backend.free_device(offsets_buf.take());
    backend.free_device(symbols_buf.take());
    backend.free_device(decode_status_buf.take());

    Ok(JpegSymbolDecode {
        symbols: symbols_host,
        phase2_passes,
    })
}

/// Download the first `len` `u32`s of `buf`.
fn download_u32s<B: GpuBackend>(
    backend: &B,
    buf: &DeviceBufferGuard<'_, B>,
    len: usize,
) -> Result<Vec<u32>> {
    let mut host = vec![0u32; len];
    let dst = bytemuck::cast_slice_mut::<u32, u8>(&mut host);
    let handle = backend.download_async(buf.as_ref(), dst)?;
    backend.wait_download(handle)?;
    Ok(host)
}

/// Download the per-subsequence Phase 4 exit codes and surface the
/// first non-`Ok` one as a typed error; `who` names the dispatcher.
fn check_phase4_status<B: GpuBackend>(
    backend: &B,
    decode_status: &DeviceBufferGuard<'_, B>,
    num_subsequences: u32,
    who: &str,
) -> Result<()> {
    let status_host = download_u32s(backend, decode_status, num_subsequences as usize)?;
    if let Some((failing_idx, &raw)) = status_host.iter().enumerate().find(|&(_, &s)| s != 0) {
        let kind = crate::backend::params::Phase4FailureKind::from_u32(raw);
        return Err(BackendError::msg(format!(
            "{who}: Phase 4 decode failed on subseq {failing_idx}: {} (status code {raw})",
            kind.label(),
        )));
    }
    Ok(())
}

#[cfg(all(test, feature = "gpu-validation"))]
mod tests {
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
    fn assert_phase1_parity(
        stream: &PackedBitstream,
        tables: &[CanonicalCodebook],
        subseq_bits: u32,
    ) {
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
            let (cpu, _stop) =
                phase1_walk_snapshot(stream, tables, start_bit, hard_limit, count_to);
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
        let (s_info, outcome) =
            dispatch_phase1_then_phase2(&b, &stream, &book, 128).expect("dispatch");
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
        let (s_info, outcome) =
            dispatch_jpeg_phase1_then_phase2(&b, &prep, 32).expect("dispatch ok");
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
        let syms16 =
            dispatch_jpeg_phase1_through_phase4(&b, &prep, 16).expect("subseq=16 dispatch");
        let syms32 =
            dispatch_jpeg_phase1_through_phase4(&b, &prep, 32).expect("subseq=32 dispatch");
        let syms64 =
            dispatch_jpeg_phase1_through_phase4(&b, &prep, 64).expect("subseq=64 dispatch");
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
        let gpu_syms = dispatch_jpeg_phase1_through_phase4(&b, &prep, subseq_bits)
            .expect("CUDA phase4 dispatch");
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
        let syms16 =
            dispatch_jpeg_phase1_through_phase4(&b, &prep, 16).expect("subseq=16 dispatch");
        let syms32 =
            dispatch_jpeg_phase1_through_phase4(&b, &prep, 32).expect("subseq=32 dispatch");
        let syms64 =
            dispatch_jpeg_phase1_through_phase4(&b, &prep, 64).expect("subseq=64 dispatch");
        assert_eq!(syms16, syms32, "YCbCr subseq 16 vs 32 symbol mismatch");
        assert_eq!(syms32, syms64, "YCbCr subseq 32 vs 64 symbol mismatch");
    }
}

#[cfg(all(test, feature = "vulkan", feature = "gpu-validation"))]
mod vulkan_tests {
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
    fn cross_backend_parity(
        stream: &PackedBitstream,
        tables: &[CanonicalCodebook],
        subseq_bits: u32,
    ) {
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
            let (cpu, _stop) =
                phase1_walk_snapshot(stream, tables, start_bit, hard_limit, count_to);
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
            let (cpu, _stop) =
                phase1_walk_snapshot(stream, tables, start_bit, hard_limit, count_to);
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
        let (cuda_s, cuda_o) =
            dispatch_phase1_then_phase2(&cuda, &stream, &book, 128).expect("cuda");
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
        let (vk_s, _) =
            dispatch_phase1_then_phase2(&vk, &stream, &book, 128).expect("vulkan phase1+2");
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
        let vk_decoded =
            dispatch_phase1_through_phase4(&vk, &stream, &book, 128).expect("vulkan e2e");

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
        let cuda_states = dispatch_jpeg_phase1_intra_sync(&cuda, &prep, subsequence_bits)
            .expect("CUDA JPEG Phase 1");
        let vk_states = dispatch_jpeg_phase1_intra_sync(&vk, &prep, subsequence_bits)
            .expect("Vulkan JPEG Phase 1");

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
        let gpu_states = dispatch_jpeg_phase1_intra_sync(&vk, &prep, subsequence_bits)
            .expect("Vulkan JPEG Phase 1");

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
        let (s_info, outcome) =
            dispatch_jpeg_phase1_then_phase2(&vk, &prep, 32).expect("dispatch ok");
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
        let (vk_s, vk_o) =
            dispatch_jpeg_phase1_then_phase2(&vk, &prep, 32).expect("vulkan dispatch");
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
}
