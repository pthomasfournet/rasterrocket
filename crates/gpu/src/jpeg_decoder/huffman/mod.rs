//! Host-side dispatch for the parallel-Huffman JPEG decoder kernels.
//!
//! `flatten_codebooks` packs `CanonicalCodebook`s into the device-side
//! `u32[num_components * 65536]` layout the kernels expect.
//! [`decode_jpeg_symbols`] is the production entry point; the
//! synthetic-stream dispatchers in `synthetic.rs` serve the
//! oracle-parity tests only.

// Per-phase parity tests live in the sibling `tests.rs` (CUDA) and
// `vulkan_tests.rs`; the 10-vector adversarial-corpus acceptance tests
// in `corpus.rs`.
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
    JPEG_MEMO_BYTES_PER_SUBSEQUENCE, JPEG_PHASE2_OVER_BUDGET,
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
/// number of kernel workgroups, `G = ceil(n / HUFFMAN_THREADS)`.
///
/// Slot 0 is pinned correct (its fresh start is the true stream
/// start). Each pass, every workgroup resolves its slots in order from
/// its left input (the previous pass's last slot of the workgroup
/// before it), so it leaves holding the exact chain from that input. Workgroup `g` is therefore final after pass `g`, and
/// pass `G - 1` — the `G`-th dispatch — sees every workgroup-boundary
/// slot unchanged, which is the convergence test
/// ([`jpeg_phase2_converged`]). The re-decode is deterministic even
/// for erroring streams, so exhausting the bound is unreachable and
/// the non-convergence error path is defence-in-depth only.
///
/// Single definition shared with the CPU oracle
/// (`phase2_oracle::jpeg_phase2_run_to_sync`) so the bound embedded
/// in compared `Phase2Outcome`s cannot drift between the two.
pub(in crate::jpeg_decoder) const fn jpeg_phase2_retry_bound(num_subsequences: u32) -> u32 {
    num_subsequences.div_ceil(HUFFMAN_THREADS)
}

/// Work limits of one JPEG Phase 2 propagation; exceeding either
/// abandons the GPU decode.
#[derive(Clone, Copy, Debug)]
pub(in crate::jpeg_decoder) struct Phase2Budget {
    /// Dispatches (host round trips).
    pub passes: u32,
    /// Sequential region re-decodes one workgroup may run in one
    /// dispatch ([`HuffmanParams::redecode_budget`]).
    pub redecodes: u32,
}

impl Phase2Budget {
    /// No limit beyond the [`jpeg_phase2_retry_bound`] that always
    /// suffices — the propagation the CPU oracle mirrors.
    #[cfg(all(test, feature = "gpu-validation"))]
    pub(in crate::jpeg_decoder) const UNLIMITED: Self = Self {
        passes: u32::MAX,
        redecodes: u32::MAX,
    };

    /// The limits [`decode_jpeg_symbols`] runs under.
    ///
    /// A dispatch costs up to `HUFFMAN_THREADS` sequential region
    /// re-decodes per workgroup, one per slot whose true left input
    /// matches neither memo entry — content on which fresh-start
    /// walkers do not synchronise with the true decode. Such content is
    /// decoded faster by the sequential CPU walk than by the remaining
    /// re-decodes, so the GPU decode gives up once a workgroup has
    /// re-decoded half its slots, and the decoder falls back. Within
    /// that re-decode budget a propagation converges in one or two
    /// passes; the pass budget bounds the remaining case, a run of
    /// wrong workgroup-boundary inputs, at four dispatches.
    pub(in crate::jpeg_decoder) const PRODUCTION: Self = Self {
        passes: 4,
        redecodes: HUFFMAN_THREADS / 2,
    };
}

/// How a JPEG Phase 2 propagation ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase2Run {
    /// Converged; the 0-based index of the converged pass.
    Converged(u32),
    /// The pass budget ran out before convergence.
    PassBudgetExhausted,
    /// A workgroup exceeded the re-decode budget.
    RedecodeBudgetExhausted,
}

/// Convergence test of one JPEG Phase 2 pass, given `unchanged(i)`:
/// whether slot `i`'s recomputed state equals its input state.
///
/// A pass has converged when the last slot of every workgroup but the
/// final one is unchanged. Workgroup 0's left input is the pinned slot
/// 0, so its chain is exact; workgroup `g`'s left input is the input
/// value of workgroup `g - 1`'s last slot, which equals that slot's
/// exact output when unchanged — so by induction every workgroup
/// propagated from an exact left input and the whole chain is exact.
/// Slots inside a workgroup may change without disturbing this: the
/// in-workgroup resolution makes each workgroup exact relative to its
/// left input in the same pass.
pub(in crate::jpeg_decoder) fn jpeg_phase2_converged(
    num_subsequences: usize,
    mut unchanged: impl FnMut(usize) -> bool,
) -> bool {
    let group = HUFFMAN_THREADS as usize;
    (group..num_subsequences)
        .step_by(group)
        .all(|base| unchanged(base - 1))
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
/// the Phase 1 / Phase 2 re-decode memo, and the Phase 2 fixpoint
/// flags.
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
    memo: DeviceBufferGuard<'a, B>,
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
        // Every entry Phase 2 reads is written by Phase 1 first.
        let memo = DeviceBufferGuard::alloc(
            backend,
            (num_subsequences as usize) * JPEG_MEMO_BYTES_PER_SUBSEQUENCE,
        )?;
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
            memo,
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
            memo: Some(self.memo.as_ref()),
            length_bits: self.length_bits,
            subsequence_bits: self.subsequence_bits,
            num_components: self.num_components,
            total_symbols: 0,
            blocks_per_mcu: self.blocks_per_mcu,
            redecode_budget: u32::MAX,
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
        backend.free_device(self.memo.take());
        backend.free_device(self.sync_flags.take());
    }
}

/// Run JPEG Phase 1, then drive the Phase 2 propagation to
/// convergence.
///
/// Each pass is one `JpegPhase2InterSync` dispatch reading the
/// previous pass's states (`setup.s_info`) and writing the recomputed
/// ones (`setup.s_info_scratch`), followed by a buffer swap, so
/// `setup.s_info` always holds the last-written states. Propagation
/// within a workgroup happens on-device inside the dispatch; the host
/// loop only carries truth across workgroup boundaries, and stops at
/// the first pass [`jpeg_phase2_converged`] accepts. Phase 1 is
/// recorded in the first pass's page: the recorder orders consecutive
/// dispatches of one page, so the only host wait per pass is the one
/// the convergence check needs.
///
/// Runs at most `min(budget.passes, jpeg_phase2_retry_bound(n))`
/// dispatches, each under `budget.redecodes`.
///
/// One loop shared by the oracle-parity dispatcher and the production
/// Phase 1–4 dispatcher, so the code the parity tests exercise is the
/// code production runs.
fn run_jpeg_phase1_and_phase2<B: GpuBackend>(
    backend: &B,
    setup: &mut JpegDispatchSetup<'_, B>,
    budget: Phase2Budget,
) -> Result<Phase2Run> {
    let passes = jpeg_phase2_retry_bound(setup.num_subsequences).min(budget.passes);
    let mut flags_host = vec![0u32; setup.num_subsequences as usize];
    for iter in 0..passes {
        backend.begin_page()?;
        if iter == 0 {
            backend.record_huffman(setup.base_params(HuffmanPhase::JpegPhase1IntraSync))?;
        }
        let mut params = setup.base_params(HuffmanPhase::JpegPhase2InterSync);
        params.s_info = setup.s_info_scratch.as_ref();
        params.s_info_prev = Some(setup.s_info.as_ref());
        params.sync_flags = Some(setup.sync_flags.as_ref());
        params.redecode_budget = budget.redecodes;
        backend.record_huffman(params)?;
        let fence = backend.submit_page()?;
        backend.wait_page(fence)?;

        let dst = bytemuck::cast_slice_mut::<u32, u8>(&mut flags_host);
        let handle = backend.download_async(setup.sync_flags.as_ref(), dst)?;
        backend.wait_download(handle)?;

        std::mem::swap(&mut setup.s_info, &mut setup.s_info_scratch);
        if flags_host.contains(&JPEG_PHASE2_OVER_BUDGET) {
            return Ok(Phase2Run::RedecodeBudgetExhausted);
        }
        if jpeg_phase2_converged(flags_host.len(), |i| flags_host[i] == 1) {
            return Ok(Phase2Run::Converged(iter));
        }
    }
    Ok(Phase2Run::PassBudgetExhausted)
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

/// One-shot JPEG Phase 1 dispatch returning the re-decode memo: per
/// subsequence, `[phase1 key, phase1 value, last key, last value]`.
///
/// # Errors
/// Returns `BackendError` for any alloc / upload / dispatch / download
/// failure.
#[cfg(all(test, feature = "gpu-validation"))]
fn dispatch_jpeg_phase1_memo<B: GpuBackend>(
    backend: &B,
    prep: &crate::jpeg_decoder::JpegPreparedInput,
    subsequence_bits: u32,
) -> Result<Vec<[SubsequenceState; 4]>> {
    let setup = JpegDispatchSetup::new(backend, prep, subsequence_bits)?;
    setup.run_phase1(backend)?;
    let mut out = vec![
        [<SubsequenceState as bytemuck::Zeroable>::zeroed(); 4];
        setup.num_subsequences as usize
    ];
    let dst = bytemuck::cast_slice_mut::<[SubsequenceState; 4], u8>(&mut out);
    let handle = backend.download_async(setup.memo.as_ref(), dst)?;
    backend.wait_download(handle)?;
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
    let outcome = match run_jpeg_phase1_and_phase2(backend, &mut setup, Phase2Budget::UNLIMITED)? {
        Phase2Run::Converged(iterations) => Phase2Outcome::Converged { iterations },
        Phase2Run::PassBudgetExhausted => Phase2Outcome::SyncBoundExceeded {
            bound: jpeg_phase2_retry_bound(setup.num_subsequences),
        },
        Phase2Run::RedecodeBudgetExhausted => {
            return Err(BackendError::msg(
                "dispatch_jpeg_phase1_then_phase2: an unlimited re-decode budget ran out",
            ));
        }
    };
    let s_info_out = setup.download_s_info(backend)?;
    setup.free_all(backend);
    Ok((s_info_out, outcome))
}

/// Result of one parallel-Huffman decode of a JPEG entropy stream.
#[derive(Debug)]
pub(crate) struct JpegSymbolDecode {
    /// Decoded Huffman symbol stream, one `u32` per symbol.
    pub symbols: Vec<u32>,
    /// Phase 2 dispatches (host round trips) the propagation took.
    pub phase2_passes: u32,
}

/// Test wrapper over [`decode_jpeg_symbols_within`] returning the
/// symbols of an unbudgeted decode — the propagation the oracle
/// mirrors, independent of the production give-up policy.
///
/// # Errors
/// As [`decode_jpeg_symbols`], minus the budget exhaustion.
#[cfg(all(test, feature = "gpu-validation"))]
fn dispatch_jpeg_phase1_through_phase4<B: GpuBackend>(
    backend: &B,
    prep: &crate::jpeg_decoder::JpegPreparedInput,
    subsequence_bits: u32,
) -> Result<Vec<u32>> {
    decode_jpeg_symbols_within(backend, prep, subsequence_bits, Phase2Budget::UNLIMITED)
        .map(|d| d.symbols)
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
/// `W = HUFFMAN_THREADS` lanes per workgroup, Phase 1 is one dispatch
/// in which every thread walks two regions, and Phase 4 one dispatch
/// in which every thread re-walks one. Phase 2 resolves each workgroup
/// in `W` barrier-separated steps; a step costs a memo comparison, or
/// one region re-decode on one thread when the slot's true left input
/// matches neither memo entry — a "miss": the predecessor's
/// fresh-start Phase 1 walk had not synchronised with the true decode
/// by its region end. A dispatch therefore costs the largest
/// per-workgroup miss count times one region re-decode, plus `W`
/// cheap steps. Dispatches are host round trips; a pass converges once
/// every workgroup-boundary slot was right on entry, so their count is
/// one plus the longest run of consecutive workgroup boundaries whose
/// left input was wrong, and repeat passes resolve unchanged
/// workgroups entirely from the last-exact memo entries.
///
/// Misses are content-dependent and not limited to multi-component
/// streams: over flat regions with short optimised codes, and on
/// dense high-quality streams, fresh-start walkers frequently settle
/// into a different parse (bit offset, DC/AC role, or block phase) and
/// never re-synchronise. [`Phase2Budget::PRODUCTION`] bounds both
/// factors and abandons the GPU decode on content where they explode.
///
/// # Errors
/// Returns `BackendError` if any allocation, upload, kernel dispatch,
/// or download fails; if Phase 2 exhausts [`Phase2Budget::PRODUCTION`];
/// if the converged chain stalled mid-stream on a decode error
/// (see [`jpeg_stream_stalled`]); or if Phase 4 reports a per-subseq
/// decode failure.
pub(crate) fn decode_jpeg_symbols<B: GpuBackend>(
    backend: &B,
    prep: &crate::jpeg_decoder::JpegPreparedInput,
    subsequence_bits: u32,
) -> Result<JpegSymbolDecode> {
    decode_jpeg_symbols_within(backend, prep, subsequence_bits, Phase2Budget::PRODUCTION)
}

/// [`decode_jpeg_symbols`] under an explicit Phase 2 budget.
pub(in crate::jpeg_decoder) fn decode_jpeg_symbols_within<B: GpuBackend>(
    backend: &B,
    prep: &crate::jpeg_decoder::JpegPreparedInput,
    subsequence_bits: u32,
    budget: Phase2Budget,
) -> Result<JpegSymbolDecode> {
    if prep.bitstream.length_bits == 0 {
        return Ok(JpegSymbolDecode {
            symbols: Vec::new(),
            phase2_passes: 0,
        });
    }
    let mut setup = JpegDispatchSetup::new(backend, prep, subsequence_bits)?;
    let converged_pass = match run_jpeg_phase1_and_phase2(backend, &mut setup, budget)? {
        Phase2Run::Converged(pass) => pass,
        Phase2Run::PassBudgetExhausted => {
            return Err(BackendError::msg(format!(
                "dispatch_jpeg_phase1_through_phase4: Phase 2 did not converge within \
                 {} pass(es)",
                jpeg_phase2_retry_bound(setup.num_subsequences).min(budget.passes),
            )));
        }
        Phase2Run::RedecodeBudgetExhausted => {
            return Err(BackendError::msg(format!(
                "dispatch_jpeg_phase1_through_phase4: Phase 2 did not converge within \
                 {} re-decodes per workgroup",
                budget.redecodes,
            )));
        }
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
mod tests;

#[cfg(all(test, feature = "vulkan", feature = "gpu-validation"))]
mod vulkan_tests;
