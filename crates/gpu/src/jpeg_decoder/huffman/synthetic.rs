//! One-shot dispatchers for the synthetic-stream kernels
//! (`phase1_intra_sync` / `phase2_inter_sync` / `phase4_redecode`).
//!
//! The synthetic state machine (`z += 1` per symbol, `c` rolling over
//! on the 64-symbol boundary) is exercised only by the oracle-parity
//! and adversarial-corpus tests; every dispatcher here owns its device
//! buffers for the duration of one call.

use crate::backend::params::{HuffmanParams, HuffmanPhase};
use crate::backend::{BackendError, GpuBackend, Result};
use crate::jpeg::CanonicalCodebook;
use crate::jpeg_decoder::PackedBitstream;
use crate::jpeg_decoder::dispatch_util::DeviceBufferGuard;
use crate::jpeg_decoder::phase1_oracle::SubsequenceState;
use crate::jpeg_decoder::phase2_oracle::{Phase2Outcome, retry_bound};

use super::build_gpu_codebook;

/// Device buffers and derived dimensions shared by the synthetic
/// dispatchers: bitstream, flat codebook, and the ping-ponged pair of
/// per-subsequence state buffers. `s_info` always names the most
/// recently written state buffer — [`run_phase1_and_phase2`] swaps the
/// pair after each pass.
struct SyntheticDispatchSetup<'a, B: GpuBackend> {
    bitstream: DeviceBufferGuard<'a, B>,
    codebook: DeviceBufferGuard<'a, B>,
    s_info: DeviceBufferGuard<'a, B>,
    s_info_scratch: DeviceBufferGuard<'a, B>,
    length_bits: u32,
    subsequence_bits: u32,
    num_subsequences: u32,
    num_components: u32,
}

impl<'a, B: GpuBackend> SyntheticDispatchSetup<'a, B> {
    /// Validate the arguments, then allocate and upload the shared
    /// buffers. `who` names the calling dispatcher in error messages.
    ///
    /// # Errors
    /// Returns `BackendError` for zero `subsequence_bits`, an empty or
    /// oversized `codebooks` slice, or any allocation/upload failure.
    fn new(
        backend: &'a B,
        who: &str,
        stream: &PackedBitstream,
        codebooks: &[CanonicalCodebook],
        subsequence_bits: u32,
    ) -> Result<Self> {
        if subsequence_bits == 0 {
            return Err(BackendError::msg(format!(
                "{who}: subsequence_bits must be > 0"
            )));
        }
        if codebooks.is_empty() {
            return Err(BackendError::msg(format!(
                "{who}: at least one codebook required"
            )));
        }
        let num_components = u32::try_from(codebooks.len()).map_err(|_| {
            BackendError::msg(format!(
                "more codebooks ({}) than fit in u32",
                codebooks.len()
            ))
        })?;
        let length_bits = stream.length_bits;
        let num_subsequences = length_bits.div_ceil(subsequence_bits);

        // peek16 reads `word_idx + 1` even at the stream tail, so the
        // bitstream buffer needs one zero word of headroom; the zeroed
        // allocation supplies it and only the stream's words are
        // uploaded.
        let bitstream_bytes = ((length_bits.div_ceil(32) as usize) + 1) * 4;
        let codebook_flat = build_gpu_codebook(codebooks);
        let s_info_bytes = (num_subsequences as usize) * std::mem::size_of::<SubsequenceState>();

        let bitstream = DeviceBufferGuard::alloc_zeroed(backend, bitstream_bytes)?;
        let codebook = DeviceBufferGuard::alloc(backend, codebook_flat.len() * 4)?;
        let s_info = DeviceBufferGuard::alloc(backend, s_info_bytes)?;
        let s_info_scratch = DeviceBufferGuard::alloc(backend, s_info_bytes)?;

        // Uploads are ordered before the first recorded kernel by the
        // backend (same CUDA stream; Vulkan cross-queue barrier at
        // submit), so the fences are not awaited.
        let _up1 = backend.upload_async(bitstream.as_ref(), bytemuck::cast_slice(&stream.words))?;
        let _up2 = backend.upload_async(codebook.as_ref(), bytemuck::cast_slice(&codebook_flat))?;

        Ok(Self {
            bitstream,
            codebook,
            s_info,
            s_info_scratch,
            length_bits,
            subsequence_bits,
            num_subsequences,
            num_components,
        })
    }

    /// `HuffmanParams` skeleton with the shared buffers and dims.
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
            dc_codebook: None,
            mcu_schedule: None,
            memo: None,
            length_bits: self.length_bits,
            subsequence_bits: self.subsequence_bits,
            num_components: self.num_components,
            total_symbols: 0,
            blocks_per_mcu: 0,
            phase,
        }
    }

    /// Record and run Phase 1 into `s_info`.
    fn run_phase1(&self, backend: &B) -> Result<()> {
        backend.begin_page()?;
        backend.record_huffman(self.base_params(HuffmanPhase::Phase1IntraSync))?;
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
}

/// Run Phase 1, then drive the synthetic Phase 2 sync loop: each pass
/// reads the previous pass's states (`setup.s_info`), writes the
/// carried-over or advanced ones (`setup.s_info_scratch`), and swaps
/// the pair, so `setup.s_info` always holds the last-written states.
/// Phase 1 is recorded in the first pass's page.
///
/// Returns the converged pass index, or `SyncBoundExceeded` after
/// `retry_bound(n) + 1` passes without every flag reading 1.
fn run_phase1_and_phase2<B: GpuBackend>(
    backend: &B,
    setup: &mut SyntheticDispatchSetup<'_, B>,
    sync_flags: &DeviceBufferGuard<'_, B>,
) -> Result<Phase2Outcome> {
    let bound = retry_bound(setup.num_subsequences as usize);
    let mut flags_host = vec![0u32; setup.num_subsequences as usize];
    for iter in 0..=bound {
        backend.begin_page()?;
        if iter == 0 {
            backend.record_huffman(setup.base_params(HuffmanPhase::Phase1IntraSync))?;
        }
        let mut params = setup.base_params(HuffmanPhase::Phase2InterSync);
        params.s_info = setup.s_info_scratch.as_ref();
        params.s_info_prev = Some(setup.s_info.as_ref());
        params.sync_flags = Some(sync_flags.as_ref());
        backend.record_huffman(params)?;
        let fence = backend.submit_page()?;
        backend.wait_page(fence)?;

        let dst = bytemuck::cast_slice_mut::<u32, u8>(&mut flags_host);
        let handle = backend.download_async(sync_flags.as_ref(), dst)?;
        backend.wait_download(handle)?;

        std::mem::swap(&mut setup.s_info, &mut setup.s_info_scratch);
        if flags_host.iter().all(|&f| f == 1) {
            return Ok(Phase2Outcome::Converged { iterations: iter });
        }
    }
    Ok(Phase2Outcome::SyncBoundExceeded { bound })
}

/// One-shot dispatch of the Phase 1 intra-sync kernel.
///
/// Empty bitstream returns `Ok(Vec::new())` as a no-op; other invalid
/// inputs (zero `subsequence_bits`, no codebooks) return a typed
/// `BackendError` rather than panicking.
///
/// # Errors
/// Returns `BackendError` for invalid arguments, or the underlying
/// driver error if any allocation, upload, dispatch, or download
/// fails.
pub(super) fn dispatch_phase1_intra_sync<B: GpuBackend>(
    backend: &B,
    stream: &PackedBitstream,
    codebooks: &[CanonicalCodebook],
    subsequence_bits: u32,
) -> Result<Vec<SubsequenceState>> {
    if stream.length_bits == 0 {
        return Ok(Vec::new());
    }
    let setup = SyntheticDispatchSetup::new(
        backend,
        "dispatch_phase1_intra_sync",
        stream,
        codebooks,
        subsequence_bits,
    )?;
    setup.run_phase1(backend)?;
    setup.download_s_info(backend)
}

/// One-shot dispatch of Phase 1 + Phase 2 (bounded retry sync loop).
///
/// Returns the final `s_info` either way; the outcome tells the caller
/// whether convergence happened.
///
/// # Errors
/// Returns `BackendError` if any alloc/upload/dispatch/download
/// fails. Bound-exceeded is reported via the outcome, not as an
/// error.
pub(super) fn dispatch_phase1_then_phase2<B: GpuBackend>(
    backend: &B,
    stream: &PackedBitstream,
    codebooks: &[CanonicalCodebook],
    subsequence_bits: u32,
) -> Result<(Vec<SubsequenceState>, Phase2Outcome)> {
    if stream.length_bits == 0 {
        return Ok((Vec::new(), Phase2Outcome::Converged { iterations: 0 }));
    }
    let mut setup = SyntheticDispatchSetup::new(
        backend,
        "dispatch_phase1_then_phase2",
        stream,
        codebooks,
        subsequence_bits,
    )?;
    let sync_flags = DeviceBufferGuard::alloc(backend, (setup.num_subsequences as usize) * 4)?;
    let outcome = run_phase1_and_phase2(backend, &mut setup, &sync_flags)?;
    Ok((setup.download_s_info(backend)?, outcome))
}

/// One-shot dispatch of all four phases: 1 → 2 → 3 → 4.
///
/// Returns the final decoded symbol stream (u32 per symbol; the kernel
/// writes a u8 value but the buffer is u32-strided to match the
/// kernel signature). Phase 3's exclusive scan runs host-side — one
/// u32 per subsequence.
///
/// # Errors
/// Returns `BackendError` if any alloc/upload/dispatch/download
/// fails, if Phase 2 fails to converge within the retry bound, if the
/// total symbol count overflows `u32`, or if Phase 4 reports a
/// per-subsequence decode failure.
pub(super) fn dispatch_phase1_through_phase4<B: GpuBackend>(
    backend: &B,
    stream: &PackedBitstream,
    codebooks: &[CanonicalCodebook],
    subsequence_bits: u32,
) -> Result<Vec<u32>> {
    if stream.length_bits == 0 {
        return Ok(Vec::new());
    }
    let mut setup = SyntheticDispatchSetup::new(
        backend,
        "dispatch_phase1_through_phase4",
        stream,
        codebooks,
        subsequence_bits,
    )?;
    let per_subseq_bytes = (setup.num_subsequences as usize) * 4;
    let sync_flags = DeviceBufferGuard::alloc(backend, per_subseq_bytes)?;
    if !matches!(
        run_phase1_and_phase2(backend, &mut setup, &sync_flags)?,
        Phase2Outcome::Converged { .. }
    ) {
        return Err(BackendError::msg(
            "dispatch_phase1_through_phase4: Phase 2 did not converge within retry bound",
        ));
    }

    // Exclusive scan + total over s_info[*].n; the checked total
    // surfaces u32 overflow (silent wrapping would let Phase 4 write
    // past the allocated symbols_out).
    let s_info_host = setup.download_s_info(backend)?;
    let counts: Vec<u32> = s_info_host.iter().map(|s| s.n).collect();
    let (offsets_host, total_symbols) = crate::jpeg_decoder::scan::exclusive_scan_checked(&counts)
        .ok_or_else(|| {
            BackendError::msg("dispatch_phase1_through_phase4: total symbol count overflows u32")
        })?;
    if total_symbols == 0 {
        // Phase 4 kernel + zero-size alloc would both reject this.
        return Ok(Vec::new());
    }

    let offsets = DeviceBufferGuard::alloc(backend, per_subseq_bytes)?;
    let _up = backend.upload_async(offsets.as_ref(), bytemuck::cast_slice(&offsets_host))?;
    let symbols = DeviceBufferGuard::alloc_zeroed(backend, (total_symbols as usize) * 4)?;
    // Zero-init = Phase4FailureKind::Ok, so a subsequence the kernel
    // does not reach reads back clean.
    let decode_status = DeviceBufferGuard::alloc_zeroed(backend, per_subseq_bytes)?;

    backend.begin_page()?;
    let mut params = setup.base_params(HuffmanPhase::Phase4Redecode);
    params.offsets = Some(offsets.as_ref());
    params.symbols_out = Some(symbols.as_ref());
    params.decode_status = Some(decode_status.as_ref());
    params.total_symbols = total_symbols;
    backend.record_huffman(params)?;
    let fence = backend.submit_page()?;
    backend.wait_page(fence)?;

    super::check_phase4_status(
        backend,
        &decode_status,
        setup.num_subsequences,
        "dispatch_phase1_through_phase4",
    )?;
    super::download_u32s(backend, &symbols, total_symbols as usize)
}

/// Phase 3: exclusive prefix-sum of per-subsequence symbol counts on
/// the device.
///
/// `offsets[i]` is the base index into the final symbol stream where
/// Phase 4 writes subsequence `i`'s symbols. Empty input returns
/// `Ok(Vec::new())`.
///
/// # Errors
/// Returns the underlying `BackendError` from
/// `dispatch_blelloch_scan` if any allocation, upload, dispatch, or
/// download fails.
pub(super) fn dispatch_phase3_offsets<B: GpuBackend>(
    backend: &B,
    s_info: &[SubsequenceState],
) -> Result<Vec<u32>> {
    let counts: Vec<u32> = s_info.iter().map(|s| s.n).collect();
    crate::jpeg_decoder::scan::dispatch_blelloch_scan(backend, &counts)
}
