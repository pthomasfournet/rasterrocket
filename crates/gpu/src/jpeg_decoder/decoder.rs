//! `JpegGpuDecoder` — end-to-end JPEG → RGBA8 device image.
//!
//! CPU pre-pass (`prepare_jpeg`) + GPU Phases 1–4 (parallel Huffman) +
//! CPU magnitude-bit extraction (`symbols_to_coefficients`) +
//! GPU Phase 5 (IDCT + dequant + colour conversion).
//!
//! The GPU Huffman path (feature `gpu-jpeg-huffman`) replaces the
//! sequential CPU Huffman walk with a parallel GPU pass; magnitude bits
//! are still extracted on the CPU in a single sequential pass, but the
//! CPU does no Huffman table lookups — only bit reads guided by the
//! symbol stream the GPU produced.

use crate::backend::params::IdctParams;
use crate::backend::{BackendError as BackendErr, GpuBackend};
use crate::jpeg::bitreader::BitReader;
use crate::jpeg::headers::{JpegFrameComponent, mcu_count};
use crate::jpeg_decoder::cpu_prepass::{JpegPreparedInput, prepare_jpeg};
use crate::jpeg_decoder::device_image::DeviceImage;
use crate::jpeg_decoder::dispatch_util::DeviceBufferGuard;
use crate::jpeg_decoder::error::JpegGpuError;

/// End-to-end JPEG decoder that runs IDCT on the GPU.
///
/// Phases 1–4 (parallel Huffman decode) are available via
/// `dispatch_jpeg_phase1_through_phase4`; this struct uses a CPU
/// coefficient extraction pass and dispatches only Phase 5 to the GPU.
pub struct JpegGpuDecoder<B: GpuBackend> {
    backend: B,
}

impl<B: GpuBackend> JpegGpuDecoder<B> {
    /// Create a new decoder bound to `backend`.
    pub const fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Borrow the underlying backend.
    pub const fn backend(&self) -> &B {
        &self.backend
    }

    /// Decode a baseline JFIF JPEG into a device-resident RGBA8 image.
    ///
    /// # Errors
    /// Returns `JpegGpuError` if the input is not a supported baseline JPEG,
    /// if coefficient extraction fails, or if any GPU operation fails.
    #[expect(
        clippy::too_many_lines,
        reason = "GPU Huffman dispatch + CPU fallback + IDCT launch; splitting obscures the linear flow"
    )]
    pub fn decode(&self, jpeg_bytes: &[u8]) -> std::result::Result<DeviceImage<B>, JpegGpuError> {
        let prep = prepare_jpeg(jpeg_bytes)?;

        // Validate component count before any allocation.
        let nc = prep.components.len();
        if nc != 1 && nc != 3 {
            return Err(JpegGpuError::UnsupportedComponents(
                u8::try_from(nc).unwrap_or(u8::MAX),
            ));
        }

        // Reject subsampled input: the IDCT kernel assumes 4:4:4 (all components
        // share the same 8×8 block grid). Non-unity sampling factors produce the
        // wrong chroma block layout and corrupt output silently.
        for comp in &prep.components {
            if comp.h_sampling != 1 || comp.v_sampling != 1 {
                return Err(JpegGpuError::UnsupportedSubsampling {
                    component: comp.id,
                    h: comp.h_sampling,
                    v: comp.v_sampling,
                });
            }
        }

        // For 3-component images, validate that quantisation tables are
        // assigned in JFIF slot order (slot 0 = luma, slot 1 = chroma):
        // the kernel hardcodes qt_sel=0 for the Y component and qt_sel=1
        // for Cb/Cr, so a reversed or non-standard assignment would
        // silently use the wrong tables. A grayscale component may
        // reference any defined slot — pack_qtables resolves it to packed
        // index 0.
        if nc == 3 {
            for (ci, comp) in prep.components.iter().enumerate() {
                let expected_slot = u8::from(ci != 0);
                if comp.quant_selector != expected_slot {
                    return Err(JpegGpuError::HeaderParse(format!(
                        "component {ci} uses quantisation table slot {} but kernel expects slot \
                         {expected_slot}; only JFIF-standard slot assignment (0=luma, 1=chroma) \
                         is supported",
                        comp.quant_selector
                    )));
                }
            }
        }

        // Attempt the GPU Phases 1–4 parallel-Huffman path.  On any error
        // (Phase 2 non-convergence, backend dispatch failure, or symbol-stream
        // inconsistency) fall through to the CPU sequential path below.
        let gpu_coefs: Option<(Vec<i32>, Vec<i32>)> = {
            #[cfg(feature = "gpu-jpeg-huffman")]
            {
                let subseq_bits = crate::jpeg_decoder::pick_subsequence_size(&prep);
                crate::jpeg_decoder::huffman::dispatch_jpeg_phase1_through_phase4(
                    &self.backend,
                    &prep,
                    subseq_bits,
                )
                .ok()
                .and_then(|symbols| symbols_to_coefficients(&prep, &symbols).ok())
            }
            #[cfg(not(feature = "gpu-jpeg-huffman"))]
            {
                None
            }
        };

        let (coef_flat, dc_flat, qt_flat, num_qtables) = if let Some((coef, dc)) = gpu_coefs {
            // GPU Huffman path succeeded; pack the referenced tables from prep.
            let (qt_flat, num_qtables) = pack_qtables(&prep).map_err(JpegGpuError::HeaderParse)?;
            (coef, dc, qt_flat, num_qtables)
        } else {
            extract_coefficients(&prep).map_err(JpegGpuError::HeaderParse)?
        };

        let width = u32::from(prep.width);
        let height = u32::from(prep.height);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "nc is 1 or 3 (checked above); trivially fits in u32"
        )]
        let num_components = nc as u32;
        let blocks_wide = width.div_ceil(8);
        let blocks_high = height.div_ceil(8);

        // block_idx = comp * BW * BH + by * BW + bx; coef_base = block_idx * 64.
        // Maximum block_idx for a 3-component JPEG: 2 * BW * BH + (BH-1)*BW + (BW-1).
        // coef_base overflows u32 when block_idx >= 2^26 = 67_108_864.
        // BW * BH <= 2^26 / 3 ≈ 22.4 M corresponds to ~4730 blocks per side = ~37 840 px.
        // Reject here so the kernel never sees an overflowing index.
        let max_block_idx = u64::from(num_components)
            .saturating_mul(u64::from(blocks_wide))
            .saturating_mul(u64::from(blocks_high));
        if max_block_idx >= (1u64 << 26) {
            return Err(JpegGpuError::Dispatch(format!(
                "image too large for IDCT kernel: {num_components} components × \
                 {blocks_wide}×{blocks_high} blocks exceeds the 26-bit block index limit"
            )));
        }

        self.dispatch_idct(
            &coef_flat,
            &qt_flat,
            &dc_flat,
            width,
            height,
            num_components,
            blocks_wide,
            blocks_high,
            num_qtables,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "10-scalar IDCT params; grouping them is IdctParams which requires a backend buffer ref"
    )]
    fn dispatch_idct(
        &self,
        coefficients: &[i32],
        qtables: &[i32],
        dc_values: &[i32],
        width: u32,
        height: u32,
        num_components: u32,
        blocks_wide: u32,
        blocks_high: u32,
        num_qtables: u32,
    ) -> std::result::Result<DeviceImage<B>, JpegGpuError> {
        let be = |e: BackendErr| JpegGpuError::Dispatch(e.to_string());

        let coef_bytes = std::mem::size_of_val(coefficients);
        let qt_bytes = std::mem::size_of_val(qtables);
        let dc_bytes = std::mem::size_of_val(dc_values);
        let px_bytes = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| {
                JpegGpuError::Dispatch("image too large: width × height × 4 overflows".into())
            })?;

        let coef_buf = DeviceBufferGuard::alloc(&self.backend, coef_bytes).map_err(be)?;
        let qt_buf = DeviceBufferGuard::alloc(&self.backend, qt_bytes).map_err(be)?;
        let dc_buf = DeviceBufferGuard::alloc(&self.backend, dc_bytes).map_err(be)?;
        let px_buf = DeviceBufferGuard::alloc_zeroed(&self.backend, px_bytes).map_err(be)?;

        // Fences from upload_async are intentionally dropped: on CUDA, uploads
        // enqueue on the same stream as the subsequent kernel launch, so stream
        // ordering guarantees the data is visible before the kernel reads it.
        // On Vulkan, the per-page recorder inserts the required cross-queue
        // barriers between the transfer and compute queues.
        let _u1 = self
            .backend
            .upload_async(coef_buf.as_ref(), bytemuck::cast_slice(coefficients))
            .map_err(be)?;
        let _u2 = self
            .backend
            .upload_async(qt_buf.as_ref(), bytemuck::cast_slice(qtables))
            .map_err(be)?;
        let _u3 = self
            .backend
            .upload_async(dc_buf.as_ref(), bytemuck::cast_slice(dc_values))
            .map_err(be)?;

        self.backend.begin_page().map_err(be)?;
        self.backend
            .record_idct(IdctParams {
                coefficients: coef_buf.as_ref(),
                qtables: qt_buf.as_ref(),
                dc_values: dc_buf.as_ref(),
                pixels_rgba: px_buf.as_ref(),
                width,
                height,
                num_components,
                blocks_wide,
                blocks_high,
                num_qtables,
            })
            .map_err(be)?;
        let fence = self.backend.submit_page().map_err(be)?;
        self.backend.wait_page(fence).map_err(be)?;

        self.backend.free_device(coef_buf.take());
        self.backend.free_device(qt_buf.take());
        self.backend.free_device(dc_buf.take());

        Ok(DeviceImage {
            buffer: px_buf.take(),
            width,
            height,
        })
    }
}

/// Reconstruct DCT coefficient arrays from `prep`.
///
/// Returns `(coefficients, dc_values, qtables, num_qtables)`:
/// - `coefficients`: `num_components × blocks_per_comp × 64` i32 in zigzag order
/// - `dc_values`: `num_components × blocks_per_comp` i32 (absolute DC from pre-pass)
/// - `qtables`: `num_qtables × 64` i32 in zigzag order (verbatim DQT bytes)
/// - `num_qtables`: count of populated quantisation table slots
#[expect(
    clippy::type_complexity,
    reason = "4-tuple private fn return; a named struct is overkill here"
)]
#[expect(
    clippy::too_many_lines,
    reason = "AC walk + DC copy + QT copy; split would obscure the single-pass structure"
)]
fn extract_coefficients(
    prep: &JpegPreparedInput,
) -> std::result::Result<(Vec<i32>, Vec<i32>, Vec<i32>, u32), String> {
    let num_comp = prep.components.len();
    let blocks_wide = usize::from(prep.width.div_ceil(8));
    let blocks_high = usize::from(prep.height.div_ceil(8));
    let blocks_per_comp = blocks_wide * blocks_high;

    let mut coef_flat = vec![0i32; num_comp * blocks_per_comp * 64];

    // Recover raw bitstream bytes for BitReader:
    // PackedBitstream words use u32::from_be_bytes packing, so each word
    // unpacks to bytes via to_be_bytes().
    let raw_bytes: Vec<u8> = prep
        .bitstream
        .words
        .iter()
        .flat_map(|w| w.to_be_bytes())
        .collect();
    let mut bits = BitReader::new(&raw_bytes);

    let totalmcus = mcu_count(prep.width, prep.height, &prep.components);
    let mut block_counts = vec![0usize; num_comp];

    for mcu in 0..totalmcus {
        for (ci, comp) in prep.components.iter().enumerate() {
            let dc_sel = usize::from(
                *prep
                    .dc_selectors
                    .get(ci)
                    .ok_or_else(|| format!("no dc_selector for comp {ci}"))?,
            );
            let ac_sel = usize::from(
                *prep
                    .ac_selectors
                    .get(ci)
                    .ok_or_else(|| format!("no ac_selector for comp {ci}"))?,
            );
            let dc_cb = prep.dc_codebooks[dc_sel]
                .as_ref()
                .ok_or_else(|| format!("missing DC codebook {dc_sel}"))?;
            let ac_cb = prep.ac_codebooks[ac_sel]
                .as_ref()
                .ok_or_else(|| format!("missing AC codebook {ac_sel}"))?;

            let bpm = blocks_permcu_count(*comp);
            for _b in 0..bpm {
                let block_idx = block_counts[ci];
                let coef_base = (ci * blocks_per_comp + block_idx) * 64;
                block_counts[ci] += 1;

                // DC: peek + consume codeword, then read `category` magnitude bits.
                let peek = bits.peek_u16().ok_or_else(|| {
                    format!("bitstream empty at DC codeword (mcu={mcu} comp={ci})")
                })?;
                let dc_entry = dc_cb.lookup(peek);
                bits.consume(usize::from(dc_entry.num_bits));
                let category = dc_entry.symbol;
                if category > 0 {
                    let _ = bits.read_bits(usize::from(category)).ok_or_else(|| {
                        format!("DC magnitude truncated (mcu={mcu} comp={ci} cat={category})")
                    })?;
                }

                // AC: read each symbol + magnitude bits.
                let mut zz = 1usize;
                while zz < 64 {
                    let peek = bits.peek_u16().ok_or_else(|| {
                        format!("bitstream empty at AC (mcu={mcu} comp={ci} zz={zz})")
                    })?;
                    let ac_entry = ac_cb.lookup(peek);
                    bits.consume(usize::from(ac_entry.num_bits));
                    let sym_byte = ac_entry.symbol;

                    if sym_byte == 0x00 {
                        break; // EOB
                    }
                    if sym_byte == 0xF0 {
                        zz += 16; // ZRL
                        continue;
                    }
                    let run = (sym_byte >> 4) as usize;
                    let size = sym_byte & 0x0F;
                    zz += run;
                    if zz >= 64 {
                        break;
                    }
                    let ac_val = if size == 0 {
                        0i32
                    } else {
                        let raw = bits.read_bits(usize::from(size)).ok_or_else(|| {
                            format!("AC magnitude truncated (mcu={mcu} comp={ci} zz={zz})")
                        })?;
                        jpeg_extend(raw.cast_signed(), size)
                    };
                    if coef_base + zz >= coef_flat.len() {
                        return Err(format!(
                            "coefficient overflow: coef_base={coef_base} zz={zz} \
                             coef_flat.len()={} (mcu={mcu} comp={ci})",
                            coef_flat.len()
                        ));
                    }
                    coef_flat[coef_base + zz] = ac_val;
                    zz += 1;
                }
            }
        }
    }

    // DC values: pre-resolved absolute DC chain from the pre-pass.
    let mut dc_flat = vec![0i32; num_comp * blocks_per_comp];
    for (ci, dc_vec) in prep
        .dc_values
        .per_component
        .iter()
        .take(num_comp)
        .enumerate()
    {
        let base = ci * blocks_per_comp;
        for (bi, &dc) in dc_vec.iter().enumerate() {
            let idx = base + bi;
            if idx >= dc_flat.len() {
                return Err(format!(
                    "dc_values overflow: comp={ci} block={bi} idx={idx} \
                     dc_flat.len()={}",
                    dc_flat.len()
                ));
            }
            dc_flat[idx] = dc;
        }
    }

    let (qt_flat, num_qtables) = pack_qtables(prep)?;

    Ok((coef_flat, dc_flat, qt_flat, num_qtables))
}

/// Pack the quantisation tables the frame components actually reference,
/// in the order the kernel selects them: index 0 is component 0's table
/// (`qt_sel = 0`), index 1 the chroma table for 3-component images
/// (`qt_sel = 1`).
///
/// Packing the *referenced* slots — rather than every populated slot in
/// slot order — keeps a grayscale image correct when its component
/// references slot 1 while an unused slot 0 is also defined. For
/// 3-component images `decode` has already pinned the JFIF slot
/// assignment (0 = luma, 1 = chroma).
fn pack_qtables(prep: &JpegPreparedInput) -> std::result::Result<(Vec<i32>, u32), String> {
    let slots: &[usize] = if prep.components.len() == 1 {
        &[usize::from(prep.components[0].quant_selector)]
    } else {
        &[0, 1]
    };

    let mut qt_flat = Vec::<i32>::with_capacity(slots.len() * 64);
    for &slot in slots {
        let qt = prep
            .quant_tables
            .get(slot)
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                format!("component references undefined quantisation table slot {slot}")
            })?;
        if qt.values.len() != 64 {
            return Err(format!(
                "quant table {slot} has {} entries (expected 64)",
                qt.values.len()
            ));
        }
        qt_flat.extend(qt.values.iter().map(|&v| i32::from(v)));
    }
    // slots has 1 or 2 entries, so qt_flat.len() / 64 ≤ 2 — fits in u32.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "at most 2 packed tables; quotient fits in u32"
    )]
    let num_qtables = (qt_flat.len() / 64) as u32;
    Ok((qt_flat, num_qtables))
}

/// Number of 8×8 blocks this component contributes per MCU.
/// Non-interleaved (1-component scan): always 1.
/// Interleaved: `h_sampling` × `v_sampling`.
const fn blocks_permcu_count(comp: JpegFrameComponent) -> usize {
    (comp.h_sampling as usize) * (comp.v_sampling as usize)
}

/// Reconstruct AC coefficient arrays from a GPU-decoded symbol stream.
///
/// The GPU Phase 4 output contains Huffman symbol bytes only — no magnitude
/// bits. This function does a single sequential pass over the raw bitstream,
/// consuming only the magnitude bits interleaved between codewords; it
/// performs no Huffman table lookups.
///
/// Returns `(coef_flat, dc_flat)`:
/// - `coef_flat`: `num_components × blocks_per_comp × 64` i32 in zigzag order.
///   DC coefficient at index 0 of each block is filled from `prep.dc_values`;
///   AC coefficients 1..63 come from the symbol stream + magnitude bits.
/// - `dc_flat`: `num_components × blocks_per_comp` i32 (absolute DC values
///   from the CPU pre-pass, copied directly from `prep.dc_values`).
///
/// Returns `Err(String)` if the symbol stream length doesn't match the
/// expected MCU count, or if magnitude bits overflow the bitstream.
#[expect(
    clippy::too_many_lines,
    reason = "DC + AC magnitude walk + dc_flat copy; same structure as extract_coefficients"
)]
fn symbols_to_coefficients(
    prep: &JpegPreparedInput,
    symbols: &[u32],
) -> std::result::Result<(Vec<i32>, Vec<i32>), String> {
    let num_comp = prep.components.len();
    let blocks_wide = usize::from(prep.width.div_ceil(8));
    let blocks_high = usize::from(prep.height.div_ceil(8));
    let blocks_per_comp = blocks_wide * blocks_high;

    let mut coef_flat = vec![0i32; num_comp * blocks_per_comp * 64];

    // Reconstruct the raw byte stream for magnitude bit extraction.
    let raw_bytes: Vec<u8> = prep
        .bitstream
        .words
        .iter()
        .flat_map(|w| w.to_be_bytes())
        .collect();
    let mut bits = BitReader::new(&raw_bytes);

    let totalmcus = mcu_count(prep.width, prep.height, &prep.components);
    let mut block_counts = vec![0usize; num_comp];
    let mut sym_idx = 0usize;

    for mcu in 0..totalmcus {
        for (ci, comp) in prep.components.iter().enumerate() {
            let bpm = blocks_permcu_count(*comp);
            for _b in 0..bpm {
                let block_idx = block_counts[ci];
                let coef_base = (ci * blocks_per_comp + block_idx) * 64;
                block_counts[ci] += 1;

                // DC: consume the DC symbol from the stream.  The magnitude bits
                // in the bitstream must be consumed to keep the bit position
                // aligned with the AC symbols that follow.
                if sym_idx >= symbols.len() {
                    return Err(format!(
                        "symbol stream too short at DC (mcu={mcu} comp={ci})"
                    ));
                }
                let dc_sym = (symbols[sym_idx] & 0xFF) as u8;
                sym_idx += 1;
                let category = dc_sym;
                if category > 0 {
                    let _ = bits.read_bits(usize::from(category)).ok_or_else(|| {
                        format!("DC magnitude truncated (mcu={mcu} comp={ci} cat={category})")
                    })?;
                }
                // DC coefficient value comes from the CPU-resolved chain; coef[0] is left 0
                // and dc_flat is filled separately below.

                // AC: read each symbol + magnitude bits until EOB or 63 slots filled.
                let mut zz = 1usize;
                while zz < 64 {
                    if sym_idx >= symbols.len() {
                        return Err(format!(
                            "symbol stream too short at AC (mcu={mcu} comp={ci} zz={zz})"
                        ));
                    }
                    let sym_byte = (symbols[sym_idx] & 0xFF) as u8;
                    sym_idx += 1;

                    if sym_byte == 0x00 {
                        // EOB: remaining ACs are zero (already initialised).
                        break;
                    }
                    if sym_byte == 0xF0 {
                        // ZRL: 16 zeros, no magnitude bits.
                        zz += 16;
                        continue;
                    }
                    let run = (sym_byte >> 4) as usize;
                    let size = sym_byte & 0x0F;
                    zz += run;
                    if zz >= 64 {
                        break;
                    }
                    let ac_val = if size == 0 {
                        0i32
                    } else {
                        let raw = bits.read_bits(usize::from(size)).ok_or_else(|| {
                            format!(
                                "AC magnitude truncated (mcu={mcu} comp={ci} zz={zz} size={size})"
                            )
                        })?;
                        jpeg_extend(raw.cast_signed(), size)
                    };
                    if coef_base + zz >= coef_flat.len() {
                        return Err(format!(
                            "coefficient overflow: coef_base={coef_base} zz={zz} \
                             coef_flat.len()={} (mcu={mcu} comp={ci})",
                            coef_flat.len()
                        ));
                    }
                    coef_flat[coef_base + zz] = ac_val;
                    zz += 1;
                }
            }
        }
    }

    // DC values: pre-resolved absolute DC chain from the pre-pass.
    let mut dc_flat = vec![0i32; num_comp * blocks_per_comp];
    for (ci, dc_vec) in prep
        .dc_values
        .per_component
        .iter()
        .take(num_comp)
        .enumerate()
    {
        let base = ci * blocks_per_comp;
        for (bi, &dc) in dc_vec.iter().enumerate() {
            let idx = base + bi;
            if idx >= dc_flat.len() {
                return Err(format!(
                    "dc_values overflow: comp={ci} block={bi} idx={idx} dc_flat.len()={}",
                    dc_flat.len()
                ));
            }
            dc_flat[idx] = dc;
        }
    }

    Ok((coef_flat, dc_flat))
}

/// JPEG EXTEND: sign-extend an `nbits`-wide magnitude into a signed integer.
const fn jpeg_extend(value: i32, nbits: u8) -> i32 {
    if nbits == 0 {
        return 0;
    }
    let vt = 1i32 << (nbits - 1);
    if value < vt {
        value + (-1 << nbits) + 1
    } else {
        value
    }
}

/// Test-only re-export of the private `extract_coefficients` so sibling
/// test modules (e.g., `cpu_prepass` DRI tests) can call it without
/// duplicating the coefficient-extraction logic.
#[cfg(test)]
pub(crate) fn extract_coefficients_pub(
    prep: &crate::jpeg_decoder::cpu_prepass::JpegPreparedInput,
) -> std::result::Result<(Vec<i32>, Vec<i32>, Vec<i32>, u32), String> {
    extract_coefficients(prep)
}

#[cfg(test)]
mod qt_selection_tests {
    use crate::jpeg_decoder::cpu_prepass::prepare_jpeg;

    /// Build a grayscale JPEG that defines DQT slots 0 (the fixture's own
    /// table) and 1 (all-2s), with the single frame component referencing
    /// slot 1.  Encoders legally emit both tables even when only one is
    /// used.
    fn gray_jpeg_with_tq1() -> Vec<u8> {
        let base = crate::jpeg::test_fixtures::GRAY_16X16_JPEG;
        let sof = base
            .windows(2)
            .position(|w| w == [0xff, 0xc0])
            .expect("fixture must contain an SOF0 marker");

        let mut out = Vec::with_capacity(base.len() + 69);
        out.extend_from_slice(&base[..sof]);
        // DQT for slot 1: Pq/Tq byte 0x01, then 64 quantiser bytes of 2.
        out.extend_from_slice(&[0xff, 0xdb, 0x00, 0x43, 0x01]);
        out.extend_from_slice(&[0x02; 64]);
        out.extend_from_slice(&base[sof..]);

        // SOF0 layout: ff c0 len(2) precision dims(4) nc, then per-component
        // (id, sampling, Tq) — Tq is byte 12 of the segment.
        let tq = sof + 69 + 12;
        assert_eq!(out[tq], 0x00, "fixture component must reference slot 0");
        out[tq] = 0x01;
        out
    }

    /// The kernel dequantises component 0 with qt index 0, so index 0 of
    /// the packed tables must hold the table the component references —
    /// not whichever populated slot sorts first.
    #[test]
    fn grayscale_component_gets_its_referenced_qtable() {
        let bytes = gray_jpeg_with_tq1();
        let prep = prepare_jpeg(&bytes).expect("prepare crafted grayscale JPEG");
        assert_eq!(prep.components.len(), 1);
        assert_eq!(prep.components[0].quant_selector, 1);

        let (_coef, _dc, qt_flat, num_qtables) =
            super::extract_coefficients_pub(&prep).expect("extract");
        assert_eq!(
            &qt_flat[..64],
            &[2i32; 64][..],
            "packed table 0 must be the component's referenced (slot 1) table"
        );
        assert_eq!(num_qtables, 1, "grayscale packs exactly one table");
    }
}

#[cfg(all(test, feature = "gpu-validation"))]
mod tests {
    use super::*;
    use crate::backend::cuda::CudaBackend;

    #[test]
    fn decoder_decodes_grayscale_jpeg_on_cuda() {
        let backend = CudaBackend::new().expect("CUDA backend");
        let dec = JpegGpuDecoder::new(backend);
        let bytes = crate::jpeg::test_fixtures::GRAY_16X16_JPEG;
        let img = dec.decode(bytes).expect("decode");
        assert_eq!(img.width, 16);
        assert_eq!(img.height, 16);
    }

    /// Download device pixels and compare to a zune-jpeg reference decode.
    ///
    /// Tolerance: peak absolute error ≤ 1 LSB per channel, mean absolute
    /// error ≤ 0.05 LSB.  This is a cross-implementation check, not an
    /// accuracy proof: zune's IDCT uses 12-bit fixed-point constants where
    /// ours uses 13-bit, so the two conforming decoders disagree by ±1 on
    /// a few percent of samples (observed ≈ 0.033 mean on these fixtures).
    /// Absolute accuracy is pinned by `idct_kernel_matches_float_reference`.
    fn assert_close_to_reference_decoder(gpu_rgba: &[u8], zune_rgba: &[u8], label: &str) {
        assert_eq!(
            gpu_rgba.len(),
            zune_rgba.len(),
            "{label}: pixel buffer length mismatch"
        );
        assert_eq!(
            gpu_rgba.len() % 4,
            0,
            "{label}: buffer length not a multiple of 4"
        );
        let mut peak = 0u8;
        let mut sum: u64 = 0;
        let mut colour_count: u64 = 0;
        // Compare only the three colour channels (R, G, B) per pixel.
        // Alpha is unconditionally 0xFF on both sides; including it would
        // inflate the sample count by 33% and deflate the mean error.
        for (gp, rp) in gpu_rgba.chunks_exact(4).zip(zune_rgba.chunks_exact(4)) {
            for (&g, &r) in gp[..3].iter().zip(rp[..3].iter()) {
                let diff = g.abs_diff(r);
                if diff > peak {
                    peak = diff;
                }
                sum += u64::from(diff);
                colour_count += 1;
            }
        }
        let mean = sum as f64 / colour_count as f64;
        assert!(
            peak <= 1,
            "{label}: peak error {peak} > 1 LSB vs reference decoder"
        );
        assert!(
            mean <= 0.05,
            "{label}: mean error {mean:.4} > 0.05 LSB vs reference decoder"
        );
    }

    /// Decode `bytes` with zune-jpeg in RGBA8 and return the flat pixel buffer.
    fn zune_decode_rgba(bytes: &[u8]) -> Vec<u8> {
        use zune_jpeg::JpegDecoder;
        use zune_jpeg::zune_core::bytestream::ZCursor;
        use zune_jpeg::zune_core::colorspace::ColorSpace;
        use zune_jpeg::zune_core::options::DecoderOptions;
        let opts = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGBA);
        let mut dec = JpegDecoder::new_with_options(ZCursor::new(bytes), opts);
        dec.decode().expect("zune-jpeg decode")
    }

    /// Download RGBA8 bytes from a `DeviceImage` via `download_async`.
    fn download_image(backend: &CudaBackend, img: &DeviceImage<CudaBackend>) -> Vec<u8> {
        let n = (img.width as usize) * (img.height as usize) * 4;
        let mut buf = vec![0u8; n];
        let handle = backend
            .download_async(&img.buffer, &mut buf)
            .expect("download_async");
        backend.wait_download(handle).expect("wait_download");
        buf
    }

    #[test]
    fn real_jpeg_q95_pixel_diff_close_to_reference_decoder() {
        let bytes = include_bytes!("../../../../tests/fixtures/jpeg/q95_scan.jpg");
        let backend = CudaBackend::new().expect("CUDA backend");
        let dec = JpegGpuDecoder::new(backend);
        let img = dec.decode(bytes).expect("decode q95_scan.jpg");
        let gpu_rgba = download_image(dec.backend(), &img);
        let zune_rgba = zune_decode_rgba(bytes);
        assert_close_to_reference_decoder(&gpu_rgba, &zune_rgba, "q95_scan.jpg");
    }

    #[test]
    fn real_jpeg_q20_pixel_diff_close_to_reference_decoder() {
        let bytes = include_bytes!("../../../../tests/fixtures/jpeg/q20.jpg");
        let backend = CudaBackend::new().expect("CUDA backend");
        let dec = JpegGpuDecoder::new(backend);
        let img = dec.decode(bytes).expect("decode q20.jpg");
        let gpu_rgba = download_image(dec.backend(), &img);
        let zune_rgba = zune_decode_rgba(bytes);
        assert_close_to_reference_decoder(&gpu_rgba, &zune_rgba, "q20.jpg");
    }

    /// Standard JPEG zigzag order: zigzag index → natural (row-major) index.
    const ZIGZAG_TO_NATURAL: [usize; 64] = [
        0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27,
        20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
    ];

    /// GPU IDCT accuracy against the exact inverse transform.
    ///
    /// Random level-shifted spatial blocks are forward-transformed in f64,
    /// rounded to integer coefficients, and pushed through the kernel with
    /// identity quantisation tables, so the kernel's dequant + IDCT is
    /// measured against the mathematically exact f64 inverse of the same
    /// coefficients: peak ≤ 1 LSB, mean ≤ 0.02 LSB.
    #[test]
    fn idct_kernel_matches_float_reference() {
        use std::f64::consts::{FRAC_1_SQRT_2, PI};

        const BW: usize = 16;
        const BH: usize = 16;
        let w = (BW * 8) as u32;
        let h = (BH * 8) as u32;

        let mut nat_to_zz = [0usize; 64];
        for (zz, &nat) in ZIGZAG_TO_NATURAL.iter().enumerate() {
            nat_to_zz[nat] = zz;
        }

        let mut state = 0x2468_ace1u32;
        let mut next_u8 = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        };
        let basis =
            |i: usize, k: usize| (f64::from((2 * i + 1) as u32) * k as f64 * PI / 16.0).cos();
        let norm = |k: usize| if k == 0 { FRAC_1_SQRT_2 } else { 1.0 };

        let n_blocks = BW * BH;
        let mut coef_flat = vec![0i32; n_blocks * 64];
        let mut dc_flat = vec![0i32; n_blocks];
        let mut expect = vec![0u8; (w as usize) * (h as usize)];

        for block_idx in 0..n_blocks {
            let (bx, by) = (block_idx % BW, block_idx / BW);

            let mut f = [[0f64; 8]; 8];
            for row in &mut f {
                for v in row.iter_mut() {
                    *v = f64::from(next_u8()) - 128.0;
                }
            }

            // Forward DCT, rounded to the integers the kernel will consume.
            let mut coef = [[0i64; 8]; 8];
            for v in 0..8 {
                for u in 0..8 {
                    let mut s = 0.0;
                    for (y, row) in f.iter().enumerate() {
                        for (x, &px) in row.iter().enumerate() {
                            s += px * basis(x, u) * basis(y, v);
                        }
                    }
                    let c = (0.25 * norm(u) * norm(v) * s).round();
                    coef[v][u] = c as i64;
                    let nat = v * 8 + u;
                    coef_flat[block_idx * 64 + nat_to_zz[nat]] = c as i32;
                }
            }
            dc_flat[block_idx] = coef[0][0] as i32;

            // Exact inverse of the rounded coefficients.
            for y in 0..8 {
                for x in 0..8 {
                    let mut s = 0.0;
                    for v in 0..8 {
                        for u in 0..8 {
                            s += norm(u) * norm(v) * coef[v][u] as f64 * basis(x, u) * basis(y, v);
                        }
                    }
                    let px = (0.25 * s + 128.0).round().clamp(0.0, 255.0) as u8;
                    expect[(by * 8 + y) * (w as usize) + bx * 8 + x] = px;
                }
            }
        }

        let backend = CudaBackend::new().expect("CUDA backend");
        let dec = JpegGpuDecoder::new(backend);
        let qt = vec![1i32; 64];
        let img = dec
            .dispatch_idct(&coef_flat, &qt, &dc_flat, w, h, 1, BW as u32, BH as u32, 1)
            .expect("dispatch_idct");
        let rgba = download_image(dec.backend(), &img);

        let mut peak = 0u8;
        let mut sum = 0u64;
        for (px, &e) in rgba.chunks_exact(4).zip(expect.iter()) {
            let d = px[0].abs_diff(e);
            peak = peak.max(d);
            sum += u64::from(d);
        }
        let mean = sum as f64 / expect.len() as f64;
        assert!(peak <= 1, "peak {peak} > 1 LSB vs f64 reference IDCT");
        assert!(
            mean <= 0.02,
            "mean {mean:.4} > 0.02 LSB vs f64 reference IDCT"
        );
    }
}
