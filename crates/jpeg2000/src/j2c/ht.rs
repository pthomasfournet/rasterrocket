//! The HTJ2K block decoder (ITU-T T.814): the cleanup pass and the optional
//! significance-propagation and magnitude-refinement passes of one HT set.
//!
//! A code-block's cleanup segment holds three bitstreams: MagSgn growing
//! forward from its start, and MEL growing forward and VLC growing backward
//! from the two ends of its last `Scup` bytes.  The refinement segment holds
//! SigProp growing forward and MagRef growing backward.
//!
//! Samples are produced in a doubled domain, one bit below the coefficient's
//! own least significant bit: a sample decoded down to bitplane `b` holds
//! `2μ + 1` shifted to bit `b`, so its lowest set bit marks the centre of
//! its quantisation bin.  [`HtDecodeContext::coefficient_rows`] turns that
//! into the magnitude and decoded bit position the reconstruction expects.
//!
//! Copyright (c) 2021, Aous Naman
//! Copyright (c) 2021, Kakadu Software Pty Ltd, Australia
//! Copyright (c) 2021, The University of New South Wales, Australia
//!
//! Redistribution and use in source and binary forms, with or without
//! modification, are permitted provided that the following conditions are
//! met:
//!
//! 1. Redistributions of source code must retain the above copyright
//!    notice, this list of conditions and the following disclaimer.
//!
//! 2. Redistributions in binary form must reproduce the above copyright
//!    notice, this list of conditions and the following disclaimer in the
//!    documentation and/or other materials provided with the distribution.
//!
//! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
//! IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
//! TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//! PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
//! HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//! SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
//! TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//! PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//! LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//! NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//! SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

use alloc::vec::Vec;

use super::bitplane::{Coefficient, CoefficientState};
use super::ht_tables::{VLC_TBL0, VLC_TBL1};

/// Entries of a significance (σ) or membership (mbr) row: 4 rows × 8
/// columns per `u32`, one nibble per column, for code-blocks up to 1024
/// samples wide, plus room for the right neighbour of the last column.
const STRIPE_WORDS: usize = 132;

/// One line-state byte per quad of a 1024-sample row, plus a guard on
/// either side.
const LINE_STATE_LEN: usize = 528;

/// The widest code-block (T.814 and Table A.18 bound both sides by 1024 and
/// the area by 4096).
const MAX_BLOCK_SIDE: u32 = 1024;
const MAX_BLOCK_AREA: u32 = 4096;

/// Reusable state of the HT block decoder and its output.
#[derive(Default)]
pub(crate) struct HtDecodeContext {
    /// Decoded samples, row-major: bit 31 the sign, bits 0–30 the doubled
    /// magnitude (see the module documentation).
    samples: Vec<u32>,
    width: u32,
    height: u32,
    coefficients: Vec<Coefficient>,
    states: Vec<CoefficientState>,
}

impl HtDecodeContext {
    /// The decoded coefficients, one row at a time.
    pub(crate) fn coefficient_rows(
        &self,
    ) -> impl Iterator<Item = (&[Coefficient], &[CoefficientState])> {
        let w = self.width.max(1) as usize;
        self.coefficients
            .chunks_exact(w)
            .zip(self.states.chunks_exact(w))
    }

    /// Convert the doubled-domain samples into coefficients.
    fn finish(&mut self) {
        self.coefficients.clear();
        self.states.clear();
        for &s in &self.samples {
            let mag = s & 0x7FFF_FFFF;
            if mag == 0 {
                self.coefficients.push(Coefficient::default());
                self.states.push(CoefficientState::default());
                continue;
            }
            let centre = mag.trailing_zeros();
            let magnitude = (mag ^ (1 << centre)) >> 1;
            self.coefficients
                .push(Coefficient::from_sign_magnitude(s >> 31, magnitude));
            self.states
                .push(CoefficientState::with_decoded_bit_position(centre as u8));
        }
    }
}

/// The parameters of one HT code-block.
pub(crate) struct HtBlock<'a> {
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// The cleanup segment.
    pub(crate) cleanup: &'a [u8],
    /// The refinement segment (SigProp, then MagRef growing backward).
    pub(crate) refinement: &'a [u8],
    /// Passes of the HT set to decode: 0 (no set: the block is zero),
    /// 1 (cleanup), 2 (+ SigProp) or 3 (+ MagRef).
    pub(crate) passes: u8,
    /// The bitplane `Mb` count of the sub-band (Equation E-2).
    pub(crate) bitplanes: u8,
    /// Missing most significant bitplanes, including those of placeholder
    /// HT sets.
    pub(crate) missing_msbs: u8,
    /// Vertically causal context formation (code-block style bit 3).
    pub(crate) stripe_causal: bool,
}

/// Decode one HT code-block into `ctx`.  `None` when the code-block is
/// malformed.
pub(crate) fn decode(block: &HtBlock<'_>, ctx: &mut HtDecodeContext) -> Result<(), HtError> {
    let (width, height) = (block.width, block.height);
    if width > MAX_BLOCK_SIDE || height > MAX_BLOCK_SIDE || width * height > MAX_BLOCK_AREA {
        return Err(HtError::Size);
    }
    ctx.width = width;
    ctx.height = height;
    ctx.samples.clear();
    ctx.samples.resize((width * height) as usize, 0);

    let decoded = if width > 0 && height > 0 && block.passes > 0 && !block.cleanup.is_empty() {
        BlockDecoder::new(block, &mut ctx.samples).and_then(|mut decoder| decoder.run())
    } else {
        Some(())
    };
    if decoded.is_none() {
        ctx.samples.fill(0);
    }
    ctx.finish();
    decoded.ok_or(HtError::Data)
}

/// Why an HT code-block did not decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HtError {
    /// The block exceeds the code-block size limits; nothing was decoded.
    Size,
    /// The block's data is malformed; the context holds the block zeroed.
    Data,
}

/// Magnitude bitplanes an HT block holds at most: the doubled domain keeps
/// `2μ + 1` at bit `p − 1` within 31 bits.
pub(crate) const MAX_BITPLANES: u8 = 30;

// ── Bitstream readers ───────────────────────────────────────────────────────

/// The MEL bitstream: forward from `Lcup − Scup`, most significant bit
/// first.  After a `0xFF` byte the next byte carries 7 bits.  The last byte
/// it may read is shared with VLC, whose half it reads as ones; past it, it
/// reads `0xFF`.
struct Mel<'a> {
    data: &'a [u8],
    pos: usize,
    remaining: usize,
    tmp: u64,
    bits: u32,
    unstuff: bool,
    k: usize,
}

impl<'a> Mel<'a> {
    fn new(cleanup: &'a [u8], lcup: usize, scup: usize) -> Self {
        Self {
            data: cleanup,
            pos: lcup - scup,
            remaining: scup - 1,
            tmp: 0,
            bits: 0,
            unstuff: false,
            k: 0,
        }
    }

    /// Append the next byte.  `None` on a stuffing violation.
    fn feed(&mut self) -> Option<()> {
        let mut d = if self.remaining > 0 {
            *self.data.get(self.pos)?
        } else {
            0xFF
        };
        if self.remaining > 0 {
            if self.unstuff && d > 0x8F {
                return None;
            }
            if self.remaining == 1 {
                d |= 0x0F;
            }
            self.pos += 1;
            self.remaining -= 1;
        }
        let d_bits = 8 - u32::from(self.unstuff);
        let v = u64::from(d) & ((1 << d_bits) - 1);
        self.tmp |= v << (64 - self.bits - d_bits);
        self.bits += d_bits;
        self.unstuff = d == 0xFF;
        Some(())
    }

    /// The next run: twice the number of 0 events, plus 1 when the run ends
    /// in a 1 event.
    fn run(&mut self) -> Option<i32> {
        const MEL_EXP: [u32; 13] = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5];
        while self.bits < 6 {
            self.feed()?;
        }
        let eval = MEL_EXP[self.k];
        let run = if self.tmp & (1 << 63) != 0 {
            self.k = (self.k + 1).min(12);
            self.tmp <<= 1;
            self.bits -= 1;
            ((1 << eval) - 1) << 1
        } else {
            let r = ((self.tmp >> (63 - eval)) & ((1 << eval) - 1)) as i32;
            self.k = self.k.saturating_sub(1);
            self.tmp <<= eval + 1;
            self.bits -= eval + 1;
            (r << 1) + 1
        };
        Some(run)
    }
}

/// A bitstream growing backward (VLC, MagRef), least significant bit first.
/// A byte after one above `0x8F` that equals `0x7F` in its low 7 bits
/// carries 7 bits.  Past its data it reads zeros.
struct Reverse<'a> {
    data: &'a [u8],
    /// Index of the next byte, counting down; `remaining` bytes are left.
    pos: usize,
    remaining: usize,
    tmp: u64,
    bits: u32,
    unstuff: bool,
}

impl<'a> Reverse<'a> {
    /// The VLC bitstream: backward from `Lcup − 2`, whose upper half is its
    /// first 4 bits (the bytes after it hold `Scup`).
    fn vlc(cleanup: &'a [u8], lcup: usize, scup: usize) -> Self {
        let d = u32::from(cleanup[lcup - 2]);
        let tmp = u64::from(d >> 4);
        let mut r = Self {
            data: cleanup,
            pos: lcup - 2,
            remaining: scup - 2,
            tmp,
            bits: 4 - u32::from(tmp & 7 == 7),
            // Only the high nibble is VLC's; `d > 0x8F` tests it alone.
            unstuff: d > 0x8F,
        };
        r.fill();
        r
    }

    /// The MagRef bitstream: backward from the end of the refinement
    /// segment.
    fn mag_ref(refinement: &'a [u8]) -> Self {
        let mut r = Self {
            data: refinement,
            pos: refinement.len(),
            remaining: refinement.len(),
            tmp: 0,
            bits: 0,
            unstuff: true,
        };
        r.fill();
        r
    }

    fn feed(&mut self) {
        let d = if self.remaining > 0 {
            self.pos -= 1;
            self.remaining -= 1;
            self.data.get(self.pos).copied().unwrap_or(0)
        } else {
            0
        };
        let d_bits = 8 - u32::from(self.unstuff && d & 0x7F == 0x7F);
        self.tmp |= u64::from(d) << self.bits;
        self.bits += d_bits;
        self.unstuff = d > 0x8F;
    }

    fn fill(&mut self) {
        while self.bits < 32 {
            self.feed();
        }
    }

    /// At least the next 32 bits.
    fn fetch(&mut self) -> u32 {
        self.fill();
        self.tmp as u32
    }

    fn advance(&mut self, n: u32) -> u32 {
        debug_assert!(n <= self.bits);
        self.tmp >>= n;
        self.bits -= n.min(self.bits);
        self.tmp as u32
    }
}

/// A bitstream growing forward (MagSgn, SigProp), least significant bit
/// first.  After a `0xFF` byte the next byte carries 7 bits.  Past its data
/// it reads `fill` (`0xFF` for MagSgn, `0` for SigProp).
struct Forward<'a> {
    data: &'a [u8],
    pos: usize,
    tmp: u64,
    bits: u32,
    unstuff: bool,
    fill: u8,
}

impl<'a> Forward<'a> {
    fn new(data: &'a [u8], fill: u8) -> Self {
        Self {
            data,
            pos: 0,
            tmp: 0,
            bits: 0,
            unstuff: false,
            fill,
        }
    }

    fn fetch(&mut self) -> u32 {
        while self.bits < 32 {
            let d = self.data.get(self.pos).copied().unwrap_or(self.fill);
            self.pos += 1;
            self.tmp |= u64::from(d) << self.bits;
            self.bits += 8 - u32::from(self.unstuff);
            self.unstuff = d == 0xFF;
        }
        self.tmp as u32
    }

    fn advance(&mut self, n: u32) {
        debug_assert!(n <= self.bits);
        self.tmp >>= n;
        self.bits -= n.min(self.bits);
    }
}

// ── u-value decoding (T.814, Table 3) ──────────────────────────────────────

/// Prefix codewords of `u`, indexed by the next 3 VLC bits: bits 0–1 the
/// prefix length, bits 2–4 the suffix length, bits 5–7 the prefix value.
const UVLC_PREFIX: [u32; 8] = [
    3 | (5 << 2) | (5 << 5),
    1 | (1 << 5),
    2 | (2 << 5),
    1 | (1 << 5),
    3 | (1 << 2) | (3 << 5),
    1 | (1 << 5),
    2 | (2 << 5),
    1 | (1 << 5),
];

/// Decode one `u` prefix from `vlc`: `(prefix entry, bits consumed)`.
fn uvlc_prefix(vlc: u32) -> (u32, u32) {
    let d = UVLC_PREFIX[(vlc & 7) as usize];
    (d, d & 3)
}

/// `u` from prefix entry `d` and the suffix bits at the head of `vlc`.
fn uvlc_value(d: u32, vlc: u32) -> (u32, u32) {
    let suffix_len = (d >> 2) & 7;
    ((d >> 5) + (vlc & ((1 << suffix_len) - 1)), suffix_len)
}

/// The `u` values (plus κ = 1) of a quad pair in the first row.  `mode`
/// holds the two `u_off` bits, plus 1 when both are set and the MEL event
/// is 1.  Returns the bits consumed.
fn decode_init_uvlc(mut vlc: u32, mode: u32, u: &mut [u32; 2]) -> u32 {
    match mode {
        0..=2 => uvlc_one_of_pair(vlc, mode, u),
        3 => {
            let (d1, len1) = uvlc_prefix(vlc);
            vlc >>= len1;
            if len1 > 2 {
                // The first prefix is "000" or "001": the second quad's
                // u is 1 or 2, coded in one bit.
                u[1] = (vlc & 1) + 2;
                let (v, suffix) = uvlc_value(d1, vlc >> 1);
                u[0] = v + 1;
                len1 + 1 + suffix
            } else {
                let (d2, len2) = uvlc_prefix(vlc);
                let (v, suffixes) = uvlc_suffixes(d1, d2, vlc >> len2);
                *u = [v[0] + 1, v[1] + 1];
                len1 + len2 + suffixes
            }
        }
        _ => {
            let (v, consumed) = uvlc_both(vlc);
            *u = [v[0] + 3, v[1] + 3];
            consumed
        }
    }
}

/// The `u` values (plus κ = 1) of a quad pair in a later row; `mode` holds
/// the two `u_off` bits.  Returns the bits consumed.
fn decode_noninit_uvlc(vlc: u32, mode: u32, u: &mut [u32; 2]) -> u32 {
    match mode {
        0..=2 => uvlc_one_of_pair(vlc, mode, u),
        _ => {
            let (v, consumed) = uvlc_both(vlc);
            *u = [v[0] + 1, v[1] + 1];
            consumed
        }
    }
}

/// The `u` values of a quad pair of which `mode` 1 (the first) or 2 (the
/// second) has a coded offset, or `mode` 0 neither: one prefix and suffix.
/// Returns the bits consumed.
fn uvlc_one_of_pair(vlc: u32, mode: u32, u: &mut [u32; 2]) -> u32 {
    if mode == 0 {
        *u = [1, 1];
        return 0;
    }
    let (d, len) = uvlc_prefix(vlc);
    let (v, suffix) = uvlc_value(d, vlc >> len);
    *u = if mode == 1 { [v + 1, 1] } else { [1, v + 1] };
    len + suffix
}

/// The offsets of both quads of a pair, two prefixes then two suffixes, and
/// the bits consumed.
fn uvlc_both(vlc: u32) -> ([u32; 2], u32) {
    let (d1, len1) = uvlc_prefix(vlc);
    let (d2, len2) = uvlc_prefix(vlc >> len1);
    let (v, suffixes) = uvlc_suffixes(d1, d2, vlc >> (len1 + len2));
    (v, len1 + len2 + suffixes)
}

/// The values of the prefixes `d1` and `d2` from their suffixes at the
/// bottom of `vlc`, and the suffix bits consumed.
fn uvlc_suffixes(d1: u32, d2: u32, vlc: u32) -> ([u32; 2], u32) {
    let (v1, s1) = uvlc_value(d1, vlc);
    let (v2, s2) = uvlc_value(d2, vlc >> s1);
    ([v1, v2], s1 + s2)
}

/// The significance of a quad pair, `qinf` of each, in the packed stripe
/// layout: the first quad's two columns of two in the low byte, the
/// second's four bits above.
const fn sigma_bits(qinf: [u32; 2]) -> u32 {
    let [q0, q1] = qinf;
    (((q0 & 0x30) >> 4) | ((q0 & 0xC0) >> 2)) | (((q1 & 0x30) | ((q1 & 0xC0) << 2)) << 4)
}

/// The significance `sig` of a group of eight columns spread one column
/// left and right, the neighbouring groups `prev` and `next` contributing
/// across the group's edges.
const fn horizontal_spread(prev: u32, sig: u32, next: u32) -> u32 {
    sig | (prev >> 28) | (sig << 4) | (sig >> 4) | (next << 28)
}

// ── The block decoder ──────────────────────────────────────────────────────

struct BlockDecoder<'a, 'b> {
    width: u32,
    height: u32,
    stride: usize,
    /// Bit of the doubled domain holding the cleanup pass's least
    /// significant magnitude bit: one above the bitplane it decodes.
    p: u32,
    passes: u8,
    /// The largest `u` (plus κ) a quad may carry: missing MSBs + 2.
    max_u: u32,
    stripe_causal: bool,
    samples: &'b mut [u32],
    mel: Mel<'a>,
    vlc: Reverse<'a>,
    mag_sgn: Forward<'a>,
    sig_prop: Forward<'a>,
    mag_ref: Reverse<'a>,
    /// Significance of the cleanup pass, alternating per stripe of 4 rows.
    sigma: [[u32; STRIPE_WORDS]; 2],
    /// SigProp membership, alternating per stripe.
    mbr: [[u32; STRIPE_WORDS]; 2],
    /// Per quad: bit 7 the significance of the quad's lower row (σⁿ), bits
    /// 0–6 the largest exponent there (Eⁿ).
    line_state: [u8; LINE_STATE_LEN],
}

impl<'a, 'b> BlockDecoder<'a, 'b> {
    fn new(block: &HtBlock<'a>, samples: &'b mut [u32]) -> Option<Self> {
        let cleanup = block.cleanup;
        let mut passes = block.passes;
        if passes > 3 {
            return None;
        }
        if block.bitplanes > MAX_BITPLANES || block.missing_msbs >= block.bitplanes {
            return None;
        }
        // With `Z` missing most significant bitplanes, the cleanup pass
        // decodes bitplane `Mb − 1 − Z`; `p = Mb − Z` is the bit just above
        // it, where the doubled domain places a sample's `2μ + 1`.
        let p = u32::from(block.bitplanes - block.missing_msbs);
        if p == 1 || block.refinement.is_empty() {
            // No bitplane below the cleanup's, or no refinement data.
            passes = 1;
        }
        let lcup = cleanup.len();
        if lcup < 2 {
            return None;
        }
        let scup = (usize::from(cleanup[lcup - 1]) << 4) | usize::from(cleanup[lcup - 2] & 0xF);
        if scup < 2 || scup > lcup || scup > 4079 {
            return None;
        }
        Some(Self {
            width: block.width,
            height: block.height,
            stride: block.width as usize,
            p,
            passes,
            max_u: u32::from(block.missing_msbs) + 2,
            stripe_causal: block.stripe_causal,
            samples,
            mel: Mel::new(cleanup, lcup, scup),
            vlc: Reverse::vlc(cleanup, lcup, scup),
            mag_sgn: Forward::new(&cleanup[..lcup - scup], 0xFF),
            sig_prop: Forward::new(block.refinement, 0),
            mag_ref: Reverse::mag_ref(if passes > 2 { block.refinement } else { &[] }),
            sigma: [[0; STRIPE_WORDS]; 2],
            mbr: [[0; STRIPE_WORDS]; 2],
            line_state: [0; LINE_STATE_LEN],
        })
    }

    fn sample_mut(&mut self, row: u32, col: u32) -> Option<&mut u32> {
        if col >= self.width {
            return None;
        }
        self.samples
            .get_mut(row as usize * self.stride + col as usize)
    }

    /// Decode one significant sample of a quad: `bit` is its index (0 top
    /// left, 1 bottom left, 2 top right, 3 bottom right).  Returns the
    /// sample and its `v_n` (twice the magnitude less one, with the bin
    /// centre).
    fn mag_sgn_sample(&mut self, qinf: u32, u_q: u32, bit: u32) -> Option<(u32, u32)> {
        let ms_val = self.mag_sgn.fetch();
        let m_n = u_q.checked_sub((qinf >> (12 + bit)) & 1)?;
        if m_n > 30 {
            return None;
        }
        self.mag_sgn.advance(m_n);
        let sign = ms_val << 31;
        let mut v_n = ms_val & ((1 << m_n) - 1);
        v_n |= ((qinf >> (8 + bit)) & 1) << m_n;
        v_n |= 1;
        // In 64 bits, so that no bit shifted past 31 goes unseen.
        let value = u32::try_from(u64::from(v_n + 2) << (self.p - 1))
            .ok()
            .filter(|value| value >> 31 == 0)?;
        Some((sign | value, v_n))
    }

    /// Decode the samples of one quad at `(row, col)` and update the line
    /// state at `ls` (this quad) as the next row's context.
    fn decode_quad(&mut self, qinf: u32, u_q: u32, row: u32, col: u32, ls: usize) -> Option<()> {
        // The left column: top, then bottom.
        if qinf & 0x10 != 0 {
            let (s, _) = self.mag_sgn_sample(qinf, u_q, 0)?;
            *self.sample_mut(row, col)? = s;
        }
        if qinf & 0x20 != 0 {
            let (s, v_n) = self.mag_sgn_sample(qinf, u_q, 1)?;
            *self.sample_mut(row + 1, col)? = s;
            let t = u32::from(self.line_state[ls] & 0x7F);
            let e = 32 - v_n.leading_zeros();
            self.line_state[ls] = (0x80 | t.max(e)) as u8;
        }
        // The right column, whose lower sample is the next quad's NW.
        if qinf & 0x40 != 0 {
            let (s, _) = self.mag_sgn_sample(qinf, u_q, 2)?;
            *self.sample_mut(row, col + 1)? = s;
        }
        self.line_state[ls + 1] = 0;
        if qinf & 0x80 != 0 {
            let (s, v_n) = self.mag_sgn_sample(qinf, u_q, 3)?;
            *self.sample_mut(row + 1, col + 1)? = s;
            self.line_state[ls + 1] = (0x80 | (32 - v_n.leading_zeros())) as u8;
        }
        Some(())
    }

    /// The sample-location mask of the quad pair at column `x` of a row
    /// pair with `two_rows` rows inside the block; `None` when the VLC
    /// marked a sample outside it significant.
    fn check_locs(&self, x: u32, two_rows: bool, qinf: [u32; 2]) -> Option<()> {
        let mut locs = 0xFFu32;
        if x + 4 > self.width {
            locs >>= (x + 4 - self.width) << 1;
        }
        if !two_rows {
            locs &= 0x55;
        }
        if (((qinf[0] & 0xF0) >> 4) | (qinf[1] & 0xF0)) & !locs != 0 {
            return None;
        }
        Some(())
    }

    /// Consume one MEL event for a quad in the zero context: keep `qinf`
    /// when the event is 1.
    fn mel_event(&mut self, run: &mut i32) -> Option<bool> {
        *run -= 2;
        let event = *run == -1;
        if *run < 0 {
            *run = self.mel.run()?;
        }
        Some(event)
    }

    fn run(&mut self) -> Option<()> {
        let (width, height) = (self.width, self.height);
        let mut run = self.mel.run()?;
        let mut sip = 0usize;
        let mut sip_shift = 0u32;

        // The first row of quads.
        let mut c_q = 0u32;
        let mut ls = 0usize;
        self.line_state[0] = 0;
        let mut x = 0;
        while x < width {
            let mut qinf = [0u32; 2];
            let mut vlc_val = self.vlc.fetch();
            qinf[0] = u32::from(VLC_TBL0[((c_q << 7) | (vlc_val & 0x7F)) as usize]);
            if c_q == 0 && !self.mel_event(&mut run)? {
                qinf[0] = 0;
            }
            c_q = ((qinf[0] & 0x10) >> 4) | ((qinf[0] & 0xE0) >> 5);
            vlc_val = self.vlc.advance(qinf[0] & 7);

            if x + 2 < width {
                qinf[1] = u32::from(VLC_TBL0[((c_q << 7) | (vlc_val & 0x7F)) as usize]);
                if c_q == 0 && !self.mel_event(&mut run)? {
                    qinf[1] = 0;
                }
                c_q = ((qinf[1] & 0x10) >> 4) | ((qinf[1] & 0xE0) >> 5);
                vlc_val = self.vlc.advance(qinf[1] & 7);
            }
            self.sigma[0][sip] |= sigma_bits(qinf) << sip_shift;
            if x & 7 != 0 {
                sip += 1;
            }
            sip_shift ^= 0x10;

            let mut mode = ((qinf[0] & 0x8) >> 3) | ((qinf[1] & 0x8) >> 2);
            if mode == 3 && self.mel_event(&mut run)? {
                mode += 1;
            }
            let mut u_q = [0u32; 2];
            let consumed = decode_init_uvlc(vlc_val, mode, &mut u_q);
            if u_q[0] > self.max_u || u_q[1] > self.max_u {
                return None;
            }
            self.vlc.advance(consumed);

            self.check_locs(x, height > 1, qinf)?;
            self.decode_quad(qinf[0], u_q[0], 0, x, ls)?;
            self.decode_quad(qinf[1], u_q[1], 0, x + 2, ls + 1)?;
            ls += 2;
            x += 4;
        }

        // Later rows of quads.
        let mut y = 2;
        while y < height {
            sip_shift ^= 0x2;
            sip_shift &= !0x10;
            let si = usize::from(y & 4 != 0);
            let mut sip = 0usize;
            let mut ls = 0usize;
            let mut ls0 = self.line_state[0];
            self.line_state[0] = 0;
            let mut c_q = 0u32;
            let mut x = 0;
            while x < width {
                let mut qinf = [0u32; 2];
                c_q |= u32::from(ls0 >> 7);
                c_q |= u32::from(self.line_state[ls + 1] >> 5) & 0x4;
                let mut vlc_val = self.vlc.fetch();
                qinf[0] = u32::from(VLC_TBL1[((c_q << 7) | (vlc_val & 0x7F)) as usize]);
                if c_q == 0 && !self.mel_event(&mut run)? {
                    qinf[0] = 0;
                }
                c_q = ((qinf[0] & 0x40) >> 5) | ((qinf[0] & 0x80) >> 6);
                vlc_val = self.vlc.advance(qinf[0] & 7);

                if x + 2 < width {
                    c_q |= u32::from(self.line_state[ls + 1] >> 7);
                    c_q |= u32::from(self.line_state[ls + 2] >> 5) & 0x4;
                    qinf[1] = u32::from(VLC_TBL1[((c_q << 7) | (vlc_val & 0x7F)) as usize]);
                    if c_q == 0 && !self.mel_event(&mut run)? {
                        qinf[1] = 0;
                    }
                    c_q = ((qinf[1] & 0x40) >> 5) | ((qinf[1] & 0x80) >> 6);
                    vlc_val = self.vlc.advance(qinf[1] & 7);
                }
                self.sigma[si][sip] |= sigma_bits(qinf) << sip_shift;
                if x & 7 != 0 {
                    sip += 1;
                }
                sip_shift ^= 0x10;

                let mode = ((qinf[0] & 0x8) >> 3) | ((qinf[1] & 0x8) >> 2);
                let mut u_q = [0u32; 2];
                let consumed = decode_noninit_uvlc(vlc_val, mode, &mut u_q);
                self.vlc.advance(consumed);

                // E^max: a quad with more than one significant sample adds
                // the largest exponent above it (Equations 5 and 6).
                let more_than_one = |q: u32| (q & 0xF0) & (q & 0xF0).wrapping_sub(1) != 0;
                if more_than_one(qinf[0]) {
                    let e = u32::from(ls0 & 0x7F).max(u32::from(self.line_state[ls + 1] & 0x7F));
                    u_q[0] += e.saturating_sub(2);
                }
                if more_than_one(qinf[1]) {
                    let e = u32::from(self.line_state[ls + 1] & 0x7F)
                        .max(u32::from(self.line_state[ls + 2] & 0x7F));
                    u_q[1] += e.saturating_sub(2);
                }
                if u_q[0] > self.max_u || u_q[1] > self.max_u {
                    return None;
                }

                ls0 = self.line_state[ls + 2];
                self.line_state[ls + 1] = 0;
                self.line_state[ls + 2] = 0;

                self.check_locs(x, y + 2 <= height, qinf)?;
                self.decode_quad(qinf[0], u_q[0], y, x, ls)?;
                self.decode_quad(qinf[1], u_q[1], y, x + 2, ls + 1)?;
                ls += 2;
                x += 4;
            }
            y += 2;

            if self.passes > 1 && y & 3 == 0 {
                if self.passes > 2 {
                    self.mag_ref_stripe(y - 4)?;
                }
                self.init_membership(y - 4);
                if y >= 8 {
                    self.add_membership_below(y - 8);
                    self.sig_prop_stripe(y - 8, 0xFFFF_FFFF)?;
                    let cur = Self::stripe_slot(y - 8);
                    let groups = (width as usize).div_ceil(8) + 1;
                    self.sigma[cur][..groups].fill(0);
                }
            }
        }

        if self.passes > 1 {
            let partial = height & 3 == 1 || height & 3 == 2;
            if partial {
                let row0 = height & !3;
                if self.passes > 2 {
                    self.mag_ref_stripe(row0)?;
                }
                self.init_membership(row0);
            }
            let st = if height > 6 {
                height - (((height + 1) & 3) + 3)
            } else {
                0
            };
            let mut y = st;
            while y < height {
                let pattern = match height - y {
                    3 => 0x7777_7777,
                    2 => 0x3333_3333,
                    1 => 0x1111_1111,
                    _ => 0xFFFF_FFFF,
                };
                if height - y > 4 {
                    self.add_membership_below(y);
                }
                self.sig_prop_stripe(y, pattern)?;
                y += 4;
            }
        }
        Some(())
    }

    /// The σ/mbr slot of the stripe starting at `row0`.
    fn stripe_slot(row0: u32) -> usize {
        usize::from(row0 & 4 != 0)
    }

    /// MagRef for the cleanup-significant samples of the stripe at `row0`.
    fn mag_ref_stripe(&mut self, row0: u32) -> Option<()> {
        let cur = Self::stripe_slot(row0);
        let p = self.p;
        let half = 1u32 << (p - 2);
        let mut i = 0;
        let mut g = 0;
        while i < self.width {
            let mut cwd = self.mag_ref.fetch();
            let sig = self.sigma[cur][g];
            if sig != 0 {
                for j in 0..8 {
                    for r in 0..4 {
                        if sig & (1 << (4 * j + r)) != 0 {
                            let sym = cwd & 1;
                            let s = self.sample_mut(row0 + r, i + j)?;
                            *s ^= (1 - sym) << (p - 1);
                            *s |= half;
                            cwd >>= 1;
                        }
                    }
                }
            }
            self.mag_ref.advance(sig.count_ones());
            i += 8;
            g += 1;
        }
        Some(())
    }

    /// SigProp membership of the stripe at `row0` from its own cleanup
    /// significance: the neighbours of significant samples, less those
    /// samples.
    fn init_membership(&mut self, row0: u32) {
        let slot = Self::stripe_slot(row0);
        let mut prev = 0u32;
        let groups = (self.width as usize).div_ceil(8);
        for g in 0..groups {
            let sig = self.sigma[slot][g];
            let next = self.sigma[slot][g + 1];
            let mut m = horizontal_spread(prev, sig, next);
            prev = sig;
            let t = m;
            m |= (t & 0x7777_7777) << 1;
            m |= (t & 0xEEEE_EEEE) >> 1;
            self.mbr[slot][g] = m & !sig;
        }
    }

    /// Add to the membership of the stripe at `row0` the samples above the
    /// next stripe's significant samples (unless contexts are vertically
    /// causal), and drop those already significant.
    fn add_membership_below(&mut self, row0: u32) {
        let cur = Self::stripe_slot(row0);
        let nxt = 1 - cur;
        let mut prev = 0u32;
        let groups = (self.width as usize).div_ceil(8);
        for g in 0..groups {
            let ns = self.sigma[nxt][g];
            let t = horizontal_spread(prev, ns, self.sigma[nxt][g + 1]);
            prev = ns;
            if !self.stripe_causal {
                self.mbr[cur][g] |= (t & 0x1111_1111) << 3;
            }
            self.mbr[cur][g] &= !self.sigma[cur][g];
        }
    }

    /// SigProp for the stripe at `row0`, restricted to the rows in
    /// `pattern`: newly significant samples get their sign and the value
    /// `1.5` at the bitplane below the cleanup's, and their neighbours join
    /// the membership, including the next stripe's.
    fn sig_prop_stripe(&mut self, row0: u32, pattern: u32) -> Option<()> {
        let cur = Self::stripe_slot(row0);
        let nxt = 1 - cur;
        let val = 3u32 << (self.p - 2);
        let width = self.width;
        let mut i = 0;
        let mut g = 0usize;
        while i < width {
            let mut mbr = self.mbr[cur][g] & pattern;
            let mut new_sig = 0u32;
            if mbr != 0 {
                for n in [0u32, 4] {
                    let mut cwd = self.sig_prop.fetch();
                    let mut cnt = 0;
                    let inv_sig = !self.sigma[cur][g] & pattern;
                    let end = (n + 4).min(width - i);
                    const SPREAD: [u32; 4] = [0x32, 0x74, 0xE8, 0xC0];
                    for j in n..end {
                        if mbr & (0xF << (4 * j)) == 0 {
                            continue;
                        }
                        for (r, spread) in SPREAD.iter().enumerate() {
                            let sample_mask = 1u32 << (4 * j + r as u32);
                            if mbr & sample_mask != 0 {
                                if cwd & 1 != 0 {
                                    new_sig |= sample_mask;
                                    mbr |= (spread << (j * 4)) & inv_sig;
                                }
                                cwd >>= 1;
                                cnt += 1;
                            }
                        }
                    }
                    if new_sig & (0xFFFF << (4 * n)) != 0 {
                        for j in n..end {
                            if new_sig & (0xF << (4 * j)) == 0 {
                                continue;
                            }
                            for r in 0..4 {
                                if new_sig & (1 << (4 * j + r)) != 0 {
                                    let s = self.sample_mut(row0 + r, i + j)?;
                                    *s |= ((cwd & 1) << 31) | val;
                                    cwd >>= 1;
                                    cnt += 1;
                                }
                            }
                        }
                    }
                    self.sig_prop.advance(cnt);
                    if n == 4 {
                        let mut t = new_sig >> 28;
                        t |= ((t & 0xE) >> 1) | ((t & 7) << 1);
                        self.mbr[cur][g + 1] |= t & !self.sigma[cur][g + 1];
                    }
                }
            }
            // Propagate into the next stripe.
            new_sig |= self.sigma[cur][g];
            let ux = (new_sig & 0x8888_8888) >> 3;
            let tx = ux | (ux << 4) | (ux >> 4);
            if g > 0 {
                self.mbr[nxt][g - 1] |= (ux << 28) & !self.sigma[nxt][g - 1];
            }
            self.mbr[nxt][g] |= tx & !self.sigma[nxt][g];
            self.mbr[nxt][g + 1] |= (ux >> 28) & !self.sigma[nxt][g + 1];
            i += 8;
            g += 1;
        }
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::{HtBlock, HtDecodeContext, HtError, MAX_BITPLANES, decode};

    /// A block of `width` × `height` with a cleanup segment `cleanup`.
    fn block(cleanup: &[u8], bitplanes: u8, missing_msbs: u8, passes: u8) -> HtBlock<'_> {
        HtBlock {
            width: 4,
            height: 4,
            cleanup,
            refinement: &[],
            passes,
            bitplanes,
            missing_msbs,
            stripe_causal: false,
        }
    }

    /// `Scup` in the cleanup's last two bytes: its high bits, then its low
    /// four in the low nibble.
    fn cleanup(len: usize, scup: usize) -> Vec<u8> {
        let mut data = vec![0u8; len];
        data[len - 1] = u8::try_from(scup >> 4).unwrap();
        data[len - 2] = u8::try_from(scup & 0xF).unwrap();
        data
    }

    #[test]
    fn malformed_blocks_decode_as_zero_and_say_so() {
        let mut ctx = HtDecodeContext::default();
        let good = cleanup(8, 4);
        let cases = [
            (vec![0u8], 8, 0, 1),           // Lcup < 2
            (cleanup(8, 1), 8, 0, 1),       // Scup < 2
            (cleanup(8, 9), 8, 0, 1),       // Scup > Lcup
            (cleanup(5000, 4080), 8, 0, 1), // Scup > 4079
            (good.clone(), 8, 8, 1),        // Z >= Mb
            (good.clone(), MAX_BITPLANES + 1, 0, 1),
            (good, 8, 0, 4), // more than three passes
        ];
        for (data, bitplanes, missing, passes) in cases {
            ctx.samples = vec![7; 16];
            let result = decode(&block(&data, bitplanes, missing, passes), &mut ctx);
            assert_eq!(
                result,
                Err(HtError::Data),
                "Mb {bitplanes}, Z {missing}, {passes} passes"
            );
            assert!(ctx.samples.iter().all(|&s| s == 0), "the block is zeroed");
        }
    }

    #[test]
    fn oversized_blocks_are_refused_without_decoding() {
        let mut ctx = HtDecodeContext::default();
        let data = cleanup(8, 4);
        let mut big = block(&data, 8, 0, 1);
        big.width = 4096;
        big.height = 4096;
        assert_eq!(decode(&big, &mut ctx), Err(HtError::Size));
    }
}
