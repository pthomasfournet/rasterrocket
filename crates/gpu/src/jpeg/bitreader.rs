//! MSB-first bit reader over an unstuffed JPEG entropy-coded segment.
//!
//! Shared between the CPU pre-pass DC-chain walker
//! ([`crate::jpeg::dc_chain`]) and the on-GPU decoder's CPU oracle
//! ([`crate::jpeg_decoder::jpeg_framing`]).  Both walk the same wire
//! format; keeping the reader in one place stops the two copies from
//! silently drifting on bug fixes.
//!
//! Bit ordering matches JPEG ISO/IEC 10918-1 § F.2.2.4 — the first bit
//! of every byte is its MSB, codewords pack left-to-right.

/// Bit reader over an unstuffed JPEG entropy-coded stream.  MSB-first.
pub(crate) struct BitReader<'a> {
    src: &'a [u8],
    /// Byte position of the next byte to load into the buffer.
    byte_pos: usize,
    /// Buffered bits, MSB-aligned.  `cap` bits are valid in the high end;
    /// the low `64 - cap` bits are zero.
    buf: u64,
    /// Number of valid bits in `buf` (0..=64).
    cap: u32,
}

impl<'a> BitReader<'a> {
    pub(crate) const fn new(src: &'a [u8]) -> Self {
        Self {
            src,
            byte_pos: 0,
            buf: 0,
            cap: 0,
        }
    }

    /// Refill `buf` with as many full bytes as fit, up to 8 bytes / 64 bits
    /// or until input runs out.
    fn refill(&mut self) {
        // Fast path: when the buffer is empty and at least 8 bytes
        // remain, fill all 64 bits in one big-endian load.  This avoids
        // 8× the branches and shifts of the scalar loop on the typical
        // Huffman codeword path.  The byte-by-byte loop below handles
        // the tail and any case where we're already mid-byte.
        //
        // `src.len() - byte_pos >= 8` is the overflow-safe phrasing of
        // `byte_pos + 8 <= src.len()` — the latter wraps to a true
        // result when `byte_pos > usize::MAX - 8` (impossible in
        // practice, but the safe form documents the invariant).
        if self.cap == 0 && self.byte_pos <= self.src.len() && self.src.len() - self.byte_pos >= 8 {
            let bytes: [u8; 8] = self.src[self.byte_pos..self.byte_pos + 8]
                .try_into()
                .expect("slice length checked above");
            self.buf = u64::from_be_bytes(bytes);
            self.byte_pos += 8;
            self.cap = 64;
            return;
        }
        while self.cap <= 56 && self.byte_pos < self.src.len() {
            let byte = u64::from(self.src[self.byte_pos]);
            self.byte_pos += 1;
            self.buf |= byte << (56 - self.cap);
            self.cap += 8;
        }
    }

    /// Peek the next 16 bits MSB-first; pads with zeros if fewer remain.
    /// Returns `None` only if the buffer is completely empty.
    pub(crate) fn peek_u16(&mut self) -> Option<u16> {
        self.refill();
        if self.cap == 0 {
            return None;
        }
        // Top 16 bits of buf, with zero-pad if cap < 16 (high bits remain
        // as we left them; refill above appends from the high end).  The
        // shift makes the truncation provably lossless.
        Some((self.buf >> 48) as u16)
    }

    /// Consume `n` bits if at least `n` remain in the stream, refilling
    /// first. Returns `false` — leaving the reader untouched — when the
    /// stream is too short.
    ///
    /// This is the truncation-safe form of the peek/lookup/consume
    /// pattern: [`Self::peek_u16`] zero-pads past the stream end, so a
    /// padded prefix can match a codeword longer than the bits that
    /// actually remain — wire data, not a caller bug, and it must
    /// surface as the caller's typed truncation error.
    #[must_use]
    pub(crate) fn try_consume(&mut self, n: usize) -> bool {
        self.refill();
        if (self.cap as usize) < n {
            return false;
        }
        let n_u32 = u32::try_from(n).expect("n ≤ cap ≤ 64 fits in u32");
        // checked_shl covers n == 64 (a full-buffer consume), where a
        // plain shift would overflow.
        self.buf = self.buf.checked_shl(n_u32).unwrap_or(0);
        self.cap -= n_u32;
        true
    }

    /// Read `n` bits MSB-first as an unsigned integer.  Returns `None` if
    /// fewer than `n` bits remain in the stream.
    ///
    /// Callers that don't need the value (e.g., the oracle skipping AC
    /// magnitude bits) can drop it with `.is_some()`.
    ///
    /// # Panics
    ///
    /// Panics (in both debug and release) if `n > 16`.  JPEG codewords
    /// cap at 16 bits per ISO/IEC 10918-1 § F.1.2; values above that
    /// would corrupt the right-shift arithmetic (returning truncated
    /// or undefined bits) and almost always indicate a buggy caller.
    pub(crate) fn read_bits(&mut self, n: usize) -> Option<u32> {
        if n == 0 {
            return Some(0);
        }
        // Hard cap: 16 bits is the JPEG spec maximum and the
        // mathematically safe shift range for this code path.
        // Asking for more is a structural caller bug.
        assert!(n <= 16, "BitReader::read_bits: n = {n} exceeds 16-bit cap");
        self.refill();
        if (self.cap as usize) < n {
            return None;
        }
        let n_u32 = u32::try_from(n).expect("n ≤ 16 (asserted above)");
        // 64 - n_u32 is in 48..=63 (since 1 ≤ n_u32 ≤ 16) — always a
        // well-defined u64 shift.  The narrowing cast keeps the
        // low-order `n` bits of the right-shifted value; the high
        // 64 − n bits of buf are zero by construction.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "n ≤ 16 implies the right-shifted value fits in u32"
        )]
        let bits = (self.buf >> (64 - n_u32)) as u32;
        self.buf <<= n_u32;
        self.cap -= n_u32;
        Some(bits)
    }

    /// Discard the buffer and seek to `byte_offset` on a byte boundary.
    /// Used when crossing an RST marker — the entropy stream is byte-aligned
    /// at every RST per JPEG § F.1.1.5, so we drop any in-flight bits.
    /// Clamps to the end of the input slice rather than panicking.
    pub(crate) fn realign_to_byte_at(&mut self, byte_offset: usize) {
        self.buf = 0;
        self.cap = 0;
        self.byte_pos = byte_offset.min(self.src.len());
    }

    /// Position of the byte the reader is currently consuming from,
    /// measured in input bytes.  Reports the byte the reader is
    /// currently *inside* (`cap % 8 != 0`) or about to read next
    /// (`cap % 8 == 0`); the ceiling division gives the inside-byte
    /// semantics.
    ///
    /// Test-only.  Production callers wanting RST-boundary detection
    /// should drive the walker by MCU index instead — see
    /// [`crate::jpeg::dc_chain::resolve_dc_chain`] for the canonical
    /// pattern.  The accessor is preserved here to document the
    /// invariant for future readers.
    #[cfg(test)]
    pub const fn byte_position(&self) -> usize {
        let bytes_unconsumed = (self.cap as usize).div_ceil(8);
        self.byte_pos.saturating_sub(bytes_unconsumed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_bytes_msb_first() {
        let src = [0b1010_1100, 0b1111_0000];
        let mut br = BitReader::new(&src);
        assert_eq!(br.read_bits(4), Some(0b1010));
        assert_eq!(br.read_bits(4), Some(0b1100));
        assert_eq!(br.read_bits(4), Some(0b1111));
        assert_eq!(br.read_bits(4), Some(0b0000));
        assert_eq!(br.read_bits(1), None);
    }

    #[test]
    fn handles_mixed_widths() {
        let src = [0b1110_0101, 0b1100_0011];
        let mut br = BitReader::new(&src);
        assert_eq!(br.read_bits(3), Some(0b111));
        assert_eq!(br.read_bits(5), Some(0b00101));
        assert_eq!(br.read_bits(8), Some(0b1100_0011));
    }

    #[test]
    fn peek_does_not_consume() {
        let src = [0xAB, 0xCD];
        let mut br = BitReader::new(&src);
        let p1 = br.peek_u16().unwrap();
        let p2 = br.peek_u16().unwrap();
        assert_eq!(p1, p2);
        assert_eq!(p1, 0xABCD);
    }

    #[test]
    fn pads_short_input_with_zeros_in_peek() {
        let src = [0xAB];
        let mut br = BitReader::new(&src);
        assert_eq!(br.peek_u16(), Some(0xAB00));
    }

    #[test]
    fn realign_clears_buffer_and_seeks() {
        let src = [0xAA, 0xBB, 0xCC, 0xDD];
        let mut br = BitReader::new(&src);
        // Consume part of one byte.
        let _ = br.read_bits(4);
        br.realign_to_byte_at(2);
        // Next read should now come from src[2] = 0xCC.
        assert_eq!(br.read_bits(8), Some(0xCC));
    }

    #[test]
    fn realign_clamps_past_end() {
        let src = [0xAA];
        let mut br = BitReader::new(&src);
        br.realign_to_byte_at(99);
        assert_eq!(br.peek_u16(), None);
    }

    #[test]
    fn byte_position_tracks_mid_byte() {
        // Pin down the byte_position contract: it must report the byte
        // the reader is currently *inside*, not the next one.  This is
        // the load-bearing invariant for RST-boundary detection.
        let src = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x11, 0x22];
        let mut br = BitReader::new(&src);
        // Force a refill to a known state.
        let _ = br.peek_u16();
        assert_eq!(br.byte_position(), 0);
        // Consume one full byte: position advances to byte 1.
        let _ = br.read_bits(8);
        assert_eq!(br.byte_position(), 1);
        // Mid-byte: still inside byte 1.
        let _ = br.read_bits(4);
        assert_eq!(br.byte_position(), 1);
        // Cross into byte 2.
        let _ = br.read_bits(4);
        assert_eq!(br.byte_position(), 2);
    }

    #[test]
    fn empty_input_yields_no_peek() {
        let src: [u8; 0] = [];
        let mut br = BitReader::new(&src);
        assert_eq!(br.peek_u16(), None);
        assert_eq!(br.read_bits(1), None);
    }

    #[test]
    fn read_zero_bits_succeeds_without_consuming() {
        let src = [0xFF, 0xFF];
        let mut br = BitReader::new(&src);
        assert_eq!(br.read_bits(0), Some(0));
        assert_eq!(br.peek_u16(), Some(0xFFFF));
    }

    #[test]
    fn read_exactly_16_bits_is_allowed() {
        // The 16-bit cap is inclusive — pin it down so a future
        // tightening to `< 16` (off-by-one) trips this test.
        let src = [0xAB, 0xCD];
        let mut br = BitReader::new(&src);
        assert_eq!(br.read_bits(16), Some(0xABCD));
    }

    #[test]
    #[should_panic(expected = "exceeds 16-bit cap")]
    fn read_bits_panics_on_too_wide_request() {
        // 17 bits is structurally invalid for baseline JPEG; the
        // reader must refuse loudly rather than silently truncate or
        // shift-overflow.
        let src = [0xFF, 0xFF, 0xFF];
        let mut br = BitReader::new(&src);
        let _ = br.read_bits(17);
    }

    #[test]
    fn try_consume_refuses_past_stream_end_without_mutating() {
        // A zero-padded peek can match a codeword longer than the bits
        // that remain; try_consume must refuse and leave the reader
        // usable rather than panicking (wire data, not a caller bug).
        let mut br = BitReader::new(&[0b1000_0000]);
        assert!(br.try_consume(7));
        assert!(!br.try_consume(2), "only 1 bit remains");
        assert_eq!(br.read_bits(1), Some(0), "refusal must not consume");
        assert!(!br.try_consume(1), "stream exhausted");
    }

    #[test]
    fn try_consume_full_buffer_is_well_defined() {
        // n == 64 exercises the checked shift (a plain `<<= 64` would
        // overflow).
        let mut br = BitReader::new(&[0xAA; 8]);
        let _ = br.peek_u16(); // force a full 64-bit refill
        assert!(br.try_consume(64));
        assert_eq!(br.peek_u16(), None);
    }
}
