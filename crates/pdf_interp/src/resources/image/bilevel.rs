//! One-bit samples of a bilevel image (`CCITTFaxDecode`, `JBIG2Decode`).

use std::sync::OnceLock;

/// The samples of a `width × height` image of two values, one bit each:
/// a set bit is the sample [`Self::set`], a clear bit [`Self::clear`].
///
/// Each row is [`Self::words`] 64-bit words, its first sample in the most
/// significant bit of its first word; the bits past the width are clear.
/// The samples as bytes are expanded on first request ([`Self::samples`])
/// and kept.
#[derive(Debug)]
pub struct Bilevel {
    width: u32,
    height: u32,
    words: usize,
    bits: Vec<u64>,
    set: u8,
    clear: u8,
    bytes: OnceLock<Option<Vec<u8>>>,
}

/// For each byte of 8 bits, the 8 bytes `0xFF` where its bit is set and
/// `0x00` where it is clear, first bit first.
static BYTE_MASKS: [[u8; 8]; 256] = {
    let mut table = [[0; 8]; 256];
    let mut byte = 0;
    while byte < 256 {
        let mut bit = 0;
        while bit < 8 {
            if byte & (0x80 >> bit) != 0 {
                table[byte][bit] = 0xFF;
            }
            bit += 1;
        }
        byte += 1;
    }
    table
};

impl Bilevel {
    /// An image of clear bits, or `None` when its bits cannot be allocated.
    pub(crate) fn new(width: u32, height: u32, set: u8, clear: u8) -> Option<Self> {
        let words = (width as usize).div_ceil(64);
        let len = words.checked_mul(height as usize)?;
        let mut bits = Vec::new();
        bits.try_reserve_exact(len)
            .map_err(|e| log::warn!("image: cannot allocate {} bytes: {e}", len * 8))
            .ok()?;
        bits.resize(len, 0);
        Some(Self {
            width,
            height,
            words,
            bits,
            set,
            clear,
            bytes: OnceLock::new(),
        })
    }

    #[must_use]
    pub(crate) const fn width(&self) -> u32 {
        self.width
    }

    #[must_use]
    pub(crate) const fn height(&self) -> u32 {
        self.height
    }

    /// The sample of a set bit.
    #[must_use]
    pub(crate) const fn set(&self) -> u8 {
        self.set
    }

    /// The sample of a clear bit.
    #[must_use]
    pub(crate) const fn clear(&self) -> u8 {
        self.clear
    }

    /// The bits of row `y`.
    #[must_use]
    pub(crate) fn row(&self, y: usize) -> &[u64] {
        &self.bits[y * self.words..][..self.words]
    }

    /// The bits of row `y`, to be written; bits past the width must stay
    /// clear.
    pub(crate) fn row_mut(&mut self, y: usize) -> &mut [u64] {
        self.bytes = OnceLock::new();
        &mut self.bits[y * self.words..][..self.words]
    }

    /// Set the bits `from .. to` of row `y`; `to` is at most the width.
    pub(crate) fn set_run(&mut self, y: usize, from: usize, to: usize) {
        debug_assert!(to <= self.width as usize);
        if from >= to {
            return;
        }
        let row = self.row_mut(y);
        let (first, last) = (from / 64, (to - 1) / 64);
        let head = u64::MAX >> (from % 64);
        let tail = u64::MAX << (63 - (to - 1) % 64);
        if first == last {
            row[first] |= head & tail;
        } else {
            row[first] |= head;
            row[first + 1..last].fill(u64::MAX);
            row[last] |= tail;
        }
    }

    /// Give set bits the sample `set` and clear bits `clear`.
    pub(crate) fn relabel(&mut self, set: u8, clear: u8) {
        if (set, clear) != (self.set, self.clear) {
            self.set = set;
            self.clear = clear;
            self.bytes = OnceLock::new();
        }
    }

    /// The samples, one byte each, row after row: expanded on the first
    /// call; `None` when they cannot be allocated.
    #[must_use]
    pub(crate) fn samples(&self) -> Option<&[u8]> {
        self.bytes.get_or_init(|| self.expand()).as_deref()
    }

    /// The samples, one byte each ([`Self::samples`]), owned.
    pub(crate) fn into_samples(mut self) -> Option<Vec<u8>> {
        self.take_samples()
    }

    /// The samples, one byte each ([`Self::samples`]), moved out of the
    /// expansion kept by [`Self::samples`] or expanded afresh; the bits are
    /// left intact, so a `None` (the bytes cannot be allocated) loses
    /// nothing.
    pub(crate) fn take_samples(&mut self) -> Option<Vec<u8>> {
        match self.bytes.take() {
            Some(Some(bytes)) => Some(bytes),
            _ => self.expand(),
        }
    }

    /// The bytes of memory the bits and their expansion to bytes occupy:
    /// the expansion is counted whether made yet or not, as every draw
    /// through [`Self::samples`] makes and keeps it.
    #[must_use]
    pub(crate) const fn heap_bytes(&self) -> usize {
        size_of::<u64>() * self.bits.len() + self.sample_count()
    }

    /// The number of samples, `width × height`.
    #[must_use]
    pub(crate) const fn sample_count(&self) -> usize {
        self.width as usize * self.height as usize
    }

    fn expand(&self) -> Option<Vec<u8>> {
        let width = self.width as usize;
        let mut out = super::try_filled(self.sample_count(), self.clear)?;
        if width == 0 {
            return Some(out);
        }
        let flip = u64::from_ne_bytes([self.set ^ self.clear; 8]);
        let clear = u64::from_ne_bytes([self.clear; 8]);
        for (row, out) in self
            .bits
            .chunks_exact(self.words)
            .zip(out.chunks_exact_mut(width))
        {
            for (&word, out) in row.iter().zip(out.chunks_mut(64)) {
                if word == 0 {
                    continue;
                }
                for (byte, out) in word.to_be_bytes().into_iter().zip(out.chunks_mut(8)) {
                    if byte != 0 {
                        let mask = u64::from_ne_bytes(BYTE_MASKS[usize::from(byte)]);
                        let samples = (clear ^ (flip & mask)).to_ne_bytes();
                        out.copy_from_slice(&samples[..out.len()]);
                    }
                }
            }
        }
        Some(out)
    }
}

#[cfg(test)]
impl Bilevel {
    /// `rows` of `'0'`/`'1'` as a bilevel image.
    pub(crate) fn from_rows(rows: &[&str], set: u8, clear: u8) -> Self {
        let width = u32::try_from(rows[0].len()).unwrap();
        let height = u32::try_from(rows.len()).unwrap();
        let mut b = Self::new(width, height, set, clear).unwrap();
        for (y, r) in rows.iter().enumerate() {
            for (x, c) in r.bytes().enumerate() {
                if c == b'1' {
                    b.set_run(y, x, x + 1);
                }
            }
        }
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_follow_the_bits() {
        for width in [1, 7, 8, 9, 63, 64, 65, 130] {
            let rows: Vec<String> = (0..3)
                .map(|y| {
                    (0..width)
                        .map(|x| if (x * 7 + y * 3) % 5 < 2 { '1' } else { '0' })
                        .collect()
                })
                .collect();
            let rows: Vec<&str> = rows.iter().map(String::as_str).collect();
            for (set, clear) in [(0x00, 0xFF), (0xFF, 0x00), (0x30, 0xC8)] {
                let b = Bilevel::from_rows(&rows, set, clear);
                let expected: Vec<u8> = rows
                    .iter()
                    .flat_map(|r| r.bytes().map(|c| if c == b'1' { set } else { clear }))
                    .collect();
                assert_eq!(b.samples().unwrap(), expected, "width {width}");
                assert_eq!(b.into_samples().unwrap(), expected, "width {width}");
            }
        }
    }

    #[test]
    fn set_run_sets_exactly_its_bits() {
        for (from, to) in [
            (0, 1),
            (3, 64),
            (0, 64),
            (63, 65),
            (5, 129),
            (64, 130),
            (7, 7),
        ] {
            let mut b = Bilevel::new(130, 1, 0x00, 0xFF).unwrap();
            b.set_run(0, from, to);
            for x in 0..130 {
                let bit = b.row(0)[x / 64] >> (63 - x % 64) & 1;
                assert_eq!(
                    bit == 1,
                    (from..to).contains(&x),
                    "run {from}..{to}, bit {x}"
                );
            }
        }
    }

    #[test]
    fn heap_bytes_counts_the_bits_and_their_expansion() {
        let b = Bilevel::new(130, 3, 0x00, 0xFF).unwrap();
        // Three 64-bit words a row, three rows, and a byte a sample.
        assert_eq!(b.heap_bytes(), 3 * 3 * 8 + 130 * 3);
        let data = super::super::ImageData::Bilevel(Box::new(b));
        assert_eq!(data.heap_bytes(), 3 * 3 * 8 + 130 * 3);
    }

    #[test]
    fn take_samples_keeps_the_bits() {
        let mut b = Bilevel::from_rows(&["0110"], 0x00, 0xFF);
        assert_eq!(b.samples().unwrap(), [0xFF, 0x00, 0x00, 0xFF]);
        assert_eq!(b.take_samples().unwrap(), [0xFF, 0x00, 0x00, 0xFF]);
        assert_eq!(b.take_samples().unwrap(), [0xFF, 0x00, 0x00, 0xFF]);
    }

    #[test]
    fn relabel_redraws_the_samples() {
        let mut b = Bilevel::from_rows(&["0110"], 0x00, 0xFF);
        assert_eq!(b.samples().unwrap(), [0xFF, 0x00, 0x00, 0xFF]);
        b.relabel(0xFF, 0x00);
        assert_eq!(b.samples().unwrap(), [0x00, 0xFF, 0xFF, 0x00]);
    }
}
