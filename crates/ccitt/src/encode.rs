//! An encoder for CCITT fax images: Group 3 one-dimensional (MH) and
//! two-dimensional (MR) coding (ITU-T T.4) and Group 4 (MMR) coding
//! (ITU-T T.6).
//!
//! Rows are given as the positions at which their colour changes, the form
//! [`crate::Decoder::push_row`] receives them in; [`changes_of_row`]
//! derives them from packed pixels.  Bits are written most significant
//! first (`FillOrder` 1).
//!
//! The output is the coded rows alone: Group 3 output carries no
//! return-to-control sequence (six EOLs), which a container that records
//! the row count, such as TIFF, does not use; a standalone T.4 stream
//! appends it.

use alloc::vec::Vec;
use core::num::NonZeroU32;

use crate::tables::{
    BLACK_MAKEUP, BLACK_TERMINATING, COMMON_MAKEUP, WHITE_MAKEUP, WHITE_TERMINATING,
};

/// How rows are coded.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EncodeScheme {
    /// Group 3: every row is preceded by an end-of-line code (EOL), padded
    /// with zero fill bits so that the EOL ends on a byte boundary when
    /// `byte_aligned_eol` is set.
    ///
    /// With `k` set, rows are coded two-dimensionally (MR): every `k`-th row
    /// from the first one-dimensionally and the others against the row
    /// above, each EOL followed by a tag bit, 1 before a one-dimensional
    /// row.  T.4 sets `k` to 2 at standard and 4 at fine resolution.
    /// Without `k`, every row is coded one-dimensionally (MH).
    Group3 {
        /// The period of one-dimensional rows in two-dimensional coding.
        k: Option<NonZeroU32>,
        /// Whether each EOL is padded to end on a byte boundary.
        byte_aligned_eol: bool,
    },
    /// Group 4: every row coded against the row above, the first against
    /// an all-white row, and the image ended by an end-of-facsimile-block
    /// code (EOFB).  No EOLs.
    Group4,
}

/// The end-of-line code, eleven zeros and a one.
const EOL: (u32, u8) = (1, 12);

/// The vertical-mode codes `(code, length)` of `a1 − b1` from −3 to 3
/// (T.4 Table 4): VL3, VL2, VL1, V0, VR1, VR2, VR3.
const VERTICAL: [(u32, u8); 7] = [
    (0b000_0010, 7),
    (0b00_0010, 6),
    (0b010, 3),
    (0b1, 1),
    (0b011, 3),
    (0b00_0011, 6),
    (0b000_0011, 7),
];

/// The run of the make-up codes of the colour tables, 64 to 1728 in steps
/// of 64, and of the common make-up codes, 1792 to 2560: the encoder
/// indexes them by run.
const _: () = {
    let mut i = 0;
    while i < 64 {
        assert!(WHITE_TERMINATING[i].0 as usize == i && BLACK_TERMINATING[i].0 as usize == i);
        i += 1;
    }
    let mut i = 0;
    while i < 27 {
        assert!(WHITE_MAKEUP[i].0 as usize == 64 * (i + 1));
        assert!(BLACK_MAKEUP[i].0 as usize == 64 * (i + 1));
        i += 1;
    }
    let mut i = 0;
    while i < 13 {
        assert!(COMMON_MAKEUP[i].0 as usize == 64 * (i + 28));
        i += 1;
    }
};

/// A CCITT encoder for an image `width` pixels wide.
#[derive(Debug)]
pub struct Encoder {
    width: u32,
    scheme: EncodeScheme,
    /// The changes of the previous row; empty for the all-white row.
    reference: Vec<u32>,
    /// The index of the next row within the period `k` of Group 3
    /// two-dimensional coding: 0 for a one-dimensional row.
    phase: u32,
    /// The changes of the row being coded by [`Self::encode_packed_row`].
    scratch: Vec<u32>,
    out: BitWriter,
}

impl Encoder {
    /// An encoder for rows of `width` pixels coded by `scheme`.
    #[must_use]
    pub fn new(width: NonZeroU32, scheme: EncodeScheme) -> Self {
        Self {
            width: width.get(),
            scheme,
            reference: Vec::new(),
            phase: 0,
            scratch: Vec::new(),
            out: BitWriter::default(),
        }
    }

    /// Code the next row, whose colour changes are `changes`: strictly
    /// ascending positions below the width, the row white before the first,
    /// as [`crate::Decoder::push_row`] receives them.
    ///
    /// # Panics
    ///
    /// Panics if `changes` is not strictly ascending or holds a position
    /// at or beyond the width.
    pub fn encode_row(&mut self, changes: &[u32]) {
        assert!(
            changes.windows(2).all(|w| w[0] < w[1])
                && changes.last().is_none_or(|&c| c < self.width),
            "row changes must ascend strictly and lie within the width"
        );
        match self.scheme {
            EncodeScheme::Group3 {
                k,
                byte_aligned_eol,
            } => {
                if byte_aligned_eol {
                    // Fill bits so that the EOL's last bit ends a byte.
                    let used = (self.out.pending + EOL.1) % 8;
                    self.out.push(0, (8 - used) % 8);
                }
                self.out.push(EOL.0, EOL.1);
                match k {
                    None => self.code_one_d(changes),
                    Some(k) => {
                        let one_d = self.phase == 0;
                        self.out.push(u32::from(one_d), 1);
                        if one_d {
                            self.code_one_d(changes);
                        } else {
                            self.code_two_d(changes);
                        }
                        self.phase = (self.phase + 1) % k.get();
                    }
                }
            }
            EncodeScheme::Group4 => self.code_two_d(changes),
        }
        self.reference.clear();
        self.reference.extend_from_slice(changes);
    }

    /// Code the next row, packed one bit per pixel, most significant first,
    /// 1 black.
    ///
    /// # Panics
    ///
    /// Panics if `row` holds fewer bits than the width.
    pub fn encode_packed_row(&mut self, row: &[u8]) {
        let mut changes = core::mem::take(&mut self.scratch);
        changes.clear();
        push_changes(row, self.width, &mut changes);
        self.encode_row(&changes);
        self.scratch = changes;
    }

    /// The coded image: for Group 4 ended by EOFB, padded to a byte.
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        if self.scheme == EncodeScheme::Group4 {
            self.out.push(EOL.0, EOL.1);
            self.out.push(EOL.0, EOL.1);
        }
        self.out.into_bytes()
    }

    /// One-dimensional coding: the runs of the row, alternately white and
    /// black, starting with a white run that may be empty.
    fn code_one_d(&mut self, changes: &[u32]) {
        let mut start = 0;
        let mut black = false;
        for &end in changes.iter().chain(core::iter::once(&self.width)) {
            self.out.run(end - start, black);
            start = end;
            black = !black;
        }
    }

    /// Two-dimensional coding against the reference row (T.4 §4.2.1.3,
    /// T.6 §2.2): pass mode while the reference row's next change pair ends
    /// before the coding row's next change, vertical mode for a change
    /// within three pixels of the reference change, horizontal mode
    /// otherwise.
    fn code_two_d(&mut self, changes: &[u32]) {
        let width = i64::from(self.width);
        let at = |list: &[u32], i: usize| list.get(i).map_or(width, |&c| i64::from(c));
        // a0 starts on an imaginary white pixel before the row.
        let mut a0: i64 = -1;
        let mut black = false;
        // Indices of the first coding and reference changes after a0; a0
        // only advances, so both cursors do.
        let mut next = 0;
        let mut after = 0;
        loop {
            while next < changes.len() && i64::from(changes[next]) <= a0 {
                next += 1;
            }
            while after < self.reference.len() && i64::from(self.reference[after]) <= a0 {
                after += 1;
            }
            let a1 = at(changes, next);
            let a2 = at(changes, next + 1);
            // b1: the first reference change after a0 to the colour
            // opposite a0's.  Changes at even indices turn the row black.
            let mut j = after;
            if j % 2 != usize::from(black) {
                j += 1;
            }
            let b1 = at(&self.reference, j);
            let b2 = at(&self.reference, j + 1);

            if b2 < a1 {
                self.out.push(0b0001, 4);
                a0 = b2;
            } else if let Ok(offset) = usize::try_from(a1 - b1 + 3)
                && let Some(&(code, len)) = VERTICAL.get(offset)
            {
                self.out.push(code, len);
                a0 = a1;
                black = !black;
            } else {
                self.out.push(0b001, 3);
                let start = a0.max(0);
                self.out.run(run_length(a1 - start), black);
                self.out.run(run_length(a2 - a1), !black);
                a0 = a2;
            }
            if a0 >= width {
                break;
            }
        }
    }
}

/// `len`, a run within a row of at most `u32::MAX` pixels.
fn run_length(len: i64) -> u32 {
    u32::try_from(len).expect("a run lies within the row")
}

/// The colour changes of a row of `width` pixels packed one bit per pixel,
/// most significant first, 1 black: the form [`Encoder::encode_row`] takes.
///
/// # Panics
///
/// Panics if `row` holds fewer than `width` bits.
#[must_use]
pub fn changes_of_row(row: &[u8], width: u32) -> Vec<u32> {
    let mut changes = Vec::new();
    push_changes(row, width, &mut changes);
    changes
}

/// Append the colour changes of the first `width` bits of `row` to
/// `changes`.  A byte all of the current colour holds no change and is
/// skipped whole.
fn push_changes(row: &[u8], width: u32, changes: &mut Vec<u32>) {
    assert!(
        row.len() as u64 * 8 >= u64::from(width),
        "the row holds fewer bits than its width"
    );
    let mut black = false;
    let mut x = 0;
    while x < width {
        let byte = row[(x / 8) as usize];
        if x % 8 == 0 && width - x >= 8 && byte == if black { 0xFF } else { 0 } {
            x += 8;
            continue;
        }
        let bit = byte & (0x80 >> (x % 8)) != 0;
        if bit != black {
            changes.push(x);
            black = bit;
        }
        x += 1;
    }
}

/// Bits written most significant first.
#[derive(Debug, Default)]
struct BitWriter {
    bytes: Vec<u8>,
    /// Pending bits, right-aligned.
    acc: u32,
    /// The number of pending bits, below 8 between calls: the bits written
    /// so far modulo 8.
    pending: u8,
}

impl BitWriter {
    /// Append the low `n` bits of `code`, `n` at most 24.
    fn push(&mut self, code: u32, n: u8) {
        debug_assert!(n <= 24);
        if n == 0 {
            return;
        }
        self.acc = (self.acc << n) | (code & ((1 << n) - 1));
        self.pending += n;
        while self.pending >= 8 {
            self.pending -= 8;
            self.bytes.push((self.acc >> self.pending) as u8);
        }
        self.acc &= (1 << self.pending) - 1;
    }

    /// Append the code of a run of `len` pixels of one colour: make-up
    /// codes for its multiples of 64 (at most 2560 each), then the
    /// terminating code of the rest.  The tables are indexed by run (see
    /// the assertion on their layout above).
    fn run(&mut self, mut len: u32, black: bool) {
        let (terminating, makeup) = if black {
            (&BLACK_TERMINATING, &BLACK_MAKEUP)
        } else {
            (&WHITE_TERMINATING, &WHITE_MAKEUP)
        };
        while len >= 64 {
            let sixty_fours = len.min(2560) / 64;
            let (_, n, code) = match sixty_fours {
                1..=27 => makeup[sixty_fours as usize - 1],
                _ => COMMON_MAKEUP[sixty_fours as usize - 28],
            };
            self.push(u32::from(code), n);
            len -= sixty_fours * 64;
        }
        let (_, n, code) = terminating[len as usize];
        self.push(u32::from(code), n);
    }

    /// The bytes written, the last padded with zeros.
    fn into_bytes(mut self) -> Vec<u8> {
        if self.pending > 0 {
            self.bytes.push((self.acc << (8 - self.pending)) as u8);
        }
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;
    use core::num::NonZeroU32;

    use super::{EncodeScheme, Encoder, changes_of_row};
    use crate::{DecodeSettings, Decoder, DecoderContext, EncodingMode, decode};

    struct Rows(Vec<Vec<u32>>);

    impl Decoder for Rows {
        fn push_row(&mut self, changes: &[u32]) {
            self.0.push(changes.to_vec());
        }
    }

    const K2: NonZeroU32 = NonZeroU32::new(2).unwrap();
    const K4: NonZeroU32 = NonZeroU32::new(4).unwrap();

    /// `n` columns.
    fn width(n: u32) -> NonZeroU32 {
        NonZeroU32::new(n).expect("a test image has columns")
    }

    const SCHEMES: [EncodeScheme; 5] = [
        EncodeScheme::Group4,
        EncodeScheme::Group3 {
            k: None,
            byte_aligned_eol: false,
        },
        EncodeScheme::Group3 {
            k: None,
            byte_aligned_eol: true,
        },
        EncodeScheme::Group3 {
            k: Some(K2),
            byte_aligned_eol: false,
        },
        EncodeScheme::Group3 {
            k: Some(K4),
            byte_aligned_eol: true,
        },
    ];

    /// `rows` coded by `scheme` and decoded again.
    fn round_trip(width: u32, rows: &[Vec<u32>], scheme: EncodeScheme) -> Vec<Vec<u32>> {
        let mut encoder = Encoder::new(self::width(width), scheme);
        for row in rows {
            encoder.encode_row(row);
        }
        let data = encoder.finish();
        let mut settings = DecodeSettings::new(width, u32::try_from(rows.len()).unwrap());
        settings.strict = true;
        match scheme {
            EncodeScheme::Group4 => settings.encoding = EncodingMode::Group4,
            EncodeScheme::Group3 {
                k,
                byte_aligned_eol,
            } => {
                settings.end_of_line = true;
                settings.rows_are_byte_aligned = byte_aligned_eol;
                settings.encoding = if k.is_some() {
                    EncodingMode::Group3_2D
                } else {
                    EncodingMode::Group3_1D
                };
            }
        }
        let mut out = Rows(Vec::new());
        decode(&data, &mut out, &mut DecoderContext::new(settings)).expect("decodes");
        out.0
    }

    /// Rows exercising every mode: blank and full rows, single pixels at
    /// both ends, changes shifted by up to three (vertical), by more
    /// (horizontal), and reference pairs ending before the next change
    /// (pass).
    fn rows(width: u32) -> Vec<Vec<u32>> {
        let mut rows = vec![
            vec![],
            vec![0],
            vec![0, 1],
            vec![width - 1],
            vec![10, 20, 30, 40],
            vec![11, 19, 33, 40],
            vec![13, 17, 36, 37],
            vec![100, 200],
            vec![5, 6, 7, 8, 9, 10, 300, 301],
            vec![],
        ];
        let mut state = 0x1234_5678u32;
        for _ in 0..40 {
            let mut row = Vec::new();
            let mut x = 0;
            loop {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                x += 1 + (state >> 20) % 90;
                if x >= width {
                    break;
                }
                row.push(x);
            }
            rows.push(row);
        }
        rows
    }

    #[test]
    fn every_scheme_round_trips_through_the_decoder() {
        for width in [1, 7, 64, 1728, 2600] {
            let rows: Vec<Vec<u32>> = if width < 400 {
                vec![vec![], vec![0], vec![], vec![0]]
                    .into_iter()
                    .chain((width > 3).then(|| vec![1, 2]))
                    .collect()
            } else {
                rows(width)
            };
            for scheme in SCHEMES {
                assert_eq!(
                    round_trip(width, &rows, scheme),
                    rows,
                    "{width} px, {scheme:?}"
                );
            }
        }
    }

    #[test]
    fn runs_beyond_the_largest_make_up_code_split() {
        let width = 9000;
        let rows = vec![vec![], vec![6000], vec![1, 8999], vec![2559, 2560, 5121]];
        for scheme in SCHEMES {
            assert_eq!(round_trip(width, &rows, scheme), rows, "{scheme:?}");
        }
    }

    #[test]
    fn eols_end_on_byte_boundaries_when_aligned() {
        let mut encoder = Encoder::new(
            width(100),
            EncodeScheme::Group3 {
                k: None,
                byte_aligned_eol: true,
            },
        );
        encoder.encode_row(&[3, 50]);
        encoder.encode_row(&[]);
        let data = encoder.finish();
        // The first EOL fills its byte pair: 4 zero bits, then 0000 0000 0001.
        assert_eq!(&data[..2], &[0x00, 0x01]);
    }

    #[test]
    fn changes_of_a_packed_row() {
        assert_eq!(changes_of_row(&[0b0011_1000, 0b0000_0001], 16), [2, 5, 15]);
        assert_eq!(changes_of_row(&[0xFF], 5), [0]);
        assert!(changes_of_row(&[0x00], 8).is_empty());
        // Whole bytes of the current colour are skipped, including across
        // a change at a byte boundary and in a final partial byte.
        assert_eq!(changes_of_row(&[0x00, 0xFF, 0xFF, 0x0F], 32), [8, 24, 28]);
        assert_eq!(changes_of_row(&[0xFF, 0xFF, 0xFE], 23), [0]);
        assert_eq!(changes_of_row(&[0xFF, 0xFF, 0xFE], 24), [0, 23]);
    }

    #[test]
    fn packed_rows_code_as_their_changes() {
        // A deterministic mix of whole-byte runs and scattered bits.
        let width: u32 = 203;
        let rows: Vec<Vec<u8>> = (0u32..40)
            .map(|y| {
                (0..width.div_ceil(8))
                    .map(|i| match (i + y) % 5 {
                        0 => 0x00,
                        1 => 0xFF,
                        _ => (i * 37 + y * 11).to_le_bytes()[0],
                    })
                    .collect()
            })
            .collect();
        for scheme in SCHEMES {
            let mut by_changes = Encoder::new(self::width(width), scheme);
            let mut packed = Encoder::new(self::width(width), scheme);
            for row in &rows {
                by_changes.encode_row(&changes_of_row(row, width));
                packed.encode_packed_row(row);
            }
            assert_eq!(packed.finish(), by_changes.finish(), "{scheme:?}");
        }
    }

    /// `rows` of `columns` pixels coded by `scheme`.
    fn coded(columns: u32, rows: &[&[u32]], scheme: EncodeScheme) -> Vec<u8> {
        let mut encoder = Encoder::new(width(columns), scheme);
        for row in rows {
            encoder.encode_row(row);
        }
        encoder.finish()
    }

    /// Bit strings written out from the code tables of T.4 (Tables 1–4)
    /// and T.6, independent of `tables.rs`, which encoder and decoder share.
    #[test]
    fn output_matches_the_recommendations_code_tables() {
        let mh = EncodeScheme::Group3 {
            k: None,
            byte_aligned_eol: false,
        };
        // EOL 000000000001, white 3 = 1000, black 2 = 11, white 15 = 110101.
        assert_eq!(coded(20, &[&[3, 5]], mh), [0x00, 0x18, 0xF5]);
        // EOL, white make-up 1728 = 010011011, white 0 = 00110101.
        assert_eq!(coded(1728, &[&[]], mh), [0x00, 0x14, 0xD9, 0xA8]);
        // MR, k = 2: EOL, tag 1, the row one-dimensionally; EOL, tag 0, the
        // same row as three V0 = 1.
        let mr = EncodeScheme::Group3 {
            k: Some(K2),
            byte_aligned_eol: false,
        };
        assert_eq!(
            coded(20, &[&[3, 5], &[3, 5]], mr),
            [0x00, 0x1C, 0x7A, 0x80, 0x0B, 0x80]
        );
        // MMR against the white row: horizontal 001, white 3, black 2; V0;
        // EOFB, two EOLs.
        assert_eq!(
            coded(20, &[&[3, 5]], EncodeScheme::Group4),
            [0x31, 0xC0, 0x04, 0x00, 0x40]
        );
    }

    #[test]
    fn the_one_dimensional_period_restarts_every_k_rows() {
        let mr = EncodeScheme::Group3 {
            k: Some(K2),
            byte_aligned_eol: true,
        };
        // Blank rows of 8: each EOL padded to end a byte, then tag 1 and
        // white 8 = 10011 on rows 0 and 2, tag 0 and V0 = 1 on rows 1 and 3.
        assert_eq!(
            coded(8, &[&[], &[], &[], &[]], mr),
            [
                0x00, 0x01, 0xCC, 0x00, 0x01, 0x40, 0x01, 0xCC, 0x00, 0x01, 0x40
            ]
        );
    }
}
