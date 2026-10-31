//! `CCITTFaxDecode` (PDF §7.4.6): Group 3 and Group 4 fax streams.

use pdf::Object;

use crate::resources::dict_ext::DictExt;
use crate::resources::image::colorspace::{Layout, Samples};

// ── CCITTFaxDecode ─────────────────────────────────────────────────────────────

/// Decode a `CCITTFaxDecode` stream (PDF §7.4.6).
///
/// `DecodeParms` keys honoured:
/// - `K`: `< 0` Group 4 (T.6), `0` Group 3 1-D (T.4), `> 0` Group 3 mixed
///   1-D/2-D (T.4 MR).
/// - `Columns`: pixels per encoded row; defaults to `width`.  Rows are
///   clamped or white-padded to `width` in the output.
/// - `Rows`: rows to decode; defaults to `height`, and is capped at it.
/// - `EncodedByteAlign`: each encoded row starts on a byte boundary.  Group 3
///   streams honour it.  Group 4 streams that declare it occur both aligned
///   and unaligned, so a Group 4 stream is read unaligned and, if that stops
///   short, re-read aligned, keeping the reading that decodes more rows.
/// - `EndOfLine`: EOL codes are required.  They are accepted either way.
/// - `EndOfBlock`: the stream may end with EOFB / RTC (default `true`).
/// - `BlackIs1`: 1 bits are black, so output samples are inverted.
///
/// The output is exactly `width × height` bytes, `0x00` black and `0xFF`
/// white (swapped under `BlackIs1`).  A stream that ends early or fails
/// part-way keeps its decoded rows and pads the rest white; one that yields
/// no row at all returns `None`.
pub(in crate::resources::image) fn decode_ccitt(
    data: &[u8],
    width: u32,
    height: u32,
    is_mask: bool,
    parms: Option<&Object>,
) -> Option<Samples> {
    use hayro_ccitt::{DecodeSettings, DecoderContext, EncodingMode};

    let parms_dict = parms.and_then(|o| o.as_dict());
    let flag =
        |key: &[u8], default: bool| parms_dict.and_then(|d| d.get_bool(key)).unwrap_or(default);
    let positive = |key: &[u8]| {
        parms_dict
            .and_then(|d| d.get_i64(key))
            .and_then(|v| u32::try_from(v).ok())
            .filter(|&v| v > 0)
    };

    let k = parms_dict.and_then(|d| d.get_i64(b"K")).unwrap_or(0);
    let encoding = match k.cmp(&0) {
        std::cmp::Ordering::Less => EncodingMode::Group4,
        std::cmp::Ordering::Equal => EncodingMode::Group3_1D,
        std::cmp::Ordering::Greater => EncodingMode::Group3_2D,
    };
    let black_is_1 = flag(b"BlackIs1", false);
    let declared_aligned = flag(b"EncodedByteAlign", false);
    // Rows past the image's height are discarded, so the decoder stops
    // there; a larger `/Rows` would otherwise let a stream of one-bit rows
    // grow the output without bound.
    let rows = positive(b"Rows").map_or(height, |r| r.min(height));
    let columns = positive(b"Columns").unwrap_or(width);

    let capacity = (width as usize).checked_mul(height as usize)?;
    // White in the output convention: 0xFF, or 0x00 when 1 bits are black.
    let white = if black_is_1 { 0x00 } else { 0xFF };
    let run = |aligned: bool| {
        let mut settings = DecodeSettings::new(columns, rows);
        settings.end_of_block = flag(b"EndOfBlock", true);
        settings.end_of_line = flag(b"EndOfLine", false);
        settings.rows_are_byte_aligned = aligned;
        settings.encoding = encoding;
        let mut collector = HayroCcittCollector::new(capacity, width, columns, white)?;
        let mut ctx = DecoderContext::new(settings);
        let result = hayro_ccitt::decode(data, &mut collector, &mut ctx);
        Some((collector, result))
    };

    // Group 4 streams that declare `EncodedByteAlign` exist both aligned and
    // unaligned.  The unaligned reading is tried first; when it stops short,
    // the aligned reading is tried and whichever decodes more rows is kept.
    let (mut collector, mut result) = run(k >= 0 && declared_aligned)?;
    if k < 0 && declared_aligned && result.is_err() && collector.rows_decoded() < rows {
        let (aligned, aligned_result) = run(true)?;
        if aligned.rows_decoded() > collector.rows_decoded() {
            collector = aligned;
            result = aligned_result;
        }
    }

    if let Err(e) = result {
        if collector.rows_decoded() == 0 {
            log::warn!("image: CCITTFaxDecode ({encoding:?}) failed: {e}");
            return None;
        }
        log::warn!(
            "image: CCITTFaxDecode ({encoding:?}) error after {}/{height} rows: {e}",
            collector.rows_decoded()
        );
    }

    let rows_decoded = collector.rows_decoded();
    let mut data_out = collector.finish();
    if data_out.len() > capacity {
        log::debug!(
            "image: CCITTFaxDecode: output too long ({} > {capacity}), truncating",
            data_out.len()
        );
        data_out.truncate(capacity);
    } else if data_out.len() < capacity {
        log::debug!("image: CCITTFaxDecode: got {rows_decoded}/{height} rows, padding remainder");
        data_out.resize(capacity, white);
    }

    let layout = if is_mask {
        Layout::Mask
    } else {
        Layout::Color(1)
    };
    let mut samples = Samples::cpu(width, height, layout, data_out);
    if rows_decoded < height {
        samples.partial = Some(format!(
            "CCITTFaxDecode: {rows_decoded} of {height} rows decoded; the rest are white"
        ));
    }
    Some(samples)
}

/// [`hayro_ccitt::Decoder`] that expands each row of colour changes into one
/// byte per pixel: `white` for white runs and its complement for black runs.
///
/// Every row is exactly `width` bytes.  Decoded rows span `columns` pixels:
/// a narrower row is padded with `white`, a wider one is clipped.  Rows
/// past the reserved capacity are counted but not stored, so the output
/// never exceeds it.
struct HayroCcittCollector {
    out: Vec<u8>,
    width: u32,
    /// The pixel at which an unterminated black run ends: `min(columns, width)`.
    row_end: u32,
    white: u8,
    rows: u32,
}

impl HayroCcittCollector {
    /// `None` when `capacity` bytes cannot be allocated.
    fn new(capacity: usize, width: u32, columns: u32, white: u8) -> Option<Self> {
        Some(Self {
            out: crate::resources::image::try_with_capacity(capacity)?,
            width,
            row_end: columns.min(width),
            white,
            rows: 0,
        })
    }

    const fn rows_decoded(&self) -> u32 {
        self.rows
    }

    fn finish(self) -> Vec<u8> {
        self.out
    }
}

impl hayro_ccitt::Decoder for HayroCcittCollector {
    fn push_row(&mut self, changes: &[u32]) {
        self.rows += 1;
        let start = self.out.len();
        if self.out.capacity() - start < self.width as usize {
            return;
        }
        self.out.resize(start + self.width as usize, self.white);
        let row = &mut self.out[start..];
        // Changes alternate white→black and black→white; an odd count leaves
        // the row black to its end.
        for run in changes.chunks(2) {
            let from = run[0].min(self.row_end) as usize;
            let to = run.get(1).map_or(self.row_end, |&x| x.min(self.row_end)) as usize;
            row[from..to].fill(!self.white);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pdf::Dictionary;

    /// Pack a string of `'0'`/`'1'` into bytes, MSB first, zero-padded.
    fn pack_bits(bits: &str) -> Vec<u8> {
        let bits: Vec<u8> = bits.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
        bits.chunks(8)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u8, |acc, (i, &b)| acc | (u8::from(b == b'1') << (7 - i)))
            })
            .collect()
    }

    /// `DecodeParms` with the given integer and boolean entries.
    fn ccitt_parms(ints: &[(&[u8], i64)], flags: &[(&[u8], bool)]) -> Object {
        let mut d = Dictionary::new();
        for &(k, v) in ints {
            d.set(k, Object::Integer(v));
        }
        for &(k, v) in flags {
            d.set(k, Object::Boolean(v));
        }
        Object::Dictionary(d)
    }

    fn ccitt_bytes(stream: &str, w: u32, h: u32, parms: &Object) -> Vec<u8> {
        let img = decode_ccitt(&pack_bits(stream), w, h, false, Some(parms)).expect("decodes");
        img.data.as_cpu().expect("host-resident").to_vec()
    }

    /// Two rows of `white 2, black 4, white 2` (8 px).
    const TWO_ROWS: [u8; 16] = [
        0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, //
        0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF,
    ];
    /// T.4 one-dimensional code for one `white 2, black 4, white 2` row.
    const ROW_1D: &str = "0111 011 0111";
    const EOL: &str = "000000000001";

    #[test]
    fn ccitt_group4_decodes_horizontal_and_vertical_modes() {
        // Row 0 against the imaginary white line: H(white 2, black 4), V0.
        // Row 1 against row 0: V0 three times.  Then EOFB.
        let stream = format!("001 0111 011 1   1 1 1   {EOL} {EOL}");
        let parms = ccitt_parms(&[(b"K", -1)], &[]);
        assert_eq!(ccitt_bytes(&stream, 8, 2, &parms), TWO_ROWS);
    }

    #[test]
    fn ccitt_group4_ignores_encoded_byte_align() {
        // The unaligned stream above, declared byte-aligned.
        let stream = format!("001 0111 011 1   1 1 1   {EOL} {EOL}");
        let parms = ccitt_parms(&[(b"K", -1)], &[(b"EncodedByteAlign", true)]);
        assert_eq!(ccitt_bytes(&stream, 8, 2, &parms), TWO_ROWS);
    }

    #[test]
    fn ccitt_group4_honours_encoded_byte_align_when_rows_are_aligned() {
        // The rows above, each padded to a byte boundary: row 0 is 11 bits
        // (+5 zero bits), row 1 is 3 bits (+5).  Read unaligned, the padding
        // is decoded as codes and the stream fails after row 0.
        let stream = format!("001 0111 011 1 00000   1 1 1 00000   {EOL} {EOL}");
        let parms = ccitt_parms(&[(b"K", -1)], &[(b"EncodedByteAlign", true)]);
        assert_eq!(ccitt_bytes(&stream, 8, 2, &parms), TWO_ROWS);
    }

    #[test]
    fn ccitt_group3_2d_decodes_one_dimensional_rows() {
        // K = 1: each row is EOL + tag bit 1 (1-D coded), then RTC.
        let stream = format!(
            "{EOL} 1 {ROW_1D} {EOL} 1 {ROW_1D} {}",
            format!("{EOL} 1 ").repeat(6)
        );
        let parms = ccitt_parms(&[(b"K", 1)], &[]);
        assert_eq!(ccitt_bytes(&stream, 8, 2, &parms), TWO_ROWS);
    }

    #[test]
    fn ccitt_encoded_byte_align_starts_each_row_on_a_byte() {
        // Group 3 1-D without EOLs: each 11-bit row is zero-filled to 16 bits.
        let stream = format!("{ROW_1D} 00000 {ROW_1D} 00000");
        let parms = ccitt_parms(&[(b"K", 0)], &[(b"EncodedByteAlign", true)]);
        assert_eq!(ccitt_bytes(&stream, 8, 2, &parms), TWO_ROWS);

        // Read unaligned, the fill bits are decoded as codes.
        let unaligned = ccitt_parms(&[(b"K", 0)], &[]);
        let misread = decode_ccitt(&pack_bits(&stream), 8, 2, false, Some(&unaligned))
            .and_then(|img| img.data.as_cpu().map(<[u8]>::to_vec));
        assert_ne!(misread.as_deref(), Some(&TWO_ROWS[..]));
    }

    #[test]
    fn ccitt_rows_narrower_than_the_image_are_padded_white() {
        // `Columns 6`: rows of white 2, black 4, padded to the 8-px width.
        let row = "0111 011";
        let stream = format!("{EOL} {row} {EOL} {row} {}", EOL.repeat(6));
        let parms = ccitt_parms(&[(b"K", 0), (b"Columns", 6)], &[]);
        assert_eq!(ccitt_bytes(&stream, 8, 2, &parms), TWO_ROWS);
    }

    #[test]
    fn ccitt_black_is_1_inverts_samples_and_padding() {
        // One decoded row of two; the missing row pads with white, which is
        // 0x00 once 1 bits mean black.
        let stream = format!("{EOL} {ROW_1D} {}", EOL.repeat(6));
        let parms = ccitt_parms(&[(b"K", 0)], &[(b"BlackIs1", true)]);
        let got = ccitt_bytes(&stream, 8, 2, &parms);
        assert_eq!(got[..8], [0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00]);
        assert_eq!(got[8..], [0x00; 8]);
    }

    #[test]
    fn ccitt_collector_expands_changes_and_clamps_to_the_width() {
        let mut c = HayroCcittCollector::new(12, 4, 9, 0xFF).expect("collector");
        // Black from 0 to 3.
        hayro_ccitt::Decoder::push_row(&mut c, &[0, 3]);
        // Black from 2 to a change past the width (`Columns` > width).
        hayro_ccitt::Decoder::push_row(&mut c, &[2, 9]);
        // An odd count: black from 1 to the end of the row.
        hayro_ccitt::Decoder::push_row(&mut c, &[1]);
        assert_eq!(c.rows_decoded(), 3);
        assert_eq!(
            c.finish(),
            [
                0x00, 0x00, 0x00, 0xFF, //
                0xFF, 0xFF, 0x00, 0x00, //
                0xFF, 0x00, 0x00, 0x00,
            ]
        );
    }

    /// Rows past the reserved capacity are not stored.
    #[test]
    fn ccitt_collector_stops_at_its_capacity() {
        let mut c = HayroCcittCollector::new(8, 4, 4, 0xFF).expect("collector");
        for _ in 0..5 {
            hayro_ccitt::Decoder::push_row(&mut c, &[0, 1]);
        }
        assert_eq!(c.finish().len(), 8);
    }

    /// A `/Rows` larger than the image decodes only the image's rows: one
    /// Group 4 all-white row per `1` bit, 64 rows in the stream, 2 kept.
    #[test]
    fn ccitt_rows_beyond_the_height_are_not_decoded() {
        let parms = ccitt_parms(&[(b"K", -1), (b"Rows", 1 << 30)], &[]);
        let img = decode_ccitt(&[0xFF; 8], 8, 2, false, Some(&parms)).expect("decodes");
        assert_eq!(img.data.as_cpu().map(<[u8]>::len), Some(16));
        assert!(img.partial.is_none());
    }
}
