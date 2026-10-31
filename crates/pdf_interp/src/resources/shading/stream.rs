//! The packed vertex data of mesh shadings (types 4–7, PDF §8.7.4.5.5–8).
//!
//! A mesh stream packs each vertex as a coordinate pair of
//! `BitsPerCoordinate` bits per value followed by `BitsPerComponent` bits per
//! colour field, most significant bit first; the patch and free-form types
//! prefix each record with a `BitsPerFlag`-bit flag.  Raw values map
//! linearly from `[0, 2^bits − 1]` onto their `Decode` interval, and the
//! coordinates then through the CTM to device space.

use pdf::Dictionary;

use super::mesh::{CornerSource, MeshVertex};
use super::transform_point;
use crate::resources::dict_ext::DictExt;
use crate::resources::obj_to_f64;

/// Bit-stream reader for packed mesh vertex data.
///
/// Values are read most significant bit first; a record realigns to a byte
/// boundary with [`Self::align`].
pub(super) struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    /// Bits already fetched from `data` but not yet consumed.
    /// Right-justified (LSB-aligned): live bits occupy `bit_buf[bits_in_buf-1..0]`.
    /// Extraction uses `(bit_buf >> bits_in_buf) & mask` to pull from the top of
    /// the live region.  64-bit so `n=32` never overflows the accumulator.
    bit_buf: u64,
    bits_in_buf: u8,
}

impl<'a> BitReader<'a> {
    /// A reader at the first bit of `data`.
    pub(super) const fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_buf: 0,
            bits_in_buf: 0,
        }
    }

    /// Skip to the next byte boundary, discarding the unread bits of the
    /// current byte.  Every byte loaded into the buffer is whole, so the
    /// partial byte's unread bits are `bits_in_buf mod 8`.
    pub(super) const fn align(&mut self) {
        self.bits_in_buf -= self.bits_in_buf % 8;
    }

    /// Read exactly `n` bits (1–32) as a right-justified `u32`; `None` when
    /// the stream ends first.  `n` outside `1..=32` reads nothing and
    /// yields 0 (callers pass validated field widths only).
    pub(super) fn read_bits(&mut self, n: u8) -> Option<u32> {
        debug_assert!(
            (1..=32).contains(&n),
            "read_bits: n={n} out of range 1..=32"
        );
        if n == 0 || n > 32 {
            return Some(0);
        }
        while self.bits_in_buf < n {
            let &byte = self.data.get(self.byte_pos)?;
            // At most 31 live bits remain before a byte is appended, so the
            // buffer never holds more than 39 bits.
            self.bit_buf = (self.bit_buf << 8) | u64::from(byte);
            self.byte_pos += 1;
            self.bits_in_buf += 8;
        }
        self.bits_in_buf -= n;
        let mask = (1u64 << n) - 1;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "mask is at most u32::MAX; the extracted value fits u32"
        )]
        Some(((self.bit_buf >> self.bits_in_buf) & mask) as u32)
    }
}

/// Legal `BitsPerCoordinate` and `BitsPerComponent` values (Tables 83–86).
const VALID_BITS: &[u8] = &[1, 2, 4, 8, 12, 16, 24, 32];

/// Legal `BitsPerFlag` values (Tables 83, 85, 86).
const VALID_FLAG_BITS: &[u8] = &[2, 4, 8];

/// The integer under `key` when it is one of `valid`; `None`, with a
/// warning naming mesh type `shading_type`, otherwise.
fn parse_bits_field(sh: &Dictionary, key: &str, valid: &[u8], shading_type: u8) -> Option<u8> {
    let Some(v) = sh.get_i64(key.as_bytes()) else {
        log::warn!("shading/type{shading_type}: {key} is missing — skipping the shading");
        return None;
    };
    let bits = u8::try_from(v).ok().filter(|b| valid.contains(b));
    if bits.is_none() {
        log::warn!(
            "shading/type{shading_type}: {key} {v} is not one of {valid:?} — skipping the shading"
        );
    }
    bits
}

/// `BitsPerFlag` of mesh type `shading_type`; `None`, logged, when absent
/// or not a legal value.
pub(super) fn parse_bits_per_flag(sh: &Dictionary, shading_type: u8) -> Option<u8> {
    parse_bits_field(sh, "BitsPerFlag", VALID_FLAG_BITS, shading_type)
}

/// The vertex packing of one mesh stream and its mapping to device space.
pub(super) struct VertexFormat<'a> {
    bits_per_coord: u8,
    bits_per_comp: u8,
    /// `[xmin xmax ymin ymax c0min c0max …]`, as [`read_decode_array`]
    /// returns it.
    decode: &'a [f64],
    ctm: &'a [f64; 6],
    page_h: f64,
}

impl<'a> VertexFormat<'a> {
    /// The format of mesh type `shading_type` with dictionary `sh`, decode
    /// intervals `decode` and target-space CTM `ctm`.  `None`, logged, when
    /// `BitsPerCoordinate` or `BitsPerComponent` is absent or not legal.
    pub(super) fn parse(
        sh: &Dictionary,
        shading_type: u8,
        decode: &'a [f64],
        ctm: &'a [f64; 6],
        page_h: f64,
    ) -> Option<Self> {
        Some(Self {
            bits_per_coord: parse_bits_field(sh, "BitsPerCoordinate", VALID_BITS, shading_type)?,
            bits_per_comp: parse_bits_field(sh, "BitsPerComponent", VALID_BITS, shading_type)?,
            decode,
            ctm,
            page_h,
        })
    }

    /// The `Decode` interval of slot `i` (0 = x, 1 = y, then one per
    /// colour field), `[0, 1]` when the array lacks it.
    fn interval(&self, i: usize) -> (f64, f64) {
        match self.decode.get(2 * i..2 * i + 2) {
            Some(&[lo, hi]) => (lo, hi),
            _ => (0.0, 1.0),
        }
    }

    /// Read one coordinate pair, mapped to device space.
    pub(super) fn read_point(&self, reader: &mut BitReader<'_>) -> Option<[f64; 2]> {
        let raw_x = reader.read_bits(self.bits_per_coord)?;
        let raw_y = reader.read_bits(self.bits_per_coord)?;
        let max = max_value(self.bits_per_coord);
        let map = |raw: u32, (lo, hi): (f64, f64)| (f64::from(raw) / max).mul_add(hi - lo, lo);
        Some(
            transform_point(
                self.ctm,
                map(raw_x, self.interval(0)),
                map(raw_y, self.interval(1)),
                self.page_h,
            )
            .into(),
        )
    }

    /// Read one colour of `source.components()` fields as `source`'s corner
    /// value.  `fields` is scratch space of that length.
    pub(super) fn read_corner<S: CornerSource>(
        &self,
        reader: &mut BitReader<'_>,
        source: &S,
        fields: &mut [f64],
    ) -> Option<S::Corner> {
        let max = max_value(self.bits_per_comp);
        for field in fields.iter_mut() {
            *field = f64::from(reader.read_bits(self.bits_per_comp)?) / max;
        }
        let components: Vec<f64> = fields
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let (lo, hi) = self.interval(2 + i);
                v.mul_add(hi - lo, lo)
            })
            .collect();
        Some(source.corner(&components, fields))
    }

    /// Read one vertex: a coordinate pair, then a colour.
    pub(super) fn read_vertex<S: CornerSource>(
        &self,
        reader: &mut BitReader<'_>,
        source: &S,
        fields: &mut [f64],
    ) -> Option<MeshVertex<S::Corner>> {
        let [x, y] = self.read_point(reader)?;
        let corner = self.read_corner(reader, source, fields)?;
        Some(MeshVertex { x, y, corner })
    }
}

/// `2^bits − 1` for a legal field width.
const fn max_value(bits: u8) -> f64 {
    #[expect(clippy::cast_precision_loss, reason = "at most 2^32 − 1, exact in f64")]
    let max = ((1u64 << bits) - 1) as f64;
    max
}

/// The `Decode` array of a mesh shading with `n_channels` colour fields:
/// `[xmin xmax ymin ymax c0min c0max …]` (PDF Table 83).  An absent array,
/// or one of the wrong length, is logged and replaced by `[0, 1]` for each
/// slot.
pub(super) fn read_decode_array(sh: &Dictionary, n_channels: usize) -> Vec<f64> {
    let expected = 4 + n_channels * 2;
    let parsed: Option<Vec<f64>> = sh
        .get(b"Decode")
        .and_then(pdf::Object::as_array)
        .map(|arr| arr.iter().filter_map(obj_to_f64).collect());
    match parsed {
        Some(v) if v.len() == expected => v,
        Some(v) => {
            log::warn!(
                "shading: Decode array has {} numbers, expected {expected} — using [0,1] defaults",
                v.len()
            );
            default_decode(n_channels)
        }
        None => {
            log::warn!(
                "shading: Decode array missing (required for mesh shadings) — using [0,1] defaults"
            );
            default_decode(n_channels)
        }
    }
}

/// `[0, 1]` for x, for y and for each of `n_channels` colour fields.
pub(super) fn default_decode(n_channels: usize) -> Vec<f64> {
    [0.0, 1.0].repeat(2 + n_channels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_reader_align_skips_to_the_next_byte() {
        let mut r = BitReader::new(&[0b1010_0000, 0xC3, 0x5A]);
        assert_eq!(r.read_bits(3), Some(0b101));
        r.align();
        assert_eq!(r.read_bits(8), Some(0xC3));
        r.align(); // already aligned: no bits skipped
        assert_eq!(r.read_bits(4), Some(0x5));
        r.align();
        assert_eq!(r.read_bits(1), None);
    }

    #[test]
    fn bit_reader_reads_32_bit_fields_across_a_partial_byte() {
        let mut r = BitReader::new(&[0b1000_0000, 0xFF, 0xFF, 0xFF, 0xFE]);
        assert_eq!(r.read_bits(1), Some(1));
        assert_eq!(r.read_bits(32), Some(0x01FF_FFFF));
        assert_eq!(r.read_bits(8), None);
    }

    #[test]
    fn default_decode_has_one_unit_interval_per_slot() {
        assert_eq!(default_decode(1), [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn illegal_field_widths_are_rejected() {
        let mut sh = Dictionary::new();
        sh.set("BitsPerCoordinate", pdf::Object::Integer(8));
        sh.set("BitsPerComponent", pdf::Object::Integer(3));
        let ctm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        assert!(VertexFormat::parse(&sh, 4, &[], &ctm, 0.0).is_none());
        sh.set("BitsPerComponent", pdf::Object::Integer(16));
        assert!(VertexFormat::parse(&sh, 4, &[], &ctm, 0.0).is_some());
        assert_eq!(parse_bits_per_flag(&sh, 4), None);
        sh.set("BitsPerFlag", pdf::Object::Integer(8));
        assert_eq!(parse_bits_per_flag(&sh, 4), Some(8));
    }
}
