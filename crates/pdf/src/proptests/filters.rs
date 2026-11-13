//! The stream filters invert reference encoders.
//!
//! Each general-purpose filter of ISO 32000-2 §7.4 other than Brotli has an
//! encoder here (Flate and LZW through the same crates the decoders use, the
//! others written from the specification), as do the PNG and TIFF
//! predictors.  A chain of up to three filters, each Flate or LZW stage with
//! a random predictor, must decode the encoded data back to the original
//! without reporting a truncation.

use std::io::Write as _;

use proptest::prelude::*;

use super::config;
use crate::dictionary::Dictionary;
use crate::object::Object;
use crate::stream::{MAX_DECOMPRESSED, decode_run_length, decode_stream};

/// A predictor and the parameters it is applied with.
#[derive(Debug, Clone, Copy)]
struct Predictor {
    /// 2 (TIFF) or 10..=15 (PNG, every row tagged with its own filter).
    kind: i64,
    colors: usize,
    bits: usize,
    columns: usize,
}

impl Predictor {
    /// Bytes of a row of samples, without the PNG tag byte.
    const fn stride(self) -> usize {
        (self.columns * self.colors * self.bits).div_ceil(8)
    }

    fn parms(self, early_change: Option<i64>) -> Object {
        let mut d = Dictionary::new();
        d.set("Predictor", Object::Integer(self.kind));
        for (key, v) in [
            ("Colors", self.colors),
            ("BitsPerComponent", self.bits),
            ("Columns", self.columns),
        ] {
            d.set(key, Object::Integer(i64::try_from(v).expect("small")));
        }
        if let Some(e) = early_change {
            d.set("EarlyChange", Object::Integer(e));
        }
        Object::Dictionary(d)
    }

    /// `data`, whole rows of [`Self::stride`] bytes, predicted; `tags`
    /// chooses each PNG row's filter.
    fn encode(self, data: &[u8], tags: &[u8]) -> Vec<u8> {
        let stride = self.stride();
        if self.kind == 2 {
            let mut out = data.to_vec();
            for row in out.chunks_exact_mut(stride) {
                tiff_difference(row, self.colors, self.bits, self.columns);
            }
            return out;
        }
        let bpp = (self.colors * self.bits).div_ceil(8);
        let mut out = Vec::with_capacity(data.len() + data.len() / stride.max(1));
        let zero = vec![0u8; stride];
        for (i, row) in data.chunks_exact(stride).enumerate() {
            let prev = if i == 0 {
                &zero[..]
            } else {
                &data[(i - 1) * stride..i * stride]
            };
            let tag = tags.get(i % tags.len().max(1)).map_or(0, |t| t % 5);
            out.push(tag);
            for x in 0..stride {
                let a = if x >= bpp { row[x - bpp] } else { 0 };
                let b = prev[x];
                let c = if x >= bpp { prev[x - bpp] } else { 0 };
                let predicted = match tag {
                    0 => 0,
                    1 => a,
                    2 => b,
                    3 => a.midpoint(b),
                    _ => paeth(a, b, c),
                };
                out.push(row[x].wrapping_sub(predicted));
            }
        }
        out
    }
}

/// The Paeth predictor of RFC 2083 §6.6, written from its definition.
fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let p = i16::from(a) + i16::from(b) - i16::from(c);
    let (pa, pb, pc) = (
        (p - i16::from(a)).abs(),
        (p - i16::from(b)).abs(),
        (p - i16::from(c)).abs(),
    );
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// TIFF predictor 2 on one row: each sample replaced by its difference from
/// the same component one pixel left, modulo `2^bits`, from the right so
/// every difference is taken against an original sample.  Bits past the
/// last sample are kept.
fn tiff_difference(row: &mut [u8], colors: usize, bits: usize, columns: usize) {
    let samples = columns * colors;
    let get = |row: &[u8], i: usize| -> u32 {
        match bits {
            8 => u32::from(row[i]),
            16 => u32::from(u16::from_be_bytes([row[2 * i], row[2 * i + 1]])),
            _ => {
                let bit = i * bits;
                u32::from(row[bit / 8] >> (8 - bits - bit % 8)) & ((1 << bits) - 1)
            }
        }
    };
    let set = |row: &mut [u8], i: usize, v: u32| match bits {
        8 => row[i] = v.to_be_bytes()[3],
        16 => row[2 * i..2 * i + 2].copy_from_slice(&v.to_be_bytes()[2..]),
        _ => {
            let bit = i * bits;
            let shift = 8 - bits - bit % 8;
            let mask = u8::try_from((1u32 << bits) - 1).expect("bits < 8") << shift;
            let v = u8::try_from(v & ((1 << bits) - 1)).expect("masked");
            row[bit / 8] = (row[bit / 8] & !mask) | (v << shift);
        }
    };
    let modulus = 1u32 << bits;
    for i in (colors..samples).rev() {
        let d = (get(row, i) + modulus - get(row, i - colors)) % modulus;
        set(row, i, d);
    }
}

/// One stage of a filter chain.
#[derive(Debug, Clone)]
enum Filter {
    Flate(Option<Predictor>),
    Lzw {
        early_change: bool,
        predictor: Option<Predictor>,
    },
    AsciiHex,
    Ascii85,
    RunLength,
}

impl Filter {
    fn name(&self) -> &'static str {
        match self {
            Self::Flate(_) => "FlateDecode",
            Self::Lzw { .. } => "LZWDecode",
            Self::AsciiHex => "ASCIIHexDecode",
            Self::Ascii85 => "ASCII85Decode",
            Self::RunLength => "RunLengthDecode",
        }
    }

    fn parms(&self) -> Object {
        match self {
            Self::Flate(Some(p)) => p.parms(None),
            Self::Lzw {
                early_change,
                predictor,
            } => {
                let early = i64::from(*early_change);
                predictor.map_or_else(
                    || {
                        let mut d = Dictionary::new();
                        d.set("EarlyChange", Object::Integer(early));
                        Object::Dictionary(d)
                    },
                    |p| p.parms(Some(early)),
                )
            }
            _ => Object::Null,
        }
    }

    const fn predictor(&self) -> Option<Predictor> {
        match self {
            Self::Flate(p) | Self::Lzw { predictor: p, .. } => *p,
            _ => None,
        }
    }

    /// `data` encoded; `style` picks among equivalent encodings.
    fn encode(&self, data: &[u8], style: &[u8]) -> Vec<u8> {
        let data = self
            .predictor()
            .map_or_else(|| data.to_vec(), |p| p.encode(data, style));
        match self {
            Self::Flate(_) => {
                let mut z =
                    flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
                z.write_all(&data).expect("writing to a Vec");
                z.finish().expect("writing to a Vec")
            }
            Self::Lzw { early_change, .. } => {
                let mut encoder = if *early_change {
                    weezl::encode::Encoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8)
                } else {
                    weezl::encode::Encoder::new(weezl::BitOrder::Msb, 8)
                };
                encoder.encode(&data).expect("8-bit symbols")
            }
            Self::AsciiHex => ascii_hex(&data, style),
            Self::Ascii85 => ascii85(&data, style),
            Self::RunLength => run_length(&data, style),
        }
    }
}

/// Byte `i` of `style`, cyclically, or 0 when it is empty.
fn pick(style: &[u8], i: usize) -> u8 {
    if style.is_empty() {
        0
    } else {
        style[i % style.len()]
    }
}

/// `ASCIIHexDecode` input: digits of either case, whitespace between them,
/// and a `>` that is optional at the end of the data.
fn ascii_hex(data: &[u8], style: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 3 + 1);
    for (i, &b) in data.iter().enumerate() {
        let s = pick(style, i);
        let digits: &[u8; 16] = if s & 1 == 0 {
            b"0123456789abcdef"
        } else {
            b"0123456789ABCDEF"
        };
        out.push(digits[usize::from(b >> 4)]);
        if s & 0b110 == 0 {
            out.push(b' ');
        }
        out.push(digits[usize::from(b & 15)]);
        if s & 0b1_1000 == 0 {
            out.push(b'\n');
        }
    }
    if pick(style, data.len()) & 1 == 0 {
        out.push(b'>');
    }
    out
}

/// `ASCII85Decode` input (§7.4.3): `z` for a zero group where the style
/// allows it, line breaks between characters, a final partial group, and
/// the `~>` end marker.
fn ascii85(data: &[u8], style: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 5 / 4 + 8);
    for (i, chunk) in data.chunks(4).enumerate() {
        let s = pick(style, i);
        let mut group = [0u8; 4];
        group[..chunk.len()].copy_from_slice(chunk);
        let mut v = u32::from_be_bytes(group);
        if chunk.len() == 4 && v == 0 && s & 1 == 0 {
            out.push(b'z');
            continue;
        }
        let mut digits = [0u8; 5];
        for d in digits.iter_mut().rev() {
            *d = u8::try_from(v % 85).expect("below 85") + b'!';
            v /= 85;
        }
        out.extend_from_slice(&digits[..=chunk.len()]);
        if s & 0b110 == 0 {
            out.extend_from_slice(b"\r\n");
        }
    }
    out.extend_from_slice(b"~>");
    out
}

/// `RunLengthDecode` input (§7.4.5): runs and literal blocks of lengths the
/// style chooses, and the end-of-data byte unless the style omits it.
fn run_length(data: &[u8], style: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 2 + 1);
    let mut i = 0;
    let mut k = 0;
    while i < data.len() {
        let s = usize::from(pick(style, k));
        k += 1;
        let run = data[i..]
            .iter()
            .take(128)
            .take_while(|&&b| b == data[i])
            .count();
        if run >= 2 && s & 1 == 0 {
            let n = 2 + s % (run - 1);
            out.push(u8::try_from(257 - n).expect("n in 2..=128"));
            out.push(data[i]);
            i += n;
        } else {
            let n = 1 + s % (data.len() - i).min(128);
            out.push(u8::try_from(n - 1).expect("n in 1..=128"));
            out.extend_from_slice(&data[i..i + n]);
            i += n;
        }
    }
    if pick(style, k) & 3 != 0 {
        out.push(128);
    }
    out
}

fn predictor() -> impl Strategy<Value = Option<Predictor>> {
    let some = (
        prop_oneof![Just(2i64), 10i64..=15],
        1usize..=4,
        prop_oneof![Just(1usize), Just(2), Just(4), Just(8), Just(16)],
        1usize..=40,
    )
        .prop_map(|(kind, colors, bits, columns)| {
            Some(Predictor {
                kind,
                colors,
                bits,
                columns,
            })
        });
    prop_oneof![Just(None), some]
}

fn filter() -> impl Strategy<Value = Filter> {
    prop_oneof![
        predictor().prop_map(Filter::Flate),
        (any::<bool>(), predictor()).prop_map(|(early_change, predictor)| Filter::Lzw {
            early_change,
            predictor,
        }),
        Just(Filter::AsciiHex),
        Just(Filter::Ascii85),
        Just(Filter::RunLength),
    ]
}

impl Filter {
    /// This filter without a predictor.
    fn unpredicted(self) -> Self {
        match self {
            Self::Flate(_) => Self::Flate(None),
            Self::Lzw { early_change, .. } => Self::Lzw {
                early_change,
                predictor: None,
            },
            f => f,
        }
    }
}

/// A chain with the data it decodes to.  Only the last filter, the first
/// applied in encoding, may carry a predictor: an earlier one would see
/// encoded bytes that need not form whole rows.  The data is whole rows of
/// that predictor.
fn chain_and_data() -> impl Strategy<Value = (Vec<Filter>, Vec<u8>)> {
    (prop::collection::vec(filter(), 0..=2), filter()).prop_flat_map(|(head, last)| {
        let stride = last.predictor().map_or(1, Predictor::stride);
        let mut chain: Vec<Filter> = head.into_iter().map(Filter::unpredicted).collect();
        chain.push(last);
        let data = (0..=4096 / stride)
            .prop_flat_map(move |rows| prop::collection::vec(any::<u8>(), rows * stride));
        (Just(chain), data)
    })
}

/// A TIFF predictor with data of whole rows and then a partial row.
fn tiff_and_data() -> impl Strategy<Value = (Predictor, Vec<u8>)> {
    predictor()
        .prop_filter_map("a predictor", |p| p.map(|p| Predictor { kind: 2, ..p }))
        .prop_flat_map(|p| {
            let stride = p.stride();
            let data = (0usize..8, 0..stride).prop_flat_map(move |(rows, tail)| {
                prop::collection::vec(any::<u8>(), rows * stride + tail)
            });
            (Just(p), data)
        })
}

proptest! {
    #![proptest_config(config(1024))]

    /// Decoding the encoded data returns it whole.
    #[test]
    fn filter_chains_invert_their_encoders(
        (chain, data) in chain_and_data(),
        style in prop::collection::vec(any::<u8>(), 1..32),
    ) {
        let mut encoded = data.clone();
        for filter in chain.iter().rev() {
            encoded = filter.encode(&encoded, &style);
        }
        let names = chain.iter().map(|f| Object::Name(f.name().into())).collect();
        let parms = chain.iter().map(Filter::parms).collect();
        let mut dict = Dictionary::new();
        dict.set("Filter", Object::Array(names));
        dict.set("DecodeParms", Object::Array(parms));
        let decoded = decode_stream(&encoded, &dict).map_err(TestCaseError::fail)?;
        prop_assert_eq!(decoded.truncated, None);
        prop_assert_eq!(decoded.bytes, data);
    }

    /// TIFF predictor 2 passes a final partial row through unchanged.
    #[test]
    fn tiff_predictor_keeps_a_partial_last_row((p, data) in tiff_and_data()) {
        let whole = data.len() / p.stride() * p.stride();
        let mut predicted = p.encode(&data[..whole], &[]);
        predicted.extend_from_slice(&data[whole..]);
        let mut dict = Dictionary::new();
        dict.set("Filter", Object::Name(b"FlateDecode".to_vec()));
        dict.set("DecodeParms", p.parms(None));
        let compressed = Filter::Flate(None).encode(&predicted, &[]);
        let decoded = decode_stream(&compressed, &dict).map_err(TestCaseError::fail)?;
        prop_assert_eq!(decoded.bytes, data);
    }

    /// `RunLengthDecode` under an output bound returns a prefix of the
    /// unbounded output no longer than the bound, marked truncated exactly
    /// when it is shorter than the unbounded output.
    #[test]
    fn run_length_output_bound_is_a_prefix(
        input in prop::collection::vec(any::<u8>(), 0..1024),
        bound in 0usize..4096,
    ) {
        let full = decode_run_length(&input, MAX_DECOMPRESSED);
        let cut = decode_run_length(&input, bound);
        prop_assert!(cut.bytes.len() <= bound);
        prop_assert!(full.bytes.starts_with(&cut.bytes));
        if cut.bytes.len() < full.bytes.len() {
            prop_assert!(cut.truncated.is_some());
        }
        if full.bytes.len() <= bound {
            prop_assert_eq!(cut, full);
        }
    }
}
