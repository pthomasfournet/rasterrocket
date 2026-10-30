#![no_main]

//! Stream filter chain: Flate, LZW, ASCIIHex, ASCII85, RunLength and Brotli,
//! chained up to three deep, with PNG / TIFF predictor parameters.
//!
//! Layout: `[count, f0, f1, f2, predictor, colors, bpc, columns, early]`,
//! then the stream body.

use libfuzzer_sys::fuzz_target;
use pdf::{Dictionary, Object, Stream};

const FILTERS: [&[u8]; 6] = [
    b"FlateDecode",
    b"LZWDecode",
    b"ASCIIHexDecode",
    b"ASCII85Decode",
    b"RunLengthDecode",
    b"BrotliDecode",
];
const PREDICTORS: [i64; 8] = [1, 2, 10, 11, 12, 13, 14, 15];
const BPC: [i64; 5] = [1, 2, 4, 8, 16];

fuzz_target!(|data: &[u8]| {
    let Some((sel, body)) = rasterrocket_fuzz::split_selectors::<9>(data) else {
        return;
    };
    let count = usize::from(sel[0] % 3) + 1;
    let filters = sel[1..=count]
        .iter()
        .map(|&b| Object::Name(FILTERS[usize::from(b) % FILTERS.len()].to_vec()))
        .collect();
    let parms = Object::Dictionary(Dictionary::from_iter([
        (
            b"Predictor".to_vec(),
            Object::Integer(PREDICTORS[usize::from(sel[4]) % PREDICTORS.len()]),
        ),
        (
            b"Colors".to_vec(),
            Object::Integer(i64::from(sel[5] % 4) + 1),
        ),
        (
            b"BitsPerComponent".to_vec(),
            Object::Integer(BPC[usize::from(sel[6]) % BPC.len()]),
        ),
        (b"Columns".to_vec(), Object::Integer(i64::from(sel[7]) + 1)),
        (
            b"EarlyChange".to_vec(),
            Object::Integer(i64::from(sel[8] & 1)),
        ),
    ]));
    let dict = Dictionary::from_iter([
        (b"Filter".to_vec(), Object::Array(filters)),
        (b"DecodeParms".to_vec(), Object::Array(vec![parms; count])),
    ]);
    drop(Stream::new(dict, body.to_vec()).decompressed_content());
});
