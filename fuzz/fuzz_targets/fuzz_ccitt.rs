#![no_main]

//! CCITT fax decoder through the image path, on each `/K` encoding.
//!
//! Layout: `[selector, width_lo, width_hi, height_lo, height_hi]`, then the
//! encoded data.  `selector & 0b11` picks the encoding:
//!
//! - 0: `K = 0` (Group 3, 1-D), with no `/DecodeParms`;
//! - 1: `K = -1` (Group 4, 2-D);
//! - 2: `K = 1` (Group 3, mixed 1-D/2-D);
//! - 3: `K = 1`, decoded as an image mask.
//!
//! Bit 7 of `selector` sets `/BlackIs1` on encodings 1–3.

use libfuzzer_sys::fuzz_target;
use pdf::{Dictionary, Object};
use pdf_interp::fuzz_helpers::decode_ccitt;

fuzz_target!(|data: &[u8]| {
    let Some(([selector, dims @ ..], payload)) = rasterrocket_fuzz::split_selectors::<5>(data)
    else {
        return;
    };
    let (width, height) = rasterrocket_fuzz::image_dimensions(dims);
    let black_is_1 = selector & 0b1000_0000 != 0;
    let (k, is_mask) = match selector & 0b11 {
        0 => {
            std::hint::black_box(decode_ccitt(payload, width, height, false, None));
            return;
        }
        1 => (-1, false),
        2 => (1, false),
        _ => (1, true),
    };
    let parms = Object::Dictionary(Dictionary::from_iter([
        (b"K".to_vec(), Object::Integer(k)),
        (b"BlackIs1".to_vec(), Object::Boolean(black_is_1)),
    ]));
    std::hint::black_box(decode_ccitt(payload, width, height, is_mask, Some(&parms)));
});
