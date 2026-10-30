#![no_main]

//! CMap program parser and code extraction.  The input is the CMap stream;
//! the input bytes themselves are then split into codes through the parsed
//! code-space ranges and mapped.

use libfuzzer_sys::fuzz_target;
use pdf_interp::resources::cmap::parse_cmap;

fuzz_target!(|data: &[u8]| {
    let Some(cmap) = parse_cmap(data) else {
        return;
    };
    let mut rest = data;
    while !rest.is_empty() {
        let (code, len) = cmap.next_code(rest);
        std::hint::black_box(cmap.lookup(code));
        rest = &rest[usize::from(len.max(1)).min(rest.len())..];
    }
});
