#![no_main]

//! CMap program parser and code extraction.  The input is the CMap stream;
//! the input bytes themselves are then split into codes through the parsed
//! code-space ranges and mapped.

use libfuzzer_sys::fuzz_target;
use pdf_interp::fuzz_helpers::parse_and_map_cmap;

fuzz_target!(|data: &[u8]| parse_and_map_cmap(data));
