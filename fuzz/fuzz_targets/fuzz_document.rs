#![no_main]

//! Whole-document target: the cross-reference and object parser, the filter
//! chain, the content tokenizer and every resource the first pages draw
//! (fonts, CMaps, images, shadings, functions, patterns, annotations), as
//! the public render path runs them.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    rasterrocket_fuzz::render_document(data.to_vec());
});
