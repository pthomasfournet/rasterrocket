#![no_main]

//! The parser without the renderer: the cross-reference reader and its
//! reconstruction fallback, the object and object-stream parsers, both
//! page-tree walkers, inherited page attributes and the `/Contents` filter
//! chain, on arbitrary bytes.  An iteration costs no rendering, so the
//! structural parsers see far more inputs than through `fuzz_document`.

use libfuzzer_sys::fuzz_target;
use pdf::Document;

/// Object numbers resolved per input, from 0: every object of a small
/// file, a missing number taking the header-scan fallback.
const OBJECTS: u32 = 512;

/// Pages whose tree position, attributes and content are read per input.
const PAGES: u32 = 8;

fuzz_target!(|data: &[u8]| {
    let Ok(doc) = Document::from_bytes_owned(data.to_vec()) else {
        return;
    };
    for num in 0..OBJECTS {
        drop(doc.get_object((num, 0)));
    }
    drop(doc.linearization_hints());
    for idx in 0..doc.page_count_fast().min(PAGES) {
        drop(doc.get_page(idx));
    }
    for (_, page) in doc.get_pages().take(PAGES as usize) {
        drop(doc.get_inherited_page_attr(page, b"MediaBox"));
        drop(doc.get_page_resource_dict(page, b"Font"));
        drop(doc.page_content(page));
    }
});
