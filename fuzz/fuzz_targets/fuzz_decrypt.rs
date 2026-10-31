#![no_main]

//! Standard security handler: the `/Encrypt` dictionary, key derivation
//! (R2–R6) and per-object RC4 / AES decryption, on arbitrary bytes with
//! decryption authorised.  The password is the user password the seed
//! corpus is encrypted with, so mutants of the seeds reach object
//! decryption rather than stopping at authentication.

use libfuzzer_sys::fuzz_target;
use pdf::{Document, OpenOptions};

fuzz_target!(|data: &[u8]| {
    let options = OpenOptions::new(true, Some("user".into()));
    let Ok(doc) = Document::from_bytes_with(data.to_vec(), &options) else {
        return;
    };
    let Ok(page) = doc.get_page(0) else {
        return;
    };
    drop(doc.get_page_fonts(page));
    drop(doc.get_page_content(page));
});
