#![no_main]

//! Comic archives: the ZIP, 7-Zip, TAR and RAR containers, the
//! decompressed-size budget, entry filtering and natural ordering, content
//! sniffing, and the pages they hold, through `open_comic_bytes`.
//!
//! Layout: `[container, flags]`, then the archive.  The low two bits of
//! `container` select `.cbz`, `.cb7`, `.cbt` or `.cbr`; `flags` as
//! `rasterrocket_fuzz::render_comic` reads them.

use libfuzzer_sys::fuzz_target;

const NAMES: [&str; 4] = ["book.cbz", "book.cb7", "book.cbt", "book.cbr"];

fuzz_target!(|data: &[u8]| {
    let Some(([container, flags], archive)) = rasterrocket_fuzz::split_selectors::<2>(data) else {
        return;
    };
    rasterrocket_fuzz::render_comic(NAMES[usize::from(container & 3)], archive.to_vec(), flags);
});
