#![no_main]

//! Comic page images: content sniffing and the JPEG, PNG, WebP, TIFF
//! (every image of a multi-page file), JPEG XL and AVIF decoders (the
//! HEIF container, item properties, grids and the AV1 still decoder),
//! with their dimension budgets, conversion to gray at 8 or 16 bits,
//! orientation and deskew.  The image is the one entry of a TAR archive, so
//! no container checksum stands between the fuzzer and the decoders.
//!
//! Layout: `[flags]`, then the image file; `flags` as
//! `rasterrocket_fuzz::render_comic` reads them.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Some(([flags], image)) = rasterrocket_fuzz::split_selectors::<1>(data) else {
        return;
    };
    rasterrocket_fuzz::render_comic("book.cbt", rasterrocket_fuzz::tar_of("1.png", image), flags);
});
