#![no_main]

//! JBIG2 decoder through the image path, as an embedded stream without
//! `/JBIG2Globals` (the common case in PDF files).
//!
//! Layout: `[width_lo, width_hi, height_lo, height_hi]`, then the embedded
//! JBIG2 stream.

use libfuzzer_sys::fuzz_target;
use rasterrocket_fuzz::common::assemble_pdf;

fuzz_target!(|data: &[u8]| {
    let Some((dims, payload)) = rasterrocket_fuzz::split_selectors::<4>(data) else {
        return;
    };
    let (width, height) = rasterrocket_fuzz::image_dimensions(dims);
    // The decoder takes the document only to resolve `/JBIG2Globals`; without
    // `parms` an empty page tree suffices, and parsing it resolves no object.
    let doc = pdf::Document::from_bytes_owned(assemble_pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [] /Count 0 >>".to_vec(),
    ]))
    .expect("the empty page tree parses");
    std::hint::black_box(pdf_interp::fuzz_helpers::decode_jbig2(
        &doc, payload, width, height, false, None,
    ));
});
