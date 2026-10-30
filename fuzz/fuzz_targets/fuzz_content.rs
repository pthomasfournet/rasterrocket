#![no_main]

//! Content-stream tokenizer and operator interpreter.  The input is the page
//! content stream, drawn with a standard font (`/F1`), an extended graphics
//! state (`/G1`: blend mode, constant alpha) and an axial shading (`/S1`) in
//! scope.

use libfuzzer_sys::fuzz_target;

const RESOURCES: &str = "<< /Font << /F1 5 0 R >> /ExtGState << /G1 6 0 R >> \
                         /Shading << /S1 7 0 R >> >>";

fuzz_target!(|data: &[u8]| {
    rasterrocket_fuzz::render_page(
        RESOURCES,
        data,
        vec![
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
            b"<< /Type /ExtGState /BM /Multiply /ca 0.5 /CA 0.5 /SMask /None >>".to_vec(),
            b"<< /ShadingType 2 /ColorSpace /DeviceRGB /Coords [0 0 100 0] \
              /Function << /FunctionType 2 /Domain [0 1] /C0 [1 0 0] /C1 [0 0 1] /N 1 >> >>"
                .to_vec(),
        ],
    );
});
