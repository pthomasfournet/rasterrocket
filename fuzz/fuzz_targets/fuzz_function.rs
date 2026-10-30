#![no_main]

//! PDF functions evaluated by shadings: the PostScript calculator (type 4)
//! compiler and interpreter, and sampled (type 0) functions, under a
//! one-input axial shading and a two-input function-based shading.
//!
//! Layout: `[kind, bits per sample, size]`, then the function stream body
//! (PostScript code for type 4, sample data for type 0).

use libfuzzer_sys::fuzz_target;
use rasterrocket_fuzz::common::stream_object;

const BITS_PER_SAMPLE: [u8; 8] = [1, 2, 4, 8, 12, 16, 24, 32];

fuzz_target!(|data: &[u8]| {
    let Some((sel, body)) = rasterrocket_fuzz::split_selectors::<3>(data) else {
        return;
    };
    let two_inputs = sel[0] & 1 != 0;
    let domain = if two_inputs { "[0 1 0 1]" } else { "[0 1]" };
    let range = "[0 1 0 1 0 1]";
    let function = if sel[0] & 2 == 0 {
        format!("/FunctionType 4 /Domain {domain} /Range {range}")
    } else {
        let n = u32::from(sel[2] % 16) + 1;
        let size = if two_inputs {
            format!("[{n} {n}]")
        } else {
            format!("[{n}]")
        };
        format!(
            "/FunctionType 0 /Domain {domain} /Range {range} /Size {size} \
             /BitsPerSample {} /Order {}",
            BITS_PER_SAMPLE[usize::from(sel[1]) % BITS_PER_SAMPLE.len()],
            if sel[0] & 4 == 0 { 1 } else { 3 },
        )
    };
    let shading = if two_inputs {
        "<< /ShadingType 1 /ColorSpace /DeviceRGB /Domain [0 1 0 1] \
         /Matrix [100 0 0 100 0 0] /Function 6 0 R >>"
    } else {
        "<< /ShadingType 2 /ColorSpace /DeviceRGB /Coords [0 0 100 0] \
         /Extend [true true] /Function 6 0 R >>"
    };
    rasterrocket_fuzz::render_page(
        "<< /Shading << /S1 5 0 R >> >>",
        b"/S1 sh",
        vec![shading.as_bytes().to_vec(), stream_object(&function, body)],
    );
});
