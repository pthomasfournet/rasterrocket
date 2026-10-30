#![no_main]

//! Image XObjects through the render path: the DCT, JPX, Flate, LZW,
//! RunLength, CCITT and JBIG2 decoders, device, ICC-based, indexed, Lab and
//! separation colour spaces, `/Decode`, `/ImageMask`, `/SMaskInData` and
//! image interpolation.
//!
//! Layout: `[filter, colour space, bpc, width, height, flags, icc_len_hi,
//! icc_len_lo]`, then the ICC profile (`icc_len` bytes, used by `/ICCBased`)
//! and the image stream body.

use libfuzzer_sys::fuzz_target;
use rasterrocket_fuzz::common::stream_object;

const FILTERS: [&str; 8] = [
    "",
    "/Filter /DCTDecode",
    "/Filter /JPXDecode",
    "/Filter /FlateDecode",
    "/Filter /LZWDecode",
    "/Filter /RunLengthDecode",
    "/Filter /CCITTFaxDecode /DecodeParms << /K -1 /Columns 16 >>",
    "/Filter /JBIG2Decode",
];
/// Colour spaces; `6 0 R` is the ICC profile stream, `7 0 R` a PostScript
/// tint transform.
const SPACES: [&str; 8] = [
    "/DeviceGray",
    "/DeviceRGB",
    "/DeviceCMYK",
    "[/ICCBased 6 0 R]",
    "[/Indexed /DeviceRGB 3 <ff000000ff000000ff000000>]",
    "[/Lab << /WhitePoint [0.9505 1 1.089] /Range [-100 100 -100 100] >>]",
    "[/Separation /Spot /DeviceCMYK << /FunctionType 2 /Domain [0 1] \
     /C0 [0 0 0 0] /C1 [0 1 1 0] /N 1 >>]",
    "[/DeviceN [/C /M] /DeviceCMYK 7 0 R]",
];
const BPC: [u8; 5] = [1, 2, 4, 8, 16];
/// Profile component counts, by colour-space selector bits.
const ICC_N: [u8; 3] = [1, 3, 4];

fuzz_target!(|data: &[u8]| {
    let Some((sel, rest)) = rasterrocket_fuzz::split_selectors::<8>(data) else {
        return;
    };
    let icc_len = usize::from(u16::from_be_bytes([sel[6], sel[7]])).min(rest.len());
    let (icc, body) = rest.split_at(icc_len);

    let flags = sel[5];
    let mut dict = format!(
        "/Type /XObject /Subtype /Image /Width {} /Height {} {}",
        u32::from(sel[3]) + 1,
        u32::from(sel[4]) + 1,
        FILTERS[usize::from(sel[0]) % FILTERS.len()],
    );
    if flags & 1 != 0 {
        dict.push_str(" /ImageMask true /BitsPerComponent 1");
    } else {
        dict.push_str(&format!(
            " /ColorSpace {} /BitsPerComponent {}",
            SPACES[usize::from(sel[1]) % SPACES.len()],
            BPC[usize::from(sel[2]) % BPC.len()],
        ));
    }
    if flags & 2 != 0 {
        dict.push_str(" /Decode [1 0 1 0 1 0 1 0]");
    }
    if flags & 4 != 0 {
        dict.push_str(" /Interpolate true");
    }
    if flags & 8 != 0 {
        dict.push_str(&format!(" /SMaskInData {}", (flags >> 4) % 3));
    }

    let icc_n = ICC_N[usize::from(sel[1] >> 4) % ICC_N.len()];
    rasterrocket_fuzz::render_page(
        "<< /XObject << /Im1 5 0 R >> >>",
        b"q 100 0 0 100 0 0 cm /Im1 Do Q",
        vec![
            stream_object(&dict, body),
            stream_object(&format!("/N {icc_n}"), icc),
            stream_object(
                "/FunctionType 4 /Domain [0 1 0 1] /Range [0 1 0 1 0 1 0 1]",
                b"{ 0 0 }",
            ),
        ],
    );
});
