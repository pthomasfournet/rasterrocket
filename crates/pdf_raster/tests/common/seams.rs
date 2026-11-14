//! A synthetic page whose objects cross window edges: the content every
//! test of windowed rendering (regions, bands) draws, so that each window
//! must receive the pixels the whole page places there.

use std::fmt::Write as _;

use super::{one_page, stream_object};

/// Content whose objects all cross several window edges of a 150 dpi
/// render: fills, a stroke, a hairline, clips, axial, radial,
/// function-based and Gouraud shadings,
/// a soft mask with a backdrop and a transfer function, nested isolated
/// and knockout groups with a blend mode, coloured and uncoloured tiling
/// patterns on a skewed lattice, a skewed interpolated image with a soft
/// mask, a box-reduced image, an image mask, a shading pattern with its
/// own graphics state filled and stroked, text, a Type 3 glyph run and a
/// text clip.
pub const CONTENT: &str = "\
q 0 0 200 100 re W n /G4 sh Q \
q 0.2 0.4 0.8 rg 13.3 7.7 m 187.1 52.9 l 61.4 93.2 l h f Q \
q 0.8 0 0 RG 2.7 w 1 J 5 95 m 195 5 l S Q \
q 0 w 0 0 0 RG 10 3 m 190 47.3 l S Q \
q 30 20 140 60 re W n 1 0 0 rg 0 0 200 100 re f Q \
q 120 30 60 40 re W n 1 0 0 1 120 30 cm /Ax sh Q \
q 10 60 50 35 re W* n /Rad sh Q \
q 70 5 40 40 re W n 40 0 0 40 70 5 cm /Fn sh Q \
q /SM gs 0 0.5 0 rg 40 10 120 80 re f Q \
q /Grp Do Q \
q /Pattern cs /P1 scn 140 5 55 90 re f Q \
q /PCS cs 0.1 0.7 0.3 /P2 scn 5 5 40 40 re f Q \
q /Pattern cs /SP scn 100 60 90 35 re f /Pattern CS /SP SCN 4 w 20 20 m 180 80 l S Q \
q 31.3 6.9 -9.2 27.6 40.7 22.1 cm /Im Do Q \
q 10 0 0 10 150 45 cm /Big Do Q \
q 0.5 0 0.5 rg 20 0 0 6 100 47 cm BI /IM true /W 8 /H 2 /BPC 1 /F /AHx ID A55A> EI Q \
BT /F 17.3 Tf 21.1 41.7 Td (Bands) Tj ET \
BT /T3 23 Tf 120 44 Td (aaa) Tj ET \
q BT 7 Tr /F 30 Tf 60 30 Td (Clip) Tj ET 0 0 1 rg 0 0 200 100 re f Q";

/// A `w × h` gray image whose samples vary along both axes, as hex.
fn gradient_hex(w: u32, h: u32) -> String {
    let mut hex = String::new();
    for y in 0..h {
        for x in 0..w {
            let _ = write!(hex, "{:02x}", (x * 7 + y * 13) % 256);
        }
    }
    hex.push('>');
    hex
}

/// A 200 × 100 pt page with [`CONTENT`] and the page dictionary entries
/// `extra`.
pub fn seam_page(extra: &str) -> Vec<u8> {
    let page = format!(
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] {extra} \
         /Resources << /Font << /F 5 0 R /T3 17 0 R >> \
         /Shading << /Ax 6 0 R /Rad 7 0 R /Fn 8 0 R /G4 19 0 R >> \
         /ExtGState << /SM 9 0 R >> \
         /XObject << /Grp 11 0 R /Im 15 0 R /Big 20 0 R >> \
         /Pattern << /P1 13 0 R /P2 14 0 R /SP 23 0 R >> \
         /ColorSpace << /PCS [/Pattern /DeviceRGB] >> >> \
         /Contents 4 0 R >>"
    );
    let rgb_image = "ff0000 00ff00 0000ff ffff00 00ffff ff00ff 808080 202020 \
                     c0c0c0 ff8000 0080ff 80ff00 400000 004000 000040 ffffff>";
    one_page(
        &page,
        vec![
            // 4: content
            stream_object("", CONTENT.as_bytes()),
            // 5: text font
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
            // 6: axial shading
            b"<< /ShadingType 2 /ColorSpace /DeviceRGB /Coords [0 0 60 0] \
               /Function << /FunctionType 2 /Domain [0 1] /C0 [1 1 0] /C1 [0 0.5 1] /N 1 >> \
               /Extend [true true] >>"
                .to_vec(),
            // 7: radial shading
            b"<< /ShadingType 3 /ColorSpace /DeviceRGB /Coords [35 77 2 35 77 30] \
               /Function << /FunctionType 2 /Domain [0 1] /C0 [1 0 0] /C1 [0 0 1] /N 1 >> \
               /Extend [true true] >>"
                .to_vec(),
            // 8: function-based shading over the unit square
            b"<< /ShadingType 1 /ColorSpace /DeviceRGB /Domain [0 1 0 1] /Function 22 0 R >>"
                .to_vec(),
            // 9: soft-mask graphics state
            b"<< /Type /ExtGState /ca 0.8 /SMask << /Type /Mask /S /Luminosity /G 10 0 R \
               /BC [0.3] /TR << /FunctionType 2 /Domain [0 1] /C0 [0] /C1 [1] /N 2 >> >> >>"
                .to_vec(),
            // 10: soft-mask group
            stream_object(
                "/Type /XObject /Subtype /Form /BBox [0 0 200 100] \
                 /Group << /S /Transparency /CS /DeviceGray >>",
                b"0.9 g 0 0 100 100 re f 0.4 g 60 20 100 60 re f",
            ),
            // 11: isolated knockout group holding a non-isolated group
            stream_object(
                "/Type /XObject /Subtype /Form /BBox [0 0 200 100] \
                 /Group << /S /Transparency /I true /K true >> \
                 /Resources << /ExtGState << /Mul 12 0 R >> /XObject << /Inner 21 0 R >> >>",
                b"/Mul gs 1 1 0 rg 50 20 60 50 re f 0 1 1 rg 80 35 60 50 re f /Inner Do",
            ),
            // 12: multiply blend
            b"<< /Type /ExtGState /ca 0.6 /BM /Multiply >>".to_vec(),
            // 13: coloured tiling pattern on a skewed lattice
            stream_object(
                "/Type /Pattern /PatternType 1 /PaintType 1 /TilingType 1 \
                 /BBox [0 0 7 5] /XStep 7 /YStep 5 /Matrix [1 0.3 0 1 0 0] /Resources << >>",
                b"1 0 0 rg 0 0 3 3 re f 0 0 1 rg 3.5 2 2 2 re f",
            ),
            // 14: uncoloured tiling pattern with cells larger than a band
            stream_object(
                "/Type /Pattern /PatternType 1 /PaintType 2 /TilingType 1 \
                 /BBox [0 0 13 13] /XStep 15 /YStep 15 /Resources << >>",
                b"0 0 m 13 6 l 2 13 l h f",
            ),
            // 15: interpolated RGB image with a soft mask
            stream_object(
                "/Type /XObject /Subtype /Image /Width 4 /Height 4 /ColorSpace /DeviceRGB \
                 /BitsPerComponent 8 /Interpolate true /SMask 16 0 R /Filter /ASCIIHexDecode",
                rgb_image.as_bytes(),
            ),
            // 16: its soft mask
            stream_object(
                "/Type /XObject /Subtype /Image /Width 4 /Height 4 /ColorSpace /DeviceGray \
                 /BitsPerComponent 8 /Filter /ASCIIHexDecode",
                b"ff c0 80 40 c0 ff c0 80 80 c0 ff c0 40 80 c0 ff>",
            ),
            // 17: Type 3 font
            b"<< /Type /Font /Subtype /Type3 /FontBBox [0 0 100 100] \
               /FontMatrix [0.01 0 0 0.01 0 0] /CharProcs << /a 18 0 R >> \
               /Encoding << /Differences [97 /a] >> /FirstChar 97 /LastChar 97 /Widths [110] >>"
                .to_vec(),
            // 18: its glyph
            stream_object("", b"100 0 0 0 100 100 d1 10 0 m 90 0 l 50 100 l h f"),
            // 19: Gouraud triangle
            stream_object(
                "/ShadingType 4 /ColorSpace /DeviceRGB /BitsPerCoordinate 8 \
                 /BitsPerComponent 8 /BitsPerFlag 8 /Decode [0 200 0 100 0 1 0 1 0 1] \
                 /Filter /ASCIIHexDecode",
                b"00 19 19 ff 00 00  00 e6 4c 00 ff 00  00 73 f2 00 00 ff>",
            ),
            // 20: a 64 x 64 image drawn at 21 px, box-reduced
            stream_object(
                "/Type /XObject /Subtype /Image /Width 64 /Height 64 /ColorSpace /DeviceGray \
                 /BitsPerComponent 8 /Filter /ASCIIHexDecode",
                gradient_hex(64, 64).as_bytes(),
            ),
            // 21: non-isolated inner group
            stream_object(
                "/Type /XObject /Subtype /Form /BBox [0 0 200 100] \
                 /Group << /S /Transparency >> /Resources << /ExtGState << /Mul 12 0 R >> >>",
                b"q /Mul gs 0.5 0 1 rg 20 50 160 20 re f Q",
            ),
            // 22: the function-based shading's function
            stream_object(
                "/FunctionType 4 /Domain [0 1 0 1] /Range [0 1 0 1 0 1]",
                b"{ 2 copy mul 3 1 roll }",
            ),
            // 23: a shading pattern with its own graphics state
            b"<< /PatternType 2 /Shading 7 0 R /Matrix [1 0.2 0 1 10 0] \
               /ExtGState << /ca 0.7 /CA 0.6 >> >>"
                .to_vec(),
        ],
    )
}

/// Page dictionary entries the seam tests cover.
pub const GEOMETRIES: [&str; 6] = [
    "",
    "/Rotate 90",
    "/Rotate 180",
    "/Rotate 270",
    "/CropBox [17 9 181 97]",
    "/UserUnit 1.5 /CropBox [17 9 181 97] /Rotate 90",
];
