//! A `JPXDecode` image is interpreted through its PDF dictionary
//! (PDF 32000-1 §7.4.9, §8.9.5): an `/Indexed` `/ColorSpace` maps the
//! codestream's samples through the PDF palette, and an alpha channel in
//! the codestream is the soft mask only when `/SMaskInData` is nonzero —
//! premultiplied by it when `/SMaskInData` is 2.

#[expect(
    dead_code,
    reason = "shared helpers; this test uses PDF assembly and sessions"
)]
mod common;

use common::{assemble_pdf, session, stream_object};
use rasterrocket::{BackendPolicy, render_page_rgb};

/// 8 × 8 one-channel 8-bit JP2; row `r` holds the value of `GRAY8_ROWS[r]`.
const GRAY8: &[u8] = include_bytes!("../../pdf_interp/src/resources/image/testdata/gray8.jp2");
const GRAY8_ROWS: [u8; 8] = [0x00, 0x24, 0x48, 0x6d, 0x91, 0xb6, 0xda, 0xff];

/// 8 × 8 RGBA JP2: columns 0–3 are (10, 220, 30) at alpha 255, columns
/// 4–7 (200, 100, 50) at alpha 127.
const RGBA8: &[u8] = include_bytes!("../../pdf_interp/src/resources/image/testdata/rgba8.jp2");

/// Page side in points; it renders at one pixel per point, one pixel per
/// image sample.
const SIDE: u32 = 8;

/// Render the JPX `data` with extra dictionary entries `entries` over a
/// white page and return the pixel at column `col` of row `row`.
fn pixel(entries: &str, data: &[u8], col: usize, row: usize) -> [u8; 3] {
    let objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {SIDE} {SIDE}] \
             /Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>"
        )
        .into_bytes(),
        stream_object(
            "",
            format!("q {SIDE} 0 0 {SIDE} 0 0 cm /Im0 Do Q").as_bytes(),
        ),
        stream_object(
            &format!(
                "/Type /XObject /Subtype /Image /Width {SIDE} /Height {SIDE} \
                 /BitsPerComponent 8 /Filter /JPXDecode {entries}"
            ),
            data,
        ),
    ];
    let s = session(assemble_pdf(&objects), BackendPolicy::CpuOnly);
    let bmp = render_page_rgb(&s, 1, 1.0).expect("render page");
    let at = row * bmp.stride + col * 3;
    let px = &bmp.data()[at..at + 3];
    [px[0], px[1], px[2]]
}

fn assert_near(name: &str, got: [u8; 3], want: [u8; 3]) {
    let close = got.iter().zip(want).all(|(&g, w)| g.abs_diff(w) <= 4);
    assert!(close, "{name}: got {got:?}, want about {want:?}");
}

#[test]
fn indexed_colour_space_applies_the_pdf_palette() {
    // Entry i of the palette is (i, 0, 255 − i).
    let palette: String = (0..=255u8)
        .map(|i| format!("{i:02X}00{:02X}", 255 - i))
        .collect();
    let entries = format!("/ColorSpace [/Indexed /DeviceRGB 255 <{palette}>]");
    for (row, &v) in GRAY8_ROWS.iter().enumerate() {
        assert_near(
            &format!("row {row}"),
            pixel(&entries, GRAY8, 3, row),
            [v, 0, 255 - v],
        );
    }
}

#[test]
fn codestream_alpha_follows_smask_in_data() {
    let rgb = "/ColorSpace /DeviceRGB";
    // Opaque columns are unaffected.
    assert_near("opaque", pixel(rgb, RGBA8, 1, 4), [10, 220, 30]);
    // Absent or 0: the alpha channel is ignored.
    assert_near("absent", pixel(rgb, RGBA8, 6, 4), [200, 100, 50]);
    assert_near(
        "SMaskInData 0",
        pixel(&format!("{rgb} /SMaskInData 0"), RGBA8, 6, 4),
        [200, 100, 50],
    );
    // 1: alpha 127 over white.
    assert_near(
        "SMaskInData 1",
        pixel(&format!("{rgb} /SMaskInData 1"), RGBA8, 6, 4),
        [228, 178, 153],
    );
    // 2: the colour is premultiplied, c = c'/α, clamped, then composited.
    assert_near(
        "SMaskInData 2",
        pixel(&format!("{rgb} /SMaskInData 2"), RGBA8, 6, 4),
        [255, 228, 178],
    );
}
