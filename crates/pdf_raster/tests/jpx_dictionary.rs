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

use std::fmt::Write as _;

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
    let palette = (0..=255u8).fold(String::new(), |mut s, i| {
        write!(s, "{i:02X}00{:02X}", 255 - i).expect("writing to a String");
        s
    });
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

/// 8 × 8 RGB raw codestream: a 16-colour gradient (see `rgb8.rgb` beside it).
const RGB8_J2K: &[u8] = include_bytes!("../../pdf_interp/src/resources/image/testdata/rgb8.j2k");

/// `codestream` wrapped in a JP2 file whose `colr` box carries the ICC
/// profile `profile` (method 2) for `channels` 8-bit channels.
fn jp2_with_icc(codestream: &[u8], channels: u16, profile: &[u8]) -> Vec<u8> {
    fn boxed(kind: [u8; 4], body: &[u8]) -> Vec<u8> {
        let mut b = u32::try_from(8 + body.len())
            .unwrap()
            .to_be_bytes()
            .to_vec();
        b.extend_from_slice(&kind);
        b.extend_from_slice(body);
        b
    }
    let mut ihdr = SIDE.to_be_bytes().to_vec();
    ihdr.extend_from_slice(&SIDE.to_be_bytes());
    ihdr.extend_from_slice(&channels.to_be_bytes());
    // 8 bits unsigned, JPEG 2000 compression, colour space known, no IPR.
    ihdr.extend_from_slice(&[7, 7, 0, 0]);
    // METH 2: a restricted ICC profile.
    let mut colr = vec![2, 0, 0];
    colr.extend_from_slice(profile);
    let jp2h = [boxed(*b"ihdr", &ihdr), boxed(*b"colr", &colr)].concat();
    [
        boxed(*b"jP  ", b"\r\n\x87\n"),
        boxed(*b"ftyp", b"jp2 \0\0\0\0jp2 "),
        boxed(*b"jp2h", &jp2h),
        boxed(*b"jp2c", codestream),
    ]
    .concat()
}

/// Without a `/ColorSpace`, the codestream's own ICC profile converts the
/// samples (ISO 32000-2 Table 89, `JPXDecode`); with one, the dictionary's
/// space does and the profile is ignored.
#[test]
fn embedded_icc_profile_applies_only_without_a_pdf_colour_space() {
    let jp2 = jp2_with_icc(
        RGB8_J2K,
        3,
        &common::icc::rgb_profile(common::icc::GAMMA_1_8),
    );
    let plain = pixel("/ColorSpace /DeviceRGB", &jp2, 4, 4);
    let through_profile = pixel("", &jp2, 4, 4);
    // Gamma 1.8 lightens every mid-tone: encode((v/255)^1.8) > v.
    assert!(
        through_profile
            .iter()
            .zip(plain)
            .all(|(&p, d)| (d > 20 && d < 235 && p > d) || p.abs_diff(d) <= 1),
        "profile {through_profile:?} should lighten the device reading {plain:?}"
    );
    assert_ne!(through_profile, plain);
}
