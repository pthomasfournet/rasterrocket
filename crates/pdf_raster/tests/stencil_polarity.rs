//! A stencil mask paints where its decoded sample is 0 and leaves the page
//! unchanged where it is 1, and `/Decode [1 0]` swaps the two (PDF 32000-1
//! §8.9.6.2).  The same holds for every way a stencil reaches the
//! renderer — an image `XObject` or an inline image, raw or through a
//! transport filter or a bilevel codec — and for an explicit `/Mask`
//! stencil, which shows its base image where it would paint.

#[expect(
    dead_code,
    reason = "shared helpers; this test uses PDF assembly and sessions"
)]
mod common;

use common::{assemble_pdf, session, stream_object};
use rasterrocket::{BackendPolicy, render_page_rgb};

/// Stencil side in samples; the page is the same size in points and
/// renders at one pixel per point, so each sample covers one pixel.
const SIDE: u32 = 16;

/// Raw 1-bit rows: the left eight samples are 1, the right eight 0.
const RAW_LEFT_ONES: [u8; 32] = [
    0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00,
    0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00,
];

/// A CCITT Group 4 (`K -1`) coding of a 16 × 16 image whose left eight
/// columns are coded white and right eight black: it decodes, with
/// `BlackIs1 false`, to sample 1 on the left and 0 on the right.
const G4_RIGHT_BLACK: [u8; 9] = [0x33, 0x17, 0xFF, 0xFF, 0xFF, 0xF0, 0x01, 0x00, 0x10];

/// An embedded JBIG2 stream (page information and one immediate lossless
/// generic region) of a 16 × 16 image whose left eight columns are black:
/// it decodes to sample 0 on the left and 1 on the right.
const JBIG2_LEFT_BLACK: [u8; 80] = [
    0x00, 0x00, 0x00, 0x00, 0x30, 0x00, 0x01, 0x00, 0x00, 0x00, 0x13, 0x00, 0x00, 0x00, 0x10, 0x00,
    0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x01, 0x26, 0x00, 0x01, 0x00, 0x00, 0x00, 0x27, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
    0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0xFF, 0xFD, 0xFF, 0x02,
    0xFE, 0xFE, 0xFE, 0xFE, 0x99, 0x9A, 0xC8, 0x01, 0xAE, 0x01, 0xFF, 0x7F, 0xFF, 0x7F, 0xFF, 0xAC,
];

/// Which half of the page a stencil painted.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Painted {
    Left,
    Right,
}

impl Painted {
    const fn swapped(self) -> Self {
        match self {
            Self::Left => Self::Right,
            Self::Right => Self::Left,
        }
    }
}

/// How the stencil reaches the page.
#[derive(Clone, Copy)]
enum Use {
    /// `/Im0 Do` of an `/ImageMask true` image, filled red.
    XObject,
    /// `BI /IM true … ID … EI`, filled red.
    Inline,
    /// `/Mask` of a red 16 × 16 `DeviceRGB` image.
    ExplicitMask,
}

/// `HexString` encoding for an `/ASCIIHexDecode` stream body.
fn hex(data: &[u8]) -> Vec<u8> {
    let mut out: Vec<u8> = data
        .iter()
        .flat_map(|b| format!("{b:02X}").into_bytes())
        .collect();
    out.push(b'>');
    out
}

/// Render the stencil described by `entries` (its `/Filter`, `/DecodeParms`
/// and `/Decode`, in full-key form) and `data`, and report the red half.
fn painted(how: Use, entries: &str, data: &[u8]) -> Painted {
    painted_with_bpc(how, "/BitsPerComponent 1", entries, data)
}

/// [`painted`] with the stencil's `/BitsPerComponent` entry, in full-key
/// form, given as `bpc` (empty to omit it).
fn painted_with_bpc(how: Use, bpc: &str, entries: &str, data: &[u8]) -> Painted {
    let s = session(
        assemble_pdf(&stencil_objects(how, bpc, entries, data)),
        BackendPolicy::CpuOnly,
    );
    let bmp = render_page_rgb(&s, 1, 1.0).expect("render page");
    let row = &bmp.data()[8 * bmp.stride..];
    let red = |x: usize| row[x * 3] > 200 && row[x * 3 + 1] < 60 && row[x * 3 + 2] < 60;
    match (red(2), red(13)) {
        (true, false) => Painted::Left,
        (false, true) => Painted::Right,
        other => panic!("stencil painted neither half alone: (left, right) = {other:?}"),
    }
}

/// The objects of a one-page document painting the stencil described by
/// `bpc`, `entries` and `data` as `how` says.
fn stencil_objects(how: Use, bpc: &str, entries: &str, data: &[u8]) -> Vec<Vec<u8>> {
    let mut objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {SIDE} {SIDE}] \
             /Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>"
        )
        .into_bytes(),
    ];
    let stencil_dict = format!(
        "/Type /XObject /Subtype /Image /Width {SIDE} /Height {SIDE} \
         /ImageMask true {bpc} {entries}"
    );
    match how {
        Use::XObject => {
            objects.push(stream_object(
                "",
                format!("1 0 0 rg q {SIDE} 0 0 {SIDE} 0 0 cm /Im0 Do Q").as_bytes(),
            ));
            objects.push(stream_object(&stencil_dict, data));
        }
        Use::Inline => {
            let params = format!("{bpc} {entries}")
                .replace("/BitsPerComponent", "/BPC")
                .replace("/Filter", "/F")
                .replace("/DecodeParms", "/DP")
                .replace("/Decode ", "/D ")
                .replace("/ASCIIHexDecode", "/AHx")
                .replace("/CCITTFaxDecode", "/CCF");
            let mut content = format!(
                "1 0 0 rg q {SIDE} 0 0 {SIDE} 0 0 cm BI /W {SIDE} /H {SIDE} /IM true \
                 {params} ID "
            )
            .into_bytes();
            content.extend_from_slice(data);
            content.extend_from_slice(b"\nEI Q");
            objects.push(stream_object("", &content));
            objects.push(b"null".to_vec());
        }
        Use::ExplicitMask => {
            objects.push(stream_object(
                "",
                format!("q {SIDE} 0 0 {SIDE} 0 0 cm /Im0 Do Q").as_bytes(),
            ));
            objects.push(stream_object(
                &format!(
                    "/Type /XObject /Subtype /Image /Width {SIDE} /Height {SIDE} \
                     /ColorSpace /DeviceRGB /BitsPerComponent 8 /Mask 6 0 R"
                ),
                &[255, 0, 0].repeat((SIDE * SIDE) as usize),
            ));
            objects.push(stream_object(&stencil_dict, data));
        }
    }
    objects
}

/// Every source of stencil samples, with the half its default decoding
/// paints: the half where the decoded sample is 0.
fn sources() -> Vec<(&'static str, String, Vec<u8>, Painted)> {
    vec![
        ("raw", String::new(), RAW_LEFT_ONES.to_vec(), Painted::Right),
        (
            "ASCIIHex",
            "/Filter /ASCIIHexDecode".into(),
            hex(&RAW_LEFT_ONES),
            Painted::Right,
        ),
        (
            "CCITT G4",
            format!(
                "/Filter /CCITTFaxDecode /DecodeParms << /K -1 /Columns {SIDE} /Rows {SIDE} >>"
            ),
            G4_RIGHT_BLACK.to_vec(),
            Painted::Right,
        ),
        (
            "CCITT G4, array /DecodeParms",
            format!(
                "/Filter [/CCITTFaxDecode] \
                 /DecodeParms [<< /K -1 /Columns {SIDE} /Rows {SIDE} >>]"
            ),
            G4_RIGHT_BLACK.to_vec(),
            Painted::Right,
        ),
        (
            "ASCIIHex then CCITT G4",
            format!(
                "/Filter [/ASCIIHexDecode /CCITTFaxDecode] \
                 /DecodeParms [null << /K -1 /Columns {SIDE} /Rows {SIDE} >>]"
            ),
            hex(&G4_RIGHT_BLACK),
            Painted::Right,
        ),
        (
            "JBIG2",
            "/Filter /JBIG2Decode".into(),
            JBIG2_LEFT_BLACK.to_vec(),
            Painted::Left,
        ),
    ]
}

fn check(how: Use, what: &str, skip: &[&str]) {
    for (name, entries, data, default) in sources() {
        if skip.contains(&name) {
            continue;
        }
        assert_eq!(
            painted(how, &entries, &data),
            default,
            "{what}, {name}, default /Decode"
        );
        let inverted = format!("{entries} /Decode [1 0]");
        assert_eq!(
            painted(how, &inverted, &data),
            default.swapped(),
            "{what}, {name}, /Decode [1 0]"
        );
    }
}

#[test]
fn xobject_stencils_paint_sample_zero() {
    check(Use::XObject, "XObject stencil", &[]);
}

#[test]
fn inline_stencils_paint_sample_zero() {
    // An inline image cannot carry JBIG2 globals; the case adds nothing
    // the XObject one does not cover.
    check(Use::Inline, "inline stencil", &["JBIG2"]);
}

#[test]
fn explicit_mask_shows_the_image_where_sample_is_zero() {
    check(Use::ExplicitMask, "explicit /Mask", &[]);
}

/// `/BitsPerComponent` is optional on an image mask, which is one bit per
/// sample; inline, `/IM true` with no `/BPC`.
#[test]
fn stencil_without_bits_per_component_is_one_bit() {
    for how in [Use::XObject, Use::Inline, Use::ExplicitMask] {
        assert_eq!(
            painted_with_bpc(how, "", "", &RAW_LEFT_ONES),
            Painted::Right
        );
        assert_eq!(
            painted_with_bpc(how, "", "/Decode [1 0]", &RAW_LEFT_ONES),
            Painted::Left
        );
    }
}

/// A stencil whose stream ends early leaves the missing rows unpainted
/// under either polarity.
#[test]
fn truncated_raw_stencil_leaves_missing_rows_unpainted() {
    for decode in ["", "/Decode [1 0]"] {
        // Only the top four of sixteen rows are present.
        let objects = stencil_objects(
            Use::XObject,
            "/BitsPerComponent 1",
            decode,
            &RAW_LEFT_ONES[..8],
        );
        let s = session(assemble_pdf(&objects), BackendPolicy::CpuOnly);
        let bmp = render_page_rgb(&s, 1, 1.0).expect("render page");
        let px = |x: usize, y: usize| &bmp.data()[y * bmp.stride + x * 3..][..3];
        assert_eq!(
            px(13, 1)[1] < 60,
            decode.is_empty(),
            "{decode}: present row"
        );
        for y in 4..SIDE as usize {
            for x in 0..SIDE as usize {
                assert_eq!(
                    px(x, y),
                    [255, 255, 255],
                    "{decode}: missing row {y} at {x}"
                );
            }
        }
    }
}
