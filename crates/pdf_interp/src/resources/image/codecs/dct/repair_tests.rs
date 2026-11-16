//! `decode_dct` on streams with the producer faults [`super::stream::repair`]
//! mends, encoded by an independent JPEG encoder.

use jpeg_encoder::{ColorType, Encoder};

use super::decode_dct;
use super::stream::jpeg_sof;
use crate::resources::image::ImageDecodeCtx;
use crate::resources::image::colorspace::Samples;

/// A `w × h` RGB gradient, JPEG-coded baseline or progressive.
fn rgb_jpeg(w: u16, h: u16, progressive: bool) -> Vec<u8> {
    let byte = |v: u16| u8::try_from(v % 256).expect("reduced mod 256");
    let pixels: Vec<u8> = (0..h)
        .flat_map(|y| (0..w).flat_map(move |x| [byte(x * 7), byte(y * 5), byte(x + y)]))
        .collect();
    let mut out = Vec::new();
    let mut enc = Encoder::new(&mut out, 85);
    enc.set_progressive(progressive);
    enc.encode(&pixels, w, h, ColorType::Rgb).expect("encode");
    out
}

/// `decode_dct` on the CPU, as the dictionary declares `pdf_w × pdf_h`.
fn decode(data: &[u8], pdf_w: u32, pdf_h: u32) -> Option<Samples> {
    let doc = crate::test_helpers::make_doc_with_objects::<&[u8]>(&[]);
    decode_dct(
        data,
        pdf_w,
        pdf_h,
        &mut ImageDecodeCtx::cpu(&doc),
        #[cfg(feature = "gpu-jpeg-huffman")]
        None::<&mut gpu::jpeg_decoder::JpegGpuDecoder<gpu::backend::cuda::CudaBackend>>,
        #[cfg(feature = "gpu-jpeg-huffman")]
        None,
        #[cfg(feature = "cache")]
        None,
    )
}

fn pixels(s: &Samples) -> &[u8] {
    s.data.as_cpu().expect("CPU samples")
}

#[test]
fn a_complete_stream_without_eoi_decodes_whole() {
    for progressive in [false, true] {
        let full = rgb_jpeg(64, 48, progressive);
        let reference = decode(&full, 64, 48).expect("reference decode");
        assert!(reference.partial.is_none());
        let no_eoi = &full[..full.len() - 2];
        let img = decode(no_eoi, 64, 48).expect("decode without EOI");
        assert_eq!(
            pixels(&img),
            pixels(&reference),
            "progressive {progressive}"
        );
        assert!(img.partial.is_none(), "{:?}", img.partial);
    }
}

/// A stream that ends at a marker short of its data (a scan cut at a
/// restart marker, the scans after the first few missing) has no error for
/// the decoder to meet; the rows or coefficients it lacks are reported.
#[test]
fn a_stream_cut_at_a_marker_is_reported_partial() {
    for progressive in [false, true] {
        let mut full = Vec::new();
        let mut enc = Encoder::new(&mut full, 85);
        enc.set_progressive(progressive);
        enc.set_restart_interval(1);
        let pixels: Vec<u8> = (0..64_u32 * 48 * 3)
            .map(|i| i.wrapping_mul(2_654_435_761).to_be_bytes()[0])
            .collect();
        enc.encode(&pixels, 64, 48, ColorType::Rgb).expect("encode");
        let reference = decode(&full, 64, 48).expect("reference decode");
        assert!(reference.partial.is_none(), "{:?}", reference.partial);

        // Every marker after the first scan header: restart markers
        // inside a scan, and the headers of later scans.
        let first_scan = full
            .windows(2)
            .position(|m| m == [0xFF, 0xDA])
            .expect("SOS");
        let cuts: Vec<usize> = (first_scan + 2..full.len() - 2)
            .filter(|&at| full[at] == 0xFF && !matches!(full[at + 1], 0x00 | 0xFF))
            .collect();
        assert!(cuts.len() > 4, "{} markers to cut at", cuts.len());
        for at in cuts {
            for eoi in [false, true] {
                let mut cut = full[..at].to_vec();
                if eoi {
                    cut.extend_from_slice(&[0xFF, 0xD9]);
                }
                let what = format!("progressive {progressive}, cut at {at}, EOI {eoi}");
                let img = decode(&cut, 64, 48).unwrap_or_else(|| panic!("{what}: no image"));
                assert!(img.partial.is_some(), "{what}: reported whole");
            }
        }
    }
}

#[test]
fn a_truncated_stream_is_drawn_and_reported_partial() {
    for progressive in [false, true] {
        let full = rgb_jpeg(64, 48, progressive);
        let cut = &full[..full.len() * 6 / 10];
        let img = decode(cut, 64, 48).expect("lenient decode of a truncated stream");
        assert_eq!((img.width, img.height), (64, 48));
        let why = img.partial.expect("a truncated stream is reported partial");
        assert!(why.contains("incomplete"), "{why}");
    }
}

/// An encoder that optimises its Huffman tables writes a baseline frame
/// with one scan per component; the scans of a flat image are two bytes
/// long.
#[test]
fn a_frame_with_a_scan_per_component_decodes_whole() {
    let colour = [206_u8, 81, 103];
    for (w, h) in [(16_u16, 16_u16), (40, 24), (8, 64)] {
        let mut data = Vec::new();
        let mut enc = Encoder::new(&mut data, 100);
        enc.set_optimized_huffman_tables(true);
        enc.encode(
            &colour.repeat(usize::from(w) * usize::from(h)),
            w,
            h,
            ColorType::Rgb,
        )
        .expect("encode");
        let scans = data.windows(2).filter(|m| m == &[0xFF, 0xDA]).count();
        assert_eq!(scans, 3, "the encoder writes one scan per component");
        let img = decode(&data, u32::from(w), u32::from(h)).expect("decode");
        assert!(img.partial.is_none(), "{:?}", img.partial);
        for px in pixels(&img).as_chunks::<3>().0 {
            for (c, want) in px.iter().zip(colour) {
                assert!(c.abs_diff(want) <= 3, "{w}x{h}: {px:?}");
            }
        }
    }
}

#[test]
fn bytes_before_soi_are_skipped() {
    let full = rgb_jpeg(16, 16, false);
    let reference = decode(&full, 16, 16).expect("reference decode");
    let prefixed = [b"%garbage\r\n".as_slice(), &full].concat();
    let img = decode(&prefixed, 16, 16).expect("decode after leading bytes");
    assert_eq!(pixels(&img), pixels(&reference));
}

#[test]
fn an_unknown_frame_height_takes_the_dictionary_height() {
    let mut data = rgb_jpeg(32, 24, false);
    let at = jpeg_sof(&data).expect("SOF").height_at;
    data[at..at + 2].copy_from_slice(&[0xFF, 0xFF]);
    let img = decode(&data, 32, 24).expect("decode with the dictionary height");
    assert_eq!((img.width, img.height), (32, 24));
    assert!(img.partial.is_none(), "{:?}", img.partial);
    assert!(
        decode(&data, 33, 24).is_none(),
        "an unknown height is refused when the widths differ"
    );
}

/// A frame height of 0 with the line count in a DNL segment after the
/// scan (ITU T.81 B.2.5) takes the dictionary height, and the segment
/// ends the scan after its last row.
#[test]
fn a_dnl_frame_height_takes_the_dictionary_height() {
    let whole = rgb_jpeg(32, 24, false);
    let reference = decode(&whole, 32, 24).expect("reference decode");
    let at = jpeg_sof(&whole).expect("SOF").height_at;
    let eoi = whole.len() - 2;
    let mut data = whole[..eoi].to_vec();
    data[at..at + 2].copy_from_slice(&[0, 0]);
    data.extend_from_slice(&[0xFF, 0xDC, 0x00, 0x04, 0x00, 24]);
    data.extend_from_slice(&whole[eoi..]);
    let img = decode(&data, 32, 24).expect("decode with the dictionary height");
    assert_eq!((img.width, img.height), (32, 24));
    assert_eq!(pixels(&img), pixels(&reference));
    assert!(img.partial.is_none(), "{:?}", img.partial);
    assert!(
        decode(&data, 33, 24).is_none(),
        "an unknown height is refused when the widths differ"
    );
}

#[test]
fn a_frame_taller_than_16384_rows_decodes() {
    let mut out = Vec::new();
    Encoder::new(&mut out, 50)
        .encode(&vec![128u8; 8 * 20_000], 8, 20_000, ColorType::Luma)
        .expect("encode");
    let img = decode(&out, 8, 20_000).expect("tall decode");
    assert_eq!((img.width, img.height), (8, 20_000));
}

#[test]
fn a_frame_over_the_decode_budget_is_refused_before_decoding() {
    let mut data = rgb_jpeg(16, 16, false);
    let at = jpeg_sof(&data).expect("SOF").height_at;
    data[at..at + 4].copy_from_slice(&[0xFF, 0xFE, 0xFF, 0xFE]);
    assert!(decode(&data, 16, 16).is_none());
}

/// A progressive frame whose samples fit the decode budget, but not with
/// the coefficients its decoder holds besides them, is refused before
/// either is allocated.
#[test]
fn a_progressive_frame_whose_coefficients_exceed_the_budget_is_refused() {
    let mut data = Vec::new();
    let mut enc = Encoder::new(&mut data, 85);
    enc.set_progressive(true);
    enc.encode(&[128; 16 * 16], 16, 16, ColorType::Luma)
        .expect("encode");
    let at = jpeg_sof(&data).expect("SOF").height_at;
    // 30 000 × 30 000 gray samples: 0.84 GiB, and twice that in coefficients.
    data[at..at + 4].copy_from_slice(&[0x75, 0x30, 0x75, 0x30]);
    assert!(decode(&data, 30_000, 30_000).is_none());
}
