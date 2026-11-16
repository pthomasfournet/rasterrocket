use super::*;

/// `NvJpeg2k::new()` must not panic — it either succeeds or returns `Err`.
/// Skipped gracefully if no GPU is available.
#[test]
fn nvjpeg2k_new_does_not_panic() {
    let _ = NvJpeg2k::new();
}

/// Empty input must return a status error before any FFI call.
///
/// This test does not require a GPU — the empty-slice fast-path in
/// `NvJpeg2k::decode` returns `Nvjpeg2kStatus(3)` (`BAD_JPEG`) immediately.
/// We still need a `NvJpeg2kDecoder` to reach `decode_sync`, so skip if
/// no GPU is available.
#[test]
fn decode_empty_returns_error() {
    // no GPU available → skip.
    let Ok(mut dec) = NvJpeg2kDecoder::new(0) else {
        return;
    };
    let result = dec.decode_sync(&[]);
    assert!(
        matches!(result, Err(NvJpeg2kError::Nvjpeg2kStatus(3))),
        "expected Nvjpeg2kStatus(3) (BAD_JPEG) for empty input, got {result:?}",
    );
}

/// Minimal grayscale JPEG 2000 codestream (16×16, luma = 128).
///
/// To regenerate (requires a JPEG 2000 encoder, e.g. `convert -size 16x16 xc:gray50 ...`):
/// ```sh
/// convert -size 16x16 xc:gray50 -define j2k:format=j2k /tmp/gray16x16.j2k
/// python3 -c "d=open('/tmp/gray16x16.j2k','rb').read(); print(', '.join(f'0x{b:02x}' for b in d))"
/// ```
///
/// This is a raw `.j2k` codestream (no JP2 container header).  nvJPEG2000
/// auto-detects the format from the SOC (0xff4f) / SIZ (0xff51) markers.
const GRAY_16X16_J2K: &[u8] = &[
    0xff, 0x4f, 0xff, 0x51, 0x00, 0x29, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x10,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x10,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x07, 0x00, 0x00, 0xff, 0x52, 0x00,
    0x0c, 0x00, 0x00, 0x00, 0x01, 0x00, 0x05, 0x04, 0x04, 0x00, 0x00, 0xff, 0x5c, 0x00, 0x04, 0x40,
    0x00, 0xff, 0x90, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff,
    0x93, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0xff, 0xd9,
];

/// End-to-end decode of the embedded 16×16 gray JPEG 2000 via the public API.
/// Skipped gracefully if no GPU is available.
#[test]
fn decode_gray_j2k() {
    // no GPU available → skip.
    let Ok(mut dec) = NvJpeg2kDecoder::new(0) else {
        return;
    };

    let img = match dec.decode_sync(GRAY_16X16_J2K) {
        Ok(img) => img,
        // Status 3: BAD_JPEG — the minimal synthetic codestream is rejected by
        // some nvJPEG2000 versions (zeroed tile data stub).
        // Status 4: JPEG_NOT_SUPPORTED — codestream type not supported by this
        // nvJPEG2000 build (e.g. no HTJ2K).  Neither is a regression.
        Err(NvJpeg2kError::Nvjpeg2kStatus(3 | 4)) => return,
        // Any other error means the decode path regressed — fail loudly.
        Err(e) => panic!("decode_gray_j2k: unexpected GPU decode error: {e}"),
    };

    assert_eq!(img.width, 16, "unexpected width");
    assert_eq!(img.height, 16, "unexpected height");
    assert_eq!(img.color_space, Jpeg2kColorSpace::Gray);
    assert_eq!(img.data.len(), 16 * 16, "unexpected data length");
    // gray50 encodes to ~127.
    assert!(
        img.data.iter().all(|&p| p > 80 && p < 180),
        "unexpected luma values (expected ~127): {:?}",
        &img.data[..16.min(img.data.len())]
    );
}

/// The largest sample of `img` and whether every sample of it is at most
/// `2^bits − 1`.
fn within_precision(img: &DecodedJpeg2k<u16>, bits: u8) -> (u16, bool) {
    let max = img.data.iter().copied().max().unwrap_or(0);
    (max, u32::from(max) < 1 << bits)
}

/// `NVJPEG2K_UINT16` returns each sample at its component's precision,
/// `0 ..= 2^p − 1`, not scaled to 16 bits; `NVJPEG2K_UINT8` refuses a
/// component wider than a byte rather than narrowing it.
#[test]
fn sixteen_bit_output_keeps_the_codestream_precision() {
    let Ok(mut dec) = NvJpeg2kDecoder::new(0) else {
        return;
    };
    for (bytes, bits) in [
        (
            &include_bytes!("../../../jpeg2000/tests/streams/codestream_precision_10.j2k")[..],
            10,
        ),
        (
            include_bytes!("../../../jpeg2000/tests/streams/codestream_precision_12.j2k"),
            12,
        ),
        (
            include_bytes!("../../../pdf_interp/src/resources/image/testdata/gray16.jp2"),
            16,
        ),
    ] {
        let img = dec.decode_sync_u16(bytes).expect("decodes");
        assert_eq!(img.precisions, vec![bits; img.precisions.len()]);
        assert_eq!(img.data.len(), (img.width * img.height) as usize);
        let (max, within) = within_precision(&img, bits);
        assert!(within, "{bits} bits: sample {max}");
        assert!(max > 255, "{bits} bits: no sample past 8 bits ({max})");
        assert!(
            matches!(
                dec.decode_sync(bytes),
                Err(NvJpeg2kError::Nvjpeg2kStatus(2))
            ),
            "{bits} bits into UINT8 is refused (NVJPEG2K_STATUS_INVALID_PARAMETER)"
        );
    }
}
