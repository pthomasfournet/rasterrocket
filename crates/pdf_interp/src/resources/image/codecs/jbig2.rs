//! `JBIG2Decode` through `hayro-jbig2`, with `JBIG2Globals` decoded once
//! per document.

use std::sync::Arc;

use pdf::{Document, Object};

use crate::resources::image::colorspace::{Layout, Samples};

// ── JBIG2Decode ──────────────────────────────────────────────────────────────

/// Decode a `JBIG2Decode` stream via `hayro-jbig2`.
///
/// PDF JBIG2 uses the "embedded organisation" (Annex D.3 of T.88): the stream
/// contains page segments only; global segments live in a separate
/// `JBIG2Globals` stream referenced from `DecodeParms`.
///
/// The decoder produces one byte per pixel: 0x00 = black, 0xFF = white (for
/// grayscale images) or 0x00 = paint, 0xFF = transparent (for `ImageMask`).
/// JBIG2 convention is 0 = white, 1 = black, inverted here to the image
/// pipeline's convention.
pub(in crate::resources::image) fn decode_jbig2(
    doc: &Document,
    data: &[u8],
    width: u32,
    height: u32,
    is_mask: bool,
    parms: Option<&Object>,
) -> Option<Samples> {
    // Resolve the optional JBIG2Globals stream from DecodeParms.  Its
    // segments are decoded once per document and shared by every image
    // that refers to the stream.
    let settings = hayro_jbig2::DecodeSettings::default();
    let globals: Option<Arc<Option<hayro_jbig2::Globals>>> = match parms
        .and_then(|o| o.as_dict())
        .and_then(|d| d.get(b"JBIG2Globals"))
    {
        None => None,
        // `Some(None)` is an unreadable globals stream, which the image is
        // decoded without; `None` is a globals stream that fails to parse,
        // which fails the image.
        Some(&Object::Reference(id)) => Some(doc.derived(id, || {
            let Some(g_bytes) = doc
                .get_object(id)
                .ok()
                .and_then(|g| g.as_stream()?.decompressed_content().ok())
            else {
                return Some(None);
            };
            hayro_jbig2::Globals::new(&g_bytes, &settings)
                .map_err(|e| log::warn!("image: JBIG2Decode globals parse error: {e}"))
                .ok()
                .map(Some)
        })?),
        Some(_) => {
            // PDF spec requires JBIG2Globals to be an indirect reference to
            // a stream object; any other form is malformed.
            log::warn!(
                "image: JBIG2Decode: JBIG2Globals is not an indirect reference — \
                 ignoring globals"
            );
            None
        }
    };
    let img = hayro_jbig2::Page::new_embedded(
        data,
        globals.as_deref().and_then(Option::as_ref),
        &settings,
    )
    .map_err(|e| log::warn!("image: JBIG2Decode parse error: {e}"))
    .ok()?;

    let jw = img.width();
    let jh = img.height();

    // Validate decoded dimensions against the PDF image dict.
    if jw != width || jh != height {
        log::warn!(
            "image: JBIG2Decode dimension mismatch: image dict says {width}×{height}, JBIG2 stream says {jw}×{jh} — using stream dimensions"
        );
    }

    // The page size of the stream, not the dictionary's, sizes the
    // decoder's bitmap and the collected samples.
    if let Err(e) = crate::resources::image::decode_budget(jw, jh, 1) {
        log::warn!("image: JBIG2Decode page {e}");
        return None;
    }
    let n_pixels = (jw as usize).checked_mul(jh as usize)?;
    let mut ctx = hayro_jbig2::DecoderContext::default();
    let bitmap = img
        .decode(&mut ctx)
        .map_err(|e| log::warn!("image: JBIG2Decode decode error: {e}"))
        .ok()?;
    if (bitmap.width(), bitmap.height()) != (jw, jh) {
        log::warn!(
            "image: JBIG2Decode produced a {}×{} bitmap, expected {jw}×{jh} — skipping",
            bitmap.width(),
            bitmap.height()
        );
        return None;
    }
    let mut data = crate::resources::image::try_with_capacity(n_pixels)?;
    data.resize(n_pixels, 0xFF);
    expand_rows(bitmap.rows(), jw as usize, &mut data);

    let layout = if is_mask {
        Layout::Mask
    } else {
        Layout::Color(1)
    };
    Some(Samples::cpu(jw, jh, layout, data))
}

/// The 8 samples of a byte of 8 packed pixels, first pixel in the most
/// significant bit: a set bit (JBIG2 black) is 0x00, a clear one 0xFF.
static BYTE_SAMPLES: [[u8; 8]; 256] = {
    let mut table = [[0xFF; 8]; 256];
    let mut byte = 0;
    while byte < 256 {
        let mut bit = 0;
        while bit < 8 {
            if byte & (0x80 >> bit) != 0 {
                table[byte][bit] = 0x00;
            }
            bit += 1;
        }
        byte += 1;
    }
    table
};

/// Write the pixels of `rows` (JBIG2 words, 1 = black, first pixel in the
/// most significant bit, bits past `width` ignored) into `out`, `width`
/// samples a row, which holds 0xFF on entry.
///
/// Both output conventions share one polarity: `ImageColorSpace::Gray`
/// (0x00 = black) and `ImageColorSpace::Mask` (0x00 = paint) take JBIG2
/// black (1) to 0x00 and white (0) to 0xFF.
fn expand_rows<'a>(rows: impl Iterator<Item = &'a [u32]>, width: usize, out: &mut [u8]) {
    if width == 0 {
        return;
    }
    for (row, out) in rows.zip(out.chunks_exact_mut(width)) {
        for (&word, out) in row.iter().zip(out.chunks_mut(32)) {
            match word {
                0 => {}
                u32::MAX => out.fill(0x00),
                _ => {
                    for (byte, out) in word.to_be_bytes().into_iter().zip(out.chunks_mut(8)) {
                        if byte != 0 {
                            out.copy_from_slice(&BYTE_SAMPLES[usize::from(byte)][..out.len()]);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Expand rows given as strings of `'0'`/`'1'`, one word per 32 pixels,
    /// with `pad` set past the width.
    fn expand(rows: &[&str], pad: bool) -> Vec<u8> {
        let width = rows[0].len();
        let words: Vec<Vec<u32>> = rows
            .iter()
            .map(|r| {
                let mut row = vec![if pad { u32::MAX } else { 0 }; width.div_ceil(32)];
                for (x, c) in r.bytes().enumerate() {
                    let bit = 0x8000_0000 >> (x % 32);
                    if c == b'1' {
                        row[x / 32] |= bit;
                    } else {
                        row[x / 32] &= !bit;
                    }
                }
                row
            })
            .collect();
        let mut out = vec![0xFF; width * rows.len()];
        expand_rows(words.iter().map(Vec::as_slice), width, &mut out);
        out
    }

    /// The samples of `rows`, pixel by pixel: black (1) is 0x00.
    fn reference(rows: &[&str]) -> Vec<u8> {
        rows.iter()
            .flat_map(|r| r.bytes().map(|c| if c == b'1' { 0x00 } else { 0xFF }))
            .collect()
    }

    #[test]
    fn expand_rows_matches_pixel_by_pixel() {
        // Widths around the byte and word edges; all-white, all-black and
        // mixed words and bytes; padding bits set and clear.
        for width in [1, 7, 8, 9, 31, 32, 33, 64, 70] {
            let patterns: [&dyn Fn(usize) -> bool; 4] =
                [&|_| false, &|_| true, &|x| x % 3 == 0, &|x| {
                    (x / 8) % 2 == 0
                }];
            let rows: Vec<String> = patterns
                .iter()
                .map(|p| (0..width).map(|x| if p(x) { '1' } else { '0' }).collect())
                .collect();
            let rows: Vec<&str> = rows.iter().map(String::as_str).collect();
            for pad in [false, true] {
                assert_eq!(expand(&rows, pad), reference(&rows), "width {width}");
            }
        }
    }

    #[test]
    fn jbig2_decode_invalid_data_returns_none() {
        // Corrupt/empty JBIG2 data must not panic — it must return None.
        let doc = crate::test_helpers::empty_doc();
        let result = decode_jbig2(&doc, b"\x00\x01\x02\x03", 4, 4, false, None);
        assert!(result.is_none());
    }

    #[test]
    fn jbig2_decode_oversized_page_returns_none() {
        // A page information segment declaring a 2^32 × 2^28 page: the
        // decoder rejects the bitmap, and no buffer of the declared size is
        // reserved before it does.
        let mut data = vec![
            0x00, 0x00, 0x00, 0x00, // segment number 0
            0x30, // type 48: page information
            0x00, // no referred-to segments
            0x01, // page association 1
            0x00, 0x00, 0x00, 0x13, // data length 19
        ];
        data.extend_from_slice(&0xFFFF_FFF0_u32.to_be_bytes()); // width
        data.extend_from_slice(&0x1000_0000_u32.to_be_bytes()); // height
        data.extend_from_slice(&[0; 8]); // resolutions unknown
        data.extend_from_slice(&[0x00, 0x00, 0x00]); // flags, striping
        let doc = crate::test_helpers::empty_doc();
        assert!(decode_jbig2(&doc, &data, 4, 4, false, None).is_none());
    }

    #[test]
    fn jbig2_page_over_the_decode_budget_returns_none() {
        // A 65 535 × 65 535 page, within the decoder's limits but 4 GiB of
        // samples, under a dictionary that declares 4 × 4: the stream's
        // page size is held to the budget before the decoder allocates.
        let mut data = vec![
            0x00, 0x00, 0x00, 0x00, // segment number 0
            0x30, // type 48: page information
            0x00, // no referred-to segments
            0x01, // page association 1
            0x00, 0x00, 0x00, 0x13, // data length 19
        ];
        data.extend_from_slice(&0xFFFF_u32.to_be_bytes()); // width
        data.extend_from_slice(&0xFFFF_u32.to_be_bytes()); // height
        data.extend_from_slice(&[0; 8]); // resolutions unknown
        data.extend_from_slice(&[0x00, 0x00, 0x00]); // flags, striping
        let doc = crate::test_helpers::empty_doc();
        assert!(decode_jbig2(&doc, &data, 4, 4, false, None).is_none());
    }
}
