//! `JBIG2Decode` through `hayro-jbig2`, with `JBIG2Globals` decoded once
//! per document.

use std::sync::Arc;

use hayro_jbig2::Decoder as Jbig2Decoder;
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
    let mut collector = Jbig2Collector::new(n_pixels);

    img.decode(&mut collector)
        .map_err(|e| log::warn!("image: JBIG2Decode decode error: {e}"))
        .ok()?;

    if collector.data.len() != n_pixels {
        log::warn!(
            "image: JBIG2Decode produced {} pixels, expected {n_pixels} — skipping",
            collector.data.len()
        );
        return None;
    }

    let layout = if is_mask {
        Layout::Mask
    } else {
        Layout::Color(1)
    };
    Some(Samples::cpu(jw, jh, layout, collector.data))
}

/// Pixel collector for `hayro_jbig2::Decoder`.
///
/// JBIG2 convention: 0 = white, 1 = black.
/// `ImageColorSpace::Gray` convention: 0x00 = black, 0xFF = white.
/// `ImageColorSpace::Mask` convention: 0x00 = paint (== JBIG2 black), 0xFF = transparent.
///
/// Both output conventions share the same polarity flip: JBIG2 black (1) → 0x00,
/// JBIG2 white (0) → 0xFF.  `ImageColorSpace` selection is handled by the caller.
struct Jbig2Collector {
    data: Vec<u8>,
    /// Total expected pixel count — used to cap `push_pixel_chunk` so a stream
    /// whose run lengths aren't a multiple of 8 doesn't overflow the buffer.
    n_pixels: usize,
    /// The page buffer could not be reserved; every pixel is dropped, so
    /// the collected length falls short and the image is refused.
    alloc_failed: bool,
}

impl Jbig2Collector {
    const fn new(n_pixels: usize) -> Self {
        Self {
            data: Vec::new(),
            n_pixels,
            alloc_failed: false,
        }
    }

    /// Reserves the whole page on the first emitted pixel, and reports
    /// whether the page buffer is held.  The decoder emits only after it
    /// has allocated the page bitmap, so `n_pixels` — taken from the
    /// stream's page information segment — is known to be a dimension the
    /// decoder accepted by the time it is reserved.
    fn reserve_page(&mut self) -> bool {
        if self.data.capacity() == 0
            && !self.alloc_failed
            && self.n_pixels > 0
            && let Err(e) = self.data.try_reserve_exact(self.n_pixels)
        {
            log::warn!(
                "image: JBIG2Decode: cannot allocate {} bytes: {e}",
                self.n_pixels
            );
            self.alloc_failed = true;
        }
        !self.alloc_failed
    }
}

impl Jbig2Decoder for Jbig2Collector {
    fn push_pixel(&mut self, black: bool) {
        if self.reserve_page() && self.data.len() < self.n_pixels {
            self.data.push(if black { 0x00 } else { 0xFF });
        }
    }

    fn push_pixel_chunk(&mut self, black: bool, chunk_count: u32) {
        if !self.reserve_page() {
            return;
        }
        let byte = if black { 0x00 } else { 0xFF };
        // chunk_count is in 8-pixel chunks; cap to the remaining pixels so that
        // run lengths that aren't an exact multiple of 8 don't produce a short
        // buffer (which would cause the caller to discard the image entirely).
        let remaining = self.n_pixels.saturating_sub(self.data.len());
        let n = (chunk_count as usize).saturating_mul(8).min(remaining);
        self.data.extend(std::iter::repeat_n(byte, n));
    }

    fn next_line(&mut self) {
        // Row boundary — nothing to do; pixels are already stored flat.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jbig2_collector_push_pixel_grayscale() {
        // JBIG2: black=true → Gray 0x00, black=false → Gray 0xFF.
        let mut c = Jbig2Collector::new(2);
        Jbig2Decoder::push_pixel(&mut c, true);
        Jbig2Decoder::push_pixel(&mut c, false);
        assert_eq!(c.data, [0x00, 0xFF]);
    }

    #[test]
    fn jbig2_collector_push_pixel_chunk() {
        let mut c = Jbig2Collector::new(16);
        // chunk_count=2 → 16 pixels of white (0xFF each).
        Jbig2Decoder::push_pixel_chunk(&mut c, false, 2);
        assert_eq!(c.data.len(), 16);
        assert!(c.data.iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn jbig2_collector_push_pixel_chunk_overrun() {
        // A chunk whose pixel count overshoots n_pixels (a page width that
        // is not a multiple of 8) is capped at the page.
        let mut c = Jbig2Collector::new(10);
        // chunk_count=2 → 16 pixels, but n_pixels=10 — must cap at 10.
        Jbig2Decoder::push_pixel_chunk(&mut c, false, 2);
        assert_eq!(c.data.len(), 10);
        assert!(c.data.iter().all(|&b| b == 0xFF));
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
