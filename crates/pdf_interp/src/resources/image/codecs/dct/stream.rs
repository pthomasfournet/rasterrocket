//! The JPEG codestream of a `DCTDecode` stream: the frame-header scan used
//! to route and budget a decode, and the repair of two producer faults
//! before any decoder reads the stream.

use std::borrow::Cow;

/// The JPEG start-of-image marker.
const SOI: [u8; 2] = [0xFF, 0xD8];

/// The frame-header height some producers write when the line count is not
/// known when the header is written; the true count is the dictionary's.
const UNKNOWN_HEIGHT: u16 = 0xFFFF;

/// Frame header fields of a JPEG's first SOF marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct JpegSof {
    /// Number of lines (`Y`); 0 means the height is defined by a later DNL marker.
    pub height: u16,
    /// Number of samples per line (`X`).
    pub width: u16,
    /// Number of image components (`Nf`).
    pub components: u8,
    /// Byte offset of the two-byte `Y` field in the scanned slice.
    pub height_at: usize,
}

/// Scan `data` for the first JPEG SOF marker and return its frame header.
/// Returns `None` if `data` is not a valid JPEG or the SOF payload is too
/// short. Zero allocations; stops at the first SOF found.
///
/// This avoids a full header-decode probe pass before calling the main
/// decode. The `gpu` crate has the same logic but is an optional dependency
/// (requires CUDA at link time), so the CPU path carries its own copy of the
/// scan.
pub(super) fn jpeg_sof(data: &[u8]) -> Option<JpegSof> {
    if data.get(0..2) != Some(&SOI) {
        return None;
    }
    let mut pos = 2usize;
    loop {
        let ff_start = pos;
        while data.get(pos).copied() == Some(0xFF) {
            pos += 1;
        }
        if pos == ff_start {
            return None;
        }
        let &marker = data.get(pos)?;
        pos += 1;
        if matches!(marker, 0x01 | 0xD0..=0xD9) {
            continue;
        }
        let hi = *data.get(pos)?;
        let lo = *data.get(pos + 1)?;
        let seg_len = ((hi as usize) << 8) | (lo as usize);
        if seg_len < 2 {
            return None;
        }
        // SOF layout from the length field: length(2) P(1) Y(2) X(2) Nf(1).
        if matches!(marker, 0xC0..=0xCF) && !matches!(marker, 0xC4 | 0xCC) {
            let frame = data.get(pos + 3..pos + 8)?;
            return Some(JpegSof {
                height: u16::from_be_bytes([frame[0], frame[1]]),
                width: u16::from_be_bytes([frame[2], frame[3]]),
                components: frame[4],
                height_at: pos + 3,
            });
        }
        if marker == 0xDA {
            return None;
        }
        pos = pos.checked_add(seg_len)?;
    }
}

/// Repair the faults of `data` that would otherwise make a decoder refuse
/// the whole image, each the way the stream evidently intends:
///
/// - Bytes before the SOI marker are skipped.
/// - A frame height of `0xFFFF` whose width equals `/Width` is replaced by
///   `/Height` (`pdf_h`): the producer wrote the header before it knew the
///   line count.  The width match keeps an unrelated stream untouched.
///
/// A missing EOI marker is left missing: the decoder reads to the end of the
/// data without it, and an appended marker would hide scan data that ends
/// short of the frame from the strict decode that detects it.  A stream
/// without an SOI marker is no JPEG and is returned unchanged.  The stream
/// is copied only when the height is repaired.
pub(super) fn repair(data: &[u8], pdf_w: u32, pdf_h: u32) -> Cow<'_, [u8]> {
    let Some(start) = data.windows(2).position(|w| w == SOI) else {
        return Cow::Borrowed(data);
    };
    let data = &data[start..];
    let mut bytes = Cow::Borrowed(data);

    if let Some(sof) = jpeg_sof(data)
        && sof.height == UNKNOWN_HEIGHT
        && u32::from(sof.width) == pdf_w
        && let Ok(height) = u16::try_from(pdf_h)
        && height != 0
        && height != UNKNOWN_HEIGHT
    {
        log::debug!("image: DCTDecode: frame height 0xFFFF taken as /Height {height}");
        bytes.to_mut()[sof.height_at..sof.height_at + 2].copy_from_slice(&height.to_be_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::{JpegSof, jpeg_sof, repair};

    /// SOI, an APP0 segment to skip, then a baseline SOF0 frame header.
    const FRAME: &[u8] = &[
        0xFF, 0xD8, // SOI
        0xFF, 0xE0, 0x00, 0x04, 0x00, 0x00, // APP0, length 4
        0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x01, 0x2C, 0x02, 0x80, 0x01, 0x01, 0x11, 0x00, // SOF0
    ];

    /// [`FRAME`] followed by an EOI marker.
    fn complete() -> Vec<u8> {
        [FRAME, &[0xFF, 0xD9]].concat()
    }

    #[test]
    fn sof_yields_the_frame_dimensions() {
        assert_eq!(
            jpeg_sof(FRAME),
            Some(JpegSof {
                height: 300,
                width: 640,
                components: 1,
                height_at: 13,
            })
        );
    }

    #[test]
    fn truncated_sof_yields_none() {
        assert_eq!(jpeg_sof(&FRAME[..FRAME.len() - 7]), None);
        assert_eq!(jpeg_sof(b"not a jpeg"), None);
    }

    #[test]
    fn a_sound_stream_is_borrowed_unchanged() {
        let data = complete();
        let r = repair(&data, 640, 300);
        assert!(matches!(r, std::borrow::Cow::Borrowed(_)));
        assert_eq!(&*r, data.as_slice());
    }

    #[test]
    fn leading_bytes_are_skipped() {
        let data = [b"junk\n".as_slice(), &complete()].concat();
        let r = repair(&data, 640, 300);
        assert_eq!(&*r, complete().as_slice());
        assert!(matches!(r, std::borrow::Cow::Borrowed(_)));
    }

    #[test]
    fn an_unknown_height_takes_the_dictionary_height() {
        let mut data = complete();
        data[13..15].copy_from_slice(&[0xFF, 0xFF]);
        let r = repair(&data, 640, 300);
        assert_eq!(jpeg_sof(&r).map(|s| s.height), Some(300));
    }

    #[test]
    fn an_unknown_height_stays_when_the_width_differs() {
        let mut data = complete();
        data[13..15].copy_from_slice(&[0xFF, 0xFF]);
        for (w, h) in [(641, 300), (640, 0), (640, 0xFFFF), (640, 70_000)] {
            let r = repair(&data, w, h);
            assert_eq!(jpeg_sof(&r).map(|s| s.height), Some(0xFFFF), "{w}×{h}");
        }
    }

    #[test]
    fn a_stream_without_soi_is_left_alone() {
        for data in [b"not a jpeg".as_slice(), b"", b"\xFF"] {
            let r = repair(data, 1, 1);
            assert_eq!(&*r, data);
        }
    }
}
