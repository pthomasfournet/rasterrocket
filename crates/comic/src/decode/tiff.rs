//! TIFF decode for comic/scan pages, via the pure-Rust `tiff` crate.
//!
//! A TIFF file can hold several pages, one per image file directory (IFD);
//! [`pages`] lists the directories that are pages, and [`decode`] decodes one.
//! A directory marked as a reduced-resolution copy or a transparency mask
//! (`NewSubfileType` bits 0 and 2) is no page.  The crate returns `WhiteIsZero`
//! samples inverted to `BlackIsZero`, gray samples of fewer than eight bits
//! packed most significant first with each row padded to a byte, and a planar
//! image (`PlanarConfiguration` 2) one plane after another.

use tiff::ColorType;
use tiff::decoder::{Decoder, DecodingResult};
use tiff::tags::Tag;

use super::{
    CodecPixels, DecodeError, gray_from_tight, guard_dimensions, keep_leading_samples,
    rgb_from_tight, unpack_gray_bits,
};

/// `PlanarConfiguration` value of an image stored one plane per component.
const PLANAR: u32 = 2;

/// `NewSubfileType` bits of a directory that is no page of its own: a
/// reduced-resolution copy (bit 0) or a transparency mask (bit 2).
const NOT_A_PAGE: u32 = 0b101;

/// The most pages read from one TIFF; the directories after it are ignored
/// with a warning.
const MAX_PAGES: usize = 10_000;

/// The directory indices of the pages of a TIFF, in file order.  A file
/// whose every directory is marked as no page yields its first directory.
/// A directory chain that breaks after the first page ends the list there,
/// with a warning; the tiff decoder refuses a chain that loops.
///
/// # Errors
///
/// [`DecodeError::Codec`] when the header or the first directory is
/// unreadable.
pub(super) fn pages(bytes: &[u8]) -> Result<Vec<usize>, DecodeError> {
    let codec = |e: tiff::TiffError| DecodeError::Codec(format!("tiff: {e}"));
    let mut dec = Decoder::new(std::io::Cursor::new(bytes)).map_err(codec)?;
    let mut pages = Vec::new();
    for ifd in 0.. {
        let subfile = dec.get_tag_u32(Tag::NewSubfileType).unwrap_or(0);
        if subfile & NOT_A_PAGE == 0 {
            if pages.len() == MAX_PAGES {
                log::warn!("tiff: more than {MAX_PAGES} pages; the rest are not read");
                break;
            }
            pages.push(ifd);
        }
        if !dec.more_images() {
            break;
        }
        if let Err(e) = dec.next_image() {
            log::warn!(
                "tiff: directory {} unreadable ({e}); the pages after it are not read",
                ifd + 1
            );
            break;
        }
    }
    if pages.is_empty() {
        pages.push(0);
    }
    Ok(pages)
}

/// Decode the page in image file directory `ifd` (an index [`pages`]
/// lists): a gray page of 1 to 16 bits per sample yields a `Gray8` bitmap of
/// its first sample, an RGB page an `Rgb8` bitmap with any alpha channel
/// dropped.
///
/// # Errors
///
/// [`DecodeError::TooLarge`] when the header dimensions exceed the page
/// limits; [`DecodeError::Codec`] on malformed or unsupported streams.
pub(super) fn decode(bytes: &[u8], ifd: usize) -> Result<CodecPixels, DecodeError> {
    let codec = |e: tiff::TiffError| DecodeError::Codec(format!("tiff: {e}"));
    let mut dec = Decoder::new(std::io::Cursor::new(bytes)).map_err(codec)?;
    if ifd > 0 {
        dec.seek_to_image(ifd).map_err(codec)?;
    }
    let (w, h) = dec.dimensions().map_err(codec)?;
    if w == 0 || h == 0 {
        return Err(DecodeError::Codec(format!("tiff: zero dimensions {w}x{h}")));
    }
    guard_dimensions(w, h)?;
    let color = dec.colortype().map_err(codec)?;
    let planar = dec.get_tag_u32(Tag::PlanarConfiguration).ok() == Some(PLANAR);

    let mut img = DecodingResult::U8(Vec::new());
    let layout = dec.read_image_to_buffer(&mut img).map_err(codec)?;
    let mut u8s: Vec<u8> = match img {
        DecodingResult::U8(v) => v,
        // 16-bit → 8-bit: keep the high byte (the standard sample reduction).
        DecodingResult::U16(v) => v.iter().map(|&x| (x >> 8) as u8).collect(),
        _ => {
            return Err(DecodeError::Codec(
                "tiff: unsupported sample format (expected 8/16-bit integer)".to_owned(),
            ));
        }
    };
    // A gray image with extra samples (alpha among them) is `Multiband`.
    let (channels, bits) = match color {
        ColorType::Gray(bits) => (1, bits),
        ColorType::Multiband {
            bit_depth,
            num_samples,
        } => (usize::from(num_samples), bit_depth),
        ColorType::RGB(bits) => (3, bits),
        ColorType::RGBA(bits) => (4, bits),
        other => {
            return Err(DecodeError::Codec(format!(
                "tiff: unsupported color type {other:?}"
            )));
        }
    };
    if planar && channels > 1 {
        let complete = layout.complete_len / usize::from(bits.div_ceil(8));
        if u8s.len() < complete {
            return Err(DecodeError::Codec(format!(
                "tiff: planar image of {complete} samples exceeds the decoding buffer limit"
            )));
        }
        u8s = interleave_planes(&u8s, channels);
    }
    match (color, bits) {
        (ColorType::Gray(_), 1 | 2 | 4) => {
            gray_from_tight(w, h, unpack_gray_bits(&u8s, w, h, bits)?)
        }
        (ColorType::Gray(_), _) => gray_from_tight(w, h, u8s),
        (ColorType::Multiband { .. }, 8 | 16) => {
            keep_leading_samples(&mut u8s, channels, 1);
            gray_from_tight(w, h, u8s)
        }
        (ColorType::RGB(_), 8 | 16) => rgb_from_tight(w, h, u8s),
        (ColorType::RGBA(_), 8 | 16) => {
            keep_leading_samples(&mut u8s, 4, 3);
            rgb_from_tight(w, h, u8s)
        }
        (other, bits) => Err(DecodeError::Codec(format!(
            "tiff: unsupported color type {other:?} at {bits} bits per sample"
        ))),
    }
}

/// Interleave `channels` equal planes stored one after another into
/// chunky pixels.
fn interleave_planes(planes: &[u8], channels: usize) -> Vec<u8> {
    let plane = planes.len() / channels;
    let mut out = vec![0; plane * channels];
    for (c, src) in planes.chunks_exact(plane).enumerate() {
        for (dst, &s) in out[c..].iter_mut().step_by(channels).zip(src) {
            *dst = s;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::interleave_planes;

    #[test]
    fn planes_interleave_into_pixels() {
        assert_eq!(
            interleave_planes(&[1, 2, 3, 10, 20, 30], 2),
            [1, 10, 2, 20, 3, 30]
        );
    }
}
