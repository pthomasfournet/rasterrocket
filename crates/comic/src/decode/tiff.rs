//! TIFF decode for comic/scan pages, via the pure-Rust `tiff` crate.
//!
//! A TIFF file can hold several pages, one per image file directory (IFD);
//! [`pages`] walks the directory chain once and lists the offsets of the
//! directories that are pages, and [`decode`] decodes one, opening the file
//! at that offset so no directory before it is read again.
//! A directory marked as a reduced-resolution copy or a transparency mask
//! (`NewSubfileType` bits 0 and 2) is no page.  The crate returns `WhiteIsZero`
//! samples inverted to `BlackIsZero`, gray samples of fewer than eight bits
//! packed most significant first with each row padded to a byte, and a planar
//! image (`PlanarConfiguration` 2) one plane after another.

use std::io::{Read, Seek, SeekFrom};

use tiff::ColorType;
use tiff::decoder::{Decoder, DecodingResult};
use tiff::tags::Tag;

use super::{
    CodecPixels, DecodeError, Target, gray_from_tight, gray16_from_tight, guard_dimensions,
    keep_leading_samples, rgb_from_tight, rgb16_from_tight, unpack_gray_bits,
};

/// `PlanarConfiguration` value of an image stored one plane per component.
const PLANAR: u32 = 2;

/// `NewSubfileType` bits of a directory that is no page of its own: a
/// reduced-resolution copy (bit 0) or a transparency mask (bit 2).
const NOT_A_PAGE: u32 = 0b101;

/// The most pages read from one TIFF; the directories after it are ignored
/// with a warning.
const MAX_PAGES: usize = 10_000;

/// The most directories, pages or not, read from one TIFF; the directories
/// after it are ignored with a warning.  Bounds the walk of a chain of
/// directories that are no pages.
const MAX_DIRECTORIES: usize = 2 * MAX_PAGES;

/// The key [`pages`] gives the first directory: the one the file header
/// names, read without rebasing.  No directory lies at offset 0, where the
/// header is.
const FIRST_DIRECTORY: usize = 0;

fn codec(e: impl std::fmt::Display) -> DecodeError {
    DecodeError::Codec(format!("tiff: {e}"))
}

/// The pages of a TIFF, in file order, each named by the key [`decode`]
/// takes: the byte offset of its directory, or [`FIRST_DIRECTORY`] for the
/// first.  A file whose every directory is marked as no page yields its
/// first directory.  A directory chain that breaks after the first page ends
/// the list there, with a warning; the tiff decoder refuses a chain that
/// loops.  At most [`MAX_DIRECTORIES`] directories are walked.
///
/// # Errors
///
/// [`DecodeError::Codec`] when the header or the first directory is
/// unreadable.
pub(super) fn pages(bytes: &[u8]) -> Result<Vec<usize>, DecodeError> {
    let mut dec = Decoder::new(std::io::Cursor::new(bytes)).map_err(codec)?;
    let mut pages = Vec::new();
    for ifd in 0.. {
        if ifd == MAX_DIRECTORIES {
            log::warn!("tiff: more than {MAX_DIRECTORIES} directories; the rest are not read");
            break;
        }
        let subfile = dec.get_tag_u32(Tag::NewSubfileType).unwrap_or(0);
        if subfile & NOT_A_PAGE == 0 {
            if pages.len() == MAX_PAGES {
                log::warn!("tiff: more than {MAX_PAGES} pages; the rest are not read");
                break;
            }
            let key = if ifd == 0 {
                Some(FIRST_DIRECTORY)
            } else {
                dec.ifd_pointer()
                    .and_then(|pointer| usize::try_from(pointer.0).ok())
            };
            let Some(key) = key else {
                log::warn!(
                    "tiff: directory {ifd} lies past the addressable range; \
                     the pages from it on are not read"
                );
                break;
            };
            pages.push(key);
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

/// Decode the page whose directory is `page` (a key [`pages`] lists): the
/// file is read as if its header named that directory first.  A gray page
/// of 1 to 16 bits per sample yields a gray bitmap of its first sample, an
/// RGB page an RGB bitmap with any alpha channel dropped; 16-bit samples
/// are kept on a 16-bit target (`Gray16`, `Rgb16`) and reduced to their
/// high byte otherwise.
///
/// # Errors
///
/// [`DecodeError::TooLarge`] when the header dimensions exceed the page
/// limits; [`DecodeError::Codec`] on malformed or unsupported streams.
pub(super) fn decode(
    bytes: &[u8],
    page: usize,
    target: Target,
) -> Result<CodecPixels, DecodeError> {
    if page == FIRST_DIRECTORY {
        decode_first(
            Decoder::new(std::io::Cursor::new(bytes)).map_err(codec)?,
            target,
        )
    } else {
        decode_first(
            Decoder::new(Rebased::new(bytes, page as u64)?).map_err(codec)?,
            target,
        )
    }
}

/// Decode the current (first) image of `dec`.
fn decode_first<R: Read + Seek>(
    mut dec: Decoder<R>,
    target: Target,
) -> Result<CodecPixels, DecodeError> {
    let (w, h) = dec.dimensions().map_err(codec)?;
    if w == 0 || h == 0 {
        return Err(DecodeError::Codec(format!("tiff: zero dimensions {w}x{h}")));
    }
    guard_dimensions(w, h, target)?;
    let color = dec.colortype().map_err(codec)?;
    let planar = dec.get_tag_u32(Tag::PlanarConfiguration).ok() == Some(PLANAR);

    let mut img = DecodingResult::U8(Vec::new());
    let layout = dec.read_image_to_buffer(&mut img).map_err(codec)?;
    let samples = match img {
        DecodingResult::U8(v) => Samples::U8(v),
        DecodingResult::U16(v) if target.deep() => Samples::U16(v),
        // 16-bit → 8-bit: keep the high byte (the standard sample reduction).
        DecodingResult::U16(v) => Samples::U8(v.iter().map(|&x| (x >> 8) as u8).collect()),
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
    let planar = planar && channels > 1;
    let complete = layout.complete_len / usize::from(bits.div_ceil(8));
    let unsupported = || {
        DecodeError::Codec(format!(
            "tiff: unsupported color type {color:?} at {bits} bits per sample"
        ))
    };
    let mut u8s = match samples {
        Samples::U8(v) => chunky(v, channels, planar, complete)?,
        Samples::U16(v) => {
            let mut v = chunky(v, channels, planar, complete)?;
            return match (color, bits) {
                (ColorType::Gray(_), 16) => gray16_from_tight(w, h, v),
                (ColorType::Multiband { .. }, 16) => {
                    keep_leading_samples(&mut v, channels, 1);
                    gray16_from_tight(w, h, v)
                }
                (ColorType::RGB(_), 16) => rgb16_from_tight(w, h, v),
                (ColorType::RGBA(_), 16) => {
                    keep_leading_samples(&mut v, 4, 3);
                    rgb16_from_tight(w, h, v)
                }
                _ => Err(unsupported()),
            };
        }
    };
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
        _ => Err(unsupported()),
    }
}

/// The decoded samples of an image: 8-bit, or 16-bit kept for a 16-bit
/// target.
enum Samples {
    U8(Vec<u8>),
    U16(Vec<u16>),
}

/// `samples` in chunky order: as they are, or for a `planar` image of
/// `channels` planes interleaved, after checking that the decoding buffer
/// held all `complete` samples.
fn chunky<T: Copy + Default>(
    samples: Vec<T>,
    channels: usize,
    planar: bool,
    complete: usize,
) -> Result<Vec<T>, DecodeError> {
    if !planar {
        return Ok(samples);
    }
    if samples.len() < complete {
        return Err(DecodeError::Codec(format!(
            "tiff: planar image of {complete} samples exceeds the decoding buffer limit"
        )));
    }
    Ok(interleave_planes(&samples, channels))
}

/// A TIFF file read with the first-directory offset of its header replaced,
/// so that a decoder opened on it starts at any directory of the file
/// without reading the ones before it.  Every other byte reads as in the
/// file.
struct Rebased<'a> {
    file: &'a [u8],
    /// The file's header with the offset replaced: 8 bytes, or 16 for a
    /// `BigTIFF`.
    header: [u8; 16],
    header_len: usize,
    pos: u64,
}

impl<'a> Rebased<'a> {
    /// `file` with its header naming the directory at `offset` first.
    ///
    /// # Errors
    ///
    /// [`DecodeError::Codec`] when the header is truncated or unknown, or
    /// `offset` does not fit its offset field.
    fn new(file: &'a [u8], offset: u64) -> Result<Self, DecodeError> {
        let bad = || DecodeError::Codec("tiff: header unreadable".to_owned());
        let little = match file.get(..2) {
            Some(b"II") => true,
            Some(b"MM") => false,
            _ => return Err(bad()),
        };
        let magic = file.get(2..4).ok_or_else(bad)?;
        let magic = if little {
            u16::from_le_bytes([magic[0], magic[1]])
        } else {
            u16::from_be_bytes([magic[0], magic[1]])
        };
        let header_len = match magic {
            42 => 8,
            43 => 16,
            _ => return Err(bad()),
        };
        let mut header = [0; 16];
        header[..header_len].copy_from_slice(file.get(..header_len).ok_or_else(bad)?);
        if header_len == 8 {
            let offset = u32::try_from(offset).map_err(|_| bad())?;
            header[4..8].copy_from_slice(&if little {
                offset.to_le_bytes()
            } else {
                offset.to_be_bytes()
            });
        } else {
            header[8..16].copy_from_slice(&if little {
                offset.to_le_bytes()
            } else {
                offset.to_be_bytes()
            });
        }
        Ok(Self {
            file,
            header,
            header_len,
            pos: 0,
        })
    }
}

impl Read for Rebased<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let at = usize::try_from(self.pos).unwrap_or(usize::MAX);
        let source = if at < self.header_len {
            &self.header[at..self.header_len]
        } else {
            self.file.get(at..).unwrap_or_default()
        };
        let n = source.len().min(buf.len());
        buf[..n].copy_from_slice(&source[..n]);
        self.pos += n as u64;
        Ok(n)
    }
}

impl Seek for Rebased<'_> {
    fn seek(&mut self, to: SeekFrom) -> std::io::Result<u64> {
        let pos = match to {
            SeekFrom::Start(n) => Some(n),
            SeekFrom::End(d) => (self.file.len() as u64).checked_add_signed(d),
            SeekFrom::Current(d) => self.pos.checked_add_signed(d),
        };
        self.pos = pos.ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "seek before the file start",
            )
        })?;
        Ok(self.pos)
    }
}

/// Interleave `channels` equal planes stored one after another into
/// chunky pixels.
fn interleave_planes<T: Copy + Default>(planes: &[T], channels: usize) -> Vec<T> {
    let plane = planes.len() / channels;
    let mut out = vec![T::default(); plane * channels];
    for (c, src) in planes.chunks_exact(plane).enumerate() {
        for (dst, &s) in out[c..].iter_mut().step_by(channels).zip(src) {
            *dst = s;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{FIRST_DIRECTORY, MAX_DIRECTORIES, decode, interleave_planes, pages};
    use crate::decode::CodecPixels;

    #[test]
    fn planes_interleave_into_pixels() {
        assert_eq!(
            interleave_planes(&[1, 2, 3, 10, 20, 30], 2),
            [1, 10, 2, 20, 3, 30]
        );
    }

    /// A little-endian TIFF of 1 × 1 8-bit gray images, one directory per
    /// `subfile` (its `NewSubfileType`), the image of directory `i` of
    /// sample `i mod 256`, every directory linked to the next.
    fn chain(subfiles: &[u32]) -> Vec<u8> {
        let mut out = b"II*\0\x08\0\0\0".to_vec();
        for (i, &subfile) in subfiles.iter().enumerate() {
            let at = u32::try_from(out.len()).unwrap();
            // Count, seven entries, next offset, then the sample.
            let data = at + 2 + 7 * 12 + 4;
            let next = if i + 1 == subfiles.len() { 0 } else { data + 2 };
            out.extend_from_slice(&7u16.to_le_bytes());
            // (tag, type, value); type 3 SHORT, 4 LONG; count 1.
            for (tag, kind, value) in [
                (254u16, 4u16, subfile),
                (256, 4, 1),
                (257, 4, 1),
                (258, 3, 8),
                (262, 3, 1),
                (273, 4, data),
                (279, 4, 1),
            ] {
                out.extend_from_slice(&tag.to_le_bytes());
                out.extend_from_slice(&kind.to_le_bytes());
                out.extend_from_slice(&1u32.to_le_bytes());
                out.extend_from_slice(&value.to_le_bytes());
            }
            out.extend_from_slice(&next.to_le_bytes());
            out.extend_from_slice(&[u8::try_from(i % 256).unwrap(), 0]);
        }
        out
    }

    /// The sample of the 1 × 1 gray page `page` of `file`.
    fn sample(file: &[u8], page: usize) -> u8 {
        match decode(file, page, crate::decode::EIGHT) {
            Ok(CodecPixels::Gray(bmp)) => bmp.data()[0],
            Ok(_) => panic!("a gray page decodes to 8-bit gray"),
            Err(e) => panic!("page {page}: {e}"),
        }
    }

    #[test]
    fn pages_are_decoded_from_their_own_directory() {
        let file = chain(&[0, 1, 0, 4, 0]);
        let keys = pages(&file).unwrap();
        assert_eq!(keys.len(), 3);
        assert_eq!(keys[0], FIRST_DIRECTORY);
        let samples: Vec<u8> = keys.iter().map(|&key| sample(&file, key)).collect();
        assert_eq!(samples, [0, 2, 4]);
    }

    #[test]
    fn a_long_chain_of_thumbnails_is_cut_at_the_directory_cap() {
        // A page after more thumbnail directories than the cap: the walk
        // stops before reaching it, so only the first directory stands in.
        let mut subfiles = vec![1; MAX_DIRECTORIES + 10];
        subfiles.push(0);
        let file = chain(&subfiles);
        assert_eq!(pages(&file).unwrap(), [FIRST_DIRECTORY]);
    }

    #[test]
    fn decoding_every_page_of_a_long_file_is_linear() {
        // 5000 pages: walking the chain again from the first directory for
        // each page reads 1.25·10^7 directories; opening each page at its
        // own directory reads 5000.
        let file = chain(&vec![0; 5000]);
        let keys = pages(&file).unwrap();
        assert_eq!(keys.len(), 5000);
        let start = std::time::Instant::now();
        for (i, &key) in keys.iter().enumerate() {
            assert_eq!(usize::from(sample(&file, key)), i % 256);
        }
        let elapsed = start.elapsed();
        assert!(elapsed < std::time::Duration::from_secs(5), "{elapsed:?}");
    }

    #[test]
    fn a_rebased_header_keeps_every_other_byte() {
        let file = chain(&[0, 0]);
        let second = pages(&file).unwrap()[1];
        let mut rebased = super::Rebased::new(&file, second as u64).unwrap();
        let mut read = Vec::new();
        assert_eq!(
            std::io::Read::read_to_end(&mut rebased, &mut read).unwrap(),
            file.len()
        );
        assert_eq!(read[4..8], u32::try_from(second).unwrap().to_le_bytes());
        assert_eq!(read[..4], file[..4]);
        assert_eq!(read[8..], file[8..]);
        assert!(super::Rebased::new(b"II*\0", 8).is_err());
        assert!(super::Rebased::new(b"XX*\0\0\0\0\0", 8).is_err());
        assert!(super::Rebased::new(&file, 1 << 33).is_err());
    }
}
