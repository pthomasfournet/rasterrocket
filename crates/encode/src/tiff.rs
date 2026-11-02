//! Baseline TIFF 6.0 writer: one or more pages of RGB, gray or bilevel
//! pixels, each in strips, uncompressed or compressed with `PackBits`, LZW
//! or Deflate, and bilevel pages also with CCITT Group 3 or Group 4 fax
//! coding.
//!
//! The file is little-endian.  Each page's strips are followed by its image
//! file directory (IFD), whose values that do not fit in an entry follow
//! the directory.  Pages are encoded when they are added; the file is laid
//! out when it is written, every offset computed and the size checked
//! against the 4 GiB a TIFF addresses before the first byte goes out, then
//! streamed to the sink.

use std::io::Write;
use std::num::NonZeroU32;

use color::{Gray8, Rgb8};
use hayro_ccitt::{EncodeScheme, Encoder};
use raster::Bitmap;

use crate::{EncodeError, pack_bilevel_row};

/// The compression of a TIFF's strips.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TiffCompression {
    /// Uncompressed (TIFF compression 1).
    None,
    /// `PackBits` run-length coding (32773).
    PackBits,
    /// LZW with the TIFF code-size switch (5).
    Lzw,
    /// zlib-wrapped Deflate, the code TIFF assigns it (8).
    Deflate,
    /// CCITT Group 3 fax coding (3), bilevel pages only: every row after an
    /// end-of-line code padded to end on a byte boundary, coded
    /// two-dimensionally with period `k` when set, else one-dimensionally.
    /// ITU-T T.4 sets `k` to 2 at standard and 4 at fine resolution.
    Group3 {
        /// The period of one-dimensional rows in two-dimensional coding.
        k: Option<NonZeroU32>,
    },
    /// CCITT Group 4 fax coding (4), bilevel pages only.
    Group4,
}

impl TiffCompression {
    const fn code(self) -> u16 {
        match self {
            Self::None => 1,
            Self::PackBits => 32773,
            Self::Lzw => 5,
            Self::Deflate => 8,
            Self::Group3 { .. } => 3,
            Self::Group4 => 4,
        }
    }

    /// Whether this is a CCITT fax coding.
    const fn is_fax(self) -> bool {
        matches!(self, Self::Group3 { .. } | Self::Group4)
    }

    /// The options tag of a CCITT page: `T4Options` (292), bit 0 for
    /// two-dimensional coding and bit 2 for fill bits before each EOL, or
    /// `T6Options` (293), no options.
    const fn options_tag(self) -> Option<(u16, u32)> {
        match self {
            Self::Group3 { k } => Some((292, 4 | k.is_some() as u32)),
            Self::Group4 => Some((293, 0)),
            Self::None | Self::PackBits | Self::Lzw | Self::Deflate => None,
        }
    }
}

/// The pixels of a page.
#[derive(Clone, Copy)]
pub enum TiffImage<'a> {
    /// 8-bit RGB.
    Rgb(&'a Bitmap<Rgb8>),
    /// 8-bit gray, 0 black.
    Gray(&'a Bitmap<Gray8>),
    /// One bit per pixel from gray luma, black below
    /// [`crate::BLACK_BELOW`].  Written `WhiteIsZero`, a 1 bit black, the
    /// interpretation the fax profile of TIFF (RFC 2306) prescribes.
    Bilevel(&'a Bitmap<Gray8>),
}

impl TiffImage<'_> {
    const fn size(&self) -> (u32, u32) {
        match self {
            Self::Rgb(b) => (b.width, b.height),
            Self::Gray(b) | Self::Bilevel(b) => (b.width, b.height),
        }
    }

    /// Samples per pixel, bits per sample and photometric interpretation.
    const fn layout(&self) -> (u16, u16, u16) {
        match self {
            Self::Rgb(_) => (3, 8, PHOTOMETRIC_RGB),
            Self::Gray(_) => (1, 8, PHOTOMETRIC_BLACK_IS_ZERO),
            Self::Bilevel(_) => (1, 1, PHOTOMETRIC_WHITE_IS_ZERO),
        }
    }

    /// The length of a stored row: its samples' bits padded to a byte.
    fn row_bytes(&self) -> usize {
        let (samples, bits, _) = self.layout();
        (self.size().0 as usize * usize::from(samples) * usize::from(bits)).div_ceil(8)
    }

    /// Append the bytes of row `y` as stored: live pixels only, bilevel
    /// rows packed most significant bit first and padded to a byte.
    fn row(&self, y: u32, out: &mut Vec<u8>) {
        match self {
            Self::Rgb(b) => out.extend_from_slice(&b.row_bytes(y)[..b.width as usize * 3]),
            Self::Gray(b) => out.extend_from_slice(&b.row_bytes(y)[..b.width as usize]),
            Self::Bilevel(b) => {
                let start = out.len();
                out.resize(start + (b.width as usize).div_ceil(8), 0);
                pack_bilevel_row(&b.row_bytes(y)[..b.width as usize], &mut out[start..]);
            }
        }
    }
}

const PHOTOMETRIC_WHITE_IS_ZERO: u16 = 0;
const PHOTOMETRIC_BLACK_IS_ZERO: u16 = 1;
const PHOTOMETRIC_RGB: u16 = 2;

/// The refusal of a file beyond the 32-bit offsets of TIFF.
const TOO_LARGE: EncodeError =
    EncodeError::InvalidInput("TIFF file exceeds 4 GiB; BigTIFF is not written");

/// Target size of a strip before compression.
const STRIP_BYTES: usize = 64 * 1024;

/// A page encoded into strips, waiting for its directory: encoded on its
/// own, in any order or thread, and laid out when pushed to a
/// [`TiffWriter`].
pub struct TiffPage {
    width: u32,
    height: u32,
    /// Samples per pixel, bits per sample, photometric interpretation.
    layout: (u16, u16, u16),
    compression: TiffCompression,
    rows_per_strip: u32,
    strips: Vec<Vec<u8>>,
    /// Resolution in pixels per inch, as rationals.
    resolution: [(u32, u32); 2],
}

impl TiffPage {
    /// Encode `image` at `dpi` (horizontal, vertical) pixels per inch,
    /// compressed with `compression`.
    ///
    /// # Errors
    ///
    /// [`EncodeError::InvalidInput`] for an empty image or a resolution
    /// outside what a TIFF rational holds; [`EncodeError::UnsupportedMode`]
    /// for CCITT compression of a page that is not bilevel.
    pub fn encode(
        image: TiffImage<'_>,
        dpi: (f64, f64),
        compression: TiffCompression,
    ) -> Result<Self, EncodeError> {
        let (width, height) = image.size();
        let columns =
            NonZeroU32::new(width)
                .filter(|_| height > 0)
                .ok_or(EncodeError::InvalidInput(
                    "zero-width or zero-height bitmap: TIFF cannot encode an empty image",
                ))?;
        let fax = compression.is_fax();
        if fax && !matches!(image, TiffImage::Bilevel(_)) {
            return Err(EncodeError::UnsupportedMode(
                "CCITT fax compression codes bilevel pages only",
            ));
        }
        let resolution = [rational(dpi.0)?, rational(dpi.1)?];
        let row_bytes = image.row_bytes();
        // A CCITT page is one strip, as the fax profile of TIFF (RFC 2306)
        // recommends; each strip would restart the coding's reference row.
        let rows_per_strip = if fax {
            height
        } else {
            u32::try_from((STRIP_BYTES / row_bytes).max(1))
                .unwrap_or(u32::MAX)
                .min(height)
        };
        let mut strips = Vec::new();
        let mut raw = Vec::with_capacity(row_bytes * rows_per_strip as usize);
        for first in (0..height).step_by(rows_per_strip as usize) {
            raw.clear();
            for y in first..height.min(first.saturating_add(rows_per_strip)) {
                image.row(y, &mut raw);
            }
            strips.push(compress(&raw, row_bytes, columns, compression)?);
        }
        Ok(Self {
            width,
            height,
            layout: image.layout(),
            compression,
            rows_per_strip,
            strips,
            resolution,
        })
    }
}

/// A TIFF file built page by page.
#[derive(Default)]
pub struct TiffWriter {
    pages: Vec<TiffPage>,
    /// Whether every page is marked a page of a multi-page document, as
    /// the fax profile of TIFF (RFC 2306) requires; otherwise only the
    /// pages of a file of more than one are.
    fax_document: bool,
}

impl TiffWriter {
    /// An empty file.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty fax document: every page, even a single one, is marked a
    /// page of a multi-page document (`NewSubfileType` bit 1).
    #[must_use]
    pub const fn fax_document() -> Self {
        Self {
            pages: Vec::new(),
            fax_document: true,
        }
    }

    /// Encode `image` as the next page, at `dpi` (horizontal, vertical)
    /// pixels per inch, compressed with `compression`.
    ///
    /// # Errors
    ///
    /// As [`TiffPage::encode`].
    pub fn add_page(
        &mut self,
        image: TiffImage<'_>,
        dpi: (f64, f64),
        compression: TiffCompression,
    ) -> Result<(), EncodeError> {
        self.pages.push(TiffPage::encode(image, dpi, compression)?);
        Ok(())
    }

    /// Append `page`, encoded beforehand, as the next page.
    pub fn push_page(&mut self, page: TiffPage) {
        self.pages.push(page);
    }

    /// Write the file to `out`.
    ///
    /// # Errors
    ///
    /// [`EncodeError::InvalidInput`] without pages, with more than 65535,
    /// or when the file would exceed the 4 GiB a TIFF can address, all
    /// found before anything is written; [`EncodeError::Io`] on a write
    /// failure.
    pub fn write<W: Write>(self, mut out: W) -> Result<(), EncodeError> {
        if self.pages.is_empty() {
            return Err(EncodeError::InvalidInput("a TIFF needs at least one page"));
        }
        let count = u16::try_from(self.pages.len())
            .map_err(|_| EncodeError::InvalidInput("a TIFF holds at most 65535 pages"))?;
        let one_of_many = self.fax_document || count > 1;

        // Lay the file out: header, then per page its strips, a pad byte
        // to an even offset, and its directory with the values that
        // overflow it.
        let mut layouts = Vec::with_capacity(self.pages.len());
        let mut end: u64 = 8;
        for page in &self.pages {
            let mut strip_offsets = Vec::with_capacity(page.strips.len());
            for strip in &page.strips {
                strip_offsets.push(end);
                end += strip.len() as u64;
            }
            end += end % 2;
            let ifd = end;
            end += directory_len(page) as u64;
            layouts.push((strip_offsets, ifd));
        }
        let file_offset = |at: u64| u32::try_from(at).map_err(|_| TOO_LARGE);
        if u32::try_from(end).is_err() {
            return Err(TOO_LARGE);
        }

        out.write_all(b"II*\0")?;
        out.write_all(&file_offset(layouts[0].1)?.to_le_bytes())?;
        let mut at: u64 = 8;
        for (number, (page, (strip_offsets, ifd))) in
            (0..count).zip(self.pages.iter().zip(&layouts))
        {
            for strip in &page.strips {
                out.write_all(strip)?;
                at += strip.len() as u64;
            }
            if at % 2 == 1 {
                out.write_all(&[0])?;
                at += 1;
            }
            debug_assert_eq!(at, *ifd, "the directory lands where it was laid out");
            let offsets = strip_offsets
                .iter()
                .map(|&o| file_offset(o))
                .collect::<Result<Vec<_>, _>>()?;
            let next = match layouts.get(usize::from(number) + 1) {
                Some(&(_, next)) => file_offset(next)?,
                None => 0,
            };
            let directory = directory(
                page,
                &offsets,
                file_offset(*ifd)?,
                next,
                (number, count),
                one_of_many,
            );
            out.write_all(&directory)?;
            at += directory.len() as u64;
        }
        out.flush()?;
        Ok(())
    }
}

/// Write `image` as a one-page TIFF to `out`.
///
/// # Errors
///
/// As [`TiffPage::encode`] and [`TiffWriter::write`].
pub fn write_tiff<W: Write>(
    image: TiffImage<'_>,
    dpi: (f64, f64),
    compression: TiffCompression,
    out: W,
) -> Result<(), EncodeError> {
    let mut writer = TiffWriter::new();
    writer.add_page(image, dpi, compression)?;
    writer.write(out)
}

/// `bytes`, stored rows of `row_bytes` each and `columns` pixels,
/// compressed with `compression`.  `PackBits` runs do not cross a row.
fn compress(
    bytes: &[u8],
    row_bytes: usize,
    columns: NonZeroU32,
    compression: TiffCompression,
) -> Result<Vec<u8>, EncodeError> {
    Ok(match compression {
        TiffCompression::None => bytes.to_vec(),
        TiffCompression::PackBits => {
            let mut out = Vec::with_capacity(
                bytes.len() + bytes.len().div_ceil(row_bytes) * row_bytes.div_ceil(128),
            );
            for row in bytes.chunks(row_bytes) {
                pack_bits(row, &mut out);
            }
            out
        }
        TiffCompression::Lzw => {
            weezl::encode::Encoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8)
                .encode(bytes)
                .expect("every byte is a symbol of an 8-bit LZW alphabet")
        }
        TiffCompression::Deflate => {
            let mut encoder =
                flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
            encoder.write_all(bytes)?;
            encoder.finish()?
        }
        TiffCompression::Group3 { k } => fax_code(
            bytes,
            row_bytes,
            columns,
            EncodeScheme::Group3 {
                k,
                byte_aligned_eol: true,
            },
        ),
        TiffCompression::Group4 => fax_code(bytes, row_bytes, columns, EncodeScheme::Group4),
    })
}

/// Packed bilevel rows of `columns` pixels, `row_bytes` each, fax-coded by
/// `scheme`.
fn fax_code(bytes: &[u8], row_bytes: usize, columns: NonZeroU32, scheme: EncodeScheme) -> Vec<u8> {
    let mut encoder = Encoder::new(columns, scheme);
    for row in bytes.chunks(row_bytes) {
        encoder.encode_packed_row(row);
    }
    encoder.finish()
}

/// Append `row` `PackBits`-coded to `out`: runs of 2 to 128 equal bytes as
/// `1 - n` and the byte, other bytes in literal blocks of up to 128 as
/// `n - 1` and the bytes.  A literal block absorbs a pair of equal bytes,
/// which coded as a run would cost a header more; a run of three or more,
/// or a pair at the start of a block, is coded as a run.
fn pack_bits(row: &[u8], out: &mut Vec<u8>) {
    let run_at = |i: usize| {
        row[i..]
            .iter()
            .take(128)
            .take_while(|&&b| b == row[i])
            .count()
    };
    let mut i = 0;
    while i < row.len() {
        let run = run_at(i);
        if run >= 2 {
            out.push((1 - i16::try_from(run).expect("run ≤ 128")).to_le_bytes()[0]);
            out.push(row[i]);
            i += run;
            continue;
        }
        let start = i;
        while i < row.len() && i - start < 128 && (i == start || run_at(i) < 3) {
            i += 1;
        }
        out.push(u8::try_from(i - start - 1).expect("literal block ≤ 128"));
        out.extend_from_slice(&row[start..i]);
    }
}

/// `dpi` as a TIFF rational: a whole number of pixels per inch over 1, or
/// thousandths reduced by their common divisor.
fn rational(dpi: f64) -> Result<(u32, u32), EncodeError> {
    let scaled = (dpi * 1000.0).round();
    if !(dpi.is_finite() && scaled >= 1.0 && scaled <= f64::from(u32::MAX)) {
        return Err(EncodeError::InvalidInput(
            "TIFF resolution must lie between 0.001 and 4294967 pixels per inch",
        ));
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "scaled is a whole number within 1..=u32::MAX, checked above"
    )]
    let thousandths = scaled as u32;
    let divisor = gcd(thousandths, 1000);
    Ok((thousandths / divisor, 1000 / divisor))
}

/// The greatest common divisor of `a` and `b`.
const fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// A directory entry's value: its type code and the bytes of its values.
struct Entry {
    tag: u16,
    kind: u16,
    count: u32,
    bytes: Vec<u8>,
}

const SHORT: u16 = 3;
const LONG: u16 = 4;
const RATIONAL: u16 = 5;

fn shorts(tag: u16, values: &[u16]) -> Entry {
    Entry {
        tag,
        kind: SHORT,
        count: u32::try_from(values.len()).expect("few values"),
        bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

fn longs(tag: u16, values: &[u32]) -> Entry {
    Entry {
        tag,
        kind: LONG,
        count: u32::try_from(values.len()).expect("strip count fits u32"),
        bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

fn rational_entry(tag: u16, (n, d): (u32, u32)) -> Entry {
    Entry {
        tag,
        kind: RATIONAL,
        count: 1,
        bytes: [n.to_le_bytes(), d.to_le_bytes()].concat(),
    }
}

/// The directory entries of `page`, whose strips start at `offsets` and
/// hold `byte_counts` bytes, sorted by tag.  Their sizes do not depend on
/// the offset and count values.
fn entries(
    page: &TiffPage,
    offsets: &[u32],
    byte_counts: &[u32],
    position: (u16, u16),
    one_of_many: bool,
) -> Vec<Entry> {
    let (samples, bits, photometric) = page.layout;
    // NewSubfileType bit 1: a page of a multi-page document.
    let mut entries = vec![
        longs(254, &[if one_of_many { 2 } else { 0 }]),
        longs(256, &[page.width]),
        longs(257, &[page.height]),
        shorts(258, &vec![bits; usize::from(samples)]),
        shorts(259, &[page.compression.code()]),
        shorts(262, &[photometric]),
        longs(273, offsets),
        shorts(277, &[samples]),
        longs(278, &[page.rows_per_strip]),
        longs(279, byte_counts),
        rational_entry(282, page.resolution[0]),
        rational_entry(283, page.resolution[1]),
        shorts(284, &[1]),
        shorts(296, &[2]),
        shorts(297, &<[u16; 2]>::from(position)),
    ];
    if bits == 1 {
        entries.push(shorts(266, &[1]));
    }
    if let Some((tag, options)) = page.compression.options_tag() {
        entries.push(longs(tag, &[options]));
    }
    entries.sort_by_key(|e| e.tag);
    entries
}

/// The length in bytes of `page`'s directory with its overflowing values:
/// two bytes of count, twelve per entry, four of link, then each value
/// longer than four bytes padded to an even length.
fn directory_len(page: &TiffPage) -> usize {
    let placeholder = vec![0; page.strips.len()];
    let entries = entries(page, &placeholder, &placeholder, (0, 1), false);
    2 + 12 * entries.len()
        + 4
        + entries
            .iter()
            .filter(|e| e.bytes.len() > 4)
            .map(|e| e.bytes.len() + e.bytes.len() % 2)
            .sum::<usize>()
}

/// The directory of `page` placed at file offset `at`, its strips at
/// `offsets`, linking to the directory at `next` (0 for none), followed by
/// its overflowing values.  The caller has checked that the file, the
/// directory included, ends within 4 GiB.
fn directory(
    page: &TiffPage,
    offsets: &[u32],
    at: u32,
    next: u32,
    position: (u16, u16),
    one_of_many: bool,
) -> Vec<u8> {
    let byte_counts: Vec<u32> = page
        .strips
        .iter()
        .map(|s| u32::try_from(s.len()).expect("a strip lies within the 4 GiB file"))
        .collect();
    let entries = entries(page, offsets, &byte_counts, position, one_of_many);
    let n = u16::try_from(entries.len()).expect("fewer than 65536 entries");
    let mut overflow = at as usize + 2 + 12 * entries.len() + 4;
    let mut directory = Vec::with_capacity(directory_len(page));
    let mut spilled = Vec::new();
    directory.extend_from_slice(&n.to_le_bytes());
    for e in &entries {
        directory.extend_from_slice(&e.tag.to_le_bytes());
        directory.extend_from_slice(&e.kind.to_le_bytes());
        directory.extend_from_slice(&e.count.to_le_bytes());
        if e.bytes.len() <= 4 {
            let mut inline = [0u8; 4];
            inline[..e.bytes.len()].copy_from_slice(&e.bytes);
            directory.extend_from_slice(&inline);
        } else {
            let value_at =
                u32::try_from(overflow).expect("the directory's values lie within the 4 GiB file");
            directory.extend_from_slice(&value_at.to_le_bytes());
            overflow += e.bytes.len() + e.bytes.len() % 2;
            spilled.extend_from_slice(&e.bytes);
            if e.bytes.len() % 2 == 1 {
                spilled.push(0);
            }
        }
    }
    directory.extend_from_slice(&next.to_le_bytes());
    directory.extend_from_slice(&spilled);
    directory
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use ::tiff::decoder::{Decoder, DecodingResult};
    use ::tiff::tags::Tag;
    use color::{Gray8, Rgb8};
    use raster::Bitmap;

    use std::num::NonZeroU32;

    use super::{TiffCompression, TiffImage, TiffPage, TiffWriter, pack_bits, write_tiff};
    use crate::EncodeError;

    const ALL: [TiffCompression; 4] = [
        TiffCompression::None,
        TiffCompression::PackBits,
        TiffCompression::Lzw,
        TiffCompression::Deflate,
    ];

    /// A `w × h` gray bitmap with runs and noise, so every coder has both.
    fn gray(w: u32, h: u32) -> Bitmap<Gray8> {
        let mut bmp = Bitmap::new(w, h, 4, false);
        for y in 0..h {
            for (x, px) in bmp.row_bytes_mut(y)[..w as usize].iter_mut().enumerate() {
                let x = u32::try_from(x).expect("small");
                *px = if x < w / 2 {
                    255
                } else {
                    (x * 37 + y * 11).to_le_bytes()[0]
                };
            }
        }
        bmp
    }

    fn rgb(w: u32, h: u32) -> Bitmap<Rgb8> {
        let mut bmp = Bitmap::new(w, h, 4, false);
        for y in 0..h {
            for (i, px) in bmp.row_bytes_mut(y)[..w as usize * 3]
                .iter_mut()
                .enumerate()
            {
                *px = (i * 7 + y as usize * 3).to_le_bytes()[0];
            }
        }
        bmp
    }

    fn decode(bytes: &[u8]) -> Decoder<Cursor<&[u8]>> {
        Decoder::new(Cursor::new(bytes)).expect("the tiff crate reads the file")
    }

    fn bytes(result: DecodingResult) -> Vec<u8> {
        match result {
            DecodingResult::U8(v) => v,
            _ => panic!("8-bit samples expected"),
        }
    }

    #[test]
    fn gray_and_rgb_pages_read_back_under_every_compression() {
        let (g, c) = (gray(37, 300), rgb(21, 50));
        for compression in ALL {
            let mut out = Vec::new();
            write_tiff(TiffImage::Gray(&g), (150.0, 150.0), compression, &mut out).unwrap();
            let pixels = bytes(decode(&out).read_image().unwrap());
            let want: Vec<u8> = (0..g.height)
                .flat_map(|y| g.row_bytes(y)[..37].to_vec())
                .collect();
            assert_eq!(pixels, want, "gray, {compression:?}");

            let mut out = Vec::new();
            write_tiff(TiffImage::Rgb(&c), (72.0, 96.0), compression, &mut out).unwrap();
            let mut decoder = decode(&out);
            assert_eq!(decoder.dimensions().unwrap(), (21, 50));
            let pixels = bytes(decoder.read_image().unwrap());
            let want: Vec<u8> = (0..c.height)
                .flat_map(|y| c.row_bytes(y)[..63].to_vec())
                .collect();
            assert_eq!(pixels, want, "rgb, {compression:?}");
        }
    }

    #[test]
    fn bilevel_rows_pack_black_as_one() {
        let g = gray(13, 3);
        for compression in ALL {
            let mut out = Vec::new();
            write_tiff(
                TiffImage::Bilevel(&g),
                (204.0, 196.0),
                compression,
                &mut out,
            )
            .unwrap();
            // The reader inverts WhiteIsZero, so its 1 bits are white.
            let pixels = bytes(decode(&out).read_image().unwrap());
            for y in 0..3u32 {
                for x in 0..13usize {
                    let white = pixels[y as usize * 2 + x / 8] & (0x80 >> (x % 8)) != 0;
                    assert_eq!(
                        white,
                        g.row_bytes(y)[x] >= 128,
                        "({x}, {y}) {compression:?}"
                    );
                }
            }
        }
    }

    /// A bilevel test page: a black frame, a diagonal and bands of short
    /// strokes, so fax coding meets every mode.
    fn drawing(w: u32, h: u32) -> Bitmap<Gray8> {
        let mut bmp = Bitmap::new(w, h, 4, false);
        for y in 0..h {
            for (x, px) in bmp.row_bytes_mut(y)[..w as usize].iter_mut().enumerate() {
                let x = u32::try_from(x).expect("small");
                let ink = x < 3
                    || y < 3
                    || x + 3 >= w
                    || y + 3 >= h
                    || x.abs_diff(y) < 2
                    || (y % 9 < 4 && (x / 3 + y / 2) % 5 == 0);
                *px = if ink { 0 } else { 255 };
            }
        }
        bmp
    }

    /// The packed rows of `g` as stored bilevel: 1 black.
    fn packed(g: &Bitmap<Gray8>) -> Vec<u8> {
        let mut rows = Vec::new();
        for y in 0..g.height {
            TiffImage::Bilevel(g).row(y, &mut rows);
        }
        rows
    }

    #[test]
    fn group4_pages_read_back_through_an_independent_decoder() {
        let g = drawing(1728, 120);
        let mut out = Vec::new();
        write_tiff(
            TiffImage::Bilevel(&g),
            (204.0, 196.0),
            TiffCompression::Group4,
            &mut out,
        )
        .unwrap();
        let mut decoder = decode(&out);
        assert_eq!(decoder.get_tag_u32(Tag::Unknown(293)).unwrap(), 0);
        assert_eq!(
            decoder.get_tag_u32(Tag::NewSubfileType).unwrap(),
            0,
            "a single page outside a fax document is not marked"
        );
        assert_eq!(
            decoder.get_tag_u32(Tag::RowsPerStrip).unwrap(),
            120,
            "one strip"
        );
        // The reader inverts WhiteIsZero: its 1 bits are white.
        let pixels: Vec<u8> = bytes(decoder.read_image().unwrap())
            .into_iter()
            .map(|b| !b)
            .collect();
        assert_eq!(pixels, packed(&g));
        assert!(
            out.len() * 3 < pixels.len(),
            "{} of {} bytes",
            out.len(),
            pixels.len()
        );
    }

    #[test]
    fn group3_pages_decode_and_declare_their_options() {
        use hayro_ccitt::{DecodeSettings, Decoder as Fax, DecoderContext, EncodingMode};

        /// Rows decoded to packed bits, 1 black.
        struct Packed(Vec<u8>, u32);
        impl Fax for Packed {
            fn push_row(&mut self, changes: &[u32]) {
                let start = self.0.len();
                self.0.resize(start + self.1.div_ceil(8) as usize, 0);
                let ends = changes.iter().copied().chain([self.1]);
                for (i, (from, to)) in changes.iter().copied().zip(ends.skip(1)).enumerate() {
                    if i % 2 == 0 {
                        for x in from..to {
                            self.0[start + (x / 8) as usize] |= 0x80 >> (x % 8);
                        }
                    }
                }
            }
        }

        let g = drawing(203, 77);
        for (k, options) in [(None, 4), (NonZeroU32::new(2), 5), (NonZeroU32::new(4), 5)] {
            let mut out = Vec::new();
            write_tiff(
                TiffImage::Bilevel(&g),
                (204.0, 98.0),
                TiffCompression::Group3 { k },
                &mut out,
            )
            .unwrap();
            let mut decoder = decode(&out);
            assert_eq!(decoder.get_tag_u32(Tag::Compression).unwrap(), 3);
            assert_eq!(decoder.get_tag_u32(Tag::Unknown(292)).unwrap(), options);
            let offset = decoder.get_tag_u32(Tag::StripOffsets).unwrap() as usize;
            let len = decoder.get_tag_u32(Tag::StripByteCounts).unwrap() as usize;

            let mut settings = DecodeSettings::new(203, 77);
            settings.strict = true;
            settings.end_of_line = true;
            settings.rows_are_byte_aligned = true;
            settings.encoding = if k.is_some() {
                EncodingMode::Group3_2D
            } else {
                EncodingMode::Group3_1D
            };
            let mut rows = Packed(Vec::new(), 203);
            let _consumed = hayro_ccitt::decode(
                &out[offset..offset + len],
                &mut rows,
                &mut DecoderContext::new(settings),
            )
            .expect("the strip decodes");
            assert_eq!(rows.0, packed(&g), "k = {k:?}");
        }
    }

    #[test]
    fn fax_coding_refuses_pages_that_are_not_bilevel() {
        let g = gray(16, 4);
        for compression in [TiffCompression::Group4, TiffCompression::Group3 { k: None }] {
            let result = TiffPage::encode(TiffImage::Gray(&g), (204.0, 196.0), compression);
            assert!(matches!(result, Err(EncodeError::UnsupportedMode(_))));
        }
    }

    #[test]
    fn a_fax_document_marks_even_a_single_page() {
        let g = drawing(16, 4);
        for (mut writer, flag) in [(TiffWriter::new(), 0), (TiffWriter::fax_document(), 2)] {
            writer
                .add_page(
                    TiffImage::Bilevel(&g),
                    (204.0, 196.0),
                    TiffCompression::Group4,
                )
                .unwrap();
            let mut out = Vec::new();
            writer.write(&mut out).unwrap();
            assert_eq!(decode(&out).get_tag_u32(Tag::NewSubfileType).unwrap(), flag);
        }
    }

    #[test]
    fn pages_carry_their_number_and_resolution() {
        let g = gray(8, 8);
        let mut writer = TiffWriter::new();
        for _ in 0..3 {
            writer
                .add_page(
                    TiffImage::Bilevel(&g),
                    (204.0, 97.5),
                    TiffCompression::Deflate,
                )
                .unwrap();
        }
        let mut out = Vec::new();
        writer.write(&mut out).unwrap();
        let mut decoder = decode(&out);
        for page in 0..3u16 {
            assert_eq!(
                decoder.get_tag_u16_vec(Tag::Unknown(297)).unwrap(),
                [page, 3]
            );
            assert_eq!(decoder.get_tag_u32(Tag::NewSubfileType).unwrap(), 2);
            let y_res = decoder.get_tag_u32_vec(Tag::YResolution).unwrap();
            assert_eq!(y_res, [195, 2], "97.5 as a reduced fraction");
            assert_eq!(decoder.more_images(), page < 2);
            if page < 2 {
                decoder.next_image().unwrap();
            }
        }
    }

    #[test]
    fn pages_encoded_apart_are_laid_out_in_push_order() {
        let (a, b) = (drawing(40, 30), gray(40, 30));
        let pages = std::thread::scope(|scope| {
            let second = scope.spawn(|| {
                TiffPage::encode(TiffImage::Gray(&b), (100.0, 100.0), TiffCompression::Lzw)
            });
            let first = TiffPage::encode(
                TiffImage::Bilevel(&a),
                (204.0, 196.0),
                TiffCompression::Group4,
            );
            [first.unwrap(), second.join().unwrap().unwrap()]
        });
        let mut writer = TiffWriter::new();
        for page in pages {
            writer.push_page(page);
        }
        let mut out = Vec::new();
        writer.write(&mut out).unwrap();
        let mut decoder = decode(&out);
        assert_eq!(decoder.get_tag_u32(Tag::Compression).unwrap(), 4);
        decoder.next_image().unwrap();
        assert_eq!(decoder.get_tag_u32(Tag::Compression).unwrap(), 5);
        let want: Vec<u8> = (0..30)
            .flat_map(|y| b.row_bytes(y)[..40].to_vec())
            .collect();
        assert_eq!(bytes(decoder.read_image().unwrap()), want);
    }

    #[test]
    fn invalid_pages_are_refused() {
        let empty: Bitmap<Gray8> = Bitmap::new(0, 4, 1, false);
        let g = gray(4, 4);
        let invalid = |image, dpi| {
            matches!(
                TiffPage::encode(image, dpi, TiffCompression::None),
                Err(EncodeError::InvalidInput(_))
            )
        };
        assert!(invalid(TiffImage::Gray(&empty), (72.0, 72.0)));
        assert!(invalid(TiffImage::Gray(&g), (0.0, 72.0)));
        assert!(invalid(TiffImage::Gray(&g), (f64::NAN, 72.0)));
        assert!(invalid(TiffImage::Gray(&g), (5e6, 72.0)));
        assert!(matches!(
            TiffWriter::new().write(Vec::new()),
            Err(EncodeError::InvalidInput(_))
        ));
        let mut file = Vec::new();
        write_tiff(
            TiffImage::Gray(&g),
            (72.0, 204.0),
            TiffCompression::None,
            &mut file,
        )
        .unwrap();
        let decoder = &mut decode(&file);
        assert_eq!(decoder.get_tag_u32_vec(Tag::XResolution).unwrap(), [72, 1]);
        assert_eq!(decoder.get_tag_u32_vec(Tag::YResolution).unwrap(), [204, 1]);
    }

    /// Decode `PackBits` per the TIFF 6.0 description.
    fn unpack(mut data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        while let [n, rest @ ..] = data {
            let n = i8::from_le_bytes([*n]);
            if n >= 0 {
                let len = usize::from(n.unsigned_abs()) + 1;
                out.extend_from_slice(&rest[..len]);
                data = &rest[len..];
            } else if n != -128 {
                out.extend(std::iter::repeat_n(
                    rest[0],
                    usize::from(n.unsigned_abs()) + 1,
                ));
                data = &rest[1..];
            } else {
                data = rest;
            }
        }
        out
    }

    #[test]
    fn pack_bits_keeps_a_pair_inside_a_literal_block() {
        // "ab cc de": one literal block of six bytes, not three blocks.
        let mut out = Vec::new();
        pack_bits(b"abccde", &mut out);
        assert_eq!(out, b"\x05abccde");
        // A pair at the start of a block and a run of three are runs.
        let mut out = Vec::new();
        pack_bits(b"aabcccd", &mut out);
        assert_eq!(unpack(&out), b"aabcccd");
        assert_eq!(out, [0xFF, b'a', 0x00, b'b', 0xFE, b'c', 0x00, b'd']);
    }

    #[test]
    fn pack_bits_round_trips_runs_literals_and_long_blocks() {
        let mut row = vec![7u8; 300];
        row.extend((0..=255u8).cycle().take(400));
        row.extend([1, 1, 2, 3, 3, 3]);
        let mut out = Vec::new();
        pack_bits(&row, &mut out);
        assert_eq!(unpack(&out), row);
        assert!(out.len() < row.len(), "the runs compress");
    }
}
