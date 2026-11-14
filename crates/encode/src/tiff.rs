//! Baseline TIFF 6.0 writer: one or more pages of RGB (with unassociated
//! alpha when the bitmap has an alpha plane), gray or bilevel pixels, each
//! in strips, uncompressed or compressed with `PackBits`, LZW or Deflate,
//! and bilevel pages also with CCITT Group 3 or Group 4 fax coding.
//!
//! The file is little-endian.  Each page's strips are followed by its image
//! file directory (IFD), whose values that do not fit in an entry follow
//! the directory.  Pages are encoded when they are added; the file is laid
//! out when it is written, every offset computed and the size checked
//! against the 4 GiB a TIFF addresses before the first byte goes out, then
//! streamed to the sink.

use std::io::{Seek, SeekFrom, Write};
use std::num::NonZeroU32;

use hayro_ccitt::{EncodeScheme, Encoder};

use crate::EncodeError;
use crate::rows::{RowCount, RowImage, RowLayout};

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

const PHOTOMETRIC_WHITE_IS_ZERO: u16 = 0;
const PHOTOMETRIC_BLACK_IS_ZERO: u16 = 1;
const PHOTOMETRIC_RGB: u16 = 2;

/// Samples per pixel, bits per sample and photometric interpretation of
/// rows stored as `layout`.  RGBA's alpha is unassociated (`ExtraSamples`
/// 2); a bilevel page is `WhiteIsZero`, a 1 bit black, the interpretation
/// the fax profile of TIFF (RFC 2306) prescribes.
const fn tiff_layout(layout: RowLayout) -> (u16, u16, u16) {
    match layout {
        RowLayout::Rgba => (4, 8, PHOTOMETRIC_RGB),
        RowLayout::Rgb => (3, 8, PHOTOMETRIC_RGB),
        RowLayout::Gray => (1, 8, PHOTOMETRIC_BLACK_IS_ZERO),
        RowLayout::Bilevel => (1, 1, PHOTOMETRIC_WHITE_IS_ZERO),
    }
}

/// The refusal of a classic file whose data passed 4 GiB as it was written.
const TOO_LARGE: EncodeError = EncodeError::InvalidInput(
    "the TIFF's compressed data passed the 4 GiB of a classic TIFF as it was written; \
     choose another compression, or none, which is laid out as BigTIFF from the start",
);

/// The two file formats: classic TIFF, of 32-bit offsets, and `BigTIFF`, of
/// 64-bit ones.  A file is written classic unless its offsets would not
/// fit 32 bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    Classic,
    Big,
}

impl Format {
    /// The format of a file of `len` bytes.
    fn for_len(len: u64) -> Self {
        if u32::try_from(len).is_ok() {
            Self::Classic
        } else {
            Self::Big
        }
    }

    /// The header, linking to the first directory at `ifd`.
    fn header(self, ifd: u64) -> Vec<u8> {
        match self {
            Self::Classic => {
                let ifd = u32::try_from(ifd).expect("a classic file's offsets fit 32 bits");
                [b"II*\0".as_slice(), &ifd.to_le_bytes()].concat()
            }
            // Version 43, offsets of 8 bytes, then a reserved 0.
            Self::Big => [b"II+\0\x08\0\0\0".as_slice(), &ifd.to_le_bytes()].concat(),
        }
    }

    const fn header_len(self) -> u64 {
        match self {
            Self::Classic => 8,
            Self::Big => 16,
        }
    }

    /// The offset of the header's link to the first directory.
    const fn link_at(self) -> u64 {
        match self {
            Self::Classic => 4,
            Self::Big => 8,
        }
    }

    /// Bytes of an offset, and of an entry's value field.
    const fn offset_bytes(self) -> usize {
        match self {
            Self::Classic => 4,
            Self::Big => 8,
        }
    }

    /// Bytes of a directory's entry count.
    const fn count_bytes(self) -> usize {
        match self {
            Self::Classic => 2,
            Self::Big => 8,
        }
    }

    /// Bytes of a directory entry: tag, type, count and value field.
    const fn entry_bytes(self) -> usize {
        4 + self.offset_bytes() * 2
    }

    /// `v` as an offset of this format.
    fn offset(self, v: u64) -> Vec<u8> {
        match self {
            Self::Classic => u32::try_from(v)
                .expect("a classic file's offsets fit 32 bits")
                .to_le_bytes()
                .to_vec(),
            Self::Big => v.to_le_bytes().to_vec(),
        }
    }
}

/// Target size of a strip before compression.
const STRIP_BYTES: usize = 64 * 1024;

/// What the directory of a page records besides its strips.
#[derive(Clone, Copy)]
struct PageMeta {
    width: u32,
    height: u32,
    /// Samples per pixel, bits per sample, photometric interpretation.
    layout: (u16, u16, u16),
    compression: TiffCompression,
    rows_per_strip: u32,
    /// Resolution in pixels per inch, as rationals.
    resolution: [(u32, u32); 2],
}

/// A page encoded into strips, waiting for its directory: encoded on its
/// own, in any order or thread, and laid out when pushed to a
/// [`TiffWriter`].
pub struct TiffPage {
    meta: PageMeta,
    strips: Vec<Vec<u8>>,
}

impl TiffPage {
    /// Encode `image` at `dpi` (horizontal, vertical) pixels per inch,
    /// compressed with `compression`: RGB (RGBA with an alpha plane), gray
    /// or bilevel as `image` is.
    ///
    /// # Errors
    ///
    /// As [`TiffPageEncoder::new`].
    pub fn encode(
        image: RowImage<'_>,
        dpi: (f64, f64),
        compression: TiffCompression,
    ) -> Result<Self, EncodeError> {
        let (width, height) = image.size();
        let mut encoder = TiffPageEncoder::new(width, height, image.layout(), dpi, compression)?;
        encoder.push(image)?;
        encoder.finish()
    }
}

/// A page encoded into strips as its rows arrive, band by band: a strip is
/// compressed once its rows are in, so a page in bands is held as its
/// compressed strips and at most one strip of rows.
pub struct TiffPageEncoder {
    meta: PageMeta,
    rows: RowCount,
    layout: RowLayout,
    row_bytes: usize,
    /// Rows of the strip being filled, as stored.
    raw: Vec<u8>,
    /// The fax coder of a CCITT page, whose single strip spans the page.
    fax: Option<Encoder>,
    /// Strips compressed and not yet taken.
    strips: Vec<Vec<u8>>,
}

impl TiffPageEncoder {
    /// An encoder of a `width × height` page stored as `layout`, at `dpi`
    /// (horizontal, vertical) pixels per inch, compressed with
    /// `compression`.
    ///
    /// # Errors
    ///
    /// [`EncodeError::InvalidInput`] for an empty page or a resolution
    /// outside what a TIFF rational holds; [`EncodeError::UnsupportedMode`]
    /// for CCITT compression of a page that is not bilevel.
    pub fn new(
        width: u32,
        height: u32,
        layout: RowLayout,
        dpi: (f64, f64),
        compression: TiffCompression,
    ) -> Result<Self, EncodeError> {
        let rows = RowCount::new(width, height).map_err(|_| {
            EncodeError::InvalidInput(
                "zero-width or zero-height bitmap: TIFF cannot encode an empty image",
            )
        })?;
        let Some(columns) = NonZeroU32::new(width) else {
            unreachable!("RowCount refuses a zero width")
        };
        let fax = compression.is_fax();
        if fax && layout != RowLayout::Bilevel {
            return Err(EncodeError::UnsupportedMode(
                "CCITT fax compression codes bilevel pages only",
            ));
        }
        let resolution = [rational(dpi.0)?, rational(dpi.1)?];
        let row_bytes = layout.row_bytes(width);
        // A CCITT page is one strip, as the fax profile of TIFF (RFC 2306)
        // recommends; each strip would restart the coding's reference row.
        let rows_per_strip = if fax {
            height
        } else {
            u32::try_from((STRIP_BYTES / row_bytes).max(1))
                .unwrap_or(u32::MAX)
                .min(height)
        };
        let fax = match compression {
            TiffCompression::Group3 { k } => Some(Encoder::new(
                columns,
                EncodeScheme::Group3 {
                    k,
                    byte_aligned_eol: true,
                },
            )),
            TiffCompression::Group4 => Some(Encoder::new(columns, EncodeScheme::Group4)),
            TiffCompression::None
            | TiffCompression::PackBits
            | TiffCompression::Lzw
            | TiffCompression::Deflate => None,
        };
        let strip_capacity = if fax.is_some() {
            row_bytes
        } else {
            row_bytes * rows_per_strip as usize
        };
        Ok(Self {
            meta: PageMeta {
                width,
                height,
                layout: tiff_layout(layout),
                compression,
                rows_per_strip,
                resolution,
            },
            rows,
            layout,
            row_bytes,
            raw: Vec::with_capacity(strip_capacity),
            fax,
            strips: Vec::new(),
        })
    }

    /// A bound on the bytes of a one-page file of this page: its header,
    /// its strips at the most their coding expands the rows to, and its
    /// directory.
    fn file_bound(&self) -> u64 {
        let rows = u64::from(self.meta.height);
        let raw = self.row_bytes as u64 * rows;
        let strips = raw.div_ceil(u64::from(self.meta.rows_per_strip) * self.row_bytes as u64);
        let coded = match self.meta.compression {
            TiffCompression::None => raw,
            // A header byte per 128 bytes of a row.
            TiffCompression::PackBits => raw + rows * (self.row_bytes as u64).div_ceil(128),
            // At most 12 bits a byte, a clear code at most every 2^9
            // codes, and an end code a strip.
            TiffCompression::Lzw => raw.div_ceil(2) * 3 + raw / 256 + 4 * strips,
            // Stored blocks: 5 bytes a 64 KiB block, and the zlib wrapper.
            TiffCompression::Deflate => raw + 5 * raw.div_ceil(65_535) + 11 * strips,
            // A change costs at most 14 bits of code in G3 and G4, and a
            // row holds at most one change a pixel.
            TiffCompression::Group3 { .. } | TiffCompression::Group4 => {
                rows * (u64::from(self.meta.width) * 2 + 8)
            }
        };
        let directory = directory_len(
            &self.meta,
            usize::try_from(strips).unwrap_or(usize::MAX),
            Format::Big,
        );
        Format::Big.header_len() + coded + 1 + directory as u64
    }

    /// Encode the rows of `band`, the next rows of the page.
    ///
    /// # Errors
    ///
    /// [`EncodeError::InvalidInput`] for a band of another width, of more
    /// rows than the page has left, or of colour for a gray page or the
    /// reverse.
    pub fn push(&mut self, band: RowImage<'_>) -> Result<(), EncodeError> {
        self.rows.take(&band, self.layout)?;
        for y in 0..band.size().1 {
            band.row(y, self.layout, &mut self.raw);
            if let Some(fax) = &mut self.fax {
                fax.encode_packed_row(&self.raw);
                self.raw.clear();
            } else if self.raw.len() == self.row_bytes * self.meta.rows_per_strip as usize {
                self.seal_strip()?;
            }
        }
        Ok(())
    }

    /// Compress the rows held as the next strip.
    fn seal_strip(&mut self) -> Result<(), EncodeError> {
        let strip = compress(&self.raw, self.row_bytes, self.meta.compression)?;
        self.strips.push(strip);
        self.raw.clear();
        Ok(())
    }

    /// The strips compressed since the last call, in page order.
    fn take_strips(&mut self) -> Vec<Vec<u8>> {
        std::mem::take(&mut self.strips)
    }

    /// Compress the last strip and return what is left of the page: its
    /// directory's record and the strips not yet taken.
    fn finish_strips(mut self) -> Result<(PageMeta, Vec<Vec<u8>>), EncodeError> {
        self.rows.finished()?;
        if let Some(fax) = self.fax.take() {
            self.strips.push(fax.finish());
        } else if !self.raw.is_empty() {
            self.seal_strip()?;
        }
        Ok((self.meta, self.strips))
    }

    /// The page, once every row is pushed.
    ///
    /// # Errors
    ///
    /// [`EncodeError::InvalidInput`] when rows are missing.
    pub fn finish(self) -> Result<TiffPage, EncodeError> {
        let (meta, strips) = self.finish_strips()?;
        Ok(TiffPage { meta, strips })
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
        image: RowImage<'_>,
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
    /// [`EncodeError::InvalidInput`] without pages or with more than
    /// 65535, found before anything is written; [`EncodeError::Io`] on a
    /// write failure.  A file past the 4 GiB of classic TIFF is written as
    /// `BigTIFF`.
    pub fn write<W: Write>(self, mut out: W) -> Result<(), EncodeError> {
        if self.pages.is_empty() {
            return Err(EncodeError::InvalidInput("a TIFF needs at least one page"));
        }
        let count = u16::try_from(self.pages.len())
            .map_err(|_| EncodeError::InvalidInput("a TIFF holds at most 65535 pages"))?;
        let one_of_many = self.fax_document || count > 1;

        // Lay the file out: header, then per page its strips, a pad byte
        // to an even offset, and its directory with the values that
        // overflow it.  A file whose classic layout passes 4 GiB is laid
        // out as BigTIFF.
        let layout = |format: Format| {
            let mut layouts = Vec::with_capacity(self.pages.len());
            let mut end = format.header_len();
            for page in &self.pages {
                let mut strip_offsets = Vec::with_capacity(page.strips.len());
                for strip in &page.strips {
                    strip_offsets.push(end);
                    end += strip.len() as u64;
                }
                end += end % 2;
                let ifd = end;
                end += directory_len(&page.meta, page.strips.len(), format) as u64;
                layouts.push((strip_offsets, ifd));
            }
            (layouts, end)
        };
        let (mut layouts, end) = layout(Format::Classic);
        let format = Format::for_len(end);
        if format == Format::Big {
            layouts = layout(Format::Big).0;
        }

        out.write_all(&format.header(layouts[0].1))?;
        let mut at = format.header_len();
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
            let next = layouts
                .get(usize::from(number) + 1)
                .map_or(0, |&(_, next)| next);
            let directory = directory(
                &page.meta,
                strip_offsets,
                &byte_counts(&page.strips),
                (*ifd, next),
                (number, count),
                one_of_many,
                format,
            );
            out.write_all(&directory)?;
            at += directory.len() as u64;
        }
        out.flush()?;
        Ok(())
    }
}

/// A one-page TIFF written as its rows arrive, band by band, to a seekable
/// sink.
///
/// Each strip is written as it is compressed, the directory last, and the
/// header's link to it patched in place.  The file holds the bytes
/// [`write_tiff`] writes for the same page.
pub struct TiffStream<W: Write + Seek> {
    file: StripFile<W>,
    encoder: TiffPageEncoder,
}

/// A one-page file being written: its sink and where its strips went.
struct StripFile<W> {
    out: W,
    format: Format,
    /// The sink's position at the header.
    start: u64,
    /// Bytes written from the header on.
    at: u64,
    offsets: Vec<u64>,
    counts: Vec<u64>,
}

impl<W: Write + Seek> StripFile<W> {
    /// Append `strips`, recording where each lies.
    fn write_strips(&mut self, strips: &[Vec<u8>]) -> Result<(), EncodeError> {
        for strip in strips {
            self.offsets.push(self.at);
            self.counts.push(strip.len() as u64);
            self.out.write_all(strip)?;
            self.at += strip.len() as u64;
        }
        if self.format == Format::Classic && u32::try_from(self.at).is_err() {
            return Err(TOO_LARGE);
        }
        Ok(())
    }

    /// Write the directory of `meta` after the strips, patch the header's
    /// link to it, and return the sink at the end of the file.
    fn finish(mut self, meta: &PageMeta) -> Result<W, EncodeError> {
        if self.at % 2 == 1 {
            self.out.write_all(&[0])?;
            self.at += 1;
        }
        let ifd = self.at;
        let end = ifd + directory_len(meta, self.offsets.len(), self.format) as u64;
        if self.format == Format::Classic && u32::try_from(end).is_err() {
            return Err(TOO_LARGE);
        }
        let directory = directory(
            meta,
            &self.offsets,
            &self.counts,
            (ifd, 0),
            (0, 1),
            false,
            self.format,
        );
        self.out.write_all(&directory)?;
        let _: u64 = self
            .out
            .seek(SeekFrom::Start(self.start + self.format.link_at()))?;
        self.out.write_all(&self.format.offset(ifd))?;
        let _: u64 = self.out.seek(SeekFrom::Start(self.start + end))?;
        self.out.flush()?;
        Ok(self.out)
    }
}

impl<W: Write + Seek> TiffStream<W> {
    /// Begin a `width × height` page stored as `layout` at `dpi`
    /// (horizontal, vertical) pixels per inch, compressed with
    /// `compression`, on `out` from its position: write the header.  The
    /// file is `BigTIFF` when the most its compressed strips can hold would
    /// pass the 4 GiB of classic TIFF.
    ///
    /// # Errors
    ///
    /// As [`TiffPageEncoder::new`]; [`EncodeError::Io`] on a write
    /// failure.
    pub fn new(
        out: W,
        (width, height): (u32, u32),
        layout: RowLayout,
        dpi: (f64, f64),
        compression: TiffCompression,
    ) -> Result<Self, EncodeError> {
        let encoder = TiffPageEncoder::new(width, height, layout, dpi, compression)?;
        let format = Format::for_len(encoder.file_bound());
        Self::with_format(out, encoder, format)
    }

    /// [`Self::new`] for `encoder`'s page, laid out as `format`.
    fn with_format(
        mut out: W,
        encoder: TiffPageEncoder,
        format: Format,
    ) -> Result<Self, EncodeError> {
        let start = out.stream_position()?;
        // The directory's offset, patched once it is known.
        out.write_all(&format.header(0))?;
        Ok(Self {
            file: StripFile {
                out,
                format,
                start,
                at: format.header_len(),
                offsets: Vec::new(),
                counts: Vec::new(),
            },
            encoder,
        })
    }

    /// Encode the rows of `band`, the next rows of the page, and write the
    /// strips they complete.
    ///
    /// # Errors
    ///
    /// As [`TiffPageEncoder::push`]; [`EncodeError::InvalidInput`] once a
    /// classic file passes 4 GiB, which a coding expanding its rows past
    /// the bound [`Self::new`] chose the format by can make it do;
    /// [`EncodeError::Io`] on a write failure.
    pub fn push(&mut self, band: RowImage<'_>) -> Result<(), EncodeError> {
        self.encoder.push(band)?;
        let strips = self.encoder.take_strips();
        self.file.write_strips(&strips)
    }

    /// Write the last strip and the directory, and return the sink, at the
    /// end of the file.
    ///
    /// # Errors
    ///
    /// As [`TiffPageEncoder::finish`] and [`Self::push`].
    pub fn finish(self) -> Result<W, EncodeError> {
        let Self { mut file, encoder } = self;
        let (meta, strips) = encoder.finish_strips()?;
        file.write_strips(&strips)?;
        file.finish(&meta)
    }
}

/// Write `image` as a one-page TIFF to `out`.
///
/// # Errors
///
/// As [`TiffPage::encode`] and [`TiffWriter::write`].
pub fn write_tiff<W: Write>(
    image: RowImage<'_>,
    dpi: (f64, f64),
    compression: TiffCompression,
    out: W,
) -> Result<(), EncodeError> {
    let mut writer = TiffWriter::new();
    writer.add_page(image, dpi, compression)?;
    writer.write(out)
}

/// The byte counts of `strips`.
fn byte_counts(strips: &[Vec<u8>]) -> Vec<u64> {
    strips.iter().map(|s| s.len() as u64).collect()
}

/// `bytes`, stored rows of `row_bytes` each, compressed with
/// `compression`, which is not a CCITT coding.  `PackBits` runs do not
/// cross a row.
fn compress(
    bytes: &[u8],
    row_bytes: usize,
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
        TiffCompression::Group3 { .. } | TiffCompression::Group4 => {
            unreachable!("a CCITT page is coded row by row as its rows arrive")
        }
    })
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
const LONG8: u16 = 16;

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

/// Strip offsets or byte counts: `LONG` in a classic file, `LONG8` in `BigTIFF`.
fn offsets_entry(tag: u16, values: &[u64], format: Format) -> Entry {
    Entry {
        tag,
        kind: match format {
            Format::Classic => LONG,
            Format::Big => LONG8,
        },
        count: u32::try_from(values.len()).expect("strip count fits u32"),
        bytes: values.iter().flat_map(|&v| format.offset(v)).collect(),
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
/// hold `byte_counts` bytes, in a file of `format`, sorted by tag.  Their
/// sizes do not depend on the offset and count values.
fn entries(
    page: &PageMeta,
    (offsets, byte_counts): (&[u64], &[u64]),
    position: (u16, u16),
    one_of_many: bool,
    format: Format,
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
        offsets_entry(273, offsets, format),
        shorts(277, &[samples]),
        longs(278, &[page.rows_per_strip]),
        offsets_entry(279, byte_counts, format),
        rational_entry(282, page.resolution[0]),
        rational_entry(283, page.resolution[1]),
        shorts(284, &[1]),
        shorts(296, &[2]),
        shorts(297, &<[u16; 2]>::from(position)),
    ];
    if bits == 1 {
        entries.push(shorts(266, &[1]));
    }
    // ExtraSamples: the fourth sample of an RGB page is unassociated alpha.
    if photometric == PHOTOMETRIC_RGB && samples == 4 {
        entries.push(shorts(338, &[2]));
    }
    if let Some((tag, options)) = page.compression.options_tag() {
        entries.push(longs(tag, &[options]));
    }
    entries.sort_by_key(|e| e.tag);
    entries
}

/// The length in bytes of the directory of `page`, of `strips` strips,
/// in a file of `format`, with its overflowing values: the entry count,
/// the entries, the link, then each value longer than an entry's value
/// field padded to an even length.
fn directory_len(page: &PageMeta, strips: usize, format: Format) -> usize {
    let placeholder = vec![0; strips];
    let entries = entries(page, (&placeholder, &placeholder), (0, 1), false, format);
    format.count_bytes()
        + format.entry_bytes() * entries.len()
        + format.offset_bytes()
        + entries
            .iter()
            .filter(|e| e.bytes.len() > format.offset_bytes())
            .map(|e| e.bytes.len() + e.bytes.len() % 2)
            .sum::<usize>()
}

/// The directory of `page` placed at file offset `at`, its strips at
/// `offsets` holding `byte_counts` bytes, linking to the directory at
/// `next` (0 for none), followed by its overflowing values, in a file of
/// `format`.  The caller has checked that a classic file, the directory
/// included, ends within 4 GiB.
fn directory(
    page: &PageMeta,
    offsets: &[u64],
    byte_counts: &[u64],
    (at, next): (u64, u64),
    position: (u16, u16),
    one_of_many: bool,
    format: Format,
) -> Vec<u8> {
    let entries = entries(page, (offsets, byte_counts), position, one_of_many, format);
    let inline = format.offset_bytes();
    let mut overflow =
        at + (format.count_bytes() + format.entry_bytes() * entries.len() + inline) as u64;
    let mut directory = Vec::with_capacity(directory_len(page, offsets.len(), format));
    let mut spilled = Vec::new();
    let n = entries.len() as u64;
    directory.extend_from_slice(&n.to_le_bytes()[..format.count_bytes()]);
    for e in &entries {
        directory.extend_from_slice(&e.tag.to_le_bytes());
        directory.extend_from_slice(&e.kind.to_le_bytes());
        directory.extend_from_slice(&u64::from(e.count).to_le_bytes()[..inline]);
        if e.bytes.len() <= inline {
            let mut field = [0u8; 8];
            field[..e.bytes.len()].copy_from_slice(&e.bytes);
            directory.extend_from_slice(&field[..inline]);
        } else {
            directory.extend_from_slice(&format.offset(overflow));
            overflow += (e.bytes.len() + e.bytes.len() % 2) as u64;
            spilled.extend_from_slice(&e.bytes);
            if e.bytes.len() % 2 == 1 {
                spilled.push(0);
            }
        }
    }
    directory.extend_from_slice(&format.offset(next));
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

    use super::{
        RowImage, RowLayout, TiffCompression, TiffPage, TiffWriter, pack_bits, write_tiff,
    };
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
            write_tiff(RowImage::Gray(&g), (150.0, 150.0), compression, &mut out).unwrap();
            let pixels = bytes(decode(&out).read_image().unwrap());
            let want: Vec<u8> = (0..g.height)
                .flat_map(|y| g.row_bytes(y)[..37].to_vec())
                .collect();
            assert_eq!(pixels, want, "gray, {compression:?}");

            let mut out = Vec::new();
            write_tiff(RowImage::Rgb(&c), (72.0, 96.0), compression, &mut out).unwrap();
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
    fn an_rgb_page_with_alpha_reads_back_as_rgba_under_every_compression() {
        let c = rgb(9, 20);
        let alpha: Vec<u8> = (0..180u32).map(|i| (i * 13).to_le_bytes()[0]).collect();
        let mut with_alpha = Bitmap::<Rgb8>::new(9, 20, 4, true);
        for y in 0..20 {
            with_alpha.row_bytes_mut(y)[..27].copy_from_slice(&c.row_bytes(y)[..27]);
        }
        with_alpha
            .alpha_plane_mut()
            .unwrap()
            .copy_from_slice(&alpha);
        let want: Vec<u8> = (0..20u32)
            .flat_map(|y| {
                let rgb = with_alpha.row_bytes(y)[..27].to_vec();
                let a = with_alpha.alpha_row(y).unwrap().to_vec();
                rgb.chunks(3)
                    .zip(a)
                    .flat_map(|(px, a)| [px[0], px[1], px[2], a])
                    .collect::<Vec<u8>>()
            })
            .collect();
        for compression in ALL {
            let mut out = Vec::new();
            write_tiff(
                RowImage::Rgb(&with_alpha),
                (72.0, 72.0),
                compression,
                &mut out,
            )
            .unwrap();
            let mut decoder = decode(&out);
            assert_eq!(
                decoder.colortype().unwrap(),
                ::tiff::ColorType::RGBA(8),
                "{compression:?}"
            );
            assert_eq!(
                bytes(decoder.read_image().unwrap()),
                want,
                "{compression:?}"
            );
        }
    }

    #[test]
    fn bilevel_rows_pack_black_as_one() {
        let g = gray(13, 3);
        for compression in ALL {
            let mut out = Vec::new();
            write_tiff(RowImage::Bilevel(&g), (204.0, 196.0), compression, &mut out).unwrap();
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
            RowImage::Bilevel(g).row(y, RowLayout::Bilevel, &mut rows);
        }
        rows
    }

    #[test]
    fn group4_pages_read_back_through_an_independent_decoder() {
        let g = drawing(1728, 120);
        let mut out = Vec::new();
        write_tiff(
            RowImage::Bilevel(&g),
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
                RowImage::Bilevel(&g),
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
            let result = TiffPage::encode(RowImage::Gray(&g), (204.0, 196.0), compression);
            assert!(matches!(result, Err(EncodeError::UnsupportedMode(_))));
        }
    }

    #[test]
    fn a_fax_document_marks_even_a_single_page() {
        let g = drawing(16, 4);
        for (mut writer, flag) in [(TiffWriter::new(), 0), (TiffWriter::fax_document(), 2)] {
            writer
                .add_page(
                    RowImage::Bilevel(&g),
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
                    RowImage::Bilevel(&g),
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
                TiffPage::encode(RowImage::Gray(&b), (100.0, 100.0), TiffCompression::Lzw)
            });
            let first = TiffPage::encode(
                RowImage::Bilevel(&a),
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
        assert!(invalid(RowImage::Gray(&empty), (72.0, 72.0)));
        assert!(invalid(RowImage::Gray(&g), (0.0, 72.0)));
        assert!(invalid(RowImage::Gray(&g), (f64::NAN, 72.0)));
        assert!(invalid(RowImage::Gray(&g), (5e6, 72.0)));
        assert!(matches!(
            TiffWriter::new().write(Vec::new()),
            Err(EncodeError::InvalidInput(_))
        ));
        let mut file = Vec::new();
        write_tiff(
            RowImage::Gray(&g),
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

    /// A page laid out as `BigTIFF`, as a file past 4 GiB is, reads back
    /// through an independent decoder, whole and streamed alike.
    #[test]
    fn bigtiff_pages_read_back() {
        use super::{Format, TiffPageEncoder, TiffStream};
        let (g, c) = (gray(37, 300), rgb(21, 50));
        for compression in ALL {
            for (image, layout, w, h) in [
                (RowImage::Gray(&g), RowLayout::Gray, 37, 300),
                (RowImage::Rgb(&c), RowLayout::Rgb, 21, 50),
            ] {
                let encoder =
                    TiffPageEncoder::new(w, h, layout, (150.0, 150.0), compression).unwrap();
                let mut stream =
                    TiffStream::with_format(Cursor::new(Vec::new()), encoder, Format::Big).unwrap();
                stream.push(image).unwrap();
                let file = stream.finish().unwrap().into_inner();
                assert_eq!(&file[..4], b"II+\0", "a BigTIFF header");
                let mut classic = Vec::new();
                write_tiff(image, (150.0, 150.0), compression, &mut classic).unwrap();
                assert_eq!(
                    bytes(decode(&file).read_image().unwrap()),
                    bytes(decode(&classic).read_image().unwrap()),
                    "{layout:?} {compression:?}"
                );
            }
        }
    }

    /// A file is classic while its last byte lies within 32-bit offsets.
    #[test]
    fn the_format_follows_the_file_length() {
        use super::Format;
        assert_eq!(Format::for_len(u64::from(u32::MAX)), Format::Classic);
        assert_eq!(Format::for_len(u64::from(u32::MAX) + 1), Format::Big);
    }

    /// The bound a streamed page chooses its format by covers what each
    /// coding writes, on rows that compress worst: noise.
    #[test]
    fn the_file_bound_covers_the_written_file() {
        use super::{TiffPageEncoder, TiffStream};
        let mut noise = Bitmap::<Gray8>::new(255, 64, 1, false);
        let mut state = 0x9e37_79b9_u32;
        for v in noise.data_mut() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *v = state.to_le_bytes()[0];
        }
        for compression in ALL.into_iter().chain([TiffCompression::Group4]) {
            let layout = if compression.is_fax() {
                RowLayout::Bilevel
            } else {
                RowLayout::Gray
            };
            let bound = TiffPageEncoder::new(255, 64, layout, (72.0, 72.0), compression)
                .unwrap()
                .file_bound();
            let mut stream = TiffStream::new(
                Cursor::new(Vec::new()),
                (255, 64),
                layout,
                (72.0, 72.0),
                compression,
            )
            .unwrap();
            stream.push(RowImage::Gray(&noise)).unwrap();
            let written = stream.finish().unwrap().into_inner().len() as u64;
            assert!(written <= bound, "{compression:?}: {written} > {bound}");
        }
    }
}
