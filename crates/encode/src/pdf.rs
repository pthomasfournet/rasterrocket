//! Raster pages as a PDF: each page one 8- or 16-bit image,
//! Flate-compressed, spanning a media box of the image's size at its
//! resolution.  A file of 16-bit pages is PDF 1.5, the first version
//! whose images take 16 bits per component.  A file tagged with an ICC
//! profile holds it once, as object 3, and every page image's colour space
//! is `[/ICCBased 3 0 R]`; the file's version is then at least the one
//! that admits the profile's (PDF 1.7, Table 66: ICC 4.0 from PDF 1.5,
//! 4.1 from 1.6, 4.2 from 1.7).
//!
//! The file is written as it goes: a page's image, content stream and page
//! object when the page is added, the page tree, catalogue and
//! cross-reference table when it is finished.  Objects 1 and 2 are the
//! catalogue and the page tree, written last; a page takes the three
//! objects that follow the previous page's.

use std::fs::File;
use std::io::{self, BufWriter, Read, Write};

use flate2::Compression;
use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;

use crate::EncodeError;
use crate::icc::IccProfile;
use crate::rows::{ByteOrder, RowCount, RowImage, RowLayout};
use color::SampleDepth;

/// The largest byte offset a cross-reference entry's ten digits hold.
const MAX_XREF_OFFSET: u64 = 9_999_999_999;

/// The smallest page side, in points, that four decimals write as nonzero.
const MIN_PAGE_PTS: f64 = 0.001;

/// A page's image, compressed, and its size.
pub struct PdfPage {
    width: u32,
    height: u32,
    rgb: bool,
    depth: SampleDepth,
    /// The page's width and height in points.
    size_pts: (f64, f64),
    data: PageData,
}

/// Where a page's compressed image is held.
enum PageData {
    Memory(Vec<u8>),
    /// In `file`, from its start, `len` bytes.
    Spilled {
        file: File,
        len: u64,
    },
}

impl PageData {
    const fn len(&self) -> u64 {
        match self {
            Self::Memory(data) => data.len() as u64,
            Self::Spilled { len, .. } => *len,
        }
    }

    /// Copy the bytes to `out`.
    fn copy_to(&self, out: &mut impl Write) -> io::Result<()> {
        match self {
            Self::Memory(data) => out.write_all(data),
            Self::Spilled { file, len } => {
                let copied = io::copy(&mut SpillReader::new(file, *len), out)?;
                if copied == *len {
                    Ok(())
                } else {
                    Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "a page's spilled image is shorter than was written",
                    ))
                }
            }
        }
    }
}

/// The first `len` bytes of a spilled page's `file`, read by positional
/// reads, which neither use nor move the file's cursor on Unix and do not
/// depend on it on Windows: readers of one page, alive at once, do not
/// move each other.
struct SpillReader<'a> {
    file: &'a File,
    at: u64,
    len: u64,
}

impl<'a> SpillReader<'a> {
    const fn new(file: &'a File, len: u64) -> Self {
        Self { file, at: 0, len }
    }
}

impl Read for SpillReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let left = usize::try_from(self.len - self.at).unwrap_or(usize::MAX);
        let want = left.min(buf.len());
        if want == 0 {
            return Ok(0);
        }
        let read = read_at(self.file, &mut buf[..want], self.at)?;
        self.at += u64::try_from(read).map_err(io::Error::other)?;
        Ok(read)
    }
}

/// Read into `buf` from `file` at byte `at`.
#[cfg(unix)]
fn read_at(file: &File, buf: &mut [u8], at: u64) -> io::Result<usize> {
    std::os::unix::fs::FileExt::read_at(file, buf, at)
}

/// Read into `buf` from `file` at byte `at`.
#[cfg(windows)]
fn read_at(file: &File, buf: &mut [u8], at: u64) -> io::Result<usize> {
    std::os::windows::fs::FileExt::seek_read(file, buf, at)
}

impl PdfPage {
    /// Compress `image`, a page at `dpi` (horizontal, vertical) pixels per
    /// inch: `DeviceRGB` for an RGB image, `DeviceGray` (0 black) for a
    /// gray or bilevel one, of the image's depth (a bilevel image is 8-bit
    /// gray).
    ///
    /// # Errors
    ///
    /// As [`PdfPageEncoder::new`].
    pub fn encode(image: RowImage<'_>, dpi: (f64, f64)) -> Result<Self, EncodeError> {
        let layout = match image {
            RowImage::Rgb(_) => RowLayout::Rgb,
            RowImage::Gray(_) | RowImage::Bilevel(_) => RowLayout::Gray,
            RowImage::Rgb16(_) => RowLayout::Rgb16,
            RowImage::Gray16(_) => RowLayout::Gray16,
            RowImage::Cmyk(_) => RowLayout::Cmyk,
            RowImage::Cmyk16(_) => RowLayout::Cmyk16,
        };
        let mut encoder = PdfPageEncoder::new(image.size(), layout, dpi, None)?;
        encoder.push(image)?;
        encoder.finish()
    }

    /// The image's width and height in pixels.
    #[must_use]
    pub const fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Whether the image is RGB; else it is gray.
    #[must_use]
    pub const fn is_rgb(&self) -> bool {
        self.rgb
    }

    /// The depth of the image's samples.
    #[must_use]
    pub const fn depth(&self) -> SampleDepth {
        self.depth
    }

    /// The page's width and height in points: the image's size at its
    /// resolution.
    #[must_use]
    pub const fn size_pts(&self) -> (f64, f64) {
        self.size_pts
    }

    /// The image's rows, decompressed, top row first: per pixel R, G and B
    /// for an RGB image and one gray sample (0 black) otherwise, a 16-bit
    /// sample most significant byte first, rows unpadded.  Any number of
    /// readers of one page may be alive at once.
    #[must_use]
    pub fn rows(&self) -> impl Read + '_ {
        let data: Box<dyn Read + '_> = match &self.data {
            PageData::Memory(data) => Box::new(data.as_slice()),
            PageData::Spilled { file, len } => Box::new(SpillReader::new(file, *len)),
        };
        ZlibDecoder::new(data)
    }
}

/// The sink of a page's compressed image.
enum Compressing {
    Memory(ZlibEncoder<Vec<u8>>),
    Spill(ZlibEncoder<Counting<BufWriter<File>>>),
}

/// A page image compressed as its rows arrive, band by band, into memory
/// or into a file of its own, so a page in bands is held as compressed
/// data and one row.
pub struct PdfPageEncoder {
    width: u32,
    height: u32,
    layout: RowLayout,
    size_pts: (f64, f64),
    rows: RowCount,
    row: Vec<u8>,
    sink: Compressing,
}

impl PdfPageEncoder {
    /// An encoder of a `width × height` page image stored as `layout` (RGB
    /// or gray, of 8 or 16 bits) at `dpi` (horizontal, vertical) pixels per
    /// inch, compressed
    /// into `spill` from its start when given (an empty file of the
    /// caller's) and into memory otherwise.
    ///
    /// # Errors
    ///
    /// [`EncodeError::InvalidInput`] for an empty image, a resolution that
    /// is not positive and finite, a page side under 0.001 point, or a
    /// layout other than RGB and gray (alpha and bilevel layouts).
    pub fn new(
        (width, height): (u32, u32),
        layout: RowLayout,
        dpi: (f64, f64),
        spill: Option<File>,
    ) -> Result<Self, EncodeError> {
        let rows = RowCount::new(width, height).map_err(|_| {
            EncodeError::InvalidInput(
                "zero-width or zero-height bitmap: a PDF page cannot hold an empty image",
            )
        })?;
        if !matches!(
            layout,
            RowLayout::Rgb | RowLayout::Gray | RowLayout::Rgb16 | RowLayout::Gray16
        ) {
            return Err(EncodeError::InvalidInput(
                "a PDF page image is RGB or gray, of 8 or 16 bits",
            ));
        }
        if !(dpi.0.is_finite() && dpi.1.is_finite() && dpi.0 > 0.0 && dpi.1 > 0.0) {
            return Err(EncodeError::InvalidInput(
                "a PDF page needs a positive resolution",
            ));
        }
        let size_pts = (
            f64::from(width) * 72.0 / dpi.0,
            f64::from(height) * 72.0 / dpi.1,
        );
        if size_pts.0 < MIN_PAGE_PTS || size_pts.1 < MIN_PAGE_PTS {
            return Err(EncodeError::InvalidInput(
                "a PDF page side is at least 0.001 point: lower the resolution",
            ));
        }
        let sink = spill.map_or_else(
            || Compressing::Memory(ZlibEncoder::new(Vec::new(), Compression::fast())),
            |file| {
                Compressing::Spill(ZlibEncoder::new(
                    Counting {
                        out: BufWriter::with_capacity(crate::PACKED_WRITE_BYTES, file),
                        written: 0,
                    },
                    Compression::fast(),
                ))
            },
        );
        Ok(Self {
            width,
            height,
            layout,
            size_pts,
            rows,
            row: Vec::with_capacity(layout.row_bytes(width)),
            sink,
        })
    }

    /// Compress the rows of `band`, the next rows of the page.
    ///
    /// # Errors
    ///
    /// [`EncodeError::InvalidInput`] for a band of another width, of more
    /// rows than the page has left, or of colour for a gray page or the
    /// reverse; [`EncodeError::Io`] on a write failure.
    pub fn push(&mut self, band: RowImage<'_>) -> Result<(), EncodeError> {
        self.rows.take(&band, self.layout)?;
        for y in 0..band.size().1 {
            self.row.clear();
            band.row(y, self.layout, ByteOrder::Big, &mut self.row);
            match &mut self.sink {
                Compressing::Memory(z) => z.write_all(&self.row)?,
                Compressing::Spill(z) => z.write_all(&self.row)?,
            }
        }
        Ok(())
    }

    /// The page, once every row is pushed.
    ///
    /// # Errors
    ///
    /// [`EncodeError::InvalidInput`] when rows are missing;
    /// [`EncodeError::Io`] on a write failure.
    pub fn finish(self) -> Result<PdfPage, EncodeError> {
        self.rows.finished()?;
        let data = match self.sink {
            Compressing::Memory(z) => PageData::Memory(z.finish()?),
            Compressing::Spill(z) => {
                let counting = z.finish()?;
                let len = counting.written;
                let file = counting
                    .out
                    .into_inner()
                    .map_err(io::IntoInnerError::into_error)?;
                PageData::Spilled { file, len }
            }
        };
        Ok(PdfPage {
            width: self.width,
            height: self.height,
            rgb: matches!(self.layout, RowLayout::Rgb | RowLayout::Rgb16),
            depth: if self.layout.bits_per_sample() == 16 {
                SampleDepth::Sixteen
            } else {
                SampleDepth::Eight
            },
            size_pts: self.size_pts,
            data,
        })
    }
}

/// A PDF of raster pages, written page by page.
#[derive(Debug)]
pub struct PdfWriter<W: Write> {
    out: Counting<W>,
    /// The byte offset of each object, from object 1.
    offsets: Vec<u64>,
    /// The page objects.
    pages: Vec<usize>,
    /// The deepest page image the file's version admits.
    depth: SampleDepth,
    /// The colour channels of the profile every page image is tagged
    /// with, object 3.
    icc_channels: Option<u8>,
}

impl<W: Write> PdfWriter<W> {
    /// Begin a PDF on `out` whose page images are at most `depth` deep
    /// and, when `icc` is given, are all of its colour space and tagged
    /// with it: write its header, of PDF 1.4 for 8-bit pages and of PDF
    /// 1.5 for 16-bit ones, or later for the profile's version, and the
    /// profile.
    ///
    /// # Errors
    ///
    /// [`EncodeError::Io`] on a write failure.
    pub fn new(out: W, depth: SampleDepth, icc: Option<&IccProfile>) -> Result<Self, EncodeError> {
        let mut out = Counting { out, written: 0 };
        let minor = match depth {
            SampleDepth::Eight => 4,
            SampleDepth::Sixteen => 5,
        };
        let minor = icc.map_or(minor, |icc| match icc.version() {
            (4, 0) => minor.max(5),
            (4, 1) => minor.max(6),
            (4, _) => 7,
            _ => minor,
        });
        write!(out, "%PDF-1.{minor}")?;
        out.write_all(b"\n%\xE2\xE3\xCF\xD3\n")?;
        let mut writer = Self {
            out,
            // The catalogue and the page tree, placed when written.
            offsets: vec![0, 0],
            pages: Vec::new(),
            depth,
            icc_channels: icc.map(IccProfile::channels),
        };
        if let Some(icc) = icc {
            writer.write_profile(icc)?;
        }
        Ok(writer)
    }

    /// Write `icc` as object 3, Flate-compressed, its alternate space the
    /// device space of its channels.
    fn write_profile(&mut self, icc: &IccProfile) -> Result<(), EncodeError> {
        let mut z = ZlibEncoder::new(Vec::new(), Compression::default());
        z.write_all(icc.bytes())?;
        let data = z.finish()?;
        self.begin(3)?;
        write!(
            self.out,
            "<</N {}/Alternate/{}/Filter/FlateDecode/Length {}>>\nstream\n",
            icc.channels(),
            device_space(icc.channels() == 3),
            data.len()
        )?;
        self.out.write_all(&data)?;
        self.out.write_all(b"\nendstream\nendobj\n")?;
        Ok(())
    }

    /// Append `page`.
    ///
    /// # Errors
    ///
    /// [`EncodeError::InvalidInput`] for a 16-bit page in a writer made for
    /// 8-bit pages, or a page of another colour space than the writer's
    /// profile; [`EncodeError::Io`] on a write failure.
    pub fn write_page(&mut self, page: &PdfPage) -> Result<(), EncodeError> {
        if page.depth == SampleDepth::Sixteen && self.depth == SampleDepth::Eight {
            return Err(EncodeError::InvalidInput(
                "a 16-bit page image needs a PDF 1.5 file: make the writer for 16-bit pages",
            ));
        }
        let space = match self.icc_channels {
            None => format!("/{}", device_space(page.rgb)),
            Some(channels) if (channels == 3) == page.rgb => "[/ICCBased 3 0 R]".to_owned(),
            Some(_) => {
                return Err(EncodeError::InvalidInput(
                    "a page image of another colour space than the PDF's ICC profile",
                ));
            }
        };
        let image = self.offsets.len() + 1;
        let (w, h) = (fmt_num(page.size_pts.0), fmt_num(page.size_pts.1));

        self.begin(image)?;
        write!(
            self.out,
            "<</Type/XObject/Subtype/Image/Width {}/Height {}/ColorSpace{space}\
             /BitsPerComponent {}/Filter/FlateDecode/Length {}>>\nstream\n",
            page.width,
            page.height,
            page.depth.bits(),
            page.data.len()
        )?;
        page.data.copy_to(&mut self.out)?;
        self.out.write_all(b"\nendstream\nendobj\n")?;

        let content = format!("q {w} 0 0 {h} 0 0 cm /Im0 Do Q");
        self.begin(image + 1)?;
        write!(
            self.out,
            "<</Length {}>>\nstream\n{content}\nendstream\nendobj\n",
            content.len()
        )?;

        self.begin(image + 2)?;
        write!(
            self.out,
            "<</Type/Page/Parent 2 0 R/MediaBox[0 0 {w} {h}]\
             /Resources<</XObject<</Im0 {image} 0 R>>>>/Contents {} 0 R>>\nendobj\n",
            image + 1
        )?;
        self.pages.push(image + 2);
        Ok(())
    }

    /// Write the page tree, catalogue and cross-reference table, and
    /// return the sink.
    ///
    /// # Errors
    ///
    /// [`EncodeError::InvalidInput`] when no page was written, as a PDF
    /// holds at least one; [`EncodeError::Io`] on a write failure.
    pub fn finish(mut self) -> Result<W, EncodeError> {
        if self.pages.is_empty() {
            return Err(EncodeError::InvalidInput("a PDF needs at least one page"));
        }
        self.offsets[1] = self.indexable_offset()?;
        let kids: Vec<String> = self.pages.iter().map(|p| format!("{p} 0 R")).collect();
        write!(
            self.out,
            "2 0 obj\n<</Type/Pages/Kids[{}]/Count {}>>\nendobj\n",
            kids.join(" "),
            self.pages.len()
        )?;
        self.offsets[0] = self.indexable_offset()?;
        self.out
            .write_all(b"1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n")?;

        let xref = self.out.written;
        let objects = self.offsets.len() + 1;
        write!(self.out, "xref\n0 {objects}\n0000000000 65535 f\r\n")?;
        for offset in &self.offsets {
            write!(self.out, "{offset:010} 00000 n\r\n")?;
        }
        write!(
            self.out,
            "trailer\n<</Size {objects}/Root 1 0 R>>\nstartxref\n{xref}\n%%EOF\n"
        )?;
        Ok(self.out.out)
    }

    /// Record object `id` as starting here and write its header.
    fn begin(&mut self, id: usize) -> Result<(), EncodeError> {
        debug_assert_eq!(id, self.offsets.len() + 1);
        let offset = self.indexable_offset()?;
        self.offsets.push(offset);
        writeln!(self.out, "{id} 0 obj")?;
        Ok(())
    }

    /// The current byte offset, refused past what a cross-reference entry
    /// holds.
    const fn indexable_offset(&self) -> Result<u64, EncodeError> {
        if self.out.written > MAX_XREF_OFFSET {
            return Err(EncodeError::InvalidInput(
                "a PDF of more than 9 999 999 999 bytes cannot be indexed by a \
                 cross-reference table: print fewer pages at once",
            ));
        }
        Ok(self.out.written)
    }
}

impl std::fmt::Debug for PdfPage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PdfPage")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("rgb", &self.rgb)
            .field("depth", &self.depth)
            .field("size_pts", &self.size_pts)
            .field("data_bytes", &self.data.len())
            .finish()
    }
}

/// The device colour space of an RGB image, or of a gray one.
const fn device_space(rgb: bool) -> &'static str {
    if rgb { "DeviceRGB" } else { "DeviceGray" }
}

/// A number as a PDF real: at most four decimals, trailing zeros dropped.
fn fmt_num(v: f64) -> String {
    let s = format!("{v:.4}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    s.to_owned()
}

/// A sink that counts the bytes written to it.
#[derive(Debug)]
struct Counting<W> {
    out: W,
    written: u64,
}

impl<W: Write> Write for Counting<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.out.write(buf)?;
        self.written += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.out.flush()
    }
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use color::{Gray8, Rgb8, SampleDepth};
    use flate2::read::ZlibDecoder;
    use raster::Bitmap;

    use super::{PdfPage, PdfWriter, RowImage, fmt_num};

    fn gray(w: u32, h: u32, f: impl Fn(u32, u32) -> u8) -> Bitmap<Gray8> {
        let mut bmp: Bitmap<Gray8> = Bitmap::new(w, h, 4, false);
        for y in 0..h {
            for (x, px) in (0..w).zip(bmp.row_bytes_mut(y).iter_mut()) {
                *px = f(x, y);
            }
        }
        bmp
    }

    fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
        hay.windows(needle.len()).position(|w| w == needle)
    }

    #[test]
    fn numbers_drop_trailing_zeros() {
        assert_eq!(fmt_num(612.0), "612");
        assert_eq!(fmt_num(595.275_590_55), "595.2756");
        assert_eq!(fmt_num(0.5), "0.5");
    }

    #[test]
    fn a_page_is_its_image_at_its_resolution() {
        // Letter at 100 dpi.
        let bmp = gray(850, 1100, |x, y| u8::try_from((x + y) % 256).unwrap());
        let page = PdfPage::encode(RowImage::Gray(&bmp), (100.0, 100.0)).unwrap();
        let mut writer = PdfWriter::new(Vec::new(), SampleDepth::Eight, None).unwrap();
        writer.write_page(&page).unwrap();
        let pdf = writer.finish().unwrap();
        let text = String::from_utf8_lossy(&pdf);
        assert!(text.starts_with("%PDF-1.4\n"));
        assert!(text.contains("/MediaBox[0 0 612 792]"));
        assert!(text.contains("q 612 0 0 792 0 0 cm /Im0 Do Q"));
        assert!(text.contains("/Width 850/Height 1100/ColorSpace/DeviceGray"));
        // The image stream inflates to the pixels, row by row.
        let start = find(&pdf, b"stream\n").unwrap() + 7;
        let mut pixels = Vec::new();
        let n = ZlibDecoder::new(&pdf[start..])
            .read_to_end(&mut pixels)
            .unwrap();
        assert_eq!(n, 850 * 1100);
        assert_eq!(&pixels[850 * 3..850 * 4], &bmp.row_bytes(3)[..850]);
    }

    /// A page reads back as its rows, held in memory or in a spill file,
    /// with its size, colour, depth and size in points; a spilled page
    /// reads back more than once.
    #[test]
    fn a_page_reads_back_as_its_rows() {
        let bmp = gray(5, 3, |x, y| u8::try_from(x * 10 + y).unwrap());
        let mut encoder = super::PdfPageEncoder::new(
            (5, 3),
            crate::RowLayout::Gray,
            (144.0, 72.0),
            Some(tempfile::tempfile().unwrap()),
        )
        .unwrap();
        encoder.push(RowImage::Gray(&bmp)).unwrap();
        let spilled = encoder.finish().unwrap();
        let memory = PdfPage::encode(RowImage::Gray(&bmp), (144.0, 72.0)).unwrap();
        let want: Vec<u8> = (0..3)
            .flat_map(|y| bmp.row_bytes(y)[..5].to_vec())
            .collect();
        for page in [&memory, &spilled, &spilled] {
            assert_eq!(page.size(), (5, 3));
            assert!(!page.is_rgb());
            assert_eq!(page.depth(), SampleDepth::Eight);
            assert_eq!(page.size_pts(), (2.5, 3.0));
            let mut rows = Vec::new();
            let _ = page.rows().read_to_end(&mut rows).unwrap();
            assert_eq!(rows, want);
        }
    }

    /// Two readers of one spill file, read in turns a byte at a time, each
    /// read its bytes whole: neither moves the other.
    #[test]
    fn readers_of_one_spill_file_do_not_move_each_other() {
        use std::io::Write as _;

        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"abcdefg").unwrap();
        let (mut a, mut b) = (
            super::SpillReader::new(&file, 6),
            super::SpillReader::new(&file, 6),
        );
        let (mut got_a, mut got_b) = (Vec::new(), Vec::new());
        let mut byte = [0];
        for _ in 0..8 {
            if a.read(&mut byte).unwrap() == 1 {
                got_a.push(byte[0]);
            }
            let mut pair = [0; 2];
            let n = b.read(&mut pair).unwrap();
            got_b.extend_from_slice(&pair[..n]);
        }
        assert_eq!(got_a, b"abcdef");
        assert_eq!(got_b, b"abcdef");
    }

    #[test]
    fn the_cross_reference_table_points_at_every_object() {
        let mut rgb: Bitmap<Rgb8> = Bitmap::new(3, 2, 4, false);
        rgb.row_bytes_mut(1)[..9].copy_from_slice(&[9; 9]);
        let page = PdfPage::encode(RowImage::Rgb(&rgb), (72.0, 144.0)).unwrap();
        let mut writer = PdfWriter::new(Vec::new(), SampleDepth::Eight, None).unwrap();
        writer.write_page(&page).unwrap();
        writer.write_page(&page).unwrap();
        let pdf = writer.finish().unwrap();
        let text = String::from_utf8_lossy(&pdf);
        assert!(text.contains("/MediaBox[0 0 3 1]"));
        assert!(text.contains("/ColorSpace/DeviceRGB"));
        assert!(text.contains("/Kids[5 0 R 8 0 R]/Count 2"));
        let startxref: usize = text
            .rsplit("startxref\n")
            .next()
            .unwrap()
            .lines()
            .next()
            .unwrap()
            .parse()
            .unwrap();
        // Offsets count bytes: read the table from the bytes themselves.
        let table = String::from_utf8(pdf[startxref..].to_vec()).unwrap();
        assert!(table.starts_with("xref\n0 9\n"));
        let entries = table
            .lines()
            .skip(3)
            .take(8)
            .map(|l| l[..10].parse::<usize>().unwrap());
        for (i, offset) in entries.enumerate() {
            let header = format!("{} 0 obj\n", i + 1);
            assert!(
                pdf[offset..].starts_with(header.as_bytes()),
                "object {}",
                i + 1
            );
        }
    }

    #[test]
    fn offsets_past_ten_digits_are_refused() {
        let page = PdfPage::encode(RowImage::Gray(&gray(1, 1, |_, _| 0)), (72.0, 72.0)).unwrap();
        let mut writer = PdfWriter::new(Vec::new(), SampleDepth::Eight, None).unwrap();
        writer.out.written = super::MAX_XREF_OFFSET + 1;
        assert!(writer.write_page(&page).is_err());
    }

    #[test]
    fn empty_documents_images_and_resolutions_are_refused() {
        assert!(
            PdfWriter::new(Vec::new(), SampleDepth::Eight, None)
                .unwrap()
                .finish()
                .is_err()
        );
        let empty: Bitmap<Gray8> = Bitmap::new(0, 4, 4, false);
        assert!(PdfPage::encode(RowImage::Gray(&empty), (300.0, 300.0)).is_err());
        let bmp = gray(2, 2, |_, _| 0);
        assert!(PdfPage::encode(RowImage::Gray(&bmp), (0.0, 300.0)).is_err());
        assert!(PdfPage::encode(RowImage::Gray(&bmp), (f64::NAN, 300.0)).is_err());
        // Two pixels at 2 000 000 dpi measure 0.000072 point.
        assert!(PdfPage::encode(RowImage::Gray(&bmp), (2.0e6, 300.0)).is_err());
    }
}
