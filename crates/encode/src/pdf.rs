//! Raster pages as a PDF: each page one 8-bit image, Flate-compressed,
//! spanning a media box of the image's size at its resolution.
//!
//! The file is written as it goes: a page's image, content stream and page
//! object when the page is added, the page tree, catalogue and
//! cross-reference table when it is finished.  Objects 1 and 2 are the
//! catalogue and the page tree, written last; a page takes the three
//! objects that follow the previous page's.

use std::io::{self, Write};

use color::{Gray8, Rgb8};
use flate2::Compression;
use flate2::write::ZlibEncoder;
use raster::Bitmap;

use crate::EncodeError;

/// The image of a PDF page.
#[derive(Clone, Copy)]
pub enum PdfImage<'a> {
    /// 8-bit RGB, written `DeviceRGB`.
    Rgb(&'a Bitmap<Rgb8>),
    /// 8-bit gray, 0 black, written `DeviceGray`.
    Gray(&'a Bitmap<Gray8>),
}

/// A page's image, compressed, and its size.
#[derive(Clone, Debug)]
pub struct PdfPage {
    width: u32,
    height: u32,
    rgb: bool,
    /// The page's width and height in points.
    size_pts: (f64, f64),
    data: Vec<u8>,
}

impl PdfPage {
    /// Compress `image`, a page at `dpi` (horizontal, vertical) pixels per
    /// inch.
    ///
    /// # Errors
    ///
    /// [`EncodeError::InvalidInput`] for an empty image or a resolution
    /// that is not positive and finite.
    pub fn encode(image: PdfImage<'_>, dpi: (f64, f64)) -> Result<Self, EncodeError> {
        let (width, height, rgb) = match image {
            PdfImage::Rgb(b) => (b.width, b.height, true),
            PdfImage::Gray(b) => (b.width, b.height, false),
        };
        if width == 0 || height == 0 {
            return Err(EncodeError::InvalidInput(
                "zero-width or zero-height bitmap: a PDF page cannot hold an empty image",
            ));
        }
        if !(dpi.0.is_finite() && dpi.1.is_finite() && dpi.0 > 0.0 && dpi.1 > 0.0) {
            return Err(EncodeError::InvalidInput(
                "a PDF page needs a positive resolution",
            ));
        }
        let mut z = ZlibEncoder::new(Vec::new(), Compression::fast());
        match image {
            PdfImage::Rgb(b) => crate::write_live_rows(b, &mut z)?,
            PdfImage::Gray(b) => crate::write_live_rows(b, &mut z)?,
        }
        Ok(Self {
            width,
            height,
            rgb,
            size_pts: (
                f64::from(width) * 72.0 / dpi.0,
                f64::from(height) * 72.0 / dpi.1,
            ),
            data: z.finish()?,
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
}

impl<W: Write> PdfWriter<W> {
    /// Begin a PDF on `out`: write its header.
    ///
    /// # Errors
    ///
    /// [`EncodeError::Io`] on a write failure.
    pub fn new(out: W) -> Result<Self, EncodeError> {
        let mut out = Counting { out, written: 0 };
        out.write_all(b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n")?;
        Ok(Self {
            out,
            // The catalogue and the page tree, placed when written.
            offsets: vec![0, 0],
            pages: Vec::new(),
        })
    }

    /// Append `page`.
    ///
    /// # Errors
    ///
    /// [`EncodeError::Io`] on a write failure.
    pub fn write_page(&mut self, page: &PdfPage) -> Result<(), EncodeError> {
        let image = self.offsets.len() + 1;
        let (w, h) = (fmt_num(page.size_pts.0), fmt_num(page.size_pts.1));

        self.begin(image)?;
        write!(
            self.out,
            "<</Type/XObject/Subtype/Image/Width {}/Height {}/ColorSpace/{}\
             /BitsPerComponent 8/Filter/FlateDecode/Length {}>>\nstream\n",
            page.width,
            page.height,
            if page.rgb { "DeviceRGB" } else { "DeviceGray" },
            page.data.len()
        )?;
        self.out.write_all(&page.data)?;
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
        self.offsets[1] = self.out.written;
        let kids: Vec<String> = self.pages.iter().map(|p| format!("{p} 0 R")).collect();
        write!(
            self.out,
            "2 0 obj\n<</Type/Pages/Kids[{}]/Count {}>>\nendobj\n",
            kids.join(" "),
            self.pages.len()
        )?;
        self.offsets[0] = self.out.written;
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
    fn begin(&mut self, id: usize) -> io::Result<()> {
        debug_assert_eq!(id, self.offsets.len() + 1);
        self.offsets.push(self.out.written);
        writeln!(self.out, "{id} 0 obj")
    }
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

    use color::{Gray8, Rgb8};
    use flate2::read::ZlibDecoder;
    use raster::Bitmap;

    use super::{PdfImage, PdfPage, PdfWriter, fmt_num};

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
        let page = PdfPage::encode(PdfImage::Gray(&bmp), (100.0, 100.0)).unwrap();
        let mut writer = PdfWriter::new(Vec::new()).unwrap();
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

    #[test]
    fn the_cross_reference_table_points_at_every_object() {
        let mut rgb: Bitmap<Rgb8> = Bitmap::new(3, 2, 4, false);
        rgb.row_bytes_mut(1)[..9].copy_from_slice(&[9; 9]);
        let page = PdfPage::encode(PdfImage::Rgb(&rgb), (72.0, 144.0)).unwrap();
        let mut writer = PdfWriter::new(Vec::new()).unwrap();
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
    fn empty_documents_images_and_resolutions_are_refused() {
        assert!(PdfWriter::new(Vec::new()).unwrap().finish().is_err());
        let empty: Bitmap<Gray8> = Bitmap::new(0, 4, 4, false);
        assert!(PdfPage::encode(PdfImage::Gray(&empty), (300.0, 300.0)).is_err());
        let bmp = gray(2, 2, |_, _| 0);
        assert!(PdfPage::encode(PdfImage::Gray(&bmp), (0.0, 300.0)).is_err());
        assert!(PdfPage::encode(PdfImage::Gray(&bmp), (f64::NAN, 300.0)).is_err());
    }
}
