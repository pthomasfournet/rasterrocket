//! PNG encoder (via the `png` crate).
//!
//! Grayscale bitmaps are written as 8-bit grayscale; every colour mode is
//! converted to 8-bit RGB as for PPM (see [`crate::ppm`]).  A bitmap's alpha
//! plane, when present, becomes the PNG alpha channel.  `Mono1` returns
//! [`EncodeError::UnsupportedMode`].
//!
//! PNG is lossless and carries transparency; prefer it to PPM when the bitmap
//! has an alpha plane.

use std::cell::RefCell;
use std::io::{self, Write};
use std::rc::Rc;

use color::{Pixel, PixelMode};
use raster::Bitmap;

use crate::EncodeError;
use crate::ppm::RgbSource;
use crate::rows::{RowCount, RowImage, RowLayout};

/// Write `bitmap` to `out` as a PNG image.
///
/// The sink `out` is consumed; wrap in `std::io::BufWriter` if buffering is
/// needed.
///
/// | Mode | PNG colour type |
/// |------|----------------|
/// | `Mono8` | `Grayscale` (8-bit) |
/// | `Rgb8`, `Bgr8`, `Xbgr8`, `Cmyk8`, `DeviceN8` | `RGB`, or `RGBA` with an alpha plane |
///
/// # Errors
///
/// Returns [`EncodeError::UnsupportedMode`] for `Mono1` bitmaps, or when the
/// pixel buffer size overflows `usize`.
/// Returns [`EncodeError::Io`] or [`EncodeError::PngEncoder`] on failure.
pub fn write_png<P: Pixel, W: Write>(bitmap: &Bitmap<P>, out: W) -> Result<(), EncodeError> {
    if P::MODE == PixelMode::Mono8 {
        return write_png_gray(bitmap, out);
    }
    let source = RgbSource::of(P::MODE).ok_or(EncodeError::UnsupportedMode(
        "1-bit bitmap: PNG output takes Mono8 or a colour mode",
    ))?;
    write_png_rgb(bitmap, source, out)
}

/// Build a configured [`png::Writer`] ready to accept image data.
fn png_encoder<W: Write>(
    out: W,
    width: u32,
    height: u32,
    color: ::png::ColorType,
    depth: ::png::BitDepth,
) -> Result<::png::Writer<W>, EncodeError> {
    let mut encoder = ::png::Encoder::new(out, width, height);
    encoder.set_color(color);
    encoder.set_depth(depth);
    // Deflate level 1 keeps encode time low across hundreds of pages.  Set
    // before the filter: `set_compression`/`set_deflate_compression` also
    // pick a filter, so choosing the filter afterwards is what makes it stick.
    encoder.set_deflate_compression(::png::DeflateCompression::Level(1));
    // Adaptive picks a filter per row rather than committing to one for the
    // whole image, which matters because page renders are not homogeneous:
    // measured over 21 rendered pages at 150 dpi, `Up` is 8.0% smaller than
    // Paeth on scanned content but only 1.0% smaller on text/vector, while
    // Adaptive is 0.9% and 3.1% smaller respectively.  Adaptive is the only
    // filter that never loses to Paeth on either content type, for ~2% more
    // encode time.
    encoder.set_filter(::png::Filter::Adaptive);
    Ok(encoder.write_header()?)
}

/// `width × height × channels` bytes, or an error when that overflows `usize`.
fn image_len<P: Pixel>(bitmap: &Bitmap<P>, channels: usize) -> Result<usize, EncodeError> {
    (bitmap.width as usize)
        .checked_mul(bitmap.height as usize)
        .and_then(|px| px.checked_mul(channels))
        .ok_or(EncodeError::UnsupportedMode(
            "image too large: pixel buffer would overflow usize",
        ))
}

/// Write a colour bitmap as PNG RGB, or RGBA when it has an alpha plane.
fn write_png_rgb<P: Pixel, W: Write>(
    bitmap: &Bitmap<P>,
    source: RgbSource,
    out: W,
) -> Result<(), EncodeError> {
    let w = bitmap.width as usize;
    let (channels, color_type) = if bitmap.has_alpha() {
        (4, ::png::ColorType::Rgba)
    } else {
        (3, ::png::ColorType::Rgb)
    };
    let mut buf = Vec::with_capacity(image_len(bitmap, channels)?);
    let mut rgb = vec![0u8; w * 3];
    for y in 0..bitmap.height {
        source.convert(bitmap.row_bytes(y), &mut rgb);
        match bitmap.alpha_row(y) {
            Some(alpha) => crate::push_rgba(&rgb, alpha, &mut buf),
            None => buf.extend_from_slice(&rgb),
        }
    }
    let mut writer = png_encoder(
        out,
        bitmap.width,
        bitmap.height,
        color_type,
        ::png::BitDepth::Eight,
    )?;
    writer.write_image_data(&buf)?;
    Ok(())
}

/// Write a `Gray8` bitmap as PNG.
fn write_png_gray<P: Pixel, W: Write>(bitmap: &Bitmap<P>, out: W) -> Result<(), EncodeError> {
    let mut writer = png_encoder(
        out,
        bitmap.width,
        bitmap.height,
        ::png::ColorType::Grayscale,
        ::png::BitDepth::Eight,
    )?;
    writer.write_image_data(&crate::live_bytes(bitmap))?;
    Ok(())
}

/// The colour type of a PNG written by [`PngWriter`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PngColor {
    /// 8-bit gray, from gray bands.
    Gray,
    /// 8-bit RGB, from RGB bands; an alpha plane is dropped.
    Rgb,
    /// 8-bit RGBA, from RGB bands, the alpha plane's (opaque without one).
    Rgba,
}

impl PngColor {
    const fn layout(self) -> RowLayout {
        match self {
            Self::Gray => RowLayout::Gray,
            Self::Rgb => RowLayout::Rgb,
            Self::Rgba => RowLayout::Rgba,
        }
    }

    const fn color_type(self) -> ::png::ColorType {
        match self {
            Self::Gray => ::png::ColorType::Grayscale,
            Self::Rgb => ::png::ColorType::Rgb,
            Self::Rgba => ::png::ColorType::Rgba,
        }
    }
}

/// A PNG written as its rows arrive, band by band.
///
/// Each row is filtered and compressed into the image data as it comes,
/// with the encoder settings of [`write_png`].  An image written whole and
/// the same image written in bands give the same bytes.
pub struct PngWriter<W: Write + 'static> {
    stream: ::png::StreamWriter<'static, Recording<W>>,
    sink: Rc<RefCell<Sink<W>>>,
    color: PngColor,
    rows: RowCount,
    row: Vec<u8>,
}

/// The sink of a [`PngWriter`], and the first error a write to it met.
struct Sink<W> {
    out: W,
    error: Option<io::Error>,
}

/// A handle on a [`Sink`] given to the encoder, which writes its last
/// chunk when it is dropped and discards the error of that write: the
/// sink keeps it, for [`PngWriter::finish`] to report.
struct Recording<W>(Rc<RefCell<Sink<W>>>);

impl<W: Write> Write for Recording<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut sink = self.0.borrow_mut();
        sink.out.write(buf).inspect_err(|e| {
            if sink.error.is_none() {
                sink.error = Some(io::Error::new(e.kind(), e.to_string()));
            }
        })
    }

    fn flush(&mut self) -> io::Result<()> {
        let mut sink = self.0.borrow_mut();
        sink.out.flush().inspect_err(|e| {
            if sink.error.is_none() {
                sink.error = Some(io::Error::new(e.kind(), e.to_string()));
            }
        })
    }
}

impl<W: Write + 'static> PngWriter<W> {
    /// Begin a `width × height` image of `color` on `out`: write the
    /// header chunks.
    ///
    /// # Errors
    ///
    /// [`EncodeError::InvalidInput`] for an empty image;
    /// [`EncodeError::Io`] or [`EncodeError::PngEncoder`] on failure.
    pub fn new(out: W, (width, height): (u32, u32), color: PngColor) -> Result<Self, EncodeError> {
        let rows = RowCount::new(width, height)?;
        let sink = Rc::new(RefCell::new(Sink { out, error: None }));
        let writer = png_encoder(
            Recording(Rc::clone(&sink)),
            width,
            height,
            color.color_type(),
            ::png::BitDepth::Eight,
        )?;
        Ok(Self {
            stream: writer.into_stream_writer()?,
            sink,
            color,
            rows,
            row: Vec::with_capacity(color.layout().row_bytes(width)),
        })
    }

    /// Write the rows of `band`, the next rows of the image.
    ///
    /// # Errors
    ///
    /// [`EncodeError::InvalidInput`] for a band of another width, of more
    /// rows than the image has left, or of colour for a gray image or the
    /// reverse; [`EncodeError::Io`] on a write failure.
    pub fn push(&mut self, band: RowImage<'_>) -> Result<(), EncodeError> {
        let layout = self.color.layout();
        self.rows.take(&band, layout)?;
        for y in 0..band.size().1 {
            self.row.clear();
            band.row(y, layout, &mut self.row);
            self.stream.write_all(&self.row)?;
        }
        Ok(())
    }

    /// Write the end of the image data and the trailing chunk, flush the
    /// sink and return it.
    ///
    /// # Errors
    ///
    /// [`EncodeError::InvalidInput`] when rows are missing;
    /// [`EncodeError::Io`] or [`EncodeError::PngEncoder`] on failure,
    /// the trailing chunk's included.
    pub fn finish(self) -> Result<W, EncodeError> {
        self.rows.finished()?;
        self.stream.finish()?;
        let sink = Rc::try_unwrap(self.sink)
            .map_err(|_| {
                EncodeError::InvalidInput("the PNG encoder still holds its sink after finishing")
            })?
            .into_inner();
        if let Some(error) = sink.error {
            return Err(error.into());
        }
        let mut out = sink.out;
        out.flush()?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use color::{Cmyk8, Gray8, Rgb8};
    use raster::Bitmap;

    fn make_rgb_bitmap(w: u32, h: u32, fill: [u8; 3]) -> Bitmap<Rgb8> {
        let mut bmp = Bitmap::new(w, h, 1, false);
        for y in 0..h {
            let row = bmp.row_bytes_mut(y);
            for chunk in row.as_chunks_mut::<3>().0 {
                *chunk = fill;
            }
        }
        bmp
    }

    fn make_gray_bitmap(w: u32, h: u32, fill: u8) -> Bitmap<Gray8> {
        let mut bmp = Bitmap::new(w, h, 1, false);
        for y in 0..h {
            bmp.row_bytes_mut(y).fill(fill);
        }
        bmp
    }

    /// Decode a PNG from bytes and return `(width, height, raw_pixels)`.
    fn decode_png(data: &[u8]) -> (u32, u32, Vec<u8>) {
        let decoder = ::png::Decoder::new(std::io::Cursor::new(data));
        let mut reader = decoder.read_info().expect("png decode header");
        let mut buf = vec![0u8; reader.output_buffer_size().expect("output buffer size")];
        let frame = reader.next_frame(&mut buf).expect("png decode frame");
        let info = reader.info();
        (info.width, info.height, buf[..frame.buffer_size()].to_vec())
    }

    #[test]
    fn rgb_png_roundtrip() {
        let bmp = make_rgb_bitmap(4, 2, [100, 150, 200]);
        let mut out = Vec::new();
        write_png::<Rgb8, _>(&bmp, &mut out).unwrap();

        let (w, h, pixels) = decode_png(&out);
        assert_eq!((w, h), (4, 2));
        assert_eq!(pixels.len(), 24, "4×2 pixels × 3 bytes");
        for chunk in pixels.as_chunks::<3>().0 {
            assert_eq!(chunk, &[100, 150, 200], "pixel mismatch");
        }
    }

    #[test]
    fn gray_png_roundtrip() {
        let bmp = make_gray_bitmap(3, 3, 77);
        let mut out = Vec::new();
        write_png::<Gray8, _>(&bmp, &mut out).unwrap();

        let (w, h, pixels) = decode_png(&out);
        assert_eq!((w, h), (3, 3));
        assert!(pixels.iter().all(|&v| v == 77), "grayscale pixel mismatch");
    }

    #[test]
    fn rgb_with_alpha_writes_rgba_png() {
        // Bitmap with alpha plane: every pixel [255,0,0] with alpha 128.
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(2, 1, 1, true);
        let row = bmp.row_bytes_mut(0);
        row[..6].copy_from_slice(&[255, 0, 0, 255, 0, 0]);
        if let Some(a) = bmp.alpha_plane_mut() {
            a.fill(128);
        }

        let mut out = Vec::new();
        write_png::<Rgb8, _>(&bmp, &mut out).unwrap();

        let (w, h, pixels) = decode_png(&out);
        assert_eq!((w, h), (2, 1));
        assert_eq!(pixels.len(), 8, "2 pixels × 4 bytes (RGBA)");
        assert_eq!(&pixels[..4], &[255, 0, 0, 128], "pixel 0 RGBA");
        assert_eq!(&pixels[4..8], &[255, 0, 0, 128], "pixel 1 RGBA");
    }

    #[test]
    fn stride_padding_not_included() {
        // width=3 with pad=4 → stride=4 for Gray8.
        let bmp: Bitmap<Gray8> = Bitmap::new(3, 1, 4, false);
        let mut out = Vec::new();
        write_png::<Gray8, _>(&bmp, &mut out).unwrap();
        let (w, h, pixels) = decode_png(&out);
        assert_eq!((w, h), (3, 1));
        assert_eq!(
            pixels.len(),
            3,
            "stride padding must not appear in PNG output"
        );
    }

    #[test]
    fn rgb_with_alpha_multi_row_promotion() {
        // The 2×1 rgb_with_alpha_writes_rgba_png test only exercises a single
        // row; this 2×2 fixture verifies the per-row offset math is correct
        // for at least two rows of RGB+alpha promotion.
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(2, 2, 1, true);
        // Row 0 RGB: [10, 20, 30, 40, 50, 60]; Row 1 RGB: [70, 80, 90, 100, 110, 120].
        bmp.row_bytes_mut(0)
            .copy_from_slice(&[10, 20, 30, 40, 50, 60]);
        bmp.row_bytes_mut(1)
            .copy_from_slice(&[70, 80, 90, 100, 110, 120]);
        let alpha = bmp.alpha_plane_mut().expect("alpha plane present");
        alpha.copy_from_slice(&[200, 210, 220, 230]);

        let mut out = Vec::new();
        write_png::<Rgb8, _>(&bmp, &mut out).unwrap();
        let (w, h, pixels) = decode_png(&out);
        assert_eq!((w, h), (2, 2));
        assert_eq!(pixels.len(), 16);
        assert_eq!(&pixels[0..4], &[10, 20, 30, 200]);
        assert_eq!(&pixels[4..8], &[40, 50, 60, 210]);
        assert_eq!(&pixels[8..12], &[70, 80, 90, 220]);
        assert_eq!(&pixels[12..16], &[100, 110, 120, 230]);
    }

    /// CMYK converts to RGB through the process-ink model, as for PPM.
    #[test]
    fn cmyk_writes_converted_rgb() {
        let mut bmp: Bitmap<Cmyk8> = Bitmap::new(2, 1, 1, false);
        bmp.row_bytes_mut(0)
            .copy_from_slice(&[0, 0, 0, 0, 10, 20, 30, 40]);
        let mut out = Vec::new();
        write_png::<Cmyk8, _>(&bmp, &mut out).unwrap();
        let (w, h, pixels) = decode_png(&out);
        assert_eq!((w, h), (2, 1));
        let mut want = vec![255, 255, 255];
        want.extend_from_slice(&color::cmyk::cmyk_to_rgb(10, 20, 30, 40));
        assert_eq!(pixels, want);
    }
}
