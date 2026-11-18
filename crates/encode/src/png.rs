//! PNG encoder (via the `png` crate).
//!
//! Grayscale bitmaps are written as 8-bit grayscale, bilevel ones
//! ([`write_png_bilevel`], [`PngColor::Bilevel`]) as grayscale of bit
//! depth 1; every colour mode is converted to 8-bit RGB as for PPM (see
//! [`crate::ppm`]).  A bitmap's alpha plane, when present, becomes the PNG
//! alpha channel.  `Mono1` returns [`EncodeError::UnsupportedMode`].  A
//! resolution, when given, is written as a `pHYs` chunk.
//!
//! PNG is lossless and carries transparency; prefer it to PPM when the bitmap
//! has an alpha plane.

use std::borrow::Cow;
use std::cell::RefCell;
use std::io::{self, Write};
use std::rc::Rc;

use color::{Pixel, PixelMode};
use raster::Bitmap;

use color::Gray8;

use crate::icc::{self, IccProfile};
use crate::ppm::RgbSource;
use crate::rows::{ByteOrder, RowCount, RowImage, RowLayout};
use crate::{EncodeError, pack_bilevel_row};

/// Write `bitmap` to `out` as a PNG image, at `dpi` (horizontal, vertical)
/// pixels per inch (a `pHYs` chunk) and tagged with `icc` (an `iCCP`
/// chunk) when given.
///
/// The sink `out` is consumed; wrap in `std::io::BufWriter` if buffering is
/// needed.
///
/// | Mode | PNG colour type |
/// |------|----------------|
/// | `Mono8` | `Grayscale` (8-bit), or `GrayscaleAlpha` with an alpha plane |
/// | `Rgb8`, `Bgr8`, `Xbgr8`, `Cmyk8`, `DeviceN8` | `RGB`, or `RGBA` with an alpha plane |
///
/// # Errors
///
/// Returns [`EncodeError::UnsupportedMode`] for `Mono1` bitmaps, or when the
/// pixel buffer size overflows `usize`; [`EncodeError::InvalidInput`] for a
/// profile of another colour space than the image's, or a resolution
/// outside what `pHYs` holds (see [`pixels_per_metre`]).
/// Returns [`EncodeError::Io`] or [`EncodeError::PngEncoder`] on failure.
pub fn write_png<P: Pixel<Sample = u8>, W: Write>(
    bitmap: &Bitmap<P>,
    dpi: Option<(f64, f64)>,
    out: W,
    icc: Option<&IccProfile>,
) -> Result<(), EncodeError> {
    let header = Header { dpi, icc };
    if P::MODE == PixelMode::Mono8 {
        icc::check(icc, RowLayout::Gray)?;
        return write_png_gray(bitmap, out, header);
    }
    let source = RgbSource::of(P::MODE).ok_or(EncodeError::UnsupportedMode(
        "1-bit bitmap: PNG output takes Mono8 or a colour mode",
    ))?;
    icc::check(icc, RowLayout::Rgb)?;
    write_png_rgb(bitmap, source, out, header)
}

/// Write the luma of `luma` to `out` as a PNG image of bit depth 1, black
/// below [`crate::BLACK_BELOW`], white otherwise, at `dpi` and tagged with
/// `icc` when given, as [`write_png`].
///
/// # Errors
///
/// As [`write_png`], and [`EncodeError::InvalidInput`] for an empty image.
pub fn write_png_bilevel<W: Write>(
    luma: &Bitmap<Gray8>,
    dpi: Option<(f64, f64)>,
    out: W,
    icc: Option<&IccProfile>,
) -> Result<(), EncodeError> {
    if luma.width == 0 || luma.height == 0 {
        return Err(EncodeError::InvalidInput(
            "zero-width or zero-height image: there is nothing to encode",
        ));
    }
    icc::check(icc, RowLayout::Bilevel)?;
    let row_bytes = RowLayout::Bilevel.row_bytes(luma.width);
    let mut buf = vec![
        0u8;
        row_bytes.checked_mul(luma.height as usize).ok_or(
            EncodeError::UnsupportedMode("image too large: pixel buffer would overflow usize")
        )?
    ];
    for (y, row) in (0..luma.height).zip(buf.chunks_exact_mut(row_bytes)) {
        pack_bilevel_row(&luma.row_bytes(y)[..luma.width as usize], row);
        black_is_zero(row, luma.width);
    }
    let mut writer = png_encoder(
        out,
        (luma.width, luma.height),
        (::png::ColorType::Grayscale, ::png::BitDepth::One),
        Header { dpi, icc },
    )?;
    writer.write_image_data(&buf)?;
    Ok(())
}

/// Turn a packed bilevel row of `width` pixels, 1 for black, into PNG's
/// sense, 0 for black, the padding bits of its last byte 0.
fn black_is_zero(row: &mut [u8], width: u32) {
    for byte in &mut *row {
        *byte = !*byte;
    }
    let tail = width % 8;
    if let Some(last) = row.last_mut()
        && tail != 0
    {
        *last &= 0xFF << (8 - tail);
    }
}

/// The chunks of a PNG's header besides `IHDR`: its resolution (`pHYs`)
/// and its profile (`iCCP`), each when given.
#[derive(Clone, Copy)]
struct Header<'a> {
    dpi: Option<(f64, f64)>,
    icc: Option<&'a IccProfile>,
}

/// `dpi` pixels per inch as the whole pixels per metre `pHYs` stores:
/// `round(dpi / 0.0254)`, within `1..=2^31 − 1`, that is from about
/// 0.0127 to 54.5 million pixels per inch.
///
/// # Errors
///
/// [`EncodeError::InvalidInput`] for a resolution outside that range.
fn pixels_per_metre(dpi: f64) -> Result<u32, EncodeError> {
    crate::whole_units(dpi / 0.0254, i32::MAX.unsigned_abs()).ok_or(EncodeError::InvalidInput(
        "PNG resolution must lie between 0.0127 and 54 500 000 pixels per inch",
    ))
}

/// Build a configured [`png::Writer`] ready to accept image data, its
/// header chunks written, those of `header` among them.
fn png_encoder<W: Write>(
    out: W,
    (width, height): (u32, u32),
    (color, depth): (::png::ColorType, ::png::BitDepth),
    header: Header<'_>,
) -> Result<::png::Writer<W>, EncodeError> {
    let mut info = ::png::Info::with_size(width, height);
    info.icc_profile = header.icc.map(|p| Cow::Owned(p.bytes().to_vec()));
    info.pixel_dims = header
        .dpi
        .map(|(x, y)| -> Result<_, EncodeError> {
            Ok(::png::PixelDimensions {
                xppu: pixels_per_metre(x)?,
                yppu: pixels_per_metre(y)?,
                unit: ::png::Unit::Meter,
            })
        })
        .transpose()?;
    let mut encoder = ::png::Encoder::with_info(out, info)?;
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
fn image_len<P: Pixel<Sample = u8>>(
    bitmap: &Bitmap<P>,
    channels: usize,
) -> Result<usize, EncodeError> {
    (bitmap.width as usize)
        .checked_mul(bitmap.height as usize)
        .and_then(|px| px.checked_mul(channels))
        .ok_or(EncodeError::UnsupportedMode(
            "image too large: pixel buffer would overflow usize",
        ))
}

/// Write a colour bitmap as PNG RGB, or RGBA when it has an alpha plane.
fn write_png_rgb<P: Pixel<Sample = u8>, W: Write>(
    bitmap: &Bitmap<P>,
    source: RgbSource,
    out: W,
    header: Header<'_>,
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
        (bitmap.width, bitmap.height),
        (color_type, ::png::BitDepth::Eight),
        header,
    )?;
    writer.write_image_data(&buf)?;
    Ok(())
}

/// Write a `Gray8` bitmap as PNG gray, or gray and alpha when it has an
/// alpha plane.
fn write_png_gray<P: Pixel<Sample = u8>, W: Write>(
    bitmap: &Bitmap<P>,
    out: W,
    header: Header<'_>,
) -> Result<(), EncodeError> {
    let color_type = if bitmap.has_alpha() {
        ::png::ColorType::GrayscaleAlpha
    } else {
        ::png::ColorType::Grayscale
    };
    let mut writer = png_encoder(
        out,
        (bitmap.width, bitmap.height),
        (color_type, ::png::BitDepth::Eight),
        header,
    )?;
    if !bitmap.has_alpha() {
        writer.write_image_data(&crate::live_bytes(bitmap))?;
        return Ok(());
    }
    let w = bitmap.width as usize;
    let mut buf = Vec::with_capacity(image_len(bitmap, 2)?);
    for y in 0..bitmap.height {
        let gray = &bitmap.row_bytes(y)[..w];
        let alpha = bitmap.alpha_row(y).unwrap_or_default();
        for (&v, &a) in gray.iter().zip(alpha) {
            buf.extend_from_slice(&[v, a]);
        }
    }
    writer.write_image_data(&buf)?;
    Ok(())
}

/// The colour type and bit depth of a PNG written by [`PngWriter`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PngColor {
    /// Gray of bit depth 1, from gray bands: black below
    /// [`crate::BLACK_BELOW`], white otherwise.
    Bilevel,
    /// 8-bit gray, from gray bands; an alpha plane is dropped.
    Gray,
    /// 8-bit gray and alpha, from gray bands, the alpha plane's (opaque
    /// without one).
    GrayAlpha,
    /// 8-bit RGB, from RGB bands; an alpha plane is dropped.
    Rgb,
    /// 8-bit RGBA, from RGB bands, the alpha plane's (opaque without one).
    Rgba,
    /// 16-bit gray, from 16-bit gray bands; an alpha plane is dropped.
    Gray16,
    /// 16-bit gray and alpha, from 16-bit gray bands, the alpha plane's
    /// (opaque without one).
    GrayAlpha16,
    /// 16-bit RGB, from 16-bit RGB bands; an alpha plane is dropped.
    Rgb16,
    /// 16-bit RGBA, from 16-bit RGB bands, the alpha plane's (opaque
    /// without one).
    Rgba16,
}

impl PngColor {
    const fn layout(self) -> RowLayout {
        match self {
            Self::Bilevel => RowLayout::Bilevel,
            Self::Gray => RowLayout::Gray,
            Self::GrayAlpha => RowLayout::GrayAlpha,
            Self::Rgb => RowLayout::Rgb,
            Self::Rgba => RowLayout::Rgba,
            Self::Gray16 => RowLayout::Gray16,
            Self::GrayAlpha16 => RowLayout::GrayAlpha16,
            Self::Rgb16 => RowLayout::Rgb16,
            Self::Rgba16 => RowLayout::Rgba16,
        }
    }

    const fn color_type(self) -> ::png::ColorType {
        match self {
            Self::Bilevel | Self::Gray | Self::Gray16 => ::png::ColorType::Grayscale,
            Self::GrayAlpha | Self::GrayAlpha16 => ::png::ColorType::GrayscaleAlpha,
            Self::Rgb | Self::Rgb16 => ::png::ColorType::Rgb,
            Self::Rgba | Self::Rgba16 => ::png::ColorType::Rgba,
        }
    }

    const fn bit_depth(self) -> ::png::BitDepth {
        match self {
            Self::Bilevel => ::png::BitDepth::One,
            Self::Gray | Self::GrayAlpha | Self::Rgb | Self::Rgba => ::png::BitDepth::Eight,
            Self::Gray16 | Self::GrayAlpha16 | Self::Rgb16 | Self::Rgba16 => {
                ::png::BitDepth::Sixteen
            }
        }
    }
}

/// A PNG written as its rows arrive, band by band.
///
/// Each row is filtered and compressed into the image data as it comes,
/// with the encoder settings of [`write_png`].  An image written whole and
/// the same image written in bands give the same bytes, and a
/// [`PngColor::Bilevel`] image those of [`write_png_bilevel`].
pub struct PngWriter<W: Write + 'static> {
    stream: ::png::StreamWriter<'static, Recording<W>>,
    sink: Rc<RefCell<Sink<W>>>,
    color: PngColor,
    width: u32,
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
    /// Begin a `width × height` image of `color` on `out`, at `dpi` and
    /// tagged with `icc` when given, as [`write_png`]: write the header
    /// chunks.
    ///
    /// # Errors
    ///
    /// [`EncodeError::InvalidInput`] for an empty image, a profile of
    /// another colour space than `color`'s or a resolution `pHYs` does not
    /// hold; [`EncodeError::Io`] or [`EncodeError::PngEncoder`] on failure.
    pub fn new(
        out: W,
        (width, height): (u32, u32),
        color: PngColor,
        dpi: Option<(f64, f64)>,
        icc: Option<&IccProfile>,
    ) -> Result<Self, EncodeError> {
        let rows = RowCount::new(width, height)?;
        icc::check(icc, color.layout())?;
        let sink = Rc::new(RefCell::new(Sink { out, error: None }));
        let writer = png_encoder(
            Recording(Rc::clone(&sink)),
            (width, height),
            (color.color_type(), color.bit_depth()),
            Header { dpi, icc },
        )?;
        Ok(Self {
            stream: writer.into_stream_writer()?,
            sink,
            color,
            width,
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
            band.row(y, layout, ByteOrder::Big, &mut self.row);
            if self.color == PngColor::Bilevel {
                black_is_zero(&mut self.row, self.width);
            }
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
        write_png::<Rgb8, _>(&bmp, None, &mut out, None).unwrap();

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
        write_png::<Gray8, _>(&bmp, None, &mut out, None).unwrap();

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
        write_png::<Rgb8, _>(&bmp, None, &mut out, None).unwrap();

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
        write_png::<Gray8, _>(&bmp, None, &mut out, None).unwrap();
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
        write_png::<Rgb8, _>(&bmp, None, &mut out, None).unwrap();
        let (w, h, pixels) = decode_png(&out);
        assert_eq!((w, h), (2, 2));
        assert_eq!(pixels.len(), 16);
        assert_eq!(&pixels[0..4], &[10, 20, 30, 200]);
        assert_eq!(&pixels[4..8], &[40, 50, 60, 210]);
        assert_eq!(&pixels[8..12], &[70, 80, 90, 220]);
        assert_eq!(&pixels[12..16], &[100, 110, 120, 230]);
    }

    /// The colour type of a PNG.
    fn color_type(data: &[u8]) -> ::png::ColorType {
        let decoder = ::png::Decoder::new(std::io::Cursor::new(data));
        decoder
            .read_info()
            .expect("png decode header")
            .info()
            .color_type
    }

    #[test]
    fn gray_with_alpha_writes_gray_alpha_png() {
        let mut bmp: Bitmap<Gray8> = Bitmap::new(3, 2, 4, true);
        bmp.row_bytes_mut(0)[..3].copy_from_slice(&[1, 2, 3]);
        bmp.row_bytes_mut(1)[..3].copy_from_slice(&[4, 5, 6]);
        bmp.alpha_plane_mut()
            .unwrap()
            .copy_from_slice(&[0, 9, 255, 7, 8, 128]);
        let want = [1, 0, 2, 9, 3, 255, 4, 7, 5, 8, 6, 128];
        let mut whole = Vec::new();
        write_png::<Gray8, _>(&bmp, None, &mut whole, None).unwrap();
        assert_eq!(color_type(&whole), ::png::ColorType::GrayscaleAlpha);
        assert_eq!(decode_png(&whole), (3, 2, want.to_vec()));
        let mut streamed =
            PngWriter::new(Vec::new(), (3, 2), PngColor::GrayAlpha, None, None).unwrap();
        streamed.push(RowImage::Gray(&bmp)).unwrap();
        assert_eq!(
            decode_png(&streamed.finish().unwrap()),
            (3, 2, want.to_vec())
        );
    }

    #[test]
    fn sixteen_bit_gray_alpha_stores_big_endian_pairs() {
        use color::Gray16;
        let mut bmp: Bitmap<Gray16> = Bitmap::new(2, 1, 1, true);
        bmp.row_samples_mut(0).copy_from_slice(&[0x0102, 0xFFFF]);
        bmp.alpha_plane_mut().unwrap().copy_from_slice(&[0x0304, 0]);
        let mut png =
            PngWriter::new(Vec::new(), (2, 1), PngColor::GrayAlpha16, None, None).unwrap();
        png.push(RowImage::Gray16(&bmp)).unwrap();
        let out = png.finish().unwrap();
        assert_eq!(color_type(&out), ::png::ColorType::GrayscaleAlpha);
        assert_eq!(decode_png(&out).2, [1, 2, 3, 4, 0xFF, 0xFF, 0, 0]);
    }

    #[test]
    fn a_gray_band_without_alpha_is_opaque_as_gray_alpha() {
        let bmp = make_gray_bitmap(2, 1, 40);
        let mut png = PngWriter::new(Vec::new(), (2, 1), PngColor::GrayAlpha, None, None).unwrap();
        png.push(RowImage::Gray(&bmp)).unwrap();
        assert_eq!(decode_png(&png.finish().unwrap()).2, [40, 255, 40, 255]);
        let rgb = make_rgb_bitmap(2, 1, [0, 0, 0]);
        let mut png = PngWriter::new(Vec::new(), (2, 1), PngColor::GrayAlpha, None, None).unwrap();
        assert!(matches!(
            png.push(RowImage::Rgb(&rgb)),
            Err(EncodeError::InvalidInput(_))
        ));
    }

    /// CMYK converts to RGB through the process-ink model, as for PPM.
    #[test]
    fn cmyk_writes_converted_rgb() {
        let mut bmp: Bitmap<Cmyk8> = Bitmap::new(2, 1, 1, false);
        bmp.row_bytes_mut(0)
            .copy_from_slice(&[0, 0, 0, 0, 10, 20, 30, 40]);
        let mut out = Vec::new();
        write_png::<Cmyk8, _>(&bmp, None, &mut out, None).unwrap();
        let (w, h, pixels) = decode_png(&out);
        assert_eq!((w, h), (2, 1));
        let mut want = vec![255, 255, 255];
        want.extend_from_slice(&color::cmyk::cmyk_to_rgb(10, 20, 30, 40));
        assert_eq!(pixels, want);
    }

    /// The `pHYs` chunk of `png`, read back.
    fn pixel_dims(png: &[u8]) -> Option<::png::PixelDimensions> {
        let decoder = ::png::Decoder::new(std::io::Cursor::new(png));
        decoder.read_info().expect("png header").info().pixel_dims
    }

    /// A resolution is written as whole pixels per metre, by the whole
    /// and the banded writers alike; none is written without one, and one
    /// `pHYs` cannot hold is refused.
    #[test]
    fn the_resolution_is_written_as_phys() {
        let bmp = make_gray_bitmap(3, 2, 9);
        let mut whole = Vec::new();
        write_png(&bmp, Some((300.0, 150.0)), &mut whole, None).unwrap();
        let dims = pixel_dims(&whole).expect("a pHYs chunk");
        assert_eq!((dims.xppu, dims.yppu), (11_811, 5_906));
        assert_eq!(dims.unit, ::png::Unit::Meter);
        let mut banded = PngWriter::new(
            Vec::new(),
            (3, 2),
            PngColor::Gray,
            Some((300.0, 150.0)),
            None,
        )
        .unwrap();
        banded.push(RowImage::Gray(&bmp)).unwrap();
        let banded = banded.finish().unwrap();
        assert_eq!(decode_png(&banded), decode_png(&whole));
        let dims = pixel_dims(&banded).expect("a pHYs chunk");
        assert_eq!((dims.xppu, dims.yppu), (11_811, 5_906));

        let mut bare = Vec::new();
        write_png(&bmp, None, &mut bare, None).unwrap();
        assert!(pixel_dims(&bare).is_none());
        for dpi in [0.0, -72.0, f64::NAN, f64::INFINITY, 6.0e7] {
            assert!(
                matches!(
                    write_png(&bmp, Some((dpi, 72.0)), Vec::new(), None),
                    Err(EncodeError::InvalidInput(_))
                ),
                "{dpi}"
            );
        }
    }

    /// A bilevel image is gray of bit depth 1, 0 for black, the padding
    /// bits of each row 0; the banded writer writes the same image.
    #[test]
    fn a_bilevel_image_is_written_at_one_bit() {
        let mut bmp: Bitmap<Gray8> = Bitmap::new(10, 2, 1, false);
        bmp.row_bytes_mut(0)[..10].copy_from_slice(&[0, 255, 0, 255, 127, 128, 0, 0, 255, 0]);
        bmp.row_bytes_mut(1)[..10].fill(255);
        let mut whole = Vec::new();
        write_png_bilevel(&bmp, Some((200.0, 200.0)), &mut whole, None).unwrap();

        let decoder = ::png::Decoder::new(std::io::Cursor::new(&whole));
        let mut reader = decoder.read_info().expect("png header");
        let info = reader.info();
        assert_eq!(info.color_type, ::png::ColorType::Grayscale);
        assert_eq!(info.bit_depth, ::png::BitDepth::One);
        let mut buf = vec![0u8; reader.output_buffer_size().expect("size")];
        let frame = reader.next_frame(&mut buf).expect("frame");
        assert_eq!(
            &buf[..frame.buffer_size()],
            &[0b0101_0100, 0b1000_0000, 0b1111_1111, 0b1100_0000]
        );

        let mut banded = PngWriter::new(
            Vec::new(),
            (10, 2),
            PngColor::Bilevel,
            Some((200.0, 200.0)),
            None,
        )
        .unwrap();
        for y in 0..2 {
            let mut row: Bitmap<Gray8> = Bitmap::new(10, 1, 1, false);
            row.row_bytes_mut(0)[..10].copy_from_slice(&bmp.row_bytes(y)[..10]);
            banded.push(RowImage::Gray(&row)).unwrap();
        }
        let banded = banded.finish().unwrap();
        assert_eq!(decode_png(&banded), decode_png(&whole));
        assert!(pixel_dims(&banded).is_some_and(|d| d.xppu == 7_874));
        assert!(write_png_bilevel(&Bitmap::new(0, 2, 1, false), None, Vec::new(), None).is_err());
    }
}
