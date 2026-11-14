//! The row-streamed writers: an image written in bands of any height is the
//! image written as one band, byte for byte, and the whole-image functions
//! write the same bytes where they share a format.

use std::io::Cursor;

use color::{Gray8, Rgb8};
use raster::Bitmap;
use rasterrocket_encode::{
    PdfPage, PdfPageEncoder, PdfWriter, PngColor, PngWriter, PnmKind, PnmWriter, RowImage,
    RowLayout, TiffCompression, TiffStream, write_pbm, write_pgm, write_ppm, write_tiff,
};

const W: u32 = 37;
const H: u32 = 101;

/// Band heights covering one row, uneven splits, a band past the image's
/// end and the whole image.
const BANDS: [u32; 5] = [1, 7, 64, 100, H];

fn rgb(alpha: bool) -> Bitmap<Rgb8> {
    let mut bmp = Bitmap::new(W, H, 4, alpha);
    for y in 0..H {
        for (i, px) in bmp.row_bytes_mut(y)[..W as usize * 3]
            .iter_mut()
            .enumerate()
        {
            *px = (i * 7 + y as usize * 3).to_le_bytes()[0];
        }
    }
    if let Some(a) = bmp.alpha_plane_mut() {
        for (i, v) in a.iter_mut().enumerate() {
            *v = (i * 13).to_le_bytes()[0];
        }
    }
    bmp
}

fn gray() -> Bitmap<Gray8> {
    let mut bmp = Bitmap::new(W, H, 4, false);
    for y in 0..H {
        for (x, px) in bmp.row_bytes_mut(y)[..W as usize].iter_mut().enumerate() {
            *px = if x < 15 {
                255
            } else {
                (x * 37 + y as usize * 11).to_le_bytes()[0]
            };
        }
    }
    bmp
}

/// Rows `y0..y0 + n` of `bmp` as a bitmap of their own, its alpha plane's
/// rows with them.
fn band<P: color::Pixel>(bmp: &Bitmap<P>, y0: u32, n: u32) -> Bitmap<P> {
    let mut out = Bitmap::new(bmp.width, n, 1, bmp.has_alpha());
    let live = bmp.width as usize * P::BYTES;
    for y in 0..n {
        out.row_bytes_mut(y)[..live].copy_from_slice(&bmp.row_bytes(y0 + y)[..live]);
    }
    if let (Some(src), Some(dst)) = (bmp.alpha_plane(), out.alpha_plane_mut()) {
        let w = bmp.width as usize;
        dst.copy_from_slice(&src[y0 as usize * w..(y0 + n) as usize * w]);
    }
    out
}

/// The bands of `bmp`, `rows` high (the last shorter).
fn bands<P: color::Pixel>(bmp: &Bitmap<P>, rows: u32) -> Vec<Bitmap<P>> {
    (0..bmp.height)
        .step_by(rows as usize)
        .map(|y0| band(bmp, y0, rows.min(bmp.height - y0)))
        .collect()
}

/// For each band height, `write` given the bands as `RowImage`s made by
/// `wrap`; every result equals `want`.
fn same_for_every_band_height<P: color::Pixel>(
    bmp: &Bitmap<P>,
    want: &[u8],
    what: &str,
    write: impl Fn(&[Bitmap<P>]) -> Vec<u8>,
) {
    for rows in BANDS {
        assert_eq!(
            write(&bands(bmp, rows)),
            want,
            "{what}, bands of {rows} rows"
        );
    }
}

#[test]
fn netpbm_bands_are_the_whole_image() {
    let (c, g) = (rgb(false), gray());
    let pnm = |kind, images: &[RowImage<'_>]| {
        let mut w = PnmWriter::new(Vec::new(), (W, H), kind).unwrap();
        for &image in images {
            w.push(image).unwrap();
        }
        w.finish().unwrap()
    };
    let mut want = Vec::new();
    write_ppm(&c, &mut want).unwrap();
    same_for_every_band_height(&c, &want, "PPM", |b| {
        pnm(
            PnmKind::Ppm,
            &b.iter().map(RowImage::Rgb).collect::<Vec<_>>(),
        )
    });
    let mut want = Vec::new();
    write_pgm(&g, &mut want).unwrap();
    same_for_every_band_height(&g, &want, "PGM", |b| {
        pnm(
            PnmKind::Pgm,
            &b.iter().map(RowImage::Gray).collect::<Vec<_>>(),
        )
    });
    let mut want = Vec::new();
    write_pbm(&g, &mut want).unwrap();
    same_for_every_band_height(&g, &want, "PBM", |b| {
        pnm(
            PnmKind::Pbm,
            &b.iter().map(RowImage::Gray).collect::<Vec<_>>(),
        )
    });
}

#[test]
fn png_bands_are_the_whole_image() {
    let png = |color, images: &[RowImage<'_>]| {
        let mut w = PngWriter::new(Vec::new(), (W, H), color).unwrap();
        for &image in images {
            w.push(image).unwrap();
        }
        w.finish().unwrap()
    };
    let g = gray();
    let whole = png(PngColor::Gray, &[RowImage::Gray(&g)]);
    same_for_every_band_height(&g, &whole, "gray PNG", |b| {
        png(
            PngColor::Gray,
            &b.iter().map(RowImage::Gray).collect::<Vec<_>>(),
        )
    });
    for (alpha, color) in [(false, PngColor::Rgb), (true, PngColor::Rgba)] {
        let c = rgb(alpha);
        let whole = png(color, &[RowImage::Rgb(&c)]);
        same_for_every_band_height(&c, &whole, &format!("{color:?} PNG"), |b| {
            png(color, &b.iter().map(RowImage::Rgb).collect::<Vec<_>>())
        });
        let mut reader = ::png::Decoder::new(Cursor::new(&whole))
            .read_info()
            .unwrap();
        let mut pixels = vec![0; reader.output_buffer_size().unwrap()];
        let n = reader.next_frame(&mut pixels).unwrap().buffer_size();
        let channels = if alpha { 4 } else { 3 };
        assert_eq!(n, (W * H) as usize * channels, "{color:?} decodes");
    }
}

#[test]
fn streamed_tiff_is_the_whole_page() {
    let compressions = [
        TiffCompression::None,
        TiffCompression::PackBits,
        TiffCompression::Lzw,
        TiffCompression::Deflate,
    ];
    let stream = |layout, compression, images: &[RowImage<'_>]| {
        let mut w = TiffStream::new(
            Cursor::new(Vec::new()),
            (W, H),
            layout,
            (150.0, 75.0),
            compression,
        )
        .unwrap();
        for &image in images {
            w.push(image).unwrap();
        }
        w.finish().unwrap().into_inner()
    };
    let g = gray();
    for compression in compressions
        .iter()
        .copied()
        .chain([TiffCompression::Group4])
    {
        let mut want = Vec::new();
        write_tiff(RowImage::Bilevel(&g), (150.0, 75.0), compression, &mut want).unwrap();
        same_for_every_band_height(&g, &want, &format!("bilevel {compression:?}"), |b| {
            stream(
                RowLayout::Bilevel,
                compression,
                &b.iter().map(RowImage::Bilevel).collect::<Vec<_>>(),
            )
        });
    }
    for compression in compressions {
        let mut want = Vec::new();
        write_tiff(RowImage::Gray(&g), (150.0, 75.0), compression, &mut want).unwrap();
        same_for_every_band_height(&g, &want, &format!("gray {compression:?}"), |b| {
            stream(
                RowLayout::Gray,
                compression,
                &b.iter().map(RowImage::Gray).collect::<Vec<_>>(),
            )
        });
        for (alpha, layout) in [(false, RowLayout::Rgb), (true, RowLayout::Rgba)] {
            let c = rgb(alpha);
            let mut want = Vec::new();
            write_tiff(RowImage::Rgb(&c), (150.0, 75.0), compression, &mut want).unwrap();
            same_for_every_band_height(&c, &want, &format!("{layout:?} {compression:?}"), |b| {
                stream(
                    layout,
                    compression,
                    &b.iter().map(RowImage::Rgb).collect::<Vec<_>>(),
                )
            });
        }
    }
}

/// The document holding `page` alone.
fn document(page: &PdfPage) -> Vec<u8> {
    let mut writer = PdfWriter::new(Vec::new()).unwrap();
    writer.write_page(page).unwrap();
    writer.finish().unwrap()
}

#[test]
fn a_pdf_page_in_bands_or_spilled_is_the_whole_page() {
    let c = rgb(false);
    let want = document(&PdfPage::encode(RowImage::Rgb(&c), (150.0, 150.0)).unwrap());
    for spill in [false, true] {
        same_for_every_band_height(&c, &want, &format!("spilled {spill}"), |b| {
            let file = spill.then(|| tempfile::tempfile().unwrap());
            let mut e = PdfPageEncoder::new((W, H), RowLayout::Rgb, (150.0, 150.0), file).unwrap();
            for band in b {
                e.push(RowImage::Rgb(band)).unwrap();
            }
            document(&e.finish().unwrap())
        });
    }
}

#[test]
fn bands_that_do_not_continue_the_image_are_refused() {
    let (c, g) = (rgb(false), gray());
    let mut w = PnmWriter::new(Vec::new(), (W, H), PnmKind::Ppm).unwrap();
    assert!(
        w.push(RowImage::Gray(&g)).is_err(),
        "gray rows into a colour image"
    );
    let narrow = Bitmap::<Rgb8>::new(W - 1, 1, 1, false);
    assert!(w.push(RowImage::Rgb(&narrow)).is_err(), "another width");
    w.push(RowImage::Rgb(&band(&c, 0, 100))).unwrap();
    assert!(
        w.push(RowImage::Rgb(&band(&c, 0, 2))).is_err(),
        "past the last row"
    );
    assert!(w.finish().is_err(), "a row missing");
    let mut t = TiffStream::new(
        Cursor::new(Vec::new()),
        (W, H),
        RowLayout::Gray,
        (72.0, 72.0),
        TiffCompression::Lzw,
    )
    .unwrap();
    t.push(RowImage::Gray(&band(&g, 0, 50))).unwrap();
    assert!(t.finish().is_err(), "rows missing");
    assert!(
        PngWriter::new(Vec::new(), (0, H), PngColor::Gray).is_err(),
        "an empty image"
    );
}
