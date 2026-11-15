//! The row-streamed writers: an image written in bands of any height is the
//! image written as one band, byte for byte, and the whole-image functions
//! write the same bytes where they share a format.

use std::io::Cursor;

use color::{Cmyk8, Cmyk16, Gray8, Gray16, Rgb8, Rgb16, SampleDepth};
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
        let mut w = PngWriter::new(Vec::new(), (W, H), color, None, None).unwrap();
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
    let whole = png(PngColor::Bilevel, &[RowImage::Gray(&g)]);
    same_for_every_band_height(&g, &whole, "bilevel PNG", |b| {
        png(
            PngColor::Bilevel,
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
            None,
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

// ── CMYK images ───────────────────────────────────────────────────────────────

fn cmyk() -> Bitmap<Cmyk8> {
    let mut bmp = Bitmap::new(W, H, 4, false);
    for y in 0..H {
        for (i, v) in bmp.row_bytes_mut(y)[..W as usize * 4]
            .iter_mut()
            .enumerate()
        {
            *v = if i % 16 < 8 {
                0
            } else {
                (i * 5 + y as usize * 9).to_le_bytes()[0]
            };
        }
    }
    bmp
}

fn cmyk16() -> Bitmap<Cmyk16> {
    let mut bmp = Bitmap::new(W, H, 1, false);
    for y in 0..H {
        for (i, v) in bmp.row_samples_mut(y)[..W as usize * 4]
            .iter_mut()
            .enumerate()
        {
            *v = u16::try_from((i * 4099 + y as usize * 31) % 65_536).expect("in range");
        }
    }
    bmp
}

#[test]
fn a_streamed_cmyk_tiff_is_the_whole_page_at_both_depths() {
    let stream = |layout, compression, images: &[RowImage<'_>]| {
        let mut w = TiffStream::new(
            Cursor::new(Vec::new()),
            (W, H),
            layout,
            (150.0, 75.0),
            compression,
            None,
        )
        .unwrap();
        for &image in images {
            w.push(image).unwrap();
        }
        w.finish().unwrap().into_inner()
    };
    let (narrow, wide) = (cmyk(), cmyk16());
    for compression in [
        TiffCompression::None,
        TiffCompression::PackBits,
        TiffCompression::Lzw,
        TiffCompression::Deflate,
    ] {
        let mut want = Vec::new();
        write_tiff(
            RowImage::Cmyk(&narrow),
            (150.0, 75.0),
            compression,
            &mut want,
        )
        .unwrap();
        same_for_every_band_height(&narrow, &want, &format!("CMYK {compression:?}"), |b| {
            stream(
                RowLayout::Cmyk,
                compression,
                &b.iter().map(RowImage::Cmyk).collect::<Vec<_>>(),
            )
        });
        let mut want = Vec::new();
        write_tiff(
            RowImage::Cmyk16(&wide),
            (150.0, 75.0),
            compression,
            &mut want,
        )
        .unwrap();
        same_for_every_band_height(&wide, &want, &format!("CMYK16 {compression:?}"), |b| {
            stream(
                RowLayout::Cmyk16,
                compression,
                &b.iter().map(RowImage::Cmyk16).collect::<Vec<_>>(),
            )
        });
    }
}

#[test]
fn cmyk_bands_are_refused_by_the_formats_without_cmyk() {
    let c = cmyk();
    let mut pnm = PnmWriter::new(Vec::new(), (W, H), PnmKind::Ppm).unwrap();
    assert!(pnm.push(RowImage::Cmyk(&c)).is_err(), "PPM");
    let mut png = PngWriter::new(Vec::new(), (W, H), PngColor::Rgb, None, None).unwrap();
    assert!(png.push(RowImage::Cmyk(&c)).is_err(), "PNG");
    assert!(
        PdfPageEncoder::new((W, H), RowLayout::Cmyk, (72.0, 72.0), None).is_err(),
        "a PDF page"
    );
    assert!(
        PdfPage::encode(RowImage::Cmyk(&c), (72.0, 72.0)).is_err(),
        "a PDF page"
    );
}

/// The document holding `page` alone.
fn document(page: &PdfPage) -> Vec<u8> {
    document_of(page, SampleDepth::Eight)
}

/// The document of pages up to `depth` deep holding `page` alone.
fn document_of(page: &PdfPage, depth: SampleDepth) -> Vec<u8> {
    let mut writer = PdfWriter::new(Vec::new(), depth, None).unwrap();
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
        None,
    )
    .unwrap();
    t.push(RowImage::Gray(&band(&g, 0, 50))).unwrap();
    assert!(t.finish().is_err(), "rows missing");
    assert!(
        PngWriter::new(Vec::new(), (0, H), PngColor::Gray, None, None).is_err(),
        "an empty image"
    );
}

// ── 16-bit images ─────────────────────────────────────────────────────────────

/// A 16-bit RGB image whose samples use all 16 bits, with a varied alpha
/// plane when `alpha`.
fn rgb16(alpha: bool) -> Bitmap<Rgb16> {
    let mut bmp = Bitmap::new(W, H, 4, alpha);
    for y in 0..H {
        for (i, v) in bmp.row_samples_mut(y)[..W as usize * 3]
            .iter_mut()
            .enumerate()
        {
            *v = u16::try_from((i * 1_009 + y as usize * 7_919) % 65_536).unwrap();
        }
    }
    if let Some(a) = bmp.alpha_plane_mut() {
        for (i, v) in a.iter_mut().enumerate() {
            *v = u16::try_from((i * 4_099) % 65_536).unwrap();
        }
    }
    bmp
}

fn gray16() -> Bitmap<Gray16> {
    let mut bmp = Bitmap::new(W, H, 4, false);
    for y in 0..H {
        for (x, v) in bmp.row_samples_mut(y)[..W as usize].iter_mut().enumerate() {
            *v = u16::try_from((x * 1_741 + y as usize * 613) % 65_536).unwrap();
        }
    }
    bmp
}

/// The live samples of `bmp`, row by row, big-endian; RGBA with its alpha
/// plane when `alpha` (opaque without one).
fn be_samples<P: color::Pixel<Sample = u16>>(bmp: &Bitmap<P>, alpha: bool) -> Vec<u8> {
    let n = P::NCOMPS;
    let mut out = Vec::new();
    for y in 0..bmp.height {
        let row = &bmp.row_samples(y)[..bmp.width as usize * n];
        for (x, px) in row.chunks(n).enumerate() {
            for &v in px {
                out.extend_from_slice(&v.to_be_bytes());
            }
            if alpha {
                let a = bmp.alpha_row(y).map_or(u16::MAX, |a| a[x]);
                out.extend_from_slice(&a.to_be_bytes());
            }
        }
    }
    out
}

#[test]
fn a_16_bit_netpbm_image_has_maxval_65535_and_big_endian_samples() {
    let (c, g) = (rgb16(true), gray16());
    let pnm = |kind, images: &[RowImage<'_>]| {
        let mut w = PnmWriter::new(Vec::new(), (W, H), kind).unwrap();
        for &image in images {
            w.push(image).unwrap();
        }
        w.finish().unwrap()
    };
    let want = [
        format!("P6\n{W} {H}\n65535\n").into_bytes(),
        be_samples(&c, false),
    ]
    .concat();
    same_for_every_band_height(&c, &want, "16-bit PPM", |b| {
        pnm(
            PnmKind::Ppm16,
            &b.iter().map(RowImage::Rgb16).collect::<Vec<_>>(),
        )
    });
    let want = [
        format!("P5\n{W} {H}\n65535\n").into_bytes(),
        be_samples(&g, false),
    ]
    .concat();
    same_for_every_band_height(&g, &want, "16-bit PGM", |b| {
        pnm(
            PnmKind::Pgm16,
            &b.iter().map(RowImage::Gray16).collect::<Vec<_>>(),
        )
    });
}

#[test]
fn a_16_bit_png_decodes_to_its_samples() {
    let png = |color, images: &[RowImage<'_>]| {
        let mut w = PngWriter::new(Vec::new(), (W, H), color, None, None).unwrap();
        for &image in images {
            w.push(image).unwrap();
        }
        w.finish().unwrap()
    };
    let decode = |bytes: &[u8]| {
        let mut reader = ::png::Decoder::new(Cursor::new(bytes)).read_info().unwrap();
        assert_eq!(reader.info().bit_depth, ::png::BitDepth::Sixteen);
        let mut pixels = vec![0; reader.output_buffer_size().unwrap()];
        let n = reader.next_frame(&mut pixels).unwrap().buffer_size();
        pixels.truncate(n);
        pixels
    };
    let g = gray16();
    let whole = png(PngColor::Gray16, &[RowImage::Gray16(&g)]);
    assert_eq!(decode(&whole), be_samples(&g, false), "gray");
    same_for_every_band_height(&g, &whole, "16-bit gray PNG", |b| {
        png(
            PngColor::Gray16,
            &b.iter().map(RowImage::Gray16).collect::<Vec<_>>(),
        )
    });
    for (alpha, color) in [(false, PngColor::Rgb16), (true, PngColor::Rgba16)] {
        let c = rgb16(true);
        let whole = png(color, &[RowImage::Rgb16(&c)]);
        assert_eq!(decode(&whole), be_samples(&c, alpha), "{color:?}");
        same_for_every_band_height(&c, &whole, &format!("{color:?} PNG"), |b| {
            png(color, &b.iter().map(RowImage::Rgb16).collect::<Vec<_>>())
        });
    }
}

#[test]
fn a_16_bit_tiff_reads_back_as_its_samples() {
    let stream = |layout, compression, images: &[RowImage<'_>]| {
        let mut w = TiffStream::new(
            Cursor::new(Vec::new()),
            (W, H),
            layout,
            (300.0, 300.0),
            compression,
            None,
        )
        .unwrap();
        for &image in images {
            w.push(image).unwrap();
        }
        w.finish().unwrap().into_inner()
    };
    let read = |bytes: Vec<u8>| {
        let mut decoder = tiff::decoder::Decoder::new(Cursor::new(bytes)).unwrap();
        match decoder.read_image().unwrap() {
            tiff::decoder::DecodingResult::U16(v) => v,
            other => panic!("not 16-bit samples: {other:?}"),
        }
    };
    let native = |bmp_rows: Vec<u8>| -> Vec<u16> {
        bmp_rows
            .chunks(2)
            .map(|b| u16::from_be_bytes([b[0], b[1]]))
            .collect()
    };
    let (c, g) = (rgb16(true), gray16());
    for compression in [
        TiffCompression::None,
        TiffCompression::PackBits,
        TiffCompression::Lzw,
        TiffCompression::Deflate,
    ] {
        let whole = stream(RowLayout::Gray16, compression, &[RowImage::Gray16(&g)]);
        assert_eq!(
            read(whole.clone()),
            native(be_samples(&g, false)),
            "gray {compression:?}"
        );
        same_for_every_band_height(&g, &whole, &format!("16-bit gray {compression:?}"), |b| {
            stream(
                RowLayout::Gray16,
                compression,
                &b.iter().map(RowImage::Gray16).collect::<Vec<_>>(),
            )
        });
        for (alpha, layout) in [(false, RowLayout::Rgb16), (true, RowLayout::Rgba16)] {
            let whole = stream(layout, compression, &[RowImage::Rgb16(&c)]);
            assert_eq!(
                read(whole),
                native(be_samples(&c, alpha)),
                "{layout:?} {compression:?}"
            );
        }
    }
}

#[test]
fn a_16_bit_pdf_page_is_pdf_1_5_of_16_bits_per_component() {
    let c = rgb16(false);
    let page = PdfPage::encode(RowImage::Rgb16(&c), (150.0, 150.0)).unwrap();
    let doc = document_of(&page, SampleDepth::Sixteen);
    assert!(doc.starts_with(b"%PDF-1.5\n"));
    let text = String::from_utf8_lossy(&doc);
    assert!(
        text.contains("/ColorSpace/DeviceRGB/BitsPerComponent 16/"),
        "{text:.300}"
    );
    let start = doc.windows(7).position(|w| w == b"stream\n").unwrap() + 7;
    let mut data = Vec::new();
    let _: usize = std::io::Read::read_to_end(
        &mut flate2::read::ZlibDecoder::new(&doc[start..]),
        &mut data,
    )
    .unwrap();
    assert_eq!(data, be_samples(&c, false));
    let mut eight = PdfWriter::new(Vec::new(), SampleDepth::Eight, None).unwrap();
    assert!(
        eight.write_page(&page).is_err(),
        "a 16-bit page in a PDF 1.4 file"
    );
    let g = gray16();
    let want = document_of(
        &PdfPage::encode(RowImage::Gray16(&g), (150.0, 150.0)).unwrap(),
        SampleDepth::Sixteen,
    );
    same_for_every_band_height(&g, &want, "16-bit gray PDF page", |b| {
        let mut e = PdfPageEncoder::new((W, H), RowLayout::Gray16, (150.0, 150.0), None).unwrap();
        for band in b {
            e.push(RowImage::Gray16(band)).unwrap();
        }
        document_of(&e.finish().unwrap(), SampleDepth::Sixteen)
    });
}

#[test]
fn bands_of_the_other_depth_are_refused() {
    let (c8, c16) = (rgb(false), rgb16(false));
    let mut w = PngWriter::new(Vec::new(), (W, H), PngColor::Rgb16, None, None).unwrap();
    assert!(
        w.push(RowImage::Rgb(&c8)).is_err(),
        "8-bit rows into a 16-bit image"
    );
    let mut w = PngWriter::new(Vec::new(), (W, H), PngColor::Rgb, None, None).unwrap();
    assert!(
        w.push(RowImage::Rgb16(&c16)).is_err(),
        "16-bit rows into an 8-bit image"
    );
}
