//! ICC profiles embedded by the writers read back, byte for byte, through
//! independent decoders; a profile of another colour space than the
//! image's is refused.

use std::io::Cursor;

use color::{Gray8, Rgb8, Rgb16, SampleDepth};
use raster::Bitmap;
use rasterrocket_encode::{
    EncodeError, IccProfile, JpegOptions, PdfPage, PdfWriter, PngColor, PngWriter, RowImage,
    RowLayout, TiffCompression, TiffStream, write_jpeg, write_png,
};

/// A profile of `space` (`RGB ` or `GRAY`) at version `major.minor`,
/// `len` bytes long, its tail a recognisable pattern.
fn profile(space: [u8; 4], (major, minor): (u8, u8), len: u32) -> IccProfile {
    let mut bytes: Vec<u8> = (0..len).map(|i| i.to_le_bytes()[0]).collect();
    bytes[..4].copy_from_slice(&len.to_be_bytes());
    bytes[8] = major;
    bytes[9] = minor << 4;
    bytes[16..20].copy_from_slice(&space);
    bytes[36..40].copy_from_slice(b"acsp");
    IccProfile::new(bytes).expect("a valid header")
}

fn rgb() -> Bitmap<Rgb8> {
    let mut bmp = Bitmap::new(9, 5, 1, false);
    for (i, v) in bmp.data_mut().iter_mut().enumerate() {
        *v = (i * 7).to_le_bytes()[0];
    }
    bmp
}

fn gray() -> Bitmap<Gray8> {
    let mut bmp = Bitmap::new(9, 5, 1, false);
    for (i, v) in bmp.data_mut().iter_mut().enumerate() {
        *v = (i * 5).to_le_bytes()[0];
    }
    bmp
}

fn png_profile(bytes: &[u8]) -> Option<Vec<u8>> {
    let reader = ::png::Decoder::new(Cursor::new(bytes)).read_info().unwrap();
    reader.info().icc_profile.as_ref().map(|p| p.to_vec())
}

#[test]
fn png_carries_the_profile_in_iccp() {
    let icc = profile(*b"RGB ", (4, 3), 3000);
    let mut whole = Vec::new();
    write_png(&rgb(), None, &mut whole, Some(&icc)).unwrap();
    assert_eq!(png_profile(&whole).as_deref(), Some(icc.bytes()));

    let mut w = PngWriter::new(Vec::new(), (9, 5), PngColor::Rgb, None, Some(&icc)).unwrap();
    w.push(RowImage::Rgb(&rgb())).unwrap();
    assert_eq!(
        png_profile(&w.finish().unwrap()).as_deref(),
        Some(icc.bytes())
    );

    let mut untagged = Vec::new();
    write_png(&rgb(), None, &mut untagged, None).unwrap();
    assert_eq!(png_profile(&untagged), None);

    let gray_icc = profile(*b"GRAY", (2, 1), 200);
    let mut g = Vec::new();
    write_png(&gray(), None, &mut g, Some(&gray_icc)).unwrap();
    assert_eq!(png_profile(&g).as_deref(), Some(gray_icc.bytes()));
}

#[test]
fn tiff_carries_the_profile_in_tag_34675() {
    let icc = profile(*b"RGB ", (4, 2), 1001);
    let img = Bitmap::<Rgb16>::new(9, 5, 1, false);
    for compression in [TiffCompression::None, TiffCompression::Lzw] {
        let mut s = TiffStream::new(
            Cursor::new(Vec::new()),
            (9, 5),
            RowLayout::Rgb16,
            (300.0, 300.0),
            compression,
            Some(&icc),
        )
        .unwrap();
        s.push(RowImage::Rgb16(&img)).unwrap();
        let bytes = s.finish().unwrap().into_inner();
        let mut decoder = ::tiff::decoder::Decoder::new(Cursor::new(bytes)).unwrap();
        assert_eq!(
            decoder
                .get_tag_u8_vec(::tiff::tags::Tag::IccProfile)
                .unwrap(),
            icc.bytes(),
            "{compression:?}"
        );
        let _ = decoder.read_image().unwrap();
    }
}

#[test]
fn jpeg_carries_the_profile_in_app2() {
    let icc = profile(*b"RGB ", (4, 3), 70_000);
    let mut bytes = Vec::new();
    write_jpeg(&rgb(), JpegOptions::default(), None, &mut bytes, Some(&icc)).unwrap();
    let mut decoder =
        zune_jpeg::JpegDecoder::new(zune_jpeg::zune_core::bytestream::ZCursor::new(&bytes));
    decoder.decode_headers().unwrap();
    assert_eq!(decoder.icc_profile().as_deref(), Some(icc.bytes()));
}

#[test]
fn a_pdf_holds_the_profile_once_for_every_page() {
    let text = |icc: &IccProfile, depth| {
        let mut writer = PdfWriter::new(Vec::new(), depth, Some(icc)).unwrap();
        for _ in 0..2 {
            let page = PdfPage::encode(RowImage::Rgb(&rgb()), (72.0, 72.0)).unwrap();
            writer.write_page(&page).unwrap();
        }
        String::from_utf8_lossy(&writer.finish().unwrap()).into_owned()
    };
    let v2 = text(&profile(*b"RGB ", (2, 1), 400), SampleDepth::Eight);
    assert!(v2.starts_with("%PDF-1.4\n"), "{}", &v2[..9]);
    assert_eq!(v2.matches("/N 3/Alternate/DeviceRGB").count(), 1);
    assert_eq!(v2.matches("/ColorSpace[/ICCBased 3 0 R]").count(), 2);
    assert!(!v2.contains("/ColorSpace/DeviceRGB"));
    for (version, header) in [
        ((4, 0), "%PDF-1.5"),
        ((4, 1), "%PDF-1.6"),
        ((4, 4), "%PDF-1.7"),
    ] {
        let doc = text(&profile(*b"RGB ", version, 400), SampleDepth::Eight);
        assert!(doc.starts_with(header), "{version:?}: {}", &doc[..9]);
    }
    let deep = text(&profile(*b"RGB ", (2, 1), 400), SampleDepth::Sixteen);
    assert!(deep.starts_with("%PDF-1.5\n"));
}

#[test]
fn a_profile_of_another_colour_space_is_refused() {
    let rgb_icc = profile(*b"RGB ", (4, 3), 200);
    let gray_icc = profile(*b"GRAY", (4, 3), 200);
    let refused = |r: Result<(), EncodeError>| matches!(r, Err(EncodeError::InvalidInput(_)));
    assert!(refused(write_png(
        &gray(),
        None,
        Vec::new(),
        Some(&rgb_icc)
    )));
    assert!(refused(write_png(
        &rgb(),
        None,
        Vec::new(),
        Some(&gray_icc)
    )));
    assert!(refused(write_jpeg(
        &gray(),
        JpegOptions::default(),
        None,
        Vec::new(),
        Some(&rgb_icc)
    )));
    assert!(refused(
        PngWriter::new(Vec::new(), (9, 5), PngColor::Gray, None, Some(&rgb_icc)).map(drop)
    ));
    assert!(refused(
        TiffStream::new(
            Cursor::new(Vec::new()),
            (9, 5),
            RowLayout::Bilevel,
            (300.0, 300.0),
            TiffCompression::Group4,
            Some(&gray_icc),
        )
        .map(drop)
    ));
    let mut pdf = PdfWriter::new(Vec::new(), SampleDepth::Eight, Some(&gray_icc)).unwrap();
    let page = PdfPage::encode(RowImage::Rgb(&rgb()), (72.0, 72.0)).unwrap();
    assert!(refused(pdf.write_page(&page)));
}
