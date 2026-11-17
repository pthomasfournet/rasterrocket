//! Tests of archive opening, page order and per-page rendering.

use super::*;

#[test]
fn options_default_matches_raster_defaults() {
    let o = ComicOptions::default();
    assert!((o.dpi - 300.0).abs() < f32::EPSILON);
    assert_eq!(o.first_page, 1);
    assert_eq!(o.last_page, u32::MAX);
    assert!(!o.deskew);
}

use crate::test_helpers::make_cbz;

/// A solid grayscale JPEG page, `w`×`h`, every pixel `v`.
fn jpeg_page(w: u32, h: u32, v: u8) -> Vec<u8> {
    use color::Gray8;
    use raster::Bitmap;
    let mut b = Bitmap::<Gray8>::new(w, h, 1, false);
    for y in 0..h {
        for x in 0..w as usize {
            b.row_bytes_mut(y)[x] = v;
        }
    }
    encode::jpeg_gray::<Gray8>(&b, 90).unwrap()
}

/// A minimal valid single-page PDF (one empty 612×792-pt page) — a
/// known-good fixture for the embedded-PDF routing tests.
fn minimal_pdf() -> Vec<u8> {
    b"%PDF-1.4\n\
1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n\
2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n\
3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]>>\nendobj\n\
xref\n0 4\n\
0000000000 65535 f\r\n\
0000000009 00000 n\r\n\
0000000056 00000 n\r\n\
0000000111 00000 n\r\n\
trailer\n<</Size 4 /Root 1 0 R>>\n\
startxref\n180\n%%EOF"
        .to_vec()
}

fn write_temp_cbz(name: &str, bytes: &[u8]) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(name);
    std::fs::write(&path, bytes).unwrap();
    (dir, path)
}

/// An archive held in memory opens as the file would, its container
/// chosen by the extension of the name given, in any case.
#[test]
fn open_comic_bytes_reads_an_archive_in_memory() {
    let cbz = make_cbz(&[("1.jpg", &jpeg_page(4, 4, 50))]);
    let opts = ComicOptions::default();
    let pages: Vec<_> = open_comic_bytes("Book.CBZ", cbz.clone(), &opts)
        .unwrap()
        .collect();
    assert_eq!(pages.len(), 1);
    assert_eq!(pages[0].1.as_ref().unwrap().width, 4);
    assert!(matches!(
        open_comic_bytes("book.cbt", cbz, &opts),
        Err(ComicError::BadArchive(_))
    ));
}

#[test]
fn open_comic_yields_ordered_pages() {
    // Two solid pages + one non-image entry; names out of natural order.
    let cbz = make_cbz(&[
        ("page10.jpg", &jpeg_page(4, 4, 200)),
        ("page2.jpg", &jpeg_page(4, 4, 100)),
        ("ComicInfo.xml", b"<x/>"),
    ]);
    let (_dir, path) = write_temp_cbz("book.cbz", &cbz);

    let pages: Vec<_> = open_comic(&path, &ComicOptions::default())
        .unwrap()
        .collect();
    assert_eq!(pages.len(), 2, "two images, xml filtered out");
    // page2 (value ~100) must come before page10 (value ~200): natural sort.
    let (n1, r1) = &pages[0];
    let (n2, r2) = &pages[1];
    assert_eq!((*n1, *n2), (1, 2));
    let p1 = r1.as_ref().unwrap();
    let p2 = r2.as_ref().unwrap();
    assert!(
        p1.pixels.as_u8().unwrap()[0] < p2.pixels.as_u8().unwrap()[0],
        "page2 darker than page10"
    );
    assert_eq!((p1.width, p1.height), (4, 4));
}

#[test]
fn an_avif_page_renders_from_a_cbz() {
    // 8 × 4 lossless, pixel (30x, 60y, 255 − 30x): see avif-fixtures.py.
    let avif: &[u8] = include_bytes!("../tests/data/rgb-8x4.avif");
    let cbz = make_cbz(&[("2.avif", avif), ("1.jpg", &jpeg_page(4, 4, 200))]);
    let (_dir, path) = write_temp_cbz("book.cbz", &cbz);
    let pages: Vec<_> = open_comic(&path, &ComicOptions::default())
        .unwrap()
        .collect();
    assert_eq!(pages.len(), 2);
    let page = pages[1].1.as_ref().unwrap();
    assert_eq!((page.width, page.height), (8, 4));
    // Gray falls along the row as red gives way to blue's smaller
    // weight... the first pixel (0, 0, 255) is darker than the last of
    // the row (210, 0, 45).
    assert!(
        page.pixels.as_u8().unwrap()[0] < page.pixels.as_u8().unwrap()[7],
        "{:?}",
        &page.pixels.as_u8().unwrap()[..8]
    );
}

#[test]
fn the_session_rotation_turns_page_images() {
    // 8 × 4: the first pixel of the row is darker than the last.
    let avif: &[u8] = include_bytes!("../tests/data/rgb-8x4.avif");
    let cbz = make_cbz(&[("1.avif", avif)]);
    let (_dir, path) = write_temp_cbz("book.cbz", &cbz);
    let render = |rotation| {
        let mut opts = ComicOptions::default();
        opts.session.rotation = rotation;
        let mut pages: Vec<_> = open_comic(&path, &opts).unwrap().collect();
        pages.remove(0).1.unwrap()
    };
    let upright = render(pdf_raster::Rotation::None);
    let turned = render(pdf_raster::Rotation::Cw90);
    assert_eq!((turned.width, turned.height), (4, 8));
    // The top row, left to right, is now the right column, top to bottom.
    let right_column: Vec<u8> = (0..8)
        .map(|y| turned.pixels.as_u8().unwrap()[y * 4 + 3])
        .collect();
    assert_eq!(right_column, upright.pixels.as_u8().unwrap()[..8]);
}

/// At 16 bits a 16-bit image keeps its samples through the turn, an
/// 8-bit one is widened, and an embedded PDF renders at 16 bits.
#[test]
fn sixteen_bit_pages_come_from_16_bit_images_and_pdfs() {
    let n = crate::decode::tests::noise16(6);
    let png = crate::decode::tests::png16(3, 2, ::png::ColorType::Grayscale, &n);
    let cbz = make_cbz(&[("1.png", &png), ("2.jpg", &jpeg_page(4, 4, 200))]);
    let (_dir, path) = write_temp_cbz("book.cbz", &cbz);
    let mut opts = ComicOptions {
        depth: SampleDepth::Sixteen,
        ..ComicOptions::default()
    };
    opts.session.rotation = pdf_raster::Rotation::Cw180;
    let pages: Vec<_> = open_comic(&path, &opts).unwrap().collect();
    let deep = pages[0].1.as_ref().unwrap();
    assert_eq!(deep.format, pdf_raster::PixelFormat::Gray16);
    let reversed: Vec<u16> = n.iter().rev().copied().collect();
    assert_eq!(deep.pixels, pdf_raster::Pixels::U16(reversed));
    let widened = pages[1].1.as_ref().unwrap();
    let pdf_raster::Pixels::U16(samples) = &widened.pixels else {
        panic!("a 16-bit page");
    };
    assert!(samples.iter().all(|v| v.is_multiple_of(257)), "{samples:?}");
    let (_dir, path) = write_temp_cbz("pdf.cbz", &make_cbz(&[("book.pdf", &minimal_pdf())]));
    let (_, page) = open_comic(&path, &opts).unwrap().next().unwrap();
    assert_eq!(page.unwrap().format, pdf_raster::PixelFormat::Gray16);
}

#[test]
fn a_multi_page_tiff_entry_yields_a_page_per_image() {
    let tif = crate::decode::tests::multi_tiff_gray(4, 4, &[(10, 0), (99, 1), (30, 2)]);
    let cbz = make_cbz(&[("1.tif", &tif), ("2.jpg", &jpeg_page(4, 4, 200))]);
    let (_dir, path) = write_temp_cbz("book.cbz", &cbz);
    let pages: Vec<_> = open_comic(&path, &ComicOptions::default())
        .expect("open")
        .map(|(n, page)| (n, page.expect("page").pixels.as_u8().unwrap()[0]))
        .collect();
    let numbers: Vec<u32> = pages.iter().map(|&(n, _)| n).collect();
    assert_eq!(
        numbers,
        [1, 2, 3],
        "two TIFF pages, the thumbnail skipped, then the JPEG"
    );
    assert!(
        pages[0].1 < pages[1].1 && pages[1].1 < pages[2].1,
        "{pages:?}"
    );
}

#[test]
fn a_solid_cbr_renders_its_pages_in_order() {
    let mut builder = rars::Builder::new(rars::ArchiveVersion::Rar50).solid(true);
    for (name, value) in [("p10.jpg", 200), ("p2.jpg", 100)] {
        builder
            .add_bytes(name.as_bytes().to_vec(), jpeg_page(4, 4, value), None, None)
            .expect("entry");
    }
    let (_dir, path) = write_temp_cbz("book.cbr", &builder.to_bytes().expect("archive"));
    let pages: Vec<_> = open_comic(&path, &ComicOptions::default())
        .expect("open")
        .map(|(n, page)| (n, page.expect("page").pixels.as_u8().unwrap()[0]))
        .collect();
    assert_eq!(pages.len(), 2);
    assert_eq!((pages[0].0, pages[1].0), (1, 2));
    assert!(pages[0].1 < pages[1].1, "p2 comes before p10: {pages:?}");
}

#[test]
fn invalid_options_are_refused_before_the_archive_is_read() {
    let (_dir, path) = write_temp_cbz("book.cbz", &make_cbz(&[("1.jpg", &jpeg_page(4, 4, 9))]));
    let mut nan = ComicOptions::new(f32::NAN);
    let mut inverted = ComicOptions::new(300.0);
    inverted.first_page = 5;
    inverted.last_page = 3;
    let mut zero = ComicOptions::new(300.0);
    zero.first_page = 0;
    let mut no_budget = ComicOptions::new(300.0);
    no_budget.max_archive_bytes = 0;
    for opts in [&mut nan, &mut inverted, &mut zero, &mut no_budget] {
        assert!(
            matches!(open_comic(&path, opts), Err(ComicError::InvalidOptions(_))),
            "{opts:?}"
        );
    }
}

#[test]
fn pages_decode_lazily_and_retain_skips_unwanted_ones() {
    let cbz = make_cbz(&[
        ("1.jpg", &jpeg_page(4, 4, 10)),
        ("2.jpg", b"\xFF\xD8\xFF corrupt after the marker"),
        ("3.jpg", &jpeg_page(4, 4, 30)),
    ]);
    let (_dir, path) = write_temp_cbz("book.cbz", &cbz);
    let mut pages = open_comic(&path, &ComicOptions::default()).unwrap();
    assert_eq!(pages.len(), 3);
    pages.retain(|page| page != 2);
    assert_eq!(pages.len(), 2);
    let got: Vec<_> = pages.map(|(n, r)| (n, r.is_ok())).collect();
    assert_eq!(
        got,
        [(1, true), (3, true)],
        "the corrupt page was never decoded"
    );
}

#[test]
fn an_archive_over_its_budget_is_refused_whole() {
    let page = jpeg_page(64, 64, 9);
    let cbz = make_cbz(&[("1.jpg", &page), ("2.jpg", &page)]);
    let (_dir, path) = write_temp_cbz("book.cbz", &cbz);
    let mut opts = ComicOptions {
        max_archive_bytes: u64::try_from(page.len() * 2 - 1).unwrap(),
        ..ComicOptions::default()
    };
    let limit = opts.max_archive_bytes;
    assert!(matches!(
        open_comic(&path, &opts),
        Err(ComicError::ArchiveTooLarge { limit: l }) if l == limit
    ));
    opts.max_archive_bytes += 1;
    assert_eq!(open_comic(&path, &opts).unwrap().len(), 2);
}

#[test]
fn empty_archive_is_no_images() {
    let cbz = make_cbz(&[]);
    let (_dir, path) = write_temp_cbz("empty.cbz", &cbz);
    assert!(matches!(
        open_comic(&path, &ComicOptions::default()),
        Err(ComicError::NoImages)
    ));
}

#[test]
fn a_damaged_cbr_is_a_bad_archive() {
    let (_dir, path) = write_temp_cbz("x.cbr", b"Rar!\x1a\x07\x00");
    assert!(matches!(
        open_comic(&path, &ComicOptions::default()),
        Err(ComicError::BadArchive(_))
    ));
}

#[test]
fn lying_extension_is_skipped_not_errored() {
    // An entry named .jpg whose bytes are not an image must be skipped
    // (warned), not surfaced as a page — and must not consume a page number.
    let cbz = make_cbz(&[
        ("001.jpg", b"NOT AN IMAGE AT ALL"),
        ("002.jpg", &jpeg_page(4, 4, 120)),
    ]);
    let (_dir, path) = write_temp_cbz("mixed.cbz", &cbz);

    let pages: Vec<_> = open_comic(&path, &ComicOptions::default())
        .unwrap()
        .collect();
    assert_eq!(
        pages.len(),
        1,
        "the junk entry is skipped, the real one kept"
    );
    assert_eq!(pages[0].0, 1, "kept page is numbered 1, not 2");
    assert!(pages[0].1.is_ok());
}

#[test]
fn cbz_with_only_a_pdf_renders_it() {
    let cbz = make_cbz(&[("book.pdf", &minimal_pdf())]);
    let (_dir, path) = write_temp_cbz("book.cbz", &cbz);
    let pages: Vec<_> = open_comic(&path, &ComicOptions::default())
        .unwrap()
        .collect();
    assert!(!pages.is_empty(), "the embedded PDF must yield pages");
    assert_eq!(pages[0].0, 1, "PDF's own 1-based page numbering");
    assert!(pages[0].1.is_ok());
}

/// A PDF of the objects `objects`, numbered from 1; object 1 is the
/// catalog.
fn pdf_of(objects: &[&str]) -> Vec<u8> {
    let mut pdf = b"%PDF-1.4\n".to_vec();
    let mut offsets = Vec::new();
    for (i, body) in objects.iter().enumerate() {
        offsets.push(pdf.len());
        pdf.extend_from_slice(format!("{} 0 obj\n{body}\nendobj\n", i + 1).as_bytes());
    }
    let xref = pdf.len();
    pdf.extend_from_slice(
        format!("xref\n0 {}\n0000000000 65535 f\r\n", objects.len() + 1).as_bytes(),
    );
    for offset in offsets {
        pdf.extend_from_slice(format!("{offset:010} 00000 n\r\n").as_bytes());
    }
    pdf.extend_from_slice(
        format!(
            "trailer\n<</Size {} /Root 1 0 R>>\nstartxref\n{xref}\n%%EOF",
            objects.len() + 1
        )
        .as_bytes(),
    );
    pdf
}

/// An embedded PDF is rendered with the options' session: a page whose
/// only content is a black annotation renders black with annotations
/// drawn and white with them hidden.
#[test]
fn embedded_pdf_uses_the_session_options() {
    let pdf = pdf_of(&[
        "<</Type /Catalog /Pages 2 0 R>>",
        "<</Type /Pages /Kids [3 0 R] /Count 1>>",
        "<</Type /Page /Parent 2 0 R /MediaBox [0 0 8 8] \
         /Annots [<</Type /Annot /Subtype /Square /Rect [0 0 8 8] /AP <</N 4 0 R>> >>]>>",
        "<</Type /XObject /Subtype /Form /BBox [0 0 8 8] /Length 16>>\nstream\n0 g 0 0 8 8 re f\nendstream",
    ]);
    let (_dir, path) = write_temp_cbz("book.cbz", &make_cbz(&[("book.pdf", &pdf)]));
    let render = |annotations: bool| {
        let mut opts = ComicOptions::new(72.0);
        opts.session.annotations = annotations;
        let (_, page) = open_comic(&path, &opts).unwrap().next().unwrap();
        page.unwrap().pixels.as_u8().unwrap().to_vec()
    };
    let shown = render(true);
    assert!(
        !shown.is_empty() && shown.iter().all(|&v| v == 0),
        "{shown:?}"
    );
    let hidden = render(false);
    assert!(hidden.iter().all(|&v| v == 255), "{hidden:?}");
}

#[test]
fn cbz_with_two_pdfs_is_ambiguous() {
    let cbz = make_cbz(&[("vol1.pdf", &minimal_pdf()), ("vol2.pdf", &minimal_pdf())]);
    let (_dir, path) = write_temp_cbz("two.cbz", &cbz);
    assert!(matches!(
        open_comic(&path, &ComicOptions::default()),
        Err(ComicError::AmbiguousArchive(_))
    ));
}

#[test]
fn images_win_over_pdf() {
    // Both an image page and a PDF present: render the image, ignore the PDF.
    let cbz = make_cbz(&[
        ("page1.jpg", &jpeg_page(4, 4, 90)),
        ("book.pdf", &minimal_pdf()),
    ]);
    let (_dir, path) = write_temp_cbz("both.cbz", &cbz);
    let pages: Vec<_> = open_comic(&path, &ComicOptions::default())
        .unwrap()
        .collect();
    assert_eq!(pages.len(), 1, "only the image page is rendered");
    let page = pages[0].1.as_ref().unwrap();
    assert_eq!(
        (page.width, page.height),
        (4, 4),
        "the rendered page is the 4×4 image, not a 612×792-pt PDF page"
    );
}

/// The one-page archives of `tests/data`, one per container (RAR in its
/// 4 and 5 formats), each hold an 8 × 8 JPEG page and a `ComicInfo.xml`.
#[test]
fn committed_archives_open_in_every_container() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    for name in [
        "one-page.cbz",
        "one-page.cb7",
        "one-page.cbt",
        "one-page-rar4.cbr",
        "one-page-rar5.cbr",
    ] {
        let pages: Vec<_> = open_comic(&dir.join(name), &ComicOptions::default())
            .unwrap_or_else(|e| panic!("{name}: {e}"))
            .collect();
        assert_eq!(pages.len(), 1, "{name}");
        let page = pages[0]
            .1
            .as_ref()
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!((page.width, page.height), (8, 8), "{name}");
    }
}

/// The PNG, WebP and TIFF pages of `tests/data` decode, a two-page TIFF to
/// two pages.
#[test]
fn committed_page_images_decode() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    for (name, count) in [
        ("rgb-16x8.png", 1),
        ("rgb-16x8-lossless.webp", 1),
        ("rgb-16x8-lossy.webp", 1),
        ("gray-16x8-packbits.tif", 1),
        ("gray-16x8-two-pages.tif", 2),
    ] {
        let image = std::fs::read(dir.join(name)).expect("fixture");
        let cbz = make_cbz(&[(name, &image)]);
        let pages: Vec<_> = open_comic_bytes("book.cbz", cbz, &ComicOptions::default())
            .unwrap_or_else(|e| panic!("{name}: {e}"))
            .collect();
        assert_eq!(pages.len(), count, "{name}");
        for (_, page) in &pages {
            let page = page.as_ref().unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!((page.width, page.height), (16, 8), "{name}");
        }
    }
}
