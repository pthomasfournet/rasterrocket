//! End-to-end tests of the `rrocket` binary on PDFs assembled in memory.
//!
//! Pixel correctness is asserted by the library's rendering suites
//! (`crates/pdf_raster/tests`); these tests assert the contract between the
//! binary and the library: output naming, page selection, file formats,
//! exit status, and that a page written by the binary carries exactly the
//! pixels the library renders for it.

// The shared PDF assembly helpers name the library `rasterrocket`; this crate
// depends on it as `pdf_raster`.
extern crate pdf_raster as rasterrocket;

#[path = "../../pdf_raster/tests/common/mod.rs"]
#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use common::{assemble_pdf, stream_object, with_page_tree};
use rasterrocket::{BackendPolicy, SessionConfig, open_session_from_bytes, render_page_rgb};

/// Page 1: 100 × 50 pt, a black square at (10, 10) of side 20.
/// Page 2: 60 × 80 pt, a blue rectangle over the lower half.
fn two_page_pdf() -> Vec<u8> {
    assemble_pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 50] /Contents 5 0 R >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 60 80] /Contents 6 0 R >>".to_vec(),
        stream_object("", b"0 g 10 10 20 20 re f"),
        stream_object("", b"0 0 1 rg 0 0 60 40 re f"),
    ])
}

/// One 40 × 40 pt page drawing an image whose JPEG data is not a JPEG.
fn broken_image_pdf() -> Vec<u8> {
    with_page_tree(&[
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 40 40] \
           /Resources << /XObject << /Im1 5 0 R >> >> /Contents 4 0 R >>"
            .to_vec(),
        stream_object("", b"q 40 0 0 40 0 0 cm /Im1 Do Q"),
        stream_object(
            "/Type /XObject /Subtype /Image /Width 8 /Height 8 /ColorSpace /DeviceGray \
             /BitsPerComponent 8 /Filter /DCTDecode",
            b"not a jpeg",
        ),
    ])
}

/// A scratch directory holding `pdf` as `in.pdf`.
fn workdir(pdf: &[u8]) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("create temp dir");
    let input = dir.path().join("in.pdf");
    std::fs::write(&input, pdf).expect("write input PDF");
    (dir, input)
}

/// Run `rrocket` on `input` with `args`, writing under `<dir>/out`.
fn rrocket(dir: &Path, input: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rrocket"))
        .args(["--no-ram", "--backend", "cpu"])
        .args(args)
        .arg(input)
        .arg(dir.join("out"))
        .output()
        .expect("run rrocket")
}

/// The names of the files in `dir`, sorted, without `in.pdf`.
fn outputs(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .expect("read temp dir")
        .map(|e| {
            e.expect("dir entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .filter(|n| n != "in.pdf")
        .collect();
    names.sort();
    names
}

/// `(magic, width, height, pixel bytes)` of a binary PNM file; a P4 bitmap
/// has no maximum value.
fn read_pnm(path: &Path) -> (String, u32, u32, Vec<u8>) {
    let bytes = std::fs::read(path).expect("read PNM");
    let header_fields = if bytes.starts_with(b"P4") { 3 } else { 4 };
    let mut fields = Vec::new();
    let mut pos = 0;
    while fields.len() < header_fields {
        while bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        let start = pos;
        while !bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        fields.push(String::from_utf8_lossy(&bytes[start..pos]).into_owned());
    }
    // One whitespace byte separates the header from the raster.
    let data = bytes[pos + 1..].to_vec();
    let num = |s: &str| s.parse::<u32>().expect("PNM header number");
    if header_fields == 4 {
        assert_eq!(fields[3], "255", "maximum value");
    }
    (fields[0].clone(), num(&fields[1]), num(&fields[2]), data)
}

fn assert_success(out: &Output) {
    assert!(
        out.status.success(),
        "rrocket failed ({}): {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn every_page_is_written_with_the_library_pixels() {
    let pdf = two_page_pdf();
    let (dir, input) = workdir(&pdf);
    let out = rrocket(dir.path(), &input, &["-r", "72"]);
    assert_success(&out);
    assert_eq!(outputs(dir.path()), ["out-1.ppm", "out-2.ppm"]);

    let session = open_session_from_bytes(pdf, &SessionConfig::with_policy(BackendPolicy::CpuOnly))
        .expect("library opens the PDF");
    for (page, (w, h)) in [(1, (100, 50)), (2, (60, 80))] {
        let (magic, width, height, data) = read_pnm(&dir.path().join(format!("out-{page}.ppm")));
        assert_eq!((magic.as_str(), width, height), ("P6", w, h), "page {page}");
        let lib = render_page_rgb(&session, page, 1.0).expect("library renders");
        assert_eq!(
            data,
            lib.data(),
            "page {page}: binary and library pixels differ"
        );
    }
}

#[test]
fn page_range_selects_pages() {
    let (dir, input) = workdir(&two_page_pdf());
    assert_success(&rrocket(
        dir.path(),
        &input,
        &["-r", "36", "-f", "2", "-l", "2"],
    ));
    assert_eq!(outputs(dir.path()), ["out-2.ppm"]);
    let (_, width, height, _) = read_pnm(&dir.path().join("out-2.ppm"));
    assert_eq!((width, height), (30, 40));
}

#[test]
fn gray_output_is_pgm() {
    let (dir, input) = workdir(&two_page_pdf());
    assert_success(&rrocket(
        dir.path(),
        &input,
        &["-r", "72", "--gray", "-l", "1"],
    ));
    let (magic, width, height, data) = read_pnm(&dir.path().join("out-1.pgm"));
    assert_eq!((magic.as_str(), width, height), ("P5", 100, 50));
    // The square spans x 10..30 and, from the top, y 20..40.
    assert_eq!(data[25 * 100 + 15], 0, "inside the square");
    assert_eq!(data[5 * 100 + 50], 255, "outside the square");
}

#[test]
fn png_output_has_the_png_signature() {
    let (dir, input) = workdir(&two_page_pdf());
    assert_success(&rrocket(dir.path(), &input, &["-r", "72", "--png"]));
    assert_eq!(outputs(dir.path()), ["out-1.png", "out-2.png"]);
    let png = std::fs::read(dir.path().join("out-1.png")).expect("read PNG");
    assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"));
    // IHDR: width and height, big-endian, at bytes 16..24.
    assert_eq!(&png[16..24], &[0, 0, 0, 100, 0, 0, 0, 50]);
}

#[test]
fn mono_tiff_carries_the_pbm_bits_under_every_compression() {
    let (dir, input) = workdir(&two_page_pdf());
    assert_success(&rrocket(
        dir.path(),
        &input,
        &["-r", "72", "--mono", "-l", "1"],
    ));
    let (magic, width, height, pbm) = read_pnm(&dir.path().join("out-1.pbm"));
    assert_eq!((magic.as_str(), width, height), ("P4", 100, 50));
    for method in ["none", "packbits", "lzw", "deflate", "g4"] {
        assert_success(&rrocket(
            dir.path(),
            &input,
            &[
                "-r",
                "72",
                "--mono",
                "-l",
                "1",
                "--tiff",
                "--tiffcompression",
                method,
            ],
        ));
        let file = std::fs::read(dir.path().join("out-1.tif")).expect("read TIFF");
        let mut decoder = tiff::decoder::Decoder::new(std::io::Cursor::new(file)).expect("a TIFF");
        assert_eq!(decoder.dimensions().expect("dimensions"), (100, 50));
        let tiff::decoder::DecodingResult::U8(bits) = decoder.read_image().expect("decodes") else {
            panic!("1-bit samples read as bytes");
        };
        // The reader inverts WhiteIsZero; PBM stores 1 for black.
        let black: Vec<u8> = bits.iter().map(|b| !b).collect();
        assert_eq!(black, pbm, "{method}");
    }
}

#[test]
fn invert_auto_turns_a_dark_page_to_ink_on_paper() {
    // A black page with a white square: the reversed-out case.
    let pdf = with_page_tree(&[
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 50] /Contents 4 0 R >>".to_vec(),
        stream_object("", b"0 g 0 0 100 50 re f 1 g 10 10 20 20 re f"),
    ]);
    let (dir, input) = workdir(&pdf);
    for (flag, inside, outside) in [("--invert=never", 255, 0), ("--invert=auto", 0, 255)] {
        assert_success(&rrocket(dir.path(), &input, &["-r", "72", "--gray", flag]));
        let (_, _, _, data) = read_pnm(&dir.path().join("out-1.pgm"));
        assert_eq!(data[25 * 100 + 15], inside, "{flag}: inside the square");
        assert_eq!(data[5 * 100 + 50], outside, "{flag}: the ground");
    }
    // The light page of two_page_pdf is left alone by auto.
    let (dir, input) = workdir(&two_page_pdf());
    assert_success(&rrocket(
        dir.path(),
        &input,
        &["-r", "72", "--gray", "-l", "1", "--invert=auto"],
    ));
    let (_, _, _, data) = read_pnm(&dir.path().join("out-1.pgm"));
    assert_eq!(data[25 * 100 + 15], 0, "the black square stays black");
}

/// The pages of a Group 4 fax document: `(width, height, ink fraction)`
/// each, from the tiff crate, which decodes Group 4 but not Group 3.
fn fax_pages(path: &Path) -> Vec<(u32, u32, f64)> {
    let file = std::fs::read(path).expect("read the fax document");
    let mut decoder = tiff::decoder::Decoder::new(std::io::Cursor::new(file)).expect("a TIFF");
    let mut pages = Vec::new();
    loop {
        let (w, h) = decoder.dimensions().expect("dimensions");
        let tiff::decoder::DecodingResult::U8(bits) = decoder.read_image().expect("decodes") else {
            panic!("1-bit samples read as bytes");
        };
        // The reader inverts WhiteIsZero: its 0 bits are ink.
        let ink: u32 = bits.iter().map(|b| b.count_zeros()).sum();
        pages.push((w, h, f64::from(ink) / f64::from(w * h)));
        if !decoder.more_images() {
            return pages;
        }
        decoder.next_image().expect("next page");
    }
}

#[test]
fn fax_writes_one_document_of_fax_width_pages() {
    // A portrait page with a small square, a landscape page, a dark page.
    let pdf = assemble_pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R 4 0 R 5 0 R] /Count 3 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 150] /Contents 6 0 R >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Contents 6 0 R >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 150] /Contents 7 0 R >>".to_vec(),
        stream_object("", b"0 g 10 10 20 20 re f"),
        stream_object("", b"0 g 0 0 100 150 re f"),
    ]);
    let (dir, input) = workdir(&pdf);
    assert_success(&rrocket(dir.path(), &input, &["--fax"]));
    assert_eq!(outputs(dir.path()), ["out.tif"]);
    let file = std::fs::read(dir.path().join("out.tif")).expect("read the fax document");
    let mut decoder = tiff::decoder::Decoder::new(std::io::Cursor::new(file)).expect("a TIFF");
    for page in 0..3 {
        let compression = decoder.get_tag_u32(tiff::tags::Tag::Compression).unwrap();
        assert_eq!(compression, 3, "page {page}: Group 3 by default");
        if page < 2 {
            decoder.next_image().expect("next page");
        }
    }
    assert!(!decoder.more_images());

    assert_success(&rrocket(
        dir.path(),
        &input,
        &["--fax", "--tiffcompression", "g4"],
    ));
    let pages = fax_pages(&dir.path().join("out.tif"));
    assert_eq!(pages.len(), 3);
    // 1728 pixels across: the portrait page by its width, the landscape
    // page turned and spanned by its height; 196/204 as many lines per
    // point along: 150 and 200 points at 17.28 × 196/204 lines each.
    let lines = |pts: f64| pts * 17.28 * 196.0 / 204.0;
    for ((w, h, _), pts) in pages.iter().zip([150.0, 200.0, 150.0]) {
        assert_eq!(*w, 1728);
        assert!(
            (f64::from(*h) - lines(pts)).abs() <= 1.0,
            "{h} lines for {pts} pt"
        );
    }
    // The square inks 400 of the page's 15000 square points; the black
    // page is inverted to blank.
    assert!(
        (pages[0].2 - 400.0 / 15000.0).abs() < 0.002,
        "{}",
        pages[0].2
    );
    assert!(
        pages[2].2 < 0.001,
        "the dark page is inverted: {}",
        pages[2].2
    );

    assert_success(&rrocket(
        dir.path(),
        &input,
        &[
            "--fax",
            "--tiffcompression",
            "g4",
            "--invert=never",
            "-f",
            "3",
        ],
    ));
    let pages = fax_pages(&dir.path().join("out.tif"));
    assert_eq!(pages.len(), 1);
    assert!(pages[0].2 > 0.999, "the dark page kept: {}", pages[0].2);
}

/// `--fax-to` hands the written document to the fax service: here a
/// stand-in `sendfax` on `PATH` that records its arguments and the document,
/// then exits with `status`.
#[cfg(unix)]
fn fax_to_through(status: u8) -> (tempfile::TempDir, Output) {
    use std::os::unix::fs::PermissionsExt as _;

    let (dir, input) = workdir(&two_page_pdf());
    let bin = dir.path().join("bin");
    std::fs::create_dir(&bin).expect("create bin dir");
    let script = bin.join("sendfax");
    std::fs::write(
        &script,
        format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"{dir}/args\"\ncp \"$5\" \"{dir}/sent.tif\"\nexit {status}\n",
            dir = dir.path().display()
        ),
    )
    .expect("write sendfax");
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755))
        .expect("make sendfax executable");
    let path = format!(
        "{}:{}",
        bin.display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let out = Command::new(env!("CARGO_BIN_EXE_rrocket"))
        .env("PATH", path)
        .args(["--no-ram", "--backend", "cpu", "--fax-to", "+1 555 0100"])
        .arg(&input)
        .arg(dir.path().join("out"))
        .output()
        .expect("run rrocket");
    (dir, out)
}

#[cfg(unix)]
#[test]
fn fax_to_submits_the_document_to_sendfax() {
    let (dir, out) = fax_to_through(0);
    assert_success(&out);
    let args = std::fs::read_to_string(dir.path().join("args")).expect("sendfax ran");
    let out_tif = dir.path().join("out.tif");
    assert_eq!(
        args.lines().collect::<Vec<_>>(),
        [
            "-n",
            "-d",
            "+1 555 0100",
            "--",
            out_tif.to_str().expect("UTF-8 path")
        ]
    );
    assert_eq!(
        std::fs::read(dir.path().join("sent.tif")).expect("the document was handed over"),
        std::fs::read(&out_tif).expect("the document was written")
    );

    let (_dir, out) = fax_to_through(1);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("not sent") && stderr.contains("sendfax refused"),
        "{stderr}"
    );
}

#[test]
fn missing_input_fails_without_output() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let out = rrocket(dir.path(), &dir.path().join("absent.pdf"), &[]);
    assert!(!out.status.success());
    assert!(!out.stderr.is_empty(), "a failure is reported on stderr");
    assert_eq!(outputs(dir.path()), Vec::<String>::new());
}

#[test]
fn strict_fails_a_page_with_an_undecodable_image() {
    let (dir, input) = workdir(&broken_image_pdf());
    assert_success(&rrocket(dir.path(), &input, &["-r", "72"]));
    assert_eq!(outputs(dir.path()), ["out-1.ppm"]);

    let (dir, input) = workdir(&broken_image_pdf());
    let out = rrocket(dir.path(), &input, &["-r", "72", "--strict"]);
    assert!(!out.status.success(), "--strict accepts a failed image");
}

#[test]
fn failed_run_removes_its_ram_directory() {
    let (dir, input) = workdir(&broken_image_pdf());
    let ram = dir.path().join("ram");
    let out = Command::new(env!("CARGO_BIN_EXE_rrocket"))
        .args(["--backend", "cpu", "--strict", "-r", "72", "--ram-path"])
        .arg(&ram)
        .arg(&input)
        .arg("out")
        .output()
        .expect("run rrocket");
    assert!(!out.status.success(), "--strict fails the page");
    assert!(!ram.exists(), "the RAM directory outlived a failed run");
}

/// One 40 × 40 pt white page carrying a `/Square` annotation whose
/// appearance fills (10, 10)–(30, 30) black.
fn annotated_pdf() -> Vec<u8> {
    with_page_tree(&[
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 40 40] /Annots [4 0 R] >>".to_vec(),
        b"<< /Type /Annot /Subtype /Square /Rect [10 10 30 30] /AP << /N 5 0 R >> >>".to_vec(),
        stream_object(
            "/Type /XObject /Subtype /Form /BBox [0 0 20 20]",
            b"0 g 0 0 20 20 re f",
        ),
    ])
}

/// `(width, height, gray pixels)` that `rrocket -r 72 --gray <args>` writes
/// for page 1 of `pdf`.
fn gray_page(pdf: &[u8], args: &[&str]) -> (u32, u32, Vec<u8>) {
    let (dir, input) = workdir(pdf);
    let mut all = vec!["-r", "72", "--gray"];
    all.extend_from_slice(args);
    assert_success(&rrocket(dir.path(), &input, &all));
    let (_, w, h, data) = read_pnm(&dir.path().join("out-1.pgm"));
    (w, h, data)
}

#[test]
fn unequal_rx_and_ry_give_non_square_pixels() {
    // Page 1 is 100 × 50 pt: at 144 × 36 dpi it is 200 × 25 px, and the
    // black square at (10, 10) of side 20 pt covers columns 20..60 and,
    // counting rows from the top, 10..20.
    let (w, h, data) = gray_page(&two_page_pdf(), &["--rx", "144", "--ry", "36"]);
    assert_eq!((w, h), (200, 25));
    let at = |x: usize, y: usize| data[y * 200 + x];
    assert_eq!((at(40, 15), at(70, 15), at(40, 5)), (0, 255, 255));
}

#[test]
fn crop_flags_write_the_selected_area_of_the_page() {
    let pdf = two_page_pdf();
    let (w, _, full) = gray_page(&pdf, &[]);
    let (cw, ch, crop) = gray_page(&pdf, &["-x", "5", "-y", "12", "-W", "30", "-H", "25"]);
    assert_eq!((cw, ch), (30, 25));
    for row in 0..25 {
        let at = (12 + row) * w as usize + 5;
        assert_eq!(
            crop[row * 30..row * 30 + 30],
            full[at..at + 30],
            "row {row} of the crop is the page's row {}",
            12 + row
        );
    }
    // A width past the page edge is clipped to it.  Page 2 is only 60 pt
    // wide, so offset 90 lies outside it.
    let (cw, ch, _) = gray_page(&pdf, &["-l", "1", "-x", "90", "-W", "500"]);
    assert_eq!((cw, ch), (10, 50));
}

#[test]
fn scale_to_flags_size_the_page() {
    let pdf = two_page_pdf(); // page 1 is 100 × 50 pt
    let size = |args: &[&str]| {
        let (w, h, _) = gray_page(&pdf, args);
        (w, h)
    };
    assert_eq!(size(&["--scale-to", "300"]), (300, 150));
    assert_eq!(size(&["--scale-to-x", "40"]), (40, 20));
    assert_eq!(size(&["--scale-to-y", "40"]), (80, 40));
    assert_eq!(
        size(&["--scale-to-x", "60", "--scale-to-y", "90"]),
        (60, 90)
    );
}

/// One 120 × 60 pt page with a diagonal-edged fill and a line of text.
fn edges_pdf() -> Vec<u8> {
    with_page_tree(&[
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 120 60] \
           /Resources << /Font << /F 5 0 R >> >> /Contents 4 0 R >>"
            .to_vec(),
        stream_object(
            "",
            b"0 g 3.3 2.7 m 117.1 22.9 l 31.4 57.2 l h f \
              BT /F 13.7 Tf 5.3 40.1 Td (Edges) Tj ET",
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
    ])
}

/// Number of pixels that are neither black nor white.
fn grey_count(pixels: &[u8]) -> usize {
    pixels.iter().filter(|&&v| v != 0 && v != 255).count()
}

#[test]
fn antialias_flags_switch_edge_coverage() {
    let pdf = edges_pdf();
    let both = gray_page(&pdf, &[]);
    let no_text = gray_page(&pdf, &["--aa", "no"]);
    let no_vector = gray_page(&pdf, &["--aaVector", "no"]);
    let neither = gray_page(&pdf, &["--aa", "no", "--aaVector", "no"]);
    assert!(grey_count(&both.2) > 0, "antialiased edges are partial");
    assert_ne!(no_text.2, both.2, "--aa no changes the glyphs");
    assert_ne!(no_vector.2, both.2, "--aaVector no changes the fill");
    assert_ne!(no_text.2, no_vector.2, "the flags govern different content");
    assert_eq!(
        grey_count(&neither.2),
        0,
        "with both off every edge is hard"
    );
}

#[test]
fn hide_annotations_draws_the_page_content_alone() {
    let pdf = annotated_pdf();
    let shown = gray_page(&pdf, &[]);
    let hidden = gray_page(&pdf, &["--hide-annotations"]);
    assert_eq!(
        shown.2[20 * 40 + 20],
        0,
        "the annotation is drawn by default"
    );
    assert!(
        hidden.2.iter().all(|&v| v == 255),
        "--hide-annotations leaves the page blank"
    );
}
