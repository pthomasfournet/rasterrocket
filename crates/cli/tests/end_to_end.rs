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

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use common::{assemble_pdf, stream_object, with_page_tree};
use rasterrocket::{
    BackendPolicy, FinishedCanvas, RenderParams, SampleDepth, SessionConfig,
    open_session_from_bytes, render_page_canvas, render_page_rgb, render_page_rgb_with,
};

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
    rrocket_with_env(dir, input, args, &[])
}

/// [`rrocket`] with the environment variables `env` set.
fn rrocket_with_env(dir: &Path, input: &Path, args: &[&str], env: &[(&str, &str)]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rrocket"))
        .envs(env.iter().copied())
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

/// Page 1 of [`two_page_pdf`] as the library renders it over a transparent
/// backdrop at 72 dpi: RGBA, alpha not premultiplied.
fn library_rgba() -> Vec<u8> {
    let session = open_session_from_bytes(
        two_page_pdf(),
        &SessionConfig::with_policy(BackendPolicy::CpuOnly),
    )
    .expect("open session");
    let params = RenderParams::new(1.0).with_transparent_background();
    let bmp = render_page_rgb_with(&session, 1, &params).expect("render page 1");
    let alpha = bmp.alpha_plane().expect("an alpha plane");
    bmp.data()
        .as_chunks::<3>()
        .0
        .iter()
        .zip(alpha)
        .flat_map(|(px, &a)| [px[0], px[1], px[2], a])
        .collect()
}

#[test]
fn transparent_png_and_tiff_carry_the_library_rgba() {
    let want = library_rgba();
    assert!(want.as_chunks::<4>().0.iter().any(|px| px[3] == 0));
    assert!(want.as_chunks::<4>().0.iter().any(|px| px[3] == 255));
    let (dir, input) = workdir(&two_page_pdf());
    assert_success(&rrocket(
        dir.path(),
        &input,
        &["-r", "72", "-l", "1", "--transparent", "--png"],
    ));
    let png = std::fs::read(dir.path().join("out-1.png")).expect("read PNG");
    // IHDR: bit depth 8, colour type 6 (RGBA), at bytes 24 and 25.
    assert_eq!(&png[24..26], &[8, 6], "an 8-bit RGBA PNG");
    for method in ["none", "lzw", "deflate"] {
        assert_success(&rrocket(
            dir.path(),
            &input,
            &[
                "-r",
                "72",
                "-l",
                "1",
                "--transparent",
                "--tiff",
                "--tiffcompression",
                method,
            ],
        ));
        let file = std::fs::read(dir.path().join("out-1.tif")).expect("read TIFF");
        let mut decoder = tiff::decoder::Decoder::new(std::io::Cursor::new(file)).expect("a TIFF");
        assert_eq!(
            decoder.colortype().expect("colour type"),
            tiff::ColorType::RGBA(8)
        );
        let tiff::decoder::DecodingResult::U8(rgba) = decoder.read_image().expect("decodes") else {
            panic!("8-bit samples");
        };
        assert_eq!(rgba, want, "{method}");
    }
}

/// `--transparent --gray` writes the library's gray and alpha page: a PNG
/// of colour type 4 and a two-sample TIFF of its samples.
#[test]
fn transparent_gray_png_and_tiff_carry_the_library_gray_alpha() {
    let session = open_session_from_bytes(
        two_page_pdf(),
        &SessionConfig::with_policy(BackendPolicy::CpuOnly),
    )
    .expect("open session");
    let mut opts = rasterrocket::RasterOptions::new(72.0);
    opts.transparent_background = true;
    opts.gray_alpha = true;
    let page = rasterrocket::render_page_gray_alpha(&session, 1, &opts).expect("render page 1");
    let want = page.pixels.as_u8().expect("8-bit samples").to_vec();
    let rgba = library_rgba();
    let alpha: Vec<u8> = want.as_chunks::<2>().0.iter().map(|px| px[1]).collect();
    let rgba_alpha: Vec<u8> = rgba.as_chunks::<4>().0.iter().map(|px| px[3]).collect();
    assert_eq!(alpha, rgba_alpha);
    let (dir, input) = workdir(&two_page_pdf());
    assert_success(&rrocket(
        dir.path(),
        &input,
        &["-r", "72", "-l", "1", "--transparent", "--gray", "--png"],
    ));
    let png = std::fs::read(dir.path().join("out-1.png")).expect("read PNG");
    // IHDR: bit depth 8, colour type 4 (gray and alpha), at bytes 24 and 25.
    assert_eq!(&png[24..26], &[8, 4], "an 8-bit gray and alpha PNG");
    assert_success(&rrocket(
        dir.path(),
        &input,
        &[
            "-r",
            "72",
            "-l",
            "1",
            "--transparent",
            "--gray",
            "--tiff",
            "--tiffcompression",
            "deflate",
        ],
    ));
    let file = std::fs::read(dir.path().join("out-1.tif")).expect("read TIFF");
    let mut decoder = tiff::decoder::Decoder::new(std::io::Cursor::new(file)).expect("a TIFF");
    assert_eq!(
        decoder.colortype().expect("colour type"),
        tiff::ColorType::Multiband {
            bit_depth: 8,
            num_samples: 2
        }
    );
    let tiff::decoder::DecodingResult::U8(pixels) = decoder.read_image().expect("decodes") else {
        panic!("8-bit samples");
    };
    assert_eq!(pixels, want);
}

#[test]
fn transparent_refuses_outputs_without_alpha() {
    for (flags, needle) in [
        (&["--transparent"][..], "PPM has none; add --png or --tiff"),
        (&["--transparent", "--jpeg"], "JPEG has none"),
        (&["--transparent", "--gray"], "PGM has none"),
        (&["--transparent", "--png", "--mono"], "drop --mono"),
    ] {
        let (dir, input) = workdir(&two_page_pdf());
        let out = rrocket(dir.path(), &input, flags);
        assert!(!out.status.success(), "{flags:?} succeeded");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains(needle), "{flags:?}: {stderr}");
        assert_eq!(outputs(dir.path()), Vec::<String>::new(), "{flags:?}");
    }
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
fn a_fax_page_longer_than_b4_is_sent_in_a4_pieces() {
    // 100 × 400 points: 6637 lines at fine resolution, past the 2808 of the
    // B4 length, so it goes as 297 mm pieces of 2291 lines and the rest.
    let pdf = assemble_pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 400] /Contents 4 0 R >>".to_vec(),
        stream_object("", b"0 g 10 10 20 20 re f"),
    ]);
    let (dir, input) = workdir(&pdf);
    assert_success(&rrocket(
        dir.path(),
        &input,
        &["--fax", "--tiffcompression", "g4"],
    ));
    let heights: Vec<u32> = fax_pages(&dir.path().join("out.tif"))
        .iter()
        .map(|&(w, h, _)| {
            assert_eq!(w, 1728);
            h
        })
        .collect();
    let total: u32 = heights.iter().sum();
    assert!(
        (f64::from(total) - 400.0 * 17.28 * 196.0 / 204.0).abs() <= 1.0,
        "{heights:?}"
    );
    assert_eq!(heights[..2], [2291, 2291], "{heights:?}");
    assert_eq!(heights.len(), 3, "{heights:?}");
}

#[test]
fn fax_writes_one_document_of_fax_width_pages() {
    // A portrait page with a small square, a landscape page, a dark page.
    let pdf = assemble_pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R 4 0 R 5 0 R] /Count 3 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 150] /Contents 6 0 R >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 150 100] /Contents 6 0 R >>".to_vec(),
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
    // point along: 150 points at 17.28 × 196/204 lines each.
    let lines = |pts: f64| pts * 17.28 * 196.0 / 204.0;
    for ((w, h, _), pts) in pages.iter().zip([150.0, 150.0, 150.0]) {
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
    let out = rrocket_with_env(
        dir.path(),
        &input,
        &["--fax-to", "+1 555 0100"],
        &[("PATH", &path)],
    );
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
            "+15550100",
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
        stderr.contains("not sent")
            && stderr.contains("sendfax refused")
            && stderr.contains("the document is at"),
        "{stderr}"
    );
}

#[test]
fn fax_with_a_bare_stem_writes_the_document_to_disk() {
    let (dir, input) = workdir(&two_page_pdf());
    let out = Command::new(env!("CARGO_BIN_EXE_rrocket"))
        .current_dir(dir.path())
        .args(["--backend", "cpu", "--fax"])
        .arg(&input)
        .arg("fax")
        .output()
        .expect("run rrocket");
    assert_success(&out);
    assert!(
        dir.path().join("fax.tif").is_file(),
        "the document outlives the run: {:?}",
        outputs(dir.path())
    );
}

/// `-` reads the PDF from standard input and renders it as the same file
/// named by path does; an empty standard input is refused.
#[test]
fn a_pdf_on_standard_input_renders_as_from_a_file() {
    use std::io::Write;
    let pdf = two_page_pdf();
    let (dir, input) = workdir(&pdf);
    assert_success(&rrocket(dir.path(), &input, &["-r", "72"]));
    let from_file = outputs(dir.path());

    let piped = tempfile::tempdir().expect("create temp dir");
    let mut child = Command::new(env!("CARGO_BIN_EXE_rrocket"))
        .current_dir(piped.path())
        .args(["--no-ram", "--backend", "cpu", "-r", "72", "-", "./out"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("run rrocket");
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(&pdf)
        .expect("write the PDF");
    let out = child.wait_with_output().expect("wait for rrocket");
    assert_success(&out);
    assert_eq!(outputs(piped.path()), from_file);
    for name in &from_file {
        assert_eq!(
            std::fs::read(piped.path().join(name)).unwrap(),
            std::fs::read(dir.path().join(name)).unwrap(),
            "{name}"
        );
    }

    let empty = Command::new(env!("CARGO_BIN_EXE_rrocket"))
        .current_dir(piped.path())
        .args(["--no-ram", "--backend", "cpu", "-", "./empty"])
        .stdin(std::process::Stdio::null())
        .output()
        .expect("run rrocket");
    assert!(!empty.status.success());
    assert!(
        String::from_utf8_lossy(&empty.stderr).contains("standard input is empty"),
        "{}",
        String::from_utf8_lossy(&empty.stderr)
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
fn box_flag_selects_the_rendered_page_box() {
    // MediaBox 100 × 50 pt; CropBox [20 10 80 40]; a black square at the
    // MediaBox origin, outside the CropBox.
    let pdf = with_page_tree(&[
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 50] /CropBox [20 10 80 40] \
           /Contents 4 0 R >>"
            .to_vec(),
        stream_object("", b"0 g 0 0 10 10 re f"),
    ]);
    let (w, h, media) = gray_page(&pdf, &["--box", "media"]);
    assert_eq!((w, h), (100, 50));
    let at = |x: usize, y: usize| media[y * 100 + x];
    assert_eq!((at(5, 45), at(50, 25)), (0, 255));

    for args in [
        &[][..],
        &["--cropbox"],
        &["--box", "crop"],
        &["--box", "trim"],
    ] {
        let (w, h, crop) = gray_page(&pdf, args);
        assert_eq!((w, h), (60, 30), "{args:?}");
        assert!(crop.iter().all(|&p| p == 255), "{args:?}");
    }
}

#[test]
fn rotate_flag_turns_the_page_before_it_is_sized_and_cropped() {
    // 100 × 50 pt; a black 10 pt square in the lower-left corner.
    let pdf = with_page_tree(&[
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 50] /Contents 4 0 R >>".to_vec(),
        stream_object("", b"0 g 0 0 10 10 re f"),
    ]);
    // Lower-left turns to upper-left at 90° clockwise, to lower-right at
    // 90° counter-clockwise, to upper-right at 180°.
    for (deg, size, black) in [
        ("90", (50, 100), (5, 5)),
        ("-90", (50, 100), (45, 95)),
        ("180", (100, 50), (95, 5)),
        ("360", (100, 50), (5, 45)),
    ] {
        let (w, h, px) = gray_page(&pdf, &["--rotate", deg]);
        assert_eq!((w, h), size, "--rotate {deg}");
        let (x, y) = black;
        assert_eq!(px[y * w as usize + x], 0, "--rotate {deg}");
        assert_eq!(
            px.iter().map(|&p| usize::from(p == 0)).sum::<usize>(),
            100,
            "--rotate {deg}"
        );
    }
    let (w, h, _) = gray_page(&pdf, &["--rotate", "90", "--scale-to-x", "25"]);
    assert_eq!((w, h), (25, 50));
    let (w, h, crop) = gray_page(&pdf, &["--rotate", "90", "-W", "10", "-H", "10"]);
    assert_eq!((w, h), (10, 10));
    assert!(
        crop.iter().all(|&p| p == 0),
        "the crop is the turned square"
    );
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

/// A `.cbt` (tar) archive holding the single file `name` of contents `data`.
fn tar_of(name: &str, data: &[u8]) -> Vec<u8> {
    let mut header = [0u8; 512];
    let mut put = |at: usize, field: &[u8]| header[at..at + field.len()].copy_from_slice(field);
    put(0, name.as_bytes());
    put(100, b"0000644\0");
    put(108, b"0000000\0");
    put(116, b"0000000\0");
    put(124, format!("{:011o}\0", data.len()).as_bytes());
    put(136, b"00000000000\0");
    put(148, b"        ");
    put(156, b"0");
    put(257, b"ustar\x0000");
    let sum: u32 = header.iter().map(|&b| u32::from(b)).sum();
    header[148..156].copy_from_slice(format!("{sum:06o}\0 ").as_bytes());
    let mut tar = header.to_vec();
    tar.extend_from_slice(data);
    tar.resize(tar.len().next_multiple_of(512) + 1024, 0);
    tar
}

/// A PDF inside a comic archive is rendered with the session flags of the
/// PDF path: `--hide-annotations` leaves the annotated page blank.
#[test]
fn comic_embedded_pdf_honours_session_flags() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let input = dir.path().join("in.cbt");
    std::fs::write(&input, tar_of("book.pdf", &annotated_pdf())).expect("write archive");
    let page = |args: &[&str]| {
        let mut all = vec!["-r", "72", "--gray"];
        all.extend_from_slice(args);
        assert_success(&rrocket(dir.path(), &input, &all));
        read_pnm(&dir.path().join("out-1.pgm")).3
    };
    assert_eq!(
        page(&[])[20 * 40 + 20],
        0,
        "the annotation is drawn by default"
    );
    assert!(
        page(&["--hide-annotations"]).iter().all(|&v| v == 255),
        "--hide-annotations leaves the page blank"
    );
}

/// A comic run reports each page under `--progress`, writes its pages in
/// order, and a page selection that selects nothing reads as on the PDF
/// path, not as a malformed archive.
#[test]
fn comic_pages_report_progress_and_selection_errors_name_no_archive() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let input = dir.path().join("in.cbt");
    std::fs::write(&input, tar_of("book.pdf", &two_page_pdf())).expect("write archive");

    let out = rrocket(dir.path(), &input, &["-r", "36", "--gray", "--progress"]);
    assert_success(&out);
    let stderr = String::from_utf8_lossy(&out.stderr);
    for page in ["page 1 done", "page 2 done", "[2/2]"] {
        assert!(stderr.contains(page), "no {page:?} in {stderr}");
    }
    assert!(dir.path().join("out-1.pgm").is_file());
    assert!(dir.path().join("out-2.pgm").is_file());

    let out = rrocket(dir.path(), &input, &["-f", "2", "-l", "1"]);
    assert!(!out.status.success(), "an empty selection fails");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("nothing to render"), "{stderr}");
    assert!(!stderr.contains("malformed"), "{stderr}");
}

/// A page rendered in bands writes the file it writes rendered whole, in
/// every banded format and colour model: the seam page at 600 dpi is
/// 1 667 × 833 px, four bands of 1 MiB.
#[test]
fn a_page_in_bands_writes_the_file_of_the_whole_page() {
    let (dir, input) = workdir(&common::seams::seam_page(""));
    for format in [
        &[][..],
        &["--png"],
        &["--tiff", "--tiffcompression", "deflate"],
        &["--gray"],
        &["--png", "--gray", "--invert=always"],
        &["--mono", "--dither", "halftone"],
        &["--tiff", "--mono"],
        &["--depth", "16"],
        &["--png", "--depth", "16", "--gray"],
        &["--tiff", "--depth", "16", "--invert=always"],
    ] {
        let mut files = Vec::new();
        for band_mib in ["1", "64"] {
            let out_dir = dir.path().join(format!("{}-{band_mib}", format.join("")));
            std::fs::create_dir(&out_dir).expect("create output dir");
            let mut args = vec!["-r", "600", "--band-mib", band_mib];
            args.extend_from_slice(format);
            assert_success(&rrocket(&out_dir, &input, &args));
            let names = outputs(&out_dir);
            assert_eq!(names.len(), 1, "{format:?}: {names:?}");
            files.push(std::fs::read(out_dir.join(&names[0])).expect("read output"));
        }
        assert!(
            files[0] == files[1],
            "{format:?}: bands of 1 MiB differ from one band"
        );
    }
}

/// The header fields of a binary PNM file: magic, width, height and, but
/// for a P4 bitmap, the maximum value.
fn pnm_header(path: &Path) -> Vec<String> {
    let bytes = std::fs::read(path).expect("read PNM");
    String::from_utf8_lossy(&bytes[..bytes.len().min(32)])
        .split_ascii_whitespace()
        .take(4)
        .map(str::to_owned)
        .collect()
}

/// `--depth 16` writes 16-bit files: a PNG of bit depth 16, a TIFF of 16
/// bits a sample carrying the library's 16-bit samples, PPM and PGM of
/// maximum value 65535.
#[test]
fn depth_16_writes_16_bit_files() {
    let (dir, input) = workdir(&two_page_pdf());
    let base = ["-r", "72", "-l", "1", "--depth", "16"];
    let run = |extra: &[&str]| {
        let mut args = base.to_vec();
        args.extend_from_slice(extra);
        assert_success(&rrocket(dir.path(), &input, &args));
    };
    for (extra, colour_type) in [
        (&["--png"][..], 2),
        (&["--png", "--gray"], 0),
        (&["--png", "--transparent"], 6),
    ] {
        run(extra);
        let png = std::fs::read(dir.path().join("out-1.png")).expect("read PNG");
        // IHDR: bit depth and colour type at bytes 24 and 25.
        assert_eq!(&png[24..26], &[16, colour_type], "{extra:?}");
    }
    run(&[]);
    assert_eq!(
        pnm_header(&dir.path().join("out-1.ppm")),
        ["P6", "100", "50", "65535"]
    );
    run(&["--gray"]);
    assert_eq!(
        pnm_header(&dir.path().join("out-1.pgm")),
        ["P5", "100", "50", "65535"]
    );

    let session = open_session_from_bytes(
        two_page_pdf(),
        &SessionConfig::with_policy(BackendPolicy::CpuOnly),
    )
    .expect("open session");
    let params = RenderParams::new(1.0).with_depth(SampleDepth::Sixteen);
    let FinishedCanvas::Rgb16(want) = render_page_canvas(&session, 1, &params).expect("renders")
    else {
        panic!("a 16-bit render");
    };
    for method in ["none", "deflate"] {
        run(&["--tiff", "--tiffcompression", method]);
        let file = std::fs::read(dir.path().join("out-1.tif")).expect("read TIFF");
        let mut decoder = tiff::decoder::Decoder::new(std::io::Cursor::new(file)).expect("a TIFF");
        assert_eq!(
            decoder.colortype().expect("colour type"),
            tiff::ColorType::RGB(16)
        );
        let tiff::decoder::DecodingResult::U16(rgb) = decoder.read_image().expect("decodes") else {
            panic!("16-bit samples");
        };
        assert_eq!(rgb, want.data(), "{method}");
    }
}

/// `--print --depth 16` hands `lp` a PDF 1.5 spool of 16-bit page images:
/// here a stand-in `lp` on `PATH` that keeps what it is given.
#[cfg(unix)]
#[test]
fn depth_16_prints_a_spool_of_16_bit_images() {
    use std::os::unix::fs::PermissionsExt as _;

    let (dir, input) = workdir(&two_page_pdf());
    let bin = dir.path().join("bin");
    std::fs::create_dir(&bin).expect("create bin dir");
    let script = bin.join("lp");
    std::fs::write(
        &script,
        format!("#!/bin/sh\ncat > \"{}/spool.pdf\"\n", dir.path().display()),
    )
    .expect("write lp");
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755))
        .expect("make lp executable");
    let path = format!(
        "{}:{}",
        bin.display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let out = Command::new(env!("CARGO_BIN_EXE_rrocket"))
        .env("PATH", &path)
        .args(["--backend", "cpu", "--print", "-r", "72", "--depth", "16"])
        .arg(&input)
        .output()
        .expect("run rrocket");
    assert_success(&out);
    let spool = std::fs::read(dir.path().join("spool.pdf")).expect("lp was given the spool");
    assert!(spool.starts_with(b"%PDF-1.5"), "the spool is PDF 1.5");
    let text = String::from_utf8_lossy(&spool);
    assert_eq!(
        text.matches("/BitsPerComponent 16").count(),
        2,
        "two 16-bit pages"
    );
}

/// The Windows printer that writes a PDF.
#[cfg(windows)]
const PRINT_TO_PDF: &str = "Microsoft Print to PDF";

/// Whether `--list-printers` lists Microsoft Print to PDF.  Its absence
/// skips the test, logged, unless `RR_REQUIRE_PRINT_TO_PDF=1` makes it a
/// failure.
#[cfg(windows)]
fn print_to_pdf(test: &str) -> bool {
    let out = Command::new(env!("CARGO_BIN_EXE_rrocket"))
        .arg("--list-printers")
        .output()
        .expect("run rrocket");
    let list = String::from_utf8_lossy(&out.stdout);
    if list
        .lines()
        .any(|l| l.split('\t').next() == Some(PRINT_TO_PDF))
    {
        eprintln!("{test}: ran");
        return true;
    }
    let reason = format!(
        "{PRINT_TO_PDF:?} is not listed: {list:?}, {:?}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        std::env::var_os("RR_REQUIRE_PRINT_TO_PDF").is_none_or(|v| v != "1"),
        "RR_REQUIRE_PRINT_TO_PDF=1: {reason}"
    );
    eprintln!("{test}: skipped: {reason}");
    false
}

/// `--print` on Windows: Microsoft Print to PDF, a printer that prints to
/// a file, is refused without `--print-file`, writing nothing, and with it
/// writes the pages as a PDF.
#[cfg(windows)]
#[test]
fn print_to_pdf_writes_the_pages_to_the_print_file() {
    if !print_to_pdf("print_to_pdf_writes_the_pages_to_the_print_file") {
        return;
    }
    let (dir, input) = workdir(&two_page_pdf());
    let print = |extra: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_rrocket"))
            .args(["--backend", "cpu", "--printer", PRINT_TO_PDF])
            .args(extra)
            .arg(&input)
            .output()
            .expect("run rrocket")
    };
    let refused = print(&[]);
    assert_eq!(refused.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&refused.stderr);
    assert!(stderr.contains("--print-file PATH"), "{stderr}");
    let file = dir.path().join("printed.pdf");
    assert_success(&print(&[
        "--print-file",
        file.to_str().unwrap(),
        "-r",
        "150",
    ]));
    let pdf = std::fs::read(&file).expect("the print file is written");
    let session = open_session_from_bytes(pdf, &SessionConfig::default()).expect("a PDF");
    assert_eq!(session.total_pages(), 2);
}

/// `--depth 16` is refused, at startup and writing nothing, with the flags
/// whose pages are 8-bit or bilevel; a comic archive renders at 16 bits.
#[test]
fn depth_16_refuses_8_bit_and_bilevel_outputs() {
    for (flags, needle) in [
        (&["--mono"][..], "use --gray"),
        (&["--dither", "halftone"], "--dither"),
        (&["--fax"], "--fax"),
        (&["--jpeg"], "use --png or --tiff"),
    ] {
        let (dir, input) = workdir(&two_page_pdf());
        let mut args = vec!["--depth", "16"];
        args.extend_from_slice(flags);
        let out = rrocket(dir.path(), &input, &args);
        assert!(!out.status.success(), "{flags:?} succeeded");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("--depth 16 is refused") && stderr.contains(needle),
            "{flags:?}: {stderr}"
        );
        assert_eq!(outputs(dir.path()), Vec::<String>::new(), "{flags:?}");
    }
    let dir = tempfile::tempdir().expect("create temp dir");
    let input = dir.path().join("in.cbt");
    std::fs::write(&input, tar_of("book.pdf", &annotated_pdf())).expect("write archive");
    assert_success(&rrocket(dir.path(), &input, &["--depth", "16", "--gray"]));
    assert_eq!(pnm_header(&dir.path().join("out-1.pgm"))[3], "65535");
}

/// The levels of the gradient page's axial ramp at each depth, read back
/// from PPM files: 2048 at 16 bits, at most 256 at 8.
#[test]
fn depth_16_keeps_the_levels_of_a_ramp_end_to_end() {
    let (dir, input) = workdir(&common::gradients::gradient_page());
    let levels = |depth: &str| {
        assert_success(&rrocket(
            dir.path(),
            &input,
            &["-r", "72", "--depth", depth],
        ));
        let bytes = std::fs::read(dir.path().join("out-1.ppm")).expect("read PPM");
        let header: usize = String::from_utf8_lossy(&bytes[..32])
            .split_ascii_whitespace()
            .take(4)
            .map(str::len)
            .sum::<usize>()
            + 4;
        let data = &bytes[header..];
        let width = common::gradients::WIDTH as usize;
        let y = common::gradients::BANDS[0].1 as usize;
        let sample = if depth == "16" { 2 } else { 1 };
        let row = &data[y * width * 3 * sample..][..width * 3 * sample];
        row.chunks_exact(3 * sample)
            .map(|px| &px[..sample])
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    };
    let (wide, narrow) = (levels("16"), levels("8"));
    assert!(
        wide >= 2000 && narrow <= 256,
        "{wide} levels at 16 bits, {narrow} at 8"
    );
}

/// `--invert=auto` at 16 bits measures the whole page and inverts a
/// reversed-out one, as at 8 bits.
#[test]
fn depth_16_inverts_a_reversed_out_page() {
    let pdf = with_page_tree(&[
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 50] /Contents 4 0 R >>".to_vec(),
        stream_object("", b"0 g 0 0 100 50 re f 1 g 10 10 20 20 re f"),
    ]);
    let (dir, input) = workdir(&pdf);
    for (flag, inside, outside) in [
        ("--invert=never", u16::MAX, 0),
        ("--invert=auto", 0, u16::MAX),
    ] {
        assert_success(&rrocket(
            dir.path(),
            &input,
            &["-r", "72", "--gray", "--depth", "16", flag],
        ));
        let bytes = std::fs::read(dir.path().join("out-1.pgm")).expect("read PGM");
        let data = &bytes[bytes.len() - 100 * 50 * 2..];
        let at = |i: usize| u16::from_be_bytes([data[2 * i], data[2 * i + 1]]);
        assert_eq!(at(25 * 100 + 15), inside, "{flag}: inside the square");
        assert_eq!(at(5 * 100 + 50), outside, "{flag}: the ground");
    }
}

/// The test profile `name` of `tests/fixtures/icc`.
fn icc_fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/icc")
        .join(name)
}

/// The samples of page 1 of [`two_page_pdf`] at 72 dpi converted to the
/// profile `name`, as the library renders them.
fn library_converted(name: &str) -> Vec<u8> {
    let mut config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    let bytes = std::fs::read(icc_fixture(name)).expect("read profile");
    config.output_profile = Some(std::sync::Arc::new(
        rasterrocket::OutputProfile::from_bytes(bytes).expect("an output profile"),
    ));
    let session = open_session_from_bytes(two_page_pdf(), &config).expect("open session");
    match render_page_canvas(&session, 1, &RenderParams::new(1.0)).expect("renders") {
        FinishedCanvas::Rgb8(b) => b.data().to_vec(),
        FinishedCanvas::Gray8(b) => b.data().to_vec(),
        other => panic!("an 8-bit canvas, not {other:?}"),
    }
}

/// A TIFF's ICC profile (tag 34675) and its 8-bit samples.
fn tiff_profile_and_samples(path: &Path) -> (Vec<u8>, Vec<u8>) {
    let file = std::fs::read(path).expect("read TIFF");
    let mut decoder = tiff::decoder::Decoder::new(std::io::Cursor::new(file)).expect("a TIFF");
    let icc = decoder
        .get_tag_u8_vec(tiff::tags::Tag::IccProfile)
        .expect("an ICC profile tag");
    let tiff::decoder::DecodingResult::U8(samples) = decoder.read_image().expect("decodes") else {
        panic!("8-bit samples");
    };
    (icc, samples)
}

#[test]
fn output_profile_converts_the_pages_and_tags_the_files() {
    let (dir, input) = workdir(&two_page_pdf());
    for (name, gray) in [("printer-rgb-v4.icc", false), ("printer-gray-v4.icc", true)] {
        let profile = icc_fixture(name);
        let profile_arg = profile.to_str().unwrap();
        let profile_bytes = std::fs::read(&profile).unwrap();
        let want = library_converted(name);
        let base = ["-r", "72", "-l", "1", "--output-profile", profile_arg];

        assert_success(&rrocket(
            dir.path(),
            &input,
            &[&base[..], &["--tiff"]].concat(),
        ));
        let (icc, samples) = tiff_profile_and_samples(&dir.path().join("out-1.tif"));
        assert_eq!(icc, profile_bytes, "{name}: the TIFF's profile");
        assert_eq!(samples, want, "{name}: the TIFF's samples");

        // --invert auto renders the page whole and writes it through the
        // whole-page encoders.
        assert_success(&rrocket(
            dir.path(),
            &input,
            &[&base[..], &["--png", "--invert=auto"]].concat(),
        ));
        let png = std::fs::read(dir.path().join("out-1.png")).unwrap();
        assert!(
            png.windows(4).any(|w| w == b"iCCP"),
            "{name}: the PNG's iCCP"
        );
        assert_eq!(
            png[25],
            if gray { 0 } else { 2 },
            "{name}: the PNG colour type"
        );

        assert_success(&rrocket(
            dir.path(),
            &input,
            &[&base[..], &["--jpeg"]].concat(),
        ));
        let jpeg = std::fs::read(dir.path().join("out-1.jpg")).unwrap();
        assert!(
            jpeg.windows(11).any(|w| w == b"ICC_PROFILE"),
            "{name}: the JPEG's APP2"
        );

        let out = rrocket(dir.path(), &input, &base);
        assert_success(&out);
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("cannot carry an ICC profile"),
            "{name}: Netpbm output warns"
        );
        let (magic, _, _, pixels) =
            read_pnm(
                &dir.path()
                    .join(if gray { "out-1.pgm" } else { "out-1.ppm" }),
            );
        assert_eq!(magic, if gray { "P5" } else { "P6" });
        assert_eq!(pixels, want, "{name}: the Netpbm samples");
    }
}

#[test]
fn output_profile_refusals_name_the_conflict() {
    let (dir, input) = workdir(&two_page_pdf());
    let rgb = icc_fixture("printer-rgb-v4.icc");
    let rgb = rgb.to_str().unwrap();
    let cmyk = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../jpeg2000/assets/CGATS001Compat-v2-micro.icc");
    let cmyk_printer = icc_fixture("printer-cmyk-v4.icc");
    for (args, message) in [
        (
            vec!["--intent", "perceptual"],
            "--intent applies to --output-profile and --proof only",
        ),
        (
            vec!["--bpc"],
            "--bpc applies to --output-profile and --proof only",
        ),
        (
            vec!["--output-profile", rgb, "--bpc", "--intent", "absolute"],
            "--bpc does not apply to --intent absolute",
        ),
        (
            vec!["--output-profile", rgb, "--mono"],
            "refused with --mono",
        ),
        (
            vec!["--output-profile", rgb, "--gray"],
            "is an RGB profile and --gray",
        ),
        (
            vec!["--output-profile", "/nonexistent.icc"],
            "cannot read the output profile",
        ),
        (
            vec!["--output-profile", cmyk.to_str().unwrap()],
            "input profile",
        ),
        (
            vec!["--output-profile", cmyk_printer.to_str().unwrap()],
            "add --cmyk with --tiff or --jpeg",
        ),
        (
            vec!["--output-profile", rgb, "--cmyk", "--tiff"],
            "give a CMYK output profile",
        ),
        (
            vec!["--working-space", "wide", "--depth", "16"],
            "--working-space wide carries its gamut to the conversion to the output \
             profile: add --output-profile",
        ),
        (
            vec!["--preserve-black"],
            "--preserve-black applies to a CMYK --output-profile and --proof only",
        ),
        (
            vec!["--output-profile", rgb, "--preserve-black=k-plane"],
            "--preserve-black applies to CMYK ink converted to a CMYK --output-profile's ink",
        ),
        (
            vec![
                "--output-profile",
                cmyk_printer.to_str().unwrap(),
                "--cmyk",
                "--tiff",
                "--intent",
                "absolute",
                "--preserve-black",
            ],
            "--preserve-black does not apply to --intent absolute",
        ),
    ] {
        let out = rrocket(dir.path(), &input, &args);
        assert!(!out.status.success(), "{args:?} succeeded");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains(message), "{args:?}: {stderr}");
        assert!(outputs(dir.path()).is_empty(), "{args:?} wrote files");
    }
}

/// `--transparent` under a gray output profile writes gray and alpha, the
/// alpha the library's RGBA page's.
#[test]
fn a_gray_output_profile_writes_transparent_pages_as_gray_and_alpha() {
    let (dir, input) = workdir(&two_page_pdf());
    let gray = icc_fixture("printer-gray-v4.icc");
    assert_success(&rrocket(
        dir.path(),
        &input,
        &[
            "-r",
            "72",
            "-l",
            "1",
            "--output-profile",
            gray.to_str().unwrap(),
            "--transparent",
            "--tiff",
        ],
    ));
    let file = std::fs::read(dir.path().join("out-1.tif")).expect("read TIFF");
    let mut decoder = tiff::decoder::Decoder::new(std::io::Cursor::new(file)).expect("a TIFF");
    let tiff::decoder::DecodingResult::U8(pixels) = decoder.read_image().expect("decodes") else {
        panic!("8-bit samples");
    };
    let alpha: Vec<u8> = pixels.as_chunks::<2>().0.iter().map(|px| px[1]).collect();
    let want: Vec<u8> = library_rgba()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|px| px[3])
        .collect();
    assert_eq!(alpha, want);
}

/// A 30 × 10 pt PDF/X page on the synthetic CMYK press: paper, sRGB red
/// and `DeviceCMYK` black, 10 pt each.
fn pdfx_swatch_pdf() -> Vec<u8> {
    let press = std::fs::read(icc_fixture("printer-cmyk-v4.icc")).expect("read profile");
    assemble_pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R /OutputIntents [<< /Type /OutputIntent /S /GTS_PDFX \
          /OutputConditionIdentifier (Test press) /DestOutputProfile 5 0 R >>] >>"
            .to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 30 10] /Contents 4 0 R >>".to_vec(),
        stream_object("", b"1 0 0 rg 10 0 10 10 re f 0 0 0 1 k 20 0 10 10 re f"),
        stream_object("/N 4", &press),
    ])
}

/// `--proof` shows the page on the press: the paper white, the red dulled
/// to the press's, black ink as the press's black; `--proof=absolute`
/// tints the paper.
#[test]
fn proof_shows_the_page_on_the_press() {
    let (dir, input) = workdir(&pdfx_swatch_pdf());
    let pixel = |args: &[&str], x: usize| {
        let out = rrocket(dir.path(), &input, args);
        assert!(
            out.status.success(),
            "{args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let (magic, width, height, pixels) = read_pnm(&dir.path().join("out-1.ppm"));
        assert_eq!((magic.as_str(), width, height), ("P6", 30, 10));
        let at = (5 * 30 + x) * 3;
        [pixels[at], pixels[at + 1], pixels[at + 2]]
    };
    let plain = ["-r", "72"];
    assert_eq!(pixel(&plain, 15), [255, 0, 0]);
    let relative = ["-r", "72", "--proof"];
    let paper = pixel(&relative, 5);
    assert!(paper.iter().all(|&c| c >= 252), "paper {paper:?}");
    let red = pixel(&relative, 15);
    assert!(red[0] < 245 && red[1] > 60, "red on the press {red:?}");
    let black = pixel(&relative, 25);
    assert!(
        black.iter().all(|&c| (18..=30).contains(&c)),
        "black ink {black:?}"
    );
    let tinted = pixel(&["-r", "72", "--proof=absolute"], 5);
    assert!(tinted[2] + 5 <= tinted[0], "paper tint {tinted:?}");
}

#[test]
fn proof_refusals_name_the_conflict() {
    let (dir, input) = workdir(&two_page_pdf());
    let cmyk_printer = icc_fixture("printer-cmyk-v4.icc");
    for (args, message) in [
        (vec!["--proof"], "carries no /OutputIntents"),
        (
            vec!["--proof", "--cmyk", "--tiff"],
            "--proof shows the press on a display",
        ),
        (
            vec!["--proof", "--transparent", "--png"],
            "drop --transparent",
        ),
        (vec!["--proof", "--mono"], "--proof is refused with --mono"),
        (
            vec![
                "--proof",
                "--output-profile",
                cmyk_printer.to_str().unwrap(),
            ],
            "is a CMYK profile and --proof",
        ),
        (vec!["--proof=perceptual"], "invalid value"),
    ] {
        let out = rrocket(dir.path(), &input, &args);
        assert!(!out.status.success(), "{args:?} succeeded");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains(message), "{args:?}: {stderr}");
        assert!(outputs(dir.path()).is_empty(), "{args:?} wrote files");
    }
}

/// A 60 × 10 pt page of six `DeviceCMYK` swatches, 10 pt each, and an RGB
/// red and a gray square over the last two.
fn cmyk_swatch_pdf() -> Vec<u8> {
    with_page_tree(&[
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 60 10] /Contents 4 0 R >>".to_vec(),
        stream_object(
            "",
            b"0.87 0.01 0 0 k 0 0 10 10 re f 1 0.46 0.05 0.18 k 10 0 10 10 re f \
              0 0 0 1 k 20 0 10 10 re f 0.5 0.5 0.5 0 k 30 0 10 10 re f \
              1 0 0 rg 40 0 10 10 re f 0.5 g 50 0 10 10 re f",
        ),
    ])
}

/// The ink of each swatch of [`cmyk_swatch_pdf`]: its `DeviceCMYK`
/// values, the red as magenta and yellow, the gray as black ink alone.
const SWATCH_INK: [[u8; 4]; 6] = [
    [222, 3, 0, 0],
    [255, 117, 13, 46],
    [0, 0, 0, 255],
    [128, 128, 128, 0],
    [0, 255, 255, 0],
    [0, 0, 0, 127],
];

/// A TIFF's photometric interpretation, ink set and samples, widened to
/// 16 bits.
fn cmyk_tiff(path: &Path) -> (u32, u32, Vec<u16>) {
    let file = std::fs::read(path).expect("read TIFF");
    let mut decoder = tiff::decoder::Decoder::new(std::io::Cursor::new(file)).expect("a TIFF");
    let photometric = decoder
        .get_tag_u32(tiff::tags::Tag::PhotometricInterpretation)
        .expect("a photometric interpretation");
    let ink_set = decoder
        .get_tag_u32(tiff::tags::Tag::Unknown(332))
        .expect("an ink set");
    let samples = match decoder.read_image().expect("decodes") {
        tiff::decoder::DecodingResult::U8(v) => v.into_iter().map(u16::from).collect(),
        tiff::decoder::DecodingResult::U16(v) => v,
        _ => panic!("8- or 16-bit samples"),
    };
    (photometric, ink_set, samples)
}

/// The ink at the centre of each swatch of a 60 × 10 pixel CMYK page.
fn swatch_centres(samples: &[u16]) -> Vec<[u16; 4]> {
    (0..6)
        .map(|i| {
            let at = (5 * 60 + i * 10 + 5) * 4;
            samples[at..at + 4].try_into().unwrap()
        })
        .collect()
}

#[test]
fn cmyk_tiff_keeps_device_cmyk_at_both_depths() {
    let (dir, input) = workdir(&cmyk_swatch_pdf());
    // At 16 bits each component rounds at 16 bits, and the gray's sRGB
    // value 32768 leaves 32767 of black.
    let wide = [
        [57_015, 655, 0, 0],
        [65_535, 30_146, 3_277, 11_796],
        [0, 0, 0, 65_535],
        [32_768, 32_768, 32_768, 0],
        [0, 65_535, 65_535, 0],
        [0, 0, 0, 32_767],
    ];
    let narrow = SWATCH_INK.map(|ink| ink.map(u16::from));
    for (depth, want) in [("8", narrow), ("16", wide)] {
        let out = rrocket(
            dir.path(),
            &input,
            &["--cmyk", "--tiff", "-r", "72", "--depth", depth],
        );
        assert_success(&out);
        let (photometric, ink_set, samples) = cmyk_tiff(&dir.path().join("out-1.tif"));
        assert_eq!((photometric, ink_set), (5, 1), "separated, CMYK");
        assert_eq!(swatch_centres(&samples), want, "depth {depth}");
    }
}

#[test]
fn jpegcmyk_writes_an_adobe_ycck_jpeg() {
    let (dir, input) = workdir(&cmyk_swatch_pdf());
    let out = rrocket(dir.path(), &input, &["--jpegcmyk", "-r", "72"]);
    assert_success(&out);
    let jpeg = std::fs::read(dir.path().join("out-1.jpg")).expect("a JPEG");
    let app14 = jpeg
        .windows(2)
        .position(|w| w == [0xFF, 0xEE])
        .expect("an Adobe APP14 marker");
    assert_eq!(&jpeg[app14 + 4..app14 + 9], b"Adobe");
    assert_eq!(jpeg[app14 + 15], 2, "colour transform YCCK");
    let sof = jpeg
        .windows(2)
        .position(|w| w == [0xFF, 0xC0])
        .expect("a baseline frame");
    assert_eq!(jpeg[sof + 9], 4, "four components");
    let same = rrocket(dir.path(), &input, &["--jpeg", "--cmyk", "-r", "72"]);
    assert_success(&same);
    assert_eq!(std::fs::read(dir.path().join("out-1.jpg")).unwrap(), jpeg);
}

#[test]
fn cmyk_output_profile_tags_the_tiff_and_keeps_device_cmyk() {
    let (dir, input) = workdir(&cmyk_swatch_pdf());
    let profile = icc_fixture("printer-cmyk-v4.icc");
    let out = rrocket(
        dir.path(),
        &input,
        &[
            "--cmyk",
            "--tiff",
            "-r",
            "72",
            "--output-profile",
            profile.to_str().unwrap(),
        ],
    );
    assert_success(&out);
    let path = dir.path().join("out-1.tif");
    let (icc, _) = tiff_profile_and_samples(&path);
    assert_eq!(
        icc,
        std::fs::read(&profile).unwrap(),
        "tagged with the profile"
    );
    let (_, _, samples) = cmyk_tiff(&path);
    let centres = swatch_centres(&samples);
    let device: Vec<[u16; 4]> = SWATCH_INK[..4]
        .iter()
        .map(|ink| ink.map(u16::from))
        .collect();
    assert_eq!(centres[..4], device[..], "DeviceCMYK passes through");
    let [c, m, y, k] = centres[5];
    assert!(
        k > 100 && c.max(m).max(y) < 12,
        "gray separates to black: {:?}",
        centres[5]
    );
}

#[test]
fn cmyk_refusals_name_the_conflict() {
    let (dir, input) = workdir(&cmyk_swatch_pdf());
    for (args, message) in [
        (vec!["--cmyk"], "add --tiff or --jpeg"),
        (vec!["--cmyk", "--png"], "PNG holds no CMYK"),
        (vec!["--jpegcmyk", "--mono"], "--mono"),
        (vec!["--cmyk", "--tiff", "--gray"], "--gray"),
        (vec!["--cmyk", "--tiff", "--transparent"], "opaque"),
        (vec!["--jpegcmyk", "--print"], "print spool"),
    ] {
        let out = rrocket(dir.path(), &input, &args);
        assert!(!out.status.success(), "{args:?} succeeded");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains(message), "{args:?}: {stderr}");
        assert!(outputs(dir.path()).is_empty(), "{args:?} wrote files");
    }
}

/// A light sheet with bars of text on a darker desk, 400 × 500 pt.
fn page_on_desk_pdf() -> Vec<u8> {
    let mut content = b"0.3 g 0 0 400 500 re f 0.95 g 50 60 300 380 re f 0.1 g".to_vec();
    for line in 0..12 {
        content.extend_from_slice(format!(" 90 {} 220 8 re", 100 + line * 25).as_bytes());
    }
    content.extend_from_slice(b" f");
    assemble_pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 500] /Contents 4 0 R >>".to_vec(),
        stream_object("", &content),
    ])
}

/// The size, colour type and samples of the TIFF at `path`.
fn read_tiff(path: &Path) -> (u32, u32, tiff::ColorType, tiff::decoder::DecodingResult) {
    let file = std::fs::read(path).expect("read TIFF");
    let mut decoder = tiff::decoder::Decoder::new(std::io::Cursor::new(file)).expect("a TIFF");
    let (w, h) = decoder.dimensions().expect("dimensions");
    let colour = decoder.colortype().expect("colour type");
    (w, h, colour, decoder.read_image().expect("decodes"))
}

/// The camera stages run on 16-bit pages and write them at 16 bits, cut as
/// at 8 bits; a stage that leaves the page as it is writes the file of the
/// page without it.
#[test]
fn camera_stages_clean_16_bit_pages() {
    let (dir, input) = workdir(&page_on_desk_pdf());
    let page = |args: &[&str]| {
        let mut all = vec!["-r", "72", "--tiff"];
        all.extend_from_slice(args);
        assert_success(&rrocket(dir.path(), &input, &all));
        read_tiff(&dir.path().join("out-1.tif"))
    };
    let (w8, h8, colour, _) = page(&["--crop"]);
    assert_eq!(colour, tiff::ColorType::RGB(8));
    assert!(w8 < 400 && h8 < 500, "the crop applies: {w8}×{h8}");
    let (w, h, colour, samples) = page(&["--crop", "--depth", "16"]);
    assert_eq!(colour, tiff::ColorType::RGB(16));
    assert!(
        w.abs_diff(w8) <= 2 && h.abs_diff(h8) <= 2,
        "{w}×{h} vs {w8}×{h8}"
    );
    let tiff::decoder::DecodingResult::U16(samples) = samples else {
        panic!("16-bit samples");
    };
    assert_eq!(samples.len(), (w * h * 3) as usize);
    let (.., colour, _) = page(&["--crop", "--depth", "16", "--gray"]);
    assert_eq!(colour, tiff::ColorType::Gray(16));

    let file = |args: &[&str]| {
        let mut all = vec!["-r", "72", "--png", "--depth", "16"];
        all.extend_from_slice(args);
        assert_success(&rrocket(dir.path(), &input, &all));
        std::fs::read(dir.path().join("out-1.png")).expect("read PNG")
    };
    // Dewarp needs five tracked text lines of curved text, and declines.
    assert!(
        file(&["--dewarp"]) == file(&[]),
        "an unchanged page is written as it is"
    );
}

/// A comic archive renders at 16 bits: a 16-bit image keeps its samples.
#[test]
fn comic_pages_render_at_16_bits() {
    use tiff::encoder::{TiffEncoder, colortype};
    let samples: Vec<u16> = (0..12u16).map(|i| i * 5000 + 7).collect();
    let mut image = std::io::Cursor::new(Vec::new());
    TiffEncoder::new(&mut image)
        .expect("encoder")
        .write_image::<colortype::Gray16>(4, 3, &samples)
        .expect("encode");
    let dir = tempfile::tempdir().expect("create temp dir");
    let input = dir.path().join("in.cbt");
    std::fs::write(&input, tar_of("1.tif", &image.into_inner())).expect("write archive");
    assert_success(&rrocket(
        dir.path(),
        &input,
        &["--depth", "16", "--tiff", "--gray"],
    ));
    let (w, h, colour, got) = read_tiff(&dir.path().join("out-1.tif"));
    assert_eq!((w, h, colour), (4, 3, tiff::ColorType::Gray(16)));
    let tiff::decoder::DecodingResult::U16(got) = got else {
        panic!("16-bit samples");
    };
    assert_eq!(got, samples);
}

/// The `(bit depth, colour type)` of a PNG's `IHDR` and the pixels per
/// metre of its `pHYs` chunk, if any.
fn png_header(png: &[u8]) -> ((u8, u8), Option<(u32, u32)>) {
    assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"), "a PNG");
    let phys = png.windows(4).position(|w| w == b"pHYs").map(|at| {
        let be = |i: usize| u32::from_be_bytes(png[i..i + 4].try_into().expect("4 bytes"));
        assert_eq!(png[at + 12], 1, "pHYs in metres");
        (be(at + 4), be(at + 8))
    });
    ((png[24], png[25]), phys)
}

/// `--mono --png` writes a PNG of bit depth 1, banded and whole (under a
/// camera stage, which renders the page whole) alike; every PNG and JPEG
/// records the render resolution.
#[test]
fn png_and_jpeg_record_the_resolution_and_mono_png_is_one_bit() {
    let (dir, input) = workdir(&two_page_pdf());
    let read = |name: &str| std::fs::read(dir.path().join(name)).expect("read output");
    // 150 dpi is 5906 pixels per metre, 75 dpi 2953.
    for extra in [&[][..], &["--crop"]] {
        let mut args = vec!["--rx", "150", "--ry", "75", "-l", "1", "--png", "--mono"];
        args.extend_from_slice(extra);
        assert_success(&rrocket(dir.path(), &input, &args));
        assert_eq!(
            png_header(&read("out-1.png")),
            ((1, 0), Some((5906, 2953))),
            "{extra:?}"
        );
    }
    for (extra, ihdr) in [(&["--gray"][..], (8, 0)), (&[], (8, 2))] {
        let mut args = vec!["-r", "150", "-l", "1", "--png"];
        args.extend_from_slice(extra);
        assert_success(&rrocket(dir.path(), &input, &args));
        assert_eq!(png_header(&read("out-1.png")), (ihdr, Some((5906, 5906))));
    }
    assert_success(&rrocket(
        dir.path(),
        &input,
        &["-r", "150", "-l", "1", "--jpeg"],
    ));
    let jpeg = read("out-1.jpg");
    let at = jpeg
        .windows(5)
        .position(|w| w == b"JFIF\0")
        .expect("a JFIF segment");
    // Units 1 (dots per inch), then the horizontal and vertical density.
    assert_eq!(&jpeg[at + 7..at + 12], &[1, 0, 150, 0, 150]);
}

/// A page of text-like bars tilted by 3°, 300 × 200 pt.
fn tilted_bars_pdf() -> Vec<u8> {
    let (sin, cos) = 3f64.to_radians().sin_cos();
    let words: String = (0..12)
        .flat_map(|line| {
            (0..6).map(move |word| {
                let (x, y) = (-125 + word * 42 + (line * 7) % 11, -66 + line * 12);
                format!("{x} {y} {} 5 re ", 24 + (line + word) % 9)
            })
        })
        .collect();
    let content = format!(
        "q {cos:.6} {sin:.6} {:.6} {cos:.6} 150 100 cm 0 g {words}f Q",
        -sin
    );
    with_page_tree(&[
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Contents 4 0 R >>".to_vec(),
        stream_object("", content.as_bytes()),
    ])
}

/// `--deskew` straightens a tilted gray page, leaves a page without a
/// confident angle as it is, and is refused with colour output as the
/// library refuses it.
#[test]
fn deskew_straightens_gray_pages_and_refuses_colour() {
    let (dir, input) = workdir(&tilted_bars_pdf());
    let page = |extra: &[&str]| {
        let mut args = vec!["-r", "150", "--gray"];
        args.extend_from_slice(extra);
        assert_success(&rrocket(dir.path(), &input, &args));
        read_pnm(&dir.path().join("out-1.pgm"))
    };
    let (plain, straight) = (page(&[]), page(&["--deskew"]));
    assert_eq!(
        (straight.1, straight.2),
        (plain.1, plain.2),
        "the size is kept"
    );
    assert_ne!(straight.3, plain.3, "the tilted page is rotated");

    let (dir, input) = workdir(&two_page_pdf());
    let blank = |extra: &[&str]| {
        let mut args = vec!["-r", "72", "-l", "1", "--gray"];
        args.extend_from_slice(extra);
        assert_success(&rrocket(dir.path(), &input, &args));
        read_pnm(&dir.path().join("out-1.pgm")).3
    };
    assert_eq!(
        blank(&["--deskew"]),
        blank(&[]),
        "a page without lines is kept"
    );

    for (flags, needle) in [
        (&["--deskew"][..], "--deskew applies to gray pages only"),
        (
            &["--deskew", "--png"],
            "--deskew applies to gray pages only",
        ),
        (
            &["--deskew", "--cmyk", "--tiff"],
            "--cmyk is refused with --deskew",
        ),
        (&["--deskew", "--transparent", "--png"], "drop --deskew"),
    ] {
        let (dir, input) = workdir(&two_page_pdf());
        let out = rrocket(dir.path(), &input, flags);
        assert!(!out.status.success(), "{flags:?} succeeded");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains(needle), "{flags:?}: {stderr}");
        assert_eq!(outputs(dir.path()), Vec::<String>::new(), "{flags:?}");
    }
    let (dir, input) = workdir(&two_page_pdf());
    for flags in [
        &["--deskew", "--mono", "--png"][..],
        &["--deskew", "--gray", "--depth", "16", "--png"],
    ] {
        let mut args = vec!["-r", "72", "-l", "1"];
        args.extend_from_slice(flags);
        assert_success(&rrocket(dir.path(), &input, &args));
    }
}

/// A born-digital letter page: lines of Helvetica text on white paper.
fn letter_text_pdf() -> Vec<u8> {
    let mut lines = String::new();
    for line in 0..30 {
        let _ = write!(
            lines,
            "BT /F1 11 Tf 72 {} Td (Line {line}: the quick brown fox jumps over the lazy dog.) Tj ET ",
            720 - line * 20
        );
    }
    with_page_tree(&[
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
           /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
            .to_vec(),
        stream_object("", lines.as_bytes()),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
    ])
}

/// A page as a scanner delivers it, 300 × 400 pt: paper lit from the left,
/// its gray falling from 0.92 to 0.45 across the page, bars of dark text,
/// and a strip of dark scanner bed 12 pt wide along the left edge.
fn uneven_scan_pdf() -> Vec<u8> {
    let mut paper = String::new();
    for i in 0..60 {
        let gray = 0.47f64.mul_add(-f64::from(i) / 59.0, 0.92);
        let _ = write!(paper, "{gray:.4} g {} 0 5 400 re f ", i * 5);
    }
    let text: String = (0..14)
        .flat_map(|line| {
            (0..5).map(move |word| format!("{} {} 38 7 re ", 40 + word * 48, 330 - line * 20))
        })
        .collect();
    let content = format!("{paper}0.05 g {text}f 0.08 g 0 0 12 400 re f");
    with_page_tree(&[
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 400] /Contents 4 0 R >>".to_vec(),
        stream_object("", content.as_bytes()),
    ])
}

/// `--clean-borders` and `--clean-background` leave a born-digital page
/// byte-identical, in gray, colour and bilevel output.
#[test]
fn ocr_prep_leaves_born_digital_pages_as_they_are() {
    let (dir, input) = workdir(&letter_text_pdf());
    let file = |extra: &[&str], name: &str| {
        let mut args = vec!["-r", "150"];
        args.extend_from_slice(extra);
        assert_success(&rrocket(dir.path(), &input, &args));
        std::fs::read(dir.path().join(name)).expect("read output")
    };
    for (output, name) in [
        (&["--gray"][..], "out-1.pgm"),
        (&[], "out-1.ppm"),
        (&["--mono"], "out-1.pbm"),
    ] {
        let plain = file(output, name);
        for flags in [
            &["--clean-borders"][..],
            &["--clean-background"],
            &["--clean-borders", "--clean-background"],
        ] {
            let with = file(&[output, flags].concat(), name);
            assert!(with == plain, "{output:?} {flags:?} changed the page");
        }
    }
}

/// On an unevenly lit scan, `--clean-borders` whitens the scanner bed,
/// `--clean-background` whitens the dim paper, and `--threshold sauvola`
/// keeps dim paper white where the fixed threshold blackens it.
#[test]
fn ocr_prep_cleans_an_uneven_scan() {
    let (dir, input) = workdir(&uneven_scan_pdf());
    let page = |extra: &[&str]| {
        let mut args = vec!["-r", "150"];
        args.extend_from_slice(extra);
        assert_success(&rrocket(dir.path(), &input, &args));
        let name = if extra.contains(&"--mono") {
            "out-1.pbm"
        } else {
            "out-1.pgm"
        };
        read_pnm(&dir.path().join(name))
    };
    let at = |p: &(String, u32, u32, Vec<u8>), x: u32, y: u32| p.3[(y * p.1 + x) as usize];
    let plain = page(&["--gray"]);
    // Pixel (10, 10) is scanner bed; (590, 10) is the dimmest paper.
    assert!(at(&plain, 10, 10) < 64 && at(&plain, 590, 10) < 128);
    let borders = page(&["--gray", "--clean-borders"]);
    assert_eq!(at(&borders, 10, 10), 255);
    assert_eq!(at(&borders, 590, 10), at(&plain, 590, 10));
    let both = page(&["--gray", "--clean-borders", "--clean-background"]);
    assert_eq!(at(&both, 590, 10), 255);

    let is_black = |p: &(String, u32, u32, Vec<u8>), x: u32, y: u32| {
        p.3[(y * p.1.div_ceil(8) + x / 8) as usize] & (0x80 >> (x % 8)) != 0
    };
    let (fixed, sauvola) = (
        page(&["--mono"]),
        page(&["--mono", "--threshold", "sauvola"]),
    );
    assert!(is_black(&fixed, 590, 10) && !is_black(&sauvola, 590, 10));
    // Text stays black: (100, 133) lies in the first line's first word.
    assert!(is_black(&fixed, 100, 133) && is_black(&sauvola, 100, 133));
}

/// The OCR-preparation flags are refused where they cannot apply.
#[test]
fn ocr_prep_flags_are_refused_where_they_cannot_apply() {
    for (flags, needle) in [
        (&["--threshold", "sauvola"][..], "bilevel output only"),
        (&["--threshold", "sauvola", "--gray"], "add --mono or --fax"),
        (
            &["--threshold", "sauvola", "--mono", "--dither", "halftone"],
            "drop one",
        ),
        (
            &["--clean-borders", "--cmyk", "--tiff"],
            "--cmyk is refused with --clean-borders",
        ),
        (
            &["--clean-background", "--cmyk", "--tiff"],
            "--cmyk is refused with --clean-background",
        ),
        (
            &["--clean-borders", "--transparent", "--png"],
            "drop --clean-borders",
        ),
        (
            &["--clean-background", "--transparent", "--png"],
            "drop --clean-background",
        ),
    ] {
        let (dir, input) = workdir(&two_page_pdf());
        let out = rrocket(dir.path(), &input, flags);
        assert!(!out.status.success(), "{flags:?} succeeded");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains(needle), "{flags:?}: {stderr}");
        assert_eq!(outputs(dir.path()), Vec::<String>::new(), "{flags:?}");
    }
    let (dir, input) = workdir(&two_page_pdf());
    for flags in [
        &["--threshold", "sauvola", "--fax"][..],
        &[
            "--clean-background",
            "--clean-borders",
            "--gray",
            "--depth",
            "16",
            "--png",
        ],
        &["--clean-background", "--clean-borders", "--fax"],
    ] {
        let mut args = vec!["-l", "1"];
        args.extend_from_slice(flags);
        assert_success(&rrocket(dir.path(), &input, &args));
    }
}

/// A 20 × 10 pt page in `[/ICCBased <printer-cmyk-v4>]`: black ink alone
/// at 80 %, then a blue of the four inks.
fn icc_cmyk_black_pdf() -> Vec<u8> {
    let press = std::fs::read(icc_fixture("printer-cmyk-v4.icc")).expect("read profile");
    assemble_pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 20 10] /Contents 4 0 R \
          /Resources << /ColorSpace << /P [/ICCBased 5 0 R] >> >> >>"
            .to_vec(),
        stream_object(
            "",
            b"/P cs 0 0 0 0.8 sc 0 0 10 10 re f 0.8 0.5 0 0.5 sc 10 0 10 10 re f",
        ),
        stream_object("/N 4", &press),
    ])
}

/// Converted to another press, black ink alone becomes four-colour black;
/// `--preserve-black` keeps it black ink alone and the blue as it was, and
/// `=k-plane` also gives the blue the K of the K → K curve, not the
/// four-colour conversion's.
#[test]
fn preserve_black_keeps_black_ink_alone() {
    let (dir, input) = workdir(&icc_cmyk_black_pdf());
    let press = icc_fixture("printer-cmyk-v2.icc");
    let ink = |extra: &[&str]| {
        let mut args = vec!["--cmyk", "--tiff", "-r", "72", "--output-profile"];
        args.push(press.to_str().unwrap());
        args.extend_from_slice(extra);
        let out = rrocket(dir.path(), &input, &args);
        assert_success(&out);
        let (_, _, samples) = cmyk_tiff(&dir.path().join("out-1.tif"));
        let at = |x: usize| -> [u16; 4] { samples[(5 * 20 + x) * 4..][..4].try_into().unwrap() };
        (at(5), at(15))
    };
    let (black, blue) = ink(&[]);
    assert!(
        black[..3].iter().all(|&v| v > 0),
        "four-colour black {black:?}"
    );
    let (only, only_blue) = ink(&["--preserve-black"]);
    assert_eq!(only[..3], [0, 0, 0], "black ink alone {only:?}");
    assert!(only[3] > 0);
    assert_eq!(only_blue, blue, "k-only leaves other colours");
    let (plane, plane_blue) = ink(&["--preserve-black=k-plane"]);
    assert_eq!(plane, only);
    assert!(
        plane_blue[3] != blue[3],
        "k-plane {plane_blue:?} against {blue:?}"
    );
}
