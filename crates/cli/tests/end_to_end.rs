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
    reason = "shared helpers; this suite uses the assembly half"
)]
mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use common::{assemble_pdf, stream_object};
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
    assemble_pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
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

/// `(magic, width, height, pixel bytes)` of a binary PNM file.
fn read_pnm(path: &Path) -> (String, u32, u32, Vec<u8>) {
    let bytes = std::fs::read(path).expect("read PNM");
    let mut fields = Vec::new();
    let mut pos = 0;
    while fields.len() < 4 {
        while bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        let start = pos;
        while !bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        fields.push(String::from_utf8_lossy(&bytes[start..pos]).into_owned());
    }
    // One whitespace byte separates the maximum value from the raster.
    let data = bytes[pos + 1..].to_vec();
    let num = |s: &str| s.parse::<u32>().expect("PNM header number");
    assert_eq!(fields[3], "255", "maximum value");
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
