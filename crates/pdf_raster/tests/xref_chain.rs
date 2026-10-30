//! Incremental updates chain cross-reference sections through `/Prev`
//! (ISO 32000-1 §7.5.6), and a hybrid-reference file adds an `/XRefStm`
//! stream that a reader consults after the section's own entries and before
//! `/Prev` (§7.5.8.4).  The newest section that mentions an object decides
//! it, whether it defines the object or frees it; an entry that leads to a
//! different object's header leads to nothing.

#[expect(
    dead_code,
    reason = "shared helpers; this test uses PDF assembly and sessions"
)]
mod common;

use std::fmt::Write as _;

use common::{session, stream_object};
use rasterrocket::{BackendPolicy, render_page_rgb};

/// Catalog 1, Pages 2, Page 3 (200 × 100 pt) drawing `/Contents 7 0 R`.
fn page_objects() -> Vec<(u32, Vec<u8>)> {
    vec![
        (1, b"<< /Type /Catalog /Pages 2 0 R >>".to_vec()),
        (2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec()),
        (
            3,
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Contents 7 0 R >>".to_vec(),
        ),
    ]
}

/// A content stream filling `[10 90] × [10 90]` with `rgb`.
fn rect(rgb: &str) -> Vec<u8> {
    stream_object("", format!("{rgb} rg 10 10 80 80 re f").as_bytes())
}

/// Write `objects` and return their header offsets.
fn write_objects(pdf: &mut Vec<u8>, objects: &[(u32, Vec<u8>)]) -> Vec<(u32, usize)> {
    objects
        .iter()
        .map(|(num, body)| {
            let off = pdf.len();
            pdf.extend_from_slice(format!("{num} 0 obj\n").as_bytes());
            pdf.extend_from_slice(body);
            pdf.extend_from_slice(b"\nendobj\n");
            (*num, off)
        })
        .collect()
}

/// Append a classic xref section with one subsection per entry, a trailer
/// with `extra` entries, and `startxref`.  Returns the section offset.
fn write_xref(pdf: &mut Vec<u8>, entries: &[(u32, usize, u16, char)], extra: &str) -> usize {
    let at = pdf.len();
    let mut s = String::from("xref\n");
    for (num, off, generation, flag) in entries {
        let _ = write!(s, "{num} 1\n{off:010} {generation:05} {flag} \n");
    }
    let _ = write!(
        s,
        "trailer\n<< /Size 10 /Root 1 0 R {extra} >>\nstartxref\n{at}\n%%EOF\n"
    );
    pdf.extend_from_slice(s.as_bytes());
    at
}

/// The first section defines the page with a blue content stream; an
/// update redefines the content stream as green through an `/XRefStm`
/// cross-reference stream, while the update's `/Prev` still names the blue
/// one.
fn hybrid_pdf() -> Vec<u8> {
    let mut pdf = b"%PDF-1.5\n%\xE2\xE3\xCF\xD3\n".to_vec();
    let mut objects = page_objects();
    objects.push((7, rect("0 0 1")));
    let offsets = write_objects(&mut pdf, &objects);
    let entries: Vec<_> = offsets.iter().map(|&(n, o)| (n, o, 0, 'n')).collect();
    let prev = write_xref(&mut pdf, &entries, "");

    let green = write_objects(&mut pdf, &[(7, rect("0 1 0"))])[0].1;
    let mut row = vec![1u8];
    row.extend_from_slice(&u32::try_from(green).expect("offset").to_be_bytes());
    row.push(0);
    let stm = write_objects(
        &mut pdf,
        &[(
            9,
            stream_object("/Type /XRef /Size 10 /W [1 4 1] /Index [7 1]", &row),
        )],
    )[0]
    .1;
    write_xref(
        &mut pdf,
        &[(9, stm, 0, 'n')],
        &format!("/Prev {prev} /XRefStm {stm}"),
    );
    pdf
}

/// The first section defines the page with a blue content stream; an
/// update frees the content stream's number while the page still refers
/// to it, and the body is still in the file.
fn freed_contents_pdf() -> Vec<u8> {
    let mut pdf = b"%PDF-1.5\n%\xE2\xE3\xCF\xD3\n".to_vec();
    let mut objects = page_objects();
    objects.push((7, rect("0 0 1")));
    let offsets = write_objects(&mut pdf, &objects);
    let entries: Vec<_> = offsets.iter().map(|&(n, o)| (n, o, 0, 'n')).collect();
    let prev = write_xref(&mut pdf, &entries, "");
    write_xref(&mut pdf, &[(7, 0, 1, 'f')], &format!("/Prev {prev}"));
    pdf
}

/// The page's `/Contents 7 0 R` has an xref entry at the header of object
/// 8, a red content stream, and no object 7 exists anywhere in the file.
fn entry_to_another_object_pdf() -> Vec<u8> {
    let mut pdf = b"%PDF-1.5\n%\xE2\xE3\xCF\xD3\n".to_vec();
    let mut objects = page_objects();
    objects.push((8, rect("1 0 0")));
    let offsets = write_objects(&mut pdf, &objects);
    let mut entries: Vec<_> = offsets.iter().map(|&(n, o)| (n, o, 0, 'n')).collect();
    let eight = offsets.iter().find(|(n, _)| *n == 8).expect("object 8").1;
    entries.push((7, eight, 0, 'n'));
    write_xref(&mut pdf, &entries, "");
    pdf
}

/// The colour at (50, 50) of the page rendered at one pixel per point.
fn centre_colour(pdf: Vec<u8>) -> [u8; 3] {
    let s = session(pdf, BackendPolicy::CpuOnly);
    let bmp = render_page_rgb(&s, 1, 1.0).expect("render page");
    let at = 50 * bmp.stride + 50 * 3;
    let px = &bmp.data()[at..at + 3];
    [px[0], px[1], px[2]]
}

/// The `/XRefStm` entry of the newest section shadows the `/Prev` chain.
#[test]
fn hybrid_xrefstm_entry_shadows_prev_section() {
    assert_eq!(centre_colour(hybrid_pdf()), [0, 255, 0]);
}

/// A freed object the page still uses is drawn from its body in the file.
#[test]
fn freed_but_referenced_contents_still_render() {
    assert_eq!(centre_colour(freed_contents_pdf()), [0, 0, 255]);
}

/// An entry that leads to another object's header does not draw that
/// object as the page's contents: the page is blank.
#[test]
fn entry_to_another_object_draws_nothing() {
    assert_eq!(
        centre_colour(entry_to_another_object_pdf()),
        [255, 255, 255]
    );
}
