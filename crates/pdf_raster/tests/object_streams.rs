//! Objects packed into a `/Type /ObjStm` stream (ISO 32000-1 §7.5.7) are
//! reached through type-2 entries of a cross-reference stream.  Each object
//! stands on its own: one that does not parse is missing without taking
//! its neighbours with it.

#[expect(
    dead_code,
    reason = "shared helpers; this test uses PDF assembly and sessions"
)]
mod common;

use std::fmt::Write as _;

use common::{session, stream_object};
use rasterrocket::{BackendPolicy, render_page_rgb};

/// A PDF-1.5 file whose Catalog (1), Pages (2) and Page (3) live in object
/// stream 5 together with `extra` as object 8, with the page's blue
/// content stream as object 4 and a cross-reference stream as object 6.
fn objstm_pdf(extra: &str) -> Vec<u8> {
    objstm_pdf_with(extra, None, false)
}

/// [`objstm_pdf`] with `startxref` pointing at `startxref` instead of the
/// cross-reference stream, and with the object stream's own entry placing
/// it at index 0 of itself when `self_nested`.
fn objstm_pdf_with(extra: &str, startxref: Option<usize>, self_nested: bool) -> Vec<u8> {
    let bodies: [(u32, &str); 4] = [
        (1, "<< /Type /Catalog /Pages 2 0 R >>"),
        (2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        (
            3,
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Contents 4 0 R >>",
        ),
        (8, extra),
    ];
    let mut directory = String::new();
    let mut data = String::new();
    for (num, body) in bodies {
        write!(directory, "{num} {} ", data.len()).expect("writing to a String");
        data.push_str(body);
        data.push('\n');
    }
    let first = directory.len();
    let objstm = stream_object(
        &format!("/Type /ObjStm /N 4 /First {first}"),
        format!("{directory}{data}").as_bytes(),
    );
    let contents = stream_object("", b"0 0 1 rg 10 10 80 80 re f");

    let mut pdf = b"%PDF-1.5\n%\xE2\xE3\xCF\xD3\n".to_vec();
    let mut offsets = [0usize; 7];
    for (num, body) in [(5, &objstm), (4, &contents)] {
        offsets[num] = pdf.len();
        pdf.extend_from_slice(format!("{num} 0 obj\n").as_bytes());
        pdf.extend_from_slice(body);
        pdf.extend_from_slice(b"\nendobj\n");
    }
    offsets[6] = pdf.len();
    // /W [1 4 1]: type, field 2 as four bytes, field 3 as one byte.
    let mut rows = Vec::new();
    let mut row = |kind: u8, f2: usize, f3: u8| {
        rows.push(kind);
        rows.extend_from_slice(&u32::try_from(f2).expect("field").to_be_bytes());
        rows.push(f3);
    };
    row(0, 0, 255);
    row(2, 5, 0);
    row(2, 5, 1);
    row(2, 5, 2);
    row(1, offsets[4], 0);
    if self_nested {
        row(2, 5, 0);
    } else {
        row(1, offsets[5], 0);
    }
    row(1, offsets[6], 0);
    row(0, 0, 0);
    row(2, 5, 3);
    pdf.extend_from_slice(b"6 0 obj\n");
    pdf.extend_from_slice(&stream_object(
        "/Type /XRef /Size 9 /Root 1 0 R /W [1 4 1]",
        &rows,
    ));
    pdf.extend_from_slice(
        format!(
            "\nendobj\nstartxref\n{}\n%%EOF\n",
            startxref.unwrap_or(offsets[6])
        )
        .as_bytes(),
    );
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

/// A well-formed object stream resolves the page tree.
#[test]
fn page_tree_in_object_stream_renders() {
    assert_eq!(centre_colour(objstm_pdf("<< /Unused true >>")), [0, 0, 255]);
}

/// An unparseable object in the same object stream as the page tree does
/// not lose the document.
#[test]
fn unparseable_neighbour_in_object_stream_keeps_the_page() {
    assert_eq!(centre_colour(objstm_pdf(")))garbage(((")), [0, 0, 255]);
}

/// With `startxref` pointing past the end of the file, the page tree is
/// recovered from the object stream and the root from the
/// cross-reference stream's dictionary.
#[test]
fn page_tree_in_object_stream_is_reconstructed_without_startxref() {
    assert_eq!(
        centre_colour(objstm_pdf_with("<< /Unused true >>", Some(999_999), false)),
        [0, 0, 255]
    );
}

/// An object stream whose own cross-reference entry places it inside
/// itself is read from its header in the file rather than through that
/// entry, so the page tree it holds still resolves.
#[test]
fn self_nested_object_stream_entry_keeps_the_page() {
    assert_eq!(
        centre_colour(objstm_pdf_with("<< /Unused true >>", None, true)),
        [0, 0, 255]
    );
}
