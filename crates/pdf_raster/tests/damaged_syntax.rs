//! Syntax damage confined to one token or one object degrades only what it
//! touches: the rest of the page still renders.

#[expect(
    dead_code,
    reason = "shared helpers; this test uses PDF assembly and ink coverage"
)]
mod common;

use common::{SQUARE, assemble_pdf, ink, one_page, stream_object};

/// A bare keyword as a page-dictionary value loses that key only; the page
/// and its content are still read.
#[test]
fn bare_keyword_value_drops_only_its_key() {
    let pdf = one_page(
        "<< /Type /Page /Parent 2 0 R /Foo bar /MediaBox [0 0 200 100] /Contents 4 0 R >>",
        vec![stream_object("", SQUARE)],
    );
    let ink = ink(pdf);
    assert!((ink - 0.32).abs() < 0.01, "square must render, ink {ink}");
}

/// A malformed number in a content stream is an operand with the value of
/// its longest well-formed prefix (`80-` is 80), so the operator after it
/// still receives its operands.
#[test]
fn malformed_content_number_keeps_its_operator() {
    let pdf = one_page(
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Contents 4 0 R >>",
        vec![stream_object("", b"0 g 10 10 80 80- re f")],
    );
    let ink = ink(pdf);
    assert!((ink - 0.32).abs() < 0.01, "square must render, ink {ink}");
}

/// A `/Pages` node whose `/Count` equals its number of kids may still have
/// an interior node among them: a root `/Count 1 /Kids [4 0 R]` whose kid
/// is `/Pages /Count 1 /Kids [3 0 R]` reaches page 3, not node 4.
#[test]
fn single_child_interior_node_is_descended() {
    let pdf = assemble_pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 4 0 R /MediaBox [0 0 200 100] /Contents 5 0 R >>".to_vec(),
        b"<< /Type /Pages /Parent 2 0 R /Kids [3 0 R] /Count 1 >>".to_vec(),
        stream_object("", SQUARE),
    ]);
    let ink = ink(pdf);
    assert!((ink - 0.32).abs() < 0.01, "square must render, ink {ink}");
}
