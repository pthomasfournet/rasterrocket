//! Syntax damage confined to one token or one object degrades only what it
//! touches: the rest of the page still renders.

#[expect(
    dead_code,
    reason = "shared helpers; this test uses PDF assembly, sessions and ink coverage"
)]
mod common;

use common::{SQUARE, assemble_pdf, ink, one_page, session, stream_object};
use rasterrocket::{
    BackendPolicy, RasterError, RasterOptions, raster_pdf_from_bytes, render_page_rgb,
};

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

/// A `/Contents` element that cannot be decoded costs only its own content:
/// the streams before it still render.
#[test]
fn undecodable_content_element_keeps_the_rest_of_the_page() {
    // A stream under an unknown filter, and one whose ASCII85 data breaks
    // off at an illegal byte after `0 1 0 rg`.
    for (dict, data) in [
        ("/Filter /FooDecode", &b"hello"[..]),
        (
            "/Filter /ASCII85Decode",
            b"0H`(m0HbCK\x7f+>GT)+>GPm1b^%/0HbCI+D(r",
        ),
    ] {
        let pdf = one_page(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Contents [4 0 R 5 0 R] >>",
            vec![stream_object("", SQUARE), stream_object(dict, data)],
        );
        let ink = ink(pdf);
        assert!(
            (ink - 0.32).abs() < 0.01,
            "{dict}: square must render, ink {ink}"
        );
    }
}

/// A skipped `/Contents` element is reported in the page diagnostics, and a
/// strict render fails the page instead.
#[test]
fn undecodable_content_element_is_reported() {
    let pdf = one_page(
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Contents [4 0 R 5 0 R] >>",
        vec![
            stream_object("", SQUARE),
            stream_object("/Filter /FooDecode", b"hello"),
        ],
    );
    let render = |strict| {
        let mut opts = RasterOptions::default();
        opts.strict = strict;
        let mut pages: Vec<_> = raster_pdf_from_bytes(pdf.clone(), &opts).collect();
        assert_eq!(pages.len(), 1, "one page expected");
        pages.remove(0).1
    };

    let page = render(false).expect("page renders");
    let failures = &page.diagnostics.content_failures;
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert_eq!(failures[0].object, 5);
    assert!(
        failures[0].reason.contains("FooDecode"),
        "reason names the filter: {}",
        failures[0].reason
    );

    match render(true) {
        Err(RasterError::ContentDecodeFailed(msgs)) => {
            assert_eq!(msgs.len(), 1, "{msgs:?}");
            assert!(msgs[0].starts_with("content stream obj 5"), "{}", msgs[0]);
        }
        other => panic!("expected ContentDecodeFailed, got {other:?}"),
    }
}

/// A `/Contents` element whose data breaks off mid-stream draws what decoded
/// before the break and is reported as partial; a strict render fails the
/// page.
#[test]
fn truncated_content_element_is_drawn_and_reported_partial() {
    // ASCII85 of `0 g 10 10 80 80 re f `, then an illegal byte, then that
    // of `0 g 100 10 80 80 re f`: the second square is lost.
    let mut data = b"0Hb!N0ea_)0H`>/+?2%tEb-A*+9".to_vec();
    data.push(0x7f);
    data.extend_from_slice(b"0Hb!N0eb:(0ea_00H`>/+ED$9Ac");
    let pdf = one_page(
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Contents 4 0 R >>",
        vec![stream_object("/Filter /ASCII85Decode", &data)],
    );
    let render = |strict| {
        let mut opts = RasterOptions::default();
        opts.strict = strict;
        let mut pages: Vec<_> = raster_pdf_from_bytes(pdf.clone(), &opts).collect();
        assert_eq!(pages.len(), 1, "one page expected");
        pages.remove(0).1
    };

    let page = render(false).expect("page renders");
    let failures = &page.diagnostics.content_failures;
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert_eq!(failures[0].object, 4);
    assert!(failures[0].partial);
    assert!(
        failures[0].reason.starts_with("ASCII85Decode"),
        "{}",
        failures[0].reason
    );
    assert!(
        (ink(pdf.clone()) - 0.32).abs() < 0.01,
        "the decoded square renders"
    );

    match render(true) {
        Err(RasterError::ContentDecodeFailed(msgs)) => {
            assert_eq!(
                msgs,
                [format!(
                    "content stream obj 4 (partial): {}",
                    failures[0].reason
                )]
            );
        }
        other => panic!("expected ContentDecodeFailed, got {other:?}"),
    }
}

/// A page none of whose content streams decode fails rather than rendering
/// blank.
#[test]
fn page_with_no_decodable_content_fails() {
    let pdf = one_page(
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Contents [4 0 R 5 0 R] >>",
        vec![
            stream_object("/Filter /FooDecode", b"hello"),
            stream_object("/Filter /FooDecode", b"world"),
        ],
    );
    let s = session(pdf, BackendPolicy::CpuOnly);
    assert!(render_page_rgb(&s, 1, 1.0).is_err());
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
