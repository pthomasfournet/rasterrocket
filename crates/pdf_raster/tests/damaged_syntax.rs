//! Syntax damage confined to one token or one object degrades only what it
//! touches: the rest of the page still renders.

#[expect(
    dead_code,
    reason = "shared helpers; this test uses PDF assembly and sessions"
)]
mod common;

use common::{assemble_pdf, session, stream_object};
use rasterrocket::{BackendPolicy, render_page_rgb};

/// Catalog 1, Pages 2, then `page` as object 3 and `rest` from object 4 on.
fn one_page(page: &str, rest: Vec<Vec<u8>>) -> Vec<u8> {
    let mut objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        page.as_bytes().to_vec(),
    ];
    objects.extend(rest);
    assemble_pdf(&objects)
}

/// Fraction of the page's pixels that are not white, at 72 dpi.
fn ink(pdf: Vec<u8>) -> f64 {
    let s = session(pdf, BackendPolicy::CpuOnly);
    let bmp = render_page_rgb(&s, 1, 1.0).expect("page renders");
    let px = bmp.data().chunks_exact(3);
    let total = px.len();
    let dark = px.filter(|p| p.iter().any(|&c| c < 128)).count();
    #[expect(
        clippy::cast_precision_loss,
        reason = "pixel counts of a test page are far below 2^52"
    )]
    let frac = dark as f64 / total as f64;
    frac
}

/// A 200 × 100 pt page whose content fills the 80 × 80 square at (10, 10):
/// 32 % of the page.
const SQUARE: &[u8] = b"0 g 10 10 80 80 re f";

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
