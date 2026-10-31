//! The document's interactive form type is reported by the session and in
//! every page's diagnostics, so a dynamic XFA form's placeholder page can be
//! told from real content.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use common::{assemble_pdf, stream_object};
use rasterrocket::{
    BackendPolicy, FormType, RasterOptions, open_session_from_bytes, prescan_session,
    raster_pdf_from_bytes,
};

/// A one-page document whose catalog carries `catalog_extra`.
fn document(catalog_extra: &str) -> Vec<u8> {
    assemble_pdf(&[
        format!("<< /Type /Catalog /Pages 2 0 R {catalog_extra} >>").into_bytes(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Contents 4 0 R >>".to_vec(),
        stream_object("", b"0 g 10 10 80 80 re f"),
        stream_object("", b"<xdp:xdp/>"),
    ])
}

/// The form type as the session, the page diagnostics of a render and the
/// prescan report it.
fn reported(catalog_extra: &str) -> [FormType; 3] {
    let pdf = document(catalog_extra);
    let config = rasterrocket::SessionConfig::with_policy(BackendPolicy::CpuOnly);
    let session = open_session_from_bytes(pdf.clone(), &config).expect("open session");
    let opts = RasterOptions::new(36.0);
    let (_, page) = raster_pdf_from_bytes(pdf, &opts).next().expect("one page");
    let rendered = page.expect("page renders").diagnostics.form_type;
    let prescanned = prescan_session(&session, 1).expect("prescan").form_type;
    [session.form_type(), rendered, prescanned]
}

#[test]
fn a_document_without_a_form() {
    assert_eq!(reported(""), [FormType::NoForm; 3]);
}

#[test]
fn an_acroform() {
    assert_eq!(
        reported("/AcroForm << /Fields [] >>"),
        [FormType::AcroForm; 3]
    );
}

#[test]
fn a_static_xfa_form() {
    assert_eq!(
        reported("/AcroForm << /Fields [] /XFA 5 0 R >>"),
        [FormType::XfaForeground; 3]
    );
}

#[test]
fn a_dynamic_xfa_form() {
    assert_eq!(
        reported("/AcroForm << /Fields [] /XFA 5 0 R >> /NeedsRendering true"),
        [FormType::XfaFull; 3]
    );
}
