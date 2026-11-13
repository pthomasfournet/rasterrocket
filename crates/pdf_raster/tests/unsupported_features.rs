//! Features the renderer ignores are reported as typed data: the
//! document-level ones by the session, a page's in its diagnostics from
//! both a render and a prescan.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use common::{assemble_pdf, stream_object};
use rasterrocket::{
    BackendPolicy, MediaKind, RasterOptions, RenderParams, ScriptLocation, SessionConfig,
    UnsupportedFeature, open_session_from_bytes, prescan_session, raster_pdf_from_bytes,
    render_page_rgb_with_diagnostics,
};

/// A one-page document whose catalog carries `catalog_extra` and whose page
/// has `/AA` `page_aa` and three annotations: a 3D annotation with a black
/// poster appearance, a link with a JavaScript action and a widget with a
/// JavaScript calculate action.
fn document(catalog_extra: &str, page_aa: &str) -> Vec<u8> {
    assemble_pdf(&[
        format!("<< /Type /Catalog /Pages 2 0 R {catalog_extra} >>").into_bytes(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Contents 4 0 R \
             /Annots [5 0 R 6 0 R 7 0 R] {page_aa} >>"
        )
        .into_bytes(),
        stream_object("", b""),
        b"<< /Type /Annot /Subtype /3D /Rect [20 20 80 80] /F 4 /AP << /N 8 0 R >> >>".to_vec(),
        b"<< /Type /Annot /Subtype /Link /Rect [0 0 10 10] /A << /S /JavaScript /JS (app.alert(1)) >> >>"
            .to_vec(),
        b"<< /Type /Annot /Subtype /Widget /Rect [0 90 10 100] /AA << /C << /S /JavaScript /JS (1) >> >> >>"
            .to_vec(),
        stream_object("/Type /XObject /Subtype /Form /BBox [0 0 1 1]", b"0 g 0 0 1 1 re f"),
    ])
}

/// No features.
const NONE: [UnsupportedFeature; 0] = [];

fn page_features() -> Vec<UnsupportedFeature> {
    vec![
        UnsupportedFeature::JavaScript(ScriptLocation::PageActions),
        UnsupportedFeature::MediaAnnotation {
            kind: MediaKind::ThreeD,
            index: 0,
            has_appearance: true,
        },
        UnsupportedFeature::JavaScript(ScriptLocation::AnnotationAction { index: 1 }),
        UnsupportedFeature::JavaScript(ScriptLocation::AnnotationActions { index: 2 }),
    ]
}

#[test]
fn the_session_reports_the_document_level_features() {
    let pdf = document(
        "/OpenAction << /S /JavaScript /JS (1) >> /Collection << /View /D >> \
         /AcroForm << /Fields [] /XFA [] >>",
        "",
    );
    let config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    let session = open_session_from_bytes(pdf, &config).expect("open session");
    assert_eq!(
        session.unsupported_features(),
        [
            UnsupportedFeature::JavaScript(ScriptLocation::OpenAction),
            UnsupportedFeature::Xfa { dynamic: false },
            UnsupportedFeature::Portfolio,
        ]
    );
}

#[test]
fn a_page_reports_its_features_from_render_and_prescan() {
    let pdf = document("", "/AA << /O << /S /JavaScript /JS (1) >> >>");
    let config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    let session = open_session_from_bytes(pdf.clone(), &config).expect("open session");
    assert_eq!(session.unsupported_features(), NONE);

    let (bitmap, diag) = render_page_rgb_with_diagnostics(&session, 1, &RenderParams::new(1.0))
        .expect("page renders");
    assert_eq!(diag.unsupported_features, page_features());
    // The 3D annotation's poster appearance is drawn in its place.
    let centre = (50 * bitmap.width as usize + 50) * 3;
    assert_eq!(&bitmap.data()[centre..centre + 3], [0, 0, 0]);

    let prescanned = prescan_session(&session, 1).expect("prescan");
    assert_eq!(prescanned.unsupported_features, page_features());

    let (_, page) = raster_pdf_from_bytes(pdf, &RasterOptions::new(72.0))
        .next()
        .expect("one page");
    assert_eq!(
        page.expect("page renders").diagnostics.unsupported_features,
        page_features()
    );
}

#[test]
fn a_page_without_features_reports_none() {
    let pdf = assemble_pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Contents 4 0 R >>".to_vec(),
        stream_object("", b"0 g 10 10 80 80 re f"),
    ]);
    let config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    let session = open_session_from_bytes(pdf, &config).expect("open session");
    let (_, diag) = render_page_rgb_with_diagnostics(&session, 1, &RenderParams::new(1.0))
        .expect("page renders");
    assert_eq!(session.unsupported_features(), NONE);
    assert_eq!(diag.unsupported_features, NONE);
}
