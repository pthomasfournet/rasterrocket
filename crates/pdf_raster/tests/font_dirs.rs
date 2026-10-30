//! `SessionConfig::font_dirs`: extra directories of installed fonts,
//! searched before the standard ones for the substitute of a font the
//! document does not embed and for the faces that draw the characters a
//! substitute lacks.

#[expect(dead_code, reason = "shared helpers; this test uses PDF assembly only")]
mod common;

use std::path::PathBuf;

use common::{assemble_pdf, stream_object};
use rasterrocket::{BackendPolicy, SessionConfig, open_session_from_bytes, render_page_rgb};

/// A 64 × 32 pt page showing `text` at 24 pt in the non-embedded font
/// `font` (a font dictionary body).
fn page(font: &str, text: &str) -> Vec<u8> {
    assemble_pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 64 32] \
          /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
            .to_vec(),
        stream_object("", format!("BT /F1 24 Tf 4 8 Td {text} Tj ET").as_bytes()),
        font.as_bytes().to_vec(),
    ])
}

/// Dark pixels of page 1 of `pdf` rendered at one pixel per point with
/// the font directories `font_dirs`.
fn ink(pdf: Vec<u8>, font_dirs: Vec<PathBuf>) -> usize {
    let mut config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    config.font_dirs = font_dirs;
    let session = open_session_from_bytes(pdf, &config).expect("session");
    let bmp = render_page_rgb(&session, 1, 1.0).expect("render page");
    bmp.data().chunks(3).filter(|px| px[0] < 128).count()
}

/// A new directory under the test target's scratch space holding `bytes`
/// as `file`.
fn font_dir(name: &str, file: &str, bytes: &[u8]) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::create_dir_all(&dir).expect("creates the font directory");
    std::fs::write(dir.join(file), bytes).expect("writes the font");
    dir
}

/// A font directory is searched first for the installed substitute: a
/// file named as the free sans-serif family draws Helvetica.  Here it is
/// Ahem, whose glyphs are solid squares.
#[test]
fn font_dirs_supply_the_installed_substitute() {
    let dir = font_dir(
        "font-dirs-substitute",
        "LiberationSans-Regular.ttf",
        include_bytes!("../../font_test_data/test_data/ttf/ahem.ttf"),
    );
    let helvetica = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>";
    let standard = ink(page(helvetica, "(AB)"), Vec::new());
    let squares = ink(page(helvetica, "(AB)"), vec![dir]);
    assert!(
        squares > 2 * standard,
        "Ahem squares ({squares} dark pixels) against Helvetica ({standard})"
    );
}

/// A font directory supplies the face that draws a character the
/// substitute lacks: a private-use symbol named by `Differences`.
#[test]
fn font_dirs_supply_fallback_glyphs() {
    let dir = font_dir(
        "font-dirs-fallback",
        "material_symbols_subset.ttf",
        include_bytes!("../../font_test_data/test_data/ttf/material_symbols_subset.ttf"),
    );
    let font = "<< /Type /Font /Subtype /TrueType /BaseFont /Arial \
                /Encoding << /Differences [65 /uniEB2F] >> >>";
    let dark = ink(page(font, "(A)"), vec![dir]);
    assert!(dark > 20, "the symbol draws ({dark} dark pixels)");
}
