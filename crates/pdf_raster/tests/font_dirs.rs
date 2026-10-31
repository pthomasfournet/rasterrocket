//! `SessionConfig::font_dirs`: extra directories of installed fonts,
//! searched before the standard ones for the substitute of a font the
//! document does not embed and for the faces that draw the characters a
//! substitute lacks.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use std::path::PathBuf;

use common::{dark_pixels_with, text_page as page};
use rasterrocket::{BackendPolicy, SessionConfig};

/// Dark pixels of page 1 of `pdf` rendered at one pixel per point with
/// the font directories `font_dirs`.
fn ink(pdf: Vec<u8>, font_dirs: Vec<PathBuf>) -> usize {
    let mut config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    config.font_dirs = font_dirs;
    dark_pixels_with(pdf, &config).len()
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
