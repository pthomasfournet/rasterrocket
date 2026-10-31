//! The bundled fallback faces (`bundled-fonts` feature): with no installed
//! font at all, non-embedded text in Greek, Cyrillic, Hebrew, Arabic and
//! Devanagari still draws its glyphs.
#![cfg(feature = "bundled-fonts")]

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use std::path::PathBuf;

use common::{stream_object, with_page_tree};
use rasterrocket::{BackendPolicy, SessionConfig, open_session_from_bytes, render_page_rgb};

/// The characters shown, one per 40 pt cell: Greek alpha, Cyrillic zhe,
/// Hebrew alef, Arabic beh, Devanagari ka.
const NAMES: [&str; 5] = ["uni03B1", "uni0416", "uni05D0", "uni0628", "uni0915"];

/// A "bare server": an empty font directory and no system font lookup, so
/// only the faces compiled into the library can draw.  Every cell must
/// hold a glyph; a code no face covers would leave its cell blank.
#[test]
fn bundled_faces_draw_scripts_without_installed_fonts() {
    let empty = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("bundled-fonts-empty");
    std::fs::create_dir_all(&empty).expect("creates the empty font directory");
    let differences = NAMES.map(|n| format!("/{n}")).join(" ");
    let font = format!(
        "<< /Type /Font /Subtype /TrueType /BaseFont /Arial \
         /Encoding << /BaseEncoding /WinAnsiEncoding /Differences [65 {differences}] >> >>"
    );
    let mut content = String::new();
    for i in 0..NAMES.len() {
        let line = format!("BT /F1 24 Tf {} 8 Td <{:02X}> Tj ET\n", 8 + 40 * i, 65 + i);
        content.push_str(&line);
    }
    let pdf = with_page_tree(&[
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 208 40] \
          /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
            .to_vec(),
        stream_object("", content.as_bytes()),
        font.into_bytes(),
    ]);
    let mut config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    config.font_dirs = vec![empty];
    config.system_fonts = false;
    let session = open_session_from_bytes(pdf, &config).expect("session");
    let page = render_page_rgb(&session, 1, 1.0).expect("render page");
    let width = page.width as usize;
    for (i, name) in NAMES.iter().enumerate() {
        let inked = (0..page.height as usize)
            .flat_map(|y| (40 * i..40 * (i + 1)).map(move |x| (x, y)))
            .filter(|&(x, y)| page.data()[(y * width + x) * 3] < 128)
            .count();
        assert!(inked > 20, "{name} draws ({inked} dark pixels)");
    }
}
