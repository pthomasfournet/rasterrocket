//! Antialiasing switches: `SessionConfig::text_antialias` governs glyph
//! bitmaps and `SessionConfig::vector_antialias` governs path fills,
//! strokes and clips.  With a switch off, the content it governs is drawn
//! with bilevel edges: every pixel is either the paint colour or the paper.

#[expect(dead_code, reason = "shared helpers; this test uses only PDF assembly")]
mod common;

use common::{one_page, stream_object};
use rasterrocket::{BackendPolicy, SessionConfig, open_session_from_bytes, render_page_rgb};

/// A 200 × 100 pt page drawing `content` in black, with Helvetica as `/F`.
fn page(content: &[u8]) -> Vec<u8> {
    one_page(
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] \
         /Resources << /Font << /F 5 0 R >> >> /Contents 4 0 R >>",
        vec![
            stream_object("", content),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
        ],
    )
}

/// Diagonal fill edges.
const FILL: &[u8] = b"0 g 13.3 7.7 m 187.1 52.9 l 61.4 93.2 l h f";
/// A diagonal stroke.
const STROKE: &[u8] = b"0 G 2.7 w 5 95 m 195 5 l S";
/// A rectangle fill clipped by a diagonal path.
const CLIP: &[u8] = b"10 10 m 190 30 l 100 95 l h W n 0 g 0 0 200 100 re f";
/// Text only.
const TEXT: &[u8] = b"0 g BT /F 23.7 Tf 11.3 41.7 Td (Bilevel glyphs) Tj ET";

/// Number of pixels of page 1 that are neither black nor white.
fn grey_pixels(pdf: &[u8], text: bool, vector: bool) -> usize {
    let mut config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    config.text_antialias = text;
    config.vector_antialias = vector;
    let s = open_session_from_bytes(pdf.to_vec(), &config).expect("open session");
    let bmp = render_page_rgb(&s, 1, 1.7).expect("render page");
    let mut grey = 0;
    for row in 0..bmp.height {
        let (px, _) = bmp.row_bytes(row)[..bmp.width as usize * 3].as_chunks::<3>();
        grey += px
            .iter()
            .filter(|p| **p != [0, 0, 0] && **p != [255, 255, 255])
            .count();
    }
    grey
}

#[test]
fn vector_antialias_off_draws_paths_bilevel() {
    for (what, content) in [("fill", FILL), ("stroke", STROKE), ("clip", CLIP)] {
        let pdf = page(content);
        assert!(
            grey_pixels(&pdf, true, true) > 0,
            "{what}: antialiased edges carry partial coverage"
        );
        assert_eq!(
            grey_pixels(&pdf, true, false),
            0,
            "{what}: vector antialiasing off leaves no partial coverage"
        );
    }
}

#[test]
fn text_antialias_off_draws_glyphs_bilevel() {
    let pdf = page(TEXT);
    assert!(
        grey_pixels(&pdf, true, true) > 0,
        "antialiased glyphs carry partial coverage"
    );
    assert_eq!(
        grey_pixels(&pdf, false, true),
        0,
        "text antialiasing off leaves no partial coverage"
    );
    assert!(
        grey_pixels(&pdf, true, false) > 0,
        "vector antialiasing does not govern glyph bitmaps"
    );
}
