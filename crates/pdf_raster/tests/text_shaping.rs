//! Shaping of non-embedded text in scripts that need it: the codes of one
//! show string drawn by the same substitute face are shaped together.
//!
//! The faces are the bundled ones (no installed fonts are searched), so the
//! glyphs are fixed.  Showing a word one code per `Tj` places every code at
//! the same pen position as one `Tj` but shapes nothing, which gives the
//! unshaped drawing to compare with.
#![cfg(feature = "bundled-fonts")]

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use std::path::PathBuf;

use common::{stream_object, with_page_tree};
use rasterrocket::{BackendPolicy, SessionConfig, open_session_from_bytes, render_page_rgb};

const WIDTH: u32 = 160;
const HEIGHT: u32 = 60;

/// Render `show`, a text object body, with a font whose codes from 65 on
/// are the glyphs `names`, using the bundled faces alone.  The page as
/// rows of booleans, `true` where inked.
fn render(names: &[&str], show: &str) -> Vec<Vec<bool>> {
    let empty = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("text-shaping-empty");
    std::fs::create_dir_all(&empty).expect("creates the empty font directory");
    let differences = names
        .iter()
        .map(|n| format!("/{n}"))
        .collect::<Vec<_>>()
        .join(" ");
    let font = format!(
        "<< /Type /Font /Subtype /TrueType /BaseFont /Arial \
         /Encoding << /BaseEncoding /WinAnsiEncoding /Differences [65 {differences}] >> >>"
    );
    let content = format!("BT /F1 36 Tf 20 15 Td {show} ET\n");
    let pdf = with_page_tree(&[
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {WIDTH} {HEIGHT}] \
             /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        )
        .into_bytes(),
        stream_object("", content.as_bytes()),
        font.into_bytes(),
    ]);
    let mut config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    config.font_dirs = vec![empty];
    config.system_fonts = false;
    let session = open_session_from_bytes(pdf, &config).expect("session");
    let page = render_page_rgb(&session, 1, 1.0).expect("render page");
    let width = page.width as usize;
    (0..page.height as usize)
        .map(|y| {
            (0..width)
                .map(|x| page.data()[(y * width + x) * 3] < 128)
                .collect()
        })
        .collect()
}

/// The codes 65.. of `n` glyphs as one show string, and as one per `Tj`.
fn shown_whole_and_apart(n: u8) -> (String, String) {
    use std::fmt::Write;
    let codes: Vec<u8> = (65..65 + n).collect();
    let mut whole = String::from("<");
    for c in &codes {
        let _ = write!(whole, "{c:02X}");
    }
    whole.push_str("> Tj");
    let apart = codes
        .iter()
        .map(|c| format!("<{c:02X}> Tj"))
        .collect::<Vec<_>>()
        .join(" ");
    (whole, apart)
}

/// The inked columns between the first and the last inked one that hold
/// no ink: the gaps between unconnected glyphs.
fn gaps(page: &[Vec<bool>]) -> usize {
    let inked: Vec<bool> = (0..page[0].len())
        .map(|x| page.iter().any(|row| row[x]))
        .collect();
    let Some(first) = inked.iter().position(|&i| i) else {
        return 0;
    };
    let last = inked.iter().rposition(|&i| i).expect("a column is inked");
    inked[first..=last].iter().filter(|&&i| !i).count()
}

/// Beh, yeh, teh ("بيت") in visual order, right to left: drawn apart the
/// isolated forms stand separately; shaped, the initial, medial and final
/// forms join into one connected word.
#[test]
fn arabic_letters_of_one_string_join() {
    let (whole, apart) = shown_whole_and_apart(3);
    let names = ["uni062A", "uni064A", "uni0628"];
    let joined = render(&names, &whole);
    let isolated = render(&names, &apart);
    assert!(
        gaps(&isolated) > 0,
        "isolated forms leave gaps between letters"
    );
    assert_eq!(gaps(&joined), 0, "the shaped word is connected");
    // Teh's two dots, the word's highest ink, sit over its final form at
    // the left end; a run shaped in the wrong order puts teh at the right.
    let top = joined
        .iter()
        .position(|row| row.contains(&true))
        .expect("ink");
    let inked = |x: usize| joined.iter().any(|row| row[x]);
    let first = (0..joined[0].len()).find(|&x| inked(x)).expect("ink");
    let last = (0..joined[0].len()).rfind(|&x| inked(x)).expect("ink");
    let dots = joined[top]
        .iter()
        .rposition(|&i| i)
        .expect("ink on the top row");
    assert!(
        dots < usize::midpoint(first, last),
        "teh's dots lie in the left half of the word ({dots} in {first}..={last})"
    );
}

/// The mean column of the ink above the headline, the row with the most
/// ink: the loop of the vowel sign i.
fn loop_centre(page: &[Vec<bool>]) -> f64 {
    let headline = (0..page.len())
        .max_by_key(|&y| page[y].iter().filter(|&&i| i).count())
        .expect("the page has rows");
    let columns: Vec<usize> = page[..headline]
        .iter()
        .flat_map(|row| row.iter().enumerate().filter(|(_, i)| **i).map(|(x, _)| x))
        .collect();
    assert!(!columns.is_empty(), "ink above the headline");
    #[expect(
        clippy::cast_precision_loss,
        reason = "column sums and counts of a 160-pixel page are exact in f64"
    )]
    let centre = columns.iter().sum::<usize>() as f64 / columns.len() as f64;
    centre
}

/// Ka with the vowel sign i ("कि"): the sign is written after ka but drawn
/// before it, its loop over ka, so shaping moves the ink above the
/// headline to the left of where the unshaped sign draws it.
#[test]
fn devanagari_vowel_sign_i_is_drawn_before_its_consonant() {
    let (whole, apart) = shown_whole_and_apart(2);
    let names = ["uni0915", "uni093F"];
    let shaped = loop_centre(&render(&names, &whole));
    let unshaped = loop_centre(&render(&names, &apart));
    assert!(
        shaped + 10.0 < unshaped,
        "the loop moves left of the unshaped sign's ({shaped} vs {unshaped})"
    );
}

/// Latin text shows glyph by glyph whether or not it is split.
#[test]
fn latin_text_is_not_shaped() {
    let (whole, apart) = shown_whole_and_apart(3);
    let names = ["f", "f", "i"];
    assert_eq!(render(&names, &whole), render(&names, &apart));
}
