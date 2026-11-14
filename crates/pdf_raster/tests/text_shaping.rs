//! Shaping of non-embedded text in scripts that need it: the codes of one
//! show string drawn by the same substitute face are shaped together, in
//! the context of the strings shown next to it on its baseline.
//!
//! The faces are the bundled ones (no installed fonts are searched), so the
//! glyphs are fixed.  Showing a word one code per `Tj`, with `q Q` between
//! the strings, places every code at the same pen position as one `Tj` but
//! shapes nothing, since `q` ends the context of the string before it; that
//! gives the unshaped drawing to compare with.  A text field's value, in
//! logical order, is drawn through the same path once its appearance has
//! put each line in visual order.
#![cfg(feature = "bundled-fonts")]

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use common::{assemble_pdf, font_dir, isolated_fonts, stream_object, with_page_tree};
use rasterrocket::{open_session_from_bytes, render_page_rgb};

const WIDTH: u32 = 160;
const HEIGHT: u32 = 60;

/// The first page of `pdf` rendered with the bundled faces alone, as rows
/// of booleans, `true` where inked.
fn rendered(pdf: Vec<u8>) -> Vec<Vec<bool>> {
    let config = isolated_fonts(vec![font_dir("text-shaping-empty", &[])]);
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

/// Render `show`, a text object body, with a font whose codes from 65 on
/// are the glyphs `names`, selected as `/F1` and as `/F2`.
fn render(names: &[&str], show: &str) -> Vec<Vec<bool>> {
    render_with_form(names, show, "")
}

/// [`render`], with the form `/Fm` whose content is `form` and whose
/// resources select the same font as `/F1`.
fn render_with_form(names: &[&str], show: &str, form: &str) -> Vec<Vec<bool>> {
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
             /Resources << /Font << /F1 5 0 R /F2 5 0 R >> /XObject << /Fm 6 0 R >> >> \
             /Contents 4 0 R >>"
        )
        .into_bytes(),
        stream_object("", content.as_bytes()),
        font.into_bytes(),
        stream_object(
            &format!(
                "/Type /XObject /Subtype /Form /BBox [0 0 {WIDTH} {HEIGHT}] \
                 /Resources << /Font << /F1 5 0 R >> >>"
            ),
            form.as_bytes(),
        ),
    ]);
    rendered(pdf)
}

/// The codes 65.. of `n` glyphs as one show string, and as one per `Tj`
/// with no shaping context between them.
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
        .join(" q Q ");
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
    assert_joined_beh_yeh_teh(&joined);
}

/// Beh, yeh, teh split between show strings that follow one another on
/// the baseline, by a `TJ` adjustment, by `Tf` selecting the font anew and
/// by consecutive `Tj`s: each part takes the forms that join it to the
/// other, and each part's glyphs continue where the other's end, so the
/// word is drawn as when shown whole.  A small adjustment that tightens
/// the word keeps it connected.
#[test]
fn an_arabic_word_split_between_strings_joins_across_the_split() {
    let names = ["uni062A", "uni064A", "uni0628"];
    let whole = render(&names, "<414243> Tj");
    for show in [
        "[<41> 0 <4243>] TJ",
        "<41> Tj /F2 36 Tf <42> Tj /F1 36 Tf <43> Tj",
        "<41> Tj <42> Tj <43> Tj",
    ] {
        assert!(
            render(&names, show) == whole,
            "{show} is drawn as one string"
        );
    }
    assert_joined_beh_yeh_teh(&render(&names, "[<4142> 20 <43>] TJ"));
}

/// A `TJ` adjustment as wide as a word space separates the strings around
/// it: neither part joins the other.
#[test]
fn a_word_space_between_strings_keeps_them_apart() {
    let names = ["uni062A", "uni064A", "uni0628"];
    let spaced = render(&names, "[<41> -500 <4243>] TJ");
    let apart = render(&names, "[<41> -500] TJ q Q <4243> Tj");
    assert_eq!(spaced, apart);
}

/// A text rise moves a string off the baseline of the string before it:
/// a raised part of a word is drawn as after an operator that ends the
/// context, on either side of the rise.
#[test]
fn a_raised_string_is_no_neighbour() {
    let names = ["uni062A", "uni064A", "uni0628"];
    for (raised, apart) in [
        ("<41> Tj 4 Ts <4243> Tj", "<41> Tj q Q 4 Ts <4243> Tj"),
        (
            "[<41> 0 <42>] TJ 4 Ts <43> Tj",
            "[<41> 0 <42>] TJ q Q 4 Ts <43> Tj",
        ),
    ] {
        assert!(
            render(&names, raised) == render(&names, apart),
            "{raised} is drawn as {apart}"
        );
    }
}

/// A form that ends inside a text object gives the string shown after it
/// no context: the outer string, drawn where the form's last string ends,
/// is drawn as after an operator that ends the context.  An invisible teh
/// moves the outer pen to that end.
#[test]
fn a_form_ends_the_context_of_its_strings() {
    let names = ["uni062A", "uni064A", "uni0628"];
    let form = "BT /F1 36 Tf 20 15 Td <41> Tj";
    let after_form = render_with_form(&names, "3 Tr <41> Tj 0 Tr /Fm Do <4243> Tj", form);
    let apart = render_with_form(&names, "3 Tr <41> Tj 0 Tr /Fm Do q Q <4243> Tj", form);
    assert!(gaps(&apart) > 0, "the form's teh stands apart");
    assert!(
        after_form == apart,
        "the outer string takes no context from the form"
    );
}

/// A text field whose value is beh, yeh, teh in logical order: its
/// appearance shows the word right to left, joined.
#[test]
fn an_arabic_field_value_is_reordered_and_joined() {
    let pdf = assemble_pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R /AcroForm << /Fields [4 0 R] \
          /DR << /Font << /Helv 5 0 R >> >> >> >>"
            .to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {WIDTH} {HEIGHT}] /Annots [4 0 R] >>")
            .into_bytes(),
        b"<< /Type /Annot /Subtype /Widget /FT /Tx /T (a) /Rect [10 5 150 55] \
          /V <FEFF0628064A062A> /DA (/Helv 36 Tf 0 g) >>"
            .to_vec(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
            .to_vec(),
    ]);
    assert_joined_beh_yeh_teh(&rendered(pdf));
}

/// A text field whose value is beh, beh with three dots below (of the
/// Arabic Supplement), teh, and another holding a zero width non-joiner
/// between meem-yeh and khah: each word is set in one show string, the
/// first connected and the second broken at the non-joiner.
#[test]
fn an_arabic_field_value_across_unicode_blocks_is_one_string() {
    let field = |value: &str| {
        assemble_pdf(&[
            b"<< /Type /Catalog /Pages 2 0 R /AcroForm << /Fields [4 0 R] \
              /DR << /Font << /Helv 5 0 R >> >> >> >>"
                .to_vec(),
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
            format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {WIDTH} {HEIGHT}] /Annots [4 0 R] >>"
            )
            .into_bytes(),
            format!(
                "<< /Type /Annot /Subtype /Widget /FT /Tx /T (a) /Rect [10 5 150 55] \
                 /V <FEFF{value}> /DA (/Helv 36 Tf 0 g) >>"
            )
            .into_bytes(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
                .to_vec(),
        ])
    };
    assert_eq!(
        gaps(&rendered(field("06280750062A"))),
        0,
        "one connected word"
    );
    let with_non_joiner = rendered(field("064506CC200C062E"));
    assert!(gaps(&with_non_joiner) > 0, "the non-joiner breaks the word");
}

/// `page` shows beh, yeh, teh as one connected word with teh at the left
/// end: teh's two dots, the word's highest ink, sit over its final form.
/// A word shaped or ordered left to right puts teh at the right.
fn assert_joined_beh_yeh_teh(joined: &[Vec<bool>]) {
    assert_eq!(gaps(joined), 0, "the shaped word is connected");
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

/// Lamed, qamats, shin ("שָׁל" without the shin dot) in visual order, right
/// to left: the qamats, stored before shin, is shin's mark.  Drawn at the
/// end of a string or as a string of its own, it lies where the word shaped
/// as one string puts it, under shin at the right of the word.
#[test]
fn a_hebrew_mark_lies_under_its_letter_across_strings() {
    let names = ["uni05DC", "uni05B8", "uni05E9"];
    // The centre of the ink more than a pixel below the baseline, 15
    // units above the bottom edge: the qamats alone; and the middle of
    // the word's inked columns.
    let qamats = |show: &str| {
        let page = render(&names, show);
        let inked = |x: usize, rows: &[Vec<bool>]| rows.iter().any(|row| row[x]);
        let columns: Vec<usize> = (0..page[0].len()).filter(|&x| inked(x, &page)).collect();
        let below = &page[page.len() - 13..];
        let mark: Vec<usize> = columns
            .iter()
            .copied()
            .filter(|&x| inked(x, below))
            .collect();
        assert!(!mark.is_empty(), "{show}: the qamats is drawn");
        (
            usize::midpoint(mark[0], mark[mark.len() - 1]),
            usize::midpoint(columns[0], columns[columns.len() - 1]),
        )
    };
    let (shaped, middle) = qamats("<414243> Tj");
    assert!(shaped > middle, "the shaped qamats lies under shin");
    for show in [
        "[<4142> 0 <43>] TJ",
        "<4142> Tj <43> Tj",
        "<41> Tj <42> Tj <43> Tj",
    ] {
        let (centre, _) = qamats(show);
        assert!(
            centre.abs_diff(shaped) <= 1,
            "{show}: the qamats lies under shin ({centre}, shaped {shaped})"
        );
    }
}

/// Latin text shows glyph by glyph whether or not it is split.
#[test]
fn latin_text_is_not_shaped() {
    let (whole, apart) = shown_whole_and_apart(3);
    let names = ["f", "f", "i"];
    assert_eq!(render(&names, &whole), render(&names, &apart));
}
