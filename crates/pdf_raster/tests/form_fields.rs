//! Form field widgets without a normal appearance, or whose form sets
//! `/NeedAppearances`, are drawn from the field value, `/DA`, `/MK` and the
//! form's `/DR` (ISO 32000-2 §12.7): text and choice fields show their
//! value, check boxes their mark when on.

#[expect(dead_code, reason = "shared helpers; this test uses only PDF assembly")]
mod common;

use common::{assemble_pdf, stream_object};
use rasterrocket::{BackendPolicy, render_page_rgb};

/// A 200 × 100 pt page, rendered at one pixel per point.
const W: u32 = 200;
const H: u32 = 100;

/// A page whose `/Annots` are the widgets (objects 5 on) `widgets`, in a
/// form with the entries `acroform`.  Object 4 is Helvetica, `/Helv` in
/// the form's `/DR`.
fn document(acroform: &str, widgets: &[&str], extra: Vec<Vec<u8>>) -> Vec<u8> {
    let refs: Vec<String> = (0..widgets.len())
        .map(|i| format!("{} 0 R", i + 5))
        .collect();
    let refs = refs.join(" ");
    let mut objects = vec![
        format!(
            "<< /Type /Catalog /Pages 2 0 R /AcroForm << /Fields [{refs}] \
             /DR << /Font << /Helv 4 0 R >> >> {acroform} >> >>"
        )
        .into_bytes(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W} {H}] /Annots [{refs}] >>")
            .into_bytes(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
            .to_vec(),
    ];
    objects.extend(
        widgets
            .iter()
            .map(|w| format!("<< /Type /Annot /Subtype /Widget {w} >>").into_bytes()),
    );
    objects.extend(extra);
    assemble_pdf(&objects)
}

/// The number of dark pixels in the user-space box `[x0 y0 x1 y1]`.
fn dark_in(pdf: Vec<u8>, b: [u32; 4]) -> usize {
    let s = common::session(pdf, BackendPolicy::CpuOnly);
    let bmp = render_page_rgb(&s, 1, 1.0).expect("render page");
    let data = bmp.data();
    let mut n = 0;
    for y in b[1]..b[3] {
        for x in b[0]..b[2] {
            let i = (H - 1 - y) as usize * bmp.stride + x as usize * 3;
            if data[i..i + 3].iter().all(|&c| c < 100) {
                n += 1;
            }
        }
    }
    n
}

/// The interior of the text field at `[10 60 190 90]`, inside its border.
const FIELD: [u32; 4] = [12, 62, 188, 88];

const TEXT_FIELD: &str = "/FT /Tx /T (name) /Rect [10 60 190 90] /DA (/Helv 14 Tf 0 g)";

#[test]
fn a_text_field_without_an_appearance_shows_its_value() {
    let with = dark_in(
        document("", &[&format!("{TEXT_FIELD} /V (Hello OCR)")], vec![]),
        FIELD,
    );
    let without = dark_in(document("", &[TEXT_FIELD], vec![]), FIELD);
    assert!(with > 100, "{with} dark pixels");
    assert_eq!(without, 0);
}

#[test]
fn need_appearances_regenerates_a_text_field_appearance() {
    // The stored appearance is empty; the value is only in /V.
    let widget = format!("{TEXT_FIELD} /V (Hello OCR) /AP << /N 6 0 R >>");
    let empty = vec![stream_object(
        "/Type /XObject /Subtype /Form /BBox [0 0 180 30]",
        b"",
    )];
    let regenerated = dark_in(
        document("/NeedAppearances true", &[&widget], empty.clone()),
        FIELD,
    );
    let stored = dark_in(document("", &[&widget], empty), FIELD);
    assert!(regenerated > 100, "{regenerated} dark pixels");
    assert_eq!(
        stored, 0,
        "without NeedAppearances the stored appearance is used"
    );
}

#[test]
fn a_field_inherits_its_type_value_and_appearance_from_its_parent() {
    let parent = b"<< /FT /Tx /T (name) /V (Hello OCR) /DA (/Helv 14 Tf 0 g) >>".to_vec();
    let dark = dark_in(
        document("", &["/Parent 6 0 R /Rect [10 60 190 90]"], vec![parent]),
        FIELD,
    );
    assert!(dark > 100, "{dark} dark pixels");
}

#[test]
fn a_combo_box_shows_the_display_text_of_its_value() {
    let combo = "/FT /Ch /Ff 131072 /T (c) /Rect [10 60 190 90] /DA (/Helv 14 Tf 0 g) \
                 /Opt [[(a) (Apples and oranges)] [(b) (B)]]";
    let long = dark_in(document("", &[&format!("{combo} /V (a)")], vec![]), FIELD);
    let short = dark_in(document("", &[&format!("{combo} /V (b)")], vec![]), FIELD);
    assert!(long > 2 * short && short > 0, "{long} vs {short}");
}

#[test]
fn a_list_box_shows_its_options() {
    let list = "/FT /Ch /T (l) /Rect [10 10 190 90] /DA (/Helv 12 Tf 0 g) \
                /Opt [(First) (Second) (Third)]";
    let dark = dark_in(document("", &[list], vec![]), [12, 12, 188, 88]);
    assert!(dark > 100, "{dark} dark pixels");
}

/// A check box at `[10 10 40 40]` with the entries `state`.
fn check_box(state: &str) -> usize {
    let widget = format!(
        "/FT /Btn /T (cb) /Rect [10 10 40 40] /DA (/ZaDb 0 Tf 0 g) /MK << /CA (4) >> {state}"
    );
    dark_in(document("", &[&widget], vec![]), [12, 12, 38, 38])
}

#[test]
fn a_check_box_shows_its_mark_when_on() {
    let on = check_box("/V /Yes /AS /Yes");
    assert!(on > 50, "{on} dark pixels");
    assert_eq!(check_box("/V /Off /AS /Off"), 0);
    assert!(check_box("/V /Yes") > 50, "the value decides without /AS");
}

#[test]
fn a_button_with_an_appearance_keeps_it_under_need_appearances() {
    // The stored appearance draws nothing even though the box is on.
    let widget = "/FT /Btn /T (cb) /Rect [10 10 40 40] /V /Yes /AS /Yes \
                  /AP << /N << /Yes 6 0 R >> >>";
    let empty = vec![stream_object(
        "/Type /XObject /Subtype /Form /BBox [0 0 30 30]",
        b"",
    )];
    let dark = dark_in(
        document("/NeedAppearances true", &[widget], empty),
        [12, 12, 38, 38],
    );
    assert_eq!(dark, 0);
}
