//! TrueType programs whose glyphs are assembled by their instructions.
//!
//! The fixtures (see `data/generate.py`) share one composite glyph, `A`:
//! the square `B` (x 100..400, y 0..300 of 1000 units) twice at the same
//! offset, the second copy moved right by half an em by the composite's
//! instructions.  Grid-fitted it is two squares, x 100..400 and 600..900;
//! unhinted it is one.

use super::{bounds, face_at, ink, sample_programs, subpaths, try_face};

const BY_NAME: &[u8] = include_bytes!("data/hint-assembled-by-name.ttf");
pub(super) const BY_TABLES: &[u8] = include_bytes!("data/hint-assembled-by-tables.ttf");
const PLAIN: &[u8] = include_bytes!("data/hint-assembled-plain.ttf");

const A: u32 = 0x41;
const B: u32 = 0x42;

/// `(min_x, max_x)` of glyph `code` in device pixels.
fn x_extent(data: &[u8], mat: [f64; 4], code: u32) -> (f64, f64, usize) {
    let face = try_face(data, mat).expect("font loads");
    let path = face.glyph_path(code).expect("glyph has an outline");
    let (x0, _, x1, _) = bounds(&path);
    (x0, x1, subpaths(&path))
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 0.5
}

#[test]
fn hint_reliant_programs_are_recognised_by_name_and_by_tables() {
    assert!(face_at(BY_NAME, 20.0).program().requires_hinting());
    assert!(face_at(BY_TABLES, 20.0).program().requires_hinting());
    assert!(!face_at(PLAIN, 20.0).program().requires_hinting());
    for (name, data, _) in sample_programs() {
        let expected = name.starts_with("hint-assembled");
        let program_requires = face_at(&data, 20.0).program().requires_hinting();
        assert_eq!(program_requires, expected, "{name}");
    }
}

#[test]
fn instructions_assemble_the_composite_glyph() {
    for data in [BY_NAME, BY_TABLES] {
        // 40 px per em: squares at 4..16 and 24..36.
        let (x0, x1, contours) = x_extent(data, [40.0, 0.0, 0.0, 40.0], A);
        assert!(close(x0, 4.0) && close(x1, 36.0), "x {x0}..{x1}");
        assert_eq!(contours, 2);
        let (x0, x1, _) = x_extent(data, [40.0, 0.0, 0.0, 40.0], B);
        assert!(close(x0, 4.0) && close(x1, 16.0), "x {x0}..{x1}");
    }
}

#[test]
fn unrecognised_program_keeps_its_unhinted_outline() {
    let (x0, x1, contours) = x_extent(PLAIN, [40.0, 0.0, 0.0, 40.0], A);
    assert!(close(x0, 4.0) && close(x1, 16.0), "x {x0}..{x1}");
    assert_eq!(contours, 2, "both components, overlapping");
}

#[test]
fn assembled_glyph_is_grid_fitted_at_each_size() {
    for px in [9.0, 12.0, 17.0, 33.0, 100.0, 250.0] {
        let (_, x1, _) = x_extent(BY_NAME, [px, 0.0, 0.0, px], A);
        // The shift is ppem / 2 pixels: the right edge sits at 0.9 em.
        let expected = 0.9 * px;
        assert!(
            close(x1, expected),
            "{px} px: right edge {x1}, expected {expected}"
        );
    }
}

#[test]
fn matrix_is_applied_after_grid_fitting() {
    // Fitted at the vertical size (40 px per em), then stretched 2× along x.
    let (x0, x1, _) = x_extent(BY_NAME, [80.0, 0.0, 0.0, 40.0], A);
    assert!(close(x0, 8.0) && close(x1, 72.0), "x {x0}..{x1}");
    // A 90° rotation maps the shift onto y.
    let face = try_face(BY_NAME, [0.0, 40.0, -40.0, 0.0]).expect("font loads");
    let (_, y0, _, y1) = bounds(&face.glyph_path(A).expect("outline"));
    assert!(close(y0, 4.0) && close(y1, 36.0), "y {y0}..{y1}");
}

#[test]
fn assembled_glyph_bitmap_has_both_squares() {
    let face = face_at(BY_NAME, 40.0);
    let pair = face.make_glyph(A, 0).expect("A renders");
    let square = face.make_glyph(B, 0).expect("B renders");
    assert!((ink(&pair) - ink(&square) - ink(&square)).abs() < 1.0);
    assert!(pair.width >= 32, "width {}", pair.width);
    let plain = face_at(PLAIN, 40.0);
    let overlapped = plain.make_glyph(A, 0).expect("A renders");
    assert!((ink(&overlapped) - ink(&square)).abs() < 1.0);
}

#[test]
fn advance_comes_from_the_metrics() {
    let face = face_at(BY_NAME, 40.0);
    assert_eq!(face.glyph_advance(A), Some(1.0));
}
