//! The clipping operators `W`/`W*`: the path-painting operator that ends
//! the path paints unclipped by it, and the clip takes effect after the
//! paint (ISO 32000-1 §8.5.4).

use super::*;

/// The page side of the tests in pixels at 72 dpi.
const SIZE: usize = 16;

/// `content` rendered on a [`SIZE`]-pixel white page by [`render_gray`].
fn render(content: &[u8]) -> Vec<u8> {
    render_gray(16, content)
}

/// The gray value of pixel `(x, y)` of `px`.
const fn at(px: &[u8], x: usize, y: usize) -> u8 {
    px[y * SIZE + x]
}

/// `content`, then a black fill of the whole page: the clip `content`
/// left in force shows as the black region.
fn render_then_flood(content: &[u8]) -> Vec<u8> {
    let content = [content, b" 0 g 0 0 16 16 re f".as_slice()].concat();
    render(&content)
}

/// A 2-unit black stroke of the square `[4, 12]²`: its outer half covers
/// device column 3 and row 3, outside the square the clip encloses.
const SQUARE_STROKE: &[u8] = b"0 G 2 w 4 4 8 8 re";

#[test]
fn clip_then_stroke_paints_the_whole_stroke() {
    let content = [SQUARE_STROKE, b" W S".as_slice()].concat();
    let px = render(&content);
    assert_eq!(at(&px, 3, 8), 0, "outer half of the left edge");
    assert_eq!(at(&px, 8, 12), 0, "outer half of the bottom edge");
    assert_eq!(at(&px, 4, 8), 0, "inner half of the left edge");
}

#[test]
fn clip_then_fill_and_stroke_paints_the_whole_stroke() {
    let content = [SQUARE_STROKE, b" 0.5 g W B".as_slice()].concat();
    let px = render(&content);
    assert_eq!(at(&px, 3, 8), 0, "outer half of the left edge");
    assert_eq!(at(&px, 8, 8), 128, "the fill");
}

#[test]
fn clip_of_a_painted_path_applies_after_the_paint() {
    for paint in [b" W S".as_slice(), b" W f", b" W B", b" W n"] {
        let content = [SQUARE_STROKE, paint].concat();
        let px = render_then_flood(&content);
        for (x, y) in [(0, 0), (1, 8), (8, 1), (15, 15)] {
            assert_eq!(at(&px, x, y), 255, "{paint:?}: ({x}, {y}) outside the clip");
        }
        assert_eq!(at(&px, 8, 8), 0, "{paint:?}: (8, 8) inside the clip");
    }
}

/// A triangle whose hypotenuse crosses pixels partially: its antialiased
/// edge pixels carry the fill's coverage once, clipped or not.
#[test]
fn clip_then_fill_matches_the_fill_at_antialiased_edges() {
    let triangle = b"0 g 1 1 m 13 1 l 1 9 l h";
    let fill = render(&[triangle.as_slice(), b" f"].concat());
    let clipped = render(&[triangle.as_slice(), b" W f"].concat());
    assert!(
        fill.iter().any(|&v| v != 0 && v != 255),
        "the triangle has partially covered pixels"
    );
    assert_eq!(clipped, fill);
}

#[test]
fn clip_then_end_path_paints_nothing_and_clips() {
    let px = render(b"0 0 8 16 re W n");
    assert!(px.iter().all(|&v| v == 255), "W n paints nothing");
    let px = render_then_flood(b"0 0 8 16 re W n");
    for y in 0..SIZE {
        for x in 0..SIZE {
            let want = if x < 8 { 0 } else { 255 };
            assert_eq!(at(&px, x, y), want, "pixel ({x}, {y})");
        }
    }
}
