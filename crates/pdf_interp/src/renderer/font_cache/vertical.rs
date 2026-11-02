//! Vertical forms drawn from horizontal glyphs.
//!
//! A vertical-form CID of a character collection draws the vertical
//! alternate of its character (PDF §9.7.4.3): a rotated bracket, dash or
//! long-vowel mark, or punctuation and small kana moved within the em box.
//! A substitute whose `GSUB` table has no `vert` / `vrt2` form for the
//! character draws its horizontal glyph instead, mapped in em space by the
//! transform [`glyph_transform`] gives.

use crate::resources::font::VerticalMetrics;

/// How a horizontal glyph stands in for its character's vertical form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VerticalFallback {
    /// Turned a quarter turn clockwise about the centre of the vertical em
    /// box.
    Rotate,
    /// Moved so that its box is point-reflected through the centre of the
    /// vertical em box: a mark in the lower-left quarter moves to the
    /// upper-right one, a centred mark stays.
    Reposition,
}

/// How the horizontal glyph of `code` stands in for the vertical form of
/// `code`; `None` when it is drawn as it is.
///
/// The classes follow the vertical orientation of Unicode Standard Annex
/// #50: code points with a transformed-upright vertical form (ideographic
/// comma and full stop, their full-width and half-width forms, small kana,
/// spacing sound marks) are repositioned; upright ones (vertical
/// presentation forms, which are already vertical, degree and prime signs,
/// pointing hands, scissors, geta mark, squared words, full-width
/// exclamation and question marks) are drawn as they are; every other
/// character with a vertical form (brackets, dashes, arrows, box drawing,
/// the long-vowel mark, the wave dash) is rotated.
#[must_use]
pub(crate) const fn vertical_fallback(code: u32) -> Option<VerticalFallback> {
    match code {
        0xFE10..=0xFE19
        | 0xFE30..=0xFE4F
        | 0x00B0
        | 0x2032
        | 0x2033
        | 0x261A..=0x261F
        | 0x2702
        | 0x3013
        | 0x3300..=0x33FF
        | 0xFF01
        | 0xFF1F => None,
        0x3001
        | 0x3002
        | 0xFF0C
        | 0xFF0E
        | 0xFF61
        | 0xFF64
        | 0xFF67..=0xFF6F
        | 0x309B
        | 0x309C
        | 0x3041
        | 0x3043
        | 0x3045
        | 0x3047
        | 0x3049
        | 0x3063
        | 0x3083
        | 0x3085
        | 0x3087
        | 0x308E
        | 0x3095
        | 0x3096
        | 0x30A1
        | 0x30A3
        | 0x30A5
        | 0x30A7
        | 0x30A9
        | 0x30C3
        | 0x30E3
        | 0x30E5
        | 0x30E7
        | 0x30EE
        | 0x30F5
        | 0x30F6
        | 0x31F0..=0x31FF => Some(VerticalFallback::Reposition),
        _ => Some(VerticalFallback::Rotate),
    }
}

/// A displacement, in em, below which a repositioned glyph stays put.
const NEGLIGIBLE_EM: f64 = 1e-3;

/// The em-space affine `[a, b, c, d, e, f]` (`x' = a·x + c·y + e`,
/// `y' = b·x + d·y + f`, y up, relative to the glyph's horizontal origin)
/// that draws a horizontal glyph as `fallback` for a CID of vertical
/// metrics `metrics`.  `bounds` is the glyph's box in em units, which
/// [`VerticalFallback::Reposition`] needs; `None` when the transform moves
/// the glyph by less than [`NEGLIGIBLE_EM`].
///
/// The vertical em box of the CID spans `w1y` downwards from its vertical
/// origin, which lies at `(vx, vy)` from the horizontal origin (PDF
/// §9.7.4.3); its centre is `(vx, vy + w1y / 2)`.
#[must_use]
pub(crate) fn glyph_transform(
    fallback: VerticalFallback,
    metrics: VerticalMetrics,
    bounds: Option<[f64; 4]>,
) -> Option<[f64; 6]> {
    let cx = f64::from(metrics.vx) / 1000.0;
    let cy = f64::from(metrics.w1y).mul_add(0.5, f64::from(metrics.vy)) / 1000.0;
    match fallback {
        // `(x, y) → (y, −x)` about the centre `C`: translation `C − R·C`.
        VerticalFallback::Rotate => Some([0.0, -1.0, 1.0, 0.0, cx - cy, cy + cx]),
        VerticalFallback::Reposition => {
            let [x0, y0, x1, y1] = bounds?;
            let (dx, dy) = (
                2.0f64.mul_add(cx, -(x0 + x1)),
                2.0f64.mul_add(cy, -(y0 + y1)),
            );
            (dx.abs().max(dy.abs()) >= NEGLIGIBLE_EM).then_some([1.0, 0.0, 0.0, 1.0, dx, dy])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FULL_WIDTH: VerticalMetrics = VerticalMetrics {
        w1y: -1000,
        vx: 500,
        vy: 880,
    };

    /// Apply the affine `t` to `(x, y)`.
    fn apply(t: [f64; 6], (x, y): (f64, f64)) -> (f64, f64) {
        (
            t[0].mul_add(x, t[2].mul_add(y, t[4])),
            t[1].mul_add(x, t[3].mul_add(y, t[5])),
        )
    }

    fn close((ax, ay): (f64, f64), (bx, by): (f64, f64)) -> bool {
        (ax - bx).abs() < 1e-9 && (ay - by).abs() < 1e-9
    }

    /// The classes of the characters with vertical forms.
    #[test]
    fn code_points_classify_by_vertical_orientation() {
        use VerticalFallback::{Reposition, Rotate};
        for (code, class) in [
            (0x30FC, Some(Rotate)),
            (0x301C, Some(Rotate)),
            (0x300C, Some(Rotate)),
            (0x2026, Some(Rotate)),
            (0x2192, Some(Rotate)),
            (0x3001, Some(Reposition)),
            (0x3002, Some(Reposition)),
            (0x3041, Some(Reposition)),
            (0x30C3, Some(Reposition)),
            (0xFE41, None),
            (0x00B0, None),
            (0x3300, None),
            (0xFF01, None),
        ] {
            assert_eq!(vertical_fallback(code), class, "U+{code:04X}");
        }
    }

    /// A rotated full-width glyph keeps its em box: `[0, 1] × [−0.12,
    /// 0.88]` turns onto itself, its left edge to the top and its baseline
    /// direction downwards.
    #[test]
    fn rotation_turns_the_em_box_onto_itself() {
        let t = glyph_transform(VerticalFallback::Rotate, FULL_WIDTH, None).expect("rotates");
        assert!(close(apply(t, (0.0, -0.12)), (0.0, 0.88)));
        assert!(close(apply(t, (1.0, 0.88)), (1.0, -0.12)));
        assert!(close(apply(t, (0.5, 0.38)), (0.5, 0.38)), "centre is fixed");
        // A horizontal stroke becomes a vertical one.
        let (a, b) = (apply(t, (0.1, 0.38)), apply(t, (0.9, 0.38)));
        assert!((a.0 - b.0).abs() < 1e-9 && a.1 > b.1);
    }

    /// Repositioning reflects the glyph box through the em-box centre: a
    /// comma in the lower-left quarter moves to the upper-right one, a
    /// centred mark stays.
    #[test]
    fn reposition_reflects_the_box_through_the_centre() {
        let comma = [0.1, -0.05, 0.3, 0.15];
        let t =
            glyph_transform(VerticalFallback::Reposition, FULL_WIDTH, Some(comma)).expect("moves");
        assert!(close(apply(t, (0.1, -0.05)), (0.7, 0.61)));
        assert!(close(apply(t, (0.3, 0.15)), (0.9, 0.81)));
        let centred = [0.4, 0.28, 0.6, 0.48];
        assert_eq!(
            glyph_transform(VerticalFallback::Reposition, FULL_WIDTH, Some(centred)),
            None
        );
        assert_eq!(
            glyph_transform(VerticalFallback::Reposition, FULL_WIDTH, None),
            None
        );
    }
}
