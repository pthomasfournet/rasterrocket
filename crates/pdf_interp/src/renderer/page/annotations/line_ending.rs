//! Line endings (ISO 32000-2 §12.5.6.7, Table 179): the shapes drawn at
//! the ends of a `Line` or `PolyLine` and at the start of a free text
//! callout line.
//!
//! An ending is written in a local frame whose origin is the end point and
//! whose x axis points away from the line, so the shapes below are fixed
//! paths scaled by the ending size.  The size is six times the border
//! width, at most half the length of the segment it ends.

use std::fmt::Write as _;

use super::content::{circle, n, rectangle};

/// Ratio of the ending size to the border width.
const SIZE_PER_WIDTH: f64 = 6.0;

/// `tan 30°`: the half-angle of an arrowhead.
const TAN_30: f64 = 0.577_350_27;

/// `(sin 30°, cos 30°)`: the direction of a slash, 30° clockwise from the
/// perpendicular to the line.
const SLASH: (f64, f64) = (0.5, 0.866_025_4);

/// A line ending style.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LineEnding {
    Square,
    Circle,
    Diamond,
    OpenArrow,
    ClosedArrow,
    Butt,
    ROpenArrow,
    RClosedArrow,
    Slash,
}

impl LineEnding {
    /// The ending a `/LE` name selects; `None` for `/None` and unknown names.
    pub(super) const fn from_name(name: &[u8]) -> Option<Self> {
        Some(match name {
            b"Square" => Self::Square,
            b"Circle" => Self::Circle,
            b"Diamond" => Self::Diamond,
            b"OpenArrow" => Self::OpenArrow,
            b"ClosedArrow" => Self::ClosedArrow,
            b"Butt" => Self::Butt,
            b"ROpenArrow" => Self::ROpenArrow,
            b"RClosedArrow" => Self::RClosedArrow,
            b"Slash" => Self::Slash,
            _ => return None,
        })
    }

    /// Whether the ending is a closed shape, filled with the interior
    /// colour.
    const fn closed(self) -> bool {
        matches!(
            self,
            Self::Square | Self::Circle | Self::Diamond | Self::ClosedArrow | Self::RClosedArrow
        )
    }

    /// The size of an ending on a segment `length` long with border
    /// `width`.
    pub(super) fn size(width: f64, length: f64) -> f64 {
        (SIZE_PER_WIDTH * width).min(length / 2.0).max(0.0)
    }

    /// Append the ending at `tip`, the end of a segment coming from `from`,
    /// at `size`; closed shapes are filled with `fill` when given.  The
    /// stroke colour and line width in effect stroke it, undashed.
    pub(super) fn write(
        self,
        s: &mut String,
        tip: (f64, f64),
        from: (f64, f64),
        size: f64,
        fill: Option<&str>,
    ) {
        let (dx, dy) = (tip.0 - from.0, tip.1 - from.1);
        let length = dx.hypot(dy);
        if size <= 0.0 || length <= 0.0 {
            return;
        }
        let (cos, sin) = (dx / length, dy / length);
        let _ = writeln!(
            s,
            "q [] 0 d {} {} {} {} {} {} cm",
            n(cos),
            n(sin),
            n(-sin),
            n(cos),
            n(tip.0),
            n(tip.1)
        );
        let fill = fill.filter(|_| self.closed());
        if let Some(fill) = fill {
            let _ = writeln!(s, "{fill}");
        }
        let half = size / 2.0;
        let wing = size * TAN_30;
        let points = |s: &mut String, pts: &[(f64, f64)]| {
            for (i, (x, y)) in pts.iter().enumerate() {
                let op = if i == 0 { "m" } else { "l" };
                let _ = writeln!(s, "{} {} {op}", n(*x), n(*y));
            }
        };
        match self {
            Self::Square => rectangle(s, [-half, -half, half, half]),
            Self::Circle => circle(s, [-half, -half, half, half]),
            Self::Diamond => {
                points(s, &[(half, 0.0), (0.0, half), (-half, 0.0), (0.0, -half)]);
                s.push_str("h\n");
            }
            Self::OpenArrow | Self::ClosedArrow => {
                points(s, &[(-size, wing), (0.0, 0.0), (-size, -wing)]);
            }
            Self::ROpenArrow | Self::RClosedArrow => {
                points(s, &[(size, wing), (0.0, 0.0), (size, -wing)]);
            }
            Self::Butt => points(s, &[(0.0, half), (0.0, -half)]),
            Self::Slash => points(
                s,
                &[
                    (SLASH.0 * half, SLASH.1 * half),
                    (-SLASH.0 * half, -SLASH.1 * half),
                ],
            ),
        }
        let op = match (self.closed(), fill.is_some()) {
            (true, true) => "b",
            (true, false) => "s",
            (false, _) => "S",
        };
        let _ = writeln!(s, "{op} Q");
    }
}

#[cfg(test)]
mod tests {
    use super::LineEnding;

    #[test]
    fn names_select_endings_and_none_selects_nothing() {
        assert_eq!(
            LineEnding::from_name(b"OpenArrow"),
            Some(LineEnding::OpenArrow)
        );
        assert_eq!(LineEnding::from_name(b"None"), None);
        assert_eq!(LineEnding::from_name(b"Bogus"), None);
    }

    #[test]
    fn the_size_is_six_widths_capped_at_half_the_segment() {
        assert!((LineEnding::size(1.0, 100.0) - 6.0).abs() < 1e-12);
        assert!((LineEnding::size(4.0, 10.0) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn an_ending_is_drawn_in_a_frame_pointing_along_the_line() {
        let mut s = String::new();
        LineEnding::ClosedArrow.write(&mut s, (10.0, 0.0), (0.0, 0.0), 6.0, Some("0 0 1 rg"));
        assert!(s.starts_with("q [] 0 d 1 0 0 1 10 0 cm\n"), "{s}");
        assert!(s.contains("0 0 1 rg") && s.contains("-6 3.4641 m"), "{s}");
        assert!(s.trim_end().ends_with("b Q"), "{s}");
        let mut s = String::new();
        LineEnding::OpenArrow.write(&mut s, (0.0, 0.0), (0.0, 10.0), 6.0, Some("0 0 1 rg"));
        assert!(s.starts_with("q [] 0 d 0 -1 1 0 0 0 cm\n"), "{s}");
        assert!(!s.contains("rg") && s.trim_end().ends_with("S Q"), "{s}");
    }
}
