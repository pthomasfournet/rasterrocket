//! A clockwise rotation in quarter turns, added to a page's `/Rotate`.

use crate::PageGeometry;

/// A clockwise rotation by a whole number of quarter turns.
///
/// [`PageGeometry::rotated`] adds it to the page's `/Rotate` (ISO 32000-2
/// §7.7.3.3, Table 31) before the page is drawn, so the content is
/// rasterised in the turned orientation rather than turned as a bitmap
/// afterwards.  The default is [`None`](Self::None).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Rotation {
    /// No rotation.
    #[default]
    None,
    /// A quarter turn clockwise.
    Cw90,
    /// A half turn.
    Cw180,
    /// Three quarter turns clockwise: a quarter turn counter-clockwise.
    Cw270,
}

/// A rotation that is not a multiple of 90 degrees.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidRotation {
    /// The degrees asked for.
    pub degrees: i64,
}

impl std::fmt::Display for InvalidRotation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "rotation {}° is not a multiple of 90 degrees",
            self.degrees
        )
    }
}

impl std::error::Error for InvalidRotation {}

impl Rotation {
    /// The rotation of `degrees` clockwise, a negative value counter-clockwise,
    /// reduced modulo 360: `-90` and `270` are [`Cw270`](Self::Cw270),
    /// `450` is [`Cw90`](Self::Cw90).
    ///
    /// # Errors
    /// [`InvalidRotation`] when `degrees` is not a multiple of 90.
    pub const fn from_degrees(degrees: i64) -> Result<Self, InvalidRotation> {
        match degrees.rem_euclid(360) {
            0 => Ok(Self::None),
            90 => Ok(Self::Cw90),
            180 => Ok(Self::Cw180),
            270 => Ok(Self::Cw270),
            _ => Err(InvalidRotation { degrees }),
        }
    }

    /// The rotation in clockwise degrees: 0, 90, 180 or 270.
    #[must_use]
    pub const fn degrees(self) -> u16 {
        match self {
            Self::None => 0,
            Self::Cw90 => 90,
            Self::Cw180 => 180,
            Self::Cw270 => 270,
        }
    }

    /// Whether the rotation exchanges the page's width and height.
    #[must_use]
    pub const fn is_quarter(self) -> bool {
        matches!(self, Self::Cw90 | Self::Cw270)
    }
}

impl PageGeometry {
    /// This geometry with `extra` added to the page's rotation: the
    /// rotation is the sum modulo 360, and the width and height are
    /// exchanged when `extra` is a quarter turn.  The box origin,
    /// `UserUnit` and the boxes are unchanged.
    #[must_use]
    pub const fn rotated(self, extra: Rotation) -> Self {
        let rotate_cw = (self.rotate_cw % 360 + extra.degrees()) % 360;
        let (width_pts, height_pts) = if extra.is_quarter() {
            (self.height_pts, self.width_pts)
        } else {
            (self.width_pts, self.height_pts)
        };
        Self {
            width_pts,
            height_pts,
            rotate_cw,
            ..self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{InvalidRotation, Rotation};
    use crate::PageGeometry;

    #[test]
    fn multiples_of_ninety_are_normalised() {
        for (degrees, want) in [
            (0, Rotation::None),
            (90, Rotation::Cw90),
            (180, Rotation::Cw180),
            (270, Rotation::Cw270),
            (360, Rotation::None),
            (450, Rotation::Cw90),
            (-90, Rotation::Cw270),
            (-180, Rotation::Cw180),
            (-270, Rotation::Cw90),
            (-360, Rotation::None),
            (720_000_090, Rotation::Cw90),
        ] {
            assert_eq!(Rotation::from_degrees(degrees), Ok(want), "{degrees}");
        }
    }

    #[test]
    fn other_angles_are_rejected() {
        for degrees in [1, 45, 89, 91, -45, 359, i64::MAX, i64::MIN] {
            assert_eq!(
                Rotation::from_degrees(degrees),
                Err(InvalidRotation { degrees }),
                "{degrees}"
            );
        }
    }

    #[test]
    fn rotation_adds_to_the_page_rotation_and_swaps_on_quarter_turns() {
        let geom = PageGeometry {
            width_pts: 200.0,
            height_pts: 100.0,
            rotate_cw: 90,
            user_unit: 1.0,
            origin_x: 5.0,
            origin_y: 7.0,
            media_box: None,
            crop_box: None,
        };
        for (extra, rotate_cw, size) in [
            (Rotation::None, 90, (200.0, 100.0)),
            (Rotation::Cw90, 180, (100.0, 200.0)),
            (Rotation::Cw180, 270, (200.0, 100.0)),
            (Rotation::Cw270, 0, (100.0, 200.0)),
        ] {
            let r = geom.rotated(extra);
            assert_eq!(r.rotate_cw, rotate_cw, "{extra:?}");
            assert_eq!((r.width_pts, r.height_pts), size, "{extra:?}");
            assert_eq!((r.origin_x, r.origin_y), (5.0, 7.0));
        }
    }

    /// A rotation outside `0..360`, as a caller may write into the public
    /// field, is reduced before the sum, which then cannot overflow.
    #[test]
    fn rotation_of_an_out_of_range_page_rotation_does_not_overflow() {
        let geom = PageGeometry {
            width_pts: 200.0,
            height_pts: 100.0,
            rotate_cw: u16::MAX,
            user_unit: 1.0,
            origin_x: 0.0,
            origin_y: 0.0,
            media_box: None,
            crop_box: None,
        };
        assert_eq!(
            geom.rotated(Rotation::Cw270).rotate_cw,
            285,
            "65 535 is 15 mod 360"
        );
    }
}
