//! Quarter-turn rotation of a decoded page image.

use color::Gray8;
use pdf_raster::Rotation;
use raster::Bitmap;

/// `src` turned clockwise by `rotation`; `src` itself when the rotation is
/// [`Rotation::None`].  Every pixel is moved, none resampled, so turning by
/// a rotation and back restores `src` exactly.
pub(crate) fn turn(src: Bitmap<Gray8>, rotation: Rotation) -> Bitmap<Gray8> {
    match rotation {
        Rotation::None => src,
        Rotation::Cw90 => src.turned_cw(1),
        Rotation::Cw180 => src.turned_cw(2),
        Rotation::Cw270 => src.turned_cw(3),
    }
}

#[cfg(test)]
mod tests {
    use super::turn;
    use color::Gray8;
    use pdf_raster::Rotation;
    use raster::Bitmap;

    /// A 3 × 2 image whose pixels are numbered 0..6 in row order.
    fn numbered() -> Bitmap<Gray8> {
        let mut b = Bitmap::<Gray8>::new(3, 2, 1, false);
        for y in 0..2u8 {
            for (x, p) in (0..3u8).zip(b.row_mut(u32::from(y)).iter_mut()) {
                p.v = y * 3 + x;
            }
        }
        b
    }

    fn pixels(b: &Bitmap<Gray8>) -> (u32, u32, Vec<u8>) {
        let v = (0..b.height)
            .flat_map(|y| b.row(y).iter().map(|p| p.v))
            .collect();
        (b.width, b.height, v)
    }

    #[test]
    fn quarter_turns_move_every_pixel() {
        // 0 1 2      90°: 3 0    180°: 5 4 3    270°: 2 5
        // 3 4 5           4 1          2 1 0          1 4
        //                 5 2                         0 3
        for (rotation, want) in [
            (Rotation::None, (3, 2, vec![0, 1, 2, 3, 4, 5])),
            (Rotation::Cw90, (2, 3, vec![3, 0, 4, 1, 5, 2])),
            (Rotation::Cw180, (3, 2, vec![5, 4, 3, 2, 1, 0])),
            (Rotation::Cw270, (2, 3, vec![2, 5, 1, 4, 0, 3])),
        ] {
            assert_eq!(pixels(&turn(numbered(), rotation)), want, "{rotation:?}");
        }
    }
}
