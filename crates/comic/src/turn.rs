//! Quarter-turn rotation of a decoded page image.

use color::Gray8;
use pdf_raster::Rotation;
use raster::Bitmap;

/// `src` turned clockwise by `rotation`; `src` itself when the rotation is
/// [`Rotation::None`].  Every pixel is moved, none resampled, so turning by
/// a rotation and back restores `src` exactly.
pub(crate) fn turn(src: Bitmap<Gray8>, rotation: Rotation) -> Bitmap<Gray8> {
    if rotation == Rotation::None {
        return src;
    }
    let (w, h) = (src.width, src.height);
    let (dw, dh) = if rotation.is_quarter() {
        (h, w)
    } else {
        (w, h)
    };
    let mut dst = Bitmap::<Gray8>::new(dw, dh, 1, false);
    for dy in 0..dh {
        let row = dst.row_mut(dy);
        for (dx, out) in (0..dw).zip(row.iter_mut()) {
            // The source pixel that lands at (dx, dy).
            let (sx, sy) = match rotation {
                Rotation::Cw90 => (dy, h - 1 - dx),
                Rotation::Cw180 => (w - 1 - dx, h - 1 - dy),
                Rotation::Cw270 => (w - 1 - dy, dx),
                Rotation::None => (dx, dy),
            };
            *out = src.row(sy)[sx as usize];
        }
    }
    dst
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
