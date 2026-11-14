//! Quarter-turn rotation of a bitmap.

use crate::bitmap::Bitmap;
use color::Pixel;

impl<P: Pixel> Bitmap<P> {
    /// `self` turned clockwise by `quarters` quarter turns, taken modulo 4.
    ///
    /// Every pixel is moved, none resampled, so turning by `q` and then by
    /// `4 - q` restores the original exactly.  A quarter or three-quarter
    /// turn swaps width and height.  The alpha plane, when present, is
    /// turned with the colour data.  The result's rows are unpadded
    /// (`row_pad` 1).
    #[must_use]
    pub fn turned_cw(&self, quarters: u8) -> Self {
        let q = quarters % 4;
        let (w, h) = (self.width, self.height);
        let (dw, dh) = if q % 2 == 1 { (h, w) } else { (w, h) };
        // The source pixel that lands at destination (dx, dy).
        let source = |dx: u32, dy: u32| match q {
            1 => (dy, h - 1 - dx),
            2 => (w - 1 - dx, h - 1 - dy),
            3 => (w - 1 - dy, dx),
            _ => (dx, dy),
        };
        let mut dst = Self::new(dw, dh, 1, self.has_alpha());
        for dy in 0..dh {
            for (dx, out) in (0..dw).zip(dst.row_mut(dy).iter_mut()) {
                let (sx, sy) = source(dx, dy);
                *out = self.row(sy)[sx as usize];
            }
        }
        if let (Some(src), Some(out)) = (self.alpha_plane(), dst.alpha_plane_mut()) {
            let (sw, dwu) = (w as usize, dw as usize);
            for dy in 0..dh {
                for dx in 0..dw {
                    let (sx, sy) = source(dx, dy);
                    out[dy as usize * dwu + dx as usize] = src[sy as usize * sw + sx as usize];
                }
            }
        }
        dst
    }
}

#[cfg(test)]
mod tests {
    use crate::bitmap::Bitmap;
    use color::Gray8;

    /// A 3 × 2 image whose pixels, and alpha bytes, are numbered 0..6 in
    /// row order.
    fn numbered() -> Bitmap<Gray8> {
        let mut b = Bitmap::<Gray8>::new(3, 2, 4, true);
        for y in 0..2u8 {
            for (x, p) in (0..3u8).zip(b.row_mut(u32::from(y)).iter_mut()) {
                p.v = y * 3 + x;
            }
        }
        for (i, a) in (0u8..).zip(b.alpha_plane_mut().expect("alpha").iter_mut()) {
            *a = i + 10;
        }
        b
    }

    fn pixels(b: &Bitmap<Gray8>) -> (u32, u32, Vec<u8>, Vec<u8>) {
        let v = (0..b.height)
            .flat_map(|y| b.row(y).iter().map(|p| p.v))
            .collect();
        (
            b.width,
            b.height,
            v,
            b.alpha_plane().expect("alpha").to_vec(),
        )
    }

    #[test]
    fn quarter_turns_move_every_pixel_and_alpha_byte() {
        // 0 1 2      90°: 3 0    180°: 5 4 3    270°: 2 5
        // 3 4 5           4 1          2 1 0          1 4
        //                 5 2                         0 3
        for (q, (w, h, order)) in [
            (0, (3, 2, [0, 1, 2, 3, 4, 5])),
            (1, (2, 3, [3, 0, 4, 1, 5, 2])),
            (2, (3, 2, [5, 4, 3, 2, 1, 0])),
            (3, (2, 3, [2, 5, 1, 4, 0, 3])),
            (5, (2, 3, [3, 0, 4, 1, 5, 2])),
        ] {
            let alpha = order.iter().map(|i| i + 10).collect();
            assert_eq!(
                pixels(&numbered().turned_cw(q)),
                (w, h, order.to_vec(), alpha),
                "{q}"
            );
        }
    }

    #[test]
    fn a_turn_and_its_complement_restore_the_image() {
        let b = numbered();
        for q in 0..4 {
            assert_eq!(pixels(&b.turned_cw(q).turned_cw(4 - q)), pixels(&b), "{q}");
        }
    }
}
