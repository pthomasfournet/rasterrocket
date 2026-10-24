//! Synthetic analysis planes shared by the camera unit tests.

use super::plane::{Plane, floor_index, to_f64};

/// A `w × h` plane with pixel `(x, y) = f(x, y)`.
pub fn plane(w: usize, h: usize, f: impl Fn(usize, usize) -> u8) -> Plane {
    let mut data = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            data.push(f(x, y));
        }
    }
    Plane { w, h, data }
}

/// Lines of word-like bars at a 30 px pitch whose centres follow
/// `y = base + bow(x, base)`, on light paper.
pub fn curved_text(w: usize, h: usize, bow: impl Fn(f64, f64) -> f64) -> Plane {
    let mut data = vec![235u8; w * h];
    for line in 0..((h - 120) / 30) {
        let base = 30.0f64.mul_add(to_f64(line), 60.0);
        for x in 40..w - 40 {
            if (x / 37) % 6 == 5 {
                continue; // word space
            }
            let y0 = floor_index(base + bow(to_f64(x), base) - 6.0);
            for y in y0..(y0 + 12).min(h) {
                if x % 7 != 0 {
                    data[y * w + x] = 25;
                }
            }
        }
    }
    Plane { w, h, data }
}

/// The plane as a tightly packed gray bitmap.
pub fn to_bitmap(p: &Plane) -> raster::Bitmap<color::Gray8> {
    let w = u32::try_from(p.w).expect("test plane width");
    let h = u32::try_from(p.h).expect("test plane height");
    let mut bmp = raster::Bitmap::<color::Gray8>::new(w, h, 1, false);
    bmp.data_mut().copy_from_slice(&p.data);
    bmp
}
