//! Synthetic images shared by the unit tests.

use crate::image::GrayPage;
use crate::num::{floor_index, to_f64};

/// A `w × h` page with pixel `(x, y) = f(x, y)`.
pub(crate) fn plane(w: usize, h: usize, f: impl Fn(usize, usize) -> u8) -> GrayPage {
    let mut data = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            data.push(f(x, y));
        }
    }
    GrayPage {
        width: w,
        height: h,
        data,
    }
}

/// Lines of word-like bars at a 30 px pitch whose centres follow
/// `y = base + bow(x, base)`, on light paper.
pub(crate) fn curved_text(w: usize, h: usize, bow: impl Fn(f64, f64) -> f64) -> GrayPage {
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
    GrayPage {
        width: w,
        height: h,
        data,
    }
}
