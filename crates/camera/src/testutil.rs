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

/// FNV-1a 64 of the dimensions and samples of `pages`: a golden digest of
/// resampled output.
pub(crate) fn digest(pages: &[GrayPage]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |b: u8| {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    };
    for p in pages {
        p.width.to_le_bytes().into_iter().for_each(&mut eat);
        p.height.to_le_bytes().into_iter().for_each(&mut eat);
        p.data.iter().copied().for_each(&mut eat);
    }
    h
}

/// A light page tilted by about 4° with text bars, on a darker desk.
pub(crate) fn tilted_page(w: usize, h: usize) -> GrayPage {
    let (cx, cy) = (to_f64(w) / 2.0, to_f64(h) / 2.0);
    let (sin, cos) = 0.07f64.sin_cos();
    plane(w, h, |x, y| {
        let (dx, dy) = (to_f64(x) - cx, to_f64(y) - cy);
        let (along, across) = (cos.mul_add(dx, sin * dy), cos.mul_add(dy, -sin * dx));
        let (pw, ph) = (to_f64(w) * 0.36, to_f64(h) * 0.38);
        if along.abs() > pw || across.abs() > ph {
            70
        } else if along.abs() < pw - 40.0
            && across.abs() < ph - 40.0
            && (floor_index(across + ph) / 8).is_multiple_of(3)
            && floor_index(along + pw) / 19 % 5 != 4
        {
            25
        } else {
            238
        }
    })
}
