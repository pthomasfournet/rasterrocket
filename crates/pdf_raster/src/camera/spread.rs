//! Gutter detection for double-page spreads.
//!
//! # Model
//!
//! A spread is wider than tall and its two pages meet at a gutter near the
//! centre.  Per column of a downsampled luma plane (top and bottom 5 % of
//! rows excluded) two statistics are taken:
//!
//! - the paper level `bg(x)`, the 90th luma percentile of the column —
//!   insensitive to text and rules, it follows illumination only;
//! - the ink count `ink(x)`, pixels darker than `0.6 · bg(x)`.
//!
//! The gutter is accepted on either of two signatures, searched for in the
//! central [`WINDOW`] of the width:
//!
//! - **shadow**: a valley in the smoothed `bg` at least [`MIN_SHADOW_DEPTH`]
//!   levels (and [`MIN_SHADOW_REL`] relative) below the paper level on both
//!   sides, over an ink-free column — the fold of a bound book;
//! - **margins**: an ink-free run at least [`MIN_GAP_FRAC`] of the width,
//!   bounded by ink on both sides and centred in the middle fifth — the
//!   two inner page margins.  A column gutter of a single landscape page
//!   is several times narrower and does not qualify.
//!
//! Either way both halves must carry at least [`MIN_SIDE_INK`] of the ink.

use super::plane::{Plane, floor_index, histogram, percentile, round_index, to_f64};

/// Width of the analysis plane, in pixels.
const ANALYSIS_SIDE: usize = 1200;
/// Width-to-height ratios accepted as a possible spread.
const ASPECT: std::ops::RangeInclusive<f64> = 1.15..=2.6;
/// Fractional horizontal range searched for the gutter.
const WINDOW: (f64, f64) = (0.35, 0.65);
/// Minimum absolute depth of a gutter shadow, in luma levels.
const MIN_SHADOW_DEPTH: f64 = 18.0;
/// Minimum depth of a gutter shadow relative to the paper level.
const MIN_SHADOW_REL: f64 = 0.08;
/// Minimum width of an ink-free gutter, as a fraction of the image width.
const MIN_GAP_FRAC: f64 = 0.06;
/// Minimum share of the total ink on each side of the gutter.
const MIN_SIDE_INK: f64 = 0.15;

/// The column at which `luma` splits into two pages (the left page is
/// `[0, x)`), or `None` when the image is not recognisably a spread.
pub fn find_gutter(luma: &Plane) -> Option<usize> {
    let aspect = to_f64(luma.w) / to_f64(luma.h.max(1));
    if !ASPECT.contains(&aspect) {
        return None;
    }
    let factor = luma.factor_for(ANALYSIS_SIDE);
    let small = luma.downsample(factor);
    let (w, h) = (small.w, small.h);
    if w < 64 || h < 32 {
        return None;
    }
    let (y0, y1) = (h / 20, h - h / 20);
    let rows = y1 - y0;
    let mut bg = vec![0.0f64; w];
    let mut ink = vec![0usize; w];
    for x in 0..w {
        let column = (y0..y1).map(|y| small.at(x, y));
        let level = percentile(&histogram(column.clone()), 0.9);
        bg[x] = f64::from(level);
        ink[x] = column
            .filter(|&v| u32::from(v) * 10 < u32::from(level) * 6)
            .count();
    }
    let total_ink: usize = ink.iter().sum();
    if total_ink == 0 {
        return None;
    }
    let balanced = |split: usize| {
        let left: usize = ink[..split].iter().sum();
        let min_side = MIN_SIDE_INK * to_f64(total_ink);
        to_f64(left) >= min_side && to_f64(total_ink - left) >= min_side
    };
    let lo = floor_index(WINDOW.0 * to_f64(w));
    let hi = floor_index(WINDOW.1 * to_f64(w)).min(w - 1);

    let split = shadow_valley(&bg, &ink, rows, lo, hi)
        .filter(|&x| balanced(x))
        .or_else(|| margin_gap(&ink, rows, w).filter(|&x| balanced(x)))?;
    log::debug!("camera: spread gutter at column {split} of {w} (analysis scale)");
    Some((split * factor).min(luma.w))
}

/// Column of the deepest qualifying paper-level valley in `[lo, hi]`.
fn shadow_valley(bg: &[f64], ink: &[usize], rows: usize, lo: usize, hi: usize) -> Option<usize> {
    let w = bg.len();
    let radius = (w / 300).max(1);
    let smooth: Vec<f64> = (0..w)
        .map(|x| {
            let (a, b) = (x.saturating_sub(radius), (x + radius + 1).min(w));
            bg[a..b].iter().sum::<f64>() / to_f64(b - a)
        })
        .collect();
    let x_min = (lo..=hi).min_by(|&a, &b| smooth[a].total_cmp(&smooth[b]))?;
    let reach = (w * 3 / 20).max(radius + 1);
    let left_peak = smooth[x_min.saturating_sub(reach)..=x_min]
        .iter()
        .copied()
        .fold(f64::MIN, f64::max);
    let right_peak = smooth[x_min..(x_min + reach + 1).min(w)]
        .iter()
        .copied()
        .fold(f64::MIN, f64::max);
    let paper = left_peak.min(right_peak);
    let depth = paper - smooth[x_min];
    let (a, b) = (x_min.saturating_sub(radius), (x_min + radius + 1).min(w));
    let ink_here = to_f64(ink[a..b].iter().sum::<usize>()) / to_f64(b - a);
    let ink_free = ink_here <= 0.03 * to_f64(rows);
    (depth >= MIN_SHADOW_DEPTH && depth >= MIN_SHADOW_REL * paper && ink_free).then(|| x_min + 1)
}

/// Centre of the qualifying ink-free run closest to the image centre.
fn margin_gap(ink: &[usize], rows: usize, w: usize) -> Option<usize> {
    let empty_max = (rows * 3 / 1000).max(1);
    let min_len = MIN_GAP_FRAC * to_f64(w);
    let centre = to_f64(w) / 2.0;
    let mut best: Option<(f64, usize)> = None;
    let mut x = 0;
    while x < w {
        if ink[x] > empty_max {
            x += 1;
            continue;
        }
        let start = x;
        while x < w && ink[x] <= empty_max {
            x += 1;
        }
        let end = x; // exclusive
        let mid = to_f64(start + end) / 2.0;
        let bounded = start > 0 && end < w;
        let centred = (mid - centre).abs() <= 0.1 * to_f64(w);
        if bounded && centred && to_f64(end - start) >= min_len {
            let off = (mid - centre).abs();
            if best.is_none_or(|(b, _)| off < b) {
                best = Some((off, round_index(mid)));
            }
        }
    }
    best.map(|(_, x)| x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::testutil::plane;

    /// Text-like bars in the column ranges `cols`, paper elsewhere.
    fn text_columns(
        w: usize,
        h: usize,
        cols: &[(usize, usize)],
        shade: impl Fn(usize) -> u8,
    ) -> Plane {
        plane(w, h, |x, y| {
            let in_text =
                cols.iter().any(|&(a, b)| (a..b).contains(&x)) && (h / 10..h * 9 / 10).contains(&y);
            if in_text && (y / 8) % 3 == 0 && (x / 17) % 5 != 4 {
                30
            } else {
                shade(x)
            }
        })
    }

    #[test]
    fn margin_gutter_found_at_centre() {
        let (w, h) = (1400, 1000);
        let p = text_columns(w, h, &[(80, 600), (800, 1320)], |_| 235);
        let x = find_gutter(&p).expect("gutter");
        assert!(x.abs_diff(700) <= 12, "split at {x}");
    }

    #[test]
    fn shadow_gutter_found_off_centre() {
        let (w, h) = (1400, 1000);
        let fold = 640.0;
        let p = text_columns(w, h, &[(80, 600), (680, 1320)], |x| {
            let d = (to_f64(x) - fold) / 25.0;
            super::super::plane::clamp_u8(70.0f64.mul_add(-(-d * d).exp(), 235.0))
        });
        let x = find_gutter(&p).expect("gutter");
        assert!(x.abs_diff(640) <= 6, "split at {x}");
    }

    #[test]
    fn two_column_landscape_page_is_not_split() {
        let (w, h) = (1400, 1000);
        let p = text_columns(w, h, &[(100, 680), (720, 1300)], |_| 235);
        assert!(find_gutter(&p).is_none());
    }

    #[test]
    fn portrait_page_is_not_split() {
        let p = text_columns(800, 1100, &[(80, 380), (420, 720)], |_| 235);
        assert!(find_gutter(&p).is_none());
    }

    #[test]
    fn one_sided_ink_is_not_split() {
        let p = text_columns(1400, 1000, &[(80, 600)], |_| 235);
        assert!(find_gutter(&p).is_none());
    }
}
