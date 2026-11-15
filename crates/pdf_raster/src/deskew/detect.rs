//! Skew angle detection by an intensity-weighted projection profile.
//!
//! The foreground weight of a pixel is `255 − pixel` on the 8-bit gray
//! image: dark pixels contribute in proportion to their darkness, so no
//! binarisation threshold enters the measurement.
//!
//! # Algorithm
//!
//! 1. **Reduce** the image 2× and 4× by box averaging.
//! 2. **Sweep** the 57 angles `k · 0.25°`, `|k| ≤ 28`, that span
//!    ±[`MAX_SEARCH_DEG`], on the 4× image, Rayon-parallel.  The sweep is
//!    symmetric about 0°, so a page and its vertical mirror image score
//!    the same at opposite angles.
//! 3. **Confidence**: the ratio of the best to the worst sweep score, 0
//!    when the worst score lies below [`WORST_SCORE_FLOOR`] per squared
//!    column (a blank or uniform page, whose profile is flat at every
//!    angle).  A page whose confidence is below [`MIN_CONFIDENCE`] has no
//!    reliable angle: its profile varies little with the angle, as for a
//!    photograph or noise.
//! 4. **Refine** on the 2× image: three-point interval halving around the
//!    best sweep angle until the half-width falls below
//!    [`REFINE_STOP_DEG`].
//! 5. **Reject** a refined angle within one sweep step of either end of
//!    the sweep: the score there may still rise beyond the range searched.
//!
//! # Score function
//!
//! For a trial angle θ, pixel `(x, y)` is projected onto row
//! `y + round(x · tan θ) − round(w/2 · tan θ)` of an image `w` pixels
//! wide: a shear about the vertical centre line, whose offsets change
//! column by column as those of a shear about the left edge do.  The
//! projection of a page turned upside down at −θ is that of the page at θ
//! reversed, so the two score the same.  Only the rows to which every column
//! contributes at every searched angle are scored, the same window at
//! every angle, so the sheared edges of the page, where a dark or
//! textured page would ramp up and down at a slope that grows with the
//! angle, add nothing.  The score is the differential square sum
//! `Σ (s[i] − s[i−1])²` of the projected row sums `s` over that window,
//! which is largest when text lines are parallel to the projection.  The
//! window excludes about `0.06 · w` rows at the top and at the bottom of
//! the page, and is empty on a page more than about 8 times wider than it
//! is tall, which then has no reliable angle.
//!
//! # Accuracy
//!
//! The projection changes when the shear offset of the last column changes
//! by one row, so on an image `w` pixels wide the score resolves angles of
//! about `1 / w` radians.  Refinement at the 2× reduction resolves about
//! `2 / W` radians on a page `W` pixels wide, a climb of two pixels across
//! the page: 0.05° for an A4 page at 300 dpi (2480 px), 0.09° at 150 dpi.
//! A page tilted by less than that may measure as straight.  Pages
//! narrower than about 1200 px measure less reliably.

//! # Cost
//!
//! The inner loop scatters weights into a `u32` row-sum array at
//! data-dependent indices, scalar code that does not vectorise.  The sweep
//! angles run in parallel; each refinement step evaluates its two new
//! angles in parallel.

use color::Gray8;
use raster::Bitmap;
use rayon::prelude::*;

use super::{MAX_SEARCH_DEG, MIN_CONFIDENCE, Skew};

/// Angle between adjacent sweep angles, in degrees.
const SWEEP_STEP_DEG: f32 = 0.25;

/// Sweep angles on either side of 0°: `MAX_SEARCH_DEG / SWEEP_STEP_DEG`.
const SWEEP_HALF: i32 = 28;

#[expect(
    clippy::cast_precision_loss,
    reason = "SWEEP_HALF = 28 is exact in f32"
)]
const _: () = assert!(
    SWEEP_HALF as f32 * SWEEP_STEP_DEG == MAX_SEARCH_DEG,
    "the sweep spans exactly ±MAX_SEARCH_DEG"
);

/// Refinement stops when the half-width of its interval falls below this.
const REFINE_STOP_DEG: f32 = 0.01;

/// The worst sweep score, per squared column of the 4× image, below which
/// the confidence is 0: `2 · 10⁻⁶` squared full-weight (255) units.
const WORST_SCORE_FLOOR: f64 = 2.0e-6 * 255.0 * 255.0;

/// Smallest side, in pixels, of an image whose skew is measured.
const MIN_SIDE: u32 = 32;

/// The skew of `img`, or `None` when it has no reliable angle: a side
/// below `MIN_SIDE` px, a confidence below [`MIN_CONFIDENCE`], or a best
/// angle within one sweep step of ±[`MAX_SEARCH_DEG`].
#[must_use]
pub(crate) fn detect(img: &Bitmap<Gray8>) -> Option<Skew> {
    if img.width < MIN_SIDE || img.height < MIN_SIDE {
        return None;
    }
    let mid = downsample(img);
    let small = downsample(&mid);

    let scores: Vec<(f32, f64)> = (-SWEEP_HALF..=SWEEP_HALF)
        .into_par_iter()
        .map(|k| {
            #[expect(clippy::cast_precision_loss, reason = "|k| ≤ 28; exact in f32")]
            let deg = k as f32 * SWEEP_STEP_DEG;
            (deg, score_angle(&small, deg))
        })
        .collect();
    let (best_deg, best) = scores
        .iter()
        .copied()
        .reduce(|a, b| if better(b, a) { b } else { a })?;
    let worst = scores.iter().map(|&(_, s)| s).fold(f64::INFINITY, f64::min);

    let floor = WORST_SCORE_FLOOR * f64::from(small.width) * f64::from(small.width);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a ratio of scores; f32 holds it to well within the confidence's use"
    )]
    let confidence = if worst < floor {
        0.0
    } else {
        (best / worst) as f32
    };
    if confidence < MIN_CONFIDENCE {
        return None;
    }

    let angle_deg = refine(&mid, best_deg, SWEEP_STEP_DEG / 2.0);
    if angle_deg.abs() > MAX_SEARCH_DEG - SWEEP_STEP_DEG {
        return None;
    }
    Some(Skew {
        angle_deg,
        confidence,
    })
}

/// Whether `(deg, score)` `a` beats `b`: a higher score, else on equal
/// scores the smaller magnitude, else the positive angle.  The order does
/// not depend on the order of evaluation.
fn better(a: (f32, f64), b: (f32, f64)) -> bool {
    a.1.total_cmp(&b.1)
        .then_with(|| b.0.abs().total_cmp(&a.0.abs()))
        .then_with(|| a.0.total_cmp(&b.0))
        .is_gt()
}

/// Three-point interval halving around `centre` on `img`: the best of
/// `centre − delta`, `centre` and `centre + delta` (within
/// ±[`MAX_SEARCH_DEG`]) becomes the centre and `delta` halves, until
/// `delta < REFINE_STOP_DEG`.
fn refine(img: &Bitmap<Gray8>, mut centre: f32, mut delta: f32) -> f32 {
    let mut centre_score = score_angle(img, centre);
    #[expect(
        clippy::while_float,
        reason = "delta halves each iteration from a positive start, so the loop ends"
    )]
    while delta >= REFINE_STOP_DEG {
        let left = (centre - delta).max(-MAX_SEARCH_DEG);
        let right = (centre + delta).min(MAX_SEARCH_DEG);
        let (sl, sr) = rayon::join(|| score_angle(img, left), || score_angle(img, right));
        let mut best = (centre, centre_score);
        for candidate in [(left, sl), (right, sr)] {
            if better(candidate, best) {
                best = candidate;
            }
        }
        (centre, centre_score) = best;
        delta *= 0.5;
    }
    centre
}

/// The differential square sum of the row projections of `img` sheared by
/// `deg` degrees; see the module documentation.
fn score_angle(img: &Bitmap<Gray8>, deg: f32) -> f64 {
    // Sides are at most DESKEW_MAX_SIDE / 2 = 16384 after the 2× reduction.
    let w = img.width as usize;
    let h = img.height as usize;
    let tan = deg.to_radians().tan();
    #[expect(clippy::cast_precision_loss, reason = "w ≤ 16384 is exact in f32")]
    let cx = w as f32 / 2.0;
    let margin = window_margin(cx);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "|cx · tan| < 1100 fits in isize"
    )]
    let shift = (cx * tan).round() as isize;
    let Some(rows) = h.checked_sub(2 * margin).filter(|&n| n >= 2) else {
        return 0.0;
    };

    // Row i of the window sums the pixels (x, y) with
    // y + round(x · tan) − shift = margin + i; |round(x · tan) − shift| is
    // at most cx · |tan| + 1 ≤ margin, so every column reaches every row
    // of the window.
    let mut sums = vec![0u32; rows];
    for y in 0..h {
        #[expect(clippy::cast_possible_truncation, reason = "y < h = img.height, a u32")]
        let row = &img.row_bytes(y as u32)[..w];
        for (x, &px) in row.iter().enumerate() {
            let weight = u32::from(255 - px);
            if weight == 0 {
                continue;
            }
            #[expect(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                reason = "x < 16384 is exact in f32; |round(x · tan)| < 2100 fits in isize"
            )]
            let offset = (x as f32 * tan).round() as isize - shift;
            if let Some(slot) = y
                .checked_add_signed(offset)
                .and_then(|r| r.checked_sub(margin))
                .and_then(|i| sums.get_mut(i))
            {
                *slot += weight;
            }
        }
    }

    sums.windows(2)
        .map(|pair| {
            let diff = i64::from(pair[1]) - i64::from(pair[0]);
            #[expect(
                clippy::cast_precision_loss,
                reason = "diff² ≤ (255 · 16384)² < 2^45 is exact in f64; a score needs no more than f64's relative precision"
            )]
            let sq = (diff * diff) as f64;
            sq
        })
        .sum()
}

/// Rows dropped at the top and at the bottom of the projection window on
/// an image of half-width `cx`, `⌈cx · tan MAX_SEARCH_DEG⌉ + 1`: the window
/// holds only rows to which every column contributes at every angle within
/// ±[`MAX_SEARCH_DEG`], so the sheared page edges add nothing to any score.
fn window_margin(cx: f32) -> usize {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "cx · tan 7° < 1100 is non-negative and fits in usize"
    )]
    let margin = (cx * MAX_SEARCH_DEG.to_radians().tan()).ceil() as usize;
    margin + 1
}

/// The 2× box-average reduction of `src`: each output pixel the truncated
/// mean of a 2 × 2 block; a last odd row or column is dropped.
fn downsample(src: &Bitmap<Gray8>) -> Bitmap<Gray8> {
    let (ow, oh) = (src.width / 2, src.height / 2);
    let mut dst = Bitmap::<Gray8>::new(ow.max(1), oh.max(1), 1, false);
    for oy in 0..oh {
        let (top, bottom) = (src.row_bytes(2 * oy), src.row_bytes(2 * oy + 1));
        let dst_row = &mut dst.row_bytes_mut(oy)[..ow as usize];
        for ((px, t), b) in dst_row
            .iter_mut()
            .zip(top.as_chunks::<2>().0)
            .zip(bottom.as_chunks::<2>().0)
        {
            let sum = u16::from(t[0]) + u16::from(t[1]) + u16::from(b[0]) + u16::from(b[1]);
            #[expect(clippy::cast_possible_truncation, reason = "sum / 4 ≤ 255")]
            {
                *px = (sum / 4) as u8;
            }
        }
    }
    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A blank page and a uniformly dark page score 0 at the unsheared
    /// angle, so their confidence is 0.
    #[test]
    fn flat_pages_have_no_reliable_angle() {
        for fill in [0, 128, 250, 255] {
            let img = Bitmap::<Gray8>::new_filled(256, 256, 1, false, fill);
            assert_eq!(detect(&img), None, "fill {fill}");
        }
    }

    #[test]
    fn a_tiny_image_has_no_reliable_angle() {
        let img = Bitmap::<Gray8>::new(MIN_SIDE - 1, 64, 1, false);
        assert_eq!(detect(&img), None);
    }

    /// The projection index stays in range at both ends of the sweep, on
    /// an image whose ink reaches every edge.
    #[test]
    fn score_angle_holds_every_pixel_at_the_sweep_ends() {
        let img = Bitmap::<Gray8>::new(64, 64, 1, false);
        for deg in [-MAX_SEARCH_DEG, 0.0, MAX_SEARCH_DEG] {
            assert!(score_angle(&img, deg).is_finite());
        }
    }

    #[test]
    fn the_reduction_halves_each_side_and_averages_blocks() {
        let mut src = Bitmap::<Gray8>::new(5, 3, 1, false);
        src.row_bytes_mut(0)[..5].copy_from_slice(&[0, 4, 8, 8, 9]);
        src.row_bytes_mut(1)[..5].copy_from_slice(&[4, 4, 8, 9, 9]);
        let dst = downsample(&src);
        assert_eq!((dst.width, dst.height), (2, 1));
        assert_eq!(&dst.row_bytes(0)[..2], &[3, 8]);
    }

    /// Equal scores resolve to the smaller magnitude, then the positive
    /// angle, whichever is evaluated first.
    #[test]
    fn ties_resolve_independently_of_order() {
        let (a, b, c) = ((0.25, 1.0), (-0.25, 1.0), (0.5, 1.0));
        assert!(better(a, b) && !better(b, a));
        assert!(better(a, c) && !better(c, a));
        assert!(better((1.0, 2.0), a));
    }
}
