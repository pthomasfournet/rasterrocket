//! The source samples a device pixel reads and their blend.
//!
//! [`Taps`] name the one, two or four samples around a pixel centre's
//! source coordinate and the weight of the second sample along each axis,
//! out of `1 << WEIGHT_BITS`.  [`Taps::blend_px`] forms their weighted mean
//! in integer arithmetic, rounded to nearest; [`BlendedRows`] forms the same
//! means a row at a time for a separable grid, with identical results.

use color::{Sample, Wide};

use super::{HALF_SAMPLE, WEIGHT_BITS};

/// The source samples one device pixel reads and their weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::renderer::page) struct Taps {
    /// Left and right source columns; equal when the column is not blended.
    x: [u32; 2],
    /// Upper and lower source rows; equal when the row is not blended.
    y: [u32; 2],
    /// Weight of `x[1]`, out of `1 << WEIGHT_BITS`.
    wx: u32,
    /// Weight of `y[1]`, out of `1 << WEIGHT_BITS`.
    wy: u32,
}

impl Taps {
    /// The taps whose columns and weight are `x` and whose rows and weight
    /// are `y`.
    #[inline]
    pub(super) const fn from_axes(x: AxisTaps, y: AxisTaps) -> Self {
        Self {
            x: x.0,
            y: y.0,
            wx: x.1,
            wy: y.1,
        }
    }

    /// The single sample `(x, y)`.
    #[cfg(test)]
    pub(super) const fn point(x: u32, y: u32) -> Self {
        Self {
            x: [x, x],
            y: [y, y],
            wx: 0,
            wy: 0,
        }
    }

    /// The weighted mean over the taps of each of the `N` channels
    /// `fetch(x, y)` returns, rounded to nearest; each tap is fetched once.
    #[inline]
    pub(in crate::renderer::page) fn blend_px<S: Sample, const N: usize>(
        &self,
        fetch: impl Fn(u32, u32) -> [S; N],
    ) -> [S; N] {
        if self.wx == 0 && self.wy == 0 {
            return fetch(self.x[0], self.y[0]);
        }
        let row = |y| horizontal_blend(fetch(self.x[0], y), fetch(self.x[1], y), self.wx);
        vertical_blend(&row(self.y[0]), &row(self.y[1]), self.wy)
    }
}

/// `left × (1 − wx) + right × wx` per channel, in units of `2^-WEIGHT_BITS`:
/// a row's horizontal blend before the vertical one.  The sums are formed
/// in [`Sample::Wide`], which holds a sample times `2^(2·WEIGHT_BITS)`.
#[inline]
fn horizontal_blend<S: Sample, const N: usize>(
    left: [S; N],
    right: [S; N],
    wx: u32,
) -> [S::Wide; N] {
    let w = <S::Wide as Wide>::of;
    let one = 1 << WEIGHT_BITS;
    std::array::from_fn(|ch| left[ch].wide() * w(one - wx) + right[ch].wide() * w(wx))
}

/// The blend of two horizontal blends `top` and `bottom` with `bottom`
/// weighted `wy`, rounded to nearest: a weighted mean of samples with
/// weights summing to `2^(2·WEIGHT_BITS)`, so a sample after the shift.
#[inline]
fn vertical_blend<S: Sample, const N: usize>(
    top: &[S::Wide; N],
    bottom: &[S::Wide; N],
    wy: u32,
) -> [S; N] {
    let w = <S::Wide as Wide>::of;
    let one = 1 << WEIGHT_BITS;
    std::array::from_fn(|ch| {
        let sum = top[ch] * w(one - wy) + bottom[ch] * w(wy);
        S::from_wide((sum + w(1 << (2 * WEIGHT_BITS - 1))) >> (2 * WEIGHT_BITS))
    })
}

/// The taps of one axis: the two samples and the weight of the second.
pub(in crate::renderer::page) type AxisTaps = ([u32; 2], u32);

/// Whether integer coordinate `i` lies on an axis of `n` samples.
pub(super) fn on_axis(i: i32, n: u32) -> bool {
    u32::try_from(i).is_ok_and(|i| i < n)
}

/// The first `N` samples of pixel `(x, y)` of an image `width` pixels wide
/// stored at `bpp` samples per pixel, or the largest sample in every
/// channel when the pixel lies past the end of `samples`.
#[inline]
pub(in crate::renderer::page) fn sample_bytes<S: Sample, const N: usize>(
    samples: &[S],
    width: usize,
    bpp: usize,
    x: u32,
    y: u32,
) -> [S; N] {
    let at = (y as usize * width + x as usize) * bpp;
    samples
        .get(at..)
        .and_then(<[S]>::first_chunk::<N>)
        .copied()
        .unwrap_or([S::MAX; N])
}

/// The taps of a [`SampleGrid`](super::SampleGrid) whose source x depends on the device column
/// alone and whose source y depends on the device row alone, as one entry
/// per column, combined with a row's taps by [`Self::taps`].  Equal to
/// [`SampleGrid::sample`](super::SampleGrid::sample) at every pixel.
pub(in crate::renderer::page) struct SeparableTaps {
    /// Per bounding-box column: its x taps, or `None` when its source x
    /// lies off the image.
    pub(super) cols: Vec<Option<AxisTaps>>,
}

/// A source row blended horizontally at every bounding-box column.
type HorizontalRow<const N: usize, S> = Vec<[<S as Sample>::Wide; N]>;

/// [`Taps::blend_px`] of a whole row of a separable grid, in two passes:
/// each source row is blended horizontally once, at every bounding-box
/// column, and kept while the device rows that read it are produced; each
/// device row is then the vertical blend of its two source rows.  The
/// arithmetic is [`Taps::blend_px`]'s, so the samples are identical.
pub(in crate::renderer::page) struct BlendedRows<'a, const N: usize, S: Sample = u8> {
    cols: &'a SeparableTaps,
    /// Two horizontally blended source rows, each with its row index;
    /// `None` before the slot is first filled.
    slots: [(Option<u32>, HorizontalRow<N, S>); 2],
    /// The samples of the last device row blended.
    out: Vec<[S; N]>,
}

impl<'a, const N: usize, S: Sample> BlendedRows<'a, N, S> {
    pub(in crate::renderer::page) fn new(cols: &'a SeparableTaps) -> Self {
        let n = cols.cols.len();
        let zero = [<S::Wide as Wide>::ZERO; N];
        Self {
            cols,
            slots: [(None, vec![zero; n]), (None, vec![zero; n])],
            out: vec![[S::ZERO; N]; n],
        }
    }

    /// The samples of the device row whose y taps are `row`, one per
    /// bounding-box column; `fetch(x, y)` reads one source sample.  A
    /// column whose source x lies off the image reads 0.
    pub(in crate::renderer::page) fn blend_row(
        &mut self,
        row: AxisTaps,
        fetch: impl Fn(u32, u32) -> [S; N],
    ) -> &[[S; N]] {
        let ([upper, lower], wy) = row;
        // Slot 0 holds the upper row and, when it is weighted, slot 1 the
        // lower; a slot already holding either row is kept, so a draw
        // stepping through the source rows in either direction blends each
        // row once.
        let lower_needed = wy != 0;
        let holds = |slot: usize, y: u32| self.slots[slot].0 == Some(y);
        if !holds(0, upper) && (holds(1, upper) || (lower_needed && holds(0, lower))) {
            self.slots.swap(0, 1);
        }
        if self.slots[0].0 != Some(upper) {
            self.fill(0, upper, &fetch);
        }
        if lower_needed && self.slots[1].0 != Some(lower) {
            self.fill(1, lower, &fetch);
        }
        let [(_, top), (_, bottom)] = &self.slots;
        // An unweighted lower row contributes nothing.
        let bottom = if lower_needed { bottom } else { top };
        for ((o, t), b) in self.out.iter_mut().zip(top).zip(bottom) {
            *o = vertical_blend(t, b, wy);
        }
        &self.out
    }

    /// Blend source row `y` horizontally into slot `slot`.
    fn fill(&mut self, slot: usize, y: u32, fetch: &impl Fn(u32, u32) -> [S; N]) {
        let (held, blended) = &mut self.slots[slot];
        for (h, col) in blended.iter_mut().zip(&self.cols.cols) {
            *h = match *col {
                Some(([left, right], wx)) => {
                    let l = fetch(left, y);
                    let r = if wx == 0 { l } else { fetch(right, y) };
                    horizontal_blend(l, r, wx)
                }
                None => [<S::Wide as Wide>::ZERO; N],
            };
        }
        *held = Some(y);
    }
}

impl SeparableTaps {
    /// Taps of the pixel in bounding-box column `col` of a row whose y taps
    /// are `row`; `None` when the pixel centre falls outside the image.
    #[inline]
    pub(in crate::renderer::page) fn taps(&self, col: usize, row: AxisTaps) -> Option<Taps> {
        Some(Taps::from_axes((*self.cols.get(col)?)?, row))
    }
}

/// Taps along one axis of `n` samples at integer part `i` and fraction
/// `frac` of a pixel centre's source coordinate, with `i` clamped onto the
/// axis.  When `blend` is set and `i` lies on the axis, the taps are the
/// samples whose centres bracket the coordinate, clamped to the axis, and
/// the weight is the coordinate's position between them; otherwise the
/// single sample `i`.
#[inline]
#[expect(
    clippy::cast_sign_loss,
    reason = "each index is clamped to be non-negative before the u32 cast"
)]
pub(super) fn axis_taps(i: i32, frac: u32, n: u32, blend: bool) -> AxisTaps {
    let last = n - 1;
    if i < 0 {
        return ([0, 0], 0);
    }
    let i_u = i as u32;
    if i_u > last {
        return ([last, last], 0);
    }
    if !blend {
        return ([i_u, i_u], 0);
    }
    // Measured from sample centres: `i + frac − ½`, floored, with its
    // fraction's top bits as the weight of the upper sample.
    let lower = i - i32::from(frac < HALF_SAMPLE);
    let weight = frac.wrapping_sub(HALF_SAMPLE) >> (32 - WEIGHT_BITS);
    let lo = lower.max(0) as u32;
    let hi = ((lower + 1) as u32).min(last);
    // Past the outer sample centres both taps clamp to the edge sample.
    ([lo, hi], if lo == hi { 0 } else { weight })
}

#[cfg(test)]
mod tests {
    use super::super::{Filter, GridView, SampleGrid};
    use super::*;
    use crate::renderer::gstate::Ctm;

    /// Clamping keeps a rounding-edge centre on the nearest mask pixel.
    #[test]
    fn sample_clamped_stays_on_the_image() {
        let ctm = [-200.0, 0.0, 0.0, -100.0, 250.0, 150.0];
        let grid =
            SampleGrid::new(&ctm, GridView::whole(300, 200), 3, 3, Filter::NEAREST).expect("grid");
        for dy in grid.y0..grid.y1 {
            for dx in grid.x0..grid.x1 {
                let taps = grid.sample_clamped(dx, dy);
                assert!(taps.x.iter().chain(&taps.y).all(|&v| v < 3));
                if let Some(p) = grid.sample(dx, dy) {
                    assert_eq!(p, taps);
                }
            }
        }
        // Device pixels outside the bounding box clamp onto its edge.
        assert_eq!(
            grid.sample_clamped(0, 0),
            grid.sample_clamped(grid.x0, grid.y0)
        );
        assert_eq!(
            grid.sample_clamped(299, 199),
            grid.sample_clamped(grid.x1 - 1, grid.y1 - 1)
        );
    }

    /// Interpolated taps bracket the pixel centre between sample centres,
    /// weighted by its position, and clamp to the edge samples outside them.
    #[test]
    fn bilinear_taps_ramp_between_sample_centres() {
        // Two samples over 100 px: centres at device x 25 and 75.
        let ctm = [100.0, 0.0, 0.0, 10.0, 0.0, 0.0];
        let filter = Filter { x: true, y: false };
        let grid = SampleGrid::new(&ctm, GridView::whole(100, 10), 2, 1, filter).expect("grid");
        let taps = |dx| grid.sample(dx, 5).expect("inside");
        assert_eq!(taps(10), Taps::point(0, 0));
        assert_eq!(taps(90), Taps::point(1, 0));
        let mid = taps(49);
        assert_eq!((mid.x, mid.wx), ([0, 1], 125));
        let ramp: Vec<u8> = (0..100)
            .map(|dx| taps(dx).blend_px(|x, _| [[0, 255][x as usize]])[0])
            .collect();
        assert!(ramp.windows(2).all(|p| p[0] <= p[1]));
        assert_eq!((ramp[24], ramp[75]), (0, 255));
        assert!((126..=130).contains(&ramp[50]), "{}", ramp[50]);
    }

    /// At one sample per pixel on an integer grid, every pixel centre is a
    /// sample centre: interpolation reproduces the samples exactly.
    #[test]
    fn bilinear_at_one_to_one_is_a_pure_copy() {
        let ctm = [40.0, 0.0, 0.0, 30.0, 10.0, 20.0];
        let filter = Filter { x: true, y: true };
        let grid = SampleGrid::new(&ctm, GridView::whole(100, 100), 40, 30, filter).expect("grid");
        for dy in 50..80 {
            for dx in 10..50 {
                let taps = grid.sample(dx, dy).expect("inside");
                assert_eq!((taps.wx, taps.wy), (0, 0));
                assert_eq!((taps.x[0], taps.y[0]), (dx - 10, dy - 50));
            }
        }
    }

    #[test]
    fn blend_rounds_to_nearest() {
        let taps = Taps {
            x: [0, 1],
            y: [0, 0],
            wx: 128,
            wy: 0,
        };
        assert_eq!(taps.blend_px(|x, _| [[0u8, 255][x as usize]]), [128]);
        let corner = Taps {
            x: [0, 1],
            y: [0, 1],
            wx: 64,
            wy: 192,
        };
        // 0.25 · 0.75 of the (1, 1) sample.
        let [v] = corner.blend_px(|x, y| [if (x, y) == (1, 1) { 255u8 } else { 0 }]);
        assert_eq!(v, 48);
    }

    #[test]
    fn separable_taps_equal_per_pixel_taps() {
        // Scales above and below one sample per pixel, flips, fractional
        // offsets, and an image overhanging the page edges.
        let ctms: [Ctm; 6] = [
            [213.4, 0.0, 0.0, 331.7, 12.25, 40.5],
            [40.0, 0.0, 0.0, 30.0, 3.3, 7.9],
            [-97.0, 0.0, 0.0, 120.0, 150.2, -10.0],
            [80.0, 0.0, 0.0, -64.0, -20.6, 90.1],
            [300.0, 0.0, 0.0, 300.0, -50.0, -50.0],
            [17.0, 0.0, 0.0, 9.0, 60.0, 60.0],
        ];
        for ctm in &ctms {
            for (img_w, img_h) in [(37, 53), (400, 300), (1, 1)] {
                for filter in [
                    Filter::NEAREST,
                    Filter { x: true, y: true },
                    Filter { x: true, y: false },
                ] {
                    let grid =
                        SampleGrid::new(ctm, GridView::whole(200, 160), img_w, img_h, filter)
                            .unwrap();
                    let cols = grid.separable().expect("an unrotated draw is separable");
                    for dy in grid.y0..grid.y1 {
                        let row = grid.row_taps(dy);
                        for (col, dx) in (0..).zip(grid.x0..grid.x1) {
                            let fast = row.and_then(|r| cols.taps(col, r));
                            assert_eq!(
                                fast,
                                grid.sample(dx, dy),
                                "{ctm:?} {img_w}x{img_h} ({dx},{dy})"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Every sample [`BlendedRows`] produces equals [`Taps::blend_px`] of
    /// the pixel's own taps: draws that magnify (rows reused), reduce (rows
    /// skipped), flip (rows stepped backwards), sample one axis nearest,
    /// and cover a single-sample image, each blended at every row and at
    /// every third row (as a clip that admits only some rows does).
    #[test]
    fn blended_rows_equal_per_pixel_blends() {
        let rgb = |x: u32, y: u32| {
            [0_u32, 1, 2].map(|ch| u8::try_from((x * 89 + y * 37 + ch * 113) % 256).unwrap())
        };
        let gray = |x: u32, y: u32| [rgb(x, y)[1]];
        let both = Filter { x: true, y: true };
        let cases: [(Ctm, Filter, u32, u32); 6] = [
            ([90.0, 0.0, 0.0, 70.0, 5.3, 8.1], both, 23, 17),
            ([15.0, 0.0, 0.0, 11.0, 20.0, 20.0], Filter::NEAREST, 23, 17),
            ([-80.0, 0.0, 0.0, -60.0, 95.0, 75.0], both, 23, 17),
            ([80.0, 0.0, 0.0, -60.0, 5.0, 75.0], both, 23, 17),
            (
                [70.0, 0.0, 0.0, 90.0, 2.0, 1.0],
                Filter { x: false, y: true },
                23,
                17,
            ),
            ([40.0, 0.0, 0.0, 30.0, 10.0, 10.0], both, 1, 1),
        ];
        for (ctm, filter, img_w, img_h) in &cases {
            let grid =
                SampleGrid::new(ctm, GridView::whole(120, 100), *img_w, *img_h, *filter).unwrap();
            let cols = grid.separable().unwrap();
            for step in [1, 3] {
                let mut rgb_rows = BlendedRows::<3>::new(&cols);
                let mut gray_rows = BlendedRows::<1>::new(&cols);
                for dy in (grid.y0..grid.y1).step_by(step) {
                    let Some(row) = grid.row_taps(dy) else {
                        continue;
                    };
                    let rgb_out = rgb_rows.blend_row(row, rgb).to_vec();
                    let gray_out = gray_rows.blend_row(row, gray);
                    for (col, dx) in (0..).zip(grid.x0..grid.x1) {
                        if let Some(taps) = grid.sample(dx, dy) {
                            assert_eq!(rgb_out[col], taps.blend_px(rgb), "{ctm:?} ({dx},{dy})");
                            assert_eq!(gray_out[col], taps.blend_px(gray), "{ctm:?} ({dx},{dy})");
                        }
                    }
                }
            }
        }
    }
}
