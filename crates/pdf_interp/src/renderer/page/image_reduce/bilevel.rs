//! The resampling of bilevel samples ([`Bilevel`]), read from their bits.
//!
//! A source sample is one of two values, `set` or `clear`, so the vertical
//! pass's sum over the taps of an output row is
//! `Σ wtⱼ·vⱼ = clear·W + (set − clear)·B`, where `W` is the sum of the
//! row's weights and `B` the sum of the weights of the taps whose bit is
//! set.  The sum is the one [`Plan::resample`] makes from the samples as
//! bytes, exactly: its true value fits `u32`, so the wrapping arithmetic
//! that forms it (with `set < clear`) lands on it.  The horizontal pass is
//! [`Plan::resample`]'s own.  An output row none of whose taps has a set
//! bit in the columns read is a row of `clear`, computed once.

use std::ops::Range;

use super::{Plan, WEIGHT_BITS};
use crate::resources::image::Bilevel;

impl Plan {
    /// [`Self::resample`] of `src`'s samples as bytes, from its bits.
    /// `None` when `src` is not the plan's source size.
    pub(super) fn resample_bilevel(&self, src: &Bilevel) -> Option<Vec<u8>> {
        if (src.width(), src.height()) != (self.src_w, self.src_h) {
            return None;
        }
        let (out_w, out_h) = (self.out_w as usize, self.out_h as usize);
        if out_w == 0 || out_h == 0 {
            return Some(Vec::new());
        }
        let mut out = vec![0u8; out_w.checked_mul(out_h)?];
        let (set, clear) = (u32::from(src.set()), u32::from(src.clear()));
        let delta = set.wrapping_sub(clear);
        let width = self.src_w as usize;
        let span = self.col_span.0 as usize..self.col_span.1 as usize;
        let inked: Vec<bool> = (0..self.src_h as usize)
            .map(|y| any_set(src.row(y), &span))
            .collect();
        // `acc` is the vertical pass's row, `set_weight` its `B`.
        let mut acc = vec![0u32; width];
        let mut set_weight = vec![0u32; width];
        // The output row of a window of `clear·W` alone, for the `W` it
        // was made for.
        let mut blank: Option<(u32, Vec<u8>)> = None;

        for (oy, out_row) in out.chunks_exact_mut(out_w).enumerate() {
            let (first, weights, frac) = self.rows.as_ref().map_or((oy, &[1][..], 0), |rows| {
                let (first, weights) = rows.taps(oy);
                (first, weights, WEIGHT_BITS)
            });
            let total = weights.iter().sum::<u32>();
            let base = clear * total;
            let taps = weights
                .iter()
                .enumerate()
                .filter(|&(j, &wt)| wt != 0 && inked[first + j]);
            if taps.clone().next().is_none() {
                let row = match &blank {
                    Some((b, row)) if *b == base => row,
                    _ => {
                        acc[span.clone()].fill(base);
                        let mut row = vec![0u8; out_w];
                        self.finish_row(&mut acc, span.clone(), frac, 1, &mut row);
                        &blank.insert((base, row)).1
                    }
                };
                out_row.copy_from_slice(row);
                continue;
            }
            set_weight[span.clone()].fill(0);
            for (j, &wt) in taps {
                add_set_bits(src.row(first + j), &span, wt, &mut set_weight);
            }
            for (a, &b) in acc[span.clone()].iter_mut().zip(&set_weight[span.clone()]) {
                *a = base.wrapping_add(delta.wrapping_mul(b));
            }
            self.finish_row(&mut acc, span.clone(), frac, 1, out_row);
        }
        Some(out)
    }
}

/// The words of a row holding the samples `span`, each with the bits
/// outside `span` cleared, and the sample of its first bit.
fn span_words<'a>(row: &'a [u64], span: &Range<usize>) -> impl Iterator<Item = (usize, u64)> + 'a {
    let (start, end) = (span.start, span.end);
    let words = start / 64..end.div_ceil(64);
    row[words.clone()].iter().zip(words).map(move |(&word, i)| {
        let mut word = word;
        if i == start / 64 {
            word &= u64::MAX >> (start % 64);
        }
        if i == (end - 1) / 64 && end % 64 != 0 {
            word &= u64::MAX << (64 - end % 64);
        }
        (i * 64, word)
    })
}

/// Whether a bit of `row` within `span` is set.
fn any_set(row: &[u64], span: &Range<usize>) -> bool {
    !span.is_empty() && span_words(row, span).any(|(_, word)| word != 0)
}

/// `weights[x] += wt` for every set bit `x` of `row` within `span`.
fn add_set_bits(row: &[u64], span: &Range<usize>, wt: u32, weights: &mut [u32]) {
    if span.is_empty() {
        return;
    }
    for (x0, mut word) in span_words(row, span) {
        if word.count_ones() > SPARSE_BITS {
            // Every lane of the word, its set bits weighted.
            let end = weights.len().min(x0 + 64);
            for (i, w) in weights[x0..end].iter_mut().enumerate() {
                *w += wt & 0u32.wrapping_sub(u32::from(word >> (63 - i) & 1 != 0));
            }
            continue;
        }
        while word != 0 {
            let bit = word.trailing_zeros() as usize;
            weights[x0 + 63 - bit] += wt;
            word &= word - 1;
        }
    }
}

/// A word with more set bits than this adds its weight to all 64 lanes,
/// masked, instead of bit by bit.
const SPARSE_BITS: u32 = 8;

#[cfg(test)]
mod tests {
    use super::super::{DeviceWindow, Plan};
    use crate::resources::image::Bilevel;

    /// A deterministic pseudo-random bilevel image: runs of set and clear
    /// bits, a blank band, and an inked band.
    fn image(width: u32, height: u32, seed: u64, set: u8, clear: u8) -> Bilevel {
        let mut state = seed | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut b = Bilevel::new(width, height, set, clear).unwrap();
        for y in 0..height as usize {
            // Rows in the first sixth stay clear; those in the last sixth
            // are dense.
            if y * 6 < height as usize {
                continue;
            }
            let dense = y * 6 >= 5 * height as usize;
            let mut x = 0usize;
            while x < width as usize {
                let run = 1 + usize::try_from(next() % if dense { 3 } else { 40 }).unwrap();
                let end = (x + run).min(width as usize);
                if next() % if dense { 2 } else { 4 } == 0 {
                    b.set_run(y, x, end);
                }
                x = end;
            }
        }
        b
    }

    /// The bilevel resampling equals the byte resampling of the expanded
    /// samples, over reduced, kept, flipped, quarter-turned, skewed and
    /// windowed draws and every pair of sample values.
    #[test]
    fn matches_the_byte_resampling() {
        let ctms: [[f64; 6]; 9] = [
            [25.0, 0.0, 0.0, 31.0, 3.3, 4.1],
            [-25.5, 0.0, 0.0, 30.2, 40.7, 2.2],
            [26.0, 0.0, 0.0, -29.0, 0.4, 50.9],
            [25.0, 0.0, 0.0, 200.0, 0.0, 0.0],
            [200.0, 0.0, 0.0, 31.0, 0.0, 0.0],
            [0.0, 27.0, -33.0, 0.0, 60.0, 1.0],
            [0.0, -27.0, 33.0, 0.0, 1.0, 60.0],
            [20.0, 9.0, -8.0, 25.0, 30.0, 5.0],
            [7.0, 0.0, 0.0, 9.0, -2.5, 1.5],
        ];
        let windows = [
            DeviceWindow::UNBOUNDED,
            DeviceWindow {
                x: (5.0, 18.0),
                y: (7.0, 20.0),
            },
        ];
        let mut compared = 0;
        for (seed, (w, h)) in [(1, (100, 130)), (2, (64, 64)), (3, (129, 77)), (4, (1, 90))] {
            for (set, clear) in [(0x00, 0xFF), (0xFF, 0x00), (0x40, 0xC0), (0xD0, 0x10)] {
                let src = image(w, h, seed, set, clear);
                let bytes = src.samples().unwrap().to_vec();
                for ctm in &ctms {
                    for window in &windows {
                        let Some(plan) = Plan::new(ctm, w, h, window) else {
                            continue;
                        };
                        assert_eq!(
                            plan.resample_bilevel(&src),
                            plan.resample(&bytes, 1),
                            "{w}×{h} seed {seed}, values {set}/{clear}, ctm {ctm:?}, {window:?}"
                        );
                        compared += 1;
                    }
                }
            }
        }
        assert!(compared >= 200, "{compared} plans compared");
    }
}
