//! Process inks that reproduce an sRGB colour from a given set of inks.
//!
//! [`ink_reproducing`] solves the quadrilinear model of
//! [`CORNERS`](super::CORNERS) for the inks printing a target colour, with
//! two constraints that keep the inks' role on an ink canvas: an ink given
//! as 0 stays 0, so the channels the colour paints under nonzero overprint
//! mode are unchanged, and of the inks reproducing the target the ones
//! nearest the given amounts are chosen.  The cost is
//!
//! `Σ (F(x) − target)² + ANCHOR · Σ (x − given)²`
//!
//! over the free inks in `[0, 1]`, with `F` the unrounded model, minimised
//! by projected Gauss-Newton.  The model is affine in each ink, so the
//! Jacobian column of an ink is the exact difference of the model at that
//! ink solid and absent.

use super::CORNERS;

/// Gauss-Newton iterations at most.
const MAX_STEPS: usize = 40;

/// Halvings of a step in the line search.
const HALVINGS: usize = 10;

/// Weight of the pull of each ink toward its given amount, in squared
/// colour counts per squared unit of ink: a change of one byte of ink
/// weighs as 1/255 of a squared count, so the colour decides and the pull
/// selects among the inks reproducing it.
const ANCHOR: f64 = 255.0;

/// The largest channel error, in counts, of inks that print a colour:
/// the model's own fixed-point rounding.
const PRINTS: f64 = 1.0;

/// The bit of corner index `k·8 + c·4 + m·2 + y` of each ink of
/// `[c, m, y, k]`.
const SHIFT: [usize; 4] = [2, 1, 0, 3];

/// The quadrilinear weight of corner `corner` at inks `t`.
fn weight(corner: usize, t: [f64; 4]) -> f64 {
    (0..4)
        .map(|a| {
            if (corner >> SHIFT[a]) & 1 == 1 {
                t[a]
            } else {
                1.0 - t[a]
            }
        })
        .product()
}

/// The model at inks `t`, unrounded.
fn forward(t: [f64; 4]) -> [f64; 3] {
    let mut out = [0.0; 3];
    for (i, corner) in CORNERS.iter().enumerate() {
        let w = weight(i, t);
        for (o, &v) in out.iter_mut().zip(corner) {
            *o = w.mul_add(f64::from(v), *o);
        }
    }
    out
}

/// The cost of inks `x` (see the module documentation).
fn cost(x: [f64; 4], target: [f64; 3], given: [f64; 4]) -> f64 {
    let colour: f64 = forward(x)
        .iter()
        .zip(target)
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    let pull: f64 = x.iter().zip(given).map(|(a, b)| (a - b) * (a - b)).sum();
    ANCHOR.mul_add(pull, colour)
}

/// Solve `a·x = b` for a 4×4 system by Gaussian elimination with partial
/// pivoting; `None` if singular.
fn solve4(mut a: [[f64; 4]; 4], mut b: [f64; 4]) -> Option<[f64; 4]> {
    for col in 0..4 {
        let pivot = (col..4).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..4 {
            let f = a[row][col] / a[col][col];
            for k in col..4 {
                a[row][k] = (-f).mul_add(a[col][k], a[row][k]);
            }
            b[row] = (-f).mul_add(b[col], b[row]);
        }
    }
    let mut x = [0.0; 4];
    for row in (0..4).rev() {
        let tail: f64 = (row + 1..4).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - tail) / a[row][row];
    }
    Some(x)
}

/// `[0, 1]` ink amounts as bytes.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is clamped to [0, 255] and rounded first"
)]
fn ink_bytes(inks: [f64; 4]) -> [u8; 4] {
    inks.map(|v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
}

/// The CMYK ink bytes printing `rgb` in the process-ink model, from the
/// inks `given` (`[c, m, y, k]`, `[0, 1]` each, clamped; NaN is no ink).
///
/// An ink given as 0 is 0 in the result; of the amounts of the others that
/// print `rgb`, or the nearest colour to it those inks print, the ones
/// nearest `given`.  `given` is the result, rounded to bytes, when it
/// prints `rgb` within [`PRINTS`] counts per channel.
#[must_use]
pub fn ink_reproducing(rgb: [u8; 3], given: [f64; 4]) -> [u8; 4] {
    let given = given.map(|v| if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) });
    let open = given.map(|v| v > 0.0);
    let target = rgb.map(f64::from);
    if forward(given)
        .iter()
        .zip(target)
        .all(|(a, b)| (a - b).abs() <= PRINTS)
    {
        return ink_bytes(given);
    }
    let mut x = given;
    let mut err = cost(x, target, given);
    for _ in 0..MAX_STEPS {
        let f0 = forward(x);
        let mut jac = [[0.0; 4]; 3];
        for a in 0..4 {
            let (mut hi, mut lo) = (x, x);
            hi[a] = 1.0;
            lo[a] = 0.0;
            let (f_hi, f_lo) = (forward(hi), forward(lo));
            for (row, line) in jac.iter_mut().enumerate() {
                line[a] = f_hi[row] - f_lo[row];
            }
        }
        let res = [0, 1, 2].map(|i| f0[i] - target[i]);
        let grad = [0, 1, 2, 3].map(|a| {
            let colour: f64 = (0..3).map(|r| jac[r][a] * res[r]).sum();
            ANCHOR.mul_add(x[a] - given[a], colour)
        });
        // An ink given as 0, or at a bound the gradient pushes past, is
        // held for the step: an identity row with no right-hand side.
        let free = [0, 1, 2, 3].map(|a| {
            open[a] && !((x[a] <= 0.0 && grad[a] > 0.0) || (x[a] >= 1.0 && grad[a] < 0.0))
        });
        let mut normal = [[0.0; 4]; 4];
        let mut rhs = [0.0; 4];
        for i in 0..4 {
            if free[i] {
                for j in 0..4 {
                    if free[j] {
                        normal[i][j] = (0..3).map(|r| jac[r][i] * jac[r][j]).sum();
                    }
                }
                normal[i][i] += ANCHOR;
                rhs[i] = -grad[i];
            } else {
                normal[i][i] = 1.0;
            }
        }
        let Some(step) = solve4(normal, rhs) else {
            break;
        };
        let mut scale = 1.0;
        let mut improved = false;
        for _ in 0..HALVINGS {
            let cand = [0, 1, 2, 3].map(|a| step[a].mul_add(scale, x[a]).clamp(0.0, 1.0));
            let e = cost(cand, target, given);
            if e < err {
                improved = err - e > 1e-9;
                x = cand;
                err = e;
                break;
            }
            scale *= 0.5;
        }
        if !improved {
            break;
        }
    }
    ink_bytes(x)
}

#[cfg(test)]
mod tests {
    use super::super::cmyk_to_rgb;
    use super::ink_reproducing;

    fn unit(b: [u8; 4]) -> [f64; 4] {
        b.map(|v| f64::from(v) / 255.0)
    }

    #[test]
    fn a_colour_the_given_inks_print_keeps_them() {
        let mut s = 7u32;
        for _ in 0..500 {
            let ink = [0; 4].map(|_: u8| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                s.to_be_bytes()[0]
            });
            let got = ink_reproducing(cmyk_to_rgb(ink[0], ink[1], ink[2], ink[3]), unit(ink));
            let back = cmyk_to_rgb(got[0], got[1], got[2], got[3]);
            let want = cmyk_to_rgb(ink[0], ink[1], ink[2], ink[3]);
            for ch in 0..3 {
                assert!(back[ch].abs_diff(want[ch]) <= 1, "{ink:?} -> {got:?}");
            }
            for a in 0..4 {
                assert!(got[a].abs_diff(ink[a]) <= 2, "{ink:?} -> {got:?}");
            }
        }
    }

    #[test]
    fn an_ink_given_as_zero_stays_zero() {
        // A warm dark grey from black alone: black prints only greys, so the
        // nearest is the grey of the colour's mean.
        let got = ink_reproducing([35, 31, 32], [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(got[..3], [0, 0, 0]);
        let grey = cmyk_to_rgb(0, 0, 0, got[3]);
        assert!(grey[0].abs_diff(33) <= 1, "{got:?} prints {grey:?}");
        // A magenta given as M and Y keeps C and K at zero.
        let got = ink_reproducing([220, 30, 60], [0.0, 1.0, 0.6, f64::NAN]);
        assert_eq!((got[0], got[3]), (0, 0));
        assert_eq!(ink_reproducing([200, 10, 90], [0.0; 4]), [0; 4]);
    }

    #[test]
    fn a_printable_colour_is_reproduced_from_other_inks() {
        // The cyan the model prints at 60 % from 50 % given, M and Y free
        // but pulled to their given zero-ish amounts.
        let want = cmyk_to_rgb(153, 20, 10, 0);
        let got = ink_reproducing(want, [0.5, 0.05, 0.05, 0.0]);
        let back = cmyk_to_rgb(got[0], got[1], got[2], got[3]);
        for ch in 0..3 {
            assert!(back[ch].abs_diff(want[ch]) <= 1, "{got:?} prints {back:?}");
        }
        assert_eq!(got[3], 0);
    }
}
