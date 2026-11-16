//! The preservation of black ink in the conversion of one CMYK profile's
//! ink to another's ([`PreserveBlack`]).
//!
//! The K → K curve is tabulated at [`TONE_SAMPLES`] inks: the darkness
//! `1 − L* / 100` of the source's `(0, 0, 0, K)` under the intent's device
//! → PCS table, then the inverse of the destination's darkness curve,
//! interpolated linearly between its samples.  Where the destination's
//! curve is flat the inverse takes the highest ink of the flat run; a
//! darkness beyond the destination's range extrapolates its last segment,
//! clipped to `[0, 1]`.  A curve that falls by more than [`RIPPLE`]
//! anywhere is not a K → K map, and no preservation is built.
//!
//! Under [`PreserveBlack::KPlane`] the colour to keep is the colorimetric
//! conversion's ink measured through the destination's colorimetric
//! (`AToB1`) table, in the Lab encoding of the PCS
//! ([`crate::pcs::lut::lab_encoding`]).  Newton iteration from that ink's
//! `C M Y`, its K fixed at the curve's, steps by the forward-difference
//! Jacobian ([`JACOBIAN_STEP`], stepping down at the top of the range),
//! each step clipped to `[0, 1]` and halved up to [`HALVINGS`] times until
//! it brings the ink nearer the colour; it stops after [`ITERATIONS`]
//! steps, on convergence, or when no halving helps.  An ink whose colour
//! misses by more than [`MAX_MISS`], or a singular Jacobian, leaves the
//! colorimetric ink: the colour lies outside what the preserved K can
//! print.  The destination's total ink limit is the largest ink sum of its
//! perceptual PCS → device table over a grid of 6 `L*` × 74 `a*` × 74
//! `b*` levels; an ink over it scales its `C M Y` down to meet it.

use moxcms::{ColorProfile, RenderingIntent};

use crate::PreserveBlack;
use crate::pcs::D50;
use crate::pcs::black::y_to_lab_l;
use crate::pcs::lut::{Direction, ProfileLut, lab_decoding, lab_encoding};

/// The number of inks at which the K → K curve is tabulated.
const TONE_SAMPLES: usize = 4096;

/// The fall a K → K curve may show between two samples and still be taken
/// as rising: 2 levels of 65 535.
const RIPPLE: f64 = 2.0 / 65535.0;

/// The difference of K below which the colorimetric conversion's K is the
/// curve's: 3 levels of 65 535.
const K_TOLERANCE: f64 = 3.0 / 65535.0;

/// The ink step of the forward-difference Jacobian.
const JACOBIAN_STEP: f64 = 0.001;

/// The determinant below which the Jacobian is singular.
const SINGULAR: f64 = 1e-4;

/// The largest number of Newton steps.
const ITERATIONS: usize = 30;

/// The number of times a Newton step is halved before the iteration stops.
const HALVINGS: usize = 8;

/// The distance, in the Lab encoding, below which the iteration has
/// converged.
const CONVERGED: f64 = 1e-6;

/// The largest CIE 1976 difference between the colour to keep and the
/// colour of the ink found, beyond which the colour keeps its colorimetric
/// conversion.
const MAX_MISS: f64 = 1.0;

/// The preservation of black between a source and a destination CMYK
/// profile under one intent.
pub(super) struct Preservation {
    /// The K → K curve at [`TONE_SAMPLES`] inks over `[0, 1]`.
    tone: Vec<f64>,
    /// The colour measurement and ink limit of [`PreserveBlack::KPlane`].
    plane: Option<Plane>,
}

/// What [`PreserveBlack::KPlane`] measures and limits with.
struct Plane {
    /// The destination's colorimetric device → PCS table.
    measure: ProfileLut,
    /// The destination's total ink limit, as a sum of inks in `[0, 1]`.
    limit: f64,
}

impl Preservation {
    /// The preservation `mode` of black from `source`'s ink to `dest`'s
    /// under `intent`; `None` when either profile lacks a table it reads
    /// or the K → K curve is not rising.
    pub(super) fn new(
        source: &ColorProfile,
        dest: &ColorProfile,
        intent: RenderingIntent,
        mode: PreserveBlack,
    ) -> Option<Self> {
        let tone = tone(&darkness(source, intent)?, &darkness(dest, intent)?)?;
        let plane = match mode {
            PreserveBlack::KOnly => None,
            PreserveBlack::KPlane => Some(Plane {
                measure: ProfileLut::new(
                    dest,
                    RenderingIntent::RelativeColorimetric,
                    Direction::ToPcs,
                    4,
                )?,
                limit: ink_limit(dest)?,
            }),
        };
        Some(Self { tone, plane })
    }

    /// The destination ink of the source ink `ink`, whose colorimetric
    /// conversion is `colorimetric`, into `out`.
    pub(super) fn convert(
        &self,
        ink: &[f64; 4],
        colorimetric: impl FnOnce(&mut [f64; 4]),
        out: &mut [f64; 4],
    ) {
        let k = sample(&self.tone, ink[3]);
        if ink[..3].iter().all(|&v| v == 0.0) {
            *out = [0.0, 0.0, 0.0, k];
            return;
        }
        colorimetric(out);
        let Some(plane) = &self.plane else {
            return;
        };
        if (out[3] - k).abs() < K_TOLERANCE {
            return;
        }
        let target = lab_encoding(plane.measure.to_pcs(out.as_slice()));
        if let Some(cmy) = plane.solve(target, [out[0], out[1], out[2]], k) {
            let sum = cmy.iter().sum::<f64>();
            let ratio = if sum + k > plane.limit && sum > 0.0 {
                (1.0 - (sum + k - plane.limit) / sum).max(0.0)
            } else {
                1.0
            };
            *out = [cmy[0] * ratio, cmy[1] * ratio, cmy[2] * ratio, k];
        }
    }
}

impl Plane {
    /// The `C M Y` from `hint` whose ink with black `k` the destination
    /// measures within [`MAX_MISS`] of `target`; `None` at a singular
    /// Jacobian or when the nearest ink found misses by more.
    fn solve(&self, target: [f64; 3], hint: [f64; 3], k: f64) -> Option<[f64; 3]> {
        let lab = |x: [f64; 3]| lab_encoding(self.measure.to_pcs(&[x[0], x[1], x[2], k]));
        let mut x = hint;
        let mut fx = lab(x);
        let mut error = distance(fx, target);
        for _ in 0..ITERATIONS {
            if error <= CONVERGED {
                break;
            }
            let mut jacobian = [[0.0; 3]; 3];
            for j in 0..3 {
                let mut xd = x;
                let step = if xd[j] < 1.0 - JACOBIAN_STEP {
                    JACOBIAN_STEP
                } else {
                    -JACOBIAN_STEP
                };
                xd[j] += step;
                let fxd = lab(xd);
                for (row, (a, b)) in jacobian.iter_mut().zip(fxd.iter().zip(&fx)) {
                    row[j] = (a - b) / step;
                }
            }
            let residual = [0, 1, 2].map(|i| fx[i] - target[i]);
            let delta = solve3(&jacobian, residual)?;
            // The Newton step, halved until it brings the ink nearer.
            let mut scale = 1.0f64;
            let nearer = (0..HALVINGS).find_map(|_| {
                let next = [0, 1, 2].map(|i| scale.mul_add(-delta[i], x[i]).clamp(0.0, 1.0));
                let f_next = lab(next);
                let e_next = distance(f_next, target);
                scale /= 2.0;
                (e_next < error).then_some((next, f_next, e_next))
            });
            let Some((next, f_next, e_next)) = nearer else {
                break;
            };
            (x, fx, error) = (next, f_next, e_next);
        }
        (delta_e76(fx, target) <= MAX_MISS).then_some(x)
    }
}

/// The CIE 1976 colour difference of two colours in the Lab encoding of
/// [`lab_encoding`].
fn delta_e76(a: [f64; 3], b: [f64; 3]) -> f64 {
    let scale = [100.0, 255.0, 255.0];
    distance(
        std::array::from_fn(|i| a[i] * scale[i]),
        std::array::from_fn(|i| b[i] * scale[i]),
    )
}

/// The Euclidean distance of `a` and `b`.
fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.iter()
        .zip(&b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// The solution `x` of `m · x = b`, by the adjugate; `None` when the
/// determinant's magnitude is below [`SINGULAR`].
fn solve3(m: &[[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let cofactor = |r: usize, c: usize| {
        let (r0, r1) = ((r + 1) % 3, (r + 2) % 3);
        let (c0, c1) = ((c + 1) % 3, (c + 2) % 3);
        m[r0][c0].mul_add(m[r1][c1], -(m[r0][c1] * m[r1][c0]))
    };
    let det = (0..3).map(|c| m[0][c] * cofactor(0, c)).sum::<f64>();
    if det.abs() < SINGULAR {
        return None;
    }
    // The inverse is the transposed cofactor matrix over the determinant.
    Some([0, 1, 2].map(|r| (0..3).map(|c| cofactor(c, r) * b[c]).sum::<f64>() / det))
}

/// The darkness `1 − L* / 100` of `profile`'s black ink `(0, 0, 0, K)`
/// under `intent` at [`TONE_SAMPLES`] inks over `[0, 1]`; `None` when the
/// profile has no device → PCS table it reads.
fn darkness(profile: &ColorProfile, intent: RenderingIntent) -> Option<Vec<f64>> {
    let lut = ProfileLut::new(profile, intent, Direction::ToPcs, 4)?;
    Some(
        (0..TONE_SAMPLES)
            .map(|i| {
                let k = unit(i, TONE_SAMPLES);
                1.0 - y_to_lab_l(lut.to_pcs(&[0.0, 0.0, 0.0, k])[1]) / 100.0
            })
            .collect(),
    )
}

/// The K → K curve of the darkness curves `source` and `dest` (see the
/// module documentation); `None` when it is not rising.
fn tone(source: &[f64], dest: &[f64]) -> Option<Vec<f64>> {
    let inverse = inverse(dest);
    let tone: Vec<f64> = source.iter().map(|&d| sample(&inverse, d)).collect();
    tone.windows(2)
        .all(|w| w[0] - w[1] <= RIPPLE)
        .then_some(tone)
}

/// The inverse of the curve of `samples` over `[0, 1]` at as many values
/// over `[0, 1]` (see the module documentation).
fn inverse(samples: &[f64]) -> Vec<f64> {
    let n = samples.len();
    let rising = samples[0] < samples[n - 1];
    let (mut a, mut b) = (0.0, 0.0);
    (0..n)
        .map(|i| {
            let y = unit(i, n);
            let within = |j: &usize| {
                let (lo, hi) = (
                    samples[*j].min(samples[j + 1]),
                    samples[*j].max(samples[j + 1]),
                );
                (lo..=hi).contains(&y)
            };
            let found = if rising {
                (0..n - 1).rev().find(within)
            } else {
                (0..n - 1).find(within)
            };
            if let Some(j) = found {
                let (x1, x2) = (samples[j], samples[j + 1]);
                let (y1, y2) = (unit(j, n), unit(j + 1, n));
                #[expect(
                    clippy::float_cmp,
                    reason = "an interval collapses only when its ends are equal"
                )]
                if x1 == x2 {
                    return if rising { y2 } else { y1 };
                }
                a = (y2 - y1) / (x2 - x1);
                b = a.mul_add(-x2, y2);
            }
            a.mul_add(y, b).clamp(0.0, 1.0)
        })
        .collect()
}

/// The curve of `samples` over `[0, 1]` at `x`, interpolated linearly.
fn sample(samples: &[f64], x: f64) -> f64 {
    let last = samples.len() - 1;
    #[expect(clippy::cast_precision_loss, reason = "a curve has 4096 samples")]
    let pos = x.clamp(0.0, 1.0) * last as f64;
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "pos lies in [0, last]"
    )]
    let lo = (pos.floor() as usize).min(last - 1);
    #[expect(clippy::cast_precision_loss, reason = "lo < 4096")]
    let f = pos - lo as f64;
    f.mul_add(samples[lo + 1] - samples[lo], samples[lo])
}

/// The value `i / (n − 1)`.
fn unit(i: usize, n: usize) -> f64 {
    #[expect(clippy::cast_precision_loss, reason = "a sample index is below 2^52")]
    let (i, n) = (i as f64, n as f64);
    i / (n - 1.0)
}

/// The largest ink sum of `dest`'s perceptual PCS → device table over the
/// PCS grid of the module documentation, the Lab components at the levels
/// `round(i · 65535 / (n − 1))` of the 16-bit Lab encoding, the PCS black
/// mapped to the destination's where a perceptual conversion to it
/// compensates; `None` when the profile has no such table it reads.
fn ink_limit(dest: &ColorProfile) -> Option<f64> {
    let to_device = ProfileLut::new(dest, RenderingIntent::Perceptual, Direction::ToDevice, 4)?;
    // The PCS black lifted to the destination's own, as a perceptual
    // conversion to a version 4 profile compensates (see `link`).
    let black = super::source::black_lift(dest, RenderingIntent::Perceptual, false)
        .ok()?
        .unwrap_or(0.0);
    let level = |i: usize, n: usize| (unit(i, n) * 65535.0).round();
    let mut limit = 0.0f64;
    let mut ink = [0.0; 4];
    for li in 0..6 {
        let l = level(li, 6) * 100.0 / 65535.0;
        for ai in 0..74 {
            let a = level(ai, 74) / 257.0 - 128.0;
            for bi in 0..74 {
                let b = level(bi, 74) / 257.0 - 128.0;
                let xyz = lab_decoding([l / 100.0, (a + 128.0) / 255.0, (b + 128.0) / 255.0]);
                let lifted = std::array::from_fn(|i| black.mul_add(D50[i], (1.0 - black) * xyz[i]));
                to_device.to_device(lifted, &mut ink);
                limit = limit.max(ink.iter().sum());
            }
        }
    }
    Some(limit)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The inverse of a rising curve composes with it to the identity, a
    /// flat run inverting to its highest value, and a darkness beyond the
    /// curve's range extrapolating its last segment.
    #[test]
    fn the_inverse_undoes_the_curve() {
        let n = 101;
        let curve: Vec<f64> = (0..n)
            .map(|i| unit(i, n).mul_add(0.5, unit(i, n).powi(2) * 0.3))
            .collect();
        let inv = inverse(&curve);
        for i in 0..n {
            let x = unit(i, n);
            assert!((sample(&inv, sample(&curve, x)) - x).abs() < 2e-3, "{x}");
        }
        assert!((sample(&inv, 0.9) - 1.0).abs() < 1e-12);
        // Samples at 0, 1/2, 1 of a curve flat over the first half: the
        // darkness 0 inverts to the run's top, 1/2.
        assert!((inverse(&[0.0, 0.0, 1.0])[0] - 0.5).abs() < 1e-12);
    }

    /// The total ink limits of the test printers, as an independent
    /// converter detects them (286.9 % and 336.2 %).
    #[test]
    fn the_ink_limit_is_the_largest_separated_sum() {
        for (name, want) in [("printer-cmyk-v4", 2.869), ("printer-cmyk-v2", 3.362)] {
            let path = format!(
                "{}/../../tests/fixtures/icc/{name}.icc",
                env!("CARGO_MANIFEST_DIR")
            );
            let profile =
                ColorProfile::new_from_slice(&std::fs::read(&path).unwrap()).expect("parses");
            let limit = ink_limit(&profile).expect("a perceptual PCS → device table");
            assert!((limit - want).abs() < 0.005, "{name}: {limit}");
        }
    }

    /// A falling curve is no K → K map.
    #[test]
    fn a_falling_tone_is_refused() {
        let n = 64;
        let rising: Vec<f64> = (0..n).map(|i| unit(i, n)).collect();
        assert!(tone(&rising, &rising).is_some());
        let mut bumpy = rising.clone();
        bumpy[40] = bumpy[10];
        assert!(tone(&bumpy, &rising).is_none());
    }

    #[test]
    fn a_linear_system_solves() {
        let m = [[2.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 3.0]];
        let x = solve3(&m, [3.0, 2.0, 4.0]).unwrap();
        for (r, want) in m.iter().zip([3.0, 2.0, 4.0]) {
            let got: f64 = r.iter().zip(&x).map(|(a, b)| a * b).sum();
            assert!((got - want).abs() < 1e-12, "{x:?}");
        }
        assert!(
            solve3(
                &[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 0.0, 1.0]],
                [0.0; 3]
            )
            .is_none()
        );
    }
}
