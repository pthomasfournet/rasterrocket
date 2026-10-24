//! Page-curvature correction from text-line geometry.
//!
//! # Surface model
//!
//! In normalised coordinates `t = (x − cₓ)/(w/2)`, `u = (y − c_y)/(h/2)`,
//! a page bent about a vertical axis (a book opened toward its spine) and
//! imaged from above moves each point vertically by an amount that varies
//! smoothly with `t` and linearly with `u`: perspective scales a column's
//! height by its distance from the camera, and an off-axis camera adds a
//! column-dependent shift.  A text line whose flat height is `hₗ` is
//! therefore imaged on
//!
//! ```text
//! u = hₗ + a(t) + b(t) · hₗ,    a(t) = Σₖ αₖ tᵏ,  b(t) = Σₖ βₖ tᵏ,  k = 1..=K
//! ```
//!
//! The absent constant terms fix the gauge: the centre column is the
//! reference, so the model has no freedom to shift or scale the page.
//!
//! # Fit
//!
//! Given tracked lines, the model is bilinear in `(α, β)` and `h`; it is
//! solved by alternating least squares (linear solve for `(α, β)` with `h`
//! fixed, closed-form `hₗ` update with `(α, β)` fixed), with one round of
//! 3σ outlier rejection.  `β` is only fitted when the tracked lines span at
//! least a quarter of the page height; otherwise it is held at zero.
//!
//! # Acceptance
//!
//! - at least [`MIN_TRACKS`] tracked lines;
//! - the lines are curved: the RMS residual of independent straight-line
//!   fits per track exceeds the larger of [`MIN_CURVE_PX`] and 0.02 pitch;
//! - the model explains the curvature: its RMS residual is at most
//!   [`MAX_RESIDUAL_RATIO`] of the straight-line residual;
//! - the map is tame over the fitted range: `|a| ≤ 0.25` and
//!   `0.75 ≤ 1 + b ≤ 1.33`.
//!
//! # Remap
//!
//! Output pixel `(x, y)` samples the source at `(x, y_src)` with
//! `u_src = u + a(t) + b(t)·u`, `t` clamped to the range covered by the
//! tracks so the polynomials are never extrapolated sideways.  Horizontal
//! foreshortening of the curved region is not modelled.

use crate::linalg::solve;
use crate::num::to_f64;
use crate::textlines::TextLines;

/// Polynomial degree of `a(t)` and `b(t)`.
const K: usize = 4;
/// Minimum number of tracked lines.
const MIN_TRACKS: usize = 5;
/// Minimum straight-line residual, in pixels, for a page to count as curved.
const MIN_CURVE_PX: f64 = 0.35;
/// Maximum ratio of model residual to straight-line residual.
const MAX_RESIDUAL_RATIO: f64 = 0.9;
/// Minimum bend the correction must apply, in pixels.
const MIN_BEND_PX: f64 = 1.5;
/// Minimum bend the correction must apply, in line pitches.
const MIN_BEND_PITCH: f64 = 0.1;

/// A fitted page surface, mapping output rows to source rows.
#[derive(Debug, Clone)]
pub(crate) struct Surface {
    alpha: [f64; K],
    beta: [f64; K],
    t_range: (f64, f64),
    centre: (f64, f64),
    half: (f64, f64),
}

impl Surface {
    fn poly(c: &[f64; K], t: f64) -> f64 {
        c.iter().rev().fold(0.0f64, |acc, &ck| acc.mul_add(t, ck)) * t
    }

    /// Source row sampled by output pixel `(x, y)`.
    pub(crate) fn source_y(&self, x: f64, y: f64) -> f64 {
        let t = ((x - self.centre.0) / self.half.0).clamp(self.t_range.0, self.t_range.1);
        let u = (y - self.centre.1) / self.half.1;
        let u_src = Self::poly(&self.beta, t).mul_add(u, u + Self::poly(&self.alpha, t));
        u_src.mul_add(self.half.1, self.centre.1)
    }
}

struct Sample {
    t: f64,
    u: f64,
    track: usize,
    weight: f64,
}

/// Fit the surface to `lines` found on a `w × h` page.  `Err` holds the
/// reason when the page is flat, the lines are too few, or the fit fails
/// acceptance.
pub(crate) fn fit(lines: &TextLines, w: usize, h: usize) -> Result<Surface, String> {
    if lines.tracks.len() < MIN_TRACKS {
        return Err(format!(
            "{} text lines tracked; at least {MIN_TRACKS} needed",
            lines.tracks.len()
        ));
    }
    let centre = ((to_f64(w) - 1.0) / 2.0, (to_f64(h) - 1.0) / 2.0);
    let half = (to_f64(w) / 2.0, to_f64(h) / 2.0);
    let mut samples: Vec<Sample> = Vec::new();
    for (i, tr) in lines.tracks.iter().enumerate() {
        for (p, &wt) in tr.points.iter().zip(&tr.weights) {
            samples.push(Sample {
                t: (p[0] - centre.0) / half.0,
                u: (p[1] - centre.1) / half.1,
                track: i,
                weight: wt,
            });
        }
    }
    let n_tracks = lines.tracks.len();

    let straight_px = straightness_px(lines, h);
    let curve_floor = MIN_CURVE_PX.max(0.02 * lines.pitch);
    if straight_px.is_nan() || straight_px <= curve_floor {
        return Err(format!(
            "text lines are straight (rms curvature {straight_px:.2} px, at most {curve_floor:.2} px)"
        ));
    }

    let mut heights = track_means(&samples, n_tracks);
    let (lo, hi) = heights
        .iter()
        .fold((f64::MAX, f64::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let fit_beta = hi - lo >= 0.5;
    let mut params = vec![0.0; 2 * K];
    let mut rms = 0.0;
    for round in 0..2 {
        params = alternate(&samples, &mut heights, fit_beta)
            .ok_or_else(|| "surface fit did not converge to a finite solution".to_owned())?;
        rms = model_rms(&samples, &heights, &params);
        if round == 0 {
            let limit = 3.0 * rms;
            samples.retain(|s| residual(s, &heights, &params).abs() <= limit);
        }
    }
    let model_px = rms * half.1;
    if model_px.is_nan() || model_px > MAX_RESIDUAL_RATIO * straight_px {
        return Err(format!(
            "surface model leaves rms {model_px:.2} px of {straight_px:.2} px curvature; \
             at most {MAX_RESIDUAL_RATIO} of it may remain"
        ));
    }

    let t_range = samples.iter().fold((f64::MAX, f64::MIN), |(lo, hi), s| {
        (lo.min(s.t), hi.max(s.t))
    });
    if !(t_range.0 <= t_range.1 && t_range.0.is_finite() && t_range.1.is_finite()) {
        return Err("no text-line samples survive outlier rejection".to_owned());
    }
    let mut alpha = [0.0; K];
    let mut beta = [0.0; K];
    alpha.copy_from_slice(&params[..K]);
    beta.copy_from_slice(&params[K..]);
    let surface = Surface {
        alpha,
        beta,
        t_range,
        centre,
        half,
    };
    for i in 0..=64u32 {
        let t = (t_range.1 - t_range.0).mul_add(f64::from(i) / 64.0, t_range.0);
        let a = Surface::poly(&surface.alpha, t);
        let scale = 1.0 + Surface::poly(&surface.beta, t);
        if a.is_nan() || a.abs() > 0.25 || !(0.75..=1.33).contains(&scale) {
            return Err(format!(
                "surface displacement {a:.3} or scale {scale:.3} out of range at t = {t:.2}"
            ));
        }
    }
    let amplitude = bend_amplitude(&surface, &samples, &heights) * half.1;
    let min_amplitude = MIN_BEND_PX.max(MIN_BEND_PITCH * lines.pitch);
    if amplitude.is_nan() || amplitude < min_amplitude {
        return Err(format!(
            "correction would bend lines by {amplitude:.2} px; at least {min_amplitude:.2} px needed"
        ));
    }
    Ok(surface)
}

/// Upper-quartile, over tracks, of the largest deviation of the modelled
/// line `u(t) = hₗ + a(t) + b(t)·hₗ` from its own best straight line, over
/// the track's extent: how far the correction would actually bend lines,
/// in normalised units.
fn bend_amplitude(surface: &Surface, samples: &[Sample], heights: &[f64]) -> f64 {
    let mut extent = vec![(f64::MAX, f64::MIN); heights.len()];
    for s in samples {
        let e = &mut extent[s.track];
        *e = (e.0.min(s.t), e.1.max(s.t));
    }
    let mut bends: Vec<f64> = extent
        .iter()
        .zip(heights)
        .filter(|((lo, hi), _)| hi > lo)
        .map(|(&(lo, hi), &h)| {
            let curve: Vec<[f64; 2]> = (0..=16u32)
                .map(|i| {
                    let t = (hi - lo).mul_add(f64::from(i) / 16.0, lo);
                    let u = Surface::poly(&surface.beta, t)
                        .mul_add(h, h + Surface::poly(&surface.alpha, t));
                    [t, u]
                })
                .collect();
            let n = 17.0;
            let mt = curve.iter().map(|p| p[0]).sum::<f64>() / n;
            let mu = curve.iter().map(|p| p[1]).sum::<f64>() / n;
            let stt: f64 = curve.iter().map(|p| (p[0] - mt).powi(2)).sum();
            let stu: f64 = curve.iter().map(|p| (p[0] - mt) * (p[1] - mu)).sum();
            let m = stu / stt;
            curve
                .iter()
                .map(|p| (p[1] - m.mul_add(p[0] - mt, mu)).abs())
                .fold(0.0, f64::max)
        })
        .collect();
    if bends.is_empty() {
        return 0.0;
    }
    bends.sort_by(f64::total_cmp);
    bends[bends.len() * 3 / 4]
}

/// Weighted RMS deviation, in pixels, of the tracked lines of a page of
/// height `h` from per-line straight fits.
pub(crate) fn straightness_px(lines: &TextLines, h: usize) -> f64 {
    let half = to_f64(h) / 2.0;
    let samples: Vec<Sample> = lines
        .tracks
        .iter()
        .enumerate()
        .flat_map(|(i, tr)| {
            tr.points
                .iter()
                .zip(&tr.weights)
                .map(move |(p, &weight)| Sample {
                    t: p[0] / half,
                    u: p[1] / half,
                    track: i,
                    weight,
                })
        })
        .collect();
    straight_residual(&samples, lines.tracks.len()) * half
}

fn features(t: f64, height: f64, fit_beta: bool) -> [f64; 2 * K] {
    let mut phi = [0.0; 2 * K];
    let mut tk = 1.0;
    for k in 0..K {
        tk *= t;
        phi[k] = tk;
        if fit_beta {
            phi[K + k] = tk * height;
        }
    }
    phi
}

fn residual(s: &Sample, heights: &[f64], params: &[f64]) -> f64 {
    let h = heights[s.track];
    let phi = features(s.t, h, true);
    let pred: f64 = phi.iter().zip(params).map(|(f, p)| f * p).sum();
    s.u - h - pred
}

fn model_rms(samples: &[Sample], heights: &[f64], params: &[f64]) -> f64 {
    let (num, den) = samples.iter().fold((0.0, 0.0), |(n, d), s| {
        (
            residual(s, heights, params).powi(2).mul_add(s.weight, n),
            d + s.weight,
        )
    });
    (num / den.max(f64::EPSILON)).sqrt()
}

/// Weighted mean `u` of every track.
fn track_means(samples: &[Sample], n_tracks: usize) -> Vec<f64> {
    let mut acc = vec![(0.0, 0.0); n_tracks];
    for s in samples {
        acc[s.track].0 = s.u.mul_add(s.weight, acc[s.track].0);
        acc[s.track].1 += s.weight;
    }
    acc.into_iter()
        .map(|(su, sw)| su / sw.max(f64::EPSILON))
        .collect()
}

/// Weighted RMS residual of an independent straight line `u = c + m·t`
/// fitted to each track: the curvature that no rotation can remove.
fn straight_residual(samples: &[Sample], n_tracks: usize) -> f64 {
    // Per-track weighted sums: w, wt, wu, wtt, wtu, wuu.
    let mut acc = vec![[0.0f64; 6]; n_tracks];
    for s in samples {
        let a = &mut acc[s.track];
        let w = s.weight;
        a[0] += w;
        a[1] = w.mul_add(s.t, a[1]);
        a[2] = w.mul_add(s.u, a[2]);
        a[3] = (w * s.t).mul_add(s.t, a[3]);
        a[4] = (w * s.t).mul_add(s.u, a[4]);
        a[5] = (w * s.u).mul_add(s.u, a[5]);
    }
    let (mut sse, mut wsum) = (0.0, 0.0);
    for a in acc {
        if a[0] <= 0.0 {
            continue;
        }
        let (mt, mu) = (a[1] / a[0], a[2] / a[0]);
        let stt = a[0].mul_add(-mt * mt, a[3]);
        let stu = a[0].mul_add(-mt * mu, a[4]);
        let suu = a[0].mul_add(-mu * mu, a[5]);
        let explained = if stt > 0.0 { stu * stu / stt } else { 0.0 };
        sse += (suu - explained).max(0.0);
        wsum += a[0];
    }
    (sse / wsum.max(f64::EPSILON)).sqrt()
}

/// Alternating least squares for `(α, β)` and the track heights.
fn alternate(samples: &[Sample], heights: &mut [f64], fit_beta: bool) -> Option<Vec<f64>> {
    let n = if fit_beta { 2 * K } else { K };
    let mut params = vec![0.0; 2 * K];
    for _ in 0..60 {
        let mut a = vec![0.0; n * n];
        let mut b = vec![0.0; n];
        let mut wsum = 0.0;
        for s in samples {
            let h = heights[s.track];
            let phi = features(s.t, h, fit_beta);
            let target = s.u - h;
            for r in 0..n {
                for c in 0..n {
                    a[r * n + c] = (s.weight * phi[r]).mul_add(phi[c], a[r * n + c]);
                }
                b[r] = (s.weight * phi[r]).mul_add(target, b[r]);
            }
            wsum += s.weight;
        }
        // Ridge term: keeps the system regular when a coefficient is
        // unobservable (e.g. all lines on one side of the centre).
        for d in 0..n {
            a[d * n + d] = 1e-6f64.mul_add(wsum, a[d * n + d]);
        }
        solve(&mut a, &mut b, n)?;
        let mut next = vec![0.0; 2 * K];
        next[..n].copy_from_slice(&b);

        // Closed-form height update: u − a(t) = h · (1 + b(t)).
        let mut acc = vec![(0.0, 0.0); heights.len()];
        for s in samples {
            let mut ta = [0.0; K];
            let mut tb = [0.0; K];
            ta.copy_from_slice(&next[..K]);
            tb.copy_from_slice(&next[K..]);
            let av = Surface::poly(&ta, s.t);
            let bv = 1.0 + Surface::poly(&tb, s.t);
            acc[s.track].0 = (s.weight * (s.u - av)).mul_add(bv, acc[s.track].0);
            acc[s.track].1 = (s.weight * bv).mul_add(bv, acc[s.track].1);
        }
        for (h, (num, den)) in heights.iter_mut().zip(acc) {
            if den > 0.0 {
                *h = num / den;
            }
        }
        let delta: f64 = next.iter().zip(&params).map(|(x, y)| (x - y).abs()).sum();
        params = next;
        if delta < 1e-10 {
            break;
        }
    }
    params.iter().all(|v| v.is_finite()).then_some(params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::resample;
    use crate::testutil::curved_text;
    use crate::textlines;

    #[test]
    fn flat_text_is_not_dewarped() {
        let p = curved_text(1200, 1000, |_, _| 0.0);
        let lines = textlines::find(&p.view()).expect("lines");
        assert!(fit(&lines, p.width, p.height).is_err());
    }

    #[test]
    fn rotated_flat_text_is_not_dewarped() {
        let p = curved_text(1200, 1000, |x, _| 0.03 * x);
        let lines = textlines::find(&p.view()).expect("lines");
        assert!(fit(&lines, p.width, p.height).is_err());
    }

    #[test]
    fn bow_scaling_with_height_is_recovered() {
        // Lines bend toward the right edge, more strongly far from the
        // vertical centre: a cylinder viewed from above.
        let (w, h) = (1200usize, 1000usize);
        let cy = to_f64(h - 1) / 2.0;
        let bow = |x: f64, base: f64| {
            let s = (x / to_f64(w)).powi(3);
            0.1f64.mul_add(base - cy, 40.0) * s
        };
        let p = curved_text(w, h, bow);
        let before = textlines::find(&p.view()).expect("lines");
        let surface = fit(&before, w, h).expect("surface");
        let out = resample(p.view(), w, h, |x, y| (x, surface.source_y(x, y)));
        let after = textlines::find(&out.view()).expect("lines after");
        let (rms_before, rms_after) = (straightness_px(&before, h), straightness_px(&after, h));
        assert!(
            rms_before > 3.0,
            "synthetic bow too weak: {rms_before:.2} px"
        );
        assert!(
            rms_after < 0.6,
            "residual curvature {rms_after:.2} px (was {rms_before:.2})"
        );
    }
}
