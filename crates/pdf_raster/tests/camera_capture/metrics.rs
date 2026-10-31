//! Recovery metrics, independent of the implementation under test.

use crate::synth::Gray;

/// Pearson correlation of the text layout of `a` and `b`: both are resized
/// to a third of `b`'s size and high-passed (each pixel minus its 13 × 13
/// local mean).  The reduction tolerates the 1–2 px offsets that would
/// decorrelate individual strokes; the high-pass removes illumination
/// gradients and shadows, which carry no layout.
pub(crate) fn correlation(a: &Gray, b: &Gray) -> f64 {
    let (w, h) = (b.w / 3, b.h / 3);
    let (a, b) = (
        high_pass(&area_resize(a, w, h)),
        high_pass(&area_resize(b, w, h)),
    );
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(&b) {
        sab += x * y;
        saa += x * x;
        sbb += y * y;
    }
    sab / (saa * sbb).sqrt().max(f64::EPSILON)
}

/// Downscale by averaging each output pixel's footprint (no aliasing).
fn area_resize(g: &Gray, w: usize, h: usize) -> Gray {
    let mut out = Gray::new(w, h, 255);
    for y in 0..h {
        let (y0, y1) = (y * g.h / h, ((y + 1) * g.h / h).max(y * g.h / h + 1));
        for x in 0..w {
            let (x0, x1) = (x * g.w / w, ((x + 1) * g.w / w).max(x * g.w / w + 1));
            let mut sum = 0u64;
            for yy in y0..y1 {
                for xx in x0..x1 {
                    sum += u64::from(g.at(xx, yy));
                }
            }
            out.px[y * w + x] = (sum / ((x1 - x0) * (y1 - y0)) as u64) as u8;
        }
    }
    out
}

fn high_pass(g: &Gray) -> Vec<f64> {
    let r = 6usize;
    let mut out = Vec::with_capacity(g.w * g.h);
    for y in 0..g.h {
        for x in 0..g.w {
            let (x0, x1) = (x.saturating_sub(r), (x + r + 1).min(g.w));
            let (y0, y1) = (y.saturating_sub(r), (y + r + 1).min(g.h));
            let mut sum = 0.0;
            for yy in y0..y1 {
                for xx in x0..x1 {
                    sum += f64::from(g.at(xx, yy));
                }
            }
            out.push(f64::from(g.at(x, y)) - sum / ((x1 - x0) * (y1 - y0)) as f64);
        }
    }
    out
}

/// Largest corner distance between two quads.
pub(crate) fn corner_error(a: &[[f64; 2]; 4], b: &[[f64; 2]; 4]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(p, q)| (p[0] - q[0]).hypot(p[1] - q[1]))
        .fold(0.0, f64::max)
}

/// Text-line displacement of `img` against the reference `flat` (same
/// size): `(rms, max)`, in pixels, of the local vertical offsets of the
/// text that no single row-affine map `y ↦ c₀ + c₁·y` explains.
///
/// The page is cut into 16 vertical strips and 96-row windows; in each
/// window carrying text, the offset is the shift in `[-20, 20]` maximising
/// the correlation of the darkness profiles of `img` and `flat`, refined to
/// sub-pixel precision by a parabola through the peak.  Straight,
/// undistorted lines give offsets that are affine in `y` (zero residual);
/// curved lines leave a residual.
pub(crate) fn line_displacement(img: &Gray, flat: &Gray) -> (f64, f64) {
    assert_eq!((img.w, img.h), (flat.w, flat.h));
    let strips = 16;
    let sw = img.w / strips;
    let win = 96usize;
    let max_shift = 20i64;
    let profile = |g: &Gray, s: usize| -> Vec<f64> {
        (0..g.h)
            .map(|y| {
                (s * sw..(s + 1) * sw)
                    .map(|x| 255.0 - f64::from(g.at(x, y)))
                    .sum::<f64>()
            })
            .collect()
    };
    let mut obs: Vec<(f64, f64)> = Vec::new(); // (y, offset)
    for s in 1..strips - 1 {
        let (pi, pf) = (profile(img, s), profile(flat, s));
        let mut y0 = max_shift as usize;
        while y0 + win + (max_shift as usize) < img.h {
            let window = &pf[y0..y0 + win];
            let mean = window.iter().sum::<f64>() / win as f64;
            let var = window.iter().map(|v| (v - mean).powi(2)).sum::<f64>();
            // Skip windows without text structure.
            if var.sqrt() / (win as f64).sqrt() < 0.04 * 255.0 * sw as f64 {
                y0 += win;
                continue;
            }
            let corr: Vec<f64> = (-max_shift..=max_shift)
                .map(|d| {
                    let seg = &pi[(y0 as i64 + d) as usize..(y0 as i64 + d) as usize + win];
                    let ms = seg.iter().sum::<f64>() / win as f64;
                    let (mut sab, mut sbb) = (0.0, 0.0);
                    for (a, b) in window.iter().zip(seg) {
                        sab += (a - mean) * (b - ms);
                        sbb += (b - ms).powi(2);
                    }
                    sab / (var * sbb).sqrt().max(f64::EPSILON)
                })
                .collect();
            let i = (0..corr.len())
                .max_by(|&a, &b| corr[a].total_cmp(&corr[b]))
                .unwrap_or(0);
            if corr[i] > 0.6 {
                // Sub-pixel peak by parabolic interpolation.
                let frac = if i > 0 && i + 1 < corr.len() {
                    let (l, c, r) = (corr[i - 1], corr[i], corr[i + 1]);
                    let den = l - 2.0 * c + r;
                    if den < 0.0 { 0.5 * (l - r) / den } else { 0.0 }
                } else {
                    0.0
                };
                obs.push(((y0 + win / 2) as f64, i as f64 - max_shift as f64 + frac));
            }
            y0 += win;
        }
    }
    assert!(obs.len() >= 40, "too few text windows ({})", obs.len());
    // Remove the best row-affine offset.
    let n = obs.len() as f64;
    let (my, mo) = (
        obs.iter().map(|o| o.0).sum::<f64>() / n,
        obs.iter().map(|o| o.1).sum::<f64>() / n,
    );
    let syy: f64 = obs.iter().map(|o| (o.0 - my).powi(2)).sum();
    let syo: f64 = obs.iter().map(|o| (o.0 - my) * (o.1 - mo)).sum();
    let slope = syo / syy;
    let residuals: Vec<f64> = obs.iter().map(|o| o.1 - mo - slope * (o.0 - my)).collect();
    let rms = (residuals.iter().map(|r| r * r).sum::<f64>() / n).sqrt();
    let max = residuals.iter().fold(0.0f64, |m, r| m.max(r.abs()));
    (rms, max)
}
