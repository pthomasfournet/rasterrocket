//! Luma histograms: Otsu threshold and percentiles.

use crate::num::{to_f64, to_f64_u64};

/// 256-bin histogram of `values`.
pub(crate) fn histogram(values: impl Iterator<Item = u8>) -> [u64; 256] {
    let mut hist = [0u64; 256];
    for v in values {
        hist[usize::from(v)] += 1;
    }
    hist
}

/// Otsu's threshold of a histogram: the `t` maximising the between-class
/// variance of `{v < t}` and `{v ≥ t}`, together with the separability
/// `σ_between² / σ_total²` in `[0, 1]`.  `None` for a constant histogram.
pub(crate) fn otsu(hist: &[u64; 256]) -> Option<(u8, f64)> {
    let total: u64 = hist.iter().sum();
    if total == 0 {
        return None;
    }
    let n = to_f64_u64(total);
    let sum: f64 = hist
        .iter()
        .enumerate()
        .map(|(v, &c)| to_f64(v) * to_f64_u64(c))
        .sum();
    let mean = sum / n;
    let var_total: f64 = hist
        .iter()
        .enumerate()
        .map(|(v, &c)| (to_f64(v) - mean).powi(2) * to_f64_u64(c))
        .sum::<f64>()
        / n;
    if var_total <= f64::EPSILON {
        return None;
    }
    let (mut w0, mut sum0) = (0.0f64, 0.0f64);
    let (mut best_t, mut best_var) = (0usize, -1.0f64);
    for t in 1..256 {
        w0 += to_f64_u64(hist[t - 1]);
        sum0 = to_f64(t - 1).mul_add(to_f64_u64(hist[t - 1]), sum0);
        let w1 = n - w0;
        if w0 <= 0.0 || w1 <= 0.0 {
            continue;
        }
        let m0 = sum0 / w0;
        let m1 = (sum - sum0) / w1;
        let var_between = w0 * w1 * (m0 - m1).powi(2) / (n * n);
        if var_between > best_var {
            best_var = var_between;
            best_t = t;
        }
    }
    let t = u8::try_from(best_t).ok()?;
    Some((t, best_var / var_total))
}

/// The value below which a fraction `q ∈ [0, 1]` of the histogram mass lies.
pub(crate) fn percentile(hist: &[u64; 256], q: f64) -> u8 {
    let total: u64 = hist.iter().sum();
    let target = q.clamp(0.0, 1.0) * to_f64_u64(total);
    let mut acc = 0u64;
    for (v, &c) in hist.iter().enumerate() {
        acc += c;
        if to_f64_u64(acc) >= target && acc > 0 {
            return u8::try_from(v).unwrap_or(u8::MAX);
        }
    }
    u8::MAX
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn otsu_separates_bimodal() {
        let mut hist = [0u64; 256];
        hist[40] = 500;
        hist[220] = 500;
        let Some((t, eta)) = otsu(&hist) else {
            panic!("bimodal histogram has a threshold");
        };
        assert!((41..=220).contains(&t), "threshold {t}");
        assert!(eta > 0.99, "separability {eta}");
    }

    #[test]
    fn otsu_constant_is_none() {
        let mut hist = [0u64; 256];
        hist[128] = 10;
        assert!(otsu(&hist).is_none());
    }

    #[test]
    fn percentile_bounds() {
        let hist = histogram([10u8, 20, 30, 40].into_iter());
        assert_eq!(percentile(&hist, 0.0), 10);
        assert_eq!(percentile(&hist, 0.5), 20);
        assert_eq!(percentile(&hist, 1.0), 40);
    }
}
