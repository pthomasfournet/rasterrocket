//! Small dense linear algebra: Gaussian elimination and total-least-squares
//! line fitting.

/// Solve the `n × n` system `a · x = b` in place by Gaussian elimination with
/// partial pivoting.  `a` is row-major.  On success `b` holds `x`.
///
/// Returns `None` when a pivot falls below `1e-12` times the largest
/// absolute entry of `a` (singular or numerically rank-deficient system).
pub(crate) fn solve(a: &mut [f64], b: &mut [f64], n: usize) -> Option<()> {
    debug_assert_eq!(a.len(), n * n);
    debug_assert_eq!(b.len(), n);
    let scale = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if scale <= 0.0 || !scale.is_finite() {
        return None;
    }
    let tiny = scale * 1e-12;
    for col in 0..n {
        let pivot =
            (col..n).max_by(|&i, &j| a[i * n + col].abs().total_cmp(&a[j * n + col].abs()))?;
        if a[pivot * n + col].abs() <= tiny {
            return None;
        }
        if pivot != col {
            for k in 0..n {
                a.swap(col * n + k, pivot * n + k);
            }
            b.swap(col, pivot);
        }
        let diag = a[col * n + col];
        for row in col + 1..n {
            let factor = a[row * n + col] / diag;
            if factor == 0.0 {
                continue;
            }
            for k in col..n {
                a[row * n + k] = factor.mul_add(-a[col * n + k], a[row * n + k]);
            }
            b[row] = factor.mul_add(-b[col], b[row]);
        }
    }
    for col in (0..n).rev() {
        let mut acc = b[col];
        for k in col + 1..n {
            acc = a[col * n + k].mul_add(-b[k], acc);
        }
        b[col] = acc / a[col * n + col];
    }
    b.iter().all(|v| v.is_finite()).then_some(())
}

/// A 2-D line in normal form: the points `p` with `n · p = d`, `|n| = 1`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Line {
    pub(crate) nx: f64,
    pub(crate) ny: f64,
    pub(crate) d: f64,
}

impl Line {
    /// The line through `a` and `b`; `None` when they coincide.
    pub(crate) fn through(a: [f64; 2], b: [f64; 2]) -> Option<Self> {
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len = dx.hypot(dy);
        if len <= f64::EPSILON {
            return None;
        }
        let (nx, ny) = (-dy / len, dx / len);
        Some(Self {
            nx,
            ny,
            d: nx.mul_add(a[0], ny * a[1]),
        })
    }

    /// Signed distance of `p` from the line.
    pub(crate) fn distance(&self, p: [f64; 2]) -> f64 {
        self.nx.mul_add(p[0], self.ny * p[1]) - self.d
    }

    /// Intersection with `other`; `None` when the lines are near-parallel.
    pub(crate) fn intersect(&self, other: &Self) -> Option<[f64; 2]> {
        let det = self.nx.mul_add(other.ny, -(self.ny * other.nx));
        if det.abs() < 1e-9 {
            return None;
        }
        let x = self.d.mul_add(other.ny, -(self.ny * other.d)) / det;
        let y = self.nx.mul_add(other.d, -(self.d * other.nx)) / det;
        Some([x, y])
    }

    /// Total-least-squares fit: the line minimising the sum of squared
    /// orthogonal distances to `points`.  `None` for fewer than two points
    /// or a degenerate (isotropic) point cloud.
    pub(crate) fn fit(points: &[[f64; 2]]) -> Option<Self> {
        if points.len() < 2 {
            return None;
        }
        let n = crate::num::to_f64(points.len());
        let (sx, sy) = points
            .iter()
            .fold((0.0, 0.0), |(sx, sy), p| (sx + p[0], sy + p[1]));
        let (mx, my) = (sx / n, sy / n);
        let (mut cxx, mut cxy, mut cyy) = (0.0, 0.0, 0.0);
        for p in points {
            let (dx, dy) = (p[0] - mx, p[1] - my);
            cxx = dx.mul_add(dx, cxx);
            cxy = dx.mul_add(dy, cxy);
            cyy = dy.mul_add(dy, cyy);
        }
        // The line direction is the principal eigenvector of the scatter
        // matrix; its normal is the eigenvector of the smaller eigenvalue.
        let theta = 0.5 * (2.0 * cxy).atan2(cxx - cyy);
        let spread = (cxx - cyy).hypot(2.0 * cxy);
        if spread <= f64::EPSILON * (cxx + cyy).max(1.0) {
            return None;
        }
        let (nx, ny) = (-theta.sin(), theta.cos());
        Some(Self {
            nx,
            ny,
            d: nx.mul_add(mx, ny * my),
        })
    }

    /// [`Self::fit`] followed by `rounds` passes that discard points farther
    /// than `max(floor, k · rms)` from the current line and refit.
    pub(crate) fn fit_trimmed(
        points: &[[f64; 2]],
        rounds: usize,
        k: f64,
        floor: f64,
    ) -> Option<Self> {
        let mut line = Self::fit(points)?;
        let mut kept: Vec<[f64; 2]> = points.to_vec();
        for _ in 0..rounds {
            let n = crate::num::to_f64(kept.len());
            let rms = (kept.iter().map(|p| line.distance(*p).powi(2)).sum::<f64>() / n).sqrt();
            let limit = (k * rms).max(floor);
            let next: Vec<[f64; 2]> = kept
                .iter()
                .copied()
                .filter(|p| line.distance(*p).abs() <= limit)
                .collect();
            if next.len() < 2 || next.len() == kept.len() {
                break;
            }
            kept = next;
            line = Self::fit(&kept)?;
        }
        Some(line)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_recovers_known_solution() {
        let mut a = vec![2.0, 1.0, -1.0, -3.0, -1.0, 2.0, -2.0, 1.0, 2.0];
        let mut b = vec![8.0, -11.0, -3.0];
        solve(&mut a, &mut b, 3).unwrap();
        for (got, want) in b.iter().zip([2.0, 3.0, -1.0]) {
            assert!((got - want).abs() < 1e-9, "{got} vs {want}");
        }
    }

    #[test]
    fn solve_rejects_singular() {
        let mut a = vec![1.0, 2.0, 2.0, 4.0];
        let mut b = vec![1.0, 2.0];
        assert!(solve(&mut a, &mut b, 2).is_none());
    }

    #[test]
    fn fit_and_intersect() {
        let horizontal: Vec<[f64; 2]> = (0..10).map(|i| [f64::from(i), 5.0]).collect();
        let vertical: Vec<[f64; 2]> = (0..10).map(|i| [3.0, f64::from(i)]).collect();
        let h = Line::fit(&horizontal).unwrap();
        let v = Line::fit(&vertical).unwrap();
        let p = h.intersect(&v).unwrap();
        assert!(
            (p[0] - 3.0).abs() < 1e-9 && (p[1] - 5.0).abs() < 1e-9,
            "{p:?}"
        );
    }

    #[test]
    fn trimmed_fit_ignores_outliers() {
        let mut pts: Vec<[f64; 2]> = (0..50)
            .map(|i| [f64::from(i), 0.5f64.mul_add(f64::from(i), 1.0)])
            .collect();
        pts.push([10.0, 60.0]);
        pts.push([20.0, -40.0]);
        let line = Line::fit_trimmed(&pts, 4, 2.5, 0.1).unwrap();
        assert!(line.distance([30.0, 16.0]).abs() < 1e-6);
    }
}
