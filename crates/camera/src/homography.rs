//! Plane-to-plane homographies and the rectified size of a page quad.

use crate::Quad;
use crate::linalg::solve;
use crate::num::{round_index, to_f64};

/// A projective map `(x, y) ↦ ((h₀x + h₁y + h₂)/w, (h₃x + h₄y + h₅)/w)`
/// with `w = h₆x + h₇y + 1`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Homography([f64; 8]);

impl Homography {
    /// The homography taking each `src[i]` to `dst[i]`.  `None` when three
    /// of the points are collinear (the 8 × 8 system is singular).
    pub(crate) fn from_points(src: &[[f64; 2]; 4], dst: &[[f64; 2]; 4]) -> Option<Self> {
        let mut matrix = vec![0.0f64; 64];
        let mut rhs = vec![0.0f64; 8];
        for (i, (from, to)) in src.iter().zip(dst).enumerate() {
            let ([x, y], [u, v]) = (*from, *to);
            let row = 16 * i;
            matrix[row..row + 8].copy_from_slice(&[x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y]);
            matrix[row + 8..row + 16].copy_from_slice(&[0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y]);
            rhs[2 * i] = u;
            rhs[2 * i + 1] = v;
        }
        solve(&mut matrix, &mut rhs, 8)?;
        let mut coeffs = [0.0; 8];
        coeffs.copy_from_slice(&rhs);
        Some(Self(coeffs))
    }

    /// Apply the map.  Points on the line at infinity map to `(NaN, NaN)`,
    /// which every resampler treats as out of bounds.
    pub(crate) fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let h = &self.0;
        let w = h[6].mul_add(x, h[7].mul_add(y, 1.0));
        if w.abs() < 1e-12 {
            return (f64::NAN, f64::NAN);
        }
        (
            h[0].mul_add(x, h[1].mul_add(y, h[2])) / w,
            h[3].mul_add(x, h[4].mul_add(y, h[5])) / w,
        )
    }

    /// The map as a row-major 3 × 3 matrix acting on homogeneous
    /// coordinates `(x, y, 1)`.
    pub(crate) const fn matrix(&self) -> [f64; 9] {
        let h = &self.0;
        [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0]
    }
}

/// Width-to-height ratio of the physical rectangle imaged as `quad`.
///
/// Pinhole model with square pixels and the principal point at the image
/// centre `(cx, cy)`.  When both pairs of opposite edges converge, their
/// vanishing points fix the focal length; otherwise (one pair parallel in
/// the image, or an estimate outside `[0.3, 5] · f_default`) the focal
/// length is taken as `f_default`.  The aspect then follows from the
/// back-projected edge directions.  When the result is not physical (ratio
/// outside `[0.2, 5]`, or more than 25 % away from the ratio of mean
/// opposite edge lengths), the edge-length ratio is returned instead.
pub(crate) fn physical_aspect(quad: &Quad, cx: f64, cy: f64, f_default: f64) -> f64 {
    let [tl, tr, br, bl] = quad.corners;
    let fallback = (dist(tl, tr) + dist(bl, br)) / (dist(tl, bl) + dist(tr, br)).max(f64::EPSILON);
    // Corners relative to the principal point make (u0, v0) = (0, 0).
    let m = |p: [f64; 2]| [p[0] - cx, p[1] - cy, 1.0];
    let (m1, m2, m3, m4) = (m(tl), m(tr), m(bl), m(br));
    let k2 = dot(cross(m1, m4), m3) / dot(cross(m2, m4), m3);
    let k3 = dot(cross(m1, m4), m2) / dot(cross(m3, m4), m2);
    if !k2.is_finite() || !k3.is_finite() {
        return fallback;
    }
    // n2, n3: images of the rectangle's edge directions, up to scale.
    let n2 = [
        k2.mul_add(m2[0], -m1[0]),
        k2.mul_add(m2[1], -m1[1]),
        k2 - 1.0,
    ];
    let n3 = [
        k3.mul_add(m3[0], -m1[0]),
        k3.mul_add(m3[1], -m1[1]),
        k3 - 1.0,
    ];
    // `k − 1` vanishes when a pair of edges is parallel in the image; the
    // focal-length equation then degenerates to 0/0.
    let estimated = (n2[2].abs() > 0.01 && n3[2].abs() > 0.01)
        .then(|| -n2[0].mul_add(n3[0], n2[1] * n3[1]) / (n2[2] * n3[2]))
        .filter(|f2| f2.is_finite() && *f2 > 0.0)
        .map(f64::sqrt)
        .filter(|f| (0.3 * f_default..=5.0 * f_default).contains(f));
    let f2 = estimated.unwrap_or(f_default).powi(2);
    let num = n2[2].mul_add(n2[2], n2[0].mul_add(n2[0], n2[1] * n2[1]) / f2);
    let den = n3[2].mul_add(n3[2], n3[0].mul_add(n3[0], n3[1] * n3[1]) / f2);
    let ratio = (num / den).sqrt();
    if ratio.is_finite() && (0.2..=5.0).contains(&ratio) && (ratio / fallback - 1.0).abs() <= 0.25 {
        ratio
    } else {
        fallback
    }
}

/// Focal length, in pixels, assumed when the page quad does not determine
/// it, relative to the longer image side (a normal lens, ≈ 53° across the
/// long side).
const DEFAULT_FOCAL_REL: f64 = 1.0;

/// Output dimensions for rectifying `quad` from an `img_w × img_h` image.
/// The width-to-height ratio is [`physical_aspect`]; the scale is the
/// largest that no quad edge is shrunk by.  Declined when either side would
/// be < 16 px or the area would exceed twice the source area.
pub(crate) fn rectified_size(
    quad: &Quad,
    img_w: usize,
    img_h: usize,
) -> Result<(usize, usize), String> {
    let [tl, tr, br, bl] = quad.corners;
    let cx = (to_f64(img_w) - 1.0) / 2.0;
    let cy = (to_f64(img_h) - 1.0) / 2.0;
    let f_default = DEFAULT_FOCAL_REL * to_f64(img_w.max(img_h));
    let aspect = physical_aspect(quad, cx, cy, f_default);
    let h = dist(tl, bl)
        .max(dist(tr, br))
        .max(dist(tl, tr).max(dist(bl, br)) / aspect);
    let w = h * aspect;
    if !(w >= 16.0 && h >= 16.0) {
        return Err(format!(
            "rectified page would be {w:.1}×{h:.1} px; both sides must be at least 16 px"
        ));
    }
    if w * h > 2.0 * to_f64(img_w) * to_f64(img_h) {
        return Err(format!(
            "rectified page would be {w:.0}×{h:.0} px, more than twice the {img_w}×{img_h} source"
        ));
    }
    Ok((round_index(w), round_index(h)))
}

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1].mul_add(b[2], -(a[2] * b[1])),
        a[2].mul_add(b[0], -(a[0] * b[2])),
        a[0].mul_add(b[1], -(a[1] * b[0])),
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0].mul_add(b[0], a[1].mul_add(b[1], a[2] * b[2]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn homography_maps_corners() {
        let src = [[0.0, 0.0], [99.0, 0.0], [99.0, 149.0], [0.0, 149.0]];
        let dst = [[12.0, 7.0], [180.0, 20.0], [170.0, 260.0], [5.0, 240.0]];
        let h = Homography::from_points(&src, &dst).unwrap();
        for (s, d) in src.iter().zip(&dst) {
            let (x, y) = h.apply(s[0], s[1]);
            assert!((x - d[0]).abs() < 1e-6 && (y - d[1]).abs() < 1e-6);
        }
    }

    #[test]
    fn collinear_points_are_rejected() {
        let src = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]];
        let dst = [[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [0.0, 1.0]];
        assert!(Homography::from_points(&src, &dst).is_none());
    }

    /// Corners of a `width × height` rectangle, rotated by `tilt_y` about
    /// its vertical axis and then by `tilt_x` about its horizontal axis,
    /// placed 700 units from a pinhole camera of focal length `focal` whose
    /// principal point is `(500, 400)`.
    fn project_rectangle(width: f64, height: f64, tilt_x: f64, tilt_y: f64, focal: f64) -> Quad {
        let project = |px: f64, py: f64| {
            let (px, py) = (px - width / 2.0, py - height / 2.0);
            let (x1, z1) = (px * tilt_y.cos(), -px * tilt_y.sin());
            let (y2, z2) = (
                py.mul_add(tilt_x.cos(), -(z1 * tilt_x.sin())),
                py.mul_add(tilt_x.sin(), z1 * tilt_x.cos()),
            );
            let depth = z2 + 700.0;
            [
                focal.mul_add(x1 / depth, 500.0),
                focal.mul_add(y2 / depth, 400.0),
            ]
        };
        Quad {
            corners: [
                project(0.0, 0.0),
                project(width, 0.0),
                project(width, height),
                project(0.0, height),
            ],
        }
    }

    #[test]
    fn aspect_recovered_from_two_vanishing_points() {
        let quad = project_rectangle(210.0, 297.0, 0.25, -0.18, 900.0);
        // The focal length is estimated from the quad; the default is a decoy.
        let got = physical_aspect(&quad, 500.0, 400.0, 1500.0);
        let want = 210.0 / 297.0;
        assert!((got - want).abs() / want < 1e-6, "aspect {got} vs {want}");
    }

    #[test]
    fn aspect_with_one_vanishing_point_uses_default_focal() {
        // Tilt about the horizontal axis only: the top and bottom edges stay
        // parallel and the focal length is unobservable.
        let quad = project_rectangle(210.0, 297.0, 0.3, 0.0, 900.0);
        let got = physical_aspect(&quad, 500.0, 400.0, 900.0);
        let want = 210.0 / 297.0;
        assert!((got - want).abs() / want < 1e-6, "aspect {got} vs {want}");
    }

    #[test]
    fn affine_quad_uses_image_aspect() {
        let quad = Quad {
            corners: [[10.0, 10.0], [110.0, 10.0], [110.0, 210.0], [10.0, 210.0]],
        };
        assert!((physical_aspect(&quad, 60.0, 110.0, 200.0) - 0.5).abs() < 1e-9);
    }
}
