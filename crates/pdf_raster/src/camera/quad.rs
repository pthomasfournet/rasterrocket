//! Page-boundary quadrilateral detection.
//!
//! # Model
//!
//! A camera capture shows a light sheet of paper against a darker
//! surround (desk, stand, hands).  On a box-downsampled luma plane:
//!
//! 1. Otsu's threshold splits paper from surround; the largest 4-connected
//!    bright component, with its enclosed holes (text, figures) filled, is
//!    the page mask.
//! 2. The convex hull of the mask is reduced to its maximum-area inscribed
//!    quadrilateral, whose vertices seed the four page edges.
//! 3. Each edge is refitted by trimmed total least squares to the mask
//!    boundary points near it, then refined again on the full-resolution
//!    plane by locating the paper/surround luma step along the edge normal
//!    to sub-pixel precision.  Adjacent edge lines intersect in the corners.
//!
//! # Acceptance
//!
//! The quad is returned only when every one of these holds; otherwise the
//! detector reports no page and the image is left untouched:
//!
//! - Otsu separability ≥ [`MIN_SEPARABILITY`];
//! - mask area within [`MIN_AREA_FRAC`]..=[`MAX_AREA_FRAC`] of the image
//!   (a page filling the frame has nothing to crop);
//! - median paper luma exceeds median surround luma by ≥ [`MIN_CONTRAST`];
//! - the quad is convex with interior angles in `[50°, 130°]`;
//! - quad/mask intersection-over-union ≥ [`MIN_IOU`];
//! - at least [`MIN_MARGIN_SIDES`] quad sides are separated from the image
//!   border by surround, which rejects a dark band running off the edge of
//!   an otherwise full-frame page.

use super::Quad;
use super::linalg::Line;
use super::plane::{Plane, histogram, otsu, percentile, to_f64};

/// Longest side of the analysis plane, in pixels.
const ANALYSIS_SIDE: usize = 640;
/// Minimum Otsu separability of paper versus surround.
const MIN_SEPARABILITY: f64 = 0.45;
/// Minimum page-mask area as a fraction of the image.
const MIN_AREA_FRAC: f64 = 0.15;
/// Maximum page-mask area as a fraction of the image.
const MAX_AREA_FRAC: f64 = 0.96;
/// Minimum median-luma difference between paper and surround.
const MIN_CONTRAST: u8 = 35;
/// Minimum intersection-over-union of the quad with the page mask.
const MIN_IOU: f64 = 0.92;
/// Minimum number of quad sides with surround between them and the border.
const MIN_MARGIN_SIDES: usize = 3;
/// Upper bound on hull vertices entering the quadratic-cubic quad search.
const MAX_HULL_VERTICES: usize = 160;

/// Detect the page quadrilateral in a full-resolution luma plane.
pub fn detect(luma: &Plane) -> Option<Quad> {
    let factor = luma.factor_for(ANALYSIS_SIDE);
    let small = luma.downsample(factor);
    let (w, h) = (small.w, small.h);
    if w < 32 || h < 32 {
        return None;
    }
    let (threshold, separability) = otsu(&histogram(small.data.iter().copied()))?;
    if separability < MIN_SEPARABILITY {
        log::debug!("camera: page quad rejected: separability {separability:.2}");
        return None;
    }
    let mask = page_mask(&small, threshold)?;
    let area = mask.iter().filter(|&&m| m).count();
    let frac = to_f64(area) / to_f64(w * h);
    if !(MIN_AREA_FRAC..=MAX_AREA_FRAC).contains(&frac) {
        log::debug!("camera: page quad rejected: mask covers {frac:.3} of the frame");
        return None;
    }
    let inside = histogram(
        small
            .data
            .iter()
            .zip(&mask)
            .filter(|(_, m)| **m)
            .map(|(v, _)| *v),
    );
    let outside = histogram(
        small
            .data
            .iter()
            .zip(&mask)
            .filter(|(_, m)| !**m)
            .map(|(v, _)| *v),
    );
    let contrast = percentile(&inside, 0.5).saturating_sub(percentile(&outside, 0.5));
    if contrast < MIN_CONTRAST {
        log::debug!("camera: page quad rejected: contrast {contrast}");
        return None;
    }

    let hull = reduce_hull(convex_hull(mask_extremes(&mask, w, h)), MAX_HULL_VERTICES);
    let seed = order_corners(max_area_quad(&hull)?);
    let boundary = boundary_points(&mask, w, h);
    let diag = to_f64(w).hypot(to_f64(h));
    let coarse = refine_from_points(&seed, &boundary, diag);
    if !plausible(&coarse, &mask, w, h) {
        return None;
    }

    let up = |p: [f64; 2]| {
        [
            Plane::upscale_coord(p[0], factor),
            Plane::upscale_coord(p[1], factor),
        ]
    };
    let full = coarse.map(up);
    let refined = refine_on_plane(luma, &full, factor);
    Some(Quad { corners: refined })
}

// ── Mask ────────────────────────────────────────────────────────────────────

/// Largest 4-connected component of `{v ≥ threshold}` with its holes filled.
fn page_mask(plane: &Plane, threshold: u8) -> Option<Vec<bool>> {
    let (w, h) = (plane.w, plane.h);
    let bright: Vec<bool> = plane.data.iter().map(|&v| v >= threshold).collect();
    let mut label = vec![0u32; w * h];
    let mut best = (0u32, 0usize);
    let mut next = 0u32;
    let mut stack = Vec::new();
    for start in 0..w * h {
        if !bright[start] || label[start] != 0 {
            continue;
        }
        next += 1;
        let size = flood(start, w, h, &mut stack, |i| {
            if bright[i] && label[i] == 0 {
                label[i] = next;
                true
            } else {
                false
            }
        });
        if size > best.1 {
            best = (next, size);
        }
    }
    if best.1 == 0 {
        return None;
    }
    let component: Vec<bool> = label.iter().map(|&l| l == best.0).collect();
    // Everything not reachable from the border through non-page pixels is
    // either the page or enclosed by it.
    let mut outside = vec![false; w * h];
    for i in border_indices(w, h) {
        if !component[i] && !outside[i] {
            let _ = flood(i, w, h, &mut stack, |j| {
                if component[j] || outside[j] {
                    false
                } else {
                    outside[j] = true;
                    true
                }
            });
        }
    }
    Some(outside.into_iter().map(|o| !o).collect())
}

/// Iterative 4-connected flood fill from `start`; `visit` claims a pixel and
/// reports whether it belongs to the region.  Returns the region size.
fn flood(
    start: usize,
    w: usize,
    h: usize,
    stack: &mut Vec<usize>,
    mut visit: impl FnMut(usize) -> bool,
) -> usize {
    stack.clear();
    if !visit(start) {
        return 0;
    }
    stack.push(start);
    let mut size = 0;
    while let Some(i) = stack.pop() {
        size += 1;
        let (x, y) = (i % w, i / w);
        let mut push = |j: usize, stack: &mut Vec<usize>| {
            if visit(j) {
                stack.push(j);
            }
        };
        if x > 0 {
            push(i - 1, stack);
        }
        if x + 1 < w {
            push(i + 1, stack);
        }
        if y > 0 {
            push(i - w, stack);
        }
        if y + 1 < h {
            push(i + w, stack);
        }
    }
    size
}

fn border_indices(w: usize, h: usize) -> impl Iterator<Item = usize> {
    let rows = (0..w).flat_map(move |x| [x, (h - 1) * w + x]);
    let cols = (0..h).flat_map(move |y| [y * w, y * w + w - 1]);
    rows.chain(cols)
}

/// Corners of the outermost mask pixel squares on every row: a point set
/// whose convex hull equals the hull of the mask's pixel squares.
fn mask_extremes(mask: &[bool], w: usize, h: usize) -> Vec<[f64; 2]> {
    let mut pts = Vec::with_capacity(4 * h);
    for y in 0..h {
        let row = &mask[y * w..(y + 1) * w];
        let (Some(l), Some(r)) = (row.iter().position(|&m| m), row.iter().rposition(|&m| m)) else {
            continue;
        };
        let (fy, fl, fr) = (to_f64(y), to_f64(l), to_f64(r));
        pts.extend([
            [fl - 0.5, fy - 0.5],
            [fl - 0.5, fy + 0.5],
            [fr + 0.5, fy - 0.5],
            [fr + 0.5, fy + 0.5],
        ]);
    }
    pts
}

/// Mask boundary locations: the midpoint of every edge between a mask pixel
/// and an in-image non-mask 4-neighbour.  Edges against the image border are
/// not page boundary and are omitted.
fn boundary_points(mask: &[bool], w: usize, h: usize) -> Vec<[f64; 2]> {
    let mut pts = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if !mask[y * w + x] {
                continue;
            }
            let (fx, fy) = (to_f64(x), to_f64(y));
            if x > 0 && !mask[y * w + x - 1] {
                pts.push([fx - 0.5, fy]);
            }
            if x + 1 < w && !mask[y * w + x + 1] {
                pts.push([fx + 0.5, fy]);
            }
            if y > 0 && !mask[(y - 1) * w + x] {
                pts.push([fx, fy - 0.5]);
            }
            if y + 1 < h && !mask[(y + 1) * w + x] {
                pts.push([fx, fy + 0.5]);
            }
        }
    }
    pts
}

// ── Hull and quad ───────────────────────────────────────────────────────────

fn cross(o: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - o[0]).mul_add(b[1] - o[1], -((a[1] - o[1]) * (b[0] - o[0])))
}

/// Convex hull by Andrew's monotone chain, counter-clockwise in a y-up
/// frame, collinear points removed.
fn convex_hull(mut pts: Vec<[f64; 2]>) -> Vec<[f64; 2]> {
    pts.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    pts.dedup();
    if pts.len() < 3 {
        return pts;
    }
    let mut hull: Vec<[f64; 2]> = Vec::with_capacity(2 * pts.len());
    for pass in 0..2 {
        let start = hull.len();
        let iter: Box<dyn Iterator<Item = &[f64; 2]>> = if pass == 0 {
            Box::new(pts.iter())
        } else {
            Box::new(pts.iter().rev())
        };
        for &p in iter {
            while hull.len() >= start + 2
                && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0
            {
                let _ = hull.pop();
            }
            hull.push(p);
        }
        let _ = hull.pop();
    }
    hull
}

/// Drop the vertex spanning the smallest triangle with its neighbours until
/// at most `max` remain; the area lost per step is minimal.
fn reduce_hull(mut hull: Vec<[f64; 2]>, max: usize) -> Vec<[f64; 2]> {
    while hull.len() > max.max(4) {
        let n = hull.len();
        let victim = (0..n)
            .min_by(|&i, &j| {
                let area =
                    |k: usize| cross(hull[(k + n - 1) % n], hull[k], hull[(k + 1) % n]).abs();
                area(i).total_cmp(&area(j))
            })
            .unwrap_or(0);
        let _ = hull.remove(victim);
    }
    hull
}

/// The four hull vertices enclosing the largest area, in hull order.
fn max_area_quad(hull: &[[f64; 2]]) -> Option<[[f64; 2]; 4]> {
    let n = hull.len();
    if n < 4 {
        return None;
    }
    let tri = |a: usize, b: usize, c: usize| cross(hull[a], hull[b], hull[c]).abs();
    let mut best = (0.0f64, [0usize; 4]);
    for i in 0..n {
        for k in i + 2..n {
            let Some((ja, j)) = (i + 1..k)
                .map(|j| (tri(i, j, k), j))
                .max_by(|a, b| a.0.total_cmp(&b.0))
            else {
                continue;
            };
            let Some((la, l)) = (k + 1..n)
                .map(|l| (tri(i, k, l), l))
                .max_by(|a, b| a.0.total_cmp(&b.0))
            else {
                continue;
            };
            if ja + la > best.0 {
                best = (ja + la, [i, j, k, l]);
            }
        }
    }
    (best.0 > 0.0).then(|| best.1.map(|i| hull[i]))
}

/// Order four corners as top-left, top-right, bottom-right, bottom-left
/// (clockwise on screen, y down).
fn order_corners(mut c: [[f64; 2]; 4]) -> [[f64; 2]; 4] {
    let cx = c.iter().map(|p| p[0]).sum::<f64>() / 4.0;
    let cy = c.iter().map(|p| p[1]).sum::<f64>() / 4.0;
    c.sort_by(|a, b| {
        (a[1] - cy)
            .atan2(a[0] - cx)
            .total_cmp(&(b[1] - cy).atan2(b[0] - cx))
    });
    let first = (0..4)
        .min_by(|&i, &j| (c[i][0] + c[i][1]).total_cmp(&(c[j][0] + c[j][1])))
        .unwrap_or(0);
    c.rotate_left(first);
    c
}

// ── Edge refinement ─────────────────────────────────────────────────────────

/// Intersect the four edge lines into corners.  A corner that moves farther
/// than `max_shift` from `fallback` keeps the fallback position.
fn corners_from_lines(
    lines: &[Line; 4],
    fallback: &[[f64; 2]; 4],
    max_shift: f64,
) -> [[f64; 2]; 4] {
    let mut out = *fallback;
    for (i, corner) in out.iter_mut().enumerate() {
        // Corner i joins edge i−1 (ending at it) and edge i (starting at it).
        if let Some(p) = lines[(i + 3) % 4].intersect(&lines[i]) {
            let moved = (p[0] - fallback[i][0]).hypot(p[1] - fallback[i][1]);
            if moved <= max_shift {
                *corner = p;
            }
        }
    }
    out
}

/// Number of stations along an edge in [`near_edge`].
const EDGE_BINS: usize = 64;

/// The outermost point of `pts` within `tol` of edge `a → b` in each of
/// [`EDGE_BINS`] stations along it, excluding the 8 % at each end where
/// rounded or occluded corners bend away from the edge line.
///
/// Dark content touching the page edge notches the mask inward; keeping
/// only the outermost point per station discards the notches.  `a → b`
/// must run clockwise on screen (the [`order_corners`] order), so the
/// outward side is the negative side of [`Line::through`].
fn near_edge(
    pts: &[[f64; 2]],
    a: [f64; 2],
    b: [f64; 2],
    tol: f64,
) -> Option<(Line, Vec<[f64; 2]>)> {
    let line = Line::through(a, b)?;
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len2 = dx.mul_add(dx, dy * dy);
    let mut best: Vec<Option<(f64, [f64; 2])>> = vec![None; EDGE_BINS];
    for &p in pts {
        let t = (p[0] - a[0]).mul_add(dx, (p[1] - a[1]) * dy) / len2;
        let outward = -line.distance(p);
        if !(0.08..0.92).contains(&t) || outward.abs() > tol {
            continue;
        }
        let bin =
            super::plane::floor_index((t - 0.08) / 0.84 * to_f64(EDGE_BINS)).min(EDGE_BINS - 1);
        if best[bin].is_none_or(|(o, _)| outward > o) {
            best[bin] = Some((outward, p));
        }
    }
    Some((line, best.into_iter().flatten().map(|(_, p)| p).collect()))
}

/// Refit each edge to the mask boundary points near it.
fn refine_from_points(seed: &[[f64; 2]; 4], boundary: &[[f64; 2]], diag: f64) -> [[f64; 2]; 4] {
    let tol = (0.03 * diag).max(2.0);
    let mut lines = [Line {
        nx: 0.0,
        ny: 0.0,
        d: 0.0,
    }; 4];
    for (i, line) in lines.iter_mut().enumerate() {
        let (a, b) = (seed[i], seed[(i + 1) % 4]);
        let Some((through, near)) = near_edge(boundary, a, b, tol) else {
            return *seed;
        };
        *line = if near.len() >= 12 {
            Line::fit_trimmed(&near, 4, 2.0, 0.75).unwrap_or(through)
        } else {
            through
        };
    }
    corners_from_lines(&lines, seed, 0.05 * diag)
}

/// Refine each edge on the full-resolution plane: along the outward normal
/// at 48 stations per edge, locate the half-way crossing between the paper
/// and surround luma levels, then refit the edge to the crossings.  Edges
/// without a clear step (e.g. lying on the image border) keep their line.
fn refine_on_plane(luma: &Plane, quad: &[[f64; 2]; 4], factor: usize) -> [[f64; 2]; 4] {
    let reach = 2.0f64.mul_add(to_f64(factor), 3.0);
    let cx = quad.iter().map(|p| p[0]).sum::<f64>() / 4.0;
    let cy = quad.iter().map(|p| p[1]).sum::<f64>() / 4.0;
    let mut lines = [Line {
        nx: 0.0,
        ny: 0.0,
        d: 0.0,
    }; 4];
    for (i, line) in lines.iter_mut().enumerate() {
        let (a, b) = (quad[i], quad[(i + 1) % 4]);
        let Some(through) = Line::through(a, b) else {
            return *quad;
        };
        // Orient the normal away from the quad centre.
        let sign = if through.distance([cx, cy]) > 0.0 {
            -1.0
        } else {
            1.0
        };
        let (nx, ny) = (through.nx * sign, through.ny * sign);
        let mut crossings = Vec::with_capacity(48);
        for k in 0..48u32 {
            let t = (f64::from(k) + 0.5).mul_add(0.84 / 48.0, 0.08);
            let p = [t.mul_add(b[0] - a[0], a[0]), t.mul_add(b[1] - a[1], a[1])];
            if let Some(o) = step_offset(luma, p, [nx, ny], reach) {
                crossings.push([o.mul_add(nx, p[0]), o.mul_add(ny, p[1])]);
            }
        }
        *line = if crossings.len() >= 16 {
            Line::fit_trimmed(&crossings, 4, 2.0, 0.5).unwrap_or(through)
        } else {
            through
        };
    }
    corners_from_lines(&lines, quad, 2.0 * reach)
}

/// Offset along `n` from `p`, within `±reach`, at which the luma profile
/// crosses midway between its inner (paper) and outer (surround) levels.
fn step_offset(luma: &Plane, p: [f64; 2], n: [f64; 2], reach: f64) -> Option<f64> {
    let step = 0.5;
    let samples = super::plane::floor_index(2.0 * reach / step) + 1;
    let at = |o: f64| luma.sample(o.mul_add(n[0], p[0]), o.mul_add(n[1], p[1]));
    let offsets: Vec<f64> = (0..samples)
        .map(|i| to_f64(i).mul_add(step, -reach))
        .collect();
    let profile: Vec<f64> = offsets.iter().map(|&o| at(o)).collect();
    let inner = profile[..3].iter().sum::<f64>() / 3.0;
    let outer = profile[samples - 3..].iter().sum::<f64>() / 3.0;
    if inner - outer < 20.0 {
        return None;
    }
    let mid = f64::midpoint(inner, outer);
    let i = profile.windows(2).position(|w| w[0] >= mid && w[1] < mid)?;
    let frac = (profile[i] - mid) / (profile[i] - profile[i + 1]);
    Some(frac.mul_add(step, offsets[i]))
}

// ── Plausibility ────────────────────────────────────────────────────────────

fn plausible(q: &[[f64; 2]; 4], mask: &[bool], w: usize, h: usize) -> bool {
    // Convexity and interior angles.
    let mut orientation: Option<bool> = None;
    for i in 0..4 {
        let (prev, corner, next) = (q[(i + 3) % 4], q[i], q[(i + 1) % 4]);
        let turn = cross(prev, corner, next);
        let positive = turn > 0.0;
        if turn.abs() <= f64::EPSILON || orientation.is_some_and(|o| o != positive) {
            log::debug!("camera: page quad rejected: not convex");
            return false;
        }
        orientation = Some(positive);
        let (ux, uy) = (prev[0] - corner[0], prev[1] - corner[1]);
        let (vx, vy) = (next[0] - corner[0], next[1] - corner[1]);
        let angle = (ux.mul_add(vx, uy * vy) / (ux.hypot(uy) * vx.hypot(vy)))
            .acos()
            .to_degrees();
        if !(50.0..=130.0).contains(&angle) {
            log::debug!("camera: page quad rejected: corner angle {angle:.1}°");
            return false;
        }
    }
    // Agreement with the mask.
    let sign = if orientation == Some(false) {
        -1.0
    } else {
        1.0
    };
    let (mut inter, mut union) = (0usize, 0usize);
    for y in 0..h {
        for x in 0..w {
            let pt = [to_f64(x), to_f64(y)];
            let inside = (0..4).all(|i| cross(q[i], q[(i + 1) % 4], pt) * sign >= 0.0);
            let m = mask[y * w + x];
            inter += usize::from(inside && m);
            union += usize::from(inside || m);
        }
    }
    let iou = to_f64(inter) / to_f64(union.max(1));
    if iou < MIN_IOU {
        log::debug!("camera: page quad rejected: IoU {iou:.3}");
        return false;
    }
    // Surround between the quad and the image border.
    let (mx, my) = ((0.01 * to_f64(w)).max(2.0), (0.01 * to_f64(h)).max(2.0));
    let (fw, fh) = (to_f64(w) - 0.5, to_f64(h) - 0.5);
    let sides = [
        q[0][1].min(q[1][1]) > my - 0.5,
        q[1][0].max(q[2][0]) < fw - mx,
        q[2][1].max(q[3][1]) < fh - my,
        q[0][0].min(q[3][0]) > mx - 0.5,
    ];
    let margin_sides = sides.iter().filter(|&&s| s).count();
    if margin_sides < MIN_MARGIN_SIDES {
        log::debug!("camera: page quad rejected: surround on {margin_sides} side(s)");
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::testutil::plane;

    /// A light quadrilateral with dark "text" bars on a dark surround.
    fn page_on_desk(w: usize, h: usize, q: [[f64; 2]; 4]) -> Plane {
        plane(w, h, |x, y| {
            let p = [to_f64(x), to_f64(y)];
            let inside = (0..4).all(|i| cross(q[i], q[(i + 1) % 4], p) >= 0.0);
            if !inside {
                return 70;
            }
            if (y / 9) % 3 == 0 && (x / 23) % 4 != 3 {
                40
            } else {
                235
            }
        })
    }

    #[test]
    fn detects_skewed_quad_to_subpixel() {
        // Clockwise on screen (y down): TL, TR, BR, BL.
        let q = [
            [120.3, 80.7],
            [880.2, 130.4],
            [840.6, 1180.1],
            [90.4, 1120.9],
        ];
        let p = page_on_desk(1000, 1300, q);
        let got = detect(&p).expect("quad");
        for (g, want) in got.corners.iter().zip(&q) {
            let err = (g[0] - want[0]).hypot(g[1] - want[1]);
            assert!(err < 1.5, "corner {g:?} vs {want:?} (err {err:.2})");
        }
    }

    #[test]
    fn full_frame_page_is_not_a_quad() {
        let p = plane(800, 1000, |x, y| {
            if (y / 9) % 3 == 0 && (x / 23) % 4 != 3 {
                30
            } else {
                240
            }
        });
        assert!(detect(&p).is_none());
    }

    #[test]
    fn dark_band_off_one_edge_is_rejected() {
        // A full-width dark header band: the bright region is a clean
        // rectangle, but surround borders it on one side only.
        let p = plane(800, 1000, |_, y| if y < 150 { 50 } else { 240 });
        assert!(detect(&p).is_none());
    }

    #[test]
    fn low_contrast_surround_is_rejected() {
        let q = [
            [100.0, 100.0],
            [700.0, 100.0],
            [700.0, 900.0],
            [100.0, 900.0],
        ];
        let p = plane(800, 1000, |x, y| {
            let inside = (0..4).all(|i| cross(q[i], q[(i + 1) % 4], [to_f64(x), to_f64(y)]) >= 0.0);
            if inside { 235 } else { 220 }
        });
        assert!(detect(&p).is_none());
    }

    #[test]
    fn order_corners_is_clockwise_from_top_left() {
        let c = order_corners([[10.0, 90.0], [90.0, 10.0], [10.0, 10.0], [90.0, 90.0]]);
        assert_eq!(c, [[10.0, 10.0], [90.0, 10.0], [90.0, 90.0], [10.0, 90.0]]);
    }
}
