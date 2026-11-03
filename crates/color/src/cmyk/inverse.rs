//! RGB → CMYK inverse of the process-ink model, and ink-canvas → RGB conversion.
//!
//! [`rgb_to_ink`] inverts the quadrilinear model of [`CORNERS`](super::CORNERS)
//! with grey-component replacement: of the inks printing a colour it
//! chooses ones with the most black.  A slice of the model at black `k` is
//! a convex combination of its eight corners, so a colour it prints has
//! no channel above that channel's largest corner value; the largest `k`
//! meeting this bound on every channel ([`black_bound`]) caps the black.
//! The bound is not the grey of the lightest channel: the M + K and
//! M + Y + K corners are redder than K alone.  The blacks reproducing a
//! colour form bands below the bound, not always reaching 0: in the dark
//! slices adding cyan lightens blue, so a dark colour may print only
//! within a narrow band.  A node's black is found by sampling a ladder of
//! slices down from the bound and bisecting the upper edge of the band
//! around the highest reproducing sample, then stepping [`MARGIN`] inside
//! the band where the band allows, since at the edge the CMY sit on a
//! bound of the slice; the colour's CMY is the preimage on that slice.  A colour nothing
//! prints takes the ladder sample whose least-squares CMY comes closest.
//!
//! Each slice is the trilinear interpolation of the eight corners at that
//! black, inverted by projected Gauss-Newton in `f64` (channels at a bound
//! with an outward-pointing gradient are frozen and the step is solved on
//! the free channels).  The inverse is tabulated on a 65³ grid over the
//! sRGB cube; where adjacent nodes chose different bands, relaxation
//! passes re-solve a node far from its neighbours' mean on their band.  A
//! query is the trilinear interpolation of the eight surrounding nodes,
//! except that a neutral takes the inks of its grey from a 256-entry
//! table — black alone down to the floor ([`BLACK_FLOOR`]), solid black
//! with C, M and Y below it — since interpolating the chromatic neighbours
//! of the grey axis would add C, M and Y to it.
//!
//! The table reproduces every colour the model prints within two counts
//! per channel, except for about one in two thousand dark colours
//! (channel mean below 64) at the edge of a thin band, within four.

use std::sync::LazyLock;

use super::{CORNERS, cmyk_to_rgb, cmyk_to_rgb_row};

/// Nodes per axis of the inverse table.
const GRID: u8 = 65;
/// Projected Gauss-Newton iterations per slice solve.
const MAX_STEPS: usize = 40;
/// Bisection steps for the largest reproducing black.
const BISECTIONS: usize = 6;
/// Squared residual below which a slice solve reproduces the colour.
const EXACT: f64 = 0.25;
/// Black stepped back from the edge of a band of reproducing blacks.
const MARGIN: f64 = 0.03;
/// Relaxation passes over the table after the node solves.
const PASSES: usize = 3;
/// Ink distance from the neighbours' mean at which a node is re-solved.
const OUTLIER: f64 = 0.08;
/// Pixels converted per stack buffer in [`ink_to_rgb_row`].
const CHUNK: usize = 256;

/// The darkest grey black ink alone prints: the solid-black corner.
pub const BLACK_FLOOR: u8 = CORNERS[8][0];

/// Process-ink equivalent of each tint (0–255) of one spot colorant:
/// entry `t` is the `[C, M, Y, K]` bytes printing the same colour as the
/// spot at tint `t`.
pub type SpotLut = [[u8; 4]; 256];

/// The eight CMY corners of the model at black `k`: `CORNERS[i]` and
/// `CORNERS[8 + i]` interpolated, index `c·4 + m·2 + y`.
fn slice_corners(k: f64) -> [[f64; 3]; 8] {
    std::array::from_fn(|i| {
        std::array::from_fn(|ch| {
            let (no_k, solid_k) = (f64::from(CORNERS[i][ch]), f64::from(CORNERS[8 + i][ch]));
            (solid_k - no_k).mul_add(k, no_k)
        })
    })
}

/// The offsets `[c, m, y]` (each 0 or 1) of trilinear corner `corner`,
/// indexed `c·4 + m·2 + y`.
const fn corner_bits(corner: usize) -> [usize; 3] {
    [(corner >> 2) & 1, (corner >> 1) & 1, corner & 1]
}

/// The trilinear weight of corner `corner` at fractional position `t`.
fn corner_weight(corner: usize, t: [f64; 3]) -> f64 {
    corner_bits(corner)
        .iter()
        .zip(t)
        .map(|(&bit, t)| if bit == 1 { t } else { 1.0 - t })
        .product()
}

/// Forward model on a black slice, unrounded: trilinear over `corners`.
fn forward(corners: &[[f64; 3]; 8], cmy: [f64; 3]) -> [f64; 3] {
    let mut out = [0.0; 3];
    for (i, corner) in corners.iter().enumerate() {
        let w = corner_weight(i, cmy);
        for (o, &v) in out.iter_mut().zip(corner) {
            *o = w.mul_add(v, *o);
        }
    }
    out
}

/// Sum of squared residuals of `forward(corners, cmy)` against `target`.
fn residual(corners: &[[f64; 3]; 8], cmy: [f64; 3], target: [f64; 3]) -> f64 {
    forward(corners, cmy)
        .iter()
        .zip(target)
        .map(|(a, b)| (a - b) * (a - b))
        .sum()
}

/// Solve `a·x = b` for a 3×3 system by Cramer's rule; `None` if singular.
fn solve3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let det = |mat: [[f64; 3]; 3]| {
        mat[0][2].mul_add(
            mat[1][0].mul_add(mat[2][1], -(mat[1][1] * mat[2][0])),
            mat[0][1].mul_add(
                mat[1][2].mul_add(mat[2][0], -(mat[1][0] * mat[2][2])),
                mat[0][0] * mat[1][1].mul_add(mat[2][2], -(mat[1][2] * mat[2][1])),
            ),
        )
    };
    let d = det(a);
    if d.abs() < 1e-9 {
        return None;
    }
    let mut x = [0.0; 3];
    for (col, xc) in x.iter_mut().enumerate() {
        let mut replaced = a;
        for (row, line) in replaced.iter_mut().enumerate() {
            line[col] = b[row];
        }
        *xc = det(replaced) / d;
    }
    Some(x)
}

/// CMY in `[0, 1]³` minimising the squared distance of the slice image to
/// `target` (RGB bytes as `f64`), by projected Gauss-Newton from `start`;
/// the minimiser and its squared residual.
///
/// The residual is non-increasing across iterations.  A channel sitting at a
/// bound whose cost gradient points out of the box is frozen for the
/// iteration; the Gauss-Newton system is solved on the remaining channels
/// and the step is projected back into the box under a halving line search.
fn invert_slice(corners: &[[f64; 3]; 8], target: [f64; 3], start: [f64; 3]) -> ([f64; 3], f64) {
    const H: f64 = 1.0 / 64.0;
    let mut x = start.map(|v| v.clamp(0.0, 1.0));
    let mut err = residual(corners, x, target);
    for _ in 0..MAX_STEPS {
        if err < 1e-9 {
            break;
        }
        let f0 = forward(corners, x);
        // The model is affine in each channel separately, so an inward
        // one-sided difference over any width is exact.
        let mut jac = [[0.0; 3]; 3];
        for col in 0..3 {
            let dir = if x[col] > 0.5 { -H } else { H };
            let mut probe = x;
            probe[col] += dir;
            let f1 = forward(corners, probe);
            for row in 0..3 {
                jac[row][col] = (f1[row] - f0[row]) / dir;
            }
        }
        let res = [0, 1, 2].map(|i| f0[i] - target[i]);
        let grad = [0, 1, 2].map(|c| (0..3).map(|r| jac[r][c] * res[r]).sum::<f64>());
        let free =
            [0, 1, 2].map(|c| !((x[c] <= 0.0 && grad[c] > 0.0) || (x[c] >= 1.0 && grad[c] < 0.0)));
        // Normal equations on the free channels; frozen ones get an identity row.
        let mut normal = [[0.0; 3]; 3];
        let mut rhs = [0.0; 3];
        for i in 0..3 {
            if free[i] {
                for j in 0..3 {
                    if free[j] {
                        normal[i][j] = (0..3).map(|r| jac[r][i] * jac[r][j]).sum();
                    }
                }
                // Levenberg term keeps near-singular reduced systems solvable.
                normal[i][i] += 1e-9;
                rhs[i] = -grad[i];
            } else {
                normal[i][i] = 1.0;
            }
        }
        let Some(step) = solve3(normal, rhs) else {
            break;
        };
        let mut scale = 1.0;
        let mut improved = false;
        for _ in 0..10 {
            let cand = [0, 1, 2].map(|i| step[i].mul_add(scale, x[i]).clamp(0.0, 1.0));
            let e = residual(corners, cand, target);
            if e < err {
                improved = err - e > 1e-12;
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
    (x, err)
}

/// The slice solve of `target` at black `k` from `start`: `(cmy, k, err)`.
fn solve_at(target: [f64; 3], k: f64, start: [f64; 3]) -> ([f64; 3], f64, f64) {
    let (cmy, err) = invert_slice(&slice_corners(k), target, start);
    (cmy, k, err)
}

/// Whether every channel of `target` lies within the corner hull of the
/// slice at black `k`, up to the reproduction tolerance.
fn within_hull(target: [f64; 3], k: f64) -> bool {
    let corners = slice_corners(k);
    (0..3).all(|ch| {
        let top = corners.iter().map(|c| c[ch]).fold(f64::MIN, f64::max);
        target[ch] <= top + EXACT.sqrt()
    })
}

/// The largest black whose slice can print `target`: the largest
/// `k ∈ [0, 1]` with `target` inside the slice's corner hull on every
/// channel (see the module documentation).
///
/// Each channel's hull top is a maximum of affine functions of `k`, so
/// convex; the black-alone corner reaches 255 at `k = 0`, so `k = 0`
/// always qualifies.  The bound is found on a ladder of `HULL_STEPS`
/// blacks down from 1 and refined by bisection.
fn black_bound(target: [f64; 3]) -> f64 {
    const HULL_STEPS: u8 = 32;
    let at = |step: u8| f64::from(step) / f64::from(HULL_STEPS);
    let Some(top) = (0..=HULL_STEPS).rev().find(|&s| within_hull(target, at(s))) else {
        return 0.0;
    };
    if top == HULL_STEPS {
        return 1.0;
    }
    let (mut lo, mut hi) = (at(top), at(top + 1));
    for _ in 0..BISECTIONS {
        let mid = f64::midpoint(lo, hi);
        if within_hull(target, mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// The inks `[c, m, y, k]` in `[0, 1]⁴` of `target` (RGB bytes as `f64`)
/// with the most black that reproduces it; the nearest printable colour's
/// inks when none does.
///
/// The blacks reproducing a colour need not reach down to 0 (in the dark
/// slices adding cyan lightens blue, so a dark colour may be printable
/// only within a band of blacks), so the slice solve is sampled on a
/// ladder of `LADDER` steps from [`black_bound`] down to 0, each solve
/// warm-started from the one above, and the upper edge of the band around
/// the highest reproducing sample is bisected.  With no reproducing sample
/// the sample of least residual is the answer.
fn invert_node(target: [f64; 3]) -> [f64; 4] {
    const LADDER: u8 = 12;
    let k_max = black_bound(target);
    let mut start = target.map(|v| 1.0 - v / f64::from(CORNERS[0][0]));
    let mut nearest = solve_at(target, k_max, start);
    let mut above = k_max;
    for step in 0..=LADDER {
        let k = k_max * f64::from(LADDER - step) / f64::from(LADDER);
        let at = if step == 0 {
            nearest
        } else {
            solve_at(target, k, start)
        };
        start = at.0;
        if at.2 <= EXACT {
            // The band's upper edge lies in `[k, above]`; at the top of
            // the ladder the interval is empty.
            let (mut lo, mut hi) = (at, above);
            for _ in 0..BISECTIONS {
                if hi <= lo.1 {
                    break;
                }
                let mid = f64::midpoint(lo.1, hi);
                let at_mid = solve_at(target, mid, lo.0);
                if at_mid.2 <= EXACT {
                    lo = at_mid;
                } else {
                    hi = mid;
                }
            }
            // At the edge the CMY sit on a bound of the slice; a step
            // inside the band keeps the node solution interior, where
            // neighbouring nodes interpolate to a reproducing ink.
            for margin in [MARGIN, MARGIN / 2.0] {
                if lo.1 - margin >= k {
                    let inside = solve_at(target, lo.1 - margin, lo.0);
                    if inside.2 <= EXACT {
                        return inks(inside);
                    }
                }
            }
            return inks(lo);
        }
        if at.2 < nearest.2 {
            nearest = at;
        }
        above = k;
    }
    inks(nearest)
}

/// `[c, m, y, k]` of a slice solve.
const fn inks((cmy, k, _): ([f64; 3], f64, f64)) -> [f64; 4] {
    [cmy[0], cmy[1], cmy[2], k]
}

/// The inverse tables.
struct Inverse {
    /// Node `(r, g, b)` at index `(r·GRID + g)·GRID + b`, inks in `[0, 1]`.
    nodes: Vec<[f32; 4]>,
    /// The inks of each grey: at or above the floor, the black alone
    /// printing the nearest grey; below it, solid black with the C, M and
    /// Y reproducing the grey on the solid-black slice.
    neutral: [[u8; 4]; 256],
}

/// `[0, 1]` ink amounts as bytes.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is clamped to [0, 255] and rounded first"
)]
fn ink_bytes<const N: usize>(inks: [f64; N]) -> [u8; N] {
    inks.map(|v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
}

/// Re-solve the nodes whose inks lie far from the mean of their six axis
/// neighbours, from that mean on its black, keeping the result when it
/// reproduces the node's colour.
///
/// Where several bands of black reproduce a colour, adjacent nodes may
/// pick different bands, and the interpolant between two bands prints
/// nothing near either; the mean of the neighbours lies on the band most
/// of them chose.  `axis[i]` is the RGB value of grid coordinate `i`.
fn relax(nodes: &mut [[f64; 4]], axis: &[f64]) {
    let n = axis.len();
    let index = |r: usize, g: usize, b: usize| (r * n + g) * n + b;
    for _ in 0..PASSES {
        for r in 0..n {
            for g in 0..n {
                for b in 0..n {
                    let mut mean = [0.0; 4];
                    let mut count = 0.0;
                    for (dr, dg, db) in [(1, 0, 0), (0, 1, 0), (0, 0, 1)] {
                        for sign in [-1isize, 1] {
                            let at =
                                |v: usize, d: usize| v.checked_add_signed(sign * d.cast_signed());
                            if let (Some(nr), Some(ng), Some(nb)) =
                                (at(r, dr), at(g, dg), at(b, db))
                                && nr < n
                                && ng < n
                                && nb < n
                            {
                                for (m, v) in mean.iter_mut().zip(nodes[index(nr, ng, nb)]) {
                                    *m += v;
                                }
                                count += 1.0;
                            }
                        }
                    }
                    let mean = mean.map(|v| v / count);
                    let node = nodes[index(r, g, b)];
                    let far = node.iter().zip(mean).any(|(a, b)| (a - b).abs() > OUTLIER);
                    if !far {
                        continue;
                    }
                    let target = [axis[r], axis[g], axis[b]];
                    let k = mean[3].clamp(0.0, 1.0);
                    let (cmy, err) =
                        invert_slice(&slice_corners(k), target, [mean[0], mean[1], mean[2]]);
                    if err <= EXACT {
                        nodes[index(r, g, b)] = [cmy[0], cmy[1], cmy[2], k];
                    }
                }
            }
        }
    }
}

static INVERSE: LazyLock<Inverse> = LazyLock::new(|| {
    let step = 255.0 / f64::from(GRID - 1);
    let axis: Vec<f64> = (0..GRID).map(|i| (f64::from(i) * step).round()).collect();
    let mut raw = Vec::with_capacity(usize::from(GRID).pow(3));
    for &r in &axis {
        for &g in &axis {
            for &b in &axis {
                raw.push(invert_node([r, g, b]));
            }
        }
    }
    relax(&mut raw, &axis);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "ink amounts lie in [0, 1]; f32 resolution exceeds a byte"
    )]
    let nodes: Vec<[f32; 4]> = raw.iter().map(|inks| inks.map(|v| v as f32)).collect();
    let solid = slice_corners(1.0);
    let mut neutral = [[0u8; 4]; 256];
    for (grey, inks) in (0..=u8::MAX).zip(neutral.iter_mut()) {
        *inks = if grey >= BLACK_FLOOR {
            // Black alone prints a grey that falls with k.
            let k = (0..=u8::MAX)
                .min_by_key(|&k| cmyk_to_rgb(0, 0, 0, k)[0].abs_diff(grey))
                .unwrap_or(u8::MAX);
            [0, 0, 0, k]
        } else {
            let target = [f64::from(grey); 3];
            let (cmy, _) = invert_slice(&solid, target, [0.5; 3]);
            let [c, m, y] = ink_bytes(cmy);
            [c, m, y, u8::MAX]
        };
    }
    Inverse { nodes, neutral }
});

/// Convert sRGB to CMYK ink bytes.
///
/// Of the inks printing a colour, ones with the most black
/// (grey-component replacement), except where relaxation kept a node on
/// the band of its neighbours: a neutral at or above [`BLACK_FLOOR`] is
/// black ink alone; a darker neutral is solid black with C, M and Y on
/// top; a solid process colour has at most a trace of black (two counts),
/// from table interpolation.  The result maps back through
/// [`cmyk_to_rgb`] to within two counts per channel for every colour the
/// four-ink model prints, except for about one in two thousand dark
/// colours (channel mean below 64), within four; and to the nearest
/// printable colour otherwise, up to table interpolation.
#[must_use]
pub fn rgb_to_ink(rgb: [u8; 3]) -> [u8; 4] {
    let table = &*INVERSE;
    if rgb[0] == rgb[1] && rgb[1] == rgb[2] {
        return table.neutral[usize::from(rgb[0])];
    }
    let span = u16::from(GRID - 1);
    // Exact grid coordinate of each channel is `v · (GRID − 1) / 255`: integer cell
    // index and fraction numerator over 255.
    let scaled = rgb.map(|v| u16::from(v) * span);
    let cell = scaled.map(|s| (s / 255).min(span - 1));
    let frac = [0, 1, 2].map(|i| f64::from(scaled[i] - cell[i] * 255) / 255.0);
    let lo = cell.map(usize::from);
    let n = usize::from(GRID);
    let mut inks = [0.0f64; 4];
    for corner in 0..8 {
        let w = corner_weight(corner, frac);
        let bits = corner_bits(corner);
        let node = table.nodes[((lo[0] + bits[0]) * n + lo[1] + bits[1]) * n + lo[2] + bits[2]];
        for (o, v) in inks.iter_mut().zip(node) {
            *o = w.mul_add(f64::from(v), *o);
        }
    }
    ink_bytes(inks)
}

/// Convert an ink canvas row to sRGB.
///
/// `ink` holds 8 bytes per pixel `[C, M, Y, K, S0, S1, S2, S3]` (0 = no
/// ink).  Each spot with a [`SpotLut`] contributes its process equivalent at
/// the pixel's tint, combined with the running process value as a
/// probabilistic union ([`ink_union`]) per channel; spots
/// without a table are ignored.  The combined CMYK row is converted by
/// [`cmyk_to_rgb_row`] in stack-buffered chunks.
///
/// # Panics
///
/// Panics if `ink.len()` is not a multiple of 8 or
/// `out.len() != ink.len() / 8 * 3`.
pub fn ink_to_rgb_row(ink: &[u8], spots: &[Option<&SpotLut>; 4], out: &mut [u8]) {
    assert!(
        ink.len().is_multiple_of(8),
        "ink_to_rgb_row: ink length {} is not a multiple of 8",
        ink.len()
    );
    assert_eq!(
        out.len(),
        ink.len() / 8 * 3,
        "ink_to_rgb_row: output length must be 3 bytes per ink pixel ({} ink bytes)",
        ink.len()
    );
    let mut cmyk = [0u8; CHUNK * 4];
    for (src, dst) in ink.chunks(CHUNK * 8).zip(out.chunks_mut(CHUNK * 3)) {
        let n = src.len() / 8;
        for (px, quad) in src
            .as_chunks::<8>()
            .0
            .iter()
            .zip(cmyk.as_chunks_mut::<4>().0)
        {
            quad.copy_from_slice(&px[..4]);
            for (lut, &tint) in spots.iter().zip(&px[4..]) {
                if let Some(lut) = lut {
                    for (c, &q) in quad.iter_mut().zip(&lut[usize::from(tint)]) {
                        *c = ink_union(*c, q);
                    }
                }
            }
        }
        cmyk_to_rgb_row(&cmyk[..n * 4], dst);
    }
}

/// The probabilistic union of two ink coverages `a` and `b` (0 = no ink):
/// `a + b − a·b/255`, rounded to nearest, computed as the complement of
/// the product of the uncovered fractions.
///
/// `a·b/255` never has a fractional part of exactly one half (255 is odd),
/// so the rounding is unambiguous.
#[must_use]
pub(crate) fn ink_union(a: u8, b: u8) -> u8 {
    let uncovered = u32::from(255 - a) * u32::from(255 - b);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "uncovered ≤ 255², so (uncovered + 127) / 255 ≤ 255"
    )]
    let paper = ((uncovered + 127) / 255) as u8;
    255 - paper
}
