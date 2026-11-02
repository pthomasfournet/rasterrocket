//! RGB → CMYK inverse of the process-ink model, and ink-canvas → RGB conversion.
//!
//! The forward model restricted to `K = 0` is the trilinear interpolation of
//! the eight `K = 0` entries of [`CORNERS`](super::CORNERS).  [`rgb_to_cmyk`]
//! inverts it through a 65³ table over the sRGB cube: each node holds the
//! CMY triple in `[0, 1]³` minimising the squared RGB distance between its
//! forward image and the node, found by projected Gauss-Newton in `f64`
//! (channels at a bound with an outward-pointing gradient are frozen and
//! the step is solved on the free channels).  In-gamut nodes are exact
//! preimages; out-of-gamut nodes are the box-constrained least-squares
//! minimum, a local one since the model is multilinear.  A query is the
//! trilinear interpolation of the eight surrounding nodes and nothing else.

use std::sync::LazyLock;

use super::{CORNERS, cmyk_to_rgb_row};

/// Nodes per axis of the inverse table.
const GRID: u8 = 65;
/// Projected Gauss-Newton iterations per node.
const MAX_STEPS: usize = 40;
/// Pixels converted per stack buffer in [`ink_to_rgb_row`].
const CHUNK: usize = 256;

/// Process-ink equivalent of each tint (0–255) of one spot colorant:
/// entry `t` is the `[C, M, Y, K]` bytes printing the same colour as the
/// spot at tint `t`.
pub type SpotLut = [[u8; 4]; 256];

/// Forward `K = 0` model on `[0, 1]³`, unrounded: trilinear over
/// `CORNERS[0..8]`, index `c·4 + m·2 + y`.
fn forward(cmy: [f64; 3]) -> [f64; 3] {
    let mut out = [0.0; 3];
    for (i, corner) in CORNERS[..8].iter().enumerate() {
        let w: f64 = [(i >> 2) & 1, (i >> 1) & 1, i & 1]
            .iter()
            .zip(cmy)
            .map(|(&bit, t)| if bit == 1 { t } else { 1.0 - t })
            .product();
        for (o, &v) in out.iter_mut().zip(corner) {
            *o = w.mul_add(f64::from(v), *o);
        }
    }
    out
}

/// Sum of squared residuals of `forward(cmy)` against `target`.
fn residual(cmy: [f64; 3], target: [f64; 3]) -> f64 {
    forward(cmy)
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

/// CMY in `[0, 1]³` minimising the squared distance of the forward image to
/// `target` (RGB bytes as `f64`), by projected Gauss-Newton from the
/// additive complement.
///
/// The residual is non-increasing across iterations.  A channel sitting at a
/// bound whose cost gradient points out of the box is frozen for the
/// iteration; the Gauss-Newton system is solved on the remaining channels
/// and the step is projected back into the box under a halving line search.
fn invert_node(target: [f64; 3]) -> [f64; 3] {
    const H: f64 = 1.0 / 64.0;
    let mut x = target.map(|v| 1.0 - v / 255.0);
    let mut err = residual(x, target);
    for _ in 0..MAX_STEPS {
        let f0 = forward(x);
        // The model is affine in each channel separately, so an inward
        // one-sided difference over any width is exact.
        let mut jac = [[0.0; 3]; 3];
        for col in 0..3 {
            let dir = if x[col] > 0.5 { -H } else { H };
            let mut probe = x;
            probe[col] += dir;
            let f1 = forward(probe);
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
            let e = residual(cand, target);
            if e < err {
                improved = err - e > 1e-12;
                x = cand;
                err = e;
                break;
            }
            scale *= 0.5;
        }
        if !improved || err < 1e-9 {
            break;
        }
    }
    x
}

/// Inverse table, node `(r, g, b)` at index `(r·65 + g)·65 + b`, CMY in `[0, 1]`.
static INVERSE: LazyLock<Vec<[f32; 3]>> = LazyLock::new(|| {
    let step = 255.0 / f64::from(GRID - 1);
    let axis = || (0..GRID).map(|i| (f64::from(i) * step).round());
    let mut table = Vec::with_capacity(usize::from(GRID).pow(3));
    for r in axis() {
        for g in axis() {
            for b in axis() {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "ink amounts lie in [0, 1]; f32 resolution exceeds a byte"
                )]
                table.push(invert_node([r, g, b]).map(|v| v as f32));
            }
        }
    }
    table
});

/// Convert sRGB to CMYK ink bytes with `K = 0`.
///
/// Trilinear lookup in a 65³ inverse of the forward model (built on first
/// use).  The result maps back through [`cmyk_to_rgb`](super::cmyk_to_rgb)
/// to within two counts for in-gamut colours; out-of-gamut colours map to
/// the printable CMY minimising the RGB distance, up to table
/// interpolation.
#[must_use]
pub fn rgb_to_cmyk(rgb: [u8; 3]) -> [u8; 4] {
    let table = &*INVERSE;
    let span = u16::from(GRID - 1);
    // Exact grid coordinate of each channel is `v · (GRID − 1) / 255`: integer cell
    // index and fraction numerator over 255.
    let scaled = rgb.map(|v| u16::from(v) * span);
    let cell = scaled.map(|s| (s / 255).min(span - 1));
    let frac = [0, 1, 2].map(|i| f32::from(scaled[i] - cell[i] * 255) / 255.0);
    let lo = cell.map(usize::from);
    let mut cmy = [0.0f32; 3];
    for corner in 0..8usize {
        let bits = [(corner >> 2) & 1, (corner >> 1) & 1, corner & 1];
        let mut w = 1.0f32;
        for i in 0..3 {
            w *= if bits[i] == 1 { frac[i] } else { 1.0 - frac[i] };
        }
        let n = usize::from(GRID);
        let node = table[((lo[0] + bits[0]) * n + lo[1] + bits[1]) * n + lo[2] + bits[2]];
        for (o, v) in cmy.iter_mut().zip(node) {
            *o = w.mul_add(v, *o);
        }
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the value is clamped to [0, 255] and rounded first"
    )]
    let [c, m, y] = cmy.map(|v| (v * 255.0).round().clamp(0.0, 255.0) as u8);
    [c, m, y, 0]
}

/// The process ink of sRGB `rgb`: black ink alone for a neutral, the
/// nearest in-gamut CMY ([`rgb_to_cmyk`]) for any other colour.
///
/// A neutral (`r = g = b`) is the grey `g` of `DeviceGray` and converts as
/// `DeviceGray` does to `DeviceCMYK`, `K = 1 − g` with no C, M or Y
/// (ISO 32000-2 §10.4.2.3).  Black therefore prints as K alone, not as a
/// mixture of C, M and Y.
#[must_use]
pub fn rgb_to_ink(rgb: [u8; 3]) -> [u8; 4] {
    let [r, g, b] = rgb;
    if r == g && g == b {
        [0, 0, 0, u8::MAX - r]
    } else {
        rgb_to_cmyk(rgb)
    }
}

/// Convert an ink canvas row to sRGB.
///
/// `ink` holds 8 bytes per pixel `[C, M, Y, K, S0, S1, S2, S3]` (0 = no
/// ink).  Each spot with a [`SpotLut`] contributes its process equivalent at
/// the pixel's tint, combined with the running process value as a
/// probabilistic union `c' = c + q − (c·q + 127) / 255` per channel; spots
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
        "ink_to_rgb_row: output length must be 3 bytes per ink pixel"
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
                        let (c0, q) = (u16::from(*c), u16::from(q));
                        // c + q − c·q/255 ≤ 255 for c, q ≤ 255, so the sum fits a byte.
                        *c = u8::try_from(c0 + q - (c0 * q + 127) / 255).unwrap_or(u8::MAX);
                    }
                }
            }
        }
        cmyk_to_rgb_row(&cmyk[..n * 4], dst);
    }
}
