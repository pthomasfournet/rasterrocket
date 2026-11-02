//! RGB → CMYK inverse of the process-ink model, and ink-canvas → RGB conversion.
//!
//! The forward model restricted to `K = 0` is the trilinear interpolation of
//! the eight `K = 0` entries of [`CORNERS`](super::CORNERS).  [`rgb_to_cmyk`]
//! inverts it through a 33³ table over the sRGB cube: each node holds the
//! CMY triple whose forward image is the node, found by damped Newton
//! iteration in `f64`; a query is the trilinear interpolation of the eight
//! surrounding nodes.  Nodes outside the forward gamut hold the clamped
//! least-residual CMY found by the iteration, so out-of-gamut colours map to
//! the nearest printable ink amounts rather than failing.

use std::sync::LazyLock;

use super::{CORNERS, cmyk_to_rgb_row};

/// Nodes per axis of the inverse table.
const GRID: u8 = 33;
/// Newton iterations per node.
const MAX_STEPS: usize = 8;
/// Newton iterations applied to an interpolated table result.
const POLISH_STEPS: usize = 1;
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

/// CMY whose forward image best matches `target` (RGB bytes as `f64`),
/// iterating from the additive complement.
fn invert_node(target: [f64; 3]) -> [f64; 3] {
    refine(target.map(|v| 1.0 - v / 255.0), target, MAX_STEPS)
}

/// Up to `steps` damped, clamped Newton iterations from `x` towards `target`.
///
/// The residual is non-increasing, so the result is never worse than `x`.
fn refine(mut x: [f64; 3], target: [f64; 3], steps: usize) -> [f64; 3] {
    const H: f64 = 1.0 / 64.0;
    let mut err = residual(x, target);
    for _ in 0..steps {
        let f0 = forward(x);
        let mut jac = [[0.0; 3]; 3];
        for col in 0..3 {
            // One-sided difference stepping inward so the probe stays in [0, 1].
            let dir = if x[col] > 0.5 { -H } else { H };
            let mut probe = x;
            probe[col] += dir;
            let f1 = forward(probe);
            for row in 0..3 {
                jac[row][col] = (f1[row] - f0[row]) / dir;
            }
        }
        let rhs = [0, 1, 2].map(|i| target[i] - f0[i]);
        let Some(step) = solve3(jac, rhs) else { break };
        let mut scale = 1.0;
        let mut improved = false;
        for _ in 0..4 {
            let cand = [0, 1, 2].map(|i| step[i].mul_add(scale, x[i]).clamp(0.0, 1.0));
            let e = residual(cand, target);
            if e < err {
                x = cand;
                err = e;
                improved = true;
                break;
            }
            scale *= 0.5;
        }
        if !improved || err < 1e-6 {
            break;
        }
    }
    x
}

/// Inverse table, node `(r, g, b)` at index `(r·33 + g)·33 + b`, CMY in `[0, 1]`.
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
/// Trilinear lookup in a 33³ inverse of the forward model (built on first
/// use), followed by one Newton step against the exact target.  The result
/// maps back through [`cmyk_to_rgb`](super::cmyk_to_rgb) to within two
/// counts for in-gamut colours; out-of-gamut colours map to the nearest
/// printable CMY.
#[must_use]
pub fn rgb_to_cmyk(rgb: [u8; 3]) -> [u8; 4] {
    let table = &*INVERSE;
    let span = u16::from(GRID - 1);
    // Exact grid coordinate of each channel is `v · 32 / 255`: integer cell
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
    // Interpolating across the gamut-boundary kink of the inverse leaves a
    // few counts of error on the boundary axes; one Newton step from the
    // interpolant removes it.
    let target = rgb.map(f64::from);
    let polished = refine(cmy.map(f64::from), target, POLISH_STEPS);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the value is clamped to [0, 255] and rounded first"
    )]
    let [c, m, y] = polished.map(|v| (v * 255.0).round().clamp(0.0, 255.0) as u8);
    [c, m, y, 0]
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
